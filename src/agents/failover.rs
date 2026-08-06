//! Classifying agent failures and walking down the ranked list.
//!
//! Whether an agent is logged in and entitled to a model is only knowable by calling
//! it. So pstore calls, and when a call fails it reads the exit code and stderr to
//! decide *why* — then remembers the verdict ([`super::detect::remember_failure`])
//! and moves to the next candidate.

use std::path::Path;
use std::sync::mpsc::Sender;
use std::time::Duration;

use super::detect::{self, Detected, Unavailable};
use super::launch::{self, Cancel, Line, Output};
use super::registry::Effort;
use crate::router::{Choice, Ranking};

/// Time budget for one headless call.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(180);

/// Classify a failed run from its exit code and stderr.
///
/// Patterns are matched case-insensitively against the combined output, most
/// specific first — quota messages often also mention authentication, so ordering
/// matters.
pub fn classify(out: &Output) -> Option<Unavailable> {
    if out.cancelled {
        // The user stopped it; that says nothing about the agent's health.
        return None;
    }
    if out.timed_out {
        return Some(Unavailable::Timeout);
    }
    if out.ok() {
        return None;
    }

    let haystack = format!("{}\n{}", out.stderr, out.stdout).to_ascii_lowercase();
    let has = |needles: &[&str]| needles.iter().any(|n| haystack.contains(n));

    if has(&[
        "rate limit",
        "rate-limit",
        "quota",
        "usage limit",
        "insufficient credit",
        "insufficient_quota",
        "billing",
        "429",
        "too many requests",
    ]) {
        return Some(Unavailable::QuotaExhausted(pick_line(
            &out.stderr,
            &out.stdout,
        )));
    }
    if has(&[
        "not logged in",
        "please log in",
        "please login",
        "run /login",
        "unauthorized",
        "unauthenticated",
        "authentication",
        "invalid api key",
        "no api key",
        "api key not found",
        "credentials",
        "401",
        "403",
    ]) {
        return Some(Unavailable::NotLoggedIn(pick_line(
            &out.stderr,
            &out.stdout,
        )));
    }
    if has(&[
        "model not found",
        "unknown model",
        "invalid model",
        "unsupported model",
        "does not have access",
        "no access to model",
        "model unavailable",
        "not entitled",
    ]) {
        return Some(Unavailable::ModelDenied(pick_line(
            &out.stderr,
            &out.stdout,
        )));
    }
    Some(Unavailable::Other(pick_line(&out.stderr, &out.stdout)))
}

/// First non-empty line of stderr, falling back to stdout, then to a placeholder.
fn pick_line(stderr: &str, stdout: &str) -> String {
    for src in [stderr, stdout] {
        if let Some(line) = src.lines().map(str::trim).find(|l| !l.is_empty()) {
            return line.to_string();
        }
    }
    "failed with no output".to_string()
}

/// What a successful run produced, and which candidate produced it.
#[derive(Debug, Clone)]
pub struct Completed {
    /// Agent that answered.
    pub agent_id: &'static str,
    /// Model that answered, empty if the agent chose.
    pub model_id: &'static str,
    /// Effort that was requested.
    pub effort: Effort,
    /// Combined text output.
    pub text: String,
    /// How long it took.
    pub elapsed: Duration,
    /// Candidates that failed before this one, with reasons.
    pub attempts: Vec<(&'static str, String)>,
}

/// Every candidate failed.
#[derive(Debug, Clone)]
pub struct AllFailed {
    /// Each candidate tried, with why it failed.
    pub attempts: Vec<(&'static str, String)>,
}

impl AllFailed {
    /// Human summary for the UI.
    pub fn summary(&self) -> String {
        if self.attempts.is_empty() {
            return "no usable agent was detected".into();
        }
        let list = self
            .attempts
            .iter()
            .map(|(id, why)| format!("{id} ({why})"))
            .collect::<Vec<_>>()
            .join(", ");
        format!("all candidates failed: {list}")
    }
}

/// Deduplicate a ranking down to one candidate per agent, preserving rank order.
///
/// Failover is about trying a *different agent* when one is unusable; retrying the
/// same agent's other 19 model/effort pairs after a login failure only wastes time.
/// The exception is [`Unavailable::might_work_with_another_model`], handled by the
/// caller keeping the next entry for that agent.
fn one_per_agent(ranking: &Ranking) -> Vec<&Choice> {
    let mut seen = Vec::new();
    let mut out = Vec::new();
    for c in &ranking.choices {
        if !seen.contains(&c.agent_id) {
            seen.push(c.agent_id);
            out.push(c);
        }
    }
    out
}

/// Run `prompt` against the best candidate, falling back down the ranking on failure.
///
/// `tx` receives output lines as they arrive. `dir` is the prompt folder, used to
/// persist verdicts.
// Plumbing: every argument is a distinct, unrelated input (program, argv,
// stdin, cwd, timeout, cancellation, sink). Bundling them into a struct
// would add a type to thread through without making any call site clearer.
#[allow(clippy::too_many_arguments)]
pub fn run_with_failover(
    detected: &[Detected],
    ranking: &Ranking,
    prompt: &str,
    cwd: &Path,
    dir: &Path,
    timeout: Duration,
    cancel: Option<&Cancel>,
    tx: &Sender<Line>,
) -> Result<Completed, AllFailed> {
    let mut attempts: Vec<(&'static str, String)> = Vec::new();

    for candidate in one_per_agent(ranking) {
        // Cancelling must stop the walk down the list, not just the current child.
        if launch::raised(cancel) {
            return Err(AllFailed { attempts });
        }
        let Some(agent) = detected.iter().find(|d| d.spec.id == candidate.agent_id) else {
            continue;
        };

        let effort = candidate.effort_selectable.then_some(candidate.effort);
        let (args, stdin) =
            launch::headless_args(agent.spec, Some(candidate.model_id), effort, prompt);

        let out = match launch::run_streaming(
            &agent.path,
            &args,
            stdin.as_deref(),
            Some(cwd),
            timeout,
            cancel,
            tx,
        ) {
            Ok(o) => o,
            Err(e) => {
                attempts.push((candidate.agent_id, format!("could not start: {e}")));
                continue;
            }
        };

        if out.cancelled {
            return Err(AllFailed { attempts });
        }

        match classify(&out) {
            None => {
                detect::remember_success(dir, candidate.agent_id);
                return Ok(Completed {
                    agent_id: candidate.agent_id,
                    model_id: candidate.model_id,
                    effort: candidate.effort,
                    text: out.stdout,
                    elapsed: out.elapsed,
                    attempts,
                });
            }
            Some(why) => {
                // A model-entitlement failure says nothing about the agent as a
                // whole, so don't blacklist it — just move on to the next agent.
                if !why.might_work_with_another_model() {
                    detect::remember_failure(dir, candidate.agent_id, why.clone());
                }
                attempts.push((candidate.agent_id, why.reason()));
            }
        }
    }

    Err(AllFailed { attempts })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn failed(stderr: &str) -> Output {
        Output {
            code: Some(1),
            stdout: String::new(),
            stderr: stderr.to_string(),
            timed_out: false,
            cancelled: false,
            elapsed: Duration::from_millis(10),
        }
    }

    #[test]
    fn success_is_not_a_failure() {
        let ok = Output {
            code: Some(0),
            stdout: "done".into(),
            stderr: String::new(),
            timed_out: false,
            cancelled: false,
            elapsed: Duration::from_millis(5),
        };
        assert!(classify(&ok).is_none());
    }

    #[test]
    fn recognises_login_failures() {
        for msg in [
            "Error: not logged in. Run /login to continue.",
            "Please log in first",
            "HTTP 401 Unauthorized",
            "Invalid API key provided",
            "authentication failed",
        ] {
            match classify(&failed(msg)) {
                Some(Unavailable::NotLoggedIn(_)) => {}
                other => panic!("{msg:?} classified as {other:?}"),
            }
        }
    }

    #[test]
    fn recognises_quota_failures_before_auth() {
        // Quota messages frequently mention the API key too; quota must win.
        for msg in [
            "Rate limit exceeded, retry after 60s",
            "You have exceeded your usage limit for this api key",
            "HTTP 429 Too Many Requests",
            "insufficient credit balance",
        ] {
            match classify(&failed(msg)) {
                Some(Unavailable::QuotaExhausted(_)) => {}
                other => panic!("{msg:?} classified as {other:?}"),
            }
        }
    }

    #[test]
    fn recognises_model_entitlement_failures() {
        for msg in [
            "unknown model: opus",
            "Your account does not have access to this model",
            "invalid model name",
        ] {
            match classify(&failed(msg)) {
                Some(Unavailable::ModelDenied(_)) => {}
                other => panic!("{msg:?} classified as {other:?}"),
            }
        }
    }

    #[test]
    fn timeout_wins_over_message_content() {
        let out = Output {
            code: None,
            stdout: String::new(),
            stderr: "not logged in".into(),
            timed_out: true,
            cancelled: false,
            elapsed: Duration::from_secs(200),
        };
        assert_eq!(classify(&out), Some(Unavailable::Timeout));
    }

    #[test]
    fn cancellation_is_not_an_agent_fault() {
        // Blacklisting an agent because the user pressed Stop would make the next
        // ranking silently skip a perfectly good agent.
        let out = Output {
            code: None,
            stdout: String::new(),
            stderr: "killed".into(),
            timed_out: false,
            cancelled: true,
            elapsed: Duration::from_millis(50),
        };
        assert_eq!(classify(&out), None);
        assert!(!out.ok(), "a cancelled run is still not a success");
    }

    #[test]
    fn unrecognised_failures_keep_the_agents_own_message() {
        match classify(&failed("segfault in tokenizer")) {
            Some(Unavailable::Other(m)) => assert_eq!(m, "segfault in tokenizer"),
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn failure_with_no_output_still_reports_something() {
        match classify(&failed("")) {
            Some(Unavailable::Other(m)) => assert_eq!(m, "failed with no output"),
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn classification_reads_stdout_when_stderr_is_empty() {
        let out = Output {
            code: Some(1),
            stdout: "Error: please login".into(),
            stderr: String::new(),
            timed_out: false,
            cancelled: false,
            elapsed: Duration::from_millis(1),
        };
        match classify(&out) {
            Some(Unavailable::NotLoggedIn(m)) => assert!(m.contains("please login")),
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn one_per_agent_keeps_rank_order_and_dedups() {
        use crate::agents::registry::Tier;

        let mk = |agent: &'static str, model: &'static str, fit: f32| Choice {
            agent_id: agent,
            agent_display: agent,
            model_id: model,
            model_display: model,
            tier: Tier::Mid,
            effort: Effort::High,
            effort_selectable: true,
            metered: false,
            relative_latency: 1.0,
            relative_price: 1.0,
            fit,
            rationale: String::new(),
        };
        let ranking = Ranking {
            choices: vec![
                mk("claude", "opus", 99.0),
                mk("claude", "sonnet", 95.0),
                mk("codex", "gpt", 90.0),
                mk("claude", "haiku", 70.0),
                mk("crush", "", 60.0),
            ],
            ..Ranking::default()
        };
        let picked: Vec<_> = one_per_agent(&ranking)
            .iter()
            .map(|c| (c.agent_id, c.model_id))
            .collect();
        assert_eq!(
            picked,
            vec![("claude", "opus"), ("codex", "gpt"), ("crush", "")],
            "one attempt per agent, best model first"
        );
    }

    /// End-to-end against a real installed agent. Ignored by default: it spawns a
    /// real process and spends a small amount of the user's quota. Run explicitly with
    /// `cargo test -- --ignored real_agent`.
    #[test]
    #[ignore = "spawns a real agent and uses quota"]
    fn real_agent_answers_through_the_launcher() {
        use crate::agents::detect::{self, Probe};
        use crate::agents::launch;

        let dir = std::env::temp_dir().join("pstore-live-check");
        std::fs::create_dir_all(&dir).unwrap();
        let detected = detect::detect_in(&dir, &Probe::from_env());
        let agent = detected
            .iter()
            .find(|d| d.spec.id == "claude" && d.usable())
            .expect("this check needs a usable claude on PATH");

        // The cheapest, fastest configuration that still exercises the real path.
        let (args, stdin) = launch::headless_args(
            agent.spec,
            Some("haiku"),
            Some(Effort::Low),
            "Reply with exactly: PSTORE_OK",
        );

        let (tx, rx) = std::sync::mpsc::channel();
        let out = launch::run_streaming(
            &agent.path,
            &args,
            stdin.as_deref(),
            Some(&dir),
            Duration::from_secs(120),
            None,
            &tx,
        )
        .expect("spawning the agent");
        drop(tx);

        let streamed: String = rx
            .into_iter()
            .filter_map(|l| match l {
                Line::Out(t) => Some(t),
                Line::Err(_) => None,
            })
            .collect();

        assert!(
            out.ok(),
            "agent failed: code={:?} classified={:?}\nstderr: {}",
            out.code,
            classify(&out),
            out.stderr
        );
        // The real assertion: our stream-json parsing produced the model's text,
        // not raw JSON envelopes.
        assert!(
            streamed.contains("PSTORE_OK"),
            "extracted text did not contain the answer.\nextracted: {streamed:?}\nraw stdout: {}",
            out.stdout
        );
        assert!(
            !streamed.contains("\"type\":"),
            "raw JSON leaked into the panel text: {streamed:?}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn all_failed_summary_lists_attempts() {
        let empty = AllFailed {
            attempts: Vec::new(),
        };
        assert!(empty.summary().contains("no usable agent"));

        let some = AllFailed {
            attempts: vec![
                ("claude", "not logged in".into()),
                ("codex", "timed out".into()),
            ],
        };
        let s = some.summary();
        assert!(s.contains("claude (not logged in)"), "got {s}");
        assert!(s.contains("codex (timed out)"), "got {s}");
    }
}
