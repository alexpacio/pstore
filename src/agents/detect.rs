//! Agent detection.
//!
//! Three stages, none of which spend tokens:
//!
//! 1. **PATH probe** — is the binary there, and what does `--version` say?
//! 2. **Credential probe** — does a known config/credential file exist? A *hint* only.
//! 3. **Lazy verification** — you cannot know an agent is logged in and entitled to a
//!    model without calling it, so the first real call classifies its own failure
//!    ([`super::failover::classify`]) and the verdict is remembered here.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};

use super::registry::{self, AgentSpec};
use crate::config::SIDECAR;

/// How long a cached verdict is trusted before being re-probed.
const TTL: Duration = Duration::from_secs(6 * 60 * 60);

/// Why an agent cannot currently be used.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "detail")]
pub enum Unavailable {
    /// Binary not found on `PATH`.
    NotInstalled,
    /// Installed, but the agent reports no valid session.
    NotLoggedIn(String),
    /// Authenticated but out of quota or rate-limited.
    QuotaExhausted(String),
    /// The requested model is not available to this account.
    ModelDenied(String),
    /// The process exceeded its time budget.
    Timeout,
    /// Anything else, with the agent's own message.
    Other(String),
}

impl Unavailable {
    /// Short human explanation for the UI.
    pub fn reason(&self) -> String {
        match self {
            Unavailable::NotInstalled => "not installed".into(),
            Unavailable::NotLoggedIn(m) => format!("not logged in: {}", first_line(m)),
            Unavailable::QuotaExhausted(m) => format!("quota exhausted: {}", first_line(m)),
            Unavailable::ModelDenied(m) => format!("model unavailable: {}", first_line(m)),
            Unavailable::Timeout => "timed out".into(),
            Unavailable::Other(m) => first_line(m).to_string(),
        }
    }

    /// Whether a different model on the same agent might still work.
    pub fn might_work_with_another_model(&self) -> bool {
        matches!(self, Unavailable::ModelDenied(_))
    }
}

fn first_line(s: &str) -> &str {
    let line = s
        .lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("")
        .trim();
    if line.len() > 140 { &line[..140] } else { line }
}

/// What pstore currently believes about one agent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Status {
    /// Binary present and no failure observed yet. Not proof of a valid login.
    Ready,
    /// Binary present, a credential file exists, and a real call has succeeded.
    Verified,
    /// Known to be unusable, with the reason.
    Blocked(Unavailable),
}

impl Status {
    /// Whether the ranker may consider this agent.
    pub fn usable(&self) -> bool {
        matches!(self, Status::Ready | Status::Verified)
    }
}

/// A detected agent.
#[derive(Debug, Clone)]
pub struct Detected {
    /// The registry row this came from.
    pub spec: &'static AgentSpec,
    /// Resolved absolute path to the executable.
    pub path: PathBuf,
    /// Trimmed first line of `--version`, when it answered.
    pub version: Option<String>,
    /// Whether a known credential/config path exists.
    pub has_credentials: bool,
    /// Current belief about usability.
    pub status: Status,
    /// The model this agent's own config says it will run, for the agents pstore cannot tell.
    ///
    /// `None` either because the agent takes a `--model` flag (its models are in the registry)
    /// or because its config did not say — see [`super::configured`]. The second case is what
    /// keeps a nameless candidate out of the ranking.
    pub configured_model: Option<String>,
}

impl Detected {
    /// Whether the ranker may consider this agent.
    pub fn usable(&self) -> bool {
        self.status.usable()
    }
}

/// Persisted verdicts, so a failure discovered once is remembered across restarts.
#[derive(Debug, Default, Serialize, Deserialize)]
struct Cache {
    /// Unix seconds when the cache was written.
    #[serde(default)]
    stamped: u64,
    #[serde(default)]
    verdicts: BTreeMap<String, Status>,
}

fn cache_path(dir: &Path) -> PathBuf {
    dir.join(SIDECAR).join("agents.json")
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn load_cache(dir: &Path) -> Cache {
    let cache: Cache = std::fs::read_to_string(cache_path(dir))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();
    // Expire wholesale: a stale "not logged in" must not hide a fresh login.
    if now_secs().saturating_sub(cache.stamped) > TTL.as_secs() {
        return Cache::default();
    }
    cache
}

fn save_cache(dir: &Path, cache: &Cache) {
    let path = cache_path(dir);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(json) = serde_json::to_string_pretty(cache) {
        let _ = std::fs::write(path, json);
    }
}

/// Where to look for agents and their credentials.
///
/// Taken as data rather than read from globals so detection is testable against a
/// fixture directory — `std::env::set_var` is unsafe in edition 2024 and mutating a
/// process-wide `PATH` from tests is racy besides.
#[derive(Debug, Clone)]
pub struct Probe {
    /// Value to search for executables, in `PATH` format.
    pub path: std::ffi::OsString,
    /// Home directory to resolve credential paths against.
    pub home: Option<PathBuf>,
}

impl Probe {
    /// Read the real environment.
    pub fn from_env() -> Self {
        Self {
            path: std::env::var_os("PATH").unwrap_or_default(),
            home: dirs::home_dir(),
        }
    }
}

/// Resolve `bin` against a `PATH`-formatted value, honouring `PATHEXT` on Windows.
fn which(path: &std::ffi::OsStr, bin: &str) -> Option<PathBuf> {
    let exts: Vec<String> = if cfg!(windows) {
        std::env::var("PATHEXT")
            .unwrap_or_else(|_| ".EXE;.CMD;.BAT;.COM".into())
            .split(';')
            .filter(|e| !e.is_empty())
            .map(|e| e.to_ascii_lowercase())
            .collect()
    } else {
        Vec::new()
    };

    for dir in std::env::split_paths(path) {
        let direct = dir.join(bin);
        if is_executable(&direct) {
            return Some(direct);
        }
        for ext in &exts {
            let candidate = dir.join(format!("{bin}{ext}"));
            if is_executable(&candidate) {
                return Some(candidate);
            }
        }
    }
    None
}

#[cfg(unix)]
fn is_executable(p: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(p).is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
}

#[cfg(not(unix))]
fn is_executable(p: &Path) -> bool {
    p.is_file()
}

/// Ask `bin --version`, with a short timeout so a hung agent can't stall startup.
fn probe_version(path: &Path) -> Option<String> {
    let out = super::launch::run_capture(
        path,
        &["--version".to_string()],
        None,
        None,
        Duration::from_secs(5),
    )
    .ok()?;
    let text = if out.stdout.trim().is_empty() {
        out.stderr
    } else {
        out.stdout
    };
    text.lines()
        .find(|l| !l.trim().is_empty())
        .map(|l| l.trim().to_string())
}

fn credentials_present(spec: &AgentSpec, home: Option<&Path>) -> bool {
    let Some(home) = home else { return false };
    spec.creds.iter().any(|rel| home.join(rel).exists())
}

/// Detect every known agent, reading the real environment.
///
/// `dir` is the prompt folder, used for the verdict cache.
pub fn detect_all(dir: &Path) -> Vec<Detected> {
    detect_in(dir, &Probe::from_env())
}

/// Detect against an explicit [`Probe`].
pub fn detect_in(dir: &Path, probe: &Probe) -> Vec<Detected> {
    let cache = load_cache(dir);
    let mut out = Vec::new();

    for spec in registry::AGENTS {
        let Some(path) = which(&probe.path, spec.bin) else {
            continue;
        };
        let has_credentials = credentials_present(spec, probe.home.as_deref());
        // A cached Blocked verdict survives; Ready/Verified are re-derived so a
        // newly-created credential file is noticed.
        let status = match cache.verdicts.get(spec.id) {
            Some(Status::Blocked(u)) => Status::Blocked(u.clone()),
            _ if has_credentials => Status::Verified,
            _ => Status::Ready,
        };
        out.push(Detected {
            spec,
            version: probe_version(&path),
            path,
            has_credentials,
            status,
            // Only for the agents that choose their own model; the rest get theirs from the
            // registry, and asking their config would be a second answer to a settled question.
            configured_model: super::configured::model_for(spec, dir),
        });
    }
    out
}

/// Record a failure so the ranker stops proposing this agent until the TTL expires.
pub fn remember_failure(dir: &Path, agent_id: &str, why: Unavailable) {
    let mut cache = load_cache(dir);
    cache.stamped = now_secs();
    cache
        .verdicts
        .insert(agent_id.to_string(), Status::Blocked(why));
    save_cache(dir, &cache);
}

/// Record that a real call against this agent succeeded.
pub fn remember_success(dir: &Path, agent_id: &str) {
    let mut cache = load_cache(dir);
    cache.stamped = now_secs();
    cache
        .verdicts
        .insert(agent_id.to_string(), Status::Verified);
    save_cache(dir, &cache);
}

/// Drop all cached verdicts, forcing a clean re-probe.
pub fn clear_cache(dir: &Path) {
    let _ = std::fs::remove_file(cache_path(dir));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "pstore-detect-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::remove_dir_all(&p).ok();
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    /// Build a directory of stub executables so detection can be tested without
    /// the real agents installed. Returns the dir to prepend to `PATH`.
    #[cfg(unix)]
    fn stub_bins(dir: &Path, bins: &[(&str, &str)]) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let bin_dir = dir.join("bin");
        std::fs::create_dir_all(&bin_dir).unwrap();
        for (name, version_output) in bins {
            let path = bin_dir.join(name);
            std::fs::write(
                &path,
                format!("#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then echo '{version_output}'; exit 0; fi\nexit 0\n"),
            )
            .unwrap();
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        bin_dir
    }

    #[test]
    #[cfg(unix)]
    fn detects_only_binaries_on_the_given_path() {
        let dir = tmpdir("path");
        let bin_dir = stub_bins(&dir, &[("claude", "1.2.3 (stub)"), ("crush", "crush v0.9")]);
        let probe = Probe {
            path: bin_dir.into(),
            home: None,
        };

        let found = detect_in(&dir, &probe);
        let ids: Vec<_> = found.iter().map(|d| d.spec.id).collect();
        assert_eq!(ids.len(), 2, "got {ids:?}");
        assert!(ids.contains(&"claude"));
        assert!(ids.contains(&"crush"));

        let claude = found.iter().find(|d| d.spec.id == "claude").unwrap();
        assert_eq!(claude.version.as_deref(), Some("1.2.3 (stub)"));
        assert!(
            claude.usable(),
            "a present binary is usable until proven otherwise"
        );
        assert!(!claude.has_credentials, "no home was given, so no creds");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    #[cfg(unix)]
    fn credentials_upgrade_ready_to_verified() {
        let dir = tmpdir("creds");
        let bin_dir = stub_bins(&dir, &[("claude", "1.0")]);

        let no_home = detect_in(
            &dir,
            &Probe {
                path: bin_dir.clone().into(),
                home: None,
            },
        );
        assert_eq!(no_home[0].status, Status::Ready);

        // Create the credential file claude's registry row points at.
        let home = dir.join("home");
        std::fs::create_dir_all(&home).unwrap();
        std::fs::write(home.join(".claude.json"), "{}").unwrap();
        let with_home = detect_in(
            &dir,
            &Probe {
                path: bin_dir.into(),
                home: Some(home),
            },
        );
        assert!(with_home[0].has_credentials);
        assert_eq!(with_home[0].status, Status::Verified);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    #[cfg(unix)]
    fn a_remembered_failure_survives_redetection() {
        let dir = tmpdir("remembered");
        let bin_dir = stub_bins(&dir, &[("claude", "1.0")]);
        let home = dir.join("home");
        std::fs::create_dir_all(&home).unwrap();
        std::fs::write(home.join(".claude.json"), "{}").unwrap();
        let probe = Probe {
            path: bin_dir.into(),
            home: Some(home),
        };

        remember_failure(
            &dir,
            "claude",
            Unavailable::NotLoggedIn("run /login".into()),
        );
        let found = detect_in(&dir, &probe);
        // Credentials exist, but a real call already failed — the verdict must win,
        // otherwise the ranker would keep proposing a broken agent.
        assert!(
            !found[0].usable(),
            "blocked verdict must outrank a credential file"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn empty_path_finds_nothing() {
        let dir = tmpdir("empty-path");
        let found = detect_in(
            &dir,
            &Probe {
                path: Default::default(),
                home: None,
            },
        );
        assert!(found.is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn blocked_verdicts_persist_and_expire() {
        let dir = tmpdir("verdict");
        remember_failure(
            &dir,
            "claude",
            Unavailable::NotLoggedIn("run /login".into()),
        );

        let cache = load_cache(&dir);
        match cache.verdicts.get("claude") {
            Some(Status::Blocked(u)) => assert!(u.reason().contains("not logged in")),
            other => panic!("expected blocked, got {other:?}"),
        }

        // Rewind the stamp past the TTL; the whole cache must be discarded.
        let mut stale = load_cache(&dir);
        stale.stamped = now_secs().saturating_sub(TTL.as_secs() + 60);
        stale
            .verdicts
            .insert("claude".into(), Status::Blocked(Unavailable::Timeout));
        save_cache(&dir, &stale);
        assert!(
            load_cache(&dir).verdicts.is_empty(),
            "stale cache must expire"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn success_overwrites_a_failure() {
        let dir = tmpdir("success");
        remember_failure(&dir, "claude", Unavailable::Timeout);
        remember_success(&dir, "claude");
        assert_eq!(
            load_cache(&dir).verdicts.get("claude"),
            Some(&Status::Verified)
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn reasons_are_single_line_and_bounded() {
        let u = Unavailable::Other("first line\nsecond line".into());
        assert_eq!(u.reason(), "first line");

        let long = Unavailable::Other("x".repeat(500));
        assert!(long.reason().len() <= 140);

        // Leading blank lines are skipped rather than yielding an empty reason.
        let padded = Unavailable::NotLoggedIn("\n\n  please log in  \n".into());
        assert_eq!(padded.reason(), "not logged in: please log in");
    }

    #[test]
    fn only_model_unavailable_suggests_retrying_another_model() {
        assert!(Unavailable::ModelDenied("no opus".into()).might_work_with_another_model());
        assert!(!Unavailable::NotLoggedIn("x".into()).might_work_with_another_model());
        assert!(!Unavailable::QuotaExhausted("x".into()).might_work_with_another_model());
        assert!(!Unavailable::NotInstalled.might_work_with_another_model());
    }

    #[test]
    fn clear_cache_removes_the_file() {
        let dir = tmpdir("clear");
        remember_failure(&dir, "crush", Unavailable::Timeout);
        assert!(cache_path(&dir).exists());
        clear_cache(&dir);
        assert!(!cache_path(&dir).exists());
        std::fs::remove_dir_all(&dir).ok();
    }
}
