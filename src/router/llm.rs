//! The one place that runs the model.
//!
//! Every local inference in pstore — ranking the agents, finding personal data — is a
//! `llama-cli` invocation made from here. The module is deliberately the only thing that
//! knows the command line, so the flags that bound memory (see [`fit_context`]) cannot be
//! forgotten at one call site and applied at another.
//!
//! **One process per call.** There is no resident server: `llama-completion` starts, maps
//! the weights, generates, and exits.
//!
//! **What a call actually costs.** Measured warm on an M4-Pro-class laptop against a
//! fifteen-candidate ranking prompt, on the 1-bit checkpoint this module was first written
//! for:
//!
//! | Phase | Cost |
//! | --- | --- |
//! | process start, mmap and the `-fit` probe | ~1.1 s |
//! | prompt evaluation | ~10 ms/token (~100 tok/s) |
//! | generation | ~41 ms/token (~24 tok/s) |
//!
//! Those rates are the checkpoint's own, not a misconfiguration: PrismML publish 26 tok/s
//! generation and 133 tok/s prompt evaluation for this class of machine, and the numbers
//! here sit right on that. A ranking call lands at **~13 s** with reasoning off and **~26 s**
//! with it, and the ternary weights move roughly twice the bytes per token. This module used
//! to claim ~1.4 s per call and ~27 tok/s of *prompt* evaluation; the first was out by an
//! order of magnitude and the second had the two phases the wrong way round.
//!
//! So both halves are worth minding. Generation costs 4× more per token than prompt
//! evaluation, which is why [`rank_grammar`] permits no whitespace and no long strings;
//! prompt evaluation is linear with no fixed floor, which is why [`rank_prompt`] stays
//! terse. The rule that keeps this affordable is **one invocation per user action**;
//! anything needing more should be one prompt instead.
//!
//! A resident server would buy back the ~1.1 s of startup and, more usefully, let the
//! unchanging head of the prompt — the rules and the candidate list — stay in the KV cache
//! instead of being re-evaluated every call. That is seconds, not milliseconds, and it is
//! the one structural argument for a port to bind and a health check to write. It is not
//! taken here.
//!
//! **Nothing outlives the window.** One process per call is only half of that promise:
//! closing a window does not kill its children, so a generation still in flight would keep
//! 7.17 GB of weights resident with nobody left to show the answer to. Every child is
//! therefore registered while it runs and killed by [`shutdown`], which the app calls on its
//! way out.
//!
//! **The chat template is the model's own** — `--jinja`. This is not a detail. Without that
//! flag llama.cpp silently falls back to a legacy ChatML template, and the checkpoint — a
//! thinking model whose own template opens a `<think>` block — is prompted in a shape it was
//! never trained on. Asked to rank fifteen (model, effort) pairs for a hard three-file
//! refactor it answered `Haiku 4.5, effort medium`, followed by indices 2, 3, 4 and 5 in
//! order, scoring every one of them `fit: 85` with a copy-pasted reason: it was not
//! discriminating at all, it was counting. With `--jinja` and nothing else changed it
//! answered Opus 5 at high effort, then Opus at medium and low, then Sonnet — with fits that
//! descend. One flag.
//!
//! **Reasoning is bounded by the grammar.** Given room to think, the model thinks well: it
//! identifies the task as a hard refactor, rules out the light models by name, and weighs
//! effort against latency. What it does not do is stop — unbounded, one routing call spent
//! 1 399 tokens re-litigating its own conclusion and never reached an answer. So the grammar
//! allows a reasoning block of at most `model_reasoning_budget` characters and then
//! *requires* `</think>`: the budget is enforced by the sampler rather than hoped for. Set it
//! to zero to skip reasoning, which is ~20 s faster and measurably worse.
//!
//! **Output is grammar-constrained.** The sampler cannot emit anything that does not parse,
//! so parsing is a `serde_json` call rather than a best-effort scrape, and a malformed reply
//! is a bug in the grammar rather than a Tuesday. One trap, learned the hard way: never give
//! a JSON grammar an unbounded whitespace rule. `ws ::= [ \t\n]*` between two tokens is a
//! legal place to emit spaces forever, and the model did exactly that — hundreds of tokens
//! of blanks, then the generation limit, and no answer. Every rule here is bounded.

use std::io::Read;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use serde_json::{Value, json};

use crate::models;
use crate::runtime;

/// Rough characters per token, for sizing the context window.
///
/// Deliberately pessimistic. Real Qwen-family tokenizers average nearer 3.7 on English
/// prose and better on code; 3.0 over-counts, which is the safe direction — see
/// [`fit_context`].
const CHARS_PER_TOKEN: f32 = 3.0;

/// Smallest context worth asking for. Below this the savings are noise and the risk of
/// clipping a prompt is not.
const MIN_CONTEXT: usize = 512;

/// Context is requested in multiples of this, so nearly-identical prompts reuse the same
/// allocation size rather than each landing on its own.
const CONTEXT_STEP: usize = 256;

/// Size the context window to the work actually being done.
///
/// The checkpoint natively supports 262 144 tokens. Running there would cost ~12 GB of KV
/// cache for prompts that are, in every one of pstore's uses, a few hundred tokens. So the
/// window is fitted per call: at the sizes pstore asks for, the cache is tens of megabytes
/// and the 7.17 GB of weights is essentially the entire footprint.
///
/// The estimate errs high on purpose. `llama-cli` silently truncates a prompt that does not
/// fit, which would mean PII spans pointing into text the model never saw — a wrong answer
/// that looks like a right one. Over-estimating costs a few megabytes; under-estimating
/// costs correctness.
pub fn fit_context(prompt: &str, max_output_tokens: usize, ceiling: usize) -> usize {
    let prompt_tokens = (prompt.len() as f32 / CHARS_PER_TOKEN).ceil() as usize;
    // 25% headroom over an already-pessimistic estimate, plus the chat template's own
    // wrapper tokens, which are not in `prompt`.
    let needed = prompt_tokens + prompt_tokens / 4 + max_output_tokens + 256;
    let stepped = needed.div_ceil(CONTEXT_STEP) * CONTEXT_STEP;
    stepped.clamp(MIN_CONTEXT, ceiling.max(MIN_CONTEXT))
}

/// What the runtime and the weights add up to: either a way to run the model, or the
/// reason there isn't one.
///
/// Returns the selected checkpoint alongside the paths, because everything downstream reports
/// progress against *its* row on the status board. Reading [`models::active`] once here and
/// passing it along is what stops a mid-call preference change from marking one build's row
/// `Ready` on the strength of the other build's run.
fn ready() -> Result<(PathBuf, PathBuf, models::Checkpoint), String> {
    let prefs = crate::config::prefs_snapshot();

    let rt = runtime::locate(prefs.llama_cli_path.as_deref())
        .ok_or_else(|| runtime::missing_reason(prefs.llama_cli_path.as_deref()))?;

    let checkpoint = prefs.local_model.checkpoint();
    if !models::is_cached(&checkpoint) {
        models::set(checkpoint.id, models::Phase::Absent);
        return Err(format!(
            "{} not downloaded — open the Models window to fetch it ({}), or switch to the \
             other build there",
            checkpoint.title,
            checkpoint.size_label()
        ));
    }
    let weights = hub_path(&checkpoint)?;
    Ok((rt.path, weights, checkpoint))
}

/// Resolve a checkpoint's path in the shared Hugging Face cache.
fn hub_path(checkpoint: &models::Checkpoint) -> Result<PathBuf, String> {
    let file = checkpoint
        .files
        .last()
        .expect("the checkpoint lists its weights");
    super::hub::cached(checkpoint.repo, file)
}

/// How the sampler is to be constrained — and, with it, how it should sample.
///
/// Two mechanisms because two jobs. A JSON Schema is the better thing to write and maintain,
/// and it is what the extraction path wants; but it constrains the very first sampled token,
/// which leaves no room for a reasoning block. Where reasoning earns its seconds — ranking —
/// the grammar is written out by hand instead.
pub enum Constrain<'a> {
    /// Compile a JSON Schema into the sampler. No reasoning is possible.
    Schema(&'a Value),
    /// A GBNF grammar, which may allow a `<think>` block before the JSON.
    Grammar(String),
}

impl Constrain<'_> {
    /// Whether this call is asking the model to reason before answering.
    ///
    /// It decides the sampling settings, and the difference is not cosmetic. The
    /// checkpoint's published `temp 0.7 / top-p 0.95 / top-k 20` are the settings **its
    /// thinking-mode benchmarks were run at**, and reasoning at temperature zero collapses
    /// into repetition — the same `reason` string on all five picks, every time.
    ///
    /// Extraction is the opposite case and wants greedy decoding. Finding an address in a
    /// paragraph has one right answer, and there is no deliberation for temperature to
    /// diversify — only a copy to get exactly right. Running the personal-data scan at 0.7
    /// cost it a live finding: asked for the personal data in
    /// `Contact Mario Rossi at mario@example.com`, it returned the name and dropped the
    /// address. Sampling settings are not a global taste; they belong to the task.
    fn reasons(&self) -> bool {
        matches!(self, Constrain::Grammar(g) if g.contains(END_OF_THOUGHT))
    }
}

/// Run the model once and return its (grammar-constrained) JSON reply.
///
/// Blocking, and slow in units of seconds rather than milliseconds — see the module header
/// for the breakdown. Call it from a worker thread. `max_output_tokens` bounds both
/// generation and the context window, so it has to cover the reasoning block as well as the
/// JSON, or the grammar will still be waiting for `</think>` when the token budget runs out.
///
/// Ends early, with the reason, if [`shutdown`] runs while the model is generating.
pub fn complete_json(
    prompt: &str,
    constrain: Constrain<'_>,
    max_output_tokens: usize,
) -> Result<Value, String> {
    let (binary, weights, checkpoint) = ready()?;
    let ceiling = crate::config::prefs_snapshot().model_context_ceiling;
    let ctx = fit_context(prompt, max_output_tokens, ceiling);

    let mut cmd = Command::new(&binary);
    let reasons = constrain.reasons();
    match constrain {
        Constrain::Schema(schema) => cmd.arg("--json-schema").arg(schema.to_string()),
        Constrain::Grammar(gbnf) => cmd.arg("--grammar").arg(gbnf),
    };
    cmd.arg("-m")
        .arg(&weights)
        .arg("-p")
        .arg(prompt)
        // The model's own chat template, not llama.cpp's legacy ChatML guess. Without this
        // the checkpoint is prompted in a shape it was not trained on and ranks visibly
        // worse — see the module header, where the before and after are written out.
        .arg("--jinja")
        // Echoing the prompt back would double the output for nothing.
        .arg("--no-display-prompt")
        // Closed stdin is what makes it exit. Even `llama-completion` drops into a prompt
        // after generating; with no stdin it takes the EOF and leaves. `--log-disable` is
        // deliberately *not* used — it silences the load errors on stderr too, which are
        // the only diagnostic when a run fails.
        .stdin(Stdio::null())
        .args(["-n", &max_output_tokens.to_string()])
        .args(["-c", &ctx.to_string()])
        // Memory: 4-bit KV cache and flash attention. Small at these context sizes, but
        // free, and they keep the ceiling honest if someone raises it.
        .args(["--cache-type-k", "q4_0", "--cache-type-v", "q4_0"])
        .args(["--flash-attn", "on"])
        // Nothing warms up or gets pinned: this process exists for one generation.
        .args(["--no-warmup", "-ngl", "999"])
        // Pinned either way, so a call is reproducible whichever branch it takes.
        .args(["--seed", "1"]);

    // Sampling belongs to the task, not to the module — see `Constrain::reasons`.
    if reasons {
        cmd.args(["--temp", "0.7", "--top-p", "0.95", "--top-k", "20"]);
    } else {
        cmd.args(["--temp", "0"]);
    }

    models::set(checkpoint.id, models::Phase::Loading);
    let run = supervise(cmd).map_err(|e| {
        models::set(checkpoint.id, models::Phase::Failed(e.clone()));
        format!("running {}: {e}", binary.display())
    })?;

    if run.stopped {
        // Nothing is wrong with the model or the machine, so this is not a `Failed` phase:
        // the app is closing and took the process with it.
        models::set(checkpoint.id, models::Phase::Cached);
        return Err("the model was stopped because pstore is closing".into());
    }

    if !run.ok {
        let why = run.stderr.trim();
        // The tail is the useful part; the head is load-time chatter about tensors.
        let tail: String = why.lines().rev().take(4).collect::<Vec<_>>().join(" / ");
        models::set(checkpoint.id, models::Phase::Failed(tail.clone()));
        return Err(format!("llama-cli failed: {tail}"));
    }

    models::set(checkpoint.id, models::Phase::Ready);
    parse_reply(&run.stdout)
}

/// How often a run looks to see whether its child has finished.
///
/// The child is polled rather than waited on: a blocking `wait` would hold its lock for the
/// whole generation, leaving [`shutdown`] queued behind the very thing it is trying to kill.
/// 20 ms against a call measured in seconds is noise.
const POLL: Duration = Duration::from_millis(20);

/// Every model process alive right now, keyed by pid.
static LIVE: Mutex<Vec<(u32, Arc<Mutex<Child>>)>> = Mutex::new(Vec::new());

/// Raised once the app is on its way out, so no further weights are mapped.
static CLOSING: AtomicBool = AtomicBool::new(false);

/// A poisoned lock means a thread panicked mid-update. Recovering the guard is better than
/// refusing to kill the process it was holding.
fn recover<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    match m.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// One finished — or killed — model process.
struct Run {
    /// Whether it exited successfully.
    ok: bool,
    stdout: String,
    stderr: String,
    /// Whether it was killed on the way out rather than left to finish.
    stopped: bool,
}

/// Run `cmd` to completion, killing it if the app closes first.
///
/// The child is registered in [`LIVE`] for the whole of its life, which is what makes
/// [`shutdown`] possible. Both pipes are drained on their own threads: `llama-completion`
/// writes kilobytes of load-time chatter to stderr, and an unread pipe would stall the
/// generation rather than fail it.
fn supervise(mut cmd: Command) -> Result<Run, String> {
    // Refused rather than started: a worker that reached here between resolving the weights
    // and spawning must not map 7.17 GB as the window disappears.
    if CLOSING.load(Ordering::Relaxed) {
        return Ok(Run {
            ok: false,
            stdout: String::new(),
            stderr: String::new(),
            stopped: true,
        });
    }

    let mut child = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| e.to_string())?;

    let mut out_pipe = child.stdout.take();
    let mut err_pipe = child.stderr.take();
    let out = std::thread::spawn(move || drain(out_pipe.as_mut()));
    let err = std::thread::spawn(move || drain(err_pipe.as_mut()));

    let pid = child.id();
    let child = Arc::new(Mutex::new(child));
    recover(&LIVE).push((pid, Arc::clone(&child)));

    // The lock is only ever held across a non-blocking `try_wait`, so `shutdown` can take
    // it at any point during the generation.
    let status = loop {
        match recover(&child).try_wait() {
            Ok(Some(s)) => break Some(s),
            // Already reaped by `shutdown`, or the child is gone. Either way it is over.
            Err(_) => break None,
            Ok(None) => {}
        }
        std::thread::sleep(POLL);
    };
    recover(&LIVE).retain(|(p, _)| *p != pid);

    let ok = status.is_some_and(|s| s.success());
    Ok(Run {
        ok,
        stdout: out.join().unwrap_or_default(),
        stderr: err.join().unwrap_or_default(),
        // A run that finished cleanly as the app closed still counts as finished — its
        // answer is good, even if nothing is left to display it.
        stopped: !ok && CLOSING.load(Ordering::Relaxed),
    })
}

fn drain<R: Read>(pipe: Option<&mut R>) -> String {
    let mut s = String::new();
    if let Some(p) = pipe {
        let _ = p.read_to_string(&mut s);
    }
    s
}

/// Kill and reap every model process running now. Returns how many there were.
fn stop_live() -> usize {
    // Taken rather than iterated, so a run that finishes by itself in the meantime cannot be
    // waited on from two places.
    let live: Vec<(u32, Arc<Mutex<Child>>)> = std::mem::take(&mut *recover(&LIVE));
    for (_, child) in &live {
        let mut c = recover(child);
        // Both calls fail on a child that has just exited on its own, which is the state
        // this function exists to reach.
        let _ = c.kill();
        // Reaped here rather than left to the supervising thread: a zombie holds its slot,
        // and the parent is about to leave.
        let _ = c.wait();
    }
    live.len()
}

/// Where the model's reasoning stops and its answer starts.
const END_OF_THOUGHT: &str = "</think>";

/// Pull the JSON object out of `llama-cli`'s stdout.
///
/// The grammar guarantees the *model's* output parses, but the process still prints its own
/// framing around it, and which framing depends on build flags. Rather than depend on that,
/// take the outermost braces.
///
/// The reasoning block is dropped first, and that ordering matters: reasoning about a routing
/// decision quotes JSON, braces and all, so scanning for the first `{` across the whole reply
/// would parse the model's rough notes instead of its answer. Everything up to the *last*
/// `</think>` goes.
fn parse_reply(stdout: &str) -> Result<Value, String> {
    let stdout = match stdout.rfind(END_OF_THOUGHT) {
        Some(i) => &stdout[i + END_OF_THOUGHT.len()..],
        None => stdout,
    };
    let start = stdout
        .find('{')
        .ok_or_else(|| format!("no JSON in the model's reply: {:?}", truncate(stdout, 200)))?;
    let end = stdout.rfind('}').ok_or_else(|| {
        format!(
            "truncated JSON in the model's reply: {:?}",
            truncate(stdout, 200)
        )
    })?;
    if end < start {
        return Err(format!(
            "malformed JSON in the model's reply: {:?}",
            truncate(stdout, 200)
        ));
    }
    serde_json::from_str(&stdout[start..=end])
        .map_err(|e| format!("could not parse the model's reply: {e}"))
}

fn truncate(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

// ---------------------------------------------------------------------------
// Ranking
// ---------------------------------------------------------------------------

/// Longest `reason` the grammar will allow, in characters.
///
/// Every one of these is a token the model spends at ~41 ms, five times over, so the cap is
/// tight and the prompt asks for twelve words. A reason cut off mid-word reads as a bug, so
/// the prompt's instruction and this number have to stay in agreement.
const REASON_CHARS: usize = 90;

/// Rank `candidates` against `text`.
///
/// The model chooses **by index** into the list it was given, never by naming an agent or a
/// model string. An index either maps onto a real launch configuration or is rejected; a
/// name could be plausible and wrong, and pstore would then try to launch it.
///
/// Expect **seconds**: ~13 s with reasoning off and ~30–40 s with it, scaling with the
/// candidate list. This is the call the whole "one invocation per user action" rule exists to
/// ration.
pub fn rank(
    text: &str,
    candidates: &[super::Candidate],
    excluded: Vec<(&'static str, String)>,
) -> Result<super::Ranking, String> {
    let started = std::time::Instant::now();
    let want = super::SHORTLIST.min(candidates.len());
    let budget = crate::config::prefs_snapshot().model_reasoning_budget;

    let reply = complete_json(
        &rank_prompt(text, candidates, want),
        Constrain::Grammar(rank_grammar(candidates.len(), want, budget)),
        rank_output_tokens(want, budget),
    )?;
    let mut choices = build_choices(&reply, candidates)?;
    normalise_latency(&mut choices);

    Ok(super::Ranking {
        choices,
        considered: candidates.len(),
        excluded,
        elapsed: started.elapsed(),
    })
}

/// Generation budget for a ranking call: the reasoning block plus the JSON.
///
/// Undersizing this is not a slow answer but no answer — the grammar is still waiting for
/// `</think>` when the tokens run out, and the reply has no JSON in it at all. So the
/// reasoning allowance is converted at the pessimistic [`CHARS_PER_TOKEN`] and then given
/// room to spare.
fn rank_output_tokens(want: usize, reasoning_budget: usize) -> usize {
    // ~35 tokens covers one `{"index":12,"fit":95,"reason":"..."}` with a full-length reason.
    let json = 16 + want * 40;
    let thought = if reasoning_budget == 0 {
        0
    } else {
        (reasoning_budget as f32 / CHARS_PER_TOKEN).ceil() as usize + 16
    };
    json + thought
}

/// The grammar for a ranking reply: an optional bounded reasoning block, then the JSON.
///
/// Written by hand rather than compiled from a JSON Schema, because a schema constrains the
/// first sampled token and the whole point here is to leave room for `<think>` first. Three
/// things are deliberate:
///
/// * **No whitespace rule at all.** The JSON is emitted dense. An unbounded `ws` rule between
///   tokens is somewhere the sampler can legally sit forever, and it does.
/// * **`index` is an alternation of the literal indices**, not a digit pattern with a range
///   check afterwards. The list is short and this way an out-of-range pick is unrepresentable
///   rather than merely rejected — [`build_choices`] still drops what does not map, but it
///   should never have to.
/// * **Every repetition is bounded.** `{0,n}` throughout, so a run cannot become unbounded by
///   any path through the grammar.
fn rank_grammar(candidates: usize, want: usize, reasoning_budget: usize) -> String {
    use std::fmt::Write;

    // The template has already opened the block, so the grammar only has to close it.
    let root = if reasoning_budget == 0 {
        "root ::= answer".to_string()
    } else {
        format!(
            "root ::= thought \"{END_OF_THOUGHT}\" answer\nthought ::= [^<]{{0,{reasoning_budget}}}"
        )
    };

    let mut index = String::new();
    for i in 0..candidates.max(1) {
        if i > 0 {
            index.push_str(" | ");
        }
        let _ = write!(index, "\"{i}\"");
    }

    // `want - 1` repetitions after the first, so the array is exactly `want` long the way
    // `minItems`/`maxItems` used to make it.
    let more = want.saturating_sub(1);
    format!(
        "{root}\n\
         answer ::= \"{{\\\"choices\\\":[\" choice (\",\" choice){{{more}}} \"]}}\"\n\
         choice ::= \"{{\\\"index\\\":\" index \",\\\"fit\\\":\" fit \",\\\"reason\\\":\" reason \"}}\"\n\
         index ::= {index}\n\
         fit ::= [0-9] | [1-9] [0-9] | \"100\"\n\
         reason ::= \"\\\"\" [^\"\\\\\\n]{{0,{REASON_CHARS}}} \"\\\"\"\n"
    )
}

/// Build the ranking prompt.
///
/// The candidate list carries price and metering because the model is being asked to make
/// the judgement pstore used to encode by hand — and "this one bills per token" is exactly
/// the sort of thing that should shift a recommendation, but only when the extra capability
/// is actually needed.
fn rank_prompt(text: &str, candidates: &[super::Candidate], want: usize) -> String {
    let mut list = String::new();
    for (i, c) in candidates.iter().enumerate() {
        use std::fmt::Write;
        // Terse on purpose. Prompt evaluation runs at ~27 tokens/second on this checkpoint,
        // so the candidate list — thirty lines on a machine with several agents installed —
        // dominates the cost of a ranking call. Everything here earns its tokens: the two
        // words that changed a routing decision in testing are the tier and the effort.
        // "included in subscription" was dropped because it is the default and saying so
        // thirty times cost more than the one line that flags the exception.
        let _ = write!(
            list,
            "\n{i}: {} via {} [{}, effort {}{}]{}",
            c.model_display,
            c.agent_display,
            c.tier,
            c.effort,
            if c.effort_selectable { "" } else { "?" },
            if c.metered { " PAID-PER-TOKEN" } else { "" },
        );
    }

    format!(
        "Pick the {want} best options for the prompt below, best first.\n\
         \n\
         Rules:\n\
         - Match capability to demand: no frontier model for simple work, no light model \
         for a hard multi-file refactor.\n\
         - Higher effort costs latency. Prefer lower unless the prompt needs the reasoning.\n\
         - PAID-PER-TOKEN costs extra money; rank one high only if clearly better than \
         every other option here.\n\
         - `?` after the effort means it cannot be set, only predicted.\n\
         - `fit` is 0-100 for THIS prompt, and the five must differ.\n\
         - `reason`: under 10 words, about THIS option. A different reason for each — \
         say what distinguishes it from the pick above.\n\
         \n\
         Options:{list}\n\
         \n\
         Prompt to route:\n\
         <prompt>\n{text}\n</prompt>\n"
    )
}

/// Map the model's indices back onto real candidates.
///
/// Duplicates are dropped rather than repeated: the grammar cannot express "distinct", so a
/// model that picks the same option twice would otherwise produce a shortlist that looks
/// like a bug in pstore.
fn build_choices(
    reply: &Value,
    candidates: &[super::Candidate],
) -> Result<Vec<super::Choice>, String> {
    let raw = reply
        .get("choices")
        .and_then(Value::as_array)
        .ok_or("the model's reply had no choices")?;

    let mut seen = Vec::new();
    let mut out = Vec::new();
    for item in raw {
        let idx = item
            .get("index")
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("a choice had no usable index: {item}"))?
            as usize;
        let Some(c) = candidates.get(idx) else {
            // The grammar bounds this, so reaching here means the schema and the list
            // disagree — skip rather than launching something arbitrary.
            continue;
        };
        if seen.contains(&idx) {
            continue;
        }
        seen.push(idx);

        out.push(super::Choice {
            agent_id: c.agent_id,
            agent_display: c.agent_display,
            model_id: c.model_id,
            model_display: c.model_display,
            tier: c.tier,
            effort: c.effort,
            effort_selectable: c.effort_selectable,
            metered: c.metered,
            relative_latency: c.effort.latency_factor(),
            relative_price: c.relative_price,
            fit: item.get("fit").and_then(Value::as_f64).unwrap_or(0.0) as f32,
            rationale: item
                .get("reason")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .trim()
                .to_string(),
        });
    }

    if out.is_empty() {
        return Err("the model returned no usable choices".into());
    }
    // Emitted order **is** the ranking, and it is deliberately not re-sorted by `fit`.
    //
    // Against the real checkpoint, a "fix a typo" prompt came back correctly ordered — the
    // light model first, with a reason saying so — but scored `fit: 0` and `fit: 1`. Sorting
    // on those numbers inverted the answer the model had actually given. The ordering is
    // what it was asked for and what it gets right; `fit` is a self-assessment that is
    // useful for the table and the hint tolerance, and not to be trusted as a sort key.
    Ok(out)
}

/// Normalise latency against the fastest choice present, so the column reads as "×
/// slower than the quickest option here" rather than as an absolute.
pub fn normalise_latency(choices: &mut [super::Choice]) {
    if let Some(min) = choices
        .iter()
        .map(|c| c.relative_latency)
        .fold(None::<f32>, |acc, v| Some(acc.map_or(v, |a: f32| a.min(v))))
        && min > 0.0
    {
        for c in choices {
            c.relative_latency /= min;
        }
    }
}

// ---------------------------------------------------------------------------
// Personal data
// ---------------------------------------------------------------------------

/// PII output scales with the input, unlike ranking. A chunk is [`crate::pii::CHUNK_CHARS`]
/// characters; this bounds the reply at roughly one finding per 40 characters of input,
/// which no real prompt approaches.
const PII_OUTPUT_TOKENS: usize = 1200;

/// Find personal data in `text`.
///
/// Returns spans with byte offsets into `text`. The model is asked for the matched text
/// rather than trusted with offsets — see [`locate_spans`].
///
/// **One process per chunk, and they run in sequence.** At [`crate::pii::CHUNK_CHARS`] a
/// six-thousand-character prompt is four calls, so a scan is four times a ranking call's
/// worth of seconds. That is the one place in pstore where the "one invocation per user
/// action" rule is broken, and it is broken on purpose: a single call would have to hold the
/// whole document, and the context ceiling exists to keep the KV cache small.
///
/// Reasoning is deliberately **not** enabled here, unlike [`rank`]. Finding an address in a
/// paragraph is extraction, not judgement — there is no deliberation to be had — and the cost
/// would be paid once per chunk.
pub fn detect_pii(text: &str) -> Result<Vec<crate::pii::Finding>, String> {
    let schema = json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["findings"],
        "properties": {
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["tag", "text"],
                    "properties": {
                        "tag": {"type": "string", "enum": crate::pii::TAGS},
                        "text": {"type": "string", "minLength": 1, "maxLength": 200}
                    }
                }
            }
        }
    });

    let mut out = Vec::new();
    // Long prompts are chunked so no single call approaches the context ceiling. Offsets
    // are translated back to the whole document by `locate_spans`.
    for (offset, chunk) in crate::pii::segments(text, crate::pii::CHUNK_CHARS) {
        let reply = complete_json(
            &pii_prompt(chunk),
            Constrain::Schema(&schema),
            PII_OUTPUT_TOKENS,
        )?;
        let findings = reply
            .get("findings")
            .and_then(Value::as_array)
            .ok_or("the model's reply had no findings")?;

        let pairs: Vec<(String, String)> = findings
            .iter()
            .filter_map(|f| {
                Some((
                    f.get("tag")?.as_str()?.to_string(),
                    f.get("text")?.as_str()?.to_string(),
                ))
            })
            .collect();

        out.extend(locate_spans(chunk, offset, &pairs));
    }
    Ok(out)
}

/// Turn `(tag, text)` pairs into spans by finding the text in the source.
///
/// The model is never asked for byte offsets. It cannot count bytes reliably — especially
/// across multi-byte characters, which is exactly where Italian and German names live — and
/// a wrong offset does not fail loudly: it masks the wrong span, leaving the real personal
/// data in the prompt while corrupting something else. Searching for the returned text
/// instead means a span either exists in the document or is dropped.
///
/// The cursor only moves forward, so repeated values (a name mentioned three times) match
/// three distinct occurrences rather than the same one three times.
fn locate_spans(
    chunk: &str,
    offset: usize,
    pairs: &[(String, String)],
) -> Vec<crate::pii::Finding> {
    let mut out = Vec::new();
    let mut cursors: Vec<(&str, usize)> = Vec::new();

    for (tag, needle) in pairs {
        if needle.trim().is_empty() {
            continue;
        }
        let from = cursors
            .iter()
            .find(|(n, _)| *n == needle.as_str())
            .map(|(_, c)| *c)
            .unwrap_or(0);
        let Some(rel) = chunk.get(from..).and_then(|hay| hay.find(needle.as_str())) else {
            // The model quoted something that is not in the text. Dropping it is the whole
            // point of locating rather than trusting: a hallucinated span cannot mask
            // anything.
            continue;
        };
        let start = from + rel;
        let end = start + needle.len();

        match cursors.iter_mut().find(|(n, _)| *n == needle.as_str()) {
            Some((_, c)) => *c = end,
            None => cursors.push((needle.as_str(), end)),
        }

        out.push(crate::pii::Finding {
            tag: tag.clone(),
            start: offset + start,
            end: offset + end,
            text: needle.clone(),
            score: 1.0,
        });
    }
    out
}

fn pii_prompt(chunk: &str) -> String {
    format!(
        "Find every piece of personal data in the text below.\n\
         \n\
         For each one, return the tag and the EXACT text as it appears — copy it \
         character for character, do not normalise, reformat or correct it.\n\
         \n\
         Tags: {tags}\n\
         \n\
         Notes:\n\
         - CF is an Italian codice fiscale, PIVA an Italian VAT number, TARGA a vehicle \
         plate, CATASTO a land-registry reference.\n\
         - Include identifiers even when they look like examples or placeholders.\n\
         - Do not report code identifiers, file paths, library names or API endpoints.\n\
         - Return an empty list if there is no personal data.\n\
         \n\
         <text>\n{chunk}\n</text>\n",
        tags = crate::pii::TAGS.join(", ")
    )
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

/// Check that the model could run, without running it.
///
/// Called from the Models window so a missing runtime or checkpoint is something the user
/// can see and fix, rather than a failure that first appears when they press a button.
pub fn preload() -> Result<(), String> {
    ready().map(|(_, _, checkpoint)| {
        models::set(checkpoint.id, models::Phase::Cached);
    })
}

/// Re-check the runtime and checkpoint on the next call.
///
/// Nothing is cached in this process — each call re-resolves and re-spawns — so this only
/// has to correct the status board, which may be showing a stale failure.
pub fn reset() {
    models::probe_cache();
}

/// Let go of the model, because pstore is closing.
///
/// The weights live in the child's address space, not this one, so "unloading" the model is
/// exactly this: end the process holding it. Killing the GUI does not kill its children — a
/// ranking or a PII scan still generating would go on holding 7.17 GB of resident pages, with
/// no window left to show the answer to — so every live run is killed and reaped here, and
/// further runs are refused so nothing maps the weights again on the way out.
///
/// Idempotent, quick, and safe to call when nothing is running. Returns once the processes
/// are gone, which is the point: returning early would leave the very orphan it exists to
/// prevent.
pub fn shutdown() {
    CLOSING.store(true, Ordering::Relaxed);
    if stop_live() > 0 {
        // Weights on disk, nothing running: what is true a moment before the app exits. Set
        // on whichever build was selected — the run that was killed can only have been that
        // one, and marking the other build's row would be a claim about a file that may not
        // even be downloaded.
        models::set(models::active().id, models::Phase::Cached);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::registry::{Effort, Tier};

    fn candidate(
        agent: &'static str,
        model: &'static str,
        effort: Effort,
    ) -> super::super::Candidate {
        super::super::Candidate {
            agent_id: agent,
            agent_display: "Agent",
            model_id: model,
            model_display: model,
            tier: Tier::Mid,
            effort,
            effort_selectable: true,
            metered: false,
            relative_price: 1.0,
        }
    }

    /// The whole memory argument rests on this: a short prompt must not open a large
    /// window. If this regresses, every call quietly costs hundreds of megabytes more.
    #[test]
    fn context_is_fitted_to_the_prompt() {
        let small = fit_context("rank this", 400, 8192);
        assert!(
            small <= 1024,
            "a nine-character prompt asked for {small} tokens of context"
        );
        assert!(small >= MIN_CONTEXT);
        assert_eq!(small % CONTEXT_STEP, 0, "context should be stepped");

        // Bigger prompt, bigger window — but still nowhere near the native 262k.
        let big = fit_context(&"x".repeat(6_000), 400, 8192);
        assert!(big > small);
        assert!(big <= 8192);
    }

    /// Clipping a prompt is silent and produces confidently wrong answers, so the estimate
    /// has to be generous rather than tight.
    #[test]
    fn context_estimate_errs_high() {
        let text = "Refactor the authentication layer across three files.".repeat(20);
        // A real tokenizer will not exceed one token per two characters for prose like
        // this; the fitted window must clear that comfortably.
        let generous_upper_bound = text.len() / 2;
        assert!(
            fit_context(&text, 100, 32_768) > generous_upper_bound,
            "fitted context leaves no headroom over a pessimistic token count"
        );
    }

    /// The ceiling is a hard bound, so a user who lowers it to fit a smaller machine
    /// actually gets a smaller allocation.
    #[test]
    fn ceiling_bounds_the_window() {
        let huge = "x".repeat(200_000);
        assert_eq!(fit_context(&huge, 400, 4096), 4096);
        assert_eq!(fit_context(&huge, 400, 2048), 2048);
        // An absurdly low ceiling still leaves a usable window rather than zero.
        assert_eq!(fit_context(&huge, 400, 1), MIN_CONTEXT);
    }

    /// `llama-cli` prints its own framing around the generation, and which framing depends
    /// on how it was built — so the parser takes the outermost object rather than trusting
    /// the output to be bare JSON.
    #[test]
    fn reply_parsing_ignores_surrounding_noise() {
        let v = parse_reply("load time = 1.2s\n{\"choices\":[]}\n\nllama_perf: 12ms").unwrap();
        assert!(v.get("choices").is_some());

        assert!(parse_reply("no braces here").is_err());
        assert!(parse_reply("} backwards {").is_err());
        assert!(parse_reply("{not json}").is_err());
    }

    /// The model picks by index; anything that does not map onto a real candidate must be
    /// dropped rather than launched.
    #[test]
    fn choices_map_back_onto_real_candidates() {
        let cands = [
            candidate("claude", "sonnet", Effort::Medium),
            candidate("codex", "gpt", Effort::High),
        ];
        let reply = json!({"choices": [
            {"index": 1, "fit": 80, "reason": "strong at refactors"},
            {"index": 0, "fit": 90, "reason": "cheaper and enough"},
        ]});

        let out = build_choices(&reply, &cands).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].model_id, "gpt", "emitted order is the ranking");
        assert_eq!(out[0].rationale, "strong at refactors");
        assert_eq!(out[1].model_id, "sonnet");
        // Latency comes from the registry, not from the model.
        assert_eq!(out[0].relative_latency, Effort::High.latency_factor());
    }

    /// Regression from a live run: the checkpoint ordered a shortlist correctly but scored
    /// it `fit: 0` then `fit: 1`. Re-sorting on `fit` inverted the answer it had given, so
    /// the emitted order has to win.
    #[test]
    fn a_useless_fit_scale_does_not_reorder_the_ranking() {
        let cands = [
            candidate("claude", "haiku", Effort::Low),
            candidate("claude", "opus", Effort::High),
        ];
        let reply = json!({"choices": [
            {"index": 0, "fit": 0, "reason": "simple task, light model is enough"},
            {"index": 1, "fit": 1, "reason": "more than this needs"},
        ]});

        let out = build_choices(&reply, &cands).unwrap();
        assert_eq!(
            out[0].model_id, "haiku",
            "the model's own ordering must survive its own scoring"
        );
    }

    #[test]
    fn duplicate_and_out_of_range_choices_are_dropped() {
        let cands = [candidate("claude", "sonnet", Effort::Medium)];
        let reply = json!({"choices": [
            {"index": 0, "fit": 90, "reason": "good"},
            {"index": 0, "fit": 70, "reason": "same option again"},
            {"index": 99, "fit": 99, "reason": "does not exist"},
        ]});

        let out = build_choices(&reply, &cands).unwrap();
        assert_eq!(out.len(), 1, "one real option, listed once");
        assert_eq!(out[0].fit, 90.0);

        // Nothing usable at all is an error, not an empty shortlist the UI would render as
        // "no models fit".
        let none = json!({"choices": [{"index": 99, "fit": 99, "reason": "nope"}]});
        assert!(build_choices(&none, &cands).is_err());
    }

    /// Byte offsets come from searching the source, never from the model. This is what
    /// makes a hallucinated finding harmless.
    #[test]
    fn spans_are_located_in_the_source() {
        let chunk = "Contact Mario Rossi at mario@example.com about it.";
        let pairs = [
            ("FULLNAME".to_string(), "Mario Rossi".to_string()),
            ("EMAIL".to_string(), "mario@example.com".to_string()),
            ("FULLNAME".to_string(), "Giulia Bianchi".to_string()),
        ];
        let found = locate_spans(chunk, 0, &pairs);

        assert_eq!(found.len(), 2, "the invented name should be dropped");
        for f in &found {
            assert_eq!(&chunk[f.start..f.end], f.text, "span must match its text");
        }
        assert_eq!(found[0].tag, "FULLNAME");
        assert_eq!(found[0].start, 8);
    }

    /// Multi-byte text is where offsets guessed by a model go wrong, and where masking the
    /// wrong span would leave real personal data in place.
    #[test]
    fn spans_survive_multibyte_text() {
        let chunk = "Società di Niccolò Rossè, IBAN IT60X0542811101000000123456.";
        let pairs = [
            ("FULLNAME".to_string(), "Niccolò Rossè".to_string()),
            (
                "IBAN".to_string(),
                "IT60X0542811101000000123456".to_string(),
            ),
        ];
        let found = locate_spans(chunk, 0, &pairs);

        assert_eq!(found.len(), 2);
        for f in &found {
            assert_eq!(&chunk[f.start..f.end], f.text);
        }
    }

    /// A value that appears twice must produce two distinct spans, or the second occurrence
    /// stays unmasked in the sanitised prompt.
    #[test]
    fn repeated_values_match_distinct_occurrences() {
        let chunk = "Mario Rossi wrote it; ask Mario Rossi again.";
        let pairs = [
            ("FULLNAME".to_string(), "Mario Rossi".to_string()),
            ("FULLNAME".to_string(), "Mario Rossi".to_string()),
        ];
        let found = locate_spans(chunk, 0, &pairs);

        assert_eq!(found.len(), 2);
        assert_ne!(found[0].start, found[1].start, "same span found twice");
        assert!(
            found[1].start > found[0].start,
            "cursor should move forward"
        );
    }

    /// Chunk offsets have to be translated back to the whole document, or every finding
    /// past the first chunk masks the wrong part of the prompt.
    #[test]
    fn chunk_offsets_are_translated_to_the_document() {
        let chunk = "email mario@example.com";
        let pairs = [("EMAIL".to_string(), "mario@example.com".to_string())];
        let found = locate_spans(chunk, 1_000, &pairs);
        assert_eq!(found[0].start, 1_006);
        assert_eq!(found[0].end, 1_006 + "mario@example.com".len());
    }

    // -----------------------------------------------------------------------
    // Process lifetime
    // -----------------------------------------------------------------------

    /// `CLOSING` is process-wide and refuses *every* run, so the test that raises it has to
    /// keep other tests out of `supervise` while it does. Held by every test here.
    ///
    /// The registry itself is deliberately **not** guarded: other tests reach the real model
    /// through `complete_json`, so their children appear in `LIVE` alongside these ones.
    /// That is exactly the situation a closing app faces, so the tests below identify their
    /// own child by pid rather than assuming the registry is theirs.
    static PROCESSES: Mutex<()> = Mutex::new(());

    /// Pids currently registered as live.
    fn live_pids() -> Vec<u32> {
        recover(&LIVE).iter().map(|(p, _)| *p).collect()
    }

    /// Wait for `f`, up to `secs`, rather than sleeping a fixed amount.
    fn until(secs: u64, mut f: impl FnMut() -> bool) -> bool {
        for _ in 0..(secs * 100) {
            if f() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        false
    }

    fn sh(script: &str) -> Command {
        let mut cmd = Command::new("/bin/sh");
        cmd.arg("-c").arg(script).stdin(Stdio::null());
        cmd
    }

    #[test]
    #[cfg(unix)]
    fn a_supervised_run_captures_both_streams() {
        let _guard = recover(&PROCESSES);
        let run = supervise(sh("echo out; echo err >&2")).expect("the child should start");

        assert!(run.ok, "a clean exit should be reported as success");
        assert!(!run.stopped);
        assert!(run.stdout.contains("out"));
        // stderr matters: it is the only diagnostic when a real run fails to load.
        assert!(run.stderr.contains("err"));

        let failed = supervise(sh("exit 3")).expect("the child should start");
        assert!(!failed.ok);
        assert!(!failed.stopped, "a failure is not a shutdown");
    }

    /// The whole point of the registry: a generation in flight when the window closes has to
    /// die with it, or 7.17 GB of weights stays resident with nothing to show for it.
    #[test]
    #[cfg(unix)]
    fn closing_kills_a_run_in_flight() {
        let _guard = recover(&PROCESSES);
        let before = live_pids();
        let started = std::time::Instant::now();
        // A child that would far outlive the test, so only being killed can end it.
        let worker = std::thread::spawn(|| supervise(sh("sleep 30")));

        // Its own pid, not merely "something is registered": another test may be running the
        // real model at the same time.
        let mut mine = None;
        assert!(
            until(5, || {
                mine = live_pids().into_iter().find(|p| !before.contains(p));
                mine.is_some()
            }),
            "the child was never registered, so nothing could have killed it"
        );
        let mine = mine.expect("just checked");
        assert!(stop_live() >= 1, "the live run should have been found");

        let run = worker.join().expect("the supervising thread").unwrap();
        assert!(!run.ok, "a killed child must not report success");
        assert!(
            !live_pids().contains(&mine),
            "a finished run must not stay in the registry"
        );
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "took {:?} — the child was waited for, not killed",
            started.elapsed()
        );
        // Nothing of ours left running, and an empty sweep is harmless.
        stop_live();
    }

    /// Shutdown also has to close the door behind it: a worker that reaches the spawn a
    /// moment later must not map the weights again on the way out.
    #[test]
    #[cfg(unix)]
    fn a_closing_app_refuses_to_start_the_model() {
        let _guard = recover(&PROCESSES);
        CLOSING.store(true, Ordering::Relaxed);

        let run = supervise(sh("echo should not run"));
        // Restored before any assertion, so a failure here cannot leave the flag raised and
        // break every later test in the binary.
        CLOSING.store(false, Ordering::Relaxed);
        let run = run.expect("refusal is not an error");

        assert!(run.stopped, "the run should report that it never started");
        assert!(!run.ok);
        assert!(
            run.stdout.is_empty(),
            "nothing should have run: {:?}",
            run.stdout
        );
    }

    /// Whether the OS still knows about `pid`. A reaped child is gone from `ps` entirely; a
    /// zombie would still be listed, which is why [`stop_live`] waits on what it kills.
    #[cfg(unix)]
    fn alive(pid: u32) -> bool {
        Command::new("ps")
            .args(["-p", &pid.to_string()])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// The claim this whole section exists to make, checked against the real weights: quit
    /// mid-generation and the process holding 7.17 GB is gone, not orphaned.
    ///
    /// Ignored by default — it needs the download and the runtime.
    ///
    /// `cargo test -- --ignored live_model_is_killed --nocapture`
    #[test]
    #[cfg(unix)]
    #[ignore = "needs the downloaded checkpoint and a provisioned runtime"]
    fn live_model_is_killed_when_the_app_closes() {
        let _guard = recover(&PROCESSES);
        let before = live_pids();
        let started = std::time::Instant::now();
        // Long enough to chunk, so the scan is certain to still be generating.
        let text = "Contact Mario Rossi at mario@example.com about invoice 42. ".repeat(60);
        let worker = std::thread::spawn(move || detect_pii(&text));

        let mut pid = None;
        assert!(
            until(60, || {
                pid = live_pids().into_iter().find(|p| !before.contains(p));
                pid.is_some()
            }),
            "the model never started, so this test proves nothing"
        );
        let pid = pid.expect("just checked");

        shutdown();
        let err = worker
            .join()
            .expect("the scanning thread")
            .expect_err("a scan cannot finish once its model is gone");
        // Restored so the rest of the binary can still run the model.
        CLOSING.store(false, Ordering::Relaxed);

        eprintln!("  killed pid {pid} after {:?}: {err}", started.elapsed());
        assert!(
            err.contains("closing"),
            "the reason should say the app is closing, got {err:?}"
        );
        assert!(
            !alive(pid),
            "pid {pid} outlived the shutdown — the weights are still resident"
        );
        assert!(
            started.elapsed() < Duration::from_secs(120),
            "shutdown waited for the generation instead of killing it"
        );
    }

    /// End-to-end against the real checkpoint and the real runtime. Ignored by default — it
    /// needs one of the builds downloaded, and takes tens of seconds per call.
    ///
    /// `cargo test -- --ignored live_model --nocapture`
    ///
    /// Runs against whichever build is actually on disk rather than insisting on the default
    /// one: this is the test that proves the grammar, the template and the reply parser work
    /// against real weights, and it should not be un-runnable on a machine that has the other
    /// 3.8 GB sitting right there.
    #[test]
    #[ignore = "needs a downloaded checkpoint and a provisioned runtime"]
    fn live_model_ranks_and_finds_personal_data() {
        use crate::config::LocalModel;

        let downloaded = [LocalModel::Ternary, LocalModel::OneBit]
            .into_iter()
            .find(|m| models::is_cached(&m.checkpoint()));
        let Some(choice) = downloaded else {
            eprintln!("neither build is downloaded — nothing to run against");
            return;
        };
        // Every build that is on disk must resolve to a real file through the same path the
        // app uses. Selecting a build that pstore then cannot locate is the one way this
        // switch can fail silently, and it costs nothing to check both here.
        for m in [LocalModel::Ternary, LocalModel::OneBit] {
            let c = m.checkpoint();
            if models::is_cached(&c) {
                let path = hub_path(&c).unwrap_or_else(|e| panic!("{} cached but {e}", c.title));
                assert!(path.exists(), "{} resolved to {path:?}", c.title);
            }
        }

        let mut prefs = crate::config::prefs_snapshot();
        prefs.local_model = choice;
        crate::config::publish(&prefs);
        eprintln!("running against {}", choice.checkpoint().title);

        let cands = [
            candidate("claude", "haiku-4.5", Effort::Low),
            candidate("claude", "opus-5", Effort::High),
        ];

        let ranking =
            rank("fix a typo in the README", &cands, Vec::new()).expect("the model should rank");
        for c in &ranking.choices {
            eprintln!("  {} · fit {} · {}", c.model_display, c.fit, c.rationale);
        }
        assert_eq!(ranking.considered, 2);
        assert!(!ranking.choices.is_empty());
        // A typo fix must not come back wanting the frontier model at high effort. This is
        // the whole premise of routing, checked against the real weights rather than a
        // fixture.
        assert_eq!(
            ranking.best().map(|c| c.model_id),
            Some("haiku-4.5"),
            "a trivial prompt was routed to {:?}",
            ranking.best().map(|c| c.model_id)
        );

        let found = detect_pii("Contact Mario Rossi at mario@example.com about invoice 42.")
            .expect("the model should scan");
        for f in &found {
            eprintln!("  {} = {:?}", f.tag, f.text);
        }
        // Spans are located in the source, so whatever comes back must be real text.
        let src = "Contact Mario Rossi at mario@example.com about invoice 42.";
        for f in &found {
            assert_eq!(&src[f.start..f.end], f.text, "span does not match its text");
        }
        assert!(
            found.iter().any(|f| f.text.contains("mario@example.com")),
            "the address should have been found: {found:?}"
        );
    }

    /// Every repetition in the grammar has to be bounded. An unbounded rule is a place the
    /// sampler can legally sit forever, and it has: a `ws ::= [ \t\n]*` between two JSON
    /// tokens once cost a whole generation budget in spaces, and the call returned no answer
    /// at all. This is cheaper to assert than to rediscover.
    #[test]
    fn the_grammar_has_no_unbounded_repetition() {
        let g = rank_grammar(15, 5, 1400);
        for (n, line) in g.lines().enumerate() {
            assert!(
                !line.contains('*') && !line.contains('+'),
                "line {n} can repeat without bound: {line}"
            );
        }
        // And no whitespace rule at all: the JSON is emitted dense, because every space is a
        // token at ~41 ms.
        assert!(!g.contains("ws"), "{g}");
    }

    /// The model picks by index, and the grammar is what makes an impossible index
    /// impossible. A digit pattern would let it emit `29` against a fifteen-item list and
    /// leave `build_choices` to drop a pick the user then cannot account for.
    #[test]
    fn the_grammar_admits_exactly_the_real_indices() {
        let g = rank_grammar(3, 2, 0);
        let index = g
            .lines()
            .find(|l| l.starts_with("index ::="))
            .expect("an index rule");
        assert_eq!(index, "index ::= \"0\" | \"1\" | \"2\"");

        // Reasoning off means no think block to close, and the answer starts immediately.
        assert!(g.starts_with("root ::= answer"), "{g}");
        assert!(!g.contains(END_OF_THOUGHT), "{g}");

        // One candidate is still a valid list, and must not produce an empty alternation.
        assert!(rank_grammar(1, 1, 0).contains("index ::= \"0\""));
    }

    /// With a budget the reasoning block is *permitted* up to the cap and `</think>` is then
    /// *required* — that requirement is the whole mechanism. If the close became optional the
    /// model would think until the token budget ran out, which is the failure this replaced.
    #[test]
    fn a_reasoning_budget_is_a_cap_and_a_forced_close() {
        let g = rank_grammar(15, 5, 900);
        assert!(
            g.starts_with(&format!("root ::= thought \"{END_OF_THOUGHT}\" answer")),
            "{g}"
        );
        assert!(g.contains("thought ::= [^<]{0,900}"), "{g}");
    }

    /// The array has to come out exactly `want` long, the way `minItems`/`maxItems` used to
    /// guarantee. Off by one here is a shortlist that is quietly the wrong length.
    #[test]
    fn the_grammar_asks_for_exactly_the_shortlist_length() {
        assert!(rank_grammar(15, 5, 0).contains("choice (\",\" choice){4}"));
        assert!(rank_grammar(15, 1, 0).contains("choice (\",\" choice){0}"));
    }

    /// Sampling follows the task. Reasoning wants the checkpoint's published thinking-mode
    /// settings; extraction wants greedy decoding, and running it at 0.7 measurably lost a
    /// personal-data finding.
    #[test]
    fn only_the_reasoning_path_samples_at_temperature() {
        let schema = json!({"type": "object"});
        assert!(
            !Constrain::Schema(&schema).reasons(),
            "extraction is greedy"
        );

        assert!(Constrain::Grammar(rank_grammar(3, 2, 900)).reasons());
        assert!(
            !Constrain::Grammar(rank_grammar(3, 2, 0)).reasons(),
            "a zero budget means no reasoning, so no temperature either"
        );
    }

    /// The generation budget has to cover the reasoning block as well as the JSON. Undersized,
    /// the grammar is still waiting for `</think>` when the tokens run out and the reply
    /// contains no JSON whatsoever — a failure that reads like a broken model.
    #[test]
    fn the_token_budget_covers_the_reasoning_block() {
        let with = rank_output_tokens(5, 1400);
        let without = rank_output_tokens(5, 0);
        assert!(
            with >= without + (1400.0 / CHARS_PER_TOKEN) as usize,
            "{with} leaves no room for 1400 characters of reasoning"
        );
        // Reasoning off should not pay for a block it will not produce.
        assert!(without < 250, "{without} tokens for JSON alone");
    }

    /// Reasoning about a routing decision quotes JSON — braces and all — so the answer has to
    /// be taken from *after* the reasoning, never by scanning the whole reply for a `{`.
    #[test]
    fn the_reasoning_block_is_not_mistaken_for_the_answer() {
        let reply = "The schema wants {\"choices\":[...]} so I will emit\n\
                     </think>{\"choices\":[{\"index\":3,\"fit\":90,\"reason\":\"ok\"}]}";
        let v = parse_reply(reply).expect("the answer follows the reasoning");
        let choices = v.get("choices").and_then(Value::as_array).unwrap();
        assert_eq!(choices.len(), 1, "parsed the notes instead of the answer");
        assert_eq!(choices[0]["index"], 3);

        // A reply with no reasoning at all still parses.
        assert!(parse_reply("{\"choices\":[]}").is_ok());
    }

    /// The ranking prompt has to carry the facts the model is being asked to weigh —
    /// especially metering, which is the one judgement pstore used to hard-code.
    #[test]
    fn rank_prompt_states_the_tradeoffs() {
        let mut metered = candidate("claude", "fable", Effort::Max);
        metered.metered = true;
        let cands = [candidate("claude", "sonnet", Effort::Low), metered];

        let p = rank_prompt("fix a typo", &cands, 2);
        assert!(
            p.contains("fix a typo"),
            "the prompt itself must be in there"
        );
        assert!(p.contains("PAID-PER-TOKEN"), "metering must be flagged");
        assert!(p.contains("0: sonnet"), "options must be indexed from zero");
        assert!(p.contains("1: fable"));
        assert!(p.contains("2 best"));
    }

    /// Prompt evaluation is the dominant cost of a ranking call, and it scales with the
    /// candidate list. A machine with several agents installed offers ~30 combinations, so
    /// a few wasted words per line is a real number of seconds.
    #[test]
    fn the_candidate_list_stays_compact() {
        let cands: Vec<_> = (0..30)
            .map(|_| candidate("claude", "claude-sonnet-5", Effort::Medium))
            .collect();
        let list_bytes =
            rank_prompt("do a thing", &cands, 5).len() - rank_prompt("do a thing", &[], 5).len();
        let per_line = list_bytes / 30;
        assert!(
            per_line < 60,
            "{per_line} bytes per candidate line is too verbose for a 30-line list"
        );
    }
}
