//! The one place that runs the model.
//!
//! Every local inference in pstore — ranking the agents, judging how hard a prompt is, finding
//! personal data, compressing a document — is a call made from here, through a
//! [`Session`](super::session::Session).
//!
//! **One load per operation, and nothing between them.** A session maps the weights, answers
//! every call that one user action needs, and dies with it. Ranking is two calls, a shrink is one
//! per chunk, a personal-data scan is one per chunk; all of them load once. Between operations
//! there is no process, no port and no memory held, which is both the privacy property and what
//! makes the context window honest — it is computed from the prompts this operation will actually
//! send. See [`super::session`] for why that is the server binary rather than one-shot completion.
//!
//! **Every operation is tuned to what it is.** Two kinds of work happen here and they want
//! opposite settings, so [`Task`] carries them rather than a global default:
//!
//! | | Judgement — ranking | Extraction — difficulty, personal data, shrink, model recall |
//! | --- | --- | --- |
//! | sampling | `temp 0.7 · top-p 0.95 · top-k 20` | greedy |
//! | reasoning | a bounded `<think>` block | none |
//! | why | the checkpoint's published thinking-mode settings, and reasoning at temperature zero collapses into repetition — the same `reason` on all five picks | one right answer and nothing to deliberate; running the personal-data scan at 0.7 cost it a live finding, returning the name in `Contact Mario Rossi at mario@example.com` and dropping the address |
//!
//! **Output is grammar-constrained**, so the sampler cannot emit anything that fails to parse and
//! parsing is a `serde_json` call rather than a best-effort scrape. One trap, learned the hard
//! way: never give a JSON grammar an unbounded whitespace rule. `ws ::= [ \t\n]*` between two
//! tokens is a legal place to emit spaces forever, and the model did exactly that — hundreds of
//! tokens of blanks, then the generation limit, and no answer. Every rule here is bounded.
//!
//! **Nothing outlives the window.** Closing a window does not kill its children, so every session
//! is registered while it runs and killed by [`shutdown`], which each front end calls on its way
//! out.
//!
//! **What a call costs**, warm, on an M4-Pro-class laptop:
//!
//! | Phase | 1-bit | ternary |
//! | --- | --- | --- |
//! | mapping the weights, once per operation | ~1.5 s | ~2.5 s |
//! | prompt evaluation | ~95 tok/s | ~50 tok/s |
//! | generation | ~25 tok/s | ~13 tok/s |
//!
//! Generation costs ~4× more per token than prompt evaluation, which is why [`rank_grammar`]
//! permits no whitespace and caps every string; prompt evaluation is linear with no fixed floor,
//! which is why [`rank_prompt`] stays terse.

use std::path::PathBuf;
use std::process::Child;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use serde_json::{Value, json};

use super::session::Session;
use crate::agents::registry::{Effort, Tier};
use crate::models;
use crate::runtime;

/// Rough characters per token, for sizing the context window.
///
/// Deliberately pessimistic. Real Qwen-family tokenizers average nearer 3.7 on English prose and
/// better on code; 3.0 over-counts, which is the safe direction — see [`fit_context`].
const CHARS_PER_TOKEN: f32 = 3.0;

/// Smallest context worth asking for. Below this the savings are noise and the risk of clipping a
/// prompt is not.
const MIN_CONTEXT: usize = 512;

/// Context is requested in multiples of this, so nearly-identical operations reuse the same
/// allocation size rather than each landing on its own.
const CONTEXT_STEP: usize = 256;

/// Size the context window to the work actually being done.
///
/// The checkpoint natively supports 262 144 tokens. Running there would cost ~12 GB of KV cache
/// for prompts that are, in every one of pstore's uses, a few hundred tokens. So the window is
/// fitted per **operation**: at the sizes pstore asks for the cache is tens of megabytes and the
/// weights are essentially the entire footprint.
///
/// The estimate errs high on purpose. A prompt that does not fit is silently truncated, which
/// would mean PII spans pointing into text the model never saw — a wrong answer that looks like a
/// right one. Over-estimating costs a few megabytes; under-estimating costs correctness.
pub fn fit_context(prompt: &str, max_output_tokens: usize, ceiling: usize) -> usize {
    fit_context_for(&[(prompt.len(), max_output_tokens)], ceiling)
}

/// The window an operation needs: the largest of its calls, since one session serves them all.
///
/// Taking the maximum rather than the sum is the whole point of sizing per operation: a
/// four-chunk shrink needs room for one chunk at a time, not for the document.
pub fn fit_context_for(calls: &[(usize, usize)], ceiling: usize) -> usize {
    let needed = calls
        .iter()
        .map(|(chars, output)| {
            let prompt_tokens = (*chars as f32 / CHARS_PER_TOKEN).ceil() as usize;
            // 25% headroom over an already-pessimistic estimate, plus the chat template's own
            // wrapper tokens, which are not in the prompt.
            prompt_tokens + prompt_tokens / 4 + output + 256
        })
        .max()
        .unwrap_or(0);
    let stepped = needed.div_ceil(CONTEXT_STEP) * CONTEXT_STEP;
    stepped.clamp(MIN_CONTEXT, ceiling.max(MIN_CONTEXT))
}

/// What the runtime and the weights add up to: either a way to run the model, or the reason there
/// isn't one.
///
/// Returns the selected checkpoint alongside the paths, because everything downstream reports
/// progress against *its* row on the status board. Reading [`models::active`] once here and
/// passing it along is what stops a mid-operation preference change from marking one build's row
/// `Ready` on the strength of the other build's run.
pub(super) fn ready() -> Result<(PathBuf, PathBuf, models::Checkpoint), String> {
    let prefs = crate::config::prefs_snapshot();

    let rt = runtime::locate(prefs.llama_path.as_deref())
        .ok_or_else(|| runtime::missing_reason(prefs.llama_path.as_deref()))?;

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

/// How the sampler is to be constrained.
///
/// Two mechanisms because two jobs. A JSON Schema is the better thing to write and maintain, and
/// it is what the extraction paths want; but it constrains the very first sampled token, which
/// leaves no room for a reasoning block. Where reasoning earns its seconds — ranking — the
/// grammar is written out by hand instead.
#[derive(Debug, Clone)]
pub enum Constrain {
    /// A JSON Schema, compiled into the sampler. No reasoning is possible.
    Schema(Value),
    /// A GBNF grammar, which may allow a `<think>` block before the JSON.
    Grammar(String),
}

/// One call: what constrains it, how much it may write, and how it should sample.
///
/// Constructed by the operation rather than defaulted, because the two kinds of work here want
/// opposite settings and getting that wrong is invisible — a personal-data scan at temperature
/// 0.7 does not fail, it quietly misses an address. See the module header for the measurements.
#[derive(Debug, Clone)]
pub struct Task {
    /// How the sampler is constrained.
    pub constrain: Constrain,
    /// Upper bound on generated tokens. Must cover the reasoning block as well as the JSON, or
    /// the grammar will still be waiting for `</think>` when the budget runs out.
    pub max_output: usize,
    /// `0.0` is greedy.
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    /// Penalty applied to tokens already generated. `1.0` is off.
    ///
    /// Only meaningful where the model *composes* a list. A schema that permits ten steps
    /// is a shape the model can satisfy by writing one step and then nine copies of it, and
    /// that is exactly what a compressed checkpoint does when nothing discourages it.
    pub repeat_penalty: f32,
}

impl Task {
    /// A call that copies, classifies or extracts: greedy, no reasoning.
    ///
    /// Most of what pstore asks for. There is one right answer and nothing for temperature to
    /// diversify — only a copy to get exactly right.
    pub fn extraction(schema: Value, max_output: usize) -> Self {
        Task {
            constrain: Constrain::Schema(schema),
            max_output,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            repeat_penalty: 1.0,
        }
    }

    /// A call that *writes* something: schema-constrained, but sampled and penalised.
    ///
    /// Between [`Self::extraction`] and [`Self::judgement`], and needed because the two
    /// existing profiles each get half of this wrong. Extraction is greedy, which is right
    /// when the answer is mostly copied out of the input and catastrophic when it is not:
    /// asked for ten steps at temperature zero, the checkpoint writes step one and then
    /// nine verbatim copies of it. Judgement samples correctly but is built on a hand-
    /// written grammar so it can open a `<think>` block, and there is nothing here worth
    /// thinking about — the fields are the structure.
    ///
    /// So: the schema, the checkpoint's published thinking-mode sampling, and a repetition
    /// penalty, which is the part that actually stops the loop.
    pub fn composition(schema: Value, max_output: usize) -> Self {
        Task {
            constrain: Constrain::Schema(schema),
            max_output,
            temperature: 0.7,
            top_p: 0.95,
            top_k: 20,
            repeat_penalty: 1.15,
        }
    }

    /// A call that classifies against a hand-written grammar: greedy, no reasoning.
    ///
    /// [`Self::extraction`] is the same sampling and the better thing to write, and it is what
    /// this should be — except that a JSON Schema is compiled by the runtime into a grammar
    /// pstore does not control, and that grammar has a whitespace rule in it. The module header
    /// records what an unbounded whitespace rule does; the difficulty call hit it. Asked for
    /// three fields inside 80 tokens, the checkpoint spent them on runs of tabs between the
    /// fields and the reply came back truncated with no JSON in it at all — which fails the whole
    /// ranking, because the difficulty is the first call it makes.
    ///
    /// A grammar written here has no whitespace rule and **fixes the field order**, which the
    /// schema path also gets wrong: `serde_json` sorts a map's keys, so the properties reach the
    /// converter alphabetically and `because` was emitted before the two labels it exists to
    /// justify. Under greedy sampling each field conditions the next, so that is not a cosmetic
    /// difference — it is the model committing to a sentence and then picking labels to match it.
    pub fn classification(grammar: String, max_output: usize) -> Self {
        Task {
            constrain: Constrain::Grammar(grammar),
            max_output,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            repeat_penalty: 1.0,
        }
    }

    /// A call that weighs options: the checkpoint's published thinking-mode sampling.
    ///
    /// Those numbers are not a taste. They are the settings its thinking-mode benchmarks were run
    /// at, and reasoning at temperature zero collapses into repetition — the same `reason` string
    /// on all five picks, every time.
    pub fn judgement(grammar: String, max_output: usize) -> Self {
        Task {
            constrain: Constrain::Grammar(grammar),
            max_output,
            temperature: 0.7,
            top_p: 0.95,
            top_k: 20,
            repeat_penalty: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Process lifetime
// ---------------------------------------------------------------------------

/// Every session alive right now: its pid, the checkpoint it is holding, and its handle.
///
/// The checkpoint id is what makes "never two builds resident at once" enforceable. Without it a
/// switch could only kill everything or nothing, and there would be no way to answer the question
/// this registry exists to answer: *which* weights is that 4 GB?
static LIVE: Mutex<Vec<LiveSession>> = Mutex::new(Vec::new());

/// One live session, shared between the [`Session`] that owns it and this registry.
type LiveSession = (u32, &'static str, Arc<Mutex<Child>>);

/// Raised once the app is on its way out, so no further weights are mapped.
static CLOSING: AtomicBool = AtomicBool::new(false);

/// What to tell the user when a call is refused or cut short by the app closing.
pub(super) const CLOSING_REASON: &str = "the model was stopped because pstore is closing";

/// A poisoned lock means a thread panicked mid-update. Recovering the guard is better than
/// refusing to kill the process it was holding.
fn recover<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    match m.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// Whether the app is on its way out.
pub(super) fn stopping() -> bool {
    CLOSING.load(Ordering::Relaxed)
}

/// Refuse to map weights that should not be mapped.
///
/// Two races that both end with gigabytes resident that should not be: a worker between resolving
/// the weights and spawning must not map them as the window disappears — and it must not map the
/// *old* build after the user has switched, which is the one way two builds could end up resident
/// at once.
pub(super) fn refuse_if_stopping(checkpoint: &models::Checkpoint) -> Result<(), String> {
    if stopping() {
        return Err(CLOSING_REASON.into());
    }
    if checkpoint.id != models::active().id {
        return Err(
            "the model was stopped because the local model was switched — try again".into(),
        );
    }
    Ok(())
}

/// Record a session while it runs, so [`shutdown`] and a build switch can reach it.
pub(super) fn register(pid: u32, checkpoint: &'static str, child: Arc<Mutex<Child>>) {
    recover(&LIVE).push((pid, checkpoint, child));
}

/// Forget a session that has ended.
pub(super) fn deregister(pid: u32) {
    recover(&LIVE).retain(|(p, _, _)| *p != pid);
}

/// Kill and reap every live session the predicate selects, by checkpoint id.
///
/// Returns the checkpoints that were actually holding weights, so the caller can correct their
/// rows on the status board — a killed session leaves a checkpoint marked `Loading` or `Ready`
/// that is now neither.
///
/// The doomed entries are removed from [`LIVE`] under the lock and killed outside it, so a session
/// that ends by itself in the meantime cannot be waited on from two places, and a second caller
/// cannot pick up the same child. Killing an already-exited child and waiting an already-reaped
/// one both fail harmlessly, which is the state this function exists to reach.
fn stop_live_where(mut wanted: impl FnMut(&str) -> bool) -> Vec<&'static str> {
    let doomed: Vec<LiveSession> = {
        let mut live = recover(&LIVE);
        let (doomed, keep) = std::mem::take(&mut *live)
            .into_iter()
            .partition(|(_, id, _)| wanted(id));
        *live = keep;
        doomed
    };

    for (_, _, child) in &doomed {
        end(child);
    }
    doomed.into_iter().map(|(_, id, _)| id).collect()
}

/// End a child and reap it. Idempotent: both calls fail on a process that has already gone.
pub(super) fn end(child: &Arc<Mutex<Child>>) {
    let mut c = recover(child);
    let _ = c.kill();
    // Reaped here rather than left to the operating system: a zombie holds its slot, and the port
    // it was bound to stays taken until the process is really gone.
    let _ = c.wait();
}

/// Kill and reap every session running now. Returns how many there were.
fn stop_live() -> usize {
    stop_live_where(|_| true).len()
}

/// Stop anything still holding a build that is no longer the selected one.
///
/// **The invariant this exists for: at most one build's weights are resident at any moment.**
/// Nothing is resident *between* operations, so switching build is normally free. The exception is
/// switching *during* one: that session goes on holding its 3.8 or 7.17 GB until it finishes, and
/// the next operation would map the other build alongside it. On a machine chosen for the small
/// build because memory is tight, that is 11 GB at once, which is the situation the choice was
/// made to avoid.
///
/// So the old session is killed rather than waited out. It cannot produce a useful answer anyway:
/// its result would come from the build the user has just decided against.
///
/// Call it **after** publishing the new preference, so [`models::active`] already reflects the
/// switch — that ordering is also what makes [`refuse_if_stopping`] catch a worker that is
/// mid-flight. Cheap and safe when nothing is running, which is the common case.
pub fn unload_other_builds() -> usize {
    let keep = models::active().id;
    let stopped = stop_live_where(|id| id != keep);
    for id in &stopped {
        // Weights on disk, nothing running: true of the build we just left.
        models::set(id, models::Phase::Cached);
    }
    stopped.len()
}

/// Where the model's reasoning stops and its answer starts.
const END_OF_THOUGHT: &str = "</think>";

/// Pull the JSON object out of a reply.
///
/// The grammar guarantees the model's output parses, but the reasoning block is dropped first and
/// that ordering matters: reasoning about a routing decision quotes JSON, braces and all, so
/// scanning for the first `{` across the whole reply would parse the model's rough notes instead
/// of its answer. Everything up to the *last* `</think>` goes.
pub(super) fn parse_reply(reply: &str) -> Result<Value, String> {
    let reply = match reply.rfind(END_OF_THOUGHT) {
        Some(i) => &reply[i + END_OF_THOUGHT.len()..],
        None => reply,
    };
    let start = reply
        .find('{')
        .ok_or_else(|| format!("no JSON in the model's reply: {:?}", truncate(reply, 200)))?;
    let end = reply.rfind('}').ok_or_else(|| {
        format!(
            "truncated JSON in the model's reply: {:?}",
            truncate(reply, 200)
        )
    })?;
    if end < start {
        return Err(format!(
            "malformed JSON in the model's reply: {:?}",
            truncate(reply, 200)
        ));
    }
    serde_json::from_str(&reply[start..=end])
        .map_err(|e| format!("could not parse the model's reply: {e}"))
}

fn truncate(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

/// Open a session sized for one call, run it, and let the weights go.
///
/// The single-call shape. Operations that make several calls open the session themselves and keep
/// it for all of them — that is the whole point of [`Session`].
fn once(task: &Task, prompt: &str) -> Result<Value, String> {
    let ceiling = crate::config::prefs_snapshot().model_context_ceiling;
    let session = Session::open(fit_context(prompt, task.max_output, ceiling))?;
    session.run(task, prompt)
}

// ---------------------------------------------------------------------------
// Ranking
// ---------------------------------------------------------------------------

/// Longest `reason` the grammar will allow, in characters.
///
/// Every one of these is a token the model spends at ~41 ms, several times over, so the cap is
/// tight and the prompt asks for twelve words. A reason cut off mid-word reads as a bug, so
/// the prompt's instruction and this number have to stay in agreement.
const REASON_CHARS: usize = 90;

/// How many picks to ask each build for.
///
/// Not the same number for both, and that is the single most effective change made to ranking on
/// the small build. Five picks over a thirty-row grid is a task the 1-bit checkpoint does not do:
/// asked for five it emitted indices 1, 2, 3, 4, 5 with `fit: 85` on every one and a copy-pasted
/// reason — it was not discriminating, it was counting to five. Three picks over a list of
/// *models* is a question it can answer.
///
/// The ternary build keeps five, which it handles: enough to show the shape of the field — a
/// frontier model, a cheap one, a couple of efforts between — without asking for separations at
/// the tail that nothing could justify.
fn shortlist_for(checkpoint: &models::Checkpoint) -> usize {
    if checkpoint.id == models::LLM_1BIT.id {
        3
    } else {
        super::SHORTLIST
    }
}

/// How much capability a prompt actually demands, decided in its own call.
///
/// **Why this is a separate call.** Ranking asks for two judgements at once: how hard is this
/// work, and which of twenty options best matches that. The 1-bit build can do the first and
/// visibly cannot do both — asked for a shortlist on a three-file refactor it returned Haiku
/// first, "weak on multi-file reasoning, prompt requires multi-file", and Opus last, "best for
/// hard refactors, prompt is hard refactor". Its analysis was right in both rows and its
/// *ordering* ignored it. Splitting the question means the difficulty arrives in the ranking
/// prompt as a stated fact rather than as a second thing to work out while sorting.
///
/// It costs one extra call, which the module header's "one invocation per user action" rule
/// otherwise forbids. It is worth it here and cheap in the right way: the prompt is the document
/// with no candidate list, and the reply is one word, so it is a fraction of a ranking call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Demand {
    /// A typo, a rename, a comment, one small well-specified edit.
    Easy,
    /// Ordinary work: something to get right, but nothing to work out first.
    Moderate,
    /// The approach is not clear from the request: the cause has to be found, a design has to
    /// be chosen, or a guarantee has to be preserved that the change could silently break.
    Hard,
}

impl Demand {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Demand::Easy => "easy",
            Demand::Moderate => "moderate",
            Demand::Hard => "hard",
        }
    }

    /// Which weight class this difficulty calls for.
    fn tier(self) -> Tier {
        match self {
            Demand::Easy => Tier::Cheap,
            Demand::Moderate => Tier::Mid,
            Demand::Hard => Tier::Top,
        }
    }

    /// What ranking should do about it, spelled out for the model that has to act on it.
    ///
    /// **Tier only.** Effort used to be named here too ("at low effort", "at high effort"), which
    /// made it a property of the difficulty alone — and difficulty alone cannot tell a one-line
    /// deadlock fix from a repo-wide async conversion, so `xhigh` and `max` were never asked for
    /// by any prompt at all. Effort now comes from [`target_effort`], which reads breadth as well,
    /// and is stated as its own rule.
    fn instruction(self) -> &'static str {
        match self {
            Demand::Easy => {
                "Rank the LIGHTEST adequate model FIRST. A frontier model is the wrong answer \
                 here: it costs latency and quota this prompt does not need."
            }
            Demand::Moderate => {
                "Rank a mid-tier model FIRST. Neither the lightest model nor the frontier one is \
                 right for this."
            }
            Demand::Hard => {
                "Rank the most CAPABLE model FIRST. A light model is the wrong answer here even \
                 though it is cheaper and faster — put it last if at all."
            }
        }
    }
}

/// How much of the codebase a request reaches across, judged alongside [`Demand`].
///
/// **Why difficulty was not enough.** The two questions a routing decision needs answered are
/// *how good does the model have to be* and *how long should it think*, and difficulty only
/// answers the first. "Fix the race condition in the distributed lock renewal" and "convert the
/// whole IO layer to async" are both hard, and one is a paragraph of work while the other is a
/// week of it. With difficulty as the only input, [`Effort::XHigh`] and [`Effort::Max`] were
/// unreachable: no prompt could ask for them, because nothing in the judgement distinguished the
/// two cases. Breadth is what separates them — see [`target_effort`].
///
/// It is asked for in the same call and the same reply as the difficulty, so it costs a handful
/// of output tokens rather than a second model load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Breadth {
    /// One place: one function, or one file, however many lines that takes.
    OneEdit,
    /// A handful of files, or one feature end to end.
    FewFiles,
    /// More than a handful, a whole subsystem, or too many to know without reading the code.
    ManyFiles,
}

impl Breadth {
    /// The spelling the grammar admits and the prompt defines, and the one `judge_demand` reads
    /// back. All three come from here so they cannot drift apart.
    fn as_str(self) -> &'static str {
        match self {
            Breadth::OneEdit => "one_edit",
            Breadth::FewFiles => "few_files",
            Breadth::ManyFiles => "many_files",
        }
    }

    /// How to say it to a person, for the line every front end shows above the shortlist.
    pub(super) fn label(self) -> &'static str {
        match self {
            Breadth::OneEdit => "one edit",
            Breadth::FewFiles => "a few files",
            Breadth::ManyFiles => "many files",
        }
    }
}

/// The effort level the shortlist should aim for.
///
/// **The whole reason breadth is judged at all.** Difficulty picks the weight class; the two
/// together pick how long that model should think. Reading down a column shows what breadth buys
/// and across a row what difficulty buys:
///
/// | | one edit | a few files | many files |
/// | --- | --- | --- | --- |
/// | **easy** | low | low | medium |
/// | **moderate** | low | medium | high |
/// | **hard** | high | xhigh | max |
///
/// Two properties worth stating, because both were bugs before this table existed:
///
/// * **every level is reachable.** `xhigh` and `max` were dead code in the routing sense — the
///   registry offered them, the grammar admitted them, and no judgement ever called for one.
/// * **the easy row never leaves the fast end.** A verbose request for a one-word fix is still a
///   one-word fix, and paying `high` for it is the failure this whole path exists to prevent.
pub fn target_effort(demand: Demand, breadth: Breadth) -> Effort {
    use Breadth::*;
    use Demand::*;
    match (demand, breadth) {
        (Easy, OneEdit | FewFiles) => Effort::Low,
        (Easy, ManyFiles) => Effort::Medium,
        (Moderate, OneEdit) => Effort::Low,
        (Moderate, FewFiles) => Effort::Medium,
        (Moderate, ManyFiles) => Effort::High,
        (Hard, OneEdit) => Effort::High,
        (Hard, FewFiles) => Effort::XHigh,
        (Hard, ManyFiles) => Effort::Max,
    }
}

/// Judge how much capability `text` demands, and how far it reaches.
///
/// Greedy and without reasoning: these are classifications with one right answer each, and the
/// whole point of asking them separately from the ranking is that they are the cheap half. Both
/// come back in one reply — they are read off the same request, and a second call to ask "and how
/// many files?" would double the cheap half's cost to learn something the first pass already had
/// in front of it.
///
/// `because` is asked for because a model that has to justify a label picks it more carefully —
/// and because it is worth showing the user why their prompt was routed the way it was.
pub fn judge_demand(session: &Session, text: &str) -> Result<(Demand, Breadth, String), String> {
    let reply = session.run(&demand_task(), &demand_prompt(text))?;
    let demand = match reply.get("difficulty").and_then(Value::as_str) {
        Some("easy") => Demand::Easy,
        Some("hard") => Demand::Hard,
        // The schema admits only the three, and "moderate" is the answer that biases least if a
        // future one slips through.
        _ => Demand::Moderate,
    };
    let breadth = match reply.get("breadth").and_then(Value::as_str) {
        Some("one_edit") => Breadth::OneEdit,
        Some("many_files") => Breadth::ManyFiles,
        // Same reasoning as above: the middle is the answer that costs least when it is wrong.
        _ => Breadth::FewFiles,
    };
    let because = tidy(
        reply
            .get("because")
            .and_then(Value::as_str)
            .unwrap_or_default(),
        BECAUSE_CHARS,
    );
    Ok((demand, breadth, because))
}

/// Longest phrase the model may give for *why* it judged the prompt as it did.
///
/// Shown in every front end, so it is trimmed at a word boundary when the sampler stops here —
/// "a single typo in the README, which is a one" reads as pstore having mangled it.
const BECAUSE_CHARS: usize = 70;

/// Room for the difficulty reply: two labels and a short phrase, emitted dense.
///
/// Generous against what [`demand_grammar`] can actually produce — two words and
/// [`BECAUSE_CHARS`] characters is well under this — because the cost of overshooting is a
/// fraction of a second and the cost of undershooting is no ranking at all.
const DEMAND_OUTPUT: usize = 96;

/// The grammar for the difficulty reply.
///
/// Written out rather than compiled from a schema, for the two reasons in
/// [`Task::classification`]: no whitespace rule, and an order pstore chooses.
///
/// **The order is breadth, then difficulty, then the justification**, and it is the order the
/// model answers in. Breadth goes first because it is the countable one — "how many places
/// change" is read off the request, where "how hard is it" is a judgement about work nobody has
/// done yet — and under greedy sampling the first field is the one answered on its own merits.
/// With difficulty first the two collapsed onto each other: in a sweep of thirty prompts every
/// single request judged `hard` was then called `many_files`, including a one-line lock-renewal
/// fix, and `few_files` was chosen once in thirty.
fn demand_grammar() -> String {
    let alternatives = |values: &[&str]| {
        values
            .iter()
            .map(|v| format!("\"{v}\""))
            .collect::<Vec<_>>()
            .join(" | ")
    };
    let breadth = alternatives(&[
        Breadth::OneEdit.as_str(),
        Breadth::FewFiles.as_str(),
        Breadth::ManyFiles.as_str(),
    ]);
    let difficulty = alternatives(&[
        Demand::Easy.as_str(),
        Demand::Moderate.as_str(),
        Demand::Hard.as_str(),
    ]);
    format!(
        "root ::= \"{{\\\"breadth\\\":\\\"\" breadth \"\\\",\\\"difficulty\\\":\\\"\" difficulty \
         \"\\\",\\\"because\\\":\\\"\" because \"\\\"}}\"\n\
         breadth ::= {breadth}\n\
         difficulty ::= {difficulty}\n\
         because ::= [^\"\\\\\\n]{{1,{BECAUSE_CHARS}}}\n"
    )
}

fn demand_task() -> Task {
    Task::classification(demand_grammar(), DEMAND_OUTPUT)
}

/// Ask for both halves of the routing judgement in one reply.
///
/// Every line below the definitions is there because the definitions alone got something wrong in
/// a measured sweep, and each one names the mistake it exists to prevent:
///
/// * **"Count the places that change, not the places you would read."** Without it, breadth was
///   an echo of difficulty — anything `hard` came back `many_files`, because reading a subsystem
///   to find a one-line bug felt like reaching across it.
/// * **"needing to read code first does not make it hard."** The definition once said `hard`
///   covered "code that has to be understood before it can be changed", which sounds
///   discriminating and matches everything: memoizing one function came back `hard` because it
///   "implies understanding" the loader. Difficulty is about whether the *approach* is known, not
///   about whether the file has been opened.
/// * **length is not difficulty, brevity is not ease.** A polite four-sentence request to fix one
///   misspelled word is a one-word edit; "fix the race condition in the distributed lock renewal"
///   is eleven words and is the hardest thing in a codebase to get right.
fn demand_prompt(text: &str) -> String {
    format!(
        "Judge this coding request on two separate things.\n\
         \n\
         breadth — how many places in the code change?\n\
         - one_edit: one place — one function, or one file — however many lines that \
         takes.\n\
         - few_files: a handful of files, two to five, or one feature end to end.\n\
         - many_files: more than a handful, a whole subsystem, or too many to know without \
         reading the code.\n\
         Count the places that CHANGE, not the places you would read to find them.\n\
         \n\
         difficulty — how good does the model have to be?\n\
         - easy: the change itself is obvious. A typo, a rename, a comment, a constant.\n\
         - moderate: the approach is clear and it just has to be done correctly. Ordinary \
         feature work, a located bug, tests for known behaviour.\n\
         - hard: the approach is NOT clear from the request. The cause has to be found, a design \
         has to be chosen, or a guarantee has to be preserved that the change could silently \
         break.\n\
         Needing to read some code first does not make a request hard — almost every request \
         needs that.\n\
         \n\
         The two are independent. A change confined to one place can still be hard, and a change \
         spread over a hundred files can still be easy. Judge each on its own.\n\
         \n\
         Judge the WORK the request implies, not how long the request is.\n\
         \n\
         Then say, in under ten words, the thing about the request that decided it.\n\
         \n\
         <request>\n{text}\n</request>\n"
    )
}

/// One (agent, model) pair, with every effort level it can be asked for.
///
/// **The unit the model ranks is this, not a (model, effort) candidate**, and the difference is
/// what makes the small build work at all. The candidate grid holds one row per effort, so five
/// efforts of Opus look like five nearly-identical lines; ranking those means separating things
/// that differ in one word, which is where the checkpoint stops discriminating and starts
/// enumerating. Ranking models and asking for the effort as a *field* poses the same question
/// with a fifth of the rows and no near-duplicates in them.
struct Row {
    agent_id: &'static str,
    agent_display: &'static str,
    model_id: super::Name,
    model_display: super::Name,
    tier: crate::agents::registry::Tier,
    metered: bool,
    relative_price: f32,
    /// How fast this drains the subscription's allowance. See [`crate::agents::registry::ModelSpec::quota_weight`].
    quota_weight: f32,
    /// Efforts this (agent, model) can actually be asked for, ascending.
    efforts: Vec<crate::agents::registry::Effort>,
    /// Whether pstore can select the effort or is only predicting it.
    effort_selectable: bool,
    /// What pstore can tell the model about this model, or empty if it needs no telling.
    note: String,
    /// Where that note came from, so a placement can be accounted for in the UI.
    fact_source: Option<crate::knowledge::Source>,
}

/// Collapse the candidate grid into one row per (agent, model), preserving order.
fn rows(candidates: &[super::Candidate], brief: &crate::knowledge::Brief) -> Vec<Row> {
    let mut out: Vec<Row> = Vec::new();
    for c in candidates {
        match out
            .iter_mut()
            .find(|r| r.agent_id == c.agent_id && r.model_id == c.model_id)
        {
            Some(row) => {
                if !row.efforts.contains(&c.effort) {
                    row.efforts.push(c.effort);
                }
            }
            None => out.push(Row {
                agent_id: c.agent_id,
                agent_display: c.agent_display,
                model_id: c.model_id.clone(),
                model_display: c.model_display.clone(),
                tier: c.tier,
                metered: c.metered,
                relative_price: c.relative_price,
                quota_weight: c.quota_weight,
                efforts: vec![c.effort],
                effort_selectable: c.effort_selectable,
                note: brief.note(&c.model_id).unwrap_or_default().to_string(),
                fact_source: brief.source(&c.model_id),
            }),
        }
    }
    for row in &mut out {
        row.efforts.sort_unstable();
    }
    out
}

/// Put the options the difficulty calls for at the top of the list.
///
/// **A small model has positional bias, and this is what turns that from a hazard into a
/// tailwind.** Asked to shortlist a hard three-file refactor from a list running
/// light → frontier, the 1-bit build returned options 0, 1 and 2 in order — Haiku first — with
/// reasons that contradicted its own ranking ("may struggle with complex threading logic"). It
/// had the difficulty right and was reading the list rather than judging it.
///
/// Sorting by how well each row's tier matches the difficulty pstore has *already decided* means
/// that failure mode now produces a defensible answer instead of an inverted one, and a model
/// that really is ranking is unaffected: it picks by index either way, and the indices are the
/// same options.
///
/// The sort is stable, so rows of equal tier keep the order the registry gave them and the result
/// does not depend on how a `HashMap` felt that day.
///
/// **Ties break downwards.** A `moderate` prompt puts light and frontier the same distance from
/// mid, and something has to separate them or the answer depends on which agent the user happens
/// to have installed first. The lighter one goes above, because the two mistakes are not
/// symmetrical: routing moderate work to a light model costs a retry, and routing it to a
/// frontier model at 25× quota burn costs an allowance the developer needs later in the week.
fn order_for(rows: &mut [Row], demand: Demand) {
    let rank_of = |t: Tier| match t {
        Tier::Cheap => 0i32,
        Tier::Mid => 1,
        Tier::Top => 2,
    };
    let wanted = rank_of(demand.tier());
    rows.sort_by_key(|r| ((rank_of(r.tier) - wanted).abs(), rank_of(r.tier)));
}

/// Rank `candidates` against `text`.
///
/// The model chooses **by index** into the list it was given, never by naming an agent or a
/// model string. An index either maps onto a real launch configuration or is rejected; a
/// name could be plausible and wrong, and pstore would then try to launch it.
///
/// Expect **seconds**, and where they go is worth knowing before optimising anything here.
/// Measured on the 1-bit build over a nine-model field: mapping the weights 0.7 s, the difficulty
/// call 7 s (368 prompt tokens, 34 generated), and the ranking call 40 s — of which 9.7 s is
/// evaluating its 729-token prompt and **30.7 s is generating the reply**. Generation dominates,
/// so the reasoning block is the only lever that moves the total; it is off by default now, which
/// takes a ranking from ~36 s to ~17 s. See [`crate::config::Prefs::model_reasoning_budget`].
///
/// This is the call the whole "one invocation per user action" rule exists to ration — and the
/// reason a degenerate answer is retried **once** and no more.
pub fn rank(
    text: &str,
    candidates: &[super::Candidate],
    excluded: Vec<(&'static str, String)>,
    brief: &crate::knowledge::Brief,
) -> Result<super::Ranking, String> {
    let started = std::time::Instant::now();
    let mut rows = rows(candidates, brief);
    let want = shortlist_for(&models::active()).min(rows.len());
    let prefs = crate::config::prefs_snapshot();
    let budget = prefs.model_reasoning_budget;
    let max_output = rank_output_tokens(want, budget);

    // The window covers both calls, because one session serves both. Sized against the *retry*
    // wording of the ranking prompt, which is the longest this operation can send — a window that
    // fits only the first attempt would silently truncate the second. The difficulty and breadth
    // it is sized with are placeholders: they change a handful of words, never the length that
    // matters, and the real ones are not known until the first call has run.
    let widest = rank_prompt(text, &rows, want, Demand::Moderate, Breadth::FewFiles, true);
    let ctx = fit_context_for(
        &[
            (demand_prompt(text).len(), DEMAND_OUTPUT),
            (widest.len(), max_output),
        ],
        prefs.model_context_ceiling,
    );

    let session = Session::open(ctx)?;

    // The cheap judgement first, so the expensive one has one thing to do rather than two. See
    // [`Demand`] for the live failure that motivates the split.
    let (demand, breadth, because) = judge_demand(&session, text)?;
    // ...and then the options are presented in the order that judgement implies. See
    // [`order_for`]: on a small model this is the difference between a ranking and the first
    // three rows of the list.
    order_for(&mut rows, demand);

    // The grammar is built here rather than above because the effort it anchors the top pick to
    // is not known until the judgement has been made. See [`rank_grammar`].
    let effort = target_effort(demand, breadth);
    let task = Task::judgement(rank_grammar(&rows, want, effort, budget), max_output);

    // Two attempts at most. A ranking call is tens of seconds, so this is not a retry loop that
    // can be widened later without someone noticing — and a second degenerate answer is reported
    // as such rather than retried into a third.
    let mut degenerate = None;
    let mut choices = Vec::new();
    for attempt in 0..2 {
        let prompt = rank_prompt(text, &rows, want, demand, breadth, attempt > 0);
        let reply = session.run(&task, &prompt)?;
        choices = build_choices(&reply, &rows, candidates)?;

        degenerate = degeneracy(&choices, rows.len());
        if degenerate.is_none() {
            break;
        }
    }
    normalise_latency(&mut choices);

    Ok(super::Ranking {
        choices,
        considered: candidates.len(),
        excluded,
        judged: Some(super::Judgement {
            demand: demand.as_str(),
            breadth: breadth.label(),
            effort,
            because,
        }),
        described: brief
            .known
            .iter()
            .filter(|k| k.source != crate::knowledge::Source::Checkpoint)
            .count(),
        degenerate,
        elapsed: started.elapsed(),
    })
}

/// Whether the model produced a ranking or merely a list.
///
/// This exists because the failure it detects is **invisible**: five rows, five scores, five
/// reasons, every field populated, and the order is whatever the options happened to be in. A
/// user cannot tell that from a real answer, and acting on it means running the wrong model. So
/// it is named, and [`super::Ranking::degenerate`] carries the reason to the UI.
///
/// Two signatures, both taken from live runs on the 1-bit build:
///
/// * **the same reason on every pick** — the checkpoint wrote one sentence and repeated it, which
///   is what it does when it has not actually compared the options;
/// * **consecutive indices** over a list long enough for that to be a coincidence worth
///   disbelieving — `1, 2, 3, 4` out of thirty is enumeration, not judgement.
///
/// Identical `fit` values were the third signature and are now unrepresentable: the grammar
/// gives each position its own band (see [`fit_band`]). The check stays because a grammar can be
/// changed by someone who does not know that is why it was written that way.
fn degeneracy(choices: &[super::Choice], available: usize) -> Option<String> {
    if choices.len() < 2 {
        return (available > 2).then(|| {
            format!(
                "the local model returned {} pick(s) out of {available} options — it did not \
                 rank the field",
                choices.len()
            )
        });
    }

    let first = choices[0].rationale.trim().to_ascii_lowercase();
    if !first.is_empty()
        && choices
            .iter()
            .all(|c| c.rationale.trim().to_ascii_lowercase() == first)
    {
        return Some(
            "the local model gave every pick the same reason, so it did not compare them"
                .to_string(),
        );
    }

    if choices.iter().all(|c| c.fit == choices[0].fit) {
        return Some(
            "the local model scored every pick identically, so it did not separate them"
                .to_string(),
        );
    }

    // There was a third signature — picks at consecutive indices — and it has been removed
    // rather than tightened. Now that [`order_for`] presents the options best-first for the
    // difficulty, "the first three in order" is what a *correct* answer often looks like, so the
    // heuristic would fire on exactly the results it was meant to protect.
    None
}

/// Generation budget for a ranking call: the reasoning block plus the JSON.
///
/// Undersizing this is not a slow answer but no answer — the grammar is still waiting for
/// `</think>` when the tokens run out, and the reply has no JSON in it at all. So the
/// reasoning allowance is converted at the pessimistic [`CHARS_PER_TOKEN`] and then given
/// room to spare.
fn rank_output_tokens(want: usize, reasoning_budget: usize) -> usize {
    // ~48 tokens covers one `{"index":12,"effort":"medium","fit":95,"reason":"…"}` with a
    // full-length reason.
    let json = 16 + want * 52;
    let thought = if reasoning_budget == 0 {
        0
    } else {
        (reasoning_budget as f32 / CHARS_PER_TOKEN).ceil() as usize + 16
    };
    json + thought
}

/// The two-digit range a pick at `position` must score in.
///
/// **Descending bands, enforced by the sampler.** The model used to be asked for five numbers
/// between 0 and 100 with the instruction that they must differ, and on the small build it
/// returned `85, 85, 85, 85, 85`; on the ternary build it once returned `0` and `1` for a
/// correctly-ordered shortlist, which a sort on `fit` then inverted. Both are instructions the
/// grammar can simply make unrepresentable, so it does: the first pick scores 85–99, the second
/// 70–84, and so on down. Within its band the model still has fifteen values to say *how much*
/// better this pick is than the next, which is what [`super::Ranking::fastest_within`] reads.
fn fit_band(position: usize) -> (u32, u32) {
    /// Width of each band. Five bands of 15 reach from 99 down to 25, which leaves the bottom
    /// of the scale for nothing — a pick that is in the shortlist is not a bad option.
    const WIDTH: u32 = 15;
    let top = 99u32.saturating_sub(position as u32 * WIDTH);
    (top.saturating_sub(WIDTH - 1).max(1), top.max(1))
}

/// A GBNF alternation matching exactly the two-digit integers in `lo..=hi`.
///
/// Written out rather than left to a digit pattern with a range check afterwards, for the same
/// reason indices are: what the grammar forbids cannot come back and be dropped later.
fn digits_in_range(lo: u32, hi: u32) -> String {
    use std::fmt::Write;
    let mut parts: Vec<String> = Vec::new();
    for tens in (lo / 10)..=(hi / 10) {
        let low_unit = lo.max(tens * 10) % 10;
        let high_unit = hi.min(tens * 10 + 9) % 10;
        let mut part = String::new();
        let _ = write!(part, "\"{tens}\" ");
        if low_unit == high_unit {
            let _ = write!(part, "\"{low_unit}\"");
        } else {
            let _ = write!(part, "[{low_unit}-{high_unit}]");
        }
        parts.push(part);
    }
    parts.join(" | ")
}

/// The grammar for a ranking reply: an optional bounded reasoning block, then the JSON.
///
/// Written by hand rather than compiled from a JSON Schema, because a schema constrains the
/// first sampled token and the whole point here is to leave room for `<think>` first. Four
/// things are deliberate:
///
/// * **No whitespace rule at all.** The JSON is emitted dense. An unbounded `ws` rule between
///   tokens is somewhere the sampler can legally sit forever, and it does.
/// * **`index` is an alternation of the literal indices**, not a digit pattern with a range
///   check afterwards. The list is short and this way an out-of-range pick is unrepresentable
///   rather than merely rejected — [`build_choices`] still drops what does not map, but it
///   should never have to.
/// * **Each position has its own `fit` band**, so the scores descend by construction. See
///   [`fit_band`] for the two live failures that instruction could not prevent on its own.
/// * **The top pick's `effort` is bounded to the target and its neighbours**, for the same
///   reason. The prompt asks for the effort [`target_effort`] computed; asking was not enough —
///   over a sweep of eighteen prompts the checkpoint answered `low` on ten of them, including
///   every prompt it had itself judged `moderate`, because `low` is the first alternative in the
///   rule and nothing forbade it. One step either side leaves it room to disagree by a level, which is the
///   most disagreement a self-assessment at this size can support. Later positions keep the full
///   alternation: they are alternatives, and an alternative at the same effort as the pick above
///   is not one.
/// * **Every repetition is bounded.** `{0,n}` throughout, so a run cannot become unbounded by
///   any path through the grammar.
fn rank_grammar(rows: &[Row], want: usize, target: Effort, reasoning_budget: usize) -> String {
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
    for i in 0..rows.len().max(1) {
        if i > 0 {
            index.push_str(" | ");
        }
        let _ = write!(index, "\"{i}\"");
    }

    // Every effort any row offers. Which row supports which cannot be expressed here — the
    // grammar has no way to make one field depend on another — so an effort the chosen model
    // cannot be asked for is snapped to its nearest neighbour in `build_choices`.
    let mut efforts: Vec<crate::agents::registry::Effort> = Vec::new();
    for row in rows {
        for e in &row.efforts {
            if !efforts.contains(e) {
                efforts.push(*e);
            }
        }
    }
    efforts.sort_unstable();
    let alternation = |set: &[Effort]| {
        set.iter()
            .map(|e| format!("\"\\\"{e}\\\"\""))
            .collect::<Vec<_>>()
            .join(" | ")
    };
    let effort_rule = alternation(&efforts);

    // The top pick's band: the target and one step either side, intersected with what the field
    // actually offers. Intersected rather than assumed, because a machine with only Gemini
    // installed offers one effort in total, and a rule matching nothing at all would make the
    // whole reply unrepresentable — the sampler cannot answer, so there is no ranking rather than
    // an unanchored one.
    let near_target: Vec<Effort> = efforts
        .iter()
        .copied()
        .filter(|e| (step_of(*e) - step_of(target)).abs() <= 1)
        .collect();
    let effort0_rule = if near_target.is_empty() {
        effort_rule.clone()
    } else {
        alternation(&near_target)
    };

    // Positional choice rules, so each pick carries its own descending band.
    let mut answer = String::from("answer ::= \"{\\\"choices\\\":[\"");
    let mut choices = String::new();
    for position in 0..want.max(1) {
        if position > 0 {
            let _ = write!(answer, " \",\"");
        }
        let _ = write!(answer, " choice{position}");
        let (lo, hi) = fit_band(position);
        // Only the top pick is anchored. It is the one pstore would actually launch, and the
        // ones below it are there to show the shape of the field.
        let effort_ref = if position == 0 { "effort0" } else { "effort" };
        let _ = write!(
            choices,
            "choice{position} ::= \"{{\\\"index\\\":\" index \",\\\"effort\\\":\" {effort_ref} \
             \",\\\"fit\\\":\" fit{position} \",\\\"reason\\\":\" reason \"}}\"\n\
             fit{position} ::= {}\n",
            digits_in_range(lo, hi)
        );
    }
    let _ = write!(answer, " \"]}}\"");

    format!(
        "{root}\n\
         {answer}\n\
         {choices}\
         index ::= {index}\n\
         effort ::= {effort_rule}\n\
         effort0 ::= {effort0_rule}\n\
         reason ::= \"\\\"\" [^\"\\\\\\n]{{0,{REASON_CHARS}}} \"\\\"\"\n"
    )
}

/// Where an effort sits on the ladder, so "one step either side" is arithmetic rather than a
/// hand-written table that would go stale the moment a level is added.
fn step_of(effort: Effort) -> i32 {
    Effort::ALL.iter().position(|e| *e == effort).unwrap_or(0) as i32
}

/// Build the ranking prompt.
///
/// Three things earn their tokens here, and nothing else is allowed to:
///
/// * **what each model is** — the notes from [`crate::knowledge`]. Without them the checkpoint is
///   ranking names: its training ended before Opus 5 and Gemini 3 existed, and a model asked to
///   judge a name it does not know does not decline, it invents. This is the single largest
///   quality change to ranking, and it is also why a model nothing can describe is not in this
///   list at all;
/// * **the tier and the available efforts** — the two words that changed a routing decision in
///   testing;
/// * **`PAID-PER-TOKEN`** — the exception. "Included in subscription" was dropped because it is
///   the default, and saying so thirty times cost more than the one line that flags the
///   exception.
fn rank_prompt(
    text: &str,
    rows: &[Row],
    want: usize,
    demand: Demand,
    breadth: Breadth,
    retry: bool,
) -> String {
    use std::fmt::Write;

    let mut list = String::new();
    for (i, r) in rows.iter().enumerate() {
        let efforts = r
            .efforts
            .iter()
            .map(|e| e.as_str())
            .collect::<Vec<_>>()
            .join("|");
        let _ = write!(
            list,
            "\n{i}: {} via {} [{}, effort {efforts}{}]{}",
            r.model_display,
            r.agent_display,
            r.tier,
            if r.effort_selectable { "" } else { "?" },
            if r.metered { " PAID-PER-TOKEN" } else { "" },
        );
        // Quota burn, stated only where it is decision-relevant. Every model in this list costs
        // the same nothing extra on a subscription, so price says nothing useful here — but they
        // drain the plan's limits at very different rates, and a developer who spends a weekly cap
        // on work a light model would have done is blocked all the same. Written as a plain
        // multiple rather than a score so the model weighs it against fit instead of summing it.
        //
        // Below 2x it is left off: a marker on nearly every row stops carrying information, and
        // the difference is not worth trading any fit for.
        if r.quota_weight >= 2.0 {
            let _ = write!(list, " burns {:.0}x quota", r.quota_weight);
        }
        if !r.note.is_empty() {
            let _ = write!(list, "\n   {}", r.note);
        }
    }

    // Only on the second attempt, and only naming the thing that went wrong. A prompt that
    // always carried this would spend tokens teaching the model to avoid a mistake it was not
    // making, and on a small checkpoint that is its own kind of noise.
    //
    // **Appended, not inserted**, and that placement is worth the awkwardness of it coming after
    // the request. `cache_prompt` reuses an exact prefix, so a correction spliced into the middle
    // of the prompt invalidates everything after it and the retry re-evaluates all ~730 tokens —
    // ten seconds, to say one sentence the model has already read the context for. Appended, the
    // retry shares the whole prompt with the attempt it is correcting and pays only for the
    // sentence.
    let insist = if retry {
        "\nThe previous answer repeated itself. Each pick must be a DIFFERENT option, with a \
         different reason naming what distinguishes it.\n"
    } else {
        ""
    };

    format!(
        "Pick the {want} best options for the prompt below, best first.\n\
         \n\
         This prompt has already been judged **{difficulty}**, reaching across **{reach}**.\n\
         {act}\n\
         \n\
         Rules:\n\
         - Ask for **{effort}** effort, or the nearest an option offers. That is what this \
         difficulty and this much code together need — do not spend more, and do not spend less.\n\
         - PAID-PER-TOKEN costs extra money; rank one high only if clearly better than \
         every other option here.\n\
         - `?` after the effort means it cannot be set, only predicted.\n\
         - `fit` is 0-100 for THIS prompt. Score the best pick highest.\n\
         - `reason`: under 10 words, about THIS option, and it must AGREE with where you put \
         it. Never rank an option first while saying it is risky or insufficient.\n\
         - A different reason for each: say what distinguishes it from the pick above.\n\
         - Pick each option at most once.\n\
         \n\
         Options:{list}\n\
         \n\
         Prompt to route:\n\
         <prompt>\n{text}\n</prompt>\n\
         {insist}",
        difficulty = demand.as_str(),
        reach = breadth.label(),
        act = demand.instruction(),
        effort = target_effort(demand, breadth).as_str(),
    )
}

/// The effort to use for `row`, given what the model asked for.
///
/// The grammar offers the union of every row's efforts because it cannot make one field depend on
/// another, so a model can legally ask Gemini for `max` when Gemini's effort is not settable at
/// all. Snapped to the nearest level the row actually offers rather than rejected: the pick — this
/// model, for this prompt — is the judgement being asked for, and throwing it away over a detail
/// pstore can resolve itself would cost the user their answer.
fn snap_effort(
    row: &Row,
    asked: crate::agents::registry::Effort,
) -> crate::agents::registry::Effort {
    use crate::agents::registry::Effort;

    if row.efforts.contains(&asked) {
        return asked;
    }
    let target = step_of(asked);
    row.efforts
        .iter()
        .copied()
        .min_by_key(|e| (step_of(*e) - target).abs())
        .unwrap_or(Effort::High)
}

/// Clean up a reason, and mark it when the grammar cut it short.
///
/// The prompt asks for under ten words and [`REASON_CHARS`] is the sampler's hard stop; when the
/// model writes past it the reply ends mid-word, and "excels at repo-scale edits and complex
/// refactoring logi" reads as a bug in pstore rather than as a model being verbose. So a reason
/// that came back at exactly the cap — which is what truncation looks like — is cut back to its
/// last whole word and ellipsised, which reads as what it is.
fn tidy_reason(raw: &str) -> String {
    tidy(raw, REASON_CHARS)
}

/// Clean up a sampled string, and mark it when it stopped at `cap` rather than at its own end.
///
/// Shared by the two short strings the model writes for a person to read — a choice's reason and
/// the phrase that decided the difficulty — because both are capped and both look like a bug in
/// pstore when the cap lands mid-word.
fn tidy(raw: &str, cap: usize) -> String {
    // Measured before trimming: the cap applies to what the sampler emitted, and a string that
    // happened to stop on a space would otherwise slip under the test and keep its dangling word.
    let truncated = raw.chars().count() >= cap;
    let trimmed = raw.trim();
    if !truncated {
        return trimmed.to_string();
    }
    match trimmed.rsplit_once(' ') {
        Some((head, _)) => format!("{}…", head.trim_end_matches([',', ';', ':'])),
        // One long word at the cap: nothing to cut back to, so say it was cut.
        None => format!("{trimmed}…"),
    }
}

/// Map the model's indices back onto real candidates.
///
/// Duplicates are dropped rather than repeated: the grammar cannot express "distinct", so a
/// model that picks the same option twice would otherwise produce a shortlist that looks
/// like a bug in pstore.
fn build_choices(
    reply: &Value,
    rows: &[Row],
    candidates: &[super::Candidate],
) -> Result<Vec<super::Choice>, String> {
    use crate::agents::registry::Effort;

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
        let Some(row) = rows.get(idx) else {
            // The grammar bounds this, so reaching here means the grammar and the list
            // disagree — skip rather than launching something arbitrary.
            continue;
        };
        if seen.contains(&idx) {
            continue;
        }
        seen.push(idx);

        let asked = item
            .get("effort")
            .and_then(Value::as_str)
            .and_then(|s| Effort::ALL.iter().copied().find(|e| e.as_str() == s))
            // No effort field at all is not something the grammar allows; if it happens, the
            // middle of what this row offers is a better answer than refusing the pick.
            .unwrap_or(Effort::High);
        let effort = snap_effort(row, asked);

        // The grid is what pstore can actually launch, so the (row, effort) pair is confirmed
        // against it rather than assumed. A pair missing from the grid means the row was built
        // from candidates that no longer exist, which is a bug, not a launch to attempt.
        if !candidates
            .iter()
            .any(|c| c.agent_id == row.agent_id && c.model_id == row.model_id && c.effort == effort)
        {
            continue;
        }

        out.push(super::Choice {
            agent_id: row.agent_id,
            agent_display: row.agent_display,
            model_id: row.model_id.clone(),
            model_display: row.model_display.clone(),
            tier: row.tier,
            effort,
            effort_selectable: row.effort_selectable,
            metered: row.metered,
            relative_latency: effort.latency_factor(),
            relative_price: row.relative_price,
            quota_weight: row.quota_weight,
            note: row.note.clone(),
            fact_source: row.fact_source,
            fit: item.get("fit").and_then(Value::as_f64).unwrap_or(0.0) as f32,
            rationale: tidy_reason(
                item.get("reason")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
            ),
            row_index: idx,
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
    // what it was asked for and what it gets right; `fit` is a self-assessment, useful for the
    // table and the hint tolerance, and not to be trusted as a sort key. The grammar's
    // descending bands now keep the two consistent, which is a reason to leave this alone
    // rather than a reason to revisit it.
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
// Which models the checkpoint actually knows
// ---------------------------------------------------------------------------

/// Ask the checkpoint which of `models` it can describe from its own knowledge.
///
/// Returns indices into `models`. This is the second step of [`crate::knowledge::resolve`] and it
/// only runs for names pstore's own table does not cover, so on a stock installation it never
/// runs at all.
///
/// **Greedy, and no reasoning.** "Do you know this name?" is recall, not judgement: there is
/// nothing to deliberate and nothing for temperature to diversify. Reasoning here would cost a
/// second ranking call's worth of seconds to answer a question the first sampled token settles.
///
/// The prompt pushes hard towards omission, because the failure that matters is asymmetric. A
/// model wrongly left out is ranked from a web lookup or reported as unknown — visible, and
/// recoverable. A model wrongly claimed as known is ranked on an invention, which is exactly
/// what this whole path exists to stop.
pub fn known_models(models: &[String]) -> Result<Vec<usize>, String> {
    if models.is_empty() {
        return Ok(Vec::new());
    }
    let schema = json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["known"],
        "properties": {
            "known": {
                "type": "array",
                "maxItems": models.len(),
                "items": {"type": "integer", "minimum": 0, "maximum": models.len() - 1}
            }
        }
    });

    let list = models
        .iter()
        .enumerate()
        .map(|(i, m)| format!("\n{i}: {m}"))
        .collect::<String>();
    let prompt = format!(
        "Below are names of AI language models.\n\
         \n\
         Return the indices of ONLY the ones you genuinely know — where you could state which \
         company makes it and what it is good at.\n\
         \n\
         If a name is unfamiliar, or you are guessing from the way it is spelled, LEAVE IT OUT. \
         An empty list is the right answer when you know none of them. Do not guess.\n\
         \n\
         Models:{list}\n"
    );

    // 8 tokens per index, plus the wrapper. The list is short by construction.
    // Its own session, and the only operation that gets one for a single call. It runs at all
    // only when [`FACTS`](crate::knowledge::FACTS) does not cover the field, which on a stock
    // installation is never — so the common ranking path still loads the weights exactly once.
    let task = Task::extraction(schema, 32 + models.len() * 8);
    let reply = once(&task, &prompt)?;
    Ok(reply
        .get("known")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_u64)
                .map(|i| i as usize)
                .filter(|i| *i < models.len())
                .collect()
        })
        .unwrap_or_default())
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
    let task = Task::extraction(
        json!({
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
        }),
        PII_OUTPUT_TOKENS,
    );

    let chunks = crate::pii::segments(text, crate::pii::CHUNK_CHARS);
    if chunks.is_empty() {
        return Ok(Vec::new());
    }
    // One window, sized for the largest chunk — every chunk goes through the same session, so the
    // cache has to hold the biggest of them and no more.
    let widest = chunks
        .iter()
        .map(|(_, c)| pii_prompt(c).len())
        .max()
        .unwrap_or(0);
    let ctx = fit_context_for(
        &[(widest, PII_OUTPUT_TOKENS)],
        crate::config::prefs_snapshot().model_context_ceiling,
    );
    let session = Session::open(ctx)?;

    let mut out = Vec::new();
    for (offset, chunk) in chunks {
        let reply = session.run(&task, &pii_prompt(chunk))?;
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
// Shrinking
// ---------------------------------------------------------------------------

/// The largest chunk a shrink pass may compress in one call, in characters.
///
/// A shrink is the one call whose *output* is the size of its input: the rewrite is shorter
/// than the original, but only by a fraction, so the window has to hold the instruction, the
/// chunk, and a second copy of the chunk. Solving [`fit_context`]'s arithmetic for the chunk
/// gives the bound below — under the default 8 192-token ceiling that is ~8 900 characters,
/// which the cap then brings down.
///
/// The cap is not about memory. Instruction-following on a 2-bit checkpoint degrades over a
/// long passage — the register slips back to prose halfway down — and a chunk that fits the
/// window can still be more than the model will rewrite evenly.
///
/// The answer is four fifths of what fits, not all of it, because [`crate::shrink::chunks`]
/// may overrun by a quarter to keep a fenced code block whole. Truncation here is silent —
/// llama.cpp drops the tail of a prompt that does not fit — so the room has to be left
/// before it is needed.
pub fn shrink_chunk_chars() -> usize {
    /// Below this, a chunk is too small to compress usefully.
    const MIN: usize = 400;
    /// Above this, the rewrite gets uneven regardless of what fits.
    const MAX: usize = 6000;

    let ceiling = crate::config::prefs_snapshot().model_context_ceiling;
    let instruction = crate::shrink::INSTRUCTION.len();
    // fit_context: (instruction + chunk)/CHARS_PER_TOKEN * 1.25 + output + 256 <= ceiling,
    // with output bounded by `shrink_output_tokens` at the chunk's own token count + 25%.
    let budget = (CHARS_PER_TOKEN / 1.25) * ceiling.saturating_sub(320) as f32;
    let fits = (budget as usize).saturating_sub(instruction) / 2;
    (fits * 4 / 5).clamp(MIN, MAX)
}

/// Room for the rewrite: the chunk's own length, plus headroom.
///
/// Bounded above the input on purpose. A rewrite that saves nothing is a legitimate answer —
/// an already-terse prompt has nothing to give — and clipping the reply at the token cap
/// would not produce a shorter prompt, it would produce invalid JSON and an error.
fn shrink_output_tokens_for(chars: usize) -> usize {
    let tokens = (chars as f32 / CHARS_PER_TOKEN).ceil() as usize;
    (tokens + tokens / 4 + 64).clamp(128, 4096)
}

/// A shrink in progress: one load of the weights, one call per chunk.
///
/// Held by [`crate::shrink::run`] for the whole pass rather than opened per chunk. A long document
/// is four or five calls, and mapping the weights for each of them would be seconds of pure
/// overhead on work that is otherwise linear in the text.
pub struct ShrinkPass {
    session: Session,
    task: Task,
}

impl ShrinkPass {
    /// Open a pass sized for the largest chunk it will be given.
    ///
    /// `widest` is the longest chunk in characters, which the caller already knows from
    /// [`crate::shrink::chunks`]. Sizing from the actual chunks rather than from
    /// [`shrink_chunk_chars`] means a short document opens a small window.
    pub fn open(widest: usize) -> Result<Self, String> {
        let output = shrink_output_tokens_for(widest);
        let instruction = crate::shrink::INSTRUCTION.len();
        let ctx = fit_context_for(
            &[(instruction + widest, output)],
            crate::config::prefs_snapshot().model_context_ceiling,
        );
        Ok(ShrinkPass {
            session: Session::open(ctx)?,
            task: Task::extraction(
                json!({
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["prompt"],
                    "properties": {"prompt": {"type": "string", "minLength": 1}}
                }),
                output,
            ),
        })
    }

    /// Rewrite one chunk in the compressed form [`crate::shrink`] asks for.
    ///
    /// Greedy, like the personal-data scan and for the same reason: most of the output is text
    /// copied from the input — paths, identifiers, code — and there is nothing for temperature to
    /// diversify but the copy. Reasoning is not enabled either; deliberating about a sentence
    /// costs more seconds than the sentence is worth, once per chunk.
    pub fn chunk(&self, body: &str) -> Result<String, String> {
        let reply = self
            .session
            .run(&self.task, &crate::shrink::compose(body))?;
        reply
            .get("prompt")
            .and_then(Value::as_str)
            // The schema constrains the shape, not the manners: a model told "no preamble" still
            // opens with one often enough that the cleanup has to stay.
            .map(crate::shrink::clean)
            .ok_or_else(|| "the model's reply had no rewritten prompt".to_string())
    }
}

/// Room for a plan: several times the request, because planning *adds* structure.
///
/// The opposite budget to a shrink, and the floor matters far more than the slope — the
/// shortest requests are the ones that expand the most. A one-line request still has to
/// come back as six fields with steps and acceptance criteria, which is a thousand tokens
/// before the request itself contributes anything.
///
/// Undersizing this does not produce a shorter plan. The reply is a single JSON object, so
/// running out mid-array yields no object at all — the failure is "truncated JSON in the
/// model's reply", after the whole generation has been paid for.
/// The floor covers the schema's own worst case — 34 entries of 240 characters plus the
/// JSON scaffolding, about 3 000 tokens — so a plan can always be finished, and the slope
/// is what a longer request adds on top by quoting more of itself back.
fn plan_output_tokens_for(chars: usize) -> usize {
    let tokens = (chars as f32 / CHARS_PER_TOKEN).ceil() as usize;
    (tokens + 3072).clamp(3072, 4608)
}

/// Turn a rough request into an agent-ready instruction, on the local checkpoint.
///
/// One call, not a chunked pass: a plan is a single structure over the whole request, and
/// planning halves of it separately would produce two objectives and two sets of
/// acceptance criteria. That caps the request at the context ceiling — see
/// [`crate::plan::run`], which reports an over-long prompt rather than letting llama.cpp
/// silently drop its tail.
///
/// Greedy, like the shrink: the parts that matter most are paths, identifiers and
/// commands copied out of the request, and there is nothing for temperature to diversify
/// but the copy.
pub fn plan(text: &str) -> Result<String, String> {
    let instruction = crate::plan::INSTRUCTION.len();
    let output = plan_output_tokens_for(text.len());
    let ceiling = crate::config::prefs_snapshot().model_context_ceiling;
    let ctx = fit_context_for(&[(instruction + text.len(), output)], ceiling);

    let session = Session::open(ctx)?;
    // One field per section, rather than one string containing the whole plan. Asked for the
    // latter, this checkpoint fills it with an account of the plan it is about to write —
    // the schema is the only thing that reliably stops that, because it leaves nowhere to
    // put the preamble. `minItems` on the two load-bearing lists is the same idea: a plan
    // with no steps and no acceptance criteria is not a short plan, it is not a plan.
    // Bounded on purpose, and this is load-bearing rather than tidiness. An unbounded array
    // of unbounded strings is a grammar the model can stay inside forever: it will keep
    // adding plausible constraints until the token budget runs out, and because the reply is
    // one JSON object, running out means there is no object at all — the whole generation is
    // paid for and thrown away as "truncated JSON". The caps are what make the worst case
    // finite, and `plan_output_tokens_for` is sized to cover it.
    let entry = json!({"type": "string", "minLength": 1, "maxLength": 240});
    let list = json!({"type": "array", "maxItems": 6, "items": entry});
    let steps = json!({"type": "array", "minItems": 1, "maxItems": 10, "items": entry});
    let criteria = json!({"type": "array", "minItems": 1, "maxItems": 6, "items": entry});
    let task = Task::composition(
        json!({
            "type": "object",
            "additionalProperties": false,
            "required": ["objective", "context", "steps", "constraints", "done_when",
                         "open_questions"],
            "properties": {
                "objective": {"type": "string", "minLength": 1, "maxLength": 240},
                "context": list,
                "steps": steps,
                "constraints": list,
                "done_when": criteria,
                "open_questions": list,
            }
        }),
        output,
    );

    let reply = session.run(&task, &crate::plan::compose(text))?;
    let objective = reply
        .get("objective")
        .and_then(Value::as_str)
        .ok_or("the model's reply had no objective")?;
    let sections: Vec<(&str, Vec<String>)> = crate::plan::FIELDS
        .iter()
        .skip(1)
        .map(|(key, heading)| (*heading, strings(&reply, key)))
        .collect();
    Ok(crate::plan::render(objective, &sections))
}

/// One field of a fill-in-the-fields reply, as a list of strings.
///
/// A missing or malformed list reads as empty rather than as an error: the schema already
/// requires the ones that have to be there, so what reaches this is an optional section the
/// model had nothing to put in.
fn strings(reply: &Value, key: &str) -> Vec<String> {
    reply
        .get(key)
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// Room for a postmortem: the largest output pstore asks this checkpoint for.
///
/// Nine fields rather than the planner's six, and the floor is what almost always applies —
/// incident notes are pasted, so the input is long and the output is a fixed-size document
/// over it rather than something that grows with it.
///
/// The floor covers the schema's own worst case, and the two are meant to be read together:
/// every field filled to its cap is 13 450 characters, about 4 490 tokens at
/// [`CHARS_PER_TOKEN`], with the JSON scaffolding around nine keys on top. Undersizing it
/// fails the way [`plan_output_tokens_for`] describes — one JSON object, so running out
/// mid-array yields no object at all, after the whole generation has been paid for.
///
/// The caps in [`rca`] cannot be loosened without raising this, and this cannot be raised
/// far: it is subtracted from the same ceiling the notes have to fit inside, and notes are
/// the longest input pstore takes. [`rca_input_chars`] is where that trade-off shows up.
///
/// A constant, unlike [`plan_output_tokens_for`], and the difference is the point. A plan
/// quotes its request back, so a longer request earns a longer plan. A postmortem does not:
/// the schema's caps are the same whatever the notes weigh, so scaling this with the input
/// would reserve room the model is not permitted to use — and every token reserved here is
/// one the notes cannot have.
const RCA_OUTPUT_TOKENS: usize = 4864;

/// Turn incident notes into a root cause analysis and postmortem, on the local checkpoint.
///
/// One call over the whole of the notes, for the reason [`plan`] is: an incident has one
/// timeline and one root cause, and analysing halves of it separately would produce two of
/// each. [`crate::rca::run`] refuses notes too long for that rather than letting llama.cpp
/// drop their tail — which here would silently truncate the incident itself.
///
/// Greedy. What matters most in a postmortem is the material copied out of the notes —
/// times, hostnames, error strings, measured quantities — and there is nothing for
/// temperature to diversify but the copy.
pub fn rca(text: &str) -> Result<String, String> {
    let instruction = crate::rca::INSTRUCTION.len();
    let output = RCA_OUTPUT_TOKENS;
    let ceiling = crate::config::prefs_snapshot().model_context_ceiling;
    let ctx = fit_context_for(&[(instruction + text.len(), output)], ceiling);

    let session = Session::open(ctx)?;
    // One field per section, and bounded at both ends, for the reasons spelled out in
    // `plan`. Two differences from the planner's schema, both learned from what came back:
    //
    // The entries are longer. A `maxLength` under a constrained grammar is not a hint — the
    // closing quote is forced at the limit, mid-word — and 240 characters, ample for "edit
    // src/net/retry.rs", cuts an entry naming a service, a time and an error string in half.
    // The instruction asks for entries under 200 so that the cap is headroom rather than the
    // thing shaping the answer.
    //
    // And `minItems` marks only the three sections whose absence would make this something
    // other than a postmortem. `resolution` is deliberately *not* among them: notes that stop
    // while the incident is still burning have no resolution to record, and requiring one
    // makes the model write the fix it thinks ought to happen as though it already had.
    // Per field rather than one size for all of them, because the ceiling makes it a real
    // budget: everything allowed here has to be generatable inside `output`, and `output` has
    // to leave room for notes worth analysing. A timeline entry carrying a time, a host and
    // an error string needs the room; an action item is a sentence.
    let entry = |chars: usize| json!({"type": "string", "minLength": 1, "maxLength": chars});
    let list =
        |max: usize, chars: usize| json!({"type": "array", "maxItems": max, "items": entry(chars)});
    let required = |min: usize, max: usize, chars: usize| json!({"type": "array", "minItems": min, "maxItems": max, "items": entry(chars)});
    let task = Task::composition(
        json!({
            "type": "object",
            "additionalProperties": false,
            "required": ["summary", "impact", "timeline", "root_cause", "contributing_factors",
                         "detection", "resolution", "action_items", "open_questions"],
            "properties": {
                "summary": entry(1200),
                "impact": list(4, 300),
                "timeline": required(1, 12, 300),
                "root_cause": required(1, 4, 400),
                "contributing_factors": list(5, 250),
                "detection": list(4, 250),
                "resolution": list(4, 250),
                // The one field whose shape is enforced rather than requested. Asked in prose
                // to begin each item with 'prevent:', 'detect:' or 'mitigate:', this
                // checkpoint complies on short notes and quietly stops on long ones — and
                // the prefix is what the list is sorted and exported by, so losing it costs
                // more than a formatting slip. An enum is not advice: the grammar cannot
                // emit anything else.
                "action_items": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 8,
                    "items": {
                        "type": "object",
                        "additionalProperties": false,
                        "required": ["buys", "task"],
                        "properties": {
                            "buys": {"enum": ["prevent", "detect", "mitigate"]},
                            "task": entry(200),
                        }
                    }
                },
                "open_questions": list(5, 200),
            }
        }),
        output,
    );

    let reply = session.run(&task, &crate::rca::compose(text))?;
    let summary = reply
        .get("summary")
        .and_then(Value::as_str)
        .ok_or("the model's reply had no summary")?;
    let sections: Vec<(&str, Vec<String>)> = crate::rca::FIELDS
        .iter()
        .skip(1)
        .map(|(key, heading)| {
            let items = match *key {
                "action_items" => action_items(&reply),
                _ => strings(&reply, key),
            };
            (*heading, items)
        })
        .collect();
    Ok(crate::rca::render(summary, &sections))
}

/// The action items, flattened back to the `buys: task` line the document is written in.
///
/// The split into two fields exists to make the grammar enforce the prefix; nothing
/// downstream wants the pair. An item missing either half is dropped rather than rendered
/// half-formed — the schema requires both, so this is only reachable if that stops being
/// true.
fn action_items(reply: &Value) -> Vec<String> {
    reply
        .get("action_items")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| {
                    let buys = item.get("buys").and_then(Value::as_str)?;
                    let task = item.get("task").and_then(Value::as_str)?;
                    Some(format!("{buys}: {}", task.trim()))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Largest set of notes [`rca`] can take without the context ceiling truncating it.
///
/// Found by search rather than by hand, for the reason [`plan_input_chars`] is.
pub fn rca_input_chars() -> usize {
    let ceiling = crate::config::prefs_snapshot().model_context_ceiling;
    let instruction = crate::rca::INSTRUCTION.len();
    let fits = |chars: usize| {
        let prompt_tokens = ((instruction + chars) as f32 / CHARS_PER_TOKEN).ceil() as usize;
        prompt_tokens + prompt_tokens / 4 + RCA_OUTPUT_TOKENS + 256 <= ceiling
    };
    search_input_chars(fits)
}

/// Answer a hint on the local checkpoint.
///
/// Composed by [`crate::hints::compose`], same as the agent path, so the two differ in who
/// answers and nothing else.
///
/// Sampled rather than greedy, and for the reason [`Task::composition`] exists: an answer
/// is written, not copied, and at temperature zero this checkpoint restates its first
/// sentence until the budget runs out.
pub fn hint(prompt: &str) -> Result<String, String> {
    let output = HINT_OUTPUT_TOKENS;
    let ceiling = crate::config::prefs_snapshot().model_context_ceiling;
    let session = Session::open(fit_context(prompt, output, ceiling))?;
    let task = Task::composition(
        json!({
            "type": "object",
            "additionalProperties": false,
            "required": ["answer"],
            "properties": {"answer": {"type": "string", "minLength": 1, "maxLength": 4000}}
        }),
        output,
    );
    session
        .run(&task, prompt)?
        .get("answer")
        .and_then(Value::as_str)
        .map(crate::shrink::clean)
        .ok_or_else(|| "the model's reply had no answer".to_string())
}

/// Room for a hint. Bounded because a hint is read in a side panel mid-edit: an answer
/// longer than this is one the developer will not read, whatever it says.
const HINT_OUTPUT_TOKENS: usize = 1536;

/// Largest request [`plan`] can take without the context ceiling truncating it.
///
/// Found by search rather than by solving the budget by hand: [`plan_output_tokens_for`]
/// is clamped at both ends, so the relationship between input size and window size is
/// piecewise, and a closed form for it would be a second copy of the sizing rule free to
/// drift from the first. This asks the real functions instead.
pub fn plan_input_chars() -> usize {
    let ceiling = crate::config::prefs_snapshot().model_context_ceiling;
    let instruction = crate::plan::INSTRUCTION.len();
    let fits = |chars: usize| {
        // `fit_context_for` clamps its answer to the ceiling, so a request that overruns
        // comes back looking like an exact fit. Compare against the unclamped need.
        let output = plan_output_tokens_for(chars);
        let prompt_tokens = ((instruction + chars) as f32 / CHARS_PER_TOKEN).ceil() as usize;
        prompt_tokens + prompt_tokens / 4 + output + 256 <= ceiling
    };
    search_input_chars(fits)
}

/// The largest input `fits` still accepts.
///
/// Shared by [`plan_input_chars`] and [`rca_input_chars`], which differ only in the budget
/// they hand in. `fits` must be monotonic — true for every size below its answer — which the
/// sizing rules are: a longer input never needs a smaller window.
fn search_input_chars(fits: impl Fn(usize) -> bool) -> usize {
    if !fits(0) {
        return 0;
    }
    let (mut lo, mut hi) = (0usize, 1usize << 20);
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if fits(mid) { lo = mid } else { hi = mid - 1 }
    }
    lo
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
    use crate::knowledge::Brief;

    fn candidate(
        agent: &'static str,
        model: &'static str,
        effort: Effort,
    ) -> super::super::Candidate {
        super::super::Candidate {
            agent_id: agent,
            agent_display: "Agent",
            model_id: model.into(),
            model_display: model.into(),
            tier: Tier::Mid,
            effort,
            effort_selectable: true,
            metered: false,
            relative_price: 1.0,
            quota_weight: 1.0,
        }
    }

    /// The burn signal has to reach the prompt, because the prompt is the only place it acts —
    /// pstore never re-sorts what the model emits (see `rank_prompt`'s caller), so a weight that
    /// stays in the struct changes nothing at all.
    #[test]
    fn quota_burn_reaches_the_ranking_prompt() {
        let heavy = real("claude", "opus", Effort::High);
        let light = real("claude", "haiku", Effort::Low);
        assert!(
            heavy.quota_weight > light.quota_weight,
            "fixture precondition"
        );

        let rows = rows(&[heavy, light], &Brief::default());
        let prompt = rank_prompt(
            "refactor this",
            &rows,
            2,
            Demand::Hard,
            Breadth::FewFiles,
            false,
        );

        assert!(
            prompt.contains("burns 5x quota"),
            "the heavy model must be marked:\n{prompt}"
        );
        assert!(
            !prompt.contains("burns 1x quota"),
            "a marker on every row carries no information:\n{prompt}"
        );
    }

    /// The one thing the burn signal must not become. `relative_price` is dollars-per-token and
    /// the ranker is forbidden from seeing it; quota burn is a different fact about an allowance
    /// already paid for. Letting price leak in under a new name would undo that decision quietly.
    #[test]
    fn the_prompt_states_burn_but_never_price() {
        let rows = rows(&[real("claude", "fable", Effort::Max)], &Brief::default());
        let prompt = rank_prompt("anything", &rows, 1, Demand::Hard, Breadth::FewFiles, false);

        assert!(prompt.contains("burns 10x quota"));
        assert!(
            !prompt.to_lowercase().contains("price")
                && !prompt.contains("$")
                && !prompt.contains("per MTok"),
            "price must not reach the ranker under any spelling:\n{prompt}"
        );
    }

    /// A candidate carrying the registry's real facts for `model`, so a test that depends on a
    /// model's tier — the ordering does — is not quietly asserting against `Tier::Mid`.
    fn real(agent: &'static str, model: &'static str, effort: Effort) -> super::super::Candidate {
        let spec = crate::agents::registry::find(agent).expect("an agent in the registry");
        let m = spec
            .models
            .iter()
            .find(|m| m.id == model)
            .unwrap_or_else(|| panic!("{agent} does not expose {model}"));
        super::super::Candidate {
            agent_id: spec.id,
            agent_display: spec.display,
            model_id: m.id.into(),
            model_display: m.display.into(),
            tier: m.tier,
            effort,
            effort_selectable: spec.effort_flag.is_supported(),
            metered: m.metered,
            relative_price: m.relative_price,
            quota_weight: m.quota_weight,
        }
    }

    /// A field of `n` distinct models, each at one effort — enough to exercise the grammar and
    /// the index alternation without naming anything real.
    fn test_rows(n: usize) -> Vec<Row> {
        let cands: Vec<super::super::Candidate> = (0..n)
            .map(|i| {
                let mut c = candidate("claude", "m", Effort::High);
                c.model_id = format!("model-{i}").into();
                c.model_display = c.model_id.clone();
                c
            })
            .collect();
        rows(&cands, &Brief::default())
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

    /// A shrink chunk, its stretch, the instruction and the rewrite all have to fit the
    /// window at once. If they do not, `llama-cli` drops the tail of the prompt without
    /// saying so and the model rewrites a document it only partly saw.
    #[test]
    fn a_stretched_shrink_chunk_still_fits_the_window() {
        let chunk = shrink_chunk_chars();
        assert!(chunk >= 400, "unusably small chunk: {chunk}");

        // The worst case chunks() can produce: max_chars plus a quarter, to keep a fenced
        // code block whole.
        let stretched = "x".repeat(chunk + chunk / 4);
        let prompt = crate::shrink::compose(&stretched);
        let ceiling = crate::config::prefs_snapshot().model_context_ceiling;
        let ctx = fit_context(&prompt, shrink_output_tokens_for(stretched.len()), ceiling);
        assert!(
            ctx < ceiling,
            "a stretched chunk needs {ctx} tokens against a {ceiling} ceiling — \
             shrink_chunk_chars is over-estimating what fits"
        );
    }

    /// The rewrite is allowed to be as long as the original. A prompt with nothing to give
    /// back must come home as valid JSON, not as a reply clipped at the token cap.
    #[test]
    fn shrink_output_has_room_for_an_unchanged_rewrite() {
        let text = "x".repeat(3_000);
        let tokens = shrink_output_tokens_for(text.len());
        assert!(
            tokens as f32 > text.len() as f32 / CHARS_PER_TOKEN,
            "{tokens} tokens cannot hold a rewrite the size of its input"
        );
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
        let rows = rows(&cands, &Brief::default());
        let reply = json!({"choices": [
            {"index": 1, "effort": "high", "fit": 90, "reason": "strong at refactors"},
            {"index": 0, "effort": "medium", "fit": 78, "reason": "cheaper and enough"},
        ]});

        let out = build_choices(&reply, &rows, &cands).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].model_id, "gpt", "emitted order is the ranking");
        assert_eq!(out[0].rationale, "strong at refactors");
        assert_eq!(out[1].model_id, "sonnet");
        // Latency comes from the registry, not from the model.
        assert_eq!(out[0].relative_latency, Effort::High.latency_factor());
        // And the index is kept, so a run of consecutive picks can be recognised later.
        assert_eq!(out[0].row_index, 1);
    }

    /// The grid is one row per (model, effort); the model ranks one row per *model* and names
    /// the effort. Getting that mapping wrong would launch an effort the agent never offered.
    #[test]
    fn efforts_collapse_into_one_row_per_model() {
        let cands = [
            candidate("claude", "opus", Effort::Low),
            candidate("claude", "opus", Effort::High),
            candidate("claude", "opus", Effort::Max),
            candidate("claude", "haiku", Effort::Low),
        ];
        let rows = rows(&cands, &Brief::default());

        assert_eq!(rows.len(), 2, "four candidates, two models");
        assert_eq!(rows[0].model_id, "opus");
        assert_eq!(
            rows[0].efforts,
            vec![Effort::Low, Effort::High, Effort::Max],
            "every effort the grid offered for that model, ascending"
        );
        assert_eq!(rows[1].efforts, vec![Effort::Low]);

        // The chosen effort has to survive onto the Choice, since that is what gets launched.
        let reply = json!({"choices": [
            {"index": 0, "effort": "max", "fit": 95, "reason": "hard refactor"},
        ]});
        let out = build_choices(&reply, &rows, &cands).unwrap();
        assert_eq!(out[0].effort, Effort::Max);
    }

    /// The grammar offers every effort any row supports, because it cannot tie one field to
    /// another — so an effort this model cannot be asked for has to be resolved, not launched.
    #[test]
    fn an_unavailable_effort_snaps_to_the_nearest_one_offered() {
        let cands = [
            candidate("claude", "opus", Effort::Low),
            candidate("claude", "opus", Effort::Max),
            // Gemini cannot be told an effort at all: the grid offers it exactly one.
            candidate("gemini", "flash", Effort::High),
        ];
        let rows = rows(&cands, &Brief::default());

        let gemini = rows.iter().find(|r| r.agent_id == "gemini").unwrap();
        assert_eq!(snap_effort(gemini, Effort::Max), Effort::High);
        assert_eq!(snap_effort(gemini, Effort::Low), Effort::High);

        let opus = rows.iter().find(|r| r.model_id == "opus").unwrap();
        assert_eq!(snap_effort(opus, Effort::Max), Effort::Max, "exact match");
        // Medium is one step from Low and two from Max, so it snaps down.
        assert_eq!(snap_effort(opus, Effort::Medium), Effort::Low);

        // End to end: asking Gemini for `max` yields a launchable choice, not a dropped one.
        let reply = json!({"choices": [
            {"index": 1, "effort": "max", "fit": 90, "reason": "fast enough"},
        ]});
        let out = build_choices(&reply, &rows, &cands).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].agent_id, "gemini");
        assert_eq!(out[0].effort, Effort::High);
    }

    /// Regression from a live run: the checkpoint ordered a shortlist correctly but scored
    /// it `fit: 0` then `fit: 1`. Re-sorting on `fit` inverted the answer it had given, so
    /// the emitted order has to win. The grammar's bands now make that reply unreachable, and
    /// the property still holds if someone changes them.
    #[test]
    fn a_useless_fit_scale_does_not_reorder_the_ranking() {
        let cands = [
            candidate("claude", "haiku", Effort::Low),
            candidate("claude", "opus", Effort::High),
        ];
        let rows = rows(&cands, &Brief::default());
        let reply = json!({"choices": [
            {"index": 0, "effort": "low", "fit": 0, "reason": "simple task, light model is enough"},
            {"index": 1, "effort": "high", "fit": 1, "reason": "more than this needs"},
        ]});

        let out = build_choices(&reply, &rows, &cands).unwrap();
        assert_eq!(
            out[0].model_id, "haiku",
            "the model's own ordering must survive its own scoring"
        );
    }

    #[test]
    fn duplicate_and_out_of_range_choices_are_dropped() {
        let cands = [candidate("claude", "sonnet", Effort::Medium)];
        let rows = rows(&cands, &Brief::default());
        let reply = json!({"choices": [
            {"index": 0, "effort": "medium", "fit": 90, "reason": "good"},
            {"index": 0, "effort": "medium", "fit": 70, "reason": "same option again"},
            {"index": 99, "effort": "medium", "fit": 99, "reason": "does not exist"},
        ]});

        let out = build_choices(&reply, &rows, &cands).unwrap();
        assert_eq!(out.len(), 1, "one real option, listed once");
        assert_eq!(out[0].fit, 90.0);

        // Nothing usable at all is an error, not an empty shortlist the UI would render as
        // "no models fit".
        let none = json!({"choices": [{"index": 99, "effort": "low", "fit": 99, "reason": "no"}]});
        assert!(build_choices(&none, &rows, &cands).is_err());
    }

    /// What pstore knows about a model has to reach the prompt, because the checkpoint's
    /// training predates every model in the field. Without this the ranker is judging names.
    #[test]
    fn model_facts_reach_the_prompt() {
        use crate::knowledge::{Known, Source};

        let cands = [candidate("claude", "opus", Effort::High)];
        let brief = Brief {
            known: vec![Known {
                model: "opus".into(),
                note: "frontier model, best for hard refactors".into(),
                source: Source::Table,
            }],
            unknown: Vec::new(),
        };
        let rows = rows(&cands, &brief);
        assert_eq!(rows[0].note, "frontier model, best for hard refactors");

        let prompt = rank_prompt(
            "do a thing",
            &rows,
            1,
            Demand::Hard,
            Breadth::FewFiles,
            false,
        );
        assert!(
            prompt.contains("frontier model, best for hard refactors"),
            "the note is missing from the prompt: {prompt}"
        );
        // And the retry instruction is only added when there is something to correct.
        assert!(!prompt.contains("repeated itself"));
        assert!(
            rank_prompt(
                "do a thing",
                &rows,
                1,
                Demand::Hard,
                Breadth::FewFiles,
                true
            )
            .contains("repeated itself")
        );
    }

    /// Observed on both builds: the model writes past the sampler's stop and the reply ends
    /// mid-word. A shortlist row reading "excels at repo-scale edits and complex refactoring
    /// logi" looks like pstore corrupted it.
    #[test]
    fn a_truncated_reason_is_cut_back_to_a_word() {
        let cut = "Coding-specialized frontier model excels at repo-scale edits and complex refac";
        assert!(
            cut.len() < REASON_CHARS,
            "this fixture should be under the cap"
        );
        assert_eq!(tidy_reason(cut), cut, "a short reason is left alone");

        // At the cap, which is what the sampler stopping mid-word looks like.
        let at_cap: String =
            std::iter::repeat_n("word ", 20).collect::<String>()[..REASON_CHARS].to_string();
        let tidied = tidy_reason(&at_cap);
        assert!(tidied.ends_with('…'), "{tidied:?}");
        assert!(!tidied.contains("wor…"), "cut mid-word: {tidied:?}");

        // Trailing punctuation left dangling by the cut goes with it.
        let dangling = format!("{}, and", "x".repeat(REASON_CHARS - 6));
        assert!(!tidy_reason(&dangling).contains(",…"), "{dangling:?}");

        // One unbroken token at the cap has nothing to cut back to, but must still be marked.
        let one_word = "x".repeat(REASON_CHARS);
        assert_eq!(tidy_reason(&one_word), format!("{one_word}…"));

        assert_eq!(tidy_reason("  spaced out  "), "spaced out");

        // The same trim serves the difficulty phrase, at its own cap — the two strings the model
        // writes for a person to read are both capped and both look mangled when cut mid-word.
        assert_eq!(
            tidy("a single typo in the README, which is a one", 42),
            "a single typo in the README, which is a…"
        );
        assert_eq!(tidy("short enough", 42), "short enough");
    }

    /// The bands are what make "all five scored 85" unrepresentable rather than merely
    /// discouraged, so they have to descend and not overlap.
    #[test]
    fn fit_bands_descend_without_overlapping() {
        let mut previous: Option<(u32, u32)> = None;
        for position in 0..super::super::SHORTLIST {
            let (lo, hi) = fit_band(position);
            assert!(lo <= hi, "band {position} is inverted: {lo}..={hi}");
            assert!(hi <= 99, "a fit over 99 needs three digits");
            assert!(lo >= 10, "a one-digit fit would not match the grammar");
            if let Some((prev_lo, _)) = previous {
                assert!(
                    hi < prev_lo,
                    "band {position} ({lo}..={hi}) overlaps the one above it"
                );
            }
            previous = Some((lo, hi));
        }
        assert_eq!(fit_band(0).1, 99, "the best pick can score full marks");
    }

    /// The digit alternation has to match exactly the range it claims, or the sampler either
    /// forbids a legal score or allows one outside the band.
    #[test]
    fn digit_ranges_cover_exactly_their_bounds() {
        assert_eq!(digits_in_range(85, 99), "\"8\" [5-9] | \"9\" [0-9]");
        assert_eq!(digits_in_range(70, 84), "\"7\" [0-9] | \"8\" [0-4]");
        // A range inside one decade, and a single value.
        assert_eq!(digits_in_range(40, 45), "\"4\" [0-5]");
        assert_eq!(digits_in_range(50, 50), "\"5\" \"0\"");
    }

    /// A degenerate answer is populated in every field and wrong in the only one that matters,
    /// so it has to be recognised rather than shown. Both signatures come from live 1-bit runs.
    #[test]
    fn enumeration_is_told_apart_from_ranking() {
        let pick = |index: usize, fit: f32, reason: &str| super::super::Choice {
            agent_id: "claude",
            agent_display: "Claude Code",
            model_id: "m".into(),
            model_display: "M".into(),
            tier: Tier::Mid,
            effort: Effort::High,
            effort_selectable: true,
            metered: false,
            relative_latency: 1.0,
            relative_price: 1.0,
            fit,
            rationale: reason.into(),
            quota_weight: 1.0,
            note: String::new(),
            fact_source: None,
            row_index: index,
        };

        // A real ranking: distinct options, descending scores, reasons that differ.
        let good = vec![
            pick(4, 95.0, "frontier model for a hard refactor"),
            pick(0, 80.0, "cheaper and probably enough"),
            pick(2, 65.0, "fast but likely to miss cases"),
        ];
        assert_eq!(degeneracy(&good, 15), None);

        // The one from the README: the same reason copy-pasted onto every pick.
        let same_reason = vec![
            pick(1, 95.0, "good fit for this prompt"),
            pick(2, 80.0, "good fit for this prompt"),
            pick(3, 65.0, "Good fit for this prompt"),
        ];
        let why = degeneracy(&same_reason, 15).expect("copy-pasted reasons are degenerate");
        assert!(why.contains("same reason"), "got {why}");

        // Identical scores, which the grammar now forbids and this still catches.
        let same_fit = vec![
            pick(1, 85.0, "frontier"),
            pick(2, 85.0, "mid"),
            pick(3, 85.0, "cheap"),
        ];
        assert!(degeneracy(&same_fit, 15).is_some());

        // Consecutive picks are NOT degenerate: `order_for` puts the options the difficulty
        // calls for at the top, so "the first three in order" is what a correct answer looks
        // like. Flagging it would cry wolf on exactly the results the check protects.
        let in_order = vec![
            pick(0, 95.0, "frontier, and this is a hard refactor"),
            pick(1, 80.0, "mid-tier, close behind"),
            pick(2, 65.0, "light, listed for completeness"),
            pick(3, 50.0, "slower for no gain here"),
        ];
        assert_eq!(degeneracy(&in_order, 15), None);

        // One pick out of a real field is not a ranking either.
        assert!(degeneracy(&good[..1], 15).is_some());
        // Unless that is all there was.
        assert_eq!(degeneracy(&good[..1], 1), None);
    }

    /// The difficulty read is the premise of the ranking, so what it tells the ranker to do has
    /// to differ per level and has to name the direction — that instruction is the whole fix for
    /// the 1-bit build putting Haiku first on a three-file refactor.
    #[test]
    fn each_difficulty_points_the_ranking_somewhere_different() {
        let all = [Demand::Easy, Demand::Moderate, Demand::Hard];

        let mut labels: Vec<&str> = all.iter().map(|d| d.as_str()).collect();
        labels.sort_unstable();
        labels.dedup();
        assert_eq!(labels.len(), all.len(), "levels must be distinguishable");

        let mut instructions: Vec<&str> = all.iter().map(|d| d.instruction()).collect();
        instructions.sort_unstable();
        instructions.dedup();
        assert_eq!(
            instructions.len(),
            all.len(),
            "each level needs its own steer"
        );

        // The two directions have to be stated as directions, not as preferences.
        assert!(Demand::Hard.instruction().contains("CAPABLE"));
        assert!(Demand::Easy.instruction().contains("LIGHTEST"));

        // And the steer has to reach the prompt, or the extra call bought nothing.
        let cands = [candidate("claude", "opus", Effort::High)];
        let rows = rows(&cands, &Brief::default());
        for d in all {
            let p = rank_prompt("x", &rows, 1, d, Breadth::FewFiles, false);
            assert!(p.contains(d.as_str()), "{d:?} is not stated in the prompt");
            assert!(
                p.contains(d.instruction()),
                "{d:?} does not tell the ranker what to do"
            );
        }

        // The instruction picks the weight class and says nothing about effort. Effort is
        // `target_effort`'s to decide, and a level named in two places is a level that can
        // disagree with itself — which is how `xhigh` and `max` came to be unreachable.
        for d in all {
            for level in Effort::ALL {
                assert!(
                    !d.instruction().contains(level.as_str()),
                    "{d:?} names an effort level ({level}) the breadth judgement has not seen yet"
                );
            }
        }
    }

    /// Difficulty picks the model, breadth picks how long it thinks — and the whole reason
    /// breadth is asked for at all is that difficulty alone could not reach the top of the
    /// ladder. `xhigh` and `max` were offered by the registry, admitted by the grammar, and
    /// requested by nothing.
    #[test]
    fn every_effort_level_is_reachable_from_some_judgement() {
        let all = [Demand::Easy, Demand::Moderate, Demand::Hard];
        let breadths = [Breadth::OneEdit, Breadth::FewFiles, Breadth::ManyFiles];

        let mut reached: Vec<Effort> = all
            .iter()
            .flat_map(|d| breadths.iter().map(|b| target_effort(*d, *b)))
            .collect();
        reached.sort_unstable();
        reached.dedup();
        assert_eq!(
            reached,
            Effort::ALL.to_vec(),
            "some effort level cannot be asked for by any prompt"
        );

        // Monotone in both arguments: harder never thinks less, and wider never thinks less.
        // Without this the table could reach every level and still be nonsense.
        for (i, d) in all.iter().enumerate() {
            for (j, b) in breadths.iter().enumerate() {
                if let Some(harder) = all.get(i + 1) {
                    assert!(
                        target_effort(*harder, *b) >= target_effort(*d, *b),
                        "{harder:?} thinks less than {d:?} at {b:?}"
                    );
                }
                if let Some(wider) = breadths.get(j + 1) {
                    assert!(
                        target_effort(*d, *wider) >= target_effort(*d, *b),
                        "{wider:?} thinks less than {b:?} at {d:?}"
                    );
                }
            }
        }

        // The one row that must never leave the fast end: a verbose request for a one-word fix
        // is still a one-word fix, and paying `high` for it is what this path exists to prevent.
        assert!(
            breadths
                .iter()
                .all(|b| target_effort(Demand::Easy, *b) <= Effort::Medium),
            "easy work was sent away to think"
        );
    }

    /// The retry has to be the first attempt's prompt **plus** a suffix, never a different prompt.
    /// `cache_prompt` reuses an exact prefix and nothing less, so a correction spliced into the
    /// middle costs a full re-evaluation of ~730 tokens — about ten seconds — to say one sentence.
    /// Appended, the retry pays for the sentence alone.
    #[test]
    fn the_retry_prompt_only_adds_to_the_first_one() {
        let cands = [
            candidate("claude", "opus", Effort::High),
            candidate("claude", "haiku", Effort::Low),
        ];
        let rows = rows(&cands, &Brief::default());

        let first = rank_prompt(
            "do a thing",
            &rows,
            2,
            Demand::Hard,
            Breadth::FewFiles,
            false,
        );
        let again = rank_prompt(
            "do a thing",
            &rows,
            2,
            Demand::Hard,
            Breadth::FewFiles,
            true,
        );

        assert!(
            again.starts_with(&first),
            "the retry diverges from the first attempt instead of extending it, so every token \
             is re-evaluated:\n--- first ---\n{first}\n--- retry ---\n{again}"
        );
        assert!(again.len() > first.len(), "the retry says nothing new");
        assert!(again.contains("repeated itself"), "{again}");
        assert!(!first.contains("repeated itself"), "{first}");
    }

    /// Both axes and what they came to have to reach the ranking prompt. The target effort in
    /// particular: the grammar anchors the top pick to it, and a prompt that did not also ask
    /// for it would leave the model constrained towards a level it was never told about.
    #[test]
    fn the_prompt_states_both_judgements_and_the_effort_they_imply() {
        let cands = [candidate("claude", "opus", Effort::XHigh)];
        let rows = rows(&cands, &Brief::default());

        let p = rank_prompt("x", &rows, 1, Demand::Hard, Breadth::FewFiles, false);
        assert!(p.contains("hard"), "{p}");
        assert!(p.contains(Breadth::FewFiles.label()), "{p}");
        assert!(
            p.contains("**xhigh** effort"),
            "the target effort is not asked for: {p}"
        );

        // A different breadth at the same difficulty has to change what is asked for, or the
        // second judgement is decoration.
        let wider = rank_prompt("x", &rows, 1, Demand::Hard, Breadth::ManyFiles, false);
        assert!(wider.contains("**max** effort"), "{wider}");
        assert!(wider.contains(Breadth::ManyFiles.label()), "{wider}");
    }

    /// The parser and the grammar have to agree on every spelling. They are two lists of the
    /// same strings in different places, and a mismatch is silent: the sampler emits a label the
    /// `match` in `judge_demand` does not recognise, and every prompt comes back `moderate` and
    /// `few_files` — a routing feature that has quietly stopped routing.
    #[test]
    fn the_difficulty_grammar_admits_exactly_what_the_parser_reads() {
        let g = demand_grammar();

        for label in [Demand::Easy, Demand::Moderate, Demand::Hard].map(Demand::as_str) {
            assert!(g.contains(&format!("\"{label}\"")), "{label} missing: {g}");
        }
        for label in [Breadth::OneEdit, Breadth::FewFiles, Breadth::ManyFiles].map(Breadth::as_str)
        {
            assert!(g.contains(&format!("\"{label}\"")), "{label} missing: {g}");
        }

        // Breadth is answered first, and that is the point of writing this grammar by hand:
        // greedy sampling makes each field condition the next, and with difficulty first the
        // two collapsed onto each other.
        let root = g.lines().next().expect("a root rule");
        let breadth_at = root.find("breadth").expect("breadth in the root rule");
        let difficulty_at = root
            .find("difficulty")
            .expect("difficulty in the root rule");
        let because_at = root.find("because").expect("because in the root rule");
        assert!(
            breadth_at < difficulty_at && difficulty_at < because_at,
            "the fields are answered in the wrong order: {root}"
        );

        // The trap the whole module is written around, and the one the schema path walked into:
        // an unbounded run of anything is somewhere the sampler can sit until the budget is gone.
        for (n, line) in g.lines().enumerate() {
            assert!(
                !line.contains('*') && !line.contains('+'),
                "line {n} can repeat without bound: {line}"
            );
        }
        assert!(!g.contains(" ws"), "no whitespace rule: {g}");
    }

    /// Both halves of the judgement are asked for, both are defined, and the prompt says they
    /// are independent. That last line is not padding: without it the two collapsed, and every
    /// request judged `hard` came back `many_files` — including a one-line lock-renewal fix.
    #[test]
    fn the_difficulty_prompt_defines_both_axes_separately() {
        let p = demand_prompt("rename a thing");

        for label in [Demand::Easy, Demand::Moderate, Demand::Hard].map(Demand::as_str) {
            assert!(p.contains(label), "{label} is not defined: {p}");
        }
        for label in [Breadth::OneEdit, Breadth::FewFiles, Breadth::ManyFiles].map(Breadth::as_str)
        {
            assert!(p.contains(label), "{label} is not defined: {p}");
        }
        assert!(p.contains("rename a thing"), "the request is missing: {p}");
        assert!(
            p.contains("independent"),
            "nothing stops the two judgements collapsing onto each other: {p}"
        );
        assert!(
            p.contains("not how long the request is"),
            "a polite four-sentence request for a typo fix will be read as work: {p}"
        );
    }

    /// A small model reads the top of the list, so the top of the list has to be the answer.
    /// Asked to shortlist a hard refactor from a light-to-frontier list, the 1-bit build returned
    /// options 0, 1, 2 in order — with reasons that contradicted its own ranking.
    #[test]
    fn options_are_presented_best_first_for_the_difficulty() {
        let mut cands = vec![
            candidate("claude", "haiku", Effort::Low),
            candidate("claude", "sonnet", Effort::Low),
            candidate("claude", "opus", Effort::Low),
        ];
        cands[0].tier = Tier::Cheap;
        cands[1].tier = Tier::Mid;
        cands[2].tier = Tier::Top;

        let first = |demand| {
            let mut rows = rows(&cands, &Brief::default());
            order_for(&mut rows, demand);
            rows.iter()
                .map(|r| r.model_id.to_string())
                .collect::<Vec<_>>()
        };

        assert_eq!(
            first(Demand::Hard)[0],
            "opus",
            "hard work leads with capability"
        );
        assert_eq!(
            first(Demand::Easy)[0],
            "haiku",
            "easy work leads with the light model"
        );
        assert_eq!(
            first(Demand::Moderate)[0],
            "sonnet",
            "and the middle with the middle"
        );

        // Every option is still offered — this reorders the list, it does not shorten it.
        assert_eq!(first(Demand::Hard).len(), 3);

        // The tail is ordered by distance too, so the worst fit is last rather than second.
        assert_eq!(first(Demand::Hard).last().unwrap(), "haiku");
        assert_eq!(first(Demand::Easy).last().unwrap(), "opus");

        // Light and frontier are the same distance from mid, so `moderate` would otherwise be
        // decided by whichever agent the user installed first. The lighter one goes above: the
        // two mistakes are not symmetrical, and a frontier model at 25x quota burn on ordinary
        // work costs an allowance that is needed later in the week.
        assert_eq!(
            first(Demand::Moderate),
            vec!["sonnet", "haiku", "opus"],
            "a tie in tier distance has to break downwards, not by registry order"
        );
    }

    /// The top pick's effort is bounded by the grammar rather than only asked for. Asking was
    /// tried: across a baseline sweep the checkpoint answered `low` for ten of eighteen prompts
    /// including a repo-wide migration, because `low` is the first alternative in the rule and
    /// nothing forbade it.
    #[test]
    fn the_grammar_anchors_the_top_picks_effort_to_the_target() {
        let cands: Vec<super::super::Candidate> = Effort::ALL
            .iter()
            .map(|e| candidate("claude", "opus", *e))
            .collect();
        let rows = rows(&cands, &Brief::default());

        let effort0 = |g: &str| {
            g.lines()
                .find(|l| l.starts_with("effort0 ::="))
                .expect("a rule for the top pick's effort")
                .to_string()
        };

        // The target and one step either side, and nothing else.
        let g = rank_grammar(&rows, 3, Effort::XHigh, 0);
        let top = effort0(&g);
        assert!(
            top.contains("high") && top.contains("xhigh") && top.contains("max"),
            "{top}"
        );
        assert!(!top.contains("\\\"low\\\""), "low is two steps away: {top}");
        assert!(!top.contains("medium"), "medium is two steps away: {top}");

        // At the bottom of the ladder the band is simply shorter — it is not wrapped or shifted.
        let low = effort0(&rank_grammar(&rows, 3, Effort::Low, 0));
        assert!(low.contains("low") && low.contains("medium"), "{low}");
        assert!(!low.contains("high"), "{low}");

        // Later positions keep the full alternation: they are alternatives, and an alternative
        // at the same effort as the pick above it is not one.
        let full = g
            .lines()
            .find(|l| l.starts_with("effort ::="))
            .expect("a rule for the rest");
        assert!(full.contains("low") && full.contains("max"), "{full}");
        assert!(g.contains("choice0 ::=") && g.contains(" effort0 "), "{g}");
        assert!(g.contains("choice1 ::=") && g.contains(" effort "), "{g}");
    }

    /// A field that offers nothing near the target must still be rankable. Gemini cannot be told
    /// an effort at all, so a machine with only Gemini installed offers exactly one level — and a
    /// rule matching none of it would make the whole reply unrepresentable, which is not a worse
    /// ranking but no ranking at all.
    #[test]
    fn an_effort_target_nothing_offers_falls_back_to_the_whole_field() {
        let cands = [candidate("gemini", "flash", Effort::High)];
        let rows = rows(&cands, &Brief::default());

        let g = rank_grammar(&rows, 1, Effort::Low, 0);
        let top = g
            .lines()
            .find(|l| l.starts_with("effort0 ::="))
            .expect("a rule for the top pick's effort");
        assert!(
            top.contains("high"),
            "the one effort on offer has to stay representable: {top}"
        );
    }

    /// The small build is asked an easier question than the big one. If this regresses it does
    /// so silently — the answers just get worse.
    #[test]
    fn the_small_build_is_asked_for_a_shorter_shortlist() {
        let small = shortlist_for(&models::LLM_1BIT);
        let big = shortlist_for(&models::LLM_TERNARY);
        assert!(
            small < big,
            "the 1-bit build should be asked for fewer picks ({small} vs {big})"
        );
        assert!(small >= 2, "fewer than two picks is not a ranking");
        assert_eq!(big, super::super::SHORTLIST);
    }

    /// The grammar is generated, so it is worth asserting that the generated thing contains
    /// the constraints it is generated for.
    #[test]
    fn the_grammar_binds_every_position() {
        let cands = [
            candidate("claude", "opus", Effort::Low),
            candidate("claude", "opus", Effort::High),
            candidate("claude", "haiku", Effort::Low),
        ];
        let rows = rows(&cands, &Brief::default());
        let g = rank_grammar(&rows, 2, Effort::High, 1400);

        assert!(g.contains("</think>"), "reasoning must be closed");
        assert!(
            g.contains("choice0 ::="),
            "each position needs its own rule"
        );
        assert!(g.contains("choice1 ::="));
        assert!(g.contains("fit0 ::="), "each position needs its own band");
        assert!(g.contains("fit1 ::="));
        assert!(
            g.contains("index ::= \"0\" | \"1\""),
            "one index per row: {g}"
        );
        assert!(g.contains("effort ::="), "effort is chosen, not indexed");
        assert!(
            !g.contains("[ \\t\\n]*"),
            "an unbounded whitespace rule hangs the sampler"
        );

        // Reasoning off means no think block at all, not an empty one.
        let quiet = rank_grammar(&rows, 2, Effort::High, 0);
        assert!(!quiet.contains("</think>"));
        assert!(quiet.starts_with("root ::= answer"));
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

    /// `CLOSING` is process-wide and refuses *every* session, so a test that raises it has to keep
    /// the others out of `Session::open` while it does. Held by every test in this section.
    ///
    /// The registry itself is deliberately **not** guarded: other tests reach the real model, so
    /// their sessions appear in `LIVE` alongside these ones. That is exactly the situation a
    /// closing app faces, so the tests below identify their own session by pid rather than
    /// assuming the registry is theirs.
    static PROCESSES: Mutex<()> = Mutex::new(());

    /// Pids currently registered as live.
    fn live_pids() -> Vec<u32> {
        recover(&LIVE).iter().map(|(p, _, _)| *p).collect()
    }

    /// Wait for `f`, up to `secs`, rather than sleeping a fixed amount.
    fn until(secs: u64, mut f: impl FnMut() -> bool) -> bool {
        for _ in 0..(secs * 100) {
            if f() {
                return true;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        false
    }

    #[cfg(unix)]
    fn alive(pid: u32) -> bool {
        std::process::Command::new("kill")
            .args(["-0", &pid.to_string()])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
    }

    /// A closing app must refuse to map weights rather than start something it is about to kill —
    /// the race that would otherwise leave gigabytes resident with no window left to show for it.
    #[test]
    fn a_closing_app_refuses_to_load() {
        let _guard = recover(&PROCESSES);
        CLOSING.store(true, Ordering::Relaxed);

        let why = refuse_if_stopping(&models::active()).expect_err("closing must refuse");
        assert!(why.contains("closing"), "got {why}");
        assert!(stopping());

        CLOSING.store(false, Ordering::Relaxed);
        assert!(!stopping());
    }

    /// Switching build must refuse a session that is about to map the one just abandoned, or both
    /// builds end up resident on a machine that was given the small one because memory is tight.
    #[test]
    fn a_session_for_the_abandoned_build_is_refused() {
        let _guard = recover(&PROCESSES);
        let other = if models::active().id == models::LLM_1BIT.id {
            models::LLM_TERNARY
        } else {
            models::LLM_1BIT
        };

        let why = refuse_if_stopping(&other).expect_err("the other build must be refused");
        assert!(why.contains("switched"), "got {why}");
        // And the selected one is allowed through.
        assert!(refuse_if_stopping(&models::active()).is_ok());
    }

    /// The registry is what makes both the shutdown and the build switch possible, so a session
    /// has to appear in it while it runs and leave when it ends.
    #[test]
    fn the_registry_tracks_a_session_by_pid_and_build() {
        let _guard = recover(&PROCESSES);
        let child = Arc::new(Mutex::new(
            std::process::Command::new("/bin/sh")
                .args(["-c", "sleep 30"])
                .stdin(std::process::Stdio::null())
                .spawn()
                .expect("a sleep should start"),
        ));
        let pid = recover(&child).id();

        register(pid, models::LLM_1BIT.id, Arc::clone(&child));
        assert!(live_pids().contains(&pid));

        // Stopping by build id reaches it, and does not touch anything else.
        let stopped = stop_live_where(|id| id == models::LLM_1BIT.id);
        assert!(stopped.contains(&models::LLM_1BIT.id), "got {stopped:?}");
        assert!(until(5, || !alive(pid)), "pid {pid} outlived the stop");
        assert!(!live_pids().contains(&pid), "it stayed in the registry");

        // Ending it twice is safe: that is the state both the owner's Drop and a switch reach.
        end(&child);
        deregister(pid);
    }

    /// End-to-end against the real checkpoint and the real runtime. Ignored by default — it
    /// needs a build downloaded, and takes tens of seconds per call.
    ///
    /// `cargo test -- --ignored live_model --nocapture`
    ///
    /// **Runs against every build that is on disk, not just the selected one.** The 1-bit build
    /// is the one that used to answer `Haiku 4.5, effort medium` and then indices 2, 3, 4, 5 with
    /// `fit: 85` on all of them, so a test that only exercised the default would pass while the
    /// build people choose for a small machine stayed broken. Each build gets its own section of
    /// output, and a failure names which one it was.
    #[test]
    #[ignore = "needs a downloaded checkpoint and a provisioned runtime"]
    fn live_model_ranks_and_finds_personal_data() {
        use crate::config::LocalModel;

        let cached: Vec<LocalModel> = [LocalModel::Ternary, LocalModel::OneBit]
            .into_iter()
            .filter(|m| models::is_cached(&m.checkpoint()))
            .collect();
        if cached.is_empty() {
            eprintln!("neither build is downloaded — nothing to run against");
            return;
        }

        for choice in cached {
            let checkpoint = choice.checkpoint();
            // A build that is on disk must resolve to a real file through the same path the app
            // uses. Selecting one pstore then cannot locate is the one way the switch fails
            // silently.
            let path = hub_path(&checkpoint)
                .unwrap_or_else(|e| panic!("{} cached but {e}", checkpoint.title));
            assert!(path.exists(), "{} resolved to {path:?}", checkpoint.title);

            let mut prefs = crate::config::prefs_snapshot();
            prefs.local_model = choice;
            crate::config::publish(&prefs);
            eprintln!("\n=== {} ===", checkpoint.title);

            // The full grid the app would build: two models at five efforts each, which is the
            // shape that used to be ranked by counting.
            let cands: Vec<super::super::Candidate> = ["haiku", "opus"]
                .into_iter()
                .flat_map(|model| {
                    crate::agents::registry::CLAUDE_EFFORTS
                        .iter()
                        .map(move |e| real("claude", model, *e))
                })
                .collect();

            // The facts pstore would supply, so the live run exercises the prompt the app builds
            // rather than a bare list of names.
            let names: Vec<String> = cands.iter().map(|c| c.model_id.to_string()).collect();
            let brief = crate::knowledge::resolve(
                &names,
                &|_| None,
                known_models,
                crate::knowledge::lookup,
            );
            for k in &brief.known {
                eprintln!("  {} — from {}", k.model, k.source.label());
            }

            let started = std::time::Instant::now();
            let ranking = rank("fix a typo in the README", &cands, Vec::new(), &brief)
                .unwrap_or_else(|e| panic!("{} could not rank: {e}", checkpoint.title));
            eprintln!("  ranked in {:.1}s", started.elapsed().as_secs_f32());
            for c in &ranking.choices {
                eprintln!(
                    "  {} · effort {} · fit {} · {}",
                    c.model_display, c.effort, c.fit, c.rationale
                );
            }

            assert_eq!(ranking.considered, cands.len());
            assert!(!ranking.choices.is_empty());

            // The property the whole rework is for: what comes back is a ranking, not the
            // options in the order they were listed.
            assert!(
                ranking.degenerate.is_none(),
                "{} listed the options instead of ranking them: {:?}",
                checkpoint.title,
                ranking.degenerate
            );
            // Distinct picks, and scores that descend — both now enforced by the grammar, so a
            // failure here means the grammar and the parser disagree.
            let mut picked: Vec<usize> = ranking.choices.iter().map(|c| c.row_index).collect();
            let before = picked.len();
            picked.sort_unstable();
            picked.dedup();
            assert_eq!(picked.len(), before, "{} repeated a pick", checkpoint.title);
            for pair in ranking.choices.windows(2) {
                assert!(
                    pair[0].fit > pair[1].fit,
                    "{}: fit did not descend ({} then {})",
                    checkpoint.title,
                    pair[0].fit,
                    pair[1].fit
                );
            }

            // And the routing judgement itself: a typo fix must not want the frontier model.
            let best = ranking.best().expect("a non-empty shortlist");
            assert_eq!(
                best.model_id, "haiku",
                "{} routed a trivial prompt to {} at effort {}",
                checkpoint.title, best.model_display, best.effort
            );

            let src = "Contact Mario Rossi at mario@example.com about invoice 42.";
            let found = detect_pii(src).expect("the model should scan");
            for f in &found {
                eprintln!("  {} = {:?}", f.tag, f.text);
                // Spans are located in the source, so whatever comes back must be real text.
                assert_eq!(&src[f.start..f.end], f.text, "span does not match its text");
            }
            assert!(
                found.iter().any(|f| f.text.contains("mario@example.com")),
                "{}: the address should have been found: {found:?}",
                checkpoint.title
            );
        }
    }

    /// The regression this whole rework exists for, against the real weights.
    ///
    /// `cargo test -- --ignored live_wide_field --nocapture`
    ///
    /// Asked to rank fifteen (model, effort) pairs for a hard three-file refactor, the 1-bit
    /// build answered `Haiku 4.5, effort medium` and then indices 2, 3, 4 and 5 in order,
    /// scoring all five `fit: 85` with the same reason copy-pasted onto each. It was counting,
    /// not ranking. This is that prompt, over a wider field than the original, run against every
    /// build on disk: a hard multi-file refactor must reach for capability, and the answer must
    /// be a ranking.
    #[test]
    #[ignore = "needs a downloaded checkpoint and a provisioned runtime"]
    fn live_wide_field_is_ranked_not_enumerated() {
        use crate::agents::registry::{CODEX_EFFORTS, Effort};
        use crate::config::LocalModel;

        const HARD: &str = "Refactor the authentication layer: move session handling out of \
                            src/api/handlers.rs into a new src/auth/session.rs, thread the store \
                            through src/api/mod.rs, and keep the existing 20 ms poll interval \
                            and every public signature unchanged.";

        let cached: Vec<LocalModel> = [LocalModel::Ternary, LocalModel::OneBit]
            .into_iter()
            .filter(|m| models::is_cached(&m.checkpoint()))
            .collect();
        if cached.is_empty() {
            eprintln!("neither build is downloaded — nothing to run against");
            return;
        }

        // A realistic field: three agents, six models, twenty-two combinations.
        let mut cands: Vec<super::super::Candidate> = Vec::new();
        for model in ["haiku", "sonnet", "opus"] {
            for e in crate::agents::registry::CLAUDE_EFFORTS {
                cands.push(real("claude", model, *e));
            }
        }
        for e in CODEX_EFFORTS {
            cands.push(real("codex", "gpt-5.1-codex", *e));
        }
        for model in ["gemini-3-flash", "gemini-3-pro"] {
            cands.push(real("gemini", model, Effort::High));
        }

        let mut failures: Vec<String> = Vec::new();
        let mut checked = 0usize;

        for choice in cached {
            let checkpoint = choice.checkpoint();
            let mut prefs = crate::config::prefs_snapshot();
            prefs.local_model = choice;
            crate::config::publish(&prefs);
            eprintln!(
                "\n=== {} · {} combinations ===",
                checkpoint.title,
                cands.len()
            );

            let names: Vec<String> = cands.iter().map(|c| c.model_id.to_string()).collect();
            let brief = crate::knowledge::resolve(
                &names,
                &|_| None,
                known_models,
                crate::knowledge::lookup,
            );
            let started = std::time::Instant::now();
            let ranking = rank(HARD, &cands, Vec::new(), &brief)
                .unwrap_or_else(|e| panic!("{} could not rank: {e}", checkpoint.title));
            eprintln!("  ranked in {:.1}s", started.elapsed().as_secs_f32());
            // The premise of everything below it: a shortlist that looks wrong is usually a
            // difficulty read that was wrong, and without this the failure is unattributable.
            match &ranking.judged {
                Some(j) => eprintln!("  judged {} — {}", j.summary(), j.because),
                None => eprintln!("  no difficulty read"),
            }
            for c in &ranking.choices {
                eprintln!(
                    "  [{}] {} via {} · effort {} · fit {} · {}",
                    c.row_index, c.model_display, c.agent_display, c.effort, c.fit, c.rationale
                );
            }

            // Collected rather than asserted one at a time. The point of running every build on
            // disk is to learn how each of them does, and `assert!` inside the loop throws that
            // away: the first build to fail ends the run, and the build that is actually the
            // default goes untested. Ternary is first in the list, so a ternary regression used
            // to hide the 1-bit result entirely.
            let mut check = |ok: bool, what: String| {
                checked += 1;
                if !ok {
                    failures.push(format!("{}: {what}", checkpoint.title));
                }
            };

            check(
                ranking.degenerate.is_none(),
                format!("enumerated instead of ranking ({:?})", ranking.degenerate),
            );
            // The judgement itself: this prompt names three files, an invariant to preserve and
            // every public signature. It is not work for the lightest model available.
            let best = ranking.best().expect("a non-empty shortlist");
            check(
                best.model_id != "haiku",
                format!(
                    "sent a hard three-file refactor to the lightest model at effort {}",
                    best.effort
                ),
            );
            check(
                best.model_id != "gemini-3-flash",
                "sent a hard three-file refactor to a light model".into(),
            );

            // And the other half of the judgement, which is what breadth was added for. This
            // prompt names three files and every public signature; whatever model it lands on,
            // asking that model for the cheapest thinking it offers is the wrong answer. Before
            // breadth was judged, `high` was the ceiling any prompt could reach and `low` was
            // where ten of eighteen top picks landed.
            let judged = ranking.judged.as_ref().expect("a judgement");
            check(
                judged.effort >= Effort::High,
                format!(
                    "steered a hard multi-file refactor to effort {} ({})",
                    judged.effort,
                    judged.summary()
                ),
            );
            check(
                best.effort > Effort::Low,
                "asked for the cheapest thinking on a hard refactor".into(),
            );
        }

        assert!(
            failures.is_empty(),
            "{} of {} checks failed:\n  {}",
            failures.len(),
            checked,
            failures.join("\n  ")
        );
    }

    /// Every repetition in the grammar has to be bounded. An unbounded rule is a place the
    /// sampler can legally sit forever, and it has: a `ws ::= [ \t\n]*` between two JSON
    /// tokens once cost a whole generation budget in spaces, and the call returned no answer
    /// at all. This is cheaper to assert than to rediscover.
    #[test]
    fn the_grammar_has_no_unbounded_repetition() {
        let g = rank_grammar(&test_rows(15), 5, Effort::High, 1400);
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
        let g = rank_grammar(&test_rows(3), 2, Effort::High, 0);
        let index = g
            .lines()
            .find(|l| l.starts_with("index ::="))
            .expect("an index rule");
        assert_eq!(index, "index ::= \"0\" | \"1\" | \"2\"");

        // Reasoning off means no think block to close, and the answer starts immediately.
        assert!(g.starts_with("root ::= answer"), "{g}");
        assert!(!g.contains(END_OF_THOUGHT), "{g}");

        // One candidate is still a valid list, and must not produce an empty alternation.
        assert!(rank_grammar(&test_rows(1), 1, Effort::High, 0).contains("index ::= \"0\""));
    }

    /// With a budget the reasoning block is *permitted* up to the cap and `</think>` is then
    /// *required* — that requirement is the whole mechanism. If the close became optional the
    /// model would think until the token budget ran out, which is the failure this replaced.
    #[test]
    fn a_reasoning_budget_is_a_cap_and_a_forced_close() {
        let g = rank_grammar(&test_rows(15), 5, Effort::High, 900);
        assert!(
            g.starts_with(&format!("root ::= thought \"{END_OF_THOUGHT}\" answer")),
            "{g}"
        );
        assert!(g.contains("thought ::= [^<]{0,900}"), "{g}");
    }

    /// The array has to come out exactly `want` long. Off by one here is a shortlist that is
    /// quietly the wrong length — and since each position now carries its own fit band, the
    /// count is spelled out in the `answer` rule rather than as a repetition.
    #[test]
    fn the_grammar_asks_for_exactly_the_shortlist_length() {
        // Counted by the numbered rule names, not by the word "choice" — the JSON key the rule
        // emits is `"choices"`, which contains it.
        let positions = |grammar: &str| {
            let answer = grammar
                .lines()
                .find(|l| l.starts_with("answer ::="))
                .expect("an answer rule")
                .to_string();
            (0..8)
                .filter(|i| answer.contains(&format!("choice{i}")))
                .count()
        };

        let five = rank_grammar(&test_rows(15), 5, Effort::High, 0);
        assert_eq!(positions(&five), 5, "{five}");
        assert!(five.contains("fit4 ::="), "the fifth band is missing");
        assert!(!five.contains("fit5 ::="), "a sixth position was generated");

        let one = rank_grammar(&test_rows(15), 1, Effort::High, 0);
        assert_eq!(positions(&one), 1, "{one}");
        let answer = one.lines().find(|l| l.starts_with("answer ::=")).unwrap();
        assert!(
            !answer.contains("\",\""),
            "one pick needs no separator: {answer}"
        );
    }

    /// Sampling follows the task, and the two kinds of work want opposite settings. Getting this
    /// wrong is invisible: a personal-data scan at 0.7 does not fail, it quietly drops a finding.
    #[test]
    fn each_kind_of_task_samples_the_way_its_work_needs() {
        let extract = Task::extraction(json!({"type": "object"}), 64);
        assert_eq!(extract.temperature, 0.0, "extraction must be greedy");
        assert!(matches!(extract.constrain, Constrain::Schema(_)));

        let judge = Task::judgement(rank_grammar(&test_rows(3), 2, Effort::High, 900), 400);
        assert_eq!(
            (judge.temperature, judge.top_p, judge.top_k),
            (0.7, 0.95, 20),
            "the checkpoint's published thinking-mode settings"
        );
        assert!(matches!(judge.constrain, Constrain::Grammar(_)));

        // And the reasoning block only exists on the judgement path.
        match judge.constrain {
            Constrain::Grammar(g) => assert!(g.contains(END_OF_THOUGHT)),
            Constrain::Schema(_) => panic!("a judgement must be able to think"),
        }
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
        assert!(without < 300, "{without} tokens for JSON alone");
    }

    /// Reasoning about a routing decision quotes JSON — braces and all — so the answer has to
    /// be taken from *after* the reasoning, never by scanning the whole reply for a `{`.
    #[test]
    fn the_reasoning_block_is_not_mistaken_for_the_answer() {
        let reply = "The schema wants {\"choices\":[...]} so I will emit\n\
                     </think>{\"choices\":[{\"index\":3,\"effort\":\"low\",\"fit\":90,\"reason\":\"ok\"}]}";
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
        let rows = rows(&cands, &Brief::default());

        let p = rank_prompt(
            "fix a typo",
            &rows,
            2,
            Demand::Easy,
            Breadth::OneEdit,
            false,
        );
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
        let rows = test_rows(30);
        let list_bytes = rank_prompt(
            "do a thing",
            &rows,
            5,
            Demand::Moderate,
            Breadth::FewFiles,
            false,
        )
        .len()
            - rank_prompt(
                "do a thing",
                &[],
                5,
                Demand::Moderate,
                Breadth::FewFiles,
                false,
            )
            .len();
        let per_line = list_bytes / 30;
        assert!(
            per_line < 60,
            "{per_line} bytes per candidate line is too verbose for a 30-line list"
        );
    }
}
