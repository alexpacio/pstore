//! Background work.
//!
//! Everything slow — spawning agents, loading models, downloading weights — runs on a
//! `std::thread` and reports back over an `mpsc` channel that the UI drains once per
//! frame. There is deliberately no async runtime: the only concurrency pstore needs is
//! "do this off the UI thread and tell me when it's done".

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::time::Duration;

use crate::agents::detect::Detected;
use crate::agents::failover::{self, AllFailed, Completed};
use crate::agents::registry::{Effort, PromptVia};
use crate::models::Checkpoint;
use crate::pii;
use crate::router::Ranking;

/// Identifies one running job so its output can be routed to the right panel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct JobId(pub u64);

/// What a job is for. Determines where its output lands in the UI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// An inline hint request.
    Hint,
    /// A prompt-shrinking pass.
    Shrink,
    /// Turning the prompt into an agent-ready instruction.
    Plan,
    /// Classifying the prompt and ranking the field.
    Rank,
    /// Re-probing installed agents.
    Detect,
    /// Downloading or loading local model weights.
    Models,
    /// Finding personal data in the prompt.
    Sanitize,
}

/// A message from a worker to the UI.
#[derive(Debug)]
pub enum Event {
    /// A job has begun; carries a label for the status bar.
    Started {
        id: JobId,
        kind: Kind,
        label: String,
    },
    /// Incremental output text.
    Chunk { id: JobId, text: String },
    /// Progress note (model download, agent being tried).
    Note { id: JobId, text: String },
    /// An agent run finished successfully.
    Finished {
        id: JobId,
        kind: Kind,
        result: Box<Completed>,
    },
    /// Every candidate failed.
    Failed {
        id: JobId,
        kind: Kind,
        error: String,
    },
    /// A ranking pass finished.
    Ranked { id: JobId, ranking: Box<Ranking> },
    /// A detection pass finished.
    Detected { id: JobId, agents: Vec<Detected> },
    /// A sanitisation pass finished.
    Scanned { id: JobId, scan: Box<pii::Scan> },
    /// A shrink pass finished; carries the whole rewritten prompt.
    Shrunk { id: JobId, text: String },
    /// A planning pass finished; carries the whole rewritten prompt.
    Planned { id: JobId, text: String },
    /// A job with no payload of its own finished; `note` goes to the status bar.
    ///
    /// Model downloads report through here: their detail is on the [`crate::models`]
    /// status board, which the UI polls, so the event only has to say "stopped".
    Done { id: JobId, kind: Kind, note: String },
    /// The job was cancelled by the user.
    Cancelled { id: JobId, kind: Kind },
}

impl Event {
    /// The job this event belongs to.
    pub fn id(&self) -> JobId {
        match self {
            Event::Started { id, .. }
            | Event::Chunk { id, .. }
            | Event::Note { id, .. }
            | Event::Finished { id, .. }
            | Event::Failed { id, .. }
            | Event::Ranked { id, .. }
            | Event::Detected { id, .. }
            | Event::Scanned { id, .. }
            | Event::Shrunk { id, .. }
            | Event::Planned { id, .. }
            | Event::Done { id, .. }
            | Event::Cancelled { id, .. } => *id,
        }
    }

    /// Whether this event ends the job.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            Event::Finished { .. }
                | Event::Failed { .. }
                | Event::Ranked { .. }
                | Event::Detected { .. }
                | Event::Scanned { .. }
                | Event::Shrunk { .. }
                | Event::Planned { .. }
                | Event::Done { .. }
                | Event::Cancelled { .. }
        )
    }
}

/// A handle the UI keeps for a running job.
#[derive(Debug, Clone)]
pub struct Handle {
    /// Job identifier.
    pub id: JobId,
    /// What the job is doing.
    pub kind: Kind,
    /// Human label.
    pub label: String,
    /// Set to request cancellation. Workers check it between steps.
    cancel: Arc<AtomicBool>,
}

impl Handle {
    /// Ask the worker to stop at its next checkpoint.
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    /// Whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }
}

/// Spawns jobs and hands out their events.
pub struct Runner {
    tx: Sender<Event>,
    rx: Receiver<Event>,
    next_id: AtomicU64,
}

impl Default for Runner {
    fn default() -> Self {
        Self::new()
    }
}

impl Runner {
    /// Create a runner with an unbounded channel.
    pub fn new() -> Self {
        let (tx, rx) = std::sync::mpsc::channel();
        Self {
            tx,
            rx,
            next_id: AtomicU64::new(1),
        }
    }

    fn alloc(&self) -> JobId {
        JobId(self.next_id.fetch_add(1, Ordering::Relaxed))
    }

    /// Drain everything queued. Call once per frame.
    pub fn drain(&self) -> Vec<Event> {
        self.rx.try_iter().collect()
    }

    /// Run `f` on a worker thread. `f` receives the job's sender and cancel flag.
    fn spawn<F>(&self, kind: Kind, label: String, f: F) -> Handle
    where
        F: FnOnce(JobId, Sender<Event>, Arc<AtomicBool>) + Send + 'static,
    {
        let id = self.alloc();
        let cancel = Arc::new(AtomicBool::new(false));
        let tx = self.tx.clone();
        let _ = tx.send(Event::Started {
            id,
            kind,
            label: label.clone(),
        });

        let worker_cancel = Arc::clone(&cancel);
        std::thread::Builder::new()
            .name(format!("pstore-{kind:?}-{}", id.0))
            .spawn(move || f(id, tx, worker_cancel))
            .expect("spawning a worker thread");

        Handle {
            id,
            kind,
            label,
            cancel,
        }
    }

    /// Detect installed agents.
    pub fn detect(&self, dir: PathBuf) -> Handle {
        self.spawn(Kind::Detect, "detecting agents".into(), move |id, tx, _| {
            let agents = crate::agents::detect::detect_all(&dir);
            let _ = tx.send(Event::Detected { id, agents });
        })
    }

    /// Rank the installed agents against `text` with the local model.
    ///
    /// `rank` is injected so this module does not depend on the router, and so tests can
    /// drive the job machinery without a 7.17 GB checkpoint.
    pub fn rank<R>(&self, text: String, agents: Vec<Detected>, rank: R) -> Handle
    where
        R: FnOnce(&str, &[Detected]) -> Result<Ranking, String> + Send + 'static,
    {
        self.spawn(Kind::Rank, "ranking models".into(), move |id, tx, _| {
            match rank(&text, &agents) {
                Ok(ranking) => {
                    let _ = tx.send(Event::Ranked {
                        id,
                        ranking: Box::new(ranking),
                    });
                }
                // Ranking has no fallback: if the model could not answer there is no
                // second-best ranking to show, only the reason.
                Err(error) => {
                    let _ = tx.send(Event::Failed {
                        id,
                        kind: Kind::Rank,
                        error,
                    });
                }
            }
        })
    }

    /// Look at the model cache and record what is already downloaded.
    ///
    /// Cheap and network-free, but it touches the filesystem a dozen times, so it stays
    /// off the UI thread like everything else.
    pub fn probe_models(&self) -> Handle {
        self.spawn(
            Kind::Models,
            "checking model cache".into(),
            move |id, tx, _| {
                crate::models::probe_cache();
                let _ = tx.send(Event::Done {
                    id,
                    kind: Kind::Models,
                    note: describe_cache(),
                });
            },
        )
    }

    /// Download `targets`, then load them, reporting progress on the status board.
    ///
    /// One job for the whole set so a "download everything" click is one thing to cancel.
    /// Cancellation is checked between files: a part-transferred file is resumed rather
    /// than restarted next time, because the Hub cache keys on content.
    pub fn fetch_models(&self, targets: Vec<Checkpoint>) -> Handle {
        let label = match targets.as_slice() {
            [one] => format!("downloading {} ({})", one.title, one.size_label()),
            many => format!(
                "downloading {} checkpoints ({})",
                many.len(),
                crate::models::bytes_label(many.iter().map(|c| c.bytes).sum())
            ),
        };
        self.spawn(Kind::Models, label, move |id, tx, cancel| {
            let mut failures = Vec::new();

            // The runtime first: it is 11-17 MB against the checkpoint's 7.17 GB, and
            // weights with nothing to run them are of no use. Failing here before the long
            // transfer starts is the kind thing to do.
            if let Err(e) = provision_runtime(&tx, id, &cancel) {
                failures.push(e);
            }

            for c in &targets {
                if cancel.load(Ordering::Relaxed) {
                    let _ = tx.send(Event::Cancelled {
                        id,
                        kind: Kind::Models,
                    });
                    return;
                }
                let _ = tx.send(Event::Note {
                    id,
                    text: format!("downloading {} — {}", c.title, c.size_label()),
                });
                if let Err(e) = crate::models::download(c, &cancel) {
                    failures.push(format!("{}: {e}", c.title));
                }
            }

            if cancel.load(Ordering::Relaxed) {
                let _ = tx.send(Event::Cancelled {
                    id,
                    kind: Kind::Models,
                });
                return;
            }

            // Load whatever arrived, so "downloaded" turns into "loaded" without the user
            // having to guess that scoring a prompt is what triggers it.
            failures.extend(load_downloaded(&targets, &tx, id));

            if failures.is_empty() {
                let _ = tx.send(Event::Done {
                    id,
                    kind: Kind::Models,
                    note: describe_cache(),
                });
            } else {
                let _ = tx.send(Event::Failed {
                    id,
                    kind: Kind::Models,
                    error: failures.join("\n"),
                });
            }
        })
    }

    /// Load already-downloaded checkpoints into memory.
    pub fn load_models(&self, targets: Vec<Checkpoint>) -> Handle {
        self.spawn(Kind::Models, "loading models".into(), move |id, tx, _| {
            let failures = load_downloaded(&targets, &tx, id);
            if failures.is_empty() {
                let _ = tx.send(Event::Done {
                    id,
                    kind: Kind::Models,
                    note: describe_cache(),
                });
            } else {
                let _ = tx.send(Event::Failed {
                    id,
                    kind: Kind::Models,
                    error: failures.join("\n"),
                });
            }
        })
    }

    /// Find personal data in `text`.
    pub fn sanitize(&self, text: String) -> Handle {
        self.spawn(
            Kind::Sanitize,
            "checking for personal data".into(),
            move |id, tx, _| match pii::sanitize(&text) {
                Ok(scan) => {
                    let _ = tx.send(Event::Scanned {
                        id,
                        scan: Box::new(scan),
                    });
                }
                // A scan that could not run must not arrive as an empty plan: "no personal
                // data found" would then be indistinguishable from "nothing looked".
                Err(error) => {
                    let _ = tx.send(Event::Failed {
                        id,
                        kind: Kind::Sanitize,
                        error,
                    });
                }
            },
        )
    }

    /// Compress `text` with the local model.
    ///
    /// Long prompts are several model calls in sequence — see [`crate::shrink::run`] — so
    /// cancellation is checked between them and reported as such. A pass that was stopped
    /// must not arrive as a result: a partly-compressed prompt looks exactly like a
    /// finished one in the diff.
    pub fn shrink(&self, text: String) -> Handle {
        self.spawn(
            Kind::Shrink,
            "shrinking the prompt".into(),
            move |id, tx, cancel| {
                let mut note = |text: String| {
                    let _ = tx.send(Event::Note { id, text });
                };
                let outcome = crate::shrink::run(&text, &cancel, &mut note);
                // Checked after the fact as well as between chunks: a stop pressed during
                // the last generation cannot interrupt it, but it still means the user does
                // not want the answer.
                match outcome {
                    Ok(text) if !cancel.load(Ordering::Relaxed) => {
                        let _ = tx.send(Event::Shrunk { id, text });
                    }
                    Ok(_) => {
                        let _ = tx.send(Event::Cancelled {
                            id,
                            kind: Kind::Shrink,
                        });
                    }
                    Err(_) if cancel.load(Ordering::Relaxed) => {
                        let _ = tx.send(Event::Cancelled {
                            id,
                            kind: Kind::Shrink,
                        });
                    }
                    Err(error) => {
                        let _ = tx.send(Event::Failed {
                            id,
                            kind: Kind::Shrink,
                            error,
                        });
                    }
                }
            },
        )
    }

    /// Answer a hint with the local model.
    ///
    /// The local path has nothing to stream — one call, one JSON object — so the answer
    /// arrives as a single [`Event::Chunk`], which is what the panel already accumulates.
    /// The agent path's incremental chunks and this one land in the same place.
    pub fn hint_local(&self, prompt: String) -> Handle {
        self.spawn(Kind::Hint, "hint".into(), move |id, tx, cancel| {
            let outcome = crate::router::llm::hint(&prompt);
            if cancel.load(Ordering::Relaxed) {
                let _ = tx.send(Event::Cancelled {
                    id,
                    kind: Kind::Hint,
                });
                return;
            }
            match outcome {
                Ok(text) => {
                    let _ = tx.send(Event::Chunk { id, text });
                    let _ = tx.send(Event::Done {
                        id,
                        kind: Kind::Hint,
                        note: "hint ready".into(),
                    });
                }
                Err(error) => {
                    let _ = tx.send(Event::Failed {
                        id,
                        kind: Kind::Hint,
                        error,
                    });
                }
            }
        })
    }

    /// Turn `text` into an agent-ready instruction with the local model.
    ///
    /// One model call rather than the shrink's several, so there is no point between steps
    /// at which cancellation can land — it is checked once, after the fact, because a stop
    /// pressed during the generation still means the user does not want the answer.
    pub fn plan(&self, text: String) -> Handle {
        self.spawn(
            Kind::Plan,
            "planning the prompt".into(),
            move |id, tx, cancel| {
                let outcome = crate::plan::run(&text);
                if cancel.load(Ordering::Relaxed) {
                    let _ = tx.send(Event::Cancelled {
                        id,
                        kind: Kind::Plan,
                    });
                    return;
                }
                match outcome {
                    Ok(text) => {
                        let _ = tx.send(Event::Planned { id, text });
                    }
                    Err(error) => {
                        let _ = tx.send(Event::Failed {
                            id,
                            kind: Kind::Plan,
                            error,
                        });
                    }
                }
            },
        )
    }

    /// Send `prompt` to the ranked agents, streaming output, failing over on error.
    // Plumbing: every argument is a distinct, unrelated input (program, argv,
    // stdin, cwd, timeout, cancellation, sink). Bundling them into a struct
    // would add a type to thread through without making any call site clearer.
    #[allow(clippy::too_many_arguments)]
    pub fn run_agent(
        &self,
        kind: Kind,
        label: String,
        prompt: String,
        agents: Vec<Detected>,
        ranking: Ranking,
        cwd: PathBuf,
        dir: PathBuf,
        timeout: Duration,
    ) -> Handle {
        self.spawn(kind, label, move |id, tx, cancel| {
            // Bridge the launcher's line channel onto the UI event channel, so text
            // reaches the panel as it is produced rather than at process exit.
            let (line_tx, line_rx) = std::sync::mpsc::channel::<String>();
            let pump_tx = tx.clone();
            let pump = std::thread::spawn(move || {
                for text in line_rx {
                    let _ = pump_tx.send(Event::Chunk { id, text });
                }
            });

            let outcome = failover::run_with_failover(
                &agents,
                &ranking,
                &prompt,
                &cwd,
                &dir,
                timeout,
                Some(&cancel),
                &line_tx,
            );
            drop(line_tx);
            let _ = pump.join();

            if cancel.load(Ordering::Relaxed) {
                let _ = tx.send(Event::Cancelled { id, kind });
                return;
            }

            match outcome {
                Ok(done) => {
                    for (agent, why) in &done.attempts {
                        let _ = tx.send(Event::Note {
                            id,
                            text: format!("{agent} unavailable ({why}); moved on"),
                        });
                    }
                    let _ = tx.send(Event::Finished {
                        id,
                        kind,
                        result: Box::new(done),
                    });
                }
                Err(AllFailed { attempts }) => {
                    let _ = tx.send(Event::Failed {
                        id,
                        kind,
                        error: AllFailed { attempts }.summary(),
                    });
                }
            }
        })
    }
}

/// Make sure `llama-cli` is present, downloading it if it is not.
///
/// Nothing to do when it is already resolvable — from an override, a system install, a
/// previous download, or `PATH`.
fn provision_runtime(tx: &Sender<Event>, id: JobId, cancel: &AtomicBool) -> Result<(), String> {
    let prefs = crate::config::prefs_snapshot();
    if crate::runtime::locate(prefs.llama_path.as_deref()).is_some() {
        return Ok(());
    }
    let asset = crate::runtime::asset()?;
    let _ = tx.send(Event::Note {
        id,
        text: format!(
            "downloading llama-cli {} ({})",
            crate::runtime::RELEASE_TAG,
            crate::models::bytes_label(asset.bytes)
        ),
    });
    crate::runtime::download(cancel)
        .map(|_| ())
        .map_err(|e| format!("llama-cli: {e}"))
}

/// Check whichever of `targets` are on disk, returning one message per failure.
///
/// State is reset first: a previous attempt may have recorded "not downloaded" for a
/// checkpoint that has since arrived, and that verdict is remembered on purpose so it is
/// not retried on every keystroke.
///
/// One checkpoint serves every feature, so this is a single check rather than one per
/// subsystem — and "checking" is all it is. Nothing is held in memory between calls: each
/// model call is a fresh `llama-cli` process, so there is no load step to watch, only a
/// question of whether the weights and the runtime are both present.
///
/// Only the **selected** build is checked. Downloading the one that is not in use is a
/// legitimate thing to do — fetch it now, switch to it later — and it should not provoke a
/// readiness check, still less report a failure, for a file nothing is about to run.
fn load_downloaded(targets: &[Checkpoint], tx: &Sender<Event>, id: JobId) -> Vec<String> {
    let active = crate::models::active();
    if !targets.iter().any(|c| c.id == active.id) {
        return Vec::new();
    }
    let _ = tx.send(Event::Note {
        id,
        text: "checking the local model…".into(),
    });
    crate::router::reset_classifiers();
    match crate::router::preload_classifiers() {
        Ok(()) => Vec::new(),
        Err(e) => vec![e],
    }
}

/// "Bonsai 27B (ternary) · loaded", for the status bar.
pub fn describe_cache() -> String {
    let active = crate::models::active();
    summarize_cache(active, crate::models::phase(active.id))
}

/// The wording, over a given checkpoint and phase rather than the live board.
///
/// Split out so it can be tested without racing the probe jobs that other tests start — the
/// board is process-wide, and a test that asserts against it directly is a test that
/// occasionally loses.
///
/// It names the **selected** build rather than counting the catalogue. Two builds are offered
/// but only one is ever run, so "1 of 2 local models loaded" would be arithmetic about a
/// situation nobody is in: the second one being absent is the normal case, not a shortfall.
fn summarize_cache(active: crate::models::Checkpoint, phase: crate::models::Phase) -> String {
    format!("{} · {}", active.title, phase.label())
}

/// Whether an agent needs its prompt on stdin, for the UI's "how it will be invoked"
/// readout.
pub fn prompt_delivery(via: PromptVia) -> &'static str {
    match via {
        PromptVia::Arg => "as an argument",
        PromptVia::Stdin => "on stdin",
    }
}

/// Format an effort level for the status bar, noting when it is only a prediction.
pub fn effort_label(effort: Effort, selectable: bool) -> String {
    if selectable {
        format!("effort {effort}")
    } else {
        format!("effort ~{effort} (agent's own setting)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_unique_and_increasing() {
        let r = Runner::new();
        let a = r.alloc();
        let b = r.alloc();
        assert_ne!(a, b);
        assert!(b.0 > a.0);
    }

    #[test]
    fn detect_job_reports_started_then_detected() {
        let dir = std::env::temp_dir().join(format!("pstore-jobs-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let r = Runner::new();
        let h = r.detect(dir.clone());
        assert_eq!(h.kind, Kind::Detect);

        // Wait for the terminal event rather than sleeping a fixed amount.
        let mut events = Vec::new();
        for _ in 0..400 {
            events.extend(r.drain());
            if events.iter().any(Event::is_terminal) {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }

        assert!(
            matches!(events.first(), Some(Event::Started { .. })),
            "got {events:?}"
        );
        assert!(
            events.iter().any(|e| matches!(e, Event::Detected { .. })),
            "no Detected event"
        );
        assert!(events.iter().all(|e| e.id() == h.id));
        std::fs::remove_dir_all(&dir).ok();
    }

    fn drain_until_terminal(r: &Runner) -> Vec<Event> {
        let mut events = Vec::new();
        for _ in 0..400 {
            events.extend(r.drain());
            if events.iter().any(Event::is_terminal) {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        events
    }

    #[test]
    fn rank_job_uses_the_injected_ranker() {
        let r = Runner::new();
        let h = r.rank("anything".into(), Vec::new(), |text, _agents| {
            assert_eq!(text, "anything", "the ranker receives the prompt");
            Ok(Ranking {
                considered: 7,
                ..Ranking::default()
            })
        });

        let events = drain_until_terminal(&r);
        let ranking = events
            .iter()
            .find_map(|e| match e {
                Event::Ranked { ranking, .. } => Some(ranking),
                _ => None,
            })
            .expect("no Ranked event");
        assert_eq!(ranking.considered, 7);
        assert!(events.iter().all(|e| e.id() == h.id));
    }

    /// Ranking has no fallback, so a failure has to arrive as a failure. If it came back
    /// as an empty `Ranked` the UI would render "no models fit this prompt", which is a
    /// different and wrong claim.
    #[test]
    fn a_failed_ranking_reports_the_reason() {
        let r = Runner::new();
        r.rank("anything".into(), Vec::new(), |_, _| {
            Err("model not downloaded".into())
        });

        let events = drain_until_terminal(&r);
        let error = events
            .iter()
            .find_map(|e| match e {
                Event::Failed { error, .. } => Some(error.clone()),
                _ => None,
            })
            .expect("no Failed event");
        assert_eq!(error, "model not downloaded");
        assert!(
            !events.iter().any(|e| matches!(e, Event::Ranked { .. })),
            "a failed ranking must not also emit a ranking"
        );
    }

    /// Drain until a terminal event arrives, or give up.
    fn drain_until_done(r: &Runner) -> Vec<Event> {
        let mut events = Vec::new();
        for _ in 0..600 {
            events.extend(r.drain());
            if events.iter().any(Event::is_terminal) {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        events
    }

    #[test]
    fn a_cache_probe_finishes_without_touching_the_network() {
        let r = Runner::new();
        let started = std::time::Instant::now();
        let h = r.probe_models();
        assert_eq!(h.kind, Kind::Models);

        let events = drain_until_done(&r);
        let note = events
            .iter()
            .find_map(|e| match e {
                Event::Done { note, .. } => Some(note.clone()),
                _ => None,
            })
            .unwrap_or_else(|| panic!("no Done event: {events:?}"));
        // Names *a* build rather than the currently-selected one: the preference is
        // process-wide and other tests switch it, so pinning this to `active()` would be a
        // test that occasionally loses a race it is not about.
        assert!(
            crate::models::ALL.iter().any(|c| note.contains(c.title)),
            "the readout should name the build in use, got {note}"
        );
        assert!(
            started.elapsed() < Duration::from_secs(20),
            "a probe must not wait on downloads"
        );
        // Every checkpoint now has a definite answer rather than "unknown" — including the
        // one that is not selected, which the Models window still has to describe.
        for (c, phase) in crate::models::snapshot() {
            assert_ne!(
                phase,
                crate::models::Phase::Unknown,
                "{} was left unprobed",
                c.id
            );
        }
    }

    /// A sanitize job must end in exactly one of two states: a real result, or a stated
    /// reason. What it must never do is arrive as an empty `Scanned` — "no personal data
    /// found" would then be indistinguishable from "nothing looked", and that is how
    /// personal data reaches an agent.
    ///
    /// Deliberately agnostic about whether the model is installed. An earlier version of
    /// this test asserted failure, which passed only on machines with no checkpoint and
    /// started failing the moment one was downloaded.
    #[test]
    fn a_sanitize_job_either_finds_something_or_says_why() {
        let r = Runner::new();
        let h = r.sanitize("scrivi a mario@example.com per il bonifico".into());
        assert_eq!(h.kind, Kind::Sanitize);

        let events = drain_until_done(&r);
        let scanned = events.iter().find_map(|e| match e {
            Event::Scanned { scan, .. } => Some(scan),
            _ => None,
        });
        let failed = events.iter().find_map(|e| match e {
            Event::Failed { error, .. } => Some(error.clone()),
            _ => None,
        });

        match (scanned, failed) {
            (Some(scan), None) => assert!(
                !scan.plan.items.is_empty(),
                "the model ran but reported nothing in a prompt containing an address"
            ),
            (None, Some(why)) => assert!(!why.is_empty(), "a failure needs a reason"),
            (a, b) => panic!("expected exactly one outcome, got {a:?} / {b:?}"),
        }
    }

    #[test]
    fn a_download_job_states_its_size_up_front() {
        // The label is what the status bar shows before any bytes move, so the user knows
        // what they just agreed to.
        let r = Runner::new();
        let one = r.fetch_models(vec![crate::models::LLM_TERNARY]);
        one.cancel();
        assert!(
            one.label.contains("7.17 GB") && one.label.contains("Bonsai"),
            "got {}",
            one.label
        );
    }

    /// A raised cancel flag must stop a download before it opens a connection.
    ///
    /// Only meaningful with `candle`: without it there is no download path to cancel.
    #[test]
    fn a_raised_cancel_flag_stops_a_download_before_it_starts() {
        let cancel = Arc::new(AtomicBool::new(true));
        let started = std::time::Instant::now();
        let err = crate::models::download(&crate::models::LLM_TERNARY, &cancel).unwrap_err();
        assert!(err.contains("cancelled"), "got {err}");
        // Generous on purpose: the point is that a 7.17 GB transfer did not start, not that
        // the return was instant. A tight bound here only buys flakiness on a busy machine.
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "took {:?} — it should not have reached the network",
            started.elapsed()
        );
    }

    /// The summary names the build in use, so a user reading the status bar can tell which
    /// of the two they are actually running. Two are offered and one is run; a count over the
    /// catalogue would report the normal state of affairs as a shortfall.
    #[test]
    fn the_cache_summary_names_the_selected_build() {
        use crate::models::{LLM_1BIT, LLM_TERNARY, Phase};

        // Written against given values rather than the process-wide board, which other tests
        // are concurrently probing.
        let loaded = summarize_cache(LLM_TERNARY, Phase::Ready);
        assert!(loaded.contains("ternary"), "got {loaded}");
        assert!(loaded.contains("loaded"), "got {loaded}");

        // The other build reads differently — otherwise the readout cannot do its one job.
        let small = summarize_cache(LLM_1BIT, Phase::Ready);
        assert_ne!(small, loaded);
        assert!(small.contains("1-bit"), "got {small}");

        // A failure keeps its reason rather than becoming a count of zero.
        let failed = summarize_cache(LLM_TERNARY, Phase::Failed("no network".into()));
        assert!(failed.contains("no network"), "got {failed}");

        let absent = summarize_cache(LLM_1BIT, Phase::Absent);
        assert!(absent.contains("not downloaded"), "got {absent}");
    }

    #[test]
    fn cancel_flag_is_observable() {
        let h = Handle {
            id: JobId(1),
            kind: Kind::Hint,
            label: "x".into(),
            cancel: Arc::new(AtomicBool::new(false)),
        };
        assert!(!h.is_cancelled());
        h.cancel();
        assert!(h.is_cancelled());
        // Clones share the flag, so the UI can cancel through any copy.
        let c = h.clone();
        assert!(c.is_cancelled());
    }

    #[test]
    fn terminal_events_are_classified() {
        let id = JobId(7);
        assert!(
            !Event::Chunk {
                id,
                text: "x".into()
            }
            .is_terminal()
        );
        assert!(
            !Event::Note {
                id,
                text: "x".into()
            }
            .is_terminal()
        );
        assert!(
            !Event::Started {
                id,
                kind: Kind::Hint,
                label: "x".into()
            }
            .is_terminal()
        );
        assert!(
            Event::Failed {
                id,
                kind: Kind::Hint,
                error: "x".into()
            }
            .is_terminal()
        );
        assert!(
            Event::Cancelled {
                id,
                kind: Kind::Hint
            }
            .is_terminal()
        );
    }

    #[test]
    fn drain_is_non_blocking_when_idle() {
        let r = Runner::new();
        let started = std::time::Instant::now();
        assert!(r.drain().is_empty());
        assert!(
            started.elapsed() < Duration::from_millis(50),
            "drain must not block"
        );
    }

    #[test]
    fn labels_flag_unselectable_effort() {
        assert_eq!(effort_label(Effort::High, true), "effort high");
        let predicted = effort_label(Effort::High, false);
        assert!(predicted.contains("~high"), "got {predicted}");
        assert!(predicted.contains("agent's own"), "got {predicted}");
    }

    #[test]
    fn prompt_delivery_is_described() {
        assert_eq!(prompt_delivery(PromptVia::Arg), "as an argument");
        assert_eq!(prompt_delivery(PromptVia::Stdin), "on stdin");
    }
}
