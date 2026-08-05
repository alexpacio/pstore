//! The local checkpoints pstore runs, and their live download/load state.
//!
//! Every model pstore uses itself — the two classifiers behind the ranking, and the PII
//! tagger behind the sanitiser — runs **in this process, on this machine**. Nothing here
//! calls an inference API; the only network traffic is a one-off weight download from
//! Hugging Face, which the user starts by hand from the Models window.
//!
//! Two things live here:
//!
//! * [`ALL`] — the catalogue: which repository each checkpoint comes from, which files it
//!   needs, and how big it is, so the UI can say what a download will cost *before* it
//!   starts, and
//! * the **status board** ([`phase`], [`set`], [`snapshot`]) — a small shared table the
//!   worker threads write and the UI polls once per frame. Loading a checkpoint happens
//!   deep inside a classifier on a worker thread; without somewhere neutral to record
//!   "downloading, 40% of 795 MB" the GUI could only show a spinner.

use std::sync::{Mutex, OnceLock};

/// One local checkpoint pstore can run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Checkpoint {
    /// Stable key, used by the status board and the UI.
    pub id: &'static str,
    /// Short name for the Models window.
    pub title: &'static str,
    /// What pstore uses it for, in one line.
    pub purpose: &'static str,
    /// Hugging Face repository holding it.
    pub repo: &'static str,
    /// Files that must be present before it can load, weights last.
    ///
    /// Ordered cheapest first so a failure (a moved repo, no network) surfaces after a
    /// few kilobytes rather than most of a gigabyte.
    pub files: &'static [&'static str],
    /// Total download size in bytes, as published by the Hub.
    pub bytes: u64,
    /// Licence of the weights, shown before the download starts.
    pub license: &'static str,
}

impl Checkpoint {
    /// Size as a human-readable string, for buttons and labels.
    pub fn size_label(&self) -> String {
        bytes_label(self.bytes)
    }
}

/// Brick's capability classifier: which of the six dimensions a prompt draws on.
pub const CAPABILITY: Checkpoint = Checkpoint {
    id: "capability",
    title: "Capability classifier",
    purpose: "scores the six capability dimensions a prompt draws on",
    repo: "regolo/brick-modernbert-capability-classifier",
    files: &["config.json", "tokenizer.json", "model.safetensors"],
    bytes: 795_276_408,
    license: "Apache-2.0",
};

/// NVIDIA's prompt task-and-complexity classifier: how hard the prompt is.
pub const DIFFICULTY: Checkpoint = Checkpoint {
    id: "difficulty",
    title: "Difficulty classifier",
    purpose: "rates prompt complexity, which sets the model tier and effort",
    repo: "nvidia/prompt-task-and-complexity-classifier",
    files: &["tokenizer.json", "model.safetensors"],
    bytes: 744_107_008,
    license: "NVIDIA Open Model License",
};

/// The rizzo-pii tagger: finds personal data in the prompt.
pub const PII: Checkpoint = Checkpoint {
    id: "pii",
    title: "PII tagger",
    purpose: "finds names, addresses and identifiers so they can be masked",
    repo: "rizzoaiacademy/rizzo-pii-0.3B",
    files: &["config.json", "tokenizer.json", "model.safetensors"],
    bytes: 1_264_633_910,
    license: "Apache-2.0",
};

/// Every checkpoint, in the order the Models window lists them.
pub const ALL: [Checkpoint; 3] = [CAPABILITY, DIFFICULTY, PII];

/// Where a checkpoint currently is.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum Phase {
    /// The cache has not been looked at yet.
    #[default]
    Unknown,
    /// Not downloaded.
    Absent,
    /// Downloaded, not yet loaded into memory.
    Cached,
    /// Downloading now.
    Fetching {
        /// File being transferred.
        file: &'static str,
        /// Bytes of `file` received.
        done: u64,
        /// Size of `file`, or 0 while unknown.
        total: u64,
        /// Files already finished, for a whole-checkpoint estimate.
        files_done: usize,
    },
    /// Weights are on disk and being built into a model.
    Loading,
    /// Loaded and answering.
    Ready,
    /// Something went wrong; the reason is shown in the Models window.
    Failed(String),
}

impl Phase {
    /// Whether work is happening right now, so the UI keeps repainting.
    pub fn is_busy(&self) -> bool {
        matches!(self, Phase::Fetching { .. } | Phase::Loading)
    }

    /// Whether the weights are on disk, loaded or not.
    pub fn is_downloaded(&self) -> bool {
        matches!(self, Phase::Cached | Phase::Loading | Phase::Ready)
    }

    /// One-line description for the Models window.
    pub fn label(&self) -> String {
        match self {
            Phase::Unknown => "checking…".into(),
            Phase::Absent => "not downloaded".into(),
            Phase::Cached => "downloaded".into(),
            Phase::Fetching {
                file,
                done,
                total,
                files_done,
            } => {
                let of = if *total > 0 {
                    format!(" of {}", bytes_label(*total))
                } else {
                    String::new()
                };
                format!(
                    "downloading {file} ({}{of}, {} file(s) done)",
                    bytes_label(*done),
                    files_done
                )
            }
            Phase::Loading => "loading into memory…".into(),
            Phase::Ready => "loaded".into(),
            Phase::Failed(why) => format!("failed — {why}"),
        }
    }

    /// Fraction of the current file transferred, when that is known.
    pub fn fraction(&self) -> Option<f32> {
        match self {
            Phase::Fetching {
                done,
                total: t @ 1..,
                ..
            } => Some((*done as f32 / *t as f32).clamp(0.0, 1.0)),
            _ => None,
        }
    }
}

fn board() -> &'static Mutex<Vec<(&'static str, Phase)>> {
    static BOARD: OnceLock<Mutex<Vec<(&'static str, Phase)>>> = OnceLock::new();
    BOARD.get_or_init(|| Mutex::new(ALL.iter().map(|c| (c.id, Phase::Unknown)).collect()))
}

/// A poisoned board means a worker panicked mid-update. Recovering the guard keeps the
/// app usable — the worst case is one stale row, which the next update corrects.
fn locked() -> std::sync::MutexGuard<'static, Vec<(&'static str, Phase)>> {
    match board().lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// Record where `id` has got to.
pub fn set(id: &'static str, phase: Phase) {
    let mut b = locked();
    if let Some(slot) = b.iter_mut().find(|(k, _)| *k == id) {
        slot.1 = phase;
    }
}

/// The current phase of one checkpoint.
pub fn phase(id: &str) -> Phase {
    locked()
        .iter()
        .find(|(k, _)| *k == id)
        .map(|(_, p)| p.clone())
        .unwrap_or_default()
}

/// Every checkpoint with its phase, in [`ALL`] order.
pub fn snapshot() -> Vec<(Checkpoint, Phase)> {
    ALL.iter().map(|c| (*c, phase(c.id))).collect()
}

/// Whether any checkpoint is downloading or loading.
pub fn any_busy() -> bool {
    locked().iter().any(|(_, p)| p.is_busy())
}

/// Round-trip a byte count into something readable.
pub fn bytes_label(bytes: u64) -> String {
    const MB: f64 = 1_000_000.0;
    const GB: f64 = 1_000_000_000.0;
    let b = bytes as f64;
    if b >= GB {
        format!("{:.2} GB", b / GB)
    } else if b >= MB {
        format!("{:.0} MB", b / MB)
    } else {
        format!("{bytes} B")
    }
}

/// Look at the on-disk cache and record, for every checkpoint, whether its files are
/// already there. Touches the network never, the filesystem once per file.
///
/// Phases that describe live work ([`Phase::Fetching`], [`Phase::Loading`]) or a loaded
/// model ([`Phase::Ready`]) are left alone: the cache says nothing they don't already.
pub fn probe_cache() {
    for c in ALL.iter() {
        if matches!(
            phase(c.id),
            Phase::Fetching { .. } | Phase::Loading | Phase::Ready
        ) {
            continue;
        }
        set(c.id, cache_phase(c));
    }
}

/// Whether this build can run the checkpoints at all.
///
/// False in a `--no-default-features` build, which compiles out Candle, the tokenizers and
/// the Hub client. Everything still works — routing uses the built-in estimate and
/// sanitising the checksum-backed patterns — but the local models are not merely missing,
/// they are absent from the binary, and the UI should say that rather than imply a download
/// would help.
pub const LOCAL_INFERENCE: bool = cfg!(feature = "candle");

/// What to say when there is nothing to run the weights with.
pub const NO_LOCAL_INFERENCE: &str =
    "this build has no local inference support (compiled without the `candle` feature)";

/// What the cache says about `c`.
#[cfg(feature = "candle")]
fn cache_phase(c: &Checkpoint) -> Phase {
    if is_cached(c) {
        Phase::Cached
    } else {
        Phase::Absent
    }
}

/// Without Candle there is no Hub client to ask, so "not downloaded" would be a claim this
/// build cannot make — the weights may well be on disk from another build. Report the thing
/// that is actually true.
#[cfg(not(feature = "candle"))]
fn cache_phase(_c: &Checkpoint) -> Phase {
    Phase::Failed(NO_LOCAL_INFERENCE.to_string())
}

/// Whether every file of `c` is already in the Hugging Face cache.
#[cfg(feature = "candle")]
pub fn is_cached(c: &Checkpoint) -> bool {
    c.files
        .iter()
        .all(|f| crate::router::hub::cached(c.repo, f).is_ok())
}

/// No Hub client, so nothing can be confirmed present.
#[cfg(not(feature = "candle"))]
pub fn is_cached(_c: &Checkpoint) -> bool {
    false
}

/// Download every file of `c`, keeping its phase up to date as bytes arrive.
///
/// Reuses anything already cached, so calling it on a complete checkpoint is quick and
/// harmless. Returns the reason on failure, which the Models window shows verbatim.
///
/// `cancel` is honoured **between files**, not mid-transfer: the Hub client has no abort
/// hook, so asking to stop during the weights file waits for that file to land. Nothing is
/// wasted when it does — the cache keys on content, so the next attempt reuses it — but the
/// button cannot be instant, and the UI says so rather than appearing to hang.
#[cfg(feature = "candle")]
pub fn download(c: &Checkpoint, cancel: &std::sync::atomic::AtomicBool) -> Result<(), String> {
    use std::sync::atomic::Ordering;

    for (i, file) in c.files.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            // Say what the cache actually holds now rather than assuming the worst: the
            // metadata files may well have arrived before the stop.
            // Re-probe only if this call actually fetched something: the metadata files may
            // have arrived before the stop. Cancelled before the first file, nothing has
            // changed, so leave the phase — and the filesystem — alone.
            if i > 0 {
                set(c.id, cache_phase(c));
            }
            return Err("cancelled".into());
        }
        set(
            c.id,
            Phase::Fetching {
                file,
                done: 0,
                total: 0,
                files_done: i,
            },
        );
        let id = c.id;
        let progress = std::sync::Arc::new(move |done: u64, total: u64| {
            set(
                id,
                Phase::Fetching {
                    file,
                    done,
                    total,
                    files_done: i,
                },
            );
        });
        if let Err(e) = crate::router::hub::fetch_reporting(c.repo, file, progress) {
            set(c.id, Phase::Failed(e.clone()));
            return Err(e);
        }
    }
    set(c.id, Phase::Cached);
    Ok(())
}

/// Without the `candle` feature there is nothing to run the weights with, so there is
/// nothing worth downloading either.
#[cfg(not(feature = "candle"))]
pub fn download(c: &Checkpoint, _cancel: &std::sync::atomic::AtomicBool) -> Result<(), String> {
    set(c.id, Phase::Failed(NO_LOCAL_INFERENCE.to_string()));
    Err(NO_LOCAL_INFERENCE.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalogue_is_complete_and_distinct() {
        for c in ALL.iter() {
            assert!(c.repo.contains('/'), "{} needs an owner/name repo", c.id);
            assert!(
                c.files.contains(&"model.safetensors"),
                "{} must list its weights",
                c.id
            );
            assert!(
                c.files.contains(&"tokenizer.json"),
                "{} must list its tokenizer",
                c.id
            );
            assert!(c.bytes > 1_000_000, "{} size looks wrong", c.id);
            assert!(!c.purpose.is_empty() && !c.license.is_empty());
        }
        let ids: Vec<_> = ALL.iter().map(|c| c.id).collect();
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), ids.len(), "checkpoint ids must be unique");
    }

    #[test]
    fn weights_are_fetched_last() {
        // A wrong repo id should cost a few KB, not most of a gigabyte.
        for c in ALL.iter() {
            assert_eq!(
                c.files.last(),
                Some(&"model.safetensors"),
                "{} downloads its weights before its metadata",
                c.id
            );
        }
    }

    #[test]
    fn phases_describe_themselves() {
        assert_eq!(Phase::Absent.label(), "not downloaded");
        assert!(
            Phase::Failed("no network".into())
                .label()
                .contains("no network")
        );
        let fetching = Phase::Fetching {
            file: "model.safetensors",
            done: 400_000_000,
            total: 800_000_000,
            files_done: 2,
        };
        assert!(fetching.is_busy());
        assert_eq!(fetching.fraction(), Some(0.5));
        assert!(fetching.label().contains("model.safetensors"));
        assert!(fetching.label().contains("800 MB"));

        assert!(Phase::Loading.is_busy());
        assert!(!Phase::Ready.is_busy());
        assert!(Phase::Ready.is_downloaded());
        assert!(Phase::Cached.is_downloaded());
        assert!(!Phase::Absent.is_downloaded());
        // Progress with no announced total must not divide by zero.
        assert_eq!(
            Phase::Fetching {
                file: "x",
                done: 5,
                total: 0,
                files_done: 0
            }
            .fraction(),
            None
        );
    }

    /// The board is process-wide, so this test picks phases [`probe_cache`] leaves alone —
    /// it skips rows that describe live work — and hands the row back afterwards rather
    /// than resetting it to `Unknown`, which would undo a probe another test is asserting
    /// on. Tests run in parallel; shared state has to be shared carefully.
    #[test]
    fn board_round_trips_and_ignores_unknown_ids() {
        set(CAPABILITY.id, Phase::Loading);
        assert_eq!(phase(CAPABILITY.id), Phase::Loading);
        assert!(any_busy());
        set(CAPABILITY.id, Phase::Ready);
        assert_eq!(phase(CAPABILITY.id), Phase::Ready);

        // An unknown id is dropped rather than added, so a typo cannot invent a row.
        set("not-a-checkpoint", Phase::Ready);
        assert_eq!(phase("not-a-checkpoint"), Phase::Unknown);

        let snap = snapshot();
        assert_eq!(snap.len(), ALL.len());
        assert_eq!(snap[0].0.id, CAPABILITY.id);

        // Hand the row back to the cache's own answer.
        set(CAPABILITY.id, Phase::Absent);
        probe_cache();
        assert_ne!(phase(CAPABILITY.id), Phase::Unknown);
    }

    #[test]
    fn sizes_read_in_familiar_units() {
        assert_eq!(bytes_label(500), "500 B");
        assert_eq!(bytes_label(795_276_408), "795 MB");
        assert_eq!(bytes_label(1_264_633_910), "1.26 GB");
    }
}
