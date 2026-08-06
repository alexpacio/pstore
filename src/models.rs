//! The local checkpoint pstore runs, and its live download/load state.
//!
//! Everything pstore infers for itself — the capability and difficulty reads behind the
//! ranking, and the PII scan behind the sanitiser — comes from **one** model, run on this
//! machine by a `llama-cli` subprocess. Nothing here calls an inference API; the only
//! network traffic is a one-off weight download from Hugging Face, which the user starts
//! from the Models window.
//!
//! Two things live here:
//!
//! * [`ALL`] — the catalogue: which repository the checkpoint comes from, which files it
//!   needs, and how big it is, so the UI can say what a download will cost *before* it
//!   starts, and
//! * the **status board** ([`phase`], [`set`], [`snapshot`]) — a small shared table the
//!   worker threads write and the UI polls once per frame. Fetching weights happens deep
//!   inside a worker thread; without somewhere neutral to record "downloading, 40% of
//!   3.8 GB" the GUI could only show a spinner.
//!
//! The `llama-cli` binary that runs these weights is provisioned separately — see
//! [`crate::runtime`], which reports into this same board.

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

/// Bonsai 27B, 1-bit: the one model behind routing, difficulty and PII.
///
/// PrismML's binary-weight build of Qwen3.6-27B. 3.8 GB of weights holding a 27B model,
/// which is why it replaced three purpose-built encoders totalling 2.8 GB and still costs
/// less memory than the ternary build of the same model (7.17 GB). The trade is ~89.5% of
/// FP16 quality against the ternary build's 94.6% — worth it here, where every task
/// (rating six dimensions 0–1, extracting typed spans) sits far below what a 27B can do.
///
/// Only the language weights are listed. The repository also ships an `mmproj` vision
/// tower and a `dspark` speculative-decoding drafter; pstore sends no images and gains
/// nothing from a drafter on one-shot invocations, so fetching either would be paying
/// gigabytes for capacity that never loads.
pub const LLM: Checkpoint = Checkpoint {
    id: "llm",
    title: "Bonsai 27B (1-bit)",
    purpose: "scores capability and difficulty, and finds personal data",
    repo: "prism-ml/Bonsai-27B-gguf",
    files: &["Bonsai-27B-Q1_0.gguf"],
    bytes: 3_800_000_000,
    license: "Apache-2.0",
};

/// Every checkpoint, in the order the Models window lists them.
pub const ALL: [Checkpoint; 1] = [LLM];

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

/// Whether this build can run the checkpoint at all.
///
/// False in a `--no-default-features` build, which compiles out the Hub client and the
/// runtime provisioner. The features that need the model — scoring, sanitising, shrink —
/// are then disabled rather than degraded: there is no second implementation behind them.
/// The distinction matters to the UI, which should say the support is absent from the
/// binary rather than imply a download would help.
pub const LOCAL_INFERENCE: bool = cfg!(feature = "local-llm");

/// What to say when there is nothing to run the weights with.
pub const NO_LOCAL_INFERENCE: &str =
    "this build has no local inference support (compiled without the `local-llm` feature)";

/// What the cache says about `c`.
#[cfg(feature = "local-llm")]
fn cache_phase(c: &Checkpoint) -> Phase {
    if is_cached(c) {
        Phase::Cached
    } else {
        Phase::Absent
    }
}

/// Without the Hub client there is nothing to ask, so "not downloaded" would be a claim
/// this build cannot make — the weights may well be on disk from another build. Report the
/// thing that is actually true.
#[cfg(not(feature = "local-llm"))]
fn cache_phase(_c: &Checkpoint) -> Phase {
    Phase::Failed(NO_LOCAL_INFERENCE.to_string())
}

/// Whether every file of `c` is already in the Hugging Face cache.
#[cfg(feature = "local-llm")]
pub fn is_cached(c: &Checkpoint) -> bool {
    c.files
        .iter()
        .all(|f| crate::router::hub::cached(c.repo, f).is_ok())
}

/// No Hub client, so nothing can be confirmed present.
#[cfg(not(feature = "local-llm"))]
pub fn is_cached(_c: &Checkpoint) -> bool {
    false
}

/// Download every file of `c`, keeping its phase up to date as bytes arrive.
///
/// Reuses anything already cached, so calling it on a complete checkpoint is quick and
/// harmless. Returns the reason on failure, which the Models window shows verbatim.
///
/// `cancel` is honoured **mid-transfer**, between one-megabyte chunks, so stopping a 3.8 GB
/// download is immediate. The partial file is kept and the next attempt resumes from it.
#[cfg(feature = "local-llm")]
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
        if let Err(e) = crate::router::hub::fetch_reporting(c.repo, file, progress, cancel) {
            set(c.id, Phase::Failed(e.clone()));
            return Err(e);
        }
    }
    set(c.id, Phase::Cached);
    Ok(())
}

/// Without the `candle` feature there is nothing to run the weights with, so there is
/// nothing worth downloading either.
#[cfg(not(feature = "local-llm"))]
pub fn download(c: &Checkpoint, _cancel: &std::sync::atomic::AtomicBool) -> Result<(), String> {
    set(c.id, Phase::Failed(NO_LOCAL_INFERENCE.to_string()));
    Err(NO_LOCAL_INFERENCE.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deliberately shape-agnostic about *which* files a checkpoint needs. It used to
    /// require the transformers trio (`config.json`/`tokenizer.json`/`model.safetensors`);
    /// a GGUF checkpoint is one self-contained file, and a future one may be neither. What
    /// has to hold is that the catalogue can be acted on: a real repo, some files, a size
    /// to show before the download starts.
    #[test]
    fn catalogue_is_complete_and_distinct() {
        for c in ALL.iter() {
            assert!(c.repo.contains('/'), "{} needs an owner/name repo", c.id);
            assert!(!c.files.is_empty(), "{} must list the files it needs", c.id);
            assert!(c.bytes > 1_000_000, "{} size looks wrong", c.id);
            assert!(!c.purpose.is_empty() && !c.license.is_empty());
        }
        let ids: Vec<_> = ALL.iter().map(|c| c.id).collect();
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), ids.len(), "checkpoint ids must be unique");
    }

    /// A wrong repo id should cost a few KB, not most of a gigabyte, so [`download`]
    /// fetches metadata first and weights last. Checked by extension rather than by an
    /// exact filename: the weights used to be `model.safetensors` and are now a `.gguf`.
    #[test]
    fn weights_are_fetched_last() {
        const WEIGHTS: [&str; 2] = [".gguf", ".safetensors"];
        let is_weights = |f: &str| WEIGHTS.iter().any(|ext| f.ends_with(ext));

        for c in ALL.iter() {
            let last = c.files.last().expect("checked non-empty above");
            assert!(
                is_weights(last),
                "{}: last file should be the weights, got {last:?}",
                c.id
            );
            let earlier = &c.files[..c.files.len() - 1];
            assert!(
                !earlier.iter().copied().any(is_weights),
                "{}: downloads weights before its metadata — {:?}",
                c.id,
                c.files
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
        set(LLM.id, Phase::Loading);
        assert_eq!(phase(LLM.id), Phase::Loading);
        assert!(any_busy());
        set(LLM.id, Phase::Ready);
        assert_eq!(phase(LLM.id), Phase::Ready);

        // An unknown id is dropped rather than added, so a typo cannot invent a row.
        set("not-a-checkpoint", Phase::Ready);
        assert_eq!(phase("not-a-checkpoint"), Phase::Unknown);

        let snap = snapshot();
        assert_eq!(snap.len(), ALL.len());
        assert_eq!(snap[0].0.id, LLM.id);

        // Hand the row back to the cache's own answer.
        set(LLM.id, Phase::Absent);
        probe_cache();
        assert_ne!(phase(LLM.id), Phase::Unknown);
    }

    #[test]
    fn sizes_read_in_familiar_units() {
        assert_eq!(bytes_label(500), "500 B");
        assert_eq!(bytes_label(795_276_408), "795 MB");
        assert_eq!(bytes_label(LLM.bytes), "3.80 GB");
    }
}
