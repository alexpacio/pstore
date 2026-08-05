//! Scoring a prompt, then scoring every model and effort available for it.
//!
//! Two signals come out of the classifiers, mirroring Brick's design:
//!
//! * a **capability vector** — which of the six dimensions the prompt draws on
//!   (multi-label sigmoid, so the components do not sum to 1), and
//! * a **complexity** label — how hard it leans on them.
//!
//! Each comes from its own small encoder model, and **both run in this process** — see
//! [`capability`] and [`difficulty`]. Neither sends the prompt anywhere. When a model is
//! missing or fails to load, that half of the reading degrades to [`heuristic`] and the
//! [`Reading`] says so rather than pretending.
//!
//! [`scoring`] turns those into a score for every (agent, model, effort) combination.

pub mod capability;
pub mod device;
pub mod difficulty;
pub mod heuristic;
pub mod hub;
#[cfg(feature = "candle")]
pub mod pooling;
pub mod scoring;

use std::fmt;

use crate::agents::registry::{DIMS, Vec6};

/// Multi-label capability scores in [`DIMS`] order, each in `[0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Capability {
    /// Per-dimension score.
    pub scores: Vec6,
}

impl Capability {
    /// The dimension the prompt leans on hardest, with its score.
    pub fn dominant(&self) -> (&'static str, f32) {
        let (i, v) = self
            .scores
            .iter()
            .enumerate()
            .fold(
                (0usize, f32::MIN),
                |acc, (i, v)| if *v > acc.1 { (i, *v) } else { acc },
            );
        (DIMS[i], v)
    }

    /// Dimensions above `threshold`, strongest first, for the UI readout.
    pub fn notable(&self, threshold: f32) -> Vec<(&'static str, f32)> {
        let mut v: Vec<_> = DIMS
            .iter()
            .copied()
            .zip(self.scores)
            .filter(|(_, s)| *s >= threshold)
            .collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        v
    }
}

/// Prompt difficulty, as [`difficulty`] reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum Complexity {
    /// Straightforward; a light model at low effort will do.
    Easy,
    /// The common middle.
    #[default]
    Medium,
    /// Demands a strong model, and effort to match.
    Hard,
}

impl Complexity {
    /// The label string.
    pub fn label(self) -> &'static str {
        match self {
            Complexity::Easy => "easy",
            Complexity::Medium => "medium",
            Complexity::Hard => "hard",
        }
    }
}

impl fmt::Display for Complexity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Which implementation produced a reading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// Both local classifiers ran.
    Models,
    /// The capability model ran; difficulty came from the heuristic.
    CapabilityOnly,
    /// The difficulty model ran; the capability vector came from the heuristic.
    DifficultyOnly,
    /// Built-in surface-feature estimate for both signals.
    Heuristic,
}

impl Source {
    /// Whether at least one trained classifier contributed.
    pub fn uses_a_model(self) -> bool {
        !matches!(self, Source::Heuristic)
    }
}

impl fmt::Display for Source {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Source::Models => "local classifiers",
            Source::CapabilityOnly => "local capability + built-in difficulty",
            Source::DifficultyOnly => "local difficulty + built-in capability",
            Source::Heuristic => "built-in",
        })
    }
}

/// A complete reading of one prompt.
#[derive(Debug, Clone)]
pub struct Reading {
    /// Capability demand.
    pub capability: Capability,
    /// Difficulty.
    pub complexity: Complexity,
    /// Which implementation produced this.
    pub source: Source,
    /// How long classification took.
    pub elapsed: std::time::Duration,
    /// Why a classifier was not used, when one wasn't. Surfaced so a silent downgrade to
    /// the heuristic is visible rather than mysterious.
    pub fallback_reason: Option<String>,
    /// Confidence in the complexity label, when the classifier reports one.
    pub confidence: Option<f32>,
    /// The continuous complexity score behind the label, when a model produced it.
    pub difficulty_score: Option<f32>,
    /// Per-dimension difficulty detail, when a model produced it.
    pub difficulty_detail: Option<difficulty::Dimensions>,
    /// What kind of task the difficulty model thinks this is, and how sure it is.
    pub task: Option<(&'static str, f32)>,
}

impl Reading {
    /// A blank reading, filled in by the readers below.
    fn empty() -> Self {
        Self {
            capability: Capability { scores: [0.0; 6] },
            complexity: Complexity::default(),
            source: Source::Heuristic,
            elapsed: std::time::Duration::ZERO,
            fallback_reason: None,
            confidence: None,
            difficulty_score: None,
            difficulty_detail: None,
            task: None,
        }
    }
}

/// Classify with the built-in heuristic. Always available, never blocks.
pub fn read_heuristic(text: &str) -> Reading {
    let started = std::time::Instant::now();
    Reading {
        capability: heuristic::capability(text),
        complexity: heuristic::complexity(text),
        source: Source::Heuristic,
        elapsed: started.elapsed(),
        ..Reading::empty()
    }
}

/// Classify with the local models, falling back to the heuristic.
///
/// The first call loads the weights, so this blocks for a while — call it from a worker
/// thread. Every later call is two forward passes. Any failure degrades to
/// [`read_heuristic`] for the affected half, with the reason attached, rather than failing
/// the whole ranking.
///
/// Nothing is downloaded here: weights come from the Models window, so a first run scores
/// with the heuristic instead of stalling on a 1.5 GB transfer nobody asked for.
pub fn read_best(text: &str) -> Reading {
    #[cfg(feature = "candle")]
    {
        local::read(text)
    }
    #[cfg(not(feature = "candle"))]
    {
        let mut r = read_heuristic(text);
        r.fallback_reason = Some("built without the `candle` feature".into());
        r
    }
}

#[cfg(feature = "candle")]
mod local {
    use super::{Reading, Source, capability, device, difficulty, heuristic};
    use crate::models;
    use candle_core::Device;
    use std::sync::{Mutex, OnceLock};

    /// Whatever loaded successfully.
    ///
    /// The two classifiers are tracked separately on purpose: they are different
    /// checkpoints of different sizes, either can be absent, and losing a working one
    /// because its neighbour failed would be a self-inflicted downgrade.
    #[derive(Default)]
    struct Loaded {
        cap: Option<capability::Model>,
        cx: Option<difficulty::Model>,
        /// Why the capability model is absent, if it is.
        cap_error: Option<String>,
        /// Why the difficulty model is absent, if it is.
        cx_error: Option<String>,
    }

    enum State {
        /// Not attempted yet.
        Unloaded,
        /// Load attempted; some, all, or none of it succeeded.
        Attempted(Box<Loaded>),
    }

    fn state() -> &'static Mutex<State> {
        static STATE: OnceLock<Mutex<State>> = OnceLock::new();
        STATE.get_or_init(|| Mutex::new(State::Unloaded))
    }

    /// A poisoned lock means a previous classification panicked; recover the guard rather
    /// than taking the whole app down over a classifier.
    fn locked() -> std::sync::MutexGuard<'static, State> {
        match state().lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    /// Load a checkpoint, retrying on the CPU if the GPU refuses it.
    ///
    /// A missing Metal or CUDA kernel is a property of the build, not of the weights, and
    /// these models are small enough that CPU is a real answer rather than a token one.
    fn load_anywhere<T>(
        c: &'static models::Checkpoint,
        load: impl Fn(Device) -> Result<T, String>,
    ) -> Result<T, String> {
        if !models::is_cached(c) {
            let why = format!(
                "{} not downloaded — open the Models window to fetch it ({})",
                c.title,
                c.size_label()
            );
            models::set(c.id, models::Phase::Absent);
            return Err(why);
        }
        let (dev, backend) = device::pick();
        match load(dev) {
            Ok(m) => Ok(m),
            Err(gpu) if backend.is_gpu() => {
                load(Device::Cpu).map_err(|cpu| format!("on {backend}: {gpu}; and on CPU: {cpu}"))
            }
            Err(e) => Err(e),
        }
    }

    /// Load both classifiers if they have not been tried yet.
    fn ensure_loaded(guard: &mut State) {
        if matches!(*guard, State::Unloaded) {
            let (cap, cap_error) = match load_anywhere(&models::CAPABILITY, capability::Model::load)
            {
                Ok(m) => (Some(m), None),
                Err(e) => (None, Some(e)),
            };
            let (cx, cx_error) = match load_anywhere(&models::DIFFICULTY, difficulty::Model::load) {
                Ok(m) => (Some(m), None),
                Err(e) => (None, Some(e)),
            };
            *guard = State::Attempted(Box::new(Loaded {
                cap,
                cx,
                cap_error,
                cx_error,
            }));
        }
    }

    /// Load both classifiers now, without classifying anything.
    ///
    /// Called from the Models window so "loading into memory…" is a step the user can
    /// watch and, if it fails, read the reason for — rather than a surprise pause the
    /// first time they score a prompt.
    pub fn preload() -> Result<(), String> {
        let mut guard = locked();
        ensure_loaded(&mut guard);
        match &*guard {
            State::Attempted(l) => match (&l.cap_error, &l.cx_error) {
                (None, None) => Ok(()),
                (Some(a), Some(b)) => Err(format!("capability: {a}; difficulty: {b}")),
                (Some(a), None) => Err(format!("capability: {a}")),
                (None, Some(b)) => Err(format!("difficulty: {b}")),
            },
            State::Unloaded => Err("classifiers not loaded".into()),
        }
    }

    /// Classify, loading the models on first use.
    pub fn read(text: &str) -> Reading {
        let started = std::time::Instant::now();
        let mut guard = locked();
        ensure_loaded(&mut guard);

        let State::Attempted(loaded) = &*guard else {
            let mut r = super::read_heuristic(text);
            r.fallback_reason = Some("classifiers not loaded".into());
            return r;
        };

        // Capability: the model if available, else the heuristic.
        let (cap_scores, cap_reason) = match loaded.cap.as_ref().map(|m| m.classify(text)) {
            Some(Ok(c)) => (c, None),
            Some(Err(e)) => (heuristic::capability(text), Some(e)),
            None => (
                heuristic::capability(text),
                loaded
                    .cap_error
                    .clone()
                    .or(Some("capability model unavailable".into())),
            ),
        };

        // Difficulty: the model if available, else the heuristic.
        let (verdict, cx_reason) = match loaded.cx.as_ref().map(|m| m.classify(text)) {
            Some(Ok(v)) => (Some(v), None),
            Some(Err(e)) => (None, Some(e)),
            None => (
                None,
                loaded
                    .cx_error
                    .clone()
                    .or(Some("difficulty model unavailable".into())),
            ),
        };

        let source = match (cap_reason.is_none(), cx_reason.is_none()) {
            (true, true) => Source::Models,
            (true, false) => Source::CapabilityOnly,
            (false, true) => Source::DifficultyOnly,
            (false, false) => Source::Heuristic,
        };
        // Report only what actually degraded, so the message stays actionable.
        let fallback_reason = match (&cap_reason, &cx_reason) {
            (None, None) => None,
            (Some(a), Some(b)) if a == b => Some(a.clone()),
            (Some(a), Some(b)) => Some(format!("capability: {a}; difficulty: {b}")),
            (Some(a), None) => Some(format!("capability: {a}")),
            (None, Some(b)) => Some(format!("difficulty: {b}")),
        };

        Reading {
            capability: cap_scores,
            complexity: verdict
                .map(|v| v.complexity)
                .unwrap_or_else(|| heuristic::complexity(text)),
            source,
            elapsed: started.elapsed(),
            fallback_reason,
            confidence: verdict.map(|v| v.confidence),
            difficulty_score: verdict.map(|v| v.score),
            difficulty_detail: verdict.map(|v| v.dimensions),
            task: verdict.map(|v| v.task),
        }
    }

    /// Discard any loaded models and any remembered failure, so the next
    /// classification retries from scratch.
    pub fn reset() {
        *locked() = State::Unloaded;
        for c in models::ALL.iter() {
            if matches!(models::phase(c.id), models::Phase::Ready) {
                models::set(c.id, models::Phase::Cached);
            }
        }
    }
}

/// Forget any loaded classifiers and retry loading on the next classification.
pub fn reset_classifiers() {
    #[cfg(feature = "candle")]
    local::reset();
}

/// Load the classifiers now rather than on the next classification.
///
/// Returns the reason when a checkpoint could not be loaded, so the Models window can show
/// it. Blocking; call from a worker thread.
pub fn preload_classifiers() -> Result<(), String> {
    #[cfg(feature = "candle")]
    {
        local::preload()
    }
    #[cfg(not(feature = "candle"))]
    {
        Err("built without the `candle` feature".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// End-to-end against the real checkpoints. Ignored by default, and it only proves
    /// anything once the weights are downloaded — run
    /// `cargo test -- --ignored classifiers --nocapture` after fetching them from the
    /// Models window (or let this test's first call download them itself, ~1.5 GB).
    #[test]
    #[ignore = "needs ~1.5 GB of downloaded model weights"]
    #[cfg(feature = "candle")]
    fn local_classifiers_load_and_classify() {
        for c in crate::models::ALL.iter().take(2) {
            crate::models::download(c, &std::sync::atomic::AtomicBool::new(false))
                .unwrap_or_else(|e| panic!("fetching {}: {e}", c.title));
        }

        let hard = read_best(
            "Refactor the authentication layer across src/auth/mod.rs, \
             src/auth/session.rs and src/api/routes.rs without breaking backwards \
             compatibility, and fix the race condition in the token refresh.",
        );

        // Printed so `--ignored --nocapture` doubles as a diagnostic of what the
        // status bar will actually say.
        eprintln!(
            "source={} complexity={} score={:?} task={:?} confidence={:?} elapsed={:?}\n\
             reason={:?}\nscores={:?}\ndetail={:?}",
            hard.source,
            hard.complexity,
            hard.difficulty_score,
            hard.task,
            hard.confidence,
            hard.elapsed,
            hard.fallback_reason,
            hard.capability.scores,
            hard.difficulty_detail,
        );

        assert_eq!(
            hard.source,
            Source::Models,
            "both classifiers should have loaded ({:?})",
            hard.fallback_reason
        );

        // The label-order gotcha, checked against the live checkpoint: a code-heavy
        // prompt must come back dominated by `coding`, not by whichever dimension
        // happens to sit at index 0 in the file.
        assert_eq!(
            hard.capability.dominant().0,
            "coding",
            "capability vector looks permuted: {:?}",
            hard.capability.scores
        );
        // Multi-label sigmoid, not softmax: several dimensions can be high at once.
        assert!(
            hard.capability.scores.iter().sum::<f32>() > 1.0,
            "scores look softmaxed (sum {:.3}), which would distort every ranking",
            hard.capability.scores.iter().sum::<f32>()
        );
        // The difficulty model recognises what kind of prompt this is.
        assert_eq!(hard.task.map(|t| t.0), Some("Code Generation"));
        // And a three-file refactor with a race condition in it is not an easy prompt.
        // This is the end-to-end check on the routing calibration; the per-dimension
        // version lives in `difficulty::tests`.
        assert_eq!(
            hard.complexity,
            Complexity::Hard,
            "a multi-file refactor scored {:?}",
            hard.difficulty_score
        );

        // A trivial prompt must not outrank a multi-file refactor in difficulty.
        let easy = read_best("fix this typo");
        assert!(
            hard.complexity >= easy.complexity,
            "a refactor ranked easier than a typo: {} ({:?}) vs {} ({:?})",
            hard.complexity,
            hard.difficulty_score,
            easy.complexity,
            easy.difficulty_score,
        );
        assert!(
            hard.difficulty_score > easy.difficulty_score,
            "difficulty scores did not separate: {:?} vs {:?}",
            hard.difficulty_score,
            easy.difficulty_score
        );

        // Models stay loaded, so a second call must be fast.
        let again = read_best("add a --verbose flag");
        assert!(
            again.elapsed < std::time::Duration::from_secs(5),
            "cached classification took {:?}",
            again.elapsed
        );
    }

    #[test]
    fn every_source_describes_itself_distinctly() {
        let all = [
            Source::Models,
            Source::CapabilityOnly,
            Source::DifficultyOnly,
            Source::Heuristic,
        ];
        let mut seen: Vec<String> = all.iter().map(|s| s.to_string()).collect();
        seen.sort();
        seen.dedup();
        assert_eq!(
            seen.len(),
            all.len(),
            "the status bar must never be ambiguous"
        );
        assert!(!Source::Heuristic.uses_a_model());
        assert!(Source::CapabilityOnly.uses_a_model());
        assert!(Source::DifficultyOnly.uses_a_model());
        assert!(Source::Models.uses_a_model());
    }

    #[test]
    fn complexity_labels_are_distinct_and_lowercase() {
        let labels =
            [Complexity::Easy, Complexity::Medium, Complexity::Hard].map(|c| c.label().to_string());
        assert_eq!(labels, ["easy", "medium", "hard"]);
        assert_eq!(
            Complexity::Hard.to_string(),
            "hard",
            "Display uses the label"
        );
    }

    #[test]
    fn complexity_orders_easy_to_hard() {
        assert!(Complexity::Easy < Complexity::Medium);
        assert!(Complexity::Medium < Complexity::Hard);
        assert_eq!(Complexity::default(), Complexity::Medium);
    }

    #[test]
    fn dominant_and_notable_read_the_vector() {
        // [instruction_following, coding, math, world, planning, creative]
        let c = Capability {
            scores: [0.4, 0.9, 0.1, 0.2, 0.6, 0.05],
        };
        assert_eq!(c.dominant(), ("coding", 0.9));

        let notable = c.notable(0.35);
        assert_eq!(
            notable.iter().map(|(d, _)| *d).collect::<Vec<_>>(),
            vec!["coding", "planning_agentic", "instruction_following"],
            "strongest first, below-threshold dropped"
        );
        assert!(c.notable(0.95).is_empty());
    }

    #[test]
    fn heuristic_reading_is_labelled_and_instant() {
        let r = read_heuristic("Refactor src/main.rs to extract the parser.");
        assert_eq!(r.source, Source::Heuristic);
        assert_eq!(r.source.to_string(), "built-in");
        assert!(r.elapsed < std::time::Duration::from_millis(100));
        assert!(r.capability.scores.iter().all(|v| (0.0..=1.0).contains(v)));
        // The heuristic has no model behind it, so it must not claim model-only detail.
        assert!(r.difficulty_score.is_none());
        assert!(r.difficulty_detail.is_none());
        assert!(r.task.is_none());
        assert!(r.confidence.is_none());
    }
}
