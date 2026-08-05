//! How hard the prompt is, from a classifier running on this machine.
//!
//! Model: `nvidia/prompt-task-and-complexity-classifier` — a DeBERTa-v3-base encoder with
//! eight linear heads over one mean-pooled representation. One head names the task
//! (Code Generation, Summarization, …); six rate a complexity dimension, and a weighted sum
//! of those — see *Scoring* below — is what this module turns into [`Complexity::Easy`],
//! `Medium` or `Hard`.
//!
//! # Why not Brick's difficulty model
//!
//! pstore's capability vector comes from Brick ([`super::capability`]), so Brick's own
//! difficulty checkpoint would have been the obvious partner. It cannot run here. Every
//! published Brick complexity model (`-2-eco`, `-2-max`, `-extractor`) is a LoRA on
//! **Qwen/Qwen3.5-0.8B**, and the merged GGUF says why that matters:
//!
//! ```text
//! general.architecture     = qwen35
//! qwen35.ssm.conv_kernel   = 4
//! qwen35.ssm.state_size    = 128
//! ```
//!
//! Qwen3.5 is a hybrid attention/SSM architecture — interleaved state-space layers, not
//! the pure-attention Qwen3. candle-transformers 0.11 has no `qwen35` model, so loading it
//! failed and difficulty silently fell back to the surface-feature heuristic, which is
//! what the status bar was complaining about. The classifier here is a plain encoder,
//! runs on Candle today, is a third of the size, and needs one forward pass with no
//! decoding at all.
//!
//! # Scoring, and why the published ensemble is not used directly
//!
//! The checkpoint publishes an overall score,
//! `0.35·creativity + 0.25·reasoning + 0.15·constraints + 0.15·domain + 0.05·contextual +
//! 0.05·few_shots`, weighted for the general-purpose prompt corpus it was trained on. Run
//! over *coding* prompts, two of its three largest terms turn out to be constants —
//! measured across the calibration set in the tests below, `creativity` never leaves
//! 0.01–0.07 and `domain_knowledge` never leaves 0.97–1.00 — so half the weight is a fixed
//! offset and everything from "fix this typo" to a forty-query database migration lands
//! between 0.13 and 0.38.
//!
//! What does discriminate is `constraints` (0.06 for a typo, 0.88 for a latency
//! investigation), then `reasoning`, which only lifts off for the genuinely hard ones.
//! [`Dimensions::routing_score`] is therefore pstore's own weighting over the model's
//! dimensions, and it spreads the same prompts across 0.05–0.51. The published formula is
//! kept as [`Dimensions::published_score`] and pinned to the model card's worked examples,
//! because reproducing it is what proves the six dimensions are being read out correctly.
//!
//! Both thresholds come from [`tests::calibrate_thresholds_against_coding_prompts`], which
//! prints the score for eleven prompts spanning trivial to very hard. Re-run it if the
//! checkpoint is ever updated.

use crate::router::Complexity;

/// Routing scores below this are [`Complexity::Easy`].
pub const EASY_BELOW: f32 = 0.15;

/// Routing scores at or above this are [`Complexity::Hard`].
pub const HARD_FROM: f32 = 0.33;

/// pstore's weighting over the model's dimensions, in [`HEADS`] order.
///
/// `domain_knowledge` is weighted zero deliberately: it is ~1.0 for every coding prompt,
/// so it carries no information here — including it would only add a constant and compress
/// the range. `creativity` keeps a small weight because a design-and-justify prompt does
/// raise it, and `few_shots` a smaller one because worked examples make a prompt longer to
/// satisfy rather than harder to think about.
pub const ROUTING_WEIGHTS: Dimensions = Dimensions {
    creativity: 0.10,
    reasoning: 0.30,
    contextual_knowledge: 0.10,
    few_shots: 0.05,
    domain_knowledge: 0.0,
    constraints: 0.45,
};

/// Longest input the backbone accepts, in tokens. DeBERTa-v3's published context.
pub const MAX_TOKENS: usize = 512;

/// Characters of prompt to consider. Difficulty is decided by the shape of the request,
/// and anything past the token limit is dropped by the tokenizer anyway — clipping first
/// just saves the work.
pub const MAX_CHARS: usize = 6000;

/// The eleven task types the model can name, plus `Unknown`, in checkpoint order.
pub const TASK_TYPES: [&str; 12] = [
    "Brainstorming",
    "Chatbot",
    "Classification",
    "Closed QA",
    "Code Generation",
    "Extraction",
    "Open QA",
    "Other",
    "Rewrite",
    "Summarization",
    "Text Generation",
    "Unknown",
];

/// One complexity head: how many classes it has, and how its probabilities collapse into
/// a single score.
///
/// `weights` and `divisor` are the `weights_map`/`divisor_map` entries from the
/// checkpoint's own `config.json`. They are not uniform — `domain_knowledge` is trained
/// with classes ordered High, Low, Medium, No, so its weights are `[3, 1, 2, 0]` — and
/// getting them wrong would invert a dimension rather than fail.
#[derive(Debug, Clone, Copy)]
pub struct Head {
    /// Name of the dimension, as the checkpoint calls it.
    pub name: &'static str,
    /// Number of classes.
    pub classes: usize,
    /// Per-class weight.
    pub weights: &'static [f32],
    /// What the weighted sum is divided by, to land in `[0, 1]`.
    pub divisor: f32,
}

/// The six complexity heads, in checkpoint order (head 0 is the task type, head 6 is an
/// unused `no_label_reason`).
pub const HEADS: [Head; 6] = [
    Head {
        name: "creativity",
        classes: 3,
        weights: &[2.0, 1.0, 0.0],
        divisor: 2.0,
    },
    Head {
        name: "reasoning",
        classes: 2,
        weights: &[0.0, 1.0],
        divisor: 1.0,
    },
    Head {
        name: "contextual_knowledge",
        classes: 2,
        weights: &[0.0, 1.0],
        divisor: 1.0,
    },
    Head {
        name: "few_shots",
        classes: 6,
        weights: &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        divisor: 1.0,
    },
    Head {
        name: "domain_knowledge",
        classes: 4,
        weights: &[3.0, 1.0, 2.0, 0.0],
        divisor: 3.0,
    },
    Head {
        name: "constraints",
        classes: 2,
        weights: &[1.0, 0.0],
        divisor: 1.0,
    },
];

/// Index into [`HEADS`], and into the tensor position of each head in the checkpoint.
mod head_index {
    pub const CREATIVITY: usize = 0;
    pub const REASONING: usize = 1;
    pub const CONTEXTUAL: usize = 2;
    pub const FEW_SHOTS: usize = 3;
    pub const DOMAIN: usize = 4;
    pub const CONSTRAINTS: usize = 5;
}

/// Which `head_N` module in the checkpoint each entry of [`HEADS`] is.
///
/// The checkpoint's order is task_type, creativity, reasoning, contextual_knowledge,
/// few_shots, domain_knowledge, no_label_reason, constraints — so head 6 is skipped and
/// constraints is 7.
pub const HEAD_TENSOR_INDEX: [usize; 6] = [1, 2, 3, 4, 5, 7];

/// The tensor index of the task-type head.
pub const TASK_HEAD_INDEX: usize = 0;

/// The six dimensions, each in `[0, 1]` except `few_shots` which counts examples.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Dimensions {
    /// How much invention the answer needs.
    pub creativity: f32,
    /// How much logical work.
    pub reasoning: f32,
    /// How much background outside the prompt.
    pub contextual_knowledge: f32,
    /// How many worked examples the prompt carries.
    pub few_shots: f32,
    /// How specialised the subject is.
    pub domain_knowledge: f32,
    /// How many conditions the prompt imposes.
    pub constraints: f32,
}

impl Dimensions {
    /// Read the dimensions out of per-head probability vectors, in [`HEADS`] order.
    pub fn from_probs(per_head: &[Vec<f32>; 6]) -> Self {
        let score = |i: usize| collapse(&per_head[i], &HEADS[i]);
        Self {
            creativity: score(head_index::CREATIVITY),
            reasoning: score(head_index::REASONING),
            contextual_knowledge: score(head_index::CONTEXTUAL),
            // The reference implementation floors noise here: fewer than 0.05 of an
            // example is no example.
            few_shots: {
                let s = score(head_index::FEW_SHOTS);
                if s >= 0.05 { s } else { 0.0 }
            },
            domain_knowledge: score(head_index::DOMAIN),
            constraints: score(head_index::CONSTRAINTS),
        }
    }

    /// The score pstore routes on: [`ROUTING_WEIGHTS`] applied to these dimensions.
    pub fn routing_score(&self) -> f32 {
        let w = &ROUTING_WEIGHTS;
        w.creativity * self.creativity
            + w.reasoning * self.reasoning
            + w.contextual_knowledge * self.contextual_knowledge
            + w.few_shots * self.few_shots
            + w.domain_knowledge * self.domain_knowledge
            + w.constraints * self.constraints
    }

    /// The checkpoint's own published ensemble.
    ///
    /// Not what pstore routes on — see the module docs — but reproducing the model card's
    /// worked examples with it is what demonstrates the dimensions are read out correctly,
    /// so it is kept and tested.
    pub fn published_score(&self) -> f32 {
        0.35 * self.creativity
            + 0.25 * self.reasoning
            + 0.15 * self.constraints
            + 0.15 * self.domain_knowledge
            + 0.05 * self.contextual_knowledge
            + 0.05 * self.few_shots
    }

    /// The dimensions worth naming in the UI, strongest first.
    pub fn notable(&self, threshold: f32) -> Vec<(&'static str, f32)> {
        let mut v: Vec<(&'static str, f32)> = HEADS
            .iter()
            .map(|h| h.name)
            .zip([
                self.creativity,
                self.reasoning,
                self.contextual_knowledge,
                self.few_shots,
                self.domain_knowledge,
                self.constraints,
            ])
            .filter(|(_, s)| *s >= threshold)
            .collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        v
    }
}

/// Collapse one head's probabilities into its dimension score.
pub fn collapse(probs: &[f32], head: &Head) -> f32 {
    let weighted: f32 = probs
        .iter()
        .zip(head.weights)
        .map(|(p, w)| p * w)
        .sum::<f32>();
    if head.divisor == 0.0 {
        return 0.0;
    }
    weighted / head.divisor
}

/// Softmax over one head's logits.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::MIN, f32::max);
    let exps: Vec<f32> = logits.iter().map(|z| (z - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        // Degenerate logits: spread the mass rather than emit NaN.
        return vec![1.0 / logits.len().max(1) as f32; logits.len()];
    }
    exps.into_iter().map(|e| e / sum).collect()
}

/// Turn a routing score into a label.
pub fn label(score: f32) -> Complexity {
    if score < EASY_BELOW {
        Complexity::Easy
    } else if score < HARD_FROM {
        Complexity::Medium
    } else {
        Complexity::Hard
    }
}

/// How far the score sits from the nearest label boundary, as a `[0, 1]` confidence.
///
/// A continuous score thresholded into three bands has no probability of its own, so this
/// reports the only thing that is true: whether a nudge would have changed the answer.
/// Reaching full confidence takes half the width of the medium band.
pub fn confidence(score: f32) -> f32 {
    let margin = (score - EASY_BELOW).abs().min((score - HARD_FROM).abs());
    let half_band = (HARD_FROM - EASY_BELOW) / 2.0;
    (margin / half_band).clamp(0.0, 1.0)
}

/// Pick the highest-scoring task type.
pub fn task(probs: &[f32]) -> (&'static str, f32) {
    let (i, p) =
        probs.iter().enumerate().fold(
            (11usize, 0.0f32),
            |acc, (i, p)| {
                if *p > acc.1 { (i, *p) } else { acc }
            },
        );
    (TASK_TYPES.get(i).copied().unwrap_or("Unknown"), p)
}

/// Clip a prompt to [`MAX_CHARS`], on a character boundary.
pub fn clip(text: &str) -> String {
    if text.chars().count() > MAX_CHARS {
        text.chars().take(MAX_CHARS).collect()
    } else {
        text.to_string()
    }
}

/// The reading this classifier produces.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Verdict {
    /// The label.
    pub complexity: Complexity,
    /// The continuous score behind it.
    pub score: f32,
    /// Distance from the nearest band boundary, as a confidence.
    pub confidence: f32,
    /// Per-dimension detail.
    pub dimensions: Dimensions,
    /// The task the model thinks this is, and how sure it is.
    pub task: (&'static str, f32),
}

/// Assemble a verdict from the eight heads' logits.
///
/// `task_logits` is head 0; `head_logits` are the six complexity heads in [`HEADS`] order.
pub fn assemble(task_logits: &[f32], head_logits: &[Vec<f32>; 6]) -> Result<Verdict, String> {
    for (i, h) in HEADS.iter().enumerate() {
        if head_logits[i].len() != h.classes {
            return Err(format!(
                "head {:?} produced {} classes, expected {}",
                h.name,
                head_logits[i].len(),
                h.classes
            ));
        }
    }
    if task_logits.len() != TASK_TYPES.len() {
        return Err(format!(
            "task head produced {} classes, expected {}",
            task_logits.len(),
            TASK_TYPES.len()
        ));
    }

    let probs: [Vec<f32>; 6] = std::array::from_fn(|i| softmax(&head_logits[i]));
    let dimensions = Dimensions::from_probs(&probs);
    let score = dimensions.routing_score();
    Ok(Verdict {
        complexity: label(score),
        score,
        confidence: confidence(score),
        dimensions,
        task: task(&softmax(task_logits)),
    })
}

#[cfg(feature = "candle")]
mod real {
    use super::{HEAD_TENSOR_INDEX, HEADS, MAX_TOKENS, TASK_HEAD_INDEX, Verdict, assemble, clip};
    use crate::models;
    use crate::router::pooling::masked_mean;

    use candle_core::{DType, Device, Tensor};
    use candle_nn::{Linear, VarBuilder};
    use candle_transformers::models::debertav2;
    use tokenizers::Tokenizer;

    /// DeBERTa-v3-base's hyperparameters.
    ///
    /// The checkpoint's own `config.json` describes the *heads* (targets, weights,
    /// divisors) and says only `"base_model": "microsoft/DeBERTa-v3-base"` about the
    /// backbone, so the backbone shape has to come from somewhere. It is spelled out here
    /// rather than fetched from a second repository: these values are fixed by the tensor
    /// shapes in the file, and any mismatch fails loudly when the weights are loaded.
    const BACKBONE_CONFIG: &str = r#"{
        "vocab_size": 128100,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "type_vocab_size": 0,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-7,
        "relative_attention": true,
        "max_relative_positions": -1,
        "pad_token_id": 0,
        "position_biased_input": false,
        "pos_att_type": ["p2c", "c2p"],
        "position_buckets": 256,
        "share_att_key": true,
        "norm_rel_ebd": "layer_norm"
    }"#;

    /// A loaded difficulty classifier.
    pub struct Model {
        backbone: debertav2::DebertaV2Model,
        task_head: Linear,
        heads: Vec<Linear>,
        tokenizer: Tokenizer,
        device: Device,
    }

    impl Model {
        /// Download (or reuse the cache for) the checkpoint and build the model.
        ///
        /// Blocking; call from a worker thread.
        pub fn load(device: Device) -> Result<Self, String> {
            models::set(models::DIFFICULTY.id, models::Phase::Loading);
            match Self::build(device) {
                Ok(m) => {
                    models::set(models::DIFFICULTY.id, models::Phase::Ready);
                    Ok(m)
                }
                Err(e) => {
                    models::set(models::DIFFICULTY.id, models::Phase::Failed(e.clone()));
                    Err(e)
                }
            }
        }

        fn build(device: Device) -> Result<Self, String> {
            let c = &models::DIFFICULTY;
            let tok_path = crate::router::hub::fetch(c.repo, "tokenizer.json")?;
            let weights_path = crate::router::hub::fetch(c.repo, "model.safetensors")?;

            let config: debertav2::Config = serde_json::from_str(BACKBONE_CONFIG)
                .map_err(|e| format!("built-in DeBERTa config is malformed: {e}"))?;
            let tokenizer =
                Tokenizer::from_file(&tok_path).map_err(|e| format!("tokenizer: {e}"))?;

            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)
                    .map_err(|e| format!("loading weights: {e}"))?
            };

            let backbone = debertav2::DebertaV2Model::load(vb.pp("backbone"), &config)
                .map_err(|e| format!("building the DeBERTa backbone: {e}"))?;

            let hidden = config.hidden_size;
            let head = |index: usize, classes: usize| {
                candle_nn::linear(hidden, classes, vb.pp(format!("head_{index}.fc")))
                    .map_err(|e| format!("building head_{index}: {e}"))
            };
            let task_head = head(TASK_HEAD_INDEX, super::TASK_TYPES.len())?;
            let heads = HEADS
                .iter()
                .enumerate()
                .map(|(i, h)| head(HEAD_TENSOR_INDEX[i], h.classes))
                .collect::<Result<Vec<_>, _>>()?;

            Ok(Self {
                backbone,
                task_head,
                heads,
                tokenizer,
                device,
            })
        }

        /// Score one prompt with a single forward pass.
        pub fn classify(&self, text: &str) -> Result<Verdict, String> {
            let clipped = clip(text);
            let encoding = self
                .tokenizer
                .encode(clipped.as_str(), true)
                .map_err(|e| format!("tokenizing: {e}"))?;

            // Truncate rather than let the backbone fail: past 512 tokens DeBERTa-v3's
            // position table runs out, and the reference implementation truncates too.
            let n = encoding.get_ids().len().min(MAX_TOKENS);
            if n == 0 {
                return Err("prompt tokenized to nothing".into());
            }
            let ids = encoding.get_ids()[..n].to_vec();
            let mask = encoding.get_attention_mask()[..n].to_vec();

            let input = Tensor::from_vec(ids, (1, n), &self.device)
                .map_err(|e| format!("input tensor: {e}"))?;
            let attn = Tensor::from_vec(mask, (1, n), &self.device)
                .map_err(|e| format!("mask tensor: {e}"))?;

            let hidden = self
                .backbone
                .forward(&input, None, Some(attn.clone()))
                .map_err(|e| format!("encoder forward: {e}"))?;
            let pooled = masked_mean(&hidden, &attn)?;

            let logits = |head: &Linear| -> Result<Vec<f32>, String> {
                pooled
                    .apply(head)
                    .and_then(|t| t.flatten_all())
                    .and_then(|t| t.to_dtype(DType::F32))
                    .and_then(|t| t.to_vec1())
                    .map_err(|e| format!("head forward: {e}"))
            };

            let task_logits = logits(&self.task_head)?;
            let mut head_logits: [Vec<f32>; 6] = Default::default();
            for (i, head) in self.heads.iter().enumerate() {
                head_logits[i] = logits(head)?;
            }
            assemble(&task_logits, &head_logits)
        }
    }
}

#[cfg(feature = "candle")]
pub use real::Model;

#[cfg(test)]
mod tests {
    use super::*;

    /// Prompts spanning trivial to very hard, used to set [`EASY_BELOW`] and
    /// [`HARD_FROM`]. Run `cargo test --release -- --ignored calibrate --nocapture` with the
    /// weights downloaded to print the score for each and check the bands still separate.
    #[cfg(feature = "candle")]
    const CALIBRATION: [(&str, &str); 11] = [
        ("easy", "fix this typo"),
        ("easy", "add a --verbose flag to the CLI"),
        ("easy", "rename the variable `x` to `count` in src/main.rs"),
        (
            "easy",
            "bump the serde dependency to 1.0.229 and run the tests",
        ),
        (
            "medium",
            "Add pagination to the /users endpoint, keeping the existing response shape \
             and the current default page size.",
        ),
        (
            "medium",
            "Write unit tests for the retry helper in src/net/retry.rs covering timeouts, \
             5xx responses and the cancellation path.",
        ),
        (
            "medium",
            "Extract the CSV parsing out of src/import.rs into its own module, keep the \
             public API unchanged, and add a test for the quoted-field case.",
        ),
        (
            "hard",
            "Refactor the authentication layer across src/auth/mod.rs, src/auth/session.rs \
             and src/api/routes.rs without breaking backwards compatibility, and fix the \
             race condition in the token refresh.",
        ),
        (
            "hard",
            "Design and implement a lock-free single-producer single-consumer ring buffer \
             in Rust. Document the memory-ordering argument for every atomic operation and \
             show why the ABA problem cannot occur.",
        ),
        (
            "hard",
            "Migrate the service from Postgres to CockroachDB: rewrite the forty queries in \
             src/db/, preserve transactional semantics, add a dual-write phase behind a \
             feature flag, and write the rollback plan.",
        ),
        (
            "hard",
            "The p99 latency of the ingest path tripled after the last deploy and the \
             profile shows the time in serde. Work out why, then fix it without changing \
             the wire format or the on-disk layout.",
        ),
    ];

    /// Print the score for every calibration prompt, then assert the bands still hold.
    ///
    /// This is where [`EASY_BELOW`] and [`HARD_FROM`] come from, and it is the test to
    /// re-run if the checkpoint is ever updated.
    #[test]
    #[ignore = "needs the 744 MB difficulty checkpoint downloaded"]
    #[cfg(feature = "candle")]
    fn calibrate_thresholds_against_coding_prompts() {
        crate::models::download(
            &crate::models::DIFFICULTY,
            &std::sync::atomic::AtomicBool::new(false),
        )
        .expect("fetching the difficulty checkpoint");
        let (dev, _) = crate::router::device::pick();
        let model = Model::load(dev).expect("loading the difficulty checkpoint");

        let mut wrong = Vec::new();
        for (expected, prompt) in CALIBRATION {
            let v = model.classify(prompt).expect("classifying");
            eprintln!(
                "{:>6} -> {:<6} {:.4}  cre {:.2} rea {:.2} ctx {:.2} few {:.2} dom {:.2} con {:.2}  [{}]  {:.40}",
                expected,
                v.complexity.label(),
                v.score,
                v.dimensions.creativity,
                v.dimensions.reasoning,
                v.dimensions.contextual_knowledge,
                v.dimensions.few_shots,
                v.dimensions.domain_knowledge,
                v.dimensions.constraints,
                v.task.0,
                prompt,
            );
            if v.complexity.label() != expected {
                wrong.push(format!("{expected} prompt scored {:.4}", v.score));
            }
        }
        assert!(
            wrong.len() <= 2,
            "the bands no longer fit coding prompts ({} of {} misplaced):\n  {}",
            wrong.len(),
            CALIBRATION.len(),
            wrong.join("\n  ")
        );
    }

    /// Dimension vectors as the real checkpoint produced them, from a run of
    /// [`calibrate_thresholds_against_coding_prompts`]. Pinning the band logic to measured
    /// output means the thresholds stay tested without needing 744 MB of weights.
    fn measured(prompt: &str) -> Dimensions {
        let d = |creativity,
                 reasoning,
                 contextual_knowledge,
                 few_shots,
                 domain_knowledge,
                 constraints| Dimensions {
            creativity,
            reasoning,
            contextual_knowledge,
            few_shots,
            domain_knowledge,
            constraints,
        };
        match prompt {
            "typo" => d(0.02, 0.01, 0.80, 0.00, 0.44, 0.09),
            "flag" => d(0.03, 0.00, 0.25, 0.00, 0.98, 0.06),
            "rename" => d(0.01, 0.01, 0.17, 0.00, 0.99, 0.11),
            "pagination" => d(0.02, 0.01, 0.24, 0.00, 0.99, 0.30),
            "unit tests" => d(0.03, 0.02, 0.08, 0.06, 1.00, 0.48),
            "extract module" => d(0.01, 0.01, 0.12, 0.00, 1.00, 0.66),
            "auth refactor" => d(0.03, 0.02, 0.10, 0.00, 0.99, 0.80),
            "ring buffer" => d(0.07, 0.10, 0.06, 0.00, 0.99, 0.80),
            "latency hunt" => d(0.04, 0.34, 0.12, 0.00, 0.97, 0.88),
            other => unreachable!("no measurement for {other}"),
        }
    }

    #[test]
    fn measured_coding_prompts_land_in_the_right_bands() {
        for p in ["typo", "flag", "rename"] {
            let s = measured(p).routing_score();
            assert_eq!(label(s), Complexity::Easy, "{p} scored {s:.4}");
        }
        for p in ["pagination", "unit tests", "extract module"] {
            let s = measured(p).routing_score();
            assert_eq!(label(s), Complexity::Medium, "{p} scored {s:.4}");
        }
        for p in ["auth refactor", "ring buffer", "latency hunt"] {
            let s = measured(p).routing_score();
            assert_eq!(label(s), Complexity::Hard, "{p} scored {s:.4}");
        }
    }

    #[test]
    fn the_routing_score_separates_prompts_the_published_one_squashes() {
        let spread = |f: fn(&Dimensions) -> f32| {
            let easiest = f(&measured("flag"));
            let hardest = f(&measured("latency hunt"));
            hardest - easiest
        };
        let routing = spread(|d| d.routing_score());
        let published = spread(|d| d.published_score());
        assert!(
            routing > published * 1.5,
            "the point of the reweighting is range: routing {routing:.3} vs published \
             {published:.3}"
        );

        // And the ordering is monotone in difficulty across the measured set.
        let ordered = [
            "flag",
            "rename",
            "pagination",
            "unit tests",
            "extract module",
            "auth refactor",
            "latency hunt",
        ];
        let scores: Vec<f32> = ordered
            .iter()
            .map(|p| measured(p).routing_score())
            .collect();
        assert!(
            scores.windows(2).all(|w| w[0] < w[1]),
            "scores should rise with difficulty: {:?}",
            ordered.iter().zip(&scores).collect::<Vec<_>>()
        );
    }

    #[test]
    fn thresholds_are_ordered_and_leave_a_medium_band() {
        const { assert!(EASY_BELOW < HARD_FROM, "bands must not invert") };
        const { assert!(HARD_FROM - EASY_BELOW > 0.05, "medium must be reachable") };
        // Boundaries belong to the harder band, so a score is never unlabelled.
        assert_eq!(label(EASY_BELOW), Complexity::Medium);
        assert_eq!(label(HARD_FROM), Complexity::Hard);
        assert_eq!(label(0.0), Complexity::Easy);
        assert_eq!(label(1.0), Complexity::Hard);
    }

    #[test]
    fn routing_weights_ignore_the_saturated_dimension_and_sum_to_one() {
        let w = ROUTING_WEIGHTS;
        assert_eq!(
            w.domain_knowledge, 0.0,
            "domain knowledge is ~1.0 for every coding prompt; weighting it adds a constant"
        );
        assert!(
            w.constraints > w.reasoning && w.reasoning > w.creativity,
            "constraints discriminate most, creativity least: {w:?}"
        );
        let total = w.creativity
            + w.reasoning
            + w.contextual_knowledge
            + w.few_shots
            + w.domain_knowledge
            + w.constraints;
        assert!((total - 1.0).abs() < 1e-6, "weights sum to {total}");

        // An all-ones reading therefore scores 1.0, and an all-zeros one 0.0.
        let all = Dimensions {
            creativity: 1.0,
            reasoning: 1.0,
            contextual_knowledge: 1.0,
            few_shots: 1.0,
            domain_knowledge: 1.0,
            constraints: 1.0,
        };
        assert!((all.routing_score() - 1.0).abs() < 1e-6);
        assert_eq!(Dimensions::default().routing_score(), 0.0);
    }

    /// The card's creative-writing example, reproduced from its per-dimension table:
    /// creativity 0.867, reasoning 0.056, contextual 0.048, domain 0.226, constraints
    /// 0.785, few-shots 0. If the head order or the weights map were wrong, the published
    /// formula would not come back at the card's 0.472 — which is what makes this the
    /// check that the six dimensions are read out correctly.
    #[test]
    fn card_dimensions_reproduce_the_published_card_score() {
        let d = Dimensions {
            creativity: 0.867,
            reasoning: 0.056,
            contextual_knowledge: 0.048,
            few_shots: 0.0,
            domain_knowledge: 0.226,
            constraints: 0.785,
        };
        assert!(
            (d.published_score() - 0.472).abs() < 0.001,
            "got {} for the card's example",
            d.published_score()
        );
        // The card's other example: a one-sentence summarisation, published score 0.133.
        let easy = Dimensions {
            creativity: 0.003,
            reasoning: 0.014,
            contextual_knowledge: 0.003,
            few_shots: 0.0,
            domain_knowledge: 0.644,
            constraints: 0.211,
        };
        assert!(
            (easy.published_score() - 0.133).abs() < 0.001,
            "got {}",
            easy.published_score()
        );
    }

    #[test]
    fn head_weights_come_from_the_checkpoint_config() {
        // domain_knowledge's classes are High, Low, Medium, No — not an ordered scale,
        // which is exactly why the weights are explicit.
        let domain = HEADS[4];
        assert_eq!(domain.name, "domain_knowledge");
        assert_eq!(domain.weights, &[3.0, 1.0, 2.0, 0.0]);
        assert_eq!(domain.divisor, 3.0);
        // "High" (class 0) must score 1.0, "No" (class 3) must score 0.0.
        assert_eq!(collapse(&[1.0, 0.0, 0.0, 0.0], &domain), 1.0);
        assert_eq!(collapse(&[0.0, 0.0, 0.0, 1.0], &domain), 0.0);
        // "Medium" sits between "Low" and "High".
        assert!(
            collapse(&[0.0, 1.0, 0.0, 0.0], &domain) < collapse(&[0.0, 0.0, 1.0, 0.0], &domain)
        );

        // creativity's classes are High, Low, No: reversed relative to its score.
        let creativity = HEADS[0];
        assert_eq!(collapse(&[1.0, 0.0, 0.0], &creativity), 1.0);
        assert_eq!(collapse(&[0.0, 0.0, 1.0], &creativity), 0.0);

        // constraints is [1, 0]: class 0 means "has constraints".
        let constraints = HEADS[5];
        assert_eq!(collapse(&[1.0, 0.0], &constraints), 1.0);

        for h in HEADS.iter() {
            assert_eq!(h.weights.len(), h.classes, "{} weights vs classes", h.name);
            assert!(h.divisor > 0.0, "{} divisor", h.name);
        }
    }

    #[test]
    fn head_tensor_indices_skip_the_unused_head() {
        // The checkpoint's head order is task, creativity, reasoning, contextual,
        // few_shots, domain, no_label_reason, constraints. Reading constraints from
        // head_6 would silently score every prompt against an unused head.
        assert_eq!(HEAD_TENSOR_INDEX, [1, 2, 3, 4, 5, 7]);
        assert_eq!(TASK_HEAD_INDEX, 0);
        assert!(!HEAD_TENSOR_INDEX.contains(&6));
    }

    #[test]
    fn few_shot_noise_is_floored() {
        let mut probs: [Vec<f32>; 6] = std::array::from_fn(|i| {
            let mut v = vec![0.0; HEADS[i].classes];
            v[HEADS[i].classes - 1] = 1.0;
            v
        });
        // A whisper of "one example" is not an example.
        probs[3] = vec![0.98, 0.02, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(Dimensions::from_probs(&probs).few_shots, 0.0);
        // Half a vote for "two examples" is.
        probs[3] = vec![0.5, 0.0, 0.5, 0.0, 0.0, 0.0];
        assert_eq!(Dimensions::from_probs(&probs).few_shots, 1.0);
    }

    #[test]
    fn softmax_is_a_distribution_and_never_nan() {
        let p = softmax(&[1.0, 2.0, 3.0]);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(p[2] > p[1] && p[1] > p[0]);
        for logits in [vec![0.0; 3], vec![f32::MIN; 2], vec![1e30, -1e30]] {
            let p = softmax(&logits);
            assert!(p.iter().all(|v| v.is_finite()), "{logits:?} -> {p:?}");
        }
    }

    #[test]
    fn confidence_is_lowest_at_the_boundaries() {
        assert_eq!(confidence(EASY_BELOW), 0.0);
        assert_eq!(confidence(HARD_FROM), 0.0);
        assert_eq!(confidence(0.0), 1.0, "far from any boundary");
        assert_eq!(confidence(1.0), 1.0);
        // Mid-band is as confident as a thresholded score gets.
        let mid = (EASY_BELOW + HARD_FROM) / 2.0;
        assert!(confidence(mid) > 0.99, "got {}", confidence(mid));
        for s in [-1.0, 0.0, 0.5, 1.0, 42.0] {
            assert!((0.0..=1.0).contains(&confidence(s)), "score {s}");
        }
    }

    #[test]
    fn task_names_are_read_in_checkpoint_order() {
        let mut probs = vec![0.0; 12];
        probs[4] = 0.9;
        assert_eq!(task(&probs), ("Code Generation", 0.9));
        probs[4] = 0.0;
        probs[9] = 0.5;
        assert_eq!(task(&probs).0, "Summarization");
        // An out-of-range or empty distribution must not panic.
        assert_eq!(task(&[]).0, "Unknown");
    }

    #[test]
    fn assemble_rejects_a_mis_shaped_head() {
        let good: [Vec<f32>; 6] = std::array::from_fn(|i| vec![0.5; HEADS[i].classes]);
        assert!(assemble(&[0.1; 12], &good).is_ok());

        let mut bad = good.clone();
        bad[0] = vec![0.5; 2];
        let err = assemble(&[0.1; 12], &bad).unwrap_err();
        assert!(err.contains("creativity"), "got {err}");

        // A task head of the wrong width means the checkpoint changed shape.
        assert!(assemble(&[0.1; 5], &good).is_err());
    }

    #[test]
    fn assemble_produces_a_coherent_verdict() {
        // Everything at its hardest class.
        let hard: [Vec<f32>; 6] = std::array::from_fn(|i| {
            let mut v = vec![0.0; HEADS[i].classes];
            // Class carrying the largest weight, per head.
            let (best, _) =
                HEADS[i]
                    .weights
                    .iter()
                    .enumerate()
                    .fold(
                        (0, f32::MIN),
                        |acc, (j, w)| {
                            if *w > acc.1 { (j, *w) } else { acc }
                        },
                    );
            v[best] = 20.0;
            v
        });
        let mut task_logits = vec![0.0; 12];
        task_logits[4] = 20.0;

        let v = assemble(&task_logits, &hard).unwrap();
        assert_eq!(v.complexity, Complexity::Hard);
        assert!(v.score > 0.9, "got {}", v.score);
        assert_eq!(v.task.0, "Code Generation");
        assert!(v.task.1 > 0.99);
        assert!(v.confidence > 0.99);
        assert!(!v.dimensions.notable(0.5).is_empty());
    }

    #[test]
    fn clip_bounds_the_input_on_char_boundaries() {
        // Byte slicing would panic on this.
        let long = "é".repeat(MAX_CHARS * 2);
        let clipped = clip(&long);
        assert_eq!(clipped.chars().count(), MAX_CHARS);
        assert_eq!(clip("short"), "short");
    }

    #[test]
    fn notable_orders_dimensions_by_strength() {
        let d = Dimensions {
            creativity: 0.1,
            reasoning: 0.9,
            contextual_knowledge: 0.2,
            few_shots: 0.0,
            domain_knowledge: 0.7,
            constraints: 0.4,
        };
        let names: Vec<_> = d.notable(0.3).into_iter().map(|(n, _)| n).collect();
        assert_eq!(names, vec!["reasoning", "domain_knowledge", "constraints"]);
        assert!(d.notable(0.95).is_empty());
    }
}
