//! Brick's capability classifier, running in-process on Candle.
//!
//! Model: `regolo/brick-modernbert-capability-classifier` — ModernBERT plus a
//! `Linear(hidden → 6)` head, trained multi-label. The head is **sigmoid**, not
//! softmax: a prompt can draw on several capabilities at once and the six scores do
//! not sum to 1.
//!
//! The label order in the checkpoint is read from `config.json`'s `id2label` and
//! permuted into [`DIMS`]. That indirection is load-bearing: the model card's order
//! differs from the alphabetical order used in Brick's own `skill_router.models`
//! YAML, and hardcoding either one would silently mis-route every prompt.

use crate::agents::registry::DIMS;
use crate::router::Capability;

/// Hugging Face repo holding the classifier.
///
/// Taken from the catalogue so the Models window and the loader can never disagree about
/// which checkpoint "the capability classifier" means.
pub const REPO: &str = crate::models::CAPABILITY.repo;

/// Map the checkpoint's label order onto [`DIMS`].
///
/// Returns, for each position in `DIMS`, the index to read from the model's logits.
/// Fails loudly rather than guessing when a dimension is missing — a wrong
/// permutation is worse than no classifier at all.
pub fn permutation(id2label: &[(usize, String)]) -> Result<[usize; 6], String> {
    let mut perm = [usize::MAX; 6];
    for (dim_idx, dim) in DIMS.iter().enumerate() {
        let found = id2label
            .iter()
            .find(|(_, name)| name.trim().eq_ignore_ascii_case(dim))
            .map(|(id, _)| *id);
        match found {
            Some(i) => perm[dim_idx] = i,
            None => {
                return Err(format!(
                    "checkpoint has no label for capability dimension {dim:?}; \
                     found {:?}",
                    id2label.iter().map(|(_, n)| n.as_str()).collect::<Vec<_>>()
                ));
            }
        }
    }
    Ok(perm)
}

/// Parse `id2label` out of a Hugging Face `config.json`.
pub fn parse_id2label(config_json: &str) -> Result<Vec<(usize, String)>, String> {
    let v: serde_json::Value =
        serde_json::from_str(config_json).map_err(|e| format!("config.json is not JSON: {e}"))?;
    let map = v
        .get("id2label")
        .and_then(|m| m.as_object())
        .ok_or_else(|| "config.json has no id2label object".to_string())?;

    let mut out = Vec::with_capacity(map.len());
    for (k, val) in map {
        let id: usize = k
            .parse()
            .map_err(|_| format!("id2label key {k:?} is not an integer"))?;
        let name = val
            .as_str()
            .ok_or_else(|| format!("id2label[{k}] is not a string"))?
            .to_string();
        out.push((id, name));
    }
    out.sort_by_key(|(id, _)| *id);
    Ok(out)
}

/// Sigmoid, applied element-wise. The head is multi-label, so this is the correct
/// activation — softmax would force the six scores to compete.
pub fn sigmoid(logits: &[f32]) -> Vec<f32> {
    logits.iter().map(|z| 1.0 / (1.0 + (-z).exp())).collect()
}

/// Assemble a [`Capability`] from raw logits and a label permutation.
pub fn assemble(logits: &[f32], perm: &[usize; 6]) -> Result<Capability, String> {
    let probs = sigmoid(logits);
    let mut scores = [0.0f32; 6];
    for (dim_idx, &src) in perm.iter().enumerate() {
        scores[dim_idx] = *probs
            .get(src)
            .ok_or_else(|| format!("logit index {src} out of range ({} present)", probs.len()))?;
    }
    Ok(Capability { scores })
}

/// Bring a ModernBERT `config.json` into the shape candle expects.
///
/// This checkpoint was exported by transformers 5.x, which nests the RoPE settings
/// under `rope_parameters`. Candle 0.11 still wants the older flat `global_rope_theta`
/// and `local_rope_theta`, so translate rather than fail: the values are present, just
/// in a different place.
pub fn normalize_config(config_json: &str) -> Result<String, String> {
    let mut v: serde_json::Value =
        serde_json::from_str(config_json).map_err(|e| format!("config.json is not JSON: {e}"))?;

    let rope = |name: &str| -> Option<f64> {
        v.get("rope_parameters")?
            .get(name)?
            .get("rope_theta")?
            .as_f64()
    };
    // `full_attention` is the global (every-nth-layer) attention; `sliding_attention`
    // is the local window.
    let global = rope("full_attention");
    let local = rope("sliding_attention");

    let obj = v
        .as_object_mut()
        .ok_or_else(|| "config.json is not an object".to_string())?;
    if !obj.contains_key("global_rope_theta")
        && let Some(g) = global
    {
        obj.insert("global_rope_theta".into(), g.into());
    }
    if !obj.contains_key("local_rope_theta") {
        // Some exports omit the local theta entirely; the upstream default is 10000.
        obj.insert("local_rope_theta".into(), local.unwrap_or(10_000.0).into());
    }

    serde_json::to_string(&v).map_err(|e| format!("re-encoding config: {e}"))
}

#[cfg(feature = "candle")]
mod real {
    use super::{REPO, assemble, normalize_config, parse_id2label, permutation};
    use crate::models;
    use crate::router::Capability;
    use crate::router::pooling::masked_mean;

    use candle_core::{DType, Device, Tensor};
    use candle_nn::{LayerNorm, Linear, VarBuilder};
    use candle_transformers::models::modernbert;
    use tokenizers::Tokenizer;

    /// A loaded capability classifier.
    ///
    /// Assembled from parts rather than using candle's
    /// `ModernBertForSequenceClassification`, for one decisive reason: that wrapper
    /// applies **softmax** to the logits, and this checkpoint is trained multi-label
    /// (`problem_type: multi_label_classification`) so it needs **sigmoid**. Softmax
    /// would force the six capabilities to compete for a fixed budget and quietly
    /// distort every ranking.
    pub struct Model {
        encoder: modernbert::ModernBert,
        head_dense: Linear,
        head_norm: LayerNorm,
        classifier: Linear,
        tokenizer: Tokenizer,
        perm: [usize; 6],
        device: Device,
    }

    impl Model {
        /// Download (or reuse the cache for) the checkpoint and build the model,
        /// recording each step on the [`models`] status board so the GUI can show it.
        ///
        /// Blocking; call from a worker thread.
        pub fn load(device: Device) -> Result<Self, String> {
            models::set(models::CAPABILITY.id, models::Phase::Loading);
            match Self::build(device) {
                Ok(m) => {
                    models::set(models::CAPABILITY.id, models::Phase::Ready);
                    Ok(m)
                }
                Err(e) => {
                    models::set(models::CAPABILITY.id, models::Phase::Failed(e.clone()));
                    Err(e)
                }
            }
        }

        fn build(device: Device) -> Result<Self, String> {
            let fetch = crate::router::hub::fetch;
            let config_path = fetch(REPO, "config.json")?;
            let tok_path = fetch(REPO, "tokenizer.json")?;
            let weights_path = fetch(REPO, "model.safetensors")?;

            let raw_config = std::fs::read_to_string(&config_path)
                .map_err(|e| format!("reading config: {e}"))?;
            let perm = permutation(&parse_id2label(&raw_config)?)?;
            let config: modernbert::Config = serde_json::from_str(&normalize_config(&raw_config)?)
                .map_err(|e| format!("config.json is not a ModernBERT config: {e}"))?;

            let tokenizer =
                Tokenizer::from_file(&tok_path).map_err(|e| format!("tokenizer: {e}"))?;

            // fp32 keeps the classifier numerically boring; it is small enough that the
            // memory saved by fp16 would not be worth the risk.
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)
                    .map_err(|e| format!("loading weights: {e}"))?
            };

            // `ModernBert::load` applies the `model.` prefix itself — pass the root.
            let encoder = modernbert::ModernBert::load(vb.clone(), &config)
                .map_err(|e| format!("building encoder: {e}"))?;

            let hidden = config.hidden_size;
            // Tensor shapes in the checkpoint: head.dense.weight [h, h] (no bias),
            // head.norm.weight [h] (no bias), classifier.{weight [6, h], bias [6]}.
            let head_dense = candle_nn::linear_no_bias(hidden, hidden, vb.pp("head.dense"))
                .map_err(|e| format!("building head.dense: {e}"))?;
            let head_norm_weight = vb
                .pp("head.norm")
                .get(hidden, "weight")
                .map_err(|e| format!("building head.norm: {e}"))?;
            let head_norm = LayerNorm::new_no_bias(head_norm_weight, config.layer_norm_eps);
            let classifier = candle_nn::linear(hidden, 6, vb.pp("classifier"))
                .map_err(|e| format!("building classifier: {e}"))?;

            Ok(Self {
                encoder,
                head_dense,
                head_norm,
                classifier,
                tokenizer,
                perm,
                device,
            })
        }

        /// Score one prompt.
        pub fn classify(&self, text: &str) -> Result<Capability, String> {
            let encoding = self
                .tokenizer
                .encode(text, true)
                .map_err(|e| format!("tokenizing: {e}"))?;
            let ids: Vec<u32> = encoding.get_ids().to_vec();
            let mask: Vec<u32> = encoding.get_attention_mask().to_vec();
            if ids.is_empty() {
                return Err("prompt tokenized to nothing".into());
            }

            let n = ids.len();
            let input = Tensor::from_vec(ids, (1, n), &self.device)
                .map_err(|e| format!("input tensor: {e}"))?;
            let attn = Tensor::from_vec(mask, (1, n), &self.device)
                .map_err(|e| format!("mask tensor: {e}"))?;

            let hidden = self
                .encoder
                .forward(&input, &attn)
                .map_err(|e| format!("encoder forward: {e}"))?;

            // `classifier_pooling: mean` in the config: mask-aware mean over positions,
            // then head, then the 6-way classifier. Order matters — pooling happens
            // before the head, matching the reference implementation.
            let pooled = masked_mean(&hidden, &attn)?;
            let logits = pooled
                .apply(&self.head_dense)
                .and_then(|t| t.gelu_erf())
                .and_then(|t| t.apply(&self.head_norm))
                .and_then(|t| t.apply(&self.classifier))
                .map_err(|e| format!("head forward: {e}"))?;
            let logits: Vec<f32> = logits
                .flatten_all()
                .and_then(|t| t.to_vec1())
                .map_err(|e| format!("reading logits: {e}"))?;

            assemble(&logits, &self.perm)
        }
    }
}

#[cfg(feature = "candle")]
pub use real::Model;

#[cfg(test)]
mod tests {
    use super::*;

    /// The order published on the model card. Deliberately not alphabetical.
    fn card_order() -> Vec<(usize, String)> {
        [
            "instruction_following",
            "coding",
            "math_reasoning",
            "world_knowledge",
            "planning_agentic",
            "creative_synthesis",
        ]
        .iter()
        .enumerate()
        .map(|(i, s)| (i, s.to_string()))
        .collect()
    }

    #[test]
    fn card_order_maps_to_identity() {
        let perm = permutation(&card_order()).unwrap();
        assert_eq!(perm, [0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn alphabetical_order_is_permuted_not_assumed() {
        // This is the trap: Brick's YAML example lists dimensions alphabetically,
        // which is a *different* order from the checkpoint. Reading id2label must
        // reorder rather than pass through.
        let alphabetical: Vec<(usize, String)> = [
            "coding",
            "creative_synthesis",
            "instruction_following",
            "math_reasoning",
            "planning_agentic",
            "world_knowledge",
        ]
        .iter()
        .enumerate()
        .map(|(i, s)| (i, s.to_string()))
        .collect();

        let perm = permutation(&alphabetical).unwrap();
        assert_ne!(perm, [0, 1, 2, 3, 4, 5], "must not be the identity");
        // DIMS[0] is instruction_following, at alphabetical index 2.
        assert_eq!(perm[0], 2);
        // DIMS[1] is coding, at alphabetical index 0.
        assert_eq!(perm[1], 0);
        // DIMS[5] is creative_synthesis, at alphabetical index 1.
        assert_eq!(perm[5], 1);
    }

    #[test]
    fn label_matching_tolerates_case_and_whitespace() {
        let noisy: Vec<(usize, String)> = [
            " Instruction_Following ",
            "CODING",
            "math_reasoning",
            "world_knowledge",
            "planning_agentic",
            "creative_synthesis",
        ]
        .iter()
        .enumerate()
        .map(|(i, s)| (i, s.to_string()))
        .collect();
        assert_eq!(permutation(&noisy).unwrap(), [0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn missing_dimension_is_an_error_not_a_guess() {
        let short: Vec<(usize, String)> = ["coding", "math_reasoning"]
            .iter()
            .enumerate()
            .map(|(i, s)| (i, s.to_string()))
            .collect();
        let err = permutation(&short).unwrap_err();
        assert!(err.contains("instruction_following"), "got {err}");
    }

    #[test]
    fn parses_id2label_and_sorts_by_id() {
        let json = r#"{
            "architectures": ["ModernBertForSequenceClassification"],
            "id2label": {"1": "coding", "0": "instruction_following", "2": "math_reasoning",
                         "3": "world_knowledge", "4": "planning_agentic", "5": "creative_synthesis"},
            "hidden_size": 768
        }"#;
        let parsed = parse_id2label(json).unwrap();
        assert_eq!(parsed[0], (0, "instruction_following".to_string()));
        assert_eq!(parsed[1], (1, "coding".to_string()));
        assert_eq!(permutation(&parsed).unwrap(), [0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn config_without_id2label_is_rejected() {
        assert!(parse_id2label(r#"{"hidden_size": 768}"#).is_err());
        assert!(parse_id2label("not json").is_err());
    }

    #[test]
    fn sigmoid_is_multilabel_not_softmax() {
        let probs = sigmoid(&[0.0, 2.0, -2.0]);
        assert!((probs[0] - 0.5).abs() < 1e-6);
        assert!(probs[1] > 0.85 && probs[1] < 1.0);
        assert!(probs[2] > 0.0 && probs[2] < 0.15);
        // Independent labels: several can be high at once, and they need not sum to 1.
        let all_high = sigmoid(&[4.0; 6]);
        assert!(all_high.iter().all(|p| *p > 0.9));
        assert!(
            all_high.iter().sum::<f32>() > 5.0,
            "softmax would cap this at 1.0"
        );
    }

    #[test]
    fn assemble_applies_the_permutation() {
        // Logits in alphabetical order; perm maps them into DIMS order.
        let alphabetical: Vec<(usize, String)> = [
            "coding",
            "creative_synthesis",
            "instruction_following",
            "math_reasoning",
            "planning_agentic",
            "world_knowledge",
        ]
        .iter()
        .enumerate()
        .map(|(i, s)| (i, s.to_string()))
        .collect();
        let perm = permutation(&alphabetical).unwrap();

        // Make "coding" (alphabetical index 0) the clear winner.
        let logits = [5.0, -5.0, -5.0, -5.0, -5.0, -5.0];
        let cap = assemble(&logits, &perm).unwrap();
        assert_eq!(cap.dominant().0, "coding", "permutation was not applied");
        assert!(cap.scores[1] > 0.99, "coding lands at DIMS index 1");
        assert!(cap.scores[0] < 0.01, "instruction_following should be low");
    }

    #[test]
    fn assemble_rejects_short_logit_vectors() {
        let perm = permutation(&card_order()).unwrap();
        let err = assemble(&[0.1, 0.2], &perm).unwrap_err();
        assert!(err.contains("out of range"), "got {err}");
    }

    #[test]
    fn normalize_config_translates_nested_rope_parameters() {
        // The shape transformers 5.x writes, which candle 0.11 does not understand.
        let raw = r#"{
            "hidden_size": 1024,
            "rope_parameters": {
                "full_attention": {"rope_theta": 160000.0, "rope_type": "default"},
                "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
            }
        }"#;
        let out = normalize_config(raw).unwrap();
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["global_rope_theta"].as_f64(), Some(160000.0));
        assert_eq!(v["local_rope_theta"].as_f64(), Some(10000.0));
        // Existing fields survive.
        assert_eq!(v["hidden_size"].as_u64(), Some(1024));
    }

    #[test]
    fn normalize_config_leaves_an_already_flat_config_alone() {
        let raw = r#"{"global_rope_theta": 5.0, "local_rope_theta": 6.0}"#;
        let v: serde_json::Value = serde_json::from_str(&normalize_config(raw).unwrap()).unwrap();
        assert_eq!(v["global_rope_theta"].as_f64(), Some(5.0));
        assert_eq!(v["local_rope_theta"].as_f64(), Some(6.0));
    }

    #[test]
    fn normalize_config_defaults_a_missing_local_theta() {
        let raw = r#"{"rope_parameters": {"full_attention": {"rope_theta": 160000.0}}}"#;
        let v: serde_json::Value = serde_json::from_str(&normalize_config(raw).unwrap()).unwrap();
        assert_eq!(v["global_rope_theta"].as_f64(), Some(160000.0));
        assert_eq!(
            v["local_rope_theta"].as_f64(),
            Some(10000.0),
            "upstream default"
        );
    }

    #[test]
    fn normalize_config_rejects_non_json() {
        assert!(normalize_config("nope").is_err());
        assert!(
            normalize_config("[1,2,3]").is_err(),
            "an array is not a config"
        );
    }

    #[test]
    fn repo_id_is_the_published_one() {
        assert_eq!(REPO, "regolo/brick-modernbert-capability-classifier");
    }
}
