//! The PII tagger, running in-process on Candle.
//!
//! Model: `rizzoaiacademy/rizzo-pii-0.3B` — an mmBERT (ModernBERT architecture) encoder
//! with a token-classification head over 44 BIO labels covering 22 kinds of personal data.
//! The upstream project ships it as a desktop app with a local Flask server; pstore loads
//! the same checkpoint straight into its own process, so there is no server, no port and
//! no request leaving the machine.
//!
//! Structurally this is the same shape as the capability classifier — encoder, prediction
//! head, linear classifier — with one difference that matters: the head is applied **per
//! token** rather than to a pooled vector, because the output has to say *where* in the
//! text each entity is, not just that one is present.

use std::sync::{Mutex, OnceLock};

use candle_core::{DType, Device, Tensor};
use candle_nn::{LayerNorm, Linear, VarBuilder};
use candle_transformers::models::modernbert;
use tokenizers::Tokenizer;

use crate::models;
use crate::pii::{CHUNK_CHARS, Finding, TokenTag, decode_bio, segments};
use crate::router::capability::normalize_config;

/// Longest sequence handed to the encoder in one pass.
///
/// ModernBERT materialises a `seq × seq` attention mask, so this bounds memory as well as
/// time. A chunk that still tokenizes longer than this is split rather than truncated —
/// silently dropping the tail of a chunk would silently stop masking the data in it.
pub const MAX_TOKENS: usize = 1024;

/// How many times a chunk may be halved before the tail is given up on.
const MAX_SPLIT_DEPTH: usize = 8;

/// A loaded tagger.
pub struct Model {
    encoder: modernbert::ModernBert,
    head_dense: Linear,
    head_norm: LayerNorm,
    classifier: Linear,
    tokenizer: Tokenizer,
    /// Label per class index, from the checkpoint's `id2label`.
    labels: Vec<String>,
    device: Device,
}

enum State {
    Unloaded,
    Attempted(Box<Result<Model, String>>),
}

fn state() -> &'static Mutex<State> {
    static STATE: OnceLock<Mutex<State>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(State::Unloaded))
}

fn locked() -> std::sync::MutexGuard<'static, State> {
    match state().lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// Parse `id2label` into a label-per-index vector.
///
/// Reading the order from the file rather than assuming it is the same defensiveness the
/// capability classifier needs: a permuted label list would mask amounts as names.
pub fn parse_labels(config_json: &str) -> Result<Vec<String>, String> {
    let pairs = crate::router::capability::parse_id2label(config_json)?;
    let mut labels = vec![String::new(); pairs.len()];
    for (id, name) in &pairs {
        if *id >= labels.len() {
            return Err(format!(
                "id2label has index {id} but only {} labels",
                labels.len()
            ));
        }
        labels[*id] = name.clone();
    }
    if let Some(gap) = labels.iter().position(String::is_empty) {
        return Err(format!("id2label has no entry for class {gap}"));
    }
    if !labels.iter().any(|l| l == "O") {
        return Err("id2label has no outside (\"O\") class; this is not a BIO tagger".into());
    }
    Ok(labels)
}

impl Model {
    /// Download (or reuse the cache for) the checkpoint and build the tagger.
    ///
    /// Blocking; call from a worker thread.
    pub fn load(device: Device) -> Result<Self, String> {
        models::set(models::PII.id, models::Phase::Loading);
        match Self::build(device) {
            Ok(m) => {
                models::set(models::PII.id, models::Phase::Ready);
                Ok(m)
            }
            Err(e) => {
                models::set(models::PII.id, models::Phase::Failed(e.clone()));
                Err(e)
            }
        }
    }

    fn build(device: Device) -> Result<Self, String> {
        let repo = models::PII.repo;
        let fetch = crate::router::hub::fetch;
        let config_path = fetch(repo, "config.json")?;
        let tok_path = fetch(repo, "tokenizer.json")?;
        let weights_path = fetch(repo, "model.safetensors")?;

        let raw_config =
            std::fs::read_to_string(&config_path).map_err(|e| format!("reading config: {e}"))?;
        let labels = parse_labels(&raw_config)?;
        let config: modernbert::Config = serde_json::from_str(&normalize_config(&raw_config)?)
            .map_err(|e| format!("config.json is not a ModernBERT config: {e}"))?;

        let tokenizer = Tokenizer::from_file(&tok_path).map_err(|e| format!("tokenizer: {e}"))?;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)
                .map_err(|e| format!("loading weights: {e}"))?
        };

        let encoder = modernbert::ModernBert::load(vb.clone(), &config)
            .map_err(|e| format!("building encoder: {e}"))?;

        let hidden = config.hidden_size;
        let head_dense = candle_nn::linear_no_bias(hidden, hidden, vb.pp("head.dense"))
            .map_err(|e| format!("building head.dense: {e}"))?;
        let head_norm_weight = vb
            .pp("head.norm")
            .get(hidden, "weight")
            .map_err(|e| format!("building head.norm: {e}"))?;
        let head_norm = LayerNorm::new_no_bias(head_norm_weight, config.layer_norm_eps);
        let classifier = candle_nn::linear(hidden, labels.len(), vb.pp("classifier"))
            .map_err(|e| format!("building classifier: {e}"))?;

        Ok(Self {
            encoder,
            head_dense,
            head_norm,
            classifier,
            tokenizer,
            labels,
            device,
        })
    }

    /// Tag a whole prompt, one chunk at a time.
    ///
    /// Chunks are cut at paragraph, line or word boundaries, so an entity can in principle
    /// straddle a seam and come back as two findings ("Mario" then "Rossi"). Both are still
    /// masked, which is the property that matters; the placeholders just read as two.
    pub fn tag(&self, text: &str) -> Result<Vec<Finding>, String> {
        let mut out = Vec::new();
        for (offset, chunk) in segments(text, CHUNK_CHARS) {
            if chunk.trim().is_empty() {
                continue;
            }
            out.extend(self.tag_chunk(text, chunk, offset, 0)?);
        }
        Ok(out)
    }

    /// Tag one chunk, halving it if the tokenizer produces more than the encoder takes.
    fn tag_chunk(
        &self,
        full: &str,
        chunk: &str,
        offset: usize,
        depth: usize,
    ) -> Result<Vec<Finding>, String> {
        let encoding = self
            .tokenizer
            .encode(chunk, true)
            .map_err(|e| format!("tokenizing: {e}"))?;

        if encoding.get_ids().len() > MAX_TOKENS {
            if depth < MAX_SPLIT_DEPTH && chunk.chars().count() > 16 {
                let half = chunk
                    .char_indices()
                    .nth(chunk.chars().count() / 2)
                    .map(|(i, _)| i)
                    .unwrap_or(chunk.len());
                let mut found = self.tag_chunk(full, &chunk[..half], offset, depth + 1)?;
                found.extend(self.tag_chunk(full, &chunk[half..], offset + half, depth + 1)?);
                return Ok(found);
            }
            // Pathological input (a single enormous token-dense run). Report rather than
            // pretend the tail was checked.
            return Err(format!(
                "a {}-character run tokenized past the {MAX_TOKENS}-token limit and could \
                 not be split; that part of the prompt was not tagged",
                chunk.chars().count()
            ));
        }

        let n = encoding.get_ids().len();
        if n == 0 {
            return Ok(Vec::new());
        }
        let input = Tensor::from_vec(encoding.get_ids().to_vec(), (1, n), &self.device)
            .map_err(|e| format!("input tensor: {e}"))?;
        let attn = Tensor::from_vec(encoding.get_attention_mask().to_vec(), (1, n), &self.device)
            .map_err(|e| format!("mask tensor: {e}"))?;

        let hidden = self
            .encoder
            .forward(&input, &attn)
            .map_err(|e| format!("encoder forward: {e}"))?;

        // Per token, not pooled: the head runs on every position, then the classifier.
        let logits = hidden
            .apply(&self.head_dense)
            .and_then(|t| t.gelu_erf())
            .and_then(|t| t.apply(&self.head_norm))
            .and_then(|t| t.apply(&self.classifier))
            .and_then(|t| t.squeeze(0))
            .map_err(|e| format!("head forward: {e}"))?;
        let rows: Vec<Vec<f32>> = logits
            .to_dtype(DType::F32)
            .and_then(|t| t.to_vec2())
            .map_err(|e| format!("reading logits: {e}"))?;

        let specials = encoding.get_special_tokens_mask();
        let offsets = encoding.get_offsets();
        let mut tokens = Vec::with_capacity(rows.len());
        for (i, row) in rows.iter().enumerate() {
            // [CLS]/[SEP]/padding carry no span, and their labels are meaningless.
            if specials.get(i).copied().unwrap_or(0) == 1 {
                continue;
            }
            let (s, e) = match offsets.get(i) {
                Some((s, e)) if e > s => (*s, *e),
                _ => continue,
            };
            let (class, score) = argmax_softmax(row);
            let Some(label) = self.labels.get(class) else {
                continue;
            };
            tokens.push(TokenTag {
                start: offset + s,
                end: offset + e,
                label: label.clone(),
                score,
            });
        }

        // Decode against the whole prompt so spans are trimmed with the real neighbouring
        // characters, and the offsets returned are the ones the caller can slice.
        Ok(decode_bio(full, &tokens))
    }
}

/// The winning class and its softmax probability.
pub fn argmax_softmax(logits: &[f32]) -> (usize, f32) {
    let max = logits.iter().copied().fold(f32::MIN, f32::max);
    let (best, _) =
        logits.iter().enumerate().fold(
            (0usize, f32::MIN),
            |acc, (i, v)| {
                if *v > acc.1 { (i, *v) } else { acc }
            },
        );
    let sum: f32 = logits.iter().map(|z| (z - max).exp()).sum();
    let p = if sum > 0.0 && sum.is_finite() {
        (logits[best] - max).exp() / sum
    } else {
        0.0
    };
    (best, p)
}

/// Load the tagger, retrying on the CPU if the GPU refuses it.
fn load_anywhere() -> Result<Model, String> {
    let c = &models::PII;
    if !models::is_cached(c) {
        let why = format!(
            "{} not downloaded — open the Models window to fetch it ({})",
            c.title,
            c.size_label()
        );
        models::set(c.id, models::Phase::Absent);
        return Err(why);
    }
    let (dev, backend) = crate::router::device::pick();
    match Model::load(dev) {
        Ok(m) => Ok(m),
        Err(gpu) if backend.is_gpu() => Model::load(Device::Cpu)
            .map_err(|cpu| format!("on {backend}: {gpu}; and on CPU: {cpu}")),
        Err(e) => Err(e),
    }
}

/// Tag `text`, loading the model on first use.
pub fn tag(text: &str) -> Result<Vec<Finding>, String> {
    let mut guard = locked();
    if matches!(*guard, State::Unloaded) {
        *guard = State::Attempted(Box::new(load_anywhere()));
    }
    match &*guard {
        State::Attempted(result) => match &**result {
            Ok(model) => model.tag(text),
            Err(e) => Err(e.clone()),
        },
        State::Unloaded => Err("tagger not loaded".into()),
    }
}

/// Load the tagger now, without tagging anything.
pub fn preload() -> Result<(), String> {
    let mut guard = locked();
    if matches!(*guard, State::Unloaded) {
        *guard = State::Attempted(Box::new(load_anywhere()));
    }
    match &*guard {
        State::Attempted(result) => match &**result {
            Ok(_) => Ok(()),
            Err(e) => Err(e.clone()),
        },
        State::Unloaded => Err("tagger not loaded".into()),
    }
}

/// Drop the loaded tagger so the next pass builds it again.
pub fn reset() {
    *locked() = State::Unloaded;
    if matches!(models::phase(models::PII.id), models::Phase::Ready) {
        models::set(models::PII.id, models::Phase::Cached);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_are_read_in_checkpoint_order() {
        let json = r#"{"id2label": {"2": "O", "0": "B-FULLNAME", "1": "I-FULLNAME"}}"#;
        assert_eq!(
            parse_labels(json).unwrap(),
            vec!["B-FULLNAME", "I-FULLNAME", "O"]
        );
    }

    #[test]
    fn a_label_set_with_holes_or_no_outside_class_is_rejected() {
        // Missing class 1: reading class 1 as class 2 would mislabel every token after it.
        let gap = r#"{"id2label": {"0": "B-FULLNAME", "2": "O"}}"#;
        assert!(parse_labels(gap).is_err());
        // Not a BIO tagger at all.
        let no_o = r#"{"id2label": {"0": "coding", "1": "math_reasoning"}}"#;
        let err = parse_labels(no_o).unwrap_err();
        assert!(err.contains("BIO"), "got {err}");
        assert!(parse_labels("not json").is_err());
    }

    #[test]
    fn argmax_reports_the_winner_and_its_probability() {
        let (i, p) = argmax_softmax(&[0.0, 10.0, 1.0]);
        assert_eq!(i, 1);
        assert!(p > 0.99, "got {p}");

        // A flat distribution is honest about being unsure.
        let (_, p) = argmax_softmax(&[1.0, 1.0, 1.0, 1.0]);
        assert!((p - 0.25).abs() < 1e-5, "got {p}");

        // Degenerate logits must not produce NaN.
        for logits in [vec![f32::MIN; 3], vec![1e30, -1e30], vec![0.0]] {
            let (_, p) = argmax_softmax(&logits);
            assert!(p.is_finite(), "{logits:?} -> {p}");
        }
    }

    /// End-to-end against the real checkpoint. Ignored by default: it needs the 1.26 GB
    /// tagger on disk. Run with `cargo test -- --ignored pii --nocapture`.
    #[test]
    #[ignore = "needs the 1.26 GB PII checkpoint downloaded"]
    fn tagger_finds_a_name_and_an_address() {
        crate::models::download(&models::PII, &std::sync::atomic::AtomicBool::new(false))
            .expect("fetching the tagger");

        let text = "Il cliente Mario Rossi, residente in Via Garibaldi 12, Milano, \
                    ha scritto a mario.rossi@example.com riguardo al bonifico su \
                    IT60X0542811101000000123456.";
        let scan = crate::pii::sanitize(text);
        eprintln!(
            "used_model={} reason={:?} summary={}",
            scan.used_model,
            scan.fallback_reason,
            scan.plan.summary()
        );
        for item in &scan.plan.items {
            eprintln!(
                "  {:>18} {:?} {:.2} {}",
                item.finding.tag,
                item.finding.text,
                item.finding.score,
                item.finder_label()
            );
        }

        assert!(
            scan.used_model,
            "tagger did not run: {:?}",
            scan.fallback_reason
        );
        let tags: Vec<&str> = scan
            .plan
            .items
            .iter()
            .map(|i| i.finding.tag.as_str())
            .collect();
        assert!(tags.contains(&"FULLNAME"), "no name found in {tags:?}");
        assert!(tags.contains(&"EMAIL"), "no email found in {tags:?}");
        assert!(tags.contains(&"IBAN"), "no IBAN found in {tags:?}");

        let out = scan.plan.apply(text);
        assert!(!out.contains("Mario Rossi"), "name survived: {out}");
        assert!(
            !out.contains("mario.rossi@example.com"),
            "email survived: {out}"
        );
        assert!(
            !out.contains("IT60X0542811101000000123456"),
            "IBAN survived: {out}"
        );
        assert!(out.contains("[FULLNAME_1]"), "got {out}");
        // The request itself must still be readable.
        assert!(out.contains("bonifico"), "got {out}");

        // The house number is one value, not one placeholder per digit — the tagger splits
        // "12" into sub-word pieces and labels both `B-`, which `join_touching` repairs.
        let numbers: Vec<&str> = scan
            .plan
            .items
            .iter()
            .filter(|i| i.finding.tag == "BUILDINGNUM")
            .map(|i| i.finding.text.as_str())
            .collect();
        assert!(
            numbers.iter().all(|n| n.len() > 1) || numbers.is_empty(),
            "a house number came back in pieces: {numbers:?}"
        );
        assert!(
            !out.contains("[BUILDINGNUM_2]"),
            "one house number should need one placeholder: {out}"
        );
    }
}
