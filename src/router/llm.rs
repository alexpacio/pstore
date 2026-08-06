//! The one place that runs the model.
//!
//! Every local inference in pstore — ranking the agents, finding personal data — is a
//! `llama-cli` invocation made from here. The module is deliberately the only thing that
//! knows the command line, so the flags that bound memory (see [`fit_context`]) cannot be
//! forgotten at one call site and applied at another.
//!
//! **One process per call.** There is no resident server: `llama-cli` starts, maps the
//! weights, generates, and exits. That costs seconds of startup on every call, which is
//! affordable because each call is a button press the user already expects to wait for,
//! and it buys a great deal — no port to bind, no health-check, no orphaned process
//! outliving the app, no HTTP client. The rule that keeps it affordable is **one
//! invocation per user action**; anything that needs more should be one prompt instead.
//!
//! **Output is grammar-constrained.** `--json-schema` compiles the schema into the
//! sampler, so the model cannot emit anything that does not parse. Parsing is therefore a
//! `serde_json` call rather than a best-effort scrape, and a malformed reply is a bug in
//! the schema rather than a Tuesday.

use std::path::PathBuf;
use std::process::Command;

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
/// and the 3.8 GB of weights is essentially the entire footprint.
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
fn ready() -> Result<(PathBuf, PathBuf), String> {
    let prefs = crate::config::prefs_snapshot();

    let rt = runtime::locate(prefs.llama_cli_path.as_deref())
        .ok_or_else(|| runtime::missing_reason(prefs.llama_cli_path.as_deref()))?;

    if !models::is_cached(&models::LLM) {
        models::set(models::LLM.id, models::Phase::Absent);
        return Err(format!(
            "{} not downloaded — open the Models window to fetch it ({})",
            models::LLM.title,
            models::LLM.size_label()
        ));
    }
    let weights = hub_path()?;
    Ok((rt.path, weights))
}

/// Resolve the checkpoint's path in the shared Hugging Face cache.
fn hub_path() -> Result<PathBuf, String> {
    let file = models::LLM
        .files
        .last()
        .expect("the checkpoint lists its weights");
    super::hub::cached(models::LLM.repo, file)
}

/// Run the model once and return its (schema-constrained) JSON reply.
///
/// Blocking, and slow by design — the weights are mapped on every call. `max_output_tokens`
/// bounds both generation and the context window, so it should reflect what the schema can
/// actually produce rather than a round number.
pub fn complete_json(
    prompt: &str,
    schema: &Value,
    max_output_tokens: usize,
) -> Result<Value, String> {
    let (binary, weights) = ready()?;
    let ceiling = crate::config::prefs_snapshot().model_context_ceiling;
    let ctx = fit_context(prompt, max_output_tokens, ceiling);

    models::set(models::LLM.id, models::Phase::Loading);
    let output = Command::new(&binary)
        .arg("-m")
        .arg(&weights)
        .arg("--json-schema")
        .arg(schema.to_string())
        .arg("-p")
        .arg(prompt)
        // Echoing the prompt back would double the output for nothing.
        .arg("--no-display-prompt")
        // Closed stdin is what makes it exit. Even `llama-completion` drops into a prompt
        // after generating; with no stdin it takes the EOF and leaves. `--log-disable` is
        // deliberately *not* used — it silences the load errors on stderr too, which are
        // the only diagnostic when a run fails.
        .stdin(std::process::Stdio::null())
        .args(["-n", &max_output_tokens.to_string()])
        .args(["-c", &ctx.to_string()])
        // Memory: 4-bit KV cache and flash attention. Small at these context sizes, but
        // free, and they keep the ceiling honest if someone raises it.
        .args(["--cache-type-k", "q4_0", "--cache-type-v", "q4_0"])
        .args(["--flash-attn", "on"])
        // Nothing warms up or gets pinned: this process exists for one generation.
        .args(["--no-warmup", "-ngl", "999"])
        // Deterministic: the same prompt should rank the same way twice.
        .args(["--temp", "0", "--seed", "1"])
        .output()
        .map_err(|e| {
            models::set(models::LLM.id, models::Phase::Failed(e.to_string()));
            format!("running {}: {e}", binary.display())
        })?;

    if !output.status.success() {
        let why = String::from_utf8_lossy(&output.stderr);
        let why = why.trim();
        // The tail is the useful part; the head is load-time chatter about tensors.
        let tail: String = why.lines().rev().take(4).collect::<Vec<_>>().join(" / ");
        models::set(models::LLM.id, models::Phase::Failed(tail.clone()));
        return Err(format!("llama-cli failed: {tail}"));
    }

    models::set(models::LLM.id, models::Phase::Ready);
    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_reply(&stdout)
}

/// Pull the JSON object out of `llama-cli`'s stdout.
///
/// The grammar guarantees the *model's* output parses, but the process still prints its own
/// framing around it, and which framing depends on build flags. Rather than depend on that,
/// take the outermost braces.
fn parse_reply(stdout: &str) -> Result<Value, String> {
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

/// Ranking asks for five objects of three small fields. 64 tokens each is generous.
const RANK_OUTPUT_TOKENS: usize = 400;

/// Rank `candidates` against `text`.
///
/// The model chooses **by index** into the list it was given, never by naming an agent or a
/// model string. An index either maps onto a real launch configuration or is rejected; a
/// name could be plausible and wrong, and pstore would then try to launch it.
pub fn rank(
    text: &str,
    candidates: &[super::Candidate],
    excluded: Vec<(&'static str, String)>,
) -> Result<super::Ranking, String> {
    let started = std::time::Instant::now();
    let want = super::SHORTLIST.min(candidates.len());

    let schema = json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["choices"],
        "properties": {
            "choices": {
                "type": "array",
                "minItems": want,
                "maxItems": want,
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["index", "fit", "reason"],
                    "properties": {
                        "index": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": candidates.len() - 1
                        },
                        "fit": {"type": "integer", "minimum": 0, "maximum": 100},
                        "reason": {"type": "string", "maxLength": 90}
                    }
                }
            }
        }
    });

    let reply = complete_json(
        &rank_prompt(text, candidates, want),
        &schema,
        RANK_OUTPUT_TOKENS,
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
        let _ = write!(
            list,
            "\n{i}: {} via {} · {} tier · effort={}{} · relative price {:.1}x{}",
            c.model_display,
            c.agent_display,
            c.tier,
            c.effort,
            if c.effort_selectable {
                ""
            } else {
                " (predicted)"
            },
            c.relative_price,
            if c.metered {
                " · BILLED PER TOKEN"
            } else {
                " · included in subscription"
            },
        );
    }

    format!(
        "You are choosing which coding-agent model should answer a prompt.\n\
         \n\
         Rank the {want} best options from the numbered list below. Judge how well each \
         option's capability matches what the prompt actually demands.\n\
         \n\
         Guidance:\n\
         - Match capability to demand. Do not reach for a frontier model on simple work, \
         and do not send a hard multi-file refactor to a light one.\n\
         - Higher effort buys depth at the cost of latency. Prefer lower effort unless the \
         prompt needs the reasoning.\n\
         - Options marked BILLED PER TOKEN cost real money on top of the subscription. \
         Rank one highly only if it is clearly better than every included option for this \
         prompt, not merely equal.\n\
         - `fit` is 0-100: how well that option suits THIS prompt.\n\
         - `reason` is at most 12 words, concrete, and about this prompt.\n\
         \n\
         Options:{list}\n\
         \n\
         The prompt to route:\n\
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
        let reply = complete_json(&pii_prompt(chunk), &schema, PII_OUTPUT_TOKENS)?;
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
    ready().map(|_| {
        models::set(models::LLM.id, models::Phase::Cached);
    })
}

/// Re-check the runtime and checkpoint on the next call.
///
/// Nothing is cached in this process — each call re-resolves and re-spawns — so this only
/// has to correct the status board, which may be showing a stale failure.
pub fn reset() {
    models::probe_cache();
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

    /// End-to-end against the real checkpoint and the real runtime. Ignored by default —
    /// it needs the 3.8 GB download and takes ~10s per call.
    ///
    /// `cargo test -- --ignored live_model --nocapture`
    #[test]
    #[ignore = "needs the downloaded checkpoint and a provisioned runtime"]
    fn live_model_ranks_and_finds_personal_data() {
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
        assert!(p.contains("BILLED PER TOKEN"));
        assert!(p.contains("included in subscription"));
        assert!(p.contains("0: sonnet"), "options must be indexed from zero");
        assert!(p.contains("1: fable"));
        assert!(p.contains("Rank the 2 best"));
    }
}
