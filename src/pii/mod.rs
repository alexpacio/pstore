//! Masking personal data out of a prompt, on this machine.
//!
//! A prompt written against real work tends to carry real details — a client's name, an
//! address in a contract, an IBAN in a payment bug. Handing that to a coding agent hands
//! it to whatever runs behind the agent. This module finds those details and replaces each
//! one with a stable, type-carrying placeholder — `[FULLNAME_1]`, `[IBAN_1]` — so the
//! prompt still reads as the same request while the values stay local.
//!
//! Two detectors, deliberately different in kind:
//!
//! * [`model`] — `rizzoaiacademy/rizzo-pii-0.3B`, a ModernBERT token classifier over 22
//!   tags, which is what finds the things only context identifies: names, streets,
//!   organisations. It runs in this process on Candle, like the routing classifiers.
//! * [`detect`] — patterns whose match is confirmed by a **checksum**: IBAN, credit card,
//!   codice fiscale, VAT number, plus plain email and URL shapes. No download needed, and
//!   arithmetic beats a model's guess when the two disagree.
//!
//! Nothing is rewritten without being shown first: [`sanitize`] returns a [`Plan`], every
//! finding in it can be turned off individually, and applying it is one undo step over a
//! file whose previous text is already in version history.

pub mod detect;
#[cfg(feature = "candle")]
pub mod model;

use std::collections::HashMap;
use std::time::Duration;

/// The 22 tags the tagger can emit, plus the two this module adds by pattern.
///
/// Kept as a list so the UI can offer them all and a checkpoint that changes its label set
/// is noticed rather than silently mapped onto nothing.
pub const TAGS: [&str; 23] = [
    "FULLNAME",
    "AGE",
    "GENDER",
    "DATE",
    "TIME",
    "STREET",
    "BUILDINGNUM",
    "ZIPCODE",
    "CITY",
    "PROVINCE",
    "EMAIL",
    "TELEPHONENUM",
    "CF",
    "PIVA",
    "ID_DOC",
    "IBAN",
    "CREDITCARDNUMBER",
    "AMOUNT",
    "TARGA",
    "ORG",
    "DOCID",
    "CATASTO",
    "URL",
];

/// Whether a tag is masked unless the user says otherwise.
///
/// Four are off by default. A prompt for a coding agent is full of dates, times, amounts
/// and documentation links that are part of the *request*, not personal data — masking
/// them by default would damage far more prompts than it would protect. They stay
/// detected, listed, and one click away.
pub fn tag_masked_by_default(tag: &str) -> bool {
    !matches!(tag, "DATE" | "TIME" | "AMOUNT" | "URL")
}

/// Which detector produced a finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Finder {
    /// The neural tagger.
    Model,
    /// A pattern, confirmed by a checksum where the identifier has one.
    Pattern,
}

impl Finder {
    /// Label for the review window.
    pub fn label(self) -> &'static str {
        match self {
            Finder::Model => "tagger",
            Finder::Pattern => "checked pattern",
        }
    }

    /// Findings that survived arithmetic take precedence over findings that did not.
    fn precedence(self) -> u8 {
        match self {
            Finder::Pattern => 1,
            Finder::Model => 0,
        }
    }
}

/// One span of the prompt identified as personal data.
#[derive(Debug, Clone, PartialEq)]
pub struct Finding {
    /// Tag from [`TAGS`].
    pub tag: String,
    /// Byte offset of the span's start in the source text.
    pub start: usize,
    /// Byte offset of the span's end.
    pub end: usize,
    /// The matched text.
    pub text: String,
    /// Model confidence, or 1.0 for a checksum-confirmed match.
    pub score: f32,
    /// Which detector found it.
    pub finder: Finder,
}

/// One finding, with the placeholder that would replace it.
#[derive(Debug, Clone, PartialEq)]
pub struct Masked {
    /// What was found.
    pub finding: Finding,
    /// What it would be replaced with, e.g. `[FULLNAME_2]`.
    pub placeholder: String,
    /// Whether to actually replace it. Starts from [`tag_masked_by_default`].
    pub masked: bool,
}

/// A reviewed, non-overlapping set of replacements.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Plan {
    /// Findings in document order.
    pub items: Vec<Masked>,
}

impl Plan {
    /// Resolve raw findings into a plan: drop overlaps, then name each distinct value.
    ///
    /// Overlaps are real — the tagger tags an IBAN *and* the pattern confirms it — and
    /// replacing both would corrupt the text. The checksum-confirmed span wins; between
    /// two of a kind, the longer one does, because a partial mask leaks the rest.
    pub fn build(mut findings: Vec<Finding>) -> Self {
        findings.retain(|f| f.end > f.start && !f.text.trim().is_empty());
        findings.sort_by(|a, b| {
            a.start
                .cmp(&b.start)
                .then(b.finder.precedence().cmp(&a.finder.precedence()))
                .then((b.end - b.start).cmp(&(a.end - a.start)))
                .then(b.score.total_cmp(&a.score))
        });

        let mut kept: Vec<Finding> = Vec::with_capacity(findings.len());
        for f in findings {
            if kept.last().is_some_and(|prev| f.start < prev.end) {
                continue;
            }
            kept.push(f);
        }

        // Numbering is per tag and per distinct value, so the same person keeps the same
        // placeholder throughout the prompt — which is what lets the rewritten text still
        // make sense to read.
        let mut counters: HashMap<&str, usize> = HashMap::new();
        let mut assigned: HashMap<(String, String), String> = HashMap::new();
        let items = kept
            .into_iter()
            .map(|finding| {
                let key = (finding.tag.clone(), finding.text.clone());
                let placeholder = assigned
                    .entry(key)
                    .or_insert_with(|| {
                        let n = counters.entry(tag_key(&finding.tag)).or_insert(0);
                        *n += 1;
                        format!("[{}_{}]", finding.tag, n)
                    })
                    .clone();
                Masked {
                    masked: tag_masked_by_default(&finding.tag),
                    placeholder,
                    finding,
                }
            })
            .collect();
        Self { items }
    }

    /// Apply every enabled replacement, returning the sanitised text.
    pub fn apply(&self, text: &str) -> String {
        let mut out = String::with_capacity(text.len());
        let mut cursor = 0usize;
        for item in self.items.iter().filter(|i| i.masked) {
            // Ignore anything that does not line up with the text handed in: applying a
            // stale plan must not panic or splice bytes mid-character.
            if item.finding.start < cursor
                || item.finding.end > text.len()
                || !text.is_char_boundary(item.finding.start)
                || !text.is_char_boundary(item.finding.end)
            {
                continue;
            }
            out.push_str(&text[cursor..item.finding.start]);
            out.push_str(&item.placeholder);
            cursor = item.finding.end;
        }
        out.push_str(&text[cursor.min(text.len())..]);
        out
    }

    /// Counts per tag over the enabled findings, strongest first, for the summary line.
    pub fn counts(&self) -> Vec<(&str, usize)> {
        let mut counts: Vec<(&str, usize)> = Vec::new();
        for item in self.items.iter().filter(|i| i.masked) {
            match counts.iter_mut().find(|(t, _)| *t == item.finding.tag) {
                Some((_, n)) => *n += 1,
                None => counts.push((item.finding.tag.as_str(), 1)),
            }
        }
        counts.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
        counts
    }

    /// How many findings would be replaced.
    pub fn enabled(&self) -> usize {
        self.items.iter().filter(|i| i.masked).count()
    }

    /// One-line summary for the status bar.
    pub fn summary(&self) -> String {
        if self.items.is_empty() {
            return "no personal data found".into();
        }
        let enabled = self.enabled();
        if enabled == 0 {
            return format!("{} finding(s), all switched off", self.items.len());
        }
        let detail = self
            .counts()
            .iter()
            .map(|(tag, n)| format!("{n}× {tag}"))
            .collect::<Vec<_>>()
            .join(", ");
        format!("{enabled} to mask — {detail}")
    }

    /// Turn every finding of `tag` on or off.
    pub fn set_tag(&mut self, tag: &str, masked: bool) {
        for item in self.items.iter_mut().filter(|i| i.finding.tag == tag) {
            item.masked = masked;
        }
    }
}

/// A tag's key in the placeholder counters. Borrowed from [`TAGS`] where possible so the
/// counter map does not allocate per finding.
fn tag_key(tag: &str) -> &'static str {
    TAGS.iter().find(|t| **t == tag).copied().unwrap_or("OTHER")
}

impl Masked {
    /// Which detector found this, for the review table.
    pub fn finder_label(&self) -> &'static str {
        self.finding.finder.label()
    }
}

/// One token, as the tagger classified it.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenTag {
    /// Byte offset of the token's start in the source text.
    pub start: usize,
    /// Byte offset of the token's end.
    pub end: usize,
    /// The winning label, e.g. `B-FULLNAME`, `I-FULLNAME` or `O`.
    pub label: String,
    /// Probability of that label.
    pub score: f32,
}

/// Stitch per-token BIO labels into whole-entity findings.
///
/// Tolerant on purpose. A token classifier will occasionally open an entity with `I-` or
/// switch tag mid-entity, and the useful reading of that is "an entity is here" rather
/// than "discard it": under-masking is the failure that matters. Spans are trimmed of
/// surrounding whitespace, because a tokenizer that attaches the preceding space to a word
/// would otherwise mask the space too and quietly change the text's shape.
///
/// A second pass then joins same-tag findings that touch with **no character between them**
/// — see [`join_touching`]. That is not cosmetic: the tagger splits a house number like
/// `12` into the sub-word pieces `1` and `2` and labels both `B-BUILDINGNUM`, which the BIO
/// rules alone turn into two entities and two placeholders for one value.
pub fn decode_bio(text: &str, tokens: &[TokenTag]) -> Vec<Finding> {
    let mut out: Vec<Finding> = Vec::new();
    let mut current: Option<(String, usize, usize, f32, usize)> = None;

    let flush = |current: &mut Option<(String, usize, usize, f32, usize)>,
                 out: &mut Vec<Finding>| {
        let Some((tag, start, end, total, n)) = current.take() else {
            return;
        };
        let Some((start, end)) = trim_span(text, start, end) else {
            return;
        };
        out.push(Finding {
            tag,
            start,
            end,
            text: text[start..end].to_string(),
            score: if n > 0 { total / n as f32 } else { 0.0 },
            finder: Finder::Model,
        });
    };

    for t in tokens {
        let (prefix, tag) = match t.label.split_once('-') {
            Some((p, tag)) if p == "B" || p == "I" => (p, tag),
            // "O", or a label with no prefix: nothing to continue.
            _ => {
                flush(&mut current, &mut out);
                continue;
            }
        };
        match &mut current {
            Some((open_tag, _, end, total, n)) if prefix == "I" && open_tag == tag => {
                *end = t.end.max(*end);
                *total += t.score;
                *n += 1;
            }
            _ => {
                flush(&mut current, &mut out);
                current = Some((tag.to_string(), t.start, t.end, t.score, 1));
            }
        }
    }
    flush(&mut current, &mut out);
    join_touching(text, out)
}

/// Merge consecutive same-tag findings with nothing at all between them.
///
/// Sub-word pieces of one value ("1" and "2" of a house number `12`) come back as separate
/// entities when the model labels each piece `B-`. Two genuinely separate entities are
/// separated by at least a space or a comma in the text, so requiring a zero-length gap
/// after trimming keeps "Mario Rossi" and "Luigi Bianchi" apart while joining `1` and `2`.
fn join_touching(text: &str, findings: Vec<Finding>) -> Vec<Finding> {
    let mut out: Vec<Finding> = Vec::with_capacity(findings.len());
    for f in findings {
        match out.last_mut() {
            Some(prev) if prev.tag == f.tag && prev.end == f.start => {
                // Weight each piece's score by its length, so a long confident run is not
                // dragged down by a one-character tail.
                let (a, b) = ((prev.end - prev.start) as f32, (f.end - f.start) as f32);
                prev.score = (prev.score * a + f.score * b) / (a + b);
                prev.end = f.end;
                prev.text = text
                    .get(prev.start..prev.end)
                    .map(str::to_string)
                    .unwrap_or_else(|| std::mem::take(&mut prev.text));
            }
            _ => out.push(f),
        }
    }
    out
}

/// Narrow `start..end` to the text it actually contains, or `None` if it is all
/// whitespace or does not line up with `text`.
fn trim_span(text: &str, start: usize, end: usize) -> Option<(usize, usize)> {
    if end > text.len()
        || start >= end
        || !text.is_char_boundary(start)
        || !text.is_char_boundary(end)
    {
        return None;
    }
    let slice = &text[start..end];
    let lead = slice.len() - slice.trim_start().len();
    let trail = slice.len() - slice.trim_end().len();
    let (start, end) = (start + lead, end - trail);
    (start < end).then_some((start, end))
}

/// What one sanitisation pass found, and with what.
#[derive(Debug, Clone)]
pub struct Scan {
    /// The proposed replacements.
    pub plan: Plan,
    /// Whether the neural tagger contributed.
    pub used_model: bool,
    /// Why the tagger did not, when it did not. The checksum detectors always run, so
    /// this is a reduction in coverage rather than a failure.
    pub fallback_reason: Option<String>,
    /// How long the pass took.
    pub elapsed: Duration,
}

impl Scan {
    /// Where the findings came from, for the review window's header.
    pub fn source_label(&self) -> String {
        if self.used_model {
            format!(
                "local tagger + checked patterns · {:.1}s",
                self.elapsed.as_secs_f32()
            )
        } else {
            "checked patterns only".into()
        }
    }
}

/// Longest chunk of text handed to the tagger in one pass, in characters.
///
/// The encoder's attention is quadratic in sequence length, and a prompt can be long, so
/// the text is split on paragraph boundaries and each piece classified separately. Offsets
/// are translated back, so the caller never sees the seam.
pub const CHUNK_CHARS: usize = 1600;

/// Split `text` into chunks of at most `max_chars` characters, preferring paragraph, then
/// line, then word boundaries. Returns each chunk with its byte offset in `text`.
pub fn segments(text: &str, max_chars: usize) -> Vec<(usize, &str)> {
    let mut out = Vec::new();
    let mut start = 0usize;
    let max_chars = max_chars.max(1);

    while start < text.len() {
        let rest = &text[start..];
        let Some((limit, _)) = rest.char_indices().nth(max_chars) else {
            out.push((start, rest));
            break;
        };
        let window = &rest[..limit];
        // Only take a break that leaves a chunk worth classifying; a paragraph break in
        // the first few characters would otherwise produce a pass per line.
        let floor = limit / 2;
        let cut = [
            window.rfind("\n\n").map(|i| i + 2),
            window.rfind('\n').map(|i| i + 1),
            window.rfind(' ').map(|i| i + 1),
        ]
        .into_iter()
        .flatten()
        .find(|c| *c > floor)
        .unwrap_or(limit);

        out.push((start, &rest[..cut]));
        start += cut;
    }
    out
}

/// Find personal data in `text`.
///
/// The checksum detectors always run. The tagger runs too when its weights are on disk;
/// otherwise the reason is reported and the pass carries on with what it has. Blocking on
/// the first call (the model has to load) — call from a worker thread.
pub fn sanitize(text: &str) -> Scan {
    let started = std::time::Instant::now();
    let (tagged, used_model, fallback_reason) = model_pass(text);
    let mut findings = detect::scan(text);
    findings.extend(tagged);

    Scan {
        plan: Plan::build(findings),
        used_model,
        fallback_reason,
        elapsed: started.elapsed(),
    }
}

/// Run the tagger, if this build has one and its weights are on disk.
///
/// Returns what it found, whether it ran, and why not when it didn't.
#[cfg(feature = "candle")]
fn model_pass(text: &str) -> (Vec<Finding>, bool, Option<String>) {
    match model::tag(text) {
        Ok(found) => (found, true, None),
        Err(e) => (Vec::new(), false, Some(e)),
    }
}

/// Without the `candle` feature there is no tagger, so the checked patterns are the whole
/// pass — reduced coverage, reported as such rather than presented as a clean result.
#[cfg(not(feature = "candle"))]
fn model_pass(_text: &str) -> (Vec<Finding>, bool, Option<String>) {
    (
        Vec::new(),
        false,
        Some("built without the `candle` feature".to_string()),
    )
}

/// Forget the loaded tagger, so the next pass loads it again.
pub fn reset() {
    #[cfg(feature = "candle")]
    model::reset();
}

/// Load the tagger now rather than on the next pass. Blocking.
pub fn preload() -> Result<(), String> {
    #[cfg(feature = "candle")]
    {
        model::preload()
    }
    #[cfg(not(feature = "candle"))]
    {
        Err("built without the `candle` feature".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finding(tag: &str, start: usize, end: usize, text: &str, finder: Finder) -> Finding {
        Finding {
            tag: tag.into(),
            start,
            end,
            text: text.into(),
            score: 0.9,
            finder,
        }
    }

    #[test]
    fn every_tag_is_known_and_defaults_are_deliberate() {
        assert_eq!(TAGS.len(), 23);
        assert!(TAGS.contains(&"FULLNAME") && TAGS.contains(&"IBAN"));
        // The four that would otherwise wreck ordinary technical prompts.
        for off in ["DATE", "TIME", "AMOUNT", "URL"] {
            assert!(!tag_masked_by_default(off), "{off} should start off");
        }
        for on in ["FULLNAME", "EMAIL", "IBAN", "CF", "CREDITCARDNUMBER", "ORG"] {
            assert!(tag_masked_by_default(on), "{on} should start on");
        }
    }

    #[test]
    fn placeholders_are_stable_per_value_and_numbered_per_tag() {
        let text = "Mario wrote to Mario about Luigi";
        let plan = Plan::build(vec![
            finding("FULLNAME", 0, 5, "Mario", Finder::Model),
            finding("FULLNAME", 15, 20, "Mario", Finder::Model),
            finding("FULLNAME", 27, 32, "Luigi", Finder::Model),
        ]);

        assert_eq!(plan.items[0].placeholder, "[FULLNAME_1]");
        assert_eq!(
            plan.items[1].placeholder, "[FULLNAME_1]",
            "the same person must keep the same placeholder"
        );
        assert_eq!(plan.items[2].placeholder, "[FULLNAME_2]");
        assert_eq!(
            plan.apply(text),
            "[FULLNAME_1] wrote to [FULLNAME_1] about [FULLNAME_2]"
        );
    }

    #[test]
    fn overlapping_findings_resolve_to_the_checked_one() {
        let text = "IBAN IT60X0542811101000000123456 ok";
        // The tagger tagged a wider, sloppier span; the pattern confirmed the exact one.
        let plan = Plan::build(vec![
            finding("IBAN", 5, 30, "IT60X05428111010000001", Finder::Model),
            Finding {
                tag: "IBAN".into(),
                start: 5,
                end: 32,
                text: "IT60X0542811101000000123456".into(),
                score: 1.0,
                finder: Finder::Pattern,
            },
        ]);
        assert_eq!(plan.items.len(), 1, "one span, not two: {:?}", plan.items);
        assert_eq!(plan.items[0].finder_label(), "checked pattern");
        assert_eq!(plan.apply(text), "IBAN [IBAN_1] ok");
    }

    #[test]
    fn the_longer_of_two_model_spans_wins() {
        // A partial mask would leak the rest of the name.
        let text = "call Mario Rossi now";
        let plan = Plan::build(vec![
            finding("FULLNAME", 5, 10, "Mario", Finder::Model),
            finding("FULLNAME", 5, 16, "Mario Rossi", Finder::Model),
        ]);
        assert_eq!(plan.items.len(), 1);
        assert_eq!(plan.apply(text), "call [FULLNAME_1] now");
    }

    #[test]
    fn disabled_findings_are_left_in_the_text() {
        let text = "on 2024-01-01 Mario paid";
        let mut plan = Plan::build(vec![
            finding("DATE", 3, 13, "2024-01-01", Finder::Model),
            finding("FULLNAME", 14, 19, "Mario", Finder::Model),
        ]);
        // DATE starts off by default.
        assert_eq!(plan.apply(text), "on 2024-01-01 [FULLNAME_1] paid");
        assert_eq!(plan.enabled(), 1);

        plan.set_tag("DATE", true);
        assert_eq!(plan.apply(text), "on [DATE_1] [FULLNAME_1] paid");
        plan.set_tag("FULLNAME", false);
        assert_eq!(plan.apply(text), "on [DATE_1] Mario paid");
    }

    #[test]
    fn applying_a_stale_plan_cannot_panic_or_corrupt() {
        let plan = Plan::build(vec![finding("FULLNAME", 100, 120, "Mario", Finder::Model)]);
        // The text was edited since the scan: the span no longer exists.
        assert_eq!(plan.apply("short"), "short");

        // A span that starts inside a multi-byte character must be ignored, not sliced.
        // In "società" the final "à" occupies bytes 6..8, so 7 is mid-character.
        let text = "società";
        let plan = Plan::build(vec![finding("ORG", 7, 8, "?", Finder::Model)]);
        assert_eq!(plan.apply(text), text);
    }

    #[test]
    fn empty_and_whitespace_findings_are_dropped() {
        let plan = Plan::build(vec![
            finding("FULLNAME", 0, 0, "", Finder::Model),
            finding("FULLNAME", 2, 3, "  ", Finder::Model),
            finding("FULLNAME", 5, 3, "backwards", Finder::Model),
        ]);
        assert!(plan.items.is_empty(), "got {:?}", plan.items);
        assert_eq!(plan.summary(), "no personal data found");
    }

    #[test]
    fn summary_reads_as_a_sentence() {
        let mut plan = Plan::build(vec![
            finding("FULLNAME", 0, 5, "Mario", Finder::Model),
            finding("EMAIL", 6, 20, "m@example.com", Finder::Pattern),
            finding("FULLNAME", 21, 26, "Luigi", Finder::Model),
        ]);
        let s = plan.summary();
        assert!(s.contains("3 to mask"), "got {s}");
        assert!(s.contains("2× FULLNAME"), "got {s}");
        assert_eq!(plan.counts()[0], ("FULLNAME", 2), "commonest tag first");

        plan.items.iter_mut().for_each(|i| i.masked = false);
        assert!(plan.summary().contains("all switched off"));
    }

    #[test]
    fn segments_split_on_paragraph_boundaries_and_cover_the_text() {
        let text = "first paragraph here\n\nsecond paragraph here\n\nthird one";
        let segs = segments(text, 25);
        assert!(segs.len() > 1, "should have split: {segs:?}");
        // Every byte is accounted for exactly once, in order.
        let mut rebuilt = String::new();
        let mut expected_start = 0;
        for (start, chunk) in &segs {
            assert_eq!(*start, expected_start, "chunks must be contiguous");
            assert_eq!(&text[*start..*start + chunk.len()], *chunk);
            rebuilt.push_str(chunk);
            expected_start += chunk.len();
        }
        assert_eq!(rebuilt, text, "segmentation must be lossless");
    }

    #[test]
    fn segments_handle_multibyte_text_and_words_with_no_breaks() {
        // No whitespace at all: the split has to fall back to the character limit, and
        // must not land mid-character.
        let text = "é".repeat(100);
        let segs = segments(&text, 10);
        assert_eq!(segs.len(), 10);
        assert_eq!(
            segs.iter().map(|(_, c)| c.len()).sum::<usize>(),
            text.len(),
            "lossless"
        );
        for (_, chunk) in &segs {
            assert_eq!(chunk.chars().count(), 10);
        }

        // Short text is one chunk.
        assert_eq!(segments("short", 100), vec![(0, "short")]);
        assert!(segments("", 100).is_empty());
        // A zero limit must not loop forever.
        assert!(!segments("abc", 0).is_empty());
    }

    #[test]
    fn a_scan_with_no_model_still_finds_checked_identifiers() {
        let text = "Il cliente scrive da mario@example.com, IBAN IT60X0542811101000000123456.";
        let plan = Plan::build(detect::scan(text));
        let out = plan.apply(text);
        assert!(out.contains("[EMAIL_1]"), "got {out}");
        assert!(out.contains("[IBAN_1]"), "got {out}");
        assert!(!out.contains("mario@example.com"));
        assert!(!out.contains("0542811101000000123456"));
    }

    fn token(start: usize, end: usize, label: &str, score: f32) -> TokenTag {
        TokenTag {
            start,
            end,
            label: label.into(),
            score,
        }
    }

    #[test]
    fn bio_tokens_stitch_into_whole_entities() {
        let text = "call Mario Rossi today";
        let found = decode_bio(
            text,
            &[
                token(0, 4, "O", 0.99),
                token(5, 10, "B-FULLNAME", 0.9),
                token(10, 16, "I-FULLNAME", 0.8),
                token(17, 22, "O", 0.99),
            ],
        );
        assert_eq!(found.len(), 1, "got {found:?}");
        assert_eq!(found[0].text, "Mario Rossi");
        assert_eq!(found[0].tag, "FULLNAME");
        assert_eq!(found[0].finder, Finder::Model);
        // The score is the mean over the entity's tokens.
        assert!(
            (found[0].score - 0.85).abs() < 1e-6,
            "got {}",
            found[0].score
        );
    }

    /// The real failure this fixes, taken from a run against the live checkpoint: in
    /// "Via Garibaldi 12" the tagger splits `12` into the pieces `1` and `2` and opens
    /// both with `B-BUILDINGNUM`, which used to produce two findings and two placeholders
    /// for one house number.
    #[test]
    fn subword_pieces_of_one_value_become_one_finding() {
        let text = "Via Garibaldi 12, Milano";
        let found = decode_bio(
            text,
            &[
                token(0, 3, "B-STREET", 0.99),
                token(3, 13, "I-STREET", 0.99),
                token(13, 15, "B-BUILDINGNUM", 1.0),
                token(15, 16, "B-BUILDINGNUM", 0.68),
                token(16, 17, "O", 0.99),
                token(17, 24, "B-CITY", 0.99),
            ],
        );
        let summary: Vec<(&str, &str)> = found
            .iter()
            .map(|f| (f.tag.as_str(), f.text.as_str()))
            .collect();
        assert_eq!(
            summary,
            vec![
                ("STREET", "Via Garibaldi"),
                ("BUILDINGNUM", "12"),
                ("CITY", "Milano"),
            ],
            "got {found:?}"
        );
        // The merged score is length-weighted, so the confident first piece dominates.
        let number = &found[1];
        assert!(
            number.score > 0.8 && number.score < 1.0,
            "got {}",
            number.score
        );

        let plan = Plan::build(found);
        assert_eq!(plan.apply(text), "[STREET_1] [BUILDINGNUM_1], [CITY_1]");
    }

    #[test]
    fn two_adjacent_entities_do_not_merge() {
        let text = "Mario Luigi";
        let found = decode_bio(
            text,
            &[
                token(0, 5, "B-FULLNAME", 0.9),
                token(5, 11, "B-FULLNAME", 0.9),
            ],
        );
        assert_eq!(found.len(), 2, "a new B- starts a new entity: {found:?}");
        assert_eq!(found[0].text, "Mario");
        assert_eq!(found[1].text, "Luigi");
    }

    #[test]
    fn a_tag_change_mid_entity_splits_it() {
        let text = "Roma Milano";
        let found = decode_bio(
            text,
            &[token(0, 4, "B-CITY", 0.9), token(4, 11, "I-PROVINCE", 0.7)],
        );
        assert_eq!(found.len(), 2);
        assert_eq!(found[0].tag, "CITY");
        assert_eq!(found[1].tag, "PROVINCE");
        assert_eq!(found[1].text, "Milano", "leading space must be trimmed");
    }

    #[test]
    fn an_entity_opened_with_i_is_still_reported() {
        // Under-masking is the failure that matters, so a stray I- is not discarded.
        let text = "Mario";
        let found = decode_bio(text, &[token(0, 5, "I-FULLNAME", 0.6)]);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].text, "Mario");
    }

    #[test]
    fn whitespace_only_and_out_of_range_tokens_are_dropped() {
        let text = "Mario  ";
        assert!(decode_bio(text, &[token(5, 7, "B-FULLNAME", 0.9)]).is_empty());
        assert!(decode_bio(text, &[token(0, 99, "B-FULLNAME", 0.9)]).is_empty());
        // A span starting mid-character must not be sliced: "à" is bytes 6..8.
        assert!(decode_bio("società", &[token(7, 8, "B-ORG", 0.9)]).is_empty());
    }

    #[test]
    fn an_entity_at_the_very_end_is_flushed() {
        let text = "scrive Mario";
        let found = decode_bio(
            text,
            &[token(0, 6, "O", 0.9), token(6, 12, "B-FULLNAME", 0.9)],
        );
        assert_eq!(found.len(), 1, "the last entity must not be lost");
        assert_eq!(found[0].text, "Mario");
    }
}
