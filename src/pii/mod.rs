//! Masking personal data out of a prompt, on this machine.
//!
//! A prompt written against real work tends to carry real details — a client's name, an
//! address in a contract, an IBAN in a payment bug. Handing that to a coding agent hands
//! it to whatever runs behind the agent. This module finds those details and replaces each
//! one with a stable, type-carrying placeholder — `[FULLNAME_1]`, `[IBAN_1]` — so the
//! prompt still reads as the same request while the values stay local.
//!
//! Detection is the local model's job, via [`crate::router::llm::detect_pii`]. Earlier
//! versions also carried checksum-validated patterns (IBAN mod-97, Luhn, codice fiscale,
//! partita IVA) as a second, always-on detector. Those are gone: with a 27B model doing the
//! reading, maintaining a parallel implementation — and the precedence rules for when the
//! two disagreed — cost more than it caught.
//!
//! Nothing is rewritten without being shown first: [`sanitize`] returns a [`Plan`], every
//! finding in it can be turned off individually, and applying it is one undo step over a
//! file whose previous text is already in version history.

use std::collections::HashMap;
use std::time::Duration;

/// Every tag the model may return.
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
    /// Model confidence.
    pub score: f32,
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
    /// Overlaps still happen with one detector — the model can return both `Mario` and
    /// `Mario Rossi` for the same person — and replacing both would corrupt the text. The
    /// longer span wins, because a partial mask leaks the rest of the value.
    pub fn build(mut findings: Vec<Finding>) -> Self {
        findings.retain(|f| f.end > f.start && !f.text.trim().is_empty());
        findings.sort_by(|a, b| {
            a.start
                .cmp(&b.start)
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

/// What one sanitisation pass found.
#[derive(Debug, Clone)]
pub struct Scan {
    /// The proposed replacements.
    pub plan: Plan,
    /// How long the pass took, model startup included.
    pub elapsed: Duration,
}

impl Scan {
    /// Where the findings came from, for the review window's header.
    pub fn source_label(&self) -> String {
        format!("local model · {:.1}s", self.elapsed.as_secs_f32())
    }
}

/// Longest chunk of text handed to the model in one pass, in characters.
///
/// A prompt can be long, and every chunk costs a whole `llama-cli` invocation — weights
/// mapped, generated, exited — so this is a balance rather than a limit: small enough that
/// the fitted context window stays cheap, large enough that an ordinary prompt is one call.
/// Offsets are translated back, so the caller never sees the seam.
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
/// Blocking — the model is a subprocess that maps 7.17 GB of weights before it answers, so
/// call this from a worker thread.
///
/// Returns the reason on failure rather than an empty plan. An empty plan means "nothing
/// personal in this prompt", which is a claim the sanitiser must not make when it never
/// actually looked — the caller shows the error and leaves the prompt alone.
pub fn sanitize(text: &str) -> Result<Scan, String> {
    let started = std::time::Instant::now();
    let findings = model_pass(text)?;
    Ok(Scan {
        plan: Plan::build(findings),
        elapsed: started.elapsed(),
    })
}

fn model_pass(text: &str) -> Result<Vec<Finding>, String> {
    crate::router::llm::detect_pii(text)
}

/// Re-check the model on the next pass.
pub fn reset() {
    crate::router::llm::reset();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finding(tag: &str, start: usize, end: usize, text: &str) -> Finding {
        Finding {
            tag: tag.into(),
            start,
            end,
            text: text.into(),
            score: 0.9,
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
            finding("FULLNAME", 0, 5, "Mario"),
            finding("FULLNAME", 15, 20, "Mario"),
            finding("FULLNAME", 27, 32, "Luigi"),
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
    fn overlapping_findings_resolve_to_one_span() {
        let text = "IBAN IT60X0542811101000000123456 ok";
        // The model returned the same identifier twice, once clipped short. Masking both
        // would splice the text; masking the shorter one would leak the tail.
        let plan = Plan::build(vec![
            finding("IBAN", 5, 27, "IT60X05428111010000001"),
            finding("IBAN", 5, 32, "IT60X0542811101000000123456"),
        ]);
        assert_eq!(plan.items.len(), 1, "one span, not two: {:?}", plan.items);
        assert_eq!(plan.apply(text), "IBAN [IBAN_1] ok");
    }

    #[test]
    fn the_longer_of_two_model_spans_wins() {
        // A partial mask would leak the rest of the name.
        let text = "call Mario Rossi now";
        let plan = Plan::build(vec![
            finding("FULLNAME", 5, 10, "Mario"),
            finding("FULLNAME", 5, 16, "Mario Rossi"),
        ]);
        assert_eq!(plan.items.len(), 1);
        assert_eq!(plan.apply(text), "call [FULLNAME_1] now");
    }

    #[test]
    fn disabled_findings_are_left_in_the_text() {
        let text = "on 2024-01-01 Mario paid";
        let mut plan = Plan::build(vec![
            finding("DATE", 3, 13, "2024-01-01"),
            finding("FULLNAME", 14, 19, "Mario"),
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
        let plan = Plan::build(vec![finding("FULLNAME", 100, 120, "Mario")]);
        // The text was edited since the scan: the span no longer exists.
        assert_eq!(plan.apply("short"), "short");

        // A span that starts inside a multi-byte character must be ignored, not sliced.
        // In "società" the final "à" occupies bytes 6..8, so 7 is mid-character.
        let text = "società";
        let plan = Plan::build(vec![finding("ORG", 7, 8, "?")]);
        assert_eq!(plan.apply(text), text);
    }

    #[test]
    fn empty_and_whitespace_findings_are_dropped() {
        let plan = Plan::build(vec![
            finding("FULLNAME", 0, 0, ""),
            finding("FULLNAME", 2, 3, "  "),
            finding("FULLNAME", 5, 3, "backwards"),
        ]);
        assert!(plan.items.is_empty(), "got {:?}", plan.items);
        assert_eq!(plan.summary(), "no personal data found");
    }

    #[test]
    fn summary_reads_as_a_sentence() {
        let mut plan = Plan::build(vec![
            finding("FULLNAME", 0, 5, "Mario"),
            finding("EMAIL", 6, 20, "m@example.com"),
            finding("FULLNAME", 21, 26, "Luigi"),
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

    /// An empty prompt must not cost a model invocation: `llama-cli` would map 7.17 GB of
    /// weights to find nothing in nothing.
    #[test]
    fn nothing_to_scan_is_not_a_model_call() {
        assert!(segments("", CHUNK_CHARS).is_empty());
    }
}
