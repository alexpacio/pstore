//! Shrinking a prompt without losing what makes it work.
//!
//! The constraint set below is the whole point: compression that drops a file path, a
//! version number or an error string has not shrunk the prompt, it has broken it. So
//! the instruction is explicit about what must survive byte-for-byte, and the result
//! is shown as a diff for approval before it touches the document.

/// The compression instruction.
///
/// Written as "preserve these, collapse those" rather than "be shorter", because a
/// bare "shorten this" reliably eats the specifics a coding agent needs.
pub const INSTRUCTION: &str = "\
Rewrite the prompt below so it is shorter while remaining exactly as useful to a \
coding agent.

Preserve verbatim, without exception:
- every code block and inline code span
- every file path, directory name, and glob
- every identifier: function, type, variable, module, crate, and package names
- every version number, error message, log line, and command
- every explicit constraint, requirement, and acceptance criterion
- the markdown structure: headings, lists, and code fences

Remove only:
- repetition and restatement
- filler, hedging, and pleasantries
- background the agent can infer from the code itself
- verbose phrasing that a shorter phrase covers exactly

Do not add new facts, requirements, examples, or interpretation. Do not answer the \
prompt or act on it. Output only the rewritten prompt, with no preamble, no commentary, \
and no code fence wrapping the whole thing.";

/// Compose the request sent to an agent.
pub fn compose(document: &str) -> String {
    format!("{INSTRUCTION}\n\n---\n\n{document}")
}

/// Strip anything an agent wrapped around the rewritten prompt.
///
/// Agents sometimes ignore "no preamble" and lead with a sentence, or fence the whole
/// document despite being told not to. Both are recoverable; silently keeping them
/// would corrupt the prompt.
pub fn clean(response: &str) -> String {
    let mut text = response.trim();

    // Drop a leading conversational line if a fence or heading follows it.
    if let Some((first, rest)) = text.split_once('\n') {
        let f = first.trim();
        let looks_conversational = f.ends_with(':')
            && f.split_whitespace().count() <= 12
            && !f.starts_with('#')
            && !f.starts_with("```");
        if looks_conversational {
            text = rest.trim();
        }
    }

    // Unwrap a fence that encloses the entire response.
    if text.starts_with("```")
        && let Some(after_open) = text.find('\n')
    {
        let body = &text[after_open + 1..];
        if let Some(close) = body.rfind("```") {
            // Only unwrap when the closing fence really is at the end.
            if body[close + 3..].trim().is_empty() {
                return body[..close].trim_end().to_string();
            }
        }
    }
    text.to_string()
}

/// What changed, for the approval dialog.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Savings {
    /// Characters before.
    pub before_chars: usize,
    /// Characters after.
    pub after_chars: usize,
    /// Words before.
    pub before_words: usize,
    /// Words after.
    pub after_words: usize,
}

impl Savings {
    /// Measure a proposed rewrite.
    pub fn measure(before: &str, after: &str) -> Self {
        Self {
            before_chars: before.chars().count(),
            after_chars: after.chars().count(),
            before_words: before.split_whitespace().count(),
            after_words: after.split_whitespace().count(),
        }
    }

    /// Percentage of characters removed. Negative when the rewrite grew.
    pub fn percent_saved(&self) -> f32 {
        if self.before_chars == 0 {
            return 0.0;
        }
        let delta = self.before_chars as f32 - self.after_chars as f32;
        delta / self.before_chars as f32 * 100.0
    }

    /// Rough token delta, at the usual ~4 characters per token.
    pub fn approx_tokens_saved(&self) -> i64 {
        (self.before_chars as i64 - self.after_chars as i64) / 4
    }

    /// Whether the rewrite is worth offering.
    pub fn worthwhile(&self) -> bool {
        self.after_chars < self.before_chars && self.percent_saved() >= 2.0
    }

    /// One-line summary for the dialog.
    pub fn summary(&self) -> String {
        format!(
            "{} → {} chars ({:.0}% smaller, ~{} tokens), {} → {} words",
            self.before_chars,
            self.after_chars,
            self.percent_saved(),
            self.approx_tokens_saved(),
            self.before_words,
            self.after_words
        )
    }
}

/// Whether a token looks like a file path or filename.
///
/// Deliberately stricter than "contains a dot": a sentence-final `README.` or an `e.g.`
/// must not read as a file reference, or the integrity check reports paths that were
/// never there.
pub fn looks_like_path(token: &str) -> bool {
    let t = token.trim_matches(|c: char| {
        !c.is_ascii_alphanumeric() && c != '/' && c != '.' && c != '_' && c != '-'
    });
    if t.len() < 3 || t.starts_with("http") {
        return false;
    }
    if t.contains('/') && t.chars().any(|c| c.is_ascii_alphanumeric()) {
        return true;
    }
    // `name.ext` with a plausible extension.
    match t.rsplit_once('.') {
        Some((stem, ext)) => {
            !stem.is_empty()
                && (1..=5).contains(&ext.len())
                && ext.chars().all(|c| c.is_ascii_alphanumeric())
        }
        None => false,
    }
}

/// The path a token refers to, with surrounding punctuation removed.
///
/// `looks_like_path` tolerates a trailing `.` because `retry.rs.` at the end of a sentence
/// is still a path. Comparing the two texts needs the *same* string on both sides, though,
/// and the rewrite will not have kept the sentence — so the trailing dot has to go before
/// anything is compared, or every path that ended a sentence reads as dropped.
pub fn path_token(token: &str) -> Option<String> {
    if !looks_like_path(token) {
        return None;
    }
    let trimmed = token
        .trim_matches(|c: char| {
            !c.is_ascii_alphanumeric() && c != '/' && c != '.' && c != '_' && c != '-'
        })
        .trim_end_matches('.');
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

/// Things that must appear in the rewrite as often as in the original.
///
/// A cheap structural check before showing the diff: if a code fence or a file path
/// went missing, the rewrite broke the prompt regardless of how good the prose looks.
pub fn integrity_warnings(before: &str, after: &str) -> Vec<String> {
    let mut warnings = Vec::new();

    let fences_before = before.matches("```").count() / 2;
    let fences_after = after.matches("```").count() / 2;
    if fences_after < fences_before {
        warnings.push(format!(
            "{} of {fences_before} code blocks were dropped",
            fences_before - fences_after
        ));
    }

    let paths = |s: &str| -> Vec<String> { s.split_whitespace().filter_map(path_token).collect() };
    let before_paths = paths(before);
    let after_paths = paths(after);
    let missing: Vec<_> = before_paths
        .iter()
        .filter(|p| !after_paths.contains(p))
        .cloned()
        .collect();
    if !missing.is_empty() {
        warnings.push(format!("file references dropped: {}", missing.join(", ")));
    }

    warnings
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bug this guards: a path that ended a sentence in the original but not in the
    /// rewrite would otherwise be reported as dropped on every single shrink.
    #[test]
    fn path_tokens_normalise_trailing_punctuation() {
        assert_eq!(path_token("src/main.rs"), Some("src/main.rs".into()));
        assert_eq!(path_token("src/main.rs."), Some("src/main.rs".into()));
        assert_eq!(path_token("`src/main.rs`,"), Some("src/main.rs".into()));
        assert_eq!(path_token("word"), None);
        assert_eq!(path_token("e.g."), None);
    }

    #[test]
    fn path_detection_ignores_ordinary_punctuation() {
        assert!(looks_like_path("src/main.rs"));
        assert!(looks_like_path("config.rs"));
        assert!(
            looks_like_path("src/store/version.rs,"),
            "trailing comma is stripped"
        );
        assert!(looks_like_path("`src/lib.rs`"), "backticks are stripped");

        // The bug this guards: prose that merely ends a sentence.
        assert!(!looks_like_path("README."));
        assert!(!looks_like_path("e.g."));
        assert!(!looks_like_path("word"));
        assert!(!looks_like_path("it."));
        assert!(
            !looks_like_path("https://example.com/a.rs"),
            "URLs are not local files"
        );
    }

    #[test]
    fn instruction_names_what_must_survive() {
        for must in [
            "verbatim",
            "file path",
            "identifier",
            "version number",
            "constraint",
        ] {
            assert!(INSTRUCTION.contains(must), "instruction omits {must:?}");
        }
        assert!(INSTRUCTION.contains("Do not add new facts"));
        assert!(INSTRUCTION.contains("Do not answer the prompt"));
    }

    #[test]
    fn compose_puts_the_document_after_a_separator() {
        let out = compose("# My prompt");
        assert!(out.starts_with(INSTRUCTION));
        assert!(out.contains("\n---\n"));
        assert!(out.ends_with("# My prompt"));
    }

    #[test]
    fn clean_passes_through_a_well_behaved_response() {
        let text = "# Task\n\nRefactor `parse()` in src/lib.rs.";
        assert_eq!(clean(text), text);
    }

    #[test]
    fn clean_strips_a_leading_preamble_line() {
        let out = clean("Here is the shortened prompt:\n\n# Task\n\nDo the thing.");
        assert!(out.starts_with("# Task"), "got {out:?}");
    }

    #[test]
    fn clean_unwraps_a_whole_document_fence() {
        let out = clean("```markdown\n# Task\n\nDo the thing.\n```");
        assert_eq!(out, "# Task\n\nDo the thing.");
    }

    #[test]
    fn clean_keeps_internal_code_fences_intact() {
        // The prompt legitimately contains a fenced snippet; unwrapping it would
        // destroy the prompt.
        let text = "# Task\n\nFix this:\n\n```rust\nfn main() {}\n```\n\nThen run tests.";
        assert_eq!(clean(text), text, "an internal fence is not a wrapper");
    }

    #[test]
    fn clean_does_not_strip_a_heading_that_ends_with_a_colon() {
        let text = "# Steps:\n\nFirst do this.";
        assert_eq!(clean(text), text);
    }

    #[test]
    fn clean_does_not_strip_a_long_first_line() {
        let text = "This prompt asks the agent to do a great many things in sequence and ends with a colon:\nnext line";
        assert_eq!(clean(text), text, "long lines are content, not preamble");
    }

    #[test]
    fn savings_measures_and_summarises() {
        let s = Savings::measure("aaaa bbbb cccc", "aaaa bbbb");
        assert_eq!(s.before_chars, 14);
        assert_eq!(s.after_chars, 9);
        assert_eq!(s.before_words, 3);
        assert_eq!(s.after_words, 2);
        assert!(s.percent_saved() > 30.0);
        assert!(s.worthwhile());
        assert!(s.summary().contains("14 → 9 chars"));
    }

    #[test]
    fn a_rewrite_that_grew_is_not_worthwhile() {
        let s = Savings::measure("short", "much longer than before");
        assert!(!s.worthwhile());
        assert!(s.percent_saved() < 0.0);
        assert!(s.approx_tokens_saved() < 0);
    }

    #[test]
    fn a_trivial_saving_is_not_worthwhile() {
        let before = "x".repeat(1000);
        let after = "x".repeat(995);
        assert!(
            !Savings::measure(&before, &after).worthwhile(),
            "0.5% is not worth a diff"
        );
    }

    #[test]
    fn empty_input_does_not_divide_by_zero() {
        let s = Savings::measure("", "");
        assert_eq!(s.percent_saved(), 0.0);
        assert!(!s.worthwhile());
    }

    #[test]
    fn integrity_check_flags_dropped_code_blocks() {
        let before = "Fix:\n```rust\nfn a() {}\n```\nand\n```rust\nfn b() {}\n```";
        let after = "Fix:\n```rust\nfn a() {}\n```\nand fn b";
        let warnings = integrity_warnings(before, after);
        assert_eq!(warnings.len(), 1);
        assert!(
            warnings[0].contains("1 of 2 code blocks"),
            "got {warnings:?}"
        );
    }

    #[test]
    fn integrity_check_flags_dropped_file_paths() {
        let before = "Update src/main.rs and src/lib.rs to match.";
        let after = "Update src/main.rs to match.";
        let warnings = integrity_warnings(before, after);
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("src/lib.rs"), "got {warnings:?}");
    }

    #[test]
    fn integrity_check_is_quiet_on_a_faithful_rewrite() {
        let before = "Please could you kindly update src/main.rs, and I would also \
                      really appreciate it if you ran the tests afterwards.";
        let after = "Update src/main.rs, then run the tests.";
        assert!(integrity_warnings(before, after).is_empty());
    }
}
