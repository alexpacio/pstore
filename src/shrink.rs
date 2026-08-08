//! Shrinking a prompt without losing what makes it work.
//!
//! The target register is telegraphic — caveman speech. Articles, copulas and politeness
//! carry no instruction, so they go; `fix the retry loop that is in src/jobs.rs, and please
//! keep the poll at 20 ms` becomes `fix retry loop in src/jobs.rs, keep 20 ms poll`. Prose
//! written for a person is mostly connective tissue, and an agent does not need it.
//!
//! What makes that safe rather than lossy is that the local model does the cutting. A
//! mechanical stripper — drop every `the`, every `is`, every word on a stop list — cannot
//! tell `the retry loop` from `the 20 ms poll must not change`, and it cannot see that a
//! `the` inside a code span is code. The model reads the whole passage and decides per
//! word, which is the only way "shorter" and "still means the same thing" hold at once.
//!
//! The constraint set below is the other half: compression that drops a file path, a
//! version number or an error string has not shrunk the prompt, it has broken it. So the
//! instruction is explicit about what must survive byte-for-byte, [`integrity_warnings`]
//! checks the structural part of that afterwards, and the result is shown as a diff for
//! approval before it touches the document.
//!
//! It runs on this machine, on the same checkpoint as ranking and the personal-data scan —
//! no coding agent is involved and the prompt does not leave. [`run`] is the entry point;
//! it is blocking and belongs on a worker thread.

use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

/// The compression instruction.
///
/// Written as "preserve these, collapse those" rather than "be shorter", because a
/// bare "shorten this" reliably eats the specifics a coding agent needs. The last line
/// is the one that keeps the register honest: telegraphic is only a win while the agent
/// would still do the same thing, and a word whose removal changes that is not filler.
pub const INSTRUCTION: &str = "\
Rewrite the prompt below in the fewest words a coding agent can still act on. Write it \
telegraphically, as instructions rather than prose.

Compress by:
- dropping articles (a, an, the) and copulas (is, are, was, will be) where the sense survives
- dropping pleasantries, hedging, and self-reference: please, I would like you to, I think, \
if possible
- turning sentences into imperative fragments: \"fix retry loop in src/jobs.rs, keep 20 ms poll\"
- stating each fact once, and cutting restatement, summary, and background the agent can \
infer from the code itself
- merging sentences that share a subject, while keeping one bullet per requirement

Copy verbatim, without exception:
- every code block and inline code span, compressing nothing inside them
- every file path, directory name, and glob
- every identifier: function, type, variable, module, crate, and package names
- every version number, error message, log line, and command
- every explicit constraint, requirement, and acceptance criterion
- the markdown structure: headings, lists, and code fences

Ambiguity is not compression: keep any word whose removal would change what the agent does, \
including negations, conditions, and the subject of an instruction.

Do not add new facts, requirements, examples, or interpretation. Do not answer the prompt or \
act on it. Return only the rewritten prompt, with no preamble, no commentary, and no code \
fence wrapping the whole thing.";

/// Compose the request sent to the local model.
pub fn compose(document: &str) -> String {
    format!("{INSTRUCTION}\n\n---\n\n{document}")
}

/// Shrink a whole prompt, one model call per chunk.
///
/// Blocking, and slow in units of seconds per chunk — the model is a subprocess that maps
/// the weights before it answers — so call it from a worker thread. `note` reports progress
/// for the status bar, and `cancel` is honoured between chunks, which is the only place a
/// pass can be interrupted: a generation already in flight runs to completion.
///
/// Returns the reason on failure rather than a partial document. Half a shrunk prompt is
/// not a shorter prompt, it is a truncated one, and it must never reach the diff.
pub fn run(
    text: &str,
    cancel: &AtomicBool,
    note: &mut dyn FnMut(String),
) -> Result<String, String> {
    let pieces = chunks(text, crate::router::llm::shrink_chunk_chars());
    let total = pieces.len();
    let mut out = String::with_capacity(text.len());

    // One load of the weights for the whole pass, sized for the longest chunk in it. Opening this
    // per chunk would pay the load again for every part of a long document.
    let widest = pieces.iter().map(|(body, _)| body.len()).max().unwrap_or(0);
    let pass = crate::router::llm::ShrinkPass::open(widest)?;

    for (n, (body, separator)) in pieces.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled".into());
        }
        // Whitespace between chunks, and any run of blank lines, has nothing to compress.
        if body.trim().is_empty() {
            out.push_str(body);
            out.push_str(separator);
            continue;
        }
        if total > 1 {
            note(format!("shrinking… part {} of {total}", n + 1));
        }
        let rewritten = pass.chunk(body)?;
        if rewritten.trim().is_empty() {
            return Err(format!(
                "the model returned nothing for part {} of {total}",
                n + 1
            ));
        }
        out.push_str(&rewritten);
        out.push_str(separator);
    }

    Ok(out)
}

/// How far a chunk may exceed `max_chars` to keep a fenced code block whole.
///
/// [`crate::router::llm::shrink_chunk_chars`] leaves this much room under the context
/// window on purpose, so a stretched chunk still fits rather than being silently truncated
/// by llama.cpp.
const STRETCH: usize = 4; // max_chars + max_chars / STRETCH

/// Split `text` into pieces small enough for one model call.
///
/// Each piece is `(body, separator)`: the text to rewrite, and the whitespace that followed
/// it in the original. Rewrites are joined back with those separators, so a paragraph break
/// between two chunks survives a pass that neither chunk knew about.
///
/// Cuts prefer a blank line **outside** a fenced code block, and a fence that would be split
/// is either pushed whole into the next chunk or kept whole by stretching this one. A chunk
/// boundary inside a fence would hand the model an unterminated code block — the one thing
/// the instruction insists it copy byte-for-byte, presented in a shape that no longer looks
/// like code. A single fenced block longer than `max_chars + max_chars / 4` is the one case
/// that cannot be honoured; it is split like prose, and [`integrity_warnings`] is what
/// reports the damage if the model then mangles it.
pub fn chunks(text: &str, max_chars: usize) -> Vec<(&str, &str)> {
    let max = max_chars.max(1);
    let (breaks, fences) = layout(text);
    let mut out = Vec::new();
    let mut start = 0usize;

    while start < text.len() {
        let rest = &text[start..];
        let end = match rest.char_indices().nth(max) {
            // What is left fits in one call.
            None => text.len(),
            Some((limit, _)) => {
                let hard = start + limit;
                // Only take a break that leaves a chunk worth the call; a blank line in the
                // first few characters would otherwise produce a pass per paragraph.
                let floor = start + limit / 2;
                let paragraph = breaks
                    .iter()
                    .rev()
                    .copied()
                    .find(|p| *p > floor && *p <= hard);

                match (
                    paragraph,
                    fences.iter().find(|(a, b)| (*a..*b).contains(&hard)),
                ) {
                    (Some(p), _) => p,
                    // The fence opens inside this chunk: end the chunk where it opens and
                    // let the block start the next one.
                    (None, Some((a, _))) if *a > start => *a,
                    // The chunk already begins in the block, so the only way to keep it
                    // whole is to carry it to its close.
                    (None, Some((_, b))) if *b <= start + max + max / STRETCH => *b,
                    (None, _) => fallback_cut(rest, limit) + start,
                }
            }
        };
        let piece = &text[start..end];
        let body = piece.trim_end();
        out.push((body, &piece[body.len()..]));
        start = end;
    }
    out
}

/// Where the document may be cut, and where it may not.
///
/// Returns the offsets just past a blank line outside any code block, and the byte ranges of
/// the fenced blocks themselves. An unterminated fence runs to the end of the document,
/// which is what the markdown renderer does with it too.
fn layout(text: &str) -> (Vec<usize>, Vec<(usize, usize)>) {
    let mut breaks = Vec::new();
    let mut fences = Vec::new();
    let mut open: Option<usize> = None;
    let mut offset = 0usize;

    for line in text.split_inclusive('\n') {
        let start = offset;
        offset += line.len();

        if line.trim_start().starts_with("```") {
            match open.take() {
                Some(a) => fences.push((a, offset)),
                None => open = Some(start),
            }
            continue;
        }
        if open.is_none() && line.trim().is_empty() {
            breaks.push(offset);
        }
    }
    if let Some(a) = open {
        fences.push((a, text.len()));
    }
    (breaks, fences)
}

/// Where to cut when no blank line falls in the window.
///
/// A line break first, then a word break, then the window edge — which is a character
/// boundary because the caller found it with `char_indices`. Prose with no paragraph breaks
/// still has to be split somewhere, and mid-word is the only unacceptable answer.
fn fallback_cut(rest: &str, limit: usize) -> usize {
    let window = &rest[..limit];
    let floor = limit / 2;
    [
        window.rfind('\n').map(|i| i + 1),
        window.rfind(' ').map(|i| i + 1),
    ]
    .into_iter()
    .flatten()
    .find(|c| *c > floor)
    .unwrap_or(limit)
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
    // A slash alone is not a path. English uses it as "or" — `difficulty/capability`,
    // `read/write`, `and/or` — and treating those as file references means warning that a
    // path was dropped on rewrites that dropped nothing, on almost every prompt. A real
    // reference either carries an extension (handled below) or is anchored: rooted at `/`,
    // relative with `./`, under `~/`, or ending in `/` as a directory.
    let anchored = t.starts_with('/')
        || t.starts_with("./")
        || t.starts_with("../")
        || t.starts_with("~/")
        || t.ends_with('/');
    if anchored && t.chars().any(|c| c.is_ascii_alphanumeric()) {
        return true;
    }
    /// `name.ext` with a plausible extension.
    fn has_extension(s: &str) -> bool {
        match s.rsplit_once('.') {
            Some((stem, ext)) => {
                !stem.is_empty()
                    && (1..=5).contains(&ext.len())
                    && ext.chars().all(|c| c.is_ascii_alphanumeric())
                    // A decimal number is not a file. `4.10`, `2.1` and `99.2` all satisfy
                    // stem-dot-extension, and prompts carrying measurements — latencies,
                    // percentages, version-less figures — are full of them, so without this
                    // every such number reads as a dropped file reference.
                    && !(stem.bytes().all(|b| b.is_ascii_digit())
                        && ext.bytes().all(|b| b.is_ascii_digit()))
            }
            None => false,
        }
    }

    if t.contains('/') {
        // The extension lives on the final segment, and a sentence-final `.` is discounted
        // first: `edit src/main.rs.` names a path. Safe to strip here precisely because the
        // slash has already ruled out the abbreviations the trailing dot is guarding against.
        let core = t.trim_end_matches('.');
        return has_extension(core.rsplit('/').next().unwrap_or(core));
    }
    // No slash: strict, and the trailing dot is load-bearing. `e.g.` and a sentence-final
    // `README.` must not read as file references.
    has_extension(t)
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

/// Every path token in `text`, in the order it appears.
fn path_tokens(text: &str) -> Vec<String> {
    text.split_whitespace().filter_map(path_token).collect()
}

/// A token of the rewrite with its surrounding punctuation removed, whatever it turns out to be.
///
/// Deliberately *not* [`path_token`]. That one answers "is this a file reference?", and it is
/// strict on purpose — a bare `README.` at the end of a sentence must not read as one. Here the
/// question is only "does the rewrite still name the path we already found in the original", so
/// the strictness has nothing to do and does real harm: it is what made `Update main.rs.` fail
/// to satisfy a reference to `src/main.rs`, the rewrite having put it at the end of a sentence.
/// One pass of trimming is not enough here: `` `src/main.rs`. `` ends in a character the
/// punctuation trim keeps (`.`), so the backtick behind it is never reached and the token comes
/// out as ``src/main.rs` ``, which matches nothing. So the two trims alternate until neither
/// takes anything. [`looks_like_path`] deliberately does *not* do this — there the trailing dot
/// is load-bearing, and eating it would make `e.g.` a file called `g`.
fn bare_token(token: &str) -> Option<&str> {
    let mut t = token;
    loop {
        let next = t
            .trim_matches(|c: char| {
                !c.is_ascii_alphanumeric() && c != '/' && c != '.' && c != '_' && c != '-'
            })
            .trim_end_matches('.');
        if next == t {
            break;
        }
        t = next;
    }
    (!t.is_empty()).then_some(t)
}

/// Whether `wanted` is still named anywhere in `after`.
///
/// Not token equality, because the two texts tokenise differently and a rewrite is allowed to
/// re-punctuate. The live failure: a prompt naming `Next.js` three times was rewritten to
/// `React/Next.js`, and comparing whole whitespace tokens reported the reference dropped —
/// from a rewrite whose very first line contains it.
///
/// So a reference counts as kept when either side is the other's trailing path segment:
/// `Next.js` is satisfied by `React/Next.js`, and `src/main.rs` by `main.rs`. It stays a
/// *segment* boundary rather than a plain substring, so `a/config.rs` and `b/config.rs` remain
/// different files and `next.js` does not answer for `xnext.js` — which is the whole point of
/// the check.
fn still_named(wanted: &str, after: &str) -> bool {
    after.split_whitespace().filter_map(bare_token).any(|p| {
        p == wanted || p.ends_with(&format!("/{wanted}")) || wanted.ends_with(&format!("/{p}"))
    })
}

/// File references in `before` that `after` no longer names, each reported once.
///
/// Shared with [`crate::plan::warnings`], which asks the same question of a plan and had the
/// same two bugs when it asked it itself.
///
/// **Presence, not frequency.** Deduplicated because a repeat is not a second file: reporting
/// `Next.js, Next.js, Next.js` reads as three dropped references when it is one, and the count
/// it repeated was the count in the *original*, which says nothing about what the rewrite kept.
/// Counting occurrences would be wrong here anyway — stating each fact once is exactly what
/// shrink is for, so a path mentioned three times and kept once is a success, not a warning.
pub fn dropped_paths(before: &str, after: &str) -> Vec<String> {
    let mut missing: Vec<String> = Vec::new();
    for p in path_tokens(before) {
        if !still_named(&p, after) && !missing.contains(&p) {
            missing.push(p);
        }
    }
    missing
}

/// Things that must still be in the rewrite.
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

    let missing = dropped_paths(before, after);
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

        // The bug this guards: a slash is English for "or" at least as often as it is a
        // path separator. Reading these as file references made `pstore plan` warn that a
        // path had been dropped on prompts that never named one.
        for prose in ["difficulty/capability", "read/write", "and/or", "GUI/TUI"] {
            assert!(!looks_like_path(prose), "{prose} is not a path");
        }

        // The bug this guards: a measurement has the shape of `name.ext`. Telemetry pasted
        // into a prompt is mostly numbers, and reading them as files made every rewrite
        // report a dozen dropped paths that were never paths.
        for number in ["4.10", "2.1", "99.2", "0.01", "44.2"] {
            assert!(!looks_like_path(number), "{number} is a number, not a path");
        }
        // A version-like name is still a file, and a numeric stem is fine with a real
        // extension — only the all-digits-both-sides case is a measurement.
        assert!(looks_like_path("v2.rs"), "numeric stem, real extension");
        assert!(looks_like_path("2026.log"));

        // Still paths, without an extension to prove it.
        assert!(looks_like_path("/usr/local/bin"), "rooted");
        assert!(looks_like_path("./scripts"), "explicitly relative");
        assert!(
            looks_like_path("src/agents/"),
            "trailing slash is a directory"
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

    /// The register is the feature. An instruction that only says "shorter" produces prose
    /// that is 10% shorter; the words below are what produce instructions instead.
    #[test]
    fn instruction_asks_for_the_telegraphic_register() {
        for must in ["telegraphically", "articles", "copulas", "imperative"] {
            assert!(INSTRUCTION.contains(must), "instruction omits {must:?}");
        }
        assert!(
            INSTRUCTION.contains("Ambiguity is not compression"),
            "the limit on how far to compress has to be stated"
        );
    }

    #[test]
    fn a_short_document_is_one_chunk_with_no_separator() {
        let pieces = chunks("Fix the retry loop.", 100);
        assert_eq!(pieces, vec![("Fix the retry loop.", "")]);
    }

    #[test]
    fn chunks_reassemble_into_the_original() {
        let text = "# Task\n\nFirst paragraph, which is fairly long.\n\nSecond paragraph, \
                    also long.\n\nThird one.\n";
        for max in [10, 25, 40, 60, 1000] {
            let joined: String = chunks(text, max)
                .iter()
                .map(|(body, sep)| format!("{body}{sep}"))
                .collect();
            assert_eq!(joined, text, "max_chars = {max}");
        }
    }

    #[test]
    fn chunks_cut_on_blank_lines_and_hand_back_the_break() {
        let text = "Alpha beta gamma delta.\n\nEpsilon zeta eta theta.";
        let pieces = chunks(text, 30);
        assert_eq!(
            pieces,
            vec![
                ("Alpha beta gamma delta.", "\n\n"),
                ("Epsilon zeta eta theta.", "")
            ]
        );
    }

    /// The bug this guards: a cut inside a fence hands the model an unterminated code
    /// block, which is exactly the content the instruction insists it copy verbatim.
    ///
    /// The block below is 33 characters, so every window here can hold it — at 30 only by
    /// stretching, which is the point.
    #[test]
    fn chunks_keep_a_code_fence_whole() {
        let text = "Intro line here.\n\n```rust\nfn a() {}\n\nfn b() {}\n```\n\nOutro line.";
        for max in [30, 40, 60] {
            let pieces = chunks(text, max);
            for (body, _) in &pieces {
                assert_eq!(
                    body.matches("```").count() % 2,
                    0,
                    "chunk splits a fence at max_chars = {max}: {body:?}"
                );
            }
            assert!(
                pieces
                    .iter()
                    .any(|(b, _)| b.contains("fn a() {}") && b.contains("fn b() {}")),
                "the block should arrive in one piece at max_chars = {max}: {pieces:?}"
            );
        }
    }

    /// A fence longer than a chunk plus its stretch has to be split — there is no window
    /// that holds it. What must still hold is that the document survives the round trip.
    #[test]
    fn an_oversized_fence_is_split_rather_than_looping() {
        let body: String = (0..40).map(|i| format!("let x{i} = {i};\n")).collect();
        let text = format!("Intro.\n\n```rust\n{body}```\n\nOutro.");
        let pieces = chunks(&text, 60);
        assert!(pieces.len() > 2, "got {} pieces", pieces.len());
        let joined: String = pieces.iter().map(|(b, s)| format!("{b}{s}")).collect();
        assert_eq!(joined, text);
    }

    #[test]
    fn chunks_split_unbroken_prose_on_a_word_boundary() {
        let text = "alpha beta gamma delta epsilon zeta eta theta iota kappa";
        let pieces = chunks(text, 20);
        assert!(pieces.len() > 1, "got {pieces:?}");
        for (body, _) in &pieces {
            assert!(!body.is_empty() && !body.starts_with(' '), "got {pieces:?}");
        }
        let joined: String = pieces.iter().map(|(b, s)| format!("{b}{s}")).collect();
        assert_eq!(joined, text);
    }

    #[test]
    fn chunking_terminates_on_multibyte_text_and_degenerate_limits() {
        let text = "però — naïve café\n\nsecondo paragrafo però";
        assert_eq!(
            chunks(text, 1)
                .iter()
                .map(|(b, s)| format!("{b}{s}"))
                .collect::<String>(),
            text
        );
        assert!(chunks("", 100).is_empty());
        assert!(!chunks("abc", 0).is_empty());
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

    /// The bug this guards, seen on a real `pstore shrink` run: a prompt naming `Next.js`
    /// three times was rewritten to `React/Next.js + SQLite`, and the check reported the
    /// reference dropped — from a rewrite whose first line contains it. The comparison was
    /// whole whitespace tokens, so re-punctuating a reference read as losing it.
    #[test]
    fn a_reference_the_rewrite_repunctuated_is_not_reported_dropped() {
        for (before, after, why) in [
            (
                "Use Next.js for the frontend.",
                "Build with React/Next.js.",
                "a reference that gained a leading segment",
            ),
            (
                "Edit src/main.rs.",
                "Edit main.rs.",
                "a reference that lost a leading segment",
            ),
            (
                "Update config.toml and README.md.",
                "Update `config.toml`, `README.md`.",
                "references the rewrite wrapped in backticks",
            ),
            (
                "Touch src/a.rs, src/b.rs and src/c.rs.",
                "Touch src/a.rs src/b.rs src/c.rs",
                "several references, all kept",
            ),
        ] {
            assert!(
                dropped_paths(before, after).is_empty(),
                "{why}: {:?}",
                dropped_paths(before, after)
            );
        }
    }

    /// Segment-boundary, not substring: two files with the same name in different folders
    /// are different files, and saying otherwise would make the check miss the real thing it
    /// exists to catch.
    #[test]
    fn a_different_file_with_the_same_name_is_still_dropped() {
        for (before, after) in [
            ("Edit a/config.rs.", "Edit b/config.rs."),
            ("Edit src/net/retry.rs.", "Edit src/net/backoff.rs."),
            // Segment boundary, not substring: one name ending in the other is not the other.
            ("Edit xnext.js now.", "Edit next.js now."),
        ] {
            assert_eq!(
                dropped_paths(before, after).len(),
                1,
                "{before:?} → {after:?} should report exactly one dropped reference"
            );
        }
    }

    /// Presence, not frequency, and reported once. Repeating the token as many times as the
    /// *original* mentioned it — which is what it used to do — reads as several different
    /// files dropped, and the number came from the wrong side of the comparison.
    #[test]
    fn a_dropped_reference_is_reported_once_however_often_it_was_mentioned() {
        let before = "Edit src/main.rs. Then src/main.rs again. And src/main.rs once more.";
        assert_eq!(dropped_paths(before, "Do nothing."), ["src/main.rs"]);

        // Mentioned three times, kept once: shrink's whole job, and not a warning.
        assert!(dropped_paths(before, "Edit src/main.rs.").is_empty());

        // Distinct files are still listed separately, in the order the original named them.
        let w = dropped_paths("Edit src/b.rs then src/a.rs then src/b.rs.", "Nothing.");
        assert_eq!(w, ["src/b.rs", "src/a.rs"]);
    }
}
