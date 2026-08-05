//! Zero-dependency capability and complexity estimation.
//!
//! The Candle classifiers in [`super::capability`] and [`super::complexity`] are the
//! real implementation. This exists so the app is usable *before* their weights have
//! been downloaded, and offline — it produces the same `(Capability, Complexity)`
//! shape from surface features of the text.

use super::{Capability, Complexity};

/// Whether a token looks like a file path or filename.
///
/// Deliberately stricter than "contains a dot": a sentence-final `README.` or an
/// `e.g.` must not read as a file reference, or difficulty inflates on ordinary prose.
/// Also used by the shrinker's integrity check.
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

/// Estimate what capabilities a prompt draws on, from surface features.
pub fn capability(text: &str) -> Capability {
    let lower = text.to_ascii_lowercase();

    let code_fences = lower.matches("```").count() / 2;
    let inline_code = lower.matches('`').count().saturating_sub(code_fences * 6);
    let paths = text
        .split_whitespace()
        .filter(|t| looks_like_path(t))
        .count();
    let identifiers = text
        .split_whitespace()
        .filter(|t| {
            t.contains('_')
                || t.contains("()")
                || (t.chars().any(|c| c.is_ascii_uppercase())
                    && t.chars().any(|c| c.is_ascii_lowercase())
                    && !t.chars().next().is_some_and(|c| c.is_ascii_uppercase()))
        })
        .count();

    let hits =
        |needles: &[&str]| -> f32 { needles.iter().filter(|n| lower.contains(*n)).count() as f32 };

    let coding_signal = hits(&[
        "refactor",
        "function",
        "class",
        "compile",
        "bug",
        "test",
        "api",
        "type",
        "import",
        "module",
        "struct",
        "async",
        "crate",
        "package",
        "build",
        "lint",
        "stack trace",
        "exception",
        "regression",
    ]) + code_fences as f32 * 2.0
        + paths as f32
        + identifiers as f32 * 0.5
        + inline_code as f32 * 0.3;

    let math_signal = hits(&[
        "calculate",
        "complexity",
        "algorithm",
        "proof",
        "optimi",
        "big-o",
        "o(n",
        "probability",
        "equation",
        "derive",
        "statistic",
        "numeric",
        "matrix",
    ]) + text.matches(|c: char| c.is_ascii_digit()).count() as f32 * 0.02;

    let planning_signal = hits(&[
        "step",
        "plan",
        "then",
        "first",
        "next",
        "finally",
        "migrate",
        "orchestrat",
        "pipeline",
        "workflow",
        "phase",
        "across",
        "multiple files",
        "end-to-end",
        "sequence",
        "coordinate",
    ]) + text
        .lines()
        .filter(|l| {
            let t = l.trim_start();
            t.starts_with("- ")
                || t.starts_with("* ")
                || t.starts_with(|c: char| c.is_ascii_digit())
        })
        .count() as f32
        * 0.4;

    let world_signal = hits(&[
        "what is",
        "who",
        "when did",
        "history",
        "explain",
        "background",
        "docs",
        "documentation",
        "standard",
        "spec",
        "convention",
        "best practice",
        "compare",
    ]);

    let creative_signal = hits(&[
        "write",
        "draft",
        "story",
        "tone",
        "style",
        "rephrase",
        "summar",
        "headline",
        "blog",
        "copy",
        "narrative",
        "describe",
        "name ",
    ]);

    let instruction_signal = hits(&[
        "must",
        "should",
        "do not",
        "don't",
        "only",
        "exactly",
        "format",
        "output",
        "constraint",
        "require",
        "ensure",
        "follow",
        "json",
        "schema",
        "return",
    ]) + 1.5; // every prompt asks to be followed

    // Squash each raw count into [0, 1] with a soft curve. No length scaling: the
    // signals count *distinct* markers rather than occurrences, so they don't inflate
    // with prompt size — and scaling by length would wrongly damp a short prompt that
    // is plainly multi-step.
    let squash = |x: f32| (x / (x + 3.0)).clamp(0.0, 1.0);

    Capability {
        scores: [
            squash(instruction_signal),
            squash(coding_signal),
            squash(math_signal),
            squash(world_signal),
            squash(planning_signal),
            squash(creative_signal),
        ],
    }
}

/// Estimate difficulty from length, structure and multi-step markers.
///
/// Deliberately not a pure length check: a short prompt with several hard
/// constraints outranks a long but simple one.
pub fn complexity(text: &str) -> Complexity {
    let lower = text.to_ascii_lowercase();
    let words = lower.split_whitespace().count();
    let mut points = 0i32;

    points += match words {
        0..=12 => 0,
        13..=60 => 1,
        61..=200 => 2,
        _ => 3,
    };

    let fences = lower.matches("```").count() / 2;
    points += (fences as i32).min(2);

    let files = text
        .split_whitespace()
        .filter(|t| looks_like_path(t))
        .count();
    if files >= 3 {
        points += 2;
    } else if files >= 1 {
        points += 1;
    }

    let multi_step = [
        "refactor",
        "migrate",
        "across",
        "architecture",
        "redesign",
        "end-to-end",
        "backwards compat",
        "concurren",
        "race condition",
        "deadlock",
        "performance",
        "optimi",
        "trade-off",
        "tradeoff",
        "proof",
        "invariant",
        "distributed",
    ]
    .iter()
    .filter(|k| lower.contains(*k))
    .count();
    points += (multi_step as i32).min(3);

    let trivial = [
        "typo",
        "rename",
        "format",
        "indent",
        "comment out",
        "bump version",
    ]
    .iter()
    .any(|k| lower.contains(k));
    if trivial && words < 40 {
        points -= 2;
    }

    let constraints = [
        "must",
        "must not",
        "do not",
        "ensure",
        "without breaking",
        "only if",
    ]
    .iter()
    .filter(|k| lower.contains(*k))
    .count();
    points += (constraints as i32).min(2);

    match points {
        i32::MIN..=1 => Complexity::Easy,
        2..=4 => Complexity::Medium,
        _ => Complexity::Hard,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::registry::DIMS;

    fn dominant(c: &Capability) -> &'static str {
        let (i, _) =
            c.scores.iter().enumerate().fold(
                (0, f32::MIN),
                |acc, (i, v)| if *v > acc.1 { (i, *v) } else { acc },
            );
        DIMS[i]
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
    fn scores_stay_in_range() {
        for text in [
            "",
            "fix typo",
            "refactor the auth module across src/a.rs and src/b.rs without breaking the API",
            &"word ".repeat(5000),
        ] {
            let c = capability(text);
            for (i, v) in c.scores.iter().enumerate() {
                assert!(
                    (0.0..=1.0).contains(v),
                    "{} out of range for {text:?}: {v}",
                    DIMS[i]
                );
            }
        }
    }

    #[test]
    fn code_heavy_prompts_lean_on_coding() {
        let text = "Refactor the `parse_config` function in src/config.rs; the test in \
                    tests/config_test.rs fails to compile after the struct change.";
        assert_eq!(dominant(&capability(text)), "coding");
    }

    #[test]
    fn prose_requests_lean_on_creative_synthesis() {
        let c = capability("Draft a short blog post about our launch; keep the tone warm.");
        assert!(
            c.scores[5] > c.scores[1],
            "creative {} should beat coding {}",
            c.scores[5],
            c.scores[1]
        );
    }

    #[test]
    fn multi_step_work_registers_as_planning() {
        let text = "First migrate the schema, then update every caller across the service, \
                    then run the end-to-end pipeline and coordinate the rollout phases.";
        let c = capability(text);
        assert!(c.scores[4] > 0.4, "planning was only {}", c.scores[4]);
    }

    #[test]
    fn trivial_edits_are_easy() {
        assert_eq!(complexity("fix this typo"), Complexity::Easy);
        assert_eq!(
            complexity("rename the variable to `count`"),
            Complexity::Easy
        );
    }

    #[test]
    fn multi_file_constrained_refactors_are_hard() {
        let text = "Refactor the authentication layer across src/auth/mod.rs, \
                    src/auth/session.rs and src/api/routes.rs. You must not break \
                    backwards compatibility, and ensure the concurrent session test \
                    still passes. Watch for the race condition in the token refresh.";
        assert_eq!(complexity(text), Complexity::Hard);
    }

    #[test]
    fn short_but_dense_beats_long_but_simple() {
        let dense = "Fix the deadlock in the distributed lock; must not break the \
                     existing invariant.";
        let long_simple = "please ".repeat(200);
        assert!(
            complexity(dense) > complexity(&long_simple),
            "difficulty is not just length"
        );
    }

    #[test]
    fn empty_input_is_easy_and_does_not_panic() {
        assert_eq!(complexity(""), Complexity::Easy);
        let c = capability("");
        assert!(c.scores.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn complexity_is_ordered_across_a_ladder() {
        let easy = complexity("bump version");
        let mid = complexity(
            "Add a `--verbose` flag to the CLI in src/main.rs and document it in the README.",
        );
        let hard = complexity(
            "Redesign the storage architecture for backwards compatibility across \
             src/store/mod.rs, src/store/index.rs and src/api.rs; you must not break \
             existing snapshots, and optimise the concurrent read path.",
        );
        assert!(easy < mid, "{easy:?} !< {mid:?}");
        assert!(mid < hard, "{mid:?} !< {hard:?}");
    }
}
