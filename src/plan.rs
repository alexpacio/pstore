//! Turning a rough prompt into a plan a coding agent can execute.
//!
//! This is the other half of what [`crate::shrink`] does. Shrink takes a prompt that says
//! the right things at too much length; Plan takes a prompt that knows what it wants but
//! has not said it in an order an agent can act on — "fix the auth flow and also the tests
//! are flaky, oh and we should probably migrate the config format".
//!
//! The output is **not** a document for a human to read. It is the next prompt: a
//! structured instruction to be pasted into a coding agent. That distinction drives every
//! line of the instruction below — no preamble, no "here's my plan!", no summary of what it
//! just did, and no offer to help further. Anything the agent would skim past is noise that
//! the developer then has to delete by hand.
//!
//! Like shrink, nothing is applied unreviewed: the result arrives as a proposal with a diff
//! against the current prompt, and accepting it is one undo step.

/// The planning instruction.
///
/// Deliberately prescriptive about *shape*. A model asked to "make a plan" will happily
/// produce prose with a numbered list in the middle; an agent handed that has to infer the
/// task boundaries, and infers them differently each run.
pub const INSTRUCTION: &str = "\
Rewrite the request below as a precise instruction for a coding agent.

The output IS the next prompt — it will be pasted directly into a coding agent. Write it \
for that reader, not for a person.

Structure it as:
- **Objective** — one sentence stating what must be true when the work is done.
- **Context** — only facts the agent cannot discover from the code: decisions already \
made, constraints from outside the repository, things that were tried and rejected.
- **Steps** — ordered, each one independently checkable. Name the files, functions and \
commands involved. Split anything that bundles two decisions.
- **Constraints** — what must not change: public APIs, file formats, behaviour other code \
depends on.
- **Done when** — concrete acceptance criteria. A command that must pass, an output that \
must appear, a behaviour that must hold.

Rules:
- Preserve verbatim every code block, file path, identifier, version number, error \
message and command from the original.
- Do not invent requirements, file names, or APIs. If the request is ambiguous, put the \
ambiguity under a final **Open questions** heading rather than guessing.
- Do not perform the work, write the code, or describe how you would approach it.
- Output only the instruction. No preamble, no commentary, no summary, and no code fence \
wrapping the whole thing.";

/// Compose the request sent to an agent.
pub fn compose(document: &str) -> String {
    format!("{INSTRUCTION}\n\n---\n\n{document}")
}

/// Headings the instruction asks for, in order.
const SECTIONS: [&str; 5] = ["Objective", "Context", "Steps", "Constraints", "Done when"];

/// Structural problems in a produced plan.
///
/// A plan is only worth pasting into an agent if it has the shape that makes it
/// unambiguous. These are warnings rather than rejections — a short request legitimately
/// produces a short plan — but they are shown, because a plan missing its acceptance
/// criteria is the one that quietly wastes an agent run.
pub fn warnings(plan: &str, original: &str) -> Vec<String> {
    let mut out = Vec::new();

    let missing: Vec<&str> = SECTIONS
        .iter()
        .copied()
        .filter(|s| !plan.to_lowercase().contains(&s.to_lowercase()))
        .collect();
    if !missing.is_empty() {
        out.push(format!("no {} section", missing.join(", no ")));
    }

    // The point of planning is to add structure, so a plan shorter than the request has
    // almost certainly dropped something rather than tightened it.
    if plan.len() < original.len() / 2 {
        out.push(format!(
            "the plan is much shorter than the request ({} vs {} chars) — check nothing was dropped",
            plan.len(),
            original.len()
        ));
    }

    // Paths are the thing an agent needs most and a planner drops most easily.
    let paths = |s: &str| -> Vec<String> {
        s.split_whitespace()
            .filter_map(crate::shrink::path_token)
            .collect()
    };
    let after = paths(plan);
    let dropped: Vec<String> = paths(original)
        .into_iter()
        .filter(|p| !after.contains(p))
        .collect();
    if !dropped.is_empty() {
        out.push(format!("file paths dropped: {}", dropped.join(", ")));
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The instruction's job is to produce something machine-facing. If it stops saying so,
    /// models revert to writing a friendly document with a plan buried in it.
    #[test]
    fn the_instruction_targets_an_agent_not_a_reader() {
        let i = INSTRUCTION.to_lowercase();
        assert!(i.contains("coding agent"));
        assert!(
            i.contains("output is the next prompt") || i.contains("output is the next prompt"),
            "it must say the output is itself a prompt"
        );
        assert!(i.contains("no preamble"));
        assert!(i.contains("do not perform the work"));
        // Guessing is the failure mode that costs a whole agent run.
        assert!(i.contains("open questions"));
        for section in SECTIONS {
            assert!(
                i.contains(&section.to_lowercase()),
                "the instruction must ask for {section}"
            );
        }
    }

    #[test]
    fn compose_keeps_the_document_after_the_instruction() {
        let out = compose("make the retry logic configurable");
        assert!(out.starts_with(INSTRUCTION));
        assert!(out.ends_with("make the retry logic configurable"));
    }

    #[test]
    fn a_well_formed_plan_raises_nothing() {
        let original = "Make the retry count configurable in src/net/retry.rs.";
        let plan = "**Objective**\nRetry count is configurable.\n\n\
                    **Context**\nCurrently hard-coded.\n\n\
                    **Steps**\n1. Edit src/net/retry.rs to read the value.\n\n\
                    **Constraints**\nDo not change the public API.\n\n\
                    **Done when**\n`cargo test` passes and the value is read from config.";
        assert!(
            warnings(plan, original).is_empty(),
            "{:?}",
            warnings(plan, original)
        );
    }

    /// The section most often missing is the one that matters most: without acceptance
    /// criteria an agent decides for itself when it is finished.
    #[test]
    fn a_missing_section_is_reported() {
        let original = "Make the retry count configurable in src/net/retry.rs.";
        let plan = "**Objective**\nRetry count is configurable.\n\n\
                    **Context**\nHard-coded today.\n\n\
                    **Steps**\n1. Edit src/net/retry.rs.\n\n\
                    **Constraints**\nNone.";
        let w = warnings(plan, original);
        assert!(
            w.iter().any(|w| w.contains("Done when")),
            "missing acceptance criteria should be called out: {w:?}"
        );
    }

    /// A dropped path is the failure that sends an agent looking in the wrong place.
    #[test]
    fn dropped_file_paths_are_reported() {
        let original = "Update src/net/retry.rs and src/config.rs to share a constant.";
        let plan = "**Objective**\nShare the constant.\n\n**Context**\nNone.\n\n\
                    **Steps**\n1. Edit src/net/retry.rs.\n\n**Constraints**\nNone.\n\n\
                    **Done when**\nBoth read the same value.";
        let w = warnings(plan, original);
        assert!(w.iter().any(|w| w.contains("src/config.rs")), "got {w:?}");
    }

    #[test]
    fn a_suspiciously_short_plan_is_flagged() {
        let original = "a".repeat(400);
        let w = warnings("**Objective** do it", &original);
        assert!(w.iter().any(|w| w.contains("much shorter")), "got {w:?}");
    }
}
