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
//! Like shrink, this runs on the local checkpoint — the same one as ranking, compression
//! and the personal-data scan. No coding agent is involved and the prompt does not leave
//! the machine. Planning is a rewrite of the request, not work on the repository, so the
//! thing that should read the code is the agent the plan is *handed to*, afterwards.
//!
//! Like shrink, nothing is applied unreviewed: the result arrives as a proposal with a diff
//! against the current prompt, and accepting it is one undo step. [`run`] is the entry
//! point; it is blocking and belongs on a worker thread.

/// The planning instruction.
///
/// Describes the *fields* rather than a markdown layout, because the layout is not the
/// model's job here — [`render`] assembles it. Asking a 27B checkpoint compressed to one
/// bit per weight to "structure your answer as five headings" gets a paragraph explaining
/// how it intends to structure its answer as five headings; asking it to fill in `steps`
/// gets steps. The schema in [`crate::router::llm::plan`] is what actually enforces this,
/// and this text only has to say what belongs in each field.
pub const INSTRUCTION: &str = "\
Turn the request below into an instruction for a coding agent. Fill in each field.

objective — one sentence stating what must be true when the work is done.
context — facts the agent cannot discover from the code: decisions already made, \
constraints from outside the repository, things already tried and rejected. Empty if there \
are none.
steps — the work, in order. One action each, independently checkable. Name the files, \
functions and commands involved. Split anything that bundles two decisions.
constraints — what must not change: public APIs, file formats, behaviour other code \
depends on. Empty if there are none.
done_when — acceptance criteria. A command that must pass, an output that must appear, a \
behaviour that must hold.
open_questions — anything the request leaves ambiguous. Put it here instead of guessing. \
Empty if the request is clear.

Copy verbatim every code block, file path, identifier, version number, error message and \
command from the request. Do not invent requirements, file names or APIs. Do not perform \
the work, write the code, or describe how you would approach it. Write each entry as the \
instruction itself, not as a description of what the entry contains.";

/// Fields the model fills in, in the order [`render`] lays them out.
///
/// The JSON key, and the heading it becomes.
pub const FIELDS: [(&str, &str); 6] = [
    ("objective", "Objective"),
    ("context", "Context"),
    ("steps", "Steps"),
    ("constraints", "Constraints"),
    ("done_when", "Done when"),
    ("open_questions", "Open questions"),
];

/// Assemble the model's fields into the instruction that gets pasted into an agent.
///
/// `objective` is a sentence; every other field is a list. Empty lists are dropped rather
/// than rendered as an empty heading — "Constraints:" with nothing under it reads as a
/// section the planner forgot, when it means there are none. `steps` is numbered because
/// its order is load-bearing; the rest are bullets because theirs is not.
///
/// Each entry has its own list marker stripped first. The numbering and the bullets here are
/// pstore's, and a model asked for a list of constraints writes one anyway — a real run of
/// `pstore plan` returned `[0] The public API ... must remain unchanged`, which this laid out
/// as `- [0] The public API ...`. [`crate::rca`] met the same thing and already knows the
/// markers, including the ones that are content rather than punctuation.
pub fn render(objective: &str, sections: &[(&str, Vec<String>)]) -> String {
    let mut out = format!(
        "**Objective**\n{}\n",
        crate::rca::strip_leading_markers(objective)
    );
    for (heading, items) in sections {
        let items: Vec<String> = items
            .iter()
            .map(|i| crate::rca::strip_leading_markers(i))
            .filter(|i| !i.is_empty())
            .collect();
        if items.is_empty() {
            continue;
        }
        out.push_str(&format!("\n**{heading}**\n"));
        for (n, item) in items.iter().enumerate() {
            if *heading == "Steps" {
                out.push_str(&format!("{}. {item}\n", n + 1));
            } else {
                out.push_str(&format!("- {item}\n"));
            }
        }
    }
    out
}

/// Compose the request sent to the local model.
pub fn compose(document: &str) -> String {
    format!("{INSTRUCTION}\n\n---\n\n{document}")
}

/// Plan `text` on the local checkpoint.
///
/// Blocking and slow in units of seconds — the model is a subprocess that maps the weights
/// before it answers — so call it from a worker thread.
///
/// A request too long for the context window is refused rather than planned. llama.cpp
/// truncates a prompt that does not fit, silently, and a plan built from the first half of
/// a request is the failure that looks most like a success: it has all five headings and
/// omits half the work.
pub fn run(text: &str) -> Result<String, String> {
    let limit = crate::router::llm::plan_input_chars();
    if text.len() > limit {
        return Err(format!(
            "the prompt is too long to plan in one pass ({} characters, limit {limit}) — \
             shrink it first, or raise `model_context_ceiling` in .pstore/config.json",
            text.len()
        ));
    }
    let planned = crate::router::llm::plan(text)?;
    if planned.trim().is_empty() {
        return Err("the model returned an empty plan".into());
    }
    Ok(planned)
}

/// Problems in a produced plan that its schema could not rule out.
///
/// The five headings are guaranteed by [`render`] and the two that matter are non-empty by
/// schema, so what is left to check is content: whether the plan still refers to the things
/// the request named, and whether it is long enough to have said anything. These are
/// warnings rather than rejections — a short request legitimately produces a short plan.
pub fn warnings(plan: &str, original: &str) -> Vec<String> {
    let mut out = Vec::new();

    // The point of planning is to add structure, so a plan shorter than the request has
    // almost certainly dropped something rather than tightened it.
    if plan.len() < original.len() / 2 {
        out.push(format!(
            "the plan is much shorter than the request ({} vs {} chars) — check nothing was dropped",
            plan.len(),
            original.len()
        ));
    }

    // Paths are the thing an agent needs most and a planner drops most easily. The comparison
    // is `shrink`'s so that the two checks cannot disagree about what "dropped" means — they
    // ask the same question, and asking it twice is how they came to answer it differently.
    let dropped = crate::shrink::dropped_paths(original, plan);
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
        assert!(i.contains("do not perform the work"));
        // Every field the schema requires has to be described, or the model is being asked
        // to fill in something the instruction never explained.
        for (key, _) in FIELDS {
            assert!(i.contains(key), "the instruction must describe {key}");
        }
    }

    /// The headings, their order, and the numbering of the steps are pstore's, not the
    /// model's — that is the whole reason the model is asked for fields.
    #[test]
    fn render_lays_out_the_sections_in_order() {
        let out = render(
            "Retry count is configurable.",
            &[
                ("Context", vec!["Hard-coded today.".into()]),
                (
                    "Steps",
                    vec!["Edit src/net/retry.rs.".into(), "Add a test.".into()],
                ),
                ("Constraints", vec![]),
                ("Done when", vec!["`cargo test` passes.".into()]),
                ("Open questions", vec![]),
            ],
        );
        assert_eq!(
            out,
            "**Objective**\nRetry count is configurable.\n\n\
             **Context**\n- Hard-coded today.\n\n\
             **Steps**\n1. Edit src/net/retry.rs.\n2. Add a test.\n\n\
             **Done when**\n- `cargo test` passes.\n"
        );
    }

    /// An empty list means "there are none", which is a fact. A heading with nothing under
    /// it reads as a section the planner forgot, which is a different claim.
    #[test]
    fn an_empty_section_is_omitted_not_left_blank() {
        let out = render(
            "Do it.",
            &[("Constraints", vec![" ".into(), String::new()])],
        );
        assert!(!out.contains("Constraints"), "got {out:?}");
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

    /// The numbering and the bullets are pstore's. A model asked for a *list* of constraints
    /// writes its own marker anyway, and laying that out again produces `- [0] ...` — which is
    /// what a real `pstore plan` run returned before this stripped them.
    #[test]
    fn a_models_own_list_marker_is_not_laid_out_again() {
        for marker in ["- ", "* ", "• ", "1. ", "2) ", "[0] ", "[3]"] {
            let out = render(
                "Do it.",
                &[("Constraints", vec![format!("{marker}Keep the public API.")])],
            );
            assert!(
                out.contains("- Keep the public API.\n"),
                "marker {marker:?} survived into {out:?}"
            );
        }

        // Steps get pstore's numbers, so a model that numbered them itself must not end up
        // with two — and the numbers must be pstore's, not the model's.
        let out = render(
            "Do it.",
            &[(
                "Steps",
                vec!["3. Edit src/a.rs.".into(), "- Edit src/b.rs.".into()],
            )],
        );
        assert!(
            out.contains("1. Edit src/a.rs.\n2. Edit src/b.rs.\n"),
            "got {out:?}"
        );

        // Content that merely opens with digits is not a marker and must survive intact.
        let out = render(
            "Do it.",
            &[("Done when", vec!["404 is never returned.".into()])],
        );
        assert!(out.contains("- 404 is never returned."), "got {out:?}");
    }

    /// An entry that is nothing but a marker is an empty entry, and an empty section is
    /// omitted rather than rendered as a heading with a stray bullet under it.
    #[test]
    fn a_section_of_bare_markers_is_omitted() {
        let out = render(
            "Do it.",
            &[("Constraints", vec!["- ".into(), "[0] ".into()])],
        );
        assert!(!out.contains("Constraints"), "got {out:?}");
    }

    /// The two path checks answer the same question and used to answer it differently. This
    /// is the case that showed it: a plan is allowed to re-punctuate a reference.
    #[test]
    fn the_dropped_path_check_agrees_with_shrink() {
        let original = "Update src/net/retry.rs and src/net/retry.rs again.";

        // Named once, in backticks, at the end of a sentence: kept, and reported so.
        let plan = "**Objective**\nRetry is configurable.\n\n\
                    **Steps**\n1. Edit `src/net/retry.rs`.";
        assert!(
            warnings(plan, original).is_empty(),
            "{:?}",
            warnings(plan, original)
        );

        // Genuinely dropped, and named twice in the original: reported once, not twice.
        let w = warnings(
            "**Objective**\nDo something else entirely, at length.",
            original,
        );
        let dropped: Vec<&String> = w.iter().filter(|w| w.contains("dropped")).collect();
        assert_eq!(dropped.len(), 1, "got {w:?}");
        assert_eq!(
            dropped[0].matches("src/net/retry.rs").count(),
            1,
            "one file, reported once: {:?}",
            dropped[0]
        );
    }
}
