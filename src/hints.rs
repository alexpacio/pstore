//! Asking an LLM for help *while* writing a prompt.
//!
//! Two entry points, both requested by the workflow: the developer either selects
//! some of what they've written and asks about it, or types a fresh question. Either
//! way the request carries the surrounding prompt as context, so the answer is about
//! *this* prompt rather than floating free.
//!
//! The answer lands in a panel, never silently in the document — inserting is an
//! explicit second step, and one undo step when taken.

/// What the developer is asking about.
///
/// Three cases rather than two. Selecting a passage *and* typing a question about it is the
/// normal way to use this — "is this constraint specific enough?" only means anything
/// alongside the constraint — so it gets its own variant instead of being flattened into a
/// question with the selection glued on. The distinction survives into [`compose`], which
/// can then tell the agent which part is the subject and which part is the ask.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Subject {
    /// Text selected in the editor, with no question typed.
    Selection(String),
    /// A question typed into the hint box, with nothing selected.
    Question(String),
    /// A question about a specific selection.
    About {
        /// The selected passage.
        selection: String,
        /// What they want to know about it.
        question: String,
    },
}

impl Subject {
    /// Build a subject from the current selection and question box.
    ///
    /// Returns `None` only when both are empty — a selection does not suppress the
    /// question box, and typing one does not discard the selection.
    pub fn resolve(selection: Option<&str>, question: &str) -> Option<Self> {
        let q = question.trim();
        match selection.map(str::trim).filter(|s| !s.is_empty()) {
            Some(sel) if q.is_empty() => Some(Subject::Selection(sel.to_string())),
            Some(sel) => Some(Subject::About {
                selection: sel.to_string(),
                question: q.to_string(),
            }),
            None if !q.is_empty() => Some(Subject::Question(q.to_string())),
            None => None,
        }
    }

    /// Short label for the panel header.
    pub fn label(&self) -> &'static str {
        match self {
            Subject::Selection(_) => "selection",
            Subject::Question(_) => "question",
            Subject::About { .. } => "question about the selection",
        }
    }

    /// The text being asked about.
    pub fn text(&self) -> &str {
        match self {
            Subject::Selection(s) | Subject::Question(s) => s,
            Subject::About { question, .. } => question,
        }
    }
}

/// Instruction prefix for hint requests. Kept short: hints are read in a side panel
/// mid-edit, so a wall of prose defeats the purpose.
const PREAMBLE: &str = "\
You are helping a developer draft a prompt for a coding agent. Answer concisely and \
concretely — they are mid-edit and will read this in a side panel. Do not rewrite \
their whole prompt unless asked. If they are missing information a coding agent would \
need, say which.";

/// How much surrounding prompt to include as context.
const CONTEXT_BUDGET: usize = 4000;

/// Compose the full text to send to an agent.
pub fn compose(subject: &Subject, document: &str) -> String {
    let doc = document.trim();
    let mut out = String::with_capacity(PREAMBLE.len() + doc.len() + 256);
    out.push_str(PREAMBLE);

    if !doc.is_empty() {
        out.push_str("\n\n<prompt-being-written>\n");
        out.push_str(&clip(doc, CONTEXT_BUDGET));
        out.push_str("\n</prompt-being-written>");
    }

    match subject {
        Subject::Selection(sel) => {
            out.push_str("\n\nThey selected this and want help with it:\n\n");
            out.push_str(sel);
        }
        Subject::Question(q) => {
            out.push_str("\n\nTheir question:\n\n");
            out.push_str(q);
        }
        // Kept apart so the agent can tell the passage from the ask. Concatenating them
        // reads as one long question and the answer drifts off the selection.
        Subject::About {
            selection,
            question,
        } => {
            out.push_str("\n\nThey selected this part of the prompt:\n\n");
            out.push_str(selection);
            out.push_str("\n\nand asked about it:\n\n");
            out.push_str(question);
        }
    }
    out
}

/// Trim `text` to roughly `budget` characters, keeping the head and tail.
///
/// The beginning states the task and the end is where the developer is working, so
/// dropping the middle preserves the useful parts.
fn clip(text: &str, budget: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= budget {
        return text.to_string();
    }
    let head: String = chars.iter().take(budget * 2 / 3).collect();
    let tail: String = chars[chars.len() - budget / 3..].iter().collect();
    format!(
        "{head}\n\n[... {} characters omitted ...]\n\n{tail}",
        chars.len() - budget
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_alone_becomes_a_selection_subject() {
        let s = Subject::resolve(Some("the retry logic"), "").unwrap();
        assert_eq!(s, Subject::Selection("the retry logic".into()));
        assert_eq!(s.label(), "selection");
    }

    #[test]
    fn question_alone_becomes_a_question_subject() {
        let s = Subject::resolve(None, "how do I phrase the constraint?").unwrap();
        assert_eq!(s.label(), "question");
        assert_eq!(s.text(), "how do I phrase the constraint?");
    }

    /// The case the hint panel is actually for: highlight something, ask about it. Neither
    /// input suppresses the other, and both reach the agent as distinct things.
    #[test]
    fn a_selection_and_a_question_are_both_kept() {
        let s = Subject::resolve(Some("retry three times"), "is this specific enough?").unwrap();
        assert_eq!(
            s,
            Subject::About {
                selection: "retry three times".into(),
                question: "is this specific enough?".into(),
            }
        );
        assert_eq!(s.label(), "question about the selection");
        assert_eq!(
            s.text(),
            "is this specific enough?",
            "the ask, not the passage"
        );

        // What the agent receives has to distinguish them, or the answer drifts off the
        // selection and onto the prompt as a whole.
        let composed = compose(&s, "Write a retry policy. retry three times. Log failures.");
        assert!(composed.contains("retry three times"));
        assert!(composed.contains("is this specific enough?"));
        let sel_at = composed
            .find("They selected this part")
            .expect("selection framing");
        let q_at = composed
            .find("and asked about it")
            .expect("question framing");
        assert!(sel_at < q_at, "the passage should come before the ask");
    }

    #[test]
    fn nothing_to_ask_about_yields_none() {
        assert!(Subject::resolve(None, "").is_none());
        assert!(Subject::resolve(Some("   "), "  ").is_none());
        assert!(Subject::resolve(None, "\n\t ").is_none());
    }

    #[test]
    fn whitespace_only_selection_falls_through_to_the_question() {
        let s = Subject::resolve(Some("  \n "), "what next?").unwrap();
        assert_eq!(s, Subject::Question("what next?".into()));
    }

    #[test]
    fn compose_includes_document_and_subject() {
        let doc = "# Task\nRefactor the parser.";
        let s = Subject::resolve(None, "what am I missing?").unwrap();
        let out = compose(&s, doc);
        assert!(out.starts_with(PREAMBLE));
        assert!(out.contains("<prompt-being-written>"));
        assert!(out.contains("Refactor the parser."));
        assert!(out.contains("what am I missing?"));
    }

    #[test]
    fn compose_omits_the_context_block_for_an_empty_document() {
        let s = Subject::resolve(None, "how do I start?").unwrap();
        let out = compose(&s, "   \n  ");
        assert!(
            !out.contains("<prompt-being-written>"),
            "no empty context block"
        );
        assert!(out.contains("how do I start?"));
    }

    #[test]
    fn long_documents_are_clipped_keeping_both_ends() {
        let doc = format!("HEAD-MARKER{}TAIL-MARKER", "x".repeat(50_000));
        let s = Subject::resolve(None, "q").unwrap();
        let out = compose(&s, &doc);
        assert!(
            out.len() < 12_000,
            "context must be bounded, got {}",
            out.len()
        );
        assert!(
            out.contains("HEAD-MARKER"),
            "the task statement is at the top"
        );
        assert!(
            out.contains("TAIL-MARKER"),
            "the working edge is at the bottom"
        );
        assert!(
            out.contains("characters omitted"),
            "clipping must be visible"
        );
    }

    #[test]
    fn clipping_respects_char_boundaries() {
        // Byte slicing would panic here.
        let doc = "é".repeat(10_000);
        let out = clip(&doc, 100);
        assert!(out.contains('é'));
        assert!(out.chars().count() < 400);
    }

    #[test]
    fn short_documents_are_passed_through_untouched() {
        assert_eq!(clip("short", 100), "short");
    }
}
