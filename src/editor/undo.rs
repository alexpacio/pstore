//! Snapshot-based undo/redo.
//!
//! egui's built-in `TextEdit` undoer is deliberately not used: it has no redo (upstream
//! TODO), and it snapshots at its own cadence, which would smear a single programmatic
//! edit — a shrink, a hint insertion, a version restore — across many undo steps.
//!
//! This stack instead:
//!   * coalesces typing into word/line granules, holding the in-flight edit in
//!     [`UndoStack::pending`] until a boundary or an idle gap commits it, and
//!   * exposes [`UndoStack::push_atomic`] so one logical operation is exactly one step,
//!     carrying a label the UI can show ("Undo shrink").

/// How long the buffer must be idle before a typing snapshot is committed.
pub const COALESCE_IDLE: std::time::Duration = std::time::Duration::from_millis(400);

/// Hard cap on retained history, to bound memory on very large prompts.
const CAP: usize = 200;

/// One point in history: the full text plus where the caret was.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Snapshot {
    /// Full buffer contents.
    pub text: String,
    /// Caret index in characters, used to restore the cursor on undo.
    pub caret: usize,
    /// Human label describing the edit that produced this state.
    pub label: &'static str,
}

/// Undo/redo history for a single buffer.
#[derive(Debug)]
pub struct UndoStack {
    /// Committed states, oldest first. Always non-empty.
    entries: Vec<Snapshot>,
    /// Index of the current committed state within `entries`.
    cursor: usize,
    /// Typing that has not yet earned its own history entry.
    pending: Option<Snapshot>,
}

impl UndoStack {
    /// Create a stack whose initial state is `text`.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            entries: vec![Snapshot {
                text: text.into(),
                caret: 0,
                label: "open",
            }],
            cursor: 0,
            pending: None,
        }
    }

    /// Reset history to a new baseline, discarding everything (used when switching prompts).
    pub fn reset(&mut self, text: impl Into<String>) {
        self.entries = vec![Snapshot {
            text: text.into(),
            caret: 0,
            label: "open",
        }];
        self.cursor = 0;
        self.pending = None;
    }

    /// The current committed text.
    pub fn committed(&self) -> &str {
        &self.entries[self.cursor].text
    }

    /// Whether there is an earlier state to return to.
    pub fn can_undo(&self) -> bool {
        self.cursor > 0 || self.pending.is_some()
    }

    /// Whether there is a later state to return to.
    ///
    /// Uncommitted typing suppresses redo: the tail is about to be invalidated anyway.
    pub fn can_redo(&self) -> bool {
        self.pending.is_none() && self.cursor + 1 < self.entries.len()
    }

    /// Label of the edit that undo would reverse, for menu text.
    pub fn undo_label(&self) -> Option<&'static str> {
        if self.pending.is_some() {
            return Some("typing");
        }
        (self.cursor > 0).then(|| self.entries[self.cursor].label)
    }

    /// Record an in-progress typing edit.
    ///
    /// Commits immediately when the edit closes a word or line (so Ctrl+Z walks back in
    /// meaningful granules); otherwise the edit is held pending until
    /// [`UndoStack::commit_pending`] fires on idle.
    ///
    /// Returns `true` if a history entry was created.
    pub fn on_edit(&mut self, text: &str, caret: usize) -> bool {
        if text == self.committed() {
            // Typing was reverted by hand back to the committed state.
            self.pending = None;
            return false;
        }
        let grew = text.len()
            > self
                .pending
                .as_ref()
                .map_or(self.committed(), |p| p.text.as_str())
                .len();
        let closes_granule = grew
            && text
                .chars()
                .nth(caret.saturating_sub(1))
                .is_some_and(|c| c.is_whitespace());
        if closes_granule {
            self.pending = None;
            self.commit(text, caret, "typing");
            true
        } else {
            self.pending = Some(Snapshot {
                text: text.to_string(),
                caret,
                label: "typing",
            });
            false
        }
    }

    /// Commit any pending typing as a single entry. Call after [`COALESCE_IDLE`] of idle.
    pub fn commit_pending(&mut self) -> bool {
        match self.pending.take() {
            Some(p) => {
                self.commit(&p.text, p.caret, "typing");
                true
            }
            None => false,
        }
    }

    /// Record one programmatic edit as exactly one undo step.
    ///
    /// Any pending keystrokes are committed first, so they remain separately undoable
    /// instead of being swallowed by this step.
    pub fn push_atomic(&mut self, text: &str, caret: usize, label: &'static str) {
        self.commit_pending();
        if text == self.committed() {
            return;
        }
        self.commit(text, caret, label);
    }

    fn commit(&mut self, text: &str, caret: usize, label: &'static str) {
        // A new edit invalidates any redo tail.
        self.entries.truncate(self.cursor + 1);
        self.entries.push(Snapshot {
            text: text.to_string(),
            caret,
            label,
        });
        if self.entries.len() > CAP {
            let overflow = self.entries.len() - CAP;
            self.entries.drain(0..overflow);
        }
        self.cursor = self.entries.len() - 1;
    }

    /// Step back one state. Returns the state to apply to the buffer.
    pub fn undo(&mut self) -> Option<Snapshot> {
        // Pending keystrokes are themselves a step: commit them, then this undo
        // lands on the state just before the typing began.
        self.commit_pending();
        if self.cursor == 0 {
            return None;
        }
        self.cursor -= 1;
        Some(self.entries[self.cursor].clone())
    }

    /// Step forward one state. Returns the state to apply to the buffer.
    pub fn redo(&mut self) -> Option<Snapshot> {
        if !self.can_redo() {
            return None;
        }
        self.cursor += 1;
        Some(self.entries[self.cursor].clone())
    }

    /// Number of retained committed states.
    ///
    /// Private, and only the tests ask: the one that matters holds [`CAP`] honest, because a
    /// history that quietly stopped being bounded would cost memory on a long prompt and
    /// nothing would say so.
    #[cfg(test)]
    fn len(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn undo_returns_to_previous_state() {
        let mut s = UndoStack::new("");
        s.push_atomic("hello", 5, "typing");
        s.push_atomic("hello world", 11, "typing");

        assert_eq!(s.undo().unwrap().text, "hello");
        assert_eq!(s.undo().unwrap().text, "");
        assert!(s.undo().is_none(), "cannot undo past the baseline");
    }

    #[test]
    fn redo_walks_forward_again() {
        let mut s = UndoStack::new("");
        s.push_atomic("a", 1, "typing");
        s.push_atomic("ab", 2, "typing");

        s.undo();
        s.undo();
        assert_eq!(s.redo().unwrap().text, "a");
        assert_eq!(s.redo().unwrap().text, "ab");
        assert!(s.redo().is_none());
    }

    #[test]
    fn new_edit_truncates_redo_tail() {
        let mut s = UndoStack::new("");
        s.push_atomic("one", 3, "typing");
        s.push_atomic("two", 3, "typing");
        s.undo();
        assert!(s.can_redo());

        s.push_atomic("three", 5, "typing");
        assert!(!s.can_redo(), "a fresh edit must discard the redo tail");
        assert_eq!(s.undo().unwrap().text, "one");
    }

    #[test]
    fn push_atomic_is_exactly_one_step() {
        let mut s = UndoStack::new("original text here");
        let before = s.len();
        // A shrink rewrites the whole document; it must cost one undo, not many.
        s.push_atomic("short", 5, "shrink");
        assert_eq!(s.len(), before + 1);
        assert_eq!(s.undo_label(), Some("shrink"));
        assert_eq!(s.undo().unwrap().text, "original text here");
    }

    #[test]
    fn typing_commits_on_word_boundary_not_every_char() {
        let mut s = UndoStack::new("");
        assert!(!s.on_edit("h", 1), "mid-word keystroke does not commit");
        assert!(!s.on_edit("he", 2));
        assert!(!s.on_edit("hey", 3));
        assert!(s.on_edit("hey ", 4), "whitespace closes a word granule");
        assert_eq!(s.len(), 2);

        assert!(!s.on_edit("hey t", 5));
        assert!(s.commit_pending(), "idle commits the tail");
        assert_eq!(s.len(), 3);
        assert!(!s.commit_pending(), "nothing new to commit");
    }

    #[test]
    fn uncommitted_typing_is_undoable() {
        let mut s = UndoStack::new("base");
        s.on_edit("based", 5); // pending, never idle-committed
        assert!(s.can_undo());
        assert_eq!(s.undo_label(), Some("typing"));
        assert_eq!(s.undo().unwrap().text, "base");
    }

    #[test]
    fn atomic_edit_after_typing_keeps_both_steps() {
        let mut s = UndoStack::new("");
        s.on_edit("draft", 5); // pending
        s.push_atomic("SHRUNK", 6, "shrink");

        assert_eq!(
            s.undo().unwrap().text,
            "draft",
            "typing survives as its own step"
        );
        assert_eq!(s.undo().unwrap().text, "");
    }

    #[test]
    fn typing_back_to_committed_state_drops_pending() {
        let mut s = UndoStack::new("abc");
        s.on_edit("abcd", 4);
        assert!(s.can_undo());
        s.on_edit("abc", 3); // backspaced to where we started
        assert!(!s.can_undo(), "no net change means nothing to undo");
    }

    #[test]
    fn atomic_noop_does_not_create_a_step() {
        let mut s = UndoStack::new("same");
        s.push_atomic("same", 4, "shrink");
        assert_eq!(s.len(), 1, "a shrink that changed nothing is not history");
        assert!(!s.can_undo());
    }

    #[test]
    fn history_is_capped_and_still_usable() {
        let mut s = UndoStack::new("");
        for i in 0..(CAP + 50) {
            s.push_atomic(&format!("v{i}"), 2, "typing");
        }
        assert_eq!(s.len(), CAP);
        assert!(s.undo().is_some());
    }

    #[test]
    fn reset_clears_history() {
        let mut s = UndoStack::new("a");
        s.push_atomic("b", 1, "typing");
        s.reset("fresh");
        assert!(!s.can_undo());
        assert!(!s.can_redo());
        assert_eq!(s.len(), 1);
        assert_eq!(s.committed(), "fresh");
    }
}
