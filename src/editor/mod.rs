//! The editable prompt buffer: text, caret/selection mirror, dirty tracking, undo.

pub mod undo;

use std::time::Instant;

use undo::{COALESCE_IDLE, UndoStack};

/// A character range within the buffer. Mirrors what the egui `TextEdit` reports so
/// hint requests can read the user's selection.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Selection {
    /// Start index, in characters.
    pub start: usize,
    /// End index, in characters.
    pub end: usize,
}

impl Selection {
    /// Ordered `(low, high)` bounds.
    pub fn sorted(self) -> (usize, usize) {
        if self.start <= self.end {
            (self.start, self.end)
        } else {
            (self.end, self.start)
        }
    }

    /// Whether the selection covers no characters.
    pub fn is_empty(self) -> bool {
        self.start == self.end
    }
}

/// The prompt being edited.
#[derive(Debug)]
pub struct Buffer {
    /// Live text, bound directly to the `TextEdit` widget.
    pub text: String,
    /// Caret/selection as last reported by the widget.
    pub selection: Selection,
    /// Undo/redo history.
    pub history: UndoStack,
    /// Whether `text` differs from what is on disk.
    dirty: bool,
    /// Last time the text changed, for idle-based undo coalescing and autosave.
    last_change: Option<Instant>,
}

impl Buffer {
    /// Create a buffer holding `text`, considered clean.
    pub fn new(text: impl Into<String>) -> Self {
        let text = text.into();
        Self {
            history: UndoStack::new(text.clone()),
            text,
            selection: Selection::default(),
            dirty: false,
            last_change: None,
        }
    }

    /// Replace the contents wholesale and reset history — used when opening a prompt.
    pub fn load(&mut self, text: impl Into<String>) {
        self.text = text.into();
        self.history.reset(self.text.clone());
        self.selection = Selection::default();
        self.dirty = false;
        self.last_change = None;
    }

    /// Whether there are unsaved changes.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Mark the buffer as matching disk (after a successful save).
    pub fn mark_saved(&mut self) {
        self.dirty = false;
    }

    /// The currently selected text, or `None` when the selection is empty.
    pub fn selected_text(&self) -> Option<String> {
        let (lo, hi) = self.selection.sorted();
        if lo == hi {
            return None;
        }
        let s: String = self.text.chars().skip(lo).take(hi - lo).collect();
        (!s.trim().is_empty()).then_some(s)
    }

    /// Notify the buffer that the widget edited `text`. Drives undo coalescing.
    pub fn on_widget_edit(&mut self, caret: usize) {
        self.selection = Selection {
            start: caret,
            end: caret,
        };
        self.dirty = true;
        self.last_change = Some(Instant::now());
        let text = self.text.clone();
        self.history.on_edit(&text, caret);
    }

    /// Commit pending typing to history once the buffer has been idle long enough.
    /// Call once per frame.
    pub fn tick(&mut self) {
        if let Some(t) = self.last_change
            && t.elapsed() >= COALESCE_IDLE
        {
            self.history.commit_pending();
            self.last_change = None;
        }
    }

    /// How long the buffer has been idle, for the autosave timer.
    pub fn idle_for(&self) -> Option<std::time::Duration> {
        self.last_change.map(|t| t.elapsed())
    }

    /// Apply a programmatic whole-document rewrite as a single undo step.
    pub fn replace_all(&mut self, text: impl Into<String>, label: &'static str) {
        self.text = text.into();
        let caret = self.text.chars().count();
        self.selection = Selection {
            start: caret,
            end: caret,
        };
        self.history.push_atomic(&self.text, caret, label);
        self.dirty = true;
        self.last_change = None;
    }

    /// Insert `snippet` at the caret as a single undo step.
    pub fn insert_at_caret(&mut self, snippet: &str, label: &'static str) {
        let (lo, _) = self.selection.sorted();
        let mut out: String = self.text.chars().take(lo).collect();
        out.push_str(snippet);
        out.extend(self.text.chars().skip(lo));
        let caret = lo + snippet.chars().count();
        self.text = out;
        self.selection = Selection {
            start: caret,
            end: caret,
        };
        self.history.push_atomic(&self.text, caret, label);
        self.dirty = true;
        self.last_change = None;
    }

    /// Replace the current selection with `snippet` as a single undo step.
    /// Falls back to insertion when nothing is selected.
    pub fn replace_selection(&mut self, snippet: &str, label: &'static str) {
        let (lo, hi) = self.selection.sorted();
        if lo == hi {
            return self.insert_at_caret(snippet, label);
        }
        let mut out: String = self.text.chars().take(lo).collect();
        out.push_str(snippet);
        out.extend(self.text.chars().skip(hi));
        let caret = lo + snippet.chars().count();
        self.text = out;
        self.selection = Selection {
            start: caret,
            end: caret,
        };
        self.history.push_atomic(&self.text, caret, label);
        self.dirty = true;
        self.last_change = None;
    }

    /// Undo one step, applying it to the buffer. Returns whether anything changed.
    pub fn undo(&mut self) -> bool {
        // Any in-flight keystrokes must reach history before we step back.
        self.last_change = None;
        match self.history.undo() {
            Some(snap) => {
                self.text = snap.text;
                self.selection = Selection {
                    start: snap.caret,
                    end: snap.caret,
                };
                self.dirty = true;
                true
            }
            None => false,
        }
    }

    /// Redo one step, applying it to the buffer. Returns whether anything changed.
    pub fn redo(&mut self) -> bool {
        match self.history.redo() {
            Some(snap) => {
                self.text = snap.text;
                self.selection = Selection {
                    start: snap.caret,
                    end: snap.caret,
                };
                self.dirty = true;
                true
            }
            None => false,
        }
    }
}

impl Default for Buffer {
    fn default() -> Self {
        Self::new(String::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_text_reads_the_range() {
        let mut b = Buffer::new("hello world");
        assert_eq!(b.selected_text(), None);
        b.selection = Selection { start: 6, end: 11 };
        assert_eq!(b.selected_text().as_deref(), Some("world"));
        // Reversed selections (drag right-to-left) read the same.
        b.selection = Selection { start: 11, end: 6 };
        assert_eq!(b.selected_text().as_deref(), Some("world"));
        // Whitespace-only selections are not useful as a hint subject.
        b.selection = Selection { start: 5, end: 6 };
        assert_eq!(b.selected_text(), None);
    }

    #[test]
    fn selection_is_char_indexed_not_byte_indexed() {
        // Multi-byte text would panic or slice mid-character under byte indexing.
        let mut b = Buffer::new("héllo wörld ⚙ tail");
        b.selection = Selection { start: 6, end: 11 };
        assert_eq!(b.selected_text().as_deref(), Some("wörld"));
        b.insert_at_caret("X", "hint");
        assert!(b.text.starts_with("héllo X"), "got {:?}", b.text);
    }

    #[test]
    fn insert_and_replace_are_single_undo_steps() {
        let mut b = Buffer::new("one two");
        b.selection = Selection { start: 4, end: 7 };
        b.replace_selection("TWO", "insert hint");
        assert_eq!(b.text, "one TWO");
        assert!(b.undo(), "one step reverses the whole insertion");
        assert_eq!(b.text, "one two");
        assert!(b.redo());
        assert_eq!(b.text, "one TWO");
    }

    #[test]
    fn replace_all_is_one_step_and_marks_dirty() {
        let mut b = Buffer::new("a long verbose prompt");
        assert!(!b.is_dirty());
        b.replace_all("short", "shrink");
        assert!(b.is_dirty());
        assert_eq!(b.history.undo_label(), Some("shrink"));
        b.undo();
        assert_eq!(b.text, "a long verbose prompt");
    }

    #[test]
    fn caret_moves_to_end_of_inserted_text() {
        let mut b = Buffer::new("abc");
        b.selection = Selection { start: 1, end: 1 };
        b.insert_at_caret("XY", "hint");
        assert_eq!(b.text, "aXYbc");
        assert_eq!(b.selection.start, 3, "caret sits after the insertion");
    }

    #[test]
    fn load_resets_dirty_and_history() {
        let mut b = Buffer::new("first");
        b.replace_all("edited", "typing");
        assert!(b.is_dirty());
        b.load("second");
        assert!(!b.is_dirty());
        assert!(
            !b.history.can_undo(),
            "opening a prompt starts fresh history"
        );
        assert_eq!(b.text, "second");
    }

    #[test]
    fn mark_saved_clears_dirty() {
        let mut b = Buffer::new("x");
        b.replace_all("y", "typing");
        assert!(b.is_dirty());
        b.mark_saved();
        assert!(!b.is_dirty());
    }
}
