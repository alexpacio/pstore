//! The terminal interface: the same editor, over ratatui.
//!
//! One of three front ends over [`crate::app::App`] — see [`crate`]. Nothing about ranking,
//! shrinking, sanitising, versioning or launching an agent is implemented here; this file reads
//! that state and draws it, and turns keystrokes into the same method calls the window makes.
//!
//! **Why it exists rather than being a second-class view.** The GUI needs a window server, and a
//! good deal of prompt work happens where there isn't one: over ssh on the machine that has the
//! 7.17 GB of weights on it, in a tmux pane beside the agent being driven, on a workstation whose
//! window manager is a terminal. The CLI covers scripting; it does not cover *authoring*, which
//! is iterative and needs the history, the diff and the shortlist in front of you.
//!
//! Two differences from the window, both about what a terminal can do well:
//!
//! * **Diffs are unified, not side by side.** Every proposal — shrink, plan, rca, sanitize —
//!   arrives as the same unified diff the version history shows, in one scrollable pane.
//! * **Panes rather than floating windows.** The right-hand pane is one of the ranking, the
//!   version history, or a hint answer, cycled with a key; a review takes over the centre.
//!
//! What is *not* different is the rule that nothing is applied unreviewed. `a` accepts a proposal
//! and `r` rejects it, exactly as the buttons do, and accepting is one undo step.

use std::io::{self, Stdout};
use std::time::Duration;

use crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyEventKind,
    KeyModifiers,
};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, Paragraph, Wrap};

use crate::app::App;
use crate::config::{Config, HintSource};
use crate::store::version::Note;

/// How long to wait for a key before redrawing anyway.
///
/// The redraw is not decoration: background jobs report through a channel that
/// [`App::tick`] drains, so this interval is also how quickly a finished ranking or a
/// download's progress appears. 100 ms is under the threshold where a spinner looks stuck and far
/// above the cost of drawing a few hundred cells.
const TICK: Duration = Duration::from_millis(100);

/// Which pane has the keyboard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Focus {
    /// The prompt list on the left.
    Prompts,
    /// The text.
    Editor,
}

/// What the right-hand pane is showing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Side {
    /// Nothing — the editor gets the whole width.
    Hidden,
    /// The latest ranking.
    Ranking,
    /// Version history for the open prompt.
    History,
    /// The last hint answer.
    Hint,
}

impl Side {
    /// The next pane in the cycle.
    fn next(self) -> Self {
        match self {
            Side::Hidden => Side::Ranking,
            Side::Ranking => Side::History,
            Side::History => Side::Hint,
            Side::Hint => Side::Hidden,
        }
    }

    fn title(self) -> &'static str {
        match self {
            Side::Hidden => "",
            Side::Ranking => "ranking",
            Side::History => "versions",
            Side::Hint => "hint",
        }
    }
}

/// An overlay that takes the centre and the keyboard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Overlay {
    /// The key map.
    Help,
    /// The local checkpoints and the runtime.
    Models,
    /// A proposal awaiting accept or reject.
    Review(Review),
    /// Typing a name for a new prompt.
    NewPrompt,
    /// Typing a hint question.
    Hint,
}

/// Which proposal is under review. All four are accept-or-reject over a unified diff.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Review {
    Shrink,
    Plan,
    Rca,
    Sanitize,
}

/// The terminal front end: application state, plus what a terminal needs and a window does not.
struct Tui {
    app: App,
    focus: Focus,
    side: Side,
    overlay: Option<Overlay>,
    /// First visible line of the editor.
    scroll: u16,
    /// Scroll offset within an overlay's diff.
    review_scroll: u16,
    /// Selected row in the prompt list.
    selected: usize,
    /// Text being typed into an overlay's one-line field.
    input: String,
    /// Set when the user has asked to leave.
    quit: bool,
}

/// Open the terminal interface and run until the user quits.
///
/// The terminal is restored on every path out, including a panic: a front end that leaves the
/// terminal in raw mode with the alternate screen up has effectively broken the user's shell, and
/// "run `reset`" is not an acceptable thing to make someone discover.
pub fn launch(config: Config) -> Result<(), String> {
    let mut terminal = enter().map_err(|e| format!("could not set up the terminal: {e}"))?;

    // The hook runs before the unwind reaches `launch`'s caller, so the terminal is usable by the
    // time the panic message is printed — otherwise it lands invisibly on the alternate screen.
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = leave();
        previous(info);
    }));

    let outcome = Tui::new(config).run(&mut terminal);

    let _ = leave();
    // Whatever happened to the window, the model must not outlive it.
    crate::router::shutdown_model();
    outcome.map_err(|e| format!("terminal interface: {e}"))
}

type Backend = CrosstermBackend<Stdout>;

fn enter() -> io::Result<Terminal<Backend>> {
    enable_raw_mode()?;
    let mut out = io::stdout();
    execute!(out, EnterAlternateScreen, EnableMouseCapture)?;
    Terminal::new(CrosstermBackend::new(out))
}

fn leave() -> io::Result<()> {
    disable_raw_mode()?;
    execute!(io::stdout(), LeaveAlternateScreen, DisableMouseCapture)
}

impl Tui {
    fn new(config: Config) -> Self {
        let app = App::new(config);
        Self {
            app,
            focus: Focus::Editor,
            side: Side::Hidden,
            overlay: None,
            scroll: 0,
            review_scroll: 0,
            selected: 0,
            input: String::new(),
            quit: false,
        }
    }

    fn run(mut self, terminal: &mut Terminal<Backend>) -> io::Result<()> {
        while !self.quit {
            // Drains job events, commits undo granules, runs the autosave timer — the same call
            // the window makes once per frame.
            self.app.tick();
            self.sync_overlay();
            terminal.draw(|frame| self.draw(frame))?;

            if event::poll(TICK)?
                && let Event::Key(key) = event::read()?
                && key.kind == KeyEventKind::Press
            {
                self.on_key(key);
            }
        }
        // A dirty buffer on the way out is saved rather than lost. The window does the same, and
        // in a terminal it matters more: Ctrl+Q is next to everything.
        if self.app.buffer.is_dirty() {
            self.app.save(Note::Manual);
        }
        Ok(())
    }

    /// Open a review when a job has produced one, and close it when the proposal is gone.
    ///
    /// Proposals arrive from worker threads through [`App::tick`], so there is no keystroke to
    /// hang the overlay off — it follows the state instead, which also means accepting or
    /// rejecting through any path closes it.
    fn sync_overlay(&mut self) {
        let pending = if self.app.shrink.is_some() {
            Some(Overlay::Review(Review::Shrink))
        } else if self.app.plan.is_some() {
            Some(Overlay::Review(Review::Plan))
        } else if self.app.rca.is_some() {
            Some(Overlay::Review(Review::Rca))
        } else if self.app.pii.is_some() {
            Some(Overlay::Review(Review::Sanitize))
        } else {
            None
        };

        match (self.overlay, pending) {
            // A new proposal takes the centre, from whatever was there.
            (_, Some(review)) if self.overlay != pending => {
                self.overlay = Some(review);
                self.review_scroll = 0;
            }
            // The proposal it was showing has been dealt with.
            (Some(Overlay::Review(_)), None) => self.overlay = None,
            _ => {}
        }
    }

    // -----------------------------------------------------------------------
    // Keys
    // -----------------------------------------------------------------------

    fn on_key(&mut self, key: KeyEvent) {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);

        // Quit is checked before anything else can claim the key. It used to be checked after the
        // overlay dispatch, where `q` closes the help pane — so Ctrl+Q with help open closed the
        // help and stayed in the app. A quit binding that silently does something else is the
        // worst kind of nearly-working.
        if ctrl && key.code == KeyCode::Char('q') {
            self.quit = true;
            return;
        }

        // An error is modal in the sense that it must be acknowledged, but it must never trap
        // anyone: any key clears it and is then handled normally.
        if self.app.error.is_some() {
            self.app.error = None;
            if matches!(key.code, KeyCode::Esc | KeyCode::Enter) {
                return;
            }
        }

        if let Some(overlay) = self.overlay
            && self.on_overlay_key(overlay, key, ctrl)
        {
            return;
        }

        match (key.code, ctrl) {
            (KeyCode::Char('s'), true) => self.app.save(Note::Manual),
            (KeyCode::Char('r'), true) => self.app.rank(),
            (KeyCode::Char('z'), true) => {
                if self.app.buffer.undo() {
                    self.app.status = "undo".into();
                }
            }
            (KeyCode::Char('y'), true) => {
                if self.app.buffer.redo() {
                    self.app.status = "redo".into();
                }
            }
            (KeyCode::Char('n'), true) => {
                self.input.clear();
                self.overlay = Some(Overlay::NewPrompt);
            }
            (KeyCode::Char('h'), true) => {
                self.input.clear();
                self.overlay = Some(Overlay::Hint);
            }
            (KeyCode::Tab, _) => {
                self.focus = match self.focus {
                    Focus::Editor => Focus::Prompts,
                    Focus::Prompts => Focus::Editor,
                }
            }
            (KeyCode::F(1), _) => self.overlay = Some(Overlay::Help),
            (KeyCode::F(2), _) => self.app.request_shrink(),
            (KeyCode::F(3), _) => self.app.request_plan(),
            (KeyCode::F(4), _) => self.app.request_sanitize(),
            (KeyCode::F(5), _) => self.app.rank(),
            (KeyCode::F(6), _) => self.overlay = Some(Overlay::Models),
            (KeyCode::F(7), _) => self.side = Side::History,
            (KeyCode::F(8), _) => self.app.send_to_agent(),
            (KeyCode::F(9), _) => self.side = self.side.next(),
            (KeyCode::F(10), _) => self.app.request_rca(),
            (KeyCode::Esc, _) => {
                // Nothing to close: stop whatever is running, which is the other thing Esc is
                // for in every editor.
                self.cancel_running();
            }
            _ => match self.focus {
                Focus::Prompts => self.on_list_key(key),
                Focus::Editor => self.on_editor_key(key, ctrl),
            },
        }
    }

    /// Handle a key for the active overlay. Returns whether it was consumed.
    fn on_overlay_key(&mut self, overlay: Overlay, key: KeyEvent, ctrl: bool) -> bool {
        match overlay {
            // `q` unmodified only: with `ctrl` folded in, this arm ate Ctrl+Q. Quit is handled
            // before this function is reached, and this guard is what keeps it that way.
            Overlay::Help | Overlay::Models => match key.code {
                KeyCode::Esc | KeyCode::Enter | KeyCode::F(1) => {
                    self.overlay = None;
                    true
                }
                KeyCode::Char('q') if !ctrl => {
                    self.overlay = None;
                    true
                }
                _ => false,
            },
            Overlay::Review(review) => match key.code {
                KeyCode::Char('a') => {
                    match review {
                        Review::Shrink => self.app.accept_shrink(),
                        Review::Plan => self.app.accept_plan(),
                        Review::Rca => self.app.accept_rca(),
                        Review::Sanitize => self.app.accept_sanitize(),
                    }
                    self.overlay = None;
                    true
                }
                KeyCode::Char('r') | KeyCode::Esc => {
                    match review {
                        // Rejecting is dropping the proposal. The buffer is untouched either
                        // way — nothing has been applied yet.
                        Review::Shrink => self.app.shrink = None,
                        Review::Plan => self.app.reject_plan(),
                        Review::Rca => self.app.reject_rca(),
                        Review::Sanitize => self.app.pii = None,
                    }
                    self.overlay = None;
                    true
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    self.review_scroll = self.review_scroll.saturating_add(1);
                    true
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    self.review_scroll = self.review_scroll.saturating_sub(1);
                    true
                }
                KeyCode::PageDown => {
                    self.review_scroll = self.review_scroll.saturating_add(15);
                    true
                }
                KeyCode::PageUp => {
                    self.review_scroll = self.review_scroll.saturating_sub(15);
                    true
                }
                _ => false,
            },
            Overlay::NewPrompt | Overlay::Hint => match key.code {
                KeyCode::Esc => {
                    self.overlay = None;
                    self.input.clear();
                    true
                }
                KeyCode::Enter => {
                    let text = std::mem::take(&mut self.input);
                    self.overlay = None;
                    match overlay {
                        Overlay::NewPrompt if !text.trim().is_empty() => {
                            self.app.create_prompt(text.trim());
                            self.selected = self.app.open.unwrap_or(0);
                        }
                        Overlay::Hint => {
                            self.app.hint_input = text;
                            self.app.request_hint();
                            self.side = Side::Hint;
                        }
                        _ => {}
                    }
                    true
                }
                KeyCode::Backspace => {
                    self.input.pop();
                    true
                }
                // Who answers, switched from inside the question box: it is a property of
                // the question being typed, not a setting to go and find.
                KeyCode::Char('l') if ctrl && overlay == Overlay::Hint => {
                    let prefs = &mut self.app.config.prefs;
                    prefs.hint_source = match prefs.hint_source {
                        HintSource::Local => HintSource::Agent,
                        HintSource::Agent => HintSource::Local,
                    };
                    prefs.save(&self.app.config.dir);
                    true
                }
                KeyCode::Char(c) if !ctrl => {
                    self.input.push(c);
                    true
                }
                _ => false,
            },
        }
    }

    fn on_list_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => self.selected = self.selected.saturating_sub(1),
            KeyCode::Down | KeyCode::Char('j') => {
                self.selected = (self.selected + 1).min(self.app.prompts.len().saturating_sub(1));
            }
            KeyCode::Enter if self.selected < self.app.prompts.len() => {
                self.app.open_prompt(self.selected);
                self.scroll = 0;
                self.focus = Focus::Editor;
            }
            _ => {}
        }
    }

    fn on_editor_key(&mut self, key: KeyEvent, ctrl: bool) {
        let buffer = &mut self.app.buffer;
        let caret = buffer.caret();
        let text = buffer.text.clone();
        let extend = key.modifiers.contains(KeyModifiers::SHIFT);

        match key.code {
            KeyCode::Char(c) if !ctrl => buffer.type_char(c),
            KeyCode::Enter => buffer.type_char('\n'),
            KeyCode::Tab => buffer.type_char('\t'),
            KeyCode::Backspace => buffer.backspace(),
            KeyCode::Delete => buffer.delete(),
            KeyCode::Left => buffer.move_caret(caret.saturating_sub(1), extend),
            KeyCode::Right => buffer.move_caret(caret + 1, extend),
            KeyCode::Home => buffer.move_caret(line_start(&text, caret), extend),
            KeyCode::End => buffer.move_caret(line_end(&text, caret), extend),
            KeyCode::Up => {
                let to = step_line(&text, caret, -1);
                buffer.move_caret(to, extend);
            }
            KeyCode::Down => {
                let to = step_line(&text, caret, 1);
                buffer.move_caret(to, extend);
            }
            KeyCode::PageUp => {
                let to = step_line(&text, caret, -15);
                buffer.move_caret(to, extend);
            }
            KeyCode::PageDown => {
                let to = step_line(&text, caret, 15);
                buffer.move_caret(to, extend);
            }
            _ => {}
        }
        self.follow_caret();
    }

    /// Keep the caret on screen. Called after every edit and movement.
    fn follow_caret(&mut self) {
        let row = row_of(&self.app.buffer.text, self.app.buffer.caret()) as u16;
        // A viewport height is not known here, so hold the caret a few lines inside whatever it
        // is: scrolling exactly to the edge makes the next keystroke jump.
        if row < self.scroll {
            self.scroll = row;
        } else if row > self.scroll + 20 {
            self.scroll = row.saturating_sub(20);
        }
    }

    /// Stop whatever is running, so a 30-second ranking is not something to sit through.
    fn cancel_running(&mut self) {
        if self.app.shrink_job.is_some() {
            self.app.cancel_shrink();
        } else if self.app.hint.as_ref().is_some_and(|h| h.job.is_some()) {
            self.app.cancel_hint();
        } else if self.app.pii_job.is_some() {
            self.app.cancel_sanitize();
        } else if self.app.plan_job.is_some() {
            self.app.cancel_plan();
        } else if self.app.rca_job.is_some() {
            self.app.cancel_rca();
        } else if self.app.models_job.is_some() {
            self.app.cancel_models();
        }
    }

    // -----------------------------------------------------------------------
    // Drawing
    // -----------------------------------------------------------------------

    fn draw(&mut self, frame: &mut ratatui::Frame) {
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1), // title
                Constraint::Min(3),    // body
                Constraint::Length(1), // status
                Constraint::Length(1), // keys
            ])
            .split(frame.area());

        self.draw_title(frame, rows[0]);
        self.draw_body(frame, rows[1]);
        self.draw_status(frame, rows[2]);
        self.draw_keys(frame, rows[3]);

        if let Some(overlay) = self.overlay {
            self.draw_overlay(frame, overlay, rows[1]);
        }
        if let Some(error) = self.app.error.clone() {
            self.draw_error(frame, &error, rows[1]);
        }
    }

    fn draw_title(&self, frame: &mut ratatui::Frame, area: Rect) {
        let open = self.app.current();
        let title =
            crate::app::window_title(&self.app.config.dir, open, self.app.buffer.is_dirty());
        frame.render_widget(
            Paragraph::new(title).style(Style::default().add_modifier(Modifier::REVERSED)),
            area,
        );
    }

    fn draw_body(&mut self, frame: &mut ratatui::Frame, area: Rect) {
        let mut widths = vec![
            Constraint::Length(self.app.config.prefs.sidebar_width.clamp(16.0, 60.0) as u16),
            Constraint::Min(20),
        ];
        if self.side != Side::Hidden {
            widths.push(Constraint::Percentage(38));
        }
        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(widths)
            .split(area);

        self.draw_prompts(frame, cols[0]);
        self.draw_editor(frame, cols[1]);
        if self.side != Side::Hidden {
            self.draw_side(frame, cols[2]);
        }
    }

    fn draw_prompts(&self, frame: &mut ratatui::Frame, area: Rect) {
        let items: Vec<ListItem> = self
            .app
            .prompts
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let open = self.app.open == Some(i);
                let mark = if open { "● " } else { "  " };
                let style = if i == self.selected && self.focus == Focus::Prompts {
                    Style::default().add_modifier(Modifier::REVERSED)
                } else if open {
                    Style::default().add_modifier(Modifier::BOLD)
                } else {
                    Style::default()
                };
                ListItem::new(format!("{mark}{}", p.name)).style(style)
            })
            .collect();

        let title = if self.app.prompts.is_empty() {
            "prompts — Ctrl+N to create one".to_string()
        } else {
            format!("prompts ({})", self.app.prompts.len())
        };
        frame.render_widget(
            List::new(items).block(bordered(&title, self.focus == Focus::Prompts)),
            area,
        );
    }

    fn draw_editor(&mut self, frame: &mut ratatui::Frame, area: Rect) {
        let block = bordered(
            &match self.app.current() {
                Some(p) => p.name.clone(),
                None => "no prompt open".to_string(),
            },
            self.focus == Focus::Editor,
        );
        let inner = block.inner(area);
        frame.render_widget(block, area);

        let text = &self.app.buffer.text;
        frame.render_widget(
            Paragraph::new(text.as_str())
                .wrap(Wrap { trim: false })
                .scroll((self.scroll, 0)),
            inner,
        );

        // The real caret rather than a drawn block: a terminal has one, and the shell's own
        // cursor shape and blink are what the user has configured.
        if self.focus == Focus::Editor && self.overlay.is_none() {
            let caret = self.app.buffer.caret();
            let row = row_of(text, caret) as u16;
            let col = (caret - line_start(text, caret)) as u16;
            if row >= self.scroll && row < self.scroll + inner.height {
                frame.set_cursor_position((
                    inner.x + col.min(inner.width.saturating_sub(1)),
                    inner.y + row - self.scroll,
                ));
            }
        }
    }

    fn draw_side(&self, frame: &mut ratatui::Frame, area: Rect) {
        let block = bordered(self.side.title(), false);
        let inner = block.inner(area);
        frame.render_widget(block, area);

        let body: Text = match self.side {
            Side::Hidden => Text::default(),
            Side::Ranking => self.ranking_text(),
            Side::History => self.history_text(),
            Side::Hint => match &self.app.hint {
                Some(h) => {
                    let mut lines = vec![Line::from(Span::styled(
                        h.answered_by.clone().unwrap_or_else(|| "asking…".into()),
                        Style::default().add_modifier(Modifier::DIM),
                    ))];
                    lines.extend(h.answer.lines().map(Line::from));
                    Text::from(lines)
                }
                None => Text::from("Ctrl+H asks about the selection, a question, or both"),
            },
        };
        frame.render_widget(Paragraph::new(body).wrap(Wrap { trim: false }), inner);
    }

    fn ranking_text(&self) -> Text<'static> {
        let Some(ranking) = &self.app.ranking else {
            return Text::from("Ctrl+R ranks the installed models against this prompt");
        };
        let mut lines = vec![Line::from(Span::styled(
            format!(
                "top {} of {} · {:.0}s",
                ranking.choices.len(),
                ranking.considered,
                ranking.elapsed.as_secs_f32()
            ),
            Style::default().add_modifier(Modifier::DIM),
        ))];
        if let Some(judged) = &ranking.judged {
            lines.push(Line::from(vec![
                Span::styled(
                    format!("judged {}", judged.summary()),
                    Style::default().add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    judged.because_suffix(),
                    Style::default().add_modifier(Modifier::DIM),
                ),
            ]));
        }

        // Said before the table, not after it: a degenerate answer looks exactly like a real one
        // in the rows below.
        if let Some(why) = &ranking.degenerate {
            lines.push(Line::from(Span::styled(
                format!("! a list, not a ranking — {why}"),
                Style::default().fg(Color::Yellow),
            )));
        }
        for c in &ranking.choices {
            lines.push(Line::from(vec![
                Span::styled(
                    format!("{:>3} ", c.fit),
                    Style::default().add_modifier(Modifier::BOLD),
                ),
                Span::raw(format!("{} · {}", c.model_display, c.agent_display)),
                Span::styled(
                    format!(
                        " · {}{}",
                        if c.effort_selectable { "" } else { "~" },
                        c.effort
                    ),
                    Style::default().add_modifier(Modifier::DIM),
                ),
            ]));
            // The GUI puts the costs in their own columns with a tooltip each. There is no hover
            // in a terminal, so they go inline — the numbers matter more than the width they
            // cost, and a shortlist that hides what a pick spends is the thing being fixed.
            let mut cost = format!("{:.1}× price", c.relative_price);
            if c.quota_weight >= 2.0 {
                cost.push_str(&format!(", {:.0}× quota", c.quota_weight));
            }
            if c.metered {
                cost.push_str(", PAID PER TOKEN");
            }
            lines.push(Line::from(Span::styled(
                format!("    {}", c.rationale),
                Style::default().add_modifier(Modifier::DIM),
            )));
            lines.push(Line::from(Span::styled(
                format!("    {cost}"),
                Style::default().add_modifier(Modifier::DIM),
            )));
        }
        for (id, why) in &ranking.excluded {
            lines.push(Line::from(Span::styled(
                format!("- {id}: {why}"),
                Style::default().add_modifier(Modifier::DIM),
            )));
        }
        Text::from(lines)
    }

    fn history_text(&self) -> Text<'static> {
        let view = &self.app.history;
        if view.versions.is_empty() {
            return Text::from("no saved versions yet — Ctrl+S takes a snapshot");
        }
        let mut lines: Vec<Line> = view
            .versions
            .iter()
            .map(|v| {
                Line::from(format!(
                    "{}  {:<10} {} bytes",
                    v.ts,
                    v.note.label(),
                    v.bytes
                ))
            })
            .collect();
        if !view.diff.is_empty() {
            lines.push(Line::from(""));
            lines.extend(view.diff.lines().map(diff_line));
        }
        Text::from(lines)
    }

    fn draw_status(&self, frame: &mut ratatui::Frame, area: Rect) {
        let busy = self.app.models_job.is_some()
            || self.app.shrink_job.is_some()
            || self.app.plan_job.is_some()
            || self.app.rca_job.is_some()
            || self.app.pii_job.is_some();
        let prefix = if busy { "… " } else { "" };
        frame.render_widget(
            Paragraph::new(format!("{prefix}{}", self.app.status))
                .style(Style::default().add_modifier(Modifier::DIM)),
            area,
        );
    }

    fn draw_keys(&self, frame: &mut ratatui::Frame, area: Rect) {
        let keys = match self.overlay {
            Some(Overlay::Review(_)) => "a accept · r reject · ↑↓ scroll",
            Some(Overlay::NewPrompt) | Some(Overlay::Hint) => "Enter confirm · Esc cancel",
            Some(_) => "Esc close",
            None => {
                "F1 help · F2 shrink · F3 plan · F4 sanitize · F5 rank · F6 models · \
                 F9 pane · F10 rca · ^S save · ^N new · ^H hint · ^Q quit"
            }
        };
        frame.render_widget(
            Paragraph::new(keys).style(Style::default().add_modifier(Modifier::REVERSED)),
            area,
        );
    }

    fn draw_overlay(&mut self, frame: &mut ratatui::Frame, overlay: Overlay, area: Rect) {
        let (title, body): (String, Text) = match overlay {
            Overlay::Help => ("keys".into(), help_text()),
            Overlay::Models => ("local model".into(), self.models_text()),
            Overlay::Review(review) => self.review_text(review),
            Overlay::NewPrompt => (
                "new prompt".into(),
                Text::from(format!("name: {}▏", self.input)),
            ),
            Overlay::Hint => (
                "ask about this prompt".into(),
                Text::from(vec![
                    Line::from(format!("question: {}▏", self.input)),
                    Line::from(Span::styled(
                        match self.app.buffer.selected_text() {
                            Some(s) => format!("with the selection ({} chars)", s.len()),
                            None => "no selection — the question stands alone".into(),
                        },
                        Style::default().add_modifier(Modifier::DIM),
                    )),
                    Line::from(Span::styled(
                        format!(
                            "answered by: {}  (Ctrl+L to switch)",
                            match self.app.config.prefs.hint_source {
                                HintSource::Local => "the local model",
                                HintSource::Agent => "the ranked coding agent",
                            }
                        ),
                        Style::default().add_modifier(Modifier::DIM),
                    )),
                ]),
            ),
        };

        let area = centred(area, 84, 80);
        frame.render_widget(Clear, area);
        let block = bordered(&title, true);
        let inner = block.inner(area);
        frame.render_widget(block, area);
        frame.render_widget(
            Paragraph::new(body)
                .wrap(Wrap { trim: false })
                .scroll((self.review_scroll, 0)),
            inner,
        );
    }

    /// The diff and the warnings for a proposal, whichever kind it is.
    fn review_text(&self, review: Review) -> (String, Text<'static>) {
        let mut lines: Vec<Line> = Vec::new();
        let title = match review {
            Review::Shrink => {
                let Some(p) = &self.app.shrink else {
                    return ("shrink".into(), Text::default());
                };
                lines.push(Line::from(p.savings.summary()));
                warn_lines(&mut lines, &p.warnings);
                lines.extend(p.diff.lines().map(diff_line));
                "shrink — accept?"
            }
            Review::Plan => {
                let Some(p) = &self.app.plan else {
                    return ("plan".into(), Text::default());
                };
                warn_lines(&mut lines, &p.warnings);
                lines.extend(p.diff.lines().map(diff_line));
                "plan — accept?"
            }
            Review::Rca => {
                let Some(p) = &self.app.rca else {
                    return ("rca".into(), Text::default());
                };
                warn_lines(&mut lines, &p.warnings);
                lines.extend(p.diff.lines().map(diff_line));
                "postmortem — accept?"
            }
            Review::Sanitize => {
                let Some(p) = &self.app.pii else {
                    return ("sanitize".into(), Text::default());
                };
                lines.push(Line::from(p.scan.plan.summary()));
                for item in &p.scan.plan.items {
                    lines.push(Line::from(Span::styled(
                        format!(
                            "  {:<10} {} → {}",
                            item.finding.tag,
                            item.finding.text,
                            if item.masked {
                                item.placeholder.as_str()
                            } else {
                                "(left as is)"
                            }
                        ),
                        Style::default().add_modifier(Modifier::DIM),
                    )));
                }
                lines.push(Line::from(""));
                lines.extend(p.diff.lines().map(diff_line));
                "personal data — mask it?"
            }
        };
        (title.into(), Text::from(lines))
    }

    fn models_text(&self) -> Text<'static> {
        let prefs = crate::config::prefs_snapshot();
        let mut lines = vec![match crate::runtime::locate(prefs.llama_path.as_deref()) {
            Some(rt) => Line::from(format!("runtime: {} ({})", rt.path.display(), rt.origin)),
            None => Line::from(Span::styled(
                format!(
                    "runtime: {}",
                    crate::runtime::missing_reason(prefs.llama_path.as_deref())
                ),
                Style::default().fg(Color::Yellow),
            )),
        }];
        lines.push(Line::from(""));

        let active = crate::models::active();
        for (c, phase) in crate::models::snapshot() {
            let marker = if c.id == active.id { "→" } else { " " };
            lines.push(Line::from(format!(
                "{marker} {:<24} {:<10} {}",
                c.title,
                c.size_label(),
                phase.label()
            )));
            lines.push(Line::from(Span::styled(
                format!("    {}", crate::models::tradeoff(c.id)),
                Style::default().add_modifier(Modifier::DIM),
            )));
        }
        lines.push(Line::from(""));
        // Downloading gigabytes is not something to start from a keystroke in a pane this small,
        // and saying where it is done beats a progress bar nobody asked for.
        lines.push(Line::from(Span::styled(
            "downloads and the build switch live in the window (`pstore`) and in \
             .pstore/config.json",
            Style::default().add_modifier(Modifier::DIM),
        )));
        Text::from(lines)
    }

    fn draw_error(&self, frame: &mut ratatui::Frame, error: &str, area: Rect) {
        let area = centred(area, 70, 40);
        frame.render_widget(Clear, area);
        frame.render_widget(
            Paragraph::new(error.to_string())
                .wrap(Wrap { trim: false })
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title(" problem — any key dismisses ")
                        .border_style(Style::default().fg(Color::Red)),
                ),
            area,
        );
    }
}

// ---------------------------------------------------------------------------
// Presentation helpers
// ---------------------------------------------------------------------------

fn bordered(title: &str, focused: bool) -> Block<'static> {
    Block::default()
        .borders(Borders::ALL)
        .title(format!(" {title} "))
        .border_style(if focused {
            Style::default().fg(Color::Cyan)
        } else {
            Style::default().add_modifier(Modifier::DIM)
        })
}

/// Colour a unified-diff line by its marker.
fn diff_line(raw: &str) -> Line<'static> {
    let style = match raw.chars().next() {
        Some('+') => Style::default().fg(Color::Green),
        Some('-') => Style::default().fg(Color::Red),
        Some('@') => Style::default().fg(Color::Cyan),
        _ => Style::default(),
    };
    Line::from(Span::styled(raw.to_string(), style))
}

fn warn_lines(lines: &mut Vec<Line<'static>>, warnings: &[String]) {
    for w in warnings {
        lines.push(Line::from(Span::styled(
            format!("! {w}"),
            Style::default().fg(Color::Yellow),
        )));
    }
    if !warnings.is_empty() {
        lines.push(Line::from(""));
    }
}

fn help_text() -> Text<'static> {
    let rows = [
        ("Tab", "move between the prompt list and the text"),
        ("Ctrl+N / Enter", "new prompt / open the selected one"),
        ("Ctrl+S", "save — every save is a version"),
        ("Ctrl+Z / Ctrl+Y", "undo / redo"),
        (
            "Ctrl+R or F5",
            "rank the installed models against this prompt",
        ),
        (
            "F2",
            "shrink: rewrite the selection, or all of it, telegraphically",
        ),
        (
            "F3",
            "plan: rewrite it as an instruction for a coding agent (local model)",
        ),
        ("F4", "sanitize: find personal data and offer to mask it"),
        (
            "F10",
            "rca: turn incident notes into a postmortem and action items (local model)",
        ),
        ("Ctrl+H", "ask about the selection, a question, or both"),
        ("Ctrl+L", "in the hint box: local model or coding agent"),
        ("F8", "send the prompt to the best-ranked agent"),
        ("F6", "what the local model and its runtime are doing"),
        ("F7 / F9", "version history / cycle the right-hand pane"),
        ("Esc", "stop whatever is running"),
        ("Ctrl+Q", "quit — an unsaved buffer is saved first"),
        ("a / r", "in a review: accept or reject the proposal"),
    ];
    let mut lines: Vec<Line> = rows
        .iter()
        .map(|(key, what)| {
            Line::from(vec![
                Span::styled(
                    format!("{key:<16}"),
                    Style::default().add_modifier(Modifier::BOLD),
                ),
                Span::raw(*what),
            ])
        })
        .collect();
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "Ranking, shrinking, planning, analysing an incident and sanitising all run the same code as \
         model runs on this machine and nothing about your prompt leaves it.",
        Style::default().add_modifier(Modifier::DIM),
    )));
    Text::from(lines)
}

/// A rectangle `pct_x` × `pct_y` percent of `area`, centred in it.
fn centred(area: Rect, pct_x: u16, pct_y: u16) -> Rect {
    let h = area.height * pct_y / 100;
    let w = area.width * pct_x / 100;
    Rect {
        x: area.x + (area.width.saturating_sub(w)) / 2,
        y: area.y + (area.height.saturating_sub(h)) / 2,
        width: w.min(area.width),
        height: h.min(area.height),
    }
}

// ---------------------------------------------------------------------------
// Caret arithmetic
// ---------------------------------------------------------------------------
//
// All of it in characters, matching [`crate::editor::Selection`], and all of it here rather than
// in the buffer: "up one line" depends on how the text is laid out, which is a property of the
// front end and not of the document.

/// Character index of the start of the line containing `caret`.
fn line_start(text: &str, caret: usize) -> usize {
    text.chars()
        .take(caret)
        .collect::<Vec<_>>()
        .iter()
        .rposition(|c| *c == '\n')
        .map_or(0, |i| i + 1)
}

/// Character index of the end of the line containing `caret`.
fn line_end(text: &str, caret: usize) -> usize {
    let chars: Vec<char> = text.chars().collect();
    chars
        .iter()
        .skip(caret)
        .position(|c| *c == '\n')
        .map_or(chars.len(), |i| caret + i)
}

/// Which line `caret` is on, counting from zero.
fn row_of(text: &str, caret: usize) -> usize {
    text.chars().take(caret).filter(|c| *c == '\n').count()
}

/// Move `caret` `delta` lines, keeping the column where the new line is long enough.
fn step_line(text: &str, caret: usize, delta: isize) -> usize {
    let column = caret - line_start(text, caret);
    let lines: Vec<&str> = text.split('\n').collect();
    let row = row_of(text, caret) as isize;
    let target = (row + delta).clamp(0, lines.len().saturating_sub(1) as isize) as usize;

    // Sum the lines before the target, plus one for each newline.
    let start: usize = lines[..target].iter().map(|l| l.chars().count() + 1).sum();
    start + column.min(lines[target].chars().count())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEXT: &str = "first line\nsecond\n\nfourth line here";

    #[test]
    fn line_bounds_are_found_in_characters() {
        assert_eq!(line_start(TEXT, 0), 0);
        assert_eq!(line_start(TEXT, 5), 0);
        assert_eq!(line_end(TEXT, 0), 10, "before the newline");
        // Second line runs 11..17.
        assert_eq!(line_start(TEXT, 13), 11);
        assert_eq!(line_end(TEXT, 13), 17);
        // The empty third line is a single position.
        assert_eq!(line_start(TEXT, 18), 18);
        assert_eq!(line_end(TEXT, 18), 18);
        // The last line has no trailing newline to stop at.
        assert_eq!(line_end(TEXT, 20), TEXT.chars().count());
    }

    #[test]
    fn rows_are_counted_from_zero() {
        assert_eq!(row_of(TEXT, 0), 0);
        assert_eq!(row_of(TEXT, 13), 1);
        assert_eq!(row_of(TEXT, 18), 2);
        assert_eq!(row_of(TEXT, 25), 3);
    }

    /// Vertical movement keeps the column where it can, which is what makes arrow keys feel
    /// right — and clamps at both ends rather than running off the buffer.
    #[test]
    fn vertical_movement_keeps_the_column_and_clamps() {
        // Column 7 of line 0 → line 1 is only 6 long, so it lands at its end.
        assert_eq!(step_line(TEXT, 7, 1), 17);
        // From there down onto the empty line.
        assert_eq!(step_line(TEXT, 17, 1), 18);
        // And onto line 3, which is long enough to keep the column.
        assert_eq!(step_line(TEXT, 13, 2), 19 + 2);

        // Up from the first line stays on it rather than going negative.
        assert_eq!(step_line(TEXT, 5, -1), 5);
        // Down from the last stays on it.
        let last = TEXT.chars().count();
        assert_eq!(step_line(TEXT, last, 1), last);
        // A page-sized jump clamps too.
        assert_eq!(step_line(TEXT, 0, 100), 19);
    }

    /// Multi-byte text is where character and byte indices diverge, and where getting this wrong
    /// panics on the next edit rather than merely looking odd.
    #[test]
    fn caret_arithmetic_is_in_characters_not_bytes() {
        let text = "società\nnaïve caffè\nx";
        assert_eq!(line_start(text, 9), 8);
        assert_eq!(line_end(text, 9), 19, "11 characters, not 13 bytes");
        assert_eq!(row_of(text, 9), 1);
        assert_eq!(step_line(text, 9, -1), 1, "column 1 of the first line");
    }

    /// A front end over a temporary prompt folder. No terminal involved: [`Tui::on_key`] is
    /// where the behaviour worth testing lives, and it only needs the state.
    fn harness(tag: &str) -> Tui {
        let dir = std::env::temp_dir().join(format!(
            "pstore-tui-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("one.md"), "first prompt").unwrap();

        Tui::new(Config {
            dir,
            prefs: crate::config::Prefs::default(),
            warnings: Vec::new(),
        })
    }

    fn press(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn ctrl(c: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(c), KeyModifiers::CONTROL)
    }

    /// Regression, found by driving the real binary through a pty: with the help pane open, `q`
    /// closed it — and Ctrl+Q matched the same arm, so the quit key closed the help and left the
    /// user in the app. Quit now outranks every overlay.
    #[test]
    fn ctrl_q_quits_even_with_an_overlay_open() {
        let mut tui = harness("quit");

        tui.on_key(press(KeyCode::F(1)));
        assert_eq!(tui.overlay, Some(Overlay::Help), "F1 should open the help");

        tui.on_key(ctrl('q'));
        assert!(tui.quit, "Ctrl+Q was swallowed by the overlay");

        // Plain `q` still closes an overlay rather than typing into the document behind it.
        let mut tui = harness("close");
        tui.on_key(press(KeyCode::F(6)));
        assert_eq!(tui.overlay, Some(Overlay::Models));
        tui.on_key(press(KeyCode::Char('q')));
        assert_eq!(tui.overlay, None);
        assert!(!tui.quit);
        assert!(
            !tui.app.buffer.text.contains('q'),
            "the key that closed the overlay also reached the buffer"
        );
    }

    /// The action keys have to reach the actions. Each of these is one line in the key bar, and a
    /// binding that silently does nothing is indistinguishable from a broken feature.
    #[test]
    fn the_action_keys_are_wired() {
        let mut tui = harness("keys");
        assert!(tui.app.open.is_some(), "the fixture prompt should be open");

        // Overlays that are pure display.
        for (key, want) in [
            (KeyCode::F(1), Overlay::Help),
            (KeyCode::F(6), Overlay::Models),
        ] {
            tui.overlay = None;
            tui.on_key(press(key));
            assert_eq!(tui.overlay, Some(want), "{key:?} did not open its overlay");
        }

        // The two that take typed input, and Esc backing out of them.
        tui.overlay = None;
        tui.on_key(ctrl('n'));
        assert_eq!(tui.overlay, Some(Overlay::NewPrompt));
        tui.on_key(press(KeyCode::Char('x')));
        assert_eq!(tui.input, "x", "typing should go to the overlay's field");
        tui.on_key(press(KeyCode::Esc));
        assert_eq!(tui.overlay, None);
        assert!(tui.input.is_empty(), "a cancelled field should not persist");

        tui.on_key(ctrl('h'));
        assert_eq!(tui.overlay, Some(Overlay::Hint));
        tui.on_key(press(KeyCode::Esc));

        // Tab moves the focus, and the pane key cycles.
        assert_eq!(tui.focus, Focus::Editor);
        tui.on_key(press(KeyCode::Tab));
        assert_eq!(tui.focus, Focus::Prompts);
        tui.on_key(press(KeyCode::Tab));
        assert_eq!(tui.focus, Focus::Editor);

        let before = tui.side;
        tui.on_key(press(KeyCode::F(9)));
        assert_ne!(tui.side, before, "F9 should change the right-hand pane");
    }

    /// Typing has to reach the buffer, and it has to coalesce into undoable words rather than
    /// per-keystroke steps — which is why the edit primitives live on `Buffer` and not here.
    #[test]
    fn typing_edits_the_buffer_and_undoes_as_one_step() {
        let mut tui = harness("typing");
        let original = tui.app.buffer.text.clone();

        tui.on_key(press(KeyCode::Home));
        for c in "note".chars() {
            tui.on_key(press(KeyCode::Char(c)));
        }
        assert!(
            tui.app.buffer.text.starts_with("note"),
            "typing did not reach the buffer: {:?}",
            tui.app.buffer.text
        );
        assert!(tui.app.buffer.is_dirty());

        // Four keystrokes, one undo step: the granule is a word, not a character. This is the
        // property that makes the edit primitives belong on `Buffer` — a front end mutating the
        // text itself would get per-keystroke undo and nobody would notice until they used it.
        tui.on_key(ctrl('z'));
        assert_eq!(
            tui.app.buffer.text, original,
            "a word of typing should be one undo step"
        );

        // Backspace and delete both act on the document, and at a boundary neither runs off it.
        tui.on_key(press(KeyCode::Home));
        tui.on_key(press(KeyCode::Backspace));
        assert_eq!(
            tui.app.buffer.text, original,
            "backspace at the start of the buffer should do nothing"
        );
        tui.on_key(press(KeyCode::Delete));
        assert_eq!(tui.app.buffer.text, original[1..]);
    }

    /// A proposal arriving from a worker thread has to take the centre by itself — there is no
    /// keystroke to hang it off — and accepting or rejecting it has to give the editor back.
    #[test]
    fn a_proposal_opens_and_closes_the_review_overlay() {
        let mut tui = harness("review");

        tui.app.plan = Some(crate::app::PlanProposal {
            after: "Objective: do the thing".into(),
            warnings: vec!["no Done when section".into()],
            diff: "-before\n+after\n".into(),
        });
        tui.sync_overlay();
        assert_eq!(tui.overlay, Some(Overlay::Review(Review::Plan)));

        // The diff and the warning both have to be in what gets drawn, or the review is a
        // yes/no question with the evidence missing.
        let (title, body) = tui.review_text(Review::Plan);
        assert!(title.contains("plan"));
        let rendered: String = body
            .lines
            .iter()
            .flat_map(|l| l.spans.iter().map(|s| s.content.to_string()))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(rendered.contains("no Done when section"), "{rendered}");
        assert!(rendered.contains("+after"), "{rendered}");

        // Rejecting drops the proposal and closes the overlay.
        tui.on_key(press(KeyCode::Char('r')));
        assert!(tui.app.plan.is_none());
        assert_eq!(tui.overlay, None);

        // Every kind of proposal goes through the same overlay, so a new one that forgot to
        // register here would be produced by a worker and then never shown.
        tui.app.rca = Some(crate::app::RcaProposal {
            after: "**Summary**\nIt broke.".into(),
            warnings: vec!["times dropped from the timeline: 09:20".into()],
            diff: "-notes\n+postmortem\n".into(),
        });
        tui.sync_overlay();
        assert_eq!(tui.overlay, Some(Overlay::Review(Review::Rca)));
        let (title, body) = tui.review_text(Review::Rca);
        assert!(title.contains("postmortem"), "{title}");
        let rendered: String = body
            .lines
            .iter()
            .flat_map(|l| l.spans.iter().map(|s| s.content.to_string()))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(rendered.contains("09:20"), "{rendered}");
        tui.on_key(press(KeyCode::Char('r')));
        assert!(tui.app.rca.is_none());
        assert_eq!(tui.overlay, None);
    }

    /// The right-hand pane cycles back to where it started, so a user pressing one key
    /// repeatedly cannot get stuck in it.
    #[test]
    fn the_side_pane_cycles() {
        let mut seen = vec![Side::Hidden];
        let mut side = Side::Hidden;
        for _ in 0..4 {
            side = side.next();
            seen.push(side);
        }
        assert_eq!(side, Side::Hidden, "four steps should return to hidden");
        assert_eq!(
            seen.len(),
            5,
            "every pane should appear once before repeating"
        );
        for s in [Side::Ranking, Side::History, Side::Hint] {
            assert!(!s.title().is_empty(), "{s:?} has no title");
        }
    }

    /// A centred overlay has to stay inside its parent even when the terminal is tiny — a
    /// rectangle wider than the screen is a panic in ratatui, not a cosmetic problem.
    #[test]
    fn overlays_stay_inside_a_small_terminal() {
        for (w, h) in [(80, 24), (20, 5), (1, 1), (200, 60)] {
            let parent = Rect {
                x: 0,
                y: 0,
                width: w,
                height: h,
            };
            let inner = centred(parent, 84, 80);
            assert!(inner.right() <= parent.right(), "{w}x{h}: {inner:?}");
            assert!(inner.bottom() <= parent.bottom(), "{w}x{h}: {inner:?}");
        }
    }

    /// Diff colouring reads the marker column, which is the only thing distinguishing an added
    /// line from a removed one in a unified diff.
    #[test]
    fn diff_lines_are_coloured_by_their_marker() {
        let colour = |s: &str| diff_line(s).spans[0].style.fg;
        assert_eq!(colour("+added"), Some(Color::Green));
        assert_eq!(colour("-removed"), Some(Color::Red));
        assert_eq!(colour("@@ -1,2 +1,3 @@"), Some(Color::Cyan));
        assert_eq!(colour(" context"), None);
        assert_eq!(colour(""), None, "an empty line must not panic");
    }
}
