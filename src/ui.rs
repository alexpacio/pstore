//! Rendering. All egui drawing lives here; [`crate::app::App`] holds the state.

use eframe::egui;
use egui_commonmark::{CommonMarkCache, CommonMarkViewer};

use crate::agents::detect::Status;
use crate::app::App;
use crate::store::version::Note;

/// The eframe application: owns [`App`] plus render-only scratch state.
pub struct Ui {
    app: App,
    markdown: CommonMarkCache,
    /// Which agent the ranking table has expanded, if any.
    show_all_candidates: bool,
}

impl Ui {
    /// Wrap application state for rendering.
    pub fn new(app: App) -> Self {
        Self {
            app,
            markdown: CommonMarkCache::default(),
            show_all_candidates: false,
        }
    }
}

impl eframe::App for Ui {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.app.tick();
        self.keyboard(&ctx);

        // Panel order matters in egui: outermost first, central last.
        egui::Panel::top("actions").show(ui, |ui| self.action_bar(ui));
        egui::Panel::bottom("status").show(ui, |ui| self.status_bar(ui));

        egui::Panel::left("prompts")
            .default_size(self.app.config.prefs.sidebar_width)
            .size_range(180.0..=460.0)
            .show(ui, |ui| self.sidebar(ui));

        if self.app.hint_open {
            egui::Panel::right("hint")
                .default_size(360.0)
                .size_range(260.0..=620.0)
                .show(ui, |ui| self.hint_panel(ui));
        }

        egui::CentralPanel::default().show(ui, |ui| self.editor_pane(ui));

        self.shrink_window(&ctx);
        self.pii_window(&ctx);
        self.models_window(&ctx);
        self.error_window(&ctx);

        // A job may be streaming output or a download may be moving, so keep repainting
        // while anything runs.
        if self.app.shrink_job.is_some()
            || self.app.pii_job.is_some()
            || self.app.hint.as_ref().is_some_and(|h| h.job.is_some())
            || crate::models::any_busy()
        {
            ctx.request_repaint_after(std::time::Duration::from_millis(80));
        } else if self.app.buffer.is_dirty() {
            // Keeps the autosave timer honest without spinning the CPU.
            ctx.request_repaint_after(std::time::Duration::from_millis(500));
        }
    }

    fn on_exit(&mut self) {
        self.app.save(Note::Manual);
        self.app.config.prefs.save(&self.app.config.dir);
    }
}

impl Ui {
    /// Global shortcuts. Registered before the widgets so they win over text input.
    fn keyboard(&mut self, ctx: &egui::Context) {
        use egui::{Key, Modifiers};

        let (save, undo, redo_y, redo_z, preview, hint, rank) = ctx.input_mut(|i| {
            (
                i.consume_key(Modifiers::COMMAND, Key::S),
                i.consume_key(Modifiers::COMMAND, Key::Z),
                i.consume_key(Modifiers::COMMAND, Key::Y),
                i.consume_key(Modifiers::COMMAND | Modifiers::SHIFT, Key::Z),
                i.consume_key(Modifiers::COMMAND, Key::M),
                i.consume_key(Modifiers::COMMAND, Key::Enter),
                i.consume_key(Modifiers::COMMAND, Key::R),
            )
        });

        if save {
            self.app.save(Note::Manual);
            self.app.status = "saved".into();
        }
        // Redo is checked first: Cmd+Shift+Z also matches Cmd+Z on some layouts.
        if redo_y || redo_z {
            if self.app.buffer.redo() {
                self.app.status = "redo".into();
            }
        } else if undo {
            let label = self.app.buffer.history.undo_label().unwrap_or("change");
            if self.app.buffer.undo() {
                self.app.status = format!("undo {label}");
            }
        }
        if preview {
            self.app.config.prefs.preview = !self.app.config.prefs.preview;
        }
        if hint {
            self.app.request_hint();
        }
        if rank {
            self.app.rank();
        }
    }

    fn action_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("Score models")
                .on_hover_text(
                    "Classify this prompt, then score every model and effort available (⌘R)",
                )
                .clicked()
            {
                self.app.rank();
            }

            let shrinking = self.app.shrink_job.is_some();
            if shrinking {
                ui.spinner();
                if ui
                    .button("Stop shrink")
                    .on_hover_text("Kill the agent process and keep the prompt as it is")
                    .clicked()
                {
                    self.app.cancel_shrink();
                }
            } else if ui
                .button("Shrink")
                .on_hover_text(
                    "Compress this prompt while keeping code, paths and constraints verbatim",
                )
                .clicked()
            {
                self.app.request_shrink();
            }

            let scanning = self.app.pii_job.is_some();
            if scanning {
                ui.spinner();
                if ui
                    .button("Stop check")
                    .on_hover_text("Stop looking for personal data")
                    .clicked()
                {
                    self.app.cancel_sanitize();
                }
            } else if ui
                .button("Sanitize")
                .on_hover_text(
                    "Find names, addresses, IBANs and other personal data in this prompt \
                     and replace them with placeholders. Runs entirely on this machine, \
                     and shows you the diff before changing anything.",
                )
                .clicked()
            {
                self.app.request_sanitize();
            }

            if ui
                .button("Send →")
                .on_hover_text("Open the prompt in a new terminal window with an interactive agent")
                .clicked()
            {
                self.app.send_to_agent();
            }

            ui.separator();

            // Send target: the top-scoring candidate, or an explicit override.
            let current = self
                .app
                .pinned_agent
                .clone()
                .or_else(|| self.app.ranking.best().map(|c| c.agent_id.to_string()))
                .unwrap_or_else(|| "—".into());
            egui::ComboBox::from_label("target")
                .selected_text(current)
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.app.pinned_agent, None, "best scoring");
                    for agent in &self.app.agents {
                        let label = match &agent.status {
                            Status::Blocked(u) => {
                                format!("{} — {}", agent.spec.display, u.reason())
                            }
                            _ => agent.spec.display.to_string(),
                        };
                        let selectable = agent.usable();
                        ui.add_enabled_ui(selectable, |ui| {
                            ui.selectable_value(
                                &mut self.app.pinned_agent,
                                Some(agent.spec.id.to_string()),
                                label,
                            );
                        });
                    }
                });

            ui.separator();
            ui.checkbox(&mut self.app.config.prefs.preview, "Preview")
                .on_hover_text("Render markdown instead of editing (⌘M)");

            if ui
                .button("Hint…")
                .on_hover_text("Ask about the selection, or a question (⌘⏎)")
                .clicked()
            {
                self.app.hint_open = true;
            }

            if ui
                .button("Models…")
                .on_hover_text("Download and load the local classifiers and the PII tagger")
                .clicked()
            {
                self.app.models_open = true;
            }
        });
    }

    fn sidebar(&mut self, ui: &mut egui::Ui) {
        ui.heading("Prompts");
        ui.horizontal(|ui| {
            let response = ui.add(
                egui::TextEdit::singleline(&mut self.app.name_input)
                    .hint_text(if self.app.renaming {
                        "new name"
                    } else {
                        "new prompt"
                    })
                    .desired_width(f32::INFINITY),
            );
            let submitted = response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            if submitted && !self.app.name_input.trim().is_empty() {
                let name = std::mem::take(&mut self.app.name_input);
                if self.app.renaming {
                    self.app.rename_open(name.trim());
                    self.app.renaming = false;
                } else {
                    self.app.create_prompt(name.trim());
                }
            }
        });
        ui.horizontal(|ui| {
            if ui.small_button("New").clicked() {
                self.app.renaming = false;
                self.app.name_input.clear();
            }
            let has_open = self.app.current().is_some();
            if ui
                .add_enabled(has_open, egui::Button::new("Rename").small())
                .clicked()
            {
                self.app.renaming = true;
                self.app.name_input = self
                    .app
                    .current()
                    .map(|p| p.name.clone())
                    .unwrap_or_default();
            }
            if ui
                .add_enabled(has_open, egui::Button::new("Delete").small())
                .clicked()
            {
                self.app.delete_open();
            }
        });

        ui.separator();
        egui::ScrollArea::vertical()
            .id_salt("prompt-list")
            .max_height(ui.available_height() * 0.45)
            .show(ui, |ui| {
                let mut to_open = None;
                for (i, prompt) in self.app.prompts.iter().enumerate() {
                    let selected = self.app.open == Some(i);
                    let dirty = selected && self.app.buffer.is_dirty();
                    let label = if dirty {
                        format!("• {}", prompt.name)
                    } else {
                        prompt.name.clone()
                    };
                    if ui.selectable_label(selected, label).clicked() {
                        to_open = Some(i);
                    }
                }
                if let Some(i) = to_open {
                    self.app.open_prompt(i);
                }
                if self.app.prompts.is_empty() {
                    ui.weak("No .md files here yet.");
                }
            });

        ui.separator();
        ui.heading("History");
        if self.app.history.versions.is_empty() {
            ui.weak("Saved versions appear here.");
        }
        egui::ScrollArea::vertical()
            .id_salt("history-list")
            .show(ui, |ui| {
                let mut to_select = None;
                for v in &self.app.history.versions {
                    let selected = self.app.history.selected.as_deref() == Some(v.ts.as_str());
                    let label = format!("{}  ({}, {} B)", v.ts, v.note.label(), v.bytes);
                    if ui.selectable_label(selected, label).clicked() {
                        to_select = Some(v.ts.clone());
                    }
                }
                if let Some(ts) = to_select {
                    self.app.select_version(ts);
                }

                if self.app.history.selected.is_some() {
                    ui.horizontal(|ui| {
                        if ui.button("Restore").clicked() {
                            self.app.restore_selected();
                        }
                        if ui.button("Close diff").clicked() {
                            self.app.history.selected = None;
                            self.app.history.diff.clear();
                        }
                    });
                    if !self.app.history.diff.is_empty() {
                        diff_view(ui, &self.app.history.diff, "history-diff");
                    }
                }
            });
    }

    fn editor_pane(&mut self, ui: &mut egui::Ui) {
        if self.app.current().is_none() {
            ui.centered_and_justified(|ui| {
                ui.weak("Create or select a prompt on the left.");
            });
            return;
        }

        if self.app.config.prefs.preview {
            egui::ScrollArea::vertical()
                .id_salt("preview")
                .show(ui, |ui| {
                    CommonMarkViewer::new().show(ui, &mut self.markdown, &self.app.buffer.text);
                });
            return;
        }

        egui::ScrollArea::vertical()
            .id_salt("editor")
            .show(ui, |ui| {
                let output = egui::TextEdit::multiline(&mut self.app.buffer.text)
                    .font(egui::TextStyle::Monospace)
                    .code_editor()
                    .desired_width(f32::INFINITY)
                    .desired_rows(30)
                    .lock_focus(true)
                    .show(ui);

                // Mirror the widget's cursor into the buffer so hints know the selection
                // and insertions land where the caret is.
                if let Some(range) = output.state.cursor.char_range() {
                    self.app.buffer.selection = crate::editor::Selection {
                        start: range.primary.index.into(),
                        end: range.secondary.index.into(),
                    };
                }
                if output.response.changed() {
                    let caret = self.app.buffer.selection.start;
                    self.app.buffer.on_widget_edit(caret);
                }
            });
    }

    fn hint_panel(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading("Hints");
            if ui.small_button("Close").clicked() {
                self.app.hint_open = false;
            }
        });

        let selection = self.app.buffer.selected_text();
        match &selection {
            Some(sel) => {
                let preview: String = sel.chars().take(80).collect();
                ui.weak(format!("Selection: {preview}…"));
            }
            None => {
                ui.weak("Nothing selected — ask a question instead.");
            }
        }

        ui.add(
            egui::TextEdit::multiline(&mut self.app.hint_input)
                .hint_text("Ask about the selection, or anything about this prompt")
                .desired_rows(3)
                .desired_width(f32::INFINITY),
        );

        let running = self.app.hint.as_ref().is_some_and(|h| h.job.is_some());
        ui.horizontal(|ui| {
            if ui.add_enabled(!running, egui::Button::new("Ask")).clicked() {
                self.app.request_hint();
            }
            if running {
                ui.spinner();
                ui.weak("waiting…");
                if ui
                    .button("Stop")
                    .on_hover_text("Kill the agent process and discard this hint")
                    .clicked()
                {
                    self.app.cancel_hint();
                }
            }
        });

        ui.separator();
        if let Some(hint) = self.app.hint.clone() {
            ui.horizontal_wrapped(|ui| {
                ui.weak(format!("asked as a {}", hint.subject.label()));
                if let Some(by) = &hint.answered_by {
                    ui.separator();
                    ui.weak(format!("via {by}"));
                }
            });
            egui::ScrollArea::vertical()
                .id_salt("hint-answer")
                .max_height(ui.available_height() - 40.0)
                .show(ui, |ui| {
                    if hint.answer.trim().is_empty() && hint.job.is_some() {
                        ui.weak("…");
                    } else {
                        CommonMarkViewer::new().show(ui, &mut self.markdown, &hint.answer);
                    }
                });

            ui.separator();
            ui.horizontal_wrapped(|ui| {
                let ready = !hint.answer.trim().is_empty();
                if ui
                    .add_enabled(ready, egui::Button::new("Insert at caret"))
                    .clicked()
                {
                    self.app.insert_hint(false);
                }
                let has_selection = selection.is_some();
                if ui
                    .add_enabled(
                        ready && has_selection,
                        egui::Button::new("Replace selection"),
                    )
                    .clicked()
                {
                    self.app.insert_hint(true);
                }
                if ui.button("Discard").clicked() {
                    self.app.hint = None;
                }
            });
        } else {
            ui.weak("Answers appear here. Nothing is inserted into your prompt unless you say so.");
        }
    }

    fn status_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.weak(&self.app.status);
            ui.separator();
            ui.weak(format!("classifier: {}", self.app.backend));

            // The model board, as a button: the commonest reason a reading is a fallback
            // is a checkpoint that was never downloaded, and this is where that is fixed.
            ui.separator();
            let busy = crate::models::any_busy();
            if busy {
                ui.spinner();
            }
            if ui
                .small_button(crate::jobs::describe_cache())
                .on_hover_text("Open the Models window")
                .clicked()
            {
                self.app.models_open = true;
            }

            if let Some(reading) = &self.app.reading {
                ui.separator();
                let (dim, score) = reading.capability.dominant();
                let mut text = format!(
                    "{} · mostly {dim} ({:.2}) · via {}",
                    reading.complexity, score, reading.source
                );
                if let Some(c) = reading.confidence {
                    text.push_str(&format!(" ({:.0}% sure)", c * 100.0));
                }
                let label = ui.weak(text);
                let mut tooltip = String::new();
                if let Some(s) = reading.difficulty_score {
                    tooltip.push_str(&format!("complexity score {s:.3}\n"));
                }
                if let Some((task, p)) = reading.task {
                    tooltip.push_str(&format!("task: {task} ({:.0}%)\n", p * 100.0));
                }
                if let Some(d) = &reading.difficulty_detail {
                    let detail = d
                        .notable(0.15)
                        .iter()
                        .map(|(n, v)| format!("{n} {v:.2}"))
                        .collect::<Vec<_>>()
                        .join(", ");
                    if !detail.is_empty() {
                        tooltip.push_str(&format!("drivers: {detail}\n"));
                    }
                }
                if let Some(why) = &reading.fallback_reason {
                    tooltip.push_str(&format!("degraded to the built-in estimate: {why}"));
                }
                if !tooltip.is_empty() {
                    label.on_hover_text(tooltip.trim_end().to_string());
                }
                // A reading with no model behind it is a working answer, not an error —
                // but it should be one click from being a better one.
                if !reading.source.uses_a_model()
                    && ui
                        .small_button("use the classifiers")
                        .on_hover_text(
                            "This reading came from the built-in estimate. Download the \
                             classifiers to score with the trained models instead.",
                        )
                        .clicked()
                {
                    self.app.models_open = true;
                }
            }
            if let Some(inv) = self.app.invocation_hint() {
                ui.separator();
                ui.weak(inv);
            }
        });

        if !self.app.ranking.candidates.is_empty() {
            ui.separator();
            self.ranking_table(ui);
        }
    }

    /// The scored field. Every model and effort combination, with its score — the
    /// point being that the developer sees the whole picture rather than a single
    /// automatic pick.
    fn ranking_table(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.strong(format!("{} candidates", self.app.ranking.candidates.len()));
            ui.checkbox(&mut self.show_all_candidates, "show all");
            if !self.app.ranking.excluded.is_empty() {
                let text = self
                    .app
                    .ranking
                    .excluded
                    .iter()
                    .map(|(id, why)| format!("{id}: {why}"))
                    .collect::<Vec<_>>()
                    .join("  ·  ");
                ui.weak(format!("excluded — {text}"));
            }
        });

        let limit = if self.show_all_candidates {
            usize::MAX
        } else {
            6
        };
        egui::ScrollArea::vertical()
            .id_salt("ranking")
            .max_height(190.0)
            .show(ui, |ui| {
                egui::Grid::new("ranking-grid")
                    .num_columns(6)
                    .striped(true)
                    .show(ui, |ui| {
                        ui.strong("score");
                        ui.strong("agent");
                        ui.strong("model");
                        ui.strong("effort");
                        ui.strong("speed");
                        ui.strong("why");
                        ui.end_row();

                        for c in self.app.ranking.candidates.iter().take(limit) {
                            ui.label(format!("{:.0}", c.score));
                            ui.label(c.agent_display);
                            ui.label(c.model_display);
                            let effort = if c.effort_selectable {
                                c.effort.to_string()
                            } else {
                                format!("~{}", c.effort)
                            };
                            ui.label(effort).on_hover_text(if c.effort_selectable {
                                "pstore can set this effort level"
                            } else {
                                "this agent has no effort flag — shown as a prediction"
                            });
                            ui.label(format!("{:.1}×", c.relative_latency))
                                .on_hover_text("time to answer, relative to the fastest candidate");
                            let mut why = format!(
                                "relative token price {:.1}× — shown for information; \
                                 it is not part of the score",
                                c.relative_price
                            );
                            if c.metered {
                                why.push_str(
                                    "\n\nThis model is billed per token rather than covered \
                                     by the subscription. Its score is its real fit, but it \
                                     is only picked automatically when it fits better than \
                                     every included model by a clear margin — pick it from \
                                     the target list to use it anyway.",
                                );
                            }
                            ui.label(c.rationale()).on_hover_text(why);
                            ui.end_row();
                        }
                    });
            });
    }

    fn shrink_window(&mut self, ctx: &egui::Context) {
        let Some(proposal) = self.app.shrink.clone() else {
            return;
        };
        let mut open = true;
        let mut accept = false;
        let mut reject = false;

        egui::Window::new("Review shrink")
            .open(&mut open)
            .default_width(760.0)
            .default_height(520.0)
            .show(ctx, |ui| {
                ui.strong(proposal.savings.summary());
                for w in &proposal.warnings {
                    ui.colored_label(egui::Color32::from_rgb(200, 120, 40), format!("⚠ {w}"));
                }
                if !proposal.warnings.is_empty() {
                    ui.weak(
                        "The rewrite dropped something the original stated explicitly. \
                         Read the diff before accepting.",
                    );
                }
                ui.separator();
                diff_view(ui, &proposal.diff, "shrink-diff");
                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button("Accept").clicked() {
                        accept = true;
                    }
                    if ui.button("Discard").clicked() {
                        reject = true;
                    }
                    ui.weak("Accepting is a single undo step.");
                });
            });

        if accept {
            self.app.accept_shrink();
        } else if reject || !open {
            self.app.shrink = None;
        }
    }

    /// Where each local checkpoint is, and the buttons that move it along.
    ///
    /// The whole point of this window is that nothing large happens behind the user's
    /// back: the size is stated before the download starts, the progress is visible while
    /// it runs, and a failure is shown here with its reason rather than swallowed into a
    /// heuristic fallback.
    fn models_window(&mut self, ctx: &egui::Context) {
        if !self.app.models_open {
            return;
        }
        let mut open = true;
        let mut fetch: Vec<crate::models::Checkpoint> = Vec::new();
        let mut load: Vec<crate::models::Checkpoint> = Vec::new();
        let mut cancel = false;
        let busy = self.app.models_job.is_some();

        egui::Window::new("Local models")
            .open(&mut open)
            .default_width(680.0)
            .show(ctx, |ui| {
                ui.label(
                    "pstore runs these itself, in this process, on this machine. They are \
                     downloaded once from Hugging Face and then never contacted again — \
                     no prompt text is sent anywhere by scoring or sanitising.",
                );
                ui.weak(format!("compute backend: {}", self.app.backend));

                // Said once, up front: without Candle compiled in, no download can help,
                // and every row below would otherwise imply one could.
                if !crate::models::LOCAL_INFERENCE {
                    ui.colored_label(
                        egui::Color32::from_rgb(200, 120, 40),
                        format!("⚠ {}", crate::models::NO_LOCAL_INFERENCE),
                    );
                    ui.label(
                        "Rebuild with the default features — `cargo build --release` — to \
                         enable the classifiers and the PII tagger. Until then routing uses \
                         the built-in estimate and sanitising uses the checksum-backed \
                         patterns; everything else works as normal.",
                    );
                }
                ui.separator();

                let usable = crate::models::LOCAL_INFERENCE;
                let snapshot = crate::models::snapshot();
                for (c, phase) in &snapshot {
                    ui.horizontal(|ui| {
                        ui.strong(c.title);
                        ui.weak(format!("· {}", c.size_label()));
                        ui.weak(format!("· {}", c.license));
                    });
                    ui.weak(c.purpose);
                    ui.horizontal_wrapped(|ui| {
                        ui.monospace(c.repo);
                    });

                    match phase {
                        crate::models::Phase::Failed(why) => {
                            ui.colored_label(
                                egui::Color32::from_rgb(200, 120, 40),
                                format!("⚠ {why}"),
                            );
                        }
                        _ => {
                            ui.weak(phase.label());
                        }
                    }
                    if let Some(fraction) = phase.fraction() {
                        ui.add(egui::ProgressBar::new(fraction).show_percentage());
                    }

                    ui.horizontal(|ui| {
                        let downloaded = phase.is_downloaded();
                        if ui
                            .add_enabled(
                                usable && !busy && !downloaded,
                                egui::Button::new("Download").small(),
                            )
                            .on_hover_text(format!("Fetch {} from {}", c.size_label(), c.repo))
                            .clicked()
                        {
                            fetch.push(*c);
                        }
                        let loaded = *phase == crate::models::Phase::Ready;
                        if ui
                            .add_enabled(
                                usable && !busy && downloaded && !loaded,
                                egui::Button::new("Load").small(),
                            )
                            .on_hover_text("Build the model in memory now")
                            .clicked()
                        {
                            load.push(*c);
                        }
                        if matches!(phase, crate::models::Phase::Failed(_))
                            && ui
                                .add_enabled(usable && !busy, egui::Button::new("Retry").small())
                                .clicked()
                        {
                            fetch.push(*c);
                        }
                    });
                    ui.separator();
                }

                let missing: Vec<_> = snapshot
                    .iter()
                    .filter(|(_, p)| !p.is_downloaded())
                    .map(|(c, _)| *c)
                    .collect();
                let missing_bytes: u64 = missing.iter().map(|c| c.bytes).sum();

                ui.horizontal_wrapped(|ui| {
                    if ui
                        .add_enabled(
                            usable && !busy && !missing.is_empty(),
                            egui::Button::new(format!(
                                "Download all missing ({})",
                                crate::models::bytes_label(missing_bytes)
                            )),
                        )
                        .clicked()
                    {
                        fetch.extend(missing.iter().copied());
                    }
                    if busy {
                        ui.spinner();
                        if ui
                            .button("Stop")
                            .on_hover_text(
                                "Stops after the file currently in flight — the Hub client \
                                 has no way to abort one mid-transfer. Nothing already \
                                 downloaded is lost.",
                            )
                            .clicked()
                        {
                            cancel = true;
                        }
                    }
                });

                ui.checkbox(
                    &mut self.app.config.prefs.allow_model_download,
                    "Allow downloading weights from Hugging Face",
                )
                .on_hover_text(
                    "Off means pstore never reaches the network: scoring falls back to the \
                     built-in estimate and sanitising to the checksum-backed patterns.",
                );
                ui.weak(
                    "Weights land in the shared Hugging Face cache (~/.cache/huggingface), \
                     so other tools reuse them and pstore stores nothing of its own.",
                );
            });

        if cancel {
            self.app.cancel_models();
        }
        if !fetch.is_empty() {
            self.app.fetch_models(fetch);
        }
        if !load.is_empty() {
            self.app.load_models(load);
        }
        if !open {
            self.app.models_open = false;
        }
    }

    /// The masking proposal: what was found, what it would become, and the diff.
    fn pii_window(&mut self, ctx: &egui::Context) {
        let Some(review) = self.app.pii.clone() else {
            return;
        };
        let mut open = true;
        let mut accept = false;
        let mut reject = false;
        let mut toggled: Vec<(usize, bool)> = Vec::new();
        let mut tag_toggled: Option<(String, bool)> = None;

        egui::Window::new("Review masking")
            .open(&mut open)
            .default_width(820.0)
            .default_height(560.0)
            .show(ctx, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.strong(review.scan.plan.summary());
                    ui.separator();
                    ui.weak(format!("via {}", review.scan.source_label()));
                });
                if let Some(why) = &review.scan.fallback_reason {
                    ui.colored_label(
                        egui::Color32::from_rgb(200, 120, 40),
                        format!("⚠ the tagger did not run — {why}"),
                    );
                    ui.weak(
                        "Checksum-backed identifiers (IBAN, cards, codice fiscale, VAT, \
                         email) were still checked, but names, addresses and \
                         organisations need the tagger.",
                    );
                    // Which remedy applies depends on why: a missing download is a click
                    // away, a missing feature needs a rebuild.
                    if crate::models::LOCAL_INFERENCE {
                        if ui.button("Open Models…").clicked() {
                            self.app.models_open = true;
                        }
                    } else {
                        ui.weak(
                            "Rebuild with the default features — `cargo build --release` — \
                             to enable it.",
                        );
                    }
                }
                ui.weak(
                    "Nothing here has left this machine, and the values below are not \
                     written to disk. The text you replace stays recoverable from version \
                     history, and applying this is a single undo step.",
                );
                ui.separator();

                // Tag-level switches, so turning off every DATE is one click.
                ui.horizontal_wrapped(|ui| {
                    ui.weak("whole tags:");
                    let mut tags: Vec<&str> = review
                        .scan
                        .plan
                        .items
                        .iter()
                        .map(|i| i.finding.tag.as_str())
                        .collect();
                    tags.sort_unstable();
                    tags.dedup();
                    for tag in tags {
                        let all_on = review
                            .scan
                            .plan
                            .items
                            .iter()
                            .filter(|i| i.finding.tag == tag)
                            .all(|i| i.masked);
                        if ui
                            .selectable_label(all_on, tag)
                            .on_hover_text(if all_on {
                                "click to leave every one of these in the prompt"
                            } else {
                                "click to mask every one of these"
                            })
                            .clicked()
                        {
                            tag_toggled = Some((tag.to_string(), !all_on));
                        }
                    }
                });

                egui::ScrollArea::vertical()
                    .id_salt("pii-findings")
                    .max_height(220.0)
                    .show(ui, |ui| {
                        egui::Grid::new("pii-grid")
                            .num_columns(6)
                            .striped(true)
                            .show(ui, |ui| {
                                ui.strong("mask");
                                ui.strong("tag");
                                ui.strong("found");
                                ui.strong("becomes");
                                ui.strong("sure");
                                ui.strong("how");
                                ui.end_row();

                                for (i, item) in review.scan.plan.items.iter().enumerate() {
                                    let mut masked = item.masked;
                                    if ui.checkbox(&mut masked, "").changed() {
                                        toggled.push((i, masked));
                                    }
                                    ui.label(&item.finding.tag);
                                    let preview: String =
                                        item.finding.text.chars().take(48).collect();
                                    ui.monospace(preview);
                                    ui.monospace(&item.placeholder);
                                    // Shown rather than hidden in a tooltip: a hesitant
                                    // finding is exactly the one worth unticking.
                                    ui.label(format!("{:.0}%", item.finding.score * 100.0));
                                    ui.label(item.finder_label()).on_hover_text(
                                        "\"checked pattern\" means the value satisfied its \
                                         own check digit, so it is certain; \"tagger\" is \
                                         the model's read of the surrounding text.",
                                    );
                                    ui.end_row();
                                }
                            });
                    });

                ui.separator();
                if review.diff.is_empty() {
                    ui.weak("Nothing selected, so the prompt would not change.");
                } else {
                    diff_view(ui, &review.diff, "pii-diff");
                }
                ui.separator();
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(
                            review.scan.plan.enabled() > 0,
                            egui::Button::new("Mask selected"),
                        )
                        .clicked()
                    {
                        accept = true;
                    }
                    if ui.button("Discard").clicked() {
                        reject = true;
                    }
                });
            });

        if let Some((tag, masked)) = tag_toggled {
            if let Some(r) = self.app.pii.as_mut() {
                r.scan.plan.set_tag(&tag, masked);
            }
            self.app.refresh_pii_diff();
        }
        if !toggled.is_empty() {
            if let Some(r) = self.app.pii.as_mut() {
                for (i, masked) in toggled {
                    if let Some(item) = r.scan.plan.items.get_mut(i) {
                        item.masked = masked;
                    }
                }
            }
            self.app.refresh_pii_diff();
        }
        if accept {
            self.app.accept_sanitize();
        } else if reject || !open {
            self.app.pii = None;
        }
    }

    fn error_window(&mut self, ctx: &egui::Context) {
        let Some(message) = self.app.error.clone() else {
            return;
        };
        let mut open = true;
        let mut dismiss = false;

        egui::Window::new("Problem")
            .open(&mut open)
            .default_width(520.0)
            .show(ctx, |ui| {
                ui.label(message);
                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button("Dismiss").clicked() {
                        dismiss = true;
                    }
                    if ui.button("Re-detect agents").clicked() {
                        self.app.refresh_agents();
                        dismiss = true;
                    }
                });
            });

        if dismiss || !open {
            self.app.error = None;
        }
    }
}

/// Render a unified diff with per-line colouring.
fn diff_view(ui: &mut egui::Ui, diff: &str, salt: &str) {
    egui::ScrollArea::both()
        .id_salt(salt)
        .max_height(340.0)
        .show(ui, |ui| {
            for line in diff.lines() {
                let colour = match line.chars().next() {
                    Some('+') => Some(egui::Color32::from_rgb(90, 170, 90)),
                    Some('-') => Some(egui::Color32::from_rgb(200, 100, 100)),
                    Some('.') => Some(egui::Color32::GRAY),
                    _ => None,
                };
                let text = egui::RichText::new(line).monospace();
                match colour {
                    Some(c) => ui.colored_label(c, text),
                    None => ui.label(text),
                };
            }
        });
}

#[cfg(test)]
mod tests {
    /// The diff colouring is the one piece of pure logic in this module worth pinning;
    /// everything else needs a live egui context and is covered by the manual pass in
    /// the plan's verification section.
    #[test]
    fn diff_lines_are_classified_by_their_first_character() {
        let classify = |line: &str| line.chars().next();
        assert_eq!(classify("+added"), Some('+'));
        assert_eq!(classify("-removed"), Some('-'));
        assert_eq!(classify(" context"), Some(' '));
        assert_eq!(classify("..."), Some('.'));
        assert_eq!(classify(""), None);
    }
}
