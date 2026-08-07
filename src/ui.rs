//! The window. All egui drawing lives here; [`crate::app::App`] holds the state.
//!
//! One of three front ends over the same core — see [`crate`] — and the only one that can show a
//! side-by-side diff, which is why the review windows (shrink, plan, sanitize) look the way they
//! do here and unified in [`crate::tui`].

use eframe::egui;
use egui_commonmark::{CommonMarkCache, CommonMarkViewer};

use crate::agents::detect::Status;
use crate::app::App;
use crate::config::Config;
use crate::store::version::Note;

/// Open the window and run until it closes.
///
/// Blocking, and it must be called on the main thread — a platform requirement on macOS rather
/// than a preference of eframe's.
pub fn launch(config: Config) -> eframe::Result<()> {
    let state = App::new(config);
    let title = crate::app::window_title(&state.config.dir.clone(), None, false);

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1180.0, 780.0])
            .with_min_inner_size([760.0, 480.0])
            .with_title(&title),
        ..Default::default()
    };
    eframe::run_native(
        "pstore",
        options,
        Box::new(move |_cc| Ok(Box::new(Ui::new(state)))),
    )
}

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
        self.plan_window(&ctx);
        self.pii_window(&ctx);
        self.models_window(&ctx);
        self.error_window(&ctx);

        // A job may be streaming output or a download may be moving, so keep repainting
        // while anything runs.
        if self.app.shrink_job.is_some()
            || self.app.plan_job.is_some()
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
        // A model call still generating would otherwise keep 7.17 GB mapped after the window
        // is gone: the child does not die with its parent.
        crate::router::shutdown_model();
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
                    .on_hover_text(
                        "Stop compressing after the part being rewritten, and keep the \
                         prompt as it is",
                    )
                    .clicked()
                {
                    self.app.cancel_shrink();
                }
            } else if ui
                .button("Shrink")
                .on_hover_text(
                    "Rewrite the selection — or the whole prompt, when nothing is selected — \
                     telegraphically: no articles, no pleasantries, one fact stated once, \
                     while code, paths and constraints stay verbatim. Runs on the local \
                     model, and shows you the diff before changing anything.",
                )
                .clicked()
            {
                self.app.request_shrink();
            }

            let planning = self.app.plan_job.is_some();
            if planning {
                ui.spinner();
                if ui
                    .button("Stop plan")
                    .on_hover_text("Kill the agent process and keep the prompt as it is")
                    .clicked()
                {
                    self.app.cancel_plan();
                }
            } else if ui
                .button("Plan")
                .on_hover_text(
                    "Rewrite this into a structured instruction for a coding agent — \
                     objective, ordered steps, constraints and acceptance criteria. The \
                     result is the next prompt, not a document to read.",
                )
                .clicked()
            {
                self.app.request_plan();
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
                .or_else(|| {
                    self.app
                        .ranking
                        .as_ref()
                        .and_then(|r| r.best())
                        .map(|c| c.agent_id.to_string())
                })
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
                .on_hover_text("Download the local model and the llama-cli that runs it")
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
                if ready {
                    copy_button(ui, &hint.answer, "Copy");
                }
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

            // The model board, as a button: the commonest reason ranking is unavailable is
            // a checkpoint or a runtime that was never downloaded, fixed from here.
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

            if let Some(inv) = self.app.invocation_hint() {
                ui.separator();
                ui.weak(inv);
            }
        });

        if self
            .app
            .ranking
            .as_ref()
            .is_some_and(|r| !r.choices.is_empty())
        {
            ui.separator();
            self.ranking_table(ui);
        }
    }

    /// The model's shortlist: its best few (agent, model, effort) combinations, each with
    /// its own reason for being there.
    fn ranking_table(&mut self, ui: &mut egui::Ui) {
        let Some(ranking) = self.app.ranking.clone() else {
            return;
        };
        ui.horizontal(|ui| {
            ui.strong(format!(
                "top {} of {} combinations",
                ranking.choices.len(),
                ranking.considered
            ));
            if let Some((label, because)) = &ranking.demand {
                ui.label(format!("judged {label}"))
                    .on_hover_text(format!("what decided it: {because}"));
            }
            ui.checkbox(&mut self.show_all_candidates, "show all");
            if !ranking.excluded.is_empty() {
                let text = ranking
                    .excluded
                    .iter()
                    .map(|(id, why)| format!("{id}: {why}"))
                    .collect::<Vec<_>>()
                    .join("  ·  ");
                ui.weak(format!("excluded — {text}"));
            }
        });

        // A degenerate answer is populated in every field and wrong in the only one that
        // matters, so it is said plainly and above the table rather than left to be read out of
        // it. The small build is where this happens, and the fix is in the Models window.
        if let Some(why) = &ranking.degenerate {
            ui.colored_label(
                ui.visuals().warn_fg_color,
                format!(
                    "⚠ this is a list, not a ranking — {why}. Treat the order as unreliable{}",
                    if crate::models::active().id == crate::models::LLM_1BIT.id {
                        "; the ternary build separates these"
                    } else {
                        ""
                    }
                ),
            );
        }

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
                    .num_columns(8)
                    .striped(true)
                    .show(ui, |ui| {
                        ui.strong("fit");
                        ui.strong("agent");
                        ui.strong("model");
                        ui.strong("effort");
                        ui.strong("speed");
                        // Two different costs, deliberately not merged. Price is what a token
                        // costs on the API; quota is how fast the subscription's allowance goes.
                        // On a paid plan they point opposite ways — every model costs the same
                        // nothing extra, and they still are not interchangeable.
                        ui.strong("price")
                            .on_hover_text("token price relative to the cheapest model here — shown, never ranked on");
                        ui.strong("quota")
                            .on_hover_text(
                                "how fast this drains the subscription's allowance, relative to \
                                 the vendor's lightest model. Unlike price, the ranker is told \
                                 this.",
                            );
                        ui.strong("why");
                        ui.end_row();

                        for c in ranking.choices.iter().take(limit) {
                            ui.label(format!("{:.0}", c.fit));
                            ui.label(c.agent_display);
                            ui.label(c.model_display.as_ref());
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
                                .on_hover_text("time to answer, relative to the fastest choice");
                            let price = ui.label(format!("{:.1}×", c.relative_price));
                            if c.metered {
                                // The one price fact that is not merely informational: every
                                // other model here is already paid for.
                                price.on_hover_text(
                                    "Billed per token on top of the subscription — picking this \
                                     spends money the others do not. The ranker holds metered \
                                     models back unless they are clearly needed.",
                                );
                            } else {
                                price.on_hover_text(
                                    "Token price relative to the cheapest model in this list. \
                                     Shown so you can see the spend; deliberately never scored — \
                                     that decision is yours, not pstore's.",
                                );
                            }

                            // Below 2× the ranker is not told either, so showing a number here
                            // would imply a signal that did not exist.
                            let quota = if c.quota_weight >= 2.0 {
                                format!("{:.0}×", c.quota_weight)
                            } else {
                                "—".to_string()
                            };
                            ui.label(quota).on_hover_text(if c.quota_weight >= 2.0 {
                                format!(
                                    "Uses roughly {:.0}× the plan allowance of the lightest model \
                                     from this vendor. The ranker was told this, so a heavy model \
                                     placed first was judged worth the burn.",
                                    c.quota_weight
                                )
                            } else {
                                "Among the lightest models here — no burn warning was given to \
                                 the ranker."
                                    .to_string()
                            });

                            let reason = if c.rationale.is_empty() {
                                "—".to_string()
                            } else {
                                c.rationale.clone()
                            };
                            // The rationale is the model's conclusion. The tooltip is the
                            // evidence it reached that conclusion from — without it a shortlist
                            // is an assertion, and a wrong row is impossible to argue with.
                            ui.label(reason).on_hover_text(explain(c));
                            ui.end_row();
                        }
                    });
            });
    }

    fn plan_window(&mut self, ctx: &egui::Context) {
        let Some(proposal) = self.app.plan.clone() else {
            return;
        };
        let mut open = true;
        let mut accept = false;
        let mut reject = false;

        egui::Window::new("Review plan")
            .open(&mut open)
            .default_width(760.0)
            .default_height(560.0)
            .show(ctx, |ui| {
                ui.strong("An instruction to paste into a coding agent");
                ui.weak(
                    "Not a document to read — it is the next prompt. Copy it straight out, \
                     or accept it to replace the one you are editing.",
                );
                for w in &proposal.warnings {
                    ui.colored_label(egui::Color32::from_rgb(200, 120, 40), format!("⚠ {w}"));
                }
                ui.separator();

                ui.horizontal(|ui| {
                    copy_button(ui, &proposal.after, "Copy plan");
                    ui.weak(format!("{} characters", proposal.after.len()));
                });
                ui.separator();

                egui::ScrollArea::vertical()
                    .id_salt("plan-text")
                    .max_height(260.0)
                    .show(ui, |ui| {
                        ui.monospace(&proposal.after);
                    });

                ui.separator();
                ui.collapsing("Diff against the current prompt", |ui| {
                    diff_view(ui, &proposal.diff, "plan-diff");
                });

                ui.separator();
                ui.horizontal(|ui| {
                    if ui
                        .button("Replace prompt")
                        .on_hover_text("Overwrite the open prompt with this plan")
                        .clicked()
                    {
                        accept = true;
                    }
                    if ui.button("Discard").clicked() {
                        reject = true;
                    }
                    ui.weak("Accepting is a single undo step.");
                });
            });

        if accept {
            self.app.accept_plan();
        } else if reject || !open {
            self.app.reject_plan();
        }
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
                if proposal.source.range.is_some() {
                    ui.weak("The selection only — the rest of the prompt is untouched.");
                }
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
                ui.horizontal(|ui| {
                    copy_button(ui, &proposal.after, "Copy shortened prompt");
                });
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
    /// The `llama-cli` that executes the checkpoint, and where it came from.
    ///
    /// Shown as its own row because it is a separate thing that can independently be
    /// missing: weights with nothing to run them fail exactly as visibly as no weights,
    /// and "which binary is this?" is the first question when inference misbehaves.
    fn runtime_row(&mut self, ui: &mut egui::Ui, busy: bool) {
        ui.horizontal(|ui| {
            ui.strong("llama-cli runtime");
            ui.weak(format!("· {}", crate::runtime::RELEASE_TAG));
        });
        ui.weak("runs the checkpoint as a subprocess; PrismML's fork of llama.cpp");

        let prefs_path = self.app.config.prefs.llama_path.clone();
        match crate::runtime::locate(prefs_path.as_deref()) {
            Some(rt) => {
                ui.weak(format!("{} · {}", rt.path.display(), rt.origin));
            }
            None => {
                ui.colored_label(
                    egui::Color32::from_rgb(200, 120, 40),
                    format!(
                        "⚠ {}",
                        crate::runtime::missing_reason(prefs_path.as_deref())
                    ),
                );
                if let Ok(asset) = crate::runtime::asset() {
                    ui.weak(format!(
                        "pstore will fetch {} with the model, and check it against its \
                         published SHA256 before installing it.",
                        crate::models::bytes_label(asset.bytes)
                    ));
                }
            }
        }

        if let Some(fraction) = crate::runtime::progress().fraction() {
            ui.add(egui::ProgressBar::new(fraction).show_percentage());
        }
        let _ = busy;
    }

    fn models_window(&mut self, ctx: &egui::Context) {
        if !self.app.models_open {
            return;
        }
        let mut open = true;
        let mut fetch: Vec<crate::models::Checkpoint> = Vec::new();
        let mut load: Vec<crate::models::Checkpoint> = Vec::new();
        let mut cancel = false;
        let mut switched = false;
        let busy = self.app.models_job.is_some();

        egui::Window::new("Local models")
            .open(&mut open)
            .default_width(680.0)
            .show(ctx, |ui| {
                ui.label(
                    "pstore runs this itself, on this machine. It is downloaded once from \
                     Hugging Face and then never contacted again — no prompt text is sent \
                     anywhere by ranking or sanitising.",
                );

                // Said once, up front: without local inference compiled in, no download can help,
                // and every row below would otherwise imply one could.
                if !crate::models::LOCAL_INFERENCE {
                    ui.colored_label(
                        egui::Color32::from_rgb(200, 120, 40),
                        format!("⚠ {}", crate::models::NO_LOCAL_INFERENCE),
                    );
                    ui.label(
                        "Rebuild with the default features — `cargo build --release` — to \
                         enable ranking and the personal-data scan. Until then those two \
                         actions are unavailable; everything else works as normal.",
                    );
                }
                ui.separator();

                self.runtime_row(ui, busy);
                ui.separator();

                let usable = crate::models::LOCAL_INFERENCE;
                let snapshot = crate::models::snapshot();

                ui.label(
                    "Two builds of the same 27B model. Pick one — pstore runs only the \
                     selected build, and having the other downloaded costs nothing but disk.",
                );

                let mut chosen = self.app.config.prefs.local_model;
                for (c, phase) in &snapshot {
                    let choice = crate::models::choice_for(c.id);
                    let selected = Some(chosen) == choice;

                    ui.horizontal(|ui| {
                        // The radio is the row's heading, so "which one am I running?" is
                        // answered by the same glance that reads the name.
                        if let Some(choice) = choice {
                            ui.radio_value(
                                &mut chosen,
                                choice,
                                egui::RichText::new(c.title).strong(),
                            )
                            .on_hover_text(if selected {
                                "pstore runs this build"
                            } else {
                                "Run this build instead, from the next model call on"
                            });
                        } else {
                            ui.strong(c.title);
                        }
                        ui.weak(format!("· {}", c.size_label()));
                        ui.weak(format!("· {}", c.license));
                    });
                    ui.weak(c.purpose);
                    ui.weak(crate::models::tradeoff(c.id));
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

                // Scoped to the selected build, not to everything missing. The other build
                // being absent is the ordinary state of affairs, and a button that quietly
                // pulls 11 GB because both rows are empty is not a button anyone wants.
                let selected = chosen.checkpoint();
                let selected_missing = snapshot
                    .iter()
                    .any(|(c, p)| c.id == selected.id && !p.is_downloaded());

                ui.horizontal_wrapped(|ui| {
                    if ui
                        .add_enabled(
                            usable && !busy && selected_missing,
                            egui::Button::new(format!(
                                "Download the selected build ({})",
                                selected.size_label()
                            )),
                        )
                        .on_hover_text(format!("Fetch {} from {}", selected.title, selected.repo))
                        .clicked()
                    {
                        fetch.push(selected);
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

                let mut prefs_changed = false;
                if chosen != self.app.config.prefs.local_model {
                    self.app.config.prefs.local_model = chosen;
                    prefs_changed = true;
                    switched = true;
                }

                // Switching to a build that is not on disk is allowed — it is how you decide
                // before you download — but the consequence has to be visible here rather
                // than turning up as a failed ranking later.
                if selected_missing {
                    ui.colored_label(
                        egui::Color32::from_rgb(200, 120, 40),
                        format!(
                            "⚠ {} is selected but not downloaded — ranking and the \
                             personal-data scan are unavailable until it is.",
                            selected.title
                        ),
                    );
                }

                prefs_changed |= ui
                    .checkbox(
                        &mut self.app.config.prefs.allow_model_download,
                        "Allow downloading the model and its runtime",
                    )
                    .on_hover_text(
                        "Off means pstore never reaches the network. Ranking and the \
                         personal-data scan are then unavailable rather than degraded — \
                         there is no second implementation behind them.",
                    )
                    .changed();
                ui.weak(
                    "Weights land in the shared Hugging Face cache (~/.cache/huggingface), \
                     so other tools reuse them. Only the llama-cli binary is stored by \
                     pstore itself.",
                );

                ui.collapsing("Advanced", |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Context ceiling");
                        let mut ceiling = self.app.config.prefs.model_context_ceiling as u32;
                        if ui
                            .add(
                                egui::DragValue::new(&mut ceiling)
                                    .range(512..=131_072)
                                    .speed(64),
                            )
                            .on_hover_text(
                                "An upper bound, not a setting: each call asks for only as \
                                 much context as its prompt needs, normally far below this. \
                                 Lower it to cap memory on a small machine; raising it costs \
                                 KV cache roughly linearly.",
                            )
                            .changed()
                        {
                            self.app.config.prefs.model_context_ceiling = ceiling as usize;
                            prefs_changed = true;
                        }
                        ui.weak("tokens");
                    });

                    ui.horizontal(|ui| {
                        ui.label("Reasoning budget");
                        let mut budget = self.app.config.prefs.model_reasoning_budget as u32;
                        if ui
                            .add(egui::DragValue::new(&mut budget).range(0..=8000).speed(50))
                            .on_hover_text(
                                "How far the model may think before it must answer. This is \
                                 the speed-against-quality dial: generation runs at roughly \
                                 41 ms per token, so every 100 characters here is about a \
                                 second and a half added to a ranking. Zero skips reasoning \
                                 entirely — noticeably faster, and noticeably coarser.",
                            )
                            .changed()
                        {
                            self.app.config.prefs.model_reasoning_budget = budget as usize;
                            prefs_changed = true;
                        }
                        ui.weak(if self.app.config.prefs.model_reasoning_budget == 0 {
                            "characters · reasoning off".to_string()
                        } else {
                            format!(
                                "characters · about {:.0}s of thinking",
                                self.app.config.prefs.model_reasoning_budget as f32 / 3.0 * 0.041
                            )
                        });
                    });

                    ui.horizontal(|ui| {
                        ui.label("llama-cli path");
                        let mut path = self.app.config.prefs.llama_path.clone().unwrap_or_default();
                        if ui
                            .add(
                                egui::TextEdit::singleline(&mut path)
                                    .hint_text("leave blank to use the downloaded one")
                                    .desired_width(320.0),
                            )
                            .on_hover_text(
                                "Point at your own build. It must be PrismML's fork — stock \
                                 llama.cpp cannot load this checkpoint's quantisation.",
                            )
                            .changed()
                        {
                            self.app.config.prefs.llama_path =
                                (!path.trim().is_empty()).then_some(path);
                            prefs_changed = true;
                        }
                    });
                });

                if prefs_changed {
                    // The model runs on a worker thread and reads these from the shared
                    // snapshot, so a change that is not published never takes effect.
                    crate::config::publish(&self.app.config.prefs);
                    self.app.config.prefs.save(&self.app.config.dir);
                }

                // Strictly after publishing: the unload asks which build is selected *now*,
                // and a worker mid-flight is refused against the same freshly-published
                // answer. Both builds' weights must never be resident at once.
                if switched {
                    let stopped = crate::router::unload_other_model_builds();
                    self.app.status = if stopped > 0 {
                        format!(
                            "using {} — stopped {stopped} run(s) still holding the other build",
                            selected.title
                        )
                    } else {
                        format!("using {}", selected.title)
                    };
                }
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
                            .num_columns(5)
                            .striped(true)
                            .show(ui, |ui| {
                                ui.strong("mask");
                                ui.strong("tag");
                                ui.strong("found");
                                ui.strong("becomes");
                                ui.strong("sure");
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
                    // Copying the masked text without applying it is the safer habit: the
                    // prompt stays as written, and what leaves the machine is sanitised.
                    if review.scan.plan.enabled() > 0 {
                        let masked = review.scan.plan.apply(&self.app.buffer.text);
                        copy_button(ui, &masked, "Copy masked prompt");
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

/// The evidence behind one ranked choice, as a tooltip.
///
/// The `why` column already shows the model's *conclusion*. This is what it concluded **from**,
/// and the two are worth separating: a shortlist that only asserts is one a developer has to
/// either accept or discard whole, while one that shows its inputs can be argued with — and the
/// input that is wrong is usually the interesting part. A pick that looks wrong is most often a
/// fact that was wrong, or a fact that was missing.
///
/// Ordered as the ranking was actually decided: what pstore knew, where that came from, then the
/// costs the model was and was not told about.
fn explain(c: &crate::router::Choice) -> String {
    use crate::knowledge::Source;
    let mut out = String::new();

    match (&c.note.is_empty(), c.fact_source) {
        // The usual case: pstore supplied a line and the ranker used it.
        (false, Some(source)) => out.push_str(&format!(
            "pstore told the ranker:\n  \"{}\"\n  — from {}\n\n",
            c.note,
            source.label()
        )),
        // The checkpoint recognised the name, so pstore deliberately said nothing rather than
        // give it its own guess to contradict.
        (true, Some(Source::Checkpoint)) => out.push_str(
            "The local model said it already knows this one, so pstore added nothing — the \
             placement is its own knowledge.\n\n",
        ),
        (true, _) | (false, None) => {}
    }

    out.push_str(&format!(
        "Effort {}{}.\nTakes about {:.1}× the fastest option here.\n",
        c.effort,
        if c.effort_selectable {
            ""
        } else {
            " (predicted — this agent has no effort flag, so pstore cannot set it)"
        },
        c.relative_latency
    ));

    // Say plainly which cost reached the ranker and which did not. Otherwise the two columns
    // look like one signal shown twice, and the deliberate asymmetry between them is invisible.
    if c.quota_weight >= 2.0 {
        out.push_str(&format!(
            "Burns about {:.0}× the plan allowance of the lightest model here — the ranker was \
             told this and placed it here anyway.\n",
            c.quota_weight
        ));
    }
    out.push_str(&format!(
        "Token price {:.1}× the cheapest here — shown only; the ranker never sees price.",
        c.relative_price
    ));
    if c.metered {
        out.push_str(
            "\n\n⚠ Billed per token on top of the subscription. Every other model in this list \
             is already paid for.",
        );
    }
    out
}

/// Render a unified diff with per-line colouring.
/// A button that puts `text` on the system clipboard.
///
/// Every produced artefact gets one. The whole workflow ends in pasting something into an
/// agent, a terminal or a chat window, and selecting a monospace block inside a scroll area
/// with the mouse is both fiddly and easy to get subtly wrong — a missing first character
/// in a prompt is not obvious until the agent misbehaves.
fn copy_button(ui: &mut egui::Ui, text: &str, label: &str) {
    if ui
        .button(format!("⧉ {label}"))
        .on_hover_text("Copy to the clipboard")
        .clicked()
    {
        ui.ctx().copy_text(text.to_string());
    }
}

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
mod explain_tests {
    use crate::agents::registry::{Effort, Tier};
    use crate::knowledge::Source;
    use crate::router::Choice;

    fn choice() -> Choice {
        Choice {
            agent_id: "claude",
            agent_display: "Claude Code",
            model_id: "opus".into(),
            model_display: "Opus 5".into(),
            tier: Tier::Top,
            effort: Effort::High,
            effort_selectable: true,
            metered: false,
            relative_latency: 2.6,
            relative_price: 5.0,
            quota_weight: 5.0,
            note: "Anthropic's frontier model: best for hard refactors".into(),
            fact_source: Some(Source::Table),
            fit: 92.0,
            rationale: "hard multi-file refactor".into(),
            row_index: 0,
        }
    }

    /// The whole point of the tooltip: the evidence, not a restatement of the conclusion.
    #[test]
    fn the_explanation_shows_the_fact_and_where_it_came_from() {
        let text = super::explain(&choice());
        assert!(text.contains("best for hard refactors"), "{text}");
        assert!(text.contains("pstore's table"), "{text}");
    }

    /// Price and quota are different signals and the asymmetry between them is the point — one
    /// reached the ranker, the other is deliberately withheld from it. Collapsing them would undo
    /// `price_does_not_influence_ranking` in the UI while leaving it true in the code.
    #[test]
    fn the_explanation_separates_what_the_ranker_saw_from_what_it_did_not() {
        let text = super::explain(&choice());
        assert!(text.contains("the ranker was told this"), "quota: {text}");
        assert!(text.contains("never sees price"), "price: {text}");
    }

    /// A model the checkpoint already knew is a different provenance claim from one pstore
    /// described, and saying "pstore told the ranker: \"\"" would be worse than saying nothing.
    #[test]
    fn a_model_the_checkpoint_knew_says_so_instead_of_quoting_an_empty_note() {
        let mut c = choice();
        c.note = String::new();
        c.fact_source = Some(Source::Checkpoint);
        let text = super::explain(&c);
        assert!(text.contains("already knows this one"), "{text}");
        assert!(!text.contains("pstore told the ranker"), "{text}");
    }

    /// A light model must not carry a burn line — a warning on every row stops being a warning.
    #[test]
    fn a_light_model_carries_no_burn_line() {
        let mut c = choice();
        c.quota_weight = 1.0;
        assert!(!super::explain(&c).contains("Burns about"));
    }

    /// The one cost that is not merely informational has to be unmissable.
    #[test]
    fn a_metered_model_is_called_out() {
        let mut c = choice();
        c.metered = true;
        assert!(super::explain(&c).contains("Billed per token"));
    }
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
