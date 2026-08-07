//! Application state and the per-frame update loop.

use std::path::Path;
use std::time::Duration;

use crate::agents::detect::Detected;
use crate::agents::registry;
use crate::config::{Config, HintSource};
use crate::editor::Buffer;
use crate::hints::Subject;
use crate::jobs::{self, Event, Handle, JobId, Kind, Runner};
use crate::router::{self, Ranking};
use crate::shrink::Savings;
use crate::store::version::{self, Note, VersionMeta};
use crate::store::{Prompt, PromptStore};

/// Autosave once the buffer has been idle this long.
const AUTOSAVE_IDLE: Duration = Duration::from_secs(5);

/// A finished sanitisation pass, awaiting approval.
#[derive(Debug, Clone)]
pub struct PiiReview {
    /// What was found, and with what.
    pub scan: crate::pii::Scan,
    /// Unified diff of the masked text against the buffer.
    pub diff: String,
}

/// An action waiting on a ranking before it can start.
///
/// A hint answered by an agent has to pick which one, and picking now means asking the
/// model — seconds, not microseconds. Rather than refuse until the user has ranked
/// manually, the request is parked here and dispatched when the ranking arrives.
///
/// Shrink and plan are not here: both run on the local model and need no agent at all.
#[derive(Debug, Clone)]
enum Pending {
    /// Ask for a hint about this subject.
    Hint(Subject),
}

/// A pending plan awaiting approval.
#[derive(Debug, Clone)]
pub struct PlanProposal {
    /// The proposed instruction text.
    pub after: String,
    /// Structural problems detected in the plan.
    pub warnings: Vec<String>,
    /// Unified diff for display.
    pub diff: String,
}

/// A pending root cause analysis awaiting approval.
///
/// Separate from [`PlanProposal`] despite the same three fields, because the two are not the
/// same kind of thing to a reader: a plan replaces the prompt with what to do next, and a
/// postmortem replaces incident notes with the write-up of them. The review windows say
/// different things for that reason, and merging the state would invite merging those too.
#[derive(Debug, Clone)]
pub struct RcaProposal {
    /// The proposed postmortem.
    pub after: String,
    /// Problems detected in the analysis — dropped times, dropped paths, one-note actions.
    pub warnings: Vec<String>,
    /// Unified diff for display.
    pub diff: String,
}

/// What a shrink pass was asked to compress.
///
/// Captured when the pass starts rather than read back when it finishes: a shrink is
/// seconds of local inference, and the selection that asked for it is usually gone by then.
#[derive(Debug, Clone)]
pub struct ShrinkSource {
    /// The text handed to the model.
    pub text: String,
    /// The character range it came from, or `None` for the whole document.
    pub range: Option<(usize, usize)>,
}

/// A pending shrink awaiting approval.
#[derive(Debug, Clone)]
pub struct ShrinkProposal {
    /// What was compressed, and where it came from.
    pub source: ShrinkSource,
    /// The proposed replacement text.
    pub after: String,
    /// Size comparison.
    pub savings: Savings,
    /// Structural problems detected in the rewrite.
    pub warnings: Vec<String>,
    /// Unified diff for display.
    pub diff: String,
}

/// A hint request and its streamed answer.
#[derive(Debug, Clone)]
pub struct HintState {
    /// What was asked.
    pub subject: Subject,
    /// Accumulated answer.
    pub answer: String,
    /// Job producing the answer, if still running.
    pub job: Option<JobId>,
    /// Which candidate answered.
    pub answered_by: Option<String>,
}

/// What the version panel is showing.
#[derive(Debug, Clone, Default)]
pub struct HistoryView {
    /// Versions of the open prompt, newest first.
    pub versions: Vec<VersionMeta>,
    /// Selected version, if any.
    pub selected: Option<String>,
    /// Diff between the selected version and the live buffer.
    pub diff: String,
}

/// Everything the UI draws.
pub struct App {
    /// Resolved configuration.
    pub config: Config,
    /// Prompt file access.
    pub store: PromptStore,
    /// Prompts in the folder.
    pub prompts: Vec<Prompt>,
    /// Index into `prompts` of the open one.
    pub open: Option<usize>,
    /// The editable buffer.
    pub buffer: Buffer,
    /// Version history for the open prompt.
    pub history: HistoryView,

    /// Background work.
    pub runner: Runner,
    /// Detected agents.
    pub agents: Vec<Detected>,
    /// Latest ranking of the open prompt, once the model has produced one.
    ///
    /// `None` until it has. There is no cheap estimate to show in the meantime: ranking
    /// costs a model invocation, and inventing a placeholder ranking would be presenting a
    /// guess as an answer.
    pub ranking: Option<Ranking>,
    /// What to do once the ranking this job is producing arrives.
    ///
    /// An agent-answered hint needs a ranking to choose one, and getting it now means
    /// waiting on the model. Rather than make the user press "Score models" first, the
    /// request is remembered and dispatched when the ranking lands.
    pending: Option<Pending>,

    /// Hint panel state.
    pub hint: Option<HintState>,
    /// Contents of the hint question box.
    pub hint_input: String,
    /// Whether the hint panel is open.
    pub hint_open: bool,
    /// Pending shrink awaiting approval.
    pub shrink: Option<ShrinkProposal>,
    /// Job currently producing a shrink.
    pub shrink_job: Option<JobId>,
    /// What that job was handed, captured when it started.
    pub shrink_source: Option<ShrinkSource>,
    /// Pending plan awaiting approval.
    pub plan: Option<PlanProposal>,
    /// Job currently producing a plan.
    pub plan_job: Option<JobId>,
    /// Pending root cause analysis awaiting approval.
    pub rca: Option<RcaProposal>,
    /// Job currently producing a root cause analysis.
    pub rca_job: Option<JobId>,
    /// Pending PII masking awaiting approval.
    pub pii: Option<PiiReview>,
    /// Job currently scanning for personal data.
    pub pii_job: Option<JobId>,
    /// Whether the Models window is open.
    pub models_open: bool,
    /// Job currently downloading or loading weights.
    pub models_job: Option<JobId>,
    /// Handles for running jobs, so the user can stop them.
    running: Vec<Handle>,

    /// Agent chosen manually for Send, overriding the top-scoring one.
    pub pinned_agent: Option<String>,
    /// Transient status line.
    pub status: String,
    /// Last error, shown until dismissed.
    pub error: Option<String>,
    /// Name being typed in the new/rename box.
    pub name_input: String,
    /// Whether the rename box targets the open prompt.
    pub renaming: bool,
}

impl App {
    /// Build the initial state and kick off agent detection.
    pub fn new(config: Config) -> Self {
        let store = PromptStore::new(config.dir.clone());
        let prompts = store.list();
        let runner = Runner::new();
        runner.detect(config.dir.clone());
        // Ask the model cache what is already there, so the Models window and the status
        // bar are honest from the first frame rather than after the first classification.
        runner.probe_models();

        let mut app = Self {
            config,
            store,
            prompts,
            open: None,
            buffer: Buffer::default(),
            history: HistoryView::default(),
            runner,
            agents: Vec::new(),
            ranking: None,
            pending: None,
            hint: None,
            hint_input: String::new(),
            hint_open: false,
            shrink: None,
            shrink_job: None,
            shrink_source: None,
            plan: None,
            plan_job: None,
            rca: None,
            rca_job: None,
            pii: None,
            pii_job: None,
            models_open: false,
            models_job: None,
            running: Vec::new(),
            pinned_agent: None,
            status: "detecting agents…".into(),
            error: None,
            name_input: String::new(),
            renaming: false,
        };
        if !app.prompts.is_empty() {
            app.open_prompt(0);
        }
        // A config layer that failed to parse is shown at startup rather than swallowed.
        // The layer most likely to be malformed is a hand-edited policy file, and the
        // symptom — a model filter that stopped applying — is otherwise invisible.
        if !app.config.warnings.is_empty() {
            app.error = Some(app.config.warnings.join("\n"));
        }
        app
    }

    /// The open prompt, if any.
    pub fn current(&self) -> Option<&Prompt> {
        self.open.and_then(|i| self.prompts.get(i))
    }

    /// Open the prompt at `index`, saving the current one first if dirty.
    pub fn open_prompt(&mut self, index: usize) {
        if self.open == Some(index) {
            return;
        }
        self.save(Note::Manual);
        let Some(prompt) = self.prompts.get(index).cloned() else {
            return;
        };
        match self.store.read(&prompt) {
            Ok(text) => {
                self.buffer.load(text);
                self.open = Some(index);
                self.ranking = None;
                self.pending = None;
                self.shrink = None;
                self.plan = None;
                self.rca = None;
                self.refresh_history();
                self.status = format!("opened {}", prompt.name);
            }
            Err(e) => self.error = Some(format!("could not read {}: {e}", prompt.name)),
        }
    }

    /// Reload the prompt list, keeping the open file selected where possible.
    pub fn refresh_prompts(&mut self) {
        let open_path = self.current().map(|p| p.path.clone());
        self.prompts = self.store.list();
        self.open = open_path.and_then(|p| self.prompts.iter().position(|q| q.path == p));
    }

    /// Reload version history for the open prompt.
    pub fn refresh_history(&mut self) {
        let Some(prompt) = self.current() else {
            self.history = HistoryView::default();
            return;
        };
        let versions = version::list(&self.config.dir, &prompt.slug);
        self.history = HistoryView {
            versions,
            selected: None,
            diff: String::new(),
        };
    }

    /// Select a version and compute its diff against the live buffer.
    pub fn select_version(&mut self, ts: String) {
        let Some(prompt) = self.current() else { return };
        match version::read(&self.config.dir, &prompt.slug, &ts) {
            Ok(old) => {
                self.history.diff = version::diff(&old, &self.buffer.text);
                self.history.selected = Some(ts);
            }
            Err(e) => self.error = Some(format!("could not read version {ts}: {e}")),
        }
    }

    /// Restore the selected version, snapshotting the current text first.
    pub fn restore_selected(&mut self) {
        let Some(ts) = self.history.selected.clone() else {
            return;
        };
        let Some(prompt) = self.current().cloned() else {
            return;
        };

        // Make sure the text about to be replaced is recoverable from history, not
        // only from the undo stack. Saving first covers unsaved edits; the explicit
        // snapshot covers the already-saved case and is deduplicated when the newest
        // version is already identical — which means it is already recoverable.
        self.save(Note::Manual);
        let _ = version::snapshot(
            &self.config.dir,
            &prompt.slug,
            &self.buffer.text,
            Note::Restore,
        );
        match version::read(&self.config.dir, &prompt.slug, &ts) {
            Ok(old) => {
                self.buffer.replace_all(old, "restore");
                self.save(Note::Manual);
                self.refresh_history();
                self.status = format!("restored version {ts}");
            }
            Err(e) => self.error = Some(format!("could not restore {ts}: {e}")),
        }
    }

    /// Write the buffer to disk and snapshot it. No-op when clean.
    pub fn save(&mut self, note: Note) {
        if !self.buffer.is_dirty() {
            return;
        }
        let Some(prompt) = self.current().cloned() else {
            return;
        };
        if let Err(e) = self.store.write(&prompt, &self.buffer.text) {
            self.error = Some(format!("could not save {}: {e}", prompt.name));
            return;
        }
        let _ = version::snapshot(&self.config.dir, &prompt.slug, &self.buffer.text, note);
        self.buffer.mark_saved();
        self.refresh_history();
    }

    /// Create a new prompt and open it.
    pub fn create_prompt(&mut self, name: &str) {
        self.save(Note::Manual);
        match self.store.create(name) {
            Ok(created) => {
                self.refresh_prompts();
                if let Some(i) = self.prompts.iter().position(|p| p.path == created.path) {
                    // `open_prompt` early-returns when the index is already open, so
                    // clear the selection first.
                    self.open = None;
                    self.open_prompt(i);
                }
                self.status = format!("created {}", created.name);
            }
            Err(e) => self.error = Some(format!("could not create {name}: {e}")),
        }
    }

    /// Rename the open prompt.
    pub fn rename_open(&mut self, new_name: &str) {
        let Some(prompt) = self.current().cloned() else {
            return;
        };
        match self.store.rename(&prompt, new_name) {
            Ok(renamed) => {
                self.refresh_prompts();
                self.open = self.prompts.iter().position(|p| p.path == renamed.path);
                self.refresh_history();
                self.status = format!("renamed to {}", renamed.name);
            }
            Err(e) => self.error = Some(format!("could not rename: {e}")),
        }
    }

    /// Delete the open prompt.
    pub fn delete_open(&mut self) {
        let Some(prompt) = self.current().cloned() else {
            return;
        };
        if let Err(e) = self.store.delete(&prompt) {
            self.error = Some(format!("could not delete: {e}"));
            return;
        }
        self.open = None;
        self.buffer.load(String::new());
        self.refresh_prompts();
        if !self.prompts.is_empty() {
            self.open_prompt(0);
        }
        self.refresh_history();
        self.status = format!("deleted {}", prompt.name);
    }

    /// Rank the field for the open prompt.
    pub fn rank(&mut self) {
        self.rank_then(None);
    }

    /// Rank the field, then optionally act on the result.
    ///
    /// The table is left as it was until the model answers. It used to be filled instantly
    /// with a surface-feature estimate and then overwritten, which meant the ranking a user
    /// read might be the guess or might be the answer, with nothing on screen to say which.
    fn rank_then(&mut self, next: Option<Pending>) {
        if self.buffer.text.trim().is_empty() {
            self.status = "nothing to rank yet".into();
            return;
        }
        if self.agents.is_empty() {
            self.error = Some("no coding agents detected on PATH".into());
            return;
        }
        let text = self.buffer.text.clone();
        let agents = self.agents.clone();
        self.pending = next;
        let filter = self.config.prefs.filter.clone();
        self.runner
            .rank(text, agents, move |t, a| router::rank(t, a, &filter));
        self.status = match self.pending {
            Some(Pending::Hint(_)) => "ranking models for the hint…".into(),
            None => "ranking models…".into(),
        };
    }

    /// Download `targets` and load them, unless the user has turned downloads off.
    pub fn fetch_models(&mut self, targets: Vec<crate::models::Checkpoint>) {
        if targets.is_empty() {
            return;
        }
        if !self.config.prefs.allow_model_download {
            self.error = Some(
                "downloading model weights is switched off — turn it back on in the \
                 Models window first"
                    .into(),
            );
            return;
        }
        if self.models_job.is_some() {
            self.status = "already downloading — wait for it or stop it first".into();
            return;
        }
        let job = self.runner.fetch_models(targets);
        self.models_job = Some(job.id);
        self.running.push(job);
    }

    /// Load already-downloaded checkpoints into memory.
    pub fn load_models(&mut self, targets: Vec<crate::models::Checkpoint>) {
        let downloaded: Vec<_> = targets
            .into_iter()
            .filter(|c| crate::models::phase(c.id).is_downloaded())
            .collect();
        if downloaded.is_empty() {
            self.status = "nothing downloaded to load yet".into();
            return;
        }
        if self.models_job.is_some() {
            self.status = "already working on the models".into();
            return;
        }
        let job = self.runner.load_models(downloaded);
        self.models_job = Some(job.id);
        self.running.push(job);
    }

    /// Stop a running download.
    pub fn cancel_models(&mut self) {
        if let Some(id) = self.models_job {
            self.cancel(id);
        }
    }

    /// Scan the open prompt for personal data.
    ///
    /// Runs on a worker: the tagger is a 0.3B encoder, and the first call also loads it.
    pub fn request_sanitize(&mut self) {
        if self.buffer.text.trim().is_empty() {
            self.status = "nothing to check".into();
            return;
        }
        if self.pii_job.is_some() {
            self.status = "already checking".into();
            return;
        }
        let job = self.runner.sanitize(self.buffer.text.clone());
        self.pii_job = Some(job.id);
        self.running.push(job);
        self.pii = None;
        self.status = "checking for personal data…".into();
    }

    /// Stop a running sanitisation pass.
    pub fn cancel_sanitize(&mut self) {
        if let Some(id) = self.pii_job {
            self.cancel(id);
        }
    }

    /// Recompute the preview diff after the user has changed which findings to mask.
    pub fn refresh_pii_diff(&mut self) {
        let Some(review) = self.pii.as_mut() else {
            return;
        };
        let masked = review.scan.plan.apply(&self.buffer.text);
        review.diff = version::diff(&self.buffer.text, &masked);
    }

    /// Apply the approved masking as one undo step and one snapshot.
    pub fn accept_sanitize(&mut self) {
        let Some(review) = self.pii.take() else {
            return;
        };
        if review.scan.plan.enabled() == 0 {
            self.status = "nothing selected to mask".into();
            return;
        }
        let masked = review.scan.plan.apply(&self.buffer.text);
        if masked == self.buffer.text {
            self.status = "nothing changed".into();
            return;
        }
        let summary = review.scan.plan.summary();
        self.buffer.replace_all(masked, "mask personal data");
        self.save(Note::Sanitize);
        self.status = format!("masked — {summary}");
    }

    /// Ask for a hint about the selection, or the typed question.
    pub fn request_hint(&mut self) {
        let selection = self.buffer.selected_text();
        let Some(subject) = Subject::resolve(selection.as_deref(), &self.hint_input) else {
            self.status = "select some text or type a question first".into();
            return;
        };
        if self.config.prefs.hint_source == HintSource::Local {
            return self.launch_hint_locally(subject);
        }
        if self.agents.is_empty() {
            self.error = Some("no coding agents detected on PATH".into());
            return;
        }

        match self.ranking.clone() {
            Some(ranking) => self.launch_hint(subject, &ranking),
            // No ranking yet: get one, then come back here.
            None => self.rank_then(Some(Pending::Hint(subject))),
        }
    }

    /// Answer the hint on the local checkpoint.
    ///
    /// No ranking first, and no agent needed: there is only one thing that can answer, so
    /// there is nothing to choose between.
    fn launch_hint_locally(&mut self, subject: Subject) {
        let prompt = crate::hints::compose(&subject, &self.buffer.text);
        let job = self.runner.hint_local(prompt);
        self.running.push(job.clone());
        self.hint = Some(HintState {
            subject,
            answer: String::new(),
            job: Some(job.id),
            answered_by: Some("local model".into()),
        });
        self.hint_open = true;
        self.hint_input.clear();
    }

    /// Start the hint agent against an existing ranking.
    fn launch_hint(&mut self, subject: Subject, ranking: &Ranking) {
        let prompt = crate::hints::compose(&subject, &self.buffer.text);
        // Hints are latency-sensitive, so bias to the quickest choice that still fits
        // close to the best.
        let narrowed = narrow_to_fastest(ranking, self.config.prefs.hint_score_tolerance);
        let picked = narrowed
            .best()
            .map(|c| format!("{} · {} · {}", c.agent_display, c.model_display, c.effort));

        let job = self.runner.run_agent(
            Kind::Hint,
            "hint".into(),
            prompt,
            self.agents.clone(),
            narrowed,
            self.config.dir.clone(),
            self.config.dir.clone(),
            Duration::from_secs(90),
        );
        self.running.push(job.clone());
        self.hint = Some(HintState {
            subject,
            answer: String::new(),
            job: Some(job.id),
            answered_by: picked,
        });
        self.hint_open = true;
        self.hint_input.clear();
    }

    /// Compress the selection with the local model, or the whole prompt when nothing is
    /// selected.
    ///
    /// Runs on a worker: a long prompt is several model calls in sequence, each of which
    /// maps the weights.
    pub fn request_shrink(&mut self) {
        if self.buffer.text.trim().is_empty() {
            self.status = "nothing to shrink".into();
            return;
        }
        if self.shrink_job.is_some() {
            self.status = "already shrinking".into();
            return;
        }

        let source = self.shrink_target();
        let scope = match source.range {
            Some(_) => "the selection",
            None => "the prompt",
        };

        let job = self.runner.shrink(source.text.clone());
        self.shrink_job = Some(job.id);
        self.shrink_source = Some(source);
        self.running.push(job);
        self.shrink = None;
        self.status = format!("shrinking {scope}…");
    }

    /// What a shrink would compress right now: the selection, or the whole prompt.
    ///
    /// Separate from [`request_shrink`](Self::request_shrink) so the choice can be tested
    /// without starting a model.
    fn shrink_target(&self) -> ShrinkSource {
        match self.buffer.selected_text() {
            Some(text) => ShrinkSource {
                text,
                range: Some(self.buffer.selection.sorted()),
            },
            None => ShrinkSource {
                text: self.buffer.text.clone(),
                range: None,
            },
        }
    }

    /// Turn the open prompt into an instruction a coding agent can execute.
    ///
    /// Runs on the local checkpoint, so unlike the handoff it needs no installed agent and
    /// no ranking first — planning rewrites the request, it does not work on the repo.
    pub fn request_plan(&mut self) {
        if self.buffer.text.trim().is_empty() {
            self.status = "nothing to plan".into();
            return;
        }
        let job = self.runner.plan(self.buffer.text.clone());
        self.plan_job = Some(job.id);
        self.running.push(job);
        self.plan = None;
        self.status = "planning…".into();
    }

    /// Accept the proposed plan, replacing the buffer with it.
    pub fn accept_plan(&mut self) {
        let Some(proposal) = self.plan.take() else {
            return;
        };
        self.buffer.replace_all(&proposal.after, "plan");
        self.save(Note::Plan);
        self.status = "plan applied".into();
    }

    /// Discard the proposed plan.
    pub fn reject_plan(&mut self) {
        self.plan = None;
        self.status = "plan discarded".into();
    }

    /// Stop a running plan job.
    pub fn cancel_plan(&mut self) {
        if let Some(id) = self.plan_job
            && let Some(h) = self.running.iter().find(|h| h.id == id)
        {
            h.cancel();
        }
    }

    /// Turn the open prompt — incident notes — into a root cause analysis and postmortem.
    ///
    /// Runs on the local checkpoint, like shrinking and planning: no installed agent, no
    /// ranking first, and nothing about the incident leaves the machine. That last part is
    /// the reason this is worth having at all — incident notes carry hostnames, customer
    /// counts and stack traces, which is the material least appropriate to hand to a hosted
    /// model for summarising.
    pub fn request_rca(&mut self) {
        if self.buffer.text.trim().is_empty() {
            self.status = "nothing to analyse".into();
            return;
        }
        let job = self.runner.rca(self.buffer.text.clone());
        self.rca_job = Some(job.id);
        self.running.push(job);
        self.rca = None;
        self.status = "analysing the incident…".into();
    }

    /// Accept the proposed analysis, replacing the buffer with it.
    ///
    /// The notes it was built from are not lost: the snapshot taken here sits on top of them
    /// in version history, and the undo it creates is a single step.
    pub fn accept_rca(&mut self) {
        let Some(proposal) = self.rca.take() else {
            return;
        };
        self.buffer.replace_all(&proposal.after, "analysis");
        self.save(Note::Rca);
        self.status = "postmortem applied".into();
    }

    /// Discard the proposed analysis.
    pub fn reject_rca(&mut self) {
        self.rca = None;
        self.status = "analysis discarded".into();
    }

    /// Stop a running analysis job.
    pub fn cancel_rca(&mut self) {
        if let Some(id) = self.rca_job
            && let Some(h) = self.running.iter().find(|h| h.id == id)
        {
            h.cancel();
        }
    }

    /// Apply a proposed shrink as one undo step and one snapshot.
    ///
    /// A shrink of a selection lands back in the range it came from — and only if that range
    /// still holds what was compressed. Typing during the pass moves everything after the
    /// caret, so an unchecked write would replace the wrong passage with a rewrite of a
    /// different one.
    pub fn accept_shrink(&mut self) {
        let Some(proposal) = self.shrink.take() else {
            return;
        };
        match proposal.source.range {
            Some((lo, hi)) => {
                if self.buffer.range_text(lo, hi).as_deref() != Some(proposal.source.text.as_str())
                {
                    self.status =
                        "the prompt changed while it was shrinking — select it and try again"
                            .into();
                    return;
                }
                self.buffer.replace_range(lo, hi, &proposal.after, "shrink");
            }
            None => self.buffer.replace_all(proposal.after, "shrink"),
        }
        self.save(Note::Shrink);
        self.status = format!("shrunk — {}", proposal.savings.summary());
    }

    /// Insert the current hint answer at the caret as one undo step.
    pub fn insert_hint(&mut self, replace_selection: bool) {
        let Some(hint) = self.hint.as_ref() else {
            return;
        };
        let text = hint.answer.trim().to_string();
        if text.is_empty() {
            return;
        }
        if replace_selection {
            self.buffer.replace_selection(&text, "insert hint");
        } else {
            self.buffer.insert_at_caret(&text, "insert hint");
        }
        self.save(Note::Hint);
        self.status = "hint inserted".into();
    }

    /// Open the top-scoring (or pinned) agent in a new terminal window.
    pub fn send_to_agent(&mut self) {
        self.save(Note::Manual);
        if self.buffer.text.trim().is_empty() {
            self.status = "nothing to send".into();
            return;
        }

        let chosen = self
            .pinned_agent
            .as_deref()
            .and_then(registry::find)
            .or_else(|| {
                self.ranking
                    .as_ref()
                    .and_then(|r| r.best())
                    .and_then(|c| registry::find(c.agent_id))
            })
            .or_else(|| self.agents.first().map(|d| d.spec));

        let Some(spec) = chosen else {
            self.error = Some("no coding agents detected on PATH".into());
            return;
        };
        let slug = self
            .current()
            .map(|p| p.slug.clone())
            .unwrap_or_else(|| "prompt".into());
        match crate::agents::launch::open_in_terminal(
            spec,
            &self.config.dir,
            &self.buffer.text,
            &slug,
        ) {
            Ok(()) => self.status = format!("opened {} in a new terminal", spec.display),
            Err(e) => self.error = Some(e),
        }
    }

    /// Re-probe installed agents, discarding cached verdicts.
    pub fn refresh_agents(&mut self) {
        crate::agents::detect::clear_cache(&self.config.dir);
        // A failed model load is remembered so it isn't retried on every keystroke; an
        // explicit refresh is the user saying "try again".
        router::reset_classifiers();
        crate::pii::reset();
        self.runner.detect(self.config.dir.clone());
        self.runner.probe_models();
        self.status = "re-detecting agents…".into();
    }

    /// Ask a running job to stop, killing its child process.
    pub fn cancel(&mut self, id: JobId) {
        if let Some(h) = self.running.iter().find(|h| h.id == id) {
            h.cancel();
            self.status = format!("stopping {}…", h.label);
        }
    }

    /// Stop the running hint, if any.
    pub fn cancel_hint(&mut self) {
        if let Some(id) = self.hint.as_ref().and_then(|h| h.job) {
            self.cancel(id);
        }
    }

    /// Stop the running shrink, if any.
    pub fn cancel_shrink(&mut self) {
        if let Some(id) = self.shrink_job {
            self.cancel(id);
        }
    }

    /// Drain worker events and run periodic upkeep. Call once per frame.
    pub fn tick(&mut self) {
        self.buffer.tick();

        for event in self.runner.drain() {
            if event.is_terminal() {
                let id = event.id();
                self.running.retain(|h| h.id != id);
            }
            self.handle(event);
        }

        // Debounced autosave.
        if self.buffer.is_dirty() && self.buffer.idle_for().is_some_and(|d| d >= AUTOSAVE_IDLE) {
            self.save(Note::Autosave);
        }
    }

    fn handle(&mut self, event: Event) {
        match event {
            Event::Started { .. } => {}
            Event::Note { text, .. } => self.status = text,
            Event::Chunk { id, text } => {
                if self.hint.as_ref().and_then(|h| h.job) == Some(id)
                    && let Some(h) = self.hint.as_mut()
                {
                    h.answer.push_str(&text);
                }
            }
            Event::Done { id, kind, note } => {
                if kind == Kind::Models && self.models_job == Some(id) {
                    self.models_job = None;
                }
                // A local hint has no `Finished` event to clear its job — the answer came
                // as a chunk and this is the terminal event.
                if kind == Kind::Hint
                    && let Some(h) = self.hint.as_mut()
                    && h.job == Some(id)
                {
                    h.job = None;
                }
                self.status = note;
            }
            Event::Scanned { id, scan } => {
                if self.pii_job != Some(id) {
                    return;
                }
                self.pii_job = None;
                if scan.plan.items.is_empty() {
                    // Say what looked, so "nothing found" is not mistaken for "nothing
                    // ran". A scan that could not run arrives as `Failed`, never here.
                    self.status = format!("{} · {}", scan.plan.summary(), scan.source_label());
                    return;
                }
                self.status = scan.plan.summary();
                let masked = scan.plan.apply(&self.buffer.text);
                self.pii = Some(PiiReview {
                    diff: version::diff(&self.buffer.text, &masked),
                    scan: *scan,
                });
            }
            Event::Shrunk { id, text } => {
                if self.shrink_job != Some(id) {
                    return;
                }
                self.shrink_job = None;
                let Some(source) = self.shrink_source.take() else {
                    return;
                };
                let after = crate::shrink::clean(&text);
                let savings = Savings::measure(&source.text, &after);
                if after.trim().is_empty() {
                    self.error = Some("the shrinker returned nothing".into());
                } else if !savings.worthwhile() {
                    // Not an error: an already-terse prompt has nothing to give, and a diff
                    // that saves two characters is not worth reading.
                    self.status = format!("no useful reduction ({})", savings.summary());
                } else {
                    self.shrink = Some(ShrinkProposal {
                        diff: version::diff(&source.text, &after),
                        warnings: crate::shrink::integrity_warnings(&source.text, &after),
                        source,
                        savings,
                        after,
                    });
                    self.status = "shrink ready for review".into();
                }
            }
            Event::Planned { id, text } => {
                if self.plan_job != Some(id) {
                    return;
                }
                self.plan_job = None;
                // The schema constrains the shape, not the manners — same cleanup as shrink.
                let after = crate::shrink::clean(&text);
                if after.trim().is_empty() {
                    self.error = Some("the planner returned nothing".into());
                } else {
                    self.plan = Some(PlanProposal {
                        diff: version::diff(&self.buffer.text, &after),
                        warnings: crate::plan::warnings(&after, &self.buffer.text),
                        after,
                    });
                    self.status = "plan ready for review".into();
                }
            }
            Event::Analysed { id, text } => {
                if self.rca_job != Some(id) {
                    return;
                }
                self.rca_job = None;
                // The schema constrains the shape, not the manners — same cleanup as shrink.
                let after = crate::shrink::clean(&text);
                if after.trim().is_empty() {
                    self.error = Some("the analysis came back empty".into());
                } else {
                    self.rca = Some(RcaProposal {
                        diff: version::diff(&self.buffer.text, &after),
                        warnings: crate::rca::warnings(&after, &self.buffer.text),
                        after,
                    });
                    self.status = "postmortem ready for review".into();
                }
            }
            Event::Detected { agents, .. } => {
                let n = agents.len();
                let usable = agents.iter().filter(|a| a.usable()).count();
                self.agents = agents;
                self.status = if n == 0 {
                    "no coding agents found on PATH".into()
                } else {
                    format!("{usable} of {n} detected agents usable")
                };
            }
            Event::Ranked { ranking, .. } => {
                self.status = match ranking.best() {
                    Some(best) => format!(
                        "{} · {} — picked from {} combinations in {:.1}s",
                        best.agent_display,
                        best.model_display,
                        ranking.considered,
                        ranking.elapsed.as_secs_f32(),
                    ),
                    None => "the model returned no usable ranking".into(),
                };
                self.ranking = Some(*ranking);

                // Whatever was waiting on this ranking can now run.
                if let (Some(pending), Some(ranking)) = (self.pending.take(), self.ranking.clone())
                {
                    match pending {
                        Pending::Hint(subject) => self.launch_hint(subject, &ranking),
                    }
                }
            }
            // The hint is the only agent-backed action left; everything else pstore infers
            // runs on the local checkpoint and reports through its own event.
            Event::Finished {
                id,
                kind: Kind::Hint,
                result,
            } => {
                if let Some(h) = self.hint.as_mut()
                    && h.job == Some(id)
                {
                    h.job = None;
                    let model = if result.model_id.is_empty() {
                        "agent default"
                    } else {
                        &result.model_id
                    };
                    h.answered_by = Some(format!(
                        "{} · {model} · effort {} ({:.1}s)",
                        result.agent_id,
                        result.effort,
                        result.elapsed.as_secs_f32()
                    ));
                }
                self.status = "hint ready".into();
            }
            Event::Finished { .. } => {}
            Event::Failed { id, kind, error } => {
                if kind == Kind::Shrink && self.shrink_job == Some(id) {
                    self.shrink_job = None;
                    self.shrink_source = None;
                }
                if kind == Kind::Plan && self.plan_job == Some(id) {
                    self.plan_job = None;
                }
                if kind == Kind::Rca && self.rca_job == Some(id) {
                    self.rca_job = None;
                }
                if kind == Kind::Models && self.models_job == Some(id) {
                    self.models_job = None;
                }
                if kind == Kind::Sanitize && self.pii_job == Some(id) {
                    self.pii_job = None;
                }
                if self.hint.as_ref().and_then(|h| h.job) == Some(id)
                    && let Some(h) = self.hint.as_mut()
                {
                    h.job = None;
                }
                self.error = Some(error);
            }
            Event::Cancelled { id, kind } => {
                if kind == Kind::Shrink && self.shrink_job == Some(id) {
                    self.shrink_job = None;
                    self.shrink_source = None;
                }
                if kind == Kind::Plan && self.plan_job == Some(id) {
                    self.plan_job = None;
                }
                if kind == Kind::Rca && self.rca_job == Some(id) {
                    self.rca_job = None;
                }
                if kind == Kind::Models && self.models_job == Some(id) {
                    self.models_job = None;
                }
                if kind == Kind::Sanitize && self.pii_job == Some(id) {
                    self.pii_job = None;
                }
                if let Some(h) = self.hint.as_mut()
                    && h.job == Some(id)
                {
                    h.job = None;
                }
                self.status = "cancelled".into();
            }
        }
    }

    /// Human description of how the top candidate would be invoked.
    pub fn invocation_hint(&self) -> Option<String> {
        let best = self.ranking.as_ref()?.best()?;
        let spec = registry::find(best.agent_id)?;
        Some(format!(
            "{} · {} · {} · prompt {}",
            spec.display,
            best.model_display,
            jobs::effort_label(best.effort, best.effort_selectable),
            jobs::prompt_delivery(spec.prompt_via),
        ))
    }
}

/// Reduce a ranking to the single fastest candidate scoring within `tolerance`.
///
/// Used for hints only, where the user asked for speed. The remaining candidates are
/// kept below it so failover still has somewhere to go.
fn narrow_to_fastest(full: &Ranking, tolerance: f32) -> Ranking {
    let Some(quick) = full.fastest_within(tolerance) else {
        return full.clone();
    };
    let mut choices = vec![quick.clone()];
    choices.extend(
        full.choices
            .iter()
            .filter(|c| c.agent_id != quick.agent_id)
            .cloned(),
    );
    Ranking {
        choices,
        ..full.clone()
    }
}

/// Title for the OS window.
pub fn window_title(dir: &Path, open: Option<&Prompt>, dirty: bool) -> String {
    let name = open.map(|p| p.name.as_str()).unwrap_or("no prompt");
    let mark = if dirty { "•" } else { "" };
    format!("pstore — {mark}{name} — {}", dir.display())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::detect::Status;
    use crate::config::Prefs;
    use std::path::PathBuf;

    fn tmp_config(tag: &str) -> Config {
        let dir = std::env::temp_dir().join(format!(
            "pstore-app-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        Config {
            dir,
            prefs: Prefs::default(),
            warnings: Vec::new(),
        }
    }

    fn choice(agent: &'static str, effort: registry::Effort, fit: f32) -> router::Choice {
        router::Choice {
            agent_id: agent,
            agent_display: agent,
            model_id: "m".into(),
            model_display: "M".into(),
            tier: registry::Tier::Mid,
            effort,
            effort_selectable: true,
            metered: false,
            relative_latency: effort.latency_factor(),
            relative_price: 1.0,
            fit,
            rationale: String::new(),
            quota_weight: 1.0,
            note: String::new(),
            fact_source: None,
            row_index: 0,
        }
    }

    #[test]
    fn opening_and_editing_tracks_dirty_state() {
        let cfg = tmp_config("open");
        let dir = cfg.dir.clone();
        let store = PromptStore::new(&dir);
        let p = store.create("first").unwrap();
        store.write(&p, "hello").unwrap();

        let mut app = App::new(cfg);
        assert_eq!(app.prompts.len(), 1);
        assert_eq!(app.buffer.text, "hello");
        assert!(!app.buffer.is_dirty());

        app.buffer.replace_all("hello world", "typing");
        assert!(app.buffer.is_dirty());
        app.save(Note::Manual);
        assert!(!app.buffer.is_dirty());
        assert_eq!(std::fs::read_to_string(&p.path).unwrap(), "hello world");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn creating_a_prompt_opens_it() {
        let cfg = tmp_config("create");
        let dir = cfg.dir.clone();
        let mut app = App::new(cfg);
        app.create_prompt("notes");
        assert_eq!(app.current().map(|p| p.name.as_str()), Some("notes"));
        assert_eq!(app.buffer.text, "");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn switching_prompts_saves_the_previous_one() {
        let cfg = tmp_config("switch");
        let dir = cfg.dir.clone();
        let mut app = App::new(cfg);
        app.create_prompt("a");
        app.buffer.replace_all("content of a", "typing");
        app.create_prompt("b");

        // "a" must have been flushed to disk by the switch.
        let a = app.prompts.iter().find(|p| p.name == "a").unwrap();
        assert_eq!(std::fs::read_to_string(&a.path).unwrap(), "content of a");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn restore_snapshots_current_text_first() {
        let cfg = tmp_config("restore");
        let dir = cfg.dir.clone();
        let mut app = App::new(cfg);
        app.create_prompt("doc");

        app.buffer.replace_all("version one", "typing");
        app.save(Note::Manual);
        app.buffer.replace_all("version two", "typing");
        app.save(Note::Manual);

        let oldest = app.history.versions.last().unwrap().ts.clone();
        app.select_version(oldest.clone());
        assert!(!app.history.diff.is_empty(), "diff should show the change");
        app.restore_selected();

        assert_eq!(app.buffer.text, "version one");
        // The guarantee that matters: the text we replaced is still recoverable from
        // history. Which *note* it carries is cosmetic — an identical snapshot is
        // deduplicated, so the pre-restore entry may be the save that preceded it.
        let slug = app.current().unwrap().slug.clone();
        let recoverable = app
            .history
            .versions
            .iter()
            .filter_map(|v| version::read(&dir, &slug, &v.ts).ok())
            .any(|text| text == "version two");
        assert!(
            recoverable,
            "pre-restore text lost: {:?}",
            app.history.versions
        );

        // And Ctrl+Z still walks it back.
        assert!(app.buffer.undo());
        assert_eq!(app.buffer.text, "version two");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn restoring_over_unsaved_edits_keeps_them_in_history() {
        let cfg = tmp_config("restore-dirty");
        let dir = cfg.dir.clone();
        let mut app = App::new(cfg);
        app.create_prompt("doc");
        app.buffer.replace_all("committed text", "typing");
        app.save(Note::Manual);

        // Edit without saving, then restore an older version over the top.
        app.buffer.replace_all("unsaved work", "typing");
        let target = app.history.versions.last().unwrap().ts.clone();
        app.select_version(target);
        app.restore_selected();

        let slug = app.current().unwrap().slug.clone();
        let found = app
            .history
            .versions
            .iter()
            .filter_map(|v| version::read(&dir, &slug, &v.ts).ok())
            .any(|text| text == "unsaved work");
        assert!(found, "unsaved work must not be lost by a restore");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn deleting_the_open_prompt_falls_back_to_another() {
        let cfg = tmp_config("delete");
        let dir = cfg.dir.clone();
        let mut app = App::new(cfg);
        app.create_prompt("keep");
        app.create_prompt("doomed");
        assert_eq!(app.current().map(|p| p.name.as_str()), Some("doomed"));

        app.delete_open();
        assert_eq!(app.prompts.len(), 1);
        assert_eq!(app.current().map(|p| p.name.as_str()), Some("keep"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn actions_refuse_politely_with_no_agents() {
        let cfg = tmp_config("noagents");
        let dir = cfg.dir.clone();
        let mut app = App::new(cfg);
        app.create_prompt("doc");
        app.buffer.replace_all("some prompt text", "typing");
        app.agents.clear();

        app.rank();
        assert!(
            app.error
                .as_deref()
                .is_some_and(|e| e.contains("no coding agents"))
        );

        // A hint answered by an agent is the action that has to refuse.
        app.error = None;
        app.hint_input = "what does this do?".into();
        app.request_hint();
        assert!(
            app.error
                .as_deref()
                .is_some_and(|e| e.contains("no coding agents")),
            "got {:?}",
            app.error
        );

        // Plan must *not* refuse: it runs on the local checkpoint, so an installed agent
        // has nothing to do with whether it can run. Same for shrink, untested here.
        app.error = None;
        app.request_plan();
        assert_eq!(
            app.error, None,
            "planning is local and must not require an agent"
        );

        // The analysis even more so: the reason it is worth having is that the incident
        // never leaves the machine, so an agent must not be on its path at all.
        app.error = None;
        app.request_rca();
        assert_eq!(
            app.error, None,
            "analysing an incident is local and must not require an agent"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// An empty buffer is a request with nothing in it, and the model costs seconds: every
    /// local action says so in the status bar instead of starting.
    #[test]
    fn an_empty_buffer_is_not_analysed() {
        let cfg = tmp_config("emptyrca");
        let dir = cfg.dir.clone();
        let mut app = App::new(cfg);
        app.create_prompt("notes");
        app.buffer.replace_all("   \n\n", "typing");

        app.request_rca();
        assert!(
            app.rca_job.is_none(),
            "an empty buffer started a model call"
        );
        assert!(app.status.contains("nothing to analyse"), "{}", app.status);

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Choosing the local model is also the answer to having no agent installed, so it must
    /// not go through the check that refuses without one.
    #[test]
    fn a_local_hint_needs_no_agent_and_no_ranking() {
        let cfg = tmp_config("localhint");
        let dir = cfg.dir.clone();
        let mut app = App::new(cfg);
        app.create_prompt("doc");
        app.buffer.replace_all("some prompt text", "typing");
        app.agents.clear();
        app.ranking = None;
        app.config.prefs.hint_source = HintSource::Local;

        app.hint_input = "is this clear?".into();
        app.request_hint();

        assert_eq!(app.error, None, "a local hint must not require an agent");
        let hint = app.hint.as_ref().expect("the hint started");
        assert!(hint.job.is_some(), "it dispatched without a ranking first");
        assert_eq!(hint.answered_by.as_deref(), Some("local model"));
        assert_eq!(app.hint_input, "", "the question box is consumed");

        app.cancel_hint();
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The terminal event for a local hint is `Done`, not `Finished` — it has no agent run
    /// to report. Without this the spinner never stops.
    #[test]
    fn a_local_hint_finishes_on_its_done_event() {
        let cfg = tmp_config("localhintdone");
        let dir = cfg.dir.clone();
        let mut app = App::new(cfg);
        app.hint = Some(HintState {
            subject: Subject::Question("q".into()),
            answer: String::new(),
            job: Some(JobId(7)),
            answered_by: Some("local model".into()),
        });

        app.handle(Event::Chunk {
            id: JobId(7),
            text: "the answer".into(),
        });
        app.handle(Event::Done {
            id: JobId(7),
            kind: Kind::Hint,
            note: "hint ready".into(),
        });

        let hint = app.hint.as_ref().expect("still there");
        assert_eq!(hint.answer, "the answer");
        assert_eq!(hint.job, None, "the job must be cleared");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn empty_prompt_actions_report_rather_than_erroring() {
        let cfg = tmp_config("emptyactions");
        let dir = cfg.dir.clone();
        let mut app = App::new(cfg);
        app.create_prompt("blank");

        app.rank();
        assert!(app.error.is_none(), "an empty prompt is not an error");
        assert!(app.status.contains("nothing to rank"));

        app.request_shrink();
        assert!(app.status.contains("nothing to shrink"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn hint_requires_a_selection_or_a_question() {
        let cfg = tmp_config("hintempty");
        let dir = cfg.dir.clone();
        let mut app = App::new(cfg);
        app.create_prompt("doc");
        app.buffer.replace_all("text", "typing");

        app.request_hint();
        assert!(app.hint.is_none());
        assert!(app.status.contains("select some text or type a question"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn narrow_to_fastest_puts_the_quick_candidate_first_and_keeps_failover() {
        let full = Ranking {
            choices: vec![
                choice("claude", registry::Effort::Max, 100.0),
                choice("claude", registry::Effort::Low, 98.0),
                choice("codex", registry::Effort::High, 97.0),
            ],
            ..Ranking::default()
        };
        let narrowed = narrow_to_fastest(&full, 5.0);
        assert_eq!(
            narrowed.choices[0].effort,
            registry::Effort::Low,
            "fastest first"
        );
        assert!(
            narrowed.choices.iter().any(|c| c.agent_id == "codex"),
            "other agents must remain for failover"
        );
    }

    #[test]
    fn narrow_to_fastest_respects_a_zero_tolerance() {
        let full = Ranking {
            choices: vec![
                choice("claude", registry::Effort::Max, 100.0),
                choice("claude", registry::Effort::Low, 80.0),
            ],
            ..Ranking::default()
        };
        let narrowed = narrow_to_fastest(&full, 0.0);
        assert_eq!(
            narrowed.choices[0].effort,
            registry::Effort::Max,
            "with no tolerance the best score wins outright"
        );
    }

    #[test]
    fn window_title_marks_unsaved_changes() {
        let dir = PathBuf::from("/work");
        let clean = window_title(&dir, None, false);
        assert!(clean.contains("no prompt"));
        assert!(!clean.contains('•'));

        let store = PromptStore::new("/work");
        let p = store.list().first().cloned();
        if let Some(p) = p {
            assert!(window_title(&dir, Some(&p), true).contains('•'));
        }
    }

    #[test]
    fn detection_event_updates_the_agent_list() {
        let cfg = tmp_config("detectevent");
        let dir = cfg.dir.clone();
        let mut app = App::new(cfg);

        app.handle(Event::Detected {
            id: JobId(1),
            agents: vec![Detected {
                spec: registry::find("claude").unwrap(),
                path: PathBuf::from("/usr/bin/claude"),
                version: Some("1.0".into()),
                has_credentials: true,
                status: Status::Verified,
                configured_model: None,
                models: Vec::new(),
            }],
        });
        assert_eq!(app.agents.len(), 1);
        assert!(app.status.contains("1 of 1"), "got {}", app.status);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A whole-document source, as `request_shrink` would have captured it.
    fn whole(text: &str) -> ShrinkSource {
        ShrinkSource {
            text: text.into(),
            range: None,
        }
    }

    #[test]
    fn a_shrink_that_saves_nothing_is_not_offered() {
        let cfg = tmp_config("shrinknoop");
        let dir = cfg.dir.clone();
        let mut app = App::new(cfg);
        app.create_prompt("doc");
        app.buffer.replace_all("keep this text exactly", "typing");
        app.shrink_job = Some(JobId(9));
        app.shrink_source = Some(whole(&app.buffer.text));

        app.handle(Event::Shrunk {
            id: JobId(9),
            text: "keep this text exactly".into(),
        });
        assert!(
            app.shrink.is_none(),
            "an identical rewrite must not be proposed"
        );
        assert!(app.status.contains("no useful reduction"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_real_shrink_is_proposed_with_a_diff_and_applies_as_one_undo() {
        let cfg = tmp_config("shrinkok");
        let dir = cfg.dir.clone();
        let mut app = App::new(cfg);
        app.create_prompt("doc");
        let original = "Please could you kindly update src/main.rs, and I would also \
                        really appreciate it if you ran the tests afterwards.";
        app.buffer.replace_all(original, "typing");
        app.save(Note::Manual);
        app.shrink_job = Some(JobId(4));
        app.shrink_source = Some(whole(&app.buffer.text));

        app.handle(Event::Shrunk {
            id: JobId(4),
            text: "Update src/main.rs, run tests.".into(),
        });

        let proposal = app.shrink.clone().expect("shrink should be proposed");
        assert!(proposal.savings.worthwhile());
        assert!(!proposal.diff.is_empty());
        assert!(
            proposal.warnings.is_empty(),
            "src/main.rs was kept: {:?}",
            proposal.warnings
        );

        app.accept_shrink();
        assert!(app.buffer.text.starts_with("Update src/main.rs"));
        assert!(app.buffer.undo(), "one undo reverses the whole shrink");
        assert_eq!(app.buffer.text, original);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_shrink_that_drops_a_file_reference_is_flagged() {
        let cfg = tmp_config("shrinkwarn");
        let dir = cfg.dir.clone();
        let mut app = App::new(cfg);
        app.create_prompt("doc");
        app.buffer.replace_all(
            "Update src/main.rs and src/lib.rs and also the docs please.",
            "typing",
        );
        app.shrink_job = Some(JobId(5));
        app.shrink_source = Some(whole(&app.buffer.text));

        app.handle(Event::Shrunk {
            id: JobId(5),
            text: "Update src/main.rs.".into(),
        });

        let proposal = app.shrink.clone().expect("proposal expected");
        assert!(
            proposal.warnings.iter().any(|w| w.contains("src/lib.rs")),
            "dropped path must be flagged: {:?}",
            proposal.warnings
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Shrinking a selection must leave the rest of the prompt exactly as it was, including
    /// the paragraph the user is still working on.
    #[test]
    fn shrinking_a_selection_replaces_only_the_selection() {
        let cfg = tmp_config("shrinkselection");
        let dir = cfg.dir.clone();
        let mut app = App::new(cfg);
        app.create_prompt("doc");
        let original =
            "Keep this line.\nPlease could you kindly update src/main.rs.\nKeep this too.";
        app.buffer.replace_all(original, "typing");
        app.save(Note::Manual);

        let nothing_selected = app.shrink_target();
        assert_eq!(nothing_selected.range, None, "no selection means the lot");
        assert_eq!(nothing_selected.text, original);

        // The middle line, as a drag would leave it.
        let start = original.chars().position(|c| c == '\n').unwrap() + 1;
        let end = start
            + "Please could you kindly update src/main.rs."
                .chars()
                .count();
        app.buffer.selection = crate::editor::Selection { start, end };

        let source = app.shrink_target();
        assert_eq!(source.range, Some((start, end)));
        assert_eq!(source.text, "Please could you kindly update src/main.rs.");

        // Straight to the result, so the test does not start a model.
        app.shrink_job = Some(JobId(7));
        app.shrink_source = Some(source);
        app.handle(Event::Shrunk {
            id: JobId(7),
            text: "Update src/main.rs.".into(),
        });
        app.accept_shrink();

        assert_eq!(
            app.buffer.text,
            "Keep this line.\nUpdate src/main.rs.\nKeep this too."
        );
        assert!(app.buffer.undo(), "one undo reverses the whole shrink");
        assert_eq!(app.buffer.text, original);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The bug this guards: the model takes seconds, and typing in the meantime moves every
    /// character after the caret. Writing the rewrite back at the captured offsets would
    /// then replace a passage that is no longer the one that was compressed.
    #[test]
    fn a_selection_shrink_refuses_to_apply_over_edited_text() {
        let cfg = tmp_config("shrinkmoved");
        let dir = cfg.dir.clone();
        let mut app = App::new(cfg);
        app.create_prompt("doc");
        app.buffer
            .replace_all("Alpha please.\nBeta please.", "typing");
        let moved = "PREFIX Alpha please.\nBeta please.";

        app.shrink = Some(ShrinkProposal {
            source: ShrinkSource {
                text: "Beta please.".into(),
                range: Some((14, 26)),
            },
            after: "Beta.".into(),
            savings: Savings::measure("Beta please.", "Beta."),
            warnings: Vec::new(),
            diff: String::new(),
        });
        // The user kept typing while the model ran.
        app.buffer.replace_all(moved, "typing");

        app.accept_shrink();
        assert_eq!(app.buffer.text, moved, "the buffer must be left alone");
        assert!(
            app.status.contains("changed while it was shrinking"),
            "got {}",
            app.status
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn hint_chunks_accumulate_into_the_answer() {
        let cfg = tmp_config("hintchunks");
        let dir = cfg.dir.clone();
        let mut app = App::new(cfg);
        app.hint = Some(HintState {
            subject: Subject::Question("q".into()),
            answer: String::new(),
            job: Some(JobId(3)),
            answered_by: None,
        });

        app.handle(Event::Chunk {
            id: JobId(3),
            text: "Hello ".into(),
        });
        app.handle(Event::Chunk {
            id: JobId(3),
            text: "world".into(),
        });
        // A chunk from an unrelated job must not leak in.
        app.handle(Event::Chunk {
            id: JobId(99),
            text: "IGNORED".into(),
        });

        assert_eq!(app.hint.as_ref().unwrap().answer, "Hello world");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn masking_applies_the_selected_findings_as_one_undo_step() {
        let cfg = tmp_config("pii");
        let dir = cfg.dir.clone();
        let mut app = App::new(cfg);
        app.create_prompt("doc");
        let original = "Il cliente Mario ha scritto da mario@example.com il 2024-01-01.";
        app.buffer.replace_all(original, "typing");
        app.save(Note::Manual);
        app.pii_job = Some(JobId(11));

        // Built directly rather than by scanning: `sanitize` now needs the model, and this
        // test is about what the app does with a scan, not about producing one.
        let scan = crate::pii::Scan {
            plan: crate::pii::Plan::build(vec![
                crate::pii::Finding {
                    tag: "FULLNAME".into(),
                    start: 11,
                    end: 16,
                    text: "Mario".into(),
                    score: 0.97,
                },
                crate::pii::Finding {
                    tag: "EMAIL".into(),
                    start: 34,
                    end: 51,
                    text: "mario@example.com".into(),
                    score: 0.99,
                },
                crate::pii::Finding {
                    tag: "DATE".into(),
                    start: 55,
                    end: 65,
                    text: "2024-01-01".into(),
                    score: 0.95,
                },
            ]),
            elapsed: std::time::Duration::from_millis(1),
        };

        app.handle(Event::Scanned {
            id: JobId(11),
            scan: Box::new(scan),
        });

        let review = app.pii.clone().expect("a review should be offered");
        assert!(!review.diff.is_empty(), "the diff should show the masking");
        assert!(
            review
                .scan
                .plan
                .items
                .iter()
                .any(|i| i.finding.tag == "EMAIL" && i.masked),
            "the email must be found and on by default: {:?}",
            review.scan.plan.items
        );

        app.accept_sanitize();
        assert!(!app.buffer.text.contains("mario@example.com"));
        assert!(
            !app.buffer.text.contains("Mario "),
            "got {}",
            app.buffer.text
        );
        assert!(
            app.buffer.text.contains("[EMAIL_1]"),
            "got {}",
            app.buffer.text
        );
        // A date is part of the request, not personal data, so it stays unless asked for.
        assert!(
            app.buffer.text.contains("2024-01-01"),
            "got {}",
            app.buffer.text
        );
        // The request itself survives.
        assert!(app.buffer.text.contains("Il cliente"));

        assert!(app.buffer.undo(), "one undo reverses the whole masking");
        assert_eq!(app.buffer.text, original);

        // And the pre-masking text is in version history too.
        let slug = app.current().unwrap().slug.clone();
        assert!(
            version::list(&dir, &slug)
                .iter()
                .filter_map(|v| version::read(&dir, &slug, &v.ts).ok())
                .any(|t| t == original),
            "the original must stay recoverable"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_scan_that_finds_nothing_reports_rather_than_opening_a_window() {
        let cfg = tmp_config("piiclean");
        let dir = cfg.dir.clone();
        let mut app = App::new(cfg);
        app.create_prompt("doc");
        app.buffer
            .replace_all("Refactor src/main.rs to extract the parser.", "typing");
        app.pii_job = Some(JobId(12));

        app.handle(Event::Scanned {
            id: JobId(12),
            scan: Box::new(crate::pii::Scan {
                plan: crate::pii::Plan::default(),
                elapsed: std::time::Duration::from_millis(1),
            }),
        });
        assert!(app.pii.is_none(), "nothing to review");
        assert!(app.pii_job.is_none());
        assert!(
            app.status.contains("no personal data"),
            "the status must say the check ran: {}",
            app.status
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn sanitising_an_empty_prompt_is_not_an_error() {
        let cfg = tmp_config("piiempty");
        let dir = cfg.dir.clone();
        let mut app = App::new(cfg);
        app.create_prompt("blank");
        app.request_sanitize();
        assert!(app.error.is_none());
        assert!(app.pii_job.is_none(), "no worker for an empty buffer");
        assert!(app.status.contains("nothing to check"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn downloads_are_refused_while_they_are_switched_off() {
        let cfg = tmp_config("nodownload");
        let dir = cfg.dir.clone();
        let mut app = App::new(cfg);
        app.config.prefs.allow_model_download = false;

        app.fetch_models(vec![crate::models::LLM_TERNARY]);
        assert!(app.models_job.is_none(), "no job should start");
        assert!(
            app.error
                .as_deref()
                .is_some_and(|e| e.contains("switched off")),
            "got {:?}",
            app.error
        );

        // Nothing to load either, since nothing is downloaded in a fresh cache probe.
        app.error = None;
        app.load_models(vec![crate::models::LLM_TERNARY]);
        assert!(app.error.is_none(), "a missing checkpoint is not an error");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The status line after a ranking should name the pick and how wide a field it came
    /// from — "picked from 30" is what tells the user the shortlist is a selection rather
    /// than everything installed.
    #[test]
    fn a_ranked_event_names_the_pick_and_the_field() {
        let cfg = tmp_config("rankstatus");
        let dir = cfg.dir.clone();
        let mut app = App::new(cfg);

        app.handle(Event::Ranked {
            id: JobId(1),
            ranking: Box::new(Ranking {
                choices: vec![choice("claude", registry::Effort::High, 91.0)],
                considered: 30,
                ..Ranking::default()
            }),
        });

        assert!(app.status.contains("claude"), "got {}", app.status);
        assert!(app.status.contains("30"), "got {}", app.status);
        assert!(app.ranking.is_some(), "the ranking should be kept");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A ranking that produced nothing must not read as a successful ranking of nothing.
    #[test]
    fn an_empty_ranking_says_so() {
        let cfg = tmp_config("rankempty");
        let dir = cfg.dir.clone();
        let mut app = App::new(cfg);

        app.handle(Event::Ranked {
            id: JobId(1),
            ranking: Box::new(Ranking::default()),
        });
        assert!(
            app.status.contains("no usable ranking"),
            "got {}",
            app.status
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn inserting_a_hint_is_one_undo_step() {
        let cfg = tmp_config("hintinsert");
        let dir = cfg.dir.clone();
        let mut app = App::new(cfg);
        app.create_prompt("doc");
        app.buffer.replace_all("before after", "typing");
        app.save(Note::Manual);
        app.buffer.selection = crate::editor::Selection { start: 7, end: 7 };
        app.hint = Some(HintState {
            subject: Subject::Question("q".into()),
            answer: "INSERTED ".into(),
            job: None,
            answered_by: None,
        });

        app.insert_hint(false);
        assert_eq!(app.buffer.text, "before INSERTEDafter");
        assert!(app.buffer.undo());
        assert_eq!(app.buffer.text, "before after");
        std::fs::remove_dir_all(&dir).ok();
    }
}
