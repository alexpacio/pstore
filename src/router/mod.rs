//! Picking which agent, model and effort should answer a prompt.
//!
//! pstore enumerates every (agent, model, effort) combination the machine can actually
//! run — from [`crate::agents::detect`] crossed with the registry — hands that list to the
//! local model along with the prompt, and gets back a ranked shortlist with a reason for
//! each pick. See [`llm`].
//!
//! This replaced a hand-built scorer: a six-dimension capability vector per prompt, a
//! difficulty classifier, per-dimension skill vectors for every model, and a weighted-RMS
//! shortfall between them. That machinery existed to approximate a judgement — "is this
//! model right for this prompt?" — that the model can now simply be asked. The skill
//! vectors in particular were maintained by hand and went stale every time a vendor
//! shipped anything.
//!
//! There is deliberately **no fallback**. Earlier versions degraded to a surface-feature
//! estimate when the weights were missing, which meant every ranking carried an invisible
//! question of which implementation produced it. Now [`rank`] either answers from the model
//! or returns the reason it cannot, and the caller disables the feature rather than quietly
//! ranking worse.
//!
//! **The prompt is judged on two axes before anything is ranked**, in one cheap call — see
//! [`llm::Demand`] and [`llm::Breadth`]. Difficulty picks the weight class; breadth picks how
//! long that model should think. They are asked separately because they answer different
//! questions and genuinely come apart: renaming a method at 40 call sites is easy and wide, and
//! finding the off-by-one that only shows under concurrent writers is hard and narrow. With
//! difficulty as the only input, [`Effort::XHigh`] and [`Effort::Max`] were unreachable — the
//! registry offered them and no prompt could ask for one. See [`llm::target_effort`] for the
//! table the two combine through.

pub mod hub;
pub mod llm;
pub mod session;

use std::borrow::Cow;

use crate::agents::detect::Detected;
use crate::agents::registry::{Effort, Tier};
use crate::filter::Filter;
use crate::knowledge::Brief;

/// A model name, which is either a registry constant or a string read out of an agent's config.
///
/// `&'static str` everywhere would be simpler, and was — until pstore started discovering the
/// model an agent is really configured with (see [`crate::agents::configured`]). Those names are
/// only known at runtime, and a `Cow` keeps the registry path allocation-free while letting a
/// discovered name travel the same code.
pub type Name = Cow<'static, str>;

/// How many candidates the model is asked to return.
///
/// Five is enough to show the shape of the field — a frontier model, a cheap one, a couple
/// of efforts in between — without asking the model to rank thirty combinations it has no
/// real basis to separate at the tail.
pub const SHORTLIST: usize = 5;

/// One ranked (agent, model, effort) combination.
#[derive(Debug, Clone, PartialEq)]
pub struct Choice {
    /// Agent id from the registry.
    pub agent_id: &'static str,
    /// Agent label for the UI.
    pub agent_display: &'static str,
    /// Model id to pass to the agent, empty when the agent picks its own.
    pub model_id: Name,
    /// Model label for the UI.
    pub model_display: Name,
    /// Weight class, shown as context.
    pub tier: Tier,
    /// Effort level to request.
    pub effort: Effort,
    /// Whether pstore can actually select this effort, or is only predicting it.
    pub effort_selectable: bool,
    /// Whether this model is billed per token rather than covered by the subscription.
    pub metered: bool,
    /// Relative time-to-answer, `1.0` being the fastest effort. Registry data, not the
    /// model's opinion.
    pub relative_latency: f32,
    /// Relative token price. **Display only** — never scored, and shown so the developer can
    /// see the spend they are being handed rather than have it decided for them.
    pub relative_price: f32,
    /// How fast this drains the subscription's allowance, relative to the vendor's lightest
    /// model. Unlike [`Self::relative_price`], the ranker *is* told this — see
    /// [`crate::agents::registry::ModelSpec::quota_weight`].
    pub quota_weight: f32,
    /// The facts pstore handed the ranker about this model, or empty if it needed none.
    ///
    /// Kept on the choice so the UI can show *what the placement was made from*, not just what
    /// the model concluded. A shortlist is easier to trust — or to correct — when the evidence
    /// behind each row is visible.
    pub note: String,
    /// Where [`Self::note`] came from. `None` when the checkpoint needed no telling.
    pub fact_source: Option<crate::knowledge::Source>,
    /// How well the model judged this fits, `0..=100`.
    ///
    /// Bounded by position: the grammar gives each rank its own descending band, so the number
    /// is consistent with the order rather than a free-floating self-assessment. See
    /// [`llm::fit_band`].
    pub fit: f32,
    /// The model's one-line reason for the placement.
    pub rationale: String,
    /// Which option in the list the model picked, kept so [`llm::degeneracy`] can tell a
    /// ranking from an enumeration.
    pub row_index: usize,
}

/// What the prompt was judged to need, before any model was ranked against it.
///
/// The premise of the shortlist rather than a summary of it, which is why every front end shows
/// it above the table: a ranking that looks wrong is usually a judgement that was wrong, and this
/// is the line that lets someone see which of the two to argue with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Judgement {
    /// How much capability the work needs: `easy`, `moderate` or `hard`. See [`llm::Demand`].
    pub demand: &'static str,
    /// How much of the codebase it reaches across, in words. See [`llm::Breadth`].
    pub breadth: &'static str,
    /// The effort the shortlist was steered towards, from the two together.
    ///
    /// Kept alongside the labels rather than recomputed by each front end: the mapping is
    /// [`llm::target_effort`]'s to own, and three copies of it would be three chances to disagree
    /// with the prompt the ranker was actually given.
    pub effort: Effort,
    /// The phrase the model gave for why it judged the prompt this way.
    pub because: String,
}

impl Judgement {
    /// The judgement in one line, for a front end that has room for one.
    pub fn summary(&self) -> String {
        format!(
            "{} · {} · effort {}",
            self.demand, self.breadth, self.effort
        )
    }

    /// The phrase that decided it, with its separator — or nothing at all.
    ///
    /// `because` is whatever the model put in that field, and the field is optional: a reply
    /// that omits it leaves this empty. Every front end wants the same thing then, which is
    /// silence rather than a dangling `judged hard · one edit · effort high — `, so the check
    /// lives here instead of in each of the three.
    pub fn because_suffix(&self) -> String {
        if self.because.trim().is_empty() {
            String::new()
        } else {
            format!(" — {}", self.because)
        }
    }
}

/// A ranked shortlist over the detected agents.
#[derive(Debug, Clone, Default)]
pub struct Ranking {
    /// Choices, best first. At most [`SHORTLIST`] of them.
    pub choices: Vec<Choice>,
    /// How many combinations were offered to the model.
    pub considered: usize,
    /// Agents that were detected but excluded, with the reason.
    ///
    /// Two kinds of exclusion arrive here: an agent that cannot run, and one whose model
    /// nothing could describe. Both belong in the same place — the question they answer is
    /// "why is that not in the list?".
    pub excluded: Vec<(&'static str, String)>,
    /// What the prompt was judged to need, and the phrase that decided it.
    ///
    /// Its own model call, made before ranking — see [`llm::Demand`], which explains why that is
    /// worth an extra few seconds. Shown because it is the premise of everything below it: a
    /// shortlist that looks wrong is usually a judgement that was wrong, and the two axes fail
    /// differently — a bad difficulty read puts the wrong model first, a bad breadth read puts
    /// the right model at the wrong effort.
    pub judged: Option<Judgement>,
    /// How many of the ranked models pstore had to describe to the checkpoint itself.
    ///
    /// Provenance rather than trivia: it is the difference between a ranking the checkpoint made
    /// from its own knowledge and one it made from facts pstore supplied. See
    /// [`crate::knowledge`].
    pub described: usize,
    /// Set when the model listed the options instead of ranking them, with the evidence.
    ///
    /// A degenerate answer is populated in every field and wrong in the only one that matters, so
    /// it cannot be left to look like a result. See [`llm::degeneracy`].
    pub degenerate: Option<String>,
    /// How long ranking took, model startup included.
    pub elapsed: std::time::Duration,
}

impl Ranking {
    /// The choice pstore would actually use.
    pub fn best(&self) -> Option<&Choice> {
        self.choices.first()
    }

    /// The fastest choice within `tolerance` points of the best.
    ///
    /// The hint path: hints are wanted quickly, and latency is a real-time property rather
    /// than a matter of taste, so it is read from the registry rather than asked of the
    /// model.
    pub fn fastest_within(&self, tolerance: f32) -> Option<&Choice> {
        let best = self.best()?.fit;
        self.choices
            .iter()
            .filter(|c| c.fit + tolerance >= best)
            .min_by(|a, b| a.relative_latency.total_cmp(&b.relative_latency))
    }
}

/// One combination offered to the model for ranking.
///
/// Built from the registry so that whatever comes back can be mapped straight onto real
/// launch parameters — the model chooses among these by index and never names an agent or
/// a model string itself, so it cannot invent a combination that does not exist.
#[derive(Debug, Clone)]
pub struct Candidate {
    pub agent_id: &'static str,
    pub agent_display: &'static str,
    pub model_id: Name,
    pub model_display: Name,
    pub tier: Tier,
    pub effort: Effort,
    pub effort_selectable: bool,
    pub metered: bool,
    pub relative_price: f32,
    /// How fast this burns the subscription's allowance. See [`ModelSpec::quota_weight`].
    pub quota_weight: f32,
}

/// Every combination the detected agents can run **and** policy permits, plus the agents
/// that were excluded outright.
///
/// Filtering happens here rather than after ranking, so a model the developer has ruled out
/// is never offered to the local model at all. Ranking it and then discarding the result
/// would waste the only expensive step and, worse, could leave the shortlist empty for
/// reasons the user cannot see.
pub fn candidates(
    detected: &[Detected],
    filter: &Filter,
) -> (Vec<Candidate>, Vec<(&'static str, String)>) {
    let mut out = Vec::new();
    let mut excluded = Vec::new();

    for agent in detected {
        if !agent.usable() {
            let reason = match &agent.status {
                crate::agents::detect::Status::Blocked(u) => u.reason(),
                _ => "unavailable".to_string(),
            };
            excluded.push((agent.spec.id, reason));
            continue;
        }
        let selectable = agent.spec.effort_flag.is_supported();
        let mut offered = 0usize;

        for model in models_of(agent) {
            if !filter.allows_model(agent.spec.id, &model.id, &model.display, model.metered) {
                continue;
            }
            for &effort in agent.spec.scoreable_efforts() {
                if !filter.allows_effort(effort) {
                    continue;
                }
                offered += 1;
                out.push(Candidate {
                    agent_id: agent.spec.id,
                    agent_display: agent.spec.display,
                    model_id: model.id.clone(),
                    model_display: model.display.clone(),
                    tier: model.tier,
                    effort,
                    effort_selectable: selectable,
                    metered: model.metered,
                    relative_price: model.relative_price,
                    quota_weight: model.quota_weight,
                });
            }
        }

        // A usable agent with nothing left is a configuration outcome, not a detection
        // one, and it reads as a bug unless it is stated.
        if offered == 0 {
            excluded.push((agent.spec.id, "every model filtered out by config".into()));
        }
    }
    (out, excluded)
}

/// One model an agent could be asked to run, whatever the source of its name.
struct Offer {
    id: Name,
    display: Name,
    tier: Tier,
    metered: bool,
    relative_price: f32,
    quota_weight: f32,
}

/// The models to offer for `agent`: its own catalog, the registry table, or the one its config
/// names.
///
/// Four cases, in descending order of how much pstore can trust the answer:
///
/// * the agent publishes a catalog — those are the models it will actually accept, so they win
///   outright over the registry's hand-written guess at the same list;
/// * it does not, but the registry lists models — pstore can pass `--model`, so it offers each;
/// * neither, but the agent's config names a model — that name is offered, unselectable but real,
///   so the ranker is judging the model the agent will actually run;
/// * none of the above — a single nameless offer, which [`crate::knowledge`] then refuses to rank.
///   It is still *produced* rather than skipped here, so the reason the agent is missing from the
///   shortlist can be stated instead of the agent just vanishing.
fn models_of(agent: &Detected) -> Vec<Offer> {
    // A discovered catalog outranks the table because it cannot be stale: it is what the agent
    // fetched for itself. The table is what someone wrote down the last time they looked.
    if !agent.models.is_empty() {
        return agent
            .models
            .iter()
            .map(|m| Offer {
                id: Cow::Owned(m.id.clone()),
                display: Cow::Owned(m.display.clone()),
                tier: m.tier,
                metered: false,
                relative_price: m.relative_price,
                quota_weight: m.quota_weight,
            })
            .collect();
    }

    if !agent.spec.models.is_empty() {
        return agent
            .spec
            .models
            .iter()
            .map(|m| Offer {
                id: Cow::Borrowed(m.id),
                display: Cow::Borrowed(m.display),
                tier: m.tier,
                metered: m.metered,
                relative_price: m.relative_price,
                quota_weight: m.quota_weight,
            })
            .collect();
    }

    let placeholder = &crate::agents::registry::UNKNOWN_MODEL;
    let (id, display) = match &agent.configured_model {
        // Both from the discovered name: the id is what the agent would run, and it is also
        // the only honest label for it — inventing a prettier display name would mean the UI
        // showing something that appears in nobody's config.
        Some(found) => (Cow::Owned(found.clone()), Cow::Owned(found.clone())),
        None => (
            Cow::Borrowed(placeholder.id),
            Cow::Borrowed(placeholder.display),
        ),
    };
    vec![Offer {
        id,
        display,
        // Unknown rather than flattering: a discovered model has no tier pstore can vouch for,
        // and `Mid` is the claim that biases least.
        tier: placeholder.tier,
        metered: placeholder.metered,
        relative_price: placeholder.relative_price,
        quota_weight: placeholder.quota_weight,
    }]
}

/// Drop the candidates whose model nothing can describe, and say why.
///
/// Only reachable with local inference: without it there is no ranking to protect.
///
/// The ranking list and the exclusion list are updated together on purpose: a candidate removed
/// without a stated reason is an agent that silently disappeared from the shortlist, which is
/// the bug report "why is Crush never suggested?" and no way to answer it.
///
/// Exclusions are recorded once per agent, not once per (model, effort) pair — five efforts of
/// one nameless model are one problem, and listing it five times would bury the others.
fn withhold_unknown(
    candidates: &mut Vec<Candidate>,
    excluded: &mut Vec<(&'static str, String)>,
    brief: &Brief,
) {
    let mut said: Vec<(&'static str, String)> = Vec::new();
    candidates.retain(|c| {
        if brief.permits(&c.model_id) {
            return true;
        }
        let why = brief
            .unknown
            .iter()
            .find(|(m, _)| *m == c.model_id)
            .map(|(_, why)| why.clone())
            .unwrap_or_else(|| format!("nothing describes {}", c.model_display));
        let entry = (c.agent_id, why);
        if !said.contains(&entry) {
            said.push(entry);
        }
        false
    });
    excluded.extend(said);
}

/// Rank the runnable combinations against `text` with the local model.
///
/// Blocking — call it from a worker thread. Each call spawns `llama-cli`, which maps the
/// weights before generating, so expect seconds rather than milliseconds.
///
/// Returns the reason on failure (model not downloaded, runtime not provisioned, malformed
/// output) so the caller can disable the feature and say why. Nothing is downloaded here:
/// weights and runtime both come from the Models window, so a first run reports "not
/// downloaded" instead of stalling on a 7.17 GB transfer nobody asked for.
pub fn rank(text: &str, detected: &[Detected], filter: &Filter) -> Result<Ranking, String> {
    // `mut` on both because ranking withholds undescribed models from the field, moving them
    // from `candidates` into `excluded` with a stated reason rather than scoring them blind.
    let (mut candidates, mut excluded) = candidates(detected, filter);
    if candidates.is_empty() {
        // Two very different problems, and sending the user to the wrong one wastes their
        // time: nothing installed, or everything installed ruled out by their own config.
        return Err(if detected.iter().any(|d| d.usable()) {
            format!(
                "every installed model is excluded by the model filter ({})",
                filter.summary()
            )
        } else {
            "no usable coding agents were detected".into()
        });
    }

    {
        // Before the field is ranked, work out what can truthfully be said about each model in
        // it — and withhold the ones nothing can describe. Ranking a model pstore cannot name
        // does not produce a worse row; it moves every real row below it.
        let names: Vec<String> = candidates.iter().map(|c| c.model_id.to_string()).collect();
        // Models discovered from an agent's own catalog arrive with the vendor's description
        // attached; hand it to `resolve` so a model that shipped after pstore's table was written
        // is still rankable. See [`crate::agents::catalog::CatalogModel::note`].
        let described = |name: &str| {
            detected
                .iter()
                .flat_map(|d| d.models.iter())
                .find(|m| m.id == name)
                .map(|m| m.note.clone())
        };
        let brief = crate::knowledge::resolve(
            &names,
            &described,
            llm::known_models,
            crate::knowledge::lookup,
        );
        withhold_unknown(&mut candidates, &mut excluded, &brief);

        if candidates.is_empty() {
            return Err(format!(
                "no model in the field could be identified, so there is nothing to rank — {}",
                brief
                    .unknown
                    .iter()
                    .map(|(_, why)| why.as_str())
                    .collect::<Vec<_>>()
                    .join("; ")
            ));
        }
        llm::rank(text, &candidates, excluded, &brief)
    }
}

/// Forget any provisioned state and re-check on the next ranking.
pub fn reset_classifiers() {
    llm::reset();
}

/// Stop the local model because the app is closing, so no `llama-completion` outlives the
/// window with the weights still mapped. See [`llm::shutdown`].
///
/// Blocks until the processes are gone — milliseconds — and does nothing when none are
/// running or when the build has no local inference.
pub fn shutdown_model() {
    llm::shutdown();
}

/// Stop anything still running the build the user has just switched away from, so the two
/// builds' weights are never resident at the same time. See [`llm::unload_other_builds`].
///
/// Call it after publishing the new preference. Returns how many runs were stopped, which is
/// normally zero — nothing is resident between calls.
pub fn unload_other_model_builds() -> usize {
    llm::unload_other_builds()
}

/// Check the model and runtime are ready now rather than on the next ranking.
///
/// Returns the reason when either is missing, so the Models window can show it. Blocking;
/// call from a worker thread.
pub fn preload_classifiers() -> Result<(), String> {
    llm::preload()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::detect::{Detected, Status, Unavailable};
    use crate::agents::registry;
    use std::path::PathBuf;

    /// A filter that gets in the way of nothing, so tests about enumeration are not also
    /// tests about policy.
    fn open_filter() -> Filter {
        Filter {
            block: Vec::new(),
            allow: Vec::new(),
            efforts: Vec::new(),
            block_metered: false,
        }
    }

    fn detected(id: &str, status: Status) -> Detected {
        detected_with(id, status, None)
    }

    /// A detected agent whose own config names `model`, the way
    /// [`crate::agents::configured`] would have found it.
    fn detected_with(id: &str, status: Status, model: Option<&str>) -> Detected {
        let spec = registry::AGENTS
            .iter()
            .find(|a| a.id == id)
            .unwrap_or_else(|| panic!("no agent {id} in the registry"));
        Detected {
            spec,
            status,
            path: PathBuf::from("/usr/bin/x"),
            version: None,
            has_credentials: true,
            configured_model: model.map(str::to_string),
            models: Vec::new(),
        }
    }

    /// The list handed to the model has to be exactly what pstore can launch: every entry
    /// must carry real launch parameters, because the model picks by index and pstore then
    /// runs whatever that index pointed at.
    #[test]
    fn candidates_cover_only_runnable_combinations() {
        let agents = [detected("claude", Status::Ready)];
        let (cands, excluded) = candidates(&agents, &open_filter());

        assert!(excluded.is_empty());
        assert!(!cands.is_empty(), "a ready agent should offer combinations");

        let spec = agents[0].spec;
        let expected = spec.scoreable_models().len() * spec.scoreable_efforts().len();
        assert_eq!(cands.len(), expected, "every model × effort pair, once");

        for c in &cands {
            assert_eq!(c.agent_id, "claude");
            assert!(
                spec.scoreable_efforts().contains(&c.effort),
                "{:?} is not an effort this agent can be asked for",
                c.effort
            );
            assert!(
                spec.scoreable_models().iter().any(|m| m.id == c.model_id),
                "{} is not a model this agent exposes",
                c.model_id
            );
        }
    }

    /// An agent that cannot run must be reported, not silently dropped: "why isn't Codex
    /// in the list?" is otherwise unanswerable from the UI.
    #[test]
    fn unusable_agents_are_excluded_with_a_reason() {
        let agents = [
            detected("claude", Status::Ready),
            detected("codex", Status::Blocked(Unavailable::NotInstalled)),
        ];
        let (cands, excluded) = candidates(&agents, &open_filter());

        assert!(cands.iter().all(|c| c.agent_id == "claude"));
        assert_eq!(excluded.len(), 1);
        assert_eq!(excluded[0].0, "codex");
        assert!(!excluded[0].1.is_empty(), "an exclusion needs a reason");
    }

    /// An agent that chooses its own model gets that model's real name into the candidate list,
    /// so the ranker judges what will actually run rather than a placeholder.
    #[test]
    fn a_configured_model_reaches_the_candidate_list() {
        let agents = [detected_with(
            "crush",
            Status::Ready,
            Some("anthropic/claude-sonnet-4-5"),
        )];
        let (cands, excluded) = candidates(&agents, &open_filter());

        assert!(excluded.is_empty(), "got {excluded:?}");
        assert_eq!(cands.len(), 1, "one model, one effort");
        assert_eq!(cands[0].model_id, "anthropic/claude-sonnet-4-5");
        assert_eq!(
            cands[0].model_display, "anthropic/claude-sonnet-4-5",
            "the label has to be the name that is in the config, not a prettier invention"
        );
        assert!(
            !cands[0].effort_selectable,
            "pstore still cannot choose this agent's effort"
        );
    }

    /// An agent whose config says nothing still produces a candidate — a nameless one — so that
    /// the reason it is missing from the shortlist can be stated. It is [`withhold_unknown`]
    /// that keeps it out of the ranking, not silence here.
    #[test]
    fn an_agent_that_names_no_model_still_yields_a_nameless_candidate() {
        let agents = [detected_with("crush", Status::Ready, None)];
        let (cands, _) = candidates(&agents, &open_filter());
        assert_eq!(cands.len(), 1);
        assert!(cands[0].model_id.is_empty(), "got {:?}", cands[0].model_id);
    }

    /// The poisoning fix, stated as a property: a model nothing can describe does not reach the
    /// ranker, and the agent it belonged to is accounted for instead of vanishing.
    #[test]
    fn undescribed_models_are_withheld_and_accounted_for() {
        use crate::knowledge::{Brief, Known, Source};

        let agents = [
            detected("claude", Status::Ready),
            detected_with("crush", Status::Ready, None),
        ];
        let (mut cands, mut excluded) = candidates(&agents, &open_filter());
        let before = cands.len();
        assert!(
            cands.iter().any(|c| c.agent_id == "crush"),
            "the nameless candidate should be present before withholding"
        );

        // Everything Claude offers is described; the nameless one is not.
        let brief = Brief {
            known: cands
                .iter()
                .filter(|c| c.agent_id == "claude")
                .map(|c| Known {
                    model: c.model_id.to_string(),
                    note: "described".into(),
                    source: Source::Table,
                })
                .collect(),
            unknown: vec![(String::new(), "its config does not say which".into())],
        };
        withhold_unknown(&mut cands, &mut excluded, &brief);

        assert!(cands.len() < before, "nothing was withheld");
        assert!(
            cands.iter().all(|c| c.agent_id == "claude"),
            "an undescribed model reached the ranker"
        );
        let crush: Vec<_> = excluded.iter().filter(|(id, _)| *id == "crush").collect();
        assert_eq!(
            crush.len(),
            1,
            "one problem, stated once — not once per effort level: {excluded:?}"
        );
        assert!(crush[0].1.contains("does not say"), "got {:?}", crush[0].1);
    }

    /// Ranking with nothing to rank is a distinct failure from ranking without a model,
    /// and the message has to say which — otherwise the user goes looking for a download
    /// when the real problem is that no agent is installed.
    #[test]
    fn ranking_without_agents_says_so() {
        let why = rank("do a thing", &[], &open_filter()).expect_err("nothing to rank");
        assert!(
            why.contains("agent"),
            "the reason should name the missing agents, got {why:?}"
        );
    }

    #[test]
    fn best_and_fastest_read_the_shortlist() {
        let choice = |fit: f32, effort: Effort| Choice {
            agent_id: "claude",
            agent_display: "Claude Code",
            model_id: "m".into(),
            model_display: "M".into(),
            tier: Tier::Mid,
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
        };
        let r = Ranking {
            choices: vec![
                choice(90.0, Effort::Max),
                choice(86.0, Effort::Low),
                choice(50.0, Effort::Low),
            ],
            considered: 3,
            ..Ranking::default()
        };

        assert_eq!(r.best().map(|c| c.fit), Some(90.0));
        // Within tolerance, the faster effort wins even though it scored lower.
        assert_eq!(r.fastest_within(5.0).map(|c| c.effort), Some(Effort::Low));
        assert_eq!(r.fastest_within(5.0).map(|c| c.fit), Some(86.0));
        // Too tight a tolerance leaves only the best.
        assert_eq!(r.fastest_within(1.0).map(|c| c.effort), Some(Effort::Max));
        assert!(Ranking::default().best().is_none());
        assert!(Ranking::default().fastest_within(10.0).is_none());
    }

    /// Policy is applied before ranking, not after: a blocked model must never reach the
    /// list the local model is asked to choose from.
    #[test]
    fn filtered_models_are_never_offered() {
        let agents = [detected("claude", Status::Ready)];
        let (all, _) = candidates(&agents, &open_filter());
        assert!(
            all.iter().any(|c| c.metered),
            "this test needs a metered model in the registry to be meaningful"
        );

        let (kept, _) = candidates(&agents, &Filter::default());
        assert!(
            kept.iter().all(|c| !c.metered),
            "a per-token model reached the ranking list"
        );
        assert!(
            !kept.is_empty(),
            "the default filter should not empty the field"
        );
    }

    /// An effort whitelist has to narrow the field too, not just the model list.
    #[test]
    fn effort_filtering_narrows_the_field() {
        let agents = [detected("claude", Status::Ready)];
        let only_low = Filter {
            efforts: vec![Effort::Low],
            ..open_filter()
        };
        let (kept, _) = candidates(&agents, &only_low);
        assert!(!kept.is_empty());
        assert!(kept.iter().all(|c| c.effort == Effort::Low));
    }

    /// The line every front end prints above the shortlist has to carry both halves of the
    /// judgement and what they came to. Showing only the difficulty is what it used to do, and
    /// it made the effort in the table below unaccountable: there was nothing on screen that
    /// explained why one hard prompt got `high` and another got `max`.
    #[test]
    fn the_judgement_line_shows_both_axes_and_the_effort() {
        let j = Judgement {
            demand: "hard",
            breadth: "many files",
            effort: Effort::Max,
            because: "whole-crate async conversion".into(),
        };
        let line = j.summary();
        assert!(line.contains("hard"), "{line}");
        assert!(line.contains("many files"), "{line}");
        assert!(line.contains("max"), "{line}");

        // Every effort the table can reach has to survive into the line, or the one number the
        // user can act on is the one that goes missing.
        for effort in Effort::ALL {
            let j = Judgement {
                effort,
                ..j.clone()
            };
            assert!(
                j.summary().contains(effort.as_str()),
                "{effort:?} did not reach the line: {}",
                j.summary()
            );
        }
    }

    /// `because` is an optional field of the model's reply, so all three front ends have to
    /// cope with it being absent. Appending the separator regardless leaves a dangling dash on
    /// the end of the line, which reads as pstore having truncated its own explanation.
    #[test]
    fn a_judgement_with_no_reason_carries_no_separator() {
        let j = Judgement {
            demand: "easy",
            breadth: "one edit",
            effort: Effort::Low,
            because: String::new(),
        };
        assert_eq!(j.because_suffix(), "", "an absent reason is silence");

        // Whitespace is absent too — a model that answered with a space should not produce a
        // separator with nothing after it either.
        let blank = Judgement {
            because: "   ".into(),
            ..j.clone()
        };
        assert_eq!(blank.because_suffix(), "");

        let given = Judgement {
            because: "one file, one variable".into(),
            ..j.clone()
        };
        assert_eq!(given.because_suffix(), " — one file, one variable");
    }

    /// "Nothing installed" and "you excluded everything" need different answers — the
    /// second one is fixed in a config file, not by installing an agent.
    #[test]
    fn an_over_tight_filter_says_it_was_the_filter() {
        let agents = [detected("claude", Status::Ready)];
        let nothing = Filter {
            allow: vec!["not-a-real-model".into()],
            ..open_filter()
        };
        let (kept, excluded) = candidates(&agents, &nothing);
        assert!(kept.is_empty());
        assert!(
            excluded
                .iter()
                .any(|(id, why)| *id == "claude" && why.contains("filtered out")),
            "got {excluded:?}"
        );

        let why = rank("do a thing", &agents, &nothing).expect_err("nothing to rank");
        assert!(
            why.contains("filter"),
            "the reason should point at the config, got {why:?}"
        );
    }
}
