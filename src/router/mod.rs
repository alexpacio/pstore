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

pub mod hub;
#[cfg(feature = "local-llm")]
pub mod llm;

use crate::agents::detect::Detected;
use crate::agents::registry::{Effort, Tier};
use crate::filter::Filter;

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
    pub model_id: &'static str,
    /// Model label for the UI.
    pub model_display: &'static str,
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
    /// Relative token price. **Display only.**
    pub relative_price: f32,
    /// How well the model judged this fits, `0..=100`.
    pub fit: f32,
    /// The model's one-line reason for the placement.
    pub rationale: String,
}

/// A ranked shortlist over the detected agents.
#[derive(Debug, Clone, Default)]
pub struct Ranking {
    /// Choices, best first. At most [`SHORTLIST`] of them.
    pub choices: Vec<Choice>,
    /// How many combinations were offered to the model.
    pub considered: usize,
    /// Agents that were detected but excluded, with the reason.
    pub excluded: Vec<(&'static str, String)>,
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
    pub model_id: &'static str,
    pub model_display: &'static str,
    pub tier: Tier,
    pub effort: Effort,
    pub effort_selectable: bool,
    pub metered: bool,
    pub relative_price: f32,
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

        for model in agent.spec.scoreable_models() {
            if !filter.allows_model(agent.spec.id, model.id, model.display, model.metered) {
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
                    model_id: model.id,
                    model_display: model.display,
                    tier: model.tier,
                    effort,
                    effort_selectable: selectable,
                    metered: model.metered,
                    relative_price: model.relative_price,
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
    let (candidates, excluded) = candidates(detected, filter);
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

    #[cfg(feature = "local-llm")]
    {
        llm::rank(text, &candidates, excluded)
    }
    #[cfg(not(feature = "local-llm"))]
    {
        let _ = (text, excluded);
        Err(crate::models::NO_LOCAL_INFERENCE.to_string())
    }
}

/// Forget any provisioned state and re-check on the next ranking.
pub fn reset_classifiers() {
    #[cfg(feature = "local-llm")]
    llm::reset();
}

/// Stop the local model because the app is closing, so no `llama-completion` outlives the
/// window with the weights still mapped. See [`llm::shutdown`].
///
/// Blocks until the processes are gone — milliseconds — and does nothing when none are
/// running or when the build has no local inference.
pub fn shutdown_model() {
    #[cfg(feature = "local-llm")]
    llm::shutdown();
}

/// Stop anything still running the build the user has just switched away from, so the two
/// builds' weights are never resident at the same time. See [`llm::unload_other_builds`].
///
/// Call it after publishing the new preference. Returns how many runs were stopped, which is
/// normally zero — nothing is resident between calls.
pub fn unload_other_model_builds() -> usize {
    #[cfg(feature = "local-llm")]
    {
        llm::unload_other_builds()
    }
    #[cfg(not(feature = "local-llm"))]
    {
        0
    }
}

/// Check the model and runtime are ready now rather than on the next ranking.
///
/// Returns the reason when either is missing, so the Models window can show it. Blocking;
/// call from a worker thread.
pub fn preload_classifiers() -> Result<(), String> {
    #[cfg(feature = "local-llm")]
    {
        llm::preload()
    }
    #[cfg(not(feature = "local-llm"))]
    {
        Err(crate::models::NO_LOCAL_INFERENCE.to_string())
    }
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
            model_id: "m",
            model_display: "M",
            tier: Tier::Mid,
            effort,
            effort_selectable: true,
            metered: false,
            relative_latency: effort.latency_factor(),
            relative_price: 1.0,
            fit,
            rationale: String::new(),
        };
        let r = Ranking {
            choices: vec![
                choice(90.0, Effort::Max),
                choice(86.0, Effort::Low),
                choice(50.0, Effort::Low),
            ],
            considered: 3,
            excluded: Vec::new(),
            elapsed: std::time::Duration::ZERO,
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

    /// Without the model there is no ranking at all — not a worse ranking. This is the
    /// property the whole no-fallback design rests on, so it is asserted rather than
    /// assumed.
    #[test]
    #[cfg(not(feature = "local-llm"))]
    fn a_build_without_inference_refuses_rather_than_guessing() {
        let agents = [detected("claude", Status::Ready)];
        let why = rank("Refactor src/main.rs", &agents, &open_filter())
            .expect_err("no local inference should mean no ranking");
        assert!(
            why.contains("local-llm"),
            "the reason should name what is missing, got {why:?}"
        );
    }
}
