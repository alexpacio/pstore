//! Scoring every available (agent, model, effort) combination against a prompt.
//!
//! pstore **reports fit, it does not shop for bargains.** The score answers one
//! question — "how well does this model, at this effort, match what the prompt
//! actually demands?" — and every candidate gets one, so the developer can see the
//! whole field and decide. Token price is displayed alongside the score and is
//! deliberately excluded from the ranking; see
//! [`tests::price_does_not_influence_ranking`].
//!
//! The fit term follows Brick's spatial-routing idea (distance in a 6-dim capability
//! space) with one deliberate change: only *shortfalls* count. A model stronger than
//! the prompt needs is not penalised, because over-capacity is only a cost problem
//! and cost is not ours to optimise.
//!
//! # The one exception: metered models
//!
//! Ignoring price works because the models are already paid for — picking a stronger one
//! than necessary wastes nothing the developer has not already spent. A model billed *per
//! token* breaks that assumption: choosing it spends new money. Since a frontier metered
//! model also tends to dominate its included siblings on every dimension, pure fit would
//! hand it every hard prompt, and — ties breaking alphabetically — a fair number of easy
//! ones too.
//!
//! So metered candidates are **held back**: they sort below every included candidate unless
//! they fit better than the best included one by at least [`METERED_MARGIN`] points. They
//! are never hidden, and they are never excluded — when nothing included is adequate, the
//! margin is met and the metered model wins on merit. See
//! [`tests::a_metered_model_is_held_back_until_it_is_clearly_needed`].

use crate::agents::detect::Detected;
use crate::agents::registry::{DIMS, Effort, ModelSpec, Tier, Vec6};

use super::{Capability, Complexity};

/// How many points of extra fit a metered model must offer before it outranks an
/// included one.
///
/// Sized against the gap it has to be able to cross: Fable's skill vector sits 0.02–0.03
/// above Opus's, which is worth two or three points when both are comfortable — so a margin
/// of five keeps it out of the way on ordinary work. It becomes reachable only when the
/// included models genuinely fall short, which is when the extra spend is the point.
pub const METERED_MARGIN: f32 = 5.0;

/// One scored (agent, model, effort) combination.
#[derive(Debug, Clone)]
pub struct Candidate {
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
    /// Overall fit, `0..=100`. Higher is better.
    pub score: f32,
    /// Per-dimension shortfall (`demand - reach`, clamped at zero), in [`DIMS`] order.
    pub shortfall: Vec6,
    /// Relative time-to-answer, `1.0` being the fastest candidate in the field.
    pub relative_latency: f32,
    /// Relative token price. **Display only** — never part of `score`.
    pub relative_price: f32,
    /// Whether this model is billed per token rather than covered by the subscription.
    pub metered: bool,
    /// Whether this candidate was held back for being metered when an included model was
    /// good enough.
    ///
    /// Held-back candidates keep their real `score` — the number is still the honest answer
    /// to "how well does this fit?" — but sort below everything included, and are skipped by
    /// [`Ranking::best`] and [`Ranking::fastest_within`].
    pub held_back: bool,
}

impl Candidate {
    /// The dimension this candidate is weakest on, when it falls short anywhere.
    pub fn weakest_dimension(&self) -> Option<&'static str> {
        let (idx, worst) =
            self.shortfall
                .iter()
                .enumerate()
                .fold(
                    (0usize, 0.0f32),
                    |acc, (i, v)| if *v > acc.1 { (i, *v) } else { acc },
                );
        (worst > 0.01).then_some(DIMS[idx])
    }

    /// One-line explanation for the ranking table.
    pub fn rationale(&self) -> String {
        let fit = match self.weakest_dimension() {
            Some(dim) => format!("{:.0}/100 — light on {dim}", self.score),
            None => format!("{:.0}/100 — covers every dimension", self.score),
        };
        // A held-back candidate can sit at the bottom of the table with the highest score
        // in it. Say why, or the ranking looks broken.
        if self.held_back {
            format!("{fit} · billed per token, held back")
        } else if self.metered {
            format!("{fit} · billed per token")
        } else {
            fit
        }
    }
}

/// A full ranking over the detected agents.
#[derive(Debug, Clone, Default)]
pub struct Ranking {
    /// Candidates, best first.
    pub candidates: Vec<Candidate>,
    /// Agents that were detected but excluded, with the reason.
    pub excluded: Vec<(&'static str, String)>,
    /// The capability demand the scores were computed against.
    pub demand: Vec6,
    /// Complexity the classifier reported.
    pub complexity: Complexity,
}

impl Ranking {
    /// The candidate pstore would actually use.
    ///
    /// Held-back candidates are skipped rather than merely sorted low, so a metered model
    /// can never become the automatic pick through a sort-order accident.
    pub fn best(&self) -> Option<&Candidate> {
        self.candidates.iter().find(|c| !c.held_back)
    }

    /// The fastest candidate scoring within `tolerance` points of the best.
    ///
    /// This is the hint path: the user asked for speed on hints specifically, and
    /// latency is a real-time property, not a price. Held-back candidates are excluded —
    /// a hint is the last place to start metered spending.
    pub fn fastest_within(&self, tolerance: f32) -> Option<&Candidate> {
        let best = self.best()?.score;
        self.candidates
            .iter()
            .filter(|c| !c.held_back && c.score + tolerance >= best)
            .min_by(|a, b| a.relative_latency.total_cmp(&b.relative_latency))
    }
}

/// How much capability the prompt demands, per dimension.
///
/// The classifier says *which* capabilities the prompt draws on; complexity says how
/// hard it leans on them. Multiplying gives the bar a candidate has to clear.
fn demand_vector(cap: &Capability, complexity: Complexity) -> Vec6 {
    let ceiling = match complexity {
        Complexity::Easy => 0.55,
        Complexity::Medium => 0.80,
        Complexity::Hard => 0.97,
    };
    let mut out = [0.0f32; 6];
    for (slot, score) in out.iter_mut().zip(cap.scores) {
        *slot = (score * ceiling).clamp(0.0, 1.0);
    }
    out
}

/// What a model actually reaches at a given effort.
fn reach(model: &ModelSpec, effort: Effort) -> Vec6 {
    let h = effort.headroom();
    let mut out = [0.0f32; 6];
    for (slot, ceiling) in out.iter_mut().zip(model.skill) {
        *slot = ceiling * h;
    }
    out
}

/// Score one combination against `demand`.
///
/// Returns `(score, shortfall)`. Only dimensions where the candidate falls short of
/// the demand reduce the score, weighted by how much the prompt leans on them.
fn score_against(demand: &Vec6, reach: &Vec6, cap: &Capability) -> (f32, Vec6) {
    let mut shortfall = [0.0f32; 6];
    let mut penalty = 0.0f32;
    let mut weight_total = 0.0f32;

    for i in 0..6 {
        let gap = (demand[i] - reach[i]).max(0.0);
        shortfall[i] = gap;
        // A gap on a dimension the prompt barely touches matters less than a gap on
        // its dominant one, so weight by the raw capability score.
        let w = cap.scores[i].max(0.02);
        penalty += w * gap * gap;
        weight_total += w;
    }

    let normalised = if weight_total > 0.0 {
        (penalty / weight_total).sqrt()
    } else {
        0.0
    };
    // A full-scale miss on the dominant dimension is a gap of ~1.0, so mapping
    // [0, 1] onto [100, 0] keeps the scale readable.
    let score = ((1.0 - normalised) * 100.0).clamp(0.0, 100.0);
    (score, shortfall)
}

/// Mark metered candidates that an included model already covers.
///
/// The comparison is against the best *included* fit in the whole field, not against the
/// metered candidate's own agent: if Codex or Gemini handles the prompt, that is still a
/// reason not to start billing per token.
///
/// With no included candidate at all — a machine where the only usable model is a metered
/// one — nothing is held back, because holding everything back would leave the ranking empty
/// and the developer with no pick rather than an informed one.
fn hold_back_metered(candidates: &mut [Candidate]) {
    let best_included = candidates
        .iter()
        .filter(|c| !c.metered)
        .map(|c| c.score)
        .fold(None::<f32>, |acc, v| Some(acc.map_or(v, |a: f32| a.max(v))));

    let Some(bar) = best_included else {
        return;
    };
    for c in candidates.iter_mut().filter(|c| c.metered) {
        c.held_back = c.score <= bar + METERED_MARGIN;
    }
}

/// Score every (model, effort) pair on every usable detected agent.
pub fn rank(detected: &[Detected], cap: &Capability, complexity: Complexity) -> Ranking {
    let demand = demand_vector(cap, complexity);
    let mut candidates = Vec::new();
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
        for model in agent.spec.scoreable_models() {
            for &effort in agent.spec.scoreable_efforts() {
                let r = reach(model, effort);
                let (score, shortfall) = score_against(&demand, &r, cap);
                candidates.push(Candidate {
                    agent_id: agent.spec.id,
                    agent_display: agent.spec.display,
                    model_id: model.id,
                    model_display: model.display,
                    tier: model.tier,
                    effort,
                    effort_selectable: selectable,
                    score,
                    shortfall,
                    relative_latency: effort.latency_factor(),
                    relative_price: model.relative_price,
                    metered: model.metered,
                    held_back: false,
                });
            }
        }
    }

    hold_back_metered(&mut candidates);

    // Included candidates first, then best score, then the faster one. A metered candidate
    // loses every remaining tie to an included one — otherwise `model_id` would decide it
    // alphabetically, which is how "fable" used to beat "opus" on identical scores.
    candidates.sort_by(|a, b| {
        a.held_back
            .cmp(&b.held_back)
            .then_with(|| b.score.total_cmp(&a.score))
            .then_with(|| a.relative_latency.total_cmp(&b.relative_latency))
            .then_with(|| a.metered.cmp(&b.metered))
            .then_with(|| a.agent_id.cmp(b.agent_id))
            .then_with(|| a.model_id.cmp(b.model_id))
    });

    // Normalise latency against the fastest candidate present.
    if let Some(min) = candidates
        .iter()
        .map(|c| c.relative_latency)
        .fold(None::<f32>, |acc, v| Some(acc.map_or(v, |a: f32| a.min(v))))
        && min > 0.0
    {
        for c in &mut candidates {
            c.relative_latency /= min;
        }
    }

    Ranking {
        candidates,
        excluded,
        demand,
        complexity,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::detect::{Detected, Status, Unavailable};
    use crate::agents::registry;
    use std::path::PathBuf;

    fn detected(id: &str, status: Status) -> Detected {
        Detected {
            spec: registry::find(id).unwrap(),
            path: PathBuf::from(format!("/usr/bin/{id}")),
            version: None,
            has_credentials: true,
            status,
        }
    }

    /// A coding-heavy prompt.
    fn coding_cap() -> Capability {
        // [instruction_following, coding, math, world, planning, creative]
        Capability {
            scores: [0.7, 0.95, 0.3, 0.2, 0.6, 0.1],
        }
    }

    #[test]
    fn every_model_and_effort_gets_a_score() {
        let agents = vec![detected("claude", Status::Verified)];
        let r = rank(&agents, &coding_cap(), Complexity::Medium);
        // Claude: 4 models x 5 effort levels.
        assert_eq!(
            r.candidates.len(),
            20,
            "each model/effort pair must be scored"
        );
        assert!(
            r.candidates
                .iter()
                .all(|c| (0.0..=100.0).contains(&c.score))
        );
    }

    /// Fable is the only Claude model billed per token, and its skill vector dominates
    /// Opus on every dimension — so pure fit picked it for anything hard, and the
    /// alphabetical tie-break ("fable" < "opus") picked it for plenty that wasn't.
    #[test]
    fn a_metered_model_is_held_back_until_it_is_clearly_needed() {
        let agents = vec![detected("claude", Status::Verified)];

        for complexity in [Complexity::Easy, Complexity::Medium, Complexity::Hard] {
            let r = rank(&agents, &coding_cap(), complexity);
            let best = r.best().expect("a pick");
            assert_ne!(
                best.model_id, "fable",
                "{complexity} picked the metered model: {} at {}",
                best.model_display, best.effort
            );
            assert!(
                !best.metered,
                "the automatic pick must be an included model"
            );

            // Held back, not hidden: still listed, still scored honestly.
            let fable: Vec<_> = r
                .candidates
                .iter()
                .filter(|c| c.model_id == "fable")
                .collect();
            assert_eq!(fable.len(), 5, "every Fable effort must still be listed");
            assert!(
                fable.iter().all(|c| c.metered),
                "Fable must be marked metered"
            );
            assert!(
                fable.iter().any(|c| c.score > 90.0),
                "{complexity}: its real fit must be reported, not zeroed"
            );
            // And the table explains why a 99/100 candidate is at the bottom.
            let top_fable = fable
                .iter()
                .max_by(|a, b| a.score.total_cmp(&b.score))
                .unwrap();
            if top_fable.held_back {
                assert!(
                    top_fable.rationale().contains("held back"),
                    "got {:?}",
                    top_fable.rationale()
                );
            }

            // Held-back candidates sort after every included one.
            let first_held = r.candidates.iter().position(|c| c.held_back);
            let last_open = r.candidates.iter().rposition(|c| !c.held_back);
            if let (Some(held), Some(open)) = (first_held, last_open) {
                assert!(held > open, "held-back candidates must sort last");
            }
        }
    }

    /// The exact situation that exposed this: the capability vector the real classifier
    /// produced for a three-file refactor, at `Hard`. Opus at Max covers every dimension,
    /// so both it and Fable score 100 — and a tie used to be settled by model id, which
    /// spells "fable" before "opus".
    #[test]
    fn a_saturating_hard_prompt_still_picks_the_included_model() {
        let measured = Capability {
            scores: [
                0.46354946,
                0.9593354,
                0.033743963,
                0.13441071,
                0.59820795,
                0.03761472,
            ],
        };
        let agents = vec![
            detected("claude", Status::Verified),
            detected("codex", Status::Verified),
        ];
        let r = rank(&agents, &measured, Complexity::Hard);
        let best = r.best().unwrap();

        assert_eq!(
            (best.agent_id, best.model_id),
            ("claude", "opus"),
            "got {} {} at {}",
            best.agent_display,
            best.model_display,
            best.effort
        );
        assert_eq!(best.score, 100.0, "Opus covers this prompt outright");
        // Fable scores just as well, and is still held back — the tie is not a need.
        let fable_top = r
            .candidates
            .iter()
            .filter(|c| c.model_id == "fable")
            .max_by(|a, b| a.score.total_cmp(&b.score))
            .unwrap();
        assert_eq!(fable_top.score, 100.0);
        assert!(
            fable_top.held_back,
            "a tie must not unlock metered spending"
        );
    }

    /// The hint path is the last place to start metered spending.
    #[test]
    fn the_fastest_candidate_is_never_a_metered_one() {
        let agents = vec![detected("claude", Status::Verified)];
        for complexity in [Complexity::Easy, Complexity::Medium, Complexity::Hard] {
            let r = rank(&agents, &coding_cap(), complexity);
            for tolerance in [0.0, 8.0, 100.0] {
                let quick = r.fastest_within(tolerance).expect("a quick pick");
                assert!(
                    !quick.held_back,
                    "{complexity} at tolerance {tolerance} chose a held-back model: {}",
                    quick.model_display
                );
            }
        }
    }

    /// When an included model would do, the metered one must lose the tie — this is the
    /// exact regression: identical scores used to be broken by model id, alphabetically.
    #[test]
    fn an_identical_score_goes_to_the_included_model() {
        let mut candidates = vec![
            Candidate {
                metered: true,
                model_id: "fable",
                ..candidate(100.0, false)
            },
            Candidate {
                model_id: "opus",
                ..candidate(100.0, false)
            },
        ];
        hold_back_metered(&mut candidates);
        candidates.sort_by(|a, b| {
            a.held_back
                .cmp(&b.held_back)
                .then_with(|| b.score.total_cmp(&a.score))
                .then_with(|| a.metered.cmp(&b.metered))
                .then_with(|| a.model_id.cmp(b.model_id))
        });
        assert_eq!(candidates[0].model_id, "opus");
    }

    #[test]
    fn a_metered_model_wins_when_it_is_clearly_better() {
        // Nothing included comes close: the margin is met, so the spend is the point.
        let mut candidates = vec![
            Candidate {
                metered: true,
                model_id: "fable",
                ..candidate(95.0, false)
            },
            Candidate {
                model_id: "opus",
                ..candidate(80.0, false)
            },
        ];
        hold_back_metered(&mut candidates);
        assert!(
            !candidates[0].held_back,
            "a {}-point lead should clear the {METERED_MARGIN}-point margin",
            95.0 - 80.0
        );

        // Just inside the margin is not "clearly better".
        let mut close = vec![
            Candidate {
                metered: true,
                model_id: "fable",
                ..candidate(84.0, false)
            },
            Candidate {
                model_id: "opus",
                ..candidate(80.0, false)
            },
        ];
        hold_back_metered(&mut close);
        assert!(close[0].held_back, "4 points is not a clear need");
    }

    #[test]
    fn a_metered_model_is_not_held_back_when_it_is_the_only_option() {
        // Holding back the only candidate would leave the developer with no pick at all,
        // which is worse than an informed one.
        let mut only = vec![Candidate {
            metered: true,
            model_id: "fable",
            ..candidate(70.0, false)
        }];
        hold_back_metered(&mut only);
        assert!(!only[0].held_back);
    }

    /// A minimal candidate for the hold-back unit tests.
    fn candidate(score: f32, metered: bool) -> Candidate {
        Candidate {
            agent_id: "claude",
            agent_display: "Claude Code",
            model_id: "m",
            model_display: "M",
            tier: registry::Tier::Top,
            effort: Effort::High,
            effort_selectable: true,
            score,
            shortfall: [0.0; 6],
            relative_latency: 1.0,
            relative_price: 1.0,
            metered,
            held_back: false,
        }
    }

    #[test]
    fn price_does_not_influence_ranking() {
        // The load-bearing guarantee: perturbing price must not reorder anything.
        let agents = vec![detected("claude", Status::Verified)];
        let cap = coding_cap();
        let r = rank(&agents, &cap, Complexity::Hard);
        let order: Vec<_> = r
            .candidates
            .iter()
            .map(|c| (c.model_id, c.effort, c.score.to_bits()))
            .collect();

        // The cheapest model must be able to outrank the dearest when it fits better.
        let cheapest_top = r
            .candidates
            .iter()
            .filter(|c| c.model_id == "haiku")
            .map(|c| c.score)
            .fold(f32::MIN, f32::max);
        let priciest_low = r
            .candidates
            .iter()
            .find(|c| c.model_id == "fable" && c.effort == Effort::Low)
            .unwrap()
            .score;
        assert!(
            cheapest_top > 0.0 && priciest_low > 0.0,
            "both extremes must be scored, not filtered by price"
        );

        // Scores are a pure function of capability and effort — recomputing with the
        // same inputs is identical, and price never appears in the computation.
        let again = rank(&agents, &cap, Complexity::Hard);
        let order2: Vec<_> = again
            .candidates
            .iter()
            .map(|c| (c.model_id, c.effort, c.score.to_bits()))
            .collect();
        assert_eq!(order, order2);

        // And a dearer model does not automatically rank above a cheaper one.
        let by_price_would_be: Vec<_> = {
            let mut v = r.candidates.clone();
            v.sort_by(|a, b| a.relative_price.total_cmp(&b.relative_price));
            v.iter().map(|c| (c.model_id, c.effort)).collect()
        };
        let actual: Vec<_> = r
            .candidates
            .iter()
            .map(|c| (c.model_id, c.effort))
            .collect();
        assert_ne!(
            actual, by_price_would_be,
            "ranking must not be a price sort"
        );
    }

    #[test]
    fn higher_effort_never_scores_worse() {
        let agents = vec![detected("claude", Status::Verified)];
        let r = rank(&agents, &coding_cap(), Complexity::Hard);
        for model in ["haiku", "sonnet", "opus", "fable"] {
            let mut by_effort: Vec<_> = r
                .candidates
                .iter()
                .filter(|c| c.model_id == model)
                .map(|c| (c.effort, c.score))
                .collect();
            by_effort.sort_by_key(|(e, _)| *e);
            for w in by_effort.windows(2) {
                assert!(
                    w[1].1 >= w[0].1 - f32::EPSILON,
                    "{model}: {:?} scored below {:?}",
                    w[1].0,
                    w[0].0
                );
            }
        }
    }

    #[test]
    fn harder_prompts_lower_the_scores_of_weak_candidates() {
        let agents = vec![detected("claude", Status::Verified)];
        let cap = coding_cap();
        let easy = rank(&agents, &cap, Complexity::Easy);
        let hard = rank(&agents, &cap, Complexity::Hard);

        let pick = |r: &Ranking| {
            r.candidates
                .iter()
                .find(|c| c.model_id == "haiku" && c.effort == Effort::Low)
                .unwrap()
                .score
        };
        assert!(
            pick(&hard) < pick(&easy),
            "a weak candidate must score worse on a hard prompt"
        );

        // On an easy prompt, even the light model should comfortably clear the bar.
        assert!(
            pick(&easy) > 90.0,
            "easy prompts are well served by light models"
        );
    }

    #[test]
    fn strong_candidates_score_full_marks_on_easy_prompts() {
        let agents = vec![detected("claude", Status::Verified)];
        let r = rank(&agents, &coding_cap(), Complexity::Easy);
        let best = r.best().unwrap();
        assert_eq!(best.score, 100.0, "no shortfall must mean a perfect score");
        assert_eq!(best.weakest_dimension(), None);
        assert!(best.rationale().contains("covers every dimension"));
    }

    #[test]
    fn overshoot_is_not_penalised() {
        // A frontier model on a trivial prompt must not be marked down — that would
        // be a cost judgement, which is not ours to make.
        let agents = vec![detected("claude", Status::Verified)];
        let r = rank(&agents, &coding_cap(), Complexity::Easy);
        let fable_max = r
            .candidates
            .iter()
            .find(|c| c.model_id == "fable" && c.effort == Effort::Max)
            .unwrap();
        assert_eq!(fable_max.score, 100.0);
    }

    #[test]
    fn shortfall_names_the_weak_dimension() {
        let agents = vec![detected("claude", Status::Verified)];
        // A math-dominant prompt against a model weak at math.
        let cap = Capability {
            scores: [0.3, 0.2, 0.98, 0.2, 0.2, 0.1],
        };
        let r = rank(&agents, &cap, Complexity::Hard);
        let haiku_low = r
            .candidates
            .iter()
            .find(|c| c.model_id == "haiku" && c.effort == Effort::Low)
            .unwrap();
        assert_eq!(haiku_low.weakest_dimension(), Some("math_reasoning"));
        assert!(haiku_low.rationale().contains("math_reasoning"));
    }

    #[test]
    fn blocked_agents_are_excluded_with_a_reason() {
        let agents = vec![
            detected(
                "claude",
                Status::Blocked(Unavailable::NotLoggedIn("run /login".into())),
            ),
            detected("codex", Status::Verified),
        ];
        let r = rank(&agents, &coding_cap(), Complexity::Medium);
        assert!(r.candidates.iter().all(|c| c.agent_id == "codex"));
        assert_eq!(r.excluded.len(), 1);
        assert_eq!(r.excluded[0].0, "claude");
        assert!(r.excluded[0].1.contains("not logged in"));
    }

    #[test]
    fn agents_without_knobs_still_appear() {
        // crush can set neither model nor effort, but must still be rankable.
        let agents = vec![detected("crush", Status::Verified)];
        let r = rank(&agents, &coding_cap(), Complexity::Medium);
        assert_eq!(r.candidates.len(), 1);
        let c = &r.candidates[0];
        assert_eq!(c.model_display, "(agent default)");
        assert!(
            !c.effort_selectable,
            "must be flagged as a prediction, not a choice"
        );
        assert!(c.score > 0.0);
    }

    #[test]
    fn fastest_within_tolerance_prefers_low_latency() {
        let agents = vec![detected("claude", Status::Verified)];
        // Easy prompt: everything fits, so the tie-break is pure speed.
        let r = rank(&agents, &coding_cap(), Complexity::Easy);
        let quick = r.fastest_within(1.0).unwrap();
        assert_eq!(
            quick.effort,
            Effort::Low,
            "hints should take the fastest adequate option"
        );
        assert_eq!(quick.relative_latency, 1.0);
    }

    #[test]
    fn latency_is_normalised_to_the_fastest_candidate() {
        let agents = vec![detected("claude", Status::Verified)];
        let r = rank(&agents, &coding_cap(), Complexity::Medium);
        let min = r
            .candidates
            .iter()
            .map(|c| c.relative_latency)
            .fold(f32::MAX, f32::min);
        assert!(
            (min - 1.0).abs() < 1e-6,
            "fastest candidate should read 1.0, got {min}"
        );
    }

    #[test]
    fn empty_detection_yields_an_empty_ranking() {
        let r = rank(&[], &coding_cap(), Complexity::Medium);
        assert!(r.candidates.is_empty());
        assert!(r.best().is_none());
        assert!(r.fastest_within(5.0).is_none());
    }

    #[test]
    fn ranking_is_deterministic_across_agent_order() {
        let a = vec![
            detected("claude", Status::Verified),
            detected("codex", Status::Verified),
        ];
        let b = vec![
            detected("codex", Status::Verified),
            detected("claude", Status::Verified),
        ];
        let ra = rank(&a, &coding_cap(), Complexity::Hard);
        let rb = rank(&b, &coding_cap(), Complexity::Hard);
        let key = |r: &Ranking| -> Vec<_> {
            r.candidates
                .iter()
                .map(|c| (c.agent_id, c.model_id, c.effort))
                .collect()
        };
        assert_eq!(
            key(&ra),
            key(&rb),
            "input order must not change the ranking"
        );
    }
}
