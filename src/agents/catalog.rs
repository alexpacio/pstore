//! Model catalogs an agent publishes for itself.
//!
//! [`super::registry`] carries a hand-written model table per agent, and that table is wrong the
//! day a vendor ships. The `GPT-5.1 Codex` row sat there while Codex had moved on to a whole
//! family — sol, terra, luna — and pstore had no way to know, because nothing ever asked Codex.
//!
//! Some agents already keep the answer on disk. Codex writes the catalog it fetched from OpenAI
//! into `~/.codex/models_cache.json`: every model it will accept for `-m`, with a display name, a
//! vendor description, a visibility flag, and a priority ordering. Reading that is strictly better
//! than a table maintained here — it is what the agent itself believes, updated when the agent
//! updates.
//!
//! **This is discovery, like [`super::configured`], and it fails the same way.** The file is a
//! cache the vendor owns and may reshape without warning, so every step is fallible and a failure
//! returns nothing rather than a guess. An empty result leaves the registry table in charge, which
//! is the old behaviour — stale, but never invented.

use std::path::Path;

use super::registry::{ModelSpec, Tier};

/// One model an agent's own catalog says it can run.
///
/// The owned counterpart to [`ModelSpec`], which is `&'static` because the registry is a compile
/// time table. A discovered model cannot be, so the two are separate types rather than one made
/// generic over its lifetime.
#[derive(Debug, Clone, PartialEq)]
pub struct CatalogModel {
    /// Value to pass to the agent's model flag.
    pub id: String,
    /// Human label, as the vendor writes it.
    pub display: String,
    /// The vendor's own one-line description.
    ///
    /// Load-bearing rather than decorative: a model pstore cannot describe is withheld from
    /// ranking entirely ([`crate::knowledge`]), so a freshly-shipped model with no entry in
    /// pstore's table would be discovered and then immediately excluded. The vendor's description
    /// is what keeps it rankable — and it is a better source than a guess here, because the
    /// vendor is describing a model it actually built.
    pub note: String,
    /// Weight class. Informational, never scored.
    pub tier: Tier,
    /// Relative price, on the same scale as [`ModelSpec::relative_price`].
    pub relative_price: f32,
    /// How fast this model spends the subscription's allowance, relative to the lightest model
    /// the same vendor offers. See [`quota_weight_for`].
    pub quota_weight: f32,
}

/// How fast each Codex model burns the ChatGPT plan's Codex allowance, relative to the lightest.
///
/// **Derived from OpenAI's published Codex rate card**, output-token credits per million:
/// Sol 750, GPT-5.5 750, GPT-5.4 375, Terra 300, GPT-5.4-mini 113, Luna 30 — normalised so Luna
/// is `1.0`. Output rates rather than input because output dominates an agentic coding run, and
/// because the input/output ratio is near-constant across the family, so either choice gives the
/// same ordering.
///
/// This is a different quantity from [`CatalogModel::relative_price`]. Price is dollars per token
/// on the API and is never scored. This is how quickly a model consumes an allowance the developer
/// has *already paid for* — the thing that decides whether they can keep working this week. Every
/// model here costs the same zero dollars extra; they do not cost the same quota.
///
/// A slug with no published rate falls back to its tier, because the alternative — treating an
/// unrecognised model as free — would route work to it precisely because pstore knows nothing
/// about it.
const CODEX_QUOTA_WEIGHTS: &[(&str, f32)] = &[
    ("gpt-5.6-sol", 25.0),
    ("gpt-5.5", 25.0),
    ("gpt-5.4", 12.5),
    ("gpt-5.6-terra", 10.0),
    ("gpt-5.4-mini", 3.8),
    ("gpt-5.6-luna", 1.0),
];

/// The published burn rate for `id`, or a tier-shaped estimate when the rate card does not name it.
fn quota_weight_for(id: &str, tier: Tier) -> f32 {
    if let Some((_, weight)) = CODEX_QUOTA_WEIGHTS.iter().find(|(slug, _)| *slug == id) {
        return *weight;
    }
    // Not on the rate card — a model newer than this table. Estimate from the vendor's own
    // ordering rather than assuming the cheapest, so an unknown frontier model is not preferred
    // for trivial work just because pstore has no number for it.
    match tier {
        Tier::Top => 25.0,
        Tier::Mid => 10.0,
        Tier::Cheap => 3.0,
    }
}

impl CatalogModel {
    /// Whether this is the model the agent's config currently names.
    pub fn is(&self, configured: &str) -> bool {
        self.id == configured
    }
}

/// Where an agent publishes the catalog it fetched for itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CatalogSource {
    /// Path relative to `$HOME`.
    pub file: &'static str,
}

/// Codex refreshes this from OpenAI and reads it back to populate its own model picker.
pub const CODEX_CATALOG: CatalogSource = CatalogSource {
    file: ".codex/models_cache.json",
};

/// Read the catalog `source` points at, or nothing if it cannot be understood.
///
/// Cheap: one small file parsed, once per agent per detection pass. Never on the ranking path.
pub fn read(source: CatalogSource, home: Option<&Path>) -> Vec<CatalogModel> {
    let Some(home) = home else {
        return Vec::new();
    };
    let Ok(text) = std::fs::read_to_string(home.join(source.file)) else {
        return Vec::new();
    };
    parse(&text)
}

/// Pull the usable models out of a Codex-shaped catalog document.
///
/// Shape, as of `codex-cli 0.147.0`: `{"models": [{"slug", "display_name", "description",
/// "visibility", "priority", ...}]}`. Every field is treated as optional — a document missing the
/// pieces pstore needs yields no models rather than half-populated ones.
fn parse(text: &str) -> Vec<CatalogModel> {
    let Ok(doc) = serde_json::from_str::<serde_json::Value>(text) else {
        return Vec::new();
    };
    let Some(models) = doc.get("models").and_then(|m| m.as_array()) else {
        return Vec::new();
    };

    let mut out: Vec<(i64, CatalogModel)> = models
        .iter()
        .filter_map(|m| {
            // `hide` marks routing aliases and internal models — `gpt-5.6-sol-wm`, the Work Mode
            // alias, and `codex-auto-review`. They are real ids the API accepts, which is exactly
            // why they must be filtered: offering a routing alias as a peer of the model it routes
            // to would put the same model in the ranking twice under two names.
            if m.get("visibility").and_then(|v| v.as_str()) != Some("list") {
                return None;
            }
            let id = m.get("slug").and_then(|s| s.as_str())?;
            if !plausible_id(id) {
                return None;
            }
            // Priority is the vendor's own ordering, ascending, and the only capability signal in
            // the document that is not prose.
            let priority = m.get("priority").and_then(|p| p.as_i64()).unwrap_or(i64::MAX);
            Some((
                priority,
                CatalogModel {
                    id: id.to_string(),
                    display: m
                        .get("display_name")
                        .and_then(|d| d.as_str())
                        .unwrap_or(id)
                        .to_string(),
                    note: m
                        .get("description")
                        .and_then(|d| d.as_str())
                        .unwrap_or_default()
                        .trim()
                        .to_string(),
                    tier: tier_for(priority),
                    relative_price: price_for(priority),
                    quota_weight: quota_weight_for(id, tier_for(priority)),
                },
            ))
        })
        .collect();

    // Best first, so the ranking and the `pstore agents` listing read in the order the vendor
    // considers most capable rather than in JSON order.
    out.sort_by_key(|(priority, _)| *priority);
    out.into_iter().map(|(_, m)| m).collect()
}

/// Whether `raw` looks like a model id rather than a stray string.
///
/// Deliberately the same shape-check discipline [`super::configured`] applies to config files:
/// this document is read without a schema pstore controls, so a field that moves could otherwise
/// put a sentence into the ranking as a model name.
fn plausible_id(raw: &str) -> bool {
    !raw.is_empty()
        && raw.len() <= 80
        && raw
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

/// Weight class from the vendor's priority ordering.
///
/// Coarse on purpose. Tier is shown and never scored, so the cost of getting a boundary slightly
/// wrong is a label; the cost of inventing a finer scale from a number whose semantics the vendor
/// never documented would be a claim pstore cannot support.
fn tier_for(priority: i64) -> Tier {
    match priority {
        ..=2 => Tier::Top,
        3..=9 => Tier::Mid,
        _ => Tier::Cheap,
    }
}

/// Relative price for a discovered model.
///
/// These models are covered by the Codex subscription, so there is no per-token rate to report and
/// nothing here is metered. The number exists because the UI shows one; like [`Tier`], the ranker
/// never reads it — see [`ModelSpec::relative_price`].
fn price_for(priority: i64) -> f32 {
    match tier_for(priority) {
        Tier::Top => 4.0,
        Tier::Mid => 2.5,
        Tier::Cheap => 1.5,
    }
}

/// A discovered model rendered as the registry row it stands in for.
///
/// Lets a caller that only knows [`ModelSpec`] treat a discovered model the same way, at the cost
/// of leaking the owned strings — acceptable because it is called once per detection pass over a
/// handful of models, never in a loop.
pub fn as_spec(model: &CatalogModel) -> ModelSpec {
    ModelSpec {
        id: Box::leak(model.id.clone().into_boxed_str()),
        display: Box::leak(model.display.clone().into_boxed_str()),
        tier: model.tier,
        relative_price: model.relative_price,
        metered: false,
        quota_weight: model.quota_weight,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real shape, trimmed to the fields pstore reads. Taken from an actual
    /// `~/.codex/models_cache.json` written by `codex-cli 0.147.0`.
    const CODEX_CACHE: &str = r#"{
      "fetched_at": "2026-08-07T05:44:00.349171Z",
      "models": [
        {"slug": "gpt-5.6-sol", "display_name": "GPT-5.6-Sol",
         "description": "Latest frontier agentic coding model.",
         "visibility": "list", "priority": 1},
        {"slug": "gpt-5.6-sol-wm", "display_name": "GPT-5.6-Sol-WM",
         "description": "Work Mode routing alias for GPT-5.6 Sol.",
         "visibility": "hide", "priority": 1},
        {"slug": "gpt-5.6-terra", "display_name": "GPT-5.6-Terra",
         "description": "Balanced agentic coding model for everyday work.",
         "visibility": "list", "priority": 2},
        {"slug": "gpt-5.6-luna", "display_name": "GPT-5.6-Luna",
         "description": "Fast and affordable agentic coding model.",
         "visibility": "list", "priority": 3},
        {"slug": "gpt-5.4-mini", "display_name": "GPT-5.4-Mini",
         "description": "Small, fast, and cost-efficient model for simpler coding tasks.",
         "visibility": "list", "priority": 23},
        {"slug": "codex-auto-review", "display_name": "Codex Auto Review",
         "description": "Automatic approval review model for Codex.",
         "visibility": "hide", "priority": 43}
      ]
    }"#;

    #[test]
    fn reads_the_visible_models_best_first() {
        let models = parse(CODEX_CACHE);
        let ids: Vec<_> = models.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(
            ids,
            ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.4-mini"],
            "visible models, in the vendor's own priority order"
        );

        let sol = &models[0];
        assert_eq!(sol.display, "GPT-5.6-Sol");
        assert_eq!(sol.note, "Latest frontier agentic coding model.");
        assert_eq!(sol.tier, Tier::Top);
    }

    /// The hidden entries are the reason this filter exists: `-wm` is an alias that routes to a
    /// model already in the list, so offering both would rank one model twice.
    #[test]
    fn hidden_models_are_not_offered() {
        let ids: Vec<_> = parse(CODEX_CACHE).into_iter().map(|m| m.id).collect();
        assert!(!ids.iter().any(|id| id == "gpt-5.6-sol-wm"));
        assert!(!ids.iter().any(|id| id == "codex-auto-review"));
    }

    /// Every discovered model must carry a note, because a model nothing can describe is withheld
    /// from ranking — discovering it and then excluding it would be worse than not discovering it.
    #[test]
    fn every_discovered_model_can_be_described() {
        for m in parse(CODEX_CACHE) {
            assert!(!m.note.is_empty(), "{} has no description", m.id);
        }
    }

    /// A cache pstore cannot understand has to leave the registry table in charge, not produce
    /// half-built models — the same rule [`super::super::configured`] follows.
    #[test]
    fn an_unreadable_cache_yields_nothing() {
        for junk in [
            "",
            "not json",
            "{}",
            r#"{"models": null}"#,
            r#"{"models": []}"#,
            r#"{"models": [{"visibility": "list"}]}"#,
            "<html><body>nope</body></html>",
        ] {
            assert!(parse(junk).is_empty(), "{junk:?} should yield no models");
        }
    }

    /// A moved field must not turn a sentence into a model id.
    #[test]
    fn implausible_ids_are_rejected() {
        let doc = r#"{"models": [
            {"slug": "the model to use", "visibility": "list", "priority": 1},
            {"slug": "https://api.example.com/v1", "visibility": "list", "priority": 2},
            {"slug": "gpt-5.6-terra", "visibility": "list", "priority": 3}
        ]}"#;
        let ids: Vec<_> = parse(doc).into_iter().map(|m| m.id).collect();
        assert_eq!(ids, ["gpt-5.6-terra"]);
    }

    /// A model with no description still has a usable id and label; the note being empty is what
    /// downstream uses to decide it cannot be ranked.
    #[test]
    fn a_model_without_a_description_keeps_its_identity() {
        let doc = r#"{"models": [{"slug": "gpt-9", "visibility": "list", "priority": 1}]}"#;
        let models = parse(doc);
        assert_eq!(models[0].id, "gpt-9");
        assert_eq!(models[0].display, "gpt-9", "falls back to the id");
        assert!(models[0].note.is_empty());
    }

    #[test]
    fn tiers_follow_the_vendors_ordering() {
        assert_eq!(tier_for(1), Tier::Top);
        assert_eq!(tier_for(2), Tier::Top);
        assert_eq!(tier_for(3), Tier::Mid);
        assert_eq!(tier_for(23), Tier::Cheap);
        assert_eq!(tier_for(i64::MAX), Tier::Cheap, "an absent priority is not a claim");
    }

    /// The whole point of the burn signal: on a subscription these models cost the same nothing
    /// extra, and drain the allowance at rates that differ by a factor of 25.
    #[test]
    fn quota_weight_follows_the_published_rate_card() {
        let by_id: std::collections::HashMap<_, _> = parse(CODEX_CACHE)
            .into_iter()
            .map(|m| (m.id.clone(), m.quota_weight))
            .collect();
        assert_eq!(by_id["gpt-5.6-sol"], 25.0);
        assert_eq!(by_id["gpt-5.6-terra"], 10.0);
        assert_eq!(by_id["gpt-5.6-luna"], 1.0);
        assert!(
            by_id["gpt-5.6-sol"] > by_id["gpt-5.6-terra"]
                && by_id["gpt-5.6-terra"] > by_id["gpt-5.6-luna"],
            "the ordering is the part that must hold even if the rates are re-baselined"
        );
    }

    /// A model shipped after this table was written must not read as free — that would route work
    /// to it *because* pstore knows nothing about it.
    #[test]
    fn an_unlisted_model_is_not_assumed_cheap() {
        let doc = r#"{"models": [{"slug": "gpt-6-titan", "description": "New frontier model.",
                       "visibility": "list", "priority": 1}]}"#;
        let models = parse(doc);
        assert_eq!(models[0].tier, Tier::Top);
        assert!(
            models[0].quota_weight >= 25.0,
            "an unknown frontier model is estimated heavy, not free"
        );
    }

    #[test]
    fn a_missing_home_reads_nothing() {
        assert!(read(CODEX_CATALOG, None).is_empty());
    }
}
