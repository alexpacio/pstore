//! Static table of known coding-agent CLIs, their models, and their effort levels.
//!
//! One row per agent. Everything pstore needs to *drive* an agent lives here, so
//! adapting to upstream flag changes is a one-line edit rather than a code change.

use std::fmt;

use super::catalog::CatalogSource;
use super::configured::ModelSource;

/// Cost/capability tier. Informational: shown in the UI, never scored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tier {
    /// Fast, lightweight models.
    Cheap,
    /// The general-purpose middle.
    Mid,
    /// Frontier models.
    Top,
}

impl fmt::Display for Tier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Tier::Cheap => "light",
            Tier::Mid => "mid",
            Tier::Top => "frontier",
        })
    }
}

/// Reasoning-effort level. Higher effort raises a model's effective capability and
/// its latency.
///
/// Serialised in lowercase so a config file names levels the way the agent CLIs do
/// (`"low"`, `"xhigh"`) rather than the way Rust spells them.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "lowercase")]
pub enum Effort {
    /// Minimal reasoning — fastest.
    Low,
    /// Balanced.
    Medium,
    /// Thorough; the usual default.
    High,
    /// Deeper than `High`, for hard coding and agentic work.
    XHigh,
    /// Maximum depth, slowest.
    Max,
}

impl Effort {
    /// All levels, ascending.
    pub const ALL: [Effort; 5] = [
        Effort::Low,
        Effort::Medium,
        Effort::High,
        Effort::XHigh,
        Effort::Max,
    ];

    /// The value passed to an agent's effort flag.
    pub fn as_str(self) -> &'static str {
        match self {
            Effort::Low => "low",
            Effort::Medium => "medium",
            Effort::High => "high",
            Effort::XHigh => "xhigh",
            Effort::Max => "max",
        }
    }

    /// Relative time-to-answer, `1.0` at [`Effort::Low`]. Used for the
    /// speed-sensitive hint path and shown in the UI.
    pub fn latency_factor(self) -> f32 {
        match self {
            Effort::Low => 1.0,
            Effort::Medium => 1.6,
            Effort::High => 2.6,
            Effort::XHigh => 4.0,
            Effort::Max => 6.0,
        }
    }
}

impl fmt::Display for Effort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// How an agent accepts an effort level on the command line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffortFlag {
    /// A dedicated flag, e.g. `--effort high`.
    Flag(&'static str),
    /// A config override taking `key=value`, e.g. `-c model_reasoning_effort=high`.
    ConfigKv(&'static str, &'static str),
    /// No per-invocation control; effort comes from the agent's own settings.
    Unsupported,
}

impl EffortFlag {
    /// Arguments that select `effort`, or empty when unsupported.
    pub fn args(self, effort: Effort) -> Vec<String> {
        match self {
            EffortFlag::Flag(f) => vec![f.to_string(), effort.as_str().to_string()],
            EffortFlag::ConfigKv(flag, key) => {
                vec![flag.to_string(), format!("{key}={}", effort.as_str())]
            }
            EffortFlag::Unsupported => Vec::new(),
        }
    }

    /// Whether pstore can choose the effort for this agent.
    pub fn is_supported(self) -> bool {
        !matches!(self, EffortFlag::Unsupported)
    }
}

/// How a prompt is handed to the agent process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptVia {
    /// Appended as the final positional argument.
    Arg,
    /// Written to the child's stdin.
    Stdin,
}

/// One selectable model exposed by an agent.
#[derive(Debug, Clone)]
pub struct ModelSpec {
    /// Value passed to the agent's model flag.
    pub id: &'static str,
    /// Human label for the UI.
    pub display: &'static str,
    /// Weight class. Informational only.
    pub tier: Tier,
    /// Relative price for the same work, normalised so the cheapest known model is
    /// `1.0`.
    ///
    /// **Displayed, never scored.** The ranker deliberately ignores this: pstore
    /// reports how well each model and effort fits the prompt and leaves spending
    /// decisions to the developer. Enforced by
    /// `scoring::tests::price_does_not_influence_ranking`.
    pub relative_price: f32,
    /// Billed per token on top of the subscription, rather than included in it.
    ///
    /// This is **not** the price signal — that is [`Self::relative_price`], which the
    /// ranker ignores on purpose. It is a different kind of fact: every other model here
    /// is already paid for, so picking one costs nothing extra, whereas picking a metered
    /// one spends money the developer has not spent yet. A router that treats those as
    /// interchangeable will quietly bill someone, so the ranker holds metered models back
    /// unless they are clearly needed — see [`crate::router::scoring::METERED_MARGIN`].
    pub metered: bool,
    /// How fast this model spends a subscription's allowance, relative to the vendor's lightest.
    ///
    /// **Not [`Self::relative_price`].** Price is dollars per token on the API, and the ranker is
    /// forbidden from reading it — spending decisions belong to the developer. This is a different
    /// fact: on a subscription every one of these models costs the same zero dollars extra, but
    /// they drain the plan's limits at wildly different rates, and a developer who burns a weekly
    /// cap on work a light model would have done is blocked whether or not they were billed.
    ///
    /// Anthropic publishes no per-model multiplier — only that "Opus costs several times more per
    /// turn than Sonnet, and Sonnet more than Haiku". These values take published *output-token*
    /// rates as the documented stand-in for that ordering, normalised so Haiku 4.5 is `1.0`:
    /// Haiku $5, Sonnet $15, Opus $25, Fable $50 per MTok. Treat them as an ordering pstore can
    /// defend, not a multiplier Anthropic has stated.
    pub quota_weight: f32,
}

/// A coding-agent CLI installed (or installable) on the system.
#[derive(Debug, Clone)]
pub struct AgentSpec {
    /// Stable identifier used in config and cache files.
    pub id: &'static str,
    /// Human label for the UI.
    pub display: &'static str,
    /// Executable name, looked up on `PATH`.
    pub bin: &'static str,
    /// Arguments that put the agent in non-interactive mode, before the prompt.
    pub headless: &'static [&'static str],
    /// Flag used to pin a model, if the CLI has one.
    ///
    /// `None` means the model comes from the agent's own config file and pstore
    /// cannot choose it — a real capability gap the ranker has to respect.
    pub model_flag: Option<&'static str>,
    /// How this agent accepts an effort level.
    pub effort_flag: EffortFlag,
    /// Effort levels this agent actually accepts, ascending.
    pub efforts: &'static [Effort],
    /// Extra arguments for the headless path (streaming/quiet flags).
    pub headless_extra: &'static [&'static str],
    /// How the prompt reaches the process in headless mode.
    pub prompt_via: PromptVia,
    /// Arguments for an interactive session, before the prompt.
    pub interactive: &'static [&'static str],
    /// Paths (relative to `$HOME`) whose presence hints the agent is configured.
    pub creds: &'static [&'static str],
    /// Models pstore may select. Empty means "whatever the agent is configured with".
    pub models: &'static [ModelSpec],
    /// Where to look for the model this agent is configured with.
    ///
    /// For the agents whose [`Self::models`] is empty, this is load-bearing: their model lives
    /// in their own config, and without reading it the ranker would be handed a candidate with
    /// no name and no properties. See [`crate::agents::configured`], which also explains why a
    /// source that finds nothing is better than a guess.
    ///
    /// An agent whose [`Self::models`] is *not* empty may still declare a source here — the
    /// ranker keeps choosing from the static table (a live config value is never fed into
    /// scoring or launched with), but `pstore agents` uses it to show what the agent is actually
    /// running right now, which drifts from a hand-maintained table the moment the vendor ships
    /// a new default. Codex is the current example: it takes `-m`, so pstore still picks from
    /// [`CODEX_MODELS`], but its config also names whichever model the interactive CLI last
    /// selected, and that is worth surfacing rather than silently contradicting.
    pub model_config: &'static [ModelSource],
    /// Where this agent publishes the model catalog it fetched for itself, when it does.
    ///
    /// Takes precedence over [`Self::models`] whenever it yields anything: the agent's own catalog
    /// is what the agent will actually accept, and the registry table is a hand-written guess at
    /// the same thing that goes stale the day the vendor ships. The table stays as the fallback
    /// for when the catalog is missing or unreadable. See [`crate::agents::catalog`].
    pub catalog: Option<CatalogSource>,
}

/// Stand-in model for agents that don't let pstore choose one. Scored so those
/// agents still appear in the ranking instead of silently vanishing.
pub const UNKNOWN_MODEL: ModelSpec = ModelSpec {
    id: "",
    display: "(agent default)",
    tier: Tier::Mid,
    relative_price: 3.0,
    metered: false,
    // A model pstore cannot name gets the mid estimate: assuming it is light would route work to
    // it for being unknown, which is the same poisoning `knowledge` exists to prevent.
    quota_weight: 3.0,
};

impl AgentSpec {
    /// Models to score for this agent: its own table, or a single placeholder when
    /// the model is fixed by the agent's configuration.
    pub fn scoreable_models(&self) -> &[ModelSpec] {
        if self.models.is_empty() {
            std::slice::from_ref(&UNKNOWN_MODEL)
        } else {
            self.models
        }
    }

    /// Effort levels to score for this agent. Agents with no effort control are
    /// scored at a single representative level.
    pub fn scoreable_efforts(&self) -> &'static [Effort] {
        if self.efforts.is_empty() {
            &[Effort::High]
        } else {
            self.efforts
        }
    }
}

// [instruction_following, coding, math_reasoning, world_knowledge, planning_agentic, creative_synthesis]

pub const CLAUDE_EFFORTS: &[Effort] = &[
    Effort::Low,
    Effort::Medium,
    Effort::High,
    Effort::XHigh,
    Effort::Max,
];
pub const CODEX_EFFORTS: &[Effort] = &[Effort::Low, Effort::Medium, Effort::High, Effort::XHigh];

/// Claude Code — the Claude 5 family. `relative_price` follows published input-token
/// rates (Haiku 4.5 $1, Sonnet 5 $3, Opus 5 $5, Fable 5 $10 per MTok) and is shown
/// for information only.
const CLAUDE_MODELS: &[ModelSpec] = &[
    ModelSpec {
        id: "haiku",
        display: "Haiku 4.5",
        tier: Tier::Cheap,
        relative_price: 1.0,
        metered: false,
        quota_weight: 1.0,
    },
    ModelSpec {
        id: "sonnet",
        display: "Sonnet 5",
        tier: Tier::Mid,
        relative_price: 3.0,
        metered: false,
        quota_weight: 3.0,
    },
    ModelSpec {
        id: "opus",
        display: "Opus 5",
        tier: Tier::Top,
        relative_price: 5.0,
        metered: false,
        quota_weight: 5.0,
    },
    // The one model here that is not covered by a Claude Code subscription: it is billed
    // per token. Its skill vector also dominates Opus on every dimension, so without
    // `metered` the ranker would pick it for anything hard — and, because ties break
    // alphabetically, for plenty that was not hard at all.
    ModelSpec {
        id: "fable",
        display: "Fable 5",
        tier: Tier::Top,
        relative_price: 10.0,
        metered: true,
        quota_weight: 10.0,
    },
];

const CODEX_MODELS: &[ModelSpec] = &[ModelSpec {
    id: "gpt-5.1-codex",
    display: "GPT-5.1 Codex",
    tier: Tier::Top,
    relative_price: 4.0,
    metered: false,
        quota_weight: 25.0,
}];

const GEMINI_MODELS: &[ModelSpec] = &[
    ModelSpec {
        id: "gemini-3-flash",
        display: "Gemini 3 Flash",
        tier: Tier::Cheap,
        relative_price: 1.2,
        metered: false,
        quota_weight: 1.0,
    },
    ModelSpec {
        id: "gemini-3-pro",
        display: "Gemini 3 Pro",
        tier: Tier::Mid,
        relative_price: 3.5,
        metered: false,
        quota_weight: 3.5,
    },
];

// ---------------------------------------------------------------------------
// Where the agents that choose their own model write it down
// ---------------------------------------------------------------------------
//
// One block per agent, and every one of them is a *place to look* rather than a format pstore
// claims to know — see [`crate::agents::configured`]. Several keys each because these names get
// renamed upstream, and trying the previous spelling costs one lookup in a document already
// parsed. A source that finds nothing leaves the agent's model unknown, which excludes it from
// ranking with a stated reason; that is the intended outcome, not a failure.

/// Cursor Agent keeps CLI settings beside the editor's.
const CURSOR_MODEL: &[ModelSource] = &[ModelSource {
    file: ".cursor/cli-config.json",
    keys: &["model", "defaultModel", "editor.model"],
}];

/// Crush names a model per size class, and the large one is what a prompt from pstore hits.
const CRUSH_MODEL: &[ModelSource] = &[ModelSource {
    file: ".config/crush/crush.json",
    keys: &[
        "models.large.model",
        "models.large.id",
        "model",
        "default_model",
    ],
}];

/// Aider reads a project file if there is one, and falls back to the one in `$HOME`.
const AIDER_MODEL: &[ModelSource] = &[
    ModelSource {
        file: "./.aider.conf.yml",
        keys: &["model"],
    },
    ModelSource {
        file: ".aider.conf.yml",
        keys: &["model"],
    },
];

/// Goose spells its settings as environment-variable names inside YAML.
const GOOSE_MODEL: &[ModelSource] = &[ModelSource {
    file: ".config/goose/config.yaml",
    keys: &["GOOSE_MODEL", "model"],
}];

/// Qwen Code is a Gemini-CLI derivative and keeps its settings the same way.
const QWEN_MODEL: &[ModelSource] = &[ModelSource {
    file: ".qwen/settings.json",
    keys: &["model", "model.name", "defaultModel"],
}];

/// Factory Droid.
const DROID_MODEL: &[ModelSource] = &[ModelSource {
    file: ".factory/config.json",
    keys: &["model", "defaultModel", "custom_models.0.model"],
}];

/// Codex CLI writes the model the interactive session last picked into its own config, in the
/// same plain `key = "value"` shape `configured::scan_lines` already reads for Aider and Goose.
/// Unlike those two, Codex also has [`CODEX_MODELS`] — this source is not what the ranker
/// chooses from, only what `pstore agents` reports as the model currently in effect.
const CODEX_MODEL: &[ModelSource] = &[ModelSource {
    file: ".codex/config.toml",
    keys: &["model"],
}];

/// Every agent pstore knows how to drive.
pub const AGENTS: &[AgentSpec] = &[
    AgentSpec {
        id: "claude",
        display: "Claude Code",
        bin: "claude",
        headless: &["-p"],
        model_flag: Some("--model"),
        effort_flag: EffortFlag::Flag("--effort"),
        efforts: CLAUDE_EFFORTS,
        headless_extra: &["--output-format", "stream-json", "--verbose"],
        prompt_via: PromptVia::Arg,
        interactive: &[],
        creds: &[".claude.json", ".claude"],
        models: CLAUDE_MODELS,
        model_config: &[],
        catalog: None,
    },
    AgentSpec {
        id: "codex",
        display: "OpenAI Codex",
        bin: "codex",
        headless: &["exec"],
        model_flag: Some("-m"),
        effort_flag: EffortFlag::ConfigKv("-c", "model_reasoning_effort"),
        efforts: CODEX_EFFORTS,
        headless_extra: &["--skip-git-repo-check"],
        prompt_via: PromptVia::Arg,
        interactive: &[],
        creds: &[".codex/auth.json", ".codex"],
        models: CODEX_MODELS,
        model_config: CODEX_MODEL,
        catalog: Some(super::catalog::CODEX_CATALOG),
    },
    AgentSpec {
        id: "gemini",
        display: "Gemini CLI",
        bin: "gemini",
        headless: &["-p"],
        model_flag: Some("-m"),
        // Gemini takes thinkingLevel from settings.json; there is no per-run flag.
        effort_flag: EffortFlag::Unsupported,
        efforts: &[],
        headless_extra: &[],
        prompt_via: PromptVia::Arg,
        interactive: &[],
        creds: &[".gemini"],
        models: GEMINI_MODELS,
        model_config: &[],
        catalog: None,
    },
    AgentSpec {
        id: "cursor",
        display: "Cursor Agent",
        bin: "cursor-agent",
        headless: &["-p"],
        model_flag: Some("-m"),
        effort_flag: EffortFlag::Unsupported,
        efforts: &[],
        headless_extra: &[],
        prompt_via: PromptVia::Arg,
        interactive: &[],
        creds: &[".cursor"],
        models: &[],
        model_config: CURSOR_MODEL,
        catalog: None,
    },
    AgentSpec {
        id: "crush",
        display: "Crush",
        bin: "crush",
        // No model or effort flag: both come from ~/.config/crush/crush.json.
        headless: &["run"],
        model_flag: None,
        effort_flag: EffortFlag::Unsupported,
        efforts: &[],
        headless_extra: &["-q"],
        prompt_via: PromptVia::Stdin,
        interactive: &[],
        creds: &[".config/crush/crush.json"],
        models: &[],
        model_config: CRUSH_MODEL,
        catalog: None,
    },
    AgentSpec {
        id: "aider",
        display: "Aider",
        bin: "aider",
        headless: &["--message"],
        model_flag: Some("--model"),
        effort_flag: EffortFlag::Unsupported,
        efforts: &[],
        headless_extra: &["--no-auto-commits", "--yes"],
        prompt_via: PromptVia::Arg,
        interactive: &[],
        creds: &[".aider.conf.yml"],
        models: &[],
        model_config: AIDER_MODEL,
        catalog: None,
    },
    AgentSpec {
        id: "goose",
        display: "Goose",
        bin: "goose",
        headless: &["run", "-t"],
        model_flag: None,
        effort_flag: EffortFlag::Unsupported,
        efforts: &[],
        headless_extra: &[],
        prompt_via: PromptVia::Arg,
        interactive: &["session"],
        creds: &[".config/goose/config.yaml"],
        models: &[],
        model_config: GOOSE_MODEL,
        catalog: None,
    },
    AgentSpec {
        id: "qwen",
        display: "Qwen Code",
        bin: "qwen",
        headless: &["-p"],
        model_flag: Some("-m"),
        effort_flag: EffortFlag::Unsupported,
        efforts: &[],
        headless_extra: &[],
        prompt_via: PromptVia::Arg,
        interactive: &[],
        creds: &[".qwen"],
        models: &[],
        model_config: QWEN_MODEL,
        catalog: None,
    },
    AgentSpec {
        id: "copilot",
        display: "GitHub Copilot CLI",
        bin: "copilot",
        headless: &["-p"],
        model_flag: Some("--model"),
        effort_flag: EffortFlag::Unsupported,
        efforts: &[],
        headless_extra: &[],
        prompt_via: PromptVia::Arg,
        interactive: &[],
        creds: &[".config/github-copilot"],
        models: &[],
        model_config: &[],
        catalog: None,
    },
    AgentSpec {
        id: "droid",
        display: "Factory Droid",
        bin: "droid",
        headless: &["exec"],
        model_flag: Some("-m"),
        effort_flag: EffortFlag::Unsupported,
        efforts: &[],
        headless_extra: &[],
        prompt_via: PromptVia::Arg,
        interactive: &[],
        creds: &[".factory"],
        models: &[],
        model_config: DROID_MODEL,
        catalog: None,
    },
    AgentSpec {
        id: "amp",
        display: "Amp",
        bin: "amp",
        headless: &["-x"],
        model_flag: None,
        effort_flag: EffortFlag::Unsupported,
        efforts: &[],
        headless_extra: &[],
        prompt_via: PromptVia::Stdin,
        interactive: &[],
        creds: &[".config/amp"],
        models: &[],
        model_config: &[],
        catalog: None,
    },
];

/// Look up an agent by id.
pub fn find(id: &str) -> Option<&'static AgentSpec> {
    AGENTS.iter().find(|a| a.id == id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_and_bins_are_unique() {
        let mut ids: Vec<_> = AGENTS.iter().map(|a| a.id).collect();
        ids.sort_unstable();
        let before = ids.len();
        ids.dedup();
        assert_eq!(ids.len(), before, "duplicate agent id");

        let mut bins: Vec<_> = AGENTS.iter().map(|a| a.bin).collect();
        bins.sort_unstable();
        let before = bins.len();
        bins.dedup();
        assert_eq!(bins.len(), before, "duplicate binary name");
    }

    /// Latency ordering is what the hint path sorts on, so it has to be monotonic — a
    /// higher effort that claimed to be faster would send hints to the slowest option.
    #[test]
    fn effort_latency_rises_with_effort() {
        for pair in Effort::ALL.windows(2) {
            let (lo, hi) = (pair[0], pair[1]);
            assert!(
                lo.latency_factor() < hi.latency_factor(),
                "{lo} latency >= {hi}"
            );
        }
        assert_eq!(
            Effort::Low.latency_factor(),
            1.0,
            "Low is the latency baseline"
        );
    }

    #[test]
    fn effort_flags_render_per_agent_syntax() {
        let claude = find("claude").unwrap();
        assert_eq!(
            claude.effort_flag.args(Effort::XHigh),
            vec!["--effort".to_string(), "xhigh".to_string()]
        );

        let codex = find("codex").unwrap();
        assert_eq!(
            codex.effort_flag.args(Effort::High),
            vec!["-c".to_string(), "model_reasoning_effort=high".to_string()]
        );

        let gemini = find("gemini").unwrap();
        assert!(gemini.effort_flag.args(Effort::Max).is_empty());
        assert!(!gemini.effort_flag.is_supported());
    }

    #[test]
    fn declared_efforts_match_flag_support() {
        // Listing effort levels pstore cannot actually select would mislead the ranking.
        for agent in AGENTS {
            if !agent.effort_flag.is_supported() {
                assert!(
                    agent.efforts.is_empty(),
                    "{} cannot set effort but declares levels",
                    agent.id
                );
            } else {
                assert!(
                    !agent.efforts.is_empty(),
                    "{} has an effort flag but no levels",
                    agent.id
                );
            }
        }
    }

    #[test]
    fn agents_without_a_model_flag_declare_no_models() {
        for agent in AGENTS {
            if agent.model_flag.is_none() {
                assert!(
                    agent.models.is_empty(),
                    "{} has no model flag but lists models",
                    agent.id
                );
            }
        }
    }

    #[test]
    fn every_agent_is_scoreable() {
        // Even agents pstore can't configure must produce at least one candidate,
        // so they appear in the ranking with a score rather than disappearing.
        for agent in AGENTS {
            assert!(
                !agent.scoreable_models().is_empty(),
                "{} has no candidates",
                agent.id
            );
            assert!(
                !agent.scoreable_efforts().is_empty(),
                "{} has no efforts",
                agent.id
            );
        }
        let crush = find("crush").unwrap();
        assert_eq!(crush.scoreable_models().len(), 1);
        assert_eq!(crush.scoreable_models()[0].display, "(agent default)");
    }
}
