//! Which models pstore is allowed to pick.
//!
//! Ranking is only useful over a field the developer would actually accept. Two things
//! routinely make a model unacceptable regardless of how well it fits a prompt: it bills
//! per token on top of a subscription that is already paid for, or the organisation simply
//! does not permit it. Neither is a judgement the local model should be making, so both are
//! policy, expressed here and applied before anything is offered for ranking.
//!
//! Two modes, because they suit opposite situations:
//!
//! * **Blocking** — everything is allowed except what matches [`Filter::block`]. Right when
//!   most of the field is fine and a few models are not.
//! * **Allowing** — nothing is allowed except what matches [`Filter::allow`]. Right when
//!   the acceptable set is small and naming it is less work than naming the rest, which is
//!   the common case under a procurement policy.
//!
//! `allow` wins when both are set: a model must match `allow` *and* not match `block`. That
//! ordering means a broad allow list can still be narrowed, and it fails closed.

use serde::{Deserialize, Serialize};

use crate::agents::registry::Effort;

/// Model and effort policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Filter {
    /// Patterns that disqualify a model. Empty blocks nothing.
    ///
    /// Matched case-insensitively against both the model's id and its display name, and
    /// against `agent/model` so one agent's copy of a shared model can be singled out.
    pub block: Vec<String>,
    /// Patterns that qualify a model. Empty allows everything not blocked.
    pub allow: Vec<String>,
    /// Effort levels pstore may request. Empty allows every level an agent supports.
    pub efforts: Vec<Effort>,
    /// Refuse models billed per token rather than covered by a subscription.
    ///
    /// On by default. Every other model in the registry is already paid for, so picking one
    /// costs nothing extra; a metered model spends money the developer has not agreed to
    /// spend. A ranker that treats those as interchangeable will eventually bill someone by
    /// accident, and the failure is silent — it shows up on an invoice, not on screen.
    pub block_metered: bool,
}

impl Default for Filter {
    fn default() -> Self {
        Self {
            // Named explicitly as well as covered by `block_metered`, so the default
            // configuration also demonstrates the pattern syntax to anyone who opens it.
            block: vec!["*fable*".into()],
            allow: Vec::new(),
            efforts: Vec::new(),
            block_metered: true,
        }
    }
}

impl Filter {
    /// Whether a model may be offered for ranking.
    ///
    /// `agent` and `id` identify it; `display` is matched too because that is the name a
    /// user reads in the UI and will naturally reach for when writing a pattern.
    pub fn allows_model(&self, agent: &str, id: &str, display: &str, metered: bool) -> bool {
        if self.block_metered && metered {
            return false;
        }
        let names = [
            id.to_string(),
            display.to_string(),
            format!("{agent}/{id}"),
            format!("{agent}/{display}"),
        ];
        let hits = |pats: &[String]| {
            pats.iter()
                .any(|p| names.iter().any(|n| matches(p.trim(), n)))
        };

        if !self.allow.is_empty() && !hits(&self.allow) {
            return false;
        }
        !hits(&self.block)
    }

    /// Whether pstore may request this effort level.
    pub fn allows_effort(&self, effort: Effort) -> bool {
        self.efforts.is_empty() || self.efforts.contains(&effort)
    }

    /// One line describing what this filter does, for the UI.
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        if !self.allow.is_empty() {
            parts.push(format!("only {}", self.allow.join(", ")));
        }
        if self.block_metered {
            parts.push("no per-token models".into());
        }
        if !self.block.is_empty() {
            parts.push(format!("not {}", self.block.join(", ")));
        }
        if !self.efforts.is_empty() {
            parts.push(format!(
                "effort {}",
                self.efforts
                    .iter()
                    .map(|e| e.as_str())
                    .collect::<Vec<_>>()
                    .join("/")
            ));
        }
        if parts.is_empty() {
            "every installed model".into()
        } else {
            parts.join(" · ")
        }
    }
}

/// Case-insensitive glob match supporting `*` (any run, including empty) and `?` (one
/// character).
///
/// A pattern with no wildcards must match the whole name, not merely appear in it — so
/// `sonnet` does not block `sonnet-thinking` unless the user writes `sonnet*`. Guessing at
/// substring semantics would silently block more than was asked for, and this is a setting
/// whose whole job is to be predictable.
pub fn matches(pattern: &str, name: &str) -> bool {
    let p: Vec<char> = pattern.to_lowercase().chars().collect();
    let n: Vec<char> = name.to_lowercase().chars().collect();
    glob(&p, &n)
}

/// Iterative glob with backtracking on `*`. Linear in practice, and cannot blow the stack
/// on a pattern like `****` the way the naive recursion can.
fn glob(pattern: &[char], name: &[char]) -> bool {
    let (mut p, mut n) = (0usize, 0usize);
    // Where to resume if the current `*` turns out to have consumed too little.
    let (mut star, mut resume) = (None, 0usize);

    while n < name.len() {
        match pattern.get(p) {
            Some('*') => {
                star = Some(p);
                resume = n;
                p += 1;
            }
            Some('?') => {
                p += 1;
                n += 1;
            }
            Some(c) if *c == name[n] => {
                p += 1;
                n += 1;
            }
            _ => match star {
                Some(s) => {
                    p = s + 1;
                    resume += 1;
                    n = resume;
                }
                None => return false,
            },
        }
    }
    pattern[p..].iter().all(|c| *c == '*')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patterns_match_whole_names_unless_wildcarded() {
        assert!(matches("sonnet", "sonnet"));
        assert!(matches("SONNET", "sonnet"), "case-insensitive");
        // The bug this guards: a bare name silently behaving as a substring, which would
        // block more models than the user named.
        assert!(!matches("sonnet", "sonnet-thinking"));
        assert!(matches("sonnet*", "sonnet-thinking"));
        assert!(matches("*fable*", "claude-fable-5"));
        assert!(matches("*", "anything"));
        assert!(matches("gpt-?", "gpt-5"));
        assert!(!matches("gpt-?", "gpt-51"));
        assert!(!matches("opus", "sonnet"));
    }

    /// Backtracking has to actually backtrack: `*-5` against `gpt-5-mini-5` must find the
    /// second `-5`, not give up at the first.
    #[test]
    fn wildcards_backtrack() {
        assert!(matches("*-5", "gpt-5-mini-5"));
        assert!(!matches("*-5", "gpt-5-mini"));
        assert!(matches("*a*b*c*", "xxaxxbxxcxx"));
        assert!(!matches("*a*b*c*", "xxaxxcxxbxx"));
        // Degenerate patterns must terminate rather than exploding.
        assert!(matches("****", "abc"));
        assert!(matches("", ""));
        assert!(!matches("", "abc"));
    }

    #[test]
    fn metered_models_are_blocked_by_default() {
        let f = Filter::default();
        assert!(
            !f.allows_model("claude", "claude-fable-5", "Fable 5", true),
            "a per-token model must not be offered unless asked for"
        );
        assert!(f.allows_model("claude", "claude-sonnet-5", "Sonnet 5", false));
    }

    /// Turning the metered rule off has to be enough on its own — but the default also
    /// names Fable by pattern, so both have to be cleared to reach it. That is deliberate,
    /// and this test pins it so nobody "fixes" the redundancy without noticing.
    #[test]
    fn reaching_a_metered_model_takes_two_deliberate_changes() {
        let mut f = Filter {
            block_metered: false,
            ..Filter::default()
        };
        assert!(
            !f.allows_model("claude", "claude-fable-5", "Fable 5", true),
            "the explicit pattern should still hold it back"
        );
        f.block.clear();
        assert!(f.allows_model("claude", "claude-fable-5", "Fable 5", true));
    }

    /// The whitelist is for a tight, policy-defined field: naming what is permitted should
    /// exclude everything else without the user enumerating it.
    #[test]
    fn an_allow_list_excludes_everything_it_does_not_name() {
        let f = Filter {
            allow: vec!["*sonnet*".into(), "gpt-5*".into()],
            block: Vec::new(),
            block_metered: false,
            efforts: Vec::new(),
        };
        assert!(f.allows_model("claude", "claude-sonnet-5", "Sonnet 5", false));
        assert!(f.allows_model("codex", "gpt-5.1-codex", "GPT-5.1 Codex", false));
        assert!(!f.allows_model("claude", "claude-opus-5", "Opus 5", false));
        assert!(!f.allows_model("gemini", "gemini-3-pro", "Gemini 3 Pro", false));
    }

    /// Both lists together must fail closed: allow admits a family, block carves one out.
    #[test]
    fn block_narrows_an_allow_list() {
        let f = Filter {
            allow: vec!["claude/*".into()],
            block: vec!["*opus*".into()],
            block_metered: false,
            efforts: Vec::new(),
        };
        assert!(f.allows_model("claude", "claude-sonnet-5", "Sonnet 5", false));
        assert!(!f.allows_model("claude", "claude-opus-5", "Opus 5", false));
        assert!(!f.allows_model("codex", "gpt-5.1-codex", "GPT-5.1", false));
    }

    /// Matching on the display name matters because that is the string the user sees in
    /// the ranking table and will copy into their config.
    #[test]
    fn display_names_and_agent_paths_are_matchable() {
        let f = Filter {
            block: vec!["Opus 5".into()],
            allow: Vec::new(),
            block_metered: false,
            efforts: Vec::new(),
        };
        assert!(!f.allows_model("claude", "claude-opus-5", "Opus 5", false));

        let scoped = Filter {
            block: vec!["crush/*".into()],
            allow: Vec::new(),
            block_metered: false,
            efforts: Vec::new(),
        };
        assert!(!scoped.allows_model("crush", "x", "X", false));
        assert!(scoped.allows_model("claude", "x", "X", false));
    }

    #[test]
    fn effort_filtering_is_opt_in() {
        let none = Filter::default();
        assert!(Effort::ALL.iter().all(|e| none.allows_effort(*e)));

        let low_only = Filter {
            efforts: vec![Effort::Low, Effort::Medium],
            ..Filter::default()
        };
        assert!(low_only.allows_effort(Effort::Low));
        assert!(!low_only.allows_effort(Effort::Max));
    }

    #[test]
    fn summary_describes_the_policy() {
        assert!(Filter::default().summary().contains("per-token"));
        let open = Filter {
            block: Vec::new(),
            allow: Vec::new(),
            efforts: Vec::new(),
            block_metered: false,
        };
        assert_eq!(open.summary(), "every installed model");

        let tight = Filter {
            allow: vec!["*sonnet*".into()],
            ..Filter::default()
        };
        assert!(tight.summary().contains("only *sonnet*"));
    }
}
