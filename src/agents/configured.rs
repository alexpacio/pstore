//! Which model an agent is *actually* configured to use.
//!
//! Half the agents in the registry do not let pstore choose a model: the model comes from the
//! agent's own config file, and [`AgentSpec::models`](super::registry::AgentSpec::models) is
//! empty for them. Ranking used to offer those agents a placeholder called `(agent default)`,
//! which is a candidate with **no information attached at all** — no name, no tier, nothing
//! for the local model to judge. Asked to rank it against Opus 5 and Haiku 4.5, the checkpoint
//! has nothing to go on and invents something, and one invented row moves everything below it.
//! That is the poisoning this module exists to stop.
//!
//! So pstore reads the agent's own config and finds the model name in it. What comes back is a
//! real name — `anthropic/claude-sonnet-4-5`, `gpt-5`, `qwen3-coder-plus` — which
//! [`crate::knowledge`] can then either describe or refuse to rank.
//!
//! **This is discovery, not a schema.** Every agent here spells its config differently and
//! changes it between releases, so a source is a *list of places to look* rather than a parser
//! for a format pstore claims to know. A key that is absent, a file that has moved, a format
//! that changed — all of them return `None`, which is a candidate excluded from ranking with a
//! reason the user can read. Guessing a model name would be worse than not knowing one: pstore
//! would rank a model the agent is not going to run.

use std::path::{Path, PathBuf};

use super::registry::AgentSpec;

/// One place an agent's model name might be written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelSource {
    /// Where the file is, relative to `$HOME` — or to the project when prefixed `./`.
    ///
    /// Project-local first for the agents that support both, because that is the one the agent
    /// itself would prefer when run in this directory.
    pub file: &'static str,
    /// Dotted key paths to try, in order. First one that yields a plausible name wins.
    ///
    /// Several per agent on purpose: these keys have been renamed upstream before, and trying
    /// the old spelling as well costs one `get` on a document already parsed.
    pub keys: &'static [&'static str],
}

impl ModelSource {
    /// Resolve this source against a project directory and `$HOME`.
    fn path(&self, project: &Path) -> Option<PathBuf> {
        match self.file.strip_prefix("./") {
            Some(rel) => Some(project.join(rel)),
            None => dirs::home_dir().map(|h| h.join(self.file)),
        }
    }
}

/// Find the model `spec` is configured with, if its config says.
///
/// Cheap: a few `stat`s and at most one small file parsed. Called once per agent per detection
/// pass, never on the ranking path.
pub fn model_for(spec: &AgentSpec, project: &Path) -> Option<String> {
    spec.model_config
        .iter()
        .filter_map(|source| {
            let path = source.path(project)?;
            let text = std::fs::read_to_string(&path).ok()?;
            find_in(&text, source.keys)
        })
        .next()
}

/// Pull the first plausible model name out of `text` using `keys`.
///
/// JSON is parsed properly. Anything else — YAML, TOML, an `.env`-shaped file — is scanned
/// line by line for `key: value` or `key = value`, matching the **last** segment of a dotted
/// key. That is deliberately loose: a real parser for each of three formats would be a lot of
/// code to reach one string, and a loose scan that finds nothing is exactly as harmless as a
/// strict one that finds nothing.
fn find_in(text: &str, keys: &[&str]) -> Option<String> {
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(text)
        && let Some(found) = keys.iter().find_map(|k| json_path(&json, k))
    {
        return Some(found);
    }
    keys.iter().find_map(|k| scan_lines(text, k))
}

/// Walk a dotted path through a JSON document.
///
/// Numeric segments index arrays, so `custom_models.0.model` reaches the first entry of a list.
fn json_path(root: &serde_json::Value, path: &str) -> Option<String> {
    let mut cursor = root;
    for segment in path.split('.') {
        cursor = match segment.parse::<usize>() {
            Ok(i) => cursor.get(i)?,
            Err(_) => cursor.get(segment)?,
        };
    }
    cursor.as_str().and_then(plausible)
}

/// Look for `key: value` or `key = value`, by the last segment of a dotted path.
fn scan_lines(text: &str, path: &str) -> Option<String> {
    let wanted = path.rsplit('.').next()?;
    for line in text.lines() {
        // A `#` inside a quoted value would be eaten here. No model name contains one, and
        // keeping a real comment parser out of this is worth more than that edge.
        let line = line.split('#').next().unwrap_or(line).trim();
        let Some((key, value)) = line.split_once(':').or_else(|| line.split_once('=')) else {
            continue;
        };
        // `- model: x` in a YAML list, and `"model": x` in JSON that failed to parse.
        let key = key
            .trim()
            .trim_start_matches("- ")
            .trim_matches(['"', '\'']);
        if key != wanted {
            continue;
        }
        if let Some(found) = plausible(value.trim().trim_matches(['"', '\'', ','])) {
            return Some(found);
        }
    }
    None
}

/// Whether `raw` looks like a model name rather than a placeholder or a sentence.
///
/// The filter matters more than it looks. These files are read without knowing their schema, so
/// the wrong key can match — and a value like `true`, `default` or a whole prose line would
/// then be carried into the ranking as if it were a model. Anything that does not look like an
/// identifier is dropped, which lands the candidate in "no model information" where it belongs.
fn plausible(raw: &str) -> Option<String> {
    /// Longest a real model name runs. `anthropic/claude-sonnet-4-5-20250929` is 36.
    const MAX: usize = 80;

    let raw = raw.trim();
    if raw.is_empty() || raw.len() > MAX {
        return None;
    }
    // A name is one token: no spaces, and nothing that reads as prose.
    if raw.contains(char::is_whitespace) {
        return None;
    }
    // An endpoint, not a model. These files sit next to each other in the same config — Crush
    // and Aider both take a `base_url` beside the model — and `:` and `/` are legal in a
    // qualified model name, so the two cannot be told apart by their characters alone.
    if raw.contains("://") || raw.starts_with("http") {
        return None;
    }
    // Values that are settings rather than names.
    if matches!(
        raw.to_ascii_lowercase().as_str(),
        "true" | "false" | "null" | "none" | "default" | "auto" | "" | "unset"
    ) {
        return None;
    }
    // Model names are made of these. A path, a URL or a JSON fragment is not a model.
    if !raw
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '/' | ':'))
    {
        return None;
    }
    // A digit-only value is a version or a count, never a name.
    if raw.chars().all(|c| c.is_ascii_digit() || c == '.') {
        return None;
    }
    Some(raw.to_string())
}

/// The part of a name a person — or a model — would recognise.
///
/// Agent configs qualify models by provider (`anthropic/claude-sonnet-4-5`,
/// `openai:gpt-5`). The provider is right to keep for launching the agent and wrong to
/// keep when asking what a model *is*, so [`crate::knowledge`] looks it up by the tail.
pub fn bare_name(model: &str) -> &str {
    model
        .rsplit(['/', ':'])
        .find(|part| !part.is_empty())
        .unwrap_or(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_configs_are_read_by_path() {
        let crush = r#"{"models": {"large": {"model": "claude-sonnet-4-5",
                        "provider": "anthropic"}}}"#;
        assert_eq!(
            find_in(crush, &["models.large.model", "model"]),
            Some("claude-sonnet-4-5".into())
        );

        // An array index reaches into a list.
        let droid = r#"{"custom_models": [{"model": "gpt-5-codex"}]}"#;
        assert_eq!(
            find_in(droid, &["custom_models.0.model"]),
            Some("gpt-5-codex".into())
        );

        // Keys are tried in order, and a missing one simply moves on.
        assert_eq!(
            find_in(r#"{"model": "gpt-5"}"#, &["defaultModel", "model"]),
            Some("gpt-5".into())
        );
    }

    /// YAML and `.env`-shaped files are scanned rather than parsed, so the loose path has to
    /// work on the shapes these agents actually write.
    #[test]
    fn line_shaped_configs_are_scanned() {
        let goose = "GOOSE_PROVIDER: openai\nGOOSE_MODEL: gpt-5\n";
        assert_eq!(
            find_in(goose, &["GOOSE_MODEL"]),
            Some("gpt-5".into()),
            "a YAML key: value line"
        );

        let aider = "# my settings\nmodel: openai/gpt-5\nauto-commits: false\n";
        assert_eq!(find_in(aider, &["model"]), Some("openai/gpt-5".into()));

        // TOML spelling, and a quoted value.
        assert_eq!(
            find_in("model = \"claude-opus-4-5\"\n", &["model"]),
            Some("claude-opus-4-5".into())
        );

        // A dotted key matches on its last segment when the file is not JSON.
        assert_eq!(
            find_in("  model: qwen3-coder\n", &["models.large.model"]),
            Some("qwen3-coder".into())
        );
    }

    /// The whole point of reading these files is to avoid ranking a model pstore invented, so
    /// a value that is not a name has to be rejected rather than carried through.
    #[test]
    fn implausible_values_are_rejected() {
        for junk in [
            "true",
            "false",
            "null",
            "default",
            "auto",
            "",
            "   ",
            "the model to use for large requests",
            "1.5",
            "42",
            "{\"nested\": 1}",
            "https://api.example.com/v1",
        ] {
            assert_eq!(plausible(junk), None, "{junk:?} should not be a model name");
        }
        // A name longer than any real one is a paragraph that happened to lack spaces.
        assert_eq!(plausible(&"x".repeat(200)), None);

        // And the shapes that are real names all survive.
        for good in [
            "gpt-5",
            "claude-sonnet-4-5",
            "anthropic/claude-opus-4-5",
            "openai:gpt-5-codex",
            "gemini-2.5-pro",
            "qwen3_coder",
        ] {
            assert_eq!(plausible(good), Some(good.to_string()), "{good:?}");
        }
    }

    /// A key found nowhere must leave the candidate unknown rather than half-guessed.
    #[test]
    fn a_config_without_a_model_yields_nothing() {
        assert_eq!(find_in(r#"{"theme": "dark"}"#, &["model"]), None);
        assert_eq!(find_in("unrelated: yes\n", &["model"]), None);
        assert_eq!(find_in("", &["model"]), None);
        // A file that is not even a config.
        assert_eq!(find_in("<html><body>nope</body></html>", &["model"]), None);
    }

    #[test]
    fn provider_prefixes_are_stripped_for_lookup() {
        assert_eq!(
            bare_name("anthropic/claude-sonnet-4-5"),
            "claude-sonnet-4-5"
        );
        assert_eq!(bare_name("openai:gpt-5"), "gpt-5");
        assert_eq!(bare_name("gpt-5"), "gpt-5");
        assert_eq!(bare_name("openrouter/anthropic/claude-opus"), "claude-opus");
    }

    /// Sources resolve against `$HOME` unless they are project-local, and getting that
    /// backwards would read one project's config for every project.
    #[test]
    fn project_local_sources_resolve_against_the_project() {
        let project = Path::new("/tmp/some-project");
        let local = ModelSource {
            file: "./.aider.conf.yml",
            keys: &["model"],
        };
        assert_eq!(
            local.path(project),
            Some(project.join(".aider.conf.yml")),
            "a ./ source belongs to the project"
        );

        let home = ModelSource {
            file: ".config/goose/config.yaml",
            keys: &["GOOSE_MODEL"],
        };
        let resolved = home.path(project).expect("a home directory");
        assert!(resolved.ends_with(".config/goose/config.yaml"));
        assert!(
            !resolved.starts_with(project),
            "a home source must not be read from the project"
        );
    }

    /// Every agent that cannot be told which model to run should have somewhere to look, and
    /// every declared source should name a file and at least one key.
    ///
    /// An agent with its own model table (`models` non-empty) may still declare a source: that
    /// is not two answers to the same question, because the ranker never consults
    /// `model_config` while `models` is non-empty — see [`super::registry::AgentSpec::model_config`].
    /// It exists so `pstore agents` can report the live value alongside the static one.
    #[test]
    fn declared_sources_are_well_formed() {
        for agent in super::super::registry::AGENTS {
            for source in agent.model_config {
                assert!(!source.file.is_empty(), "{}: empty source path", agent.id);
                assert!(
                    !source.keys.is_empty(),
                    "{}: {} has no keys to try",
                    agent.id,
                    source.file
                );
                assert!(
                    !source.file.starts_with('/'),
                    "{}: {} must be relative to $HOME or the project",
                    agent.id,
                    source.file
                );
            }
        }
    }
}
