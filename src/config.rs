//! Resolved runtime configuration: where prompts live, plus persisted preferences.
//!
//! Preferences come from three layers, each overriding the one before it:
//!
//! 1. **System** — `/etc/pstore/config.json` (`%PROGRAMDATA%\pstore\config.json`). What an
//!    administrator sets for everyone on the machine.
//! 2. **User** — `~/.config/pstore/config.json`. What this person prefers everywhere.
//! 3. **Local** — `.pstore/config.json` beside the prompts. What this project needs.
//!
//! Layering exists mainly for [`crate::filter::Filter`]: "which models may we use" is
//! usually an organisation-wide answer with per-project exceptions, and expressing that by
//! copying the same block list into every checkout is how it goes stale.
//!
//! Only the **local** layer is written. A GUI that edited `/etc` would need privileges it
//! should not ask for, and silently rewriting a shared file from one project's window would
//! surprise everyone else on the machine.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::filter::Filter;

/// Directory name for all pstore sidecar state, created inside the prompt dir.
pub const SIDECAR: &str = ".pstore";

/// File name used at every layer.
const FILE: &str = "config.json";

/// Runtime configuration, resolved once at startup.
#[derive(Debug, Clone)]
pub struct Config {
    /// Folder holding the `.md` prompts.
    pub dir: PathBuf,
    /// Persisted user preferences, with every layer applied.
    pub prefs: Prefs,
    /// Config layers that could not be read, so a policy file that silently stopped
    /// applying is visible rather than merely absent.
    pub warnings: Vec<String>,
}

/// User preferences persisted to `.pstore/config.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Prefs {
    /// How many points of fit a hint may give up in exchange for a faster answer.
    ///
    /// Speed is a real-time property, not a price: hints are explicitly latency
    /// sensitive, so the hint path takes the quickest candidate scoring within this
    /// many points of the best. `0` always uses the top-scoring candidate.
    pub hint_score_tolerance: f32,
    /// Render the markdown preview instead of the editor.
    pub preview: bool,
    /// Width of the left column, in points.
    pub sidebar_width: f32,
    /// Preferred agent id to send to, overriding the ranker's pick.
    pub pinned_agent: Option<String>,
    /// Allow pstore to download the checkpoint and the `llama-cli` runtime.
    ///
    /// Covers both: with this off, pstore makes no network request of any kind, and the
    /// features that need the model are disabled rather than degraded.
    pub allow_model_download: bool,
    /// Use this `llama-cli` instead of a provisioned or discovered one.
    ///
    /// For a build the user made themselves — an accelerated one, say. It must be
    /// PrismML's fork; stock llama.cpp cannot load the checkpoint's quantisation.
    pub llama_cli_path: Option<String>,
    /// Hard upper bound on the context window any single model call may request.
    ///
    /// A **cap**, not a setting: the window is fitted to each prompt (see
    /// [`crate::router::llm::fit_context`]) and is normally far below this. Lowering it
    /// bounds memory on a small machine; raising it costs KV cache roughly linearly and is
    /// only needed for prompts larger than anything pstore currently sends.
    pub model_context_ceiling: usize,
    /// Which models and effort levels pstore may pick from.
    pub filter: Filter,
}

impl Default for Prefs {
    fn default() -> Self {
        Self {
            hint_score_tolerance: 8.0,
            preview: false,
            sidebar_width: 260.0,
            pinned_agent: None,
            allow_model_download: true,
            llama_cli_path: None,
            model_context_ceiling: 8192,
            filter: Filter::default(),
        }
    }
}

/// One layer of configuration, as read from disk.
///
/// Every field is optional so that "absent" and "set to the default value" stay
/// distinguishable — without that, a user layer would silently reimpose defaults over
/// whatever the system layer had set, and a config file could only ever be all-or-nothing.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Layer {
    hint_score_tolerance: Option<f32>,
    preview: Option<bool>,
    sidebar_width: Option<f32>,
    pinned_agent: Option<String>,
    allow_model_download: Option<bool>,
    llama_cli_path: Option<String>,
    model_context_ceiling: Option<usize>,
    filter: Option<Filter>,
}

impl Layer {
    /// Read one layer, or nothing if it is missing or unreadable.
    ///
    /// A malformed file is reported rather than ignored: a policy layer that fails to parse
    /// is precisely the case where silently continuing with defaults is wrong — an
    /// administrator's block list would vanish with no indication.
    fn read(path: &Path) -> (Option<Self>, Option<String>) {
        let Ok(text) = std::fs::read_to_string(path) else {
            return (None, None);
        };
        match serde_json::from_str(&text) {
            Ok(layer) => (Some(layer), None),
            Err(e) => (None, Some(format!("ignoring {}: {e}", path.display()))),
        }
    }

    /// Apply this layer over `base`.
    fn apply(self, base: &mut Prefs) {
        if let Some(v) = self.hint_score_tolerance {
            base.hint_score_tolerance = v;
        }
        if let Some(v) = self.preview {
            base.preview = v;
        }
        if let Some(v) = self.sidebar_width {
            base.sidebar_width = v;
        }
        if self.pinned_agent.is_some() {
            base.pinned_agent = self.pinned_agent;
        }
        if let Some(v) = self.allow_model_download {
            base.allow_model_download = v;
        }
        if self.llama_cli_path.is_some() {
            base.llama_cli_path = self.llama_cli_path;
        }
        if let Some(v) = self.model_context_ceiling {
            base.model_context_ceiling = v;
        }
        // Replaced wholesale rather than merged. A half-merged policy — this layer's allow
        // list against that layer's block list — is not something anyone can reason about
        // from reading either file.
        if let Some(v) = self.filter {
            base.filter = v;
        }
    }
}

/// The machine-wide configuration path, if this platform has one.
pub fn system_config() -> Option<PathBuf> {
    #[cfg(windows)]
    return std::env::var_os("PROGRAMDATA").map(|p| PathBuf::from(p).join("pstore").join(FILE));
    #[cfg(not(windows))]
    return Some(PathBuf::from("/etc/pstore").join(FILE));
}

/// The per-user configuration path.
pub fn user_config() -> Option<PathBuf> {
    dirs::config_dir().map(|d| d.join("pstore").join(FILE))
}

impl Config {
    /// Resolve the prompt directory from an explicit argument, `PSTORE_DIR`, or the cwd.
    pub fn resolve(explicit: Option<PathBuf>) -> std::io::Result<Self> {
        let dir = match explicit.or_else(|| std::env::var_os("PSTORE_DIR").map(PathBuf::from)) {
            Some(d) => d,
            None => std::env::current_dir()?,
        };
        // Canonicalize so the window title and child-process cwd agree, but tolerate
        // a not-yet-existing dir by creating it first.
        if !dir.exists() {
            std::fs::create_dir_all(&dir)?;
        }
        let dir = dir.canonicalize().unwrap_or(dir);
        let (prefs, warnings) = Prefs::load(&dir);
        publish(&prefs);
        Ok(Self {
            dir,
            prefs,
            warnings,
        })
    }
}

/// The preferences the model paths read, shared across threads.
///
/// [`crate::router::llm`] runs on a worker thread, several calls deep, and needs two
/// settings: which `llama-cli` to run and how large a context it may ask for. Threading a
/// `&Config` down to it would put a lifetime through every job signature to deliver two
/// scalars. Instead the app publishes a snapshot whenever preferences change, and the model
/// paths read it.
fn shared() -> &'static std::sync::RwLock<Prefs> {
    static SHARED: std::sync::OnceLock<std::sync::RwLock<Prefs>> = std::sync::OnceLock::new();
    SHARED.get_or_init(|| std::sync::RwLock::new(Prefs::default()))
}

/// Publish `prefs` for the model paths to read. Called on startup and on every change.
pub fn publish(prefs: &Prefs) {
    match shared().write() {
        Ok(mut g) => *g = prefs.clone(),
        Err(poisoned) => *poisoned.into_inner() = prefs.clone(),
    }
}

/// The current published preferences.
pub fn prefs_snapshot() -> Prefs {
    match shared().read() {
        Ok(g) => g.clone(),
        Err(poisoned) => poisoned.into_inner().clone(),
    }
}

impl Prefs {
    /// The local layer's path — the only one pstore writes.
    pub fn path(dir: &Path) -> PathBuf {
        dir.join(SIDECAR).join(FILE)
    }

    /// Load every layer in order, returning the result and any layer that failed to parse.
    pub fn load(dir: &Path) -> (Self, Vec<String>) {
        let layers = [system_config(), user_config(), Some(Self::path(dir))];
        Self::layered(layers.into_iter().flatten())
    }

    /// Apply `paths` in order over the defaults. Split out so the ordering is testable
    /// without writing to `/etc`.
    fn layered(paths: impl Iterator<Item = PathBuf>) -> (Self, Vec<String>) {
        let mut prefs = Prefs::default();
        let mut warnings = Vec::new();
        for path in paths {
            let (layer, warning) = Layer::read(&path);
            if let Some(layer) = layer {
                layer.apply(&mut prefs);
            }
            warnings.extend(warning);
        }
        (prefs, warnings)
    }

    /// Persist preferences. Best-effort: a read-only folder must not break the app.
    pub fn save(&self, dir: &Path) {
        let path = Self::path(dir);
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(json) = serde_json::to_string_pretty(self) {
            let _ = std::fs::write(path, json);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "pstore-cfg-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::remove_dir_all(&d).ok();
        std::fs::create_dir_all(d.join(SIDECAR)).unwrap();
        d
    }

    #[test]
    fn prefs_roundtrip_and_default_on_garbage() {
        let dir = tmp("roundtrip");
        let p = Prefs {
            hint_score_tolerance: 3.0,
            preview: true,
            ..Default::default()
        };
        p.save(&dir);
        let (back, warnings) = Prefs::load(&dir);
        assert_eq!(back.hint_score_tolerance, 3.0);
        assert!(back.preview);
        assert!(warnings.is_empty());

        std::fs::write(Prefs::path(&dir), "not json").unwrap();
        let (back, warnings) = Prefs::load(&dir);
        assert_eq!(
            back.hint_score_tolerance,
            Prefs::default().hint_score_tolerance,
            "corrupt prefs fall back to defaults"
        );
        // But not silently: a policy file that stopped applying has to be visible.
        assert_eq!(warnings.len(), 1, "got {warnings:?}");
        assert!(warnings[0].contains("config.json"), "got {warnings:?}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The layering rule the whole design rests on: later layers override earlier ones,
    /// and a field a layer does not mention is left alone rather than reset to its default.
    #[test]
    fn later_layers_override_and_absent_fields_are_left_alone() {
        let dir = tmp("layers");
        let system = dir.join("system.json");
        let user = dir.join("user.json");

        // System sets a policy and a width; user changes only the width.
        std::fs::write(
            &system,
            r#"{"sidebar_width": 100.0, "filter": {"allow": ["*sonnet*"], "block": [],
               "efforts": [], "block_metered": true}}"#,
        )
        .unwrap();
        std::fs::write(&user, r#"{"sidebar_width": 300.0}"#).unwrap();

        let (prefs, warnings) = Prefs::layered([system, user].into_iter());
        assert!(warnings.is_empty(), "got {warnings:?}");
        assert_eq!(prefs.sidebar_width, 300.0, "the later layer wins");
        assert_eq!(
            prefs.filter.allow,
            vec!["*sonnet*".to_string()],
            "the user layer must not erase a policy it never mentioned"
        );
    }

    /// A layer that is simply absent must contribute nothing — the common case, since most
    /// machines have no system config at all.
    #[test]
    fn a_missing_layer_is_not_an_error() {
        let (prefs, warnings) =
            Prefs::layered([PathBuf::from("/definitely/not/here/config.json")].into_iter());
        assert!(warnings.is_empty());
        assert_eq!(prefs.sidebar_width, Prefs::default().sidebar_width);
    }

    #[test]
    fn defaults_leave_room_for_a_faster_hint() {
        assert!(
            Prefs::default().hint_score_tolerance > 0.0,
            "hints are latency sensitive by default"
        );
    }
}
