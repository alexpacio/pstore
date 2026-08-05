//! Resolved runtime configuration: where prompts live, plus persisted preferences.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Directory name for all pstore sidecar state, created inside the prompt dir.
pub const SIDECAR: &str = ".pstore";

/// Runtime configuration, resolved once at startup.
#[derive(Debug, Clone)]
pub struct Config {
    /// Folder holding the `.md` prompts.
    pub dir: PathBuf,
    /// Persisted user preferences.
    pub prefs: Prefs,
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
    /// Allow downloading Brick classifier weights from Hugging Face.
    pub allow_model_download: bool,
}

impl Default for Prefs {
    fn default() -> Self {
        Self {
            hint_score_tolerance: 8.0,
            preview: false,
            sidebar_width: 260.0,
            pinned_agent: None,
            allow_model_download: true,
        }
    }
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
        let prefs = Prefs::load(&dir);
        Ok(Self { dir, prefs })
    }
}

impl Prefs {
    fn path(dir: &Path) -> PathBuf {
        dir.join(SIDECAR).join("config.json")
    }

    /// Load preferences, falling back to defaults on any error (missing, corrupt, unreadable).
    pub fn load(dir: &Path) -> Self {
        std::fs::read_to_string(Self::path(dir))
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
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

    #[test]
    fn prefs_roundtrip_and_default_on_garbage() {
        let tmp = std::env::temp_dir().join(format!("pstore-prefs-{}", std::process::id()));
        std::fs::create_dir_all(tmp.join(SIDECAR)).unwrap();
        let p = Prefs {
            hint_score_tolerance: 3.0,
            preview: true,
            ..Default::default()
        };
        p.save(&tmp);
        let back = Prefs::load(&tmp);
        assert_eq!(back.hint_score_tolerance, 3.0);
        assert!(back.preview);

        std::fs::write(Prefs::path(&tmp), "not json").unwrap();
        assert_eq!(
            Prefs::load(&tmp).hint_score_tolerance,
            Prefs::default().hint_score_tolerance,
            "corrupt prefs fall back to defaults"
        );
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn defaults_leave_room_for_a_faster_hint() {
        assert!(
            Prefs::default().hint_score_tolerance > 0.0,
            "hints are latency sensitive by default"
        );
    }
}
