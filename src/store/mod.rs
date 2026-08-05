//! Prompt files on disk: enumeration, creation, rename, delete.
//!
//! Prompts are plain `.md` files directly in the working folder so they stay usable
//! outside pstore. All pstore-specific state lives in the `.pstore/` sidecar.

pub mod version;

use std::path::{Path, PathBuf};

use crate::config::SIDECAR;

/// A single prompt file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Prompt {
    /// Absolute path to the `.md` file.
    pub path: PathBuf,
    /// Filename stem, used as the display name.
    pub name: String,
    /// Kebab-cased identifier used for sidecar paths.
    pub slug: String,
}

impl Prompt {
    fn from_path(path: PathBuf) -> Option<Self> {
        let name = path.file_stem()?.to_string_lossy().into_owned();
        let slug = slugify(&name);
        Some(Self { path, name, slug })
    }
}

/// Lowercase, kebab-case, filesystem-safe identifier for a prompt name.
///
/// Collapses any run of non-alphanumeric characters to a single `-`. Empty results
/// become `untitled` so sidecar paths are never empty.
pub fn slugify(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut pending_dash = false;
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() {
            if pending_dash && !out.is_empty() {
                out.push('-');
            }
            pending_dash = false;
            out.push(ch.to_ascii_lowercase());
        } else if ch.is_alphanumeric() {
            // Keep non-ASCII letters/digits rather than mangling them.
            if pending_dash && !out.is_empty() {
                out.push('-');
            }
            pending_dash = false;
            out.extend(ch.to_lowercase());
        } else {
            pending_dash = true;
        }
    }
    if out.is_empty() {
        "untitled".to_string()
    } else {
        out
    }
}

/// Enumerates and mutates the prompt files in one directory.
#[derive(Debug, Clone)]
pub struct PromptStore {
    dir: PathBuf,
}

impl PromptStore {
    /// Create a store over `dir`.
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    /// The directory being managed.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// List `.md` prompts, newest-modified first. Non-recursive; skips `.pstore/`.
    pub fn list(&self) -> Vec<Prompt> {
        let Ok(entries) = std::fs::read_dir(&self.dir) else {
            return Vec::new();
        };
        let mut with_time: Vec<(std::time::SystemTime, Prompt)> = entries
            .filter_map(Result::ok)
            .filter_map(|e| {
                let path = e.path();
                if path.file_name().is_some_and(|n| n == SIDECAR) {
                    return None;
                }
                if !path.is_file() {
                    return None;
                }
                if path.extension()?.to_string_lossy().to_ascii_lowercase() != "md" {
                    return None;
                }
                let mtime = e
                    .metadata()
                    .and_then(|m| m.modified())
                    .unwrap_or(std::time::UNIX_EPOCH);
                Some((mtime, Prompt::from_path(path)?))
            })
            .collect();
        // Newest first; ties broken by name so ordering is stable across runs.
        with_time.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.name.cmp(&b.1.name)));
        with_time.into_iter().map(|(_, p)| p).collect()
    }

    /// Create a new empty prompt named `name`, returning it.
    ///
    /// If the name is taken, a numeric suffix is appended rather than overwriting.
    pub fn create(&self, name: &str) -> std::io::Result<Prompt> {
        let base = if name.trim().is_empty() {
            "untitled"
        } else {
            name.trim()
        };
        let mut candidate = self.dir.join(format!("{base}.md"));
        let mut n = 2;
        while candidate.exists() {
            candidate = self.dir.join(format!("{base}-{n}.md"));
            n += 1;
        }
        std::fs::write(&candidate, "")?;
        Prompt::from_path(candidate)
            .ok_or_else(|| std::io::Error::other("could not derive prompt name"))
    }

    /// Read a prompt's contents.
    pub fn read(&self, prompt: &Prompt) -> std::io::Result<String> {
        std::fs::read_to_string(&prompt.path)
    }

    /// Write a prompt's contents.
    pub fn write(&self, prompt: &Prompt, text: &str) -> std::io::Result<()> {
        std::fs::write(&prompt.path, text)
    }

    /// Rename a prompt, moving its version history with it. Returns the updated prompt.
    pub fn rename(&self, prompt: &Prompt, new_name: &str) -> std::io::Result<Prompt> {
        let new_name = new_name.trim();
        if new_name.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "name cannot be empty",
            ));
        }
        let target = self.dir.join(format!("{new_name}.md"));
        if target != prompt.path && target.exists() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                format!("{new_name}.md already exists"),
            ));
        }
        std::fs::rename(&prompt.path, &target)?;
        let renamed = Prompt::from_path(target)
            .ok_or_else(|| std::io::Error::other("could not derive prompt name"))?;
        // Carry history across so a rename does not orphan snapshots.
        version::rename_history(&self.dir, &prompt.slug, &renamed.slug)?;
        Ok(renamed)
    }

    /// Delete a prompt and its version history.
    pub fn delete(&self, prompt: &Prompt) -> std::io::Result<()> {
        std::fs::remove_file(&prompt.path)?;
        version::delete_history(&self.dir, &prompt.slug)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "pstore-store-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::remove_dir_all(&p).ok();
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn slugify_collapses_and_lowercases() {
        assert_eq!(slugify("Refactor Auth!!"), "refactor-auth");
        assert_eq!(slugify("  leading/trailing  "), "leading-trailing");
        assert_eq!(slugify("already-kebab"), "already-kebab");
        assert_eq!(slugify("!!!"), "untitled");
        assert_eq!(slugify("v1.2.3"), "v1-2-3");
    }

    #[test]
    fn create_avoids_clobbering_and_list_skips_sidecar() {
        let dir = tmpdir("create");
        let store = PromptStore::new(&dir);

        let a = store.create("notes").unwrap();
        let b = store.create("notes").unwrap();
        assert_ne!(a.path, b.path);
        assert!(b.path.ends_with("notes-2.md"));

        // Sidecar dir and non-md files must not show up as prompts.
        std::fs::create_dir_all(dir.join(SIDECAR)).unwrap();
        std::fs::write(dir.join(SIDECAR).join("x.md"), "hidden").unwrap();
        std::fs::write(dir.join("readme.txt"), "nope").unwrap();

        let names: Vec<_> = store.list().into_iter().map(|p| p.name).collect();
        assert_eq!(names.len(), 2, "got {names:?}");
        assert!(names.contains(&"notes".to_string()));
        assert!(names.contains(&"notes-2".to_string()));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rename_moves_history_and_refuses_collision() {
        let dir = tmpdir("rename");
        let store = PromptStore::new(&dir);
        let p = store.create("old").unwrap();
        store.write(&p, "body").unwrap();
        version::snapshot(&dir, &p.slug, "body", version::Note::Manual).unwrap();
        assert_eq!(version::list(&dir, &p.slug).len(), 1);

        let other = store.create("taken").unwrap();
        assert!(store.rename(&p, "taken").is_err());
        assert!(p.path.exists(), "failed rename must not move the file");

        let renamed = store.rename(&p, "new").unwrap();
        assert_eq!(renamed.slug, "new");
        assert_eq!(
            version::list(&dir, "new").len(),
            1,
            "history follows rename"
        );
        assert!(version::list(&dir, "old").is_empty());

        store.delete(&other).unwrap();
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn delete_removes_file_and_history() {
        let dir = tmpdir("delete");
        let store = PromptStore::new(&dir);
        let p = store.create("doomed").unwrap();
        version::snapshot(&dir, &p.slug, "x", version::Note::Manual).unwrap();
        store.delete(&p).unwrap();
        assert!(!p.path.exists());
        assert!(version::list(&dir, &p.slug).is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }
}
