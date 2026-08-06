//! Sidecar version history for prompts.
//!
//! Layout inside the working folder:
//!
//! ```text
//! .pstore/
//!   index.json                     { "<slug>": [VersionMeta, ...] }
//!   versions/<slug>/<ts>.md        full snapshot of the prompt at <ts>
//! ```
//!
//! Snapshots are whole-file copies, not diffs: prompts are small, and a plain `.md`
//! per version means the history stays readable without pstore.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::config::SIDECAR;

/// Why a snapshot was taken. Recorded so the history reads as a story.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Note {
    /// Explicit save (Ctrl+S) or switching away from a dirty buffer.
    Manual,
    /// Debounced idle autosave.
    Autosave,
    /// Result of the prompt shrinker being accepted.
    Shrink,
    /// Result of the planner being accepted.
    Plan,
    /// State captured immediately before restoring an older version.
    Restore,
    /// A hint was inserted into the document.
    Hint,
    /// Personal data was masked out.
    Sanitize,
}

impl Note {
    /// Short human label for the history list.
    pub fn label(self) -> &'static str {
        match self {
            Note::Manual => "saved",
            Note::Autosave => "autosave",
            Note::Shrink => "shrunk",
            Note::Plan => "planned",
            Note::Restore => "pre-restore",
            Note::Hint => "hint",
            Note::Sanitize => "masked",
        }
    }
}

/// Metadata for one stored version.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VersionMeta {
    /// Timestamp id, also the snapshot filename stem (`YYYYMMDD-HHMMSS`).
    pub ts: String,
    /// Size of the snapshot in bytes.
    pub bytes: u64,
    /// Provenance of this snapshot.
    pub note: Note,
}

/// The on-disk index: slug -> versions, oldest first.
#[derive(Debug, Default, Serialize, Deserialize)]
struct Index {
    #[serde(flatten)]
    by_slug: BTreeMap<String, Vec<VersionMeta>>,
}

fn index_path(dir: &Path) -> PathBuf {
    dir.join(SIDECAR).join("index.json")
}

fn versions_dir(dir: &Path, slug: &str) -> PathBuf {
    dir.join(SIDECAR).join("versions").join(slug)
}

fn load_index(dir: &Path) -> Index {
    std::fs::read_to_string(index_path(dir))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn save_index(dir: &Path, index: &Index) -> std::io::Result<()> {
    let path = index_path(dir);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(index).map_err(std::io::Error::other)?;
    std::fs::write(path, json)
}

/// Take a snapshot of `text`.
///
/// Returns `Ok(None)` when the newest existing snapshot is byte-identical — repeated
/// saves of unchanged text must not fill the history with noise.
pub fn snapshot(
    dir: &Path,
    slug: &str,
    text: &str,
    note: Note,
) -> std::io::Result<Option<VersionMeta>> {
    let mut index = load_index(dir);
    let entries = index.by_slug.entry(slug.to_string()).or_default();

    if let Some(last) = entries.last() {
        // Compare against the newest snapshot's bytes before writing anything.
        if last.bytes == text.len() as u64
            && let Ok(prev) =
                std::fs::read_to_string(versions_dir(dir, slug).join(format!("{}.md", last.ts)))
            && prev == text
        {
            return Ok(None);
        }
    }

    let vdir = versions_dir(dir, slug);
    std::fs::create_dir_all(&vdir)?;

    // Second-resolution stamps can collide on rapid saves; disambiguate with a suffix.
    let stamp = chrono::Local::now().format("%Y%m%d-%H%M%S").to_string();
    let mut ts = stamp.clone();
    let mut n = 1;
    while vdir.join(format!("{ts}.md")).exists() {
        ts = format!("{stamp}-{n}");
        n += 1;
    }

    std::fs::write(vdir.join(format!("{ts}.md")), text)?;
    let meta = VersionMeta {
        ts,
        bytes: text.len() as u64,
        note,
    };
    entries.push(meta.clone());
    save_index(dir, &index)?;
    Ok(Some(meta))
}

/// List versions for a slug, **newest first**.
pub fn list(dir: &Path, slug: &str) -> Vec<VersionMeta> {
    let mut v = load_index(dir).by_slug.remove(slug).unwrap_or_default();
    v.reverse();
    v
}

/// Read the stored text of one version.
pub fn read(dir: &Path, slug: &str, ts: &str) -> std::io::Result<String> {
    std::fs::read_to_string(versions_dir(dir, slug).join(format!("{ts}.md")))
}

/// Move a slug's history after the prompt file was renamed.
pub fn rename_history(dir: &Path, from: &str, to: &str) -> std::io::Result<()> {
    if from == to {
        return Ok(());
    }
    let (src, dst) = (versions_dir(dir, from), versions_dir(dir, to));
    if src.exists() {
        if let Some(parent) = dst.parent() {
            std::fs::create_dir_all(parent)?;
        }
        if dst.exists() {
            // Merge rather than fail: move individual snapshots into the destination.
            for entry in std::fs::read_dir(&src)?.filter_map(Result::ok) {
                let target = dst.join(entry.file_name());
                if !target.exists() {
                    std::fs::rename(entry.path(), target)?;
                }
            }
            std::fs::remove_dir_all(&src)?;
        } else {
            std::fs::rename(&src, &dst)?;
        }
    }
    let mut index = load_index(dir);
    if let Some(mut entries) = index.by_slug.remove(from) {
        let merged = index.by_slug.entry(to.to_string()).or_default();
        merged.append(&mut entries);
        merged.sort_by(|a, b| a.ts.cmp(&b.ts));
        save_index(dir, &index)?;
    }
    Ok(())
}

/// Delete a slug's entire history.
pub fn delete_history(dir: &Path, slug: &str) -> std::io::Result<()> {
    let vdir = versions_dir(dir, slug);
    if vdir.exists() {
        std::fs::remove_dir_all(vdir)?;
    }
    let mut index = load_index(dir);
    if index.by_slug.remove(slug).is_some() {
        save_index(dir, &index)?;
    }
    Ok(())
}

/// Unified diff from `old` to `new`, suitable for monospace display.
pub fn diff(old: &str, new: &str) -> String {
    use similar::{ChangeTag, TextDiff};

    let diff = TextDiff::from_lines(old, new);
    let mut out = String::new();
    for group in diff.grouped_ops(3) {
        for op in group {
            for change in diff.iter_changes(&op) {
                out.push(match change.tag() {
                    ChangeTag::Delete => '-',
                    ChangeTag::Insert => '+',
                    ChangeTag::Equal => ' ',
                });
                out.push_str(change.value());
                if change.missing_newline() {
                    out.push('\n');
                }
            }
        }
        out.push_str("...\n");
    }
    if out.is_empty() {
        out.push_str("(no differences)\n");
    } else if out.ends_with("...\n") {
        // Trailing separator adds nothing after the final hunk.
        out.truncate(out.len() - 4);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "pstore-ver-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::remove_dir_all(&p).ok();
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn snapshot_roundtrip_and_ordering() {
        let dir = tmpdir("roundtrip");
        let m1 = snapshot(&dir, "p", "first", Note::Manual).unwrap().unwrap();
        let m2 = snapshot(&dir, "p", "second", Note::Autosave)
            .unwrap()
            .unwrap();

        let listed = list(&dir, "p");
        assert_eq!(listed.len(), 2);
        assert_eq!(listed[0].ts, m2.ts, "newest first");
        assert_eq!(listed[1].ts, m1.ts);
        assert_eq!(listed[0].note, Note::Autosave);

        assert_eq!(read(&dir, "p", &m1.ts).unwrap(), "first");
        assert_eq!(read(&dir, "p", &m2.ts).unwrap(), "second");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn identical_content_is_deduped() {
        let dir = tmpdir("dedupe");
        assert!(snapshot(&dir, "p", "same", Note::Manual).unwrap().is_some());
        assert!(
            snapshot(&dir, "p", "same", Note::Manual).unwrap().is_none(),
            "byte-identical save must not create a version"
        );
        assert_eq!(list(&dir, "p").len(), 1);
        // A real change is recorded again.
        assert!(
            snapshot(&dir, "p", "different", Note::Manual)
                .unwrap()
                .is_some()
        );
        assert_eq!(list(&dir, "p").len(), 2);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rapid_snapshots_do_not_collide() {
        let dir = tmpdir("collide");
        // Same-second writes with differing content must produce distinct ids.
        let a = snapshot(&dir, "p", "a", Note::Manual).unwrap().unwrap();
        let b = snapshot(&dir, "p", "b", Note::Manual).unwrap().unwrap();
        let c = snapshot(&dir, "p", "c", Note::Manual).unwrap().unwrap();
        assert_ne!(a.ts, b.ts);
        assert_ne!(b.ts, c.ts);
        assert_eq!(read(&dir, "p", &a.ts).unwrap(), "a");
        assert_eq!(read(&dir, "p", &c.ts).unwrap(), "c");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn index_survives_reload_and_garbage() {
        let dir = tmpdir("reload");
        snapshot(&dir, "p", "x", Note::Manual).unwrap();
        // Fresh load from disk (no in-memory state) still sees the version.
        assert_eq!(list(&dir, "p").len(), 1);

        std::fs::write(index_path(&dir), "{{{ not json").unwrap();
        assert!(
            list(&dir, "p").is_empty(),
            "corrupt index degrades to empty"
        );
        // And writing again recovers rather than erroring.
        assert!(snapshot(&dir, "p", "y", Note::Manual).unwrap().is_some());
        assert_eq!(list(&dir, "p").len(), 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn diff_reports_changes_and_equality() {
        assert!(diff("same\n", "same\n").contains("no differences"));
        let d = diff("one\ntwo\n", "one\nTWO\n");
        assert!(d.contains("-two") || d.contains("- two"), "got: {d}");
        assert!(d.contains("+TWO") || d.contains("+ TWO"), "got: {d}");
    }

    #[test]
    fn rename_merges_into_existing_history() {
        let dir = tmpdir("merge");
        snapshot(&dir, "a", "from-a", Note::Manual).unwrap();
        snapshot(&dir, "b", "from-b", Note::Manual).unwrap();
        rename_history(&dir, "a", "b").unwrap();
        assert!(list(&dir, "a").is_empty());
        assert_eq!(list(&dir, "b").len(), 2, "both histories preserved");
        std::fs::remove_dir_all(&dir).ok();
    }
}
