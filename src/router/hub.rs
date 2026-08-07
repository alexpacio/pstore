//! Fetching a file from Hugging Face into the shared cache.
//!
//! This used to delegate to the `hf-hub` crate. It does not any more: hf-hub 1.0.0 stalls
//! **deterministically** at 11,021,778 bytes into `Bonsai-27B-Q1_0.gguf` — from a clean
//! cache, every time, while `curl` and `ureq` both stream the same URL at full speed. A
//! 7.17 GB download that stops 0.3% in and never resumes is not something the app can work
//! around, so the transfer is done here instead.
//!
//! What is *not* reinvented is the cache **layout**. Weights land in the same
//! `~/.cache/huggingface/hub` structure the Python and Rust ecosystems use, so other tools
//! reuse them and pstore stores nothing of its own:
//!
//! ```text
//! models--{owner}--{name}/
//!   refs/main                       the commit this snapshot came from
//!   blobs/{sha256}                  content-addressed file (LFS)
//!   blobs/{git-sha1}                content-addressed file (plain)
//!   snapshots/{commit}/{filename}   symlink to ../../blobs/{id}
//! ```
//!
//! Two things fall out of doing it here that hf-hub did not offer: transfers **resume**
//! from a partial `.incomplete` file with a `Range` request, and they **cancel mid-file**
//! rather than only between files.

use std::path::PathBuf;

/// Split a `owner/name` repo id.
pub fn split_repo(repo_id: &str) -> Result<(&str, &str), String> {
    repo_id
        .split_once('/')
        .filter(|(o, n)| !o.is_empty() && !n.is_empty())
        .ok_or_else(|| format!("{repo_id:?} is not an owner/name repository id"))
}

/// Directory name Hugging Face uses for a repo.
pub fn repo_folder(repo_id: &str) -> Result<String, String> {
    let (owner, name) = split_repo(repo_id)?;
    Ok(format!("models--{owner}--{name}"))
}

/// Root of the shared cache.
///
/// Honours `HF_HUB_CACHE` and `HF_HOME` the way the rest of the ecosystem does, so a
/// machine that has already relocated its cache does not end up with a second one.
pub fn cache_root() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("HF_HUB_CACHE") {
        return Some(PathBuf::from(p));
    }
    if let Some(p) = std::env::var_os("HF_HOME") {
        return Some(PathBuf::from(p).join("hub"));
    }
    dirs::home_dir().map(|h| h.join(".cache").join("huggingface").join("hub"))
}

/// Resolve `filename` from the local cache only.
///
/// `Err` means "not downloaded" rather than "broken": this is how the Models window answers
/// whether the checkpoint is present without starting a 7.17 GB transfer.
pub fn cached(repo_id: &str, filename: &str) -> Result<PathBuf, String> {
    let dir = cache_root()
        .ok_or("no home directory to hold the Hugging Face cache")?
        .join(repo_folder(repo_id)?);

    let commit = std::fs::read_to_string(dir.join("refs").join("main"))
        .map_err(|_| format!("{repo_id} is not in the local cache"))?;
    let path = dir
        .join("snapshots")
        .join(commit.trim())
        .join(filename.trim_start_matches('/'));

    // `is_file` follows the symlink, so a snapshot entry pointing at a blob that was pruned
    // reads as absent rather than as a path that fails later at open time.
    if path.is_file() {
        Ok(path)
    } else {
        Err(format!("{repo_id}/{filename} is not in the local cache"))
    }
}

pub use real::fetch_reporting;

mod real {
    use super::{cache_root, repo_folder};
    use std::io::{Read, Seek, Write};
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    /// What the Hub says about one file.
    struct Meta {
        /// Commit the snapshot belongs to.
        commit: String,
        /// Blob id: the LFS sha256, or the git sha1 for a plain file.
        blob: String,
        /// Size in bytes, or 0 when the Hub did not say.
        size: u64,
        /// Whether `blob` is a sha256 that can be verified after downloading.
        verifiable: bool,
    }

    /// Ask the Hub for a file's commit, blob id and size.
    fn meta(repo_id: &str, filename: &str) -> Result<Meta, String> {
        let url = format!("https://huggingface.co/api/models/{repo_id}?blobs=true");
        let body = ureq::get(&url)
            .call()
            .map_err(|e| format!("asking about {repo_id}: {e}"))?
            .into_body()
            .read_to_string()
            .map_err(|e| format!("reading the reply about {repo_id}: {e}"))?;

        let v: serde_json::Value =
            serde_json::from_str(&body).map_err(|e| format!("{repo_id} metadata: {e}"))?;

        let commit = v
            .get("sha")
            .and_then(|s| s.as_str())
            .ok_or_else(|| format!("{repo_id} metadata has no commit"))?
            .to_string();

        let sibling = v
            .get("siblings")
            .and_then(|s| s.as_array())
            .and_then(|all| {
                all.iter()
                    .find(|f| f.get("rfilename").and_then(|n| n.as_str()) == Some(filename))
            })
            .ok_or_else(|| format!("{repo_id} has no file called {filename}"))?;

        // An LFS file is addressed by the sha256 of its contents, which is exactly what we
        // want to verify against afterwards. A plain file is addressed by its git blob sha1,
        // which is a hash of the content *plus a header* — computable, but not worth it for
        // the small metadata files, so those are simply not verified.
        match sibling.get("lfs") {
            Some(lfs) => Ok(Meta {
                commit,
                blob: lfs
                    .get("sha256")
                    .and_then(|s| s.as_str())
                    .ok_or_else(|| format!("{filename} has no LFS sha256"))?
                    .to_string(),
                size: lfs.get("size").and_then(|s| s.as_u64()).unwrap_or(0),
                verifiable: true,
            }),
            None => Ok(Meta {
                commit,
                blob: sibling
                    .get("blobId")
                    .and_then(|s| s.as_str())
                    .ok_or_else(|| format!("{filename} has no blob id"))?
                    .to_string(),
                size: sibling.get("size").and_then(|s| s.as_u64()).unwrap_or(0),
                verifiable: false,
            }),
        }
    }

    /// Fetch `filename`, calling `progress(bytes_done, bytes_total)` as it arrives.
    ///
    /// Blocking; call from a worker thread. Reuses an already-cached blob without touching
    /// the network, resumes a partial transfer, and honours `cancel` between chunks.
    pub fn fetch_reporting(
        repo_id: &str,
        filename: &str,
        progress: Arc<dyn Fn(u64, u64) + Send + Sync>,
        cancel: &AtomicBool,
    ) -> Result<PathBuf, String> {
        // Already complete? Then this is free, and the common case on every launch.
        if let Ok(p) = super::cached(repo_id, filename) {
            return Ok(p);
        }

        let root = cache_root().ok_or("no home directory to hold the Hugging Face cache")?;
        let dir = root.join(repo_folder(repo_id)?);
        let meta = meta(repo_id, filename)?;

        let blobs = dir.join("blobs");
        std::fs::create_dir_all(&blobs)
            .map_err(|e| format!("creating {}: {e}", blobs.display()))?;
        let blob = blobs.join(&meta.blob);

        if !blob.is_file() {
            download(repo_id, filename, &meta, &blob, &progress, cancel)?;
        }
        link(&dir, &meta, filename, &blob)
    }

    /// Stream the file into `blob`, resuming if a partial transfer is already there.
    fn download(
        repo_id: &str,
        filename: &str,
        meta: &Meta,
        blob: &Path,
        progress: &Arc<dyn Fn(u64, u64) + Send + Sync>,
        cancel: &AtomicBool,
    ) -> Result<(), String> {
        let partial = blob.with_extension("incomplete");
        let have = std::fs::metadata(&partial).map(|m| m.len()).unwrap_or(0);

        // A leftover larger than the file itself means the partial belongs to something
        // else, or the file changed. Starting over is cheaper than reasoning about it.
        let have = if meta.size > 0 && have >= meta.size {
            let _ = std::fs::remove_file(&partial);
            0
        } else {
            have
        };

        let url = format!("https://huggingface.co/{repo_id}/resolve/main/{filename}");
        let mut request = ureq::get(&url);
        if have > 0 {
            request = request.header("Range", &format!("bytes={have}-"));
        }
        let response = request
            .call()
            .map_err(|e| format!("downloading {repo_id}/{filename}: {e}"))?;

        // A server that ignored the Range header sends 200 and the whole file; writing that
        // after the bytes we already have would corrupt the blob.
        let resuming = response.status() == 206;
        let start = if resuming { have } else { 0 };

        let total = response
            .headers()
            .get("content-length")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok())
            .map(|len| start + len)
            .filter(|t| *t > 0)
            .unwrap_or(meta.size);

        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(!resuming)
            .open(&partial)
            .map_err(|e| format!("opening {}: {e}", partial.display()))?;
        if resuming {
            file.seek(std::io::SeekFrom::End(0))
                .map_err(|e| format!("seeking {}: {e}", partial.display()))?;
        }

        let mut reader = response.into_body().into_reader();
        let mut buf = vec![0u8; 1024 * 1024];
        let mut done = start;
        progress(done, total);

        loop {
            if cancel.load(Ordering::Relaxed) {
                // The partial file stays: the next attempt resumes from here rather than
                // starting the 7.17 GB again.
                let _ = file.flush();
                return Err("cancelled".into());
            }
            let n = reader
                .read(&mut buf)
                .map_err(|e| format!("reading {repo_id}/{filename}: {e}"))?;
            if n == 0 {
                break;
            }
            file.write_all(&buf[..n])
                .map_err(|e| format!("writing {}: {e}", partial.display()))?;
            done += n as u64;
            progress(done, total);
        }
        file.flush()
            .map_err(|e| format!("flushing {}: {e}", partial.display()))?;
        drop(file);

        if meta.verifiable {
            verify(&partial, &meta.blob)?;
        }
        std::fs::rename(&partial, blob)
            .map_err(|e| format!("finishing {}: {e}", blob.display()))?;
        Ok(())
    }

    /// Check the downloaded bytes against the sha256 the Hub published for them.
    ///
    /// These weights are about to be executed. "The transfer ended without an error" is not
    /// the same as "the file is what it should be" — a truncated or corrupted blob would
    /// otherwise sit in the cache looking complete, and fail much later as a confusing
    /// model-load error.
    fn verify(path: &Path, expected: &str) -> Result<(), String> {
        use sha2::{Digest, Sha256};

        let mut file =
            std::fs::File::open(path).map_err(|e| format!("verifying {}: {e}", path.display()))?;
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; 1024 * 1024];
        loop {
            let n = file
                .read(&mut buf)
                .map_err(|e| format!("verifying {}: {e}", path.display()))?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
        let got = hasher.finalize().iter().fold(String::new(), |mut s, b| {
            use std::fmt::Write;
            let _ = write!(s, "{b:02x}");
            s
        });

        if got == expected {
            Ok(())
        } else {
            // Removed rather than left behind: a bad blob that survives would be resumed
            // from on the next attempt and fail identically forever.
            let _ = std::fs::remove_file(path);
            Err(format!(
                "checksum mismatch (expected {expected}, got {got}) — the download was discarded"
            ))
        }
    }

    /// Point `snapshots/{commit}/{filename}` at the blob, and record the ref.
    fn link(dir: &Path, meta: &Meta, filename: &str, blob: &Path) -> Result<PathBuf, String> {
        let refs = dir.join("refs");
        std::fs::create_dir_all(&refs).map_err(|e| format!("creating {}: {e}", refs.display()))?;
        std::fs::write(refs.join("main"), &meta.commit)
            .map_err(|e| format!("writing {}/main: {e}", refs.display()))?;

        let snapshot = dir.join("snapshots").join(&meta.commit);
        let link = snapshot.join(filename.trim_start_matches('/'));
        if let Some(parent) = link.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("creating {}: {e}", parent.display()))?;
        }
        if link.is_file() {
            return Ok(link);
        }
        // Stale symlink (points at a pruned blob): replace rather than fail.
        let _ = std::fs::remove_file(&link);

        // Relative, like the rest of the ecosystem writes them, so the cache survives being
        // moved or mounted at a different path.
        let target = relative_to_blob(&link, blob)?;
        symlink(&target, &link).map_err(|e| format!("linking {}: {e}", link.display()))?;
        Ok(link)
    }

    /// The `../../blobs/{id}` form used inside a snapshot directory.
    ///
    /// Counted rather than hard-coded to `../..`, because a `filename` containing a
    /// directory (`onnx/model.onnx`, say) sits one level deeper and needs one more `..`.
    fn relative_to_blob(link: &Path, blob: &Path) -> Result<PathBuf, String> {
        let repo_dir = blob
            .parent()
            .and_then(Path::parent)
            .ok_or("blob is not inside a repo directory")?;
        let depth = link
            .parent()
            .and_then(|p| p.strip_prefix(repo_dir).ok())
            .map(|rest| rest.components().count())
            .ok_or("could not place the snapshot relative to the blobs directory")?;

        let mut out = PathBuf::new();
        for _ in 0..depth {
            out.push("..");
        }
        out.push("blobs");
        out.push(blob.file_name().ok_or("blob has no file name")?);
        Ok(out)
    }

    #[cfg(unix)]
    fn symlink(target: &Path, link: &Path) -> std::io::Result<()> {
        std::os::unix::fs::symlink(target, link)
    }

    /// Windows needs a privilege for symlinks that a normal user may not have, so the blob
    /// is copied instead. It costs the disk space the layout exists to save, but a working
    /// cache beats an elegant one that needs an administrator.
    #[cfg(windows)]
    fn symlink(target: &Path, link: &Path) -> std::io::Result<()> {
        let absolute = link
            .parent()
            .map(|p| p.join(target))
            .ok_or_else(|| std::io::Error::other("snapshot has no parent"))?;
        std::os::windows::fs::symlink_file(&absolute, link).or_else(|_| {
            std::fs::copy(&absolute, link)?;
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_owner_and_name() {
        assert_eq!(split_repo("nvidia/model"), Ok(("nvidia", "model")));
        assert!(split_repo("nvidia").is_err());
        assert!(split_repo("/model").is_err());
        assert!(split_repo("nvidia/").is_err());
    }

    /// The folder name is how pstore's cache entries line up with everyone else's. Getting
    /// it wrong would not fail — it would quietly download a second copy of 7.17 GB beside
    /// the one another tool already has.
    #[test]
    fn repo_folders_match_the_hub_convention() {
        assert_eq!(
            repo_folder("prism-ml/Bonsai-27B-gguf").unwrap(),
            "models--prism-ml--Bonsai-27B-gguf"
        );
        assert!(repo_folder("nope").is_err());
    }

    /// An empty cache must read as "not downloaded", not as an error the UI shows as a
    /// failure — it is the state of every first run.
    #[test]
    fn a_missing_entry_is_reported_as_absent() {
        let why = cached("prism-ml/definitely-not-a-real-repo", "x.gguf").unwrap_err();
        assert!(why.contains("not in the local cache"), "got {why}");
    }
}
