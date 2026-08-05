//! The one place that knows how to pull a file from Hugging Face.
//!
//! Every classifier needs weights, a config and a tokenizer. Keeping the Hub call in a
//! single function means the (fairly involved) client API appears once, and downloads
//! reuse the shared on-disk cache automatically.
//!
//! Three entry points, because the app needs to ask three different questions:
//! [`cached`] ("is it already here?", never touches the network), [`fetch`] ("give me the
//! path, downloading if you must") and [`fetch_reporting`] (the same, but telling the
//! caller how it is going so a progress bar can move).

// Only the download paths deal in paths, and those need the Hub client.
#[cfg(feature = "candle")]
use std::path::PathBuf;

/// Split a `owner/name` repo id.
pub fn split_repo(repo_id: &str) -> Result<(&str, &str), String> {
    repo_id
        .split_once('/')
        .filter(|(o, n)| !o.is_empty() && !n.is_empty())
        .ok_or_else(|| format!("{repo_id:?} is not an owner/name repository id"))
}

#[cfg(feature = "candle")]
mod real {
    use super::{PathBuf, split_repo};

    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    use hf_hub::progress::{DownloadEvent, ProgressEvent, ProgressHandler};

    /// Fetch `filename` from `repo_id`, returning its path in the local cache.
    ///
    /// Blocking, and the first call for a given file downloads it — call from a worker
    /// thread, never from the UI thread.
    pub fn fetch(repo_id: &str, filename: &str) -> Result<PathBuf, String> {
        download(repo_id, filename, false, None)
    }

    /// Resolve `filename` from the local cache only.
    ///
    /// `Err` means "not downloaded" rather than "broken": this is how the Models window
    /// answers whether a checkpoint is present without starting a 795 MB transfer.
    pub fn cached(repo_id: &str, filename: &str) -> Result<PathBuf, String> {
        download(repo_id, filename, true, None)
    }

    /// Fetch `filename`, calling `progress(bytes_done, bytes_total)` as it arrives.
    ///
    /// `bytes_total` is 0 until the Hub reports the size. A cache hit may report nothing
    /// at all before returning, so callers must treat "no progress events" as success
    /// rather than as a stall.
    ///
    /// The callback is shared rather than borrowed because hf-hub keeps the handler in an
    /// `Arc` and may call it from its own threads; it is invoked on the transfer's read
    /// path, so it must return promptly.
    pub fn fetch_reporting(
        repo_id: &str,
        filename: &str,
        progress: Arc<dyn Fn(u64, u64) + Send + Sync>,
    ) -> Result<PathBuf, String> {
        download(
            repo_id,
            filename,
            false,
            Some(Reporter {
                progress,
                total: AtomicU64::new(0),
            }),
        )
    }

    fn download(
        repo_id: &str,
        filename: &str,
        local_only: bool,
        handler: Option<Reporter>,
    ) -> Result<PathBuf, String> {
        let (owner, name) = split_repo(repo_id)?;
        let client = hf_hub::HFClientSync::new()
            .map_err(|e| format!("could not reach the Hugging Face cache: {e}"))?;
        client
            .model(owner, name)
            .download_file()
            .filename(filename)
            .local_files_only(local_only)
            .maybe_progress(handler.map(hf_hub::progress::Progress::new))
            .send()
            .map_err(|e| format!("{repo_id}/{filename}: {e}"))
    }

    /// Turns hf-hub's download events into `(done, total)` byte pairs.
    struct Reporter {
        progress: Arc<dyn Fn(u64, u64) + Send + Sync>,
        /// Remembered from `Start`, because per-file `Progress` events may report a
        /// total of 0.
        total: AtomicU64,
    }

    impl Reporter {
        fn emit(&self, done: u64, total: u64) {
            let total = if total > 0 {
                self.total.store(total, Ordering::Relaxed);
                total
            } else {
                self.total.load(Ordering::Relaxed)
            };
            (self.progress)(done, total);
        }
    }

    impl ProgressHandler for Reporter {
        fn on_progress(&self, event: &ProgressEvent) {
            match event {
                ProgressEvent::Download(DownloadEvent::Start { total_bytes, .. }) => {
                    self.emit(0, *total_bytes);
                }
                // One file per call, so the last delta is the whole story.
                ProgressEvent::Download(DownloadEvent::Progress { files }) => {
                    if let Some(f) = files.last() {
                        self.emit(f.bytes_completed, f.total_bytes);
                    }
                }
                ProgressEvent::Download(DownloadEvent::AggregateProgress {
                    bytes_completed,
                    total_bytes,
                    ..
                }) => {
                    self.emit(*bytes_completed, *total_bytes);
                }
                ProgressEvent::Download(DownloadEvent::Complete) => {
                    let total = self.total.load(Ordering::Relaxed);
                    self.emit(total, total);
                }
                ProgressEvent::Upload(_) => {}
            }
        }
    }
}

#[cfg(feature = "candle")]
pub use real::{cached, fetch, fetch_reporting};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_owner_and_name() {
        assert_eq!(
            split_repo("regolo/brick-eco").unwrap(),
            ("regolo", "brick-eco")
        );
        // Extra slashes belong to the name, not to a third component.
        assert_eq!(split_repo("a/b/c").unwrap(), ("a", "b/c"));
    }

    #[test]
    fn rejects_malformed_ids() {
        for bad in ["", "no-slash", "/name", "owner/"] {
            assert!(split_repo(bad).is_err(), "{bad:?} should be rejected");
        }
    }

    /// A cache lookup must answer without a network round-trip, and must not invent a
    /// path for something that was never downloaded.
    #[test]
    #[cfg(feature = "candle")]
    fn cache_lookup_is_local_and_honest() {
        let started = std::time::Instant::now();
        let missing = cached(
            "pstore-test/definitely-not-a-real-repo",
            "model.safetensors",
        );
        assert!(missing.is_err(), "got {missing:?}");
        assert!(
            started.elapsed() < std::time::Duration::from_secs(5),
            "a cache probe must not wait on the network"
        );
    }
}
