//! Finding — and if necessary fetching — the `llama-cli` that runs the checkpoint.
//!
//! pstore does not link an inference engine. It runs the model the same way it runs a
//! coding agent: as a child process. That process has to come from somewhere, and asking
//! the user to clone and build a C++ project before the app works would make local
//! inference an expert feature. So pstore provisions it.
//!
//! It has to be **PrismML's fork** of llama.cpp, not upstream: the Bonsai checkpoints are
//! quantised with a `Q1_0_g128` scheme whose kernels exist only there. Upstream
//! `llama-cli` loads the file and fails on the tensor types. The fork publishes prebuilt
//! binaries for every platform pstore targets, which is what makes hands-off provisioning
//! possible at all.
//!
//! The release tag is **pinned** rather than resolved to "latest". A new upstream build
//! could change a flag name or a kernel and break inference for everyone on the next
//! launch; upgrading is a code change that can be tested, not something that happens to a
//! user overnight.

use std::path::PathBuf;

use crate::models::Phase;

/// The llama.cpp fork release pstore is built and tested against.
///
/// Bumping this means re-checking the flags in [`crate::router::llm`] — they have been
/// renamed upstream before.
pub const RELEASE_TAG: &str = "prism-b9599-9ca265a";

/// Where the pinned release lives.
pub const RELEASE_URL: &str = "https://github.com/PrismML-Eng/llama.cpp/releases";

/// The binary pstore looks for and runs.
///
/// **Not** `llama-cli`. That one refuses `--no-conversation` at runtime — despite listing it
/// in `--help` — and says to use `llama-completion` instead. `llama-completion` is the
/// non-interactive one, which is all pstore ever wants.
pub const BINARY: &str = if cfg!(windows) {
    "llama-completion.exe"
} else {
    "llama-completion"
};

/// One platform's prebuilt archive, with the digest GitHub publishes for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Asset {
    /// File name within the release.
    pub name: &'static str,
    /// Lowercase hex SHA256, as published alongside the asset.
    pub sha256: &'static str,
    /// Download size, so the Models window can say what it will cost.
    pub bytes: u64,
}

impl Asset {
    /// Full download URL for the pinned release.
    pub fn url(&self) -> String {
        format!("{RELEASE_URL}/download/{RELEASE_TAG}/{}", self.name)
    }
}

/// The asset for the host platform, or why there isn't one.
///
/// Accelerated Linux builds (CUDA, ROCm, Vulkan) exist too, but picking between them means
/// probing the machine's driver stack, and guessing wrong yields a binary that fails to
/// start rather than one that runs slowly. The portable build always works; a user with a
/// GPU can point `llama_cli_path` at an accelerated build they chose themselves.
pub fn asset() -> Result<Asset, String> {
    // macOS: Metal is in the standard arm64 build, no separate variant needed.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    return Ok(Asset {
        name: "llama-prism-b9599-9ca265a-bin-macos-arm64.tar.gz",
        sha256: "0452e5ffbab947b54cb03abbafe3ab9e4745e0aba2d9fabcf83440349e7b1cd0",
        bytes: 11_200_000,
    });

    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    return Ok(Asset {
        name: "llama-prism-b9599-9ca265a-bin-macos-x64.tar.gz",
        sha256: "7c6179a43265c0f24c7336c976b5aae81589b33cb979543421d538ae25175d88",
        bytes: 11_200_000,
    });

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    return Ok(Asset {
        name: "llama-prism-b9599-9ca265a-bin-ubuntu-x64.tar.gz",
        sha256: "35e935c828f58f9837d3da357a86846deb11d4feffc835d9e94e7d8fc1bbd55e",
        bytes: 16_100_000,
    });

    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    return Ok(Asset {
        name: "llama-prism-b9599-9ca265a-bin-ubuntu-arm64.tar.gz",
        sha256: "885679964ff476cb626f2e70d593f9a12c5fedb253260dc100d31df44ba68b81",
        bytes: 13_100_000,
    });

    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    return Ok(Asset {
        name: "llama-bin-win-cpu-x64.zip",
        sha256: "9253ff142d5d08cc4f80e22893a44a80e088ae0589dfa47acc2e5548a4f6818f",
        bytes: 17_400_000,
    });

    #[cfg(all(target_os = "windows", target_arch = "aarch64"))]
    return Ok(Asset {
        name: "llama-bin-win-cpu-arm64.zip",
        sha256: "e88c0d35344c052cae97ccff921508e18bdc9995c6f88bf790f3366fade192f1",
        bytes: 11_200_000,
    });

    #[allow(unreachable_code)]
    Err(format!(
        "no prebuilt llama-cli for {}-{}; build the PrismML fork yourself and set \
         `llama_cli_path` in .pstore/config.json",
        std::env::consts::OS,
        std::env::consts::ARCH
    ))
}

/// Where a working `llama-cli` was found.
///
/// Shown in the Models window: provisioning that cannot be inspected is indistinguishable
/// from magic, and "which binary is this actually running?" is the first question when
/// inference misbehaves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    /// `llama_cli_path` in the config.
    Override,
    /// A machine-wide install an administrator provisioned.
    System,
    /// pstore's own assets directory.
    Managed,
    /// Found on `PATH`.
    Path,
}

impl std::fmt::Display for Origin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Origin::Override => "configured path",
            Origin::System => "system install",
            Origin::Managed => "downloaded by pstore",
            Origin::Path => "found on PATH",
        })
    }
}

/// A resolved runtime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Runtime {
    /// Absolute path to the binary.
    pub path: PathBuf,
    /// How it was found.
    pub origin: Origin,
}

/// pstore's own assets directory, where a downloaded runtime is kept.
///
/// Per-user rather than machine-wide. A machine-wide path would have to be written as
/// root, and a GUI app asking for a password to unpack 11 MB is worse than keeping its own
/// copy — [`system_dir`] still *reads* a shared install when an administrator provisioned
/// one.
pub fn managed_dir() -> Option<PathBuf> {
    dirs::data_dir().map(|d| d.join("pstore").join("bin"))
}

/// A machine-wide install, if this platform has a conventional place for one.
fn system_dir() -> Option<PathBuf> {
    #[cfg(target_os = "macos")]
    return Some(PathBuf::from("/Library/Application Support/pstore/bin"));
    #[cfg(target_os = "linux")]
    return Some(PathBuf::from("/usr/local/share/pstore/bin"));
    #[cfg(target_os = "windows")]
    return std::env::var_os("PROGRAMDATA").map(|p| PathBuf::from(p).join("pstore").join("bin"));
    #[allow(unreachable_code)]
    None
}

/// Whether `p` is a file this process could execute.
fn is_executable(p: &std::path::Path) -> bool {
    if !p.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::metadata(p).is_ok_and(|m| m.permissions().mode() & 0o111 != 0)
    }
    // Windows has no execute bit; being a file is as much as can be checked here.
    #[cfg(not(unix))]
    true
}

/// Find `llama-cli` without touching the network.
///
/// Order is deliberate: an explicit override always wins, then a machine-wide install
/// (an administrator chose it for everyone), then pstore's own copy, then `PATH` last —
/// a `llama-cli` on `PATH` is most likely stock llama.cpp, which cannot load these
/// weights, so it is a fallback rather than a preference.
pub fn locate(override_path: Option<&str>) -> Option<Runtime> {
    if let Some(p) = override_path.filter(|p| !p.trim().is_empty()) {
        let path = PathBuf::from(p);
        if is_executable(&path) {
            return Some(Runtime {
                path,
                origin: Origin::Override,
            });
        }
        // A configured path that does not resolve is a mistake worth surfacing rather than
        // silently papering over with a different binary.
        return None;
    }

    for (dir, origin) in [
        (system_dir(), Origin::System),
        (managed_dir(), Origin::Managed),
    ] {
        if let Some(candidate) = dir.map(|d| d.join(BINARY)).filter(|p| is_executable(p)) {
            return Some(Runtime {
                path: candidate,
                origin,
            });
        }
    }

    which_on_path().map(|path| Runtime {
        path,
        origin: Origin::Path,
    })
}

/// Search `PATH` for the binary.
fn which_on_path() -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|d| d.join(BINARY))
        .find(|p| is_executable(p))
}

/// Why the runtime is not usable, phrased so the Models window can show it verbatim.
pub fn missing_reason(override_path: Option<&str>) -> String {
    match override_path.filter(|p| !p.trim().is_empty()) {
        Some(p) => format!("`llama_cli_path` is set to {p:?}, which is not an executable file"),
        None => format!(
            "{BINARY} not installed — pstore can download it ({}), or set `llama_cli_path`",
            match asset() {
                Ok(a) => crate::models::bytes_label(a.bytes),
                Err(_) => "unavailable for this platform".into(),
            }
        ),
    }
}

#[cfg(feature = "local-llm")]
pub use provision::{download, progress};

/// Provisioning never starts in a build with no local inference, so there is nothing to
/// report on.
#[cfg(not(feature = "local-llm"))]
pub fn progress() -> Phase {
    Phase::Unknown
}

#[cfg(feature = "local-llm")]
mod provision {
    use super::{Asset, BINARY, Origin, PathBuf, Phase, Runtime, asset, managed_dir};
    use std::io::Read;
    use std::sync::{Mutex, OnceLock};

    fn slot() -> &'static Mutex<Phase> {
        static SLOT: OnceLock<Mutex<Phase>> = OnceLock::new();
        SLOT.get_or_init(|| Mutex::new(Phase::Unknown))
    }

    fn set(p: Phase) {
        match slot().lock() {
            Ok(mut g) => *g = p,
            Err(poisoned) => *poisoned.into_inner() = p,
        }
    }

    /// Live provisioning state, polled by the Models window once per frame.
    ///
    /// Separate from [`crate::models`]'s board, which tracks *checkpoints* — the runtime is
    /// a different kind of thing, and giving it a fake checkpoint row would put a binary in
    /// a catalogue of weights.
    pub fn progress() -> Phase {
        match slot().lock() {
            Ok(g) => g.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        }
    }

    /// Download, verify and unpack the pinned `llama-cli` into pstore's assets directory.
    ///
    /// Blocking; call from a worker thread. Reports into [`PROGRESS`] as bytes arrive.
    /// Returns the installed runtime, or the reason it could not be installed.
    ///
    /// `cancel` is honoured between chunks, so stopping is prompt — unlike the weight
    /// download, this transfer is small enough to restart from scratch.
    pub fn download(cancel: &std::sync::atomic::AtomicBool) -> Result<Runtime, String> {
        use std::sync::atomic::Ordering;

        let asset = asset()?;
        let dir = managed_dir().ok_or("no user data directory to install into")?;

        set(Phase::Fetching {
            file: "llama-cli",
            done: 0,
            total: asset.bytes,
            files_done: 0,
        });

        let bytes = match fetch(&asset, cancel) {
            Ok(b) => b,
            Err(e) => {
                set(if cancel.load(Ordering::Relaxed) {
                    Phase::Absent
                } else {
                    Phase::Failed(e.clone())
                });
                return Err(e);
            }
        };

        set(Phase::Loading);
        let installed = verify(&asset, &bytes).and_then(|()| unpack(&bytes, &dir));
        match installed {
            Ok(path) => {
                set(Phase::Ready);
                Ok(Runtime {
                    path,
                    origin: Origin::Managed,
                })
            }
            Err(e) => {
                set(Phase::Failed(e.clone()));
                Err(e)
            }
        }
    }

    /// Stream the asset into memory, reporting progress.
    ///
    /// Held in memory rather than spooled to disk: the largest asset pstore selects is
    /// ~17 MB, and a partial file on disk is a thing that can be found later and executed.
    fn fetch(asset: &Asset, cancel: &std::sync::atomic::AtomicBool) -> Result<Vec<u8>, String> {
        use std::sync::atomic::Ordering;

        let url = asset.url();
        let response = ureq::get(&url)
            .call()
            .map_err(|e| format!("downloading {}: {e}", asset.name))?;

        let total = response
            .headers()
            .get("content-length")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(asset.bytes);

        let mut reader = response.into_body().into_reader();
        let mut out = Vec::with_capacity(total as usize);
        let mut buf = [0u8; 64 * 1024];
        loop {
            if cancel.load(Ordering::Relaxed) {
                return Err("cancelled".into());
            }
            let n = reader
                .read(&mut buf)
                .map_err(|e| format!("reading {}: {e}", asset.name))?;
            if n == 0 {
                break;
            }
            out.extend_from_slice(&buf[..n]);
            set(Phase::Fetching {
                file: "llama-cli",
                done: out.len() as u64,
                total,
                files_done: 0,
            });
        }
        Ok(out)
    }

    /// Check the download against the digest GitHub publishes for it.
    ///
    /// This binary is about to be executed on the user's machine, so "it downloaded
    /// without an error" is not the bar. A mismatch is a hard failure, never a warning.
    fn verify(asset: &Asset, bytes: &[u8]) -> Result<(), String> {
        use sha2::{Digest, Sha256};

        let got = Sha256::digest(bytes);
        let got = got.iter().fold(String::new(), |mut s, b| {
            use std::fmt::Write;
            let _ = write!(s, "{b:02x}");
            s
        });
        if got == asset.sha256 {
            Ok(())
        } else {
            Err(format!(
                "{} failed its checksum (expected {}, got {got}) — refusing to install it",
                asset.name, asset.sha256
            ))
        }
    }

    /// Unpack the archive and return the path to the installed binary.
    ///
    /// Only `llama-cli` and the shared libraries beside it are extracted; the release also
    /// carries a dozen other tools pstore never runs. Entries are flattened into `dir`
    /// after their file name is checked, so a crafted archive cannot write outside it.
    fn unpack(bytes: &[u8], dir: &std::path::Path) -> Result<PathBuf, String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("creating {}: {e}", dir.display()))?;
        // The binary plus every shared library beside it. `libllama-completion-impl.dylib`
        // in particular is not optional: the binary is a thin shim over it.
        let wanted = |name: &str| {
            name == BINARY
                || name.ends_with(".so")
                || name.ends_with(".dylib")
                || name.ends_with(".dll")
                || name.ends_with(".metal")
                || name.ends_with(".metallib")
        };

        #[cfg(not(target_os = "windows"))]
        {
            let decoder = flate2::read::GzDecoder::new(bytes);
            let mut archive = tar::Archive::new(decoder);
            for entry in archive
                .entries()
                .map_err(|e| format!("reading the archive: {e}"))?
            {
                let mut entry = entry.map_err(|e| format!("reading the archive: {e}"))?;
                let path = entry
                    .path()
                    .map_err(|e| format!("reading the archive: {e}"))?
                    .into_owned();
                let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                    continue;
                };
                if !wanted(name) {
                    continue;
                }
                let dest = dir.join(name);
                entry
                    .unpack(&dest)
                    .map_err(|e| format!("unpacking {name}: {e}"))?;
            }
        }

        #[cfg(target_os = "windows")]
        {
            let cursor = std::io::Cursor::new(bytes);
            let mut archive =
                zip::ZipArchive::new(cursor).map_err(|e| format!("reading the archive: {e}"))?;
            for i in 0..archive.len() {
                let mut entry = archive
                    .by_index(i)
                    .map_err(|e| format!("reading the archive: {e}"))?;
                let Some(name) = entry
                    .enclosed_name()
                    .and_then(|p| p.file_name().map(|n| n.to_owned()))
                    .and_then(|n| n.to_str().map(str::to_owned))
                else {
                    continue;
                };
                if !wanted(&name) {
                    continue;
                }
                let mut out = std::fs::File::create(dir.join(&name))
                    .map_err(|e| format!("writing {name}: {e}"))?;
                std::io::copy(&mut entry, &mut out).map_err(|e| format!("writing {name}: {e}"))?;
            }
        }

        let binary = dir.join(BINARY);
        if !binary.is_file() {
            return Err(format!(
                "the release archive did not contain {BINARY} — the pinned asset may have \
                 changed shape"
            ));
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(0o755))
                .map_err(|e| format!("making {BINARY} executable: {e}"))?;
        }
        Ok(binary)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pinned asset has to name a real file in a real release, and carry a digest that
    /// could actually match one. Both are transcribed by hand from the GitHub release, and
    /// a typo would only surface as a failed install on a user's first run.
    #[test]
    fn the_pinned_asset_is_well_formed() {
        let Ok(a) = asset() else {
            // An unsupported platform is a legitimate outcome; nothing to check.
            return;
        };
        assert!(
            a.name.contains(RELEASE_TAG) || a.name.starts_with("llama-bin-win"),
            "{} does not belong to release {RELEASE_TAG}",
            a.name
        );
        assert_eq!(a.sha256.len(), 64, "a SHA256 is 64 hex characters");
        assert!(
            a.sha256
                .chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
            "digest should be lowercase hex: {}",
            a.sha256
        );
        assert!(a.bytes > 1_000_000, "asset size looks wrong");
        assert!(
            a.url().starts_with("https://"),
            "downloads must be over TLS"
        );
        assert!(a.url().contains(RELEASE_TAG));
    }

    /// An override that does not resolve must not silently fall through to some other
    /// binary: the user asked for a specific one, and running a different one would make
    /// the setting look broken in a way that is very hard to diagnose.
    #[test]
    fn a_bad_override_does_not_fall_through() {
        assert!(locate(Some("/definitely/not/a/real/llama-cli")).is_none());
        // Blank is treated as unset rather than as a broken path.
        let blank = locate(Some("   "));
        assert_eq!(blank.is_some(), locate(None).is_some());
    }

    #[test]
    fn missing_reason_names_the_setting() {
        let configured = missing_reason(Some("/nope/llama-cli"));
        assert!(configured.contains("/nope/llama-cli"));
        assert!(configured.contains("llama_cli_path"));

        let absent = missing_reason(None);
        assert!(absent.contains(BINARY), "got {absent}");
    }

    #[test]
    fn origins_describe_themselves_distinctly() {
        let all = [
            Origin::Override,
            Origin::System,
            Origin::Managed,
            Origin::Path,
        ];
        let mut seen: Vec<String> = all.iter().map(|o| o.to_string()).collect();
        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), all.len());
    }
}
