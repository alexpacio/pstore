//! The checkpoint as a background service, for other tools to use.
//!
//! Everything else in pstore runs the model as a one-shot subprocess: prompt in, JSON out,
//! process exits. That is right for pstore's own calls, which are one per user action and
//! want nothing left running. It is wrong for a coding agent, which makes many calls in a
//! session and would pay the ~1.1 s of start-up and — far worse — re-evaluate the whole
//! prompt every turn, at ~10 ms a token.
//!
//! So this module runs the *same* weights, from the *same* release, behind `llama-server`:
//! resident, HTTP, OpenAI-compatible. An agent points at `http://127.0.0.1:<port>/v1` and
//! uses the model already sitting on this disk instead of a vendor's API. Nothing about the
//! prompt leaves the machine, which is the same promise pstore's own inference makes.
//!
//! **It is off by default and explicit to start.** A resident 27B is 5–8.4 GB of RAM held
//! until it is stopped, which is not something to switch on behind someone's back. The button
//! is in the Models window and the state is visible there for as long as it runs.
//!
//! **One build at a time, still.** The server holds weights resident by definition, so it is
//! the one thing in pstore that can violate "never two builds at once" for an unbounded
//! period. Switching build therefore stops it — see [`crate::router::unload_other_model_builds`]
//! — rather than leaving 7.17 GB of the build you just left mapped for the rest of the
//! session.
//!
//! **Nothing outlives the window.** Like every other child, the server is killed on the way
//! out. A resident model with no window left to stop it is the worst orphan pstore could
//! leave, because it would keep the memory *and* the port.

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use crate::models;

/// Where the server binds. Loopback only, never `0.0.0.0`.
///
/// The model runs on this machine for the same reason pstore's own inference does, and a
/// listener on every interface would quietly turn a privacy property into a service anyone on
/// the network could reach. Users who want that can run `llama-server` themselves and mean it.
pub const HOST: &str = "127.0.0.1";

/// How long to wait for the server to answer `/health` before giving up.
///
/// Generous: this covers mapping up to 7.17 GB of weights, which on a cold page cache is disk
/// I/O rather than work. Measured warm it is a couple of seconds.
const READY_TIMEOUT: Duration = Duration::from_secs(180);

/// How often to ask, while waiting.
const POLL: Duration = Duration::from_millis(200);

/// A running server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Serving {
    /// Port it is listening on.
    pub port: u16,
    /// The checkpoint whose weights it holds.
    pub checkpoint: &'static str,
    /// Its process id, so the Models window can say what to kill if all else fails.
    pub pid: u32,
}

impl Serving {
    /// The OpenAI-compatible base URL an agent should be pointed at.
    pub fn base_url(&self) -> String {
        format!("http://{HOST}:{}/v1", self.port)
    }

    /// The model name the endpoint advertises, which is the weights file's own name.
    ///
    /// `llama-server` names the model after the file it loaded, and agents send that string
    /// back in the `model` field. Guessing it wrong is a 404 at the first turn, so it is
    /// derived from the checkpoint rather than typed out anywhere.
    pub fn model_name(&self) -> String {
        models::ALL
            .iter()
            .find(|c| c.id == self.checkpoint)
            .and_then(|c| c.files.last())
            .map(|f| (*f).to_string())
            .unwrap_or_else(|| "bonsai".into())
    }

    /// Human title of the build being served.
    pub fn title(&self) -> &'static str {
        models::ALL
            .iter()
            .find(|c| c.id == self.checkpoint)
            .map(|c| c.title)
            .unwrap_or("the local model")
    }
}

/// The server, if one is running. At most one, ever.
static SERVER: Mutex<Option<(Child, Serving)>> = Mutex::new(None);

/// A poisoned lock means a thread panicked mid-update. Recovering the guard is better than
/// losing the ability to stop a resident 27B.
fn recover<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    match m.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// What is running now, if anything.
///
/// Reaps first: a server that died on its own — an occupied port, a bad checkpoint — must not
/// go on being reported as running, or the Models window offers to stop something that is
/// already gone and never offers to start it again.
pub fn status() -> Option<Serving> {
    let mut slot = recover(&SERVER);
    if let Some((child, _)) = slot.as_mut()
        && matches!(child.try_wait(), Ok(Some(_)) | Err(_))
    {
        *slot = None;
    }
    slot.as_ref().map(|(_, s)| s.clone())
}

/// The base URL of the running server, if there is one.
pub fn base_url() -> Option<String> {
    status().map(|s| s.base_url())
}

/// Start the selected build as a background service, and wait until it answers.
///
/// Blocking — call it from a worker thread. Returns once `/health` is `ok`, so a caller that
/// gets `Ok` can point an agent at the endpoint immediately rather than racing the load.
///
/// Starting when a server is already up is not an error: the running one is returned, because
/// the caller's intent — "I want it serving" — is already satisfied. Starting a *different*
/// build replaces the running one, which is the invariant rather than a convenience.
#[cfg(feature = "local-llm")]
pub fn start() -> Result<Serving, String> {
    let prefs = crate::config::prefs_snapshot();
    let checkpoint = prefs.local_model.checkpoint();

    if let Some(running) = status() {
        if running.checkpoint == checkpoint.id {
            return Ok(running);
        }
        // Two builds resident at once is the one thing this must never do.
        stop();
    }

    let binary =
        crate::runtime::locate_server(prefs.llama_cli_path.as_deref()).ok_or_else(|| {
            format!(
                "{} is not installed beside the runtime — re-download the runtime from the \
                 Models window",
                crate::runtime::SERVER_BINARY
            )
        })?;
    if !models::is_cached(&checkpoint) {
        return Err(format!(
            "{} not downloaded — fetch it before serving it ({})",
            checkpoint.title,
            checkpoint.size_label()
        ));
    }
    let weights = weights_path(&checkpoint)?;
    let port = prefs.model_server_port;

    let child = Command::new(&binary)
        .arg("-m")
        .arg(&weights)
        .args(["--host", HOST])
        .args(["--port", &port.to_string()])
        // The model's own chat template, for the same reason every other call uses it: this
        // checkpoint is a thinking model, and served through the legacy ChatML guess it
        // answers in a shape it was not trained on. With `--jinja` the server also splits the
        // reasoning into `reasoning_content` and leaves `content` as the answer, which is
        // exactly what an agent wants to render.
        .arg("--jinja")
        .args(["-ngl", "999"])
        .args(["-c", &prefs.model_server_context.to_string()])
        // Same memory settings as the one-shot path, and they matter more here: this process
        // stays up, so its KV cache is resident for as long as the session lasts.
        .args(["--cache-type-k", "q4_0", "--cache-type-v", "q4_0"])
        .args(["--flash-attn", "on"])
        .stdin(Stdio::null())
        // Discarded rather than piped. Nothing drains these pipes for the life of a resident
        // process, and a full pipe would wedge the server mid-session — the one failure mode
        // that would look like the model hanging.
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("starting {}: {e}", binary.display()))?;

    let serving = Serving {
        port,
        checkpoint: checkpoint.id,
        pid: child.id(),
    };
    *recover(&SERVER) = Some((child, serving.clone()));
    models::set(checkpoint.id, models::Phase::Loading);

    match wait_until_ready(port) {
        Ok(()) => {
            models::set(checkpoint.id, models::Phase::Ready);
            Ok(serving)
        }
        Err(e) => {
            // A server that never came up must not be left holding a port and some memory
            // while the board says it failed.
            stop();
            models::set(checkpoint.id, models::Phase::Failed(e.clone()));
            Err(e)
        }
    }
}

/// Without local inference there is nothing to serve.
#[cfg(not(feature = "local-llm"))]
pub fn start() -> Result<Serving, String> {
    Err(crate::models::NO_LOCAL_INFERENCE.to_string())
}

/// Poll `/health` until the server answers or the timeout runs out.
///
/// The distinction that matters here is "not up yet" against "up and broken". A refused
/// connection is the former and worth retrying; the process having exited is the latter, and
/// waiting the full three minutes on a server that is already gone helps nobody.
#[cfg(feature = "local-llm")]
fn wait_until_ready(port: u16) -> Result<(), String> {
    let url = format!("http://{HOST}:{port}/health");
    let started = Instant::now();

    while started.elapsed() < READY_TIMEOUT {
        if let Some((child, _)) = recover(&SERVER).as_mut()
            && matches!(child.try_wait(), Ok(Some(_)) | Err(_))
        {
            return Err(
                "the server exited while starting — the port may be in use, or the \
                 checkpoint may not load with this runtime"
                    .into(),
            );
        }
        if ureq::get(&url).call().is_ok() {
            return Ok(());
        }
        std::thread::sleep(POLL);
    }
    Err(format!(
        "the server did not answer on port {port} within {}s",
        READY_TIMEOUT.as_secs()
    ))
}

/// Resolve the checkpoint's weights in the shared Hugging Face cache.
fn weights_path(checkpoint: &models::Checkpoint) -> Result<PathBuf, String> {
    let file = checkpoint
        .files
        .last()
        .expect("the checkpoint lists its weights");
    crate::router::hub::cached(checkpoint.repo, file)
}

/// Stop the server, if one is running. Returns whether there was one.
///
/// Killed and reaped here rather than left to exit on its own: a zombie holds its slot, and
/// the port stays bound until the process is really gone — which is the difference between
/// "stopped" and "cannot be restarted".
pub fn stop() -> bool {
    let Some((mut child, serving)) = recover(&SERVER).take() else {
        return false;
    };
    let _ = child.kill();
    let _ = child.wait();
    // Weights on disk, nothing running: what is true the moment after.
    models::set(serving.checkpoint, models::Phase::Cached);
    true
}

/// Stop the server unless it is serving `keep`.
///
/// The build-switch path. A resident server is the one thing that can hold the weights of a
/// build the user has moved away from for an unbounded time, so the switch has to reach it.
pub fn stop_unless(keep: &str) -> bool {
    match status() {
        Some(s) if s.checkpoint != keep => stop(),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Nothing running is a state, not a failure: every accessor has to say so plainly, and
    /// stopping nothing has to be safe — the app calls it on the way out regardless.
    #[test]
    fn an_idle_server_reports_itself_as_idle() {
        // This test never starts one; if another test did, it is not this test's to assert on.
        if status().is_none() {
            assert_eq!(base_url(), None);
            assert!(!stop(), "stopping nothing should report nothing stopped");
            assert!(!stop_unless("llm-ternary"));
        }
    }

    /// The endpoint has to be loopback and OpenAI-shaped, because that string is pasted into
    /// other tools' configs. `0.0.0.0` here would turn a local-only promise into a service on
    /// the network.
    #[test]
    fn the_endpoint_is_loopback_and_openai_shaped() {
        let s = Serving {
            port: 8787,
            checkpoint: models::LLM_TERNARY.id,
            pid: 1,
        };
        assert_eq!(s.base_url(), "http://127.0.0.1:8787/v1");
        assert!(s.base_url().starts_with("http://127.0.0.1:"));
        assert!(s.base_url().ends_with("/v1"));
        assert_eq!(
            HOST, "127.0.0.1",
            "the server must not bind every interface"
        );
    }

    /// End-to-end against the real server: start it, use it the way an agent would, wire an
    /// agent's config to it, and stop it. Ignored by default — it maps the whole checkpoint.
    ///
    /// `cargo test -- --ignored live_server --nocapture`
    ///
    /// This is the test that proves the claim the Models window makes: that what comes up is
    /// an OpenAI-compatible endpoint a coding agent can actually talk to.
    #[test]
    #[ignore = "needs a downloaded checkpoint and a provisioned runtime"]
    #[cfg(feature = "local-llm")]
    fn live_server_answers_an_openai_request() {
        let checkpoint = crate::config::prefs_snapshot().local_model.checkpoint();
        if !models::is_cached(&checkpoint) {
            eprintln!("{} is not downloaded — nothing to serve", checkpoint.title);
            return;
        }

        let s = match start() {
            Ok(s) => s,
            Err(e) => {
                // A busy port is an environment problem, not a failing assertion.
                eprintln!("could not start: {e}");
                return;
            }
        };
        eprintln!("serving {} at {}", s.title(), s.base_url());
        assert_eq!(
            status().as_ref(),
            Some(&s),
            "it should report itself running"
        );

        // Starting again is a no-op rather than a second 7 GB mapping.
        assert_eq!(
            start().unwrap(),
            s,
            "a second start must not spawn a second server"
        );

        // Exactly the shape an agent sends: chat completions, by the model name the endpoint
        // advertises. If this 404s, every wiring this module writes is wrong.
        let body = serde_json::json!({
            "model": s.model_name(),
            "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
            "max_tokens": 400,
        });
        // Sent as a raw body rather than through ureq's `json` feature: enabling a dependency
        // feature for one test would put it in the shipped binary too.
        let text = ureq::post(&format!("{}/chat/completions", s.base_url()))
            .header("Content-Type", "application/json")
            .send(body.to_string())
            .expect("the endpoint should answer")
            .body_mut()
            .read_to_string()
            .expect("a readable reply");
        let reply: serde_json::Value = serde_json::from_str(&text).expect("a JSON reply");

        let message = &reply["choices"][0]["message"];
        eprintln!("content: {}", message["content"]);
        assert!(
            message["content"]
                .as_str()
                .is_some_and(|c| !c.trim().is_empty()),
            "no answer in {reply}"
        );
        // `--jinja` is what splits the thinking out of the answer. Without it the reasoning
        // arrives inside `content` and every agent renders the model's notes as its reply.
        assert!(
            message.get("reasoning_content").is_some(),
            "reasoning should be separated from the answer: {message}"
        );

        // And the generated configs should name this endpoint, not a guess at one.
        let dir = std::env::temp_dir().join(format!("pstore-serve-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for w in crate::agents::wire::WIRINGS {
            let outcome = crate::agents::wire::apply(w, &dir, &s);
            assert!(outcome.wired(), "{}: {outcome:?}", w.display);
            let written = std::fs::read_to_string(w.path(&dir)).unwrap();
            assert!(written.contains(&s.base_url()));
            assert!(written.contains(&s.model_name()));
        }
        std::fs::remove_dir_all(&dir).ok();

        assert!(stop(), "the server should have been running");
        assert_eq!(status(), None, "and gone afterwards");
        assert!(!stop(), "stopping twice is not an error");
    }

    /// Agents send back the model name the endpoint advertises, and `llama-server` names the
    /// model after the file it loaded. Getting this wrong is a 404 on the agent's first turn.
    #[test]
    fn the_model_name_is_the_weights_file() {
        for c in models::ALL.iter() {
            let s = Serving {
                port: 1,
                checkpoint: c.id,
                pid: 1,
            };
            assert_eq!(s.model_name(), *c.files.last().unwrap());
            assert_eq!(s.title(), c.title);
        }

        // An id that is not in the catalogue still yields something usable rather than a panic.
        let unknown = Serving {
            port: 1,
            checkpoint: "gone",
            pid: 1,
        };
        assert!(!unknown.model_name().is_empty());
        assert!(!unknown.title().is_empty());
    }
}
