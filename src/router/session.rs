//! One load of the weights, for one thing the user asked for.
//!
//! **The unit is the operation, not the call.** Ranking is two model calls — how hard is this
//! prompt, then which model suits it. Shrinking a long document is one call per chunk. A
//! personal-data scan is one per chunk too. Loading 3.8 GB separately for each of those is
//! seconds of pure overhead per chunk, and it buys nothing: it is the same weights answering the
//! same user's single request.
//!
//! So a [`Session`] maps the weights once, answers every call the operation needs, and is killed
//! when the operation ends. Between operations **nothing is resident** — no process, no port, no
//! memory held — which is also what makes the context window honest: it is computed for the calls
//! this operation will actually make, rather than sized for whatever might come later.
//!
//! ## What it is not
//!
//! It is not a service. The process binds `127.0.0.1` on a port the kernel chooses, behind a
//! random key generated for this session and never written down; the web UI is off; and it lives
//! for the seconds an operation takes. Nothing can find it, nothing is meant to, and there is no
//! setting to make it stay. That is the difference between this and the background server pstore
//! used to offer, which was a feature and is gone.
//!
//! Loopback rather than a Unix socket only because this fork does not bind one: `--host` with a
//! `.sock` path is accepted by the argument parser and then fails at `bind`. If that is fixed
//! upstream this module should move to a socket file and drop the key.
//!
//! ## Why the server binary rather than one-shot completion
//!
//! `llama-completion` takes one prompt and exits, so N calls means N loads. `llama-server` takes
//! a prompt per request with its **own grammar per request**, which is what the operations here
//! need: ranking constrains one call with a JSON schema and the next with a hand-written grammar.
//! An interactive `llama-completion` session would fix one grammar for its whole life.
//!
//! The prompt is rendered by `/apply-template`, which applies the model's own chat template and
//! opens the `<think>` block exactly as `--jinja` does for the one-shot binary — so the grammars,
//! which only have to *close* that block, are unchanged.

use std::io::Read;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use super::llm::{Constrain, Task};
use crate::models;

/// Where a session listens. Loopback, never `0.0.0.0`.
const HOST: &str = "127.0.0.1";

/// How long to wait for the weights to map before giving up.
///
/// Generous because a cold page cache makes this disk I/O rather than work: 3.8 GB off a slow
/// disk is minutes. Measured warm it is one to two seconds.
const READY_TIMEOUT: Duration = Duration::from_secs(180);

/// How often to ask whether it is up.
const POLL: Duration = Duration::from_millis(50);

/// How long any one call may take.
///
/// A ranking call on the ternary build is tens of seconds and a shrink chunk can be longer, so
/// this is a backstop against a wedged process rather than a latency budget.
const CALL_TIMEOUT: Duration = Duration::from_secs(600);

/// One load of the weights, answering every call of one operation.
pub struct Session {
    /// Shared with [`crate::router::llm`]'s registry, so a build switch or the app closing can
    /// end this process without waiting on the operation that owns it.
    child: Arc<Mutex<Child>>,
    pid: u32,
    port: u16,
    key: String,
    checkpoint: models::Checkpoint,
    /// Diagnostics from the process, appended by a thread that drains the pipe — an unread pipe
    /// would wedge the process rather than fail it, and the tail is the only clue when the
    /// weights will not load.
    stderr: Arc<Mutex<String>>,
}

impl Session {
    /// Map the weights and wait until they will answer.
    ///
    /// `context` is the window this operation needs — see [`crate::router::llm::Plan`], which
    /// computes it from the actual prompts rather than from a ceiling. Blocking, in units of
    /// seconds; call it from a worker thread.
    pub fn open(context: usize) -> Result<Self, String> {
        let (binary, weights, checkpoint) = super::llm::ready()?;
        super::llm::refuse_if_stopping(&checkpoint)?;

        let port = free_port()?;
        let key = one_time_key();

        let mut child = Command::new(&binary)
            .arg("-m")
            .arg(&weights)
            .args(["--host", HOST])
            .args(["--port", &port.to_string()])
            // Nothing but this process knows the key, and it is never written to disk. It is
            // belt-and-braces over binding loopback: another local user must not be able to
            // drive these weights just because they guessed the port.
            .args(["--api-key", &key])
            // A web UI on a private inference process is attack surface with no user.
            .arg("--no-webui")
            // One caller, one call at a time. More slots would divide the KV cache for
            // concurrency nothing here uses.
            .args(["--parallel", "1"])
            // The model's own chat template, which `/apply-template` then renders with.
            .arg("--jinja")
            .args(["-c", &context.to_string()])
            .args(["-ngl", "999"])
            // Memory: 4-bit KV cache and flash attention, as the one-shot path used.
            .args(["--cache-type-k", "q4_0", "--cache-type-v", "q4_0"])
            .args(["--flash-attn", "on"])
            // Nothing is pre-touched: this process exists for one operation.
            .arg("--no-warmup")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("starting {}: {e}", binary.display()))?;

        let mut pipe = child.stderr.take();
        let stderr = Arc::new(Mutex::new(String::new()));
        let sink = Arc::clone(&stderr);
        std::thread::spawn(move || {
            let mut buf = [0u8; 8192];
            while let Some(p) = pipe.as_mut() {
                match p.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => {
                        if let Ok(mut s) = sink.lock() {
                            s.push_str(&String::from_utf8_lossy(&buf[..n]));
                            // Bounded: this process writes kilobytes of load-time chatter about
                            // tensors, and only the tail ever explains anything.
                            if s.len() > 8192 {
                                let cut = s.len() - 4096;
                                *s = s[cut..].to_string();
                            }
                        }
                    }
                }
            }
        });

        models::set(checkpoint.id, models::Phase::Loading);
        let pid = child.id();
        let child = Arc::new(Mutex::new(child));
        super::llm::register(pid, checkpoint.id, Arc::clone(&child));

        let mut session = Session {
            child,
            pid,
            port,
            key,
            checkpoint,
            stderr,
        };

        match session.wait_until_ready() {
            Ok(()) => {
                models::set(checkpoint.id, models::Phase::Ready);
                Ok(session)
            }
            Err(e) => {
                models::set(checkpoint.id, models::Phase::Failed(e.clone()));
                Err(e)
            }
        }
    }

    /// Poll until it answers, or until it dies trying.
    ///
    /// The distinction that matters is "not up yet" against "up and broken": a refused connection
    /// is the former and worth retrying, the process having exited is the latter, and waiting the
    /// full three minutes on something already gone helps nobody.
    fn wait_until_ready(&mut self) -> Result<(), String> {
        let url = format!("http://{HOST}:{}/health", self.port);
        let started = Instant::now();

        while started.elapsed() < READY_TIMEOUT {
            // The lock is only ever held across a non-blocking `try_wait`, so a build switch or
            // the app closing can take it at any point during the load.
            let gone = match self.child.lock() {
                Ok(mut c) => matches!(c.try_wait(), Ok(Some(_)) | Err(_)),
                Err(_) => true,
            };
            if gone {
                return Err(format!("the model exited while loading — {}", self.why()));
            }
            if super::llm::stopping() {
                return Err(super::llm::CLOSING_REASON.into());
            }
            if agent().get(&url).call().is_ok() {
                return Ok(());
            }
            std::thread::sleep(POLL);
        }
        Err(format!(
            "the model did not load within {}s",
            READY_TIMEOUT.as_secs()
        ))
    }

    /// Run one call and return its parsed JSON reply.
    ///
    /// The constraint and the sampling both come from `task`, because they belong to the job
    /// being done rather than to this module — see [`Task`].
    pub fn run(&self, task: &Task, prompt: &str) -> Result<Value, String> {
        let rendered = self.render(prompt)?;

        let mut body = json!({
            "prompt": rendered,
            "n_predict": task.max_output,
            "temperature": task.temperature,
            // Pinned, so a call is reproducible whichever branch it takes.
            "seed": 1,
            // Free, and it makes a repeated identical call — the retry path, or the same prompt
            // ranked twice — near-instant. This fork reuses an exact prefix only, so it does
            // nothing for two chunks that share an instruction.
            "cache_prompt": true,
        });
        if task.temperature > 0.0 {
            body["top_p"] = json!(task.top_p);
            body["top_k"] = json!(task.top_k);
        }
        if task.repeat_penalty > 1.0 {
            body["repeat_penalty"] = json!(task.repeat_penalty);
            // The default window is 64 tokens, which is shorter than two entries of a list.
            // A penalty that cannot see the previous item cannot notice it is being copied.
            body["repeat_last_n"] = json!(1024);
        }
        match &task.constrain {
            Constrain::Schema(schema) => body["json_schema"] = (*schema).clone(),
            Constrain::Grammar(gbnf) => body["grammar"] = json!(gbnf),
        }

        let reply = self.post("/completion", &body)?;
        let content = reply
            .get("content")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("the model's reply had no content: {reply}"))?;
        super::llm::parse_reply(content)
    }

    /// Apply the model's own chat template to `prompt`.
    ///
    /// Not a nicety. Without it the checkpoint — a thinking model whose template opens a
    /// `<think>` block — is prompted in a shape it was never trained on, and ranks visibly worse:
    /// the same call that returns Opus 5 at high effort returns `Haiku 4.5, effort medium` and
    /// then the next four options in order.
    fn render(&self, prompt: &str) -> Result<String, String> {
        let body = json!({"messages": [{"role": "user", "content": prompt}]});
        let reply = self.post("/apply-template", &body)?;
        reply
            .get("prompt")
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| "the model's runtime did not apply its chat template".to_string())
    }

    fn post(&self, path: &str, body: &Value) -> Result<Value, String> {
        if super::llm::stopping() {
            return Err(super::llm::CLOSING_REASON.into());
        }
        let text = agent()
            .post(format!("http://{HOST}:{}{path}", self.port))
            .header("Authorization", format!("Bearer {}", self.key))
            .header("Content-Type", "application/json")
            .send(body.to_string())
            .map_err(|e| format!("the model did not answer: {e} — {}", self.why()))?
            .body_mut()
            .read_to_string()
            .map_err(|e| format!("reading the model's reply: {e}"))?;
        serde_json::from_str(&text).map_err(|e| format!("could not parse the model's reply: {e}"))
    }

    /// The last few lines the process wrote, for an error message.
    ///
    /// The head is load-time chatter about tensors; the tail is the reason.
    fn why(&self) -> String {
        let captured = match self.stderr.lock() {
            Ok(s) => s.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        };
        let tail: Vec<&str> = captured
            .lines()
            .filter(|l| !l.trim().is_empty())
            .rev()
            .take(3)
            .collect();
        if tail.is_empty() {
            "it said nothing about why".to_string()
        } else {
            tail.into_iter().rev().collect::<Vec<_>>().join(" / ")
        }
    }
}

impl Drop for Session {
    /// End the process, and with it the weights.
    ///
    /// The weights live in the child's address space, not this one, so "unloading the model" is
    /// exactly this. Killed and reaped rather than left to exit: a zombie holds its slot, and the
    /// port stays bound until the process is really gone.
    fn drop(&mut self) {
        super::llm::end(&self.child);
        super::llm::deregister(self.pid);
        // Weights on disk, nothing running: what is true the moment after.
        models::set(self.checkpoint.id, models::Phase::Cached);
    }
}

/// An HTTP client for talking to a process on this machine.
fn agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(CALL_TIMEOUT))
        .build()
        .into()
}

/// A port the kernel says is free.
///
/// Bound and immediately released, so there is a window in which something else could take it.
/// The alternative is a fixed port, which is worse: two pstore windows would collide every time
/// rather than approximately never.
fn free_port() -> Result<u16, String> {
    let listener = std::net::TcpListener::bind((HOST, 0))
        .map_err(|e| format!("could not reserve a loopback port: {e}"))?;
    listener
        .local_addr()
        .map(|a| a.port())
        .map_err(|e| format!("could not read the reserved port: {e}"))
}

/// A key for one session, from the OS's own randomness.
///
/// `RandomState` is seeded per instance from the platform's secure random source — it is what
/// makes `HashMap` resistant to collision attacks — so this is real entropy rather than a
/// timestamp dressed up as one, and it costs no dependency.
fn one_time_key() -> String {
    use std::hash::{BuildHasher, Hasher, RandomState};

    let mut out = String::with_capacity(32);
    for _ in 0..2 {
        let mut h = RandomState::new().build_hasher();
        h.write_u8(0);
        out.push_str(&format!("{:016x}", h.finish()));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two sessions must never be handed the same port, and the key must be unguessable and
    /// different every time — it is the only thing between these weights and another local user.
    #[test]
    fn each_session_gets_its_own_port_and_key() {
        let a = free_port().expect("a free port");
        let b = free_port().expect("another free port");
        assert!(a > 1024, "{a} is a privileged port");
        assert!(b > 1024);

        let keys: Vec<String> = (0..8).map(|_| one_time_key()).collect();
        for k in &keys {
            assert_eq!(k.len(), 32, "{k} is not 128 bits of hex");
            assert!(k.chars().all(|c| c.is_ascii_hexdigit()), "{k}");
        }
        let mut unique = keys.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), keys.len(), "keys repeat: {keys:?}");
    }

    /// The host is checked rather than assumed: `0.0.0.0` here would turn a private inference
    /// process into something every machine on the network can drive.
    #[test]
    fn a_session_is_reachable_only_from_this_machine() {
        assert_eq!(HOST, "127.0.0.1");
    }
}
