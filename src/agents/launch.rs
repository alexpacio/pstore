//! Running agents: headless capture, headless streaming, and interactive terminal spawn.
//!
//! Prompts always reach the child through an argument or stdin — never interpolated
//! into a shell string. That sidesteps quoting bugs and argument-length limits.

use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc::Sender;
use std::time::{Duration, Instant};

use super::registry::{AgentSpec, Effort, PromptVia};

/// Result of a completed headless run.
#[derive(Debug, Clone, Default)]
pub struct Output {
    /// Process exit code, or `None` if it was killed.
    pub code: Option<i32>,
    /// Captured stdout, verbatim.
    ///
    /// For an agent speaking `stream-json` this is the JSON transcript, not anything a
    /// human would read. It stays raw because [`crate::agents::failover::classify`] greps
    /// it for quota and authentication strings that only appear inside the envelopes —
    /// use [`Self::text`] or [`Self::result`] for anything user-facing.
    pub stdout: String,
    /// The assistant text pulled out of `stdout`, concatenated in arrival order.
    ///
    /// Everything the agent said, which for a tool-using agent includes the narration
    /// between tool calls. [`Self::result`] is the better choice when it is populated.
    pub text: String,
    /// The agent's final answer, when it reports one as a distinct event.
    ///
    /// Claude Code closes a `stream-json` run with `{"type":"result","result":"…"}`. That
    /// payload is the answer *alone*, with none of the narration that [`Self::text`]
    /// accumulates, which is what a one-shot transformation like `plan` wants. `None` for
    /// agents that just print to stdout.
    pub result: Option<String>,
    /// Captured stderr.
    pub stderr: String,
    /// Whether the run was killed for exceeding its time budget.
    pub timed_out: bool,
    /// Whether the run was killed because the user cancelled it.
    pub cancelled: bool,
    /// Wall-clock duration, used to rank agents by observed latency.
    pub elapsed: Duration,
}

impl Output {
    /// Whether the process reported success.
    pub fn ok(&self) -> bool {
        self.code == Some(0) && !self.timed_out && !self.cancelled
    }
}

/// A cancellation flag a caller can raise to stop a run in flight.
pub type Cancel = std::sync::Arc<std::sync::atomic::AtomicBool>;

/// Whether `cancel` has been raised.
pub fn raised(cancel: Option<&Cancel>) -> bool {
    cancel.is_some_and(|c| c.load(std::sync::atomic::Ordering::Relaxed))
}

/// Build the argv (excluding the program) for a headless run.
///
/// Model and effort are applied only where the agent's CLI actually supports them —
/// an empty model id or an unsupported effort is silently omitted rather than passed
/// as a bogus flag.
///
/// Every headless run is also a read-only one, so [`AgentSpec::readonly_extra`] is always
/// applied. That is the whole distinction between pstore's two ways of running an agent:
/// headless produces text *about* the code, and the interactive handoff
/// ([`terminal_command`]) is the one that changes it. An agent with no read-only mode is
/// launched unchanged.
///
/// Returns `None` for the prompt when it should go to stdin instead.
pub fn headless_args(
    spec: &AgentSpec,
    model: Option<&str>,
    effort: Option<Effort>,
    prompt: &str,
) -> (Vec<String>, Option<String>) {
    let mut args: Vec<String> = spec.headless.iter().map(|s| s.to_string()).collect();
    if let (Some(flag), Some(m)) = (spec.model_flag, model)
        && !m.is_empty()
    {
        args.push(flag.to_string());
        args.push(m.to_string());
    }
    if let Some(e) = effort {
        args.extend(spec.effort_flag.args(e));
    }
    args.extend(spec.headless_extra.iter().map(|s| s.to_string()));
    args.extend(spec.readonly_extra.iter().map(|s| s.to_string()));

    match spec.prompt_via {
        PromptVia::Arg => {
            args.push(prompt.to_string());
            (args, None)
        }
        PromptVia::Stdin => (args, Some(prompt.to_string())),
    }
}

/// Run a process to completion, capturing both streams.
///
/// The child is killed if it exceeds `timeout`; `timed_out` is set on the result.
pub fn run_capture(
    program: &Path,
    args: &[String],
    stdin_data: Option<&str>,
    cwd: Option<&Path>,
    timeout: Duration,
) -> std::io::Result<Output> {
    let started = Instant::now();
    let mut cmd = Command::new(program);
    cmd.args(args)
        .stdin(if stdin_data.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(d) = cwd {
        cmd.current_dir(d);
    }
    let mut child = cmd.spawn()?;

    if let (Some(data), Some(mut sink)) = (stdin_data, child.stdin.take()) {
        // A child that exits before reading stdin gives us a broken pipe, which is
        // not an error from our side — the exit code is what matters.
        let _ = sink.write_all(data.as_bytes());
        drop(sink);
    }

    // Drain both pipes on threads so a full stderr buffer can't deadlock stdout.
    let mut out_pipe = child.stdout.take();
    let mut err_pipe = child.stderr.take();
    let out_handle = std::thread::spawn(move || {
        let mut s = String::new();
        if let Some(p) = out_pipe.as_mut() {
            let _ = p.read_to_string(&mut s);
        }
        s
    });
    let err_handle = std::thread::spawn(move || {
        let mut s = String::new();
        if let Some(p) = err_pipe.as_mut() {
            let _ = p.read_to_string(&mut s);
        }
        s
    });

    let (status, timed_out, cancelled) = wait_for(&mut child, started, timeout, None)?;

    // Same per-line fold as the streaming path, so both agree on what the agent said.
    let stdout = out_handle.join().unwrap_or_default();
    let mut text = String::new();
    let mut result = None;
    for line in stdout.lines() {
        absorb_line(line, &mut text, &mut result);
    }

    Ok(Output {
        code: status.and_then(|s| s.code()),
        stdout,
        text,
        result,
        stderr: err_handle.join().unwrap_or_default(),
        timed_out,
        cancelled,
        elapsed: started.elapsed(),
    })
}

/// Wait for `child`, killing it if `timeout` elapses or `cancel` is raised.
///
/// Returns `(exit status, timed out, cancelled)`. The status is `None` when the child
/// was killed rather than allowed to finish.
fn wait_for(
    child: &mut std::process::Child,
    started: Instant,
    timeout: Duration,
    cancel: Option<&Cancel>,
) -> std::io::Result<(Option<std::process::ExitStatus>, bool, bool)> {
    loop {
        if let Some(s) = child.try_wait()? {
            return Ok((Some(s), false, false));
        }
        if raised(cancel) {
            let _ = child.kill();
            let _ = child.wait();
            return Ok((None, false, true));
        }
        if started.elapsed() >= timeout {
            let _ = child.kill();
            let _ = child.wait();
            return Ok((None, true, false));
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// Run headlessly, forwarding output line-by-line so the UI can render tokens as
/// they arrive. Blocks until the child exits; call from a worker thread.
// Plumbing: every argument is a distinct, unrelated input (program, argv,
// stdin, cwd, timeout, cancellation, sink). Bundling them into a struct
// would add a type to thread through without making any call site clearer.
#[allow(clippy::too_many_arguments)]
pub fn run_streaming(
    program: &Path,
    args: &[String],
    stdin_data: Option<&str>,
    cwd: Option<&Path>,
    timeout: Duration,
    cancel: Option<&Cancel>,
    tx: &Sender<String>,
) -> std::io::Result<Output> {
    let started = Instant::now();
    let mut cmd = Command::new(program);
    cmd.args(args)
        .stdin(if stdin_data.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(d) = cwd {
        cmd.current_dir(d);
    }
    let mut child = cmd.spawn()?;

    if let (Some(data), Some(mut sink)) = (stdin_data, child.stdin.take()) {
        let _ = sink.write_all(data.as_bytes());
        drop(sink);
    }

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    let out_tx = tx.clone();
    let out_handle = std::thread::spawn(move || {
        let mut collected = String::new();
        let mut text = String::new();
        let mut result = None;
        if let Some(p) = stdout {
            for line in BufReader::new(p).lines().map_while(Result::ok) {
                collected.push_str(&line);
                collected.push('\n');
                if let Some(chunk) = absorb_line(&line, &mut text, &mut result) {
                    let _ = out_tx.send(chunk);
                }
            }
        }
        (collected, text, result)
    });

    // stderr is collected but never forwarded: it is read for failure classification, not shown
    // as output, and an agent's progress chatter in the answer panel is noise the user then has
    // to read past.
    let err_handle = std::thread::spawn(move || {
        let mut collected = String::new();
        if let Some(p) = stderr {
            for line in BufReader::new(p).lines().map_while(Result::ok) {
                collected.push_str(&line);
                collected.push('\n');
            }
        }
        collected
    });

    let (status, timed_out, cancelled) = wait_for(&mut child, started, timeout, cancel)?;

    let (stdout, text, result) = out_handle.join().unwrap_or_default();

    Ok(Output {
        code: status.and_then(|s| s.code()),
        stdout,
        text,
        result,
        stderr: err_handle.join().unwrap_or_default(),
        timed_out,
        cancelled,
        elapsed: started.elapsed(),
    })
}

/// Pull display text out of one stdout line.
///
/// Agents that speak `stream-json` (Claude Code) emit one JSON object per line; we
/// want only the assistant text deltas. Anything that isn't recognisable JSON is
/// passed through verbatim, which covers the plain-text agents.
fn extract_text(line: &str) -> Option<String> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }
    if !trimmed.starts_with('{') {
        return Some(line.to_string());
    }
    let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) else {
        // Looked like JSON but wasn't — show it rather than silently dropping output.
        return Some(line.to_string());
    };

    // Claude Code stream-json: {"type":"assistant","message":{"content":[{"type":"text","text":"..."}]}}
    if let Some(content) = v
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_array())
    {
        let mut out = String::new();
        for block in content {
            if block.get("type").and_then(|t| t.as_str()) == Some("text")
                && let Some(t) = block.get("text").and_then(|t| t.as_str())
            {
                out.push_str(t);
            }
        }
        return Some(out);
    }
    // Incremental delta shapes.
    for path in [["delta", "text"], ["content_block", "text"]] {
        if let Some(t) = v
            .get(path[0])
            .and_then(|d| d.get(path[1]))
            .and_then(|t| t.as_str())
        {
            return Some(t.to_string());
        }
    }
    if let Some(t) = v.get("text").and_then(|t| t.as_str()) {
        return Some(t.to_string());
    }
    // A structural event (init, tool_use, result) with no text to show.
    None
}

/// Pull the *final answer* out of one stdout line, if this line is the run's result event.
///
/// Claude Code ends a `stream-json` run with
/// `{"type":"result","subtype":"success","result":"…"}`. That is deliberately not part of
/// [`extract_text`]: the value duplicates text already streamed, so forwarding it to the
/// panel would print the answer twice. It is worth capturing separately because it is the
/// answer *without* the narration a tool-using agent emits between tool calls — for
/// `plan`, that narration is the difference between a pasteable instruction and a
/// transcript.
///
/// An error result is not an answer, so only `success` is taken.
fn extract_result(line: &str) -> Option<String> {
    let trimmed = line.trim();
    if !trimmed.starts_with('{') {
        return None;
    }
    let v = serde_json::from_str::<serde_json::Value>(trimmed).ok()?;
    if v.get("type").and_then(|t| t.as_str()) != Some("result") {
        return None;
    }
    if v.get("subtype")
        .and_then(|s| s.as_str())
        .is_some_and(|s| s != "success")
    {
        return None;
    }
    let text = v.get("result").and_then(|r| r.as_str())?;
    (!text.trim().is_empty()).then(|| text.to_string())
}

/// Fold one stdout line into the accumulated text and result of a run.
///
/// Shared so [`run_capture`] and [`run_streaming`] cannot drift into disagreeing about
/// what an agent said — the bug that made `plan` return raw JSON was exactly one of the
/// two paths keeping the parsed text and the other keeping the envelopes.
fn absorb_line(line: &str, text: &mut String, result: &mut Option<String>) -> Option<String> {
    if let Some(r) = extract_result(line) {
        *result = Some(r);
        return None;
    }
    let chunk = extract_text(line)?;
    if chunk.is_empty() {
        return None;
    }
    text.push_str(&chunk);
    // A passthrough line is a whole line of a plain-text agent's output, and the line
    // break is part of what it said — a plan's own list items depend on it. Text lifted
    // out of a JSON envelope is a fragment of one message and must not gain breaks it
    // never had. `extract_text` returns the line itself in exactly the passthrough case.
    if chunk == line {
        text.push('\n');
    }
    Some(chunk)
}

/// How to open an interactive agent session in a new OS window.
#[derive(Debug, Clone)]
pub struct TerminalCommand {
    /// Program to execute.
    pub program: String,
    /// Arguments to pass.
    pub args: Vec<String>,
}

/// Shell-single-quote a string for embedding in a `sh -c` command line.
fn shell_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', r"'\''"))
}

/// Build the command that opens `agent` interactively in a new terminal window,
/// with `prompt_file` supplying the prompt.
///
/// Pure so it can be unit-tested on every platform without spawning anything.
pub fn terminal_command(
    spec: &AgentSpec,
    cwd: &Path,
    prompt_file: &Path,
    available: &dyn Fn(&str) -> bool,
) -> Result<TerminalCommand, String> {
    // The prompt is substituted from the file at run time, so it never appears in
    // an argv or an AppleScript string.
    let mut inner = format!("cd {} && {}", shell_quote(&cwd.to_string_lossy()), spec.bin);
    for a in spec.interactive {
        inner.push(' ');
        inner.push_str(a);
    }
    inner.push_str(&format!(
        " \"$(cat {})\"",
        shell_quote(&prompt_file.to_string_lossy())
    ));

    if cfg!(target_os = "macos") {
        // iTerm2 if present, else Terminal.app. `do script` runs in a new window.
        let app = if available("iTerm") {
            "iTerm"
        } else {
            "Terminal"
        };
        let script = format!(
            "tell application \"{app}\" to do script \"{}\"\ntell application \"{app}\" to activate",
            inner.replace('\\', "\\\\").replace('"', "\\\"")
        );
        return Ok(TerminalCommand {
            program: "osascript".into(),
            args: vec!["-e".into(), script],
        });
    }

    if cfg!(target_os = "windows") {
        if available("wt.exe") {
            return Ok(TerminalCommand {
                program: "wt.exe".into(),
                args: vec![
                    "-d".into(),
                    cwd.to_string_lossy().into_owned(),
                    "cmd".into(),
                    "/k".into(),
                    format!("type {} | {}", prompt_file.to_string_lossy(), spec.bin),
                ],
            });
        }
        return Ok(TerminalCommand {
            program: "cmd".into(),
            args: vec![
                "/c".into(),
                "start".into(),
                String::new(), // window title placeholder — `start` needs it
                "cmd".into(),
                "/k".into(),
                format!("cd /d {} && {}", cwd.to_string_lossy(), spec.bin),
            ],
        });
    }

    // Linux and the BSDs: emulators differ in how they take a command.
    // `-e` takes the rest of argv; `--` is the modern form for several.
    const CANDIDATES: &[(&str, &[&str])] = &[
        ("kitty", &[]),
        ("alacritty", &["-e"]),
        ("wezterm", &["start", "--"]),
        ("gnome-terminal", &["--"]),
        ("konsole", &["-e"]),
        ("xfce4-terminal", &["-x"]),
        ("xterm", &["-e"]),
    ];
    for (bin, lead) in CANDIDATES {
        if available(bin) {
            let mut args: Vec<String> = lead.iter().map(|s| s.to_string()).collect();
            args.extend(["sh".to_string(), "-c".to_string(), inner.clone()]);
            return Ok(TerminalCommand {
                program: (*bin).to_string(),
                args,
            });
        }
    }
    Err("no supported terminal emulator found (tried kitty, alacritty, wezterm, gnome-terminal, konsole, xfce4-terminal, xterm)".into())
}

/// Write `prompt` to a temp file that outlives this call, for terminal handoff.
pub fn write_prompt_file(slug: &str, prompt: &str) -> std::io::Result<PathBuf> {
    let mut path = std::env::temp_dir();
    path.push(format!("pstore-{slug}-{}.md", std::process::id()));
    std::fs::write(&path, prompt)?;
    Ok(path)
}

/// Spawn an interactive session in a new terminal window and return immediately.
pub fn open_in_terminal(
    spec: &AgentSpec,
    cwd: &Path,
    prompt: &str,
    slug: &str,
) -> Result<(), String> {
    let file = write_prompt_file(slug, prompt).map_err(|e| format!("writing prompt file: {e}"))?;
    let cmd = terminal_command(spec, cwd, &file, &|bin| which_exists(bin))?;
    Command::new(&cmd.program)
        .args(&cmd.args)
        .spawn()
        .map(|_| ())
        .map_err(|e| format!("launching {}: {e}", cmd.program))
}

fn which_exists(bin: &str) -> bool {
    if cfg!(target_os = "macos") {
        // Terminal emulators are .app bundles, not PATH entries.
        return Path::new(&format!("/Applications/{bin}.app")).exists();
    }
    std::env::var_os("PATH").is_some_and(|p| {
        std::env::split_paths(&p).any(|d| {
            let c = d.join(bin);
            c.is_file()
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::registry;

    #[test]
    fn headless_args_place_model_effort_and_prompt() {
        let claude = registry::find("claude").unwrap();
        let (args, stdin) = headless_args(claude, Some("haiku"), Some(Effort::Low), "hello");
        assert_eq!(stdin, None, "claude takes the prompt as an argument");
        assert_eq!(args[0], "-p");
        let m = args.iter().position(|a| a == "--model").unwrap();
        assert_eq!(args[m + 1], "haiku");
        let e = args.iter().position(|a| a == "--effort").unwrap();
        assert_eq!(args[e + 1], "low");
        assert_eq!(
            args.last().unwrap(),
            "hello",
            "prompt is the final argument"
        );
    }

    /// The transcript is not the answer.
    ///
    /// This is the bug `pstore plan` shipped with: the raw `stream-json` lines were kept
    /// as the run's output and the parsed text was only ever sent to the streaming
    /// channel, so `plan` — the one feature reading the completed run rather than the
    /// channel — printed JSON envelopes at the user. Fold a realistic Claude Code
    /// transcript, narration and tool call included, and assert on what a caller sees.
    #[test]
    fn a_stream_json_transcript_yields_the_answer_not_the_envelopes() {
        let transcript = [
            r#"{"type":"system","subtype":"init","session_id":"abc"}"#,
            r#"{"type":"assistant","message":{"content":[{"type":"text","text":"Let me look at the retry logic."}]}}"#,
            r#"{"type":"assistant","message":{"content":[{"type":"tool_use","name":"Read","input":{}}]}}"#,
            r#"{"type":"user","message":{"content":[{"type":"tool_result","content":"fn retry() {}"}]}}"#,
            r#"{"type":"result","subtype":"success","result":"**Objective**\nRetry count is configurable."}"#,
        ];

        let mut text = String::new();
        let mut result = None;
        for line in transcript {
            absorb_line(line, &mut text, &mut result);
        }

        // What `Completed.text` is built from.
        let answer = result.clone().unwrap_or_else(|| text.clone());
        assert_eq!(answer, "**Objective**\nRetry count is configurable.");
        assert!(!answer.contains("\"type\":"), "JSON leaked: {answer:?}");
        assert!(
            !answer.contains("Let me look"),
            "narration between tool calls leaked into the answer: {answer:?}"
        );

        // The narration is still available for the live panel — it is just not the answer.
        assert_eq!(text, "Let me look at the retry logic.");
    }

    /// A plain-text agent has no result event, so the concatenated text is the answer —
    /// and its line structure is part of what it said.
    #[test]
    fn a_plain_text_agent_keeps_its_line_breaks() {
        let mut text = String::new();
        let mut result = None;
        for line in [
            "**Objective**",
            "Ship it.",
            "",
            "**Steps**",
            "1. Do the thing.",
        ] {
            absorb_line(line, &mut text, &mut result);
        }
        assert_eq!(result, None);
        assert_eq!(
            text,
            "**Objective**\nShip it.\n**Steps**\n1. Do the thing.\n"
        );
    }

    /// An error result is not an answer; falling back to the narration beats reporting
    /// the failure message as though it were the plan.
    #[test]
    fn an_error_result_is_not_taken_as_the_answer() {
        let mut text = String::new();
        let mut result = None;
        absorb_line(
            r#"{"type":"assistant","message":{"content":[{"type":"text","text":"partial"}]}}"#,
            &mut text,
            &mut result,
        );
        absorb_line(
            r#"{"type":"result","subtype":"error_max_turns","result":"hit the turn limit"}"#,
            &mut text,
            &mut result,
        );
        assert_eq!(result, None);
        assert_eq!(text, "partial");
    }

    /// Planning must not be able to perform the work it is describing.
    #[test]
    fn headless_runs_ask_for_the_agents_read_only_mode() {
        let claude = registry::find("claude").unwrap();
        let (args, _) = headless_args(claude, Some("haiku"), None, "plan this");
        assert!(
            args.iter()
                .any(|a| a.starts_with("--disallowedTools=") && a.contains("Write")),
            "got {args:?}"
        );

        // Aider's headless flags include `--yes`, which auto-approves edits.
        let aider = registry::find("aider").unwrap();
        let (args, _) = headless_args(aider, None, None, "plan this");
        assert!(args.iter().any(|a| a == "--chat-mode=ask"), "got {args:?}");
    }

    /// A read-only flag that takes a list is variadic in at least one agent's CLI
    /// (`claude --disallowedTools`), and pstore appends the prompt as the last positional
    /// argument. Written as a separate flag and value, such a flag consumes the prompt and
    /// the run dies with "Input must be provided". The `=` form is what prevents that, so
    /// no read-only argument may be a bare flag expecting a following value.
    #[test]
    fn read_only_flags_cannot_swallow_the_prompt() {
        for spec in registry::AGENTS {
            for arg in spec.readonly_extra {
                assert!(
                    arg.contains('='),
                    "{}: read-only argument {arg:?} must bind its value with `=`",
                    spec.id
                );
            }
            if spec.prompt_via == PromptVia::Arg {
                let (args, _) = headless_args(spec, None, None, "THE PROMPT");
                assert_eq!(
                    args.last().map(String::as_str),
                    Some("THE PROMPT"),
                    "{}: prompt is no longer the final argument: {args:?}",
                    spec.id
                );
            }
        }
    }

    #[test]
    fn codex_effort_uses_its_config_override_syntax() {
        let codex = registry::find("codex").unwrap();
        let (args, _) = headless_args(codex, Some("gpt-5.1-codex"), Some(Effort::XHigh), "go");
        let c = args.iter().position(|a| a == "-c").unwrap();
        assert_eq!(args[c + 1], "model_reasoning_effort=xhigh");
    }

    #[test]
    fn agents_without_flags_ignore_model_and_effort() {
        let crush = registry::find("crush").unwrap();
        let (args, stdin) =
            headless_args(crush, Some("some-model"), Some(Effort::Max), "explain this");
        assert!(
            !args.iter().any(|a| a.contains("some-model")),
            "got {args:?}"
        );
        assert!(!args.iter().any(|a| a.contains("max")), "got {args:?}");
        assert_eq!(stdin.as_deref(), Some("explain this"), "crush reads stdin");
        assert!(args.contains(&"-q".to_string()));
    }

    #[test]
    fn empty_model_id_is_not_passed_as_a_flag() {
        // The placeholder model used for agents pstore can't configure has an empty
        // id; emitting `--model ""` would be a hard error from the agent.
        let claude = registry::find("claude").unwrap();
        let (args, _) = headless_args(claude, Some(""), None, "hi");
        assert!(!args.contains(&"--model".to_string()), "got {args:?}");
    }

    #[test]
    fn extract_text_handles_stream_json_and_plain_text() {
        // Claude Code assistant message
        let line = r#"{"type":"assistant","message":{"content":[{"type":"text","text":"Hi"},{"type":"text","text":" there"}]}}"#;
        assert_eq!(extract_text(line).as_deref(), Some("Hi there"));

        // Delta shape
        assert_eq!(
            extract_text(r#"{"delta":{"text":"tok"}}"#).as_deref(),
            Some("tok")
        );

        // Structural event carries no text
        assert_eq!(extract_text(r#"{"type":"system","subtype":"init"}"#), None);

        // Plain text passes through
        assert_eq!(extract_text("just words").as_deref(), Some("just words"));

        // Malformed JSON is shown rather than dropped
        assert_eq!(extract_text("{not json").as_deref(), Some("{not json"));

        // Blank lines are skipped
        assert_eq!(extract_text("   "), None);
    }

    #[test]
    fn shell_quote_escapes_embedded_quotes() {
        assert_eq!(shell_quote("plain"), "'plain'");
        assert_eq!(shell_quote("it's"), r"'it'\''s'");
        // A path containing a quote cannot break out of the quoting.
        let q = shell_quote("/tmp/a'b");
        assert!(q.starts_with('\'') && q.ends_with('\''));
    }

    #[test]
    fn terminal_command_never_inlines_the_prompt() {
        let claude = registry::find("claude").unwrap();
        let cmd = terminal_command(
            claude,
            Path::new("/work/dir"),
            Path::new("/tmp/p.md"),
            &|_| true,
        )
        .unwrap();
        let joined = format!("{} {}", cmd.program, cmd.args.join(" "));
        assert!(
            joined.contains("/tmp/p.md"),
            "prompt file must be referenced: {joined}"
        );
        assert!(
            joined.contains("cat") || joined.contains("type"),
            "prompt is read from the file at run time: {joined}"
        );
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn macos_prefers_iterm_when_present() {
        let claude = registry::find("claude").unwrap();
        let with = terminal_command(claude, Path::new("/w"), Path::new("/tmp/p.md"), &|b| {
            b == "iTerm"
        })
        .unwrap();
        assert_eq!(with.program, "osascript");
        assert!(with.args[1].contains("iTerm"));

        let without =
            terminal_command(claude, Path::new("/w"), Path::new("/tmp/p.md"), &|_| false).unwrap();
        assert!(without.args[1].contains("Terminal"));
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn linux_picks_the_first_available_emulator_and_errors_when_none() {
        let claude = registry::find("claude").unwrap();
        let kitty = terminal_command(claude, Path::new("/w"), Path::new("/tmp/p.md"), &|b| {
            b == "kitty"
        })
        .unwrap();
        assert_eq!(kitty.program, "kitty");

        let konsole = terminal_command(claude, Path::new("/w"), Path::new("/tmp/p.md"), &|b| {
            b == "konsole"
        })
        .unwrap();
        assert_eq!(konsole.program, "konsole");
        assert_eq!(konsole.args[0], "-e", "konsole needs -e before the command");

        assert!(
            terminal_command(claude, Path::new("/w"), Path::new("/tmp/p.md"), &|_| false).is_err(),
            "no emulator must be a clear error, not a silent no-op"
        );
    }

    #[test]
    #[cfg(unix)]
    fn run_capture_reports_code_stdout_and_stderr() {
        let out = run_capture(
            Path::new("/bin/sh"),
            &["-c".into(), "echo out; echo err >&2; exit 3".into()],
            None,
            None,
            Duration::from_secs(10),
        )
        .unwrap();
        assert_eq!(out.code, Some(3));
        assert!(!out.ok());
        assert!(out.stdout.contains("out"));
        assert!(out.stderr.contains("err"));
        assert!(!out.timed_out);
    }

    #[test]
    #[cfg(unix)]
    fn run_capture_feeds_stdin() {
        let out = run_capture(
            Path::new("/bin/sh"),
            &["-c".into(), "cat".into()],
            Some("piped input"),
            None,
            Duration::from_secs(10),
        )
        .unwrap();
        assert!(out.ok());
        assert!(out.stdout.contains("piped input"));
    }

    #[test]
    #[cfg(unix)]
    fn run_capture_kills_a_hung_child() {
        let out = run_capture(
            Path::new("/bin/sh"),
            &["-c".into(), "sleep 30".into()],
            None,
            None,
            Duration::from_millis(300),
        )
        .unwrap();
        assert!(out.timed_out);
        assert!(!out.ok());
        assert!(
            out.elapsed < Duration::from_secs(5),
            "must not wait for the child"
        );
    }

    #[test]
    #[cfg(unix)]
    fn run_streaming_forwards_lines_as_they_arrive() {
        let (tx, rx) = std::sync::mpsc::channel();
        let out = run_streaming(
            Path::new("/bin/sh"),
            &["-c".into(), "echo one; echo two; echo oops >&2".into()],
            None,
            None,
            Duration::from_secs(10),
            None,
            &tx,
        )
        .unwrap();
        drop(tx);
        assert!(out.ok());

        // Only stdout is forwarded. stderr is collected for classification — `oops` is in
        // `out.stderr` — and deliberately kept out of the stream the answer panel renders.
        let streamed: Vec<String> = rx.into_iter().collect();
        assert_eq!(streamed, vec!["one", "two"]);
        assert!(out.stderr.contains("oops"), "got {:?}", out.stderr);
    }

    #[test]
    #[cfg(unix)]
    fn a_raised_cancel_flag_kills_the_child_promptly() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        let cancel: Cancel = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&cancel);
        // Raise the flag shortly after the child starts.
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(150));
            flag.store(true, Ordering::Relaxed);
        });

        let (tx, _rx) = std::sync::mpsc::channel();
        let out = run_streaming(
            Path::new("/bin/sh"),
            &["-c".into(), "sleep 30".into()],
            None,
            None,
            // A timeout far longer than the test: only cancellation can end this.
            Duration::from_secs(120),
            Some(&cancel),
            &tx,
        )
        .unwrap();

        assert!(out.cancelled, "the run should report cancellation");
        assert!(!out.timed_out, "it was cancelled, not timed out");
        assert!(!out.ok());
        assert!(
            out.elapsed < Duration::from_secs(10),
            "the child must be killed promptly, took {:?}",
            out.elapsed
        );
    }

    #[test]
    #[cfg(unix)]
    fn an_unraised_cancel_flag_does_not_interfere() {
        use std::sync::Arc;
        use std::sync::atomic::AtomicBool;

        let cancel: Cancel = Arc::new(AtomicBool::new(false));
        let (tx, _rx) = std::sync::mpsc::channel();
        let out = run_streaming(
            Path::new("/bin/sh"),
            &["-c".into(), "echo done".into()],
            None,
            None,
            Duration::from_secs(10),
            Some(&cancel),
            &tx,
        )
        .unwrap();
        assert!(out.ok());
        assert!(!out.cancelled);
    }
}
