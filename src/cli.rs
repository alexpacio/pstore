//! The command line: one action, one exit code, no window.
//!
//! Every subcommand here is the same code the GUI and the TUI run — [`crate::router::rank`],
//! [`crate::shrink::run`], [`crate::pii::sanitize`] — so `pstore rank` and the **Score models**
//! button give the same answer for the same prompt. What differs is only how the result is
//! presented, and that is the point of the split: a front end is a renderer.
//!
//! Two conventions the whole surface keeps, because a CLI is something other programs use:
//!
//! * **`--json` on everything that produces a result.** The human table is for reading and is
//!   allowed to change; the JSON is a contract. Nothing goes to stdout except the result, so
//!   `pstore rank p.md --json | jq .best.model` works and progress notes go to stderr.
//! * **Exit codes mean something.** `0` succeeded, `1` the operation ran and reported a problem
//!   (nothing to rank, a scan found nothing), `2` pstore could not do the thing at all (no such
//!   file, no model downloaded). A script can tell "no personal data in this prompt" from
//!   "the checkpoint is missing" without parsing English.
//!
//! Without a subcommand pstore opens its window, which is why the positional directory argument
//! lives on the top-level command rather than under a `gui` subcommand.

use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand, ValueEnum};
use serde_json::{Value, json};

use crate::config::{self, Config};
use crate::router::Ranking;
use crate::store::version::Note;

/// Exit code for "the operation ran, and the answer is no".
const REPORTED: i32 = 1;
/// Exit code for "pstore could not do this".
const FAILED: i32 = 2;

/// Command-line arguments.
#[derive(Debug, Parser)]
#[command(
    name = "pstore",
    version,
    about = "Write versioned prompts; see how every installed agent, model and effort level scores against them."
)]
pub struct Args {
    /// Subcommand, if any. Without one, pstore opens its window.
    #[command(subcommand)]
    command: Option<Command>,

    /// Folder holding the prompts. Defaults to the current directory.
    #[arg(value_name = "DIR", global = true)]
    dir: Option<PathBuf>,

    /// Folder holding the prompts (same as the positional argument).
    ///
    /// When neither form is given, `PSTORE_DIR` is consulted before the current
    /// directory — see [`config::Config::resolve`].
    #[arg(long, value_name = "DIR", conflicts_with = "dir", global = true)]
    prompt_dir: Option<PathBuf>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Open the terminal interface instead of the window.
    Tui,

    /// Rank the installed agents, models and effort levels against a prompt.
    Rank {
        /// Prompt file, or `-` for stdin.
        #[arg(value_name = "FILE")]
        file: String,
        /// Emit JSON rather than a table.
        #[arg(long)]
        json: bool,
    },

    /// Rewrite a prompt telegraphically with the local model.
    Shrink {
        /// Prompt file, or `-` for stdin.
        #[arg(value_name = "FILE")]
        file: String,
        /// Overwrite the file, taking a version snapshot first.
        #[arg(long, conflicts_with = "json")]
        write: bool,
        /// Emit JSON rather than the rewritten prompt.
        #[arg(long)]
        json: bool,
    },

    /// Turn a rough prompt into a structured instruction, using an installed agent.
    Plan {
        /// Prompt file, or `-` for stdin.
        #[arg(value_name = "FILE")]
        file: String,
        /// Use this agent instead of ranking first. An id from `pstore agents`.
        #[arg(long, value_name = "ID")]
        agent: Option<String>,
        /// Overwrite the file, taking a version snapshot first.
        #[arg(long, conflicts_with = "json")]
        write: bool,
        /// Emit JSON rather than the plan.
        #[arg(long)]
        json: bool,
    },

    /// Find personal data in a prompt and report or mask it.
    Sanitize {
        /// Prompt file, or `-` for stdin.
        #[arg(value_name = "FILE")]
        file: String,
        /// Print the masked prompt rather than a report.
        #[arg(long)]
        masked: bool,
        /// Overwrite the file with the masked prompt, taking a version snapshot first.
        #[arg(long, conflicts_with = "json")]
        write: bool,
        /// Emit JSON rather than a report.
        #[arg(long)]
        json: bool,
    },

    /// Show which coding agents are installed and usable.
    Agents {
        /// Emit JSON rather than a report.
        #[arg(long)]
        json: bool,
    },

    /// Show the local checkpoints and the runtime that executes them.
    Models {
        /// Emit JSON rather than a report.
        #[arg(long)]
        json: bool,
    },

    /// List the prompts in the folder.
    List {
        /// Emit JSON rather than a table.
        #[arg(long)]
        json: bool,
    },

    /// Show a prompt's version history.
    Versions {
        /// Prompt file.
        #[arg(value_name = "FILE")]
        file: String,
        /// Print one version's text instead of the list.
        #[arg(long, value_name = "STAMP")]
        show: Option<String>,
        /// Diff one version against the current text.
        #[arg(long, value_name = "STAMP", conflicts_with = "show")]
        diff: Option<String>,
        /// Emit JSON rather than a table.
        #[arg(long)]
        json: bool,
    },

    /// Write a starter config file, without opening a window.
    New {
        /// Which layer to create. Defaults to the current folder.
        #[arg(value_enum, long, default_value_t = Scope::Local)]
        scope: Scope,

        /// Shorthand for `--scope local`.
        #[arg(long, conflicts_with_all = ["scope", "user", "system"])]
        local: bool,
        /// Shorthand for `--scope user`.
        #[arg(long, conflicts_with_all = ["scope", "local", "system"])]
        user: bool,
        /// Shorthand for `--scope system`.
        #[arg(long, conflicts_with_all = ["scope", "local", "user"])]
        system: bool,
    },
}

/// Which configuration layer `pstore new` should create.
///
/// The layers are read in this order, each overriding the one before, so which one to
/// write is a question of scope of intent: a policy for the machine, a preference for the
/// person, or a setting for this project.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Scope {
    /// `.pstore/config.json` beside the prompts.
    Local,
    /// `~/.config/pstore/config.json`.
    User,
    /// `/etc/pstore/config.json` — usually needs elevated privileges.
    System,
}

impl Command {
    /// Resolve `new`'s scope from either the flag or the shorthand.
    fn scope(&self) -> Scope {
        match self {
            Command::New {
                scope,
                local,
                user,
                system,
            } => match (local, user, system) {
                (true, _, _) => Scope::Local,
                (_, true, _) => Scope::User,
                (_, _, true) => Scope::System,
                _ => *scope,
            },
            _ => Scope::Local,
        }
    }
}

/// Parse the command line and do what it says. Returns the process exit code.
pub fn main() -> i32 {
    run(Args::parse())
}

/// Dispatch already-parsed arguments, so the whole surface is testable without a process.
pub fn run(args: Args) -> i32 {
    let dir = args.dir.clone().or_else(|| args.prompt_dir.clone());

    // `new` writes a config rather than reading one, so it runs before resolution — resolving
    // first would create the very folder it is about to report on.
    if let Some(cmd @ Command::New { .. }) = &args.command {
        return init_config(cmd.scope(), dir);
    }

    let config = match Config::resolve(dir) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("pstore: could not open the prompt folder: {e}");
            return FAILED;
        }
    };
    for warning in &config.warnings {
        eprintln!("pstore: {warning}");
    }

    match args.command {
        None => gui(config),
        Some(Command::Tui) => tui(config),
        Some(Command::Rank { file, json }) => rank(&config, &file, json),
        Some(Command::Shrink { file, write, json }) => shrink(&config, &file, write, json),
        Some(Command::Plan {
            file,
            agent,
            write,
            json,
        }) => plan(&config, &file, agent.as_deref(), write, json),
        Some(Command::Sanitize {
            file,
            masked,
            write,
            json,
        }) => sanitize(&config, &file, masked, write, json),
        Some(Command::Agents { json }) => agents(&config, json),
        Some(Command::Models { json }) => models(&config, json),
        Some(Command::List { json }) => list(&config, json),
        Some(Command::Versions {
            file,
            show,
            diff,
            json,
        }) => versions(&config, &file, show.as_deref(), diff.as_deref(), json),
        Some(Command::New { .. }) => unreachable!("handled above, before config resolution"),
    }
}

// ---------------------------------------------------------------------------
// Front ends
// ---------------------------------------------------------------------------

#[cfg(feature = "gui")]
fn gui(config: Config) -> i32 {
    match crate::ui::launch(config) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("pstore: could not open a window: {e}");
            eprintln!("       try `pstore tui` for the terminal interface");
            FAILED
        }
    }
}

/// A build without the GUI still has two working front ends, so it says which rather than
/// failing as though pstore were broken.
#[cfg(not(feature = "gui"))]
fn gui(_config: Config) -> i32 {
    eprintln!("pstore: this build has no window (compiled without the `gui` feature)");
    eprintln!("       run `pstore tui`, or one of the subcommands — see `pstore --help`");
    FAILED
}

#[cfg(feature = "tui")]
fn tui(config: Config) -> i32 {
    match crate::tui::launch(config) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("pstore: {e}");
            FAILED
        }
    }
}

#[cfg(not(feature = "tui"))]
fn tui(_config: Config) -> i32 {
    eprintln!("pstore: this build has no terminal interface (compiled without the `tui` feature)");
    FAILED
}

// ---------------------------------------------------------------------------
// Reading and writing prompts
// ---------------------------------------------------------------------------

/// Read the prompt named by `file`, which may be `-` for stdin.
///
/// A bare name is resolved inside the prompt folder as well as against the working directory, so
/// `pstore rank refactor.md` works from anywhere with `PSTORE_DIR` set — that is the form a hook
/// or a Makefile will use.
fn read_prompt(config: &Config, file: &str) -> Result<(String, Option<PathBuf>), String> {
    if file == "-" {
        use std::io::Read;
        let mut text = String::new();
        std::io::stdin()
            .read_to_string(&mut text)
            .map_err(|e| format!("reading stdin: {e}"))?;
        return Ok((text, None));
    }

    let direct = PathBuf::from(file);
    let candidates = if direct.is_absolute() {
        vec![direct]
    } else {
        vec![direct.clone(), config.dir.join(&direct)]
    };
    for path in candidates {
        if path.is_file() {
            let text = std::fs::read_to_string(&path)
                .map_err(|e| format!("reading {}: {e}", path.display()))?;
            return Ok((text, Some(path)));
        }
    }
    Err(format!("no such prompt: {file}"))
}

/// Replace a prompt's text, taking a version snapshot first.
///
/// The snapshot is not optional and not a courtesy. `--write` on a rewrite that the user has not
/// seen is the one place the CLI destroys something, and the same history the GUI keeps is what
/// makes it recoverable — `pstore versions` will show it.
fn write_prompt(config: &Config, path: &Path, text: &str, note: Note) -> Result<(), String> {
    let slug = path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| format!("{} has no usable name", path.display()))?;
    let before = std::fs::read_to_string(path).unwrap_or_default();

    crate::store::version::snapshot(&config.dir, slug, &before, note)
        .map_err(|e| format!("could not snapshot {slug}: {e}"))?;
    std::fs::write(path, text).map_err(|e| format!("writing {}: {e}", path.display()))?;
    eprintln!("pstore: wrote {} (previous version kept)", path.display());
    Ok(())
}

/// Print `value` as JSON on stdout.
fn emit(value: &Value) {
    println!(
        "{}",
        serde_json::to_string_pretty(value).unwrap_or_else(|_| "{}".into())
    );
}

// ---------------------------------------------------------------------------
// rank
// ---------------------------------------------------------------------------

fn rank(config: &Config, file: &str, as_json: bool) -> i32 {
    let (text, _) = match read_prompt(config, file) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("pstore: {e}");
            return FAILED;
        }
    };
    if text.trim().is_empty() {
        eprintln!("pstore: the prompt is empty — nothing to rank");
        return REPORTED;
    }

    eprintln!("pstore: ranking with the local model — this takes tens of seconds…");
    let detected = crate::agents::detect::detect_all(&config.dir);
    let ranking = match crate::router::rank(&text, &detected, &config.prefs.filter) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("pstore: {e}");
            return FAILED;
        }
    };

    if as_json {
        emit(&ranking_json(&ranking));
    } else {
        print_ranking(&ranking);
    }
    // A ranking the model did not actually make is reported, not returned as success: a script
    // that acts on `.best` should be able to tell.
    if ranking.degenerate.is_some() {
        return REPORTED;
    }
    0
}

/// The JSON shape for a ranking. Stable: other programs read this.
fn ranking_json(ranking: &Ranking) -> Value {
    let choice = |c: &crate::router::Choice| {
        json!({
            "agent": c.agent_id,
            "agent_name": c.agent_display,
            "model": c.model_id,
            "model_name": c.model_display,
            "tier": c.tier.to_string(),
            "effort": c.effort.as_str(),
            "effort_selectable": c.effort_selectable,
            "metered": c.metered,
            "fit": c.fit,
            "relative_latency": c.relative_latency,
            "reason": c.rationale,
        })
    };
    json!({
        "difficulty": ranking.demand.as_ref().map(|(label, _)| *label),
        "difficulty_because": ranking.demand.as_ref().map(|(_, why)| why.clone()),
        "best": ranking.best().map(choice),
        "choices": ranking.choices.iter().map(choice).collect::<Vec<_>>(),
        "considered": ranking.considered,
        "described": ranking.described,
        "excluded": ranking.excluded.iter()
            .map(|(id, why)| json!({"agent": id, "reason": why}))
            .collect::<Vec<_>>(),
        // Present and null when the answer is sound, so a consumer can check the field exists.
        "degenerate": ranking.degenerate,
        "seconds": ranking.elapsed.as_secs_f32(),
    })
}

fn print_ranking(ranking: &Ranking) {
    println!(
        "top {} of {} combinations · {:.1}s",
        ranking.choices.len(),
        ranking.considered,
        ranking.elapsed.as_secs_f32()
    );
    // The premise of everything below: a shortlist that looks wrong is usually a difficulty read
    // that was wrong, and this is the line that lets someone see which.
    if let Some((label, because)) = &ranking.demand {
        println!(
            "judged {label}{}",
            if because.is_empty() {
                String::new()
            } else {
                format!(" — {because}")
            }
        );
    }
    if let Some(why) = &ranking.degenerate {
        println!("\n!! this is a list, not a ranking — {why}");
        println!("   treat the order below as unreliable\n");
    }
    println!(
        "{:>4}  {:<18} {:<26} {:<8} why",
        "fit", "agent", "model", "effort"
    );
    for c in &ranking.choices {
        let effort = if c.effort_selectable {
            c.effort.to_string()
        } else {
            format!("~{}", c.effort)
        };
        println!(
            "{:>4.0}  {:<18} {:<26} {:<8} {}",
            c.fit, c.agent_display, c.model_display, effort, c.rationale
        );
    }
    for (id, why) in &ranking.excluded {
        println!("  excluded · {id}: {why}");
    }
}

// ---------------------------------------------------------------------------
// shrink
// ---------------------------------------------------------------------------

fn shrink(config: &Config, file: &str, write: bool, as_json: bool) -> i32 {
    let (text, path) = match read_prompt(config, file) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("pstore: {e}");
            return FAILED;
        }
    };
    if write && path.is_none() {
        eprintln!("pstore: --write needs a file, not stdin");
        return FAILED;
    }

    let cancel = std::sync::atomic::AtomicBool::new(false);
    let mut note = |text: String| eprintln!("pstore: {text}");
    let shrunk = match crate::shrink::run(&text, &cancel, &mut note) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("pstore: {e}");
            return FAILED;
        }
    };
    let savings = crate::shrink::Savings::measure(&text, &shrunk);
    let warnings = crate::shrink::integrity_warnings(&text, &shrunk);

    if as_json {
        emit(&json!({
            "before": text,
            "after": shrunk,
            "before_chars": savings.before_chars,
            "after_chars": savings.after_chars,
            "saved_percent": savings.percent_saved(),
            "approx_tokens_saved": savings.approx_tokens_saved(),
            "worthwhile": savings.worthwhile(),
            "warnings": warnings,
        }));
    } else {
        eprintln!("pstore: {}", savings.summary());
        for w in &warnings {
            eprintln!("pstore: warning — {w}");
        }
        if write {
            if let Err(e) = write_prompt(
                config,
                path.as_ref().expect("checked above"),
                &shrunk,
                Note::Shrink,
            ) {
                eprintln!("pstore: {e}");
                return FAILED;
            }
        } else {
            print!("{shrunk}");
        }
    }
    // A rewrite that saved nothing is a legitimate answer about an already-terse prompt, and a
    // script that pipes it should still be able to notice.
    if savings.worthwhile() { 0 } else { REPORTED }
}

// ---------------------------------------------------------------------------
// plan
// ---------------------------------------------------------------------------

fn plan(config: &Config, file: &str, agent: Option<&str>, write: bool, as_json: bool) -> i32 {
    let (text, path) = match read_prompt(config, file) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("pstore: {e}");
            return FAILED;
        }
    };
    if write && path.is_none() {
        eprintln!("pstore: --write needs a file, not stdin");
        return FAILED;
    }

    let detected = crate::agents::detect::detect_all(&config.dir);
    // Planning runs on an installed coding agent, not the local checkpoint, so it spends whatever
    // that agent costs. `--agent` skips the ranking call, which is the slower half of the wait.
    let ranking = match agent {
        Some(id) => match pinned(&detected, id) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("pstore: {e}");
                return FAILED;
            }
        },
        None => {
            eprintln!("pstore: choosing an agent with the local model…");
            match crate::router::rank(&text, &detected, &config.prefs.filter) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("pstore: {e}");
                    return FAILED;
                }
            }
        }
    };
    if let Some(best) = ranking.best() {
        eprintln!(
            "pstore: planning with {} · {} · effort {}",
            best.agent_display, best.model_display, best.effort
        );
    }

    let (tx, rx) = std::sync::mpsc::channel::<String>();
    // Drained on its own thread: the launcher streams into this channel, and nothing reading it
    // would stall the agent rather than fail it.
    let drain = std::thread::spawn(move || rx.into_iter().count());
    let outcome = crate::agents::failover::run_with_failover(
        &detected,
        &ranking,
        &crate::plan::compose(&text),
        &config.dir,
        &config.dir,
        std::time::Duration::from_secs(600),
        None,
        &tx,
    );
    drop(tx);
    let _ = drain.join();

    let done = match outcome {
        Ok(done) => done,
        Err(failed) => {
            eprintln!("pstore: {}", failed.summary());
            return FAILED;
        }
    };
    let planned = crate::shrink::clean(&done.text);
    let warnings = crate::plan::warnings(&planned, &text);

    if as_json {
        emit(&json!({
            "plan": planned,
            "agent": done.agent_id,
            "model": done.model_id,
            "effort": done.effort.as_str(),
            "seconds": done.elapsed.as_secs_f32(),
            "warnings": warnings,
        }));
    } else {
        for w in &warnings {
            eprintln!("pstore: warning — {w}");
        }
        if write {
            if let Err(e) = write_prompt(
                config,
                path.as_ref().expect("checked above"),
                &planned,
                Note::Plan,
            ) {
                eprintln!("pstore: {e}");
                return FAILED;
            }
        } else {
            println!("{planned}");
        }
    }
    0
}

/// A one-choice ranking naming `id`, for `--agent`.
///
/// Built from the registry so the launch parameters are the real ones. The model is the agent's
/// first, and the effort its lowest: with nothing ranked there is no judgement to honour, and
/// guessing upwards would spend the user's quota on a decision they did not ask pstore to make.
fn pinned(detected: &[crate::agents::detect::Detected], id: &str) -> Result<Ranking, String> {
    let found = detected
        .iter()
        .find(|d| d.spec.id == id)
        .ok_or_else(|| format!("{id} is not installed — see `pstore agents`"))?;
    if !found.usable() {
        return Err(format!("{id} is installed but not usable"));
    }
    let model = found.spec.models.first();
    let effort = *found
        .spec
        .scoreable_efforts()
        .first()
        .expect("every agent has an effort");

    Ok(Ranking {
        choices: vec![crate::router::Choice {
            agent_id: found.spec.id,
            agent_display: found.spec.display,
            model_id: model.map(|m| m.id.into()).unwrap_or_default(),
            model_display: model
                .map(|m| m.display.into())
                .or_else(|| found.configured_model.clone().map(Into::into))
                .unwrap_or_else(|| "(agent default)".into()),
            tier: model.map_or(crate::agents::registry::Tier::Mid, |m| m.tier),
            effort,
            effort_selectable: found.spec.effort_flag.is_supported(),
            metered: model.is_some_and(|m| m.metered),
            relative_latency: 1.0,
            relative_price: 1.0,
            fit: 0.0,
            rationale: "chosen with --agent".into(),
            row_index: 0,
        }],
        considered: 1,
        ..Ranking::default()
    })
}

// ---------------------------------------------------------------------------
// sanitize
// ---------------------------------------------------------------------------

fn sanitize(config: &Config, file: &str, masked: bool, write: bool, as_json: bool) -> i32 {
    let (text, path) = match read_prompt(config, file) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("pstore: {e}");
            return FAILED;
        }
    };
    if write && path.is_none() {
        eprintln!("pstore: --write needs a file, not stdin");
        return FAILED;
    }

    // A scan that could not run is a failure, never an empty result: "no personal data found" is
    // a claim, and making it without having looked is how personal data reaches an agent.
    let scan = match crate::pii::sanitize(&text) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("pstore: {e}");
            return FAILED;
        }
    };
    let out = scan.plan.apply(&text);

    if as_json {
        emit(&json!({
            "findings": scan.plan.items.iter().map(|item| json!({
                "tag": item.finding.tag,
                "text": item.finding.text,
                "start": item.finding.start,
                "end": item.finding.end,
                "masked": item.masked,
                "placeholder": item.placeholder,
            })).collect::<Vec<_>>(),
            "counts": scan.plan.counts().iter()
                .map(|(tag, n)| json!({"tag": tag, "count": n}))
                .collect::<Vec<_>>(),
            "summary": scan.plan.summary(),
            "masked_prompt": out,
        }));
    } else if masked && !write {
        print!("{out}");
    } else {
        println!("{}", scan.plan.summary());
        for item in &scan.plan.items {
            println!(
                "  {:<10} {:<40} → {}",
                item.finding.tag,
                item.finding.text,
                if item.masked {
                    item.placeholder.as_str()
                } else {
                    "(left as is)"
                }
            );
        }
        if write
            && let Err(e) = write_prompt(
                config,
                path.as_ref().expect("checked above"),
                &out,
                Note::Sanitize,
            )
        {
            eprintln!("pstore: {e}");
            return FAILED;
        }
    }
    // Nothing found is the good outcome and still worth a distinct code: a pre-commit hook wants
    // to know whether this prompt is clean without reading prose.
    if scan.plan.enabled() > 0 { REPORTED } else { 0 }
}

// ---------------------------------------------------------------------------
// agents, models, list, versions
// ---------------------------------------------------------------------------

fn agents(config: &Config, as_json: bool) -> i32 {
    let detected = crate::agents::detect::detect_all(&config.dir);

    if as_json {
        emit(&json!({
            "dir": config.dir.display().to_string(),
            "agents": detected.iter().map(|d| json!({
                "id": d.spec.id,
                "name": d.spec.display,
                "path": d.path.display().to_string(),
                "version": d.version,
                "usable": d.usable(),
                "status": match &d.status {
                    crate::agents::detect::Status::Verified => "configured".to_string(),
                    crate::agents::detect::Status::Ready => "present".to_string(),
                    crate::agents::detect::Status::Blocked(u) => u.reason(),
                },
                "has_credentials": d.has_credentials,
                "configured_model": d.configured_model,
                "models": d.spec.models.iter().map(|m| m.id).collect::<Vec<_>>(),
                "efforts": d.spec.scoreable_efforts().iter()
                    .map(|e| e.as_str()).collect::<Vec<_>>(),
            })).collect::<Vec<_>>(),
            "known": crate::agents::registry::AGENTS.iter()
                .map(|a| a.bin).collect::<Vec<_>>(),
        }));
        return 0;
    }

    println!("prompt folder: {}", config.dir.display());
    if detected.is_empty() {
        println!("\nNo coding agents found on PATH.");
        println!(
            "pstore looks for: {}",
            crate::agents::registry::AGENTS
                .iter()
                .map(|a| a.bin)
                .collect::<Vec<_>>()
                .join(", ")
        );
        return REPORTED;
    }

    println!("\n{} agent(s) detected:", detected.len());
    for d in &detected {
        let state = match &d.status {
            crate::agents::detect::Status::Verified => "configured".to_string(),
            crate::agents::detect::Status::Ready => "present (login not yet verified)".to_string(),
            crate::agents::detect::Status::Blocked(u) => format!("unusable — {}", u.reason()),
        };
        println!("  {:<22} {}", d.spec.display, state);
        println!("    binary   {}", d.path.display());
        if let Some(v) = &d.version {
            println!("    version  {v}");
        }
        // The model pstore will actually rank for this agent, which for half of them comes from
        // the agent's own config rather than from the registry.
        let models = if !d.spec.models.is_empty() {
            d.spec
                .models
                .iter()
                .map(|m| m.display)
                .collect::<Vec<_>>()
                .join(", ")
        } else {
            match &d.configured_model {
                Some(m) => format!("{m} (from its own config)"),
                None => "unknown — its config names none, so pstore will not rank it".to_string(),
            }
        };
        println!("    models   {models}");
        let efforts = if d.spec.efforts.is_empty() {
            "not settable by pstore".to_string()
        } else {
            d.spec
                .efforts
                .iter()
                .map(|e| e.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        };
        println!("    effort   {efforts}");
    }
    0
}

fn models(config: &Config, as_json: bool) -> i32 {
    let prefs = &config.prefs;
    let runtime = crate::runtime::locate(prefs.llama_path.as_deref());
    crate::models::probe_cache();
    let board = crate::models::snapshot();
    let active = crate::models::active();

    if as_json {
        emit(&json!({
            "runtime": runtime.as_ref().map(|rt| json!({
                "path": rt.path.display().to_string(),
                "origin": rt.origin.to_string(),
            })),
            "runtime_problem": runtime.is_none()
                .then(|| crate::runtime::missing_reason(prefs.llama_path.as_deref())),
            "selected": active.id,
            "checkpoints": board.iter().map(|(c, phase)| json!({
                "id": c.id,
                "title": c.title,
                "repo": c.repo,
                "bytes": c.bytes,
                "state": phase.label(),
                "downloaded": phase.is_downloaded(),
                "selected": c.id == active.id,
            })).collect::<Vec<_>>(),
        }));
        return if runtime.is_some() { 0 } else { REPORTED };
    }

    match &runtime {
        Some(rt) => println!("runtime: {} ({})", rt.path.display(), rt.origin),
        None => println!(
            "runtime: {}",
            crate::runtime::missing_reason(prefs.llama_path.as_deref())
        ),
    }
    println!("\nlocal model (runs on this machine; nothing is sent to a service):");
    for (c, phase) in &board {
        let marker = if c.id == active.id { "→" } else { " " };
        println!(
            "{marker} {:<22} {:<10} {}",
            c.title,
            c.size_label(),
            phase.label()
        );
        println!("    {}", c.repo);
    }
    if board.iter().any(|(_, p)| !p.is_downloaded()) {
        println!("  → download the missing ones from the Models window in the app.");
    }
    if runtime.is_some() { 0 } else { REPORTED }
}

fn list(config: &Config, as_json: bool) -> i32 {
    let store = crate::store::PromptStore::new(config.dir.clone());
    let prompts = store.list();

    if as_json {
        emit(&json!({
            "dir": config.dir.display().to_string(),
            "prompts": prompts.iter().map(|p| json!({
                "name": p.name,
                "slug": p.slug,
                "path": p.path.display().to_string(),
            })).collect::<Vec<_>>(),
        }));
        return 0;
    }
    if prompts.is_empty() {
        println!("no prompts in {}", config.dir.display());
        return REPORTED;
    }
    for p in &prompts {
        println!("{:<40} {}", p.name, p.path.display());
    }
    0
}

fn versions(
    config: &Config,
    file: &str,
    show: Option<&str>,
    diff: Option<&str>,
    as_json: bool,
) -> i32 {
    let (text, path) = match read_prompt(config, file) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("pstore: {e}");
            return FAILED;
        }
    };
    let Some(slug) = path
        .as_deref()
        .and_then(Path::file_stem)
        .and_then(|s| s.to_str())
    else {
        eprintln!("pstore: version history needs a file, not stdin");
        return FAILED;
    };

    if let Some(stamp) = show.or(diff) {
        let old = match crate::store::version::read(&config.dir, slug, stamp) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("pstore: no version {stamp} of {slug}: {e}");
                return FAILED;
            }
        };
        if diff.is_some() {
            print!("{}", crate::store::version::diff(&old, &text));
        } else {
            print!("{old}");
        }
        return 0;
    }

    let history = crate::store::version::list(&config.dir, slug);
    if as_json {
        emit(&json!({
            "prompt": slug,
            "versions": history.iter().map(|v| json!({
                "stamp": v.ts,
                "note": v.note.label(),
                "bytes": v.bytes,
            })).collect::<Vec<_>>(),
        }));
        return 0;
    }
    if history.is_empty() {
        println!("no saved versions of {slug}");
        return REPORTED;
    }
    for v in &history {
        println!("{:<24} {:<12} {} bytes", v.ts, v.note.label(), v.bytes);
    }
    0
}

// ---------------------------------------------------------------------------
// new
// ---------------------------------------------------------------------------

/// Write a starter config file for `scope`, and report what happened.
///
/// Refuses to overwrite an existing file: these are hand-edited policy files, and clobbering an
/// administrator's block list because someone typed the wrong subcommand is not a recoverable
/// mistake.
fn init_config(scope: Scope, dir: Option<PathBuf>) -> i32 {
    let path = match scope {
        Scope::Local => {
            let dir = dir
                .or_else(|| std::env::var_os("PSTORE_DIR").map(PathBuf::from))
                .or_else(|| std::env::current_dir().ok());
            dir.map(|d| config::Prefs::path(&d))
        }
        Scope::User => config::user_config(),
        Scope::System => config::system_config(),
    };

    let Some(path) = path else {
        eprintln!("pstore: no {scope:?} configuration path on this platform");
        return FAILED;
    };

    if path.exists() {
        eprintln!(
            "pstore: {} already exists — leaving it alone",
            path.display()
        );
        return REPORTED;
    }
    if let Some(parent) = path.parent()
        && let Err(e) = std::fs::create_dir_all(parent)
    {
        // The usual cause for `--system` is running without privileges, so say so rather
        // than reporting a bare permission error.
        eprintln!("pstore: could not create {}: {e}", parent.display());
        if scope == Scope::System {
            eprintln!("       a system-wide config usually needs sudo");
        }
        return FAILED;
    }

    let defaults = config::Prefs::default();
    let json = match serde_json::to_string_pretty(&defaults) {
        Ok(j) => j,
        Err(e) => {
            eprintln!("pstore: could not render the default config: {e}");
            return FAILED;
        }
    };
    if let Err(e) = std::fs::write(&path, format!("{json}\n")) {
        eprintln!("pstore: could not write {}: {e}", path.display());
        if scope == Scope::System {
            eprintln!("       a system-wide config usually needs sudo");
        }
        return FAILED;
    }

    println!("wrote {}", path.display());
    // Layering is the thing people get wrong, so state it at the moment it becomes relevant
    // rather than only in the README.
    println!(
        "config layers apply in order: system → user → local, each overriding the last.\n\
         model filter: {}",
        defaults.filter.summary()
    );
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_defaults_to_the_local_scope() {
        let args = Args::parse_from(["pstore", "new"]);
        let cmd = args.command.expect("a subcommand");
        assert_eq!(cmd.scope(), Scope::Local, "`pstore new` means --local");
    }

    /// The shorthands are the forms people will actually type, so each has to reach the
    /// scope it names — writing a user config when `--system` was asked for would be a
    /// silent no-op from the administrator's point of view.
    #[test]
    fn new_shorthands_select_their_scope() {
        for (flag, want) in [
            ("--local", Scope::Local),
            ("--user", Scope::User),
            ("--system", Scope::System),
        ] {
            let args = Args::parse_from(["pstore", "new", flag]);
            assert_eq!(args.command.unwrap().scope(), want, "for {flag}");
        }
        // The explicit form works too.
        let args = Args::parse_from(["pstore", "new", "--scope", "user"]);
        assert_eq!(args.command.unwrap().scope(), Scope::User);
    }

    #[test]
    fn new_scope_flags_are_mutually_exclusive() {
        assert!(Args::try_parse_from(["pstore", "new", "--user", "--system"]).is_err());
        assert!(Args::try_parse_from(["pstore", "new", "--local", "--scope", "user"]).is_err());
    }

    /// `pstore new` must write a config the app can actually read back, or the starter file
    /// is worse than no file.
    #[test]
    fn new_writes_a_config_that_loads_and_refuses_to_clobber() {
        let dir = std::env::temp_dir().join(format!(
            "pstore-new-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();

        assert_eq!(init_config(Scope::Local, Some(dir.clone())), 0);
        let path = config::Prefs::path(&dir);
        assert!(path.is_file(), "no config written");

        let (prefs, warnings) = config::Prefs::load(&dir);
        assert!(
            warnings.is_empty(),
            "the file it wrote must parse: {warnings:?}"
        );
        assert!(
            prefs.filter.block_metered,
            "the starter config should carry the default policy"
        );

        // A second run must not overwrite a file someone may have edited.
        std::fs::write(&path, r#"{"sidebar_width": 999.0}"#).unwrap();
        assert_eq!(init_config(Scope::Local, Some(dir.clone())), REPORTED);
        let (prefs, _) = config::Prefs::load(&dir);
        assert_eq!(prefs.sidebar_width, 999.0, "the edited file was clobbered");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn positional_dir_is_accepted() {
        let args = Args::parse_from(["pstore", "/tmp/prompts"]);
        assert_eq!(args.dir, Some(PathBuf::from("/tmp/prompts")));
        assert!(args.command.is_none(), "a bare directory opens the window");
    }

    #[test]
    fn positional_and_flag_forms_conflict() {
        // Accepting both would leave the working folder ambiguous.
        assert!(Args::try_parse_from(["pstore", "/a", "--prompt-dir", "/b"]).is_err());
    }

    #[test]
    fn no_arguments_is_valid() {
        let args = Args::parse_from(["pstore"]);
        assert!(args.dir.is_none());
        assert!(args.command.is_none());
    }

    /// Every front end has to be reachable, and every result-producing command has to offer
    /// `--json` — a subcommand that can only be read by a person is one a hook cannot use.
    #[test]
    fn the_command_surface_is_reachable_and_scriptable() {
        assert!(matches!(
            Args::parse_from(["pstore", "tui"]).command,
            Some(Command::Tui)
        ));

        for cmd in [
            vec!["rank", "p.md"],
            vec!["shrink", "p.md"],
            vec!["plan", "p.md"],
            vec!["sanitize", "p.md"],
            vec!["agents"],
            vec!["models"],
            vec!["list"],
            vec!["versions", "p.md"],
        ] {
            let mut argv = vec!["pstore"];
            argv.extend(cmd.iter());
            argv.push("--json");
            assert!(
                Args::try_parse_from(&argv).is_ok(),
                "{cmd:?} should accept --json"
            );
        }
    }

    /// `--write` destroys the file it is given, so it must not be combinable with the flag that
    /// says "print this instead", and it must refuse stdin rather than write somewhere arbitrary.
    #[test]
    fn write_and_json_are_mutually_exclusive() {
        for cmd in ["shrink", "sanitize", "plan"] {
            assert!(
                Args::try_parse_from(["pstore", cmd, "p.md", "--write", "--json"]).is_err(),
                "{cmd} allowed --write with --json"
            );
            assert!(Args::try_parse_from(["pstore", cmd, "p.md", "--write"]).is_ok());
        }
    }

    /// A bare prompt name is looked for in the prompt folder as well as the working directory,
    /// because that is the form a hook or a Makefile will use.
    #[test]
    fn prompts_resolve_against_the_prompt_folder() {
        let dir = std::env::temp_dir().join(format!(
            "pstore-cli-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("thing.md"), "the prompt").unwrap();

        let config = Config {
            dir: dir.clone(),
            prefs: config::Prefs::default(),
            warnings: Vec::new(),
        };
        let (text, path) = read_prompt(&config, "thing.md").expect("found in the prompt folder");
        assert_eq!(text, "the prompt");
        assert_eq!(path, Some(dir.join("thing.md")));

        // An absolute path is taken as given.
        let (text, _) = read_prompt(&config, dir.join("thing.md").to_str().unwrap()).unwrap();
        assert_eq!(text, "the prompt");

        // And a name that is nowhere is an error rather than an empty prompt, which would
        // otherwise be ranked or scanned as though it were the user's document.
        let why = read_prompt(&config, "absent.md").expect_err("no such prompt");
        assert!(why.contains("absent.md"), "got {why}");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// `--agent` skips ranking, so the choice it builds has to be launchable: a real registry
    /// row, with an effort the agent accepts.
    #[test]
    fn a_pinned_agent_becomes_a_launchable_choice() {
        use crate::agents::detect::{Detected, Status};

        let spec = crate::agents::registry::find("claude").unwrap();
        let detected = vec![Detected {
            spec,
            path: PathBuf::from("/usr/bin/claude"),
            version: None,
            has_credentials: true,
            status: Status::Ready,
            configured_model: None,
        }];

        let ranking = pinned(&detected, "claude").expect("claude is installed here");
        let best = ranking.best().expect("one choice");
        assert_eq!(best.agent_id, "claude");
        assert!(
            spec.models.iter().any(|m| m.id == best.model_id),
            "{} is not a model claude exposes",
            best.model_id
        );
        assert!(
            spec.scoreable_efforts().contains(&best.effort),
            "{:?} is not an effort claude accepts",
            best.effort
        );

        // An agent that is not there has to say so rather than producing an unlaunchable pick.
        assert!(pinned(&detected, "codex").is_err());
    }
}
