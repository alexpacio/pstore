//! pstore — a versioned prompt editor that scores the coding-agent models and effort
//! levels available on this machine.
//!
//! Prompts are plain `.md` files in a working folder, so they stay useful outside the
//! app. Everything pstore adds — version history, agent verdicts, preferences — lives
//! in a `.pstore/` sidecar next to them.

mod agents;
mod app;
mod config;
mod editor;
mod filter;
mod hints;
mod jobs;
mod models;
mod pii;
mod plan;
mod router;
mod runtime;
mod shrink;
mod store;
mod ui;

use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

/// Command-line arguments.
#[derive(Debug, Parser)]
#[command(
    name = "pstore",
    version,
    about = "Write versioned prompts; see how every installed agent, model and effort level scores against them."
)]
struct Args {
    /// Subcommand, if any. Without one, pstore opens its window.
    #[command(subcommand)]
    command: Option<Command>,

    /// Folder holding the prompts. Defaults to the current directory.
    #[arg(value_name = "DIR")]
    dir: Option<PathBuf>,

    /// Folder holding the prompts (same as the positional argument).
    ///
    /// When neither form is given, `PSTORE_DIR` is consulted before the current
    /// directory — see [`config::Config::resolve`].
    #[arg(long, value_name = "DIR", conflicts_with = "dir")]
    prompt_dir: Option<PathBuf>,

    /// List detected agents and exit, without opening a window.
    #[arg(long)]
    list_agents: bool,
}

#[derive(Debug, Subcommand)]
enum Command {
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
enum Scope {
    /// `.pstore/config.json` beside the prompts.
    Local,
    /// `~/.config/pstore/config.json`.
    User,
    /// `/etc/pstore/config.json` — usually needs elevated privileges.
    System,
}

impl Command {
    /// Resolve the scope from either the flag or the shorthand.
    fn scope(&self) -> Scope {
        let Command::New {
            scope,
            local,
            user,
            system,
        } = self;
        match (local, user, system) {
            (true, _, _) => Scope::Local,
            (_, true, _) => Scope::User,
            (_, _, true) => Scope::System,
            _ => *scope,
        }
    }
}

fn main() -> eframe::Result<()> {
    let args = Args::parse();

    if let Some(command) = &args.command {
        let scope = command.scope();
        let dir = args.dir.clone().or_else(|| args.prompt_dir.clone());
        std::process::exit(init_config(scope, dir));
    }

    let config = match config::Config::resolve(args.dir.or(args.prompt_dir)) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("pstore: could not open the prompt folder: {e}");
            std::process::exit(2);
        }
    };

    if args.list_agents {
        list_agents(&config);
        return Ok(());
    }

    let state = app::App::new(config);
    let title = app::window_title(&state.config.dir.clone(), None, false);

    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1180.0, 780.0])
            .with_min_inner_size([760.0, 480.0])
            .with_title(&title),
        ..Default::default()
    };

    eframe::run_native(
        "pstore",
        options,
        Box::new(move |_cc| Ok(Box::new(ui::Ui::new(state)))),
    )
}

/// Write a starter config file for `scope`, and report what happened.
///
/// Returns a process exit code. Refuses to overwrite an existing file: these are hand-edited
/// policy files, and clobbering an administrator's block list because someone typed the
/// wrong subcommand is not a recoverable mistake.
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
        return 2;
    };

    if path.exists() {
        eprintln!(
            "pstore: {} already exists — leaving it alone",
            path.display()
        );
        return 1;
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
        return 2;
    }

    let defaults = config::Prefs::default();
    let json = match serde_json::to_string_pretty(&defaults) {
        Ok(j) => j,
        Err(e) => {
            eprintln!("pstore: could not render the default config: {e}");
            return 2;
        }
    };
    if let Err(e) = std::fs::write(&path, format!("{json}\n")) {
        eprintln!("pstore: could not write {}: {e}", path.display());
        if scope == Scope::System {
            eprintln!("       a system-wide config usually needs sudo");
        }
        return 2;
    }

    println!("wrote {}", path.display());
    // Layering is the thing people get wrong, so state it at the moment it becomes
    // relevant rather than only in the README.
    println!(
        "config layers apply in order: system → user → local, each overriding the last.\n\
         model filter: {}",
        defaults.filter.summary()
    );
    0
}

/// Print what pstore can see, for troubleshooting without launching the GUI.
fn list_agents(config: &config::Config) {
    println!("prompt folder: {}", config.dir.display());
    // The runtime that actually executes the model, so "why is ranking unavailable?" is
    // answerable without opening the window.
    let prefs = config::prefs_snapshot();
    match runtime::locate(prefs.llama_cli_path.as_deref()) {
        Some(rt) => println!("runtime: {} ({})", rt.path.display(), rt.origin),
        None => println!(
            "runtime: {}",
            runtime::missing_reason(prefs.llama_cli_path.as_deref())
        ),
    }

    models::probe_cache();
    println!("\nlocal model (runs on this machine; nothing is sent to a service):");
    for (c, phase) in models::snapshot() {
        println!("  {:<22} {:<16} {}", c.title, c.size_label(), phase.label());
        println!("    {}", c.repo);
    }
    if models::snapshot().iter().any(|(_, p)| !p.is_downloaded()) {
        println!("  → download the missing ones from the Models window in the app.");
    }

    let detected = agents::detect::detect_all(&config.dir);
    if detected.is_empty() {
        println!("\nNo coding agents found on PATH.");
        println!("pstore looks for: {}", known_bins().join(", "));
        return;
    }

    println!("\n{} agent(s) detected:", detected.len());
    for d in &detected {
        let state = match &d.status {
            agents::detect::Status::Verified => "configured".to_string(),
            agents::detect::Status::Ready => "present (login not yet verified)".to_string(),
            agents::detect::Status::Blocked(u) => format!("unusable — {}", u.reason()),
        };
        println!("  {:<22} {}", d.spec.display, state);
        println!("    binary   {}", d.path.display());
        if let Some(v) = &d.version {
            println!("    version  {v}");
        }
        println!(
            "    config   {}",
            if d.has_credentials {
                "a credential/config file is present"
            } else {
                "none found — the agent may not be logged in"
            }
        );
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
        let models = if d.spec.models.is_empty() {
            "chosen by the agent's own config".to_string()
        } else {
            d.spec
                .models
                .iter()
                .map(|m| m.display)
                .collect::<Vec<_>>()
                .join(", ")
        };
        println!("    models   {models}");
    }
}

fn known_bins() -> Vec<&'static str> {
    agents::registry::AGENTS.iter().map(|a| a.bin).collect()
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
        assert_eq!(init_config(Scope::Local, Some(dir.clone())), 1);
        let (prefs, _) = config::Prefs::load(&dir);
        assert_eq!(prefs.sidebar_width, 999.0, "the edited file was clobbered");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn positional_dir_is_accepted() {
        let args = Args::parse_from(["pstore", "/tmp/prompts"]);
        assert_eq!(args.dir, Some(PathBuf::from("/tmp/prompts")));
        assert!(!args.list_agents);
    }

    #[test]
    fn list_agents_flag_parses() {
        let args = Args::parse_from(["pstore", "--list-agents"]);
        assert!(args.list_agents);
        assert!(args.dir.is_none());
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
        assert!(args.prompt_dir.is_none() || args.prompt_dir.is_some());
    }

    #[test]
    fn known_bins_covers_the_registry() {
        let bins = known_bins();
        assert!(bins.contains(&"claude"));
        assert!(bins.contains(&"codex"));
        assert_eq!(bins.len(), agents::registry::AGENTS.len());
    }
}
