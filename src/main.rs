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
mod hints;
mod jobs;
mod models;
mod pii;
mod router;
mod shrink;
mod store;
mod ui;

use std::path::PathBuf;

use clap::Parser;

/// Command-line arguments.
#[derive(Debug, Parser)]
#[command(
    name = "pstore",
    version,
    about = "Write versioned prompts; see how every installed agent, model and effort level scores against them."
)]
struct Args {
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

fn main() -> eframe::Result<()> {
    let args = Args::parse();

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

/// Print what pstore can see, for troubleshooting without launching the GUI.
fn list_agents(config: &config::Config) {
    println!("prompt folder: {}", config.dir.display());
    println!("classifier backend: {}", router::device::probe());

    // The local checkpoints, so "why is it using the built-in scorer?" is answerable
    // without opening the window.
    models::probe_cache();
    println!("\nlocal models (all run in-process; nothing is sent to a service):");
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
