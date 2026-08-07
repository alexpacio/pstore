//! pstore — versioned prompt authoring, with the local model that scores your installed
//! coding agents against what you wrote.
//!
//! **This crate is the shared core, and it has no user interface in it.** Three front ends sit on
//! top of exactly these modules:
//!
//! | Front end | Where | What it is for |
//! | --- | --- | --- |
//! | GUI | [`ui`] (feature `gui`) | the editor: writing, versions, diffs, review of every proposal |
//! | TUI | [`tui`] (feature `tui`) | the same editor over a terminal, for a remote machine or a tiling window manager |
//! | CLI | [`cli`] | one action, one exit code, no window — for scripts, hooks and CI |
//!
//! The split is a library boundary rather than a convention. [`app::App`] holds every piece of
//! state a front end needs and every operation one can perform, and it does not name a widget
//! toolkit: it owns the buffer, the version store, the job runner and the proposals awaiting
//! review, and it advances by being handed [`jobs::Event`]s. A front end renders that state and
//! calls those methods. Nothing about ranking, shrinking, planning, sanitising, launching an agent
//! or reading a config file is reachable only from one of them — which is what makes
//! `pstore rank` and the Score models button the same code path with the same answer.
//!
//! What is *not* shared is presentation, and deliberately: a GUI can show a side-by-side diff, a
//! terminal shows a unified one, and a CLI prints JSON when asked. That is the only layer where
//! the three differ.

pub mod agents;
pub mod app;
pub mod cli;
pub mod config;
pub mod editor;
pub mod filter;
pub mod hints;
pub mod jobs;
pub mod knowledge;
pub mod models;
pub mod pii;
pub mod plan;
pub mod router;
pub mod runtime;
pub mod shrink;
pub mod store;
#[cfg(feature = "tui")]
pub mod tui;
#[cfg(feature = "gui")]
pub mod ui;
