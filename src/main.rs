//! pstore's binary: parse the command line, hand over to a front end, and make sure nothing
//! outlives the process.
//!
//! Everything else is in the library — see [`pstore`], which explains the split. This file exists
//! to own the two things a binary has to: the exit code, and the guarantee below.

use std::process::ExitCode;

fn main() -> ExitCode {
    let code = pstore::cli::main();

    // The one thing that must not survive `main`: a `llama-completion` still generating, holding
    // 3.8 or 7.17 GB of weights with nothing left to show the answer to. Each front end stops the
    // model on its own way out; this covers the paths that never reach that — a windowing error, a
    // panic in a front end, a terminal that went away — because closing a process does not kill
    // its children.
    pstore::router::shutdown_model();

    ExitCode::from(u8::try_from(code).unwrap_or(2))
}
