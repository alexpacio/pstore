//! The command line as another program sees it.
//!
//! Everything else in the suite tests pstore's functions. This tests its *binary*: the exit
//! codes, and the promise that `--json` puts the result on stdout and nothing else.
//!
//! That promise is the one with no other coverage and the most to lose. The README tells
//! people to write `pstore sanitize prompt.md --json` into a pre-commit hook and branch on the
//! status — "so a hook does not have to parse English" — and a hook that reads the wrong code
//! either blocks every commit or blocks none of them. Three codes, each meaning something
//! different:
//!
//! | code | meaning |
//! | --- | --- |
//! | 0 | it ran and the answer is yes |
//! | 1 | it ran and the answer is no — nothing to rank, personal data *was* found, a rewrite saved nothing |
//! | 2 | it could not run at all — no such file, no checkpoint |
//!
//! The distinction that matters is 1 against 2. Both are "not a success", and collapsing them
//! turns "this prompt is clean" into "pstore is broken" or the reverse.
//!
//! **Nothing here runs the local model.** Every case is a command that answers from the
//! filesystem, or one that fails before it would have loaded a checkpoint — so the suite stays
//! runnable on a machine that has never downloaded one, and takes milliseconds rather than the
//! tens of seconds a ranking costs.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// The binary this test was built alongside.
const EXE: &str = env!("CARGO_BIN_EXE_pstore");

/// A directory of its own for one test, removed and recreated so a rerun starts clean.
fn workdir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("pstore-cli-{}-{name}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Run pstore in `dir` and hand back everything the shell would see.
///
/// `PSTORE_DIR` is cleared rather than left alone: it is consulted when no folder is given, so
/// a developer who exports it would otherwise have these tests reading their own prompts.
fn run(dir: &Path, args: &[&str]) -> Output {
    Command::new(EXE)
        .arg(dir)
        .args(args)
        .env_remove("PSTORE_DIR")
        .output()
        .expect("pstore should be runnable")
}

fn code(out: &Output) -> i32 {
    out.status
        .code()
        .expect("pstore should exit, not be signalled")
}

fn stdout(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// `2` is "could not do it at all", and a name that is nowhere is the commonest way to get
/// there. Every command that takes a prompt has to agree, because a hook that runs two of them
/// should not need to know which one it is reading.
#[test]
fn a_prompt_that_does_not_exist_is_two_from_every_command() {
    let dir = workdir("missing");
    for cmd in ["rank", "shrink", "plan", "rca", "sanitize", "versions"] {
        let out = run(&dir, &[cmd, "nope.md"]);
        assert_eq!(code(&out), 2, "`{cmd} nope.md` should be 2: {out:?}");
        assert!(
            stdout(&out).trim().is_empty(),
            "`{cmd}` put its complaint on stdout, which a pipe would swallow"
        );
        assert!(
            String::from_utf8_lossy(&out.stderr).contains("nope.md"),
            "`{cmd}` should name the prompt it could not find"
        );
    }
}

/// The same, with `--json`: a failure must not print half a document. A consumer that pipes
/// into `jq` should get nothing to parse rather than something that parses wrongly.
#[test]
fn a_failure_writes_no_json() {
    let dir = workdir("nojson");
    for cmd in ["rank", "shrink", "plan", "rca", "sanitize"] {
        let out = run(&dir, &[cmd, "nope.md", "--json"]);
        assert_eq!(code(&out), 2, "`{cmd} --json` should be 2");
        assert!(
            stdout(&out).trim().is_empty(),
            "`{cmd} --json` emitted {:?} for a prompt that does not exist",
            stdout(&out)
        );
    }
}

/// An empty folder is a legitimate answer, not a breakage — `1`, the "the answer is no" code.
/// With `--json` it is `0`, because an empty list *is* the result and it parsed.
#[test]
fn an_empty_folder_lists_as_no_rather_than_broken() {
    let dir = workdir("empty");

    let out = run(&dir, &["list"]);
    assert_eq!(code(&out), 1, "an empty folder is 1, not 2");

    let out = run(&dir, &["list", "--json"]);
    assert_eq!(code(&out), 0, "an empty list is still a result");
    let v: serde_json::Value = serde_json::from_str(&stdout(&out)).expect("stdout must be JSON");
    assert_eq!(v["prompts"].as_array().map(Vec::len), Some(0));

    // ...and with prompts in it, 0 and both names, in a stable order.
    std::fs::write(dir.join("b.md"), "second").unwrap();
    std::fs::write(dir.join("a.md"), "first").unwrap();
    let out = run(&dir, &["list"]);
    assert_eq!(code(&out), 0);
    assert!(stdout(&out).contains("a") && stdout(&out).contains("b"));

    let out = run(&dir, &["list", "--json"]);
    let v: serde_json::Value = serde_json::from_str(&stdout(&out)).unwrap();
    let names: Vec<&str> = v["prompts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p["name"].as_str().unwrap_or_default())
        .collect();
    assert_eq!(names, ["a", "b"], "prompts should list in a stable order");
}

/// A prompt with no history is "nothing to show", not a failure — the distinction a script
/// restoring the last version needs, and the one that separates 1 from 2.
#[test]
fn a_prompt_with_no_versions_is_one_not_two() {
    let dir = workdir("versions");
    std::fs::write(dir.join("p.md"), "some prompt").unwrap();

    let out = run(&dir, &["versions", "p.md"]);
    assert_eq!(code(&out), 1, "no history is 1: {out:?}");

    // The file genuinely missing is the other code, from the same command.
    let out = run(&dir, &["versions", "gone.md"]);
    assert_eq!(code(&out), 2);
}

/// `pstore new` writes a starter config and then refuses to touch it — the second run is `1`
/// ("it ran, and the answer is no"), not `0`, so a provisioning script can tell whether it was
/// the one that created the file.
#[test]
fn new_writes_once_and_then_declines() {
    let dir = workdir("new");

    let out = run(&dir, &["new"]);
    assert_eq!(code(&out), 0, "the first run should write the file");
    let cfg = dir.join(".pstore").join("config.json");
    assert!(cfg.is_file(), "no config at {}", cfg.display());

    // What it wrote has to be readable by the thing that will read it.
    let text = std::fs::read_to_string(&cfg).unwrap();
    let v: serde_json::Value = serde_json::from_str(&text).expect("the starter config must parse");
    assert_eq!(
        v["filter"]["block_metered"], true,
        "the starter config should carry the default policy: {text}"
    );

    // A second run must not clobber an edited file.
    std::fs::write(&cfg, r#"{"sidebar_width": 999.0}"#).unwrap();
    let out = run(&dir, &["new"]);
    assert_eq!(code(&out), 1, "a second run is 1, not 0 and not 2");
    assert_eq!(
        std::fs::read_to_string(&cfg).unwrap(),
        r#"{"sidebar_width": 999.0}"#,
        "the edited config was overwritten"
    );
}

/// `--version` and `--help` are how a script checks it is talking to pstore at all.
#[test]
fn the_binary_identifies_itself() {
    let out = Command::new(EXE).arg("--version").output().unwrap();
    assert_eq!(out.status.code(), Some(0));
    assert!(
        stdout(&out).contains(env!("CARGO_PKG_VERSION")),
        "--version should print the crate version, got {:?}",
        stdout(&out)
    );

    let out = Command::new(EXE).arg("--help").output().unwrap();
    assert_eq!(out.status.code(), Some(0));
    let help = stdout(&out);
    // Every subcommand the README documents has to be reachable from the help text, or it
    // cannot be discovered by the person the help text is for.
    for cmd in [
        "tui", "rank", "shrink", "plan", "rca", "sanitize", "agents", "models", "list", "versions",
        "new",
    ] {
        assert!(help.contains(cmd), "`{cmd}` is missing from --help");
    }
}

/// `--write` edits the file in place and `--json` says "print it instead". Accepting both
/// would leave it unclear which happened, so clap has to refuse — and refusing is a usage
/// error, which is `2`.
#[test]
fn write_and_json_together_are_refused_before_anything_happens() {
    let dir = workdir("conflict");
    std::fs::write(dir.join("p.md"), "some prompt").unwrap();
    let before = std::fs::read_to_string(dir.join("p.md")).unwrap();

    for cmd in ["shrink", "plan", "rca", "sanitize"] {
        let out = run(&dir, &[cmd, "p.md", "--write", "--json"]);
        assert_eq!(code(&out), 2, "`{cmd} --write --json` should be refused");
        assert_eq!(
            std::fs::read_to_string(dir.join("p.md")).unwrap(),
            before,
            "`{cmd}` modified the prompt before refusing the arguments"
        );
    }
}

/// `models` answers from the filesystem, so it works — and reports honestly — on a machine
/// that has never downloaded a checkpoint. This is what the Models window reads.
#[test]
fn models_reports_the_local_checkpoints_without_the_network() {
    let dir = workdir("models");
    let out = run(&dir, &["models", "--json"]);
    assert_eq!(code(&out), 0, "models should succeed: {out:?}");

    let v: serde_json::Value = serde_json::from_str(&stdout(&out)).expect("stdout must be JSON");
    let checkpoints = v["checkpoints"].as_array().expect("a list of checkpoints");
    assert_eq!(checkpoints.len(), 2, "both builds are always offered");

    // Exactly one is selected, and `selected` names it — the two must not disagree.
    let selected: Vec<&serde_json::Value> = checkpoints
        .iter()
        .filter(|c| c["selected"] == true)
        .collect();
    assert_eq!(selected.len(), 1, "exactly one build is in use: {v}");
    assert_eq!(v["selected"], selected[0]["id"]);

    // Every build states its size and whether it is on disk, downloaded or not.
    for c in checkpoints {
        assert!(c["bytes"].as_u64().unwrap_or(0) > 0, "{c} has no size");
        assert!(c["downloaded"].is_boolean(), "{c}");
        assert!(c["repo"].as_str().is_some_and(|r| r.contains('/')), "{c}");
    }
}
