# pstore

**A versioned editor for the prompts you hand to coding agents** — write and revise them like real artifacts, not scratch text. Get inline hints, automatic difficulty/capability scoring to pick the cheapest adequate model, and one-keystroke handoff to a real agent session, from a native window, a terminal UI, or the command line.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=flat-square)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.85%2B-orange?style=flat-square&logo=rust)](https://rust-lang.org)

---

## What it does

| Feature | Description |
|---------|-------------|
| **Versioned prompts** | Every save creates a snapshot in `.pstore/versions/`. Full history, diffs, one-click restore — all plain markdown. |
| **Inline hints** | Select text, type a question, or both — the selection and the question reach the answerer as distinct things. Answered by the local model or by the ranked coding agent, your pick per question. Answers land in a panel, never silently in the document. |
| **Plan** | Rewrites a rough request into a structured instruction for a coding agent: objective, ordered steps, constraints, acceptance criteria. Runs on the local model, and asks it for the fields rather than for a document — the headings and their order are pstore's. The output *is* the next prompt, not a document to read. |
| **RCA** | Turns incident notes into a root cause analysis and postmortem — impact, timeline, root cause, contributing factors, detection, resolution, action items and what is still unknown. Blameless by instruction, and **checked for invention**: a section the notes never established says so rather than being filled in, and any figure in the write-up that the notes did not give is reported back to you before you read it. Each action item is tagged `prevent`/`detect`/`mitigate` — enforced by the schema, not asked for — and `pstore rca --actions` prints just those, one per line. Runs on the local model, so hostnames, customer numbers and stack traces stay on the machine. |
| **Shrink** | Rewrites the selection — or the whole prompt — telegraphically: no articles, no pleasantries, each fact stated once, while code, paths, identifiers and constraints stay verbatim. Arrives as a diff with a size summary and a warning if a file reference or code block went missing. |
| **Smart routing** | Every (agent, model, effort) combination your machine can run is handed to a local model along with your prompt; it returns a ranked shortlist with a reason for each pick. |
| **Only models it can describe** | A candidate pstore cannot name and characterise is **withheld from the ranking**, with the reason shown, rather than ranked blind. |
| **PII sanitizer** | Finds names, addresses, IBANs, tax codes and card numbers and swaps them for placeholders — before the prompt reaches an agent. |
| **One local model** | A single checkpoint — your pick of two sizes — runs everything pstore infers, as a subprocess on your machine. Nothing about your prompt leaves it. |
| **Model policy** | Block models by pattern, or whitelist the only ones you are allowed to use. Per-token models are blocked by default so nothing bills you by accident. Configurable machine-wide, per-user, or per-project. |
| **One-key handoff** | `Ctrl+Enter` launches the selected agent with your prompt. Supports 11 agents. |
| **Three front ends** | The same core behind a native window, a full-screen terminal UI, and a scriptable CLI with `--json` on everything. `pstore rank` and the **Score models** button are one code path. |
| **Copy anything** | Every produced artefact — hint, shrink, plan, masked prompt — has a copy button. The workflow ends in a paste. |
| **Plain files** | Prompts are `.md` in your project. Sidecar in `.pstore/`. Works with git, grep, any editor. |

Nothing is applied unreviewed: shrink, plan, RCA and sanitize all arrive as a diff, and accepting is one undo step with the previous text already in version history.

---

## The local model

Routing, planning, incident analysis, shrinking and sanitizing all run on one local checkpoint:
[**Bonsai 27B**](https://huggingface.co/prism-ml/Bonsai-27B-gguf), from
[PrismML](https://docs.prismml.com/models/bonsai-27b) — a 27B model (derived from Qwen3.6-27B)
compressed to binary or ternary weights rather than trained small from the start. Ordinary
quantization below ~4 bits/weight causes a real model to keep sounding fluent while its
reasoning, tool calls and multi-step plans quietly stop working — exactly the behaviors pstore's
own routing, planning and shrinking depend on. Per PrismML's published benchmarks, Bonsai's
representation holds up far below that line:

| Build | Bits/weight | Size | Benchmark score vs. FP16 |
| --- | --- | --- | --- |
| **1-bit** (binary, pstore's default) | 1.125 | ~3.9 GB | 89.5% |
| **Ternary** | 1.71 | ~5.9–7.2 GB on disk | 94.6% |

**Why 1-bit is the default:** it is the smaller download and the smaller memory footprint, and
routing accuracy on pstore's own workload holds up well at that size once the prompt is asked the
right way — see the model's own docs for how it's built. Pick **ternary** instead in the Models
window or `.pstore/config.json` (`"local_model": "ternary"`) if your machine has memory to spare
and you want a longer, more separated shortlist.

Weights are cached in the shared Hugging Face location (`~/.cache/huggingface`) and run as a
short-lived subprocess — nothing stays resident between operations, and nothing about your
prompt leaves the machine.

### Memory and context sizing

pstore never loads the checkpoint at its native 262K-token context — that would cost several
gigabytes of KV cache for prompts that are, in every one of pstore's own uses, a few hundred to a
few thousand tokens. Instead, each operation sizes its own context window from what it is actually
about to send:

```text
ctx = round_up(prompt_tokens × 1.25 + max_output_tokens + 256, step), capped at model_context_ceiling
```

`model_context_ceiling` defaults to 8192 and is a hard cap, not a target — a ranking call
typically fits in ~1,000–2,000 tokens. The 25% headroom is deliberate: pstore would rather waste a
few megabytes of context than silently truncate a prompt, which is how you'd end up sanitizing
text the model never actually saw.

PrismML's own measurements put peak memory (weights + KV cache + activations, FP16 KV, no
compression) at:

| Build | Weights only | +4K context | +10K context | +100K context |
| --- | --- | --- | --- | --- |
| 1-bit | ~3.8 GB | ~5.2 GB | ~5.6 GB | ~11.6 GB |
| Ternary | ~7.2 GB | ~8.4 GB | ~8.7 GB | ~14.7 GB |

With the 4-bit KV cache pstore actually runs with, that context-dependent growth shrinks roughly
4×: PrismML report the 1-bit build's 100K-token peak dropping to ~6.8 GB and the ternary build's
to ~10.1 GB, with the full 262K-token window fitting in ~9.4 GB and ~12.8 GB respectively.

In practice pstore's own calls stay far below all of this — the context-sizing formula above caps
a ranking or a shrink at `model_context_ceiling` (8,192 tokens by default), so expect peak memory
close to the "weights only" column, not the wide-context end of it. The larger figures matter if
you raise `model_context_ceiling` yourself or point `llama_path` at your own PrismML build and run
it outside pstore.

---

## Installation

### Download a binary (recommended)

Grab the archive for your platform from the [Releases page](https://github.com/alexpacio/pstore/releases/latest), then:

#### macOS (Apple Silicon)

```bash
tar xzf pstore-macos-arm64.tar.gz
xattr -d com.apple.quarantine pstore   # unsigned binary; clears Gatekeeper's block
sudo mv pstore /usr/local/bin/
```

Intel Macs aren't built by CI; use `cargo install` or build from source below instead.

#### Linux

```bash
tar xzf pstore-linux-x86_64.tar.gz
chmod +x pstore
sudo mv pstore /usr/local/bin/
```

#### Windows

```powershell
# Unzip pstore-windows-x86_64.zip, then run the extracted pstore.exe.
# Optionally move it onto your PATH.
```

### Via Cargo (requires Rust 1.85+)

```bash
cargo install --git https://github.com/alexpacio/pstore
```

### From source

```bash
git clone https://github.com/alexpacio/pstore
cd pstore

cargo run --release          # just try it, without installing anything
cargo install --path . --force   # install onto your PATH (~/.cargo/bin/pstore)
```

`--force` overwrites a previous install; drop it if this is the first one.

**No other prerequisites.** pstore downloads the local model and the runtime that executes it on first use, from the Models window.

**One binary, three front ends**, each behind a build feature so you can leave out what you will not run — both on by default:

```bash
# Headless machine — no windowing dependencies compiled at all
cargo build --release --no-default-features --features tui

# Window only
cargo build --release --no-default-features --features gui
```

The local model is **not** a feature. Ranking, planning, analysing an incident, shrinking and sanitizing are all
the local checkpoint, so a build without it is not a smaller pstore — it is a text editor
that refuses every button.

---

## Quick start

```bash
pstore                 # open the window, in the current directory
pstore tui              # the terminal interface — e.g. over ssh on a headless machine
pstore ~/my-prompts     # either front end, pointed at a specific prompt folder
pstore tui ~/my-prompts
```

`pstore rank` and the **Score models** button run the same code and give the same answer — the three front ends only differ in how the result is shown.

### Using the CLI

Every command that produces a result takes `--json`; nothing but the result goes to stdout, so progress notes and warnings stay on stderr and a pipe stays clean.

```bash
pstore rank refactor.md              # ranked shortlist, with the difficulty verdict
pstore rank refactor.md --json | jq -r .best.model
pstore shrink refactor.md            # telegraphic rewrite to stdout
pstore shrink refactor.md --write    # ...or in place, snapshotting the previous version
pstore plan refactor.md              # structured instruction, via an installed agent
pstore plan refactor.md --agent claude   # skip the ranking call and use this one
pstore rca incident.md               # root cause analysis and postmortem, to stdout
pstore rca incident.md --actions     # just the action items, one per line, for the ticket tracker
pstore sanitize refactor.md          # what personal data is in here, and what would mask it
pstore sanitize refactor.md --masked # the masked prompt, ready to pipe

pstore agents                        # what is installed, and which model each will actually run
pstore models                        # the local checkpoints and the runtime that executes them
pstore list                          # prompts in the folder
pstore versions refactor.md          # history; --show STAMP or --diff STAMP for one of them

pstore new                # .pstore/config.json here — same as --local
pstore new --user         # ~/.config/pstore/config.json
sudo pstore new --system  # /etc/pstore/config.json
```

Exit codes mean something, so a hook does not have to parse English: `0` succeeded, `1` the operation ran and the answer is no (nothing to rank, a scan *found* personal data, a rewrite saved nothing), `2` pstore could not do it at all (no such file, no checkpoint downloaded). A pre-commit hook is `pstore sanitize prompt.md --json` and a test on the exit code.

### Using the TUI

Launch with `pstore tui`. Full-screen terminal interface, same operations as the window:

| Action | Key |
| --- | --- |
| Save (creates a version) | `Ctrl+S` |
| Rank the installed models | `Ctrl+R` or `F5` |
| Shrink · Plan · Sanitize | `F2` · `F3` · `F4` |
| Root cause analysis | `F10` |
| Ask about the selection or a question | `Ctrl+H` |
| Undo / Redo | `Ctrl+Z` / `Ctrl+Y` |
| Accept / reject a proposal | `a` / `r` |
| New prompt | `Ctrl+N` |
| Local model status | `F6` |
| Version history / cycle the side pane | `F7` / `F9` |
| Move between the prompt list and the text | `Tab` |
| Stop whatever is running | `Esc` |
| Help | `F1` |
| Quit | `Ctrl+Q` |

### Using the GUI

Launch with `pstore` (no subcommand). Native window with a sidebar of prompts, an editor, and an action bar: **Score models** · **Shrink** · **Plan** · **RCA** · **Sanitize** · **Send →** · **Hint…** · **Models…**

| Action | Key |
| --- | --- |
| Save (creates a version) | `Ctrl+S` |
| Rank the installed models | `Ctrl+R` |
| Ask about the selection or a question | `Ctrl+Enter` |
| Undo / Redo | `Ctrl+Z` / `Ctrl+Shift+Z` |
| Toggle preview | `Ctrl+M` |
| Quit | close the window |

---

## Supported agents

pstore knows how to drive these coding agents out of the box. For the ones whose model it cannot choose, it reads their config to find out what they will run — and if it cannot, it says so and leaves them out of the ranking rather than guessing.

| Agent | Model | Effort |
| --- | --- | --- |
| **Claude Code** | Haiku 4.5 / Sonnet 5 / Opus 5 / Fable 5 | `--effort`, all five levels |
| **OpenAI Codex** | GPT-5.1 Codex | `-c model_reasoning_effort` |
| **Gemini CLI** | Gemini 3 Flash / Pro | from `settings.json` |
| **Cursor Agent** | read from `.cursor/cli-config.json` | its own |
| **Crush** | read from `.config/crush/crush.json` | its own |
| **Aider** | read from `.aider.conf.yml`, project first | its own |
| **Goose** | read from `.config/goose/config.yaml` | its own |
| **Qwen Code** | read from `.qwen/settings.json` | its own |
| **Factory Droid** | read from `.factory/config.json` | its own |
| **GitHub Copilot CLI** | not discoverable — reported, not ranked | its own |
| **Amp** | not selectable — reported, not ranked | its own |

---

## Configuration

Three layers, each overriding the one before it:

| Layer | Path | For |
| --- | --- | --- |
| System | `/etc/pstore/config.json` (`%PROGRAMDATA%\pstore\config.json`) | Machine-wide policy |
| User | `~/.config/pstore/config.json` | Your preferences everywhere |
| Local | `.pstore/config.json` beside the prompts | This project |

A field a layer does not mention is left alone. Only the **local** layer is ever written by the app. `pstore new [--local|--user|--system]` writes a starter file; it refuses to overwrite an existing one.

```jsonc
{
  "hint_score_tolerance": 8.0,     // points of fit a hint may trade for speed
  "pinned_agent": null,            // override the ranker's pick
  "allow_model_download": true,    // may pstore reach the network at all
  "allow_model_lookup": true,      // may it look up a model it cannot describe (name only)
  "local_model": "1-bit",          // or "ternary" — see the Models window for the trade-off
  "filter": {
    "block": ["*fable*"],          // patterns that disqualify a model
    "allow": [],                   // if non-empty, ONLY these are permitted
    "efforts": [],                 // if non-empty, ONLY these effort levels
    "block_metered": true          // refuse models billed per token
  }
}
```

`allow_model_download: false` means pstore never reaches the network — for the runtime, the weights and model lookups alike — and the features that need the model are disabled rather than degraded. `filter` decides which models are offered for ranking at all, before the model is asked.

---

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

---

## Links

- **Repository:** [github.com/alexpacio/pstore](https://github.com/alexpacio/pstore)
- **Issues:** [github.com/alexpacio/pstore/issues](https://github.com/alexpacio/pstore/issues)
