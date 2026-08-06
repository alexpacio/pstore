# pstore

**A native GUI that makes prompt authoring a first-class activity** — versioned markdown files, inline LLM hints, automatic difficulty/capability scoring to pick the cheapest adequate model, and one-keystroke handoff to a real agent session.

[![Website](https://img.shields.io/badge/Website-pstore.dev-00d4aa?style=flat-square)](https://alexpacio.github.io/pstore)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.78%2B-orange?style=flat-square&logo=rust)](https://rust-lang.org)

---

## Why pstore?

Writing a good prompt for a coding agent is iterative. Today that work happens in a scratch buffer with:
- **No history** — every edit is destructive
- **No help while writing** — you're on your own
- **No idea which agent/model to use** — overpaying for Opus when Haiku would do, or underpowered for the task

pstore treats prompt authoring as a first-class workflow:

| Feature | Description |
|---------|-------------|
| **Versioned prompts** | Every save creates a snapshot in `.pstore/versions/`. Full history, diffs, one-click restore — all plain markdown. |
| **Inline hints** | Select text, type a question, or both — the selection and the question reach the agent as distinct things. Answers land in a panel, never silently in the document. |
| **Plan** | Rewrites a rough request into a structured instruction for a coding agent: objective, ordered steps, constraints, acceptance criteria. The output *is* the next prompt, not a document to read. |
| **Smart routing** | Every (agent, model, effort) combination your machine can run is handed to a local 27B model along with your prompt; it returns a ranked shortlist with a reason for each pick. |
| **PII sanitizer** | Finds names, addresses, IBANs, tax codes and card numbers and swaps them for placeholders — before the prompt reaches an agent. |
| **One local model** | A single 3.8 GB checkpoint runs everything pstore infers, as a subprocess on your machine. Nothing about your prompt leaves it. |
| **Model policy** | Block models by pattern, or whitelist the only ones you are allowed to use. Per-token models are blocked by default so nothing bills you by accident. Configurable machine-wide, per-user, or per-project. |
| **One-key handoff** | `Ctrl+Enter` launches the selected agent with your prompt. Supports 12+ agents. |
| **Copy anything** | Every produced artefact — hint, shrink, plan, masked prompt — has a copy button. The workflow ends in a paste. |
| **Plain files** | Prompts are `.md` in your project. Sidecar in `.pstore/`. Works with git, grep, any editor. |

---

## The local model

pstore uses **one** checkpoint for everything it infers itself — which agent and model
should answer a prompt, and where the personal data in it is:

| Checkpoint | What it does | Download |
| --- | --- | --- |
| [`prism-ml/Bonsai-27B-gguf`](https://huggingface.co/prism-ml/Bonsai-27B-gguf) (`Q1_0`) | Ranks the installed (agent, model, effort) combinations against your prompt, and finds personal data in it | 3.8 GB |

It is PrismML's 1-bit build of Qwen3.6-27B: a 27B model in 3.8 GB of weights, retaining
~89.5% of FP16 quality. The ternary build of the same model scores higher (94.6%) but costs
7.17 GB — twice the memory for tasks that sit far below what a 27B can do.

Weights go to the shared Hugging Face cache (`~/.cache/huggingface`), so other tools reuse
them.

### How it runs

pstore does not link an inference engine. It runs the model the way it runs a coding agent:
as a one-shot `llama-cli` subprocess, prompt in, JSON out, process exits. No server, no
port, no HTTP client, nothing left running when pstore closes.

That binary has to be [PrismML's fork of llama.cpp](https://github.com/PrismML-Eng/llama.cpp)
— the `Q1_0_g128` quantisation has kernels that exist nowhere else, and stock llama.cpp
fails to load the file. **pstore downloads it for you** (~11 MB, checked against its
published SHA256 before it is installed) into `~/Library/Application Support/pstore/bin`,
`~/.local/share/pstore/bin`, or `%LOCALAPPDATA%\pstore\bin`. A machine-wide install is used
if an administrator has provisioned one; pstore never asks for privileges to create it.

Output is **grammar-constrained**: the JSON schema is compiled into the sampler with
`--json-schema`, so the model cannot emit anything that fails to parse.

The cost of one process per call is that the weights are mapped every time — expect seconds,
not milliseconds. That is affordable because every one of these is a button press you were
already waiting on, and it buys the whole simplification above.

### Memory footprint

The context window is **fitted to each call** rather than pinned:

```
ctx = round_up(prompt_tokens + max_output_tokens + margin, 256), clamped to the ceiling
```

A routing call runs at ~512–1024 tokens of context instead of the checkpoint's native
262,144. At those sizes the KV cache is tens of megabytes, so the weights are essentially
the entire footprint: **expect ~5 GB peak** (3.8 GB weights + ~1.3 GB runtime), against the
model card's 5.9 GB at 4K context and 12.2 GB at 100K.

| Technique | How | Effect |
| --- | --- | --- |
| 1-bit checkpoint | `Bonsai-27B-Q1_0.gguf` | **The dominant lever** — 3.8 GB vs 7.17 GB for the ternary build |
| No vision tower | `mmproj` never fetched | 0.63–0.93 GB never downloaded or loaded; pstore sends no images |
| No drafter | `dspark` never fetched | Speculative decoding buys throughput with memory — the wrong trade for one-shot calls |
| Fitted context | `--ctx-size <computed>` | KV cache is linear in context; most calls need a fraction of the ceiling |
| 4-bit KV cache | `--cache-type-k q4_0 --cache-type-v q4_0` | ~4× smaller cache than f16 |
| Flash attention | `--flash-attn on` | Removes the attention scratch buffer |
| No warmup or mlock | `--no-warmup` | Nothing is pre-touched or pinned on a short-lived process |

`model_context_ceiling` (default 8192) **caps** the fitted value rather than setting it.
Lower it to bound memory on a small machine; raising it costs KV cache roughly linearly.
On Apple Silicon this is unified memory, so it counts against overall RAM pressure rather
than a separate VRAM pool.

> **Why one generative model instead of three classifiers?**
> pstore used to run three encoders on [Candle](https://github.com/huggingface/candle): a
> capability classifier, a difficulty classifier, and a PII tagger, behind a hand-built
> scorer with per-dimension skill vectors for every model. Those vectors were maintained by
> hand and went stale every time a vendor shipped anything, and the whole apparatus existed
> to approximate a judgement — "is this model right for this prompt?" — that a capable model
> can simply be asked.
>
> Candle could not have run this one in any case: Qwen3.6 uses hybrid Gated-DeltaNet +
> Gated-Attention, and `candle-transformers` has no implementation for it — the same wall
> that made Brick's difficulty checkpoint unusable.

### Nothing degrades silently

There is **no fallback**. Earlier versions dropped to a surface-feature estimate when the
weights were missing, which meant every ranking carried an invisible question of which
implementation produced it. Now, if the model or its runtime is unavailable, **Score
models**, **Plan**, **Shrink** and **Sanitize** report why and do nothing. Editing,
versioning, diffing and agent handoff are unaffected.

`sanitize` in particular returns an error rather than an empty result: "no personal data
found" is a claim, and making it without having looked is how personal data reaches an
agent.

The **hint**, **shrink** and **plan** features drive *your* installed coding agent (Claude
Code, Codex, Gemini CLI, …), which is the point of the tool. Those are the only paths where
prompt text leaves the machine, they are always an explicit action, and the sanitizer exists
to run before them.

---

## Installation

```bash
# Via cargo (requires Rust 1.78+)
cargo install --git https://github.com/alexpacio/pstore

# Or download a pre-built binary from Releases
```

### From Source
```bash
git clone https://github.com/alexpacio/pstore
cd pstore
cargo build --release
./target/release/pstore
```

**No other prerequisites.** pstore downloads the model and the `llama-cli` that runs it on
first use, from the Models window. To build without local inference at all:

```bash
cargo build --release --no-default-features
```

That drops the download machinery entirely; ranking, planning, shrinking and sanitizing are
then unavailable, and say so.

---

## Quick Start

```bash
# In your project directory
pstore

# Or specify a prompt directory
pstore ~/my-prompts

# Write a starter config file (see Configuration below)
pstore new                # .pstore/config.json here — same as --local
pstore new --user         # ~/.config/pstore/config.json
sudo pstore new --system  # /etc/pstore/config.json

# See what pstore can find, without opening a window
pstore --list-agents
```

**Key bindings:**
- `Ctrl+S` — Save (creates version snapshot)
- `Ctrl+R` — Rank the installed models against this prompt
- `Ctrl+Enter` — Ask about the selection, the typed question, or both
- `Ctrl+Z` / `Ctrl+Shift+Z` — Undo / Redo
- `Ctrl+M` — Toggle preview mode

**Action bar:** `Score models` · `Shrink` · `Plan` · `Sanitize` · `Send →` · `Hint…` · `Models…`

---

## Supported Agents

pstore knows how to drive these coding agents out of the box:

- **Claude Code** — Haiku 4.5 / Sonnet 5 / Opus 5 / Fable 5
- **OpenAI Codex** — GPT-5.1 Codex
- **Gemini CLI** — Gemini 3 Flash / Pro
- **Cursor Agent** — Model from Cursor config
- **OpenCode** — Model from OpenCode config
- **Crush** — Model from `~/.config/crush/crush.json`
- **Aider** — Model flag support
- **Goose** — Session-based interaction
- **Qwen Code** — Model flag support
- **GitHub Copilot CLI** — Model flag support
- **Factory Droid** — Model flag support
- **Amp** — Stdin-based prompt delivery

---

## Architecture

```
pstore/
├── src/
│   ├── main.rs              # Application entry point
│   ├── config.rs            # Layered config (system → user → local) & preferences
│   ├── filter.rs            # Which models policy permits: glob patterns, allow/block
│   ├── models.rs            # Checkpoint catalogue + download status board
│   ├── runtime.rs           # Finding/fetching/verifying the llama-cli that runs it
│   ├── plan.rs              # Planning instruction + structural checks on the result
│   ├── shrink.rs            # Compression instruction + integrity checks
│   ├── hints.rs             # Hint subjects (selection, question, or both) + composition
│   ├── pii/
│   │   └── mod.rs           # Findings, placeholder plan, overlap resolution
│   ├── router/
│   │   ├── mod.rs           # Candidate enumeration, Ranking/Choice
│   │   ├── llm.rs           # The one place that runs the model: ranking, PII, ctx fitting
│   │   └── hub.rs           # Hugging Face cache probe / download with progress
│   ├── store/
│   │   ├── mod.rs           # PromptStore: list/create/read/write/rename/delete
│   │   └── version.rs       # Version history (snapshots, diffs, index)
│   ├── editor/
│   │   ├── mod.rs           # Buffer: text, selection, dirty tracking
│   │   └── undo.rs          # Snapshot-based undo/redo with coalescing
│   └── agents/
│       └── registry.rs      # Static agent/model specs
├── Cargo.toml
└── README.md
```

**Key design decisions:**
- **Plain files** — Prompts are `.md`. Sidecar state in `.pstore/`. No database, no lock-in.
- **Snapshot-based undo** — Coalesces typing into word/line granules. Programmatic edits are single atomic steps.
- **Static agent registry** — All agent CLIs and model specs in one file. One-line updates for flag changes.
- **The model does the judging** — Ranking is a prompt, not a formula. No hand-maintained skill vectors to go stale.
- **Subprocess inference** — The model runs as a `llama-cli` child process, the same way agents do. Nothing is linked in, nothing stays resident.
- **No silent degradation** — Model-dependent features are disabled with a reason when the model is unavailable, never quietly replaced by something worse.
- **Nothing applied unreviewed** — Shrink, plan and sanitize all propose a diff. Accepting is one undo step, and the previous text is already in version history.
- **egui/eframe** — Native GUI, no Electron. Small binary, fast startup, cross-platform.

---

## Configuration

Three layers, each overriding the one before it:

| Layer | Path | For |
| --- | --- | --- |
| System | `/etc/pstore/config.json` (`%PROGRAMDATA%\pstore\config.json`) | Machine-wide policy |
| User | `~/.config/pstore/config.json` | Your preferences everywhere |
| Local | `.pstore/config.json` beside the prompts | This project |

A field a layer does not mention is left alone, so a user layer can change one setting
without reimposing defaults over an administrator's policy. Only the **local** layer is ever
written by the app. `pstore new [--local|--user|--system]` writes a starter file; it refuses
to overwrite an existing one.

```json
{
  "hint_score_tolerance": 8.0,   // points of fit a hint may trade for speed
  "preview": false,              // start in preview mode
  "sidebar_width": 260,          // sidebar width in points
  "pinned_agent": null,          // override the ranker's pick
  "allow_model_download": true,  // may pstore fetch the model and its runtime at all
  "llama_cli_path": null,        // use your own PrismML llama.cpp build
  "model_context_ceiling": 8192, // hard cap on the per-call context window
  "filter": {
    "block": ["*fable*"],        // patterns that disqualify a model
    "allow": [],                 // if non-empty, ONLY these are permitted
    "efforts": [],               // if non-empty, ONLY these effort levels
    "block_metered": true        // refuse models billed per token
  }
}
```

`allow_model_download: false` means pstore never reaches the network — for the runtime as
well as the weights — and the features that need the model are disabled rather than
degraded.

### Model policy

`filter` decides which models are offered for ranking at all. Filtering happens *before* the
model is asked, so a model you have ruled out is never a candidate.

Patterns are case-insensitive globs (`*` and `?`) matched against a model's id, its display
name, and `agent/id`. **A pattern without wildcards must match the whole name** — `sonnet`
does not match `sonnet-thinking`; write `sonnet*` if that is what you meant.

```jsonc
// Blocking: everything is fine except these
{ "filter": { "block": ["*opus*", "crush/*"] } }

// Allowing: nothing is permitted except these — usually easier under a procurement policy
{ "filter": { "allow": ["*sonnet*", "gpt-5*"], "efforts": ["low", "medium"] } }
```

`allow` and `block` compose, and it fails closed: a model must match `allow` *and* not match
`block`.

`block_metered` is on by default. Every other model in the registry is covered by a
subscription you have already paid for; a metered one (Claude Fable 5) spends money you have
not agreed to spend, and the failure is silent — it shows up on an invoice, not on screen.
Reaching one takes two deliberate changes: clearing `block_metered` *and* removing the
default `*fable*` pattern.

---

## Development

```bash
# Run in dev mode
cargo run

# Run tests
cargo test

# Check formatting
cargo fmt --check

# Lint
cargo clippy -- -D warnings
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Links

- **Website:** https://alexpacio.github.io/pstore
- **Repository:** https://github.com/alexpacio/pstore
- **Issues:** https://github.com/alexpacio/pstore/issues