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
| **Inline hints** | Select text → get LLM suggestions (clarify, expand, constrain, fix). Inserted at cursor. |
| **Smart routing** | 6-dimension capability classifier plus a difficulty classifier score your prompt. Picks the cheapest adequate model automatically. |
| **PII sanitizer** | Finds names, addresses, IBANs, tax codes and card numbers and swaps them for placeholders — before the prompt reaches an agent. |
| **Local models only** | Every model pstore runs itself is an in-process encoder on [Candle](https://github.com/huggingface/candle). Download them from the Models window; after that, nothing about your prompt leaves the machine. |
| **One-key handoff** | `Ctrl+Enter` launches the selected agent with your prompt. Supports 12+ agents. |
| **The whole field, not a verdict** | Every (agent, model, effort) combination is scored and shown with its speed and rationale, so you can disagree with the pick. Token price is displayed but deliberately kept out of the score. |
| **Metered models stay out of the way** | Claude Fable 5 is the one model billed per token rather than covered by the subscription. It is scored like everything else, but only picked automatically when it fits better than every included model by a clear margin — never by a tie or a rounding accident. |
| **Plain files** | Prompts are `.md` in your project. Sidecar in `.pstore/`. Works with git, grep, any editor. |

---

## The local models

pstore uses three checkpoints, and runs all of them **in its own process, on your machine**.
None of them is contacted over the network at inference time; the only traffic is the
one-off weight download you start yourself from **Models…** in the action bar, which shows
each file's size and its progress as it arrives.

| Checkpoint | What it does | Size |
| --- | --- | --- |
| [`regolo/brick-modernbert-capability-classifier`](https://huggingface.co/regolo/brick-modernbert-capability-classifier) | Multi-label capability vector over the six routing dimensions (ModernBERT, sigmoid head) | 795 MB |
| [`nvidia/prompt-task-and-complexity-classifier`](https://huggingface.co/nvidia/prompt-task-and-complexity-classifier) | Task type plus six complexity dimensions, reweighted for coding prompts into easy/medium/hard (DeBERTa-v3) | 744 MB |
| [`rizzoaiacademy/rizzo-pii-0.3B`](https://huggingface.co/rizzoaiacademy/rizzo-pii-0.3B) | BIO token classification over 22 kinds of personal data, from [rizzo-pii](https://github.com/Rizzo-AI-Academy/rizzo-pii) (mmBERT) | 1.26 GB |

Weights go to the shared Hugging Face cache (`~/.cache/huggingface`), so other tools reuse
them. Until a checkpoint is downloaded, the feature that needs it degrades **visibly**: the
router falls back to its built-in surface-feature estimate and says so in the status bar,
and the sanitizer falls back to checksum-validated patterns (IBAN mod-97, Luhn, codice
fiscale, partita IVA, email) and says so in the review window.

> **Why not Brick's difficulty model?** Every published Brick complexity checkpoint is a LoRA
> on Qwen3.5-0.8B — a hybrid attention/SSM architecture candle-transformers has no
> implementation for — so it could never load, and difficulty quietly fell back to the
> heuristic. The NVIDIA classifier is the difficulty half instead: a plain encoder, one
> forward pass, a third of the size.
>
> Its published complexity score is weighted for general-purpose prompts, where two of the
> three largest terms (`creativity`, `domain_knowledge`) turn out to be near-constant across
> coding prompts — which squashes everything from "fix this typo" to a forty-query database
> migration into 0.13–0.38. pstore therefore reweights the model's own dimensions towards the
> ones that discriminate here (`constraints`, then `reasoning`), which spreads the same
> prompts across 0.05–0.51. The weights, the thresholds and the eleven-prompt calibration set
> they came from are all in `src/router/difficulty.rs`.

The **hint** and **shrink** features are different in kind — they deliberately drive
*your* installed coding agent (Claude Code, Codex, Gemini CLI, …), which is the point of
the tool. Those are the only paths where prompt text leaves the process, they are always an
explicit action, and the sanitizer exists to run before them.

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

---

## Quick Start

```bash
# In your project directory
pstore

# Or specify a prompt directory
pstore ~/my-prompts
```

**Key bindings:**
- `Ctrl+S` — Save (creates version snapshot)
- `Ctrl+R` — Score every installed model and effort level against this prompt
- `Ctrl+Enter` — Request a hint about the selection or the typed question
- `Ctrl+Z` / `Ctrl+Shift+Z` — Undo / Redo
- `Ctrl+M` — Toggle preview mode

**Action bar:** `Score models` · `Shrink` · `Sanitize` · `Send →` · `Hint…` · `Models…`

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
│   ├── config.rs            # Runtime config & persisted preferences
│   ├── models.rs            # Local checkpoint catalogue + download/load status board
│   ├── pii/
│   │   ├── mod.rs           # Findings, placeholder plan, BIO decoding
│   │   ├── detect.rs        # Checksum-validated identifiers (IBAN, Luhn, CF, P.IVA)
│   │   └── model.rs         # rizzo-pii token classifier on Candle
│   ├── router/
│   │   ├── capability.rs    # Brick ModernBERT capability classifier
│   │   ├── difficulty.rs    # NVIDIA DeBERTa task/complexity classifier
│   │   ├── heuristic.rs     # Surface-feature fallback for both signals
│   │   ├── hub.rs           # Hugging Face cache probe / download with progress
│   │   └── scoring.rs       # (agent, model, effort) ranking
│   ├── store/
│   │   ├── mod.rs           # PromptStore: list/create/read/write/rename/delete
│   │   └── version.rs       # Version history (snapshots, diffs, index)
│   ├── editor/
│   │   ├── mod.rs           # Buffer: text, selection, dirty tracking
│   │   └── undo.rs          # Snapshot-based undo/redo with coalescing
│   └── agents/
│       └── registry.rs      # Static agent/model specs & capability vectors
├── Cargo.toml
└── README.md
```

**Key design decisions:**
- **Plain files** — Prompts are `.md`. Sidecar state in `.pstore/`. No database, no lock-in.
- **Snapshot-based undo** — Coalesces typing into word/line granules. Programmatic edits are single atomic steps.
- **Static agent registry** — All agent CLIs and model specs in one file. One-line updates for flag changes.
- **Capability routing** — 6-dimension skill vector per model. Quality/cost tradeoff via single `r` parameter.
- **Local inference only** — Every model pstore runs itself is loaded in-process via Candle. Weights are downloaded on request, never at startup, and the app degrades visibly rather than silently when one is missing.
- **Nothing applied unreviewed** — Shrink and sanitize both propose a diff. Accepting is one undo step, and the previous text is already in version history.
- **egui/eframe** — Native GUI, no Electron. Small binary, fast startup, cross-platform.

---

## Configuration

Settings stored in `.pstore/config.json` in your prompt directory:

```json
{
  "hint_score_tolerance": 8.0, // points of fit a hint may trade for speed
  "preview": false,            // start in preview mode
  "sidebar_width": 260,        // sidebar width in points
  "pinned_agent": null,        // override auto-selected agent
  "allow_model_download": true // may pstore fetch weights from Hugging Face at all
}
```

Setting `allow_model_download` to `false` (or unticking it in the Models window) means
pstore never reaches the network: routing uses the built-in estimate and sanitizing uses the
checksum-backed patterns.

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