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
| **Shrink** | Rewrites the selection — or the whole prompt — telegraphically: no articles, no pleasantries, each fact stated once, while code, paths, identifiers and constraints stay verbatim. The local model decides what a word is carrying, so `the` in a code span and `the 20 ms poll must not change` survive where a stop-word stripper would eat both. Arrives as a diff with a size summary and a warning if a file reference or code block went missing. |
| **Smart routing** | Every (agent, model, effort) combination your machine can run is handed to a local 27B model along with your prompt; it returns a ranked shortlist with a reason for each pick. |
| **Only models it can describe** | A candidate pstore cannot name and characterise is **withheld from the ranking**, with the reason shown, rather than ranked blind. Half the agents choose their own model, and one unidentified row moves every real one below it. |
| **PII sanitizer** | Finds names, addresses, IBANs, tax codes and card numbers and swaps them for placeholders — before the prompt reaches an agent. |
| **One local model** | A single checkpoint — your pick of two sizes — runs everything pstore infers, as a subprocess on your machine. Nothing about your prompt leaves it. |
| **Model policy** | Block models by pattern, or whitelist the only ones you are allowed to use. Per-token models are blocked by default so nothing bills you by accident. Configurable machine-wide, per-user, or per-project. |
| **One-key handoff** | `Ctrl+Enter` launches the selected agent with your prompt. Supports 11 agents. |
| **Three front ends** | The same core behind a native window, a full-screen terminal UI, and a scriptable CLI with `--json` on everything. `pstore rank` and the **Score models** button are one code path. |
| **Copy anything** | Every produced artefact — hint, shrink, plan, masked prompt — has a copy button. The workflow ends in a paste. |
| **Plain files** | Prompts are `.md` in your project. Sidecar in `.pstore/`. Works with git, grep, any editor. |

---

## The local model

pstore runs **one** checkpoint for everything it infers itself — which agent and model
should answer a prompt, where the personal data in it is, and how to say the same thing in
fewer words. You choose which of two builds of it to run — in the Models window, in
`.pstore/config.json`, or from `pstore models`:

| Build | Quality | Peak memory | Download |
| --- | --- | --- | --- |
| [`Bonsai-27B`](https://huggingface.co/prism-ml/Bonsai-27B-gguf) (`Q1_0`) — **default** | 89.5% of FP16 | ~5 GB | 3.8 GB |
| [`Ternary-Bonsai-27B`](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf) (`Q2_0`) | 94.6% of FP16 | ~8.4 GB | 7.17 GB |

Same 27B model, same 262K context, same template — PrismML's ternary `{-1, 0, +1}` build at a
true 1.71 bits, or their binary `{-1, +1}` build at 1.125. What changes is how much of the
full-precision model survives, and what it costs you in memory and in seconds. Both can sit on
disk at once; only the selected one is ever run.

**The small build is the default, and it routes as well as the large one.** That took work, and
it was not always true. Routing is an instruction-following and judgement task, and those are the
two categories the binary build gives up most on — IFBench 52.4 and τ²-Bench 61.3, against FP16's
68.0 and 82.9. Asked to rank fifteen (model, effort) pairs for a hard three-file refactor it used
to answer `Haiku 4.5, effort medium`, then indices 2, 3, 4 and 5 in order, scoring all five
`fit: 85` with the same reason copy-pasted onto each. It was not ranking; it was counting.

It was the *question*, not the weights. Asked one judgement at a time, against a list of models
rather than a grid of (model, effort) pairs, with the scores banded by the sampler and the options
presented best-first for the difficulty already decided — see
[How routing asks the question](#how-routing-asks-the-question) — the same 3.8 GB build now
answers, over a twenty-one-combination field:

| Prompt | 1-bit (default) | ternary |
| --- | --- | --- |
| a three-file refactor with an invariant to preserve | Opus 5 · high · 95, then GPT-5.1 Codex · high · 80 | Opus 5 · xhigh · 98, then GPT-5.1 Codex · xhigh · 80 |
| fix a typo in the README | Haiku 4.5 · low · 95 | Haiku 4.5 · low · 95 |
| time for that ranking | **26 s** | 41 s |

**What the ternary build still buys**, at double the memory and roughly double the wait: a longer
shortlist it can separate honestly — five picks against the small build's three — and sharper
reasons for each. Nothing about which model ends up at the top. Choose it if the machine has the
memory to spare and you read the whole list; otherwise the default is the better trade.

Set it in the Models window, or by hand:

```json
{ "local_model": "ternary" }   // or "1-bit"
```

Switching takes effect on the next model call, not the next launch — nothing is resident
between calls.

**The two are never resident at the same time.** Since every call is its own subprocess,
switching is normally free; the exception is switching *while* a call is running, where the
old process would go on holding its 3.8 or 7.17 GB and the next call would map the other build
alongside it — both at once, on a machine that may have been given the small build precisely
because memory is tight. So the run holding the build you left is stopped rather than waited
out (its answer would come from the build you just rejected anyway), and a worker that is
mid-flight when you switch is refused before it maps anything. The status bar says how many
runs were stopped, and the affected action reports that it was interrupted rather than failing.

Weights go to the shared Hugging Face cache (`~/.cache/huggingface`), so other tools reuse
them.

### How it runs

pstore does not link an inference engine. It runs the model the way it runs a coding agent: as a
child process.

**One load per operation, and nothing between them.** A ranking is two model calls — how hard is
this prompt, then which model suits it. A shrink is one call per chunk; a personal-data scan is one
per chunk too. Those all belong to *one thing the user asked for*, so they share one load of the
weights and the process dies with the operation. Between operations there is no process, no port
and no memory held.

That is the privacy property and it is also what keeps the context window honest: the window is
computed from the prompts this operation is actually going to send, taking the largest, rather than
being sized for whatever might come later. A ranking runs at ~1 000–2 000 tokens of context instead
of the checkpoint's native 262 144.

| | before | now |
| --- | --- | --- |
| a ranking (2 calls) | 2 loads | 1 |
| a 3-chunk personal-data scan | 3 loads | 1 |
| a 5-chunk shrink | 5 loads | 1 |

**It is not a service.** The process binds `127.0.0.1` on a port the kernel picks, behind a random
key generated for that session and never written anywhere, with the web UI off, for the seconds the
operation takes. Nothing can find it, nothing is meant to, and there is no setting to make it stay.
That is the difference between this and the background server pstore used to offer, which was a
feature and is gone. (Loopback rather than a socket file only because this fork accepts `--host
x.sock` and then fails to bind it.)

Closing a window does not kill its children, so every session is registered while it runs and
killed on the way out. Quit mid-generation and the 3.8 GB goes with it — the call you were waiting
on reports that pstore is closing instead of finishing into a window that no longer exists.

The runtime has to be [PrismML's fork of llama.cpp](https://github.com/PrismML-Eng/llama.cpp) — the
`Q1_0`/`Q2_0_g128` quantisations have kernels that exist nowhere else, and stock llama.cpp fails to
load the file. **pstore downloads it for you** (~11 MB, checked against its published SHA256 before
it is installed) into `~/Library/Application Support/pstore/bin`, `~/.local/share/pstore/bin`, or
`%LOCALAPPDATA%\pstore\bin`. A machine-wide install is used if an administrator has provisioned
one; pstore never asks for privileges to create it.

The prompt is rendered with the **model's own chat template**, which opens the `<think>` block a
thinking model expects. Without it llama.cpp falls back to a legacy ChatML template and the
checkpoint is prompted in a shape it was never trained on: that one detail is the difference
between `Haiku 4.5, effort medium` followed by the next four options in order, and Opus 5 at high
effort with scores that descend.

**Every operation is tuned to what it is**, because two kinds of work happen here and they want
opposite settings:

| | Judgement — ranking | Extraction — difficulty, personal data, shrink, model recall |
| --- | --- | --- |
| sampling | `temp 0.7 · top-p 0.95 · top-k 20` | greedy |
| reasoning | a bounded `<think>` block | none |
| why | the checkpoint's published thinking-mode settings; reasoning at temperature zero collapses into repetition — the same `reason` on all five picks | one right answer and nothing to deliberate. Running the personal-data scan at 0.7 cost it a live finding: asked for the personal data in `Contact Mario Rossi at mario@example.com`, it returned the name and dropped the address |

Output is **grammar-constrained**, so the model cannot emit anything that fails to parse and
parsing is a `serde_json` call rather than a best-effort scrape. The ranking grammar allows a
bounded **reasoning block** before the JSON and then *requires* `</think>`: given room to think the
model reasons well, but it does not stop on its own — unbounded, one routing call spent 1,399
tokens re-litigating its own conclusion and never answered. `model_reasoning_budget` caps it
(default 1,400 characters; `0` disables it). One trap, learned the hard way: never give a JSON
grammar an unbounded whitespace rule. `ws ::= [ \t\n]*` between two tokens is a legal place to
emit spaces forever, and the model did exactly that.

**Measured cost**, warm, on an M4-Pro-class laptop:

| Phase | 1-bit | ternary |
| --- | --- | --- |
| mapping the weights, once per operation | ~1.5 s | ~2.5 s |
| prompt evaluation | ~95 tok/s | ~50 tok/s |
| generation | ~25 tok/s | ~13 tok/s |
| **a ranking over 21 combinations** | **~26 s** | ~41 s |

Generation costs ~4× more per token than prompt evaluation, so the reply grammar permits no
whitespace at all and every string is capped; prompt evaluation is linear with no fixed floor, so
the candidate list stays terse.

**Speculative decoding is not the answer here.** The repository ships a DSpark drafter, and it is
lossless — verification preserves the target distribution exactly, so it can only cost speed, never
quality. But its measured 1.34× is on the CUDA serving path, PrismML do not enable it on Apple
Silicon at all (a batch-1 verification pass does not amortise there), and it accelerates only
generation — under half of a ranking call.

### How routing asks the question

Four things about the shape of the ranking call, each of which came from watching it fail:

**The difficulty is decided in its own call, first.** Ranking used to ask for two judgements at
once: how hard is this work, and which of twenty options best matches. The 1-bit build can do the
first and demonstrably cannot do both — asked to shortlist a three-file refactor it returned
Haiku first, *"weak on multi-file reasoning, prompt requires multi-file"*, and Opus last,
*"best for hard refactors, prompt is hard refactor"*. Its analysis was right in both rows and its
**ordering ignored it**. So `easy | moderate | hard` is settled in a short greedy call — the
document, no candidate list, a one-word reply — and reaches the ranking prompt as a stated fact
with an explicit instruction attached (*"rank the most capable model first; a light model is the
wrong answer here even though it is cheaper"*). It is the one place pstore knowingly spends two
calls on one user action, and it is what makes the small build route correctly. The verdict is
shown, because it is the premise of everything under it: a shortlist that looks wrong is usually a
difficulty read that was wrong.

**The model ranks models, not (model, effort) pairs.** The candidate grid holds one row per
effort, so five efforts of Opus are five nearly-identical lines, and separating those is where the
checkpoint stops discriminating and starts enumerating. Ranking *models* and asking for the effort
as a field poses the same question with a fifth of the rows and no near-duplicates in them.

**Each rank has its own score band, enforced by the sampler.** `fit` used to be "0-100, and the
five must differ", which produced `85, 85, 85, 85, 85` on the small build and a correctly-ordered
shortlist scored `0` then `1` on the large one. Now the first pick must score 85-99, the second
70-84, and so on down: identical scores are **unrepresentable** rather than discouraged, and
within its band the model still has fifteen values to say how much better one pick is than the
next.

**The options are listed best-first for the difficulty already decided.** A small model has
positional bias, and this is what turns that from a hazard into a tailwind. Given a list running
light → frontier, the 1-bit build returned options 0, 1 and 2 in order for a hard refactor — Haiku
first — with reasons that contradicted its own ranking ("may struggle with complex threading
logic"). It had the difficulty right and was reading the list rather than judging it. Sorting the
rows by how well their tier matches the difficulty means that failure mode now produces a
defensible answer instead of an inverted one, and a model that really is ranking is unaffected: it
picks by index either way, and the indices are the same options.

**A list is told apart from a ranking, and said so.** Two signatures, both from live runs: the
same reason copy-pasted onto every pick, and identical scores across the shortlist. Either one is
retried once with the mistake named, and if it survives that the shortlist is shown with a warning
rather than presented as an answer — because a degenerate reply is populated in every field and
wrong in the only one that matters, which no user can see by looking at it. (A third signature —
picks at consecutive indices — was removed rather than tightened: now that the list is ordered,
"the first three in order" is what a *correct* answer often looks like.)

The small build is also asked for **three** picks rather than five. It answers a shorter question
better, and five was never a number it could fill honestly.

**Measured cost**, warm, on an M4-Pro-class laptop against a fifteen-candidate prompt:

| Phase | Cost |
| --- | --- |
| process start, mmap, `-fit` probe | ~1.1 s |
| prompt evaluation | ~10 ms/token (~100 tok/s) |
| generation | ~41 ms/token (~24 tok/s) |
| a difficulty read (its own call, see below) | ~3-6 s |
| **a ranking call** | **~26-40 s**, difficulty read included |

Those are the checkpoint's own rates — PrismML publish 26 tok/s generation and 133 tok/s
prompt evaluation for this class of machine — and the ternary weights move about twice the
bytes per token, so budget accordingly. Earlier versions of this file claimed ~1.4 s per call
and ~27 tok/s of *prompt* evaluation; the first was out by an order of magnitude and the
second had the two phases the wrong way round.

Generation costs 4× more per token than prompt evaluation, so the reply grammar permits no
whitespace at all; prompt evaluation is linear with no fixed floor, so the candidate list is
terse. A **resident server** would buy back the startup and, more usefully, keep the
unchanging head of the prompt in the KV cache instead of re-evaluating it every call — that
is the one real argument against one-process-per-call, and it is seconds, not milliseconds.
It is deliberately not taken: keeping a 27B resident means holding 3.8-8.4 GB between calls, and
a port or a socket for something to talk to. One process per call means the memory is held only
while an answer is being produced, and nothing is listening for anything, ever.

**Speculative decoding is not the answer here.** The repository ships a DSpark drafter, and
it is lossless — verification preserves the target distribution exactly, so it can only cost
speed, never quality. But its measured 1.34× is on the CUDA serving path, PrismML do not
enable it on Apple Silicon at all (a batch-1 verification pass does not amortise there), and
it accelerates only generation — under half of a ranking call.

### Only models pstore can describe are ranked

Ranking is a judgement about models, and a judgement needs facts. Half the agents in the
registry do not let pstore choose a model — Crush, Goose, Aider, Cursor and the rest take it
from their own config — and those used to be offered to the ranker as a row reading
`(agent default) via Crush [mid, effort high]`: a candidate with no name, no vendor and no
capability, sitting in the same list as Opus 5 and Haiku 4.5.

A model asked for five ranked choices cannot answer "I don't know what that is". It invents a
placement, and **one invented row displaces every real one below it**. That is the failure this
layer exists to prevent, and it is why a shortlist of four models pstore understands is worth
more than a shortlist of five where one is fiction.

So before anything is ranked, every distinct model in the field has to be identified — three
answers, tried in this order:

| Source | What it is | Cost |
| --- | --- | --- |
| **pstore's table** | One line per model, in `src/knowledge.rs` beside the registry: what it is, what it is good at, where it falls short. | free, offline |
| **the checkpoint itself** | For a name pstore does not describe, the model is asked — in its own call — whether it actually knows it, and told to say nothing rather than guess. | one short call |
| **a web lookup** | A name neither knows is searched for, and the result goes into the ranking prompt as facts. Cached on disk, so a name is looked up once. | one HTTP GET |

The table comes **first**, ahead of the checkpoint's own memory, and that ordering is
deliberate: the checkpoint's training ended before every model in it, so on `Opus 5` or
`Gemini 3` its recollection is not knowledge but a plausible-sounding guess about a name whose
shape it recognises. A stated fact beats a remembered one. It also means the common path — a
stock install, where the table covers the whole field — spends **no** extra call and touches no
network.

A model that survives none of the three is excluded, and the reason appears next to the
shortlist:

```
excluded · crush: this agent picks its own model and its config does not say which —
                 pstore will not rank a model it cannot name
```

Where pstore *can* read it, it does: `.config/crush/crush.json`, `.config/goose/config.yaml`,
`.aider.conf.yml`, `.qwen/settings.json`, `.cursor/cli-config.json`, `.factory/config.json`.
That is discovery rather than a schema pstore claims to know — a key that has been renamed, a
file that has moved, a value that is not a name all end in "unknown", which is the honest
answer and not a guess. `pstore agents` shows what it found for each one.

**What leaves the machine.** Only a model's *name*, only for a name nothing local describes, and
only to a search engine — never the prompt, never the file, never the project. `allow_model_lookup:
false` turns it off, and `allow_model_download: false` (which already means "make no network
request") covers it too.

### Memory footprint

The context window is **fitted to each call** rather than pinned:

```
ctx = round_up(max over the operation's calls of
                 (prompt_tokens + max_output_tokens + margin), 256), clamped to the ceiling
```

A routing operation runs at ~1 000–2 000 tokens of context instead of the checkpoint's native
262,144. At those sizes the KV cache is tens of megabytes, so the weights are essentially the
entire footprint: **expect ~5 GB peak** on the default build (3.8 GB weights + ~1.2 GB runtime),
or ~8.4 GB on the ternary one.

| Technique | How | Effect |
| --- | --- | --- |
| Choice of checkpoint | `local_model` | 3.8 GB and ~5 GB peak by default; `"ternary"` raises it to 7.17 GB and ~8.4 GB for a longer shortlist |
| No vision tower | `mmproj` never fetched | 0.63–0.93 GB never downloaded or loaded; pstore sends no images |
| No drafter | `dspark` never fetched | 1.95 GB for a generation speedup PrismML do not enable on Apple Silicon |
| Fitted context | `--ctx-size <computed>` | KV cache is linear in context; sized per operation from the prompts it will send |
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

The **hint** and **plan** features drive *your* installed coding agent (Claude Code, Codex,
Gemini CLI, …), which is the point of the tool. Together with **Send →** those are the only
paths where prompt text leaves the machine, they are always an explicit action, and the
sanitizer exists to run before them. **Shrink** is not among them: it compresses on the
local checkpoint, so a prompt can be tightened before anything has seen it.

The only other request pstore makes is a **model lookup** — a name, not your text — and only for a
model nothing local can describe. See
[Only models pstore can describe](#only-models-pstore-can-describe-are-ranked).

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

**No other prerequisites.** pstore downloads the model and the `llama-server` binary that runs it
on first use, from the Models window.

**One binary, three front ends**, each behind a feature so you can leave out what you will not
run. All are on by default:

```bash
# Headless machine — no windowing dependencies compiled at all. The largest saving there is.
cargo build --release --no-default-features --features local-llm,tui

# Window only
cargo build --release --no-default-features --features local-llm,gui

# No local inference: editing, versioning, diffing and agent handoff still work; ranking,
# planning, shrinking and sanitizing are disabled and say why.
cargo build --release --no-default-features --features gui,tui
```

---

## Quick Start

Three ways in, over the same core. `pstore rank` and the **Score models** button run the same
code and give the same answer; what differs is only how the result is presented.

```bash
# The window, in the current directory
pstore

# The terminal interface — for the machine that has the weights on it, over ssh
pstore tui

# Or specify a prompt directory, with either
pstore ~/my-prompts
pstore tui ~/my-prompts
```

### The command line

Every command that produces a result takes `--json`; nothing but the result goes to stdout, so
progress notes and warnings stay on stderr and a pipe stays clean.

```bash
pstore rank refactor.md              # ranked shortlist, with the difficulty verdict
pstore rank refactor.md --json | jq -r .best.model
pstore shrink refactor.md            # telegraphic rewrite to stdout
pstore shrink refactor.md --write    # ...or in place, snapshotting the previous version
pstore plan refactor.md              # structured instruction, via an installed agent
pstore plan refactor.md --agent claude   # skip the ranking call and use this one
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

**Exit codes mean something**, so a hook does not have to parse English: `0` succeeded, `1` the
operation ran and the answer is no (nothing to rank, a scan *found* personal data, a rewrite saved
nothing, the ranking came back degenerate), `2` pstore could not do it at all (no such file, no
checkpoint downloaded). A pre-commit hook is `pstore sanitize prompt.md --json` and a test on the
code.

### Key bindings

| | Window | Terminal |
| --- | --- | --- |
| Save (creates a version) | `Ctrl+S` | `Ctrl+S` |
| Rank the installed models | `Ctrl+R` | `Ctrl+R` or `F5` |
| Shrink · Plan · Sanitize | action bar | `F2` · `F3` · `F4` |
| Ask about the selection or a question | `Ctrl+Enter` | `Ctrl+H` |
| Undo / Redo | `Ctrl+Z` / `Ctrl+Shift+Z` | `Ctrl+Z` / `Ctrl+Y` |
| Accept / reject a proposal | buttons | `a` / `r` |
| New prompt | sidebar | `Ctrl+N` |
| Local model status | `Models…` | `F6` |
| Version history / cycle the side pane | panel | `F7` / `F9` |
| Move between the prompt list and the text | click | `Tab` |
| Toggle preview | `Ctrl+M` | — |
| Stop whatever is running | `Stop` | `Esc` |
| Help | — | `F1` |
| Quit | close the window | `Ctrl+Q` |

**Action bar:** `Score models` · `Shrink` · `Plan` · `Sanitize` · `Send →` · `Hint…` · `Models…`

Nothing is applied unreviewed in either: shrink, plan and sanitize all arrive as a diff, and
accepting is one undo step with the previous text already in version history.

---

## Supported Agents

pstore knows how to drive these coding agents out of the box. For the ones whose model it cannot
choose, it reads their config to find out what they will run — and if it cannot, it says so and
leaves them out of the ranking rather than guessing (see
[Only models pstore can describe](#only-models-pstore-can-describe-are-ranked)).

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

## Architecture

**The crate is a library with no user interface in it**, and three front ends over exactly the
same modules. `App` holds every piece of state a front end needs and every operation one can
perform, and it names no widget toolkit: it owns the buffer, the version store, the job runner and
the proposals awaiting review, and advances by being handed job events. A front end renders that
and calls those methods.

```
pstore/
├── src/
│   ├── lib.rs               # The shared core. Nothing below here draws anything.
│   ├── main.rs              # The binary: parse, dispatch, and stop the model on the way out
│   ├── cli.rs               # Front end: one action, one exit code, --json on everything
│   ├── ui.rs                # Front end: the window (feature `gui`)
│   ├── tui.rs               # Front end: the terminal (feature `tui`)
│   ├── app.rs               # All state and every operation — UI-agnostic
│   ├── jobs.rs              # Worker threads and the event channel the front ends drain
│   ├── config.rs            # Layered config (system → user → local) & preferences
│   ├── filter.rs            # Which models policy permits: glob patterns, allow/block
│   ├── knowledge.rs         # What can be said about a model, and what happens when nothing can
│   ├── models.rs            # Checkpoint catalogue + download status board
│   ├── runtime.rs           # Finding/fetching/verifying the binary that runs it
│   ├── plan.rs              # Planning instruction + structural checks on the result
│   ├── shrink.rs            # Telegraphic rewrite: instruction, chunking, integrity checks
│   ├── hints.rs             # Hint subjects (selection, question, or both) + composition
│   ├── pii/
│   │   └── mod.rs           # Findings, placeholder plan, overlap resolution
│   ├── router/
│   │   ├── mod.rs           # Candidate enumeration, withholding, Ranking/Choice
│   │   ├── llm.rs           # The one place that runs the model: difficulty, ranking, PII
│   │   └── hub.rs           # Hugging Face cache probe / download with progress
│   ├── store/
│   │   ├── mod.rs           # PromptStore: list/create/read/write/rename/delete
│   │   └── version.rs       # Version history (snapshots, diffs, index)
│   ├── editor/
│   │   ├── mod.rs           # Buffer: text, selection, caret edits, dirty tracking
│   │   └── undo.rs          # Snapshot-based undo/redo with coalescing
│   └── agents/
│       ├── registry.rs      # Static agent/model specs
│       ├── configured.rs    # Which model an agent's own config says it will run
│       ├── detect.rs        # What is installed, and whether it works
│       ├── launch.rs        # Running one, with streaming output
│       └── failover.rs      # Down the shortlist when one is unavailable
├── Cargo.toml
└── README.md
```

**Key design decisions:**
- **Plain files** — Prompts are `.md`. Sidecar state in `.pstore/`. No database, no lock-in.
- **Snapshot-based undo** — Coalesces typing into word/line granules. Programmatic edits are single atomic steps.
- **Static agent registry** — All agent CLIs and model specs in one file. One-line updates for flag changes.
- **The model does the judging** — Ranking is a prompt, not a formula. No hand-maintained skill vectors to go stale.
- **One core, three renderers** — `app.rs` holds the state and the operations and names no toolkit; `ui.rs`, `tui.rs` and `cli.rs` only present them. A feature nobody can reach from the CLI would be a bug, not a design.
- **Nothing ranked blind** — A model pstore cannot identify is withheld with a reason, not scored on a guess. A shorter honest shortlist beats a full one with fiction in it.
- **Subprocess inference, one load per operation** — The model runs as a child process, the same way agents do. Nothing is linked in, nothing stays resident *between* operations, and a session still generating is killed when the app quits rather than orphaned with the weights mapped.
- **No silent degradation** — Model-dependent features are disabled with a reason when the model is unavailable, never quietly replaced by something worse.
- **Nothing applied unreviewed** — Shrink, plan and sanitize all propose a diff. Accepting is one undo step, and the previous text is already in version history.
- **egui/eframe and ratatui** — Native GUI, no Electron; and a terminal UI that is a peer of it, not a cut-down view. Both optional at build time.

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

```jsonc
{
  "hint_score_tolerance": 8.0,     // points of fit a hint may trade for speed
  "preview": false,                // start in preview mode
  "sidebar_width": 260,            // sidebar width in points
  "pinned_agent": null,            // override the ranker's pick
  "allow_model_download": true,    // may pstore reach the network at all
  "allow_model_lookup": true,      // may it look up a model it cannot describe (name only)
  "llama_path": null,              // use your own PrismML llama-server build
  "local_model": "1-bit",          // or "ternary"
  "model_context_ceiling": 8192,   // hard cap; the window is fitted per operation, far below it
  "model_reasoning_budget": 1400,  // characters of reasoning before it must answer; 0 disables
  "filter": {
    "block": ["*fable*"],          // patterns that disqualify a model
    "allow": [],                   // if non-empty, ONLY these are permitted
    "efforts": [],                 // if non-empty, ONLY these effort levels
    "block_metered": true          // refuse models billed per token
  }
}
```

`allow_model_download: false` means pstore never reaches the network — for the runtime, the
weights and model lookups alike — and the features that need the model are disabled rather than
degraded. `allow_model_lookup: false` switches off only the lookup, which is the one request that
is not a download: a model's *name*, sent to a search engine, when nothing local can describe it.
Never the prompt.

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
# Run in dev mode — window, terminal, or a one-shot command
cargo run
cargo run -- tui
cargo run -- rank prompt.md

# Run tests. 278 of them, none of which need the checkpoint.
cargo test

# The ones that do need it are ignored by default and take minutes. They run against *every*
# build on disk, not just the selected one — the 1-bit build is the one that used to rank by
# counting, so a test that only exercised the default would pass while it stayed broken.
cargo test -- --ignored live_model --nocapture
cargo test -- --ignored live_wide_field --nocapture

# Check formatting
cargo fmt --check

# Lint
cargo clippy --all-targets -- -D warnings

# Every feature combination has to build clean
for f in "" "local-llm" "tui" "gui" "gui,tui" "local-llm,gui,tui"; do
  cargo check --no-default-features --features "$f" --all-targets
done
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Links

- **Website:** https://alexpacio.github.io/pstore
- **Repository:** https://github.com/alexpacio/pstore
- **Issues:** https://github.com/alexpacio/pstore/issues