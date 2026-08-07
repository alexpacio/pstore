//! What pstore can tell the ranker about a model — and what it does when it can tell it
//! nothing.
//!
//! Ranking is a judgement about models, and a judgement needs facts. The ranker used to be
//! handed lines like `(agent default) via Crush [mid, effort high]`: a candidate with no name,
//! no vendor and no capability, sitting in the same list as Opus 5 and Haiku 4.5. The
//! checkpoint cannot say "I don't know what that is" — it is being asked for five ranked
//! choices — so it invents a placement, and one invented row displaces every real one below
//! it. **That is the failure this module exists to prevent.**
//!
//! Three answers, in this order, for every distinct model in the field:
//!
//! 1. **[`FACTS`] — pstore's own table.** One line per model, maintained beside the registry.
//!    It comes first, ahead of the checkpoint's own memory, and that ordering is deliberate:
//!    the checkpoint's training predates every model in the table, so on `Opus 5` or
//!    `Gemini 3` its recollection is not knowledge but a plausible-sounding guess about a name
//!    it has seen a pattern of. A stated fact beats a remembered one.
//! 2. **Ask the checkpoint.** For a name pstore does not describe — one read out of an agent's
//!    own config, typically — the model is asked whether it actually knows it, in its own
//!    call, before any ranking happens. It answers with indices, and it is told to say nothing
//!    rather than guess.
//! 3. **Look it up.** A name neither pstore nor the checkpoint knows is searched for, and the
//!    result is passed into the ranking prompt as facts. Only the **name** leaves the machine —
//!    never the prompt, never the file, never the project — and the answer is cached on disk so
//!    a given name is looked up once. `allow_model_lookup: false` (or `allow_model_download:
//!    false`) turns this off.
//!
//! A model that survives none of the three is **excluded from ranking**, with a reason the user
//! can read next to the shortlist. That is the point of the whole module: a shortlist of four
//! models pstore understands is worth more than a shortlist of five where one is fiction.

use std::collections::BTreeMap;
use std::path::PathBuf;

/// Where a model's facts came from. Shown in the UI so a ranking can be accounted for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// [`FACTS`], pstore's own table.
    Table,
    /// The checkpoint said it knows this model, so it needs no telling.
    Checkpoint,
    /// Looked up over the network.
    Web,
}

impl Source {
    /// How to describe this provenance in one word.
    pub fn label(self) -> &'static str {
        match self {
            Source::Table => "pstore's table",
            Source::Checkpoint => "the local model's own knowledge",
            Source::Web => "a web lookup",
        }
    }
}

/// What the ranker is told about one model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Known {
    /// The name as the candidate carries it.
    pub model: String,
    /// One line of facts, or empty when the checkpoint already knows the model.
    pub note: String,
    /// Where the line came from.
    pub source: Source,
}

/// The outcome of resolving a field of models.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Brief {
    /// Models the ranker may be asked about, with whatever pstore can tell it.
    pub known: Vec<Known>,
    /// Models withheld from ranking, each with the reason.
    pub unknown: Vec<(String, String)>,
}

impl Brief {
    /// Whether `model` may be ranked.
    pub fn permits(&self, model: &str) -> bool {
        self.known.iter().any(|k| k.model == model)
    }

    /// Where the facts about `model` came from, if pstore has any.
    ///
    /// Shown next to a ranked choice so a placement can be accounted for: a pick made from
    /// pstore's own table is a different kind of claim from one the checkpoint made unaided.
    pub fn source(&self, model: &str) -> Option<Source> {
        self.known.iter().find(|k| k.model == model).map(|k| k.source)
    }

    /// The note for `model`, if there is one worth spending tokens on.
    pub fn note(&self, model: &str) -> Option<&str> {
        self.known
            .iter()
            .find(|k| k.model == model)
            .map(|k| k.note.as_str())
            .filter(|n| !n.is_empty())
    }
}

/// Why a model cannot be ranked, when nothing describes it.
///
/// Phrased as something the user can act on. "Unknown model" would send them looking for a bug;
/// each of these points at the thing that would fix it.
fn why_unknown(model: &str) -> String {
    if model.trim().is_empty() {
        return "this agent picks its own model and its config does not say which — pstore will \
                not rank a model it cannot name"
            .into();
    }
    let prefs = crate::config::prefs_snapshot();
    if !prefs.allow_model_lookup || !prefs.allow_model_download {
        format!(
            "nothing describes {model}: it is not in pstore's table, the local model does not \
             know it, and lookups are switched off"
        )
    } else {
        format!(
            "nothing describes {model}: it is not in pstore's table, the local model does not \
             know it, and the lookup found nothing"
        )
    }
}

/// Work out what can be said about every model in the field.
///
/// Both slow steps are injected rather than called directly, which keeps this module free of
/// process handling and HTTP, and lets the resolution *order* — the part that is easy to get
/// wrong and invisible when it is — be tested without a 7.17 GB checkpoint or a network
/// connection:
///
/// * `probe` asks the local checkpoint which of a list of names it knows, returning indices into
///   that list. Pass [`crate::router::llm::known_models`].
/// * `find` looks a name up over the network. Pass [`lookup`].
///
/// Neither runs when [`FACTS`] already covers the field, which on a stock installation is always:
/// the common path spends no extra model call and touches no network.
///
/// A probe that fails is not fatal. Its failure means "the checkpoint told us nothing", the
/// lookup still gets its turn, and whatever is really wrong with the model surfaces from the
/// ranking call itself with a better message than this one could give.
/// `supplied` answers for models whose own vendor described them — see
/// [`crate::agents::catalog`]. It is consulted after [`FACTS`] and before the checkpoint, because
/// pstore's own line is written for ranking and the vendor's is marketing, but a vendor's
/// description of a model it shipped last week beats a checkpoint that has never heard of it.
/// Without this, discovering a new model would surface it and then immediately withhold it.
pub fn resolve(
    models: &[String],
    supplied: &dyn Fn(&str) -> Option<String>,
    probe: impl FnOnce(&[String]) -> Result<Vec<usize>, String>,
    find: impl Fn(&str) -> Option<String>,
) -> Brief {
    let mut brief = Brief::default();
    let mut pending: Vec<String> = Vec::new();

    for model in dedup(models) {
        match from_table(&model).map(str::to_string).or_else(|| {
            supplied(&model)
                .map(|note| trim_note(&note))
                .filter(|note| !note.is_empty())
        }) {
            Some(note) => brief.known.push(Known {
                model,
                note,
                source: Source::Table,
            }),
            // A nameless candidate skips the checkpoint and the network alike: there is no
            // question to ask about it, which is the whole problem with it.
            None if model.trim().is_empty() => {
                brief.unknown.push((model.clone(), why_unknown(&model)));
            }
            None => pending.push(model),
        }
    }

    if pending.is_empty() {
        return brief;
    }

    let recognised = probe(&pending).unwrap_or_default();
    for (i, model) in pending.into_iter().enumerate() {
        if recognised.contains(&i) {
            brief.known.push(Known {
                model,
                // Nothing to tell it: it says it knows this one, and repeating pstore's guess
                // back at it would only give it something to contradict.
                note: String::new(),
                source: Source::Checkpoint,
            });
            continue;
        }
        match find(&model).map(|note| trim_note(&note)) {
            Some(note) => brief.known.push(Known {
                model,
                note,
                source: Source::Web,
            }),
            None => {
                let why = why_unknown(&model);
                brief.unknown.push((model, why));
            }
        }
    }
    brief
}

/// Distinct names, order preserved.
///
/// The field arrives as one entry per (model, effort) pair, so a five-effort agent asks the same
/// question five times. Resolving once per *model* is what keeps a lookup — and the notes in the
/// prompt — from being paid for repeatedly.
fn dedup(models: &[String]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for m in models {
        if !out.contains(m) {
            out.push(m.clone());
        }
    }
    out
}

/// One line of facts about one model.
///
/// Terse on purpose: every character here is a token the checkpoint evaluates on every ranking
/// call, and the notes are the only part of the prompt that grows with the number of agents
/// installed. Twelve words that change a decision, not a spec sheet.
#[derive(Debug, Clone, Copy)]
pub struct Fact {
    /// Names this line answers to, lowercase. The registry id and display name, plus the
    /// spellings agents' own configs use for the same model.
    pub names: &'static [&'static str],
    /// What the ranker needs to know.
    pub note: &'static str,
}

/// pstore's own table of model facts.
///
/// Maintained by hand, beside the registry, and that is a cost worth being explicit about: it
/// goes stale when vendors ship. It is much the smaller of two evils. The alternative is asking
/// a checkpoint whose training ended before any of these models existed to recall them, and
/// what comes back is not a blank but a confident invention — which is indistinguishable from
/// knowledge in a ranking, and the reason ranking got poisoned in the first place.
///
/// Note the entries with no counterpart in [`crate::agents::registry`]: those are the models
/// people actually configure Crush, Goose, Aider and Cursor with, discovered by
/// [`crate::agents::configured`] and previously ranked as `(agent default)` — that is, ranked
/// blind.
pub const FACTS: &[Fact] = &[
    // --- Anthropic, as Claude Code exposes them ------------------------------------------
    Fact {
        names: &["haiku", "haiku 4.5", "claude-haiku-4-5", "claude-3-5-haiku"],
        note: "Anthropic's light model: fastest and cheapest, fine for small well-specified \
               edits, weak on multi-file reasoning",
    },
    Fact {
        names: &["sonnet", "sonnet 5", "claude-sonnet-5", "claude-sonnet-4-5"],
        note: "Anthropic's mid tier: strong general coding, the sensible default for ordinary \
               multi-file work",
    },
    Fact {
        names: &["opus", "opus 5", "claude-opus-5", "claude-opus-4-5"],
        note: "Anthropic's frontier model: best available for hard refactors, unfamiliar \
               codebases and long agentic runs; slowest",
    },
    Fact {
        names: &["fable", "fable 5", "claude-fable-5"],
        note: "Anthropic frontier model billed per token: strongest at long-horizon planning, \
               and the only option here that costs extra money",
    },
    // --- OpenAI --------------------------------------------------------------------------
    Fact {
        names: &["gpt-5.1-codex", "gpt-5.1 codex", "gpt-5-codex", "codex"],
        note: "OpenAI's coding-specialised frontier model: strong at repo-scale edits, tests \
               and tool use",
    },
    Fact {
        names: &["gpt-5", "gpt-5.1", "gpt-4o", "o3", "o4-mini"],
        note: "OpenAI general-purpose model: capable across coding and reasoning, not \
               coding-specialised",
    },
    // --- Google --------------------------------------------------------------------------
    Fact {
        names: &["gemini-3-flash", "gemini 3 flash", "gemini-2.5-flash"],
        note: "Google's light model: very fast with a very large context, weaker on hard \
               reasoning",
    },
    Fact {
        names: &["gemini-3-pro", "gemini 3 pro", "gemini-2.5-pro"],
        note: "Google's strong model: good reasoning with a very large context, competitive on \
               coding",
    },
    // --- Models people configure the other agents with -----------------------------------
    Fact {
        names: &["qwen3-coder", "qwen3-coder-plus", "qwen2.5-coder"],
        note: "Alibaba's open coding model: solid on single-file and well-scoped edits, below \
               the frontier on long reasoning",
    },
    Fact {
        names: &[
            "deepseek-v3",
            "deepseek-chat",
            "deepseek-r1",
            "deepseek-coder",
        ],
        note: "DeepSeek's open model: strong coding and maths for its cost, weaker at long \
               agentic tool use",
    },
    Fact {
        names: &["kimi-k2", "kimi-k2-instruct"],
        note: "Moonshot's open model: large, agentic tool use its strength, coding respectable",
    },
    Fact {
        names: &["glm-4.6", "glm-4.5", "glm-4.6-air"],
        note: "Zhipu's open model: capable general coding at low cost, below the frontier on \
               hard reasoning",
    },
    Fact {
        names: &["grok-code-fast-1", "grok-4", "grok-code"],
        note: "xAI's model: fast, coding-tuned, mid-tier on hard multi-file work",
    },
    Fact {
        names: &["mistral-large", "codestral", "devstral"],
        note: "Mistral's model: competent on focused code edits, mid-tier on reasoning",
    },
    Fact {
        names: &["llama-3.3-70b", "llama-4-maverick", "llama-3.1-405b"],
        note: "Meta's open model: general-purpose, below current coding-tuned models on \
               refactors",
    },
];

/// What [`FACTS`] says about `model`, if anything.
///
/// Matched on the bare name — provider prefixes are stripped, so `anthropic/claude-sonnet-4-5`
/// and `claude-sonnet-4-5` are the same question — and case-insensitively, because these
/// strings are typed into config files by people.
pub fn from_table(model: &str) -> Option<&'static str> {
    let bare = crate::agents::configured::bare_name(model).to_ascii_lowercase();
    if bare.is_empty() {
        return None;
    }
    FACTS
        .iter()
        .find(|f| f.names.iter().any(|n| *n == bare))
        .map(|f| f.note)
}

/// Longest note pstore will put in a ranking prompt, in characters.
///
/// A web lookup returns a paragraph; the ranker needs a line. Everything past this is cost
/// without effect — and worse, a long note about one candidate makes it loom over the ones
/// described in twelve words.
pub const NOTE_CHARS: usize = 200;

/// Trim a note to something worth its tokens, cutting at a word boundary.
pub fn trim_note(raw: &str) -> String {
    let flat = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= NOTE_CHARS {
        return flat;
    }
    let cut: String = flat.chars().take(NOTE_CHARS).collect();
    match cut.rsplit_once(' ') {
        Some((head, _)) => format!("{head}…"),
        None => cut,
    }
}

// ---------------------------------------------------------------------------
// Looked-up notes, cached on disk
// ---------------------------------------------------------------------------

/// Where looked-up notes are remembered between runs.
///
/// The user cache directory rather than the project: what a model is does not depend on which
/// project is open, and re-looking-up the same name once per checkout would be both slower and
/// more traffic for no gain.
fn cache_path() -> Option<PathBuf> {
    dirs::cache_dir().map(|d| d.join("pstore").join("models.json"))
}

/// Notes previously looked up, keyed by lowercase bare name.
///
/// A miss is cached as an empty string, deliberately: a name that is not a public model — a
/// typo in someone's config, an internal deployment id — must not cost a network round trip on
/// every ranking for the rest of time.
fn load_cache() -> BTreeMap<String, String> {
    cache_path()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn save_cache(cache: &BTreeMap<String, String>) {
    let Some(path) = cache_path() else { return };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(json) = serde_json::to_string_pretty(cache) {
        let _ = std::fs::write(path, json);
    }
}

/// Find out what `model` is, from the network, and remember the answer.
///
/// Returns `None` when the lookup is not permitted, cannot reach anything, or finds nothing
/// that reads like a description — and `None` means the model is excluded from ranking rather
/// than ranked on a guess.
///
/// Blocking, with a short timeout: this runs on the ranking worker, and a search engine that
/// has decided to be slow today must not turn a 13-second ranking into a minute of nothing.
pub fn lookup(model: &str) -> Option<String> {
    let prefs = crate::config::prefs_snapshot();
    // `allow_model_download` is the wider promise — "pstore makes no network request" — so it
    // covers this too. Either switch being off is enough to stay offline.
    if !prefs.allow_model_lookup || !prefs.allow_model_download {
        return None;
    }

    let key = crate::agents::configured::bare_name(model).to_ascii_lowercase();
    if key.is_empty() {
        return None;
    }
    let mut cache = load_cache();
    if let Some(hit) = cache.get(&key) {
        // An empty entry is a remembered miss, and it stays a miss.
        return Some(hit.clone()).filter(|n| !n.is_empty());
    }

    let found = web::describe(&key).map(|n| trim_note(&n));
    cache.insert(key, found.clone().unwrap_or_default());
    save_cache(&cache);
    found
}

/// Searching the public web for what a model is.
///
/// **This is the one place in pstore where something about the user's setup leaves the
/// machine**, and what leaves is a model's name — the same string that is printed on the
/// vendor's pricing page. Never the prompt, never the file, never the project.
///
/// Two sources, cheapest first. Both are best-effort by nature: an answer that does not arrive,
/// or does not look like prose, leaves the model unranked rather than ranked on nonsense.
mod web {
    use std::time::Duration;

    /// How long to wait on a search before giving up on it.
    ///
    /// Short on purpose. This is a nicety in the middle of a ranking call; a model whose
    /// description takes longer than this to find is a model pstore reports as unknown.
    const TIMEOUT: Duration = Duration::from_secs(6);

    /// Shortest reply worth treating as a description.
    const MIN_CHARS: usize = 40;

    /// Ask what `name` is, or return nothing.
    pub fn describe(name: &str) -> Option<String> {
        instant_answer(name).or_else(|| first_snippet(name))
    }

    fn agent() -> ureq::Agent {
        ureq::Agent::config_builder()
            .timeout_global(Some(TIMEOUT))
            // Identifying the caller honestly is both good manners and what keeps a scraped
            // endpoint from treating pstore as something to block.
            .user_agent(concat!("pstore/", env!("CARGO_PKG_VERSION")))
            .build()
            .into()
    }

    fn get(url: &str) -> Option<String> {
        agent()
            .get(url)
            .call()
            .ok()?
            .body_mut()
            .read_to_string()
            .ok()
    }

    /// DuckDuckGo's Instant Answer API: documented, JSON, no key, no scraping.
    ///
    /// Tried first because it is the only source here with a contract. It answers for models
    /// notable enough to have an encyclopaedia entry and returns an empty abstract for the
    /// rest, which is what the second source is for.
    fn instant_answer(name: &str) -> Option<String> {
        let url = format!(
            "https://api.duckduckgo.com/?q={}&format=json&no_html=1&no_redirect=1",
            encode(&format!("{name} AI model"))
        );
        let body = get(&url)?;
        let json: serde_json::Value = serde_json::from_str(&body).ok()?;
        let abstract_text = json.get("AbstractText").and_then(|v| v.as_str())?;
        plausible(abstract_text)
    }

    /// The first result snippet from DuckDuckGo's HTML-only endpoint.
    ///
    /// A scrape, with everything that implies: no contract, and it will break when the markup
    /// changes. That is survivable here precisely because the failure mode is "pstore reports
    /// this model as unknown", which is the same thing that happens with no network at all.
    fn first_snippet(name: &str) -> Option<String> {
        let url = format!(
            "https://lite.duckduckgo.com/lite/?q={}",
            encode(&format!("{name} language model capabilities"))
        );
        let body = get(&url)?;
        body.split("result-snippet")
            .skip(1)
            .filter_map(|chunk| {
                let text = chunk.split('<').next().unwrap_or_default();
                plausible(&strip_entities(text.trim_start_matches(['"', '>', ' '])))
            })
            .next()
    }

    /// Whether a reply reads like a description rather than boilerplate.
    fn plausible(text: &str) -> Option<String> {
        let text = text.trim();
        if text.chars().count() < MIN_CHARS {
            return None;
        }
        // A description is prose. A cookie banner, a login wall or a JSON fragment is not.
        if !text.contains(' ') || text.starts_with('{') || text.starts_with('<') {
            return None;
        }
        Some(text.to_string())
    }

    /// The handful of entities a search snippet actually contains.
    fn strip_entities(s: &str) -> String {
        s.replace("&quot;", "\"")
            .replace("&#x27;", "'")
            .replace("&#39;", "'")
            .replace("&amp;", "&")
            .replace("&nbsp;", " ")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
    }

    /// Percent-encode a query. Small by hand rather than a dependency for one call site.
    fn encode(s: &str) -> String {
        let mut out = String::with_capacity(s.len() * 3);
        for b in s.bytes() {
            match b {
                b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                    out.push(b as char);
                }
                b' ' => out.push('+'),
                other => out.push_str(&format!("%{other:02X}")),
            }
        }
        out
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn queries_are_encoded() {
            assert_eq!(encode("gpt-5 AI model"), "gpt-5+AI+model");
            assert_eq!(encode("a/b:c"), "a%2Fb%3Ac");
            assert_eq!(encode("caffè"), "caff%C3%A8");
        }

        /// The filter is what stops a cookie banner becoming a model's description.
        #[test]
        fn only_prose_survives() {
            assert_eq!(plausible("short"), None);
            assert_eq!(plausible(""), None);
            assert_eq!(
                plausible(&"x".repeat(100)),
                None,
                "one long word is not prose"
            );
            assert_eq!(
                plausible("{\"error\": \"this is a json body, not prose\"}"),
                None
            );
            assert_eq!(
                plausible("<div>markup, not prose, but long enough</div>"),
                None
            );

            let real = "Claude is a family of large language models developed by Anthropic.";
            assert_eq!(plausible(real), Some(real.to_string()));
        }

        #[test]
        fn entities_are_unescaped() {
            assert_eq!(
                strip_entities("Anthropic&#x27;s &quot;Claude&quot; &amp; friends"),
                "Anthropic's \"Claude\" & friends"
            );
        }

        /// Against the real endpoints. Ignored by default: it needs the network, and a test
        /// that fails on a train is a test people learn to skip.
        ///
        /// `cargo test -- --ignored live_lookup --nocapture`
        #[test]
        #[ignore = "needs network access"]
        fn live_lookup_finds_a_well_known_model() {
            for name in ["claude", "gpt-4", "llama"] {
                match describe(name) {
                    Some(note) => eprintln!("{name}: {note}"),
                    None => eprintln!("{name}: nothing found"),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The table is the first thing consulted for every ranking, so a duplicated or empty
    /// entry is a silent behaviour change rather than a cosmetic problem.
    #[test]
    fn the_table_is_well_formed() {
        let mut seen: Vec<&str> = Vec::new();
        for fact in FACTS {
            assert!(!fact.names.is_empty(), "a fact with no names: {fact:?}");
            assert!(
                fact.note.len() > 20,
                "a note that says nothing: {:?}",
                fact.note
            );
            assert!(
                fact.note.chars().count() <= NOTE_CHARS,
                "note is longer than the ranker will accept: {:?}",
                fact.note
            );
            for name in fact.names {
                assert_eq!(
                    *name,
                    name.to_ascii_lowercase(),
                    "names are matched lowercase, so they must be written that way"
                );
                assert!(
                    !seen.contains(name),
                    "{name} is described twice — which note wins is then an accident of order"
                );
                seen.push(name);
            }
        }
    }

    /// Every model the registry offers has to be described, or ranking is poisoned by the
    /// models pstore ships with rather than the ones it discovers. This is the test that fails
    /// when a vendor's model is added to the registry and nobody wrote its line.
    #[test]
    fn every_registry_model_is_described() {
        for agent in crate::agents::registry::AGENTS {
            for model in agent.models {
                assert!(
                    from_table(model.id).is_some(),
                    "{}/{} has no entry in FACTS — it would be ranked blind",
                    agent.id,
                    model.id
                );
                assert!(
                    from_table(model.display).is_some(),
                    "{}/{} is described by id but not by display name",
                    agent.id,
                    model.display
                );
            }
        }
    }

    #[test]
    fn lookup_is_by_bare_name_and_case_insensitive() {
        assert!(from_table("anthropic/claude-sonnet-4-5").is_some());
        assert!(from_table("Sonnet 5").is_some());
        assert!(from_table("openai:gpt-5").is_some());
        assert_eq!(from_table("gpt-5"), from_table("GPT-5"));

        // The placeholder for "the agent chose, and did not say" must not resolve to
        // anything: it is the exact case that has to reach the unknown list.
        assert_eq!(from_table(""), None);
        assert_eq!(from_table("(agent default)"), None);
        assert_eq!(from_table("some-model-nobody-has-heard-of"), None);
    }

    #[test]
    fn notes_are_trimmed_at_a_word_boundary() {
        let long = "word ".repeat(200);
        let trimmed = trim_note(&long);
        assert!(trimmed.chars().count() <= NOTE_CHARS + 1, "{trimmed:?}");
        assert!(trimmed.ends_with('…'));
        assert!(!trimmed.contains("  "), "whitespace should be collapsed");

        // Something already short comes back unchanged apart from its whitespace.
        assert_eq!(trim_note("  a\n  short   note "), "a short note");
    }

    #[test]
    fn a_brief_answers_what_may_be_ranked() {
        let brief = Brief {
            known: vec![
                Known {
                    model: "opus".into(),
                    note: "frontier".into(),
                    source: Source::Table,
                },
                Known {
                    model: "mystery".into(),
                    note: String::new(),
                    source: Source::Checkpoint,
                },
            ],
            unknown: vec![("".into(), "the agent did not say".into())],
        };

        assert!(brief.permits("opus"));
        assert_eq!(brief.note("opus"), Some("frontier"));
        // The checkpoint knows this one, so there is nothing to spend tokens telling it.
        assert!(brief.permits("mystery"));
        assert_eq!(brief.note("mystery"), None);
        assert!(!brief.permits(""));
        assert!(!Brief::default().permits("opus"));
    }

    /// The resolution order is the design: pstore's stated facts beat the checkpoint's
    /// recollection, and the checkpoint is asked before anything reaches the network.
    #[test]
    fn the_table_wins_and_the_probe_is_only_asked_the_rest() {
        let asked = std::cell::RefCell::new(Vec::new());
        let brief = resolve(
            &[
                "opus".into(),
                "opus".into(), // the same model at another effort level
                "some-internal-deployment".into(),
            ],
            &|_| None,
            |pending| {
                asked.borrow_mut().extend_from_slice(pending);
                Ok(Vec::new()) // knows none of them
            },
            |_| None, // and nothing is found for them either
        );

        assert_eq!(
            asked.into_inner(),
            vec!["some-internal-deployment".to_string()],
            "a model pstore describes must not cost a probe, and neither must a duplicate"
        );
        assert!(brief.permits("opus"));
        assert_eq!(
            brief
                .known
                .iter()
                .find(|k| k.model == "opus")
                .map(|k| k.source),
            Some(Source::Table)
        );
    }

    /// A model the checkpoint recognises is ranked without a note and without a lookup: it has
    /// its own facts, and pstore's guess would only be something for it to argue with.
    #[test]
    fn a_model_the_checkpoint_knows_needs_no_telling() {
        let brief = resolve(
            &["mystery-model".into()],
            &|_| None,
            |_| Ok(vec![0]),
            |_| panic!("a model the checkpoint knows must not be looked up"),
        );
        assert!(brief.permits("mystery-model"));
        assert_eq!(brief.note("mystery-model"), None);
        assert_eq!(brief.known[0].source, Source::Checkpoint);
        assert!(brief.unknown.is_empty());
    }

    /// The whole point: a candidate nothing can describe is withheld from the ranking, with a
    /// reason, rather than ranked on an invention.
    #[test]
    fn a_nameless_candidate_is_excluded_with_a_reason() {
        // The `(agent default)` case — an agent that chooses its own model and whose config
        // did not say which. It must not even reach the probe.
        let probed = std::cell::Cell::new(false);
        let brief = resolve(
            &[String::new()],
            &|_| None,
            |_| {
                probed.set(true);
                Ok(vec![0])
            },
            |_| panic!("a nameless model must not be looked up either"),
        );

        assert!(
            !probed.get(),
            "there is nothing to ask about a nameless model"
        );
        assert!(brief.known.is_empty());
        assert_eq!(brief.unknown.len(), 1);
        assert!(
            brief.unknown[0].1.contains("does not say"),
            "the reason should point at the agent's config, got {:?}",
            brief.unknown[0].1
        );
    }

    /// A probe that cannot run must not take the ranking down with it — the model may still be
    /// describable from the table or a lookup, and the real failure will surface with a better
    /// message when ranking itself runs.
    #[test]
    fn a_failing_probe_degrades_to_unknown_rather_than_erroring() {
        let brief = resolve(
            &["opus".into(), "obscure-thing".into()],
            &|_| None,
            |_| Err("the model is not downloaded".into()),
            |_| None,
        );
        assert!(brief.permits("opus"), "the table does not need the probe");
        assert!(!brief.permits("obscure-thing"));
    }

    /// Every provenance has to read differently: "why is this model in the list?" is a
    /// question the UI answers with this string.
    #[test]
    fn sources_describe_themselves_distinctly() {
        let all = [Source::Table, Source::Checkpoint, Source::Web];
        let mut labels: Vec<&str> = all.iter().map(|s| s.label()).collect();
        labels.sort_unstable();
        labels.dedup();
        assert_eq!(labels.len(), all.len());
    }
}
