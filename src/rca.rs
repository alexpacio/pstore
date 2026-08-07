//! Turning incident notes into a root cause analysis, a postmortem and the work that follows.
//!
//! The third of the local rewrites, after [`crate::shrink`] and [`crate::plan`], and the one
//! whose input is not a request at all. What goes in is what an incident leaves behind — a
//! pasted alert, a scrollback of what was tried, a chat log with times in it. What comes out
//! is the document an on-call team owes the rest of the organisation afterwards: what broke,
//! for whom, why, and what is being done so it does not happen again.
//!
//! Unlike a plan, this output *is* a document for a human to read — several humans, usually,
//! most of whom were asleep when it happened. That changes what the instruction asks for but
//! not where it runs: an incident write-up quotes hostnames, customer counts, stack traces and
//! internal service names, which is exactly the material that should not leave the machine to
//! be summarised. So this runs on the same local checkpoint as everything else pstore infers.
//!
//! Two rules carry more weight than the rest, and both exist because of how this fails rather
//! than how it succeeds:
//!
//! * **Nothing is invented.** A model asked why a service fell over will always produce an
//!   answer, and a confident wrong root cause in a postmortem is worse than no postmortem —
//!   it closes the question. Anything the notes do not establish belongs in `open_questions`.
//! * **Nobody is named.** Postmortems are blameless because a culture that names an engineer
//!   in one stops getting honest notes for the next. Systems, services and roles only.
//!
//! The first of those is defended three times over, because asking once did not hold. On
//! telemetry whose notes carried no impact figures at all, an earlier version of this module
//! produced "users affected: 342" and "requests failed: 1,204" in the same register as the
//! measured numbers, and described a rollback that had not happened. So: the rule now leads
//! the instruction rather than closing it; the schema no longer *requires* a resolution, which
//! is what had been forcing one to be written; and [`warnings`] reads the finished document
//! back against the notes and reports every figure in it that the notes never gave. The last
//! of those is the one that catches a checkpoint having a bad day, and it is deliberately
//! quiet — see [`figures_not_in_the_notes`] for what it declines to flag and why.
//!
//! Like the others, nothing is applied unreviewed: the result arrives as a proposal with a
//! diff against the current prompt, and accepting it is one undo step with the raw notes
//! already in version history. [`run`] is the entry point; it is blocking and belongs on a
//! worker thread.

/// The analysis instruction.
///
/// Describes the *fields* rather than a markdown layout, for the reason
/// [`crate::plan::INSTRUCTION`] does: the layout is [`render`]'s job, and a checkpoint asked
/// to produce nine headings produces a paragraph about the nine headings it is about to
/// produce. The schema in [`crate::router::llm::rca`] enforces the shape; this text only has
/// to say what belongs in each field.
pub const INSTRUCTION: &str = "\
Turn the incident notes below into a root cause analysis and postmortem.

The rule that outranks every other instruction here: every time, number, quantity, hostname, \
service name and error string in your answer must already appear in the notes. You have no \
information except the notes. If the notes do not record something — how many users were \
affected, when it was detected, whether it was fixed — leave that field empty and put the \
question in open_questions. An empty field is a correct answer. A plausible figure that is \
not in the notes is the worst thing this document can contain, because it will be read as \
measured.

Write plain sentences. Never write JSON, braces, or key/value pairs in a field, whatever the \
notes look like. One point per entry, one or two sentences, under 200 characters — a longer \
entry is cut off where it stands. Say the finding itself, not what the entry is about.

Fill in each field.

summary — one paragraph: what broke, what it affected, when it started, and whether the notes \
show it ending.
impact — the damage, only in figures the notes give: users affected, requests failed, data \
lost, money, duration. Empty if the notes give none.
timeline — what happened, in order. Begin each entry with the time exactly as the notes write \
it, then what happened at that time: '14:02:15 redis-primary-01 logged OOM command not \
allowed'. One event per entry, and every time the notes record gets an entry.
root_cause — the causal chain, one link per entry, ending at the condition or change that \
made the failure possible. Not the trigger alone: the trigger is what set it off, the root \
cause is why setting it off broke anything. Prefer the earliest change the notes record over \
the first symptom they show.
contributing_factors — what made it worse, longer, or harder to diagnose, but did not cause \
it. Empty if there are none.
detection — how the failure was noticed and how long after it began, if the notes say. Empty \
if they do not.
resolution — what the notes record as stopping the impact, saying which entries were \
temporary mitigation and which are permanent. Empty if the notes do not show it being \
resolved — do not describe a fix that has not happened.
action_items — the work this incident justifies. One task each, imperative and independently \
completable, naming the system, service or file it touches, with 'buys' saying whether the \
task prevents a recurrence, detects one sooner, or reduces the damage when it happens.
open_questions — what the notes do not establish, including every figure you were unable to \
give above.

Write blamelessly: name systems, services and roles, never individual people. Copy timestamps, \
error messages, identifiers, file paths and version numbers verbatim. Propose fixes in \
action_items and nowhere else.";

/// Fields the model fills in, in the order [`render`] lays them out.
///
/// The JSON key, and the heading it becomes. `summary` is first and is a single string; every
/// other field is a list.
pub const FIELDS: [(&str, &str); 9] = [
    ("summary", "Summary"),
    ("impact", "Impact"),
    ("timeline", "Timeline"),
    ("root_cause", "Root cause"),
    ("contributing_factors", "Contributing factors"),
    ("detection", "Detection"),
    ("resolution", "Resolution"),
    ("action_items", "Action items"),
    ("open_questions", "Open questions"),
];

/// Headings whose entries are numbered rather than bulleted.
///
/// Order is load-bearing in the timeline — it is the record of what happened when. Action
/// items are numbered for a different reason: they leave this document to become tickets, and
/// "action item 3" has to mean something in the meeting where they are handed out.
const NUMBERED: [&str; 2] = ["Timeline", "Action items"];

/// Headings that state their own absence rather than disappearing.
///
/// For most sections an empty list means "there were none", and a bare heading reads as
/// something the author forgot — so [`render`] drops them, as [`crate::plan::render`] does.
/// These four are the opposite: their emptiness is the finding. Notes that never say how many
/// users were affected, or that stop while the incident is still burning, produce an empty
/// Impact and an empty Resolution, and a reader has to be able to tell that from a document
/// where nobody filled them in. Dropping them here would also hide exactly the restraint the
/// instruction asks for, which is the behaviour most worth being able to see.
const STATE_IF_EMPTY: [&str; 4] = ["Impact", "Detection", "Resolution", "Open questions"];

/// What those headings say when the notes did not establish them.
const NOT_RECORDED: &str = "Not recorded in the notes.";

/// Assemble the model's fields into the postmortem.
pub fn render(summary: &str, sections: &[(&str, Vec<String>)]) -> String {
    let mut out = format!("**Summary**\n{}\n", entry_text(summary));
    for (heading, items) in sections {
        let items: Vec<String> = items
            .iter()
            .map(|i| entry_text(i))
            .filter(|i| !i.is_empty())
            .collect();
        if items.is_empty() {
            if STATE_IF_EMPTY.contains(heading) {
                out.push_str(&format!("\n**{heading}**\n- {NOT_RECORDED}\n"));
            }
            continue;
        }
        out.push_str(&format!("\n**{heading}**\n"));
        for (n, item) in items.iter().enumerate() {
            if NUMBERED.contains(heading) {
                out.push_str(&format!("{}. {item}\n", n + 1));
            } else {
                out.push_str(&format!("- {item}\n"));
            }
        }
    }
    out
}

/// One entry, reduced to the line it has to become.
///
/// Three things get cleaned up, each of them seen in real output rather than guarded against
/// on principle:
///
/// * **Its own list marker.** A model asked for a list of findings frequently writes the
///   bullet too, and `- • why did the deploy run unattended?` is the result of laying that
///   out again.
/// * **Newlines.** An entry holding two sentences on two lines silently becomes two list
///   items when it is written into a markdown document, so the second one loses its bullet
///   and its number.
/// * **Wrapping JSON.** Notes that contain a JSON audit log lead this checkpoint to answer
///   in kind, filling the string field with `{"timestamp": ..., "description": ...}`. The
///   instruction forbids it and the schema cannot; when it happens anyway, the readable part
///   is recovered rather than shown as a broken object.
fn entry_text(entry: &str) -> String {
    let mut text = entry.split_whitespace().collect::<Vec<_>>().join(" ");
    if text.starts_with('{') || text.starts_with("[{") {
        text = json_fragment(&text);
    }
    while let Some(rest) = without_leading_marker(&text) {
        text = rest.to_string();
    }
    text
}

/// `text` without one leading list marker, or `None` if it does not open with one.
///
/// The distinction this has to get right: a timeline entry is *supposed* to start with
/// digits. `09:14 deploy went out` and `1,400 failed checkouts` open with a number that is
/// the content, while `1. prevent: gate it` opens with one that is punctuation. So a numeric
/// marker counts only when the digits are followed by `.` or `)` and then a space — which
/// `09:14` and `1,400` are not.
fn without_leading_marker(text: &str) -> Option<&str> {
    let t = text.trim_start();
    if let Some(rest) = t.strip_prefix(['-', '*', '•', '·', '–']) {
        return rest.starts_with(' ').then(|| rest.trim_start());
    }
    let digits = t.len() - t.trim_start_matches(|c: char| c.is_ascii_digit()).len();
    if digits == 0 {
        return None;
    }
    let rest = &t[digits..];
    let rest = rest.strip_prefix(['.', ')'])?;
    rest.starts_with(' ').then(|| rest.trim_start())
}

/// The prose inside a JSON object the model wrote into a string field.
///
/// Splitting on `"` puts the quoted strings at odd positions, alternating key, value, key,
/// value; this takes the values and joins them, which for the shape this actually arrives in
/// — a timestamp and a description — recovers the entry. Best effort by construction, so a
/// fragment too mangled to read that way falls back to itself rather than to nothing: a
/// reviewer can still see what the model produced.
fn json_fragment(text: &str) -> String {
    let values: Vec<&str> = text
        .split('"')
        .skip(3)
        .step_by(4)
        .filter(|s| !s.trim().is_empty())
        .collect();
    if values.is_empty() {
        return text.to_string();
    }
    values.join(" ")
}

/// Compose the request sent to the local model.
pub fn compose(document: &str) -> String {
    format!("{INSTRUCTION}\n\n---\n\n{document}")
}

/// Analyse `text` on the local checkpoint.
///
/// Blocking and slow in units of seconds — the model is a subprocess that maps the weights
/// before it answers — so call it from a worker thread.
///
/// Notes too long for the context window are refused rather than analysed. This matters more
/// here than anywhere else in pstore: incident notes are the longest thing anyone pastes in,
/// llama.cpp truncates silently, and a postmortem built from the first half of an incident is
/// the failure that looks most like a success — it has every heading, a plausible root cause,
/// and it stops at the point where the notes were cut off.
pub fn run(text: &str) -> Result<String, String> {
    let limit = crate::router::llm::rca_input_chars();
    if text.len() > limit {
        return Err(format!(
            "the incident notes are too long to analyse in one pass ({} characters, limit \
             {limit}) — shrink them first, or raise `model_context_ceiling` in \
             .pstore/config.json",
            text.len()
        ));
    }
    let analysis = crate::router::llm::rca(text)?;
    if analysis.trim().is_empty() {
        return Err("the model returned an empty analysis".into());
    }
    Ok(analysis)
}

/// The action items, read back out of a rendered analysis.
///
/// The one section that leaves the document: it becomes tickets, and whoever cuts them wants
/// the list on its own rather than the whole write-up. Read back from the rendered text
/// rather than kept alongside it so that an edited postmortem yields its edited items — the
/// document is the artefact pstore versions, and a second copy of the list would be the one
/// that quietly went stale.
///
/// Numbering is stripped; the caller is presenting or scripting these, not reading them in
/// place.
pub fn action_items(analysis: &str) -> Vec<String> {
    analysis
        .lines()
        .skip_while(|l| l.trim() != "**Action items**")
        .skip(1)
        .take_while(|l| !l.trim_start().starts_with("**"))
        .filter_map(|l| {
            let item = l
                .trim()
                .trim_start_matches(|c: char| c.is_ascii_digit())
                .trim_start_matches(['.', '-', ')'])
                .trim();
            (!item.is_empty()).then(|| item.to_string())
        })
        .collect()
}

/// Problems in a produced analysis that its schema could not rule out.
///
/// The headings are guaranteed by [`render`] and the load-bearing lists are non-empty by
/// schema, so what is left to check is the relationship between the document and the notes it
/// claims to be about: what it dropped, and what it added. These are warnings rather than
/// rejections — a reader who has been told what to distrust can still use the document — but
/// [`figures_not_in_the_notes`] is close to a rejection in practice, and is meant to be read
/// that way.
pub fn warnings(analysis: &str, notes: &str) -> Vec<String> {
    let mut out = Vec::new();

    // The one that matters most. Everything else here is about material going missing; this
    // is about material appearing, which is the failure a postmortem cannot survive — a
    // reader has no way to tell an invented "1,204 requests failed" from a measured one, and
    // will quote it in the incident review.
    let invented = figures_not_in_the_notes(analysis, notes);
    if !invented.is_empty() {
        out.push(format!(
            "figures that do not appear in the notes: {} — the model was told to invent \
             nothing, so treat these as unmeasured until you have checked them",
            invented.join(", ")
        ));
    }

    // A timeline is a claim about the record. Times the notes established and the analysis
    // dropped are the ones nobody will notice are missing, because the section reads as
    // complete either way.
    let times = clock_times(notes);
    if times.is_empty() {
        out.push(
            "the notes contain no times — the timeline's order is the model's reading, not \
             the record's"
                .into(),
        );
    } else {
        let kept = clock_times(analysis);
        let dropped: Vec<String> = times.into_iter().filter(|t| !kept.contains(t)).collect();
        if !dropped.is_empty() {
            out.push(format!(
                "times dropped from the timeline: {}",
                dropped.join(", ")
            ));
        }
    }

    // Same check the planner makes, and for the same reason: a path is the thing a reader
    // needs to go look at, and the easiest thing for a summariser to paraphrase away.
    let paths = |s: &str| -> Vec<String> {
        s.split_whitespace()
            .filter_map(crate::shrink::path_token)
            .collect()
    };
    let after = paths(analysis);
    let mut dropped: Vec<String> = paths(notes)
        .into_iter()
        .filter(|p| !after.contains(p))
        .collect();
    // A log paste names the same endpoint on every line, so without this the warning is one
    // path repeated a dozen times and the reader stops reading it.
    dropped.sort();
    dropped.dedup();
    if !dropped.is_empty() {
        out.push(format!("file paths dropped: {}", dropped.join(", ")));
    }

    // Every action item pointing at the same word is the shape of a postmortem that only
    // learned one thing: usually "add more alerting", proposed three ways. The prefixes are
    // asked for in the instruction, so this reads what the model was told to write.
    let kinds: Vec<&str> = ["prevent:", "detect:", "mitigate:"]
        .into_iter()
        .filter(|k| analysis.to_lowercase().contains(k))
        .collect();
    if kinds.len() == 1 {
        out.push(format!(
            "every action item is a '{}' — check nothing else was learned",
            kinds[0].trim_end_matches(':')
        ));
    }

    out
}

/// Quantities the analysis states that the notes never did.
///
/// The check that exists because of what this feature does when it goes wrong. Asked to write
/// up an incident whose notes carry no impact figures, a model will supply them — "342 users
/// affected", "1,204 requests failed" — in the same register as the measured ones, and
/// nothing downstream can tell the two apart. The instruction forbids it; this is what
/// notices when it happens anyway.
///
/// Deliberately narrow, because a false alarm here trains the reader to skip the one warning
/// they must not skip:
///
/// * **Only the digits are compared.** A write-up quotes a figure without quoting the way the
///   notes spaced it: the notes' `8.00 GB` comes back as `8.00GB`, and `4.10 GB` as `4.1GB`.
///   Comparing whole words flagged all four of those as invented, on a document that had
///   invented nothing — which is the failure mode that would retire this warning within a
///   week. So each token is reduced to its numeric core and matched as a substring, making
///   `4.1` ⊂ `4.10` and `1,204` ⊂ `1204` land where they should.
/// * **Two digits minimum.** A lone `5` matches something in any set of notes, so checking it
///   proves nothing either way.
/// * **The list markers [`render`] writes are never candidates** — they are pstore's digits,
///   not the model's.
fn figures_not_in_the_notes(analysis: &str, notes: &str) -> Vec<String> {
    let haystack: Vec<String> = split_figures(notes).filter_map(numeric_core).collect();
    let known = |figure: &str| haystack.iter().any(|t| t.contains(figure));

    let mut out: Vec<String> = Vec::new();
    let mut prescriptive = false;
    for line in analysis.lines() {
        // Action items are the one section whose figures are *meant* to be new. "alert when
        // the miss rate passes 90%" proposes a threshold; it does not claim one was
        // measured, and flagging it is how a warning about invented findings turns into
        // noise the reader learns to skip.
        if let Some(heading) = line.trim().strip_prefix("**") {
            prescriptive = heading.starts_with("Action items");
        }
        if prescriptive {
            continue;
        }
        // Read past the bullet or number this line was laid out with, so pstore's own
        // digits are never candidates.
        let body = without_leading_marker(line.trim()).unwrap_or(line.trim());
        let words: Vec<&str> = split_figures(body).collect();
        for (i, token) in words.iter().enumerate() {
            // A time is checked by `clock_times`, and a date carries digits that the notes
            // write differently everywhere they appear.
            if token.contains(':') {
                continue;
            }
            let Some(figure) = numeric_core(token) else {
                continue;
            };
            if figure.bytes().filter(|b| b.is_ascii_digit()).count() < 2 {
                continue;
            }
            // A duration in whole minutes or hours is the one figure a postmortem is
            // expected to compute rather than quote — "down for 22 minutes", from a timeline
            // the reader can check a few lines further up. Flagging it would fire on every
            // correct write-up, which is the fastest way to make this warning ignorable.
            // Sub-second units are not exempt: `5000ms` is copied from a log line rather than
            // derived, so an invented one is worth catching.
            let unit = words.get(i + 1).map(|w| {
                w.trim_matches(|c: char| !c.is_ascii_alphabetic())
                    .to_lowercase()
            });
            if unit.is_some_and(|u| COARSE_DURATIONS.contains(&u.as_str())) {
                continue;
            }
            if known(&figure) {
                continue;
            }
            // Reported as the document wrote it, not as it was normalised — the reader has to
            // find it on the page.
            let shown = token.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '%');
            if !shown.is_empty() && !out.contains(&shown.to_string()) {
                out.push(shown.to_string());
            }
        }
    }
    out
}

/// Units that make a preceding number a duration the write-up was expected to work out.
const COARSE_DURATIONS: [&str; 8] = [
    "minute", "minutes", "min", "mins", "hour", "hours", "hr", "hrs",
];

/// Split text into candidate figures.
///
/// `/` is a separator here and nowhere else in this module: `100/100 connections` is two
/// readings of the same gauge, and joining them into `100100` would make a figure that
/// appears in neither document.
fn split_figures(text: &str) -> impl Iterator<Item = &str> {
    text.split(|c: char| c.is_whitespace() || matches!(c, '(' | ')' | '/' | ';'))
}

/// The digits and decimal point in `token`, with thousands separators and units removed.
///
/// `None` when there are none — the token is a word, and words are not this check's business.
fn numeric_core(token: &str) -> Option<String> {
    let core: String = token
        .chars()
        .filter(|c| c.is_ascii_digit() || *c == '.')
        .collect();
    let core = core.trim_matches('.').to_string();
    core.bytes().any(|b| b.is_ascii_digit()).then_some(core)
}

/// Clock times appearing in `text`, as written.
///
/// Deliberately only `HH:MM`, with optional seconds: it is the form incident notes and chat
/// logs actually use, and the one a summariser drops. Matching dates as well would flag the
/// header line of every log paste as a missing timeline entry.
fn clock_times(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    // `T` splits so that an ISO timestamp yields its clock part: notes written
    // `[2026-08-07T14:02:15Z]` and a timeline that answers in either format have to compare
    // equal, or every entry reads as a dropped time. The date half loses its colons in the
    // split and is ignored below, which is what should happen to it.
    for token in text.split(|c: char| c.is_whitespace() || matches!(c, '[' | ']' | ',' | 'T')) {
        let t = token.trim_matches(|c: char| !c.is_ascii_digit());
        let mut parts = t.split(':');
        let (Some(h), Some(m)) = (parts.next(), parts.next()) else {
            continue;
        };
        let s = parts.next();
        if parts.next().is_some() {
            continue;
        }
        let digits = |p: &str, n: usize| p.len() == n && p.bytes().all(|b| b.is_ascii_digit());
        if (1..=2).contains(&h.len())
            && h.bytes().all(|b| b.is_ascii_digit())
            && digits(m, 2)
            && s.is_none_or(|s| digits(s, 2))
            && !out.contains(&t.to_string())
        {
            out.push(t.to_string());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rules that make the difference between a postmortem and a plausible story about
    /// an outage. If any of them stops being asked for, the model reverts to supplying the
    /// missing facts itself — which it did, on real telemetry, when the no-invention rule
    /// was one clause in a closing paragraph instead of the first thing said.
    #[test]
    fn the_instruction_forbids_invention_and_blame() {
        let i = INSTRUCTION.to_lowercase();
        assert!(i.contains("must already appear in the notes"));
        assert!(i.contains("an empty field is a correct answer"));
        assert!(i.contains("blamelessly"));
        assert!(i.contains("never individual people"));
        // The anti-invention rule has to come before the fields it governs: buried under
        // them, it is read as a caveat on the last one rather than as the premise.
        let rule = i.find("must already appear in the notes").unwrap();
        assert!(
            rule < i.find("fill in each field").unwrap(),
            "rule at {rule}"
        );
        // Two format failures seen in real output, both now named explicitly.
        assert!(i.contains("never write json"));
        assert!(i.contains("under 200 characters"));
        // Every field the schema requires has to be described, or the model is being asked to
        // fill in something the instruction never explained.
        for (key, _) in FIELDS {
            assert!(i.contains(key), "the instruction must describe {key}");
        }
    }

    #[test]
    fn render_numbers_the_timeline_and_the_action_items() {
        let out = render(
            "Checkout was down for 22 minutes.",
            &[
                ("Impact", vec!["1,400 failed checkouts.".into()]),
                (
                    "Timeline",
                    vec!["09:14 deploy went out.".into(), "09:36 rolled back.".into()],
                ),
                (
                    "Action items",
                    vec!["prevent: gate the migration behind a flag.".into()],
                ),
                ("Open questions", vec!["Why did the canary pass?".into()]),
            ],
        );
        assert_eq!(
            out,
            "**Summary**\nCheckout was down for 22 minutes.\n\n\
             **Impact**\n- 1,400 failed checkouts.\n\n\
             **Timeline**\n1. 09:14 deploy went out.\n2. 09:36 rolled back.\n\n\
             **Action items**\n1. prevent: gate the migration behind a flag.\n\n\
             **Open questions**\n- Why did the canary pass?\n"
        );
    }

    /// The sections where silence is the finding say so, and the ones where it means "there
    /// were none" disappear. Notes that stop mid-incident produce an empty Resolution, and a
    /// reader has to be able to tell that from a document where nobody filled it in — this is
    /// also the only visible sign that the model declined to invent one.
    #[test]
    fn sections_the_notes_did_not_establish_say_so() {
        let out = render(
            "It broke.",
            &[
                ("Impact", vec![]),
                ("Contributing factors", vec![]),
                ("Resolution", vec![]),
                ("Open questions", vec![]),
            ],
        );
        for heading in ["Impact", "Resolution", "Open questions"] {
            assert!(
                out.contains(&format!("**{heading}**\n- {NOT_RECORDED}")),
                "{heading} should state its absence, got {out:?}"
            );
        }
        assert!(
            !out.contains("Contributing factors"),
            "an empty list of aggravating factors means there were none, got {out:?}"
        );
    }

    /// The bug this guards, found by the test above failing: an entry is stripped of the
    /// list marker a model writes for itself, and a timeline entry is *supposed* to open
    /// with digits. Stripping leading digits unconditionally turned `09:14` into `:14`.
    #[test]
    fn a_leading_number_is_kept_and_a_leading_marker_is_not() {
        assert_eq!(entry_text("09:14 deploy went out"), "09:14 deploy went out");
        assert_eq!(
            entry_text("1,400 checkouts failed"),
            "1,400 checkouts failed"
        );
        assert_eq!(
            entry_text("3. detect: alert sooner"),
            "detect: alert sooner"
        );
        assert_eq!(entry_text("- • why did it run?"), "why did it run?");
        // Newlines inside one entry would become list items with no bullet.
        assert_eq!(
            entry_text("first link.\nsecond link."),
            "first link. second link."
        );
        // The shape seen when the notes contain a JSON audit log.
        assert_eq!(
            entry_text(r#"{"timestamp": "13:58:10Z", "description": "policy set to noeviction"}"#),
            "13:58:10Z policy set to noeviction"
        );
    }

    /// The copy button and `pstore rca --actions` both read the list back out of the
    /// rendered document, so this has to survive the numbering [`render`] puts on it and
    /// stop at the next heading rather than swallowing the section after it.
    #[test]
    fn action_items_are_read_back_out_of_the_document() {
        let analysis = render(
            "It broke.",
            &[
                ("Resolution", vec!["Rolled back.".into()]),
                (
                    "Action items",
                    vec![
                        "prevent: gate the migration.".into(),
                        "detect: alert on lock waits.".into(),
                    ],
                ),
                ("Open questions", vec!["Why did the canary pass?".into()]),
            ],
        );
        assert_eq!(
            action_items(&analysis),
            [
                "prevent: gate the migration.",
                "detect: alert on lock waits."
            ]
        );
        // A document with no such section yields nothing rather than the whole thing.
        assert!(action_items("**Summary**\nIt broke.").is_empty());
    }

    #[test]
    fn compose_keeps_the_notes_after_the_instruction() {
        let out = compose("pager fired at 03:02, disk full on db-2");
        assert!(out.starts_with(INSTRUCTION));
        assert!(out.ends_with("pager fired at 03:02, disk full on db-2"));
    }

    #[test]
    fn clock_times_reads_the_forms_notes_are_written_in() {
        let found = clock_times("[03:02] alert fired, 3:07 acked, resolved 03:41:18 — build 12345");
        assert_eq!(found, vec!["03:02", "3:07", "03:41:18"]);
        // A ratio, a version and a bare number are not times.
        assert!(clock_times("ratio 1:400 in v2:3 of 900").is_empty());
    }

    /// Machine-generated notes carry ISO timestamps and the write-up may answer in either
    /// form. Reading only one of them made every entry of a correct timeline look dropped.
    #[test]
    fn an_iso_timestamp_and_a_bare_clock_are_the_same_time() {
        assert_eq!(clock_times("[2026-08-07T14:02:15Z] OOM"), vec!["14:02:15"]);
        let notes = "[2026-08-07T14:02:15Z] redis OOM, [2026-08-07T14:02:30Z] gateway 504";
        let analysis = "**Timeline**\n1. 14:02:15 redis logged OOM.\n\
                        2. 2026-08-07T14:02:30Z gateway returned 504.\n\n\
                        **Action items**\n1. prevent: set a policy.\n2. detect: alert on it.";
        assert!(
            !warnings(analysis, notes)
                .iter()
                .any(|w| w.contains("times dropped")),
            "{:?}",
            warnings(analysis, notes)
        );
    }

    #[test]
    fn a_well_formed_analysis_raises_nothing() {
        let notes = "09:14 deploy of src/db/migrate.rs, 09:20 alerts, 09:36 rollback";
        let analysis = "**Summary**\nCheckout failed for 22 minutes.\n\n\
                        **Timeline**\n1. 09:14 deploy of src/db/migrate.rs.\n\
                        2. 09:20 alerts fired.\n3. 09:36 rolled back.\n\n\
                        **Action items**\n1. prevent: gate it.\n2. detect: alert on the table lock.";
        assert!(
            warnings(analysis, notes).is_empty(),
            "{:?}",
            warnings(analysis, notes)
        );
    }

    /// The failure this check was written for, taken from real output: notes carrying no
    /// impact figures at all, and a postmortem stating three of them in the register of
    /// something measured.
    #[test]
    fn figures_the_notes_never_gave_are_reported() {
        let notes = "09:14 deploy, 09:20 alerts fired on checkout-api, 09:36 rollback";
        let analysis = "**Impact**\n- users affected: 342\n- requests failed: 1,204\n\
                        - refund processing cost $42.30\n\n\
                        **Timeline**\n1. 09:14 deploy.\n\n\
                        **Action items**\n1. prevent: gate it.\n2. detect: alert sooner.";
        let w = warnings(analysis, notes);
        let flagged = w
            .iter()
            .find(|w| w.starts_with("figures that do not appear"))
            .unwrap_or_else(|| panic!("nothing flagged, got {w:?}"));
        for invented in ["342", "1,204", "42.30"] {
            assert!(flagged.contains(invented), "{invented} missed: {flagged}");
        }
    }

    /// An action item proposes a threshold; it does not claim one was measured. Reading the
    /// two the same way is what makes a warning about invented findings unreadable.
    #[test]
    fn figures_proposed_by_an_action_item_are_not_reported() {
        let notes = "14:02 cache_miss_rate 98.4%, connections 100/100";
        let analysis = "**Timeline**\n1. 14:02 cache_miss_rate hit 98.4%.\n\n\
                        **Action items**\n1. detect: alert when the miss rate passes 90%.\n\
                        2. mitigate: raise max_connections from 100 to 250.";
        assert!(
            figures_not_in_the_notes(analysis, notes).is_empty(),
            "{:?}",
            figures_not_in_the_notes(analysis, notes)
        );
    }

    /// The check has to survive its own document. pstore numbers the list itself, the notes
    /// write figures with their own separators, and a duration is worked out rather than
    /// quoted — a warning that fires on all three is one nobody reads.
    #[test]
    fn figures_taken_from_the_notes_are_not_reported() {
        let notes = "14:02 redis-primary-01 at 8.00 GB, 100/100 connections, 98.4% miss rate, \
                     query timeout after 5000ms";
        let analysis = "**Summary**\nCheckout was down for 22 minutes.\n\n\
                        **Impact**\n- 98.4% cache miss rate\n- 100/100 connections in use\n\n\
                        **Timeline**\n1. 14:02 redis-primary-01 reached 8.00 GB.\n\
                        2. Query timeout after 5000ms.\n\n\
                        **Action items**\n1. prevent: set an eviction policy.\n\
                        2. detect: alert on memory use.";
        assert!(
            figures_not_in_the_notes(analysis, notes).is_empty(),
            "{:?}",
            figures_not_in_the_notes(analysis, notes)
        );
    }

    /// The false positive that would have retired this warning, caught on real telemetry: a
    /// write-up quotes a figure without quoting the whitespace around it. Every one of these
    /// is in the notes, and flagging them made a document that had invented nothing look as
    /// though it had invented four things.
    #[test]
    fn a_figure_respaced_by_the_write_up_is_still_the_notes_figure() {
        let notes = "redis-primary-01 memory_used_bytes 8.00 GB, baseline 4.10 GB, \
                     pg_active_connections 100 / 100, cache_miss_rate 98.4 %";
        let analysis = "**Timeline**\n1. Memory rose to 8.00GB against a 4.1GB baseline.\n\
                        2. Connections reached 100/100 and the miss rate hit 98.4%.\n\n\
                        **Action items**\n1. prevent: set a policy.\n2. detect: alert on it.";
        assert!(
            figures_not_in_the_notes(analysis, notes).is_empty(),
            "{:?}",
            figures_not_in_the_notes(analysis, notes)
        );
    }

    /// A timeline that quietly loses the moment the alert fired still reads as a timeline.
    #[test]
    fn times_the_analysis_dropped_are_reported() {
        let notes = "09:14 deploy, 09:20 alerts fired, 09:36 rollback";
        let analysis = "**Timeline**\n1. 09:14 deploy.\n2. 09:36 rollback.\n\n\
                        **Action items**\n1. prevent: gate it.\n2. detect: alert sooner.";
        let w = warnings(analysis, notes);
        assert!(w.iter().any(|w| w.contains("09:20")), "got {w:?}");
    }

    #[test]
    fn notes_without_times_are_flagged_as_an_unverifiable_order() {
        let w = warnings(
            "**Timeline**\n1. it broke.",
            "the service broke and we fixed it",
        );
        assert!(w.iter().any(|w| w.contains("no times")), "got {w:?}");
    }

    #[test]
    fn dropped_file_paths_are_reported() {
        let notes = "03:00 src/db/migrate.rs and src/db/pool.rs both changed";
        let analysis = "**Timeline**\n1. 03:00 src/db/migrate.rs changed.\n\n\
                        **Action items**\n1. prevent: review it.\n2. detect: alert on it.";
        let w = warnings(analysis, notes);
        assert!(w.iter().any(|w| w.contains("src/db/pool.rs")), "got {w:?}");
    }

    /// Three ways to say "add an alert" is one lesson, not three, and the reader should be
    /// told that before the tickets are cut.
    #[test]
    fn action_items_that_all_buy_the_same_thing_are_flagged() {
        let notes = "03:00 disk filled on db-2";
        let analysis = "**Timeline**\n1. 03:00 disk filled on db-2.\n\n\
                        **Action items**\n1. detect: alert on disk use.\n\
                        2. detect: alert on inode use.\n3. detect: page on the growth rate.";
        let w = warnings(analysis, notes);
        assert!(w.iter().any(|w| w.contains("'detect'")), "got {w:?}");
    }
}
