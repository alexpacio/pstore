//! Structured identifiers, found by pattern and confirmed by arithmetic.
//!
//! The neural tagger ([`super::model`]) is what finds names, addresses and organisations —
//! things only context can identify. This module handles the other kind: identifiers with
//! a *checksum*. An IBAN, a credit card, an Italian codice fiscale or VAT number either
//! satisfies its check digit or it does not, and that answer does not depend on a model
//! being downloaded, on a threshold, or on the surrounding language.
//!
//! Two consequences shape the code below:
//!
//! * **Arithmetic overrides the model.** A validated match is reported with full
//!   confidence and wins any overlap, because "these 27 characters are a valid IBAN" is a
//!   stronger statement than "the tagger thinks this looks like an IBAN".
//! * **A pattern alone is never enough.** `\b\d{11}\b` matches a great many things that
//!   are not VAT numbers — timestamps, ids, phone numbers — so nothing is reported unless
//!   its check digit agrees. In a prompt full of code that difference is the difference
//!   between a useful tool and one that mangles the input.

use std::sync::OnceLock;

use regex::Regex;

use super::{Finder, Finding};

/// Compile once, on first use. A malformed pattern here is a bug, not a runtime
/// condition, so panicking on it is the honest response — and the unit tests below reach
/// every pattern, so it cannot survive a test run.
fn re(cell: &'static OnceLock<Regex>, pattern: &'static str) -> &'static Regex {
    cell.get_or_init(|| Regex::new(pattern).expect("built-in PII pattern must compile"))
}

/// Everything the deterministic pass can find, in document order.
pub fn scan(text: &str) -> Vec<Finding> {
    let mut out = Vec::new();
    out.extend(emails(text));
    out.extend(urls(text));
    out.extend(ibans(text));
    out.extend(credit_cards(text));
    out.extend(codici_fiscali(text));
    out.extend(partite_iva(text));
    out.sort_by_key(|f| (f.start, f.end));
    out
}

fn found(tag: &str, text: &str, start: usize, end: usize) -> Finding {
    Finding {
        tag: tag.to_string(),
        start,
        end,
        text: text[start..end].to_string(),
        // A pattern that had to satisfy a checksum is not a guess.
        score: 1.0,
        finder: Finder::Pattern,
    }
}

/// Email addresses. Deliberately not RFC-complete: the goal is the shapes people write.
pub fn emails(text: &str) -> Vec<Finding> {
    static RE: OnceLock<Regex> = OnceLock::new();
    re(
        &RE,
        r"(?i)\b[a-z0-9._%+\-]+@[a-z0-9\-]+(?:\.[a-z0-9\-]+)*\.[a-z]{2,24}\b",
    )
    .find_iter(text)
    .map(|m| found("EMAIL", text, m.start(), m.end()))
    .collect()
}

/// `http(s)` URLs. Reported, but masked only if the user asks — see
/// [`super::tag_masked_by_default`].
pub fn urls(text: &str) -> Vec<Finding> {
    static RE: OnceLock<Regex> = OnceLock::new();
    re(&RE, r"(?i)\bhttps?://[^\s<>()\[\]{}'\x22]+")
        .find_iter(text)
        // Trailing sentence punctuation is not part of the URL.
        .map(|m| {
            let trimmed = m.as_str().trim_end_matches(['.', ',', ';', ':', '!', '?']);
            found("URL", text, m.start(), m.start() + trimmed.len())
        })
        .collect()
}

/// IBANs, confirmed by the ISO 7064 mod-97 check.
pub fn ibans(text: &str) -> Vec<Finding> {
    static RE: OnceLock<Regex> = OnceLock::new();
    re(&RE, r"\b([A-Z]{2}[0-9]{2}(?:[ ]?[A-Z0-9]{2,4}){2,8})\b")
        .find_iter(text)
        .filter(|m| iban_is_valid(m.as_str()))
        .map(|m| found("IBAN", text, m.start(), m.end()))
        .collect()
}

/// Card numbers, confirmed by the Luhn check.
pub fn credit_cards(text: &str) -> Vec<Finding> {
    static RE: OnceLock<Regex> = OnceLock::new();
    re(&RE, r"\b(?:[0-9]{4}[ \-]?){3}[0-9]{1,7}\b")
        .find_iter(text)
        .filter(|m| {
            let digits = digits_of(m.as_str());
            (13..=19).contains(&digits.len()) && luhn_is_valid(&digits)
        })
        .map(|m| found("CREDITCARDNUMBER", text, m.start(), m.end()))
        .collect()
}

/// Italian tax codes, confirmed by their check letter.
pub fn codici_fiscali(text: &str) -> Vec<Finding> {
    static RE: OnceLock<Regex> = OnceLock::new();
    re(
        &RE,
        r"\b[A-Za-z]{6}[0-9LMNPQRSTUVlmnpqrstuv]{2}[A-EHLMPRSTabehlmprst][0-9LMNPQRSTUVlmnpqrstuv]{2}[A-Za-z][0-9LMNPQRSTUVlmnpqrstuv]{3}[A-Za-z]\b",
    )
    .find_iter(text)
    .filter(|m| codice_fiscale_is_valid(m.as_str()))
    .map(|m| found("CF", text, m.start(), m.end()))
    .collect()
}

/// Italian VAT numbers, confirmed by their check digit.
pub fn partite_iva(text: &str) -> Vec<Finding> {
    static RE: OnceLock<Regex> = OnceLock::new();
    re(&RE, r"\b(?:IT[ ]?)?([0-9]{11})\b")
        .find_iter(text)
        .filter(|m| partita_iva_is_valid(m.as_str()))
        .map(|m| found("PIVA", text, m.start(), m.end()))
        .collect()
}

/// Just the digits of `s`.
fn digits_of(s: &str) -> Vec<u32> {
    s.chars().filter_map(|c| c.to_digit(10)).collect()
}

/// The Luhn check, over digits already extracted.
pub fn luhn_is_valid(digits: &[u32]) -> bool {
    if digits.len() < 2 {
        return false;
    }
    let sum: u32 = digits
        .iter()
        .rev()
        .enumerate()
        .map(|(i, d)| {
            if i % 2 == 1 {
                let doubled = d * 2;
                if doubled > 9 { doubled - 9 } else { doubled }
            } else {
                *d
            }
        })
        .sum();
    sum.is_multiple_of(10)
}

/// ISO 7064 mod-97-10, as IBANs use it.
pub fn iban_is_valid(candidate: &str) -> bool {
    let compact: String = candidate
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_uppercase())
        .collect();
    // Shortest is Norway at 15, longest defined is 34.
    if !(15..=34).contains(&compact.len()) {
        return false;
    }
    let (head, tail) = compact.split_at(4);
    if !head[..2].chars().all(|c| c.is_ascii_alphabetic())
        || !head[2..].chars().all(|c| c.is_ascii_digit())
    {
        return false;
    }

    // Rearranged, letters expanded to two digits, then mod 97 — computed a digit at a
    // time so no number ever needs more than 32 bits.
    let mut remainder: u32 = 0;
    for c in tail.chars().chain(head.chars()) {
        let value = if c.is_ascii_digit() {
            c as u32 - '0' as u32
        } else {
            c as u32 - 'A' as u32 + 10
        };
        remainder = if value > 9 {
            (remainder * 100 + value) % 97
        } else {
            (remainder * 10 + value) % 97
        };
    }
    remainder == 1
}

/// The odd-position table of the codice fiscale check character.
const CF_ODD: [u32; 36] = [
    1, 0, 5, 7, 9, 13, 15, 17, 19, 21, // 0-9
    1, 0, 5, 7, 9, 13, 15, 17, 19, 21, // A-J
    2, 4, 18, 20, 11, 3, 6, 8, 12, 14, // K-T
    16, 10, 22, 25, 24, 23, // U-Z
];

/// Validate an Italian codice fiscale by recomputing its final check letter.
pub fn codice_fiscale_is_valid(candidate: &str) -> bool {
    let up: Vec<char> = candidate.trim().to_ascii_uppercase().chars().collect();
    if up.len() != 16 || !up.iter().all(|c| c.is_ascii_alphanumeric()) {
        return false;
    }

    // Positions are 1-based in the specification: odd positions use CF_ODD, even ones
    // take the character's plain value.
    let value = |c: char| -> Option<u32> {
        if c.is_ascii_digit() {
            Some(c as u32 - '0' as u32)
        } else {
            Some(c as u32 - 'A' as u32 + 10)
        }
    };
    let mut sum = 0u32;
    for (i, c) in up[..15].iter().enumerate() {
        let v = match value(*c) {
            Some(v) if (v as usize) < CF_ODD.len() => v,
            _ => return false,
        };
        sum += if i % 2 == 0 {
            CF_ODD[v as usize]
        } else if v > 9 {
            v - 10
        } else {
            v
        };
    }
    let expected = (b'A' + (sum % 26) as u8) as char;
    up[15] == expected
}

/// Validate an Italian VAT number by recomputing its check digit.
pub fn partita_iva_is_valid(candidate: &str) -> bool {
    let digits = digits_of(candidate);
    if digits.len() != 11 {
        return false;
    }
    // Odd positions (1-based) count as themselves; even ones are doubled with nine
    // subtracted past ten. Same shape as Luhn, anchored from the left.
    let sum: u32 = digits[..10]
        .iter()
        .enumerate()
        .map(|(i, d)| {
            if i % 2 == 0 {
                *d
            } else {
                let doubled = d * 2;
                if doubled > 9 { doubled - 9 } else { doubled }
            }
        })
        .sum();
    let check = (10 - (sum % 10)) % 10;
    digits[10] == check
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_email_addresses_but_not_bare_words() {
        let f = emails("write to mario.rossi+legal@studio-rossi.it about it");
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].text, "mario.rossi+legal@studio-rossi.it");
        assert_eq!(f[0].tag, "EMAIL");
        assert!(emails("no address here @ all").is_empty());
        // Offsets must index the original string.
        let text = "ciao mario@example.com";
        let f = emails(text);
        assert_eq!(&text[f[0].start..f[0].end], "mario@example.com");
    }

    #[test]
    fn finds_urls_without_swallowing_punctuation() {
        let f = urls("see https://example.com/docs/page.html, then stop");
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].text, "https://example.com/docs/page.html");
        // Markdown links are common in prompts and must not lose their closing bracket.
        let f = urls("[docs](https://example.com/a)");
        assert_eq!(f[0].text, "https://example.com/a");
    }

    #[test]
    fn validates_real_ibans_and_rejects_mistyped_ones() {
        // Published example IBANs.
        assert!(iban_is_valid("IT60X0542811101000000123456"));
        assert!(iban_is_valid("GB82 WEST 1234 5698 7654 32"));
        assert!(iban_is_valid("DE89370400440532013000"));
        // One digit changed: the check must fail.
        assert!(!iban_is_valid("IT60X0542811101000000123457"));
        assert!(!iban_is_valid("GB82 WEST 1234 5698 7654 33"));
        // Too short, and not an IBAN shape at all.
        assert!(!iban_is_valid("IT60X05428"));
        assert!(!iban_is_valid("1234567890123456"));
    }

    #[test]
    fn scans_ibans_with_and_without_spacing() {
        let text = "bonifico su IT60X0542811101000000123456 entro venerdì";
        let f = ibans(text);
        assert_eq!(f.len(), 1, "got {f:?}");
        assert_eq!(f[0].tag, "IBAN");
        assert_eq!(&text[f[0].start..f[0].end], "IT60X0542811101000000123456");
        assert_eq!(f[0].score, 1.0, "a checksum match is not a guess");

        let spaced = "IBAN GB82 WEST 1234 5698 7654 32 grazie";
        let f = ibans(spaced);
        assert_eq!(f.len(), 1, "got {f:?}");
        assert_eq!(f[0].text, "GB82 WEST 1234 5698 7654 32");
    }

    #[test]
    fn validates_cards_by_luhn_only() {
        // Standard test numbers.
        assert!(luhn_is_valid(&digits_of("4111111111111111")));
        assert!(luhn_is_valid(&digits_of("5500 0000 0000 0004")));
        assert!(!luhn_is_valid(&digits_of("4111111111111112")));

        let text = "card 4111 1111 1111 1111 expires soon";
        let f = credit_cards(text);
        assert_eq!(f.len(), 1, "got {f:?}");
        assert_eq!(f[0].tag, "CREDITCARDNUMBER");

        // A 16-digit number that fails Luhn is not reported at all.
        assert!(credit_cards("id 4111 1111 1111 1112 here").is_empty());
    }

    #[test]
    fn validates_codici_fiscali() {
        // Known-good examples.
        assert!(codice_fiscale_is_valid("RSSMRA85T10A562S"));
        assert!(
            codice_fiscale_is_valid("rssmra85t10a562s"),
            "case-insensitive"
        );
        // Wrong check letter.
        assert!(!codice_fiscale_is_valid("RSSMRA85T10A562T"));
        // Wrong length.
        assert!(!codice_fiscale_is_valid("RSSMRA85T10A562"));

        let text = "il CF è RSSMRA85T10A562S, grazie";
        let f = codici_fiscali(text);
        assert_eq!(f.len(), 1, "got {f:?}");
        assert_eq!(f[0].tag, "CF");
        assert_eq!(f[0].text, "RSSMRA85T10A562S");
    }

    #[test]
    fn validates_partite_iva_and_ignores_lookalike_numbers() {
        assert!(partita_iva_is_valid("00743110157"));
        assert!(partita_iva_is_valid("IT00743110157"));
        assert!(!partita_iva_is_valid("00743110158"));

        // An eleven-digit timestamp is not a VAT number, and must be left alone.
        assert!(partite_iva("epoch 17356789012 seconds").is_empty());
        let f = partite_iva("P.IVA 00743110157");
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].tag, "PIVA");
    }

    #[test]
    fn scan_returns_everything_in_document_order() {
        let text = "Mario (mario@example.com, CF RSSMRA85T10A562S) \
                    paga con IT60X0542811101000000123456";
        let found = scan(text);
        let tags: Vec<&str> = found.iter().map(|f| f.tag.as_str()).collect();
        assert_eq!(tags, vec!["EMAIL", "CF", "IBAN"], "got {found:?}");
        assert!(
            found.windows(2).all(|w| w[0].start <= w[1].start),
            "findings must be sorted"
        );
        for f in &found {
            assert_eq!(&text[f.start..f.end], f.text, "offsets must match the text");
            assert_eq!(f.finder, Finder::Pattern);
        }
    }

    #[test]
    fn a_prompt_full_of_code_is_left_alone() {
        // The failure mode that matters: false positives in ordinary technical prose.
        let code = "Refactor src/main.rs:42 to use u128 counters, bump 1.2.3 to 1.3.0, \
                    keep the 0x1234567890abcdef mask and the 12345678901 row id. \
                    Version 4111111111111112 is not a card.";
        let found = scan(code);
        assert!(found.is_empty(), "false positives: {found:?}");
    }

    #[test]
    fn multibyte_text_offsets_stay_on_char_boundaries() {
        let text = "Società Rossi — scrivi a mario@example.com però";
        let found = scan(text);
        assert_eq!(found.len(), 1);
        // Slicing at a bad offset would panic here.
        assert_eq!(&text[found[0].start..found[0].end], "mario@example.com");
    }
}
