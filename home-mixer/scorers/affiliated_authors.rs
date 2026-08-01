//! Accounts affiliated with X, xAI, Tesla and SpaceX.
//!
//! This is an operator editorial list, not a measured signal, so it lives in its own file under
//! its own name: one roster that can be read and changed in one place, rather than a set of
//! special cases buried inside a scorer.
//!
//! The roster is split in two on purpose. Company-run accounts are stable — the handle belongs to
//! the company, so an entry stays correct for as long as the account exists. Individual accounts
//! are not: people change employers, and an entry that was right when it was added quietly becomes
//! a person being deranked for a job they no longer have. That half needs pruning, and keeping it
//! separate is what makes the maintenance visible.
//!
//! Matching is exact on the normalized handle — case-insensitive, ignoring a leading '@' and
//! surrounding whitespace. Substrings never match, so lookalike handles are unaffected. Lookup
//! fails open when screen name hydration is missing: a degraded Gizmoduck hydration costs
//! precision here rather than deranking posts that never matched the roster.

use crate::models::candidate::PostCandidate;

/// Handles the companies operate themselves.
const COMPANY_ACCOUNTS: [(&str, &str); 14] = [
    ("x", "X"),
    ("xdevelopers", "X"),
    ("xeng", "X"),
    ("xdesign", "X"),
    ("safety", "X"),
    ("support", "X"),
    ("premium", "X"),
    ("xai", "xAI"),
    ("grok", "xAI"),
    ("tesla", "Tesla"),
    ("teslaai", "Tesla"),
    ("tesla_optimus", "Tesla"),
    ("teslaenergy", "Tesla"),
    ("spacex", "SpaceX"),
];

/// Individual accounts. Seeded only with people whose affiliation they state publicly themselves;
/// extend it as needed, and delete an entry the day the person leaves.
const AFFILIATED_PEOPLE: [(&str, &str); 4] = [
    ("elonmusk", "X, xAI, Tesla, SpaceX"),
    ("aelluswamy", "Tesla"),
    ("thegregyang", "xAI"),
    ("tobyphln", "xAI"),
];

fn normalize(screen_name: &str) -> String {
    screen_name
        .trim()
        .trim_start_matches('@')
        .to_ascii_lowercase()
}

/// The affiliation on record for a handle, or `None` for everyone else.
pub fn affiliation_of_screen_name(screen_name: &str) -> Option<&'static str> {
    let normalized = normalize(screen_name);
    COMPANY_ACCOUNTS
        .iter()
        .chain(AFFILIATED_PEOPLE.iter())
        .find(|(handle, _)| *handle == normalized)
        .map(|(_, affiliation)| *affiliation)
}

/// The affiliation carried by a post, whether the affiliated account wrote it or the post is a
/// repost of one. A repost carries the same content, so it carries the same affiliation.
pub fn affiliation_of(candidate: &PostCandidate) -> Option<&'static str> {
    [
        candidate.author_screen_name.as_deref(),
        candidate.retweeted_screen_name.as_deref(),
    ]
    .into_iter()
    .flatten()
    .find_map(affiliation_of_screen_name)
}

pub fn is_affiliated(candidate: &PostCandidate) -> bool {
    affiliation_of(candidate).is_some()
}

#[cfg(test)]
mod tests {
    use super::{
        AFFILIATED_PEOPLE, COMPANY_ACCOUNTS, affiliation_of, affiliation_of_screen_name,
        is_affiliated, normalize,
    };
    use crate::models::candidate::PostCandidate;

    fn candidate(author_screen_name: &str) -> PostCandidate {
        PostCandidate {
            author_screen_name: Some(author_screen_name.to_string()),
            ..Default::default()
        }
    }

    #[test]
    fn every_roster_entry_is_already_normalized() {
        // A "@Tesla" or "Tesla" entry would match nothing, silently. Catch that at the source
        // rather than at a lookup that never fires.
        for (handle, _) in COMPANY_ACCOUNTS.iter().chain(AFFILIATED_PEOPLE.iter()) {
            assert_eq!(*handle, normalize(handle), "{handle} is not normalized");
        }
    }

    #[test]
    fn the_roster_has_no_duplicate_handles() {
        let mut handles: Vec<&str> = COMPANY_ACCOUNTS
            .iter()
            .chain(AFFILIATED_PEOPLE.iter())
            .map(|(handle, _)| *handle)
            .collect();
        let total = handles.len();
        handles.sort_unstable();
        handles.dedup();

        assert_eq!(handles.len(), total);
    }

    #[test]
    fn matches_company_accounts_case_insensitively() {
        assert_eq!(affiliation_of_screen_name("SpaceX"), Some("SpaceX"));
        assert_eq!(affiliation_of_screen_name(" @xAI "), Some("xAI"));
    }

    #[test]
    fn matches_individual_accounts() {
        assert_eq!(
            affiliation_of_screen_name("elonmusk"),
            Some("X, xAI, Tesla, SpaceX")
        );
    }

    #[test]
    fn lookalike_handles_are_untouched() {
        for handle in ["teslarati", "notspacex", "xaidaily", "elonmuskfan", "xx"] {
            assert_eq!(affiliation_of_screen_name(handle), None, "{handle}");
        }
    }

    #[test]
    fn reposts_of_affiliated_accounts_carry_the_affiliation() {
        let repost = PostCandidate {
            author_screen_name: Some("someoneelse".to_string()),
            retweeted_screen_name: Some("@Tesla".to_string()),
            ..Default::default()
        };

        assert_eq!(affiliation_of(&repost), Some("Tesla"));
    }

    #[test]
    fn unaffiliated_authors_are_not_matched() {
        assert!(!is_affiliated(&candidate("cascadiawire")));
    }

    #[test]
    fn missing_screen_name_hydration_fails_open() {
        assert!(!is_affiliated(&PostCandidate::default()));
    }
}
