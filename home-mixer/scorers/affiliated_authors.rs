//! Accounts affiliated with X, xAI, Tesla and SpaceX, identified by the affiliate badge their
//! account carries.
//!
//! The badge is the parent organization's own statement that an account belongs to it, granted
//! and revoked through Verified Organizations. That is what makes it worth keying on: someone who
//! leaves loses the badge and stops being deranked without anyone editing this file. A list of
//! handles could not do that — it was right the day it was written and wrong from then on.
//!
//! What the badge is not is a roster of everyone who works there. It is applied at the
//! organization's discretion; the organizations' own accounts carry a business badge rather than
//! an affiliate badge pointing at themselves, and the people who run them typically carry no
//! badge at all. Those accounts are not covered here, deliberately: this file states one rule
//! from one source, and the accounts outside it are outside it.
//!
//! The organization list below is the target set, not a roster of people — four handles that
//! change only if the companies do.

use crate::models::candidate::PostCandidate;

/// Organizations whose affiliates are deranked. Matched against the handle the badge points at.
const AFFILIATED_ORGANIZATIONS: [&str; 4] = ["x", "xai", "tesla", "spacex"];

fn normalize(handle: &str) -> String {
    handle.trim().trim_start_matches('@').to_ascii_lowercase()
}

/// The organization on the list this badge points at, or `None` for a badge from anyone else.
pub fn affiliation_of_handle(parent_handle: &str) -> Option<&'static str> {
    let normalized = normalize(parent_handle);
    AFFILIATED_ORGANIZATIONS
        .into_iter()
        .find(|organization| *organization == normalized)
}

/// The badge a post carries, whether the badged account wrote it or the post is a repost of one.
/// A repost carries the same content, so it carries the same affiliation.
pub fn badge_of(candidate: &PostCandidate) -> Option<&str> {
    [
        candidate.author_affiliate_handle.as_deref(),
        candidate.retweeted_affiliate_handle.as_deref(),
    ]
    .into_iter()
    .flatten()
    .next()
}

pub fn affiliation_of(candidate: &PostCandidate) -> Option<&'static str> {
    [
        candidate.author_affiliate_handle.as_deref(),
        candidate.retweeted_affiliate_handle.as_deref(),
    ]
    .into_iter()
    .flatten()
    .find_map(affiliation_of_handle)
}

pub fn is_affiliated(candidate: &PostCandidate) -> bool {
    affiliation_of(candidate).is_some()
}

#[cfg(test)]
mod tests {
    use super::{
        AFFILIATED_ORGANIZATIONS, affiliation_of, affiliation_of_handle, badge_of, is_affiliated,
        normalize,
    };
    use crate::models::candidate::PostCandidate;

    fn badged(parent_handle: &str) -> PostCandidate {
        PostCandidate {
            author_affiliate_handle: Some(parent_handle.to_string()),
            ..Default::default()
        }
    }

    #[test]
    fn every_organization_is_already_normalized() {
        // An "@Tesla" or "Tesla" entry would match nothing, silently.
        for organization in AFFILIATED_ORGANIZATIONS {
            assert_eq!(organization, normalize(organization), "{organization}");
        }
    }

    #[test]
    fn a_badge_from_a_listed_organization_matches() {
        assert_eq!(affiliation_of_handle("Tesla"), Some("tesla"));
        assert_eq!(affiliation_of_handle(" @xAI "), Some("xai"));
    }

    #[test]
    fn a_badge_from_anyone_else_does_not() {
        for parent in ["nasa", "teslarati", "boeing", "xaicorp", ""] {
            assert_eq!(affiliation_of_handle(parent), None, "{parent}");
        }
    }

    #[test]
    fn an_affiliate_of_a_listed_organization_is_matched() {
        assert!(is_affiliated(&badged("SpaceX")));
    }

    #[test]
    fn a_repost_of_an_affiliate_carries_the_affiliation() {
        let repost = PostCandidate {
            author_affiliate_handle: None,
            retweeted_affiliate_handle: Some("tesla".to_string()),
            ..Default::default()
        };

        assert_eq!(affiliation_of(&repost), Some("tesla"));
    }

    #[test]
    fn an_account_with_no_badge_is_untouched() {
        assert!(!is_affiliated(&PostCandidate::default()));
        assert_eq!(badge_of(&PostCandidate::default()), None);
    }

    #[test]
    fn a_badge_from_elsewhere_is_still_visible_to_the_metric() {
        // Reported as an observed badge that matched no organization, which is how a payload
        // arriving under a different field name is told apart from one that never arrives.
        let candidate = badged("nasa");

        assert_eq!(badge_of(&candidate), Some("nasa"));
        assert_eq!(affiliation_of(&candidate), None);
    }
}
