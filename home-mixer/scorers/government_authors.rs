//! Government accounts subject to feed-policy downranking.
//!
//! "Domestic" means the United States. The current Trump administration is an explicit exception
//! to that domestic-government exemption. Foreign-government coverage is an auditable exact-match
//! roster of national executive, head-of-government and foreign-affairs accounts; it is seeded
//! across the non-U.S. G20 and additional high-volume governments. State-controlled media is
//! deliberately excluded.
//!
//! Both the visible account handle and the parent handle from an affiliate badge are checked.
//! Reposts are checked too because they carry the same content. Missing hydration and accounts not
//! on a roster fail open. This favors false negatives over silently treating political discussion,
//! journalism, parody or a similarly named account as a government.
//!
//! Role accounts need review when an administration changes. Personal accounts need review when an
//! officeholder changes. Keeping both rosters here makes that maintenance visible.

use crate::models::candidate::PostCandidate;

const TRUMP_ADMINISTRATION_HANDLES: &[&str] = &[
    "whitehouse",
    "potus",
    "realdonaldtrump",
    "rapidresponse47",
    "vp",
    "jdvance",
    "presssec",
    "flotus",
    "melaniatrump",
    "firstladyoffice",
    "slotus",
];

const FOREIGN_GOVERNMENT_HANDLES: &[&str] = &[
    "oprargentina",
    "jmilei",
    "ausgov",
    "albomp",
    "govbrazil",
    "lulaoficial",
    "canadianpm",
    "markjcarney",
    "mfa_china",
    "elysee",
    "emmanuelmacron",
    "bundeskanzler",
    "germanydiplo",
    "pmoindia",
    "narendramodi",
    "setkabgoid",
    "prabowo",
    "palazzo_chigi",
    "giorgiameloni",
    "jpn_pmo",
    "mofajapan_en",
    "gobiernomx",
    "claudiashein",
    "kremlinrussia_e",
    "mfa_russia",
    "ksamofaen",
    "presidencyza",
    "governmentza",
    "president_kr",
    "tcbestepe",
    "rterdogan",
    "mfaturkiye",
    "10downingstreet",
    "keir_starmer",
    "fcdogovuk",
    "eu_commission",
    "eucouncil",
    "vonderleyen",
    "zelenskyyua",
    "ukraine",
    "mfa_ukraine",
    "israel",
    "israelmfa",
    "netanyahu",
    "khamenei_ir",
    "irimfa_en",
    "taiwanpresident",
    "mofa_taiwan",
];

fn normalize(handle: &str) -> String {
    handle.trim().trim_start_matches('@').to_ascii_lowercase()
}

fn is_listed(handle: &str, roster: &[&str]) -> bool {
    let normalized = normalize(handle);
    roster.contains(&normalized.as_str())
}

fn screen_names(candidate: &PostCandidate) -> impl Iterator<Item = &str> {
    [
        candidate.author_screen_name.as_deref(),
        candidate.retweeted_screen_name.as_deref(),
    ]
    .into_iter()
    .flatten()
}

fn affiliate_handles(candidate: &PostCandidate) -> impl Iterator<Item = &str> {
    [
        candidate.author_affiliate_handle.as_deref(),
        candidate.retweeted_affiliate_handle.as_deref(),
    ]
    .into_iter()
    .flatten()
}

fn matches_roster(candidate: &PostCandidate, roster: &[&str]) -> bool {
    screen_names(candidate)
        .chain(affiliate_handles(candidate))
        .any(|handle| is_listed(handle, roster))
}

pub fn is_trump_administration(candidate: &PostCandidate) -> bool {
    matches_roster(candidate, TRUMP_ADMINISTRATION_HANDLES)
}

pub fn is_foreign_government(candidate: &PostCandidate) -> bool {
    matches_roster(candidate, FOREIGN_GOVERNMENT_HANDLES)
}

#[cfg(test)]
mod tests {
    use super::{
        FOREIGN_GOVERNMENT_HANDLES, TRUMP_ADMINISTRATION_HANDLES, is_foreign_government, is_listed,
        is_trump_administration, normalize,
    };
    use crate::models::candidate::PostCandidate;

    fn authored_by(screen_name: &str) -> PostCandidate {
        PostCandidate {
            author_screen_name: Some(screen_name.to_string()),
            ..Default::default()
        }
    }

    #[test]
    fn every_roster_entry_is_normalized_and_unique() {
        let mut all: Vec<&str> = TRUMP_ADMINISTRATION_HANDLES
            .iter()
            .chain(FOREIGN_GOVERNMENT_HANDLES)
            .copied()
            .collect();

        for handle in &all {
            assert_eq!(*handle, normalize(handle), "{handle}");
        }
        all.sort_unstable();
        assert!(
            all.windows(2).all(|pair| pair[0] != pair[1]),
            "government rosters must not overlap or contain duplicates"
        );
    }

    #[test]
    fn current_trump_administration_accounts_match_exactly() {
        assert!(is_trump_administration(&authored_by("@WhiteHouse")));
        assert!(is_trump_administration(&authored_by("POTUS")));
        assert!(is_trump_administration(&authored_by("RapidResponse47")));
        assert!(!is_trump_administration(&authored_by("WhiteHouse45")));
    }

    #[test]
    fn foreign_government_accounts_match_exactly() {
        assert!(is_foreign_government(&authored_by("10DowningStreet")));
        assert!(is_foreign_government(&authored_by("@CanadianPM")));
        assert!(is_foreign_government(&authored_by("MFA_China")));
        assert!(!is_foreign_government(&authored_by("CanadianPMFan")));
    }

    #[test]
    fn reposts_carry_the_government_classification() {
        let trump_repost = PostCandidate {
            retweeted_screen_name: Some("PressSec".to_string()),
            ..Default::default()
        };
        let foreign_repost = PostCandidate {
            retweeted_screen_name: Some("Elysee".to_string()),
            ..Default::default()
        };

        assert!(is_trump_administration(&trump_repost));
        assert!(is_foreign_government(&foreign_repost));
    }

    #[test]
    fn affiliate_badges_cover_government_staff_accounts() {
        let foreign_affiliate = PostCandidate {
            author_screen_name: Some("a_diplomat".to_string()),
            author_affiliate_handle: Some("10DowningStreet".to_string()),
            ..Default::default()
        };

        assert!(is_foreign_government(&foreign_affiliate));
    }

    #[test]
    fn other_us_government_and_state_media_are_out_of_scope() {
        for handle in ["StateDept", "SenateFloor", "RT_com", "CGTNOfficial"] {
            let candidate = authored_by(handle);
            assert!(!is_trump_administration(&candidate), "{handle}");
            assert!(!is_foreign_government(&candidate), "{handle}");
        }
    }

    #[test]
    fn political_text_does_not_classify_an_unlisted_author() {
        let candidate = PostCandidate {
            author_screen_name: Some("localreporter".to_string()),
            tweet_text: "A report on the Trump administration and several foreign governments"
                .to_string(),
            ..Default::default()
        };

        assert!(!is_trump_administration(&candidate));
        assert!(!is_foreign_government(&candidate));
    }

    #[test]
    fn normalization_is_case_insensitive_and_ignores_at_prefix() {
        assert!(is_listed(" @POTUS ", TRUMP_ADMINISTRATION_HANDLES));
        assert!(is_listed(" @EUCouncil ", FOREIGN_GOVERNMENT_HANDLES));
    }
}
