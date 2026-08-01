use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use xai_candidate_pipeline::filter::{Filter, FilterResult};

/// Accounts excluded from the For You feed, by screen name, without the leading `@`.
///
/// This is an operator editorial decision, not a learned or measured quality signal. It lives in
/// its own file under its own name so that it is a single auditable list rather than a special
/// case buried inside a heuristic: anyone reading the pipeline can see exactly who is on it and
/// change it in one place.
const SUPPRESSED_SCREEN_NAMES: &[&str] = &[
    "dogedesigner",
    "farzyness",
    "kettlebelldan",
    "xfreeze",
    "wholemars",
    "sawyermerritt",
    "teslaownerssv",
];

/// Removes posts authored by, or reposted from, an account on `SUPPRESSED_SCREEN_NAMES`.
///
/// Matching is on screen name, which `GizmoduckCandidateHydrator` populates. When that hydration
/// is missing the candidate is kept: a suppression list should fail open, so that a degraded
/// hydrator costs some precision on this list rather than removing posts it never matched.
pub struct SuppressedAuthorsFilter;

impl SuppressedAuthorsFilter {
    fn is_suppressed(screen_name: Option<&str>) -> bool {
        screen_name.is_some_and(|screen_name| {
            let normalized = screen_name
                .trim()
                .trim_start_matches('@')
                .to_ascii_lowercase();
            SUPPRESSED_SCREEN_NAMES.contains(&normalized.as_str())
        })
    }

    fn should_remove(candidate: &PostCandidate) -> bool {
        Self::is_suppressed(candidate.author_screen_name.as_deref())
            || Self::is_suppressed(candidate.retweeted_screen_name.as_deref())
    }
}

impl Filter<ScoredPostsQuery, PostCandidate> for SuppressedAuthorsFilter {
    fn filter(
        &self,
        _query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> FilterResult<PostCandidate> {
        let (removed, kept): (Vec<_>, Vec<_>) =
            candidates.into_iter().partition(Self::should_remove);

        FilterResult { kept, removed }
    }
}

#[cfg(test)]
mod tests {
    use super::{SUPPRESSED_SCREEN_NAMES, SuppressedAuthorsFilter};
    use crate::models::candidate::PostCandidate;
    use crate::models::query::ScoredPostsQuery;
    use xai_candidate_pipeline::filter::Filter;

    fn authored_by(screen_name: Option<&str>) -> PostCandidate {
        PostCandidate {
            tweet_id: 1,
            author_screen_name: screen_name.map(str::to_string),
            ..Default::default()
        }
    }

    fn is_removed(candidate: PostCandidate) -> bool {
        let result = SuppressedAuthorsFilter.filter(&ScoredPostsQuery::default(), vec![candidate]);
        result.kept.is_empty() && result.removed.len() == 1
    }

    #[test]
    fn removes_suppressed_authors() {
        assert!(is_removed(authored_by(Some("dogedesigner"))));
    }

    #[test]
    fn matching_is_case_insensitive() {
        assert!(is_removed(authored_by(Some("DogeDesigner"))));
    }

    #[test]
    fn matching_ignores_a_leading_at_sign_and_whitespace() {
        assert!(is_removed(authored_by(Some("  @DogeDesigner "))));
    }

    #[test]
    fn removes_reposts_of_suppressed_authors() {
        let repost = PostCandidate {
            tweet_id: 1,
            author_screen_name: Some("someone_else".to_string()),
            retweeted_screen_name: Some("@TeslaOwnersSV".to_string()),
            ..Default::default()
        };

        assert!(is_removed(repost));
    }

    #[test]
    fn keeps_everyone_else() {
        assert!(!is_removed(authored_by(Some("someone_else"))));
    }

    #[test]
    fn keeps_candidates_missing_screen_name_hydration() {
        // Fail open: a degraded Gizmoduck hydration must not start removing unrelated posts.
        assert!(!is_removed(authored_by(None)));
    }

    #[test]
    fn does_not_match_on_a_substring_of_a_listed_name() {
        assert!(!is_removed(authored_by(Some("dogedesigner_fan"))));
        assert!(!is_removed(authored_by(Some("not_dogedesigner"))));
    }

    #[test]
    fn the_list_is_stored_normalized() {
        // Entries are compared against a trimmed, lowercased, '@'-stripped screen name, so an
        // entry that is not already in that form would silently never match.
        for name in SUPPRESSED_SCREEN_NAMES {
            assert_eq!(*name, name.trim().trim_start_matches('@').to_lowercase());
        }
    }
}
