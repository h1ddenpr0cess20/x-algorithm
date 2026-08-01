use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::util::candidates_util::get_related_post_ids;
use std::collections::HashSet;
use xai_candidate_pipeline::filter::{Filter, FilterResult};

pub struct PreviouslySeenPostsBackupFilter;

impl Filter<ScoredPostsQuery, PostCandidate> for PreviouslySeenPostsBackupFilter {
    fn filter(
        &self,
        query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> FilterResult<PostCandidate> {
        if query.impressed_post_ids.is_empty() {
            return FilterResult {
                kept: candidates,
                removed: Vec::new(),
            };
        }

        // Built once instead of scanning the impressed id list per candidate.
        let impressed_post_ids: HashSet<u64> = query.impressed_post_ids.iter().copied().collect();

        let (removed, kept): (Vec<_>, Vec<_>) = candidates.into_iter().partition(|c| {
            get_related_post_ids(c)
                .iter()
                .any(|id| impressed_post_ids.contains(id))
        });

        FilterResult { kept, removed }
    }
}

#[cfg(test)]
mod tests {
    use super::PreviouslySeenPostsBackupFilter;
    use crate::models::candidate::PostCandidate;
    use crate::models::query::ScoredPostsQuery;
    use xai_candidate_pipeline::filter::Filter;

    #[test]
    fn removes_impressed_posts_after_history_hydration() {
        let query = ScoredPostsQuery {
            impressed_post_ids: vec![42],
            ..Default::default()
        };
        let candidates = vec![
            PostCandidate {
                tweet_id: 42,
                ..Default::default()
            },
            PostCandidate {
                tweet_id: 43,
                ..Default::default()
            },
        ];

        let result = PreviouslySeenPostsBackupFilter.filter(&query, candidates);

        assert_eq!(result.kept.len(), 1);
        assert_eq!(result.kept[0].tweet_id, 43);
        assert_eq!(result.removed.len(), 1);
        assert_eq!(result.removed[0].tweet_id, 42);
    }

    #[test]
    fn removes_retweets_of_impressed_source_posts() {
        let query = ScoredPostsQuery {
            impressed_post_ids: vec![42],
            ..Default::default()
        };
        let candidate = PostCandidate {
            tweet_id: 99,
            retweeted_tweet_id: Some(42),
            ..Default::default()
        };

        let result = PreviouslySeenPostsBackupFilter.filter(&query, vec![candidate]);

        assert!(result.kept.is_empty());
        assert_eq!(result.removed.len(), 1);
    }
}
