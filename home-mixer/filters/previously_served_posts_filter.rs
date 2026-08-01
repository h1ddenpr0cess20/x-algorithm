use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::util::candidates_util::get_related_post_ids;
use std::collections::HashSet;
use xai_candidate_pipeline::filter::{Filter, FilterResult};

pub struct PreviouslyServedPostsFilter;

impl Filter<ScoredPostsQuery, PostCandidate> for PreviouslyServedPostsFilter {
    fn filter(
        &self,
        query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> FilterResult<PostCandidate> {
        // Built once instead of scanning the served id list per candidate: the filter now runs on
        // every request, so this was a linear scan per candidate per related id.
        let served_ids: HashSet<u64> = query.served_ids.iter().copied().collect();

        let (removed, kept): (Vec<_>, Vec<_>) = candidates.into_iter().partition(|c| {
            get_related_post_ids(c)
                .iter()
                .any(|id| served_ids.contains(id))
        });

        FilterResult { kept, removed }
    }
}

#[cfg(test)]
mod tests {
    use super::PreviouslyServedPostsFilter;
    use crate::models::candidate::PostCandidate;
    use crate::models::query::ScoredPostsQuery;
    use xai_candidate_pipeline::filter::Filter;

    #[test]
    fn removes_served_posts_on_every_request_type() {
        let query = ScoredPostsQuery {
            served_ids: vec![42],
            is_bottom_request: false,
            is_top_request: true,
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

        let result = PreviouslyServedPostsFilter.filter(&query, candidates);

        assert_eq!(result.kept.len(), 1);
        assert_eq!(result.kept[0].tweet_id, 43);
        assert_eq!(result.removed.len(), 1);
        assert_eq!(result.removed[0].tweet_id, 42);
    }

    #[test]
    fn removes_retweets_of_a_served_source_post() {
        let query = ScoredPostsQuery {
            served_ids: vec![42],
            ..Default::default()
        };
        let candidate = PostCandidate {
            tweet_id: 99,
            retweeted_tweet_id: Some(42),
            ..Default::default()
        };

        let result = PreviouslyServedPostsFilter.filter(&query, vec![candidate]);

        assert!(result.kept.is_empty());
        assert_eq!(result.removed.len(), 1);
    }
}
