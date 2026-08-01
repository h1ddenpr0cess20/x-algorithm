use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use std::collections::HashSet;
use xai_candidate_pipeline::filter::{Filter, FilterResult};

/// Posts shorter than this once normalized are never deduplicated. Short posts ("good morning",
/// "this is huge") collide constantly between unrelated authors without being spam.
const MIN_TEMPLATE_CHARACTER_COUNT: usize = 40;
const MIN_TEMPLATE_WORD_COUNT: usize = 6;

/// Removes posts that are literal template duplicates of another candidate in the same batch:
/// identical wording once mentions, links and numbers are masked out. This is the only text
/// signal in the pipeline confident enough for hard removal — everything softer (engagement
/// bait, clickbait) is demoted by `FeedPolicyScorer` instead, so a false positive costs a post
/// some rank rather than deleting it from the feed.
pub struct TemplateSpamFilter;

impl TemplateSpamFilter {
    /// Reduces a post to its structural template: `@handles`, links and numbers become
    /// placeholders, punctuation and casing are dropped. Returns `None` when the result is too
    /// short to be a meaningful fingerprint.
    fn normalize_template(text: &str) -> Option<String> {
        let normalized = text
            .split_whitespace()
            .filter_map(|token| {
                let lowercase = token.to_lowercase();
                if lowercase.starts_with("http://") || lowercase.starts_with("https://") {
                    return Some("<url>".to_string());
                }
                if lowercase.starts_with('@') {
                    return Some("<user>".to_string());
                }

                let cleaned: String = lowercase
                    .chars()
                    .filter(|character| character.is_alphanumeric())
                    .collect();
                if cleaned.is_empty() {
                    None
                } else if cleaned.chars().all(|character| character.is_numeric()) {
                    Some("<number>".to_string())
                } else {
                    Some(cleaned)
                }
            })
            .collect::<Vec<_>>()
            .join(" ");

        let word_count = normalized.split_whitespace().count();
        (normalized.len() >= MIN_TEMPLATE_CHARACTER_COUNT && word_count >= MIN_TEMPLATE_WORD_COUNT)
            .then_some(normalized)
    }
}

impl Filter<ScoredPostsQuery, PostCandidate> for TemplateSpamFilter {
    fn filter(
        &self,
        _query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> FilterResult<PostCandidate> {
        let mut seen_templates = HashSet::new();
        let mut kept = Vec::new();
        let mut removed = Vec::new();

        for candidate in candidates {
            let is_duplicate = Self::normalize_template(&candidate.tweet_text)
                .is_some_and(|template| !seen_templates.insert(template));

            if is_duplicate {
                removed.push(candidate);
            } else {
                kept.push(candidate);
            }
        }

        FilterResult { kept, removed }
    }
}

#[cfg(test)]
mod tests {
    use super::TemplateSpamFilter;
    use crate::models::candidate::PostCandidate;
    use crate::models::query::ScoredPostsQuery;
    use xai_candidate_pipeline::filter::Filter;

    fn candidate(id: u64, text: &str) -> PostCandidate {
        PostCandidate {
            tweet_id: id,
            in_network: Some(true),
            tweet_text: text.to_string(),
            ..Default::default()
        }
    }

    fn filter(candidates: Vec<PostCandidate>) -> (Vec<u64>, Vec<u64>) {
        let result = TemplateSpamFilter.filter(&ScoredPostsQuery::default(), candidates);
        (
            result.kept.iter().map(|c| c.tweet_id).collect(),
            result.removed.iter().map(|c| c.tweet_id).collect(),
        )
    }

    #[test]
    fn removes_repeated_normalized_templates() {
        let (kept, removed) = filter(vec![
            candidate(
                1,
                "Daily update for @alice has the same repeated template number 100 https://one.test",
            ),
            candidate(
                2,
                "Daily update for @bob has the same repeated template number 200 https://two.test",
            ),
        ]);

        assert_eq!(kept, vec![1]);
        assert_eq!(removed, vec![2]);
    }

    #[test]
    fn keeps_distinct_posts() {
        let (kept, removed) = filter(vec![
            candidate(
                1,
                "The city council approved the new transit budget this morning",
            ),
            candidate(
                2,
                "Rain is expected across the region for most of the weekend",
            ),
        ]);

        assert_eq!(kept, vec![1, 2]);
        assert!(removed.is_empty());
    }

    #[test]
    fn never_dedupes_short_posts() {
        let (kept, removed) = filter(vec![
            candidate(1, "good morning"),
            candidate(2, "good morning"),
            candidate(3, "good morning"),
        ]);

        assert_eq!(kept, vec![1, 2, 3]);
        assert!(removed.is_empty());
    }

    #[test]
    fn keeps_candidates_missing_network_hydration() {
        // Missing hydration is a data problem, not a quality signal: dropping these would
        // silently empty the feed whenever the in-network hydrator degrades.
        let unhydrated = PostCandidate {
            tweet_id: 1,
            in_network: None,
            tweet_text: "An ordinary post with no duplicate anywhere in this batch".to_string(),
            ..Default::default()
        };

        let (kept, removed) = filter(vec![unhydrated]);

        assert_eq!(kept, vec![1]);
        assert!(removed.is_empty());
    }
}
