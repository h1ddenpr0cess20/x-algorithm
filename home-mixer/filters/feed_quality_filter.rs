use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use std::collections::HashSet;
use xai_candidate_pipeline::filter::{Filter, FilterResult};

const MAX_SHORT_FORM_VIDEO_DURATION_MS: i32 = 60_000;
const MIN_TEMPLATE_CHARACTER_COUNT: usize = 40;
const MIN_TEMPLATE_WORD_COUNT: usize = 6;

const RESTRICTED_SCREEN_NAMES: [&str; 7] = [
    "dogedesigner",
    "farzyness",
    "kettlebelldan",
    "xfreeze",
    "wholemars",
    "sawyermerritt",
    "teslaownerssv",
    "nikitabier"
];

const ENGAGEMENT_BAIT_PHRASES: [&str; 9] = [
    "like and repost",
    "like & repost",
    "repost if you",
    "retweet if you",
    "share if you",
    "follow for more",
    "comment below",
    "tag someone who",
    "who agrees with me",
];

const CLICKBAIT_PHRASES: [&str; 6] = [
    "you won't believe",
    "what happened next",
    "they don't want you to know",
    "this one weird trick",
    "the truth will shock you",
    "wait until you see",
];

pub struct FeedQualityFilter;

impl FeedQualityFilter {
    fn is_short_form_video(candidate: &PostCandidate) -> bool {
        [
            candidate.min_video_duration_ms,
            candidate.quoted_video_duration_ms,
        ]
        .into_iter()
        .flatten()
        .any(|duration| duration <= MAX_SHORT_FORM_VIDEO_DURATION_MS)
    }

    fn contains_phrase(text: &str, phrases: &[&str]) -> bool {
        let lowercase = text.to_lowercase();
        phrases.iter().any(|phrase| lowercase.contains(phrase))
    }

    fn is_engagement_bait(text: &str) -> bool {
        Self::contains_phrase(text, &ENGAGEMENT_BAIT_PHRASES)
    }

    fn is_clickbait(text: &str) -> bool {
        Self::contains_phrase(text, &CLICKBAIT_PHRASES)
    }

    fn is_restricted_screen_name(screen_name: Option<&str>) -> bool {
        screen_name.is_some_and(|screen_name| {
            let normalized = screen_name
                .trim()
                .trim_start_matches('@')
                .to_ascii_lowercase();
            RESTRICTED_SCREEN_NAMES.contains(&normalized.as_str())
        })
    }

    fn is_restricted_author(candidate: &PostCandidate) -> bool {
        Self::is_restricted_screen_name(candidate.author_screen_name.as_deref())
            || Self::is_restricted_screen_name(candidate.retweeted_screen_name.as_deref())
    }

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

        // Counted in characters, not bytes: String::len is a byte count, which made the same
        // 40-character post pass in a multi-byte script and fail in a single-byte one.
        let character_count = normalized.chars().count();
        let word_count = normalized.split_whitespace().count();
        (character_count >= MIN_TEMPLATE_CHARACTER_COUNT
            && word_count >= MIN_TEMPLATE_WORD_COUNT)
            .then_some(normalized)
    }

    fn should_remove(candidate: &PostCandidate) -> bool {
        candidate.in_network.is_none()
            || Self::is_short_form_video(candidate)
            || Self::is_restricted_author(candidate)
            || Self::is_engagement_bait(&candidate.tweet_text)
            || Self::is_clickbait(&candidate.tweet_text)
    }
}

impl Filter<ScoredPostsQuery, PostCandidate> for FeedQualityFilter {
    fn filter(
        &self,
        _query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> FilterResult<PostCandidate> {
        let mut seen_templates = HashSet::new();
        let mut kept = Vec::new();
        let mut removed = Vec::new();

        for candidate in candidates {
            if Self::should_remove(&candidate) {
                removed.push(candidate);
                continue;
            }

            if let Some(template) = Self::normalize_template(&candidate.tweet_text)
                && !seen_templates.insert(template)
            {
                removed.push(candidate);
                continue;
            }

            kept.push(candidate);
        }

        FilterResult { kept, removed }
    }
}

#[cfg(test)]
mod tests {
    use super::{FeedQualityFilter, MAX_SHORT_FORM_VIDEO_DURATION_MS, MIN_TEMPLATE_CHARACTER_COUNT};
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

    #[test]
    fn removes_unknown_network_candidates() {
        let unknown = PostCandidate {
            tweet_id: 1,
            in_network: None,
            ..Default::default()
        };
        let result = FeedQualityFilter.filter(&ScoredPostsQuery::default(), vec![unknown]);

        assert!(result.kept.is_empty());
        assert_eq!(result.removed.len(), 1);
    }

    #[test]
    fn removes_short_form_video_candidates() {
        let mut short_video = candidate(1, "A short video");
        short_video.min_video_duration_ms = Some(MAX_SHORT_FORM_VIDEO_DURATION_MS);
        let result = FeedQualityFilter.filter(&ScoredPostsQuery::default(), vec![short_video]);

        assert!(result.kept.is_empty());
        assert_eq!(result.removed.len(), 1);
    }

    #[test]
    fn preserves_long_form_video_candidates() {
        let mut long_video = candidate(1, "A long video");
        long_video.min_video_duration_ms = Some(MAX_SHORT_FORM_VIDEO_DURATION_MS + 1);
        let result = FeedQualityFilter.filter(&ScoredPostsQuery::default(), vec![long_video]);

        assert_eq!(result.kept.len(), 1);
        assert!(result.removed.is_empty());
    }

    #[test]
    fn removes_explicit_engagement_bait() {
        let bait = candidate(1, "Like and repost if this made your day");
        let result = FeedQualityFilter.filter(&ScoredPostsQuery::default(), vec![bait]);

        assert!(result.kept.is_empty());
        assert_eq!(result.removed.len(), 1);
    }

    #[test]
    fn removes_high_confidence_clickbait() {
        let clickbait = candidate(1, "You won't believe what happened next in this story");
        let result = FeedQualityFilter.filter(&ScoredPostsQuery::default(), vec![clickbait]);

        assert!(result.kept.is_empty());
        assert_eq!(result.removed.len(), 1);
    }

    #[test]
    fn removes_restricted_direct_authors_case_insensitively() {
        let mut restricted = candidate(1, "A regular post");
        restricted.author_screen_name = Some("DogeDesigner".to_string());
        let result = FeedQualityFilter.filter(&ScoredPostsQuery::default(), vec![restricted]);

        assert!(result.kept.is_empty());
        assert_eq!(result.removed.len(), 1);
    }

    #[test]
    fn removes_retweets_of_restricted_authors() {
        let mut restricted = candidate(1, "A retweeted post");
        restricted.retweeted_screen_name = Some("@TeslaOwnersSV".to_string());
        let result = FeedQualityFilter.filter(&ScoredPostsQuery::default(), vec![restricted]);

        assert!(result.kept.is_empty());
        assert_eq!(result.removed.len(), 1);
    }

    #[test]
    fn template_length_is_measured_in_characters_not_bytes() {
        // Two identical Cyrillic posts of 26 characters but 44 bytes. A byte-based threshold
        // treated them as long enough to dedupe, while the English equivalent of the same
        // character length was left alone.
        let text = "аб вг де жз ий кл мн оп рс";
        assert_eq!(text.chars().count(), 26);
        assert!(text.len() >= MIN_TEMPLATE_CHARACTER_COUNT);

        let result = FeedQualityFilter.filter(
            &ScoredPostsQuery::default(),
            vec![candidate(1, text), candidate(2, text)],
        );

        assert_eq!(result.kept.len(), 2);
        assert!(result.removed.is_empty());
    }

    #[test]
    fn long_multibyte_templates_are_still_deduped() {
        let text = "аб вг де жз ий кл мн оп рс ту фх цч шщ ыэ юя аб вг де жз ий";
        assert!(text.chars().count() >= 40);

        let result = FeedQualityFilter.filter(
            &ScoredPostsQuery::default(),
            vec![candidate(1, text), candidate(2, text)],
        );

        assert_eq!(result.kept.len(), 1);
        assert_eq!(result.removed.len(), 1);
    }

    #[test]
    fn removes_repeated_normalized_templates() {
        let first = candidate(
            1,
            "Daily update for @alice has the same repeated template number 100 https://one.test",
        );
        let second = candidate(
            2,
            "Daily update for @bob has the same repeated template number 200 https://two.test",
        );
        let result =
            FeedQualityFilter.filter(&ScoredPostsQuery::default(), vec![first, second]);

        assert_eq!(result.kept.len(), 1);
        assert_eq!(result.removed.len(), 1);
        assert_eq!(result.kept[0].tweet_id, 1);
    }
}
