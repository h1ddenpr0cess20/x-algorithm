use crate::filters::topic_ids_filter::TopicIdExpansion;
use crate::models::candidate::{CandidateHelpers, PostCandidate};
use crate::models::query::ScoredPostsQuery;
use crate::params::topics::{
    XAI_AI, XAI_ANIME, XAI_BIOTECH, XAI_CELEBRITY, XAI_CRIME, XAI_EDUCATION, XAI_ELECTIONS,
    XAI_J_POP, XAI_K_POP, XAI_MEMES, XAI_MOVIES_TV, XAI_NATURAL_DISASTERS, XAI_NEWS, XAI_POLITICS,
    XAI_ROBOTICS, XAI_SCIENCE, XAI_SOFTWARE_DEVELOPMENT, XAI_SPACE, XAI_SPORTS_REAL,
    XAI_STOCKS_ECONOMY, XAI_STREAMING, XAI_TECHNOLOGY, XAI_US_IRAN_WAR,
};
use std::collections::HashMap;
use tonic::async_trait;
use xai_candidate_pipeline::scorer::Scorer;

const IN_NETWORK_WEIGHT_FACTOR: f64 = 2.0;
const OUT_OF_NETWORK_WEIGHT_FACTOR: f64 = 0.1;
const HARD_NEWS_WEIGHT_FACTOR: f64 = 1.5;
const OVEREXPOSED_ELON_TOPIC_WEIGHT_FACTOR: f64 = 0.5;
const ELON_MENTION_RATIO_THRESHOLD: f64 = 0.25;
const MIN_AUTHOR_POSTS_FOR_TOPIC_RATIO: usize = 4;

const STEM_WEIGHT_FACTOR: f64 = 1.25;
const SPORTS_WEIGHT_FACTOR: f64 = 0.6;
const FANDOM_WEIGHT_FACTOR: f64 = 0.6;

const HARD_NEWS_TOPIC_IDS: [i64; 7] = [
    XAI_NEWS,
    XAI_NATURAL_DISASTERS,
    XAI_POLITICS,
    XAI_ELECTIONS,
    XAI_US_IRAN_WAR,
    XAI_CRIME,
    XAI_STOCKS_ECONOMY,
];

const STEM_TOPIC_IDS: &[i64] = &[
    XAI_SCIENCE,
    XAI_TECHNOLOGY,
    XAI_SOFTWARE_DEVELOPMENT,
    XAI_SPACE,
    XAI_ROBOTICS,
    XAI_BIOTECH,
    XAI_AI,
    XAI_EDUCATION,
];

/// Fan culture around people and franchises, as distinct from the work itself. Music, art,
/// photography and design are deliberately absent: those are people posting what they made.
const FANDOM_TOPIC_IDS: &[i64] = &[
    XAI_CELEBRITY,
    XAI_K_POP,
    XAI_J_POP,
    XAI_ANIME,
    XAI_MOVIES_TV,
    XAI_STREAMING,
    XAI_MEMES,
];

pub struct FeedPolicyScorer;

impl FeedPolicyScorer {
    fn network_weight(in_network: Option<bool>) -> f64 {
        match in_network {
            Some(true) => IN_NETWORK_WEIGHT_FACTOR,
            Some(false) => OUT_OF_NETWORK_WEIGHT_FACTOR,
            None => 0.0,
        }
    }

    fn contains_hard_news_topic(topic_ids: Option<&[i64]>) -> bool {
        topic_ids.is_some_and(|ids| {
            ids.iter()
                .any(|topic_id| HARD_NEWS_TOPIC_IDS.contains(topic_id))
        })
    }

    fn is_hard_news(candidate: &PostCandidate) -> bool {
        Self::contains_hard_news_topic(candidate.filtered_topic_ids.as_deref())
            || Self::contains_hard_news_topic(candidate.unfiltered_topic_ids.as_deref())
    }

    fn has_topic(candidate: &PostCandidate, topic_ids: &[i64]) -> bool {
        [
            candidate.filtered_topic_ids.as_deref(),
            candidate.unfiltered_topic_ids.as_deref(),
        ]
        .into_iter()
        .flatten()
        .any(|ids| ids.iter().any(|topic_id| topic_ids.contains(topic_id)))
    }

    /// The full sports cluster — every league and discipline already grouped under
    /// `XAI_SPORTS_REAL` — so this stays in sync as topics are added there.
    fn sports_topic_ids() -> &'static [i64] {
        TopicIdExpansion::category_ids(XAI_SPORTS_REAL).unwrap_or(&[XAI_SPORTS_REAL])
    }

    /// Boosts STEM, reduces sports and fan culture. Independent of the hard-news boost, which
    /// keeps its own weight.
    fn category_weight(candidate: &PostCandidate) -> f64 {
        let stem_weight = if Self::has_topic(candidate, STEM_TOPIC_IDS) {
            STEM_WEIGHT_FACTOR
        } else {
            1.0
        };
        let sports_weight = if Self::has_topic(candidate, Self::sports_topic_ids()) {
            SPORTS_WEIGHT_FACTOR
        } else {
            1.0
        };
        let fandom_weight = if Self::has_topic(candidate, FANDOM_TOPIC_IDS) {
            FANDOM_WEIGHT_FACTOR
        } else {
            1.0
        };

        stem_weight * sports_weight * fandom_weight
    }

    fn mentions_elon_musk(text: &str) -> bool {
        text.split(|character: char| !character.is_alphanumeric())
            .any(|token| {
                token.eq_ignore_ascii_case("elon") || token.eq_ignore_ascii_case("elonmusk")
            })
    }

    // The public Home Mixer does not expose author timelines. Use the current candidate batch as
    // a conservative proxy and require multiple observations before applying an author-level rule.
    fn author_elon_mention_ratios(candidates: &[PostCandidate]) -> HashMap<u64, f64> {
        let mut counts: HashMap<u64, (usize, usize)> = HashMap::new();
        for candidate in candidates {
            let entry = counts.entry(candidate.get_original_author_id()).or_default();
            entry.0 += 1;
            if Self::mentions_elon_musk(&candidate.tweet_text) {
                entry.1 += 1;
            }
        }

        counts
            .into_iter()
            .filter_map(|(author_id, (post_count, mention_count))| {
                (post_count >= MIN_AUTHOR_POSTS_FOR_TOPIC_RATIO).then_some((
                    author_id,
                    mention_count as f64 / post_count as f64,
                ))
            })
            .collect()
    }

    fn policy_weight(candidate: &PostCandidate, elon_mention_ratio: Option<f64>) -> f64 {
        let news_weight = if Self::is_hard_news(candidate) {
            HARD_NEWS_WEIGHT_FACTOR
        } else {
            1.0
        };
        let topic_weight = if elon_mention_ratio
            .is_some_and(|ratio| ratio > ELON_MENTION_RATIO_THRESHOLD)
        {
            OVEREXPOSED_ELON_TOPIC_WEIGHT_FACTOR
        } else {
            1.0
        };

        Self::network_weight(candidate.in_network)
            * news_weight
            * topic_weight
            * Self::category_weight(candidate)
    }
}

#[async_trait]
impl Scorer<ScoredPostsQuery, PostCandidate> for FeedPolicyScorer {
    async fn score(
        &self,
        _query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Vec<Result<PostCandidate, String>> {
        let elon_mention_ratios = Self::author_elon_mention_ratios(candidates);

        candidates
            .iter()
            .map(|candidate| {
                let elon_mention_ratio = elon_mention_ratios
                    .get(&candidate.get_original_author_id())
                    .copied();
                Ok(PostCandidate {
                    score: candidate
                        .score
                        .map(|score| score * Self::policy_weight(candidate, elon_mention_ratio)),
                    ..Default::default()
                })
            })
            .collect()
    }

    fn update(&self, candidate: &mut PostCandidate, scored: PostCandidate) {
        candidate.score = scored.score;
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ELON_MENTION_RATIO_THRESHOLD, FANDOM_WEIGHT_FACTOR, FeedPolicyScorer,
        HARD_NEWS_WEIGHT_FACTOR, IN_NETWORK_WEIGHT_FACTOR, OUT_OF_NETWORK_WEIGHT_FACTOR,
        OVEREXPOSED_ELON_TOPIC_WEIGHT_FACTOR, SPORTS_WEIGHT_FACTOR, STEM_WEIGHT_FACTOR,
        XAI_CELEBRITY, XAI_NEWS, XAI_SCIENCE,
    };
    use crate::models::candidate::PostCandidate;
    use crate::params::topics::{XAI_NBA, XAI_SOCCER};

    #[test]
    fn followed_accounts_receive_double_weight() {
        assert_eq!(
            FeedPolicyScorer::network_weight(Some(true)),
            IN_NETWORK_WEIGHT_FACTOR
        );
    }

    #[test]
    fn out_of_network_candidates_keep_ten_percent() {
        assert_eq!(
            FeedPolicyScorer::network_weight(Some(false)),
            OUT_OF_NETWORK_WEIGHT_FACTOR
        );
    }

    #[test]
    fn unknown_network_candidates_receive_zero_weight() {
        assert_eq!(FeedPolicyScorer::network_weight(None), 0.0);
    }

    #[test]
    fn hard_news_receives_additional_boost() {
        let candidate = PostCandidate {
            in_network: Some(false),
            filtered_topic_ids: Some(vec![XAI_NEWS]),
            ..Default::default()
        };

        let expected = OUT_OF_NETWORK_WEIGHT_FACTOR * HARD_NEWS_WEIGHT_FACTOR;
        assert!(
            (FeedPolicyScorer::policy_weight(&candidate, None) - expected).abs() < f64::EPSILON
        );
    }

    #[test]
    fn deranks_authors_above_elon_mention_threshold() {
        let candidates = [
            candidate(7, "Elon announced another product"),
            candidate(7, "A second Elon post"),
            candidate(7, "An unrelated post"),
            candidate(7, "Another unrelated post"),
        ];
        let ratios = FeedPolicyScorer::author_elon_mention_ratios(&candidates);
        let ratio = ratios.get(&7).copied();

        assert!(ratio.is_some_and(|value| value > ELON_MENTION_RATIO_THRESHOLD));
        assert_eq!(
            FeedPolicyScorer::policy_weight(&candidates[0], ratio),
            IN_NETWORK_WEIGHT_FACTOR * OVEREXPOSED_ELON_TOPIC_WEIGHT_FACTOR
        );
    }

    #[test]
    fn does_not_derank_authors_at_exactly_twenty_five_percent() {
        let candidates = [
            candidate(7, "Elon announced another product"),
            candidate(7, "An unrelated post"),
            candidate(7, "Another unrelated post"),
            candidate(7, "One more unrelated post"),
        ];
        let ratios = FeedPolicyScorer::author_elon_mention_ratios(&candidates);

        assert_eq!(ratios.get(&7), Some(&ELON_MENTION_RATIO_THRESHOLD));
        assert_eq!(
            FeedPolicyScorer::policy_weight(&candidates[0], ratios.get(&7).copied()),
            IN_NETWORK_WEIGHT_FACTOR
        );
    }

    #[test]
    fn requires_multiple_posts_before_deranking_an_author() {
        let candidates = [candidate(7, "Elon announced another product")];

        assert!(FeedPolicyScorer::author_elon_mention_ratios(&candidates).is_empty());
    }

    fn candidate(author_id: u64, text: &str) -> PostCandidate {
        PostCandidate {
            author_id,
            in_network: Some(true),
            tweet_text: text.to_string(),
            ..Default::default()
        }
    }

    fn topical(topic_id: i64) -> PostCandidate {
        PostCandidate {
            filtered_topic_ids: Some(vec![topic_id]),
            ..candidate(7, "a post")
        }
    }

    #[test]
    fn stem_receives_a_small_boost() {
        assert_eq!(
            FeedPolicyScorer::policy_weight(&topical(XAI_SCIENCE), None),
            IN_NETWORK_WEIGHT_FACTOR * STEM_WEIGHT_FACTOR
        );
    }

    #[test]
    fn stem_boost_is_smaller_than_the_hard_news_boost() {
        const {
            assert!(STEM_WEIGHT_FACTOR > 1.0 && STEM_WEIGHT_FACTOR < HARD_NEWS_WEIGHT_FACTOR)
        };
    }

    #[test]
    fn sports_is_reduced() {
        assert_eq!(
            FeedPolicyScorer::policy_weight(&topical(XAI_NBA), None),
            IN_NETWORK_WEIGHT_FACTOR * SPORTS_WEIGHT_FACTOR
        );
    }

    #[test]
    fn sports_reduction_covers_the_whole_league_cluster() {
        // Sourced from the existing XAI_SPORTS_REAL grouping rather than a second hand-kept list,
        // so a topic added there is covered here automatically.
        let sports_topic_ids = FeedPolicyScorer::sports_topic_ids();

        assert!(sports_topic_ids.contains(&XAI_NBA));
        assert!(sports_topic_ids.contains(&XAI_SOCCER));
        assert!(
            sports_topic_ids.len() > 20,
            "expected the full league cluster, got {}",
            sports_topic_ids.len()
        );
    }

    #[test]
    fn fandom_is_reduced() {
        assert_eq!(
            FeedPolicyScorer::policy_weight(&topical(XAI_CELEBRITY), None),
            IN_NETWORK_WEIGHT_FACTOR * FANDOM_WEIGHT_FACTOR
        );
    }

    #[test]
    fn uncategorized_posts_are_unaffected_by_the_new_weights() {
        assert_eq!(
            FeedPolicyScorer::category_weight(&candidate(7, "an ordinary post")),
            1.0
        );
    }

    #[test]
    fn new_weights_read_the_unfiltered_topic_set_too() {
        let stem = PostCandidate {
            unfiltered_topic_ids: Some(vec![XAI_SCIENCE]),
            ..candidate(7, "a new telescope image")
        };

        assert_eq!(FeedPolicyScorer::category_weight(&stem), STEM_WEIGHT_FACTOR);
    }

    #[test]
    fn hard_news_boost_still_applies_on_top_of_the_new_weights() {
        let sports_news = PostCandidate {
            filtered_topic_ids: Some(vec![XAI_NEWS, XAI_NBA]),
            ..candidate(7, "league announces a rule change")
        };

        assert_eq!(
            FeedPolicyScorer::policy_weight(&sports_news, None),
            IN_NETWORK_WEIGHT_FACTOR * HARD_NEWS_WEIGHT_FACTOR * SPORTS_WEIGHT_FACTOR
        );
    }
}
