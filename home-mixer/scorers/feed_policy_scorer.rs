use crate::filters::topic_ids_filter::TopicIdExpansion;
use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::topics::{
    XAI_AI, XAI_ANIME, XAI_BIOTECH, XAI_CELEBRITY, XAI_CRIME, XAI_EDUCATION, XAI_ELECTIONS,
    XAI_J_POP, XAI_K_POP, XAI_MEMES, XAI_MOVIES_TV, XAI_NATURAL_DISASTERS, XAI_NEWS, XAI_POLITICS,
    XAI_ROBOTICS, XAI_SCIENCE, XAI_SOFTWARE_DEVELOPMENT, XAI_SPACE, XAI_SPORTS_REAL,
    XAI_STOCKS_ECONOMY, XAI_STREAMING, XAI_TECHNOLOGY, XAI_US_IRAN_WAR,
};
use std::collections::{HashMap, HashSet};
use tonic::async_trait;
use xai_candidate_pipeline::scorer::Scorer;

const HARD_NEWS_WEIGHT_FACTOR: f64 = 1.5;
const STEM_WEIGHT_FACTOR: f64 = 1.25;
const SPORTS_WEIGHT_FACTOR: f64 = 0.6;
const FANDOM_WEIGHT_FACTOR: f64 = 0.6;
const BAIT_WEIGHT_FACTOR: f64 = 0.4;

/// Share of a single response any one entity may occupy before its posts start losing rank.
const MAX_ENTITY_FEED_SHARE: f64 = 0.15;
/// Floor on the concentration penalty. An overexposed entity is demoted, never suppressed: a
/// genuinely dominant story (an election, a disaster) still surfaces, it just stops crowding out
/// everything else.
const MIN_ENTITY_WEIGHT_FACTOR: f64 = 0.4;
/// Below this batch size a share is noise rather than a measurement of the feed.
const MIN_CANDIDATES_FOR_ENTITY_CONCENTRATION: usize = 20;
/// A sigil on its own ("@", "#") names nobody. Anything past that counts, including
/// single-character handles — excluding those would be an arbitrary carve-out.
const MIN_ENTITY_LENGTH: usize = 1;

const HARD_NEWS_TOPIC_IDS: &[i64] = &[
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

/// Applies feed-level editorial policy on top of the learned ranking score.
///
/// Four rules, all stated in terms of content properties rather than named accounts or subjects,
/// so they apply identically to everyone:
///
/// 1. Hard news (politics, elections, disasters, crime, markets) and STEM (science, technology,
///    space, education) get a boost — the categories where a timely feed has the most value
///    beyond entertainment.
/// 2. Sports and fan culture are reduced. Neither is removed: someone who follows a team or a
///    franchise still sees it, it just stops being the default filler that out-ranks everything
///    else on engagement alone.
/// 3. Posts whose text solicits engagement or withholds their own point are demoted.
/// 4. No single entity may dominate a response. Whichever `@handle` or `#hashtag` is
///    overrepresented in this batch gets demoted, whoever it is.
///
/// Boosts take the max of whatever applies and demotions take the min, so a post cannot be
/// inflated or buried by accumulating category tags.
///
/// Network weighting is deliberately absent: `RankingScorer` already applies the
/// out-of-network factor from params, and stacking a second multiplier here would compound into
/// near-total suppression of out-of-network discovery.
pub struct FeedPolicyScorer;

impl FeedPolicyScorer {
    fn contains_phrase(text: &str, phrases: &[&str]) -> bool {
        let lowercase = text.to_lowercase();
        phrases.iter().any(|phrase| lowercase.contains(phrase))
    }

    fn is_bait(text: &str) -> bool {
        Self::contains_phrase(text, &ENGAGEMENT_BAIT_PHRASES)
            || Self::contains_phrase(text, &CLICKBAIT_PHRASES)
    }

    /// Topic ids are hydrated from either the experiment-filtered or unfiltered set depending on
    /// the request, so a candidate matches if the topic appears in either.
    fn has_topic(candidate: &PostCandidate, topic_ids: &[i64]) -> bool {
        [
            candidate.filtered_topic_ids.as_deref(),
            candidate.unfiltered_topic_ids.as_deref(),
        ]
        .into_iter()
        .flatten()
        .any(|ids| ids.iter().any(|topic_id| topic_ids.contains(topic_id)))
    }

    /// The full sports cluster — every league and discipline the repo already groups under
    /// `XAI_SPORTS_REAL` — so this stays in sync as topics are added there.
    fn sports_topic_ids() -> &'static [i64] {
        TopicIdExpansion::category_ids(XAI_SPORTS_REAL).unwrap_or(&[XAI_SPORTS_REAL])
    }

    /// The single largest boost a post qualifies for, or 1.0. Boosts take the max rather than
    /// the product so that tagging a post into more categories cannot inflate it.
    fn boost_weight(candidate: &PostCandidate) -> f64 {
        [
            (HARD_NEWS_TOPIC_IDS, HARD_NEWS_WEIGHT_FACTOR),
            (STEM_TOPIC_IDS, STEM_WEIGHT_FACTOR),
        ]
        .into_iter()
        .filter(|(topic_ids, _)| Self::has_topic(candidate, topic_ids))
        .map(|(_, weight)| weight)
        .fold(1.0, f64::max)
    }

    /// The single strongest demotion a post qualifies for, or 1.0. Symmetrically with boosts
    /// these take the min, so a post that is both sports and fandom is demoted once.
    fn demotion_weight(candidate: &PostCandidate) -> f64 {
        [
            (Self::sports_topic_ids(), SPORTS_WEIGHT_FACTOR),
            (FANDOM_TOPIC_IDS, FANDOM_WEIGHT_FACTOR),
        ]
        .into_iter()
        .filter(|(topic_ids, _)| Self::has_topic(candidate, topic_ids))
        .map(|(_, weight)| weight)
        .fold(1.0, f64::min)
    }

    /// The distinct `@handles` and `#hashtags` a post is about. Deduplicated per post so that
    /// repeating a handle within one post cannot inflate its own share.
    fn entities(text: &str) -> HashSet<String> {
        text.split_whitespace()
            .filter_map(|token| {
                let mut characters = token.chars();
                let sigil = characters.next()?;
                if sigil != '@' && sigil != '#' {
                    return None;
                }

                let name: String = characters
                    .take_while(|character| character.is_alphanumeric() || *character == '_')
                    .collect::<String>()
                    .to_lowercase();

                (name.chars().count() >= MIN_ENTITY_LENGTH).then(|| format!("{sigil}{name}"))
            })
            .collect()
    }

    /// Demotion factor for every entity that occupies more than `MAX_ENTITY_FEED_SHARE` of the
    /// batch. The penalty scales with the overshoot, so an entity at twice the cap is halved,
    /// and entities under the cap are untouched.
    fn entity_weights(candidates: &[PostCandidate]) -> HashMap<String, f64> {
        if candidates.len() < MIN_CANDIDATES_FOR_ENTITY_CONCENTRATION {
            return HashMap::new();
        }

        let total = candidates.len() as f64;
        let mut counts: HashMap<String, usize> = HashMap::new();
        for candidate in candidates {
            for entity in Self::entities(&candidate.tweet_text) {
                *counts.entry(entity).or_default() += 1;
            }
        }

        counts
            .into_iter()
            .filter_map(|(entity, count)| {
                let share = count as f64 / total;
                (share > MAX_ENTITY_FEED_SHARE).then(|| {
                    let weight = (MAX_ENTITY_FEED_SHARE / share).max(MIN_ENTITY_WEIGHT_FACTOR);
                    (entity, weight)
                })
            })
            .collect()
    }

    /// A post is governed by its most overexposed entity.
    fn entity_weight(entity_weights: &HashMap<String, f64>, text: &str) -> f64 {
        Self::entities(text)
            .iter()
            .filter_map(|entity| entity_weights.get(entity).copied())
            .fold(1.0, f64::min)
    }

    fn policy_weight(candidate: &PostCandidate, entity_weight: f64) -> f64 {
        let bait_weight = if Self::is_bait(&candidate.tweet_text) {
            BAIT_WEIGHT_FACTOR
        } else {
            1.0
        };

        Self::boost_weight(candidate)
            * Self::demotion_weight(candidate)
            * bait_weight
            * entity_weight
    }
}

#[async_trait]
impl Scorer<ScoredPostsQuery, PostCandidate> for FeedPolicyScorer {
    async fn score(
        &self,
        _query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Vec<Result<PostCandidate, String>> {
        let entity_weights = Self::entity_weights(candidates);

        candidates
            .iter()
            .map(|candidate| {
                let entity_weight = Self::entity_weight(&entity_weights, &candidate.tweet_text);
                Ok(PostCandidate {
                    score: candidate
                        .score
                        .map(|score| score * Self::policy_weight(candidate, entity_weight)),
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
        BAIT_WEIGHT_FACTOR, FANDOM_WEIGHT_FACTOR, FeedPolicyScorer, HARD_NEWS_WEIGHT_FACTOR,
        MIN_CANDIDATES_FOR_ENTITY_CONCENTRATION, MIN_ENTITY_WEIGHT_FACTOR, SPORTS_WEIGHT_FACTOR,
        STEM_WEIGHT_FACTOR, XAI_CELEBRITY, XAI_NEWS, XAI_SCIENCE,
    };
    use crate::models::candidate::PostCandidate;
    use crate::params::topics::XAI_NBA;
    use std::collections::HashMap;

    fn candidate(text: &str) -> PostCandidate {
        PostCandidate {
            tweet_text: text.to_string(),
            ..Default::default()
        }
    }

    fn topical(text: &str, topic_id: i64) -> PostCandidate {
        PostCandidate {
            filtered_topic_ids: Some(vec![topic_id]),
            ..candidate(text)
        }
    }

    /// A batch large enough to measure concentration on, in which `mentions` of the posts
    /// mention `@subject` and the rest are unrelated.
    fn batch(mentions: usize) -> Vec<PostCandidate> {
        (0..MIN_CANDIDATES_FOR_ENTITY_CONCENTRATION)
            .map(|i| {
                if i < mentions {
                    candidate("a post about @subject")
                } else {
                    candidate("an unrelated post")
                }
            })
            .collect()
    }

    #[test]
    fn hard_news_receives_a_boost() {
        let news = topical("the council approved the budget", XAI_NEWS);

        assert_eq!(
            FeedPolicyScorer::policy_weight(&news, 1.0),
            HARD_NEWS_WEIGHT_FACTOR
        );
    }

    #[test]
    fn stem_receives_a_smaller_boost() {
        let stem = topical("a new result on protein folding", XAI_SCIENCE);
        let weight = FeedPolicyScorer::policy_weight(&stem, 1.0);

        assert_eq!(weight, STEM_WEIGHT_FACTOR);
        assert!(weight > 1.0 && weight < HARD_NEWS_WEIGHT_FACTOR);
    }

    #[test]
    fn category_boosts_do_not_compound() {
        let both = PostCandidate {
            filtered_topic_ids: Some(vec![XAI_NEWS, XAI_SCIENCE]),
            ..candidate("a science story in the news")
        };

        assert_eq!(
            FeedPolicyScorer::policy_weight(&both, 1.0),
            HARD_NEWS_WEIGHT_FACTOR
        );
    }

    #[test]
    fn sports_is_reduced() {
        let sports = topical("final score from last night", XAI_NBA);
        let weight = FeedPolicyScorer::policy_weight(&sports, 1.0);

        assert_eq!(weight, SPORTS_WEIGHT_FACTOR);
        assert!(weight > 0.0, "sports must lose rank, not disappear");
    }

    #[test]
    fn sports_demotion_covers_the_whole_league_cluster() {
        // Sourced from the repo's own XAI_SPORTS_REAL grouping rather than a second hand-kept
        // list, so a topic added there is covered here automatically.
        let sports_topic_ids = FeedPolicyScorer::sports_topic_ids();

        assert!(sports_topic_ids.contains(&XAI_NBA));
        assert!(
            sports_topic_ids.len() > 20,
            "expected the full league cluster, got {}",
            sports_topic_ids.len()
        );
    }

    #[test]
    fn fandom_is_reduced() {
        let fandom = topical("the cast reunion everyone is talking about", XAI_CELEBRITY);

        assert_eq!(
            FeedPolicyScorer::policy_weight(&fandom, 1.0),
            FANDOM_WEIGHT_FACTOR
        );
    }

    #[test]
    fn demotions_do_not_compound() {
        let both = PostCandidate {
            filtered_topic_ids: Some(vec![XAI_NBA, XAI_CELEBRITY]),
            ..candidate("a player spotted courtside")
        };
        let weight = FeedPolicyScorer::policy_weight(&both, 1.0);

        assert_eq!(weight, SPORTS_WEIGHT_FACTOR.min(FANDOM_WEIGHT_FACTOR));
        assert!(weight >= SPORTS_WEIGHT_FACTOR * FANDOM_WEIGHT_FACTOR);
    }

    #[test]
    fn news_about_sports_keeps_its_news_boost_but_is_still_netted_down() {
        let sports_news = PostCandidate {
            filtered_topic_ids: Some(vec![XAI_NEWS, XAI_NBA]),
            ..candidate("league announces a rule change")
        };

        assert_eq!(
            FeedPolicyScorer::policy_weight(&sports_news, 1.0),
            HARD_NEWS_WEIGHT_FACTOR * SPORTS_WEIGHT_FACTOR
        );
    }

    #[test]
    fn topics_are_read_from_the_unfiltered_set_too() {
        let stem = PostCandidate {
            filtered_topic_ids: None,
            unfiltered_topic_ids: Some(vec![XAI_SCIENCE]),
            ..candidate("a new telescope image")
        };

        assert_eq!(
            FeedPolicyScorer::policy_weight(&stem, 1.0),
            STEM_WEIGHT_FACTOR
        );
    }

    #[test]
    fn bait_is_demoted_rather_than_removed() {
        let bait = candidate("Like and repost if this made your day");
        let weight = FeedPolicyScorer::policy_weight(&bait, 1.0);

        assert_eq!(weight, BAIT_WEIGHT_FACTOR);
        assert!(weight > 0.0, "bait must lose rank, not disappear");
    }

    #[test]
    fn clickbait_is_demoted() {
        let clickbait = candidate("You won't believe what happened next");

        assert_eq!(
            FeedPolicyScorer::policy_weight(&clickbait, 1.0),
            BAIT_WEIGHT_FACTOR
        );
    }

    #[test]
    fn ordinary_posts_are_left_alone() {
        let ordinary = candidate("Rain expected across the region this weekend");

        assert_eq!(FeedPolicyScorer::policy_weight(&ordinary, 1.0), 1.0);
    }

    #[test]
    fn entities_under_the_cap_are_untouched() {
        // 2 of 20 posts = 10% share, under the 15% cap.
        assert!(FeedPolicyScorer::entity_weights(&batch(2)).is_empty());
    }

    #[test]
    fn overexposed_entities_are_demoted_in_proportion_to_the_overshoot() {
        // 6 of 20 posts = 30% share, twice the cap, so the weight halves.
        let weights = FeedPolicyScorer::entity_weights(&batch(6));

        assert_eq!(weights.get("@subject"), Some(&0.5));
    }

    #[test]
    fn demotion_never_falls_below_the_floor() {
        let weights =
            FeedPolicyScorer::entity_weights(&batch(MIN_CANDIDATES_FOR_ENTITY_CONCENTRATION));

        assert_eq!(weights.get("@subject"), Some(&MIN_ENTITY_WEIGHT_FACTOR));
    }

    #[test]
    fn concentration_is_not_measured_on_small_batches() {
        let tiny = vec![candidate("@subject again"), candidate("@subject once more")];

        assert!(FeedPolicyScorer::entity_weights(&tiny).is_empty());
    }

    #[test]
    fn the_rule_does_not_name_anyone() {
        // The same 30% share produces the same demotion regardless of who the entity is: the
        // policy constrains concentration, not identity.
        let one = FeedPolicyScorer::entity_weights(&batch(6));
        let other: Vec<PostCandidate> = batch(6)
            .iter()
            .map(|c| candidate(&c.tweet_text.replace("@subject", "@someone_else")))
            .collect();
        let other = FeedPolicyScorer::entity_weights(&other);

        assert_eq!(one.get("@subject"), other.get("@someone_else"));
    }

    #[test]
    fn hashtags_and_handles_are_counted_separately() {
        let entities = FeedPolicyScorer::entities("@alice tagged #alice in a post");

        assert!(entities.contains("@alice"));
        assert!(entities.contains("#alice"));
    }

    #[test]
    fn repeating_a_handle_within_one_post_counts_once() {
        let entities = FeedPolicyScorer::entities("@alice @alice @alice");

        assert_eq!(entities.len(), 1);
    }

    #[test]
    fn single_character_handles_still_count() {
        let entities = FeedPolicyScorer::entities("a post mentioning @x");

        assert!(entities.contains("@x"));
    }

    #[test]
    fn a_bare_sigil_names_nobody() {
        let entities = FeedPolicyScorer::entities("email me @ the address # 4");

        assert!(entities.is_empty());
    }

    #[test]
    fn trailing_punctuation_is_stripped_from_entities() {
        let entities = FeedPolicyScorer::entities("thanks @alice! see #news, tomorrow");

        assert!(entities.contains("@alice"));
        assert!(entities.contains("#news"));
    }

    #[test]
    fn a_post_is_governed_by_its_most_overexposed_entity() {
        let weights = HashMap::from([("@a".to_string(), 0.8), ("@b".to_string(), 0.5)]);

        assert_eq!(
            FeedPolicyScorer::entity_weight(&weights, "@a and @b in one post"),
            0.5
        );
    }

    #[test]
    fn penalties_compose_multiplicatively() {
        let news_bait = topical("Follow for more updates on the election", XAI_NEWS);

        assert_eq!(
            FeedPolicyScorer::policy_weight(&news_bait, 0.5),
            HARD_NEWS_WEIGHT_FACTOR * BAIT_WEIGHT_FACTOR * 0.5
        );
    }
}
