use crate::params::RESULT_SIZE;
use std::sync::LazyLock;
use xai_home_mixer_proto::{BrandSafetyVerdict, FeedItem, ScoredPost, feed_item};
use xai_post_text::TweetTokenizer;
use xai_recsys_proto::{AdIndexInfo, BrandSafetyRiskLevel};
use xai_stats_receiver::global_stats_receiver;

static TWEET_TOKENIZER: LazyLock<TweetTokenizer> = LazyLock::new(TweetTokenizer::new);

/// Floor on how many posts sit between two ads, whatever the ad server asks for. The requested
/// spacing arrives from outside this service and used to be taken as given, which set the feed's
/// ad load by a number nobody here chose. This is the number this service chooses: ads may be
/// spaced further apart than this, never closer.
pub(crate) const MIN_ORGANIC_RUN: usize = 6;

/// A response too short to hold one ad at the floor spacing carries none.
pub(crate) const MIN_POSTS_FOR_ADS: usize = MIN_ORGANIC_RUN + 1;

pub(crate) const MIN_REQUESTED_GAP: usize = 3;

pub(crate) const DEFAULT_SPACING: AdSpacing = AdSpacing {
    requested: MIN_ORGANIC_RUN,
    min: MIN_ORGANIC_RUN.div_ceil(2),
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AdSpacing {
    pub(crate) requested: usize,
    pub(crate) min: usize,
}

pub(crate) fn has_avoid(post: &ScoredPost) -> bool {
    post.brand_safety_verdict() == BrandSafetyVerdict::MediumRisk
}

pub(crate) fn find_safe_gaps(scored_posts: &[ScoredPost]) -> Vec<usize> {
    let n = scored_posts.len();
    let mut safe = Vec::new();
    for g in 1..n {
        if has_avoid(&scored_posts[g - 1]) {
            continue;
        }
        if g < n && has_avoid(&scored_posts[g]) {
            continue;
        }
        safe.push(g);
    }
    safe
}

pub(crate) fn compute_spacing(ads: &[AdIndexInfo]) -> AdSpacing {
    if ads.len() < 2 {
        return DEFAULT_SPACING;
    }

    let mut positions: Vec<i32> = ads.iter().take(4).map(|a| a.insert_position).collect();
    positions.sort_unstable();

    let min_diff = positions
        .windows(2)
        .map(|w| (w[1] - w[0]) as usize)
        .filter(|&d| d > 0)
        .min();

    match min_diff {
        Some(requested) if requested >= MIN_REQUESTED_GAP => {
            let requested = requested.max(MIN_ORGANIC_RUN);
            AdSpacing {
                requested,
                min: requested.div_ceil(2),
            }
        }
        _ => DEFAULT_SPACING,
    }
}

pub(crate) fn is_bsr_low_ad(ad: &AdIndexInfo) -> bool {
    let risk = ad
        .ad_adjacency_control
        .as_ref()
        .map(|c| c.brand_safety_risk())
        .unwrap_or(BrandSafetyRiskLevel::BsrUnknown);
    matches!(
        risk,
        BrandSafetyRiskLevel::BsrLow | BrandSafetyRiskLevel::BsrIas
    )
}

pub(crate) fn should_drop_bsr_low(
    ad: &AdIndexInfo,
    above: Option<&ScoredPost>,
    below: Option<&ScoredPost>,
) -> bool {
    let risk = ad
        .ad_adjacency_control
        .as_ref()
        .map(|c| c.brand_safety_risk())
        .unwrap_or(BrandSafetyRiskLevel::BsrUnknown);
    if !matches!(
        risk,
        BrandSafetyRiskLevel::BsrLow | BrandSafetyRiskLevel::BsrIas
    ) {
        return false;
    }
    let is_lr = |p: &ScoredPost| p.brand_safety_verdict() == BrandSafetyVerdict::LowRisk;
    above.map(is_lr).unwrap_or(false) || below.map(is_lr).unwrap_or(false)
}

pub(crate) fn should_drop_handle(
    ad: &AdIndexInfo,
    above: Option<&ScoredPost>,
    below: Option<&ScoredPost>,
) -> bool {
    let handles = match ad.ad_adjacency_control.as_ref() {
        Some(ctrl) if !ctrl.handles.is_empty() => &ctrl.handles,
        _ => return false,
    };
    above
        .map(|p| handles.contains(&(p.author_id as i64)))
        .unwrap_or(false)
        || below
            .map(|p| handles.contains(&(p.author_id as i64)))
            .unwrap_or(false)
}

pub(crate) fn should_drop_keyword(
    ad: &AdIndexInfo,
    above: Option<&ScoredPost>,
    below: Option<&ScoredPost>,
) -> bool {
    let keywords = match ad.ad_adjacency_control.as_ref() {
        Some(ctrl) if !ctrl.keywords.is_empty() => &ctrl.keywords,
        _ => return false,
    };

    let tokenizer = &*TWEET_TOKENIZER;

    let tokenized_keywords: Vec<_> = keywords
        .iter()
        .map(|kw| tokenizer.tokenize(kw))
        .filter(|seq| !seq.is_empty())
        .collect();

    if tokenized_keywords.is_empty() {
        return false;
    }

    let text_matches = |p: &ScoredPost| {
        if p.tweet_text.is_empty() {
            return false;
        }
        let tweet_tokens = tokenizer.tokenize(&p.tweet_text);
        if tweet_tokens.is_empty() {
            return false;
        }
        tokenized_keywords
            .iter()
            .any(|kw_tokens| tweet_tokens.contains_keyword_sequence(kw_tokens))
    };
    above.map(text_matches).unwrap_or(false) || below.map(text_matches).unwrap_or(false)
}

pub(crate) fn posts_to_feed_items(scored_posts: Vec<ScoredPost>) -> Vec<FeedItem> {
    scored_posts
        .into_iter()
        .enumerate()
        .map(|(i, post)| FeedItem {
            position: i as i32,
            item: Some(feed_item::Item::Post(post)),
        })
        .collect()
}

pub(crate) fn interleave_and_finalize(
    scored_posts: Vec<ScoredPost>,
    ads: Vec<AdIndexInfo>,
    placements: &[usize],
) -> Vec<FeedItem> {
    let n = scored_posts.len();
    let mut items: Vec<FeedItem> = Vec::with_capacity(n + placements.len());
    let mut ads_iter = ads.into_iter();
    let mut pi = 0;

    for (i, post) in scored_posts.into_iter().enumerate() {
        if pi < placements.len() && placements[pi] == i {
            items.push(FeedItem {
                position: 0,
                item: Some(feed_item::Item::Ad(ads_iter.next().unwrap())),
            });
            pi += 1;
        }
        items.push(FeedItem {
            position: 0,
            item: Some(feed_item::Item::Post(post)),
        });
    }

    items.truncate(RESULT_SIZE);
    if matches!(items.last(), Some(item) if matches!(item.item, Some(feed_item::Item::Ad(_)))) {
        items.pop();
    }

    for (i, item) in items.iter_mut().enumerate() {
        item.position = i as i32;
    }

    items
}

const VERDICT_METRIC: &str = "AdsBlender.post_brand_safety_verdict";
const RISK_METRIC: &str = "AdsBlender.ad_brand_safety_risk";

pub(crate) fn record_post_verdict_stats(posts: &[ScoredPost]) {
    let Some(receiver) = global_stats_receiver() else {
        return;
    };

    for post in posts {
        let label = post.brand_safety_verdict().as_str_name();
        receiver.incr(VERDICT_METRIC, &[("verdict", label)], 1);
    }
}

pub(crate) fn record_ad_risk_stats(ads: &[AdIndexInfo]) {
    let Some(receiver) = global_stats_receiver() else {
        return;
    };

    for ad in ads {
        let risk_level = ad
            .ad_adjacency_control
            .as_ref()
            .map(|c| c.brand_safety_risk())
            .unwrap_or(BrandSafetyRiskLevel::BsrUnknown);

        receiver.incr(RISK_METRIC, &[("risk", risk_level.as_str_name())], 1);
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AdSpacing, DEFAULT_SPACING, MIN_ORGANIC_RUN, MIN_POSTS_FOR_ADS, MIN_REQUESTED_GAP,
        compute_spacing,
    };
    use xai_recsys_proto::AdIndexInfo;

    fn requesting(positions: &[i32]) -> Vec<AdIndexInfo> {
        positions
            .iter()
            .map(|&insert_position| AdIndexInfo {
                insert_position,
                ..Default::default()
            })
            .collect()
    }

    #[test]
    fn a_single_ad_carries_the_default_spacing() {
        assert_eq!(compute_spacing(&requesting(&[2])), DEFAULT_SPACING);
    }

    #[test]
    fn a_tighter_request_than_the_floor_is_raised_to_it() {
        // The ad server asking for one every three posts gets one every six.
        let spacing = compute_spacing(&requesting(&[3, 6, 9, 12]));

        assert_eq!(spacing.requested, MIN_ORGANIC_RUN);
    }

    #[test]
    fn a_request_wider_than_the_floor_is_left_alone() {
        let spacing = compute_spacing(&requesting(&[4, 14, 24]));

        assert_eq!(
            spacing,
            AdSpacing {
                requested: 10,
                min: 5
            }
        );
    }

    #[test]
    fn an_incoherent_request_falls_back_to_the_default() {
        // Positions closer together than MIN_REQUESTED_GAP are not a spacing the server meant.
        let spacing = compute_spacing(&requesting(&[4, 5]));

        assert!(1 < MIN_REQUESTED_GAP);
        assert_eq!(spacing, DEFAULT_SPACING);
    }

    #[test]
    fn the_default_spacing_sits_at_the_floor() {
        const { assert!(DEFAULT_SPACING.requested == MIN_ORGANIC_RUN) };
    }

    #[test]
    fn a_response_that_cannot_hold_an_ad_at_the_floor_carries_none() {
        const { assert!(MIN_POSTS_FOR_ADS > MIN_ORGANIC_RUN) };
    }
}
