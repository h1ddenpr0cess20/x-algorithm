use crate::ads::{AdsBlender, PartitionOrganicAdsBlender, SafeGapAdsBlender};
use crate::models::query::ScoredPostsQuery;
use crate::params::{AdsBlenderType, PROMPTS_POSITION, WHO_TO_FOLLOW_POSITION};
use xai_candidate_pipeline::selector::{SelectResult, Selector};
use xai_home_mixer_proto::{
    FeedItem, Prompt, PushToHomePost, ScoredPost, WhoToFollowModule, feed_item,
};
use xai_recsys_proto::AdIndexInfo;

pub struct BlenderSelector {
    safe_gap_blender: SafeGapAdsBlender,
    partition_organic_blender: PartitionOrganicAdsBlender,
}

impl BlenderSelector {
    pub fn new() -> Self {
        Self {
            safe_gap_blender: SafeGapAdsBlender,
            partition_organic_blender: PartitionOrganicAdsBlender,
        }
    }
}

impl Selector<ScoredPostsQuery, FeedItem> for BlenderSelector {
    fn select(
        &self,
        query: &ScoredPostsQuery,
        candidates: Vec<FeedItem>,
    ) -> SelectResult<FeedItem> {
        let PartitionedFeedItems {
            posts,
            ads,
            wtf_modules,
            prompts,
            push_to_home,
        } = partition_feed_items(candidates);

        let input_post_count = posts.len();
        let input_ad_count = ads.len();

        let blender_type = query.params.get(AdsBlenderType);
        let blender: &dyn AdsBlender = match blender_type.as_str() {
            "safe_gap" => &self.safe_gap_blender,
            _ => &self.partition_organic_blender,
        };

        let mut blended = blender.blend(posts, ads);

        insert_prompts(&mut blended, prompts);
        insert_who_to_follow(&mut blended, wtf_modules);
        pin_push_to_home(&mut blended, push_to_home);

        let output_post_count = blended
            .iter()
            .filter(|i| matches!(i.item, Some(feed_item::Item::Post(_))))
            .count();
        let output_ad_count = blended
            .iter()
            .filter(|i| matches!(i.item, Some(feed_item::Item::Ad(_))))
            .count();

        let dropped_posts = input_post_count.saturating_sub(output_post_count);
        let dropped_ads = input_ad_count.saturating_sub(output_ad_count);
        let non_selected = build_non_selected_placeholders(dropped_posts, dropped_ads);

        SelectResult {
            selected: blended,
            non_selected,
        }
    }

    fn score(&self, _candidate: &FeedItem) -> f64 {
        0.0
    }
}

/// Prompts are injections — follow suggestions, upsells, nudges to turn something on. Every one
/// the injection service returned used to be inserted above every post in the response, so a
/// stack of them was the first thing in the feed. One per response, and never before the feed has
/// shown some of what was asked for.
const MAX_PROMPTS_PER_RESPONSE: usize = 1;
const MIN_POSTS_BEFORE_PROMPT: usize = 3;

fn insert_prompts(blended: &mut Vec<FeedItem>, prompts: Vec<Prompt>) {
    // Follows the who-to-follow pattern: the declared position is the slot, taken as a 1-based
    // index, and it is only ever pushed later by the floor — never earlier.
    let declared = PROMPTS_POSITION.max(1) as usize - 1;
    let index = declared.max(MIN_POSTS_BEFORE_PROMPT);

    for (offset, prompt) in prompts
        .into_iter()
        .take(MAX_PROMPTS_PER_RESPONSE)
        .enumerate()
    {
        blended.insert(
            (index + offset).min(blended.len()),
            FeedItem {
                position: PROMPTS_POSITION,
                item: Some(feed_item::Item::Prompt(prompt)),
            },
        );
    }
}

fn insert_who_to_follow(blended: &mut Vec<FeedItem>, wtf_modules: Vec<WhoToFollowModule>) {
    let Some(wtf) = wtf_modules.into_iter().next() else {
        return;
    };
    let insert_idx = WHO_TO_FOLLOW_POSITION.saturating_sub(1).min(blended.len());
    blended.insert(
        insert_idx,
        FeedItem {
            position: WHO_TO_FOLLOW_POSITION as i32,
            item: Some(feed_item::Item::WhoToFollow(wtf)),
        },
    );
}

fn pin_push_to_home(blended: &mut Vec<FeedItem>, push_to_home: Option<PushToHomePost>) {
    let Some(pth) = push_to_home else {
        return;
    };
    blended.insert(
        0,
        FeedItem {
            position: 0,
            item: Some(feed_item::Item::PushToHome(pth)),
        },
    );
}

fn build_non_selected_placeholders(dropped_posts: usize, dropped_ads: usize) -> Vec<FeedItem> {
    let mut non_selected = Vec::with_capacity(dropped_posts + dropped_ads);
    for _ in 0..dropped_posts {
        non_selected.push(FeedItem {
            position: 0,
            item: Some(feed_item::Item::Post(ScoredPost::default())),
        });
    }
    for _ in 0..dropped_ads {
        non_selected.push(FeedItem {
            position: 0,
            item: Some(feed_item::Item::Ad(AdIndexInfo::default())),
        });
    }
    non_selected
}

struct PartitionedFeedItems {
    posts: Vec<ScoredPost>,
    ads: Vec<AdIndexInfo>,
    wtf_modules: Vec<WhoToFollowModule>,
    prompts: Vec<Prompt>,
    push_to_home: Option<PushToHomePost>,
}

fn partition_feed_items(items: Vec<FeedItem>) -> PartitionedFeedItems {
    let mut posts = Vec::new();
    let mut ads = Vec::new();
    let mut wtf_modules = Vec::new();
    let mut prompts = Vec::new();
    let mut push_to_home = None;
    for item in items {
        match item.item {
            Some(feed_item::Item::Post(post)) => posts.push(post),
            Some(feed_item::Item::Ad(ad)) => ads.push(ad),
            Some(feed_item::Item::WhoToFollow(wtf)) => wtf_modules.push(wtf),
            Some(feed_item::Item::Prompt(prompt)) => prompts.push(prompt),
            Some(feed_item::Item::PushToHome(pth)) => push_to_home = Some(pth),
            None => {}
        }
    }
    PartitionedFeedItems {
        posts,
        ads,
        wtf_modules,
        prompts,
        push_to_home,
    }
}

#[cfg(test)]
mod tests {
    use super::{MAX_PROMPTS_PER_RESPONSE, MIN_POSTS_BEFORE_PROMPT, insert_prompts};
    use xai_home_mixer_proto::{FeedItem, Prompt, ScoredPost, feed_item};

    fn posts(count: usize) -> Vec<FeedItem> {
        (0..count)
            .map(|i| FeedItem {
                position: i as i32,
                item: Some(feed_item::Item::Post(ScoredPost::default())),
            })
            .collect()
    }

    fn prompt_indices(blended: &[FeedItem]) -> Vec<usize> {
        blended
            .iter()
            .enumerate()
            .filter(|(_, item)| matches!(item.item, Some(feed_item::Item::Prompt(_))))
            .map(|(index, _)| index)
            .collect()
    }

    #[test]
    fn only_one_prompt_reaches_the_response() {
        let mut blended = posts(20);
        insert_prompts(&mut blended, vec![Prompt::default(); 4]);

        assert_eq!(prompt_indices(&blended).len(), MAX_PROMPTS_PER_RESPONSE);
    }

    #[test]
    fn the_prompt_never_comes_before_the_first_posts() {
        let mut blended = posts(20);
        insert_prompts(&mut blended, vec![Prompt::default()]);

        assert!(prompt_indices(&blended)[0] >= MIN_POSTS_BEFORE_PROMPT);
    }

    #[test]
    fn a_response_shorter_than_the_lead_puts_the_prompt_last() {
        let mut blended = posts(2);
        insert_prompts(&mut blended, vec![Prompt::default()]);

        assert_eq!(prompt_indices(&blended), vec![2]);
        assert_eq!(blended.len(), 3);
    }

    #[test]
    fn no_prompts_leaves_the_response_untouched() {
        let mut blended = posts(5);
        insert_prompts(&mut blended, vec![]);

        assert!(prompt_indices(&blended).is_empty());
        assert_eq!(blended.len(), 5);
    }
}
