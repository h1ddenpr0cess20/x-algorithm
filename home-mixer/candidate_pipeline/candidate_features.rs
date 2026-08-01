use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub struct PureCoreData {
    pub author_id: u64,
    pub text: String,
    pub source_tweet_id: Option<u64>,
    pub source_user_id: Option<u64>,
    pub in_reply_to_tweet_id: Option<u64>,
    pub in_reply_to_user_id: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub struct ExclusiveTweetControl {
    pub conversation_author_id: i64,
}

pub type MediaEntities = Vec<MediaEntity>;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub struct MediaEntity {
    pub media_info: Option<MediaInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub enum MediaInfo {
    VideoInfo(VideoInfo),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub struct VideoInfo {
    pub duration_millis: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub struct Share {
    pub source_tweet_id: u64,
    pub source_user_id: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub struct Reply {
    pub in_reply_to_tweet_id: Option<u64>,
    pub in_reply_to_user_id: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub struct GizmoduckUserCounts {
    pub followers_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub struct GizmoduckUserProfile {
    pub screen_name: String,
}

/// The affiliate badge: the parent-organization mark beside a handle, granted and revoked by that
/// organization through Verified Organizations. It is the organization's own statement that this
/// account belongs to it, which is why the feed policy keys on it rather than on a list of names
/// somebody here has to maintain.
///
/// This is the one struct in this file whose wire shape cannot be checked against its source: the
/// Gizmoduck client that decides which field groups are requested is not part of this repository.
/// It is therefore written to be unable to break the rest of the user: only the parent handle is
/// named, and only as a string. Anything else in the payload — a nested link, a badge image, a
/// parent id — is ignored rather than parsed, because a field named here with the wrong type
/// would fail the whole `GizmoduckUser` and cost every candidate its screen name and follower
/// count. A payload that carries the handle somewhere else leaves this `None`, which means no
/// post is deranked and `FeedPolicyScorer` reports zero badges observed. Whoever wires the client
/// maps the real field into `screen_name`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub struct GizmoduckAffiliateLabel {
    #[serde(default, alias = "parentScreenName")]
    pub screen_name: Option<String>,
}

impl GizmoduckAffiliateLabel {
    /// The parent organization's handle, normalized: lowercase, no leading '@'.
    pub fn parent_handle(&self) -> Option<String> {
        self.screen_name
            .as_deref()
            .map(|handle| handle.trim().trim_start_matches('@').to_ascii_lowercase())
            .filter(|handle| !handle.is_empty())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub struct GizmoduckUser {
    pub user_id: u64,
    pub profile: GizmoduckUserProfile,
    pub counts: GizmoduckUserCounts,
    #[serde(default, alias = "affiliatesHighlightedLabel")]
    pub affiliate_label: Option<GizmoduckAffiliateLabel>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub struct GizmoduckUserResult {
    pub user: Option<GizmoduckUser>,
}

#[cfg(test)]
mod tests {
    use super::{GizmoduckAffiliateLabel, GizmoduckUser};

    fn label(screen_name: Option<&str>) -> GizmoduckAffiliateLabel {
        GizmoduckAffiliateLabel {
            screen_name: screen_name.map(str::to_string),
        }
    }

    #[test]
    fn the_parent_handle_is_normalized() {
        assert_eq!(label(Some("@Tesla")).parent_handle(), Some("tesla".into()));
        assert_eq!(
            label(Some(" SpaceX ")).parent_handle(),
            Some("spacex".into())
        );
    }

    #[test]
    fn an_empty_label_yields_no_handle() {
        assert_eq!(label(None).parent_handle(), None);
        assert_eq!(label(Some("  ")).parent_handle(), None);
    }

    #[test]
    fn a_user_payload_without_the_field_still_deserializes() {
        // The field is absent for almost every account, and absent entirely if the client never
        // projects it. Neither case may fail the whole user.
        let user: GizmoduckUser = serde_json::from_str(
            r#"{"userId":7,"profile":{"screenName":"someone"},"counts":{"followersCount":3}}"#,
        )
        .expect("a user without an affiliate label should deserialize");

        assert!(user.affiliate_label.is_none());
    }

    #[test]
    fn both_spellings_of_the_handle_are_accepted() {
        for body in [
            r#"{"userId":7,"profile":{"screenName":"a"},"counts":{"followersCount":1},"affiliateLabel":{"screenName":"Tesla"}}"#,
            r#"{"userId":7,"profile":{"screenName":"a"},"counts":{"followersCount":1},"affiliatesHighlightedLabel":{"parentScreenName":"@Tesla"}}"#,
        ] {
            let user: GizmoduckUser = serde_json::from_str(body).expect("label should deserialize");

            assert_eq!(
                user.affiliate_label.and_then(|label| label.parent_handle()),
                Some("tesla".to_string()),
                "{body}"
            );
        }
    }

    #[test]
    fn an_unrecognized_label_shape_costs_the_badge_and_nothing_else() {
        // The shape this is most likely to arrive in if it arrives at all: a nested object whose
        // handle sits somewhere this projection does not name. The user still deserializes.
        let body = r#"{"userId":7,"profile":{"screenName":"a"},"counts":{"followersCount":1},
            "affiliateLabel":{"badge":{"url":"https://pbs.x.com/badge.png"},
            "url":{"url":"https://x.com/Tesla","urlType":"DeepLink"}}}"#;

        let user: GizmoduckUser =
            serde_json::from_str(body).expect("an unknown label shape must not fail the user");

        assert_eq!(user.profile.screen_name, "a");
        assert_eq!(
            user.affiliate_label.and_then(|label| label.parent_handle()),
            None
        );
    }
}
