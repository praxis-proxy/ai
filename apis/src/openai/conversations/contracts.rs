// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Runtime JSON contracts for locally handled Conversations operations.

#![expect(
    clippy::large_stack_frames,
    reason = "utoipa macro-generated schema builders allocate large temporary values"
)]

use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use utoipa::{
    ToSchema,
    openapi::schema::{Object, ObjectBuilder, Type},
};

/// Maximum number of items accepted by create operations.
pub(super) const MAX_ITEMS_PER_REQUEST: usize = 20;

/// Request body accepted by `POST /conversations`.
#[derive(Debug, Deserialize, ToSchema)]
pub(super) struct CreateConversationRequest {
    /// Optional metadata map. Missing and null both produce empty metadata.
    pub(super) metadata: Option<Metadata>,

    /// Optional initial items to add to the conversation.
    #[serde(default)]
    #[schema(max_items = 20)]
    pub(super) items: Vec<ConversationItem>,
}

/// Request body accepted by `POST /conversations/{conversation_id}`.
#[derive(Debug, Deserialize, ToSchema)]
pub(super) struct UpdateConversationRequest {
    /// Optional metadata replacement. Null clears existing metadata.
    #[serde(default)]
    #[schema(value_type = Option<Metadata>)]
    pub(super) metadata: MetadataUpdate,
}

/// Request body accepted by `POST /conversations/{conversation_id}/items`.
#[derive(Debug, Deserialize, ToSchema)]
pub(super) struct CreateConversationItemsRequest {
    /// Items to create.
    #[serde(default)]
    #[schema(value_type = Vec<ConversationItem>, required = true, max_items = 20)]
    pub(super) items: Option<Vec<ConversationItem>>,
}

/// Metadata supplied with a conversation.
///
/// The runtime keeps the original JSON object ordering. Validation enforces
/// string values before the value crosses into storage.
#[derive(Debug, Deserialize, Serialize, ToSchema)]
#[serde(transparent)]
#[schema(value_type = std::collections::BTreeMap<String, String>)]
pub(super) struct Metadata(Value);

impl Metadata {
    /// Borrow the underlying JSON value for validation.
    pub(super) const fn as_value(&self) -> &Value {
        &self.0
    }

    /// Move the underlying JSON value into storage.
    pub(super) fn into_value(self) -> Value {
        self.0
    }

    /// Wrap metadata read from storage for response serialization.
    pub(super) const fn from_value(value: Value) -> Self {
        Self(value)
    }
}

/// Metadata update semantics for the update operation.
#[derive(Debug, Default)]
pub(super) enum MetadataUpdate {
    /// The metadata field was absent; preserve the stored value.
    #[default]
    Missing,
    /// The metadata field was null; clear the stored value.
    Clear,
    /// Replace the stored metadata with this value.
    Replace(Metadata),
}

impl<'de> Deserialize<'de> for MetadataUpdate {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Option::<Metadata>::deserialize(deserializer).map(|metadata| match metadata {
            Some(metadata) => Self::Replace(metadata),
            None => Self::Clear,
        })
    }
}

/// Polymorphic conversation item stored and returned as an opaque JSON object.
///
/// Message-specific normalization happens in the handler. Other item kinds are
/// deliberately preserved so new provider variants do not require proxy code
/// changes.
#[derive(Debug, Deserialize, Serialize, ToSchema)]
#[serde(transparent)]
#[schema(value_type = Object)]
pub(super) struct ConversationItem(Value);

impl ConversationItem {
    /// Move an input item into runtime normalization.
    pub(super) fn into_value(self) -> Value {
        self.0
    }

    /// Wrap a stored item for response serialization.
    pub(super) const fn from_value(value: Value) -> Self {
        Self(value)
    }
}

/// Local conversation response object.
#[derive(Debug, Serialize, ToSchema)]
pub(super) struct ConversationResource {
    /// Conversation ID.
    id: String,
    /// Object discriminator.
    #[schema(schema_with = conversation_object_schema)]
    object: ConversationObject,
    /// Creation timestamp measured in seconds since the Unix epoch.
    #[schema(format = "unixtime")]
    created_at: i64,
    /// Conversation metadata.
    metadata: Metadata,
}

impl ConversationResource {
    /// Construct a conversation response from runtime-owned fields.
    pub(super) const fn new(id: String, created_at: i64, metadata: Metadata) -> Self {
        Self {
            id,
            object: ConversationObject::Conversation,
            created_at,
            metadata,
        }
    }
}

/// Delete conversation response object.
#[derive(Debug, Serialize, ToSchema)]
pub(super) struct DeletedConversationResource {
    /// Conversation ID.
    id: String,
    /// Object discriminator.
    #[schema(schema_with = deleted_conversation_object_schema)]
    object: DeletedConversationObject,
    /// Whether the object was deleted.
    deleted: bool,
}

impl DeletedConversationResource {
    /// Construct a successful delete response.
    pub(super) fn deleted(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            object: DeletedConversationObject::ConversationDeleted,
            deleted: true,
        }
    }
}

/// Conversation item list response object.
#[derive(Debug, Serialize, ToSchema)]
pub(super) struct ConversationItemList {
    /// Object discriminator.
    #[schema(schema_with = list_object_schema)]
    object: ListObject,
    /// Conversation items.
    data: Vec<ConversationItem>,
    /// Whether more items are available.
    has_more: bool,
    /// First item ID in this page.
    first_id: String,
    /// Last item ID in this page.
    last_id: String,
}

impl ConversationItemList {
    /// Construct one page of conversation items.
    pub(super) const fn new(data: Vec<ConversationItem>, has_more: bool, first_id: String, last_id: String) -> Self {
        Self {
            object: ListObject::List,
            data,
            has_more,
            first_id,
            last_id,
        }
    }
}

/// Conversation object discriminator.
#[derive(Debug, Serialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub(super) enum ConversationObject {
    /// Conversation resource.
    Conversation,
}

/// Deleted conversation object discriminator.
#[derive(Debug, Serialize, ToSchema)]
pub(super) enum DeletedConversationObject {
    /// Deleted conversation resource.
    #[serde(rename = "conversation.deleted")]
    #[schema(rename = "conversation.deleted")]
    ConversationDeleted,
}

/// List object discriminator.
#[derive(Debug, Serialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub(super) enum ListObject {
    /// List resource.
    List,
}

/// Supported item list ordering.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub(super) enum ItemOrder {
    /// Oldest item first.
    Asc,
    /// Newest item first.
    #[default]
    Desc,
}

impl ItemOrder {
    /// Whether records should be returned oldest-first.
    pub(super) const fn is_ascending(self) -> bool {
        matches!(self, Self::Asc)
    }
}

/// Generate the fixed conversation discriminator schema.
fn conversation_object_schema() -> Object {
    fixed_string_schema("conversation", true)
}

/// Generate the fixed deleted-conversation discriminator schema.
fn deleted_conversation_object_schema() -> Object {
    fixed_string_schema("conversation.deleted", true)
}

/// Generate the fixed list discriminator schema.
fn list_object_schema() -> Object {
    fixed_string_schema("list", false)
}

/// Build an inline string schema for a single discriminator value.
fn fixed_string_schema(value: &str, include_default: bool) -> Object {
    let mut schema = ObjectBuilder::new()
        .schema_type(Type::String)
        .enum_values(Some([value]));
    if include_default {
        schema = schema.default(Some(Value::String(value.to_owned())));
    }
    schema.build()
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests")]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn create_request_distinguishes_missing_and_null_items() {
        let missing: CreateConversationRequest = serde_json::from_value(json!({})).unwrap();
        assert!(missing.items.is_empty());

        let null = serde_json::from_value::<CreateConversationRequest>(json!({"items": null}));
        assert!(null.is_err(), "explicit null items must remain invalid");
    }

    #[test]
    fn update_request_preserves_metadata_field_state() {
        let missing: UpdateConversationRequest = serde_json::from_value(json!({})).unwrap();
        assert!(matches!(missing.metadata, MetadataUpdate::Missing));

        let null: UpdateConversationRequest = serde_json::from_value(json!({"metadata": null})).unwrap();
        assert!(matches!(null.metadata, MetadataUpdate::Clear));

        let replacement: UpdateConversationRequest =
            serde_json::from_value(json!({"metadata": {"project": "praxis"}})).unwrap();
        assert!(matches!(replacement.metadata, MetadataUpdate::Replace(_)));
    }

    #[test]
    fn conversation_item_preserves_unknown_object_variants() {
        let value = json!({"type": "future_provider_item", "provider_data": {"enabled": true}});
        let item: ConversationItem = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(serde_json::to_value(item).unwrap(), value);
    }
}
