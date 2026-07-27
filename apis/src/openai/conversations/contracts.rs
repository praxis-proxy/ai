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

/// Optional response fields supported by Conversation item endpoints.
///
/// The spellings match OpenAI's `IncludeEnum` exactly. Runtime query parsing
/// and generated `OpenAPI` both consume this enum.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, ToSchema)]
pub(super) enum IncludeField {
    /// Include file search result payloads.
    #[serde(rename = "file_search_call.results")]
    #[schema(rename = "file_search_call.results")]
    FileSearchCallResults,
    /// Include web search result payloads.
    #[serde(rename = "web_search_call.results")]
    #[schema(rename = "web_search_call.results")]
    WebSearchCallResults,
    /// Include the sources used by web search actions.
    #[serde(rename = "web_search_call.action.sources")]
    #[schema(rename = "web_search_call.action.sources")]
    WebSearchCallActionSources,
    /// Include image URLs in message input-image parts.
    #[serde(rename = "message.input_image.image_url")]
    #[schema(rename = "message.input_image.image_url")]
    MessageInputImageImageUrl,
    /// Include image URLs in computer-call outputs.
    #[serde(rename = "computer_call_output.output.image_url")]
    #[schema(rename = "computer_call_output.output.image_url")]
    ComputerCallOutputImageUrl,
    /// Include code-interpreter output payloads.
    #[serde(rename = "code_interpreter_call.outputs")]
    #[schema(rename = "code_interpreter_call.outputs")]
    CodeInterpreterCallOutputs,
    /// Include encrypted reasoning content.
    #[serde(rename = "reasoning.encrypted_content")]
    #[schema(rename = "reasoning.encrypted_content")]
    ReasoningEncryptedContent,
    /// Include token log probabilities in message output-text parts.
    #[serde(rename = "message.output_text.logprobs")]
    #[schema(rename = "message.output_text.logprobs")]
    MessageOutputTextLogprobs,
}

impl IncludeField {
    /// Parse one decoded query value using the official enum spelling.
    pub(super) fn parse(value: &str) -> Option<Self> {
        match value {
            "file_search_call.results" => Some(Self::FileSearchCallResults),
            "web_search_call.results" => Some(Self::WebSearchCallResults),
            "web_search_call.action.sources" => Some(Self::WebSearchCallActionSources),
            "message.input_image.image_url" => Some(Self::MessageInputImageImageUrl),
            "computer_call_output.output.image_url" => Some(Self::ComputerCallOutputImageUrl),
            "code_interpreter_call.outputs" => Some(Self::CodeInterpreterCallOutputs),
            "reasoning.encrypted_content" => Some(Self::ReasoningEncryptedContent),
            "message.output_text.logprobs" => Some(Self::MessageOutputTextLogprobs),
            _ => None,
        }
    }

    /// Return this field's bit in the compact runtime include set.
    const fn bit(self) -> u8 {
        match self {
            Self::FileSearchCallResults => 1 << 0,
            Self::WebSearchCallResults => 1 << 1,
            Self::WebSearchCallActionSources => 1 << 2,
            Self::MessageInputImageImageUrl => 1 << 3,
            Self::ComputerCallOutputImageUrl => 1 << 4,
            Self::CodeInterpreterCallOutputs => 1 << 5,
            Self::ReasoningEncryptedContent => 1 << 6,
            Self::MessageOutputTextLogprobs => 1 << 7,
        }
    }
}

/// Set of requested optional item fields.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct IncludeFields(u8);

impl IncludeFields {
    /// Add one requested field.
    pub(super) fn insert(&mut self, field: IncludeField) {
        self.0 |= field.bit();
    }

    /// Return whether a field was requested.
    pub(super) const fn contains(self, field: IncludeField) -> bool {
        self.0 & field.bit() != 0
    }
}

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
