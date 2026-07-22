// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Request handlers for the `/v1/conversations` endpoints.

use std::{borrow::Cow, collections::HashSet, fmt, marker::PhantomData};

use percent_encoding::percent_decode_str;
use praxis_filter::{FilterAction, FilterError, HttpFilterContext, Rejection};
use serde::{
    Deserializer as _, Serialize,
    de::{DeserializeOwned, MapAccess, Visitor, value::MapAccessDeserializer},
};
use serde_json::{Map, Value};
use tracing::debug;

use super::{
    contracts::{
        ConversationItem, ConversationItemList, ConversationResource, CreateConversationItemsRequest,
        CreateConversationRequest, DeletedConversationResource, IncludeField, IncludeFields, ItemOrder,
        MAX_ITEMS_PER_REQUEST, Metadata, MetadataUpdate, UpdateConversationRequest,
    },
    validate::validate_metadata,
};
use crate::{
    openai::responses::{
        DEFAULT_TENANT_ID, TENANT_METADATA_KEY,
        store::{DEFAULT_PAGE_LIMIT, MAX_PAGE_LIMIT},
    },
    store::{ConversationItemRecord, ConversationItemStore, ConversationRecord, StoreError},
};

// -----------------------------------------------------------------------------
// ItemListParams
// -----------------------------------------------------------------------------

/// Cursor pagination parameters for conversation item listing.
struct ItemListParams {
    /// Item ID to page after.
    after_item_id: Option<String>,

    /// Maximum number of items to return.
    limit: u32,

    /// Result ordering.
    order: ItemOrder,
}

impl Default for ItemListParams {
    fn default() -> Self {
        Self {
            after_item_id: None,
            limit: DEFAULT_PAGE_LIMIT,
            order: ItemOrder::default(),
        }
    }
}

impl ItemListParams {
    /// Return the effective limit clamped to the API bounds.
    fn effective_limit(&self) -> u32 {
        self.limit.clamp(1, MAX_PAGE_LIMIT)
    }
}

// -----------------------------------------------------------------------------
// Conversation Lifecycle
// -----------------------------------------------------------------------------

/// Handle `POST /v1/conversations` — create a new conversation.
#[expect(clippy::too_many_lines, reason = "sequential guard-clause pipeline")]
pub(super) async fn handle_create_conversation(
    ctx: &HttpFilterContext<'_>,
    store: &dyn ConversationItemStore,
    body: &[u8],
) -> Result<FilterAction, FilterError> {
    let tenant_id = ctx.get_metadata(TENANT_METADATA_KEY).unwrap_or(DEFAULT_TENANT_ID);
    let input: CreateConversationRequest = match parse_json_body(body) {
        Ok(v) => v,
        Err(msg) => return Ok(FilterAction::Reject(invalid_input_response(&msg)?)),
    };
    let metadata = match input.metadata {
        Some(metadata) => {
            if let Err(msg) = validate_metadata(metadata.as_value()) {
                return Ok(FilterAction::Reject(invalid_input_response(&msg)?));
            }
            metadata.into_value()
        },
        None => Value::Object(Map::new()),
    };

    let raw_id = ctx.id_generator.generate(ctx.time_source);
    let conversation_id = format!("conv_{raw_id}");
    let created_at = current_timestamp(ctx);
    if let Err(msg) = validate_item_count(input.items.len()) {
        return Ok(FilterAction::Reject(invalid_input_response(&msg)?));
    }
    let item_values = input.items.into_iter().map(ConversationItem::into_value);
    let item_records = match build_item_records(ctx, tenant_id, &conversation_id, created_at, 1, item_values) {
        Ok(records) => records,
        Err(msg) => return Ok(FilterAction::Reject(invalid_input_response(&msg)?)),
    };
    if let Some(item_id) = duplicate_item_id(&item_records) {
        return Ok(FilterAction::Reject(invalid_input_response(
            &duplicate_item_id_message(item_id),
        )?));
    }
    let messages = Value::Array(item_records.iter().map(|item| item.item_data.clone()).collect());

    let record = ConversationRecord {
        conversation_id: conversation_id.clone(),
        tenant_id: tenant_id.to_owned(),
        created_at,
        metadata,
        messages,
    };

    if let Err(e) = store.upsert_conversation(&record).await {
        return Ok(FilterAction::Reject(store_error_response(&e)?));
    }
    if !item_records.is_empty()
        && let Err(e) = store.create_conversation_items(&item_records).await
    {
        return Ok(FilterAction::Reject(store_error_response(&e)?));
    }
    debug!(conversation_id, tenant_id, "conversation created");

    let body = conversation_response(record);
    Ok(FilterAction::Reject(json_response(200, &body)?))
}

/// Handle `GET /v1/conversations/{id}` — retrieve a conversation.
pub(super) async fn handle_get_conversation(
    ctx: &HttpFilterContext<'_>,
    store: &dyn ConversationItemStore,
    conversation_id: &str,
) -> Result<FilterAction, FilterError> {
    let tenant_id = ctx.get_metadata(TENANT_METADATA_KEY).unwrap_or(DEFAULT_TENANT_ID);

    match store.get_conversation(tenant_id, conversation_id).await {
        Ok(Some(record)) => {
            let body = conversation_response(record);
            Ok(FilterAction::Reject(json_response(200, &body)?))
        },
        Ok(None) => {
            debug!(conversation_id, "conversation not found");
            Ok(FilterAction::Reject(not_found_response(&format!(
                "No conversation found with id: '{conversation_id}'."
            ))?))
        },
        Err(e) => Ok(FilterAction::Reject(store_error_response(&e)?)),
    }
}

/// Handle `POST /v1/conversations/{id}` — update a conversation.
#[expect(clippy::too_many_lines, reason = "sequential guard-clause pipeline")]
pub(super) async fn handle_update_conversation(
    ctx: &HttpFilterContext<'_>,
    store: &dyn ConversationItemStore,
    conversation_id: &str,
    body: &[u8],
) -> Result<FilterAction, FilterError> {
    let tenant_id = ctx.get_metadata(TENANT_METADATA_KEY).unwrap_or(DEFAULT_TENANT_ID);
    let input: UpdateConversationRequest = match parse_json_body(body) {
        Ok(v) => v,
        Err(msg) => return Ok(FilterAction::Reject(invalid_input_response(&msg)?)),
    };
    if let MetadataUpdate::Replace(metadata) = &input.metadata
        && let Err(msg) = validate_metadata(metadata.as_value())
    {
        return Ok(FilterAction::Reject(invalid_input_response(&msg)?));
    }

    let existing = match store.get_conversation(tenant_id, conversation_id).await {
        Ok(record) => record,
        Err(e) => return Ok(FilterAction::Reject(store_error_response(&e)?)),
    };
    let Some(existing) = existing else {
        debug!(conversation_id, "conversation not found for update");
        return Ok(FilterAction::Reject(not_found_response(&format!(
            "No conversation found with id: '{conversation_id}'."
        ))?));
    };

    let metadata = match input.metadata {
        MetadataUpdate::Missing => existing.metadata,
        MetadataUpdate::Clear => Value::Object(Map::new()),
        MetadataUpdate::Replace(metadata) => metadata.into_value(),
    };

    let record = ConversationRecord {
        conversation_id: conversation_id.to_owned(),
        tenant_id: tenant_id.to_owned(),
        created_at: existing.created_at,
        metadata,
        messages: existing.messages,
    };

    if let Err(e) = store.upsert_conversation(&record).await {
        return Ok(FilterAction::Reject(store_error_response(&e)?));
    }
    debug!(conversation_id, tenant_id, "conversation updated");

    let body = conversation_response(record);
    Ok(FilterAction::Reject(json_response(200, &body)?))
}

/// Handle `DELETE /v1/conversations/{id}` — delete a conversation.
///
/// This intentionally deletes only the conversation record. The OpenAI
/// Conversations API specifies that deleting a conversation does not delete
/// its items; item cleanup belongs to item deletion or a separate retention
/// policy, not this endpoint.
pub(super) async fn handle_delete_conversation(
    ctx: &HttpFilterContext<'_>,
    store: &dyn ConversationItemStore,
    conversation_id: &str,
) -> Result<FilterAction, FilterError> {
    let tenant_id = ctx.get_metadata(TENANT_METADATA_KEY).unwrap_or(DEFAULT_TENANT_ID);

    match store.delete_conversation(tenant_id, conversation_id).await {
        Ok(true) => {
            debug!(conversation_id, tenant_id, "conversation deleted");
            let body = DeletedConversationResource::deleted(conversation_id);
            Ok(FilterAction::Reject(json_response(200, &body)?))
        },
        Ok(false) => {
            debug!(conversation_id, "conversation not found for delete");
            Ok(FilterAction::Reject(not_found_response(&format!(
                "No conversation found with id: '{conversation_id}'."
            ))?))
        },
        Err(e) => Ok(FilterAction::Reject(store_error_response(&e)?)),
    }
}

// -----------------------------------------------------------------------------
// Conversation Items
// -----------------------------------------------------------------------------

/// Handle `POST /v1/conversations/{id}/items` — create items.
#[expect(clippy::too_many_lines, reason = "sequential guard-clause pipeline")]
#[expect(clippy::large_stack_frames, reason = "Pingora context types are large")]
pub(super) async fn handle_create_items(
    ctx: &HttpFilterContext<'_>,
    store: &dyn ConversationItemStore,
    conversation_id: &str,
    body: &[u8],
) -> Result<FilterAction, FilterError> {
    let tenant_id = ctx.get_metadata(TENANT_METADATA_KEY).unwrap_or(DEFAULT_TENANT_ID);
    let input: CreateConversationItemsRequest = match parse_json_body(body) {
        Ok(v) => v,
        Err(msg) => return Ok(FilterAction::Reject(invalid_input_response(&msg)?)),
    };
    let includes = match parse_include_fields(ctx.request.uri.query()) {
        Ok(includes) => includes,
        Err(msg) => return Ok(FilterAction::Reject(invalid_input_response(&msg)?)),
    };
    let existing = match store.get_conversation(tenant_id, conversation_id).await {
        Ok(record) => record,
        Err(e) => return Ok(FilterAction::Reject(store_error_response(&e)?)),
    };
    let Some(existing) = existing else {
        debug!(conversation_id, "conversation not found for item create");
        return Ok(FilterAction::Reject(not_found_response(
            &conversation_not_found_message(conversation_id),
        )?));
    };

    let Some(items) = input.items else {
        return Ok(FilterAction::Reject(invalid_input_response("'items' is required")?));
    };
    if let Err(msg) = validate_item_count(items.len()) {
        return Ok(FilterAction::Reject(invalid_input_response(&msg)?));
    }
    let item_values = items.into_iter().map(ConversationItem::into_value);
    let start_position = match store.max_item_position(tenant_id, conversation_id).await {
        Ok(pos) => pos.saturating_add(1),
        Err(e) => return Ok(FilterAction::Reject(store_error_response(&e)?)),
    };
    let created_at = current_timestamp(ctx);
    let item_records =
        match build_item_records(ctx, tenant_id, conversation_id, created_at, start_position, item_values) {
            Ok(records) => records,
            Err(msg) => return Ok(FilterAction::Reject(invalid_input_response(&msg)?)),
        };
    if let Some(item_id) = duplicate_item_id(&item_records) {
        return Ok(FilterAction::Reject(invalid_input_response(
            &duplicate_item_id_message(item_id),
        )?));
    }
    let requested_ids: Vec<&str> = item_records.iter().map(|r| r.item_id.as_str()).collect();
    let already_present = match store
        .get_existing_conversation_item_ids(tenant_id, conversation_id, &requested_ids)
        .await
    {
        Ok(ids) => ids,
        Err(e) => return Ok(FilterAction::Reject(store_error_response(&e)?)),
    };
    if let Some(item_id) = already_present.first() {
        return Ok(FilterAction::Reject(invalid_input_response(
            &existing_item_id_message(item_id),
        )?));
    }

    if let Err(e) = store.create_conversation_items(&item_records).await {
        return Ok(FilterAction::Reject(store_error_response(&e)?));
    }
    if let Err(e) = sync_conversation_messages(store, existing).await {
        return Ok(FilterAction::Reject(store_error_response(&e)?));
    }
    debug!(
        conversation_id,
        tenant_id,
        count = item_records.len(),
        "conversation items created"
    );

    let body = conversation_items_response(item_records, false, includes);
    Ok(FilterAction::Reject(json_response(200, &body)?))
}

/// Handle `GET /v1/conversations/{id}/items` — list items.
#[expect(clippy::too_many_lines, reason = "sequential guard-clause pipeline")]
pub(super) async fn handle_list_items(
    ctx: &HttpFilterContext<'_>,
    store: &dyn ConversationItemStore,
    conversation_id: &str,
) -> Result<FilterAction, FilterError> {
    let tenant_id = ctx.get_metadata(TENANT_METADATA_KEY).unwrap_or(DEFAULT_TENANT_ID);
    let includes = match parse_include_fields(ctx.request.uri.query()) {
        Ok(includes) => includes,
        Err(msg) => return Ok(FilterAction::Reject(invalid_input_response(&msg)?)),
    };
    match store.get_conversation(tenant_id, conversation_id).await {
        Ok(Some(_)) => {},
        Ok(None) => {
            debug!(conversation_id, "conversation not found for item list");
            return Ok(FilterAction::Reject(not_found_response(
                &conversation_not_found_message(conversation_id),
            )?));
        },
        Err(e) => return Ok(FilterAction::Reject(store_error_response(&e)?)),
    }

    let params = parse_item_list_params(ctx.request.uri.query());
    let limit = params.effective_limit();
    let rows = match store
        .list_conversation_items(
            tenant_id,
            conversation_id,
            params.after_item_id.as_deref(),
            limit.saturating_add(1),
            params.order.is_ascending(),
        )
        .await
    {
        Ok(rows) => rows,
        Err(e) => return Ok(FilterAction::Reject(store_error_response(&e)?)),
    };
    let take_limit = usize::try_from(limit).unwrap_or(usize::MAX);
    let has_more = rows.len() > take_limit;
    let data: Vec<_> = rows.into_iter().take(take_limit).collect();

    let body = conversation_items_response(data, has_more, includes);
    Ok(FilterAction::Reject(json_response(200, &body)?))
}

/// Handle `GET /v1/conversations/{id}/items/{item_id}` — retrieve one item.
pub(super) async fn handle_get_item(
    ctx: &HttpFilterContext<'_>,
    store: &dyn ConversationItemStore,
    conversation_id: &str,
    item_id: &str,
) -> Result<FilterAction, FilterError> {
    let tenant_id = ctx.get_metadata(TENANT_METADATA_KEY).unwrap_or(DEFAULT_TENANT_ID);
    let includes = match parse_include_fields(ctx.request.uri.query()) {
        Ok(includes) => includes,
        Err(msg) => return Ok(FilterAction::Reject(invalid_input_response(&msg)?)),
    };
    let item_id = match decode_item_id_path_segment(item_id) {
        Ok(id) => id,
        Err(msg) => return Ok(FilterAction::Reject(invalid_input_response(&msg)?)),
    };
    let item_id = item_id.as_ref();
    match store.get_conversation_item(tenant_id, conversation_id, item_id).await {
        Ok(Some(record)) => {
            let mut item_data = record.item_data;
            project_conversation_item(&mut item_data, includes);
            let item = ConversationItem::from_value(item_data);
            Ok(FilterAction::Reject(json_response(200, &item)?))
        },
        Ok(None) => {
            debug!(conversation_id, item_id, "conversation item not found");
            Ok(FilterAction::Reject(not_found_response(&item_not_found_message(
                item_id,
            ))?))
        },
        Err(e) => Ok(FilterAction::Reject(store_error_response(&e)?)),
    }
}

/// Handle `DELETE /v1/conversations/{id}/items/{item_id}` — delete one item.
#[expect(clippy::too_many_lines, reason = "sequential guard-clause pipeline")]
#[expect(clippy::cognitive_complexity, reason = "tracing macros inflate complexity")]
pub(super) async fn handle_delete_item(
    ctx: &HttpFilterContext<'_>,
    store: &dyn ConversationItemStore,
    conversation_id: &str,
    item_id: &str,
) -> Result<FilterAction, FilterError> {
    let tenant_id = ctx.get_metadata(TENANT_METADATA_KEY).unwrap_or(DEFAULT_TENANT_ID);
    let item_id = match decode_item_id_path_segment(item_id) {
        Ok(id) => id,
        Err(msg) => return Ok(FilterAction::Reject(invalid_input_response(&msg)?)),
    };
    let item_id = item_id.as_ref();
    let existing = match store.get_conversation(tenant_id, conversation_id).await {
        Ok(Some(record)) => record,
        Ok(None) => {
            debug!(conversation_id, item_id, "conversation not found for item delete");
            return Ok(FilterAction::Reject(not_found_response(
                &conversation_not_found_message(conversation_id),
            )?));
        },
        Err(e) => return Ok(FilterAction::Reject(store_error_response(&e)?)),
    };

    match store
        .delete_conversation_item(tenant_id, conversation_id, item_id)
        .await
    {
        Ok(true) => {
            if let Err(e) = sync_conversation_messages(store, existing).await {
                return Ok(FilterAction::Reject(store_error_response(&e)?));
            }
            debug!(conversation_id, item_id, tenant_id, "conversation item deleted");
            match store.get_conversation(tenant_id, conversation_id).await {
                Ok(Some(record)) => {
                    let body = conversation_response(record);
                    Ok(FilterAction::Reject(json_response(200, &body)?))
                },
                Ok(None) => Ok(FilterAction::Reject(not_found_response(
                    &conversation_not_found_message(conversation_id),
                )?)),
                Err(e) => Ok(FilterAction::Reject(store_error_response(&e)?)),
            }
        },
        Ok(false) => {
            debug!(conversation_id, item_id, "conversation item not found for delete");
            Ok(FilterAction::Reject(not_found_response(&item_not_found_message(
                item_id,
            ))?))
        },
        Err(e) => Ok(FilterAction::Reject(store_error_response(&e)?)),
    }
}

// -----------------------------------------------------------------------------
// JSON Helpers
// -----------------------------------------------------------------------------

/// Parse a request body into its runtime contract.
fn parse_json_body<T: DeserializeOwned>(body: &[u8]) -> Result<T, String> {
    let mut deserializer = serde_json::Deserializer::from_slice(body);
    let value = deserializer
        .deserialize_map(JsonObjectVisitor(PhantomData))
        .map_err(|e| format!("invalid JSON body: {e}"))?;
    deserializer.end().map_err(|e| format!("invalid JSON body: {e}"))?;
    Ok(value)
}

/// Deserialize a typed contract only from a top-level JSON object.
struct JsonObjectVisitor<T>(PhantomData<T>);

impl<'de, T: DeserializeOwned> Visitor<'de> for JsonObjectVisitor<T> {
    type Value = T;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON object")
    }

    fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<Self::Value, A::Error> {
        T::deserialize(MapAccessDeserializer::new(map))
    }
}

/// Validate the shared item-count bound after deserialization.
fn validate_item_count(item_count: usize) -> Result<(), String> {
    if item_count > MAX_ITEMS_PER_REQUEST {
        return Err(format!("items may contain at most {MAX_ITEMS_PER_REQUEST} entries"));
    }
    Ok(())
}

/// Return the first duplicate item ID in a create request.
fn duplicate_item_id(items: &[ConversationItemRecord]) -> Option<&str> {
    let mut seen = HashSet::new();
    for item in items {
        if !seen.insert(item.item_id.as_str()) {
            return Some(item.item_id.as_str());
        }
    }
    None
}

/// Build store records for normalized conversation item JSON values.
#[expect(clippy::too_many_arguments, reason = "factoring into struct would add indirection")]
pub(super) fn build_item_records(
    ctx: &HttpFilterContext<'_>,
    tenant_id: &str,
    conversation_id: &str,
    created_at: i64,
    start_position: i64,
    items: impl IntoIterator<Item = Value>,
) -> Result<Vec<ConversationItemRecord>, String> {
    items
        .into_iter()
        .enumerate()
        .map(|(index, item)| {
            let (item_id, item_data) = normalize_item(ctx, item)?;
            let offset = i64::try_from(index).unwrap_or(i64::MAX);
            Ok(ConversationItemRecord {
                item_id,
                tenant_id: tenant_id.to_owned(),
                conversation_id: conversation_id.to_owned(),
                item_data,
                created_at,
                position: start_position.saturating_add(offset),
            })
        })
        .collect()
}

/// Ensure an item is an object and has a usable ID.
pub(super) fn normalize_item(ctx: &HttpFilterContext<'_>, item: Value) -> Result<(String, Value), String> {
    let Value::Object(mut map) = item else {
        return Err("each item must be a JSON object".to_owned());
    };
    let item_id = match map.get("id") {
        Some(Value::String(id)) if !id.is_empty() => id.clone(),
        Some(Value::String(_)) => return Err("item id must not be empty".to_owned()),
        Some(Value::Null) | None => generated_item_id(ctx),
        Some(_) => return Err("item id must be a string".to_owned()),
    };
    map.insert("id".to_owned(), Value::String(item_id.clone()));
    normalize_message_item(&mut map)?;
    Ok((item_id, Value::Object(map)))
}

/// Normalize easy SDK message inputs into conversation message response objects.
fn normalize_message_item(map: &mut Map<String, Value>) -> Result<(), String> {
    if map.get("type").and_then(Value::as_str) != Some("message") {
        return Ok(());
    }

    let role = match map.get("role") {
        Some(Value::String(role)) if !role.is_empty() => role.clone(),
        Some(Value::String(_)) => return Err("message role must not be empty".to_owned()),
        Some(_) => return Err("message role must be a string".to_owned()),
        None => return Err("message role is required".to_owned()),
    };

    let content = map
        .remove("content")
        .ok_or_else(|| "message content is required".to_owned())?;
    map.insert("content".to_owned(), normalize_message_content(&role, content)?);
    map.entry("status".to_owned())
        .or_insert_with(|| Value::String("completed".to_owned()));

    Ok(())
}

/// Convert string message content to the list-form content returned by the API.
fn normalize_message_content(role: &str, content: Value) -> Result<Value, String> {
    match content {
        Value::String(text) => {
            let content_item = if role == "assistant" {
                serde_json::json!({
                    "type": "output_text",
                    "text": text,
                    "annotations": [],
                })
            } else {
                serde_json::json!({
                    "type": "input_text",
                    "text": text,
                })
            };
            Ok(Value::Array(vec![content_item]))
        },
        Value::Array(_) => Ok(content),
        _ => Err("message content must be a string or array".to_owned()),
    }
}

/// Generate a conversation item ID.
pub(super) fn generated_item_id(ctx: &HttpFilterContext<'_>) -> String {
    let raw_id = ctx.id_generator.generate(ctx.time_source);
    format!("item_{raw_id}")
}

/// Decode an item ID path segment the same way clients encode path parameters.
fn decode_item_id_path_segment(item_id: &str) -> Result<Cow<'_, str>, String> {
    percent_decode_str(item_id)
        .decode_utf8()
        .map_err(|e| format!("item id path segment must be valid UTF-8: {e}"))
}

/// Move a stored conversation into its public response contract.
fn conversation_response(record: ConversationRecord) -> ConversationResource {
    ConversationResource::new(
        record.conversation_id,
        record.created_at,
        Metadata::from_value(record.metadata),
    )
}

/// Move item records into an `OpenAI` list response without copying item JSON.
fn conversation_items_response(
    records: Vec<ConversationItemRecord>,
    has_more: bool,
    includes: IncludeFields,
) -> ConversationItemList {
    let record_count = records.len();
    let mut first_id = String::new();
    let mut last_id = String::new();
    let mut data = Vec::with_capacity(record_count);

    for (index, record) in records.into_iter().enumerate() {
        if record_count == 1 {
            first_id.clone_from(&record.item_id);
            last_id = record.item_id;
        } else if index == 0 {
            first_id = record.item_id;
        } else if index + 1 == record_count {
            last_id = record.item_id;
        }
        let mut item_data = record.item_data;
        project_conversation_item(&mut item_data, includes);
        data.push(ConversationItem::from_value(item_data));
    }

    ConversationItemList::new(data, has_more, first_id, last_id)
}

/// Remove optional fields that were not requested through `include`.
///
/// Projection changes only the response-owned value after the complete item
/// representation has crossed the storage boundary.
fn project_conversation_item(item: &mut Value, includes: IncludeFields) {
    let Some(object) = item.as_object_mut() else {
        return;
    };
    match projection_kind(object) {
        ProjectionKind::Reasoning => remove_unless_included(
            object,
            "encrypted_content",
            includes.contains(IncludeField::ReasoningEncryptedContent),
        ),
        ProjectionKind::FileSearch => remove_unless_included(
            object,
            "results",
            includes.contains(IncludeField::FileSearchCallResults),
        ),
        ProjectionKind::WebSearch => project_web_search_fields(object, includes),
        ProjectionKind::CodeInterpreter => remove_unless_included(
            object,
            "outputs",
            includes.contains(IncludeField::CodeInterpreterCallOutputs),
        ),
        ProjectionKind::ComputerOutput => project_computer_output_fields(object, includes),
        ProjectionKind::Message => project_message_fields(object, includes),
        ProjectionKind::Other => {},
    }
}

/// Item variants with fields controlled by `include`.
#[derive(Clone, Copy)]
enum ProjectionKind {
    /// Reasoning item with optional encrypted content.
    Reasoning,
    /// File-search call with optional results.
    FileSearch,
    /// Web-search call with optional results and sources.
    WebSearch,
    /// Code-interpreter call with optional outputs.
    CodeInterpreter,
    /// Computer-call output with an optional image URL.
    ComputerOutput,
    /// Message with optional fields in typed content parts.
    Message,
    /// Item without any fields controlled by `include`.
    Other,
}

/// Classify an item without retaining a borrow into the mutable object.
fn projection_kind(object: &Map<String, Value>) -> ProjectionKind {
    match object.get("type").and_then(Value::as_str) {
        Some("reasoning") => ProjectionKind::Reasoning,
        Some("file_search_call") => ProjectionKind::FileSearch,
        Some("web_search_call") => ProjectionKind::WebSearch,
        Some("code_interpreter_call") => ProjectionKind::CodeInterpreter,
        Some("computer_call_output") => ProjectionKind::ComputerOutput,
        Some("message") => ProjectionKind::Message,
        _ => ProjectionKind::Other,
    }
}

/// Remove one top-level field unless it was explicitly requested.
fn remove_unless_included(object: &mut Map<String, Value>, field: &str, included: bool) {
    if !included {
        object.remove(field);
    }
}

/// Project web-search fields controlled by independent include values.
fn project_web_search_fields(object: &mut Map<String, Value>, includes: IncludeFields) {
    remove_unless_included(object, "results", includes.contains(IncludeField::WebSearchCallResults));
    if !includes.contains(IncludeField::WebSearchCallActionSources)
        && let Some(action) = object.get_mut("action").and_then(Value::as_object_mut)
    {
        action.remove("sources");
    }
}

/// Project the nested image URL from a computer-call output.
fn project_computer_output_fields(object: &mut Map<String, Value>, includes: IncludeFields) {
    if !includes.contains(IncludeField::ComputerCallOutputImageUrl)
        && let Some(output) = object.get_mut("output").and_then(Value::as_object_mut)
    {
        output.remove("image_url");
    }
}

/// Project optional fields from typed message content parts.
fn project_message_fields(object: &mut Map<String, Value>, includes: IncludeFields) {
    let Some(content) = object.get_mut("content").and_then(Value::as_array_mut) else {
        return;
    };
    for part in content {
        let Some(part) = part.as_object_mut() else {
            continue;
        };
        if part.get("type").and_then(Value::as_str) == Some("input_image")
            && !includes.contains(IncludeField::MessageInputImageImageUrl)
        {
            part.remove("image_url");
        } else if part.get("type").and_then(Value::as_str) == Some("output_text")
            && !includes.contains(IncludeField::MessageOutputTextLogprobs)
        {
            part.remove("logprobs");
        }
    }
}

/// Parse both official SDK encodings for the array-valued `include` query:
/// repeated `include=value` pairs and bracketed `include[]=value` pairs.
fn parse_include_fields(query: Option<&str>) -> Result<IncludeFields, String> {
    let Some(query) = query else {
        return Ok(IncludeFields::default());
    };

    let mut includes = IncludeFields::default();
    for pair in query.split('&') {
        let Some((raw_key, raw_value)) = pair.split_once('=') else {
            let key = decode_query_component_strict(pair)?;
            if matches!(key.as_ref(), "include" | "include[]") {
                return Err("'include' query parameter requires a value".to_owned());
            }
            continue;
        };
        let key = decode_query_component_strict(raw_key)?;
        if !matches!(key.as_ref(), "include" | "include[]") {
            continue;
        }
        let value = decode_query_component_strict(raw_value)?;
        let field = IncludeField::parse(&value).ok_or_else(|| format!("unsupported include value: '{value}'"))?;
        includes.insert(field);
    }
    Ok(includes)
}

/// Strictly decode one query component, including form-style `+` spaces.
fn decode_query_component_strict(value: &str) -> Result<Cow<'_, str>, String> {
    if value.contains('+') {
        let normalized = value.replace('+', " ");
        return percent_decode_str(&normalized)
            .decode_utf8()
            .map(|decoded| Cow::Owned(decoded.into_owned()))
            .map_err(|e| format!("query parameter must be valid UTF-8: {e}"));
    }
    percent_decode_str(value)
        .decode_utf8()
        .map_err(|e| format!("query parameter must be valid UTF-8: {e}"))
}

/// Parse cursor-based pagination parameters from a query string.
fn parse_item_list_params(query: Option<&str>) -> ItemListParams {
    let Some(qs) = query else {
        return ItemListParams::default();
    };

    let mut params = ItemListParams::default();
    for pair in qs.split('&') {
        let Some((key, value)) = pair.split_once('=') else {
            continue;
        };
        match key {
            "after" => {
                params.after_item_id = Some(decode_query_component(value));
            },
            "limit" => {
                if let Ok(n) = value.parse::<u32>() {
                    params.limit = n;
                }
            },
            "order" => match value {
                "asc" => params.order = ItemOrder::Asc,
                "desc" => params.order = ItemOrder::Desc,
                _ => {},
            },
            _ => {},
        }
    }
    params
}

/// Decode one application/x-www-form-urlencoded query component.
fn decode_query_component(value: &str) -> String {
    let normalized = value.replace('+', " ");
    percent_decode_str(&normalized).decode_utf8_lossy().into_owned()
}

/// Return the current Unix timestamp as an `i64`.
pub(super) fn current_timestamp(ctx: &HttpFilterContext<'_>) -> i64 {
    i64::try_from(ctx.time_source.now().as_secs()).unwrap_or(i64::MAX)
}

/// Build a JSON response with the given status code.
fn json_response<T: Serialize + ?Sized>(status: u16, body: &T) -> Result<Rejection, FilterError> {
    let bytes = serde_json::to_vec(body)
        .map_err(|e| FilterError::from(format!("openai_conversations: serialize failed: {e}")))?;
    Ok(Rejection::status(status)
        .with_header("content-type", "application/json")
        .with_body(bytes))
}

/// Build a 400 JSON response for invalid input.
fn invalid_input_response(message: &str) -> Result<Rejection, FilterError> {
    json_response(
        400,
        &serde_json::json!({
            "error": {
                "message": message,
                "type": "invalid_request_error",
            }
        }),
    )
}

/// Build a 404 JSON response.
fn not_found_response(message: &str) -> Result<Rejection, FilterError> {
    json_response(
        404,
        &serde_json::json!({
            "error": {
                "message": message,
                "type": "invalid_request_error",
            }
        }),
    )
}

/// Build the standard conversation not-found message.
fn conversation_not_found_message(conversation_id: &str) -> String {
    format!("No conversation found with id: '{conversation_id}'.")
}

/// Build the standard item not-found message.
fn item_not_found_message(item_id: &str) -> String {
    format!("No conversation item found with id: '{item_id}'.")
}

/// Build a duplicate-item client error message.
fn duplicate_item_id_message(item_id: &str) -> String {
    format!("duplicate item id in request: '{item_id}'")
}

/// Build an existing-item client error message.
fn existing_item_id_message(item_id: &str) -> String {
    format!("item id already exists in conversation: '{item_id}'")
}

/// Build a 500 JSON response from a store error.
fn store_error_response(error: &StoreError) -> Result<Rejection, FilterError> {
    let message = match error {
        StoreError::InvalidInput(msg) => {
            return json_response(
                400,
                &serde_json::json!({
                    "error": {
                        "message": msg,
                        "type": "invalid_request_error",
                    }
                }),
            );
        },
        _ => "Internal server error.",
    };
    json_response(
        500,
        &serde_json::json!({
            "error": {
                "message": message,
                "type": "server_error",
            }
        }),
    )
}

/// Refresh the denormalized conversation message cache from item rows.
///
/// This currently re-reads all items on every mutation. Conversations are not
/// assumed to be small: the OpenAI contract has no cumulative item or byte
/// ceiling. Replace this full-history rebuild with incremental processing; do
/// not add a non-spec conversation limit as a workaround. Tracked in #532.
pub(super) async fn sync_conversation_messages(
    store: &dyn ConversationItemStore,
    record: ConversationRecord,
) -> Result<(), StoreError> {
    let messages =
        Value::Array(collect_conversation_messages(store, &record.tenant_id, &record.conversation_id).await?);
    let updated = store
        .update_conversation_messages(&record.tenant_id, &record.conversation_id, &messages)
        .await?;
    if updated {
        Ok(())
    } else {
        Err(StoreError::Database(format!(
            "conversation disappeared during message sync: {}",
            record.conversation_id
        )))
    }
}

/// Collect all item JSON values for a conversation in ascending order.
async fn collect_conversation_messages(
    store: &dyn ConversationItemStore,
    tenant_id: &str,
    conversation_id: &str,
) -> Result<Vec<Value>, StoreError> {
    let mut after = None;
    let mut messages = Vec::new();
    loop {
        let rows = store
            .list_conversation_items(tenant_id, conversation_id, after.as_deref(), MAX_PAGE_LIMIT, true)
            .await?;
        if rows.is_empty() {
            break;
        }
        after = rows.last().map(|record| record.item_id.clone());
        let row_count = rows.len();
        messages.extend(rows.into_iter().map(|record| record.item_data));
        if row_count < usize::try_from(MAX_PAGE_LIMIT).unwrap_or(usize::MAX) {
            break;
        }
    }
    Ok(messages)
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing, reason = "tests")]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // store_error_response
    // -------------------------------------------------------------------------

    #[test]
    fn store_error_invalid_input_returns_400() {
        let error = StoreError::InvalidInput("bad cursor".to_owned());
        let rejection = store_error_response(&error).unwrap();
        assert_eq!(rejection.status, 400);
        let body: Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["message"], "bad cursor");
    }

    #[test]
    fn store_error_database_returns_500() {
        let error = StoreError::Database("connection lost".to_owned());
        let rejection = store_error_response(&error).unwrap();
        assert_eq!(rejection.status, 500);
        let body: Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();
        assert_eq!(body["error"]["type"], "server_error");
        assert_eq!(body["error"]["message"], "Internal server error.");
    }

    // -------------------------------------------------------------------------
    // parse_item_list_params
    // -------------------------------------------------------------------------

    #[test]
    fn parse_params_skips_pair_without_separator() {
        let params = parse_item_list_params(Some("noseparator&limit=5"));
        assert_eq!(params.limit, 5);
        assert!(params.after_item_id.is_none());
    }

    #[test]
    fn parse_params_unknown_order_stays_default() {
        let params = parse_item_list_params(Some("order=random"));
        assert!(
            !params.order.is_ascending(),
            "unknown order should keep default descending"
        );
    }

    #[test]
    fn parse_params_non_numeric_limit_uses_default() {
        let params = parse_item_list_params(Some("limit=abc"));
        assert_eq!(params.limit, DEFAULT_PAGE_LIMIT);
    }

    // -------------------------------------------------------------------------
    // decode_query_component / decode_item_id_path_segment
    // -------------------------------------------------------------------------

    #[test]
    fn decode_query_component_invalid_utf8_uses_lossy() {
        let result = decode_query_component("%FF%FE");
        assert!(
            result.contains('\u{FFFD}'),
            "invalid UTF-8 should produce replacement characters"
        );
    }

    #[test]
    fn decode_item_id_path_segment_invalid_utf8_returns_error() {
        let result = decode_item_id_path_segment("%FF%FE");
        assert!(result.is_err(), "invalid UTF-8 should return error");
        assert!(
            result.unwrap_err().contains("valid UTF-8"),
            "error should mention UTF-8 requirement"
        );
    }

    // -------------------------------------------------------------------------
    // ItemListParams::effective_limit
    // -------------------------------------------------------------------------

    #[test]
    fn effective_limit_clamps_zero_to_one() {
        let params = ItemListParams {
            limit: 0,
            ..ItemListParams::default()
        };
        assert_eq!(params.effective_limit(), 1);
    }

    #[test]
    fn effective_limit_clamps_above_max() {
        let params = ItemListParams {
            limit: MAX_PAGE_LIMIT + 50,
            ..ItemListParams::default()
        };
        assert_eq!(params.effective_limit(), MAX_PAGE_LIMIT);
    }

    #[test]
    fn effective_limit_returns_value_within_range() {
        let params = ItemListParams {
            limit: 50,
            ..ItemListParams::default()
        };
        assert_eq!(params.effective_limit(), 50);
    }

    // -------------------------------------------------------------------------
    // store_error_response — catch-all variants
    // -------------------------------------------------------------------------

    #[test]
    fn store_error_serialization_returns_500() {
        let error = StoreError::Serialization("corrupt data".to_owned());
        let rejection = store_error_response(&error).unwrap();
        assert_eq!(rejection.status, 500);
        let body: Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();
        assert_eq!(body["error"]["type"], "server_error");
        assert_eq!(body["error"]["message"], "Internal server error.");
    }

    #[test]
    fn store_error_unavailable_returns_500() {
        let error = StoreError::Unavailable("not connected".to_owned());
        let rejection = store_error_response(&error).unwrap();
        assert_eq!(rejection.status, 500);
        let body: Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();
        assert_eq!(body["error"]["type"], "server_error");
        assert_eq!(body["error"]["message"], "Internal server error.");
    }

    // -------------------------------------------------------------------------
    // parse_item_list_params — additional edges
    // -------------------------------------------------------------------------

    #[test]
    fn parse_params_none_query_returns_defaults() {
        let params = parse_item_list_params(None);
        assert_eq!(params.limit, DEFAULT_PAGE_LIMIT);
        assert!(!params.order.is_ascending());
        assert!(params.after_item_id.is_none());
    }

    #[test]
    fn parse_params_valid_after_parameter() {
        let params = parse_item_list_params(Some("after=item_abc123&limit=10"));
        assert_eq!(params.after_item_id.as_deref(), Some("item_abc123"));
        assert_eq!(params.limit, 10);
    }

    #[test]
    fn parse_params_asc_order() {
        let params = parse_item_list_params(Some("order=asc"));
        assert!(params.order.is_ascending(), "order=asc should set ascending");
    }

    #[test]
    fn parse_params_desc_order() {
        let params = parse_item_list_params(Some("order=desc"));
        assert!(!params.order.is_ascending(), "order=desc should set descending");
    }

    #[test]
    fn parse_params_negative_limit_uses_default() {
        let params = parse_item_list_params(Some("limit=-5"));
        assert_eq!(
            params.limit, DEFAULT_PAGE_LIMIT,
            "negative limit should not parse as u32"
        );
    }

    #[test]
    fn parse_params_percent_encoded_after() {
        let params = parse_item_list_params(Some("after=item%20with+space"));
        assert_eq!(
            params.after_item_id.as_deref(),
            Some("item with space"),
            "percent-encoded and plus-encoded values should decode"
        );
    }

    // -------------------------------------------------------------------------
    // include parsing and projection
    // -------------------------------------------------------------------------

    #[test]
    fn parse_include_fields_supports_python_and_node_sdk_encodings() {
        let includes = parse_include_fields(Some(
            "include=reasoning.encrypted_content&include%5B%5D=message.output_text.logprobs",
        ))
        .unwrap();

        assert!(
            includes.contains(IncludeField::ReasoningEncryptedContent),
            "repeated-key encoding should parse reasoning encrypted content"
        );
        assert!(
            includes.contains(IncludeField::MessageOutputTextLogprobs),
            "bracket encoding should parse output-text log probabilities"
        );
        assert!(
            !includes.contains(IncludeField::FileSearchCallResults),
            "unrequested include values must remain absent"
        );
    }

    #[test]
    fn parse_include_fields_rejects_unknown_or_malformed_values() {
        let unknown = parse_include_fields(Some("include=future.secret_field")).unwrap_err();
        assert!(
            unknown.contains("unsupported include value"),
            "unknown values should produce an unsupported-value diagnostic: {unknown}"
        );

        let missing = parse_include_fields(Some("include")).unwrap_err();
        assert!(
            missing.contains("requires a value"),
            "missing include values should identify the required value: {missing}"
        );

        let invalid_utf8 = parse_include_fields(Some("include=%FF")).unwrap_err();
        assert!(
            invalid_utf8.contains("valid UTF-8"),
            "invalid encoding should identify the UTF-8 requirement: {invalid_utf8}"
        );
    }

    #[test]
    #[expect(clippy::too_many_lines, reason = "one fixture covers every include projection path")]
    fn projection_removes_every_unrequested_include_gated_field() {
        let mut items = vec![
            serde_json::json!({
                "type": "reasoning",
                "encrypted_content": "secret",
                "summary": []
            }),
            serde_json::json!({
                "type": "file_search_call",
                "results": [{"file_id": "file_1"}],
                "status": "completed"
            }),
            serde_json::json!({
                "type": "web_search_call",
                "results": [{"url": "https://example.com"}],
                "action": {
                    "type": "search",
                    "sources": [{"type": "url", "url": "https://example.com"}]
                }
            }),
            serde_json::json!({
                "type": "code_interpreter_call",
                "outputs": [{"type": "logs", "logs": "done"}],
                "status": "completed"
            }),
            serde_json::json!({
                "type": "computer_call_output",
                "output": {"type": "computer_screenshot", "image_url": "data:image/png;base64,AA=="}
            }),
            serde_json::json!({
                "type": "message",
                "content": [
                    {"type": "input_image", "image_url": "https://example.com/image.png", "detail": "auto"},
                    {"type": "output_text", "text": "answer", "annotations": [], "logprobs": []},
                    {"type": "input_text", "text": "keep me"}
                ]
            }),
        ];

        for item in &mut items {
            project_conversation_item(item, IncludeFields::default());
        }

        assert!(
            items[0].get("encrypted_content").is_none(),
            "reasoning encrypted content should be omitted"
        );
        assert!(
            items[1].get("results").is_none(),
            "file-search results should be omitted"
        );
        assert!(
            items[2].get("results").is_none(),
            "web-search results should be omitted"
        );
        assert!(
            items[2]["action"].get("sources").is_none(),
            "web-search action sources should be omitted"
        );
        assert!(
            items[3].get("outputs").is_none(),
            "code-interpreter outputs should be omitted"
        );
        assert!(
            items[4]["output"].get("image_url").is_none(),
            "computer-output image URLs should be omitted"
        );
        assert!(
            items[5]["content"][0].get("image_url").is_none(),
            "message input-image URLs should be omitted"
        );
        assert!(
            items[5]["content"][1].get("logprobs").is_none(),
            "message output-text log probabilities should be omitted"
        );
        assert_eq!(items[5]["content"][2]["text"], "keep me");
    }

    #[test]
    fn projection_preserves_every_requested_include_gated_field() {
        let mut includes = IncludeFields::default();
        for field in [
            IncludeField::FileSearchCallResults,
            IncludeField::WebSearchCallResults,
            IncludeField::WebSearchCallActionSources,
            IncludeField::MessageInputImageImageUrl,
            IncludeField::ComputerCallOutputImageUrl,
            IncludeField::CodeInterpreterCallOutputs,
            IncludeField::ReasoningEncryptedContent,
            IncludeField::MessageOutputTextLogprobs,
        ] {
            includes.insert(field);
        }
        let original = serde_json::json!({
            "type": "message",
            "content": [
                {"type": "input_image", "image_url": "https://example.com/image.png"},
                {"type": "output_text", "logprobs": [{"token": "x"}]}
            ]
        });
        let mut projected = original.clone();

        project_conversation_item(&mut projected, includes);

        assert_eq!(projected, original);
    }

    // -------------------------------------------------------------------------
    // decode_item_id_path_segment — additional cases
    // -------------------------------------------------------------------------

    #[test]
    fn decode_item_id_plain_ascii_passes_through() {
        let result = decode_item_id_path_segment("item_abc123").unwrap();
        assert_eq!(result.as_ref(), "item_abc123");
    }

    #[test]
    fn decode_item_id_percent_encoded_ascii() {
        let result = decode_item_id_path_segment("item%5Fabc").unwrap();
        assert_eq!(result.as_ref(), "item_abc", "percent-encoded underscore should decode");
    }
}
