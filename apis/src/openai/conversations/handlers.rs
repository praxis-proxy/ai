// SPDX-License-Identifier: Apache-2.0
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
use tracing::{debug, warn};

use super::{
    contracts::{
        ConversationItem, ConversationItemList, ConversationResource, CreateConversationItemsRequest,
        CreateConversationRequest, DeletedConversationResource, ItemOrder, MAX_ITEMS_PER_REQUEST, Metadata,
        UpdateConversationRequest,
    },
    validate::{MetadataError, validate_metadata},
};
use crate::{
    openai::{
        include::{IncludeFields, decode_query_component_strict, parse_include, project_item},
        responses::{
            DEFAULT_TENANT_ID, TENANT_METADATA_KEY,
            store::{DEFAULT_PAGE_LIMIT, MAX_PAGE_LIMIT},
        },
    },
    store::{ConversationItemRecord, ConversationItemStore, ConversationRecord, StoreError},
};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Maximum compare-and-swap retries before [`sync_conversation_messages`] gives
/// up on refreshing the cache. Each failed swap means another writer refreshed
/// the cache first from the same authoritative rows, so a small bound absorbs
/// realistic append contention.
const MAX_SYNC_ATTEMPTS: usize = 8;

// -----------------------------------------------------------------------------
// ItemListParams
// -----------------------------------------------------------------------------

/// Cursor pagination parameters for conversation item listing.
#[derive(Debug)]
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
    let input = if body.is_empty() {
        CreateConversationRequest::default()
    } else {
        match parse_json_body(body) {
            Ok(v) => v,
            Err(msg) => return Ok(FilterAction::Reject(invalid_input_response(&msg)?)),
        }
    };
    let metadata = match input.metadata {
        Some(metadata) => {
            if let Err(e) = validate_metadata(metadata.as_value()) {
                return Ok(FilterAction::Reject(invalid_input_response(&e.to_string())?));
            }
            metadata.into_value()
        },
        None => Value::Object(Map::new()),
    };

    let raw_id = ctx.id_generator.generate(ctx.time_source);
    let conversation_id = format!("conv_{raw_id}");
    let created_at = current_timestamp(ctx);
    let items = input.items.unwrap_or_default();
    if let Err(msg) = validate_item_count(items.len()) {
        return Ok(FilterAction::Reject(invalid_input_response(&msg)?));
    }
    let item_values = items.into_iter().map(ConversationItem::into_value);
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
    let conversation_id = match decoded_path_param("conversation id", conversation_id) {
        Ok(id) => id,
        Err(action) => return action,
    };
    let conversation_id = conversation_id.as_ref();

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
    let conversation_id = match decoded_path_param("conversation id", conversation_id) {
        Ok(id) => id,
        Err(action) => return action,
    };
    let conversation_id = conversation_id.as_ref();
    if body.is_empty() {
        return Ok(FilterAction::Reject(invalid_input_response_with(
            "Missing required parameter: 'metadata'.",
            Some("missing_required_parameter"),
            Some("metadata"),
        )?));
    }
    let input: UpdateConversationRequest = match parse_json_body(body) {
        Ok(v) => v,
        Err(msg) => {
            return Ok(FilterAction::Reject(classify_update_error(&msg)?));
        },
    };
    if let Err(e) = validate_metadata(input.metadata.as_value()) {
        return Ok(FilterAction::Reject(match e {
            MetadataError::InvalidType(_) => {
                invalid_input_response_with(&e.to_string(), Some("invalid_type"), Some("metadata"))?
            },
            MetadataError::ConstraintViolation(_) => invalid_input_response(&e.to_string())?,
        }));
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

    let metadata = input.metadata.into_value();

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
    let conversation_id = match decoded_path_param("conversation id", conversation_id) {
        Ok(id) => id,
        Err(action) => return action,
    };
    let conversation_id = conversation_id.as_ref();

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
pub(super) async fn handle_create_items(
    ctx: &HttpFilterContext<'_>,
    store: &dyn ConversationItemStore,
    conversation_id: &str,
    body: &[u8],
) -> Result<FilterAction, FilterError> {
    let tenant_id = ctx.get_metadata(TENANT_METADATA_KEY).unwrap_or(DEFAULT_TENANT_ID);
    let conversation_id = match decoded_path_param("conversation id", conversation_id) {
        Ok(id) => id,
        Err(action) => return action,
    };
    let conversation_id = conversation_id.as_ref();
    let input: CreateConversationItemsRequest = match parse_json_body(body) {
        Ok(v) => v,
        Err(msg) => return Ok(FilterAction::Reject(invalid_input_response(&msg)?)),
    };
    let includes = match parse_include(ctx.request.uri.query()) {
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
    if let Err(e) = sync_conversation_messages(store, tenant_id, conversation_id, Some(existing.messages)).await {
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
    let conversation_id = match decoded_path_param("conversation id", conversation_id) {
        Ok(id) => id,
        Err(action) => return action,
    };
    let conversation_id = conversation_id.as_ref();
    let includes = match parse_include(ctx.request.uri.query()) {
        Ok(includes) => includes,
        Err(msg) => return Ok(FilterAction::Reject(invalid_input_response(&msg)?)),
    };
    let params = match parse_item_list_params(ctx.request.uri.query()) {
        Ok(params) => params,
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

    let limit = params.limit;
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
#[expect(clippy::too_many_lines, reason = "decode both path parameters then look up")]
pub(super) async fn handle_get_item(
    ctx: &HttpFilterContext<'_>,
    store: &dyn ConversationItemStore,
    conversation_id: &str,
    item_id: &str,
) -> Result<FilterAction, FilterError> {
    let tenant_id = ctx.get_metadata(TENANT_METADATA_KEY).unwrap_or(DEFAULT_TENANT_ID);
    let conversation_id = match decoded_path_param("conversation id", conversation_id) {
        Ok(id) => id,
        Err(action) => return action,
    };
    let conversation_id = conversation_id.as_ref();
    let includes = match parse_include(ctx.request.uri.query()) {
        Ok(includes) => includes,
        Err(msg) => return Ok(FilterAction::Reject(invalid_input_response(&msg)?)),
    };
    let item_id = match decoded_path_param("item id", item_id) {
        Ok(id) => id,
        Err(action) => return action,
    };
    let item_id = item_id.as_ref();
    match store.get_conversation_item(tenant_id, conversation_id, item_id).await {
        Ok(Some(record)) => {
            let mut item_data = record.item_data;
            project_item(&mut item_data, includes);
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
    let conversation_id = match decoded_path_param("conversation id", conversation_id) {
        Ok(id) => id,
        Err(action) => return action,
    };
    let conversation_id = conversation_id.as_ref();
    let item_id = match decoded_path_param("item id", item_id) {
        Ok(id) => id,
        Err(action) => return action,
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
            if let Err(e) = sync_conversation_messages(store, tenant_id, conversation_id, Some(existing.messages)).await
            {
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

/// Decode a URI path parameter the same way clients encode path segments.
fn decode_path_segment<'a>(kind: &str, value: &'a str) -> Result<Cow<'a, str>, String> {
    percent_decode_str(value)
        .decode_utf8()
        .map_err(|e| format!("{kind} path segment must be valid UTF-8: {e}"))
}

/// Decode a path parameter or return the invalid-input rejection.
fn decoded_path_param<'a>(
    kind: &'static str,
    value: &'a str,
) -> Result<Cow<'a, str>, Result<FilterAction, FilterError>> {
    decode_path_segment(kind, value).map_err(|msg| invalid_input_response(&msg).map(FilterAction::Reject))
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
        project_item(&mut item_data, includes);
        data.push(ConversationItem::from_value(item_data));
    }

    ConversationItemList::new(data, has_more, first_id, last_id)
}

/// Parse and validate cursor-based pagination parameters from a query string.
#[expect(
    clippy::too_many_lines,
    reason = "query parser benefits from single-function locality"
)]
fn parse_item_list_params(query: Option<&str>) -> Result<ItemListParams, String> {
    let Some(qs) = query else {
        return Ok(ItemListParams::default());
    };

    let mut params = ItemListParams::default();
    let mut seen_limit = false;
    let mut seen_order = false;
    let mut seen_after = false;

    for pair in qs.split('&') {
        if pair.is_empty() {
            continue;
        }
        let Some((raw_key, raw_value)) = pair.split_once('=') else {
            let key = decode_query_component_strict(pair)?;
            if matches!(key.as_ref(), "include" | "include[]") {
                continue;
            }
            if matches!(key.as_ref(), "limit" | "order" | "after") {
                return Err(format!("Missing value for query parameter '{key}'."));
            }
            return Err(format!("Unknown query parameter: '{key}'."));
        };
        let key = decode_query_component_strict(raw_key)?;
        match key.as_ref() {
            "after" => {
                if seen_after {
                    return Err("Duplicate query parameter: 'after'.".to_owned());
                }
                seen_after = true;
                let value = decode_query_component_strict(raw_value)?;
                if value.is_empty() {
                    return Err("Invalid value for 'after': cursor must not be empty.".to_owned());
                }
                params.after_item_id = Some(value.into_owned());
            },
            "limit" => {
                if seen_limit {
                    return Err("Duplicate query parameter: 'limit'.".to_owned());
                }
                seen_limit = true;
                let value = decode_query_component_strict(raw_value)?;
                params.limit = parse_limit(&value)?;
            },
            "order" => {
                if seen_order {
                    return Err("Duplicate query parameter: 'order'.".to_owned());
                }
                seen_order = true;
                let value = decode_query_component_strict(raw_value)?;
                params.order = parse_order(&value)?;
            },
            "include" | "include[]" => {},
            _ => return Err(format!("Unknown query parameter: '{key}'.")),
        }
    }
    Ok(params)
}

/// Parse and validate a `limit` query-string value.
fn parse_limit(value: &str) -> Result<u32, String> {
    let n: u32 = value
        .parse()
        .map_err(|_e| format!("Invalid value for 'limit': '{value}' is not a valid integer."))?;
    if n > MAX_PAGE_LIMIT {
        return Err(format!(
            "Invalid value for 'limit': must be between 0 and {MAX_PAGE_LIMIT}, got {n}."
        ));
    }
    Ok(n)
}

/// Parse and validate an `order` query-string value.
fn parse_order(value: &str) -> Result<ItemOrder, String> {
    match value {
        "asc" => Ok(ItemOrder::Asc),
        "desc" => Ok(ItemOrder::Desc),
        _ => Err(format!(
            "Invalid value for 'order': must be 'asc' or 'desc', got '{value}'."
        )),
    }
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

/// Build a 400 JSON response with optional OpenAI error code and parameter.
fn invalid_input_response_with(
    message: &str,
    code: Option<&str>,
    param: Option<&str>,
) -> Result<Rejection, FilterError> {
    json_response(
        400,
        &serde_json::json!({
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": code,
                "param": param,
            }
        }),
    )
}

/// Map update deserialization errors to OpenAI-style error codes.
fn classify_update_error(msg: &str) -> Result<Rejection, FilterError> {
    if msg.contains("missing field") && msg.contains("metadata") {
        return invalid_input_response_with(
            "Missing required parameter: 'metadata'.",
            Some("missing_required_parameter"),
            Some("metadata"),
        );
    }
    if msg.contains("metadata must be an object") {
        return invalid_input_response_with(
            "Invalid type for 'metadata': expected an object.",
            Some("invalid_type"),
            Some("metadata"),
        );
    }
    invalid_input_response(msg)
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
/// The cache is rebuilt from the authoritative item rows and written back with
/// a compare-and-swap so a slower writer cannot clobber a newer cache with a
/// stale snapshot. Concurrent item appends each insert their rows and then
/// refresh this cache; without the compare-and-swap a slow writer could commit
/// an older `messages` snapshot after a newer writer, dropping the newer
/// writer's items from the denormalized history that rehydration consumes
/// (#662).
///
/// A caller that read the cache just before its row mutation passes that value
/// as `snapshot` to seed the first swap's expected value, skipping a redundant
/// reload; the value written is always a fresh rebuild, so the snapshot only
/// decides whether that first swap lands. Callers without a snapshot (the
/// response append-back path) pass `None`. On a conflict the snapshot is dropped
/// and later attempts re-read the live cache, so every committed row is visible
/// to the retry's rebuild and the cache converges to include all items.
///
/// Callers invoke this only after the authoritative item-row mutation is durably
/// committed, so the cache is a derived view that any later write rebuilds. If
/// the retries are exhausted under sustained contention this returns `Ok(())`
/// rather than an error: failing here would report an already-committed
/// create/delete as failed and drive the client into duplicate-item or
/// not-found retries. Genuine store errors — including the conversation
/// disappearing mid-sync — still propagate.
///
/// This currently re-reads all items on every mutation. Conversations are not
/// assumed to be small: the OpenAI contract has no cumulative item or byte
/// ceiling. Replace this full-history rebuild with incremental processing; do
/// not add a non-spec conversation limit as a workaround. Tracked in #532.
pub(super) async fn sync_conversation_messages(
    store: &dyn ConversationItemStore,
    tenant_id: &str,
    conversation_id: &str,
    mut snapshot: Option<Value>,
) -> Result<(), StoreError> {
    for _ in 0..MAX_SYNC_ATTEMPTS {
        if Box::pin(try_sync_conversation_messages(
            store,
            tenant_id,
            conversation_id,
            snapshot.take(),
        ))
        .await?
        {
            return Ok(());
        }
    }

    // The item-row mutation this refresh follows is already durably committed and
    // any later write rebuilds the cache, so abandoning the refresh must not fail
    // the request that already succeeded.
    warn!(
        conversation_id,
        attempts = MAX_SYNC_ATTEMPTS,
        "conversation message cache refresh abandoned after repeated compare-and-swap \
         contention; a later write will rebuild it"
    );
    Ok(())
}

/// Attempt one compare-and-swap refresh of the denormalized message cache.
///
/// `expected` seeds the swap's expected value: `Some` is the caller's snapshot,
/// read before its row mutation, which lets this attempt skip reloading the
/// cache; `None` re-reads the live cache (and reports a disappeared
/// conversation). Either way the value written is a fresh rebuild from the item
/// rows, so a stale `expected` only ever loses the swap — it never writes stale
/// data.
///
/// Returns `Ok(true)` when the cache is up to date — either already current or
/// swapped in this call — and `Ok(false)` when a concurrent writer won the swap
/// and the caller should retry with a freshly read snapshot.
async fn try_sync_conversation_messages(
    store: &dyn ConversationItemStore,
    tenant_id: &str,
    conversation_id: &str,
    expected: Option<Value>,
) -> Result<bool, StoreError> {
    // Optimistic first pass: the caller's pre-mutation snapshot is the expected
    // value, so no cache reload is needed. The rebuild below is still fresh, so a
    // snapshot that no longer matches only costs a lost swap and a retry.
    if let Some(expected) = expected {
        let messages = Value::Array(collect_conversation_messages(store, tenant_id, conversation_id).await?);
        return store
            .compare_and_swap_conversation_messages(tenant_id, conversation_id, &expected, &messages)
            .await;
    }

    // No snapshot: read the live cache as the expected value before rebuilding, so
    // the rebuilt view is at least as fresh as the cache it replaces and a
    // successful swap can never drop a committed row.
    let Some(current) = store.get_conversation(tenant_id, conversation_id).await? else {
        return Err(StoreError::Database(format!(
            "conversation disappeared during message sync: {conversation_id}"
        )));
    };
    let messages = Value::Array(collect_conversation_messages(store, tenant_id, conversation_id).await?);
    if current.messages == messages {
        return Ok(true);
    }
    store
        .compare_and_swap_conversation_messages(tenant_id, conversation_id, &current.messages, &messages)
        .await
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
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use async_trait::async_trait;

    use super::*;
    use crate::store::SqliteResponseStore;

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
    fn parse_params_unknown_key_only_rejected() {
        let err = parse_item_list_params(Some("noseparator&limit=5")).unwrap_err();
        assert!(
            err.contains("Unknown query parameter"),
            "unknown key-only component should be rejected: {err}"
        );
    }

    #[test]
    fn parse_params_unknown_order_rejected() {
        let err = parse_item_list_params(Some("order=random")).unwrap_err();
        assert!(
            err.contains("must be 'asc' or 'desc'"),
            "unknown order should be rejected: {err}"
        );
    }

    #[test]
    fn parse_params_non_numeric_limit_rejected() {
        let err = parse_item_list_params(Some("limit=abc")).unwrap_err();
        assert!(
            err.contains("not a valid integer"),
            "non-numeric limit should be rejected: {err}"
        );
    }

    // -------------------------------------------------------------------------
    // decode_path_segment
    // -------------------------------------------------------------------------

    #[test]
    fn decode_item_id_path_segment_invalid_utf8_returns_error() {
        let result = decode_path_segment("item id", "%FF%FE");
        assert!(result.is_err(), "invalid UTF-8 should return error");
        assert!(
            result.unwrap_err().contains("valid UTF-8"),
            "error should mention UTF-8 requirement"
        );
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
        let params = parse_item_list_params(None).unwrap();
        assert_eq!(params.limit, DEFAULT_PAGE_LIMIT);
        assert!(!params.order.is_ascending());
        assert!(params.after_item_id.is_none());
    }

    #[test]
    fn parse_params_valid_after_parameter() {
        let params = parse_item_list_params(Some("after=item_abc123&limit=10")).unwrap();
        assert_eq!(params.after_item_id.as_deref(), Some("item_abc123"));
        assert_eq!(params.limit, 10);
    }

    #[test]
    fn parse_params_asc_order() {
        let params = parse_item_list_params(Some("order=asc")).unwrap();
        assert!(params.order.is_ascending(), "order=asc should set ascending");
    }

    #[test]
    fn parse_params_desc_order() {
        let params = parse_item_list_params(Some("order=desc")).unwrap();
        assert!(!params.order.is_ascending(), "order=desc should set descending");
    }

    #[test]
    fn parse_params_negative_limit_rejected() {
        let err = parse_item_list_params(Some("limit=-5")).unwrap_err();
        assert!(
            err.contains("not a valid integer"),
            "negative limit should be rejected: {err}"
        );
    }

    #[test]
    fn parse_params_percent_encoded_after() {
        let params = parse_item_list_params(Some("after=item%20with+space")).unwrap();
        assert_eq!(
            params.after_item_id.as_deref(),
            Some("item with space"),
            "percent-encoded and plus-encoded values should decode"
        );
    }

    #[test]
    fn parse_params_limit_zero_accepted() {
        let params = parse_item_list_params(Some("limit=0")).unwrap();
        assert_eq!(params.limit, 0, "limit=0 should be accepted");
    }

    #[test]
    fn parse_params_limit_above_max_rejected() {
        let err = parse_item_list_params(Some(&format!("limit={}", MAX_PAGE_LIMIT + 1))).unwrap_err();
        assert!(
            err.contains("must be between 0 and"),
            "limit above max should be rejected: {err}"
        );
    }

    #[test]
    fn parse_params_limit_at_max_accepted() {
        let params = parse_item_list_params(Some(&format!("limit={MAX_PAGE_LIMIT}"))).unwrap();
        assert_eq!(
            params.limit, MAX_PAGE_LIMIT,
            "limit at MAX_PAGE_LIMIT should be accepted"
        );
    }

    #[test]
    fn parse_params_duplicate_limit_rejected() {
        let err = parse_item_list_params(Some("limit=5&limit=10")).unwrap_err();
        assert!(
            err.contains("Duplicate query parameter: 'limit'"),
            "duplicate limit should be rejected: {err}"
        );
    }

    #[test]
    fn parse_params_duplicate_order_rejected() {
        let err = parse_item_list_params(Some("order=asc&order=desc")).unwrap_err();
        assert!(
            err.contains("Duplicate query parameter: 'order'"),
            "duplicate order should be rejected: {err}"
        );
    }

    #[test]
    fn parse_params_duplicate_after_rejected() {
        let err = parse_item_list_params(Some("after=a&after=b")).unwrap_err();
        assert!(
            err.contains("Duplicate query parameter: 'after'"),
            "duplicate after should be rejected: {err}"
        );
    }

    #[test]
    fn parse_params_unknown_param_rejected() {
        let err = parse_item_list_params(Some("foo=bar")).unwrap_err();
        assert!(
            err.contains("Unknown query parameter: 'foo'"),
            "unknown parameter should be rejected: {err}"
        );
    }

    #[test]
    fn parse_params_known_key_only_rejected() {
        let err = parse_item_list_params(Some("limit")).unwrap_err();
        assert!(
            err.contains("Missing value for query parameter 'limit'"),
            "key-only known param should be rejected: {err}"
        );
    }

    #[test]
    fn parse_params_empty_after_rejected() {
        let err = parse_item_list_params(Some("after=")).unwrap_err();
        assert!(
            err.contains("cursor must not be empty"),
            "empty after should be rejected: {err}"
        );
    }

    #[test]
    fn parse_params_invalid_utf8_key_rejected() {
        let err = parse_item_list_params(Some("%FF=1")).unwrap_err();
        assert!(
            err.contains("valid UTF-8"),
            "invalid UTF-8 key should be rejected: {err}"
        );
    }

    #[test]
    fn parse_params_invalid_utf8_value_rejected() {
        let err = parse_item_list_params(Some("limit=%FF")).unwrap_err();
        assert!(
            err.contains("valid UTF-8"),
            "invalid UTF-8 value should be rejected: {err}"
        );
    }

    #[test]
    fn parse_params_repeated_include_allowed() {
        let params = parse_item_list_params(Some(
            "include=reasoning.encrypted_content&include=message.output_text.logprobs",
        ))
        .unwrap();
        assert_eq!(
            params.limit, DEFAULT_PAGE_LIMIT,
            "repeated include should not affect other defaults"
        );
    }

    #[test]
    fn parse_params_empty_components_ignored() {
        let params = parse_item_list_params(Some("&&limit=5&")).unwrap();
        assert_eq!(params.limit, 5, "empty components should be silently ignored");
    }

    #[test]
    fn parse_params_encoded_duplicate_key_rejected() {
        let err = parse_item_list_params(Some("limit=5&%6Cimit=10")).unwrap_err();
        assert!(
            err.contains("Duplicate query parameter: 'limit'"),
            "encoded duplicate key should be rejected: {err}"
        );
    }

    // -------------------------------------------------------------------------
    // decode_path_segment — additional cases
    // -------------------------------------------------------------------------

    #[test]
    fn decode_item_id_plain_ascii_passes_through() {
        let result = decode_path_segment("item id", "item_abc123").unwrap();
        assert_eq!(result.as_ref(), "item_abc123");
    }

    #[test]
    fn decode_item_id_percent_encoded_ascii() {
        let result = decode_path_segment("item id", "item%5Fabc").unwrap();
        assert_eq!(result.as_ref(), "item_abc", "percent-encoded underscore should decode");
    }

    #[test]
    fn decode_conversation_id_percent_encoded_ascii() {
        let result = decode_path_segment("conversation id", "conv%5Fabc").unwrap();
        assert_eq!(
            result.as_ref(),
            "conv_abc",
            "percent-encoded conversation underscore should decode"
        );
    }

    #[test]
    fn decode_conversation_id_invalid_utf8_returns_error() {
        let result = decode_path_segment("conversation id", "%FF%FE");
        assert!(result.is_err(), "invalid UTF-8 should return error");
        assert!(
            result.unwrap_err().contains("conversation id"),
            "error should name the conversation id segment"
        );
    }

    // -------------------------------------------------------------------------
    // sync_conversation_messages — denormalized cache consistency (#662)
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn sync_repairs_stale_cache_from_item_rows() {
        let store = cache_sync_store().await;
        seed_conversation(&store, "conv_stale", &["a", "b", "c"]).await;
        // Corrupt the denormalized cache so it omits every item.
        store
            .update_conversation_messages("tenant_a", "conv_stale", &Value::Array(vec![]))
            .await
            .expect("stale update should succeed");

        // The append-back path supplies no snapshot, so it re-reads the live cache.
        sync_conversation_messages(&store, "tenant_a", "conv_stale", None)
            .await
            .expect("sync should succeed");

        // Sync should rebuild the cache from all item rows.
        assert_cache_ids(&store, "conv_stale", &["a", "b", "c"]).await;
    }

    #[tokio::test]
    async fn sync_does_not_clobber_concurrent_cache_update() {
        let inner = cache_sync_store().await;
        // Seed conversation with item "a" and a matching cache.
        seed_conversation(&inner, "conv_race", &["a"]).await;
        // Writer A reads the cache (the create path's snapshot) before appending its
        // own item "x" (rows now [a, x]; cache still [a]).
        let snapshot = read_cache(&inner, "conv_race").await;
        inner
            .create_conversation_items(&[cache_item("x", "conv_race", 2)])
            .await
            .expect("append should succeed");

        // A concurrent writer commits item "b" and refreshes the cache during
        // A's rebuild read.
        let store = InterferingStore::new(
            inner,
            Interference::ConcurrentAppendOnFirstList,
            Some(cache_item("b", "conv_race", 3)),
        );

        sync_conversation_messages(&store, "tenant_a", "conv_race", Some(snapshot))
            .await
            .expect("sync should converge under contention");

        // The cache must retain both the concurrent writer's item (b) and writer A's (x).
        assert_cache_ids(&store, "conv_race", &["a", "b", "x"]).await;
    }

    #[tokio::test]
    async fn sync_missing_conversation_reports_error() {
        let store = cache_sync_store().await;
        let err = sync_conversation_messages(&store, "tenant_a", "conv_missing", None)
            .await
            .expect_err("sync on a missing conversation should error");
        assert!(
            matches!(err, StoreError::Database(_)),
            "missing conversation should map to a database error, got {err:?}"
        );
    }

    #[tokio::test]
    async fn sync_reports_success_when_cache_never_converges() {
        let inner = cache_sync_store().await;
        seed_conversation(&inner, "conv_contended", &["a", "b"]).await;
        // Leave the cache stale versus the item rows so every attempt reaches the
        // compare-and-swap instead of returning early on an already-current cache.
        inner
            .update_conversation_messages("tenant_a", "conv_contended", &Value::Array(vec![]))
            .await
            .expect("stale update should succeed");

        // Every compare-and-swap loses the race, so the retry budget is exhausted.
        let store = InterferingStore::new(inner, Interference::AlwaysLoseCas, None);

        // The item mutation this refresh follows is already durable, so exhausting
        // retries must not surface as an error — callers turn a sync error into an
        // HTTP 500 for a committed create/delete, which clients retry into
        // duplicate-item or not-found responses (#662 review follow-up).
        sync_conversation_messages(&store, "tenant_a", "conv_contended", None)
            .await
            .expect("cache-sync exhaustion must not fail an already-committed mutation");

        assert_eq!(
            store.cas_attempts.load(Ordering::SeqCst),
            MAX_SYNC_ATTEMPTS,
            "every retry should attempt the compare-and-swap before giving up"
        );
    }

    #[tokio::test]
    async fn sync_uses_caller_snapshot_without_reloading_cache() {
        let inner = cache_sync_store().await;
        seed_conversation(&inner, "conv_snap", &["a"]).await;
        // The create path reads the cache before appending, so capture that snapshot.
        let snapshot = read_cache(&inner, "conv_snap").await;
        inner
            .create_conversation_items(&[cache_item("x", "conv_snap", 2)])
            .await
            .expect("append should succeed");
        let store = InterferingStore::new(inner, Interference::PassThrough, None);

        sync_conversation_messages(&store, "tenant_a", "conv_snap", Some(snapshot))
            .await
            .expect("sync should succeed on the first attempt");

        assert_eq!(
            store.read_attempts.load(Ordering::SeqCst),
            0,
            "a matching snapshot must satisfy the first swap without reloading the cache"
        );
        assert_eq!(
            store.cas_attempts.load(Ordering::SeqCst),
            1,
            "a valid snapshot should land the swap on the first attempt"
        );

        // The swap should write the freshly rebuilt history.
        assert_cache_ids(&store, "conv_snap", &["a", "x"]).await;
    }

    #[tokio::test]
    async fn sync_falls_back_to_fresh_read_after_snapshot_conflict() {
        let inner = cache_sync_store().await;
        seed_conversation(&inner, "conv_fallback", &["a"]).await;
        inner
            .create_conversation_items(&[cache_item("x", "conv_fallback", 2)])
            .await
            .expect("append should succeed");
        let store = InterferingStore::new(inner, Interference::PassThrough, None);

        // A stale snapshot (empty) never matches the live cache, so the first swap
        // loses and the sync must re-read the live cache to converge.
        sync_conversation_messages(&store, "tenant_a", "conv_fallback", Some(Value::Array(vec![])))
            .await
            .expect("sync should converge after falling back to a fresh read");

        assert_eq!(
            store.cas_attempts.load(Ordering::SeqCst),
            2,
            "the stale snapshot should cost one lost swap before the fresh-read retry"
        );
        assert_eq!(
            store.read_attempts.load(Ordering::SeqCst),
            1,
            "exactly one fresh cache read should follow the snapshot conflict"
        );

        // The fallback retry should rebuild the full history.
        assert_cache_ids(&store, "conv_fallback", &["a", "x"]).await;
    }

    // -------------------------------------------------------------------------
    // Test Utilities
    // -------------------------------------------------------------------------

    /// Build an items-enabled in-memory SQLite store for cache-sync tests.
    async fn cache_sync_store() -> SqliteResponseStore {
        SqliteResponseStore::new(
            "sqlite::memory:",
            "test_responses",
            "test_conversations",
            Some("test_conversation_items"),
            None,
        )
        .await
        .expect("store creation should succeed")
    }

    /// A conversation item whose `item_data` carries its own id for assertions.
    fn cache_item(item_id: &str, conversation_id: &str, position: i64) -> ConversationItemRecord {
        ConversationItemRecord {
            item_id: item_id.to_owned(),
            tenant_id: "tenant_a".to_owned(),
            conversation_id: conversation_id.to_owned(),
            item_data: serde_json::json!({
                "id": item_id,
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": item_id}],
            }),
            created_at: 1000,
            position,
        }
    }

    /// Sorted item ids extracted from a denormalized `messages` cache value.
    fn cache_ids(messages: &Value) -> Vec<String> {
        let mut ids: Vec<String> = messages
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|m| m.get("id").and_then(Value::as_str).map(str::to_owned))
            .collect();
        ids.sort();
        ids
    }

    /// Read the denormalized `messages` cache for a conversation.
    async fn read_cache(store: &SqliteResponseStore, conversation_id: &str) -> Value {
        ConversationItemStore::get_conversation(store, "tenant_a", conversation_id)
            .await
            .expect("get should succeed")
            .expect("conversation should exist")
            .messages
    }

    /// Assert the denormalized cache holds exactly `expected` item ids (sorted).
    async fn assert_cache_ids(store: &dyn ConversationItemStore, conversation_id: &str, expected: &[&str]) {
        let refreshed = store
            .get_conversation("tenant_a", conversation_id)
            .await
            .expect("get should succeed")
            .expect("conversation should exist");
        let expected: Vec<String> = expected.iter().copied().map(str::to_owned).collect();
        assert_eq!(
            cache_ids(&refreshed.messages),
            expected,
            "cache should contain exactly the expected item ids"
        );
    }

    /// Seed a conversation record plus its item rows and a matching cache.
    async fn seed_conversation(store: &SqliteResponseStore, conversation_id: &str, item_ids: &[&str]) {
        let items: Vec<ConversationItemRecord> = item_ids
            .iter()
            .enumerate()
            .map(|(i, id)| cache_item(id, conversation_id, i64::try_from(i + 1).unwrap()))
            .collect();
        let messages = Value::Array(items.iter().map(|it| it.item_data.clone()).collect());
        store
            .upsert_conversation(&ConversationRecord {
                conversation_id: conversation_id.to_owned(),
                tenant_id: "tenant_a".to_owned(),
                created_at: 1000,
                metadata: Value::Object(Map::new()),
                messages,
            })
            .await
            .expect("seed conversation should succeed");
        if !items.is_empty() {
            store
                .create_conversation_items(&items)
                .await
                .expect("seed items should succeed");
        }
    }

    /// How an [`InterferingStore`] perturbs an in-flight `sync_conversation_messages`.
    enum Interference {
        /// Every compare-and-swap reports a lost race, modeling a conversation under
        /// sustained contention where this writer never wins the swap.
        AlwaysLoseCas,
        /// The first `list_conversation_items` call (the rebuild read) returns the
        /// pre-injection rows, then commits an extra item row and refreshes the
        /// denormalized cache to include it — the #662 interleaving where a slower
        /// writer must not clobber the newer cache with its stale snapshot.
        ConcurrentAppendOnFirstList,
        /// No injected fault: delegate every operation. Used to observe the snapshot
        /// fast path and its cache-read count in isolation.
        PassThrough,
    }

    /// Store wrapper that injects a controlled concurrency fault into the cache-sync
    /// path while delegating every other operation to a real SQLite store.
    struct InterferingStore {
        cas_attempts: AtomicUsize,
        fired: AtomicBool,
        injected_item: Option<ConversationItemRecord>,
        inner: SqliteResponseStore,
        interference: Interference,
        read_attempts: AtomicUsize,
    }

    impl InterferingStore {
        /// Wrap a real store with a controlled fault and zeroed observation counters.
        fn new(
            inner: SqliteResponseStore,
            interference: Interference,
            injected_item: Option<ConversationItemRecord>,
        ) -> Self {
            Self {
                cas_attempts: AtomicUsize::new(0),
                fired: AtomicBool::new(false),
                injected_item,
                inner,
                interference,
                read_attempts: AtomicUsize::new(0),
            }
        }

        /// Commit the concurrent writer's item and refresh the cache from all rows.
        async fn commit_concurrent_writer(&self, tenant_id: &str, conversation_id: &str) -> Result<(), StoreError> {
            let Some(item) = self.injected_item.as_ref() else {
                return Ok(());
            };
            self.inner.create_conversation_items(std::slice::from_ref(item)).await?;
            let all = self
                .inner
                .list_conversation_items(tenant_id, conversation_id, None, MAX_PAGE_LIMIT, true)
                .await?;
            let cache = Value::Array(all.into_iter().map(|r| r.item_data).collect());
            self.inner
                .update_conversation_messages(tenant_id, conversation_id, &cache)
                .await
                .map(|_updated| ())
        }
    }

    #[async_trait]
    impl ConversationItemStore for InterferingStore {
        async fn upsert_conversation(&self, record: &ConversationRecord) -> Result<(), StoreError> {
            self.inner.upsert_conversation(record).await
        }

        async fn update_conversation_messages(
            &self,
            tenant_id: &str,
            conversation_id: &str,
            messages: &Value,
        ) -> Result<bool, StoreError> {
            self.inner
                .update_conversation_messages(tenant_id, conversation_id, messages)
                .await
        }

        async fn compare_and_swap_conversation_messages(
            &self,
            tenant_id: &str,
            conversation_id: &str,
            expected_messages: &Value,
            messages: &Value,
        ) -> Result<bool, StoreError> {
            self.cas_attempts.fetch_add(1, Ordering::SeqCst);
            if matches!(self.interference, Interference::AlwaysLoseCas) {
                // Model a writer that is always beaten to the swap.
                return Ok(false);
            }
            self.inner
                .compare_and_swap_conversation_messages(tenant_id, conversation_id, expected_messages, messages)
                .await
        }

        async fn get_conversation(
            &self,
            tenant_id: &str,
            conversation_id: &str,
        ) -> Result<Option<ConversationRecord>, StoreError> {
            self.read_attempts.fetch_add(1, Ordering::SeqCst);
            ConversationItemStore::get_conversation(&self.inner, tenant_id, conversation_id).await
        }

        async fn delete_conversation(&self, tenant_id: &str, conversation_id: &str) -> Result<bool, StoreError> {
            self.inner.delete_conversation(tenant_id, conversation_id).await
        }

        async fn create_conversation_items(&self, items: &[ConversationItemRecord]) -> Result<(), StoreError> {
            self.inner.create_conversation_items(items).await
        }

        async fn list_conversation_items(
            &self,
            tenant_id: &str,
            conversation_id: &str,
            after_item_id: Option<&str>,
            limit: u32,
            ascending: bool,
        ) -> Result<Vec<ConversationItemRecord>, StoreError> {
            let rows = self
                .inner
                .list_conversation_items(tenant_id, conversation_id, after_item_id, limit, ascending)
                .await?;
            if matches!(self.interference, Interference::ConcurrentAppendOnFirstList)
                && !self.fired.swap(true, Ordering::SeqCst)
            {
                self.commit_concurrent_writer(tenant_id, conversation_id).await?;
            }
            Ok(rows)
        }

        async fn get_existing_conversation_item_ids(
            &self,
            tenant_id: &str,
            conversation_id: &str,
            item_ids: &[&str],
        ) -> Result<Vec<String>, StoreError> {
            self.inner
                .get_existing_conversation_item_ids(tenant_id, conversation_id, item_ids)
                .await
        }

        async fn get_conversation_item(
            &self,
            tenant_id: &str,
            conversation_id: &str,
            item_id: &str,
        ) -> Result<Option<ConversationItemRecord>, StoreError> {
            self.inner
                .get_conversation_item(tenant_id, conversation_id, item_id)
                .await
        }

        async fn delete_conversation_item(
            &self,
            tenant_id: &str,
            conversation_id: &str,
            item_id: &str,
        ) -> Result<bool, StoreError> {
            self.inner
                .delete_conversation_item(tenant_id, conversation_id, item_id)
                .await
        }

        async fn conversation_item_position(
            &self,
            tenant_id: &str,
            conversation_id: &str,
            item_id: &str,
        ) -> Result<Option<i64>, StoreError> {
            self.inner
                .conversation_item_position(tenant_id, conversation_id, item_id)
                .await
        }

        async fn max_item_position(&self, tenant_id: &str, conversation_id: &str) -> Result<i64, StoreError> {
            self.inner.max_item_position(tenant_id, conversation_id).await
        }
    }
}
