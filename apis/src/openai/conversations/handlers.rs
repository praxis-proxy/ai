// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Request handlers for the `/v1/conversations` endpoints.

use std::{borrow::Cow, fmt, marker::PhantomData};

use percent_encoding::percent_decode_str;
use praxis_ai_store::{ConversationItemRecord, ConversationRecord, StoreError};
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
        CreateConversationRequest, DeletedConversationResource, InputItem, ItemOrder, Metadata,
        UpdateConversationRequest,
    },
    validate::{MetadataError, validate_metadata},
};
use crate::{
    openai::include::{IncludeFields, decode_query_component_strict, parse_include, project_item},
    service::{
        conversations::{build_item_records, duplicate_item_id, validate_item_count},
        responses::{MAX_PAGE_LIMIT, input_items::DEFAULT_PAGE_LIMIT},
    },
    store::OwnerScopedResponseStore,
};

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
    store: &OwnerScopedResponseStore,
    body: &[u8],
) -> Result<FilterAction, FilterError> {
    let owner = store.owner();
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
    if let Err(e) = validate_item_count(input.items.len()) {
        return Ok(FilterAction::Reject(store_error_response(&e)?));
    }
    let item_values = input.items.into_iter().map(InputItem::into_value);
    let item_records = match build_item_records(owner, &conversation_id, created_at, 0, item_values, || {
        generated_item_id(ctx)
    }) {
        Ok(records) => records,
        Err(e) => return Ok(FilterAction::Reject(store_error_response(&e)?)),
    };
    if let Some(item_id) = duplicate_item_id(&item_records) {
        return Ok(FilterAction::Reject(invalid_input_response(
            &duplicate_item_id_message(item_id),
        )?));
    }

    let record = ConversationRecord {
        conversation_id: conversation_id.clone(),
        owner: owner.clone(),
        created_at,
        metadata,
        messages: Value::Array(Vec::new()),
    };

    if let Err(e) = store.upsert_conversation(&record).await {
        return Ok(FilterAction::Reject(store_error_response(&e)?));
    }
    if !item_records.is_empty()
        && let Err(e) = store
            .create_items_and_sync_messages(&conversation_id, &item_records)
            .await
    {
        return Ok(FilterAction::Reject(store_error_response(&e)?));
    }
    debug!(conversation_id, "conversation created");

    let body = conversation_response(record);
    Ok(FilterAction::Reject(json_response(200, &body)?))
}

/// Handle `GET /v1/conversations/{id}` — retrieve a conversation.
pub(super) async fn handle_get_conversation(
    store: &OwnerScopedResponseStore,
    conversation_id: &str,
) -> Result<FilterAction, FilterError> {
    let conversation_id = match decoded_path_param("conversation id", conversation_id) {
        Ok(id) => id,
        Err(action) => return action,
    };
    let conversation_id = conversation_id.as_ref();

    match store.get_conversation(conversation_id).await {
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
    store: &OwnerScopedResponseStore,
    conversation_id: &str,
    body: &[u8],
) -> Result<FilterAction, FilterError> {
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

    let existing = match store.get_conversation(conversation_id).await {
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

    // Update only the metadata column. Round-tripping the whole record through
    // `upsert_conversation` would write back the `messages` snapshot read above,
    // clobbering a denormalized cache that a concurrent item append rebuilt in the
    // meantime and dropping those committed items from conversation-backed
    // rehydration (#1144). `created_at` is immutable, so the read above still
    // supplies it for the response.
    match store.update_conversation_metadata(conversation_id, &metadata).await {
        Ok(true) => {},
        Ok(false) => {
            // The conversation was deleted between the read above and this write.
            debug!(conversation_id, "conversation not found for update");
            return Ok(FilterAction::Reject(not_found_response(&format!(
                "No conversation found with id: '{conversation_id}'."
            ))?));
        },
        Err(e) => return Ok(FilterAction::Reject(store_error_response(&e)?)),
    }
    debug!(conversation_id, "conversation updated");

    let body = ConversationResource::new(
        conversation_id.to_owned(),
        existing.created_at,
        Metadata::from_value(metadata),
    );
    Ok(FilterAction::Reject(json_response(200, &body)?))
}

/// Handle `DELETE /v1/conversations/{id}` — delete a conversation.
///
/// This intentionally deletes only the conversation record. The OpenAI
/// Conversations API specifies that deleting a conversation does not delete
/// its items; item cleanup belongs to item deletion or a separate retention
/// policy, not this endpoint.
pub(super) async fn handle_delete_conversation(
    store: &OwnerScopedResponseStore,
    conversation_id: &str,
) -> Result<FilterAction, FilterError> {
    let conversation_id = match decoded_path_param("conversation id", conversation_id) {
        Ok(id) => id,
        Err(action) => return action,
    };
    let conversation_id = conversation_id.as_ref();

    match store.delete_conversation(conversation_id).await {
        Ok(true) => {
            debug!(conversation_id, "conversation deleted");
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
    store: &OwnerScopedResponseStore,
    conversation_id: &str,
    body: &[u8],
) -> Result<FilterAction, FilterError> {
    let owner = store.owner();
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
    match store.get_conversation(conversation_id).await {
        Ok(Some(_)) => {},
        Ok(None) => {
            debug!(conversation_id, "conversation not found for item create");
            return Ok(FilterAction::Reject(not_found_response(
                &conversation_not_found_message(conversation_id),
            )?));
        },
        Err(e) => return Ok(FilterAction::Reject(store_error_response(&e)?)),
    };

    let Some(items) = input.items else {
        return Ok(FilterAction::Reject(invalid_input_response("'items' is required")?));
    };
    if let Err(e) = validate_item_count(items.len()) {
        return Ok(FilterAction::Reject(store_error_response(&e)?));
    }
    let item_values = items.into_iter().map(InputItem::into_value);
    let created_at = current_timestamp(ctx);
    let item_records = match build_item_records(owner, conversation_id, created_at, 0, item_values, || {
        generated_item_id(ctx)
    }) {
        Ok(records) => records,
        Err(e) => return Ok(FilterAction::Reject(store_error_response(&e)?)),
    };
    if let Some(item_id) = duplicate_item_id(&item_records) {
        return Ok(FilterAction::Reject(invalid_input_response(
            &duplicate_item_id_message(item_id),
        )?));
    }
    let requested_ids: Vec<&str> = item_records.iter().map(|r| r.item_id.as_str()).collect();
    let already_present = match store
        .get_existing_conversation_item_ids(conversation_id, &requested_ids)
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

    match store
        .create_items_and_sync_messages(conversation_id, &item_records)
        .await
    {
        Ok(()) => {},
        Err(StoreError::NotFound) => {
            return Ok(FilterAction::Reject(not_found_response(
                &conversation_not_found_message(conversation_id),
            )?));
        },
        Err(e) => return Ok(FilterAction::Reject(store_error_response(&e)?)),
    }
    debug!(
        conversation_id,
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
    store: &OwnerScopedResponseStore,
    conversation_id: &str,
) -> Result<FilterAction, FilterError> {
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
    match store.get_conversation(conversation_id).await {
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
    store: &OwnerScopedResponseStore,
    conversation_id: &str,
    item_id: &str,
) -> Result<FilterAction, FilterError> {
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
    match store.get_conversation(conversation_id).await {
        Ok(Some(_)) => {},
        Ok(None) => {
            debug!(conversation_id, item_id, "conversation not found for item get");
            return Ok(FilterAction::Reject(not_found_response(
                &conversation_not_found_message(conversation_id),
            )?));
        },
        Err(e) => return Ok(FilterAction::Reject(store_error_response(&e)?)),
    }
    match store.get_conversation_item(conversation_id, item_id).await {
        Ok(Some(record)) => {
            let item = conversation_item_response(record.item_data, includes);
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
    store: &OwnerScopedResponseStore,
    conversation_id: &str,
    item_id: &str,
) -> Result<FilterAction, FilterError> {
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
    match store.get_conversation(conversation_id).await {
        Ok(Some(_)) => {},
        Ok(None) => {
            debug!(conversation_id, item_id, "conversation not found for item delete");
            return Ok(FilterAction::Reject(not_found_response(
                &conversation_not_found_message(conversation_id),
            )?));
        },
        Err(e) => return Ok(FilterAction::Reject(store_error_response(&e)?)),
    };

    match store.delete_item_and_sync_messages(conversation_id, item_id).await {
        Ok(true) => {
            debug!(conversation_id, item_id, "conversation item deleted");
            match store.get_conversation(conversation_id).await {
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
        Err(StoreError::NotFound) => Ok(FilterAction::Reject(not_found_response(
            &conversation_not_found_message(conversation_id),
        )?)),
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
        data.push(conversation_item_response(record.item_data, includes));
    }

    ConversationItemList::new(data, has_more, first_id, last_id)
}

/// Project stored items for list and retrieve without copying their JSON.
fn conversation_item_response(mut item: Value, includes: IncludeFields) -> ConversationItem {
    if item.get("type").and_then(Value::as_str) == Some("mcp_call")
        && let Some(map) = item.as_object_mut()
        && map.get("error").is_some_and(Value::is_string)
        && let Some(content) = map.remove("error")
    {
        // Historical records predate the tagged MCP error contract. Move the
        // legacy content into that shape only at the response boundary.
        let mut tagged = Map::new();
        tagged.insert("type".to_owned(), Value::String("mcp_tool_execution_error".to_owned()));
        tagged.insert("content".to_owned(), content);
        map.insert("error".to_owned(), Value::Object(tagged));
    }
    project_item(&mut item, includes);
    ConversationItem::from_value(item)
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

#[cfg(test)]
#[cfg(feature = "store-sqlite")]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing, reason = "tests")]
mod tests {
    use super::*;

    #[test]
    fn legacy_mcp_errors_normalize_at_the_read_boundary() {
        let legacy = serde_json::json!({
            "type": "mcp_call", "id": "mcp_legacy", "status": "failed",
            "server_label": "weather", "name": "lookup", "arguments": "{}",
            "output": null, "error": "legacy failure",
        });
        let item = conversation_item_response(legacy, IncludeFields::default());
        let output = serde_json::to_value(item).unwrap();
        assert_eq!(
            output["error"],
            serde_json::json!({
                "type": "mcp_tool_execution_error", "content": "legacy failure",
            }),
            "legacy errors must be tagged before serialization"
        );
        assert!(
            super::super::item_schema::validate_output_item(&output).is_ok(),
            "normalized legacy MCP output must satisfy the contract"
        );

        for error in [
            serde_json::json!({"type": "mcp_protocol_error", "code": -1, "message": "failed"}),
            Value::Null,
        ] {
            let input = serde_json::json!({"type": "mcp_call", "error": error});
            let output =
                serde_json::to_value(conversation_item_response(input.clone(), IncludeFields::default())).unwrap();
            assert_eq!(output, input, "structured and null errors must not change");
        }
    }

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
}
