// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Conversation item-record assembly and validation shared by the HTTP handlers.
//!
//! Store operations use the already owner-scoped handle resolved by each caller.

use std::collections::HashSet;

use serde_json::{Map, Value};

use crate::{
    openai::conversations::{contracts::MAX_ITEMS_PER_REQUEST, item_schema::validate_output_item},
    state_owner::StateOwner,
    store::{ConversationItemRecord, StoreError},
};

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::too_many_lines,
    reason = "tests"
)]
mod tests;

// -----------------------------------------------------------------------------
// Item-record assembly
// -----------------------------------------------------------------------------

/// Reject a create request whose item count exceeds the shared bound.
///
/// # Errors
///
/// Returns [`StoreError::InvalidInput`] when `item_count` is over the limit.
pub(crate) fn validate_item_count(item_count: usize) -> Result<(), StoreError> {
    if item_count > MAX_ITEMS_PER_REQUEST {
        return Err(StoreError::InvalidInput(format!(
            "items may contain at most {MAX_ITEMS_PER_REQUEST} entries"
        )));
    }
    Ok(())
}

/// Return the first duplicate item ID in a create request.
#[must_use]
pub(crate) fn duplicate_item_id(items: &[ConversationItemRecord]) -> Option<&str> {
    let mut seen = HashSet::new();
    for item in items {
        if !seen.insert(item.item_id.as_str()) {
            return Some(item.item_id.as_str());
        }
    }
    None
}

/// Build store records for conversation item JSON values.
///
/// `created_at` and `generate_item_id` are supplied by the caller so this stays
/// transport-neutral: `generate_item_id` mints an ID for an item that carries
/// none.
///
/// # Errors
///
/// Returns [`StoreError::InvalidInput`] when an item is malformed.
#[expect(clippy::too_many_arguments, reason = "factoring into a struct would add indirection")]
pub(crate) fn build_item_records(
    owner: &StateOwner,
    conversation_id: &str,
    created_at: i64,
    start_position: i64,
    items: impl IntoIterator<Item = Value>,
    mut generate_item_id: impl FnMut() -> String,
) -> Result<Vec<ConversationItemRecord>, StoreError> {
    items
        .into_iter()
        .enumerate()
        .map(|(index, item)| {
            let (item_id, item_data) = normalize_item(item, &mut generate_item_id)?;
            let offset = i64::try_from(index).unwrap_or(i64::MAX);
            Ok(ConversationItemRecord {
                item_id,
                owner: owner.clone(),
                conversation_id: conversation_id.to_owned(),
                item_data,
                created_at,
                position: start_position.saturating_add(offset),
            })
        })
        .collect()
}

/// Ensure an item is an object and has a usable ID.
fn normalize_item(item: Value, generate_item_id: &mut impl FnMut() -> String) -> Result<(String, Value), StoreError> {
    let Value::Object(mut map) = item else {
        return Err(invalid("each item must be a JSON object"));
    };
    let item_id = match map.get("id") {
        Some(Value::String(id)) if !id.is_empty() => id.clone(),
        Some(Value::String(_)) => return Err(invalid("item id must not be empty")),
        Some(Value::Null) | None => generate_item_id(),
        Some(_) => return Err(invalid("item id must be a string")),
    };
    map.insert("id".to_owned(), Value::String(item_id.clone()));
    normalize_message_item(&mut map)?;
    default_item_status(&mut map);
    let item = Value::Object(map);
    validate_output_item(&item).map_err(StoreError::InvalidInput)?;
    Ok((item_id, item))
}

/// Default a missing or `null` item `status` to `completed`.
///
/// Some backends emit `status: null` (or omit it) on output items such as
/// reasoning; treat a present-but-null status the same as absent so append-back
/// does not fail closed on schema validation.
fn default_item_status(map: &mut Map<String, Value>) {
    if map.get("status").is_none_or(Value::is_null) {
        map.insert("status".to_owned(), Value::String("completed".to_owned()));
    }
}

/// Normalize easy SDK message inputs into conversation message response objects.
fn normalize_message_item(map: &mut Map<String, Value>) -> Result<(), StoreError> {
    // Responses accepts an easy input message without `type`, but persisted
    // Conversation items require the explicit message discriminator.
    if !map.contains_key("type") && map.contains_key("role") && map.contains_key("content") {
        map.insert("type".to_owned(), Value::String("message".to_owned()));
    }
    if map.get("type").and_then(Value::as_str) != Some("message") {
        return Ok(());
    }

    let role = match map.get("role") {
        Some(Value::String(role)) if !role.is_empty() => role.clone(),
        Some(Value::String(_)) => return Err(invalid("message role must not be empty")),
        Some(_) => return Err(invalid("message role must be a string")),
        None => return Err(invalid("message role is required")),
    };

    let content = map
        .remove("content")
        .ok_or_else(|| invalid("message content is required"))?;
    map.insert("content".to_owned(), normalize_message_content(&role, content)?);
    map.entry("status".to_owned())
        .or_insert_with(|| Value::String("completed".to_owned()));

    Ok(())
}

/// Convert string message content to the list-form content returned by the API.
fn normalize_message_content(role: &str, content: Value) -> Result<Value, StoreError> {
    match content {
        Value::String(text) => {
            let content_item = if role == "assistant" {
                serde_json::json!({
                    "type": "output_text",
                    "text": text,
                    "annotations": [],
                    "logprobs": [],
                })
            } else {
                serde_json::json!({
                    "type": "input_text",
                    "text": text,
                })
            };
            Ok(Value::Array(vec![content_item]))
        },
        Value::Array(mut parts) => {
            if role == "assistant" {
                normalize_assistant_content_parts(&mut parts);
            }
            Ok(Value::Array(parts))
        },
        _ => Err(invalid("message content must be a string or array")),
    }
}

/// Fill in the optional `annotations` and `logprobs` fields that some backends
/// omit or send as `null` on assistant `output_text` parts.
fn normalize_assistant_content_parts(parts: &mut [Value]) {
    for part in parts {
        let Some(part) = part.as_object_mut() else {
            continue;
        };
        if part.get("type").and_then(Value::as_str) != Some("output_text") {
            continue;
        }
        if part.get("annotations").is_none_or(Value::is_null) {
            part.insert("annotations".to_owned(), Value::Array(Vec::new()));
        }
        if part.get("logprobs").is_none_or(Value::is_null) {
            part.insert("logprobs".to_owned(), Value::Array(Vec::new()));
        }
    }
}

/// Build an invalid-input store error from a client-facing message.
fn invalid(message: &str) -> StoreError {
    StoreError::InvalidInput(message.to_owned())
}
