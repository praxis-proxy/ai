// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Responses record assembly and input-item listing.
//!
//! Store operations use the already owner-scoped handle resolved by each caller.

pub(crate) mod input_items;

pub(crate) use input_items::{InputItemPage, ListParams, MAX_PAGE_LIMIT, Order, list_input_items};
use serde_json::Value;
use tracing::warn;

use crate::{openai::responses::append_stored_input_items, state_owner::StateOwner, store::ResponseRecord};

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

/// Assemble a persisted [`ResponseRecord`] from a completed Responses API
/// response object.
///
/// Returns `None` when the object is null (an incomplete stream) or is
/// missing a required field (`id`, `created_at`, `model`), i.e. it is not
/// persistable. `request_input` is the original create-request `input`;
/// `state_messages` is the accumulated persistence history when rehydrate
/// populated it.
pub(crate) fn build_record(
    response_object: Value,
    owner: StateOwner,
    request_input: Option<Value>,
    state_messages: Option<Vec<Value>>,
) -> Option<ResponseRecord> {
    if response_object.is_null() {
        warn!("response persistence: response_object is null (incomplete stream?)");
        return None;
    }

    let id = response_object.get("id").and_then(Value::as_str);
    let created_at = response_object.get("created_at").and_then(Value::as_i64);
    let model = response_object.get("model").and_then(Value::as_str);

    let (Some(id), Some(created_at), Some(model)) = (id, created_at, model) else {
        warn!("response persistence: missing required field (id, created_at, or model)");
        return None;
    };

    let capture = ResponseCapture::from_response_json(&response_object, request_input, state_messages);

    Some(ResponseRecord {
        id: id.to_owned(),
        owner,
        created_at,
        model: model.to_owned(),
        response_object,
        input: capture.input,
        messages: capture.messages,
    })
}

/// Stored input and message history extracted from a Responses API exchange.
struct ResponseCapture {
    /// Original request input used by rehydration.
    input: Value,

    /// Full message history used by rehydration.
    messages: Value,
}

impl ResponseCapture {
    /// Extract stored input and output from a Responses API exchange.
    fn from_response_json(json: &Value, request_input: Option<Value>, state_messages: Option<Vec<Value>>) -> Self {
        let input = request_input
            .or_else(|| json.get("input").cloned())
            .unwrap_or(Value::Null);
        let history_input = state_messages.map_or_else(|| input.clone(), Value::Array);
        let messages = assemble_stored_messages(history_input, json.get("output"));

        Self { input, messages }
    }
}

/// Build the stored conversation history from response input and output.
fn assemble_stored_messages(input: Value, output: Option<&Value>) -> Value {
    let mut messages = Vec::new();

    append_stored_input_items(&mut messages, input);

    match output {
        Some(Value::Array(items)) => {
            // A streamed response can contain state-owned input items followed
            // by new output. Drop the largest suffix/prefix overlap so a
            // compaction (or any other replayable item) is stored only once.
            let overlap = (0..=messages.len().min(items.len()))
                .rev()
                .find(|&length| {
                    messages
                        .get(messages.len() - length..)
                        .zip(items.get(..length))
                        .is_some_and(|(message_suffix, output_prefix)| message_suffix == output_prefix)
                })
                .unwrap_or(0);
            let new_items = items
                .iter()
                .skip(overlap)
                .filter(|item| {
                    item.get("type").and_then(Value::as_str) != Some("compaction")
                        || !messages.iter().any(|message| message == *item)
                })
                .cloned()
                .collect::<Vec<_>>();
            messages.extend(new_items);
        },
        Some(output) if !output.is_null() && messages.last() != Some(output) => messages.push(output.clone()),
        Some(_) | None => {},
    }

    Value::Array(messages)
}
