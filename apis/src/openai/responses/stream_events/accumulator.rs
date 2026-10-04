// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Dispatches parsed SSE events to mutate [`ResponsesState`].
//!
//! Terminal events (`response.completed`, `response.incomplete`,
//! `response.failed`) are authoritative — their payloads overwrite
//! any incrementally accumulated state. Incremental events
//! (`output_item.added`, `function_call_arguments.done`) provide
//! fallback state in case the terminal event is missing.

use std::collections::hash_map::Entry;

use praxis_filter::HttpFilterContext;
use serde_json::Value;
use tracing::{debug, warn};

use super::StreamEventsState;
use crate::openai::{
    responses::{state::ResponsesState, usage::merge_usage},
    sse::responses::ResponsesEvent,
};

/// Process a single SSE event, updating `ResponsesState` in
/// extensions and per-filter accumulation state.
pub(super) fn accumulate_event(
    ctx: &mut HttpFilterContext<'_>,
    filter_state: &mut StreamEventsState,
    event: &mut ResponsesEvent,
) {
    match event {
        ResponsesEvent::ResponseCompleted(payload) => handle_terminal_event(ctx, payload, "completed"),
        ResponsesEvent::ResponseIncomplete(payload) => handle_terminal_event(ctx, payload, "incomplete"),
        ResponsesEvent::ResponseFailed(payload) => handle_terminal_event(ctx, payload, "failed"),

        ResponsesEvent::OutputItemAdded(payload) => {
            handle_output_item_added(ctx, payload);
        },
        ResponsesEvent::OutputItemDone(payload) => {
            handle_output_item_done(ctx, payload);
        },

        ResponsesEvent::FunctionCallArgumentsDelta(payload) => {
            handle_function_call_delta(filter_state, payload);
        },
        ResponsesEvent::FunctionCallArgumentsDone(payload) => handle_function_call_done(ctx, filter_state, payload),

        ResponsesEvent::Error(payload) => {
            warn!(error = %payload, "streaming error event received");
        },

        ResponsesEvent::Unknown { event_type, .. } => {
            debug!(event_type, "unknown SSE event type (forward-compat)");
        },

        _ => {},
    }
}

/// Overwrite `ResponsesState` from a terminal event's authoritative payload.
fn handle_terminal_event(ctx: &mut HttpFilterContext<'_>, payload: &mut Value, status: &str) {
    // The parsed event owns the terminal response. Move it into the canonical
    // state owner before the event is deferred; the remaining envelope keeps
    // only the small lifecycle metadata. Malformed bare-response events retain
    // the old fallback copy so their original envelope remains available.
    let response = match payload.as_object_mut().and_then(|object| object.get_mut("response")) {
        Some(response) => std::mem::take(response),
        None => payload.clone(),
    };
    let _ = accumulate_response_object(ctx, response, Some(status));
}

/// Overwrite response fields from an authoritative complete response object.
///
/// `status_override` is authoritative for SSE terminal events. Other callers
/// use the response object's own `status` field.
pub(super) fn accumulate_response_object(
    ctx: &mut HttpFilterContext<'_>,
    mut response: Value,
    status_override: Option<&str>,
) -> bool {
    let status = status_override
        .map(str::to_owned)
        .or_else(|| response.get("status").and_then(Value::as_str).map(str::to_owned))
        .unwrap_or_else(|| "unknown".to_owned());
    let had_prior_usage = {
        let state = ctx.extensions.get_or_insert_with(ResponsesState::default);
        let had_prior_usage = !state.usage.is_null();
        if let Some(usage) = response.get("usage").filter(|usage| !usage.is_null()) {
            merge_usage(&mut state.usage, usage);
        }
        if !state.usage.is_null()
            && let Some(object) = response.as_object_mut()
        {
            object.insert("usage".to_owned(), state.usage.clone());
        }
        // Completed calls stay in the response's output array until the loop
        // moves that array to its canonical accumulated owner.
        state.tool_calls.clear();
        state.response_object = response;
        state.local_completion_response_template = Value::Null;
        had_prior_usage
    };
    ctx.set_metadata("responses.status", status.clone());

    debug!(status, "complete response received, ResponsesState updated");
    had_prior_usage
}

/// Push a new output item to the incremental accumulator.
fn handle_output_item_added(ctx: &mut HttpFilterContext<'_>, payload: &Value) {
    let state = ctx.extensions.get_or_insert_with(ResponsesState::default);

    if let Some(item) = payload.get("item") {
        state.output_items_mut().push(item.clone());
    }
}

/// Replace an existing output item by index or id, or append if new.
fn handle_output_item_done(ctx: &mut HttpFilterContext<'_>, payload: &Value) {
    let state = ctx.extensions.get_or_insert_with(ResponsesState::default);

    let Some(item) = payload.get("item") else {
        return;
    };

    if let Some(idx) = payload
        .get("output_index")
        .and_then(Value::as_u64)
        .and_then(|v| usize::try_from(v).ok())
        && let Some(slot) = state.output_items_mut().get_mut(idx)
    {
        item.clone_into(slot);
        return;
    }

    if let Some(id) = item.get("id").and_then(Value::as_str)
        && let Some(existing) = state
            .output_items_mut()
            .iter_mut()
            .find(|i| i.get("id").and_then(Value::as_str) == Some(id))
    {
        item.clone_into(existing);
        return;
    }

    state.output_items_mut().push(item.clone());
}

/// Append a function-call argument delta to the running buffer.
fn handle_function_call_delta(filter_state: &mut StreamEventsState, payload: &Value) {
    let Some(key) = tool_call_key(payload) else {
        return;
    };
    let Some(delta) = payload.get("delta").and_then(Value::as_str) else {
        return;
    };
    if filter_state.rejected_tool_call_args.contains(&key) {
        return;
    }

    let limit = filter_state.max_tool_call_argument_bytes;
    match filter_state.tool_call_args.entry(key) {
        Entry::Occupied(mut occupied) => {
            if occupied.get().len().saturating_add(delta.len()) > limit {
                let key = occupied.remove_entry().0;
                reject_overflowing_tool_call(filter_state, key);
                return;
            }
            occupied.get_mut().push_str(delta);
        },
        Entry::Vacant(vacant) => {
            if delta.len() > limit {
                let key = vacant.into_key();
                reject_overflowing_tool_call(filter_state, key);
                return;
            }
            vacant.insert(delta.to_owned());
        },
    }
}

/// Drop an overflowing argument buffer and reject every later event for `key`.
fn reject_overflowing_tool_call(filter_state: &mut StreamEventsState, key: String) {
    warn!(
        key,
        limit = filter_state.max_tool_call_argument_bytes,
        "accumulated tool-call arguments exceed max_tool_call_argument_bytes, dropping"
    );
    filter_state.rejected_tool_call_args.insert(key);
}

/// Finalize a function call in the stream's canonical output item.
fn handle_function_call_done(ctx: &mut HttpFilterContext<'_>, filter_state: &mut StreamEventsState, payload: &Value) {
    let Some(key) = tool_call_key(payload) else {
        return;
    };
    if filter_state.rejected_tool_call_args.contains(&key) {
        return;
    }
    if reject_oversized_done(filter_state, &key, payload) {
        return;
    }

    let accumulated = filter_state.tool_call_args.remove(&key);
    let arguments = payload
        .get("arguments")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .or(accumulated)
        .unwrap_or_default();

    finalize_function_call(ctx, &key, payload, arguments);
}

/// Reject a completed function call whose `arguments` already exceed the cap.
///
/// Returns `true` when the call was rejected and finalization must stop.
fn reject_oversized_done(filter_state: &mut StreamEventsState, key: &str, payload: &Value) -> bool {
    let oversized = payload
        .get("arguments")
        .and_then(Value::as_str)
        .is_some_and(|arguments| arguments.len() > filter_state.max_tool_call_argument_bytes);
    if !oversized {
        return false;
    }
    warn!(
        key,
        limit = filter_state.max_tool_call_argument_bytes,
        "completed tool-call arguments exceed max_tool_call_argument_bytes, dropping"
    );
    filter_state.tool_call_args.remove(key);
    filter_state.rejected_tool_call_args.insert(key.to_owned());
    true
}

/// Apply finalized arguments to the matching output item and store the tool call.
///
/// The completed item stays in the stream's output array. No dispatch copy is
/// created: the loop records its index after moving the array to the canonical
/// accumulated output at the round boundary.
fn finalize_function_call(ctx: &mut HttpFilterContext<'_>, key: &str, payload: &Value, arguments: String) {
    let state = ctx.extensions.get_or_insert_with(ResponsesState::default);
    let Some(item) = find_output_item_mut(state.output_items_mut(), payload) else {
        warn!(
            key,
            "dropping function-call arguments.done without matching output item"
        );
        return;
    };

    if !complete_function_call_item(item, arguments) {
        warn!(
            key,
            "dropping function-call arguments.done for non-function output item"
        );
    }
}

/// Build the stable key used by argument delta/done events.
pub(super) fn tool_call_key(payload: &Value) -> Option<String> {
    payload
        .get("item_id")
        .and_then(Value::as_str)
        .map(|item_id| format!("item:{item_id}"))
        .or_else(|| {
            payload
                .get("output_index")
                .and_then(Value::as_u64)
                .map(|output_index| format!("index:{output_index}"))
        })
}

/// Read-only lookup of the accumulated output item matching an event payload's
/// tool-call key, for #1159 artifact capture without a mutable borrow.
///
/// Mirrors [`find_output_item_mut`]'s matching: stored output items carry an `id`
/// (never the events' top-level `item_id`/`output_index`), so match the payload's
/// `item_id` against each item's `id`, then fall back to positional `output_index`.
pub(super) fn find_output_item<'a>(items: &'a [Value], payload: &Value) -> Option<&'a Value> {
    if let Some(item_id) = payload.get("item_id").and_then(Value::as_str)
        && let Some(item) = items
            .iter()
            .find(|item| item.get("id").and_then(Value::as_str) == Some(item_id))
    {
        return Some(item);
    }

    let output_index = payload
        .get("output_index")
        .and_then(Value::as_u64)
        .and_then(|v| usize::try_from(v).ok())?;
    items.get(output_index)
}

/// Find the output item targeted by a function-call arguments event.
fn find_output_item_mut<'a>(output_items: &'a mut [Value], payload: &Value) -> Option<&'a mut Value> {
    if let Some(item_id) = payload.get("item_id").and_then(Value::as_str)
        && let Some(index) = output_items
            .iter()
            .position(|item| item.get("id").and_then(Value::as_str) == Some(item_id))
    {
        return output_items.get_mut(index);
    }

    let output_index = payload
        .get("output_index")
        .and_then(Value::as_u64)
        .and_then(|v| usize::try_from(v).ok())?;
    output_items.get_mut(output_index)
}

/// Apply finalized arguments to an existing function-call item.
fn complete_function_call_item(item: &mut Value, arguments: String) -> bool {
    let Some(obj) = item.as_object_mut() else { return false };
    if obj.get("type").and_then(Value::as_str) != Some("function_call") {
        return false;
    }

    obj.insert("arguments".to_owned(), Value::String(arguments));
    if !matches!(
        obj.get("status").and_then(Value::as_str),
        Some("completed" | "incomplete")
    ) {
        obj.insert("status".to_owned(), Value::String("completed".to_owned()));
    }

    true
}
