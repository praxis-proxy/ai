// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Dispatches parsed SSE events to mutate [`ResponsesState`].
//!
//! Terminal events (`response.completed`, `response.incomplete`,
//! `response.failed`) are authoritative — their payloads overwrite
//! any incrementally accumulated state. Incremental events
//! (`output_item.added`, `function_call_arguments.done`) provide
//! fallback state in case the terminal event is missing.

use praxis_filter::HttpFilterContext;
use serde_json::Value;
use tracing::{debug, warn};

use super::StreamEventsState;
use crate::openai::{responses::state::ResponsesState, sse::responses::ResponsesEvent};

/// Process a single SSE event, updating `ResponsesState` in
/// extensions and per-filter accumulation state.
pub(super) fn accumulate_event(
    ctx: &mut HttpFilterContext<'_>,
    filter_state: &mut StreamEventsState,
    event: &ResponsesEvent,
) {
    match event {
        ResponsesEvent::ResponseCompleted(payload)
        | ResponsesEvent::ResponseIncomplete(payload)
        | ResponsesEvent::ResponseFailed(payload) => {
            handle_terminal_event(ctx, payload, event);
        },

        ResponsesEvent::OutputItemAdded(payload) => {
            handle_output_item_added(ctx, payload);
        },
        ResponsesEvent::OutputItemDone(payload) => {
            handle_output_item_done(ctx, payload);
        },

        ResponsesEvent::FunctionCallArgumentsDelta(payload) => {
            handle_function_call_delta(filter_state, payload);
        },
        ResponsesEvent::FunctionCallArgumentsDone(payload) => {
            handle_function_call_done(ctx, filter_state, payload);
        },

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
fn handle_terminal_event(ctx: &mut HttpFilterContext<'_>, payload: &Value, event: &ResponsesEvent) {
    let response = payload.get("response").unwrap_or(payload);

    let status = match event {
        ResponsesEvent::ResponseCompleted(_) => "completed",
        ResponsesEvent::ResponseIncomplete(_) => "incomplete",
        ResponsesEvent::ResponseFailed(_) => "failed",
        _ => "unknown",
    };
    // SSE payloads are borrowed from the frame parser, so terminal accumulation
    // is the one path that must clone a complete response object.
    let _ = accumulate_response_object(ctx, response.clone(), Some(status));
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
        if let Some(Value::Array(output)) = response.get("output") {
            state.output_items_mut().clone_from(output);
        }
        state.response_object = response;
        had_prior_usage
    };
    ctx.set_metadata("responses.status", status.clone());

    debug!(status, "complete response received, ResponsesState updated");
    had_prior_usage
}

/// Saturating recursive sum for numeric token-usage fields.
fn merge_usage(accumulated: &mut Value, current: &Value) {
    match (accumulated, current) {
        (Value::Object(accumulated), Value::Object(current)) => {
            for (key, value) in current {
                match accumulated.get_mut(key) {
                    Some(existing) => merge_usage(existing, value),
                    None => {
                        accumulated.insert(key.clone(), value.clone());
                    },
                }
            }
        },
        (Value::Number(accumulated), Value::Number(current)) => {
            if let (Some(left), Some(right)) = (accumulated.as_u64(), current.as_u64()) {
                *accumulated = serde_json::Number::from(left.saturating_add(right));
            } else {
                *accumulated = current.clone();
            }
        },
        (accumulated, current) => current.clone_into(accumulated),
    }
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

    let buf = filter_state.tool_call_args.entry(key.clone()).or_default();
    if buf.len().saturating_add(delta.len()) > filter_state.max_tool_call_argument_bytes {
        warn!(
            key,
            limit = filter_state.max_tool_call_argument_bytes,
            "accumulated tool-call arguments exceed max_tool_call_argument_bytes, dropping"
        );
        filter_state.tool_call_args.remove(&key);
        return;
    }
    buf.push_str(delta);
}

/// Finalize a function call from the done event's payload and push to `tool_calls`.
fn handle_function_call_done(ctx: &mut HttpFilterContext<'_>, filter_state: &mut StreamEventsState, payload: &Value) {
    let state = ctx.extensions.get_or_insert_with(ResponsesState::default);

    let Some(key) = tool_call_key(payload) else {
        return;
    };

    let accumulated = filter_state.tool_call_args.remove(&key);
    let arguments = payload
        .get("arguments")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .or(accumulated)
        .unwrap_or_default();

    let tool_call = {
        let Some(item) = find_output_item_mut(state.output_items_mut(), payload) else {
            warn!(
                key,
                "dropping function-call arguments.done without matching output item"
            );
            return;
        };

        let Some(tool_call) = complete_function_call_item(item, &arguments) else {
            warn!(
                key,
                "dropping function-call arguments.done for non-function output item"
            );
            return;
        };
        tool_call
    };

    upsert_tool_call(&mut state.tool_calls, tool_call);
}

/// Build the stable key used by argument delta/done events.
fn tool_call_key(payload: &Value) -> Option<String> {
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
fn complete_function_call_item(item: &mut Value, arguments: &str) -> Option<Value> {
    let obj = item.as_object_mut()?;
    if obj.get("type").and_then(Value::as_str) != Some("function_call") {
        return None;
    }

    obj.insert("arguments".to_owned(), Value::String(arguments.to_owned()));
    if !matches!(
        obj.get("status").and_then(Value::as_str),
        Some("completed" | "incomplete")
    ) {
        obj.insert("status".to_owned(), Value::String("completed".to_owned()));
    }

    Some(item.clone())
}

/// Insert or replace a completed function-call item.
fn upsert_tool_call(tool_calls: &mut Vec<Value>, tool_call: Value) {
    let id = tool_call.get("id").and_then(Value::as_str);
    let call_id = tool_call.get("call_id").and_then(Value::as_str);

    if let Some(existing) = tool_calls.iter_mut().find(|existing| {
        let existing_id = existing.get("id").and_then(Value::as_str);
        let existing_call_id = existing.get("call_id").and_then(Value::as_str);
        id.is_some_and(|id| existing_id == Some(id)) || call_id.is_some_and(|call_id| existing_call_id == Some(call_id))
    }) {
        tool_call.clone_into(existing);
        return;
    }

    tool_calls.push(tool_call);
}
