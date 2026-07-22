// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Request-scoped state for the Responses API filter set.
//!
//! [`ResponsesState`] is stored in [`RequestExtensions`] and shared
//! across filter phases. It holds the heavy data needed by the
//! validate → rehydrate → `openai_tool_parse` → `openai_responses_proxy` →
//! `stream_events` → `tool_dispatch` pipeline.
//!
//! [`RequestExtensions`]: praxis_filter::RequestExtensions

use std::collections::HashMap;

/// Maximum citation file mappings retained across execution and rehydration.
pub(crate) const MAX_CITATION_FILES: usize = 1_024;

/// Request-scoped state shared across Responses API filters.
///
/// Stored in [`RequestExtensions`] by the validate filter and read
/// or mutated by subsequent filters. Uses [`serde_json::Value`] for
/// flexibility while the Responses API types stabilize; can be
/// refactored to typed structs later without affecting external
/// callers.
///
/// [`RequestExtensions`]: praxis_filter::RequestExtensions
#[expect(clippy::struct_excessive_bools, reason = "independent provider request policies")]
pub(crate) struct ResponsesState {
    /// Maps `file_id` to filename for citation markers injected by
    /// the file-search filter and consumed by response annotation extraction.
    pub citation_files: HashMap<String, String>,

    /// Truncation strategy for managing context window limits.
    ///
    /// Preserves the full object from the request so filters can
    /// inspect both the strategy type and any parameters.
    pub context_management: Option<serde_json::Value>,

    /// Tool choice derived for internal continuation inference.
    ///
    /// The effective public policy remains in [`Self::tool_choice`] for public
    /// response projection.
    pub continuation_tool_choice: Option<serde_json::Value>,

    /// Initial input retained once until the first response is accumulated.
    ///
    /// Standalone file-search creates do not need materialized replay histories
    /// while the original request is forwarded. Deferring those histories avoids
    /// retaining three deep copies of a potentially large `input` value during
    /// the first inference pass.
    pub deferred_input: Option<Vec<serde_json::Value>>,

    /// Number of leading output items produced before continuation inference.
    ///
    /// Before the next backend response is accumulated, these items remain at
    /// the start of [`Self::output_items`]. The count avoids cloning the full
    /// public output merely to preserve it across a continuation.
    pub continuation_output_count: usize,

    /// Start of private persistence added by local search in this request.
    ///
    /// Conversation append-back uses this boundary to overlay only current-turn
    /// markers when history came from another provider-owned source.
    pub local_file_search_persistence_start: Option<usize>,

    /// Conversation scope for multi-turn state.
    ///
    /// Can be a string ID or an object with `id`. Controls which
    /// stored conversation this request belongs to.
    pub conversation: Option<serde_json::Value>,

    /// Additional fields to include in the response.
    ///
    /// E.g. `["usage"]`, `["file_search_results"]`. Filters that
    /// construct the response object check this to decide which
    /// optional sections to populate.
    pub include: Vec<String>,

    /// Whether `rehydrate` successfully resolved request history into this state.
    ///
    /// A request carrying a previous-response ID remains provider-owned when no
    /// rehydrate filter is configured. The proxy strips that ID only after this
    /// flag is set; `conversation` is removed independently on every path.
    pub history_rehydrated: bool,

    /// The current request's input items, immutable after construction.
    ///
    /// Preserved as-is so downstream filters can inspect what the
    /// client actually sent, independent of conversation history
    /// resolved by `rehydrate`.
    pub input: Vec<serde_json::Value>,

    /// Current agentic loop iteration (0-indexed). Incremented by
    /// `tool_dispatch` at the start of each new inference round.
    pub iteration: u32,

    /// Maximum number of tool-call rounds in the agentic loop.
    ///
    /// `tool_dispatch` checks this to cap iterations. `None` means
    /// no explicit limit was set by the client.
    pub max_tool_calls: Option<u32>,

    /// Resolved MCP tool definitions keyed by `(server_label,
    /// tool_name)`.
    ///
    /// Built by `openai_mcp_tool_resolve` from `tools/list` responses.
    /// Consumed by `mcp_tool` (#27) for dispatch routing.
    pub mcp_tool_map: HashMap<(String, String), serde_json::Value>,

    /// Resolved conversation history sent to the backend.
    ///
    /// Initialized from the current request's input. When
    /// `previous_response_id` is set, `rehydrate` prepends stored
    /// history. `tool_dispatch` appends tool results during agentic
    /// loops. `openai_responses_proxy` reads this as the authoritative
    /// conversation to send to the backend. Output-only metadata
    /// items must be omitted from this field.
    pub messages: Vec<serde_json::Value>,

    /// Whether tool calls may execute concurrently within an
    /// iteration. Defaults to `true` per the API spec.
    pub parallel_tool_calls: bool,

    /// Full message history to persist for future rehydration.
    ///
    /// This may include output-only metadata items omitted from
    /// [`Self::messages`] because it is not forwarded to backend
    /// inference.
    pub persisted_messages: Vec<serde_json::Value>,

    /// ID of a previous response to continue from.
    ///
    /// When set, `rehydrate` fetches the stored conversation
    /// history for this response and prepends it to `messages`.
    pub previous_response_id: Option<String>,

    /// MCP tool listings recovered from the previous response.
    pub previous_tools: Vec<serde_json::Value>,

    /// Token usage reported by the previous response.
    pub previous_usage: Option<serde_json::Value>,

    /// Parsed request fields not separately owned by deferred state.
    ///
    /// Fully materialized state preserves the complete request. Deferred
    /// first-pass file-search state moves `input` and `tools` into their owned
    /// fields so each potentially large value is retained only once.
    pub request_body: serde_json::Value,

    /// The constructed response object for the current iteration.
    pub response_object: serde_json::Value,

    /// Tool calls from the current inference response only.
    ///
    /// Cleared by `tool_dispatch` at the start of each iteration
    /// before `stream_events` writes new ones. Without explicit
    /// clearing, stale tool calls from a previous iteration cause
    /// duplicate dispatch.
    pub tool_calls: Vec<serde_json::Value>,

    /// Effective tool choice setting. Request-only shorthand is canonicalized
    /// so this value is valid when restored into a public response.
    pub tool_choice: serde_json::Value,

    /// Whether the client explicitly supplied [`Self::tool_choice`].
    pub tool_choice_present: bool,

    /// Processed tool definitions from the request.
    pub tools: Vec<serde_json::Value>,

    /// Token usage accumulated across all iterations within the
    /// request. `stream_events` merges per-iteration usage into
    /// the running total.
    pub usage: serde_json::Value,
}

impl Default for ResponsesState {
    fn default() -> Self {
        Self {
            citation_files: HashMap::new(),
            context_management: None,
            continuation_tool_choice: None,
            deferred_input: None,
            continuation_output_count: 0,
            local_file_search_persistence_start: None,
            conversation: None,
            include: Vec::new(),
            history_rehydrated: false,
            input: Vec::new(),
            iteration: 0,
            max_tool_calls: None,
            mcp_tool_map: HashMap::new(),
            messages: Vec::new(),
            parallel_tool_calls: true,
            persisted_messages: Vec::new(),
            previous_response_id: None,
            previous_tools: Vec::new(),
            previous_usage: None,
            request_body: serde_json::Value::Null,
            response_object: serde_json::Value::Null,
            tool_calls: Vec::new(),
            tool_choice: serde_json::Value::String("auto".to_owned()),
            tool_choice_present: false,
            tools: Vec::new(),
            usage: serde_json::Value::Null,
        }
    }
}

impl ResponsesState {
    /// Create initial state from a parsed request body.
    pub(crate) fn from_request_body(body: serde_json::Value) -> Self {
        let messages = normalize_input(&body);
        let persisted_messages = messages.clone();
        let tool_choice_present = body.get("tool_choice").is_some();
        let tool_choice = effective_response_tool_choice(body.get("tool_choice").cloned());

        let tools = extract_array_field(&body, "tools");

        Self {
            context_management: body.get("context_management").cloned(),
            conversation: body.get("conversation").cloned(),
            include: extract_string_array(&body, "include"),
            input: messages.clone(),
            max_tool_calls: extract_u32(&body, "max_tool_calls"),
            messages,
            parallel_tool_calls: extract_bool_or(&body, "parallel_tool_calls", true),
            persisted_messages,
            previous_response_id: extract_string(&body, "previous_response_id"),
            request_body: body,
            tool_choice,
            tool_choice_present,
            tools,
            ..Default::default()
        }
    }

    /// Create lightweight state for a standalone hosted file-search request.
    ///
    /// The original request bytes remain authoritative for the first backend
    /// pass. Move `input` and `tools` out of the parsed fallback body so state
    /// retains each large value once, then materialize replay histories when a
    /// response actually arrives.
    pub(crate) fn from_file_search_request_body(mut body: serde_json::Value) -> Self {
        let input = take_field(&mut body, "input");
        let tools = take_array_field(&mut body, "tools");
        let context_management = take_field(&mut body, "context_management");
        let conversation = take_field(&mut body, "conversation");
        let include = take_string_array_field(&mut body, "include");
        let previous_response_id = take_string_field(&mut body, "previous_response_id");
        let tool_choice_present = body.get("tool_choice").is_some();
        let tool_choice = effective_response_tool_choice(take_field(&mut body, "tool_choice"));

        Self {
            context_management,
            conversation,
            deferred_input: Some(normalize_input_value(input)),
            include,
            max_tool_calls: extract_u32(&body, "max_tool_calls"),
            parallel_tool_calls: extract_bool_or(&body, "parallel_tool_calls", true),
            previous_response_id,
            request_body: body,
            tool_choice,
            tool_choice_present,
            tools,
            ..Default::default()
        }
    }

    /// Materialize model and persistence histories after the first response.
    pub(crate) fn materialize_deferred_history(&mut self) {
        let Some(input) = self.deferred_input.take() else {
            return;
        };
        self.input = input;
        self.messages.clone_from(&self.input);
        self.persisted_messages.clone_from(&self.input);
    }

    /// Borrow normalized request input before or after deferred materialization.
    pub(crate) fn request_input_items(&self) -> &[serde_json::Value] {
        if self.input.is_empty() {
            self.deferred_input.as_deref().unwrap_or(&self.input)
        } else {
            &self.input
        }
    }

    /// Return whether initial history is still retained in deferred form.
    #[cfg(test)]
    pub(crate) fn has_deferred_history(&self) -> bool {
        self.deferred_input.is_some()
    }

    /// Borrow the public output owned by [`Self::response_object`].
    pub(crate) fn output_items(&self) -> &[serde_json::Value] {
        self.response_object
            .get("output")
            .and_then(serde_json::Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default()
    }

    /// Mutably borrow public output, creating a valid array when absent.
    pub(crate) fn output_items_mut(&mut self) -> &mut Vec<serde_json::Value> {
        if !self.response_object.is_object() {
            self.response_object = serde_json::Value::Object(serde_json::Map::new());
        }
        let serde_json::Value::Object(response) = &mut self.response_object else {
            unreachable!("response_object was normalized to an object")
        };
        let output = response
            .entry("output".to_owned())
            .or_insert_with(|| serde_json::Value::Array(Vec::new()));
        if !output.is_array() {
            *output = serde_json::Value::Array(Vec::new());
        }
        let serde_json::Value::Array(items) = output else {
            unreachable!("output was normalized to an array")
        };
        items
    }

    /// Move public output out while retaining an empty response array.
    pub(crate) fn take_output_items(&mut self) -> Vec<serde_json::Value> {
        match self.response_object.get_mut("output") {
            Some(serde_json::Value::Array(items)) => std::mem::take(items),
            _ => Vec::new(),
        }
    }

    /// Replace the public output owned by [`Self::response_object`].
    pub(crate) fn replace_output_items(&mut self, items: Vec<serde_json::Value>) {
        *self.output_items_mut() = items;
    }
}

/// Canonicalize request-only tool-choice shorthand for response projection.
fn effective_response_tool_choice(value: Option<serde_json::Value>) -> serde_json::Value {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return serde_json::Value::String("auto".to_owned());
    };
    let mut choice = match value {
        serde_json::Value::Object(choice) => choice,
        other => return other,
    };
    if choice.get("type").and_then(serde_json::Value::as_str) != Some("allowed_tools") {
        return serde_json::Value::Object(choice);
    }
    if let Some(serde_json::Value::Object(mut allowed)) = choice.remove("allowed_tools") {
        allowed.insert("type".to_owned(), serde_json::Value::String("allowed_tools".to_owned()));
        choice = allowed;
    }
    if !choice.get("mode").is_some_and(serde_json::Value::is_string) {
        choice.insert("mode".to_owned(), serde_json::Value::String("auto".to_owned()));
    }
    serde_json::Value::Object(choice)
}

/// Normalize the `input` field into a message array.
///
/// The Responses API `input` can be a string (single user message)
/// or an array of message objects. Normalizes both forms to a
/// `Vec<Value>`.
fn normalize_input(body: &serde_json::Value) -> Vec<serde_json::Value> {
    match body.get("input") {
        Some(serde_json::Value::Array(arr)) => arr.clone(),
        Some(serde_json::Value::String(s)) => {
            vec![serde_json::json!({
                "type": "message",
                "role": "user",
                "content": s,
            })]
        },
        _ => Vec::new(),
    }
}

/// Normalize an owned `input` value without cloning array items.
fn normalize_input_value(input: Option<serde_json::Value>) -> Vec<serde_json::Value> {
    match input {
        Some(serde_json::Value::Array(items)) => items,
        Some(serde_json::Value::String(text)) => {
            vec![serde_json::json!({
                "type": "message",
                "role": "user",
                "content": text,
            })]
        },
        _ => Vec::new(),
    }
}

/// Move one field out of an object request body.
fn take_field(body: &mut serde_json::Value, field: &str) -> Option<serde_json::Value> {
    body.as_object_mut()?.remove(field)
}

/// Move an array field out of an object request body, defaulting to empty.
fn take_array_field(body: &mut serde_json::Value, field: &str) -> Vec<serde_json::Value> {
    match take_field(body, field) {
        Some(serde_json::Value::Array(items)) => items,
        _ => Vec::new(),
    }
}

/// Move a string field out of an object body, leaving malformed values intact.
fn take_string_field(body: &mut serde_json::Value, field: &str) -> Option<String> {
    match take_field(body, field)? {
        serde_json::Value::String(value) => Some(value),
        other => {
            body.as_object_mut()?.insert(field.to_owned(), other);
            None
        },
    }
}

/// Move and normalize an array of strings from an object body.
fn take_string_array_field(body: &mut serde_json::Value, field: &str) -> Vec<String> {
    let Some(value) = take_field(body, field) else {
        return Vec::new();
    };
    match value {
        serde_json::Value::Array(values) if values.iter().all(serde_json::Value::is_string) => values
            .into_iter()
            .map(|value| match value {
                serde_json::Value::String(value) => value,
                _ => unreachable!("include values were validated as strings"),
            })
            .collect(),
        other => {
            if let Some(object) = body.as_object_mut() {
                object.insert(field.to_owned(), other);
            }
            Vec::new()
        },
    }
}

/// Extract a JSON array field by name, defaulting to empty.
fn extract_array_field(body: &serde_json::Value, field: &str) -> Vec<serde_json::Value> {
    body.get(field)
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default()
}

/// Extract a string field by name.
fn extract_string(body: &serde_json::Value, field: &str) -> Option<String> {
    body.get(field)
        .and_then(serde_json::Value::as_str)
        .map(ToOwned::to_owned)
}

/// Extract an array of strings by name, defaulting to empty.
fn extract_string_array(body: &serde_json::Value, field: &str) -> Vec<String> {
    body.get(field)
        .and_then(serde_json::Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(serde_json::Value::as_str)
                .map(ToOwned::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

/// Extract a `u32` field by name, logging when a value is present
/// but not representable as `u32`.
fn extract_u32(body: &serde_json::Value, field: &str) -> Option<u32> {
    let raw = body.get(field)?;
    let result = raw.as_u64().and_then(|v| u32::try_from(v).ok());
    if result.is_none() {
        tracing::debug!(field, %raw, "ignoring non-u32 value");
    }
    result
}

/// Extract a bool field by name, returning a default if absent.
fn extract_bool_or(body: &serde_json::Value, field: &str, default: bool) -> bool {
    body.get(field).and_then(serde_json::Value::as_bool).unwrap_or(default)
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests"
)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn from_request_body_extracts_string_input() {
        let body = json!({
            "model": "gpt-4o",
            "input": "Hello, world!"
        });
        let state = ResponsesState::from_request_body(body);
        assert_eq!(state.input.len(), 1, "string input should produce one item");
        assert_eq!(
            state.input[0]["role"], "user",
            "string input should default to user role"
        );
        assert_eq!(
            state.input[0]["type"], "message",
            "string input should produce a Responses message item"
        );
        assert_eq!(state.input[0]["content"], "Hello, world!");
    }

    #[test]
    fn from_request_body_extracts_array_input() {
        let body = json!({
            "model": "gpt-4o",
            "input": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "second"}
            ]
        });
        let state = ResponsesState::from_request_body(body);
        assert_eq!(state.input.len(), 2, "array input should preserve all items");
    }

    #[test]
    fn from_request_body_empty_input() {
        let body = json!({"model": "gpt-4o"});
        let state = ResponsesState::from_request_body(body);
        assert!(state.input.is_empty(), "missing input should produce empty input");
    }

    #[test]
    fn request_tool_choice_shorthand_is_canonicalized_for_public_responses() {
        let null_choice = ResponsesState::from_request_body(json!({
            "model":"gpt-4o","input":"hello","tool_choice":null
        }));
        assert_eq!(null_choice.tool_choice, "auto");
        assert!(null_choice.tool_choice_present);

        let allowed_choice = ResponsesState::from_file_search_request_body(json!({
            "model":"gpt-4o",
            "input":"hello",
            "tools":[{"type":"file_search","vector_store_ids":["vs-1"]}],
            "tool_choice":{
                "type":"allowed_tools",
                "tools":[{"type":"function","name":"lookup"}]
            }
        }));
        assert_eq!(allowed_choice.tool_choice["type"], "allowed_tools");
        assert_eq!(allowed_choice.tool_choice["mode"], "auto");
        assert_eq!(allowed_choice.tool_choice["tools"][0]["name"], "lookup");
    }

    #[test]
    fn input_and_messages_start_identical() {
        let body = json!({
            "model": "gpt-4o",
            "input": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"}
            ]
        });
        let state = ResponsesState::from_request_body(body);
        assert_eq!(
            state.input, state.messages,
            "input and messages should be identical at construction"
        );
        assert_eq!(
            state.input, state.persisted_messages,
            "input and persisted_messages should be identical at construction"
        );
    }

    #[test]
    fn file_search_state_defers_large_input_history_without_duplication() {
        let input = json!([
            {"type": "message", "role": "user", "content": "large request content"},
            {"type": "message", "role": "assistant", "content": "more history"}
        ]);
        let tools = json!([
            {"type": "file_search", "vector_store_ids": ["vs_1"]},
            {"type": "function", "name": "large_schema", "parameters": {"type": "object"}}
        ]);
        let mut state = ResponsesState::from_file_search_request_body(json!({
            "model": "gpt-4.1", "input": input, "tools": tools, "max_tool_calls": 4
        }));
        assert!(state.has_deferred_history());
        assert!(state.input.is_empty());
        assert!(state.messages.is_empty());
        assert!(state.persisted_messages.is_empty());
        assert!(state.request_body.get("input").is_none());
        assert!(state.request_body.get("tools").is_none());
        assert_eq!(state.tools, tools.as_array().unwrap().clone());
        assert_eq!(state.max_tool_calls, Some(4));
        assert_eq!(state.request_body["max_tool_calls"], 4);
        state.materialize_deferred_history();
        assert!(!state.has_deferred_history());
        assert_eq!(state.messages, input.as_array().unwrap().clone());
        assert_eq!(state.persisted_messages, state.messages);
        assert_eq!(state.input, input.as_array().unwrap().clone());
    }

    #[test]
    #[expect(clippy::too_many_lines, reason = "covers every separately owned deferred field")]
    fn file_search_state_moves_large_rebuild_fields_out_of_fallback_body() {
        let context_management = json!({"type": "custom", "payload": "c".repeat(32_768)});
        let conversation = json!({"id": "", "metadata": "v".repeat(32_768)});
        let include = json!(["file_search_call.results", "i".repeat(32_768)]);
        let previous_response_id = "p".repeat(1_024);
        let tool_choice = json!({
            "type": "allowed_tools",
            "mode": "required",
            "tools": [
                {"type": "file_search"},
                {"type": "function", "name": "f", "description": "d".repeat(32_768)}
            ]
        });
        let state = ResponsesState::from_file_search_request_body(json!({
            "model": "gpt-4.1",
            "input": "search",
            "tools": [{"type": "file_search", "vector_store_ids": ["vs_1"]}],
            "context_management": context_management,
            "conversation": conversation,
            "include": include,
            "previous_response_id": previous_response_id,
            "tool_choice": tool_choice,
        }));

        assert_eq!(state.context_management.as_ref(), Some(&context_management));
        assert_eq!(state.conversation.as_ref(), Some(&conversation));
        assert_eq!(
            state.include,
            include
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_str().unwrap().to_owned())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            state.previous_response_id.as_deref(),
            Some(previous_response_id.as_str())
        );
        assert_eq!(state.tool_choice, tool_choice);
        for field in [
            "input",
            "tools",
            "context_management",
            "conversation",
            "include",
            "previous_response_id",
            "tool_choice",
        ] {
            assert!(state.request_body.get(field).is_none(), "{field} should have one owner");
        }
    }

    #[test]
    fn file_search_state_keeps_malformed_typed_fields_in_fallback_body() {
        let state = ResponsesState::from_file_search_request_body(json!({
            "model": "gpt-4.1",
            "input": "search",
            "tools": [{"type": "file_search", "vector_store_ids": ["vs_1"]}],
            "include": ["valid", 7],
            "previous_response_id": null,
        }));

        assert!(state.include.is_empty());
        assert!(!state.history_rehydrated);
        assert_eq!(state.request_body["include"], json!(["valid", 7]));
        assert!(state.previous_response_id.is_none());
        assert!(state.request_body["previous_response_id"].is_null());
    }

    #[test]
    fn from_request_body_extracts_tools() {
        let body = json!({
            "model": "gpt-4o",
            "input": "test",
            "tools": [{"type": "function", "name": "get_weather"}]
        });
        let state = ResponsesState::from_request_body(body);
        assert_eq!(state.tools.len(), 1, "should extract one tool");
    }

    #[test]
    fn from_request_body_default_tool_choice() {
        let body = json!({"model": "gpt-4o", "input": "test"});
        let state = ResponsesState::from_request_body(body);
        assert_eq!(state.tool_choice, json!("auto"), "default tool_choice should be auto");
    }

    #[test]
    fn from_request_body_explicit_tool_choice() {
        let body = json!({
            "model": "gpt-4o",
            "input": "test",
            "tool_choice": "required"
        });
        let state = ResponsesState::from_request_body(body);
        assert_eq!(
            state.tool_choice,
            json!("required"),
            "should preserve explicit tool_choice"
        );
    }

    #[test]
    fn initial_state_has_zero_iteration() {
        let body = json!({"model": "gpt-4o", "input": "test"});
        let state = ResponsesState::from_request_body(body);
        assert_eq!(state.iteration, 0, "initial iteration should be 0");
    }

    #[test]
    fn initial_state_has_empty_tool_calls() {
        let body = json!({"model": "gpt-4o", "input": "test"});
        let state = ResponsesState::from_request_body(body);
        assert!(state.tool_calls.is_empty(), "initial tool_calls should be empty");
    }

    #[test]
    fn initial_state_has_null_usage() {
        let body = json!({"model": "gpt-4o", "input": "test"});
        let state = ResponsesState::from_request_body(body);
        assert!(state.usage.is_null(), "initial usage should be null");
    }

    #[test]
    fn request_body_is_preserved() {
        let body = json!({"model": "gpt-4o", "input": "hello", "temperature": 0.7});
        let state = ResponsesState::from_request_body(body.clone());
        assert_eq!(state.request_body, body, "original request body should be preserved");
    }

    #[test]
    fn extracts_previous_response_id() {
        let body = json!({"model": "gpt-4o", "input": "test", "previous_response_id": "resp_abc123"});
        let state = ResponsesState::from_request_body(body);
        assert_eq!(state.previous_response_id.as_deref(), Some("resp_abc123"));
    }

    #[test]
    fn previous_response_id_defaults_to_none() {
        let body = json!({"model": "gpt-4o", "input": "test"});
        let state = ResponsesState::from_request_body(body);
        assert!(state.previous_response_id.is_none());
    }

    #[test]
    fn extracts_conversation_string() {
        let body = json!({"model": "gpt-4o", "input": "test", "conversation": "conv_xyz"});
        let state = ResponsesState::from_request_body(body);
        assert_eq!(state.conversation, Some(json!("conv_xyz")));
    }

    #[test]
    fn extracts_conversation_object() {
        let body = json!({"model": "gpt-4o", "input": "test", "conversation": {"id": "conv_xyz"}});
        let state = ResponsesState::from_request_body(body);
        assert_eq!(state.conversation, Some(json!({"id": "conv_xyz"})));
    }

    #[test]
    fn extracts_context_management() {
        let body = json!({
            "model": "gpt-4o",
            "input": "test",
            "context_management": {"type": "truncation", "max_tokens": 4096}
        });
        let state = ResponsesState::from_request_body(body);
        assert_eq!(
            state.context_management,
            Some(json!({"type": "truncation", "max_tokens": 4096}))
        );
    }

    #[test]
    fn extracts_include() {
        let body = json!({"model": "gpt-4o", "input": "test", "include": ["usage", "file_search_results"]});
        let state = ResponsesState::from_request_body(body);
        assert_eq!(state.include, vec!["usage", "file_search_results"]);
    }

    #[test]
    fn include_defaults_to_empty() {
        let body = json!({"model": "gpt-4o", "input": "test"});
        let state = ResponsesState::from_request_body(body);
        assert!(state.include.is_empty());
    }

    #[test]
    fn extracts_max_tool_calls() {
        let body = json!({"model": "gpt-4o", "input": "test", "max_tool_calls": 5});
        let state = ResponsesState::from_request_body(body);
        assert_eq!(state.max_tool_calls, Some(5));
    }

    #[test]
    fn max_tool_calls_defaults_to_none() {
        let body = json!({"model": "gpt-4o", "input": "test"});
        let state = ResponsesState::from_request_body(body);
        assert!(state.max_tool_calls.is_none());
    }

    #[test]
    fn parallel_tool_calls_defaults_to_true() {
        let body = json!({"model": "gpt-4o", "input": "test"});
        let state = ResponsesState::from_request_body(body);
        assert!(state.parallel_tool_calls);
    }

    #[test]
    fn default_produces_expected_values() {
        let state = ResponsesState::default();
        assert!(state.context_management.is_none());
        assert!(state.continuation_tool_choice.is_none());
        assert_eq!(state.continuation_output_count, 0);
        assert!(state.local_file_search_persistence_start.is_none());
        assert!(state.conversation.is_none());
        assert!(state.include.is_empty());
        assert!(state.input.is_empty());
        assert_eq!(state.iteration, 0);
        assert!(state.max_tool_calls.is_none());
        assert!(state.mcp_tool_map.is_empty());
        assert!(state.messages.is_empty());
        assert!(state.output_items().is_empty());
        assert!(state.parallel_tool_calls);
        assert!(state.persisted_messages.is_empty());
        assert!(state.previous_response_id.is_none());
        assert!(state.previous_tools.is_empty());
        assert!(state.previous_usage.is_none());
        assert!(state.request_body.is_null());
        assert!(state.response_object.is_null());
        assert!(state.tool_calls.is_empty());
        assert_eq!(state.tool_choice, json!("auto"));
        assert!(!state.tool_choice_present);
        assert!(state.tools.is_empty());
        assert!(state.usage.is_null());
    }

    #[test]
    fn response_object_is_the_single_output_owner() {
        let first = json!({"type": "message", "id": "msg_1"});
        let second = json!({"type": "reasoning", "id": "rs_1"});
        let mut state = ResponsesState {
            response_object: json!({"id": "resp_1", "output": [first.clone()]}),
            ..Default::default()
        };

        state.output_items_mut().push(second.clone());
        assert_eq!(state.output_items(), &[first.clone(), second.clone()]);
        assert_eq!(state.response_object["output"], json!([first, second]));

        let moved = state.take_output_items();
        assert_eq!(moved.len(), 2);
        assert!(state.output_items().is_empty());
        assert_eq!(state.response_object["output"], json!([]));

        state.replace_output_items(moved);
        assert_eq!(state.output_items().len(), 2);
        assert_eq!(state.response_object["id"], "resp_1");
    }

    #[test]
    fn mutable_output_normalizes_missing_or_malformed_response_output() {
        let mut state = ResponsesState {
            response_object: json!({"id": "resp_1", "output": "invalid"}),
            ..Default::default()
        };

        state.output_items_mut().push(json!({"type": "message"}));

        assert_eq!(state.output_items().len(), 1);
        assert!(state.response_object["output"].is_array());
        assert_eq!(state.response_object["id"], "resp_1");
    }

    #[test]
    fn parallel_tool_calls_explicit_false() {
        let body = json!({"model": "gpt-4o", "input": "test", "parallel_tool_calls": false});
        let state = ResponsesState::from_request_body(body);
        assert!(!state.parallel_tool_calls);
    }

    #[test]
    fn default_has_empty_citation_files() {
        let state = ResponsesState::default();
        assert!(
            state.citation_files.is_empty(),
            "initial citation_files should be empty"
        );
    }

    #[test]
    fn default_has_empty_mcp_tool_map() {
        let state = ResponsesState::default();
        assert!(state.mcp_tool_map.is_empty(), "default mcp_tool_map should be empty");
    }

    #[test]
    fn from_request_body_has_empty_mcp_tool_map() {
        let body = json!({"model": "gpt-4o", "input": "test"});
        let state = ResponsesState::from_request_body(body);
        assert!(state.mcp_tool_map.is_empty(), "initial mcp_tool_map should be empty");
    }

    #[test]
    fn from_request_body_has_empty_citation_files() {
        let body = json!({"model": "gpt-4o", "input": "test"});
        let state = ResponsesState::from_request_body(body);
        assert!(
            state.citation_files.is_empty(),
            "citation_files should be empty on construction"
        );
    }
}
