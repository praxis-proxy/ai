// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Rehydrate filter: validates `previous_response_id` by
//! fetching the stored response, confirming its status is
//! `"completed"`, and populating [`ResponsesState`] with the
//! full conversation history (stored turns + current input).
//!
//! The request body is **not** modified; downstream filters
//! read from `ResponsesState.messages` instead.
//!
//! [`ResponsesState`]: super::state::ResponsesState

use std::collections::HashSet;

use async_trait::async_trait;
use bytes::Bytes;
use praxis_filter::{
    FilterAction, FilterError, HttpFilter, HttpFilterContext,
    body::{BodyAccess, BodyMode, MAX_JSON_BODY_BYTES},
    parse_filter_config,
};
use serde_json::Value;
use tracing::{debug, trace, warn};

use super::{
    DEFAULT_STORE_NAME, DEFAULT_TENANT_ID, TENANT_METADATA_KEY, canonical_openresponses_replay_item,
    error::responses_error_rejection, state::ResponsesState,
};
use crate::store::{ConversationRecord, ResponseRecord, ResponseStoreRegistry};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Metadata key for previous response input token count.
const PREV_USAGE_INPUT_KEY: &str = "responses.previous_usage_input_tokens";

/// Metadata key for previous response output token count.
const PREV_USAGE_OUTPUT_KEY: &str = "responses.previous_usage_output_tokens";

/// Metadata key for previous response total token count.
const PREV_USAGE_TOTAL_KEY: &str = "responses.previous_usage_total_tokens";

// -----------------------------------------------------------------------------
// RehydrateFilter
// -----------------------------------------------------------------------------

/// Validates `previous_response_id` by fetching the stored
/// response, confirming its status is `"completed"`, and
/// populating `ResponsesState` with the full conversation
/// history (stored turns + current input).
///
/// The request body is **not** modified; downstream filters
/// read from `ResponsesState.messages` instead.
///
/// # YAML
///
/// ```yaml
/// filter: openai_responses_rehydrate
/// ```
pub struct RehydrateFilter {
    /// Maximum serialized byte size of stored conversation history.
    max_history_bytes: usize,
    /// Optional cap on the number of stored history items.
    max_history_items: Option<usize>,
}

impl RehydrateFilter {
    /// Create a filter from YAML config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the YAML config contains unknown
    /// fields, or `max_history_bytes` / `max_history_items` is zero.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let empty = serde_yaml::Value::Mapping(serde_yaml::Mapping::new());
        let cfg = if config.is_null() { &empty } else { config };
        let validated: RehydrateConfig = parse_filter_config("openai_responses_rehydrate", cfg)?;
        if validated.max_history_bytes == 0 {
            return Err(FilterError::from(
                "openai_responses_rehydrate: max_history_bytes must be greater than 0",
            ));
        }
        if validated.max_history_items == Some(0) {
            return Err(FilterError::from(
                "openai_responses_rehydrate: max_history_items must be greater than 0",
            ));
        }
        Ok(Box::new(Self {
            max_history_bytes: validated.max_history_bytes,
            max_history_items: validated.max_history_items,
        }))
    }

    /// Parse body, resolve rehydration source (`previous_response_id` or
    /// `conversation`), and populate [`ResponsesState`] with the full
    /// conversation history.
    ///
    /// `previous_response_id` takes precedence when both fields are
    /// present.
    async fn rehydrate(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &Option<Bytes>,
        streaming: bool,
    ) -> Result<FilterAction, FilterError> {
        let Some(bytes) = body.as_ref() else {
            return Ok(FilterAction::Release);
        };
        match parse_body_and_extract_id(bytes, streaming) {
            Ok((body, Some(id))) => self.rehydrate_from_response(ctx, body, id, streaming).await,
            Ok((body, None)) => self.rehydrate_from_conversation(ctx, body, streaming).await,
            Err(action) => Ok(action),
        }
    }

    /// Rehydrate from a stored response.
    async fn rehydrate_from_response(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        parsed_body: Value,
        prev_id: String,
        streaming: bool,
    ) -> Result<FilterAction, FilterError> {
        let tenant_id = ctx
            .get_metadata(TENANT_METADATA_KEY)
            .unwrap_or(DEFAULT_TENANT_ID)
            .to_owned();
        let record = match fetch_and_validate_previous(ctx, &tenant_id, &prev_id, streaming).await {
            Ok(r) => r,
            Err(action) => return Ok(action),
        };
        let stored =
            match stored_messages_for_response(&record, self.max_history_bytes, self.max_history_items, streaming) {
                Ok(s) => s,
                Err(action) => return Ok(action),
            };
        let previous_tools = collect_mcp_tool_listings(&record);
        let previous_usage = record.response_object.get("usage").filter(|u| !u.is_null()).cloned();
        let state = build_state(parsed_body, stored, previous_tools, previous_usage);
        write_previous_usage_metadata(ctx, state.previous_usage.as_ref());
        ctx.extensions.insert(state);
        debug!(previous_response_id = %prev_id, "previous response validated, state populated");
        ctx.set_metadata("responses.previous_response_id", prev_id);
        Ok(FilterAction::Release)
    }

    /// Rehydrate from a stored conversation when no `previous_response_id`
    /// is present.
    async fn rehydrate_from_conversation(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        parsed_body: Value,
        streaming: bool,
    ) -> Result<FilterAction, FilterError> {
        let conv_id = match resolve_conversation_id(&parsed_body, streaming) {
            Ok(id) => id,
            Err(action) => return Ok(action),
        };
        let tenant_id = ctx
            .get_metadata(TENANT_METADATA_KEY)
            .unwrap_or(DEFAULT_TENANT_ID)
            .to_owned();
        let record = match fetch_conversation(ctx, &tenant_id, &conv_id, streaming).await {
            Ok(r) => r,
            Err(action) => return Ok(action),
        };
        let stored = match stored_messages_for_conversation(
            &record,
            self.max_history_bytes,
            self.max_history_items,
            streaming,
        ) {
            Ok(s) => s,
            Err(action) => return Ok(action),
        };
        let state = build_state(parsed_body, stored, vec![], None);
        write_previous_usage_metadata(ctx, state.previous_usage.as_ref());
        ctx.extensions.insert(state);
        debug!(conversation_id = %conv_id, "conversation rehydrated, state populated");
        Ok(FilterAction::Release)
    }
}

/// YAML configuration for [`RehydrateFilter`].
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RehydrateConfig {
    /// Maximum serialized byte size of stored conversation history. Default: 2,097,152 (2 MiB).
    #[serde(default = "default_max_history_bytes")]
    max_history_bytes: usize,
    /// Optional cap on the number of stored history items.
    #[serde(default)]
    max_history_items: Option<usize>,
}

/// Default maximum byte size for stored conversation history (2 MiB).
fn default_max_history_bytes() -> usize {
    2_097_152 // 2 MiB
}

#[async_trait]
impl HttpFilter for RehydrateFilter {
    fn name(&self) -> &'static str {
        "openai_responses_rehydrate"
    }

    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    /// `StreamBuffer` so the protocol layer assembles the complete
    /// request body before delivering it at end-of-stream.
    fn request_body_mode(&self) -> BodyMode {
        BodyMode::StreamBuffer {
            max_bytes: Some(MAX_JSON_BODY_BYTES),
        }
    }

    async fn on_request(&self, _ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        Ok(FilterAction::Continue)
    }

    async fn on_request_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if !end_of_stream {
            return Ok(FilterAction::Continue);
        }

        if ctx.request.method != http::Method::POST {
            return Ok(FilterAction::Continue);
        }

        if is_responses_cancel_path(ctx.request.uri.path()) {
            return Ok(FilterAction::Release);
        }

        if ctx.get_metadata("openai_responses_format.format") != Some("openai_responses") {
            return Ok(FilterAction::Release);
        }

        let streaming = ctx
            .get_metadata("openai_responses_format.stream")
            .is_some_and(|v| v == "true");

        self.rehydrate(ctx, body, streaming).await
    }
}

/// Return whether this request targets the body-less Responses cancel endpoint.
fn is_responses_cancel_path(path: &str) -> bool {
    let path = path.trim_end_matches('/');

    let Some(response_id) = path
        .strip_prefix("/v1/responses/")
        .and_then(|rest| rest.strip_suffix("/cancel"))
    else {
        return false;
    };

    !response_id.is_empty() && !response_id.contains('/')
}

/// Reject when `items` exceeds the configured byte-size or item-count cap.
fn check_history_limits(
    items: &[Value],
    max_bytes: usize,
    max_items: Option<usize>,
    streaming: bool,
) -> Result<(), FilterAction> {
    if let Some(max) = max_items {
        let count = items.len();
        if count > max {
            return Err(reject_too_large(
                &format!(
                    "stored conversation history contains {count} items, \
                   exceeding the {max} item limit; \
                   compact or shorten the conversation before continuing"
                ),
                streaming,
            ));
        }
    }

    let byte_size = serde_json::to_string(items).map_or(usize::MAX, |s| s.len());
    if byte_size > max_bytes {
        return Err(reject_too_large(
            &format!(
                "stored conversation history is {byte_size} bytes, \
               exceeding the {max_bytes} byte limit; \
               compact or shorten the conversation before continuing"
            ),
            streaming,
        ));
    }

    Ok(())
}

// -----------------------------------------------------------------------------
// Stored message extraction
// -----------------------------------------------------------------------------

/// Stored messages from a response record, checking limits before cloning.
fn stored_messages_for_response(
    record: &ResponseRecord,
    max_bytes: usize,
    max_items: Option<usize>,
    streaming: bool,
) -> Result<Vec<Value>, FilterAction> {
    if let Some(messages) = record.messages.as_array().filter(|a| !a.is_empty()) {
        check_history_limits(messages, max_bytes, max_items, streaming)?;
        return Ok(messages.clone());
    }
    reconstruct_messages_from_public_response(record, max_bytes, max_items, streaming)
}

/// Stored messages from a conversation record, checking limits before cloning.
fn stored_messages_for_conversation(
    record: &ConversationRecord,
    max_bytes: usize,
    max_items: Option<usize>,
    streaming: bool,
) -> Result<Vec<Value>, FilterAction> {
    let empty: &[Value] = &[];
    let messages = record.messages.as_array().map_or(empty, Vec::as_slice);
    check_history_limits(messages, max_bytes, max_items, streaming)?;
    Ok(messages.to_vec())
}

/// Fetch the previous response and validate its status in one step.
async fn fetch_and_validate_previous(
    ctx: &HttpFilterContext<'_>,
    tenant_id: &str,
    prev_id: &str,
    streaming: bool,
) -> Result<ResponseRecord, FilterAction> {
    let record = fetch_previous_response(ctx, tenant_id, prev_id, streaming).await?;
    validate_response_status(&record, streaming)?;
    Ok(record)
}

/// Resolve the conversation ID from the request body, returning a
/// `Release` when no conversation field is present or a `Reject`
/// when the field is malformed.
fn resolve_conversation_id(body: &Value, streaming: bool) -> Result<String, FilterAction> {
    let has_field = body.get("conversation").is_some();
    extract_conversation_id(body).ok_or_else(|| {
        if has_field {
            FilterAction::Reject(responses_error_rejection(
                400,
                "invalid_request_error",
                "invalid conversation value: expected a string ID or {\"id\": \"...\"}",
                streaming,
            ))
        } else {
            FilterAction::Release
        }
    })
}

/// Extract a conversation ID from the request body.
///
/// Accepts both string and object forms:
/// - `"conversation": "conv_abc"`
/// - `"conversation": {"id": "conv_abc"}`
fn extract_conversation_id(body: &Value) -> Option<String> {
    body.get("conversation").and_then(|c| {
        c.as_str()
            .or_else(|| c.get("id").and_then(Value::as_str))
            .map(ToOwned::to_owned)
    })
}

/// Fetch a conversation record from the store.
async fn fetch_conversation(
    ctx: &HttpFilterContext<'_>,
    tenant_id: &str,
    conv_id: &str,
    streaming: bool,
) -> Result<ConversationRecord, FilterAction> {
    let registry = ctx.extensions.get::<ResponseStoreRegistry>().ok_or_else(|| {
        warn!("rehydrate: response store registry not available");
        reject_server_error("response store is not available", streaming)
    })?;

    let store = registry.get(DEFAULT_STORE_NAME).ok_or_else(|| {
        warn!("rehydrate: default response store not registered");
        reject_server_error("response store is not available", streaming)
    })?;

    let record = store.get_conversation(tenant_id, conv_id).await.map_err(|e| {
        warn!(error = %e, "rehydrate: failed to fetch conversation");
        reject_server_error("failed to fetch conversation", streaming)
    })?;

    record.ok_or_else(|| {
        debug!(id = %conv_id, "rehydrate: conversation not found");
        reject_invalid(&format!("conversation '{conv_id}' not found"), streaming)
    })
}

/// Build [`ResponsesState`] by prepending stored messages before the current input.
fn build_state(
    parsed_body: Value,
    stored: Vec<Value>,
    previous_tools: Vec<Value>,
    previous_usage: Option<Value>,
) -> ResponsesState {
    let replay = replay_messages_from_stored(&stored);
    let mut state = ResponsesState::from_request_body(parsed_body);
    state.history_rehydrated = true;
    state.messages.splice(0..0, replay);
    state.persisted_messages.splice(0..0, stored);
    state.previous_tools = previous_tools;
    state.previous_usage = previous_usage;
    state
}

/// Return stored history, reconstructing from public fields for
/// records created before hidden messages were persisted.
fn reconstruct_messages_from_public_response(
    record: &ResponseRecord,
    max_bytes: usize,
    max_items: Option<usize>,
    streaming: bool,
) -> Result<Vec<Value>, FilterAction> {
    let mut messages = Vec::new();

    append_stored_input_items(&mut messages, record.input.clone());

    if let Some(output) = record.response_object.get("output").filter(|output| !output.is_null()) {
        append_stored_output_items(&mut messages, output);
    }

    check_history_limits(&messages, max_bytes, max_items, streaming)?;
    Ok(messages)
}

/// Append stored response input as Responses API item params.
fn append_stored_input_items(messages: &mut Vec<Value>, input: Value) {
    match input {
        Value::Null => {},
        Value::String(text) => messages.push(user_message_item(&text)),
        Value::Array(items) => messages.extend(items),
        other => messages.push(other),
    }
}

/// Append stored response output items to the persisted conversation history.
fn append_stored_output_items(messages: &mut Vec<Value>, output: &Value) {
    if let Value::Array(items) = output {
        messages.extend(items.iter().cloned());
    } else {
        messages.push(output.clone());
    }
}

/// Return stored items that should be replayed as backend request input.
fn replay_messages_from_stored(stored: &[Value]) -> Vec<Value> {
    stored.iter().filter_map(canonical_openresponses_replay_item).collect()
}

/// Build a Responses API user message item from string input.
fn user_message_item(text: &str) -> Value {
    serde_json::json!({
        "type": "message",
        "role": "user",
        "content": text,
    })
}

/// Parse the request body and extract `previous_response_id`.
///
/// Returns the parsed body alongside the optional ID so callers
/// can reuse it for [`ResponsesState`] construction.
fn parse_body_and_extract_id(bytes: &[u8], streaming: bool) -> Result<(Value, Option<String>), FilterAction> {
    let parsed: Value = serde_json::from_slice(bytes).map_err(|e| {
        debug!(error = %e, "rehydrate: invalid request JSON");
        reject_invalid(&format!("invalid request body: {e}"), streaming)
    })?;

    let id = match parsed.get("previous_response_id") {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) => Some(s.clone()),
        Some(_) => return Err(reject_invalid("previous_response_id must be a string", streaming)),
    };

    Ok((parsed, id))
}

// -----------------------------------------------------------------------------
// Fetch & Validate
// -----------------------------------------------------------------------------

/// Fetch the previous response record from the store.
async fn fetch_previous_response(
    ctx: &HttpFilterContext<'_>,
    tenant_id: &str,
    prev_id: &str,
    streaming: bool,
) -> Result<ResponseRecord, FilterAction> {
    let registry = ctx.extensions.get::<ResponseStoreRegistry>().ok_or_else(|| {
        warn!("rehydrate: response store registry not available");
        reject_server_error("response store is not available", streaming)
    })?;

    let store = registry.get(DEFAULT_STORE_NAME).ok_or_else(|| {
        warn!("rehydrate: default response store not registered");
        reject_server_error("response store is not available", streaming)
    })?;

    let record = store.get_response(tenant_id, prev_id).await.map_err(|e| {
        warn!(error = %e, "rehydrate: failed to fetch previous response");
        reject_server_error("failed to fetch previous response", streaming)
    })?;

    record.ok_or_else(|| {
        debug!(id = %prev_id, "rehydrate: previous response not found");
        reject_invalid(&format!("response '{prev_id}' not found"), streaming)
    })
}

/// Validate that the stored response has status `"completed"`.
fn validate_response_status(record: &ResponseRecord, streaming: bool) -> Result<(), FilterAction> {
    let status = record
        .response_object
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("unknown");

    if status != "completed" {
        return Err(reject_invalid(
            &format!("cannot continue from response with status '{status}'"),
            streaming,
        ));
    }

    Ok(())
}

// -----------------------------------------------------------------------------
// MCP Tool & Usage Extraction
// -----------------------------------------------------------------------------

/// Recover MCP tool listings from stored history and response output.
fn collect_mcp_tool_listings(record: &ResponseRecord) -> Vec<Value> {
    let mut listings = Vec::new();
    let mut seen = HashSet::new();

    if let Some(messages) = record.messages.as_array() {
        collect_mcp_tool_listings_from_items(messages, &mut seen, &mut listings);
    }

    if let Some(output) = record.response_object.get("output").and_then(Value::as_array) {
        collect_mcp_tool_listings_from_items(output, &mut seen, &mut listings);
    }

    listings
}

/// Append MCP tool listings from a sequence of response items.
fn collect_mcp_tool_listings_from_items(
    items: &[Value],
    seen: &mut HashSet<(String, Vec<String>)>,
    listings: &mut Vec<Value>,
) {
    listings.extend(items.iter().filter_map(|item| {
        if item.get("type").and_then(Value::as_str) != Some("mcp_list_tools") {
            return None;
        }

        let label = item.get("server_label").and_then(Value::as_str)?;
        let tools = item.get("tools").and_then(Value::as_array)?;
        let names = mcp_tool_names(tools);
        let mut dedupe_names = names.clone();
        dedupe_names.sort();
        dedupe_names.dedup();

        if !seen.insert((label.to_owned(), dedupe_names)) {
            return None;
        }

        let mut map = serde_json::Map::new();
        map.insert("server_label".to_owned(), Value::String(label.to_owned()));
        map.insert("tools".to_owned(), Value::Array(tools.clone()));
        if let Some(url) = item.get("server_url").and_then(Value::as_str) {
            map.insert("server_url".to_owned(), Value::String(url.to_owned()));
        }
        Some(Value::Object(map))
    }));
}

/// Extract tool names from MCP tool definitions.
fn mcp_tool_names(tools: &[Value]) -> Vec<String> {
    tools
        .iter()
        .filter_map(|tool| tool.get("name").and_then(Value::as_str).map(ToOwned::to_owned))
        .collect()
}

/// Extract token usage from the previous response and set
/// metadata keys for downstream auto-compaction.
///
/// Writes `input_tokens`, `output_tokens`, and `total_tokens` as
/// individual string metadata values when present.
fn write_previous_usage_metadata(ctx: &mut HttpFilterContext<'_>, usage: Option<&Value>) {
    let Some(usage) = usage else {
        return;
    };

    if let Some(input) = usage.get("input_tokens").and_then(Value::as_u64) {
        ctx.set_metadata(PREV_USAGE_INPUT_KEY, input.to_string());
    }

    if let Some(output) = usage.get("output_tokens").and_then(Value::as_u64) {
        ctx.set_metadata(PREV_USAGE_OUTPUT_KEY, output.to_string());
    }

    if let Some(total) = usage.get("total_tokens").and_then(Value::as_u64) {
        ctx.set_metadata(PREV_USAGE_TOTAL_KEY, total.to_string());
    }

    trace!("extracted previous response usage");
}

// -----------------------------------------------------------------------------
// Rejection Helpers
// -----------------------------------------------------------------------------

/// Build a 400 rejection with a Responses API error body.
fn reject_invalid(message: &str, streaming: bool) -> FilterAction {
    FilterAction::Reject(responses_error_rejection(
        400,
        "invalid_request_error",
        message,
        streaming,
    ))
}

/// Build a 500 rejection with a Responses API error body.
fn reject_server_error(message: &str, streaming: bool) -> FilterAction {
    FilterAction::Reject(responses_error_rejection(500, "server_error", message, streaming))
}

/// Build a 413 rejection with a Responses API error body.
fn reject_too_large(message: &str, streaming: bool) -> FilterAction {
    FilterAction::Reject(responses_error_rejection(
        413,
        "invalid_request_error",
        message,
        streaming,
    ))
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
    clippy::needless_pass_by_value,
    clippy::panic,
    clippy::needless_raw_strings,
    clippy::needless_raw_string_hashes,
    clippy::too_many_lines,
    reason = "tests"
)]
mod tests;
