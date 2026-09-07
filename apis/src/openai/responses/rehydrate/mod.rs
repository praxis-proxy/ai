// SPDX-License-Identifier: Apache-2.0
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
    DEFAULT_STORE_NAME, DEFAULT_TENANT_ID, TENANT_METADATA_KEY, append_stored_input_items,
    canonical_openresponses_replay_item, error::responses_error_rejection, extract_conversation_id,
    state::ResponsesState,
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
        let mut state = build_state(parsed_body, stored, previous_tools, previous_usage);
        state.response_id = ctx.get_metadata("responses.response_id").map(ToOwned::to_owned);
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
        let mut state = build_state(parsed_body, stored, vec![], None);
        state.response_id = ctx.get_metadata("responses.response_id").map(ToOwned::to_owned);
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

    /// `ReadWrite` so the response phase can restore the caller's
    /// `previous_response_id` into the response body.
    fn response_body_access(&self) -> BodyAccess {
        BodyAccess::ReadWrite
    }

    /// Streaming by default. A finite JSON response that requires the
    /// `previous_response_id` restore selects a bounded `StreamBuffer`
    /// dynamically in [`Self::on_response`].
    fn response_body_mode(&self) -> BodyMode {
        BodyMode::Stream
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

    async fn on_response(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        // The response header is only available in this header phase, so the
        // eligibility decision (status + content-type + rehydration state) is
        // made here and carried into `on_response_body` via filter state.
        let Some(prev_id) = eligible_previous_response_id(ctx) else {
            return Ok(FilterAction::Continue);
        };

        // Buffer the finite JSON response so `on_response_body` can restore the
        // caller's `previous_response_id`. Re-serializing the body invalidates
        // every header that describes the exact upstream bytes, so drop them all
        // here: `Content-Length` (core recomputes it from the buffered body) and
        // the body validators / integrity digests (`ETag`, `Last-Modified`,
        // `Content-MD5`, `Digest`, `Content-Digest`, `Repr-Digest`). Leaving a
        // validator behind would assert the rewritten payload is byte-identical
        // to the upstream representation and break cache revalidation or integrity
        // verification — rewriting `previous_response_id` changes both bytes and
        // API-visible semantics, so those validators no longer describe this
        // response. (`Last-Modified` is a weak, mtime-granularity validator, but
        // it still names the origin representation and is dropped for symmetry.)
        //
        // `Content-Encoding` and `Content-Range` are deliberately left alone:
        // `eligible_previous_response_id` already declined encoded and ranged
        // responses, so an eligible body is a complete, identity-coded
        // representation. Caching-policy (`Cache-Control`, `Age`, ...), routing,
        // tracing, and `Content-Type` headers are unrelated to the byte content
        // and are preserved so the response reaches the client almost unchanged.
        ctx.set_response_body_mode(BodyMode::StreamBuffer {
            max_bytes: Some(MAX_JSON_BODY_BYTES),
        });
        if let Some(response) = &mut ctx.response_header {
            let headers = &mut response.headers;
            headers.remove(http::header::CONTENT_LENGTH);
            headers.remove(http::header::ETAG);
            headers.remove(http::header::LAST_MODIFIED);
            // `Content-MD5` (RFC 1864, obsolete) and the RFC 9530 digest fields
            // have no typed constants in `http`; remove them by lowercase name.
            headers.remove("content-md5");
            headers.remove("digest");
            headers.remove("content-digest");
            headers.remove("repr-digest");
        }
        ctx.response_headers_modified = true;
        ctx.insert_filter_state(RestorePreviousResponseId {
            previous_response_id: prev_id,
        });

        Ok(FilterAction::Continue)
    }

    fn on_response_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if !end_of_stream {
            return Ok(FilterAction::Continue);
        }

        // `response_header` is gone by the body phase; the armed decision from
        // `on_response` is the sole signal that a restore is required.
        let Some(armed) = ctx.remove_filter_state::<RestorePreviousResponseId>() else {
            return Ok(FilterAction::Continue);
        };

        restore_previous_response_id(armed.previous_response_id, body);
        Ok(FilterAction::Continue)
    }
}

/// Marker carrying the caller's `previous_response_id` from the response header
/// phase to the response body phase.
///
/// The eligibility decision requires `ctx.response_header`, which is only
/// present during [`RehydrateFilter::on_response`]; this state hands the ID to
/// [`RehydrateFilter::on_response_body`], where the header is no longer available.
struct RestorePreviousResponseId {
    /// The `previous_response_id` the caller supplied, to echo back.
    previous_response_id: String,
}

/// Return the caller's `previous_response_id` when it must be restored into the
/// response body, or `None` when the response is ineligible.
///
/// When history was rehydrated, the proxy strips `previous_response_id` from the
/// upstream request (prior turns are replayed via the `input` array and the
/// backend never sees the ID), so the backend echoes `null`. The Responses API
/// contract always echoes the caller's `previous_response_id`, so it is restored
/// on the way out. Only a finite, identity-coded, complete `200 OK` JSON response
/// is eligible; streaming SSE, content-encoded, ranged/partial, and non-`200`
/// responses are left untouched.
fn eligible_previous_response_id(ctx: &HttpFilterContext<'_>) -> Option<String> {
    // Clone the ID at the ownership boundary so it outlives the borrow on
    // `extensions` and can be carried into the body phase.
    let prev_id = ctx
        .extensions
        .get::<ResponsesState>()
        .filter(|state| state.history_rehydrated)
        .and_then(|state| state.previous_response_id.clone())?;

    let resp = ctx.response_header.as_ref()?;
    // Require an ordinary `200 OK`. Restoring the ID re-serializes the full body,
    // so any other success shape is not a complete, self-contained Responses
    // resource to rewrite: `206 Partial Content` is a byte fragment, `204 No
    // Content` has no body, and `202 Accepted` background handoffs are echoed
    // verbatim. All are passed through untouched.
    if resp.status != http::StatusCode::OK {
        return None;
    }

    // Decline any response carrying a `Content-Encoding` or `Content-Range`.
    // Restoring the ID means parsing the buffered body as JSON and re-serializing
    // it: a compressed body is opaque bytes that would fail to parse, and a
    // ranged (partial) body is a fragment of a larger representation that cannot
    // be soundly rewritten. Dropping the framing headers while leaving the body
    // encoded or partial would ship those bytes mislabeled as a full, identity
    // JSON representation. Leaving such a response ineligible passes it through
    // verbatim. (Mirrors the encoded-SSE decline in `openai_responses`
    // stream_events; see issue #668 for the same defense on the streaming path.)
    if resp.headers.contains_key(http::header::CONTENT_ENCODING)
        || resp.headers.contains_key(http::header::CONTENT_RANGE)
    {
        return None;
    }

    let content_type = resp
        .headers
        .get(http::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default();
    let is_json = content_type
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .eq_ignore_ascii_case("application/json");

    is_json.then_some(prev_id)
}

/// Restore the caller's `previous_response_id` into the buffered Responses
/// resource, replacing the `null` the backend echoed after the proxy stripped
/// the ID from the upstream request.
fn restore_previous_response_id(prev_id: String, body: &mut Option<Bytes>) {
    let Some(bytes) = body.as_ref() else {
        return;
    };
    let Ok(mut parsed) = serde_json::from_slice::<Value>(bytes) else {
        debug!("rehydrate: response body is not JSON, skipping previous_response_id restore");
        return;
    };
    let Some(object) = parsed.as_object_mut() else {
        return;
    };
    // Only rewrite a Responses resource, never an unexpected success shape.
    if object.get("object").and_then(Value::as_str) != Some("response") {
        return;
    }

    object.insert("previous_response_id".to_owned(), Value::String(prev_id));
    match serde_json::to_vec(&parsed) {
        Ok(serialized) => {
            *body = Some(Bytes::from(serialized));
            trace!("restored caller previous_response_id into response body");
        },
        Err(error) => warn!(error = %error, "rehydrate: failed to re-serialize response body"),
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
