// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Responses API proxy filter.
//!
//! Body-preparation waypoint in the Responses API filter pipeline.
//! Sits between upstream enrichment filters (`rehydrate`, `openai_tool_parse`)
//! and downstream consumption filters (`stream_events`, `tool_dispatch`).
//! Named `inference` in pipeline configs so branch chains can
//! `rejoin` here for the agentic tool loop.
//!
//! When `ResponsesState` is present in `RequestExtensions`, replaces
//! the request input with `state.messages` only after conversation
//! history has changed it. Provider-owned conversation continuations
//! send only their new message delta. It strips `previous_response_id`
//! and `conversation` only after local rehydration consumes them.

mod config;
mod selected_input;

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::needless_raw_strings,
    clippy::needless_raw_string_hashes,
    clippy::too_many_lines,
    reason = "tests"
)]
mod tests;

use std::{borrow::Cow, collections::HashSet};

use async_trait::async_trait;
use base64::Engine as _;
use bytes::Bytes;
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, SelectedUpstreamBodyOutcome,
    SubRequestResponseMode, body::MAX_JSON_BODY_BYTES, parse_filter_config,
};
use serde::{Deserialize, ser::SerializeMap as _};
use tracing::{debug, trace};

use self::config::{ResponsesProxyConfig, build_config};
use super::{
    body_limits::reject_rewritten_body_too_large,
    enforce_agentic_stream_guard,
    error::responses_error_rejection,
    state::{ResponsesState, normalize_input_owned},
};
use crate::json_body::SerializedJson;

// -----------------------------------------------------------------------------
// ResponsesProxyFilter
// -----------------------------------------------------------------------------

/// Rebuilds the request body from `ResponsesState` when present.
///
/// Reads the assembled conversation history from
/// `ResponsesState::messages` and replaces the `input` field in
/// the outbound body when it differs from the original normalized
/// input. Strips `previous_response_id` after Praxis resolves it
/// locally via the rehydrate filter.
///
/// When no `ResponsesState` exists, preserves the request body unchanged.
///
/// This filter always advertises the Praxis streaming capability. When the
/// effective outbound body contains `"stream": true` it selects Praxis's
/// streaming transport; otherwise it selects the buffered transport. There is
/// no operator opt-in — the removed `terminal_streaming` flag is rejected via
/// `deny_unknown_fields` so stale configs fail to build. Classifier metadata
/// remains descriptive client intent; this final serializer owns the transport
/// decision. IRR can resume one downstream stream across response-dependent
/// transitions, but every response-body filter in a step composed with this
/// filter must use `BodyMode::Stream` (or explicitly reject streaming requests)
/// rather than silently buffering them.
///
/// # YAML
///
/// ```yaml
/// filter: openai_responses_proxy
/// ```
///
/// # Full YAML
///
/// ```yaml
/// filter: openai_responses_proxy
/// max_rewritten_body_bytes: 67108864
/// ```
///
/// # Example
///
/// ```rust
/// use praxis_ai_apis::openai::ResponsesProxyFilter;
///
/// let yaml = serde_yaml::Value::Null;
/// let filter = ResponsesProxyFilter::from_config(&yaml).unwrap();
/// assert_eq!(filter.name(), "openai_responses_proxy");
/// ```
pub struct ResponsesProxyFilter {
    /// Parsed and validated configuration.
    config: ResponsesProxyConfig,
}

impl ResponsesProxyFilter {
    /// Create from parsed YAML config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the YAML config contains unknown fields.
    ///
    /// [`FilterError`]: praxis_filter::FilterError
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: ResponsesProxyConfig = if config.is_null() {
            ResponsesProxyConfig::default()
        } else {
            parse_filter_config("openai_responses_proxy", config)?
        };
        let validated = build_config(cfg)?;
        Ok(Box::new(Self { config: validated }))
    }

    /// Serialize the rebuilt body from conversation state.
    fn serialize_body(
        &self,
        state: &ResponsesState,
        preserve_native_compaction: bool,
    ) -> Result<Result<Vec<u8>, FilterAction>, FilterError> {
        let serialized = serialize_outbound_body(state, preserve_native_compaction)
            .map_err(|e| -> FilterError { format!("openai_responses_proxy: {e}").into() })?;
        if serialized.len() > self.config.max_rewritten_body_bytes {
            debug!(
                body_bytes = serialized.len(),
                max_bytes = self.config.max_rewritten_body_bytes,
                "rebuilt request body exceeds maximum size"
            );
            return Ok(Err(reject_rewritten_body_too_large(
                serialized.len(),
                self.config.max_rewritten_body_bytes,
            )));
        }

        debug!(
            messages = state.messages.len(),
            body_bytes = serialized.len(),
            "rebuilt request body from ResponsesState"
        );

        Ok(Ok(serialized))
    }

    /// Reconcile the selected backend's compaction projection with the body
    /// produced by the earlier request-body phase.
    ///
    /// The selected-upstream phase runs after other request-body filters. Use
    /// that live body as the source so downstream changes, such as a model
    /// rewrite, are retained instead of rebuilding from the older
    /// [`ResponsesState`] snapshot.
    #[expect(
        clippy::too_many_lines,
        reason = "streaming splice keeps the large body out of serde_json::Value"
    )]
    fn serialize_selected_body(
        &self,
        body: &Bytes,
        state: &ResponsesState,
        preserve_native_compaction: bool,
    ) -> Result<Result<Vec<u8>, FilterAction>, FilterError> {
        let members = scan_top_level_object(body).map_err(|error| -> FilterError {
            format!("openai_responses_proxy: invalid selected request body: {error}").into()
        })?;
        let messages = selected_input::reconcile(body, &members, state, preserve_native_compaction)?;
        let discovered =
            ResponsesState::provider_compaction_ids_from_messages(messages.iter().filter_map(|item| match item {
                Cow::Owned(item) => Some(item),
                Cow::Borrowed(_) => None,
            }));
        let provider_compaction_ids = if discovered.is_subset(&state.provider_compaction_ids) {
            Cow::Borrowed(&state.provider_compaction_ids)
        } else {
            let mut ids = state.provider_compaction_ids.clone();
            ids.extend(discovered);
            Cow::Owned(ids)
        };
        let input_replacement = serde_json::to_vec(&SelectedMessages {
            messages: &messages,
            preserve_native_compaction,
            provider_compaction_ids: provider_compaction_ids.as_ref(),
        })
        .map_err(|error| -> FilterError { format!("openai_responses_proxy: {error}").into() })?;
        let stream_replacement = state
            .request_body
            .get("stream")
            .map(serde_json::to_vec)
            .transpose()
            .map_err(|error| -> FilterError { format!("openai_responses_proxy: {error}").into() })?;

        let mut serialized = Vec::with_capacity(body.len());
        serialized.push(b'{');
        let mut wrote_member = false;
        let mut wrote_input = false;
        let mut wrote_tools = false;
        let mut wrote_tool_choice = false;
        let mut wrote_reasoning = false;
        for member in members {
            if state.history_rehydrated
                && matches!(
                    member.name,
                    TopLevelField::PreviousResponseId | TopLevelField::Conversation
                )
            {
                continue;
            }
            let state_replacement = selected_state_field(state, member.name)?;
            if wrote_member {
                serialized.push(b',');
            }
            serialized.extend_from_slice(selected_body_slice(body, member.key_start, member.value_start)?);
            if member.name == TopLevelField::Input {
                serialized.extend_from_slice(&input_replacement);
                wrote_input = true;
            } else if let Some(replacement) = state_replacement {
                serialized.extend_from_slice(&replacement);
                wrote_tools |= member.name == TopLevelField::Tools;
                wrote_tool_choice |= member.name == TopLevelField::ToolChoice;
            } else if member.name == TopLevelField::Reasoning {
                let live = selected_body_slice(body, member.value_start, member.value_end)?;
                if let Some(replacement) = selected_reasoning_default(state, Some(live))? {
                    serialized.extend_from_slice(&replacement);
                } else {
                    serialized.extend_from_slice(live);
                }
                wrote_reasoning = true;
            } else if member.name == TopLevelField::Stream {
                if let Some(replacement) = &stream_replacement {
                    serialized.extend_from_slice(replacement);
                } else {
                    serialized.extend_from_slice(selected_body_slice(body, member.value_start, member.value_end)?);
                }
            } else {
                serialized.extend_from_slice(selected_body_slice(body, member.value_start, member.value_end)?);
            }
            wrote_member = true;
        }
        if !wrote_input {
            if wrote_member {
                serialized.push(b',');
            }
            serialized.extend_from_slice(br#""input":"#);
            serialized.extend_from_slice(&input_replacement);
            wrote_member = true;
        }
        let state_fields: [(TopLevelField, bool, &[u8]); 2] = [
            (TopLevelField::Tools, wrote_tools, br#""tools":"#),
            (TopLevelField::ToolChoice, wrote_tool_choice, br#""tool_choice":"#),
        ];
        for (field, wrote, key) in state_fields {
            if wrote {
                continue;
            }
            let Some(replacement) = selected_state_field(state, field)? else {
                continue;
            };
            if wrote_member {
                serialized.push(b',');
            }
            serialized.extend_from_slice(key);
            serialized.extend_from_slice(&replacement);
            wrote_member = true;
        }
        if !wrote_reasoning && let Some(replacement) = selected_reasoning_default(state, None)? {
            if wrote_member {
                serialized.push(b',');
            }
            serialized.extend_from_slice(br#""reasoning":"#);
            serialized.extend_from_slice(&replacement);
        }
        serialized.push(b'}');
        if serialized.len() > self.config.max_rewritten_body_bytes {
            return Ok(Err(reject_rewritten_body_too_large(
                serialized.len(),
                self.config.max_rewritten_body_bytes,
            )));
        }
        Ok(Ok(serialized))
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
/// Recognized fields in the request's top-level JSON object.
enum TopLevelField {
    /// The Responses input array.
    Input,
    /// The effective streaming flag.
    Stream,
    /// Provider-visible tool declarations owned by [`ResponsesState`].
    Tools,
    /// Provider-visible tool choice owned by [`ResponsesState`].
    ToolChoice,
    /// Live reasoning settings that may inherit a stored effort default.
    Reasoning,
    /// A locally consumed response selector.
    PreviousResponseId,
    /// A locally consumed conversation selector.
    Conversation,
    /// Any field preserved byte-for-byte.
    Other,
}

/// A top-level JSON member and the byte ranges needed for splicing it.
struct TopLevelMember {
    /// The recognized field name.
    name: TopLevelField,
    /// Start of the encoded field name.
    key_start: usize,
    /// Start of the encoded field value.
    value_start: usize,
    /// End of the encoded field value.
    value_end: usize,
}

/// Return the state-owned replacement for a selected-body member, if one is
/// required by the current rebuild.
fn selected_state_field(state: &ResponsesState, field: TopLevelField) -> Result<Option<Vec<u8>>, FilterError> {
    let key = match field {
        TopLevelField::Tools => "tools",
        TopLevelField::ToolChoice => "tool_choice",
        _ => return Ok(None),
    };
    if !state.request_body_requires_rebuild() {
        return Ok(None);
    }
    let Some(value) = state.request_body.get(key) else {
        return Ok(None);
    };
    serde_json::to_vec(value)
        .map(Some)
        .map_err(|error| format!("openai_responses_proxy: {error}").into())
}

/// Fill only an omitted effort in locally rehydrated requests. Live settings,
/// including explicit nulls and later filter edits, take precedence over state.
fn selected_reasoning_default(state: &ResponsesState, live: Option<&[u8]>) -> Result<Option<Vec<u8>>, FilterError> {
    if !state.history_rehydrated {
        return Ok(None);
    }
    let Some(effort) = state
        .request_body
        .get("reasoning")
        .and_then(|reasoning| reasoning.get("effort"))
    else {
        return Ok(None);
    };
    let reasoning = if let Some(live) = live {
        let mut reasoning: serde_json::Value = serde_json::from_slice(live)
            .map_err(|error| -> FilterError { format!("openai_responses_proxy: {error}").into() })?;
        let Some(object) = reasoning.as_object_mut() else {
            return Ok(None);
        };
        if object.contains_key("effort") {
            return Ok(None);
        }
        // Only the small default needs ownership in the newly parsed reasoning object.
        object.insert("effort".to_owned(), effort.clone());
        reasoning
    } else {
        serde_json::json!({"effort": effort})
    };
    serde_json::to_vec(&reasoning)
        .map(Some)
        .map_err(|error| format!("openai_responses_proxy: {error}").into())
}

/// Borrow a validated byte range from the selected request body.
fn selected_body_slice(body: &[u8], start: usize, end: usize) -> Result<&[u8], FilterError> {
    body.get(start..end)
        .ok_or_else(|| "openai_responses_proxy: invalid selected request body range".into())
}

/// Borrow reconciled messages and translate only individual summaries.
struct SelectedMessages<'a> {
    /// Borrowed history and the few owned live edits.
    messages: &'a [Cow<'a, serde_json::Value>],
    /// Selected native Responses protocol.
    preserve_native_compaction: bool,
    /// Known provider-originated compaction IDs.
    provider_compaction_ids: &'a HashSet<String>,
}

impl serde::Serialize for SelectedMessages<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq as _;
        let mut sequence = serializer.serialize_seq(Some(self.messages.len()))?;
        for item in self.messages {
            let projected = messages_for_backend(
                std::slice::from_ref(item.as_ref()),
                self.preserve_native_compaction,
                self.provider_compaction_ids,
            );
            for item in projected.iter() {
                sequence.serialize_element(item)?;
            }
        }
        sequence.end()
    }
}

/// Locate top-level members without materializing the complete JSON value.
#[expect(
    clippy::too_many_lines,
    reason = "scanner handles objects, strings, and separators explicitly"
)]
fn scan_top_level_object(body: &[u8]) -> Result<Vec<TopLevelMember>, &'static str> {
    let mut cursor = skip_json_whitespace(body, 0);
    if body.get(cursor) != Some(&b'{') {
        return Err("expected a JSON object");
    }
    cursor = skip_json_whitespace(body, cursor + 1);
    let mut members = Vec::new();
    if body.get(cursor) == Some(&b'}') {
        return Ok(members);
    }

    loop {
        let key_start = cursor;
        let key_end = scan_json_string(body, cursor)?;
        let key_bytes = body.get(key_start..key_end).ok_or("invalid JSON field range")?;
        let key = serde_json::from_slice::<String>(key_bytes).map_err(|_error| "invalid JSON field name")?;
        cursor = skip_json_whitespace(body, key_end);
        if body.get(cursor) != Some(&b':') {
            return Err("expected a colon after a JSON field name");
        }
        let value_start = skip_json_whitespace(body, cursor + 1);
        let value_end = scan_json_value(body, value_start)?;
        members.push(TopLevelMember {
            name: match key.as_str() {
                "input" => TopLevelField::Input,
                "stream" => TopLevelField::Stream,
                "tools" => TopLevelField::Tools,
                "tool_choice" => TopLevelField::ToolChoice,
                "reasoning" => TopLevelField::Reasoning,
                "previous_response_id" => TopLevelField::PreviousResponseId,
                "conversation" => TopLevelField::Conversation,
                _ => TopLevelField::Other,
            },
            key_start,
            value_start,
            value_end,
        });
        cursor = skip_json_whitespace(body, value_end);
        match body.get(cursor) {
            Some(b',') => cursor = skip_json_whitespace(body, cursor + 1),
            Some(b'}') => return Ok(members),
            _ => return Err("expected a comma or closing brace"),
        }
    }
}

/// Skip JSON whitespace from `cursor`.
fn skip_json_whitespace(body: &[u8], mut cursor: usize) -> usize {
    while body.get(cursor).is_some_and(u8::is_ascii_whitespace) {
        cursor += 1;
    }
    cursor
}

/// Return the exclusive end of a JSON string.
fn scan_json_string(body: &[u8], start: usize) -> Result<usize, &'static str> {
    if body.get(start) != Some(&b'"') {
        return Err("expected a JSON string");
    }
    let mut cursor = start + 1;
    let mut escaped = false;
    while let Some(byte) = body.get(cursor).copied() {
        cursor += 1;
        if escaped {
            escaped = false;
        } else if byte == b'\\' {
            escaped = true;
        } else if byte == b'"' {
            return Ok(cursor);
        }
    }
    Err("unterminated JSON string")
}

/// Return the exclusive end of one JSON value.
#[expect(clippy::too_many_lines, reason = "scanner handles nested JSON delimiters explicitly")]
fn scan_json_value(body: &[u8], start: usize) -> Result<usize, &'static str> {
    match body.get(start).copied() {
        Some(b'"') => scan_json_string(body, start),
        Some(b'{' | b'[') => {
            let opening = body.get(start).copied().ok_or("missing JSON value")?;
            let closing = if opening == b'{' { b'}' } else { b']' };
            let mut cursor = start + 1;
            let mut depth = 1_usize;
            while cursor < body.len() {
                match body.get(cursor).copied().ok_or("unterminated JSON value")? {
                    b'"' => cursor = scan_json_string(body, cursor)?,
                    b'{' | b'[' => {
                        depth += 1;
                        cursor += 1;
                    },
                    b'}' | b']' => {
                        depth = depth.checked_sub(1).ok_or("unbalanced JSON value")?;
                        cursor += 1;
                        if depth == 0 {
                            if body.get(cursor - 1).copied() != Some(closing) {
                                return Err("mismatched JSON delimiters");
                            }
                            return Ok(cursor);
                        }
                    },
                    _ => cursor += 1,
                }
            }
            Err("unterminated JSON value")
        },
        Some(_) => {
            let mut cursor = start;
            while let Some(byte) = body.get(cursor) {
                if matches!(byte, b',' | b'}' | b']') {
                    break;
                }
                cursor += 1;
            }
            let end = body
                .get(..cursor)
                .ok_or("invalid JSON value range")?
                .iter()
                .rposition(|byte| !byte.is_ascii_whitespace())
                .map_or(start, |i| i + 1);
            (end > start).then_some(end).ok_or("empty JSON value")
        },
        None => Err("missing JSON value"),
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "request and selected-upstream body phases share one filter implementation"
)]
#[async_trait]
impl HttpFilter for ResponsesProxyFilter {
    fn name(&self) -> &'static str {
        "openai_responses_proxy"
    }

    fn selected_upstream_request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadWrite
    }

    fn request_body_mode(&self) -> BodyMode {
        // The pipeline clamps this per-filter bound to the effective
        // body_limits.max_request_bytes. The selected-upstream phase also
        // checks its rebuilt projection against that same effective ceiling;
        // max_rewritten_body_bytes remains the independent rewriter limit.
        BodyMode::StreamBuffer {
            max_bytes: Some(MAX_JSON_BODY_BYTES),
        }
    }

    fn may_select_streaming_subrequest_response(&self) -> bool {
        // Always advertise the capability: transport follows the effective
        // outbound `stream` field, chosen per-request after upstream selection.
        // There is no operator opt-in. A runtime guard in Praxis still
        // validates the actual streaming terminal action.
        true
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
            trace!("buffering request body chunk");
            return Ok(FilterAction::Continue);
        }
        let Some(state) = ctx.extensions.get::<ResponsesState>() else {
            select_terminal_response_mode(ctx, body);
            return Ok(FilterAction::Continue);
        };

        if !request_needs_rebuild(state) {
            select_terminal_response_mode(ctx, body);
            return Ok(FilterAction::Continue);
        }

        let serialized = match self.serialize_body(state, true)? {
            Ok(bytes) => bytes,
            Err(action) => return Ok(action),
        };
        SerializedJson::from_bytes(serialized).commit(body, self.name(), "body");
        select_terminal_response_mode(ctx, body);
        Ok(FilterAction::Continue)
    }

    async fn on_selected_upstream_request_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
    ) -> Result<SelectedUpstreamBodyOutcome, FilterError> {
        let preserve_native_compaction = selected_backend_uses_native_responses(ctx);
        let Some(state) = ctx.extensions.get::<ResponsesState>() else {
            select_terminal_response_mode(ctx, body);
            if let Some(rejection) = enforce_agentic_stream_guard(ctx) {
                return Ok(SelectedUpstreamBodyOutcome::Reject(rejection));
            }
            debug!("no ResponsesState in extensions, passthrough");
            return Ok(SelectedUpstreamBodyOutcome::Continue);
        };

        if !request_needs_rebuild(state) {
            select_terminal_response_mode(ctx, body);
            if let Some(rejection) = enforce_agentic_stream_guard(ctx) {
                return Ok(SelectedUpstreamBodyOutcome::Reject(rejection));
            }
            debug!("ResponsesState does not require an outbound rewrite, passthrough");
            return Ok(SelectedUpstreamBodyOutcome::Continue);
        }

        // Splice the selected body's state-owned projection into the live body
        // produced by the earlier request-body phase. This keeps later body
        // filter edits to `input` and unrelated fields intact while allowing
        // state changes, such as compaction replay, to reach the backend.
        let Some(current_body) = body.as_ref() else {
            return Ok(SelectedUpstreamBodyOutcome::Continue);
        };
        let serialized = match self.serialize_selected_body(current_body, state, preserve_native_compaction)? {
            Ok(bytes) => bytes,
            Err(FilterAction::Reject(rejection)) => return Ok(SelectedUpstreamBodyOutcome::Reject(rejection)),
            Err(_) => return Err("openai_responses_proxy: invalid selected-upstream body outcome".into()),
        };

        if let Some(limit) = effective_request_body_limit(ctx)
            && serialized.len() > limit
        {
            debug!(
                body_bytes = serialized.len(),
                max_bytes = limit,
                "selected rebuilt request body exceeds effective body limit"
            );
            return Ok(SelectedUpstreamBodyOutcome::Reject(responses_error_rejection(
                413,
                "invalid_request_error",
                &format!(
                    "selected request body ({} bytes) exceeds maximum ({} bytes)",
                    serialized.len(),
                    limit
                ),
            )));
        }

        SerializedJson::from_bytes(serialized).commit(body, self.name(), "body");
        select_terminal_response_mode(ctx, body);
        if let Some(rejection) = enforce_agentic_stream_guard(ctx) {
            return Ok(SelectedUpstreamBodyOutcome::Reject(rejection));
        }

        Ok(SelectedUpstreamBodyOutcome::Continue)
    }
}

/// Return the request ceiling applied by the pipeline to this filter.
///
/// Selected-upstream body participants are validated to use a bounded
/// `StreamBuffer`, so this is the effective `body_limits.max_request_bytes`
/// after the pipeline has clamped the filter's declared mode. The other
/// variants keep this helper safe for direct unit-test invocation and future
/// pipeline changes.
fn effective_request_body_limit(ctx: &HttpFilterContext<'_>) -> Option<usize> {
    match ctx.request_body_mode {
        BodyMode::StreamBuffer { max_bytes } => max_bytes,
        BodyMode::SizeLimit { max_bytes } => Some(max_bytes),
        _ => None,
    }
}

/// Narrow deserialization target for the provider-visible stream bit.
///
/// Only `stream` participates in transport selection; all other request fields
/// are intentionally ignored.
#[derive(Deserialize)]
struct EffectiveResponseMode {
    /// Whether the effective outbound Responses request asks for SSE.
    #[serde(default)]
    stream: bool,
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// Whether the selected backend uses the native Responses wire protocol.
///
/// Compaction preservation follows the selected wire protocol rather than the
/// provider name, since one provider may support multiple application protocols.
fn selected_backend_uses_native_responses(ctx: &HttpFilterContext<'_>) -> bool {
    ctx.selected_application_protocol() == Some("openai_responses")
}

/// Align the typed Praxis response mode with the effective serialized request.
///
/// Classifier metadata describes client intent, but request transformations can
/// change the provider-visible body. The final serializer therefore owns this
/// transport decision and reads the bytes it actually leaves for the upstream.
fn select_terminal_response_mode(ctx: &mut HttpFilterContext<'_>, body: &Option<Bytes>) {
    let mode = if body
        .as_deref()
        .and_then(|bytes| serde_json::from_slice::<EffectiveResponseMode>(bytes).ok())
        .is_some_and(|selection| selection.stream)
    {
        SubRequestResponseMode::Streaming
    } else {
        SubRequestResponseMode::Buffered
    };
    ctx.set_subrequest_response_mode(mode);
}

/// Borrowed view of the outbound request body.
///
/// This keeps the original request and message history borrowed while
/// replacing `input` and omitting locally consumed fields during
/// serialization, avoiding full-body and message clones.
struct OutboundBody<'a> {
    /// Shared request state to project into the provider body.
    state: &'a ResponsesState,
    /// Preserve provider-native compaction items instead of translating them
    /// to Chat-style assistant messages.
    preserve_native_compaction: bool,
    /// IDs of compaction items known to have come from a provider response.
    provider_compaction_ids: &'a HashSet<String>,
}

impl serde::Serialize for OutboundBody<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let Some(object) = self.state.request_body.as_object() else {
            return self.state.request_body.serialize(serializer);
        };

        let start = if provider_owns_conversation(self.state) && self.state.iteration > 0 {
            self.state.provider_history_len
        } else {
            0
        };
        let backend_messages = BackendMessages {
            history: &self.state.messages,
            start,
            preserve_native_compaction: self.preserve_native_compaction,
            provider_compaction_ids: self.provider_compaction_ids,
        };
        let mut map = serializer.serialize_map(None)?;
        let mut wrote_input = false;
        for (name, value) in object {
            match name.as_str() {
                "input" => {
                    map.serialize_entry(name, &backend_messages)?;
                    wrote_input = true;
                },
                "previous_response_id" | "conversation" if self.state.history_rehydrated => {},
                _ => map.serialize_entry(name, value)?,
            }
        }
        if !wrote_input {
            map.serialize_entry("input", &backend_messages)?;
        }
        map.end()
    }
}

/// Whether the upstream provider, rather than local rehydration, owns history.
fn provider_owns_conversation(state: &ResponsesState) -> bool {
    !state.history_rehydrated
        && state
            .conversation
            .as_ref()
            .is_some_and(|conversation| !conversation.is_null())
}

/// Serialize the outbound body without cloning request state.
fn serialize_outbound_body(
    state: &ResponsesState,
    preserve_native_compaction: bool,
) -> Result<Vec<u8>, serde_json::Error> {
    serde_json::to_vec(&OutboundBody {
        state,
        preserve_native_compaction,
        provider_compaction_ids: &state.provider_compaction_ids,
    })
}

/// Borrow history directly, translating only compaction items while serializing.
struct BackendMessages<'a> {
    /// Shared canonical messages.
    history: &'a super::history::MessageHistory,
    /// First message not already owned by the provider.
    start: usize,
    /// Preserve provider-native compaction items instead of translating them
    /// to Chat-style assistant messages.
    preserve_native_compaction: bool,
    /// IDs of compaction items known to have come from a provider response.
    provider_compaction_ids: &'a HashSet<String>,
}

/// Project compaction items into the selected backend's input format.
///
/// Returns `Cow::Borrowed` when no compaction items are present, avoiding
/// allocation. Native mode borrows only provider-originated compaction items;
/// locally generated Praxis summaries are still translated to assistant
/// messages because they are not opaque provider state.
fn messages_for_backend<'a>(
    messages: &'a [serde_json::Value],
    preserve_native_compaction: bool,
    provider_compaction_ids: &HashSet<String>,
) -> Cow<'a, [serde_json::Value]> {
    let mut translated: Option<Vec<serde_json::Value>> = None;

    for (i, m) in messages.iter().enumerate() {
        let is_provider_compaction = preserve_native_compaction
            && m.get("type").and_then(serde_json::Value::as_str) == Some("compaction")
            && m.get(crate::openai::responses::state::LOCAL_COMPACTION_MARKER)
                .and_then(serde_json::Value::as_bool)
                != Some(true)
            && match m.get("id").and_then(serde_json::Value::as_str) {
                Some(id) => provider_compaction_ids.contains(id),
                None => true,
            };
        if m.get("type").and_then(serde_json::Value::as_str) == Some("compaction") && !is_provider_compaction {
            let vec = translated.get_or_insert_with(|| messages.get(..i).unwrap_or(&[]).to_vec());
            vec.push(compaction_to_assistant_message(m));
        } else if let Some(vec) = &mut translated {
            vec.push(m.clone());
        }
    }

    match translated {
        Some(vec) => Cow::Owned(vec),
        None => Cow::Borrowed(messages),
    }
}

impl serde::Serialize for BackendMessages<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeSeq as _;

        let mut sequence = serializer.serialize_seq(Some(self.history.len().saturating_sub(self.start)))?;
        for item in self.history.iter().skip(self.start) {
            let is_provider_compaction = self.preserve_native_compaction
                && item.get("type").and_then(serde_json::Value::as_str) == Some("compaction")
                && item
                    .get(crate::openai::responses::state::LOCAL_COMPACTION_MARKER)
                    .and_then(serde_json::Value::as_bool)
                    != Some(true)
                && match item.get("id").and_then(serde_json::Value::as_str) {
                    Some(id) => self.provider_compaction_ids.contains(id),
                    None => true,
                };
            if item.get("type").and_then(serde_json::Value::as_str) == Some("compaction") && !is_provider_compaction {
                sequence.serialize_element(&compaction_to_assistant_message(item))?;
            } else {
                sequence.serialize_element(item)?;
            }
        }
        sequence.end()
    }
}

/// Translate a compaction item to a Chat Completions assistant message.
fn compaction_to_assistant_message(m: &serde_json::Value) -> serde_json::Value {
    let summary = m
        .get("encrypted_content")
        .and_then(serde_json::Value::as_str)
        .and_then(|e| base64::engine::general_purpose::STANDARD.decode(e).ok())
        .and_then(|b| String::from_utf8(b).ok())
        .unwrap_or_default();
    let prefix = m
        .get("summary_prefix")
        .and_then(serde_json::Value::as_str)
        .unwrap_or(crate::openai::translation::chat_completions::DEFAULT_SUMMARY_PREFIX);
    serde_json::json!({
        "role": "assistant",
        "content": format!("{prefix}{summary}")
    })
}

/// Count the exact serialized bytes for both outbound compaction projections.
///
/// The selected provider is not known when pre-selection rewrite filters
/// enforce their configured cap. Measuring both the native and translated
/// forms prevents the smaller translated assistant placeholder from masking a
/// larger opaque provider item that the selected-upstream hook will preserve.
pub(super) fn serialized_outbound_body_len(state: &ResponsesState) -> Result<usize, serde_json::Error> {
    let translated = serialized_outbound_body_len_for(state, false)?;
    if !state
        .messages
        .iter()
        .any(|message| message.get("type").and_then(serde_json::Value::as_str) == Some("compaction"))
    {
        return Ok(translated);
    }
    let native = serialized_outbound_body_len_for(state, true)?;
    Ok(translated.max(native))
}

/// Count one concrete outbound projection without allocating the body.
fn serialized_outbound_body_len_for(
    state: &ResponsesState,
    preserve_native_compaction: bool,
) -> Result<usize, serde_json::Error> {
    let mut counter = ByteCounter::default();
    serde_json::to_writer(
        &mut counter,
        &OutboundBody {
            state,
            preserve_native_compaction,
            provider_compaction_ids: &state.provider_compaction_ids,
        },
    )?;
    Ok(counter.bytes)
}

/// Writer that counts serialized bytes without allocating a second body.
#[derive(Default)]
struct ByteCounter {
    /// Number of bytes written by the serializer.
    bytes: usize,
}

impl std::io::Write for ByteCounter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.bytes = self.bytes.saturating_add(buf.len());
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Return whether state requires parsing and rebuilding the outbound body.
fn request_needs_rebuild(state: &ResponsesState) -> bool {
    state.request_body_requires_rebuild()
        || state.messages != state.input
        || (state.history_rehydrated
            && (state.previous_response_id.is_some()
                || state.conversation.is_some()
                || state.request_body.get("previous_response_id").is_some()
                || state.request_body.get("conversation").is_some()))
}
