// SPDX-License-Identifier: MIT
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
//! history has changed it. It strips `previous_response_id` only after
//! the rehydrate filter has resolved it locally. Every path removes the
//! Praxis-owned `conversation` field before forwarding upstream.

mod config;

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

use std::borrow::Cow;

use async_trait::async_trait;
use bytes::Bytes;
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, parse_filter_config,
};
use serde::ser::SerializeMap as _;
use tracing::{debug, trace};

use self::config::{ResponsesProxyConfig, build_config};
use super::{error::responses_error_rejection, state::ResponsesState};

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
/// When no `ResponsesState` exists, preserves the request body apart
/// from removing the Praxis-owned `conversation` field.
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
/// max_body_bytes: 67108864
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
        current_body: Option<&Bytes>,
        state: &ResponsesState,
        streaming: bool,
    ) -> Result<Result<Vec<u8>, FilterAction>, FilterError> {
        let current_body = current_body
            .map(|bytes| serde_json::from_slice::<serde_json::Value>(bytes))
            .transpose()
            .map_err(|error| -> FilterError {
                format!("openai_responses_proxy: invalid current body: {error}").into()
            })?;
        let source = current_body.as_ref().unwrap_or(&state.request_body);
        let serialized = serialize_outbound_body(source, state)
            .map_err(|e| -> FilterError { format!("openai_responses_proxy: {e}").into() })?;
        if serialized.len() > self.config.max_body_bytes {
            debug!(
                body_bytes = serialized.len(),
                max_bytes = self.config.max_body_bytes,
                "rebuilt request body exceeds maximum size"
            );
            return Ok(Err(FilterAction::Reject(responses_error_rejection(
                413,
                "invalid_request_error",
                "request body exceeds maximum size",
                streaming,
            ))));
        }

        debug!(
            messages = state.messages.len(),
            body_bytes = serialized.len(),
            "rebuilt request body from ResponsesState"
        );

        Ok(Ok(serialized))
    }
}

#[async_trait]
impl HttpFilter for ResponsesProxyFilter {
    fn name(&self) -> &'static str {
        "openai_responses_proxy"
    }

    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadWrite
    }

    fn request_body_mode(&self) -> BodyMode {
        BodyMode::StreamBuffer {
            max_bytes: Some(self.config.max_body_bytes),
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
            trace!("buffering request body chunk");
            return Ok(FilterAction::Continue);
        }

        let Some(state) = ctx.extensions.get::<ResponsesState>() else {
            strip_conversation_field(ctx, body);
            debug!("no ResponsesState in extensions, passthrough");
            return Ok(FilterAction::Continue);
        };

        if !request_needs_rebuild(state) {
            strip_conversation_field(ctx, body);
            debug!("ResponsesState does not require an outbound rewrite, passthrough");
            return Ok(FilterAction::Continue);
        }

        let streaming = ctx
            .get_metadata("openai_responses_format.stream")
            .is_some_and(|v| v == "true");

        let serialized = match self.serialize_body(body.as_ref(), state, streaming)? {
            Ok(bytes) => bytes,
            Err(action) => return Ok(action),
        };

        let len = serialized.len();
        *body = Some(Bytes::from(serialized));
        ctx.extra_request_headers
            .push((Cow::Borrowed("content-length"), len.to_string()));

        Ok(FilterAction::Continue)
    }
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// Defensively strip `conversation` from a passthrough body so it never
/// leaks to the backend even when no [`ResponsesState`] was produced.
fn strip_conversation_field(ctx: &mut HttpFilterContext<'_>, body: &mut Option<Bytes>) {
    let Some(bytes) = body.as_ref() else {
        return;
    };
    let Ok(mut parsed) = serde_json::from_slice::<serde_json::Value>(bytes) else {
        return;
    };
    if parsed
        .as_object_mut()
        .is_some_and(|obj| obj.remove("conversation").is_some())
    {
        debug!("stripped conversation from passthrough body");
        if let Ok(serialized) = serde_json::to_vec(&parsed) {
            let len = serialized.len();
            *body = Some(Bytes::from(serialized));
            ctx.extra_request_headers
                .push((Cow::Borrowed("content-length"), len.to_string()));
        }
    }
}

/// Borrowed projection of the current request and response state.
///
/// Large request fields and replay messages remain borrowed during
/// serialization. Only the exhausted continuation tool choice may require a
/// small owned value.
struct OutboundBody<'a> {
    /// Current request body, including rewrites from earlier filters.
    source: &'a serde_json::Value,
    /// Shared Responses state.
    state: &'a ResponsesState,
    /// Owned policy used only after exhausting the built-in tool budget.
    exhausted_tool_choice: Option<serde_json::Value>,
    /// Remaining provider-visible built-in tool allowance.
    remaining_tool_calls: Option<u32>,
}

impl<'a> OutboundBody<'a> {
    /// Build a borrowed outbound projection.
    fn new(source: &'a serde_json::Value, state: &'a ResponsesState) -> Self {
        let remaining_tool_calls = (state.iteration != 0)
            .then(|| remaining_max_tool_calls(state))
            .flatten();
        let exhausted_tool_choice =
            (remaining_tool_calls == Some(0)).then(|| exhausted_continuation_tool_choice(state));
        Self {
            source,
            state,
            exhausted_tool_choice,
            remaining_tool_calls,
        }
    }

    /// Resolve the provider-visible tool policy for this request.
    fn tool_choice(&self) -> Option<&serde_json::Value> {
        if self.state.iteration != 0 {
            return self.exhausted_tool_choice.as_ref().or_else(|| {
                self.state
                    .continuation_tool_choice
                    .as_ref()
                    .or(Some(&self.state.tool_choice))
            });
        }
        self.state.tool_choice_present.then_some(&self.state.tool_choice)
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "projects separately owned request fields without cloning"
)]
impl serde::Serialize for OutboundBody<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let Some(object) = self.source.as_object() else {
            return self.source.serialize(serializer);
        };

        let mut map = serializer.serialize_map(None)?;
        let mut wrote_context_management = false;
        let mut wrote_include = false;
        let mut wrote_input = false;
        let mut wrote_previous_response_id = false;
        let mut wrote_tool_choice = false;
        let mut wrote_tools = false;

        for (name, value) in object {
            match name.as_str() {
                "conversation" => {},
                "context_management" => {
                    map.serialize_entry(name, value)?;
                    wrote_context_management = true;
                },
                "include" => {
                    map.serialize_entry(name, value)?;
                    wrote_include = true;
                },
                "input" => {
                    if self.state.messages == self.state.input {
                        map.serialize_entry(name, value)?;
                    } else {
                        map.serialize_entry(name, &self.state.messages)?;
                    }
                    wrote_input = true;
                },
                "max_tool_calls" if self.state.iteration != 0 => {
                    if let Some(remaining @ 1..) = self.remaining_tool_calls {
                        map.serialize_entry(name, &remaining)?;
                    }
                },
                "previous_response_id" if self.state.history_rehydrated => {},
                "previous_response_id" => {
                    map.serialize_entry(name, value)?;
                    wrote_previous_response_id = true;
                },
                "tool_choice" if self.state.iteration != 0 => {
                    if let Some(tool_choice) = self.tool_choice() {
                        map.serialize_entry(name, tool_choice)?;
                        wrote_tool_choice = true;
                    }
                },
                "tool_choice" => {
                    map.serialize_entry(name, value)?;
                    wrote_tool_choice = true;
                },
                "tools" => {
                    map.serialize_entry(name, value)?;
                    wrote_tools = true;
                },
                _ => map.serialize_entry(name, value)?,
            }
        }

        if !wrote_input {
            map.serialize_entry("input", &self.state.messages)?;
        }
        if !wrote_tools && !self.state.tools.is_empty() {
            map.serialize_entry("tools", &self.state.tools)?;
        }
        if !wrote_context_management && let Some(context_management) = &self.state.context_management {
            map.serialize_entry("context_management", context_management)?;
        }
        if !wrote_include && !self.state.include.is_empty() {
            map.serialize_entry("include", &self.state.include)?;
        }
        if !wrote_previous_response_id
            && !self.state.history_rehydrated
            && let Some(previous_response_id) = &self.state.previous_response_id
        {
            map.serialize_entry("previous_response_id", previous_response_id)?;
        }
        if !wrote_tool_choice && let Some(tool_choice) = self.tool_choice() {
            map.serialize_entry("tool_choice", tool_choice)?;
        }
        if self.state.iteration != 0
            && object.get("max_tool_calls").is_none()
            && let Some(remaining @ 1..) = self.remaining_tool_calls
        {
            map.serialize_entry("max_tool_calls", &remaining)?;
        }

        map.end()
    }
}

/// Serialize an outbound body without cloning request state or replay history.
fn serialize_outbound_body(source: &serde_json::Value, state: &ResponsesState) -> Result<Vec<u8>, serde_json::Error> {
    serde_json::to_vec(&OutboundBody::new(source, state))
}

/// Count the exact bytes serialized from state-owned request data.
pub(super) fn serialized_outbound_body_len(state: &ResponsesState) -> Result<usize, serde_json::Error> {
    let mut counter = ByteCounter::default();
    serde_json::to_writer(&mut counter, &OutboundBody::new(&state.request_body, state))?;
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

/// Restrict an exhausted built-in budget to eligible custom tool references.
#[expect(
    clippy::too_many_lines,
    reason = "preserves global, scoped, and forced custom choices"
)]
fn exhausted_continuation_tool_choice(state: &ResponsesState) -> serde_json::Value {
    let current = state.continuation_tool_choice.as_ref().unwrap_or(&state.tool_choice);
    let tools = match current {
        serde_json::Value::String(mode) if mode == "auto" || mode == "required" => {
            state.tools.iter().filter_map(custom_tool_choice_reference).collect()
        },
        serde_json::Value::Object(choice)
            if choice.get("type").and_then(serde_json::Value::as_str) == Some("allowed_tools") =>
        {
            let allowed = choice
                .get("allowed_tools")
                .and_then(serde_json::Value::as_object)
                .unwrap_or(choice);
            allowed
                .get("tools")
                .and_then(serde_json::Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(custom_tool_choice_reference)
                .collect()
        },
        serde_json::Value::Object(_) => custom_tool_choice_reference(current).into_iter().collect(),
        _ => Vec::new(),
    };
    if tools.is_empty() {
        serde_json::Value::String("none".to_owned())
    } else {
        serde_json::json!({
            "type": "allowed_tools",
            "mode": "auto",
            "tools": tools,
        })
    }
}

/// Build the minimal allowed-tools reference for a custom function or MCP tool.
fn custom_tool_choice_reference(tool: &serde_json::Value) -> Option<serde_json::Value> {
    let tool = tool.as_object()?;
    let tool_type = tool.get("type")?.as_str()?;
    let mut reference = serde_json::Map::new();
    reference.insert("type".to_owned(), serde_json::Value::String(tool_type.to_owned()));
    match tool_type {
        "function" | "custom" => {
            let name = tool.get("name").filter(|name| name.is_string())?;
            reference.insert("name".to_owned(), name.clone());
        },
        "mcp" => {
            let server_label = tool.get("server_label").filter(|label| label.is_string())?;
            reference.insert("server_label".to_owned(), server_label.clone());
            if let Some(name) = tool.get("name").filter(|name| name.is_string()) {
                reference.insert("name".to_owned(), name.clone());
            }
        },
        _ => return None,
    }
    Some(serde_json::Value::Object(reference))
}

/// Return whether state requires parsing and rebuilding the outbound body.
fn request_needs_rebuild(state: &ResponsesState) -> bool {
    state.iteration != 0
        || state.messages != state.input
        || (state.history_rehydrated
            && (state.previous_response_id.is_some()
                || state.conversation.is_some()
                || state.request_body.get("previous_response_id").is_some()
                || state.request_body.get("conversation").is_some()))
}

/// Derive the provider-visible allowance for an internal continuation.
fn remaining_max_tool_calls(state: &ResponsesState) -> Option<u32> {
    let max_tool_calls = state.max_tool_calls?;
    let used_calls = state
        .output_items()
        .iter()
        .filter(|item| is_builtin_tool_call(item))
        .count();
    let used_calls = u32::try_from(used_calls).unwrap_or(u32::MAX);
    Some(max_tool_calls.saturating_sub(used_calls))
}

/// Return whether an output item consumes the built-in tool-call allowance.
fn is_builtin_tool_call(item: &serde_json::Value) -> bool {
    matches!(
        item.get("type").and_then(serde_json::Value::as_str),
        Some(
            "apply_patch_call"
                | "code_interpreter_call"
                | "computer_call"
                | "file_search_call"
                | "image_generation_call"
                | "local_shell_call"
                | "multi_agent_call"
                | "shell_call"
                | "tool_search_call"
                | "web_search_call"
        )
    )
}
