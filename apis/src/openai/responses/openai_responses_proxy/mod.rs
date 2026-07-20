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
        state: &ResponsesState,
        streaming: bool,
    ) -> Result<Result<Vec<u8>, FilterAction>, FilterError> {
        let serialized = serialize_outbound_body(state)
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

        let serialized = match self.serialize_body(state, streaming)? {
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

/// Borrowed view of the outbound request body.
///
/// This keeps the original request and message history borrowed while
/// replacing `input` and omitting locally consumed fields during
/// serialization, avoiding full-body and message clones.
struct OutboundBody<'a> {
    /// Shared request state to project into the provider body.
    state: &'a ResponsesState,
}

impl serde::Serialize for OutboundBody<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let Some(object) = self.state.request_body.as_object() else {
            return self.state.request_body.serialize(serializer);
        };

        let mut map = serializer.serialize_map(None)?;
        let mut wrote_input = false;
        for (name, value) in object {
            match name.as_str() {
                "input" => {
                    map.serialize_entry(name, &self.state.messages)?;
                    wrote_input = true;
                },
                "previous_response_id" | "conversation" => {},
                _ => map.serialize_entry(name, value)?,
            }
        }
        if !wrote_input {
            map.serialize_entry("input", &self.state.messages)?;
        }
        map.end()
    }
}

/// Serialize the outbound body without cloning request state.
fn serialize_outbound_body(state: &ResponsesState) -> Result<Vec<u8>, serde_json::Error> {
    serde_json::to_vec(&OutboundBody { state })
}

/// Count the exact bytes the proxy will serialize for an outbound body.
pub(super) fn serialized_outbound_body_len(state: &ResponsesState) -> Result<usize, serde_json::Error> {
    let mut counter = ByteCounter::default();
    serde_json::to_writer(&mut counter, &OutboundBody { state })?;
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
    state.messages != state.input
        || (state.history_rehydrated
            && (state.previous_response_id.is_some()
                || state.conversation.is_some()
                || state.request_body.get("previous_response_id").is_some()
                || state.request_body.get("conversation").is_some()))
}
