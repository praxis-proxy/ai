// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Accumulates state from native Responses API SSE event streams.
//!
//! Parses backend SSE chunks using [`ResponsesSseParser`], dispatches
//! typed events to update [`ResponsesState`] in request extensions.
//! The filter is read-only: it observes body chunks without modifying
//! them.
//!
//! [`ResponsesSseParser`]: crate::openai::sse::responses::ResponsesSseParser
//! [`ResponsesState`]: super::state::ResponsesState

mod accumulator;
mod config;

use async_trait::async_trait;
use bytes::Bytes;
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, parse_filter_config,
};
use tracing::{debug, trace, warn};

use self::{accumulator::accumulate_event, config::StreamEventsConfig};
use crate::openai::sse::{SseParserConfig, responses::ResponsesSseParser};

/// Per-request parser and accumulation state.
pub(super) struct StreamEventsState {
    /// Layered SSE parser that converts raw bytes into typed events.
    parser: ResponsesSseParser,
    /// Accumulated function-call argument deltas, keyed by item id or output index.
    tool_call_args: std::collections::HashMap<String, String>,
}

/// Accumulates state from native Responses API SSE event streams.
///
/// # YAML
///
/// ```yaml
/// filter: openai_stream_events
/// # All fields optional:
/// # max_buffer_bytes: 10485760
/// # max_events: 100000
/// # timeout_secs: 300
/// ```
pub struct OpenaiStreamEventsFilter {
    /// Configuration for the SSE frame parser.
    parser_config: SseParserConfig,
}

impl OpenaiStreamEventsFilter {
    /// Create a filter from parsed YAML config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the YAML config is invalid.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: StreamEventsConfig = parse_filter_config("openai_stream_events", config)?;
        Ok(Box::new(Self {
            parser_config: cfg.to_parser_config(),
        }))
    }

    /// Whether per-request parser state has been installed.
    fn is_armed(ctx: &HttpFilterContext<'_>) -> bool {
        ctx.get_filter_state::<StreamEventsState>().is_some()
    }
}

#[async_trait]
impl HttpFilter for OpenaiStreamEventsFilter {
    fn name(&self) -> &'static str {
        "openai_stream_events"
    }

    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::None
    }

    fn request_body_mode(&self) -> BodyMode {
        BodyMode::Stream
    }

    fn response_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    fn response_body_mode(&self) -> BodyMode {
        BodyMode::Stream
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        let is_responses = ctx.get_metadata("openai_responses_format.format") == Some("openai_responses");
        let is_streaming = ctx.get_metadata("openai_responses_format.stream") == Some("true");

        if is_responses && is_streaming {
            trace!("arming stream_events for streaming Responses API request");
            ctx.insert_filter_state(StreamEventsState {
                parser: ResponsesSseParser::new(&self.parser_config),
                tool_call_args: std::collections::HashMap::new(),
            });
        }

        Ok(FilterAction::Continue)
    }

    fn on_response_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if !Self::is_armed(ctx) {
            debug!("stream_events not armed, passing through");
            return Ok(FilterAction::Continue);
        }

        if let Some(bytes) = body.as_ref() {
            process_chunk(ctx, bytes);
        }

        if end_of_stream {
            validate_stream_end(ctx);
        }

        Ok(FilterAction::Continue)
    }
}

/// Parse an SSE chunk and dispatch events to the accumulator.
fn process_chunk(ctx: &mut HttpFilterContext<'_>, bytes: &Bytes) {
    let Some(mut state) = ctx.remove_filter_state::<StreamEventsState>() else {
        return;
    };
    match state.parser.parse_chunk(bytes) {
        Ok(events) => {
            for event in &events {
                accumulate_event(ctx, &mut state, event);
            }
        },
        Err(e) => {
            warn!(error = %e, "SSE parse error in stream_events");
            ctx.set_metadata("responses.stream_parse_error", "true".to_owned());
        },
    }
    ctx.insert_filter_state(state);
}

/// Check that the SSE stream terminated with a terminal event.
fn validate_stream_end(ctx: &mut HttpFilterContext<'_>) {
    if let Some(state) = ctx.get_filter_state::<StreamEventsState>()
        && let Err(e) = state.parser.validate_complete()
    {
        warn!(error = %e, "stream did not terminate cleanly");
        ctx.set_metadata("responses.stream_incomplete", "true".to_owned());
    }
    debug!("stream_events processing complete");
}

#[cfg(test)]
mod tests;
