// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Incremental Anthropic Messages SSE to OpenAI Chat Completions SSE conversion.

mod config;

use std::collections::BTreeMap;

use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, Rejection, parse_filter_config,
    sse::{SseDecoder, SseLimits, SseRecord, encode},
};
use serde_json::{Value, json};
use tracing::warn;

use self::config::OpenaiChatCompletionsToAnthropicStreamConfig;
use super::openai_chat_completions_to_anthropic::{
    BOUND_PREFIX, CLIENT_MODEL, INCLUDE_USAGE, REQUEST_STREAM, map_stop_reason, total_input_tokens,
};

/// Registered filter identifier.
const FILTER_NAME: &str = "openai_chat_completions_to_anthropic_stream";
/// Marker indicating that this request's Anthropic SSE response is armed.
const ARMED: &str = "openai_chat_completions_to_anthropic_stream.armed";

/// Per-request state. The decoder retains only an unfinished SSE record; this
/// state stores protocol metadata and tool indices, never accumulated output.
#[expect(
    clippy::struct_excessive_bools,
    reason = "These flags represent independent SSE protocol milestones"
)]
struct StreamState {
    /// Incremental decoder retaining only a partial event between chunks.
    decoder: SseDecoder,
    /// Original client model used in downstream chunks.
    client_model: String,
    /// Anthropic message ID from `message_start`.
    message_id: Option<String>,
    /// Whether `message_start` has been observed.
    started: bool,
    /// Whether `message_delta` has been observed.
    saw_message_delta: bool,
    /// Whether conversion emitted a terminal event.
    finished: bool,
    /// Whether the client requested usage in the terminal chunk.
    include_usage: bool,
    /// Prompt token count from Anthropic `message_start`.
    input_tokens: u64,
    /// Mapping from Anthropic content-block index to OpenAI tool-call index.
    tool_indices: BTreeMap<u64, u64>,
    /// Next OpenAI tool-call index.
    next_tool_index: u64,
}

impl StreamState {
    /// Create request-local stream conversion state with a bounded event decoder.
    fn new(config: &OpenaiChatCompletionsToAnthropicStreamConfig, client_model: String, include_usage: bool) -> Self {
        let limits = SseLimits {
            max_line_bytes: config.max_partial_event_bytes,
            max_record_bytes: config.max_partial_event_bytes,
            ..SseLimits::default()
        };
        Self {
            decoder: SseDecoder::with_limits(limits),
            client_model,
            message_id: None,
            started: false,
            saw_message_delta: false,
            finished: false,
            include_usage,
            input_tokens: 0,
            tool_indices: BTreeMap::new(),
            next_tool_index: 0,
        }
    }
}

/// Converts the selected Anthropic provider's event stream incrementally.
/// OpenAI-compatible candidates are left byte-for-byte unchanged.
///
/// Place this after `intelligent_route` and before the request converter. The
/// filter binds to the immutable provider profile saved by the request
/// converter and retains only a partial SSE event plus bounded tool indexes.
///
/// ```yaml
/// filter: openai_chat_completions_to_anthropic_stream
/// max_partial_event_bytes: 1048576
/// max_tool_blocks: 256
/// ```
pub struct OpenaiChatCompletionsToAnthropicStreamFilter {
    /// SSE record and tool-count limits.
    config: OpenaiChatCompletionsToAnthropicStreamConfig,
}

impl OpenaiChatCompletionsToAnthropicStreamFilter {
    /// Create the stream filter from `ExtProc` YAML.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown configuration keys or invalid limits.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let config: OpenaiChatCompletionsToAnthropicStreamConfig = parse_filter_config(FILTER_NAME, config)?;
        Ok(Box::new(Self {
            config: config.validate()?,
        }))
    }

    /// Emit a terminal OpenAI SSE error and `[DONE]` record.
    fn append_error(output: &mut BytesMut, message: &str) -> Result<(), FilterError> {
        append_data(output, &json!({"error":{"message":message,"type":"server_error"}}))?;
        append_done(output)?;
        Ok(())
    }

    #[expect(
        clippy::too_many_lines,
        reason = "This loop is the exhaustive Anthropic SSE event dispatcher"
    )]
    /// Convert a decoded batch of Anthropic events in order.
    fn process_records(
        &self,
        state: &mut StreamState,
        records: Vec<SseRecord>,
        output: &mut BytesMut,
    ) -> Result<(), String> {
        for record in records {
            if state.finished {
                continue;
            }
            if !record.is_event() {
                continue;
            }
            let event = record
                .event()
                .and_then(|value| std::str::from_utf8(value).ok())
                .unwrap_or_default();
            let data = record.data();
            let data = std::str::from_utf8(&data)
                .map_err(|error| format!("Anthropic SSE event data is not UTF-8: {error}"))?;
            if event == "ping" {
                continue;
            }
            if event == "error" {
                let value: Value =
                    serde_json::from_str(data).map_err(|error| format!("invalid Anthropic error event: {error}"))?;
                let message = value
                    .pointer("/error/message")
                    .and_then(Value::as_str)
                    .unwrap_or("Anthropic provider stream failed");
                Self::append_error(output, message).map_err(|error| error.to_string())?;
                state.finished = true;
                continue;
            }
            let value: Value =
                serde_json::from_str(data).map_err(|error| format!("invalid Anthropic {event} event JSON: {error}"))?;
            match event {
                "message_start" => Self::message_start(state, &value, output)?,
                "content_block_start" => self.content_block_start(state, &value, output)?,
                "content_block_delta" => Self::content_block_delta(state, &value, output)?,
                "content_block_stop" | "ping" => {},
                "message_delta" => Self::message_delta(state, &value, output)?,
                "message_stop" => {
                    if !state.started {
                        return Err("Anthropic message_stop arrived before message_start".to_owned());
                    }
                    if !state.saw_message_delta {
                        return Err("Anthropic message_stop arrived before message_delta".to_owned());
                    }
                    append_done(output).map_err(|error| error.to_string())?;
                    state.finished = true;
                },
                other => return Err(format!("unsupported Anthropic SSE event {other:?}")),
            }
        }
        Ok(())
    }

    /// Convert Anthropic's message header to the first OpenAI chunk.
    fn message_start(state: &mut StreamState, value: &Value, output: &mut BytesMut) -> Result<(), String> {
        if state.started {
            return Err("Anthropic stream contains more than one message_start".to_owned());
        }
        let message = value
            .get("message")
            .and_then(Value::as_object)
            .ok_or("Anthropic message_start requires message")?;
        let id = message
            .get("id")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .ok_or("Anthropic message_start requires message.id")?;
        state.message_id = Some(format!("chatcmpl-{id}"));
        let usage = message
            .get("usage")
            .and_then(Value::as_object)
            .ok_or("Anthropic message_start requires usage")?;
        state.input_tokens = total_input_tokens(usage)?;
        append_chunk(output, state, &json!({"role":"assistant"}), &Value::Null).map_err(|error| error.to_string())?;
        state.started = true;
        Ok(())
    }

    #[expect(
        clippy::too_many_lines,
        reason = "The event validates and registers text or tool block starts"
    )]
    /// Register one Anthropic text or tool content block.
    fn content_block_start(&self, state: &mut StreamState, value: &Value, output: &mut BytesMut) -> Result<(), String> {
        let block_index = value
            .get("index")
            .and_then(Value::as_u64)
            .ok_or("Anthropic content_block_start requires index")?;
        let block = value
            .get("content_block")
            .and_then(Value::as_object)
            .ok_or("Anthropic content_block_start requires content_block")?;
        match block.get("type").and_then(Value::as_str) {
            Some("text") => Ok(()),
            Some("tool_use") => {
                if state.tool_indices.len() >= self.config.max_tool_blocks {
                    return Err("Anthropic tool block limit exceeded".to_owned());
                }
                let id = block
                    .get("id")
                    .and_then(Value::as_str)
                    .filter(|value| !value.is_empty())
                    .ok_or("Anthropic tool_use requires id")?;
                let name = block
                    .get("name")
                    .and_then(Value::as_str)
                    .filter(|value| !value.is_empty())
                    .ok_or("Anthropic tool_use requires name")?;
                let tool_index = state.next_tool_index;
                state.next_tool_index = state.next_tool_index.checked_add(1).ok_or("tool index overflow")?;
                if state.tool_indices.insert(block_index, tool_index).is_some() {
                    return Err("Anthropic stream reused a content block index".to_owned());
                }
                let delta = json!({"tool_calls":[{"index":tool_index,"id":id,"type":"function","function":{"name":name,"arguments":""}}]});
                append_chunk(output, state, &delta, &Value::Null).map_err(|error| error.to_string())
            },
            Some(kind) => Err(format!("unsupported Anthropic content block {kind:?}")),
            None => Err("Anthropic content_block_start requires content_block.type".to_owned()),
        }
    }

    #[expect(clippy::too_many_lines, reason = "The event maps text and partial tool JSON deltas")]
    /// Convert one Anthropic content delta without retaining streamed output.
    fn content_block_delta(state: &StreamState, value: &Value, output: &mut BytesMut) -> Result<(), String> {
        let index = value
            .get("index")
            .and_then(Value::as_u64)
            .ok_or("Anthropic content_block_delta requires index")?;
        let delta = value
            .get("delta")
            .and_then(Value::as_object)
            .ok_or("Anthropic content_block_delta requires delta")?;
        match delta.get("type").and_then(Value::as_str) {
            Some("text_delta") => {
                let text = delta
                    .get("text")
                    .and_then(Value::as_str)
                    .ok_or("Anthropic text_delta requires text")?;
                append_chunk(output, state, &json!({"content":text}), &Value::Null).map_err(|error| error.to_string())
            },
            Some("input_json_delta") => {
                let partial = delta
                    .get("partial_json")
                    .and_then(Value::as_str)
                    .ok_or("Anthropic input_json_delta requires partial_json")?;
                let tool_index = state
                    .tool_indices
                    .get(&index)
                    .copied()
                    .ok_or("Anthropic input_json_delta has no matching tool_use block")?;
                append_chunk(
                    output,
                    state,
                    &json!({"tool_calls":[{"index":tool_index,"function":{"arguments":partial}}]}),
                    &Value::Null,
                )
                .map_err(|error| error.to_string())
            },
            Some(kind) => Err(format!("unsupported Anthropic content delta {kind:?}")),
            None => Err("Anthropic content_block_delta requires delta.type".to_owned()),
        }
    }

    #[expect(
        clippy::too_many_lines,
        reason = "The terminal event validates stop and usage before emitting completion"
    )]
    /// Convert Anthropic stop and token-usage data into OpenAI terminal chunks.
    fn message_delta(state: &mut StreamState, value: &Value, output: &mut BytesMut) -> Result<(), String> {
        if !state.started {
            return Err("Anthropic message_delta arrived before message_start".to_owned());
        }
        if state.saw_message_delta {
            return Err("Anthropic stream contains more than one message_delta".to_owned());
        }
        let delta = value
            .get("delta")
            .and_then(Value::as_object)
            .ok_or("Anthropic message_delta requires delta")?;
        let reason = delta
            .get("stop_reason")
            .and_then(Value::as_str)
            .ok_or("Anthropic message_delta requires stop_reason")?;
        let finish_reason = map_stop_reason(reason)?;
        let usage = value
            .get("usage")
            .and_then(Value::as_object)
            .ok_or("Anthropic message_delta requires usage")?;
        let completion_tokens = usage
            .get("output_tokens")
            .and_then(Value::as_u64)
            .ok_or("Anthropic message_delta requires output_tokens")?;
        append_chunk(output, state, &json!({}), &json!(finish_reason)).map_err(|error| error.to_string())?;
        if state.include_usage {
            let total = state
                .input_tokens
                .checked_add(completion_tokens)
                .ok_or("usage token count overflow")?;
            let chunk = json!({
                "id":state.message_id.as_deref().unwrap_or("chatcmpl-unknown"),
                "object":"chat.completion.chunk",
                "created":0,
                "model":state.client_model,
                "choices":[],
                "usage":{"prompt_tokens":state.input_tokens,"completion_tokens":completion_tokens,"total_tokens":total}
            });
            append_data(output, &chunk).map_err(|error| error.to_string())?;
        }
        state.saw_message_delta = true;
        Ok(())
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "The response callbacks keep incremental stream state local to one request"
)]
#[async_trait]
impl HttpFilter for OpenaiChatCompletionsToAnthropicStreamFilter {
    /// Return the registry name used by the `ExtProc` filter chain.
    fn name(&self) -> &'static str {
        "openai_chat_completions_to_anthropic_stream"
    }

    /// Leave request bodies to the JSON translator or selected OpenAI pass-through path.
    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::None
    }

    /// Do not buffer the request in this response-only filter.
    fn request_body_mode(&self) -> BodyMode {
        BodyMode::Stream
    }

    /// Allow incremental conversion of the selected Anthropic response stream.
    fn response_body_access(&self) -> BodyAccess {
        BodyAccess::ReadWrite
    }

    /// Keep upstream SSE progressive; only the unfinished event is retained.
    fn response_body_mode(&self) -> BodyMode {
        BodyMode::Stream
    }

    /// This filter has no request-side action; selection happens in the JSON filter.
    async fn on_request(&self, _ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        Ok(FilterAction::Continue)
    }

    /// Arm conversion only when the bound provider is Anthropic and the client requested streaming.
    async fn on_response(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        if ctx.get_metadata(&format!("{BOUND_PREFIX}api_format")) != Some("messages")
            || ctx.get_metadata(REQUEST_STREAM) != Some("true")
        {
            return Ok(FilterAction::Continue);
        }
        let Some(response) = ctx.response_header.as_ref() else {
            return Err("openai_chat_completions_to_anthropic_stream: response headers are missing".into());
        };
        if !response.status.is_success()
            || !response
                .headers
                .get(http::header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .is_some_and(crate::is_event_stream_content_type)
        {
            return Ok(FilterAction::Continue);
        }
        if response.headers.contains_key(http::header::CONTENT_ENCODING) {
            return Ok(FilterAction::Reject(
                Rejection::status(502)
                    .with_header("content-type", "application/json")
                    .with_body(
                    br#"{"error":{"message":"compressed Anthropic SSE cannot be translated","type":"server_error"}}"#
                        .as_slice(),
                ),
            ));
        }
        let model = ctx
            .get_metadata(CLIENT_MODEL)
            .filter(|value| !value.is_empty())
            .ok_or("OpenAI client model is missing from the selected request profile")?;
        ctx.insert_filter_state(StreamState::new(
            &self.config,
            model.to_owned(),
            ctx.get_metadata(INCLUDE_USAGE) == Some("true"),
        ));
        ctx.set_metadata(ARMED, "true");
        if let Some(response) = &mut ctx.response_header {
            response.headers.remove(http::header::CONTENT_LENGTH);
            response.headers.insert(
                http::header::CONTENT_TYPE,
                http::HeaderValue::from_static("text/event-stream"),
            );
            ctx.response_headers_modified = true;
        }
        Ok(FilterAction::Continue)
    }

    /// Decode complete SSE events from chunks and emit equivalent OpenAI chunks immediately.
    #[expect(
        clippy::too_many_lines,
        reason = "This callback bounds and transforms each SSE body chunk in place"
    )]
    fn on_response_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if ctx.get_metadata(ARMED) != Some("true") {
            return Ok(FilterAction::Continue);
        }
        let state = ctx
            .get_filter_state_mut::<StreamState>()
            .ok_or("OpenAI SSE response state is missing")?;
        if state.finished {
            *body = Some(Bytes::new());
            return Ok(FilterAction::Continue);
        }
        let mut output = BytesMut::new();
        let batch = body
            .as_ref()
            .map_or_else(Default::default, |bytes| state.decoder.push(bytes));
        if let Some(error) = batch.error {
            warn!(error = %error, "Anthropic SSE decoder limit or protocol error");
            OpenaiChatCompletionsToAnthropicStreamFilter::append_error(
                &mut output,
                "Anthropic provider event exceeded the configured streaming limit",
            )?;
            state.finished = true;
        } else if let Err(error) = self.process_records(state, batch.records, &mut output) {
            warn!(error = error.as_str(), "Anthropic SSE event could not be translated");
            OpenaiChatCompletionsToAnthropicStreamFilter::append_error(&mut output, &error)?;
            state.finished = true;
        }
        if end_of_stream && !state.finished {
            let tail = state.decoder.finish();
            if tail.error.is_some() || tail.trailing.is_some() || !state.started {
                OpenaiChatCompletionsToAnthropicStreamFilter::append_error(
                    &mut output,
                    "Anthropic provider stream ended before a complete message",
                )?;
            } else {
                OpenaiChatCompletionsToAnthropicStreamFilter::append_error(
                    &mut output,
                    "Anthropic provider stream omitted message_stop",
                )?;
            }
            state.finished = true;
        }
        *body = Some(output.freeze());
        Ok(FilterAction::Continue)
    }
}

/// Serialize one OpenAI-compatible SSE chunk into the current response bytes.
fn append_chunk(
    output: &mut BytesMut,
    state: &StreamState,
    delta: &Value,
    finish_reason: &Value,
) -> Result<(), FilterError> {
    let id = state
        .message_id
        .as_deref()
        .ok_or("Anthropic stream has no message ID")?;
    append_data(
        output,
        &json!({
            "id":id,
            "object":"chat.completion.chunk",
            "created":0,
            "model":state.client_model,
            "choices":[{"index":0,"delta":delta,"finish_reason":finish_reason}]
        }),
    )
}

/// Encode a JSON SSE data record.
fn append_data(output: &mut BytesMut, value: &Value) -> Result<(), FilterError> {
    let data = serde_json::to_vec(value).map_err(|error| FilterError::from(error.to_string()))?;
    let record = SseRecord::builder()
        .data(Bytes::from(data))
        .build()
        .map_err(|error| FilterError::from(error.to_string()))?;
    output.extend_from_slice(&encode(&record));
    Ok(())
}

/// Encode the OpenAI stream termination record.
fn append_done(output: &mut BytesMut) -> Result<(), FilterError> {
    let record = SseRecord::builder()
        .data("[DONE]")
        .build()
        .map_err(|error| FilterError::from(error.to_string()))?;
    output.extend_from_slice(&encode(&record));
    Ok(())
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::too_many_lines,
    reason = "test assertions and streamed response exchanges"
)]
mod tests;
