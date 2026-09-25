// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Anthropic Messages streaming SSE transformation filter.
//!
//! Transforms Chat Completions SSE events into Anthropic Messages SSE
//! events per-chunk while buffering partial events. Any Chat
//! Completions-compatible backend is a valid source, not only OpenAI.

mod config;

use std::borrow::Cow;

use async_trait::async_trait;
use bytes::Bytes;
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, parse_filter_config,
};
use serde_json::Value;
use tracing::debug;

use self::config::{AnthropicMessagesToChatCompletionsStreamConfig, build_config};
use crate::{
    anthropic::{
        messages_to_chat_completions::client_stop_sequences,
        wire::{ContentBlock, MessageDeltaUsage, MessageUsage},
    },
    is_event_stream_content_type,
};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Metadata key for the partial line buffer between chunks.
const LINE_BUFFER_KEY: &str = "anthropic_stream.line_buffer";

/// Metadata key for the internal streaming state.
const STREAM_STATE_KEY: &str = "anthropic_stream.state";

/// Internal stream state value recorded after emitting `message_start`.
const STREAM_STATE_STARTED: &str = "started";

/// Internal stream state value recorded after emitting a terminal error event.
const STREAM_STATE_FAILED: &str = "failed";

/// OpenAI Chat Completions SSE sentinel that marks logical stream completion.
const OPENAI_DONE_SENTINEL: &str = "[DONE]";

/// Metadata key tracking whether a text content block is open.
const TEXT_BLOCK_OPEN_KEY: &str = "anthropic_stream.text_block_open";

/// Metadata key prefix for per-tool-call content block state.
const TOOL_BLOCK_KEY_PREFIX: &str = "anthropic_stream.tool_block.";

/// Metadata key suffix for a tool call's Anthropic content block index.
const TOOL_BLOCK_INDEX_SUFFIX: &str = ".index";

/// Metadata key suffix tracking whether a tool call's content block is open.
const TOOL_BLOCK_OPEN_SUFFIX: &str = ".open";

/// Metadata key counting distinct tool-call content blocks opened so far.
///
/// Each opened block pins per-block state (index and open/closed flag)
/// for the response's lifetime; this count bounds that growth against
/// `max_tool_blocks`. The trailing token is `tool_block_count`, not
/// `tool_block.<key>`, so it never matches [`TOOL_BLOCK_KEY_PREFIX`].
const TOOL_BLOCK_COUNT_KEY: &str = "anthropic_stream.tool_block_count";

/// Metadata key for the finish reason from the upstream provider.
const FINISH_REASON_KEY: &str = "anthropic_stream.finish_reason";

/// Metadata key for the client stop sequence the upstream reported as matched.
const STOP_SEQUENCE_KEY: &str = "anthropic_stream.stop_sequence";

/// Metadata key for accumulated output token count.
const OUTPUT_TOKENS_KEY: &str = "anthropic_stream.output_tokens";

/// Metadata key for accumulated input (prompt) token count.
const INPUT_TOKENS_KEY: &str = "anthropic_stream.input_tokens";

/// Metadata key for cached input token count.
const CACHE_READ_TOKENS_KEY: &str = "anthropic_stream.cache_read_tokens";

/// Metadata key for the current content block index.
const BLOCK_INDEX_KEY: &str = "anthropic_stream.block_index";

/// Metadata key for incomplete UTF-8 bytes (hex-encoded) between chunks.
const UTF8_BUFFER_KEY: &str = "anthropic_stream.utf8_buffer";

/// Metadata key indicating the filter is armed for streaming transformation.
const ARMED_KEY: &str = "anthropic_stream.armed";

// -----------------------------------------------------------------------------
// AnthropicMessagesToChatCompletionsStreamFilter
// -----------------------------------------------------------------------------

/// Transforms streaming SSE responses between the Chat Completions and
/// Anthropic Messages formats, processing each chunk as it arrives.
///
/// Arms automatically when an upstream classifier or transform
/// filter sets `anthropic_messages_format.stream` or
/// `anthropic_messages_to_chat_completions.streaming` metadata to `"true"` and
/// the backend response has `Content-Type: text/event-stream`
/// (with or without parameters such as `charset=utf-8`) and does
/// not carry a `Content-Encoding` header.
/// No `response_conditions` configuration is needed.
///
/// # YAML
///
/// ```yaml
/// filter: anthropic_messages_to_chat_completions_stream
/// ```
///
/// # Full YAML
///
/// ```yaml
/// filter: anthropic_messages_to_chat_completions_stream
/// max_partial_event_bytes: 10485760
/// max_tool_blocks: 10000
/// ```
pub struct AnthropicMessagesToChatCompletionsStreamFilter {
    /// Parsed and validated configuration.
    config: AnthropicMessagesToChatCompletionsStreamConfig,
}

impl AnthropicMessagesToChatCompletionsStreamFilter {
    /// Create a filter from parsed YAML config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the YAML config is invalid.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: AnthropicMessagesToChatCompletionsStreamConfig =
            parse_filter_config("anthropic_messages_to_chat_completions_stream", config)?;
        let validated = build_config(cfg)?;
        Ok(Box::new(Self { config: validated }))
    }

    /// Decode and transform one response body chunk under the filter's
    /// configured partial-event and tool-block limits.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if a partial SSE event or the retained
    /// tool-call block count exceeds its configured limit, or if a complete
    /// data event is neither the `[DONE]` sentinel nor a valid Chat
    /// Completions JSON object.
    fn process_response_chunk(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        bytes: &Bytes,
        end_of_stream: bool,
    ) -> Result<Option<Bytes>, FilterError> {
        decode_and_process_chunk(
            ctx,
            bytes,
            end_of_stream,
            self.config.max_partial_event_bytes,
            self.config.max_tool_blocks,
        )
    }
}

#[async_trait]
impl HttpFilter for AnthropicMessagesToChatCompletionsStreamFilter {
    fn name(&self) -> &'static str {
        "anthropic_messages_to_chat_completions_stream"
    }

    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::None
    }

    fn request_body_mode(&self) -> BodyMode {
        BodyMode::Stream
    }

    fn response_body_access(&self) -> BodyAccess {
        BodyAccess::ReadWrite
    }

    fn response_body_mode(&self) -> BodyMode {
        BodyMode::Stream
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        ctx.request_headers_to_remove.push(http::header::ACCEPT_ENCODING);
        Ok(FilterAction::Continue)
    }

    async fn on_response(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        if !should_arm(ctx) {
            return Ok(FilterAction::Continue);
        }

        ctx.set_metadata(ARMED_KEY, "true".to_owned());

        if let Some(resp) = &mut ctx.response_header {
            resp.headers.remove(http::header::CONTENT_LENGTH);
            resp.headers.insert(
                http::header::CONTENT_TYPE,
                http::HeaderValue::from_static("text/event-stream"),
            );
            ctx.response_headers_modified = true;
        }
        Ok(FilterAction::Continue)
    }

    fn on_response_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if !is_armed(ctx) {
            return Ok(FilterAction::Continue);
        }

        if is_stream_failed(ctx) {
            if body.is_some() {
                *body = Some(Bytes::new());
            }
            return Ok(FilterAction::Continue);
        }

        let Some(bytes) = body.as_ref() else {
            if end_of_stream {
                let output = self
                    .process_response_chunk(ctx, &Bytes::new(), true)?
                    .unwrap_or_default();
                if !output.is_empty() {
                    *body = Some(output);
                }
            }
            return Ok(FilterAction::Continue);
        };

        let Some(output) = self.process_response_chunk(ctx, bytes, end_of_stream)? else {
            *body = Some(Bytes::new());
            return Ok(FilterAction::Continue);
        };

        *body = Some(output);
        Ok(FilterAction::Continue)
    }
}

// -----------------------------------------------------------------------------
// SSE Chunk Processing
// -----------------------------------------------------------------------------

/// Reassemble incomplete UTF-8 bytes from the previous chunk, extract
/// the valid prefix, buffer any trailing incomplete sequence, and
/// run SSE processing on the valid portion.
fn decode_and_process_chunk(
    ctx: &mut HttpFilterContext<'_>,
    bytes: &Bytes,
    end_of_stream: bool,
    max_partial_event_bytes: usize,
    max_tool_blocks: usize,
) -> Result<Option<Bytes>, FilterError> {
    let combined = combine_pending_utf8(ctx, bytes);
    let Some(valid_up_to) = valid_utf8_prefix_len(ctx, combined.as_slice(), end_of_stream) else {
        // Never mix raw upstream bytes into an Anthropic event stream.
        ctx.filter_metadata.remove(LINE_BUFFER_KEY);
        return Err(FilterError::from(
            "anthropic_messages_to_chat_completions_stream: upstream SSE contains malformed UTF-8",
        ));
    };
    let Some(valid_bytes) = combined.as_slice().get(..valid_up_to) else {
        return Ok(None);
    };
    let Some(chunk_str) = std::str::from_utf8(valid_bytes).ok() else {
        return Ok(None);
    };

    if chunk_str.is_empty() && !end_of_stream {
        return Ok(None);
    }

    process_sse_chunk(ctx, chunk_str, end_of_stream, max_partial_event_bytes, max_tool_blocks).map(Some)
}

/// Prefix any incomplete UTF-8 bytes retained from the previous chunk.
fn combine_pending_utf8<'a>(ctx: &mut HttpFilterContext<'_>, bytes: &'a Bytes) -> CombinedUtf8Chunk<'a> {
    match ctx.filter_metadata.remove(UTF8_BUFFER_KEY) {
        Some(hex) => {
            let mut pending = decode_hex_bytes(&hex);
            pending.extend_from_slice(bytes);
            CombinedUtf8Chunk::Owned(pending)
        },
        None => CombinedUtf8Chunk::Borrowed(bytes),
    }
}

/// Find the valid UTF-8 prefix and retain only a trailing incomplete suffix.
fn valid_utf8_prefix_len(ctx: &mut HttpFilterContext<'_>, combined: &[u8], end_of_stream: bool) -> Option<usize> {
    match std::str::from_utf8(combined) {
        Ok(_) => Some(combined.len()),
        Err(e) if e.error_len().is_none() => {
            if end_of_stream {
                return None;
            }
            let valid = e.valid_up_to();
            let tail = combined.get(valid..)?;
            if tail.len() > 3 {
                return None;
            }
            ctx.filter_metadata
                .insert(UTF8_BUFFER_KEY.to_owned(), encode_hex_bytes(tail));
            Some(valid)
        },
        Err(_) => None,
    }
}

/// Combined UTF-8 chunk data, borrowed unless a pending suffix had to be prefixed.
enum CombinedUtf8Chunk<'a> {
    /// Current chunk borrowed directly from Pingora.
    Borrowed(&'a Bytes),

    /// Current chunk prefixed with bytes buffered from the previous chunk.
    Owned(Vec<u8>),
}

impl CombinedUtf8Chunk<'_> {
    /// View the combined bytes without forcing an allocation.
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Borrowed(bytes) => bytes.as_ref(),
            Self::Owned(bytes) => bytes.as_slice(),
        }
    }
}

/// Parse SSE event boundaries from the combined buffer, transform
/// each complete event, and store any leftover partial data.
///
/// Handles `\r\n`, `\r`, and `\n` line endings per the SSE
/// specification. A single trailing `\r` is held back before
/// end-of-stream because it might be the first half of a `\r\n`
/// pair split across chunks.
fn process_sse_chunk(
    ctx: &mut HttpFilterContext<'_>,
    chunk_str: &str,
    end_of_stream: bool,
    max_partial_event_bytes: usize,
    max_tool_blocks: usize,
) -> Result<Bytes, FilterError> {
    let leftover = ctx.filter_metadata.get(LINE_BUFFER_KEY).map(String::as_str);
    let combined = combine_chunk_with_leftover(leftover, chunk_str);
    let (to_normalize, pending_cr) = split_deferred_trailing_cr(combined.as_ref(), end_of_stream);
    let normalized = normalize_line_endings(to_normalize);
    let mut output = Vec::new();
    let remaining = process_complete_event_blocks(ctx, &normalized, &mut output, max_tool_blocks)?;

    let to_buffer = if pending_cr {
        format!("{remaining}\r")
    } else {
        remaining.to_owned()
    };

    store_line_buffer(ctx, to_buffer, max_partial_event_bytes)?;

    if output.is_empty() {
        Ok(Bytes::new())
    } else {
        Ok(Bytes::from(output))
    }
}

/// Transform complete SSE event blocks and return the unconsumed suffix.
fn process_complete_event_blocks<'a>(
    ctx: &mut HttpFilterContext<'_>,
    mut remaining: &'a str,
    output: &mut Vec<u8>,
    max_tool_blocks: usize,
) -> Result<&'a str, FilterError> {
    while let Some((event_block, rest)) = remaining.split_once("\n\n") {
        remaining = rest;
        process_event_block(ctx, event_block, output, max_tool_blocks)?;
        if is_stream_failed(ctx) {
            return Ok("");
        }
    }
    Ok(remaining)
}

/// Prefix the current chunk with any buffered incomplete SSE text.
///
/// Returns borrowed `chunk_str` when there is no leftover prefix.
fn combine_chunk_with_leftover<'a>(leftover: Option<&str>, chunk_str: &'a str) -> Cow<'a, str> {
    match leftover {
        Some(prefix) if !prefix.is_empty() => {
            let mut combined = String::with_capacity(prefix.len() + chunk_str.len());
            combined.push_str(prefix);
            combined.push_str(chunk_str);
            Cow::Owned(combined)
        },
        _ => Cow::Borrowed(chunk_str),
    }
}

/// Hold back a single trailing CR that may be the first half of a CRLF pair.
fn split_deferred_trailing_cr(combined: &str, end_of_stream: bool) -> (&str, bool) {
    let defer = !end_of_stream && combined.ends_with('\r') && !combined.ends_with("\r\r");
    if !defer {
        return (combined, false);
    }
    combined
        .strip_suffix('\r')
        .map_or((combined, false), |without| (without, true))
}

/// Store bounded incomplete SSE event data between response chunks.
fn store_line_buffer(
    ctx: &mut HttpFilterContext<'_>,
    buffer: String,
    max_partial_event_bytes: usize,
) -> Result<(), FilterError> {
    if buffer.is_empty() {
        ctx.filter_metadata.remove(LINE_BUFFER_KEY);
        return Ok(());
    }

    if buffer.len() > max_partial_event_bytes {
        ctx.filter_metadata.remove(LINE_BUFFER_KEY);
        let msg = format!(
            "anthropic_messages_to_chat_completions_stream: incomplete SSE event exceeds {max_partial_event_bytes} bytes"
        );
        return Err(msg.into());
    }

    ctx.filter_metadata.insert(LINE_BUFFER_KEY.to_owned(), buffer);
    Ok(())
}

/// Whether the filter has been armed in the response phase.
fn is_armed(ctx: &HttpFilterContext<'_>) -> bool {
    ctx.filter_metadata.get(ARMED_KEY).is_some_and(|v| v == "true")
}

/// Whether a terminal error event has already ended the client-visible stream.
fn is_stream_failed(ctx: &HttpFilterContext<'_>) -> bool {
    ctx.filter_metadata
        .get(STREAM_STATE_KEY)
        .is_some_and(|v| v == STREAM_STATE_FAILED)
}

/// Whether the filter should arm: streaming request, SSE Content-Type, success status.
fn should_arm(ctx: &HttpFilterContext<'_>) -> bool {
    if !is_streaming_request(ctx) {
        return false;
    }

    let is_sse = ctx
        .response_header
        .as_ref()
        .and_then(|r| r.headers.get(http::header::CONTENT_TYPE))
        .and_then(|v| v.to_str().ok())
        .is_some_and(is_event_stream_content_type);

    if !is_sse {
        debug!("streaming request but non-SSE response; skipping stream transformation");
        return false;
    }

    let is_encoded = ctx
        .response_header
        .as_ref()
        .is_some_and(|response| response.headers.contains_key(http::header::CONTENT_ENCODING));
    if is_encoded {
        debug!("streaming SSE response is encoded; skipping stream transformation");
        return false;
    }

    let is_success = ctx.response_header.as_ref().is_none_or(|r| r.status.is_success());
    if !is_success {
        debug!("streaming SSE response with non-2xx status; passing through error body");
        return false;
    }

    true
}

/// Whether an upstream filter classified this as a streaming request.
fn is_streaming_request(ctx: &HttpFilterContext<'_>) -> bool {
    ctx.filter_metadata
        .get("anthropic_messages_format.stream")
        .is_some_and(|v| v == "true")
        || ctx
            .filter_metadata
            .get("anthropic_messages_to_chat_completions.streaming")
            .is_some_and(|v| v == "true")
}


/// Process a single SSE event block (lines between double-newlines).
///
/// Collects all `data` fields into one newline-delimited payload before
/// processing it. Accepts bare `data`, `data: value`, and `data:value`
/// per the SSE specification.
///
/// # Errors
///
/// Returns [`FilterError`] if a complete data event is neither the `[DONE]`
/// sentinel nor a valid Chat Completions JSON object, or if transforming the
/// chunk exceeds a configured limit. Such events are failed closed rather than
/// silently discarded, which would erase their content from a stream that still
/// reports success.
fn process_event_block(
    ctx: &mut HttpFilterContext<'_>,
    block: &str,
    output: &mut Vec<u8>,
    max_tool_blocks: usize,
) -> Result<(), FilterError> {
    let mut event_data = None::<Cow<'_, str>>;

    for line in block.lines() {
        let data = if line == "data" {
            Some("")
        } else {
            line.strip_prefix("data:").map(|d| d.strip_prefix(' ').unwrap_or(d))
        };

        if let Some(data) = data {
            match &mut event_data {
                Some(event_data) => {
                    let event_data = event_data.to_mut();
                    event_data.push('\n');
                    event_data.push_str(data);
                },
                None => event_data = Some(Cow::Borrowed(data)),
            }
        }
    }

    if let Some(data) = event_data {
        if data == OPENAI_DONE_SENTINEL {
            emit_done(ctx, output);
        } else {
            transform_data_event(ctx, &data, output, max_tool_blocks)?;
        }
    }

    Ok(())
}

/// Parse a non-sentinel SSE data event as a Chat Completions chunk and
/// transform it into Anthropic events.
///
/// # Errors
///
/// Returns [`FilterError`] when `data` is not valid JSON, is not a JSON object,
/// or when transforming the chunk exceeds a configured limit. Failing closed
/// keeps a malformed event from silently vanishing — dropping it would erase
/// any text, tool call, finish reason, or usage it carried while the
/// transformed stream still reports success.
fn transform_data_event(
    ctx: &mut HttpFilterContext<'_>,
    data: &str,
    output: &mut Vec<u8>,
    max_tool_blocks: usize,
) -> Result<(), FilterError> {
    let chunk = serde_json::from_str::<Value>(data).map_err(|err| {
        FilterError::from(format!(
            "anthropic_messages_to_chat_completions_stream: upstream SSE data event is not valid JSON: {err}"
        ))
    })?;
    if !chunk.is_object() {
        return Err(FilterError::from(
            "anthropic_messages_to_chat_completions_stream: upstream SSE data event is not a JSON object",
        ));
    }
    transform_chunk(ctx, &chunk, output, max_tool_blocks)
}

// -----------------------------------------------------------------------------
// Per-Chunk Transformation
// -----------------------------------------------------------------------------

/// Transform a single `OpenAI` SSE chunk into Anthropic events.
fn transform_chunk(
    ctx: &mut HttpFilterContext<'_>,
    chunk: &Value,
    output: &mut Vec<u8>,
    max_tool_blocks: usize,
) -> Result<(), FilterError> {
    let started = ctx
        .filter_metadata
        .get(STREAM_STATE_KEY)
        .is_some_and(|v| v == STREAM_STATE_STARTED);

    if !started {
        emit_message_start(ctx, chunk, output);
    }

    if let Some(choice) = extract_first_choice(chunk) {
        if let Some(delta) = choice.get("delta") {
            transform_delta(ctx, delta, output, max_tool_blocks)?;
        }
        if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
            ctx.set_metadata(FINISH_REASON_KEY, reason.to_owned());
        }
        // vLLM reports the matched stop string in a choice-level `stop_reason`;
        // only a client-provided sequence may be reported back as one.
        if let Some(matched) = choice.get("stop_reason").and_then(Value::as_str)
            && client_stop_sequences(ctx).iter().any(|sequence| sequence == matched)
        {
            ctx.set_metadata(STOP_SEQUENCE_KEY, matched.to_owned());
        }
    }

    extract_usage_tokens(ctx, chunk);
    Ok(())
}

/// Extract token usage from a chunk and store in filter metadata.
fn extract_usage_tokens(ctx: &mut HttpFilterContext<'_>, chunk: &Value) {
    if let Some(usage) = chunk.get("usage") {
        if let Some(ot) = usage.get("completion_tokens").and_then(Value::as_u64) {
            ctx.set_metadata(OUTPUT_TOKENS_KEY, ot.to_string());
        }
        if let Some(pt) = usage.get("prompt_tokens").and_then(Value::as_u64) {
            ctx.set_metadata(INPUT_TOKENS_KEY, pt.to_string());
        }
        if let Some(ct) = usage
            .get("prompt_tokens_details")
            .and_then(|d| d.get("cached_tokens"))
            .and_then(Value::as_u64)
        {
            ctx.set_metadata(CACHE_READ_TOKENS_KEY, ct.to_string());
        }
    }
}

/// Emit the initial `message_start` event and mark the stream as started.
fn emit_message_start(ctx: &mut HttpFilterContext<'_>, chunk: &Value, output: &mut Vec<u8>) {
    let model = chunk.get("model").and_then(Value::as_str).unwrap_or("");

    emit_event(
        output,
        "message_start",
        &serde_json::json!({
            "type": "message_start",
            "message": {
                "id": format!("msg_{:016x}", generate_timestamp_id()),
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": null,
                "stop_sequence": null,
                "stop_details": null,
                "usage": message_start_usage(),
                "container": null
            }
        }),
    );
    ctx.set_metadata(STREAM_STATE_KEY, STREAM_STATE_STARTED.to_owned());
}

/// Extract the first choice from a Chat Completions streaming chunk.
///
/// Anthropic's response format is structurally single-choice, so only
/// `choices[0]` can be mapped.
fn extract_first_choice(chunk: &Value) -> Option<&Value> {
    chunk.get("choices").and_then(Value::as_array).and_then(|c| c.first())
}

/// Generate a timestamp-based identifier for message IDs.
fn generate_timestamp_id() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        & 0xFFFF_FFFF_FFFF_FFFF_u128
}

// -----------------------------------------------------------------------------
// Delta Transformation
// -----------------------------------------------------------------------------

/// Transform a delta object from a streaming chunk.
fn transform_delta(
    ctx: &mut HttpFilterContext<'_>,
    delta: &Value,
    output: &mut Vec<u8>,
    max_tool_blocks: usize,
) -> Result<(), FilterError> {
    if let Some(content) = delta.get("content").and_then(Value::as_str) {
        emit_text_delta(ctx, content, output);
    }

    if let Some(Value::Array(tool_calls)) = delta.get("tool_calls") {
        close_text_block_if_open(ctx, output);
        for tc in tool_calls {
            transform_tool_delta(ctx, tc, output, max_tool_blocks)?;
            if is_stream_failed(ctx) {
                break;
            }
        }
    }

    Ok(())
}

/// Emit a text content delta, opening a new block if needed.
fn emit_text_delta(ctx: &mut HttpFilterContext<'_>, content: &str, output: &mut Vec<u8>) {
    if !is_text_block_open(ctx) {
        let idx = get_block_index(ctx);
        let content_block = ContentBlock::text("");
        emit_event(
            output,
            "content_block_start",
            &serde_json::json!({
                "type": "content_block_start",
                "index": idx,
                "content_block": content_block
            }),
        );
        ctx.set_metadata(TEXT_BLOCK_OPEN_KEY, "true".to_owned());
    }

    let idx = get_block_index(ctx);
    emit_event(
        output,
        "content_block_delta",
        &serde_json::json!({
            "type": "content_block_delta",
            "index": idx,
            "delta": {"type": "text_delta", "text": content}
        }),
    );
}

// -----------------------------------------------------------------------------
// Tool Delta Transformation
// -----------------------------------------------------------------------------

/// Transform a tool call delta into Anthropic content block events.
fn transform_tool_delta(
    ctx: &mut HttpFilterContext<'_>,
    tc: &Value,
    output: &mut Vec<u8>,
    max_tool_blocks: usize,
) -> Result<(), FilterError> {
    let tool_call_key = tool_call_key(tc);

    if !is_tool_block_open(ctx, &tool_call_key)
        && !emit_tool_block_start(ctx, &tool_call_key, tc, output, max_tool_blocks)?
    {
        return Ok(());
    }

    emit_tool_arguments_delta(ctx, &tool_call_key, tc, output);

    Ok(())
}

/// Close any open text content block and advance the block index.
fn close_text_block_if_open(ctx: &mut HttpFilterContext<'_>, output: &mut Vec<u8>) {
    if !is_text_block_open(ctx) {
        return;
    }

    let idx = get_block_index(ctx);
    emit_event(
        output,
        "content_block_stop",
        &serde_json::json!({"type": "content_block_stop", "index": idx}),
    );
    increment_block_index(ctx);
    ctx.set_metadata(TEXT_BLOCK_OPEN_KEY, "false".to_owned());
}

/// Return the stable key used to associate OpenAI tool-call deltas.
fn tool_call_key(tc: &Value) -> String {
    tc.get("index")
        .and_then(Value::as_u64)
        .map_or_else(|| "legacy".to_owned(), |idx| idx.to_string())
}

/// Emit a `content_block_start` for a tool-use block.
///
/// # Errors
///
/// Fails closed with a [`FilterError`] before opening the
/// `max_tool_blocks + 1`th block, bounding the per-response tool-call
/// state that would otherwise grow with every unique upstream index.
fn emit_tool_block_start(
    ctx: &mut HttpFilterContext<'_>,
    tool_call_key: &str,
    tc: &Value,
    output: &mut Vec<u8>,
    max_tool_blocks: usize,
) -> Result<bool, FilterError> {
    let opened = get_tool_block_count(ctx);
    if opened >= max_tool_blocks {
        return Err(format!(
            "anthropic_messages_to_chat_completions_stream: streaming tool-call content blocks exceed max_tool_blocks ({max_tool_blocks})"
        )
        .into());
    }

    let idx = get_block_index(ctx);
    let Some((id, name)) = extract_or_fail_tool_call(ctx, tc, output) else {
        return Ok(false);
    };
    let content_block = ContentBlock::tool_use(id, serde_json::Map::new(), name);

    emit_event(
        output,
        "content_block_start",
        &serde_json::json!({
            "type": "content_block_start",
            "index": idx,
            "content_block": content_block
        }),
    );
    set_tool_block_index(ctx, tool_call_key, idx);
    set_tool_block_open(ctx, tool_call_key, true);
    increment_block_index(ctx);
    ctx.set_metadata(TOOL_BLOCK_COUNT_KEY, (opened + 1).to_string());

    Ok(true)
}

/// Validate an upstream tool call or terminate the client-visible stream.
fn extract_or_fail_tool_call<'a>(
    ctx: &mut HttpFilterContext<'_>,
    tc: &'a Value,
    output: &mut Vec<u8>,
) -> Option<(&'a str, &'a str)> {
    match extract_tool_id_and_name(tc) {
        Ok(fields) => Some(fields),
        Err(error) => {
            debug!(%error, "invalid upstream streaming tool call");
            emit_upstream_transform_error(output);
            ctx.filter_metadata.remove(LINE_BUFFER_KEY);
            ctx.filter_metadata.remove(UTF8_BUFFER_KEY);
            ctx.set_metadata(STREAM_STATE_KEY, STREAM_STATE_FAILED.to_owned());
            None
        },
    }
}

/// Extract and validate the tool-call ID and function name from an OpenAI tool-call delta.
fn extract_tool_id_and_name(tc: &Value) -> Result<(&str, &str), FilterError> {
    let id = tc
        .get("id")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            FilterError::from(
                "anthropic_messages_to_chat_completions_stream: tool call missing required non-empty `id`",
            )
        })?;
    if !crate::anthropic::wire::is_valid_tool_use_id(id) {
        return Err(FilterError::from(
            "anthropic_messages_to_chat_completions_stream: tool call `id` must match ^[a-zA-Z0-9_-]+$",
        ));
    }
    let name = tc
        .get("function")
        .and_then(|f| f.get("name"))
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            FilterError::from(
                "anthropic_messages_to_chat_completions_stream: tool call missing required non-empty function `name`",
            )
        })?;
    Ok((id, name))
}

/// Emit an `input_json_delta` if the tool call has non-empty arguments.
fn emit_tool_arguments_delta(ctx: &HttpFilterContext<'_>, tool_call_key: &str, tc: &Value, output: &mut Vec<u8>) {
    let Some(args) = tc
        .get("function")
        .and_then(|f| f.get("arguments"))
        .and_then(Value::as_str)
    else {
        return;
    };

    if args.is_empty() {
        return;
    }

    let idx = get_tool_block_index(ctx, tool_call_key).unwrap_or_else(|| get_block_index(ctx));
    emit_event(
        output,
        "content_block_delta",
        &serde_json::json!({
            "type": "content_block_delta",
            "index": idx,
            "delta": {"type": "input_json_delta", "partial_json": args}
        }),
    );
}

/// Close a specific open tool content block.
fn close_tool_block(ctx: &mut HttpFilterContext<'_>, tool_call_key: &str, output: &mut Vec<u8>) {
    if !is_tool_block_open(ctx, tool_call_key) {
        return;
    }

    let Some(idx) = get_tool_block_index(ctx, tool_call_key) else {
        return;
    };

    emit_event(
        output,
        "content_block_stop",
        &serde_json::json!({"type": "content_block_stop", "index": idx}),
    );
    set_tool_block_open(ctx, tool_call_key, false);
}

// -----------------------------------------------------------------------------
// Stream Completion
// -----------------------------------------------------------------------------

/// Emit final events when `[DONE]` is received.
///
/// If `[DONE]` arrives before any upstream chunk initialized the stream, a
/// `message_start` is synthesized first so the terminal `message_delta` and
/// `message_stop` never appear without the required opening event, yielding a
/// structurally valid (empty) Anthropic stream instead of a malformed one.
fn emit_done(ctx: &mut HttpFilterContext<'_>, output: &mut Vec<u8>) {
    if is_stream_failed(ctx) {
        return;
    }

    let started = ctx
        .filter_metadata
        .get(STREAM_STATE_KEY)
        .is_some_and(|v| v == STREAM_STATE_STARTED);
    if !started {
        emit_message_start(ctx, &Value::Null, output);
    }

    emit_final_block_stop(ctx, output);
    emit_message_delta(ctx, output);
    emit_event(output, "message_stop", &serde_json::json!({"type": "message_stop"}));
    debug!("streaming transformation complete");
}

/// Close any open content block at end of stream.
fn emit_final_block_stop(ctx: &mut HttpFilterContext<'_>, output: &mut Vec<u8>) {
    if is_text_block_open(ctx) {
        close_text_block_if_open(ctx, output);
    }
    for (_, tool_call_key) in open_tool_blocks(ctx) {
        close_tool_block(ctx, &tool_call_key, output);
    }
}

/// Emit the `message_delta` event with stop reason and usage.
fn emit_message_delta(ctx: &HttpFilterContext<'_>, output: &mut Vec<u8>) {
    let stop_sequence = ctx.filter_metadata.get(STOP_SEQUENCE_KEY);
    let stop_reason = match ctx.filter_metadata.get(FINISH_REASON_KEY).map(String::as_str) {
        Some("stop") if stop_sequence.is_some() => "stop_sequence",
        Some(reason) => map_stop_reason(reason),
        None => "end_turn",
    };

    let usage = collect_delta_usage(ctx);

    emit_event(
        output,
        "message_delta",
        &serde_json::json!({
            "type": "message_delta",
            "delta": {
                "container": null,
                "stop_details": null,
                "stop_reason": stop_reason,
                "stop_sequence": stop_sequence
            },
            "usage": usage
        }),
    );
}

/// Collect token counts from metadata and build the terminal delta usage.
fn collect_delta_usage(ctx: &HttpFilterContext<'_>) -> MessageDeltaUsage {
    let output_tokens: u64 = ctx
        .filter_metadata
        .get(OUTPUT_TOKENS_KEY)
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    let prompt_tokens: Option<u64> = ctx.filter_metadata.get(INPUT_TOKENS_KEY).and_then(|v| v.parse().ok());

    let cache_read: Option<u64> = ctx
        .filter_metadata
        .get(CACHE_READ_TOKENS_KEY)
        .and_then(|v| v.parse().ok());

    let input_tokens = prompt_tokens.map(|pt| match cache_read {
        Some(cached) => pt.saturating_sub(cached),
        None => pt,
    });

    MessageDeltaUsage::new(output_tokens, input_tokens, cache_read)
}

/// Build a schema-complete Anthropic `Message.usage` value.
fn message_start_usage() -> MessageUsage {
    MessageUsage::new(0, 0, None)
}

/// Map `OpenAI` finish reasons to Anthropic stop reasons.
fn map_stop_reason(reason: &str) -> &str {
    match reason {
        "tool_calls" => "tool_use",
        "length" => "max_tokens",
        _ => "end_turn",
    }
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// Check whether a text content block is currently open.
fn is_text_block_open(ctx: &HttpFilterContext<'_>) -> bool {
    ctx.filter_metadata
        .get(TEXT_BLOCK_OPEN_KEY)
        .is_some_and(|v| v == "true")
}

/// Get the current block index from metadata.
fn get_block_index(ctx: &HttpFilterContext<'_>) -> u32 {
    ctx.filter_metadata
        .get(BLOCK_INDEX_KEY)
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

/// Increment the block index in metadata.
fn increment_block_index(ctx: &mut HttpFilterContext<'_>) {
    let current = get_block_index(ctx);
    ctx.set_metadata(BLOCK_INDEX_KEY, (current + 1).to_string());
}

/// Return how many tool-call content blocks have opened this response.
fn get_tool_block_count(ctx: &HttpFilterContext<'_>) -> usize {
    ctx.filter_metadata
        .get(TOOL_BLOCK_COUNT_KEY)
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

/// Build the metadata key for a tool call's Anthropic block index.
fn tool_block_index_key(tool_call_key: &str) -> String {
    format!("{TOOL_BLOCK_KEY_PREFIX}{tool_call_key}{TOOL_BLOCK_INDEX_SUFFIX}")
}

/// Build the metadata key for a tool call's open/closed state.
fn tool_block_open_key(tool_call_key: &str) -> String {
    format!("{TOOL_BLOCK_KEY_PREFIX}{tool_call_key}{TOOL_BLOCK_OPEN_SUFFIX}")
}

/// Record the Anthropic block index assigned to an OpenAI tool-call index.
fn set_tool_block_index(ctx: &mut HttpFilterContext<'_>, tool_call_key: &str, idx: u32) {
    ctx.set_metadata(tool_block_index_key(tool_call_key), idx.to_string());
}

/// Return the Anthropic block index assigned to an OpenAI tool-call index.
fn get_tool_block_index(ctx: &HttpFilterContext<'_>, tool_call_key: &str) -> Option<u32> {
    ctx.filter_metadata
        .get(&tool_block_index_key(tool_call_key))
        .and_then(|v| v.parse().ok())
}

/// Record whether a tool call's Anthropic content block remains open.
fn set_tool_block_open(ctx: &mut HttpFilterContext<'_>, tool_call_key: &str, open: bool) {
    ctx.set_metadata(tool_block_open_key(tool_call_key), open.to_string());
}

/// Check whether a tool call's Anthropic content block is currently open.
fn is_tool_block_open(ctx: &HttpFilterContext<'_>, tool_call_key: &str) -> bool {
    ctx.filter_metadata
        .get(&tool_block_open_key(tool_call_key))
        .is_some_and(|v| v == "true")
}

/// Return open tool blocks ordered by their Anthropic content block index.
fn open_tool_blocks(ctx: &HttpFilterContext<'_>) -> Vec<(u32, String)> {
    let mut blocks = ctx
        .filter_metadata
        .iter()
        .filter_map(|(key, value)| {
            if value != "true" {
                return None;
            }
            let tool_call_key = key
                .strip_prefix(TOOL_BLOCK_KEY_PREFIX)?
                .strip_suffix(TOOL_BLOCK_OPEN_SUFFIX)?;
            get_tool_block_index(ctx, tool_call_key).map(|idx| (idx, tool_call_key.to_owned()))
        })
        .collect::<Vec<_>>();
    blocks.sort_by_key(|(idx, _)| *idx);
    blocks
}

/// Write a single SSE event to the output buffer.
///
/// Capacity is not reserved per event: a guessed JSON size would force a
/// realloc on later events in the same chunk once spare capacity drops
/// below that guess.
///
/// If JSON serialization fails after a partial write, the payload is
/// truncated so `data:` is empty, matching `to_string().unwrap_or_default()`.
fn emit_event(output: &mut Vec<u8>, event_type: &str, data: &Value) {
    output.extend_from_slice(b"event: ");
    output.extend_from_slice(event_type.as_bytes());
    output.extend_from_slice(b"\ndata: ");
    let json_start = output.len();
    if serde_json::to_writer(&mut *output, data).is_err() {
        output.truncate(json_start);
    }
    output.extend_from_slice(b"\n\n");
}

/// Emit a client-safe terminal error when an upstream tool call cannot be
/// represented in Anthropic's streaming schema.
fn emit_upstream_transform_error(output: &mut Vec<u8>) {
    let body = crate::anthropic::wire::error_body("api_error", "upstream response could not be transformed", None);
    output.extend_from_slice(b"event: error\ndata: ");
    output.extend_from_slice(&body);
    output.extend_from_slice(b"\n\n");
}

/// Normalize SSE line endings: `\r\n` → `\n`, standalone `\r` → `\n`.
///
/// LF-only input is returned borrowed.
fn normalize_line_endings(s: &str) -> Cow<'_, str> {
    if s.contains('\r') {
        Cow::Owned(s.replace("\r\n", "\n").replace('\r', "\n"))
    } else {
        Cow::Borrowed(s)
    }
}

/// Hex-encode a byte slice (for buffering incomplete UTF-8 sequences).
fn encode_hex_bytes(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Decode a hex-encoded byte slice.
fn decode_hex_bytes(hex: &str) -> Vec<u8> {
    hex.as_bytes()
        .chunks_exact(2)
        .filter_map(|pair| {
            let hi = hex_nibble(*pair.first()?)?;
            let lo = hex_nibble(*pair.last()?)?;
            Some(hi << 4 | lo)
        })
        .collect()
}

/// Convert a single lowercase hex digit to its numeric value.
fn hex_nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        _ => None,
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests")]
mod tests {
    use std::borrow::Cow;

    use super::*;

    #[test]
    fn default_config_parses() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("{}").unwrap();
        let filter = AnthropicMessagesToChatCompletionsStreamFilter::from_config(&yaml).unwrap();

        assert_eq!(
            filter.name(),
            "anthropic_messages_to_chat_completions_stream",
            "filter name should match"
        );
    }

    #[tokio::test]
    async fn on_request_prevents_upstream_response_encoding() {
        let filter = make_filter();
        let req = crate::test_utils::make_request(http::Method::POST, "/v1/messages");
        let mut ctx = crate::test_utils::make_filter_context(&req);

        drop(filter.on_request(&mut ctx).await.unwrap());

        assert!(
            ctx.request_headers_to_remove.contains(&http::header::ACCEPT_ENCODING),
            "stream transformation requires an unencoded upstream representation"
        );
    }

    #[test]
    fn incremental_text_chunks_transformed_immediately() {
        let (filter, mut ctx) = make_filter_and_context();

        let chunk1 = "data: {\"id\":\"c1\",\"model\":\"gpt-4\",\"choices\":[{\"delta\":{\"role\":\"assistant\"},\"index\":0}]}\n\n";
        let mut body1 = Some(Bytes::from(chunk1));
        drop(filter.on_response_body(&mut ctx, &mut body1, false).unwrap());

        let out1 = String::from_utf8(body1.unwrap().to_vec()).unwrap();
        assert!(
            out1.contains("message_start"),
            "first chunk should emit message_start immediately"
        );

        let chunk2 = "data: {\"id\":\"c1\",\"model\":\"gpt-4\",\"choices\":[{\"delta\":{\"content\":\"Hello\"},\"index\":0}]}\n\n";
        let mut body2 = Some(Bytes::from(chunk2));
        drop(filter.on_response_body(&mut ctx, &mut body2, false).unwrap());

        let out2 = String::from_utf8(body2.unwrap().to_vec()).unwrap();
        assert!(
            out2.contains("text_delta"),
            "second chunk should emit text_delta immediately"
        );
        assert!(out2.contains("Hello"), "text content should be forwarded immediately");
        let start = event_data(&out2, "content_block_start");
        assert_eq!(
            start.pointer("/content_block/citations"),
            Some(&Value::Null),
            "streaming text citations should be null"
        );
    }

    #[test]
    fn partial_chunk_buffered_until_complete() {
        let (filter, mut ctx) = make_filter_and_context();

        let partial = "data: {\"id\":\"c1\",\"model\":\"gpt-4\",\"choices\":[{\"delta\":{\"role\":\"assistant\"";
        let mut body1 = Some(Bytes::from(partial));
        drop(filter.on_response_body(&mut ctx, &mut body1, false).unwrap());

        let out1 = body1.unwrap();
        assert!(out1.is_empty(), "partial chunk should produce no output");

        let rest = "},\"index\":0}]}\n\n";
        let mut body2 = Some(Bytes::from(rest));
        drop(filter.on_response_body(&mut ctx, &mut body2, false).unwrap());

        let out2 = String::from_utf8(body2.unwrap().to_vec()).unwrap();
        assert!(
            out2.contains("message_start"),
            "completed chunk should emit message_start"
        );
    }

    #[test]
    fn done_emits_final_events() {
        let (filter, mut ctx) = make_filter_and_context();

        let chunk1 = "data: {\"id\":\"c1\",\"model\":\"gpt-4\",\"choices\":[{\"delta\":{\"content\":\"Hi\"},\"index\":0,\"finish_reason\":\"stop\"}]}\n\n";
        let mut body1 = Some(Bytes::from(chunk1));
        drop(filter.on_response_body(&mut ctx, &mut body1, false).unwrap());

        let done = "data: [DONE]\n\n";
        let mut body2 = Some(Bytes::from(done));
        drop(filter.on_response_body(&mut ctx, &mut body2, false).unwrap());

        let out = String::from_utf8(body2.unwrap().to_vec()).unwrap();
        assert!(out.contains("message_delta"), "DONE should emit message_delta");
        assert!(out.contains("message_stop"), "DONE should emit message_stop");
        assert!(out.contains("end_turn"), "stop reason should be end_turn");
    }

    #[test]
    fn message_start_usage_matches_anthropic_schema() {
        let (filter, mut ctx) = make_filter_and_context();

        let chunk = "data: {\"id\":\"c1\",\"model\":\"gpt-4\",\"choices\":[{\"delta\":{\"role\":\"assistant\"},\"index\":0}]}\n\n";
        let mut body = Some(Bytes::from(chunk));
        drop(filter.on_response_body(&mut ctx, &mut body, false).unwrap());

        let out = String::from_utf8(body.unwrap().to_vec()).unwrap();
        let event = event_data(&out, "message_start");
        let message = event.get("message").unwrap();
        let usage = message.get("usage").unwrap();

        assert_null_fields(message, &["stop_details", "container"], "message_start");
        assert_null_fields(
            usage,
            &[
                "cache_creation",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
                "inference_geo",
                "output_tokens_details",
                "server_tool_use",
                "service_tier",
            ],
            "message_start usage",
        );
        assert_u64_field(usage, "input_tokens", 0, "message_start usage");
        assert_u64_field(usage, "output_tokens", 0, "message_start usage");
    }

    #[test]
    fn message_delta_reports_matched_stop_sequence() {
        let (filter, mut ctx) = make_filter_and_context();
        ctx.set_metadata(
            crate::anthropic::messages_to_chat_completions::STOP_SEQUENCES_KEY,
            r#"[","]"#,
        );

        let chunk1 = "data: {\"id\":\"c1\",\"model\":\"gpt-4\",\"choices\":[{\"delta\":{\"content\":\"Count: 1\"},\"index\":0,\"finish_reason\":\"stop\",\"stop_reason\":\",\"}]}\n\n";
        let mut body1 = Some(Bytes::from(chunk1));
        drop(filter.on_response_body(&mut ctx, &mut body1, false).unwrap());
        let mut body2 = Some(Bytes::from("data: [DONE]\n\n"));
        drop(filter.on_response_body(&mut ctx, &mut body2, false).unwrap());

        let out = String::from_utf8(body2.unwrap().to_vec()).unwrap();
        let event = event_data(&out, "message_delta");
        let delta = event.get("delta").unwrap();
        assert_eq!(
            delta.get("stop_reason").and_then(Value::as_str),
            Some("stop_sequence"),
            "matched stop → stop_sequence"
        );
        assert_eq!(
            delta.get("stop_sequence").and_then(Value::as_str),
            Some(","),
            "matched value is reported"
        );
    }

    #[test]
    fn message_delta_ignores_stop_reason_outside_client_sequences() {
        let (filter, mut ctx) = make_filter_and_context();
        ctx.set_metadata(
            crate::anthropic::messages_to_chat_completions::STOP_SEQUENCES_KEY,
            r#"[","]"#,
        );

        let chunk1 = "data: {\"id\":\"c1\",\"model\":\"gpt-4\",\"choices\":[{\"delta\":{\"content\":\"Hi\"},\"index\":0,\"finish_reason\":\"stop\",\"stop_reason\":\"</s>\"}]}\n\n";
        let mut body1 = Some(Bytes::from(chunk1));
        drop(filter.on_response_body(&mut ctx, &mut body1, false).unwrap());
        let mut body2 = Some(Bytes::from("data: [DONE]\n\n"));
        drop(filter.on_response_body(&mut ctx, &mut body2, false).unwrap());

        let out = String::from_utf8(body2.unwrap().to_vec()).unwrap();
        let event = event_data(&out, "message_delta");
        let delta = event.get("delta").unwrap();
        assert_eq!(
            delta.get("stop_reason").and_then(Value::as_str),
            Some("end_turn"),
            "server-side stop → end_turn"
        );
        assert_null_fields(delta, &["stop_sequence"], "server-side stop");
    }

    #[test]
    fn message_delta_usage_matches_anthropic_schema() {
        let (filter, mut ctx) = make_filter_and_context();

        let chunk1 = "data: {\"id\":\"c1\",\"model\":\"gpt-4\",\"choices\":[{\"delta\":{\"content\":\"Hi\"},\"index\":0,\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":15,\"completion_tokens\":7}}\n\n";
        let mut body1 = Some(Bytes::from(chunk1));
        drop(filter.on_response_body(&mut ctx, &mut body1, false).unwrap());

        let done = "data: [DONE]\n\n";
        let mut body2 = Some(Bytes::from(done));
        drop(filter.on_response_body(&mut ctx, &mut body2, false).unwrap());

        let out = String::from_utf8(body2.unwrap().to_vec()).unwrap();
        let event = event_data(&out, "message_delta");
        let delta = event.get("delta").unwrap();
        let usage = event.get("usage").unwrap();

        assert_null_fields(delta, &["container", "stop_details", "stop_sequence"], "message_delta");
        assert_eq!(
            delta.get("stop_reason").and_then(Value::as_str),
            Some("end_turn"),
            "message_delta should include stop_reason"
        );
        assert_null_fields(
            usage,
            &[
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
                "output_tokens_details",
                "server_tool_use",
            ],
            "message_delta usage",
        );
        assert_u64_field(usage, "output_tokens", 7, "message_delta usage");
        assert_u64_field(usage, "input_tokens", 15, "message_delta usage");
    }

    #[test]
    fn message_delta_usage_with_cached_tokens() {
        let (filter, mut ctx) = make_filter_and_context();

        let chunk1 = "data: {\"id\":\"c1\",\"model\":\"gpt-4\",\"choices\":[{\"delta\":{\"content\":\"Hi\"},\"index\":0,\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":5,\"prompt_tokens_details\":{\"cached_tokens\":80}}}\n\n";
        let mut body1 = Some(Bytes::from(chunk1));
        drop(filter.on_response_body(&mut ctx, &mut body1, false).unwrap());

        let done = "data: [DONE]\n\n";
        let mut body2 = Some(Bytes::from(done));
        drop(filter.on_response_body(&mut ctx, &mut body2, false).unwrap());

        let out = String::from_utf8(body2.unwrap().to_vec()).unwrap();
        let event = event_data(&out, "message_delta");
        let usage = event.get("usage").unwrap();

        assert_u64_field(usage, "output_tokens", 5, "message_delta usage");
        assert_u64_field(
            usage,
            "input_tokens",
            20,
            "input_tokens should exclude cached (100 - 80)",
        );
        assert_u64_field(usage, "cache_read_input_tokens", 80, "message_delta usage");
    }

    #[test]
    fn message_delta_usage_without_usage_chunk() {
        let (filter, mut ctx) = make_filter_and_context();

        let chunk1 = "data: {\"id\":\"c1\",\"model\":\"gpt-4\",\"choices\":[{\"delta\":{\"content\":\"Hi\"},\"index\":0,\"finish_reason\":\"stop\"}]}\n\n";
        let mut body1 = Some(Bytes::from(chunk1));
        drop(filter.on_response_body(&mut ctx, &mut body1, false).unwrap());

        let done = "data: [DONE]\n\n";
        let mut body2 = Some(Bytes::from(done));
        drop(filter.on_response_body(&mut ctx, &mut body2, false).unwrap());

        let out = String::from_utf8(body2.unwrap().to_vec()).unwrap();
        let event = event_data(&out, "message_delta");
        let usage = event.get("usage").unwrap();

        assert_u64_field(usage, "output_tokens", 0, "no usage chunk means zero output_tokens");
        assert_null_fields(
            usage,
            &["input_tokens", "cache_read_input_tokens"],
            "no usage chunk means null input fields",
        );
    }

    #[test]
    fn no_full_response_buffering() {
        let (filter, mut ctx) = make_filter_and_context();

        let chunks = vec![
            "data: {\"id\":\"c1\",\"model\":\"gpt-4\",\"choices\":[{\"delta\":{\"role\":\"assistant\"},\"index\":0}]}\n\n",
            "data: {\"id\":\"c1\",\"model\":\"gpt-4\",\"choices\":[{\"delta\":{\"content\":\"A\"},\"index\":0}]}\n\n",
            "data: {\"id\":\"c1\",\"model\":\"gpt-4\",\"choices\":[{\"delta\":{\"content\":\"B\"},\"index\":0}]}\n\n",
        ];

        let mut outputs_with_content = 0;
        for chunk in chunks {
            let mut body = Some(Bytes::from(chunk));
            drop(filter.on_response_body(&mut ctx, &mut body, false).unwrap());
            if !body.unwrap().is_empty() {
                outputs_with_content += 1;
            }
        }

        assert!(
            outputs_with_content >= 3,
            "each chunk should produce output immediately, got {outputs_with_content}/3"
        );
    }

    #[tokio::test]
    async fn on_response_arms_when_streaming_request_and_sse_response() {
        let filter = make_filter();
        let req = crate::test_utils::make_request(http::Method::POST, "/v1/messages");
        let mut ctx = crate::test_utils::make_filter_context(&req);
        let mut resp = crate::test_utils::make_response();
        resp.headers.insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("text/event-stream"),
        );
        ctx.response_header = Some(&mut resp);
        ctx.set_metadata("anthropic_messages_format.stream", "true".to_owned());

        drop(filter.on_response(&mut ctx).await.unwrap());

        assert!(
            is_armed(&ctx),
            "filter should be armed when streaming request meets SSE response"
        );
        assert!(
            ctx.response_headers_modified,
            "response headers should be marked modified"
        );
    }

    #[tokio::test]
    async fn on_response_arms_with_charset_parameter() {
        let filter = make_filter();
        let req = crate::test_utils::make_request(http::Method::POST, "/v1/messages");
        let mut ctx = crate::test_utils::make_filter_context(&req);
        let mut resp = crate::test_utils::make_response();
        resp.headers.insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("text/event-stream; charset=utf-8"),
        );
        ctx.response_header = Some(&mut resp);
        ctx.set_metadata("anthropic_messages_to_chat_completions.streaming", "true".to_owned());

        drop(filter.on_response(&mut ctx).await.unwrap());

        assert!(
            is_armed(&ctx),
            "filter should arm even with charset parameter in Content-Type"
        );
    }

    #[tokio::test]
    async fn on_response_arms_with_mixed_case_content_type() {
        let filter = make_filter();
        let req = crate::test_utils::make_request(http::Method::POST, "/v1/messages");
        let mut ctx = crate::test_utils::make_filter_context(&req);
        let mut resp = crate::test_utils::make_response();
        resp.headers.insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("Text/Event-Stream"),
        );
        ctx.response_header = Some(&mut resp);
        ctx.set_metadata("anthropic_messages_to_chat_completions.streaming", "true".to_owned());

        drop(filter.on_response(&mut ctx).await.unwrap());

        assert!(
            is_armed(&ctx),
            "filter should arm with case-insensitive Content-Type matching"
        );
    }

    #[tokio::test]
    async fn on_response_does_not_arm_for_non_streaming_request() {
        let filter = make_filter();
        let req = crate::test_utils::make_request(http::Method::POST, "/v1/messages");
        let mut ctx = crate::test_utils::make_filter_context(&req);
        let mut resp = crate::test_utils::make_response();
        resp.headers.insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("text/event-stream"),
        );
        ctx.response_header = Some(&mut resp);

        drop(filter.on_response(&mut ctx).await.unwrap());

        assert!(!is_armed(&ctx), "filter should not arm without streaming metadata");
        assert!(
            !ctx.response_headers_modified,
            "response headers should not be modified for non-streaming request"
        );
    }

    #[tokio::test]
    async fn on_response_does_not_arm_for_non_sse_response() {
        let filter = make_filter();
        let req = crate::test_utils::make_request(http::Method::POST, "/v1/messages");
        let mut ctx = crate::test_utils::make_filter_context(&req);
        let mut resp = crate::test_utils::make_response();
        resp.headers.insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("application/json"),
        );
        ctx.response_header = Some(&mut resp);
        ctx.set_metadata("anthropic_messages_to_chat_completions.streaming", "true".to_owned());

        drop(filter.on_response(&mut ctx).await.unwrap());

        let content_type = ctx
            .response_header
            .as_ref()
            .unwrap()
            .headers
            .get(http::header::CONTENT_TYPE);
        assert_eq!(
            content_type,
            Some(&http::HeaderValue::from_static("application/json")),
            "non-SSE response should preserve original content type"
        );
        assert!(!is_armed(&ctx), "filter should not arm for non-SSE response");
    }

    #[tokio::test]
    async fn encoded_sse_response_passes_through_unchanged() {
        let filter = make_filter();
        let req = crate::test_utils::make_request(http::Method::POST, "/v1/messages");
        let mut ctx = crate::test_utils::make_filter_context(&req);
        let mut resp = crate::test_utils::make_response();
        resp.headers.insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("text/event-stream"),
        );
        resp.headers
            .insert(http::header::CONTENT_ENCODING, http::HeaderValue::from_static("gzip"));
        ctx.response_header = Some(&mut resp);
        ctx.set_metadata("anthropic_messages_to_chat_completions.streaming", "true".to_owned());

        drop(filter.on_response(&mut ctx).await.unwrap());

        assert!(!is_armed(&ctx), "filter should not arm for an encoded SSE response");
        assert!(
            !ctx.response_headers_modified,
            "encoded SSE response headers should remain unchanged"
        );

        let encoded = Bytes::from_static(b"\x1f\x8bencoded-sse");
        let mut body = Some(encoded.clone());
        drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());

        assert_eq!(body, Some(encoded), "encoded SSE bytes should pass through unchanged");
    }

    #[tokio::test]
    async fn on_response_arms_via_messages_format_metadata() {
        let filter = make_filter();
        let req = crate::test_utils::make_request(http::Method::POST, "/v1/messages");
        let mut ctx = crate::test_utils::make_filter_context(&req);
        let mut resp = crate::test_utils::make_response();
        resp.headers.insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("text/event-stream"),
        );
        ctx.response_header = Some(&mut resp);
        ctx.set_metadata("anthropic_messages_format.stream", "true".to_owned());

        drop(filter.on_response(&mut ctx).await.unwrap());

        assert!(
            is_armed(&ctx),
            "filter should arm via anthropic_messages_format.stream metadata"
        );
    }

    #[test]
    fn on_response_body_passes_through_when_not_armed() {
        let filter = make_filter();
        let req = crate::test_utils::make_request(http::Method::POST, "/v1/messages");
        let mut ctx = crate::test_utils::make_filter_context(Box::leak(Box::new(req)));

        let chunk =
            "data: {\"id\":\"c1\",\"model\":\"gpt-4\",\"choices\":[{\"delta\":{\"content\":\"Hi\"},\"index\":0}]}\n\n";
        let mut body = Some(Bytes::from(chunk));
        drop(filter.on_response_body(&mut ctx, &mut body, false).unwrap());

        let out = String::from_utf8(body.unwrap().to_vec()).unwrap();
        assert_eq!(out, chunk, "unarmed filter should pass through body unchanged");
    }

    #[test]
    fn tool_block_is_closed_at_done() {
        let (filter, mut ctx) = make_filter_and_context();

        let tool_start = "data: {\"id\":\"c1\",\"model\":\"gpt-4\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"id\":\"call_1\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"{}\"}}]},\"index\":0}]}\n\n";
        let mut body1 = Some(Bytes::from(tool_start));
        drop(filter.on_response_body(&mut ctx, &mut body1, false).unwrap());

        let done = "data: [DONE]\n\n";
        let mut body2 = Some(Bytes::from(done));
        drop(filter.on_response_body(&mut ctx, &mut body2, false).unwrap());

        let out = String::from_utf8(body2.unwrap().to_vec()).unwrap();
        assert!(
            out.contains("content_block_stop") && out.contains(r#""index":0"#),
            "DONE should close the open tool block"
        );
        assert!(out.contains("message_stop"), "DONE should still emit message_stop");
    }

    #[test]
    fn tool_call_delta_emits_tool_use_block_and_input_delta() {
        let (filter, mut ctx) = make_filter_and_context();

        let tool_start = "data: {\"id\":\"c1\",\"model\":\"gpt-4\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"id\":\"call_1\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\"}}]},\"index\":0}]}\n\n";
        let mut body = Some(Bytes::from(tool_start));
        drop(filter.on_response_body(&mut ctx, &mut body, false).unwrap());

        let out = String::from_utf8(body.unwrap().to_vec()).unwrap();
        assert!(
            out.contains("content_block_start") && out.contains(r#""type":"tool_use""#),
            "tool delta should start an Anthropic tool_use block"
        );
        assert!(
            out.contains(r#""id":"call_1""#) && out.contains(r#""name":"get_weather""#),
            "tool_use block should preserve id and function name"
        );
        let start = event_data(&out, "content_block_start");
        assert_eq!(
            start
                .get("content_block")
                .and_then(|block| block.get("caller"))
                .and_then(|caller| caller.get("type"))
                .and_then(Value::as_str),
            Some("direct"),
            "streaming tool_use blocks should identify a direct caller"
        );
        assert!(
            out.contains("input_json_delta") && out.contains(r#""partial_json":"{\"city\":"#),
            "tool arguments should stream as input_json_delta"
        );
    }
    const INVALID_TOOL_CALL_CASES: [(&str, &str, &str); 5] = [
        (
            "missing id",
            r#"{"index":0,"function":{"name":"get_weather","arguments":"{}"}}"#,
            "non-empty `id`",
        ),
        (
            "empty id",
            r#"{"index":0,"id":"","function":{"name":"get_weather","arguments":"{}"}}"#,
            "non-empty `id`",
        ),
        (
            "missing name",
            r#"{"index":0,"id":"call_1","function":{"arguments":"{}"}}"#,
            "non-empty function `name`",
        ),
        (
            "empty name",
            r#"{"index":0,"id":"call_1","function":{"name":"","arguments":"{}"}}"#,
            "non-empty function `name`",
        ),
        (
            "invalid id",
            r#"{"index":0,"id":"call.bad","function":{"name":"get_weather","arguments":"{}"}}"#,
            "must match ^[a-zA-Z0-9_-]+$",
        ),
    ];

    #[test]
    fn invalid_unopened_tool_call_deltas_emit_one_terminal_error() {
        for (description, tool_call, expected_error) in INVALID_TOOL_CALL_CASES {
            let (filter, mut ctx) = make_filter_and_context();
            let chunk = format!(
                "data: {{\"id\":\"c1\",\"model\":\"gpt-4\",\"choices\":[{{\"delta\":{{\"tool_calls\":[{tool_call}]}},\"index\":0}}]}}\n\n"
            );
            let mut body = Some(Bytes::from(chunk));
            drop(filter.on_response_body(&mut ctx, &mut body, false).unwrap());
            let output = String::from_utf8(body.unwrap().to_vec()).unwrap();
            assert!(
                output.contains("event: error")
                    && output.contains(r#""type":"api_error""#)
                    && output.contains("upstream response could not be transformed"),
                "{description} should emit a terminal Anthropic error event: {output}"
            );
            assert!(!output.contains(r#""type":"tool_use""#));
            assert!(!output.contains("event: message_stop"));

            let error = extract_tool_id_and_name(&serde_json::from_str(tool_call).unwrap()).unwrap_err();
            assert!(error.to_string().contains(expected_error));

            let mut trailing_body = Some(Bytes::from_static(b"data: [DONE]\n\n"));
            drop(filter.on_response_body(&mut ctx, &mut trailing_body, true).unwrap());
            assert_eq!(trailing_body, Some(Bytes::new()));
        }
    }


    #[test]
    fn interleaved_tool_call_argument_delta_uses_matching_tool_block_index() {
        let (filter, mut ctx) = make_filter_and_context();

        let call_0_start = "data: {\"id\":\"c1\",\"model\":\"gpt-4\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_0\",\"function\":{\"name\":\"first\",\"arguments\":\"{\\\"first\\\":\"}}]},\"index\":0}]}\n\n";
        let mut body1 = Some(Bytes::from(call_0_start));
        drop(filter.on_response_body(&mut ctx, &mut body1, false).unwrap());

        let call_1_start = "data: {\"id\":\"c1\",\"model\":\"gpt-4\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":1,\"id\":\"call_1\",\"function\":{\"name\":\"second\",\"arguments\":\"{\\\"second\\\":\"}}]},\"index\":0}]}\n\n";
        let mut body2 = Some(Bytes::from(call_1_start));
        drop(filter.on_response_body(&mut ctx, &mut body2, false).unwrap());

        let call_0_args = "data: {\"id\":\"c1\",\"model\":\"gpt-4\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"value\\\"}\"}}]},\"index\":0}]}\n\n";
        let mut body3 = Some(Bytes::from(call_0_args));
        drop(filter.on_response_body(&mut ctx, &mut body3, false).unwrap());

        let out = String::from_utf8(body3.unwrap().to_vec()).unwrap();
        let delta = event_data(&out, "content_block_delta");
        assert_eq!(
            delta.get("index").and_then(Value::as_u64),
            Some(0),
            "id-less argument delta for tool_call index 0 should use call_0's block"
        );
    }

    #[test]
    fn done_closes_all_open_tool_blocks() {
        let (filter, mut ctx) = make_filter_and_context();

        let call_0_start = "data: {\"id\":\"c1\",\"model\":\"gpt-4\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_0\",\"function\":{\"name\":\"first\",\"arguments\":\"{\\\"first\\\":\"}}]},\"index\":0}]}\n\n";
        let mut body1 = Some(Bytes::from(call_0_start));
        drop(filter.on_response_body(&mut ctx, &mut body1, false).unwrap());

        let call_1_start = "data: {\"id\":\"c1\",\"model\":\"gpt-4\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":1,\"id\":\"call_1\",\"function\":{\"name\":\"second\",\"arguments\":\"{\\\"second\\\":\"}}]},\"index\":0}]}\n\n";
        let mut body2 = Some(Bytes::from(call_1_start));
        drop(filter.on_response_body(&mut ctx, &mut body2, false).unwrap());

        let done = "data: [DONE]\n\n";
        let mut body3 = Some(Bytes::from(done));
        drop(filter.on_response_body(&mut ctx, &mut body3, false).unwrap());

        let out = String::from_utf8(body3.unwrap().to_vec()).unwrap();
        assert_eq!(
            event_indices(&out, "content_block_stop"),
            vec![0, 1],
            "DONE should close every open tool block in block-index order"
        );
    }

    #[tokio::test]
    async fn error_response_passes_through_unchanged() {
        let filter = make_filter();
        let (mut ctx, mut resp) = make_error_context(http::StatusCode::TOO_MANY_REQUESTS);
        resp.headers.insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("text/event-stream"),
        );
        ctx.response_header = Some(&mut resp);
        ctx.set_metadata("anthropic_messages_to_chat_completions.streaming", "true".to_owned());

        drop(filter.on_response(&mut ctx).await.unwrap());

        assert!(!is_armed(&ctx), "filter should not arm for error response");
        assert!(
            !ctx.response_headers_modified,
            "error response should not modify headers"
        );

        let error_body = r#"{"type":"error","error":{"type":"rate_limit_error","message":"Rate limited"}}"#;
        let mut body = Some(Bytes::from(error_body));
        drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());

        let out = String::from_utf8(body.unwrap().to_vec()).unwrap();
        assert_eq!(out, error_body, "error body should pass through unchanged");
    }

    #[test]
    fn unknown_config_field_rejected() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("max_body_bytes: 1048576").unwrap();
        let result = AnthropicMessagesToChatCompletionsStreamFilter::from_config(&yaml);

        assert!(
            result.is_err(),
            "streaming filter should reject unused buffer-size config"
        );
    }

    #[test]
    fn split_utf8_character_buffered_across_chunks() {
        let (filter, mut ctx) = make_filter_and_context();

        let mut chunk1 = Vec::new();
        chunk1.extend_from_slice(b"data: {\"id\":\"c1\",\"model\":\"gpt-4\",\"choices\":[{\"delta\":{\"content\":\"");
        chunk1.extend_from_slice(&[0xE2, 0x82]);
        let mut body1 = Some(Bytes::from(chunk1));
        drop(filter.on_response_body(&mut ctx, &mut body1, false).unwrap());
        assert!(
            body1.unwrap().is_empty(),
            "incomplete UTF-8 at chunk boundary should produce no output"
        );

        let mut chunk2 = vec![0xAC];
        chunk2.extend_from_slice(b"\"},\"index\":0}]}\n\n");
        let mut body2 = Some(Bytes::from(chunk2));
        drop(filter.on_response_body(&mut ctx, &mut body2, false).unwrap());
        let out = String::from_utf8(body2.unwrap().to_vec()).unwrap();
        assert!(out.contains("text_delta"), "completed UTF-8 should emit text_delta");
        assert!(out.contains('\u{20ac}'), "Euro sign should appear in the output");
    }

    #[test]
    fn invalid_utf8_rejected() {
        let (filter, mut ctx) = make_filter_and_context();

        let mut body1 = Some(Bytes::from(vec![0xFF]));
        let error = filter.on_response_body(&mut ctx, &mut body1, false).unwrap_err();
        assert!(
            error.to_string().contains("upstream SSE contains malformed UTF-8"),
            "malformed UTF-8 should fail the transformed stream"
        );
        assert!(
            !ctx.filter_metadata.contains_key(UTF8_BUFFER_KEY),
            "malformed UTF-8 should not be buffered as incomplete"
        );
    }

    #[test]
    fn invalid_utf8_after_stream_start_rejected() {
        let (filter, mut ctx) = make_filter_and_context();

        let chunk =
            "data: {\"id\":\"c1\",\"model\":\"gpt-4\",\"choices\":[{\"delta\":{\"content\":\"ok\"},\"index\":0}]}\n\n";
        let mut body1 = Some(Bytes::from(chunk));
        drop(filter.on_response_body(&mut ctx, &mut body1, false).unwrap());
        assert!(
            body1.unwrap().starts_with(b"event: message_start"),
            "setup chunk should start an Anthropic event stream"
        );

        let mut body2 = Some(Bytes::from(vec![0xFF]));
        let error = filter.on_response_body(&mut ctx, &mut body2, false).unwrap_err();
        assert!(
            error.to_string().contains("upstream SSE contains malformed UTF-8"),
            "malformed UTF-8 should fail after transformed events were emitted"
        );
    }

    #[test]
    fn truncated_utf8_at_end_of_stream_rejected() {
        let (filter, mut ctx) = make_filter_and_context();

        let mut body = Some(Bytes::from(vec![0xE2, 0x82]));
        let error = filter.on_response_body(&mut ctx, &mut body, true).unwrap_err();
        assert!(
            error.to_string().contains("upstream SSE contains malformed UTF-8"),
            "truncated final UTF-8 should fail the transformed stream"
        );
        assert!(
            !ctx.filter_metadata.contains_key(UTF8_BUFFER_KEY),
            "truncated final UTF-8 should not leave buffered bytes"
        );
    }

    #[test]
    fn pending_utf8_rejected_by_none_end_of_stream_body() {
        let (filter, mut ctx) = make_filter_and_context();

        let mut body1 = Some(Bytes::from(vec![0xE2]));
        drop(filter.on_response_body(&mut ctx, &mut body1, false).unwrap());
        assert!(
            body1.unwrap().is_empty(),
            "incomplete UTF-8 should wait for the next chunk"
        );

        let mut body2 = None;
        let error = filter.on_response_body(&mut ctx, &mut body2, true).unwrap_err();
        assert!(
            error.to_string().contains("upstream SSE contains malformed UTF-8"),
            "missing final body should reject pending incomplete UTF-8"
        );
        assert!(
            !ctx.filter_metadata.contains_key(UTF8_BUFFER_KEY),
            "rejected pending UTF-8 should clear the buffer"
        );
    }

    #[test]
    fn malformed_utf8_discards_partial_sse_buffer() {
        let (filter, mut ctx) = make_filter_and_context();

        let mut chunk1 = Vec::new();
        chunk1.extend_from_slice(b"data: {\"id\":\"c1\",\"choices\":[{\"delta\":{\"content\":\"");
        chunk1.extend_from_slice(&[0xE2]);
        let mut body1 = Some(Bytes::from(chunk1));
        drop(filter.on_response_body(&mut ctx, &mut body1, false).unwrap());
        assert!(body1.unwrap().is_empty(), "setup chunk should produce no output");
        assert_stream_buffers_present(&ctx, true);

        let mut body2 = Some(Bytes::from(vec![0xFF]));
        let error = filter.on_response_body(&mut ctx, &mut body2, false).unwrap_err();
        assert!(
            error.to_string().contains("upstream SSE contains malformed UTF-8"),
            "malformed UTF-8 should fail the transformed stream"
        );
        assert!(
            !ctx.filter_metadata.contains_key(LINE_BUFFER_KEY),
            "malformed UTF-8 should discard buffered SSE data"
        );
        assert_stream_buffers_present(&ctx, false);
    }

    #[test]
    fn crlf_event_boundaries_parsed() {
        let (filter, mut ctx) = make_filter_and_context();

        let chunk = "data: {\"id\":\"c1\",\"model\":\"gpt-4\",\"choices\":[{\"delta\":{\"content\":\"Hi\"},\"index\":0}]}\r\n\r\n";
        let mut body = Some(Bytes::from(chunk));
        drop(filter.on_response_body(&mut ctx, &mut body, false).unwrap());

        let out = String::from_utf8(body.unwrap().to_vec()).unwrap();
        assert!(out.contains("message_start"), "CRLF boundaries should be recognized");
        assert!(
            out.contains("text_delta"),
            "event data should parse through CRLF boundaries"
        );
    }

    #[test]
    fn crlf_split_across_chunks_not_false_boundary() {
        let (filter, mut ctx) = make_filter_and_context();

        let chunk1 = "data: {\"id\":\"c1\",\"model\":\"gpt-4\",\"choices\":[{\"delta\":{\"role\":\"assistant\"},\"index\":0}]}\r";
        let mut body1 = Some(Bytes::from(chunk1));
        drop(filter.on_response_body(&mut ctx, &mut body1, false).unwrap());
        assert!(
            body1.unwrap().is_empty(),
            "trailing CR should not prematurely complete an event"
        );

        let chunk2 = "\n\r\n";
        let mut body2 = Some(Bytes::from(chunk2));
        drop(filter.on_response_body(&mut ctx, &mut body2, false).unwrap());
        let out = String::from_utf8(body2.unwrap().to_vec()).unwrap();
        assert!(
            out.contains("message_start"),
            "completed CRLF-delimited event should produce output"
        );
    }

    #[test]
    fn data_without_space_after_colon_accepted() {
        let (filter, mut ctx) = make_filter_and_context();

        let chunk =
            "data:{\"id\":\"c1\",\"model\":\"gpt-4\",\"choices\":[{\"delta\":{\"content\":\"test\"},\"index\":0}]}\n\n";
        let mut body = Some(Bytes::from(chunk));
        drop(filter.on_response_body(&mut ctx, &mut body, false).unwrap());

        let out = String::from_utf8(body.unwrap().to_vec()).unwrap();
        assert!(
            out.contains("text_delta"),
            "data without space after colon should be accepted"
        );
        assert!(
            out.contains("test"),
            "content should be parsed from data: without space"
        );
    }

    #[test]
    fn multiline_data_fields_are_joined_before_json_parsing() {
        let (filter, mut ctx) = make_filter_and_context();

        let chunk = "data: {\"id\":\"c1\",\"model\":\"gpt-4\",\n\
                     data: \"choices\":[{\"delta\":{\"content\":\"Hi\"},\"index\":0}]}\n\n";
        let mut body = Some(Bytes::from(chunk));
        drop(filter.on_response_body(&mut ctx, &mut body, false).unwrap());

        let out = String::from_utf8(body.unwrap().to_vec()).unwrap();
        assert!(
            out.contains("text_delta"),
            "multi-line SSE data should be parsed as one JSON payload"
        );
        assert!(out.contains("Hi"), "joined JSON content should be transformed");
    }

    #[test]
    fn done_sentinel_without_space_after_colon() {
        let (filter, mut ctx) = make_filter_and_context();

        let chunk1 = "data:{\"id\":\"c1\",\"model\":\"gpt-4\",\"choices\":[{\"delta\":{\"content\":\"Hi\"},\"index\":0,\"finish_reason\":\"stop\"}]}\n\n";
        let mut body1 = Some(Bytes::from(chunk1));
        drop(filter.on_response_body(&mut ctx, &mut body1, false).unwrap());

        let done = "data:[DONE]\n\n";
        let mut body2 = Some(Bytes::from(done));
        drop(filter.on_response_body(&mut ctx, &mut body2, false).unwrap());

        let out = String::from_utf8(body2.unwrap().to_vec()).unwrap();
        assert!(
            out.contains("message_stop"),
            "DONE without space should complete stream"
        );
    }

    #[test]
    fn bare_data_field_before_done_fails_closed() {
        let (filter, mut ctx) = make_filter_and_context();

        let chunk1 = "data: {\"id\":\"c1\",\"model\":\"gpt-4\",\"choices\":[{\"delta\":{\"content\":\"Hi\"},\"index\":0,\"finish_reason\":\"stop\"}]}\n\n";
        let mut body1 = Some(Bytes::from(chunk1));
        drop(filter.on_response_body(&mut ctx, &mut body1, false).unwrap());

        // A leading bare `data` line joins as "\n[DONE]", which is neither the
        // sentinel nor valid JSON. It must fail closed rather than be silently
        // discarded or mistaken for the DONE sentinel.
        let done = "data\ndata: [DONE]\n\n";
        let mut body2 = Some(Bytes::from(done));
        let err = filter.on_response_body(&mut ctx, &mut body2, false).unwrap_err();
        assert!(
            err.to_string().contains("not valid JSON"),
            "a preceding empty data field should prevent DONE recognition and fail closed, got: {err}"
        );
    }

    #[test]
    fn malformed_json_data_event_fails_closed() {
        let (filter, mut ctx) = make_filter_and_context();

        let chunk1 =
            "data: {\"id\":\"c1\",\"model\":\"gpt-4\",\"choices\":[{\"delta\":{\"content\":\"Hi\"},\"index\":0}]}\n\n";
        let mut body1 = Some(Bytes::from(chunk1));
        drop(filter.on_response_body(&mut ctx, &mut body1, false).unwrap());

        // A complete data event that is neither [DONE] nor valid JSON must not
        // be silently discarded; any text, tool call, finish reason, or usage
        // it carried would otherwise disappear from a successful stream.
        let mut body2 = Some(Bytes::from("data: not-json\n\n"));
        let err = filter.on_response_body(&mut ctx, &mut body2, false).unwrap_err();
        assert!(
            err.to_string().contains("not valid JSON"),
            "malformed JSON data event should fail closed, got: {err}"
        );
    }

    #[test]
    fn malformed_first_event_then_done_fails_closed() {
        let (filter, mut ctx) = make_filter_and_context();

        // Reproduction from the finding: a malformed first event followed by
        // [DONE] must fail closed instead of emitting terminal events with no
        // preceding message_start.
        let mut body = Some(Bytes::from("data: not-json\n\ndata: [DONE]\n\n"));
        let err = filter.on_response_body(&mut ctx, &mut body, false).unwrap_err();
        assert!(
            err.to_string().contains("not valid JSON"),
            "malformed first event should fail the stream before [DONE], got: {err}"
        );
    }

    #[test]
    fn non_object_json_data_event_fails_closed() {
        let (filter, mut ctx) = make_filter_and_context();

        let chunk1 =
            "data: {\"id\":\"c1\",\"model\":\"gpt-4\",\"choices\":[{\"delta\":{\"content\":\"Hi\"},\"index\":0}]}\n\n";
        let mut body1 = Some(Bytes::from(chunk1));
        drop(filter.on_response_body(&mut ctx, &mut body1, false).unwrap());

        // Valid JSON that is not an object is not a Chat Completions chunk and
        // must fail closed rather than be silently ignored.
        let mut body2 = Some(Bytes::from("data: 123\n\n"));
        let err = filter.on_response_body(&mut ctx, &mut body2, false).unwrap_err();
        assert!(
            err.to_string().contains("not a JSON object"),
            "non-object JSON data event should fail closed, got: {err}"
        );
    }

    #[test]
    fn lone_done_synthesizes_message_start_before_terminal_events() {
        let (filter, mut ctx) = make_filter_and_context();

        // [DONE] with no preceding chunk must still yield a structurally valid
        // Anthropic stream: message_start precedes message_delta and message_stop.
        let mut body = Some(Bytes::from("data: [DONE]\n\n"));
        drop(filter.on_response_body(&mut ctx, &mut body, false).unwrap());

        let out = String::from_utf8(body.unwrap().to_vec()).unwrap();
        let start = out.find("event: message_start");
        let delta = out.find("event: message_delta");
        let stop = out.find("event: message_stop");
        assert!(
            matches!((start, delta, stop), (Some(s), Some(d), Some(t)) if s < d && d < t),
            "message_start must precede message_delta and message_stop, got: {out}"
        );
    }

    #[test]
    fn cr_only_done_at_end_of_stream_emits_stop() {
        let (filter, mut ctx) = make_filter_and_context();

        let chunk1 = "data: {\"id\":\"c1\",\"model\":\"gpt-4\",\"choices\":[{\"delta\":{\"content\":\"Hi\"},\"index\":0,\"finish_reason\":\"stop\"}]}\n\n";
        let mut body1 = Some(Bytes::from(chunk1));
        drop(filter.on_response_body(&mut ctx, &mut body1, false).unwrap());

        let done = "data:[DONE]\r\r";
        let mut body2 = Some(Bytes::from(done));
        drop(filter.on_response_body(&mut ctx, &mut body2, true).unwrap());

        let out = String::from_utf8(body2.unwrap().to_vec()).unwrap();
        assert!(
            out.contains("message_stop"),
            "CR-only DONE at end of stream should complete stream"
        );
    }

    #[test]
    fn deferred_cr_flushed_by_empty_end_of_stream_body() {
        let (filter, mut ctx) = make_filter_and_context();

        let done = "data:[DONE]\r\n\r";
        let mut body1 = Some(Bytes::from(done));
        drop(filter.on_response_body(&mut ctx, &mut body1, false).unwrap());
        assert!(
            body1.unwrap().is_empty(),
            "split CRLF delimiter should wait for final boundary"
        );

        let mut body2 = Some(Bytes::new());
        drop(filter.on_response_body(&mut ctx, &mut body2, true).unwrap());

        let out = String::from_utf8(body2.unwrap().to_vec()).unwrap();
        assert!(
            out.contains("message_stop"),
            "empty end-of-stream body should flush deferred CR delimiter"
        );
    }

    #[test]
    fn deferred_cr_flushed_by_none_end_of_stream_body() {
        let (filter, mut ctx) = make_filter_and_context();

        let done = "data:[DONE]\r\n\r";
        let mut body1 = Some(Bytes::from(done));
        drop(filter.on_response_body(&mut ctx, &mut body1, false).unwrap());

        let mut body2 = None;
        drop(filter.on_response_body(&mut ctx, &mut body2, true).unwrap());

        let out = String::from_utf8(body2.unwrap().to_vec()).unwrap();
        assert!(
            out.contains("message_stop"),
            "missing end-of-stream body should flush deferred CR delimiter"
        );
    }

    #[test]
    fn split_event_above_64k_uses_configured_default_limit() {
        let (filter, mut ctx) = make_filter_and_context();
        let content = "x".repeat(70_000);
        let chunk1 =
            format!("data: {{\"id\":\"c1\",\"model\":\"gpt-4\",\"choices\":[{{\"delta\":{{\"content\":\"{content}");
        let mut body1 = Some(Bytes::from(chunk1));
        drop(filter.on_response_body(&mut ctx, &mut body1, false).unwrap());
        assert!(
            body1.unwrap().is_empty(),
            "large split SSE event should be buffered until its delimiter arrives"
        );

        let chunk2 = "\"},\"index\":0}]}\n\n";
        let mut body2 = Some(Bytes::from(chunk2));
        drop(filter.on_response_body(&mut ctx, &mut body2, false).unwrap());

        let out = String::from_utf8(body2.unwrap().to_vec()).unwrap();
        assert!(
            out.contains("text_delta"),
            "valid split SSE event larger than 64 KiB should transform"
        );
    }

    #[test]
    fn configured_oversized_partial_event_rejected() {
        let (filter, mut ctx) = make_filter_and_context_from_yaml("max_partial_event_bytes: 32");
        let filler = "x".repeat(33);
        let chunk = format!("data: {filler}");
        let mut body = Some(Bytes::from(chunk));

        let result = filter.on_response_body(&mut ctx, &mut body, false);

        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("exceeds 32 bytes"),
            "oversized incomplete SSE event should mention the configured limit"
        );
        assert!(
            !ctx.filter_metadata.contains_key(LINE_BUFFER_KEY),
            "oversized incomplete SSE event should not remain buffered"
        );
    }

    #[test]
    fn normalize_line_endings_rewrites_crlf_and_standalone_cr() {
        assert_eq!(
            normalize_line_endings("a\r\nb\rc\n").as_ref(),
            "a\nb\nc\n",
            "CRLF and standalone CR must become LF"
        );
        assert!(
            matches!(normalize_line_endings("a\r\nb"), Cow::Owned(_)),
            "input containing CR must allocate a normalized String"
        );
    }

    #[test]
    fn combine_chunk_with_leftover_prefixes_current_chunk() {
        let combined = combine_chunk_with_leftover(Some("data: {"), "\"a\":1}\n\n");
        assert_eq!(
            combined.as_ref(),
            "data: {\"a\":1}\n\n",
            "leftover prefix must be prepended to the current chunk"
        );
        assert!(
            matches!(combined, Cow::Owned(_)),
            "non-empty leftover must own the combined buffer"
        );
    }

    #[test]
    fn emit_event_matches_legacy_to_string_format_bytes() {
        let data = serde_json::json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "hello"}
        });
        let mut output = Vec::new();
        emit_event(&mut output, "content_block_delta", &data);
        let data_str = serde_json::to_string(&data).unwrap_or_default();
        let expected = format!("event: content_block_delta\ndata: {data_str}\n\n");
        assert_eq!(
            output,
            expected.as_bytes(),
            "direct Vec writes must match to_string+format! SSE bytes"
        );
    }

    #[test]
    fn normalize_line_endings_lf_path_borrows_input() {
        let input = "data: {\"id\":\"c1\"}\n\n";
        let normalized = normalize_line_endings(input);
        assert!(matches!(normalized, Cow::Borrowed(_)), "LF-only input must be borrowed");
        assert!(
            std::ptr::eq(normalized.as_ptr(), input.as_ptr()),
            "borrowed LF-only input must retain the caller's buffer"
        );
    }

    #[test]
    fn combine_without_leftover_borrows_chunk() {
        let chunk = "data: {\"a\":1}\n\n";
        let combined = combine_chunk_with_leftover(None, chunk);
        assert!(
            matches!(combined, Cow::Borrowed(_)),
            "absent leftover must borrow the current chunk"
        );
        assert!(
            std::ptr::eq(combined.as_ptr(), chunk.as_ptr()),
            "absent leftover must not copy the current chunk"
        );
    }

    // =====================================================================
    // Allocation evidence (allocation-counter)
    // =====================================================================

    /// Pre-optimisation `emit_event` baseline.
    fn emit_event_legacy(output: &mut Vec<u8>, event_type: &str, data: &Value) {
        let data_str = serde_json::to_string(data).unwrap_or_default();
        output.extend_from_slice(format!("event: {event_type}\ndata: {data_str}\n\n").as_bytes());
    }

    /// Pre-optimisation `normalize_line_endings` baseline.
    fn normalize_line_endings_legacy(s: &str) -> String {
        s.replace("\r\n", "\n").replace('\r', "\n")
    }

    /// Pre-optimisation `combine_chunk_with_leftover` baseline.
    fn combine_chunk_with_leftover_legacy(leftover: Option<&str>, chunk_str: &str) -> String {
        match leftover {
            Some(prefix) if !prefix.is_empty() => format!("{prefix}{chunk_str}"),
            _ => chunk_str.to_owned(),
        }
    }

    fn assert_emit_allocates_less(event_type: &str, payload: &Value) {
        let capacity = serde_json::to_vec(payload).unwrap().len() + 64;
        let mut via_writer = Vec::with_capacity(capacity);
        let mut via_string = Vec::with_capacity(capacity);
        let writer = allocation_counter::measure(|| {
            emit_event(&mut via_writer, event_type, payload);
        });
        let string = allocation_counter::measure(|| {
            emit_event_legacy(&mut via_string, event_type, payload);
        });
        assert_eq!(
            via_writer, via_string,
            "{event_type} SSE bytes must stay equivalent while measuring allocations"
        );
        assert!(
            writer.bytes_total < string.bytes_total,
            "{event_type} to_writer must allocate fewer bytes than to_string: writer={} string={}",
            writer.bytes_total,
            string.bytes_total
        );
    }

    #[test]
    fn emit_event_allocates_less_than_legacy_to_string_format() {
        let small = serde_json::json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "hello"}
        });
        assert_emit_allocates_less("content_block_delta", &small);

        let large = serde_json::json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "x".repeat(4096)}
        });
        assert_emit_allocates_less("content_block_delta", &large);
    }

    #[test]
    fn normalize_line_endings_lf_only_allocates_zero() {
        let input = "event: content_block_delta\ndata: {\"index\":0}\n\n";
        let optimized = allocation_counter::measure(|| {
            std::hint::black_box(normalize_line_endings(input));
        });
        let legacy = allocation_counter::measure(|| {
            std::hint::black_box(normalize_line_endings_legacy(input));
        });
        assert_eq!(
            optimized.count_total, 0,
            "LF-only normalize must not allocate at all, got {} allocations",
            optimized.count_total
        );
        assert!(
            legacy.count_total > 0,
            "legacy normalize should allocate even for LF-only input"
        );
    }

    #[test]
    fn combine_chunk_without_leftover_allocates_zero() {
        let chunk = "event: message_start\ndata: {\"type\":\"message\"}\n\n";
        let optimized = allocation_counter::measure(|| {
            std::hint::black_box(combine_chunk_with_leftover(None, chunk));
        });
        let legacy = allocation_counter::measure(|| {
            std::hint::black_box(combine_chunk_with_leftover_legacy(None, chunk));
        });
        assert_eq!(
            optimized.count_total, 0,
            "combine without leftover must not allocate, got {} allocations",
            optimized.count_total
        );
        assert!(
            legacy.count_total > 0,
            "legacy combine should allocate via to_owned even without leftover"
        );
    }

    #[test]
    fn zero_partial_event_limit_rejected() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("max_partial_event_bytes: 0").unwrap();
        let result = AnthropicMessagesToChatCompletionsStreamFilter::from_config(&yaml);

        assert!(
            result.is_err(),
            "streaming filter should reject a zero partial event limit"
        );
    }

    #[test]
    fn exceeds_max_partial_event_limit_rejected() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("max_partial_event_bytes: 67108865").unwrap();
        let result = AnthropicMessagesToChatCompletionsStreamFilter::from_config(&yaml);

        assert!(
            result.is_err(),
            "streaming filter should reject a limit above MAX_JSON_BODY_BYTES"
        );
    }

    #[test]
    fn custom_max_tool_blocks_parses() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("max_tool_blocks: 5").unwrap();
        let result = AnthropicMessagesToChatCompletionsStreamFilter::from_config(&yaml);

        assert!(
            result.is_ok(),
            "streaming filter should accept a custom max_tool_blocks"
        );
    }

    #[test]
    fn zero_max_tool_blocks_rejected() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("max_tool_blocks: 0").unwrap();
        let result = AnthropicMessagesToChatCompletionsStreamFilter::from_config(&yaml);

        assert!(result.is_err(), "streaming filter should reject a zero max_tool_blocks");
    }

    #[test]
    fn exceeding_max_tool_blocks_fails_closed() {
        let (filter, mut ctx) = make_filter_and_context_from_yaml("max_tool_blocks: 2");

        let block = |index: u64| {
            format!(
                "data: {{\"id\":\"c1\",\"model\":\"gpt-4\",\"choices\":[{{\"delta\":{{\"tool_calls\":[{{\"index\":{index},\"id\":\"call_{index}\",\"function\":{{\"name\":\"f{index}\",\"arguments\":\"{{}}\"}}}}]}},\"index\":0}}]}}\n\n"
            )
        };

        let mut body0 = Some(Bytes::from(block(0)));
        drop(filter.on_response_body(&mut ctx, &mut body0, false).unwrap());

        let mut body1 = Some(Bytes::from(block(1)));
        drop(filter.on_response_body(&mut ctx, &mut body1, false).unwrap());

        let mut body2 = Some(Bytes::from(block(2)));
        let result = filter.on_response_body(&mut ctx, &mut body2, false);

        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("max_tool_blocks"),
            "exceeding the tool-block cap should fail closed and mention max_tool_blocks, got: {err}"
        );
    }

    // Test Utilities

    fn event_data(output: &str, event_type: &str) -> Value {
        let marker = format!("event: {event_type}\n");
        let block = output.split("\n\n").find(|block| block.starts_with(&marker)).unwrap();
        let data = block.lines().find_map(|line| line.strip_prefix("data: ")).unwrap();
        serde_json::from_str(data).unwrap()
    }

    fn event_indices(output: &str, event_type: &str) -> Vec<u64> {
        let marker = format!("event: {event_type}\n");
        output
            .split("\n\n")
            .filter(|block| block.starts_with(&marker))
            .filter_map(|block| block.lines().find_map(|line| line.strip_prefix("data: ")))
            .map(|data| serde_json::from_str::<Value>(data).unwrap())
            .map(|event| event.get("index").and_then(Value::as_u64).unwrap())
            .collect()
    }

    fn assert_null_fields(value: &Value, fields: &[&str], label: &str) {
        for field in fields {
            assert!(
                value.get(*field).is_some_and(Value::is_null),
                "{label} should include null {field}"
            );
        }
    }

    fn assert_u64_field(value: &Value, field: &str, expected: u64, label: &str) {
        assert_eq!(
            value.get(field).and_then(Value::as_u64),
            Some(expected),
            "{label} should include {field}"
        );
    }

    fn assert_stream_buffers_present(ctx: &HttpFilterContext<'_>, expected: bool) {
        assert_eq!(
            ctx.filter_metadata.contains_key(LINE_BUFFER_KEY),
            expected,
            "SSE line buffer presence should be {expected}"
        );
        assert_eq!(
            ctx.filter_metadata.contains_key(UTF8_BUFFER_KEY),
            expected,
            "UTF-8 buffer presence should be {expected}"
        );
    }

    fn make_filter() -> Box<dyn HttpFilter> {
        let yaml: serde_yaml::Value = serde_yaml::from_str("{}").unwrap();
        AnthropicMessagesToChatCompletionsStreamFilter::from_config(&yaml).unwrap()
    }

    fn make_filter_from_yaml(yaml: &str) -> Box<dyn HttpFilter> {
        let yaml: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
        AnthropicMessagesToChatCompletionsStreamFilter::from_config(&yaml).unwrap()
    }

    fn make_error_context(status: http::StatusCode) -> (HttpFilterContext<'static>, praxis_filter::Response) {
        let req = crate::test_utils::make_request(http::Method::POST, "/v1/messages");
        let mut resp = crate::test_utils::make_response();
        resp.status = status;
        resp.headers.insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("application/json"),
        );
        (crate::test_utils::make_filter_context(Box::leak(Box::new(req))), resp)
    }

    fn make_filter_and_context() -> (Box<dyn HttpFilter>, HttpFilterContext<'static>) {
        make_filter_and_context_from_yaml("{}")
    }

    fn make_filter_and_context_from_yaml(yaml: &str) -> (Box<dyn HttpFilter>, HttpFilterContext<'static>) {
        let req = crate::test_utils::make_request(http::Method::POST, "/v1/messages");
        let mut ctx = crate::test_utils::make_filter_context(Box::leak(Box::new(req)));
        ctx.set_metadata(ARMED_KEY, "true".to_owned());
        (make_filter_from_yaml(yaml), ctx)
    }
}
