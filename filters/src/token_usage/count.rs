// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Token usage extraction filter for AI inference responses.
//!
//! Parses token counts from both streaming (SSE) and non-streaming (JSON)
//! AI provider responses and writes unified counts to [`filter_metadata`]
//! for downstream consumers. The filter is transparent: response bodies
//! and status codes pass through unchanged.
//!
//! Bedrock `InvokeModel` is the one exception: it reports token counts as
//! HTTP response headers rather than in the response body, so it is read
//! directly in `on_response` with no body buffering at all.
//!
//! [`filter_metadata`]: HttpFilterContext::filter_metadata

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests"
)]
mod tests;

use async_trait::async_trait;
use bytes::Bytes;
use praxis_ai_apis::json_body::MAX_JSON_BODY_BYTES;
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext,
    builtins::http::payload_processing::config_validation::validate_max_body_bytes, parse_filter_config,
};
use serde::Deserialize;
use tracing::{debug, trace};

use super::{
    StreamingTokens, TokenUsage,
    providers::{parse_anthropic, parse_bedrock, parse_google, parse_openai},
    set_cache_token_usage, set_reasoning_token_usage, set_token_status_overflow, set_token_usage, streaming,
};
use crate::agentic::a2a::sse;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Default maximum bytes for non-streaming JSON body accumulation (1 MiB).
const DEFAULT_MAX_BODY_BYTES: usize = 1_048_576; // 1 MiB

/// Default maximum scratch bytes for SSE scanner state.
///
/// Sized to hold the largest usage-carrying SSE event any provider emits:
/// the OpenAI Responses API `response.completed` event embeds the full
/// response object and grows with the output, so a 64 KiB default dropped
/// it under real agentic traffic — usage was lost and consumers that
/// ignore `token.status=overflow` booked zero spend. Matches
/// [`DEFAULT_MAX_BODY_BYTES`]: worst-case capture memory per in-flight
/// stream is 2 MiB either way.
const DEFAULT_MAX_SCRATCH_BYTES: usize = 1_048_576; // 1 MiB

/// Bedrock `InvokeModel` response header carrying the input token count.
const HEADER_BEDROCK_INPUT: &str = "x-amzn-bedrock-input-token-count";

/// Bedrock `InvokeModel` response header carrying the output token count.
const HEADER_BEDROCK_OUTPUT: &str = "x-amzn-bedrock-output-token-count";

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------

/// Filter configuration for `token_count`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TokenCountConfig {
    /// AI provider whose response format to parse.
    provider: ProviderKind,

    /// Maximum bytes to buffer for a non-streaming JSON response before
    /// giving up on locating its usage field. Must be greater than 0 and
    /// at most 64 MiB.
    #[serde(default = "default_max_body_bytes")]
    max_body_bytes: usize,

    /// Maximum scratch bytes (buffered line + in-progress event data) for
    /// the SSE scanner before an event is discarded as oversized. Must be
    /// greater than 0 and at most 64 MiB.
    #[serde(default = "default_max_scratch_bytes")]
    max_scratch_bytes: usize,
}

/// Default for [`TokenCountConfig::max_body_bytes`].
fn default_max_body_bytes() -> usize {
    DEFAULT_MAX_BODY_BYTES
}

/// Default for [`TokenCountConfig::max_scratch_bytes`].
fn default_max_scratch_bytes() -> usize {
    DEFAULT_MAX_SCRATCH_BYTES
}

/// Provider and transport shape selected by the `token_count` configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ProviderKind {
    /// OpenAI Chat Completions API.
    #[serde(rename = "openai")]
    OpenAi,
    /// Anthropic Claude API.
    Anthropic,
    /// Google Gemini API.
    Google,
    /// AWS Bedrock Converse API.
    Bedrock,
    /// AWS Bedrock `InvokeModel` token headers.
    BedrockInvokeModel,
    /// Azure OpenAI, which uses the OpenAI response schema.
    Azure,
}

impl ProviderKind {
    /// Extract complete usage from a JSON response or final SSE payload.
    fn extract_token_usage(self, body: &[u8]) -> Option<TokenUsage> {
        match self {
            Self::OpenAi | Self::Azure => parse_openai(body),
            Self::Anthropic => parse_anthropic(body),
            Self::Google => parse_google(body),
            Self::Bedrock => parse_bedrock(body),
            Self::BedrockInvokeModel => None,
        }
    }

    /// Extract partial usage from providers that split counts across events.
    fn extract_streaming_tokens(self, event_data: &[u8]) -> StreamingTokens {
        match self {
            Self::Anthropic => streaming::parse_anthropic_event(event_data),
            Self::Bedrock => streaming::parse_bedrock_event(event_data),
            Self::OpenAi | Self::Azure | Self::Google | Self::BedrockInvokeModel => StreamingTokens::default(),
        }
    }
}

// -----------------------------------------------------------------------------
// Capture State
// -----------------------------------------------------------------------------

/// What one `token_count` instance has captured of the current response.
///
/// `on_response` stores it with [`HttpFilterContext::insert_filter_state`]
/// when a success response has a supported content type; it is removed once
/// usage is published or capture is abandoned.
#[expect(
    clippy::large_enum_variant,
    reason = "filter_state boxes the capture once per response and it is only accessed by reference"
)]
enum CaptureState {
    /// `application/json`: the body is captured whole and parsed at
    /// end-of-stream, because usage may appear anywhere in it.
    Json {
        /// Body bytes captured so far, at most `max_body_bytes`.
        body: Vec<u8>,
    },

    /// `text/event-stream`: events are scanned as they arrive.
    Sse {
        /// Scanner state carried between body chunks.
        scanner: sse::SseScanState,

        /// Counts accumulated from the stream's usage-bearing events.
        counts: StreamCounts,
    },
}

/// Token counts accumulated from a stream's SSE events.
///
/// Each count is `None` until an event reports it.
#[derive(Default)]
struct StreamCounts {
    /// Input tokens, cached tokens included.
    input: Option<u64>,

    /// Output tokens.
    output: Option<u64>,

    /// Provider-supplied or parser-computed total from a complete-usage event.
    ///
    /// Kept so Google thoughts, which are not part of `token.output`, are not
    /// dropped when the total is published.
    total: Option<u64>,

    /// Input tokens served from the prompt cache.
    cache_read: Option<u64>,

    /// Input tokens written to the prompt cache.
    cache_write: Option<u64>,

    /// Reasoning / thinking tokens.
    reasoning: Option<u64>,
}

/// Stop capturing the current response, dropping anything captured so far.
fn clear_capture(ctx: &mut HttpFilterContext<'_>) {
    ctx.remove_filter_state::<CaptureState>();
}

/// Stop capturing the current response and return its captured JSON body,
/// which is empty when nothing was captured.
fn take_json_capture(ctx: &mut HttpFilterContext<'_>) -> Vec<u8> {
    match ctx.remove_filter_state::<CaptureState>() {
        Some(CaptureState::Json { body }) => body,
        Some(CaptureState::Sse { .. }) | None => Vec::new(),
    }
}

// -----------------------------------------------------------------------------
// TokenCountFilter
// -----------------------------------------------------------------------------

/// Extracts token usage from AI inference responses and writes unified
/// counts to [`filter_metadata`].
///
/// Supports both streaming (SSE) and non-streaming (JSON) responses across
/// five providers (OpenAI, Anthropic, Google, Bedrock Converse, Azure), plus
/// a header-only extraction path for Bedrock `InvokeModel`.
///
/// # YAML
///
/// ```yaml
/// filter: token_count
/// provider: openai   # openai | anthropic | google | bedrock | bedrock_invoke_model | azure
/// max_body_bytes: 1048576    # optional, JSON capture limit
/// max_scratch_bytes: 1048576 # optional, SSE per-event capture limit; must fit
///                            # the largest usage event (Responses-API
///                            # response.completed embeds the full response)
/// ```
///
/// [`filter_metadata`]: HttpFilterContext::filter_metadata
pub struct TokenCountFilter {
    /// Which provider's response format to parse.
    provider: ProviderKind,

    /// Maximum bytes to buffer for a non-streaming JSON response.
    max_body_bytes: usize,

    /// Maximum scratch bytes per SSE event before it is discarded.
    max_scratch_bytes: usize,
}

impl TokenCountFilter {
    /// Create a filter from parsed YAML config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the YAML config is invalid.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: TokenCountConfig = parse_filter_config("token_count", config)?;
        validate_max_body_bytes("token_count", cfg.max_body_bytes)?;
        validate_max_scratch_bytes(cfg.max_scratch_bytes)?;

        Ok(Box::new(Self {
            provider: cfg.provider,
            max_body_bytes: cfg.max_body_bytes,
            max_scratch_bytes: cfg.max_scratch_bytes,
        }))
    }
}

/// Reject a `max_scratch_bytes` that is zero or above the shared 64 MiB
/// capture ceiling used by other payload-processing filters.
fn validate_max_scratch_bytes(value: usize) -> Result<(), FilterError> {
    if value == 0 {
        return Err("token_count: 'max_scratch_bytes' must be greater than 0".into());
    }
    if value > MAX_JSON_BODY_BYTES {
        return Err(format!("token_count: max_scratch_bytes ({value}) exceeds maximum ({MAX_JSON_BODY_BYTES})").into());
    }
    Ok(())
}

#[async_trait]
impl HttpFilter for TokenCountFilter {
    fn name(&self) -> &'static str {
        "token_count"
    }

    fn response_body_access(&self) -> BodyAccess {
        if self.provider == ProviderKind::BedrockInvokeModel {
            BodyAccess::None
        } else {
            BodyAccess::ReadOnly
        }
    }

    fn response_body_mode(&self) -> BodyMode {
        BodyMode::Stream
    }

    async fn on_request(&self, _ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        Ok(FilterAction::Continue)
    }

    /// Skips extraction entirely for non-success statuses, for every
    /// provider. For `bedrock_invoke_model`, reads token counts directly
    /// from response headers since that provider has no body format to
    /// parse. For all other providers, detects the content-type to select
    /// the body extraction strategy used by `on_response_body`.
    async fn on_response(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        let is_success = ctx.response_header.as_ref().is_some_and(|r| r.status.is_success());

        if !is_success {
            trace!("non-success response, skipping token extraction");
            return Ok(FilterAction::Continue);
        }

        if self.provider == ProviderKind::BedrockInvokeModel {
            extract_bedrock_headers(ctx);
            return Ok(FilterAction::Continue);
        }

        let content_type = ctx
            .response_header
            .as_ref()
            .and_then(|r| r.headers.get("content-type"))
            .and_then(|v| v.to_str().ok());

        let Some(ct) = content_type else {
            return Ok(FilterAction::Continue);
        };

        if is_event_stream_content_type(ct) {
            ctx.insert_filter_state(CaptureState::Sse {
                scanner: sse::SseScanState::default(),
                counts: StreamCounts::default(),
            });
            trace!("content-type is SSE, will extract tokens from stream events");
        } else if is_json_content_type(ct) {
            ctx.insert_filter_state(CaptureState::Json { body: Vec::new() });
            trace!("content-type is JSON, will extract tokens from full body");
        }

        Ok(FilterAction::Continue)
    }

    fn on_response_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if self.provider == ProviderKind::BedrockInvokeModel {
            return Ok(FilterAction::Continue);
        }

        match ctx.get_filter_state::<CaptureState>() {
            Some(CaptureState::Sse { .. }) => {
                handle_sse_body(ctx, body, end_of_stream, self.provider, self.max_scratch_bytes);
            },
            Some(CaptureState::Json { .. }) => {
                handle_json_body(ctx, body, end_of_stream, self.provider, self.max_body_bytes);
            },
            None => {},
        }

        Ok(FilterAction::Continue)
    }
}

// -----------------------------------------------------------------------------
// Bedrock InvokeModel (Header-Only) Path
// -----------------------------------------------------------------------------

/// Reads Bedrock `InvokeModel` token counts from response headers; no-op if
/// either header is absent or unparseable. Unlike every other provider, no
/// body access is required for this path.
///
/// The response headers carry no prompt cache or reasoning breakdown, so
/// neither is recorded.
fn extract_bedrock_headers(ctx: &mut HttpFilterContext<'_>) {
    let input = ctx
        .response_header
        .as_ref()
        .and_then(|r| r.headers.get(HEADER_BEDROCK_INPUT))
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok());

    let output = ctx
        .response_header
        .as_ref()
        .and_then(|r| r.headers.get(HEADER_BEDROCK_OUTPUT))
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok());

    let (Some(input), Some(output)) = (input, output) else {
        trace!("Bedrock InvokeModel token headers not present or unparseable");
        return;
    };

    set_token_usage(ctx, input, output, None);
    crate::llm_metrics::record_token_usage(ctx, input, output, None, None);
    debug!(
        input,
        output, "extracted Bedrock InvokeModel token counts from response headers"
    );
}

// -----------------------------------------------------------------------------
// JSON (Non-Streaming) Path
// -----------------------------------------------------------------------------

/// Accumulate JSON body chunks and extract token usage on completion.
///
/// A response larger than `max_body_bytes` cannot be parsed for usage
/// (it may appear anywhere, including near the end), so extraction is
/// abandoned. Rather than leaving no signal at all, an explicit overflow
/// status is recorded so consumers do not mistake this for a genuine
/// zero-usage response.
fn handle_json_body(
    ctx: &mut HttpFilterContext<'_>,
    body: &Option<Bytes>,
    end_of_stream: bool,
    provider: ProviderKind,
    max_body_bytes: usize,
) {
    match (ctx.response_body_mode, end_of_stream, body.as_deref()) {
        // A buffered chain hands filters each raw chunk while it accumulates and
        // then re-delivers the frozen whole buffer at end-of-stream, so that body
        // must not be appended to the chunks already captured.
        (BodyMode::StreamBuffer { .. }, true, Some(eos_body)) => {
            let captured = take_json_capture(ctx);
            record_buffered_eos_usage(ctx, provider, eos_body, captured, max_body_bytes);
        },
        _ => accumulate_json_body(ctx, body, end_of_stream, provider, max_body_bytes),
    }
}

/// Captures each chunk as it arrives and parses the captured document once
/// the stream ends.
///
/// Serves plain `Stream` chains, buffered chains while they accumulate (the
/// captured chunks back the fallback in [`record_buffered_eos_usage`]), and
/// buffered chains whose end-of-stream call carries no body.
fn accumulate_json_body(
    ctx: &mut HttpFilterContext<'_>,
    body: &Option<Bytes>,
    end_of_stream: bool,
    provider: ProviderKind,
    max_body_bytes: usize,
) {
    if let Some(chunk) = body.as_ref()
        && !capture_json_chunk(ctx, chunk, max_body_bytes)
    {
        mark_json_overflow(ctx);
        clear_capture(ctx);
        return;
    }

    if end_of_stream {
        let captured = take_json_capture(ctx);
        record_json_usage(ctx, provider, &captured);
    }
}

/// Append a body chunk to the JSON capture. Returns `false`, leaving the
/// capture unchanged, if the chunk would take it past `max_bytes`, and `true`
/// without capturing anything when no JSON capture is active.
fn capture_json_chunk(ctx: &mut HttpFilterContext<'_>, chunk: &[u8], max_bytes: usize) -> bool {
    let Some(CaptureState::Json { body }) = ctx.get_filter_state_mut::<CaptureState>() else {
        return true;
    };

    if body.len().saturating_add(chunk.len()) > max_bytes {
        debug!(
            existing_bytes = body.len(),
            chunk_len = chunk.len(),
            max_bytes,
            "response body exceeds token_count capture limit"
        );
        return false;
    }

    body.extend_from_slice(chunk);
    true
}

/// Extracts usage from the body a buffered chain hands filters at
/// end-of-stream.
///
/// That body is normally the frozen whole response. After a filter returns
/// `FilterAction::Release`, though, Praxis core keeps reporting `StreamBuffer`
/// while delivering chunks the way `Stream` does (praxis-proxy/praxis#1270), so
/// the body may be only the final chunk. A body that yields no usage is
/// therefore retried as the captured chunks followed by that body, unless it
/// already starts with every captured byte: then it is the frozen buffer of a
/// response that reports no usage, and the concatenation could only fail to
/// parse or trip the capture limit. `captured` holds the chunks captured
/// before end-of-stream.
fn record_buffered_eos_usage(
    ctx: &mut HttpFilterContext<'_>,
    provider: ProviderKind,
    eos_body: &[u8],
    mut captured: Vec<u8>,
    max_body_bytes: usize,
) {
    if eos_body.len() > max_body_bytes {
        mark_json_overflow(ctx);
        return;
    }

    if record_json_usage(ctx, provider, eos_body) {
        return;
    }

    if captured.is_empty() {
        return;
    }

    if eos_body.starts_with(&captured) {
        trace!("buffered end-of-stream body repeats the captured chunks and reports no usage");
        return;
    }

    if captured.len().saturating_add(eos_body.len()) > max_body_bytes {
        mark_json_overflow(ctx);
        return;
    }

    captured.extend_from_slice(eos_body);
    record_json_usage(ctx, provider, &captured);
}

/// Parses JSON usage and records the normalized counts, prompt cache
/// breakdown, and reasoning tokens. Returns `false`, recording nothing, when
/// the body carries no recognizable usage.
fn record_json_usage(ctx: &mut HttpFilterContext<'_>, provider: ProviderKind, data: &[u8]) -> bool {
    let Some(usage) = provider.extract_token_usage(data) else {
        return false;
    };

    publish_token_usage(ctx, usage);
    debug!(
        input = usage.input_tokens(),
        output = usage.output_tokens(),
        total = usage.total_tokens(),
        cache_read = ?usage.cache_read_tokens(),
        cache_write = ?usage.cache_write_tokens(),
        reasoning = ?usage.reasoning_tokens(),
        "extracted token usage from JSON response"
    );
    true
}

/// Records the explicit overflow status for a JSON response that outgrew the
/// capture limit.
fn mark_json_overflow(ctx: &mut HttpFilterContext<'_>) {
    set_token_status_overflow(ctx);
    debug!("JSON response exceeded token_count capture limit, usage is unavailable");
}

/// Writes normalized totals plus any reported cache and reasoning breakdowns.
fn publish_token_usage(ctx: &mut HttpFilterContext<'_>, usage: TokenUsage) {
    set_token_usage(
        ctx,
        usage.input_tokens(),
        usage.output_tokens(),
        Some(usage.total_tokens()),
    );
    set_cache_token_usage(ctx, usage.cache_read_tokens(), usage.cache_write_tokens());
    set_reasoning_token_usage(ctx, usage.reasoning_tokens());
    crate::llm_metrics::record_token_usage(
        ctx,
        usage.input_tokens(),
        usage.output_tokens(),
        usage.cache_read_tokens(),
        usage.reasoning_tokens(),
    );
}

// -----------------------------------------------------------------------------
// SSE (Streaming) Path
// -----------------------------------------------------------------------------

/// Process SSE body chunks and extract token usage incrementally.
///
/// An oversized event is discarded by the scanner without aborting the
/// stream, so a terminal usage event arriving after it is still captured.
/// Final counts are only ever written at `end_of_stream`.
fn handle_sse_body(
    ctx: &mut HttpFilterContext<'_>,
    body: &Option<Bytes>,
    end_of_stream: bool,
    provider: ProviderKind,
    max_scratch_bytes: usize,
) {
    if let Some(chunk) = body.as_ref()
        && let Some(CaptureState::Sse { scanner, counts }) = ctx.get_filter_state_mut::<CaptureState>()
    {
        let result = sse::scan_sse_chunk(scanner, chunk, max_scratch_bytes);

        for payload in &result.payloads {
            counts.record_payload(payload, provider);
        }

        if result.dropped_events > 0 {
            debug!(
                dropped_events = result.dropped_events,
                "SSE event exceeded scratch limit, discarding and resuming at next event boundary"
            );
        }
    }

    if end_of_stream
        && let Some(CaptureState::Sse {
            mut scanner,
            mut counts,
        }) = ctx.remove_filter_state::<CaptureState>()
    {
        let mut pending = Vec::new();
        sse::flush_sse_state(&mut scanner, &mut pending);
        for payload in &pending {
            counts.record_payload(payload, provider);
        }

        finalize_streaming_counts(ctx, &counts, scanner.tail == sse::ScanTail::Dropped);
    }
}

impl StreamCounts {
    /// Fold one SSE `data:` payload into the counts.
    fn record_payload(&mut self, payload: &[u8], provider: ProviderKind) {
        if payload == b"[DONE]" {
            return;
        }

        if let Some(usage) = provider.extract_token_usage(payload) {
            self.record_complete_usage(usage);
            return;
        }

        self.merge_partial_usage(provider.extract_streaming_tokens(payload));
    }

    /// Record complete usage (OpenAI, Google, Azure final events).
    ///
    /// Cache and reasoning counts are replaced only when the event reports
    /// them, so an unreported count stays absent instead of being pinned to
    /// zero at finalization.
    fn record_complete_usage(&mut self, usage: TokenUsage) {
        self.input = Some(usage.input_tokens());
        self.output = Some(usage.output_tokens());
        self.total = Some(usage.total_tokens());
        self.cache_read = usage.cache_read_tokens().or(self.cache_read);
        self.cache_write = usage.cache_write_tokens().or(self.cache_write);
        self.reasoning = usage.reasoning_tokens().or(self.reasoning);

        trace!(
            input = usage.input_tokens(),
            output = usage.output_tokens(),
            total = usage.total_tokens(),
            cache_read = ?usage.cache_read_tokens(),
            cache_write = ?usage.cache_write_tokens(),
            reasoning = ?usage.reasoning_tokens(),
            "complete token usage found in SSE event"
        );
    }

    /// Merge partial counts (Anthropic, Bedrock streaming), skipping counts
    /// the event omitted.
    fn merge_partial_usage(&mut self, tokens: StreamingTokens) {
        merge_reported_count(&mut self.input, tokens.input, "input");
        merge_reported_count(&mut self.output, tokens.output, "output");
        merge_reported_count(&mut self.cache_read, tokens.cache_read, "cache_read");
        merge_reported_count(&mut self.cache_write, tokens.cache_write, "cache_write");
        merge_reported_count(&mut self.reasoning, tokens.reasoning, "reasoning");
    }
}

/// Merge one reported count into its accumulator via max: correct for
/// providers that report running totals (latest value wins) and for those
/// that report each count once (single value preserved). `count` names the
/// count in trace output.
fn merge_reported_count(accumulated: &mut Option<u64>, reported: Option<u64>, count: &'static str) {
    let Some(value) = reported else {
        return;
    };

    *accumulated = Some(accumulated.map_or(value, |existing| existing.max(value)));
    trace!(count, value, "partial token count from SSE event");
}

/// Write accumulated streaming counts to the well-known metadata keys.
///
/// If the stream ended after discarding an oversized event — including
/// the case where Anthropic/Bedrock partial counts were already stored
/// and the terminal usage event itself overflowed — an explicit overflow
/// status is recorded so this is distinguishable from a complete capture.
/// Recovered usage that arrived *after* an oversized event is treated as
/// authoritative and does not set overflow.
fn finalize_streaming_counts(ctx: &mut HttpFilterContext<'_>, counts: &StreamCounts, dropped_tail: bool) {
    let has_accumulated = counts.input.is_some() || counts.output.is_some();

    if dropped_tail {
        set_token_status_overflow(ctx);
        debug!(
            has_accumulated,
            "stream ended after an oversized SSE event; marking status overflow"
        );
    }

    if has_accumulated {
        publish_accumulated_streaming_usage(ctx, counts);
    }
}

/// Publishes streaming accumulators to the well-known `token.*` metadata keys.
fn publish_accumulated_streaming_usage(ctx: &mut HttpFilterContext<'_>, counts: &StreamCounts) {
    let input = counts.input.unwrap_or(0);
    let output = counts.output.unwrap_or(0);
    // Prefer the parser-computed total from a complete-usage event. Partial
    // streams (Anthropic / Bedrock) omit it, so TokenUsage::new falls back to
    // input + output, which is correct when reasoning is a subset of output.
    let usage = TokenUsage::new(input, output, counts.total)
        .with_cache(counts.cache_read, counts.cache_write)
        .with_reasoning(counts.reasoning);

    publish_token_usage(ctx, usage);
    debug!(
        input,
        output,
        total = usage.total_tokens(),
        cache_read = ?usage.cache_read_tokens(),
        cache_write = ?usage.cache_write_tokens(),
        reasoning = ?usage.reasoning_tokens(),
        "finalized streaming token counts"
    );
}

// -----------------------------------------------------------------------------
// Content-Type Helpers
// -----------------------------------------------------------------------------

/// Whether a content-type header value indicates `text/event-stream`.
fn is_event_stream_content_type(ct: &str) -> bool {
    ct.split(';')
        .next()
        .is_some_and(|media| media.trim().eq_ignore_ascii_case("text/event-stream"))
}

/// Whether a content-type header value indicates `application/json`.
fn is_json_content_type(ct: &str) -> bool {
    ct.split(';')
        .next()
        .is_some_and(|media| media.trim().eq_ignore_ascii_case("application/json"))
}
