// SPDX-License-Identifier: MIT
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

use std::fmt::Write as _;

use async_trait::async_trait;
use bytes::Bytes;
use praxis_ai_apis::token_usage::{TokenUsageProvider, set_token_usage};
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, parse_filter_config,
};
use serde::Deserialize;
use tracing::{debug, trace};

use crate::agentic::a2a::sse;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Default maximum bytes for non-streaming JSON body accumulation (1 MiB).
const DEFAULT_MAX_BODY_BYTES: usize = 1_048_576; // 1 MiB

/// Default maximum scratch bytes for SSE scanner state.
const DEFAULT_MAX_SCRATCH_BYTES: usize = 65_536; // 64 KiB

/// Metadata key prefix for all `token_count` working state.
const META_PREFIX: &str = "token_count.";

/// Metadata key for the extraction mode (`sse` or `json`).
const META_MODE: &str = "token_count.mode";

/// Metadata key for accumulated input tokens (streaming).
const META_INPUT: &str = "token_count.input";

/// Metadata key for accumulated output tokens (streaming).
const META_OUTPUT: &str = "token_count.output";

/// Metadata key for hex-encoded JSON body buffer (non-streaming).
const META_BUF_HEX: &str = "token_count.buf_hex";

/// Metadata key for byte count of buffered body (non-streaming).
const META_BUF_BYTES: &str = "token_count.buf_bytes";

/// Metadata key for SSE scanner line buffer.
const META_SSE_LINE_BUF: &str = "token_count.sse_line_buf_hex";

/// Metadata key for SSE scanner data buffer.
const META_SSE_DATA_HEX: &str = "token_count.sse_data_hex";

/// Metadata key for SSE scanner `has_data` flag.
const META_SSE_HAS_DATA: &str = "token_count.sse_has_data";

/// Metadata key for SSE scanner `prev_cr` flag.
const META_SSE_PREV_CR: &str = "token_count.sse_prev_cr";

/// Metadata key for SSE scanner scratch byte count.
const META_SSE_SCRATCH: &str = "token_count.sse_scratch_bytes";

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
}

/// AI provider selecting the token extraction strategy.
///
/// Distinct from [`TokenUsageProvider`] because Bedrock `InvokeModel` has no
/// body format to parse — its counts live in response headers — so it needs
/// a variant that [`TokenUsageProvider`] intentionally has no equivalent for.
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
    /// AWS Bedrock Converse API (JSON body / SSE, like other providers).
    Bedrock,
    /// AWS Bedrock `InvokeModel` API (token counts in response headers only).
    BedrockInvokeModel,
    /// Azure OpenAI (same JSON schema as OpenAI).
    Azure,
}

impl ProviderKind {
    /// Maps to the shared library provider used for body-based extraction.
    ///
    /// Returns `None` for `BedrockInvokeModel`, which has no body format
    /// and is instead handled directly via response headers.
    fn to_library_provider(self) -> Option<TokenUsageProvider> {
        match self {
            Self::OpenAi => Some(TokenUsageProvider::OpenAi),
            Self::Anthropic => Some(TokenUsageProvider::Anthropic),
            Self::Google => Some(TokenUsageProvider::Google),
            Self::Bedrock => Some(TokenUsageProvider::Bedrock),
            Self::Azure => Some(TokenUsageProvider::Azure),
            Self::BedrockInvokeModel => None,
        }
    }
}

impl From<TokenUsageProvider> for ProviderKind {
    fn from(provider: TokenUsageProvider) -> Self {
        match provider {
            TokenUsageProvider::OpenAi => Self::OpenAi,
            TokenUsageProvider::Anthropic => Self::Anthropic,
            TokenUsageProvider::Google => Self::Google,
            TokenUsageProvider::Bedrock => Self::Bedrock,
            TokenUsageProvider::Azure => Self::Azure,
        }
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
/// ```
///
/// [`filter_metadata`]: HttpFilterContext::filter_metadata
pub struct TokenCountFilter {
    /// Which provider's response format to parse.
    provider: ProviderKind,
}

impl TokenCountFilter {
    /// Create a filter from parsed YAML config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the YAML config is invalid.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: TokenCountConfig = parse_filter_config("token_count", config)?;

        Ok(Box::new(Self { provider: cfg.provider }))
    }
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
            ctx.filter_metadata.insert(META_MODE.to_owned(), "sse".to_owned());
            trace!("content-type is SSE, will extract tokens from stream events");
        } else if is_json_content_type(ct) {
            ctx.filter_metadata.insert(META_MODE.to_owned(), "json".to_owned());
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
        let Some(provider) = self.provider.to_library_provider() else {
            return Ok(FilterAction::Continue);
        };

        let mode = ctx.get_metadata(META_MODE).map(str::to_owned);

        match mode.as_deref() {
            Some("sse") => handle_sse_body(ctx, body, end_of_stream, &provider),
            Some("json") => handle_json_body(ctx, body, end_of_stream, &provider),
            _ => {},
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
    debug!(
        input,
        output, "extracted Bedrock InvokeModel token counts from response headers"
    );
}

// -----------------------------------------------------------------------------
// JSON (Non-Streaming) Path
// -----------------------------------------------------------------------------

/// Accumulate JSON body chunks and extract token usage on completion.
fn handle_json_body(
    ctx: &mut HttpFilterContext<'_>,
    body: &Option<Bytes>,
    end_of_stream: bool,
    provider: &TokenUsageProvider,
) {
    if let Some(chunk) = body.as_ref()
        && !accumulate_response_hex(ctx, chunk, DEFAULT_MAX_BODY_BYTES)
    {
        clear_all_metadata(ctx);
        return;
    }

    if end_of_stream {
        let bytes = ctx.filter_metadata.get(META_BUF_HEX).and_then(|hex| decode_hex(hex));

        if let Some(data) = bytes
            && let Some(usage) = provider.extract_token_usage(&data)
        {
            set_token_usage(
                ctx,
                usage.input_tokens(),
                usage.output_tokens(),
                Some(usage.total_tokens()),
            );
            debug!(
                input = usage.input_tokens(),
                output = usage.output_tokens(),
                total = usage.total_tokens(),
                "extracted token usage from JSON response"
            );
        }

        clear_all_metadata(ctx);
    }
}

// -----------------------------------------------------------------------------
// SSE (Streaming) Path
// -----------------------------------------------------------------------------

/// Process SSE body chunks and extract token usage incrementally.
fn handle_sse_body(
    ctx: &mut HttpFilterContext<'_>,
    body: &Option<Bytes>,
    end_of_stream: bool,
    provider: &TokenUsageProvider,
) {
    if let Some(chunk) = body.as_ref() {
        let mut state = load_sse_scan_state(ctx);

        let result = sse::scan_sse_chunk(&mut state, chunk, DEFAULT_MAX_SCRATCH_BYTES);

        for payload in &result.payloads {
            process_sse_payload(ctx, payload, provider);
        }

        if result.overflowed {
            debug!(
                scratch_bytes = state.scratch_bytes,
                "SSE scratch exceeds limit, finalizing token counts"
            );
            finalize_streaming_counts(ctx);
            clear_all_metadata(ctx);
            return;
        }

        save_sse_scan_state(ctx, &state);
    }

    if end_of_stream {
        finalize_streaming_counts(ctx);
        clear_all_metadata(ctx);
    }
}

/// Try to extract token usage from a single SSE data payload.
fn process_sse_payload(ctx: &mut HttpFilterContext<'_>, payload: &[u8], provider: &TokenUsageProvider) {
    if payload == b"[DONE]" {
        return;
    }

    if try_complete_usage(ctx, payload, provider) {
        return;
    }

    try_partial_usage(ctx, payload, provider);
}

/// Try complete usage extraction (OpenAI, Google, Azure final events).
fn try_complete_usage(ctx: &mut HttpFilterContext<'_>, payload: &[u8], provider: &TokenUsageProvider) -> bool {
    let Some(usage) = provider.extract_token_usage(payload) else {
        return false;
    };

    ctx.filter_metadata
        .insert(META_INPUT.to_owned(), usage.input_tokens().to_string());
    ctx.filter_metadata
        .insert(META_OUTPUT.to_owned(), usage.output_tokens().to_string());
    trace!(
        input = usage.input_tokens(),
        output = usage.output_tokens(),
        "complete token usage found in SSE event"
    );
    true
}

/// Try partial extraction (Anthropic, Bedrock streaming).
fn try_partial_usage(ctx: &mut HttpFilterContext<'_>, payload: &[u8], provider: &TokenUsageProvider) {
    let (input, output) = provider.extract_streaming_tokens(payload);

    if let Some(inp) = input {
        merge_accumulated_count(ctx, META_INPUT, inp);
        trace!(input = inp, "partial input tokens from SSE event");
    }
    if let Some(out) = output {
        merge_accumulated_count(ctx, META_OUTPUT, out);
        trace!(output = out, "partial output tokens from SSE event");
    }
}

/// Merge via max: correct for providers that report running totals
/// (latest value wins) and for those that report each count once
/// (single value preserved).
fn merge_accumulated_count(ctx: &mut HttpFilterContext<'_>, key: &str, value: u64) {
    let existing: u64 = ctx.filter_metadata.get(key).and_then(|v| v.parse().ok()).unwrap_or(0);

    let new_value = existing.max(value);
    ctx.filter_metadata.insert(key.to_owned(), new_value.to_string());
}

/// Write accumulated streaming counts to the well-known metadata keys.
fn finalize_streaming_counts(ctx: &mut HttpFilterContext<'_>) {
    let input: u64 = ctx
        .filter_metadata
        .get(META_INPUT)
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let output: u64 = ctx
        .filter_metadata
        .get(META_OUTPUT)
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    if input > 0 || output > 0 {
        set_token_usage(ctx, input, output, None);
        debug!(input, output, "finalized streaming token counts");
    }
}

// -----------------------------------------------------------------------------
// SSE Scanner State Persistence
// -----------------------------------------------------------------------------

/// Reconstructs scanner state from hex-encoded `filter_metadata` keys.
fn load_sse_scan_state(ctx: &HttpFilterContext<'_>) -> sse::SseScanState {
    let line_buf = ctx
        .filter_metadata
        .get(META_SSE_LINE_BUF)
        .and_then(|hex| decode_hex(hex))
        .unwrap_or_default();

    let data_buf = ctx
        .filter_metadata
        .get(META_SSE_DATA_HEX)
        .and_then(|hex| decode_hex(hex))
        .unwrap_or_default();

    let has_data = ctx.filter_metadata.get(META_SSE_HAS_DATA).is_some_and(|v| v == "true");

    let prev_cr = ctx.filter_metadata.get(META_SSE_PREV_CR).is_some_and(|v| v == "true");

    let scratch_bytes: usize = ctx
        .filter_metadata
        .get(META_SSE_SCRATCH)
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    sse::SseScanState {
        line_buf,
        data_buf,
        has_data,
        prev_cr,
        scratch_bytes,
    }
}

/// Persists scanner state back to `filter_metadata` for the next chunk.
fn save_sse_scan_state(ctx: &mut HttpFilterContext<'_>, state: &sse::SseScanState) {
    set_hex_metadata(ctx, META_SSE_LINE_BUF, &state.line_buf);
    set_hex_metadata(ctx, META_SSE_DATA_HEX, &state.data_buf);

    ctx.filter_metadata.insert(
        META_SSE_HAS_DATA.to_owned(),
        if state.has_data { "true" } else { "false" }.to_owned(),
    );
    ctx.filter_metadata.insert(
        META_SSE_PREV_CR.to_owned(),
        if state.prev_cr { "true" } else { "false" }.to_owned(),
    );
    ctx.filter_metadata
        .insert(META_SSE_SCRATCH.to_owned(), state.scratch_bytes.to_string());
}

// -----------------------------------------------------------------------------
// Hex Encoding Utilities
// -----------------------------------------------------------------------------

/// Accumulate raw bytes as hex to avoid corruption when chunk boundaries
/// split multibyte UTF-8 code points. Returns `false` if the byte limit
/// was exceeded.
fn accumulate_response_hex(ctx: &mut HttpFilterContext<'_>, chunk: &[u8], max_bytes: usize) -> bool {
    let existing_bytes: usize = ctx
        .filter_metadata
        .get(META_BUF_BYTES)
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    if existing_bytes.saturating_add(chunk.len()) > max_bytes {
        debug!(
            existing_bytes,
            chunk_len = chunk.len(),
            max_bytes,
            "response body exceeds token_count capture limit"
        );
        return false;
    }

    let hex_buf = ctx.filter_metadata.entry(META_BUF_HEX.to_owned()).or_default();
    for byte in chunk {
        _ = write!(hex_buf, "{byte:02x}");
    }

    let new_total = existing_bytes + chunk.len();
    ctx.filter_metadata
        .insert(META_BUF_BYTES.to_owned(), new_total.to_string());

    true
}

/// Inverse of the hex encoding.
fn decode_hex(hex: &str) -> Option<Vec<u8>> {
    if !hex.len().is_multiple_of(2) {
        return None;
    }

    hex.as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let hi = hex_digit(*pair.first()?)?;
            let lo = hex_digit(*pair.last()?)?;
            Some(hi << 4 | lo)
        })
        .collect()
}

/// Supports lowercase `a`-`f` only (our encoder writes lowercase).
fn hex_digit(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        _ => None,
    }
}

/// Hex-encodes raw bytes into a metadata value, or removes the key if empty.
fn set_hex_metadata(ctx: &mut HttpFilterContext<'_>, key: &str, data: &[u8]) {
    if data.is_empty() {
        ctx.filter_metadata.remove(key);
    } else {
        let mut hex = String::with_capacity(data.len() * 2);
        for byte in data {
            _ = write!(hex, "{byte:02x}");
        }
        ctx.filter_metadata.insert(key.to_owned(), hex);
    }
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

// -----------------------------------------------------------------------------
// Metadata Cleanup
// -----------------------------------------------------------------------------

/// Remove all `token_count.*` working state from `filter_metadata`.
fn clear_all_metadata(ctx: &mut HttpFilterContext<'_>) {
    ctx.filter_metadata.retain(|key, _| !key.starts_with(META_PREFIX));
}
