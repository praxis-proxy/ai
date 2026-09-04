// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Configuration for Responses-to-Chat translation.

use praxis_filter::{FilterError, body::MAX_JSON_BODY_BYTES};
use serde::Deserialize;

use super::stream::StreamLimits;
use crate::openai::responses::body_limits::validate_size_limit;

/// Default SSE reassembly buffer ceiling in bytes.
const DEFAULT_MAX_SSE_BUFFER_BYTES: usize = 1 << 20;

/// Default cap on total emitted streaming events, including the terminal.
///
/// Chosen to equal the `openai_stream_events` accumulator's default `max_events`
/// (100,000). Because that filter counts every Responses event this converter
/// emits, keeping the converter's total-event ceiling at or below the
/// accumulator's default guarantees that a stream the client receives is never
/// silently dropped from the response store when both filters run at defaults.
const DEFAULT_MAX_STREAM_EVENTS: usize = 100_000;

/// Default per-tool-call argument byte ceiling.
const DEFAULT_MAX_TOOL_CALL_ARGUMENT_BYTES: usize = 1 << 20;

/// Default cap on the number of tool calls in one response.
const DEFAULT_MAX_TOOL_CALLS: usize = 512;

/// The `openai_stream_events` accumulator's default wall-clock timeout, in
/// seconds (`SseParserConfig::default().timeout`). Duplicated as a plain integer
/// because that `Duration` default is not const-accessible here; the
/// default-compatibility test keeps the two in the required relationship.
const ACCUMULATOR_DEFAULT_TIMEOUT_SECS: u64 = 300;

/// The `openai_stream_events` accumulator's default SSE reassembly buffer
/// ceiling in bytes (`SseParserConfig::default().max_buffer_bytes`, 10 MiB).
/// Duplicated here to size the emitted-frame ceiling strictly below it; the
/// default-compatibility test keeps the two in the required relationship.
const ACCUMULATOR_DEFAULT_BUFFER_BYTES: usize = 10_485_760;

/// Default ceiling on the complete encoded size of a single emitted Responses
/// SSE frame, in bytes (8 MiB).
///
/// Set *strictly* below the `openai_stream_events` accumulator's default SSE
/// reassembly buffer so that every frame this converter emits fits in the buffer
/// the accumulator reassembles it in. The accumulator buffers each frame whole
/// (`event: ...\ndata: ...\n\n`) before parsing it, so a frame exceeding that
/// buffer would abort a stream the client already received, silently dropping it
/// from the response store. Measuring and bounding the *complete* encoded frame
/// here — rather than the response body alone — keeps the converter from ever
/// emitting a frame the accumulator cannot reassemble, at whatever the two
/// filters are configured to. Derived as the accumulator's default buffer less a
/// 2 MiB margin (yielding 8 MiB) so the headroom for the per-frame framing
/// overhead is explicit and tracks the accumulator default if it ever changes.
const DEFAULT_MAX_EMITTED_SSE_FRAME_BYTES: usize = ACCUMULATOR_DEFAULT_BUFFER_BYTES - (2 << 20);

/// Minimum accepted value for `max_emitted_sse_frame_bytes`.
///
/// When an emitted frame exceeds the ceiling the converter falls back to a
/// constant-bounded minimal `response.failed` resource (no request data), whose
/// complete encoded frame is well under 1 `KiB`. The per-frame ceiling is
/// re-applied to that fallback, so a ceiling below the minimal frame would make
/// even the fail-closed terminal overflow — ending the stream with no terminal
/// event at all and silently defeating the guarantee that every stream
/// terminates with a schema-complete `response.failed`. Reject such a
/// configuration up front. The 4 `KiB` floor clears the minimal frame with room
/// to spare while still permitting tight per-frame ceilings for small responses.
const MIN_MAX_EMITTED_SSE_FRAME_BYTES: usize = 4096;

/// Default streaming timeout in seconds; `0` disables the guard.
///
/// Set *strictly* below the `openai_stream_events` accumulator's default timeout
/// so that, at defaults, the converter fails a stalled stream — and emits its
/// `response.failed` terminal — on a callback the accumulator will still parse.
/// Equal deadlines would race: the accumulator checks its own timeout *before*
/// parsing a callback's frames and, measuring sub-second elapsed time, reaches
/// the deadline no later than this converter (which measures in whole seconds),
/// so it could reject the very callback carrying the converter's terminal. The
/// 30-second margin also covers the converter's whole-second timing granularity.
const DEFAULT_STREAM_TIMEOUT_SECS: u64 = ACCUMULATOR_DEFAULT_TIMEOUT_SECS - 30;

/// Default cap on decoded SSE frames processed per response. Set well above the
/// event cap so legitimate streams (which interleave role-only, usage-only, and
/// keepalive-adjacent frames with content frames) never trip it, while still
/// bounding an upstream that floods no-op frames.
const DEFAULT_MAX_STREAM_FRAMES: usize = 2_000_000;

/// Bounded body configuration for the translation filter.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ResponsesToChatCompletionsConfig {
    /// Maximum size in bytes of the request or finite response body this
    /// filter *produces* when translating between Responses and Chat
    /// Completions wire formats, including the accumulated semantic content of
    /// a streaming response.
    ///
    /// Raw transport body size is governed by the pipeline's `body_limits`,
    /// not this field. This bounds only the translated body, which can grow
    /// larger than the raw input.
    #[serde(default = "default_max_rewritten_body_bytes")]
    pub max_rewritten_body_bytes: usize,
    /// Maximum bytes buffered by the streaming SSE frame parser.
    #[serde(default = "default_max_sse_buffer_bytes")]
    pub max_sse_buffer_bytes: usize,
    /// Maximum total number of Responses SSE events emitted per response,
    /// including the single terminal event. Downstream accumulators count events
    /// the same way, so keep this at or below their event budget.
    #[serde(default = "default_max_stream_events")]
    pub max_stream_events: usize,
    /// Maximum accumulated argument bytes for a single streamed tool call.
    #[serde(default = "default_max_tool_call_argument_bytes")]
    pub max_tool_call_argument_bytes: usize,
    /// Maximum number of distinct tool calls in one streamed response.
    #[serde(default = "default_max_tool_calls")]
    pub max_tool_calls: usize,
    /// Wall-clock streaming timeout in seconds; `0` disables the guard.
    #[serde(default = "default_stream_timeout_secs")]
    pub stream_timeout_secs: u64,
    /// Maximum number of decoded SSE frames processed per streamed response.
    #[serde(default = "default_max_stream_frames")]
    pub max_stream_frames: usize,
    /// Maximum complete encoded size, in bytes, of a single emitted Responses
    /// SSE frame. Keep *strictly* below the downstream accumulator's SSE buffer
    /// ceiling so every emitted frame fits the buffer it is reassembled in.
    #[serde(default = "default_max_emitted_sse_frame_bytes")]
    pub max_emitted_sse_frame_bytes: usize,
}

impl Default for ResponsesToChatCompletionsConfig {
    fn default() -> Self {
        Self {
            max_rewritten_body_bytes: MAX_JSON_BODY_BYTES,
            max_sse_buffer_bytes: DEFAULT_MAX_SSE_BUFFER_BYTES,
            max_stream_events: DEFAULT_MAX_STREAM_EVENTS,
            max_tool_call_argument_bytes: DEFAULT_MAX_TOOL_CALL_ARGUMENT_BYTES,
            max_tool_calls: DEFAULT_MAX_TOOL_CALLS,
            stream_timeout_secs: DEFAULT_STREAM_TIMEOUT_SECS,
            max_stream_frames: DEFAULT_MAX_STREAM_FRAMES,
            max_emitted_sse_frame_bytes: DEFAULT_MAX_EMITTED_SSE_FRAME_BYTES,
        }
    }
}

impl ResponsesToChatCompletionsConfig {
    /// Project the streaming-relevant limits for the SSE converter.
    pub(super) fn stream_limits(&self) -> StreamLimits {
        StreamLimits {
            max_sse_buffer_bytes: self.max_sse_buffer_bytes,
            max_stream_events: self.max_stream_events,
            max_tool_call_argument_bytes: self.max_tool_call_argument_bytes,
            max_tool_calls: self.max_tool_calls,
            stream_timeout_secs: self.stream_timeout_secs,
            max_body_bytes: self.max_rewritten_body_bytes,
            max_stream_frames: self.max_stream_frames,
            max_emitted_sse_frame_bytes: self.max_emitted_sse_frame_bytes,
        }
    }
}

/// Serde default for `max_rewritten_body_bytes`.
fn default_max_rewritten_body_bytes() -> usize {
    MAX_JSON_BODY_BYTES
}

/// Return the default SSE buffer ceiling.
fn default_max_sse_buffer_bytes() -> usize {
    DEFAULT_MAX_SSE_BUFFER_BYTES
}

/// Return the default streaming event cap.
fn default_max_stream_events() -> usize {
    DEFAULT_MAX_STREAM_EVENTS
}

/// Return the default per-tool-call argument ceiling.
fn default_max_tool_call_argument_bytes() -> usize {
    DEFAULT_MAX_TOOL_CALL_ARGUMENT_BYTES
}

/// Return the default tool-call count cap.
fn default_max_tool_calls() -> usize {
    DEFAULT_MAX_TOOL_CALLS
}

/// Return the default streaming timeout.
fn default_stream_timeout_secs() -> u64 {
    DEFAULT_STREAM_TIMEOUT_SECS
}

/// Return the default decoded-frame cap.
fn default_max_stream_frames() -> usize {
    DEFAULT_MAX_STREAM_FRAMES
}

/// Return the default emitted-frame size ceiling.
fn default_max_emitted_sse_frame_bytes() -> usize {
    DEFAULT_MAX_EMITTED_SSE_FRAME_BYTES
}

/// Validate the parsed filter configuration.
pub(super) fn build_config(
    config: ResponsesToChatCompletionsConfig,
) -> Result<ResponsesToChatCompletionsConfig, FilterError> {
    validate_size_limit(
        "responses_to_chat_completions",
        "max_rewritten_body_bytes",
        config.max_rewritten_body_bytes,
    )?;
    if config.max_sse_buffer_bytes == 0 {
        return Err("responses_to_chat_completions: max_sse_buffer_bytes must be greater than zero".into());
    }
    if config.max_stream_events == 0 {
        return Err("responses_to_chat_completions: max_stream_events must be greater than zero".into());
    }
    if config.max_tool_call_argument_bytes == 0 {
        return Err("responses_to_chat_completions: max_tool_call_argument_bytes must be greater than zero".into());
    }
    if config.max_tool_calls == 0 {
        return Err("responses_to_chat_completions: max_tool_calls must be greater than zero".into());
    }
    if config.max_stream_frames == 0 {
        return Err("responses_to_chat_completions: max_stream_frames must be greater than zero".into());
    }
    if config.max_emitted_sse_frame_bytes < MIN_MAX_EMITTED_SSE_FRAME_BYTES {
        return Err(format!(
            "responses_to_chat_completions: max_emitted_sse_frame_bytes ({}) must be at least {MIN_MAX_EMITTED_SSE_FRAME_BYTES} bytes so the fail-closed response.failed terminal always fits",
            config.max_emitted_sse_frame_bytes,
        )
        .into());
    }
    Ok(config)
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::too_many_lines, reason = "tests")]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::openai::sse::SseParserConfig;

    #[test]
    fn defaults_are_compatible_with_stream_events_accumulator() {
        // The converter emits the Responses SSE stream that `openai_stream_events`
        // accumulates for persistence. At bare defaults (neither filter configured),
        // the accumulator must never abort a stream the client already received, so
        // the converter's default event and timeout ceilings must be no looser than
        // the accumulator's own defaults.
        let converter = ResponsesToChatCompletionsConfig::default();
        let accumulator = SseParserConfig::default();

        // `max_stream_events` bounds the converter's *total* emitted events
        // (terminal included) and the accumulator counts events identically, so a
        // converter default at or below the accumulator's default guarantees an
        // accepted stream is never dropped from the store.
        assert!(
            converter.max_stream_events <= accumulator.max_events,
            "converter default max_stream_events ({}) must not exceed the accumulator default max_events ({})",
            converter.max_stream_events,
            accumulator.max_events,
        );

        // The converter must fail a stalled stream *strictly* before the
        // accumulator, so its default timeout must be enabled and shorter than the
        // accumulator's. Equal deadlines race: the accumulator checks its own
        // timeout before parsing a callback and, measuring sub-second elapsed time,
        // reaches the deadline no later than this converter (whole-second clock),
        // so it could reject the callback carrying the converter's terminal.
        assert_ne!(
            converter.stream_timeout_secs, 0,
            "the converter default timeout must be enabled so a stalled stream fails within the accumulator's window",
        );
        assert!(
            Duration::from_secs(converter.stream_timeout_secs) < accumulator.timeout,
            "converter default stream timeout ({}s) must be strictly shorter than the accumulator default timeout ({:?})",
            converter.stream_timeout_secs,
            accumulator.timeout,
        );
        // The margin must exceed the converter's whole-second timing granularity so
        // a stalled stream cannot fire on the same wall-clock second in both filters.
        assert!(
            accumulator
                .timeout
                .as_secs()
                .saturating_sub(converter.stream_timeout_secs)
                >= 2,
            "converter default timeout ({}s) must trail the accumulator default ({:?}) by more than its whole-second granularity",
            converter.stream_timeout_secs,
            accumulator.timeout,
        );

        // Every frame the converter emits is reassembled whole in the
        // accumulator's SSE buffer before parsing, so the converter's default
        // per-frame ceiling must sit *strictly* below the accumulator's default
        // buffer. A frame larger than that buffer would abort a stream the client
        // already received, silently dropping it from the store.
        assert!(
            converter.max_emitted_sse_frame_bytes < accumulator.max_buffer_bytes,
            "converter default max_emitted_sse_frame_bytes ({}) must be strictly below the accumulator default max_buffer_bytes ({})",
            converter.max_emitted_sse_frame_bytes,
            accumulator.max_buffer_bytes,
        );
        // The margin must leave room for the accumulator to hold one complete
        // frame including its `event: ...\ndata: ...\n\n` framing, not just the
        // payload the ceiling measures.
        assert!(
            accumulator
                .max_buffer_bytes
                .saturating_sub(converter.max_emitted_sse_frame_bytes)
                >= 1 << 20,
            "converter default max_emitted_sse_frame_bytes ({}) must trail the accumulator default max_buffer_bytes ({}) with framing headroom",
            converter.max_emitted_sse_frame_bytes,
            accumulator.max_buffer_bytes,
        );

        // The default config must still pass validation.
        build_config(ResponsesToChatCompletionsConfig::default()).expect("default config must be valid");
    }

    #[test]
    fn sub_minimal_emitted_frame_ceiling_is_rejected() {
        // A nonzero ceiling below the minimal `response.failed` frame would make
        // even the fail-closed fallback terminal overflow, leaving the stream with
        // no terminal event. build_config must reject anything below the floor,
        // while the floor itself must remain valid so tight-but-safe ceilings are
        // still allowed.
        let below_floor = ResponsesToChatCompletionsConfig {
            max_emitted_sse_frame_bytes: MIN_MAX_EMITTED_SSE_FRAME_BYTES - 1,
            ..ResponsesToChatCompletionsConfig::default()
        };
        assert!(
            build_config(below_floor).is_err(),
            "a per-frame ceiling below the minimum must be rejected",
        );

        let at_floor = ResponsesToChatCompletionsConfig {
            max_emitted_sse_frame_bytes: MIN_MAX_EMITTED_SSE_FRAME_BYTES,
            ..ResponsesToChatCompletionsConfig::default()
        };
        build_config(at_floor).expect("the minimum per-frame ceiling must be accepted");
    }
}
