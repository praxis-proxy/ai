// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Time-to-first-token (TTFT) measurement filter.
//!
//! Records the server-side elapsed time from request receipt to the first
//! generated output delta as a Prometheus histogram labeled by `model` and
//! `cluster`. The event definition matches vLLM's TTFT (the first output
//! token chunk); vLLM measures it client-side, so values are not directly
//! comparable. The filter is transparent: response bodies and status codes
//! pass through unchanged.
//!
//! # Metric
//!
//! `praxis_ai_ttft_seconds` — histogram with `model` and `cluster` labels.
//! See `docs/llm-metrics.md` for vLLM parity and query examples.
//!
//! # YAML
//!
//! ```yaml
//! filter: time_to_first_token
//! ```

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
use metrics::histogram;
use praxis_ai_apis::is_event_stream_content_type;
use praxis_filter::{
    BodyAccess, EmptyFilterConfig, FilterAction, FilterError, HttpFilter, HttpFilterContext, parse_filter_config,
};
use tracing::debug;

use crate::{
    agentic::a2a::sse,
    llm_metrics::{cluster, resolve_model},
    llm_output_delta::payload_has_output_delta,
};

/// Prometheus histogram name for time-to-first-token measurements.
const METRIC_TTFT_SECONDS: &str = "praxis_ai_ttft_seconds";

/// Maximum bytes buffered while scanning the prefix of an SSE stream.
const MAX_SSE_SCRATCH_BYTES: usize = 1024 * 1024;

/// Per-request SSE scanner state used until the first output delta.
#[derive(Default)]
struct TtftState {
    /// Incremental SSE scanner state for the prefix of the stream.
    scan: sse::SseScanState,
}

/// Measures time-to-first-token for streaming AI responses.
///
/// Activates only for successful `text/event-stream` responses. It scans SSE
/// data payloads until the first generated output delta, then records
/// `ctx.request_start.elapsed()` once and deactivates. Role, metadata,
/// usage, and finish events do not trigger TTFT; SSE comments are ignored by
/// the scanner. The OpenAI Chat/Responses output-delta rules match
/// `vllm bench serve`.
///
/// The histogram's `model` label is read from metadata set by an upstream
/// request fact filter: `openai_responses_request` (Responses create bodies),
/// `openai_chat_completions_request` (native `POST /v1/chat/completions` bodies),
/// `anthropic_messages_request`, `anthropic_messages_to_chat_completions`
/// (Anthropic, including Anthropic-to-Chat translation), or the
/// `model_to_header`-published `model_to_header.model` value. If none runs
/// before this filter, all TTFT samples are labeled `unknown`.
///
/// The `cluster` label is the upstream cluster the request was routed to
/// (`HttpFilterContext::cluster_name`), falling back to `unknown`.
///
/// Distinct `model` label values are capped at 1024 per process lifetime;
/// values arriving after the cap are labeled `other` (see `llm_metrics`).
///
/// Chat traffic that originates as a Responses or Anthropic request and is
/// translated downstream keeps the model its originating classifier promoted.
///
/// # YAML
///
/// ```yaml
/// filter: time_to_first_token
/// ```
pub struct TimeToFirstTokenFilter;

impl TimeToFirstTokenFilter {
    /// Create a filter from parsed YAML config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the YAML config is invalid.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let _: EmptyFilterConfig = parse_filter_config("time_to_first_token", config)?;
        Ok(Box::new(Self))
    }
}

#[async_trait]
impl HttpFilter for TimeToFirstTokenFilter {
    fn name(&self) -> &'static str {
        "time_to_first_token"
    }

    fn response_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    async fn on_request(&self, _ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        Ok(FilterAction::Continue)
    }

    async fn on_response(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        ctx.extensions.remove::<TtftState>();

        let is_success = ctx.response_header.as_ref().is_some_and(|r| r.status.is_success());
        if !is_success {
            return Ok(FilterAction::Continue);
        }

        let is_sse = ctx
            .response_header
            .as_ref()
            .and_then(|r| r.headers.get("content-type"))
            .and_then(|v| v.to_str().ok())
            .is_some_and(is_event_stream_content_type);

        if is_sse {
            ctx.extensions.insert(TtftState::default());
        }

        Ok(FilterAction::Continue)
    }

    fn on_response_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        let Some(state) = ctx.extensions.get_mut::<TtftState>() else {
            return Ok(FilterAction::Continue);
        };

        let mut found = false;
        if let Some(chunk) = body.as_ref() {
            let result = sse::scan_sse_chunk(&mut state.scan, chunk, MAX_SSE_SCRATCH_BYTES);
            found = result.payloads.iter().any(|payload| payload_has_output_delta(payload));
        }

        if end_of_stream && !found {
            let mut pending = Vec::new();
            sse::flush_sse_state(&mut state.scan, &mut pending);
            found = pending.iter().any(|payload| payload_has_output_delta(payload));
        }

        if found {
            let ttft = ctx.request_start.elapsed().as_secs_f64();
            let model = resolve_model(ctx);
            let cluster = cluster(ctx);
            histogram!(METRIC_TTFT_SECONDS, "model" => model, "cluster" => cluster).record(ttft);
            ctx.extensions.remove::<TtftState>();
            debug!(ttft, "recorded time-to-first-token");
        } else if end_of_stream {
            ctx.extensions.remove::<TtftState>();
        }

        Ok(FilterAction::Continue)
    }
}
