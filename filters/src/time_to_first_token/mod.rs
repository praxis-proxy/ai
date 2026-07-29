// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Time-to-first-token (TTFT) measurement filter.
//!
//! Records the elapsed time from request receipt to the first non-empty
//! SSE response body chunk as a Prometheus histogram labeled by model.
//! The filter is transparent: response bodies and status codes pass
//! through unchanged.
//!
//! # Metric
//!
//! `praxis_ai_ttft_seconds` — histogram with a `model` label.
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
    BodyAccess, EmptyFilterConfig, FilterAction, FilterError, HttpFilter, HttpFilterContext,
    builtins::http::value_safety::is_safe_promoted_value, parse_filter_config,
};
use tracing::debug;

/// Prometheus histogram name for time-to-first-token measurements.
const METRIC_TTFT_SECONDS: &str = "praxis_ai_ttft_seconds";

/// Maximum length of a body-derived value promoted to metric labels.
const MAX_PROMOTED_VALUE_LEN: usize = 256;

/// Metadata key indicating this request is an active TTFT candidate.
/// Present after `on_response` detects SSE; removed once TTFT is recorded.
const META_ACTIVE: &str = "time_to_first_token.active";

/// Measures time-to-first-token for streaming AI responses.
///
/// Activates only for successful `text/event-stream` responses. On the
/// first non-empty body chunk, records `ctx.request_start.elapsed()` as
/// a Prometheus histogram and deactivates for the remainder of the
/// response.
///
/// The histogram's `model` label is read from metadata set by an upstream format
/// filter (`openai_responses_format`, `anthropic_messages_format`, or
/// `anthropic_to_openai`). If no format filter runs before this filter,
/// all TTFT samples are labeled `unknown`.
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
            ctx.filter_metadata.insert(META_ACTIVE.to_owned(), String::new());
        }

        Ok(FilterAction::Continue)
    }

    fn on_response_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        _end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if !ctx.filter_metadata.contains_key(META_ACTIVE) {
            return Ok(FilterAction::Continue);
        }

        if body.as_ref().is_none_or(Bytes::is_empty) {
            return Ok(FilterAction::Continue);
        }

        let ttft = ctx.request_start.elapsed().as_secs_f64();
        let model = resolve_model(ctx);

        histogram!(METRIC_TTFT_SECONDS, "model" => model).record(ttft);
        ctx.filter_metadata.remove(META_ACTIVE);

        debug!(ttft, "recorded time-to-first-token");

        Ok(FilterAction::Continue)
    }
}

/// Resolve the model label from format-filter metadata with fallback.
///
/// Values containing control characters or exceeding the promotion length
/// cap are treated as unsafe and replaced with `"unknown"` to prevent
/// malformed Prometheus labels or cardinality pressure.
fn resolve_model(ctx: &HttpFilterContext<'_>) -> String {
    ctx.get_metadata("openai_responses_format.model")
        .or_else(|| ctx.get_metadata("anthropic_messages_format.model"))
        .or_else(|| ctx.get_metadata("anthropic_to_openai.model"))
        .filter(|v| v.len() <= MAX_PROMOTED_VALUE_LEN && is_safe_promoted_value(v))
        .unwrap_or("unknown")
        .to_owned()
}
