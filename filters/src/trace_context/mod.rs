// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! W3C trace-context propagation for forwarded requests.
//!
//! Stamps `x-request-id` and `traceparent` onto the request sent
//! upstream, so the forwarded request joins the same trace as the
//! delegated calls made while handling it.
//!
//! Without this filter only delegated calls carry trace context,
//! which traces the file lookups but not the inference request they
//! belong to.
//!
//! # Placement
//!
//! Register early in the chain. Filters that make delegated calls
//! read the injected values, so anything running before this filter
//! resolves trace context independently and lands in a different
//! trace.
//!
//! # Behavior
//!
//! A valid inbound `traceparent` is continued: its trace-id and
//! flags carry forward under a fresh span-id for the upstream hop.
//! An absent or malformed one starts a new sampled trace — client-
//! supplied values are validated rather than forwarded, since they
//! would otherwise reach the telemetry backend unchecked.
//!
//! `x-request-id` follows the same precedence as the `request_id`
//! core builtin: client-supplied, then injected, then generated.
//! Running both filters is safe; this one reuses whatever
//! `request_id` injected.
//!
//! # YAML
//!
//! ```yaml
//! filter: trace_context
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

use std::borrow::Cow;

use async_trait::async_trait;
use praxis_ai_apis::correlation::TraceContext;
use praxis_filter::{EmptyFilterConfig, FilterAction, FilterError, HttpFilter, HttpFilterContext, parse_filter_config};
use tracing::debug;

/// Propagates correlation and W3C trace context to the upstream
/// request.
///
/// # YAML
///
/// ```yaml
/// filter: trace_context
/// ```
pub struct TraceContextFilter;

impl TraceContextFilter {
    /// Create a filter from parsed YAML config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the YAML config is invalid.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let _: EmptyFilterConfig = parse_filter_config("trace_context", config)?;
        Ok(Box::new(Self))
    }
}

#[async_trait]
impl HttpFilter for TraceContextFilter {
    fn name(&self) -> &'static str {
        "trace_context"
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        let context = TraceContext::get_or_init(ctx);
        let headers = context.headers_for_hop(ctx);

        for (name, value) in headers {
            // Replace any value this filter already injected, so a
            // re-entered chain does not accumulate duplicates. A
            // client-supplied header lives on ctx.request.headers
            // and is untouched here; the pipeline applies these
            // pending mutations over it.
            ctx.extra_request_headers
                .retain(|(existing, _)| !existing.eq_ignore_ascii_case(name.as_str()));
            ctx.extra_request_headers
                .push((Cow::Owned(name.as_str().to_owned()), value));
        }

        debug!(request_id = %context.request_id(), "propagating trace context upstream");

        Ok(FilterAction::Continue)
    }

    async fn on_response(&self, _ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        Ok(FilterAction::Continue)
    }
}
