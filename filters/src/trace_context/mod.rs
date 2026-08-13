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
//! One filter must come *earlier* still: the `request_id` core
//! builtin, when it is configured at all. That builtin reads only the
//! client's headers, so it generates a second, unrelated ID when it
//! runs after this filter. Pending header mutations are applied in
//! chain order with last-write-wins, so the forwarded request would
//! carry the builtin's ID while the delegated calls and the echoed
//! response header keep this filter's — exactly the split correlation
//! the filter exists to prevent. With `request_id` first, this filter
//! adopts the ID it injected and every leg agrees.
//!
//! The mismatch is detected at response time and logged, since a
//! filter cannot see what the chain places after it.
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
//! Running both filters is safe when `request_id` runs first; this
//! one then reuses whatever it injected.
//!
//! # YAML
//!
//! ```yaml
//! filter: trace_context
//! ```

use std::borrow::Cow;

use async_trait::async_trait;
use praxis_ai_apis::correlation::TraceContext;
use praxis_filter::{EmptyFilterConfig, FilterAction, FilterError, HttpFilter, HttpFilterContext, parse_filter_config};
use tracing::{debug, warn};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Header carrying the request correlation ID.
const REQUEST_ID: &str = "x-request-id";

// -----------------------------------------------------------------------------
// TraceContextFilter
// -----------------------------------------------------------------------------

/// Propagates correlation and W3C trace context to the upstream
/// request.
///
/// Register early: filters that make delegated calls read what this
/// injects. The one filter that must run even earlier is the
/// `request_id` core builtin, if configured — it reads only the
/// client's headers, so running it after this filter mints a second
/// ID that reaches the backend on the forwarded request while the
/// delegated calls keep the first.
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

    async fn on_response(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        // Every request filter has run by now, so this is the first
        // point at which a later filter's competing request ID is
        // visible.
        let Some(shared) = ctx.extensions.get::<TraceContext>() else {
            return Ok(FilterAction::Continue);
        };

        if let Some(forwarded) = competing_request_id(ctx, shared.request_id()) {
            warn!(
                correlated = %shared.request_id(),
                forwarded = %forwarded,
                "another filter injected a different x-request-id; the forwarded request and the \
                 delegated calls are in different correlation IDs. Register `request_id` before \
                 `trace_context`."
            );
        }

        Ok(FilterAction::Continue)
    }
}

// -----------------------------------------------------------------------------
// Utility Functions
// -----------------------------------------------------------------------------

/// Find a pending `x-request-id` that disagrees with the shared
/// context.
///
/// Pending mutations are applied last-write-wins, so the value that
/// actually reaches the backend is the final one. Returns `None` when
/// every pending value agrees with the correlated ID.
fn competing_request_id(ctx: &HttpFilterContext<'_>, correlated: &str) -> Option<String> {
    let forwarded = ctx
        .extra_request_headers
        .iter()
        .rfind(|(name, _)| name.eq_ignore_ascii_case(REQUEST_ID))
        .map(|(_, value)| value.clone())?;

    (forwarded != correlated).then_some(forwarded)
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

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
