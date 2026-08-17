// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Token-denominated rate limiting filter.
//!
//! Implements the uncontested MVP core of `ai#658`'s token rate limiting
//! proposal (`docs/proposals/00121_token-rate-limiting.md`, tracked by
//! epic `ai#121`): a single global token bucket, reservation-based
//! admission reconciled against actual provider-reported usage, and
//! standard 429 responses with token-denominated rate limit headers.
//!
//! Deliberately deferred, pending open design threads on `ai#658`:
//!
//! - **Bucket keys (M5)**: per-header/per-model/composite keys are still marked `TBD` in the proposal itself, and
//!   overlap with `ai#123`/`ai#129`/`praxis#189`+`praxis#232`. This filter only supports one global bucket for now, the
//!   same simplification `rate_limit`'s own `Global` mode makes.
//! - **Configurable estimation (M3)**: `estimate_tokens` is a fixed constant per rule for now, not derived from request
//!   metadata (e.g. `max_tokens`).
//! - **Token-type-aware accounting (M4)**: reconciles against `token.total` only; per-type (input/output/cached)
//!   weighting is not modeled yet.
//! - **Observability (M7/M8) and metering (S3)**: out of scope here — both are recommended to split into their own
//!   proposals on the `ai#658` review thread.
//!
//! Depends on `token_count` running earlier in the response phase to
//! populate `token.total` in `filter_metadata`; if that metadata is
//! absent when the response completes, the reservation is left as
//! final rather than guessed at.

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, reason = "tests")]
mod tests;

mod bucket;
mod config;

use std::time::Instant;

use async_trait::async_trait;
use bytes::Bytes;
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, Rejection, parse_filter_config,
};

use self::{bucket::TokenBucket, config::TokenRateLimitConfig};
use crate::token_usage::META_TOKEN_TOTAL;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Metadata key stashing this request's reserved token estimate, read
/// back during response-phase reconciliation.
const META_RESERVED: &str = "token_rate_limit.reserved";

/// Rate limit header: configured token budget.
///
/// Uses the `-Tokens` suffix per `ai#124`'s spec, distinct from the
/// existing `rate_limit` filter's unsuffixed `X-RateLimit-Limit`, to
/// avoid a header collision when both filters run in the same
/// pipeline (flagged on the `ai#658` review thread).
const HEADER_RATELIMIT_LIMIT_TOKENS: &str = "X-RateLimit-Limit-Tokens";

/// Rate limit header: remaining tokens in the current bucket.
const HEADER_RATELIMIT_REMAINING_TOKENS: &str = "X-RateLimit-Remaining-Tokens";

/// Rate limit header: Unix timestamp when the bucket fully refills.
///
/// Also `-Tokens`-suffixed (unlike `rate_limit`'s bare `X-RateLimit-Reset`)
/// — the sibling `rate_limit` filter emits that exact literal name
/// (`praxis::filter::builtins::http::traffic_management::rate_limit::limiter`),
/// so leaving this one unsuffixed would still collide when both filters
/// run in the same pipeline, defeating the point of suffixing the other
/// two headers below.
const HEADER_RATELIMIT_RESET: &str = "X-RateLimit-Reset-Tokens";

// -----------------------------------------------------------------------------
// TokenRateLimitFilter
// -----------------------------------------------------------------------------

/// Token-denominated rate limiter: reserves an estimated cost at
/// admission, reconciles against actual usage after the response
/// completes.
///
/// # YAML configuration
///
/// ```yaml
/// filter: token_rate_limit
/// rate: 1000            # tokens replenished per second
/// burst: 100000         # max bucket capacity, in tokens
/// estimate_tokens: 500  # fixed cost reserved per request at admission
/// ```
pub struct TokenRateLimitFilter {
    /// Single global bucket (per-key buckets are M5, not yet built).
    bucket: TokenBucket,

    /// Tokens replenished per second.
    rate: f64,

    /// Maximum bucket capacity, in tokens.
    burst: f64,

    /// Fixed token cost reserved at admission (M3 placeholder).
    estimate_tokens: f64,

    /// Pre-formatted burst value for the `X-RateLimit-Limit-Tokens` header.
    burst_string: String,

    /// Monotonic clock reference; all timestamps are offsets from this.
    epoch: Instant,
}

impl TokenRateLimitFilter {
    /// Create a filter from parsed YAML config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the YAML config is invalid, `rate` is
    /// not a positive finite number, `burst` is not positive, or
    /// `estimate_tokens` is negative or exceeds `burst` (which would
    /// make every request unadmittable).
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: TokenRateLimitConfig = parse_filter_config("token_rate_limit", config)?;

        if !cfg.rate.is_finite() || cfg.rate <= 0.0 {
            return Err("token_rate_limit: rate must be a finite number greater than 0".into());
        }
        if !cfg.burst.is_finite() || cfg.burst <= 0.0 {
            return Err("token_rate_limit: burst must be a finite number greater than 0".into());
        }
        if !cfg.estimate_tokens.is_finite() || cfg.estimate_tokens < 0.0 {
            return Err("token_rate_limit: estimate_tokens must be a finite number >= 0".into());
        }
        if cfg.estimate_tokens > cfg.burst {
            return Err("token_rate_limit: estimate_tokens must not exceed burst".into());
        }

        #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "burst fits u64")]
        let burst_string = (cfg.burst as u64).to_string();

        Ok(Box::new(Self {
            bucket: TokenBucket::new(cfg.burst),
            rate: cfg.rate,
            burst: cfg.burst,
            estimate_tokens: cfg.estimate_tokens,
            burst_string,
            epoch: Instant::now(),
        }))
    }

    /// Nanoseconds elapsed since this filter's epoch.
    #[expect(clippy::cast_possible_truncation, reason = "nanos fit u64")]
    fn now_nanos(&self) -> u64 {
        self.epoch.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64
    }

    /// Build rate limit headers and compute the retry-after value.
    ///
    /// Mirrors `rate_limit`'s `Retry-After` formula exactly (flagged as
    /// worth reusing on the `ai#658` review thread):
    /// `ceil((1 - remaining) / rate).max(1)`.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "token count truncation"
    )]
    fn rate_limit_headers(
        &self,
        remaining: f64,
        time_source: &dyn praxis_core::time::TimeSource,
    ) -> (Vec<(&'static str, String)>, u64) {
        let retry_secs = if remaining < 1.0 {
            ((1.0 - remaining) / self.rate).ceil().max(1.0) as u64
        } else {
            0
        };
        let now_unix = time_source.now().as_secs();
        let reset_unix = now_unix.saturating_add(retry_secs);
        let remaining_int = remaining.max(0.0) as u64;

        let headers = vec![
            (HEADER_RATELIMIT_LIMIT_TOKENS, self.burst_string.clone()),
            (HEADER_RATELIMIT_REMAINING_TOKENS, format!("{remaining_int}")),
            (HEADER_RATELIMIT_RESET, format!("{reset_unix}")),
        ];
        (headers, retry_secs)
    }

    /// Reconcile a prior reservation against actual usage, if known.
    ///
    /// No-ops if either the reservation or `token.total` metadata is
    /// absent — reservation stands as final rather than guessing.
    /// Overshoot beyond what the bucket can absorb floors at zero (see
    /// [`bucket::TokenBucket::reconcile`]); this is a known open
    /// question, not solved here.
    fn reconcile(&self, ctx: &HttpFilterContext<'_>) {
        let Some(reserved) = ctx.get_metadata(META_RESERVED).and_then(|v| v.parse::<f64>().ok()) else {
            return;
        };
        let Some(actual) = ctx.get_metadata(META_TOKEN_TOTAL).and_then(|v| v.parse::<f64>().ok()) else {
            tracing::trace!("token_rate_limit: no token.total metadata at end of stream, keeping reservation as final");
            return;
        };

        let now = self.now_nanos();
        let delta = reserved - actual;
        let remaining = self.bucket.reconcile(delta, self.rate, self.burst, now);
        tracing::debug!(
            reserved,
            actual,
            remaining,
            "token_rate_limit: reconciled reservation against actual usage"
        );
    }
}

#[async_trait]
impl HttpFilter for TokenRateLimitFilter {
    fn name(&self) -> &'static str {
        "token_rate_limit"
    }

    fn response_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    fn response_body_mode(&self) -> BodyMode {
        BodyMode::Stream
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        let now = self.now_nanos();

        if self
            .bucket
            .try_reserve(self.estimate_tokens, self.rate, self.burst, now)
            .is_some()
        {
            ctx.set_metadata(META_RESERVED, self.estimate_tokens.to_string());
            Ok(FilterAction::Continue)
        } else {
            tracing::info!(
                estimate = self.estimate_tokens,
                "token_rate_limit: rejecting request (429)"
            );
            let remaining = self.bucket.current_tokens(self.rate, self.burst, now);
            let (headers, retry_secs) = self.rate_limit_headers(remaining, ctx.time_source);

            let mut rejection = Rejection::status(429).with_header("Retry-After", format!("{retry_secs}"));
            for (name, value) in headers {
                rejection = rejection.with_header(name, value);
            }
            Ok(FilterAction::Reject(rejection))
        }
    }

    async fn on_response(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        // Reports the reservation-time snapshot: headers must be set
        // before the body streams, before actual usage is known, so
        // they can't reflect this response's own reconciliation yet.
        let now = self.now_nanos();
        let remaining = self.bucket.current_tokens(self.rate, self.burst, now);
        let (headers, _retry_secs) = self.rate_limit_headers(remaining, ctx.time_source);

        if let Some(ref mut resp) = ctx.response_header {
            for (name, value) in &headers {
                if let Ok(hv) = value.parse()
                    && let Ok(hn) = http::header::HeaderName::from_bytes(name.as_bytes())
                {
                    resp.headers.insert(hn, hv);
                    ctx.response_headers_modified = true;
                }
            }
        }

        Ok(FilterAction::Continue)
    }

    fn on_response_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        _body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if end_of_stream {
            self.reconcile(ctx);
        }
        Ok(FilterAction::Continue)
    }
}
