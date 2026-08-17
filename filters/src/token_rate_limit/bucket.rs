// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Token bucket supporting multi-token reservation and post-hoc reconciliation.
//!
//! This is deliberately a separate, minimal implementation rather than a
//! reuse of [`praxis_filter`]'s internal single-token, lock-free bucket
//! (`praxis::filter::builtins::http::traffic_management::token_bucket`):
//! that type is `pub(crate)` to its own crate and not part of the public
//! API, and its `try_acquire` only ever consumes exactly one token per
//! call, which doesn't fit reserving an estimated N-token cost up front.
//! Mutex-based rather than lock-free CAS for now; revisit if this becomes
//! a hot path bottleneck (see `praxis`'s version for the lock-free pattern
//! to follow).

use std::sync::Mutex;

// -----------------------------------------------------------------------------
// TokenBucket
// -----------------------------------------------------------------------------

/// Per-rule bucket state: current tokens and the last refill timestamp.
struct BucketState {
    /// Current token count.
    tokens: f64,

    /// Last refill timestamp in nanoseconds since the filter's epoch.
    last_refill_nanos: u64,
}

/// Token bucket that supports reserving an arbitrary token amount up
/// front, and later reconciling that reservation against actual usage.
pub(super) struct TokenBucket {
    /// Guarded bucket state, refilled lazily on each access.
    state: Mutex<BucketState>,
}

impl TokenBucket {
    /// Create a bucket pre-filled with `burst` tokens.
    pub(super) fn new(burst: f64) -> Self {
        Self {
            state: Mutex::new(BucketState {
                tokens: burst,
                last_refill_nanos: 0,
            }),
        }
    }

    /// Refill `state` based on elapsed time, capped at `burst`.
    fn refill(state: &mut BucketState, rate: f64, burst: f64, now_nanos: u64) {
        let elapsed_nanos = now_nanos.saturating_sub(state.last_refill_nanos);
        if elapsed_nanos > 0 {
            let elapsed_secs = nanos_to_secs(elapsed_nanos);
            state.tokens = (state.tokens + elapsed_secs * rate).min(burst);
            state.last_refill_nanos = now_nanos;
        }
    }

    /// Try to reserve `amount` tokens up front (the estimated cost of an
    /// admitted request).
    ///
    /// Returns `Some(remaining)` on success. Returns `None` without
    /// consuming any tokens if `amount` exceeds what's currently
    /// available — reservation is all-or-nothing, never partial.
    #[expect(
        clippy::unwrap_used,
        reason = "poisoned mutex is unrecoverable; matches praxis's std::process::abort style"
    )]
    pub(super) fn try_reserve(&self, amount: f64, rate: f64, burst: f64, now_nanos: u64) -> Option<f64> {
        let mut state = self.state.lock().unwrap();
        Self::refill(&mut state, rate, burst, now_nanos);

        if state.tokens < amount {
            return None;
        }

        state.tokens -= amount;
        Some(state.tokens)
    }

    /// Reconcile a prior reservation against actual usage.
    ///
    /// `delta` is `reserved - actual`: positive releases unused tokens
    /// back to the bucket; negative draws additional tokens for an
    /// underestimate. The result is clamped to `[0, burst]` — an
    /// underestimate that exceeds the bucket's current balance is
    /// absorbed as a floor-at-zero shortfall rather than driven negative
    /// (overshoot handling beyond this is an open design question, see
    /// the `ai#658` review thread on reconciliation overshoot).
    #[expect(
        clippy::unwrap_used,
        reason = "poisoned mutex is unrecoverable; matches praxis's std::process::abort style"
    )]
    pub(super) fn reconcile(&self, delta: f64, rate: f64, burst: f64, now_nanos: u64) -> f64 {
        let mut state = self.state.lock().unwrap();
        Self::refill(&mut state, rate, burst, now_nanos);

        state.tokens = (state.tokens + delta).clamp(0.0, burst);
        state.tokens
    }

    /// Read current tokens without modification (for header reporting).
    #[expect(
        clippy::unwrap_used,
        reason = "poisoned mutex is unrecoverable; matches praxis's std::process::abort style"
    )]
    pub(super) fn current_tokens(&self, rate: f64, burst: f64, now_nanos: u64) -> f64 {
        let mut state = self.state.lock().unwrap();
        Self::refill(&mut state, rate, burst, now_nanos);
        state.tokens
    }
}

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------

/// Convert nanoseconds to seconds without `u64`-to-`f64` precision loss.
///
/// Splits into whole seconds (exact integer division) and a sub-second
/// remainder that fits within `f64`'s 53-bit mantissa, matching
/// `praxis`'s `token_bucket::nanos_to_secs` precision approach.
#[expect(
    clippy::cast_precision_loss,
    reason = "whole_secs max ~1.8e10 (u64::MAX nanos); well within f64's 2^53 mantissa. remainder < 1e9 is exact"
)]
fn nanos_to_secs(nanos: u64) -> f64 {
    let whole_secs = nanos / 1_000_000_000;
    let remainder = nanos % 1_000_000_000;
    whole_secs as f64 + remainder as f64 / 1_000_000_000.0
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, reason = "tests")]
mod tests {
    use super::*;

    #[test]
    fn reserve_succeeds_within_burst() {
        let bucket = TokenBucket::new(100.0);
        assert_eq!(bucket.try_reserve(40.0, 10.0, 100.0, 0), Some(60.0));
    }

    #[test]
    fn reserve_fails_when_insufficient_and_does_not_partially_consume() {
        let bucket = TokenBucket::new(10.0);
        assert!(bucket.try_reserve(15.0, 10.0, 10.0, 0).is_none());
        // Bucket must be untouched: a second reservation of the full burst still succeeds.
        assert_eq!(bucket.try_reserve(10.0, 10.0, 10.0, 0), Some(0.0));
    }

    #[test]
    fn reconcile_releases_unused_tokens() {
        let bucket = TokenBucket::new(100.0);
        bucket.try_reserve(50.0, 0.0, 100.0, 0);
        // Reserved 50, actual usage was only 30 -> release 20 back.
        let remaining = bucket.reconcile(20.0, 0.0, 100.0, 0);
        assert!((remaining - 70.0).abs() < 1e-9);
    }

    #[test]
    fn reconcile_draws_additional_tokens_for_underestimate() {
        let bucket = TokenBucket::new(100.0);
        bucket.try_reserve(50.0, 0.0, 100.0, 0);
        // Reserved 50, actual usage was 80 -> draw 30 more.
        let remaining = bucket.reconcile(-30.0, 0.0, 100.0, 0);
        assert!((remaining - 20.0).abs() < 1e-9);
    }

    #[test]
    fn reconcile_floors_at_zero_on_large_underestimate() {
        let bucket = TokenBucket::new(100.0);
        bucket.try_reserve(50.0, 0.0, 100.0, 0);
        // Reserved 50, actual usage was 500 -> would go deeply negative; floors at 0.
        let remaining = bucket.reconcile(-450.0, 0.0, 100.0, 0);
        assert_eq!(remaining, 0.0);
    }

    #[test]
    fn reconcile_caps_at_burst_on_large_overestimate() {
        let bucket = TokenBucket::new(100.0);
        bucket.try_reserve(10.0, 0.0, 100.0, 0);
        // Reserved 10, actual usage was 0 -> release 10, but bucket is already near-full.
        let remaining = bucket.reconcile(10.0, 0.0, 100.0, 0);
        assert_eq!(remaining, 100.0);
    }

    #[test]
    fn refills_over_time() {
        let bucket = TokenBucket::new(10.0);
        bucket.try_reserve(10.0, 10.0, 10.0, 0);
        assert!(
            bucket.try_reserve(1.0, 10.0, 10.0, 0).is_none(),
            "no tokens immediately after full reservation"
        );
        assert_eq!(
            bucket.try_reserve(2.0, 10.0, 10.0, 200_000_000),
            Some(0.0),
            "200ms at rate=10/s refills 2 tokens"
        );
    }

    #[test]
    fn current_tokens_is_read_only() {
        let bucket = TokenBucket::new(50.0);
        bucket.try_reserve(20.0, 0.0, 50.0, 0);
        assert!((bucket.current_tokens(0.0, 50.0, 0) - 30.0).abs() < 1e-9);
        // Reading twice must not change the balance.
        assert!((bucket.current_tokens(0.0, 50.0, 0) - 30.0).abs() < 1e-9);
    }
}
