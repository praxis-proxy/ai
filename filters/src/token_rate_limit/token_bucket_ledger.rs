// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! In-process token-bucket reservation ledger.
//!
//! Reuses, unmodified, the refill formula from Praxis's own lock-free
//! `praxis_filter::builtins::http::traffic_management::token_bucket`
//! (`tokens = (tokens + elapsed_secs * rate).min(capacity)`), extended
//! with the reserve/reconcile split this filter needs: that lock-free
//! bucket is a single atomic decrement with no way to credit tokens
//! back, so a reservation that overestimates its actual cost would
//! strand unused capacity until the next natural refill. Reconcile here
//! explicitly credits back `estimate - actual` (or debits the
//! shortfall) once actual provider-reported usage is known, mirroring
//! [`super::ledger::Ledger::reconcile`]'s refund/overage semantics for
//! the sliding-window algorithm.
//!
//! Structurally mirrors [`super::ledger::Ledger`] (per-key locking,
//! active-reservation bookkeeping, bounded keys/reservations) so both
//! algorithms get the same operational guarantees; only the "how much
//! is available right now" computation differs (refill-and-cap here vs.
//! sum-in-window there).

#![allow(
    missing_docs,
    clippy::missing_docs_in_private_items,
    clippy::too_many_lines,
    reason = "private ledger implementation is covered by its public filter contract and focused tests, matching \
              the sliding-window ledger's own module-level allow"
)]

use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
};

use dashmap::DashMap;

/// Upper bound, in seconds, on `capacity / refill_rate` -- the time to
/// fill an empty bucket from scratch.
///
/// [`super::backend::TOKEN_BUCKET_RESERVE_SCRIPT`]'s Valkey/Lua path
/// folds this ratio into a millisecond `PEXPIRE` TTL. Lua 5.1's `%.14g`
/// number formatting switches to scientific notation past ~1e14, which
/// `PEXPIRE`'s strict-integer parser rejects -- and since Redis doesn't
/// roll back a script's earlier `redis.call()`s on a later error, that
/// failure would permanently drain the bucket instead of just denying
/// one request. Enforced here, shared by both backends, so a config
/// rejected on one is rejected on both. 1e9 stays ~1e5x below the
/// threshold, with margin for `reservation_timeout_ms` on top.
pub(super) const MAX_CAPACITY_REFILL_RATE_RATIO_SECS: f64 = 1e9;

/// Upper bound on `capacity`/`reserved_tokens`, matching f64's 2^53
/// mantissa: every `as f64` cast site in this module (refill math,
/// deficit/overage arithmetic) assumes its input stays below this via
/// `#[expect(cast_precision_loss)]`. A `capacity` above it would
/// silently lose precision instead of erroring.
pub(super) const MAX_F64_SAFE_INTEGER: u64 = 1 << 53;

/// Bounds and parameters for a [`TokenBucketLedger`].
#[derive(Clone, Debug)]
pub(super) struct TokenBucketConfig {
    /// Maximum tokens held at once (the bucket's ceiling), per key.
    pub(super) capacity: u64,
    /// Tokens refilled per second, up to `capacity`.
    pub(super) refill_rate: f64,
    /// Time after which an ambiguous reservation is left charged at its
    /// estimate (tokens were already decremented at reserve time; this
    /// only bounds how long it's tracked as "active").
    pub(super) reservation_timeout_ms: u64,
    /// Maximum logical keys retained by the ledger.
    pub(super) max_keys: usize,
    /// Maximum key length retained by the ledger.
    pub(super) max_key_length: usize,
    /// Maximum active reservations retained by the ledger.
    pub(super) max_active_reservations: usize,
}

/// Validate `capacity`/`refill_rate` bounds shared by both the in-memory
/// and Valkey token-bucket backends, so a config that's rejected on one
/// is rejected identically on the other (see [`MAX_F64_SAFE_INTEGER`]
/// and [`MAX_CAPACITY_REFILL_RATE_RATIO_SECS`] for why each bound
/// exists).
pub(super) fn validate_capacity_and_refill_rate(capacity: u64, refill_rate: f64) -> Result<(), String> {
    if capacity == 0 {
        return Err("capacity must be positive".into());
    }
    if capacity > MAX_F64_SAFE_INTEGER {
        return Err(format!("capacity must not exceed {MAX_F64_SAFE_INTEGER} (2^53)"));
    }
    if !refill_rate.is_finite() || refill_rate <= 0.0 {
        return Err("refill_rate must be a positive, finite number".into());
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "capacity is already checked above to not exceed f64's 2^53 mantissa"
    )]
    let capacity_f64 = capacity as f64;
    if capacity_f64 / refill_rate > MAX_CAPACITY_REFILL_RATE_RATIO_SECS {
        return Err(format!(
            "capacity / refill_rate must not exceed {MAX_CAPACITY_REFILL_RATE_RATIO_SECS} seconds (time to fill \
             an empty bucket)"
        ));
    }
    Ok(())
}

impl TokenBucketConfig {
    /// Validate configuration before constructing a ledger.
    pub(super) fn validate(&self) -> Result<(), String> {
        validate_capacity_and_refill_rate(self.capacity, self.refill_rate)?;
        if self.reservation_timeout_ms == 0 {
            return Err("reservation timeout must be positive".into());
        }
        if self.max_keys == 0 || self.max_key_length == 0 || self.max_active_reservations == 0 {
            return Err("ledger bounds must be positive".into());
        }
        Ok(())
    }
}

/// A reservation admitted against one key's bucket.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Reservation {
    /// Opaque identifier used for idempotent reconciliation.
    pub(super) id: u64,
    /// Estimated token cost reserved (and already decremented) at
    /// admission.
    pub(super) estimate: u64,
    /// Monotonic timestamp at admission, in milliseconds.
    pub(super) created_at_ms: u64,
}

/// Result of attempting admission.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Decision {
    /// Request may proceed with this reservation.
    Admitted(Reservation),
    /// Request must be rejected before routing.
    Denied {
        /// Conservative delay before the bucket refills enough to admit
        /// the same estimate.
        retry_after_ms: u64,
    },
}

/// Result of reconciling a reservation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Settlement {
    /// Actual usage was applied exactly once.
    Applied {
        /// Actual tokens charged to the bucket.
        actual: u64,
        /// Estimate credited back to the bucket.
        refund: u64,
        /// Usage above the estimate, debited from the bucket.
        overage: u64,
    },
    /// The reservation was already reconciled or conservatively expired.
    Noop,
}

#[derive(Debug)]
struct ActiveReservation {
    estimate: u64,
    created_at_ms: u64,
}

#[derive(Debug)]
struct BucketState {
    /// Current tokens, refilled lazily on access (same lazy-refill
    /// pattern as the core lock-free `TokenBucket`).
    tokens: f64,
    last_refill_ms: u64,
    active: HashMap<u64, ActiveReservation>,
}

impl BucketState {
    fn new(capacity: u64) -> Self {
        Self {
            #[expect(
                clippy::cast_precision_loss,
                reason = "token capacities are far below f64's 2^53 mantissa"
            )]
            tokens: capacity as f64,
            last_refill_ms: 0,
            active: HashMap::new(),
        }
    }

    /// Refill tokens for elapsed time, capped at `capacity`. Mirrors
    /// `praxis_filter::builtins::http::traffic_management::token_bucket::TokenBucket::try_acquire`'s
    /// refill formula exactly.
    fn refill(&mut self, now_ms: u64, config: &TokenBucketConfig) {
        let elapsed_ms = now_ms.saturating_sub(self.last_refill_ms);
        if elapsed_ms > 0 {
            #[expect(
                clippy::cast_precision_loss,
                reason = "elapsed_ms/capacity are far below f64's 2^53 mantissa for any realistic reservation window"
            )]
            {
                let elapsed_secs = elapsed_ms as f64 / 1000.0;
                self.tokens = (self.tokens + elapsed_secs * config.refill_rate).min(config.capacity as f64);
            }
            self.last_refill_ms = now_ms;
        }
    }

    /// Drop active reservations that exceeded `reservation_timeout_ms`
    /// without being reconciled.
    ///
    /// Unlike the sliding-window ledger, no further charge happens here
    /// -- the estimate was already decremented from the bucket at
    /// reserve time (immediate-decrement design), so an abandoned
    /// reservation is already charged. This only stops it from
    /// permanently occupying the active-reservation bookkeeping.
    fn reap(&mut self, now_ms: u64, config: &TokenBucketConfig) -> Vec<u64> {
        let expired: Vec<u64> = self
            .active
            .iter()
            .filter_map(|(id, reservation)| {
                (now_ms.saturating_sub(reservation.created_at_ms) >= config.reservation_timeout_ms).then_some(*id)
            })
            .collect();
        for id in &expired {
            self.active.remove(id);
        }
        expired
    }

    /// Whether this key's state can be safely forgotten: no reservations
    /// pending reconciliation, *and* fully refilled back to capacity.
    ///
    /// Unlike the sliding-window ledger (where a settled entry simply
    /// ages out of the window and becomes irrelevant), a token bucket's
    /// balance never "expires" -- it only recovers via refill over time.
    /// Evicting a key while it's still short of `capacity` would
    /// silently grant it a full, unrecovered bucket the next time it's
    /// touched (a fresh key always starts full) -- effectively free
    /// bonus tokens. Only a fully-recovered, reservation-free key is
    /// truly equivalent to "not tracked at all".
    fn is_empty(&self, capacity: u64) -> bool {
        #[expect(
            clippy::cast_precision_loss,
            reason = "token capacities are far below f64's 2^53 mantissa"
        )]
        let capacity = capacity as f64;
        self.active.is_empty() && self.tokens >= capacity
    }
}

/// Thread-safe exact local token-bucket ledger with independent locks
/// per key.
pub(super) struct TokenBucketLedger {
    config: TokenBucketConfig,
    keys: DashMap<String, Arc<Mutex<BucketState>>>,
    reservations: DashMap<u64, String>,
    next_id: AtomicU64,
    key_count: AtomicUsize,
    active_reservations: AtomicUsize,
}

impl TokenBucketLedger {
    /// Construct a validated empty ledger.
    pub(super) fn new(config: TokenBucketConfig) -> Result<Self, String> {
        config.validate()?;
        Ok(Self {
            config,
            keys: DashMap::new(),
            reservations: DashMap::new(),
            next_id: AtomicU64::new(1),
            key_count: AtomicUsize::new(0),
            active_reservations: AtomicUsize::new(0),
        })
    }

    /// The configured capacity, for bounded quota headers.
    pub(super) fn limit(&self) -> u64 {
        self.config.capacity
    }

    /// Current number of active reservations.
    pub(super) fn active_count(&self) -> usize {
        self.active_reservations.load(Ordering::Relaxed)
    }

    /// Current number of retained logical keys.
    pub(super) fn key_count(&self) -> usize {
        self.key_count.load(Ordering::Relaxed)
    }

    /// Reserve an estimate against one key's bucket: refill to `now_ms`,
    /// admit and immediately decrement if enough tokens are available,
    /// deny otherwise.
    pub(super) fn reserve(&self, key: &str, estimate: u64, now_ms: u64) -> Decision {
        if key.is_empty() || key.len() > self.config.max_key_length || estimate == 0 {
            return Decision::Denied { retry_after_ms: 0 };
        }

        let state = match self.keys.entry(key.to_owned()) {
            dashmap::mapref::entry::Entry::Occupied(entry) => Arc::clone(entry.get()),
            dashmap::mapref::entry::Entry::Vacant(entry) => {
                if self
                    .key_count
                    .fetch_update(Ordering::AcqRel, Ordering::Relaxed, |count| {
                        (count < self.config.max_keys).then_some(count + 1)
                    })
                    .is_err()
                {
                    return Decision::Denied { retry_after_ms: 0 };
                }
                let state = Arc::new(Mutex::new(BucketState::new(self.config.capacity)));
                entry.insert(Arc::clone(&state));
                state
            },
        };
        let mut state = match state.lock() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        };
        state.refill(now_ms, &self.config);
        let expired = state.reap(now_ms, &self.config);
        for id in &expired {
            self.reservations.remove(id);
        }
        self.active_reservations.fetch_sub(expired.len(), Ordering::Relaxed);

        #[expect(
            clippy::cast_precision_loss,
            reason = "token estimates are far below f64's 2^53 mantissa"
        )]
        let estimate_f64 = estimate as f64;
        if estimate_f64 > state.tokens {
            let deficit = estimate_f64 - state.tokens;
            let retry_after_ms = (deficit / self.config.refill_rate * 1000.0).ceil();
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "retry_after_ms is a small positive duration bounded by realistic refill rates"
            )]
            let retry_after_ms = retry_after_ms.max(1.0) as u64;
            return Decision::Denied { retry_after_ms };
        }
        if self
            .active_reservations
            .fetch_update(Ordering::AcqRel, Ordering::Relaxed, |active| {
                (active < self.config.max_active_reservations).then_some(active + 1)
            })
            .is_err()
        {
            return Decision::Denied {
                retry_after_ms: self.config.reservation_timeout_ms,
            };
        }

        state.tokens -= estimate_f64;
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        state.active.insert(
            id,
            ActiveReservation {
                estimate,
                created_at_ms: now_ms,
            },
        );
        self.reservations.insert(id, key.to_owned());
        drop(state);
        Decision::Admitted(Reservation {
            id,
            estimate,
            created_at_ms: now_ms,
        })
    }

    /// Reconcile actual usage: refill to `now_ms`, then credit back
    /// `estimate - actual` (capped at capacity) or debit the shortfall
    /// (floored at 0). Repeated calls for one ID are no-ops.
    pub(super) fn reconcile(&self, id: u64, actual: Option<u64>, now_ms: u64) -> Settlement {
        let Some((_, key)) = self.reservations.remove(&id) else {
            return Settlement::Noop;
        };
        let Some(state) = self.keys.get(&key).map(|entry| Arc::clone(entry.value())) else {
            return Settlement::Noop;
        };
        let mut state = match state.lock() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        };
        let Some(reservation) = state.active.remove(&id) else {
            return Settlement::Noop;
        };
        self.active_reservations.fetch_sub(1, Ordering::Relaxed);
        state.refill(now_ms, &self.config);

        let actual = actual.unwrap_or(reservation.estimate);
        let refund = reservation.estimate.saturating_sub(actual);
        let overage = actual.saturating_sub(reservation.estimate);
        #[expect(
            clippy::cast_precision_loss,
            reason = "token deltas are far below f64's 2^53 mantissa"
        )]
        {
            if refund > 0 {
                state.tokens = (state.tokens + refund as f64).min(self.config.capacity as f64);
            } else if overage > 0 {
                state.tokens = (state.tokens - overage as f64).max(0.0);
            }
        }
        drop(state);
        Settlement::Applied {
            actual,
            refund,
            overage,
        }
    }

    /// Conservatively expire a bounded number of keys and reclaim idle
    /// state.
    pub(super) fn cleanup(&self, now_ms: u64, max_keys_to_scan: usize) -> usize {
        let mut orphaned = 0;
        let keys: Vec<String> = self
            .keys
            .iter()
            .take(max_keys_to_scan)
            .map(|entry| entry.key().clone())
            .collect();
        for key in keys {
            let Some(entry) = self.keys.get_mut(&key) else {
                continue;
            };
            let state_arc = Arc::clone(entry.value());
            let mut state = match state_arc.lock() {
                Ok(state) => state,
                Err(poisoned) => poisoned.into_inner(),
            };
            state.refill(now_ms, &self.config);
            let expired = state.reap(now_ms, &self.config);
            orphaned += expired.len();
            for id in &expired {
                self.reservations.remove(id);
            }
            self.active_reservations.fetch_sub(expired.len(), Ordering::Relaxed);
            let empty = state.is_empty(self.config.capacity);
            drop(state);
            drop(entry);
            if empty
                && self
                    .keys
                    .remove_if(&key, |_, candidate| {
                        Arc::ptr_eq(candidate, &state_arc)
                            && match candidate.lock() {
                                Ok(state) => state.is_empty(self.config.capacity),
                                Err(poisoned) => poisoned.into_inner().is_empty(self.config.capacity),
                            }
                    })
                    .is_some()
            {
                self.key_count.fetch_sub(1, Ordering::Relaxed);
            }
        }
        orphaned
    }
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::manual_let_else,
    clippy::match_wildcard_for_single_variants,
    reason = "ledger tests intentionally fail fast on impossible fixture states"
)]
mod tests {
    use std::sync::{Arc, Barrier};

    use super::*;

    fn ledger(capacity: u64, refill_rate: f64) -> TokenBucketLedger {
        TokenBucketLedger::new(TokenBucketConfig {
            capacity,
            refill_rate,
            reservation_timeout_ms: 100,
            max_keys: 8,
            max_key_length: 256,
            max_active_reservations: 32,
        })
        .unwrap()
    }

    #[test]
    fn admits_within_capacity_and_denies_over_capacity() {
        let l = ledger(10, 1.0);
        assert!(matches!(l.reserve("a", 10, 0), Decision::Admitted(_)));
        assert!(matches!(l.reserve("a", 1, 0), Decision::Denied { .. }));
    }

    #[test]
    fn refills_over_time_at_the_configured_rate() {
        let l = ledger(10, 10.0); // 10 tokens/sec
        assert!(matches!(l.reserve("a", 10, 0), Decision::Admitted(_)));
        assert!(matches!(l.reserve("a", 1, 0), Decision::Denied { .. }));
        // 200ms at 10 tokens/sec refills 2 tokens.
        assert!(matches!(l.reserve("a", 2, 200), Decision::Admitted(_)));
    }

    #[test]
    fn refill_never_exceeds_capacity() {
        let l = ledger(5, 1000.0);
        assert!(matches!(l.reserve("a", 1, 0), Decision::Admitted(_)));
        // A huge amount of elapsed time should cap at capacity (5), not overflow.
        assert!(matches!(l.reserve("a", 5, 1_000_000), Decision::Admitted(_)));
        assert!(matches!(l.reserve("a", 1, 1_000_000), Decision::Denied { .. }));
    }

    #[test]
    fn reconcile_credits_back_unused_estimate_on_overestimate() {
        let l = ledger(100, 1.0);
        let r = match l.reserve("a", 50, 0) {
            Decision::Admitted(r) => r,
            other => panic!("expected admission, got {other:?}"),
        };
        assert_eq!(
            l.reconcile(r.id, Some(20), 0),
            Settlement::Applied {
                actual: 20,
                refund: 30,
                overage: 0
            }
        );
        // 100 - 50 (reserved) + 30 (refund) = 80 available.
        assert!(matches!(l.reserve("a", 80, 0), Decision::Admitted(_)));
    }

    #[test]
    fn reconcile_debits_the_shortfall_on_underestimate() {
        let l = ledger(100, 1.0);
        let r = match l.reserve("a", 50, 0) {
            Decision::Admitted(r) => r,
            other => panic!("expected admission, got {other:?}"),
        };
        assert_eq!(
            l.reconcile(r.id, Some(70), 0),
            Settlement::Applied {
                actual: 70,
                refund: 0,
                overage: 20
            }
        );
        // 100 - 50 (reserved) - 20 (extra overage debit) = 30 available.
        assert!(matches!(l.reserve("a", 30, 0), Decision::Admitted(_)));
        assert!(
            matches!(l.reserve("b", 1, 0), Decision::Admitted(_)),
            "other keys unaffected"
        );
    }

    #[test]
    fn overage_debit_floors_at_zero_rather_than_going_negative() {
        let l = ledger(10, 1.0);
        let r = match l.reserve("a", 5, 0) {
            Decision::Admitted(r) => r,
            other => panic!("expected admission, got {other:?}"),
        };
        // actual (100) wildly exceeds both the estimate and the bucket's
        // total capacity -- the bucket must floor at 0, not panic or
        // wrap on the unsigned subtraction.
        assert_eq!(
            l.reconcile(r.id, Some(100), 0),
            Settlement::Applied {
                actual: 100,
                refund: 0,
                overage: 95
            }
        );
        assert!(matches!(l.reserve("a", 1, 0), Decision::Denied { .. }));
    }

    #[test]
    fn duplicate_reconciliation_is_noop() {
        let l = ledger(100, 1.0);
        let r = match l.reserve("a", 5, 0) {
            Decision::Admitted(r) => r,
            other => panic!("expected admission, got {other:?}"),
        };
        assert!(matches!(
            l.reconcile(r.id, None, 0),
            Settlement::Applied { actual: 5, .. }
        ));
        assert_eq!(l.reconcile(r.id, Some(99), 0), Settlement::Noop);
    }

    #[test]
    fn keys_are_independent_and_bounded() {
        let l = ledger(10, 1.0);
        assert!(matches!(l.reserve("a", 10, 0), Decision::Admitted(_)));
        assert!(matches!(l.reserve("b", 10, 0), Decision::Admitted(_)));
        assert!(matches!(l.reserve("", 1, 0), Decision::Denied { .. }));
    }

    #[test]
    fn abandoned_reservation_stays_charged_after_timeout() {
        // A lost request (never reconciled) must not free its reserved
        // tokens back into the bucket just because it timed out -- the
        // tokens were already spent at reserve time under the
        // immediate-decrement design, so "abandoned" must mean "stays
        // charged", matching the sliding-window ledger's equivalent
        // guarantee.
        let l = ledger(10, 1.0);
        assert!(matches!(l.reserve("a", 10, 0), Decision::Admitted(_)));
        assert_eq!(l.active_count(), 1);
        l.cleanup(200, 8); // past reservation_timeout_ms=100
        assert_eq!(l.active_count(), 0, "orphan should be reaped from active tracking");
        // Capacity is still fully charged (no refill credited for the
        // abandoned reservation) -- only the 1 token/sec natural refill
        // over 200ms (0.2 tokens) is available, nowhere near 10.
        assert!(matches!(l.reserve("a", 1, 200), Decision::Denied { .. }));
    }

    #[test]
    fn idle_settled_keys_are_reclaimed_once_fully_refilled() {
        let l = ledger(10, 1000.0); // fast refill so full recovery is reachable in the test
        let r = match l.reserve("a", 10, 0) {
            Decision::Admitted(r) => r,
            other => panic!("expected admission, got {other:?}"),
        };
        l.reconcile(r.id, Some(10), 0);
        assert_eq!(l.key_count(), 1);
        // Not yet reclaimed: still short of full capacity at the same instant.
        l.cleanup(0, 8);
        assert_eq!(
            l.key_count(),
            1,
            "a not-yet-refilled key must not be evicted (would grant free bonus tokens)"
        );

        // 100ms at 1000 tokens/sec fully refills the 10-token bucket.
        l.cleanup(100, 8);
        assert_eq!(
            l.key_count(),
            0,
            "a fully-refilled, reservation-free key can be safely forgotten"
        );
    }

    #[test]
    fn concurrent_same_key_admission_cannot_oversubscribe() {
        let ledger = Arc::new(ledger(100, 0.000_001)); // negligible refill during the test window
        let barrier = Arc::new(Barrier::new(16));
        let handles = (0..16)
            .map(|_| {
                let ledger = Arc::clone(&ledger);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    matches!(ledger.reserve("same", 10, 0), Decision::Admitted(_))
                })
            })
            .collect::<Vec<_>>();
        let admitted = handles
            .into_iter()
            .filter_map(|handle| handle.join().ok())
            .filter(|ok| *ok)
            .count();
        assert_eq!(admitted, 10, "exactly the capacity should be admitted");
    }

    #[test]
    fn invalid_config_is_rejected() {
        assert!(
            TokenBucketLedger::new(TokenBucketConfig {
                capacity: 0,
                refill_rate: 1.0,
                reservation_timeout_ms: 1,
                max_keys: 1,
                max_key_length: 1,
                max_active_reservations: 1,
            })
            .is_err()
        );
        assert!(
            TokenBucketLedger::new(TokenBucketConfig {
                capacity: 1,
                refill_rate: 0.0,
                reservation_timeout_ms: 1,
                max_keys: 1,
                max_key_length: 1,
                max_active_reservations: 1,
            })
            .is_err()
        );
    }

    #[test]
    fn non_finite_refill_rate_is_rejected() {
        // `refill_rate <= 0.0` alone is always false for NaN/+-Infinity.
        for bad_rate in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                TokenBucketLedger::new(TokenBucketConfig {
                    capacity: 1,
                    refill_rate: bad_rate,
                    reservation_timeout_ms: 1,
                    max_keys: 1,
                    max_key_length: 1,
                    max_active_reservations: 1,
                })
                .is_err(),
                "refill_rate {bad_rate} must be rejected as non-finite/non-positive"
            );
        }
    }

    #[test]
    fn zero_reservation_timeout_is_rejected() {
        assert!(
            TokenBucketLedger::new(TokenBucketConfig {
                capacity: 1,
                refill_rate: 1.0,
                reservation_timeout_ms: 0,
                max_keys: 1,
                max_key_length: 1,
                max_active_reservations: 1,
            })
            .is_err()
        );
    }

    #[test]
    fn zero_ledger_bounds_are_rejected() {
        for (max_keys, max_key_length, max_active_reservations) in [(0, 1, 1), (1, 0, 1), (1, 1, 0)] {
            assert!(
                TokenBucketLedger::new(TokenBucketConfig {
                    capacity: 1,
                    refill_rate: 1.0,
                    reservation_timeout_ms: 1,
                    max_keys,
                    max_key_length,
                    max_active_reservations,
                })
                .is_err(),
                "bounds ({max_keys}, {max_key_length}, {max_active_reservations}) must be rejected"
            );
        }
    }

    #[test]
    fn key_capacity_denies_new_keys_beyond_the_configured_limit() {
        let l = TokenBucketLedger::new(TokenBucketConfig {
            capacity: 100,
            refill_rate: 1.0,
            reservation_timeout_ms: 100,
            max_keys: 1,
            max_key_length: 256,
            max_active_reservations: 32,
        })
        .unwrap();
        assert!(matches!(l.reserve("a", 1, 0), Decision::Admitted(_)));
        assert!(matches!(l.reserve("b", 1, 0), Decision::Denied { .. }));
    }

    #[test]
    fn reservation_capacity_denies_beyond_the_configured_limit() {
        let l = TokenBucketLedger::new(TokenBucketConfig {
            capacity: 1_000,
            refill_rate: 1.0,
            reservation_timeout_ms: 100,
            max_keys: 32,
            max_key_length: 256,
            max_active_reservations: 1,
        })
        .unwrap();
        assert!(matches!(l.reserve("a", 1, 0), Decision::Admitted(_)));
        assert!(matches!(l.reserve("b", 1, 0), Decision::Denied { .. }));
    }
}
