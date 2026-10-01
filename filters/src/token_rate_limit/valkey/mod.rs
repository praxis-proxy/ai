// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Valkey-backed state for the `token_rate_limit` filter, on plain commands.
//!
//! Admission and reconciliation are composed from plain Valkey commands
//! instead of server-side scripts. The sliding window uses commutative
//! operations (`INCRBY`, `ZADD`, `ZREMRANGEBYSCORE`, TTLs): two admissions
//! racing on one key may both pass, and the proposal accepts that momentary
//! overshoot; what is never lost is the usage they write. The token bucket's
//! refill is a read-modify-write, so it runs under `WATCH`/`MULTI`/`EXEC`.
//! In both, each reservation lives in its own key that expires with the
//! reservation timeout, and it is settled exactly once: both algorithms
//! delete it in a transaction that `WATCH` aborts when another settlement
//! got there first.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use praxis_ai_apis::hash::Sha256;

use super::backend::{BackendError, BackendSnapshot};

mod connection;
mod sliding_window;
mod token_bucket;
mod window;

pub(super) use connection::ValkeyConnection;
pub(super) use sliding_window::{ValkeySlidingWindowBackend, ValkeySlidingWindowConfig};
pub(super) use token_bucket::{ValkeyTokenBucketBackend, ValkeyTokenBucketConfig};

/// Last values a rule's Valkey backend observed, published as gauges under
/// the rule's label. Only `remaining` is the rule's own; `active` and
/// `active` is shared by every rule of the same algorithm in the namespace;
/// `keys` is scoped to this rule.
#[derive(Debug, Default)]
pub(super) struct RuleTelemetry {
    /// Remaining budget for the key this replica last decided.
    remaining: AtomicU64,
    /// Active reservations of this namespace and algorithm, after the last
    /// decision.
    active: AtomicUsize,
    /// Retained keys of this rule, after the last
    /// decision.
    keys: AtomicUsize,
}

impl RuleTelemetry {
    /// Replace the snapshot.
    pub(super) fn record(&self, remaining: u64, active: usize, keys: usize) {
        self.remaining.store(remaining, Ordering::Relaxed);
        self.active.store(active, Ordering::Relaxed);
        self.keys.store(keys, Ordering::Relaxed);
    }

    /// Read the snapshot without I/O.
    pub(super) fn snapshot(&self) -> BackendSnapshot {
        BackendSnapshot {
            budget_remaining: self
                .remaining
                .load(Ordering::Relaxed)
                .min(super::MAX_REPORTED_REMAINING),
            active_reservations: self.active.load(Ordering::Relaxed),
            active_keys: self.keys.load(Ordering::Relaxed),
        }
    }
}

/// Hex SHA-256 of `parts` joined by a zero byte: every Valkey key name
/// derived from a namespace, rule, or budget key goes through this, so no
/// user identifier appears in the keyspace.
pub(super) fn key_hash(parts: &[&[u8]]) -> String {
    let mut digest = Sha256::new();
    for (index, part) in parts.iter().enumerate() {
        if index > 0 {
            digest.update(&[0]);
        }
        digest.update(part);
    }
    digest.finish().iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Extend a namespace-wide key's TTL to at least `ttl_ms` without ever
/// shortening it: `NX` covers the key having no TTL yet (a fresh
/// namespace), `GT` then only ever lengthens an existing one, so a rule
/// with a short TTL can never cut short another rule's still-live entries
/// in the same namespace.
pub(super) fn extend_shared_ttl(pipe: &mut redis::Pipeline, key: &str, ttl_ms: u64) {
    pipe.cmd("PEXPIRE").arg(key).arg(ttl_ms).arg("NX").ignore();
    pipe.cmd("PEXPIRE").arg(key).arg(ttl_ms).arg("GT").ignore();
}

/// Parse `"estimate|admitted_at_ms"` read from a per-reservation key.
pub(super) fn parse_reservation(value: &str) -> Result<(u64, u64), BackendError> {
    let (estimate, admitted_at) = value.split_once('|').ok_or(BackendError::InvalidResponse)?;
    Ok((
        estimate.parse().map_err(|_error| BackendError::InvalidResponse)?,
        admitted_at.parse().map_err(|_error| BackendError::InvalidResponse)?,
    ))
}

/// A non-negative reply used as a count.
pub(super) fn count(value: i64) -> Result<usize, BackendError> {
    usize::try_from(value).map_err(|_error| BackendError::InvalidResponse)
}

/// A non-negative reply used as a token amount.
pub(super) fn amount(value: i64) -> Result<u64, BackendError> {
    u64::try_from(value).map_err(|_error| BackendError::InvalidResponse)
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, reason = "tests")]
mod tests {
    use super::{BackendError, RuleTelemetry, amount, count, key_hash, parse_reservation};

    #[test]
    fn key_hash_is_stable_and_separator_sensitive() {
        let a = key_hash(&[b"ns", b"rule", b"alice"]);
        assert_eq!(a, key_hash(&[b"ns", b"rule", b"alice"]), "same parts, same hash");
        assert_ne!(
            a,
            key_hash(&[b"nsrule", b"alice"]),
            "parts are separated, not concatenated"
        );
        assert_eq!(a.len(), 64, "hex sha-256");
    }

    #[test]
    fn telemetry_snapshot_reports_the_last_recorded_values() {
        let telemetry = RuleTelemetry::default();
        telemetry.record(40, 2, 3);
        let snapshot = telemetry.snapshot();
        assert_eq!(
            snapshot.budget_remaining, 40,
            "remaining must round-trip through record/snapshot"
        );
        assert_eq!(
            snapshot.active_reservations, 2,
            "active must round-trip through record/snapshot"
        );
        assert_eq!(snapshot.active_keys, 3, "keys must round-trip through record/snapshot");
    }

    #[test]
    fn negative_replies_are_invalid_responses() {
        assert!(count(-1).is_err(), "a negative count cannot come from ZCARD");
        assert!(amount(-5).is_err(), "a negative amount cannot come from a counter sum");
        assert_eq!(
            amount(7).unwrap(),
            7,
            "a non-negative amount reply must pass through unchanged"
        );
        assert_eq!(
            count(7).unwrap(),
            7,
            "a non-negative count reply must pass through unchanged"
        );
    }

    #[test]
    fn reservation_values_parse_or_are_invalid_responses() {
        assert_eq!(
            parse_reservation("60|1000").unwrap(),
            (60, 1_000),
            "estimate and admission time round-trip"
        );
        for malformed in ["60", "x|1000", "60|", "-1|1000"] {
            assert!(
                matches!(parse_reservation(malformed), Err(BackendError::InvalidResponse)),
                "{malformed:?} is not a reservation value"
            );
        }
    }
}
