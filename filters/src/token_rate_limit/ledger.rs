// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Exact local sliding-window reservation ledger.
//!
//! Adapted, unmodified in logic, from the `token_rate_limit::ledger` module
//! on nerdalert's `poc/distributed-token-rate-limit-demo` spike branch
//! (<https://github.com/nerdalert/ai/tree/poc/distributed-token-rate-limit-demo>).
//! Replaces this filter's token-bucket state with a true sliding window per
//! the proposal's design doc ("Windows are sliding: a `window: 1h` budget
//! tracks usage in the most recent 60 minutes from the current instant"),
//! and answers that same proposal's still-open "lost request handling"
//! question via `reservation_timeout_ms` + conservative charge-at-estimate
//! on expiry.

#![allow(
    missing_docs,
    clippy::missing_docs_in_private_items,
    clippy::too_many_lines,
    reason = "private ledger implementation is covered by its public filter contract and focused tests"
)]

use std::{
    collections::{HashMap, VecDeque},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
};

use dashmap::DashMap;

/// A positive rolling-window budget.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Budget {
    /// Window length in milliseconds.
    pub(super) window_ms: u64,
    /// Maximum settled plus active tokens in the window.
    pub(super) capacity: u64,
}

/// Bounds and timing for a ledger.
#[derive(Clone, Debug)]
pub(super) struct LedgerConfig {
    /// All budgets in one atomic reservation rule.
    pub(super) budgets: Vec<Budget>,
    /// Time after which an ambiguous reservation is charged at its estimate.
    pub(super) reservation_timeout_ms: u64,
    /// Maximum logical keys retained by the ledger.
    pub(super) max_keys: usize,
    /// Maximum key length retained by the ledger.
    pub(super) max_key_length: usize,
    /// Maximum active reservations retained by the ledger.
    pub(super) max_active_reservations: usize,
}

impl LedgerConfig {
    /// Validate configuration before constructing a ledger.
    pub(super) fn validate(&self) -> Result<(), String> {
        if self.budgets.is_empty() {
            return Err("at least one budget is required".into());
        }
        if self.budgets.iter().any(|b| b.window_ms == 0 || b.capacity == 0) {
            return Err("budget window and capacity must be positive".into());
        }
        if self.budgets.windows(2).any(|w| {
            w.first()
                .zip(w.get(1))
                .is_some_and(|(left, right)| left.window_ms == right.window_ms)
        }) {
            return Err("budget windows must be unique".into());
        }
        if self.reservation_timeout_ms == 0 {
            return Err("reservation timeout must be positive".into());
        }
        if self.max_keys == 0 || self.max_key_length == 0 || self.max_active_reservations == 0 {
            return Err("ledger bounds must be positive".into());
        }
        Ok(())
    }
}

/// A reservation admitted atomically across every configured budget.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Reservation {
    /// Opaque identifier used for idempotent reconciliation.
    pub(super) id: u64,
    /// Estimated token cost reserved at admission.
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
        /// Conservative delay before another admission attempt.
        retry_after_ms: u64,
        /// Bounded reason used for operational counters.
        reason: DenialReason,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DenialReason {
    InvalidKey,
    KeyCapacity,
    WindowCapacity,
    ReservationCapacity,
}

/// Result of reconciling a reservation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Settlement {
    /// Actual usage was applied exactly once.
    Applied {
        /// Actual tokens charged to the rolling ledger.
        actual: u64,
        /// Estimate returned to the ledger.
        refund: u64,
        /// Usage above the estimate.
        overage: u64,
    },
    /// The reservation was already reconciled or conservatively expired.
    Noop,
}

#[derive(Debug)]
struct Usage {
    at_ms: u64,
    tokens: u64,
}

#[derive(Debug)]
struct ActiveReservation {
    estimate: u64,
    created_at_ms: u64,
}

#[derive(Debug, Default)]
struct KeyState {
    settled: VecDeque<Usage>,
    active: HashMap<u64, ActiveReservation>,
}

impl KeyState {
    fn reap(&mut self, now_ms: u64, config: &LedgerConfig) -> Vec<u64> {
        let expired: Vec<u64> = self
            .active
            .iter()
            .filter_map(|(id, reservation)| {
                (now_ms.saturating_sub(reservation.created_at_ms) >= config.reservation_timeout_ms).then_some(*id)
            })
            .collect();

        for id in &expired {
            if let Some(reservation) = self.active.remove(id) {
                // An ambiguous request is never free traffic. Charge the
                // estimate at admission time so the normal window expiry
                // rules still apply.
                self.settled.push_back(Usage {
                    at_ms: reservation.created_at_ms,
                    tokens: reservation.estimate,
                });
            }
        }

        let max_window = config.budgets.iter().map(|b| b.window_ms).max().unwrap_or(0);
        while self
            .settled
            .front()
            .is_some_and(|entry| now_ms.saturating_sub(entry.at_ms) >= max_window)
        {
            self.settled.pop_front();
        }
        expired
    }

    fn usage_in_window(&self, now_ms: u64, window_ms: u64) -> u64 {
        let settled = self
            .settled
            .iter()
            .filter(|entry| now_ms.saturating_sub(entry.at_ms) < window_ms)
            .fold(0_u64, |sum, entry| sum.saturating_add(entry.tokens));
        let active = self
            .active
            .values()
            .fold(0_u64, |sum, reservation| sum.saturating_add(reservation.estimate));
        settled.saturating_add(active)
    }

    fn retry_after_ms(&self, now_ms: u64, config: &LedgerConfig) -> u64 {
        config
            .budgets
            .iter()
            .flat_map(|budget| {
                self.settled.iter().filter_map(move |entry| {
                    let expiry = entry.at_ms.saturating_add(budget.window_ms);
                    (expiry > now_ms).then_some(expiry - now_ms)
                })
            })
            .max()
            .unwrap_or(config.reservation_timeout_ms)
            .max(config.reservation_timeout_ms)
    }

    fn is_empty(&self) -> bool {
        self.active.is_empty() && self.settled.is_empty()
    }
}

/// Thread-safe exact local ledger with independent locks per key.
pub(super) struct Ledger {
    config: LedgerConfig,
    keys: DashMap<String, Arc<Mutex<KeyState>>>,
    reservations: DashMap<u64, String>,
    next_id: AtomicU64,
    key_count: AtomicUsize,
    active_reservations: AtomicUsize,
}

impl Ledger {
    /// Construct a validated empty ledger.
    pub(super) fn new(config: LedgerConfig) -> Result<Self, String> {
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

    /// Return the smallest configured capacity for bounded quota headers.
    pub(super) fn limit(&self) -> u64 {
        self.config
            .budgets
            .iter()
            .map(|budget| budget.capacity)
            .min()
            .unwrap_or(0)
    }

    /// Current number of active reservations.
    pub(super) fn active_count(&self) -> usize {
        self.active_reservations.load(Ordering::Relaxed)
    }

    /// Current number of retained logical keys.
    pub(super) fn key_count(&self) -> usize {
        self.key_count.load(Ordering::Relaxed)
    }

    /// Reserve an estimate atomically across all configured windows.
    pub(super) fn reserve(&self, key: &str, estimate: u64, now_ms: u64) -> Decision {
        if key.is_empty() || key.len() > self.config.max_key_length || estimate == 0 {
            return Decision::Denied {
                retry_after_ms: 0,
                reason: DenialReason::InvalidKey,
            };
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
                    return Decision::Denied {
                        retry_after_ms: 0,
                        reason: DenialReason::KeyCapacity,
                    };
                }
                let state = Arc::new(Mutex::new(KeyState::default()));
                entry.insert(Arc::clone(&state));
                state
            },
        };
        let mut state = match state.lock() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        };
        let expired = state.reap(now_ms, &self.config);
        for id in &expired {
            self.reservations.remove(id);
        }
        self.active_reservations.fetch_sub(expired.len(), Ordering::Relaxed);

        if self
            .config
            .budgets
            .iter()
            .any(|budget| state.usage_in_window(now_ms, budget.window_ms).saturating_add(estimate) > budget.capacity)
        {
            return Decision::Denied {
                retry_after_ms: state.retry_after_ms(now_ms, &self.config),
                reason: DenialReason::WindowCapacity,
            };
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
                reason: DenialReason::ReservationCapacity,
            };
        }

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

    /// Reconcile actual usage. Repeated calls for one ID are no-ops.
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
        let actual = actual.unwrap_or(reservation.estimate);
        state.settled.push_back(Usage {
            at_ms: reservation.created_at_ms,
            tokens: actual,
        });
        let refund = reservation.estimate.saturating_sub(actual);
        let overage = actual.saturating_sub(reservation.estimate);
        let expired = state.reap(now_ms, &self.config);
        for expired_id in expired {
            self.reservations.remove(&expired_id);
            self.active_reservations.fetch_sub(1, Ordering::Relaxed);
        }
        drop(state);
        Settlement::Applied {
            actual,
            refund,
            overage,
        }
    }

    /// Conservatively expire a bounded number of keys and reclaim idle state.
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
            let expired = state.reap(now_ms, &self.config);
            orphaned += expired.len();
            for id in &expired {
                self.reservations.remove(id);
            }
            self.active_reservations.fetch_sub(expired.len(), Ordering::Relaxed);
            let empty = state.is_empty();
            drop(state);
            drop(entry);
            if empty
                && self
                    .keys
                    .remove_if(&key, |_, candidate| {
                        Arc::ptr_eq(candidate, &state_arc)
                            && match candidate.lock() {
                                Ok(state) => state.is_empty(),
                                Err(poisoned) => poisoned.into_inner().is_empty(),
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

    fn ledger(budgets: &[(u64, u64)]) -> Ledger {
        Ledger::new(LedgerConfig {
            budgets: budgets
                .iter()
                .map(|&(window_ms, capacity)| Budget { window_ms, capacity })
                .collect(),
            reservation_timeout_ms: 100,
            max_keys: 8,
            max_key_length: 256,
            max_active_reservations: 32,
        })
        .unwrap()
    }

    #[test]
    fn admits_and_denies_one_window() {
        let l = ledger(&[(60_000, 10)]);
        assert!(matches!(l.reserve("a", 10, 0), Decision::Admitted(_)));
        assert!(matches!(l.reserve("a", 1, 0), Decision::Denied { .. }));
    }

    #[test]
    fn concurrent_same_key_admission_cannot_oversubscribe() {
        let ledger = Arc::new(ledger(&[(1_000, 100)]));
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
    fn concurrent_different_keys_respect_global_reservation_bound() {
        let ledger = Arc::new(
            Ledger::new(LedgerConfig {
                budgets: vec![Budget {
                    window_ms: 1_000,
                    capacity: 1_000,
                }],
                reservation_timeout_ms: 100,
                max_keys: 32,
                max_key_length: 256,
                max_active_reservations: 4,
            })
            .unwrap(),
        );
        let barrier = Arc::new(Barrier::new(16));
        let handles = (0..16)
            .map(|index| {
                let ledger = Arc::clone(&ledger);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    matches!(ledger.reserve(&format!("key-{index}"), 1, 0), Decision::Admitted(_))
                })
            })
            .collect::<Vec<_>>();
        let admitted = handles
            .into_iter()
            .filter_map(|handle| handle.join().ok())
            .filter(|ok| *ok)
            .count();
        assert_eq!(admitted, 4, "the global reservation bound must be atomic");
        assert_eq!(ledger.active_count(), 4);
    }

    #[test]
    fn exact_boundary_expires_usage() {
        let l = ledger(&[(100, 10)]);
        let r = match l.reserve("a", 10, 0) {
            Decision::Admitted(r) => r,
            _ => panic!(),
        };
        assert!(matches!(l.reconcile(r.id, Some(10), 0), Settlement::Applied { .. }));
        assert!(matches!(l.reserve("a", 10, 100), Decision::Admitted(_)));
    }

    #[test]
    fn multiple_windows_are_atomic() {
        let l = ledger(&[(100, 10), (1_000, 15)]);
        let r = match l.reserve("a", 10, 0) {
            Decision::Admitted(r) => r,
            _ => panic!(),
        };
        l.reconcile(r.id, Some(10), 0);
        assert!(matches!(l.reserve("a", 6, 0), Decision::Denied { .. }));
        assert!(matches!(l.reserve("b", 6, 0), Decision::Admitted(_)));
    }

    #[test]
    fn active_reservations_count_and_orphans_are_charged() {
        let l = ledger(&[(1_000, 10)]);
        assert!(matches!(l.reserve("a", 10, 0), Decision::Admitted(_)));
        assert_eq!(l.active_count(), 1);
        assert!(matches!(l.reserve("a", 1, 0), Decision::Denied { .. }));
        l.cleanup(100, 8);
        assert_eq!(l.active_count(), 0);
        assert!(matches!(l.reserve("a", 1, 100), Decision::Denied { .. }));
    }

    #[test]
    fn idle_settled_keys_are_reclaimed_atomically() {
        let l = ledger(&[(100, 10)]);
        let r = match l.reserve("a", 10, 0) {
            Decision::Admitted(r) => r,
            _ => panic!(),
        };
        assert!(matches!(l.reconcile(r.id, Some(10), 0), Settlement::Applied { .. }));
        assert_eq!(l.key_count(), 1);
        l.cleanup(100, 8);
        assert_eq!(l.key_count(), 0);
    }

    #[test]
    fn refund_exact_and_overage_are_recorded() {
        let l = ledger(&[(1_000, 100)]);
        let r = match l.reserve("a", 50, 0) {
            Decision::Admitted(r) => r,
            _ => panic!(),
        };
        assert_eq!(
            l.reconcile(r.id, Some(20), 0),
            Settlement::Applied {
                actual: 20,
                refund: 30,
                overage: 0
            }
        );
        let r = match l.reserve("a", 50, 0) {
            Decision::Admitted(r) => r,
            _ => panic!(),
        };
        assert_eq!(
            l.reconcile(r.id, Some(70), 0),
            Settlement::Applied {
                actual: 70,
                refund: 0,
                overage: 20
            }
        );
    }

    #[test]
    fn duplicate_reconciliation_is_noop() {
        let l = ledger(&[(1_000, 100)]);
        let r = match l.reserve("a", 5, 0) {
            Decision::Admitted(r) => r,
            _ => panic!(),
        };
        assert!(matches!(
            l.reconcile(r.id, None, 0),
            Settlement::Applied { actual: 5, .. }
        ));
        assert_eq!(l.reconcile(r.id, Some(99), 0), Settlement::Noop);
    }

    #[test]
    fn keys_are_independent_and_bounded() {
        let l = ledger(&[(1_000, 10)]);
        assert!(matches!(l.reserve("a", 10, 0), Decision::Admitted(_)));
        assert!(matches!(l.reserve("b", 10, 0), Decision::Admitted(_)));
        assert!(matches!(l.reserve("", 1, 0), Decision::Denied { .. }));
    }

    #[test]
    fn invalid_config_is_rejected() {
        assert!(
            Ledger::new(LedgerConfig {
                budgets: vec![],
                reservation_timeout_ms: 1,
                max_keys: 1,
                max_key_length: 1,
                max_active_reservations: 1
            })
            .is_err()
        );
        assert!(
            Ledger::new(LedgerConfig {
                budgets: vec![Budget {
                    window_ms: 0,
                    capacity: 1
                }],
                reservation_timeout_ms: 1,
                max_keys: 1,
                max_key_length: 1,
                max_active_reservations: 1
            })
            .is_err()
        );
    }

    #[test]
    fn duplicate_budget_windows_are_rejected() {
        assert!(
            Ledger::new(LedgerConfig {
                budgets: vec![
                    Budget {
                        window_ms: 1_000,
                        capacity: 10
                    },
                    Budget {
                        window_ms: 1_000,
                        capacity: 20
                    },
                ],
                reservation_timeout_ms: 1,
                max_keys: 1,
                max_key_length: 1,
                max_active_reservations: 1
            })
            .is_err(),
            "two budgets sharing the same window are ambiguous"
        );
    }

    #[test]
    fn zero_reservation_timeout_is_rejected() {
        assert!(
            Ledger::new(LedgerConfig {
                budgets: vec![Budget {
                    window_ms: 1_000,
                    capacity: 10
                }],
                reservation_timeout_ms: 0,
                max_keys: 1,
                max_key_length: 1,
                max_active_reservations: 1
            })
            .is_err()
        );
    }

    #[test]
    fn zero_ledger_bounds_are_rejected() {
        for (max_keys, max_key_length, max_active_reservations) in [(0, 1, 1), (1, 0, 1), (1, 1, 0)] {
            assert!(
                Ledger::new(LedgerConfig {
                    budgets: vec![Budget {
                        window_ms: 1_000,
                        capacity: 10
                    }],
                    reservation_timeout_ms: 1,
                    max_keys,
                    max_key_length,
                    max_active_reservations
                })
                .is_err(),
                "bounds ({max_keys}, {max_key_length}, {max_active_reservations}) must be rejected"
            );
        }
    }

    #[test]
    fn key_capacity_denies_new_keys_beyond_the_configured_limit() {
        let l = Ledger::new(LedgerConfig {
            budgets: vec![Budget {
                window_ms: 1_000,
                capacity: 100,
            }],
            reservation_timeout_ms: 100,
            max_keys: 1,
            max_key_length: 256,
            max_active_reservations: 32,
        })
        .unwrap();
        assert!(matches!(l.reserve("a", 1, 0), Decision::Admitted(_)));
        assert!(matches!(
            l.reserve("b", 1, 0),
            Decision::Denied {
                reason: DenialReason::KeyCapacity,
                ..
            }
        ));
    }

    #[test]
    fn expired_active_reservation_is_reaped_on_next_reserve_for_the_same_key() {
        let l = ledger(&[(100, 10)]);
        let r = match l.reserve("a", 10, 0) {
            Decision::Admitted(r) => r,
            other => panic!("expected admission, got {other:?}"),
        };
        assert_eq!(l.active_count(), 1);
        // Never reconciled, and now well past both the window and the
        // reservation timeout -- the next reserve for "a" must reap the
        // stale reservation before evaluating capacity, not deny it as
        // still active.
        assert!(matches!(l.reserve("a", 10, 200), Decision::Admitted(_)));
        assert_eq!(l.active_count(), 1, "old reservation reaped, new one admitted");
        assert_eq!(
            l.reconcile(r.id, Some(1), 200),
            Settlement::Noop,
            "the reaped id was actually dropped, not just shadowed"
        );
    }

    #[test]
    fn expired_sibling_reservation_is_reaped_during_reconcile() {
        let l = ledger(&[(1_000, 10)]);
        let stale = match l.reserve("a", 1, 0) {
            Decision::Admitted(r) => r,
            other => panic!("expected admission, got {other:?}"),
        };
        let fresh = match l.reserve("a", 1, 50) {
            Decision::Admitted(r) => r,
            other => panic!("expected admission, got {other:?}"),
        };
        assert_eq!(l.active_count(), 2);
        // Reconciling `fresh` at t=150 (>= reservation_timeout_ms=100 past
        // `stale`'s creation) must also reap `stale` as a side effect.
        assert!(matches!(
            l.reconcile(fresh.id, Some(1), 150),
            Settlement::Applied { .. }
        ));
        assert_eq!(l.active_count(), 0, "the stale sibling must be reaped too");
        assert_eq!(l.reconcile(stale.id, Some(1), 150), Settlement::Noop);
    }
}
