// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Lock-free aggregate of every retained key's last reported balance,
//! shared by the in-process sliding-window and token-bucket ledgers.

use std::sync::atomic::{AtomicU64, Ordering};

/// Sum of the last calculated remaining balance for every retained key.
///
/// Each ledger adjusts it by the delta of one key's balance while holding
/// that key's lock, so no ledger-wide lock is needed on the request path.
/// Arithmetic saturates; only the exported snapshot is clamped to the
/// largest integer Prometheus represents exactly.
#[derive(Debug, Default)]
pub(super) struct RemainingTotal(AtomicU64);

impl RemainingTotal {
    /// Add one key's balance when the key is first retained.
    pub(super) fn add(&self, increase: u64) {
        self.0.fetch_add(increase, Ordering::Relaxed);
    }

    /// Remove one key's last reported balance when the key is evicted.
    pub(super) fn subtract(&self, decrease: u64) {
        let _previous = self.0.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |total| {
            Some(total.saturating_sub(decrease))
        });
    }

    /// Move one key's contribution from `previous` to `next`.
    pub(super) fn replace(&self, previous: u64, next: u64) {
        if next >= previous {
            self.add(next - previous);
        } else {
            self.subtract(previous - next);
        }
    }

    /// The aggregate as exported to Prometheus.
    pub(super) fn reported(&self) -> u64 {
        self.0.load(Ordering::Relaxed).min(super::MAX_REPORTED_REMAINING)
    }
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, reason = "tests")]
mod tests {
    use super::RemainingTotal;

    #[test]
    fn replace_moves_a_contribution_in_both_directions() {
        let total = RemainingTotal::default();
        total.add(100);
        total.replace(100, 40);
        assert_eq!(total.reported(), 40, "a lower balance shrinks the aggregate");
        total.replace(40, 70);
        assert_eq!(total.reported(), 70, "a higher balance grows the aggregate");
    }

    #[test]
    fn subtract_saturates_instead_of_wrapping() {
        let total = RemainingTotal::default();
        total.add(5);
        total.subtract(9);
        assert_eq!(total.reported(), 0, "an over-subtraction must floor at zero");
    }

    #[test]
    fn reported_value_is_clamped_to_the_prometheus_safe_integer() {
        let total = RemainingTotal::default();
        total.add(u64::MAX);
        assert_eq!(
            total.reported(),
            super::super::MAX_REPORTED_REMAINING,
            "the export must not exceed 2^53 - 1"
        );
    }
}
