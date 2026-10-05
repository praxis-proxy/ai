// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Checked accounting for independently owned, request-scoped payloads.
//!
//! This is an accounting primitive, not a filter. A caller must reserve before
//! making a new owned copy and release only after that copy is gone. The later
//! Responses admission filters will supply the effective policy and measure
//! every payload owner.

/// Remaining capacity shared by all payload owners in one logical request.
#[derive(Debug, Eq, PartialEq)]
pub struct RetainedPayloadBudget {
    limit: usize,
    retained: usize,
}

impl RetainedPayloadBudget {
    /// Create an empty budget. Policy validation belongs to the caller.
    #[must_use]
    pub const fn new(limit: usize) -> Self {
        Self { limit, retained: 0 }
    }

    /// Return the smallest limit applied to this request.
    #[must_use]
    pub const fn limit(&self) -> usize {
        self.limit
    }

    /// Return the sum of currently reserved owned payloads.
    #[must_use]
    pub const fn retained(&self) -> usize {
        self.retained
    }

    /// Apply another loop instance's limit; the smallest limit wins.
    ///
    /// A `false` result means existing reservations already exceed the new
    /// limit, so the caller must fail the request.
    #[must_use = "a smaller limit can invalidate the current reservations"]
    pub const fn tighten(&mut self, limit: usize) -> bool {
        if limit < self.limit {
            self.limit = limit;
        }
        self.retained <= self.limit
    }

    /// Reserve one known owned payload before it is copied or allocated.
    /// Unknown sizes and arithmetic overflow fail without changing state.
    #[must_use = "a failed reservation must reject the request"]
    pub fn reserve(&mut self, bytes: Option<usize>) -> bool {
        self.reserve_many([bytes])
    }

    /// Atomically reserve several simultaneous owners.
    ///
    /// A batch is rejected if any owner has unknown size or its total exceeds
    /// the limit. No partial reservation is kept on failure.
    #[must_use = "a failed reservation must reject the request"]
    pub fn reserve_many(&mut self, owners: impl IntoIterator<Item = Option<usize>>) -> bool {
        let next = owners
            .into_iter()
            .try_fold(self.retained, |total, bytes| total.checked_add(bytes?));
        match next {
            Some(next) if next <= self.limit => {
                self.retained = next;
                true
            },
            _ => false,
        }
    }

    /// Release an owner after it has actually been dropped or moved out.
    /// An invalid release fails without changing state.
    #[must_use = "an invalid release means accounting is inconsistent"]
    pub const fn release(&mut self, bytes: usize) -> bool {
        if let Some(next) = self.retained.checked_sub(bytes) {
            self.retained = next;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::RetainedPayloadBudget;

    #[test]
    fn admits_exact_limit_and_rejects_one_more() {
        let mut budget = RetainedPayloadBudget::new(10);
        assert!(budget.reserve(Some(9)));
        assert!(budget.reserve(Some(1)));
        assert!(!budget.reserve(Some(1)));
        assert_eq!(budget.retained(), 10);
    }

    #[test]
    fn batch_counts_simultaneous_copies_without_partial_commit() {
        let mut budget = RetainedPayloadBudget::new(20);
        assert!(budget.reserve_many([Some(6), Some(6)]));
        assert!(!budget.reserve_many([Some(4), Some(5)]));
        assert_eq!(budget.retained(), 12);
        assert!(budget.release(6));
        assert!(budget.reserve_many([Some(4), Some(5)]));
        assert_eq!(budget.retained(), 15);
    }

    #[test]
    fn unknown_size_overflow_and_invalid_release_fail_closed() {
        let mut budget = RetainedPayloadBudget::new(usize::MAX);
        assert!(budget.reserve(Some(2)));
        assert!(!budget.reserve(None));
        assert!(!budget.reserve(Some(usize::MAX)));
        assert!(!budget.release(3));
        assert_eq!(budget.retained(), 2);
    }

    #[test]
    fn smallest_loop_limit_wins_even_after_reservation() {
        let mut budget = RetainedPayloadBudget::new(20);
        assert!(budget.reserve(Some(12)));
        assert!(budget.tighten(12));
        assert!(!budget.reserve(Some(1)));
        assert!(!budget.tighten(11));
        assert_eq!(budget.limit(), 11);
        assert_eq!(budget.retained(), 12);
        assert!(!budget.reserve(Some(0)));
    }
}
