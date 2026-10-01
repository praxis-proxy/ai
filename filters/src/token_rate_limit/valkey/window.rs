// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Fixed sub-window arithmetic shared by the Valkey sliding-window backend.

/// Sub-windows per window for windows of at least one minute.
const SUB_WINDOWS: u64 = 60;

/// Smallest sub-window width; windows shorter than a minute use it.
const MIN_BUCKET_MS: u64 = 1_000;

/// Width of one sub-window for a window of `window_ms`.
pub(super) fn bucket_ms(window_ms: u64) -> u64 {
    window_ms.div_ceil(SUB_WINDOWS).max(MIN_BUCKET_MS)
}

/// Index of the sub-window that contains the instant `at_ms`.
pub(super) fn bucket_index(at_ms: u64, bucket_ms: u64) -> u64 {
    at_ms / bucket_ms.max(1)
}

/// The consecutive sub-windows whose counters must be summed for one
/// admission decision: from the bucket containing `now - window` to the
/// bucket containing `now`, inclusive. The oldest bucket is only partly
/// inside the window and is counted in full, so usage leaves the window up
/// to one bucket late, never early.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct BucketRange {
    /// Index of the oldest bucket.
    pub(super) first: u64,
    /// Number of buckets, oldest to newest.
    pub(super) count: usize,
}

impl BucketRange {
    /// Buckets covering `[now_ms - window_ms, now_ms]`.
    pub(super) fn covering(now_ms: u64, window_ms: u64) -> Self {
        let width = bucket_ms(window_ms);
        let first = bucket_index(now_ms.saturating_sub(window_ms), width);
        let last = bucket_index(now_ms, width);
        Self {
            first,
            count: usize::try_from(last - first + 1).unwrap_or(usize::MAX),
        }
    }

    /// Bucket indexes, oldest first.
    pub(super) fn indexes(&self) -> impl Iterator<Item = u64> {
        let count = u64::try_from(self.count).unwrap_or(u64::MAX);
        self.first..self.first.saturating_add(count)
    }
}

/// Milliseconds until the bucket `oldest_non_empty` has aged out of a
/// `window_ms` window that ends at `now_ms`; at least 1.
pub(super) fn retry_after_ms(now_ms: u64, window_ms: u64, oldest_non_empty: u64) -> u64 {
    let width = bucket_ms(window_ms);
    let leaves_at = oldest_non_empty
        .saturating_add(1)
        .saturating_mul(width)
        .saturating_add(window_ms);
    leaves_at.saturating_sub(now_ms).max(1)
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::panic, reason = "tests")]
mod tests {
    use super::{BucketRange, SUB_WINDOWS, bucket_index, bucket_ms, retry_after_ms};

    #[test]
    fn windows_of_a_minute_or_more_get_sixty_sub_windows() {
        assert_eq!(bucket_ms(3_600_000), 60_000, "one hour splits into minutes");
        assert_eq!(bucket_ms(60_000), 1_000, "one minute splits into seconds");
        assert_eq!(bucket_ms(90_000), 1_500, "ninety seconds splits into 1.5 s buckets");
    }

    #[test]
    fn short_windows_use_one_second_sub_windows() {
        assert_eq!(bucket_ms(30_000), 1_000, "under a minute the floor is one second");
        assert_eq!(
            bucket_ms(999),
            1_000,
            "even a sub-second window uses one-second buckets"
        );
    }

    #[test]
    fn bucket_index_is_the_floor_of_time_over_width() {
        assert_eq!(bucket_index(59_999, 60_000), 0, "just before the boundary");
        assert_eq!(bucket_index(60_000, 60_000), 1, "on the boundary");
    }

    #[test]
    fn bucket_range_covers_the_whole_window_plus_the_partial_oldest_bucket() {
        let range = BucketRange::covering(3_600_000 + 30_000, 3_600_000);
        assert_eq!(
            range.count,
            usize::try_from(SUB_WINDOWS).unwrap() + 1,
            "60 full buckets and one partial"
        );
        assert_eq!(range.first, 0, "the bucket that contains now - window");
        assert_eq!(range.indexes().last(), Some(60), "up to the bucket that contains now");
    }

    #[test]
    fn bucket_range_covers_windows_that_are_not_multiples_of_sixty() {
        let range = BucketRange::covering(200_000, 90_000);
        let width = bucket_ms(90_000);
        let oldest_start = range.first * width;
        assert!(
            oldest_start <= 200_000 - 90_000,
            "the oldest bucket starts at or before now - window"
        );
        let newest_end = (range.first + u64::try_from(range.count).unwrap()) * width;
        assert!(newest_end > 200_000, "the newest bucket ends after now");
    }

    #[test]
    fn retry_after_is_the_time_until_the_oldest_usage_leaves_the_window() {
        let now = 3_600_000 + 30_000;
        let oldest = 0;
        assert_eq!(
            retry_after_ms(now, 3_600_000, oldest),
            30_000,
            "bucket 0 ends at 60 s and leaves the window at 60 s + 1 h"
        );
        assert_eq!(
            retry_after_ms(7_300_000, 3_600_000, 0),
            1,
            "a bucket that has already left the window still reports a positive delay"
        );
    }
}
