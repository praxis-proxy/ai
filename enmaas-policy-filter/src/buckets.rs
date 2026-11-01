#![allow(
    clippy::missing_docs_in_private_items,
    clippy::significant_drop_tightening,
    clippy::too_many_lines,
    reason = "private bucket internals"
)]

use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

use dashmap::DashMap;

const IDLE_BUCKET_TTL: Duration = Duration::from_secs(10 * 60);
const SWEEP_EVERY: u64 = 1024;

#[derive(Clone, Copy, Debug)]
pub(crate) struct BucketLimits {
    burst: f64,
    refill_per_second: f64,
}

impl BucketLimits {
    pub(crate) fn new(burst: u32, refill_per_second: f64) -> Self {
        Self {
            burst: f64::from(burst),
            refill_per_second,
        }
    }

    fn disabled(self) -> bool {
        self.burst <= 0.0
    }
}

#[derive(Debug)]
struct BucketState {
    tokens: f64,
    last_refill: Instant,
    last_seen: Instant,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BucketOutcome {
    Admitted,
    Rejected { retry_after: u64 },
}

pub(crate) struct LocalBuckets {
    main: BucketLimits,
    premium: BucketLimits,
    max_keys: usize,
    buckets: DashMap<String, BucketState>,
    admissions: AtomicU64,
}

impl LocalBuckets {
    pub(crate) fn new(main: BucketLimits, premium: BucketLimits, max_keys: usize) -> Self {
        Self {
            main,
            premium,
            max_keys,
            buckets: DashMap::new(),
            admissions: AtomicU64::new(0),
        }
    }

    pub(crate) fn admit(&self, class: &str, subject: &str) -> BucketOutcome {
        let limits = if class == "premium" { self.premium } else { self.main };
        if limits.disabled() {
            return BucketOutcome::Admitted;
        }

        self.maybe_sweep();
        let key = format!("{class}\u{1f}{subject}");
        let now = Instant::now();
        if !self.buckets.contains_key(&key) && self.buckets.len() >= self.max_keys {
            return BucketOutcome::Rejected { retry_after: 30 };
        }

        let mut bucket = self.buckets.entry(key).or_insert_with(|| BucketState {
            tokens: limits.burst,
            last_refill: now,
            last_seen: now,
        });
        let elapsed = now.duration_since(bucket.last_refill).as_secs_f64();
        bucket.tokens = (bucket.tokens + elapsed * limits.refill_per_second).min(limits.burst);
        bucket.last_refill = now;
        bucket.last_seen = now;
        if bucket.tokens >= 1.0 {
            bucket.tokens -= 1.0;
            BucketOutcome::Admitted
        } else {
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "retry seconds are positive and bounded by the bucket refill"
            )]
            let retry_after = ((1.0 - bucket.tokens) / limits.refill_per_second).ceil() as u64;
            BucketOutcome::Rejected {
                retry_after: retry_after.max(1),
            }
        }
    }

    fn maybe_sweep(&self) {
        let admissions = self.admissions.fetch_add(1, Ordering::Relaxed);
        if admissions.is_multiple_of(SWEEP_EVERY) {
            let now = Instant::now();
            self.buckets
                .retain(|_, bucket| now.duration_since(bucket.last_seen) < IDLE_BUCKET_TTL);
        }
    }
}
