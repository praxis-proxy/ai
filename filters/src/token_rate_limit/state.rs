// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Bucket state: a single global bucket, or one bucket per header value.
//!
//! Mirrors `praxis`'s own `rate_limit` filter's `RateLimitState::PerIp`
//! pattern (`DashMap` keyed storage, soft/hard eviction caps) -- see
//! `praxis::filter::builtins::http::traffic_management::rate_limit` --
//! applied to an arbitrary header value instead of a client IP, per
//! `ai#129`'s spec ("Configure a header name as the bucket key source",
//! "fallback to global bucket if header absent", "bucket eviction for
//! inactive keys").

use dashmap::DashMap;

use super::bucket::TokenBucket;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Maximum number of per-key entries before eviction is triggered.
///
/// Same soft cap `rate_limit` uses for per-IP entries; app/tenant
/// cardinality is expected to be far lower in practice.
const MAX_KEY_ENTRIES: usize = 100_000;

/// Hard cap on per-key entries; new keys fall back to the shared bucket above this.
const HARD_CAP_KEY_ENTRIES: usize = 200_000; // 2 * MAX_KEY_ENTRIES

/// Maximum entries to scan during a single (non-aggressive) eviction pass.
const EVICTION_SCAN_LIMIT: usize = 2_048;

/// Entry count above which eviction scans the entire map.
const AGGRESSIVE_EVICTION_THRESHOLD: usize = MAX_KEY_ENTRIES + MAX_KEY_ENTRIES / 2; // 150_000

// -----------------------------------------------------------------------------
// TokenRateLimitState
// -----------------------------------------------------------------------------

/// Per-filter bucket state: one shared bucket, or one bucket per header value.
pub(super) enum TokenRateLimitState {
    /// One shared bucket for every request.
    Global(TokenBucket),

    /// One bucket per unique value of `header_name`; `fallback` covers
    /// requests where the header is absent or not valid UTF-8, and any
    /// request once the hard cap is reached.
    PerHeader {
        /// Request header whose value keys an independent bucket.
        header_name: String,

        /// Shared bucket for requests missing `header_name`, or once the
        /// hard cap on distinct keys is reached.
        fallback: TokenBucket,

        /// Per-key buckets, created lazily on first use of a given key.
        buckets: DashMap<String, TokenBucket>,
    },
}

impl TokenRateLimitState {
    /// Build the single-global-bucket variant.
    pub(super) fn global(burst: f64) -> Self {
        Self::Global(TokenBucket::new(burst))
    }

    /// Build the per-header-value variant, per `ai#129`.
    pub(super) fn per_header(header_name: String, burst: f64) -> Self {
        Self::PerHeader {
            header_name,
            fallback: TokenBucket::new(burst),
            buckets: DashMap::new(),
        }
    }

    /// Resolve which bucket key a request should use, given the configured
    /// key header (if any) and the request's header map.
    ///
    /// Returns `None` when keying isn't configured at all, or when it is
    /// configured but this particular request's header is absent/not valid
    /// UTF-8 -- both cases route to the shared/global/fallback bucket, but
    /// callers that need to distinguish "no keying configured" from "this
    /// request fell back" can match on `self` directly.
    pub(super) fn resolve_key(&self, headers: &http::HeaderMap) -> Option<String> {
        match self {
            Self::Global(_) => None,
            Self::PerHeader { header_name, .. } => headers
                .get(header_name.as_str())
                .and_then(|v| v.to_str().ok())
                .map(str::to_owned),
        }
    }

    /// Run the given closure against the bucket for `key`, creating a new
    /// per-key bucket on first use (subject to the hard cap) or falling
    /// back to the shared/global bucket when `key` is `None` or the hard
    /// cap is reached.
    ///
    /// `rate`/`burst` are needed here (rather than read off `self`) because
    /// a freshly-created per-key bucket must be seeded with the configured
    /// burst and the eviction pass needs the configured rate, matching
    /// `rate_limit::acquire_per_ip`'s
    /// `or_insert_with(|| TokenBucket::new(self.burst))` pattern.
    #[expect(
        clippy::too_many_arguments,
        reason = "key/rate/burst/now are independent bucket-selection inputs; a wrapper struct would obscure the call site"
    )]
    pub(super) fn with_bucket<R>(
        &self,
        key: Option<&str>,
        rate: f64,
        burst: f64,
        now_nanos: u64,
        f: impl FnOnce(&TokenBucket) -> R,
    ) -> R {
        match self {
            Self::Global(bucket) => f(bucket),
            Self::PerHeader { fallback, buckets, .. } => {
                let Some(key) = key else {
                    return f(fallback);
                };

                Self::maybe_evict(buckets, rate, burst, now_nanos);

                if let Some(bucket) = buckets.get(key) {
                    return f(&bucket);
                }

                if buckets.len() >= HARD_CAP_KEY_ENTRIES {
                    tracing::warn!(
                        entries = buckets.len(),
                        hard_cap = HARD_CAP_KEY_ENTRIES,
                        "token_rate_limit: per-key map hard cap reached, falling back to shared bucket"
                    );
                    return f(fallback);
                }

                let bucket = buckets.entry(key.to_owned()).or_insert_with(|| TokenBucket::new(burst));
                f(&bucket)
            },
        }
    }

    /// Evict stale entries once the map exceeds [`MAX_KEY_ENTRIES`].
    ///
    /// Identical shape and thresholds to `rate_limit::maybe_evict`,
    /// generalized from `IpAddr` keys to `String` keys: a bucket is idle
    /// (and reclaimable) once it's been untouched for long enough to have
    /// fully refilled twice over.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "rate/burst nanos, matches rate_limit::maybe_evict"
    )]
    fn maybe_evict(buckets: &DashMap<String, TokenBucket>, rate: f64, burst: f64, now_nanos: u64) {
        if buckets.len() <= MAX_KEY_ENTRIES {
            return;
        }

        let idle_threshold_nanos = (2.0 * burst / rate * 1_000_000_000.0) as u64;
        let aggressive = buckets.len() > AGGRESSIVE_EVICTION_THRESHOLD;
        let scan_limit = if aggressive { usize::MAX } else { EVICTION_SCAN_LIMIT };
        let mut scanned = 0_usize;
        let mut evicted = 0_usize;

        buckets.retain(|_key, bucket| {
            if scanned >= scan_limit {
                return true;
            }
            scanned += 1;
            if now_nanos.saturating_sub(bucket.last_refill_nanos()) > idle_threshold_nanos {
                evicted += 1;
                return false;
            }
            true
        });

        if evicted > 0 {
            tracing::debug!(
                evicted,
                scanned,
                remaining = buckets.len(),
                aggressive,
                "token_rate_limit: evicted stale per-key entries"
            );
        }
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, reason = "tests")]
mod tests {
    use http::{HeaderMap, HeaderValue};

    use super::*;

    fn headers_with(name: &str, value: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            http::header::HeaderName::from_bytes(name.as_bytes()).unwrap(),
            HeaderValue::from_str(value).unwrap(),
        );
        headers
    }

    #[test]
    fn global_state_resolves_no_key() {
        let state = TokenRateLimitState::global(10.0);
        assert_eq!(state.resolve_key(&HeaderMap::new()), None);
    }

    #[test]
    fn per_header_state_resolves_key_from_header() {
        let state = TokenRateLimitState::per_header("x-app-id".to_owned(), 10.0);
        let headers = headers_with("x-app-id", "odin");
        assert_eq!(state.resolve_key(&headers).as_deref(), Some("odin"));
    }

    #[test]
    fn per_header_state_resolves_none_when_header_absent() {
        let state = TokenRateLimitState::per_header("x-app-id".to_owned(), 10.0);
        assert_eq!(state.resolve_key(&HeaderMap::new()), None);
    }

    #[test]
    fn with_bucket_creates_independent_buckets_per_key() {
        let state = TokenRateLimitState::per_header("x-app-id".to_owned(), 10.0);

        // odin exhausts its own bucket...
        state.with_bucket(Some("odin"), 0.0, 10.0, 0, |b| b.try_reserve(10.0, 0.0, 10.0, 0));
        assert!(
            state
                .with_bucket(Some("odin"), 0.0, 10.0, 0, |b| b.try_reserve(1.0, 0.0, 10.0, 0))
                .is_none(),
            "odin should be exhausted"
        );

        // ...but thor's independent bucket is untouched.
        assert!(
            state
                .with_bucket(Some("thor"), 0.0, 10.0, 0, |b| b.try_reserve(10.0, 0.0, 10.0, 0))
                .is_some(),
            "thor's bucket must be independent of odin's"
        );
    }

    #[test]
    fn with_bucket_falls_back_to_shared_bucket_when_key_is_none() {
        let state = TokenRateLimitState::per_header("x-app-id".to_owned(), 5.0);

        assert_eq!(
            state.with_bucket(None, 0.0, 5.0, 0, |b| b.try_reserve(5.0, 0.0, 5.0, 0)),
            Some(0.0)
        );
        assert!(
            state
                .with_bucket(None, 0.0, 5.0, 0, |b| b.try_reserve(1.0, 0.0, 5.0, 0))
                .is_none(),
            "fallback bucket should be shared across all keyless requests"
        );
    }

    #[test]
    fn with_bucket_falls_back_to_shared_bucket_above_hard_cap() {
        let state = TokenRateLimitState::per_header("x-app-id".to_owned(), 5.0);

        for i in 0..HARD_CAP_KEY_ENTRIES {
            state
                .buckets_for_test()
                .insert(format!("key-{i}"), TokenBucket::new(5.0));
        }

        // A brand-new key above the hard cap must fall back rather than
        // grow the map further.
        let remaining = state.with_bucket(Some("new-key"), 0.0, 5.0, 0, |b| b.current_tokens(0.0, 5.0, 0));
        assert_eq!(
            remaining, 5.0,
            "should read from the fallback bucket, not create a new entry"
        );
        assert_eq!(
            state.buckets_for_test().len(),
            HARD_CAP_KEY_ENTRIES,
            "map must not grow past the hard cap"
        );
    }

    #[test]
    fn maybe_evict_reclaims_idle_entries_once_over_soft_cap() {
        let state = TokenRateLimitState::per_header("x-app-id".to_owned(), 10.0);

        // Seed one idle entry (last touched at time 0) plus enough filler
        // entries to cross AGGRESSIVE_EVICTION_THRESHOLD, forcing a
        // full-map scan rather than the bounded EVICTION_SCAN_LIMIT partial
        // scan -- DashMap::retain doesn't visit entries in insertion order,
        // so a partial scan can't deterministically guarantee it reaches
        // any one specific key.
        state
            .buckets_for_test()
            .insert("idle-key".to_owned(), TokenBucket::new(10.0));
        for i in 0..=AGGRESSIVE_EVICTION_THRESHOLD {
            state
                .buckets_for_test()
                .insert(format!("filler-{i}"), TokenBucket::new(10.0));
        }

        // rate=10, burst=10 -> idle threshold is 2s. Access far beyond that
        // at a *different* key to trigger the eviction pass without
        // refreshing idle-key's own timestamp.
        let now_far_future = 5_000_000_000; // 5s
        state.with_bucket(Some("idle-key-trigger"), 10.0, 10.0, now_far_future, |b| {
            b.current_tokens(10.0, 10.0, now_far_future)
        });

        assert!(
            state.buckets_for_test().get("idle-key").is_none(),
            "idle-key should have been reclaimed by the full-scan (aggressive) eviction pass"
        );
    }

    impl TokenRateLimitState {
        /// Test-only accessor for the per-key map, to assert on its
        /// contents/size without exposing it outside `#[cfg(test)]`.
        fn buckets_for_test(&self) -> &DashMap<String, TokenBucket> {
            match self {
                Self::PerHeader { buckets, .. } => buckets,
                Self::Global(_) => unreachable!("test fixtures only construct PerHeader state"),
            }
        }
    }
}
