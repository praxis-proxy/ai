// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! [`TokenCache`] — a cache-through, on-demand-refreshed cache for a
//! single short-lived credential (an upstream bearer token, an access
//! token, ...).
//!
//! # Semantics
//!
//! There is no background refresh. Every call to
//! [`TokenCache::get_or_refresh`] checks the cache; if the cached value
//! is still valid it is returned immediately (a shared read lock, no
//! network call). If it is missing or past its safety margin, the
//! caller takes an exclusive lock, checks again in case a concurrent
//! caller already refreshed it while this one was waiting for the lock,
//! and only then calls `fetch` — at most once per group of callers that
//! all observe a stale cache at the same time. A failed fetch is
//! propagated to the caller and nothing is cached, so the next call
//! tries again; there is no server-side retry/backoff loop.

use std::time::{Duration, Instant};

use tokio::sync::RwLock;
use tracing::warn;

// -----------------------------------------------------------------------------
// Margin
// -----------------------------------------------------------------------------

/// Safety margin to subtract from a TTL before treating a cached value
/// as expired, capped at half the TTL so a short-lived credential stays
/// usable for part of its life instead of being cached already-expired.
fn effective_margin(ttl: Duration, margin: Duration) -> Duration {
    margin.min(ttl / 2)
}

// -----------------------------------------------------------------------------
// TokenCache
// -----------------------------------------------------------------------------

/// A cached value and the instant after which it must not be used.
struct Entry<T> {
    /// The cached value itself.
    value: T,

    /// Instant, already adjusted by the cache's margin, after which the
    /// value must not be used.
    expires_at: Instant,
}

/// Return a clone of `entry`'s value if it is still valid, `None`
/// otherwise (missing or past its safety-margin-adjusted expiry).
fn valid_cached_value<T: Clone>(entry: Option<&Entry<T>>) -> Option<T> {
    entry
        .filter(|entry| entry.expires_at > Instant::now())
        .map(|entry| entry.value.clone())
}

/// Cache-through cache for one credential. See the module docs.
pub struct TokenCache<T> {
    /// Safety window subtracted from a fetched value's TTL (capped at
    /// half the TTL) before it is treated as expired.
    margin: Duration,

    /// The cached entry, if any.
    cache: RwLock<Option<Entry<T>>>,
}

impl<T: Clone + Send + Sync> TokenCache<T> {
    /// Build an empty cache. `margin` is the safety window subtracted
    /// from a fetched value's TTL (capped at half the TTL) before it is
    /// treated as expired.
    pub fn new(margin: Duration) -> Self {
        Self {
            margin,
            cache: RwLock::new(None),
        }
    }

    /// Return a cached, valid value — fetching a new one first if the
    /// cache is empty or the cached value is within its safety margin
    /// of expiry.
    ///
    /// At most one concurrent caller ever calls `fetch`: everyone else
    /// who finds the cache stale queues on the exclusive lock behind
    /// the first caller and, once it releases the lock, re-checks and
    /// finds the value that caller just published.
    ///
    /// # Errors
    ///
    /// Returns whatever `fetch` returns on failure. Nothing is cached
    /// in that case, so the next call tries again.
    #[expect(
        clippy::significant_drop_tightening,
        reason = "guard's last use is immediately before each return; an explicit drop() would just \
                   restate that, not shorten the actual hold time"
    )]
    pub async fn get_or_refresh<F, Fut, E>(&self, fetch: F) -> Result<T, E>
    where
        F: FnOnce() -> Fut + Send,
        Fut: Future<Output = Result<(T, Duration), E>> + Send,
    {
        // Fast path: shared read lock, no fetch.
        let fresh = valid_cached_value(self.cache.read().await.as_ref());
        if let Some(value) = fresh {
            return Ok(value);
        }

        // Slow path: exclusive lock, re-check (another caller may have
        // already refreshed it while this one waited for the lock).
        let mut guard = self.cache.write().await;
        let fresh = valid_cached_value(guard.as_ref());
        if let Some(value) = fresh {
            return Ok(value);
        }

        let (value, ttl) = fetch().await?;
        if ttl <= self.margin {
            warn!(
                ttl_secs = ttl.as_secs(),
                margin_secs = self.margin.as_secs(),
                "token TTL is unusually short; validity margin reduced"
            );
        }
        let expires_at = Instant::now() + ttl.saturating_sub(effective_margin(ttl, self.margin));
        *guard = Some(Entry {
            value: value.clone(),
            expires_at,
        });
        Ok(value)
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, reason = "tests")]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};

    use super::{TokenCache, effective_margin};

    #[test]
    fn effective_margin_caps_at_half_ttl() {
        use std::time::Duration;
        assert_eq!(
            effective_margin(Duration::from_secs(3600), Duration::from_secs(30)),
            Duration::from_secs(30)
        );
        assert_eq!(
            effective_margin(Duration::from_secs(40), Duration::from_secs(30)),
            Duration::from_secs(20)
        );
        assert_eq!(
            effective_margin(Duration::from_secs(10), Duration::from_secs(30)),
            Duration::from_secs(5)
        );
    }

    #[tokio::test]
    async fn first_call_fetches_and_caches() {
        let cache: TokenCache<u32> = TokenCache::new(std::time::Duration::from_millis(5));
        let calls = AtomicU32::new(0);

        let value = cache
            .get_or_refresh(|| async {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok::<_, &str>((42_u32, std::time::Duration::from_secs(60)))
            })
            .await
            .expect("fetch must succeed");

        assert_eq!(value, 42);
        assert_eq!(calls.load(Ordering::SeqCst), 1, "fetch must be called exactly once");
    }

    #[tokio::test]
    async fn second_call_within_ttl_does_not_refetch() {
        let cache: TokenCache<u32> = TokenCache::new(std::time::Duration::from_millis(5));
        let calls = AtomicU32::new(0);
        let fetch = || async {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok::<_, &str>((7_u32, std::time::Duration::from_secs(60)))
        };

        let first = cache.get_or_refresh(fetch).await.expect("first fetch must succeed");
        let second = cache.get_or_refresh(fetch).await.expect("second call must succeed");

        assert_eq!(first, 7);
        assert_eq!(second, 7);
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "a still-valid cache must not trigger a second fetch"
        );
    }

    #[tokio::test]
    async fn refetch_after_expiry() {
        let cache: TokenCache<u32> = TokenCache::new(std::time::Duration::from_millis(1));
        let calls = AtomicU32::new(0);
        let fetch = || async {
            let n = calls.fetch_add(1, Ordering::SeqCst) + 1;
            Ok::<_, &str>((n, std::time::Duration::from_millis(5)))
        };

        let first = cache.get_or_refresh(fetch).await.expect("first fetch must succeed");
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        let second = cache.get_or_refresh(fetch).await.expect("second fetch must succeed");

        assert_eq!(first, 1);
        assert_eq!(second, 2, "an expired cache must trigger a fresh fetch");
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn concurrent_calls_with_empty_cache_fetch_exactly_once() {
        let cache: TokenCache<u32> = TokenCache::new(std::time::Duration::from_millis(5));
        let calls = std::sync::Arc::new(AtomicU32::new(0));
        let cache = std::sync::Arc::new(cache);

        let mut handles = Vec::new();
        for _ in 0..10 {
            let cache = std::sync::Arc::clone(&cache);
            let calls = std::sync::Arc::clone(&calls);
            handles.push(tokio::spawn(async move {
                cache
                    .get_or_refresh(|| async move {
                        calls.fetch_add(1, Ordering::SeqCst);
                        // Widen the race window so all 10 tasks are
                        // guaranteed to observe the empty cache before
                        // the first one publishes a value.
                        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                        Ok::<_, &str>((99_u32, std::time::Duration::from_secs(60)))
                    })
                    .await
                    .expect("fetch must succeed")
            }));
        }

        for handle in handles {
            assert_eq!(handle.await.expect("task must not panic"), 99);
        }
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "10 concurrent callers against an empty cache must fetch exactly once"
        );
    }

    #[tokio::test]
    async fn fetch_error_is_propagated_and_not_cached() {
        let cache: TokenCache<u32> = TokenCache::new(std::time::Duration::from_millis(5));
        let calls = AtomicU32::new(0);

        let err = cache
            .get_or_refresh(|| async {
                calls.fetch_add(1, Ordering::SeqCst);
                Err::<(u32, std::time::Duration), _>("boom")
            })
            .await
            .expect_err("failed fetch must propagate the error");

        assert_eq!(err, "boom");
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        // Nothing was cached, so the next call must try again.
        let value = cache
            .get_or_refresh(|| async {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok::<_, &str>((5_u32, std::time::Duration::from_secs(60)))
            })
            .await
            .expect("retry must succeed");
        assert_eq!(value, 5);
        assert_eq!(calls.load(Ordering::SeqCst), 2, "a failed fetch must not be cached");
    }
}
