// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Per-owner request rate limiting for local Conversations operations.
//!
//! The pinned OpenAI reference declares a `429 TooManyRequests` response with
//! an `ErrorResponse` body and an optional integer `Retry-After` header on
//! every Conversations operation. This module provides the runtime the
//! contract describes: a fixed-window counter keyed by the trusted
//! [`StateOwner`], mirroring the owner scoping of the Conversations store.
//!
//! The limiter state lives in the filter instance, which the pipeline builds
//! once per configuration. Swapping pipelines on dynamic reload creates a
//! fresh counter set, matching the lifecycle of the per-pipeline
//! [`ResponseStoreRegistry`](crate::store::ResponseStoreRegistry).

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use praxis_filter::Rejection;

use crate::state_owner::StateOwner;

/// Length of one fixed counting window.
const WINDOW: Duration = Duration::from_secs(60);

/// Configured Conversations rate limit.
///
/// A fixed window of [`WINDOW`] shared by all eight Conversations
/// operations, enforced per authenticated owner.
#[derive(Debug, Clone, Copy, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RateLimitConfig {
    /// Maximum Conversations requests one owner may make per minute.
    ///
    /// Must be at least one.
    pub requests_per_minute: u32,
}

impl RateLimitConfig {
    /// Validate the configured values.
    ///
    /// # Errors
    ///
    /// Returns a [`FilterError`] message when `requests_per_minute` is zero.
    pub(crate) fn validate(self) -> Result<(), praxis_filter::FilterError> {
        if self.requests_per_minute == 0 {
            return Err("openai_conversations: rate_limit.requests_per_minute must be at least 1".into());
        }
        Ok(())
    }
}

/// Fixed-window counters for one owner.
#[derive(Debug)]
struct OwnerWindow {
    /// Start of the current counting window.
    window_start: Instant,
    /// Requests already counted in the current window.
    count: u32,
}

/// Owner windows and the last expiration sweep, protected by one lock.
struct WindowState {
    /// Counters retained until their counting windows expire.
    windows: HashMap<StateOwner, OwnerWindow>,
    /// Time of the last sweep, limiting map scans to once per minute.
    last_prune: Instant,
}

impl WindowState {
    /// Remove expired owners without resetting active windows.
    fn prune_expired(&mut self, now: Instant) {
        if now.duration_since(self.last_prune) >= WINDOW {
            self.windows
                .retain(|_, window| now.duration_since(window.window_start) < WINDOW);
            if self.windows.len() < self.windows.capacity() / 4 {
                self.windows.shrink_to_fit();
            }
            self.last_prune = now;
        }
    }
}

/// Shared per-owner fixed-window rate limiter.
pub(crate) struct OwnerRateLimiter {
    /// Configured maximum requests per window.
    requests_per_minute: u32,
    /// Per-owner window state.
    windows: Mutex<WindowState>,
    /// Clock used for window arithmetic; injectable for tests.
    now: Box<dyn Fn() -> Instant + Send + Sync>,
}

impl OwnerRateLimiter {
    /// Build a limiter using the real clock.
    pub(crate) fn new(config: RateLimitConfig) -> Self {
        Self::with_clock(config, Box::new(Instant::now))
    }

    /// Build a limiter with an injectable clock.
    fn with_clock(config: RateLimitConfig, now: Box<dyn Fn() -> Instant + Send + Sync>) -> Self {
        let last_prune = now();
        Self {
            requests_per_minute: config.requests_per_minute,
            windows: Mutex::new(WindowState {
                windows: HashMap::new(),
                last_prune,
            }),
            now,
        }
    }

    /// Lock the window map, surviving a poisoned lock.
    fn lock_windows(&self) -> std::sync::MutexGuard<'_, WindowState> {
        // Window counters are advisory state, not correctness state: a panic
        // elsewhere while holding the lock must not take request handling
        // down with it.
        self.windows.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Count one request for `owner`.
    ///
    /// Returns the number of whole seconds until the window resets when the
    /// owner has exhausted its quota. The value is always at least one,
    /// matching the `Retry-After` schema declared by the contract.
    pub(crate) fn try_acquire(&self, owner: &StateOwner) -> Result<(), u64> {
        let mut state = self.lock_windows();
        // Sample under the lock so concurrent acquires cannot observe an older
        // time than a window installed by another acquire.
        let now = (self.now)();
        state.prune_expired(now);
        let windows = &mut state.windows;
        // Borrow first so steady-state requests never clone the owner key;
        // the clone is confined to the once-per-owner cold path.
        let window = if let Some(window) = windows.get_mut(owner) {
            window
        } else {
            windows.insert(
                owner.clone(),
                OwnerWindow {
                    window_start: now,
                    count: 0,
                },
            );
            windows
                .get_mut(owner)
                .unwrap_or_else(|| unreachable!("window inserted above"))
        };

        if now.duration_since(window.window_start) >= WINDOW {
            window.window_start = now;
            window.count = 0;
        }
        if window.count < self.requests_per_minute {
            window.count += 1;
            return Ok(());
        }
        let remaining = WINDOW.saturating_sub(now.duration_since(window.window_start));
        drop(state);
        let seconds = remaining.as_secs() + u64::from(remaining.subsec_nanos() > 0);
        Err(seconds.max(1))
    }
}

/// Build the `429` rejection the contract declares: an `ErrorResponse` body
/// carrying `type`, `message`, `param`, and `code`, plus the optional
/// `Retry-After` header.
pub(crate) fn reject_rate_limited(retry_after_secs: u64) -> Rejection {
    let body = serde_json::json!({
        "error": {
            "message": format!("Rate limit reached. Please retry after {retry_after_secs} seconds."),
            "type": "rate_limit_error",
            "param": null,
            "code": "rate_limit_exceeded",
        }
    });
    Rejection::status(429)
        .with_header("content-type", "application/json")
        .with_header("retry-after", retry_after_secs.to_string())
        .with_body(serde_json::to_vec(&body).unwrap_or_default())
}

/// Wrap [`OwnerRateLimiter`] for filter-instance storage.
pub(crate) type SharedOwnerRateLimiter = Arc<OwnerRateLimiter>;

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::indexing_slicing, clippy::unwrap_used, reason = "tests")]
mod tests {
    use std::{sync::Arc, time::Duration};

    use super::*;
    use crate::state_owner::StateOwner;

    /// Deterministic clock advanced by hand.
    struct FakeClock {
        now: Mutex<Instant>,
    }

    impl FakeClock {
        fn new() -> Self {
            Self {
                now: Mutex::new(Instant::now()),
            }
        }

        fn instant(&self) -> Instant {
            *self.now.lock().unwrap()
        }

        fn advance(&self, by: Duration) {
            *self.now.lock().unwrap() += by;
        }
    }

    fn owner(id: u32) -> StateOwner {
        StateOwner::from_trusted_parts(format!("tenant-{id}"), format!("issuer-{id}"), format!("subject-{id}")).unwrap()
    }

    fn limiter_with_clock(clock: Arc<FakeClock>, requests_per_minute: u32) -> OwnerRateLimiter {
        OwnerRateLimiter::with_clock(
            RateLimitConfig { requests_per_minute },
            Box::new(move || clock.instant()),
        )
    }

    #[test]
    fn allows_requests_up_to_the_configured_limit() {
        let clock = Arc::new(FakeClock::new());
        let limiter = limiter_with_clock(Arc::clone(&clock), 3);
        let owner = owner(1);
        for _ in 0..3 {
            assert_eq!(limiter.try_acquire(&owner), Ok(()));
        }
    }

    #[test]
    fn rejects_the_next_request_with_remaining_window_seconds() {
        let clock = Arc::new(FakeClock::new());
        let limiter = limiter_with_clock(Arc::clone(&clock), 2);
        let owner = owner(1);
        limiter.try_acquire(&owner).unwrap();
        limiter.try_acquire(&owner).unwrap();

        clock.advance(Duration::from_secs(10));
        // 50 seconds remain in the window.
        assert_eq!(limiter.try_acquire(&owner), Err(50));
    }

    #[test]
    fn retry_after_rounds_up_fractional_seconds() {
        let clock = Arc::new(FakeClock::new());
        let limiter = limiter_with_clock(Arc::clone(&clock), 1);
        let owner = owner(1);
        limiter.try_acquire(&owner).unwrap();
        clock.advance(Duration::from_millis(9_600));
        assert_eq!(
            limiter.try_acquire(&owner),
            Err(51),
            "50.4 remaining seconds must round up"
        );
        clock.advance(Duration::from_millis(49_401));
        assert_eq!(
            limiter.try_acquire(&owner),
            Err(1),
            "a fractional final second must round up"
        );
    }

    #[test]
    fn expiration_sweep_preserves_active_owner_quotas() {
        let clock = Arc::new(FakeClock::new());
        let limiter = limiter_with_clock(Arc::clone(&clock), 1);
        for id in 0..100 {
            limiter.try_acquire(&owner(id)).unwrap();
        }
        clock.advance(Duration::from_secs(30));
        let active = owner(100);
        limiter.try_acquire(&active).unwrap();
        clock.advance(Duration::from_secs(30));
        assert_eq!(
            limiter.try_acquire(&active),
            Err(30),
            "pruning must not reset active quotas"
        );
        assert_eq!(
            limiter.lock_windows().windows.len(),
            1,
            "expired owners must be evicted"
        );
        assert_eq!(
            limiter.try_acquire(&owner(0)),
            Ok(()),
            "an expired owner gets a fresh window"
        );
        clock.advance(Duration::from_secs(1));
        assert_eq!(
            limiter.try_acquire(&active),
            Err(29),
            "requests between sweeps keep their quota"
        );
    }

    #[test]
    fn retry_after_is_never_below_one_second() {
        let clock = Arc::new(FakeClock::new());
        let limiter = limiter_with_clock(Arc::clone(&clock), 1);
        let owner = owner(1);
        limiter.try_acquire(&owner).unwrap();

        clock.advance(Duration::from_secs(59));
        assert_eq!(limiter.try_acquire(&owner), Err(1));
    }

    #[test]
    fn window_resets_after_sixty_seconds() {
        let clock = Arc::new(FakeClock::new());
        let limiter = limiter_with_clock(Arc::clone(&clock), 1);
        let owner = owner(1);
        assert_eq!(limiter.try_acquire(&owner), Ok(()));
        assert_eq!(limiter.try_acquire(&owner), Err(60));

        clock.advance(Duration::from_secs(60));
        assert_eq!(limiter.try_acquire(&owner), Ok(()));
    }

    #[test]
    fn owners_are_limited_independently() {
        let clock = Arc::new(FakeClock::new());
        let limiter = limiter_with_clock(Arc::clone(&clock), 1);
        let first = owner(1);
        let second = owner(2);
        assert_eq!(limiter.try_acquire(&first), Ok(()));
        assert_eq!(limiter.try_acquire(&first), Err(60));
        assert_eq!(limiter.try_acquire(&second), Ok(()));
    }

    #[test]
    fn zero_requests_per_minute_is_rejected_at_config_validation() {
        assert!(RateLimitConfig { requests_per_minute: 0 }.validate().is_err());
        assert!(RateLimitConfig { requests_per_minute: 1 }.validate().is_ok());
    }

    #[test]
    fn rejection_matches_the_declared_contract_shape() {
        let rejection = reject_rate_limited(30);
        assert_eq!(rejection.status, 429);

        let headers: Vec<(&str, &str)> = rejection
            .headers
            .iter()
            .map(|(name, value)| (name.as_str(), value.as_str()))
            .collect();
        assert!(headers.contains(&("content-type", "application/json")));
        assert!(headers.contains(&("retry-after", "30")));

        let body: serde_json::Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();
        let error = &body["error"];
        assert_eq!(error["type"], "rate_limit_error");
        assert_eq!(error["code"], "rate_limit_exceeded");
        assert!(error["param"].is_null());
        assert!(error["message"].as_str().is_some_and(|m| m.contains("30")));
        // The contract's `ErrorResponse` marks exactly `error` required, and
        // its nested `Error` marks exactly these four fields.
        assert_eq!(body.as_object().unwrap().len(), 1);
        assert_eq!(error.as_object().unwrap().len(), 4);
    }
}
