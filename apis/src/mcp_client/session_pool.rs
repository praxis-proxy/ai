// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Per-execution pool of initialized MCP sessions (#1019).
//!
//! The pool is stored in request extensions so consecutive agentic rounds can
//! reuse an initialized rmcp service. Keys are opaque and include both the
//! dispatch-filter instance and the validated target identity: two dispatchers
//! in one execution can therefore never exchange sessions whose outbound
//! pipeline, timeout, or forwarded-header policy differs.
//!
//! A session's response limit remains immutable. The pool is keyed only by
//! identity, but checkout accepts the required limit and rejects mismatched
//! sessions for explicit background closure. This preserves the transport's
//! fixed response bound without retaining one bucket for every per-round limit.
//!
//! Idle sessions are deliberately scarce: one per identity and sixteen per
//! execution. An idle timer cancels the rmcp worker after one minute, stopping
//! its standalone GET SSE reconnect loop even when model inference is still in
//! progress. Checkout also rejects expired or already-closed services before a
//! tool request is sent, making a fresh connection safe under the at-most-once
//! policy.

use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex, MutexGuard, OnceLock,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use futures::future::join_all;
use rmcp::{RoleClient, service::RunningService};
use tokio::sync::oneshot;

use super::subrequest_transport::{TransportSignal, TransportSignalState};

/// Synchronized ownership claim for one parked session's idle cancellation.
struct IdleTimer {
    /// Wakes the timer task early after checkout or explicit closure.
    disarm: oneshot::Sender<()>,
    /// Exactly one side may claim the parked state: checkout or the timer.
    parked: Arc<AtomicBool>,
}

/// Maximum idle sessions retained per dispatcher + target identity.
pub(crate) const MAX_IDLE_PER_KEY: usize = 1;

/// Maximum idle sessions retained across one logical Responses execution.
pub(crate) const MAX_TOTAL_IDLE: usize = 16;

/// Maximum time a parked session may keep its standalone SSE stream alive.
pub(crate) const MAX_IDLE_AGE: Duration = Duration::from_secs(60);

/// Maximum graceful-shutdown wait during final request teardown.
const MAX_CLOSE_WAIT: Duration = Duration::from_secs(5);

/// Monotonic source for process-local dispatcher namespaces.
static NEXT_POOL_NAMESPACE: AtomicU64 = AtomicU64::new(1);

/// Unique namespace assigned to one `openai_mcp_dispatch` filter instance.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct McpPoolNamespace(u64);

impl McpPoolNamespace {
    /// Allocate a namespace that cannot collide with another live dispatcher.
    pub(crate) fn new() -> Self {
        let id = NEXT_POOL_NAMESPACE.fetch_add(1, Ordering::Relaxed);
        assert!(id != 0, "MCP pool namespace counter exhausted");
        Self(id)
    }
}

/// Opaque identity for one dispatcher's validated MCP target.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct McpPoolKey {
    /// Dispatcher instance whose immutable transport policy owns the session.
    namespace: McpPoolNamespace,
    /// Credential- and target-bound identity computed by dispatch admission.
    target_fingerprint: String,
}

impl McpPoolKey {
    /// Build a reusable key, returning `None` for the fail-closed empty target
    /// fingerprint sentinel.
    pub(crate) fn new(namespace: McpPoolNamespace, target_fingerprint: String) -> Option<Self> {
        (!target_fingerprint.is_empty()).then_some(Self {
            namespace,
            target_fingerprint,
        })
    }
}

/// A live, initialized MCP session ready for another exclusive `tools/call`.
pub(crate) struct PooledSession {
    /// Initialized rmcp client worker.
    service: RunningService<RoleClient, ()>,
    /// Replaceable per-call transport classification slot.
    signal_state: Arc<TransportSignalState>,
    /// Immutable result limit baked into this session's transport.
    payload_limit: usize,
    /// Instant when the session most recently entered the idle pool.
    last_used: Instant,
    /// Guard that disarms the idle cancellation task when taken or closed.
    idle_timer: Option<IdleTimer>,
}

impl PooledSession {
    /// Bind a freshly initialized service to its transport state and immutable
    /// response limit.
    pub(crate) fn new(
        service: RunningService<RoleClient, ()>,
        signal_state: Arc<TransportSignalState>,
        payload_limit: usize,
    ) -> Self {
        Self {
            service,
            signal_state,
            payload_limit,
            last_used: Instant::now(),
            idle_timer: None,
        }
    }

    /// The running rmcp client service used for issuing a tool request.
    pub(crate) fn service(&self) -> &RunningService<RoleClient, ()> {
        &self.service
    }

    /// Start a new call-error generation, isolating this request from any
    /// signal recorded by an earlier exchange on the same session.
    pub(crate) fn begin_call(&self) -> Arc<OnceLock<TransportSignal>> {
        self.signal_state.begin_exchange()
    }

    /// Clear this call's signal generation after rmcp has completed or failed it.
    pub(crate) fn finish_call(&self, signal: &Arc<OnceLock<TransportSignal>>) {
        self.signal_state.finish_exchange(signal);
    }

    /// Return whether the rmcp worker terminated while the session was parked.
    fn is_closed(&self) -> bool {
        self.service.is_closed()
    }

    /// Return whether this session has exceeded the bounded idle lifetime.
    fn is_expired_at(&self, now: Instant) -> bool {
        now.saturating_duration_since(self.last_used) >= MAX_IDLE_AGE
    }

    /// Arm cancellation of the rmcp worker if the session stays parked.
    fn park(&mut self) {
        self.last_used = Instant::now();
        let (disarm, timer_guard) = oneshot::channel();
        let parked = Arc::new(AtomicBool::new(true));
        let timer_parked = Arc::clone(&parked);
        let service_cancellation = self.service.cancellation_token();
        tokio::spawn(async move {
            tokio::select! {
                () = tokio::time::sleep(MAX_IDLE_AGE) => {
                    if timer_parked
                        .compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire)
                        .is_ok()
                    {
                        service_cancellation.cancel();
                    }
                },
                _ = timer_guard => {},
            }
        });
        self.idle_timer = Some(IdleTimer { disarm, parked });
    }

    /// Claim the parked session before its timer does.
    ///
    /// A `false` result means the timer already won and cancellation is pending,
    /// even if [`RunningService::is_closed`] has not observed it yet.
    fn unpark(&mut self) -> bool {
        if let Some(timer) = self.idle_timer.take() {
            let claimed = timer
                .parked
                .compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire)
                .is_ok();
            let _timer_was_closed = timer.disarm.send(()).is_err();
            claimed
        } else {
            true
        }
    }

    /// Close the service and await rmcp's best-effort DELETE up to `timeout`.
    async fn close_with_timeout(mut self, timeout: Duration) {
        let _claimed_before_idle_timeout = self.unpark();
        drop(self.service.close_with_timeout(timeout).await);
    }

    /// Close the service within the normal teardown bound.
    pub(crate) async fn close(self) {
        self.close_with_timeout(MAX_CLOSE_WAIT).await;
    }

    /// Close without waiting past an active tool call's absolute deadline.
    pub(crate) async fn close_before(self, deadline: tokio::time::Instant) {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        self.close_with_timeout(remaining).await;
    }

    #[cfg(test)]
    pub(crate) fn mark_expired(&mut self) {
        self.last_used = Instant::now() - MAX_IDLE_AGE;
    }

    #[cfg(test)]
    /// Simulate the timer atomically claiming cancellation before checkout.
    fn claim_idle_timeout_for_test(&self) {
        assert!(self.idle_timer.is_some(), "parked test session must have an idle timer");
        if let Some(timer) = self.idle_timer.as_ref() {
            assert!(
                timer
                    .parked
                    .compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok(),
                "test idle timer must win the parked-session claim"
            );
        }
    }
}

/// Sessions accepted and rejected by one checkout operation.
pub(crate) struct PoolCheckout {
    /// A compatible live session, when one was available.
    pub(crate) session: Option<PooledSession>,
    /// Stale, closed, or limit-mismatched sessions requiring explicit closure.
    pub(crate) rejected: Vec<PooledSession>,
}

/// Request-scoped reusable MCP sessions.
#[derive(Clone)]
pub(crate) struct McpSessionPool {
    /// Shared idle-session map and cancellation-path cleanup guard.
    inner: Arc<PoolInner>,
}

/// Shared pool state whose final drop is the cancellation-path cleanup guard.
struct PoolInner {
    /// Idle sessions grouped by opaque dispatcher + target identity.
    sessions: Mutex<HashMap<McpPoolKey, Vec<PooledSession>>>,
}

impl Drop for PoolInner {
    fn drop(&mut self) {
        let map = self
            .sessions
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let sessions: Vec<_> = std::mem::take(map).into_values().flatten().collect();
        if sessions.is_empty() {
            return;
        }

        // Normal response completion calls `drain()` and awaits shutdown. If a
        // streamed response is cancelled before its EOS body hook, final pool
        // ownership is dropped instead; keep the same explicit bounded-close
        // behavior when a runtime is still available. Without one, dropping the
        // sessions still cancels their rmcp workers through RunningService.
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            drop(runtime.spawn(close_sessions(sessions)));
        }
    }
}

impl McpSessionPool {
    /// Create an empty pool.
    pub(crate) fn new() -> Self {
        Self {
            inner: Arc::new(PoolInner {
                sessions: Mutex::new(HashMap::new()),
            }),
        }
    }

    /// Take one compatible session and return all unusable entries separately
    /// so the caller can close them outside the synchronous mutex boundary.
    pub(crate) fn checkout(&self, key: &McpPoolKey, payload_limit: usize) -> PoolCheckout {
        let mut map = self.lock();
        let Some(stack) = map.get_mut(key) else {
            return PoolCheckout {
                session: None,
                rejected: Vec::new(),
            };
        };

        let now = Instant::now();
        let mut session = None;
        let mut rejected = Vec::new();
        for mut candidate in std::mem::take(stack) {
            let idle_timer_disarmed = candidate.unpark();
            if session.is_none()
                && idle_timer_disarmed
                && candidate.payload_limit == payload_limit
                && !candidate.is_closed()
                && !candidate.is_expired_at(now)
            {
                session = Some(candidate);
            } else {
                rejected.push(candidate);
            }
        }
        map.remove(key);
        drop(map);
        PoolCheckout { session, rejected }
    }

    /// Return a healthy session. Rejected and superseded sessions are returned
    /// for explicit asynchronous closure by the caller.
    pub(crate) fn checkin(&self, key: McpPoolKey, mut session: PooledSession) -> Vec<PooledSession> {
        let mut map = self.lock();

        let mut rejected = Vec::new();
        if let Some(existing) = map.get_mut(&key) {
            let mut retained = Vec::with_capacity(existing.len());
            for prior in std::mem::take(existing) {
                if prior.payload_limit == session.payload_limit {
                    retained.push(prior);
                } else {
                    rejected.push(prior);
                }
            }
            *existing = retained;
        }

        let total: usize = map.values().map(Vec::len).sum();
        let stack = map.entry(key).or_default();
        if stack.len() >= MAX_IDLE_PER_KEY || total >= MAX_TOTAL_IDLE {
            rejected.push(session);
        } else {
            session.park();
            stack.push(session);
        }
        drop(map);
        rejected
    }

    /// Remove every idle session from the pool and close them concurrently.
    #[cfg(test)]
    pub(crate) async fn drain(&self) {
        let sessions = self.take_all();
        close_sessions(sessions).await;
    }

    /// Remove every idle session and schedule bounded graceful closure.
    pub(crate) fn drain_in_background(&self) {
        close_sessions_in_background(self.take_all());
    }

    #[cfg(test)]
    pub(crate) fn expire_all_for_test(&self) {
        for session in self.lock().values_mut().flatten() {
            session.mark_expired();
        }
    }

    #[cfg(test)]
    /// Simulate every parked timer winning its synchronization race.
    pub(crate) fn claim_all_idle_timeouts_for_test(&self) {
        for session in self.lock().values_mut().flatten() {
            session.claim_idle_timeout_for_test();
        }
    }

    /// Remove every session without awaiting while holding the pool mutex.
    fn take_all(&self) -> Vec<PooledSession> {
        let mut map = self.lock();
        std::mem::take(&mut *map).into_values().flatten().collect()
    }

    /// Recover the map even if an earlier test or caller panicked while holding
    /// the mutex; cleanup must not silently abandon live sessions.
    fn lock(&self) -> MutexGuard<'_, HashMap<McpPoolKey, Vec<PooledSession>>> {
        self.inner
            .sessions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// Explicitly close rejected sessions concurrently.
pub(crate) async fn close_sessions(sessions: Vec<PooledSession>) {
    join_all(sessions.into_iter().map(PooledSession::close)).await;
}

/// Explicitly close rejected sessions in the background.
///
/// Rejected idle sessions have not received the current tool call, so their
/// best-effort DELETE must not consume that call's delivery deadline. Each
/// close remains bounded by [`MAX_CLOSE_WAIT`].
pub(crate) fn close_sessions_in_background(sessions: Vec<PooledSession>) {
    if sessions.is_empty() {
        return;
    }
    drop(tokio::spawn(close_sessions(sessions)));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_fingerprint_has_no_pool_key() {
        assert!(
            McpPoolKey::new(McpPoolNamespace::new(), String::new()).is_none(),
            "ambiguous targets must not construct reusable keys"
        );
    }

    #[test]
    #[expect(clippy::panic, reason = "test setup invariant")]
    fn dispatcher_namespaces_isolate_identical_targets() {
        let target = "same-target".to_owned();
        let Some(first) = McpPoolKey::new(McpPoolNamespace::new(), target.clone()) else {
            panic!("non-empty target must construct a key");
        };
        let Some(second) = McpPoolKey::new(McpPoolNamespace::new(), target) else {
            panic!("non-empty target must construct a key");
        };
        assert_ne!(first, second, "separate dispatch filters must never share sessions");
    }

    #[test]
    fn pool_clone_shares_backing_map() {
        let pool = McpSessionPool::new();
        let handle = pool.clone();
        assert!(Arc::ptr_eq(&pool.inner, &handle.inner), "clone must share the map");
    }
}
