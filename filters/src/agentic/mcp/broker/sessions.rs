// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Bounded `(gateway session, server)` → backend session mapping.
//!
//! Each key owns one slot. A slot's handshake permit serializes the backend
//! handshake, so concurrent first calls for the same pair initialize the
//! backend once; its session state sits behind a short-held lock that is never
//! held across I/O, so invalidation and gateway `DELETE` always apply, even
//! while a handshake is in flight.
//!
//! The map holds at most `max_sessions` entries; idle entries expire after
//! `idle_timeout`, and inserting into a full map evicts the least recently
//! used entry. A slot is in use while a caller holds it (a handshake, a
//! lookup, or a forwarded call in flight); in-use slots are never expired or
//! evicted, so two calls for one key cannot end up on two backend sessions.
//! When every slot is in use, a new key is refused.
//!
//! Sessions dropped by eviction, expiry, or a credential change are queued as
//! retired, with the credentials that created them, so the caller can
//! terminate them off the map lock. Mappings are in memory and per filter
//! instance, so a config reload or restart starts with an empty map and
//! backends see a fresh handshake.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex, MutexGuard},
    time::{Duration, Instant},
};

use http::HeaderMap;
use praxis_ai_apis::hash::{SHA256_LEN, Sha256};

use super::backend::BackendInit;

// -----------------------------------------------------------------------------
// Types
// -----------------------------------------------------------------------------

/// Credential fingerprint a backend session is bound to.
pub(crate) type CredentialFingerprint = [u8; SHA256_LEN];

/// Map key: the client-held gateway session and the configured server index.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct BackendSessionKey {
    /// Client `Mcp-Session-Id` issued by the broker.
    pub gateway_session: Box<str>,
    /// Server index in configuration order.
    pub server: usize,
}

/// An initialized backend session and the credentials that created it.
#[derive(Clone, Debug)]
pub(crate) struct BoundSession {
    /// Credential fingerprint of the handshake's forwarded headers.
    pub credential: CredentialFingerprint,
    /// The handshake's forwarded headers, marked sensitive, kept so the
    /// session can be terminated with the credentials that created it.
    pub credentials: HeaderMap,
    /// Backend handshake result.
    pub init: BackendInit,
}

impl BoundSession {
    /// Bind `init` to the forwarded `headers` that created it.
    pub(crate) fn new(headers: &HeaderMap, init: BackendInit) -> Self {
        let mut credentials = headers.clone();
        for value in credentials.values_mut() {
            value.set_sensitive(true);
        }
        Self {
            credential: credential_fingerprint(headers),
            credentials,
            init,
        }
    }
}

/// A backend session the map no longer references, awaiting termination.
#[derive(Debug)]
pub(crate) struct RetiredSession {
    /// Server index in configuration order.
    pub server: usize,
    /// The retired session.
    pub session: BoundSession,
}

/// Result of publishing a completed handshake.
#[derive(Debug)]
pub(crate) enum Publish {
    /// The session is now current; a session bound to other credentials that
    /// it replaced is returned for termination.
    Published(Option<BoundSession>),
    /// The gateway session was deleted while the handshake ran; the new
    /// session was not published and must be terminated.
    Cancelled(BoundSession),
}

/// A handshake in flight for one slot.
#[derive(Debug)]
struct Pending {
    /// Credential fingerprint the handshake runs with.
    credential: CredentialFingerprint,
    /// Set by a gateway `DELETE` from the same credentials.
    cancelled: bool,
}

/// Session state of one slot.
#[derive(Debug, Default)]
struct SlotState {
    /// Current backend session, if initialized.
    bound: Option<BoundSession>,
    /// Handshake in flight, if any.
    pending: Option<Pending>,
}

/// One key's handshake serialization point and current session.
#[derive(Debug)]
pub(crate) struct SessionSlot {
    /// Serializes handshakes: one permit, held across handshake I/O.
    permit: tokio::sync::Semaphore,
    /// Session state; held only for short updates, never across I/O.
    state: Mutex<SlotState>,
}

impl Default for SessionSlot {
    fn default() -> Self {
        Self {
            permit: tokio::sync::Semaphore::new(1),
            state: Mutex::default(),
        }
    }
}

impl SessionSlot {
    /// The current session when it was created with `credential`.
    pub(crate) fn current(&self, credential: &CredentialFingerprint) -> Option<BackendInit> {
        self.lock_state()
            .bound
            .as_ref()
            .filter(|bound| bound.credential == *credential)
            .map(|bound| bound.init.clone())
    }

    /// Wait for the handshake permit.
    pub(crate) async fn handshake_permit(&self) -> Result<tokio::sync::SemaphorePermit<'_>, tokio::sync::AcquireError> {
        self.permit.acquire().await
    }

    /// Record that a handshake with `credential` started. Call while holding
    /// the handshake permit.
    pub(crate) fn begin(&self, credential: CredentialFingerprint) {
        self.lock_state().pending = Some(Pending {
            credential,
            cancelled: false,
        });
    }

    /// Record that the started handshake failed.
    pub(crate) fn abort(&self) {
        self.lock_state().pending = None;
    }

    /// Publish a completed handshake unless a gateway `DELETE` cancelled it.
    pub(crate) fn publish(&self, session: BoundSession) -> Publish {
        let mut state = self.lock_state();
        if state.pending.take().is_some_and(|pending| pending.cancelled) {
            return Publish::Cancelled(session);
        }
        let replaced = state.bound.replace(session);
        drop(state);
        Publish::Published(replaced)
    }

    /// Lock the session state, recovering from a poisoned lock: every
    /// mutation completes before any fallible step.
    fn lock_state(&self) -> MutexGuard<'_, SlotState> {
        self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// Map entry with LRU bookkeeping.
#[derive(Debug)]
struct Entry {
    /// Last lookup time.
    last_used: Instant,
    /// Shared slot.
    slot: Arc<SessionSlot>,
}

impl Entry {
    /// Whether a caller still holds this slot. Callers clone the slot under
    /// the map lock, so the count cannot rise while eviction inspects it.
    fn in_use(&self) -> bool {
        Arc::strong_count(&self.slot) > 1
    }
}

/// State behind the map lock.
#[derive(Debug)]
struct Inner {
    /// Live entries.
    entries: HashMap<BackendSessionKey, Entry>,
    /// Last full idle sweep.
    last_sweep: Instant,
    /// Sessions dropped from the map, awaiting termination.
    retired: Vec<RetiredSession>,
}

impl Inner {
    /// Queue the session of a removed `entry` for termination.
    fn retire(&mut self, server: usize, entry: &Entry) {
        let session = entry.slot.lock_state().bound.take();
        if let Some(session) = session {
            self.retired.push(RetiredSession { server, session });
        }
    }
}

// -----------------------------------------------------------------------------
// BackendSessionStore
// -----------------------------------------------------------------------------

/// Bounded in-memory backend session map.
#[derive(Debug)]
pub(crate) struct BackendSessionStore {
    /// Entries, sweep bookkeeping, and the retired queue.
    inner: Mutex<Inner>,
    /// Idle lifetime of an entry.
    idle_timeout: Duration,
    /// Maximum number of entries.
    max_sessions: usize,
}

impl BackendSessionStore {
    /// Create an empty store.
    pub(crate) fn new(max_sessions: usize, idle_timeout: Duration) -> Self {
        Self {
            inner: Mutex::new(Inner {
                entries: HashMap::new(),
                last_sweep: Instant::now(),
                retired: Vec::new(),
            }),
            idle_timeout,
            max_sessions,
        }
    }

    /// Return the slot for `key`, creating a fresh one when absent or idle.
    ///
    /// Returns `None` when the map is full and every slot is in use.
    pub(crate) fn slot(&self, key: &BackendSessionKey) -> Option<Arc<SessionSlot>> {
        self.slot_at(key, Instant::now())
    }

    /// [`slot`](Self::slot) with an explicit clock for tests.
    fn slot_at(&self, key: &BackendSessionKey, now: Instant) -> Option<Arc<SessionSlot>> {
        let mut inner = self.lock();
        if now.saturating_duration_since(inner.last_sweep) >= self.idle_timeout {
            inner.last_sweep = now;
            self.purge_idle(&mut inner, now);
        }
        if let Some(entry) = inner.entries.get_mut(key) {
            if self.reusable(entry, now) {
                // Overlapping callers may sample the clock out of order.
                entry.last_used = entry.last_used.max(now);
                let slot = Arc::clone(&entry.slot);
                drop(inner);
                return Some(slot);
            }
            if let Some(stale) = inner.entries.remove(key) {
                inner.retire(key.server, &stale);
            }
        }
        if !self.make_room(&mut inner, now) {
            return None;
        }
        let slot = Arc::new(SessionSlot::default());
        inner.entries.insert(
            key.clone(),
            Entry {
                last_used: now,
                slot: Arc::clone(&slot),
            },
        );
        drop(inner);
        Some(slot)
    }

    /// Whether `entry` may be returned: in use, or not yet idle.
    fn reusable(&self, entry: &Entry, now: Instant) -> bool {
        entry.in_use() || now.saturating_duration_since(entry.last_used) <= self.idle_timeout
    }

    /// Remove and retire every idle entry not in use.
    fn purge_idle(&self, inner: &mut Inner, now: Instant) {
        let idle: Vec<BackendSessionKey> = inner
            .entries
            .iter()
            .filter(|(_, entry)| !self.reusable(entry, now))
            .map(|(key, _)| key.clone())
            .collect();
        for key in idle {
            if let Some(entry) = inner.entries.remove(&key) {
                inner.retire(key.server, &entry);
            }
        }
    }

    /// Free one slot when the map is full, purging idle entries first and then
    /// the least recently used entry not in use. Returns `false` when every
    /// slot is in use.
    fn make_room(&self, inner: &mut Inner, now: Instant) -> bool {
        if inner.entries.len() < self.max_sessions {
            return true;
        }
        self.purge_idle(inner, now);
        if inner.entries.len() < self.max_sessions {
            return true;
        }
        let Some(oldest) = inner
            .entries
            .iter()
            .filter(|(_, entry)| !entry.in_use())
            .min_by_key(|(_, entry)| entry.last_used)
            .map(|(key, _)| key.clone())
        else {
            return false;
        };
        if let Some(entry) = inner.entries.remove(&oldest) {
            inner.retire(oldest.server, &entry);
        }
        true
    }

    /// Release a call's hold on `slot`. The idle interval of its entry starts
    /// now, so a long call does not leave the session looking idle.
    pub(crate) fn release(&self, key: &BackendSessionKey, slot: Arc<SessionSlot>) {
        self.release_at(key, slot, Instant::now());
    }

    /// [`Self::release`] at a given instant.
    fn release_at(&self, key: &BackendSessionKey, slot: Arc<SessionSlot>, now: Instant) {
        let mut inner = self.lock();
        if let Some(entry) = inner.entries.get_mut(key)
            && Arc::ptr_eq(&entry.slot, &slot)
        {
            entry.last_used = entry.last_used.max(now);
        }
        // Dropped under the map lock, so eviction never sees the entry idle
        // with a stale timestamp.
        drop(slot);
        drop(inner);
    }

    /// Queue a session dropped outside the map (a credential change replaced
    /// it) for termination.
    pub(crate) fn push_retired(&self, retired: RetiredSession) {
        self.lock().retired.push(retired);
    }

    /// Take the sessions dropped from the map since the last call.
    pub(crate) fn take_retired(&self) -> Vec<RetiredSession> {
        std::mem::take(&mut self.lock().retired)
    }

    /// Drop the mapping for `key` when it still holds `session_id`.
    ///
    /// Applies even while a handshake is in flight: that handshake publishes
    /// its own session afterwards.
    pub(crate) fn invalidate(&self, key: &BackendSessionKey, session_id: &http::HeaderValue) -> bool {
        let mut inner = self.lock();
        let Some(entry) = inner.entries.get(key) else {
            return false;
        };
        let mut state = entry.slot.lock_state();
        let matches = state
            .bound
            .as_ref()
            .and_then(|bound| bound.init.session_id.as_ref())
            .is_some_and(|current| current == session_id);
        if !matches {
            return false;
        }
        state.bound = None;
        let idle = state.pending.is_none();
        drop(state);
        // A caller holding the slot must keep finding it in the map, or its
        // next handshake would land on a slot no other call can see.
        if idle && !entry.in_use() {
            inner.entries.remove(key);
        }
        drop(inner);
        true
    }

    /// Remove the mappings of `gateway_session` that were created with
    /// `credential`, returning them so the caller can terminate them, and
    /// cancel that caller's handshakes still in flight so they terminate
    /// their session instead of publishing it.
    ///
    /// Mappings and handshakes of other credentials are kept: presenting a
    /// gateway session id alone must not tear down sessions another caller
    /// created.
    pub(crate) fn remove_gateway(
        &self,
        gateway_session: &str,
        credential: &CredentialFingerprint,
    ) -> Vec<RetiredSession> {
        let mut removed = Vec::new();
        self.lock().entries.retain(|key, entry| {
            if &*key.gateway_session != gateway_session {
                return true;
            }
            let mut state = entry.slot.lock_state();
            if let Some(pending) = state.pending.as_mut()
                && pending.credential == *credential
            {
                pending.cancelled = true;
            }
            if state.bound.as_ref().is_none_or(|bound| bound.credential != *credential) {
                return true;
            }
            if let Some(session) = state.bound.take() {
                removed.push(RetiredSession {
                    server: key.server,
                    session,
                });
            }
            let pending = state.pending.is_some();
            drop(state);
            // Keep a slot a caller still holds; it is now empty, so that
            // caller runs a fresh handshake on the mapped slot.
            pending || entry.in_use()
        });
        removed
    }

    /// Number of retained mappings.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.lock().entries.len()
    }

    /// Lock the map, recovering from a poisoned lock: entries stay valid
    /// because every mutation completes before any fallible step.
    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// Fingerprint the forwarded credential headers a handshake carries.
pub(crate) fn credential_fingerprint(headers: &HeaderMap) -> CredentialFingerprint {
    let mut names: Vec<&http::HeaderName> = headers.keys().collect();
    names.sort_unstable_by(|a, b| a.as_str().cmp(b.as_str()));
    let mut hasher = Sha256::new();
    for name in names {
        for value in headers.get_all(name) {
            hasher.update(name.as_str().as_bytes());
            hasher.update(&[0]);
            hasher.update(value.as_bytes());
            hasher.update(&[0]);
        }
    }
    hasher.finish()
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing, reason = "tests")]
mod tests {
    use http::HeaderValue;

    use super::*;

    fn key(gateway: &str, server: usize) -> BackendSessionKey {
        BackendSessionKey {
            gateway_session: Box::from(gateway),
            server,
        }
    }

    fn bound_with(session: &str, credential: u8) -> BoundSession {
        BoundSession {
            credential: [credential; SHA256_LEN],
            credentials: HeaderMap::new(),
            init: BackendInit {
                protocol_version: Box::from("2025-06-18"),
                session_id: Some(HeaderValue::from_str(session).unwrap()),
                endpoint: None,
            },
        }
    }

    fn bound(session: &str) -> BoundSession {
        bound_with(session, 0)
    }

    fn set(slot: &SessionSlot, session: BoundSession) {
        slot.lock_state().bound = Some(session);
    }

    fn current_id(slot: &SessionSlot) -> Option<String> {
        slot.lock_state()
            .bound
            .as_ref()
            .and_then(|bound| bound.init.session_id.as_ref())
            .map(|id| id.to_str().unwrap().to_owned())
    }

    fn retired_ids(store: &BackendSessionStore) -> Vec<String> {
        let mut ids: Vec<String> = store
            .take_retired()
            .into_iter()
            .map(|retired| retired.session.init.session_id.unwrap().to_str().unwrap().to_owned())
            .collect();
        ids.sort();
        ids
    }

    #[test]
    fn same_key_returns_same_slot() {
        let store = BackendSessionStore::new(4, Duration::from_secs(60));
        let first = store.slot(&key("g", 0)).unwrap();
        set(&first, bound("s"));
        let second = store.slot(&key("g", 0)).unwrap();
        assert!(Arc::ptr_eq(&first, &second), "same key must reuse the slot");
        assert!(
            !Arc::ptr_eq(&first, &store.slot(&key("g", 1)).unwrap()),
            "another server must get its own slot"
        );
    }

    #[test]
    fn idle_entry_is_replaced_and_retired() {
        let store = BackendSessionStore::new(4, Duration::from_secs(10));
        let start = Instant::now();
        set(&store.slot_at(&key("g", 0), start).unwrap(), bound("old"));
        let later = store.slot_at(&key("g", 0), start + Duration::from_secs(11)).unwrap();
        assert!(current_id(&later).is_none(), "replacement must be fresh");
        assert_eq!(store.len(), 1, "replacement must not grow the map");
        assert_eq!(retired_ids(&store), ["old"], "the expired session must be retired");
    }

    #[test]
    fn idle_sweep_retires_untouched_entries() {
        let store = BackendSessionStore::new(4, Duration::from_secs(10));
        let start = Instant::now();
        set(&store.slot_at(&key("a", 0), start).unwrap(), bound("a"));
        store.slot_at(&key("b", 0), start + Duration::from_secs(21));
        assert_eq!(store.len(), 1, "an idle entry must be swept without a lookup");
        assert_eq!(retired_ids(&store), ["a"], "the swept session must be retired");
    }

    #[test]
    fn idle_entry_in_use_is_kept() {
        let store = BackendSessionStore::new(4, Duration::from_secs(10));
        let start = Instant::now();
        let held = store.slot_at(&key("g", 0), start).unwrap();
        let later = store.slot_at(&key("g", 0), start + Duration::from_secs(11)).unwrap();
        assert!(
            Arc::ptr_eq(&held, &later),
            "a slot in use must outlive its idle timeout"
        );
    }

    #[test]
    fn idle_interval_starts_when_the_last_call_releases() {
        let store = BackendSessionStore::new(4, Duration::from_secs(10));
        let start = Instant::now();
        let held = store.slot_at(&key("g", 0), start).unwrap();
        set(&held, bound("live"));
        store.release_at(&key("g", 0), held, start + Duration::from_secs(20));
        let later = store.slot_at(&key("g", 0), start + Duration::from_secs(25)).unwrap();
        assert_eq!(
            current_id(&later).as_deref(),
            Some("live"),
            "a session released 5s ago must not count as idle"
        );
        assert!(retired_ids(&store).is_empty(), "nothing may be retired");
    }

    #[test]
    fn out_of_order_release_keeps_the_later_time() {
        let store = BackendSessionStore::new(4, Duration::from_secs(10));
        let start = Instant::now();
        let first = store.slot_at(&key("g", 0), start).unwrap();
        let second = store.slot_at(&key("g", 0), start).unwrap();
        set(&first, bound("live"));
        store.release_at(&key("g", 0), second, start + Duration::from_secs(28));
        store.release_at(&key("g", 0), first, start + Duration::from_secs(20));
        let later = store.slot_at(&key("g", 0), start + Duration::from_secs(31)).unwrap();
        assert_eq!(
            current_id(&later).as_deref(),
            Some("live"),
            "an earlier release sample must not move the idle start backward"
        );
    }

    #[test]
    fn eviction_skips_slots_in_use() {
        let store = BackendSessionStore::new(1, Duration::from_secs(60));
        let start = Instant::now();
        let a = store.slot_at(&key("a", 0), start).unwrap();
        assert!(
            store.slot_at(&key("b", 0), start + Duration::from_secs(1)).is_none(),
            "a full store of busy slots must refuse a new key"
        );
        let again = store.slot_at(&key("a", 0), start + Duration::from_secs(2)).unwrap();
        assert!(Arc::ptr_eq(&a, &again), "the busy slot must not have been evicted");
        drop((a, again));
        assert!(
            store.slot_at(&key("b", 0), start + Duration::from_secs(3)).is_some(),
            "a released slot must become evictable"
        );
    }

    #[test]
    fn full_store_evicts_least_recently_used_and_retires_it() {
        let store = BackendSessionStore::new(2, Duration::from_secs(60));
        let start = Instant::now();
        set(&store.slot_at(&key("a", 0), start).unwrap(), bound("a"));
        set(
            &store.slot_at(&key("b", 0), start + Duration::from_secs(1)).unwrap(),
            bound("b"),
        );
        store.slot_at(&key("a", 0), start + Duration::from_secs(2));
        store.slot_at(&key("c", 0), start + Duration::from_secs(3));

        assert_eq!(store.len(), 2, "store must stay at capacity");
        assert_eq!(
            current_id(&store.slot_at(&key("a", 0), start + Duration::from_secs(4)).unwrap()).as_deref(),
            Some("a"),
            "recently used entry must survive"
        );
        assert_eq!(retired_ids(&store), ["b"], "the evicted session must be retired");
    }

    #[test]
    fn invalidate_requires_matching_session() {
        let store = BackendSessionStore::new(4, Duration::from_secs(60));
        set(&store.slot(&key("g", 0)).unwrap(), bound("current"));

        assert!(
            !store.invalidate(&key("g", 0), &HeaderValue::from_static("stale")),
            "a stale session id must not drop a newer session"
        );
        assert_eq!(store.len(), 1, "mapping must remain");
        assert!(
            store.invalidate(&key("g", 0), &HeaderValue::from_static("current")),
            "matching session id must invalidate"
        );
        assert_eq!(store.len(), 0, "an unheld mapping must be removed");
    }

    #[test]
    fn invalidate_keeps_held_slot_mapped() {
        let store = BackendSessionStore::new(4, Duration::from_secs(60));
        let held = store.slot(&key("g", 0)).unwrap();
        set(&held, bound("s"));

        assert!(
            store.invalidate(&key("g", 0), &HeaderValue::from_static("s")),
            "matching session must invalidate"
        );
        assert!(current_id(&held).is_none(), "the session must be cleared");
        assert!(
            Arc::ptr_eq(&held, &store.slot(&key("g", 0)).unwrap()),
            "a held slot must stay mapped so the next handshake is shared"
        );
    }

    #[tokio::test]
    async fn invalidate_applies_during_a_handshake() {
        let store = BackendSessionStore::new(4, Duration::from_secs(60));
        let slot = store.slot(&key("g", 0)).unwrap();
        set(&slot, bound_with("old", 0));
        let permit = slot.handshake_permit().await;
        slot.begin([1; SHA256_LEN]);

        assert!(
            store.invalidate(&key("g", 0), &HeaderValue::from_static("old")),
            "a rejected session must be dropped while another handshake runs"
        );
        slot.abort();
        drop(permit);
        assert!(
            slot.current(&[0; SHA256_LEN]).is_none(),
            "a failed replacement must not resurrect the rejected session"
        );
    }

    #[test]
    fn publish_returns_session_of_other_credentials() {
        let slot = SessionSlot::default();
        set(&slot, bound_with("alice", 0));
        slot.begin([1; SHA256_LEN]);
        let outcome = slot.publish(bound_with("bob", 1));
        assert!(
            matches!(
                &outcome,
                Publish::Published(Some(replaced))
                    if replaced.init.session_id.as_ref().is_some_and(|id| id == "alice")
            ),
            "the replaced session must be handed back for termination: {outcome:?}"
        );
    }

    #[test]
    fn remove_gateway_returns_initialized_sessions() {
        let store = BackendSessionStore::new(8, Duration::from_secs(60));
        set(&store.slot(&key("g", 0)).unwrap(), bound("s0"));
        set(&store.slot(&key("g", 1)).unwrap(), bound("s1"));
        store.slot(&key("g", 2)).unwrap();
        set(&store.slot(&key("other", 0)).unwrap(), bound("o"));

        let mut removed = store.remove_gateway("g", &[0; SHA256_LEN]);
        removed.sort_by_key(|retired| retired.server);

        assert_eq!(removed.len(), 2, "only initialized sessions are returned");
        assert_eq!(removed[0].server, 0, "server index must be reported");
        assert_eq!(store.len(), 2, "other gateway sessions and empty slots remain");
    }

    #[test]
    fn remove_gateway_keeps_sessions_of_other_credentials() {
        let store = BackendSessionStore::new(8, Duration::from_secs(60));
        set(&store.slot(&key("g", 0)).unwrap(), bound("s0"));

        let removed = store.remove_gateway("g", &[1; SHA256_LEN]);

        assert!(removed.is_empty(), "another credential must not remove the session");
        assert_eq!(store.len(), 1, "the mapping must remain");
    }

    #[tokio::test]
    async fn remove_gateway_cancels_a_handshake_in_flight() {
        let store = BackendSessionStore::new(8, Duration::from_secs(60));
        let slot = store.slot(&key("g", 0)).unwrap();
        let permit = slot.handshake_permit().await;
        slot.begin([0; SHA256_LEN]);

        assert!(
            store.remove_gateway("g", &[1; SHA256_LEN]).is_empty(),
            "nothing is published yet"
        );
        assert!(
            matches!(slot.publish(bound_with("other", 0)), Publish::Published(None)),
            "another credential's DELETE must not cancel the handshake"
        );

        slot.begin([0; SHA256_LEN]);
        store.remove_gateway("g", &[0; SHA256_LEN]);
        let outcome = slot.publish(bound_with("late", 0));
        drop(permit);
        assert!(
            matches!(outcome, Publish::Cancelled(_)),
            "a handshake the same caller deleted must not be published"
        );
    }

    #[test]
    fn credential_fingerprint_is_order_independent_and_value_sensitive() {
        let mut a = HeaderMap::new();
        a.insert("authorization", HeaderValue::from_static("Bearer one"));
        a.insert("x-tenant", HeaderValue::from_static("t"));
        let mut b = HeaderMap::new();
        b.insert("x-tenant", HeaderValue::from_static("t"));
        b.insert("authorization", HeaderValue::from_static("Bearer one"));
        assert_eq!(
            credential_fingerprint(&a),
            credential_fingerprint(&b),
            "order must not matter"
        );

        b.insert("authorization", HeaderValue::from_static("Bearer two"));
        assert_ne!(
            credential_fingerprint(&a),
            credential_fingerprint(&b),
            "values must matter"
        );
    }
}
