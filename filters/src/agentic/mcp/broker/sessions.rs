// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Bounded `(gateway session, server)` → backend session mapping.
//!
//! Each key owns one slot whose async mutex serializes the backend handshake,
//! so concurrent first calls for the same pair initialize the backend once.
//! The map holds at most `max_sessions` entries; idle entries expire after
//! `idle_timeout`, and inserting into a full map evicts the least recently
//! used entry. A slot is in use while a caller holds it (a handshake, a
//! lookup, or a forwarded call in flight); in-use slots are never expired or evicted, so two calls
//! for one key cannot end up on two backend sessions. When every slot is in
//! use, a new key is refused. Mappings are in memory and per filter instance, so a config
//! reload or restart starts with an empty map and backends see a fresh
//! handshake.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
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
    /// Backend handshake result.
    pub init: BackendInit,
}

/// One key's handshake serialization point and current session.
#[derive(Debug, Default)]
pub(crate) struct SessionSlot {
    /// Current backend session, if initialized.
    pub state: tokio::sync::Mutex<Option<BoundSession>>,
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

// -----------------------------------------------------------------------------
// BackendSessionStore
// -----------------------------------------------------------------------------

/// Bounded in-memory backend session map.
#[derive(Debug)]
pub(crate) struct BackendSessionStore {
    /// Live entries.
    entries: Mutex<HashMap<BackendSessionKey, Entry>>,
    /// Idle lifetime of an entry.
    idle_timeout: Duration,
    /// Maximum number of entries.
    max_sessions: usize,
}

impl BackendSessionStore {
    /// Create an empty store.
    pub(crate) fn new(max_sessions: usize, idle_timeout: Duration) -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
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
        let mut entries = self.lock();
        if let Some(entry) = entries.get_mut(key) {
            if self.reusable(entry, now) {
                entry.last_used = now;
                let slot = Arc::clone(&entry.slot);
                drop(entries);
                return Some(slot);
            }
            entries.remove(key);
        }
        if !self.make_room(&mut entries, now) {
            return None;
        }
        let slot = Arc::new(SessionSlot::default());
        entries.insert(
            key.clone(),
            Entry {
                last_used: now,
                slot: Arc::clone(&slot),
            },
        );
        drop(entries);
        Some(slot)
    }

    /// Whether `entry` may be returned: in use, or not yet idle.
    fn reusable(&self, entry: &Entry, now: Instant) -> bool {
        entry.in_use() || now.saturating_duration_since(entry.last_used) <= self.idle_timeout
    }

    /// Free one slot when the map is full, purging idle entries first and then
    /// the least recently used entry not in use. Returns `false` when every
    /// slot is in use.
    fn make_room(&self, entries: &mut HashMap<BackendSessionKey, Entry>, now: Instant) -> bool {
        if entries.len() < self.max_sessions {
            return true;
        }
        entries.retain(|_, entry| self.reusable(entry, now));
        if entries.len() < self.max_sessions {
            return true;
        }
        let Some(oldest) = entries
            .iter()
            .filter(|(_, entry)| !entry.in_use())
            .min_by_key(|(_, entry)| entry.last_used)
            .map(|(key, _)| key.clone())
        else {
            return false;
        };
        entries.remove(&oldest);
        true
    }

    /// Drop the mapping for `key` when it still holds `session_id`.
    ///
    /// A slot busy with a concurrent handshake is left alone: that handshake
    /// replaces the session the backend rejected.
    pub(crate) fn invalidate(&self, key: &BackendSessionKey, session_id: &http::HeaderValue) -> bool {
        let mut entries = self.lock();
        let Some(entry) = entries.get(key) else {
            return false;
        };
        let Ok(mut state) = entry.slot.state.try_lock() else {
            return false;
        };
        let matches = state
            .as_ref()
            .and_then(|bound| bound.init.session_id.as_ref())
            .is_some_and(|current| current == session_id);
        if !matches {
            return false;
        }
        *state = None;
        drop(state);
        // A caller holding the slot must keep finding it in the map, or its
        // next handshake would land on a slot no other call can see.
        if !entry.in_use() {
            entries.remove(key);
        }
        true
    }

    /// Remove the mappings of `gateway_session` that were created with
    /// `credential`, returning them so the caller can terminate them.
    ///
    /// Mappings bound to other credentials, uninitialized slots, and slots
    /// busy with a handshake are kept: presenting a gateway session id alone
    /// must not tear down sessions another caller created.
    pub(crate) fn remove_gateway(
        &self,
        gateway_session: &str,
        credential: &CredentialFingerprint,
    ) -> Vec<(usize, BoundSession)> {
        let mut removed = Vec::new();
        self.lock().retain(|key, entry| {
            if &*key.gateway_session != gateway_session {
                return true;
            }
            let Ok(mut state) = entry.slot.state.try_lock() else {
                return true;
            };
            match state.take() {
                Some(bound) if bound.credential == *credential => {
                    removed.push((key.server, bound));
                    drop(state);
                    // Keep a slot a caller still holds; it is now empty, so
                    // that caller runs a fresh handshake on the mapped slot.
                    entry.in_use()
                },
                other => {
                    *state = other;
                    true
                },
            }
        });
        removed
    }

    /// Number of retained mappings.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.lock().len()
    }

    /// Lock the map, recovering from a poisoned lock: entries stay valid
    /// because every mutation completes before any fallible step.
    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<BackendSessionKey, Entry>> {
        self.entries.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
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

    fn bound(session: &str) -> BoundSession {
        BoundSession {
            credential: [0; SHA256_LEN],
            init: BackendInit {
                protocol_version: Box::from("2025-06-18"),
                session_id: Some(HeaderValue::from_str(session).unwrap()),
                endpoint: None,
            },
        }
    }

    #[tokio::test]
    async fn same_key_returns_same_slot() {
        let store = BackendSessionStore::new(4, Duration::from_secs(60));
        let first = store.slot(&key("g", 0)).unwrap();
        *first.state.lock().await = Some(bound("s"));
        let second = store.slot(&key("g", 0)).unwrap();
        assert!(Arc::ptr_eq(&first, &second), "same key must reuse the slot");
        assert!(
            !Arc::ptr_eq(&first, &store.slot(&key("g", 1)).unwrap()),
            "another server must get its own slot"
        );
    }

    #[test]
    fn idle_entry_is_replaced() {
        let store = BackendSessionStore::new(4, Duration::from_secs(10));
        let start = Instant::now();
        let first = Arc::downgrade(&store.slot_at(&key("g", 0), start).unwrap());
        let later = store.slot_at(&key("g", 0), start + Duration::from_secs(11)).unwrap();
        assert!(first.upgrade().is_none(), "idle entry must be replaced");
        assert!(later.state.try_lock().unwrap().is_none(), "replacement must be fresh");
        assert_eq!(store.len(), 1, "replacement must not grow the map");
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
    fn full_store_evicts_least_recently_used() {
        let store = BackendSessionStore::new(2, Duration::from_secs(60));
        let start = Instant::now();
        let a = Arc::downgrade(&store.slot_at(&key("a", 0), start).unwrap());
        store.slot_at(&key("b", 0), start + Duration::from_secs(1));
        store.slot_at(&key("a", 0), start + Duration::from_secs(2));
        store.slot_at(&key("c", 0), start + Duration::from_secs(3));

        assert_eq!(store.len(), 2, "store must stay at capacity");
        assert!(
            Arc::ptr_eq(
                &a.upgrade().unwrap(),
                &store.slot_at(&key("a", 0), start + Duration::from_secs(4)).unwrap()
            ),
            "recently used entry must survive"
        );
        assert_eq!(store.len(), 2, "lookup of a survivor must not grow the map");
    }

    #[test]
    fn full_store_purges_idle_entries_first() {
        let store = BackendSessionStore::new(2, Duration::from_secs(5));
        let start = Instant::now();
        store.slot_at(&key("a", 0), start);
        store.slot_at(&key("b", 0), start);
        store.slot_at(&key("c", 0), start + Duration::from_secs(10));
        assert_eq!(store.len(), 1, "idle entries must be purged before insert");
    }

    #[tokio::test]
    async fn invalidate_requires_matching_session() {
        let store = BackendSessionStore::new(4, Duration::from_secs(60));
        *store.slot(&key("g", 0)).unwrap().state.lock().await = Some(bound("current"));

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

    #[tokio::test]
    async fn invalidate_keeps_held_slot_mapped() {
        let store = BackendSessionStore::new(4, Duration::from_secs(60));
        let held = store.slot(&key("g", 0)).unwrap();
        *held.state.lock().await = Some(bound("s"));

        assert!(
            store.invalidate(&key("g", 0), &HeaderValue::from_static("s")),
            "matching session must invalidate"
        );
        assert!(held.state.lock().await.is_none(), "the session must be cleared");
        assert!(
            Arc::ptr_eq(&held, &store.slot(&key("g", 0)).unwrap()),
            "a held slot must stay mapped so the next handshake is shared"
        );
    }

    #[tokio::test]
    async fn invalidate_skips_busy_slot() {
        let store = BackendSessionStore::new(4, Duration::from_secs(60));
        let slot = store.slot(&key("g", 0)).unwrap();
        let mut guard = slot.state.lock().await;
        *guard = Some(bound("s"));
        assert!(
            !store.invalidate(&key("g", 0), &HeaderValue::from_static("s")),
            "a slot mid-handshake must not be invalidated"
        );
        drop(guard);
    }

    #[tokio::test]
    async fn remove_gateway_returns_initialized_sessions() {
        let store = BackendSessionStore::new(8, Duration::from_secs(60));
        *store.slot(&key("g", 0)).unwrap().state.lock().await = Some(bound("s0"));
        *store.slot(&key("g", 1)).unwrap().state.lock().await = Some(bound("s1"));
        store.slot(&key("g", 2)).unwrap();
        *store.slot(&key("other", 0)).unwrap().state.lock().await = Some(bound("o"));

        let mut removed = store.remove_gateway("g", &[0; SHA256_LEN]);
        removed.sort_by_key(|(server, _)| *server);

        assert_eq!(removed.len(), 2, "only initialized sessions are returned");
        assert_eq!(removed[0].0, 0, "server index must be reported");
        assert_eq!(store.len(), 2, "other gateway sessions and empty slots remain");
    }

    #[tokio::test]
    async fn remove_gateway_keeps_sessions_of_other_credentials() {
        let store = BackendSessionStore::new(8, Duration::from_secs(60));
        *store.slot(&key("g", 0)).unwrap().state.lock().await = Some(bound("s0"));

        let removed = store.remove_gateway("g", &[1; SHA256_LEN]);

        assert!(removed.is_empty(), "another credential must not remove the session");
        assert_eq!(store.len(), 1, "the mapping must remain");
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
