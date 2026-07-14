// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Local in-process task route store for A2A task-ownership routing.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex, RwLock},
    time::{Duration, Instant},
};

use praxis_filter::builtins::http::value_safety::contains_control_chars;
use serde_json::Value;

use super::config::TaskRoutingConfig;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Maximum length for stored IDs, matching the existing A2A dynamic-value bound.
const MAX_ID_LEN: usize = 256;

/// Maximum number of task route entries before inserts are rejected.
const MAX_TASK_ROUTES: usize = 50_000;

/// Maximum number of context route entries before inserts are rejected.
const MAX_CONTEXT_ROUTES: usize = 50_000;

/// Minimum interval between proactive eviction sweeps.
const EVICTION_INTERVAL: Duration = Duration::from_secs(30);

// -----------------------------------------------------------------------------
// RouteSource
// -----------------------------------------------------------------------------

/// Which store produced a route match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RouteSource {
    /// Route was resolved from the task-ID store.
    Task,
    /// Route was resolved from the context-ID store.
    Context,
}

impl RouteSource {
    /// String representation for tracing and metadata.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Task => "task",
            Self::Context => "context",
        }
    }
}

// -----------------------------------------------------------------------------
// TaskRoute
// -----------------------------------------------------------------------------

/// A stored mapping from a task (or context) ID to the cluster that owns it.
#[derive(Debug, Clone)]
struct TaskRoute {
    /// Cluster name selected when the task was created.
    cluster: Arc<str>,

    /// When this entry expires and should be treated as a miss.
    expires_at: Instant,
}

// -----------------------------------------------------------------------------
// ExtractedTaskRoute
// -----------------------------------------------------------------------------

/// Task route information extracted from a JSON-RPC response body.
#[derive(Debug, Clone)]
pub(crate) struct ExtractedTaskRoute {
    /// Whether the task is in a terminal state.
    pub terminal: bool,

    /// Task ID from the response.
    pub task_id: String,

    /// Context ID from the response, when present.
    pub context_id: Option<String>,
}

// -----------------------------------------------------------------------------
// LocalTaskRouteStore
// -----------------------------------------------------------------------------

/// In-process task route store backed by `RwLock<HashMap>`.
///
/// Holds locks only for short synchronous map operations.
/// Never held across `.await` boundaries.
pub(crate) struct LocalTaskRouteStore {
    /// Task ID → cluster mappings.
    tasks: RwLock<HashMap<String, TaskRoute>>,

    /// Context ID → cluster mappings.
    ///
    /// Kept as a separate lock so task and context operations are independent
    /// and cannot deadlock each other.
    contexts: RwLock<HashMap<String, TaskRoute>>,

    /// Timestamp of the last proactive task eviction sweep.
    last_task_eviction: Mutex<Instant>,

    /// Timestamp of the last proactive context eviction sweep.
    last_context_eviction: Mutex<Instant>,
}

impl LocalTaskRouteStore {
    /// Create an empty store.
    pub(crate) fn new() -> Self {
        Self {
            tasks: RwLock::new(HashMap::new()),
            contexts: RwLock::new(HashMap::new()),
            last_task_eviction: Mutex::new(Instant::now()),
            last_context_eviction: Mutex::new(Instant::now()),
        }
    }

    // ---- Task routes ----

    /// Look up a cluster by task ID. Returns `None` if absent or expired.
    /// Lazily removes expired entries on miss.
    ///
    /// # Panics
    ///
    /// Panics if the internal lock is poisoned.
    #[expect(clippy::expect_used, reason = "poisoned lock is unrecoverable")]
    pub(crate) fn get_by_task_id(&self, task_id: &str) -> Option<Arc<str>> {
        let expired = {
            let tasks = self.tasks.read().expect("task route store lock poisoned");
            match tasks.get(task_id) {
                Some(r) if Instant::now() < r.expires_at => return Some(Arc::clone(&r.cluster)),
                Some(_) => true,
                None => false,
            }
        };

        if expired {
            let mut tasks = self.tasks.write().expect("task route store lock poisoned");
            // Re-check under write lock: another request may have
            // refreshed this task between the read and write locks.
            if tasks.get(task_id).is_some_and(|r| Instant::now() >= r.expires_at) {
                tasks.remove(task_id);
            }
        }
        None
    }

    /// Store a task route mapping with the given TTL.
    ///
    /// Silently ignores task IDs that fail validation (control chars,
    /// too long) and rejects new inserts when the store is at
    /// [`MAX_TASK_ROUTES`] capacity. Overwrites of existing keys are
    /// always allowed regardless of capacity.
    ///
    /// Periodically sweeps expired entries (at most once per
    /// [`EVICTION_INTERVAL`]) to bound memory growth.
    ///
    /// # Panics
    ///
    /// Panics if the internal lock is poisoned.
    #[expect(clippy::expect_used, reason = "poisoned lock is unrecoverable")]
    pub(crate) fn put(&self, task_id: &str, cluster: &str, ttl: Duration) {
        if !validate_id(task_id) {
            return;
        }

        let route = TaskRoute {
            cluster: Arc::from(cluster),
            expires_at: Instant::now() + ttl,
        };

        let mut tasks = self.tasks.write().expect("task route store lock poisoned");
        maybe_evict_map(&mut tasks, &self.last_task_eviction);

        if tasks.len() >= MAX_TASK_ROUTES && !tasks.contains_key(task_id) {
            tracing::warn!(
                limit = MAX_TASK_ROUTES,
                "task route store: capacity reached, insert rejected"
            );
            return;
        }

        tasks.insert(task_id.to_owned(), route);
    }

    /// Remove a task route immediately (for `terminal_ttl_seconds` == 0).
    ///
    /// # Panics
    ///
    /// Panics if the internal lock is poisoned.
    #[expect(clippy::expect_used, reason = "poisoned lock is unrecoverable")]
    pub(crate) fn remove(&self, task_id: &str) {
        self.tasks
            .write()
            .expect("task route store lock poisoned")
            .remove(task_id);
    }

    // ---- Context routes ----

    /// Look up a cluster by context ID. Returns `None` if absent or expired.
    /// Lazily removes expired entries on miss.
    ///
    /// # Panics
    ///
    /// Panics if the internal lock is poisoned.
    #[expect(clippy::expect_used, reason = "poisoned lock is unrecoverable")]
    pub(crate) fn get_by_context_id(&self, context_id: &str) -> Option<Arc<str>> {
        let expired = {
            let contexts = self.contexts.read().expect("context route store lock poisoned");
            match contexts.get(context_id) {
                Some(r) if Instant::now() < r.expires_at => return Some(Arc::clone(&r.cluster)),
                Some(_) => true,
                None => false,
            }
        };

        if expired {
            let mut contexts = self.contexts.write().expect("context route store lock poisoned");
            if contexts.get(context_id).is_some_and(|r| Instant::now() >= r.expires_at) {
                contexts.remove(context_id);
            }
        }
        None
    }

    /// Store a context route mapping with the given TTL.
    ///
    /// Applies the same ID validation rules as [`Self::put`].
    /// Context routes use a separate capacity limit ([`MAX_CONTEXT_ROUTES`])
    /// and a separate eviction clock from task routes, so the two maps
    /// cannot interfere.
    ///
    /// # Panics
    ///
    /// Panics if the internal lock is poisoned.
    #[expect(clippy::expect_used, reason = "poisoned lock is unrecoverable")]
    pub(crate) fn put_context(&self, context_id: &str, cluster: &str, ttl: Duration) {
        if !validate_id(context_id) {
            return;
        }

        let route = TaskRoute {
            cluster: Arc::from(cluster),
            expires_at: Instant::now() + ttl,
        };

        let mut contexts = self.contexts.write().expect("context route store lock poisoned");
        maybe_evict_map(&mut contexts, &self.last_context_eviction);

        if contexts.len() >= MAX_CONTEXT_ROUTES && !contexts.contains_key(context_id) {
            tracing::warn!(
                limit = MAX_CONTEXT_ROUTES,
                "context route store: capacity reached, insert rejected"
            );
            return;
        }

        contexts.insert(context_id.to_owned(), route);
    }
}

// -----------------------------------------------------------------------------
// Route Resolution Helper
// -----------------------------------------------------------------------------

/// Resolve a route by checking `task_id` first (higher precedence), then
/// `context_id`. Used by the request-side lookup and directly testable
/// to verify task-over-context precedence without method-classification guards.
///
/// Semantics: if `task_id` is `Some`, only the task store is consulted —
/// a miss does **not** fall through to the context store. This mirrors
/// the A2A spec rule that task-routable methods are not context-routable:
/// supplying a `task_id` implies the caller already determined this is a
/// task-routable method, so context routing would be semantically wrong on
/// miss. When `task_id` is `None`, only `context_id` is consulted.
///
/// Returns `None` when neither ID has a live route.
pub(crate) fn attempt_route_lookup(
    store: &LocalTaskRouteStore,
    task_id: Option<&str>,
    context_id: Option<&str>,
) -> Option<(Arc<str>, RouteSource)> {
    if let Some(tid) = task_id {
        // Task lookup only: task-routable methods are not context-routable,
        // so a miss here should not silently consult the context store.
        return store.get_by_task_id(tid).map(|c| (c, RouteSource::Task));
    }
    if let Some(cid) = context_id
        && let Some(cluster) = store.get_by_context_id(cid)
    {
        return Some((cluster, RouteSource::Context));
    }
    None
}

// -----------------------------------------------------------------------------
// Response Extraction
// -----------------------------------------------------------------------------

/// Extract task route information from a parsed JSON-RPC response.
///
/// Supports these [A2A core object] response/stream shapes:
/// - `result.task.id` — full `Task` nested under result
/// - `result.id` with `result.status` — direct `Task` object in result
/// - `result.statusUpdate.taskId` — `TaskStatusUpdateEvent` (carries terminal state)
/// - `result.artifactUpdate.taskId` — `TaskArtifactUpdateEvent` (never terminal)
///
/// Returns `None` for message-only responses or malformed JSON.
///
/// [A2A core object]: https://a2a-protocol.org/latest/specification/#5-core-objects
pub(crate) fn extract_task_route(value: &Value) -> Option<ExtractedTaskRoute> {
    let result = value.get("result")?;

    if let Some(task_obj) = result.get("task") {
        return extract_from_task_object(task_obj);
    }

    if result.get("id").is_some() && result.get("status").is_some() {
        return extract_from_task_object(result);
    }

    if let Some(status_update) = result.get("statusUpdate") {
        return extract_from_status_update(status_update);
    }

    if let Some(artifact_update) = result.get("artifactUpdate") {
        return extract_from_artifact_update(artifact_update);
    }

    None
}

/// Extract route info from a task object (either `result.task` or `result` itself).
fn extract_from_task_object(task: &Value) -> Option<ExtractedTaskRoute> {
    let task_id = task.get("id")?.as_str()?;

    if !validate_id(task_id) {
        return None;
    }

    let terminal = task
        .get("status")
        .and_then(|s| s.get("state"))
        .and_then(Value::as_str)
        .is_some_and(is_terminal_state);

    let context_id = extract_context_id_from_object(task);

    Some(ExtractedTaskRoute {
        task_id: task_id.to_owned(),
        terminal,
        context_id,
    })
}

/// Extract route info from a `TaskStatusUpdateEvent` (`result.statusUpdate`).
fn extract_from_status_update(update: &Value) -> Option<ExtractedTaskRoute> {
    let task_id = update.get("taskId")?.as_str()?;

    if !validate_id(task_id) {
        return None;
    }

    let terminal = update
        .get("status")
        .and_then(|s| s.get("state"))
        .and_then(Value::as_str)
        .is_some_and(is_terminal_state);

    // A2A v1.0 TaskStatusUpdateEvent includes contextId per spec.
    let context_id = extract_context_id_from_object(update);

    Some(ExtractedTaskRoute {
        task_id: task_id.to_owned(),
        terminal,
        context_id,
    })
}

/// Extract route info from a `TaskArtifactUpdateEvent` (`result.artifactUpdate`).
///
/// Artifact updates carry no status, so they are never terminal.
fn extract_from_artifact_update(update: &Value) -> Option<ExtractedTaskRoute> {
    let task_id = update.get("taskId")?.as_str()?;

    if !validate_id(task_id) {
        return None;
    }

    // A2A v1.0 TaskArtifactUpdateEvent includes contextId per spec.
    let context_id = extract_context_id_from_object(update);

    Some(ExtractedTaskRoute {
        task_id: task_id.to_owned(),
        terminal: false,
        context_id,
    })
}

/// Extract and validate the `contextId` field from a JSON object.
fn extract_context_id_from_object(obj: &Value) -> Option<String> {
    obj.get("contextId")
        .and_then(Value::as_str)
        .filter(|id| validate_id(id))
        .map(str::to_owned)
}

/// Compute the TTL to use for a task route entry.
pub(crate) fn route_ttl(terminal: bool, config: &TaskRoutingConfig) -> Duration {
    if terminal {
        Duration::from_secs(config.terminal_ttl_seconds)
    } else {
        Duration::from_secs(config.ttl_seconds)
    }
}

// -----------------------------------------------------------------------------
// Private Utilities
// -----------------------------------------------------------------------------

/// Whether the given state string represents a terminal task state.
fn is_terminal_state(state: &str) -> bool {
    matches!(
        state,
        "TASK_STATE_COMPLETED"
            | "TASK_STATE_FAILED"
            | "TASK_STATE_CANCELED"
            | "TASK_STATE_REJECTED"
            | "completed"
            | "failed"
            | "canceled"
            | "cancelled"
            | "rejected"
    )
}

/// Whether an ID is safe for storage: no control characters, bounded length.
fn validate_id(id: &str) -> bool {
    !id.is_empty() && id.len() <= MAX_ID_LEN && !contains_control_chars(id)
}

/// Sweep expired entries if [`EVICTION_INTERVAL`] has elapsed.
///
/// Accepts a mutable reference to the map (caller already holds write lock)
/// and a separate eviction timer so task and context evictions are independent.
fn maybe_evict_map(map: &mut HashMap<String, TaskRoute>, last_eviction: &Mutex<Instant>) {
    if let Ok(mut last) = last_eviction.try_lock() {
        if last.elapsed() < EVICTION_INTERVAL {
            return;
        }
        let before = map.len();
        let now = Instant::now();
        map.retain(|_, r| now < r.expires_at);
        let evicted = before.saturating_sub(map.len());
        if evicted > 0 {
            tracing::debug!(evicted, remaining = map.len(), "route store: evicted expired entries");
        }
        *last = Instant::now();
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::disallowed_methods,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used,
    reason = "tests"
)]
mod tests {
    use std::thread::sleep;

    use super::*;

    // ---- Store Tests: Task Routes ----

    #[test]
    fn local_store_put_then_get_task_route() {
        let store = LocalTaskRouteStore::new();
        store.put("task-1", "agent-a", Duration::from_secs(60));

        let cluster = store.get_by_task_id("task-1");
        assert_eq!(
            cluster.as_deref(),
            Some("agent-a"),
            "stored task route should be retrievable"
        );
    }

    #[test]
    fn local_store_expired_task_route_misses_and_removes_entry() {
        let store = LocalTaskRouteStore::new();
        store.put("task-1", "agent-a", Duration::from_millis(50));

        sleep(Duration::from_millis(200));

        let cluster = store.get_by_task_id("task-1");
        assert!(cluster.is_none(), "expired task route should miss");

        let still_present = store.tasks.read().unwrap().contains_key("task-1");
        assert!(!still_present, "expired entry should be lazily removed from the map");
    }

    #[test]
    fn local_store_terminal_zero_ttl_removes_route() {
        let store = LocalTaskRouteStore::new();
        store.put("task-1", "agent-a", Duration::from_secs(60));
        store.remove("task-1");

        assert!(
            store.get_by_task_id("task-1").is_none(),
            "removed task route should miss"
        );
    }

    #[test]
    fn local_store_rejects_control_char_task_id() {
        let store = LocalTaskRouteStore::new();
        let bad_id = "task\n-1";
        store.put(bad_id, "agent-a", Duration::from_secs(60));

        assert!(
            store.get_by_task_id(bad_id).is_none(),
            "task ID with control chars should not be stored"
        );
    }

    #[test]
    fn local_store_rejects_too_long_task_id() {
        let store = LocalTaskRouteStore::new();
        let long_id = "x".repeat(257);
        store.put(&long_id, "agent-a", Duration::from_secs(60));

        assert!(
            store.get_by_task_id(&long_id).is_none(),
            "task ID exceeding 256 bytes should not be stored"
        );
    }

    #[test]
    fn local_store_rejects_insert_at_capacity() {
        let store = LocalTaskRouteStore::new();
        for i in 0..MAX_TASK_ROUTES {
            store.put(&format!("task-{i}"), "agent-a", Duration::from_secs(3600));
        }

        store.put("overflow", "agent-a", Duration::from_secs(3600));
        assert!(
            store.get_by_task_id("overflow").is_none(),
            "insert should be rejected when store is at capacity"
        );
        assert_eq!(
            store.tasks.read().unwrap().len(),
            MAX_TASK_ROUTES,
            "store size should not exceed MAX_TASK_ROUTES"
        );
    }

    #[test]
    fn local_store_allows_overwrite_at_capacity() {
        let store = LocalTaskRouteStore::new();
        for i in 0..MAX_TASK_ROUTES {
            store.put(&format!("task-{i}"), "agent-a", Duration::from_secs(3600));
        }

        store.put("task-0", "agent-b", Duration::from_secs(3600));
        assert_eq!(
            store.get_by_task_id("task-0").as_deref(),
            Some("agent-b"),
            "overwrite of existing key should succeed at capacity"
        );
    }

    #[test]
    fn local_store_eviction_reclaims_expired_entries() {
        let store = LocalTaskRouteStore::new();
        for i in 0..100 {
            store.put(&format!("task-{i}"), "agent-a", Duration::from_millis(50));
        }
        assert_eq!(store.tasks.read().unwrap().len(), 100, "should have 100 entries");

        sleep(Duration::from_millis(200));

        // Force eviction by setting last_eviction far in the past.
        *store.last_task_eviction.lock().unwrap() = Instant::now() - EVICTION_INTERVAL - Duration::from_secs(1);
        store.put("fresh", "agent-b", Duration::from_secs(3600));

        let remaining = store.tasks.read().unwrap().len();
        assert_eq!(
            remaining, 1,
            "eviction should have removed all 100 expired entries, leaving only 'fresh'"
        );
        assert_eq!(
            store.get_by_task_id("fresh").as_deref(),
            Some("agent-b"),
            "fresh entry should be retrievable after eviction"
        );
    }

    #[test]
    fn local_store_replaces_existing_task_route() {
        let store = LocalTaskRouteStore::new();
        store.put("task-1", "agent-a", Duration::from_secs(60));
        store.put("task-1", "agent-b", Duration::from_secs(60));

        let cluster = store.get_by_task_id("task-1");
        assert_eq!(
            cluster.as_deref(),
            Some("agent-b"),
            "later put should replace earlier route"
        );
    }

    // ---- Store Tests: Context Routes ----

    #[test]
    fn local_store_put_then_get_context_route() {
        let store = LocalTaskRouteStore::new();
        store.put_context("ctx-1", "agent-a", Duration::from_secs(60));

        let cluster = store.get_by_context_id("ctx-1");
        assert_eq!(
            cluster.as_deref(),
            Some("agent-a"),
            "stored context route should be retrievable"
        );
    }

    #[test]
    fn local_store_context_route_expiry_misses_and_removes_entry() {
        let store = LocalTaskRouteStore::new();
        store.put_context("ctx-1", "agent-a", Duration::from_millis(50));

        sleep(Duration::from_millis(200));

        let cluster = store.get_by_context_id("ctx-1");
        assert!(cluster.is_none(), "expired context route should miss");

        let still_present = store.contexts.read().unwrap().contains_key("ctx-1");
        assert!(!still_present, "expired context entry should be lazily removed");
    }

    #[test]
    fn local_store_rejects_control_char_context_id() {
        let store = LocalTaskRouteStore::new();
        let bad_id = "ctx\n-1";
        store.put_context(bad_id, "agent-a", Duration::from_secs(60));

        assert!(
            store.get_by_context_id(bad_id).is_none(),
            "context ID with control chars should not be stored"
        );
    }

    #[test]
    fn local_store_rejects_too_long_context_id() {
        let store = LocalTaskRouteStore::new();
        let long_id = "c".repeat(257);
        store.put_context(&long_id, "agent-a", Duration::from_secs(60));

        assert!(
            store.get_by_context_id(&long_id).is_none(),
            "context ID exceeding 256 bytes should not be stored"
        );
    }

    #[test]
    fn local_store_replaces_existing_context_route() {
        let store = LocalTaskRouteStore::new();
        store.put_context("ctx-1", "agent-a", Duration::from_secs(60));
        store.put_context("ctx-1", "agent-b", Duration::from_secs(60));

        assert_eq!(
            store.get_by_context_id("ctx-1").as_deref(),
            Some("agent-b"),
            "later put_context should replace earlier route"
        );
    }

    #[test]
    fn local_store_task_and_context_routes_are_independent() {
        let store = LocalTaskRouteStore::new();
        store.put("task-1", "agent-a", Duration::from_secs(60));
        store.put_context("task-1", "agent-b", Duration::from_secs(60));

        assert_eq!(
            store.get_by_task_id("task-1").as_deref(),
            Some("agent-a"),
            "task store should be independent of context store"
        );
        assert_eq!(
            store.get_by_context_id("task-1").as_deref(),
            Some("agent-b"),
            "context store should be independent of task store"
        );
    }

    #[test]
    fn local_store_capacity_applies_to_context_routes() {
        let store = LocalTaskRouteStore::new();
        for i in 0..MAX_CONTEXT_ROUTES {
            store.put_context(&format!("ctx-{i}"), "agent-a", Duration::from_secs(3600));
        }

        store.put_context("overflow", "agent-a", Duration::from_secs(3600));
        assert!(
            store.get_by_context_id("overflow").is_none(),
            "context insert should be rejected when store is at capacity"
        );
        assert_eq!(
            store.contexts.read().unwrap().len(),
            MAX_CONTEXT_ROUTES,
            "context store size should not exceed MAX_CONTEXT_ROUTES"
        );
    }

    // ---- Precedence Tests ----

    #[test]
    fn attempt_route_lookup_task_wins_over_context() {
        let store = LocalTaskRouteStore::new();
        store.put("task-1", "agent-a", Duration::from_secs(60));
        store.put_context("ctx-1", "agent-b", Duration::from_secs(60));

        let result = attempt_route_lookup(&store, Some("task-1"), Some("ctx-1"));
        assert!(result.is_some(), "should find a route");
        let (cluster, source) = result.unwrap();
        assert_eq!(cluster.as_ref(), "agent-a", "task route should win over context route");
        assert_eq!(source, RouteSource::Task, "source should be task");
    }

    #[test]
    fn attempt_route_lookup_falls_through_to_context_when_no_task_id() {
        let store = LocalTaskRouteStore::new();
        store.put_context("ctx-1", "agent-a", Duration::from_secs(60));

        let result = attempt_route_lookup(&store, None, Some("ctx-1"));
        assert!(result.is_some(), "should find context route");
        let (cluster, source) = result.unwrap();
        assert_eq!(cluster.as_ref(), "agent-a");
        assert_eq!(source, RouteSource::Context);
    }

    #[test]
    fn attempt_route_lookup_task_miss_does_not_fall_through_to_context() {
        let store = LocalTaskRouteStore::new();
        // A live context route exists, but we pass a task_id that misses.
        store.put_context("ctx-1", "agent-a", Duration::from_secs(60));

        // Supplying task_id signals a task-routable method. Task-routable
        // methods are not context-routable, so a task miss must NOT silently
        // consult the context store. In practice both IDs are never set at
        // the same time (method sets are disjoint), but the helper encodes
        // this semantics explicitly so it is directly testable.
        let result = attempt_route_lookup(&store, Some("task-missing"), Some("ctx-1"));
        assert!(
            result.is_none(),
            "task miss should not fall through to context when task_id was supplied"
        );
    }

    #[test]
    fn attempt_route_lookup_both_missing_returns_none() {
        let store = LocalTaskRouteStore::new();
        let result = attempt_route_lookup(&store, Some("no-task"), Some("no-ctx"));
        assert!(result.is_none(), "both absent should return None");
    }

    #[test]
    fn attempt_route_lookup_neither_id_returns_none() {
        let store = LocalTaskRouteStore::new();
        let result = attempt_route_lookup(&store, None, None);
        assert!(result.is_none(), "no IDs should return None");
    }

    // ---- Response Extraction Tests ----

    #[test]
    fn extract_task_route_from_result_task_includes_context_id() {
        let json = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "task": {
                    "id": "task-123",
                    "contextId": "ctx-123",
                    "status": {"state": "TASK_STATE_WORKING"}
                }
            }
        });

        let route = extract_task_route(&json).expect("should extract route");
        assert_eq!(route.task_id, "task-123");
        assert_eq!(
            route.context_id.as_deref(),
            Some("ctx-123"),
            "contextId should be captured"
        );
        assert!(!route.terminal, "TASK_STATE_WORKING is not terminal");
    }

    #[test]
    fn extract_task_route_from_direct_result_includes_context_id() {
        let json = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "id": "task-456",
                "contextId": "ctx-456",
                "status": {"state": "TASK_STATE_COMPLETED"}
            }
        });

        let route = extract_task_route(&json).expect("should extract route");
        assert_eq!(route.task_id, "task-456");
        assert_eq!(
            route.context_id.as_deref(),
            Some("ctx-456"),
            "contextId should be captured"
        );
        assert!(route.terminal, "TASK_STATE_COMPLETED is terminal");
    }

    #[test]
    fn extract_task_route_without_context_id_keeps_existing_task_behavior() {
        let json = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "task": {
                    "id": "task-no-ctx",
                    "status": {"state": "TASK_STATE_WORKING"}
                }
            }
        });

        let route = extract_task_route(&json).expect("should extract route");
        assert_eq!(route.task_id, "task-no-ctx");
        assert!(
            route.context_id.is_none(),
            "missing contextId should leave context_id as None"
        );
        assert!(!route.terminal);
    }

    #[test]
    fn message_only_response_does_not_create_task_or_context_route() {
        let json = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "message": {
                    "messageId": "msg-1",
                    "role": "ROLE_AGENT",
                    "parts": [{"text": "done"}]
                }
            }
        });

        assert!(
            extract_task_route(&json).is_none(),
            "message-only response should not produce a route"
        );
    }

    #[test]
    fn status_update_without_context_keeps_task_only_behavior() {
        let json = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "statusUpdate": {
                    "taskId": "task-su-1",
                    "status": {"state": "TASK_STATE_WORKING"}
                }
            }
        });

        let route = extract_task_route(&json).expect("should extract from statusUpdate");
        assert_eq!(route.task_id, "task-su-1");
        assert!(
            route.context_id.is_none(),
            "statusUpdate without contextId should have None context_id"
        );
        assert!(!route.terminal);
    }

    #[test]
    fn artifact_update_without_context_keeps_task_only_behavior() {
        let json = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "artifactUpdate": {
                    "taskId": "task-au-1",
                    "artifact": {"artifactId": "a1", "parts": []}
                }
            }
        });

        let route = extract_task_route(&json).expect("should extract from artifactUpdate");
        assert_eq!(route.task_id, "task-au-1");
        assert!(
            route.context_id.is_none(),
            "artifactUpdate without contextId should have None context_id"
        );
        assert!(!route.terminal);
    }

    #[test]
    fn status_update_with_context_id_captures_context_route() {
        let json = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "statusUpdate": {
                    "taskId": "task-su-ctx",
                    "contextId": "ctx-su-1",
                    "status": {"state": "TASK_STATE_WORKING"}
                }
            }
        });

        let route = extract_task_route(&json).expect("should extract from statusUpdate");
        assert_eq!(route.task_id, "task-su-ctx");
        assert_eq!(
            route.context_id.as_deref(),
            Some("ctx-su-1"),
            "statusUpdate contextId should be captured"
        );
    }

    #[test]
    fn artifact_update_with_context_id_captures_context_route() {
        let json = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "artifactUpdate": {
                    "taskId": "task-au-ctx",
                    "contextId": "ctx-au-1",
                    "artifact": {"artifactId": "a1", "parts": []}
                }
            }
        });

        let route = extract_task_route(&json).expect("should extract from artifactUpdate");
        assert_eq!(route.task_id, "task-au-ctx");
        assert_eq!(
            route.context_id.as_deref(),
            Some("ctx-au-1"),
            "artifactUpdate contextId should be captured"
        );
    }

    #[test]
    fn context_id_with_control_chars_not_extracted() {
        let json = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "task": {
                    "id": "task-1",
                    "contextId": "ctx\n-bad",
                    "status": {"state": "TASK_STATE_WORKING"}
                }
            }
        });

        let route = extract_task_route(&json).expect("should still extract task");
        assert!(
            route.context_id.is_none(),
            "contextId with control chars should not be extracted"
        );
    }

    #[test]
    fn context_id_too_long_not_extracted() {
        let long_ctx = "c".repeat(257);
        let json = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "task": {
                    "id": "task-1",
                    "contextId": long_ctx,
                    "status": {"state": "TASK_STATE_WORKING"}
                }
            }
        });

        let route = extract_task_route(&json).expect("should still extract task");
        assert!(
            route.context_id.is_none(),
            "contextId exceeding 256 bytes should not be extracted"
        );
    }

    #[test]
    fn invalid_json_response_does_not_error() {
        let json = serde_json::json!({"not": "a valid response"});
        assert!(
            extract_task_route(&json).is_none(),
            "malformed response should return None, not error"
        );
    }

    #[test]
    fn missing_cluster_does_not_store_route() {
        let json = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "task": {
                    "contextId": "ctx-1",
                    "status": {"state": "TASK_STATE_WORKING"}
                }
            }
        });

        assert!(
            extract_task_route(&json).is_none(),
            "task without id should not produce a route"
        );
    }

    #[test]
    fn terminal_state_uses_terminal_ttl() {
        let config = TaskRoutingConfig {
            ttl_seconds: 3600,
            terminal_ttl_seconds: 300,
            ..TaskRoutingConfig::default()
        };

        let ttl = route_ttl(true, &config);
        assert_eq!(ttl, Duration::from_secs(300), "terminal tasks should use terminal TTL");
    }

    #[test]
    fn context_route_uses_normal_ttl_for_terminal_task() {
        let store = LocalTaskRouteStore::new();
        let normal_ttl = Duration::from_millis(500);
        let terminal_ttl = Duration::from_millis(50);

        store.put("task-1", "agent-a", terminal_ttl);
        store.put_context("ctx-1", "agent-a", normal_ttl);

        sleep(Duration::from_millis(200));

        assert!(
            store.get_by_task_id("task-1").is_none(),
            "task route should expire after terminal TTL"
        );
        assert_eq!(
            store.get_by_context_id("ctx-1").as_deref(),
            Some("agent-a"),
            "context route should survive past terminal TTL using normal TTL"
        );
    }

    #[test]
    fn input_required_state_keeps_normal_route_ttl() {
        let json = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "task": {
                    "id": "task-1",
                    "status": {"state": "TASK_STATE_INPUT_REQUIRED"}
                }
            }
        });

        let route = extract_task_route(&json).expect("should extract route");
        assert!(!route.terminal, "TASK_STATE_INPUT_REQUIRED should not be terminal");
    }

    // ---- Streaming Event Extraction Tests ----

    #[test]
    fn extract_from_status_update_event() {
        let json = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "statusUpdate": {
                    "taskId": "task-su-1",
                    "contextId": "ctx-1",
                    "status": {"state": "TASK_STATE_WORKING"}
                }
            }
        });

        let route = extract_task_route(&json).expect("should extract from statusUpdate");
        assert_eq!(route.task_id, "task-su-1");
        assert!(!route.terminal, "TASK_STATE_WORKING is not terminal");
    }

    #[test]
    fn extract_terminal_status_update_event() {
        let json = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "statusUpdate": {
                    "taskId": "task-su-2",
                    "status": {"state": "TASK_STATE_COMPLETED"}
                }
            }
        });

        let route = extract_task_route(&json).expect("should extract from terminal statusUpdate");
        assert_eq!(route.task_id, "task-su-2");
        assert!(
            route.terminal,
            "TASK_STATE_COMPLETED from statusUpdate should be terminal"
        );
    }

    #[test]
    fn extract_from_artifact_update_event() {
        let json = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "artifactUpdate": {
                    "taskId": "task-au-1",
                    "contextId": "ctx-1",
                    "artifact": {
                        "artifactId": "art-1",
                        "parts": [{"text": "chunk"}]
                    }
                }
            }
        });

        let route = extract_task_route(&json).expect("should extract from artifactUpdate");
        assert_eq!(route.task_id, "task-au-1");
        assert!(!route.terminal, "artifactUpdate is never terminal");
    }

    #[test]
    fn status_update_without_task_id_returns_none() {
        let json = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "statusUpdate": {
                    "status": {"state": "TASK_STATE_WORKING"}
                }
            }
        });

        assert!(
            extract_task_route(&json).is_none(),
            "statusUpdate without taskId should return None"
        );
    }

    #[test]
    fn artifact_update_without_task_id_returns_none() {
        let json = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "artifactUpdate": {
                    "artifact": {"parts": []}
                }
            }
        });

        assert!(
            extract_task_route(&json).is_none(),
            "artifactUpdate without taskId should return None"
        );
    }

    #[test]
    fn all_terminal_states_detected() {
        let terminal_states = [
            "TASK_STATE_COMPLETED",
            "TASK_STATE_FAILED",
            "TASK_STATE_CANCELED",
            "TASK_STATE_REJECTED",
            "completed",
            "failed",
            "canceled",
            "cancelled",
            "rejected",
        ];

        for state in terminal_states {
            assert!(is_terminal_state(state), "{state} should be terminal");
        }
    }

    #[test]
    fn non_terminal_states_not_detected() {
        let non_terminal = [
            "TASK_STATE_WORKING",
            "TASK_STATE_INPUT_REQUIRED",
            "TASK_STATE_AUTH_REQUIRED",
            "TASK_STATE_SUBMITTED",
            "working",
            "submitted",
        ];

        for state in non_terminal {
            assert!(!is_terminal_state(state), "{state} should not be terminal");
        }
    }
}
