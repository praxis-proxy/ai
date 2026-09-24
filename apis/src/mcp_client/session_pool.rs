// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Per-execution pool of initialized MCP sessions (#1019).
//!
//! During one logical Responses execution the agentic loop re-enters
//! `openai_mcp_dispatch` once per round. Without pooling, every `tools/call`
//! opens a throwaway Streamable-HTTP transport, runs a full `initialize`
//! handshake, issues one call, then closes the peer. Repeated calls to the same
//! MCP server across consecutive rounds therefore pay one handshake per call.
//!
//! [`McpSessionPool`] keeps the live rmcp session (a
//! [`RunningService`]) after a clean call so a later call with the *same*
//! validated server + credential/header identity can skip the handshake. The
//! pool lives in the request's threaded `RequestExtensions` (alongside
//! `ResponsesState`), so it is:
//!
//! * shared across every agentic round of one execution,
//! * isolated to a single downstream request (one logical execution), and
//! * dropped — cancelling every idle session's worker via rmcp's `DropGuard` — when the request completes, is
//!   cancelled, or the pipeline is reloaded.
//!
//! ## Identity keying
//!
//! Sessions are keyed by the caller's opaque
//! [`target_fingerprint`](crate::openai::responses::mcp_dispatch) — a SHA-256
//! digest over the validated server URL, per-target headers, static
//! authorization, connector id, and the forwarded-header / owner / credential
//! fingerprints — combined with the effective per-call payload limit. The
//! fingerprint contains **no raw secret** (identity values are pre-digested) and
//! deliberately excludes the tool name, so it is per-server, not per-tool. The
//! payload limit is folded in because it is baked permanently into a session's
//! transport at open time yet varies per round with the batch's call count; a
//! round whose limit differs therefore lands on a different key and never reuses
//! a session that would enforce the wrong response-size ceiling. Because the pool
//! is execution-scoped, the request-invariant gateway assertion and scoped
//! identity are constant for every entry, so the key fully determines a session's
//! security context *and* its transport bounds: two calls that differ in any way
//! that changes the effective transport can never collide on a key.
//!
//! ## Reuse safety
//!
//! A session is returned to the pool only after a call *succeeds*. Every
//! `TransportSignal` write in the subrequest transport is on an error path, so a
//! successful call never latches the session's single-shot signal `OnceLock`; a
//! reused session therefore always presents a pristine signal for the next call.
//! Any error on a reused session evicts (closes) it, so a poisoned signal is
//! never observed by a subsequent operation. A reused session's failure is never
//! retried on a fresh session: its delivery is unknown, so retrying could execute
//! a non-idempotent tool twice. rmcp transparently reinitializes and retries the
//! one provably-safe case (a 404 `SessionExpired`, which the server rejected
//! before executing the tool) inside the reused session itself.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
};

use rmcp::{RoleClient, service::RunningService};

use super::subrequest_transport::TransportSignal;

/// Maximum idle sessions retained per identity key.
///
/// Within-round parallel calls to one server open several sessions that all
/// check back in; this bounds how many are kept warm for later rounds. Extras
/// are dropped (cancelled) rather than retained.
pub(crate) const MAX_IDLE_PER_KEY: usize = 8;

/// Maximum idle sessions retained across all keys in one execution.
///
/// A defensive ceiling so a pathological fan-out across many distinct servers
/// cannot retain an unbounded number of live worker tasks for the request's
/// lifetime.
pub(crate) const MAX_TOTAL_IDLE: usize = 64;

/// A live, initialized MCP session ready to serve another `tools/call` without
/// re-running the `initialize` handshake.
pub(crate) struct PooledSession {
    /// The running rmcp client service; derefs to `Peer<RoleClient>`.
    service: RunningService<RoleClient, ()>,
    /// The subrequest transport's typed-error side-channel, read back after a
    /// call to map an oversized/SSRF/rejected exchange to its typed error.
    signal: Arc<OnceLock<TransportSignal>>,
}

impl PooledSession {
    /// Bind a freshly initialized service to its transport signal handle.
    pub(crate) fn new(service: RunningService<RoleClient, ()>, signal: Arc<OnceLock<TransportSignal>>) -> Self {
        Self { service, signal }
    }

    /// The running service (derefs to the rmcp `Peer`) for issuing calls.
    pub(crate) fn service(&self) -> &RunningService<RoleClient, ()> {
        &self.service
    }

    /// The transport signal handle for this session's most recent exchange.
    pub(crate) fn signal(&self) -> &Arc<OnceLock<TransportSignal>> {
        &self.signal
    }

    /// Close the session, awaiting the worker shutdown and best-effort session
    /// `DELETE`, instead of relying on the `DropGuard`'s detached cancellation.
    pub(crate) async fn close(mut self) {
        drop(self.service.close().await);
    }
}

/// A per-execution pool of reusable MCP sessions keyed by validated server +
/// credential/header identity.
///
/// Cloning shares the underlying map (an `Arc` handle), so a cheap clone can be
/// threaded into per-call futures while the canonical copy lives in the
/// request's `RequestExtensions`.
#[derive(Clone)]
pub(crate) struct McpSessionPool {
    /// Idle sessions grouped by identity key; each value is a stack of warm
    /// sessions for one validated server + credential/header identity.
    inner: Arc<Mutex<HashMap<String, Vec<PooledSession>>>>,
}

impl McpSessionPool {
    /// Create an empty pool.
    pub(crate) fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Take an idle session for `key`, transferring exclusive ownership to the
    /// caller. Returns `None` when no session is warm for that identity.
    ///
    /// Exclusive checkout means concurrent same-key calls never share one live
    /// peer: the second caller finds the stack empty and opens its own.
    pub(crate) fn checkout(&self, key: &str) -> Option<PooledSession> {
        let mut map = self.inner.lock().ok()?;
        let stack = map.get_mut(key)?;
        let session = stack.pop();
        if stack.is_empty() {
            map.remove(key);
        }
        session
    }

    /// Return a healthy session for reuse under `key`.
    ///
    /// Sessions beyond [`MAX_IDLE_PER_KEY`] or [`MAX_TOTAL_IDLE`] are dropped
    /// (cancelled via the `DropGuard`) rather than retained, and a poisoned lock
    /// also drops the session — teardown is always bounded.
    pub(crate) fn checkin(&self, key: String, session: PooledSession) {
        let Ok(mut map) = self.inner.lock() else {
            // Poisoned pool: drop the session so its worker is cancelled.
            return;
        };
        let total: usize = map.values().map(Vec::len).sum();
        if total >= MAX_TOTAL_IDLE {
            return;
        }
        let stack = map.entry(key).or_default();
        if stack.len() < MAX_IDLE_PER_KEY {
            stack.push(session);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkout_on_empty_pool_returns_none() {
        let pool = McpSessionPool::new();
        assert!(
            pool.checkout("missing").is_none(),
            "an empty pool must have no session to check out"
        );
    }

    #[test]
    fn pool_clone_shares_backing_map() {
        let pool = McpSessionPool::new();
        let handle = pool.clone();
        assert!(Arc::ptr_eq(&pool.inner, &handle.inner), "clone must share the map");
    }
}
