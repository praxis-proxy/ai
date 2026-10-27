// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Background TTL reaper tasks for response-store backends.
//!
//! One reaper runs per distinct backend identity (the factory's effective config
//! key, which the backend cache also uses to deduplicate cached backends 1:1, so
//! every backend is swept and none is dropped). It wakes on a fixed interval
//! and deletes records older than the configured TTL in bounded, per-batch
//! committed sweeps. The [`StoreProvisionService`] owns the reapers and
//! reconciles them to the committed generation (see
//! [`crate::store_provision`]); this module only holds the task itself.
//!
//! The task runs on the serving runtime, where the sqlx pool is bound, so it
//! must never spawn a private runtime (unlike the health and circuit helpers).
//!
//! [`StoreProvisionService`]: crate::store_provision::StoreProvisionService

#![cfg(feature = "_store-backend")]

use std::{
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use pingora_core::server::ShutdownWatch;
use praxis_ai_store::{PersistedStateBackend, ReaperPolicy};
use tokio::{task::JoinHandle, time::MissedTickBehavior};
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

/// Upper bound on a single sweep batch's database round-trip.
///
/// Wrapping each batch in a timeout keeps awaiting cancellation or shutdown from
/// hanging on a stuck query: a batch that exceeds this is abandoned and retried
/// on the next interval, with already-committed batches preserved.
const DB_WAIT_BOUND: Duration = Duration::from_secs(30);

/// A running reaper task and the control it answers to.
///
/// The coordinator cancels [`Self::cancel`] and awaits [`Self::join`] to stop a
/// reaper cooperatively (it finishes its current batch first). [`Self::policy`]
/// is the committed policy the task is running, so a reload can tell an
/// unchanged policy (keep the task) from a changed one (hand off).
///
/// [`Self::backend`] is the exact backend handle the task sweeps. A reload that
/// changes only `pool`/`compression` keeps the policy and namespace but rebuilds
/// the backend pool under a new `effective_key`, retiring (closing) the old one.
/// Comparing the backend identity lets the coordinator notice that swap and
/// re-spawn the task on the live backend, instead of silently sweeping a closed
/// pool forever.
pub(crate) struct ReaperHandle {
    /// Cooperative stop signal, checked between batches and in the wait loop.
    pub(crate) cancel: CancellationToken,
    /// Join handle for the spawned task, awaited during a handoff or shutdown.
    pub(crate) join: JoinHandle<()>,
    /// The policy this task is sweeping under.
    pub(crate) policy: ReaperPolicy,
    /// The backend handle this task sweeps, compared by `Arc` identity so a
    /// pool/compression-only reload (new backend, same policy) re-spawns the task.
    pub(crate) backend: Arc<dyn PersistedStateBackend>,
}

/// Current wall-clock time as whole seconds since the Unix epoch.
///
/// A clock before the epoch yields `0`; a value beyond `i64::MAX` saturates.
/// Both are inert for a cutoff comparison against stored `created_at` seconds.
fn now_unix_secs() -> i64 {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs());
    i64::try_from(secs).unwrap_or(i64::MAX)
}

/// TTL as whole seconds, saturating a pathologically large duration.
fn ttl_secs(ttl: Duration) -> i64 {
    i64::try_from(ttl.as_secs()).unwrap_or(i64::MAX)
}

/// Run bounded, cancellation-checked delete batches for one interval tick.
///
/// Loops until a batch deletes fewer than `batch_size` rows (every table
/// exhausted), a batch fails or exceeds [`DB_WAIT_BOUND`], or the sweep is
/// cancelled between batches. Each batch commits on its own inside the backend,
/// so an interrupted sweep keeps its progress and simply resumes next interval.
async fn sweep_once(backend: &dyn PersistedStateBackend, policy: ReaperPolicy, cancel: &CancellationToken) {
    let cutoff = now_unix_secs().saturating_sub(ttl_secs(policy.ttl));
    let batch_size = policy.batch_size.get();
    loop {
        if cancel.is_cancelled() {
            return;
        }
        match tokio::time::timeout(DB_WAIT_BOUND, backend.sweep_expired(cutoff, batch_size)).await {
            Ok(Ok(stats)) => {
                if stats.max_table() < u64::from(batch_size) {
                    return;
                }
            },
            Ok(Err(error)) => {
                warn!(%error, "ttl reaper: sweep batch failed; retrying next interval");
                return;
            },
            Err(_elapsed) => {
                warn!("ttl reaper: sweep batch exceeded the bounded database wait; retrying next interval");
                return;
            },
        }
    }
}

/// Drive one physical store's reaper until cancelled or shut down.
///
/// The first interval tick fires immediately, so a freshly committed policy
/// takes effect at once rather than after one full interval. Missed ticks are
/// skipped so a slow sweep does not queue a backlog of catch-up sweeps. Both the
/// per-reaper [`CancellationToken`] and the service `shutdown` stop the task; it
/// never interrupts an in-flight batch mid-statement.
pub(crate) async fn run_reaper(
    backend: Arc<dyn PersistedStateBackend>,
    policy: ReaperPolicy,
    cancel: CancellationToken,
    mut shutdown: ShutdownWatch,
) {
    let mut interval = tokio::time::interval(policy.sweep_interval);
    interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
    loop {
        tokio::select! {
            _ = interval.tick() => sweep_once(backend.as_ref(), policy, &cancel).await,
            () = cancel.cancelled() => break,
            _ = shutdown.changed() => break,
        }
    }
    debug!("ttl reaper: stopped");
}
