// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Prometheus metrics emitted by `intelligent_route`.
//!
//! Every label value is drawn from a closed vocabulary or from the bounded
//! snapshot contract (cluster and site names), never from client input, so
//! the series count stays bounded by configuration. Candidate `stable_id`
//! values are deliberately absent: a producer may rotate them on every
//! revision and no exporter prunes retired series.

use std::sync::{
    Mutex, PoisonError,
    atomic::{AtomicBool, AtomicU64, Ordering},
};

use metrics::{counter, gauge};

use super::{
    descriptor::{CapabilityKind, RouteCandidate},
    overlay::PickerPolicy,
};

/// Decisions by outcome and capability kind.
pub(crate) const DECISIONS_TOTAL: &str = "praxis_ai_intelligent_route_decisions_total";

/// Fresh selections by cluster, site, and selection mode.
pub(crate) const SELECTED_TOTAL: &str = "praxis_ai_intelligent_route_selected_total";

/// Session-affinity events.
pub(crate) const SESSION_TOTAL: &str = "praxis_ai_intelligent_route_session_total";

/// Candidates in the serving snapshot.
pub(crate) const SNAPSHOT_CANDIDATES: &str = "praxis_ai_intelligent_route_snapshot_candidates";

/// Overlay reload attempts by result.
pub(crate) const SNAPSHOT_RELOADS_TOTAL: &str = "praxis_ai_intelligent_route_snapshot_reloads_total";

/// Entries in the session-affinity tables of every filter instance in this
/// process, including expired entries not yet evicted.
pub(crate) const SESSION_TABLE_ENTRIES: &str = "praxis_ai_intelligent_route_session_table_entries";

/// Serializes snapshot-size publication so a request-path claim and an
/// accepted reload cannot interleave their snapshot load or store with the
/// gauge update that publishes its size.
static SNAPSHOT_CANDIDATES_LOCK: Mutex<()> = Mutex::new(());

/// The slot every instance in this process competes for.
static SNAPSHOT_GAUGE_SLOT: GaugeSlot = GaugeSlot {
    owner: AtomicU64::new(0),
};

/// Source of instance identities for [`SnapshotOwner`].
static NEXT_SNAPSHOT_OWNER: AtomicU64 = AtomicU64::new(1);

/// The holder of one snapshot gauge.
#[derive(Debug, Default)]
pub(crate) struct GaugeSlot {
    /// Identity of the instance holding the gauge; `0` when none does.
    owner: AtomicU64,
}

/// One filter instance's claim on the snapshot-size gauge.
///
/// A pipeline reload builds a replacement instance before deciding whether
/// to serve it, so construction must not publish. The instance that serves
/// requests claims the gauge on its first request, an accepted reload only
/// republishes while its instance still holds the gauge, and dropping the
/// holder clears it so a retired snapshot's size is never reported.
#[derive(Debug)]
pub(crate) struct SnapshotOwner {
    /// This instance's identity.
    id: u64,
    /// Whether this instance has published at least once.
    published: AtomicBool,
    /// The slot this instance competes for.
    slot: &'static GaugeSlot,
}

impl SnapshotOwner {
    /// A claim on the process-wide gauge.
    pub(crate) fn new() -> Self {
        Self::in_slot(&SNAPSHOT_GAUGE_SLOT)
    }

    /// A claim on a slot of its own, so a test can observe ownership without
    /// other instances in the process interfering.
    #[cfg(test)]
    pub(crate) fn isolated() -> Self {
        Self::in_slot(Box::leak(Box::default()))
    }

    /// A claim with a fresh identity on `slot`.
    fn in_slot(slot: &'static GaugeSlot) -> Self {
        Self {
            id: NEXT_SNAPSHOT_OWNER.fetch_add(1, Ordering::Relaxed),
            published: AtomicBool::new(false),
            slot,
        }
    }

    /// Claim the gauge for this instance and publish `count`.
    ///
    /// Called from the request path, so it is a single atomic load once the
    /// claim is settled: an instance republishes only on its first request
    /// or when the gauge has no holder, never on every request.
    pub(crate) fn claim(&self, count: impl FnOnce() -> usize) {
        let owner = self.slot.owner.load(Ordering::Acquire);
        if owner == self.id || (owner != 0 && self.published.load(Ordering::Relaxed)) {
            return;
        }
        let _guard = SNAPSHOT_CANDIDATES_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
        self.slot.owner.store(self.id, Ordering::Release);
        self.published.store(true, Ordering::Relaxed);
        gauge!(SNAPSHOT_CANDIDATES).set(usize_to_f64(count()));
    }

    /// Run `install`, which swaps in a new snapshot and returns its size,
    /// and publish that size if this instance holds the gauge.
    pub(crate) fn publish_after(&self, install: impl FnOnce() -> usize) {
        let _guard = SNAPSHOT_CANDIDATES_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
        let count = install();
        if self.slot.owner.load(Ordering::Acquire) == self.id {
            gauge!(SNAPSHOT_CANDIDATES).set(usize_to_f64(count));
        }
    }
}

impl Drop for SnapshotOwner {
    fn drop(&mut self) {
        let _guard = SNAPSHOT_CANDIDATES_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
        if self
            .slot
            .owner
            .compare_exchange(self.id, 0, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            gauge!(SNAPSHOT_CANDIDATES).set(0.0);
        }
    }
}

/// How a request left the filter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DecisionOutcome {
    /// A fresh candidate was selected.
    Routed,
    /// A session-bound candidate was reused.
    Reused,
    /// The capability was looked up but no admissible candidate exists.
    NoCandidate,
    /// The routing input was present but malformed.
    Invalid,
    /// No routing input was present; the request continued unrouted.
    Skipped,
    /// The path matched `skip_paths`.
    Management,
    /// An earlier filter had already chosen the cluster.
    Preserved,
}

impl DecisionOutcome {
    /// Label value and metadata vocabulary.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Routed => "routed",
            Self::Reused => "reused",
            Self::NoCandidate => "no_candidate",
            Self::Invalid => "invalid",
            Self::Skipped => "skipped",
            Self::Management => "management",
            Self::Preserved => "preserved",
        }
    }
}

/// A session-affinity transition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SessionEvent {
    /// A new binding was stored.
    Bound,
    /// An existing binding selected the candidate.
    Reused,
    /// The bound candidate was gone or ineligible; a fresh pick replaced it.
    Failover,
}

impl SessionEvent {
    /// Label value.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Bound => "bound",
            Self::Reused => "reused",
            Self::Failover => "failover",
        }
    }
}

/// The result of one overlay reload attempt.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReloadResult {
    /// A new snapshot is serving.
    Accepted,
    /// The file or its semantic revision did not change.
    Unchanged,
    /// Validation failed; the previous snapshot is retained.
    Rejected,
    /// The file could not be read; the previous snapshot is retained.
    ReadError,
}

impl ReloadResult {
    /// Label value.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Accepted => "accepted",
            Self::Unchanged => "unchanged",
            Self::Rejected => "rejected",
            Self::ReadError => "read_error",
        }
    }
}

/// Count one decision. `kind` is `None` before a capability lookup exists.
pub(crate) fn record_decision(outcome: DecisionOutcome, kind: Option<CapabilityKind>) {
    let kind = kind.map_or("none", CapabilityKind::as_str);
    counter!(DECISIONS_TOTAL, "outcome" => outcome.as_str(), "kind" => kind).increment(1);
}

/// Count one fresh selection against its cluster and site.
pub(crate) fn record_selected(candidate: &RouteCandidate, mode: PickerPolicy) {
    counter!(
        SELECTED_TOTAL,
        "cluster" => candidate.cluster.to_string(),
        "site" => candidate.site.to_string(),
        "selection_mode" => mode.as_str(),
    )
    .increment(1);
}

/// Count one session-affinity event.
pub(crate) fn record_session_event(event: SessionEvent) {
    counter!(SESSION_TOTAL, "event" => event.as_str()).increment(1);
}

/// Count one overlay reload attempt.
pub(crate) fn record_reload(result: ReloadResult) {
    counter!(SNAPSHOT_RELOADS_TOTAL, "result" => result.as_str()).increment(1);
}

/// Account for entries added to a session table.
pub(crate) fn session_table_grew(count: usize) {
    gauge!(SESSION_TABLE_ENTRIES).increment(usize_to_f64(count));
}

/// Account for entries removed from a session table.
pub(crate) fn session_table_shrank(count: usize) {
    gauge!(SESSION_TABLE_ENTRIES).decrement(usize_to_f64(count));
}

/// Gauge values are `f64`; snapshot and binding counts are bounded far below
/// the exact-integer range.
#[expect(
    clippy::cast_precision_loss,
    reason = "counts are bounded by MAX_CANDIDATES and MAX_BINDINGS"
)]
fn usize_to_f64(count: usize) -> f64 {
    count as f64
}
