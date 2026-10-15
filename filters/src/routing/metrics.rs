// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Prometheus metrics emitted by `intelligent_route`.
//!
//! Every label value is drawn from a closed vocabulary or from the bounded
//! snapshot contract (cluster and site names), never from client input, so
//! the series count stays bounded by configuration. Candidate `stable_id`
//! values are deliberately absent: a producer may rotate them on every
//! revision and no exporter prunes retired series.

use std::sync::{Mutex, MutexGuard, PoisonError};

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

/// Serializes snapshot-size publication so a first-request republish cannot
/// overwrite the value an accepted reload published in between.
static SNAPSHOT_CANDIDATES_LOCK: Mutex<()> = Mutex::new(());

/// Hold this across a snapshot load (or store) and the gauge update that
/// publishes its size.
pub(crate) fn lock_snapshot_candidates() -> MutexGuard<'static, ()> {
    SNAPSHOT_CANDIDATES_LOCK.lock().unwrap_or_else(PoisonError::into_inner)
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

/// Publish the serving snapshot's candidate count.
pub(crate) fn set_snapshot_candidates(count: usize) {
    gauge!(SNAPSHOT_CANDIDATES).set(usize_to_f64(count));
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
