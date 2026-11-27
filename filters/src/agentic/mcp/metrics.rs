// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! MCP protocol metrics emitted by the `mcp` filter.
//!
//! These provide per-tool and per-method observability for MCP traffic
//! routed through Praxis, closing the gap with competing gateways (Kong,
//! Portkey) that ship MCP-specific Prometheus metrics out of the box.
//!
//! # Label cardinality
//!
//! Every label value is drawn from either:
//! - A closed vocabulary (`method` from [`sanitize_method`], `transport_status`
//!   from HTTP status classification), or
//! - Bounded configuration (`tool` names capped by [`ToolAdmission`],
//!   `cluster` from proxy config).
//!
//! Tool names are additionally capped at [`MAX_TOOL_LABELS`] distinct
//! values per process lifetime — values arriving after the cap are labeled
//! `"other"`, matching the cardinality-bounding pattern in
//! [`llm_metrics`](crate::llm_metrics).
//!
//! # Metric rationale
//!
//! - `praxis_mcp_requests_total{method, protocol_version}`: request
//!   volumes by MCP method — capacity planning, method mix visibility.
//! - `praxis_mcp_tool_calls_total{tool, cluster, transport_status}`:
//!   per-tool call counts with HTTP transport outcome — error rate
//!   alerting, usage ranking. Note: reflects HTTP status, not JSON-RPC
//!   application-level success.
//! - `praxis_mcp_tool_call_duration_seconds{tool, cluster}`: upstream
//!   round-trip time per tool — latency SLOs, slow-tool detection.
//! - `praxis_mcp_filter_rejections_total{method, reason}`: requests
//!   rejected by the MCP filter itself — misconfiguration, attacks.

use std::{
    collections::HashSet,
    sync::{Arc, Mutex, Once},
};

use arc_swap::ArcSwap;
use metrics::{Unit, counter, describe_histogram, histogram};
use praxis_ai_apis::promotion::contains_control_chars;

// ---------------------------------------------------------------------------
// Metric names
// ---------------------------------------------------------------------------

/// MCP requests by method and protocol version, cumulative.
pub const METRIC_MCP_REQUESTS_TOTAL: &str = "praxis_mcp_requests_total";

/// Tool call invocations by tool name, upstream cluster, and HTTP transport outcome.
///
/// The `transport_status` label reflects the HTTP status code returned by
/// the upstream MCP server (`"ok"` for 2xx/3xx, `"error"` for 4xx/5xx).
/// It does **not** indicate JSON-RPC application-level success: a tool
/// that returns a JSON-RPC `{"error": ...}` inside an HTTP 200 is recorded
/// as `transport_status="ok"`. Detecting application-level errors requires
/// response body parsing, which is out of scope for this transport-level
/// metric.
pub const METRIC_MCP_TOOL_CALLS_TOTAL: &str = "praxis_mcp_tool_calls_total";

/// Upstream round-trip time for tool calls (seconds), histogram.
pub const METRIC_MCP_TOOL_CALL_DURATION: &str = "praxis_mcp_tool_call_duration_seconds";

/// Requests rejected by the MCP filter (invalid params, header mismatch, etc.).
pub const METRIC_MCP_FILTER_REJECTIONS_TOTAL: &str = "praxis_mcp_filter_rejections_total";

// ---------------------------------------------------------------------------
// Per-instance filter state for cross-phase metrics
// ---------------------------------------------------------------------------

/// Per-request metrics state stored in `filter_state` during the request
/// phase and retrieved in the response phase.
///
/// Using `filter_state` (keyed by `current_filter_id`) instead of shared
/// `filter_metadata` ensures that two MCP filter instances in the same
/// pipeline do not overwrite each other's timing or tool name.
// This PR (#2, audit logging) introduces `AuditState` as a superset of
// `McpMetricsState`, carrying both the metrics fields (start, tool_name)
// and audit fields (method, session_id, protocol_version, tenant).
// All `McpMetricsState` usage in mod.rs is replaced by `AuditState`,
// leaving this struct unused. The struct is retained so the metrics PR
// (#2030) can be reviewed and merged independently. Once both PRs land,
// this struct and annotation can be removed.
#[expect(dead_code, reason = "retained for independent review of metrics PR #2030; audit.rs uses AuditState instead")]
pub(crate) struct McpMetricsState {
    /// Monotonic start time for duration measurement.
    ///
    /// Uses [`std::time::Instant`] instead of wall-clock time so that
    /// system-clock adjustments never produce negative or inflated
    /// histogram values.
    pub start: std::time::Instant,
    /// Tool name extracted from the JSON-RPC `params.name` field.
    pub tool_name: String,
}

// ---------------------------------------------------------------------------
// Tool-name label bounding
// ---------------------------------------------------------------------------

/// Upper bound on distinct `tool` label values admitted per process.
///
/// Tool names originate from the configured upstream catalog and are
/// inherently bounded by configuration, but a misconfigured or malicious
/// client could send arbitrary tool names before the router rejects
/// them. This cap keeps the Prometheus registry bounded regardless.
/// The MCP Gateway design targets ~100 tools per tenant; 512 leaves
/// generous headroom.
pub(crate) const MAX_TOOL_LABELS: usize = 512;

/// Maximum length of a tool name label value. Names longer than this are
/// truncated at a character boundary to prevent unbounded label storage.
const MAX_TOOL_NAME_LEN: usize = 64;

/// Label for tool values that arrive once the admission cap has been reached.
const OTHER_TOOL_LABEL: &str = "other";

/// Guards the one-time histogram descriptions.
static DESCRIBED: Once = Once::new();

/// Describe histogram units once.
fn ensure_described() {
    DESCRIBED.call_once(|| {
        describe_histogram!(
            METRIC_MCP_TOOL_CALL_DURATION,
            Unit::Seconds,
            "upstream round-trip time for MCP tool calls"
        );
    });
}

// ---------------------------------------------------------------------------
// ToolAdmission — testable cardinality bounding
// ---------------------------------------------------------------------------

/// Bounded admission gate for tool-name label values.
///
/// Production code shares one process-wide instance via [`TOOL_ADMISSION`].
/// Tests can create isolated instances with a small cap to verify overflow
/// without polluting shared state.
pub(crate) struct ToolAdmission {
    /// Maximum distinct tool names this instance will admit.
    cap: usize,
    /// Lock-free snapshot of admitted names. Updated via compare-and-swap
    /// under [`Self::lock`].
    admitted: ArcSwap<HashSet<String>>,
    /// Serializes first-time admissions so only one thread clones the set.
    lock: Mutex<()>,
}

impl ToolAdmission {
    /// Create a new admission gate with the given capacity.
    pub(crate) fn new(cap: usize) -> Self {
        Self {
            cap,
            admitted: ArcSwap::from_pointee(HashSet::new()),
            lock: Mutex::new(()),
        }
    }

    /// Admit `tool` if the cap has not been reached.
    ///
    /// Fast path (already-admitted) is a lock-free snapshot load.
    /// First-time admissions serialize on the internal lock.
    /// Returns the admitted name or [`OTHER_TOOL_LABEL`] on overflow.
    pub(crate) fn admit<'a>(&self, tool: &'a str) -> &'a str {
        // Fast path: lock-free membership check.
        {
            let snapshot = self.admitted.load();
            if snapshot.contains(tool) {
                return tool;
            }
        }

        // Slow path: first-time admission.
        let _lock = self
            .lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut updated = {
            let snapshot = self.admitted.load_full();
            let current: &HashSet<String> = &snapshot;
            if current.contains(tool) {
                return tool;
            }
            if current.len() >= self.cap {
                return OTHER_TOOL_LABEL;
            }
            current.clone()
        };
        updated.insert(tool.to_owned());
        self.admitted.store(Arc::new(updated));
        tool
    }

    /// Number of distinct tool names currently admitted.
    #[cfg(test)]
    fn len(&self) -> usize {
        self.admitted.load().len()
    }
}

/// Process-wide tool admission gate used by [`tool_label`].
static TOOL_ADMISSION: std::sync::LazyLock<ToolAdmission> =
    std::sync::LazyLock::new(|| ToolAdmission::new(MAX_TOOL_LABELS));

/// Resolve a tool label, applying control-char rejection, length truncation,
/// and cardinality bound.
fn tool_label(tool: &str) -> String {
    tool_label_with(&TOOL_ADMISSION, tool)
}

/// Inner implementation that accepts an explicit [`ToolAdmission`] for testability.
fn tool_label_with(admission: &ToolAdmission, tool: &str) -> String {
    if tool.is_empty() || contains_control_chars(tool) {
        return "unknown".to_owned();
    }
    // Truncate to MAX_TOOL_NAME_LEN, respecting UTF-8 char boundaries.
    let truncated = if tool.len() > MAX_TOOL_NAME_LEN {
        match tool.char_indices().nth(MAX_TOOL_NAME_LEN) {
            Some((idx, _)) => &tool[..idx],
            None => tool,
        }
    } else {
        tool
    };
    admission.admit(truncated).to_owned()
}

// ---------------------------------------------------------------------------
// Recording helpers
// ---------------------------------------------------------------------------

/// Recognized MCP methods. Anything else is mapped to `"other"`.
const KNOWN_METHODS: &[&str] = &[
    "tools/call",
    "tools/list",
    "initialize",
    "notifications/initialized",
    "ping",
    "logging/setLevel",
    "resources/list",
    "resources/read",
    "prompts/list",
    "prompts/get",
    "completion/complete",
    "notifications/cancelled",
    "notifications/progress",
];

/// Map an MCP method to a bounded label value.
fn sanitize_method(method: &str) -> &str {
    if KNOWN_METHODS.contains(&method) {
        method
    } else {
        "other"
    }
}

/// Map a protocol version to a bounded label value.
/// Accepts ISO-date-shaped version strings (e.g. `"2024-11-05"`,
/// `"2025-11-25"`) and the literal `"unknown"`; everything else
/// becomes `"unknown"`.
fn sanitize_protocol_version(pv: &str) -> &str {
    if pv == "unknown"
        || (pv.len() == 10
            && pv.as_bytes().get(4) == Some(&b'-')
            && pv.as_bytes().get(7) == Some(&b'-')
            && pv.bytes().filter(|b| b.is_ascii_digit()).count() == 8)
    {
        pv
    } else {
        "unknown"
    }
}

/// Record one MCP request by method and protocol version.
///
/// Called from `on_request_body` after successful JSON-RPC extraction.
/// `method` is mapped to a closed set via [`sanitize_method`];
/// `protocol_version` is validated via [`sanitize_protocol_version`].
pub fn record_mcp_request(method: &str, protocol_version: &str) {
    let method = sanitize_method(method);
    let protocol_version = sanitize_protocol_version(protocol_version);
    counter!(
        METRIC_MCP_REQUESTS_TOTAL,
        "method" => method.to_owned(),
        "protocol_version" => protocol_version.to_owned()
    )
    .increment(1);
}

/// Record a completed tool call with its transport outcome and upstream duration.
///
/// Called from the response phase. `transport_status` is `"ok"` or
/// `"error"` derived from the HTTP status code (>= 400 → error).
/// This reflects **transport-level** success only; a JSON-RPC
/// application error inside an HTTP 200 is recorded as `"ok"`.
/// `duration_seconds` is wall-clock time from routing decision to
/// response headers.
pub fn record_tool_call_completed(
    tool: &str,
    cluster: &str,
    transport_status: &str,
    duration_seconds: f64,
) {
    ensure_described();
    let tool = tool_label(tool);
    let cluster = cluster.to_owned();

    counter!(
        METRIC_MCP_TOOL_CALLS_TOTAL,
        "tool" => tool.clone(),
        "cluster" => cluster.clone(),
        "transport_status" => transport_status.to_owned()
    )
    .increment(1);

    histogram!(
        METRIC_MCP_TOOL_CALL_DURATION,
        "tool" => tool,
        "cluster" => cluster
    )
    .record(duration_seconds);
}

/// Recognized rejection reasons. Anything else is mapped to `"other"`.
const KNOWN_REJECTION_REASONS: &[&str] = &[
    "invalid_params",
    "header_mismatch",
    "missing_header",
    "body_too_large",
    "invalid_jsonrpc",
];

/// Record a filter-level rejection (invalid params, header mismatch, etc.).
///
/// Called when the MCP filter returns a `FilterAction::Reject`. Both
/// `method` and `reason` are mapped to closed sets to prevent cardinality
/// explosion.
pub fn record_filter_rejection(method: &str, reason: &str) {
    let method = sanitize_method(method);
    let reason = if KNOWN_REJECTION_REASONS.contains(&reason) {
        reason
    } else {
        "other"
    };
    counter!(
        METRIC_MCP_FILTER_REJECTIONS_TOTAL,
        "method" => method.to_owned(),
        "reason" => reason.to_owned()
    )
    .increment(1);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests"
)]
mod tests {
    use metrics_util::debugging::DebuggingRecorder;

    use super::*;

    // ── Metric recording tests ──

    #[test]
    fn record_mcp_request_emits_labeled_counter() {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();

        metrics::with_local_recorder(&recorder, || {
            record_mcp_request("tools/call", "2025-11-25");
            record_mcp_request("tools/call", "2025-11-25");
            record_mcp_request("tools/list", "2026-07-28");
        });

        let snap = snapshotter.snapshot().into_vec();
        let total = |method: &str, pv: &str| -> u64 {
            snap.iter()
                .filter(|(k, ..)| {
                    k.key().name() == METRIC_MCP_REQUESTS_TOTAL
                        && k.key().labels().any(|l| l.key() == "method" && l.value() == method)
                        && k.key()
                            .labels()
                            .any(|l| l.key() == "protocol_version" && l.value() == pv)
                })
                .filter_map(|(_, _, _, v)| match v {
                    metrics_util::debugging::DebugValue::Counter(c) => Some(*c),
                    _ => None,
                })
                .sum()
        };

        assert_eq!(total("tools/call", "2025-11-25"), 2);
        assert_eq!(total("tools/list", "2026-07-28"), 1);
    }

    #[test]
    fn record_tool_call_completed_emits_counter_and_histogram() {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();

        metrics::with_local_recorder(&recorder, || {
            record_tool_call_completed("weather_lookup", "mcp-weather", "ok", 0.042);
            record_tool_call_completed("weather_lookup", "mcp-weather", "error", 1.5);
        });

        let snap = snapshotter.snapshot().into_vec();

        // Counter check
        let counter_total = |status: &str| -> u64 {
            snap.iter()
                .filter(|(k, ..)| {
                    k.key().name() == METRIC_MCP_TOOL_CALLS_TOTAL
                        && k.key()
                            .labels()
                            .any(|l| l.key() == "tool" && l.value() == "weather_lookup")
                        && k.key()
                            .labels()
                            .any(|l| l.key() == "transport_status" && l.value() == status)
                })
                .filter_map(|(_, _, _, v)| match v {
                    metrics_util::debugging::DebugValue::Counter(c) => Some(*c),
                    _ => None,
                })
                .sum()
        };
        assert_eq!(counter_total("ok"), 1);
        assert_eq!(counter_total("error"), 1);

        // Histogram check — two samples recorded
        let histogram_count: usize = snap
            .iter()
            .filter(|(k, ..)| k.key().name() == METRIC_MCP_TOOL_CALL_DURATION)
            .filter_map(|(_, _, _, v)| match v {
                metrics_util::debugging::DebugValue::Histogram(h) => Some(h.len()),
                _ => None,
            })
            .sum();
        assert_eq!(histogram_count, 2);
    }

    #[test]
    fn record_filter_rejection_emits_labeled_counter() {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();

        metrics::with_local_recorder(&recorder, || {
            record_filter_rejection("tools/call", "invalid_params");
            record_filter_rejection("tools/call", "header_mismatch");
        });

        let snap = snapshotter.snapshot().into_vec();
        let present = |reason: &str| -> bool {
            snap.iter().any(|(k, ..)| {
                k.key().name() == METRIC_MCP_FILTER_REJECTIONS_TOTAL
                    && k.key()
                        .labels()
                        .any(|l| l.key() == "reason" && l.value() == reason)
            })
        };
        assert!(present("invalid_params"));
        assert!(present("header_mismatch"));
    }

    // ── Label sanitization tests ──

    #[test]
    fn sanitize_method_rejects_unknown() {
        assert_eq!(sanitize_method("tools/call"), "tools/call");
        assert_eq!(sanitize_method("tools/list"), "tools/list");
        assert_eq!(sanitize_method("initialize"), "initialize");
        assert_eq!(sanitize_method("notifications/initialized"), "notifications/initialized");
        assert_eq!(sanitize_method("logging/setLevel"), "logging/setLevel");
        assert_eq!(sanitize_method("evil_method"), "other");
        assert_eq!(sanitize_method(""), "other");
        assert_eq!(sanitize_method("initialized"), "other");
    }

    #[test]
    fn sanitize_protocol_version_rejects_garbage() {
        assert_eq!(sanitize_protocol_version("2025-11-25"), "2025-11-25");
        assert_eq!(sanitize_protocol_version("2024-11-05"), "2024-11-05");
        assert_eq!(sanitize_protocol_version("unknown"), "unknown");
        assert_eq!(sanitize_protocol_version("evil"), "unknown");
        assert_eq!(sanitize_protocol_version(""), "unknown");
        assert_eq!(sanitize_protocol_version("99999-99-99"), "unknown");
    }

    #[test]
    fn unknown_method_maps_to_other_in_recording() {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();

        metrics::with_local_recorder(&recorder, || {
            record_mcp_request("evil_method", "2025-11-25");
        });

        let snap = snapshotter.snapshot().into_vec();
        let has_other = snap.iter().any(|(k, ..)| {
            k.key().name() == METRIC_MCP_REQUESTS_TOTAL
                && k.key().labels().any(|l| l.key() == "method" && l.value() == "other")
        });
        assert!(has_other, "unknown method should map to 'other' label");
    }

    #[test]
    fn unknown_rejection_reason_maps_to_other() {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();

        metrics::with_local_recorder(&recorder, || {
            record_filter_rejection("tools/call", "some_unknown_reason");
        });

        let snap = snapshotter.snapshot().into_vec();
        let has_other = snap.iter().any(|(k, ..)| {
            k.key().name() == METRIC_MCP_FILTER_REJECTIONS_TOTAL
                && k.key().labels().any(|l| l.key() == "reason" && l.value() == "other")
        });
        assert!(has_other, "unknown rejection reason should map to 'other' label");
    }

    // ── Tool admission tests (isolated instances) ──

    #[test]
    fn tool_label_empty_and_normal() {
        let admission = ToolAdmission::new(16);
        assert_eq!(tool_label_with(&admission, ""), "unknown");

        let name = "get_weather";
        assert_eq!(tool_label_with(&admission, name), name);
    }

    #[test]
    fn tool_label_truncates_long_names() {
        let admission = ToolAdmission::new(16);
        let long_name = "a".repeat(100);
        let label = tool_label_with(&admission, &long_name);
        assert_eq!(label.len(), MAX_TOOL_NAME_LEN);
        assert_eq!(label, "a".repeat(MAX_TOOL_NAME_LEN));
    }

    #[test]
    fn tool_label_truncates_multibyte_safely() {
        let admission = ToolAdmission::new(16);
        // 65 two-byte characters → 130 bytes, should truncate to 64 chars.
        let multibyte = "é".repeat(65);
        let label = tool_label_with(&admission, &multibyte);
        assert_eq!(label.chars().count(), MAX_TOOL_NAME_LEN);
    }

    #[test]
    fn tool_admission_overflow_returns_other() {
        let admission = ToolAdmission::new(4);

        // Admit exactly 4 names.
        assert_eq!(admission.admit("tool_a"), "tool_a");
        assert_eq!(admission.admit("tool_b"), "tool_b");
        assert_eq!(admission.admit("tool_c"), "tool_c");
        assert_eq!(admission.admit("tool_d"), "tool_d");
        assert_eq!(admission.len(), 4);

        // 5th name overflows to "other".
        assert_eq!(admission.admit("tool_e"), OTHER_TOOL_LABEL);
        assert_eq!(admission.admit("tool_f"), OTHER_TOOL_LABEL);

        // Previously admitted names still work.
        assert_eq!(admission.admit("tool_a"), "tool_a");
        assert_eq!(admission.admit("tool_c"), "tool_c");

        // Set size never exceeds cap.
        assert_eq!(admission.len(), 4);
    }

    #[test]
    fn tool_label_with_overflow_returns_other() {
        let admission = ToolAdmission::new(3);

        assert_eq!(tool_label_with(&admission, "alpha"), "alpha");
        assert_eq!(tool_label_with(&admission, "beta"), "beta");
        assert_eq!(tool_label_with(&admission, "gamma"), "gamma");

        // Cap reached — new names overflow.
        assert_eq!(tool_label_with(&admission, "delta"), OTHER_TOOL_LABEL);

        // Previously admitted still pass.
        assert_eq!(tool_label_with(&admission, "alpha"), "alpha");
    }

    #[test]
    fn tool_admission_concurrent() {
        use std::sync::Barrier;

        let admission = Arc::new(ToolAdmission::new(8));
        let barrier = Arc::new(Barrier::new(5)); // 4 threads + main
        let mut handles = Vec::new();

        for t in 0..4 {
            let admission = Arc::clone(&admission);
            let barrier = Arc::clone(&barrier);
            handles.push(std::thread::spawn(move || {
                barrier.wait();
                for i in 0..4 {
                    let name = format!("conc_t{t}_m{i}");
                    let label = admission.admit(&name);
                    assert!(
                        label == name || label == OTHER_TOOL_LABEL,
                        "unexpected label: {label}"
                    );
                }
            }));
        }
        barrier.wait();
        for h in handles {
            h.join().unwrap();
        }

        // 4 threads × 4 names = 16, but cap is 8. At most 8 admitted.
        assert!(admission.len() <= 8, "admission set exceeded cap");
    }
}
