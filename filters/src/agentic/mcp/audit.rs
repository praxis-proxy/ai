// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Structured audit logging for MCP tool calls.
//!
//! Each MCP request emits one or two structured log events at `INFO` level
//! with a `log_type = "mcp_audit"` structured field. The dedicated tracing
//! target `praxis_ai::mcp_audit` lets log collectors (OpenShift Logging /
//! Vector, Loki, Fluentd) route and store audit events independently of
//! general proxy diagnostic logs.
//!
//! # Two-phase audit model
//!
//! Because filter-level rejections happen before routing, and tool-call
//! outcomes arrive in the response phase, audit recording is split:
//!
//! 1. **Request phase** — [`emit_rejected`] fires immediately for
//!    filter-level rejections (`outcome: "rejected"`).
//!    [`emit_non_tool`] fires for non-tool methods (`tools/list`,
//!    `initialize`, `ping`) that have no response-phase follow-up.
//!    For `tools/call` requests that pass validation, an [`AuditState`]
//!    is stored via `filter_state` for the response phase.
//!
//! 2. **Response phase** — [`emit_completed`] fires after the upstream
//!    responds to a `tools/call`, recording outcome, HTTP status, latency,
//!    and cluster identity in a single complete record.
//!
//! # Cross-phase state
//!
//! Uses [`AuditState`] stored via `ctx.insert_filter_state()` /
//! `ctx.get_filter_state::<AuditState>()` — the same typed per-filter-
//! instance mechanism used by the metrics module's `McpMetricsState`.
//! This avoids key collisions between audit and metrics metadata.
//!
//! # JSON output — rejected call (request phase)
//!
//! ```json
//! {
//!   "timestamp": "2026-11-12T14:32:07.001Z",
//!   "level": "INFO",
//!   "target": "praxis_ai::mcp_audit",
//!   "fields": {
//!     "message":             "mcp.call.rejected",
//!     "log_type":            "mcp_audit",
//!     "mcp_method":          "tools/call",
//!     "mcp_name":            "query_database",
//!     "mcp_session_id":      "sess_abc123",
//!     "mcp_protocol_version":"2026-11-05",
//!     "mcp_outcome":         "rejected",
//!     "mcp_rejection_reason":"invalid_params",
//!     "tenant":              "acme-corp"
//!   }
//! }
//! ```
//!
//! # JSON output — completed tool call (response phase)
//!
//! ```json
//! {
//!   "timestamp": "2026-11-12T14:32:07.044Z",
//!   "level": "INFO",
//!   "target": "praxis_ai::mcp_audit",
//!   "fields": {
//!     "message":             "mcp.call.completed",
//!     "log_type":            "mcp_audit",
//!     "mcp_method":          "tools/call",
//!     "mcp_name":            "query_database",
//!     "mcp_session_id":      "sess_abc123",
//!     "mcp_protocol_version":"2026-11-05",
//!     "mcp_outcome":         "ok",
//!     "mcp_upstream_status": 200,
//!     "mcp_duration_ms":     42,
//!     "mcp_cluster":         "analytics-mcp",
//!     "tenant":              "acme-corp"
//!   }
//! }
//! ```

use tracing::info;

// ---------------------------------------------------------------------------
// Per-request audit state for cross-phase carry-over
// ---------------------------------------------------------------------------

/// Per-request audit state stored in `filter_state` during the request
/// phase and retrieved in the response phase.
///
/// Using `filter_state` (keyed by `current_filter_id`) instead of string-
/// based metadata ensures audit keys never collide with the metrics module's
/// `McpMetricsState` or other filter metadata.
pub(crate) struct AuditState {
    /// Monotonic start time for latency measurement.
    ///
    /// Uses [`std::time::Instant`] so system-clock adjustments never produce
    /// negative or inflated duration values.
    pub start: std::time::Instant,
    /// MCP method (always `"tools/call"` when stored).
    pub method: String,
    /// Tool name from `params.name`.
    pub tool_name: String,
    /// MCP session ID from the `Mcp-Session-Id` header, if present.
    pub session_id: Option<String>,
    /// MCP protocol version from the body, if present.
    pub protocol_version: Option<String>,
    /// Tenant identifier from `x-tenant-id` header, if present.
    pub tenant: Option<String>,
}

// ---------------------------------------------------------------------------
// Outcome vocabulary (closed set — log collectors can filter on these)
// ---------------------------------------------------------------------------

/// Request rejected by the filter before reaching any upstream.
const OUTCOME_REJECTED: &str = "rejected";

/// Non-tool MCP request (tools/list, initialize, ping …) passed the filter.
const OUTCOME_INITIATED: &str = "initiated";

/// tools/call upstream responded with a 2xx status code.
const OUTCOME_OK: &str = "ok";

/// tools/call upstream responded with a 4xx/5xx status code.
const OUTCOME_ERROR: &str = "error";

// ---------------------------------------------------------------------------
// Request-phase: rejection
// ---------------------------------------------------------------------------

/// Emit an audit event for an MCP request **rejected** at the filter layer.
///
/// `rejection_reason` is a closed vocabulary: `"invalid_params"`,
/// `"header_mismatch"`, `"missing_header"`.
pub(crate) fn emit_rejected(
    method: &str,
    name: Option<&str>,
    session_id: Option<&str>,
    protocol_version: Option<&str>,
    tenant: Option<&str>,
    rejection_reason: &str,
) {
    info!(
        target: "praxis_ai::mcp_audit",
        log_type             = "mcp_audit",
        mcp_method           = method,
        mcp_name             = name,
        mcp_session_id       = session_id,
        mcp_protocol_version = protocol_version,
        mcp_outcome          = OUTCOME_REJECTED,
        mcp_rejection_reason = rejection_reason,
        tenant               = tenant,
        "mcp.call.rejected",
    );
}

// ---------------------------------------------------------------------------
// Request-phase: non-tool methods (complete event emitted at request time)
// ---------------------------------------------------------------------------

/// Emit an audit event for a **non-`tools/call`** MCP request that passed
/// all filter checks.
///
/// Methods such as `tools/list`, `initialize`, `resources/list`, `ping`,
/// and `prompts/list` do not invoke an upstream tool and have no response-
/// phase follow-up. The entire audit record is emitted here.
pub(crate) fn emit_non_tool(
    method: &str,
    session_id: Option<&str>,
    protocol_version: Option<&str>,
    tenant: Option<&str>,
) {
    info!(
        target: "praxis_ai::mcp_audit",
        log_type             = "mcp_audit",
        mcp_method           = method,
        mcp_session_id       = session_id,
        mcp_protocol_version = protocol_version,
        mcp_outcome          = OUTCOME_INITIATED,
        tenant               = tenant,
        "mcp.call.initiated",
    );
}

// ---------------------------------------------------------------------------
// Response-phase: completed tool call
// ---------------------------------------------------------------------------

/// Emit a completed audit event for a `tools/call` upstream response.
///
/// Called from `on_response` after retrieving the [`AuditState`] stored
/// during the request phase. Computes latency from the stored `Instant`.
pub(crate) fn emit_completed(state: &AuditState, cluster: &str, upstream_status: u16) {
    let outcome = if upstream_status >= 400 {
        OUTCOME_ERROR
    } else {
        OUTCOME_OK
    };

    // Compute latency in milliseconds without u128→u64 truncation cast
    // (cast_possible_truncation is deny). `as_secs()` returns u64;
    // `subsec_millis()` returns u32 — both fit u64 natively.
    let elapsed = state.start.elapsed();
    let duration_ms = elapsed
        .as_secs()
        .saturating_mul(1000)
        .saturating_add(u64::from(elapsed.subsec_millis()));

    info!(
        target: "praxis_ai::mcp_audit",
        log_type             = "mcp_audit",
        mcp_method           = state.method.as_str(),
        mcp_name             = state.tool_name.as_str(),
        mcp_session_id       = state.session_id.as_deref(),
        mcp_protocol_version = state.protocol_version.as_deref(),
        mcp_outcome          = outcome,
        mcp_upstream_status  = upstream_status,
        mcp_duration_ms      = duration_ms,
        mcp_cluster          = cluster,
        tenant               = state.tenant.as_deref(),
        "mcp.call.completed",
    );
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
    use std::sync::{Arc, Mutex};
    use std::time::Instant;

    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

    use super::*;

    // Minimal tracing layer that captures event targets for assertion.
    struct CapturingLayer {
        events: Arc<Mutex<Vec<String>>>,
    }

    impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for CapturingLayer {
        fn on_event(
            &self,
            event: &tracing::Event<'_>,
            _ctx: tracing_subscriber::layer::Context<'_, S>,
        ) {
            self.events
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(event.metadata().target().to_owned());
        }
    }

    fn capture_targets() -> Arc<Mutex<Vec<String>>> {
        let targets = Arc::new(Mutex::new(Vec::<String>::new()));
        let layer = CapturingLayer {
            events: Arc::clone(&targets),
        };
        // best-effort — ignore if a subscriber is already set
        let _ = tracing_subscriber::registry().with(layer).try_init();
        targets
    }

    #[test]
    fn emit_rejected_uses_audit_target() {
        let targets = capture_targets();
        emit_rejected(
            "tools/call",
            Some("query_database"),
            Some("sess_abc"),
            Some("2026-11-05"),
            Some("acme-corp"),
            "invalid_params",
        );
        let locked = targets.lock().unwrap();
        assert!(
            !locked.is_empty(),
            "expected at least one tracing event, got none"
        );
    }

    #[test]
    fn emit_non_tool_uses_audit_target() {
        let targets = capture_targets();
        emit_non_tool("tools/list", Some("sess_xyz"), Some("2026-11-05"), None);
        let locked = targets.lock().unwrap();
        assert!(
            !locked.is_empty(),
            "expected at least one tracing event, got none"
        );
    }

    #[test]
    fn emit_completed_ok_uses_audit_target() {
        let targets = capture_targets();
        let state = AuditState {
            start: Instant::now(),
            method: "tools/call".to_owned(),
            tool_name: "query_database".to_owned(),
            session_id: Some("sess_abc".to_owned()),
            protocol_version: Some("2026-11-05".to_owned()),
            tenant: Some("acme-corp".to_owned()),
        };
        emit_completed(&state, "analytics-mcp", 200);
        let locked = targets.lock().unwrap();
        assert!(
            !locked.is_empty(),
            "expected at least one tracing event, got none"
        );
    }

    #[test]
    fn emit_completed_error_uses_audit_target() {
        let targets = capture_targets();
        let state = AuditState {
            start: Instant::now(),
            method: "tools/call".to_owned(),
            tool_name: "query_database".to_owned(),
            session_id: None,
            protocol_version: None,
            tenant: None,
        };
        emit_completed(&state, "analytics-mcp", 500);
        let locked = targets.lock().unwrap();
        assert!(
            !locked.is_empty(),
            "expected at least one tracing event, got none"
        );
    }

    #[test]
    fn duration_ms_is_sane() {
        let state = AuditState {
            start: Instant::now(),
            method: "tools/call".to_owned(),
            tool_name: "tool".to_owned(),
            session_id: None,
            protocol_version: None,
            tenant: None,
        };
        // No panic / overflow expected
        emit_completed(&state, "cluster", 200);
    }
}
