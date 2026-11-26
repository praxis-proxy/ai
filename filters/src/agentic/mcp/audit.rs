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

// ---------------------------------------------------------------------------
// Field-length bounds (AGENTS.md § Key Patterns: "bounded classification
// fields").  Caps prevent body-sized INFO records when an MCP request
// carries a large `params.name` or unusual header values.
// ---------------------------------------------------------------------------

/// Maximum byte length for the `mcp_name` audit field.
///
/// Tool names are typically short identifiers. 128 bytes is generous for
/// any realistic name while preventing a 60 KiB `params.name` from
/// inflating audit records.
const MAX_AUDIT_NAME_LEN: usize = 128;

/// Maximum byte length for `mcp_session_id` and `mcp_protocol_version`.
const MAX_AUDIT_FIELD_LEN: usize = 128;

/// Truncate `s` to at most `max_len` bytes at a UTF-8 character boundary.
/// Returns `None` if the input is `None`.
fn truncate(s: Option<&str>, max_len: usize) -> Option<&str> {
    s.map(|v| {
        if v.len() <= max_len {
            v
        } else {
            // Find the last char boundary at or before max_len
            let mut end = max_len;
            while end > 0 && !v.is_char_boundary(end) {
                end -= 1;
            }
            &v[..end]
        }
    })
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
        mcp_name             = truncate(name, MAX_AUDIT_NAME_LEN),
        mcp_session_id       = truncate(session_id, MAX_AUDIT_FIELD_LEN),
        mcp_protocol_version = truncate(protocol_version, MAX_AUDIT_FIELD_LEN),
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
        mcp_session_id       = truncate(session_id, MAX_AUDIT_FIELD_LEN),
        mcp_protocol_version = truncate(protocol_version, MAX_AUDIT_FIELD_LEN),
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
        mcp_method           = "tools/call",
        mcp_name             = truncate(Some(state.tool_name.as_str()), MAX_AUDIT_NAME_LEN),
        mcp_session_id       = truncate(state.session_id.as_deref(), MAX_AUDIT_FIELD_LEN),
        mcp_protocol_version = truncate(state.protocol_version.as_deref(), MAX_AUDIT_FIELD_LEN),
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

    use tracing_subscriber::layer::SubscriberExt;

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

    /// Run `f` under a thread-local scoped subscriber that captures event
    /// targets.  Unlike `try_init()`, `set_default()` installs a per-thread
    /// subscriber that does not conflict with other tests running in
    /// parallel, so every test reliably captures its own events.
    fn with_captured_events<F: FnOnce()>(f: F) -> Vec<String> {
        let targets = Arc::new(Mutex::new(Vec::<String>::new()));
        let layer = CapturingLayer {
            events: Arc::clone(&targets),
        };
        let subscriber = tracing_subscriber::registry().with(layer);
        let guard = tracing::subscriber::set_default(subscriber);
        f();
        drop(guard);
        Arc::try_unwrap(targets)
            .expect("only one reference remains")
            .into_inner()
            .unwrap()
    }

    // ── Field-level capture (for content-leak regression tests) ──────

    /// Tracing layer that captures the full textual representation of every
    /// event's fields.  Used by `capture_log_output` to verify that
    /// sensitive content (e.g. resource URIs with credentials) never reaches
    /// the audit log output.
    struct FieldCapture(Arc<Mutex<String>>);

    struct FieldWriter<'a>(&'a mut String);

    impl tracing::field::Visit for FieldWriter<'_> {
        fn record_debug(
            &mut self,
            field: &tracing::field::Field,
            value: &dyn std::fmt::Debug,
        ) {
            use std::fmt::Write as _;
            write!(self.0, " {}={value:?}", field.name())
                .expect("writing to a String cannot fail");
        }
    }

    impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for FieldCapture {
        fn on_event(
            &self,
            event: &tracing::Event<'_>,
            _ctx: tracing_subscriber::layer::Context<'_, S>,
        ) {
            let mut line = event.metadata().target().to_owned();
            event.record(&mut FieldWriter(&mut line));
            let mut out = self.0.lock().expect("field capture lock");
            out.push_str(&line);
            out.push('\n');
        }
    }

    /// Run `f` under a thread-local scoped subscriber that captures the
    /// full textual output (target + all field key=value pairs) of every
    /// tracing event.  Returns the concatenated output as a single `String`.
    ///
    /// Mirrors the `test_utils::capture_log_output` pattern used elsewhere
    /// in the codebase for content-leak regression tests.
    fn capture_log_output<F: FnOnce()>(f: F) -> String {
        let output = Arc::new(Mutex::new(String::new()));
        let subscriber =
            tracing_subscriber::registry().with(FieldCapture(Arc::clone(&output)));
        let guard = tracing::subscriber::set_default(subscriber);
        f();
        drop(guard);
        Arc::try_unwrap(output)
            .expect("only one reference remains")
            .into_inner()
            .unwrap()
    }

    #[test]
    fn emit_rejected_uses_audit_target() {
        let targets = with_captured_events(|| {
            emit_rejected(
                "tools/call",
                Some("query_database"),
                Some("sess_abc"),
                Some("2026-11-05"),
                Some("acme-corp"),
                "invalid_params",
            );
        });
        assert!(
            targets
                .iter()
                .all(|t| t == "praxis_ai::mcp_audit"),
            "every event must use the dedicated audit target, got: {targets:?}"
        );
    }

    #[test]
    fn emit_non_tool_uses_audit_target() {
        let targets = with_captured_events(|| {
            emit_non_tool("tools/list", Some("sess_xyz"), Some("2026-11-05"), None);
        });
        assert!(
            targets
                .iter()
                .all(|t| t == "praxis_ai::mcp_audit"),
            "every event must use the dedicated audit target, got: {targets:?}"
        );
    }

    #[test]
    fn emit_completed_ok_uses_audit_target() {
        let targets = with_captured_events(|| {
            let state = AuditState {
                start: Instant::now(),
                tool_name: "query_database".to_owned(),
                session_id: Some("sess_abc".to_owned()),
                protocol_version: Some("2026-11-05".to_owned()),
                tenant: Some("acme-corp".to_owned()),
            };
            emit_completed(&state, "analytics-mcp", 200);
        });
        assert!(
            targets
                .iter()
                .all(|t| t == "praxis_ai::mcp_audit"),
            "every event must use the dedicated audit target, got: {targets:?}"
        );
    }

    #[test]
    fn emit_completed_error_uses_audit_target() {
        let targets = with_captured_events(|| {
            let state = AuditState {
                start: Instant::now(),
                tool_name: "query_database".to_owned(),
                session_id: None,
                protocol_version: None,
                tenant: None,
            };
            emit_completed(&state, "analytics-mcp", 500);
        });
        assert!(
            targets
                .iter()
                .all(|t| t == "praxis_ai::mcp_audit"),
            "every event must use the dedicated audit target, got: {targets:?}"
        );
    }

    #[test]
    fn duration_ms_is_sane() {
        let state = AuditState {
            start: Instant::now(),
            tool_name: "tool".to_owned(),
            session_id: None,
            protocol_version: None,
            tenant: None,
        };
        // No panic / overflow expected
        emit_completed(&state, "cluster", 200);
    }

    // ── Field-length bounding tests ───────────────────────────────────

    #[test]
    fn oversized_tool_name_is_truncated_in_audit() {
        // A tool name longer than MAX_AUDIT_NAME_LEN must be truncated
        // so that body-sized names do not inflate audit records.
        let long_name = "a".repeat(300);
        let output = capture_log_output(|| {
            emit_rejected(
                "tools/call",
                Some(&long_name),
                Some("sess"),
                Some("2026-11-05"),
                Some("acme"),
                "invalid_params",
            );
        });
        assert!(
            output.contains("mcp.call.rejected"),
            "event must be emitted"
        );
        // The full 300-char name must NOT appear
        assert!(
            !output.contains(&long_name),
            "oversized tool name should be truncated, got: {output}"
        );
        // A truncated prefix MUST appear (first 128 chars)
        assert!(
            output.contains(&"a".repeat(super::MAX_AUDIT_NAME_LEN)),
            "truncated prefix should be present"
        );
    }

    #[test]
    fn normal_tool_name_is_not_truncated() {
        let name = "query_database";
        let output = capture_log_output(|| {
            emit_rejected(
                "tools/call",
                Some(name),
                Some("sess"),
                Some("2026-11-05"),
                None,
                "invalid_params",
            );
        });
        assert!(
            output.contains(name),
            "normal-length tool name must appear in full"
        );
    }

    // ── Content-leak regression tests ────────────────────────────────
    //
    // The MCP filter suppresses resource URIs (which may contain
    // credentials) from the audit `mcp_name` field by passing `None`
    // when `mcp_envelope.method.requires_uri()` is true.  These tests
    // prove the suppression works at the emit layer: a credential
    // marker embedded in a URI must never appear in the captured log
    // output.
    //
    // Pattern: `test_utils::capture_log_output` with a marker string,
    // per AGENTS.md § Key Patterns.

    #[test]
    fn rejected_resource_uri_credential_not_logged() {
        // Simulate a resources/read rejection where the caller (mod.rs)
        // has already suppressed the URI by passing `name: None`.
        let marker = "AUDIT_SECRET_TOKEN_12345";
        let uri_with_creds =
            format!("https://internal.corp/data?token={marker}&page=1");

        let output = capture_log_output(|| {
            // The name parameter is None — the URI is suppressed.
            // Pass the URI nowhere; it must not appear in the output.
            emit_rejected(
                "resources/read",
                None, // suppressed — URI with credential not passed
                Some("sess_resource"),
                Some("2026-11-05"),
                Some("acme-corp"),
                "missing_header",
            );
        });

        // Positive: event was emitted
        assert!(
            output.contains("mcp.call.rejected"),
            "expected rejection event in output: {output}"
        );
        // Negative: credential marker must be absent
        assert!(
            !output.contains(marker),
            "credential marker '{marker}' leaked into audit output: {output}"
        );
        // Negative: full URI must be absent
        assert!(
            !output.contains(&uri_with_creds),
            "full URI leaked into audit output: {output}"
        );
    }

    /// Prove that `McpMethod::ResourcesRead` triggers the `requires_uri()`
    /// guard that causes mod.rs to pass `name: None` to `emit_rejected`.
    /// Combined with `rejected_resource_uri_credential_not_logged` above,
    /// this covers both halves of the URI suppression contract:
    ///   1. `requires_uri()` → true for resources/read (this test)
    ///   2. `emit_rejected(…, None, …)` → credential absent (test above)
    #[test]
    fn resources_read_requires_uri_guard() {
        use super::super::envelope::McpMethod;
        assert!(
            McpMethod::ResourcesRead.requires_uri(),
            "resources/read must trigger requires_uri() so mod.rs suppresses the URI"
        );
        // tools/call must NOT trigger requires_uri — names are safe to log
        assert!(
            !McpMethod::ToolsCall.requires_uri(),
            "tools/call must not trigger requires_uri()"
        );
    }

    #[test]
    fn rejected_tools_call_name_is_logged() {
        // For tools/call, the tool name IS logged — verify it appears.
        let tool_name = "query_database";

        let output = capture_log_output(|| {
            emit_rejected(
                "tools/call",
                Some(tool_name),
                Some("sess_tool"),
                Some("2026-11-05"),
                Some("acme-corp"),
                "invalid_params",
            );
        });

        assert!(
            output.contains("mcp.call.rejected"),
            "expected rejection event in output: {output}"
        );
        assert!(
            output.contains(tool_name),
            "tool name '{tool_name}' should be present in audit output: {output}"
        );
    }
}
