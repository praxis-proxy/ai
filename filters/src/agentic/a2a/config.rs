// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Configuration types for the A2A filter.

use std::collections::BTreeMap;

use praxis_filter::{
    FilterError,
    builtins::http::payload_processing::{
        OnInvalidBehavior,
        config_validation::{validate_header_name, validate_max_body_bytes},
    },
};
use serde::Deserialize;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Default maximum request body size for `StreamBuffer` mode (64 `KiB`).
pub(crate) const DEFAULT_MAX_BODY_BYTES: usize = 65_536;

/// Default route cluster header name for task routing.
const DEFAULT_ROUTE_CLUSTER_HEADER: &str = "x-praxis-a2a-route-cluster";

/// Default TTL for non-terminal task routes (1 hour).
const DEFAULT_TTL_SECONDS: u64 = 3_600; // 1 hour

/// Default TTL for terminal task routes (5 minutes).
const DEFAULT_TERMINAL_TTL_SECONDS: u64 = 300; // 5 minutes

/// Default maximum response body bytes for task route capture (64 `KiB`).
const DEFAULT_MAX_RESPONSE_BODY_BYTES: usize = 65_536;

// -----------------------------------------------------------------------------
// Behavior Enums
// -----------------------------------------------------------------------------

/// Task route lookup miss behavior.
#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum OnLookupMiss {
    /// Continue without a route header; let the router fallback decide.
    #[default]
    Continue,
}

/// Task route store backend.
#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TaskRouteStore {
    /// In-process local store.
    #[default]
    Local,
}

// -----------------------------------------------------------------------------
// TaskRoutingConfig
// -----------------------------------------------------------------------------

/// Configuration for A2A task-ownership routing.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TaskRoutingConfig {
    /// Whether task routing is enabled.
    #[serde(default)]
    pub enabled: bool,

    /// Maximum response body bytes to buffer for task route capture.
    #[serde(default = "default_max_response_body_bytes")]
    pub max_response_body_bytes: usize,

    /// Behavior when a task route lookup misses.
    #[serde(default)]
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "validated at parse time, used in follow-up PRs")
    )]
    pub on_lookup_miss: OnLookupMiss,

    /// Internal header name injected on task or context route hit.
    #[serde(default = "default_route_cluster_header")]
    pub route_cluster_header: String,

    /// Storage backend for task routes.
    #[serde(default)]
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "validated at parse time, only local supported in this PR")
    )]
    pub store: TaskRouteStore,

    /// TTL in seconds for terminal task routes (0 = remove immediately).
    #[serde(default = "default_terminal_ttl_seconds")]
    pub terminal_ttl_seconds: u64,

    /// TTL in seconds for non-terminal task routes.
    #[serde(default = "default_ttl_seconds")]
    pub ttl_seconds: u64,
}

impl Default for TaskRoutingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_response_body_bytes: DEFAULT_MAX_RESPONSE_BODY_BYTES,
            on_lookup_miss: OnLookupMiss::default(),
            route_cluster_header: DEFAULT_ROUTE_CLUSTER_HEADER.to_owned(),
            store: TaskRouteStore::default(),
            terminal_ttl_seconds: DEFAULT_TERMINAL_TTL_SECONDS,
            ttl_seconds: DEFAULT_TTL_SECONDS,
        }
    }
}

/// Default route cluster header.
fn default_route_cluster_header() -> String {
    DEFAULT_ROUTE_CLUSTER_HEADER.to_owned()
}

/// Default TTL seconds.
fn default_ttl_seconds() -> u64 {
    DEFAULT_TTL_SECONDS
}

/// Default terminal TTL seconds.
fn default_terminal_ttl_seconds() -> u64 {
    DEFAULT_TERMINAL_TTL_SECONDS
}

/// Default max response body bytes.
fn default_max_response_body_bytes() -> usize {
    DEFAULT_MAX_RESPONSE_BODY_BYTES
}

// -----------------------------------------------------------------------------
// A2aHeaders
// -----------------------------------------------------------------------------

/// Promoted header names for A2A metadata.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct A2aHeaders {
    /// Header name for the extracted context ID (e.g. `x-praxis-a2a-context-id`).
    #[serde(default = "default_context_id_header")]
    pub context_id: Option<String>,

    /// Header name for the A2A family (e.g. `x-praxis-a2a-family`).
    #[serde(default = "default_family_header")]
    pub family: Option<String>,

    /// Header name for the JSON-RPC kind (e.g. `x-praxis-a2a-kind`).
    #[serde(default = "default_kind_header")]
    pub kind: Option<String>,

    /// Header name for the canonical A2A method (e.g. `x-praxis-a2a-method`).
    #[serde(default = "default_method_header")]
    pub method: Option<String>,

    /// Header name for streaming detection (e.g. `x-praxis-a2a-streaming`).
    #[serde(default = "default_streaming_header")]
    pub streaming: Option<String>,

    /// Header name for the extracted task ID (e.g. `x-praxis-a2a-task-id`).
    #[serde(default = "default_task_id_header")]
    pub task_id: Option<String>,

    /// Header name for A2A version (e.g. `x-praxis-a2a-version`).
    #[serde(default = "default_version_header")]
    pub version: Option<String>,
}

impl Default for A2aHeaders {
    fn default() -> Self {
        Self {
            context_id: default_context_id_header(),
            family: default_family_header(),
            kind: default_kind_header(),
            method: default_method_header(),
            streaming: default_streaming_header(),
            task_id: default_task_id_header(),
            version: default_version_header(),
        }
    }
}

/// Default context ID header name.
#[expect(
    clippy::unnecessary_wraps,
    reason = "serde default functions require Option return type"
)]
fn default_context_id_header() -> Option<String> {
    Some("x-praxis-a2a-context-id".to_owned())
}

/// Default method header name.
#[expect(
    clippy::unnecessary_wraps,
    reason = "serde default functions require Option return type"
)]
fn default_method_header() -> Option<String> {
    Some("x-praxis-a2a-method".to_owned())
}

/// Default family header name.
#[expect(
    clippy::unnecessary_wraps,
    reason = "serde default functions require Option return type"
)]
fn default_family_header() -> Option<String> {
    Some("x-praxis-a2a-family".to_owned())
}

/// Default task ID header name.
#[expect(
    clippy::unnecessary_wraps,
    reason = "serde default functions require Option return type"
)]
fn default_task_id_header() -> Option<String> {
    Some("x-praxis-a2a-task-id".to_owned())
}

/// Default kind header name.
#[expect(
    clippy::unnecessary_wraps,
    reason = "serde default functions require Option return type"
)]
fn default_kind_header() -> Option<String> {
    Some("x-praxis-a2a-kind".to_owned())
}

/// Default streaming header name.
#[expect(
    clippy::unnecessary_wraps,
    reason = "serde default functions require Option return type"
)]
fn default_streaming_header() -> Option<String> {
    Some("x-praxis-a2a-streaming".to_owned())
}

/// Default version header name.
#[expect(
    clippy::unnecessary_wraps,
    reason = "serde default functions require Option return type"
)]
fn default_version_header() -> Option<String> {
    Some("x-praxis-a2a-version".to_owned())
}

// -----------------------------------------------------------------------------
// A2aConfig
// -----------------------------------------------------------------------------

/// YAML configuration for the A2A filter.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct A2aConfig {
    /// Header names for A2A metadata promotion.
    #[serde(default)]
    pub headers: A2aHeaders,

    /// Maximum body size in bytes for `StreamBuffer`.
    #[serde(default = "default_max_body_bytes")]
    pub max_body_bytes: usize,

    /// Method aliases for compatibility (slash-delimited → `PascalCase`).
    #[serde(default)]
    pub method_aliases: BTreeMap<String, String>,

    /// Invalid input handling behavior.
    #[serde(default = "OnInvalidBehavior::default_reject")]
    pub on_invalid: OnInvalidBehavior,

    /// Task-ownership routing configuration.
    #[serde(default)]
    pub task_routing: TaskRoutingConfig,
}

/// Default max body bytes.
fn default_max_body_bytes() -> usize {
    DEFAULT_MAX_BODY_BYTES
}

// -----------------------------------------------------------------------------
// Config Validation
// -----------------------------------------------------------------------------

/// Validate and build the final configuration.
pub(crate) fn build_config(cfg: A2aConfig) -> Result<A2aConfig, FilterError> {
    validate_max_body_bytes("a2a", cfg.max_body_bytes)?;

    validate_header_name("a2a", "context_id", cfg.headers.context_id.as_deref())?;
    validate_header_name("a2a", "method", cfg.headers.method.as_deref())?;
    validate_header_name("a2a", "family", cfg.headers.family.as_deref())?;
    validate_header_name("a2a", "task_id", cfg.headers.task_id.as_deref())?;
    validate_header_name("a2a", "kind", cfg.headers.kind.as_deref())?;
    validate_header_name("a2a", "streaming", cfg.headers.streaming.as_deref())?;
    validate_header_name("a2a", "version", cfg.headers.version.as_deref())?;

    if cfg.task_routing.enabled {
        validate_task_routing(&cfg.task_routing)?;
    }

    for (alias, canonical) in &cfg.method_aliases {
        if alias.is_empty() {
            return Err("a2a: alias key must be non-empty".into());
        }
        if canonical.is_empty() {
            return Err("a2a: alias value must be non-empty".into());
        }
        if !is_known_a2a_method(canonical) {
            return Err(format!("a2a: alias target '{canonical}' is not a known A2A method").into());
        }
    }

    Ok(cfg)
}

/// Validate task routing configuration.
fn validate_task_routing(tr: &TaskRoutingConfig) -> Result<(), FilterError> {
    if tr.ttl_seconds == 0 {
        return Err("a2a: task_routing.ttl_seconds must be greater than 0".into());
    }

    validate_max_body_bytes("a2a: task_routing", tr.max_response_body_bytes)?;

    validate_header_name(
        "a2a",
        "task_routing.route_cluster_header",
        Some(&tr.route_cluster_header),
    )?;

    // The route header must use the reserved x-praxis-a2a- prefix so that
    // the protocol layer's reserved-header rejection guard prevents clients
    // from injecting it directly.
    if !tr.route_cluster_header.starts_with("x-praxis-a2a-") {
        return Err(format!(
            "a2a: task_routing.route_cluster_header '{}' must start with 'x-praxis-a2a-'",
            tr.route_cluster_header
        )
        .into());
    }

    Ok(())
}

/// Check if a method is a known canonical A2A method.
fn is_known_a2a_method(method: &str) -> bool {
    matches!(
        method,
        "SendMessage"
            | "SendStreamingMessage"
            | "GetTask"
            | "ListTasks"
            | "CancelTask"
            | "SubscribeToTask"
            | "CreateTaskPushNotificationConfig"
            | "GetTaskPushNotificationConfig"
            | "ListTaskPushNotificationConfigs"
            | "DeleteTaskPushNotificationConfig"
            | "GetExtendedAgentCard"
    )
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::needless_raw_strings,
    clippy::needless_raw_string_hashes,
    reason = "tests"
)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Serde Defaults
    // -------------------------------------------------------------------------

    #[test]
    fn on_lookup_miss_defaults_to_continue() {
        let parsed: OnLookupMiss = serde_yaml::from_str("continue").unwrap();
        assert_eq!(parsed, OnLookupMiss::Continue, "explicit 'continue' should parse");
        assert_eq!(
            OnLookupMiss::default(),
            OnLookupMiss::Continue,
            "default should be Continue"
        );
    }

    #[test]
    fn task_route_store_defaults_to_local() {
        let parsed: TaskRouteStore = serde_yaml::from_str("local").unwrap();
        assert_eq!(parsed, TaskRouteStore::Local, "explicit 'local' should parse");
        assert_eq!(
            TaskRouteStore::default(),
            TaskRouteStore::Local,
            "default should be Local"
        );
    }

    #[test]
    fn task_routing_config_deserializes_with_defaults() {
        let cfg: TaskRoutingConfig = serde_yaml::from_str("{}").unwrap();

        assert!(!cfg.enabled, "enabled should default to false");
        assert_eq!(
            cfg.max_response_body_bytes, DEFAULT_MAX_RESPONSE_BODY_BYTES,
            "max_response_body_bytes should use default constant"
        );
        assert_eq!(
            cfg.on_lookup_miss,
            OnLookupMiss::Continue,
            "on_lookup_miss should default to Continue"
        );
        assert_eq!(
            cfg.route_cluster_header, DEFAULT_ROUTE_CLUSTER_HEADER,
            "route_cluster_header should use default constant"
        );
        assert_eq!(cfg.store, TaskRouteStore::Local, "store should default to Local");
        assert_eq!(
            cfg.terminal_ttl_seconds, DEFAULT_TERMINAL_TTL_SECONDS,
            "terminal_ttl_seconds should use default constant"
        );
        assert_eq!(
            cfg.ttl_seconds, DEFAULT_TTL_SECONDS,
            "ttl_seconds should use default constant"
        );
    }

    /// Assert all seven header defaults match the expected `x-praxis-a2a-*` values.
    fn assert_all_header_defaults(h: &A2aHeaders) {
        assert_eq!(h.context_id.as_deref(), Some("x-praxis-a2a-context-id"));
        assert_eq!(h.method.as_deref(), Some("x-praxis-a2a-method"));
        assert_eq!(h.family.as_deref(), Some("x-praxis-a2a-family"));
        assert_eq!(h.task_id.as_deref(), Some("x-praxis-a2a-task-id"));
        assert_eq!(h.kind.as_deref(), Some("x-praxis-a2a-kind"));
        assert_eq!(h.streaming.as_deref(), Some("x-praxis-a2a-streaming"));
        assert_eq!(h.version.as_deref(), Some("x-praxis-a2a-version"));
    }

    #[test]
    fn a2a_headers_deserializes_with_defaults() {
        let headers: A2aHeaders = serde_yaml::from_str("{}").unwrap();
        assert_all_header_defaults(&headers);
    }

    #[test]
    fn a2a_config_deserializes_with_defaults() {
        let cfg: A2aConfig = serde_yaml::from_str("{}").unwrap();

        assert_eq!(
            cfg.max_body_bytes, DEFAULT_MAX_BODY_BYTES,
            "max_body_bytes should use default constant"
        );
        assert!(cfg.method_aliases.is_empty(), "method_aliases should default to empty");
        assert_eq!(
            cfg.on_invalid,
            OnInvalidBehavior::Reject,
            "on_invalid should default to Reject"
        );
        assert!(
            !cfg.task_routing.enabled,
            "task_routing.enabled should default to false"
        );
    }

    // -------------------------------------------------------------------------
    // Serde deny_unknown_fields
    // -------------------------------------------------------------------------

    #[test]
    fn task_routing_config_rejects_unknown_fields() {
        let result = serde_yaml::from_str::<TaskRoutingConfig>("bogus_field: true");
        assert!(result.is_err(), "unknown field should be rejected");
    }

    #[test]
    fn a2a_headers_rejects_unknown_fields() {
        let result = serde_yaml::from_str::<A2aHeaders>("bogus_field: x-foo");
        assert!(result.is_err(), "unknown field should be rejected");
    }

    #[test]
    fn a2a_config_rejects_unknown_fields() {
        let result = serde_yaml::from_str::<A2aConfig>("bogus_field: true");
        assert!(result.is_err(), "unknown field should be rejected");
    }

    // -------------------------------------------------------------------------
    // A2aHeaders with null (disabled) headers
    // -------------------------------------------------------------------------

    #[test]
    fn null_header_produces_none() {
        let headers: A2aHeaders = serde_yaml::from_str(
            r"
context_id: ~
method: ~
family: ~
task_id: ~
kind: ~
streaming: ~
version: ~
",
        )
        .unwrap();

        assert!(headers.context_id.is_none(), "null context_id should be None");
        assert!(headers.method.is_none(), "null method should be None");
        assert!(headers.family.is_none(), "null family should be None");
        assert!(headers.task_id.is_none(), "null task_id should be None");
        assert!(headers.kind.is_none(), "null kind should be None");
        assert!(headers.streaming.is_none(), "null streaming should be None");
        assert!(headers.version.is_none(), "null version should be None");
    }

    // -------------------------------------------------------------------------
    // A2aHeaders Default trait
    // -------------------------------------------------------------------------

    #[test]
    fn a2a_headers_default_trait_matches_serde_defaults() {
        let from_default = A2aHeaders::default();
        let from_serde: A2aHeaders = serde_yaml::from_str("{}").unwrap();

        // Both paths should produce the same canonical header names.
        assert_all_header_defaults(&from_default);
        assert_all_header_defaults(&from_serde);
    }

    // -------------------------------------------------------------------------
    // build_config — valid
    // -------------------------------------------------------------------------

    #[test]
    fn build_config_minimal_valid() {
        let cfg: A2aConfig = serde_yaml::from_str("{}").unwrap();
        let result = build_config(cfg);
        assert!(result.is_ok(), "minimal config should be valid");
    }

    // -------------------------------------------------------------------------
    // build_config — max_body_bytes
    // -------------------------------------------------------------------------

    #[test]
    fn build_config_rejects_zero_max_body_bytes() {
        let cfg: A2aConfig = serde_yaml::from_str("max_body_bytes: 0").unwrap();
        let result = build_config(cfg);
        assert!(result.is_err(), "zero max_body_bytes should be rejected");
    }

    // -------------------------------------------------------------------------
    // build_config — header validation
    // -------------------------------------------------------------------------

    #[test]
    fn build_config_rejects_invalid_header_name() {
        let cfg: A2aConfig = serde_yaml::from_str(
            r"
headers:
  context_id: 'invalid header name'
",
        )
        .unwrap();
        let result = build_config(cfg);
        assert!(result.is_err(), "header name with spaces should be rejected");
    }

    #[test]
    fn build_config_accepts_null_headers() {
        let cfg: A2aConfig = serde_yaml::from_str(
            r"
headers:
  context_id: ~
  method: ~
  family: ~
  task_id: ~
  kind: ~
  streaming: ~
  version: ~
",
        )
        .unwrap();
        let result = build_config(cfg);
        assert!(result.is_ok(), "null (disabled) headers should be accepted");
    }

    // -------------------------------------------------------------------------
    // build_config — alias validation
    // -------------------------------------------------------------------------

    #[test]
    fn build_config_rejects_empty_alias_key() {
        let cfg: A2aConfig = serde_yaml::from_str(
            r#"
method_aliases:
  "": SendMessage
"#,
        )
        .unwrap();
        let result = build_config(cfg);
        assert!(result.is_err(), "empty alias key should be rejected");
    }

    #[test]
    fn build_config_rejects_empty_alias_value() {
        let cfg: A2aConfig = serde_yaml::from_str(
            r#"
method_aliases:
  send: ""
"#,
        )
        .unwrap();
        let result = build_config(cfg);
        assert!(result.is_err(), "empty alias value should be rejected");
    }

    #[test]
    fn build_config_rejects_unknown_canonical_method() {
        let cfg: A2aConfig = serde_yaml::from_str(
            r"
method_aliases:
  send: NotARealMethod
",
        )
        .unwrap();
        let result = build_config(cfg);
        assert!(
            result.is_err(),
            "alias target that is not a known A2A method should be rejected"
        );
    }

    #[test]
    fn build_config_accepts_valid_alias() {
        let cfg: A2aConfig = serde_yaml::from_str(
            r"
method_aliases:
  message/send: SendMessage
  tasks/get: GetTask
",
        )
        .unwrap();
        let result = build_config(cfg);
        assert!(
            result.is_ok(),
            "valid aliases mapping to known methods should be accepted"
        );
    }

    // -------------------------------------------------------------------------
    // is_known_a2a_method
    // -------------------------------------------------------------------------

    #[test]
    fn all_known_methods_recognized() {
        let known = [
            "SendMessage",
            "SendStreamingMessage",
            "GetTask",
            "ListTasks",
            "CancelTask",
            "SubscribeToTask",
            "CreateTaskPushNotificationConfig",
            "GetTaskPushNotificationConfig",
            "ListTaskPushNotificationConfigs",
            "DeleteTaskPushNotificationConfig",
            "GetExtendedAgentCard",
        ];
        for method in &known {
            assert!(
                is_known_a2a_method(method),
                "{method} should be recognized as a known A2A method"
            );
        }
    }

    #[test]
    fn unknown_method_not_recognized() {
        assert!(
            !is_known_a2a_method("NotARealMethod"),
            "unknown method should not be recognized"
        );
        assert!(
            !is_known_a2a_method("sendmessage"),
            "lowercase variant should not be recognized"
        );
        assert!(!is_known_a2a_method(""), "empty string should not be recognized");
    }

    // -------------------------------------------------------------------------
    // validate_task_routing
    // -------------------------------------------------------------------------

    #[test]
    fn validate_task_routing_rejects_zero_ttl() {
        let tr = TaskRoutingConfig {
            enabled: true,
            ttl_seconds: 0,
            ..TaskRoutingConfig::default()
        };
        let result = validate_task_routing(&tr);
        assert!(result.is_err(), "ttl_seconds=0 should be rejected");
    }

    #[test]
    fn validate_task_routing_rejects_zero_max_response_body_bytes() {
        let tr = TaskRoutingConfig {
            enabled: true,
            max_response_body_bytes: 0,
            ..TaskRoutingConfig::default()
        };
        let result = validate_task_routing(&tr);
        assert!(result.is_err(), "max_response_body_bytes=0 should be rejected");
    }

    #[test]
    fn validate_task_routing_rejects_bad_route_cluster_header_prefix() {
        let tr = TaskRoutingConfig {
            enabled: true,
            route_cluster_header: "x-custom-header".to_owned(),
            ..TaskRoutingConfig::default()
        };
        let result = validate_task_routing(&tr);
        assert!(
            result.is_err(),
            "route_cluster_header without x-praxis-a2a- prefix should be rejected"
        );
    }

    #[test]
    fn validate_task_routing_valid_config() {
        let tr = TaskRoutingConfig::default();
        // Default has ttl_seconds > 0, valid header prefix, valid body bytes.
        let result = validate_task_routing(&tr);
        assert!(result.is_ok(), "default TaskRoutingConfig should be valid");
    }

    #[test]
    fn build_config_validates_task_routing_when_enabled() {
        let cfg: A2aConfig = serde_yaml::from_str(
            r"
task_routing:
  enabled: true
  ttl_seconds: 0
",
        )
        .unwrap();
        let result = build_config(cfg);
        assert!(
            result.is_err(),
            "enabled task_routing with ttl_seconds=0 should be rejected"
        );
    }

    #[test]
    fn build_config_skips_task_routing_validation_when_disabled() {
        let cfg: A2aConfig = serde_yaml::from_str(
            r"
task_routing:
  enabled: false
  ttl_seconds: 0
",
        )
        .unwrap();
        let result = build_config(cfg);
        assert!(
            result.is_ok(),
            "disabled task_routing should skip validation even with invalid ttl"
        );
    }

    // -------------------------------------------------------------------------
    // TaskRoutingConfig defaults match constants
    // -------------------------------------------------------------------------

    #[test]
    fn task_routing_config_default_matches_constants() {
        let cfg = TaskRoutingConfig::default();

        assert_eq!(cfg.ttl_seconds, 3_600, "default ttl_seconds should be 3600");
        assert_eq!(
            cfg.terminal_ttl_seconds, 300,
            "default terminal_ttl_seconds should be 300"
        );
        assert_eq!(
            cfg.max_response_body_bytes, 65_536,
            "default max_response_body_bytes should be 65536"
        );
        assert_eq!(
            cfg.route_cluster_header, "x-praxis-a2a-route-cluster",
            "default route_cluster_header should be x-praxis-a2a-route-cluster"
        );
    }
}
