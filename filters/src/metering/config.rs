// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Deserialized YAML configuration types for the external metering filter.

use praxis_filter::FilterError;
use serde::Deserialize;

/// Default HTTP timeout for metering service calls (5 seconds).
const DEFAULT_TIMEOUT_SECONDS: u64 = 5;

/// Default entitlement feature key for balance checks.
const DEFAULT_FEATURE_KEY: &str = "inference-tokens";

/// Default `CloudEvents` `source` field.
const DEFAULT_SOURCE: &str = "ai-gateway";

/// Default header prefix for tenant identity headers.
const DEFAULT_IDENTITY_HEADER_PREFIX: &str = "x-tenant-";

/// Deserialized YAML config for the `external_metering` filter.
///
/// ```yaml
/// filter: external_metering
/// metering_url: "http://metering-service:8080"
/// timeout_seconds: 5
/// feature_key: "inference-tokens"
/// source: "ai-gateway"
/// fail_open: true
/// identity_header_prefix: "x-tenant-"
/// default_username: "anonymous"
/// default_model: "unknown"
/// ```
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ExternalMeteringConfig {
    /// Base URL of the external metering service (required).
    pub metering_url: String,

    /// HTTP timeout in seconds for all metering calls.
    #[serde(default = "default_timeout_seconds")]
    pub timeout_seconds: u64,

    /// Entitlement feature key used in balance check URL path.
    #[serde(default = "default_feature_key")]
    pub feature_key: String,

    /// `CloudEvents` `source` field value.
    #[serde(default = "default_source")]
    pub source: String,

    /// When `true` (default), requests proceed if the metering service
    /// is unavailable. When `false`, requests are rejected with 503.
    #[serde(default = "default_true")]
    pub fail_open: bool,

    /// Prefix for tenant identity headers to capture and strip.
    /// Expected headers: `{prefix}username`, `{prefix}group`,
    /// `{prefix}subscription`, `{prefix}model`.
    #[serde(default = "default_identity_header_prefix")]
    pub identity_header_prefix: String,

    /// Fallback username when no identity header is present.
    /// If set, requests without `{prefix}username` are still metered
    /// under this name. If unset, metering is skipped entirely.
    #[serde(default)]
    pub default_username: Option<String>,

    /// Fallback model name when no identity model header is present.
    #[serde(default)]
    pub default_model: Option<String>,
}

/// Validate config at construction time.
pub(super) fn validate_config(cfg: &ExternalMeteringConfig) -> Result<(), FilterError> {
    if cfg.metering_url.is_empty() {
        return Err("external_metering: metering_url must not be empty".into());
    }

    if cfg.timeout_seconds == 0 {
        return Err("external_metering: timeout_seconds must be greater than 0".into());
    }

    if cfg.identity_header_prefix.is_empty() {
        return Err("external_metering: identity_header_prefix must not be empty".into());
    }

    Ok(())
}

/// Serde default for `timeout_seconds`.
fn default_timeout_seconds() -> u64 {
    DEFAULT_TIMEOUT_SECONDS
}

/// Serde default for `feature_key`.
fn default_feature_key() -> String {
    DEFAULT_FEATURE_KEY.to_owned()
}

/// Serde default for `source`.
fn default_source() -> String {
    DEFAULT_SOURCE.to_owned()
}

/// Serde default for `fail_open`.
fn default_true() -> bool {
    true
}

/// Serde default for `identity_header_prefix`.
fn default_identity_header_prefix() -> String {
    DEFAULT_IDENTITY_HEADER_PREFIX.to_owned()
}
