// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Deserialized YAML configuration types for the external metering filter.

use praxis_filter::FilterError;
use serde::Deserialize;

/// Default header prefix for tenant identity headers.
const DEFAULT_IDENTITY_HEADER_PREFIX: &str = "x-tenant-";

/// Deserialized YAML config for the `external_metering` filter.
///
/// ```yaml
/// filter: external_metering
/// identity_header_prefix: "x-tenant-"
/// ```
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ExternalMeteringConfig {
    /// Prefix for tenant identity headers to strip.
    /// Expected headers: `{prefix}username`, `{prefix}group`,
    /// `{prefix}subscription`, `{prefix}model`.
    #[serde(default = "default_identity_header_prefix")]
    pub identity_header_prefix: String,
}

/// Validate config at construction time.
pub(super) fn validate_config(cfg: &ExternalMeteringConfig) -> Result<(), FilterError> {
    if cfg.identity_header_prefix.is_empty() {
        return Err("external_metering: identity_header_prefix must not be empty".into());
    }

    Ok(())
}

/// Serde default for `identity_header_prefix`.
fn default_identity_header_prefix() -> String {
    DEFAULT_IDENTITY_HEADER_PREFIX.to_owned()
}
