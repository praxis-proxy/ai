// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Configuration for the identity header guard filter.

use serde::Deserialize;

// -----------------------------------------------------------------------------
// IdentityHeaderGuardConfig
// -----------------------------------------------------------------------------

/// Deserialized YAML config for the identity header guard filter.
///
/// ```yaml
/// filter: identity_header_guard
/// prefix: "x-tenant-"
/// metadata_namespace: "identity"
/// ```
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct IdentityHeaderGuardConfig {
    /// Case-insensitive header name prefix to capture and strip.
    pub prefix: String,

    /// Metadata namespace for captured headers.
    /// Headers are stored as `{namespace}.{header_name}`.
    #[serde(default = "default_namespace")]
    pub metadata_namespace: String,
}

/// Returns the default metadata namespace (`identity`).
fn default_namespace() -> String {
    "identity".to_owned()
}

// -----------------------------------------------------------------------------
// Validation
// -----------------------------------------------------------------------------

/// Validate an [`IdentityHeaderGuardConfig`], returning an error on missing required fields.
pub(super) fn validate_config(config: &IdentityHeaderGuardConfig) -> Result<(), String> {
    if config.prefix.is_empty() {
        return Err("identity_header_guard: prefix must not be empty".into());
    }
    if config.metadata_namespace.is_empty() {
        return Err("identity_header_guard: metadata_namespace must not be empty".into());
    }
    Ok(())
}
