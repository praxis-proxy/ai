// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Deserialized YAML configuration for the in-flight tracking filter.

use praxis_filter::FilterError;
use serde::Deserialize;

/// Model name attributed when neither config nor the request body supplies one.
pub(super) const DEFAULT_MODEL_FALLBACK: &str = "unknown";

/// YAML config for the `inflight_tracker` filter.
///
/// ```yaml
/// filter: inflight_tracker
/// default_model: "unknown"   # attributed when the body carries no model
/// ```
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct InFlightConfig {
    /// Model name attributed when the request body reveals none. Keeps counts
    /// from silently vanishing for clients that omit `model`.
    #[serde(default = "default_model")]
    pub default_model: String,
}

/// Validate config at construction time.
pub(super) fn validate_config(cfg: &InFlightConfig) -> Result<(), FilterError> {
    if cfg.default_model.is_empty() {
        return Err("inflight_tracker: default_model must not be empty".into());
    }
    Ok(())
}

/// Serde default for `default_model`.
fn default_model() -> String {
    DEFAULT_MODEL_FALLBACK.to_owned()
}
