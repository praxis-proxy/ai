// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Configuration types for the Responses proxy filter.

use praxis_filter::{FilterError, body::MAX_JSON_BODY_BYTES};
use serde::Deserialize;

use crate::openai::responses::body_limits::validate_size_limit;

// -----------------------------------------------------------------------------
// ResponsesProxyConfig
// -----------------------------------------------------------------------------

/// Deserialized YAML config for the Responses proxy filter.
///
/// ```yaml
/// filter: openai_responses_proxy
/// ```
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ResponsesProxyConfig {
    /// Maximum size in bytes of the request body this filter *produces*
    /// when it rebuilds the outbound body from `ResponsesState`.
    ///
    /// Raw request body size is governed by the pipeline's `body_limits`,
    /// not this field. This bounds only the rebuilt body, which can grow
    /// larger than the raw input when conversation history is rehydrated.
    #[serde(default = "default_max_rewritten_body_bytes")]
    pub max_rewritten_body_bytes: usize,
}

impl Default for ResponsesProxyConfig {
    fn default() -> Self {
        Self {
            max_rewritten_body_bytes: MAX_JSON_BODY_BYTES,
        }
    }
}

/// Serde default for `max_rewritten_body_bytes`.
fn default_max_rewritten_body_bytes() -> usize {
    MAX_JSON_BODY_BYTES
}

// -----------------------------------------------------------------------------
// Config Validation
// -----------------------------------------------------------------------------

/// Validate the parsed configuration.
pub(super) fn build_config(cfg: ResponsesProxyConfig) -> Result<ResponsesProxyConfig, FilterError> {
    validate_size_limit(
        "openai_responses_proxy",
        "max_rewritten_body_bytes",
        cfg.max_rewritten_body_bytes,
    )?;
    Ok(cfg)
}
