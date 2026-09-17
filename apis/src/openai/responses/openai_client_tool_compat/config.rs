// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Configuration for the `openai_client_tool_compat` filter.

use praxis_filter::{FilterError, body::MAX_JSON_BODY_BYTES};
use serde::Deserialize;

use crate::openai::responses::body_limits::validate_size_limit;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Default maximum number of client-owned tool declarations lowered per request.
///
/// Bounds the size of the per-request reverse map (`client_tool_lowering`) and
/// the pre-lowering echo snapshot so a request with an enormous `tools` array
/// cannot grow request-scoped state without limit.
const DEFAULT_MAX_CLIENT_TOOLS: usize = 512;

// -----------------------------------------------------------------------------
// ClientToolCompatConfig
// -----------------------------------------------------------------------------

/// YAML configuration for the `openai_client_tool_compat` filter.
///
/// Enablement is by placement in the pipeline routed to a function-only
/// Responses backend, not by a "target backend" flag. Config fields bound
/// behaviour (byte and tool-count caps) only, mirroring the other Responses
/// rewriter filters.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ClientToolCompatConfig {
    /// Maximum size in bytes of the request body this filter *produces* after
    /// lowering rich client tool declarations to `function` entries, and of the
    /// response body it *produces* after restoring typed client-owned items.
    ///
    /// Raw request body size is governed by the pipeline's `body_limits`, not
    /// this field. Lowering absorbs provider fields into function descriptions,
    /// so the rewritten body can grow larger than the raw input.
    #[serde(default = "default_max_rewritten_body_bytes")]
    pub max_rewritten_body_bytes: usize,

    /// Maximum number of client-owned tool declarations lowered per request.
    #[serde(default = "default_max_client_tools")]
    pub max_client_tools: usize,
}

impl Default for ClientToolCompatConfig {
    fn default() -> Self {
        Self {
            max_rewritten_body_bytes: default_max_rewritten_body_bytes(),
            max_client_tools: default_max_client_tools(),
        }
    }
}

/// Default max rewritten/restored body bytes (64 MiB): a post-rewrite backstop.
fn default_max_rewritten_body_bytes() -> usize {
    MAX_JSON_BODY_BYTES
}

/// Default max client tools lowered per request.
fn default_max_client_tools() -> usize {
    DEFAULT_MAX_CLIENT_TOOLS
}

/// Validate the parsed configuration.
pub(crate) fn build_config(cfg: ClientToolCompatConfig) -> Result<ClientToolCompatConfig, FilterError> {
    validate_size_limit(
        "openai_client_tool_compat",
        "max_rewritten_body_bytes",
        cfg.max_rewritten_body_bytes,
    )?;
    if cfg.max_client_tools == 0 {
        return Err("openai_client_tool_compat: max_client_tools must be greater than 0".into());
    }
    Ok(cfg)
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, reason = "tests")]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        let cfg = build_config(ClientToolCompatConfig::default()).unwrap();
        assert_eq!(cfg.max_rewritten_body_bytes, MAX_JSON_BODY_BYTES);
        assert_eq!(cfg.max_client_tools, DEFAULT_MAX_CLIENT_TOOLS);
    }

    #[test]
    fn rejects_zero_rewritten_body_bytes() {
        let cfg = ClientToolCompatConfig {
            max_rewritten_body_bytes: 0,
            max_client_tools: 1,
        };
        let err = build_config(cfg).unwrap_err();
        assert!(
            err.to_string().contains("max_rewritten_body_bytes"),
            "the error names the offending field: {err}"
        );
    }

    #[test]
    fn rejects_zero_max_client_tools() {
        let cfg = ClientToolCompatConfig {
            max_rewritten_body_bytes: MAX_JSON_BODY_BYTES,
            max_client_tools: 0,
        };
        let err = build_config(cfg).unwrap_err();
        assert!(
            err.to_string().contains("max_client_tools"),
            "the error names the offending field: {err}"
        );
    }

    #[test]
    fn rejects_above_ceiling() {
        let cfg = ClientToolCompatConfig {
            max_rewritten_body_bytes: MAX_JSON_BODY_BYTES + 1,
            max_client_tools: 1,
        };
        let err = build_config(cfg).unwrap_err();
        assert!(
            err.to_string().contains("exceeds maximum"),
            "the error explains the ceiling: {err}"
        );
    }

    #[test]
    fn rejects_unknown_field() {
        let err = serde_yaml::from_str::<ClientToolCompatConfig>("unknown_field: 1")
            .expect_err("deny_unknown_fields must reject an unknown key");
        assert!(
            err.to_string().contains("unknown_field"),
            "the error names the unknown field: {err}"
        );
    }
}
