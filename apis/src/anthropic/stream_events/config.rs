// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Configuration for the Anthropic stream events filter.

use praxis_filter::{
    FilterError,
    body::{DEFAULT_JSON_BODY_MAX_BYTES, MAX_JSON_BODY_BYTES},
};
use serde::Deserialize;

// -----------------------------------------------------------------------------
// AnthropicStreamEventsConfig
// -----------------------------------------------------------------------------

/// YAML configuration for the [`AnthropicStreamEventsFilter`].
///
/// [`AnthropicStreamEventsFilter`]: super::AnthropicStreamEventsFilter
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct AnthropicStreamEventsConfig {
    /// Maximum incomplete SSE event bytes retained between chunks.
    #[serde(default = "default_max_partial_event_bytes")]
    pub max_partial_event_bytes: usize,

    /// Maximum number of distinct streaming tool-call content blocks
    /// retained per response. Each tool-call index an upstream streams
    /// pins per-block state for the response's lifetime, so an upstream
    /// that emits unbounded unique indices would grow memory without
    /// limit. Exceeding this cap fails the stream closed. Default:
    /// 10,000.
    #[serde(default = "default_max_tool_blocks")]
    pub max_tool_blocks: usize,
}

/// Default maximum partial event bytes.
fn default_max_partial_event_bytes() -> usize {
    DEFAULT_JSON_BODY_MAX_BYTES
}

/// Default maximum retained streaming tool-call content blocks.
fn default_max_tool_blocks() -> usize {
    DEFAULT_MAX_TOOL_BLOCKS
}

/// Default cap on distinct streaming tool-call content blocks per response.
const DEFAULT_MAX_TOOL_BLOCKS: usize = 10_000;

// -----------------------------------------------------------------------------
// Config Validation
// -----------------------------------------------------------------------------

/// Validate the parsed configuration.
pub(crate) fn build_config(cfg: AnthropicStreamEventsConfig) -> Result<AnthropicStreamEventsConfig, FilterError> {
    validate_max_partial_event_bytes(cfg.max_partial_event_bytes)?;
    validate_max_tool_blocks(cfg.max_tool_blocks)?;
    Ok(cfg)
}

/// Validate the maximum retained tool-call content block count.
fn validate_max_tool_blocks(value: usize) -> Result<(), FilterError> {
    if value == 0 {
        return Err("anthropic_stream_events: 'max_tool_blocks' must be greater than 0".into());
    }

    Ok(())
}

/// Validate the maximum partial SSE event byte limit.
fn validate_max_partial_event_bytes(value: usize) -> Result<(), FilterError> {
    if value == 0 {
        return Err("anthropic_stream_events: 'max_partial_event_bytes' must be greater than 0".into());
    }

    if value > MAX_JSON_BODY_BYTES {
        return Err(format!(
            "anthropic_stream_events: max_partial_event_bytes ({value}) exceeds maximum ({MAX_JSON_BODY_BYTES})"
        )
        .into());
    }

    Ok(())
}
