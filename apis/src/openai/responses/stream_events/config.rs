// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! YAML-facing configuration for the `openai_stream_events` filter.

use std::time::Duration;

use praxis_filter::{FilterError, body::MAX_JSON_BODY_BYTES};
use serde::Deserialize;

use crate::openai::sse::SseParserConfig;

/// Configuration for the `openai_stream_events` filter.
///
/// All fields are optional; omitted values fall back to
/// [`SseParserConfig`] defaults.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct StreamEventsConfig {
    /// Maximum bytes buffered for incomplete SSE lines/data across
    /// chunk boundaries. Default: 10 MiB.
    #[serde(default)]
    pub max_buffer_bytes: Option<usize>,

    /// Maximum number of SSE events before the parser errors.
    /// Default: 100,000.
    #[serde(default)]
    pub max_events: Option<usize>,

    /// Maximum seconds from first chunk to stream completion.
    /// Default: 300 (5 minutes).
    #[serde(default)]
    pub timeout_secs: Option<u64>,

    /// Maximum bytes accumulated per function-call argument string
    /// from `function_call_arguments.delta` events. Default: 1 MiB.
    #[serde(default)]
    pub max_tool_call_argument_bytes: Option<usize>,
}

/// Default cap for accumulated function-call argument bytes (1 MiB).
const DEFAULT_MAX_TOOL_CALL_ARGUMENT_BYTES: usize = 1024 * 1024;

/// Filter name used in validation error messages.
const FILTER_NAME: &str = "openai_stream_events";

impl StreamEventsConfig {
    /// Validate explicitly set values. Omitted fields use safe defaults.
    pub(crate) fn validate(&self) -> Result<(), FilterError> {
        if let Some(v) = self.max_buffer_bytes {
            reject_zero(v, "max_buffer_bytes")?;
            reject_above_max(v, "max_buffer_bytes")?;
        }
        if let Some(v) = self.max_events {
            reject_zero(v, "max_events")?;
        }
        if let Some(v) = self.timeout_secs
            && v == 0
        {
            return Err(format!("{FILTER_NAME}: 'timeout_secs' must be greater than 0").into());
        }
        if let Some(v) = self.max_tool_call_argument_bytes {
            reject_zero(v, "max_tool_call_argument_bytes")?;
            reject_above_max(v, "max_tool_call_argument_bytes")?;
        }
        Ok(())
    }

    /// Convert to the internal parser config, applying defaults for
    /// any omitted fields.
    pub(crate) fn to_parser_config(&self) -> SseParserConfig {
        let defaults = SseParserConfig::default();
        SseParserConfig {
            max_buffer_bytes: self.max_buffer_bytes.unwrap_or(defaults.max_buffer_bytes),
            max_events: self.max_events.unwrap_or(defaults.max_events),
            timeout: self.timeout_secs.map_or(defaults.timeout, Duration::from_secs),
        }
    }

    /// Resolved cap for per-tool-call accumulated argument bytes.
    pub(crate) fn max_tool_call_argument_bytes(&self) -> usize {
        self.max_tool_call_argument_bytes
            .unwrap_or(DEFAULT_MAX_TOOL_CALL_ARGUMENT_BYTES)
    }
}

/// Reject a zero value for a named configuration field.
fn reject_zero(value: usize, field: &str) -> Result<(), FilterError> {
    if value == 0 {
        return Err(format!("{FILTER_NAME}: '{field}' must be greater than 0").into());
    }
    Ok(())
}

/// Reject a byte-cap value that exceeds `MAX_JSON_BODY_BYTES` (64 MiB).
fn reject_above_max(value: usize, field: &str) -> Result<(), FilterError> {
    if value > MAX_JSON_BODY_BYTES {
        return Err(format!("{FILTER_NAME}: {field} ({value}) exceeds maximum ({MAX_JSON_BODY_BYTES})").into());
    }
    Ok(())
}
