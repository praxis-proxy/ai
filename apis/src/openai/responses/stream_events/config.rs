// SPDX-License-Identifier: Apache-2.0
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

    /// Maximum seconds from the first SSE chunk to stream completion.
    ///
    /// The parser enforces this absolute deadline when chunks or
    /// end-of-stream arrive. Before the first chunk, cluster
    /// `read_timeout_ms` applies; `timeout_secs` does not start until
    /// the first SSE chunk. Each later chunk calls
    /// [`praxis_filter::HttpFilterContext::cap_stream_read_timeout`] with the
    /// remaining time so downstream backpressure cannot restart a relative
    /// per-read timer past the deadline. A tighter cluster `read_timeout_ms`
    /// is left in place. Default: 300 (5 minutes).
    #[serde(default)]
    pub timeout_secs: Option<u64>,

    /// Maximum bytes accepted per function-call argument string from
    /// `function_call_arguments.delta` or `function_call_arguments.done` events.
    /// Default: 1 MiB.
    #[serde(default)]
    pub max_tool_call_argument_bytes: Option<usize>,

    /// Maximum aggregate bytes accumulated across streaming output items and
    /// function-call argument buffers before the stream fails closed. Bounds
    /// total process memory even when every individual event is within
    /// `max_buffer_bytes`. Default: 64 MiB.
    #[serde(default)]
    pub max_accumulated_bytes: Option<usize>,

    /// Maximum number of streaming output items accumulated before the stream
    /// fails closed. Default: 100,000.
    #[serde(default)]
    pub max_output_items: Option<usize>,
}

/// Default cap for accumulated function-call argument bytes (1 MiB).
const DEFAULT_MAX_TOOL_CALL_ARGUMENT_BYTES: usize = 1024 * 1024;

/// Default aggregate accumulation ceiling (64 MiB), matching the JSON body cap.
const DEFAULT_MAX_ACCUMULATED_BYTES: usize = MAX_JSON_BODY_BYTES;

/// Default cap on accumulated streaming output items.
const DEFAULT_MAX_OUTPUT_ITEMS: usize = 100_000;

/// Hard upper bound on the configurable SSE event count.
const MAX_EVENTS_CEILING: usize = 10_000_000;

/// Hard upper bound on the configurable output-item count.
const MAX_OUTPUT_ITEMS_CEILING: usize = 10_000_000;

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
            reject_above(v, "max_events", MAX_EVENTS_CEILING)?;
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
        if let Some(v) = self.max_accumulated_bytes {
            reject_zero(v, "max_accumulated_bytes")?;
            reject_above_max(v, "max_accumulated_bytes")?;
        }
        if let Some(v) = self.max_output_items {
            reject_zero(v, "max_output_items")?;
            reject_above(v, "max_output_items", MAX_OUTPUT_ITEMS_CEILING)?;
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

    /// Resolved aggregate accumulation byte ceiling.
    pub(crate) fn max_accumulated_bytes(&self) -> usize {
        self.max_accumulated_bytes.unwrap_or(DEFAULT_MAX_ACCUMULATED_BYTES)
    }

    /// Resolved cap on accumulated streaming output items.
    pub(crate) fn max_output_items(&self) -> usize {
        self.max_output_items.unwrap_or(DEFAULT_MAX_OUTPUT_ITEMS)
    }
}

/// Reject a zero value for a named configuration field.
fn reject_zero(value: usize, field: &str) -> Result<(), FilterError> {
    if value == 0 {
        return Err(format!("{FILTER_NAME}: '{field}' must be greater than 0").into());
    }
    Ok(())
}

/// Reject a value for `field` that exceeds `max`.
fn reject_above(value: usize, field: &str, max: usize) -> Result<(), FilterError> {
    if value > max {
        return Err(format!("{FILTER_NAME}: {field} ({value}) exceeds maximum ({max})").into());
    }
    Ok(())
}

/// Reject a byte-cap value that exceeds `MAX_JSON_BODY_BYTES` (64 MiB).
fn reject_above_max(value: usize, field: &str) -> Result<(), FilterError> {
    reject_above(value, field, MAX_JSON_BODY_BYTES)
}
