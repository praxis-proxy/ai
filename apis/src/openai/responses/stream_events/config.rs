// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! YAML-facing configuration for the `openai_stream_events` filter.

use std::time::Duration;

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
}

impl StreamEventsConfig {
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
}
