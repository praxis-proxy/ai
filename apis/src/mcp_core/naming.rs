// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Model-facing names for MCP tools.
//!
//! Inference backends have no concept of MCP servers, so every protocol
//! adapter exposes a resolved `(server_label, tool_name)` pair as one plain
//! function or tool name and maps model-emitted calls back through
//! [`McpToolIndex`]. The encoding is shared so that admission, classification,
//! and dispatch agree on the same name for every protocol.

use std::collections::HashMap;

/// Maximum length for generated function names.
///
/// This matches the OpenAI function-name schema (`^[a-zA-Z0-9_-]+$`, max 64
/// chars), which also bounds Chat Completions translation targets and fits
/// within Anthropic's custom tool-name limit.
pub(crate) const MAX_FUNCTION_NAME_LEN: usize = 64;

/// Encode `(label, tool_name)` into a bounded, schema-valid
/// function name matching `^[a-zA-Z0-9_-]+$` with max 64 chars.
///
/// The `{label}__{tool_name}` prefix is required for dispatch:
/// the backend has no concept of MCP servers, so the proxy must
/// embed the server identity in the function name to route
/// tool-call responses back to the correct upstream server.
///
/// Replaces invalid characters with `_` and truncates to fit
/// within [`MAX_FUNCTION_NAME_LEN`]. Lossy: distinct inputs can
/// produce the same output. Each adapter must reject collisions after
/// building the full tools array to catch this.
pub(crate) fn encode_function_name(label: &str, tool_name: &str) -> String {
    // Stop at the schema limit while building the string. Truncating a long
    // allocation would leave its full capacity retained in every reference.
    let mut sanitized = String::with_capacity(MAX_FUNCTION_NAME_LEN);
    for character in label
        .chars()
        .chain("__".chars())
        .chain(tool_name.chars())
        .take(MAX_FUNCTION_NAME_LEN)
    {
        sanitized.push(
            if character.is_ascii_alphanumeric() || character == '_' || character == '-' {
                character
            } else {
                '_'
            },
        );
    }
    sanitized
}

/// Reverse lookup for model-facing MCP function names.
///
/// Building this once per filter phase avoids repeatedly encoding every
/// resolved `(server_label, tool_name)` pair for each model-emitted call.
/// Lossy name collisions remain explicit so dispatch can fail closed.
pub(crate) struct McpToolIndex<'a> {
    /// Encoded function names mapped to their unique entry or collision count.
    entries: HashMap<String, McpToolMatch<'a>>,
}

/// Resolution state for one encoded MCP function name.
#[derive(Clone, Copy)]
pub(crate) enum McpToolMatch<'a> {
    /// Exactly one resolved MCP tool owns the encoded name.
    Unique {
        /// Original `(server_label, tool_name)` key.
        key: &'a (String, String),
        /// Resolved dispatch metadata.
        entry: &'a serde_json::Value,
    },
    /// Multiple tools collapsed to the same lossy encoded name.
    Ambiguous {
        /// Number of colliding tools.
        count: usize,
    },
}

impl<'a> McpToolIndex<'a> {
    /// Build a reverse index for one resolved MCP tool map.
    pub(crate) fn new(tool_map: &'a HashMap<(String, String), serde_json::Value>) -> Self {
        let mut entries = HashMap::with_capacity(tool_map.len());
        for (key @ (label, tool_name), entry) in tool_map {
            let encoded_name = encode_function_name(label, tool_name);
            entries
                .entry(encoded_name)
                .and_modify(|existing| {
                    let count = match existing {
                        McpToolMatch::Unique { .. } => 2,
                        McpToolMatch::Ambiguous { count } => count.saturating_add(1),
                    };
                    *existing = McpToolMatch::Ambiguous { count };
                })
                .or_insert(McpToolMatch::Unique { key, entry });
        }
        Self { entries }
    }

    /// Return whether any resolved MCP tool owns `encoded_name`.
    pub(crate) fn contains(&self, encoded_name: &str) -> bool {
        self.entries.contains_key(encoded_name)
    }

    /// Return the unique entry or collision state for `encoded_name`.
    pub(crate) fn get(&self, encoded_name: &str) -> Option<McpToolMatch<'a>> {
        self.entries.get(encoded_name).copied()
    }
}
