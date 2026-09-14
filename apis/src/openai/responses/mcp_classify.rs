// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Shared MCP tool-call classification.
//!
//! This module is the single source of truth for *deciding* how one
//! model-emitted tool call relates to the resolved MCP tool map: whether it is
//! an MCP call at all and, if so, whether it may execute automatically or must
//! be returned to the client for approval. It exposes a deliberately narrow
//! surface — the [`McpDisposition`] enum and the [`classify_mcp`] function —
//! so the agentic loop owner and every dispatcher (`openai_mcp_dispatch`,
//! `openai_file_search_callout`) can agree on the disposition of a call
//! without any dispatcher importing another dispatcher's internals.
//!
//! The approval-policy helpers ([`parse_approval_policy`], [`requires_approval`])
//! live here rather than in `openai_mcp_dispatch` so that this module stays
//! neutral: `classify_mcp` composes them, and `openai_mcp_dispatch` imports them
//! from here for its private [`PendingApproval`] construction.
//!
//! [`PendingApproval`]: super::mcp_dispatch

use super::openai_mcp_tool_resolve::{McpToolIndex, McpToolMatch};

/// Disposition of a single model-emitted tool call relative to the resolved MCP
/// tool map.
///
/// This is intentionally a unit-variant enum: it carries the *decision* only,
/// never the per-call data (`call_id`/`server_label`/`tool_name`/`arguments`)
/// that `openai_mcp_dispatch` needs to build a client-visible
/// `mcp_approval_request`. That construction stays private to the dispatcher.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum McpDisposition {
    /// An MCP tool call whose policy permits automatic execution.
    Automatic,
    /// An MCP tool call that must be returned to the client for approval, or
    /// one whose encoded name is ambiguous across servers (fail-closed).
    ApprovalRequired,
    /// Not an MCP tool call — the encoded name matches no resolved MCP tool.
    NotMcp,
}

/// Classify one model-emitted tool call against the resolved MCP tool index.
///
/// The branch order is load-bearing and mirrors the pre-existing dispatch logic
/// (`is_mcp_tool_call` + `check_single_approval`):
///
/// - **no match** → [`McpDisposition::NotMcp`];
/// - **more than one match** (a lossy encoded-name collision across servers) → [`McpDisposition::ApprovalRequired`],
///   failing closed *before* any policy lookup so an ambiguous call can never execute automatically;
/// - **exactly one match** → the per-tool `require_approval` policy decides [`McpDisposition::ApprovalRequired`] vs
///   [`McpDisposition::Automatic`], fail-closed to approval when the policy is absent or unrecognized.
pub(crate) fn classify_mcp(tool_call: &serde_json::Value, tool_index: &McpToolIndex<'_>) -> McpDisposition {
    let Some(encoded_name) = tool_call.get("name").and_then(serde_json::Value::as_str) else {
        return McpDisposition::NotMcp;
    };
    match tool_index.get(encoded_name) {
        None => McpDisposition::NotMcp,
        Some(McpToolMatch::Ambiguous { .. }) => McpDisposition::ApprovalRequired,
        Some(McpToolMatch::Unique { key, entry }) => {
            if requires_approval(&parse_approval_policy(entry), &key.1) {
                McpDisposition::ApprovalRequired
            } else {
                McpDisposition::Automatic
            }
        },
    }
}

// -----------------------------------------------------------------------------
// Approval policy
// -----------------------------------------------------------------------------

/// Approval policy for MCP tool execution.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ApprovalPolicy {
    /// Always require approval.
    Always,
    /// Never require approval.
    Never,
    /// Filter: named tools always/never require approval.
    Filter {
        /// Tools that always require approval.
        always: Vec<String>,
        /// Tools that never require approval.
        never: Vec<String>,
    },
}

/// Parse `require_approval` from an MCP tool definition.
///
/// Handles:
/// - `"always"` → `Always`
/// - `"never"` → `Never`
/// - `{"always": {"tool_names": [...]}, "never": {"tool_names": [...]}}` → `Filter`
/// - absent or unrecognized → `Always` (fail-closed default)
pub(crate) fn parse_approval_policy(tool_def: &serde_json::Value) -> ApprovalPolicy {
    let Some(value) = tool_def.get("require_approval") else {
        return ApprovalPolicy::Always;
    };

    if let Some(s) = value.as_str() {
        return match s {
            "never" => ApprovalPolicy::Never,
            _ => ApprovalPolicy::Always,
        };
    }

    if let Some(obj) = value.as_object() {
        let always = extract_tool_names(obj.get("always"));
        let never = extract_tool_names(obj.get("never"));
        return ApprovalPolicy::Filter { always, never };
    }

    ApprovalPolicy::Always
}

/// Check whether a tool call requires approval under the given
/// policy.
///
/// For `Filter`: `always` takes precedence over `never`. Tools
/// not in either list default to requiring approval.
pub(crate) fn requires_approval(policy: &ApprovalPolicy, tool_name: &str) -> bool {
    match policy {
        ApprovalPolicy::Always => true,
        ApprovalPolicy::Never => false,
        ApprovalPolicy::Filter { always, never } => {
            if always.iter().any(|n| n == tool_name) {
                return true;
            }
            if never.iter().any(|n| n == tool_name) {
                return false;
            }
            true
        },
    }
}

/// Extract tool names from an `MCPToolFilter` value.
///
/// Accepts both the canonical `{"tool_names": [...]}` object form
/// and a flat `[...]` array for resilience.
fn extract_tool_names(value: Option<&serde_json::Value>) -> Vec<String> {
    let Some(v) = value else {
        return Vec::new();
    };
    if let Some(obj) = v.as_object() {
        return obj
            .get("tool_names")
            .and_then(serde_json::Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(serde_json::Value::as_str)
                    .map(ToOwned::to_owned)
                    .collect()
            })
            .unwrap_or_default();
    }
    if let Some(arr) = v.as_array() {
        return arr
            .iter()
            .filter_map(serde_json::Value::as_str)
            .map(ToOwned::to_owned)
            .collect();
    }
    Vec::new()
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::panic, clippy::needless_pass_by_value, reason = "tests")]
mod tests {
    use std::collections::HashMap;

    use serde_json::json;

    use super::{McpDisposition, classify_mcp};
    use crate::openai::responses::openai_mcp_tool_resolve::McpToolIndex;

    /// Build a one-entry tool map for `(label, tool)` with the given approval policy.
    fn tool_map_with_policy(
        label: &str,
        tool: &str,
        require_approval: serde_json::Value,
    ) -> HashMap<(String, String), serde_json::Value> {
        let mut map = HashMap::new();
        map.insert(
            (label.to_owned(), tool.to_owned()),
            json!({
                "server_label": label,
                "server_url": "https://mcp.example",
                "require_approval": require_approval,
                "tool_definition": {"name": tool},
            }),
        );
        map
    }

    #[test]
    fn unknown_name_is_not_mcp() {
        let map = tool_map_with_policy("srv", "search", json!("never"));
        let index = McpToolIndex::new(&map);
        let call = json!({"type": "function_call", "name": "some_client_fn", "call_id": "c1"});
        assert_eq!(classify_mcp(&call, &index), McpDisposition::NotMcp);
    }

    #[test]
    fn missing_name_is_not_mcp() {
        let map = tool_map_with_policy("srv", "search", json!("never"));
        let index = McpToolIndex::new(&map);
        let call = json!({"type": "function_call", "call_id": "c1"});
        assert_eq!(classify_mcp(&call, &index), McpDisposition::NotMcp);
    }

    #[test]
    fn never_policy_is_automatic() {
        let map = tool_map_with_policy("srv", "search", json!("never"));
        let index = McpToolIndex::new(&map);
        let call = json!({"type": "function_call", "name": "srv__search", "call_id": "c1"});
        assert_eq!(classify_mcp(&call, &index), McpDisposition::Automatic);
    }

    #[test]
    fn always_policy_requires_approval() {
        let map = tool_map_with_policy("srv", "search", json!("always"));
        let index = McpToolIndex::new(&map);
        let call = json!({"type": "function_call", "name": "srv__search", "call_id": "c1"});
        assert_eq!(classify_mcp(&call, &index), McpDisposition::ApprovalRequired);
    }

    #[test]
    fn absent_policy_fails_closed_to_approval() {
        let map = tool_map_with_policy("srv", "search", serde_json::Value::Null);
        let index = McpToolIndex::new(&map);
        let call = json!({"type": "function_call", "name": "srv__search", "call_id": "c1"});
        assert_eq!(classify_mcp(&call, &index), McpDisposition::ApprovalRequired);
    }

    #[test]
    fn ambiguous_encoded_name_requires_approval() {
        // Two distinct servers whose lossy encoded names collide both map to the
        // same `server__tool` function name, so the call is ambiguous.
        let mut map = HashMap::new();
        for label in ["srv/one", "srv#one"] {
            map.insert(
                (label.to_owned(), "search".to_owned()),
                json!({
                    "server_label": label,
                    "server_url": "https://mcp.example",
                    "require_approval": "never",
                    "tool_definition": {"name": "search"},
                }),
            );
        }
        let index = McpToolIndex::new(&map);
        let call = json!({"type": "function_call", "name": "srv_one__search", "call_id": "c1"});
        assert_eq!(
            classify_mcp(&call, &index),
            McpDisposition::ApprovalRequired,
            "an ambiguous encoded name must fail closed to approval, not automatic"
        );
    }
}
