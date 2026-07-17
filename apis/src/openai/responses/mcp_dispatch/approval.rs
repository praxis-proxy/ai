// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Approval policy parsing and evaluation for MCP tool calls.

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
