// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Resolved MCP server entries and the `(server_label, tool_name)` tool map.
//!
//! Protocol adapters translate their request's MCP declarations into internal
//! entry objects carrying `server_label`, and either an operator `connector_id`
//! or a client-selected `server_url`, plus optional `headers`,
//! `authorization`, `allowed_tools`, and `require_approval`. This module owns
//! the protocol-neutral policy over those entries: connector resolution and
//! the `require_connector` gate, client tool selection, operator admission,
//! and the dispatch tool map the executor reads.

use std::collections::{HashMap, HashSet};

use super::{
    admission::{
        ConnectorToolPolicy, McpAdmissionDecisions, McpAdmissionPolicy, McpDenyReason, McpToolVerdict, evaluate_tool,
    },
    connector::MAX_CONNECTOR_ID_LEN,
};

/// Client-correctable failure of MCP entry resolution.
#[derive(Debug, thiserror::Error)]
pub(crate) enum McpToolMapError {
    /// Generated function names collide after sanitization or
    /// truncation.
    #[error("generated function name collision: \"{0}\" maps to multiple tools")]
    NameCollision(String),

    /// Unknown `connector_id` was referenced.
    #[error("unknown connector_id \"{0}\"")]
    UnknownConnector(String),

    /// A direct client-selected `server_url` was supplied while connector-only
    /// routing is enabled.
    #[error(
        "Direct MCP server URLs are disabled by your administrator. Use an administrator-configured connector_id instead."
    )]
    DirectServerUrl,

    /// Both `connector_id` and `server_url` were provided.
    #[error("connector_id and server_url are mutually exclusive on entry \"{0}\"")]
    MutuallyExclusiveTarget(String),

    /// Invalid `connector_id` value (not a non-empty string).
    #[error("connector_id must be a non-empty string")]
    InvalidConnectorId,

    /// Connector entry missing required `server_label`.
    #[error("connector_id requires a non-empty server_label on entry \"{0}\"")]
    MissingConnectorLabel(String),

    /// `allowed_tools` has an invalid schema.
    #[error("{0}")]
    InvalidAllowedTools(String),
}

/// Detect duplicate tool names involving at least one generated name.
///
/// Client-supplied duplicates are the backend's concern. This only rejects
/// collisions where lossy encoding produced a generated name that clashes with
/// another tool.
///
/// # Errors
///
/// Returns [`McpToolMapError::NameCollision`] for the first generated name seen
/// twice.
pub(crate) fn detect_name_collisions<'a>(
    names: impl IntoIterator<Item = &'a str>,
    generated_names: &HashSet<String>,
) -> Result<(), McpToolMapError> {
    let mut seen: HashSet<&'a str> = HashSet::new();
    for name in names {
        if !seen.insert(name) && generated_names.contains(name) {
            return Err(McpToolMapError::NameCollision(name.to_owned()));
        }
    }
    Ok(())
}

/// Extract `server_label` from an MCP tool entry.
pub(crate) fn server_label(entry: &serde_json::Value) -> &str {
    entry
        .get("server_label")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown")
}

/// Extract `allowed_tools` from an MCP tool entry.
///
/// Handles `null` (unrestricted, matching the OpenAI schema), the
/// string-array form (`["a", "b"]`), and the `MCPToolFilter` object
/// form (`{"tool_names": ["a"]}`).
pub(crate) fn extract_allowed_tools(entry: &serde_json::Value) -> Result<AllowedTools, McpToolMapError> {
    let Some(value) = entry.get("allowed_tools") else {
        return Ok(AllowedTools::unrestricted());
    };
    if value.is_null() {
        return Ok(AllowedTools::unrestricted());
    }
    if let Some(arr) = value.as_array() {
        return validate_string_array(arr).map(|names| AllowedTools {
            names: Some(names),
            read_only: None,
        });
    }
    if let Some(obj) = value.as_object() {
        return extract_from_filter_object(obj);
    }
    Err(McpToolMapError::InvalidAllowedTools(format!(
        "allowed_tools must be an array of strings or an MCPToolFilter object, got {}",
        json_type_name(value),
    )))
}

/// Recognized fields in an `MCPToolFilter` object.
const ALLOWED_FILTER_FIELDS: &[&str] = &["tool_names", "read_only"];

/// Parse an `MCPToolFilter` object: `{tool_names?, read_only?}`.
fn extract_from_filter_object(
    obj: &serde_json::Map<String, serde_json::Value>,
) -> Result<AllowedTools, McpToolMapError> {
    for key in obj.keys() {
        if !ALLOWED_FILTER_FIELDS.contains(&key.as_str()) {
            return Err(McpToolMapError::InvalidAllowedTools(format!(
                "unknown field \"{key}\" in allowed_tools filter object",
            )));
        }
    }
    let names = match obj.get("tool_names") {
        Some(v) => {
            let arr = v.as_array().ok_or_else(|| {
                McpToolMapError::InvalidAllowedTools(format!(
                    "allowed_tools.tool_names must be an array, got {}",
                    json_type_name(v),
                ))
            })?;
            Some(validate_string_array(arr)?)
        },
        None => None,
    };
    let read_only = match obj.get("read_only") {
        Some(v) => Some(v.as_bool().ok_or_else(|| {
            McpToolMapError::InvalidAllowedTools(format!(
                "allowed_tools.read_only must be a boolean, got {}",
                json_type_name(v),
            ))
        })?),
        None => None,
    };
    Ok(AllowedTools { names, read_only })
}

/// Validate that every element in a JSON array is a string.
fn validate_string_array(arr: &[serde_json::Value]) -> Result<Vec<String>, McpToolMapError> {
    let mut result = Vec::with_capacity(arr.len());
    for (i, item) in arr.iter().enumerate() {
        match item.as_str() {
            Some(s) => result.push(s.to_owned()),
            None => {
                return Err(McpToolMapError::InvalidAllowedTools(format!(
                    "allowed_tools[{i}] must be a string, got {}",
                    json_type_name(item),
                )));
            },
        }
    }
    Ok(result)
}

/// Human-readable JSON type name for error messages.
fn json_type_name(v: &serde_json::Value) -> &'static str {
    match v {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "boolean",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

/// Outcome of parsing `allowed_tools`.
#[derive(Debug)]
pub(crate) struct AllowedTools {
    /// Optional tool-name allowlist.
    pub names: Option<Vec<String>>,
    /// `Some(true)` = only read-only tools,
    /// `Some(false)` = only writable tools,
    /// `None` = no read-only filter.
    pub read_only: Option<bool>,
}

impl AllowedTools {
    /// No filter — expose all tools.
    pub(crate) fn unrestricted() -> Self {
        Self {
            names: None,
            read_only: None,
        }
    }

    /// Return the name list as a slice, or `None` if
    /// unrestricted.
    pub(crate) fn as_names(&self) -> Option<&[String]> {
        self.names.as_deref()
    }
}

/// Filter tools by name list and/or read-only annotation.
pub(crate) fn apply_allowed_tools_filter(
    tools: Vec<serde_json::Value>,
    allowed: &AllowedTools,
) -> Vec<serde_json::Value> {
    if allowed.as_names().is_none() && allowed.read_only.is_none() {
        return tools;
    }
    tools
        .into_iter()
        .filter(|tool| is_allowed_tool(tool, allowed))
        .collect()
}

/// Match the same allowed-tools policy during preflight and final filtering.
pub(crate) fn is_allowed_tool(tool: &serde_json::Value, allowed: &AllowedTools) -> bool {
    if let Some(names) = allowed.as_names() {
        let matches_name = tool
            .get("name")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|name| names.iter().any(|allowed_name| allowed_name == name));
        if !matches_name {
            return false;
        }
    }
    allowed
        .read_only
        .is_none_or(|want_read_only| tool_read_only_hint(tool) == want_read_only)
}

/// Return whether an MCP tool has `annotations.readOnlyHint`
/// set to `true` (defaults to `false` when absent).
///
/// This is the **advisory** server hint consulted by the standard client
/// `allowed_tools.read_only` selector only. It is not a trusted classification;
/// operator-enforced read-only uses [`super::admission`] instead.
fn tool_read_only_hint(tool: &serde_json::Value) -> bool {
    tool.get("annotations")
        .and_then(|a| a.get("readOnlyHint"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
}

/// Apply operator metadata admission to the client-filtered tools of one entry.
///
/// In advisory mode this is a no-op: the tools pass through unchanged and no
/// verdict is recorded, so existing behavior is byte-for-byte preserved. In
/// protected mode each tool is evaluated against the operator policy and the
/// resolving entry's connector policy ([`tool_policy_for_entry`]); admitted
/// tools are returned (and reach the model + dispatch map), while denied tools are dropped
/// and recorded in `decisions` so a later forged call can be rejected explicitly
/// rather than mis-classified as a client function.
pub(crate) fn admit_tools(
    policy: McpAdmissionPolicy,
    tool_policy: ConnectorToolPolicy,
    label: &str,
    tools: Vec<serde_json::Value>,
    decisions: &mut McpAdmissionDecisions,
) -> Vec<serde_json::Value> {
    if !policy.is_protected() {
        return tools;
    }
    let mut admitted = Vec::with_capacity(tools.len());
    for tool in tools {
        // A tool without a name has no dispatchable identity; in protected mode
        // it is excluded rather than admitted under an unknown identity.
        let Some(tool_name) = tool.get("name").and_then(serde_json::Value::as_str).map(str::to_owned) else {
            continue;
        };
        let verdict = evaluate_tool(policy, tool_policy, &tool);
        decisions.record(label, &tool_name, verdict);
        if matches!(verdict, McpToolVerdict::Admitted(_)) {
            admitted.push(tool);
        }
    }
    admitted
}

/// Why protected mode excludes an MCP entry outright without dialing, or `None`
/// if the target may be contacted.
///
/// A target is excluded without contact when its policy is `Deny` — direct
/// `server_url` targets, denied connectors, and targetless (label-only) entries.
///
/// Centralizes the exclusion rule shared by `resolve_entry` (eager dial),
/// `collect_resolutions` (deferred/targetless entries with no resolution task),
/// and `collect_deferred_connectors` (deferred-discovery queueing) (#1724).
pub(crate) fn protected_exclusion_reason(
    tool_policy: ConnectorToolPolicy,
    policy: McpAdmissionPolicy,
) -> Option<McpDenyReason> {
    if !policy.is_protected() {
        return None;
    }
    // Exhaustive over `ConnectorToolPolicy` on purpose: a new variant must force a
    // decision here rather than silently become contactable.
    match tool_policy {
        ConnectorToolPolicy::Deny => Some(McpDenyReason::UntrustedSource),
        ConnectorToolPolicy::AllowAll | ConnectorToolPolicy::ReadOnly => None,
    }
}

/// [`protected_exclusion_reason`] resolved for one MCP entry via its connector
/// policy ([`tool_policy_for_entry`]).
pub(crate) fn protected_exclusion(
    entry: &serde_json::Value,
    policy: McpAdmissionPolicy,
    connector_tool_policies: &HashMap<String, ConnectorToolPolicy>,
) -> Option<McpDenyReason> {
    protected_exclusion_reason(tool_policy_for_entry(entry, connector_tool_policies), policy)
}

/// Resolve the operator tool policy for one MCP entry.
///
/// A direct, client-selected `server_url` entry and an unknown connector are
/// denied by default. A configured connector's policy is looked up by
/// `connector_id`; a missing entry fails closed.
pub(crate) fn tool_policy_for_entry(
    entry: &serde_json::Value,
    connector_tool_policies: &HashMap<String, ConnectorToolPolicy>,
) -> ConnectorToolPolicy {
    entry
        .get("connector_id")
        .and_then(serde_json::Value::as_str)
        .and_then(|id| connector_tool_policies.get(id))
        .copied()
        .unwrap_or_default()
}

/// Evaluate operator admission for one deferred connector's client-filtered
/// tools, returning the admitted tools and the `(tool_name, verdict)` pairs to
/// record once concurrent discovery commits (the decisions live in `state`,
/// which is not borrowed across the concurrent `tools/list` calls).
///
/// Advisory mode returns the tools unchanged with no verdicts.
pub(crate) fn admit_deferred_tools(
    policy: McpAdmissionPolicy,
    tool_policy: ConnectorToolPolicy,
    tools: Vec<serde_json::Value>,
) -> (Vec<serde_json::Value>, Vec<(String, McpToolVerdict)>) {
    if !policy.is_protected() {
        return (tools, Vec::new());
    }
    let mut admitted = Vec::with_capacity(tools.len());
    let mut verdicts = Vec::new();
    for tool in tools {
        let Some(tool_name) = tool.get("name").and_then(serde_json::Value::as_str).map(str::to_owned) else {
            continue;
        };
        let verdict = evaluate_tool(policy, tool_policy, &tool);
        verdicts.push((tool_name, verdict));
        if matches!(verdict, McpToolVerdict::Admitted(_)) {
            admitted.push(tool);
        }
    }
    (admitted, verdicts)
}

/// Insert resolved tools into the tool map keyed by
/// `(server_label, tool_name)`, consuming the definitions.
#[expect(
    clippy::too_many_lines,
    reason = "explicit map construction moves tool definitions while preserving dispatch metadata"
)]
pub(crate) fn insert_tools(
    tools: Vec<serde_json::Value>,
    entry: &serde_json::Value,
    tool_map: &mut HashMap<(String, String), serde_json::Value>,
) {
    let label = server_label(entry);
    let server_url = entry
        .get("server_url")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown");
    let headers = entry.get("headers");
    let authorization = entry.get("authorization");
    let require_approval = entry.get("require_approval");
    let connector_id = entry.get("connector_id");

    for tool in tools {
        let tool_name = tool.get("name").and_then(serde_json::Value::as_str).map(str::to_owned);
        let Some(tool_name) = tool_name else {
            continue;
        };

        let key = (label.to_owned(), tool_name);
        let mut dispatch = serde_json::Map::new();
        dispatch.insert("server_label".to_owned(), serde_json::Value::String(label.to_owned()));
        dispatch.insert(
            "server_url".to_owned(),
            serde_json::Value::String(server_url.to_owned()),
        );
        dispatch.insert(
            "headers".to_owned(),
            headers.cloned().unwrap_or(serde_json::Value::Null),
        );
        dispatch.insert(
            "authorization".to_owned(),
            authorization.cloned().unwrap_or(serde_json::Value::Null),
        );
        dispatch.insert(
            "require_approval".to_owned(),
            require_approval.cloned().unwrap_or(serde_json::Value::Null),
        );
        dispatch.insert(
            "connector_id".to_owned(),
            connector_id.cloned().unwrap_or(serde_json::Value::Null),
        );
        dispatch.insert("tool_definition".to_owned(), tool);
        tool_map.insert(key, serde_json::Value::Object(dispatch));
    }
}

/// Resolve connector IDs to server URLs for MCP tool entries.
///
/// Mutates entries in place, replacing `connector_id` with the
/// corresponding `server_url` from the configured connectors map.
pub(crate) fn resolve_connector_ids(
    connectors: &HashMap<String, url::Url>,
    entries: &mut [serde_json::Value],
) -> Result<(), McpToolMapError> {
    for entry in entries.iter_mut() {
        let Some(connector_id_value) = entry.get("connector_id") else {
            continue;
        };

        let connector_id = match connector_id_value.as_str() {
            Some(s) if !s.is_empty() && s.len() <= MAX_CONNECTOR_ID_LEN => s.to_owned(),
            _ => return Err(McpToolMapError::InvalidConnectorId),
        };

        validate_connector_entry(entry, &connector_id)?;

        let resolved_url = connectors
            .get(&connector_id)
            .ok_or(McpToolMapError::UnknownConnector(connector_id))?;

        if let Some(obj) = entry.as_object_mut() {
            obj.insert(
                "server_url".to_owned(),
                serde_json::Value::String(resolved_url.to_string()),
            );
        }
    }
    Ok(())
}

/// Reject direct client-selected MCP destinations before any identity staging or
/// outbound callout is acquired.
///
/// The error intentionally contains no request data: `server_url`,
/// `server_label`, credentials, and assertions must not leak through the
/// client-visible rejection.
pub(crate) fn reject_direct_server_urls(entries: &[serde_json::Value]) -> Result<(), McpToolMapError> {
    if entries.iter().any(|entry| entry.get("server_url").is_some()) {
        return Err(McpToolMapError::DirectServerUrl);
    }
    Ok(())
}

/// Validate that a connector entry has required fields and no conflicts.
fn validate_connector_entry(entry: &serde_json::Value, connector_id: &str) -> Result<(), McpToolMapError> {
    let label = entry
        .get("server_label")
        .and_then(serde_json::Value::as_str)
        .filter(|s| !s.is_empty());
    if label.is_none() {
        return Err(McpToolMapError::MissingConnectorLabel(connector_id.to_owned()));
    }

    if entry.get("server_url").is_some() {
        return Err(McpToolMapError::MutuallyExclusiveTarget(connector_id.to_owned()));
    }

    Ok(())
}

/// Select only configured headers from the effective body-phase request.
pub(crate) fn select_forward_headers(names: &[http::HeaderName], source: &http::HeaderMap) -> http::HeaderMap {
    let mut selected = http::HeaderMap::with_capacity(names.len());
    for name in names {
        if let Some(value) = source.get(name) {
            selected.insert(name.clone(), value.clone());
        }
    }
    selected
}
