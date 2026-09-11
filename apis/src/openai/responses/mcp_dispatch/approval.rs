// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Approval policy parsing and evaluation for MCP tool calls.
//!
//! Two concerns live here:
//!
//! 1. **Policy** ([`ApprovalPolicy`], [`parse_approval_policy`], [`requires_approval`]): whether a resolved MCP tool
//!    call must pause for human approval before execution.
//! 2. **Response round trip** ([`parse_approval_response`], [`resolve_approval`], [`ResolvedApproval`],
//!    [`ApprovalError`], [`build_approved_tool_call`], [`build_denial_message`]): parsing a client-supplied
//!    `mcp_approval_response`, correlating it to the server-owned [`PendingApprovalRecord`] the proxy wrote when it
//!    emitted the `mcp_approval_request`, binding it to that *complete* pending call (server label, tool name, and
//!    arguments) resolved against the current tool map, and shaping the decision into either a tool call to execute or
//!    a denial fed back to the model.

use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
};

use sha2::{Digest as _, Sha256};

use crate::{openai::responses::mcp_tool_resolve::encode_function_name, store::PendingApprovalRecord};

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

// -----------------------------------------------------------------------------
// Approval Response Round Trip
// -----------------------------------------------------------------------------

/// Item type of a client-supplied approval decision.
const APPROVAL_RESPONSE_TYPE: &str = "mcp_approval_response";

/// A parsed client-supplied `mcp_approval_response`.
///
/// Carries only what the client legitimately supplies: the correlation id, the
/// accept/deny verdict, and an optional reason. Everything about the pending
/// call itself (target, tool name, arguments) comes from the server-owned
/// [`PendingApprovalRecord`], never from this input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ApprovalResponseInput {
    /// Correlation id echoed by the client; must match a pending record.
    pub approval_id: String,
    /// Whether the user approved (`true`) or denied (`false`) the call.
    pub approve: bool,
    /// Optional human-supplied denial reason, preserved for the model.
    pub reason: Option<String>,
}

/// A validated approval decision bound to a concrete pending MCP call.
///
/// Produced by [`resolve_approval`]. The target fields are derived from the
/// server-owned [`PendingApprovalRecord`] (the source of truth for the pending
/// call), never from the client's `mcp_approval_response`, which only carries
/// the correlation id and the accept/deny verdict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResolvedApproval {
    /// Correlation id; equals the original function-call `call_id`.
    pub approval_id: String,
    /// Whether the user approved (`true`) or denied (`false`) the call.
    pub approve: bool,
    /// Optional human-supplied denial reason, preserved for the model.
    pub reason: Option<String>,
    /// Server label of the resolved target.
    pub server_label: String,
    /// Original (un-encoded) tool name of the resolved target.
    pub tool_name: String,
    /// Encoded function name used to route the call through dispatch.
    pub encoded_name: String,
    /// Tool arguments as a JSON string, taken from the pending record.
    pub arguments: String,
}

/// Why an `mcp_approval_response` could not be honored.
///
/// Every variant is a client-triggered condition that must fail closed with a
/// `400 invalid_request_error`: the proxy refuses to execute or resume a call
/// it cannot fully and unambiguously bind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ApprovalError {
    /// The response item is missing required fields or has wrong types.
    Malformed(String),
    /// No server-owned pending approval matches `approval_request_id`.
    UnknownApprovalId(String),
    /// The pending target is not present in the current tool map.
    TargetUnresolvable(String),
    /// The encoded target name matches more than one tool-map entry.
    AmbiguousTarget(String),
    /// The current target's identity (URL, headers, authorization, or
    /// connector) differs from the target the approval was granted for, or the
    /// stored approval carries no fingerprint to verify against.
    TargetIdentityMismatch(String),
}

impl ApprovalError {
    /// Human-readable message for the error envelope.
    pub(crate) fn message(&self) -> &str {
        match self {
            Self::Malformed(m)
            | Self::UnknownApprovalId(m)
            | Self::TargetUnresolvable(m)
            | Self::AmbiguousTarget(m)
            | Self::TargetIdentityMismatch(m) => m,
        }
    }
}

/// Return whether an item is a client-supplied `mcp_approval_response`.
pub(crate) fn is_approval_response(item: &serde_json::Value) -> bool {
    item.get("type").and_then(serde_json::Value::as_str) == Some(APPROVAL_RESPONSE_TYPE)
}

/// Borrow all `mcp_approval_response` items from a message list.
///
/// Returns references into `messages`; the caller only reads the items to
/// correlate them, so there is no need to clone the request-derived JSON.
pub(crate) fn extract_approval_responses(messages: &[serde_json::Value]) -> Vec<&serde_json::Value> {
    messages.iter().filter(|m| is_approval_response(m)).collect()
}

/// Parse a client-supplied `mcp_approval_response` into its trusted fields.
///
/// Only the correlation id, verdict, and optional reason are read; the pending
/// call itself is looked up server-side. Missing or wrongly-typed fields fail
/// closed as [`ApprovalError::Malformed`].
pub(crate) fn parse_approval_response(response: &serde_json::Value) -> Result<ApprovalResponseInput, ApprovalError> {
    let approval_id = response
        .get("approval_request_id")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| ApprovalError::Malformed("mcp_approval_response missing string approval_request_id".to_owned()))?
        .to_owned();

    let approve = response
        .get("approve")
        .and_then(serde_json::Value::as_bool)
        .ok_or_else(|| ApprovalError::Malformed("mcp_approval_response missing boolean approve".to_owned()))?;

    let reason = response
        .get("reason")
        .and_then(serde_json::Value::as_str)
        .map(ToOwned::to_owned);

    Ok(ApprovalResponseInput {
        approval_id,
        approve,
        reason,
    })
}

/// Target-bind a parsed approval response to its server-owned pending record.
///
/// The decision is bound to the **complete** pending call the proxy recorded
/// when it emitted the `mcp_approval_request`: that record's `(server_label,
/// tool_name)` must resolve to exactly one entry in the current `tool_map`, and
/// that entry's target identity (URL, headers, authorization, connector) must
/// match the fingerprint captured when approval was requested. An unresolved
/// target, ambiguous encoding, missing fingerprint, or target-identity change
/// fails closed so a stale, forged, or redirected response can never execute an
/// unintended tool against an unapproved destination.
pub(crate) fn resolve_approval(
    input: &ApprovalResponseInput,
    pending: &PendingApprovalRecord,
    tool_map: &HashMap<(String, String), serde_json::Value>,
) -> Result<ResolvedApproval, ApprovalError> {
    let approval_id = pending.approval_id.as_str();
    let server_label = pending.server_label.as_str();
    let tool_name = pending.tool_name.as_str();

    let encoded_name = encode_function_name(server_label, tool_name);
    let entry = bind_target(tool_map, &encoded_name, server_label, tool_name, approval_id)?;

    // Bind the decision to the concrete target the proxy resolved when it asked
    // for approval. A resume turn that keeps the approved (server_label,
    // tool_name) but redirects to a different URL, swaps headers or the
    // authorization credential, or points at another connector produces a
    // different fingerprint and is rejected before any call is made. An empty
    // stored fingerprint (e.g. an ambiguous target or case-colliding header names
    // at approval time) can never match, so it also fails closed.
    if pending.target_fingerprint.is_empty() {
        return Err(ApprovalError::TargetIdentityMismatch(format!(
            "stored approval '{approval_id}' is missing its target fingerprint"
        )));
    }
    if target_fingerprint(entry) != pending.target_fingerprint {
        return Err(ApprovalError::TargetIdentityMismatch(format!(
            "approval '{approval_id}' was granted for a different target than the current request resolves"
        )));
    }

    Ok(ResolvedApproval {
        approval_id: pending.approval_id.clone(),
        approve: input.approve,
        reason: input.reason.clone(),
        server_label: pending.server_label.clone(),
        tool_name: pending.tool_name.clone(),
        encoded_name,
        arguments: pending.arguments.clone(),
    })
}

/// Verify the stored target resolves to exactly one current tool-map entry
/// whose identity matches the stored `(server_label, tool_name)`, returning that
/// entry so the caller can bind against its resolved target identity.
fn bind_target<'a>(
    tool_map: &'a HashMap<(String, String), serde_json::Value>,
    encoded_name: &str,
    server_label: &str,
    tool_name: &str,
    approval_id: &str,
) -> Result<&'a serde_json::Value, ApprovalError> {
    let mut matches = tool_map
        .iter()
        .filter(|((label, name), _)| encode_function_name(label, name) == encoded_name);
    let Some((matched_key, matched_entry)) = matches.next() else {
        return Err(ApprovalError::TargetUnresolvable(format!(
            "approved tool '{encoded_name}' for approval '{approval_id}' is not in the current tool map"
        )));
    };
    if matches.next().is_some() {
        return Err(ApprovalError::AmbiguousTarget(format!(
            "approved tool '{encoded_name}' for approval '{approval_id}' matches multiple servers"
        )));
    }
    if matched_key.0 != server_label || matched_key.1 != tool_name {
        return Err(ApprovalError::TargetUnresolvable(format!(
            "approved target ({server_label}, {tool_name}) for approval '{approval_id}' \
             does not match the resolved tool-map entry ({}, {})",
            matched_key.0, matched_key.1
        )));
    }
    Ok(matched_entry)
}

/// Compute a stable, credential-safe fingerprint of a resolved MCP target.
///
/// Binds an approval to the concrete destination resolved at approval time: the
/// server URL, request headers, authorization credential, and connector id. On
/// resume the proxy recomputes this from the *current* tool-map entry and
/// rejects the call if it differs, so a client cannot keep the approved
/// `(server_label, tool_name)` while redirecting execution elsewhere or swapping
/// credentials.
///
/// A SHA-256 digest — not the raw fields — is emitted so the value is safe to
/// embed in the client-visible, at-rest `mcp_approval_request`: it never
/// discloses the authorization token, and it is deterministic across processes
/// (unlike the std hasher's per-process seed) so the request-time and
/// resume-time computations agree. Fields are length-framed and headers are
/// key-sorted so the digest is independent of JSON key ordering.
///
/// Returns the empty string as a fail-closed sentinel when the headers contain
/// case-insensitive duplicate names: the transport lowercases names and lets the
/// last one in JSON order win, so such a target is ambiguous and must never
/// produce a matchable fingerprint.
pub(crate) fn target_fingerprint(entry: &serde_json::Value) -> String {
    let mut hasher = Sha256::new();
    for field in ["server_url", "authorization", "connector_id"] {
        hash_segment(&mut hasher, field.as_bytes());
        hash_segment(&mut hasher, &scalar_bytes(entry.get(field)));
    }
    hash_segment(&mut hasher, b"headers");
    if let Some(headers) = entry.get("headers").and_then(serde_json::Value::as_object) {
        // The transport lowercases header names, so two names differing only in
        // case collapse to one and the value actually sent depends on JSON order.
        // A case-sensitive key sort would hash such an ambiguous target to a
        // stable digest that does not reflect what is transmitted, letting a
        // reordered resume reuse the approval with a different value. Fail closed
        // with the empty sentinel instead so it can never match on resume.
        let mut seen = HashSet::with_capacity(headers.len());
        for key in headers.keys() {
            if !seen.insert(key.to_ascii_lowercase()) {
                return String::new();
            }
        }
        let mut keys: Vec<&str> = headers.keys().map(String::as_str).collect();
        keys.sort_unstable();
        for key in keys {
            hash_segment(&mut hasher, key.as_bytes());
            hash_segment(&mut hasher, &scalar_bytes(headers.get(key)));
        }
    }
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        hex.push(char::from_digit(u32::from(byte >> 4), 16).unwrap_or('0'));
        hex.push(char::from_digit(u32::from(byte & 0x0F), 16).unwrap_or('0'));
    }
    hex
}

/// Length-framed update so adjacent fields cannot be confused by content that
/// happens to look like a delimiter.
fn hash_segment(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

/// Canonical bytes of an optional JSON scalar for hashing: strings hash as their
/// UTF-8 contents, absent/null as empty, anything else as its compact JSON.
fn scalar_bytes(value: Option<&serde_json::Value>) -> Cow<'_, [u8]> {
    match value {
        None | Some(serde_json::Value::Null) => Cow::Borrowed(&[]),
        Some(serde_json::Value::String(s)) => Cow::Borrowed(s.as_bytes()),
        Some(other) => Cow::Owned(other.to_string().into_bytes()),
    }
}

/// Build a function-call-shaped tool call for an approved decision.
///
/// The shape matches what `openai_stream_events` produces for a native
/// function call so the existing dispatch execution path runs it unchanged.
/// `approval_request_id` is threaded so the resulting `mcp_call` output item
/// references the approval that authorized it.
pub(crate) fn build_approved_tool_call(resolved: &ResolvedApproval) -> serde_json::Value {
    serde_json::json!({
        "type": "function_call",
        "name": resolved.encoded_name,
        "call_id": resolved.approval_id,
        "arguments": resolved.arguments,
        "approval_request_id": resolved.approval_id,
    })
}

/// Build a schema-valid `function_call_output` denial message.
///
/// A denial resumes inference with a truthful tool result rather than a
/// fabricated `mcp_call` with a made-up `denied` status. The original
/// `call_id` correlation is preserved and any supplied reason is surfaced to
/// the model.
pub(crate) fn build_denial_message(approval_id: &str, reason: Option<&str>) -> serde_json::Value {
    let output = match reason.map(str::trim).filter(|r| !r.is_empty()) {
        Some(reason) => format!("Tool call was denied by the user. Reason: {reason}"),
        None => "Tool call was denied by the user.".to_owned(),
    };
    serde_json::json!({
        "type": "function_call_output",
        "call_id": approval_id,
        "output": output,
    })
}
