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
//! `openai_file_search_dispatch`) can agree on the disposition of a call
//! without any dispatcher importing another dispatcher's internals.
//!
//! The approval-policy helpers ([`parse_approval_policy`], [`requires_approval`])
//! live here rather than in `openai_mcp_dispatch` so that this module stays
//! neutral: `classify_mcp` composes them, and `openai_mcp_dispatch` imports them
//! from here for its private [`PendingApproval`] construction.
//!
//! [`PendingApproval`]: super::mcp_dispatch

#[cfg(any(test, feature = "openai-agentic-loop"))]
use super::mcp_tool_resolve::{McpToolIndex, McpToolMatch};

/// Disposition of a single model-emitted tool call relative to the resolved MCP
/// tool map.
///
/// This is intentionally a unit-variant enum: it carries the *decision* only,
/// never the per-call data (`call_id`/`server_label`/`tool_name`/`arguments`)
/// that `openai_mcp_dispatch` needs to build a client-visible
/// `mcp_approval_request`. That construction stays private to the dispatcher.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(any(test, feature = "openai-agentic-loop"))]
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
/// - **exactly one match** → the operator floor and the per-tool `require_approval` policy together decide
///   [`McpDisposition::ApprovalRequired`] vs [`McpDisposition::Automatic`], fail-closed to approval when the policy is
///   absent or unrecognized.
///
/// `operator_floor` is combined with the client's per-tool policy **monotonically**:
///
/// ```text
/// approval_required = operator_floor_requires_approval || client_requires_approval
/// ```
///
/// The floor can only *raise* the requirement. An operator floor of
/// [`ApprovalFloor::Always`] forces approval for every resolved MCP call, so a
/// client `"require_approval": "never"` (or a server-supplied `readOnlyHint`,
/// which never feeds approval) can never suppress it. When the floor is
/// [`ApprovalFloor::None`] the classification is exactly the client-driven
/// behavior.
#[cfg(any(test, feature = "openai-agentic-loop"))]
pub(crate) fn classify_mcp(
    tool_call: &serde_json::Value,
    tool_index: &McpToolIndex<'_>,
    operator_floor: ApprovalFloor,
) -> McpDisposition {
    let Some(encoded_name) = tool_call.get("name").and_then(serde_json::Value::as_str) else {
        return McpDisposition::NotMcp;
    };
    match tool_index.get(encoded_name) {
        None => McpDisposition::NotMcp,
        Some(McpToolMatch::Ambiguous { .. }) => McpDisposition::ApprovalRequired,
        Some(McpToolMatch::Unique { key, entry }) => {
            let read_only_hint = tool_read_only_hint(entry);
            if operator_floor.forces_approval()
                || requires_approval(&parse_approval_policy(entry), &key.1, read_only_hint)
            {
                McpDisposition::ApprovalRequired
            } else {
                McpDisposition::Automatic
            }
        },
    }
}

// -----------------------------------------------------------------------------
// Operator approval floor
// -----------------------------------------------------------------------------

/// Operator-configured minimum approval requirement for the Responses MCP path.
///
/// Combined monotonically with each tool's client `require_approval` policy in
/// [`classify_mcp`]:
///
/// ```text
/// approval_required = operator_floor_requires_approval || client_requires_approval
/// ```
///
/// The floor can only *raise* the requirement; it never weakens a client that
/// already requires approval, and a client `"never"` or a server-supplied
/// `readOnlyHint` cannot suppress an operator-required approval.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum ApprovalFloor {
    /// No operator floor; the client's per-tool policy decides (existing behavior).
    #[default]
    None,
    /// Require approval for every resolved MCP call, regardless of client settings.
    Always,
}

#[cfg(any(test, feature = "openai-agentic-loop"))]
impl ApprovalFloor {
    /// Whether the operator floor forces approval for every classified MCP call.
    pub(crate) const fn forces_approval(self) -> bool {
        matches!(self, Self::Always)
    }
}

// -----------------------------------------------------------------------------
// Approval policy
// -----------------------------------------------------------------------------

/// Approval policy for MCP tool execution.
///
/// Mirrors the OpenAI `require_approval` schema: the string forms `"always"` /
/// `"never"`, or an `MCPToolApprovalFilter` object carrying an `always` and a
/// `never` [`ApprovalSelector`]. Every shape outside that schema — absent,
/// `null`, an unknown string, an unknown object field, or an invalid member
/// type — collapses to [`ApprovalPolicy::Always`], the fail-closed default that
/// forces approval before a tool can execute.
#[derive(Debug, Clone, PartialEq)]
#[cfg(any(test, feature = "openai-agentic-loop"))]
pub(crate) enum ApprovalPolicy {
    /// Always require approval.
    Always,
    /// Never require approval.
    Never,
    /// Per-tool filter. The `always` selector forces approval and takes
    /// precedence over `never`; a tool matched by neither selector defaults to
    /// requiring approval.
    Filter {
        /// Selector for tools that always require approval.
        always: ApprovalSelector,
        /// Selector for tools that never require approval.
        never: ApprovalSelector,
    },
}

/// One side (`always` or `never`) of an `MCPToolApprovalFilter`.
///
/// Represents the full `MCPToolFilter` schema: an optional `tool_names`
/// allowlist and an optional `read_only` annotation selector. A tool matches
/// the filter when it satisfies *every* selector that is present (logical AND).
/// A filter with no selectors matches no tool, so a degenerate `{}` member
/// never changes a call's disposition.
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg(any(test, feature = "openai-agentic-loop"))]
pub(crate) struct ApprovalSelector {
    /// Optional `tool_names` allowlist; `None` imposes no name constraint.
    pub(crate) names: Option<Vec<String>>,
    /// Optional `read_only` selector matched against the tool's
    /// `annotations.readOnlyHint`; `None` imposes no annotation constraint.
    pub(crate) read_only: Option<bool>,
}

#[cfg(any(test, feature = "openai-agentic-loop"))]
impl ApprovalSelector {
    /// Whether `tool_name`/`read_only_hint` satisfy every selector present on
    /// this filter. A filter carrying no selectors matches nothing.
    fn matches(&self, tool_name: &str, read_only_hint: bool) -> bool {
        let mut has_selector = false;
        if let Some(names) = &self.names {
            has_selector = true;
            if !names.iter().any(|n| n == tool_name) {
                return false;
            }
        }
        if let Some(read_only) = self.read_only {
            has_selector = true;
            if read_only != read_only_hint {
                return false;
            }
        }
        has_selector
    }
}

/// Parse `require_approval` from a resolved MCP tool entry.
///
/// Recognizes the OpenAI schema strictly and fails closed to
/// [`ApprovalPolicy::Always`] for every shape outside it, so a malformed or
/// unknown policy can never downgrade a call to automatic execution:
///
/// - absent or `null` → `Always` (the schema default);
/// - `"always"` → `Always`, `"never"` → `Never`, any other string → `Always`;
/// - `{"always"?: MCPToolFilter, "never"?: MCPToolFilter}` → `Filter`;
/// - an object with an unknown field, a non-object `always`/`never`, a non-array `tool_names`, a non-string tool name,
///   or a non-boolean `read_only` → `Always` (malformed, fail closed before dispatch).
#[cfg(any(test, feature = "openai-agentic-loop"))]
pub(crate) fn parse_approval_policy(tool_def: &serde_json::Value) -> ApprovalPolicy {
    let Some(value) = tool_def.get("require_approval") else {
        return ApprovalPolicy::Always;
    };
    if value.is_null() {
        return ApprovalPolicy::Always;
    }
    if let Some(s) = value.as_str() {
        return match s {
            "never" => ApprovalPolicy::Never,
            _ => ApprovalPolicy::Always,
        };
    }
    if let Some(obj) = value.as_object() {
        return parse_filter_policy(obj).unwrap_or(ApprovalPolicy::Always);
    }
    ApprovalPolicy::Always
}

/// Recognized keys in an `MCPToolApprovalFilter` object.
#[cfg(any(test, feature = "openai-agentic-loop"))]
const APPROVAL_FILTER_KEYS: &[&str] = &["always", "never"];

/// Parse the `{always?, never?}` approval-filter object.
///
/// Returns `None` for any shape outside the closed schema (an unknown key or a
/// member that is not a valid `MCPToolFilter`) so the caller fails closed.
#[cfg(any(test, feature = "openai-agentic-loop"))]
fn parse_filter_policy(obj: &serde_json::Map<String, serde_json::Value>) -> Option<ApprovalPolicy> {
    if obj.keys().any(|key| !APPROVAL_FILTER_KEYS.contains(&key.as_str())) {
        return None;
    }
    let always = parse_filter_member(obj.get("always"))?;
    let never = parse_filter_member(obj.get("never"))?;
    Some(ApprovalPolicy::Filter { always, never })
}

/// Recognized keys in an `MCPToolFilter` object.
#[cfg(any(test, feature = "openai-agentic-loop"))]
const TOOL_FILTER_KEYS: &[&str] = &["tool_names", "read_only"];

/// Parse one `MCPToolFilter` member of an approval policy.
///
/// An absent member is the empty [`ApprovalSelector`]. Returns `None` for any
/// shape outside the closed schema so the caller fails closed: a non-object
/// member, an unknown field, a non-array `tool_names`, a non-string tool name,
/// or a non-boolean `read_only`. An explicit `null` — as the member itself or as
/// a `tool_names`/`read_only` value — is **not** nullable in the schema, so it is
/// treated as malformed rather than a dropped selector (otherwise
/// `{"never":{"tool_names":null,"read_only":false}}` would silently exempt every
/// writable tool from approval).
#[cfg(any(test, feature = "openai-agentic-loop"))]
fn parse_filter_member(value: Option<&serde_json::Value>) -> Option<ApprovalSelector> {
    let Some(value) = value else {
        return Some(ApprovalSelector::default());
    };
    let obj = value.as_object()?;
    if obj.keys().any(|key| !TOOL_FILTER_KEYS.contains(&key.as_str())) {
        return None;
    }
    let names = match obj.get("tool_names") {
        None => None,
        Some(v) => Some(parse_tool_names(v.as_array()?)?),
    };
    let read_only = match obj.get("read_only") {
        None => None,
        Some(v) => Some(v.as_bool()?),
    };
    Some(ApprovalSelector { names, read_only })
}

/// Collect a `tool_names` array, returning `None` if any element is not a string.
#[cfg(any(test, feature = "openai-agentic-loop"))]
fn parse_tool_names(arr: &[serde_json::Value]) -> Option<Vec<String>> {
    arr.iter().map(|v| v.as_str().map(ToOwned::to_owned)).collect()
}

/// Check whether a tool call requires approval under the given policy.
///
/// `read_only_hint` is the tool's `annotations.readOnlyHint` (defaulting to
/// `false`/writable when absent). For `Filter`, an `always` match forces
/// approval and takes precedence over a `never` match; a tool matched by
/// neither selector defaults to requiring approval (fail closed).
#[cfg(any(test, feature = "openai-agentic-loop"))]
pub(crate) fn requires_approval(policy: &ApprovalPolicy, tool_name: &str, read_only_hint: bool) -> bool {
    match policy {
        ApprovalPolicy::Always => true,
        ApprovalPolicy::Never => false,
        ApprovalPolicy::Filter { always, never } => {
            if always.matches(tool_name, read_only_hint) {
                return true;
            }
            if never.matches(tool_name, read_only_hint) {
                return false;
            }
            true
        },
    }
}

/// Read `tool_definition.annotations.readOnlyHint` from a resolved MCP tool
/// entry.
///
/// The resolved entry wraps the tool under `tool_definition`; a missing hint
/// defaults to `false` (writable), matching the MCP annotation default so a
/// tool without annotations is treated as data-modifying.
#[cfg(any(test, feature = "openai-agentic-loop"))]
fn tool_read_only_hint(entry: &serde_json::Value) -> bool {
    entry
        .get("tool_definition")
        .and_then(|def| def.get("annotations"))
        .and_then(|annotations| annotations.get("readOnlyHint"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::panic, clippy::needless_pass_by_value, reason = "tests")]
mod tests {
    use std::collections::HashMap;

    use serde_json::json;

    use super::{
        ApprovalFloor, ApprovalPolicy, ApprovalSelector, McpDisposition, classify_mcp, parse_approval_policy,
        requires_approval,
    };
    use crate::openai::responses::mcp_tool_resolve::McpToolIndex;

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

    /// Like [`tool_map_with_policy`] but stamps the tool's
    /// `annotations.readOnlyHint`, so `read_only` selectors can participate in
    /// classification.
    fn tool_map_with_read_only(
        label: &str,
        tool: &str,
        read_only_hint: bool,
        require_approval: serde_json::Value,
    ) -> HashMap<(String, String), serde_json::Value> {
        let mut map = HashMap::new();
        map.insert(
            (label.to_owned(), tool.to_owned()),
            json!({
                "server_label": label,
                "server_url": "https://mcp.example",
                "require_approval": require_approval,
                "tool_definition": {"name": tool, "annotations": {"readOnlyHint": read_only_hint}},
            }),
        );
        map
    }

    #[test]
    fn unknown_name_is_not_mcp() {
        let map = tool_map_with_policy("srv", "search", json!("never"));
        let index = McpToolIndex::new(&map);
        let call = json!({"type": "function_call", "name": "some_client_fn", "call_id": "c1"});
        assert_eq!(classify_mcp(&call, &index, ApprovalFloor::None), McpDisposition::NotMcp);
    }

    #[test]
    fn missing_name_is_not_mcp() {
        let map = tool_map_with_policy("srv", "search", json!("never"));
        let index = McpToolIndex::new(&map);
        let call = json!({"type": "function_call", "call_id": "c1"});
        assert_eq!(classify_mcp(&call, &index, ApprovalFloor::None), McpDisposition::NotMcp);
    }

    #[test]
    fn never_policy_is_automatic() {
        let map = tool_map_with_policy("srv", "search", json!("never"));
        let index = McpToolIndex::new(&map);
        let call = json!({"type": "function_call", "name": "srv__search", "call_id": "c1"});
        assert_eq!(
            classify_mcp(&call, &index, ApprovalFloor::None),
            McpDisposition::Automatic
        );
    }

    #[test]
    fn always_policy_requires_approval() {
        let map = tool_map_with_policy("srv", "search", json!("always"));
        let index = McpToolIndex::new(&map);
        let call = json!({"type": "function_call", "name": "srv__search", "call_id": "c1"});
        assert_eq!(
            classify_mcp(&call, &index, ApprovalFloor::None),
            McpDisposition::ApprovalRequired
        );
    }

    #[test]
    fn absent_policy_fails_closed_to_approval() {
        let map = tool_map_with_policy("srv", "search", serde_json::Value::Null);
        let index = McpToolIndex::new(&map);
        let call = json!({"type": "function_call", "name": "srv__search", "call_id": "c1"});
        assert_eq!(
            classify_mcp(&call, &index, ApprovalFloor::None),
            McpDisposition::ApprovalRequired
        );
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
            classify_mcp(&call, &index, ApprovalFloor::None),
            McpDisposition::ApprovalRequired,
            "an ambiguous encoded name must fail closed to approval, not automatic"
        );
    }

    #[test]
    fn operator_floor_forces_approval_over_client_never() {
        // The acceptance core: a client `require_approval: "never"` resolves to
        // Automatic without a floor, but the operator floor overrides it.
        let map = tool_map_with_policy("srv", "search", json!("never"));
        let index = McpToolIndex::new(&map);
        let call = json!({"type": "function_call", "name": "srv__search", "call_id": "c1"});
        assert_eq!(
            classify_mcp(&call, &index, ApprovalFloor::None),
            McpDisposition::Automatic,
            "without the floor, a client 'never' executes automatically"
        );
        assert_eq!(
            classify_mcp(&call, &index, ApprovalFloor::Always),
            McpDisposition::ApprovalRequired,
            "the operator floor must force approval even when the client said 'never'"
        );
    }

    #[test]
    fn operator_floor_keeps_approval_for_always_client() {
        let map = tool_map_with_policy("srv", "search", json!("always"));
        let index = McpToolIndex::new(&map);
        let call = json!({"type": "function_call", "name": "srv__search", "call_id": "c1"});
        assert_eq!(
            classify_mcp(&call, &index, ApprovalFloor::Always),
            McpDisposition::ApprovalRequired,
            "the floor must not weaken a client policy that already requires approval"
        );
    }

    #[test]
    fn operator_floor_cannot_be_weakened_by_read_only_hint() {
        // `readOnlyHint` feeds tool filtering, never approval, so it cannot
        // suppress the operator floor: a read-only tool with client 'never' still
        // requires approval under the floor.
        let mut map = tool_map_with_policy("srv", "search", json!("never"));
        map.get_mut(&("srv".to_owned(), "search".to_owned())).unwrap()["tool_definition"] =
            json!({"name": "search", "annotations": {"readOnlyHint": true}});
        let index = McpToolIndex::new(&map);
        let call = json!({"type": "function_call", "name": "srv__search", "call_id": "c1"});
        assert_eq!(
            classify_mcp(&call, &index, ApprovalFloor::Always),
            McpDisposition::ApprovalRequired,
            "a server-supplied readOnlyHint must not weaken the operator floor"
        );
    }

    #[test]
    fn operator_floor_does_not_change_non_mcp_calls() {
        let map = tool_map_with_policy("srv", "search", json!("never"));
        let index = McpToolIndex::new(&map);
        let call = json!({"type": "function_call", "name": "client_only_fn", "call_id": "c1"});
        assert_eq!(
            classify_mcp(&call, &index, ApprovalFloor::Always),
            McpDisposition::NotMcp,
            "the floor must not reclassify a non-MCP call"
        );
    }

    #[test]
    fn approval_floor_forces_approval_predicate() {
        assert!(
            !ApprovalFloor::None.forces_approval(),
            "no floor must not force approval"
        );
        assert!(ApprovalFloor::Always.forces_approval(), "Always must force approval");
        assert_eq!(
            ApprovalFloor::default(),
            ApprovalFloor::None,
            "the floor must default to unset"
        );
    }

    // -------------------------------------------------------------------------
    // read_only selector classification (issue #1398)
    // -------------------------------------------------------------------------

    #[test]
    fn always_read_only_selector_overrides_never_name_match() {
        // The regression: a writable tool is named by `never.tool_names`, but
        // `always.read_only=false` also selects it. `always` precedence must
        // win and require approval — the ignored read_only selector previously
        // left only the `never` name match and returned Automatic.
        let map = tool_map_with_read_only(
            "srv",
            "delete",
            false,
            json!({"always": {"read_only": false}, "never": {"tool_names": ["delete"]}}),
        );
        let index = McpToolIndex::new(&map);
        let call = json!({"type": "function_call", "name": "srv__delete", "call_id": "c1"});
        assert_eq!(
            classify_mcp(&call, &index, ApprovalFloor::None),
            McpDisposition::ApprovalRequired,
            "always.read_only=false selects the writable tool and must win over never.tool_names"
        );
    }

    #[test]
    fn never_read_only_selector_exempts_read_only_tool() {
        // Complementary case: a read-only tool matched by `never.read_only=true`
        // and not forced by `always` executes automatically.
        let map = tool_map_with_read_only(
            "srv",
            "list",
            true,
            json!({"always": {"read_only": false}, "never": {"read_only": true}}),
        );
        let index = McpToolIndex::new(&map);
        let call = json!({"type": "function_call", "name": "srv__list", "call_id": "c1"});
        assert_eq!(
            classify_mcp(&call, &index, ApprovalFloor::None),
            McpDisposition::Automatic
        );
    }

    #[test]
    fn read_only_selector_distinguishes_by_annotation() {
        // The same policy classifies a writable and a read-only tool oppositely,
        // proving the annotation — not just the name — drives the decision.
        let policy = json!({"never": {"read_only": true}});
        let read_only_map = tool_map_with_read_only("srv", "peek", true, policy.clone());
        let writable_map = tool_map_with_read_only("srv", "poke", false, policy);
        let read_only_index = McpToolIndex::new(&read_only_map);
        let writable_index = McpToolIndex::new(&writable_map);
        assert_eq!(
            classify_mcp(
                &json!({"type": "function_call", "name": "srv__peek", "call_id": "c1"}),
                &read_only_index,
                ApprovalFloor::None,
            ),
            McpDisposition::Automatic,
            "a read-only tool matches never.read_only=true"
        );
        assert_eq!(
            classify_mcp(
                &json!({"type": "function_call", "name": "srv__poke", "call_id": "c2"}),
                &writable_index,
                ApprovalFloor::None,
            ),
            McpDisposition::ApprovalRequired,
            "a writable tool does not match never.read_only=true and defaults to approval"
        );
    }

    #[test]
    fn missing_annotation_is_treated_as_writable() {
        // A tool without annotations is writable (readOnlyHint defaults false),
        // so never.read_only=true does not exempt it.
        let map = tool_map_with_policy("srv", "search", json!({"never": {"read_only": true}}));
        let index = McpToolIndex::new(&map);
        let call = json!({"type": "function_call", "name": "srv__search", "call_id": "c1"});
        assert_eq!(
            classify_mcp(&call, &index, ApprovalFloor::None),
            McpDisposition::ApprovalRequired
        );
    }

    #[test]
    fn never_filter_requires_both_name_and_read_only() {
        // `never` carries BOTH selectors, so it exempts a tool only when the
        // name AND the read_only annotation match (logical AND). A regression to
        // OR would wrongly auto-execute the writable tool.
        let policy = json!({"never": {"tool_names": ["list"], "read_only": true}});
        let read_only_map = tool_map_with_read_only("srv", "list", true, policy.clone());
        let writable_map = tool_map_with_read_only("srv", "list", false, policy);
        assert_eq!(
            classify_mcp(
                &json!({"type": "function_call", "name": "srv__list", "call_id": "c1"}),
                &McpToolIndex::new(&read_only_map),
                ApprovalFloor::None,
            ),
            McpDisposition::Automatic,
            "read-only `list` matches both never selectors"
        );
        assert_eq!(
            classify_mcp(
                &json!({"type": "function_call", "name": "srv__list", "call_id": "c2"}),
                &McpToolIndex::new(&writable_map),
                ApprovalFloor::None,
            ),
            McpDisposition::ApprovalRequired,
            "writable `list` matches never.tool_names but not never.read_only, so AND fails"
        );
    }

    #[test]
    fn always_filter_requires_both_name_and_read_only() {
        // The same name flips disposition by annotation: `always` (AND of name +
        // read_only=false) gates only the writable tool, while `never.read_only`
        // exempts the read-only one.
        let policy = json!({"always": {"tool_names": ["delete"], "read_only": false}, "never": {"read_only": true}});
        let writable_map = tool_map_with_read_only("srv", "delete", false, policy.clone());
        let read_only_map = tool_map_with_read_only("srv", "delete", true, policy);
        assert_eq!(
            classify_mcp(
                &json!({"type": "function_call", "name": "srv__delete", "call_id": "c1"}),
                &McpToolIndex::new(&writable_map),
                ApprovalFloor::None,
            ),
            McpDisposition::ApprovalRequired,
            "writable `delete` matches both always selectors"
        );
        assert_eq!(
            classify_mcp(
                &json!({"type": "function_call", "name": "srv__delete", "call_id": "c2"}),
                &McpToolIndex::new(&read_only_map),
                ApprovalFloor::None,
            ),
            McpDisposition::Automatic,
            "read-only `delete` fails always (read_only mismatch) and is exempted by never.read_only=true"
        );
    }

    #[test]
    fn always_read_only_true_wins_over_never_read_only_true() {
        // When both sides select read-only tools, `always` precedence forces
        // approval even though `never` also matches.
        let map = tool_map_with_read_only(
            "srv",
            "audit",
            true,
            json!({"always": {"read_only": true}, "never": {"read_only": true}}),
        );
        let call = json!({"type": "function_call", "name": "srv__audit", "call_id": "c1"});
        assert_eq!(
            classify_mcp(&call, &McpToolIndex::new(&map), ApprovalFloor::None),
            McpDisposition::ApprovalRequired,
            "always must win when both selectors match a read-only tool"
        );
    }

    // -------------------------------------------------------------------------
    // malformed policies fail closed to approval
    // -------------------------------------------------------------------------

    #[test]
    fn malformed_policy_unknown_field_fails_closed() {
        let map = tool_map_with_policy(
            "srv",
            "search",
            json!({"never": {"tool_names": ["search"]}, "sometimes": {}}),
        );
        let index = McpToolIndex::new(&map);
        let call = json!({"type": "function_call", "name": "srv__search", "call_id": "c1"});
        assert_eq!(
            classify_mcp(&call, &index, ApprovalFloor::None),
            McpDisposition::ApprovalRequired,
            "an unknown approval-filter field must reject the whole policy and fail closed"
        );
    }

    #[test]
    fn malformed_policy_invalid_member_type_fails_closed() {
        // `never` is a bare array (the old non-schema fallback) and `read_only`
        // is a string — both outside the schema, so the call fails closed.
        for policy in [json!({"never": ["search"]}), json!({"never": {"read_only": "yes"}})] {
            let map = tool_map_with_policy("srv", "search", policy);
            let index = McpToolIndex::new(&map);
            let call = json!({"type": "function_call", "name": "srv__search", "call_id": "c1"});
            assert_eq!(
                classify_mcp(&call, &index, ApprovalFloor::None),
                McpDisposition::ApprovalRequired,
                "an invalid approval-filter member type must fail closed"
            );
        }
    }

    #[test]
    fn nested_null_selector_fails_closed() {
        // `tool_names` is not nullable in the schema. A `null` selector must not
        // be read as "omitted": otherwise `never.read_only=false` alone would
        // exempt this writable tool and auto-execute it.
        let map = tool_map_with_read_only(
            "srv",
            "delete",
            false,
            json!({"never": {"tool_names": null, "read_only": false}}),
        );
        let index = McpToolIndex::new(&map);
        let call = json!({"type": "function_call", "name": "srv__delete", "call_id": "c1"});
        assert_eq!(
            classify_mcp(&call, &index, ApprovalFloor::None),
            McpDisposition::ApprovalRequired,
            "an explicit null selector is malformed and must fail closed, not drop the selector"
        );
    }

    // -------------------------------------------------------------------------
    // parse_approval_policy / requires_approval units
    // -------------------------------------------------------------------------

    #[test]
    fn parse_filter_captures_names_and_read_only() {
        let def = json!({
            "require_approval": {
                "always": {"tool_names": ["a"], "read_only": false},
                "never": {"read_only": true}
            }
        });
        assert_eq!(
            parse_approval_policy(&def),
            ApprovalPolicy::Filter {
                always: ApprovalSelector {
                    names: Some(vec!["a".to_owned()]),
                    read_only: Some(false),
                },
                never: ApprovalSelector {
                    names: None,
                    read_only: Some(true),
                },
            }
        );
    }

    #[test]
    fn parse_null_and_unknown_string_default_to_always() {
        assert_eq!(
            parse_approval_policy(&json!({"require_approval": null})),
            ApprovalPolicy::Always
        );
        assert_eq!(
            parse_approval_policy(&json!({"require_approval": "maybe"})),
            ApprovalPolicy::Always
        );
        assert_eq!(parse_approval_policy(&json!({})), ApprovalPolicy::Always);
    }

    #[test]
    fn parse_malformed_filters_collapse_to_always() {
        for policy in [
            json!({"require_approval": {"always": ["x"]}}),
            json!({"require_approval": {"never": {"tool_names": [1]}}}),
            json!({"require_approval": {"never": {"read_only": 1}}}),
            json!({"require_approval": {"never": {"unknown": true}}}),
            json!({"require_approval": {"bogus": {}}}),
            json!({"require_approval": 42}),
            // Explicit nulls are not nullable members/selectors in the schema.
            json!({"require_approval": {"never": null}}),
            json!({"require_approval": {"never": {"tool_names": null}}}),
            json!({"require_approval": {"always": {"read_only": null}}}),
        ] {
            assert_eq!(
                parse_approval_policy(&policy),
                ApprovalPolicy::Always,
                "{policy} must fail closed to Always"
            );
        }
    }

    #[test]
    fn empty_filter_member_matches_nothing() {
        // `never: {}` carries no selector, so it exempts no tool and the call
        // falls through to the approval default.
        let policy = ApprovalPolicy::Filter {
            always: ApprovalSelector::default(),
            never: ApprovalSelector::default(),
        };
        assert!(
            requires_approval(&policy, "anything", false),
            "a degenerate empty filter must not auto-approve"
        );
    }

    #[test]
    fn requires_approval_filter_always_precedence_with_read_only() {
        let policy = ApprovalPolicy::Filter {
            always: ApprovalSelector {
                names: None,
                read_only: Some(false),
            },
            never: ApprovalSelector {
                names: Some(vec!["delete".to_owned()]),
                read_only: None,
            },
        };
        // Writable `delete`: matches always (read_only=false) → approval.
        assert!(requires_approval(&policy, "delete", false));
    }
}
