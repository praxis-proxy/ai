// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Operator-owned MCP tool metadata admission and read-only trust (issue #1724).
//!
//! MCP servers supply tool descriptions, schemas, and annotations. A
//! client-selected server is not an authority for its own safety claims, so its
//! `annotations.readOnlyHint` cannot establish a trusted read-only
//! classification. This module makes the trust contract explicit and keeps it
//! **operator-owned**: an admission verdict is always a typed value
//! ([`McpToolVerdict`]) computed afresh from operator configuration — the
//! filter's [`McpAdmissionPolicy`] plus the resolving entry's
//! [`ConnectorTrust`] — never a field carried in request, cached, or restored
//! JSON. Client fields, tool descriptions, annotations, labels, or a matching
//! URL can neither grant nor broaden trust.
//!
//! The verdicts are recorded in [`McpAdmissionDecisions`], a request-scoped
//! snapshot held in `ResponsesState`. Because the resolver re-runs on every
//! request (fresh, cached, or restored) and extends the decisions through
//! deferred discovery and IRR re-entry, the applicable policy and target are
//! re-evaluated each turn, so stale cached policy cannot transfer trust.
//!
//! Admitted tools flow into the model-visible function tools and the dispatch
//! `mcp_tool_map` as before. **Denied** tools are recorded here only: they never
//! reach the model, but keeping their identity lets the agentic loop recognize a
//! forged model-generated call as MCP-owned-and-denied and route it to an
//! explicit dispatch rejection rather than mis-classifying it as a client
//! function.

#[cfg(feature = "openai-mcp-tools")]
use std::collections::{HashMap, HashSet};

#[cfg(feature = "openai-mcp-tools")]
use serde_json::Value;

#[cfg(feature = "openai-mcp-tools")]
use super::mcp_tool_resolve::encode_function_name;

/// Operator admission mode for MCP tool metadata.
#[cfg(feature = "openai-mcp-tools")]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum McpAdmissionMode {
    /// Standard advisory behavior. The client selector `allowed_tools.read_only`
    /// keeps using the server-supplied `readOnlyHint`; no operator admission or
    /// enforcement is applied. This is the default and leaves existing,
    /// non-protected deployments unchanged.
    #[default]
    Advisory,
    /// Operator-enforced protection. Only tool definitions admitted by the
    /// operator trust contract reach the model and may dispatch.
    Protected,
}

/// Request-scoped snapshot of the operator admission policy.
///
/// Carries only the operator decision, never any server- or client-supplied
/// value. Mirrors the `McpConnectorContextPolicy` snapshot pattern so the
/// dispatch filter can re-enforce without re-reading filter configuration.
#[cfg(feature = "openai-mcp-tools")]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct McpAdmissionPolicy {
    /// Admission mode.
    pub mode: McpAdmissionMode,
    /// Operator-enforced read-only: admit only tools with a trusted affirmative
    /// read-only classification.
    pub require_read_only: bool,
}

#[cfg(feature = "openai-mcp-tools")]
impl McpAdmissionPolicy {
    /// Whether protected-mode admission applies.
    pub(crate) fn is_protected(self) -> bool {
        matches!(self.mode, McpAdmissionMode::Protected)
    }
}

/// Operator trust granted to one connector's tool service.
///
/// The scopes are independent: trusting a service for read-only classification
/// does not grant it catalog-admission trust. A declaration that sets neither
/// scope leaves the connector untrusted, so it follows the same protected-mode
/// admission rules as a client-selected direct target.
#[cfg(feature = "openai-mcp-tools")]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ConnectorTrust {
    /// Authoritative for catalog admission: its tools are admitted without a
    /// per-tool digest.
    pub catalog: bool,
    /// Authoritative for read-only classification: its `readOnlyHint` is a
    /// trusted capability decision.
    pub read_only: bool,
}

/// Why a discovered tool was denied admission under protected mode.
#[cfg(feature = "openai-mcp-tools")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum McpDenyReason {
    /// The resolving target has no operator-granted catalog trust — a
    /// client-selected direct `server_url` or a connector without a declared
    /// catalog-trust contract.
    UntrustedSource,
    /// Operator-enforced read-only is active but the trusted source is not
    /// authoritative for read-only classification.
    NoReadOnlyAuthority,
    /// A trusted source classifies the tool as writable.
    Writable,
    /// A trusted source provides no affirmative read-only classification: the
    /// hint is missing, malformed, or internally conflicting.
    UnverifiedReadOnly,
}

#[cfg(feature = "openai-mcp-tools")]
impl McpDenyReason {
    /// Client-safe explanation used in a truthful dispatch rejection.
    pub(crate) fn message(self) -> &'static str {
        match self {
            Self::UntrustedSource => "The platform administrator has forbidden tools from this untrusted MCP service.",
            Self::NoReadOnlyAuthority => {
                "The platform administrator permits only verified read-only tools, and this service cannot verify that requirement."
            },
            Self::Writable => "The platform administrator has forbidden tools that can make changes.",
            Self::UnverifiedReadOnly => {
                "The platform administrator permits only verified read-only tools, and this tool could not be verified."
            },
        }
    }
}

/// Admission verdict for one discovered tool.
#[cfg(feature = "openai-mcp-tools")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum McpToolVerdict {
    /// The tool is admitted. `read_only` is the trusted read-only classification
    /// (only ever `true` when a trusted source affirms it); it is advisory-free.
    Admitted {
        /// Trusted read-only classification for this tool.
        read_only: bool,
    },
    /// The tool is denied and must not reach the model or dispatch.
    Denied(McpDenyReason),
}

/// Trusted read-only classification of a single tool definition.
#[cfg(feature = "openai-mcp-tools")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReadOnlyClass {
    /// A trusted source affirms the tool is read-only.
    Affirmative,
    /// A trusted source reports the tool as writable.
    Writable,
    /// No usable classification: missing, malformed, or conflicting.
    Unverified,
}

/// Classify the read-only status asserted by a tool definition's annotations.
///
/// This is consulted **only** for a source the operator has declared
/// authoritative for read-only classification. `readOnlyHint: true` is
/// affirmative unless contradicted by `destructiveHint: true` (conflicting);
/// `readOnlyHint: false` is writable; absent or non-boolean is unverified.
#[cfg(feature = "openai-mcp-tools")]
fn classify_read_only(tool_def: &Value) -> ReadOnlyClass {
    let annotations = tool_def.get("annotations");
    let Some(hint) = annotations.and_then(|a| a.get("readOnlyHint")) else {
        return ReadOnlyClass::Unverified;
    };
    let Some(read_only) = hint.as_bool() else {
        return ReadOnlyClass::Unverified;
    };
    if !read_only {
        return ReadOnlyClass::Writable;
    }
    // `destructiveHint: true` alongside `readOnlyHint: true` is contradictory
    // (a read-only tool makes no destructive modifications): treat as unverified
    // so a conflicting classification cannot pass operator read-only mode.
    if annotations
        .and_then(|a| a.get("destructiveHint"))
        .and_then(Value::as_bool)
        == Some(true)
    {
        return ReadOnlyClass::Unverified;
    }
    ReadOnlyClass::Affirmative
}

/// Evaluate the admission verdict for one discovered tool under protected mode.
///
/// Advisory mode never reaches here — callers skip verdict construction entirely
/// so existing behavior is byte-for-byte unchanged.
#[cfg(feature = "openai-mcp-tools")]
pub(crate) fn evaluate_tool(policy: McpAdmissionPolicy, trust: ConnectorTrust, tool_def: &Value) -> McpToolVerdict {
    debug_assert!(
        policy.is_protected(),
        "evaluate_tool is only meaningful in protected mode"
    );
    if !trust.catalog {
        return McpToolVerdict::Denied(McpDenyReason::UntrustedSource);
    }
    if policy.require_read_only {
        if !trust.read_only {
            return McpToolVerdict::Denied(McpDenyReason::NoReadOnlyAuthority);
        }
        return match classify_read_only(tool_def) {
            ReadOnlyClass::Affirmative => McpToolVerdict::Admitted { read_only: true },
            ReadOnlyClass::Writable => McpToolVerdict::Denied(McpDenyReason::Writable),
            ReadOnlyClass::Unverified => McpToolVerdict::Denied(McpDenyReason::UnverifiedReadOnly),
        };
    }
    // Admission without operator read-only enforcement: the tool is admitted,
    // and its trusted read-only classification is recorded only when the source
    // is authoritative for it and affirms it.
    let read_only = trust.read_only && matches!(classify_read_only(tool_def), ReadOnlyClass::Affirmative);
    McpToolVerdict::Admitted { read_only }
}

/// Request-scoped record of every admission verdict for the current request.
///
/// The verdict map is keyed by the canonical `(server_label, tool_name)`
/// identity. A parallel index of denied encoded function names lets the agentic
/// loop and dispatcher recognize a model-generated call to a denied tool by the
/// lossy name the model actually emits, failing closed on encoded-name
/// collisions.
#[cfg(feature = "openai-mcp-tools")]
#[derive(Clone, Debug, Default)]
pub(crate) struct McpAdmissionDecisions {
    /// The operator policy snapshot this request was evaluated under.
    policy: McpAdmissionPolicy,
    /// Typed verdicts keyed by canonical `(server_label, tool_name)`.
    verdicts: HashMap<(String, String), McpToolVerdict>,
    /// Denied tools indexed by the encoded model-facing function name, with the
    /// reason, for fast ownership and dispatch lookups.
    denied_encoded: HashMap<String, McpDenyReason>,
    /// Encoded `label__` namespace prefixes of targets excluded without dialing
    /// (an untrusted target in protected mode). Its individual tool names are
    /// unknown — they were never listed — so a forged call to any tool in the
    /// namespace is recognized by prefix and rejected explicitly rather than
    /// falling through as a client function (#1724).
    excluded_namespaces: HashMap<String, McpDenyReason>,
    /// Every model-visible function name in the outbound request — admitted MCP
    /// tools and client-declared functions alike. A lossy encoding can collapse a
    /// denied tool onto the same name as one of these; the available tool is the
    /// one shown to the model, so it wins. This set lets the denied lookups
    /// ignore any name that is still a live function, preventing a denied
    /// collision from rejecting a legitimately admitted tool or a client
    /// function (#1724).
    available_encoded: HashSet<String>,
}

#[cfg(feature = "openai-mcp-tools")]
impl McpAdmissionDecisions {
    /// Create an empty decision set bound to an operator policy snapshot.
    pub(crate) fn new(policy: McpAdmissionPolicy) -> Self {
        Self {
            policy,
            verdicts: HashMap::new(),
            denied_encoded: HashMap::new(),
            excluded_namespaces: HashMap::new(),
            available_encoded: HashSet::new(),
        }
    }

    /// The operator policy these decisions were evaluated under.
    pub(crate) fn policy(&self) -> McpAdmissionPolicy {
        self.policy
    }

    /// Record a verdict for one tool identity, maintaining the encoded indexes.
    pub(crate) fn record(&mut self, server_label: &str, tool_name: &str, verdict: McpToolVerdict) {
        match verdict {
            McpToolVerdict::Denied(reason) => {
                self.denied_encoded
                    .insert(encode_function_name(server_label, tool_name), reason);
            },
            McpToolVerdict::Admitted { .. } => {
                self.available_encoded
                    .insert(encode_function_name(server_label, tool_name));
            },
        }
        self.verdicts
            .insert((server_label.to_owned(), tool_name.to_owned()), verdict);
    }

    /// Record a target excluded without dialing so forged calls into its encoded
    /// `label__` namespace can be rejected explicitly, even though its individual
    /// tool names were never listed.
    pub(crate) fn record_excluded_namespace(&mut self, server_label: &str, reason: McpDenyReason) {
        // `encode_function_name(label, "")` yields the sanitized `label__` prefix
        // every tool of this server encodes under.
        self.excluded_namespaces
            .insert(encode_function_name(server_label, ""), reason);
    }

    /// Record every model-visible function name from the outbound `tools` array.
    ///
    /// Called once the request body is rewritten so both admitted MCP function
    /// tools and client-declared functions are known. A name recorded here is
    /// never reported as denied, so a denied tool's lossy encoded-name collision
    /// cannot reject a legitimately available call.
    pub(crate) fn mark_available_functions(&mut self, tools: &[Value]) {
        for name in tools.iter().filter_map(|tool| {
            (tool.get("type").and_then(Value::as_str) == Some("function"))
                .then(|| tool.get("name").and_then(Value::as_str))
                .flatten()
        }) {
            self.available_encoded.insert(name.to_owned());
        }
    }

    /// Look up a verdict by canonical tool identity.
    pub(crate) fn verdict(&self, server_label: &str, tool_name: &str) -> Option<McpToolVerdict> {
        self.verdicts
            .get(&(server_label.to_owned(), tool_name.to_owned()))
            .copied()
    }

    /// Whether the encoded model-facing function name denotes a denied tool.
    ///
    /// A model-visible function (an admitted MCP tool or a client function) that
    /// shares the same lossy encoded name wins: the name is still available to the
    /// model, so it is not reported as denied.
    pub(crate) fn is_denied_encoded(&self, encoded_name: &str) -> bool {
        self.denied_reason(encoded_name).is_some()
    }

    /// The denial reason for an encoded model-facing function name.
    ///
    /// A model-visible function (admitted MCP tool or client function) sharing the
    /// name always wins and is never reported denied. Otherwise the name is denied
    /// if it is a specific discovered-but-denied tool, or falls within the encoded
    /// namespace of a target excluded without dialing.
    pub(crate) fn denied_reason(&self, encoded_name: &str) -> Option<McpDenyReason> {
        if self.available_encoded.contains(encoded_name) {
            return None;
        }
        if let Some(reason) = self.denied_encoded.get(encoded_name) {
            return Some(*reason);
        }
        self.excluded_namespaces
            .iter()
            .find_map(|(prefix, reason)| encoded_name.starts_with(prefix).then_some(*reason))
    }

    /// Whether this request denies any MCP call — a specific discovered-but-denied
    /// tool, or any tool inside an excluded target's namespace.
    pub(crate) fn has_denied(&self) -> bool {
        !self.denied_encoded.is_empty() || !self.excluded_namespaces.is_empty()
    }

    /// Re-enforce admission at dispatch for a call that resolved to a tool-map
    /// entry (`server_label`, `tool_name`). Returns the denial reason when the
    /// call must not execute.
    ///
    /// In advisory mode this never denies. In protected mode an admitted entry
    /// must carry a matching admitted verdict; a missing verdict (a forged or
    /// bypassed map entry) or a writable verdict under operator read-only mode
    /// fails closed. This is defense in depth: discovery already excluded denied
    /// tools, and this guarantees a model-generated call to a resolved entry is
    /// re-checked against the operator-owned verdict before any side effect.
    pub(crate) fn admitted_denial(&self, server_label: &str, tool_name: &str) -> Option<McpDenyReason> {
        if !self.policy.is_protected() {
            return None;
        }
        match self.verdict(server_label, tool_name) {
            Some(McpToolVerdict::Admitted { read_only }) => {
                (self.policy.require_read_only && !read_only).then_some(McpDenyReason::Writable)
            },
            Some(McpToolVerdict::Denied(reason)) => Some(reason),
            None => Some(McpDenyReason::UntrustedSource),
        }
    }
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing, reason = "tests")]
mod tests {
    use serde_json::json;

    use super::{
        ConnectorTrust, McpAdmissionDecisions, McpAdmissionMode, McpAdmissionPolicy, McpDenyReason, McpToolVerdict,
        evaluate_tool,
    };

    fn protected(require_read_only: bool) -> McpAdmissionPolicy {
        McpAdmissionPolicy {
            mode: McpAdmissionMode::Protected,
            require_read_only,
        }
    }

    fn tool(annotations: Option<serde_json::Value>) -> serde_json::Value {
        let mut def = json!({"name": "do_thing", "inputSchema": {"type": "object"}});
        if let Some(a) = annotations {
            def["annotations"] = a;
        }
        def
    }

    #[test]
    fn policy_denial_messages_attribute_restrictions_to_the_platform_administrator() {
        for reason in [
            McpDenyReason::UntrustedSource,
            McpDenyReason::NoReadOnlyAuthority,
            McpDenyReason::Writable,
            McpDenyReason::UnverifiedReadOnly,
        ] {
            assert!(reason.message().contains("platform administrator"));
            assert!(!reason.message().contains("protected mode"));
            assert!(!reason.message().contains("tool_admission"));
        }
    }

    #[test]
    fn untrusted_source_denied_regardless_of_hint() {
        let verdict = evaluate_tool(
            protected(false),
            ConnectorTrust::default(),
            &tool(Some(json!({"readOnlyHint": true}))),
        );
        assert_eq!(verdict, McpToolVerdict::Denied(McpDenyReason::UntrustedSource));
    }

    #[test]
    fn catalog_trust_admits_without_read_only_enforcement() {
        let trust = ConnectorTrust {
            catalog: true,
            read_only: false,
        };
        let verdict = evaluate_tool(protected(false), trust, &tool(None));
        assert_eq!(verdict, McpToolVerdict::Admitted { read_only: false });
    }

    #[test]
    fn catalog_trust_records_trusted_read_only_when_authoritative() {
        let trust = ConnectorTrust {
            catalog: true,
            read_only: true,
        };
        let verdict = evaluate_tool(protected(false), trust, &tool(Some(json!({"readOnlyHint": true}))));
        assert_eq!(verdict, McpToolVerdict::Admitted { read_only: true });
    }

    #[test]
    fn read_only_without_authority_denied_in_enforced_mode() {
        let trust = ConnectorTrust {
            catalog: true,
            read_only: false,
        };
        let verdict = evaluate_tool(protected(true), trust, &tool(Some(json!({"readOnlyHint": true}))));
        assert_eq!(verdict, McpToolVerdict::Denied(McpDenyReason::NoReadOnlyAuthority));
    }

    #[test]
    fn affirmative_read_only_admitted_in_enforced_mode() {
        let trust = ConnectorTrust {
            catalog: true,
            read_only: true,
        };
        let verdict = evaluate_tool(protected(true), trust, &tool(Some(json!({"readOnlyHint": true}))));
        assert_eq!(verdict, McpToolVerdict::Admitted { read_only: true });
    }

    #[test]
    fn writable_tool_denied_in_enforced_mode() {
        let trust = ConnectorTrust {
            catalog: true,
            read_only: true,
        };
        let verdict = evaluate_tool(protected(true), trust, &tool(Some(json!({"readOnlyHint": false}))));
        assert_eq!(verdict, McpToolVerdict::Denied(McpDenyReason::Writable));
    }

    #[test]
    fn missing_hint_denied_in_enforced_mode() {
        let trust = ConnectorTrust {
            catalog: true,
            read_only: true,
        };
        let verdict = evaluate_tool(protected(true), trust, &tool(None));
        assert_eq!(verdict, McpToolVerdict::Denied(McpDenyReason::UnverifiedReadOnly));
    }

    #[test]
    fn malformed_hint_denied_in_enforced_mode() {
        let trust = ConnectorTrust {
            catalog: true,
            read_only: true,
        };
        let verdict = evaluate_tool(protected(true), trust, &tool(Some(json!({"readOnlyHint": "yes"}))));
        assert_eq!(verdict, McpToolVerdict::Denied(McpDenyReason::UnverifiedReadOnly));
    }

    #[test]
    fn conflicting_hint_denied_in_enforced_mode() {
        let trust = ConnectorTrust {
            catalog: true,
            read_only: true,
        };
        let verdict = evaluate_tool(
            protected(true),
            trust,
            &tool(Some(json!({"readOnlyHint": true, "destructiveHint": true}))),
        );
        assert_eq!(verdict, McpToolVerdict::Denied(McpDenyReason::UnverifiedReadOnly));
    }

    #[test]
    fn decisions_track_denied_encoded_names() {
        let mut decisions = McpAdmissionDecisions::new(protected(true));
        decisions.record("gw", "delete_repo", McpToolVerdict::Denied(McpDenyReason::Writable));
        decisions.record("gw", "search", McpToolVerdict::Admitted { read_only: true });

        assert!(decisions.is_denied_encoded("gw__delete_repo"));
        assert_eq!(
            decisions.denied_reason("gw__delete_repo"),
            Some(McpDenyReason::Writable)
        );
        assert!(!decisions.is_denied_encoded("gw__search"));
        assert_eq!(
            decisions.verdict("gw", "search"),
            Some(McpToolVerdict::Admitted { read_only: true })
        );
        assert!(decisions.policy().is_protected());
        assert!(decisions.has_denied());
    }

    #[test]
    fn admitted_tool_wins_over_denied_encoded_collision() {
        // `read?docs` and `read_docs` both encode to `gw__read_docs` (lossy `?`
        // → `_`). One is denied (writable), the other admitted (read-only). The
        // admitted tool is the one shown to the model and resolvable at dispatch,
        // so the shared encoded name must not be reported as denied.
        let mut decisions = McpAdmissionDecisions::new(protected(true));
        decisions.record("gw", "read?docs", McpToolVerdict::Denied(McpDenyReason::Writable));
        decisions.record("gw", "read_docs", McpToolVerdict::Admitted { read_only: true });

        assert!(
            !decisions.is_denied_encoded("gw__read_docs"),
            "an admitted tool must not be shadowed by a denied encoded-name collision"
        );
        assert_eq!(decisions.denied_reason("gw__read_docs"), None);
        // A denied tool with no admitted collision is still reported denied.
        decisions.record("gw", "delete_all", McpToolVerdict::Denied(McpDenyReason::Writable));
        assert_eq!(decisions.denied_reason("gw__delete_all"), Some(McpDenyReason::Writable));
    }

    #[test]
    fn client_function_wins_over_denied_encoded_collision() {
        // A trusted connector's writable tool is denied under read-only policy,
        // and the request also declares a client function that encodes to the same
        // name. The client function stays model-visible, so a call to it must not
        // be rejected as denied MCP work (#1724).
        let mut decisions = McpAdmissionDecisions::new(protected(true));
        decisions.record("gw", "delete", McpToolVerdict::Denied(McpDenyReason::Writable));
        decisions.mark_available_functions(&[json!({"type": "function", "name": "gw__delete"})]);

        assert!(
            !decisions.is_denied_encoded("gw__delete"),
            "a client function sharing a denied tool's encoded name must be preserved"
        );
        assert_eq!(decisions.denied_reason("gw__delete"), None);
    }

    #[test]
    fn excluded_target_namespace_is_denied_without_dialing() {
        // An untrusted target is excluded without dialing, so its individual tool
        // names are unknown. A forged call into its encoded namespace is still
        // rejected explicitly by prefix (#1724).
        let mut decisions = McpAdmissionDecisions::new(protected(false));
        decisions.record_excluded_namespace("docs", McpDenyReason::UntrustedSource);

        assert!(decisions.has_denied());
        assert_eq!(
            decisions.denied_reason("docs__search"),
            Some(McpDenyReason::UntrustedSource)
        );
        assert_eq!(
            decisions.denied_reason("other__thing"),
            None,
            "unrelated namespaces are unaffected"
        );

        // A client function sharing the name wins and stays available.
        decisions.mark_available_functions(&[json!({"type": "function", "name": "docs__search"})]);
        assert_eq!(decisions.denied_reason("docs__search"), None);
    }
}
