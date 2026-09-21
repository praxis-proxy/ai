// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Unit tests for the `openai_client_tool_compat` filter.

use std::collections::HashSet;

use praxis_filter::body::MAX_JSON_BODY_BYTES;
use serde_json::{Value, json};

use super::*;
use crate::test_utils::{make_filter_context, make_request};

// -----------------------------------------------------------------------------
// Test helpers
// -----------------------------------------------------------------------------

/// A filter with the default caps.
fn filter() -> ClientToolCompatFilter {
    ClientToolCompatFilter {
        max_rewritten_body_bytes: MAX_JSON_BODY_BYTES,
        max_client_tools: 512,
    }
}

/// Build a set of lowered client-executed `call_id`s for history-lowering tests.
fn lowered_ids(ids: &[&str]) -> HashSet<String> {
    ids.iter().map(|id| (*id).to_owned()).collect()
}

/// Assert the action is a rejection and return its `(status, message)`.
fn reject_parts(action: &FilterAction) -> (u16, String) {
    let FilterAction::Reject(rejection) = action else {
        panic!("expected FilterAction::Reject");
    };
    let body = rejection.body.clone().expect("rejection has a body");
    let parsed: Value = serde_json::from_slice(&body).expect("rejection body is JSON");
    let message = parsed["error"]["message"]
        .as_str()
        .expect("rejection carries an error message")
        .to_owned();
    (rejection.status, message)
}

/// Restore a buffered response body and return the parsed JSON.
fn restore(state: &ResponsesState, response: &Value) -> Value {
    let serialized = filter()
        .restore_response(state, response.to_string().as_bytes())
        .expect("restoration succeeds")
        .expect("restoration rewrites the body");
    serde_json::from_slice(serialized.as_bytes()).expect("restored body is JSON")
}

// -----------------------------------------------------------------------------
// Name and id helpers
// -----------------------------------------------------------------------------

#[test]
fn namespace_member_name_is_readable_when_short() {
    assert_eq!(namespace_member_name("git", "commit"), "agentic_ns__git__commit");
}

#[test]
fn namespace_member_name_hashes_when_too_long() {
    let namespace = "n".repeat(40);
    let member = "m".repeat(40);
    let flat = namespace_member_name(&namespace, &member);
    assert!(
        flat.chars().count() <= MAX_FUNCTION_NAME_LEN,
        "must fit the schema length"
    );
    assert!(is_valid_function_name(&flat), "hashed name must be schema-valid");
    assert_eq!(
        flat,
        namespace_member_name(&namespace, &member),
        "hashing must be deterministic"
    );
    assert!(flat.starts_with("agentic_ns__"), "readable prefix is retained");
}

#[test]
fn namespace_member_name_distinguishes_distinct_long_members() {
    let namespace = "n".repeat(40);
    let a = namespace_member_name(&namespace, &"a".repeat(40));
    let b = namespace_member_name(&namespace, &"b".repeat(40));
    assert_ne!(a, b, "distinct members must not collapse to the same hashed name");
}

#[test]
fn namespace_member_name_hashed_form_is_disjoint_from_verbatim() {
    // Regression for the forward-FNV wire-name collision. The hash-truncation branch of
    // `namespace_member_name` once emitted `{readable}__{hash:016x}`, byte-shaped exactly
    // like a verbatim `..__{member}` tail. A short member whose name is the 16-hex FNV of a
    // longer member's full name therefore flattened to the *same* wire as that longer member
    // — no hash search, just forward evaluation of the public FNV constants. That aliased two
    // distinct members in the reverse map, the reclaim budget, and `tool_choice`. The hashed
    // form now carries a `___` domain-separator marker that a verbatim wire can never contain,
    // so the two branches are byte-disjoint.
    let namespace = "a".repeat(34);
    let long_member = "m".repeat(17);
    let long_full = format!("{NAMESPACE_MEMBER_PREFIX}{namespace}__{long_member}");
    assert!(
        long_full.chars().count() > MAX_FUNCTION_NAME_LEN,
        "the long member must take the hash-truncation branch"
    );

    let hashed = namespace_member_name(&namespace, &long_member);
    assert!(
        hashed.contains("___"),
        "the hashed form carries the `___` domain-separator: {hashed}"
    );
    assert!(
        hashed.chars().count() <= MAX_FUNCTION_NAME_LEN,
        "the hashed form still fits the schema length"
    );
    assert!(is_valid_function_name(&hashed), "the hashed form stays schema-valid");

    // The verbatim twin: a member whose name is the forward FNV hex of the long member's full
    // name. Its own full name is exactly 64 chars, so it takes the verbatim branch, and before
    // the fix it flattened byte-for-byte onto the long member's hashed wire.
    let colliding_member = format!("{:016x}", stable_name_hash(&long_full));
    let verbatim = namespace_member_name(&namespace, &colliding_member);
    assert_eq!(
        verbatim.chars().count(),
        MAX_FUNCTION_NAME_LEN,
        "the verbatim twin is the 64-char boundary case"
    );
    assert!(
        !verbatim.contains("___"),
        "a verbatim wire never contains three underscores: {verbatim}"
    );
    assert_ne!(
        verbatim, hashed,
        "the two distinct members must no longer flatten to one wire name"
    );
}

#[test]
fn custom_public_item_id_prefixes_and_hashes() {
    assert_eq!(custom_public_item_id("ctc_keep"), "ctc_keep");
    assert_eq!(custom_public_item_id("fc_abc"), "ctc_abc");
    let hashed = custom_public_item_id("weird");
    assert!(hashed.starts_with("ctc_"), "unprefixed ids get a hashed ctc_ id");
    assert_eq!(hashed, custom_public_item_id("weird"), "hashing is deterministic");
}

#[test]
fn shell_public_item_id_prefixes_and_hashes() {
    assert_eq!(shell_public_item_id("sh_keep"), "sh_keep");
    assert_eq!(shell_public_item_id("fc_abc"), "sh_abc");
    assert!(
        shell_public_item_id("weird").starts_with("sh_"),
        "an unprefixed id gets a hashed sh_ id"
    );
}

#[test]
fn tool_search_public_item_id_prefixes_and_hashes() {
    assert_eq!(tool_search_public_item_id("tsc_keep"), "tsc_keep");
    assert_eq!(tool_search_public_item_id("fc_abc"), "tsc_abc");
    assert!(
        tool_search_public_item_id("weird").starts_with("tsc_"),
        "an unprefixed id gets a hashed tsc_ id"
    );
}

#[test]
fn is_valid_function_name_enforces_schema() {
    assert!(
        is_valid_function_name("run_python-1"),
        "letters, digits, underscores and hyphens are accepted"
    );
    assert!(!is_valid_function_name(""), "an empty name is rejected");
    assert!(!is_valid_function_name("has space"), "whitespace is rejected");
    assert!(
        !is_valid_function_name(&"x".repeat(65)),
        "a name longer than the schema limit is rejected"
    );
}

#[test]
fn input_from_arguments_is_fail_open() {
    assert_eq!(input_from_arguments(r#""raw string""#), "raw string");
    assert_eq!(input_from_arguments(r#"{"input":"unwrapped"}"#), "unwrapped");
    assert_eq!(input_from_arguments("not json"), "not json");
    assert_eq!(input_from_arguments(r#"{"other":1}"#), r#"{"other":1}"#);
}

// -----------------------------------------------------------------------------
// Custom tool lowering
// -----------------------------------------------------------------------------

#[test]
fn lowers_custom_tool_to_function() {
    let mut state = ResponsesState::from_request_body(json!({
        "model": "m",
        "input": "hi",
        "tools": [{
            "type": "custom",
            "name": "run_python",
            "description": "Run python",
            "format": {"type": "text"},
            "grammar": "start: ..."
        }],
    }));
    filter().lower_request(&mut state, false).expect("lowering succeeds");

    let tools = state.request_body["tools"].as_array().expect("tools array");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0]["type"], "function");
    assert_eq!(tools[0]["name"], "run_python");
    assert_eq!(tools[0]["strict"], true);
    assert_eq!(tools[0]["parameters"]["properties"]["input"]["type"], "string");
    assert_eq!(tools[0]["parameters"]["required"], json!(["input"]));
    let description = tools[0]["description"].as_str().expect("description");
    assert!(description.contains("Run python"), "keeps the declared description");
    assert!(
        description.contains("Provide the raw tool input"),
        "explains the input field"
    );
    assert!(
        description.contains("\"grammar\""),
        "preserves extra declaration fields the model must respect"
    );

    let lowered = state.client_tool_lowering.get("run_python").expect("reverse entry");
    assert_eq!(lowered.restore, ClientToolRestore::Custom);
    assert_eq!(lowered.original_name, "run_python");
    assert!(lowered.namespace.is_none(), "a top-level custom tool has no namespace");

    let echo = state.client_tool_echo.as_ref().expect("echo snapshot");
    assert_eq!(echo.tools[0]["type"], "custom");
    assert!(state.request_body_requires_rebuild(), "rebuild is requested");
}

#[test]
fn custom_without_format_lowers() {
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "custom", "name": "c"}],
    }));
    filter().lower_request(&mut state, false).expect("lowering succeeds");
    assert_eq!(state.request_body["tools"][0]["type"], "function");
}

#[test]
fn deferred_top_level_custom_is_withheld() {
    // A deferred `custom` declaration is not callable until a `tool_search` loads
    // it, so it is withheld from the outbound set rather than rejected (deferred
    // custom is a valid Responses construct). When it is the only rich tool, the
    // outbound callable set is empty and the client's originals are echoed back on
    // the response.
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "custom", "name": "c", "defer_loading": true}],
    }));
    filter().lower_request(&mut state, false).expect("lowering succeeds");
    assert_eq!(
        state.request_body["tools"],
        json!([]),
        "the deferred custom is withheld from the outbound callable set"
    );
    let echo = state
        .client_tool_echo
        .as_ref()
        .expect("the client originals are echoed back on the response");
    assert_eq!(
        echo.tools[0]["type"], "custom",
        "the client's original custom tool is echoed back on the response"
    );
    assert!(
        state.client_tool_lowering.is_empty(),
        "nothing was lowered, so there is no restoration recipe"
    );
    assert!(
        state.request_body_requires_rebuild(),
        "the withheld outbound set rebuilds the request body"
    );
}

#[test]
fn custom_with_non_text_format_is_rejected() {
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{
            "type": "custom",
            "name": "c",
            "format": {"type": "grammar", "syntax": "lark", "definition": "start: ..."}
        }],
    }));
    let action = filter().lower_request(&mut state, false).expect_err("must reject");
    let (status, message) = reject_parts(&action);
    assert_eq!(status, 400);
    assert!(
        message.contains("format"),
        "the rejection names the offending field: {message}"
    );
    assert_eq!(
        state.request_body["tools"][0]["type"], "custom",
        "tools restored verbatim"
    );
}

#[test]
fn streaming_rejection_is_a_400_before_any_upstream_call() {
    // The `on_request_body` guard fails closed on a streaming request that
    // declares rich client tools: it recognizes the rich tool and produces the
    // 400 rejection, so lowered private function names never reach the wire.
    let state = ResponsesState::from_request_body(json!({
        "stream": true,
        "tools": [{"type": "custom", "name": "apply_patch"}],
    }));
    assert!(
        request_has_rich_client_tool(&state),
        "a custom tool is a rich client tool the streaming guard must catch"
    );
    let (status, message) = reject_parts(&reject_streaming_unsupported());
    assert_eq!(status, 400);
    assert!(
        message.contains("streaming is not supported") && message.contains("stream=false"),
        "the rejection explains the streaming limitation: {message}"
    );
}

#[test]
fn streaming_guard_ignores_requests_without_rich_client_tools() {
    // A streaming request that declares only a plain function tool is not rich, so
    // the guard does not fire and native streaming stays a transparent passthrough.
    let state = ResponsesState::from_request_body(json!({
        "stream": true,
        "tools": [{"type": "function", "name": "get_weather"}],
    }));
    assert!(
        !request_has_rich_client_tool(&state),
        "a plain function tool must not trip the streaming guard"
    );
}

// -----------------------------------------------------------------------------
// Namespace tool lowering
// -----------------------------------------------------------------------------

#[test]
fn lowers_namespace_members_to_flat_functions() {
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{
            "type": "namespace",
            "name": "git",
            "description": "Git version control operations.",
            "tools": [
                {"type": "function", "name": "commit", "parameters": {"type": "object"}},
                {"type": "function", "name": "push", "parameters": {"type": "object"}}
            ]
        }],
    }));
    filter().lower_request(&mut state, false).expect("lowering succeeds");

    let tools = state.request_body["tools"].as_array().expect("tools array");
    assert_eq!(tools.len(), 2);
    assert_eq!(tools[0]["type"], "function");
    assert_eq!(tools[0]["name"], "agentic_ns__git__commit");
    assert_eq!(tools[1]["name"], "agentic_ns__git__push");

    let lowered = state
        .client_tool_lowering
        .get("agentic_ns__git__commit")
        .expect("reverse entry");
    assert_eq!(lowered.restore, ClientToolRestore::Namespace);
    assert_eq!(lowered.original_name, "commit");
    assert_eq!(lowered.namespace.as_deref(), Some("git"));
}

#[test]
fn empty_namespace_is_rejected() {
    // `NamespaceToolParam.tools` is required with `minItems: 1`; an empty array is
    // schema-invalid and must fail closed rather than be silently dropped.
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "namespace", "name": "git", "description": "Git operations.", "tools": []}],
    }));
    let action = filter()
        .lower_request(&mut state, false)
        .expect_err("an empty namespace must fail closed");
    let (status, message) = reject_parts(&action);
    assert_eq!(status, 400, "an empty namespace is a bad request");
    assert!(
        message.contains("non-empty tools array"),
        "the rejection explains the empty namespace: {message}"
    );
    assert!(state.client_tool_echo.is_none(), "no echo on a rejected request");
}

#[test]
fn namespace_with_missing_tools_is_rejected() {
    // A missing `tools` array is schema-invalid and must fail closed even when
    // another tool in the request is lowerable.
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [
            {"type": "namespace", "name": "git", "description": "Git operations."},
            {"type": "custom", "name": "apply_patch"}
        ],
    }));
    let action = filter()
        .lower_request(&mut state, false)
        .expect_err("a namespace without tools must fail closed");
    let (status, _message) = reject_parts(&action);
    assert_eq!(status, 400, "a namespace without tools is a bad request");
    assert!(state.client_tool_echo.is_none(), "no echo on a rejected request");
    assert_eq!(
        state.request_body["tools"][0]["type"], "namespace",
        "the request tools are rolled back to the client's originals"
    );
}

// -----------------------------------------------------------------------------
// Shell tool lowering
// -----------------------------------------------------------------------------

#[test]
fn lowers_local_shell_to_function() {
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "shell", "environment": {"type": "local"}}],
    }));
    filter().lower_request(&mut state, false).expect("lowering succeeds");
    let tools = state.request_body["tools"].as_array().expect("tools array");
    assert_eq!(tools[0]["type"], "function");
    assert_eq!(tools[0]["name"], "shell");
    assert_eq!(tools[0]["parameters"]["required"], json!(["commands"]));
    assert_eq!(
        state.client_tool_lowering.get("shell").expect("reverse entry").restore,
        ClientToolRestore::Shell
    );
}

#[test]
fn non_local_shell_is_left_untouched() {
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "shell", "environment": {"type": "provider"}}],
    }));
    filter().lower_request(&mut state, false).expect("passthrough succeeds");
    assert_eq!(
        state.request_body["tools"][0]["type"], "shell",
        "not a rich local shell"
    );
    assert!(state.client_tool_echo.is_none(), "a provider shell takes no echo");
    assert!(
        !state.request_body_requires_rebuild(),
        "a provider shell needs no rebuild"
    );
}

// -----------------------------------------------------------------------------
// Tool search lowering
// -----------------------------------------------------------------------------

#[test]
fn lowers_client_tool_search_to_function() {
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "tool_search"}],
    }));
    filter().lower_request(&mut state, false).expect("lowering succeeds");
    let tools = state.request_body["tools"].as_array().expect("tools array");
    assert_eq!(tools[0]["type"], "function");
    assert_eq!(tools[0]["name"], "tool_search");
    assert_eq!(tools[0]["description"], "Search the client tool catalog");
    assert_eq!(
        tools[0]["parameters"]["properties"]["query"]["description"],
        "A concise description of the needed capabilities."
    );
    assert_eq!(
        state
            .client_tool_lowering
            .get("tool_search")
            .expect("reverse entry")
            .restore,
        ClientToolRestore::ToolSearch
    );
}

#[test]
fn server_executed_tool_search_is_left_untouched() {
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "tool_search", "execution": "server"}],
    }));
    filter().lower_request(&mut state, false).expect("passthrough succeeds");
    assert_eq!(state.request_body["tools"][0]["type"], "tool_search");
    assert!(
        state.client_tool_echo.is_none(),
        "a server-executed tool_search takes no echo"
    );
}

// -----------------------------------------------------------------------------
// Collision and cap fail-closed behaviour
// -----------------------------------------------------------------------------

#[test]
fn collision_between_lowered_names_is_rejected() {
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [
            {"type": "custom", "name": "shell"},
            {"type": "shell", "environment": {"type": "local"}}
        ],
    }));
    let action = filter().lower_request(&mut state, false).expect_err("must reject");
    let (status, message) = reject_parts(&action);
    assert_eq!(status, 400);
    assert!(
        message.contains("collides"),
        "the rejection explains the name collision: {message}"
    );
    assert_eq!(
        state.request_body["tools"][0]["type"], "custom",
        "tools restored verbatim"
    );
    assert_eq!(state.request_body["tools"][1]["type"], "shell");
}

#[test]
fn collision_with_passthrough_function_is_rejected() {
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [
            {"type": "function", "name": "run"},
            {"type": "custom", "name": "run"}
        ],
    }));
    let action = filter().lower_request(&mut state, false).expect_err("must reject");
    assert_eq!(reject_parts(&action).0, 400);
    assert_eq!(
        state.request_body["tools"][0]["type"], "function",
        "tools restored verbatim"
    );
}

#[test]
fn exceeding_max_client_tools_is_rejected() {
    let capped = ClientToolCompatFilter {
        max_rewritten_body_bytes: MAX_JSON_BODY_BYTES,
        max_client_tools: 1,
    };
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [
            {"type": "custom", "name": "a"},
            {"type": "custom", "name": "b"}
        ],
    }));
    let action = capped.lower_request(&mut state, false).expect_err("must reject");
    let (status, message) = reject_parts(&action);
    assert_eq!(status, 400);
    assert!(
        message.contains("maximum of 1"),
        "the rejection explains the tool-count cap: {message}"
    );
    assert_eq!(
        state.request_body["tools"][0]["type"], "custom",
        "tools restored verbatim"
    );
}

// -----------------------------------------------------------------------------
// Native passthrough
// -----------------------------------------------------------------------------

#[test]
fn native_function_tools_are_passthrough() {
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "function", "name": "f", "parameters": {"type": "object"}}],
    }));
    filter().lower_request(&mut state, false).expect("passthrough succeeds");
    assert_eq!(state.request_body["tools"][0]["type"], "function");
    assert!(state.client_tool_echo.is_none(), "native traffic takes no echo");
    assert!(state.client_tool_lowering.is_empty(), "native traffic lowers nothing");
    assert!(
        !state.request_body_requires_rebuild(),
        "native traffic needs no rebuild"
    );
}

#[test]
fn request_without_tools_is_passthrough() {
    let mut state = ResponsesState::from_request_body(json!({"model": "m", "input": "hi"}));
    filter().lower_request(&mut state, false).expect("passthrough succeeds");
    assert!(
        state.client_tool_echo.is_none(),
        "a request without tools takes no echo"
    );
    assert!(
        !state.request_body_requires_rebuild(),
        "a request without tools needs no rebuild"
    );
}

#[test]
fn native_non_array_tools_fail_closed() {
    // A non-array `tools` with no rich client tool and no discovery previously took
    // the native passthrough path and forwarded the malformed value verbatim; it now
    // fails closed uniformly, matching the discovery path — a non-array `tools` is
    // structurally malformed for this filter, whose whole contract treats `tools` as
    // an array. Null/absent still pass through (see `request_without_tools_is_passthrough`).
    let mut state = ResponsesState::from_request_body(json!({
        "model": "m",
        "input": "hi",
        "tools": {"type": "custom", "name": "x"}
    }));
    let action = filter()
        .lower_request(&mut state, false)
        .expect_err("a non-array tools must fail closed on the passthrough path too");
    let (status, message) = reject_parts(&action);
    assert_eq!(status, 400, "a malformed tools value is a bad request");
    assert!(
        message.contains("tools must be a JSON array"),
        "the rejection names the malformed field: {message}"
    );
    assert_eq!(
        state.request_body["tools"],
        json!({"type": "custom", "name": "x"}),
        "the malformed tools value is left untouched, not normalized"
    );
}

// -----------------------------------------------------------------------------
// tool_choice lowering
// -----------------------------------------------------------------------------

#[test]
fn lowers_custom_tool_choice_to_function() {
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "custom", "name": "c"}],
        "tool_choice": {"type": "custom", "name": "c"},
    }));
    filter().lower_request(&mut state, false).expect("lowering succeeds");
    assert_eq!(state.request_body["tool_choice"]["type"], "function");
    assert_eq!(state.request_body["tool_choice"]["name"], "c");
}

#[test]
fn lowers_namespaced_tool_choice_to_flat_function() {
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "namespace", "name": "git", "description": "Git operations.", "tools": [{"type": "function", "name": "commit"}]}],
        "tool_choice": {"type": "function", "name": "commit", "namespace": "git"},
    }));
    filter().lower_request(&mut state, false).expect("lowering succeeds");
    assert_eq!(state.request_body["tool_choice"]["name"], "agentic_ns__git__commit");
    assert!(
        state.request_body["tool_choice"].get("namespace").is_none(),
        "the namespace selector is flattened into the private function name"
    );
}

#[test]
fn lowers_namespaced_custom_tool_choice_to_flat_function() {
    // #1158: a forced namespaced custom member lowers to the same flat function
    // selector as a namespaced function member.
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{
            "type": "namespace",
            "name": "git",
            "description": "Git operations.",
            "tools": [{"type": "custom", "name": "freeform"}]
        }],
        "tool_choice": {"type": "custom", "name": "freeform", "namespace": "git"},
    }));
    filter().lower_request(&mut state, false).expect("lowering succeeds");
    let choice = &state.request_body["tool_choice"];
    assert_eq!(
        choice["type"], "function",
        "a namespaced custom selector lowers to a function selector: {choice}"
    );
    assert_eq!(choice["name"], "agentic_ns__git__freeform", "flattened member name");
    assert!(
        choice.get("namespace").is_none(),
        "the namespace is folded into the flat name so the backend never sees it: {choice}"
    );
}

#[test]
fn undeclared_namespaced_custom_tool_choice_is_rejected() {
    // A forced namespaced custom member that was never declared fails closed
    // rather than leaking an un-mapped private selector to the backend.
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{
            "type": "namespace",
            "name": "git",
            "description": "Git operations.",
            "tools": [{"type": "custom", "name": "freeform"}]
        }],
        "tool_choice": {"type": "custom", "name": "missing", "namespace": "git"},
    }));
    let action = filter()
        .lower_request(&mut state, false)
        .expect_err("an undeclared namespaced custom selector must fail closed");
    let (status, message) = reject_parts(&action);
    assert_eq!(status, 400, "an undeclared selector is a bad request");
    assert!(
        message.contains("missing"),
        "the rejection names the undeclared member: {message}"
    );
    assert_eq!(
        state.request_body["tool_choice"]["type"], "custom",
        "tool_choice left untouched on reject"
    );
}

#[test]
fn lowers_allowed_tools_selectors_recursively() {
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "custom", "name": "c"}],
        "tool_choice": {
            "type": "allowed_tools",
            "mode": "auto",
            "tools": [{"type": "custom", "name": "c"}]
        },
    }));
    filter().lower_request(&mut state, false).expect("lowering succeeds");
    assert_eq!(state.request_body["tool_choice"]["tools"][0]["type"], "function");
    assert_eq!(state.request_body["tool_choice"]["tools"][0]["name"], "c");
}

#[test]
fn tool_choice_for_undeclared_client_tool_is_rejected() {
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "custom", "name": "c"}],
        "tool_choice": {"type": "custom", "name": "other"},
    }));
    let action = filter().lower_request(&mut state, false).expect_err("must reject");
    let (status, message) = reject_parts(&action);
    assert_eq!(status, 400);
    assert!(
        message.contains("other"),
        "the rejection names the undeclared tool: {message}"
    );
    assert_eq!(
        state.request_body["tools"][0]["type"], "custom",
        "tools restored verbatim"
    );
    assert_eq!(
        state.request_body["tool_choice"]["type"], "custom",
        "tool_choice left untouched on reject"
    );
}

#[test]
fn tool_choice_auto_is_left_untouched() {
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "custom", "name": "c"}],
        "tool_choice": "auto",
    }));
    filter().lower_request(&mut state, false).expect("lowering succeeds");
    assert_eq!(state.request_body["tool_choice"], "auto");
}

// -----------------------------------------------------------------------------
// History lowering
// -----------------------------------------------------------------------------

#[test]
fn lowers_custom_tool_call_history_item() {
    let mut item = json!({
        "type": "custom_tool_call",
        "call_id": "call_1",
        "name": "run",
        "input": "echo hi",
        "id": "ctc_123",
        "status": "completed"
    });
    assert!(
        lower_history_item(&mut item, &HashSet::new()),
        "a custom_tool_call is always lowered"
    );
    assert_eq!(item["type"], "function_call");
    assert_eq!(item["name"], "run");
    assert_eq!(item["call_id"], "call_1");
    assert_eq!(item["arguments"], json!({"input": "echo hi"}).to_string());
    assert_eq!(item["id"], "fc_123");
    assert_eq!(item["status"], "completed");
}

#[test]
fn lowers_custom_tool_call_output_history_item() {
    let mut item = json!({
        "type": "custom_tool_call_output",
        "call_id": "call_1",
        "name": "run",
        "output": "result"
    });
    assert!(
        lower_history_item(&mut item, &HashSet::new()),
        "a custom_tool_call_output is always lowered"
    );
    assert_eq!(item["type"], "function_call_output");
    assert!(item.get("name").is_none(), "name is dropped");
    assert_eq!(item["output"], "result");
}

#[test]
fn lowers_client_executed_shell_call_history_item() {
    let mut item = json!({
        "type": "shell_call",
        "call_id": "call_1",
        "action": {"commands": ["ls"]},
        "environment": {"type": "local"},
        "id": "sh_9",
        "status": "completed"
    });
    assert!(
        lower_history_item(&mut item, &HashSet::new()),
        "a client-executed (local) shell_call is lowered"
    );
    assert_eq!(item["type"], "function_call");
    assert_eq!(item["name"], "shell");
    assert_eq!(item["arguments"], json!({"commands": ["ls"]}).to_string());
    assert_eq!(item["id"], "fc_9");
}

#[test]
fn server_owned_shell_call_history_item_is_left_untouched() {
    // A container/server-owned shell_call must keep its wire semantics across a
    // continuation turn rather than being relabelled as a client function call.
    let mut item = json!({
        "type": "shell_call",
        "call_id": "call_1",
        "action": {"commands": ["ls"]},
        "environment": {"type": "container_reference", "id": "c_1"},
        "id": "sh_9",
        "status": "completed"
    });
    assert!(
        !lower_history_item(&mut item, &HashSet::new()),
        "a non-local shell_call is not lowered"
    );
    assert_eq!(item["type"], "shell_call", "the server-owned call is unchanged");
}

#[test]
fn lowers_shell_call_output_when_its_call_was_lowered() {
    let mut item = json!({
        "type": "shell_call_output",
        "call_id": "call_1",
        "output": [{"stdout": "hi", "stderr": "", "outcome": {"type": "exit", "exit_code": 0}}]
    });
    assert!(
        lower_history_item(&mut item, &lowered_ids(&["call_1"])),
        "an output whose client-executed call was lowered is lowered too"
    );
    assert_eq!(item["type"], "function_call_output");
    assert_eq!(
        item["output"],
        json!([{"stdout": "hi", "stderr": "", "outcome": {"type": "exit", "exit_code": 0}}]).to_string()
    );
}

#[test]
fn server_owned_shell_call_output_history_item_is_left_untouched() {
    // Its matching call was server-owned (not in the lowered set), so the output
    // must keep its typed shape.
    let mut item = json!({
        "type": "shell_call_output",
        "call_id": "call_1",
        "output": [{"stdout": "hi", "stderr": "", "outcome": {"type": "exit", "exit_code": 0}}]
    });
    assert!(
        !lower_history_item(&mut item, &HashSet::new()),
        "an output with no lowered call is not lowered"
    );
    assert_eq!(
        item["type"], "shell_call_output",
        "the server-owned output is unchanged"
    );
}

#[test]
fn lowers_client_executed_tool_search_call_history_item() {
    let mut item = json!({
        "type": "tool_search_call",
        "call_id": "call_1",
        "execution": "client",
        "arguments": {"query": "foo"},
        "id": "tsc_1"
    });
    assert!(
        lower_history_item(&mut item, &HashSet::new()),
        "a client-executed tool_search_call is lowered"
    );
    assert_eq!(item["type"], "function_call");
    assert_eq!(item["name"], "tool_search");
    assert_eq!(item["arguments"], json!({"query": "foo"}).to_string());
    assert_eq!(item["id"], "tsc_1", "tool_search keeps its own id verbatim");
}

#[test]
fn server_executed_tool_search_call_history_item_is_left_untouched() {
    let mut item = json!({
        "type": "tool_search_call",
        "call_id": "call_1",
        "execution": "server",
        "arguments": {"query": "foo"},
        "id": "tsc_1"
    });
    assert!(
        !lower_history_item(&mut item, &HashSet::new()),
        "a server-executed tool_search_call is not lowered"
    );
    assert_eq!(item["type"], "tool_search_call", "the server-owned call is unchanged");
}

#[test]
fn lowers_tool_search_output_when_its_call_was_lowered() {
    let mut item = json!({
        "type": "tool_search_output",
        "call_id": "call_1",
        "tools": [{"name": "x"}]
    });
    assert!(
        lower_history_item(&mut item, &lowered_ids(&["call_1"])),
        "an output whose client-executed call was lowered is lowered too"
    );
    assert_eq!(item["type"], "function_call_output");
    assert_eq!(item["output"], json!([{"name": "x"}]).to_string());
}

#[test]
fn lowered_shell_call_output_preserves_id_status_and_caller() {
    // `FunctionShellCallOutput` carries `id`, a non-terminal `status`, and a
    // `caller`; all are representation-compatible with `FunctionToolCallOutput`, so
    // a continuation output must not reach the backend implicitly completed and
    // identity-less.
    // `ToolCallCaller` permits only `direct` or `program`; a `program` caller
    // carries a required `caller_id`, so preserving it end-to-end proves a full
    // schema-valid caller round-trips (not merely an arbitrary object).
    let mut item = json!({
        "type": "shell_call_output",
        "id": "sco_1",
        "call_id": "call_1",
        "status": "in_progress",
        "caller": {"type": "program", "caller_id": "prog_1"},
        "output": [{"stdout": "hi"}]
    });
    assert!(
        lower_history_item(&mut item, &lowered_ids(&["call_1"])),
        "a shell output whose client-executed call was lowered is lowered too"
    );
    assert_eq!(item["type"], "function_call_output");
    assert_eq!(item["id"], "sco_1", "the output id is preserved");
    assert_eq!(item["status"], "in_progress", "a non-terminal status is preserved");
    assert_eq!(
        item["caller"],
        json!({"type": "program", "caller_id": "prog_1"}),
        "a schema-valid program caller is preserved"
    );
}

#[test]
fn lowered_tool_search_output_preserves_id_and_status() {
    // `ToolSearchOutput` carries `id` and a `status`; both are preserved. It has no
    // `caller` field, so none is invented.
    let mut item = json!({
        "type": "tool_search_output",
        "id": "tso_1",
        "call_id": "call_1",
        "status": "incomplete",
        "tools": [{"name": "x"}]
    });
    assert!(
        lower_history_item(&mut item, &lowered_ids(&["call_1"])),
        "a tool_search output whose client-executed call was lowered is lowered too"
    );
    assert_eq!(item["type"], "function_call_output");
    assert_eq!(item["id"], "tso_1", "the output id is preserved");
    assert_eq!(item["status"], "incomplete", "a non-terminal status is preserved");
    assert!(
        item.get("caller").is_none(),
        "no caller field is invented for a tool_search output"
    );
}

#[test]
fn lowered_output_without_status_leaves_status_absent() {
    // An absent status must stay absent (not be forced to `completed`) so the
    // backend keeps its own default rather than an invented terminal status.
    let mut item = json!({
        "type": "shell_call_output",
        "call_id": "call_1",
        "output": [{"stdout": "hi"}]
    });
    assert!(
        lower_history_item(&mut item, &lowered_ids(&["call_1"])),
        "the output is lowered"
    );
    assert_eq!(item["type"], "function_call_output");
    assert!(
        item.get("status").is_none(),
        "an absent status is left absent, not forced to completed"
    );
    assert!(item.get("id").is_none(), "no id is invented when absent");
}

#[test]
fn lowered_output_with_non_enum_status_omits_status() {
    // A non-enum status is omitted rather than forwarded (matching call lowering),
    // so the backend never receives an invalid status string.
    let mut item = json!({
        "type": "tool_search_output",
        "call_id": "call_1",
        "status": "bogus",
        "tools": [{"name": "x"}]
    });
    assert!(
        lower_history_item(&mut item, &lowered_ids(&["call_1"])),
        "the output is lowered"
    );
    assert!(
        item.get("status").is_none(),
        "a non-enum status is omitted from the lowered output"
    );
}

#[test]
fn flattens_namespaced_function_call_history_item() {
    let mut item = json!({
        "type": "function_call",
        "call_id": "call_1",
        "name": "commit",
        "namespace": "git",
        "arguments": "{}"
    });
    assert!(
        lower_history_item(&mut item, &HashSet::new()),
        "a namespaced function_call is flattened"
    );
    assert_eq!(item["name"], "agentic_ns__git__commit");
    assert!(
        item.get("namespace").is_none(),
        "the namespace is folded into the flat name so the backend never sees it"
    );
}

#[test]
fn plain_function_call_history_item_is_unchanged() {
    let mut item = json!({"type": "function_call", "call_id": "call_1", "name": "f", "arguments": "{}"});
    assert!(
        !lower_history_item(&mut item, &HashSet::new()),
        "a plain function_call is not rewritten"
    );
    assert_eq!(item["name"], "f");
}

// -----------------------------------------------------------------------------
// Response restoration
// -----------------------------------------------------------------------------

/// Lower a single-custom-tool request and return the resulting state.
fn state_with_custom_lowered() -> ResponsesState {
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "custom", "name": "run_python", "description": "d", "format": {"type": "text"}}],
        "tool_choice": {"type": "custom", "name": "run_python"},
    }));
    filter().lower_request(&mut state, false).expect("lowering succeeds");
    state
}

#[test]
fn restores_custom_tool_call_and_echoes_tools() {
    let state = state_with_custom_lowered();
    let response = json!({
        "object": "response",
        "output": [{
            "type": "function_call",
            "id": "fc_abc",
            "call_id": "call_1",
            "name": "run_python",
            "arguments": "{\"input\":\"print(1)\"}",
            "status": "completed"
        }],
        "tools": [{"type": "function", "name": "run_python"}],
        "tool_choice": {"type": "function", "name": "run_python"}
    });
    let restored = restore(&state, &response);

    let item = &restored["output"][0];
    assert_eq!(item["type"], "custom_tool_call");
    assert_eq!(item["id"], "ctc_abc");
    assert_eq!(item["call_id"], "call_1");
    assert_eq!(item["name"], "run_python");
    assert_eq!(item["input"], "print(1)");
    assert!(
        item.get("status").is_none(),
        "custom_tool_call defines no status property; the backend status is dropped, not copied"
    );

    assert_eq!(restored["tools"][0]["type"], "custom", "tools restored from echo");
    assert_eq!(
        restored["tool_choice"]["type"], "custom",
        "tool_choice restored from echo"
    );
}

#[test]
fn restores_namespaced_function_call_in_place() {
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "namespace", "name": "git", "description": "Git operations.", "tools": [{"type": "function", "name": "commit"}]}],
    }));
    filter().lower_request(&mut state, false).expect("lowering succeeds");
    let response = json!({
        "object": "response",
        "output": [{
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "agentic_ns__git__commit",
            "arguments": "{}",
            "status": "completed"
        }],
    });
    let restored = restore(&state, &response);
    let item = &restored["output"][0];
    assert_eq!(item["type"], "function_call", "namespace members stay function calls");
    assert_eq!(item["name"], "commit", "original member name restored");
    assert_eq!(item["namespace"], "git", "namespace restored");
}

#[test]
fn restores_shell_call_with_local_environment() {
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "shell", "environment": {"type": "local"}}],
    }));
    filter().lower_request(&mut state, false).expect("lowering succeeds");
    let response = json!({
        "object": "response",
        "output": [{
            "type": "function_call",
            "id": "fc_x",
            "call_id": "call_1",
            "name": "shell",
            "arguments": "{\"commands\":[\"ls -la\"],\"timeout_ms\":1000}",
            "status": "completed"
        }],
    });
    let restored = restore(&state, &response);
    let item = &restored["output"][0];
    assert_eq!(item["type"], "shell_call");
    assert_eq!(item["id"], "sh_x");
    assert_eq!(item["action"]["commands"], json!(["ls -la"]));
    assert_eq!(item["action"]["timeout_ms"], 1000);
    assert!(
        item["action"].get("max_output_length").is_some(),
        "required-but-nullable action key is present"
    );
    assert!(
        item["action"]["max_output_length"].is_null(),
        "an omitted optional action value is normalized to explicit null"
    );
    assert_eq!(item["environment"]["type"], "local", "local environment recorded");
    assert_eq!(item["status"], "completed");
}

#[test]
fn restores_shell_call_normalizes_missing_action_fields() {
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "shell", "environment": {"type": "local"}}],
    }));
    filter().lower_request(&mut state, false).expect("lowering succeeds");
    let response = json!({
        "object": "response",
        "output": [{
            "type": "function_call",
            "id": "fc_x",
            "call_id": "call_1",
            "name": "shell",
            "arguments": "{\"commands\":[\"ls\"]}",
            "status": "completed"
        }],
    });
    let restored = restore(&state, &response);
    let action = &restored["output"][0]["action"];
    assert_eq!(action["commands"], json!(["ls"]));
    assert!(
        action["timeout_ms"].is_null() && action["max_output_length"].is_null(),
        "both required-but-nullable keys are normalized to null: {action}"
    );
}

#[test]
fn restores_incomplete_shell_call_preserves_status() {
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "shell", "environment": {"type": "local"}}],
    }));
    filter().lower_request(&mut state, false).expect("lowering succeeds");
    let response = json!({
        "object": "response",
        "output": [{
            "type": "function_call",
            "id": "fc_x",
            "call_id": "call_1",
            "name": "shell",
            "arguments": "{\"commands\":[\"ls\"]}",
            "status": "incomplete"
        }],
    });
    let restored = restore(&state, &response);
    assert_eq!(
        restored["output"][0]["status"], "incomplete",
        "an incomplete shell call must not be silently completed"
    );
}

#[test]
fn restores_shell_call_without_status_defaults_to_completed() {
    // A function-only backend (vLLM) may omit `status` on its `function_call`
    // outputs; Praxis already accepts status-less function calls elsewhere
    // (see agentic_loop). Restoration mirrors that: an absent status becomes the
    // spec-valid terminal `completed`, never a fail-closed rejection.
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "shell", "environment": {"type": "local"}}],
    }));
    filter().lower_request(&mut state, false).expect("lowering succeeds");
    let response = json!({
        "object": "response",
        "output": [{
            "type": "function_call",
            "id": "fc_x",
            "call_id": "call_1",
            "name": "shell",
            "arguments": "{\"commands\":[\"ls\"]}"
        }],
    });
    let restored = restore(&state, &response);
    let item = &restored["output"][0];
    assert_eq!(item["type"], "shell_call", "a status-less shell call still restores");
    assert_eq!(
        item["status"], "completed",
        "an absent status defaults to the terminal completed state"
    );
}

#[test]
fn shell_call_without_call_id_fails_closed() {
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "shell", "environment": {"type": "local"}}],
    }));
    filter().lower_request(&mut state, false).expect("lowering succeeds");
    let response = json!({
        "object": "response",
        "output": [{
            "type": "function_call",
            "id": "fc_x",
            "name": "shell",
            "arguments": "{\"commands\":[\"ls\"]}",
            "status": "completed"
        }],
    });
    let action = filter()
        .restore_response(&state, response.to_string().as_bytes())
        .expect_err("a shell_call without call_id must fail closed");
    let (status, message) = reject_parts(&action);
    assert_eq!(status, 502, "a schema-invalid shell_call cannot be restored");
    assert!(
        message.contains("shell_call"),
        "the rejection names shell_call: {message}"
    );
}

#[test]
fn shell_call_with_unknown_status_fails_closed() {
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "shell", "environment": {"type": "local"}}],
    }));
    filter().lower_request(&mut state, false).expect("lowering succeeds");
    let response = json!({
        "object": "response",
        "output": [{
            "type": "function_call",
            "id": "fc_x",
            "call_id": "call_1",
            "name": "shell",
            "arguments": "{\"commands\":[\"ls\"]}",
            "status": "bogus"
        }],
    });
    let action = filter()
        .restore_response(&state, response.to_string().as_bytes())
        .expect_err("an unknown status must fail closed");
    assert_eq!(reject_parts(&action).0, 502);
}

#[test]
fn shell_call_with_non_string_status_fails_closed() {
    // A non-string status is outside the `FunctionShellCallStatus` string domain;
    // it must fail closed rather than be silently coerced to `completed`.
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "shell", "environment": {"type": "local"}}],
    }));
    filter().lower_request(&mut state, false).expect("lowering succeeds");
    let response = json!({
        "object": "response",
        "output": [{
            "type": "function_call",
            "id": "fc_x",
            "call_id": "call_1",
            "name": "shell",
            "arguments": "{\"commands\":[\"ls\"]}",
            "status": 7
        }],
    });
    let action = filter()
        .restore_response(&state, response.to_string().as_bytes())
        .expect_err("a non-string status must fail closed");
    assert_eq!(reject_parts(&action).0, 502);
}

#[test]
fn shell_call_with_malformed_arguments_fails_closed() {
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "shell", "environment": {"type": "local"}}],
    }));
    filter().lower_request(&mut state, false).expect("lowering succeeds");
    let response = json!({
        "object": "response",
        "output": [{
            "type": "function_call",
            "id": "fc_x",
            "call_id": "call_1",
            "name": "shell",
            "arguments": "{not json}",
            "status": "completed"
        }],
    });
    let action = filter()
        .restore_response(&state, response.to_string().as_bytes())
        .expect_err("malformed shell args must fail closed");
    let (status, message) = reject_parts(&action);
    assert_eq!(status, 502);
    assert!(
        message.contains("shell_call"),
        "the rejection names the item that could not be restored: {message}"
    );
}

#[test]
fn shell_call_with_non_string_commands_fails_closed() {
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "shell", "environment": {"type": "local"}}],
    }));
    filter().lower_request(&mut state, false).expect("lowering succeeds");
    let response = json!({
        "object": "response",
        "output": [{
            "type": "function_call",
            "id": "fc_x",
            "call_id": "call_1",
            "name": "shell",
            "arguments": "{\"commands\":[1,2]}",
            "status": "completed"
        }],
    });
    let action = filter()
        .restore_response(&state, response.to_string().as_bytes())
        .expect_err("non-string commands must fail closed");
    assert_eq!(reject_parts(&action).0, 502);
}

#[test]
fn restores_tool_search_call() {
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "tool_search"}],
    }));
    filter().lower_request(&mut state, false).expect("lowering succeeds");
    let response = json!({
        "object": "response",
        "output": [{
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "tool_search",
            "arguments": "{\"query\":\"vector search\"}",
            "status": "completed"
        }],
    });
    let restored = restore(&state, &response);
    let item = &restored["output"][0];
    assert_eq!(item["type"], "tool_search_call");
    assert_eq!(item["id"], "tsc_1");
    assert_eq!(item["call_id"], "call_1");
    assert_eq!(item["execution"], "client");
    assert_eq!(item["arguments"]["query"], "vector search");
    assert_eq!(item["status"], "completed");
}

#[test]
fn restores_non_terminal_tool_search_call_preserves_status() {
    // `ToolSearchCall.status` is a required `FunctionCallStatus` whose domain is
    // {in_progress, completed, incomplete}; every valid value must survive
    // restoration verbatim rather than being coerced to `completed`.
    for status in ["in_progress", "incomplete"] {
        let mut state = ResponsesState::from_request_body(json!({
            "tools": [{"type": "tool_search"}],
        }));
        filter().lower_request(&mut state, false).expect("lowering succeeds");
        let response = json!({
            "object": "response",
            "output": [{
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "tool_search",
                "arguments": "{\"query\":\"x\"}",
                "status": status
            }],
        });
        let restored = restore(&state, &response);
        let item = &restored["output"][0];
        assert_eq!(item["type"], "tool_search_call", "restored to a tool_search_call");
        assert_eq!(
            item["status"], status,
            "the non-terminal status is preserved verbatim, not coerced to completed"
        );
    }
}

#[test]
fn tool_search_call_without_status_defaults_to_completed() {
    // An absent status is not a malformed call; it defaults to the terminal
    // `completed` state to tolerate a backend that omits it.
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "tool_search"}],
    }));
    filter().lower_request(&mut state, false).expect("lowering succeeds");
    let response = json!({
        "object": "response",
        "output": [{
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "tool_search",
            "arguments": "{\"query\":\"x\"}"
        }],
    });
    let restored = restore(&state, &response);
    let item = &restored["output"][0];
    assert_eq!(item["type"], "tool_search_call");
    assert_eq!(
        item["status"], "completed",
        "an absent status defaults to the terminal completed state"
    );
}

#[test]
fn tool_search_call_with_unknown_status_fails_closed() {
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "tool_search"}],
    }));
    filter().lower_request(&mut state, false).expect("lowering succeeds");
    let response = json!({
        "object": "response",
        "output": [{
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "tool_search",
            "arguments": "{\"query\":\"x\"}",
            "status": "bogus"
        }],
    });
    let action = filter()
        .restore_response(&state, response.to_string().as_bytes())
        .expect_err("an unknown status must fail closed");
    let (status, message) = reject_parts(&action);
    assert_eq!(status, 502);
    assert!(
        message.contains("tool_search_call"),
        "the rejection names the item that could not be restored: {message}"
    );
}

#[test]
fn tool_search_call_with_non_string_status_fails_closed() {
    // A non-string status is outside the `FunctionCallStatus` string domain and
    // must fail closed rather than be silently coerced to `completed`.
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "tool_search"}],
    }));
    filter().lower_request(&mut state, false).expect("lowering succeeds");
    let response = json!({
        "object": "response",
        "output": [{
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "tool_search",
            "arguments": "{\"query\":\"x\"}",
            "status": 7
        }],
    });
    let action = filter()
        .restore_response(&state, response.to_string().as_bytes())
        .expect_err("a non-string status must fail closed");
    assert_eq!(reject_parts(&action).0, 502);
}

#[test]
fn tool_search_call_with_malformed_arguments_fails_closed() {
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "tool_search"}],
    }));
    filter().lower_request(&mut state, false).expect("lowering succeeds");
    let response = json!({
        "object": "response",
        "output": [{
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "tool_search",
            "arguments": "not json",
            "status": "completed"
        }],
    });
    let action = filter()
        .restore_response(&state, response.to_string().as_bytes())
        .expect_err("malformed tool_search args must fail closed");
    assert_eq!(reject_parts(&action).0, 502);
}

#[test]
fn unrelated_function_calls_are_left_untouched() {
    let state = state_with_custom_lowered();
    let response = json!({
        "object": "response",
        "output": [{
            "type": "function_call",
            "id": "fc_other",
            "call_id": "call_2",
            "name": "some_other_function",
            "arguments": "{}",
            "status": "completed"
        }],
    });
    let restored = restore(&state, &response);
    assert_eq!(
        restored["output"][0]["type"], "function_call",
        "a call not in the reverse map is unchanged"
    );
    assert_eq!(restored["output"][0]["name"], "some_other_function");
}

#[test]
fn non_response_body_is_not_rewritten() {
    let state = state_with_custom_lowered();
    let sse = b"event: response.created\ndata: {}\n\n";
    let result = filter()
        .restore_response(&state, sse)
        .expect("streaming bodies are left for stream_events");
    assert!(result.is_none(), "an SSE body is not restored here");
}

#[test]
fn error_body_is_not_rewritten() {
    let state = state_with_custom_lowered();
    let error = json!({"object": "error", "error": {"message": "boom"}});
    let result = filter()
        .restore_response(&state, error.to_string().as_bytes())
        .expect("error bodies pass through");
    assert!(result.is_none(), "a non-response object is not restored");
}

#[test]
fn native_response_without_echo_is_not_rewritten() {
    let state = ResponsesState::from_request_body(json!({"model": "m", "input": "hi"}));
    let response = json!({
        "object": "response",
        "output": [{"type": "message", "role": "assistant", "content": []}],
    });
    let result = filter()
        .restore_response(&state, response.to_string().as_bytes())
        .expect("native responses pass through");
    assert!(result.is_none(), "no echo means no restoration");
}

#[test]
fn oversized_restored_body_is_rejected() {
    let tiny = ClientToolCompatFilter {
        max_rewritten_body_bytes: 32,
        max_client_tools: 512,
    };
    let state = state_with_custom_lowered();
    let response = json!({
        "object": "response",
        "output": [{
            "type": "function_call",
            "id": "fc_abc",
            "call_id": "call_1",
            "name": "run_python",
            "arguments": "{\"input\":\"print(1)\"}",
            "status": "completed"
        }],
    });
    let action = tiny
        .restore_response(&state, response.to_string().as_bytes())
        .expect_err("a body over the cap must be rejected");
    assert_eq!(reject_parts(&action).0, 413);
}

// -----------------------------------------------------------------------------
// Code-review follow-up coverage (#1131)
// -----------------------------------------------------------------------------

#[test]
fn local_shell_declaration_is_rejected() {
    // `local_shell` is a distinct client-owned declaration this filter cannot yet
    // restore; it fails closed before any upstream call and trips the streaming
    // guard, rather than being forwarded to a function-only backend.
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "local_shell"}],
    }));
    assert!(
        request_has_rich_client_tool(&state),
        "local_shell is a rich client tool the streaming guard must catch"
    );
    let action = filter()
        .lower_request(&mut state, false)
        .expect_err("local_shell must fail closed");
    let (status, message) = reject_parts(&action);
    assert_eq!(status, 400);
    assert!(
        message.contains("local_shell"),
        "the rejection names local_shell: {message}"
    );
    assert_eq!(
        state.request_body["tools"][0]["type"], "local_shell",
        "tools restored verbatim on reject"
    );
    assert!(state.client_tool_echo.is_none(), "no echo on a rejected request");
}

#[test]
fn lowers_namespace_custom_member_to_flat_function() {
    // #1158 requires flattening both function and custom namespace members.
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{
            "type": "namespace",
            "name": "git",
            "description": "Git version control operations.",
            "tools": [
                {"type": "function", "name": "commit", "parameters": {"type": "object"}},
                {"type": "custom", "name": "freeform", "description": "Freeform patch input."}
            ]
        }],
    }));
    filter().lower_request(&mut state, false).expect("lowering succeeds");

    let tools = state.request_body["tools"].as_array().expect("tools array");
    assert_eq!(tools.len(), 2, "both members lowered: {tools:?}");
    let custom_fn = tools
        .iter()
        .find(|tool| tool["name"] == "agentic_ns__git__freeform")
        .expect("custom member flattened to a function");
    assert_eq!(custom_fn["type"], "function", "custom member lowered to a function");
    assert_eq!(
        custom_fn["parameters"]["required"],
        json!(["input"]),
        "the custom member keeps the single freeform input contract: {custom_fn}"
    );

    let lowered = state
        .client_tool_lowering
        .get("agentic_ns__git__freeform")
        .expect("reverse entry for the custom member");
    assert_eq!(
        lowered.restore,
        ClientToolRestore::NamespaceCustom,
        "the custom member restores to a namespaced custom_tool_call"
    );
    assert_eq!(lowered.original_name, "freeform", "the bare member name is recorded");
    assert_eq!(lowered.namespace.as_deref(), Some("git"), "the namespace is recorded");
}

#[test]
fn restores_namespace_custom_member_call() {
    let reverse = HashMap::from([(
        "agentic_ns__git__freeform".to_owned(),
        LoweredClientTool {
            original_name: "freeform".to_owned(),
            namespace: Some("git".to_owned()),
            restore: ClientToolRestore::NamespaceCustom,
        },
    )]);
    let mut item = json!({
        "type": "function_call",
        "id": "fc_xyz",
        "call_id": "call_xyz",
        "name": "agentic_ns__git__freeform",
        "arguments": r#"{"input":"*** Begin Patch"}"#,
        "status": "completed"
    });
    restore_output_item(&mut item, &reverse).expect("restoration succeeds");
    assert_eq!(item["type"], "custom_tool_call", "restored to a namespaced custom call");
    assert_eq!(item["name"], "freeform", "member name recovered");
    assert_eq!(item["namespace"], "git", "namespace recovered onto the custom call");
    assert_eq!(item["call_id"], "call_xyz", "call_id preserved");
    assert_eq!(item["id"], "ctc_xyz", "public custom item id derived");
    assert_eq!(
        item["input"], "*** Begin Patch",
        "the freeform input is unwrapped from the lowered arguments: {item}"
    );
    assert!(
        item.get("status").is_none(),
        "a namespaced custom_tool_call defines no status property; the backend status is dropped, not copied"
    );
}

#[test]
fn flattens_namespaced_custom_call_history_item() {
    // A namespaced custom_tool_call in history flattens to the same private member
    // name it was lowered to, so a continuation turn correlates with the backend.
    let mut item = json!({
        "type": "custom_tool_call",
        "id": "ctc_hist",
        "call_id": "call_hist",
        "namespace": "git",
        "name": "freeform",
        "input": "*** Begin Patch",
        "status": "completed"
    });
    assert!(
        lower_history_item(&mut item, &HashSet::new()),
        "a namespaced custom history call is lowered"
    );
    assert_eq!(item["type"], "function_call", "lowered to a function_call");
    assert_eq!(
        item["name"], "agentic_ns__git__freeform",
        "flattened to the private member name: {item}"
    );
    assert_eq!(item["call_id"], "call_hist", "call_id preserved");
    assert_eq!(item["id"], "fc_hist", "item id translated to the function prefix");
    assert!(
        item.get("namespace").is_none(),
        "the namespace field is dropped from the lowered function_call: {item}"
    );
}

#[test]
fn namespace_with_local_shell_member_is_rejected() {
    // A `local_shell` member is neither function nor custom, so it must fail closed.
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{
            "type": "namespace",
            "name": "git",
            "description": "Git operations.",
            "tools": [{"type": "local_shell"}]
        }],
    }));
    let action = filter()
        .lower_request(&mut state, false)
        .expect_err("an unsupported namespace member must fail closed");
    let (status, message) = reject_parts(&action);
    assert_eq!(status, 400, "an unsupported member is a bad request");
    assert!(
        message.contains("local_shell") && message.contains("git"),
        "the rejection names the unsupported member and namespace: {message}"
    );
    assert_eq!(
        state.request_body["tools"][0]["type"], "namespace",
        "tools restored verbatim on reject"
    );
}

#[test]
fn deferred_namespace_custom_member_is_withheld() {
    // A deferred `custom` namespace member is not callable until a `tool_search`
    // loads it, so it is withheld from the outbound set on the declaration path
    // rather than rejected — a namespace member is not exempt from `defer_loading`.
    // When it is the namespace's only member the outbound callable set is empty and
    // the client's originals are echoed back on the response.
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{
            "type": "namespace",
            "name": "git",
            "description": "Git operations.",
            "tools": [{"type": "custom", "name": "freeform", "defer_loading": true}]
        }],
    }));
    filter().lower_request(&mut state, false).expect("lowering succeeds");
    assert_eq!(
        state.request_body["tools"],
        json!([]),
        "the deferred custom member is withheld from the outbound callable set"
    );
    let echo = state
        .client_tool_echo
        .as_ref()
        .expect("the client originals are echoed back on the response");
    assert_eq!(
        echo.tools[0]["type"], "namespace",
        "the client's original namespace is echoed back on the response"
    );
    assert!(
        state.client_tool_lowering.is_empty(),
        "nothing was lowered, so there is no restoration recipe"
    );
}

#[test]
fn tool_search_with_null_parameters_uses_default_schema() {
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "tool_search", "parameters": null}],
    }));
    filter()
        .lower_request(&mut state, false)
        .expect("an explicit null parameters lowers like omission");
    let tools = state.request_body["tools"].as_array().expect("tools array");
    assert_eq!(tools[0]["type"], "function");
    assert_eq!(
        tools[0]["parameters"]["properties"]["query"]["description"],
        "A concise description of the needed capabilities.",
        "null parameters lowers to the default query schema"
    );
}

#[test]
fn tool_search_with_non_object_parameters_is_rejected() {
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "tool_search", "parameters": 5}],
    }));
    let action = filter()
        .lower_request(&mut state, false)
        .expect_err("non-object, non-null parameters must fail closed");
    let (status, message) = reject_parts(&action);
    assert_eq!(status, 400);
    assert!(
        message.contains("tool_search parameters"),
        "the rejection explains the constraint: {message}"
    );
}

#[test]
fn lowers_shell_tool_choice_to_function() {
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "shell", "environment": {"type": "local"}}],
        "tool_choice": {"type": "shell"},
    }));
    filter().lower_request(&mut state, false).expect("lowering succeeds");
    assert_eq!(
        state.request_body["tool_choice"]["type"], "function",
        "a forced shell choice becomes a function selector"
    );
    assert_eq!(state.request_body["tool_choice"]["name"], "shell");
}

#[test]
fn shell_tool_choice_without_lowered_shell_is_untouched() {
    // A `{"type":"shell"}` selector with no local shell tool lowered this request
    // (a native shell this filter did not rewrite) is left as-is.
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "custom", "name": "c"}],
        "tool_choice": {"type": "shell"},
    }));
    filter().lower_request(&mut state, false).expect("lowering succeeds");
    assert_eq!(
        state.request_body["tool_choice"]["type"], "shell",
        "a shell selector with no lowered shell tool is unchanged"
    );
}

#[test]
fn lowers_shell_selector_inside_allowed_tools() {
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "shell", "environment": {"type": "local"}}],
        "tool_choice": {
            "type": "allowed_tools",
            "mode": "auto",
            "tools": [{"type": "shell"}]
        },
    }));
    filter().lower_request(&mut state, false).expect("lowering succeeds");
    assert_eq!(
        state.request_body["tool_choice"]["tools"][0]["type"], "function",
        "a shell selector nested in allowed_tools is lowered recursively"
    );
    assert_eq!(state.request_body["tool_choice"]["tools"][0]["name"], "shell");
}

#[test]
fn custom_with_allowed_callers_is_rejected() {
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "custom", "name": "c", "allowed_callers": ["programmatic"]}],
    }));
    let action = filter()
        .lower_request(&mut state, false)
        .expect_err("an unenforceable caller restriction must fail closed");
    let (status, message) = reject_parts(&action);
    assert_eq!(status, 400);
    assert!(
        message.contains("allowed_callers"),
        "the rejection names the restriction: {message}"
    );
    assert_eq!(
        state.request_body["tools"][0]["type"], "custom",
        "tools restored verbatim on reject"
    );
}

#[test]
fn shell_with_allowed_callers_is_rejected() {
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "shell", "environment": {"type": "local"}, "allowed_callers": ["direct"]}],
    }));
    let action = filter()
        .lower_request(&mut state, false)
        .expect_err("an unenforceable caller restriction must fail closed");
    assert_eq!(reject_parts(&action).0, 400);
}

#[test]
fn namespace_member_with_allowed_callers_is_rejected() {
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{
            "type": "namespace",
            "name": "git",
            "description": "Git operations.",
            "tools": [{"type": "function", "name": "commit", "allowed_callers": ["programmatic"]}]
        }],
    }));
    let action = filter()
        .lower_request(&mut state, false)
        .expect_err("an unenforceable caller restriction must fail closed");
    assert_eq!(reject_parts(&action).0, 400);
}

#[test]
fn custom_with_null_allowed_callers_lowers_and_omits_prose() {
    // A null caller restriction is permitted, and `allowed_callers` is never
    // reduced to unenforced prose in the model-visible description.
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "custom", "name": "c", "allowed_callers": null}],
    }));
    filter()
        .lower_request(&mut state, false)
        .expect("a null caller restriction is permitted");
    let description = state.request_body["tools"][0]["description"]
        .as_str()
        .expect("description");
    assert!(
        !description.contains("allowed_callers"),
        "allowed_callers must never appear as prose: {description}"
    );
}

#[test]
fn oversized_lowered_request_is_rejected_and_rolled_back() {
    let tiny = ClientToolCompatFilter {
        max_rewritten_body_bytes: 16,
        max_client_tools: 512,
    };
    let mut state = ResponsesState::from_request_body(json!({
        "model": "m",
        "input": "hi",
        "tools": [{"type": "custom", "name": "run_python", "description": "x".repeat(200)}],
        "tool_choice": {"type": "custom", "name": "run_python"},
    }));
    let action = tiny
        .lower_request(&mut state, false)
        .expect_err("a rewrite over the configured cap must fail closed");
    assert_eq!(
        reject_parts(&action).0,
        413,
        "an over-cap rewritten request body is rejected"
    );
    assert_eq!(
        state.request_body["tools"][0]["type"], "custom",
        "tools rolled back to the client originals"
    );
    assert_eq!(
        state.request_body["tool_choice"]["type"], "custom",
        "tool_choice rolled back to the client original"
    );
    assert!(state.client_tool_echo.is_none(), "no echo on a rejected request");
    assert!(
        !state.request_body_requires_rebuild(),
        "no rebuild on a rejected request"
    );
}

#[test]
fn lower_history_items_lowers_only_client_owned_calls() {
    // A local shell_call and its output are lowered together; a container-owned
    // call and its output keep their wire semantics across the continuation turn.
    let mut messages = vec![
        json!({
            "type": "shell_call", "call_id": "local_1", "action": {"commands": ["ls"]},
            "environment": {"type": "local"}, "id": "sh_1", "status": "completed"
        }),
        json!({"type": "shell_call_output", "call_id": "local_1", "output": "ok"}),
        json!({
            "type": "shell_call", "call_id": "cont_1", "action": {"commands": ["ls"]},
            "environment": {"type": "container_reference", "id": "c_1"}, "id": "sh_2", "status": "completed"
        }),
        json!({"type": "shell_call_output", "call_id": "cont_1", "output": "ok"}),
    ];
    assert!(
        lower_history_items(&mut messages).expect("supported history lowers without error"),
        "at least one client-owned item is lowered"
    );
    assert_eq!(messages[0]["type"], "function_call", "the local shell_call is lowered");
    assert_eq!(
        messages[1]["type"], "function_call_output",
        "its output is lowered in lockstep"
    );
    assert_eq!(
        messages[2]["type"], "shell_call",
        "the container shell_call is left untouched"
    );
    assert_eq!(
        messages[3]["type"], "shell_call_output",
        "the container output is left untouched"
    );
}

#[test]
fn local_shell_call_history_item_is_rejected() {
    // A `local_shell` declaration fails closed on the request; its typed history
    // item must fail closed too, so a prior turn's unsupported call cannot reach a
    // function-only backend unchanged on a continuation.
    let mut messages = vec![json!({
        "type": "local_shell_call",
        "call_id": "call_1",
        "id": "lsh_1",
        "action": {"type": "exec", "command": ["ls"]},
        "status": "completed"
    })];
    let action = lower_history_items(&mut messages).expect_err("a prior local_shell_call must fail closed");
    let (status, message) = reject_parts(&action);
    assert_eq!(status, 400, "an unsupported history item is a bad request");
    assert!(
        message.contains("local_shell"),
        "the rejection names the unsupported history item: {message}"
    );
}

#[test]
fn local_shell_call_output_history_item_is_rejected() {
    // The output half of an unsupported `local_shell` call is rejected on the same
    // grounds, even when the matching call is not present in this window.
    let mut messages = vec![json!({
        "type": "local_shell_call_output",
        "call_id": "call_1",
        "output": "listing"
    })];
    let action = lower_history_items(&mut messages).expect_err("a prior local_shell_call_output must fail closed");
    let (status, message) = reject_parts(&action);
    assert_eq!(status, 400, "an unsupported history item is a bad request");
    assert!(
        message.contains("local_shell"),
        "the rejection names the unsupported history item: {message}"
    );
}

// -----------------------------------------------------------------------------
// History-only rewrite cap (#1158 review F1)
// -----------------------------------------------------------------------------

#[test]
fn history_only_rewrite_over_cap_fails_closed() {
    // A continuation with NO rich tool declarations still lowers typed history;
    // escaping a large custom_tool_call input can expand the outbound body past a
    // configured cap smaller than the proxy's own limit. The history-only path must
    // enforce the same rewrite cap and fail closed before any upstream call.
    let tiny = ClientToolCompatFilter {
        max_rewritten_body_bytes: 64,
        max_client_tools: 512,
    };
    let mut state = ResponsesState::from_request_body(json!({
        "model": "m",
        "input": [{
            "type": "custom_tool_call",
            "call_id": "call_1",
            "name": "apply_patch",
            "input": "x".repeat(4096),
            "status": "completed"
        }],
    }));
    let action = tiny
        .lower_request(&mut state, false)
        .expect_err("a history-only rewrite over the cap must fail closed");
    assert_eq!(
        reject_parts(&action).0,
        413,
        "an over-cap history-only rewrite is rejected before any upstream call"
    );
}

#[test]
fn history_only_rewrite_within_cap_lowers_and_marks_rebuild() {
    // The positive companion: a history-only rewrite under the cap lowers the typed
    // client-owned item and marks the body for rebuild without rejecting.
    let mut state = ResponsesState::from_request_body(json!({
        "model": "m",
        "input": [{
            "type": "custom_tool_call",
            "call_id": "call_1",
            "name": "apply_patch",
            "input": "small",
            "status": "completed"
        }],
    }));
    filter()
        .lower_request(&mut state, false)
        .expect("a history-only rewrite under the cap succeeds");
    assert_eq!(
        state.messages[0]["type"], "function_call",
        "the typed client-owned history item is lowered"
    );
    assert!(
        state.request_body_requires_rebuild(),
        "lowered history marks the body for rebuild"
    );
}

// -----------------------------------------------------------------------------
// Namespace description preservation (#1158 review F2)
// -----------------------------------------------------------------------------

#[test]
fn namespace_description_is_folded_into_flattened_members() {
    // A flattened member loses its `namespace` grouping on the wire, so the
    // namespace's required model-visible description is folded into each member.
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{
            "type": "namespace",
            "name": "git",
            "description": "Version control operations.",
            "tools": [
                {"type": "function", "name": "commit", "description": "Record changes.", "parameters": {"type": "object"}},
                {"type": "custom", "name": "freeform", "description": "Freeform patch."}
            ]
        }],
    }));
    filter().lower_request(&mut state, false).expect("lowering succeeds");
    let tools = state.request_body["tools"].as_array().expect("tools array");

    let function_member = tools
        .iter()
        .find(|tool| tool["name"] == "agentic_ns__git__commit")
        .expect("function member");
    let function_description = function_member["description"].as_str().expect("description");
    assert!(
        function_description.contains("Version control operations."),
        "the namespace description is folded into the function member: {function_description}"
    );
    assert!(
        function_description.contains("Record changes."),
        "the member's own description is retained: {function_description}"
    );
    assert!(
        function_description.contains("git"),
        "the namespace name is present so the model keeps the grouping: {function_description}"
    );

    let custom_member = tools
        .iter()
        .find(|tool| tool["name"] == "agentic_ns__git__freeform")
        .expect("custom member");
    let custom_description = custom_member["description"].as_str().expect("description");
    assert!(
        custom_description.contains("Version control operations."),
        "the namespace description is folded into the custom member: {custom_description}"
    );
    assert!(
        custom_description.contains("Freeform patch."),
        "the custom member's own description is retained: {custom_description}"
    );
}

#[test]
fn namespace_without_description_is_rejected() {
    // `NamespaceToolParam.description` is required (`minLength: 1`). A flattened
    // member cannot recover the namespace grouping without it, so a namespace that
    // omits its model-visible description fails closed rather than dropping it.
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{
            "type": "namespace",
            "name": "git",
            "tools": [{"type": "function", "name": "commit"}]
        }],
    }));
    let action = filter()
        .lower_request(&mut state, false)
        .expect_err("a namespace without a description must fail closed");
    let (status, message) = reject_parts(&action);
    assert_eq!(status, 400, "a namespace without a description is a bad request");
    assert!(
        message.contains("description"),
        "the rejection explains the missing description: {message}"
    );
    assert_eq!(
        state.request_body["tools"][0]["type"], "namespace",
        "tools restored verbatim on reject"
    );
}

#[test]
fn namespace_with_blank_description_is_rejected() {
    // A whitespace-only description is not model-visible content and is treated the
    // same as an omitted one.
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{
            "type": "namespace",
            "name": "git",
            "description": "   ",
            "tools": [{"type": "function", "name": "commit"}]
        }],
    }));
    let action = filter()
        .lower_request(&mut state, false)
        .expect_err("a blank namespace description must fail closed");
    assert_eq!(reject_parts(&action).0, 400);
}

// -----------------------------------------------------------------------------
// Local-shell skills preservation (#1158 review F3)
// -----------------------------------------------------------------------------

#[test]
fn shell_environment_skills_are_folded_into_description() {
    // `LocalEnvironmentParam.skills` is valid schema data; a Codex-declared skill's
    // name/description/path must reach the model in the lowered shell function.
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{
            "type": "shell",
            "environment": {
                "type": "local",
                "skills": [
                    {"name": "deploy", "description": "Ship the service.", "path": "/skills/deploy"},
                    {"name": "lint", "description": "Run the linters.", "path": "/skills/lint"}
                ]
            }
        }],
    }));
    filter().lower_request(&mut state, false).expect("lowering succeeds");
    let description = state.request_body["tools"][0]["description"]
        .as_str()
        .expect("description");
    assert!(
        description.contains("deploy") && description.contains("Ship the service."),
        "the first skill's name and description reach the model: {description}"
    );
    assert!(
        description.contains("/skills/deploy"),
        "the skill path reaches the model: {description}"
    );
    assert!(
        description.contains("lint") && description.contains("Run the linters."),
        "the second skill reaches the model: {description}"
    );
    assert_eq!(
        state.request_body["tools"][0]["name"], "shell",
        "the tool is still lowered to the private shell function"
    );
}

#[test]
fn shell_skill_missing_name_is_rejected() {
    // `LocalSkillParam.name` is a required string; because the filter consumes the
    // declaration the backend can no longer validate it, so a missing name fails
    // closed rather than being silently erased.
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{
            "type": "shell",
            "environment": {
                "type": "local",
                "skills": [{"description": "No name here.", "path": "/skills/x"}]
            }
        }],
    }));
    let action = filter()
        .lower_request(&mut state, false)
        .expect_err("a skill without a name must fail closed");
    let (status, message) = reject_parts(&action);
    assert_eq!(status, 400, "a skill missing a required field is a bad request");
    assert!(
        message.contains("name"),
        "the rejection names the missing field: {message}"
    );
    assert_eq!(
        state.request_body["tools"][0]["type"], "shell",
        "tools restored verbatim on reject"
    );
}

#[test]
fn shell_skill_missing_description_is_rejected() {
    // `LocalSkillParam.description` is a required string.
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{
            "type": "shell",
            "environment": {
                "type": "local",
                "skills": [{"name": "deploy", "path": "/skills/deploy"}]
            }
        }],
    }));
    let action = filter()
        .lower_request(&mut state, false)
        .expect_err("a skill without a description must fail closed");
    let (status, message) = reject_parts(&action);
    assert_eq!(status, 400, "a skill missing a required field is a bad request");
    assert!(
        message.contains("description"),
        "the rejection names the missing field: {message}"
    );
}

#[test]
fn shell_skill_missing_path_is_rejected() {
    // `LocalSkillParam.path` is a required string.
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{
            "type": "shell",
            "environment": {
                "type": "local",
                "skills": [{"name": "deploy", "description": "Ship the service."}]
            }
        }],
    }));
    let action = filter()
        .lower_request(&mut state, false)
        .expect_err("a skill without a path must fail closed");
    let (status, message) = reject_parts(&action);
    assert_eq!(status, 400, "a skill missing a required field is a bad request");
    assert!(
        message.contains("path"),
        "the rejection names the missing field: {message}"
    );
}

#[test]
fn shell_skill_with_non_string_field_is_rejected() {
    // A required field present with the wrong type is schema-invalid and fails
    // closed rather than being silently coerced or erased.
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{
            "type": "shell",
            "environment": {
                "type": "local",
                "skills": [{"name": 7, "description": "d", "path": "/p"}]
            }
        }],
    }));
    let action = filter()
        .lower_request(&mut state, false)
        .expect_err("a non-string skill field must fail closed");
    let (status, message) = reject_parts(&action);
    assert_eq!(status, 400, "a non-string skill field is a bad request");
    assert!(
        message.contains("name"),
        "the rejection names the offending field: {message}"
    );
}

#[test]
fn shell_skills_non_array_is_rejected() {
    // `environment.skills` is an array; a non-array value must not be silently
    // treated as absent now that the backend can no longer validate it.
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{
            "type": "shell",
            "environment": {"type": "local", "skills": "deploy"}
        }],
    }));
    let action = filter()
        .lower_request(&mut state, false)
        .expect_err("a non-array skills value must fail closed");
    let (status, message) = reject_parts(&action);
    assert_eq!(status, 400, "a non-array skills value is a bad request");
    assert!(
        message.contains("skills") && message.contains("array"),
        "the rejection explains the array requirement: {message}"
    );
}

#[test]
fn shell_skills_null_fails_closed() {
    // `LocalEnvironmentParam.skills` is a non-nullable array, so an explicit null
    // is a schema violation and fails closed rather than being silently treated as
    // "no skills" (which would drop a malformed declaration).
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{
            "type": "shell",
            "environment": {"type": "local", "skills": null}
        }],
    }));
    let action = filter()
        .lower_request(&mut state, false)
        .expect_err("a null skills value must fail closed");
    let (status, message) = reject_parts(&action);
    assert_eq!(status, 400, "a null skills value is a bad request");
    assert!(
        message.contains("skills") && message.contains("array"),
        "the rejection explains the array requirement: {message}"
    );
}

#[test]
fn shell_skill_with_empty_strings_is_accepted() {
    // `LocalSkillParam` has no `minLength`, so schema-valid empty strings are
    // accepted (and simply not rendered) rather than imposing a stricter constraint.
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{
            "type": "shell",
            "environment": {
                "type": "local",
                "skills": [{"name": "", "description": "", "path": ""}]
            }
        }],
    }));
    filter()
        .lower_request(&mut state, false)
        .expect("a schema-valid empty-string skill is accepted");
    assert_eq!(
        state.request_body["tools"][0]["name"], "shell",
        "the shell tool is still lowered to the private function"
    );
}

#[test]
fn shell_skills_at_max_are_accepted_but_over_max_are_rejected() {
    // `LocalEnvironmentParam.skills` has `maxItems: 200`.
    let skill = |index: usize| json!({"name": format!("s{index}"), "description": "d", "path": "/p"});
    let at_max: Vec<_> = (0..200).map(skill).collect();
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "shell", "environment": {"type": "local", "skills": at_max}}],
    }));
    filter()
        .lower_request(&mut state, false)
        .expect("exactly 200 skills are within the cap");

    let over_max: Vec<_> = (0..201).map(skill).collect();
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "shell", "environment": {"type": "local", "skills": over_max}}],
    }));
    let action = filter()
        .lower_request(&mut state, false)
        .expect_err("more than 200 skills must fail closed");
    let (status, message) = reject_parts(&action);
    assert_eq!(status, 400, "exceeding the skills cap is a bad request");
    assert!(message.contains("200"), "the rejection states the maximum: {message}");
}

#[test]
fn shell_without_skills_uses_base_description() {
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "shell", "environment": {"type": "local"}}],
    }));
    filter().lower_request(&mut state, false).expect("lowering succeeds");
    assert_eq!(
        state.request_body["tools"][0]["description"], SHELL_BASE_DESCRIPTION,
        "a shell with no skills keeps the base description unchanged"
    );
}

// -----------------------------------------------------------------------------
// Tool-search discovered tools become callable (#1158 review F3)
// -----------------------------------------------------------------------------

/// Build a continuation request whose history holds a client-executed
/// `tool_search` round returning `discovered_tools`, optionally re-declaring
/// `tool_search`.
fn state_with_discovered_tools(discovered_tools: &Value, declare_tool_search: bool) -> ResponsesState {
    let tools = if declare_tool_search {
        json!([{"type": "tool_search"}])
    } else {
        json!([])
    };
    ResponsesState::from_request_body(json!({
        "tools": tools,
        "input": [
            {
                "type": "tool_search_call",
                "call_id": "call_ts",
                "execution": "client",
                "arguments": {"query": "patch"}
            },
            {
                "type": "tool_search_output",
                "call_id": "call_ts",
                "status": "completed",
                "tools": discovered_tools.clone()
            }
        ],
    }))
}

#[test]
fn discovered_custom_tool_is_hoisted_and_restored() {
    // A client `tool_search` returned a `custom` tool; a function-only backend can
    // only invoke it if the discovered definition is lowered into the outbound tool
    // set, and its returned `function_call` must restore to a `custom_tool_call`.
    let mut state = state_with_discovered_tools(
        &json!([{"type": "custom", "name": "apply_patch", "description": "Apply a patch.", "format": {"type": "text"}}]),
        true,
    );
    filter().lower_request(&mut state, false).expect("lowering succeeds");

    let tools = state.request_body["tools"].as_array().expect("outbound tools");
    let hoisted = tools
        .iter()
        .find(|tool| tool["name"] == "apply_patch")
        .expect("the discovered custom tool is hoisted into the outbound tools");
    assert_eq!(
        hoisted["type"], "function",
        "the discovered custom tool is lowered to a private function"
    );
    assert!(
        state.client_tool_lowering.contains_key("apply_patch"),
        "the discovered tool is registered for restoration"
    );

    // The backend calls the discovered tool; it must restore to a custom_tool_call.
    let response = json!({
        "object": "response",
        "output": [{
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "apply_patch",
            "arguments": "{\"input\":\"diff\"}",
            "status": "completed"
        }],
    });
    let restored = restore(&state, &response);
    let item = &restored["output"][0];
    assert_eq!(item["type"], "custom_tool_call", "the discovered call is restored");
    assert_eq!(item["name"], "apply_patch");
    assert_eq!(item["input"], "diff");
    assert!(
        restored["tools"]
            .as_array()
            .expect("echoed tools")
            .iter()
            .all(|tool| tool["name"] != "apply_patch"),
        "the discovered tool is not echoed into the response tools (it was never declared)"
    );
}

#[test]
fn discovered_tool_hoisted_without_declared_rich_tool() {
    // The continuation need not re-declare `tool_search`: a prior client-executed
    // search's discovered tool is still hoisted so it stays callable.
    let mut state = state_with_discovered_tools(
        &json!([{"type": "custom", "name": "apply_patch", "description": "Apply a patch.", "format": {"type": "text"}}]),
        false,
    );
    filter().lower_request(&mut state, false).expect("lowering succeeds");
    let tools = state.request_body["tools"].as_array().expect("outbound tools");
    assert!(
        tools.iter().any(|tool| tool["name"] == "apply_patch"),
        "the discovered tool is hoisted even without a re-declared tool_search"
    );
    assert!(
        state.request_body_requires_rebuild(),
        "hoisting a discovered tool rewrites the outbound body"
    );
}

#[test]
fn discovered_plain_function_is_hoisted_and_not_restored() {
    // A discovered plain `function` becomes callable but needs no restoration: its
    // returned `function_call` stays a `function_call`.
    let mut state = state_with_discovered_tools(
        &json!([{"type": "function", "name": "grep", "parameters": {"type": "object"}}]),
        true,
    );
    filter().lower_request(&mut state, false).expect("lowering succeeds");
    let tools = state.request_body["tools"].as_array().expect("outbound tools");
    assert!(
        tools
            .iter()
            .any(|tool| tool["name"] == "grep" && tool["type"] == "function"),
        "the discovered plain function is hoisted"
    );
    assert!(
        !state.client_tool_lowering.contains_key("grep"),
        "a plain function needs no restoration entry"
    );
}

#[test]
fn discovered_custom_with_defer_loading_becomes_callable() {
    // A `tool_search` that surfaces a `custom` tool still marked `defer_loading:
    // true` is loading it now: the deferred hint is consumed and the tool becomes
    // callable (lowered to a private function) rather than being rejected.
    let mut state = state_with_discovered_tools(
        &json!([{
            "type": "custom",
            "name": "apply_patch",
            "description": "Apply a patch.",
            "format": {"type": "text"},
            "defer_loading": true
        }]),
        true,
    );
    filter().lower_request(&mut state, false).expect("lowering succeeds");
    let tools = state.request_body["tools"].as_array().expect("outbound tools");
    let hoisted = tools
        .iter()
        .find(|tool| tool["name"] == "apply_patch")
        .expect("the discovered deferred custom is hoisted");
    assert_eq!(
        hoisted["type"], "function",
        "the discovered custom is lowered to a private function"
    );
    assert!(
        hoisted.get("defer_loading").is_none(),
        "the deferred hint is consumed on the discovery path"
    );
    assert!(
        state.client_tool_lowering.contains_key("apply_patch"),
        "the discovered tool is registered for restoration"
    );

    // The backend calls it; it must restore to a custom_tool_call.
    let response = json!({
        "object": "response",
        "output": [{
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "apply_patch",
            "arguments": "{\"input\":\"diff\"}",
            "status": "completed"
        }],
    });
    let restored = restore(&state, &response);
    assert_eq!(
        restored["output"][0]["type"], "custom_tool_call",
        "the discovered deferred custom call is restored"
    );
}

#[test]
fn discovered_namespace_custom_member_with_defer_loading_becomes_callable() {
    // A `namespace` surfaced by `tool_search` whose `custom` member is still marked
    // `defer_loading: true` is being loaded now: the member becomes callable, its
    // deferred hint consumed, and its call restores to a namespaced custom_tool_call.
    let mut state = state_with_discovered_tools(
        &json!([{
            "type": "namespace",
            "name": "git",
            "description": "Git operations.",
            "tools": [{
                "type": "custom",
                "name": "freeform",
                "description": "Freeform git.",
                "format": {"type": "text"},
                "defer_loading": true
            }]
        }]),
        true,
    );
    filter().lower_request(&mut state, false).expect("lowering succeeds");
    let flat = namespace_member_name("git", "freeform");
    let tools = state.request_body["tools"].as_array().expect("outbound tools");
    let member = tools
        .iter()
        .find(|tool| tool["name"] == flat)
        .expect("the discovered deferred custom member is hoisted");
    assert_eq!(
        member["type"], "function",
        "the member is lowered to a private function"
    );
    assert!(
        state.client_tool_lowering.contains_key(&flat),
        "the discovered member is registered for restoration"
    );

    let response = json!({
        "object": "response",
        "output": [{
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": flat,
            "arguments": "{\"input\":\"status\"}",
            "status": "completed"
        }],
    });
    let restored = restore(&state, &response);
    let item = &restored["output"][0];
    assert_eq!(
        item["type"], "custom_tool_call",
        "restores to a namespaced custom_tool_call"
    );
    assert_eq!(item["name"], "freeform", "the original member name is recovered");
    assert_eq!(
        item["namespace"], "git",
        "the namespace is recovered onto the custom call"
    );
}

#[test]
fn discovered_namespace_function_member_with_defer_loading_becomes_callable() {
    // A `namespace` surfaced by `tool_search` whose `function` member is still marked
    // `defer_loading: true` is being loaded now: the deferred hint is consumed and
    // the member becomes callable.
    let mut state = state_with_discovered_tools(
        &json!([{
            "type": "namespace",
            "name": "git",
            "description": "Git operations.",
            "tools": [{
                "type": "function",
                "name": "commit",
                "parameters": {"type": "object"},
                "defer_loading": true
            }]
        }]),
        true,
    );
    filter().lower_request(&mut state, false).expect("lowering succeeds");
    let flat = namespace_member_name("git", "commit");
    let tools = state.request_body["tools"].as_array().expect("outbound tools");
    let member = tools
        .iter()
        .find(|tool| tool["name"] == flat)
        .expect("the discovered deferred function member is hoisted");
    assert_eq!(member["type"], "function", "the member stays a function");
    assert!(
        member.get("defer_loading").is_none(),
        "the deferred hint is consumed on the discovery path"
    );
}

#[test]
fn deferred_top_level_custom_and_its_discovered_copy_do_not_collide() {
    // The end-to-end shape: the same `custom` tool is declared `defer_loading: true`
    // and also surfaced by a prior `tool_search`. The deferred declaration is
    // withheld while the discovered copy is loaded and made callable, so the two
    // never collide on the shared lowered name.
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "custom", "name": "apply_patch", "defer_loading": true}],
        "input": [
            {
                "type": "tool_search_call",
                "call_id": "call_ts",
                "execution": "client",
                "arguments": {"query": "patch"}
            },
            {
                "type": "tool_search_output",
                "call_id": "call_ts",
                "status": "completed",
                "tools": [{
                    "type": "custom",
                    "name": "apply_patch",
                    "description": "Apply a patch.",
                    "format": {"type": "text"}
                }]
            }
        ],
    }));
    filter()
        .lower_request(&mut state, false)
        .expect("the deferred declaration and its discovered copy do not collide");
    let tools = state.request_body["tools"].as_array().expect("outbound tools");
    let hoisted: Vec<_> = tools.iter().filter(|tool| tool["name"] == "apply_patch").collect();
    assert_eq!(
        hoisted.len(),
        1,
        "the discovered copy is hoisted exactly once; the deferred declaration is withheld"
    );
    assert_eq!(
        hoisted[0]["type"], "function",
        "the surviving copy is the lowered discovered tool"
    );
    assert!(
        state.client_tool_lowering.contains_key("apply_patch"),
        "the discovered copy is callable and registered for restoration"
    );
}

#[test]
fn discovered_tool_colliding_with_declared_fails_closed() {
    // A discovered tool whose lowered name collides with a declared tool is
    // genuinely ambiguous and fails closed.
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [
            {"type": "tool_search"},
            {"type": "function", "name": "apply_patch", "parameters": {"type": "object"}}
        ],
        "input": [
            {"type": "tool_search_call", "call_id": "call_ts", "execution": "client", "arguments": {}},
            {
                "type": "tool_search_output",
                "call_id": "call_ts",
                "status": "completed",
                "tools": [{"type": "custom", "name": "apply_patch", "format": {"type": "text"}}]
            }
        ],
    }));
    let action = filter()
        .lower_request(&mut state, false)
        .expect_err("a discovered tool colliding with a declared tool must fail closed");
    let (status, message) = reject_parts(&action);
    assert_eq!(status, 400, "a collision is a bad request");
    assert!(
        message.contains("apply_patch"),
        "the rejection names the collision: {message}"
    );
}

#[test]
fn server_executed_search_discovered_tools_are_not_hoisted() {
    // A server-executed `tool_search` is not a client-owned round; its discovered
    // tools are not hoisted and the output keeps its typed shape.
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "tool_search"}],
        "input": [
            {"type": "tool_search_call", "call_id": "call_ts", "execution": "server", "arguments": {}},
            {
                "type": "tool_search_output",
                "call_id": "call_ts",
                "tools": [{"type": "custom", "name": "apply_patch", "format": {"type": "text"}}]
            }
        ],
    }));
    filter().lower_request(&mut state, false).expect("lowering succeeds");
    let tools = state.request_body["tools"].as_array().expect("outbound tools");
    assert!(
        tools.iter().all(|tool| tool["name"] != "apply_patch"),
        "a server-owned search contributes no callable tools"
    );
    assert_eq!(
        state.messages[1]["type"], "tool_search_output",
        "the server-owned output keeps its typed shape"
    );
}

#[test]
fn streaming_discovered_tools_fail_closed() {
    // Streaming cannot restore hoisted private names on the SSE stream, so a
    // streaming request that would hoist discovered tools fails closed with HTTP
    // 400 rather than silently degrading them to non-callable context (#1159).
    let mut state = state_with_discovered_tools(
        &json!([{"type": "custom", "name": "apply_patch", "description": "d", "format": {"type": "text"}}]),
        false,
    );
    let action = filter()
        .lower_request(&mut state, true)
        .expect_err("a streaming request carrying discovered tools must fail closed");
    let (status, message) = reject_parts(&action);
    assert_eq!(status, 400, "streaming discovered tools is a bad request");
    assert!(
        message.contains("stream=false"),
        "the rejection tells the caller to retry without streaming: {message}"
    );
    assert!(
        state.client_tool_echo.is_none(),
        "no restoration is armed for a rejected streaming continuation"
    );
}

#[test]
fn discovered_tools_over_cap_fail_closed() {
    let capped = ClientToolCompatFilter {
        max_rewritten_body_bytes: MAX_JSON_BODY_BYTES,
        max_client_tools: 1,
    };
    let mut state = state_with_discovered_tools(
        &json!([
            {"type": "custom", "name": "a", "format": {"type": "text"}},
            {"type": "custom", "name": "b", "format": {"type": "text"}}
        ]),
        true,
    );
    let action = capped
        .lower_request(&mut state, false)
        .expect_err("more discovered tools than the cap must fail closed");
    let (status, message) = reject_parts(&action);
    assert_eq!(status, 400, "exceeding the discovered-tool cap is a bad request");
    assert!(
        message.contains("maximum of 1"),
        "the rejection explains the cap: {message}"
    );
}

#[test]
fn passthrough_functions_are_not_counted_toward_max_client_tools() {
    // `max_client_tools` bounds *lowered* plus *withheld* client tools, not plain
    // `function` tools forwarded through unchanged. A passthrough function stays in
    // the outbound body verbatim, so `max_rewritten_body_bytes` already bounds it,
    // exactly as the native passthrough path (a `tools` array of plain functions with
    // no rich tool) applies no tool-count cap at all. So one lowered custom plus three
    // passthrough functions clears a cap of 1: only the custom is lowered. Counting
    // passthrough here would reject a large plain-function set only when a rich tool
    // happens to be co-declared — see `enforce_tool_cap`.
    let capped = ClientToolCompatFilter {
        max_rewritten_body_bytes: MAX_JSON_BODY_BYTES,
        max_client_tools: 1,
    };
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [
            {"type": "custom", "name": "live", "format": {"type": "text"}},
            {"type": "function", "name": "p1", "parameters": {"type": "object"}},
            {"type": "function", "name": "p2", "parameters": {"type": "object"}},
            {"type": "function", "name": "p3", "parameters": {"type": "object"}}
        ],
    }));
    capped
        .lower_request(&mut state, false)
        .expect("passthrough functions do not count toward the lowered-tool cap");
    let tools = state.request_body["tools"].as_array().expect("outbound tools");
    assert_eq!(
        tools.len(),
        4,
        "the lowered custom and all three passthrough functions are forwarded: {tools:?}"
    );
}

// -----------------------------------------------------------------------------
// Cap counts withheld tools; discovery-only malformed `tools` (round 13)
// -----------------------------------------------------------------------------

#[test]
fn withheld_deferred_tools_count_toward_max_client_tools() {
    // Deferred declarations are withheld from the outbound callable set but still
    // echoed back in the pre-lowering snapshot, so they count toward the cap: two
    // deferred customs exceed a cap of 1 and fail closed rather than bypassing it.
    let capped = ClientToolCompatFilter {
        max_rewritten_body_bytes: MAX_JSON_BODY_BYTES,
        max_client_tools: 1,
    };
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [
            {"type": "custom", "name": "a", "defer_loading": true},
            {"type": "custom", "name": "b", "defer_loading": true}
        ],
    }));
    let action = capped
        .lower_request(&mut state, false)
        .expect_err("withheld deferred tools over the cap must fail closed");
    let (status, message) = reject_parts(&action);
    assert_eq!(status, 400, "exceeding the cap via withheld tools is a bad request");
    assert!(
        message.contains("maximum of 1"),
        "the rejection explains the cap: {message}"
    );
    assert_eq!(
        state.request_body["tools"][0]["name"], "a",
        "the client's original tools are restored verbatim on rejection"
    );
}

#[test]
fn withheld_namespace_members_count_toward_max_client_tools() {
    // Each deferred namespace member is withheld and echoed back, so members count
    // toward the cap exactly like top-level tools.
    let capped = ClientToolCompatFilter {
        max_rewritten_body_bytes: MAX_JSON_BODY_BYTES,
        max_client_tools: 1,
    };
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{
            "type": "namespace",
            "name": "git",
            "description": "Git operations.",
            "tools": [
                {"type": "function", "name": "commit", "defer_loading": true, "parameters": {"type": "object"}},
                {"type": "function", "name": "push", "defer_loading": true, "parameters": {"type": "object"}}
            ]
        }],
    }));
    let action = capped
        .lower_request(&mut state, false)
        .expect_err("withheld namespace members over the cap must fail closed");
    let (status, message) = reject_parts(&action);
    assert_eq!(status, 400, "exceeding the cap via withheld members is a bad request");
    assert!(
        message.contains("maximum of 1"),
        "the rejection explains the cap: {message}"
    );
}

#[test]
fn lowered_plus_withheld_tools_count_toward_max_client_tools() {
    // A cap of 1 is filled by one lowered custom; a second, deferred custom is
    // withheld but still counts, tipping the combined total over the cap.
    let capped = ClientToolCompatFilter {
        max_rewritten_body_bytes: MAX_JSON_BODY_BYTES,
        max_client_tools: 1,
    };
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [
            {"type": "custom", "name": "live", "format": {"type": "text"}},
            {"type": "custom", "name": "deferred", "defer_loading": true}
        ],
    }));
    let action = capped
        .lower_request(&mut state, false)
        .expect_err("lowered plus withheld over the cap must fail closed");
    let (status, _) = reject_parts(&action);
    assert_eq!(status, 400, "the combined lowered-plus-withheld count exceeds the cap");
}

#[test]
fn withheld_tools_at_the_cap_are_allowed() {
    // The cap is inclusive: exactly `max_client_tools` withheld tools succeed; only
    // the next one fails, so a boundary-sized deferred set is not falsely rejected.
    let capped = ClientToolCompatFilter {
        max_rewritten_body_bytes: MAX_JSON_BODY_BYTES,
        max_client_tools: 2,
    };
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [
            {"type": "custom", "name": "a", "defer_loading": true},
            {"type": "custom", "name": "b", "defer_loading": true}
        ],
    }));
    capped
        .lower_request(&mut state, false)
        .expect("exactly max_client_tools withheld tools are allowed");
    assert_eq!(
        state.request_body["tools"],
        json!([]),
        "both deferred customs are withheld from the outbound callable set"
    );
}

#[test]
fn deferred_custom_reloaded_by_discovery_counts_once_toward_max_client_tools() {
    // A `custom` tool declared `defer_loading: true` and also surfaced by a prior
    // `tool_search` is one logical tool: it is withheld as a declaration and lowered
    // as the discovered copy under the same wire name. It must count once, not twice,
    // so a cap of 1 admits it — `register` reclaims the withheld budget when the
    // discovered copy is loaded. Before the reclaim this tipped 1 (withheld) + 1
    // (lowered) over the cap and falsely rejected the primary defer-then-discover
    // continuation.
    let capped = ClientToolCompatFilter {
        max_rewritten_body_bytes: MAX_JSON_BODY_BYTES,
        max_client_tools: 1,
    };
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "custom", "name": "apply_patch", "defer_loading": true}],
        "input": [
            {"type": "tool_search_call", "call_id": "call_ts", "execution": "client", "arguments": {}},
            {
                "type": "tool_search_output",
                "call_id": "call_ts",
                "status": "completed",
                "tools": [{"type": "custom", "name": "apply_patch", "format": {"type": "text"}}]
            }
        ],
    }));
    capped
        .lower_request(&mut state, false)
        .expect("a deferred tool reloaded by discovery counts once and fits the cap");
    let tools = state.request_body["tools"].as_array().expect("outbound tools");
    let hoisted: Vec<_> = tools.iter().filter(|tool| tool["name"] == "apply_patch").collect();
    assert_eq!(hoisted.len(), 1, "the one logical tool is hoisted exactly once");
    assert!(
        state.client_tool_lowering.contains_key("apply_patch"),
        "the discovered copy is registered for restoration"
    );
}

#[test]
fn deferred_namespace_member_reloaded_by_discovery_counts_once_toward_max_client_tools() {
    // The same reclaim covers namespace members: a deferred `git/commit` member is
    // withheld under its flat wire name and reloaded by discovery under that same
    // name, so it counts once and a cap of 1 admits it.
    let capped = ClientToolCompatFilter {
        max_rewritten_body_bytes: MAX_JSON_BODY_BYTES,
        max_client_tools: 1,
    };
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{
            "type": "namespace",
            "name": "git",
            "description": "Git operations.",
            "tools": [
                {"type": "function", "name": "commit", "defer_loading": true, "parameters": {"type": "object"}}
            ]
        }],
        "input": [
            {"type": "tool_search_call", "call_id": "call_ts", "execution": "client", "arguments": {}},
            {
                "type": "tool_search_output",
                "call_id": "call_ts",
                "status": "completed",
                "tools": [{
                    "type": "namespace",
                    "name": "git",
                    "description": "Git operations.",
                    "tools": [{"type": "function", "name": "commit", "parameters": {"type": "object"}}]
                }]
            }
        ],
    }));
    capped
        .lower_request(&mut state, false)
        .expect("a deferred namespace member reloaded by discovery counts once and fits the cap");
    let flat = namespace_member_name("git", "commit");
    assert!(
        state.client_tool_lowering.contains_key(&flat),
        "the discovered member is registered under its flat wire name"
    );
}

#[test]
fn deferred_custom_named_shell_and_local_shell_both_count_toward_cap() {
    // Regression (false reclaim): a deferred `custom` the client named "shell" and a
    // local `shell` tool (lowered to the fixed "shell" function name) are two DISTINCT
    // logical tools that merely share a wire name. The withheld-budget reclaim fires
    // only on the discovery path, so a local shell declared alongside the deferred
    // custom must not credit its registration against the deferred custom's budget:
    // the pair counts as two and fails closed at a cap of 1 regardless of order.
    for tools in [
        json!([
            {"type": "custom", "name": "shell", "defer_loading": true, "description": "x"},
            {"type": "shell", "environment": {"type": "local"}}
        ]),
        json!([
            {"type": "shell", "environment": {"type": "local"}},
            {"type": "custom", "name": "shell", "defer_loading": true, "description": "x"}
        ]),
    ] {
        let capped = ClientToolCompatFilter {
            max_rewritten_body_bytes: MAX_JSON_BODY_BYTES,
            max_client_tools: 1,
        };
        let mut state = ResponsesState::from_request_body(json!({ "tools": tools }));
        let action = capped
            .lower_request(&mut state, false)
            .expect_err("two distinct tools sharing the wire name 'shell' exceed a cap of 1");
        let (status, message) = reject_parts(&action);
        assert_eq!(status, 400, "the pair exceeds the cap");
        assert!(
            message.contains("maximum of 1"),
            "the rejection explains the cap: {message}"
        );
    }
}

#[test]
fn deferred_custom_named_tool_search_and_client_tool_search_both_count_toward_cap() {
    // The same false-reclaim guard for the fixed `tool_search` name: a deferred
    // `custom` named "tool_search" and a client `tool_search` tool are two distinct
    // tools sharing a wire name, so the pair fails closed at a cap of 1.
    let capped = ClientToolCompatFilter {
        max_rewritten_body_bytes: MAX_JSON_BODY_BYTES,
        max_client_tools: 1,
    };
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [
            {"type": "custom", "name": "tool_search", "defer_loading": true, "description": "x"},
            {"type": "tool_search"}
        ],
    }));
    let action = capped
        .lower_request(&mut state, false)
        .expect_err("a deferred custom and a client tool_search sharing a wire name exceed a cap of 1");
    let (status, _) = reject_parts(&action);
    assert_eq!(status, 400, "the pair exceeds the cap");
}

#[test]
fn top_level_tool_using_reserved_namespace_prefix_fails_closed() {
    // The `agentic_ns__` prefix is reserved for the flat wire names synthesized for
    // namespace members. A client-declared top-level `function` or `custom` that
    // usurps it would collide with a synthesized member on the wire (corrupting
    // restoration and the cap accounting), so once the filter engages it fails closed
    // before any lowering — regardless of a generous cap. The `tool_search` sibling
    // engages the filter (a lone plain `function` is not rich and passes through
    // untouched, synthesizing no member to collide with). This is the declaration-path
    // guard for the wire-name collision the discovery-path reclaim otherwise mis-credits.
    let flat = namespace_member_name("git", "commit");
    for tool in [
        json!({"type": "function", "name": flat, "parameters": {"type": "object"}}),
        json!({"type": "function", "name": flat, "parameters": {"type": "object"}, "defer_loading": true}),
        json!({"type": "custom", "name": flat}),
        json!({"type": "custom", "name": flat, "defer_loading": true}),
    ] {
        let filter = ClientToolCompatFilter {
            max_rewritten_body_bytes: MAX_JSON_BODY_BYTES,
            max_client_tools: 512,
        };
        let mut state = ResponsesState::from_request_body(json!({ "tools": [tool, {"type": "tool_search"}] }));
        let action = filter
            .lower_request(&mut state, false)
            .expect_err("a top-level tool using the reserved namespace prefix fails closed");
        let (status, message) = reject_parts(&action);
        assert_eq!(status, 400, "the reserved prefix is rejected");
        assert!(
            message.contains("reserved") && message.contains(NAMESPACE_MEMBER_PREFIX),
            "the rejection names the reserved prefix: {message}"
        );
    }
}

#[test]
fn deferred_top_level_tool_colliding_with_discovered_namespace_member_fails_closed() {
    // The confirmed discovery-path cap bypass: a deferred top-level `custom`/`function`
    // the client named exactly a namespace member's flat wire name, plus a genuinely
    // distinct namespace member discovered via `tool_search` that flattens to the same
    // name. Before the reserved-prefix guard the discovery-path reclaim credited the
    // discovered member against the withheld top-level tool, admitting two distinct
    // logical tools under a cap of one. The reservation now fails it closed up front.
    let flat = namespace_member_name("git", "commit");
    for (top_level, member) in [
        (
            json!({"type": "custom", "name": flat, "defer_loading": true}),
            json!({"type": "custom", "name": "commit"}),
        ),
        (
            json!({"type": "function", "name": flat, "parameters": {"type": "object"}, "defer_loading": true}),
            json!({"type": "function", "name": "commit", "parameters": {"type": "object"}}),
        ),
    ] {
        let capped = ClientToolCompatFilter {
            max_rewritten_body_bytes: MAX_JSON_BODY_BYTES,
            max_client_tools: 1,
        };
        let mut state = ResponsesState::from_request_body(json!({
            "tools": [top_level, {"type": "tool_search"}],
            "input": [
                {"type": "tool_search_call", "call_id": "call_1", "execution": "client",
                 "arguments": {"query": "x"}, "id": "tsc_1"},
                {"type": "tool_search_output", "call_id": "call_1", "tools": [
                    {"type": "namespace", "name": "git", "description": "Git operations.",
                     "tools": [member]}
                ]}
            ],
        }));
        let action = capped
            .lower_request(&mut state, false)
            .expect_err("a deferred top-level tool colliding with a discovered member fails closed");
        let (status, message) = reject_parts(&action);
        assert_eq!(status, 400, "the collision is rejected");
        assert!(
            message.contains("reserved"),
            "the rejection names the reserved prefix rather than admitting the pair: {message}"
        );
    }
}

#[test]
fn discovered_top_level_tool_using_reserved_namespace_prefix_fails_closed() {
    // A hoisted (discovered) top-level `custom` is subject to the same reservation as
    // a declared one, so a `tool_search` result cannot smuggle in a top-level tool that
    // impersonates a namespace member wire name.
    let flat = namespace_member_name("git", "commit");
    let filter = ClientToolCompatFilter {
        max_rewritten_body_bytes: MAX_JSON_BODY_BYTES,
        max_client_tools: 512,
    };
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "tool_search"}],
        "input": [
            {"type": "tool_search_call", "call_id": "call_1", "execution": "client",
             "arguments": {"query": "x"}, "id": "tsc_1"},
            {"type": "tool_search_output", "call_id": "call_1", "tools": [
                {"type": "custom", "name": flat}
            ]}
        ],
    }));
    let action = filter
        .lower_request(&mut state, false)
        .expect_err("a discovered top-level tool using the reserved prefix fails closed");
    let (status, message) = reject_parts(&action);
    assert_eq!(status, 400, "the reserved prefix is rejected on the discovery path");
    assert!(
        message.contains("reserved") && message.contains(NAMESPACE_MEMBER_PREFIX),
        "the rejection names the reserved prefix: {message}"
    );
}

#[test]
fn legitimate_namespace_member_is_not_rejected_by_the_reserved_prefix() {
    // The reservation targets only client-declared top-level `function`/`custom` tools;
    // a genuine `namespace` member (whose synthesized flat name legitimately carries the
    // prefix) still lowers and registers under that flat name.
    let filter = ClientToolCompatFilter {
        max_rewritten_body_bytes: MAX_JSON_BODY_BYTES,
        max_client_tools: 512,
    };
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{
            "type": "namespace",
            "name": "git",
            "description": "Git operations.",
            "tools": [{"type": "function", "name": "commit", "parameters": {"type": "object"}}]
        }],
    }));
    filter
        .lower_request(&mut state, false)
        .expect("a legitimate namespace member lowers despite carrying the reserved prefix");
    assert!(
        state
            .client_tool_lowering
            .contains_key(&namespace_member_name("git", "commit")),
        "the namespace member is registered under its flat wire name"
    );
}

#[test]
fn namespace_name_embedding_the_reserved_delimiter_fails_closed() {
    // A `namespace` group name or member name carrying the `__` flattening delimiter
    // would make its flattened wire name ambiguous with a distinct namespace/member
    // pair, so both fail closed up front on either lowering path.
    for tool in [
        json!({
            "type": "namespace",
            "name": "a__b",
            "description": "Ambiguous group name.",
            "tools": [{"type": "function", "name": "c", "parameters": {"type": "object"}}]
        }),
        json!({
            "type": "namespace",
            "name": "a",
            "description": "Ambiguous member name.",
            "tools": [{"type": "function", "name": "b__c", "parameters": {"type": "object"}}]
        }),
    ] {
        // Declaration path.
        let mut state = ResponsesState::from_request_body(json!({ "tools": [tool.clone()] }));
        let action = filter()
            .lower_request(&mut state, false)
            .expect_err("a namespace name embedding the reserved delimiter fails closed");
        let (status, message) = reject_parts(&action);
        assert_eq!(
            status, 400,
            "the reserved delimiter is rejected on the declaration path"
        );
        assert!(
            message.contains("delimiter") && message.contains(NAMESPACE_NAME_DELIMITER),
            "the rejection names the reserved delimiter: {message}"
        );

        // Discovery path: the same reservation applies to a hoisted namespace.
        let mut state = ResponsesState::from_request_body(json!({
            "tools": [{"type": "tool_search"}],
            "input": [
                {"type": "tool_search_call", "call_id": "call_1", "execution": "client",
                 "arguments": {"query": "x"}, "id": "tsc_1"},
                {"type": "tool_search_output", "call_id": "call_1", "tools": [tool]}
            ],
        }));
        let action = filter()
            .lower_request(&mut state, false)
            .expect_err("a discovered namespace embedding the reserved delimiter fails closed");
        let (status, message) = reject_parts(&action);
        assert_eq!(status, 400, "the reserved delimiter is rejected on the discovery path");
        assert!(
            message.contains("delimiter"),
            "the discovery-path rejection names the reserved delimiter: {message}"
        );
    }
}

#[test]
fn deferred_namespace_member_colliding_via_reserved_delimiter_fails_closed() {
    // The namespace-internal analogue of the top-level cap bypass: a deferred member
    // `a`/`b__c` and a genuinely distinct discovered member `a__b`/`c` both flatten to
    // `agentic_ns__a__b__c`. The deferred member is withheld (never claimed), so without
    // the delimiter reservation the discovery-path reclaim would credit the discovered
    // member against the withheld one, admitting two logical tools under a cap of one.
    // Reserving the delimiter fails the deferred declaration closed before it is withheld.
    assert_eq!(
        namespace_member_name("a", "b__c"),
        namespace_member_name("a__b", "c"),
        "the two distinct members flatten to the same wire name absent the reservation"
    );
    let capped = ClientToolCompatFilter {
        max_rewritten_body_bytes: MAX_JSON_BODY_BYTES,
        max_client_tools: 1,
    };
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [
            {"type": "namespace", "name": "a", "description": "Deferred group.",
             "tools": [{"type": "function", "name": "b__c", "parameters": {"type": "object"},
                        "defer_loading": true}]},
            {"type": "tool_search"}
        ],
        "input": [
            {"type": "tool_search_call", "call_id": "call_1", "execution": "client",
             "arguments": {"query": "x"}, "id": "tsc_1"},
            {"type": "tool_search_output", "call_id": "call_1", "tools": [
                {"type": "namespace", "name": "a__b", "description": "Discovered group.",
                 "tools": [{"type": "function", "name": "c", "parameters": {"type": "object"}}]}
            ]}
        ],
    }));
    let action = capped
        .lower_request(&mut state, false)
        .expect_err("a deferred member colliding via the reserved delimiter fails closed");
    let (status, message) = reject_parts(&action);
    assert_eq!(status, 400, "the collision is rejected rather than admitting the pair");
    assert!(
        message.contains("delimiter"),
        "the rejection names the reserved delimiter: {message}"
    );
}

#[test]
fn namespace_names_with_single_underscores_are_accepted() {
    // The reservation targets only the `__` double-underscore delimiter; single
    // underscores are ubiquitous in tool names and must still lower cleanly.
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{
            "type": "namespace",
            "name": "file_ops",
            "description": "File operations.",
            "tools": [{"type": "function", "name": "read_file", "parameters": {"type": "object"}}]
        }],
    }));
    filter()
        .lower_request(&mut state, false)
        .expect("a namespace with single-underscore names lowers cleanly");
    assert!(
        state
            .client_tool_lowering
            .contains_key(&namespace_member_name("file_ops", "read_file")),
        "the single-underscore namespace member is registered under its flat wire name"
    );
}

#[test]
fn namespace_boundary_underscore_colliding_pair_fails_closed() {
    // The subtler namespace-flattening collision the `__`-substring guard alone misses:
    // a boundary single `_` merges with the fixed `__` separator into `___`, so distinct
    // pairs re-split to the same wire name. `a_`/`b` and `a`/`_b` both flatten to
    // `agentic_ns__a___b`, yet none of the components contains `__`. A deferred member of
    // one pair (withheld, never claimed) and a discovered member of the other would let
    // the discovery-path reclaim credit two distinct tools as one — a cap bypass — unless
    // the boundary underscore is reserved too. Confirm the collision exists, then that it
    // fails closed at a cap of one.
    assert_eq!(
        namespace_member_name("a_", "b"),
        namespace_member_name("a", "_b"),
        "the boundary underscore makes the two distinct pairs flatten identically"
    );
    let capped = ClientToolCompatFilter {
        max_rewritten_body_bytes: MAX_JSON_BODY_BYTES,
        max_client_tools: 1,
    };
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [
            {"type": "namespace", "name": "a_", "description": "Deferred group.",
             "tools": [{"type": "function", "name": "b", "parameters": {"type": "object"},
                        "defer_loading": true}]},
            {"type": "tool_search"}
        ],
        "input": [
            {"type": "tool_search_call", "call_id": "call_1", "execution": "client",
             "arguments": {"query": "x"}, "id": "tsc_1"},
            {"type": "tool_search_output", "call_id": "call_1", "tools": [
                {"type": "namespace", "name": "a", "description": "Discovered group.",
                 "tools": [{"type": "function", "name": "_b", "parameters": {"type": "object"}}]}
            ]}
        ],
    }));
    let action = capped
        .lower_request(&mut state, false)
        .expect_err("a boundary-underscore collision must fail closed rather than reclaim");
    let (status, message) = reject_parts(&action);
    assert_eq!(
        status, 400,
        "the collision is rejected rather than admitting the pair as one tool"
    );
    assert!(
        message.contains("delimiter"),
        "the rejection names the reserved delimiter: {message}"
    );
}

#[test]
fn deferred_namespace_member_hash_colliding_with_a_callable_member_fails_closed_on_tool_choice() {
    // Behavioral consequence of the forward-FNV wire collision on `tool_choice`: one namespace
    // with a callable member M1 whose name is the 16-hex FNV of a deferred member M2's full
    // name. Pre-fix M1's verbatim 64-char wire equaled M2's hashed wire, so a `tool_choice`
    // forcing the *deferred* M2 resolved through the reverse map to the *callable* M1 and was
    // silently redirected to a different tool — a fail-closed violation. With the hashed `___`
    // marker the wires differ, so the forced deferred selector is no longer callable and fails
    // closed as deferred.
    let namespace = "a".repeat(34);
    let deferred_member = "m".repeat(17);
    let deferred_full = format!("{NAMESPACE_MEMBER_PREFIX}{namespace}__{deferred_member}");
    let callable_member = format!("{:016x}", stable_name_hash(&deferred_full));
    assert_ne!(
        namespace_member_name(&namespace, &deferred_member),
        namespace_member_name(&namespace, &callable_member),
        "the callable and deferred members must no longer share a wire name"
    );
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{
            "type": "namespace", "name": namespace, "description": "Group.",
            "tools": [
                {"type": "function", "name": callable_member, "parameters": {"type": "object"}},
                {"type": "function", "name": deferred_member, "parameters": {"type": "object"},
                 "defer_loading": true}
            ]
        }],
        "tool_choice": {"type": "function", "namespace": namespace, "name": deferred_member},
    }));
    let action = filter()
        .lower_request(&mut state, false)
        .expect_err("forcing a deferred member that used to alias a callable member fails closed");
    let (status, message) = reject_parts(&action);
    assert_eq!(
        status, 400,
        "the forced deferred selector is rejected rather than redirected"
    );
    assert!(
        message.contains("deferred"),
        "the rejection reports the member as deferred: {message}"
    );
}

#[test]
fn hash_colliding_deferred_and_discovered_members_both_count_against_the_cap() {
    // Behavioral consequence of the forward-FNV collision on the reclaim budget: a deferred
    // member (hash-truncated wire) and a genuinely distinct *discovered* member whose verbatim
    // wire equaled that hashed wire pre-fix. The deferred member is withheld (never claimed), so
    // the collision slipped past `claim_lowered`; the discovery-path reclaim then removed the
    // withheld entry keyed by the shared wire, crediting two distinct tools as one and admitting
    // both under a cap that should hold one. With the `___` marker the wires differ, the reclaim
    // no longer fires, and the second tool trips the cap.
    let deferred_namespace = "a".repeat(36);
    let deferred_member = "m".repeat(15);
    let deferred_full = format!("{NAMESPACE_MEMBER_PREFIX}{deferred_namespace}__{deferred_member}");
    assert!(
        deferred_full.chars().count() > MAX_FUNCTION_NAME_LEN,
        "the deferred member must take the hash-truncation branch"
    );
    // The discovered member shares the deferred member's 46-char readable prefix and its name is
    // the forward FNV hex of the deferred full name, so its verbatim 64-char wire equaled the
    // deferred member's hashed wire before the fix.
    let discovered_namespace = "a".repeat(34);
    let discovered_member = format!("{:016x}", stable_name_hash(&deferred_full));
    assert_ne!(
        namespace_member_name(&deferred_namespace, &deferred_member),
        namespace_member_name(&discovered_namespace, &discovered_member),
        "the deferred and discovered members must no longer flatten to one wire name"
    );
    // A cap of two admits {tool_search, one member}; a correctly counted second member trips it.
    let capped = ClientToolCompatFilter {
        max_rewritten_body_bytes: MAX_JSON_BODY_BYTES,
        max_client_tools: 2,
    };
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [
            {"type": "namespace", "name": deferred_namespace, "description": "Deferred group.",
             "tools": [{"type": "function", "name": deferred_member, "parameters": {"type": "object"},
                        "defer_loading": true}]},
            {"type": "tool_search"}
        ],
        "input": [
            {"type": "tool_search_call", "call_id": "call_1", "execution": "client",
             "arguments": {"query": "x"}, "id": "tsc_1"},
            {"type": "tool_search_output", "call_id": "call_1", "tools": [
                {"type": "namespace", "name": discovered_namespace, "description": "Discovered group.",
                 "tools": [{"type": "function", "name": discovered_member, "parameters": {"type": "object"}}]}
            ]}
        ],
    }));
    let action = capped
        .lower_request(&mut state, false)
        .expect_err("two members that used to alias one wire must both count against the cap");
    let (status, message) = reject_parts(&action);
    assert_eq!(
        status, 400,
        "the cap fails closed rather than crediting two tools as one"
    );
    assert!(
        message.contains("client tools"),
        "the rejection names the client-tool cap: {message}"
    );
}

#[test]
fn namespace_component_leading_or_trailing_underscore_fails_closed() {
    // Keeping the flattening injective means reserving any boundary `_` that could abut
    // the `__` separator; the rule is applied symmetrically to both the group name and
    // member names, on the declaration path, so the behavior is pinned.
    for tool in [
        json!({
            "type": "namespace", "name": "_utils", "description": "Leading-underscore group.",
            "tools": [{"type": "function", "name": "read", "parameters": {"type": "object"}}]
        }),
        json!({
            "type": "namespace", "name": "utils_", "description": "Trailing-underscore group.",
            "tools": [{"type": "function", "name": "read", "parameters": {"type": "object"}}]
        }),
        json!({
            "type": "namespace", "name": "utils", "description": "Leading-underscore member.",
            "tools": [{"type": "function", "name": "_read", "parameters": {"type": "object"}}]
        }),
        json!({
            "type": "namespace", "name": "utils", "description": "Trailing-underscore member.",
            "tools": [{"type": "function", "name": "read_", "parameters": {"type": "object"}}]
        }),
    ] {
        let mut state = ResponsesState::from_request_body(json!({ "tools": [tool] }));
        let action = filter()
            .lower_request(&mut state, false)
            .expect_err("a boundary underscore in a namespace component fails closed");
        let (status, message) = reject_parts(&action);
        assert_eq!(status, 400, "a boundary underscore is a bad request");
        assert!(
            message.contains("begin or end with '_'"),
            "the rejection explains the boundary-underscore rule: {message}"
        );
    }
}

#[test]
fn discovery_only_non_array_tools_fails_closed() {
    // A discovery-only continuation whose `tools` is a non-array (here a string)
    // fails closed rather than having `commit_lowering` silently replace it with
    // the hoisted discovered array — validating a backend-owned field is not this
    // filter's job, so it must not paper over a malformed value.
    let mut state = ResponsesState::from_request_body(json!({
        "tools": "not-an-array",
        "input": [
            {"type": "tool_search_call", "call_id": "call_ts", "execution": "client", "arguments": {}},
            {
                "type": "tool_search_output",
                "call_id": "call_ts",
                "status": "completed",
                "tools": [{"type": "custom", "name": "apply_patch", "format": {"type": "text"}}]
            }
        ],
    }));
    let action = filter()
        .lower_request(&mut state, false)
        .expect_err("a non-array tools on the discovery path must fail closed");
    let (status, message) = reject_parts(&action);
    assert_eq!(status, 400, "a malformed tools value is a bad request");
    assert!(
        message.contains("tools must be a JSON array"),
        "the rejection names the malformed field: {message}"
    );
    assert_eq!(
        state.request_body["tools"], "not-an-array",
        "the malformed tools value is left untouched, not normalized to the discovered array"
    );
}

#[test]
fn discovery_only_null_tools_hoists_discovered() {
    // A `null` `tools` value means "no declared tools"; the discovered set is still
    // hoisted onto a fresh array rather than rejected as malformed.
    let mut state = ResponsesState::from_request_body(json!({
        "tools": null,
        "input": [
            {"type": "tool_search_call", "call_id": "call_ts", "execution": "client", "arguments": {}},
            {
                "type": "tool_search_output",
                "call_id": "call_ts",
                "status": "completed",
                "tools": [{"type": "custom", "name": "apply_patch", "format": {"type": "text"}}]
            }
        ],
    }));
    filter()
        .lower_request(&mut state, false)
        .expect("null tools hoists the discovered set onto a fresh array");
    let tools = state.request_body["tools"]
        .as_array()
        .expect("outbound tools is a fresh array");
    assert!(
        tools
            .iter()
            .any(|tool| tool["name"] == "apply_patch" && tool["type"] == "function"),
        "the discovered tool is hoisted onto a fresh array: {tools:?}"
    );
}

// -----------------------------------------------------------------------------
// Discovered-tool status gate, conflict rejection, and sanitization (round 5)
// -----------------------------------------------------------------------------

#[test]
fn discovered_tools_from_unfinished_search_are_not_hoisted() {
    // Only a `completed` tool_search_output has an authoritative tool list; an
    // in_progress/incomplete search contributes no callable tools (they stay as
    // history context) rather than being hoisted prematurely.
    for status in ["in_progress", "incomplete"] {
        let mut state = ResponsesState::from_request_body(json!({
            "tools": [{"type": "tool_search"}],
            "input": [
                {"type": "tool_search_call", "call_id": "call_ts", "execution": "client", "arguments": {}},
                {
                    "type": "tool_search_output",
                    "call_id": "call_ts",
                    "status": status,
                    "tools": [{"type": "custom", "name": "apply_patch", "format": {"type": "text"}}]
                }
            ],
        }));
        filter().lower_request(&mut state, false).expect("lowering succeeds");
        let tools = state.request_body["tools"].as_array().expect("outbound tools");
        assert!(
            tools.iter().all(|tool| tool["name"] != "apply_patch"),
            "a {status} search contributes no callable tools"
        );
    }
}

#[test]
fn discovered_tools_missing_or_null_status_are_hoisted() {
    // `status` is optional in `ToolSearchOutputItemParam` (`anyOf[..., null]`; only
    // `type` and `tools` are required), so a missing or explicit-null status is
    // authoritative — treated as `completed` — and its discovered tools are hoisted
    // into the callable set rather than left as history context.
    for output in [
        json!({
            "type": "tool_search_output",
            "call_id": "call_ts",
            "tools": [{"type": "custom", "name": "apply_patch", "format": {"type": "text"}}]
        }),
        json!({
            "type": "tool_search_output",
            "call_id": "call_ts",
            "status": null,
            "tools": [{"type": "custom", "name": "apply_patch", "format": {"type": "text"}}]
        }),
    ] {
        let mut state = ResponsesState::from_request_body(json!({
            "tools": [{"type": "tool_search"}],
            "input": [
                {"type": "tool_search_call", "call_id": "call_ts", "execution": "client", "arguments": {}},
                output,
            ],
        }));
        filter().lower_request(&mut state, false).expect("lowering succeeds");
        let tools = state.request_body["tools"].as_array().expect("outbound tools");
        assert!(
            tools
                .iter()
                .any(|tool| tool["name"] == "apply_patch" && tool["type"] == "function"),
            "a missing or null status output is authoritative and hoists its tools: {tools:?}"
        );
    }
}

#[test]
fn discovered_tools_with_malformed_status_fail_closed() {
    // A status outside the `FunctionCallItemStatus` domain — a bogus string, or a
    // non-string — is malformed input the proxy cannot interpret. Silently treating
    // it as non-terminal would strip a callable tool the client believes it loaded,
    // so a client-lowered `tool_search_output` with a malformed status fails closed
    // with a 400 rather than degrading silently (mirroring `restore_call_status`).
    for status in [json!("searching"), json!(3), json!(true)] {
        let label = status.to_string();
        let mut state = ResponsesState::from_request_body(json!({
            "tools": [{"type": "tool_search"}],
            "input": [
                {"type": "tool_search_call", "call_id": "call_ts", "execution": "client", "arguments": {}},
                {
                    "type": "tool_search_output",
                    "call_id": "call_ts",
                    "status": status,
                    "tools": [{"type": "custom", "name": "apply_patch", "format": {"type": "text"}}]
                }
            ],
        }));
        let action = filter()
            .lower_request(&mut state, false)
            .expect_err(&format!("a malformed discovered status ({label}) must fail closed"));
        let (code, message) = reject_parts(&action);
        assert_eq!(code, 400, "a malformed discovered status ({label}) is a bad request");
        assert!(
            message.contains("status"),
            "the rejection names the malformed status field ({label}): {message}"
        );
    }
}

#[test]
fn identical_discovered_redefinition_is_hoisted_once() {
    // The same (type, name) re-listed identically across searches is deduplicated
    // and hoisted a single time.
    let discovered = json!([{"type": "function", "name": "grep", "parameters": {"type": "object"}}]);
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "tool_search"}],
        "input": [
            {"type": "tool_search_call", "call_id": "call_a", "execution": "client", "arguments": {}},
            {"type": "tool_search_output", "call_id": "call_a", "status": "completed", "tools": discovered.clone()},
            {"type": "tool_search_call", "call_id": "call_b", "execution": "client", "arguments": {}},
            {"type": "tool_search_output", "call_id": "call_b", "status": "completed", "tools": discovered},
        ],
    }));
    filter().lower_request(&mut state, false).expect("lowering succeeds");
    let tools = state.request_body["tools"].as_array().expect("outbound tools");
    let hoisted = tools.iter().filter(|tool| tool["name"] == "grep").count();
    assert_eq!(hoisted, 1, "an identical re-listing is hoisted exactly once");
}

#[test]
fn conflicting_discovered_redefinition_fails_closed() {
    // The same (type, name) redefined with a conflicting definition across searches
    // is ambiguous and fails closed.
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "tool_search"}],
        "input": [
            {"type": "tool_search_call", "call_id": "call_a", "execution": "client", "arguments": {}},
            {
                "type": "tool_search_output", "call_id": "call_a", "status": "completed",
                "tools": [{"type": "function", "name": "grep", "parameters": {"type": "object"}}]
            },
            {"type": "tool_search_call", "call_id": "call_b", "execution": "client", "arguments": {}},
            {
                "type": "tool_search_output", "call_id": "call_b", "status": "completed",
                "tools": [{
                    "type": "function", "name": "grep",
                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}
                }]
            },
        ],
    }));
    let action = filter()
        .lower_request(&mut state, false)
        .expect_err("a conflicting discovered redefinition must fail closed");
    let (status, message) = reject_parts(&action);
    assert_eq!(status, 400, "an ambiguous discovery is a bad request");
    assert!(
        message.contains("grep") && message.contains("conflicting"),
        "the rejection names the ambiguous tool: {message}"
    );
}

#[test]
fn missing_status_output_participates_in_cross_output_dedup_and_conflict() {
    // A missing/null-status `tool_search_output` is authoritative (P1), so it takes
    // part in cross-output dedup and conflict detection exactly like a `completed`
    // output: an identical (type, name) across a completed and a missing-status output
    // dedups to one hoist, while a conflicting redefinition across the two fails closed.
    let grep = json!([{"type": "function", "name": "grep", "parameters": {"type": "object"}}]);
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "tool_search"}],
        "input": [
            {"type": "tool_search_call", "call_id": "call_a", "execution": "client", "arguments": {}},
            {"type": "tool_search_output", "call_id": "call_a", "status": "completed", "tools": grep.clone()},
            {"type": "tool_search_call", "call_id": "call_b", "execution": "client", "arguments": {}},
            // No status: authoritative, listing the same tool identically.
            {"type": "tool_search_output", "call_id": "call_b", "tools": grep},
        ],
    }));
    filter()
        .lower_request(&mut state, false)
        .expect("identical listings across a completed and a missing-status output dedup");
    let tools = state.request_body["tools"].as_array().expect("outbound tools");
    let hoisted = tools.iter().filter(|tool| tool["name"] == "grep").count();
    assert_eq!(hoisted, 1, "a missing-status output dedups with a completed one");

    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "tool_search"}],
        "input": [
            {"type": "tool_search_call", "call_id": "call_a", "execution": "client", "arguments": {}},
            {
                "type": "tool_search_output", "call_id": "call_a", "status": "completed",
                "tools": [{"type": "function", "name": "grep", "parameters": {"type": "object"}}]
            },
            {"type": "tool_search_call", "call_id": "call_b", "execution": "client", "arguments": {}},
            {
                // No status: authoritative, redefining the same tool with a conflicting shape.
                "type": "tool_search_output", "call_id": "call_b",
                "tools": [{
                    "type": "function", "name": "grep",
                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}
                }]
            },
        ],
    }));
    let action = filter()
        .lower_request(&mut state, false)
        .expect_err("a conflicting missing-status redefinition fails closed like a completed one");
    let (status, message) = reject_parts(&action);
    assert_eq!(status, 400, "a cross-output conflict is a bad request");
    assert!(
        message.contains("grep") && message.contains("conflicting"),
        "the rejection names the ambiguous tool: {message}"
    );
}

#[test]
fn discovered_redefinition_differing_only_in_dropped_decorator_is_hoisted_once() {
    // Two searches list the same (type, name) `function` that differ ONLY in fields
    // lowering drops (the Responses-only `defer_loading` hint and a `null`
    // `output_schema`). Both lower to a byte-identical backend function, so the
    // discovery is unambiguous: it is deduplicated and hoisted once rather than
    // falsely rejected as a conflict.
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "tool_search"}],
        "input": [
            {"type": "tool_search_call", "call_id": "call_a", "execution": "client", "arguments": {}},
            {
                "type": "tool_search_output", "call_id": "call_a", "status": "completed",
                "tools": [{
                    "type": "function", "name": "grep",
                    "parameters": {"type": "object"},
                    "defer_loading": true, "output_schema": null
                }]
            },
            {"type": "tool_search_call", "call_id": "call_b", "execution": "client", "arguments": {}},
            {
                "type": "tool_search_output", "call_id": "call_b", "status": "completed",
                "tools": [{"type": "function", "name": "grep", "parameters": {"type": "object"}}]
            },
        ],
    }));
    filter()
        .lower_request(&mut state, false)
        .expect("listings that lower identically are not a conflict");
    let tools = state.request_body["tools"].as_array().expect("outbound tools");
    let grep: Vec<_> = tools.iter().filter(|tool| tool["name"] == "grep").collect();
    assert_eq!(
        grep.len(),
        1,
        "a listing differing only in dropped fields is hoisted once"
    );
    assert!(
        grep[0].get("defer_loading").is_none(),
        "the hoisted copy carries no defer_loading hint"
    );
    assert!(
        grep[0].get("output_schema").is_none(),
        "a null output_schema is dropped by lowering"
    );
}

#[test]
fn restrictive_allowed_callers_discovered_twin_fails_closed_regardless_of_order() {
    // A clean listing and a would-reject twin (restrictive `allowed_callers`) of the
    // same (type, name) must fail closed no matter which search lists which. The
    // dedup identity preserves a non-null `allowed_callers`, so the stricter twin is
    // never silently dropped behind a clean listing that happened to come first — the
    // fail-closed guarantee is order-independent.
    let clean = json!({"type": "function", "name": "grep", "parameters": {"type": "object"}});
    let restricted = json!({
        "type": "function", "name": "grep", "parameters": {"type": "object"},
        "allowed_callers": ["programmatic"]
    });
    let assert_fails_closed = |first: &Value, second: &Value| {
        let mut state = ResponsesState::from_request_body(json!({
            "tools": [{"type": "tool_search"}],
            "input": [
                {"type": "tool_search_call", "call_id": "call_a", "execution": "client", "arguments": {}},
                {"type": "tool_search_output", "call_id": "call_a", "status": "completed", "tools": [first]},
                {"type": "tool_search_call", "call_id": "call_b", "execution": "client", "arguments": {}},
                {"type": "tool_search_output", "call_id": "call_b", "status": "completed", "tools": [second]},
            ],
        }));
        let action = filter()
            .lower_request(&mut state, false)
            .expect_err("a restrictive allowed_callers twin must fail closed in either order");
        let (status, message) = reject_parts(&action);
        assert_eq!(status, 400, "the fail-closed guarantee is order-independent");
        assert!(
            message.contains("grep") && message.contains("conflicting"),
            "the clean listing and its restricted twin are an ambiguous conflict: {message}"
        );
    };
    assert_fails_closed(&clean, &restricted);
    assert_fails_closed(&restricted, &clean);
}

#[test]
fn clean_then_malformed_output_schema_discovered_twin_fails_closed() {
    // A malformed `output_schema` on the SECOND listing, behind a clean twin, must
    // still fail closed: the identity preserves a non-null `output_schema` so the
    // clean listing cannot mask the would-reject twin (which pre-fix depended on
    // listing order).
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "tool_search"}],
        "input": [
            {"type": "tool_search_call", "call_id": "call_a", "execution": "client", "arguments": {}},
            {
                "type": "tool_search_output", "call_id": "call_a", "status": "completed",
                "tools": [{"type": "function", "name": "grep", "parameters": {"type": "object"}}]
            },
            {"type": "tool_search_call", "call_id": "call_b", "execution": "client", "arguments": {}},
            {
                "type": "tool_search_output", "call_id": "call_b", "status": "completed",
                "tools": [{
                    "type": "function", "name": "grep",
                    "parameters": {"type": "object"}, "output_schema": "not-an-object"
                }]
            },
        ],
    }));
    let action = filter()
        .lower_request(&mut state, false)
        .expect_err("a malformed output_schema twin behind a clean listing must fail closed");
    let (status, message) = reject_parts(&action);
    assert_eq!(status, 400, "a malformed discovered output_schema is a bad request");
    assert!(
        message.contains("grep") && message.contains("conflicting"),
        "the clean listing and its malformed twin are an ambiguous conflict: {message}"
    );
}

#[test]
fn deferred_top_level_function_and_discovered_copy_do_not_collide() {
    // A top-level `function` declared with `defer_loading: true` is withheld until a
    // `tool_search` hoists it, so its discovered copy is not a false collision and
    // exactly one callable `grep` (the hoisted copy, sans defer hint) reaches the
    // backend.
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [
            {"type": "tool_search"},
            {"type": "function", "name": "grep", "parameters": {"type": "object"}, "defer_loading": true}
        ],
        "input": [
            {"type": "tool_search_call", "call_id": "call_ts", "execution": "client", "arguments": {}},
            {
                "type": "tool_search_output", "call_id": "call_ts", "status": "completed",
                "tools": [{"type": "function", "name": "grep", "parameters": {"type": "object"}}]
            }
        ],
    }));
    filter()
        .lower_request(&mut state, false)
        .expect("a deferred declaration and its discovered copy do not collide");
    let tools = state.request_body["tools"].as_array().expect("outbound tools");
    let grep: Vec<_> = tools.iter().filter(|tool| tool["name"] == "grep").collect();
    assert_eq!(
        grep.len(),
        1,
        "exactly one `grep` reaches the backend (the hoisted discovered copy)"
    );
    assert!(
        grep[0].get("defer_loading").is_none(),
        "the hoisted copy carries no defer_loading hint"
    );
}

#[test]
fn deferred_top_level_function_never_discovered_is_withheld() {
    // A deferred top-level function that was never discovered is withheld: the model
    // must issue a tool_search to load it.
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [
            {"type": "custom", "name": "c", "format": {"type": "text"}},
            {"type": "function", "name": "grep", "parameters": {"type": "object"}, "defer_loading": true}
        ],
    }));
    filter().lower_request(&mut state, false).expect("lowering succeeds");
    let tools = state.request_body["tools"].as_array().expect("outbound tools");
    assert!(
        tools.iter().all(|tool| tool["name"] != "grep"),
        "a never-discovered deferred function is withheld from the outbound set"
    );
}

#[test]
fn deferred_top_level_function_forwards_verbatim_on_native_passthrough() {
    // A request with no rich client tool and no discovered tools takes the native
    // passthrough path, which never touches the top-level `tools`. A deferred plain
    // `function` therefore forwards verbatim — the `defer_loading` withhold is a
    // collision-avoidance step scoped to the rich/discovered lowering path, and a
    // function-only backend can honor or ignore the documented hint. This documents
    // that the withhold does not run when there is nothing to lower.
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [
            {"type": "function", "name": "grep", "parameters": {"type": "object"}, "defer_loading": true}
        ],
    }));
    filter()
        .lower_request(&mut state, false)
        .expect("native passthrough succeeds");
    let tools = state.request_body["tools"].as_array().expect("outbound tools");
    let grep = tools
        .iter()
        .find(|tool| tool["name"] == "grep")
        .expect("the deferred function is forwarded on the native passthrough path");
    assert_eq!(
        grep["defer_loading"], true,
        "native passthrough forwards the declaration verbatim, defer_loading included"
    );
    assert!(
        !state.request_body_requires_rebuild(),
        "a pure-function passthrough leaves the body untouched"
    );
}

#[test]
fn non_deferred_top_level_function_passes_through() {
    // A plain (non-deferred) top-level function is forwarded verbatim alongside the
    // lowered rich tool.
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [
            {"type": "custom", "name": "c", "format": {"type": "text"}},
            {"type": "function", "name": "grep", "parameters": {"type": "object"}}
        ],
    }));
    filter().lower_request(&mut state, false).expect("lowering succeeds");
    let tools = state.request_body["tools"].as_array().expect("outbound tools");
    assert!(
        tools
            .iter()
            .any(|tool| tool["name"] == "grep" && tool["type"] == "function"),
        "a non-deferred function passes through"
    );
}

#[test]
fn discovered_function_defer_loading_stripped_and_output_schema_preserved() {
    // A discovered `function` is sanitized: the Responses-only `defer_loading` hint
    // is dropped and a valid `output_schema` is preserved (the backend delegates to
    // the OpenAI SDK Tool type, which carries it).
    let mut state = state_with_discovered_tools(
        &json!([{
            "type": "function",
            "name": "grep",
            "parameters": {"type": "object"},
            "defer_loading": true,
            "output_schema": {"type": "object", "properties": {"matches": {"type": "array"}}}
        }]),
        true,
    );
    filter().lower_request(&mut state, false).expect("lowering succeeds");
    let tools = state.request_body["tools"].as_array().expect("outbound tools");
    let grep = tools
        .iter()
        .find(|tool| tool["name"] == "grep")
        .expect("the discovered function is hoisted");
    assert!(
        grep.get("defer_loading").is_none(),
        "defer_loading is stripped from the hoisted copy"
    );
    assert_eq!(
        grep["output_schema"],
        json!({"type": "object", "properties": {"matches": {"type": "array"}}}),
        "a valid output_schema is preserved"
    );
}

#[test]
fn discovered_function_with_restricted_callers_fails_closed() {
    // A discovered `function` that declares a restrictive `allowed_callers` cannot
    // be enforced on a function-only backend, so it fails closed rather than being
    // silently widened to all callers.
    let mut state = state_with_discovered_tools(
        &json!([{
            "type": "function", "name": "grep",
            "parameters": {"type": "object"}, "allowed_callers": ["programmatic"]
        }]),
        true,
    );
    let action = filter()
        .lower_request(&mut state, false)
        .expect_err("a restricted discovered function must fail closed");
    let (status, message) = reject_parts(&action);
    assert_eq!(status, 400, "a restrictive allowed_callers is a bad request");
    assert!(
        message.contains("allowed_callers"),
        "the rejection explains the restriction: {message}"
    );
}

#[test]
fn discovered_function_with_malformed_output_schema_fails_closed() {
    // A discovered `function` whose `output_schema` is neither an object nor null is
    // malformed and fails closed.
    let mut state = state_with_discovered_tools(
        &json!([{"type": "function", "name": "grep", "parameters": {"type": "object"}, "output_schema": "nope"}]),
        true,
    );
    let action = filter()
        .lower_request(&mut state, false)
        .expect_err("a malformed output_schema must fail closed");
    let (status, message) = reject_parts(&action);
    assert_eq!(status, 400, "a malformed output_schema is a bad request");
    assert!(
        message.contains("output_schema"),
        "the rejection explains the schema requirement: {message}"
    );
}

#[test]
fn deferred_namespace_function_member_is_withheld() {
    // A deferred `function` namespace member is withheld from the outbound set on the
    // declaration path until a `tool_search` loads it — a namespace `function` member
    // is not exempt from `defer_loading`. When it is the namespace's only member the
    // outbound callable set is empty and the originals are echoed back.
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{
            "type": "namespace",
            "name": "git",
            "description": "Git tools.",
            "tools": [{
                "type": "function",
                "name": "commit",
                "parameters": {"type": "object"},
                "defer_loading": true,
                "output_schema": {"type": "object"}
            }]
        }],
    }));
    filter().lower_request(&mut state, false).expect("lowering succeeds");
    assert_eq!(
        state.request_body["tools"],
        json!([]),
        "the deferred function member is withheld from the outbound callable set"
    );
    assert!(
        state.client_tool_lowering.is_empty(),
        "nothing was lowered, so there is no restoration recipe"
    );
    let echo = state
        .client_tool_echo
        .as_ref()
        .expect("the client originals are echoed back on the response");
    assert_eq!(
        echo.tools[0]["type"], "namespace",
        "the client's original namespace is echoed back on the response"
    );
}

#[test]
fn namespace_function_member_output_schema_preserved() {
    // A non-deferred `function` namespace member is sanitized and lowered like a
    // discovered function: its Responses-only decorators are reduced to the
    // backend-callable fields while a valid `output_schema` is preserved.
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{
            "type": "namespace",
            "name": "git",
            "description": "Git tools.",
            "tools": [{
                "type": "function",
                "name": "commit",
                "parameters": {"type": "object"},
                "output_schema": {"type": "object"}
            }]
        }],
    }));
    filter().lower_request(&mut state, false).expect("lowering succeeds");
    let tools = state.request_body["tools"].as_array().expect("outbound tools");
    let member = tools
        .iter()
        .find(|tool| tool["name"] == "agentic_ns__git__commit")
        .expect("the member is lowered");
    assert!(
        member.get("defer_loading").is_none(),
        "the lowered member carries no defer_loading hint"
    );
    assert_eq!(
        member["output_schema"],
        json!({"type": "object"}),
        "a valid output_schema is preserved"
    );
}

#[test]
fn namespace_function_member_with_malformed_output_schema_fails_closed() {
    // A `function` namespace member with a non-object, non-null `output_schema` is
    // malformed and fails closed.
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{
            "type": "namespace",
            "name": "git",
            "description": "Git tools.",
            "tools": [{"type": "function", "name": "commit", "parameters": {"type": "object"}, "output_schema": 7}]
        }],
    }));
    let action = filter()
        .lower_request(&mut state, false)
        .expect_err("a malformed member output_schema must fail closed");
    let (status, message) = reject_parts(&action);
    assert_eq!(status, 400, "a malformed output_schema is a bad request");
    assert!(
        message.contains("output_schema"),
        "the rejection explains the schema requirement: {message}"
    );
}

#[test]
fn local_skill_text_is_rendered_verbatim_without_trimming() {
    // Caller-supplied skill text is rendered verbatim; surrounding whitespace is not
    // trimmed, so the model sees exactly what the client declared.
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{
            "type": "shell",
            "environment": {
                "type": "local",
                "skills": [{"name": "  deploy  ", "description": "  ship it  ", "path": "  /bin/deploy  "}]
            }
        }],
    }));
    filter().lower_request(&mut state, false).expect("lowering succeeds");
    let description = state.request_body["tools"][0]["description"]
        .as_str()
        .expect("description");
    assert!(
        description.contains("  deploy  ")
            && description.contains("  ship it  ")
            && description.contains("  /bin/deploy  "),
        "skill text is rendered verbatim without trimming: {description}"
    );
}

// -----------------------------------------------------------------------------
// Cross-kind namespaced tool_choice (#1158 review F4)
// -----------------------------------------------------------------------------

#[test]
fn namespaced_custom_selector_for_function_member_is_rejected() {
    // A `type:"custom"` selector must not force a declared namespaced *function*
    // member: the selector kind disagrees with the declared member, which would
    // change the explicit tool_choice, so it fails closed.
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{
            "type": "namespace",
            "name": "git",
            "description": "Git operations.",
            "tools": [{"type": "function", "name": "commit"}]
        }],
        "tool_choice": {"type": "custom", "name": "commit", "namespace": "git"},
    }));
    let action = filter()
        .lower_request(&mut state, false)
        .expect_err("a cross-kind namespaced selector must fail closed");
    let (status, message) = reject_parts(&action);
    assert_eq!(
        status, 400,
        "a selector whose kind disagrees with the declared member is a bad request"
    );
    assert!(
        message.contains("commit"),
        "the rejection names the mismatched member: {message}"
    );
    assert_eq!(
        state.request_body["tool_choice"]["type"], "custom",
        "tool_choice left untouched on reject"
    );
}

#[test]
fn namespaced_function_selector_for_custom_member_is_rejected() {
    // Conversely, a `type:"function"` selector must not force a declared namespaced
    // *custom* member.
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{
            "type": "namespace",
            "name": "git",
            "description": "Git operations.",
            "tools": [{"type": "custom", "name": "freeform"}]
        }],
        "tool_choice": {"type": "function", "name": "freeform", "namespace": "git"},
    }));
    let action = filter()
        .lower_request(&mut state, false)
        .expect_err("a cross-kind namespaced selector must fail closed");
    let (status, message) = reject_parts(&action);
    assert_eq!(
        status, 400,
        "a selector whose kind disagrees with the declared member is a bad request"
    );
    assert!(
        message.contains("freeform"),
        "the rejection names the mismatched member: {message}"
    );
    assert_eq!(
        state.request_body["tool_choice"]["type"], "function",
        "tool_choice left untouched on reject"
    );
}

// -----------------------------------------------------------------------------
// Continuation status preservation (#1158 review F5)
// -----------------------------------------------------------------------------

#[test]
fn lowering_history_preserves_incomplete_status() {
    // `apply_message_status` must preserve all three `FunctionCallStatus` values; an
    // `incomplete` historical client-owned call must reach the backend with its
    // terminal state, not as a status-less call.
    let mut messages = vec![
        json!({
            "type": "shell_call", "call_id": "s1", "action": {"commands": ["ls"]},
            "environment": {"type": "local"}, "id": "sh_1", "status": "incomplete"
        }),
        json!({
            "type": "tool_search_call", "call_id": "t1", "execution": "client",
            "arguments": {"query": "x"}, "id": "tsc_1", "status": "incomplete"
        }),
        json!({
            "type": "custom_tool_call", "call_id": "c1", "name": "apply_patch",
            "input": "patch", "id": "ctc_1", "status": "incomplete"
        }),
    ];
    assert!(
        lower_history_items(&mut messages).expect("supported history lowers without error"),
        "client-owned items are lowered"
    );
    for (idx, item) in messages.iter().enumerate() {
        assert_eq!(
            item["type"], "function_call",
            "item {idx} lowered to a function_call: {item}"
        );
        assert_eq!(
            item["status"], "incomplete",
            "item {idx} keeps its incomplete status across the continuation: {item}"
        );
    }
}

// -----------------------------------------------------------------------------
// Regression: a withheld deferred tool must never dangle a forced tool_choice.
// -----------------------------------------------------------------------------

#[test]
fn deferred_top_level_function_forced_by_tool_choice_fails_closed() {
    // Regression (Finding A): a top-level `function` declared `defer_loading: true`
    // alongside a rich `custom` tool is withheld from the outbound callable set. A
    // `tool_choice` forcing that withheld function must fail closed with a
    // deferred-specific rejection rather than dangling a selector for a tool the
    // backend never receives.
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [
            {"type": "custom", "name": "c", "format": {"type": "text"}},
            {"type": "function", "name": "grep", "parameters": {"type": "object"}, "defer_loading": true}
        ],
        "tool_choice": {"type": "function", "name": "grep"},
    }));
    let action = filter()
        .lower_request(&mut state, false)
        .expect_err("forcing a withheld deferred function must fail closed");
    let (status, message) = reject_parts(&action);
    assert_eq!(status, 400, "a dangling deferred selector is a bad request");
    assert!(
        message.contains("grep") && message.contains("deferred"),
        "the rejection names the deferred tool and explains the deferral: {message}"
    );
    let tools = state.request_body["tools"].as_array().expect("tools restored");
    assert_eq!(tools.len(), 2, "the client's original tools are restored on reject");
    assert!(
        tools
            .iter()
            .any(|tool| tool["name"] == "grep" && tool["defer_loading"] == true),
        "the deferred function is restored verbatim: {tools:?}"
    );
    assert_eq!(
        state.request_body["tool_choice"]["type"], "function",
        "tool_choice is left untouched on reject"
    );
}

#[test]
fn deferred_top_level_function_reloaded_by_discovery_can_be_forced() {
    // Finding A companion: a deferred top-level function that a prior client
    // `tool_search` re-discovered is callable again, so a `tool_choice` forcing it
    // must forward verbatim. The deferred-selector fail-closed only fires for a
    // withheld name discovery never re-loaded (present in `callable` re-enables it).
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [
            {"type": "tool_search"},
            {"type": "function", "name": "grep", "parameters": {"type": "object"}, "defer_loading": true}
        ],
        "tool_choice": {"type": "function", "name": "grep"},
        "input": [
            {"type": "tool_search_call", "call_id": "call_ts", "execution": "client", "arguments": {}},
            {
                "type": "tool_search_output", "call_id": "call_ts", "status": "completed",
                "tools": [{"type": "function", "name": "grep", "parameters": {"type": "object"}}]
            }
        ],
    }));
    filter()
        .lower_request(&mut state, false)
        .expect("a re-discovered deferred function is callable and can be forced");
    let choice = &state.request_body["tool_choice"];
    assert_eq!(choice["type"], "function", "the selector stays a function selector");
    assert_eq!(choice["name"], "grep", "the forced name is forwarded verbatim");
    let tools = state.request_body["tools"].as_array().expect("outbound tools");
    let grep: Vec<_> = tools.iter().filter(|tool| tool["name"] == "grep").collect();
    assert_eq!(grep.len(), 1, "exactly one callable grep reaches the backend");
}

#[test]
fn deferred_top_level_custom_forced_by_tool_choice_reports_deferred() {
    // Finding D: forcing a withheld deferred `custom` tool is reported as deferred
    // (loadable via tool_search), not as an undeclared tool — the tool WAS declared,
    // its loading is merely deferred.
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [
            {"type": "custom", "name": "loader", "format": {"type": "text"}},
            {"type": "custom", "name": "apply_patch", "format": {"type": "text"}, "defer_loading": true}
        ],
        "tool_choice": {"type": "custom", "name": "apply_patch"},
    }));
    let action = filter()
        .lower_request(&mut state, false)
        .expect_err("forcing a withheld deferred custom must fail closed");
    let (status, message) = reject_parts(&action);
    assert_eq!(status, 400);
    assert!(
        message.contains("apply_patch") && message.contains("deferred"),
        "a declared-but-deferred custom is reported as deferred, not undeclared: {message}"
    );
}

#[test]
fn deferred_namespace_function_member_forced_by_tool_choice_reports_deferred() {
    // Finding A/D (namespace variant): forcing a namespaced `function` member whose
    // declaration was withheld for `defer_loading` reports it as deferred rather than
    // dangling the flat private selector for a member the backend never receives.
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{
            "type": "namespace",
            "name": "git",
            "description": "Git tools.",
            "tools": [{"type": "function", "name": "commit", "parameters": {"type": "object"}, "defer_loading": true}]
        }],
        "tool_choice": {"type": "function", "name": "commit", "namespace": "git"},
    }));
    let action = filter()
        .lower_request(&mut state, false)
        .expect_err("forcing a withheld deferred namespace member must fail closed");
    let (status, message) = reject_parts(&action);
    assert_eq!(status, 400);
    assert!(
        message.contains("commit") && message.contains("deferred"),
        "the withheld member is reported as deferred, not undeclared: {message}"
    );
}

// -----------------------------------------------------------------------------
// Regression: dedup identity drops only decorators lowering consumes.
// -----------------------------------------------------------------------------

#[test]
fn discovered_custom_redefinition_differing_only_in_defer_loading_is_hoisted_once() {
    // Findings B/C: two client `tool_search` listings of the same (type, name)
    // `custom` tool differing ONLY in the Responses-only `defer_loading` hint lower
    // to a byte-identical backend function. The discovery is unambiguous — it is
    // deduplicated and hoisted once rather than falsely rejected as a conflict.
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "tool_search"}],
        "input": [
            {"type": "tool_search_call", "call_id": "call_a", "execution": "client", "arguments": {}},
            {
                "type": "tool_search_output", "call_id": "call_a", "status": "completed",
                "tools": [{
                    "type": "custom", "name": "apply_patch",
                    "description": "Apply a patch.", "format": {"type": "text"},
                    "defer_loading": true
                }]
            },
            {"type": "tool_search_call", "call_id": "call_b", "execution": "client", "arguments": {}},
            {
                "type": "tool_search_output", "call_id": "call_b", "status": "completed",
                "tools": [{
                    "type": "custom", "name": "apply_patch",
                    "description": "Apply a patch.", "format": {"type": "text"}
                }]
            },
        ],
    }));
    filter()
        .lower_request(&mut state, false)
        .expect("custom listings differing only in defer_loading are not a conflict");
    let tools = state.request_body["tools"].as_array().expect("outbound tools");
    let apply_patch: Vec<_> = tools.iter().filter(|tool| tool["name"] == "apply_patch").collect();
    assert_eq!(
        apply_patch.len(),
        1,
        "a custom listing differing only in defer_loading is hoisted once"
    );
    assert_eq!(
        apply_patch[0]["type"], "function",
        "the surviving copy is the lowered discovered custom"
    );
    assert!(
        state.client_tool_lowering.contains_key("apply_patch"),
        "the hoisted custom is callable and registered for restoration"
    );
}

#[test]
fn discovered_namespace_member_redefinition_differing_only_in_defer_loading_is_hoisted_once() {
    // Findings B/C (namespace variant): two `tool_search` listings of the same
    // namespace differing ONLY in a member's `defer_loading` hint lower identically
    // (discovery consumes `defer_loading` per member), so they dedup to one hoisted
    // member rather than a false conflict. Exercises the per-member decorator strip.
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "tool_search"}],
        "input": [
            {"type": "tool_search_call", "call_id": "call_a", "execution": "client", "arguments": {}},
            {
                "type": "tool_search_output", "call_id": "call_a", "status": "completed",
                "tools": [{
                    "type": "namespace", "name": "git", "description": "Git tools.",
                    "tools": [{"type": "function", "name": "commit", "parameters": {"type": "object"}, "defer_loading": true}]
                }]
            },
            {"type": "tool_search_call", "call_id": "call_b", "execution": "client", "arguments": {}},
            {
                "type": "tool_search_output", "call_id": "call_b", "status": "completed",
                "tools": [{
                    "type": "namespace", "name": "git", "description": "Git tools.",
                    "tools": [{"type": "function", "name": "commit", "parameters": {"type": "object"}}]
                }]
            },
        ],
    }));
    filter()
        .lower_request(&mut state, false)
        .expect("namespace listings differing only in a member defer_loading are not a conflict");
    let tools = state.request_body["tools"].as_array().expect("outbound tools");
    let members: Vec<_> = tools
        .iter()
        .filter(|tool| tool["name"] == "agentic_ns__git__commit")
        .collect();
    assert_eq!(
        members.len(),
        1,
        "a namespace differing only in a member defer_loading is hoisted once"
    );
    assert!(
        members[0].get("defer_loading").is_none(),
        "the hoisted member carries no defer_loading hint"
    );
}

#[test]
fn discovered_namespace_function_member_redefinition_differing_only_in_null_output_schema_is_hoisted_once() {
    // Round-8 finding: the per-member identity must mirror `sanitize_backend_function`,
    // which lowers a `null`/absent `output_schema` on a function member to omission.
    // Two `tool_search` listings of the same namespace whose one function member
    // differs only in `output_schema` null vs absent lower to a byte-identical backend
    // function, so they dedup and hoist once instead of a false 400 conflict.
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "tool_search"}],
        "input": [
            {"type": "tool_search_call", "call_id": "call_a", "execution": "client", "arguments": {}},
            {
                "type": "tool_search_output", "call_id": "call_a", "status": "completed",
                "tools": [{
                    "type": "namespace", "name": "utils", "description": "Utility tools.",
                    "tools": [{"type": "function", "name": "run", "parameters": {"type": "object"}, "output_schema": null}]
                }]
            },
            {"type": "tool_search_call", "call_id": "call_b", "execution": "client", "arguments": {}},
            {
                "type": "tool_search_output", "call_id": "call_b", "status": "completed",
                "tools": [{
                    "type": "namespace", "name": "utils", "description": "Utility tools.",
                    "tools": [{"type": "function", "name": "run", "parameters": {"type": "object"}}]
                }]
            },
        ],
    }));
    filter()
        .lower_request(&mut state, false)
        .expect("namespace function members differing only in a null/absent output_schema are not a conflict");
    let tools = state.request_body["tools"].as_array().expect("outbound tools");
    let members: Vec<_> = tools
        .iter()
        .filter(|tool| tool["name"] == "agentic_ns__utils__run")
        .collect();
    assert_eq!(
        members.len(),
        1,
        "a namespace function member differing only in a null/absent output_schema is hoisted once"
    );
    assert!(
        members[0].get("output_schema").is_none(),
        "the hoisted member drops the null output_schema, matching sanitize_backend_function"
    );
}

#[test]
fn discovered_namespace_function_member_malformed_output_schema_twin_fails_closed_regardless_of_order() {
    // Order-independence guard for the member output_schema normalization: only a
    // `null`/absent `output_schema` leaves the identity. A namespace function member
    // with a reject-worthy (non-object, non-null) `output_schema` keeps it, so a clean
    // member (no schema) and a would-reject twin of the same namespace never collapse —
    // the request fails closed whichever listing is discovered first.
    let clean = json!({
        "type": "namespace", "name": "utils", "description": "Utility tools.",
        "tools": [{"type": "function", "name": "run", "parameters": {"type": "object"}}]
    });
    let reject_worthy = json!({
        "type": "namespace", "name": "utils", "description": "Utility tools.",
        "tools": [{"type": "function", "name": "run", "parameters": {"type": "object"}, "output_schema": "not-an-object"}]
    });
    for (first, second, order) in [
        (clean.clone(), reject_worthy.clone(), "clean-first"),
        (reject_worthy.clone(), clean.clone(), "reject-worthy-first"),
    ] {
        let mut state = ResponsesState::from_request_body(json!({
            "tools": [{"type": "tool_search"}],
            "input": [
                {"type": "tool_search_call", "call_id": "call_a", "execution": "client", "arguments": {}},
                {"type": "tool_search_output", "call_id": "call_a", "status": "completed", "tools": [first]},
                {"type": "tool_search_call", "call_id": "call_b", "execution": "client", "arguments": {}},
                {"type": "tool_search_output", "call_id": "call_b", "status": "completed", "tools": [second]},
            ],
        }));
        let action = filter()
            .lower_request(&mut state, false)
            .expect_err("clean vs malformed output_schema member must fail closed");
        let (status, message) = reject_parts(&action);
        assert_eq!(
            status, 400,
            "a reject-worthy output_schema member twin is a bad request ({order})"
        );
        assert!(
            message.contains("utils") && message.contains("conflicting"),
            "clean vs malformed output_schema member is an ambiguous conflict ({order}): {message}"
        );
    }
}

#[test]
fn discovered_namespace_function_member_redefinition_differing_only_in_unknown_field_is_hoisted_once() {
    // Root-cause guard: a function member is lowered by `sanitize_backend_function`, a
    // WHITELIST that drops any field outside type/name/description/parameters/strict/
    // (object) output_schema. Its conflict identity must mirror that whitelist, so two
    // listings of the same namespace whose function member differs only in an unknown
    // extra field lower identically and dedup once instead of a false 400 conflict.
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "tool_search"}],
        "input": [
            {"type": "tool_search_call", "call_id": "call_a", "execution": "client", "arguments": {}},
            {
                "type": "tool_search_output", "call_id": "call_a", "status": "completed",
                "tools": [{
                    "type": "namespace", "name": "utils", "description": "Utility tools.",
                    "tools": [{"type": "function", "name": "run", "parameters": {"type": "object"}, "x_vendor_meta": {"a": 1}}]
                }]
            },
            {"type": "tool_search_call", "call_id": "call_b", "execution": "client", "arguments": {}},
            {
                "type": "tool_search_output", "call_id": "call_b", "status": "completed",
                "tools": [{
                    "type": "namespace", "name": "utils", "description": "Utility tools.",
                    "tools": [{"type": "function", "name": "run", "parameters": {"type": "object"}, "x_vendor_meta": {"a": 2}}]
                }]
            },
        ],
    }));
    filter()
        .lower_request(&mut state, false)
        .expect("namespace function members differing only in an unwhitelisted field are not a conflict");
    let tools = state.request_body["tools"].as_array().expect("outbound tools");
    let members: Vec<_> = tools
        .iter()
        .filter(|tool| tool["name"] == "agentic_ns__utils__run")
        .collect();
    assert_eq!(
        members.len(),
        1,
        "a namespace function member differing only in an unwhitelisted field is hoisted once"
    );
    assert!(
        members[0].get("x_vendor_meta").is_none(),
        "the hoisted member drops the unwhitelisted field, matching sanitize_backend_function"
    );
}

#[test]
fn discovered_custom_redefinition_differing_in_description_still_conflicts() {
    // Guard for Findings B/C: stripping `defer_loading` from the conflict identity
    // must not over-broaden dedup. Two `custom` listings of the same (type, name)
    // that differ in an outcome-affecting field (the `description` folded into the
    // lowered function's prose) remain an ambiguous conflict and fail closed.
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "tool_search"}],
        "input": [
            {"type": "tool_search_call", "call_id": "call_a", "execution": "client", "arguments": {}},
            {
                "type": "tool_search_output", "call_id": "call_a", "status": "completed",
                "tools": [{"type": "custom", "name": "apply_patch", "description": "Apply a patch.", "format": {"type": "text"}}]
            },
            {"type": "tool_search_call", "call_id": "call_b", "execution": "client", "arguments": {}},
            {
                "type": "tool_search_output", "call_id": "call_b", "status": "completed",
                "tools": [{"type": "custom", "name": "apply_patch", "description": "Revert a patch.", "format": {"type": "text"}}]
            },
        ],
    }));
    let action = filter()
        .lower_request(&mut state, false)
        .expect_err("custom listings differing in an outcome-affecting field must fail closed");
    let (status, message) = reject_parts(&action);
    assert_eq!(status, 400, "an ambiguous discovery is a bad request");
    assert!(
        message.contains("apply_patch") && message.contains("conflicting"),
        "the differing custom listings are an ambiguous conflict: {message}"
    );
}

#[test]
fn discovered_custom_redefinition_differing_only_in_accepted_format_is_hoisted_once() {
    // Non-function dedup identity mirrors the lowering outcome: an accepted custom
    // `format` (absent or `{"type":"text"}`) is only validated, never read, so two
    // listings of the same (type, name) `custom` that differ only there lower to a
    // byte-identical backend function. They dedup and hoist once instead of a false
    // 400 conflict. (One listing omits `format`, the other declares the text default.)
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "tool_search"}],
        "input": [
            {"type": "tool_search_call", "call_id": "call_a", "execution": "client", "arguments": {}},
            {
                "type": "tool_search_output", "call_id": "call_a", "status": "completed",
                "tools": [{"type": "custom", "name": "apply_patch", "description": "Apply a patch.", "format": {"type": "text"}}]
            },
            {"type": "tool_search_call", "call_id": "call_b", "execution": "client", "arguments": {}},
            {
                "type": "tool_search_output", "call_id": "call_b", "status": "completed",
                "tools": [{"type": "custom", "name": "apply_patch", "description": "Apply a patch."}]
            },
        ],
    }));
    filter()
        .lower_request(&mut state, false)
        .expect("custom listings differing only in an accepted format are not a conflict");
    let tools = state.request_body["tools"].as_array().expect("outbound tools");
    let apply_patch: Vec<_> = tools.iter().filter(|tool| tool["name"] == "apply_patch").collect();
    assert_eq!(
        apply_patch.len(),
        1,
        "a custom listing differing only in an accepted format is hoisted once"
    );
    assert_eq!(
        apply_patch[0]["type"], "function",
        "the surviving copy is the lowered discovered custom"
    );
    assert!(
        state.client_tool_lowering.contains_key("apply_patch"),
        "the hoisted custom is callable and registered for restoration"
    );
}

#[test]
fn discovered_custom_reject_worthy_format_twin_fails_closed_regardless_of_order() {
    // Order-independence guard for the format normalization: only an *accepted*
    // format leaves the conflict identity. A reject-worthy custom `format` (any
    // non-`text` shape) stays in the identity, so a clean `{"type":"text"}` listing
    // and a would-reject grammar twin of the same (type, name) never collapse — the
    // request fails closed as a conflict whichever listing is discovered first.
    let clean =
        json!({"type": "custom", "name": "apply_patch", "description": "Apply a patch.", "format": {"type": "text"}});
    let reject_worthy = json!({
        "type": "custom", "name": "apply_patch", "description": "Apply a patch.",
        "format": {"type": "grammar", "syntax": "lark", "definition": "start: TOKEN"}
    });
    for (first, second, order) in [
        (clean.clone(), reject_worthy.clone(), "clean-first"),
        (reject_worthy.clone(), clean.clone(), "reject-worthy-first"),
    ] {
        let mut state = ResponsesState::from_request_body(json!({
            "tools": [{"type": "tool_search"}],
            "input": [
                {"type": "tool_search_call", "call_id": "call_a", "execution": "client", "arguments": {}},
                {"type": "tool_search_output", "call_id": "call_a", "status": "completed", "tools": [first]},
                {"type": "tool_search_call", "call_id": "call_b", "execution": "client", "arguments": {}},
                {"type": "tool_search_output", "call_id": "call_b", "status": "completed", "tools": [second]},
            ],
        }));
        let action = filter()
            .lower_request(&mut state, false)
            .expect_err("clean vs reject-worthy format must fail closed");
        let (status, message) = reject_parts(&action);
        assert_eq!(status, 400, "a reject-worthy format twin is a bad request ({order})");
        assert!(
            message.contains("apply_patch") && message.contains("conflicting"),
            "clean vs reject-worthy format is an ambiguous conflict ({order}): {message}"
        );
    }
}

#[test]
fn discovered_custom_redefinition_differing_only_in_null_allowed_callers_is_hoisted_once() {
    // The dedup identity drops a `null`/absent `allowed_callers` (accepted; lowers to
    // omission) just as the function branch does, so two custom listings differing
    // only in an explicit-null vs omitted restriction lower identically and hoist
    // once. A non-`null` restriction still fails closed (covered elsewhere).
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "tool_search"}],
        "input": [
            {"type": "tool_search_call", "call_id": "call_a", "execution": "client", "arguments": {}},
            {
                "type": "tool_search_output", "call_id": "call_a", "status": "completed",
                "tools": [{"type": "custom", "name": "apply_patch", "description": "Apply a patch.", "allowed_callers": null}]
            },
            {"type": "tool_search_call", "call_id": "call_b", "execution": "client", "arguments": {}},
            {
                "type": "tool_search_output", "call_id": "call_b", "status": "completed",
                "tools": [{"type": "custom", "name": "apply_patch", "description": "Apply a patch."}]
            },
        ],
    }));
    filter()
        .lower_request(&mut state, false)
        .expect("custom listings differing only in a null/absent allowed_callers are not a conflict");
    let tools = state.request_body["tools"].as_array().expect("outbound tools");
    let apply_patch: Vec<_> = tools.iter().filter(|tool| tool["name"] == "apply_patch").collect();
    assert_eq!(
        apply_patch.len(),
        1,
        "a custom listing differing only in a null/absent allowed_callers is hoisted once"
    );
    assert!(
        state.client_tool_lowering.contains_key("apply_patch"),
        "the hoisted custom is callable and registered for restoration"
    );
}

// -----------------------------------------------------------------------------
// Fix B: per-kind description normalization in the discovered-tool conflict identity
// -----------------------------------------------------------------------------

/// Build a two-listing `tool_search` discovery request that lists `first` then
/// `second` (both the same discovered `(type, name)`), so the dedup/conflict
/// identity decides whether they hoist once or fail closed.
fn discovery_of(first: Value, second: Value) -> ResponsesState {
    let mut body = json!({
        "tools": [{"type": "tool_search"}],
        "input": [
            {"type": "tool_search_call", "call_id": "call_a", "execution": "client", "arguments": {}},
            {"type": "tool_search_output", "call_id": "call_a", "status": "completed", "tools": []},
            {"type": "tool_search_call", "call_id": "call_b", "execution": "client", "arguments": {}},
            {"type": "tool_search_output", "call_id": "call_b", "status": "completed", "tools": []},
        ],
    });
    body["input"][1]["tools"] = Value::Array(vec![first]);
    body["input"][3]["tools"] = Value::Array(vec![second]);
    ResponsesState::from_request_body(body)
}

/// Assert two equivalent discovered listings dedup: `hoisted_name` appears exactly
/// once in the outbound tools as a lowered `function`.
fn assert_discovery_hoisted_once(first: Value, second: Value, hoisted_name: &str, case: &str) {
    let mut state = discovery_of(first, second);
    filter()
        .lower_request(&mut state, false)
        .unwrap_or_else(|_| panic!("equivalent listings must dedup, not conflict ({case})"));
    let tools = state.request_body["tools"].as_array().expect("outbound tools");
    let hits: Vec<_> = tools.iter().filter(|tool| tool["name"] == hoisted_name).collect();
    assert_eq!(hits.len(), 1, "equivalent listings are hoisted once ({case})");
    assert_eq!(hits[0]["type"], "function", "the surviving copy is lowered ({case})");
}

/// Assert two discovered listings fail closed as an ambiguous conflict (HTTP 400
/// naming `needle`), regardless of which listing is discovered first.
fn assert_discovery_conflicts_both_orders(a: Value, b: Value, needle: &str, case: &str) {
    for (first, second, order) in [(a.clone(), b.clone(), "a-first"), (b, a, "b-first")] {
        let mut state = discovery_of(first, second);
        let action = filter()
            .lower_request(&mut state, false)
            .expect_err("conflicting listings must fail closed");
        let (status, message) = reject_parts(&action);
        assert_eq!(
            status, 400,
            "a conflicting discovery is a bad request ({case}, {order})"
        );
        assert!(
            message.contains(needle) && message.contains("conflicting"),
            "differing listings are an ambiguous conflict ({case}, {order}): {message}"
        );
    }
}

#[test]
fn conflict_identity_mirrors_per_kind_description_folding() {
    // Fix B: the hand-built conflict identity must mirror how each discovered kind's
    // lowering folds its `description`, so two listings that lower to the same
    // model-visible text dedup (equal identity) while materially different text stays a
    // conflict (distinct identity). Each kind folds differently, so this is verified per
    // kind — including the two whose descriptions lowering keeps VERBATIM (top-level
    // `function` and `tool_search`), which must NOT be trimmed here (that would collapse
    // two listings that genuinely lower to different text — a wrong dedup).
    let func = |description: Value| {
        let mut member = json!({"type": "function", "name": "run", "parameters": {"type": "object"}});
        member
            .as_object_mut()
            .unwrap()
            .insert("description".to_owned(), description);
        member
    };
    let ns = |member: Value, description: Value| json!({"type": "namespace", "name": "grp", "description": description, "tools": [member]});
    let run = json!({"type": "function", "name": "run", "parameters": {"type": "object"}});
    let apply = |description: Value| {
        let mut member = json!({"type": "custom", "name": "apply"});
        member
            .as_object_mut()
            .unwrap()
            .insert("description".to_owned(), description);
        member
    };
    let cases: Vec<(&str, Value, Value, bool)> = vec![
        // top-level custom: trimmed, dropped-when-empty (custom_model_visible_description).
        (
            "custom: trailing-space description folds equal",
            json!({"type": "custom", "name": "c", "description": "Do it. "}),
            json!({"type": "custom", "name": "c", "description": "Do it."}),
            true,
        ),
        (
            "custom: null and absent description both fold to omitted",
            json!({"type": "custom", "name": "c", "description": null}),
            json!({"type": "custom", "name": "c"}),
            true,
        ),
        (
            "custom: whitespace-only and absent description both fold to omitted",
            json!({"type": "custom", "name": "c", "description": "   "}),
            json!({"type": "custom", "name": "c"}),
            true,
        ),
        (
            "custom: materially different description stays a conflict",
            json!({"type": "custom", "name": "c", "description": "Apply."}),
            json!({"type": "custom", "name": "c", "description": "Revert."}),
            false,
        ),
        // namespace top-level description: trimmed (namespace_header).
        (
            "namespace: trailing-space description folds equal",
            ns(run.clone(), json!("Group. ")),
            ns(run.clone(), json!("Group.")),
            true,
        ),
        (
            "namespace: materially different description stays a conflict",
            ns(run.clone(), json!("Group A.")),
            ns(run.clone(), json!("Group B.")),
            false,
        ),
        // namespace function member: null/absent coerced to "" then trimmed (lower_namespace_member).
        (
            "namespace function member: trailing-space description folds equal",
            ns(func(json!("Run. ")), json!("Group.")),
            ns(func(json!("Run.")), json!("Group.")),
            true,
        ),
        (
            "namespace function member: null and absent description both fold to empty",
            ns(func(json!(null)), json!("Group.")),
            ns(run.clone(), json!("Group.")),
            true,
        ),
        (
            "namespace function member: materially different description stays a conflict",
            ns(func(json!("Run.")), json!("Group.")),
            ns(func(json!("Stop.")), json!("Group.")),
            false,
        ),
        // namespace custom member: trimmed, dropped-when-empty (lower_namespace_custom_member).
        (
            "namespace custom member: trailing-space description folds equal",
            ns(apply(json!("Apply. ")), json!("Group.")),
            ns(apply(json!("Apply.")), json!("Group.")),
            true,
        ),
        (
            "namespace custom member: null and absent description both fold to omitted",
            ns(apply(json!(null)), json!("Group.")),
            ns(json!({"type": "custom", "name": "apply"}), json!("Group.")),
            true,
        ),
        (
            "namespace custom member: materially different description stays a conflict",
            ns(apply(json!("Apply.")), json!("Group.")),
            ns(apply(json!("Revert.")), json!("Group.")),
            false,
        ),
        // Regression locks: descriptions lowering keeps VERBATIM must NOT be trimmed here.
        (
            "top-level function: verbatim description keeps trailing-space distinct (no trim)",
            func(json!("Run. ")),
            func(json!("Run.")),
            false,
        ),
        (
            "tool_search: verbatim description keeps trailing-space distinct (no trim)",
            json!({"type": "tool_search", "description": "Search. "}),
            json!({"type": "tool_search", "description": "Search."}),
            false,
        ),
    ];
    for (case, a, b, expect_equal) in cases {
        let repr_a = discovered_conflict_repr(&a);
        let repr_b = discovered_conflict_repr(&b);
        assert_eq!(
            repr_a == repr_b,
            expect_equal,
            "conflict identity mismatch for case '{case}':\n  a = {repr_a}\n  b = {repr_b}"
        );
    }
}

#[test]
fn discovered_custom_equivalent_descriptions_are_hoisted_once() {
    // Fix B (top-level custom): a `custom` folds a trimmed, dropped-when-empty
    // description into its lowered prose, so listings differing only by trailing
    // whitespace, or by null vs absent, lower identically and hoist once end-to-end.
    assert_discovery_hoisted_once(
        json!({"type": "custom", "name": "apply_patch", "description": "Apply a patch. ", "format": {"type": "text"}}),
        json!({"type": "custom", "name": "apply_patch", "description": "Apply a patch.", "format": {"type": "text"}}),
        "apply_patch",
        "custom-trailing-space",
    );
    assert_discovery_hoisted_once(
        json!({"type": "custom", "name": "apply_patch", "description": null}),
        json!({"type": "custom", "name": "apply_patch"}),
        "apply_patch",
        "custom-null-vs-absent",
    );
}

#[test]
fn discovered_namespace_description_equivalent_is_hoisted_once() {
    // Fix B (namespace top-level): `namespace_header` trims the required description
    // before folding it into each member, so listings differing only by trailing
    // whitespace on the namespace description lower identically and hoist once.
    assert_discovery_hoisted_once(
        json!({"type": "namespace", "name": "git", "description": "Git tools. ",
               "tools": [{"type": "function", "name": "commit", "parameters": {"type": "object"}}]}),
        json!({"type": "namespace", "name": "git", "description": "Git tools.",
               "tools": [{"type": "function", "name": "commit", "parameters": {"type": "object"}}]}),
        "agentic_ns__git__commit",
        "namespace-description-trailing-space",
    );
}

#[test]
fn discovered_namespace_function_member_equivalent_descriptions_are_hoisted_once() {
    // Fix B (namespace function member): `lower_namespace_member` coerces a null/absent
    // member description to "" and trims it, so listings differing only by trailing
    // whitespace, or by null vs absent, on the member description hoist once.
    let ns = |member: Value| json!({"type": "namespace", "name": "utils", "description": "Utility tools.", "tools": [member]});
    assert_discovery_hoisted_once(
        ns(json!({"type": "function", "name": "run", "parameters": {"type": "object"}, "description": "Run it. "})),
        ns(json!({"type": "function", "name": "run", "parameters": {"type": "object"}, "description": "Run it."})),
        "agentic_ns__utils__run",
        "member-trailing-space",
    );
    assert_discovery_hoisted_once(
        ns(json!({"type": "function", "name": "run", "parameters": {"type": "object"}, "description": null})),
        ns(json!({"type": "function", "name": "run", "parameters": {"type": "object"}})),
        "agentic_ns__utils__run",
        "member-null-vs-absent",
    );
}

#[test]
fn discovered_namespace_custom_member_equivalent_descriptions_are_hoisted_once() {
    // Fix B (namespace custom member): `lower_namespace_custom_member` folds a trimmed,
    // dropped-when-empty member description, so listings differing only by trailing
    // whitespace, or by null vs absent, hoist once.
    let ns =
        |member: Value| json!({"type": "namespace", "name": "patch", "description": "Patch tools.", "tools": [member]});
    assert_discovery_hoisted_once(
        ns(json!({"type": "custom", "name": "apply", "description": "Apply. "})),
        ns(json!({"type": "custom", "name": "apply", "description": "Apply."})),
        "agentic_ns__patch__apply",
        "custom-member-trailing-space",
    );
    assert_discovery_hoisted_once(
        ns(json!({"type": "custom", "name": "apply", "description": null})),
        ns(json!({"type": "custom", "name": "apply"})),
        "agentic_ns__patch__apply",
        "custom-member-null-vs-absent",
    );
}

#[test]
fn discovered_description_materially_different_conflicts_per_kind() {
    // Fix B guard: normalizing the description in the conflict identity must not
    // over-broaden dedup — a materially different (non-whitespace) description lowers to
    // different model-visible text, so it stays an ambiguous conflict that fails closed
    // regardless of listing order, for every kind whose description is folded into prose.
    let ns = |member: Value| json!({"type": "namespace", "name": "toolgrp", "description": "Tool group.", "tools": [member]});
    assert_discovery_conflicts_both_orders(
        json!({"type": "custom", "name": "apply_patch", "description": "Apply.", "format": {"type": "text"}}),
        json!({"type": "custom", "name": "apply_patch", "description": "Revert.", "format": {"type": "text"}}),
        "apply_patch",
        "top-level-custom",
    );
    assert_discovery_conflicts_both_orders(
        json!({"type": "namespace", "name": "toolgrp", "description": "Group A.",
               "tools": [{"type": "function", "name": "run", "parameters": {"type": "object"}}]}),
        json!({"type": "namespace", "name": "toolgrp", "description": "Group B.",
               "tools": [{"type": "function", "name": "run", "parameters": {"type": "object"}}]}),
        "toolgrp",
        "namespace-description",
    );
    assert_discovery_conflicts_both_orders(
        ns(json!({"type": "function", "name": "run", "parameters": {"type": "object"}, "description": "Run."})),
        ns(json!({"type": "function", "name": "run", "parameters": {"type": "object"}, "description": "Stop."})),
        "toolgrp",
        "namespace-function-member",
    );
    assert_discovery_conflicts_both_orders(
        ns(json!({"type": "custom", "name": "apply", "description": "Apply."})),
        ns(json!({"type": "custom", "name": "apply", "description": "Revert."})),
        "toolgrp",
        "namespace-custom-member",
    );
}

#[test]
fn discovered_namespace_custom_member_reject_worthy_format_twin_fails_closed_regardless_of_order() {
    // Order-independence guard for the namespace custom member path: an equal
    // (folded-identical) description must not let a reject-worthy `format` twin dedup
    // behind a clean one. The description normalization touches only `description`, so a
    // clean text-format member and a would-reject grammar-format member of the same
    // namespace never collapse — the request fails closed whichever is discovered first.
    let ns = |format: Value| {
        json!({"type": "namespace", "name": "toolgrp", "description": "Tool group.",
               "tools": [{"type": "custom", "name": "apply", "description": "Apply.", "format": format}]})
    };
    assert_discovery_conflicts_both_orders(
        ns(json!({"type": "text"})),
        ns(json!({"type": "grammar", "syntax": "lark", "definition": "start: TOKEN"})),
        "toolgrp",
        "namespace-custom-member-format",
    );
}

// -----------------------------------------------------------------------------
// Response restoration arming (P1)
// -----------------------------------------------------------------------------

#[tokio::test]
async fn lowering_arms_response_buffer_and_strips_accept_encoding() {
    // A request that lowers rich client tools must buffer the upstream response and
    // strip `Accept-Encoding`: a chunked response would let early chunks reach the
    // client with lowered private function names un-restored, and a compressed body
    // would fail JSON parsing and pass through un-restored.
    let filter = filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "tools": [{"type": "custom", "name": "run_python", "description": "d", "format": {"type": "text"}}],
    })));

    let action = filter
        .on_request_body(&mut ctx, &mut None, true)
        .await
        .expect("on_request_body succeeds");
    assert!(
        matches!(action, FilterAction::Continue),
        "lowering continues to the backend"
    );

    assert!(
        ctx.extensions
            .get::<ResponsesState>()
            .unwrap()
            .client_tool_echo
            .is_some(),
        "lowering rich client tools records a restoration echo",
    );
    assert_eq!(
        ctx.response_body_mode,
        BodyMode::StreamBuffer {
            max_bytes: Some(MAX_JSON_BODY_BYTES)
        },
        "the response is buffered so restoration sees the whole body",
    );
    assert!(
        ctx.request_headers_to_remove.contains(&http::header::ACCEPT_ENCODING),
        "Accept-Encoding is stripped so a compliant backend returns plaintext JSON",
    );
}

#[tokio::test]
async fn passthrough_request_does_not_arm_buffering_or_strip_encoding() {
    // A native function-only request lowers nothing rich and records no echo, so it
    // must not pay to buffer the response or strip Accept-Encoding.
    let filter = filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "tools": [{"type": "function", "name": "f", "parameters": {"type": "object"}}],
    })));

    let action = filter
        .on_request_body(&mut ctx, &mut None, true)
        .await
        .expect("on_request_body succeeds");
    assert!(
        matches!(action, FilterAction::Continue),
        "a native request continues to the backend"
    );

    assert!(
        ctx.extensions
            .get::<ResponsesState>()
            .unwrap()
            .client_tool_echo
            .is_none(),
        "a native function-only request records no echo",
    );
    assert_eq!(
        ctx.response_body_mode,
        BodyMode::Stream,
        "no buffering is armed when nothing will be restored",
    );
    assert!(
        ctx.request_headers_to_remove.is_empty(),
        "Accept-Encoding is left intact when nothing will be restored",
    );
}

#[tokio::test]
async fn streaming_rich_request_rejects_without_arming() {
    // A streaming request with rich client tools fails closed before any upstream
    // call and must never arm response buffering.
    let filter = filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    ctx.set_metadata("openai_responses_format.stream", "true".to_owned());
    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "stream": true,
        "tools": [{"type": "custom", "name": "run_python", "description": "d", "format": {"type": "text"}}],
    })));

    let action = filter
        .on_request_body(&mut ctx, &mut None, true)
        .await
        .expect("on_request_body returns an action");
    assert_eq!(
        reject_parts(&action).0,
        400,
        "streaming rich client tools fail closed before any upstream call",
    );
    assert_eq!(
        ctx.response_body_mode,
        BodyMode::Stream,
        "a rejected streaming request never arms buffering",
    );
    assert!(
        ctx.request_headers_to_remove.is_empty(),
        "a rejected streaming request never strips Accept-Encoding",
    );
}

// -----------------------------------------------------------------------------
// Caller preservation on history lowering (P2)
// -----------------------------------------------------------------------------

#[test]
fn lowered_custom_tool_call_history_item_preserves_caller() {
    // `CustomToolCall` and the `FunctionToolCall` it lowers to both carry an optional
    // `caller`; a `program` caller has a required `caller_id`, so a full schema-valid
    // caller must survive lowering rather than being silently dropped.
    let mut item = json!({
        "type": "custom_tool_call",
        "call_id": "call_1",
        "name": "run",
        "input": "echo hi",
        "caller": {"type": "program", "caller_id": "prog_1"}
    });
    assert!(
        lower_history_item(&mut item, &HashSet::new()),
        "a custom_tool_call is lowered",
    );
    assert_eq!(item["type"], "function_call");
    assert_eq!(
        item["caller"],
        json!({"type": "program", "caller_id": "prog_1"}),
        "the caller survives lowering",
    );
}

#[test]
fn lowered_shell_call_history_item_preserves_caller() {
    let mut item = json!({
        "type": "shell_call",
        "call_id": "call_1",
        "action": {"commands": ["ls"]},
        "environment": {"type": "local"},
        "caller": {"type": "direct"}
    });
    assert!(
        lower_history_item(&mut item, &HashSet::new()),
        "a client-executed shell_call is lowered",
    );
    assert_eq!(item["type"], "function_call");
    assert_eq!(
        item["caller"],
        json!({"type": "direct"}),
        "the caller survives lowering",
    );
}

#[test]
fn lowered_custom_tool_call_history_item_omits_null_caller() {
    // `caller` is `ToolCallCaller | null`; a null caller is omitted rather than
    // forwarded, matching the output-lowering path.
    let mut item = json!({
        "type": "custom_tool_call",
        "call_id": "call_1",
        "name": "run",
        "input": "echo hi",
        "caller": null
    });
    assert!(
        lower_history_item(&mut item, &HashSet::new()),
        "a custom_tool_call is lowered",
    );
    assert!(item.get("caller").is_none(), "a null caller is omitted, not forwarded",);
}

// -----------------------------------------------------------------------------
// Caller preservation and fail-closed call_id on restoration (P2)
// -----------------------------------------------------------------------------

#[test]
fn restores_custom_call_preserves_caller() {
    let state = state_with_custom_lowered();
    let response = json!({
        "object": "response",
        "output": [{
            "type": "function_call",
            "id": "fc_abc",
            "call_id": "call_1",
            "name": "run_python",
            "arguments": "{\"input\":\"print(1)\"}",
            "status": "completed",
            "caller": {"type": "program", "caller_id": "prog_1"}
        }],
    });
    let restored = restore(&state, &response);
    assert_eq!(restored["output"][0]["type"], "custom_tool_call");
    assert_eq!(
        restored["output"][0]["caller"],
        json!({"type": "program", "caller_id": "prog_1"}),
        "the backend caller is restored onto the custom_tool_call",
    );
}

#[test]
fn restores_shell_call_preserves_caller() {
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "shell", "environment": {"type": "local"}}],
    }));
    filter().lower_request(&mut state, false).expect("lowering succeeds");
    let response = json!({
        "object": "response",
        "output": [{
            "type": "function_call",
            "id": "fc_x",
            "call_id": "call_1",
            "name": "shell",
            "arguments": "{\"commands\":[\"ls\"]}",
            "status": "completed",
            "caller": {"type": "direct"}
        }],
    });
    let restored = restore(&state, &response);
    assert_eq!(restored["output"][0]["type"], "shell_call");
    assert_eq!(
        restored["output"][0]["caller"],
        json!({"type": "direct"}),
        "the backend caller is restored onto the shell_call",
    );
}

#[test]
fn custom_call_without_call_id_fails_closed() {
    // `CustomToolCall` requires a non-null `call_id`; a backend call that omits it
    // must fail closed rather than restore to a schema-invalid `call_id: null` the
    // client cannot map to an output.
    let state = state_with_custom_lowered();
    let response = json!({
        "object": "response",
        "output": [{
            "type": "function_call",
            "id": "fc_abc",
            "name": "run_python",
            "arguments": "{\"input\":\"print(1)\"}",
            "status": "completed"
        }],
    });
    let action = filter()
        .restore_response(&state, response.to_string().as_bytes())
        .expect_err("a custom_tool_call without call_id must fail closed");
    let (status, message) = reject_parts(&action);
    assert_eq!(status, 502, "a schema-invalid custom_tool_call cannot be restored");
    assert!(
        message.contains("custom_tool_call"),
        "the rejection names custom_tool_call: {message}",
    );
}

#[test]
fn custom_call_with_blank_call_id_fails_closed() {
    let state = state_with_custom_lowered();
    let response = json!({
        "object": "response",
        "output": [{
            "type": "function_call",
            "id": "fc_abc",
            "call_id": "   ",
            "name": "run_python",
            "arguments": "{\"input\":\"print(1)\"}",
            "status": "completed"
        }],
    });
    let action = filter()
        .restore_response(&state, response.to_string().as_bytes())
        .expect_err("a blank call_id must fail closed");
    assert_eq!(reject_parts(&action).0, 502);
}

#[test]
fn namespace_custom_call_without_call_id_fails_closed() {
    // The namespaced custom member restoration shares `restore_custom_call`, so it
    // must also fail closed when the backend omits `call_id`.
    let reverse = HashMap::from([(
        "agentic_ns__git__freeform".to_owned(),
        LoweredClientTool {
            original_name: "freeform".to_owned(),
            namespace: Some("git".to_owned()),
            restore: ClientToolRestore::NamespaceCustom,
        },
    )]);
    let mut item = json!({
        "type": "function_call",
        "id": "fc_xyz",
        "name": "agentic_ns__git__freeform",
        "arguments": r#"{"input":"x"}"#,
        "status": "completed"
    });
    let action = restore_output_item(&mut item, &reverse)
        .expect_err("a namespaced custom call without call_id must fail closed");
    let (status, message) = reject_parts(&action);
    assert_eq!(status, 502);
    assert!(
        message.contains("custom_tool_call"),
        "the rejection names custom_tool_call: {message}",
    );
}
