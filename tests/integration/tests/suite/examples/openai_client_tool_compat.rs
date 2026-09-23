// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Integration tests for the `openai_client_tool_compat` example config.
//!
//! These tests drive the `client-tool-compat.yaml` pipeline end to end: a rich
//! Codex-style Responses client talks to a function-only Responses backend
//! (modeled by a capturing mock) through `POST /v1/responses`, with no
//! `/v1/chat/completions` in the chain and no client-owned tool executed inside
//! Praxis. They assert both directions of the adapter:
//!
//! - request phase: rich `custom`, `namespace`, local `shell`, and client-executed `tool_search` declarations are
//!   lowered to private `function` tools on the wire the backend receives;
//! - response phase: the backend's `function_call` output items are restored to their canonical typed items
//!   (`custom_tool_call`, namespaced `function_call`, `shell_call`, `tool_search_call`) and the original
//!   `tools`/`tool_choice` are echoed back to the client.
//!
//! A request that declares no rich client tools stays a byte-transparent
//! passthrough, so native traffic is unchanged.

use std::collections::HashMap;

use praxis_test_utils::{
    Backend, StatefulCapturingBackend, TempSqlite, build_pipeline, example_config_path, free_port, http_send,
    json_post, parse_body, parse_header, parse_status, patch_yaml, start_proxy,
};

// -----------------------------------------------------------------------------
// Config loader
// -----------------------------------------------------------------------------

/// Load the `client-tool-compat.yaml` example, pointing the proxy at `proxy_port`,
/// the inference backend at `model_port`, and the response store at a private temp
/// SQLite database so tests do not share persisted state.
fn load_client_tool_compat_config(proxy_port: u16, model_port: u16, db_url: &str) -> praxis_core::config::Config {
    let path = example_config_path("openai/responses/client-tool-compat.yaml");
    let yaml = std::fs::read_to_string(path).expect("read client-tool-compat example");
    let yaml = yaml.replace("sqlite://responses.db?mode=rwc", db_url);
    let yaml = patch_yaml(&yaml, proxy_port, &HashMap::from([("127.0.0.1:3001", model_port)]));
    praxis_core::config::Config::from_yaml(&yaml).expect("parse client-tool-compat config")
}

/// Like [`load_client_tool_compat_config`] but with the `openai_stream_events`
/// filter removed from the inference step, so the streaming-restoration marker is
/// never armed. Exercises the missing-owner fail-closed path (#1159).
fn load_client_tool_compat_config_without_stream_owner(
    proxy_port: u16,
    model_port: u16,
    db_url: &str,
) -> praxis_core::config::Config {
    let path = example_config_path("openai/responses/client-tool-compat.yaml");
    let yaml = std::fs::read_to_string(path).expect("read client-tool-compat example");
    let yaml = yaml.replace("sqlite://responses.db?mode=rwc", db_url);
    // Drop the streaming owner line; the remaining filters keep their order.
    let yaml = yaml
        .lines()
        .filter(|line| line.trim() != "- filter: openai_stream_events")
        .collect::<Vec<_>>()
        .join("\n");
    let yaml = patch_yaml(&yaml, proxy_port, &HashMap::from([("127.0.0.1:3001", model_port)]));
    praxis_core::config::Config::from_yaml(&yaml).expect("parse client-tool-compat config without stream owner")
}

// -----------------------------------------------------------------------------
// Pipeline build
// -----------------------------------------------------------------------------

#[test]
fn example_config_builds_pipeline() {
    let db = TempSqlite::new("client_tool_compat_build");
    let config = load_client_tool_compat_config(free_port(), 19951, db.url());
    let _pipeline = build_pipeline(&config);
}

// -----------------------------------------------------------------------------
// Round trip: custom tool lowered on the way out, restored on the way back
// -----------------------------------------------------------------------------

/// A `custom` client tool is lowered to a private `function` the backend accepts,
/// and the backend's `function_call` reply is restored to a `custom_tool_call`
/// with the original `tools` echoed back — all through `POST /v1/responses`.
#[test]
fn custom_tool_round_trip_lowers_and_restores() {
    let backend_response = serde_json::json!({
        "id": "resp_custom",
        "object": "response",
        "status": "completed",
        "output": [{
            "type": "function_call",
            "id": "fc_abc",
            "call_id": "call_abc",
            "name": "apply_patch",
            "arguments": r#"{"input":"*** Begin Patch"}"#,
            "status": "completed"
        }]
    });
    let model = StatefulCapturingBackend::new(vec![(200, backend_response.to_string())]).start_with_shutdown();
    let proxy_port = free_port();
    let db = TempSqlite::new("client_tool_compat_custom");
    let config = load_client_tool_compat_config(proxy_port, model.port(), db.url());
    let proxy = start_proxy(&config);

    let request = serde_json::json!({
        "model": "gpt-4.1",
        "input": "Apply the patch.",
        "tools": [{
            "type": "custom",
            "name": "apply_patch",
            "description": "Apply a unified diff to the workspace."
        }]
    });
    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", &serde_json::to_string(&request).unwrap()),
    );

    assert_eq!(parse_status(&raw), 200, "custom round trip should return 200: {raw}");

    // Request phase: the backend saw a lowered private `function`, never a
    // `custom` tool.
    let model_reqs = model.requests();
    assert_eq!(model_reqs.len(), 1, "one inference round for a single client tool call");
    let backend_body: serde_json::Value =
        serde_json::from_str(&model_reqs[0].body).expect("backend request body should be JSON");
    let backend_tools = backend_body["tools"]
        .as_array()
        .expect("backend request should carry lowered tools");
    assert_eq!(backend_tools.len(), 1, "one tool lowered: {backend_body}");
    assert_eq!(backend_tools[0]["type"], "function", "custom lowered to function");
    assert_eq!(backend_tools[0]["name"], "apply_patch", "lowered name preserved");
    assert!(
        backend_tools.iter().all(|tool| tool["type"] != "custom"),
        "no rich custom tool may reach the backend: {backend_body}"
    );

    // Response phase: the client sees the restored typed item and the echoed
    // original tools.
    let response: serde_json::Value = serde_json::from_str(&parse_body(&raw)).expect("client response should be JSON");
    assert_eq!(response["id"], "resp_custom", "backend response id preserved");
    let output = response["output"].as_array().expect("output array present");
    assert_eq!(output.len(), 1, "one restored output item: {response}");
    assert_eq!(
        output[0]["type"], "custom_tool_call",
        "function_call restored to custom"
    );
    assert_eq!(output[0]["name"], "apply_patch", "restored custom name");
    assert_eq!(output[0]["call_id"], "call_abc", "restored call_id");
    assert_eq!(output[0]["id"], "ctc_abc", "restored public custom item id");
    assert_eq!(
        output[0]["input"], "*** Begin Patch",
        "custom input unwrapped from the lowered arguments: {response}"
    );
    let echoed_tools = response["tools"].as_array().expect("echoed tools present");
    assert_eq!(
        echoed_tools[0]["type"], "custom",
        "the client sees its original custom tool, not the lowered function: {response}"
    );
}

// -----------------------------------------------------------------------------
// Round trip: local shell lowered on the way out, restored on the way back
// -----------------------------------------------------------------------------

/// A local `shell` client tool is lowered to a private `function`, and the
/// backend's `function_call` reply is restored to a `shell_call`. The backend's
/// `in_progress` status is preserved verbatim (never coerced to `completed`), the
/// optional `action` fields it omitted are normalized to explicit nulls, and the
/// call is stamped with a `local` environment so it stays client-executed.
#[test]
fn shell_tool_round_trip_lowers_and_restores() {
    let backend_response = serde_json::json!({
        "id": "resp_shell",
        "object": "response",
        "status": "completed",
        "output": [{
            "type": "function_call",
            "id": "fc_sh1",
            "call_id": "call_sh1",
            "name": "shell",
            // The lowered function omits timeout_ms/max_output_length, and the
            // backend reports the call still running.
            "arguments": r#"{"commands":["ls","-la"]}"#,
            "status": "in_progress"
        }]
    });
    let model = StatefulCapturingBackend::new(vec![(200, backend_response.to_string())]).start_with_shutdown();
    let proxy_port = free_port();
    let db = TempSqlite::new("client_tool_compat_shell");
    let config = load_client_tool_compat_config(proxy_port, model.port(), db.url());
    let proxy = start_proxy(&config);

    let request = serde_json::json!({
        "model": "gpt-4.1",
        "input": "List the files.",
        "tools": [{"type": "shell", "environment": {"type": "local"}}]
    });
    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", &serde_json::to_string(&request).unwrap()),
    );

    assert_eq!(parse_status(&raw), 200, "shell round trip should return 200: {raw}");

    // Request phase: the backend saw a private `function` named `shell`, never a
    // rich `shell`/`local_shell` tool.
    let model_reqs = model.requests();
    assert_eq!(model_reqs.len(), 1, "one inference round");
    let backend_body: serde_json::Value =
        serde_json::from_str(&model_reqs[0].body).expect("backend request body should be JSON");
    let backend_tools = backend_body["tools"]
        .as_array()
        .expect("backend request should carry lowered tools");
    assert_eq!(backend_tools[0]["type"], "function", "shell lowered to function");
    assert_eq!(backend_tools[0]["name"], "shell", "lowered shell name preserved");
    assert!(
        backend_tools
            .iter()
            .all(|tool| tool["type"] != "shell" && tool["type"] != "local_shell"),
        "no rich shell tool may reach the backend: {backend_body}"
    );

    // Response phase: the `function_call` is restored to a `shell_call`.
    let response: serde_json::Value = serde_json::from_str(&parse_body(&raw)).expect("client response should be JSON");
    let output = response["output"].as_array().expect("output array present");
    assert_eq!(output.len(), 1, "one restored output item: {response}");
    assert_eq!(output[0]["type"], "shell_call", "function_call restored to shell");
    assert_eq!(output[0]["call_id"], "call_sh1", "restored call_id");
    assert_eq!(output[0]["id"], "sh_sh1", "restored public shell item id");
    assert_eq!(
        output[0]["status"], "in_progress",
        "the backend's in_progress status is preserved, not coerced to completed: {response}"
    );
    assert_eq!(
        output[0]["environment"]["type"], "local",
        "the restored shell call is stamped local so it stays client-executed"
    );
    assert_eq!(
        output[0]["action"]["commands"],
        serde_json::json!(["ls", "-la"]),
        "shell commands round-trip through the lowered arguments"
    );
    let action = output[0]["action"].as_object().expect("action object present");
    assert!(
        action.contains_key("timeout_ms") && action["timeout_ms"].is_null(),
        "the omitted timeout_ms is normalized to an explicit null: {response}"
    );
    assert!(
        action.contains_key("max_output_length") && action["max_output_length"].is_null(),
        "the omitted max_output_length is normalized to an explicit null: {response}"
    );
    let echoed_tools = response["tools"].as_array().expect("echoed tools present");
    assert_eq!(
        echoed_tools[0]["type"], "shell",
        "the client sees its original shell tool, not the lowered function: {response}"
    );
}

// -----------------------------------------------------------------------------
// Round trip: namespace member lowered on the way out, restored on the way back
// -----------------------------------------------------------------------------

/// A `namespace` member is lowered to a flat private `function`, and the backend's
/// `function_call` reply is restored to its namespaced form with the member name
/// and `namespace` recovered and the original `namespace` tool echoed back.
#[test]
fn namespace_tool_round_trip_lowers_and_restores() {
    let backend_response = serde_json::json!({
        "id": "resp_ns",
        "object": "response",
        "status": "completed",
        "output": [{
            "type": "function_call",
            "id": "fc_ns1",
            "call_id": "call_ns1",
            "name": "agentic_ns__utils__read_file",
            "arguments": r#"{"path":"/etc/hosts"}"#,
            "status": "completed"
        }]
    });
    let model = StatefulCapturingBackend::new(vec![(200, backend_response.to_string())]).start_with_shutdown();
    let proxy_port = free_port();
    let db = TempSqlite::new("client_tool_compat_namespace");
    let config = load_client_tool_compat_config(proxy_port, model.port(), db.url());
    let proxy = start_proxy(&config);

    let request = serde_json::json!({
        "model": "gpt-4.1",
        "input": "Read the hosts file.",
        "tools": [{
            "type": "namespace",
            "name": "utils",
            "description": "Local filesystem utilities.",
            "tools": [{
                "type": "function",
                "name": "read_file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": false
                }
            }]
        }]
    });
    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", &serde_json::to_string(&request).unwrap()),
    );

    assert_eq!(parse_status(&raw), 200, "namespace round trip should return 200: {raw}");

    // Request phase: the backend saw a flat private function name.
    let model_reqs = model.requests();
    assert_eq!(model_reqs.len(), 1, "one inference round");
    let backend_body: serde_json::Value =
        serde_json::from_str(&model_reqs[0].body).expect("backend request body should be JSON");
    let backend_tools = backend_body["tools"]
        .as_array()
        .expect("backend request should carry lowered tools");
    assert_eq!(
        backend_tools[0]["type"], "function",
        "namespace member lowered to function"
    );
    assert_eq!(
        backend_tools[0]["name"], "agentic_ns__utils__read_file",
        "namespace member flattened to a private function name: {backend_body}"
    );
    assert!(
        backend_tools[0]["description"]
            .as_str()
            .is_some_and(|description| description.contains("Local filesystem utilities.")),
        "the namespace's model-visible description is folded into the flattened member: {backend_body}"
    );

    // Response phase: the flat `function_call` is restored to its namespaced form.
    let response: serde_json::Value = serde_json::from_str(&parse_body(&raw)).expect("client response should be JSON");
    let output = response["output"].as_array().expect("output array present");
    assert_eq!(output.len(), 1, "one restored output item: {response}");
    assert_eq!(
        output[0]["type"], "function_call",
        "a namespaced call restores to a function_call, not a custom item: {response}"
    );
    assert_eq!(output[0]["name"], "read_file", "restored member name");
    assert_eq!(output[0]["namespace"], "utils", "restored namespace");
    assert_eq!(output[0]["call_id"], "call_ns1", "restored call_id");
    assert_eq!(
        output[0]["arguments"], r#"{"path":"/etc/hosts"}"#,
        "namespace call arguments pass through unchanged"
    );
    let echoed_tools = response["tools"].as_array().expect("echoed tools present");
    assert_eq!(
        echoed_tools[0]["type"], "namespace",
        "the client sees its original namespace tool: {response}"
    );
}

// -----------------------------------------------------------------------------
// Round trip: namespaced custom member lowered on the way out, restored on the way back
// -----------------------------------------------------------------------------

/// A `custom` member of a `namespace` is lowered to a flat private `function`, and
/// the backend's `function_call` reply is restored to a namespaced
/// `custom_tool_call`: the member name and `namespace` are recovered, the single
/// string `input` is unwrapped, and the original `namespace` tool is echoed back
/// (#1158 requires flattening both function and custom namespace members).
#[test]
fn namespace_custom_member_round_trip_lowers_and_restores() {
    let backend_response = serde_json::json!({
        "id": "resp_ns_custom",
        "object": "response",
        "status": "completed",
        "output": [{
            "type": "function_call",
            "id": "fc_nsc1",
            "call_id": "call_nsc1",
            "name": "agentic_ns__git__freeform",
            "arguments": r#"{"input":"*** Begin Patch"}"#,
            "status": "completed"
        }]
    });
    let model = StatefulCapturingBackend::new(vec![(200, backend_response.to_string())]).start_with_shutdown();
    let proxy_port = free_port();
    let db = TempSqlite::new("client_tool_compat_namespace_custom");
    let config = load_client_tool_compat_config(proxy_port, model.port(), db.url());
    let proxy = start_proxy(&config);

    let request = serde_json::json!({
        "model": "gpt-4.1",
        "input": "Apply a freeform patch.",
        "tools": [{
            "type": "namespace",
            "name": "git",
            "description": "Version control operations.",
            "tools": [{
                "type": "custom",
                "name": "freeform",
                "description": "Apply a freeform patch."
            }]
        }]
    });
    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", &serde_json::to_string(&request).unwrap()),
    );

    assert_eq!(
        parse_status(&raw),
        200,
        "namespaced custom round trip should return 200: {raw}"
    );

    // Request phase: the backend saw a flat private function name and no rich
    // custom or namespace tool.
    let model_reqs = model.requests();
    assert_eq!(model_reqs.len(), 1, "one inference round");
    let backend_body: serde_json::Value =
        serde_json::from_str(&model_reqs[0].body).expect("backend request body should be JSON");
    let backend_tools = backend_body["tools"]
        .as_array()
        .expect("backend request should carry lowered tools");
    assert_eq!(
        backend_tools[0]["type"], "function",
        "namespaced custom member lowered to function"
    );
    assert_eq!(
        backend_tools[0]["name"], "agentic_ns__git__freeform",
        "namespaced custom member flattened to a private function name: {backend_body}"
    );
    assert!(
        backend_tools
            .iter()
            .all(|tool| tool["type"] != "custom" && tool["type"] != "namespace"),
        "no rich custom or namespace tool may reach the backend: {backend_body}"
    );

    // Response phase: the flat `function_call` is restored to a namespaced
    // `custom_tool_call`.
    let response: serde_json::Value = serde_json::from_str(&parse_body(&raw)).expect("client response should be JSON");
    let output = response["output"].as_array().expect("output array present");
    assert_eq!(output.len(), 1, "one restored output item: {response}");
    assert_eq!(
        output[0]["type"], "custom_tool_call",
        "a namespaced custom member restores to a custom_tool_call: {response}"
    );
    assert_eq!(output[0]["name"], "freeform", "restored member name");
    assert_eq!(output[0]["namespace"], "git", "restored namespace on the custom call");
    assert_eq!(output[0]["call_id"], "call_nsc1", "restored call_id");
    assert_eq!(output[0]["id"], "ctc_nsc1", "restored public custom item id");
    assert_eq!(
        output[0]["input"], "*** Begin Patch",
        "the freeform input is unwrapped from the lowered arguments: {response}"
    );
    let echoed_tools = response["tools"].as_array().expect("echoed tools present");
    assert_eq!(
        echoed_tools[0]["type"], "namespace",
        "the client sees its original namespace tool: {response}"
    );
}

// -----------------------------------------------------------------------------
// Round trip: client tool_search lowered on the way out, restored on the way back
// -----------------------------------------------------------------------------

/// A client-executed `tool_search` tool is lowered to a private `function`, and
/// the backend's `function_call` reply is restored to a `tool_search_call` with
/// `execution: client` and the original `tool_search` tool echoed back.
#[test]
fn tool_search_round_trip_lowers_and_restores() {
    let backend_response = serde_json::json!({
        "id": "resp_ts",
        "object": "response",
        "status": "completed",
        "output": [{
            "type": "function_call",
            "id": "fc_ts1",
            "call_id": "call_ts1",
            "name": "tool_search",
            "arguments": r#"{"query":"weather"}"#,
            "status": "completed"
        }]
    });
    let model = StatefulCapturingBackend::new(vec![(200, backend_response.to_string())]).start_with_shutdown();
    let proxy_port = free_port();
    let db = TempSqlite::new("client_tool_compat_tool_search");
    let config = load_client_tool_compat_config(proxy_port, model.port(), db.url());
    let proxy = start_proxy(&config);

    let request = serde_json::json!({
        "model": "gpt-4.1",
        "input": "Find a weather tool.",
        "tools": [{"type": "tool_search"}]
    });
    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", &serde_json::to_string(&request).unwrap()),
    );

    assert_eq!(
        parse_status(&raw),
        200,
        "tool_search round trip should return 200: {raw}"
    );

    // Request phase: the backend saw a private `function` named `tool_search`.
    let model_reqs = model.requests();
    assert_eq!(model_reqs.len(), 1, "one inference round");
    let backend_body: serde_json::Value =
        serde_json::from_str(&model_reqs[0].body).expect("backend request body should be JSON");
    let backend_tools = backend_body["tools"]
        .as_array()
        .expect("backend request should carry lowered tools");
    assert_eq!(backend_tools[0]["type"], "function", "tool_search lowered to function");
    assert_eq!(
        backend_tools[0]["name"], "tool_search",
        "lowered tool_search name preserved"
    );

    // Response phase: the `function_call` is restored to a `tool_search_call`.
    let response: serde_json::Value = serde_json::from_str(&parse_body(&raw)).expect("client response should be JSON");
    let output = response["output"].as_array().expect("output array present");
    assert_eq!(output.len(), 1, "one restored output item: {response}");
    assert_eq!(
        output[0]["type"], "tool_search_call",
        "function_call restored to tool_search_call"
    );
    assert_eq!(output[0]["execution"], "client", "restored client execution marker");
    assert_eq!(output[0]["call_id"], "call_ts1", "restored call_id");
    assert_eq!(output[0]["id"], "tsc_ts1", "restored public tool_search item id");
    assert_eq!(
        output[0]["arguments"],
        serde_json::json!({"query": "weather"}),
        "tool_search arguments parsed back to a structured object"
    );
    let echoed_tools = response["tools"].as_array().expect("echoed tools present");
    assert_eq!(
        echoed_tools[0]["type"], "tool_search",
        "the client sees its original tool_search tool: {response}"
    );
}

// -----------------------------------------------------------------------------
// Round trip: a discovered tool becomes callable on the continuation turn
// -----------------------------------------------------------------------------

/// A full search -> result -> discovered-call round trip. Turn 1 lowers a client
/// `tool_search` and restores the backend's `function_call` to a
/// `tool_search_call`. The client executes the search and returns a
/// `tool_search_output` carrying a discovered `custom` tool. Turn 2 must hoist the
/// discovered tool into the backend's declared functions so a function-only backend
/// can invoke it, and restore the returned `function_call` to a `custom_tool_call`.
#[test]
fn discovered_tool_round_trip_becomes_callable() {
    let search_response = serde_json::json!({
        "id": "resp_ts",
        "object": "response",
        "status": "completed",
        "output": [{
            "type": "function_call",
            "id": "fc_ts1",
            "call_id": "call_ts1",
            "name": "tool_search",
            "arguments": r#"{"query":"apply a patch"}"#,
            "status": "completed"
        }]
    });
    let discovered_call_response = serde_json::json!({
        "id": "resp_ap",
        "object": "response",
        "status": "completed",
        "output": [{
            "type": "function_call",
            "id": "fc_ap1",
            "call_id": "call_ap1",
            "name": "apply_patch",
            "arguments": r#"{"input":"*** Begin Patch"}"#,
            "status": "completed"
        }]
    });
    let model = StatefulCapturingBackend::new(vec![
        (200, search_response.to_string()),
        (200, discovered_call_response.to_string()),
    ])
    .start_with_shutdown();
    let proxy_port = free_port();
    let db = TempSqlite::new("client_tool_compat_discovered");
    let config = load_client_tool_compat_config(proxy_port, model.port(), db.url());
    let proxy = start_proxy(&config);

    // Turn 1: the client declares tool_search and receives a restored
    // tool_search_call.
    let turn1 = serde_json::json!({
        "model": "gpt-4.1",
        "input": "Find and apply a patch.",
        "tools": [{"type": "tool_search"}]
    });
    let raw1 = http_send(
        proxy.addr(),
        &json_post("/v1/responses", &serde_json::to_string(&turn1).unwrap()),
    );
    assert_eq!(parse_status(&raw1), 200, "turn 1 should return 200: {raw1}");
    let response1: serde_json::Value = serde_json::from_str(&parse_body(&raw1)).expect("turn 1 response is JSON");
    let search_call = response1["output"][0].clone();
    assert_eq!(
        search_call["type"], "tool_search_call",
        "turn 1 restores a tool_search_call: {response1}"
    );

    // Turn 2: the client executed the search and returns a discovered `custom`
    // tool, then re-declares tool_search for the continuation.
    let turn2 = serde_json::json!({
        "model": "gpt-4.1",
        "input": [
            {"type": "message", "role": "user", "content": "Find and apply a patch."},
            search_call,
            {
                "type": "tool_search_output",
                "call_id": "call_ts1",
                "status": "completed",
                "tools": [{
                    "type": "custom",
                    "name": "apply_patch",
                    "description": "Apply a unified diff.",
                    "format": {"type": "text"}
                }]
            }
        ],
        "tools": [{"type": "tool_search"}]
    });
    let raw2 = http_send(
        proxy.addr(),
        &json_post("/v1/responses", &serde_json::to_string(&turn2).unwrap()),
    );
    assert_eq!(parse_status(&raw2), 200, "turn 2 should return 200: {raw2}");

    // Request phase of turn 2: the backend must see the discovered `apply_patch`
    // tool declared as a private function so it can call it.
    let model_reqs = model.requests();
    assert_eq!(model_reqs.len(), 2, "two inference rounds");
    let backend_body: serde_json::Value =
        serde_json::from_str(&model_reqs[1].body).expect("turn 2 backend body is JSON");
    let backend_tools = backend_body["tools"].as_array().expect("turn 2 backend tools");
    let hoisted = backend_tools
        .iter()
        .find(|tool| tool["name"] == "apply_patch")
        .unwrap_or_else(|| panic!("the discovered tool is hoisted for the backend: {backend_body}"));
    assert_eq!(
        hoisted["type"], "function",
        "the discovered custom tool is lowered to a private function"
    );

    // Response phase of turn 2: the returned function_call is restored to a
    // custom_tool_call for the discovered tool.
    let response2: serde_json::Value = serde_json::from_str(&parse_body(&raw2)).expect("turn 2 response is JSON");
    let output = response2["output"].as_array().expect("turn 2 output array");
    assert_eq!(
        output[0]["type"], "custom_tool_call",
        "the discovered call is restored to a custom_tool_call: {response2}"
    );
    assert_eq!(output[0]["name"], "apply_patch", "restored discovered tool name");
    assert_eq!(output[0]["input"], "*** Begin Patch", "restored freeform input");
    let echoed_tools = response2["tools"].as_array().expect("turn 2 echoed tools");
    assert!(
        echoed_tools.iter().all(|tool| tool["name"] != "apply_patch"),
        "the discovered tool is not echoed into the response tools (it was never declared): {response2}"
    );
}

// -----------------------------------------------------------------------------
// Request phase: every rich client tool type lowers to a private function
// -----------------------------------------------------------------------------

/// A request mixing `custom`, `namespace`, local `shell`, and client-executed
/// `tool_search` declarations reaches the backend as private `function` tools
/// only — proving the outbound lowering covers all four rich shapes.
#[test]
fn all_client_tool_types_lower_to_functions() {
    let backend_response = serde_json::json!({
        "id": "resp_empty",
        "object": "response",
        "status": "completed",
        "output": []
    });
    let model = StatefulCapturingBackend::new(vec![(200, backend_response.to_string())]).start_with_shutdown();
    let proxy_port = free_port();
    let db = TempSqlite::new("client_tool_compat_all_types");
    let config = load_client_tool_compat_config(proxy_port, model.port(), db.url());
    let proxy = start_proxy(&config);

    let request = serde_json::json!({
        "model": "gpt-4.1",
        "input": "Use the available tools.",
        "tools": [
            {"type": "custom", "name": "apply_patch", "description": "Apply a unified diff."},
            {
                "type": "namespace",
                "name": "utils",
                "description": "Local filesystem utilities.",
                "tools": [{
                    "type": "function",
                    "name": "read_file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                        "additionalProperties": false
                    }
                }]
            },
            {"type": "shell", "environment": {"type": "local"}},
            {"type": "tool_search"}
        ]
    });
    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", &serde_json::to_string(&request).unwrap()),
    );

    assert_eq!(
        parse_status(&raw),
        200,
        "mixed rich-tool request should return 200: {raw}"
    );

    let model_reqs = model.requests();
    assert_eq!(model_reqs.len(), 1, "one inference round");
    let backend_body: serde_json::Value =
        serde_json::from_str(&model_reqs[0].body).expect("backend request body should be JSON");
    let backend_tools = backend_body["tools"]
        .as_array()
        .expect("backend request should carry lowered tools");

    assert!(
        backend_tools.iter().all(|tool| tool["type"] == "function"),
        "every rich client tool must reach the backend as a function: {backend_body}"
    );
    let names: Vec<&str> = backend_tools.iter().filter_map(|tool| tool["name"].as_str()).collect();
    assert!(names.contains(&"apply_patch"), "custom lowered: {names:?}");
    assert!(
        names.contains(&"agentic_ns__utils__read_file"),
        "namespace member flattened: {names:?}"
    );
    assert!(names.contains(&"shell"), "shell lowered: {names:?}");
    assert!(names.contains(&"tool_search"), "tool_search lowered: {names:?}");
}

// -----------------------------------------------------------------------------
// Streaming: a rich client tool fails closed without the streaming owner
// -----------------------------------------------------------------------------

/// A streaming request that declares a rich `custom` client tool fails closed
/// with HTTP 500 before any upstream call when the `openai_stream_events` logical
/// SSE owner is NOT in the pipeline to restore the lowered calls — an operator
/// misconfiguration. The lowered private function name is never streamed to the
/// client in an un-restored SSE event, and the backend must never be contacted.
#[test]
fn streaming_rich_client_tool_without_stream_owner_fails_closed() {
    // The backend would 200 if ever contacted; the test asserts it is not.
    let backend_response = serde_json::json!({
        "id": "resp_never",
        "object": "response",
        "status": "completed",
        "output": []
    });
    let model = StatefulCapturingBackend::new(vec![(200, backend_response.to_string())]).start_with_shutdown();
    let proxy_port = free_port();
    let db = TempSqlite::new("client_tool_compat_streaming_no_owner");
    // Config variant WITHOUT openai_stream_events so the marker is never armed.
    let config = load_client_tool_compat_config_without_stream_owner(proxy_port, model.port(), db.url());
    let proxy = start_proxy(&config);

    let request = serde_json::json!({
        "model": "gpt-4.1",
        "stream": true,
        "input": "Apply the patch.",
        "tools": [{
            "type": "custom",
            "name": "apply_patch",
            "description": "Apply a unified diff to the workspace."
        }]
    });
    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", &serde_json::to_string(&request).unwrap()),
    );

    assert_eq!(
        parse_status(&raw),
        500,
        "streaming rich client tools without openai_stream_events must fail closed: {raw}"
    );
    let body = parse_body(&raw);
    assert!(
        body.contains("openai_stream_events"),
        "the rejection names the missing streaming owner: {body}"
    );
    assert!(
        model.requests().is_empty(),
        "the backend must never be contacted when the request fails closed"
    );
}

// -----------------------------------------------------------------------------
// Fail closed: an unsupported legacy local_shell declaration is rejected upstream
// -----------------------------------------------------------------------------

/// The legacy `local_shell` tool restores to a distinct `local_shell_call` output
/// shape this adapter does not yet reconstruct, so a `local_shell` declaration
/// fails closed with HTTP 400 before any upstream call rather than being forwarded
/// to a function-only backend as an unsupported tool. The backend must never be
/// contacted.
#[test]
fn local_shell_declaration_fails_closed_before_upstream() {
    // The backend would 200 if ever contacted; the test asserts it is not.
    let backend_response = serde_json::json!({
        "id": "resp_never",
        "object": "response",
        "status": "completed",
        "output": []
    });
    let model = StatefulCapturingBackend::new(vec![(200, backend_response.to_string())]).start_with_shutdown();
    let proxy_port = free_port();
    let db = TempSqlite::new("client_tool_compat_local_shell");
    let config = load_client_tool_compat_config(proxy_port, model.port(), db.url());
    let proxy = start_proxy(&config);

    let request = serde_json::json!({
        "model": "gpt-4.1",
        "input": "Run a command.",
        "tools": [{"type": "local_shell"}]
    });
    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", &serde_json::to_string(&request).unwrap()),
    );

    assert_eq!(
        parse_status(&raw),
        400,
        "an unsupported local_shell declaration must fail closed: {raw}"
    );
    let body = parse_body(&raw);
    assert!(
        body.contains("local_shell"),
        "the rejection names the unsupported tool: {body}"
    );
    assert!(
        model.requests().is_empty(),
        "the backend must never be contacted when the request fails closed"
    );
}

// -----------------------------------------------------------------------------
// Native passthrough: a plain function tool is unchanged in both directions
// -----------------------------------------------------------------------------

/// A request that declares only a plain `function` tool is a transparent
/// passthrough: the backend receives the tool unchanged and the client receives
/// the backend's `function_call` reply without any typed-item restoration.
#[test]
fn native_function_tool_passes_through_unchanged() {
    let backend_response = serde_json::json!({
        "id": "resp_native",
        "object": "response",
        "status": "completed",
        "output": [{
            "type": "function_call",
            "id": "fc_weather",
            "call_id": "call_weather",
            "name": "get_weather",
            "arguments": r#"{"location":"SF"}"#,
            "status": "completed"
        }]
    });
    let model = StatefulCapturingBackend::new(vec![(200, backend_response.to_string())]).start_with_shutdown();
    let proxy_port = free_port();
    let db = TempSqlite::new("client_tool_compat_native");
    let config = load_client_tool_compat_config(proxy_port, model.port(), db.url());
    let proxy = start_proxy(&config);

    let request = serde_json::json!({
        "model": "gpt-4.1",
        "input": "What is the weather in SF?",
        "tools": [{
            "type": "function",
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
                "additionalProperties": false
            }
        }]
    });
    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", &serde_json::to_string(&request).unwrap()),
    );

    assert_eq!(parse_status(&raw), 200, "native passthrough should return 200: {raw}");

    // The backend saw the plain function tool untouched.
    let model_reqs = model.requests();
    assert_eq!(model_reqs.len(), 1, "one inference round");
    let backend_body: serde_json::Value =
        serde_json::from_str(&model_reqs[0].body).expect("backend request body should be JSON");
    let backend_tools = backend_body["tools"]
        .as_array()
        .expect("backend request should carry the function tool");
    assert_eq!(backend_tools.len(), 1, "one tool forwarded: {backend_body}");
    assert_eq!(backend_tools[0]["type"], "function", "function forwarded verbatim");
    assert_eq!(backend_tools[0]["name"], "get_weather", "function name unchanged");

    // The client sees the model's function_call unchanged — no restoration.
    let response: serde_json::Value = serde_json::from_str(&parse_body(&raw)).expect("client response should be JSON");
    let output = response["output"].as_array().expect("output array present");
    assert_eq!(output.len(), 1, "one output item: {response}");
    assert_eq!(
        output[0]["type"], "function_call",
        "a plain function_call must not be restyled: {response}"
    );
    assert_eq!(output[0]["name"], "get_weather", "function name preserved");
    assert!(
        output.iter().all(|item| item["type"] != "custom_tool_call"),
        "native traffic must not gain a restored typed item: {response}"
    );
}

// -----------------------------------------------------------------------------
// Streaming: a custom client tool is restored live in the SSE lifecycle
// -----------------------------------------------------------------------------

/// A streaming request that declares a rich `custom` client tool has the tool
/// lowered to a private `function` on the wire and the backend's streamed
/// `function_call` lifecycle restored live to a `custom_tool_call` — with the
/// single string parameter unwrapped into the plain-string `input` — by the
/// `openai_stream_events` owner (#1159). The client never sees a bare
/// `function_call` output item for the restored tool.
#[test]
fn streaming_custom_tool_restores_lifecycle() {
    let sse_body = concat!(
        "event: response.output_item.added\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"fc_1\",\"call_id\":\"call_1\",\"name\":\"apply_patch\",\"arguments\":\"\"}}\n\n",
        "event: response.function_call_arguments.delta\n",
        "data: {\"type\":\"response.function_call_arguments.delta\",\"item_id\":\"fc_1\",\"output_index\":0,\"delta\":\"{\\\"input\\\":\\\"*** Begin Patch\\\"}\"}\n\n",
        "event: response.function_call_arguments.done\n",
        "data: {\"type\":\"response.function_call_arguments.done\",\"item_id\":\"fc_1\",\"output_index\":0,\"arguments\":\"{\\\"input\\\":\\\"*** Begin Patch\\\"}\"}\n\n",
        "event: response.output_item.done\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"fc_1\",\"call_id\":\"call_1\",\"name\":\"apply_patch\",\"arguments\":\"{\\\"input\\\":\\\"*** Begin Patch\\\"}\"}}\n\n",
        "event: response.completed\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_custom_stream\",\"object\":\"response\",\"status\":\"completed\",\"output\":[{\"type\":\"function_call\",\"id\":\"fc_1\",\"call_id\":\"call_1\",\"name\":\"apply_patch\",\"arguments\":\"{\\\"input\\\":\\\"*** Begin Patch\\\"}\"}],\"tools\":[{\"type\":\"function\",\"name\":\"apply_patch\"}]}}\n\n",
        "event: done\ndata: [DONE]\n\n",
    );
    let model = Backend::fixed(sse_body)
        .header("content-type", "text/event-stream")
        .start_with_shutdown();
    let proxy_port = free_port();
    let db = TempSqlite::new("client_tool_compat_stream_custom");
    let config = load_client_tool_compat_config(proxy_port, model.port(), db.url());
    let proxy = start_proxy(&config);

    let request = serde_json::json!({
        "model": "gpt-4.1",
        "stream": true,
        "input": "Apply the patch.",
        "tools": [{
            "type": "custom",
            "name": "apply_patch",
            "description": "Apply a unified diff to the workspace."
        }]
    });
    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", &serde_json::to_string(&request).unwrap()),
    );

    assert_eq!(
        parse_status(&raw),
        200,
        "streaming custom restore should return 200: {raw}"
    );
    assert_eq!(
        parse_header(&raw, "content-type").as_deref(),
        Some("text/event-stream"),
        "streaming response keeps the SSE content type"
    );
    let body = parse_body(&raw);
    assert!(
        body.contains("\"type\":\"custom_tool_call\""),
        "the streamed function_call is restored to a custom_tool_call: {body}"
    );
    assert!(
        body.contains("*** Begin Patch"),
        "the custom input is unwrapped and streamed to the client: {body}"
    );
}

// -----------------------------------------------------------------------------
// Streaming: a local shell client tool is restored live to a shell_call
// -----------------------------------------------------------------------------

/// A streaming request that declares a local `shell` client tool has the tool
/// lowered to a private `function` named `shell` and the backend's streamed
/// `function_call` restored live to a `shell_call` with `environment.type ==
/// "local"` and the shell commands recovered (#1159).
#[test]
fn streaming_shell_tool_restores_local_shell_call() {
    let sse_body = concat!(
        "event: response.output_item.added\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"fc_sh1\",\"call_id\":\"call_sh1\",\"name\":\"shell\",\"arguments\":\"\"}}\n\n",
        "event: response.function_call_arguments.delta\n",
        "data: {\"type\":\"response.function_call_arguments.delta\",\"item_id\":\"fc_sh1\",\"output_index\":0,\"delta\":\"{\\\"commands\\\":[\\\"ls\\\",\\\"-la\\\"]}\"}\n\n",
        "event: response.function_call_arguments.done\n",
        "data: {\"type\":\"response.function_call_arguments.done\",\"item_id\":\"fc_sh1\",\"output_index\":0,\"arguments\":\"{\\\"commands\\\":[\\\"ls\\\",\\\"-la\\\"]}\"}\n\n",
        "event: response.output_item.done\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"fc_sh1\",\"call_id\":\"call_sh1\",\"name\":\"shell\",\"arguments\":\"{\\\"commands\\\":[\\\"ls\\\",\\\"-la\\\"]}\"}}\n\n",
        "event: response.completed\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_shell_stream\",\"object\":\"response\",\"status\":\"completed\",\"output\":[{\"type\":\"function_call\",\"id\":\"fc_sh1\",\"call_id\":\"call_sh1\",\"name\":\"shell\",\"arguments\":\"{\\\"commands\\\":[\\\"ls\\\",\\\"-la\\\"]}\"}],\"tools\":[{\"type\":\"function\",\"name\":\"shell\"}]}}\n\n",
        "event: done\ndata: [DONE]\n\n",
    );
    let model = Backend::fixed(sse_body)
        .header("content-type", "text/event-stream")
        .start_with_shutdown();
    let proxy_port = free_port();
    let db = TempSqlite::new("client_tool_compat_stream_shell");
    let config = load_client_tool_compat_config(proxy_port, model.port(), db.url());
    let proxy = start_proxy(&config);

    let request = serde_json::json!({
        "model": "gpt-4.1",
        "stream": true,
        "input": "List the files.",
        "tools": [{"type": "shell", "environment": {"type": "local"}}]
    });
    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", &serde_json::to_string(&request).unwrap()),
    );

    assert_eq!(
        parse_status(&raw),
        200,
        "streaming shell restore should return 200: {raw}"
    );
    let body = parse_body(&raw);
    assert!(
        body.contains("\"type\":\"shell_call\""),
        "the streamed function_call is restored to a shell_call: {body}"
    );
    assert!(
        body.contains("\"local\""),
        "the restored shell call is stamped with a local environment: {body}"
    );
    assert!(
        body.contains("\"ls\"") && body.contains("\"-la\""),
        "the shell commands are recovered onto the restored call: {body}"
    );
}

// -----------------------------------------------------------------------------
// Streaming: a namespace member never leaks its private lowered name
// -----------------------------------------------------------------------------

/// A streaming request that declares a `namespace` member has the member lowered
/// to a flat private `function` name (`agentic_ns__utils__read_file`) and the
/// backend's streamed `function_call` restored in place to its namespaced form
/// (bare member name `read_file`, `namespace` `utils` re-added). The private
/// lowered name must never appear anywhere in the streamed body (#1159).
#[test]
fn streaming_namespace_member_never_leaks_private_name() {
    let sse_body = concat!(
        "event: response.output_item.added\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"fc_ns1\",\"call_id\":\"call_ns1\",\"name\":\"agentic_ns__utils__read_file\",\"arguments\":\"\"}}\n\n",
        "event: response.function_call_arguments.delta\n",
        "data: {\"type\":\"response.function_call_arguments.delta\",\"item_id\":\"fc_ns1\",\"output_index\":0,\"delta\":\"{\\\"path\\\":\\\"/etc/hosts\\\"}\"}\n\n",
        "event: response.function_call_arguments.done\n",
        "data: {\"type\":\"response.function_call_arguments.done\",\"item_id\":\"fc_ns1\",\"output_index\":0,\"arguments\":\"{\\\"path\\\":\\\"/etc/hosts\\\"}\"}\n\n",
        "event: response.output_item.done\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"fc_ns1\",\"call_id\":\"call_ns1\",\"name\":\"agentic_ns__utils__read_file\",\"arguments\":\"{\\\"path\\\":\\\"/etc/hosts\\\"}\"}}\n\n",
        "event: response.completed\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_ns_stream\",\"object\":\"response\",\"status\":\"completed\",\"output\":[{\"type\":\"function_call\",\"id\":\"fc_ns1\",\"call_id\":\"call_ns1\",\"name\":\"agentic_ns__utils__read_file\",\"arguments\":\"{\\\"path\\\":\\\"/etc/hosts\\\"}\"}],\"tools\":[{\"type\":\"function\",\"name\":\"agentic_ns__utils__read_file\"}]}}\n\n",
        "event: done\ndata: [DONE]\n\n",
    );
    let model = Backend::fixed(sse_body)
        .header("content-type", "text/event-stream")
        .start_with_shutdown();
    let proxy_port = free_port();
    let db = TempSqlite::new("client_tool_compat_stream_namespace");
    let config = load_client_tool_compat_config(proxy_port, model.port(), db.url());
    let proxy = start_proxy(&config);

    let request = serde_json::json!({
        "model": "gpt-4.1",
        "stream": true,
        "input": "Read the hosts file.",
        "tools": [{
            "type": "namespace",
            "name": "utils",
            "description": "Local filesystem utilities.",
            "tools": [{
                "type": "function",
                "name": "read_file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": false
                }
            }]
        }]
    });
    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", &serde_json::to_string(&request).unwrap()),
    );

    assert_eq!(
        parse_status(&raw),
        200,
        "streaming namespace restore should return 200: {raw}"
    );
    let body = parse_body(&raw);
    assert!(
        !body.contains("agentic_ns__"),
        "the private lowered namespace name must never reach the client on the stream: {body}"
    );
    assert!(
        body.contains("\"name\":\"read_file\""),
        "the restored namespaced call carries the bare member name: {body}"
    );
    assert!(
        body.contains("\"namespace\":\"utils\""),
        "the restored namespaced call re-adds its namespace: {body}"
    );
}
