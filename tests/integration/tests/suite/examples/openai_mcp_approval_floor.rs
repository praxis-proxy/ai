// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for the mcp-approval-floor example config.
//!
//! The example ships the operator approval floor
//! (`openai_mcp_dispatch.minimum_approval: always`) on a complete persisted
//! agentic Responses pipeline, so these drive the real approval exchange the
//! example promises: a client that submits `"require_approval": "never"` still
//! gets an `mcp_approval_request` and runs zero `tools/call` until it approves,
//! after which the recorded call executes exactly once. A denial executes
//! nothing. Non-MCP traffic is undisturbed.

use std::collections::HashMap;

use praxis_test_utils::{
    McpMockConfig, McpToolFixture, StatefulCapturingBackend, TempSqlite, allow_loopback_endpoints, example_config_path,
    free_port, http_send, json_post, parse_body, parse_status, patch_yaml, start_mcp_mock_server_with_config,
    start_proxy,
};

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// Load the `mcp-approval-floor.yaml` example, patching the listener/backend
/// ports and pointing the Responses store at a private temp SQLite database so
/// the approval request/response round trip survives across two client requests
/// without sharing persisted state.
fn load_floor_config(proxy_port: u16, model_port: u16, db_url: &str) -> praxis_core::config::Config {
    let path = example_config_path("openai/responses/mcp-approval-floor.yaml");
    let yaml = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let yaml = yaml.replace("sqlite://responses.db?mode=rwc", db_url);
    let patched = allow_loopback_endpoints(&patch_yaml(
        &yaml,
        proxy_port,
        &HashMap::from([("127.0.0.1:3001", model_port)]),
    ));
    praxis_core::config::Config::from_yaml(&patched).expect("parse mcp-approval-floor config")
}

/// The MCP tool fixture shared by the approval round-trip tests.
fn weather_mock() -> praxis_test_utils::McpMockServerGuard {
    start_mcp_mock_server_with_config(McpMockConfig {
        tools: vec![
            McpToolFixture::new("get_weather")
                .with_description("Get the weather for a location")
                .with_input_schema(serde_json::json!({
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                    "additionalProperties": false
                })),
        ],
        ..McpMockConfig::default()
    })
}

/// The model turn that asks to call the weather tool.
fn call_response() -> String {
    serde_json::json!({
        "id": "resp_floor_1",
        "object": "response",
        "created_at": 1000,
        "model": "gpt-4.1",
        "status": "completed",
        "output": [{
            "type": "function_call",
            "id": "fc_floor",
            "call_id": "call_floor_weather",
            "name": "weather__get_weather",
            "arguments": r#"{"location":"SF"}"#,
            "status": "completed"
        }]
    })
    .to_string()
}

/// The model turn after the tool result is fed back.
fn final_response() -> String {
    serde_json::json!({
        "id": "resp_floor_final",
        "object": "response",
        "created_at": 2000,
        "model": "gpt-4.1",
        "status": "completed",
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "The weather in SF is 72F and sunny."}]
        }]
    })
    .to_string()
}

/// An MCP tool whose client policy is `"never"`, so only the operator floor can
/// require approval.
fn never_tools(mcp_port: u16) -> serde_json::Value {
    serde_json::json!([{
        "type": "mcp",
        "server_label": "weather",
        "server_url": format!("http://127.0.0.1:{mcp_port}/mcp"),
        "allowed_tools": ["get_weather"],
        "require_approval": "never"
    }])
}

/// Drive the first turn and return `(approval_request_id, previous_response_id)`.
///
/// Asserts the floor produced exactly one `mcp_approval_request` and that no tool
/// executed while the approval is pending.
fn request_floored_approval(
    proxy_addr: &str,
    mcp: &praxis_test_utils::McpMockServerGuard,
    tools: &serde_json::Value,
) -> (String, String) {
    let request = serde_json::json!({
        "model": "gpt-4.1",
        "input": "What is the weather in SF?",
        "tools": tools,
    });
    let raw = http_send(
        proxy_addr,
        &json_post("/v1/responses", &serde_json::to_string(&request).unwrap()),
    );
    assert_eq!(
        parse_status(&raw),
        200,
        "floored approval request should return 200: {raw}"
    );
    let response: serde_json::Value = serde_json::from_str(&parse_body(&raw)).expect("response should be JSON");
    let response_id = response["id"].as_str().expect("response id").to_owned();

    let output = response["output"].as_array().expect("output array");
    let approvals: Vec<&serde_json::Value> = output
        .iter()
        .filter(|item| item["type"] == "mcp_approval_request")
        .collect();
    assert_eq!(
        approvals.len(),
        1,
        "the floor must surface exactly one mcp_approval_request even though the client said never: {output:#?}"
    );
    assert_eq!(
        approvals[0]["name"], "get_weather",
        "the approval must preserve the tool name"
    );
    assert_eq!(
        approvals[0]["server_label"], "weather",
        "the approval must preserve the server label"
    );
    let approval_id = approvals[0]["id"].as_str().expect("approval id").to_owned();

    assert_eq!(
        mcp.method_count("tools/call"),
        0,
        "no tool may execute while a floored approval is pending"
    );

    (approval_id, response_id)
}

/// Build the follow-up request carrying a single `mcp_approval_response`.
fn approval_followup(
    previous_response_id: &str,
    approval_id: &str,
    approve: bool,
    tools: &serde_json::Value,
) -> String {
    serde_json::to_string(&serde_json::json!({
        "model": "gpt-4.1",
        "previous_response_id": previous_response_id,
        "tools": tools,
        "input": [{
            "type": "mcp_approval_response",
            "approval_request_id": approval_id,
            "approve": approve
        }]
    }))
    .unwrap()
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn floor_overrides_client_never_then_executes_on_approval() {
    let model =
        StatefulCapturingBackend::new(vec![(200, call_response()), (200, final_response())]).start_with_shutdown();
    let mcp = weather_mock();

    let db = TempSqlite::new("mcp_approval_floor_approve");
    let proxy_port = free_port();
    let config = load_floor_config(proxy_port, model.port(), db.url());
    let proxy = start_proxy(&config);

    let tools = never_tools(mcp.port());

    // Turn 1: the client asks for NO approval; the operator floor forces one.
    let (approval_id, previous_response_id) = request_floored_approval(proxy.addr(), &mcp, &tools);
    assert_eq!(
        approval_id, "call_floor_weather",
        "approval id must equal the model call id"
    );

    // Turn 2: approving resumes the recorded call, which executes exactly once.
    let raw = http_send(
        proxy.addr(),
        &json_post(
            "/v1/responses",
            &approval_followup(&previous_response_id, &approval_id, true, &tools),
        ),
    );
    assert_eq!(parse_status(&raw), 200, "approved follow-up should return 200: {raw}");
    let response: serde_json::Value = serde_json::from_str(&parse_body(&raw)).expect("response should be JSON");
    assert_eq!(
        response["id"], "resp_floor_final",
        "an approved round trip must complete model->tool->model"
    );
    assert_eq!(
        mcp.method_count("tools/call"),
        1,
        "a valid approval must execute the floored call exactly once"
    );
    assert_eq!(mcp.last_tool_call_name().as_deref(), Some("get_weather"));
}

#[test]
fn floor_denied_runs_no_tool() {
    let model =
        StatefulCapturingBackend::new(vec![(200, call_response()), (200, final_response())]).start_with_shutdown();
    let mcp = weather_mock();

    let db = TempSqlite::new("mcp_approval_floor_deny");
    let proxy_port = free_port();
    let config = load_floor_config(proxy_port, model.port(), db.url());
    let proxy = start_proxy(&config);

    let tools = never_tools(mcp.port());
    let (approval_id, previous_response_id) = request_floored_approval(proxy.addr(), &mcp, &tools);

    let raw = http_send(
        proxy.addr(),
        &json_post(
            "/v1/responses",
            &approval_followup(&previous_response_id, &approval_id, false, &tools),
        ),
    );
    assert_eq!(
        parse_status(&raw),
        200,
        "denied follow-up should still return 200: {raw}"
    );
    assert_eq!(
        mcp.method_count("tools/call"),
        0,
        "a denied floored approval must never execute the tool"
    );
}

#[test]
fn non_mcp_request_is_undisturbed_by_the_floor() {
    let model = StatefulCapturingBackend::new(vec![(200, final_response())]).start_with_shutdown();

    let db = TempSqlite::new("mcp_approval_floor_passthrough");
    let proxy_port = free_port();
    let config = load_floor_config(proxy_port, model.port(), db.url());
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"Hello, world!"}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "a non-MCP request should complete normally");
    let response: serde_json::Value = serde_json::from_str(&parse_body(&raw)).expect("response should be JSON");
    assert_eq!(
        response["output"][0]["content"][0]["text"], "The weather in SF is 72F and sunny.",
        "a request with no MCP tools must reach the model and return its message unchanged: {response:#?}"
    );
    let requests = model.requests();
    assert_eq!(
        requests.len(),
        1,
        "a non-MCP request must make exactly one inference call"
    );
    let forwarded: serde_json::Value = serde_json::from_str(&requests[0].body).expect("forwarded body should be JSON");
    assert!(
        forwarded.to_string().contains("Hello, world!"),
        "the floored pipeline must forward the client input unchanged: {forwarded}"
    );
}
