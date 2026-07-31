// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Integration tests for the agentic_loop filter with
//! `iterative_request_router`.
//!
//! These tests verify that IRR, request-supplied MCP resolution,
//! MCP dispatch, and the agentic inference loop function together.

use std::collections::HashMap;

use praxis_test_utils::{
    McpMockConfig, McpToolFixture, StatefulCapturingBackend, build_pipeline, example_config_path, free_port, http_send,
    json_post, load_example_config, parse_body, parse_status, patch_yaml, start_backend_with_shutdown,
    start_mcp_mock_server_with_config, start_proxy,
};

// -----------------------------------------------------------------------------
// Pipeline Build
// -----------------------------------------------------------------------------

#[test]
fn example_config_builds_pipeline() {
    let config = load_example_config(
        "openai/responses/agentic-loop.yaml",
        free_port(),
        HashMap::from([("127.0.0.1:3001", 19901_u16)]),
    );
    let _pipeline = build_pipeline(&config);
}

// -----------------------------------------------------------------------------
// Single-Pass
// -----------------------------------------------------------------------------

#[test]
fn single_pass_completes_through_irr() {
    let model = start_backend_with_shutdown(r#"{"id":"resp_1","object":"response","status":"completed","output":[]}"#);
    let proxy_port = free_port();

    let config = load_example_config(
        "openai/responses/agentic-loop.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3001", model.port())]),
    );
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"Hello"}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(
        parse_status(&raw),
        200,
        "single-pass request through IRR should return 200"
    );
}

#[test]
fn client_function_call_returns_without_server_execution() {
    let function_response = serde_json::json!({
        "id": "resp_client_tool",
        "object": "response",
        "status": "completed",
        "output": [{
            "type": "function_call",
            "id": "fc_client",
            "call_id": "call_client",
            "name": "get_weather",
            "arguments": r#"{"location":"SF"}"#,
            "status": "completed"
        }]
    });
    let model = StatefulCapturingBackend::new(vec![(
        200,
        serde_json::to_string(&function_response).expect("serialize function response"),
    )])
    .start_with_shutdown();
    let proxy_port = free_port();
    let config = load_example_config(
        "openai/responses/agentic-loop.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3001", model.port())]),
    );
    let proxy = start_proxy(&config);

    let request = serde_json::json!({
        "model": "gpt-4.1",
        "input": "What is the weather in SF?",
        "tools": [{
            "type": "function",
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}}
            }
        }]
    });
    let raw = http_send(
        proxy.addr(),
        &json_post(
            "/v1/responses",
            &serde_json::to_string(&request).expect("serialize client function request"),
        ),
    );

    assert_eq!(parse_status(&raw), 200);
    let response: serde_json::Value =
        serde_json::from_str(&parse_body(&raw)).expect("client function response should be JSON");
    assert_eq!(response["id"], "resp_client_tool");
    assert_eq!(
        model.requests().len(),
        1,
        "client-side function calls must return to the client without an internal loop"
    );
}

// -----------------------------------------------------------------------------
// Round-Trip: Resolve MCP → Inference → tools/call → Inference
// -----------------------------------------------------------------------------

#[test]
fn round_trip_captures_tool_and_model_requests() {
    let first_response = serde_json::json!({
        "id": "resp_1",
        "object": "response",
        "status": "completed",
        "output": [{
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_abc",
            "name": "weather__get_weather",
            "arguments": r#"{"location":"SF"}"#,
            "status": "completed"
        }]
    });
    let second_response = serde_json::json!({
        "id": "resp_2",
        "object": "response",
        "status": "completed",
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "The weather in SF is 72F and sunny."}]
        }]
    });

    let model = StatefulCapturingBackend::new(vec![
        (200, serde_json::to_string(&first_response).unwrap()),
        (200, serde_json::to_string(&second_response).unwrap()),
    ])
    .start_with_shutdown();

    let mcp = start_mcp_mock_server_with_config(McpMockConfig {
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
    });

    let proxy_port = free_port();
    let config = load_loopback_mcp_config(proxy_port, model.port());
    let proxy = start_proxy(&config);

    let mcp_url = format!("http://127.0.0.1:{}/mcp", mcp.port());
    let request_body = serde_json::json!({
        "model": "gpt-4.1",
        "input": "What is the weather in SF?",
        "tools": [{
            "type": "mcp",
            "server_label": "weather",
            "server_url": mcp_url,
            "allowed_tools": ["get_weather"],
            "require_approval": "never"
        }]
    });
    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", &serde_json::to_string(&request_body).unwrap()),
    );

    assert_eq!(parse_status(&raw), 200, "round-trip should return 200");
    let body = parse_body(&raw);
    let response: serde_json::Value = serde_json::from_str(&body).expect("response should be valid JSON");
    assert_eq!(
        response["id"], "resp_2",
        "final response should be the second model response"
    );

    // -------------------------------------------------------------------------
    // Assert request-supplied MCP discovery and execution
    // -------------------------------------------------------------------------
    assert!(
        mcp.method_count("tools/list") >= 1,
        "MCP resolver should call tools/list on the request-supplied server"
    );
    assert_eq!(mcp.method_count("tools/call"), 1, "MCP dispatch should call one tool");
    assert_eq!(mcp.last_tool_call_name().as_deref(), Some("get_weather"));

    let mcp_requests = mcp.received_requests();
    let call = mcp_requests
        .iter()
        .find(|request| request.json_rpc_method.as_deref() == Some("tools/call"))
        .expect("MCP server should receive tools/call");
    let call_body: serde_json::Value = serde_json::from_str(&call.body).expect("tools/call body should be JSON");
    assert_eq!(call_body["params"]["arguments"]["location"], "SF");

    // -------------------------------------------------------------------------
    // Assert resolved first request and tool-enriched second request
    // -------------------------------------------------------------------------
    let model_reqs = model.requests();
    assert_eq!(model_reqs.len(), 2, "model backend should receive exactly two requests");

    let first_model_body: serde_json::Value =
        serde_json::from_str(&model_reqs[0].body).expect("first model request body should be valid JSON");
    let resolved_tools = first_model_body["tools"]
        .as_array()
        .expect("first model request should contain resolved tools");
    assert!(
        resolved_tools
            .iter()
            .any(|tool| tool["type"] == "function" && tool["name"] == "weather__get_weather"),
        "MCP resolver should expose the request-supplied MCP tool as an encoded function"
    );

    let second_model_req = &model_reqs[1];

    let model_body: serde_json::Value =
        serde_json::from_str(&second_model_req.body).expect("second model request body should be valid JSON");
    let input = model_body["input"]
        .as_array()
        .expect("second model request input should be an array");

    let has_function_call = input.iter().any(|item| item["type"] == "function_call");
    let has_function_call_output = input.iter().any(|item| item["type"] == "function_call_output");
    assert!(
        has_function_call,
        "second model request input should contain a function_call item"
    );
    assert!(
        has_function_call_output,
        "second model request input should contain a function_call_output item"
    );
    let function_output = input
        .iter()
        .find(|item| item["type"] == "function_call_output")
        .expect("function_call_output should be present");
    assert!(
        function_output["output"]
            .as_str()
            .is_some_and(|output| output.contains("mock result for get_weather")),
        "second inference should receive the MCP tools/call result"
    );

    // -------------------------------------------------------------------------
    // Assert agentic_loop bookkeeping in second request
    // -------------------------------------------------------------------------
    assert_eq!(
        model_body["parallel_tool_calls"], false,
        "agentic_loop must force parallel_tool_calls=false on re-entry"
    );
    assert_eq!(
        model_body["tool_choice"], "auto",
        "agentic_loop must reset tool_choice to auto on re-entry"
    );
}

/// Load the checked-in example while enabling loopback only for the
/// test-owned MCP server.
fn load_loopback_mcp_config(proxy_port: u16, model_port: u16) -> praxis_core::config::Config {
    let path = example_config_path("openai/responses/agentic-loop.yaml");
    let yaml = std::fs::read_to_string(path).expect("read agentic-loop example");
    let yaml = patch_yaml(&yaml, proxy_port, &HashMap::from([("127.0.0.1:3001", model_port)]));
    let yaml = yaml.replacen(
        "      - filter: openai_mcp_tool_resolve\n",
        "      - filter: openai_mcp_tool_resolve\n        allow_loopback: true\n",
        1,
    );
    let yaml = yaml.replacen(
        "              - filter: openai_mcp_dispatch\n",
        "              - filter: openai_mcp_dispatch\n                allow_loopback: true\n",
        1,
    );
    praxis_core::config::Config::from_yaml(&yaml).expect("parse loopback MCP config")
}
