// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for the anthropic/mcp-connector.yaml example.
//!
//! Each test drives the full loop: Praxis lists the mock MCP server's tools,
//! the scripted backend calls one, Praxis runs it over MCP, and the client
//! receives Anthropic `mcp_tool_use` / `mcp_tool_result` blocks.

use std::collections::HashMap;

use praxis_test_utils::{
    McpMockConfig, McpMockServerGuard, McpToolFixture, StatefulCapturingBackend, example_config_path, free_port,
    http_send, json_post, json_post_with_header, parse_body, parse_status, patch_yaml,
    start_mcp_mock_server_with_config, start_proxy,
};
use serde_json::{Value, json};

const EXAMPLE: &str = "anthropic/mcp-connector.yaml";
const CHAT_MODEL: &str = "Qwen/Qwen3-8B";
const NATIVE_MODEL: &str = "openai/gpt-oss-20b";

// -----------------------------------------------------------------------------
// Fixtures
// -----------------------------------------------------------------------------

/// Start a mock MCP server exposing `get_weather`.
fn start_weather_mcp() -> McpMockServerGuard {
    start_mcp_mock_server_with_config(McpMockConfig {
        tools: vec![McpToolFixture::new("get_weather").with_description("Get the weather for a city")],
        ..McpMockConfig::default()
    })
}

/// Load the example with local ports and the mock MCP server URL.
fn load_config(
    proxy_port: u16,
    native_port: u16,
    chat_port: u16,
    mcp: &McpMockServerGuard,
) -> praxis_core::config::Config {
    let yaml = std::fs::read_to_string(example_config_path(EXAMPLE)).expect("read mcp-connector example");
    let yaml = patch_yaml(
        &yaml,
        proxy_port,
        &HashMap::from([("127.0.0.1:8000", native_port), ("127.0.0.1:8001", chat_port)]),
    );
    let yaml = yaml.replace("http://127.0.0.1:9100/mcp", &format!("http://{}/mcp", mcp.endpoint()));
    // Secret-bearing filters resolve at build time; use a Cargo-provided var.
    let yaml = yaml.replace("env_var: VLLM_API_KEY", "env_var: CARGO_PKG_NAME");
    praxis_core::config::Config::from_yaml(&yaml).expect("parse mcp-connector example")
}

/// A client request declaring the mock server through the MCP connector.
fn connector_request(model: &str, mcp: &McpMockServerGuard) -> String {
    json!({
        "model": model,
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
        "mcp_servers": [{"type": "url", "url": format!("http://{}/mcp", mcp.endpoint()), "name": "weather"}],
        "tools": [{"type": "mcp_toolset", "mcp_server_name": "weather"}],
    })
    .to_string()
}

/// A native Messages round calling the encoded MCP tool.
fn native_tool_round() -> String {
    json!({
        "id": "msg_1", "type": "message", "role": "assistant", "model": NATIVE_MODEL,
        "content": [
            {"type": "text", "text": "Checking."},
            {"type": "tool_use", "id": "toolu_1", "name": "weather__get_weather", "input": {"city": "Paris"}},
        ],
        "stop_reason": "tool_use", "stop_sequence": null,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    })
    .to_string()
}

/// A final native Messages round.
fn native_answer_round() -> String {
    json!({
        "id": "msg_2", "type": "message", "role": "assistant", "model": NATIVE_MODEL,
        "content": [{"type": "text", "text": "It is sunny in Paris."}],
        "stop_reason": "end_turn", "stop_sequence": null,
        "usage": {"input_tokens": 20, "output_tokens": 7},
    })
    .to_string()
}

/// A Chat Completions round calling the encoded MCP tool.
fn chat_tool_round() -> String {
    json!({
        "id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": CHAT_MODEL,
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": null,
            "tool_calls": [{"id": "call_1", "type": "function", "function": {
                "name": "weather__get_weather", "arguments": "{\"city\":\"Paris\"}"
            }}],
        }}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    })
    .to_string()
}

/// A final Chat Completions round.
fn chat_answer_round() -> String {
    json!({
        "id": "chatcmpl-2", "object": "chat.completion", "created": 2, "model": CHAT_MODEL,
        "choices": [{"index": 0, "finish_reason": "stop", "message": {
            "role": "assistant", "content": "It is sunny in Paris."
        }}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 7, "total_tokens": 27},
    })
    .to_string()
}

/// Content block types of a client response.
fn block_types(response: &Value) -> Vec<&str> {
    response["content"]
        .as_array()
        .expect("content array")
        .iter()
        .map(|block| block["type"].as_str().expect("block type"))
        .collect()
}

/// Assert the client sees the MCP call and its result from the mock server.
fn assert_mcp_blocks(response: &Value) {
    let content = response["content"].as_array().expect("content array");
    let tool_use = content
        .iter()
        .find(|block| block["type"] == "mcp_tool_use")
        .expect("mcp_tool_use block");
    assert_eq!(
        tool_use["name"], "get_weather",
        "the client sees the original MCP tool name"
    );
    assert_eq!(
        tool_use["server_name"], "weather",
        "the client sees the declared server name"
    );
    assert_eq!(tool_use["input"], json!({"city": "Paris"}), "the model input is kept");
    let result = content
        .iter()
        .find(|block| block["type"] == "mcp_tool_result")
        .expect("mcp_tool_result block");
    assert_eq!(result["tool_use_id"], tool_use["id"], "the result answers its call");
    assert_eq!(result["is_error"], false, "the MCP call succeeded");
    assert_eq!(
        result["content"][0]["text"], "mock result for get_weather",
        "the MCP server output is returned"
    );
    assert_eq!(
        content.last().expect("final block")["text"],
        "It is sunny in Paris.",
        "the model answer ends the response"
    );
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn native_route_runs_mcp_tools_and_returns_mcp_blocks() {
    let mcp = start_weather_mcp();
    let native = StatefulCapturingBackend::new(vec![(200, native_tool_round()), (200, native_answer_round())])
        .start_with_shutdown();
    let proxy = start_proxy(&load_config(free_port(), native.port(), free_port(), &mcp));

    let raw = http_send(
        proxy.addr(),
        &json_post_with_header(
            "/v1/messages",
            &connector_request(NATIVE_MODEL, &mcp),
            "anthropic-beta: mcp-client-2025-11-20",
        ),
    );

    assert_eq!(parse_status(&raw), 200, "connector loop should succeed: {raw}");
    let response: Value = serde_json::from_str(&parse_body(&raw)).expect("client response JSON");
    assert_eq!(
        block_types(&response),
        ["text", "mcp_tool_use", "mcp_tool_result", "text"]
    );
    assert_mcp_blocks(&response);
    assert_eq!(response["usage"]["input_tokens"], 30, "usage sums both rounds");
    assert_eq!(response["usage"]["output_tokens"], 12);
    assert_eq!(mcp.tool_call_count("get_weather"), 1, "the tool runs once");

    let requests = native.requests();
    assert_eq!(requests.len(), 2, "one tool round and one answer round");
    let first: Value = serde_json::from_str(&requests[0].body).expect("first round JSON");
    assert!(
        first.get("mcp_servers").is_none(),
        "the connector must not reach the backend"
    );
    assert_eq!(first["tools"][0]["name"], "weather__get_weather");
    for request in &requests {
        assert!(
            !request.headers.to_ascii_lowercase().contains("mcp-client-"),
            "the connector beta must not reach the backend on any round"
        );
    }
    let second: Value = serde_json::from_str(&requests[1].body).expect("second round JSON");
    let messages = second["messages"].as_array().expect("messages");
    let tool_result = &messages.last().expect("tool result turn")["content"][0];
    assert_eq!(tool_result["type"], "tool_result");
    assert_eq!(tool_result["tool_use_id"], "toolu_1");
    assert_eq!(tool_result["content"], "mock result for get_weather");
}

#[test]
fn chat_route_runs_mcp_tools_and_returns_mcp_blocks() {
    let mcp = start_weather_mcp();
    let chat =
        StatefulCapturingBackend::new(vec![(200, chat_tool_round()), (200, chat_answer_round())]).start_with_shutdown();
    let proxy = start_proxy(&load_config(free_port(), free_port(), chat.port(), &mcp));

    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/messages", &connector_request(CHAT_MODEL, &mcp)),
    );

    assert_eq!(
        parse_status(&raw),
        200,
        "translated connector loop should succeed: {raw}"
    );
    let response: Value = serde_json::from_str(&parse_body(&raw)).expect("client response JSON");
    assert_eq!(block_types(&response), ["mcp_tool_use", "mcp_tool_result", "text"]);
    assert_mcp_blocks(&response);

    let requests = chat.requests();
    assert_eq!(requests.len(), 2);
    let first: Value = serde_json::from_str(&requests[0].body).expect("first round JSON");
    assert_eq!(first["tools"][0]["function"]["name"], "weather__get_weather");
    let second: Value = serde_json::from_str(&requests[1].body).expect("second round JSON");
    let tool_message = second["messages"]
        .as_array()
        .expect("messages")
        .iter()
        .find(|message| message["role"] == "tool")
        .expect("tool message");
    assert_eq!(tool_message["tool_call_id"], "call_1");
    assert!(
        tool_message["content"]
            .to_string()
            .contains("mock result for get_weather"),
        "{tool_message}"
    );
}

#[test]
fn replayed_mcp_history_reaches_the_backend_as_tool_turns() {
    let mcp = start_weather_mcp();
    let native = StatefulCapturingBackend::new(vec![(200, native_answer_round())]).start_with_shutdown();
    let proxy = start_proxy(&load_config(free_port(), native.port(), free_port(), &mcp));
    let body = json!({
        "model": NATIVE_MODEL,
        "max_tokens": 256,
        "mcp_servers": [{"type": "url", "url": format!("http://{}/mcp", mcp.endpoint()), "name": "weather"}],
        "tools": [{"type": "mcp_toolset", "mcp_server_name": "weather"}],
        "messages": [
            {"role": "user", "content": "What is the weather in Paris?"},
            {"role": "assistant", "content": [
                {"type": "mcp_tool_use", "id": "mcptoolu_1", "name": "get_weather", "server_name": "weather", "input": {"city": "Paris"}},
                {"type": "mcp_tool_result", "tool_use_id": "mcptoolu_1", "is_error": false, "content": [{"type": "text", "text": "sunny"}]},
                {"type": "text", "text": "It is sunny."},
            ]},
            {"role": "user", "content": "Thanks!"},
        ],
    })
    .to_string();

    let raw = http_send(proxy.addr(), &json_post("/v1/messages", &body));

    assert_eq!(parse_status(&raw), 200, "{raw}");
    let requests = native.requests();
    let sent: Value = serde_json::from_str(&requests[0].body).expect("backend request JSON");
    let roles: Vec<&str> = sent["messages"]
        .as_array()
        .expect("messages")
        .iter()
        .map(|message| message["role"].as_str().expect("role"))
        .collect();
    assert_eq!(roles, ["user", "assistant", "user", "assistant", "user"]);
    assert_eq!(sent["messages"][1]["content"][0]["type"], "tool_use");
    assert_eq!(sent["messages"][1]["content"][0]["name"], "weather__get_weather");
    assert_eq!(sent["messages"][2]["content"][0]["type"], "tool_result");
    assert!(
        !requests[0].body.contains("mcp_tool_"),
        "no MCP blocks reach the backend"
    );
}

#[test]
fn unconfigured_server_is_rejected_before_inference() {
    let mcp = start_weather_mcp();
    let native = StatefulCapturingBackend::new(vec![(200, native_answer_round())]).start_with_shutdown();
    let proxy = start_proxy(&load_config(free_port(), native.port(), free_port(), &mcp));
    let body = json!({
        "model": NATIVE_MODEL,
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "hi"}],
        "mcp_servers": [{"type": "url", "url": "https://unknown.example/mcp", "name": "other"}],
        "tools": [{"type": "mcp_toolset", "mcp_server_name": "other"}],
    })
    .to_string();

    let raw = http_send(proxy.addr(), &json_post("/v1/messages", &body));

    assert_eq!(parse_status(&raw), 400, "{raw}");
    let error: Value = serde_json::from_str(&parse_body(&raw)).expect("error JSON");
    assert_eq!(error["type"], "error");
    assert_eq!(error["error"]["type"], "invalid_request_error");
    assert!(native.requests().is_empty(), "the backend must not be called");
    assert_eq!(mcp.method_count("tools/list"), 0, "no MCP server is dialed");
}

#[test]
fn streaming_connector_request_is_rejected() {
    let mcp = start_weather_mcp();
    let native = StatefulCapturingBackend::new(vec![(200, native_answer_round())]).start_with_shutdown();
    let proxy = start_proxy(&load_config(free_port(), native.port(), free_port(), &mcp));
    let mut body: Value = serde_json::from_str(&connector_request(NATIVE_MODEL, &mcp)).expect("request JSON");
    body["stream"] = Value::Bool(true);

    let raw = http_send(proxy.addr(), &json_post("/v1/messages", &body.to_string()));

    assert_eq!(parse_status(&raw), 400, "{raw}");
    assert!(parse_body(&raw).contains("streaming"));
    assert!(native.requests().is_empty());
}

#[test]
fn request_without_connector_passes_through() {
    let mcp = start_weather_mcp();
    let native = StatefulCapturingBackend::new(vec![(200, native_answer_round())]).start_with_shutdown();
    let proxy = start_proxy(&load_config(free_port(), native.port(), free_port(), &mcp));
    let body = json!({
        "model": NATIVE_MODEL,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}],
    })
    .to_string();

    let raw = http_send(proxy.addr(), &json_post("/v1/messages", &body));

    assert_eq!(parse_status(&raw), 200, "{raw}");
    let response: Value = serde_json::from_str(&parse_body(&raw)).expect("response JSON");
    assert_eq!(block_types(&response), ["text"]);
    assert_eq!(mcp.method_count("tools/list"), 0);
}

#[test]
fn mixed_round_runs_mcp_calls_and_returns_client_tool_calls() {
    let mcp = start_weather_mcp();
    let mixed_round = json!({
        "id": "msg_1", "type": "message", "role": "assistant", "model": NATIVE_MODEL,
        "content": [
            {"type": "tool_use", "id": "toolu_mcp", "name": "weather__get_weather", "input": {"city": "Paris"}},
            {"type": "tool_use", "id": "toolu_client", "name": "local_clock", "input": {}},
        ],
        "stop_reason": "tool_use", "stop_sequence": null,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    })
    .to_string();
    let native =
        StatefulCapturingBackend::new(vec![(200, mixed_round), (200, native_answer_round())]).start_with_shutdown();
    let proxy = start_proxy(&load_config(free_port(), native.port(), free_port(), &mcp));
    let mut body: Value = serde_json::from_str(&connector_request(NATIVE_MODEL, &mcp)).expect("request JSON");
    body["tools"]
        .as_array_mut()
        .expect("tools")
        .push(json!({"name": "local_clock", "input_schema": {"type": "object"}}));

    let raw = http_send(proxy.addr(), &json_post("/v1/messages", &body.to_string()));

    assert_eq!(parse_status(&raw), 200, "{raw}");
    let response: Value = serde_json::from_str(&parse_body(&raw)).expect("client response JSON");
    assert_eq!(block_types(&response), ["mcp_tool_use", "mcp_tool_result", "tool_use"]);
    assert_eq!(response["content"][2]["name"], "local_clock");
    assert_eq!(
        response["stop_reason"], "tool_use",
        "the client must answer its own tool"
    );
    assert_eq!(
        response["content"][1]["content"][0]["text"],
        "mock result for get_weather"
    );
    assert_eq!(mcp.tool_call_count("get_weather"), 1);
    assert_eq!(native.requests().len(), 1, "no second inference round");
}
