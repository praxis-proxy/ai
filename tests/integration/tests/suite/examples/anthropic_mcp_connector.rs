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
    load_config_with(proxy_port, native_port, chat_port, mcp, |yaml| yaml)
}

fn load_config_with(
    proxy_port: u16,
    native_port: u16,
    chat_port: u16,
    mcp: &McpMockServerGuard,
    configure: impl FnOnce(String) -> String,
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
    praxis_core::config::Config::from_yaml(&configure(yaml)).expect("parse mcp-connector example")
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
fn pending_round_can_resume_at_a_different_filter_index() {
    let mcp = start_weather_mcp();
    let backend = StatefulCapturingBackend::new(vec![(200, native_tool_round()), (200, native_answer_round())])
        .start_with_shutdown();
    let config = load_config_with(free_port(), backend.port(), free_port(), &mcp, |yaml| {
        let (steps, tail) = yaml.split_once("\ninsecure_options:").unwrap();
        let (_, filters) = steps.split_once("          - name: inference\n").unwrap();
        let resumed = filters.replacen(
            "            filters:\n",
            "            filters:\n              - filter: headers\n                request_set:\n                  - {name: x-inference-step, value: resumed}\n",
            1,
        );
        format!(
            "{}          - name: continuation\n{resumed}\ninsecure_options:{tail}",
            steps.replace("next: inference", "next: continuation")
        )
    });
    let proxy = start_proxy(&config);
    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/messages", &connector_request(NATIVE_MODEL, &mcp)),
    );
    assert_eq!(parse_status(&raw), 200, "{raw}");
    assert_mcp_blocks(&serde_json::from_str(&parse_body(&raw)).unwrap());
    assert_eq!(mcp.tool_call_count("get_weather"), 1);
    let requests = backend.requests();
    assert_eq!(requests.len(), 2);
    assert!(requests[1].headers.to_lowercase().contains("x-inference-step: resumed"));
}

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

/// The recorded model round is replayed on both sides of the Chat translator.
/// This extends the single-exchange fixture through a real MCP call and re-entry.
fn fixture_frames(side: &str) -> Vec<Value> {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../fixtures/inference/recordings/synthetic/messages/mcp-model-tool-stream.json"
    ))
    .unwrap();
    fixture["turns"][0][side]["response"]["body"]["frames"]
        .as_array()
        .unwrap()
        .iter()
        .map(|frame| serde_json::from_str(frame["data"].as_str().unwrap()).unwrap())
        .collect()
}

fn event_stream(frames: &[Value]) -> String {
    frames.iter().map(|frame| format!("data: {frame}\n\n")).collect()
}

fn fixture_round(side: &str) -> String {
    let frames = fixture_frames(side);
    let mut stream = if side == "client" {
        frames
            .iter()
            .map(|frame| format!("event: {}\ndata: {frame}\n\n", frame["type"].as_str().unwrap()))
            .collect()
    } else {
        event_stream(&frames)
    };
    if side == "upstream" {
        stream.push_str("data: [DONE]\n\n");
    }
    stream
}

fn stream_request(model: &str, mcp: &McpMockServerGuard) -> String {
    let mut request: Value = serde_json::from_str(&connector_request(model, mcp)).unwrap();
    request["stream"] = Value::Bool(true);
    request.to_string()
}

fn stream_events(raw: &str) -> Vec<Value> {
    assert_eq!(parse_status(raw), 200, "{raw}");
    parse_body(raw)
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .map(|data| serde_json::from_str(data).unwrap())
        .collect()
}

/// Records calls observed by the step's ordinary response-body pipeline.
struct CallObserver {
    observed: std::sync::mpsc::Sender<()>,
    reject: bool,
}

#[async_trait::async_trait]
impl praxis_filter::HttpFilter for CallObserver {
    fn name(&self) -> &'static str {
        "test_mcp_call_observer"
    }

    fn response_body_access(&self) -> praxis_filter::BodyAccess {
        praxis_filter::BodyAccess::ReadOnly(praxis_filter::BodyRequirement::Optional)
    }

    async fn on_request(
        &self,
        _ctx: &mut praxis_filter::HttpFilterContext<'_>,
    ) -> Result<praxis_filter::FilterAction, praxis_filter::FilterError> {
        Ok(praxis_filter::FilterAction::Continue)
    }

    fn on_response_body(
        &self,
        _ctx: &mut praxis_filter::HttpFilterContext<'_>,
        body: &mut Option<bytes::Bytes>,
        _eof: bool,
    ) -> Result<praxis_filter::FilterAction, praxis_filter::FilterError> {
        if let Some(bytes) = body
            && std::str::from_utf8(bytes)?.contains("\"mcp_tool_use\"")
        {
            self.observed.send(())?;
            if self.reject {
                return Err("test observer stopped the managed call".into());
            }
        }
        Ok(praxis_filter::FilterAction::Continue)
    }
}

#[test]
fn step_body_observer_sees_tool_frames_coalesced_with_stop() {
    assert_observed_handoff(false);
}

#[test]
fn rejected_call_chunk_stops_before_mcp_dispatch() {
    assert_observed_handoff(true);
}

fn assert_observed_handoff(reject: bool) {
    let mcp = start_weather_mcp();
    let backend = StatefulCapturingBackend::new(vec![(200, fixture_round("client")), (503, "{}".to_owned())])
        .start_with_shutdown();
    let config = load_config_with(free_port(), backend.port(), free_port(), &mcp, |yaml| {
        yaml.replacen(
            "              - filter: anthropic_mcp_connector",
            "              - filter: test_mcp_call_observer\n              - filter: anthropic_mcp_connector",
            1,
        )
    });
    let (sender, receiver) = std::sync::mpsc::channel();
    let mut registry = praxis_ai::build_full_registry(&praxis_test_utils::test_subrequest_client());
    registry
        .register(
            "test_mcp_call_observer",
            praxis_filter::FilterFactory::Http(std::sync::Arc::new(move |_| {
                Ok(Box::new(CallObserver {
                    observed: sender.clone(),
                    reject,
                }))
            })),
        )
        .unwrap();
    let proxy = praxis_test_utils::start_proxy_with_registry(&config, &registry);
    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/messages", &stream_request(NATIVE_MODEL, &mcp)),
    );
    assert_eq!(
        parse_status(&raw),
        200,
        "the model has already opened the streamed response"
    );
    assert_eq!(
        mcp.tool_call_count("get_weather"),
        usize::from(!reject),
        "only a call accepted by the step observer may execute"
    );
    assert_eq!(
        receiver.try_iter().count(),
        1,
        "the step observer sees the managed call once"
    );
}

#[test]
fn translated_and_native_streams_compose_tool_results_and_final_text() {
    for (model, side) in [(CHAT_MODEL, "upstream"), (NATIVE_MODEL, "client")] {
        let mcp = start_weather_mcp();
        let answer = if model == CHAT_MODEL {
            format!(
                "data: {}\n\ndata: [DONE]\n\n",
                json!({
                    "id":"chatcmpl_final","model":model,"choices":[{"index":0,"delta":{"content":"Sunny."},"finish_reason":"stop"}],
                    "usage":{"prompt_tokens":20,"completion_tokens":7,"total_tokens":27}
                })
            )
        } else {
            [
                json!({"type":"message_start","message":{"id":"msg_final","type":"message","role":"assistant","usage":{"input_tokens":20,"output_tokens":0}}}),
                json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}),
                json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Sunny."}}),
                json!({"type":"content_block_stop","index":0}),
                json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7}}),
                json!({"type":"message_stop"}),
            ].iter().map(|v|format!("data: {v}\n\n")).collect()
        };
        let backend =
            StatefulCapturingBackend::new(vec![(200, fixture_round(side)), (200, answer)]).start_with_shutdown();
        let proxy = start_proxy(&load_config(free_port(), backend.port(), backend.port(), &mcp));
        let raw = http_send(proxy.addr(), &json_post("/v1/messages", &stream_request(model, &mcp)));
        let events = stream_events(&raw);
        let types: Vec<_> = events.iter().map(|e| e["type"].as_str().unwrap()).collect();
        assert_eq!(
            types,
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop"
            ],
            "{model}: {events:?}"
        );
        assert_eq!(events[1]["content_block"]["type"], "mcp_tool_use");
        assert_eq!(events[1]["content_block"]["server_name"], "weather");
        assert_eq!(events[1]["content_block"]["name"], "get_weather");
        assert_eq!(events[5]["index"], 1);
        assert_eq!(events[5]["content_block"]["type"], "mcp_tool_result");
        assert_eq!(
            events[5]["content_block"]["tool_use_id"],
            events[1]["content_block"]["id"]
        );
        assert_eq!(
            events[5]["content_block"]["content"][0]["text"],
            "mock result for get_weather"
        );
        assert_eq!(events[8]["index"], 2);
        assert_eq!(events[8]["delta"]["text"], "Sunny.");
        assert_eq!(events[10]["delta"]["stop_reason"], "end_turn");
        assert_eq!(events[10]["usage"]["input_tokens"], 30);
        assert_eq!(events[10]["usage"]["output_tokens"], 12);
        assert_eq!(mcp.tool_call_count("get_weather"), 1);
        let requests = backend.requests();
        assert_eq!(requests.len(), 2);
        let resumed: Value = serde_json::from_str(&requests[1].body).unwrap();
        assert_eq!(resumed["stream"], true);
        assert!(resumed["messages"].to_string().contains("mock result for get_weather"));
    }
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

#[test]
fn mixed_stream_returns_mcp_results_and_leaves_the_client_tool_open_for_the_caller() {
    let mcp = start_weather_mcp();
    let mut round = fixture_frames("client");
    let before_delta = round.iter().position(|event| event["type"] == "message_delta").unwrap();
    round.splice(before_delta..before_delta, [
        json!({"type":"content_block_start","index":1,"content_block":{"type":"tool_use","name":"local","id":"toolu_local","input":{}}}),
        json!({"type":"content_block_stop","index":1}),
    ]);
    let backend = StatefulCapturingBackend::new(vec![(200, event_stream(&round))]).start_with_shutdown();
    let proxy = start_proxy(&load_config(free_port(), backend.port(), free_port(), &mcp));
    let mut request: Value = serde_json::from_str(&stream_request(NATIVE_MODEL, &mcp)).unwrap();
    request["tools"]
        .as_array_mut()
        .unwrap()
        .push(json!({"name":"local","input_schema":{"type":"object"}}));

    let events = stream_events(&http_send(
        proxy.addr(),
        &json_post("/v1/messages", &request.to_string()),
    ));
    let blocks: Vec<_> = events.iter().filter_map(|e| e.get("content_block")).collect();
    assert_eq!(
        blocks.iter().map(|b| b["type"].as_str().unwrap()).collect::<Vec<_>>(),
        ["mcp_tool_use", "tool_use", "mcp_tool_result"]
    );
    assert_eq!(blocks[1]["id"], "toolu_local");
    assert_eq!(blocks[2]["tool_use_id"], blocks[0]["id"]);
    assert_eq!(events[events.len() - 2]["delta"]["stop_reason"], "tool_use");
    assert_eq!(events.last().unwrap()["type"], "message_stop");
    assert_eq!(mcp.tool_call_count("get_weather"), 1);
    assert_eq!(backend.requests().len(), 1);
}

#[test]
fn incomplete_calls_and_iteration_exhaustion_end_in_one_error_before_dispatch() {
    let round = fixture_frames("client");
    let mut missing_stop = round.clone();
    assert_eq!(missing_stop.pop().unwrap()["type"], "message_stop");
    let mut bad_input = round.clone();
    bad_input[3]["delta"]["partial_json"] = json!("invalid");
    for (source, limit, reason) in [
        (missing_stop, 6, "ended before completion"),
        (round, 1, "iteration or deadline"),
        (bad_input, 6, "malformed"),
    ] {
        let mcp = start_weather_mcp();
        let backend = StatefulCapturingBackend::new(vec![(200, event_stream(&source))]).start_with_shutdown();
        let proxy = start_proxy(&load_config_with(
            free_port(),
            backend.port(),
            free_port(),
            &mcp,
            |yaml| yaml.replace("max_iterations: 6", &format!("max_iterations: {limit}")),
        ));
        let events = stream_events(&http_send(
            proxy.addr(),
            &json_post("/v1/messages", &stream_request(NATIVE_MODEL, &mcp)),
        ));
        let error = events.last().unwrap();
        assert_eq!(error["type"], "error", "{events:?}");
        assert!(error["error"]["message"].as_str().unwrap().contains(reason));
        assert_eq!(events.iter().filter(|e| e["type"] == "error").count(), 1);
        assert!(events.iter().all(|e| e["type"] != "message_stop"));
        assert_eq!(mcp.tool_call_count("get_weather"), 0);
    }
}

#[test]
fn resumed_http_failure_keeps_the_completed_result_and_emits_one_sse_error() {
    let mcp = start_weather_mcp();
    let backend = StatefulCapturingBackend::new(vec![
        (200, fixture_round("client")),
        (503, r#"{"private":"backend marker"}"#.to_owned()),
    ])
    .start_with_shutdown();
    let proxy = start_proxy(&load_config(free_port(), backend.port(), free_port(), &mcp));
    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/messages", &stream_request(NATIVE_MODEL, &mcp)),
    );
    let events = stream_events(&raw);
    assert_eq!(events.last().unwrap()["error"]["type"], "api_error");
    assert_eq!(
        events
            .iter()
            .filter(|e| e["content_block"]["type"] == "mcp_tool_result")
            .count(),
        1
    );
    assert!(events.iter().all(|e| e["type"] != "message_stop"));
    assert!(!raw.contains("backend marker"));
    assert_eq!(mcp.tool_call_count("get_weather"), 1);
}

#[test]
fn streamed_batch_over_result_budget_errors_before_any_tool_runs() {
    let mut round = fixture_frames("client");
    let second = json!({"type":"tool_use","name":"weather__get_weather","id":"toolu_second","input":{"city":"Lyon"}});
    let before_delta = round.iter().position(|event| event["type"] == "message_delta").unwrap();
    round.splice(
        before_delta..before_delta,
        [
            json!({"type":"content_block_start","index":1,"content_block":second}),
            json!({"type":"content_block_stop","index":1}),
        ],
    );
    let mcp = start_weather_mcp();
    let backend = StatefulCapturingBackend::new(vec![(200, event_stream(&round))]).start_with_shutdown();
    let proxy = start_proxy(&load_config_with(
        free_port(),
        backend.port(),
        free_port(),
        &mcp,
        |yaml| {
            yaml.replace("timeout_ms: 30000", "timeout_ms: 30000\n                max_result_bytes: 1024\n                max_total_result_bytes: 1024")
        },
    ));
    let events = stream_events(&http_send(
        proxy.addr(),
        &json_post("/v1/messages", &stream_request(NATIVE_MODEL, &mcp)),
    ));
    let error = events.last().unwrap();
    assert_eq!(error["type"], "error");
    assert!(
        error["error"]["message"]
            .as_str()
            .unwrap()
            .contains("tool results exceed")
    );
    assert_eq!(mcp.tool_call_count("get_weather"), 0);
    assert_eq!(backend.requests().len(), 1);
}

#[test]
fn token_limited_tool_input_stops_without_dispatch() {
    let mut round = fixture_frames("client");
    round[3]["delta"]["partial_json"] = json!("\"Pa");
    round.iter_mut().find(|event| event["type"] == "message_delta").unwrap()["delta"]["stop_reason"] =
        json!("max_tokens");
    let mcp = start_weather_mcp();
    let backend = StatefulCapturingBackend::new(vec![(200, event_stream(&round))]).start_with_shutdown();
    let proxy = start_proxy(&load_config(free_port(), backend.port(), free_port(), &mcp));
    let events = stream_events(&http_send(
        proxy.addr(),
        &json_post("/v1/messages", &stream_request(NATIVE_MODEL, &mcp)),
    ));
    assert_eq!(events[events.len() - 2]["delta"]["stop_reason"], "max_tokens");
    assert_eq!(events.last().unwrap()["type"], "message_stop");
    assert_eq!(mcp.tool_call_count("get_weather"), 0);
}

#[test]
fn oversized_reentry_reports_the_executed_call_and_ends_with_an_error() {
    let mcp = start_weather_mcp();
    let backend = StatefulCapturingBackend::new(vec![(200, fixture_round("client"))]).start_with_shutdown();
    let proxy = start_proxy(&load_config_with(
        free_port(),
        backend.port(),
        free_port(),
        &mcp,
        |yaml| {
            yaml.replace(
                "timeout_ms: 30000",
                "timeout_ms: 30000\n                max_body_bytes: 2048",
            )
        },
    ));
    let mut request: Value = serde_json::from_str(&stream_request(NATIVE_MODEL, &mcp)).unwrap();
    request["messages"][0]["content"] = Value::from("x".repeat(1700));
    let events = stream_events(&http_send(
        proxy.addr(),
        &json_post("/v1/messages", &request.to_string()),
    ));
    assert_eq!(mcp.tool_call_count("get_weather"), 1);
    assert_eq!(backend.requests().len(), 1);
    assert!(events.iter().any(|e| e["content_block"]["type"] == "mcp_tool_result"));
    let last = events.last().unwrap();
    assert_eq!(last["type"], "error");
    assert!(
        last["error"]["message"]
            .as_str()
            .unwrap()
            .contains("request body exceeds")
    );
}

#[test]
fn retained_round_over_router_budget_fails_before_dispatch() {
    let mcp = start_weather_mcp();
    let mut round = fixture_frames("client");
    round[3]["delta"]["partial_json"] = json!(format!("\"{}\"}}", "a".repeat(8192)));
    let backend = StatefulCapturingBackend::new(vec![(200, event_stream(&round))]).start_with_shutdown();
    let proxy = start_proxy(&load_config_with(
        free_port(),
        backend.port(),
        free_port(),
        &mcp,
        |yaml| yaml.replace("max_iterations: 6", "max_iterations: 6\n        max_state_bytes: 4096"),
    ));
    let events = stream_events(&http_send(
        proxy.addr(),
        &json_post("/v1/messages", &stream_request(NATIVE_MODEL, &mcp)),
    ));
    assert_eq!(events.last().unwrap()["type"], "error");
    assert_eq!(events.iter().filter(|event| event["type"] == "error").count(), 1);
    assert!(
        events.last().unwrap()["error"]["message"]
            .as_str()
            .unwrap()
            .contains("retained-state")
    );
    assert_eq!(mcp.tool_call_count("get_weather"), 0);
    assert_eq!(backend.requests().len(), 1);
}

#[test]
fn first_inference_connection_failure_returns_a_messages_http_error() {
    let mcp = start_weather_mcp();
    let proxy = start_proxy(&load_config(free_port(), free_port(), free_port(), &mcp));
    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/messages", &stream_request(NATIVE_MODEL, &mcp)),
    );
    assert_eq!(parse_status(&raw), 502, "{raw}");
    assert!(raw.to_ascii_lowercase().contains("content-type: application/json"));
    let body: Value = serde_json::from_str(&parse_body(&raw)).unwrap();
    assert_eq!(body["error"]["type"], "api_error");
    assert_eq!(mcp.tool_call_count("get_weather"), 0);
}
