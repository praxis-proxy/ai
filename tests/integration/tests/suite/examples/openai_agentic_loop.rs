// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Integration tests for the openai_agentic_loop filter with
//! `iterative_request_router`.
//!
//! These tests verify that IRR, request-supplied MCP resolution,
//! MCP dispatch, and the agentic inference loop function together.

use std::collections::HashMap;

use praxis_test_utils::{
    McpMockConfig, McpToolFixture, StatefulCapturingBackend, build_pipeline, example_config_path, free_port, http_send,
    json_post, parse_body, parse_status, patch_yaml, start_mcp_mock_server_with_config, start_proxy,
};

// -----------------------------------------------------------------------------
// Pipeline Build
// -----------------------------------------------------------------------------

#[test]
fn example_config_builds_pipeline() {
    let config = load_agentic_config(free_port(), 19901);
    let _pipeline = build_pipeline(&config);
}

// -----------------------------------------------------------------------------
// Single-Pass
// -----------------------------------------------------------------------------

#[test]
fn single_pass_completes_through_irr() {
    let response = r#"{"id":"resp_1","object":"response","status":"completed","output":[]}"#;
    let model = StatefulCapturingBackend::new(vec![(200, response.to_owned())]).start_with_shutdown();
    let proxy_port = free_port();

    let config = load_agentic_config(proxy_port, model.port());
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"Hello"}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(
        parse_status(&raw),
        200,
        "single-pass request through IRR should return 200"
    );

    let model_reqs = model.requests();
    assert_eq!(model_reqs.len(), 1, "model backend should receive one request");
    let model_body: serde_json::Value =
        serde_json::from_str(&model_reqs[0].body).expect("model request body should be valid JSON");
    assert_eq!(
        model_body["parallel_tool_calls"], false,
        "first inference must disable parallel tool calls when the client omits the field"
    );
}

#[test]
fn explicit_false_preserves_original_request_bytes() {
    let response = r#"{"id":"resp_1","object":"response","status":"completed","output":[]}"#;
    let model = StatefulCapturingBackend::new(vec![(200, response.to_owned())]).start_with_shutdown();
    let proxy_port = free_port();

    let config = load_agentic_config(proxy_port, model.port());
    let proxy = start_proxy(&config);

    let body = r#"{ "model": "gpt-4.1", "input": [{"role":"user","content":"Hello"}], "parallel_tool_calls": false }"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200);
    let model_reqs = model.requests();
    assert_eq!(model_reqs.len(), 1, "model backend should receive one request");
    assert_eq!(
        model_reqs[0].body, body,
        "an already-disabled request should retain byte-exact passthrough"
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
    let config = load_agentic_config(proxy_port, model.port());
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
// IRR Rejection Preservation (regression for #663)
// -----------------------------------------------------------------------------
//
// The agentic-loop filter rejects (400/508) from its `on_response_body` hook,
// which runs inside IRR. These tests assert IRR surfaces that rejection as a
// client-visible status instead of aborting the response body.
// https://github.com/praxis-proxy/ai/issues/663

#[test]
fn multiple_function_calls_returns_client_visible_400() {
    let response = serde_json::json!({
        "id": "resp_parallel_calls",
        "object": "response",
        "status": "completed",
        "output": [
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": r#"{"location":"SF"}"#,
                "status": "completed"
            },
            {
                "type": "function_call",
                "id": "fc_2",
                "call_id": "call_2",
                "name": "get_time",
                "arguments": r#"{"timezone":"PST"}"#,
                "status": "completed"
            }
        ]
    });
    let model = StatefulCapturingBackend::new(vec![(200, response.to_string())]).start_with_shutdown();
    let proxy_port = free_port();
    let config = load_agentic_rejection_config(proxy_port, model.port());
    let proxy = start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", r#"{"model":"gpt-4.1","input":"Hello"}"#),
    );

    assert_eq!(
        parse_status(&raw),
        400,
        "IRR must preserve the response-body rejection status: {raw}"
    );
    let body: serde_json::Value = serde_json::from_str(&parse_body(&raw)).expect("rejection body should be valid JSON");
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert_eq!(
        body["error"]["message"],
        "openai_agentic_loop supports exactly one function call per round"
    );
    assert_eq!(model.requests().len(), 1, "the rejection must stop iteration");
}

#[test]
fn iteration_limit_returns_client_visible_508() {
    let function_response = |id: &str, call_id: &str| {
        serde_json::json!({
            "id": id,
            "object": "response",
            "status": "completed",
            "output": [{
                "type": "function_call",
                "id": format!("fc_{call_id}"),
                "call_id": call_id,
                "name": "get_weather",
                "arguments": r#"{"location":"SF"}"#,
                "status": "completed"
            }]
        })
        .to_string()
    };
    let model = StatefulCapturingBackend::new(vec![
        (200, function_response("resp_1", "call_1")),
        (200, function_response("resp_2", "call_2")),
    ])
    .start_with_shutdown();
    let proxy_port = free_port();
    let config = load_agentic_rejection_config(proxy_port, model.port());
    let proxy = start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", r#"{"model":"gpt-4.1","input":"Hello"}"#),
    );

    assert_eq!(
        parse_status(&raw),
        508,
        "IRR must preserve the iteration-limit rejection status: {raw}"
    );
    let body: serde_json::Value = serde_json::from_str(&parse_body(&raw)).expect("rejection body should be valid JSON");
    assert_eq!(body["error"]["type"], "server_error");
    assert_eq!(body["error"]["message"], "agentic loop iteration limit exceeded");
    assert_eq!(model.requests().len(), 2, "one loop is allowed before the limit");
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
        "parallel_tool_calls": true,
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
    assert_eq!(
        first_model_body["parallel_tool_calls"], false,
        "first inference must override parallel_tool_calls=true"
    );
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
    // Assert openai_agentic_loop bookkeeping in second request
    // -------------------------------------------------------------------------
    assert_eq!(
        model_body["parallel_tool_calls"], false,
        "openai_agentic_loop must force parallel_tool_calls=false on re-entry"
    );
    assert_eq!(
        model_body["tool_choice"], "auto",
        "openai_agentic_loop must reset tool_choice to auto on re-entry"
    );
}

// -----------------------------------------------------------------------------
// Round-Trip: Web Search via IRR
// -----------------------------------------------------------------------------

#[test]
fn web_search_round_trip_executes_and_re_enters_inference() {
    let first_response = serde_json::json!({
        "id": "resp_ws_1",
        "object": "response",
        "status": "completed",
        "output": [{
            "type": "web_search_call",
            "id": "ws_1",
            "status": "completed",
            "action": {"type": "search", "query": "Rust 2025 edition"}
        }]
    });
    let second_response = serde_json::json!({
        "id": "resp_ws_2",
        "object": "response",
        "status": "completed",
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Rust 2025 brings great features."}]
        }]
    });

    let model = StatefulCapturingBackend::new(vec![
        (200, serde_json::to_string(&first_response).unwrap()),
        (200, serde_json::to_string(&second_response).unwrap()),
    ])
    .start_with_shutdown();

    let search_listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let search_port = search_listener.local_addr().unwrap().port();
    spawn_search_mock(search_listener);

    let proxy_port = free_port();
    let config = load_web_search_config(proxy_port, model.port(), search_port);
    let proxy = start_proxy(&config);

    let request_body = serde_json::json!({
        "model": "gpt-4.1",
        "input": "Search for Rust 2025 edition features",
        "tools": [{"type": "web_search_preview"}]
    });
    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", &serde_json::to_string(&request_body).unwrap()),
    );

    assert_eq!(parse_status(&raw), 200, "web search round-trip should return 200");
    let body = parse_body(&raw);
    let response: serde_json::Value = serde_json::from_str(&body).expect("response should be JSON");
    assert_eq!(
        response["id"], "resp_ws_2",
        "final response should be the second model response after web search"
    );

    // A successful search updates the model's placeholder in place, so the public
    // response carries exactly one completed web_search_call for ws_1.
    let output = response["output"].as_array().expect("final response output array");
    let search_calls: Vec<&serde_json::Value> =
        output.iter().filter(|item| item["type"] == "web_search_call").collect();
    assert_eq!(
        search_calls.len(),
        1,
        "final response must contain exactly one web_search_call, got: {output:#?}"
    );
    assert_eq!(search_calls[0]["id"], "ws_1");
    assert_eq!(search_calls[0]["status"], "completed");

    let model_reqs = model.requests();
    assert_eq!(
        model_reqs.len(),
        2,
        "model backend should receive exactly two requests (initial + post-search)"
    );

    let second_body: serde_json::Value =
        serde_json::from_str(&model_reqs[1].body).expect("second model request should be valid JSON");
    let input = second_body["input"]
        .as_array()
        .expect("second model request input should be an array");

    // #808: a hosted web_search_call is not a valid OpenResponses input item
    // (vLLM's Harmony conversion rejects it with HTTP 400), so the continuation
    // must never forward it to the inference backend.
    assert!(
        input.iter().all(|item| item["type"] != "web_search_call"),
        "second inference input must not contain hosted web_search_call items: {input:?}"
    );

    // The search result reaches the model through a backend-valid
    // function_call / function_call_output bridge instead.
    let has_web_search_call = input
        .iter()
        .any(|item| item["type"] == "function_call" && item["name"] == "web_search");
    assert!(
        has_web_search_call,
        "second inference input should carry a synthetic web_search function_call: {input:?}"
    );
    let function_output = input
        .iter()
        .find(|item| item["type"] == "function_call_output")
        .expect("second inference input should contain a function_call_output");
    assert!(
        function_output["output"]
            .as_str()
            .is_some_and(|output| output.contains("blog.rust-lang.org")),
        "second inference should receive the web search results: {function_output:?}"
    );
}

#[test]
fn web_search_provider_failure_continues_loop_with_failed_result() {
    let first_response = serde_json::json!({
        "id": "resp_ws_1",
        "object": "response",
        "status": "completed",
        "output": [{
            "type": "web_search_call",
            "id": "ws_1",
            "status": "completed",
            "action": {"type": "search", "query": "Rust 2025 edition"}
        }]
    });
    let second_response = serde_json::json!({
        "id": "resp_ws_2",
        "object": "response",
        "status": "completed",
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "I could not search, but here is what I know."}]
        }]
    });

    let model = StatefulCapturingBackend::new(vec![
        (200, serde_json::to_string(&first_response).unwrap()),
        (200, serde_json::to_string(&second_response).unwrap()),
    ])
    .start_with_shutdown();

    let search_listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let search_port = search_listener.local_addr().unwrap().port();
    spawn_failing_search_mock(search_listener);

    let proxy_port = free_port();
    let config = load_web_search_config(proxy_port, model.port(), search_port);
    let proxy = start_proxy(&config);

    let request_body = serde_json::json!({
        "model": "gpt-4.1",
        "input": "Search for Rust 2025 edition features",
        "tools": [{"type": "web_search_preview"}]
    });
    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", &serde_json::to_string(&request_body).unwrap()),
    );

    assert_eq!(
        parse_status(&raw),
        200,
        "a provider failure must not reject the Response"
    );
    let response: serde_json::Value = serde_json::from_str(&parse_body(&raw)).expect("response should be JSON");
    assert_eq!(
        response["id"], "resp_ws_2",
        "the loop must continue to a second inference after the search fails"
    );

    // The public response must carry exactly one web_search_call for ws_1, marked
    // failed — not a contradictory completed placeholder plus a failed duplicate.
    let output = response["output"].as_array().expect("final response output array");
    let search_calls: Vec<&serde_json::Value> =
        output.iter().filter(|item| item["type"] == "web_search_call").collect();
    assert_eq!(
        search_calls.len(),
        1,
        "final response must contain exactly one web_search_call, got: {output:#?}"
    );
    assert_eq!(search_calls[0]["id"], "ws_1");
    assert_eq!(
        search_calls[0]["status"], "failed",
        "the single web_search_call must reflect the failed outcome"
    );

    let model_reqs = model.requests();
    assert_eq!(
        model_reqs.len(),
        2,
        "model backend should receive two requests (initial + post-failure)"
    );

    let second_body: serde_json::Value =
        serde_json::from_str(&model_reqs[1].body).expect("second model request should be valid JSON");
    let input = second_body["input"]
        .as_array()
        .expect("second model request input should be an array");
    // The model receives the failure through a backend-valid function_call_output
    // bridge carrying the bounded notice — never a hosted web_search_call, which
    // is not a valid OpenResponses input (issue #808).
    assert!(
        input.iter().all(|item| item["type"] != "web_search_call"),
        "the continuation must not feed the model a hosted web_search_call: {input:#?}"
    );
    let has_failure_notice = input
        .iter()
        .any(|item| item["type"] == "function_call_output" && item["output"] == "Web search unavailable.");
    assert!(
        has_failure_notice,
        "the model must receive a truthful failure notice via function_call_output: {input:#?}"
    );
}

fn spawn_search_mock(listener: std::net::TcpListener) {
    use std::io::{Read as _, Write as _};
    let body = serde_json::json!({
        "web": {
            "results": [{
                "title": "Rust 2025 Edition",
                "url": "https://blog.rust-lang.org/2025",
                "description": "The Rust 2025 edition is here."
            }]
        }
    })
    .to_string();
    std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut buf = [0_u8; 4096];
        let _n = stream.read(&mut buf).unwrap();
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        stream.write_all(response.as_bytes()).unwrap();
    });
}

/// Serve a single 5xx so the search client maps the callout to a failed outcome.
fn spawn_failing_search_mock(listener: std::net::TcpListener) {
    use std::io::{Read as _, Write as _};
    let body = r#"{"error":"service unavailable"}"#;
    std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut buf = [0_u8; 4096];
        let _n = stream.read(&mut buf).unwrap();
        let response = format!(
            "HTTP/1.1 503 Service Unavailable\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        stream.write_all(response.as_bytes()).unwrap();
    });
}

fn load_web_search_config(proxy_port: u16, model_port: u16, search_port: u16) -> praxis_core::config::Config {
    let path = example_config_path("openai/responses/agentic-loop.yaml");
    let yaml = std::fs::read_to_string(path).expect("read agentic-loop example");
    let yaml = patch_yaml(&yaml, proxy_port, &HashMap::from([("127.0.0.1:3001", model_port)]));
    let yaml = yaml.replace(
        "api_key: ${WEB_SEARCH_API_KEY}",
        &format!(
            "api_key: test-key\n                base_url: http://127.0.0.1:{search_port}\n                allow_private_base_url: true"
        ),
    );
    praxis_core::config::Config::from_yaml(&yaml).expect("parse web search config")
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

fn patch_web_search_api_key(yaml: &str) -> String {
    yaml.replace("api_key: ${WEB_SEARCH_API_KEY}", "api_key: test-key")
}

fn load_agentic_config(proxy_port: u16, model_port: u16) -> praxis_core::config::Config {
    let path = example_config_path("openai/responses/agentic-loop.yaml");
    let yaml = std::fs::read_to_string(path).expect("read agentic-loop example");
    let yaml = patch_yaml(&yaml, proxy_port, &HashMap::from([("127.0.0.1:3001", model_port)]));
    let yaml = patch_web_search_api_key(&yaml);
    praxis_core::config::Config::from_yaml(&yaml).expect("parse agentic-loop config")
}

fn load_agentic_rejection_config(proxy_port: u16, model_port: u16) -> praxis_core::config::Config {
    let path = example_config_path("openai/responses/agentic-loop-fixture.yaml");
    let yaml = std::fs::read_to_string(path).expect("read agentic-loop fixture");
    let yaml = patch_yaml(&yaml, proxy_port, &HashMap::from([("127.0.0.1:3001", model_port)]));
    // Route action=loop back to inference so the loop re-enters and can reach the
    // iteration limit (508). Without this, IRR terminates after the first pass via
    // the default `done: true` branch.
    let terminal_on_result = "            on_result:\n              - default: true\n                done: true";
    let looping_on_result = "            on_result:\n              - filter: openai_agentic_loop\n                key: action\n                value: loop\n                next: inference\n              - default: true\n                done: true";
    let patched = yaml.replacen(terminal_on_result, looping_on_result, 1);
    assert_ne!(
        patched, yaml,
        "expected to inject the loop action into agentic-loop-fixture.yaml; its on_result block may have changed"
    );
    praxis_core::config::Config::from_yaml(&patched).expect("parse agentic-loop rejection config")
}

fn load_loopback_mcp_config(proxy_port: u16, model_port: u16) -> praxis_core::config::Config {
    let path = example_config_path("openai/responses/agentic-loop.yaml");
    let yaml = std::fs::read_to_string(path).expect("read agentic-loop example");
    let yaml = patch_yaml(&yaml, proxy_port, &HashMap::from([("127.0.0.1:3001", model_port)]));
    let yaml = patch_web_search_api_key(&yaml);
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
