// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Integration tests for Anthropic Messages API filters.
//!
//! Tests validate the full request/response cycle through Praxis with
//! passthrough and transform filter chains. Response data is loaded
//! from recording fixtures in `tests/integration/fixtures/anthropic/messages/`.

use praxis_core::config::Config;
use praxis_test_utils::{
    Backend, Recording, free_port, http_send, json_post, parse_body, parse_header, parse_status,
    start_backend_with_shutdown, start_proxy,
};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn non_streaming_basic() {
    let recording = Recording::load("anthropic/messages/basic.json");
    let response_body = recording.response_body();
    let request_body = recording.request_body();
    let backend = Backend::fixed(&response_body)
        .header("content-type", "application/json")
        .header("anthropic-version", "2023-06-01")
        .start_with_shutdown();
    let proxy_port = free_port();
    let config = Config::from_yaml(&passthrough_yaml(proxy_port, backend.port())).unwrap();
    let proxy = start_proxy(&config);

    let raw = http_send(proxy.addr(), &anthropic_post("/v1/messages", &request_body));
    let status = parse_status(&raw);
    let body = parse_body(&raw);
    let data: serde_json::Value = serde_json::from_str(&body).unwrap();

    assert_eq!(status, 200, "expected 200");
    assert_eq!(data["type"], "message", "type should be message");
    assert_eq!(data["role"], "assistant", "role should be assistant");
    assert!(
        data["id"].as_str().unwrap().starts_with("msg_"),
        "id should start with msg_"
    );
    let content = data["content"].as_array().unwrap();
    assert!(!content.is_empty(), "content should not be empty");
    let text_blocks: Vec<_> = content.iter().filter(|b| b["type"] == "text").collect();
    assert!(!text_blocks.is_empty(), "should have at least one text block");
    assert!(
        !text_blocks[0]["text"].as_str().unwrap().is_empty(),
        "text should not be empty"
    );
    assert!(
        ["end_turn", "max_tokens"].contains(&data["stop_reason"].as_str().unwrap()),
        "stop_reason should be end_turn or max_tokens"
    );
    assert!(
        data["usage"]["input_tokens"].as_u64().unwrap() > 0,
        "input_tokens should be > 0"
    );
    assert!(
        data["usage"]["output_tokens"].as_u64().unwrap() > 0,
        "output_tokens should be > 0"
    );
    for block in content {
        assert!(
            ["text", "thinking", "tool_use"].contains(&block["type"].as_str().unwrap()),
            "block type should be text, thinking, or tool_use"
        );
    }
}

#[test]
fn non_streaming_with_system() {
    let recording = Recording::load("anthropic/messages/system.json");
    let response_body = recording.response_body();
    let request_body = recording.request_body();
    let backend = Backend::fixed(&response_body)
        .header("content-type", "application/json")
        .header("anthropic-version", "2023-06-01")
        .start_with_shutdown();
    let proxy_port = free_port();
    let config = Config::from_yaml(&passthrough_yaml(proxy_port, backend.port())).unwrap();
    let proxy = start_proxy(&config);

    let raw = http_send(proxy.addr(), &anthropic_post("/v1/messages", &request_body));
    let status = parse_status(&raw);
    let body = parse_body(&raw);
    let data: serde_json::Value = serde_json::from_str(&body).unwrap();

    assert_eq!(status, 200, "expected 200");
    assert_eq!(data["type"], "message", "type should be message");
    let content = data["content"].as_array().unwrap();
    assert!(!content.is_empty(), "content should not be empty");
    let text_blocks: Vec<_> = content.iter().filter(|b| b["type"] == "text").collect();
    assert!(!text_blocks.is_empty(), "should have at least one text block");
    assert!(
        !text_blocks[0]["text"].as_str().unwrap().is_empty(),
        "text should not be empty"
    );
}

#[test]
fn non_streaming_multi_turn() {
    let recording = Recording::load("anthropic/messages/multi_turn.json");
    let response_body = recording.response_body();
    let request_body = recording.request_body();
    let backend = Backend::fixed(&response_body)
        .header("content-type", "application/json")
        .header("anthropic-version", "2023-06-01")
        .start_with_shutdown();
    let proxy_port = free_port();
    let config = Config::from_yaml(&passthrough_yaml(proxy_port, backend.port())).unwrap();
    let proxy = start_proxy(&config);

    let raw = http_send(proxy.addr(), &anthropic_post("/v1/messages", &request_body));
    let status = parse_status(&raw);
    let body = parse_body(&raw);
    let data: serde_json::Value = serde_json::from_str(&body).unwrap();

    assert_eq!(status, 200, "expected 200");
    assert_eq!(data["type"], "message", "type should be message");
    let content = data["content"].as_array().unwrap();
    assert!(!content.is_empty(), "content should not be empty");
    let text_blocks: Vec<_> = content.iter().filter(|b| b["type"] == "text").collect();
    assert!(!text_blocks.is_empty(), "should have at least one text block");
    let text = text_blocks[0]["text"].as_str().unwrap().to_lowercase();
    assert!(text.contains("alice"), "response should mention Alice");
}

#[test]
fn streaming_basic() {
    let recording = Recording::load("anthropic/messages/streaming_basic.json");
    let response_body = recording.response_body();
    let request_body = recording.request_body();
    let backend = Backend::fixed(&response_body)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .start_with_shutdown();
    let proxy_port = free_port();
    let config = Config::from_yaml(&passthrough_yaml(proxy_port, backend.port())).unwrap();
    let proxy = start_proxy(&config);

    let raw = http_send(proxy.addr(), &anthropic_post("/v1/messages", &request_body));
    let body = parse_body(&raw);

    let events = parse_sse_events(&body);
    let event_types: Vec<&str> = events.iter().filter_map(|e| e["_event_type"].as_str()).collect();

    assert!(event_types.contains(&"message_start"), "should have message_start");
    assert!(event_types.contains(&"message_stop"), "should have message_stop");

    let msg_start = events.iter().find(|e| e["_event_type"] == "message_start").unwrap();
    assert_eq!(
        msg_start["message"]["role"], "assistant",
        "message_start role should be assistant"
    );

    let content_deltas: Vec<_> = events
        .iter()
        .filter(|e| e["_event_type"] == "content_block_delta")
        .collect();
    assert!(
        !content_deltas.is_empty(),
        "should have at least one content_block_delta"
    );

    for delta in &content_deltas {
        assert!(
            ["text_delta", "thinking_delta"].contains(&delta["delta"]["type"].as_str().unwrap()),
            "delta type should be text_delta or thinking_delta"
        );
    }
}

#[test]
fn streaming_collects_full_text() {
    let recording = Recording::load("anthropic/messages/streaming_basic.json");
    let response_body = recording.response_body();
    let backend = Backend::fixed(&response_body)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .start_with_shutdown();
    let proxy_port = free_port();
    let config = Config::from_yaml(&passthrough_yaml(proxy_port, backend.port())).unwrap();
    let proxy = start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        &anthropic_post(
            "/v1/messages",
            r#"{"model":"mock-model","messages":[{"role":"user","content":"Count from 1 to 5, separated by commas."}],"max_tokens":64,"stream":true}"#,
        ),
    );
    let body = parse_body(&raw);
    let events = parse_sse_events(&body);

    let full_text: String = events
        .iter()
        .filter(|e| e["_event_type"] == "content_block_delta")
        .filter(|e| e["delta"]["type"] == "text_delta")
        .filter_map(|e| e["delta"]["text"].as_str())
        .collect();

    assert!(!full_text.is_empty(), "collected text should not be empty");
}

#[test]
fn streaming_tool_calls_within_cap_completes() {
    let backend = Backend::fixed(&tool_call_stream_sse())
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .start_with_shutdown();
    let proxy_port = free_port();
    let config = Config::from_yaml(&transform_yaml(proxy_port, backend.port(), 5)).unwrap();
    let proxy = start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        &anthropic_post(
            "/v1/messages",
            r#"{"model":"mock-model","messages":[{"role":"user","content":"call tools"}],"max_tokens":64,"stream":true}"#,
        ),
    );
    let body = parse_body(&raw);

    // Three distinct tool-call indices open three blocks, all under the
    // cap of 5, so the transform runs to completion.
    assert!(
        body.contains("event: message_stop"),
        "a tool-call stream within max_tool_blocks should complete with message_stop; body: {body}"
    );
}

#[test]
fn streaming_tool_calls_exceeding_cap_fails_closed() {
    let backend = Backend::fixed(&tool_call_stream_sse())
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .start_with_shutdown();
    let proxy_port = free_port();
    // Same stream as the within-cap test; only the cap changes. The third
    // distinct tool-call index exceeds max_tool_blocks and fails closed.
    let config = Config::from_yaml(&transform_yaml(proxy_port, backend.port(), 2)).unwrap();
    let proxy = start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        &anthropic_post(
            "/v1/messages",
            r#"{"model":"mock-model","messages":[{"role":"user","content":"call tools"}],"max_tokens":64,"stream":true}"#,
        ),
    );
    let body = parse_body(&raw);

    assert!(
        !body.contains("event: message_stop"),
        "exceeding max_tool_blocks should fail the stream closed before message_stop; body: {body}"
    );
}

#[test]
fn non_streaming_with_temperature() {
    let recording = Recording::load("anthropic/messages/temperature.json");
    let response_body = recording.response_body();
    let request_body = recording.request_body();
    let backend = Backend::fixed(&response_body)
        .header("content-type", "application/json")
        .header("anthropic-version", "2023-06-01")
        .start_with_shutdown();
    let proxy_port = free_port();
    let config = Config::from_yaml(&passthrough_yaml(proxy_port, backend.port())).unwrap();
    let proxy = start_proxy(&config);

    let raw = http_send(proxy.addr(), &anthropic_post("/v1/messages", &request_body));
    let status = parse_status(&raw);
    let body = parse_body(&raw);
    let data: serde_json::Value = serde_json::from_str(&body).unwrap();

    assert_eq!(status, 200, "expected 200");
    assert_eq!(data["type"], "message", "type should be message");
    assert!(
        !data["content"].as_array().unwrap().is_empty(),
        "content should not be empty"
    );
}

#[test]
fn non_streaming_with_stop_sequences() {
    let recording = Recording::load("anthropic/messages/stop_sequences.json");
    let response_body = recording.response_body();
    let request_body = recording.request_body();
    let backend = Backend::fixed(&response_body)
        .header("content-type", "application/json")
        .header("anthropic-version", "2023-06-01")
        .start_with_shutdown();
    let proxy_port = free_port();
    let config = Config::from_yaml(&passthrough_yaml(proxy_port, backend.port())).unwrap();
    let proxy = start_proxy(&config);

    let raw = http_send(proxy.addr(), &anthropic_post("/v1/messages", &request_body));
    let status = parse_status(&raw);
    let body = parse_body(&raw);
    let data: serde_json::Value = serde_json::from_str(&body).unwrap();

    assert_eq!(status, 200, "expected 200");
    assert_eq!(data["type"], "message", "type should be message");
}

#[test]
fn with_tool_definitions() {
    let recording = Recording::load("anthropic/messages/tool_defs.json");
    let response_body = recording.response_body();
    let request_body = recording.request_body();
    let backend = Backend::fixed(&response_body)
        .header("content-type", "application/json")
        .header("anthropic-version", "2023-06-01")
        .start_with_shutdown();
    let proxy_port = free_port();
    let config = Config::from_yaml(&passthrough_yaml(proxy_port, backend.port())).unwrap();
    let proxy = start_proxy(&config);

    let raw = http_send(proxy.addr(), &anthropic_post("/v1/messages", &request_body));
    let status = parse_status(&raw);
    let body = parse_body(&raw);
    let data: serde_json::Value = serde_json::from_str(&body).unwrap();

    assert_eq!(status, 200, "expected 200");
    assert_eq!(data["type"], "message", "type should be message");
    let content = data["content"].as_array().unwrap();
    assert!(!content.is_empty(), "content should not be empty");

    for block in content {
        assert!(
            ["text", "tool_use", "thinking"].contains(&block["type"].as_str().unwrap()),
            "block type should be text, tool_use, or thinking"
        );
        if block["type"] == "tool_use" {
            assert!(block["id"].is_string(), "tool_use should have id");
            assert_eq!(block["name"], "get_weather", "tool name should be get_weather");
            assert!(block["input"].is_object(), "tool_use should have input");
        }
    }
}

#[test]
fn tool_use_round_trip() {
    let recording = Recording::load("anthropic/messages/tool_result.json");
    let response_body = recording.response_body();
    let request_body = recording.request_body();
    let backend = Backend::fixed(&response_body)
        .header("content-type", "application/json")
        .header("anthropic-version", "2023-06-01")
        .start_with_shutdown();
    let proxy_port = free_port();
    let config = Config::from_yaml(&passthrough_yaml(proxy_port, backend.port())).unwrap();
    let proxy = start_proxy(&config);

    let raw = http_send(proxy.addr(), &anthropic_post("/v1/messages", &request_body));
    let status = parse_status(&raw);
    let body = parse_body(&raw);
    let data: serde_json::Value = serde_json::from_str(&body).unwrap();

    assert_eq!(status, 200, "expected 200");
    assert_eq!(data["type"], "message", "type should be message");
    assert!(
        !data["content"].as_array().unwrap().is_empty(),
        "content should not be empty"
    );
}

#[test]
fn backend_owned_missing_model_reaches_backend() {
    let backend = start_backend_with_shutdown("backend-owned");
    let proxy_port = free_port();
    let config = Config::from_yaml(&passthrough_yaml(proxy_port, backend.port())).unwrap();
    let proxy = start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        &anthropic_post(
            "/v1/messages",
            r#"{"messages":[{"role":"user","content":"Hello"}],"max_tokens":64}"#,
        ),
    );
    let status = parse_status(&raw);

    assert_eq!(status, 200, "backend-owned missing model semantics should be forwarded");
    assert_eq!(parse_body(&raw), "backend-owned", "request should reach the backend");
}

#[test]
fn backend_owned_empty_messages_reaches_backend() {
    let backend = start_backend_with_shutdown("backend-owned");
    let proxy_port = free_port();
    let config = Config::from_yaml(&passthrough_yaml(proxy_port, backend.port())).unwrap();
    let proxy = start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        &anthropic_post(
            "/v1/messages",
            r#"{"model":"mock-model","messages":[],"max_tokens":64}"#,
        ),
    );
    let status = parse_status(&raw);

    assert_eq!(
        status, 200,
        "backend-owned empty messages semantics should be forwarded"
    );
    assert_eq!(parse_body(&raw), "backend-owned", "request should reach the backend");
}

#[test]
fn response_headers() {
    let recording = Recording::load("anthropic/messages/response_headers.json");
    let response_body = recording.response_body();
    let request_body = recording.request_body();
    let backend = Backend::fixed(&response_body)
        .header("content-type", "application/json")
        .header("anthropic-version", "2023-06-01")
        .start_with_shutdown();
    let proxy_port = free_port();
    let config = Config::from_yaml(&passthrough_yaml(proxy_port, backend.port())).unwrap();
    let proxy = start_proxy(&config);

    let raw = http_send(proxy.addr(), &anthropic_post("/v1/messages", &request_body));
    let status = parse_status(&raw);

    assert_eq!(status, 200, "expected 200");
    let raw_lower = raw.to_lowercase();
    assert!(
        raw_lower.contains("anthropic-version: 2023-06-01"),
        "response should include anthropic-version header"
    );
}

#[test]
fn content_block_array() {
    let recording = Recording::load("anthropic/messages/content_block.json");
    let response_body = recording.response_body();
    let request_body = recording.request_body();
    let backend = Backend::fixed(&response_body)
        .header("content-type", "application/json")
        .header("anthropic-version", "2023-06-01")
        .start_with_shutdown();
    let proxy_port = free_port();
    let config = Config::from_yaml(&passthrough_yaml(proxy_port, backend.port())).unwrap();
    let proxy = start_proxy(&config);

    let raw = http_send(proxy.addr(), &anthropic_post("/v1/messages", &request_body));
    let status = parse_status(&raw);
    let body = parse_body(&raw);
    let data: serde_json::Value = serde_json::from_str(&body).unwrap();

    assert_eq!(status, 200, "expected 200");
    assert_eq!(data["type"], "message", "type should be message");
    assert!(
        !data["content"].as_array().unwrap().is_empty(),
        "content should not be empty"
    );
}

// -----------------------------------------------------------------------------
// Error Formatter Integration Tests
// -----------------------------------------------------------------------------

#[test]
fn proxy_failure_formats_anthropic_error_for_messages() {
    let dead_port = free_port();
    let proxy_port = free_port();

    let yaml = error_formatter_yaml(proxy_port, dead_port);
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body =
        r#"{"model":"claude-3-5-sonnet-20241022","max_tokens":1024,"messages":[{"role":"user","content":"Hi"}]}"#;
    let raw = http_send(proxy.addr(), &anthropic_post("/v1/messages", body));

    assert_eq!(
        parse_status(&raw),
        502,
        "proxy failure on unreachable upstream should return 502"
    );
    assert_eq!(
        parse_header(&raw, "content-type").as_deref(),
        Some("application/json"),
        "Content-Type should be application/json"
    );

    let parsed: serde_json::Value =
        serde_json::from_str(&parse_body(&raw)).expect("response body should be valid JSON");
    assert_eq!(parsed["type"], "error", "Anthropic top-level type should be error");
    assert_eq!(
        parsed["error"]["type"], "api_error",
        "Anthropic error type should be api_error for 502"
    );
    assert!(
        parsed["error"]["message"].is_string(),
        "error message should be a string"
    );
    assert!(
        parsed["request_id"].as_str().unwrap().starts_with("req_"),
        "request_id should be present with req_ prefix"
    );
}

#[test]
fn proxy_failure_formats_anthropic_error_with_custom_request_id() {
    let dead_port = free_port();
    let proxy_port = free_port();

    let yaml = error_formatter_yaml(proxy_port, dead_port);
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body =
        r#"{"model":"claude-3-5-sonnet-20241022","max_tokens":1024,"messages":[{"role":"user","content":"Hi"}]}"#;
    let raw = http_send(
        proxy.addr(),
        &anthropic_post_with_request_id("/v1/messages", body, "req_custom_123"),
    );

    assert_eq!(
        parse_status(&raw),
        502,
        "proxy failure on unreachable upstream should return 502"
    );
    assert_eq!(
        parse_header(&raw, "content-type").as_deref(),
        Some("application/json"),
        "Content-Type should be application/json"
    );

    let parsed: serde_json::Value =
        serde_json::from_str(&parse_body(&raw)).expect("response body should be valid JSON");
    assert_eq!(parsed["type"], "error", "Anthropic top-level type should be error");
    assert_eq!(
        parsed["request_id"], "req_custom_123",
        "request_id should match client x-request-id header"
    );
}

#[test]
fn proxy_failure_does_not_format_anthropic_error_for_unclassified_request() {
    let dead_port = free_port();
    let proxy_port = free_port();

    let yaml = error_formatter_yaml(proxy_port, dead_port);
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"unrelated_api":"data"}"#;
    let raw = http_send(proxy.addr(), &json_post("/other/endpoint", body));

    assert_eq!(
        parse_status(&raw),
        502,
        "proxy failure on unreachable upstream should return 502"
    );

    let body_str = parse_body(&raw);
    let parsed: Result<serde_json::Value, _> = serde_json::from_str(&body_str);
    if let Ok(json) = parsed {
        assert!(
            json.get("type").and_then(|t| t.as_str()) != Some("error")
                || json.get("error").and_then(|e| e.get("type")).is_none(),
            "unclassified request should not receive Anthropic formatted error envelope"
        );
    }
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

fn error_formatter_yaml(proxy_port: u16, backend_port: u16) -> String {
    format!(
        r#"
listeners:
  - name: test
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [passthrough]

filter_chains:
  - name: passthrough
    filters:
      - filter: anthropic_messages_format
        on_invalid: continue
      - filter: router
        routes:
          - path_prefix: "/"
            cluster: mock
      - filter: load_balancer
        clusters:
          - name: mock
            endpoints:
              - "127.0.0.1:{backend_port}"

insecure_options:
  allow_private_endpoints: true
"#
    )
}

fn passthrough_yaml(proxy_port: u16, backend_port: u16) -> String {
    format!(
        r#"
listeners:
  - name: test
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [passthrough]

filter_chains:
  - name: passthrough
    filters:
      - filter: anthropic_messages_format
        on_invalid: continue
      - filter: anthropic_validate
      - filter: anthropic_messages_protocol
        default_version: "2023-06-01"
      - filter: router
        routes:
          - path_prefix: "/"
            cluster: mock
      - filter: load_balancer
        clusters:
          - name: mock
            endpoints:
              - "127.0.0.1:{backend_port}"

insecure_options:
  allow_private_endpoints: true
"#
    )
}

fn transform_yaml(proxy_port: u16, backend_port: u16, max_tool_blocks: usize) -> String {
    format!(
        r#"
listeners:
  - name: test
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [transform]

filter_chains:
  - name: transform
    filters:
      - filter: anthropic_messages_format
        on_invalid: continue
      - filter: anthropic_to_openai
        max_body_bytes: 1048576
      - filter: anthropic_stream_events
        max_tool_blocks: {max_tool_blocks}
      - filter: router
        routes:
          - path_prefix: "/"
            cluster: mock
      - filter: load_balancer
        clusters:
          - name: mock
            endpoints:
              - "127.0.0.1:{backend_port}"

insecure_options:
  allow_private_endpoints: true
"#
    )
}

/// OpenAI Chat Completions SSE with three tool-call deltas at distinct
/// indices, a finish chunk, and the `[DONE]` sentinel. Each distinct
/// index opens a new Anthropic tool-use content block in the transform,
/// so the stream pins three blocks of per-block metadata.
fn tool_call_stream_sse() -> String {
    let block = |index: u64| {
        format!(
            "data: {{\"id\":\"c1\",\"model\":\"gpt-4\",\"choices\":[{{\"delta\":{{\"tool_calls\":[{{\"index\":{index},\"id\":\"call_{index}\",\"function\":{{\"name\":\"f{index}\",\"arguments\":\"{{}}\"}}}}]}},\"index\":0}}]}}\n\n"
        )
    };

    format!(
        "{}{}{}data: {{\"id\":\"c1\",\"model\":\"gpt-4\",\"choices\":[{{\"delta\":{{}},\"index\":0,\"finish_reason\":\"tool_calls\"}}]}}\n\ndata: [DONE]\n\n",
        block(0),
        block(1),
        block(2),
    )
}

fn anthropic_post(path: &str, body: &str) -> String {
    format!(
        "POST {path} HTTP/1.1\r\n\
         Host: localhost\r\n\
         Content-Type: application/json\r\n\
         anthropic-version: 2023-06-01\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\r\n\
         {body}",
        body.len()
    )
}

fn anthropic_post_with_request_id(path: &str, body: &str, request_id: &str) -> String {
    format!(
        "POST {path} HTTP/1.1\r\n\
         Host: localhost\r\n\
         Content-Type: application/json\r\n\
         anthropic-version: 2023-06-01\r\n\
         x-request-id: {request_id}\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\r\n\
         {body}",
        body.len()
    )
}

fn parse_sse_events(body: &str) -> Vec<serde_json::Value> {
    let mut events = Vec::new();
    let mut current_event_type = None;

    for line in body.lines() {
        if let Some(event_type) = line.strip_prefix("event: ") {
            current_event_type = Some(event_type.to_owned());
        } else if let Some(data) = line.strip_prefix("data: ")
            && let Ok(mut value) = serde_json::from_str::<serde_json::Value>(data)
        {
            if let Some(et) = &current_event_type {
                value["_event_type"] = serde_json::Value::String(et.clone());
            }
            events.push(value);
            current_event_type = None;
        }
    }

    events
}
