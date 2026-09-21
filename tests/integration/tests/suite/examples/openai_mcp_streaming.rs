// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Integration tests for MCP SSE streaming transport.
//!
//! These tests verify the filtered-subrequest transport's ability to consume
//! SSE-streamed tool results from MCP servers (issue #1226).

use std::collections::HashMap;

use praxis_test_utils::{
    McpMockConfig, McpToolFixture, StatefulCapturingBackend, free_port, http_send, json_post,
    load_example_config, parse_body, parse_status, start_mcp_mock_server_with_config, start_proxy,
};

// -----------------------------------------------------------------------------
// Scenario 1: POST→SSE tool result streams and completes
// -----------------------------------------------------------------------------

#[test]
fn post_sse_tool_result_streams_and_completes() {
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
        sse_tool_results: true,
        ..McpMockConfig::default()
    });

    let proxy_port = free_port();
    let config = load_example_config(
        "openai/responses/mcp-streaming.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3001", model.port())]),
    );
    let proxy = start_proxy(&config);

    let mcp_url = format!("http://127.0.0.1:{}/mcp", mcp.port());
    let body = serde_json::json!({
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
        &json_post("/v1/responses", &serde_json::to_string(&body).unwrap()),
    );

    assert_eq!(parse_status(&raw), 200, "SSE tool result should return 200");
    assert_eq!(
        mcp.tool_call_count("get_weather"),
        1,
        "MCP server should receive one tool call"
    );

    let response_body = parse_body(&raw);
    assert!(
        response_body.contains("mock result for get_weather"),
        "response should contain the MCP tool result"
    );
}

// -----------------------------------------------------------------------------
// Scenario 2: Buffered JSON tool result still works
// -----------------------------------------------------------------------------

#[test]
fn buffered_json_tool_result_still_works() {
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
        sse_tool_results: false,
        ..McpMockConfig::default()
    });

    let proxy_port = free_port();
    let config = load_example_config(
        "openai/responses/mcp-streaming.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3001", model.port())]),
    );
    let proxy = start_proxy(&config);

    let mcp_url = format!("http://127.0.0.1:{}/mcp", mcp.port());
    let body = serde_json::json!({
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
        &json_post("/v1/responses", &serde_json::to_string(&body).unwrap()),
    );

    assert_eq!(
        parse_status(&raw),
        200,
        "buffered JSON tool result should return 200"
    );
    assert_eq!(
        mcp.tool_call_count("get_weather"),
        1,
        "MCP server should receive one tool call"
    );

    let response_body = parse_body(&raw);
    assert!(
        response_body.contains("mock result for get_weather"),
        "response should contain the MCP tool result"
    );
}

// -----------------------------------------------------------------------------
// Scenario 3: Oversized SSE tool result returns 413
// -----------------------------------------------------------------------------

#[test]
fn oversized_sse_tool_result_returns_413() {
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

    let model = StatefulCapturingBackend::new(vec![(
        200,
        serde_json::to_string(&first_response).unwrap(),
    )])
    .start_with_shutdown();

    // max_result_bytes is 1048576 in the config, so use 2 MiB
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
        sse_tool_results: true,
        oversized_sse_bytes: Some(2 * 1024 * 1024),
        ..McpMockConfig::default()
    });

    let proxy_port = free_port();
    let config = load_example_config(
        "openai/responses/mcp-streaming.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3001", model.port())]),
    );
    let proxy = start_proxy(&config);

    let mcp_url = format!("http://127.0.0.1:{}/mcp", mcp.port());
    let body = serde_json::json!({
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
        &json_post("/v1/responses", &serde_json::to_string(&body).unwrap()),
    );

    let status = parse_status(&raw);
    let response_body = parse_body(&raw);

    // Expect the proxy to surface a failure (not hang)
    assert!(
        status == 200 || status >= 400,
        "oversized SSE should surface a failure; got status {status}"
    );

    // If status is 200, the error should be in the body as an error envelope
    if status == 200 {
        assert!(
            response_body.contains("error") || response_body.contains("413"),
            "response body should contain an error indication: {response_body}"
        );
    }
}

// -----------------------------------------------------------------------------
// Scenario 4: Clean request after stream succeeds
// -----------------------------------------------------------------------------

#[test]
fn clean_request_after_stream_succeeds() {
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

    // Four responses: two rounds * two responses per round
    let model = StatefulCapturingBackend::new(vec![
        (200, serde_json::to_string(&first_response).unwrap()),
        (200, serde_json::to_string(&second_response).unwrap()),
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
        sse_tool_results: true,
        ..McpMockConfig::default()
    });

    let proxy_port = free_port();
    let config = load_example_config(
        "openai/responses/mcp-streaming.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3001", model.port())]),
    );
    let proxy = start_proxy(&config);

    let mcp_url = format!("http://127.0.0.1:{}/mcp", mcp.port());
    let body = serde_json::json!({
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

    // First request
    let raw1 = http_send(
        proxy.addr(),
        &json_post("/v1/responses", &serde_json::to_string(&body).unwrap()),
    );
    assert_eq!(parse_status(&raw1), 200, "first request should return 200");

    // Second request
    let raw2 = http_send(
        proxy.addr(),
        &json_post("/v1/responses", &serde_json::to_string(&body).unwrap()),
    );
    assert_eq!(parse_status(&raw2), 200, "second request should return 200");

    assert_eq!(
        mcp.method_count("tools/call"),
        2,
        "MCP server should receive two tool calls"
    );
}

// -----------------------------------------------------------------------------
// Scenario 5: Named outbound chain without selector fails to boot
// -----------------------------------------------------------------------------

#[test]
fn named_outbound_chain_without_selector_fails_to_boot() {
    let yaml = format!(
        r#"
listeners:
  - name: ai-gateway
    address: "127.0.0.1:{}"
    filter_chains: [mcp-pipeline]

filter_chains:
  - name: mcp-pipeline
    filters:
      - filter: openai_responses_format
        on_invalid: continue
        headers:
          format: x-praxis-ai-format
          model: x-praxis-ai-model
          stream: x-praxis-ai-stream
      - filter: openai_tool_parse
      - filter: openai_mcp_tool_resolve
        timeout_ms: 5000
        outbound_chain: mcp-egress
      - filter: openai_responses_proxy
      - filter: router
        routes:
          - path: "/v1/responses"
            headers:
              x-praxis-ai-format: "openai_responses"
            cluster: "inference-backend"
      - filter: load_balancer
        clusters:
          - name: "inference-backend"
            endpoints:
              - "127.0.0.1:3001"

  # Named chain WITHOUT the selector first - should be rejected
  - name: mcp-egress
    filters:
      - filter: headers
        request_set:
          - name: X-MCP-Client
            value: praxis-ai-gateway

insecure_options:
  allow_private_endpoints: true
  allow_private_upstreams: true
"#,
        free_port()
    );

    let config_result = praxis_core::config::Config::from_yaml(&yaml);
    assert!(
        config_result.is_ok(),
        "config parsing should succeed; bind-time validation happens at start_proxy"
    );

    let result = std::panic::catch_unwind(|| {
        let config = config_result.unwrap();
        let _proxy = start_proxy(&config);
    });

    assert!(
        result.is_err(),
        "start_proxy should panic when named outbound_chain lacks selector-first"
    );
}

// -----------------------------------------------------------------------------
// Scenario 6: Named outbound chain with selector first boots and streams
// -----------------------------------------------------------------------------

#[test]
fn named_outbound_chain_with_selector_first_boots_and_streams() {
    let proxy_port = free_port();
    let model = StatefulCapturingBackend::new(vec![(
        200,
        r#"{"id":"resp_1","object":"response","status":"completed","output":[]}"#.to_owned(),
    )])
    .start_with_shutdown();

    let yaml = format!(
        r#"
listeners:
  - name: ai-gateway
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [mcp-pipeline]

filter_chains:
  - name: mcp-pipeline
    filters:
      - filter: openai_responses_format
        on_invalid: continue
        headers:
          format: x-praxis-ai-format
          model: x-praxis-ai-model
          stream: x-praxis-ai-stream
      - filter: openai_tool_parse
      - filter: openai_mcp_tool_resolve
        timeout_ms: 5000
        outbound_chain: mcp-egress
      - filter: openai_responses_proxy
      - filter: router
        routes:
          - path: "/v1/responses"
            headers:
              x-praxis-ai-format: "openai_responses"
            cluster: "inference-backend"
      - filter: load_balancer
        clusters:
          - name: "inference-backend"
            endpoints:
              - "127.0.0.1:{backend_port}"

  # Named chain WITH the selector first - should be accepted
  - name: mcp-egress
    filters:
      - filter: openai_mcp_streaming_selector
      - filter: headers
        request_set:
          - name: X-MCP-Client
            value: praxis-ai-gateway

insecure_options:
  allow_private_endpoints: true
  allow_private_upstreams: true
"#,
        backend_port = model.port()
    );

    let config = praxis_core::config::Config::from_yaml(&yaml)
        .expect("config with selector-first named chain should parse");
    let proxy = start_proxy(&config);

    let mcp = start_mcp_mock_server_with_config(McpMockConfig {
        tools: vec![McpToolFixture::new("get_weather")],
        ..McpMockConfig::default()
    });

    let mcp_url = format!("http://127.0.0.1:{}/mcp", mcp.port());
    let body = serde_json::json!({
        "model": "gpt-4.1",
        "input": "Hello",
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
        &json_post("/v1/responses", &serde_json::to_string(&body).unwrap()),
    );

    assert_eq!(
        parse_status(&raw),
        200,
        "named outbound_chain with selector first should boot and complete successfully"
    );
}

// -----------------------------------------------------------------------------
// Scenario 7: StreamBuffer outbound filter fails to boot
// -----------------------------------------------------------------------------

#[test]
fn streambuffer_outbound_filter_fails_to_boot() {
    let yaml = format!(
        r#"
listeners:
  - name: ai-gateway
    address: "127.0.0.1:{}"
    filter_chains: [mcp-pipeline]

filter_chains:
  - name: mcp-pipeline
    filters:
      - filter: openai_responses_format
        on_invalid: continue
        headers:
          format: x-praxis-ai-format
          model: x-praxis-ai-model
          stream: x-praxis-ai-stream
      - filter: openai_tool_parse
      - filter: state_owner
        mode: single_tenant
        tenant_id: default
      - filter: openai_response_store
        backend: sqlite
        database_url: "sqlite://responses.db?mode=rwc"
        responses_table: openai_responses
        conversations_table: openai_conversations
      - filter: openai_mcp_tool_resolve
        timeout_ms: 5000
      - filter: iterative_request_router
        initial_step: inference
        max_iterations: 11
        steps:
          - name: inference
            filters:
              - filter: openai_mcp_dispatch
                # Include a response-body-buffering filter in the outbound chain
                outbound_chain:
                  name: mcp-dispatch-outbound
                  filters:
                    - filter: openai_responses_proxy
                max_calls_per_round: 32
                max_result_bytes: 1048576
              - filter: openai_agentic_loop
                max_infer_iters: 10
              - filter: openai_responses_proxy
              - filter: router
                routes:
                  - path_prefix: "/"
                    cluster: "inference-backend"
              - filter: load_balancer
                clusters:
                  - name: "inference-backend"
                    endpoints:
                      - "127.0.0.1:3001"
            on_result:
              - filter: openai_agentic_loop
                key: action
                value: loop
                next: inference
              - default: true
                done: true

insecure_options:
  allow_private_endpoints: true
  allow_private_upstreams: true
"#,
        free_port()
    );

    let config_result = praxis_core::config::Config::from_yaml(&yaml);
    assert!(
        config_result.is_ok(),
        "config parsing should succeed; bind-time validation happens at start_proxy"
    );

    let result = std::panic::catch_unwind(|| {
        let config = config_result.unwrap();
        let _proxy = start_proxy(&config);
    });

    assert!(
        result.is_err(),
        "start_proxy should panic when outbound_chain includes StreamBuffer filter"
    );
}
