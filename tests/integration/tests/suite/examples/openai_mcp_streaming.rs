// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Integration tests for MCP SSE streaming transport.
//!
//! These tests verify the filtered-subrequest transport's ability to consume
//! SSE-streamed tool results from MCP servers (issue #1226).

use std::collections::HashMap;

use praxis_test_utils::{
    McpMockConfig, McpToolFixture, StatefulCapturingBackend, TempSqlite, allow_loopback_endpoints, example_config_path,
    free_port, http_send, json_post, parse_body, parse_status, patch_yaml, registry_with,
    start_mcp_mock_server_with_config, start_proxy, start_proxy_with_registry,
};

/// Bound asynchronous rmcp cleanup/stream observations without making the
/// integration test fail on a busy shared CI runner.
const RECORDED_REQUEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Load the `mcp-streaming.yaml` example, patching the listener/backend ports and
/// pointing the response store at a private temp SQLite database so tests never
/// share persisted state (a stale shared `responses.db` otherwise surfaces as a
/// spurious HTTP 500 after a schema change).
fn load_mcp_streaming_config(proxy_port: u16, model_port: u16, db_url: &str) -> praxis_core::config::Config {
    let path = example_config_path("openai/responses/mcp-streaming.yaml");
    let yaml = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let yaml = yaml.replace("sqlite://responses.db?mode=rwc", db_url);
    let patched = allow_loopback_endpoints(&patch_yaml(
        &yaml,
        proxy_port,
        &HashMap::from([("127.0.0.1:3001", model_port)]),
    ));
    praxis_core::config::Config::from_yaml(&patched).expect("parse mcp-streaming config")
}

/// Poll the mock's recorded requests until `pred` matches or the deadline
/// passes. Returns whether it matched. Used for the DELETE cleanup, which the
/// rmcp worker issues asynchronously as the transport drops.
fn wait_for_recorded<F>(mcp: &praxis_test_utils::McpMockServerGuard, pred: F) -> bool
where
    F: Fn(&[praxis_test_utils::McpRecordedRequest]) -> bool,
{
    let deadline = std::time::Instant::now() + RECORDED_REQUEST_TIMEOUT;
    loop {
        if pred(&mcp.received_requests()) {
            return true;
        }
        if std::time::Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}

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
    let db = TempSqlite::new("mcp_streaming_post_sse");
    let config = load_mcp_streaming_config(proxy_port, model.port(), db.url());
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
    let db = TempSqlite::new("mcp_streaming_buffered_json");
    let config = load_mcp_streaming_config(proxy_port, model.port(), db.url());
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

    assert_eq!(parse_status(&raw), 200, "buffered JSON tool result should return 200");
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

    let model = StatefulCapturingBackend::new(vec![(200, serde_json::to_string(&first_response).unwrap())])
        .start_with_shutdown();

    // wire cap = tool_result_wire_cap(1 MiB) = 6_356_992 B; executor backstop = 2x =
    // 12_713_984 B. Use 8 MiB: above the cap (adapter 413) but below the backstop.
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
        oversized_sse_bytes: Some(8 * 1024 * 1024),
        ..McpMockConfig::default()
    });

    let proxy_port = free_port();
    let db = TempSqlite::new("mcp_streaming_oversized_413");
    let config = load_mcp_streaming_config(proxy_port, model.port(), db.url());
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

    assert!(
        status >= 400 || (status == 200 && response_body.contains("413")),
        "oversized SSE must surface a 413 (either an HTTP >=400 status or a 413 in \
         the streamed error envelope); got status {status}, body: {response_body}"
    );
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
    let db = TempSqlite::new("mcp_streaming_clean_after_stream");
    let config = load_mcp_streaming_config(proxy_port, model.port(), db.url());
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
// Scenario 4b: GET common stream + DELETE cleanup route through the outbound chain
// -----------------------------------------------------------------------------
// NOTE: There is deliberately no Last-Event-ID / resumption test here. rmcp only
// sends Last-Event-ID when an already-established GET common stream drops
// mid-flight and reconnects; our short-lived serve -> tool-call -> drop flows
// never interrupt the stream, so the transport never emits Last-Event-ID.
// Asserting resumption in this flow would be unreachable, fabricated coverage.

#[test]
fn get_common_stream_and_delete_cleanup_route_through_outbound_chain() {
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
        serve_get_stream: true,
        ..McpMockConfig::default()
    });

    let proxy_port = free_port();
    let db = TempSqlite::new("mcp_streaming_get_common_stream");
    let config = load_mcp_streaming_config(proxy_port, model.port(), db.url());
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

    assert_eq!(parse_status(&raw), 200, "GET common stream test should return 200");
    assert_eq!(
        mcp.tool_call_count("get_weather"),
        1,
        "MCP server should receive one tool call"
    );

    // GET common stream: opened, session-scoped, AND routed through the filtered
    // outbound chain. The mcp_dispatch session's outbound chain stamps
    // x-mcp-client via its `headers` filter; a correctly routed GET carries it.
    // The rmcp worker opens this eager GET stream asynchronously and the mock
    // records connections on separate threads, so poll rather than snapshotting
    // once (same reason the DELETE below polls).
    let saw_chain_get = wait_for_recorded(&mcp, |reqs| {
        reqs.iter().any(|r| {
            r.http_method == "GET"
                && r.headers
                    .iter()
                    .any(|(k, v)| k == "x-mcp-client" && v == "praxis-ai-gateway")
        })
    });
    assert!(
        saw_chain_get,
        "the mcp_dispatch session's eager GET common stream must route through the \
         filtered outbound chain and carry x-mcp-client (within 30s)"
    );
    let reqs = mcp.received_requests();
    let chain_get = reqs
        .iter()
        .find(|r| {
            r.http_method == "GET"
                && r.headers
                    .iter()
                    .any(|(k, v)| k == "x-mcp-client" && v == "praxis-ai-gateway")
        })
        .expect("GET with x-mcp-client should be present after wait_for_recorded returned true");
    assert!(
        chain_get
            .headers
            .iter()
            .any(|(k, v)| k == "mcp-session-id" && v == "mock-mcp-session-1"),
        "GET common stream must carry the negotiated MCP session id; headers: {:?}",
        chain_get.headers
    );

    // NOTE: The top-level openai_mcp_tool_resolve discovery session has NO
    // outbound_chain, so its own GET/DELETE correctly carry no x-mcp-client. We
    // assert on the mcp_dispatch session's GET/DELETE, which DO route through the
    // chain — that is the outbound-filter-on-non-POST coverage this test exists for.

    // DELETE cleanup fires asynchronously on transport drop; poll for the
    // dispatch session's DELETE (the one that carries the egress header).
    let saw_chain_delete = wait_for_recorded(&mcp, |reqs| {
        reqs.iter().any(|r| {
            r.http_method == "DELETE"
                && r.headers
                    .iter()
                    .any(|(k, v)| k == "x-mcp-client" && v == "praxis-ai-gateway")
        })
    });
    assert!(
        saw_chain_delete,
        "the mcp_dispatch session's DELETE cleanup should route through the filtered \
         outbound chain and be recorded within 30s of transport drop"
    );

    let reqs = mcp.received_requests();
    let chain_delete = reqs
        .iter()
        .find(|r| {
            r.http_method == "DELETE"
                && r.headers
                    .iter()
                    .any(|(k, v)| k == "x-mcp-client" && v == "praxis-ai-gateway")
        })
        .expect("DELETE with x-mcp-client should be present after wait_for_recorded returned true");
    assert!(
        chain_delete
            .headers
            .iter()
            .any(|(k, v)| k == "mcp-session-id" && v == "mock-mcp-session-1"),
        "DELETE cleanup must carry the negotiated MCP session id; headers: {:?}",
        chain_delete.headers
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

    let config =
        praxis_core::config::Config::from_yaml(&yaml).expect("config with selector-first named chain should parse");
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
// bind_mcp_outbound_chain rejects, at bind time, any outbound chain whose
// aggregate response-body mode is StreamBuffer, because buffering the response
// defeats SSE streaming. The stock filters usable in an MCP outbound chain
// never report a response-body StreamBuffer, so we register a test-only filter
// (ResponseStreamBufferFilter, name "test_response_stream_buffer") that does,
// placed after the selector (so the selector-first check passes and the
// StreamBuffer arm is the reason boot fails). Scenario 6 proves the same chain
// shape boots when the trailing filter is a stock `headers` filter, so a boot
// failure here isolates the StreamBuffer rejection.

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

  # Selector first (passes the selector-first check) followed by a response-body
  # buffering filter - must be rejected at bind time.
  - name: mcp-egress
    filters:
      - filter: openai_mcp_streaming_selector
      - filter: test_response_stream_buffer

insecure_options:
  allow_private_endpoints: true
  allow_private_upstreams: true
"#,
        free_port()
    );

    let config = praxis_core::config::Config::from_yaml(&yaml)
        .expect("config parsing should succeed; bind-time validation happens at start_proxy");
    let registry = registry_with("test_response_stream_buffer", || {
        Box::new(praxis_test_utils::filters::ResponseStreamBufferFilter)
    });

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _proxy = start_proxy_with_registry(&config, &registry);
    }));

    assert!(
        result.is_err(),
        "start_proxy should panic when the MCP outbound_chain contains a response-body \
         StreamBuffer filter (incompatible with SSE streaming)"
    );
}
