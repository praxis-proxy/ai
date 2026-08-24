// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Simulator-backed environment integration tests.
//!
//! Proves the AI gateway can route OpenAI-compatible traffic
//! through generic `ext_proc` + `endpoint_selector` to an
//! `llm-d-inference-sim` container backend.

use praxis_test_utils::{free_port, json_post, parse_body, start_mock_routing_processor, start_simulator};

// -----------------------------------------------------------------------------
// Proxy Helper
// -----------------------------------------------------------------------------

fn start_sim_proxy(proxy_port: u16, processor_port: u16) -> praxis_test_utils::ProxyGuard {
    let config_yaml = format!(
        r#"
listeners:
  - name: sim-env
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [sim-chain]

filter_chains:
  - name: sim-chain
    filters:
      - filter: ext_proc
        target: "http://127.0.0.1:{processor_port}"
        message_timeout_ms: 5000
        lifecycle_timeout_ms: 10000
        status_on_error: 503
        processing_mode:
          request_body_mode: full_duplex_streamed
          response_header_mode: skip
      - filter: endpoint_selector
        source_header: x-gateway-destination-endpoint
        required: true
        status_on_required_failure: 503
        strip_header: true
"#
    );

    let config = praxis_core::config::Config::from_yaml(&config_yaml).expect("sim env config should parse");
    let subrequest_client =
        praxis_core::subrequest::SubRequestClient::new(praxis_core::subrequest::SubRequestConnector::new(8, None));
    let registry = praxis_ai::build_full_registry(&subrequest_client);
    praxis_test_utils::start_proxy_with_registry(&config, &registry)
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn simulator_chat_completion_routes_through_praxis() {
    let sim = start_simulator();
    let proc_guard = start_mock_routing_processor(&sim.endpoint());
    let proxy_port = free_port();
    let _proxy = start_sim_proxy(proxy_port, proc_guard.port());

    let body = format!(
        r#"{{"model":"{}","messages":[{{"role":"user","content":"hello"}}],"max_tokens":5}}"#,
        sim.model()
    );
    let proxy_addr = format!("127.0.0.1:{proxy_port}");
    let raw = praxis_test_utils::http_send(&proxy_addr, &json_post("/v1/chat/completions", &body));
    let status = praxis_test_utils::parse_status(&raw);
    let response_body = parse_body(&raw);
    assert_eq!(status, 200, "chat completion should succeed through Praxis");
    assert!(
        !response_body.is_empty(),
        "simulator should return a non-empty response body"
    );

    let json: serde_json::Value = serde_json::from_str(&response_body)
        .unwrap_or_else(|e| panic!("response should be valid JSON: {e}\nbody: {response_body}"));
    assert_eq!(
        json.get("model").and_then(|v| v.as_str()),
        Some(sim.model()),
        "response model should match simulator model"
    );
}

#[test]
fn simulator_spoofed_destination_header_ignored() {
    let sim = start_simulator();
    let proc_guard = start_mock_routing_processor(&sim.endpoint());
    let proxy_port = free_port();
    let _proxy = start_sim_proxy(proxy_port, proc_guard.port());

    let body = format!(
        r#"{{"model":"{}","messages":[{{"role":"user","content":"spoof"}}],"max_tokens":5}}"#,
        sim.model()
    );
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n\
         Content-Type: application/json\r\n\
         x-gateway-destination-endpoint: 10.99.99.99:9999\r\n\
         Content-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );
    let proxy_addr = format!("127.0.0.1:{proxy_port}");
    let raw = praxis_test_utils::http_send(&proxy_addr, &request);
    let status = praxis_test_utils::parse_status(&raw);
    assert_eq!(
        status, 200,
        "spoofed destination header should be ignored; request should route to simulator"
    );
}

#[test]
fn simulator_repeated_requests_no_crosstalk() {
    let sim = start_simulator();
    let proc_guard = start_mock_routing_processor(&sim.endpoint());
    let proxy_port = free_port();
    let _proxy = start_sim_proxy(proxy_port, proc_guard.port());

    let proxy_addr = format!("127.0.0.1:{proxy_port}");
    let baseline = proc_guard.stream_count();
    for i in 0..3 {
        let body = format!(
            r#"{{"model":"{}","messages":[{{"role":"user","content":"repeat {i}"}}],"max_tokens":5}}"#,
            sim.model()
        );
        let raw = praxis_test_utils::http_send(&proxy_addr, &json_post("/v1/chat/completions", &body));
        let status = praxis_test_utils::parse_status(&raw);
        assert_eq!(status, 200, "request {i} should succeed");
    }
    let streams = proc_guard.stream_count() - baseline;
    assert_eq!(
        streams, 3,
        "each request should use one Process stream (used {streams})"
    );
}

#[test]
fn simulator_health_endpoint_reachable() {
    let sim = start_simulator();
    let raw = praxis_test_utils::http_send(&sim.endpoint(), "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n");
    let status = praxis_test_utils::parse_status(&raw);
    assert_eq!(status, 200, "simulator health endpoint should return 200");
}

#[test]
fn simulator_metrics_endpoint_reachable() {
    let sim = start_simulator();
    let raw = praxis_test_utils::http_send(&sim.endpoint(), "GET /metrics HTTP/1.1\r\nHost: localhost\r\n\r\n");
    let status = praxis_test_utils::parse_status(&raw);
    let body = parse_body(&raw);

    assert_eq!(status, 200, "simulator metrics endpoint should return 200");
    assert!(!body.trim().is_empty(), "simulator metrics body should be non-empty");
}

#[test]
fn simulator_processor_failure_returns_status_on_error() {
    let sim = start_simulator();
    let unused_port = free_port();
    let proxy_port = free_port();
    let _proxy = start_sim_proxy(proxy_port, unused_port);

    let body = format!(
        r#"{{"model":"{}","messages":[{{"role":"user","content":"fail"}}],"max_tokens":5}}"#,
        sim.model()
    );
    let proxy_addr = format!("127.0.0.1:{proxy_port}");
    let raw = praxis_test_utils::http_send(&proxy_addr, &json_post("/v1/chat/completions", &body));
    let status = praxis_test_utils::parse_status(&raw);
    assert_eq!(status, 503, "processor failure should return status_on_error 503");
}
