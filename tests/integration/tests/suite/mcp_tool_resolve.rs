// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Integration tests for the `mcp_tool_resolve` filter.

use praxis_core::config::Config;
use praxis_test_utils::{
    free_port, http_send, json_post, parse_body, parse_status, start_backend_with_shutdown, start_echo_backend,
    start_proxy,
};

// =============================================================================
// Pass-Through (no MCP tools)
// =============================================================================

#[test]
fn request_without_mcp_tools_passes_through() {
    let backend_guard = start_backend_with_shutdown("inference");
    let proxy_port = free_port();

    let yaml = resolve_yaml(proxy_port, backend_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"test","tools":[{"type":"function","name":"calc"}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "non-MCP request should pass through");
    assert_eq!(
        parse_body(&raw),
        "inference",
        "non-MCP request should reach inference backend"
    );
}

#[test]
fn request_without_tools_passes_through() {
    let backend_guard = start_backend_with_shutdown("inference");
    let proxy_port = free_port();

    let yaml = resolve_yaml(proxy_port, backend_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"Hello"}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "request without tools should pass through");
    assert_eq!(
        parse_body(&raw),
        "inference",
        "request without tools should reach inference backend"
    );
}

// =============================================================================
// SSRF Rejection
// =============================================================================

#[test]
fn mcp_loopback_url_rejected_as_ssrf() {
    let backend_guard = start_backend_with_shutdown("inference");
    let proxy_port = free_port();

    let yaml = resolve_yaml(proxy_port, backend_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"test","tools":[{"type":"mcp","server_label":"evil","server_url":"http://127.0.0.1/mcp","allowed_tools":["x"]}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 502, "loopback MCP URL should be rejected");
    let response_body = parse_body(&raw);
    assert!(
        response_body.contains("SSRF"),
        "rejection should mention SSRF: {response_body}"
    );
}

#[test]
fn mcp_metadata_url_rejected_as_ssrf() {
    let backend_guard = start_backend_with_shutdown("inference");
    let proxy_port = free_port();

    let yaml = resolve_yaml(proxy_port, backend_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"test","tools":[{"type":"mcp","server_label":"meta","server_url":"http://169.254.169.254/latest/meta-data/","allowed_tools":["x"]}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 502, "metadata URL should be rejected");
    let response_body = parse_body(&raw);
    assert!(
        response_body.contains("SSRF"),
        "rejection should mention SSRF: {response_body}"
    );
}

#[test]
fn mcp_localhost_url_rejected_as_ssrf() {
    let backend_guard = start_backend_with_shutdown("inference");
    let proxy_port = free_port();

    let yaml = resolve_yaml(proxy_port, backend_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"test","tools":[{"type":"mcp","server_label":"local","server_url":"http://localhost/mcp","allowed_tools":["x"]}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 502, "localhost MCP URL should be rejected");
}

// =============================================================================
// Connection Failure
// =============================================================================

#[test]
fn mcp_unreachable_server_returns_502() {
    let backend_guard = start_backend_with_shutdown("inference");
    let proxy_port = free_port();
    let dead_port = free_port();

    let yaml = resolve_yaml_with_timeout(proxy_port, backend_guard.port(), 500);
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = format!(
        r#"{{"model":"gpt-4.1","input":"test","tools":[{{"type":"mcp","server_label":"dead","server_url":"http://192.0.2.1:{dead_port}/mcp","allowed_tools":["x"]}}]}}"#
    );
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &body));

    assert_eq!(parse_status(&raw), 502, "unreachable MCP server should produce 502");
}

// =============================================================================
// MCPToolFilter Object: read_only accepted
// =============================================================================

#[test]
fn mcp_read_only_filter_accepted() {
    let backend_guard = start_backend_with_shutdown("inference");
    let proxy_port = free_port();

    let yaml = resolve_yaml(proxy_port, backend_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"test","tools":[{"type":"mcp","server_label":"srv","server_url":"http://10.0.0.1/mcp","allowed_tools":{"read_only":true}}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    let status = parse_status(&raw);
    assert_ne!(status, 400, "read_only filter should be accepted, not rejected as 400");
}

// =============================================================================
// Non-Responses Path
// =============================================================================

#[test]
fn non_responses_path_passes_through() {
    let backend_guard = start_backend_with_shutdown("inference");
    let proxy_port = free_port();

    let yaml = resolve_yaml(proxy_port, backend_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"test","tools":[{"type":"mcp","server_label":"s","server_url":"http://127.0.0.1/mcp"}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/chat/completions", body));

    assert_eq!(
        parse_status(&raw),
        200,
        "non-Responses path should not trigger MCP resolution"
    );
    assert_eq!(
        parse_body(&raw),
        "inference",
        "non-Responses path should pass through to backend"
    );
}

// =============================================================================
// Body Preservation with responses_proxy (get_or_insert_with regression)
// =============================================================================

#[test]
fn body_preserved_through_responses_proxy_with_function_tools() {
    let backend_guard = start_echo_backend();
    let proxy_port = free_port();

    let yaml = resolve_with_responses_proxy_yaml(proxy_port, backend_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"test","tools":[{"type":"function","name":"calc"}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "should return 200");
    let echoed = parse_body(&raw);
    let parsed: serde_json::Value = serde_json::from_str(&echoed).unwrap();
    assert_eq!(parsed["model"], "gpt-4.1", "model should be preserved");
    assert_eq!(parsed["input"], "test", "input should be preserved");
    assert!(parsed["tools"].is_array(), "tools array should be preserved");
}

#[test]
fn body_preserved_through_responses_proxy_without_tools() {
    let backend_guard = start_echo_backend();
    let proxy_port = free_port();

    let yaml = resolve_with_responses_proxy_yaml(proxy_port, backend_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"Hello, world!"}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "should return 200");
    let echoed = parse_body(&raw);
    let parsed: serde_json::Value = serde_json::from_str(&echoed).unwrap();
    assert_eq!(parsed["model"], "gpt-4.1", "model should be preserved");
    assert_eq!(parsed["input"], "Hello, world!", "input should be preserved");
}

// =============================================================================
// Authorization + SSRF Interaction
// =============================================================================

#[test]
fn authorization_does_not_bypass_ssrf_check() {
    let backend_guard = start_backend_with_shutdown("inference");
    let proxy_port = free_port();

    let yaml = resolve_yaml(proxy_port, backend_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"test","tools":[{"type":"mcp","server_label":"auth","server_url":"http://127.0.0.1/mcp","authorization":"tok_secret","allowed_tools":["x"]}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 502, "SSRF should reject even with authorization");
    let response_body = parse_body(&raw);
    assert!(
        response_body.contains("SSRF"),
        "rejection should mention SSRF, not auth failure: {response_body}"
    );
}

#[test]
fn authorization_with_unreachable_server_returns_502() {
    let backend_guard = start_backend_with_shutdown("inference");
    let proxy_port = free_port();
    let dead_port = free_port();

    let yaml = resolve_yaml_with_timeout(proxy_port, backend_guard.port(), 500);
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = format!(
        r#"{{"model":"gpt-4.1","input":"test","tools":[{{"type":"mcp","server_label":"auth","server_url":"http://192.0.2.1:{dead_port}/mcp","authorization":"tok_secret","headers":{{"x-custom":"val"}},"allowed_tools":["x"]}}]}}"#
    );
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &body));

    assert_eq!(
        parse_status(&raw),
        502,
        "unreachable server with auth+headers should produce 502"
    );
}

// =============================================================================
// MCPToolFilter Object Form (tool_names accepted)
// =============================================================================

#[test]
fn mcp_tool_names_filter_object_accepted() {
    let backend_guard = start_backend_with_shutdown("inference");
    let proxy_port = free_port();
    let dead_port = free_port();

    let yaml = resolve_yaml_with_timeout(proxy_port, backend_guard.port(), 500);
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = format!(
        r#"{{"model":"gpt-4.1","input":"test","tools":[{{"type":"mcp","server_label":"srv","server_url":"http://192.0.2.1:{dead_port}/mcp","allowed_tools":{{"tool_names":["get_weather"]}}}}]}}"#
    );
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &body));

    assert_eq!(
        parse_status(&raw),
        502,
        "tool_names filter object should be accepted (502 = attempted resolution, not 400)"
    );
}

// =============================================================================
// YAML Helpers
// =============================================================================

fn resolve_yaml(proxy_port: u16, backend_port: u16) -> String {
    resolve_yaml_with_timeout(proxy_port, backend_port, 5000)
}

fn resolve_yaml_with_timeout(proxy_port: u16, backend_port: u16, timeout_ms: u64) -> String {
    format!(
        r#"
listeners:
  - name: default
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: openai_responses_format
        on_invalid: continue
      - filter: tool_parse
      - filter: mcp_tool_resolve
        timeout_ms: {timeout_ms}
      - filter: router
        routes:
          - path_prefix: "/"
            cluster: "backend"
      - filter: load_balancer
        clusters:
          - name: "backend"
            endpoints:
              - "127.0.0.1:{backend_port}"
"#
    )
}

fn resolve_with_responses_proxy_yaml(proxy_port: u16, backend_port: u16) -> String {
    format!(
        r#"
listeners:
  - name: default
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: openai_responses_format
        on_invalid: continue
      - filter: tool_parse
      - filter: mcp_tool_resolve
      - filter: responses_proxy
        name: inference
      - filter: router
        routes:
          - path_prefix: "/"
            cluster: "backend"
      - filter: load_balancer
        clusters:
          - name: "backend"
            endpoints:
              - "127.0.0.1:{backend_port}"
"#
    )
}
