// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for the mcp-tool-resolve example config.

use std::collections::HashMap;

use praxis_test_utils::{
    free_port, http_send, json_post, load_example_config, parse_body, parse_status, start_backend_with_shutdown,
    start_proxy,
};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn openai_mcp_tool_resolve_example_passes_through_non_mcp_request() {
    let backend_guard = start_backend_with_shutdown("inference");
    let proxy_port = free_port();

    let config = load_example_config(
        "openai/responses/mcp-tool-resolve.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3001", backend_guard.port())]),
    );
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"Hello, world!"}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "non-MCP request should pass through");
    assert_eq!(
        parse_body(&raw),
        "inference",
        "non-MCP request should reach inference backend"
    );
}

#[test]
fn openai_mcp_tool_resolve_example_rejects_ssrf_url() {
    let backend_guard = start_backend_with_shutdown("inference");
    let proxy_port = free_port();

    let config = load_example_config(
        "openai/responses/mcp-tool-resolve.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3001", backend_guard.port())]),
    );
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"test","tools":[{"type":"mcp","server_label":"evil","server_url":"http://127.0.0.1/mcp","allowed_tools":["x"]}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 502, "SSRF URL should be rejected");
    let response_body = parse_body(&raw);
    assert!(
        response_body.contains("SSRF"),
        "rejection should mention SSRF: {response_body}"
    );
}

#[test]
fn openai_mcp_tool_resolve_example_passes_through_function_tools() {
    let backend_guard = start_backend_with_shutdown("inference");
    let proxy_port = free_port();

    let config = load_example_config(
        "openai/responses/mcp-tool-resolve.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3001", backend_guard.port())]),
    );
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"test","tools":[{"type":"function","name":"calc"}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "function tools should pass through");
    assert_eq!(
        parse_body(&raw),
        "inference",
        "function tools should reach inference backend"
    );
}
