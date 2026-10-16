// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for the mcp-tool-admission example config (#1724).

use std::collections::HashMap;

use praxis_test_utils::{
    free_port, http_send, json_post, load_example_config, parse_body, parse_status, start_echo_backend, start_proxy,
};

/// The example config is valid and non-MCP traffic passes straight through.
#[test]
fn mcp_tool_admission_example_passes_through_non_mcp_request() {
    let backend_guard = start_echo_backend();
    let proxy_port = free_port();

    let config = load_example_config(
        "openai/responses/mcp-tool-admission.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3001", backend_guard.port())]),
    );
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"Hello, world!"}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "non-MCP request should pass through");
}

/// A client-selected direct URL is untrusted under protected mode: its tools are
/// excluded without the proxy ever dialing the untrusted endpoint, so exclusion
/// holds even against an unreachable target.
#[test]
fn mcp_tool_admission_example_excludes_untrusted_direct_target() {
    let backend_guard = start_echo_backend();
    let proxy_port = free_port();

    let config = load_example_config(
        "openai/responses/mcp-tool-admission.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3001", backend_guard.port())]),
    );
    let proxy = start_proxy(&config);

    // The target is never contacted: protected mode excludes an untrusted direct
    // URL before dialing, so an unreachable host still yields a clean exclusion.
    let body = r#"{"model":"gpt-4.1","input":"test","tools":[{"type":"mcp","server_label":"x","server_url":"https://unreachable.invalid/mcp"}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));
    assert_eq!(
        parse_status(&raw),
        200,
        "request proceeds with the tool excluded: {}",
        parse_body(&raw)
    );

    let echoed = parse_body(&raw);
    let parsed: serde_json::Value = serde_json::from_str(&echoed).unwrap();
    let tool_count = parsed["tools"].as_array().map_or(0, Vec::len);
    assert_eq!(
        tool_count, 0,
        "no untrusted tools reach the model in protected mode: {echoed}"
    );
    assert!(
        !echoed.contains("server_url"),
        "the untrusted MCP entry must be stripped from the forwarded body: {echoed}"
    );
}
