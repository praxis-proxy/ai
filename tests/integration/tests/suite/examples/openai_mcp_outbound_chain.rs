// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for the mcp-outbound-chain example config.
//!
//! Proves that an operator `outbound_chain` bound onto `openai_mcp_tool_resolve`
//! runs on the outbound MCP callout: the MCP transport stages the dial target
//! and the operator's `headers` filter then stamps a probe header on every
//! request the proxy dials to the MCP server.
//!
//! Also covers the two properties the example config only advertises in prose:
//! that `openai_mcp_tool_resolve` (a top-level filter) resolves a **named**
//! `outbound_chain` against the top-level `filter_chains`, and that a
//! client-controlled `server_url` carrying embedded credentials is refused
//! before any dial — the egress credential the operator injects through its
//! own chain stays isolated from client input.

use std::collections::HashMap;

use praxis_core::config::Config;
use praxis_test_utils::{
    McpMockConfig, McpToolFixture, free_port, http_send, json_post, load_example_config, parse_body, parse_status,
    start_echo_backend, start_mcp_mock_server_with_config, start_proxy,
};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

/// The `headers` filter in the bound `outbound_chain` must stamp
/// `x-mcp-outbound-probe` onto every outbound MCP request (`initialize`,
/// `notifications/initialized`, `tools/list`, ...), demonstrating that operator
/// filters observe and mutate the callout while the transport stages the dial
/// target.
#[test]
fn openai_mcp_outbound_chain_example_stamps_probe_header_on_callout() {
    let mcp_config = McpMockConfig {
        tools: vec![McpToolFixture::new("get_weather").with_description("Get weather")],
        ..McpMockConfig::default()
    };
    let mcp_server = start_mcp_mock_server_with_config(mcp_config);
    let backend_guard = start_echo_backend();
    let proxy_port = free_port();

    let config = load_example_config(
        "openai/responses/mcp-outbound-chain.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3001", backend_guard.port())]),
    );
    let proxy = start_proxy(&config);

    let mcp_url = format!("http://127.0.0.1:{}/mcp", mcp_server.port());
    let body = format!(
        r#"{{"model":"gpt-4.1","input":"test","tools":[{{"type":"mcp","server_label":"weather","server_url":"{mcp_url}","allowed_tools":["get_weather"]}}]}}"#
    );
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &body));

    assert_eq!(parse_status(&raw), 200, "MCP resolution should succeed");
    assert!(
        mcp_server.method_count("tools/list") >= 1,
        "the outbound callout must reach the MCP server"
    );

    let requests = mcp_server.received_requests();
    assert!(
        !requests.is_empty(),
        "the MCP server should have recorded the outbound callout"
    );
    // Every request the proxy dialed to the MCP server passed through the bound
    // outbound chain, so each must carry the header its `headers` filter stamped.
    for req in &requests {
        let has_probe = req
            .headers
            .iter()
            .any(|(name, value)| name.eq_ignore_ascii_case("x-mcp-outbound-probe") && value == "praxis-outbound");
        assert!(
            has_probe,
            "outbound {} request must carry the probe header stamped by the outbound_chain: {:?}",
            req.json_rpc_method.as_deref().unwrap_or("(non-JSON-RPC)"),
            req.headers
        );
    }
}

/// `openai_mcp_tool_resolve` is a top-level filter, so praxis core builds it
/// with a chain-binding context and its `outbound_chain` may be a **named**
/// reference to a top-level `filter_chains` entry (not just an inline chain).
/// Binding a named `mcp-egress` chain that is never attached to a listener must
/// still resolve and run on the callout, stamping its header on every outbound
/// MCP request — proving named resolution works where it is advertised.
#[test]
fn openai_mcp_named_outbound_chain_stamps_probe_header_on_callout() {
    let mcp_config = McpMockConfig {
        tools: vec![McpToolFixture::new("get_weather").with_description("Get weather")],
        ..McpMockConfig::default()
    };
    let mcp_server = start_mcp_mock_server_with_config(mcp_config);
    let backend_guard = start_echo_backend();
    let proxy_port = free_port();

    let yaml = named_outbound_chain_yaml(proxy_port, backend_guard.port());
    let config = Config::from_yaml(&yaml).expect("named outbound_chain config should build");
    let proxy = start_proxy(&config);

    let mcp_url = format!("http://127.0.0.1:{}/mcp", mcp_server.port());
    let body = format!(
        r#"{{"model":"gpt-4.1","input":"test","tools":[{{"type":"mcp","server_label":"weather","server_url":"{mcp_url}","allowed_tools":["get_weather"]}}]}}"#
    );
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &body));

    assert_eq!(parse_status(&raw), 200, "MCP resolution should succeed");
    assert!(
        mcp_server.method_count("tools/list") >= 1,
        "the outbound callout must reach the MCP server"
    );

    let requests = mcp_server.received_requests();
    assert!(
        !requests.is_empty(),
        "the MCP server should have recorded the outbound callout"
    );
    // The named chain resolved and ran, so every dialed request carries its
    // stamped header just as the inline case does.
    for req in &requests {
        let has_probe = req
            .headers
            .iter()
            .any(|(name, value)| name.eq_ignore_ascii_case("x-mcp-outbound-probe") && value == "praxis-outbound");
        assert!(
            has_probe,
            "outbound {} request must carry the probe header stamped by the named outbound_chain: {:?}",
            req.json_rpc_method.as_deref().unwrap_or("(non-JSON-RPC)"),
            req.headers
        );
    }
}

/// A client-controlled `server_url` that embeds credentials in its authority
/// (`user:secret@host`) must be refused before the proxy dials anything, even
/// when loopback destinations are allowed. This keeps any egress credential the
/// operator injects through its `outbound_chain` isolated from client-supplied
/// URLs: the client can neither smuggle its own credentials onto the wire nor
/// coax the proxy into contacting a credentialed authority.
#[test]
fn openai_mcp_client_url_with_embedded_credentials_is_never_dialed() {
    let mcp_config = McpMockConfig {
        tools: vec![McpToolFixture::new("get_weather").with_description("Get weather")],
        ..McpMockConfig::default()
    };
    let mcp_server = start_mcp_mock_server_with_config(mcp_config);
    let backend_guard = start_echo_backend();
    let proxy_port = free_port();

    let config = load_example_config(
        "openai/responses/mcp-outbound-chain.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3001", backend_guard.port())]),
    );
    let proxy = start_proxy(&config);

    // The authority carries `user:secret@` credentials the client should never
    // be able to place on the wire, pointing at the loopback mock the harness
    // otherwise permits — isolating the rejection to the embedded credentials.
    let mcp_url = format!("http://user:secret@127.0.0.1:{}/mcp", mcp_server.port());
    let body = format!(
        r#"{{"model":"gpt-4.1","input":"test","tools":[{{"type":"mcp","server_label":"weather","server_url":"{mcp_url}","allowed_tools":["get_weather"]}}]}}"#
    );
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &body));

    assert_ne!(
        parse_status(&raw),
        200,
        "a server_url embedding credentials must not resolve successfully"
    );
    assert_eq!(
        mcp_server.method_count("tools/list"),
        0,
        "the credentialed URL must be refused before any callout is dialed"
    );
    assert!(
        mcp_server.received_requests().is_empty(),
        "no request should ever reach the MCP server for a credentialed URL"
    );
    // The rejection body must not echo the client's secret back to the caller.
    let response_body = parse_body(&raw);
    assert!(
        !response_body.contains("secret"),
        "the rejection must not leak the embedded credential: {response_body}"
    );
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

/// Build a Responses pipeline whose `openai_mcp_tool_resolve` binds a **named**
/// `outbound_chain` (`mcp-egress`) defined as a standalone top-level
/// `filter_chains` entry — the shape the example config documents but does not
/// itself exercise.
fn named_outbound_chain_yaml(proxy_port: u16, backend_port: u16) -> String {
    format!(
        r#"
listeners:
  - name: ai-gateway
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [responses-pipeline]

filter_chains:
  - name: responses-pipeline
    filters:
      - filter: openai_responses_format
        on_invalid: continue
        headers:
          format: x-praxis-ai-format
          model: x-praxis-ai-model
          stream: x-praxis-ai-stream
          mode: x-praxis-responses-mode
      - filter: openai_tool_parse
      - filter: openai_mcp_tool_resolve
        timeout_ms: 5000
        outbound_chain: mcp-egress
      - filter: openai_responses_proxy
        name: inference
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

  # Standalone named chain, never attached to a listener: resolved only as the
  # MCP callout's outbound chain via the named reference above.
  - name: mcp-egress
    filters:
      - filter: headers
        request_set:
          - name: x-mcp-outbound-probe
            value: praxis-outbound

insecure_options:
  allow_private_endpoints: true
  allow_private_upstreams: true
"#
    )
}
