// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Grid route filter example configuration tests.
//!
//! These tests verify that the `grid_route` filter example configs parse
//! and route correctly end-to-end.  The filter is registered by
//! `praxis-ai-proxy` and is AI/Grid-specific — it is not a Praxis core builtin.

use praxis_core::config::Config;
use praxis_test_utils::{free_port, http_post, start_backend, start_proxy};

// -----------------------------------------------------------------------------
// Inference routing tests
// -----------------------------------------------------------------------------

#[test]
fn grid_route_inference_routes_known_local_model() {
    let local_port = start_backend("granite-response");
    let remote_port = start_backend("llama-response");
    let proxy_port = free_port();

    let yaml = make_inference_yaml(proxy_port, local_port, remote_port);
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let (status, body) = http_post(
        proxy.addr(),
        "/v1/chat/completions",
        r#"{"model":"granite-3.3-8b","messages":[]}"#,
    );
    assert_eq!(status, 200, "known local model should route");
    assert_eq!(body, "granite-response", "should select local candidate");
}

#[test]
fn grid_route_inference_routes_known_remote_model() {
    let local_port = start_backend("granite-response");
    let remote_port = start_backend("llama-response");
    let proxy_port = free_port();

    let yaml = make_inference_yaml(proxy_port, local_port, remote_port);
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let (status, body) = http_post(
        proxy.addr(),
        "/v1/chat/completions",
        r#"{"model":"llama-3.2-8b","messages":[]}"#,
    );
    assert_eq!(status, 200, "known remote model should route");
    assert_eq!(body, "llama-response", "should select remote candidate");
}

#[test]
fn grid_route_inference_rejects_unknown_model_with_404() {
    let local_port = start_backend("granite-response");
    let remote_port = start_backend("llama-response");
    let proxy_port = free_port();

    let yaml = make_inference_yaml(proxy_port, local_port, remote_port);
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let (status, _body) = http_post(
        proxy.addr(),
        "/v1/chat/completions",
        r#"{"model":"unknown-model","messages":[]}"#,
    );
    assert_eq!(status, 404, "unknown model should be rejected with 404");
}

// -----------------------------------------------------------------------------
// MCP tool routing tests
// -----------------------------------------------------------------------------

#[test]
fn grid_route_mcp_routes_known_tool() {
    let local_port = start_backend("code-search-response");
    let remote_port = start_backend("weather-response");
    let proxy_port = free_port();

    let yaml = make_mcp_yaml(proxy_port, local_port, remote_port);
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let (status, body) = http_post(
        proxy.addr(),
        "/mcp",
        r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"weather-lookup","arguments":{}}}"#,
    );
    assert_eq!(status, 200, "known MCP tool should route");
    assert_eq!(body, "weather-response", "should select the tool-owning cluster");
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

/// Build YAML config that mirrors the grid-route-inference.yaml example
/// with dynamic ports substituted in.
fn make_inference_yaml(proxy_port: u16, local_port: u16, remote_port: u16) -> String {
    format!(
        r#"
listeners:
  - name: proxy
    address: "127.0.0.1:{proxy_port}"
    filter_chains:
      - main
filter_chains:
  - name: main
    filters:
      - filter: json_body_field
        field: model
        header: X-Model
      - filter: grid_route
        local_site: site-a
        model_header: X-Model
        candidates:
          - kind: inference_model
            name: granite-3.3-8b
            site: site-a
            cluster: granite-local
            fresh: true
          - kind: inference_model
            name: llama-3.2-8b
            site: site-b
            cluster: llama-remote
            fresh: true
      - filter: load_balancer
        clusters:
          - name: granite-local
            endpoints:
              - "127.0.0.1:{local_port}"
          - name: llama-remote
            endpoints:
              - "127.0.0.1:{remote_port}"
"#
    )
}

/// Build YAML config that mirrors the grid-route-mcp.yaml example
/// with dynamic ports substituted in.
fn make_mcp_yaml(proxy_port: u16, local_port: u16, remote_port: u16) -> String {
    format!(
        r#"
listeners:
  - name: proxy
    address: "127.0.0.1:{proxy_port}"
    filter_chains:
      - main
filter_chains:
  - name: main
    filters:
      - filter: mcp
      - filter: grid_route
        local_site: site-a
        candidates:
          - kind: mcp_tool
            name: weather-lookup
            site: site-b
            cluster: tools-site-b
            fresh: true
          - kind: mcp_tool
            name: code-search
            site: site-a
            cluster: tools-site-a
            fresh: true
      - filter: load_balancer
        clusters:
          - name: tools-site-a
            endpoints:
              - "127.0.0.1:{local_port}"
          - name: tools-site-b
            endpoints:
              - "127.0.0.1:{remote_port}"
"#
    )
}
