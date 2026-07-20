// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Tests for semantic router example configuration.

use praxis_core::config::Config;
use praxis_test_utils::{free_port, http_post, start_backend_with_shutdown, start_proxy};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn semantic_router_routes_by_mock_score() {
    let port_heavy_guard = start_backend_with_shutdown("heavy-response");
    let port_heavy = port_heavy_guard.port();
    
    let port_light_guard = start_backend_with_shutdown("light-response");
    let port_light = port_light_guard.port();

    let port_default_guard = start_backend_with_shutdown("default-response");
    let port_default = port_default_guard.port();

    let proxy_port = free_port();
    let yaml = make_yaml(proxy_port, port_heavy, port_light, port_default);
    
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);
    
    // The mock backend always returns 0.1, which falls into the light_cluster bucket (max_score: 0.79)
    let (status, body) = http_post(
        proxy.addr(),
        "/v1/chat/completions",
        r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hello"}]}"#,
    );
    
    assert_eq!(status, 200, "valid request should return 200");
    assert_eq!(
        body, "light-response",
        "mock score of 0.1 should route to light backend"
    );
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

/// Build YAML config for semantic router with three clusters.
fn make_yaml(proxy_port: u16, port_heavy: u16, port_light: u16, port_default: u16) -> String {
    format!(
        r#"
listeners:
  - name: default
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: semantic_router
        backend: mock
        routes:
          - min_score: 0.8
            target_cluster: heavy_cluster
          - max_score: 0.79
            target_cluster: light_cluster
      - filter: load_balancer
        clusters:
          - name: heavy_cluster
            endpoints:
              - "127.0.0.1:{port_heavy}"
          - name: light_cluster
            endpoints:
              - "127.0.0.1:{port_light}"
          - name: default_cluster
            endpoints:
              - "127.0.0.1:{port_default}"
"#
    )
}