// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for the web-search example config.
//!
//! These tests verify that the example config parses, the filter
//! pipeline builds correctly, and requests pass through unchanged.
//! The `openai_web_search` filter is a scaffolded passthrough — it validates
//! config at startup but does not execute searches at runtime.
//!
//! The example config uses `${WEB_SEARCH_API_KEY}` for the API key.
//! Since `set_var` is `unsafe` in Rust 2024 and `unsafe_code` is
//! denied, we patch the YAML to replace the env var reference with a
//! literal test key.

use std::collections::HashMap;

use praxis_core::config::Config;
use praxis_test_utils::{
    free_port, http_send, json_post, parse_body, parse_status, patch_yaml, start_backend_with_shutdown, start_proxy,
};

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// Load the web-search example config with the env var reference
/// replaced by a literal test key.
fn load_web_search_config(proxy_port: u16, port_map: &HashMap<&str, u16>) -> Config {
    let path = praxis_test_utils::example_config_path("openai/responses/web-search.yaml");
    let yaml = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let patched = patch_yaml(&yaml, proxy_port, port_map);
    let patched = patched.replace("${WEB_SEARCH_API_KEY}", "test-key-for-config");
    Config::from_yaml(&patched).unwrap_or_else(|e| panic!("parse web-search.yaml: {e}"))
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn web_search_example_passthrough() {
    let backend_guard = start_backend_with_shutdown("inference");
    let proxy_port = free_port();

    let config = load_web_search_config(proxy_port, &HashMap::from([("127.0.0.1:3001", backend_guard.port())]));
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"Hello","tools":[{"type":"web_search"}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "request should pass through to backend");
    assert_eq!(
        parse_body(&raw),
        "inference",
        "openai_web_search filter is a passthrough — request should reach inference backend"
    );
}

#[test]
fn web_search_example_no_tools_passthrough() {
    let backend_guard = start_backend_with_shutdown("inference");
    let proxy_port = free_port();

    let config = load_web_search_config(proxy_port, &HashMap::from([("127.0.0.1:3001", backend_guard.port())]));
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"Hello"}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "no-tools request should return 200");
    assert_eq!(
        parse_body(&raw),
        "inference",
        "request without tools should route to inference backend"
    );
}
