// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Integration tests for the file-search-callout example config.
//!
//! These tests verify that the example config parses, the registered filter
//! pipeline builds, and first-pass requests reach inference unchanged. Live
//! vector-store execution requires response-driven pipeline continuation and
//! is therefore covered by state-injection tests in the API crate for now.
//!
//! The example uses `${VECTOR_STORE_API_KEY}`. Since `set_var` is `unsafe` in
//! Rust 2024 and `unsafe_code` is denied, the loader substitutes a test key.

use std::collections::HashMap;

use praxis_core::config::Config;
use praxis_test_utils::{
    free_port, http_send, json_post, parse_body, parse_status, patch_yaml, start_backend_with_shutdown, start_proxy,
};

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// Load the example with its environment reference replaced by a test key.
fn load_file_search_callout_config(proxy_port: u16, port_map: &HashMap<&str, u16>) -> Config {
    let path = praxis_test_utils::example_config_path("openai/responses/file-search-callout.yaml");
    let yaml = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let patched = patch_yaml(&yaml, proxy_port, port_map);
    let patched = patched.replace("${VECTOR_STORE_API_KEY}", "test-key-for-config");
    Config::from_yaml(&patched).unwrap_or_else(|e| panic!("parse file-search-callout.yaml: {e}"))
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn file_search_callout_example_first_pass_passthrough() {
    let response = r#"{"id":"resp_123","object":"response","output":[{"id":"msg_123","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Search requested","annotations":[]}]}]}"#;
    let backend_guard = start_backend_with_shutdown(response);
    let proxy_port = free_port();

    let config =
        load_file_search_callout_config(proxy_port, &HashMap::from([("127.0.0.1:3001", backend_guard.port())]));
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"Find the report","tools":[{"type":"file_search","vector_store_ids":["vs_123"]}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "first-pass request should reach inference");
    assert_eq!(
        parse_body(&raw),
        response,
        "file-search callout should remain dormant until response-driven continuation"
    );
}

#[test]
fn file_search_callout_example_without_tools_passthrough() {
    let response = r#"{"id":"resp_456","object":"response","output":[{"id":"msg_456","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Hello","annotations":[]}]}]}"#;
    let backend_guard = start_backend_with_shutdown(response);
    let proxy_port = free_port();

    let config =
        load_file_search_callout_config(proxy_port, &HashMap::from([("127.0.0.1:3001", backend_guard.port())]));
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"Hello"}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "request without tools should return 200");
    assert_eq!(parse_body(&raw), response, "request should reach inference backend");
}
