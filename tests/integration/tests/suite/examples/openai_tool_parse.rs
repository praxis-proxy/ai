// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for the tool-routing example config.

use std::collections::HashMap;

use praxis_test_utils::{
    free_port, http_send, json_post, load_example_config, parse_body, parse_status, start_backend_with_shutdown,
    start_proxy,
};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn tool_routing_example_routes_request_with_tools() {
    let backend_guard = start_backend_with_shutdown("inference");
    let proxy_port = free_port();

    let config = load_example_config(
        "openai/responses/tool-routing.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3001", backend_guard.port())]),
    );
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"test","tools":[{"type":"function","name":"calc"}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "request with tools should return 200");
    assert_eq!(
        parse_body(&raw),
        "inference",
        "request with function tools should reach inference backend"
    );
}

#[test]
fn tool_routing_example_routes_request_without_tools() {
    let backend_guard = start_backend_with_shutdown("inference");
    let proxy_port = free_port();

    let config = load_example_config(
        "openai/responses/tool-routing.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3001", backend_guard.port())]),
    );
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"test"}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "request without tools should return 200");
    assert_eq!(
        parse_body(&raw),
        "inference",
        "request without tools should reach inference backend"
    );
}

#[test]
fn tool_routing_example_branches_web_search() {
    let backend_guard = start_backend_with_shutdown("inference");
    let proxy_port = free_port();

    let config = load_example_config(
        "openai/responses/tool-routing.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3001", backend_guard.port())]),
    );
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"test","tools":[{"type":"web_search"}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "web_search request should return 200");
    assert_eq!(
        parse_body(&raw),
        "inference",
        "web_search should branch to web-search chain and reach inference backend"
    );
}
