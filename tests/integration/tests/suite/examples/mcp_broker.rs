// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Integration tests for MCP broker example configurations.

use std::collections::HashMap;

use praxis_test_utils::{
    free_port, http_send, load_example_config, parse_body, parse_status, start_backend_with_shutdown, start_proxy,
};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn mcp_static_catalog_example_serves_prefixed_catalog() {
    let weather_guard = start_backend_with_shutdown("weather");
    let calendar_guard = start_backend_with_shutdown("calendar");
    let proxy_port = free_port();

    let config = load_example_config(
        "payload-processing/mcp-static-catalog.yaml",
        proxy_port,
        HashMap::from([
            ("127.0.0.1:3001", weather_guard.port()),
            ("127.0.0.1:3002", calendar_guard.port()),
        ]),
    );
    let proxy = start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        &json_post("/mcp", r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#),
    );

    assert_eq!(parse_status(&raw), 200, "tools/list should return 200");
    let body = parse_body(&raw);
    assert!(
        body.contains("weather_get_weather"),
        "weather tool should include prefix: {body}"
    );
    assert!(
        body.contains("weather_forecast"),
        "weather forecast should include prefix: {body}"
    );
    assert!(
        body.contains("cal_create_event"),
        "calendar create tool should include prefix: {body}"
    );
    assert!(
        body.contains("cal_list_events"),
        "calendar list tool should include prefix: {body}"
    );
    assert!(
        body.contains(r#""city""#),
        "example inputSchema should be preserved: {body}"
    );
}

#[test]
fn mcp_stateless_broker_example_serves_discover() {
    let weather_guard = start_backend_with_shutdown("weather");
    let calendar_guard = start_backend_with_shutdown("calendar");
    let proxy_port = free_port();

    let config = load_example_config(
        "mcp-stateless-broker.yaml",
        proxy_port,
        HashMap::from([
            ("127.0.0.1:3001", weather_guard.port()),
            ("127.0.0.1:3002", calendar_guard.port()),
        ]),
    );
    let proxy = start_proxy(&config);

    let body_str = stateless_rpc_body(1, "server/discover");
    let request = stateless_post("/mcp", &body_str, "server/discover", None);
    let raw = http_send(proxy.addr(), &request);

    assert_eq!(parse_status(&raw), 200, "server/discover should return 200");
    let body = parse_body(&raw);
    let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(parsed["result"]["resultType"], "complete", "should include resultType");
    assert!(
        parsed["result"]["supportedVersions"].is_array(),
        "should include supportedVersions"
    );
    assert!(parsed["result"]["ttlMs"].is_number(), "should include ttlMs");
    assert_eq!(parsed["result"]["cacheScope"], "public", "should include cacheScope");
    assert_eq!(
        parsed["result"]["serverInfo"]["name"], "praxis",
        "should identify as praxis"
    );
    assert!(
        parsed["result"]["serverInfo"]["version"].is_string(),
        "serverInfo must include version"
    );
}

#[test]
fn mcp_stateless_broker_example_serves_tools_list() {
    let weather_guard = start_backend_with_shutdown("weather");
    let calendar_guard = start_backend_with_shutdown("calendar");
    let proxy_port = free_port();

    let config = load_example_config(
        "mcp-stateless-broker.yaml",
        proxy_port,
        HashMap::from([
            ("127.0.0.1:3001", weather_guard.port()),
            ("127.0.0.1:3002", calendar_guard.port()),
        ]),
    );
    let proxy = start_proxy(&config);

    let body_str = stateless_rpc_body(2, "tools/list");
    let request = stateless_post("/mcp", &body_str, "tools/list", None);
    let raw = http_send(proxy.addr(), &request);

    assert_eq!(parse_status(&raw), 200, "tools/list should return 200");
    let body = parse_body(&raw);
    let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(parsed["result"]["resultType"], "complete");
    assert!(parsed["result"]["tools"].is_array(), "should include tools array");
    assert!(parsed["result"]["ttlMs"].is_number(), "should include ttlMs");
    assert_eq!(parsed["result"]["cacheScope"], "public");
    assert!(
        body.contains("weather_get_weather"),
        "should contain prefixed weather tool"
    );
    assert!(
        body.contains("cal_create_event"),
        "should contain prefixed calendar tool"
    );
    assert!(!raw.contains("mcp-session-id:"), "should not include session header");
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

fn stateless_rpc_body(id: u64, method: &str) -> String {
    format!(
        r#"{{"jsonrpc":"2.0","id":{id},"method":"{method}","params":{{"_meta":{{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientInfo":{{"name":"test","version":"1.0"}},"io.modelcontextprotocol/clientCapabilities":{{}}}}}}}}"#,
    )
}

fn json_post(path: &str, body: &str) -> String {
    format!(
        "POST {path} HTTP/1.1\r\n\
         Host: localhost\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\
         \r\n\
         {body}",
        body.len(),
    )
}

fn stateless_post(path: &str, body: &str, method: &str, mcp_name: Option<&str>) -> String {
    let mut headers = format!(
        "POST {path} HTTP/1.1\r\n\
         Host: localhost\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         MCP-Protocol-Version: 2026-07-28\r\n\
         Mcp-Method: {method}\r\n",
        body.len(),
    );
    if let Some(name) = mcp_name {
        headers.push_str(&format!("Mcp-Name: {name}\r\n"));
    }
    headers.push_str("Connection: close\r\n\r\n");
    headers.push_str(body);
    headers
}
