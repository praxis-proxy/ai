// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Integration tests for MCP broker example configurations.

use std::collections::HashMap;

use praxis_test_utils::{
    McpMockConfig, McpToolFixture, free_port, http_send, load_example_config, parse_body, parse_header, parse_status,
    start_backend_with_shutdown, start_echo_backend, start_mcp_mock_server_with_config, start_proxy,
};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn mcp_session_broker_example_initializes_backend_session() {
    let weather = start_mcp_mock_server_with_config(McpMockConfig {
        enforce_sessions: true,
        tools: vec![McpToolFixture::new("get_weather")],
        ..McpMockConfig::default()
    });
    let calendar = start_mcp_mock_server_with_config(McpMockConfig {
        enforce_sessions: true,
        tools: vec![McpToolFixture::new("create_event")],
        ..McpMockConfig::default()
    });
    let proxy_port = free_port();
    let config = load_example_config(
        "mcp-session-broker.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3001", weather.port()), ("127.0.0.1:3002", calendar.port())]),
    );
    let proxy = start_proxy(&config);

    let init = http_send(
        proxy.addr(),
        &json_post(
            "/mcp",
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{}}}"#,
        ),
    );
    assert_eq!(parse_status(&init), 200, "initialize should return 200");
    let gateway = parse_header(&init, "mcp-session-id").expect("gateway session must be issued");

    let body = r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"cal_create_event","arguments":{}}}"#;
    let call = format!(
        "POST /mcp HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n\
         Accept: application/json, text/event-stream\r\nMcp-Session-Id: {gateway}\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let raw = http_send(proxy.addr(), &call);

    assert_eq!(parse_status(&raw), 200, "tools/call should succeed: {raw}");
    assert!(
        parse_body(&raw).contains("mock result for create_event"),
        "backend result must reach the client: {raw}"
    );
    assert_eq!(
        calendar.method_count("initialize"),
        1,
        "calendar backend must be initialized"
    );
    assert_eq!(
        calendar.session_ids_for("tools/call"),
        vec![Some("mock-mcp-session-1".to_owned())],
        "the call must carry the backend session"
    );
}

#[test]
fn mcp_session_broker_example_discovers_weather_tools() {
    let weather = start_mcp_mock_server_with_config(McpMockConfig {
        enforce_sessions: true,
        tools: vec![
            McpToolFixture::new("get_weather").with_description("listed description"),
            McpToolFixture::new("forecast"),
        ],
        ..McpMockConfig::default()
    });
    let calendar = start_mcp_mock_server_with_config(McpMockConfig {
        enforce_sessions: true,
        tools: vec![McpToolFixture::new("create_event")],
        ..McpMockConfig::default()
    });
    let config = load_example_config(
        "mcp-session-broker.yaml",
        free_port(),
        HashMap::from([("127.0.0.1:3001", weather.port()), ("127.0.0.1:3002", calendar.port())]),
    );
    let proxy = start_proxy(&config);
    let init = http_send(
        proxy.addr(),
        &json_post(
            "/mcp",
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{}}}"#,
        ),
    );
    let gateway = parse_header(&init, "mcp-session-id").expect("gateway session must be issued");

    let raw = http_send(
        proxy.addr(),
        &session_post(&gateway, r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#),
    );
    assert_eq!(parse_status(&raw), 200, "tools/list should succeed: {raw}");
    let body: serde_json::Value = serde_json::from_str(&parse_body(&raw)).unwrap();
    let tools = body["result"]["tools"].as_array().unwrap();
    let names: Vec<&str> = tools.iter().map(|tool| tool["name"].as_str().unwrap()).collect();
    assert_eq!(
        names,
        ["weather_get_weather", "weather_forecast", "cal_create_event"],
        "weather tools come from its tools/list: {body}"
    );
    assert_eq!(
        tools[0]["description"], "Get current weather",
        "the static entry overrides the listed description"
    );
    assert_eq!(weather.method_count("tools/list"), 1);

    let raw = http_send(
        proxy.addr(),
        &session_post(
            &gateway,
            r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"weather_forecast","arguments":{}}}"#,
        ),
    );
    assert_eq!(parse_status(&raw), 200, "a discovered tool is callable: {raw}");
    assert!(parse_body(&raw).contains("mock result for forecast"), "{raw}");
}

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

#[test]
fn mcp_stateless_broker_example_routes_tools_call() {
    let weather_guard = start_echo_backend();
    let calendar_guard = start_echo_backend();
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

    let body_str = stateless_rpc_body_with_params(
        3,
        "tools/call",
        r#""name":"weather_get_weather","arguments":{"city":"NYC"}"#,
    );
    let request = stateless_post("/mcp", &body_str, "tools/call", Some("weather_get_weather"));
    let raw = http_send(proxy.addr(), &request);

    assert_eq!(parse_status(&raw), 200, "routed tools/call should return 200");
    let body = parse_body(&raw);
    let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(
        parsed["params"]["name"], "get_weather",
        "backend should receive stripped params.name"
    );
    assert_eq!(
        parsed["params"]["arguments"]["city"], "NYC",
        "arguments should be preserved"
    );
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

fn stateless_rpc_body(id: u64, method: &str) -> String {
    format!(
        r#"{{"jsonrpc":"2.0","id":{id},"method":"{method}","params":{{"_meta":{{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientInfo":{{"name":"test","version":"1.0"}},"io.modelcontextprotocol/clientCapabilities":{{}}}}}}}}"#,
    )
}

fn stateless_rpc_body_with_params(id: u64, method: &str, extra_params: &str) -> String {
    format!(
        r#"{{"jsonrpc":"2.0","id":{id},"method":"{method}","params":{{"_meta":{{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientInfo":{{"name":"test","version":"1.0"}},"io.modelcontextprotocol/clientCapabilities":{{}}}},{extra_params}}}}}"#,
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

/// A JSON POST on the gateway session.
fn session_post(gateway: &str, body: &str) -> String {
    format!(
        "POST /mcp HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n\
         Accept: application/json, text/event-stream\r\nMcp-Session-Id: {gateway}\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
}
