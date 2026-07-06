// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Integration tests for MCP static catalog and broker behavior.

use std::collections::HashMap;

use praxis_core::config::Config;
use praxis_test_utils::{
    free_port, http_send, load_example_config, parse_body, parse_status, start_backend_with_shutdown, start_proxy,
};

// -----------------------------------------------------------------------------
// Current Tests
// -----------------------------------------------------------------------------

#[test]
fn mcp_broker_initialize_returns_session() {
    let backend_guard = start_backend_with_shutdown("backend");
    let proxy_port = free_port();

    let yaml = broker_yaml(proxy_port, backend_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body =
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{}}}"#;
    let request = json_post("/mcp", body);
    let raw = http_send(proxy.addr(), &request);

    assert_eq!(parse_status(&raw), 200, "initialize should return 200");
    let response_body = parse_body(&raw);
    assert!(
        response_body.contains("protocolVersion"),
        "should contain protocolVersion: {response_body}"
    );
    let parsed: serde_json::Value = serde_json::from_str(&response_body).unwrap();
    assert_eq!(
        parsed["result"]["serverInfo"]["name"], "praxis",
        "should contain Praxis server name: {response_body}"
    );
    assert!(
        raw.contains("mcp-session-id:"),
        "response should contain mcp-session-id header"
    );
    assert_ne!(
        response_body, "backend",
        "response should come from Praxis, not backend"
    );
}

#[test]
fn mcp_broker_tools_list_returns_prefixed_catalog() {
    let backend_guard = start_backend_with_shutdown("backend");
    let proxy_port = free_port();

    let yaml = broker_yaml(proxy_port, backend_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#;
    let request = json_post("/mcp", body);
    let raw = http_send(proxy.addr(), &request);

    assert_eq!(parse_status(&raw), 200, "tools/list should return 200");
    let response_body = parse_body(&raw);
    assert!(
        response_body.contains("weather_get_weather"),
        "should contain prefixed weather tool: {response_body}"
    );
    assert!(
        response_body.contains("cal_create_event"),
        "should contain prefixed calendar tool: {response_body}"
    );
    let parsed: serde_json::Value = serde_json::from_str(&response_body).unwrap();
    let tools = parsed["result"]["tools"].as_array().unwrap();
    assert!(
        tools.iter().all(|tool| tool.get("inputSchema").is_some()),
        "every returned tool should include inputSchema: {response_body}"
    );
    assert_eq!(
        tools[1]["inputSchema"],
        serde_json::json!({"type": "object", "additionalProperties": false}),
        "tools without configured schema should get a closed object inputSchema"
    );
    assert_ne!(
        response_body, "backend",
        "response should come from Praxis, not backend"
    );
}

#[test]
fn mcp_broker_example_serves_prefixed_catalog() {
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
fn mcp_broker_ping_returns_result() {
    let backend_guard = start_backend_with_shutdown("backend");
    let proxy_port = free_port();

    let yaml = broker_yaml(proxy_port, backend_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"jsonrpc":"2.0","id":5,"method":"ping"}"#;
    let request = json_post("/mcp", body);
    let raw = http_send(proxy.addr(), &request);

    assert_eq!(parse_status(&raw), 200, "ping should return 200");
    let response_body = parse_body(&raw);
    assert!(
        response_body.contains(r#""result":{}"#),
        "ping should return empty result: {response_body}"
    );
    assert!(
        response_body.contains(r#""id":5"#),
        "ping should preserve numeric id: {response_body}"
    );
}

#[test]
fn mcp_broker_initialized_notification_returns_202() {
    let backend_guard = start_backend_with_shutdown("backend");
    let proxy_port = free_port();

    let yaml = broker_yaml(proxy_port, backend_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#;
    let request = json_post("/mcp", body);
    let raw = http_send(proxy.addr(), &request);

    assert_eq!(parse_status(&raw), 202, "notifications/initialized should return 202");
    assert_eq!(
        parse_body(&raw),
        "",
        "accepted notifications should not include a JSON-RPC response body"
    );
}

#[test]
fn mcp_broker_ping_with_null_id_rejected() {
    let backend_guard = start_backend_with_shutdown("backend");
    let proxy_port = free_port();

    let yaml = broker_yaml(proxy_port, backend_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"jsonrpc":"2.0","id":null,"method":"ping"}"#;
    let request = json_post("/mcp", body);
    let raw = http_send(proxy.addr(), &request);

    assert_eq!(
        parse_status(&raw),
        200,
        "null request ids should return JSON-RPC errors"
    );
    let response_body = parse_body(&raw);
    assert!(
        response_body.contains("-32600"),
        "null request ids should return invalid request: {response_body}"
    );
    assert!(
        response_body.contains(r#""id":null"#),
        "invalid id response should use null id: {response_body}"
    );
}

#[test]
fn mcp_broker_ping_with_missing_id_rejected() {
    let backend_guard = start_backend_with_shutdown("backend");
    let proxy_port = free_port();

    let yaml = broker_yaml(proxy_port, backend_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"jsonrpc":"2.0","method":"ping"}"#;
    let request = json_post("/mcp", body);
    let raw = http_send(proxy.addr(), &request);

    assert_eq!(
        parse_status(&raw),
        200,
        "request methods without ids should return JSON-RPC errors"
    );
    let response_body = parse_body(&raw);
    assert!(
        response_body.contains("-32600"),
        "missing request ids should return invalid request: {response_body}"
    );
}

#[test]
fn mcp_broker_unsupported_method_returns_method_not_found() {
    let backend_guard = start_backend_with_shutdown("backend");
    let proxy_port = free_port();

    let yaml = broker_yaml(proxy_port, backend_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"jsonrpc":"2.0","id":7,"method":"resources/list"}"#;
    let request = json_post("/mcp", body);
    let raw = http_send(proxy.addr(), &request);

    assert_eq!(
        parse_status(&raw),
        200,
        "unsupported method should return a JSON-RPC error response"
    );
    let response_body = parse_body(&raw);
    assert!(
        response_body.contains("-32601"),
        "unsupported method should return -32601: {response_body}"
    );
}

#[test]
fn mcp_broker_tools_call_not_forwarded_before_routing() {
    let backend_guard = start_backend_with_shutdown("not-reachable-backend");
    let proxy_port = free_port();

    let yaml = broker_yaml(proxy_port, backend_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body =
        r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"weather_get_weather","arguments":{}}}"#;
    let request = json_post("/mcp", body);
    let raw = http_send(proxy.addr(), &request);

    let status = parse_status(&raw);
    let response_body = parse_body(&raw);

    assert_eq!(
        status, 200,
        "tools/call should return a JSON-RPC error response before backend routing is added"
    );
    assert!(
        response_body.contains("-32601"),
        "tools/call should return -32601 before backend routing is added: {response_body}"
    );
    assert!(
        !response_body.contains("not-reachable-backend"),
        "tools/call must not reach the backend before routing is added"
    );
}

#[test]
fn mcp_broker_delete_returns_controlled_response() {
    let backend_guard = start_backend_with_shutdown("backend");
    let proxy_port = free_port();

    let yaml = broker_yaml(proxy_port, backend_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let request = format!(
        "DELETE /mcp HTTP/1.1\r\n\
         Host: localhost\r\n\
         Mcp-Session-Id: mcp-test-session\r\n\
         Connection: close\r\n\
         \r\n"
    );
    let raw = http_send(proxy.addr(), &request);

    assert_eq!(parse_status(&raw), 204, "DELETE with session should return 204");
}

#[test]
fn mcp_broker_wrong_path_returns_404() {
    let backend_guard = start_backend_with_shutdown("backend");
    let proxy_port = free_port();

    let yaml = broker_yaml(proxy_port, backend_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#;
    let request = json_post("/not-mcp", body);
    let raw = http_send(proxy.addr(), &request);

    assert_eq!(parse_status(&raw), 404, "POST to /not-mcp should return 404");
    assert!(
        !parse_body(&raw).contains("backend"),
        "wrong-path request must not reach backend"
    );
}

#[test]
fn mcp_broker_ping_with_query_param() {
    let backend_guard = start_backend_with_shutdown("not-reachable-backend");
    let proxy_port = free_port();

    let yaml = broker_yaml(proxy_port, backend_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"jsonrpc":"2.0","id":9,"method":"ping"}"#;
    let request = json_post("/mcp?x=1", body);
    let raw = http_send(proxy.addr(), &request);

    assert_eq!(
        parse_status(&raw),
        200,
        "POST /mcp?x=1 should match configured MCP path"
    );
    let response_body = parse_body(&raw);
    assert!(
        response_body.contains(r#""result":{}"#),
        "ping should return empty result: {response_body}"
    );
    assert!(
        !response_body.contains("not-reachable-backend"),
        "query-param request must not reach backend"
    );
}

// -----------------------------------------------------------------------------
// Stateless Tests
// -----------------------------------------------------------------------------

#[test]
fn mcp_stateless_server_discover_returns_200() {
    let backend_guard = start_backend_with_shutdown("not-reachable-backend");
    let proxy_port = free_port();

    let yaml = stateless_broker_yaml(proxy_port, backend_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body_str = &stateless_rpc_body(1, "server/discover", None);
    let request = stateless_post("/mcp", body_str, "server/discover", None);
    let raw = http_send(proxy.addr(), &request);

    assert_eq!(parse_status(&raw), 200, "server/discover should return 200");
    let body = parse_body(&raw);
    let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(parsed["result"]["resultType"], "complete");
    assert!(parsed["result"]["supportedVersions"].is_array());
    assert!(parsed["result"]["ttlMs"].is_number());
    assert_eq!(parsed["result"]["cacheScope"], "public");
    assert_eq!(parsed["result"]["serverInfo"]["name"], "praxis");
    assert!(
        parsed["result"]["serverInfo"]["version"].is_string(),
        "serverInfo must include version"
    );
    assert!(
        !raw.contains("mcp-session-id:"),
        "stateless server/discover must not return session header"
    );
    assert!(
        !body.contains("not-reachable-backend"),
        "server/discover must not contact backend"
    );
}

#[test]
fn mcp_stateless_tools_list_returns_catalog_with_cache_metadata() {
    let backend_guard = start_backend_with_shutdown("not-reachable-backend");
    let proxy_port = free_port();

    let yaml = stateless_broker_yaml(proxy_port, backend_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body_str = &stateless_rpc_body(2, "tools/list", None);
    let request = stateless_post("/mcp", body_str, "tools/list", None);
    let raw = http_send(proxy.addr(), &request);

    assert_eq!(parse_status(&raw), 200, "stateless tools/list should return 200");
    let body = parse_body(&raw);
    let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(parsed["result"]["resultType"], "complete");
    assert!(parsed["result"]["ttlMs"].is_number());
    assert_eq!(parsed["result"]["cacheScope"], "public");
    assert!(parsed["result"]["tools"].is_array());
    assert!(body.contains("weather_get_weather"), "should include prefixed tools");
    assert!(
        !raw.contains("mcp-session-id:"),
        "stateless response must not include session header"
    );
    assert!(
        !body.contains("not-reachable-backend"),
        "tools/list must not contact backend"
    );
}

#[test]
fn mcp_stateless_tools_call_returns_unsupported_after_header_validation() {
    let backend_guard = start_backend_with_shutdown("not-reachable-backend");
    let proxy_port = free_port();

    let yaml = stateless_broker_yaml(proxy_port, backend_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body_str = &stateless_rpc_body(3, "tools/call", Some(r#""name":"weather_get_weather","arguments":{}"#));
    let request = stateless_post("/mcp", body_str, "tools/call", Some("weather_get_weather"));
    let raw = http_send(proxy.addr(), &request);

    assert_eq!(
        parse_status(&raw),
        404,
        "stateless tools/call should return 404 per draft Streamable HTTP"
    );
    let body = parse_body(&raw);
    let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(
        parsed["error"]["code"], -32601,
        "should return method not yet supported"
    );
    assert!(
        !body.contains("not-reachable-backend"),
        "tools/call must not hit backend"
    );
}

#[test]
fn mcp_stateless_missing_mcp_method_rejects_with_400_32020() {
    let backend_guard = start_backend_with_shutdown("not-reachable-backend");
    let proxy_port = free_port();

    let yaml = stateless_broker_yaml(proxy_port, backend_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body_str = stateless_rpc_body(4, "ping", None);
    let request = format!(
        "POST /mcp HTTP/1.1\r\n\
         Host: localhost\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         MCP-Protocol-Version: 2026-07-28\r\n\
         Connection: close\r\n\
         \r\n\
         {body_str}",
        body_str.len(),
    );
    let raw = http_send(proxy.addr(), &request);

    assert_eq!(parse_status(&raw), 400, "missing Mcp-Method should return 400");
    let body = parse_body(&raw);
    let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(parsed["error"]["code"], -32020, "should return -32020 header mismatch");
}

#[test]
fn mcp_stateless_notifications_initialized_rejects_with_404_32601() {
    let backend_guard = start_backend_with_shutdown("not-reachable-backend");
    let proxy_port = free_port();

    let yaml = stateless_broker_yaml(proxy_port, backend_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body_str = stateless_notification_body("notifications/initialized");
    let request = stateless_post("/mcp", &body_str, "notifications/initialized", None);
    let raw = http_send(proxy.addr(), &request);

    assert_eq!(
        parse_status(&raw),
        404,
        "stateless notifications/initialized should be removed with the initialize handshake"
    );
    let body = parse_body(&raw);
    let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(parsed["error"]["code"], -32601, "should return method not found");
    assert!(
        !body.contains("not-reachable-backend"),
        "stateless notifications/initialized must not hit backend"
    );
}

#[test]
fn mcp_stateless_unsupported_version_rejects_with_400_32022() {
    let backend_guard = start_backend_with_shutdown("not-reachable-backend");
    let proxy_port = free_port();

    let yaml = stateless_broker_yaml(proxy_port, backend_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body_str = stateless_rpc_body_with_version(5, "ping", "9999-12-31");
    let request = format!(
        "POST /mcp HTTP/1.1\r\n\
         Host: localhost\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         MCP-Protocol-Version: 9999-12-31\r\n\
         Mcp-Method: ping\r\n\
         Connection: close\r\n\
         \r\n\
         {body_str}",
        body_str.len(),
    );
    let raw = http_send(proxy.addr(), &request);

    assert_eq!(parse_status(&raw), 400, "unsupported version should return 400");
    let body = parse_body(&raw);
    let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(
        parsed["error"]["code"], -32022,
        "should return -32022 unsupported version"
    );
    assert!(
        parsed["error"]["data"]["supported"].is_array(),
        "should include supported list"
    );
    assert!(
        parsed["error"]["data"]["requested"].is_string(),
        "should include requested version"
    );
}

#[test]
fn mcp_stateless_malformed_client_info_rejected() {
    let backend_guard = start_backend_with_shutdown("not-reachable-backend");
    let proxy_port = free_port();

    let yaml = stateless_broker_yaml(proxy_port, backend_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body_str = r#"{"jsonrpc":"2.0","id":6,"method":"ping","params":{"_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientInfo":"not-an-object","io.modelcontextprotocol/clientCapabilities":{}}}}"#;
    let request = stateless_post("/mcp", body_str, "ping", None);
    let raw = http_send(proxy.addr(), &request);

    assert_eq!(parse_status(&raw), 400, "malformed clientInfo should return 400");
    let body = parse_body(&raw);
    let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(parsed["error"]["code"], -32020, "should return -32020 header mismatch");
}

#[test]
fn mcp_current_initialize_still_returns_session_header() {
    let backend_guard = start_backend_with_shutdown("backend");
    let proxy_port = free_port();

    let yaml = broker_yaml(proxy_port, backend_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body =
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{}}}"#;
    let request = json_post("/mcp", body);
    let raw = http_send(proxy.addr(), &request);

    assert_eq!(parse_status(&raw), 200, "current initialize should return 200");
    assert!(
        raw.contains("mcp-session-id:"),
        "current profile initialize must still return session header"
    );
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

/// Stateless request metadata fragment for `params._meta`.
const STATELESS_META: &str = concat!(
    r#""_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28","#,
    r#""io.modelcontextprotocol/clientInfo":{"name":"test","version":"1.0"},"#,
    r#""io.modelcontextprotocol/clientCapabilities":{}}"#,
);

fn stateless_rpc_body(id: u64, method: &str, extra_params: Option<&str>) -> String {
    let extra = extra_params.map_or(String::new(), |p| format!(",{p}"));
    format!(r#"{{"jsonrpc":"2.0","id":{id},"method":"{method}","params":{{{STATELESS_META}{extra}}}}}"#)
}

fn stateless_notification_body(method: &str) -> String {
    format!(r#"{{"jsonrpc":"2.0","method":"{method}","params":{{{STATELESS_META}}}}}"#)
}

fn stateless_rpc_body_with_version(id: u64, method: &str, version: &str) -> String {
    format!(
        r#"{{"jsonrpc":"2.0","id":{id},"method":"{method}","params":{{"_meta":{{"io.modelcontextprotocol/protocolVersion":"{version}","io.modelcontextprotocol/clientInfo":{{"name":"test","version":"1.0"}},"io.modelcontextprotocol/clientCapabilities":{{}}}}}}}}"#,
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

fn broker_yaml(proxy_port: u16, backend_port: u16) -> String {
    format!(
        r#"
listeners:
  - name: default
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: mcp
        path: /mcp
        max_body_bytes: 65536
        servers:
          - name: weather
            cluster: weather-mcp
            path: /mcp
            tool_prefix: "weather_"
            tools:
              - name: get_weather
                description: Get current weather
          - name: calendar
            cluster: calendar-mcp
            path: /mcp
            tool_prefix: "cal_"
            tools:
              - name: create_event
                description: Create a calendar event
      - filter: load_balancer
        clusters:
          - name: weather-mcp
            endpoints:
              - "127.0.0.1:{backend_port}"
          - name: calendar-mcp
            endpoints:
              - "127.0.0.1:{backend_port}"
"#,
    )
}

fn stateless_broker_yaml(proxy_port: u16, backend_port: u16) -> String {
    format!(
        r#"
listeners:
  - name: default
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: mcp
        path: /mcp
        max_body_bytes: 65536
        protocol_profile: stateless
        servers:
          - name: weather
            cluster: weather-mcp
            path: /mcp
            tool_prefix: "weather_"
            tools:
              - name: get_weather
                description: Get current weather
          - name: calendar
            cluster: calendar-mcp
            path: /mcp
            tool_prefix: "cal_"
            tools:
              - name: create_event
                description: Create a calendar event
      - filter: load_balancer
        clusters:
          - name: weather-mcp
            endpoints:
              - "127.0.0.1:{backend_port}"
          - name: calendar-mcp
            endpoints:
              - "127.0.0.1:{backend_port}"
"#,
    )
}
