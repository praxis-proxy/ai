// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Tests for the reusable MCP and A2A mock servers.

use praxis_test_utils::{
    A2aMockConfig, McpMockConfig, McpToolFixture, http_send, parse_body, parse_header, parse_status,
    start_a2a_mock_server, start_a2a_mock_server_with_config, start_mcp_mock_server, start_mcp_mock_server_with_config,
};
use serde_json::Value;

// -----------------------------------------------------------------------------
// MCP Tests
// -----------------------------------------------------------------------------

#[test]
fn mcp_mock_initialize_returns_session() {
    let server = start_mcp_mock_server();
    let body = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#;
    let raw = json_post_raw(&server.endpoint(), server.path(), body, &[]);

    let status = parse_status(&raw);
    assert_eq!(status, 200, "initialize should return 200");

    let json: Value = serde_json::from_str(&parse_body(&raw)).unwrap();
    let result = &json["result"];
    assert!(
        result["protocolVersion"].is_string(),
        "result should include protocolVersion"
    );
    assert!(result["capabilities"].is_object(), "result should include capabilities");
    assert!(
        result["serverInfo"]["name"].is_string(),
        "result should include serverInfo.name"
    );

    let session_id = parse_header(&raw, "MCP-Session-Id");
    assert_eq!(
        session_id.as_deref(),
        Some("mock-mcp-session-1"),
        "stateful server should emit MCP-Session-Id"
    );

    assert_eq!(
        server.method_count("initialize"),
        1,
        "should record exactly one initialize call"
    );
}

#[test]
fn mcp_mock_tools_list_returns_configured_tools() {
    let config = McpMockConfig {
        tools: vec![
            McpToolFixture::new("get_weather").with_description("Get the weather"),
            McpToolFixture::new("create_event"),
        ],
        ..McpMockConfig::default()
    };
    let server = start_mcp_mock_server_with_config(config);

    let body = r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#;
    let raw = json_post_raw(&server.endpoint(), server.path(), body, &[]);

    let status = parse_status(&raw);
    assert_eq!(status, 200, "tools/list should return 200");

    let json: Value = serde_json::from_str(&parse_body(&raw)).unwrap();
    let tools = json["result"]["tools"]
        .as_array()
        .expect("result.tools should be an array");
    assert_eq!(tools.len(), 2, "should return exactly two tools");

    for tool in tools {
        assert!(tool["name"].is_string(), "each tool should have a name");
        assert!(
            tool["inputSchema"].is_object(),
            "each tool should have an object inputSchema"
        );
    }

    let names: Vec<&str> = tools.iter().filter_map(|t| t["name"].as_str()).collect();
    assert!(names.contains(&"get_weather"), "should include get_weather");
    assert!(names.contains(&"create_event"), "should include create_event");
}

#[test]
fn mcp_mock_tools_call_records_body_and_headers() {
    let config = McpMockConfig {
        tools: vec![McpToolFixture::new("get_weather")],
        ..McpMockConfig::default()
    };
    let server = start_mcp_mock_server_with_config(config);

    let body = r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"get_weather","arguments":{}}}"#;
    let headers = [("MCP-Session-Id", "mock-mcp-session-1")];
    let raw = json_post_raw(&server.endpoint(), server.path(), body, &headers);

    let status = parse_status(&raw);
    assert_eq!(status, 200, "tools/call should return 200");

    assert_eq!(
        server.last_tool_call_name().as_deref(),
        Some("get_weather"),
        "should record the tool name"
    );

    let reqs = server.received_requests();
    let last = reqs.last().expect("should have at least one request");
    let has_session_header = last
        .headers
        .iter()
        .any(|(k, v)| k == "mcp-session-id" && v == "mock-mcp-session-1");
    assert!(has_session_header, "recorded headers should include the session header");
}

#[test]
fn mcp_mock_notification_returns_202() {
    let server = start_mcp_mock_server();

    let body = r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#;
    let raw = json_post_raw(&server.endpoint(), server.path(), body, &[]);

    let status = parse_status(&raw);
    assert_eq!(status, 202, "notification should return 202");

    let resp_body = parse_body(&raw);
    assert!(resp_body.is_empty(), "notification should have no response body");
}

#[test]
fn mcp_mock_query_path_matches() {
    let server = start_mcp_mock_server();

    let body = r#"{"jsonrpc":"2.0","id":4,"method":"ping"}"#;
    let path_with_query = format!("{}?trace_id=abc", server.path());
    let raw = json_post_raw(&server.endpoint(), &path_with_query, body, &[]);

    let status = parse_status(&raw);
    assert_eq!(status, 200, "query string should not prevent path matching");

    let json: Value = serde_json::from_str(&parse_body(&raw)).unwrap();
    assert!(json["result"].is_object(), "ping should return an object result");
}

#[test]
fn mcp_mock_unknown_tool_returns_error() {
    let server = start_mcp_mock_server();

    let body = r#"{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"nonexistent","arguments":{}}}"#;
    let raw = json_post_raw(&server.endpoint(), server.path(), body, &[]);

    let status = parse_status(&raw);
    assert_eq!(status, 200, "unknown tool should still return HTTP 200");

    let json: Value = serde_json::from_str(&parse_body(&raw)).unwrap();
    assert_eq!(
        json["error"]["code"], -32602,
        "unknown tool should return JSON-RPC error code -32602"
    );
}

#[test]
fn mcp_mock_unknown_method_returns_error() {
    let server = start_mcp_mock_server();

    let body = r#"{"jsonrpc":"2.0","id":6,"method":"bogus/method"}"#;
    let raw = json_post_raw(&server.endpoint(), server.path(), body, &[]);

    let status = parse_status(&raw);
    assert_eq!(status, 200, "unknown method should return HTTP 200");

    let json: Value = serde_json::from_str(&parse_body(&raw)).unwrap();
    assert_eq!(
        json["error"]["code"], -32601,
        "unknown method should return JSON-RPC error code -32601"
    );
}

#[test]
fn mcp_mock_wrong_path_returns_404() {
    let server = start_mcp_mock_server();

    let body = r#"{"jsonrpc":"2.0","id":7,"method":"ping"}"#;
    let raw = json_post_raw(&server.endpoint(), "/wrong", body, &[]);

    let status = parse_status(&raw);
    assert_eq!(status, 404, "wrong path should return 404");
}

#[test]
fn mcp_mock_delete_returns_204() {
    let server = start_mcp_mock_server();

    let req = format!(
        "DELETE {} HTTP/1.1\r\n\
         Host: localhost\r\n\
         Connection: close\r\n\r\n",
        server.path()
    );
    let raw = http_send(&server.endpoint(), &req);

    let status = parse_status(&raw);
    assert_eq!(status, 204, "DELETE should return 204");
}

#[test]
fn mcp_mock_get_method_rejected() {
    let server = start_mcp_mock_server();

    let req = format!(
        "GET {} HTTP/1.1\r\n\
         Host: localhost\r\n\
         Connection: close\r\n\r\n",
        server.path()
    );
    let raw = http_send(&server.endpoint(), &req);

    let status = parse_status(&raw);
    assert_eq!(status, 405, "GET should be rejected with 405");
}

#[test]
fn mcp_mock_put_method_rejected() {
    let server = start_mcp_mock_server();

    let body = r#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#;
    let req = format!(
        "PUT {} HTTP/1.1\r\n\
         Host: localhost\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\r\n\
         {body}",
        server.path(),
        body.len()
    );
    let raw = http_send(&server.endpoint(), &req);

    let status = parse_status(&raw);
    assert_eq!(status, 405, "PUT should be rejected with 405");
}

// -----------------------------------------------------------------------------
// A2A Tests
// -----------------------------------------------------------------------------

#[test]
fn a2a_mock_send_message_returns_task() {
    let server = start_a2a_mock_server();

    let body = r#"{"jsonrpc":"2.0","id":1,"method":"SendMessage","params":{"message":{"role":"user","parts":[{"kind":"text","text":"hello"}]}}}"#;
    let raw = json_post_raw(&server.endpoint(), server.path(), body, &[]);

    let status = parse_status(&raw);
    assert_eq!(status, 200, "SendMessage should return 200");

    let json: Value = serde_json::from_str(&parse_body(&raw)).unwrap();
    let result = &json["result"];
    assert_eq!(result["id"], "mock-task-1", "should include mock task ID");
    assert_eq!(result["contextId"], "mock-context-1", "should include mock context ID");
    assert_eq!(result["status"]["state"], "completed", "should have completed status");
}

#[test]
fn a2a_mock_send_streaming_message_returns_sse() {
    let server = start_a2a_mock_server();

    let body = r#"{"jsonrpc":"2.0","id":2,"method":"SendStreamingMessage","params":{"message":{"role":"user","parts":[{"kind":"text","text":"hello"}]}}}"#;
    let raw = json_post_raw(&server.endpoint(), server.path(), body, &[]);

    let status = parse_status(&raw);
    assert_eq!(status, 200, "SendStreamingMessage should return 200");

    let content_type = parse_header(&raw, "Content-Type");
    assert_eq!(
        content_type.as_deref(),
        Some("text/event-stream"),
        "should return SSE content type"
    );

    let resp_body = parse_body(&raw);
    assert!(resp_body.starts_with("data: "), "SSE body should start with 'data: '");

    let sse_data = resp_body.trim_start_matches("data: ").trim();
    let json: Value = serde_json::from_str(sse_data).unwrap();
    assert_eq!(json["result"]["final"], true, "SSE event should include final: true");
    assert_eq!(
        json["result"]["status"]["state"], "completed",
        "SSE event should be completed"
    );
}

#[test]
fn a2a_mock_records_version_header() {
    let server = start_a2a_mock_server();

    let body = r#"{"jsonrpc":"2.0","id":3,"method":"SendMessage","params":{"message":{"role":"user","parts":[{"kind":"text","text":"hello"}]}}}"#;
    let headers = [("A2A-Version", "0.3.0")];
    let _raw = json_post_raw(&server.endpoint(), server.path(), body, &headers);

    assert_eq!(
        server.last_a2a_version().as_deref(),
        Some("0.3.0"),
        "should record A2A-Version header"
    );
}

#[test]
fn a2a_mock_query_path_matches() {
    let server = start_a2a_mock_server();

    let body = r#"{"jsonrpc":"2.0","id":4,"method":"SendMessage","params":{"message":{"role":"user","parts":[{"kind":"text","text":"hello"}]}}}"#;
    let path_with_query = format!("{}?trace_id=abc", server.path());
    let raw = json_post_raw(&server.endpoint(), &path_with_query, body, &[]);

    let status = parse_status(&raw);
    assert_eq!(status, 200, "query string should not prevent path matching");
}

#[test]
fn a2a_mock_wrong_path_returns_404() {
    let server = start_a2a_mock_server();

    let body = r#"{"jsonrpc":"2.0","id":5,"method":"SendMessage","params":{}}"#;
    let raw = json_post_raw(&server.endpoint(), "/wrong", body, &[]);

    let status = parse_status(&raw);
    assert_eq!(status, 404, "wrong path should return 404");
}

#[test]
fn a2a_mock_cancel_task_returns_canceled() {
    let server = start_a2a_mock_server();

    let body = r#"{"jsonrpc":"2.0","id":6,"method":"CancelTask","params":{"id":"mock-task-1"}}"#;
    let raw = json_post_raw(&server.endpoint(), server.path(), body, &[]);

    let status = parse_status(&raw);
    assert_eq!(status, 200, "CancelTask should return 200");

    let json: Value = serde_json::from_str(&parse_body(&raw)).unwrap();
    assert_eq!(
        json["result"]["status"]["state"], "canceled",
        "should return canceled status"
    );
}

#[test]
fn a2a_mock_custom_config() {
    let config = A2aMockConfig {
        context_id: "ctx-42".to_owned(),
        path: "/custom-a2a".to_owned(),
        task_id: "task-42".to_owned(),
    };
    let server = start_a2a_mock_server_with_config(config);

    let body = r#"{"jsonrpc":"2.0","id":7,"method":"SendMessage","params":{"message":{"role":"user","parts":[{"kind":"text","text":"hello"}]}}}"#;
    let raw = json_post_raw(&server.endpoint(), "/custom-a2a", body, &[]);

    let status = parse_status(&raw);
    assert_eq!(status, 200, "custom path should work");

    let json: Value = serde_json::from_str(&parse_body(&raw)).unwrap();
    assert_eq!(json["result"]["id"], "task-42", "should use custom task ID");
    assert_eq!(json["result"]["contextId"], "ctx-42", "should use custom context ID");
}

#[test]
fn a2a_mock_legacy_method_aliases() {
    let server = start_a2a_mock_server();

    let body = r#"{"jsonrpc":"2.0","id":1,"method":"message/send","params":{"message":{"role":"user","parts":[{"kind":"text","text":"hello"}]}}}"#;
    let raw = json_post_raw(&server.endpoint(), server.path(), body, &[]);

    let status = parse_status(&raw);
    assert_eq!(status, 200, "legacy message/send should be accepted");

    let json: Value = serde_json::from_str(&parse_body(&raw)).unwrap();
    assert_eq!(
        json["result"]["status"]["state"], "completed",
        "legacy alias should work"
    );
}

#[test]
fn a2a_mock_get_method_rejected() {
    let server = start_a2a_mock_server();

    let req = format!(
        "GET {} HTTP/1.1\r\n\
         Host: localhost\r\n\
         Connection: close\r\n\r\n",
        server.path()
    );
    let raw = http_send(&server.endpoint(), &req);

    let status = parse_status(&raw);
    assert_eq!(status, 405, "GET should be rejected with 405");
}

#[test]
fn a2a_mock_put_with_body_rejected() {
    let server = start_a2a_mock_server();

    let body = r#"{"jsonrpc":"2.0","id":1,"method":"SendMessage","params":{}}"#;
    let req = format!(
        "PUT {} HTTP/1.1\r\n\
         Host: localhost\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\r\n\
         {body}",
        server.path(),
        body.len()
    );
    let raw = http_send(&server.endpoint(), &req);

    let status = parse_status(&raw);
    assert_eq!(status, 405, "PUT should be rejected with 405");
}

#[test]
fn a2a_mock_get_task_returns_completed() {
    let server = start_a2a_mock_server();

    let body = r#"{"jsonrpc":"2.0","id":1,"method":"GetTask","params":{"id":"mock-task-1"}}"#;
    let raw = json_post_raw(&server.endpoint(), server.path(), body, &[]);

    let status = parse_status(&raw);
    assert_eq!(status, 200, "GetTask should return 200");

    let json: Value = serde_json::from_str(&parse_body(&raw)).unwrap();
    assert_eq!(
        json["result"]["status"]["state"], "completed",
        "GetTask should return completed status"
    );
    assert_eq!(
        server.method_count("GetTask"),
        1,
        "should record exactly one GetTask call"
    );
}

#[test]
fn a2a_mock_legacy_tasks_get_alias() {
    let server = start_a2a_mock_server();

    let body = r#"{"jsonrpc":"2.0","id":1,"method":"tasks/get","params":{"id":"mock-task-1"}}"#;
    let raw = json_post_raw(&server.endpoint(), server.path(), body, &[]);

    let status = parse_status(&raw);
    assert_eq!(status, 200, "legacy tasks/get should be accepted");

    let json: Value = serde_json::from_str(&parse_body(&raw)).unwrap();
    assert_eq!(
        json["result"]["status"]["state"], "completed",
        "legacy alias should return completed"
    );

    assert_eq!(
        server.method_count("tasks/get"),
        1,
        "raw method name should be recorded, not canonical form"
    );
    assert_eq!(server.method_count("GetTask"), 0, "alias should not be canonicalized");
}

// -----------------------------------------------------------------------------
// Config Validation Tests
// -----------------------------------------------------------------------------

#[test]
#[should_panic(expected = "must start with '/'")]
fn mcp_mock_config_path_missing_slash_panics() {
    let config = McpMockConfig {
        path: "mcp".to_owned(),
        ..McpMockConfig::default()
    };
    start_mcp_mock_server_with_config(config);
}

#[test]
#[should_panic(expected = "must not contain a query string")]
fn mcp_mock_config_path_with_query_panics() {
    let config = McpMockConfig {
        path: "/mcp?x=1".to_owned(),
        ..McpMockConfig::default()
    };
    start_mcp_mock_server_with_config(config);
}

#[test]
#[should_panic(expected = "must not contain a fragment")]
fn mcp_mock_config_path_with_fragment_panics() {
    let config = McpMockConfig {
        path: "/mcp#frag".to_owned(),
        ..McpMockConfig::default()
    };
    start_mcp_mock_server_with_config(config);
}

#[test]
#[should_panic(expected = "must not contain an authority")]
fn mcp_mock_config_path_with_authority_panics() {
    let config = McpMockConfig {
        path: "//example.com/mcp".to_owned(),
        ..McpMockConfig::default()
    };
    start_mcp_mock_server_with_config(config);
}

#[test]
#[should_panic(expected = "must start with '/'")]
fn a2a_mock_config_path_missing_slash_panics() {
    let config = A2aMockConfig {
        path: "a2a".to_owned(),
        ..A2aMockConfig::default()
    };
    start_a2a_mock_server_with_config(config);
}

#[test]
#[should_panic(expected = "must not contain a query string")]
fn a2a_mock_config_path_with_query_panics() {
    let config = A2aMockConfig {
        path: "/a2a?x=1".to_owned(),
        ..A2aMockConfig::default()
    };
    start_a2a_mock_server_with_config(config);
}

#[test]
#[should_panic(expected = "must not contain a fragment")]
fn a2a_mock_config_path_with_fragment_panics() {
    let config = A2aMockConfig {
        path: "/a2a#frag".to_owned(),
        ..A2aMockConfig::default()
    };
    start_a2a_mock_server_with_config(config);
}

#[test]
#[should_panic(expected = "must not contain an authority")]
fn a2a_mock_config_path_with_authority_panics() {
    let config = A2aMockConfig {
        path: "//example.com/a2a".to_owned(),
        ..A2aMockConfig::default()
    };
    start_a2a_mock_server_with_config(config);
}

#[test]
#[should_panic(expected = "must not contain whitespace")]
fn mcp_mock_config_path_with_whitespace_panics() {
    let config = McpMockConfig {
        path: "/bad path".to_owned(),
        ..McpMockConfig::default()
    };
    start_mcp_mock_server_with_config(config);
}

#[test]
#[should_panic(expected = "inputSchema must be a JSON object")]
fn mcp_mock_non_object_schema_panics() {
    let _fixture = McpToolFixture::new("bad").with_input_schema(serde_json::json!("not-an-object"));
}

#[test]
#[should_panic(expected = "must have \"type\": \"object\"")]
fn mcp_mock_wrong_type_schema_panics() {
    let _fixture = McpToolFixture::new("bad").with_input_schema(serde_json::json!({"type": "string"}));
}

#[test]
#[should_panic(expected = "must have \"type\": \"object\"")]
fn mcp_mock_direct_field_schema_validated_at_startup() {
    let bad_tool = McpToolFixture {
        description: None,
        input_schema: serde_json::json!({"type": "string"}),
        name: "bad".to_owned(),
    };
    let config = McpMockConfig {
        tools: vec![bad_tool],
        ..McpMockConfig::default()
    };
    start_mcp_mock_server_with_config(config);
}

// -----------------------------------------------------------------------------
// HTTP Parser Tests
// -----------------------------------------------------------------------------

#[test]
fn mcp_mock_trailing_bytes_ignored() {
    let server = start_mcp_mock_server();

    let json_body = r#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#;
    let trailing = "GARBAGE_TRAILING_DATA";
    let req = format!(
        "POST {} HTTP/1.1\r\n\
         Host: localhost\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\r\n\
         {json_body}{trailing}",
        server.path(),
        json_body.len()
    );
    let raw = http_send(&server.endpoint(), &req);

    let status = parse_status(&raw);
    assert_eq!(
        status, 200,
        "should parse body using Content-Length, ignoring trailing bytes"
    );

    let reqs = server.received_requests();
    let last = reqs.last().expect("should have at least one request");
    assert_eq!(
        last.body, json_body,
        "recorded body should be exactly Content-Length bytes"
    );
}

// -----------------------------------------------------------------------------
// Behavioral Edge-Case Tests
// -----------------------------------------------------------------------------

#[test]
fn mcp_mock_stateless_omits_session_header() {
    let config = McpMockConfig {
        stateful_sessions: false,
        ..McpMockConfig::default()
    };
    let server = start_mcp_mock_server_with_config(config);

    let body = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#;
    let raw = json_post_raw(&server.endpoint(), server.path(), body, &[]);

    let status = parse_status(&raw);
    assert_eq!(status, 200, "initialize should return 200");

    let session_id = parse_header(&raw, "MCP-Session-Id");
    assert!(session_id.is_none(), "stateless server should not emit MCP-Session-Id");
}

#[test]
fn mcp_mock_tool_call_count_tracks_per_tool() {
    let config = McpMockConfig {
        tools: vec![McpToolFixture::new("get_weather"), McpToolFixture::new("create_event")],
        ..McpMockConfig::default()
    };
    let server = start_mcp_mock_server_with_config(config);

    let body = r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"get_weather","arguments":{}}}"#;
    let _raw = json_post_raw(&server.endpoint(), server.path(), body, &[]);

    assert_eq!(
        server.tool_call_count("get_weather"),
        1,
        "should count one get_weather call"
    );
    assert_eq!(
        server.tool_call_count("create_event"),
        0,
        "should count zero create_event calls"
    );
}

#[test]
fn a2a_mock_unknown_method_returns_error() {
    let server = start_a2a_mock_server();

    let body = r#"{"jsonrpc":"2.0","id":1,"method":"BogusMethod","params":{}}"#;
    let raw = json_post_raw(&server.endpoint(), server.path(), body, &[]);

    let status = parse_status(&raw);
    assert_eq!(status, 200, "unknown A2A method should return HTTP 200");

    let json: Value = serde_json::from_str(&parse_body(&raw)).unwrap();
    assert_eq!(
        json["error"]["code"], -32601,
        "unknown A2A method should return JSON-RPC -32601"
    );
}

#[test]
fn mcp_mock_content_length_zero_records_empty_body() {
    let server = start_mcp_mock_server();

    let req = format!(
        "POST {} HTTP/1.1\r\n\
         Host: localhost\r\n\
         Content-Type: application/json\r\n\
         Content-Length: 0\r\n\
         Connection: close\r\n\r\n\
         {{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}}",
        server.path()
    );
    let raw = http_send(&server.endpoint(), &req);

    let status = parse_status(&raw);
    assert_eq!(status, 400, "Content-Length: 0 should result in empty body and 400");

    let reqs = server.received_requests();
    let last = reqs.last().expect("should have at least one request");
    assert!(
        last.body.is_empty(),
        "recorded body should be empty when Content-Length is 0"
    );
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

fn json_post_raw(addr: &str, path: &str, body: &str, extra_headers: &[(&str, &str)]) -> String {
    let mut req = format!(
        "POST {path} HTTP/1.1\r\n\
         Host: localhost\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n",
        body.len()
    );

    for (name, value) in extra_headers {
        req.push_str(&format!("{name}: {value}\r\n"));
    }

    req.push_str("Connection: close\r\n\r\n");
    req.push_str(body);

    http_send(addr, &req)
}
