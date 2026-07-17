// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Integration tests for the MCP client (`list_tools` and `call_tool`)
//! against a real rmcp-based MCP server.


use std::{sync::Arc, time::Duration};

use praxis_ai_apis::mcp_client::{call_tool, list_tools};
use rmcp::{
    ServerHandler,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{ServerCapabilities, ServerInfo},
    tool, tool_handler, tool_router,
    transport::streamable_http_server::{
        StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
    },
};
use schemars::JsonSchema;
use serde::Deserialize;
use tokio_util::sync::CancellationToken;

// -----------------------------------------------------------------------------
// Test MCP Server
// -----------------------------------------------------------------------------

#[derive(Debug, Deserialize, JsonSchema)]
struct EchoRequest {
    #[schemars(description = "The message to echo back")]
    message: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct AddRequest {
    #[schemars(description = "First operand")]
    a: i32,
    #[schemars(description = "Second operand")]
    b: i32,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct FailRequest {
    #[schemars(description = "The error message to return")]
    message: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct SlowRequest {
    #[schemars(description = "Milliseconds to sleep before responding")]
    sleep_ms: u64,
}

#[derive(Debug, Clone)]
struct TestMcpServer {
    tool_router: ToolRouter<Self>,
}

#[expect(clippy::unused_self, reason = "rmcp macro-generated code")]
#[tool_router]
impl TestMcpServer {
    fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }

    #[tool(description = "Echo the input message back verbatim")]
    fn echo(&self, Parameters(req): Parameters<EchoRequest>) -> String {
        req.message
    }

    #[tool(description = "Add two integers and return the sum")]
    fn add(&self, Parameters(req): Parameters<AddRequest>) -> String {
        (req.a + req.b).to_string()
    }

    #[tool(description = "Always returns an error with the given message")]
    fn fail(&self, Parameters(req): Parameters<FailRequest>) -> Result<String, String> {
        Err(req.message)
    }

    #[tool(description = "Sleep for the specified duration then return")]
    async fn slow(&self, Parameters(req): Parameters<SlowRequest>) -> String {
        tokio::time::sleep(Duration::from_millis(req.sleep_ms)).await;
        "done".to_owned()
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for TestMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_instructions("Test MCP server for integration tests")
    }
}

// -----------------------------------------------------------------------------
// Server Helper
// -----------------------------------------------------------------------------

async fn start_test_mcp_server() -> (String, CancellationToken) {
    let ct = CancellationToken::new();
    let config = StreamableHttpServerConfig::default()
        .with_stateful_mode(false)
        .with_json_response(true)
        .with_sse_keep_alive(None)
        .with_cancellation_token(ct.child_token());

    let service: StreamableHttpService<TestMcpServer, LocalSessionManager> =
        StreamableHttpService::new(|| Ok(TestMcpServer::new()), Arc::default(), config);

    let router = axum::Router::new().nest_service("/mcp", service);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let shutdown = ct.clone();
    tokio::spawn(async move {
        drop(
            axum::serve(listener, router)
                .with_graceful_shutdown(async move { shutdown.cancelled_owned().await })
                .await,
        );
    });

    (format!("http://{addr}/mcp"), ct)
}

const TIMEOUT: Duration = Duration::from_secs(10);

// -----------------------------------------------------------------------------
// list_tools tests
// -----------------------------------------------------------------------------

#[tokio::test]
async fn list_tools_returns_all_tools() {
    let (url, ct) = start_test_mcp_server().await;
    let tools = list_tools(&url, None, None, TIMEOUT, 128, true).await.unwrap();
    ct.cancel();

    let names: Vec<&str> = tools
        .iter()
        .filter_map(|t| t.get("name").and_then(serde_json::Value::as_str))
        .collect();
    assert_eq!(names.len(), 4, "expected 4 tools, got: {names:?}");
    assert!(names.contains(&"echo"), "missing echo tool");
    assert!(names.contains(&"add"), "missing add tool");
    assert!(names.contains(&"fail"), "missing fail tool");
    assert!(names.contains(&"slow"), "missing slow tool");
}

#[tokio::test]
async fn list_tools_contains_expected_schema() {
    let (url, ct) = start_test_mcp_server().await;
    let tools = list_tools(&url, None, None, TIMEOUT, 128, true).await.unwrap();
    ct.cancel();

    let add_tool = tools
        .iter()
        .find(|t| t.get("name").and_then(serde_json::Value::as_str) == Some("add"))
        .expect("add tool should be present");

    assert!(
        add_tool.get("description").is_some(),
        "add tool should have a description"
    );

    let schema = add_tool.get("inputSchema").expect("add tool should have inputSchema");
    let props = schema
        .get("properties")
        .and_then(serde_json::Value::as_object)
        .expect("inputSchema should have properties");
    assert!(props.contains_key("a"), "schema should have property 'a'");
    assert!(props.contains_key("b"), "schema should have property 'b'");
}

#[tokio::test]
async fn list_tools_enforces_max_tools() {
    let (url, ct) = start_test_mcp_server().await;
    let result = list_tools(&url, None, None, TIMEOUT, 2, true).await;
    ct.cancel();

    let err = result.expect_err("should fail with TooManyTools");
    let msg = err.to_string();
    assert!(
        msg.contains("too many tools"),
        "error should mention too many tools: {msg}"
    );
}

#[tokio::test]
async fn list_tools_with_custom_headers() {
    let (url, ct) = start_test_mcp_server().await;
    let headers = serde_json::json!({"x-custom-header": "test-value"});
    let tools = list_tools(&url, Some(&headers), None, TIMEOUT, 128, true)
        .await
        .unwrap();
    ct.cancel();

    assert_eq!(tools.len(), 4, "should still return all 4 tools");
}

#[tokio::test]
async fn list_tools_with_authorization() {
    let (url, ct) = start_test_mcp_server().await;
    let tools = list_tools(&url, None, Some("test-token"), TIMEOUT, 128, true)
        .await
        .unwrap();
    ct.cancel();

    assert_eq!(tools.len(), 4, "should still return all 4 tools");
}

// -----------------------------------------------------------------------------
// call_tool tests
// -----------------------------------------------------------------------------

#[tokio::test]
async fn call_tool_echo() {
    let (url, ct) = start_test_mcp_server().await;
    let result = call_tool(
        &url,
        None,
        None,
        "echo",
        serde_json::json!({"message": "hello world"}),
        TIMEOUT,
        true,
    )
    .await
    .unwrap();
    ct.cancel();

    let text = result
        .content
        .first()
        .and_then(|c| c.as_text())
        .map(|t| t.text.as_str())
        .expect("expected text content");
    assert_eq!(text, "hello world", "echo should return the input message");
}

#[tokio::test]
async fn call_tool_add_with_arguments() {
    let (url, ct) = start_test_mcp_server().await;
    let result = call_tool(
        &url,
        None,
        None,
        "add",
        serde_json::json!({"a": 17, "b": 25}),
        TIMEOUT,
        true,
    )
    .await
    .unwrap();
    ct.cancel();

    let text = result
        .content
        .first()
        .and_then(|c| c.as_text())
        .map(|t| t.text.as_str())
        .expect("expected text content");
    assert_eq!(text, "42", "17 + 25 should be 42");
}

#[tokio::test]
async fn call_tool_add_with_string_arguments() {
    let (url, ct) = start_test_mcp_server().await;
    let result = call_tool(
        &url,
        None,
        None,
        "add",
        serde_json::Value::String(r#"{"a": 3, "b": 7}"#.to_owned()),
        TIMEOUT,
        true,
    )
    .await
    .unwrap();
    ct.cancel();

    let text = result
        .content
        .first()
        .and_then(|c| c.as_text())
        .map(|t| t.text.as_str())
        .expect("expected text content");
    assert_eq!(text, "10", "3 + 7 should be 10");
}

#[tokio::test]
async fn call_tool_error_returns_is_error() {
    let (url, ct) = start_test_mcp_server().await;
    let result = call_tool(
        &url,
        None,
        None,
        "fail",
        serde_json::json!({"message": "something broke"}),
        TIMEOUT,
        true,
    )
    .await
    .unwrap();
    ct.cancel();

    assert_eq!(result.is_error, Some(true), "fail tool should set is_error=true");
    let text = result
        .content
        .first()
        .and_then(|c| c.as_text())
        .map(|t| t.text.as_str())
        .expect("expected error text content");
    assert!(
        text.contains("something broke"),
        "error text should contain the message: {text}"
    );
}

#[tokio::test]
async fn call_tool_nonexistent_tool() {
    let (url, ct) = start_test_mcp_server().await;
    let result = call_tool(
        &url,
        None,
        None,
        "nonexistent_tool",
        serde_json::json!({}),
        TIMEOUT,
        true,
    )
    .await;
    ct.cancel();

    assert!(result.is_err(), "calling a nonexistent tool should fail");
}

#[tokio::test]
async fn call_tool_timeout() {
    let (url, ct) = start_test_mcp_server().await;
    let short_timeout = Duration::from_millis(200);
    let result = call_tool(
        &url,
        None,
        None,
        "slow",
        serde_json::json!({"sleep_ms": 5000}),
        short_timeout,
        true,
    )
    .await;
    ct.cancel();

    let err = result.expect_err("should time out");
    let msg = err.to_string();
    assert!(msg.contains("timed out"), "error should mention timeout: {msg}");
}
