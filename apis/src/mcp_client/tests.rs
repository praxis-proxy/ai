// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Unit tests for the MCP client wrapper.

use std::time::Duration;

use super::*;

const TEST_TIMEOUT: Duration = Duration::from_secs(5);

async fn validate_url(url: &str) -> Result<(), McpClientError> {
    validate_mcp_url(url, TEST_TIMEOUT, false).await
}

fn display_url(url: &str) -> McpDisplayUrl {
    McpDisplayUrl::from_uri(&url.parse().unwrap())
}

fn assert_error_uses_sanitized_url(error: &McpClientError) {
    let message = error.to_string();
    assert!(
        message.contains("https://example.com:8443/mcp/tools"),
        "error should retain the sanitized endpoint: {message}"
    );
    assert!(!message.contains("user"), "error must redact URL userinfo: {message}");
    assert!(!message.contains("pass"), "error must redact URL passwords: {message}");
    assert!(!message.contains("api_key"), "error must redact query names: {message}");
    assert!(
        !message.contains("TOPSECRET"),
        "error must redact query values: {message}"
    );
}

// =========================================================================
// Transport Config
// =========================================================================

#[test]
fn build_config_with_no_headers() {
    let config = build_transport_config("http://localhost:8001/mcp", None, None).unwrap();
    assert_eq!(&*config.uri, "http://localhost:8001/mcp", "URI should match");
    assert!(config.custom_headers.is_empty(), "no custom headers expected");
}

#[test]
fn build_config_with_headers() {
    let headers = serde_json::json!({"x-custom": "value", "x-other": "val2"});
    let config = build_transport_config("http://localhost:8001/mcp", Some(&headers), None).unwrap();

    assert_eq!(config.custom_headers.len(), 2, "should have 2 custom headers");
}

#[test]
fn build_config_ignores_non_string_header_values() {
    let headers = serde_json::json!({"x-good": "ok", "x-bad": 123});
    let config = build_transport_config("http://localhost:8001/mcp", Some(&headers), None).unwrap();

    assert_eq!(
        config.custom_headers.len(),
        1,
        "should only include string-valued headers"
    );
}

#[test]
fn build_config_ignores_non_object_headers() {
    let headers = serde_json::json!("not-an-object");
    let config = build_transport_config("http://localhost:8001/mcp", Some(&headers), None).unwrap();

    assert!(config.custom_headers.is_empty(), "non-object headers should be ignored");
}

// =========================================================================
// Hop-by-hop / framing header blocking
// =========================================================================

#[test]
fn hop_by_hop_headers_stripped_from_mcp_headers() {
    let headers = serde_json::json!({
        "host": "evil.example.com",
        "content-length": "999",
        "transfer-encoding": "chunked",
        "connection": "keep-alive",
        "te": "trailers",
        "trailer": "Foo",
        "upgrade": "websocket",
        "proxy-authorization": "Basic creds",
        "x-custom": "safe"
    });
    let config = build_transport_config("http://api.example.com/mcp", Some(&headers), None).unwrap();

    assert_eq!(config.custom_headers.len(), 1, "only safe header should remain");
    assert!(
        config
            .custom_headers
            .contains_key(&http::HeaderName::from_static("x-custom")),
        "x-custom should pass through"
    );
}

#[test]
fn reserved_internal_headers_stripped_from_mcp_headers() {
    let headers = serde_json::json!({
        "x-praxis-ai-format": "openai",
        "x-mcp-servername": "backend-1",
        "x-a2a-method": "task/send",
        "x-custom": "safe"
    });
    let config = build_transport_config("http://api.example.com/mcp", Some(&headers), None).unwrap();

    assert_eq!(config.custom_headers.len(), 1, "only safe header should remain");
    assert!(
        config
            .custom_headers
            .contains_key(&http::HeaderName::from_static("x-custom")),
        "x-custom should pass through"
    );
}

// =========================================================================
// Authorization
// =========================================================================

#[test]
fn authorization_injects_bearer_header() {
    let config = build_transport_config("http://api.example.com/mcp", None, Some("tok_abc")).unwrap();
    let auth = config.custom_headers.get(&http::header::AUTHORIZATION).unwrap();
    assert_eq!(auth, "Bearer tok_abc", "should inject Bearer token");
}

#[test]
fn authorization_with_custom_headers() {
    let headers = serde_json::json!({"x-custom": "val"});
    let config = build_transport_config("http://api.example.com/mcp", Some(&headers), Some("tok_xyz")).unwrap();

    assert_eq!(config.custom_headers.len(), 2, "should have both headers");
    assert_eq!(
        config.custom_headers.get(&http::header::AUTHORIZATION).unwrap(),
        "Bearer tok_xyz",
        "should have authorization"
    );
}

#[test]
fn authorization_field_overrides_headers_authorization() {
    let headers = serde_json::json!({"authorization": "Basic creds"});
    let config = build_transport_config("http://api.example.com/mcp", Some(&headers), Some("tok_real")).unwrap();

    let auth = config.custom_headers.get(&http::header::AUTHORIZATION).unwrap();
    assert_eq!(
        auth, "Bearer tok_real",
        "authorization field should win over headers.Authorization"
    );
}

#[test]
fn authorization_in_headers_stripped_when_no_field() {
    let headers = serde_json::json!({"authorization": "Basic creds", "x-custom": "val"});
    let config = build_transport_config("http://api.example.com/mcp", Some(&headers), None).unwrap();

    assert!(
        !config.custom_headers.contains_key(&http::header::AUTHORIZATION),
        "Authorization from headers should be stripped"
    );
    assert_eq!(config.custom_headers.len(), 1, "only x-custom should remain");
}

#[test]
fn no_authorization_no_header() {
    let config = build_transport_config("http://api.example.com/mcp", None, None).unwrap();
    assert!(config.custom_headers.is_empty(), "no headers expected");
}

#[test]
fn authorization_with_invalid_chars_returns_error() {
    let result = build_transport_config("http://api.example.com/mcp", None, Some("tok\x00bad"));
    assert!(result.is_err(), "invalid header chars should return error");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("invalid HTTP header"),
        "error should describe invalid header: {msg}"
    );
}

// =========================================================================
// Error Display
// =========================================================================

#[test]
fn connection_error_display() {
    let err = McpClientError::Connection {
        url: display_url("http://example.com/mcp"),
    };
    let msg = err.to_string();
    assert!(msg.contains("example.com"), "should include URL");
    assert!(msg.contains("connection failed"), "should describe failure");
}

#[test]
fn timeout_error_display() {
    let err = McpClientError::Timeout {
        url: display_url("http://example.com/mcp"),
        timeout: Duration::from_secs(5),
    };
    let msg = err.to_string();
    assert!(msg.contains("timed out"), "should describe timeout");
    assert!(msg.contains("5s"), "should include duration");
}

#[test]
fn too_many_tools_error_display() {
    let err = McpClientError::TooManyTools {
        url: display_url("http://example.com/mcp"),
        count: 200,
        max: 128,
    };
    let msg = err.to_string();
    assert!(msg.contains("200"), "should include actual count");
    assert!(msg.contains("128"), "should include max limit");
}

#[test]
fn list_tools_error_display() {
    let err = McpClientError::ListTools {
        url: display_url("http://example.com/mcp"),
    };
    let msg = err.to_string();
    assert!(msg.contains("tools/list failed"), "should describe failure");
    assert!(msg.contains("example.com"), "should include URL");
}

#[test]
fn invalid_authorization_error_display() {
    let err = McpClientError::InvalidAuthorization;
    let msg = err.to_string();
    assert!(
        msg.contains("invalid HTTP header"),
        "should describe invalid header: {msg}"
    );
}

#[test]
fn display_url_retains_only_routable_components() {
    let display = display_url("https://user:pass@example.com:8443/mcp/tools?api_key=TOPSECRET");
    assert_eq!(display.to_string(), "https://example.com:8443/mcp/tools");
}

#[test]
fn display_url_preserves_ipv6_authority() {
    let display = display_url("https://user:pass@[2001:db8::1]:8443/mcp?api_key=TOPSECRET");
    assert_eq!(display.to_string(), "https://[2001:db8::1]:8443/mcp");
}

#[test]
fn url_bearing_errors_only_format_sanitized_urls() {
    let url = display_url("https://user:pass@example.com:8443/mcp/tools?api_key=TOPSECRET");
    let errors = [
        McpClientError::Connection { url: url.clone() },
        McpClientError::ListTools { url: url.clone() },
        McpClientError::Timeout {
            url: url.clone(),
            timeout: Duration::from_secs(5),
        },
        McpClientError::TooManyTools {
            url: url.clone(),
            count: 200,
            max: 128,
        },
        McpClientError::SsrfBlocked {
            url,
            reason: "test reason",
        },
    ];

    for error in &errors {
        assert_error_uses_sanitized_url(error);
    }
}

// =========================================================================
// SSRF Validation
// =========================================================================

#[tokio::test]
async fn ssrf_blocks_ipv4_loopback() {
    assert!(validate_url("http://127.0.0.1/mcp").await.is_err());
    assert!(validate_url("http://127.0.0.99:8080/mcp").await.is_err());
}

#[tokio::test]
async fn ssrf_blocks_ipv6_loopback() {
    assert!(validate_url("http://[::1]/mcp").await.is_err());
}

#[tokio::test]
async fn ssrf_blocks_ipv6_link_local() {
    assert!(validate_url("http://[fe80::1]/mcp").await.is_err());
    assert!(validate_url("http://[fe80::1%25eth0]:8080/mcp").await.is_err());
}

#[tokio::test]
async fn ssrf_blocks_localhost_hostname() {
    assert!(validate_url("http://localhost/mcp").await.is_err());
    assert!(validate_url("http://LOCALHOST/mcp").await.is_err());
    assert!(validate_url("http://sub.localhost/mcp").await.is_err());
}

#[tokio::test]
async fn ssrf_blocks_link_local() {
    assert!(validate_url("http://169.254.169.254/latest/meta-data/").await.is_err());
    assert!(validate_url("http://169.254.0.1/mcp").await.is_err());
}

#[tokio::test]
async fn ssrf_blocks_alibaba_cloud_metadata() {
    assert!(
        validate_url("http://100.100.100.200/latest/meta-data/instance-id")
            .await
            .is_err()
    );
}

#[tokio::test]
async fn ssrf_blocks_mapped_ipv4_loopback() {
    assert!(validate_url("http://[::ffff:127.0.0.1]/mcp").await.is_err());
}

#[tokio::test]
async fn ssrf_blocks_mapped_metadata() {
    assert!(validate_url("http://[::ffff:169.254.169.254]/mcp").await.is_err());
}

#[tokio::test]
async fn ssrf_blocks_mapped_alibaba_cloud_metadata() {
    assert!(
        validate_url("http://[::ffff:100.100.100.200]/latest/meta-data/instance-id")
            .await
            .is_err()
    );
}

#[test]
fn ssrf_blocks_dns_resolved_alibaba_cloud_metadata() {
    let addrs = ["100.100.100.200:80".parse::<SocketAddr>().unwrap()];
    let url = display_url("http://metadata.example/mcp");
    assert!(
        check_resolved_addrs(&addrs, &url, false).is_err(),
        "DNS-resolved Alibaba Cloud metadata address should be blocked"
    );
}

#[tokio::test]
async fn ssrf_blocks_invalid_url() {
    assert!(validate_url("not-a-url").await.is_err());
}

#[tokio::test]
async fn ssrf_blocks_unresolvable_hostname() {
    assert!(validate_url("http://unresolvable.invalid/mcp").await.is_err());
}

#[tokio::test]
async fn ssrf_errors_redact_sensitive_url_components() {
    let urls = [
        "http://unresolvable.invalid/mcp?api_key=TOPSECRET",
        "http://127.0.0.1/mcp?api_key=TOPSECRET",
        "http://127.0.0.1/mcp?api_key=TOPSECRET#FRAGMENTSECRET",
    ];

    for url in urls {
        let message = validate_url(url).await.unwrap_err().to_string();
        assert!(
            !message.contains("TOPSECRET"),
            "error must redact URL query credentials: {message}"
        );
        assert!(
            !message.contains("FRAGMENTSECRET"),
            "error must redact URL fragments: {message}"
        );
    }
}

#[tokio::test]
async fn invalid_url_errors_use_opaque_display_value() {
    let urls = [
        "http://exa mple.com/mcp?api_key=TOPSECRET#FRAGMENTSECRET",
        "//user:pass@example.com/mcp?api_key=TOPSECRET#FRAGMENTSECRET",
        "ftp://user:pass@example.com/mcp?api_key=TOPSECRET#FRAGMENTSECRET",
    ];

    for url in urls {
        let message = validate_url(url).await.unwrap_err().to_string();
        assert!(
            message.contains("<invalid MCP URL>"),
            "invalid URL should be opaque: {message}"
        );
        assert!(!message.contains("user"), "invalid URL must redact userinfo: {message}");
        assert!(
            !message.contains("pass"),
            "invalid URL must redact passwords: {message}"
        );
        assert!(
            !message.contains("TOPSECRET"),
            "invalid URL must redact query values: {message}"
        );
        assert!(
            !message.contains("FRAGMENTSECRET"),
            "invalid URL must redact fragments: {message}"
        );
    }
}

#[tokio::test]
async fn ssrf_errors_include_actionable_reason() {
    let cases = [
        ("ftp://example.com/mcp", "scheme must be http or https"),
        (
            "http://user:pass@example.com/mcp",
            "embedded credentials are not allowed",
        ),
        ("http://localhost/mcp", "localhost hostnames are not allowed"),
        (
            "http://127.0.0.1/mcp",
            "address is loopback, link-local, unspecified, or cloud metadata",
        ),
    ];

    for (url, expected_reason) in cases {
        let message = validate_url(url).await.unwrap_err().to_string();
        assert!(
            message.contains(expected_reason),
            "error should explain how to fix the blocked URL: {message}"
        );
    }
}

#[tokio::test]
async fn ssrf_allows_public_ips() {
    assert!(validate_url("http://8.8.8.8/mcp").await.is_ok());
    assert!(validate_url("https://1.1.1.1:443/v1").await.is_ok());
}

#[tokio::test]
async fn ssrf_allows_private_rfc1918() {
    assert!(validate_url("http://10.0.0.5/mcp").await.is_ok());
    assert!(validate_url("http://192.168.1.100/mcp").await.is_ok());
}

#[test]
fn ssrf_error_display() {
    let err = McpClientError::SsrfBlocked {
        url: display_url("http://127.0.0.1/mcp"),
        reason: "loopback address is not allowed",
    };
    let msg = err.to_string();
    assert!(msg.contains("SSRF"), "should mention SSRF");
    assert!(msg.contains("127.0.0.1"), "should include the URL");
    assert!(msg.contains("loopback address"), "should include the reason");
}

#[tokio::test]
async fn ssrf_blocks_unspecified_ipv4() {
    assert!(validate_url("http://0.0.0.0/mcp").await.is_err());
}

#[tokio::test]
async fn ssrf_blocks_unspecified_ipv6() {
    assert!(validate_url("http://[::]/mcp").await.is_err());
}

#[tokio::test]
async fn ssrf_blocks_mapped_unspecified() {
    assert!(validate_url("http://[::ffff:0.0.0.0]/mcp").await.is_err());
}

#[tokio::test]
async fn ssrf_blocks_url_with_userinfo() {
    let err = validate_url("http://user:pass@example.com/mcp").await.unwrap_err();
    let msg = err.to_string();
    assert!(!msg.contains("pass"), "error must not leak credentials");
    assert!(validate_url("https://user@example.com/mcp").await.is_err());

    let ipv6_message = validate_url("http://user:pass@[::1]:8080/mcp?api_key=TOPSECRET")
        .await
        .unwrap_err()
        .to_string();
    assert!(!ipv6_message.contains("user"), "IPv6 error must redact URL userinfo");
    assert!(!ipv6_message.contains("pass"), "IPv6 error must redact URL passwords");
    assert!(
        !ipv6_message.contains("TOPSECRET"),
        "IPv6 error must redact URL queries"
    );
}

#[tokio::test]
async fn ssrf_blocks_aws_imds_ipv6() {
    assert!(validate_url("http://[fd00:ec2::254]/latest/meta-data/").await.is_err());
}

#[test]
fn aws_imds_v6_detected_by_is_ssrf_sensitive() {
    let ip = "fd00:ec2::254".parse::<IpAddr>().unwrap();
    assert!(is_ssrf_sensitive(&ip), "fd00:ec2::254 should be SSRF-sensitive");
}

#[test]
fn unspecified_ip_detected_by_is_ssrf_sensitive() {
    let v4 = "0.0.0.0".parse::<IpAddr>().unwrap();
    assert!(is_ssrf_sensitive(&v4), "0.0.0.0 should be SSRF-sensitive");
    let v6 = "::".parse::<IpAddr>().unwrap();
    assert!(is_ssrf_sensitive(&v6), ":: should be SSRF-sensitive");
}

#[test]
fn no_authorization_field_injects_no_auth_header() {
    let headers = serde_json::json!({"x-custom": "val"});
    let config = build_transport_config("http://api.example.com/mcp", Some(&headers), None).unwrap();
    assert!(
        !config.custom_headers.contains_key(&http::header::AUTHORIZATION),
        "should not inject Authorization when authorization field is absent"
    );
}

#[test]
fn ipv6_link_local_detected_by_is_ssrf_sensitive() {
    let fe80 = "fe80::1".parse::<IpAddr>().unwrap();
    assert!(is_ssrf_sensitive(&fe80), "fe80::1 should be SSRF-sensitive");
    let febf = "febf::1".parse::<IpAddr>().unwrap();
    assert!(is_ssrf_sensitive(&febf), "febf::1 should be SSRF-sensitive");
    let fe00 = "fe00::1".parse::<IpAddr>().unwrap();
    assert!(!is_ssrf_sensitive(&fe00), "fe00::1 is not link-local");
}

// =========================================================================
// allow_loopback
// =========================================================================

#[tokio::test]
async fn allow_loopback_permits_ipv4_loopback() {
    assert!(
        validate_mcp_url("http://127.0.0.1/mcp", TEST_TIMEOUT, true)
            .await
            .is_ok()
    );
}

#[tokio::test]
async fn allow_loopback_permits_localhost_hostname() {
    assert!(
        validate_mcp_url("http://localhost/mcp", TEST_TIMEOUT, true)
            .await
            .is_ok()
    );
}

#[tokio::test]
async fn allow_loopback_still_blocks_link_local() {
    assert!(
        validate_mcp_url("http://169.254.169.254/mcp", TEST_TIMEOUT, true)
            .await
            .is_err()
    );
}

#[tokio::test]
async fn allow_loopback_still_blocks_unspecified() {
    assert!(
        validate_mcp_url("http://0.0.0.0/mcp", TEST_TIMEOUT, true)
            .await
            .is_err()
    );
}

// =========================================================================
// CallTool Error
// =========================================================================

#[test]
fn call_tool_error_display() {
    let err = McpClientError::CallTool {
        url: "http://example.com/mcp".to_owned(),
        tool_name: "get_weather".to_owned(),
        source: Box::new(std::io::Error::new(std::io::ErrorKind::ConnectionRefused, "refused")),
    };
    let msg = err.to_string();
    assert!(msg.contains("tools/call failed"), "should mention tools/call: {msg}");
    assert!(msg.contains("get_weather"), "should mention tool name: {msg}");
    assert!(msg.contains("example.com"), "should mention URL: {msg}");
}

// =========================================================================
// Integration tests (real rmcp MCP server)
// =========================================================================

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

async fn start_test_mcp_server() -> (String, tokio_util::sync::CancellationToken) {
    let ct = tokio_util::sync::CancellationToken::new();
    let config = StreamableHttpServerConfig::default()
        .with_stateful_mode(false)
        .with_json_response(true)
        .with_sse_keep_alive(None)
        .with_cancellation_token(ct.child_token());

    let service: StreamableHttpService<TestMcpServer, LocalSessionManager> =
        StreamableHttpService::new(|| Ok(TestMcpServer::new()), std::sync::Arc::default(), config);

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

const INTEGRATION_TIMEOUT: Duration = Duration::from_secs(10);

#[tokio::test]
async fn list_tools_returns_all_tools() {
    let (url, ct) = start_test_mcp_server().await;
    let tools = list_tools(&url, None, None, INTEGRATION_TIMEOUT, 128, true)
        .await
        .unwrap();
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
    let tools = list_tools(&url, None, None, INTEGRATION_TIMEOUT, 128, true)
        .await
        .unwrap();
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
    let result = list_tools(&url, None, None, INTEGRATION_TIMEOUT, 2, true).await;
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
    let tools = list_tools(&url, Some(&headers), None, INTEGRATION_TIMEOUT, 128, true)
        .await
        .unwrap();
    ct.cancel();

    assert_eq!(tools.len(), 4, "should still return all 4 tools");
}

#[tokio::test]
async fn list_tools_with_authorization() {
    let (url, ct) = start_test_mcp_server().await;
    let tools = list_tools(&url, None, Some("test-token"), INTEGRATION_TIMEOUT, 128, true)
        .await
        .unwrap();
    ct.cancel();

    assert_eq!(tools.len(), 4, "should still return all 4 tools");
}

#[tokio::test]
async fn call_tool_echo() {
    let (url, ct) = start_test_mcp_server().await;
    let result = call_tool(
        &url,
        None,
        None,
        "echo",
        serde_json::json!({"message": "hello world"}),
        INTEGRATION_TIMEOUT,
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
        INTEGRATION_TIMEOUT,
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
        INTEGRATION_TIMEOUT,
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
        INTEGRATION_TIMEOUT,
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
        INTEGRATION_TIMEOUT,
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
