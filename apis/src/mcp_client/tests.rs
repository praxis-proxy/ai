// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Unit tests for the MCP client wrapper.

use std::time::Duration;

use super::*;

const TEST_TIMEOUT: Duration = Duration::from_secs(5);

async fn validate_url(url: &str) -> Result<(), McpClientError> {
    validate_mcp_url(url, TEST_TIMEOUT, false).await
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
        url: "http://example.com/mcp".to_owned(),
        source: "refused".into(),
    };
    let msg = err.to_string();
    assert!(msg.contains("example.com"), "should include URL");
    assert!(msg.contains("connection failed"), "should describe failure");
}

#[test]
fn timeout_error_display() {
    let err = McpClientError::Timeout {
        url: "http://example.com/mcp".to_owned(),
        timeout: Duration::from_secs(5),
    };
    let msg = err.to_string();
    assert!(msg.contains("timed out"), "should describe timeout");
    assert!(msg.contains("5s"), "should include duration");
}

#[test]
fn too_many_tools_error_display() {
    let err = McpClientError::TooManyTools {
        url: "http://example.com/mcp".to_owned(),
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
        url: "http://example.com/mcp".to_owned(),
        source: "protocol error".into(),
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
    assert!(
        check_resolved_addrs(&addrs, "http://metadata.example/mcp", false).is_err(),
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
    let err = McpClientError::SsrfBlocked("http://127.0.0.1/mcp".to_owned());
    let msg = err.to_string();
    assert!(msg.contains("SSRF"), "should mention SSRF");
    assert!(msg.contains("127.0.0.1"), "should include the URL");
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
