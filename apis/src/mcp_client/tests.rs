// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Unit tests for the MCP client wrapper.

use std::{
    sync::{
        Arc as StdArc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use super::{
    session_pool::{MAX_IDLE_PER_KEY, MAX_TOTAL_IDLE},
    *,
};

const TEST_TIMEOUT: Duration = Duration::from_secs(5);

async fn validate_url(url: &str) -> Result<(), McpClientError> {
    validate_mcp_target(url, TEST_TIMEOUT, false).await
}

fn display_url(url: &str) -> McpDisplayUrl {
    McpDisplayUrl::from_uri(&url.parse().unwrap())
}

fn assert_error_uses_sanitized_url(error: &McpClientError) {
    let message = error.to_string();
    assert!(
        message.contains("https://example.com:8443/mcp/tools"),
        "sanitized endpoint should survive in: {message}"
    );
    for secret in ["user", "pass", "api_key", "TOPSECRET"] {
        assert!(
            !message.contains(secret),
            "credential fragment {secret:?} leaked into error: {message}"
        );
    }
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
fn trusted_forwarded_headers_override_tool_entry_values() {
    let headers = serde_json::json!({"x-tenant-id": "spoofed", "x-tool": "kept"});
    let mut forwarded = http::HeaderMap::new();
    forwarded.insert("x-tenant-id", http::HeaderValue::from_static("tenant-a"));
    forwarded.insert(http::header::HOST, http::HeaderValue::from_static("evil.example"));

    let config = build_transport_config_with_forwarded_headers(
        "http://localhost:8001/mcp",
        Some(&headers),
        None,
        &[http::HeaderName::from_static("x-tenant-id")],
        Some(&forwarded),
        None,
    )
    .unwrap();

    assert_eq!(
        config
            .custom_headers
            .get(&http::HeaderName::from_static("x-tenant-id"))
            .unwrap(),
        "tenant-a"
    );
    assert_eq!(
        config
            .custom_headers
            .get(&http::HeaderName::from_static("x-tool"))
            .unwrap(),
        "kept"
    );
    assert!(!config.custom_headers.contains_key(&http::header::HOST));
}

#[test]
fn configured_forwarded_names_are_stripped_without_trusted_values() {
    let headers = serde_json::json!({"x-tenant-id": "spoofed", "x-tool": "kept"});

    let config = build_transport_config_with_forwarded_headers(
        "http://localhost:8001/mcp",
        Some(&headers),
        None,
        &[http::HeaderName::from_static("x-tenant-id")],
        None,
        None,
    )
    .unwrap();

    assert!(
        !config
            .custom_headers
            .contains_key(&http::HeaderName::from_static("x-tenant-id"))
    );
    assert_eq!(
        config
            .custom_headers
            .get(&http::HeaderName::from_static("x-tool"))
            .unwrap(),
        "kept"
    );
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
        "keep-alive": "timeout=5",
        "proxy-connection": "keep-alive",
        "te": "trailers",
        "trailer": "Foo",
        "upgrade": "websocket",
        "proxy-authorization": "Basic creds",
        "proxy-authenticate": "Basic realm=\"mcp\"",
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
fn keep_alive_and_proxy_connection_headers_stripped_from_mcp_headers() {
    let headers = serde_json::json!({
        "keep-alive": "timeout=5",
        "proxy-connection": "keep-alive",
        "connection": "keep-alive",
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
    assert!(
        !config
            .custom_headers
            .contains_key(&http::HeaderName::from_static("keep-alive")),
        "keep-alive must not reach outbound MCP transport"
    );
    assert!(
        !config
            .custom_headers
            .contains_key(&http::HeaderName::from_static("proxy-connection")),
        "proxy-connection must not reach outbound MCP transport"
    );
    assert!(
        !config.custom_headers.contains_key(&http::header::CONNECTION),
        "connection must stay blocked"
    );
}

#[test]
fn connection_nominated_headers_stripped_from_mcp_headers() {
    let headers = serde_json::json!({
        "connection": "x-smuggle, Keep-Alive",
        "x-smuggle": "secret",
        "x-custom": "safe"
    });
    let config = build_transport_config("http://api.example.com/mcp", Some(&headers), None).unwrap();

    assert_eq!(config.custom_headers.len(), 1, "only safe header should remain");
    assert!(
        config
            .custom_headers
            .contains_key(&http::HeaderName::from_static("x-custom")),
        "x-custom is not listed in Connection and should pass through"
    );
    assert!(
        !config
            .custom_headers
            .contains_key(&http::HeaderName::from_static("x-smuggle")),
        "fields named by Connection must not reach outbound MCP transport"
    );
    assert!(
        !config.custom_headers.contains_key(&http::header::CONNECTION),
        "connection itself must stay blocked"
    );
}

#[test]
fn proxy_authenticate_stripped_from_mcp_headers() {
    let headers = serde_json::json!({
        "proxy-authenticate": "Basic realm=\"mcp\"",
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
    assert!(
        !config
            .custom_headers
            .contains_key(&http::HeaderName::from_static("proxy-authenticate")),
        "proxy-authenticate is hop-by-hop and must not reach outbound MCP transport"
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

#[test]
fn build_transport_config_bounds_retry_to_three() {
    let config =
        build_transport_config_with_forwarded_headers("https://mcp.example/mcp", None, None, &[], None, None).unwrap();
    // A policy consulted past its max returns None (no further retry).
    assert!(
        config.retry_config.retry(3).is_none(),
        "the 4th consecutive failed re-dial must not retry"
    );
    assert!(config.retry_config.retry(0).is_some(), "the first re-dial is allowed");
}

// =========================================================================
// Cookie and forwarded header blocking
// =========================================================================

#[test]
fn cookie_and_forwarded_headers_stripped_from_mcp_headers() {
    let headers = serde_json::json!({
        "cookie": "session=abc123",
        "set-cookie": "id=xyz; Path=/",
        "forwarded": "for=192.0.2.60;proto=http",
        "x-forwarded-for": "203.0.113.50",
        "x-forwarded-host": "original.example.com",
        "x-forwarded-proto": "https",
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

#[test]
fn connector_context_overrides_client_shadow_and_static_bearer() {
    let owner = StateOwner::from_trusted_parts("tenant-a", "issuer-a", "subject-a").unwrap();
    let bearer = SecretString::from("per-user-token");
    let assertion = SecretString::from("signed-assertion");
    let context = McpConnectorContext {
        owner: &owner,
        bearer: Some(&bearer),
        assertion: Some(&assertion),
    };
    let headers = serde_json::json!({
        "authorization": "Basic spoofed",
        "x-mcp-authorized": "client-spoofed",
        "x-custom": "kept"
    });
    let config = build_transport_config_with_forwarded_headers(
        "https://mcp.example/mcp",
        Some(&headers),
        Some("static-token"),
        &[],
        None,
        Some(&context),
    )
    .unwrap();
    assert_eq!(
        config.custom_headers.get(&http::header::AUTHORIZATION).unwrap(),
        "Bearer per-user-token"
    );
    assert_eq!(
        config
            .custom_headers
            .get(&crate::callout_credentials::MCP_AUTHORIZED_HEADER)
            .unwrap(),
        "signed-assertion"
    );
}

#[test]
fn rotated_assertion_injects_latest_value_without_changing_other_context() {
    let owner = StateOwner::from_trusted_parts("tenant-a", "issuer-a", "subject-a").unwrap();
    let bearer = SecretString::from("per-user-token");
    let first_assertion = SecretString::from("assertion-v1");
    let second_assertion = SecretString::from("assertion-v2");
    let first_context = McpConnectorContext {
        owner: &owner,
        bearer: Some(&bearer),
        assertion: Some(&first_assertion),
    };
    let second_context = McpConnectorContext {
        owner: &owner,
        bearer: Some(&bearer),
        assertion: Some(&second_assertion),
    };

    let first = build_transport_config_with_forwarded_headers(
        "https://mcp.example/mcp",
        None,
        None,
        &[],
        None,
        Some(&first_context),
    )
    .unwrap();
    let second = build_transport_config_with_forwarded_headers(
        "https://mcp.example/mcp",
        None,
        None,
        &[],
        None,
        Some(&second_context),
    )
    .unwrap();

    assert_eq!(
        first
            .custom_headers
            .get(&crate::callout_credentials::MCP_AUTHORIZED_HEADER)
            .unwrap(),
        "assertion-v1"
    );
    assert_eq!(
        second
            .custom_headers
            .get(&crate::callout_credentials::MCP_AUTHORIZED_HEADER)
            .unwrap(),
        "assertion-v2"
    );
    assert_eq!(
        second.custom_headers.get(&http::header::AUTHORIZATION).unwrap(),
        "Bearer per-user-token"
    );
}

#[test]
fn direct_url_context_none_never_forwards_client_assertion() {
    let headers = serde_json::json!({"x-mcp-authorized": "client-spoofed"});
    let config =
        build_transport_config_with_forwarded_headers("https://mcp.example/mcp", Some(&headers), None, &[], None, None)
            .unwrap();
    assert!(
        !config
            .custom_headers
            .contains_key(&crate::callout_credentials::MCP_AUTHORIZED_HEADER)
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
fn listing_too_large_error_display() {
    let err = McpClientError::ListingTooLarge {
        url: display_url("http://example.com/mcp"),
        bytes: 5 * 1024 * 1024,
        max: 4 * 1024 * 1024,
    };
    let msg = err.to_string();
    assert!(msg.contains("5242880"), "should include observed byte count");
    assert!(msg.contains("4194304"), "should include cumulative max");
    assert!(msg.contains("example.com"), "should include URL");
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
fn display_url_keeps_locators_and_drops_secrets() {
    // scheme + host + port + path survive; userinfo, query, and fragment do not.
    let cases = [
        (
            "https://user:pass@example.com:8443/mcp/tools?api_key=TOPSECRET#frag",
            "https://example.com:8443/mcp/tools",
        ),
        ("http://example.com/", "http://example.com/"),
        ("https://token@host.example/path", "https://host.example/path"),
    ];
    for (raw, expected) in cases {
        assert_eq!(display_url(raw).to_string(), expected, "input: {raw}");
    }
}

#[test]
fn display_url_brackets_ipv6_and_strips_credentials() {
    assert_eq!(
        display_url("https://user:pass@[2001:db8::1]:8443/mcp?api_key=TOPSECRET").to_string(),
        "https://[2001:db8::1]:8443/mcp",
    );
    // Bare IPv6 host, no port: brackets are still restored.
    assert_eq!(display_url("http://[fd00::5]/x").to_string(), "http://[fd00::5]/x");
}

#[test]
fn every_url_bearing_error_variant_is_sanitized() {
    let url = display_url("https://user:pass@example.com:8443/mcp/tools?api_key=TOPSECRET");
    let variants = [
        McpClientError::Connection { url: url.clone() },
        McpClientError::ListTools { url: url.clone() },
        McpClientError::CallTool {
            url: url.clone(),
            tool_name: "test_tool".to_owned(),
        },
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

    for variant in &variants {
        assert_error_uses_sanitized_url(variant);
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
async fn alibaba_metadata_ipv4_is_blocked() {
    assert!(
        validate_url("http://100.100.100.200/latest/meta-data/").await.is_err(),
        "Alibaba Cloud metadata IPv4 must be treated as SSRF"
    );
}

// IPv4-mapped IPv6 literals (`[::ffff:a.b.c.d]`) are refused during target
// parsing: they would normalize to a bare IPv4 address while the SNI/Host kept
// the mapped form, so `prepare_url_target` rejects the host before the SSRF hook
// runs. The requests are still refused; the mechanism is parse-time rejection
// rather than address classification.
#[tokio::test]
async fn ssrf_blocks_mapped_ipv4_loopback() {
    assert!(validate_url("http://[::ffff:127.0.0.1]/mcp").await.is_err());
}

#[tokio::test]
async fn ssrf_blocks_mapped_metadata() {
    assert!(validate_url("http://[::ffff:169.254.169.254]/mcp").await.is_err());
}

#[tokio::test]
async fn alibaba_metadata_ipv4_mapped_ipv6_is_blocked() {
    assert!(
        validate_url("http://[::ffff:100.100.100.200]/latest/meta-data/")
            .await
            .is_err(),
        "IPv4-mapped Alibaba metadata literal must be refused during target parsing"
    );
}

#[test]
fn alibaba_metadata_via_dns_is_blocked() {
    // The upfront DNS classifier is gone: `prepare_url_target` normalizes every
    // resolved address and then applies the SSRF hook. A hostname that resolves
    // to the Alibaba metadata endpoint is refused by that hook, so assert the
    // hook itself rejects the address regardless of the private-upstream flag.
    let ip = "100.100.100.200".parse::<IpAddr>().unwrap();
    assert!(
        is_ssrf_blocked_ip(&ip, false),
        "a hostname resolving to Alibaba metadata must be blocked after DNS"
    );
    assert!(
        is_ssrf_blocked_ip(&ip, true),
        "cloud-metadata addresses stay blocked even when private upstreams are permitted"
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
async fn blocked_url_errors_hide_query_and_fragment() {
    let with_secrets = [
        "http://unresolvable.invalid/mcp?api_key=TOPSECRET",
        "http://127.0.0.1/admin?token=TOPSECRET",
        "http://169.254.169.254/latest?token=TOPSECRET#FRAGMENTSECRET",
    ];

    for raw in with_secrets {
        let message = validate_url(raw).await.unwrap_err().to_string();
        for leaked in ["TOPSECRET", "FRAGMENTSECRET"] {
            assert!(!message.contains(leaked), "{leaked} leaked from {raw}: {message}");
        }
    }
}

#[tokio::test]
async fn unshowable_urls_use_opaque_placeholder() {
    // URLs that cannot be parsed into a scheme+host at all fall back to the
    // opaque placeholder. (A scheme like `ftp` is parseable, so it renders as a
    // sanitized `ftp://host/path`; that case is covered by
    // `blocked_urls_report_actionable_reason`.)
    let malformed = [
        "http://exa mple.com/mcp?api_key=TOPSECRET#FRAGMENTSECRET",
        "//user:pass@example.com/mcp?api_key=TOPSECRET#FRAGMENTSECRET",
    ];

    for raw in malformed {
        let message = validate_url(raw).await.unwrap_err().to_string();
        assert!(
            message.contains("<invalid MCP URL>"),
            "expected opaque placeholder for {raw}: {message}"
        );
        for secret in ["user", "pass", "TOPSECRET", "FRAGMENTSECRET"] {
            assert!(!message.contains(secret), "{secret} leaked from {raw}: {message}");
        }
    }
}

#[tokio::test]
async fn blocked_urls_report_actionable_reason() {
    // SSRF address rejections consolidate on the single credential-safe reason
    // string, so a blocked literal reads identically to a blocked DNS result.
    for raw in ["http://127.0.0.1/mcp", "http://169.254.169.254/mcp"] {
        let message = validate_url(raw).await.unwrap_err().to_string();
        assert!(
            message.contains(SSRF_BLOCK_REASON),
            "missing SSRF reason for {raw}: {message}"
        );
    }

    // Structural target rejections (unsupported scheme, embedded credentials)
    // fail closed as a permanent "invalid or not allowed" error whose sanitized
    // URL never echoes the userinfo or scheme-specific guidance. Like an SSRF
    // block, these are hard rejections rather than transient connection failures.
    for raw in ["ftp://example.com/mcp", "http://user:pass@example.com/mcp"] {
        let message = validate_url(raw).await.unwrap_err().to_string();
        assert!(
            message.contains("invalid or not allowed"),
            "structural rejection should fail closed as an invalid-target error for {raw}: {message}"
        );
        assert!(
            !message.contains("user:pass"),
            "userinfo must never leak from {raw}: {message}"
        );
    }
}

// A parse-time target rejection (an SSRF-evasion host literal, an unsupported
// scheme, embedded userinfo, or a fragment) must classify as the *permanent*
// `InvalidTarget` rather than the transient `Connection`. On a streaming
// `tools/list` the two diverge sharply: `InvalidTarget` is excluded from
// `is_mcp_listing_runtime_failure`, so it stays a hard HTTP error, while
// `Connection` would degrade to a soft in-band SSE lifecycle. Pin the exact
// variant so that split cannot silently regress.
#[tokio::test]
async fn parse_time_rejections_classify_as_invalid_target() {
    for raw in [
        // IPv4-mapped IPv6 loopback: rejected during host parsing, before the
        // SSRF address hook ever runs.
        "http://[::ffff:127.0.0.1]/mcp",
        // Bracketed IPv4 literal: likewise rejected at parse time.
        "http://[127.0.0.1]/mcp",
        // Unsupported scheme and embedded credentials: structural rejections.
        "ftp://example.com/mcp",
        "http://user:pass@example.com/mcp",
    ] {
        let error = validate_url(raw).await.unwrap_err();
        assert!(
            matches!(error, McpClientError::InvalidTarget { .. }),
            "{raw} must classify as a hard InvalidTarget, got: {error:?}"
        );
    }
}

// A DNS resolution failure is transient, not a policy rejection: it must remain
// a `Connection` error so a streaming listing can degrade to the soft in-band
// lifecycle rather than a hard HTTP error.
#[tokio::test]
async fn dns_failure_classifies_as_connection() {
    let error = validate_url("http://unresolvable.invalid/mcp").await.unwrap_err();
    assert!(
        matches!(error, McpClientError::Connection { .. }),
        "an unresolvable host must stay a transient Connection error, got: {error:?}"
    );
}

// A resolved SSRF block stays the dedicated `SsrfBlocked` variant carrying the
// credential-safe reason string.
#[tokio::test]
async fn resolved_ssrf_block_classifies_as_ssrf_blocked() {
    let error = validate_url("http://127.0.0.1/mcp").await.unwrap_err();
    assert!(
        matches!(error, McpClientError::SsrfBlocked { .. }),
        "a loopback address must classify as SsrfBlocked, got: {error:?}"
    );
}

#[tokio::test]
async fn ssrf_allows_public_ips() {
    assert!(validate_url("http://8.8.8.8/mcp").await.is_ok());
    assert!(validate_url("https://1.1.1.1:443/v1").await.is_ok());
}

#[tokio::test]
async fn ssrf_blocks_private_rfc1918_by_default() {
    // The 0.5.6 migration hardens the default posture: RFC1918 ranges are
    // blocked unless the callout explicitly permits private upstreams.
    assert!(validate_url("http://10.0.0.5/mcp").await.is_err());
    assert!(validate_url("http://192.168.1.100/mcp").await.is_err());
}

#[tokio::test]
async fn ssrf_allows_private_rfc1918_when_private_permitted() {
    assert!(
        validate_mcp_target("http://10.0.0.5/mcp", TEST_TIMEOUT, true)
            .await
            .is_ok()
    );
    assert!(
        validate_mcp_target("http://192.168.1.100/mcp", TEST_TIMEOUT, true)
            .await
            .is_ok()
    );
}

#[test]
fn ssrf_blocked_display_lists_ssrf_url_and_reason() {
    let err = McpClientError::SsrfBlocked {
        url: display_url("http://127.0.0.1/mcp"),
        reason: "loopback address is not allowed",
    };
    let msg = err.to_string();
    for expected in ["SSRF", "127.0.0.1", "loopback address is not allowed"] {
        assert!(msg.contains(expected), "display missing {expected:?}: {msg}");
    }
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
async fn userinfo_urls_are_blocked_and_redacted() {
    let msg = validate_url("http://user:pass@example.com/mcp")
        .await
        .unwrap_err()
        .to_string();
    assert!(!msg.contains("pass"), "userinfo password must not leak: {msg}");
    assert!(validate_url("https://user@example.com/mcp").await.is_err());

    let ipv6 = validate_url("http://user:pass@[::1]:8080/mcp?api_key=TOPSECRET")
        .await
        .unwrap_err()
        .to_string();
    for secret in ["user", "pass", "TOPSECRET"] {
        assert!(!ipv6.contains(secret), "IPv6 error leaked {secret}: {ipv6}");
    }
}

#[tokio::test]
async fn ssrf_blocks_aws_imds_ipv6() {
    assert!(validate_url("http://[fd00:ec2::254]/latest/meta-data/").await.is_err());
}

#[test]
fn aws_imds_v6_detected_by_is_always_sensitive() {
    let ip = "fd00:ec2::254".parse::<IpAddr>().unwrap();
    assert!(is_always_sensitive(&ip), "fd00:ec2::254 should be SSRF-sensitive");
}

#[test]
fn unspecified_ip_detected_by_is_always_sensitive() {
    let v4 = "0.0.0.0".parse::<IpAddr>().unwrap();
    assert!(is_always_sensitive(&v4), "0.0.0.0 should be SSRF-sensitive");
    let v6 = "::".parse::<IpAddr>().unwrap();
    assert!(is_always_sensitive(&v6), ":: should be SSRF-sensitive");
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
fn ipv6_link_local_detected_by_is_always_sensitive() {
    let fe80 = "fe80::1".parse::<IpAddr>().unwrap();
    assert!(is_always_sensitive(&fe80), "fe80::1 should be SSRF-sensitive");
    let febf = "febf::1".parse::<IpAddr>().unwrap();
    assert!(is_always_sensitive(&febf), "febf::1 should be SSRF-sensitive");
    let fe00 = "fe00::1".parse::<IpAddr>().unwrap();
    assert!(!is_always_sensitive(&fe00), "fe00::1 is not link-local");
}

#[tokio::test]
async fn ssrf_blocks_ipv6_unique_local() {
    assert!(validate_url("http://[fc00::1]/mcp").await.is_err());
    assert!(validate_url("http://[fd00:ec2::23]/mcp").await.is_err());
}

#[test]
fn ipv6_unique_local_detected_by_is_always_sensitive() {
    let fc00 = "fc00::1".parse::<IpAddr>().unwrap();
    assert!(is_always_sensitive(&fc00), "fc00::1 should be SSRF-sensitive");
    let fd00 = "fd00:ec2::23".parse::<IpAddr>().unwrap();
    assert!(is_always_sensitive(&fd00), "fd00:ec2::23 should be SSRF-sensitive");
    let fb00 = "fb00::1".parse::<IpAddr>().unwrap();
    assert!(!is_always_sensitive(&fb00), "fb00::1 is not unique-local");
}

// =========================================================================
// allow_loopback
// =========================================================================

#[tokio::test]
async fn allow_loopback_permits_ipv4_loopback() {
    assert!(
        validate_mcp_target("http://127.0.0.1/mcp", TEST_TIMEOUT, true)
            .await
            .is_ok()
    );
}

#[tokio::test]
async fn allow_loopback_permits_localhost_hostname() {
    assert!(
        validate_mcp_target("http://localhost/mcp", TEST_TIMEOUT, true)
            .await
            .is_ok()
    );
}

#[tokio::test]
async fn allow_loopback_still_blocks_link_local() {
    assert!(
        validate_mcp_target("http://169.254.169.254/mcp", TEST_TIMEOUT, true)
            .await
            .is_err()
    );
}

#[tokio::test]
async fn allow_loopback_still_blocks_unspecified() {
    assert!(
        validate_mcp_target("http://0.0.0.0/mcp", TEST_TIMEOUT, true)
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
        url: display_url("http://example.com/mcp?api_key=TOPSECRET"),
        tool_name: "get_weather".to_owned(),
    };
    let msg = err.to_string();
    assert!(msg.contains("tools/call failed"), "should mention tools/call: {msg}");
    assert!(msg.contains("get_weather"), "should mention tool name: {msg}");
    assert!(msg.contains("example.com"), "should mention URL: {msg}");
    assert!(!msg.contains("TOPSECRET"), "error must redact query credentials: {msg}");
}

// =========================================================================
// Integration tests (real rmcp MCP server)
// =========================================================================

use rmcp::{
    ServerHandler,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{ServerCapabilities, ServerConfig},
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
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_instructions("Test MCP server for integration tests")
    }
}

async fn start_test_mcp_server() -> (String, tokio_util::sync::CancellationToken) {
    let ct = tokio_util::sync::CancellationToken::new();
    let config = StreamableHttpServerConfig::default()
        .with_legacy_session_mode(false)
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

#[derive(Debug, Clone)]
struct CapturedRequestHeaders {
    path: String,
    authorization: Option<String>,
    assertion: Option<String>,
    tenant: Option<String>,
    subject: Option<String>,
}

type CapturedRequests = StdArc<Mutex<Vec<CapturedRequestHeaders>>>;

async fn start_recording_mcp_server() -> (String, tokio_util::sync::CancellationToken, CapturedRequests) {
    let ct = tokio_util::sync::CancellationToken::new();
    let config = StreamableHttpServerConfig::default()
        .with_legacy_session_mode(false)
        .with_json_response(true)
        .with_sse_keep_alive(None)
        .with_cancellation_token(ct.child_token());
    let service: StreamableHttpService<TestMcpServer, LocalSessionManager> =
        StreamableHttpService::new(|| Ok(TestMcpServer::new()), StdArc::default(), config);

    let captured = CapturedRequests::default();
    let capture = StdArc::clone(&captured);
    let router = axum::Router::new()
        .nest_service("/mcp", service)
        .layer(axum::middleware::from_fn(
            move |request: axum::extract::Request, next: axum::middleware::Next| {
                let capture = StdArc::clone(&capture);
                async move {
                    let headers = request.headers();
                    capture.lock().unwrap().push(CapturedRequestHeaders {
                        path: request.uri().path().to_owned(),
                        authorization: headers
                            .get(http::header::AUTHORIZATION)
                            .and_then(|value| value.to_str().ok())
                            .map(str::to_owned),
                        assertion: headers
                            .get(crate::callout_credentials::MCP_AUTHORIZED_HEADER)
                            .and_then(|value| value.to_str().ok())
                            .map(str::to_owned),
                        tenant: headers
                            .get("x-tenant-id")
                            .and_then(|value| value.to_str().ok())
                            .map(str::to_owned),
                        subject: headers
                            .get("x-user-id")
                            .and_then(|value| value.to_str().ok())
                            .map(str::to_owned),
                    });
                    next.run(request).await
                }
            },
        ));
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

    (format!("http://{addr}/mcp"), ct, captured)
}

async fn start_redirecting_mcp_server() -> (String, tokio_util::sync::CancellationToken, CapturedRequests) {
    let ct = tokio_util::sync::CancellationToken::new();
    let config = StreamableHttpServerConfig::default()
        .with_legacy_session_mode(false)
        .with_json_response(true)
        .with_sse_keep_alive(None)
        .with_cancellation_token(ct.child_token());
    let service: StreamableHttpService<TestMcpServer, LocalSessionManager> =
        StreamableHttpService::new(|| Ok(TestMcpServer::new()), StdArc::default(), config);

    let captured = CapturedRequests::default();
    let capture = StdArc::clone(&captured);
    let router = axum::Router::new()
        .route(
            "/redirect",
            axum::routing::post(|| async {
                (http::StatusCode::TEMPORARY_REDIRECT, [(http::header::LOCATION, "/mcp")])
            }),
        )
        .nest_service("/mcp", service)
        .layer(axum::middleware::from_fn(
            move |request: axum::extract::Request, next: axum::middleware::Next| {
                let capture = StdArc::clone(&capture);
                async move {
                    let headers = request.headers();
                    capture.lock().unwrap().push(CapturedRequestHeaders {
                        path: request.uri().path().to_owned(),
                        authorization: headers
                            .get(http::header::AUTHORIZATION)
                            .and_then(|value| value.to_str().ok())
                            .map(str::to_owned),
                        assertion: headers
                            .get(crate::callout_credentials::MCP_AUTHORIZED_HEADER)
                            .and_then(|value| value.to_str().ok())
                            .map(str::to_owned),
                        tenant: headers
                            .get("x-tenant-id")
                            .and_then(|value| value.to_str().ok())
                            .map(str::to_owned),
                        subject: headers
                            .get("x-user-id")
                            .and_then(|value| value.to_str().ok())
                            .map(str::to_owned),
                    });
                    next.run(request).await
                }
            },
        ));
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

    (format!("http://{addr}/redirect"), ct, captured)
}

async fn start_failing_initialize_server() -> (String, tokio_util::sync::CancellationToken, StdArc<AtomicUsize>) {
    let ct = tokio_util::sync::CancellationToken::new();
    let requests = StdArc::new(AtomicUsize::new(0));
    let observed = StdArc::clone(&requests);
    let router = axum::Router::new().route(
        "/mcp",
        axum::routing::post(move || {
            let observed = StdArc::clone(&observed);
            async move {
                observed.fetch_add(1, Ordering::SeqCst);
                http::StatusCode::INTERNAL_SERVER_ERROR
            }
        }),
    );
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
    (format!("http://{addr}/mcp"), ct, requests)
}

/// Ordered log of the JSON-RPC methods observed by a recording server, used to
/// prove how many `initialize` handshakes and `tools/call` requests the session
/// pool actually issued.
type ObservedMethods = StdArc<Mutex<Vec<String>>>;

/// Extract the JSON-RPC `method` from a request body, if present. Notifications,
/// requests, and responses all carry it; DELETE/GET frames with no JSON body
/// yield `None`.
fn jsonrpc_method(body: &[u8]) -> Option<String> {
    serde_json::from_slice::<serde_json::Value>(body)
        .ok()
        .and_then(|value| {
            value
                .get("method")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned)
        })
}

/// Count how many times `method` appears in a recording server's method log.
fn method_count(methods: &ObservedMethods, method: &str) -> usize {
    methods
        .lock()
        .unwrap()
        .iter()
        .filter(|seen| seen.as_str() == method)
        .count()
}

/// A real rmcp server that records the JSON-RPC method of every POST it receives
/// so a test can assert the handshake/`tools/call` counts a pooled execution
/// produced.
async fn start_method_recording_mcp_server() -> (String, tokio_util::sync::CancellationToken, ObservedMethods) {
    let ct = tokio_util::sync::CancellationToken::new();
    let config = StreamableHttpServerConfig::default()
        .with_legacy_session_mode(false)
        .with_json_response(true)
        .with_sse_keep_alive(None)
        .with_cancellation_token(ct.child_token());
    let service: StreamableHttpService<TestMcpServer, LocalSessionManager> =
        StreamableHttpService::new(|| Ok(TestMcpServer::new()), StdArc::default(), config);

    let methods: ObservedMethods = StdArc::default();
    let record = StdArc::clone(&methods);
    let router = axum::Router::new()
        .nest_service("/mcp", service)
        .layer(axum::middleware::from_fn(
            move |request: axum::extract::Request, next: axum::middleware::Next| {
                let record = StdArc::clone(&record);
                async move {
                    let (parts, body) = request.into_parts();
                    let bytes = axum::body::to_bytes(body, 64 * 1024).await.unwrap_or_default();
                    if let Some(method) = jsonrpc_method(&bytes) {
                        record.lock().unwrap().push(method);
                    }
                    let request = axum::extract::Request::from_parts(parts, axum::body::Body::from(bytes));
                    next.run(request).await
                }
            },
        ));
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

    (format!("http://{addr}/mcp"), ct, methods)
}

/// A real rmcp server that records methods and rejects the *second* `tools/call`
/// it ever receives with a 500. The first call (round 1) succeeds and pools the
/// session; the second call (the round-2 reuse attempt) fails. A 500 (unlike a
/// 404 `SessionExpired`) is not transparently reinitialized by rmcp, so it
/// exercises the pool's at-most-once policy: the reused session is evicted and
/// the error surfaced *without* a fresh retry, so the server never sees a third
/// `tools/call`. JSON-response mode leaves the server stateless (no
/// `Mcp-Session-Id`), so the reuse attempt is distinguished by call ordinal
/// rather than by session identity.
async fn start_second_call_rejecting_mcp_server() -> (String, tokio_util::sync::CancellationToken, ObservedMethods) {
    use axum::response::IntoResponse as _;

    let ct = tokio_util::sync::CancellationToken::new();
    let config = StreamableHttpServerConfig::default()
        .with_legacy_session_mode(false)
        .with_json_response(true)
        .with_sse_keep_alive(None)
        .with_cancellation_token(ct.child_token());
    let service: StreamableHttpService<TestMcpServer, LocalSessionManager> =
        StreamableHttpService::new(|| Ok(TestMcpServer::new()), StdArc::default(), config);

    let methods: ObservedMethods = StdArc::default();
    let record = StdArc::clone(&methods);
    let tool_calls = StdArc::new(AtomicUsize::new(0));
    let router = axum::Router::new()
        .nest_service("/mcp", service)
        .layer(axum::middleware::from_fn(
            move |request: axum::extract::Request, next: axum::middleware::Next| {
                let record = StdArc::clone(&record);
                let tool_calls = StdArc::clone(&tool_calls);
                async move {
                    let (parts, body) = request.into_parts();
                    let bytes = axum::body::to_bytes(body, 64 * 1024).await.unwrap_or_default();
                    let method = jsonrpc_method(&bytes);
                    if let Some(method) = &method {
                        record.lock().unwrap().push(method.clone());
                    }
                    // Reject exactly the second tools/call (the round-2 reuse attempt).
                    // The corrected pool never retries it, so no third call arrives.
                    if method.as_deref() == Some("tools/call") && tool_calls.fetch_add(1, Ordering::SeqCst) == 1 {
                        return http::StatusCode::INTERNAL_SERVER_ERROR.into_response();
                    }
                    let request = axum::extract::Request::from_parts(parts, axum::body::Body::from(bytes));
                    next.run(request).await
                }
            },
        ));
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

    (format!("http://{addr}/mcp"), ct, methods)
}

fn assert_scoped_context_on_every_exchange(captured: &CapturedRequests) {
    let captured = captured.lock().unwrap();
    assert!(
        captured.len() >= 2,
        "initialize and the requested MCP operation should both reach the server"
    );
    for request in captured.iter() {
        assert_eq!(request.authorization.as_deref(), Some("Bearer per-user-token"));
        assert_eq!(request.assertion.as_deref(), Some("signed-assertion"));
    }
}

fn owner_projecting_mcp_callout() -> McpCallout {
    let mut registry = praxis_filter::FilterRegistry::with_builtins();
    praxis_filter::register_filters!(
        @register registry,
        http "state_owner_headers" => crate::StateOwnerHeadersFilter::from_config
    );
    let mut entries: Vec<praxis_filter::FilterEntry> = serde_yaml::from_str(
        "- filter: state_owner_headers\n  tenant_header: x-tenant-id\n  subject_header: x-user-id\n",
    )
    .unwrap();
    let mut pipeline = praxis_filter::FilterPipeline::build(&mut entries, &registry).unwrap();
    pipeline.set_allow_private_upstreams(true);
    McpCallout::fabricated(true)
        .unwrap()
        .with_pipeline_for_test(StdArc::new(pipeline))
}

const INTEGRATION_TIMEOUT: Duration = Duration::from_secs(10);
const TEST_MAX_RESULT_BYTES: usize = 1_048_576;

#[tokio::test]
async fn scoped_connector_context_reaches_initialize_and_tools_list_unchanged() {
    let (url, ct, captured) = start_recording_mcp_server().await;
    let owner = StateOwner::from_trusted_parts("tenant-a", "issuer-a", "subject-a").unwrap();
    let bearer = SecretString::from("per-user-token");
    let assertion = SecretString::from("signed-assertion");
    let context = McpConnectorContext {
        owner: &owner,
        bearer: Some(&bearer),
        assertion: Some(&assertion),
    };
    let client_shadow = serde_json::json!({
        "authorization": "Bearer client-shadow",
        "x-mcp-authorized": "client-shadow"
    });

    let tools = list_tools_with_forwarded_headers(
        &url,
        Some(&client_shadow),
        Some("static-token"),
        &[],
        None,
        Some(&context),
        INTEGRATION_TIMEOUT,
        128,
        &McpCallout::fabricated(true).unwrap(),
    )
    .await
    .unwrap();
    ct.cancel();

    assert_eq!(tools.len(), 4);
    assert_scoped_context_on_every_exchange(&captured);
}

#[tokio::test]
async fn scoped_connector_context_reaches_initialize_and_tools_call_unchanged() {
    let (url, ct, captured) = start_recording_mcp_server().await;
    let owner = StateOwner::from_trusted_parts("tenant-a", "issuer-a", "subject-a").unwrap();
    let bearer = SecretString::from("per-user-token");
    let assertion = SecretString::from("signed-assertion");
    let context = McpConnectorContext {
        owner: &owner,
        bearer: Some(&bearer),
        assertion: Some(&assertion),
    };

    let result = call_tool_with_forwarded_headers(
        None,
        &url,
        None,
        Some("static-token"),
        &[],
        None,
        Some(&context),
        "echo",
        serde_json::json!({"message": "hello"}),
        INTEGRATION_TIMEOUT,
        TEST_MAX_RESULT_BYTES,
        &McpCallout::fabricated(true).unwrap(),
    )
    .await
    .unwrap();
    ct.cancel();

    assert_eq!(
        result
            .content
            .first()
            .and_then(|content| content.as_text())
            .map(|text| text.text.as_str()),
        Some("hello")
    );
    assert_scoped_context_on_every_exchange(&captured);
}

#[tokio::test]
async fn two_user_mcp_contexts_are_isolated_across_initialize_and_list() {
    for suffix in ["a", "b"] {
        let (url, ct, captured) = start_recording_mcp_server().await;
        let owner = StateOwner::from_trusted_parts(
            format!("tenant-{suffix}"),
            "urn:integration:test",
            format!("user-{suffix}"),
        )
        .unwrap();
        let bearer = SecretString::from(format!("credential-{suffix}"));
        let assertion = SecretString::from(format!("assertion-{suffix}"));
        let context = McpConnectorContext {
            owner: &owner,
            bearer: Some(&bearer),
            assertion: Some(&assertion),
        };

        list_tools_with_forwarded_headers(
            &url,
            None,
            None,
            &[],
            None,
            Some(&context),
            INTEGRATION_TIMEOUT,
            128,
            &owner_projecting_mcp_callout(),
        )
        .await
        .unwrap();
        ct.cancel();

        let captured = captured.lock().unwrap();
        assert!(captured.len() >= 2, "initialize and list must both be observed");
        let expected_authorization = format!("Bearer credential-{suffix}");
        let expected_assertion = format!("assertion-{suffix}");
        let expected_tenant = format!("tenant-{suffix}");
        let expected_subject = format!("user-{suffix}");
        for request in captured.iter() {
            assert_eq!(request.authorization.as_deref(), Some(expected_authorization.as_str()));
            assert_eq!(request.assertion.as_deref(), Some(expected_assertion.as_str()));
            assert_eq!(request.tenant.as_deref(), Some(expected_tenant.as_str()));
            assert_eq!(request.subject.as_deref(), Some(expected_subject.as_str()));
            let other = if suffix == "a" { "b" } else { "a" };
            for leaked in [
                format!("credential-{other}"),
                format!("assertion-{other}"),
                format!("tenant-{other}"),
                format!("user-{other}"),
            ] {
                assert!(
                    request.authorization.as_deref() != Some(leaked.as_str())
                        && request.assertion.as_deref() != Some(leaked.as_str())
                        && request.tenant.as_deref() != Some(leaked.as_str())
                        && request.subject.as_deref() != Some(leaked.as_str()),
                    "user {suffix} MCP exchange leaked user {other} context"
                );
            }
        }
    }
}

#[tokio::test]
async fn direct_url_never_sends_client_shadow_as_ambient_context() {
    let (url, ct, captured) = start_recording_mcp_server().await;
    let client_shadow = serde_json::json!({
        "authorization": "Bearer client-shadow",
        "x-mcp-authorized": "client-shadow"
    });

    let tools = list_tools_with_forwarded_headers(
        &url,
        Some(&client_shadow),
        None,
        &[],
        None,
        None,
        INTEGRATION_TIMEOUT,
        128,
        &McpCallout::fabricated(true).unwrap(),
    )
    .await
    .unwrap();
    ct.cancel();

    assert_eq!(tools.len(), 4);
    let captured = captured.lock().unwrap();
    assert!(captured.len() >= 2);
    for request in captured.iter() {
        assert!(
            request.authorization.is_none(),
            "direct URL received client Authorization shadow"
        );
        assert!(
            request.assertion.is_none(),
            "direct URL received client assertion shadow"
        );
    }
}

#[tokio::test]
async fn scoped_connector_context_is_not_followed_across_redirects() {
    let (url, ct, captured) = start_redirecting_mcp_server().await;
    let owner = StateOwner::from_trusted_parts("tenant-a", "issuer-a", "subject-a").unwrap();
    let bearer = SecretString::from("per-user-token");
    let assertion = SecretString::from("signed-assertion");
    let context = McpConnectorContext {
        owner: &owner,
        bearer: Some(&bearer),
        assertion: Some(&assertion),
    };

    let result = list_tools_with_forwarded_headers(
        &url,
        None,
        None,
        &[],
        None,
        Some(&context),
        INTEGRATION_TIMEOUT,
        128,
        &McpCallout::fabricated(true).unwrap(),
    )
    .await;
    ct.cancel();

    assert!(result.is_err(), "redirect must terminate the MCP exchange");
    let captured = captured.lock().unwrap();
    assert!(captured.iter().any(|request| request.path == "/redirect"));
    assert!(
        captured.iter().all(|request| request.path != "/mcp"),
        "ambient connector context must never be replayed to a redirected target"
    );
}

#[tokio::test]
async fn failed_initialize_stops_before_service_start_without_background_retries() {
    let (url, ct, requests) = start_failing_initialize_server().await;

    let result = list_tools(
        &url,
        None,
        None,
        Duration::from_millis(500),
        128,
        &McpCallout::fabricated(true).unwrap(),
    )
    .await;
    tokio::time::sleep(Duration::from_millis(50)).await;
    ct.cancel();

    assert!(
        result.is_err(),
        "a failed initialize exchange must fail the MCP operation"
    );
    assert_eq!(
        requests.load(Ordering::SeqCst),
        1,
        "pre-running initialization failure must not leave a worker retrying in the background"
    );
}

#[tokio::test]
async fn list_tools_returns_all_tools() {
    let (url, ct) = start_test_mcp_server().await;
    let tools = list_tools(
        &url,
        None,
        None,
        INTEGRATION_TIMEOUT,
        128,
        &McpCallout::fabricated(true).unwrap(),
    )
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
    let tools = list_tools(
        &url,
        None,
        None,
        INTEGRATION_TIMEOUT,
        128,
        &McpCallout::fabricated(true).unwrap(),
    )
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
    let result = list_tools(
        &url,
        None,
        None,
        INTEGRATION_TIMEOUT,
        2,
        &McpCallout::fabricated(true).unwrap(),
    )
    .await;
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
    let tools = list_tools(
        &url,
        Some(&headers),
        None,
        INTEGRATION_TIMEOUT,
        128,
        &McpCallout::fabricated(true).unwrap(),
    )
    .await
    .unwrap();
    ct.cancel();

    assert_eq!(tools.len(), 4, "should still return all 4 tools");
}

#[tokio::test]
async fn list_tools_with_authorization() {
    let (url, ct) = start_test_mcp_server().await;
    let tools = list_tools(
        &url,
        None,
        Some("test-token"),
        INTEGRATION_TIMEOUT,
        128,
        &McpCallout::fabricated(true).unwrap(),
    )
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
        TEST_MAX_RESULT_BYTES,
        &McpCallout::fabricated(true).unwrap(),
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
        TEST_MAX_RESULT_BYTES,
        &McpCallout::fabricated(true).unwrap(),
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
        TEST_MAX_RESULT_BYTES,
        &McpCallout::fabricated(true).unwrap(),
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
        TEST_MAX_RESULT_BYTES,
        &McpCallout::fabricated(true).unwrap(),
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
        TEST_MAX_RESULT_BYTES,
        &McpCallout::fabricated(true).unwrap(),
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
        TEST_MAX_RESULT_BYTES,
        &McpCallout::fabricated(true).unwrap(),
    )
    .await;
    ct.cancel();

    let err = result.expect_err("should time out");
    let msg = err.to_string();
    assert!(msg.contains("timed out"), "error should mention timeout: {msg}");
}

// =========================================================================
// Session pooling / reuse (#1019)
// =========================================================================

/// Two `tools/call`s for the same identity across consecutive rounds share one
/// initialized session: exactly one `initialize` handshake, two `tools/call`s.
#[tokio::test]
async fn pooled_session_reused_across_rounds_runs_single_initialize() {
    let (url, ct, methods) = start_method_recording_mcp_server().await;
    let pool = McpSessionPool::new();
    let callout = McpCallout::fabricated(true).unwrap();

    for message in ["round-1", "round-2"] {
        let result = call_tool_with_forwarded_headers(
            Some((&pool, "identity-shared")),
            &url,
            None,
            None,
            &[],
            None,
            None,
            "echo",
            serde_json::json!({ "message": message }),
            INTEGRATION_TIMEOUT,
            TEST_MAX_RESULT_BYTES,
            &callout,
        )
        .await
        .unwrap();
        assert_eq!(
            result
                .content
                .first()
                .and_then(|c| c.as_text())
                .map(|t| t.text.as_str()),
            Some(message),
            "each pooled round must return its own result"
        );
    }
    ct.cancel();

    assert_eq!(
        method_count(&methods, "initialize"),
        1,
        "reusing a warm session must run the handshake only once"
    );
    assert_eq!(
        method_count(&methods, "tools/call"),
        2,
        "each round still issues its own tools/call"
    );
}

/// Sessions never cross security contexts: two calls with different identity
/// keys (same endpoint) each open their own session, so each runs its own
/// `initialize`.
#[tokio::test]
async fn distinct_identity_keys_never_reuse_a_session() {
    let (url, ct, methods) = start_method_recording_mcp_server().await;
    let pool = McpSessionPool::new();
    let callout = McpCallout::fabricated(true).unwrap();

    for key in ["identity-a", "identity-b"] {
        call_tool_with_forwarded_headers(
            Some((&pool, key)),
            &url,
            None,
            None,
            &[],
            None,
            None,
            "echo",
            serde_json::json!({ "message": key }),
            INTEGRATION_TIMEOUT,
            TEST_MAX_RESULT_BYTES,
            &callout,
        )
        .await
        .unwrap();
    }
    ct.cancel();

    assert_eq!(
        method_count(&methods, "initialize"),
        2,
        "different identities must each run their own handshake"
    );
    assert_eq!(method_count(&methods, "tools/call"), 2);
}

/// The empty-fingerprint sentinel is a fail-closed ambiguous identity: it must
/// never reuse or retain a session, so repeated calls each re-handshake.
#[tokio::test]
async fn empty_fingerprint_never_pools_a_session() {
    let (url, ct, methods) = start_method_recording_mcp_server().await;
    let pool = McpSessionPool::new();
    let callout = McpCallout::fabricated(true).unwrap();

    for _ in 0..2 {
        call_tool_with_forwarded_headers(
            Some((&pool, "")),
            &url,
            None,
            None,
            &[],
            None,
            None,
            "echo",
            serde_json::json!({ "message": "x" }),
            INTEGRATION_TIMEOUT,
            TEST_MAX_RESULT_BYTES,
            &callout,
        )
        .await
        .unwrap();
    }
    ct.cancel();

    assert_eq!(
        method_count(&methods, "initialize"),
        2,
        "an ambiguous (empty) fingerprint must never reuse a session"
    );
    assert!(
        pool.checkout("").is_none(),
        "the empty-fingerprint sentinel must never retain a session"
    );
}

/// A reused session whose `tools/call` fails is evicted and the error surfaced
/// **without** a fresh retry. This is the at-most-once guarantee: an ambiguous
/// failure (here a 5xx, whose delivery is unknown) must never re-execute the tool
/// on a new session, or a non-idempotent tool could run twice. Because there is
/// no second attempt, one logical call also cannot exceed its single `timeout`
/// budget or open a second session.
#[tokio::test]
async fn reused_session_failure_evicts_without_retry() {
    let (url, ct, methods) = start_second_call_rejecting_mcp_server().await;
    let pool = McpSessionPool::new();
    let callout = McpCallout::fabricated(true).unwrap();

    // Round 1: fresh session, clean call -> returned to the pool.
    let first = call_tool_with_forwarded_headers(
        Some((&pool, "identity-shared")),
        &url,
        None,
        None,
        &[],
        None,
        None,
        "echo",
        serde_json::json!({ "message": "round-1" }),
        INTEGRATION_TIMEOUT,
        TEST_MAX_RESULT_BYTES,
        &callout,
    )
    .await
    .unwrap();
    assert_eq!(
        first.content.first().and_then(|c| c.as_text()).map(|t| t.text.as_str()),
        Some("round-1")
    );

    // Round 2: the reused session's tools/call is rejected. The pool evicts the
    // session and propagates the error; it does NOT reinitialize and retry.
    let second = call_tool_with_forwarded_headers(
        Some((&pool, "identity-shared")),
        &url,
        None,
        None,
        &[],
        None,
        None,
        "echo",
        serde_json::json!({ "message": "round-2" }),
        INTEGRATION_TIMEOUT,
        TEST_MAX_RESULT_BYTES,
        &callout,
    )
    .await;
    assert!(
        second.is_err(),
        "a failed reused call must surface the error, not silently retry: {second:?}"
    );
    ct.cancel();

    assert_eq!(
        method_count(&methods, "initialize"),
        1,
        "the failed reused session must not be reinitialized (no fresh fallback)"
    );
    assert_eq!(
        method_count(&methods, "tools/call"),
        2,
        "at-most-once: round 1 (ok) + round 2 reuse attempt (rejected), never a third retry"
    );
    assert!(
        pool.checkout("identity-shared").is_none(),
        "a failed reused session must be evicted, not returned to the pool"
    );
}

/// Open one real, initialized session against `url` for direct pool bookkeeping
/// tests (the fast paths that only touch `checkin`/`checkout`, not a full call).
async fn open_pooled_session(url: &str, callout: &McpCallout) -> PooledSession {
    open_tool_session(
        url,
        None,
        None,
        &[],
        None,
        None,
        INTEGRATION_TIMEOUT,
        TEST_MAX_RESULT_BYTES,
        callout,
        &parse_display_url(url),
    )
    .await
    .unwrap()
}

/// Count the idle sessions the pool retains under `key`, draining it.
fn drain_key(pool: &McpSessionPool, key: &str) -> usize {
    let mut count = 0;
    while pool.checkout(key).is_some() {
        count += 1;
    }
    count
}

/// Checking a session back out empties its stack and removes the key, so a later
/// checkout for the same identity finds nothing warm and opens fresh.
#[tokio::test]
async fn checkout_removes_emptied_key() {
    let (url, ct, _methods) = start_method_recording_mcp_server().await;
    let pool = McpSessionPool::new();
    let callout = McpCallout::fabricated(true).unwrap();

    pool.checkin("k".to_owned(), open_pooled_session(&url, &callout).await);
    assert!(
        pool.checkout("k").is_some(),
        "the single warm session must check out once"
    );
    assert!(
        pool.checkout("k").is_none(),
        "the emptied key must be removed, so a second checkout finds nothing"
    );
    ct.cancel();
}

/// More check-ins for one identity than [`MAX_IDLE_PER_KEY`] (a within-round
/// parallel fan-out to one server) retain only the cap; the extras are dropped.
#[tokio::test]
async fn checkin_bounds_idle_sessions_per_key() {
    let (url, ct, _methods) = start_method_recording_mcp_server().await;
    let pool = McpSessionPool::new();
    let callout = McpCallout::fabricated(true).unwrap();

    for _ in 0..(MAX_IDLE_PER_KEY + 3) {
        pool.checkin("k".to_owned(), open_pooled_session(&url, &callout).await);
    }
    assert_eq!(
        drain_key(&pool, "k"),
        MAX_IDLE_PER_KEY,
        "a single identity must retain at most MAX_IDLE_PER_KEY warm sessions"
    );
    ct.cancel();
}

/// A pathological fan-out across many identities cannot retain more than
/// [`MAX_TOTAL_IDLE`] live sessions for the request's lifetime; check-ins past the
/// global ceiling are dropped even when no single key is at its own cap.
#[tokio::test]
async fn checkin_bounds_total_idle_sessions_across_keys() {
    let (url, ct, _methods) = start_method_recording_mcp_server().await;
    let pool = McpSessionPool::new();
    let callout = McpCallout::fabricated(true).unwrap();

    // Spread MAX_TOTAL_IDLE + 1 sessions over enough keys that the per-key cap
    // (MAX_IDLE_PER_KEY) never fires first, so only the global ceiling can bound
    // the total. The final check-in must be dropped by the global cap.
    let keys = MAX_TOTAL_IDLE / MAX_IDLE_PER_KEY + 1;
    for i in 0..=MAX_TOTAL_IDLE {
        let key = format!("k{}", i % keys);
        pool.checkin(key, open_pooled_session(&url, &callout).await);
    }

    let retained: usize = (0..keys).map(|i| drain_key(&pool, &format!("k{i}"))).sum();
    assert_eq!(
        retained, MAX_TOTAL_IDLE,
        "the pool must retain at most MAX_TOTAL_IDLE warm sessions across all keys"
    );
    ct.cancel();
}

#[derive(Debug, Clone)]
struct SlowListToolsMcpServer {
    delay_per_page: Duration,
}

impl ServerHandler for SlowListToolsMcpServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
    }

    async fn list_tools(
        &self,
        request: Option<PaginatedRequestParams>,
        _context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<rmcp::model::ListToolsResult, rmcp::ErrorData> {
        tokio::time::sleep(self.delay_per_page).await;
        let cursor = request.and_then(|p| p.cursor);
        let next_cursor = match cursor.as_deref() {
            None => Some("page2".to_owned()),
            Some("page2") => Some("page3".to_owned()),
            _ => None,
        };
        let tool = rmcp::model::Tool::new(
            "dummy".to_owned(),
            "dummy tool".to_owned(),
            std::sync::Arc::new(serde_json::Map::new()),
        );
        let mut res = rmcp::model::ListToolsResult::with_all_items(vec![tool]);
        res.next_cursor = next_cursor;
        Ok(res)
    }
}

async fn start_slow_list_mcp_server(delay_per_page: Duration) -> (String, tokio_util::sync::CancellationToken) {
    let ct = tokio_util::sync::CancellationToken::new();
    let config = StreamableHttpServerConfig::default()
        .with_legacy_session_mode(false)
        .with_json_response(true)
        .with_sse_keep_alive(None)
        .with_cancellation_token(ct.child_token());

    let service: StreamableHttpService<SlowListToolsMcpServer, LocalSessionManager> = StreamableHttpService::new(
        move || Ok(SlowListToolsMcpServer { delay_per_page }),
        std::sync::Arc::default(),
        config,
    );

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

#[tokio::test]
async fn list_tools_cumulative_pagination_timeout() {
    let (url, ct) = start_slow_list_mcp_server(Duration::from_millis(150)).await;
    let short_timeout = Duration::from_millis(200);
    let result = list_tools(
        &url,
        None,
        None,
        short_timeout,
        128,
        &McpCallout::fabricated(true).unwrap(),
    )
    .await;
    ct.cancel();

    let err = result.expect_err("cumulative pagination time exceeding timeout should time out");
    let msg = err.to_string();
    assert!(msg.contains("timed out"), "error should mention timeout: {msg}");
}

/// MCP server whose `tools/list` returns a single tool carrying an oversized
/// description, used to prove `list_tools` bounds the response body before it
/// is buffered and deserialized.
#[derive(Debug, Clone)]
struct OversizedListToolsMcpServer {
    /// Byte length of the description attached to the one returned tool.
    description_bytes: usize,
}

impl ServerHandler for OversizedListToolsMcpServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<rmcp::model::ListToolsResult, rmcp::ErrorData> {
        let tool = rmcp::model::Tool::new(
            "dummy".to_owned(),
            "x".repeat(self.description_bytes),
            std::sync::Arc::new(serde_json::Map::new()),
        );
        Ok(rmcp::model::ListToolsResult::with_all_items(vec![tool]))
    }
}

async fn start_oversized_list_mcp_server(description_bytes: usize) -> (String, tokio_util::sync::CancellationToken) {
    let ct = tokio_util::sync::CancellationToken::new();
    let config = StreamableHttpServerConfig::default()
        .with_legacy_session_mode(false)
        .with_json_response(true)
        .with_sse_keep_alive(None)
        .with_cancellation_token(ct.child_token());

    let service: StreamableHttpService<OversizedListToolsMcpServer, LocalSessionManager> = StreamableHttpService::new(
        move || Ok(OversizedListToolsMcpServer { description_bytes }),
        std::sync::Arc::default(),
        config,
    );

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

#[tokio::test]
async fn list_tools_rejects_oversized_response() {
    // A single tool whose description alone dwarfs the 1 MiB control-response
    // ceiling. `max_tools` caps only the tool *count* (here just 1), so before
    // the transport was size-bounded the entire body was downloaded and
    // deserialized regardless — the memory-exhaustion vector this guards.
    let (url, ct) = start_oversized_list_mcp_server(2 * 1024 * 1024).await;
    let result = list_tools(
        &url,
        None,
        None,
        INTEGRATION_TIMEOUT,
        128,
        &McpCallout::fabricated(true).unwrap(),
    )
    .await;
    ct.cancel();

    let err = result.expect_err("oversized tools/list response must be rejected before buffering");
    // The filtered-subrequest transport classifies the over-ceiling body as
    // `CalloutOutcome::ResponseTooLarge`; the caller reads that typed overflow
    // back out-of-band (rmcp discards the transport error) and surfaces the
    // dedicated `ResponseTooLarge` variant, which callers map to HTTP 413 —
    // distinct from the generic 502 a plain `ListTools` failure yields.
    //
    // NOTE: tools/list is a ClientRequest, so rmcp routes it through the
    // streaming post_message_with_max_sse_event_size path. The server returns
    // JSON (not SSE), so praxis buffers anyway (Blocker 5). The executor
    // backstop passed to execute_streaming is 2x the binding cap (spec §4.5 F3),
    // so when praxis buffers and trips on an oversized response, it reports the
    // 2x limit. This is intentional: the buffered fallback is memory-bounded at
    // 2x the cap (see subrequest_transport.rs streaming_executor_backstop doc).
    match err {
        McpClientError::ResponseTooLarge { limit, .. } => {
            assert_eq!(
                limit,
                2 * MAX_CONTROL_RESPONSE_BYTES,
                "buffered fallback in execute_streaming is bounded at 2x the binding cap"
            );
        },
        other => panic!("oversized response should surface as ResponseTooLarge, got: {other:?}"),
    }
}

/// MCP server that paginates `tools/list`, returning one tool per page whose
/// description is large but stays under the per-page control ceiling. Used to
/// prove the cumulative byte budget bounds pagination: each page passes the
/// per-page bound and the tool *count* stays under `max_tools`, yet their union
/// crosses the operation-wide limit.
#[derive(Debug, Clone)]
struct MultiPageListToolsMcpServer {
    /// Byte length of the description attached to the one tool per page.
    description_bytes: usize,
    /// Number of pages the server will serve before ending pagination.
    total_pages: usize,
}

impl ServerHandler for MultiPageListToolsMcpServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
    }

    async fn list_tools(
        &self,
        request: Option<PaginatedRequestParams>,
        _context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<rmcp::model::ListToolsResult, rmcp::ErrorData> {
        let page: usize = request.and_then(|p| p.cursor).and_then(|c| c.parse().ok()).unwrap_or(0);
        let tool = rmcp::model::Tool::new(
            format!("tool_{page}"),
            "x".repeat(self.description_bytes),
            std::sync::Arc::new(serde_json::Map::new()),
        );
        let mut res = rmcp::model::ListToolsResult::with_all_items(vec![tool]);
        res.next_cursor = (page + 1 < self.total_pages).then(|| (page + 1).to_string());
        Ok(res)
    }
}

async fn start_multi_page_list_mcp_server(
    description_bytes: usize,
    total_pages: usize,
) -> (String, tokio_util::sync::CancellationToken) {
    let ct = tokio_util::sync::CancellationToken::new();
    let config = StreamableHttpServerConfig::default()
        .with_legacy_session_mode(false)
        .with_json_response(true)
        .with_sse_keep_alive(None)
        .with_cancellation_token(ct.child_token());

    let service: StreamableHttpService<MultiPageListToolsMcpServer, LocalSessionManager> = StreamableHttpService::new(
        move || {
            Ok(MultiPageListToolsMcpServer {
                description_bytes,
                total_pages,
            })
        },
        std::sync::Arc::default(),
        config,
    );

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

#[tokio::test]
async fn list_tools_rejects_oversized_cumulative_pagination() {
    // Every page carries a single ~900 KiB tool: individually under the 1 MiB
    // per-page control ceiling, and the running count (one tool per page) never
    // approaches `max_tools`. Only the cumulative byte budget across pages can
    // catch this, so it exercises the aggregate bound rather than the per-page
    // bound or the count cap. `total_pages` is capped well below `MAX_PAGES` so a
    // regression that dropped the budget check fails cleanly (bounded transfer)
    // instead of streaming tens of MiB.
    let (url, ct) = start_multi_page_list_mcp_server(900 * 1024, 12).await;
    let result = list_tools(
        &url,
        None,
        None,
        INTEGRATION_TIMEOUT,
        128,
        &McpCallout::fabricated(true).unwrap(),
    )
    .await;
    ct.cancel();

    let err = result.expect_err("cumulative tools/list bytes exceeding the budget must be rejected");
    assert!(
        matches!(err, McpClientError::ListingTooLarge { .. }),
        "aggregate overflow should surface as ListingTooLarge, got: {err:?}"
    );
}

// =========================================================================
// classify_deadline
// =========================================================================

#[test]
fn classify_deadline_maps_size_signal_to_413_else_timeout() {
    use std::sync::{Arc, OnceLock};
    let url = parse_display_url("https://mcp.example/mcp");

    // No signal recorded -> generic Timeout.
    let signal: Arc<OnceLock<subrequest_transport::TransportSignal>> = Arc::new(OnceLock::new());
    let err = classify_deadline(&signal, &url, Duration::from_secs(1));
    assert!(matches!(err, McpClientError::Timeout { .. }));

    // Size signal recorded -> 413-classified ResponseTooLarge.
    let signal: Arc<OnceLock<subrequest_transport::TransportSignal>> = Arc::new(OnceLock::new());
    assert!(
        signal
            .set(subrequest_transport::TransportSignal::ResponseTooLarge { limit: 5 })
            .is_ok(),
        "signal OnceLock should be empty"
    );
    let err = classify_deadline(&signal, &url, Duration::from_secs(1));
    assert!(matches!(err, McpClientError::ResponseTooLarge { .. }));
}
