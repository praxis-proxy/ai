// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! MCP client wrapper for calling upstream MCP servers.
//!
//! Thin layer over `rmcp` that exposes [`list_tools`] for resolving
//! MCP tool declarations. Designed for reuse by `mcp_tool` (#27)
//! when `call_tool` support is added.

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests"
)]
mod tests;

use std::{
    collections::HashMap,
    net::{IpAddr, SocketAddr},
    time::Duration,
};

use rmcp::{
    Peer, RoleClient, ServiceExt as _,
    model::PaginatedRequestParams,
    transport::{StreamableHttpClientTransport, streamable_http_client::StreamableHttpClientTransportConfig},
};

// -----------------------------------------------------------------------------
// McpClientError
// -----------------------------------------------------------------------------

/// Errors from MCP server communication.
#[derive(Debug, thiserror::Error)]
pub(crate) enum McpClientError {
    /// Failed to connect to the MCP server or complete the
    /// handshake.
    #[error("mcp connection failed for {url}: {source}")]
    Connection {
        /// URL of the MCP server.
        url: String,

        /// Underlying error.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// The `tools/list` call failed or returned an invalid
    /// response.
    #[error("mcp tools/list failed for {url}: {source}")]
    ListTools {
        /// URL of the MCP server.
        url: String,

        /// Underlying error.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// Timed out waiting for the MCP server.
    #[error("mcp request timed out for {url} after {timeout:?}")]
    Timeout {
        /// URL of the MCP server.
        url: String,

        /// Configured timeout duration.
        timeout: Duration,
    },

    /// Failed to serialize tool definitions to JSON.
    #[error("failed to serialize tool definitions: {0}")]
    Serialization(
        /// Serialization error.
        #[from]
        serde_json::Error,
    ),

    /// An MCP server returned more tools than the configured cap.
    #[error("mcp server {url} returned too many tools: {count} exceeds limit of {max}")]
    TooManyTools {
        /// Server URL.
        url: String,

        /// Actual tool count.
        count: usize,

        /// Configured maximum.
        max: usize,
    },

    /// MCP server URL resolves to a blocked address.
    #[error("mcp server URL blocked (SSRF): {0}")]
    SsrfBlocked(
        /// The blocked URL.
        String,
    ),

    /// Authorization token contains invalid header characters.
    #[error("authorization token contains invalid HTTP header characters")]
    InvalidAuthorization,
}

// -----------------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------------

/// Call `tools/list` on an MCP server and return tool definitions
/// as opaque JSON values.
///
/// Creates a fresh Streamable HTTP transport per call. The
/// `previous_tools` cache in `ResponsesState` prevents redundant
/// calls across request continuations.
///
/// # Errors
///
/// Returns [`McpClientError`] on connection failure, timeout, or
/// invalid server response.
#[expect(clippy::too_many_arguments, reason = "allow_loopback extends the existing param set")]
pub(crate) async fn list_tools(
    server_url: &str,
    headers: Option<&serde_json::Value>,
    authorization: Option<&str>,
    timeout: Duration,
    max_tools: usize,
    allow_loopback: bool,
) -> Result<Vec<serde_json::Value>, McpClientError> {
    let resolved = resolve_and_validate(server_url, timeout, allow_loopback).await?;
    let transport = StreamableHttpClientTransport::with_client(
        build_pinned_client(&resolved)?,
        build_transport_config(server_url, headers, authorization)?,
    );
    let url = server_url.to_owned();
    let client = tokio::time::timeout(timeout, Box::pin(().serve(transport)))
        .await
        .map_err(|_elapsed| McpClientError::Timeout {
            url: url.clone(),
            timeout,
        })?
        .map_err(|e| McpClientError::Connection {
            url: url.clone(),
            source: Box::new(e),
        })?;
    let tools = paginate_tools(&client, timeout, max_tools, &url).await?;
    tools_to_json(tools)
}

/// Cap on pagination rounds to prevent infinite loops from
/// servers returning empty pages with valid cursors.
const MAX_PAGES: usize = 100;

/// Paginate `tools/list`, bounded by both `max_tools` and
/// [`MAX_PAGES`].
#[expect(clippy::too_many_lines, reason = "pagination loop with error branches")]
async fn paginate_tools(
    client: &Peer<RoleClient>,
    timeout: Duration,
    max_tools: usize,
    url: &str,
) -> Result<Vec<rmcp::model::Tool>, McpClientError> {
    let mut all_tools = Vec::new();
    let mut cursor = None;
    for _ in 0..MAX_PAGES {
        let params = PaginatedRequestParams::default().with_cursor(cursor);
        let page = tokio::time::timeout(timeout, Box::pin(client.list_tools(Some(params))))
            .await
            .map_err(|_elapsed| McpClientError::Timeout {
                url: url.to_owned(),
                timeout,
            })?
            .map_err(|e| McpClientError::ListTools {
                url: url.to_owned(),
                source: Box::new(e),
            })?;
        all_tools.extend(page.tools);
        if all_tools.len() > max_tools {
            return Err(McpClientError::TooManyTools {
                url: url.to_owned(),
                count: all_tools.len(),
                max: max_tools,
            });
        }
        match page.next_cursor {
            Some(next) => cursor = Some(next),
            None => return Ok(all_tools),
        }
    }
    Err(McpClientError::TooManyTools {
        url: url.to_owned(),
        count: all_tools.len(),
        max: max_tools,
    })
}

// -----------------------------------------------------------------------------
// Private Helpers
// -----------------------------------------------------------------------------

/// Build transport config from server URL, optional headers, and
/// optional `OAuth` authorization token.
///
/// # Errors
///
/// Returns [`McpClientError::InvalidAuthorization`] if the token
/// contains characters invalid in HTTP header values.
fn build_transport_config(
    server_url: &str,
    headers: Option<&serde_json::Value>,
    authorization: Option<&str>,
) -> Result<StreamableHttpClientTransportConfig, McpClientError> {
    let mut config = StreamableHttpClientTransportConfig::with_uri(server_url);
    let mut header_map = HashMap::new();

    if let Some(headers_obj) = headers.and_then(serde_json::Value::as_object) {
        for (key, value) in headers_obj {
            if let Some(value_str) = value.as_str()
                && let Ok(name) = key.parse::<http::HeaderName>()
                && !is_blocked_mcp_header(&name)
                && let Ok(val) = http::HeaderValue::from_str(value_str)
            {
                header_map.insert(name, val);
            }
        }
    }

    inject_authorization(&mut header_map, authorization)?;

    if !header_map.is_empty() {
        config = config.custom_headers(header_map);
    }

    Ok(config)
}

/// Inject `authorization` as a Bearer token.
///
/// `Authorization` headers in the `headers` field are stripped
/// upstream so the dedicated `authorization` field is the only
/// auth source.
///
/// # Errors
///
/// Returns [`McpClientError::InvalidAuthorization`] if the token
/// contains characters invalid in HTTP header values.
fn inject_authorization(
    header_map: &mut HashMap<http::HeaderName, http::HeaderValue>,
    authorization: Option<&str>,
) -> Result<(), McpClientError> {
    let Some(token) = authorization else {
        return Ok(());
    };
    let bearer = format!("Bearer {token}");
    let val = http::HeaderValue::from_str(&bearer).map_err(|_invalid| McpClientError::InvalidAuthorization)?;
    header_map.insert(http::header::AUTHORIZATION, val);
    Ok(())
}

/// Reject MCP server URLs that point at SSRF-sensitive addresses.
///
/// Lightweight validation for use on the cache-hit path where no
/// connection is made. For the connect path, use
/// [`resolve_and_validate`] to also pin resolved addresses.
pub(crate) async fn validate_mcp_url(url: &str, timeout: Duration, allow_loopback: bool) -> Result<(), McpClientError> {
    resolve_and_validate(url, timeout, allow_loopback)
        .await
        .map(|_resolved| ())
}

/// Resolved MCP URL with validated addresses pinned for
/// connect-time use, eliminating DNS rebinding between
/// validation and the actual connection.
struct ResolvedMcpUrl {
    /// Hostname to pin (present for DNS-resolved hosts, absent
    /// for literal IPs).
    hostname: Option<String>,

    /// Validated socket addresses from DNS resolution.
    addrs: Vec<SocketAddr>,
}

/// Validate an MCP server URL and resolve its addresses.
///
/// Returns the validated resolved addresses so the caller can
/// pin them on the HTTP client, closing the DNS-rebinding
/// TOCTOU window between validation and connect.
async fn resolve_and_validate(
    url: &str,
    timeout: Duration,
    allow_loopback: bool,
) -> Result<ResolvedMcpUrl, McpClientError> {
    let uri: http::Uri = url
        .parse()
        .map_err(|_parse_err| McpClientError::SsrfBlocked(url.to_owned()))?;
    let scheme = uri.scheme_str().unwrap_or_default();
    if scheme != "http" && scheme != "https" {
        return Err(McpClientError::SsrfBlocked(url.to_owned()));
    }
    if uri.authority().is_some_and(|a| a.as_str().contains('@')) {
        return Err(McpClientError::SsrfBlocked(
            "URL with embedded credentials is not allowed".to_owned(),
        ));
    }
    let Some(host) = uri.host() else {
        return Err(McpClientError::SsrfBlocked(url.to_owned()));
    };
    let host = host.trim_matches(|c| c == '[' || c == ']');
    if !allow_loopback && is_blocked_hostname(host) {
        return Err(McpClientError::SsrfBlocked(url.to_owned()));
    }
    if let Ok(ip) = host.parse::<IpAddr>() {
        check_ip(ip, url, allow_loopback)?;
        return Ok(ResolvedMcpUrl {
            hostname: None,
            addrs: Vec::new(),
        });
    }
    let port = uri.port_u16().unwrap_or(if scheme == "https" { 443 } else { 80 });
    resolve_hostname_ssrf(host, port, url, timeout, allow_loopback).await
}

/// Check a literal IP address against the SSRF block list.
fn check_ip(ip: IpAddr, url: &str, allow_loopback: bool) -> Result<(), McpClientError> {
    let ip = praxis_core::connectivity::normalize_mapped_ipv4(ip);
    if allow_loopback && ip.is_loopback() {
        return Ok(());
    }
    if is_ssrf_sensitive(&ip) {
        return Err(McpClientError::SsrfBlocked(url.to_owned()));
    }
    Ok(())
}

/// Resolve a hostname and check all resolved addresses. Fails
/// closed: DNS resolution failure or timeout blocks the request.
/// Returns validated addresses for connect-time pinning.
async fn resolve_hostname_ssrf(
    host: &str,
    port: u16,
    url: &str,
    timeout: Duration,
    allow_loopback: bool,
) -> Result<ResolvedMcpUrl, McpClientError> {
    let addrs: Vec<SocketAddr> = tokio::time::timeout(timeout, tokio::net::lookup_host((host, port)))
        .await
        .map_err(|_elapsed| McpClientError::Timeout {
            url: url.to_owned(),
            timeout,
        })?
        .map_err(|_dns_err| McpClientError::SsrfBlocked(url.to_owned()))?
        .collect();
    for addr in &addrs {
        let ip = praxis_core::connectivity::normalize_mapped_ipv4(addr.ip());
        if allow_loopback && ip.is_loopback() {
            continue;
        }
        if is_ssrf_sensitive(&ip) {
            return Err(McpClientError::SsrfBlocked(url.to_owned()));
        }
    }
    Ok(ResolvedMcpUrl {
        hostname: Some(host.to_owned()),
        addrs,
    })
}

/// Build a reqwest client with resolved addresses pinned, so
/// the connection uses the same IPs that passed SSRF validation.
fn build_pinned_client(resolved: &ResolvedMcpUrl) -> Result<reqwest::Client, McpClientError> {
    let mut builder = reqwest::Client::builder()
        .pool_max_idle_per_host(0)
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none());

    if let Some(hostname) = &resolved.hostname {
        builder = builder.resolve_to_addrs(hostname, &resolved.addrs);
    }

    builder.build().map_err(|e| McpClientError::Connection {
        url: String::new(),
        source: Box::new(e),
    })
}

/// Headers that must not pass through from client-supplied MCP
/// tool config into the proxy's outbound MCP transport.
fn is_blocked_mcp_header(name: &http::HeaderName) -> bool {
    if matches!(
        *name,
        http::header::AUTHORIZATION
            | http::header::HOST
            | http::header::CONTENT_LENGTH
            | http::header::TRANSFER_ENCODING
            | http::header::CONNECTION
            | http::header::TE
            | http::header::TRAILER
            | http::header::UPGRADE
            | http::header::PROXY_AUTHORIZATION
    ) {
        return true;
    }
    let s = name.as_str();
    s.starts_with("x-praxis-") || s.starts_with("x-mcp-") || s.starts_with("x-a2a-")
}

/// Hostnames that resolve to loopback.
fn is_blocked_hostname(host: &str) -> bool {
    let lower = host.to_ascii_lowercase();
    lower == "localhost" || lower.ends_with(".localhost")
}

/// Loopback, link-local, unspecified, and known cloud
/// metadata addresses are SSRF-sensitive.
fn is_ssrf_sensitive(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => v4.is_loopback() || v4.is_link_local() || v4.is_unspecified(),
        IpAddr::V6(v6) => {
            let [a, b, ..] = v6.octets();
            v6.is_loopback() || v6.is_unspecified() || (a == 0xFE && (b & 0xC0) == 0x80) || is_cloud_metadata_v6(v6)
        },
    }
}

/// AWS EC2 IMDS IPv6 endpoint.
fn is_cloud_metadata_v6(v6: &std::net::Ipv6Addr) -> bool {
    const AWS_IMDS_V6: std::net::Ipv6Addr = std::net::Ipv6Addr::new(0xFD00, 0x0EC2, 0, 0, 0, 0, 0, 0x0254);
    *v6 == AWS_IMDS_V6
}

/// Convert `rmcp::model::Tool` values to opaque JSON.
fn tools_to_json(tools: Vec<rmcp::model::Tool>) -> Result<Vec<serde_json::Value>, McpClientError> {
    tools
        .into_iter()
        .map(|tool| serde_json::to_value(tool).map_err(McpClientError::Serialization))
        .collect()
}
