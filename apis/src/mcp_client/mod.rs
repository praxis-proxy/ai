// SPDX-License-Identifier: Apache-2.0
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
    clippy::needless_pass_by_value,
    clippy::unused_self,
    missing_docs,
    reason = "tests"
)]
mod tests;

use std::{
    collections::HashMap,
    fmt,
    net::{IpAddr, Ipv4Addr, SocketAddr},
    time::Duration,
};

use rmcp::{
    Peer, RoleClient, ServiceExt as _,
    model::{CallToolRequestParams, PaginatedRequestParams},
    transport::{StreamableHttpClientTransport, streamable_http_client::StreamableHttpClientTransportConfig},
};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Cloud instance-metadata IPv4 endpoints that the generic loopback,
/// link-local, and unspecified checks do not already cover. Any request that
/// resolves to one of these is treated as an SSRF attempt.
const CLOUD_METADATA_IPV4: &[Ipv4Addr] = &[
    // Alibaba Cloud ECS metadata service. Lives in 100.64.0.0/10 shared
    // address space, so it is not flagged as link-local.
    Ipv4Addr::new(100, 100, 100, 200),
];

// -----------------------------------------------------------------------------
// McpDisplayUrl
// -----------------------------------------------------------------------------

/// A URL reduced to its non-secret locator parts, safe to embed in error
/// messages and logs.
///
/// Only the scheme, host, optional port, and path are kept; userinfo, query,
/// and fragment are dropped, since each of those can carry credentials.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct McpDisplayUrl(String);

impl McpDisplayUrl {
    /// Build a safe display URL from an already-parsed URI.
    ///
    /// Reassembled from individual components (scheme, host, optional port,
    /// path) rather than the raw authority string, so URL userinfo can never
    /// reach the output. Query strings and fragments (both common credential
    /// carriers) are never appended.
    pub(crate) fn from_uri(uri: &http::Uri) -> Self {
        let (Some(scheme), Some(host)) = (uri.scheme_str(), uri.host()) else {
            return Self::invalid();
        };
        let mut out = String::with_capacity(scheme.len() + host.len() + 16);
        out.push_str(scheme);
        out.push_str("://");
        // `http::Uri::host()` can return an IPv6 literal without its brackets;
        // put them back so the value stays a valid URL authority.
        if host.contains(':') && !host.starts_with('[') {
            out.push('[');
            out.push_str(host);
            out.push(']');
        } else {
            out.push_str(host);
        }
        if let Some(port) = uri.port_u16() {
            out.push(':');
            out.push_str(&port.to_string());
        }
        out.push_str(uri.path());
        Self(out)
    }

    /// Opaque replacement used when a URL cannot be shown safely.
    fn invalid() -> Self {
        Self("<invalid MCP URL>".to_owned())
    }
}

impl fmt::Display for McpDisplayUrl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Construct an [`McpClientError::SsrfBlocked`] from an already-sanitized URL
/// and a fixed, non-sensitive reason string.
fn ssrf_blocked(url: McpDisplayUrl, reason: &'static str) -> McpClientError {
    McpClientError::SsrfBlocked { url, reason }
}

// -----------------------------------------------------------------------------
// McpClientError
// -----------------------------------------------------------------------------

/// Errors from MCP server communication.
#[derive(Debug, thiserror::Error)]
pub(crate) enum McpClientError {
    /// Failed to connect to the MCP server or complete the
    /// handshake.
    ///
    /// The underlying transport error is deliberately not retained: its
    /// `Display` output can echo the full request URL, credentials in userinfo
    /// or query parameters included.
    #[error("mcp connection failed for {url}")]
    Connection {
        /// URL of the MCP server.
        url: McpDisplayUrl,
    },

    /// The `tools/list` call failed or returned an invalid
    /// response.
    ///
    /// The underlying transport error is deliberately not retained: its
    /// `Display` output can echo the full request URL, credentials in userinfo
    /// or query parameters included.
    #[error("mcp tools/list failed for {url}")]
    ListTools {
        /// URL of the MCP server.
        url: McpDisplayUrl,
    },

    /// The `tools/call` request failed.
    ///
    /// The underlying transport error is deliberately not retained: its
    /// `Display` output can echo the full request URL, credentials in userinfo
    /// or query parameters included.
    #[error("mcp tools/call failed for {url} tool {tool_name}")]
    CallTool {
        /// URL of the MCP server.
        url: McpDisplayUrl,

        /// Name of the tool that was called.
        tool_name: String,
    },

    /// Timed out waiting for the MCP server.
    #[error("mcp request timed out for {url} after {timeout:?}")]
    Timeout {
        /// URL of the MCP server.
        url: McpDisplayUrl,

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
        url: McpDisplayUrl,

        /// Actual tool count.
        count: usize,

        /// Configured maximum.
        max: usize,
    },

    /// MCP server URL is invalid or resolves to a blocked address.
    #[error("mcp server URL blocked (SSRF): {url}: {reason}")]
    SsrfBlocked {
        /// The blocked URL.
        url: McpDisplayUrl,

        /// Safe explanation of why the URL was blocked.
        reason: &'static str,
    },

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
    let display_url = resolved.display_url;
    let client = tokio::time::timeout(timeout, Box::pin(().serve(transport)))
        .await
        .map_err(|_elapsed| McpClientError::Timeout {
            url: display_url.clone(),
            timeout,
        })?
        .map_err(|_source| McpClientError::Connection {
            url: display_url.clone(),
        })?;
    let tools = paginate_tools(&client, timeout, max_tools, &display_url).await?;
    tools_to_json(tools)
}

/// Call `tools/call` on an MCP server and return the result.
///
/// Creates a fresh Streamable HTTP transport per call, same
/// pattern as [`list_tools`]. Session reuse deferred to MCP
/// Foundation PR 5.
///
/// # Errors
///
/// Returns [`McpClientError`] on connection failure, timeout, or
/// tool execution failure.
#[expect(clippy::too_many_arguments, reason = "allow_loopback extends the existing param set")]
#[expect(clippy::too_many_lines, reason = "transport setup + call follows list_tools pattern")]
#[expect(clippy::large_stack_frames, reason = "rmcp call_tool future is inherently large")]
pub(crate) async fn call_tool(
    server_url: &str,
    headers: Option<&serde_json::Value>,
    authorization: Option<&str>,
    tool_name: &str,
    arguments: serde_json::Value,
    timeout: Duration,
    allow_loopback: bool,
) -> Result<rmcp::model::CallToolResult, McpClientError> {
    let resolved = resolve_and_validate(server_url, timeout, allow_loopback).await?;
    let transport = StreamableHttpClientTransport::with_client(
        build_pinned_client(&resolved)?,
        build_transport_config(server_url, headers, authorization)?,
    );
    let display_url = resolved.display_url;

    let client = tokio::time::timeout(timeout, Box::pin(().serve(transport)))
        .await
        .map_err(|_elapsed| McpClientError::Timeout {
            url: display_url.clone(),
            timeout,
        })?
        .map_err(|_source| McpClientError::Connection {
            url: display_url.clone(),
        })?;

    let parsed_args = match &arguments {
        serde_json::Value::Object(obj) => Some(obj.clone()),
        serde_json::Value::String(s) => serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(s).ok(),
        _ => None,
    };
    let mut params = CallToolRequestParams::new(tool_name.to_owned());
    if let Some(args_obj) = parsed_args {
        params = params.with_arguments(args_obj);
    }

    tokio::time::timeout(timeout, Box::pin(client.call_tool(params)))
        .await
        .map_err(|_elapsed| McpClientError::Timeout {
            url: display_url.clone(),
            timeout,
        })?
        .map_err(|_source| McpClientError::CallTool {
            url: display_url.clone(),
            tool_name: tool_name.to_owned(),
        })
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
    url: &McpDisplayUrl,
) -> Result<Vec<rmcp::model::Tool>, McpClientError> {
    let mut all_tools = Vec::new();
    let mut cursor = None;
    for _ in 0..MAX_PAGES {
        let params = PaginatedRequestParams::default().with_cursor(cursor);
        let page = tokio::time::timeout(timeout, Box::pin(client.list_tools(Some(params))))
            .await
            .map_err(|_elapsed| McpClientError::Timeout {
                url: url.clone(),
                timeout,
            })?
            .map_err(|_source| McpClientError::ListTools { url: url.clone() })?;
        all_tools.extend(page.tools);
        if all_tools.len() > max_tools {
            return Err(McpClientError::TooManyTools {
                url: url.clone(),
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
        url: url.clone(),
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
/// connection is made. For the connect path, `resolve_and_validate`
/// also pins resolved addresses.
///
/// # Errors
///
/// Returns [`McpClientError::SsrfBlocked`] if the URL resolves to
/// a loopback, link-local, or metadata address.
pub(crate) async fn validate_mcp_url(url: &str, timeout: Duration, allow_loopback: bool) -> Result<(), McpClientError> {
    resolve_and_validate(url, timeout, allow_loopback)
        .await
        .map(|_resolved| ())
}

/// Resolved MCP URL with validated addresses pinned for
/// connect-time use, eliminating DNS rebinding between
/// validation and the actual connection.
struct ResolvedMcpUrl {
    /// Sanitized URL retained for diagnostics.
    display_url: McpDisplayUrl,

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
        .map_err(|_parse_err| ssrf_blocked(McpDisplayUrl::invalid(), "invalid URL"))?;
    let scheme = uri.scheme_str().unwrap_or_default();
    if scheme != "http" && scheme != "https" {
        return Err(ssrf_blocked(McpDisplayUrl::invalid(), "scheme must be http or https"));
    }
    let display_url = McpDisplayUrl::from_uri(&uri);
    if uri.authority().is_some_and(|a| a.as_str().contains('@')) {
        return Err(ssrf_blocked(display_url, "embedded credentials are not allowed"));
    }
    let Some(host) = uri.host() else {
        return Err(ssrf_blocked(display_url, "URL must include a host"));
    };
    let host = host.trim_matches(|c| c == '[' || c == ']');
    if !allow_loopback && is_blocked_hostname(host) {
        return Err(ssrf_blocked(display_url, "localhost hostnames are not allowed"));
    }
    if let Ok(ip) = host.parse::<IpAddr>() {
        check_ip(ip, &display_url, allow_loopback)?;
        return Ok(ResolvedMcpUrl {
            display_url,
            hostname: None,
            addrs: Vec::new(),
        });
    }
    let port = uri.port_u16().unwrap_or(if scheme == "https" { 443 } else { 80 });
    resolve_hostname_ssrf(host, port, display_url, timeout, allow_loopback).await
}

/// Check a literal IP address against the SSRF block list.
fn check_ip(ip: IpAddr, url: &McpDisplayUrl, allow_loopback: bool) -> Result<(), McpClientError> {
    let ip = praxis_core::connectivity::normalize_mapped_ipv4(ip);
    if allow_loopback && ip.is_loopback() {
        return Ok(());
    }
    if is_ssrf_sensitive(&ip) {
        return Err(ssrf_blocked(
            url.clone(),
            "address is loopback, link-local, unique-local, unspecified, or cloud metadata",
        ));
    }
    Ok(())
}

/// Resolve a hostname and check all resolved addresses. Fails
/// closed: DNS resolution failure or timeout blocks the request.
/// Returns validated addresses for connect-time pinning.
async fn resolve_hostname_ssrf(
    host: &str,
    port: u16,
    url: McpDisplayUrl,
    timeout: Duration,
    allow_loopback: bool,
) -> Result<ResolvedMcpUrl, McpClientError> {
    let addrs: Vec<SocketAddr> = tokio::time::timeout(timeout, tokio::net::lookup_host((host, port)))
        .await
        .map_err(|_elapsed| McpClientError::Timeout {
            url: url.clone(),
            timeout,
        })?
        .map_err(|_dns_err| ssrf_blocked(url.clone(), "DNS resolution failed"))?
        .collect();
    check_resolved_addrs(&addrs, &url, allow_loopback)?;
    Ok(ResolvedMcpUrl {
        display_url: url,
        hostname: Some(host.to_owned()),
        addrs,
    })
}

/// Check DNS-resolved addresses against the SSRF block list.
fn check_resolved_addrs(addrs: &[SocketAddr], url: &McpDisplayUrl, allow_loopback: bool) -> Result<(), McpClientError> {
    for addr in addrs {
        check_ip(addr.ip(), url, allow_loopback)?;
    }
    Ok(())
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

    builder.build().map_err(|_source| McpClientError::Connection {
        url: resolved.display_url.clone(),
    })
}

/// Headers that must not pass through from client-supplied MCP
/// tool config into the proxy's outbound MCP transport.
fn is_blocked_mcp_header(name: &http::HeaderName) -> bool {
    if matches!(
        *name,
        http::header::AUTHORIZATION
            | http::header::CONNECTION
            | http::header::CONTENT_LENGTH
            | http::header::COOKIE
            | http::header::FORWARDED
            | http::header::HOST
            | http::header::PROXY_AUTHORIZATION
            | http::header::SET_COOKIE
            | http::header::TE
            | http::header::TRAILER
            | http::header::TRANSFER_ENCODING
            | http::header::UPGRADE
    ) {
        return true;
    }
    let s = name.as_str();
    s.starts_with("x-forwarded-") || s.starts_with("x-praxis-") || s.starts_with("x-mcp-") || s.starts_with("x-a2a-")
}

/// Hostnames that resolve to loopback.
fn is_blocked_hostname(host: &str) -> bool {
    let lower = host.to_ascii_lowercase();
    lower == "localhost" || lower.ends_with(".localhost")
}

/// Whether `v4` matches a known cloud instance-metadata endpoint that the
/// generic loopback, link-local, and unspecified checks miss.
fn is_cloud_metadata_ipv4(v4: Ipv4Addr) -> bool {
    CLOUD_METADATA_IPV4.contains(&v4)
}

/// Loopback, link-local, unique-local, unspecified, and
/// known cloud metadata addresses are SSRF-sensitive.
fn is_ssrf_sensitive(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => v4.is_loopback() || v4.is_link_local() || v4.is_unspecified() || is_cloud_metadata_ipv4(*v4),
        IpAddr::V6(v6) => {
            let [a, b, ..] = v6.octets();
            v6.is_loopback() || v6.is_unspecified() || (a == 0xFE && (b & 0xC0) == 0x80) || (a & 0xFE) == 0xFC
        },
    }
}
/// Convert `rmcp::model::Tool` values to opaque JSON.
fn tools_to_json(tools: Vec<rmcp::model::Tool>) -> Result<Vec<serde_json::Value>, McpClientError> {
    tools
        .into_iter()
        .map(|tool| serde_json::to_value(tool).map_err(McpClientError::Serialization))
        .collect()
}
