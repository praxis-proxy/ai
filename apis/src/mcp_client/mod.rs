// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! MCP client wrapper for calling upstream MCP servers.
//!
//! Thin layer over `rmcp` that exposes [`list_tools_with_forwarded_headers`] for resolving
//! MCP tool declarations. Designed for reuse by `mcp_tool` (#27)
//! when `call_tool` support is added.

mod sse_adapter;
mod streaming_selector;
mod subrequest_transport;
#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::needless_pass_by_value,
    clippy::significant_drop_tightening,
    clippy::too_many_lines,
    clippy::unused_self,
    missing_docs,
    reason = "tests"
)]
mod tests;

use std::{
    collections::{HashMap, HashSet},
    fmt,
    net::{IpAddr, Ipv4Addr},
    time::Duration,
};

use rmcp::{
    Peer, RoleClient, ServiceExt as _,
    model::{CallToolRequestParams, PaginatedRequestParams},
    service::RunningService,
    transport::{StreamableHttpClientTransport, streamable_http_client::StreamableHttpClientTransportConfig},
};
use secrecy::{ExposeSecret as _, SecretString};

pub use self::streaming_selector::McpStreamingSelectorFilter;
use self::subrequest_transport::MAX_CONTROL_RESPONSE_BYTES;
pub(crate) use self::subrequest_transport::{
    McpCallout, bind_mcp_outbound_chain, build_bare_outbound_pipeline, transport_signal_error, validate_mcp_target,
};
use crate::StateOwner;

/// Request-scoped ambient context authorized only for a configured connector.
///
/// Direct client-selected `server_url` callsites must pass `None`. The raw
/// assertion and bearer are exposed only while constructing the RMCP transport
/// headers and never enter the tool map or retained response state.
pub(crate) struct McpConnectorContext<'a> {
    /// Trusted owner projected into each filtered MCP exchange.
    pub owner: &'a StateOwner,
    /// Optional per-user bearer token overriding a static entry token.
    pub bearer: Option<&'a SecretString>,
    /// Optional opaque MCP Gateway assertion.
    pub assertion: Option<&'a SecretString>,
}

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Cumulative byte budget for a complete `tools/list` result across all
/// paginated pages.
///
/// Each page is already wire-bounded to [`MAX_CONTROL_RESPONSE_BYTES`] before
/// deserialization, but pagination must not multiply that ceiling: without an
/// aggregate cap a server could return up to [`MAX_PAGES`] near-ceiling pages —
/// each carrying a single, count-cheap tool that stays under `max_tools` — and
/// force the proxy to retain their decoded union (on the order of 100 MiB per
/// server, amplified across concurrently resolved servers). This bounds that
/// union. It is deliberately generous relative to a realistic listing (128
/// tools averaging 32 KiB) so well-behaved servers are never rejected.
pub(super) const MAX_LISTING_RESPONSE_BYTES: usize = 4 * MAX_CONTROL_RESPONSE_BYTES;

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

    /// An MCP server's paginated `tools/list` result exceeded the cumulative
    /// byte budget for a single discovery operation.
    ///
    /// Distinct from [`TooManyTools`](Self::TooManyTools): the tool *count* can
    /// stay within `max_tools` while the retained bytes across pages do not.
    #[error("mcp server {url} returned an oversized tools/list: {bytes} bytes exceeds limit of {max}")]
    ListingTooLarge {
        /// Server URL.
        url: McpDisplayUrl,

        /// Cumulative decoded listing bytes observed when the budget was crossed.
        bytes: usize,

        /// Configured cumulative maximum.
        max: usize,
    },

    /// An MCP server returned a single response exceeding the transport's
    /// size limit.
    ///
    /// Distinct from [`ListingTooLarge`](Self::ListingTooLarge), which bounds the
    /// *cumulative* decoded `tools/list` union across pages: this is one response
    /// body crossing the per-exchange wire ceiling. The filtered-subrequest
    /// transport classifies it as
    /// [`CalloutOutcome::ResponseTooLarge`](praxis_filter::CalloutOutcome::ResponseTooLarge);
    /// callers map it to HTTP 413.
    #[error("mcp server {url} returned a response exceeding the {limit} byte limit")]
    ResponseTooLarge {
        /// Server URL (credential-safe).
        url: McpDisplayUrl,

        /// The effective response-size limit that was exceeded.
        limit: usize,
    },

    /// MCP server URL is invalid or resolves to a blocked address.
    #[error("mcp server URL blocked (SSRF): {url}: {reason}")]
    SsrfBlocked {
        /// The blocked URL.
        url: McpDisplayUrl,

        /// Safe explanation of why the URL was blocked.
        reason: &'static str,
    },

    /// The MCP server URL is structurally invalid or disallowed before any dial:
    /// an unsupported scheme, embedded userinfo, a fragment, or a
    /// malformed/disallowed host literal (for example a bracketed IPv4 or an
    /// IPv4-mapped IPv6 address rejected at parse time).
    ///
    /// A permanent request-shaping failure — the target is never contacted. Like
    /// [`SsrfBlocked`](Self::SsrfBlocked), it is a policy/validation rejection
    /// rather than a transient upstream failure, so a streaming `tools/list`
    /// retains its HTTP error instead of degrading to an in-band lifecycle event.
    #[error("mcp server URL is invalid or not allowed: {url}")]
    InvalidTarget {
        /// The rejected URL, reduced to a credential-safe display form.
        url: McpDisplayUrl,
    },

    /// Authorization token contains invalid header characters.
    #[error("authorization token contains invalid HTTP header characters")]
    InvalidAuthorization,
}

/// Parse a server URL into a safe display URL, or return invalid fallback.
pub(crate) fn parse_display_url(server_url: &str) -> McpDisplayUrl {
    server_url
        .parse::<http::Uri>()
        .map_or_else(|_| McpDisplayUrl::invalid(), |uri| McpDisplayUrl::from_uri(&uri))
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
/// The transport is size-bounded: `initialize` and `tools/list`
/// bodies are rejected before deserialization once they cross the
/// control-response ceiling, so an untrusted server cannot exhaust
/// proxy memory with an oversized response. Across pagination the
/// decoded listing is additionally bounded by
/// [`MAX_LISTING_RESPONSE_BYTES`], so a server cannot multiply the
/// per-page ceiling by returning many near-ceiling pages. `max_tools`
/// remains a further, post-deserialization limit on the tool *count*.
///
/// # Errors
///
/// Returns [`McpClientError`] on connection failure, timeout, an
/// oversized response (per page or cumulative), or an otherwise
/// invalid server response.
#[expect(
    clippy::too_many_arguments,
    reason = "callout replaces the prior allow_loopback param"
)]
#[cfg(test)]
pub(crate) async fn list_tools(
    server_url: &str,
    headers: Option<&serde_json::Value>,
    authorization: Option<&str>,
    timeout: Duration,
    max_tools: usize,
    callout: &McpCallout,
) -> Result<Vec<serde_json::Value>, McpClientError> {
    list_tools_with_forwarded_headers(
        server_url,
        headers,
        authorization,
        &[],
        None,
        None,
        timeout,
        max_tools,
        callout,
    )
    .await
}

/// Call `tools/list` with an additional trusted, operator-allowlisted header
/// set. Forwarded values override same-named client tool-entry headers.
#[expect(
    clippy::too_many_arguments,
    reason = "trusted forwarded headers extend the existing API"
)]
#[expect(clippy::too_many_lines, reason = "transport setup and bounded listing are linear")]
pub(crate) async fn list_tools_with_forwarded_headers(
    server_url: &str,
    headers: Option<&serde_json::Value>,
    authorization: Option<&str>,
    forwarded_header_names: &[http::HeaderName],
    forwarded_headers: Option<&http::HeaderMap>,
    connector_context: Option<&McpConnectorContext<'_>>,
    timeout: Duration,
    max_tools: usize,
    callout: &McpCallout,
) -> Result<Vec<serde_json::Value>, McpClientError> {
    // No upfront SSRF classifier: the subrequest transport validates the
    // dial target during the callout via `prepare_url_target`, so this path
    // resolves DNS exactly once. `initialize` and `tools/list` are
    // control-plane exchanges: the transport bounds each response body to the
    // control ceiling before deserialization, so an untrusted server cannot
    // exhaust proxy memory before `max_tools` (a count-only limit) is ever
    // evaluated. Across pagination the decoded listing is additionally
    // bounded by `MAX_LISTING_RESPONSE_BYTES` (see `paginate_tools`).
    let display_url = parse_display_url(server_url);
    let mcp_client = subrequest_transport::McpSubrequestClient::control(
        callout.clone(),
        timeout,
        connector_context.map(|context| context.owner.clone()),
    );
    // Take the signal handle before the client is moved into the rmcp
    // transport, so a `ResponseTooLarge` or `SsrfBlocked` classification
    // recorded during the exchange (which rmcp otherwise discards) can be
    // read back below.
    let signal = mcp_client.signal_handle();
    let transport = StreamableHttpClientTransport::with_client(
        mcp_client,
        build_transport_config_with_forwarded_headers(
            server_url,
            headers,
            authorization,
            forwarded_header_names,
            forwarded_headers,
            connector_context,
        )?,
    );

    let mut running: Option<RunningService<RoleClient, ()>> = None;
    let outcome = tokio::time::timeout(timeout, async {
        let client = Box::pin(().serve(transport)).await.map_err(|_source| {
            transport_signal_error(&signal, &display_url).unwrap_or_else(|| McpClientError::Connection {
                url: display_url.clone(),
            })
        })?;
        let client = running.insert(client);
        let tools = Box::pin(paginate_tools(client, max_tools, &display_url))
            .await
            .map_err(|err| transport_signal_error(&signal, &display_url).unwrap_or(err))?;
        tools_to_json(tools)
    })
    .await;

    // Close (not drop) the service on every post-serve exit so no background
    // worker task is left holding our subrequest executor. A pre-serve
    // initialization failure owns no `RunningService`; dropping the failed
    // `serve` future drops its transport without starting the service worker.
    if let Some(mut client) = running {
        drop(client.close().await);
    }

    match outcome {
        Ok(result) => result,
        Err(_elapsed) => Err(classify_deadline(&signal, &display_url, timeout)),
    }
}

/// Call `tools/call` on an MCP server and return the result.
///
/// Creates a fresh Streamable HTTP transport per call, same
/// pattern as [`list_tools_with_forwarded_headers`]. Session reuse deferred to MCP
/// Foundation PR 5.
///
/// # Errors
///
/// Returns [`McpClientError`] on connection failure, timeout, or
/// tool execution failure.
#[expect(
    clippy::too_many_arguments,
    reason = "callout replaces the prior allow_loopback param"
)]
#[cfg(test)]
pub(crate) async fn call_tool(
    server_url: &str,
    headers: Option<&serde_json::Value>,
    authorization: Option<&str>,
    tool_name: &str,
    arguments: serde_json::Value,
    timeout: Duration,
    max_result_bytes: usize,
    callout: &McpCallout,
) -> Result<rmcp::model::CallToolResult, McpClientError> {
    call_tool_with_forwarded_headers(
        server_url,
        headers,
        authorization,
        &[],
        None,
        None,
        tool_name,
        arguments,
        timeout,
        max_result_bytes,
        callout,
    )
    .await
}

/// Call `tools/call` with an additional trusted, operator-allowlisted header
/// set. Forwarded values override same-named client tool-entry headers.
#[expect(
    clippy::too_many_arguments,
    reason = "trusted forwarded headers extend the existing API"
)]
#[expect(clippy::too_many_lines, reason = "transport setup + call follows list_tools pattern")]
pub(crate) async fn call_tool_with_forwarded_headers(
    server_url: &str,
    headers: Option<&serde_json::Value>,
    authorization: Option<&str>,
    forwarded_header_names: &[http::HeaderName],
    forwarded_headers: Option<&http::HeaderMap>,
    connector_context: Option<&McpConnectorContext<'_>>,
    tool_name: &str,
    arguments: serde_json::Value,
    timeout: Duration,
    max_result_bytes: usize,
    callout: &McpCallout,
) -> Result<rmcp::model::CallToolResult, McpClientError> {
    // No upfront SSRF classifier: the subrequest transport validates the
    // dial target during the callout via `prepare_url_target`, so this path
    // resolves DNS exactly once. `initialize` uses the control ceiling; the
    // `tools/call` result is bounded to the configured `max_result_bytes` cap
    // (expanded for worst-case JSON string escaping) before deserialization.
    let display_url = parse_display_url(server_url);
    let mcp_client = subrequest_transport::McpSubrequestClient::for_tool(
        callout.clone(),
        timeout,
        max_result_bytes,
        connector_context.map(|context| context.owner.clone()),
    );
    // Take the signal handle before the client is moved into the rmcp
    // transport, so a `ResponseTooLarge` or `SsrfBlocked` classification
    // recorded during the exchange (which rmcp otherwise discards) can be
    // read back below.
    let signal = mcp_client.signal_handle();
    let transport = StreamableHttpClientTransport::with_client(
        mcp_client,
        build_transport_config_with_forwarded_headers(
            server_url,
            headers,
            authorization,
            forwarded_header_names,
            forwarded_headers,
            connector_context,
        )?,
    );

    let mut running: Option<RunningService<RoleClient, ()>> = None;
    let outcome = tokio::time::timeout(timeout, async {
        let client = Box::pin(().serve(transport)).await.map_err(|_source| {
            transport_signal_error(&signal, &display_url).unwrap_or_else(|| McpClientError::Connection {
                url: display_url.clone(),
            })
        })?;
        let client = running.insert(client);

        let parsed_args = match arguments {
            serde_json::Value::Object(obj) => Some(obj),
            serde_json::Value::String(s) => serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(&s).ok(),
            _ => None,
        };
        let mut params = CallToolRequestParams::new(tool_name.to_owned());
        if let Some(args_obj) = parsed_args {
            params = params.with_arguments(args_obj);
        }

        Box::pin(client.call_tool(params)).await.map_err(|_source| {
            transport_signal_error(&signal, &display_url).unwrap_or_else(|| McpClientError::CallTool {
                url: display_url.clone(),
                tool_name: tool_name.to_owned(),
            })
        })
    })
    .await;

    // Close (not drop) the service on every post-serve exit so no background
    // worker task is left holding our subrequest executor. A pre-serve
    // initialization failure owns no `RunningService`; dropping the failed
    // `serve` future drops its transport without starting the service worker.
    if let Some(mut client) = running {
        drop(client.close().await);
    }

    match outcome {
        Ok(result) => result,
        Err(_elapsed) => Err(classify_deadline(&signal, &display_url, timeout)),
    }
}

/// Cap on pagination rounds to prevent infinite loops from
/// servers returning empty pages with valid cursors.
const MAX_PAGES: usize = 100;

/// Paginate `tools/list`, bounded by [`MAX_LISTING_RESPONSE_BYTES`]
/// (cumulative decoded size), `max_tools` (count), and [`MAX_PAGES`].
///
/// Each page body is already capped to the control-response ceiling
/// before deserialization, but a server can still return many
/// near-ceiling pages; the cumulative byte budget bounds that
/// amplification independently of the tool count.
async fn paginate_tools(
    client: &Peer<RoleClient>,
    max_tools: usize,
    url: &McpDisplayUrl,
) -> Result<Vec<rmcp::model::Tool>, McpClientError> {
    let mut all_tools = Vec::new();
    let mut total_bytes: usize = 0;
    let mut cursor = None;
    for _ in 0..MAX_PAGES {
        let params = PaginatedRequestParams::default().with_cursor(cursor);
        let page = Box::pin(client.list_tools(Some(params)))
            .await
            .map_err(|_source| McpClientError::ListTools { url: url.clone() })?;
        accumulate_listing_bytes(&mut total_bytes, &page.tools, url)?;
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

/// Classify a step deadline: a recorded transport signal (e.g. an oversized
/// response, surfaced only when the outer deadline fired) becomes its typed
/// error (413 for a size breach); otherwise a generic timeout.
fn classify_deadline(
    signal: &std::sync::Arc<std::sync::OnceLock<subrequest_transport::TransportSignal>>,
    url: &McpDisplayUrl,
    timeout: Duration,
) -> McpClientError {
    transport_signal_error(signal, url).unwrap_or_else(|| McpClientError::Timeout {
        url: url.clone(),
        timeout,
    })
}

/// Build transport config from server URL, optional headers, and
/// optional `OAuth` authorization token.
///
/// # Errors
///
/// Returns [`McpClientError::InvalidAuthorization`] if the token
/// contains characters invalid in HTTP header values.
#[cfg(test)]
fn build_transport_config(
    server_url: &str,
    headers: Option<&serde_json::Value>,
    authorization: Option<&str>,
) -> Result<StreamableHttpClientTransportConfig, McpClientError> {
    build_transport_config_with_forwarded_headers(server_url, headers, authorization, &[], None, None)
}

/// Build transport config and overlay trusted, operator-allowlisted headers.
///
/// Every configured forwarded name is removed from client tool-entry headers
/// even when no trusted value is available or the target is a direct URL. This
/// prevents client-controlled headers from impersonating ambient identity at a
/// connector endpoint reached through an equivalent direct URL.
#[expect(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "target config, trusted overlays, and scoped context form one security boundary"
)]
fn build_transport_config_with_forwarded_headers(
    server_url: &str,
    headers: Option<&serde_json::Value>,
    authorization: Option<&str>,
    forwarded_header_names: &[http::HeaderName],
    forwarded_headers: Option<&http::HeaderMap>,
    connector_context: Option<&McpConnectorContext<'_>>,
) -> Result<StreamableHttpClientTransportConfig, McpClientError> {
    let mut config = StreamableHttpClientTransportConfig::with_uri(server_url);
    // Bound consecutive *failed* re-dials so a dead endpoint cannot drive an
    // unbounded reconnect storm. This does NOT bound a re-breaching-but-reachable
    // stream (retry_times resets to 0 on every successful reconnect); that case is
    // bounded by the caller's outer deadline + the 413 side-channel (see #1244,
    // and the F4b caller restructure). `ExponentialBackoff` is `#[non_exhaustive]`,
    // so construct via `default()` then set the public field.
    let mut backoff = rmcp::transport::common::client_side_sse::ExponentialBackoff::default();
    backoff.max_times = Some(3);
    config.retry_config = std::sync::Arc::new(backoff);
    let mut header_map = HashMap::new();

    if let Some(headers_obj) = headers.and_then(serde_json::Value::as_object) {
        let nominated = connection_nominated_from_json(headers_obj);
        for (key, value) in headers_obj {
            if let Some(value_str) = value.as_str()
                && let Ok(name) = key.parse::<http::HeaderName>()
                && !is_blocked_mcp_header(&name)
                && !forwarded_header_names.contains(&name)
                && !nominated.contains(&name)
                && let Ok(val) = http::HeaderValue::from_str(value_str)
            {
                header_map.insert(name, val);
            }
        }
    }

    if let Some(forwarded_headers) = forwarded_headers {
        for name in forwarded_headers.keys() {
            if is_blocked_mcp_header(name) {
                continue;
            }
            if let Some(value) = forwarded_headers.get(name) {
                header_map.insert(name.clone(), value.clone());
            }
        }
    }

    let effective_authorization = connector_context
        .and_then(|context| context.bearer)
        .map(SecretString::expose_secret)
        .or(authorization);
    inject_authorization(&mut header_map, effective_authorization)?;
    inject_connector_assertion(&mut header_map, connector_context)?;

    if !header_map.is_empty() {
        config = config.custom_headers(header_map);
    }

    Ok(config)
}

/// Inject the opaque Gateway assertion after every client/operator header merge.
///
/// `is_blocked_mcp_header` remains unchanged, so neither client tool-entry
/// headers nor `forward_headers` can inject or shadow this fixed field.
fn inject_connector_assertion(
    header_map: &mut HashMap<http::HeaderName, http::HeaderValue>,
    connector_context: Option<&McpConnectorContext<'_>>,
) -> Result<(), McpClientError> {
    let Some(assertion) = connector_context.and_then(|context| context.assertion) else {
        return Ok(());
    };
    let value = http::HeaderValue::from_str(assertion.expose_secret())
        .map_err(|_invalid| McpClientError::InvalidAuthorization)?;
    header_map.insert(crate::callout_credentials::MCP_AUTHORIZED_HEADER, value);
    Ok(())
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

/// Credential-safe reason attached to every SSRF rejection.
///
/// The subrequest transport's [`ssrf_validate`](subrequest_transport) hook and
/// the cache-hit [`validate_mcp_target`] check both surface this string on an
/// SSRF rejection, so a blocked literal and a blocked DNS-resolved address read
/// identically to the client without echoing which address matched.
const SSRF_BLOCK_REASON: &str =
    "address is a private, loopback, link-local, unique-local, unspecified, or cloud-metadata range";

/// Field names listed by any `Connection` value in MCP tool-config headers.
fn connection_nominated_from_json(
    headers_obj: &serde_json::Map<String, serde_json::Value>,
) -> HashSet<http::HeaderName> {
    let mut nominated = HashSet::new();
    for (key, value) in headers_obj {
        let Ok(name) = key.parse::<http::HeaderName>() else {
            continue;
        };
        if name != http::header::CONNECTION {
            continue;
        }
        let Some(value) = value.as_str() else {
            continue;
        };
        for token in crate::http_hop::connection_tokens(value) {
            if let Ok(nominated_name) = token.parse::<http::HeaderName>() {
                nominated.insert(nominated_name);
            }
        }
    }
    nominated
}

/// Headers that must not pass through from client-supplied MCP
/// tool config into the proxy's outbound MCP transport.
pub(crate) fn is_blocked_mcp_header(name: &http::HeaderName) -> bool {
    if crate::http_hop::is_hop_by_hop(name.as_str()) {
        return true;
    }
    if matches!(
        *name,
        http::header::AUTHORIZATION
            | http::header::CONTENT_LENGTH
            | http::header::COOKIE
            | http::header::FORWARDED
            | http::header::HOST
            | http::header::SET_COOKIE
    ) {
        return true;
    }
    let s = name.as_str();
    s.starts_with("x-forwarded-") || s.starts_with("x-praxis-") || s.starts_with("x-mcp-") || s.starts_with("x-a2a-")
}

/// Whether `v4` matches a known cloud instance-metadata endpoint that the
/// generic loopback, link-local, and unspecified checks miss.
fn is_cloud_metadata_ipv4(v4: Ipv4Addr) -> bool {
    CLOUD_METADATA_IPV4.contains(&v4)
}

/// Addresses refused as MCP dial targets even when the operator has enabled
/// private upstreams (`allow_private`): the unspecified address, link-local
/// ranges (which include the cloud instance-metadata endpoints), the known
/// cloud-metadata IPv4 endpoints, and IPv6 unique-local/site-local.
///
/// Loopback and the RFC1918/CGNAT private ranges are deliberately *not* here —
/// those are gated on `allow_private` by [`is_ssrf_blocked_ip`], so an operator
/// can opt into reaching them.
fn is_always_sensitive(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => v4.is_link_local() || v4.is_unspecified() || is_cloud_metadata_ipv4(*v4),
        IpAddr::V6(v6) => {
            let [a, b, ..] = v6.octets();
            v6.is_unspecified() || (a == 0xFE && (b & 0xC0) == 0x80) || (a & 0xFE) == 0xFC
        },
    }
}

/// Whether an MCP dial target IP must be refused under the SSRF policy.
///
/// Two tiers:
/// - [`is_always_sensitive`] addresses (link-local/metadata, unspecified, IPv6 unique-local) are refused
///   unconditionally, even with `allow_private`.
/// - The remaining private ranges — loopback, RFC1918, CGNAT (`100.64.0.0/10`), and `0.0.0.0/8` — are refused only when
///   `allow_private` is `false`. This matches the policy the filtered-subrequest executor applies to DNS-resolved
///   hostnames, closing the gap where a pinned literal address (which the executor's `resolve_address_checked`
///   short-circuits) would otherwise reach an RFC1918 host with private upstreams disabled.
///
/// IPv4-mapped IPv6 addresses are normalized first so a mapped private address
/// cannot slip past either tier.
fn is_ssrf_blocked_ip(ip: &IpAddr, allow_private: bool) -> bool {
    let ip = praxis_core::connectivity::normalize_mapped_ipv4(*ip);
    if is_always_sensitive(&ip) {
        return true;
    }
    !allow_private && praxis_core::connectivity::is_private_ip(&ip)
}
/// Convert `rmcp::model::Tool` values to opaque JSON.
fn tools_to_json(tools: Vec<rmcp::model::Tool>) -> Result<Vec<serde_json::Value>, McpClientError> {
    tools
        .into_iter()
        .map(|tool| serde_json::to_value(tool).map_err(McpClientError::Serialization))
        .collect()
}

/// `io::Write` sink that counts bytes written instead of retaining
/// them, so a page's serialized size can be measured without a second
/// heap buffer.
#[derive(Default)]
struct ByteCounter {
    /// Total number of bytes observed.
    count: usize,
}

impl std::io::Write for ByteCounter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.count = self.count.saturating_add(buf.len());
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Measure the JSON-serialized byte size of one decoded `tools/list`
/// page without allocating a second buffer. A serialization error
/// (which the caller would surface elsewhere) yields the bytes counted
/// so far, keeping the cumulative budget conservative.
fn measure_tools_json_bytes(tools: &[rmcp::model::Tool]) -> usize {
    let mut counter = ByteCounter::default();
    match serde_json::to_writer(&mut counter, tools) {
        Ok(()) | Err(_) => counter.count,
    }
}

/// Add one page's decoded size to the running listing total and reject the
/// whole operation once it crosses [`MAX_LISTING_RESPONSE_BYTES`], so
/// pagination cannot multiply the per-page ceiling.
fn accumulate_listing_bytes(
    total_bytes: &mut usize,
    tools: &[rmcp::model::Tool],
    url: &McpDisplayUrl,
) -> Result<(), McpClientError> {
    *total_bytes = total_bytes.saturating_add(measure_tools_json_bytes(tools));
    if *total_bytes > MAX_LISTING_RESPONSE_BYTES {
        return Err(McpClientError::ListingTooLarge {
            url: url.clone(),
            bytes: *total_bytes,
            max: MAX_LISTING_RESPONSE_BYTES,
        });
    }
    Ok(())
}
