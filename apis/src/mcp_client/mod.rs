// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! MCP client wrapper for calling upstream MCP servers.
//!
//! Thin layer over `rmcp` that exposes [`list_tools_with_forwarded_headers`] for resolving
//! MCP tool declarations. Designed for reuse by `mcp_tool` (#27)
//! when `call_tool` support is added.

mod session_pool;
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
    sync::{Arc, OnceLock},
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
use self::{session_pool::PooledSession, subrequest_transport::MAX_CONTROL_RESPONSE_BYTES};
pub(crate) use self::{
    session_pool::{McpPoolKey, McpPoolNamespace, McpSessionPool},
    subrequest_transport::{
        McpCallout, bind_mcp_outbound_chain, build_bare_outbound_pipeline, transport_signal_error, validate_mcp_target,
    },
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

/// Per-call ceilings reserved by the Responses request-wide budget before a
/// budgeted MCP side effect. Budgeted calls use fresh sessions so an older
/// pooled transport cannot retain a less restrictive response ceiling.
#[derive(Clone, Copy)]
pub(crate) struct McpBudgetedCallLimits {
    /// Maximum raw bytes in one JSON-RPC response or SSE operation.
    pub wire_limit: usize,
    /// Maximum lexical wire and JSON structure charge before rmcp parses it.
    pub parse_charge_limit: usize,
}

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
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "keeps the unbudgeted MCP listing entry point available")
)]
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
    list_tools_with_forwarded_headers_budgeted(
        server_url,
        headers,
        authorization,
        forwarded_header_names,
        forwarded_headers,
        connector_context,
        timeout,
        max_tools,
        callout,
        None,
    )
    .await
}

/// Resolve an MCP listing with the request's reserved wire, parser, and
/// cumulative decoded-listing ceilings. `None` preserves the legacy limits.
#[expect(clippy::too_many_arguments, reason = "MCP callout parameters plus request budget")]
#[expect(clippy::too_many_lines, reason = "transport setup and bounded listing are linear")]
pub(crate) async fn list_tools_with_forwarded_headers_budgeted(
    server_url: &str,
    headers: Option<&serde_json::Value>,
    authorization: Option<&str>,
    forwarded_header_names: &[http::HeaderName],
    forwarded_headers: Option<&http::HeaderMap>,
    connector_context: Option<&McpConnectorContext<'_>>,
    timeout: Duration,
    max_tools: usize,
    callout: &McpCallout,
    budget: Option<(McpBudgetedCallLimits, usize)>,
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
    let owner = connector_context.map(|context| context.owner.clone());
    let mcp_client = if let Some((limits, _)) = budget {
        subrequest_transport::McpSubrequestClient::control_with_budget(callout.clone(), timeout, owner, limits)
    } else {
        subrequest_transport::McpSubrequestClient::control(callout.clone(), timeout, owner)
    };
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
        let listing_limit = budget.map_or(MAX_LISTING_RESPONSE_BYTES, |(_, limit)| {
            limit.min(MAX_LISTING_RESPONSE_BYTES)
        });
        let tools = Box::pin(paginate_tools(client, max_tools, listing_limit, &display_url))
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
        None,
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

/// Open and initialize a fresh MCP session for a `tools/call` target.
///
/// Builds the subrequest-backed rmcp client, captures its signal handle before
/// the client is moved into the transport (so a `ResponseTooLarge`/`SsrfBlocked`
/// classification recorded during the exchange — which rmcp otherwise discards —
/// can be read back), and runs the `initialize` handshake. No upfront SSRF
/// classifier: the subrequest transport validates the dial target during the
/// callout via `prepare_url_target`, so this path resolves DNS exactly once.
///
/// This does not apply a timeout; the caller bounds the handshake.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors the tool-call boundary's forwarded-header + connector context inputs"
)]
async fn open_tool_session(
    server_url: &str,
    headers: Option<&serde_json::Value>,
    authorization: Option<&str>,
    forwarded_header_names: &[http::HeaderName],
    forwarded_headers: Option<&http::HeaderMap>,
    connector_context: Option<&McpConnectorContext<'_>>,
    timeout: Duration,
    max_result_bytes: usize,
    budgeted_limits: Option<McpBudgetedCallLimits>,
    callout: &McpCallout,
    display_url: &McpDisplayUrl,
) -> Result<PooledSession, McpClientError> {
    let mcp_client = subrequest_transport::McpSubrequestClient::for_tool_with_budget(
        callout.clone(),
        timeout,
        max_result_bytes,
        budgeted_limits,
        connector_context.map(|context| context.owner.clone()),
    );
    let signal = mcp_client.signal_handle();
    let signal_state = mcp_client.signal_state();
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
    // A pre-serve initialization failure owns no `RunningService`; dropping the
    // failed `serve` future drops its transport without starting the worker.
    let service = Box::pin(().serve(transport)).await.map_err(|_source| {
        transport_signal_error(&signal, display_url).unwrap_or_else(|| McpClientError::Connection {
            url: display_url.clone(),
        })
    })?;
    signal_state.finish_exchange(&signal);
    Ok(PooledSession::new(service, signal_state, max_result_bytes))
}

/// Issue one `tools/call` on an already-initialized session without closing it.
///
/// `initialize` uses the control ceiling; the `tools/call` result is bounded to
/// the configured `max_result_bytes` cap (expanded for worst-case JSON string
/// escaping) before deserialization, applied inside the session's transport.
async fn invoke_tool(
    session: &PooledSession,
    signal: &Arc<OnceLock<subrequest_transport::TransportSignal>>,
    tool_name: &str,
    arguments: serde_json::Value,
    display_url: &McpDisplayUrl,
) -> Result<rmcp::model::CallToolResult, McpClientError> {
    let parsed_args = match arguments {
        serde_json::Value::Object(obj) => Some(obj),
        serde_json::Value::String(s) => serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(&s).ok(),
        _ => None,
    };
    let mut params = CallToolRequestParams::new(tool_name.to_owned());
    if let Some(args_obj) = parsed_args {
        params = params.with_arguments(args_obj);
    }
    Box::pin(session.service().call_tool(params)).await.map_err(|_source| {
        transport_signal_error(signal, display_url).unwrap_or_else(|| McpClientError::CallTool {
            url: display_url.clone(),
            tool_name: tool_name.to_owned(),
        })
    })
}

/// Call `tools/call` with an additional trusted, operator-allowlisted header
/// set, optionally reusing an initialized session from `pool` (#1019).
///
/// Forwarded values override same-named client tool-entry headers.
///
/// When `pool` is `Some((pool, key))`, a warm session for
/// that exact identity is reused (skipping the handshake) and returned to the
/// pool after a clean call. A failure on a *reused* session evicts (closes) it
/// and surfaces the error **without** a fresh retry: rmcp already reinitializes
/// and retries a server-invalidated (404 `SessionExpired`) session transparently,
/// so any error observed here is a genuine failure of unknown delivery — a
/// timeout or 5xx cannot prove the tool was not already executed, and blindly
/// retrying would risk duplicating a non-idempotent side effect. The single
/// reused attempt is bounded by one `timeout`, so a logical call never exceeds
/// its configured deadline. A restarted server that rejects the old session
/// with a non-404 status therefore surfaces an error rather than opening a fresh
/// session automatically. Fresh sessions are pooled on success and closed on any
/// error; a `None` pool never reuses or retains a session.
#[expect(
    clippy::too_many_arguments,
    reason = "trusted forwarded headers and optional pooling extend the existing API"
)]
pub(crate) async fn call_tool_with_forwarded_headers(
    pool: Option<(&McpSessionPool, &McpPoolKey)>,
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
    call_tool_with_limits(
        pool,
        server_url,
        headers,
        authorization,
        forwarded_header_names,
        forwarded_headers,
        connector_context,
        tool_name,
        arguments,
        timeout,
        max_result_bytes,
        callout,
        None,
    )
    .await
}

/// Execute a budgeted call with transport limits reserved by the Responses
/// owner. The fresh session keeps a pooled transport's older, larger ceilings
/// from bypassing this request's limit.
#[expect(clippy::too_many_arguments, reason = "mirrors the unbudgeted MCP call boundary")]
pub(crate) async fn call_tool_with_forwarded_headers_budgeted(
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
    limits: McpBudgetedCallLimits,
) -> Result<rmcp::model::CallToolResult, McpClientError> {
    call_tool_with_limits(
        None,
        server_url,
        headers,
        authorization,
        forwarded_header_names,
        forwarded_headers,
        connector_context,
        tool_name,
        arguments,
        timeout,
        max_result_bytes,
        callout,
        Some(limits),
    )
    .await
}

/// Execute the common pooled or fresh MCP session path with optional budgeted
/// transport bounds. Only unbudgeted callers can reuse the session pool.
#[expect(
    clippy::too_many_arguments,
    reason = "shared budgeted and unbudgeted MCP call boundary"
)]
#[expect(
    clippy::too_many_lines,
    reason = "session reuse and fresh session cleanup share one boundary"
)]
#[expect(
    clippy::large_stack_frames,
    reason = "rmcp service setup and cleanup remain in one async owner"
)]
async fn call_tool_with_limits(
    pool: Option<(&McpSessionPool, &McpPoolKey)>,
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
    budgeted_limits: Option<McpBudgetedCallLimits>,
) -> Result<rmcp::model::CallToolResult, McpClientError> {
    let display_url = parse_display_url(server_url);
    let deadline = tokio::time::Instant::now() + timeout;
    // 1. Reuse a warm session for this exact identity, if one exists. A reused session only ever existed after a prior
    //    clean success. Checkout rejects closed, expired, or limit-mismatched sessions before any request is sent;
    //    those sessions are explicitly closed in the background so their DELETE cannot consume this call's delivery
    //    deadline, and a miss safely falls through to a fresh open. Each attempt installs a fresh transport signal so
    //    an idle GET-stream failure cannot poison this call.
    if let Some((pool, key)) = pool.filter(|_| budgeted_limits.is_none()) {
        let checkout = pool.checkout(key, max_result_bytes);
        session_pool::close_sessions_in_background(checkout.rejected);
        if let Some(session) = checkout.session {
            let signal = session.begin_call();
            let outcome = tokio::time::timeout_at(
                deadline,
                invoke_tool(&session, &signal, tool_name, arguments, &display_url),
            )
            .await;
            session.finish_call(&signal);
            return match outcome {
                Ok(Ok(result)) => {
                    let rejected = pool.checkin(key.clone(), session);
                    session_pool::close_sessions_in_background(rejected);
                    Ok(result)
                },
                // A reused session's failure is evicted, never retried: rmcp already transparently reinitializes a 404
                // `SessionExpired` session, so any error surfaced here is genuine and of unknown delivery — retrying a
                // timeout or 5xx could execute a non-idempotent tool twice (at-most-once for the reused path).
                Ok(Err(err)) => {
                    session.close_before(deadline).await;
                    Err(err)
                },
                Err(_elapsed) => {
                    // Read the signal (an oversized/SSRF exchange surfaced only when the deadline fired) before
                    // closing.
                    session.close_before(deadline).await;
                    Err(classify_deadline(&signal, &display_url, timeout))
                },
            };
        }
    }

    // 2. Fresh session: first use, a `None` pool, or the empty-fingerprint sentinel. Close (not drop) the service on
    //    every post-serve exit so no background worker task is left holding our subrequest executor.
    let mut session: Option<PooledSession> = None;
    let mut call_signal = None;
    let outcome = tokio::time::timeout_at(deadline, async {
        let opened = open_tool_session(
            server_url,
            headers,
            authorization,
            forwarded_header_names,
            forwarded_headers,
            connector_context,
            timeout,
            max_result_bytes,
            budgeted_limits,
            callout,
            &display_url,
        )
        .await?;
        let opened = session.insert(opened);
        let signal = opened.begin_call();
        call_signal = Some(Arc::clone(&signal));
        invoke_tool(opened, &signal, tool_name, arguments, &display_url).await
    })
    .await;

    if let (Some(session), Some(signal)) = (session.as_ref(), call_signal.as_ref()) {
        session.finish_call(signal);
    }

    match outcome {
        Ok(Ok(result)) => {
            match (pool.filter(|_| budgeted_limits.is_none()), session.take()) {
                (Some((pool, key)), Some(session)) => {
                    let rejected = pool.checkin(key.clone(), session);
                    session_pool::close_sessions_in_background(rejected);
                },
                (None, Some(session)) => close_fresh_session(session, deadline, budgeted_limits).await,
                (_, None) => {},
            }
            Ok(result)
        },
        Ok(Err(err)) => {
            if let Some(session) = session.take() {
                close_fresh_session(session, deadline, budgeted_limits).await;
            }
            Err(err)
        },
        Err(_elapsed) => {
            // Read the signal (if the handshake completed) before closing, so an
            // oversized/SSRF exchange surfaced only when the deadline fired still
            // maps to its typed error.
            if let Some(session) = session.take() {
                close_fresh_session(session, deadline, budgeted_limits).await;
            }
            Err(call_signal.map_or_else(
                || McpClientError::Timeout {
                    url: display_url.clone(),
                    timeout,
                },
                |signal| classify_deadline(&signal, &display_url, timeout),
            ))
        },
    }
}

/// A budgeted result can release its callout reservation only after rmcp's
/// worker has stopped retaining the transport and any DELETE response. Normal
/// unbudgeted calls keep their existing bounded close behavior.
async fn close_fresh_session(
    session: PooledSession,
    deadline: tokio::time::Instant,
    budgeted_limits: Option<McpBudgetedCallLimits>,
) {
    if budgeted_limits.is_some() {
        session.close_budgeted().await;
    } else {
        session.close_before(deadline).await;
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
    max_listing_bytes: usize,
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
        accumulate_listing_bytes(&mut total_bytes, &page.tools, max_listing_bytes, url)?;
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
    signal: &Arc<OnceLock<subrequest_transport::TransportSignal>>,
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
    config.retry_config = Arc::new(backoff);
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
    max_listing_bytes: usize,
    url: &McpDisplayUrl,
) -> Result<(), McpClientError> {
    *total_bytes = total_bytes.saturating_add(measure_tools_json_bytes(tools));
    if *total_bytes > max_listing_bytes {
        return Err(McpClientError::ListingTooLarge {
            url: url.clone(),
            bytes: *total_bytes,
            max: max_listing_bytes,
        });
    }
    Ok(())
}
