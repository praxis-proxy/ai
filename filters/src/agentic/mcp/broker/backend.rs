// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! MCP backend session callouts for the current-profile broker.
//!
//! [`McpBackendClient`] isolates the backend session lifecycle exchanges the
//! broker issues itself (`initialize`, `notifications/initialized`, and
//! session `DELETE`). `tools/call` is not a callout: once a backend session
//! exists the broker forwards the client's request through the main proxy
//! path so streaming and every main-chain policy still apply.
//!
//! [`FilteredMcpBackendClient`] runs each exchange as a filtered subrequest
//! through the broker's generated outbound chain. The executor bounds the
//! buffered response before materializing it, so oversized and chunked
//! handshake replies are rejected without retaining the complete body.

use std::{
    fmt,
    sync::Arc,
    time::{Duration, Instant},
};

use async_trait::async_trait;
use bytes::Bytes;
use http::{HeaderMap, HeaderName, HeaderValue, Method};
use praxis_core::subrequest::SubRequestClient;
use praxis_filter::{
    CalloutOutcome, CalloutResponse, FilterPipeline, FilteredSubrequestExecutor, RequestExtensions, SubRequest,
    SubResponse, SubrequestRuntime, TraceContext,
};
use tracing::debug;

use super::selector::BrokerBackendCluster;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Prefix of the JSON-RPC id the broker uses for its own backend `initialize`.
const INITIALIZE_REQUEST_ID_PREFIX: &str = "praxis-broker-initialize-";

/// Client name the broker reports to backends.
const CLIENT_NAME: &str = "praxis";

/// Client version the broker reports to backends.
const CLIENT_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Maximum accepted backend `Mcp-Session-Id` length.
const MAX_SESSION_ID_LEN: usize = 256;

/// Maximum accepted backend protocol version length.
const MAX_PROTOCOL_VERSION_LEN: usize = 64;

/// MCP session header.
pub(crate) const MCP_SESSION_ID: HeaderName = HeaderName::from_static("mcp-session-id");

/// MCP protocol version header.
pub(crate) const MCP_PROTOCOL_VERSION: HeaderName = HeaderName::from_static("mcp-protocol-version");

// -----------------------------------------------------------------------------
// Types
// -----------------------------------------------------------------------------

/// The backend MCP server a lifecycle exchange targets.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BackendTarget<'a> {
    /// Top-level cluster the handshake chain dials.
    pub cluster: &'a Arc<str>,
    /// Backend MCP endpoint path.
    pub path: &'a str,
}

/// Request-scoped resources one backend exchange runs with.
#[derive(Clone)]
pub(crate) struct BackendCallout {
    /// Shared sub-request transport.
    pub client: SubRequestClient,
    /// Absolute deadline for the exchange.
    pub deadline: Instant,
    /// Current sub-request nesting depth.
    pub depth: u8,
    /// Downstream client attributes.
    pub downstream: SubrequestRuntime,
    /// Operator-allowlisted client headers copied onto the exchange.
    pub forwarded_headers: HeaderMap,
    /// Request correlation propagated into the exchange.
    pub trace_context: Option<TraceContext>,
}

/// Result of a successful backend `initialize`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct BackendInit {
    /// Protocol version the backend selected.
    pub protocol_version: Box<str>,
    /// Backend `Mcp-Session-Id`; `None` for a backend that issues no session.
    pub session_id: Option<HeaderValue>,
    /// Endpoint address that created the session. Later exchanges and the
    /// forwarded calls are pinned to it, because a backend session exists
    /// only on the endpoint that issued it. `None` when unknown.
    pub endpoint: Option<Arc<str>>,
}

/// Structured backend lifecycle failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum McpBackendError {
    /// The exchange exceeded its deadline.
    Timeout,
    /// The exchange could not be delivered or no transport is available.
    Transport,
    /// The backend response exceeded the configured ceiling.
    ResponseTooLarge {
        /// Ceiling that was exceeded.
        limit: usize,
    },
    /// The backend answered with a non-success HTTP status.
    Rejected {
        /// HTTP status returned by the backend.
        status: u16,
    },
    /// The backend answered `initialize` with a JSON-RPC error.
    JsonRpc {
        /// JSON-RPC error code.
        code: i64,
    },
    /// The backend response is not a valid MCP lifecycle reply.
    Malformed(&'static str),
    /// The backend negotiated a protocol version the broker does not support.
    UnsupportedVersion,
    /// Every session slot is busy with an in-flight exchange.
    Capacity,
    /// The gateway session was deleted while its handshake ran.
    SessionClosed,
}

impl McpBackendError {
    /// Stable, credential-free reason label for responses and logs.
    pub(crate) fn reason(&self) -> &'static str {
        match self {
            Self::Timeout => "timeout",
            Self::Transport => "transport",
            Self::ResponseTooLarge { .. } => "response_too_large",
            Self::Rejected { .. } => "rejected",
            Self::JsonRpc { .. } => "json_rpc_error",
            Self::Malformed(_) => "malformed_response",
            Self::UnsupportedVersion => "unsupported_protocol_version",
            Self::Capacity => "session_capacity",
            Self::SessionClosed => "session_closed",
        }
    }
}

impl fmt::Display for McpBackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Timeout => f.write_str("backend exchange timed out"),
            Self::Transport => f.write_str("backend exchange failed"),
            Self::ResponseTooLarge { limit } => write!(f, "backend response exceeded {limit} bytes"),
            Self::Rejected { status } => write!(f, "backend rejected the exchange with HTTP {status}"),
            Self::JsonRpc { code } => write!(f, "backend returned JSON-RPC error {code}"),
            Self::Malformed(detail) => write!(f, "malformed backend response: {detail}"),
            Self::UnsupportedVersion => f.write_str("backend negotiated an unsupported protocol version"),
            Self::Capacity => f.write_str("backend session capacity exhausted"),
            Self::SessionClosed => f.write_str("gateway session closed during the backend handshake"),
        }
    }
}

// -----------------------------------------------------------------------------
// McpBackendClient
// -----------------------------------------------------------------------------

/// Backend MCP session lifecycle callouts.
///
/// Implementations must bound every response they read and keep timeout,
/// transport, protocol, malformed-reply, and oversized-reply failures
/// distinct.
#[async_trait]
pub(crate) trait McpBackendClient: Send + Sync {
    /// Send `initialize` and capture the backend session.
    async fn initialize(
        &self,
        target: BackendTarget<'_>,
        protocol_version: &str,
        callout: &BackendCallout,
    ) -> Result<BackendInit, McpBackendError>;

    /// Send `notifications/initialized` on an initialized session.
    async fn send_initialized(
        &self,
        target: BackendTarget<'_>,
        session: &BackendInit,
        callout: &BackendCallout,
    ) -> Result<(), McpBackendError>;

    /// Terminate a backend session with HTTP `DELETE`.
    async fn terminate(
        &self,
        target: BackendTarget<'_>,
        session: &BackendInit,
        callout: &BackendCallout,
    ) -> Result<(), McpBackendError>;
}

// -----------------------------------------------------------------------------
// FilteredMcpBackendClient
// -----------------------------------------------------------------------------

/// [`McpBackendClient`] backed by filtered subrequests through the broker's
/// generated outbound chain.
pub(crate) struct FilteredMcpBackendClient {
    /// Generated handshake chain: selector, operator filters, load balancer.
    pipeline: Arc<FilterPipeline>,
    /// Buffered response ceiling for each exchange.
    max_response_bytes: usize,
    /// Per-exchange step timeout.
    step_timeout: Duration,
}

impl FilteredMcpBackendClient {
    /// Build a client over the bound handshake chain.
    pub(crate) fn new(pipeline: Arc<FilterPipeline>, max_response_bytes: usize, step_timeout: Duration) -> Self {
        Self {
            pipeline,
            max_response_bytes,
            step_timeout,
        }
    }

    /// Run one exchange, pinned to `pinned_endpoint` when set, and return the
    /// bounded buffered response with the endpoint address that served it.
    #[expect(
        clippy::large_stack_frames,
        reason = "filtered subrequest executor futures are inherently large"
    )]
    async fn exchange(
        &self,
        target: BackendTarget<'_>,
        request: SubRequest,
        callout: &BackendCallout,
        pinned_endpoint: Option<&Arc<str>>,
    ) -> Result<(SubResponse, Option<Arc<str>>), McpBackendError> {
        let staged = BrokerBackendCluster::new(Arc::clone(target.cluster), pinned_endpoint.cloned());
        let dialed = Arc::clone(&staged.dialed);
        let mut extensions = RequestExtensions::default();
        extensions.insert(staged);
        if let Some(trace_context) = callout.trace_context.as_ref() {
            extensions.insert(trace_context.clone());
        }

        let executor = FilteredSubrequestExecutor::for_callout(
            callout.client.clone(),
            callout.downstream.clone(),
            callout.depth,
            self.max_response_bytes,
            self.step_timeout,
        );
        let outcome = Box::pin(executor.run_classified(&self.pipeline, &request, extensions, callout.deadline)).await;
        match outcome {
            Ok(CalloutOutcome::Response(CalloutResponse::Buffered(response))) => Ok((response, dialed.get().cloned())),
            Ok(CalloutOutcome::ResponseTooLarge { limit, .. }) => Err(McpBackendError::ResponseTooLarge { limit }),
            Ok(_) => Err(McpBackendError::Transport),
            Err(error) => {
                debug!(error = %error, "mcp broker backend exchange failed");
                if Instant::now() >= callout.deadline {
                    Err(McpBackendError::Timeout)
                } else {
                    Err(McpBackendError::Transport)
                }
            },
        }
    }
}

#[async_trait]
impl McpBackendClient for FilteredMcpBackendClient {
    async fn initialize(
        &self,
        target: BackendTarget<'_>,
        protocol_version: &str,
        callout: &BackendCallout,
    ) -> Result<BackendInit, McpBackendError> {
        let request_id = initialize_request_id();
        let request = post_request(target, initialize_body(&request_id, protocol_version), callout, None)?;
        let (response, endpoint) = self.exchange(target, request, callout, None).await?;
        let mut init = parse_initialize_response(&response, &request_id)?;
        init.endpoint = endpoint;
        Ok(init)
    }

    async fn send_initialized(
        &self,
        target: BackendTarget<'_>,
        session: &BackendInit,
        callout: &BackendCallout,
    ) -> Result<(), McpBackendError> {
        let body = Bytes::from_static(br#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#);
        let request = post_request(target, body, callout, Some(session))?;
        let (response, _endpoint) = self
            .exchange(target, request, callout, session.endpoint.as_ref())
            .await?;
        expect_success(response.status)
    }

    async fn terminate(
        &self,
        target: BackendTarget<'_>,
        session: &BackendInit,
        callout: &BackendCallout,
    ) -> Result<(), McpBackendError> {
        let Some(session_id) = session.session_id.as_ref() else {
            return Ok(());
        };
        let mut headers = callout.forwarded_headers.clone();
        headers.insert(MCP_SESSION_ID, session_id.clone());
        insert_protocol_version(&mut headers, &session.protocol_version)?;
        let request = SubRequest {
            method: Method::DELETE,
            uri: backend_uri(target)?,
            headers,
            body: Bytes::new(),
        };
        let (response, _endpoint) = self
            .exchange(target, request, callout, session.endpoint.as_ref())
            .await?;
        // 404 and 405 mean the backend already forgot the session or does not
        // support client-initiated termination; both leave nothing to clean up.
        if matches!(response.status, 404 | 405) {
            return Ok(());
        }
        expect_success(response.status)
    }
}

// -----------------------------------------------------------------------------
// Request Builders
// -----------------------------------------------------------------------------

/// Build a JSON-RPC POST to the backend endpoint.
fn post_request(
    target: BackendTarget<'_>,
    body: Bytes,
    callout: &BackendCallout,
    session: Option<&BackendInit>,
) -> Result<SubRequest, McpBackendError> {
    let mut headers = callout.forwarded_headers.clone();
    headers.insert(http::header::CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(
        http::header::ACCEPT,
        HeaderValue::from_static("application/json, text/event-stream"),
    );
    if let Some(session) = session {
        if let Some(session_id) = session.session_id.as_ref() {
            headers.insert(MCP_SESSION_ID, session_id.clone());
        }
        insert_protocol_version(&mut headers, &session.protocol_version)?;
    }
    Ok(SubRequest {
        method: Method::POST,
        uri: backend_uri(target)?,
        headers,
        body,
    })
}

/// Origin-form URI for the backend endpoint.
fn backend_uri(target: BackendTarget<'_>) -> Result<http::Uri, McpBackendError> {
    target
        .path
        .parse()
        .map_err(|_e| McpBackendError::Malformed("invalid backend path"))
}

/// Set the negotiated protocol version on a post-initialize exchange.
fn insert_protocol_version(headers: &mut HeaderMap, version: &str) -> Result<(), McpBackendError> {
    let value = HeaderValue::from_str(version).map_err(|_e| McpBackendError::Malformed("invalid protocol version"))?;
    headers.insert(MCP_PROTOCOL_VERSION, value);
    Ok(())
}

/// A fresh JSON-RPC id for one backend `initialize`.
///
/// Clients choose the ids of the calls the broker forwards on the same backend
/// session, and MCP forbids reusing an id within a session. A random id keeps
/// the broker's own request from colliding with any id a client picks.
fn initialize_request_id() -> String {
    format!("{INITIALIZE_REQUEST_ID_PREFIX}{:032x}", rand::random::<u128>())
}

/// Serialize the broker's backend `initialize` request.
fn initialize_body(request_id: &str, protocol_version: &str) -> Bytes {
    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "initialize",
        "params": {
            "protocolVersion": protocol_version,
            "capabilities": {},
            "clientInfo": { "name": CLIENT_NAME, "version": CLIENT_VERSION },
        },
    });
    Bytes::from(body.to_string())
}

// -----------------------------------------------------------------------------
// Response Parsing
// -----------------------------------------------------------------------------

/// Map a non-2xx status to [`McpBackendError::Rejected`].
fn expect_success(status: u16) -> Result<(), McpBackendError> {
    if (200..300).contains(&status) {
        Ok(())
    } else {
        Err(McpBackendError::Rejected { status })
    }
}

/// Validate an `initialize` reply delivered as JSON or a buffered SSE stream.
pub(crate) fn parse_initialize_response(
    response: &SubResponse,
    request_id: &str,
) -> Result<BackendInit, McpBackendError> {
    expect_success(response.status)?;
    let session_id = backend_session_id(&response.headers)?;
    let is_sse = response
        .headers
        .get(http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(praxis_ai_apis::is_event_stream_content_type);
    let message = if is_sse {
        find_sse_response(&response.body, request_id)?
    } else {
        serde_json::from_slice::<serde_json::Value>(&response.body)
            .map_err(|_e| McpBackendError::Malformed("initialize reply is not JSON"))?
    };
    let protocol_version = initialize_result_version(&message, request_id)?;
    Ok(BackendInit {
        protocol_version,
        session_id,
        endpoint: None,
    })
}

/// Read and validate the backend `Mcp-Session-Id` response header.
fn backend_session_id(headers: &HeaderMap) -> Result<Option<HeaderValue>, McpBackendError> {
    let mut values = headers.get_all(&MCP_SESSION_ID).iter();
    let Some(value) = values.next() else {
        return Ok(None);
    };
    if values.next().is_some() {
        return Err(McpBackendError::Malformed("multiple Mcp-Session-Id headers"));
    }
    let bytes = value.as_bytes();
    if bytes.is_empty() || bytes.len() > MAX_SESSION_ID_LEN || !bytes.iter().all(|b| (0x21..=0x7E).contains(b)) {
        return Err(McpBackendError::Malformed("invalid Mcp-Session-Id"));
    }
    Ok(Some(value.clone()))
}

/// Find the JSON-RPC reply to the broker's `initialize` in an SSE body.
fn find_sse_response(body: &[u8], request_id: &str) -> Result<serde_json::Value, McpBackendError> {
    praxis_ai_apis::buffered_sse_event_data(body)
        .iter()
        .find_map(|data| initialize_reply(data, request_id))
        .ok_or(McpBackendError::Malformed("initialize stream has no reply"))
}

/// Parse one SSE event payload when it answers the broker's `initialize`.
fn initialize_reply(data: &[u8], request_id: &str) -> Option<serde_json::Value> {
    let value: serde_json::Value = serde_json::from_slice(data).ok()?;
    (value.get("id").and_then(serde_json::Value::as_str) == Some(request_id)).then_some(value)
}

/// Extract the negotiated version from an `initialize` JSON-RPC reply.
fn initialize_result_version(message: &serde_json::Value, request_id: &str) -> Result<Box<str>, McpBackendError> {
    if message.get("jsonrpc").and_then(serde_json::Value::as_str) != Some("2.0") {
        return Err(McpBackendError::Malformed("initialize reply is not JSON-RPC 2.0"));
    }
    if message.get("id").and_then(serde_json::Value::as_str) != Some(request_id) {
        return Err(McpBackendError::Malformed("initialize reply id does not match"));
    }
    if let Some(error) = message.get("error") {
        let code = error
            .get("code")
            .and_then(serde_json::Value::as_i64)
            .ok_or(McpBackendError::Malformed("initialize error has no code"))?;
        return Err(McpBackendError::JsonRpc { code });
    }
    let version = message
        .get("result")
        .and_then(|r| r.get("protocolVersion"))
        .and_then(serde_json::Value::as_str)
        .ok_or(McpBackendError::Malformed("initialize result has no protocolVersion"))?;
    if version.is_empty()
        || version.len() > MAX_PROTOCOL_VERSION_LEN
        || HeaderValue::from_str(version).is_err()
        || version.bytes().any(|b| b.is_ascii_control())
    {
        return Err(McpBackendError::Malformed("invalid protocolVersion"));
    }
    Ok(Box::from(version))
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    reason = "tests"
)]
mod tests {
    use super::*;

    /// Request id the tests' `initialize` replies answer.
    const INITIALIZE_REQUEST_ID: &str = "praxis-broker-initialize-test";

    fn parse_initialize_response(response: &SubResponse) -> Result<BackendInit, McpBackendError> {
        super::parse_initialize_response(response, INITIALIZE_REQUEST_ID)
    }

    fn response(status: u16, content_type: &str, session: Option<&str>, body: &str) -> SubResponse {
        let mut headers = HeaderMap::new();
        headers.insert(http::header::CONTENT_TYPE, HeaderValue::from_str(content_type).unwrap());
        if let Some(session) = session {
            headers.insert(MCP_SESSION_ID, HeaderValue::from_str(session).unwrap());
        }
        SubResponse {
            status,
            version: http::Version::HTTP_11,
            headers,
            body: Bytes::from(body.to_owned()),
        }
    }

    fn init_reply(version: &str) -> String {
        format!(r#"{{"jsonrpc":"2.0","id":"{INITIALIZE_REQUEST_ID}","result":{{"protocolVersion":"{version}"}}}}"#)
    }

    #[test]
    fn json_initialize_captures_session_and_version() {
        let init = parse_initialize_response(&response(
            200,
            "application/json",
            Some("s-1"),
            &init_reply("2025-06-18"),
        ))
        .unwrap();
        assert_eq!(init.session_id.unwrap(), "s-1", "session id must be captured");
        assert_eq!(&*init.protocol_version, "2025-06-18", "version must be captured");
    }

    #[test]
    fn sessionless_backend_is_accepted() {
        let init =
            parse_initialize_response(&response(200, "application/json", None, &init_reply("2025-03-26"))).unwrap();
        assert!(init.session_id.is_none(), "backend without a session yields None");
    }

    #[test]
    fn sse_initialize_skips_unrelated_events() {
        let body = format!(
            ": keepalive\r\n\r\nevent: message\r\ndata: {{\"jsonrpc\":\"2.0\",\"method\":\"notifications/message\"}}\r\n\r\ndata: {}\r\n\r\n",
            init_reply("2025-06-18")
        );
        let init = parse_initialize_response(&response(200, "text/event-stream", Some("s-2"), &body)).unwrap();
        assert_eq!(&*init.protocol_version, "2025-06-18", "SSE reply must be found");
    }

    #[test]
    fn sse_initialize_without_terminal_blank_line_is_rejected() {
        let body = format!("data: {}", init_reply("2025-06-18"));
        assert!(
            parse_initialize_response(&response(200, "text/event-stream", Some("s"), &body)).is_err(),
            "an unterminated event is incomplete and must not be accepted"
        );
    }

    #[test]
    fn sse_initialize_accepts_cr_only_framing() {
        let body = format!("event: message\rdata: {}\r\r", init_reply("2025-06-18"));
        let init = parse_initialize_response(&response(200, "text/event-stream", Some("s"), &body)).unwrap();
        assert_eq!(
            &*init.protocol_version, "2025-06-18",
            "bare CR line endings must frame events"
        );
    }

    #[test]
    fn sse_initialize_accepts_leading_bom() {
        let body = format!("\u{feff}data: {}\n\n", init_reply("2025-06-18"));
        let init = parse_initialize_response(&response(200, "text/event-stream", Some("s"), &body)).unwrap();
        assert_eq!(
            &*init.protocol_version, "2025-06-18",
            "a leading BOM must not hide data:"
        );
    }

    #[test]
    fn sse_initialize_without_reply_is_malformed() {
        let body = "data: {\"jsonrpc\":\"2.0\",\"method\":\"ping\"}\n\n";
        assert!(
            matches!(
                parse_initialize_response(&response(200, "text/event-stream", Some("s"), body)),
                Err(McpBackendError::Malformed(_))
            ),
            "stream without reply must be malformed"
        );
    }

    #[test]
    fn non_success_status_is_rejected() {
        assert_eq!(
            parse_initialize_response(&response(401, "application/json", None, "{}")),
            Err(McpBackendError::Rejected { status: 401 }),
            "401 must map to Rejected"
        );
    }

    #[test]
    fn json_rpc_error_is_preserved() {
        let body =
            format!(r#"{{"jsonrpc":"2.0","id":"{INITIALIZE_REQUEST_ID}","error":{{"code":-32602,"message":"x"}}}}"#);
        assert_eq!(
            parse_initialize_response(&response(200, "application/json", None, &body)),
            Err(McpBackendError::JsonRpc { code: -32602 }),
            "JSON-RPC error code must be preserved"
        );
    }

    #[test]
    fn malformed_json_is_distinct() {
        assert_eq!(
            parse_initialize_response(&response(200, "application/json", None, "not json"))
                .unwrap_err()
                .reason(),
            "malformed_response",
            "malformed JSON must be classified"
        );
    }

    #[test]
    fn mismatched_reply_id_is_malformed() {
        let body = r#"{"jsonrpc":"2.0","id":7,"result":{"protocolVersion":"2025-06-18"}}"#;
        assert!(
            matches!(
                parse_initialize_response(&response(200, "application/json", None, body)),
                Err(McpBackendError::Malformed(_))
            ),
            "reply with another id must be rejected"
        );
    }

    #[test]
    fn invalid_session_id_is_rejected() {
        let long = "a".repeat(MAX_SESSION_ID_LEN + 1);
        assert!(
            parse_initialize_response(&response(
                200,
                "application/json",
                Some(&long),
                &init_reply("2025-06-18")
            ))
            .is_err(),
            "oversized session id must be rejected"
        );
        assert!(
            parse_initialize_response(&response(
                200,
                "application/json",
                Some("a b"),
                &init_reply("2025-06-18")
            ))
            .is_err(),
            "session id with space must be rejected"
        );
    }

    #[test]
    fn duplicate_session_headers_are_rejected() {
        let mut resp = response(200, "application/json", Some("a"), &init_reply("2025-06-18"));
        resp.headers.append(MCP_SESSION_ID, HeaderValue::from_static("b"));
        assert!(
            parse_initialize_response(&resp).is_err(),
            "ambiguous session id must be rejected"
        );
    }

    #[test]
    fn missing_protocol_version_is_malformed() {
        let body = format!(r#"{{"jsonrpc":"2.0","id":"{INITIALIZE_REQUEST_ID}","result":{{}}}}"#);
        assert!(
            parse_initialize_response(&response(200, "application/json", None, &body)).is_err(),
            "result without protocolVersion must be rejected"
        );
    }

    #[test]
    fn initialize_body_is_valid_request() {
        let body: serde_json::Value = serde_json::from_slice(&initialize_body("req-1", "2025-06-18")).unwrap();
        assert_eq!(body["method"], "initialize", "method must be initialize");
        assert_eq!(body["id"], "req-1", "the request id must be sent");
        assert_eq!(body["params"]["protocolVersion"], "2025-06-18", "version must be sent");
        assert_eq!(
            body["params"]["clientInfo"]["name"], CLIENT_NAME,
            "client name must be sent"
        );
    }

    #[test]
    fn initialize_request_ids_are_unique() {
        let first = initialize_request_id();
        assert!(
            first.starts_with(INITIALIZE_REQUEST_ID_PREFIX),
            "the id must carry the broker prefix"
        );
        assert_ne!(first, initialize_request_id(), "each handshake must get its own id");
    }

    #[test]
    fn reply_to_another_request_id_is_rejected() {
        let body =
            r#"{"jsonrpc":"2.0","id":"praxis-broker-initialize-other","result":{"protocolVersion":"2025-06-18"}}"#;
        assert!(
            parse_initialize_response(&response(200, "application/json", None, body)).is_err(),
            "a reply must answer this handshake's own id"
        );
    }

    #[test]
    fn error_reasons_are_distinct() {
        let reasons = [
            McpBackendError::Timeout.reason(),
            McpBackendError::Transport.reason(),
            McpBackendError::ResponseTooLarge { limit: 1 }.reason(),
            McpBackendError::Rejected { status: 500 }.reason(),
            McpBackendError::JsonRpc { code: 1 }.reason(),
            McpBackendError::Malformed("x").reason(),
        ];
        let unique: std::collections::HashSet<_> = reasons.iter().collect();
        assert_eq!(
            unique.len(),
            reasons.len(),
            "every failure class needs a distinct reason"
        );
    }
}
