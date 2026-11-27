// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! MCP session lifecycle for callers that keep their own session map.
//!
//! [`McpSessionClient`] runs the `initialize`, `notifications/initialized`,
//! and session `DELETE` exchanges as separate steps through the filtered
//! subrequest transport the hosted MCP paths use. `rmcp` builds and decodes
//! the JSON-RPC messages; SSE replies are framed by the crate's conforming SSE
//! parser, and nothing a server sends is logged. The caller holds the resulting
//! [`McpSessionInit`] and forwards its own requests on that session.
//!
//! No `rmcp` service runs between exchanges: a session costs no task, no
//! channel, and no standalone `GET` stream while it sits idle in the caller's
//! map, and ends only when the caller terminates it.
//!
//! The server is reached through a bound outbound chain that selects the
//! destination itself; [`McpExchangeStage`] places the facts the chain reads.

use std::{
    fmt,
    sync::{Arc, OnceLock},
    time::{Duration, Instant},
};

use bytes::{Bytes, BytesMut};
use http::{HeaderMap, HeaderName, HeaderValue, Method};
use praxis_core::subrequest::SubRequestClient;
use praxis_filter::{
    CalloutResponseTooLarge, FilterPipeline, StreamingResponseBody, SubResponse, SubrequestRuntime, TraceContext,
};
use rmcp::{
    model::{
        ClientCapabilities, ClientJsonRpcMessage, ClientNotification, ClientRequest, ClientResult, ErrorCode,
        ErrorData, Implementation, InitializeRequest, InitializeRequestParams, InitializedNotification, JsonRpcMessage,
        JsonRpcRequest, ListToolsResult, ProtocolVersion, RequestId, ServerJsonRpcMessage, ServerRequest, ServerResult,
    },
    transport::streamable_http_client::StreamableHttpError,
};

use super::{
    listing::McpToolPage,
    subrequest_transport::{McpCallout, McpExchangeStage, McpSubrequestClient, McpTransportError, TransportSignal},
};
use crate::openai::sse::SseFrameParser;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Prefix of the JSON-RPC id used for each `initialize`.
const INITIALIZE_REQUEST_ID_PREFIX: &str = "praxis-initialize-";

/// Prefix of the JSON-RPC id used for each `tools/list`.
const TOOLS_LIST_REQUEST_ID_PREFIX: &str = "praxis-tools-list-";

/// Client name reported to MCP servers.
const CLIENT_NAME: &str = "praxis";

/// Client version reported to MCP servers.
const CLIENT_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Maximum accepted `Mcp-Session-Id` length.
const MAX_SESSION_ID_LEN: usize = 256;

/// Maximum accepted negotiated protocol version length.
const MAX_PROTOCOL_VERSION_LEN: usize = 64;

/// MCP session header.
const MCP_SESSION_ID: HeaderName = HeaderName::from_static("mcp-session-id");

/// MCP protocol version header.
const MCP_PROTOCOL_VERSION: HeaderName = HeaderName::from_static("mcp-protocol-version");

/// Stateless-profile method header.
const MCP_METHOD: HeaderName = HeaderName::from_static("mcp-method");

/// Most server requests answered while one reply is read.
///
/// Before `notifications/initialized` MCP lets a server send only pings (and
/// logging, which is a notification). One ping checks liveness; a few more
/// cover a server that pings again while it waits. This client declares no
/// capabilities, so a later request has nothing more to ask. Each answer is a
/// separate POST, so the cap bounds the outbound requests one exchange can
/// trigger.
const MAX_SERVER_REQUESTS: usize = 4;

// -----------------------------------------------------------------------------
// Types
// -----------------------------------------------------------------------------

/// Request-scoped resources one lifecycle exchange runs with.
#[derive(Clone)]
pub struct McpSessionCallout {
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

/// How a `tools/list` request is sent.
#[derive(Clone, Copy, Debug)]
pub enum McpListingScope<'a> {
    /// On an initialized session, at its negotiated version.
    Session(&'a McpSessionInit),
    /// Without a session, under the MCP `2026-07-28` stateless profile.
    Stateless {
        /// Protocol version the request declares.
        protocol_version: &'a str,
    },
}

/// Result of a successful `initialize`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct McpSessionInit {
    /// Protocol version the server selected.
    pub protocol_version: Box<str>,
    /// Server `Mcp-Session-Id`; `None` for a server that issues no session.
    pub session_id: Option<HeaderValue>,
}

/// Structured lifecycle failure.
///
/// Carries no URL, header value, or server-provided text.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum McpSessionError {
    /// The exchange exceeded its deadline.
    Timeout,
    /// The exchange could not be delivered.
    Transport,
    /// The server response exceeded the configured ceiling.
    ResponseTooLarge {
        /// Ceiling that was exceeded.
        limit: usize,
    },
    /// The server answered with a non-success HTTP status.
    Rejected {
        /// HTTP status returned by the server.
        status: u16,
    },
    /// The server answered the request with a JSON-RPC error.
    JsonRpc {
        /// JSON-RPC error code.
        code: i64,
    },
    /// The server response is not a valid MCP lifecycle reply.
    Malformed(&'static str),
}

impl fmt::Display for McpSessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Timeout => f.write_str("mcp session exchange timed out"),
            Self::Transport => f.write_str("mcp session exchange failed"),
            Self::ResponseTooLarge { limit } => write!(f, "mcp server response exceeded {limit} bytes"),
            Self::Rejected { status } => write!(f, "mcp server rejected the exchange with HTTP {status}"),
            Self::JsonRpc { code } => write!(f, "mcp server returned JSON-RPC error {code}"),
            Self::Malformed(detail) => write!(f, "malformed mcp server response: {detail}"),
        }
    }
}

impl std::error::Error for McpSessionError {}

/// A failed `initialize`, with the session the server issued before it failed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct McpInitializeError {
    /// Why the exchange failed.
    pub error: McpSessionError,
    /// `Mcp-Session-Id` from the response head, when the exchange failed
    /// after the server issued one. That session exists on the server and is
    /// never confirmed, so the caller ends it with
    /// [`McpSessionClient::terminate_issued`].
    pub issued_session: Option<HeaderValue>,
}

impl From<McpSessionError> for McpInitializeError {
    fn from(error: McpSessionError) -> Self {
        Self {
            error,
            issued_session: None,
        }
    }
}

impl fmt::Display for McpInitializeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.error.fmt(f)
    }
}

impl std::error::Error for McpInitializeError {}

// -----------------------------------------------------------------------------
// McpSessionClient
// -----------------------------------------------------------------------------

/// Runs MCP session lifecycle exchanges through a bound outbound chain.
///
/// Each exchange is bounded by `max_response_bytes` before its reply is
/// parsed, by `step_timeout`, and by the callout deadline. An `initialize`
/// reply on an SSE stream completes as soon as its JSON-RPC response arrives,
/// even when the server keeps the stream open.
pub struct McpSessionClient {
    /// Outbound chain that selects the destination and applies operator policy.
    pipeline: Arc<FilterPipeline>,
    /// Response ceiling for each exchange.
    max_response_bytes: usize,
    /// Per-exchange timeout.
    step_timeout: Duration,
}

impl McpSessionClient {
    /// Build a client over the bound `pipeline`.
    #[must_use]
    pub fn new(pipeline: Arc<FilterPipeline>, max_response_bytes: usize, step_timeout: Duration) -> Self {
        Self {
            pipeline,
            max_response_bytes,
            step_timeout,
        }
    }

    /// Send `initialize` to `path` and capture the server session.
    ///
    /// # Errors
    ///
    /// Returns [`McpInitializeError`] on timeout, transport failure, an
    /// oversized reply, a non-success status, a JSON-RPC error, a reply that
    /// is not a valid `initialize` result, or a failed answer to a server
    /// request. It carries the session the server issued, if any.
    pub async fn initialize(
        &self,
        path: &str,
        stage: Arc<dyn McpExchangeStage>,
        callout: &McpSessionCallout,
        protocol_version: &str,
    ) -> Result<McpSessionInit, McpInitializeError> {
        let callout = &self.bounded(callout);
        let transport = self.transport(Arc::clone(&stage), callout);
        // Set once the response head is accepted, so a failure after it,
        // the deadline included, still reports the session to end.
        let issued = OnceLock::new();
        let exchange = self.initialize_exchange(&transport, &stage, path, callout, protocol_version, &issued);
        let result = Box::pin(within(callout.deadline, exchange)).await;
        result.map_err(|error| McpInitializeError {
            error,
            issued_session: issued.get().cloned(),
        })
    }

    /// POST `initialize` and read the session id and negotiated version from
    /// the reply, whether it arrives as JSON or on an SSE stream.
    #[expect(clippy::large_stack_frames, reason = "executor futures are inherently large")]
    #[expect(
        clippy::too_many_arguments,
        reason = "transport, stage, path, callout, version, and issued-session cell describe one exchange"
    )]
    async fn initialize_exchange(
        &self,
        transport: &McpSubrequestClient,
        stage: &Arc<dyn McpExchangeStage>,
        path: &str,
        callout: &McpSessionCallout,
        protocol_version: &str,
        issued: &OnceLock<HeaderValue>,
    ) -> Result<McpSessionInit, McpSessionError> {
        let request_id = RequestId::String(Arc::from(initialize_request_id()?));
        let message = &initialize_message(protocol_version, request_id.clone())?;
        let signal = transport.signal_handle();
        let classify = |error: StreamableHttpError<McpTransportError>| self.classify(&error, &signal, callout.deadline);
        let (response, body) =
            Box::pin(transport.send_streaming_post(path, message, None, callout.forwarded_headers.clone(), &signal))
                .await
                .map_err(classify)?;
        let (response, body, session_id) = accept_initialize_head(response, body).await?;
        if let Some(session_id) = session_id.as_ref() {
            // A cheap `Bytes` handle, kept so a failed handshake can end it.
            let _first = issued.set(session_id.clone());
        }
        let mut responder = ServerRequestResponder {
            client: self,
            path,
            stage,
            callout,
            session_id: session_id.as_ref(),
            // None is negotiated until the result arrives.
            protocol_version: None,
            answered: 0,
        };
        let reply = read_reply(&response, body, self.max_response_bytes, &mut responder)
            .await
            .map_err(|error| past_deadline(error, callout.deadline))?;
        Ok(McpSessionInit {
            protocol_version: initialize_result_version(&reply, &request_id)?,
            session_id,
        })
    }

    /// Send `notifications/initialized` on an initialized session.
    ///
    /// # Errors
    ///
    /// Returns [`McpSessionError`] on timeout, transport failure, an
    /// oversized reply, or a non-success status.
    pub async fn send_initialized(
        &self,
        path: &str,
        stage: Arc<dyn McpExchangeStage>,
        callout: &McpSessionCallout,
        session: &McpSessionInit,
    ) -> Result<(), McpSessionError> {
        let callout = &self.bounded(callout);
        let transport = self.transport(stage, callout);
        let message = ClientJsonRpcMessage::notification(ClientNotification::InitializedNotification(
            InitializedNotification::default(),
        ));
        let session_id = session_id_str(session)?;
        let headers = session_headers(&callout.forwarded_headers, None, &session.protocol_version)?;
        let signal = transport.signal_handle();
        let exchange = async {
            let (response, body) =
                Box::pin(transport.send_streaming_post(path, &message, session_id, headers, &signal))
                    .await
                    .map_err(|error| self.classify(&error, &signal, callout.deadline))?;
            let (response, body) = reject_unsuccessful(response, body).await?;
            discard_within(&response, body, self.max_response_bytes)
                .await
                .map_err(|error| past_deadline(error, callout.deadline))
        };
        Box::pin(within(callout.deadline, exchange)).await
    }

    /// Request one `tools/list` page at `cursor` (`None` for the first).
    ///
    /// On a session the request carries its `Mcp-Session-Id` and negotiated
    /// version. A stateless request carries the declared version in
    /// `MCP-Protocol-Version` and `params._meta`, with the client info and
    /// capabilities the profile requires, and `Mcp-Method: tools/list`.
    /// Server requests that arrive before the reply are answered as during
    /// `initialize`. Each tool is returned as its JSON object, decoded and
    /// shape-checked as an MCP `Tool`.
    ///
    /// # Errors
    ///
    /// Returns [`McpSessionError`] on timeout, transport failure, an
    /// oversized reply, a non-success status, a JSON-RPC error, a reply that
    /// is not a `tools/list` result, or a failed answer to a server request.
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors the other lifecycle exchanges plus scope and cursor"
    )]
    pub async fn list_tools(
        &self,
        path: &str,
        stage: Arc<dyn McpExchangeStage>,
        callout: &McpSessionCallout,
        scope: McpListingScope<'_>,
        cursor: Option<String>,
    ) -> Result<McpToolPage<serde_json::Value>, McpSessionError> {
        let callout = &self.bounded(callout);
        let transport = self.transport(Arc::clone(&stage), callout);
        let exchange = self.list_tools_exchange(&transport, &stage, path, callout, scope, cursor);
        Box::pin(within(callout.deadline, exchange)).await
    }

    /// POST `tools/list` and decode the page from the reply.
    #[expect(clippy::large_stack_frames, reason = "executor futures are inherently large")]
    #[expect(
        clippy::too_many_arguments,
        reason = "transport, stage, path, callout, scope, and cursor describe one exchange"
    )]
    #[expect(
        clippy::too_many_lines,
        reason = "request, headers, and reply handling of one exchange"
    )]
    async fn list_tools_exchange(
        &self,
        transport: &McpSubrequestClient,
        stage: &Arc<dyn McpExchangeStage>,
        path: &str,
        callout: &McpSessionCallout,
        scope: McpListingScope<'_>,
        cursor: Option<String>,
    ) -> Result<McpToolPage<serde_json::Value>, McpSessionError> {
        let request_id = request_id(TOOLS_LIST_REQUEST_ID_PREFIX)?;
        let message = tools_list_message(scope, cursor, &request_id)?;
        let (session_id, protocol_version) = match scope {
            McpListingScope::Session(session) => (session.session_id.as_ref(), &*session.protocol_version),
            McpListingScope::Stateless { protocol_version } => (None, protocol_version),
        };
        let mut headers = session_headers(&callout.forwarded_headers, None, protocol_version)?;
        if matches!(scope, McpListingScope::Stateless { .. }) {
            headers.insert(MCP_METHOD, HeaderValue::from_static("tools/list"));
        }
        let session_str = session_id
            .map(|value| {
                value
                    .to_str()
                    .map_err(|_error| McpSessionError::Malformed("invalid Mcp-Session-Id"))
            })
            .transpose()?;
        let signal = transport.signal_handle();
        let (response, body) = Box::pin(transport.send_streaming_post(path, &message, session_str, headers, &signal))
            .await
            .map_err(|error| self.classify(&error, &signal, callout.deadline))?;
        let (response, body) = reject_unsuccessful(response, body).await?;
        let mut responder = ServerRequestResponder {
            client: self,
            path,
            stage,
            callout,
            session_id,
            protocol_version: Some(protocol_version),
            answered: 0,
        };
        let reply = read_reply(&response, body, self.max_response_bytes, &mut responder)
            .await
            .map_err(|error| past_deadline(error, callout.deadline))?;
        tools_list_page(reply, &request_id)
    }

    /// Terminate a session with HTTP `DELETE`.
    ///
    /// A session without an id, and a server answering `404` (session already
    /// gone) or `405` (no client-initiated termination), leave nothing to end.
    ///
    /// # Errors
    ///
    /// Returns [`McpSessionError`] on timeout, transport failure, an
    /// oversized reply, or another non-success status.
    pub async fn terminate(
        &self,
        path: &str,
        stage: Arc<dyn McpExchangeStage>,
        callout: &McpSessionCallout,
        session: &McpSessionInit,
    ) -> Result<(), McpSessionError> {
        let Some(session_id) = session.session_id.as_ref() else {
            return Ok(());
        };
        let headers = session_headers(&callout.forwarded_headers, Some(session_id), &session.protocol_version)?;
        Box::pin(self.delete_session(path, stage, callout, headers)).await
    }

    /// Terminate a session the server issued in an `initialize` that failed.
    ///
    /// Sends no `MCP-Protocol-Version`: MCP sets it to the negotiated version,
    /// and none was negotiated before the `initialize` result arrived.
    ///
    /// # Errors
    ///
    /// Returns [`McpSessionError`] as [`Self::terminate`] does.
    pub async fn terminate_issued(
        &self,
        path: &str,
        stage: Arc<dyn McpExchangeStage>,
        callout: &McpSessionCallout,
        session_id: &HeaderValue,
    ) -> Result<(), McpSessionError> {
        let mut headers = callout.forwarded_headers.clone();
        headers.insert(MCP_SESSION_ID, session_id.clone());
        Box::pin(self.delete_session(path, stage, callout, headers)).await
    }

    /// Send the session `DELETE` with `headers`. A `404` (session already
    /// gone) or `405` (no client-initiated termination) leaves nothing to end.
    async fn delete_session(
        &self,
        path: &str,
        stage: Arc<dyn McpExchangeStage>,
        callout: &McpSessionCallout,
        headers: HeaderMap,
    ) -> Result<(), McpSessionError> {
        let callout = &self.bounded(callout);
        let transport = self.transport(stage, callout);
        let signal = Arc::new(OnceLock::new());
        let exchange = async {
            let response = Box::pin(transport.execute(
                Method::DELETE,
                path,
                Bytes::new(),
                headers,
                transport.control_response_limit(),
                &signal,
            ))
            .await
            .map_err(|error| self.classify(&error, &signal, callout.deadline))?;
            match response.status {
                200..=299 | 404 | 405 => Ok(()),
                status => Err(McpSessionError::Rejected { status }),
            }
        };
        Box::pin(within(callout.deadline, exchange)).await
    }

    /// `callout` with its deadline capped by one step, so the executor, the
    /// outer timeout, and error classification all use the same instant.
    fn bounded(&self, callout: &McpSessionCallout) -> McpSessionCallout {
        let now = Instant::now();
        let step = now.checked_add(self.step_timeout).unwrap_or(now);
        McpSessionCallout {
            deadline: callout.deadline.min(step),
            ..callout.clone()
        }
    }

    /// Transport for one exchange through the bound chain.
    fn transport(&self, stage: Arc<dyn McpExchangeStage>, callout: &McpSessionCallout) -> McpSubrequestClient {
        let mcp_callout = McpCallout::from_parts(
            callout.client.clone(),
            callout.downstream.clone(),
            Arc::clone(&self.pipeline),
            callout.depth,
            callout.trace_context.clone(),
            callout.deadline,
        );
        McpSubrequestClient::for_chain(mcp_callout, self.step_timeout, self.max_response_bytes, stage)
    }

    /// Map a failure to send the request or receive the response head to a
    /// structured, credential-free error.
    fn classify(
        &self,
        error: &StreamableHttpError<McpTransportError>,
        signal: &OnceLock<TransportSignal>,
        deadline: Instant,
    ) -> McpSessionError {
        // The executor's ceiling may be the doubled streaming backstop; callers
        // see the configured limit.
        if matches!(signal.get(), Some(TransportSignal::ResponseTooLarge { .. }))
            || matches!(error, StreamableHttpError::Client(McpTransportError::ResponseTooLarge))
        {
            return McpSessionError::ResponseTooLarge {
                limit: self.max_response_bytes,
            };
        }
        match error {
            StreamableHttpError::Client(McpTransportError::InvalidStatus) => {
                McpSessionError::Malformed("unrepresentable status or session header")
            },
            StreamableHttpError::Client(McpTransportError::Serialize) => {
                McpSessionError::Malformed("request could not be serialized")
            },
            _ => past_deadline(McpSessionError::Transport, deadline),
        }
    }
}

// -----------------------------------------------------------------------------
// Server Requests
// -----------------------------------------------------------------------------

/// Answers the requests a server sends on a response stream.
///
/// MCP lets a server send requests on that stream before its response, and
/// a ping receiver must answer. A server may hold its `initialize` result
/// until its ping is answered, so each answer is sent and accepted before
/// the stream is read further.
struct ServerRequestResponder<'a> {
    /// Client whose chain, limits, and error mapping the answers use.
    client: &'a McpSessionClient,
    /// MCP endpoint path the `initialize` was sent to.
    path: &'a str,
    /// Stage of the `initialize` exchange; answers follow its endpoint.
    stage: &'a Arc<dyn McpExchangeStage>,
    /// The `initialize` callout: credentials, correlation, and deadline.
    callout: &'a McpSessionCallout,
    /// `Mcp-Session-Id` of the session the exchange runs on.
    session_id: Option<&'a HeaderValue>,
    /// Negotiated protocol version, sent on answers; `None` during
    /// `initialize`.
    protocol_version: Option<&'a str>,
    /// Server requests answered so far.
    answered: usize,
}

impl ServerRequestResponder<'_> {
    /// Handle the message in one SSE event: a response or error is the
    /// reply, a request is answered, and a notification or an event without
    /// data is skipped.
    async fn next_reply(&mut self, data: &[u8]) -> Result<Option<ServerJsonRpcMessage>, McpSessionError> {
        if data.iter().all(u8::is_ascii_whitespace) {
            return Ok(None);
        }
        match parse_reply(data)? {
            JsonRpcMessage::Request(request) => {
                Box::pin(self.answer(&request)).await?;
                Ok(None)
            },
            JsonRpcMessage::Notification(_) => Ok(None),
            reply => Ok(Some(reply)),
        }
    }

    /// POST the answer to `request` and wait for the server to accept it.
    ///
    /// The answer goes through the same bound chain, so operator outbound
    /// filters and credential injection apply, with the forwarded headers
    /// that created the session, its `Mcp-Session-Id`, and the endpoint
    /// serving the stream. It shares the `initialize` deadline.
    ///
    /// During `initialize` no `MCP-Protocol-Version` header is sent. MCP
    /// requires it on requests after initialization, carrying the negotiated
    /// version, and none is negotiated until the `initialize` result arrives. The offered version
    /// is not a substitute: a server that will select an older one may treat
    /// it as unsupported, and MCP then requires `400 Bad Request`. A server
    /// without the header identifies the version from the session or assumes
    /// `2025-03-26`, which defines `ping`.
    async fn answer(&mut self, request: &JsonRpcRequest<ServerRequest>) -> Result<(), McpSessionError> {
        self.answered = self.answered.saturating_add(1);
        if self.answered > MAX_SERVER_REQUESTS {
            return Err(McpSessionError::Malformed("too many server requests"));
        }
        let message = server_request_answer(request);
        let session_id = self
            .session_id
            .map(|value| {
                value
                    .to_str()
                    .map_err(|_error| McpSessionError::Malformed("invalid Mcp-Session-Id"))
            })
            .transpose()?;
        let transport = self.client.transport(self.stage.follow_up(), self.callout);
        let signal = transport.signal_handle();
        let deadline = self.callout.deadline;
        let headers = self.answer_headers()?;
        let (response, body) =
            Box::pin(transport.send_streaming_post(self.path, &message, session_id, headers, &signal))
                .await
                .map_err(|error| self.client.classify(&error, &signal, deadline))?;
        let (response, body) = reject_unsuccessful(response, body)
            .await
            .map_err(|error| past_deadline(error, deadline))?;
        discard_within(&response, body, self.client.max_response_bytes)
            .await
            .map_err(|error| past_deadline(error, deadline))
    }

    /// The forwarded headers, plus the negotiated version once there is one.
    fn answer_headers(&self) -> Result<HeaderMap, McpSessionError> {
        let mut headers = self.callout.forwarded_headers.clone();
        if let Some(version) = self.protocol_version {
            let version = HeaderValue::from_str(version)
                .map_err(|_error| McpSessionError::Malformed("invalid protocol version"))?;
            headers.insert(MCP_PROTOCOL_VERSION, version);
        }
        Ok(headers)
    }
}


/// The client's answer to a server request: an empty result for `ping`, as
/// MCP requires, and JSON-RPC `-32601` (method not found) for any other
/// method. This client declares no capabilities, so sampling, roots,
/// elicitation, and unknown methods are all unavailable here.
fn server_request_answer(request: &JsonRpcRequest<ServerRequest>) -> ClientJsonRpcMessage {
    let id = request.id.clone();
    if is_ping(&request.request) {
        ClientJsonRpcMessage::response(ClientResult::empty(()), id)
    } else {
        ClientJsonRpcMessage::error(
            ErrorData::new(ErrorCode::METHOD_NOT_FOUND, "Method not found", None),
            Some(id),
        )
    }
}

/// Whether a server request is `ping`, including one whose extra members
/// made it parse as a custom request.
fn is_ping(request: &ServerRequest) -> bool {
    match request {
        ServerRequest::PingRequest(_) => true,
        ServerRequest::CustomRequest(custom) => custom.method == "ping",
        _ => false,
    }
}

// -----------------------------------------------------------------------------
// Request Builders
// -----------------------------------------------------------------------------

/// A fresh JSON-RPC id for one `initialize`.
fn initialize_request_id() -> Result<String, McpSessionError> {
    request_id(INITIALIZE_REQUEST_ID_PREFIX).map(|id| id.to_string())
}

/// A fresh JSON-RPC id starting with `prefix`.
///
/// Callers that forward their own clients' requests on the same session do
/// not choose those ids, and MCP forbids reusing an id within a session. A
/// random id keeps this request from colliding with any id a client picks.
fn request_id(prefix: &str) -> Result<RequestId, McpSessionError> {
    let mut random = [0_u8; 16];
    crate::hash::random_bytes(&mut random).map_err(|_error| McpSessionError::Transport)?;
    Ok(RequestId::String(Arc::from(format!(
        "{prefix}{:032x}",
        u128::from_be_bytes(random)
    ))))
}

/// Build the `tools/list` request for `scope` at `cursor`.
fn tools_list_message(
    scope: McpListingScope<'_>,
    cursor: Option<String>,
    id: &RequestId,
) -> Result<ClientJsonRpcMessage, McpSessionError> {
    let mut params = serde_json::Map::new();
    if let Some(cursor) = cursor {
        params.insert("cursor".to_owned(), serde_json::Value::String(cursor));
    }
    if let McpListingScope::Stateless { protocol_version } = scope {
        params.insert(
            "_meta".to_owned(),
            serde_json::json!({
                "io.modelcontextprotocol/protocolVersion": protocol_version,
                "io.modelcontextprotocol/clientInfo": { "name": CLIENT_NAME, "version": CLIENT_VERSION },
                "io.modelcontextprotocol/clientCapabilities": {},
            }),
        );
    }
    let message = serde_json::json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": "tools/list",
        "params": params,
    });
    serde_json::from_value(message)
        .map_err(|_error| McpSessionError::Malformed("tools/list request could not be built"))
}

/// Build the `initialize` request offering `protocol_version`.
fn initialize_message(protocol_version: &str, id: RequestId) -> Result<ClientJsonRpcMessage, McpSessionError> {
    let version: ProtocolVersion = serde_json::from_value(serde_json::Value::String(protocol_version.to_owned()))
        .map_err(|_error| McpSessionError::Malformed("invalid protocol version"))?;
    let mut params = InitializeRequestParams::new(
        ClientCapabilities::default(),
        Implementation::new(CLIENT_NAME, CLIENT_VERSION),
    );
    params.protocol_version = version;
    Ok(ClientJsonRpcMessage::request(
        ClientRequest::InitializeRequest(InitializeRequest::new(params)),
        id,
    ))
}

/// Headers for a post-`initialize` exchange: the forwarded set, the session
/// id when given, and the negotiated protocol version.
fn session_headers(
    forwarded: &HeaderMap,
    session_id: Option<&HeaderValue>,
    protocol_version: &str,
) -> Result<HeaderMap, McpSessionError> {
    let mut headers = forwarded.clone();
    if let Some(session_id) = session_id {
        headers.insert(MCP_SESSION_ID, session_id.clone());
    }
    let version = HeaderValue::from_str(protocol_version)
        .map_err(|_error| McpSessionError::Malformed("invalid protocol version"))?;
    headers.insert(MCP_PROTOCOL_VERSION, version);
    Ok(headers)
}

/// The session id as text, for the transport's session header.
fn session_id_str(session: &McpSessionInit) -> Result<Option<&str>, McpSessionError> {
    session
        .session_id
        .as_ref()
        .map(|value| {
            value
                .to_str()
                .map_err(|_error| McpSessionError::Malformed("invalid Mcp-Session-Id"))
        })
        .transpose()
}

// -----------------------------------------------------------------------------
// Response Handling
// -----------------------------------------------------------------------------

/// Run `exchange` until `deadline`.
async fn within<T>(
    deadline: Instant,
    exchange: impl Future<Output = Result<T, McpSessionError>>,
) -> Result<T, McpSessionError> {
    tokio::time::timeout_at(deadline.into(), exchange)
        .await
        .unwrap_or(Err(McpSessionError::Timeout))
}

/// Release a streaming body that will not be read.
async fn cancel(body: Option<Box<dyn StreamingResponseBody>>) {
    if let Some(mut body) = body {
        body.cancel().await;
    }
}

/// Pass a success response through; reject any other status after releasing
/// its body.
async fn reject_unsuccessful(
    response: SubResponse,
    body: Option<Box<dyn StreamingResponseBody>>,
) -> Result<(SubResponse, Option<Box<dyn StreamingResponseBody>>), McpSessionError> {
    if (200..300).contains(&response.status) {
        return Ok((response, body));
    }
    cancel(body).await;
    Err(McpSessionError::Rejected {
        status: response.status,
    })
}

/// Accept a success `initialize` response head and read its session id,
/// releasing the body when the head is rejected.
async fn accept_initialize_head(
    response: SubResponse,
    body: Option<Box<dyn StreamingResponseBody>>,
) -> Result<(SubResponse, Option<Box<dyn StreamingResponseBody>>, Option<HeaderValue>), McpSessionError> {
    let (response, body) = reject_unsuccessful(response, body).await?;
    match response_session_id(&response.headers) {
        Ok(session_id) => Ok((response, body, session_id)),
        Err(error) => {
            cancel(body).await;
            Err(error)
        },
    }
}

/// Read the JSON-RPC reply to a request from a success response, within
/// `limit` bytes.
///
/// A JSON body is parsed whole. An SSE body is read with the crate's
/// conforming SSE parser, which ignores unknown fields and discards an event
/// cut off at end of stream. Reading stops at the first response or error
/// event, so a backend may keep the stream open after it. Server requests
/// before it are answered through `responder`, and notifications are
/// skipped; none of their contents is logged.
async fn read_reply(
    response: &SubResponse,
    body: Option<Box<dyn StreamingResponseBody>>,
    limit: usize,
    responder: &mut ServerRequestResponder<'_>,
) -> Result<ServerJsonRpcMessage, McpSessionError> {
    let content_type = response
        .headers
        .get(http::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default();
    let is_sse = crate::is_event_stream_content_type(content_type);
    if !is_sse && !is_json_content_type(content_type) {
        cancel(body).await;
        return Err(McpSessionError::Malformed("unexpected content type"));
    }
    let Some(mut body) = body else {
        return Box::pin(buffered_reply(&response.body, is_sse, limit, responder)).await;
    };
    let reply = if is_sse {
        Box::pin(stream_sse_reply(&mut body, limit, responder)).await
    } else {
        collect_json_reply(&mut body, limit).await
    };
    body.cancel().await;
    reply
}

/// The reply in a body the chain buffered. The executor bounded it only by the
/// looser streaming backstop, so the exact limit is checked here.
///
/// The chain delivers the body only after the stream ended, so an answer
/// sent now cannot unblock the server: one that holds its result until a
/// ping is answered never ends the stream, and the executor ends the
/// exchange at its deadline with a synthesized `504`. Server requests in a
/// delivered body are still answered, in order, before the reply is
/// returned, because MCP requires an answer.
async fn buffered_reply(
    body: &[u8],
    is_sse: bool,
    limit: usize,
    responder: &mut ServerRequestResponder<'_>,
) -> Result<ServerJsonRpcMessage, McpSessionError> {
    if body.len() > limit {
        return Err(McpSessionError::ResponseTooLarge { limit });
    }
    if !is_sse {
        return parse_reply(body);
    }
    for data in crate::buffered_sse_event_data(body) {
        if let Some(reply) = responder.next_reply(&data).await? {
            return Ok(reply);
        }
    }
    Err(McpSessionError::Malformed("stream has no reply"))
}

/// Read SSE events until the first JSON-RPC response or error, answering
/// server requests as they arrive.
async fn stream_sse_reply(
    body: &mut Box<dyn StreamingResponseBody>,
    limit: usize,
    responder: &mut ServerRequestResponder<'_>,
) -> Result<ServerJsonRpcMessage, McpSessionError> {
    // The parser stores an `event:` value lossily decoded, and each invalid
    // byte becomes a three-byte U+FFFD, so three times the read limit covers
    // any input within it.
    let mut parser = SseFrameParser::new(limit.saturating_mul(3));
    // The limit covers the bytes through the event that carries the reply.
    // Only the part of each chunk that still fits is parsed, so the result
    // does not depend on how the transport splits the stream.
    let mut remaining = limit;
    while let Some(chunk) = next_chunk(body, limit).await? {
        let (within, beyond) = chunk.split_at(chunk.len().min(remaining));
        remaining = remaining.saturating_sub(within.len());
        let frames = parser
            .parse_chunk(within)
            .map_err(|_overflow| McpSessionError::ResponseTooLarge { limit })?;
        for frame in frames {
            if let Some(reply) = responder.next_reply(&frame.data).await? {
                return Ok(reply);
            }
        }
        if !beyond.is_empty() {
            return Err(McpSessionError::ResponseTooLarge { limit });
        }
    }
    Err(McpSessionError::Malformed("stream has no reply"))
}

/// Read and drop a reply body that carries no message, failing when it
/// exceeds `limit` bytes.
async fn discard_within(
    response: &SubResponse,
    body: Option<Box<dyn StreamingResponseBody>>,
    limit: usize,
) -> Result<(), McpSessionError> {
    let Some(mut body) = body else {
        return if response.body.len() > limit {
            Err(McpSessionError::ResponseTooLarge { limit })
        } else {
            Ok(())
        };
    };
    let mut read = 0_usize;
    let result = loop {
        match next_chunk(&mut body, limit).await {
            Ok(Some(chunk)) => match read.checked_add(chunk.len()).filter(|total| *total <= limit) {
                Some(total) => read = total,
                None => break Err(McpSessionError::ResponseTooLarge { limit }),
            },
            Ok(None) => break Ok(()),
            Err(error) => break Err(error),
        }
    };
    body.cancel().await;
    result
}

/// A transport failure, or a `504`, at or past `deadline` is the executor
/// ending the exchange there, so it is a timeout. The executor answers a
/// request whose response head misses the deadline with a synthesized `504`.
fn past_deadline(error: McpSessionError, deadline: Instant) -> McpSessionError {
    match error {
        McpSessionError::Transport | McpSessionError::Rejected { status: 504 } if Instant::now() >= deadline => {
            McpSessionError::Timeout
        },
        other => other,
    }
}

/// Buffer a streamed JSON body within `limit` bytes and parse it.
async fn collect_json_reply(
    body: &mut Box<dyn StreamingResponseBody>,
    limit: usize,
) -> Result<ServerJsonRpcMessage, McpSessionError> {
    let mut buf = BytesMut::new();
    while let Some(chunk) = next_chunk(body, limit).await? {
        if buf.len().checked_add(chunk.len()).is_none_or(|total| total > limit) {
            return Err(McpSessionError::ResponseTooLarge { limit });
        }
        buf.extend_from_slice(&chunk);
    }
    parse_reply(&buf)
}

/// The next body chunk. An overflow the executor reports stays a size failure,
/// reported against the configured `limit` rather than the executor's looser
/// streaming backstop; any other read failure is a transport failure.
async fn next_chunk(body: &mut Box<dyn StreamingResponseBody>, limit: usize) -> Result<Option<Bytes>, McpSessionError> {
    body.next_chunk().await.map_err(|error| {
        if error.downcast_ref::<CalloutResponseTooLarge>().is_some() {
            McpSessionError::ResponseTooLarge { limit }
        } else {
            McpSessionError::Transport
        }
    })
}

/// Parse one JSON-RPC message from the server.
fn parse_reply(bytes: &[u8]) -> Result<ServerJsonRpcMessage, McpSessionError> {
    serde_json::from_slice(bytes).map_err(|_error| McpSessionError::Malformed("initialize reply is not JSON-RPC"))
}

/// Whether a `Content-Type` value is `application/json`.
fn is_json_content_type(value: &str) -> bool {
    value
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .eq_ignore_ascii_case("application/json")
}

/// Read and validate the server `Mcp-Session-Id` response header.
///
/// MCP requires the id to be visible ASCII. A server that sends more than one
/// leaves the session ambiguous, so the reply is rejected.
fn response_session_id(headers: &HeaderMap) -> Result<Option<HeaderValue>, McpSessionError> {
    let mut values = headers.get_all(&MCP_SESSION_ID).iter();
    let Some(value) = values.next() else {
        return Ok(None);
    };
    if values.next().is_some() {
        return Err(McpSessionError::Malformed("multiple Mcp-Session-Id headers"));
    }
    let bytes = value.as_bytes();
    if bytes.is_empty() || bytes.len() > MAX_SESSION_ID_LEN || !bytes.iter().all(|b| (0x21..=0x7E).contains(b)) {
        return Err(McpSessionError::Malformed("invalid Mcp-Session-Id"));
    }
    Ok(Some(value.clone()))
}

/// Extract the negotiated version from the `initialize` reply to `request_id`.
fn initialize_result_version(
    message: &ServerJsonRpcMessage,
    request_id: &RequestId,
) -> Result<Box<str>, McpSessionError> {
    let result = match message {
        JsonRpcMessage::Response(response) if response.id == *request_id => &response.result,
        JsonRpcMessage::Error(error) if error.id.as_ref() == Some(request_id) => {
            return Err(McpSessionError::JsonRpc {
                code: i64::from(error.error.code.0),
            });
        },
        JsonRpcMessage::Response(_) | JsonRpcMessage::Error(_) => {
            return Err(McpSessionError::Malformed("initialize reply id does not match"));
        },
        _ => return Err(McpSessionError::Malformed("initialize reply is not a response")),
    };
    let ServerResult::InitializeResult(result) = result else {
        return Err(McpSessionError::Malformed(
            "initialize reply is not an initialize result",
        ));
    };
    let version = result.protocol_version.as_str();
    if version.is_empty()
        || version.len() > MAX_PROTOCOL_VERSION_LEN
        || HeaderValue::from_str(version).is_err()
        || version.bytes().any(|b| b.is_ascii_control())
    {
        return Err(McpSessionError::Malformed("invalid protocolVersion"));
    }
    Ok(Box::from(version))
}

/// Decode the `tools/list` page in the reply to `request_id`.
fn tools_list_page(
    message: ServerJsonRpcMessage,
    request_id: &RequestId,
) -> Result<McpToolPage<serde_json::Value>, McpSessionError> {
    let result = match message {
        JsonRpcMessage::Response(response) if response.id == *request_id => response.result,
        JsonRpcMessage::Error(error) if error.id.as_ref() == Some(request_id) => {
            return Err(McpSessionError::JsonRpc {
                code: i64::from(error.error.code.0),
            });
        },
        JsonRpcMessage::Response(_) | JsonRpcMessage::Error(_) => {
            return Err(McpSessionError::Malformed("tools/list reply id does not match"));
        },
        _ => return Err(McpSessionError::Malformed("tools/list reply is not a response")),
    };
    let not_a_listing =
        |_error: serde_json::Error| McpSessionError::Malformed("tools/list reply is not a tools/list result");
    // `ServerResult` is an untagged union, so a listing may decode as another
    // variant; decode it again as a listing.
    let listing = match result {
        ServerResult::ListToolsResult(listing) => listing,
        other => serde_json::to_value(other)
            .and_then(serde_json::from_value::<ListToolsResult>)
            .map_err(not_a_listing)?,
    };
    let tools = listing
        .tools
        .into_iter()
        .map(serde_json::to_value)
        .collect::<Result<Vec<_>, _>>()
        .map_err(not_a_listing)?;
    Ok(McpToolPage {
        tools,
        next_cursor: listing.next_cursor,
    })
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
    clippy::too_many_lines,
    clippy::large_futures,
    clippy::format_push_string,
    reason = "tests"
)]
mod tests;
