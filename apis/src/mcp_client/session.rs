// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! MCP session lifecycle for callers that keep their own session map.
//!
//! [`McpSessionClient`] runs the `initialize`, `notifications/initialized`,
//! and session `DELETE` exchanges as separate steps through the filtered
//! subrequest transport the hosted MCP paths use, with `rmcp` building each
//! message and parsing each reply. The caller holds the resulting
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

use bytes::Bytes;
use http::{HeaderMap, HeaderName, HeaderValue, Method};
use praxis_core::subrequest::SubRequestClient;
use praxis_filter::{FilterPipeline, StreamingResponseBody, SubResponse, SubrequestRuntime, TraceContext};
use rmcp::{
    model::{
        ClientCapabilities, ClientJsonRpcMessage, ClientNotification, ClientRequest, Implementation, InitializeRequest,
        InitializeRequestParams, InitializedNotification, JsonRpcMessage, ProtocolVersion, RequestId,
        ServerJsonRpcMessage, ServerResult,
    },
    transport::streamable_http_client::StreamableHttpError,
};

use super::subrequest_transport::{
    McpCallout, McpExchangeStage, McpSubrequestClient, McpTransportError, TransportSignal,
};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Prefix of the JSON-RPC id used for each `initialize`.
const INITIALIZE_REQUEST_ID_PREFIX: &str = "praxis-initialize-";

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
    /// The server answered `initialize` with a JSON-RPC error.
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
    /// Returns [`McpSessionError`] on timeout, transport failure, an
    /// oversized reply, a non-success status, a JSON-RPC error, or a reply
    /// that is not a valid `initialize` result.
    pub async fn initialize(
        &self,
        path: &str,
        stage: Arc<dyn McpExchangeStage>,
        callout: &McpSessionCallout,
        protocol_version: &str,
    ) -> Result<McpSessionInit, McpSessionError> {
        let transport = self.transport(stage, callout);
        let exchange = self.initialize_exchange(&transport, path, callout, protocol_version);
        Box::pin(within(callout.deadline, exchange)).await
    }

    /// POST `initialize` and read the session id and negotiated version from
    /// the reply, whether it arrives as JSON or on an SSE stream.
    #[expect(clippy::large_stack_frames, reason = "rmcp/executor futures are inherently large")]
    async fn initialize_exchange(
        &self,
        transport: &McpSubrequestClient,
        path: &str,
        callout: &McpSessionCallout,
        protocol_version: &str,
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
        let (reply, _session) = Box::pin(transport.classify_post_response(
            response,
            body,
            message,
            false,
            self.max_response_bytes,
            Arc::clone(&signal),
        ))
        .await
        .map_err(classify)?
        .expect_initialized::<McpTransportError>()
        .await
        .map_err(classify)?;
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
            let (_response, body) = reject_unsuccessful(response, body).await?;
            cancel(body).await;
            Ok(())
        };
        Box::pin(within(callout.deadline, exchange)).await
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
        let transport = self.transport(stage, callout);
        let headers = session_headers(&callout.forwarded_headers, Some(session_id), &session.protocol_version)?;
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

    /// Map a transport failure to a structured, credential-free error.
    fn classify(
        &self,
        error: &StreamableHttpError<McpTransportError>,
        signal: &OnceLock<TransportSignal>,
        deadline: Instant,
    ) -> McpSessionError {
        if let Some(TransportSignal::ResponseTooLarge { limit }) = signal.get() {
            return McpSessionError::ResponseTooLarge { limit: *limit };
        }
        match error {
            StreamableHttpError::Client(McpTransportError::ResponseTooLarge) => McpSessionError::ResponseTooLarge {
                limit: self.max_response_bytes,
            },
            StreamableHttpError::Client(
                McpTransportError::Transport
                | McpTransportError::Target
                | McpTransportError::SsrfBlocked
                | McpTransportError::Setup,
            )
            | StreamableHttpError::Io(_)
            | StreamableHttpError::Sse(sse_stream::Error::Body(_))
            | StreamableHttpError::UnexpectedEndOfStream
            | StreamableHttpError::TransportChannelClosed => {
                if Instant::now() >= deadline {
                    McpSessionError::Timeout
                } else {
                    McpSessionError::Transport
                }
            },
            StreamableHttpError::UnexpectedContentType(_) => McpSessionError::Malformed("unexpected content type"),
            StreamableHttpError::UnexpectedServerResponse(_) => {
                McpSessionError::Malformed("initialize reply is missing")
            },
            _ => McpSessionError::Malformed("initialize reply is not a JSON-RPC message"),
        }
    }
}

// -----------------------------------------------------------------------------
// Request Builders
// -----------------------------------------------------------------------------

/// A fresh JSON-RPC id for one `initialize`.
///
/// Callers that forward their own clients' requests on the same session do
/// not choose those ids, and MCP forbids reusing an id within a session. A
/// random id keeps this request from colliding with any id a client picks.
fn initialize_request_id() -> Result<String, McpSessionError> {
    let mut random = [0_u8; 16];
    crate::hash::random_bytes(&mut random).map_err(|_error| McpSessionError::Transport)?;
    Ok(format!(
        "{INITIALIZE_REQUEST_ID_PREFIX}{:032x}",
        u128::from_be_bytes(random)
    ))
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
