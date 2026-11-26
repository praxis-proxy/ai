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
//! [`FilteredMcpBackendClient`] runs each exchange through
//! [`McpSessionClient`], the MCP client the hosted MCP paths share: `rmcp`
//! builds the messages and parses the replies, and the filtered subrequest
//! transport runs them through the broker's generated outbound chain. Replies
//! are bounded before they are parsed, and an `initialize` reply on an SSE
//! stream completes when it arrives even if the backend keeps the stream open.

use std::{fmt, sync::Arc, time::Duration};

use async_trait::async_trait;
use http::{HeaderName, HeaderValue};
use praxis_ai_apis::mcp_session::{McpSessionCallout, McpSessionClient, McpSessionError, McpSessionInit};
use praxis_filter::FilterPipeline;

use super::selector::BrokerBackendCluster;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

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
pub(crate) type BackendCallout = McpSessionCallout;

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
    /// The gateway session was ended before the handshake started.
    GatewayEnded,
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
            Self::GatewayEnded => "gateway_session_ended",
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
            Self::GatewayEnded => f.write_str("gateway session ended before the backend handshake"),
        }
    }
}

impl From<McpSessionError> for McpBackendError {
    fn from(error: McpSessionError) -> Self {
        match error {
            McpSessionError::Timeout => Self::Timeout,
            McpSessionError::ResponseTooLarge { limit } => Self::ResponseTooLarge { limit },
            McpSessionError::Rejected { status } => Self::Rejected { status },
            McpSessionError::JsonRpc { code } => Self::JsonRpc { code },
            McpSessionError::Malformed(detail) => Self::Malformed(detail),
            McpSessionError::Transport => Self::Transport,
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

/// [`McpBackendClient`] backed by [`McpSessionClient`] over the broker's
/// generated outbound chain.
pub(crate) struct FilteredMcpBackendClient {
    /// Shared MCP session client bound to the handshake chain.
    client: McpSessionClient,
}

impl FilteredMcpBackendClient {
    /// Build a client over the bound handshake chain.
    pub(crate) fn new(pipeline: Arc<FilterPipeline>, max_response_bytes: usize, step_timeout: Duration) -> Self {
        Self {
            client: McpSessionClient::new(pipeline, max_response_bytes, step_timeout),
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
        // Unpinned: the load balancer picks the endpoint, and the selector
        // records it so later exchanges reach the session's endpoint.
        let staged = BrokerBackendCluster::new(Arc::clone(target.cluster), None);
        let dialed = Arc::clone(&staged.dialed);
        let init = Box::pin(
            self.client
                .initialize(target.path, Arc::new(staged), callout, protocol_version),
        )
        .await?;
        Ok(BackendInit {
            protocol_version: init.protocol_version,
            session_id: init.session_id,
            endpoint: dialed.get().cloned(),
        })
    }

    async fn send_initialized(
        &self,
        target: BackendTarget<'_>,
        session: &BackendInit,
        callout: &BackendCallout,
    ) -> Result<(), McpBackendError> {
        let stage = Arc::new(pinned_stage(target, session));
        Box::pin(
            self.client
                .send_initialized(target.path, stage, callout, &session_init(session)),
        )
        .await
        .map_err(McpBackendError::from)
    }

    async fn terminate(
        &self,
        target: BackendTarget<'_>,
        session: &BackendInit,
        callout: &BackendCallout,
    ) -> Result<(), McpBackendError> {
        let stage = Arc::new(pinned_stage(target, session));
        Box::pin(
            self.client
                .terminate(target.path, stage, callout, &session_init(session)),
        )
        .await
        .map_err(McpBackendError::from)
    }
}

/// Destination for an exchange on an existing session: its cluster, pinned to
/// the endpoint that holds the session.
fn pinned_stage(target: BackendTarget<'_>, session: &BackendInit) -> BrokerBackendCluster {
    BrokerBackendCluster::new(Arc::clone(target.cluster), session.endpoint.clone())
}

/// The client-side view of a broker backend session.
fn session_init(session: &BackendInit) -> McpSessionInit {
    McpSessionInit {
        protocol_version: session.protocol_version.clone(),
        session_id: session.session_id.clone(),
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, reason = "tests")]
mod tests {
    use super::*;

    #[test]
    fn session_errors_keep_their_class() {
        assert_eq!(
            McpBackendError::from(McpSessionError::ResponseTooLarge { limit: 7 }),
            McpBackendError::ResponseTooLarge { limit: 7 },
            "size limit must survive the mapping"
        );
        assert_eq!(
            McpBackendError::from(McpSessionError::Rejected { status: 401 }),
            McpBackendError::Rejected { status: 401 },
            "status must survive the mapping"
        );
        assert_eq!(
            McpBackendError::from(McpSessionError::JsonRpc { code: -32602 }),
            McpBackendError::JsonRpc { code: -32602 },
            "JSON-RPC code must survive the mapping"
        );
        assert_eq!(
            McpBackendError::from(McpSessionError::Timeout).reason(),
            "timeout",
            "timeouts stay distinct from transport failures"
        );
        assert_eq!(
            McpBackendError::from(McpSessionError::Transport).reason(),
            "transport",
            "transport failures keep their reason"
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

    #[test]
    fn pinned_stage_targets_the_session_endpoint() {
        let cluster: Arc<str> = Arc::from("weather-mcp");
        let target = BackendTarget {
            cluster: &cluster,
            path: "/mcp",
        };
        let session = BackendInit {
            protocol_version: Box::from("2025-06-18"),
            session_id: Some(HeaderValue::from_static("s-1")),
            endpoint: Some(Arc::from("10.0.0.2:3001")),
        };
        let stage = pinned_stage(target, &session);
        assert_eq!(&*stage.cluster, "weather-mcp", "the session's cluster is dialed");
        assert_eq!(
            stage.pinned_endpoint.as_deref(),
            Some("10.0.0.2:3001"),
            "later exchanges reach the endpoint holding the session"
        );
        assert_eq!(
            session_init(&session).session_id.unwrap(),
            "s-1",
            "the session id is carried to the client"
        );
    }
}
