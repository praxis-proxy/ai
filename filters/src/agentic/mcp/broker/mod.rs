// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! MCP static catalog filter: static tool catalog, prefix management, broker
//! behavior for `tools/list`, current-profile `initialize`, `ping`, and
//! `notifications`, stateless-profile `tools/call` routing, and
//! current-profile `tools/call` routing over lazily initialized backend
//! sessions.

mod backend;
pub(crate) mod config;
mod selector;
#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::needless_raw_strings,
    clippy::needless_raw_string_hashes,
    clippy::too_many_lines,
    reason = "tests"
)]
mod session_tests;
mod sessions;

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::err_expect,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::needless_raw_strings,
    clippy::needless_raw_string_hashes,
    reason = "tests"
)]
mod tests;

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use async_trait::async_trait;
use base64::Engine as _;
use bytes::Bytes;
use praxis_ai_apis::{RecordAiMetadata as _, promotion::contains_control_chars};
use praxis_core::{
    config::{ChainRef, FilterEntry},
    subrequest::{DEPTH_HEADER, SubRequestClient},
};
use praxis_filter::{
    BodyAccess, BodyMode, ChainBindingContext, FilterAction, FilterError, HttpFilter, HttpFilterContext,
    IterationState, Rejection, SubrequestRuntime, TraceContext,
    builtins::http::payload_processing::json_rpc::{
        config::JsonRpcConfig,
        envelope::{JsonRpcEnvelope, JsonRpcIdKind, JsonRpcKind, parse_json_rpc_value},
    },
    parse_filter_config,
};
use tracing::{debug, trace, warn};

pub use self::selector::McpBrokerBackendSelectorFilter;
use self::{
    backend::{
        BackendCallout, BackendInit, BackendTarget, FilteredMcpBackendClient, MCP_PROTOCOL_VERSION, MCP_SESSION_ID,
        McpBackendClient, McpBackendError,
    },
    config::{CacheScope, CatalogTool, McpBrokerConfig, McpServerConfig, ValidatedBackendSessions, build_config},
    sessions::{
        BackendSessionKey, BackendSessionStore, BoundSession, Publish, RetiredSession, SessionSlot,
        credential_fingerprint,
    },
};
use super::protocol::ProtocolProfile;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Server name reported in MCP responses.
const SERVER_NAME: &str = "praxis";

/// Server version reported in MCP responses.
const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// JSON-RPC error code for stateless header mismatch.
const ERR_HEADER_MISMATCH: i32 = -32020;

/// JSON-RPC error code for unsupported protocol version.
const ERR_UNSUPPORTED_VERSION: i32 = -32022;

/// JSON-RPC error code for invalid params.
const ERR_INVALID_PARAMS: i32 = -32602;

/// MCP `=?base64?` sentinel prefix.
const BASE64_SENTINEL_PREFIX: &str = "=?base64?";

/// MCP `=?base64?` sentinel suffix.
const BASE64_SENTINEL_SUFFIX: &str = "?=";

/// JSON-RPC error code for invalid requests.
const ERR_INVALID_REQUEST: i32 = -32600;

/// JSON-RPC error code for internal errors (backend session failures).
const ERR_INTERNAL: i32 = -32603;

/// Maximum accepted gateway `Mcp-Session-Id` length.
const MAX_GATEWAY_SESSION_ID_LEN: usize = 256;

/// Name of the generated backend handshake chain.
const BACKEND_CHAIN_NAME: &str = "mcp-backend-sessions";

// -----------------------------------------------------------------------------
// Backend Sessions
// -----------------------------------------------------------------------------

/// Routing facts for one configured backend server.
#[derive(Debug)]
struct BrokerServer {
    /// Top-level cluster name.
    cluster: Arc<str>,
    /// Backend MCP endpoint path.
    path: String,
}

/// Most retired backend sessions terminated concurrently; beyond that the
/// backend's own idle expiry applies.
const MAX_CONCURRENT_CLEANUPS: usize = 16;

/// Current-profile backend session runtime.
struct BackendSessions {
    /// Bounds concurrent background terminations of retired sessions.
    cleanup: Arc<tokio::sync::Semaphore>,
    /// Lifecycle callout client.
    client: Arc<dyn McpBackendClient>,
    /// Transport used when the request context exposes no shared client.
    fallback_client: Option<SubRequestClient>,
    /// Client headers copied onto handshake exchanges.
    forward_headers: Vec<http::HeaderName>,
    /// Deadline for one handshake exchange.
    handshake_timeout: Duration,
    /// Gateway → backend session map.
    store: BackendSessionStore,
    /// Protocol versions a backend may negotiate.
    supported_versions: Vec<String>,
}

/// Per-request record of a forwarded current-profile `tools/call`, read by
/// `on_response` (backend answered) or `on_request_complete` (it never did)
/// to invalidate a session that expired or whose endpoint is unreachable.
struct ForwardedCall {
    /// Whether a backend response arrived.
    answered: bool,
    /// Session map key.
    key: BackendSessionKey,
    /// The slot this call uses, held until the request completes so capacity
    /// pressure or idle expiry cannot evict the mapping mid-call and let an
    /// overlapping call open a second backend session. Dropping the request
    /// context releases it on every path.
    slot: Arc<SessionSlot>,
    /// Backend session id sent upstream.
    session_id: http::HeaderValue,
    /// Endpoint the backend session lives on.
    ///
    /// The broker resolves the session in the request-body phase, but core
    /// rebuilds `pinned_endpoint_address` empty for every phase, so the pin is
    /// carried here and applied in the header phase that runs the
    /// `load_balancer`.
    endpoint: Option<Arc<str>>,
}

// -----------------------------------------------------------------------------
// McpBrokerFilter
// -----------------------------------------------------------------------------

/// MCP static catalog filter that aggregates tool catalogs from multiple backend
/// MCP servers and handles `tools/list`, `tools/call`, and current-profile
/// broker methods (`initialize`, `ping`, and `notifications/initialized`).
///
/// In the stateless profile, `tools/call` routes to the configured backend
/// cluster by exposed tool name, stripping the public prefix from `params.name`
/// and repairing the forwarded `Mcp-Name` header before forwarding.
///
/// In the current profile, `tools/call` returns `-32601` unless
/// `backend_sessions` is configured. With it, the broker lazily runs the
/// backend `initialize` / `notifications/initialized` handshake once per
/// `(Mcp-Session-Id, server)` pair through a generated outbound chain that
/// dials the server's top-level cluster, then forwards the call to that
/// cluster with the backend `Mcp-Session-Id`. A backend `404` drops the
/// mapping so the next call reinitializes. Server clusters must be declared
/// in top-level `clusters:` so the handshake chain can reference them.
///
/// # YAML
///
/// ```yaml
/// filter: mcp
/// path: /mcp
/// max_body_bytes: 65536
/// servers:
///   - name: weather
///     cluster: weather-mcp
///     path: /mcp
///     tool_prefix: weather_
///     tools:
///       - name: get_weather
///         description: Get current weather
///   - name: calendar
///     cluster: calendar-mcp
///     path: /mcp
///     tool_prefix: cal_
///     tools:
///       - name: create_event
///         description: Create a calendar event
/// ```
pub(crate) struct McpBrokerFilter {
    /// Current-profile backend session runtime, when configured.
    backend: Option<BackendSessions>,
    /// Cache scope for stateless responses.
    cache_scope: CacheScope,
    /// Cache TTL in milliseconds for stateless responses.
    cache_ttl_ms: u64,
    /// Static tool catalog built from config.
    catalog: Vec<CatalogTool>,
    /// Protocol version the broker uses in responses.
    default_version: String,
    /// Shared JSON-RPC parser configuration.
    json_rpc_config: JsonRpcConfig,
    /// Maximum body bytes for `StreamBuffer`.
    max_body_bytes: usize,
    /// Configured protocol profile.
    protocol_profile: ProtocolProfile,
    /// Public path this MCP broker handles (e.g. `/mcp`).
    public_path: String,
    /// Configured servers in configuration order.
    servers: Vec<BrokerServer>,
    /// Implemented versions used for protocol version negotiation.
    supported_versions: Vec<String>,
}

impl McpBrokerFilter {
    /// Return true when this MCP config selects static catalog behavior.
    pub(crate) fn matches_config(config: &serde_yaml::Value) -> bool {
        config.get("servers").is_some()
    }

    /// Create a filter from parsed YAML config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the YAML config is invalid or if
    /// the static tool catalog cannot be serialized.
    #[cfg(test)]
    pub(crate) fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        Self::from_config_with_binding(config, None, None)
    }

    /// Create a filter, binding the backend handshake chain when
    /// `backend_sessions` is configured.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the config is invalid, if
    /// `backend_sessions` is configured without a chain-binding context, or
    /// if the generated handshake chain cannot be bound.
    pub(crate) fn from_config_with_binding(
        config: &serde_yaml::Value,
        binding: Option<&ChainBindingContext<'_>>,
        fallback_client: Option<SubRequestClient>,
    ) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: McpBrokerConfig = parse_filter_config("mcp", config)?;
        let handshake_filters = cfg
            .backend_sessions
            .as_ref()
            .map(|sessions| sessions.outbound_filters.clone());
        let servers_cfg = cfg.servers.clone();
        let (validated, catalog) = build_config(cfg)?;
        let backend = match (validated.backend_sessions.as_ref(), handshake_filters) {
            (Some(settings), Some(filters)) => {
                let Some(binding) = binding else {
                    return Err("mcp: backend_sessions requires chain-binding registration".into());
                };
                let pipeline = Arc::new(bind_backend_chain(&servers_cfg, filters, binding)?);
                let client = Arc::new(FilteredMcpBackendClient::new(
                    pipeline,
                    settings.max_response_bytes,
                    settings.handshake_timeout,
                ));
                Some(BackendSessions::new(
                    client,
                    fallback_client,
                    settings,
                    validated.supported_versions.clone(),
                ))
            },
            _ => None,
        };
        Ok(Box::new(Self::build(&validated, catalog, &servers_cfg, backend)))
    }

    /// Assemble the filter from validated parts.
    fn build(
        validated: &config::ValidatedBrokerConfig,
        catalog: Vec<CatalogTool>,
        servers: &[McpServerConfig],
        backend: Option<BackendSessions>,
    ) -> Self {
        let json_rpc_config = super::build_json_rpc_config(validated.max_body_bytes);

        Self {
            backend,
            cache_scope: validated.cache_scope,
            cache_ttl_ms: validated.cache_ttl_ms,
            catalog,
            default_version: validated.default_version.clone(),
            json_rpc_config,
            max_body_bytes: validated.max_body_bytes,
            protocol_profile: validated.protocol_profile,
            public_path: validated.path.clone(),
            servers: servers
                .iter()
                .map(|server| BrokerServer {
                    cluster: Arc::from(server.cluster.as_str()),
                    path: server.path.clone(),
                })
                .collect(),
            supported_versions: validated.supported_versions.clone(),
        }
    }

    // -------------------------------------------------------------------------
    // Method Dispatch
    // -------------------------------------------------------------------------

    /// Maps a JSON-RPC method to the MCP handler that owns it.
    #[expect(clippy::too_many_arguments, reason = "body reference needed for tools/call mutation")]
    async fn dispatch_method(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        value: &mut serde_json::Value,
        envelope: &JsonRpcEnvelope,
        method_str: &str,
        body: &mut Option<Bytes>,
    ) -> Result<FilterAction, FilterError> {
        match self.protocol_profile {
            ProtocolProfile::Current => self.dispatch_current(ctx, value, envelope, method_str, body).await,
            ProtocolProfile::Stateless => Ok(self.dispatch_stateless(ctx, value, envelope, method_str, body)),
        }
    }

    /// Current-profile dispatch.
    #[expect(clippy::too_many_arguments, reason = "body reference needed for tools/call mutation")]
    async fn dispatch_current(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        value: &mut serde_json::Value,
        envelope: &JsonRpcEnvelope,
        method_str: &str,
        body: &mut Option<Bytes>,
    ) -> Result<FilterAction, FilterError> {
        if method_str.starts_with("notifications/") {
            return Ok(handle_notification(envelope));
        }

        if !has_valid_request_id(envelope) {
            return Ok(invalid_request_action(envelope));
        }

        let action = match method_str {
            "initialize" => handle_initialize(ctx, value, envelope, &self.supported_versions, &self.default_version)?,
            "tools/list" => handle_tools_list(&self.catalog, envelope)?,
            "tools/call" => self.handle_current_tools_call(ctx, value, envelope, body).await,
            "ping" => handle_ping(envelope),
            _ => {
                debug!(method_len = method_str.len(), "unsupported MCP method");
                json_rpc_error_action(envelope, -32601, "method not found")
            },
        };

        Ok(action)
    }

    /// Stateless-profile dispatch: validates stateless headers, then dispatches.
    #[expect(clippy::too_many_arguments, reason = "body reference needed for tools/call mutation")]
    #[expect(clippy::too_many_lines, reason = "linear dispatch with early returns")]
    fn dispatch_stateless(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        value: &serde_json::Value,
        envelope: &JsonRpcEnvelope,
        method_str: &str,
        body: &mut Option<Bytes>,
    ) -> FilterAction {
        let is_notification = method_str.starts_with("notifications/");
        if is_notification {
            if let Some(action) = self.validate_stateless_headers(value, &ctx.request.headers, envelope, method_str) {
                return action;
            }

            return json_rpc_error_action_with_status(envelope, 404, -32601, "method not found");
        }

        if !has_valid_request_id(envelope) {
            return invalid_request_action(envelope);
        }

        if method_str == "tools/call"
            && let Err(detail) = extract_tool_name(value)
        {
            return invalid_params_action(envelope, detail);
        }

        if let Some(action) = self.validate_stateless_headers(value, &ctx.request.headers, envelope, method_str) {
            return action;
        }

        match method_str {
            "server/discover" => self.handle_server_discover(envelope),
            "tools/list" => self.handle_stateless_tools_list(envelope),
            "tools/call" => self.handle_stateless_tools_call(ctx, value, envelope, body),
            "initialize" => json_rpc_error_action_with_status(
                envelope,
                404,
                -32601,
                "method not found: use server/discover for stateless profiles",
            ),
            _ => {
                debug!(method_len = method_str.len(), "unsupported MCP method");
                json_rpc_error_action_with_status(envelope, 404, -32601, "method not found")
            },
        }
    }

    /// Profile-aware DELETE handler.
    async fn handle_delete(&self, ctx: &HttpFilterContext<'_>) -> FilterAction {
        match self.protocol_profile {
            ProtocolProfile::Current => {
                let action = handle_delete_current(ctx);
                if let (Some(backend), Some(gateway)) = (self.backend.as_ref(), gateway_session_id(ctx)) {
                    self.terminate_backend_sessions(ctx, backend, gateway).await;
                }
                action
            },
            ProtocolProfile::Stateless => FilterAction::Reject(Rejection::status(405)),
        }
    }

    // -------------------------------------------------------------------------
    // Current-Profile Backend Sessions
    // -------------------------------------------------------------------------

    /// Route a current-profile `tools/call` over a lazily initialized backend
    /// session, or return `-32601` when `backend_sessions` is not configured.
    async fn handle_current_tools_call(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        value: &mut serde_json::Value,
        envelope: &JsonRpcEnvelope,
        body: &mut Option<Bytes>,
    ) -> FilterAction {
        let Some(backend) = self.backend.as_ref() else {
            return json_rpc_error_action(envelope, -32601, "method not yet supported");
        };
        let (tool, server, key, version) = match self.current_call_route(ctx, value, envelope) {
            Ok(route) => route,
            Err(action) => return action,
        };
        let Some(callout) = backend.callout(ctx) else {
            return backend_error_action(envelope, &tool.server_name, &McpBackendError::Transport);
        };
        let target = BackendTarget {
            cluster: &server.cluster,
            path: &server.path,
        };
        let session = backend.ensure_session(&key, target, version, &callout).await;
        backend.retire(&self.servers, &callout);
        let (slot, init) = match session {
            Ok(session) => session,
            Err(error) => {
                warn!(server = %tool.server_name, reason = error.reason(), "mcp backend session handshake failed");
                return backend_error_action(envelope, &tool.server_name, &error);
            },
        };

        forward_current_tools_call(ctx, value, body, tool, &init);
        record_forwarded_call(ctx, key, slot, init);
        FilterAction::Release
    }

    /// Resolve the catalog tool, its server, the session map key, and the
    /// backend protocol version for a current-profile `tools/call`, or the
    /// error to return.
    fn current_call_route(
        &self,
        ctx: &HttpFilterContext<'_>,
        value: &serde_json::Value,
        envelope: &JsonRpcEnvelope,
    ) -> Result<(&CatalogTool, &BrokerServer, BackendSessionKey, &str), FilterAction> {
        let tool_name =
            extract_tool_name(value).map_err(|detail| json_rpc_error_action(envelope, ERR_INVALID_PARAMS, detail))?;
        let tool = self
            .catalog
            .iter()
            .find(|t| t.exposed_name == tool_name)
            .ok_or_else(|| json_rpc_error_action(envelope, ERR_INVALID_PARAMS, "unknown tool"))?;
        let gateway = gateway_session_id(ctx).ok_or_else(|| {
            json_rpc_error_action_with_status(envelope, 400, ERR_INVALID_REQUEST, "missing or invalid Mcp-Session-Id")
        })?;
        let server = self
            .servers
            .get(tool.server_index)
            .ok_or_else(|| json_rpc_error_action(envelope, ERR_INTERNAL, "unknown backend server"))?;
        let key = BackendSessionKey {
            gateway_session: Box::from(gateway),
            server: tool.server_index,
        };
        let version = self.requested_backend_version(ctx, envelope)?;
        Ok((tool, server, key, version))
    }

    /// Protocol version offered to the backend: the client's
    /// `MCP-Protocol-Version`, or the broker default when the header is absent.
    /// A present but unsupported or undecodable version is rejected with
    /// HTTP 400.
    fn requested_backend_version(
        &self,
        ctx: &HttpFilterContext<'_>,
        envelope: &JsonRpcEnvelope,
    ) -> Result<&str, FilterAction> {
        let Some(value) = ctx.request.headers.get("mcp-protocol-version") else {
            return Ok(self.default_version.as_str());
        };
        // The raw bytes are not echoed back.
        let requested = value
            .to_str()
            .map_err(|_invalid| unsupported_version_action(envelope, "", &self.supported_versions))?;
        self.supported_versions
            .iter()
            .find(|v| v.as_str() == requested)
            .map(String::as_str)
            .ok_or_else(|| unsupported_version_action(envelope, requested, &self.supported_versions))
    }

    /// Best-effort termination of every backend session a gateway session owns.
    async fn terminate_backend_sessions(&self, ctx: &HttpFilterContext<'_>, backend: &BackendSessions, gateway: &str) {
        let Some(callout) = backend.callout(ctx) else {
            return;
        };
        let credential = credential_fingerprint(&callout.forwarded_headers);
        let removed = backend.store.remove_gateway(gateway, &credential);
        for retired in removed {
            let Some(server) = self.servers.get(retired.server) else {
                continue;
            };
            let target = BackendTarget {
                cluster: &server.cluster,
                path: &server.path,
            };
            if let Err(error) = backend.client.terminate(target, &retired.session.init, &callout).await {
                debug!(reason = error.reason(), "mcp backend session termination failed");
            }
        }
        backend.retire(&self.servers, &callout);
    }

    // -------------------------------------------------------------------------
    // Stateless Header Validation
    // -------------------------------------------------------------------------

    /// Validates stateless request metadata headers.
    ///
    /// Returns `Some(FilterAction)` if validation fails, `None` if it passes.
    fn validate_stateless_headers(
        &self,
        value: &serde_json::Value,
        request_headers: &http::HeaderMap,
        envelope: &JsonRpcEnvelope,
        method_str: &str,
    ) -> Option<FilterAction> {
        let header_method = header_str(request_headers, "mcp-method");
        let header_name = header_str(request_headers, "mcp-name");
        let params_meta = value.get("params").and_then(|p| p.get("_meta"));

        if let Some(action) = self.validate_protocol_version(params_meta, request_headers, envelope) {
            return Some(action);
        }

        if let Some(action) = validate_mcp_method(header_method, envelope, method_str) {
            return Some(action);
        }

        if let Some(msg) = validate_params_meta(params_meta) {
            return Some(header_mismatch_action(envelope, msg));
        }

        validate_mcp_name_header(value, envelope, method_str, header_name)
    }

    /// Validates `MCP-Protocol-Version` header and `params._meta` version.
    fn validate_protocol_version(
        &self,
        params_meta: Option<&serde_json::Value>,
        request_headers: &http::HeaderMap,
        envelope: &JsonRpcEnvelope,
    ) -> Option<FilterAction> {
        let header_version = header_str(request_headers, "mcp-protocol-version");

        let body_version = params_meta
            .and_then(|m| m.get("io.modelcontextprotocol/protocolVersion"))
            .and_then(|v| v.as_str());

        let Some(header_version) = header_version else {
            return Some(header_mismatch_action(envelope, "missing MCP-Protocol-Version header"));
        };

        let Some(body_version) = body_version else {
            return Some(header_mismatch_action(
                envelope,
                "missing params._meta[\"io.modelcontextprotocol/protocolVersion\"]",
            ));
        };

        if header_version != body_version {
            return Some(header_mismatch_action(
                envelope,
                "MCP-Protocol-Version header does not match body _meta protocol version",
            ));
        }

        if !self.supported_versions.iter().any(|v| v == header_version) {
            return Some(unsupported_version_action(
                envelope,
                header_version,
                &self.supported_versions,
            ));
        }

        None
    }

    // -------------------------------------------------------------------------
    // Stateless Response Builders
    // -------------------------------------------------------------------------

    /// `server/discover` response for the stateless profile.
    fn handle_server_discover(&self, envelope: &JsonRpcEnvelope) -> FilterAction {
        let response = serde_json::json!({
            "jsonrpc": "2.0",
            "id": id_value(envelope),
            "result": {
                "cacheScope": self.cache_scope.as_str(),
                "capabilities": {
                    "tools": { "listChanged": false },
                },
                "resultType": "complete",
                "serverInfo": {
                    "name": SERVER_NAME,
                    "version": SERVER_VERSION,
                },
                "supportedVersions": self.supported_versions,
                "ttlMs": self.cache_ttl_ms,
            },
        });

        trace!("serving server/discover");

        FilterAction::Reject(
            Rejection::status(200)
                .with_header("content-type", "application/json")
                .with_body(Bytes::from(response.to_string())),
        )
    }

    /// Route a stateless `tools/call` to the configured backend cluster.
    fn handle_stateless_tools_call(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        value: &serde_json::Value,
        envelope: &JsonRpcEnvelope,
        body: &mut Option<Bytes>,
    ) -> FilterAction {
        let tool_name = match extract_tool_name(value) {
            Ok(name) => name,
            Err(detail) => return invalid_params_action(envelope, detail),
        };

        let Some(tool) = self.catalog.iter().find(|t| t.exposed_name == tool_name) else {
            return invalid_params_action(envelope, "unknown tool");
        };

        ctx.cluster = Some(Arc::from(tool.cluster.as_str()));
        ctx.rewritten_path = Some(tool.backend_path.clone());

        let mut mutated = value.clone();
        if let Some(p) = mutated.get_mut("params").and_then(|p| p.as_object_mut()) {
            p.insert("name".to_owned(), serde_json::Value::String(tool.original_name.clone()));
        }
        *body = Some(Bytes::from(mutated.to_string()));

        ctx.request_headers_to_set.push((
            http::header::HeaderName::from_static("mcp-name"),
            encode_mcp_name(&tool.original_name),
        ));

        trace!(
            exposed = tool_name,
            original = tool.original_name,
            cluster = tool.cluster,
            server = tool.server_name,
            "routing stateless tools/call"
        );

        FilterAction::Release
    }

    /// `tools/list` response for the stateless profile with cache metadata.
    fn handle_stateless_tools_list(&self, envelope: &JsonRpcEnvelope) -> FilterAction {
        let tools: Vec<serde_json::Value> = self.catalog.iter().map(catalog_tool_to_json).collect();

        let response = serde_json::json!({
            "jsonrpc": "2.0",
            "id": id_value(envelope),
            "result": {
                "cacheScope": self.cache_scope.as_str(),
                "resultType": "complete",
                "tools": tools,
                "ttlMs": self.cache_ttl_ms,
            },
        });

        trace!(tool_count = self.catalog.len(), "serving stateless tools/list");

        FilterAction::Reject(
            Rejection::status(200)
                .with_header("content-type", "application/json")
                .with_body(Bytes::from(response.to_string())),
        )
    }
}

#[async_trait]
impl HttpFilter for McpBrokerFilter {
    fn name(&self) -> &'static str {
        "mcp"
    }

    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadWrite(praxis_filter::BodyRequirement::Optional)
    }

    fn request_body_mode(&self) -> BodyMode {
        BodyMode::StreamBuffer {
            max_bytes: Some(self.max_body_bytes),
        }
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        if !request_path_matches(&ctx.request.uri, &self.public_path) {
            return Ok(FilterAction::Reject(Rejection::status(404)));
        }

        match ctx.request.method {
            http::Method::POST => {
                // With a pre-read body the body phase already resolved the
                // backend session; pin its endpoint for the load_balancer that
                // runs later in this phase.
                let pinned = ctx
                    .get_filter_state::<ForwardedCall>()
                    .and_then(|call| call.endpoint.clone());
                if pinned.is_some() {
                    ctx.pinned_endpoint_address = pinned;
                }
                Ok(FilterAction::Continue)
            },
            http::Method::DELETE => Ok(self.handle_delete(ctx).await),
            _ => Ok(FilterAction::Reject(Rejection::status(405))),
        }
    }

    async fn on_request_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if ctx.request.method != http::Method::POST {
            return Ok(FilterAction::Continue);
        }

        if !request_path_matches(&ctx.request.uri, &self.public_path) {
            return Ok(FilterAction::Reject(Rejection::status(404)));
        }

        if !end_of_stream {
            return Ok(FilterAction::Continue);
        }

        let Some(chunk) = body.as_ref() else {
            return Ok(FilterAction::Continue);
        };

        let Ok(mut value) = serde_json::from_slice::<serde_json::Value>(chunk) else {
            return Ok(FilterAction::Reject(Rejection::status(400)));
        };

        let Ok(Some(envelope)) = parse_json_rpc_value(&value, &self.json_rpc_config) else {
            return Ok(FilterAction::Reject(Rejection::status(400)));
        };

        let Some(method_str) = &envelope.method else {
            return Ok(FilterAction::Reject(Rejection::status(400)));
        };

        if !contains_control_chars(method_str) {
            ctx.record_ai_metadata("json_rpc.method", method_str.clone());
            ctx.record_ai_metadata("mcp.method", method_str.clone());
        }

        self.dispatch_method(ctx, &mut value, &envelope, method_str, body).await
    }

    async fn on_response(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        // Without response headers no backend answered (core runs this hook
        // during cleanup after an upstream failure); `on_request_complete`
        // settles the call with the status the client received.
        let Some(status) = ctx.response_header.as_ref().map(|response| response.status) else {
            return Ok(FilterAction::Continue);
        };
        // The record stays until completion: it holds the session slot.
        let Some(call) = ctx.get_filter_state_mut::<ForwardedCall>() else {
            return Ok(FilterAction::Continue);
        };
        call.answered = true;
        // 404: the backend forgot the session. 502: an intermediary reports
        // the pinned endpoint unreachable; the session cannot move elsewhere.
        if matches!(status, http::StatusCode::NOT_FOUND | http::StatusCode::BAD_GATEWAY)
            && let Some(backend) = self.backend.as_ref()
        {
            backend.invalidate_forwarded(
                &call.key,
                &call.session_id,
                "mcp backend session expired or unreachable; mapping invalidated",
            );
        }
        Ok(FilterAction::Continue)
    }

    fn observes_request_completion(&self) -> bool {
        self.backend.is_some()
    }

    fn on_request_complete(&self, ctx: &mut HttpFilterContext<'_>, status: u16) {
        // Completion releases the call's slot. A call no backend answered
        // that ends in a proxy 502 or 504 could not reach its pinned endpoint
        // (refused, reset, TLS failure, or timed out), so the session it holds
        // is unusable; local rejections keep it.
        let Some(ForwardedCall {
            answered,
            key,
            slot,
            session_id,
            ..
        }) = ctx.remove_filter_state::<ForwardedCall>()
        else {
            return;
        };
        if let Some(backend) = self.backend.as_ref() {
            backend.store.release(&key, slot);
        }
        if !answered
            && matches!(status, 502 | 504)
            && let Some(backend) = self.backend.as_ref()
        {
            backend.invalidate_forwarded(
                &key,
                &session_id,
                "mcp backend endpoint unreachable; mapping invalidated",
            );
        }
    }
}

impl BackendSessions {
    /// Build the runtime from validated settings.
    fn new(
        client: Arc<dyn McpBackendClient>,
        fallback_client: Option<SubRequestClient>,
        settings: &ValidatedBackendSessions,
        supported_versions: Vec<String>,
    ) -> Self {
        Self {
            cleanup: Arc::new(tokio::sync::Semaphore::new(MAX_CONCURRENT_CLEANUPS)),
            client,
            fallback_client,
            forward_headers: settings.forward_headers.clone(),
            handshake_timeout: settings.handshake_timeout,
            store: BackendSessionStore::new(settings.max_sessions, settings.idle_timeout),
            supported_versions,
        }
    }

    /// Drop the mapping a forwarded call used, if it still holds that session.
    fn invalidate_forwarded(&self, key: &BackendSessionKey, session_id: &http::HeaderValue, message: &'static str) {
        if self.store.invalidate(key, session_id) {
            debug!(server = key.server, "{message}");
        }
    }

    /// Capture request-scoped callout resources, or `None` without a transport.
    fn callout(&self, ctx: &HttpFilterContext<'_>) -> Option<BackendCallout> {
        let client = ctx
            .subrequest_client
            .cloned()
            .or_else(|| self.fallback_client.clone())?;
        let iteration_state = ctx.extensions.get::<IterationState>();
        let now = Instant::now();
        let step_deadline = now.checked_add(self.handshake_timeout).unwrap_or(now);
        let deadline = iteration_state
            .map(IterationState::deadline)
            .map_or(step_deadline, |parent| parent.min(step_deadline));
        Some(BackendCallout {
            client,
            deadline,
            depth: callout_depth(iteration_state, &ctx.request.headers),
            downstream: SubrequestRuntime::new(
                ctx.client_addr,
                ctx.downstream_tls,
                ctx.peer_identity.clone(),
                ctx.request_start,
            ),
            forwarded_headers: self.forwarded_headers(&ctx.request.headers),
            trace_context: ctx.extensions.get::<TraceContext>().cloned(),
        })
    }

    /// Copy the allowlisted client headers onto a handshake exchange.
    fn forwarded_headers(&self, request_headers: &http::HeaderMap) -> http::HeaderMap {
        let mut forwarded = http::HeaderMap::new();
        for name in &self.forward_headers {
            for value in request_headers.get_all(name) {
                forwarded.append(name.clone(), value.clone());
            }
        }
        forwarded
    }

    /// Return the backend session for `key`, running the handshake when the
    /// key has none or when the caller's credentials differ from the ones
    /// that created it. Concurrent callers for one key share one handshake.
    async fn ensure_session(
        &self,
        key: &BackendSessionKey,
        target: BackendTarget<'_>,
        protocol_version: &str,
        callout: &BackendCallout,
    ) -> Result<(Arc<SessionSlot>, BackendInit), McpBackendError> {
        let credential = credential_fingerprint(&callout.forwarded_headers);
        let slot = self.store.slot(key).ok_or(McpBackendError::Capacity)?;
        if let Some(init) = slot.current(&credential) {
            return Ok((slot, init));
        }
        let (outcome, init) = self
            .serialized_handshake(&slot, target, protocol_version, callout)
            .await?;
        match outcome {
            None => Ok((slot, init)),
            Some(Publish::Published(replaced)) => {
                if let Some(session) = replaced {
                    self.store.push_retired(RetiredSession {
                        server: key.server,
                        session,
                    });
                }
                Ok((slot, init))
            },
            Some(Publish::Cancelled(session)) => {
                self.abandon(target, &session.init, callout).await;
                Err(McpBackendError::SessionClosed)
            },
        }
    }

    /// Run the handshake under the slot's permit. Returns `None` for the
    /// outcome when a concurrent caller published a usable session first.
    async fn serialized_handshake(
        &self,
        slot: &SessionSlot,
        target: BackendTarget<'_>,
        protocol_version: &str,
        callout: &BackendCallout,
    ) -> Result<(Option<Publish>, BackendInit), McpBackendError> {
        let _permit = tokio::time::timeout_at(callout.deadline.into(), slot.handshake_permit())
            .await
            .map_err(|_elapsed| McpBackendError::Timeout)?
            .map_err(|_closed| McpBackendError::SessionClosed)?;
        let credential = credential_fingerprint(&callout.forwarded_headers);
        if let Some(init) = slot.current(&credential) {
            return Ok((None, init));
        }
        slot.begin(credential);
        match self.handshake(target, protocol_version, callout).await {
            Ok(init) => {
                let outcome = slot.publish(BoundSession::new(&callout.forwarded_headers, init.clone()));
                Ok((Some(outcome), init))
            },
            Err(error) => {
                slot.abort();
                Err(error)
            },
        }
    }

    /// Terminate the sessions the map retired (evicted, expired, or replaced
    /// after a credential change) in the background, each with the
    /// credentials that created it.
    fn retire(&self, servers: &[BrokerServer], callout: &BackendCallout) {
        for retired in self.store.take_retired() {
            self.spawn_terminate(servers, callout, retired);
        }
    }

    /// A callout outliving the request, carrying the retired session's own
    /// credentials and a fresh per-exchange deadline.
    fn detached_callout(&self, callout: &BackendCallout, forwarded_headers: http::HeaderMap) -> BackendCallout {
        let now = Instant::now();
        BackendCallout {
            client: callout.client.clone(),
            deadline: now.checked_add(self.handshake_timeout).unwrap_or(now),
            depth: callout.depth,
            downstream: callout.downstream.clone(),
            forwarded_headers,
            trace_context: callout.trace_context.clone(),
        }
    }

    /// Spawn one bounded best-effort termination.
    fn spawn_terminate(&self, servers: &[BrokerServer], callout: &BackendCallout, retired: RetiredSession) {
        let Some(server) = servers.get(retired.server) else {
            return;
        };
        let Ok(permit) = Arc::clone(&self.cleanup).try_acquire_owned() else {
            debug!(
                server = retired.server,
                "mcp backend session cleanup saturated; backend expiry applies"
            );
            return;
        };
        let cleanup = self.detached_callout(callout, retired.session.credentials);
        let client = Arc::clone(&self.client);
        let cluster = Arc::clone(&server.cluster);
        let path = server.path.clone();
        let init = retired.session.init;
        drop(tokio::spawn(async move {
            let target = BackendTarget {
                cluster: &cluster,
                path: &path,
            };
            if let Err(error) = client.terminate(target, &init, &cleanup).await {
                debug!(
                    reason = error.reason(),
                    "mcp backend retired session termination failed"
                );
            }
            drop(permit);
        }));
    }

    /// Run `initialize` and `notifications/initialized` for a new session.
    ///
    /// A backend that answers `initialize` with a version outside
    /// `supported_versions` is never confirmed: its session is terminated and
    /// no call is forwarded.
    async fn handshake(
        &self,
        target: BackendTarget<'_>,
        protocol_version: &str,
        callout: &BackendCallout,
    ) -> Result<BackendInit, McpBackendError> {
        let init = self.client.initialize(target, protocol_version, callout).await?;
        if !self
            .supported_versions
            .iter()
            .any(|v| v.as_str() == &*init.protocol_version)
        {
            self.abandon(target, &init, callout).await;
            return Err(McpBackendError::UnsupportedVersion);
        }
        if let Err(error) = self.client.send_initialized(target, &init, callout).await {
            self.abandon(target, &init, callout).await;
            return Err(error);
        }
        Ok(init)
    }

    /// Best-effort termination of a backend session that will not be used.
    ///
    /// Runs with a fresh deadline: the handshake that failed may have used up
    /// the request's.
    async fn abandon(&self, target: BackendTarget<'_>, init: &BackendInit, callout: &BackendCallout) {
        let cleanup = self.detached_callout(callout, callout.forwarded_headers.clone());
        if let Err(cleanup) = self.client.terminate(target, init, &cleanup).await {
            debug!(reason = cleanup.reason(), "mcp backend session cleanup failed");
        }
    }
}

// -----------------------------------------------------------------------------
// Backend Session Helpers
// -----------------------------------------------------------------------------

/// Bind the generated handshake chain: the backend selector, any operator
/// `outbound_filters`, and a `load_balancer` referencing each server's
/// top-level cluster.
fn bind_backend_chain(
    servers: &[McpServerConfig],
    outbound_filters: Vec<FilterEntry>,
    binding: &ChainBindingContext<'_>,
) -> Result<praxis_filter::FilterPipeline, FilterError> {
    let mut clusters: Vec<&str> = Vec::with_capacity(servers.len());
    for server in servers {
        if !clusters.contains(&server.cluster.as_str()) {
            clusters.push(server.cluster.as_str());
        }
    }
    if clusters.is_empty() {
        return Err("mcp: backend_sessions requires at least one server".into());
    }

    let (selector, load_balancer) = generated_chain_entries(&clusters)?;
    let mut filters = Vec::with_capacity(outbound_filters.len().saturating_add(2));
    filters.push(selector);
    filters.extend(outbound_filters);
    filters.push(load_balancer);

    binding
        .bind_chain(&ChainRef::Inline {
            name: BACKEND_CHAIN_NAME.to_owned(),
            filters,
        })
        .map_err(|e| {
            FilterError::from(format!(
                "mcp: backend_sessions handshake chain failed to bind (declare each server cluster in top-level \
                 `clusters:`): {e}"
            ))
        })
}

/// Build the generated selector and `load_balancer` entries for `clusters`.
fn generated_chain_entries(clusters: &[&str]) -> Result<(FilterEntry, FilterEntry), FilterError> {
    let selector = serde_yaml::to_value(serde_json::json!({
        "filter": McpBrokerBackendSelectorFilter::NAME,
        "clusters": clusters,
    }))
    .map_err(|e| FilterError::from(format!("mcp: backend selector config: {e}")))?;
    let load_balancer = serde_yaml::to_value(serde_json::json!({
        "filter": "load_balancer",
        "clusters": clusters.iter().map(|name| serde_json::json!({ "name": name })).collect::<Vec<_>>(),
    }))
    .map_err(|e| FilterError::from(format!("mcp: backend load_balancer config: {e}")))?;
    Ok((filter_entry(selector)?, filter_entry(load_balancer)?))
}

/// Sub-request nesting depth, preferring IRR-owned state over the framework
/// header (which ingress strips from client requests).
fn callout_depth(iteration_state: Option<&IterationState>, headers: &http::HeaderMap) -> u8 {
    iteration_state.map_or_else(
        || {
            header_str(headers, DEPTH_HEADER)
                .and_then(|raw| raw.parse::<u8>().ok())
                .unwrap_or(0)
        },
        IterationState::depth,
    )
}

/// Deserialize one generated chain entry through [`FilterEntry`]'s own path.
fn filter_entry(value: serde_yaml::Value) -> Result<FilterEntry, FilterError> {
    serde_yaml::from_value(value).map_err(|e| FilterError::from(format!("mcp: generated chain entry: {e}")))
}

/// Rewrite a current-profile `tools/call` for its backend and swap the
/// gateway session for the backend session.
fn forward_current_tools_call(
    ctx: &mut HttpFilterContext<'_>,
    value: &mut serde_json::Value,
    body: &mut Option<Bytes>,
    tool: &CatalogTool,
    init: &BackendInit,
) {
    trace!(
        exposed = tool.exposed_name,
        original = tool.original_name,
        cluster = tool.cluster,
        server = tool.server_name,
        "routing current-profile tools/call"
    );
    ctx.cluster = Some(Arc::from(tool.cluster.as_str()));
    ctx.rewritten_path = Some(tool.backend_path.clone());

    if tool.original_name != tool.exposed_name {
        if let Some(p) = value.get_mut("params").and_then(|p| p.as_object_mut()) {
            p.insert("name".to_owned(), serde_json::Value::String(tool.original_name.clone()));
        }
        *body = Some(Bytes::from(value.to_string()));
    }

    match init.session_id.as_ref() {
        Some(session_id) => ctx.request_headers_to_set.push((MCP_SESSION_ID, session_id.clone())),
        None => ctx.request_headers_to_remove.push(MCP_SESSION_ID),
    }
    if let Ok(version) = http::HeaderValue::from_str(&init.protocol_version) {
        ctx.request_headers_to_set.push((MCP_PROTOCOL_VERSION, version));
    }
}

/// Record a forwarded session-bearing call so the header phase can pin its
/// endpoint and the response can invalidate an expired or unreachable session.
fn record_forwarded_call(
    ctx: &mut HttpFilterContext<'_>,
    key: BackendSessionKey,
    slot: Arc<SessionSlot>,
    init: BackendInit,
) {
    if let Some(session_id) = init.session_id {
        ctx.insert_filter_state(ForwardedCall {
            answered: false,
            key,
            slot,
            session_id,
            endpoint: init.endpoint,
        });
    }
}

/// The client's gateway `Mcp-Session-Id`, when present and well formed.
fn gateway_session_id<'a>(ctx: &'a HttpFilterContext<'_>) -> Option<&'a str> {
    let value = ctx.request.headers.get(&MCP_SESSION_ID)?;
    let bytes = value.as_bytes();
    if bytes.is_empty() || bytes.len() > MAX_GATEWAY_SESSION_ID_LEN || !bytes.iter().all(|b| (0x21..=0x7E).contains(b))
    {
        return None;
    }
    value.to_str().ok()
}

/// JSON-RPC error for a failed backend handshake. `data` carries only the
/// server name and a fixed reason label, never backend payloads.
fn backend_error_action(envelope: &JsonRpcEnvelope, server: &str, error: &McpBackendError) -> FilterAction {
    let mut data = serde_json::Map::new();
    data.insert("server".to_owned(), serde_json::Value::from(server));
    data.insert("reason".to_owned(), serde_json::Value::from(error.reason()));
    if let McpBackendError::Rejected { status } = error {
        data.insert("status".to_owned(), serde_json::Value::from(*status));
    }
    let response = serde_json::json!({
        "jsonrpc": "2.0",
        "error": {
            "code": ERR_INTERNAL,
            "message": "backend session initialization failed",
            "data": data,
        },
        "id": id_value(envelope),
    });

    FilterAction::Reject(
        Rejection::status(200)
            .with_header("content-type", "application/json")
            .with_body(Bytes::from(response.to_string())),
    )
}

// -----------------------------------------------------------------------------
// Stateless Helpers
// -----------------------------------------------------------------------------

/// Extracts and validates the tool name from `params.name`.
fn extract_tool_name(value: &serde_json::Value) -> Result<&str, &'static str> {
    let Some(params) = value.get("params") else {
        return Err("missing params");
    };
    if params.is_null() {
        return Err("params is null");
    }
    let Some(params_obj) = params.as_object() else {
        return Err("params must be an object");
    };
    let Some(name) = params_obj.get("name") else {
        return Err("missing params.name");
    };
    name.as_str().ok_or("params.name must be a string")
}

/// Encode a tool name as an HTTP header value, using the MCP `=?base64?`
/// sentinel for values that contain non-visible-ASCII bytes.
#[expect(clippy::expect_used, reason = "base64 sentinel is always valid ASCII")]
fn encode_mcp_name(name: &str) -> http::header::HeaderValue {
    if name.is_ascii()
        && !name.bytes().any(|b| b < 0x20 || b == 0x7F)
        && let Ok(hv) = http::header::HeaderValue::from_str(name)
    {
        return hv;
    }

    let encoded = base64::engine::general_purpose::STANDARD.encode(name);
    let sentinel = format!("{BASE64_SENTINEL_PREFIX}{encoded}{BASE64_SENTINEL_SUFFIX}");
    http::header::HeaderValue::from_str(&sentinel).expect("base64 sentinel is valid ASCII")
}

/// Validates the `Mcp-Name` header for stateless requests.
#[expect(clippy::too_many_lines, reason = "linear validation with early returns")]
fn validate_mcp_name_header(
    value: &serde_json::Value,
    envelope: &JsonRpcEnvelope,
    method_str: &str,
    header_name: Option<&str>,
) -> Option<FilterAction> {
    let body_name = extract_body_name(value, method_str);
    let requires_name = method_requires_mcp_name(method_str);

    if requires_name {
        let Some(header_name_value) = header_name else {
            return Some(header_mismatch_action(
                envelope,
                "Mcp-Name header is required for this method",
            ));
        };

        let decoded = decode_mcp_name(header_name_value);
        let decoded_ref = match &decoded {
            Ok(Some(d)) => d.as_str(),
            Ok(None) => header_name_value,
            Err(_) => {
                return Some(header_mismatch_action(
                    envelope,
                    "malformed base64 sentinel in Mcp-Name",
                ));
            },
        };

        let Some(body_name_value) = body_name else {
            return Some(header_mismatch_action(
                envelope,
                "Mcp-Name header present but params.name/uri is missing in body",
            ));
        };

        if decoded_ref != body_name_value {
            return Some(header_mismatch_action(
                envelope,
                "Mcp-Name header does not match body params.name/uri",
            ));
        }
    } else if header_name.is_some() {
        return Some(header_mismatch_action(
            envelope,
            "Mcp-Name header must not be present for this method",
        ));
    }

    None
}

/// Validates the `Mcp-Method` transport header.
fn validate_mcp_method(
    header_method: Option<&str>,
    envelope: &JsonRpcEnvelope,
    method_str: &str,
) -> Option<FilterAction> {
    let Some(header_method) = header_method else {
        return Some(header_mismatch_action(envelope, "missing Mcp-Method header"));
    };

    if header_method != method_str {
        return Some(header_mismatch_action(
            envelope,
            "Mcp-Method header does not match JSON-RPC method",
        ));
    }

    None
}

/// Validates required `params._meta` client metadata fields.
fn validate_params_meta(params_meta: Option<&serde_json::Value>) -> Option<&'static str> {
    if let Some(msg) = validate_client_info(params_meta) {
        return Some(msg);
    }

    validate_client_capabilities(params_meta)
}

/// Validates `params._meta["io.modelcontextprotocol/clientInfo"]` is an object
/// with string `name` and `version` fields.
fn validate_client_info(params_meta: Option<&serde_json::Value>) -> Option<&'static str> {
    let Some(info) = params_meta.and_then(|m| m.get("io.modelcontextprotocol/clientInfo")) else {
        return Some("missing params._meta[\"io.modelcontextprotocol/clientInfo\"]");
    };

    let Some(obj) = info.as_object() else {
        return Some("params._meta[\"io.modelcontextprotocol/clientInfo\"] must be an object");
    };

    if !obj.get("name").is_some_and(serde_json::Value::is_string) {
        return Some("params._meta clientInfo.name must be a string");
    }

    if !obj.get("version").is_some_and(serde_json::Value::is_string) {
        return Some("params._meta clientInfo.version must be a string");
    }

    None
}

/// Validates `params._meta["io.modelcontextprotocol/clientCapabilities"]` is an object.
fn validate_client_capabilities(params_meta: Option<&serde_json::Value>) -> Option<&'static str> {
    let Some(caps) = params_meta.and_then(|m| m.get("io.modelcontextprotocol/clientCapabilities")) else {
        return Some("missing params._meta[\"io.modelcontextprotocol/clientCapabilities\"]");
    };

    if !caps.is_object() {
        return Some("params._meta[\"io.modelcontextprotocol/clientCapabilities\"] must be an object");
    }

    None
}

/// Returns `true` for methods that require the `Mcp-Name` header.
fn method_requires_mcp_name(method_str: &str) -> bool {
    matches!(method_str, "tools/call" | "resources/read" | "prompts/get")
}

/// Extracts the body-derived name for `Mcp-Name` comparison.
fn extract_body_name<'a>(value: &'a serde_json::Value, method_str: &str) -> Option<&'a str> {
    let params = value.get("params")?;
    match method_str {
        "resources/read" => params.get("uri").and_then(|v| v.as_str()),
        _ => params.get("name").and_then(|v| v.as_str()),
    }
}

/// Failure modes for MCP `=?base64?...?=` sentinel decoding.
enum McpNameDecodeError {
    /// Invalid base64 payload.
    Base64,
    /// Decoded bytes are not valid UTF-8.
    Utf8,
}

/// Decode MCP `=?base64?...?=` sentinel values.
///
/// Returns `Ok(Some(decoded))` for valid sentinels, `Ok(None)` for
/// non-sentinel values, and `Err` for malformed sentinels.
fn decode_mcp_name(value: &str) -> Result<Option<String>, McpNameDecodeError> {
    let Some(inner) = value
        .strip_prefix(BASE64_SENTINEL_PREFIX)
        .and_then(|rest| rest.strip_suffix(BASE64_SENTINEL_SUFFIX))
    else {
        return Ok(None);
    };

    let bytes = base64::engine::general_purpose::STANDARD
        .decode(inner)
        .map_err(|_e| McpNameDecodeError::Base64)?;

    String::from_utf8(bytes)
        .map(Some)
        .map_err(|_e| McpNameDecodeError::Utf8)
}

/// Gets a header value as a `&str`.
fn header_str<'a>(headers: &'a http::HeaderMap, name: &str) -> Option<&'a str> {
    headers.get(name).and_then(|v| v.to_str().ok())
}

// -----------------------------------------------------------------------------
// Error Response Builders
// -----------------------------------------------------------------------------

/// Builds an invalid-params error response (HTTP 400, JSON-RPC -32602).
fn invalid_params_action(envelope: &JsonRpcEnvelope, detail: &str) -> FilterAction {
    let response = serde_json::json!({
        "jsonrpc": "2.0",
        "error": { "code": ERR_INVALID_PARAMS, "message": detail },
        "id": id_value(envelope),
    });

    FilterAction::Reject(
        Rejection::status(400)
            .with_header("content-type", "application/json")
            .with_body(Bytes::from(response.to_string())),
    )
}

/// Builds a header-mismatch error response (HTTP 400, JSON-RPC -32020).
fn header_mismatch_action(envelope: &JsonRpcEnvelope, message: &str) -> FilterAction {
    let response = serde_json::json!({
        "jsonrpc": "2.0",
        "error": { "code": ERR_HEADER_MISMATCH, "message": message },
        "id": id_value(envelope),
    });

    FilterAction::Reject(
        Rejection::status(400)
            .with_header("content-type", "application/json")
            .with_body(Bytes::from(response.to_string())),
    )
}

/// Builds an unsupported-version error response (HTTP 400, JSON-RPC -32022).
fn unsupported_version_action(envelope: &JsonRpcEnvelope, requested: &str, supported: &[String]) -> FilterAction {
    let response = serde_json::json!({
        "jsonrpc": "2.0",
        "error": {
            "code": ERR_UNSUPPORTED_VERSION,
            "data": { "requested": requested, "supported": supported },
            "message": "unsupported protocol version",
        },
        "id": id_value(envelope),
    });

    FilterAction::Reject(
        Rejection::status(400)
            .with_header("content-type", "application/json")
            .with_body(Bytes::from(response.to_string())),
    )
}

// -----------------------------------------------------------------------------
// Current-Profile Request Handlers
// -----------------------------------------------------------------------------

/// MCP notifications are one-way messages.
fn handle_notification(envelope: &JsonRpcEnvelope) -> FilterAction {
    if matches!(envelope.kind, JsonRpcKind::Notification) && matches!(envelope.id_kind, JsonRpcIdKind::Missing) {
        FilterAction::Reject(Rejection::status(202))
    } else {
        invalid_request_action(envelope)
    }
}

/// MCP request ids are narrower than JSON-RPC's parser accepts.
fn has_valid_request_id(envelope: &JsonRpcEnvelope) -> bool {
    matches!(envelope.id_kind, JsonRpcIdKind::String | JsonRpcIdKind::Integer)
}

/// Invalid request responses use id `null` when the client omitted or nulled
/// the request id, matching JSON-RPC error-envelope conventions.
fn invalid_request_action(envelope: &JsonRpcEnvelope) -> FilterAction {
    let id_json = match envelope.id_kind {
        JsonRpcIdKind::String | JsonRpcIdKind::Integer => format_id_json(envelope),
        JsonRpcIdKind::Number | JsonRpcIdKind::Null | JsonRpcIdKind::Missing => "null".to_owned(),
    };
    json_rpc_error_action_with_id(&id_json, -32600, "invalid request")
}

/// Returns 204 when a valid `Mcp-Session-Id` header is present, 400 otherwise.
fn handle_delete_current(ctx: &HttpFilterContext<'_>) -> FilterAction {
    if ctx
        .request
        .headers
        .get("mcp-session-id")
        .and_then(|v| v.to_str().ok())
        .is_some()
    {
        FilterAction::Reject(Rejection::status(204))
    } else {
        FilterAction::Reject(Rejection::status(400))
    }
}

/// Generates a new MCP session and returns MCP capabilities.
#[expect(clippy::unnecessary_wraps, reason = "signature matches sibling handle_* fns")]
fn handle_initialize(
    ctx: &mut HttpFilterContext<'_>,
    value: &serde_json::Value,
    envelope: &JsonRpcEnvelope,
    supported_versions: &[String],
    default_version: &str,
) -> Result<FilterAction, FilterError> {
    record_client_protocol_version(ctx, value);
    let response_version = negotiate_protocol_version(value, supported_versions, default_version);
    let session_id = format!("mcp-{}", ctx.id_generator.generate(ctx.time_source));

    debug!(session_id_len = session_id.len(), "MCP initialize");
    ctx.record_ai_metadata("mcp.session_id", session_id.clone());

    Ok(FilterAction::Reject(
        Rejection::status(200)
            .with_header("content-type", "application/json")
            .with_header("mcp-session-id", &session_id)
            .with_body(Bytes::from(
                initialize_response_body(envelope, response_version).to_string(),
            )),
    ))
}

/// Echoes the client's requested version when supported, otherwise falls back.
fn negotiate_protocol_version<'a>(
    value: &serde_json::Value,
    supported_versions: &'a [String],
    default_version: &'a str,
) -> &'a str {
    let requested = value
        .get("params")
        .and_then(|p| p.get("protocolVersion"))
        .and_then(|v| v.as_str());

    if let Some(req) = requested
        && let Some(matched) = supported_versions.iter().find(|v| v.as_str() == req)
    {
        return matched.as_str();
    }

    default_version
}

/// Persist the client's advertised MCP protocol version.
fn record_client_protocol_version(ctx: &mut HttpFilterContext<'_>, value: &serde_json::Value) {
    if let Some(version) = value
        .get("params")
        .and_then(|p| p.get("protocolVersion"))
        .and_then(|v| v.as_str())
        && !contains_control_chars(version)
    {
        ctx.record_ai_metadata("mcp.protocol_version", version.to_owned());
    }
}

/// Build the initialize response from the configured protocol version.
///
/// `serverInfo` intentionally omits `version` to preserve the established
/// current-profile response shape; stateless clients get version from
/// `server/discover` instead.
fn initialize_response_body(envelope: &JsonRpcEnvelope, protocol_version: &str) -> serde_json::Value {
    serde_json::json!({
        "jsonrpc": "2.0",
        "id": id_value(envelope),
        "result": {
            "protocolVersion": protocol_version,
            "capabilities": {
                "tools": {
                    "listChanged": false,
                },
            },
            "serverInfo": {
                "name": SERVER_NAME,
            },
        },
    })
}

/// Returns the aggregated static catalog.
fn handle_tools_list(catalog: &[CatalogTool], envelope: &JsonRpcEnvelope) -> Result<FilterAction, FilterError> {
    let tools_json = serialize_catalog(catalog)?;
    let id_json = format_id_json(envelope);
    let response_body = format!(r#"{{"jsonrpc":"2.0","id":{id_json},"result":{{"tools":{tools_json}}}}}"#,);

    trace!(tool_count = catalog.len(), "serving aggregated tools/list");

    Ok(FilterAction::Reject(
        Rejection::status(200)
            .with_header("content-type", "application/json")
            .with_body(Bytes::from(response_body)),
    ))
}

/// Serialize the tool catalog to JSON.
fn serialize_catalog(catalog: &[CatalogTool]) -> Result<String, FilterError> {
    let tools: Vec<serde_json::Value> = catalog.iter().map(catalog_tool_to_json).collect();
    serde_json::to_string(&tools).map_err(|e| FilterError::from(format!("mcp: failed to serialize tool catalog: {e}")))
}

/// Produces the MCP tool object shape expected by `tools/list` responses.
fn catalog_tool_to_json(tool: &CatalogTool) -> serde_json::Value {
    let mut obj = serde_json::Map::new();
    obj.insert("name".to_owned(), serde_json::Value::String(tool.exposed_name.clone()));
    if let Some(desc) = &tool.description {
        obj.insert("description".to_owned(), serde_json::Value::String(desc.clone()));
    }
    obj.insert("inputSchema".to_owned(), tool.input_schema.clone());
    if let Some(annotations) = &tool.annotations {
        obj.insert("annotations".to_owned(), annotations.clone());
    }
    serde_json::Value::Object(obj)
}

/// Returns `{"result":{}}` with the caller's JSON-RPC id preserved.
fn handle_ping(envelope: &JsonRpcEnvelope) -> FilterAction {
    let id_json = format_id_json(envelope);
    let response_body = format!(r#"{{"jsonrpc":"2.0","id":{id_json},"result":{{}}}}"#);

    FilterAction::Reject(
        Rejection::status(200)
            .with_header("content-type", "application/json")
            .with_body(Bytes::from(response_body)),
    )
}

// -----------------------------------------------------------------------------
// JSON-RPC Helpers
// -----------------------------------------------------------------------------

/// Format the JSON-RPC `id` field for response serialization.
fn format_id_json(envelope: &JsonRpcEnvelope) -> String {
    let id = envelope.id.as_deref().unwrap_or("null");
    match envelope.id_kind {
        JsonRpcIdKind::String => serde_json::to_string(id).unwrap_or_else(|_| "null".to_owned()),
        JsonRpcIdKind::Integer | JsonRpcIdKind::Number => id.to_owned(),
        JsonRpcIdKind::Null | JsonRpcIdKind::Missing => "null".to_owned(),
    }
}

/// Convert the parsed JSON-RPC id into a response JSON value.
fn id_value(envelope: &JsonRpcEnvelope) -> serde_json::Value {
    let Some(id) = envelope.id.as_deref() else {
        return serde_json::Value::Null;
    };
    match envelope.id_kind {
        JsonRpcIdKind::String => serde_json::Value::String(id.to_owned()),
        JsonRpcIdKind::Integer | JsonRpcIdKind::Number => serde_json::from_str(id).unwrap_or(serde_json::Value::Null),
        JsonRpcIdKind::Null | JsonRpcIdKind::Missing => serde_json::Value::Null,
    }
}

/// Build a JSON-RPC error response (HTTP 200 for current-profile compatibility).
fn json_rpc_error_action(envelope: &JsonRpcEnvelope, code: i32, message: &str) -> FilterAction {
    let id_json = format_id_json(envelope);
    json_rpc_error_action_with_id(&id_json, code, message)
}

/// Build a JSON-RPC error response with a specific HTTP status code.
fn json_rpc_error_action_with_status(
    envelope: &JsonRpcEnvelope,
    http_status: u16,
    code: i32,
    message: &str,
) -> FilterAction {
    let response = serde_json::json!({
        "jsonrpc": "2.0",
        "error": { "code": code, "message": message },
        "id": id_value(envelope),
    });

    FilterAction::Reject(
        Rejection::status(http_status)
            .with_header("content-type", "application/json")
            .with_body(Bytes::from(response.to_string())),
    )
}

/// Build a JSON-RPC error response with an explicit id.
fn json_rpc_error_action_with_id(id_json: &str, code: i32, message: &str) -> FilterAction {
    let message_json = serde_json::to_string(message).unwrap_or_else(|_| "\"internal error\"".to_owned());
    let body = Bytes::from(format!(
        r#"{{"jsonrpc":"2.0","error":{{"code":{code},"message":{message_json}}},"id":{id_json}}}"#,
    ));
    FilterAction::Reject(
        Rejection::status(200)
            .with_header("content-type", "application/json")
            .with_body(body),
    )
}

// -----------------------------------------------------------------------------
// Path Matching
// -----------------------------------------------------------------------------

/// Returns `true` when the request URI path matches the configured MCP path.
fn request_path_matches(uri: &http::Uri, public_path: &str) -> bool {
    uri.path() == public_path
}
