// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! MCP static catalog filter: static tool catalog, prefix management, and broker
//! behavior for `initialize`, `tools/list`, `ping`, and `notifications`.

pub(crate) mod config;

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

use async_trait::async_trait;
use bytes::Bytes;
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, Rejection,
    builtins::http::{
        payload_processing::json_rpc::{
            config::JsonRpcConfig,
            envelope::{JsonRpcEnvelope, JsonRpcIdKind, JsonRpcKind, parse_json_rpc_value},
        },
        value_safety::contains_control_chars,
    },
    parse_filter_config,
};
use tracing::{debug, trace};

use self::config::{CacheScope, CatalogTool, McpBrokerConfig, build_config};
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

/// MCP `=?base64?` sentinel prefix.
const BASE64_SENTINEL_PREFIX: &str = "=?base64?";

/// MCP `=?base64?` sentinel suffix.
const BASE64_SENTINEL_SUFFIX: &str = "?=";

// -----------------------------------------------------------------------------
// McpBrokerFilter
// -----------------------------------------------------------------------------

/// MCP static catalog filter that aggregates tool catalogs from multiple backend
/// MCP servers and handles `initialize`, `tools/list`, `tools/call`, `ping`,
/// and `notifications/initialized` directly as a static broker.
///
/// The broker serves configured catalog operations locally while backend tool
/// routing is not implemented. It deliberately returns `-32601` for
/// `tools/call` rather than forwarding a request whose target is unresolved.
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
    pub(crate) fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: McpBrokerConfig = parse_filter_config("mcp", config)?;
        let (validated, catalog) = build_config(cfg)?;

        let json_rpc_config = build_json_rpc_config(validated.max_body_bytes);

        Ok(Box::new(Self {
            cache_scope: validated.cache_scope,
            cache_ttl_ms: validated.cache_ttl_ms,
            catalog,
            default_version: validated.default_version.clone(),
            json_rpc_config,
            max_body_bytes: validated.max_body_bytes,
            protocol_profile: validated.protocol_profile,
            public_path: validated.path.clone(),
            supported_versions: validated.supported_versions.clone(),
        }))
    }

    // -------------------------------------------------------------------------
    // Method Dispatch
    // -------------------------------------------------------------------------

    /// Maps a JSON-RPC method to the MCP handler that owns it.
    fn dispatch_method(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        value: &serde_json::Value,
        envelope: &JsonRpcEnvelope,
        method_str: &str,
    ) -> Result<FilterAction, FilterError> {
        match self.protocol_profile {
            ProtocolProfile::Current => self.dispatch_current(ctx, value, envelope, method_str),
            ProtocolProfile::Stateless => Ok(self.dispatch_stateless(ctx, value, envelope, method_str)),
        }
    }

    /// Current-profile dispatch: preserves existing behavior.
    fn dispatch_current(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        value: &serde_json::Value,
        envelope: &JsonRpcEnvelope,
        method_str: &str,
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
            "tools/call" => json_rpc_error_action(envelope, -32601, "method not yet supported"),
            "ping" => handle_ping(envelope),
            _ => {
                debug!(method_len = method_str.len(), "unsupported MCP method");
                json_rpc_error_action(envelope, -32601, "method not found")
            },
        };

        Ok(action)
    }

    /// Stateless-profile dispatch: validates stateless headers, then dispatches.
    #[expect(
        clippy::needless_pass_by_ref_mut,
        reason = "signature matches dispatch_current for consistent dispatch_method call"
    )]
    fn dispatch_stateless(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        value: &serde_json::Value,
        envelope: &JsonRpcEnvelope,
        method_str: &str,
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

        if let Some(action) = self.validate_stateless_headers(value, &ctx.request.headers, envelope, method_str) {
            return action;
        }

        match method_str {
            "server/discover" => self.handle_server_discover(envelope),
            "tools/list" => self.handle_stateless_tools_list(envelope),
            "tools/call" => json_rpc_error_action_with_status(envelope, 404, -32601, "method not yet supported"),
            "ping" => handle_ping(envelope),
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
    fn handle_delete(&self, ctx: &HttpFilterContext<'_>) -> FilterAction {
        match self.protocol_profile {
            ProtocolProfile::Current => handle_delete_current(ctx),
            ProtocolProfile::Stateless => FilterAction::Reject(Rejection::status(405)),
        }
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
        BodyAccess::ReadWrite
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
            http::Method::POST => Ok(FilterAction::Continue),
            http::Method::DELETE => Ok(self.handle_delete(ctx)),
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

        let Ok(value) = serde_json::from_slice::<serde_json::Value>(chunk) else {
            return Ok(FilterAction::Reject(Rejection::status(400)));
        };

        let Ok(Some(envelope)) = parse_json_rpc_value(&value, &self.json_rpc_config) else {
            return Ok(FilterAction::Reject(Rejection::status(400)));
        };

        let Some(method_str) = &envelope.method else {
            return Ok(FilterAction::Reject(Rejection::status(400)));
        };

        if !contains_control_chars(method_str) {
            ctx.set_metadata("json_rpc.method", method_str.clone());
            ctx.set_metadata("mcp.method", method_str.clone());
        }

        self.dispatch_method(ctx, &value, &envelope, method_str)
    }
}

// -----------------------------------------------------------------------------
// Stateless Helpers
// -----------------------------------------------------------------------------

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
    use base64::Engine as _;

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
    ctx.set_metadata("mcp.session_id", session_id.clone());

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
        ctx.set_metadata("mcp.protocol_version", version.to_owned());
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

// -----------------------------------------------------------------------------
// Shared Parser Config
// -----------------------------------------------------------------------------

/// Build a [`JsonRpcConfig`] for the shared parser.
fn build_json_rpc_config(max_body_bytes: usize) -> JsonRpcConfig {
    use praxis_filter::builtins::http::payload_processing::{
        OnInvalidBehavior,
        json_rpc::config::{BatchPolicy, JsonRpcHeaders},
    };

    JsonRpcConfig {
        batch_policy: BatchPolicy::Reject,
        headers: JsonRpcHeaders {
            id: None,
            kind: None,
            method: None,
        },
        max_body_bytes,
        on_invalid: OnInvalidBehavior::Continue,
    }
}
