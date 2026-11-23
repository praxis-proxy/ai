// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Opt-in MCP 2026-07-28 POC broker, driven by a manually refreshed tool catalog.

use std::{borrow::Cow, collections::HashSet, fs};

use async_trait::async_trait;
use bytes::Bytes;
use http::header::{CONTENT_TYPE, HeaderName, HeaderValue};
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, McpAllowedCapabilities, Rejection,
    TerminalResponse, parse_filter_config,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tracing::trace;

/// The only wire protocol version this POC filter accepts.
const PROTOCOL_VERSION: &str = "2026-07-28";
/// Default maximum buffered JSON-RPC request size.
const DEFAULT_MAX_BODY_BYTES: usize = 1_048_576;
/// Default stateless response-cache lifetime.
const DEFAULT_CACHE_TTL_MS: u64 = 300_000;
/// Trusted internal header that selects the configured cluster.
const TARGET_HEADER: &str = "x-praxis-mcp-target";

/// Filter settings; tool definitions are loaded from the discovery catalog.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct McpPocConfig {
    /// Stage in the classify/PPE/respond chain.
    stage: McpPocStage,
    /// Path to the manually generated YAML/JSON catalog.
    catalog_path: String,
    /// Public MCP endpoint path.
    #[serde(default = "default_path")]
    path: String,
    /// Maximum request body size.
    #[serde(default = "default_max_body_bytes")]
    max_body_bytes: usize,
    /// Cache scope advertised in stateless responses.
    #[serde(default = "default_cache_scope")]
    cache_scope: String,
    /// Cache lifetime advertised in stateless responses, in milliseconds.
    #[serde(default = "default_cache_ttl_ms")]
    cache_ttl_ms: u64,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
enum McpPocStage {
    Classify,
    Respond,
}

/// Exact discovery snapshot consumed by the POC broker.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CatalogDocument {
    /// MCP version negotiated by the discovery utility.
    protocol_version: String,
    /// Upstream servers and their discovered tools.
    servers: Vec<ServerCatalog>,
}

/// One upstream's routing identity, path, prefix, and discovered tool schema.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ServerCatalog {
    /// Unique server label used by the route header.
    name: String,
    /// Praxis load-balancer cluster name.
    cluster: String,
    /// MCP path on this upstream.
    path: String,
    /// Prefix added to every exposed tool name.
    tool_prefix: String,
    /// Tool definitions returned by `tools/list`.
    tools: Vec<Value>,
}

/// One exposed tool and the backend identity needed to route its call.
struct CatalogTool {
    /// Stable server identity used by PPE capability grants.
    server_name: String,
    /// Praxis load-balancer cluster selected for this tool.
    cluster: String,
    /// MCP path on the selected upstream.
    backend_path: String,
    /// Public prefixed tool name.
    exposed_name: String,
    /// Original upstream tool name.
    original_name: String,
    /// Validated upstream tool name for the forwarded `Mcp-Name` header.
    mcp_name_header: HeaderValue,
    /// Upstream tool schema and metadata with its public name substituted.
    definition: Value,
}
/// JSON-RPC success envelope serialized without copying its identifier.
#[derive(Serialize)]
struct JsonRpcResult<'a, R> {
    /// JSON-RPC protocol marker.
    jsonrpc: &'static str,
    /// Request identifier borrowed from the incoming JSON-RPC body.
    id: &'a Value,
    /// Method-specific result object.
    result: R,
}

/// Stateless tools/list result metadata and borrowed tool definitions.
#[derive(Serialize)]
struct ToolsListResult<'a> {
    /// Cache scope advertised to clients.
    #[serde(rename = "cacheScope")]
    cache_scope: &'a str,
    /// Completeness marker required by the stateless profile.
    #[serde(rename = "resultType")]
    result_type: &'static str,
    /// Tool definitions serialized from the retained catalog.
    tools: CatalogDefinitions<'a>,
    /// Cache lifetime advertised to clients.
    #[serde(rename = "ttlMs")]
    ttl_ms: u64,
}

/// Serializes catalog entries as a JSON array without cloning tool values.
struct CatalogDefinitions<'a> {
    /// Catalog entries in discovery order.
    catalog: &'a [CatalogTool],
    /// PPE-produced request capabilities.
    capabilities: &'a McpAllowedCapabilities,
}

impl Serialize for CatalogDefinitions<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeSeq as _;

        let mut sequence = serializer.serialize_seq(None)?;
        for tool in self.catalog {
            if self.capabilities.allows(&tool.server_name, &tool.original_name) {
                sequence.serialize_element(&tool.definition)?;
            }
        }
        sequence.end()
    }
}
/// Parsed request facts shared by the classify and respond instances.
struct McpPocRequest {
    id: Value,
    method: String,
    tool_name: Option<String>,
    routed: bool,
}

/// Map untrusted MCP methods to a bounded span attribute.
fn bounded_mcp_method(method: &str) -> &'static str {
    match method {
        "server/discover" => "server/discover",
        "tools/list" => "tools/list",
        "tools/call" => "tools/call",
        _ => "other",
    }
}

/// One stage of the stateless MCP POC filter.
pub struct McpPocFilter {
    /// Classifies/routes requests or returns local catalog responses.
    stage: McpPocStage,
    /// Cache scope used for local stateless responses.
    cache_scope: String,
    /// Cache lifetime used in local stateless responses.
    cache_ttl_ms: u64,
    /// Validated, prefixed tools in stable upstream discovery order.
    catalog: Vec<CatalogTool>,
    /// Bound for buffered ingress JSON-RPC bodies.
    max_body_bytes: usize,
    /// Public MCP path served by this filter.
    path: String,
}

impl McpPocFilter {
    /// Build the POC filter from its local catalog path and validate all tools.
    ///
    /// # Errors
    ///
    /// Returns an error when filter settings are invalid or the catalog cannot be loaded.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let config: McpPocConfig = parse_filter_config("mcp_poc", config)?;
        validate_config(&config)?;
        let catalog = load_catalog(&config.catalog_path)?;

        Ok(Box::new(Self {
            stage: config.stage,
            cache_scope: config.cache_scope,
            cache_ttl_ms: config.cache_ttl_ms,
            catalog,
            max_body_bytes: config.max_body_bytes,
            path: config.path,
        }))
    }
}

/// Return the default MCP path.
fn default_path() -> String {
    "/mcp".to_owned()
}

/// Return the default buffered request size.
fn default_max_body_bytes() -> usize {
    DEFAULT_MAX_BODY_BYTES
}

/// Return the default private cache scope.
fn default_cache_scope() -> String {
    "private".to_owned()
}

/// Return the default stateless cache lifetime.
fn default_cache_ttl_ms() -> u64 {
    DEFAULT_CACHE_TTL_MS
}

/// Validate filter-level limits and response cache settings.
///
/// # Errors
///
/// Returns an error when the path, body limit, cache scope, or cache lifetime is invalid.
fn validate_config(config: &McpPocConfig) -> Result<(), FilterError> {
    praxis_filter::builtins::http::payload_processing::config_validation::validate_max_body_bytes(
        "mcp_poc",
        config.max_body_bytes,
    )?;
    validate_path(&config.path)?;
    if !matches!(config.cache_scope.as_str(), "public" | "private") {
        return Err("mcp_poc: cache_scope must be 'public' or 'private'".into());
    }
    if config.cache_ttl_ms == 0 {
        return Err("mcp_poc: cache_ttl_ms must be greater than zero".into());
    }
    Ok(())
}

/// Validate an absolute MCP path and reject traversal or URL components.
///
/// # Errors
///
/// Returns an error when the path is not an absolute path or contains traversal, query, or fragment syntax.
fn validate_path(path: &str) -> Result<(), FilterError> {
    if !path.starts_with('/')
        || path.starts_with("//")
        || path.contains(['?', '#'])
        || path.split('/').any(|part| part == "." || part == "..")
    {
        return Err("mcp_poc: path must be an absolute path without query, fragment, or traversal".into());
    }
    Ok(())
}

/// Read and validate a discovery catalog from its local config file.
///
/// # Errors
///
/// Returns an error when the file cannot be read or its catalog is invalid.
fn load_catalog(path: &str) -> Result<Vec<CatalogTool>, FilterError> {
    let contents = fs::read_to_string(path).map_err(|error| format!("mcp_poc: cannot read catalog: {error}"))?;
    let catalog: CatalogDocument =
        serde_yaml::from_str(&contents).map_err(|error| format!("mcp_poc: invalid catalog: {error}"))?;
    build_catalog(catalog)
}

/// Flatten the generated catalog into prefixed tool entries.
///
/// # Errors
///
/// Returns an error for an unsupported protocol version, invalid server entry, or duplicate exposed tool.
fn build_catalog(document: CatalogDocument) -> Result<Vec<CatalogTool>, FilterError> {
    if document.protocol_version != PROTOCOL_VERSION {
        return Err(format!("mcp_poc: catalog protocol_version must be '{PROTOCOL_VERSION}'").into());
    }
    if document.servers.is_empty() {
        return Err("mcp_poc: catalog must contain at least one server".into());
    }

    let mut server_names = HashSet::with_capacity(document.servers.len());
    let mut exposed_names = HashSet::new();
    let mut tools = Vec::new();
    for server in document.servers {
        validate_server(&server, &mut server_names)?;
        add_server_tools(server, &mut exposed_names, &mut tools)?;
    }
    Ok(tools)
}

/// Validate a server's stable identity and upstream path.
///
/// # Errors
///
/// Returns an error for unsafe or duplicate names, clusters, prefixes, or paths.
fn validate_server(server: &ServerCatalog, names: &mut HashSet<String>) -> Result<(), FilterError> {
    if !is_route_token(&server.name) || !names.insert(server.name.clone()) {
        return Err(format!("mcp_poc: invalid or duplicate server name '{}'", server.name).into());
    }
    if !is_route_token(&server.cluster) {
        return Err(format!("mcp_poc: server '{}' has an invalid cluster name", server.name).into());
    }
    validate_path(&server.path)?;
    if !is_route_token(&server.tool_prefix) {
        return Err(format!("mcp_poc: server '{}' has an invalid tool_prefix", server.name).into());
    }
    Ok(())
}

/// Add one server's tools to the flattened catalog.
///
/// # Errors
///
/// Returns an error when any tool definition is malformed or duplicates an exposed name.
fn add_server_tools(
    mut server: ServerCatalog,
    exposed_names: &mut HashSet<String>,
    catalog: &mut Vec<CatalogTool>,
) -> Result<(), FilterError> {
    let definitions = std::mem::take(&mut server.tools);
    for definition in definitions {
        catalog.push(build_catalog_tool(definition, &server, exposed_names)?);
    }
    Ok(())
}

/// Validate and prefix one upstream tool definition.
///
/// # Errors
///
/// Returns an error when the name or input schema is not usable by this POC.
fn build_catalog_tool(
    mut definition: Value,
    server: &ServerCatalog,
    exposed_names: &mut HashSet<String>,
) -> Result<CatalogTool, FilterError> {
    let original_name = validate_tool_definition(&definition, &server.name)?;
    let exposed_name = format!("{}{original_name}", server.tool_prefix);
    if exposed_name.len() > 255 || !exposed_names.insert(exposed_name.clone()) {
        return Err(format!("mcp_poc: invalid or duplicate exposed tool name '{exposed_name}'").into());
    }
    let mcp_name_header = HeaderValue::from_str(&original_name)
        .map_err(|error| format!("mcp_poc: tool '{original_name}' cannot be used as an MCP name header: {error}"))?;
    if let Some(object) = definition.as_object_mut() {
        object.insert("name".to_owned(), Value::String(exposed_name.clone()));
    }
    Ok(CatalogTool {
        server_name: server.name.clone(),
        cluster: server.cluster.clone(),
        backend_path: server.path.clone(),
        exposed_name,
        original_name,
        mcp_name_header,
        definition,
    })
}

/// Validate the name and schema fields required by a proxy-routed tool.
///
/// # Errors
///
/// Returns an error for a non-object definition, unsafe name, or missing object input schema.
fn validate_tool_definition(definition: &Value, server_name: &str) -> Result<String, FilterError> {
    let Some(object) = definition.as_object() else {
        return Err(format!("mcp_poc: server '{server_name}' has a non-object tool").into());
    };
    let Some(name) = object.get("name").and_then(Value::as_str) else {
        return Err(format!("mcp_poc: server '{server_name}' has a tool without a name").into());
    };
    if !is_route_token(name) {
        return Err(format!("mcp_poc: tool name '{name}' is not a POC-safe header value").into());
    }
    if !object.get("inputSchema").is_some_and(Value::is_object) {
        return Err(format!("mcp_poc: tool '{name}' has no object inputSchema").into());
    }
    Ok(name.to_owned())
}

/// Check that a routing label is safe as an internal HTTP header value.
fn is_route_token(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 255
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
}


/// Build a JSON-RPC success response for the response stage.
fn json_rpc_response(id: &Value, result: &Value) -> FilterAction {
    let body = json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": result,
    });
    terminal_json_response(200, &body)
}

/// Build a JSON-RPC error response for parser validation failures.
fn json_rpc_error(id: Option<&Value>, status: u16, code: i32, message: &str) -> FilterAction {
    local_json_response(status, &json_rpc_error_body(id, code, message))
}

/// Build a JSON-RPC error response after PPE has run.
fn terminal_json_rpc_error(id: Option<&Value>, status: u16, code: i32, message: &str) -> FilterAction {
    terminal_json_response(status, &json_rpc_error_body(id, code, message))
}

fn json_rpc_error_body(id: Option<&Value>, code: i32, message: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "error": { "code": code, "message": message },
        "id": id.cloned().unwrap_or(Value::Null),
    })
}

/// Serialize a parser rejection and set its content type.
fn local_json_response(status: u16, body: &Value) -> FilterAction {
    local_json_response_bytes(status, Bytes::from(body.to_string()))
}

fn local_json_response_bytes(status: u16, body: Bytes) -> FilterAction {
    FilterAction::Reject(
        Rejection::status(status)
            .with_header("content-type", "application/json")
            .with_body(body),
    )
}

fn terminal_json_response(status: u16, body: &Value) -> FilterAction {
    terminal_json_response_bytes(status, Bytes::from(body.to_string()))
}

fn terminal_json_response_bytes(status: u16, body: Bytes) -> FilterAction {
    let mut headers = http::HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    FilterAction::TerminalResponse(Box::new(
        TerminalResponse::new(status).with_headers(headers).with_body(body),
    ))
}

/// Validate method, version, client metadata, and name headers for stateless MCP.
///
/// # Errors
///
/// Returns a JSON-RPC rejection when a required header or metadata field is missing or inconsistent.
fn validate_stateless_headers(
    value: &Value,
    headers: &http::HeaderMap,
    id: &Value,
    method: &str,
) -> Result<(), FilterAction> {
    validate_method_header(headers, id, method)?;
    validate_protocol_version(value, headers, id)?;
    validate_client_metadata(value, id)?;
    validate_mcp_name(value, headers, id, method)
}

/// Require the per-request MCP method header to match the JSON-RPC body.
///
/// # Errors
///
/// Returns a JSON-RPC rejection when the method header is absent or mismatched.
fn validate_method_header(headers: &http::HeaderMap, id: &Value, method: &str) -> Result<(), FilterAction> {
    let header = headers.get("mcp-method").and_then(|value| value.to_str().ok());
    if header != Some(method) {
        return Err(json_rpc_error(
            Some(id),
            400,
            -32001,
            "Mcp-Method does not match JSON-RPC method",
        ));
    }
    Ok(())
}

/// Require the target MCP version in both the HTTP header and request metadata.
///
/// # Errors
///
/// Returns a JSON-RPC rejection when the header or metadata version differs.
fn validate_protocol_version(value: &Value, headers: &http::HeaderMap, id: &Value) -> Result<(), FilterAction> {
    let header = headers
        .get("mcp-protocol-version")
        .and_then(|value| value.to_str().ok());
    let body = value
        .get("params")
        .and_then(|params| params.get("_meta"))
        .and_then(|metadata| metadata.get("io.modelcontextprotocol/protocolVersion"))
        .and_then(Value::as_str);
    if header != Some(PROTOCOL_VERSION) || body != Some(PROTOCOL_VERSION) {
        return Err(json_rpc_error(
            Some(id),
            400,
            -32002,
            "Mcp-Protocol-Version and params._meta protocolVersion must be 2026-07-28",
        ));
    }
    Ok(())
}

/// Validate the client metadata fields required by the stateless profile.
///
/// # Errors
///
/// Returns a JSON-RPC rejection when the `_meta` fields are absent or malformed.
fn validate_client_metadata(value: &Value, id: &Value) -> Result<(), FilterAction> {
    let metadata = value.get("params").and_then(|params| params.get("_meta"));
    validate_client_info(metadata, id)?;
    validate_client_capabilities(metadata, id)
}

/// Validate the client name and version carried in `_meta`.
///
/// # Errors
///
/// Returns a JSON-RPC rejection when the clientInfo object is malformed.
fn validate_client_info(metadata: Option<&Value>, id: &Value) -> Result<(), FilterAction> {
    let Some(client_info) = metadata
        .and_then(|metadata| metadata.get("io.modelcontextprotocol/clientInfo"))
        .and_then(Value::as_object)
    else {
        return Err(json_rpc_error(Some(id), 400, -32001, "clientInfo must be an object"));
    };
    if client_info.get("name").is_some_and(Value::is_string) && client_info.get("version").is_some_and(Value::is_string)
    {
        return Ok(());
    }
    Err(json_rpc_error(
        Some(id),
        400,
        -32001,
        "clientInfo requires string name and version",
    ))
}

/// Validate the capabilities object carried in `_meta`.
///
/// # Errors
///
/// Returns a JSON-RPC rejection when clientCapabilities is missing or malformed.
fn validate_client_capabilities(metadata: Option<&Value>, id: &Value) -> Result<(), FilterAction> {
    if metadata
        .and_then(|metadata| metadata.get("io.modelcontextprotocol/clientCapabilities"))
        .is_some_and(Value::is_object)
    {
        return Ok(());
    }
    Err(json_rpc_error(
        Some(id),
        400,
        -32001,
        "clientCapabilities must be an object",
    ))
}

/// Require the named-operation header to match `tools/call` and reject it elsewhere.
///
/// # Errors
///
/// Returns a JSON-RPC rejection when the header and body name differ.
fn validate_mcp_name(value: &Value, headers: &http::HeaderMap, id: &Value, method: &str) -> Result<(), FilterAction> {
    let body_name = (method == "tools/call")
        .then(|| {
            value
                .get("params")
                .and_then(|params| params.get("name"))
                .and_then(Value::as_str)
        })
        .flatten();
    let header_name = headers.get("mcp-name").and_then(|value| value.to_str().ok());
    if body_name != header_name {
        return Err(json_rpc_error(
            Some(id),
            400,
            -32001,
            "Mcp-Name does not match tools/call name",
        ));
    }
    Ok(())
}

#[expect(
    clippy::multiple_inherent_impl,
    reason = "Keep construction and request dispatch responsibilities grouped separately"
)]
impl McpPocFilter {
    /// Build the server/discover response from the shared catalog settings.
    fn server_discover(&self, id: &Value) -> FilterAction {
        let result = json!({
            "cacheScope": self.cache_scope,
            "capabilities": { "tools": { "listChanged": false } },
            "resultType": "complete",
            "serverInfo": {
                "name": "praxis-mcp-poc",
                "version": env!("CARGO_PKG_VERSION"),
            },
            "supportedVersions": [PROTOCOL_VERSION],
            "ttlMs": self.cache_ttl_ms,
        });
        json_rpc_response(id, &result)
    }

    /// Build a complete tools/list response from the caller's allowed catalog entries.
    fn tools_list(&self, id: &Value, capabilities: &McpAllowedCapabilities) -> FilterAction {
        let tools_before = u64::try_from(self.catalog.len()).unwrap_or(u64::MAX);
        let tools_after = u64::try_from(
            self.catalog
                .iter()
                .filter(|tool| capabilities.allows(&tool.server_name, &tool.original_name))
                .count(),
        )
        .unwrap_or(u64::MAX);
        tracing::info!(
            target: "mcp_poc.authorization",
            tools_before = tools_before,
            tools_after = tools_after,
            operation = "tools/list",
            "filtered MCP tools/list by authorization",
        );
        let response = JsonRpcResult {
            jsonrpc: "2.0",
            id,
            result: ToolsListResult {
                cache_scope: "private",
                result_type: "complete",
                tools: CatalogDefinitions {
                    catalog: &self.catalog,
                    capabilities,
                },
                ttl_ms: self.cache_ttl_ms,
            },
        };
        match serde_json::to_vec(&response) {
            Ok(body) => terminal_json_response_bytes(200, Bytes::from(body)),
            Err(_) => terminal_json_rpc_error(Some(id), 500, -32603, "could not serialize tools/list response"),
        }
    }

    /// Resolve a known prefixed call before PPE and stage its upstream route.
    fn classify_tool_call(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        value: &mut Value,
        id: &Value,
        body: &mut Option<Bytes>,
        name: Option<&str>,
        span: &tracing::Span,
    ) -> FilterAction {
        let Some(name) = name else {
            return FilterAction::Continue;
        };
        let Some(tool) = self.catalog.iter().find(|tool| tool.exposed_name == name) else {
            return FilterAction::Continue;
        };
        route_tool_call(ctx, value, id, body, tool, span)
    }

    /// Find a routed catalog entry only when the PPE capability set allows it.
    fn authorized_tool<'a>(
        &'a self,
        request: &McpPocRequest,
        capabilities: Option<&McpAllowedCapabilities>,
    ) -> Option<&'a CatalogTool> {
        let capabilities = capabilities?;
        if !request.routed {
            return None;
        }
        let name = request.tool_name.as_deref()?;
        self.catalog
            .iter()
            .find(|tool| tool.exposed_name == name && capabilities.allows(&tool.server_name, &tool.original_name))
    }

    /// Produce local responses after PPE, or let an authorized call reach routing.
    fn respond(&self, request: McpPocRequest, capabilities: Option<&McpAllowedCapabilities>) -> FilterAction {
        match request.method.as_str() {
            "server/discover" => self.server_discover(&request.id),
            "tools/list" => match capabilities {
                Some(capabilities) => self.tools_list(&request.id, capabilities),
                None => {
                    terminal_json_rpc_error(Some(&request.id), 200, -32003, "MCP tool authorization is unavailable")
                },
            },
            "tools/call" if request.routed && self.authorized_tool(&request, capabilities).is_some() => {
                FilterAction::Continue
            },
            "tools/call" if request.routed => {
                terminal_json_rpc_error(Some(&request.id), 200, -32003, "not authorized to call this tool")
            },
            "tools/call" if request.tool_name.is_none() => {
                terminal_json_rpc_error(Some(&request.id), 200, -32602, "tools/call requires params.name")
            },
            "tools/call" => terminal_json_rpc_error(Some(&request.id), 200, -32602, "unknown tool"),
            _ => terminal_json_rpc_error(Some(&request.id), 404, -32601, "method not found"),
        }
    }
}
/// Rewrite the upstream name and stage the route selected by a validated catalog entry.
fn route_tool_call(
    ctx: &mut HttpFilterContext<'_>,
    value: &mut Value,
    id: &Value,
    body: &mut Option<Bytes>,
    tool: &CatalogTool,
    span: &tracing::Span,
) -> FilterAction {
    let Some(params) = value.get_mut("params").and_then(Value::as_object_mut) else {
        return json_rpc_error(Some(id), 200, -32602, "tools/call params must be an object");
    };
    span.record("mcp.tool", tool.exposed_name.as_str());
    span.record("mcp.backend.tool", tool.original_name.as_str());
    params.insert("name".to_owned(), Value::String(tool.original_name.clone()));
    *body = Some(Bytes::from(value.to_string()));
    ctx.rewritten_path = Some(tool.backend_path.clone());
    ctx.extra_request_headers
        .push((Cow::Borrowed(TARGET_HEADER), tool.cluster.clone()));
    ctx.request_headers_to_set
        .push((HeaderName::from_static("mcp-name"), tool.mcp_name_header.clone()));
    trace!(
        exposed_tool = tool.exposed_name,
        backend_tool = tool.original_name,
        cluster = tool.cluster,
        "routing stateless MCP tools/call"
    );
    FilterAction::Release
}
#[expect(
    clippy::too_many_lines,
    reason = "Request admission and buffered dispatch are one filter implementation"
)]
#[async_trait]
impl HttpFilter for McpPocFilter {
    fn name(&self) -> &'static str {
        "mcp_poc"
    }

    fn request_body_access(&self) -> BodyAccess {
        match self.stage {
            McpPocStage::Classify => BodyAccess::ReadWrite,
            McpPocStage::Respond => BodyAccess::None,
        }
    }

    fn request_body_mode(&self) -> BodyMode {
        match self.stage {
            McpPocStage::Classify => BodyMode::StreamBuffer {
                max_bytes: Some(self.max_body_bytes),
            },
            McpPocStage::Respond => BodyMode::Stream,
        }
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        if self.stage == McpPocStage::Classify {
            if ctx.request.uri.path() != self.path {
                return Ok(FilterAction::Reject(Rejection::status(404)));
            }
            if ctx.request.method != http::Method::POST {
                return Ok(FilterAction::Reject(Rejection::status(405)));
            }
            if ctx.request.headers.contains_key("mcp-session-id") {
                return Ok(FilterAction::Reject(Rejection::status(400)));
            }
            return Ok(FilterAction::Continue);
        }
        let Some(request) = ctx.extensions.remove::<McpPocRequest>() else {
            return Ok(FilterAction::Reject(Rejection::status(500)));
        };
        let capabilities = ctx.extensions.get::<McpAllowedCapabilities>();
        let outcome = if request.method == "tools/call" {
            if self.authorized_tool(&request, capabilities).is_some() {
                "forward"
            } else if request.routed {
                "deny"
            } else {
                "local_response"
            }
        } else {
            "local_response"
        };
        let span = tracing::info_span!(
            "mcp_poc.respond",
            "mcp.method" = bounded_mcp_method(&request.method),
            "mcp.outcome" = outcome,
        );
        let _entered = span.enter();
        Ok(self.respond(request, capabilities))
    }

    async fn on_request_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if self.stage == McpPocStage::Respond || !end_of_stream {
            return Ok(FilterAction::Continue);
        }
        let span = tracing::info_span!(
            "mcp_poc.classify",
            "mcp.method" = tracing::field::Empty,
            "mcp.outcome" = tracing::field::Empty,
            "mcp.tool" = tracing::field::Empty,
            "mcp.backend.tool" = tracing::field::Empty,
        );
        let _entered = span.enter();
        let Some(bytes) = body.as_ref() else {
            return Ok(FilterAction::Reject(Rejection::status(400)));
        };
        let Ok(mut value) = serde_json::from_slice::<Value>(bytes) else {
            return Ok(FilterAction::Reject(Rejection::status(400)));
        };
        if value.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
            return Ok(json_rpc_error(None, 400, -32600, "invalid JSON-RPC request"));
        }
        let Some(id) = value.get("id").filter(|id| id.is_string() || id.is_number()).cloned() else {
            return Ok(json_rpc_error(
                None,
                400,
                -32600,
                "stateless MCP requests require an id",
            ));
        };
        let Some(method) = value.get("method").and_then(Value::as_str).map(str::to_owned) else {
            return Ok(json_rpc_error(Some(&id), 400, -32600, "missing JSON-RPC method"));
        };
        span.record("mcp.method", bounded_mcp_method(&method));

        if let Err(action) = validate_stateless_headers(&value, &ctx.request.headers, &id, &method) {
            return Ok(action);
        }

        let tool_name = (method == "tools/call")
            .then(|| {
                value
                    .get("params")
                    .and_then(|params| params.get("name"))
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            })
            .flatten();
        ctx.set_metadata("mcp.method", &method);
        if let Some(name) = &tool_name {
            ctx.set_metadata("mcp.name", name);
        }

        let action = if method == "tools/call" {
            self.classify_tool_call(ctx, &mut value, &id, body, tool_name.as_deref(), &span)
        } else {
            FilterAction::Continue
        };
        let routed = matches!(&action, FilterAction::Release);
        span.record("mcp.outcome", if routed { "route" } else { "respond" });

        ctx.extensions.insert(McpPocRequest {
            id,
            method,
            tool_name,
            routed,
        });
        Ok(action)
    }
}
#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests"
)]
mod tests {
    use std::fs;
    #[cfg(not(feature = "token-rate-limit-filter"))]
    use std::{cell::RefCell, collections::BTreeMap, rc::Rc, sync::LazyLock};

    use bytes::Bytes;
    use praxis_filter::{FilterAction, McpAllowedCapabilities};
    #[cfg(not(feature = "token-rate-limit-filter"))]
    use tracing::Instrument as _;
    #[cfg(not(feature = "token-rate-limit-filter"))]
    use tracing_subscriber::layer::SubscriberExt as _;

    use super::McpPocFilter;

    const CATALOG: &str = concat!(
        "protocol_version: 2026-07-28\n",
        "servers:\n",
        "  - name: python\n",
        "    cluster: python\n",
        "    path: /mcp\n",
        "    tool_prefix: py_\n",
        "    tools:\n",
        "      - name: echo\n",
        "        description: Echo from FastMCP\n",
        "        inputSchema:\n",
        "          type: object\n",
        "          properties:\n",
        "            message: {type: string}\n",
        "  - name: everything\n",
        "    cluster: everything\n",
        "    path: /mcp\n",
        "    tool_prefix: everything_\n",
        "    tools:\n",
        "      - name: get-sum\n",
        "        description: Sum from the official Everything server\n",
        "        inputSchema:\n",
        "          type: object\n",
        "          properties:\n",
        "            a: {type: number}\n",
        "            b: {type: number}\n",
    );

    fn filter_for_stage(
        directory: &tempfile::TempDir,
        stage: &str,
        cache_scope: &str,
    ) -> Box<dyn praxis_filter::HttpFilter> {
        let catalog_path = directory.path().join("catalog.yaml");
        let config = serde_yaml::from_str(&format!(
            "catalog_path: {:?}\npath: /mcp\nstage: {stage}\ncache_scope: {cache_scope}\n",
            catalog_path.to_string_lossy()
        ))
        .unwrap();
        McpPocFilter::from_config(&config).unwrap()
    }

    fn filter_pair() -> (
        tempfile::TempDir,
        Box<dyn praxis_filter::HttpFilter>,
        Box<dyn praxis_filter::HttpFilter>,
    ) {
        filter_pair_with_cache_scope("private")
    }

    fn filter_pair_with_cache_scope(
        cache_scope: &str,
    ) -> (
        tempfile::TempDir,
        Box<dyn praxis_filter::HttpFilter>,
        Box<dyn praxis_filter::HttpFilter>,
    ) {
        let directory = tempfile::tempdir().unwrap();
        let catalog_path = directory.path().join("catalog.yaml");
        fs::write(&catalog_path, CATALOG).unwrap();
        let classify = filter_for_stage(&directory, "classify", cache_scope);
        let respond = filter_for_stage(&directory, "respond", cache_scope);
        (directory, classify, respond)
    }

    fn all_capabilities() -> McpAllowedCapabilities {
        McpAllowedCapabilities::from_grants([("*".to_owned(), "*".to_owned())])
    }

    fn python_echo_capabilities() -> McpAllowedCapabilities {
        McpAllowedCapabilities::from_grants([("python".to_owned(), "echo".to_owned())])
    }

    fn request(method: &str, tool_name: Option<&str>) -> (praxis_filter::Request, Bytes) {
        let mut params = serde_json::json!({
            "_meta": {
                "io.modelcontextprotocol/protocolVersion": "2026-07-28",
                "io.modelcontextprotocol/clientInfo": {"name": "mcp-poc-tests", "version": "1.0"},
                "io.modelcontextprotocol/clientCapabilities": {}
            }
        });
        if let Some(name) = tool_name {
            params["name"] = serde_json::Value::String(name.to_owned());
            params["arguments"] = serde_json::json!({"message": "route-check"});
        }
        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 7,
            "method": method,
            "params": params,
        });
        let mut request = crate::test_utils::make_request(http::Method::POST, "/mcp");
        request
            .headers
            .insert("mcp-protocol-version", "2026-07-28".parse().unwrap());
        request.headers.insert("mcp-method", method.parse().unwrap());
        if let Some(name) = tool_name {
            request.headers.insert("mcp-name", name.parse().unwrap());
        }
        (request, Bytes::from(body.to_string()))
    }

    #[cfg(not(feature = "token-rate-limit-filter"))]
    #[derive(Debug, Clone)]
    struct CapturedSpan {
        id: u64,
        name: &'static str,
        fields: BTreeMap<&'static str, String>,
    }

    #[cfg(not(feature = "token-rate-limit-filter"))]
    #[derive(Debug, Clone, Default)]
    struct SpanCapture(Rc<RefCell<Vec<CapturedSpan>>>);

    #[cfg(not(feature = "token-rate-limit-filter"))]
    thread_local! {
        static ACTIVE_SPAN_CAPTURE: RefCell<Option<SpanCapture>> = const { RefCell::new(None) };
    }

    #[cfg(not(feature = "token-rate-limit-filter"))]
    #[derive(Debug, Default)]
    struct GlobalSpanCapture;

    #[cfg(not(feature = "token-rate-limit-filter"))]
    #[derive(Default)]
    struct SpanFields(BTreeMap<&'static str, String>);

    #[cfg(not(feature = "token-rate-limit-filter"))]
    impl tracing::field::Visit for SpanFields {
        fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
            self.0.insert(field.name(), format!("{value:?}"));
        }

        fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
            self.0.insert(field.name(), value.to_owned());
        }
    }

    #[cfg(not(feature = "token-rate-limit-filter"))]
    impl SpanCapture {
        fn install(&self) -> SpanCaptureGuard {
            static INSTALLED: LazyLock<()> = LazyLock::new(|| {
                tracing::subscriber::set_global_default(tracing_subscriber::registry().with(GlobalSpanCapture))
                    .expect("MCP POC tracing capture must own the test subscriber");
            });
            let _installed = LazyLock::force(&INSTALLED);
            ACTIVE_SPAN_CAPTURE.with(|active| SpanCaptureGuard {
                previous: active.replace(Some(self.clone())),
            })
        }

        fn spans(&self) -> Vec<CapturedSpan> {
            self.0.borrow().clone()
        }
    }

    #[cfg(not(feature = "token-rate-limit-filter"))]
    struct SpanCaptureGuard {
        previous: Option<SpanCapture>,
    }

    #[cfg(not(feature = "token-rate-limit-filter"))]
    impl Drop for SpanCaptureGuard {
        fn drop(&mut self) {
            ACTIVE_SPAN_CAPTURE.with(|active| {
                drop(active.replace(self.previous.take()));
            });
        }
    }

    #[cfg(not(feature = "token-rate-limit-filter"))]
    impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for GlobalSpanCapture {
        fn register_callsite(&self, metadata: &'static tracing::Metadata<'static>) -> tracing::subscriber::Interest {
            if matches!(
                metadata.name(),
                "mcp_poc_test_request" | "mcp_poc.classify" | "mcp_poc.respond"
            ) {
                tracing::subscriber::Interest::always()
            } else {
                tracing::subscriber::Interest::never()
            }
        }

        fn on_new_span(
            &self,
            attrs: &tracing::span::Attributes<'_>,
            id: &tracing::span::Id,
            _ctx: tracing_subscriber::layer::Context<'_, S>,
        ) {
            ACTIVE_SPAN_CAPTURE.with(|active| {
                let active = active.borrow();
                let Some(capture) = active.as_ref() else {
                    return;
                };
                let mut fields = SpanFields::default();
                attrs.record(&mut fields);
                capture.0.borrow_mut().push(CapturedSpan {
                    id: id.into_u64(),
                    name: attrs.metadata().name(),
                    fields: fields.0,
                });
            });
        }

        fn on_record(
            &self,
            id: &tracing::span::Id,
            values: &tracing::span::Record<'_>,
            _ctx: tracing_subscriber::layer::Context<'_, S>,
        ) {
            ACTIVE_SPAN_CAPTURE.with(|active| {
                let active = active.borrow();
                let Some(capture) = active.as_ref() else {
                    return;
                };
                let mut fields = SpanFields::default();
                values.record(&mut fields);
                if let Some(span) = capture.0.borrow_mut().iter_mut().find(|span| span.id == id.into_u64()) {
                    span.fields.extend(fields.0);
                }
            });
        }
    }

    #[cfg(not(feature = "token-rate-limit-filter"))]
    #[tokio::test]
    async fn classify_and_respond_spans_record_bounded_catalog_attributes() {
        let (_directory, classify, respond) = filter_pair();
        let capture = SpanCapture::default();
        let _capture_guard = capture.install();
        let request_span = tracing::info_span!("mcp_poc_test_request");

        for (method, tool_name, bounded_method, classify_outcome, respond_outcome, public_tool, backend_tool) in [
            (
                "server/discover",
                None,
                "server/discover",
                "respond",
                "local_response",
                None,
                None,
            ),
            (
                "tools/list",
                None,
                "tools/list",
                "respond",
                "local_response",
                None,
                None,
            ),
            (
                "tools/call",
                Some("py_echo"),
                "tools/call",
                "route",
                "forward",
                Some("py_echo"),
                Some("echo"),
            ),
            (
                "tools/call",
                Some("everything_get-sum"),
                "tools/call",
                "route",
                "forward",
                Some("everything_get-sum"),
                Some("get-sum"),
            ),
            (
                "tools/call",
                Some("untrusted-tool"),
                "tools/call",
                "respond",
                "local_response",
                None,
                None,
            ),
            (
                "request-method-secret",
                None,
                "other",
                "respond",
                "local_response",
                None,
                None,
            ),
        ] {
            let (request, body) = request(method, tool_name);
            let mut body = Some(body);
            let mut context = crate::test_utils::make_filter_context(&request);
            context.current_filter_id = Some(0);
            context.extensions.insert(all_capabilities());
            let start = capture.spans().len();

            let classify_action = classify
                .on_request_body(&mut context, &mut body, true)
                .instrument(request_span.clone())
                .await
                .unwrap();
            if public_tool.is_some() {
                assert!(matches!(classify_action, FilterAction::Release));
            } else {
                assert!(matches!(classify_action, FilterAction::Continue));
            }
            assert!(matches!(
                classify
                    .on_request(&mut context)
                    .instrument(request_span.clone())
                    .await
                    .unwrap(),
                FilterAction::Continue
            ));
            context.current_filter_id = Some(1);
            let respond_action = respond
                .on_request(&mut context)
                .instrument(request_span.clone())
                .await
                .unwrap();
            if respond_outcome == "forward" {
                assert!(matches!(respond_action, FilterAction::Continue));
            } else {
                assert!(matches!(respond_action, FilterAction::TerminalResponse(_)));
            }

            let spans = capture.spans();
            let request_spans = &spans[start..];
            assert_eq!(
                request_spans.len(),
                2,
                "each request should create classify and respond spans"
            );
            let classify_span = request_spans
                .iter()
                .find(|span| span.name == "mcp_poc.classify")
                .expect("missing classify span");
            assert_eq!(
                classify_span.fields.get("mcp.method").map(String::as_str),
                Some(bounded_method),
                "classify span should record only the bounded MCP method"
            );
            assert_eq!(
                classify_span.fields.get("mcp.outcome").map(String::as_str),
                Some(classify_outcome),
                "classify span should record a bounded routing decision"
            );
            assert_eq!(
                classify_span.fields.get("mcp.tool").map(String::as_str),
                public_tool,
                "classify span should record only a catalog-validated public tool name"
            );
            assert_eq!(
                classify_span.fields.get("mcp.backend.tool").map(String::as_str),
                backend_tool,
                "classify span should record only a catalog-validated backend tool name"
            );
            assert_eq!(
                classify_span.fields.len(),
                2 + usize::from(public_tool.is_some()) + usize::from(backend_tool.is_some()),
                "classify span must not capture request data"
            );
            let respond_span = request_spans
                .iter()
                .find(|span| span.name == "mcp_poc.respond")
                .expect("missing respond span");
            assert_eq!(
                respond_span.fields.get("mcp.method").map(String::as_str),
                Some(bounded_method)
            );
            assert_eq!(
                respond_span.fields.get("mcp.outcome").map(String::as_str),
                Some(respond_outcome),
                "respond span should record a bounded outcome"
            );
            assert_eq!(
                respond_span.fields.len(),
                2,
                "respond span must not capture request data"
            );
        }
    }


    #[tokio::test]
    async fn classify_and_respond_stages_serve_tools_list() {
        let (_directory, classify, respond) = filter_pair();
        let (request, body) = request("tools/list", None);
        let mut body = Some(body);
        let mut context = crate::test_utils::make_filter_context(&request);
        context.current_filter_id = Some(0);
        context.extensions.insert(all_capabilities());

        let action = classify.on_request_body(&mut context, &mut body, true).await.unwrap();

        assert!(matches!(action, FilterAction::Continue));
        assert_eq!(context.get_metadata("mcp.method"), Some("tools/list"));
        assert_eq!(context.get_metadata("mcp.name"), None);
        assert!(matches!(
            classify.on_request(&mut context).await.unwrap(),
            FilterAction::Continue
        ));

        context.current_filter_id = Some(1);
        let action = respond.on_request(&mut context).await.unwrap();
        let FilterAction::TerminalResponse(response) = action else {
            panic!("response stage should serve tools/list");
        };
        assert_eq!(response.status, 200);
        let response: serde_json::Value = serde_json::from_slice(response.body.as_ref().unwrap()).unwrap();
        let tools = response["result"]["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0]["name"], "py_echo");
        assert_eq!(tools[1]["name"], "everything_get-sum");
        assert_eq!(tools[0]["inputSchema"]["properties"]["message"]["type"], "string");
    }

    #[tokio::test]
    async fn tools_list_filters_by_grant_and_always_uses_private_cache_scope() {
        let (_directory, classify, respond) = filter_pair_with_cache_scope("public");
        let (request, body) = request("tools/list", None);
        let mut body = Some(body);
        let mut context = crate::test_utils::make_filter_context(&request);
        context.extensions.insert(python_echo_capabilities());
        context.current_filter_id = Some(0);

        assert!(matches!(
            classify.on_request_body(&mut context, &mut body, true).await.unwrap(),
            FilterAction::Continue
        ));
        assert!(matches!(
            classify.on_request(&mut context).await.unwrap(),
            FilterAction::Continue
        ));
        context.current_filter_id = Some(1);
        let FilterAction::TerminalResponse(response) = respond.on_request(&mut context).await.unwrap() else {
            panic!("response stage should filter tools/list");
        };
        let response: serde_json::Value = serde_json::from_slice(response.body.as_ref().unwrap()).unwrap();
        assert_eq!(response["result"]["cacheScope"], "private");
        let tools = response["result"]["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "py_echo");
    }

    #[tokio::test]
    async fn tools_list_without_policy_capabilities_fails_closed() {
        let (_directory, classify, respond) = filter_pair();
        let (request, body) = request("tools/list", None);
        let mut body = Some(body);
        let mut context = crate::test_utils::make_filter_context(&request);
        context.current_filter_id = Some(0);

        assert!(matches!(
            classify.on_request_body(&mut context, &mut body, true).await.unwrap(),
            FilterAction::Continue
        ));
        assert!(matches!(
            classify.on_request(&mut context).await.unwrap(),
            FilterAction::Continue
        ));
        context.current_filter_id = Some(1);
        let FilterAction::TerminalResponse(response) = respond.on_request(&mut context).await.unwrap() else {
            panic!("response stage should fail closed without policy capabilities");
        };
        let response: serde_json::Value = serde_json::from_slice(response.body.as_ref().unwrap()).unwrap();
        assert_eq!(response["error"]["code"], -32003);
    }

    #[tokio::test]
    async fn classify_and_respond_stages_serve_server_discover() {
        let (_directory, classify, respond) = filter_pair();
        let (request, body) = request("server/discover", None);
        let mut body = Some(body);
        let mut context = crate::test_utils::make_filter_context(&request);
        context.current_filter_id = Some(0);

        let action = classify.on_request_body(&mut context, &mut body, true).await.unwrap();

        assert!(matches!(action, FilterAction::Continue));
        assert!(matches!(
            classify.on_request(&mut context).await.unwrap(),
            FilterAction::Continue
        ));

        context.current_filter_id = Some(1);
        let action = respond.on_request(&mut context).await.unwrap();
        let FilterAction::TerminalResponse(response) = action else {
            panic!("response stage should serve server/discover");
        };
        assert_eq!(response.status, 200);
        let response: serde_json::Value = serde_json::from_slice(response.body.as_ref().unwrap()).unwrap();
        assert_eq!(
            response["result"]["supportedVersions"],
            serde_json::json!(["2026-07-28"])
        );
    }

    #[tokio::test]
    async fn classify_stage_routes_tool_calls_and_response_stage_passes_them() {
        let (_directory, classify, respond) = filter_pair();
        let (request, body) = request("tools/call", Some("everything_get-sum"));
        let mut body = Some(body);
        let mut context = crate::test_utils::make_filter_context(&request);
        context.current_filter_id = Some(0);
        context.extensions.insert(all_capabilities());

        let action = classify.on_request_body(&mut context, &mut body, true).await.unwrap();

        assert!(matches!(action, FilterAction::Release));
        assert_eq!(context.get_metadata("mcp.method"), Some("tools/call"));
        assert_eq!(context.get_metadata("mcp.name"), Some("everything_get-sum"));
        let forwarded: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
        assert_eq!(forwarded["params"]["name"], "get-sum");
        let route_header = context
            .extra_request_headers
            .iter()
            .find(|(name, _)| name.as_ref() == "x-praxis-mcp-target")
            .map(|(_, value)| value.as_str());
        assert_eq!(route_header, Some("everything"));
        assert_eq!(context.rewritten_path.as_deref(), Some("/mcp"));
        assert!(
            context
                .request_headers_to_set
                .iter()
                .any(|(name, value)| { name.as_str() == "mcp-name" && value.as_bytes() == b"get-sum" })
        );
        assert!(matches!(
            classify.on_request(&mut context).await.unwrap(),
            FilterAction::Continue
        ));

        context.current_filter_id = Some(1);
        assert!(matches!(
            respond.on_request(&mut context).await.unwrap(),
            FilterAction::Continue
        ));
    }

    #[tokio::test]
    async fn response_stage_denies_a_hidden_direct_tool_call_before_routing() {
        let (_directory, classify, respond) = filter_pair();
        let (request, body) = request("tools/call", Some("everything_get-sum"));
        let mut body = Some(body);
        let mut context = crate::test_utils::make_filter_context(&request);
        context.extensions.insert(python_echo_capabilities());
        context.current_filter_id = Some(0);

        assert!(matches!(
            classify.on_request_body(&mut context, &mut body, true).await.unwrap(),
            FilterAction::Release
        ));
        assert!(matches!(
            classify.on_request(&mut context).await.unwrap(),
            FilterAction::Continue
        ));
        context.current_filter_id = Some(1);
        let FilterAction::TerminalResponse(response) = respond.on_request(&mut context).await.unwrap() else {
            panic!("response stage must deny a hidden tool before the router");
        };
        let response: serde_json::Value = serde_json::from_slice(response.body.as_ref().unwrap()).unwrap();
        assert_eq!(response["error"]["code"], -32003);
        assert_eq!(response["error"]["message"], "not authorized to call this tool");
    }

    #[tokio::test]
    async fn respond_stage_returns_unknown_tool_error_after_classification() {
        let (_directory, classify, respond) = filter_pair();
        let (request, body) = request("tools/call", Some("everything_missing"));
        let mut body = Some(body);
        let mut context = crate::test_utils::make_filter_context(&request);
        context.current_filter_id = Some(0);

        let action = classify.on_request_body(&mut context, &mut body, true).await.unwrap();

        assert!(matches!(action, FilterAction::Continue));
        assert!(context.extra_request_headers.is_empty());
        assert!(matches!(
            classify.on_request(&mut context).await.unwrap(),
            FilterAction::Continue
        ));
        context.current_filter_id = Some(1);
        let FilterAction::TerminalResponse(response) = respond.on_request(&mut context).await.unwrap() else {
            panic!("response stage should reject unknown tools before routing");
        };
        assert_eq!(response.status, 200);
        let response: serde_json::Value = serde_json::from_slice(response.body.as_ref().unwrap()).unwrap();
        assert_eq!(response["error"]["code"], -32602);
    }

    #[tokio::test]
    async fn classify_stage_rejects_incomplete_stateless_client_metadata() {
        let (_directory, classify, _respond) = filter_pair();
        let (request, body) = request("tools/list", None);
        let mut value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        value["params"]["_meta"]
            .as_object_mut()
            .unwrap()
            .remove("io.modelcontextprotocol/clientCapabilities");
        let mut body = Some(Bytes::from(value.to_string()));
        let mut context = crate::test_utils::make_filter_context(&request);
        context.current_filter_id = Some(0);

        let action = classify.on_request_body(&mut context, &mut body, true).await.unwrap();

        let FilterAction::Reject(rejection) = action else {
            panic!("an incomplete stateless metadata envelope must be rejected");
        };
        assert_eq!(rejection.status, 400);
        let response: serde_json::Value = serde_json::from_slice(rejection.body.as_ref().unwrap()).unwrap();
        assert_eq!(response["error"]["code"], -32001);
    }
}
