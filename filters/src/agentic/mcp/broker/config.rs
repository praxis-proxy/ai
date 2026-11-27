// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Configuration types for the MCP static catalog filter.

use std::{collections::HashMap, time::Duration};

use praxis_ai_apis::mcp_admission::ConnectorToolPolicy;
use praxis_core::config::FilterEntry;
use praxis_filter::{
    FilterError, builtins::http::payload_processing::config_validation::validate_max_body_bytes, has_dot_dot_traversal,
};
use serde::Deserialize;

use super::super::{
    config::DEFAULT_MAX_BODY_BYTES,
    protocol::{self, ProtocolProfile},
};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Default cache TTL in milliseconds (5 minutes).
pub(super) const DEFAULT_CACHE_TTL_MS: u64 = 300_000; // 5 min

/// Default maximum number of retained backend sessions.
const DEFAULT_MAX_BACKEND_SESSIONS: usize = 4_096;

/// Upper bound on `backend_sessions.max_sessions`.
const MAX_BACKEND_SESSIONS_LIMIT: usize = 1_000_000;

/// Default idle lifetime of a backend session mapping (10 minutes).
const DEFAULT_BACKEND_IDLE_TIMEOUT_MS: u64 = 600_000;

/// Default deadline for one backend handshake exchange (5 seconds).
const DEFAULT_HANDSHAKE_TIMEOUT_MS: u64 = 5_000;

/// Default ceiling for one backend handshake response body (1 MiB).
const DEFAULT_HANDSHAKE_RESPONSE_BYTES: usize = 1_048_576;

/// Upper bound on `backend_sessions.max_response_bytes` (16 MiB).
const MAX_HANDSHAKE_RESPONSE_BYTES_LIMIT: usize = 16 * 1_048_576;

/// Default maximum number of cached `(caller, server)` tool listings.
const DEFAULT_MAX_DISCOVERY_ENTRIES: usize = 4_096;

/// Upper bound on `discovery.max_entries`.
const MAX_DISCOVERY_ENTRIES_LIMIT: usize = 1_000_000;

/// Default age after which a caller's tools for a server are rediscovered
/// (5 minutes).
const DEFAULT_DISCOVERY_MAX_AGE_MS: u64 = 300_000;

/// Default time after which a listing no request has read is dropped
/// (10 minutes).
const DEFAULT_DISCOVERY_IDLE_TIMEOUT_MS: u64 = 600_000;

/// Default number of servers discovered concurrently for one request.
const DEFAULT_DISCOVERY_CONCURRENCY: usize = 8;

/// Upper bound on `discovery.max_concurrency`.
const MAX_DISCOVERY_CONCURRENCY_LIMIT: usize = 64;

/// Default maximum number of tools one server may list.
const DEFAULT_DISCOVERY_MAX_TOOLS: usize = 512;

/// Upper bound on `discovery.max_tools`.
const MAX_DISCOVERY_MAX_TOOLS_LIMIT: usize = 10_000;

// -----------------------------------------------------------------------------
// CacheScope
// -----------------------------------------------------------------------------

/// Cache scope for stateless MCP responses.
#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum CacheScope {
    /// Response may be cached by any intermediary.
    #[default]
    Public,
    /// Response may only be cached by the requesting client.
    Private,
}

impl CacheScope {
    /// String representation for response serialization.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Public => "public",
            Self::Private => "private",
        }
    }
}

// -----------------------------------------------------------------------------
// InvalidToolPolicy
// -----------------------------------------------------------------------------

/// Behavior when a tool definition has an invalid schema.
#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum InvalidToolPolicy {
    /// Reject the entire server config at load time.
    #[default]
    RejectServer,
    /// Exclude the invalid tool from the exposed catalog, keeping
    /// the rest of the server's tools.
    FilterOut,
}

// -----------------------------------------------------------------------------
// ToolConfig
// -----------------------------------------------------------------------------

/// Tool definition in static config.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ToolConfig {
    /// Tool name on the backend.
    pub name: String,
    /// Optional description.
    pub description: Option<String>,
    /// Optional input schema. `schema` is accepted as a local shorthand.
    #[serde(rename = "inputSchema", alias = "input_schema", alias = "schema")]
    pub input_schema: Option<serde_json::Value>,
    /// Optional tool annotations.
    pub annotations: Option<serde_json::Value>,
    /// Hide the discovered tool of this name. Requires `discover` on the
    /// server.
    #[serde(default)]
    pub hidden: bool,
}

// -----------------------------------------------------------------------------
// DiscoveryToolPolicy
// -----------------------------------------------------------------------------

/// Which discovered tools a server may expose.
#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum DiscoveryToolPolicy {
    /// Expose every tool the server lists. This is the default.
    #[default]
    AllowAll,
    /// Expose only tools the server affirmatively annotates as read-only
    /// (`readOnlyHint: true` without `destructiveHint: true`).
    ReadOnly,
}

impl DiscoveryToolPolicy {
    /// The admission policy discovered tools are vetted with.
    fn admission(self) -> ConnectorToolPolicy {
        match self {
            Self::AllowAll => ConnectorToolPolicy::AllowAll,
            Self::ReadOnly => ConnectorToolPolicy::ReadOnly,
        }
    }
}

// -----------------------------------------------------------------------------
// McpServerConfig
// -----------------------------------------------------------------------------

/// MCP backend server configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct McpServerConfig {
    /// Unique server name.
    pub name: String,
    /// Backend cluster name.
    pub cluster: String,
    /// Backend MCP path.
    #[serde(default = "default_path")]
    pub path: String,
    /// Discover this server's tools from its `tools/list`.
    #[serde(default)]
    pub discover: bool,
    /// Tool prefix for this server.
    pub tool_prefix: Option<String>,
    /// Which discovered tools the server may expose. Requires `discover`.
    pub tool_policy: Option<DiscoveryToolPolicy>,
    /// Statically defined tools. With `discover`, an entry named like a
    /// discovered tool overrides its description or hides it.
    #[serde(default)]
    pub tools: Vec<ToolConfig>,
}

// -----------------------------------------------------------------------------
// DiscoveryConfig
// -----------------------------------------------------------------------------

/// Settings shared by every server with `discover: true`.
///
/// Current-profile discovery runs inside the caller's backend sessions, so
/// its credential, transport, and time budget come from `backend_sessions`.
/// The stateless-only fields configure the same for the stateless profile.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct DiscoveryConfig {
    /// Request headers copied from the client request onto discovery
    /// exchanges. Stateless profile only.
    pub forward_headers: Option<Vec<String>>,
    /// Time after which a cached listing no request has read is dropped, in
    /// milliseconds.
    #[serde(default = "default_discovery_idle_timeout_ms")]
    pub idle_timeout_ms: u64,
    /// Age after which a caller's tools for a server are rediscovered, in
    /// milliseconds. A failed rediscovery keeps serving the older listing.
    #[serde(default = "default_discovery_max_age_ms")]
    pub max_age_ms: u64,
    /// Servers discovered concurrently for one request.
    #[serde(default = "default_discovery_concurrency")]
    pub max_concurrency: usize,
    /// Maximum cached `(caller, server)` tool listings.
    #[serde(default = "default_max_discovery_entries")]
    pub max_entries: usize,
    /// Response body ceiling for each `tools/list` page, in bytes.
    /// Stateless profile only.
    pub max_response_bytes: Option<usize>,
    /// Maximum tools one server may list.
    #[serde(default = "default_discovery_max_tools")]
    pub max_tools: usize,
    /// Extra filters run on each discovery subrequest before the broker's
    /// generated `load_balancer`. Stateless profile only.
    pub outbound_filters: Option<Vec<FilterEntry>>,
    /// Budget for discovering every server one request needs, in
    /// milliseconds. Stateless profile only.
    pub timeout_ms: Option<u64>,
}

/// Normalized discovery settings.
#[derive(Debug, Clone)]
pub(crate) struct ValidatedDiscovery {
    /// Time after which a cached listing no request has read is dropped.
    pub idle_timeout: Duration,
    /// Age after which a caller's tools for a server are rediscovered.
    pub max_age: Duration,
    /// Servers discovered concurrently for one request.
    pub max_concurrency: usize,
    /// Maximum cached `(caller, server)` tool listings.
    pub max_entries: usize,
    /// Maximum tools one server may list.
    pub max_tools: usize,
    /// Per-server discovery settings, in configuration order; `None` for a
    /// server without `discover`.
    pub servers: Vec<Option<ServerDiscovery>>,
    /// Stateless-profile transport settings.
    pub stateless: Option<StatelessDiscovery>,
}

/// Stateless-profile discovery transport settings.
#[derive(Debug, Clone)]
pub(crate) struct StatelessDiscovery {
    /// Client request headers copied onto discovery exchanges.
    pub forward_headers: Vec<http::HeaderName>,
    /// Response body ceiling for each `tools/list` page.
    pub max_response_bytes: usize,
    /// Budget for discovering every server one request needs.
    pub timeout: Duration,
}

/// Discovery settings for one server.
#[derive(Debug, Clone)]
pub(crate) struct ServerDiscovery {
    /// Behavior for a discovered tool with an invalid input schema.
    pub invalid_tool_policy: InvalidToolPolicy,
    /// Static entries by backend tool name.
    pub overrides: HashMap<String, ToolOverride>,
    /// Admission policy discovered tools are vetted with.
    pub tool_policy: ConnectorToolPolicy,
}

/// Static entry applied to the discovered tool of the same name.
#[derive(Debug, Clone)]
pub(crate) struct ToolOverride {
    /// Replacement description.
    pub description: Option<String>,
    /// Whether the tool is hidden.
    pub hidden: bool,
}

// -----------------------------------------------------------------------------
// BackendSessionsConfig
// -----------------------------------------------------------------------------

/// Current-profile backend session management.
///
/// When present, current-profile `tools/call` lazily runs the backend
/// `initialize` / `notifications/initialized` handshake once per
/// `(gateway session, server)` pair and forwards the call to the server's
/// cluster with the backend `Mcp-Session-Id`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct BackendSessionsConfig {
    /// Request headers copied from the client request onto the handshake.
    #[serde(default = "default_forward_headers")]
    pub forward_headers: Vec<String>,
    /// Idle lifetime of one backend session mapping, in milliseconds.
    #[serde(default = "default_backend_idle_timeout_ms")]
    pub idle_timeout_ms: u64,
    /// Budget for one whole handshake, in milliseconds: waiting for a
    /// concurrent handshake, `initialize`, and `notifications/initialized`
    /// share it, and a parent deadline can shorten it.
    #[serde(default = "default_handshake_timeout_ms")]
    pub handshake_timeout_ms: u64,
    /// Maximum retained backend session mappings.
    #[serde(default = "default_max_backend_sessions")]
    pub max_sessions: usize,
    /// Response body ceiling for each handshake exchange, in bytes.
    #[serde(default = "default_handshake_response_bytes")]
    pub max_response_bytes: usize,
    /// Extra filters run on each handshake subrequest before the broker's
    /// generated `load_balancer` (for example credential injection).
    #[serde(default)]
    pub outbound_filters: Vec<FilterEntry>,
}

/// Normalized backend session settings.
#[derive(Debug, Clone)]
pub(crate) struct ValidatedBackendSessions {
    /// Client request headers copied onto handshake exchanges.
    pub forward_headers: Vec<http::HeaderName>,
    /// Deadline for one handshake exchange.
    pub handshake_timeout: Duration,
    /// Idle lifetime of one backend session mapping.
    pub idle_timeout: Duration,
    /// Maximum retained backend session mappings.
    pub max_sessions: usize,
    /// Response body ceiling for each handshake exchange.
    pub max_response_bytes: usize,
}

// -----------------------------------------------------------------------------
// McpBrokerConfig (raw deserialized)
// -----------------------------------------------------------------------------

/// MCP broker filter configuration.
///
/// Supports two protocol profiles: `current` (session-based, default) and
/// `stateless` (MCP 2026-07-28, configurable). Version and cache fields are
/// derived from the selected profile when omitted.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct McpBrokerConfig {
    /// Current-profile backend session management. Requires
    /// `protocol_profile: current`.
    pub backend_sessions: Option<BackendSessionsConfig>,
    /// Cache scope for stateless responses. Requires `protocol_profile: stateless`.
    pub cache_scope: Option<CacheScope>,
    /// Cache TTL in milliseconds for stateless responses. Requires `protocol_profile: stateless`.
    pub cache_ttl_ms: Option<u64>,
    /// Fallback MCP protocol version. When omitted, derived from the profile.
    pub default_version: Option<String>,
    /// Runtime tool discovery settings for servers with `discover: true`.
    pub discovery: Option<DiscoveryConfig>,
    /// Behavior when a tool has an invalid schema.
    #[serde(default)]
    pub invalid_tool_policy: InvalidToolPolicy,
    /// Maximum body size in bytes.
    #[serde(default = "default_max_body_bytes")]
    pub max_body_bytes: usize,
    /// Public MCP path handled by Praxis.
    #[serde(default = "default_path")]
    pub path: String,
    /// Protocol profile governing session semantics and header
    /// requirements for this broker instance.
    #[serde(default)]
    pub protocol_profile: ProtocolProfile,
    /// Backend server definitions.
    #[serde(default)]
    pub servers: Vec<McpServerConfig>,
    /// Protocol versions accepted during negotiation.
    /// When omitted, derived from the profile.
    pub supported_versions: Option<Vec<String>>,
}

// -----------------------------------------------------------------------------
// ValidatedBrokerConfig (normalized)
// -----------------------------------------------------------------------------

/// Normalized broker configuration with all defaults resolved.
#[derive(Debug)]
pub(super) struct ValidatedBrokerConfig {
    /// Current-profile backend session settings, when enabled.
    pub backend_sessions: Option<ValidatedBackendSessions>,
    /// Cache scope for stateless responses.
    pub cache_scope: CacheScope,
    /// Cache TTL in milliseconds for stateless responses.
    pub cache_ttl_ms: u64,
    /// Fallback MCP protocol version.
    pub default_version: String,
    /// Runtime tool discovery settings, when any server discovers.
    pub discovery: Option<ValidatedDiscovery>,
    /// Maximum body size in bytes.
    pub max_body_bytes: usize,
    /// Public MCP path handled by Praxis.
    pub path: String,
    /// Protocol profile for this broker instance.
    pub protocol_profile: ProtocolProfile,
    /// Protocol versions accepted during negotiation.
    pub supported_versions: Vec<String>,
}

// -----------------------------------------------------------------------------
// CatalogTool
// -----------------------------------------------------------------------------

/// Entry in the pre-built tool catalog.
#[derive(Debug, Clone)]
pub(crate) struct CatalogTool {
    /// Optional tool annotations.
    pub annotations: Option<serde_json::Value>,
    /// Backend MCP endpoint path.
    pub backend_path: String,
    /// Backend cluster name.
    pub cluster: String,
    /// Optional description.
    pub description: Option<String>,
    /// Exposed (prefixed) tool name visible to clients.
    pub exposed_name: String,
    /// Other fields of a discovered tool definition (`title`,
    /// `outputSchema`, ...), served unchanged. Empty for static tools.
    pub extra: serde_json::Map<String, serde_json::Value>,
    /// MCP input schema.
    pub input_schema: serde_json::Value,
    /// Original tool name on the backend.
    pub original_name: String,
    /// Index of the owning server in configuration order.
    pub server_index: usize,
    /// Backend server name from config.
    pub server_name: String,
}

// -----------------------------------------------------------------------------
// Defaults
// -----------------------------------------------------------------------------

/// Default MCP path.
fn default_path() -> String {
    "/mcp".to_owned()
}

/// Default max body bytes.
fn default_max_body_bytes() -> usize {
    DEFAULT_MAX_BODY_BYTES
}

/// Forward the client's bearer credential to the handshake by default, as the
/// forwarded `tools/call` itself carries it.
fn default_forward_headers() -> Vec<String> {
    vec!["authorization".to_owned()]
}

/// Default backend session idle timeout.
fn default_backend_idle_timeout_ms() -> u64 {
    DEFAULT_BACKEND_IDLE_TIMEOUT_MS
}

/// Default handshake deadline.
fn default_handshake_timeout_ms() -> u64 {
    DEFAULT_HANDSHAKE_TIMEOUT_MS
}

/// Default retained backend session cap.
fn default_max_backend_sessions() -> usize {
    DEFAULT_MAX_BACKEND_SESSIONS
}

/// Default handshake response ceiling.
fn default_handshake_response_bytes() -> usize {
    DEFAULT_HANDSHAKE_RESPONSE_BYTES
}

/// Default discovery cache entry cap.
fn default_max_discovery_entries() -> usize {
    DEFAULT_MAX_DISCOVERY_ENTRIES
}

/// Default discovery idle timeout.
fn default_discovery_idle_timeout_ms() -> u64 {
    DEFAULT_DISCOVERY_IDLE_TIMEOUT_MS
}

/// Default discovery age.
fn default_discovery_max_age_ms() -> u64 {
    DEFAULT_DISCOVERY_MAX_AGE_MS
}

/// Default discovery concurrency.
fn default_discovery_concurrency() -> usize {
    DEFAULT_DISCOVERY_CONCURRENCY
}

/// Default per-server tool cap.
fn default_discovery_max_tools() -> usize {
    DEFAULT_DISCOVERY_MAX_TOOLS
}

// -----------------------------------------------------------------------------
// Validation
// -----------------------------------------------------------------------------

/// Validate configuration and build the static tool catalog.
pub(super) fn build_config(cfg: McpBrokerConfig) -> Result<(ValidatedBrokerConfig, Vec<CatalogTool>), FilterError> {
    validate_max_body_bytes("mcp_broker", cfg.max_body_bytes)?;

    let profile = cfg.protocol_profile;

    let backend_sessions = validate_profile_fields(profile, &cfg)?;

    let catalog = validate_and_build_catalog(&cfg)?;

    let discovery = validate_discovery(profile, &cfg)?;

    let default_version = cfg
        .default_version
        .unwrap_or_else(|| protocol::default_version_for_profile(profile).to_owned());

    let supported_versions = cfg.supported_versions.unwrap_or_else(|| {
        protocol::supported_versions_for_profile(profile)
            .iter()
            .map(|s| (*s).to_owned())
            .collect()
    });

    let cache_scope = cfg.cache_scope.unwrap_or(CacheScope::Public);
    let cache_ttl_ms = cfg.cache_ttl_ms.unwrap_or(DEFAULT_CACHE_TTL_MS);

    validate_versions(profile, &supported_versions, &default_version)?;

    let validated = ValidatedBrokerConfig {
        backend_sessions,
        cache_scope,
        cache_ttl_ms,
        default_version,
        discovery,
        max_body_bytes: cfg.max_body_bytes,
        path: cfg.path,
        protocol_profile: profile,
        supported_versions,
    };

    Ok((validated, catalog))
}

/// Validate server definitions and build the static tool catalog.
fn validate_and_build_catalog(cfg: &McpBrokerConfig) -> Result<Vec<CatalogTool>, FilterError> {
    validate_path("mcp", &cfg.path)?;
    validate_unique_server_names(&cfg.servers)?;
    validate_server_clusters(&cfg.servers)?;
    validate_server_paths(&cfg.servers)?;
    validate_tool_names(&cfg.servers)?;

    let catalog = build_catalog(&cfg.servers, cfg.invalid_tool_policy)?;
    validate_unique_exposed_names(&catalog)?;

    Ok(catalog)
}

/// Validate profile-specific fields and normalize `backend_sessions`.
fn validate_profile_fields(
    profile: ProtocolProfile,
    cfg: &McpBrokerConfig,
) -> Result<Option<ValidatedBackendSessions>, FilterError> {
    validate_cache_fields_for_profile(profile, cfg)?;
    cfg.backend_sessions
        .as_ref()
        .map(|sessions| validate_backend_sessions(profile, sessions))
        .transpose()
}

/// Reject explicit cache config on profiles that do not use it.
fn validate_cache_fields_for_profile(profile: ProtocolProfile, cfg: &McpBrokerConfig) -> Result<(), FilterError> {
    if profile == ProtocolProfile::Current {
        if cfg.cache_scope.is_some() {
            return Err("mcp: cache_scope requires protocol_profile 'stateless'".into());
        }
        if cfg.cache_ttl_ms.is_some() {
            return Err("mcp: cache_ttl_ms requires protocol_profile 'stateless'".into());
        }
    }
    Ok(())
}

/// Validate current-profile backend session settings.
fn validate_backend_sessions(
    profile: ProtocolProfile,
    cfg: &BackendSessionsConfig,
) -> Result<ValidatedBackendSessions, FilterError> {
    if profile != ProtocolProfile::Current {
        return Err("mcp: backend_sessions requires protocol_profile 'current'".into());
    }
    validate_backend_session_limits(cfg)?;
    if cfg
        .outbound_filters
        .iter()
        .any(|entry| entry.filter_type == "load_balancer")
    {
        return Err(
            "mcp: backend_sessions.outbound_filters must not contain load_balancer; the broker routes handshakes \
             to each server's top-level cluster"
                .into(),
        );
    }

    Ok(ValidatedBackendSessions {
        forward_headers: validate_forward_headers("backend_sessions", &cfg.forward_headers)?,
        handshake_timeout: Duration::from_millis(cfg.handshake_timeout_ms),
        idle_timeout: Duration::from_millis(cfg.idle_timeout_ms),
        max_sessions: cfg.max_sessions,
        max_response_bytes: cfg.max_response_bytes,
    })
}

/// Range-check numeric backend session settings.
fn validate_backend_session_limits(cfg: &BackendSessionsConfig) -> Result<(), FilterError> {
    if cfg.max_sessions == 0 || cfg.max_sessions > MAX_BACKEND_SESSIONS_LIMIT {
        return Err(
            format!("mcp: backend_sessions.max_sessions must be between 1 and {MAX_BACKEND_SESSIONS_LIMIT}").into(),
        );
    }
    if cfg.idle_timeout_ms == 0 {
        return Err("mcp: backend_sessions.idle_timeout_ms must be greater than 0".into());
    }
    if cfg.handshake_timeout_ms == 0 {
        return Err("mcp: backend_sessions.handshake_timeout_ms must be greater than 0".into());
    }
    if cfg.max_response_bytes == 0 || cfg.max_response_bytes > MAX_HANDSHAKE_RESPONSE_BYTES_LIMIT {
        return Err(format!(
            "mcp: backend_sessions.max_response_bytes must be between 1 and {MAX_HANDSHAKE_RESPONSE_BYTES_LIMIT}"
        )
        .into());
    }
    Ok(())
}

/// Parse forwarded header names, rejecting names the broker or transport owns.
fn validate_forward_headers(block: &str, names: &[String]) -> Result<Vec<http::HeaderName>, FilterError> {
    let mut parsed = Vec::with_capacity(names.len());
    for raw in names {
        let name = http::HeaderName::from_bytes(raw.as_bytes())
            .map_err(|_e| FilterError::from(format!("mcp: {block}.forward_headers has invalid name '{raw}'")))?;
        if is_reserved_forward_header(&name) {
            return Err(format!("mcp: {block}.forward_headers must not include '{name}'").into());
        }
        if !parsed.contains(&name) {
            parsed.push(name);
        }
    }
    Ok(parsed)
}

/// Headers the handshake sets itself or that must never be copied.
fn is_reserved_forward_header(name: &http::HeaderName) -> bool {
    if praxis_ai_apis::http_hop::is_hop_by_hop(name.as_str()) {
        return true;
    }
    let s = name.as_str();
    matches!(
        *name,
        http::header::ACCEPT | http::header::CONTENT_LENGTH | http::header::CONTENT_TYPE | http::header::HOST
    ) || matches!(s, "mcp-session-id" | "mcp-protocol-version" | "last-event-id")
        || s.starts_with("x-praxis-")
}

/// Validate per-server discovery fields and normalize the `discovery` block.
///
/// Returns `None` when no server discovers.
#[expect(clippy::too_many_lines, reason = "linear validation with early returns")]
fn validate_discovery(
    profile: ProtocolProfile,
    cfg: &McpBrokerConfig,
) -> Result<Option<ValidatedDiscovery>, FilterError> {
    validate_server_discovery_fields(&cfg.servers)?;
    let any_discovers = cfg.servers.iter().any(|server| server.discover);
    if !any_discovers {
        if cfg.discovery.is_some() {
            return Err("mcp: discovery requires at least one server with discover: true".into());
        }
        return Ok(None);
    }
    let settings = cfg.discovery.clone().unwrap_or_else(|| DiscoveryConfig {
        forward_headers: None,
        idle_timeout_ms: DEFAULT_DISCOVERY_IDLE_TIMEOUT_MS,
        max_age_ms: DEFAULT_DISCOVERY_MAX_AGE_MS,
        max_concurrency: DEFAULT_DISCOVERY_CONCURRENCY,
        max_entries: DEFAULT_MAX_DISCOVERY_ENTRIES,
        max_response_bytes: None,
        max_tools: DEFAULT_DISCOVERY_MAX_TOOLS,
        outbound_filters: None,
        timeout_ms: None,
    });
    validate_discovery_limits(&settings)?;
    let stateless = match profile {
        ProtocolProfile::Current => {
            validate_current_discovery(cfg, &settings)?;
            None
        },
        ProtocolProfile::Stateless => {
            if cfg.cache_scope == Some(CacheScope::Public) {
                return Err(
                    "mcp: cache_scope 'public' conflicts with discover: true; discovered tools are per caller".into(),
                );
            }
            Some(validate_stateless_discovery(&settings)?)
        },
    };
    let servers = cfg
        .servers
        .iter()
        .map(|server| {
            server.discover.then(|| ServerDiscovery {
                invalid_tool_policy: cfg.invalid_tool_policy,
                overrides: server
                    .tools
                    .iter()
                    .map(|tool| {
                        (
                            tool.name.clone(),
                            ToolOverride {
                                description: tool.description.clone(),
                                hidden: tool.hidden,
                            },
                        )
                    })
                    .collect(),
                tool_policy: server.tool_policy.unwrap_or_default().admission(),
            })
        })
        .collect();
    Ok(Some(ValidatedDiscovery {
        idle_timeout: Duration::from_millis(settings.idle_timeout_ms),
        max_age: Duration::from_millis(settings.max_age_ms),
        max_concurrency: settings.max_concurrency,
        max_entries: settings.max_entries,
        max_tools: settings.max_tools,
        servers,
        stateless,
    }))
}

/// Reject `tool_policy` and `hidden` on servers that do not discover.
fn validate_server_discovery_fields(servers: &[McpServerConfig]) -> Result<(), FilterError> {
    for server in servers.iter().filter(|server| !server.discover) {
        if server.tool_policy.is_some() {
            return Err(format!("mcp: server '{}' tool_policy requires discover: true", server.name).into());
        }
        if let Some(tool) = server.tools.iter().find(|tool| tool.hidden) {
            return Err(format!(
                "mcp: server '{}' tool '{}' hidden requires discover: true",
                server.name, tool.name
            )
            .into());
        }
    }
    Ok(())
}

/// Range-check numeric discovery settings.
fn validate_discovery_limits(settings: &DiscoveryConfig) -> Result<(), FilterError> {
    if settings.max_entries == 0 || settings.max_entries > MAX_DISCOVERY_ENTRIES_LIMIT {
        return Err(format!("mcp: discovery.max_entries must be between 1 and {MAX_DISCOVERY_ENTRIES_LIMIT}").into());
    }
    if settings.max_age_ms == 0 {
        return Err("mcp: discovery.max_age_ms must be greater than 0".into());
    }
    if settings.idle_timeout_ms == 0 {
        return Err("mcp: discovery.idle_timeout_ms must be greater than 0".into());
    }
    if settings.max_concurrency == 0 || settings.max_concurrency > MAX_DISCOVERY_CONCURRENCY_LIMIT {
        return Err(
            format!("mcp: discovery.max_concurrency must be between 1 and {MAX_DISCOVERY_CONCURRENCY_LIMIT}").into(),
        );
    }
    if settings.max_tools == 0 || settings.max_tools > MAX_DISCOVERY_MAX_TOOLS_LIMIT {
        return Err(format!("mcp: discovery.max_tools must be between 1 and {MAX_DISCOVERY_MAX_TOOLS_LIMIT}").into());
    }
    Ok(())
}

/// Current-profile discovery runs in backend sessions: require them, and
/// reject the stateless-only transport fields.
fn validate_current_discovery(cfg: &McpBrokerConfig, settings: &DiscoveryConfig) -> Result<(), FilterError> {
    if cfg.backend_sessions.is_none() {
        return Err("mcp: discover: true requires backend_sessions in protocol_profile 'current'".into());
    }
    let stateless_only = [
        ("forward_headers", settings.forward_headers.is_some()),
        ("max_response_bytes", settings.max_response_bytes.is_some()),
        ("outbound_filters", settings.outbound_filters.is_some()),
        ("timeout_ms", settings.timeout_ms.is_some()),
    ];
    if let Some((field, _)) = stateless_only.iter().find(|(_, set)| *set) {
        return Err(format!(
            "mcp: discovery.{field} requires protocol_profile 'stateless'; current-profile discovery uses \
             backend_sessions.{}",
            match *field {
                "timeout_ms" => "handshake_timeout_ms",
                other => other,
            }
        )
        .into());
    }
    Ok(())
}

/// Normalize stateless-profile discovery transport settings.
fn validate_stateless_discovery(settings: &DiscoveryConfig) -> Result<StatelessDiscovery, FilterError> {
    let timeout_ms = settings.timeout_ms.unwrap_or(DEFAULT_HANDSHAKE_TIMEOUT_MS);
    if timeout_ms == 0 {
        return Err("mcp: discovery.timeout_ms must be greater than 0".into());
    }
    let max_response_bytes = settings.max_response_bytes.unwrap_or(DEFAULT_HANDSHAKE_RESPONSE_BYTES);
    if max_response_bytes == 0 || max_response_bytes > MAX_HANDSHAKE_RESPONSE_BYTES_LIMIT {
        return Err(format!(
            "mcp: discovery.max_response_bytes must be between 1 and {MAX_HANDSHAKE_RESPONSE_BYTES_LIMIT}"
        )
        .into());
    }
    if settings
        .outbound_filters
        .iter()
        .flatten()
        .any(|entry| entry.filter_type == "load_balancer")
    {
        return Err(
            "mcp: discovery.outbound_filters must not contain load_balancer; the broker routes discovery to each \
             server's top-level cluster"
                .into(),
        );
    }
    let forward_headers = settings.forward_headers.clone().unwrap_or_else(default_forward_headers);
    Ok(StatelessDiscovery {
        forward_headers: validate_forward_headers("discovery", &forward_headers)?,
        max_response_bytes,
        timeout: Duration::from_millis(timeout_ms),
    })
}

/// Rejects versions that the selected profile does not recognize.
fn validate_versions(
    profile: ProtocolProfile,
    supported_versions: &[String],
    default_version: &str,
) -> Result<(), FilterError> {
    if supported_versions.is_empty() {
        return Err("mcp: supported_versions must not be empty".into());
    }

    for v in supported_versions {
        if !protocol::SUPPORTED_VERSIONS.contains(&v.as_str()) {
            return Err(
                format!("mcp: supported_versions contains '{v}' which is not implemented by this build").into(),
            );
        }
        if !protocol::supported_versions_for_profile(profile).contains(&v.as_str()) {
            return Err(format!(
                "mcp: version '{v}' is not compatible with protocol_profile '{}'",
                profile.as_str()
            )
            .into());
        }
    }

    if !supported_versions.iter().any(|v| v == default_version) {
        return Err(format!("mcp: default_version '{default_version}' must appear in supported_versions",).into());
    }

    Ok(())
}

/// Validate that all server names are unique and non-empty.
fn validate_unique_server_names(servers: &[McpServerConfig]) -> Result<(), FilterError> {
    let mut seen = std::collections::HashSet::new();
    for server in servers {
        if server.name.is_empty() {
            return Err("mcp: server name must not be empty".into());
        }
        if !seen.insert(&server.name) {
            return Err(format!("mcp: duplicate server name: '{}'", server.name).into());
        }
    }
    Ok(())
}

/// Validate that all cluster names are non-empty.
pub(super) fn validate_server_clusters(servers: &[McpServerConfig]) -> Result<(), FilterError> {
    for server in servers {
        if server.cluster.is_empty() {
            return Err(format!("mcp: server '{}' cluster must not be empty", server.name).into());
        }
    }
    Ok(())
}

/// Validate server backend paths against runtime rewrite constraints.
pub(super) fn validate_server_paths(servers: &[McpServerConfig]) -> Result<(), FilterError> {
    for server in servers {
        validate_path(&format!("server '{}'", server.name), &server.path)?;
    }
    Ok(())
}

/// Shared path validator for both the public MCP path and backend
/// server paths. Rejects scheme/authority, missing leading `/`, double
/// leading `/`, traversal segments (including percent-encoded), and
/// values that fail [`http::Uri`] parsing.
fn validate_path(label: &str, path: &str) -> Result<(), FilterError> {
    if path.contains("://") {
        return Err(format!("mcp: {label} path must not contain a scheme/authority: '{path}'").into());
    }
    if !path.starts_with('/') {
        return Err(format!("mcp: {label} path must start with /: '{path}'").into());
    }
    if path.starts_with("//") {
        return Err(format!("mcp: {label} path must not start with //: '{path}'").into());
    }

    let uri: http::Uri = path
        .parse()
        .map_err(|e| FilterError::from(format!("mcp: {label} path is not a valid URI: '{path}': {e}")))?;

    if uri.scheme().is_some() || uri.authority().is_some() {
        return Err(format!("mcp: {label} path must not contain a scheme/authority: '{path}'").into());
    }

    if uri.query().is_some() {
        return Err(format!("mcp: {label} path must not contain a query string: '{path}'").into());
    }

    if has_dot_dot_traversal(uri.path()) {
        return Err(format!("mcp: {label} path contains '..' traversal: '{path}'").into());
    }
    Ok(())
}

/// Validate that all tool names are non-empty.
fn validate_tool_names(servers: &[McpServerConfig]) -> Result<(), FilterError> {
    for server in servers {
        for tool in &server.tools {
            if tool.name.is_empty() {
                return Err(format!("mcp: server '{}' has a tool with an empty name", server.name).into());
            }
        }
    }
    Ok(())
}

/// Validate that no two tools produce the same exposed name after prefixing.
fn validate_unique_exposed_names(catalog: &[CatalogTool]) -> Result<(), FilterError> {
    let mut seen = std::collections::HashSet::new();
    for tool in catalog {
        if !seen.insert(&tool.exposed_name) {
            return Err(format!("mcp: duplicate exposed tool name: '{}'", tool.exposed_name).into());
        }
    }
    Ok(())
}

/// Build the static tool catalog from configured servers.
fn build_catalog(servers: &[McpServerConfig], policy: InvalidToolPolicy) -> Result<Vec<CatalogTool>, FilterError> {
    let mut catalog = Vec::new();
    for (server_index, server) in servers.iter().enumerate() {
        for tool in server.tools.iter().filter(|tool| !tool.hidden) {
            if let Err(reason) = validate_tool_schemas(tool) {
                match policy {
                    InvalidToolPolicy::RejectServer => {
                        return Err(format!("mcp: server '{}' tool '{}' {reason}", server.name, tool.name,).into());
                    },
                    InvalidToolPolicy::FilterOut => {
                        tracing::debug!(
                            server = %server.name,
                            tool = %tool.name,
                            reason = %reason,
                            "excluding tool with non-object schema"
                        );
                        continue;
                    },
                }
            }

            catalog.push(build_catalog_entry(server_index, server, tool));
        }
    }
    Ok(catalog)
}

/// MCP tools accept object-shaped input parameters.
fn validate_tool_schemas(tool: &ToolConfig) -> Result<(), String> {
    if let Some(schema) = &tool.input_schema {
        validate_schema_object("inputSchema", schema)?;
    }
    Ok(())
}

/// Tool schemas without `type: object` confuse clients that validate calls.
pub(super) fn validate_schema_object(label: &str, schema: &serde_json::Value) -> Result<(), String> {
    if !schema.is_object() {
        return Err(format!("{label} must be a JSON object"));
    }
    if schema.get("type").and_then(serde_json::Value::as_str) != Some("object") {
        return Err(format!("{label}.type must be 'object'"));
    }
    if let Some(properties) = schema.get("properties")
        && !properties.is_object()
    {
        return Err(format!("{label}.properties must be a JSON object"));
    }
    if let Some(required) = schema.get("required")
        && !required
            .as_array()
            .is_some_and(|values| values.iter().all(serde_json::Value::is_string))
    {
        return Err(format!("{label}.required must be an array of strings"));
    }
    Ok(())
}

/// A missing configured schema means the tool declares no structured args.
fn default_input_schema() -> serde_json::Value {
    serde_json::json!({ "type": "object", "additionalProperties": false })
}

/// Routing fields stay with catalog entries so follow-up `tools/call`
/// routing can select the backend without reparsing config.
fn build_catalog_entry(server_index: usize, server: &McpServerConfig, tool: &ToolConfig) -> CatalogTool {
    let exposed_name = if let Some(prefix) = &server.tool_prefix {
        format!("{prefix}{}", tool.name)
    } else {
        tool.name.clone()
    };

    CatalogTool {
        annotations: tool.annotations.clone(),
        backend_path: server.path.clone(),
        cluster: server.cluster.clone(),
        description: tool.description.clone(),
        exposed_name,
        extra: serde_json::Map::new(),
        input_schema: tool.input_schema.clone().unwrap_or_else(default_input_schema),
        original_name: tool.name.clone(),
        server_index,
        server_name: server.name.clone(),
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::needless_raw_strings,
    clippy::needless_raw_string_hashes,
    reason = "tests"
)]
mod tests {
    use super::*;

    #[test]
    fn cache_scope_default_is_public() {
        assert_eq!(CacheScope::default(), CacheScope::Public);
    }

    #[test]
    fn cache_scope_as_str() {
        assert_eq!(CacheScope::Public.as_str(), "public");
        assert_eq!(CacheScope::Private.as_str(), "private");
    }

    #[test]
    fn cache_scope_deserializes_from_yaml() {
        assert_eq!(
            serde_yaml::from_str::<CacheScope>("public").unwrap(),
            CacheScope::Public,
        );
        assert_eq!(
            serde_yaml::from_str::<CacheScope>("private").unwrap(),
            CacheScope::Private,
        );
    }

    #[test]
    fn cache_scope_rejects_unknown_value() {
        assert!(serde_yaml::from_str::<CacheScope>("shared").is_err());
    }

    #[test]
    fn invalid_tool_policy_default_is_reject_server() {
        assert_eq!(InvalidToolPolicy::default(), InvalidToolPolicy::RejectServer);
    }

    #[test]
    fn invalid_tool_policy_deserializes() {
        assert_eq!(
            serde_yaml::from_str::<InvalidToolPolicy>("reject_server").unwrap(),
            InvalidToolPolicy::RejectServer,
        );
        assert_eq!(
            serde_yaml::from_str::<InvalidToolPolicy>("filter_out").unwrap(),
            InvalidToolPolicy::FilterOut,
        );
    }

    #[test]
    fn tool_config_minimal() {
        let tc: ToolConfig = serde_yaml::from_str("name: foo").unwrap();
        assert_eq!(tc.name, "foo");
        assert!(tc.description.is_none());
        assert!(tc.input_schema.is_none());
        assert!(tc.annotations.is_none());
    }

    #[test]
    fn tool_config_input_schema_alias() {
        let yaml = r#"
name: t
schema:
  type: object
"#;
        let tc: ToolConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(tc.input_schema.is_some());
        assert_eq!(tc.input_schema.unwrap()["type"], "object");
    }

    #[test]
    fn tool_config_rejects_unknown_fields() {
        let yaml = "name: t\nunknown_field: 42";
        assert!(serde_yaml::from_str::<ToolConfig>(yaml).is_err());
    }

    #[test]
    fn server_config_defaults() {
        let yaml = "name: s\ncluster: c";
        let sc: McpServerConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(sc.path, "/mcp", "path should default to /mcp");
        assert!(sc.tool_prefix.is_none());
        assert!(sc.tools.is_empty(), "tools should default to empty vec");
    }

    #[test]
    fn server_config_rejects_unknown_fields() {
        let yaml = "name: s\ncluster: c\nbogus: true";
        assert!(serde_yaml::from_str::<McpServerConfig>(yaml).is_err());
    }

    #[test]
    fn broker_config_minimal_deserializes() {
        let cfg: McpBrokerConfig = serde_yaml::from_str("{}").unwrap();
        assert_eq!(cfg.path, "/mcp");
        assert_eq!(cfg.max_body_bytes, DEFAULT_MAX_BODY_BYTES);
        assert_eq!(cfg.protocol_profile, ProtocolProfile::Current);
        assert_eq!(cfg.invalid_tool_policy, InvalidToolPolicy::RejectServer);
        assert!(cfg.servers.is_empty());
        assert!(cfg.cache_scope.is_none());
        assert!(cfg.cache_ttl_ms.is_none());
        assert!(cfg.default_version.is_none());
        assert!(cfg.supported_versions.is_none());
    }

    #[test]
    fn broker_config_rejects_unknown_fields() {
        assert!(serde_yaml::from_str::<McpBrokerConfig>("bogus: true").is_err());
    }

    #[test]
    fn schema_properties_must_be_object() {
        let schema = serde_json::json!({"type": "object", "properties": "not-an-object"});
        let err = validate_schema_object("inputSchema", &schema).unwrap_err();
        assert!(err.contains("properties must be a JSON object"), "{err}");
    }

    #[test]
    fn schema_required_must_be_string_array() {
        let schema = serde_json::json!({"type": "object", "required": [1, 2]});
        let err = validate_schema_object("inputSchema", &schema).unwrap_err();
        assert!(err.contains("required must be an array of strings"), "{err}");
    }

    #[test]
    fn schema_required_rejects_mixed_types() {
        let schema = serde_json::json!({"type": "object", "required": ["a", 1]});
        let err = validate_schema_object("inputSchema", &schema).unwrap_err();
        assert!(err.contains("required must be an array of strings"), "{err}");
    }

    #[test]
    fn schema_required_accepts_valid_string_array() {
        let schema = serde_json::json!({"type": "object", "required": ["a", "b"]});
        assert!(validate_schema_object("inputSchema", &schema).is_ok());
    }

    #[test]
    fn schema_not_an_object_value() {
        let schema = serde_json::json!("just a string");
        let err = validate_schema_object("inputSchema", &schema).unwrap_err();
        assert!(err.contains("must be a JSON object"), "{err}");
    }

    #[test]
    fn schema_array_value() {
        let schema = serde_json::json!([1, 2, 3]);
        let err = validate_schema_object("inputSchema", &schema).unwrap_err();
        assert!(err.contains("must be a JSON object"), "{err}");
    }

    #[test]
    fn schema_missing_type_field() {
        let schema = serde_json::json!({"properties": {}});
        let err = validate_schema_object("inputSchema", &schema).unwrap_err();
        assert!(err.contains("type must be 'object'"), "{err}");
    }

    #[test]
    fn schema_valid_minimal() {
        let schema = serde_json::json!({"type": "object"});
        assert!(validate_schema_object("inputSchema", &schema).is_ok());
    }

    #[test]
    fn schema_valid_with_properties_and_required() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        });
        assert!(validate_schema_object("inputSchema", &schema).is_ok());
    }

    #[test]
    fn schema_properties_object_accepted() {
        let schema = serde_json::json!({"type": "object", "properties": {}});
        assert!(validate_schema_object("inputSchema", &schema).is_ok());
    }

    #[test]
    fn schema_required_non_array_rejected() {
        let schema = serde_json::json!({"type": "object", "required": "not-array"});
        let err = validate_schema_object("inputSchema", &schema).unwrap_err();
        assert!(err.contains("required must be an array of strings"), "{err}");
    }

    #[test]
    fn tool_with_no_schema_is_valid() {
        let tool = ToolConfig {
            hidden: false,
            name: "t".into(),
            description: None,
            input_schema: None,
            annotations: None,
        };
        assert!(validate_tool_schemas(&tool).is_ok());
    }

    #[test]
    fn tool_with_valid_schema_is_valid() {
        let tool = ToolConfig {
            hidden: false,
            name: "t".into(),
            description: None,
            input_schema: Some(serde_json::json!({"type": "object"})),
            annotations: None,
        };
        assert!(validate_tool_schemas(&tool).is_ok());
    }

    #[test]
    fn tool_with_invalid_schema_returns_error() {
        let tool = ToolConfig {
            hidden: false,
            name: "t".into(),
            description: None,
            input_schema: Some(serde_json::json!("not-object")),
            annotations: None,
        };
        assert!(validate_tool_schemas(&tool).is_err());
    }

    #[test]
    fn catalog_entry_without_prefix() {
        let server = McpServerConfig {
            name: "srv".into(),
            cluster: "c".into(),
            path: "/backend".into(),
            discover: false,
            tool_prefix: None,
            tool_policy: None,
            tools: vec![],
        };
        let tool = ToolConfig {
            hidden: false,
            name: "my_tool".into(),
            description: Some("desc".into()),
            input_schema: Some(serde_json::json!({"type": "object"})),
            annotations: Some(serde_json::json!({"readOnly": true})),
        };
        let entry = build_catalog_entry(0, &server, &tool);
        assert_eq!(entry.exposed_name, "my_tool");
        assert_eq!(entry.original_name, "my_tool");
        assert_eq!(entry.server_name, "srv");
        assert_eq!(entry.cluster, "c");
        assert_eq!(entry.backend_path, "/backend");
        assert_eq!(entry.description.as_deref(), Some("desc"));
        assert_eq!(entry.input_schema, serde_json::json!({"type": "object"}));
        assert_eq!(entry.annotations, Some(serde_json::json!({"readOnly": true})));
    }

    #[test]
    fn catalog_entry_with_prefix() {
        let server = McpServerConfig {
            name: "srv".into(),
            cluster: "c".into(),
            path: "/mcp".into(),
            discover: false,
            tool_prefix: Some("ns_".into()),
            tool_policy: None,
            tools: vec![],
        };
        let tool = ToolConfig {
            hidden: false,
            name: "action".into(),
            description: None,
            input_schema: None,
            annotations: None,
        };
        let entry = build_catalog_entry(0, &server, &tool);
        assert_eq!(entry.exposed_name, "ns_action");
        assert_eq!(entry.original_name, "action");
        assert!(entry.description.is_none());
        assert!(entry.annotations.is_none());
        assert_eq!(
            entry.input_schema,
            serde_json::json!({"type": "object", "additionalProperties": false}),
        );
    }

    #[test]
    fn default_schema_is_closed_object() {
        let schema = default_input_schema();
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["additionalProperties"], false);
    }

    #[test]
    fn path_valid_simple() {
        assert!(validate_path("test", "/mcp").is_ok());
    }

    #[test]
    fn path_valid_nested() {
        assert!(validate_path("test", "/a/b/c").is_ok());
    }

    #[test]
    fn path_rejects_scheme() {
        let err = validate_path("test", "http://example.com/mcp").unwrap_err();
        assert!(err.to_string().contains("scheme/authority"));
    }

    #[test]
    fn path_rejects_no_leading_slash() {
        let err = validate_path("test", "mcp").unwrap_err();
        assert!(err.to_string().contains("must start with /"));
    }

    #[test]
    fn path_rejects_double_leading_slash() {
        let err = validate_path("test", "//evil").unwrap_err();
        assert!(err.to_string().contains("must not start with //"));
    }

    #[test]
    fn path_rejects_query_string() {
        let err = validate_path("test", "/mcp?x=1").unwrap_err();
        assert!(err.to_string().contains("query string"));
    }

    #[test]
    fn path_rejects_traversal() {
        let err = validate_path("test", "/a/../b").unwrap_err();
        assert!(err.to_string().contains("traversal"));
    }

    #[test]
    fn path_rejects_invalid_uri() {
        let err = validate_path("test", "/bad path").unwrap_err();
        assert!(err.to_string().contains("not a valid URI"));
    }

    #[test]
    fn unique_server_names_pass() {
        let servers = vec![
            McpServerConfig {
                name: "a".into(),
                cluster: "c".into(),
                path: "/mcp".into(),
                discover: false,
                tool_prefix: None,
                tool_policy: None,
                tools: vec![],
            },
            McpServerConfig {
                name: "b".into(),
                cluster: "c".into(),
                path: "/mcp".into(),
                discover: false,
                tool_prefix: None,
                tool_policy: None,
                tools: vec![],
            },
        ];
        assert!(validate_unique_server_names(&servers).is_ok());
    }

    #[test]
    fn duplicate_server_names_fail() {
        let servers = vec![
            McpServerConfig {
                name: "dup".into(),
                cluster: "c".into(),
                path: "/mcp".into(),
                discover: false,
                tool_prefix: None,
                tool_policy: None,
                tools: vec![],
            },
            McpServerConfig {
                name: "dup".into(),
                cluster: "c2".into(),
                path: "/mcp".into(),
                discover: false,
                tool_prefix: None,
                tool_policy: None,
                tools: vec![],
            },
        ];
        let err = validate_unique_server_names(&servers).unwrap_err();
        assert!(err.to_string().contains("duplicate server name"));
    }

    #[test]
    fn empty_server_name_fails() {
        let servers = vec![McpServerConfig {
            name: String::new(),
            cluster: "c".into(),
            path: "/mcp".into(),
            discover: false,
            tool_prefix: None,
            tool_policy: None,
            tools: vec![],
        }];
        let err = validate_unique_server_names(&servers).unwrap_err();
        assert!(err.to_string().contains("must not be empty"));
    }

    #[test]
    fn no_servers_passes() {
        assert!(validate_unique_server_names(&[]).is_ok());
    }

    #[test]
    fn empty_cluster_name_fails() {
        let servers = vec![McpServerConfig {
            name: "s".into(),
            cluster: String::new(),
            path: "/mcp".into(),
            discover: false,
            tool_prefix: None,
            tool_policy: None,
            tools: vec![],
        }];
        let err = validate_server_clusters(&servers).unwrap_err();
        assert!(err.to_string().contains("cluster must not be empty"));
    }

    #[test]
    fn empty_tool_name_fails() {
        let servers = vec![McpServerConfig {
            name: "s".into(),
            cluster: "c".into(),
            path: "/mcp".into(),
            discover: false,
            tool_prefix: None,
            tool_policy: None,
            tools: vec![ToolConfig {
                name: String::new(),
                description: None,
                input_schema: None,
                annotations: None,
                hidden: false,
            }],
        }];
        let err = validate_tool_names(&servers).unwrap_err();
        assert!(err.to_string().contains("empty name"));
    }

    #[test]
    fn valid_tool_names_pass() {
        let servers = vec![McpServerConfig {
            name: "s".into(),
            cluster: "c".into(),
            path: "/mcp".into(),
            discover: false,
            tool_prefix: None,
            tool_policy: None,
            tools: vec![
                ToolConfig {
                    name: "a".into(),
                    description: None,
                    input_schema: None,
                    annotations: None,
                    hidden: false,
                },
                ToolConfig {
                    name: "b".into(),
                    description: None,
                    input_schema: None,
                    annotations: None,
                    hidden: false,
                },
            ],
        }];
        assert!(validate_tool_names(&servers).is_ok());
    }

    #[test]
    fn duplicate_exposed_names_fail() {
        let catalog = vec![
            CatalogTool {
                annotations: None,
                backend_path: "/mcp".into(),
                cluster: "c".into(),
                description: None,
                exposed_name: "dup".into(),
                extra: serde_json::Map::new(),
                input_schema: serde_json::json!({"type": "object"}),
                original_name: "a".into(),
                server_index: 0,
                server_name: "s1".into(),
            },
            CatalogTool {
                annotations: None,
                backend_path: "/mcp".into(),
                cluster: "c2".into(),
                description: None,
                exposed_name: "dup".into(),
                extra: serde_json::Map::new(),
                input_schema: serde_json::json!({"type": "object"}),
                original_name: "b".into(),
                server_index: 0,
                server_name: "s2".into(),
            },
        ];
        let err = validate_unique_exposed_names(&catalog).unwrap_err();
        assert!(err.to_string().contains("duplicate exposed tool name"));
    }

    #[test]
    fn unique_exposed_names_pass() {
        let catalog = vec![CatalogTool {
            annotations: None,
            backend_path: "/mcp".into(),
            cluster: "c".into(),
            description: None,
            exposed_name: "unique".into(),
            extra: serde_json::Map::new(),
            input_schema: serde_json::json!({"type": "object"}),
            original_name: "x".into(),
            server_index: 0,
            server_name: "s".into(),
        }];
        assert!(validate_unique_exposed_names(&catalog).is_ok());
    }

    #[test]
    fn empty_catalog_passes() {
        assert!(validate_unique_exposed_names(&[]).is_ok());
    }

    #[test]
    fn current_profile_rejects_cache_scope() {
        let cfg: McpBrokerConfig = serde_yaml::from_str("cache_scope: public").unwrap();
        let err = validate_cache_fields_for_profile(ProtocolProfile::Current, &cfg).unwrap_err();
        assert!(err.to_string().contains("cache_scope requires"));
    }

    #[test]
    fn current_profile_rejects_cache_ttl() {
        let cfg: McpBrokerConfig = serde_yaml::from_str("cache_ttl_ms: 1000").unwrap();
        let err = validate_cache_fields_for_profile(ProtocolProfile::Current, &cfg).unwrap_err();
        assert!(err.to_string().contains("cache_ttl_ms requires"));
    }

    #[test]
    fn stateless_profile_accepts_cache_fields() {
        let cfg: McpBrokerConfig = serde_yaml::from_str("cache_scope: private\ncache_ttl_ms: 60000").unwrap();
        assert!(validate_cache_fields_for_profile(ProtocolProfile::Stateless, &cfg).is_ok());
    }

    #[test]
    fn current_profile_no_cache_fields_passes() {
        let cfg: McpBrokerConfig = serde_yaml::from_str("{}").unwrap();
        assert!(validate_cache_fields_for_profile(ProtocolProfile::Current, &cfg).is_ok());
    }

    #[test]
    fn validate_versions_empty_supported_rejected() {
        let err = validate_versions(ProtocolProfile::Current, &[], "2025-03-26").unwrap_err();
        assert!(err.to_string().contains("must not be empty"));
    }

    #[test]
    fn validate_versions_unimplemented_version_rejected() {
        let versions = vec!["9999-12-31".to_owned()];
        let err = validate_versions(ProtocolProfile::Current, &versions, "9999-12-31").unwrap_err();
        assert!(err.to_string().contains("not implemented"));
    }

    #[test]
    fn validate_versions_profile_incompatible_rejected() {
        let versions = vec![protocol::PROTOCOL_VERSION_STATELESS_2026_07_28.to_owned()];
        let err = validate_versions(
            ProtocolProfile::Current,
            &versions,
            protocol::PROTOCOL_VERSION_STATELESS_2026_07_28,
        )
        .unwrap_err();
        assert!(err.to_string().contains("not compatible"));
    }

    #[test]
    fn validate_versions_default_not_in_list_rejected() {
        let versions = vec![protocol::PROTOCOL_VERSION_CURRENT.to_owned()];
        let err = validate_versions(ProtocolProfile::Current, &versions, "other").unwrap_err();
        assert!(err.to_string().contains("must appear in supported_versions"));
    }

    #[test]
    fn validate_versions_valid_current() {
        let versions = vec![protocol::PROTOCOL_VERSION_CURRENT.to_owned()];
        assert!(validate_versions(ProtocolProfile::Current, &versions, protocol::PROTOCOL_VERSION_CURRENT).is_ok());
    }

    #[test]
    fn validate_versions_valid_stateless() {
        let versions = vec![protocol::PROTOCOL_VERSION_STATELESS_2026_07_28.to_owned()];
        assert!(
            validate_versions(
                ProtocolProfile::Stateless,
                &versions,
                protocol::PROTOCOL_VERSION_STATELESS_2026_07_28,
            )
            .is_ok()
        );
    }

    #[test]
    fn build_config_minimal() {
        let cfg: McpBrokerConfig = serde_yaml::from_str("{}").unwrap();
        let (validated, catalog) = build_config(cfg).unwrap();
        assert_eq!(validated.path, "/mcp");
        assert_eq!(validated.max_body_bytes, DEFAULT_MAX_BODY_BYTES);
        assert_eq!(validated.protocol_profile, ProtocolProfile::Current);
        assert_eq!(validated.cache_scope, CacheScope::Public);
        assert_eq!(validated.cache_ttl_ms, DEFAULT_CACHE_TTL_MS);
        assert!(catalog.is_empty());
    }

    #[test]
    fn build_config_stateless_with_cache() {
        let yaml = r#"
protocol_profile: stateless
cache_scope: private
cache_ttl_ms: 120000
"#;
        let cfg: McpBrokerConfig = serde_yaml::from_str(yaml).unwrap();
        let (validated, _) = build_config(cfg).unwrap();
        assert_eq!(validated.cache_scope, CacheScope::Private);
        assert_eq!(validated.cache_ttl_ms, 120_000);
    }

    #[test]
    fn build_config_with_tools_and_prefix() {
        let yaml = r#"
servers:
  - name: srv
    cluster: c
    tool_prefix: "p_"
    tools:
      - name: action
        description: do something
        inputSchema:
          type: object
          properties:
            x:
              type: string
          required: ["x"]
"#;
        let cfg: McpBrokerConfig = serde_yaml::from_str(yaml).unwrap();
        let (_, catalog) = build_config(cfg).unwrap();
        assert_eq!(catalog.len(), 1);
        assert_eq!(catalog[0].exposed_name, "p_action");
        assert_eq!(catalog[0].original_name, "action");
        assert_eq!(catalog[0].description.as_deref(), Some("do something"));
    }

    #[test]
    fn build_config_filter_out_bad_schema_keeps_good_tools() {
        let yaml = r#"
invalid_tool_policy: filter_out
servers:
  - name: srv
    cluster: c
    tools:
      - name: bad
        inputSchema:
          type: string
      - name: good
        inputSchema:
          type: object
"#;
        let cfg: McpBrokerConfig = serde_yaml::from_str(yaml).unwrap();
        let (_, catalog) = build_config(cfg).unwrap();
        assert_eq!(catalog.len(), 1);
        assert_eq!(catalog[0].exposed_name, "good");
    }

    #[test]
    fn build_config_reject_server_bad_schema() {
        let yaml = r#"
servers:
  - name: srv
    cluster: c
    tools:
      - name: bad
        inputSchema:
          type: array
"#;
        let cfg: McpBrokerConfig = serde_yaml::from_str(yaml).unwrap();
        let err = build_config(cfg).unwrap_err();
        assert!(err.to_string().contains("type must be 'object'"));
    }
}
