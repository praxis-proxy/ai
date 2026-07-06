// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Configuration types for the MCP static catalog filter.

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
pub(super) enum InvalidToolPolicy {
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
    /// Tool prefix for this server.
    pub tool_prefix: Option<String>,
    /// Statically defined tools.
    #[serde(default)]
    pub tools: Vec<ToolConfig>,
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
    /// Cache scope for stateless responses. Requires `protocol_profile: stateless`.
    pub cache_scope: Option<CacheScope>,
    /// Cache TTL in milliseconds for stateless responses. Requires `protocol_profile: stateless`.
    pub cache_ttl_ms: Option<u64>,
    /// Fallback MCP protocol version. When omitted, derived from the profile.
    pub default_version: Option<String>,
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
    /// Cache scope for stateless responses.
    pub cache_scope: CacheScope,
    /// Cache TTL in milliseconds for stateless responses.
    pub cache_ttl_ms: u64,
    /// Fallback MCP protocol version.
    pub default_version: String,
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
#[expect(dead_code, reason = "fields used by follow-up tools/call routing")]
pub(super) struct CatalogTool {
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
    /// MCP input schema.
    pub input_schema: serde_json::Value,
    /// Original tool name on the backend.
    pub original_name: String,
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

// -----------------------------------------------------------------------------
// Validation
// -----------------------------------------------------------------------------

/// Validate configuration and build the static tool catalog.
pub(super) fn build_config(cfg: McpBrokerConfig) -> Result<(ValidatedBrokerConfig, Vec<CatalogTool>), FilterError> {
    validate_max_body_bytes("mcp_broker", cfg.max_body_bytes)?;

    let profile = cfg.protocol_profile;

    validate_cache_fields_for_profile(profile, &cfg)?;

    let catalog = validate_and_build_catalog(&cfg)?;

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
        cache_scope,
        cache_ttl_ms,
        default_version,
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
        if !protocol::is_supported_version(v) {
            return Err(
                format!("mcp: supported_versions contains '{v}' which is not implemented by this build").into(),
            );
        }
        if !protocol::is_supported_version_for_profile(profile, v) {
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
fn validate_server_clusters(servers: &[McpServerConfig]) -> Result<(), FilterError> {
    for server in servers {
        if server.cluster.is_empty() {
            return Err(format!("mcp: server '{}' cluster must not be empty", server.name).into());
        }
    }
    Ok(())
}

/// Validate server backend paths against runtime rewrite constraints.
fn validate_server_paths(servers: &[McpServerConfig]) -> Result<(), FilterError> {
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
    for server in servers {
        for tool in &server.tools {
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

            catalog.push(build_catalog_entry(server, tool));
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
fn validate_schema_object(label: &str, schema: &serde_json::Value) -> Result<(), String> {
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
fn build_catalog_entry(server: &McpServerConfig, tool: &ToolConfig) -> CatalogTool {
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
        input_schema: tool.input_schema.clone().unwrap_or_else(default_input_schema),
        original_name: tool.name.clone(),
        server_name: server.name.clone(),
    }
}
