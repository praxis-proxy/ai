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
#[cfg_attr(not(test), expect(dead_code, reason = "fields used by follow-up tools/call routing"))]
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
            tool_prefix: None,
            tools: vec![],
        };
        let tool = ToolConfig {
            name: "my_tool".into(),
            description: Some("desc".into()),
            input_schema: Some(serde_json::json!({"type": "object"})),
            annotations: Some(serde_json::json!({"readOnly": true})),
        };
        let entry = build_catalog_entry(&server, &tool);
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
            tool_prefix: Some("ns_".into()),
            tools: vec![],
        };
        let tool = ToolConfig {
            name: "action".into(),
            description: None,
            input_schema: None,
            annotations: None,
        };
        let entry = build_catalog_entry(&server, &tool);
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
                tool_prefix: None,
                tools: vec![],
            },
            McpServerConfig {
                name: "b".into(),
                cluster: "c".into(),
                path: "/mcp".into(),
                tool_prefix: None,
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
                tool_prefix: None,
                tools: vec![],
            },
            McpServerConfig {
                name: "dup".into(),
                cluster: "c2".into(),
                path: "/mcp".into(),
                tool_prefix: None,
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
            tool_prefix: None,
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
            tool_prefix: None,
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
            tool_prefix: None,
            tools: vec![ToolConfig {
                name: String::new(),
                description: None,
                input_schema: None,
                annotations: None,
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
            tool_prefix: None,
            tools: vec![
                ToolConfig {
                    name: "a".into(),
                    description: None,
                    input_schema: None,
                    annotations: None,
                },
                ToolConfig {
                    name: "b".into(),
                    description: None,
                    input_schema: None,
                    annotations: None,
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
                input_schema: serde_json::json!({"type": "object"}),
                original_name: "a".into(),
                server_name: "s1".into(),
            },
            CatalogTool {
                annotations: None,
                backend_path: "/mcp".into(),
                cluster: "c2".into(),
                description: None,
                exposed_name: "dup".into(),
                input_schema: serde_json::json!({"type": "object"}),
                original_name: "b".into(),
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
            input_schema: serde_json::json!({"type": "object"}),
            original_name: "x".into(),
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
