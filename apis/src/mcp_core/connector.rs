// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Operator connector configuration shared by MCP resolution filters.

use std::collections::HashSet;

use praxis_filter::FilterError;
use serde::Deserialize;
use url::Url;

/// Maximum number of connectors allowed in filter configuration.
const MAX_CONNECTORS: usize = 64;
/// Maximum length in bytes for a connector ID.
pub(crate) const MAX_CONNECTOR_ID_LEN: usize = 128;
/// Maximum length in bytes for a connector server URL.
const MAX_CONNECTOR_URL_LEN: usize = 2048;

// -----------------------------------------------------------------------------
// ConnectorConfig
// -----------------------------------------------------------------------------

/// Configuration for a named MCP connector.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ConnectorConfig {
    /// Connector identifier referenced in requests.
    pub id: String,
    /// MCP server URL for this connector.
    pub server_url: String,
    /// Operator policy for tools from this connector. Defaults to `deny` and is
    /// enforced in protected mode: `allow_all` admits every tool, while
    /// `read_only` admits only tools the service affirmatively classifies as
    /// read-only. This classification does not itself prevent side effects;
    /// enforce write restrictions at the service or with scoped credentials.
    #[serde(default)]
    pub tool_policy: ConnectorToolPolicyConfig,
}

/// Operator policy for tools supplied by one connector. Defaults to `deny`.
#[derive(Debug, Default, Clone, Copy, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ConnectorToolPolicyConfig {
    /// Exclude the connector without contacting it. This is the default.
    #[default]
    Deny,
    /// Admit all tools from the connector, including tools that write.
    AllowAll,
    /// Admit only tools affirmatively classified as read-only by the connector.
    ReadOnly,
}

impl ConnectorToolPolicyConfig {
    /// Snapshot as the request-scoped admission policy.
    pub(crate) fn snapshot(self) -> super::admission::ConnectorToolPolicy {
        use super::admission::ConnectorToolPolicy;
        match self {
            Self::Deny => ConnectorToolPolicy::Deny,
            Self::AllowAll => ConnectorToolPolicy::AllowAll,
            Self::ReadOnly => ConnectorToolPolicy::ReadOnly,
        }
    }
}

/// Operator admission mode for MCP tool metadata.
#[derive(Debug, Default, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ToolAdmissionMode {
    /// Standard advisory behavior; unchanged. This is the default.
    #[default]
    Advisory,
    /// Operator-enforced protection.
    Protected,
}

/// Operator metadata-admission policy for MCP tool resolution.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ToolAdmissionConfig {
    /// Admission mode (`advisory` default, or `protected`).
    #[serde(default)]
    pub mode: ToolAdmissionMode,
}

impl ToolAdmissionConfig {
    /// Snapshot as the request-scoped admission policy type.
    pub(crate) fn snapshot(&self) -> super::admission::McpAdmissionPolicy {
        super::admission::McpAdmissionPolicy {
            mode: match self.mode {
                ToolAdmissionMode::Advisory => super::admission::McpAdmissionMode::Advisory,
                ToolAdmissionMode::Protected => super::admission::McpAdmissionMode::Protected,
            },
        }
    }
}

/// Validate connector configuration entries.
pub(crate) fn validate_connectors(filter: &str, connectors: &[ConnectorConfig]) -> Result<(), FilterError> {
    if connectors.len() > MAX_CONNECTORS {
        return Err(format!(
            "{filter}: too many connectors: {} exceeds limit of {MAX_CONNECTORS}",
            connectors.len()
        )
        .into());
    }
    let mut seen_ids = HashSet::new();
    for connector in connectors {
        validate_single_connector(filter, connector)?;
        if !seen_ids.insert(&connector.id) {
            return Err(format!("{filter}: duplicate connector id \"{}\"", connector.id).into());
        }
    }
    Ok(())
}

/// Validate a single connector's ID and URL.
fn validate_single_connector(filter: &str, connector: &ConnectorConfig) -> Result<(), FilterError> {
    validate_connector_id(filter, &connector.id)?;
    validate_connector_server_url(filter, &connector.server_url, &connector.id)
}

/// Validate connector ID is non-empty and within length limit.
fn validate_connector_id(filter: &str, id: &str) -> Result<(), FilterError> {
    if id.is_empty() {
        return Err(format!("{filter}: connector id must not be empty").into());
    }
    if id.len() > MAX_CONNECTOR_ID_LEN {
        return Err(format!("{filter}: connector id \"{id}\" exceeds {MAX_CONNECTOR_ID_LEN} bytes").into());
    }
    Ok(())
}

/// Validate connector server URL string and parse it.
fn validate_connector_server_url(filter: &str, server_url: &str, id: &str) -> Result<(), FilterError> {
    if server_url.is_empty() {
        return Err(format!("{filter}: connector \"{id}\" has empty server_url").into());
    }
    if server_url.len() > MAX_CONNECTOR_URL_LEN {
        return Err(format!("{filter}: connector \"{id}\" server_url exceeds {MAX_CONNECTOR_URL_LEN} bytes").into());
    }
    let url = Url::parse(server_url)
        .map_err(|e| FilterError::from(format!("{filter}: connector \"{id}\" has invalid server_url: {e}")))?;
    validate_connector_url(filter, &url, id)
}

/// Validate a connector URL's scheme, host, and credentials.
fn validate_connector_url(filter: &str, url: &Url, connector_id: &str) -> Result<(), FilterError> {
    match url.scheme() {
        "http" | "https" => {},
        scheme => {
            return Err(format!(
                "{filter}: connector \"{connector_id}\" server_url must use http or https, got \"{scheme}\""
            )
            .into());
        },
    }
    if url.host().is_none() {
        return Err(format!("{filter}: connector \"{connector_id}\" server_url has no host").into());
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err(format!("{filter}: connector \"{connector_id}\" server_url must not contain credentials").into());
    }
    Ok(())
}
