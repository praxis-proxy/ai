// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Configuration for the `openai_operation` classifier.

use praxis_filter::FilterError;
use serde::Deserialize;

/// Default header carrying the classified API family.
pub(crate) const DEFAULT_FAMILY_HEADER: &str = "x-praxis-ai-family";

/// Default header carrying the classified operation ID.
pub(crate) const DEFAULT_OPERATION_HEADER: &str = "x-praxis-ai-operation";

/// Configurable header names for the classified operation.
///
/// Both headers are proxy-owned. A configured name is always either
/// overwritten with the classifier's own value or removed, so a client
/// cannot supply one and influence routing.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct OperationHeaders {
    /// Header name for the API family. `null` disables the header.
    #[serde(default = "default_family_header")]
    pub family: Option<String>,

    /// Header name for the operation ID. `null` disables the header.
    #[serde(default = "default_operation_header")]
    pub operation: Option<String>,
}

impl Default for OperationHeaders {
    fn default() -> Self {
        Self {
            family: default_family_header(),
            operation: default_operation_header(),
        }
    }
}

/// Default family header name.
#[expect(
    clippy::unnecessary_wraps,
    reason = "serde default for an Option field must produce an Option"
)]
fn default_family_header() -> Option<String> {
    Some(DEFAULT_FAMILY_HEADER.to_owned())
}

/// Default operation header name.
#[expect(
    clippy::unnecessary_wraps,
    reason = "serde default for an Option field must produce an Option"
)]
fn default_operation_header() -> Option<String> {
    Some(DEFAULT_OPERATION_HEADER.to_owned())
}

/// Parsed `openai_operation` configuration.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct OperationClassifierConfig {
    /// Header names for the classified operation.
    #[serde(default)]
    pub headers: OperationHeaders,
}

/// Validated configuration with parsed header names.
#[derive(Debug, Clone)]
pub(crate) struct ValidatedConfig {
    /// Parsed family header name, when enabled.
    pub family_header: Option<http::HeaderName>,

    /// Parsed operation header name, when enabled.
    pub operation_header: Option<http::HeaderName>,
}

/// Validate configuration and pre-parse header names.
///
/// Header names are parsed once at startup so the request path never
/// re-validates them.
///
/// # Errors
///
/// Returns [`FilterError`] when a configured header name is not a valid
/// HTTP header name.
pub(crate) fn build_config(config: &OperationClassifierConfig) -> Result<ValidatedConfig, FilterError> {
    Ok(ValidatedConfig {
        family_header: parse_header(config.headers.family.as_deref(), "headers.family")?,
        operation_header: parse_header(config.headers.operation.as_deref(), "headers.operation")?,
    })
}

/// Parse one optional header name, naming the offending field on failure.
fn parse_header(value: Option<&str>, field: &str) -> Result<Option<http::HeaderName>, FilterError> {
    value
        .map(|name| {
            http::HeaderName::try_from(name)
                .map_err(|_ignored| FilterError::from(format!("openai_operation: {field} is not a valid header name")))
        })
        .transpose()
}
