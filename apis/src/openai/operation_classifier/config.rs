// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Configuration for the `openai_operation` classifier.

use praxis_filter::FilterError;
use serde::Deserialize;

/// Default header carrying the classified application protocol.
pub(crate) const DEFAULT_APPLICATION_PROTOCOL_HEADER: &str = "x-praxis-ai-application-protocol";

/// Default header carrying the classified operation ID.
pub(crate) const DEFAULT_OPERATION_HEADER: &str = "x-praxis-ai-operation";

/// Header names a classifier target may never use.
///
/// Each of these carries authentication state or HTTP framing that the proxy
/// or upstream depends on, so overwriting or removing one would corrupt the
/// exchange rather than route it. Matched case-insensitively against the
/// parsed [`http::HeaderName`], which is always lowercase.
const FORBIDDEN_HEADER_TARGETS: &[&str] = &[
    "authorization",
    "proxy-authorization",
    "cookie",
    "set-cookie",
    "host",
    "content-length",
    "content-type",
    "transfer-encoding",
    "connection",
    "upgrade",
    "te",
    "trailer",
    "expect",
];

/// Configurable header names for the classified operation.
///
/// Both headers are proxy-owned. A configured name is always either
/// overwritten with the classifier's own value or removed, so a client
/// cannot supply one and influence routing.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct OperationHeaders {
    /// Header name for the application protocol. `null` disables the header.
    #[serde(default = "default_application_protocol_header")]
    pub application_protocol: Option<String>,

    /// Header name for the operation ID. `null` disables the header.
    #[serde(default = "default_operation_header")]
    pub operation: Option<String>,
}

impl Default for OperationHeaders {
    fn default() -> Self {
        Self {
            application_protocol: default_application_protocol_header(),
            operation: default_operation_header(),
        }
    }
}

/// Default application-protocol header name.
#[expect(
    clippy::unnecessary_wraps,
    reason = "serde default for an Option field must produce an Option"
)]
fn default_application_protocol_header() -> Option<String> {
    Some(DEFAULT_APPLICATION_PROTOCOL_HEADER.to_owned())
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
    /// Parsed application-protocol header name, when enabled.
    pub application_protocol_header: Option<http::HeaderName>,

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
/// Returns [`FilterError`] when a configured header name is not a valid HTTP
/// header name, names a header the classifier must not own, or when both
/// outputs target the same header.
pub(crate) fn build_config(config: &OperationClassifierConfig) -> Result<ValidatedConfig, FilterError> {
    let application_protocol_header = parse_header(
        config.headers.application_protocol.as_deref(),
        "headers.application_protocol",
    )?;
    let operation_header = parse_header(config.headers.operation.as_deref(), "headers.operation")?;

    // Both targets are written with set semantics, so sharing a name means the
    // operation value silently replaces the application protocol.
    if let (Some(protocol), Some(operation)) = (&application_protocol_header, &operation_header)
        && protocol == operation
    {
        return Err(format!(
            "openai_operation: headers.application_protocol and headers.operation must differ, both are {protocol:?}"
        )
        .into());
    }

    Ok(ValidatedConfig {
        application_protocol_header,
        operation_header,
    })
}

/// Parse one optional header name, naming the offending field on failure.
///
/// Rejects targets carrying authentication state or HTTP framing, which the
/// classifier overwrites or removes and so must never own.
fn parse_header(value: Option<&str>, field: &str) -> Result<Option<http::HeaderName>, FilterError> {
    value
        .map(|name| {
            let header = http::HeaderName::try_from(name).map_err(|_ignored| {
                FilterError::from(format!("openai_operation: {field} is not a valid header name"))
            })?;
            if FORBIDDEN_HEADER_TARGETS.contains(&header.as_str()) {
                return Err(FilterError::from(format!(
                    "openai_operation: {field} must not target {header:?}, which carries authentication or framing state"
                )));
            }
            Ok(header)
        })
        .transpose()
}
