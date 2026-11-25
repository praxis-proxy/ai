// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Anthropic Messages protocol filter for native `/v1/messages` backends.
//!
//! Supplies a gateway-managed `anthropic-version` default for
//! internal or non-SDK callers while preserving caller-provided
//! `anthropic-version` and `anthropic-beta` headers across iterative
//! requests. Does not touch credentials or the request or response
//! body.
//!
//! Also refuses the Anthropic MCP connector (`mcp_servers` /
//! `mcp_toolset`) when the selected cluster declares an
//! `application_provider` other than `anthropic`: only Anthropic executes
//! the connector, so forwarding it elsewhere cannot succeed. The fact comes
//! from `anthropic_messages_request`, so this filter still reads no body.

mod config;

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing, reason = "tests")]
mod tests;

use std::borrow::Cow;

use async_trait::async_trait;
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, IterationState, parse_filter_config,
};
use tracing::debug;

use self::config::{AnthropicMessagesProtocolConfig, build_config};
use super::{
    messages_request::{MCP_CONNECTOR_METADATA_KEY, MCP_CONNECTOR_RESOLVED},
    wire,
};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Anthropic API version header name.
const ANTHROPIC_VERSION_HEADER: &str = "anthropic-version";

/// Anthropic beta feature header name.
const ANTHROPIC_BETA_HEADER: &str = "anthropic-beta";

/// `anthropic-beta` token prefix that negotiates the MCP connector.
const MCP_CLIENT_BETA_PREFIX: &str = "mcp-client-";

/// The `application_provider` that executes the MCP connector natively.
const ANTHROPIC_PROVIDER: &str = "anthropic";

/// Client-facing reason for refusing the MCP connector.
const MCP_CONNECTOR_UNSUPPORTED: &str =
    "the MCP connector (`mcp_servers` and `mcp_toolset` tools) is not supported by the selected backend";

// -----------------------------------------------------------------------------
// AnthropicMessagesProtocolFilter
// -----------------------------------------------------------------------------

/// Normalizes Anthropic Messages protocol headers for native
/// backends.
///
/// `default_version` applies only when the caller sends no
/// `anthropic-version`; a caller-supplied version, such as the `2026-10-21`
/// default of Anthropic SDK v2, is forwarded unchanged. The filter does not
/// validate version-specific body rules, so under `2026-10-21` the backend
/// owns rejecting `temperature`, `top_p`, `top_k`, assistant `document`
/// blocks, and non-empty citations on user text blocks. It does not touch
/// credentials: remove `x-api-key` and `authorization` separately when the
/// gateway owns the upstream credential.
///
/// A request that uses the MCP connector, as published by
/// `anthropic_messages_request`, is rejected with an Anthropic
/// `invalid_request_error` when the selected cluster declares an
/// `application_provider` other than `anthropic`. Place this filter after
/// `load_balancer` for the check to see the selected cluster; an untagged
/// cluster, or a placement before selection, forwards the request unchanged.
///
/// # YAML
///
/// ```yaml
/// filter: anthropic_messages_protocol
/// ```
///
/// # Full YAML
///
/// ```yaml
/// filter: anthropic_messages_protocol
/// default_version: "2023-06-01"
/// ```
pub struct AnthropicMessagesProtocolFilter {
    /// Parsed and validated configuration.
    config: AnthropicMessagesProtocolConfig,
}

impl AnthropicMessagesProtocolFilter {
    /// Create a filter from parsed YAML config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the YAML config is invalid.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: AnthropicMessagesProtocolConfig = parse_filter_config("anthropic_messages_protocol", config)?;
        let validated = build_config(cfg)?;
        Ok(Box::new(Self { config: validated }))
    }
}

#[async_trait]
impl HttpFilter for AnthropicMessagesProtocolFilter {
    fn name(&self) -> &'static str {
        "anthropic_messages_protocol"
    }

    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::None
    }

    fn request_body_mode(&self) -> BodyMode {
        BodyMode::Stream
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        if let Some(provider) = ctx.selected_application_provider()
            && provider != ANTHROPIC_PROVIDER
            && ctx.filter_metadata.get(MCP_CONNECTOR_METADATA_KEY) == Some("true")
        {
            debug!(provider, "rejecting MCP connector for a non-Anthropic provider");
            return Ok(FilterAction::Reject(wire::invalid_request_rejection(
                MCP_CONNECTOR_UNSUPPORTED,
            )));
        }

        // A translated route may already own the version through a queued set.
        // Do not append a native default that would override that contract.
        let has_version = ctx.request.headers.get(ANTHROPIC_VERSION_HEADER).is_some()
            || ctx
                .request_headers_to_set
                .iter()
                .any(|(name, _)| name == ANTHROPIC_VERSION_HEADER);

        if !has_version && !restore_original_header(ctx, ANTHROPIC_VERSION_HEADER) {
            debug!(
                version = self.config.default_version.as_str(),
                "injecting default anthropic-version header"
            );
            ctx.extra_request_headers.push((
                Cow::Borrowed(ANTHROPIC_VERSION_HEADER),
                self.config.default_version.clone(),
            ));
        }

        if ctx.request.headers.get(ANTHROPIC_BETA_HEADER).is_none() {
            let connector_resolved =
                ctx.filter_metadata.get(MCP_CONNECTOR_METADATA_KEY) == Some(MCP_CONNECTOR_RESOLVED);
            restore_beta_header(ctx, connector_resolved);
        }

        Ok(FilterAction::Continue)
    }

    async fn on_response(&self, _ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        Ok(FilterAction::Continue)
    }
}

/// Restore the caller's `anthropic-beta` header on an iterative request.
///
/// Once Praxis has resolved the MCP connector the backend no longer receives
/// it, so the connector's beta tokens are dropped from the restored value.
fn restore_beta_header(ctx: &mut HttpFilterContext<'_>, connector_resolved: bool) {
    let Some(value) = ctx
        .extensions
        .get::<IterationState>()
        .and_then(|state| state.original_request.headers.get(ANTHROPIC_BETA_HEADER))
        .and_then(|value| value.to_str().ok())
    else {
        return;
    };
    let value = match connector_resolved.then(|| strip_mcp_client_betas(value)).flatten() {
        Some(stripped) if stripped.is_empty() => return,
        Some(stripped) => stripped,
        None => value.to_owned(),
    };
    ctx.extra_request_headers
        .push((Cow::Borrowed(ANTHROPIC_BETA_HEADER), value));
}

/// Remove MCP connector tokens from an `anthropic-beta` value.
///
/// Returns `None` when the value carries no MCP connector token. Otherwise
/// returns the remaining comma-separated tokens, which may be empty.
pub(crate) fn strip_mcp_client_betas(value: &str) -> Option<String> {
    let mut stripped = false;
    let kept: Vec<&str> = value
        .split(',')
        .map(str::trim)
        .filter(|token| {
            let mcp = token.starts_with(MCP_CLIENT_BETA_PREFIX);
            stripped |= mcp;
            !mcp && !token.is_empty()
        })
        .collect();
    stripped.then(|| kept.join(","))
}

/// Restore a caller-supplied Anthropic protocol header when an
/// iterative request starts with a fresh header map.
fn restore_original_header(ctx: &mut HttpFilterContext<'_>, name: &'static str) -> bool {
    let Some(value) = ctx
        .extensions
        .get::<IterationState>()
        .and_then(|state| state.original_request.headers.get(name))
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned)
    else {
        return false;
    };

    ctx.extra_request_headers.push((Cow::Borrowed(name), value));
    true
}
