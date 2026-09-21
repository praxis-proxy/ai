// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Streaming-selection filter for the filtered-subrequest MCP transport.
//!
//! This plain [`HttpFilter`] is auto-injected as the first entry of the bound
//! MCP `outbound_chain` (see
//! [`bind_mcp_outbound_chain`](super::subrequest_transport::bind_mcp_outbound_chain)).
//! It carries no config and never mutates the body: it declares the streaming
//! subrequest capability and, when the transport armed the callout with
//! [`McpStreamingRequested`], flips the subrequest response mode to
//! [`SubRequestResponseMode::Streaming`] so praxis returns
//! [`CalloutResponse::Streaming`] instead of buffering the SSE body.

use async_trait::async_trait;
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, SubRequestResponseMode,
};

use super::subrequest_transport::McpStreamingRequested;

/// Auto-injected first filter of the MCP outbound chain that selects the
/// streaming subrequest transport when the callout is armed for streaming.
#[derive(Debug, Default)]
pub struct McpStreamingSelectorFilter;

impl McpStreamingSelectorFilter {
    /// Registry name; also the entry the auto-injector prepends.
    pub const NAME: &'static str = "openai_mcp_streaming_selector";

    /// Build the filter. Takes no configuration.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] only if a non-null config mapping is supplied.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        if !config.is_null() && config.as_mapping().is_some_and(|m| !m.is_empty()) {
            return Err(FilterError::from(
                "openai_mcp_streaming_selector takes no configuration".to_owned(),
            ));
        }
        Ok(Box::new(Self))
    }
}

#[async_trait]
impl HttpFilter for McpStreamingSelectorFilter {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    /// Declare that this filter may select the streaming subrequest transport.
    ///
    /// Without this, praxis never asks the pipeline for streaming and always
    /// buffers the callout response.
    fn may_select_streaming_subrequest_response(&self) -> bool {
        true
    }

    /// This filter never terminates the request.
    fn produces_terminal_response(&self) -> bool {
        false
    }

    /// Read-only response body access.
    ///
    /// Load-bearing: with the default (`None`), pipeline finalization's
    /// `clamp_body_mode` downgrades `Stream` to a size-limited buffered mode,
    /// which would re-arm an opaque inner transport limiter and defeat
    /// streaming. `ReadOnly` keeps the response body streamed through untouched.
    fn response_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    /// Stream the response body (do not buffer).
    fn response_body_mode(&self) -> BodyMode {
        BodyMode::Stream
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        if ctx.extensions.get::<McpStreamingRequested>().is_some() {
            ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
        }
        Ok(FilterAction::Continue)
    }
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, reason = "tests use unwrap for brevity")]
mod tests {
    use http::Method;
    use praxis_filter::{BodyAccess, BodyMode, FilterAction, HttpFilter as _, SubRequestResponseMode};

    use super::{McpStreamingRequested, McpStreamingSelectorFilter};
    use crate::test_utils::{make_filter_context, make_request};

    #[test]
    fn selector_declares_readonly_stream_and_streaming_capability() {
        let filter = McpStreamingSelectorFilter;
        assert_eq!(filter.name(), "openai_mcp_streaming_selector");
        assert!(
            filter.may_select_streaming_subrequest_response(),
            "selector must advertise the streaming capability"
        );
        assert!(!filter.produces_terminal_response());
        assert_eq!(
            filter.response_body_access(),
            BodyAccess::ReadOnly,
            "ReadOnly keeps clamp_body_mode from downgrading Stream to a buffered limit"
        );
        assert!(matches!(filter.response_body_mode(), BodyMode::Stream));
    }

    #[tokio::test]
    async fn on_request_sets_streaming_only_with_marker() {
        let filter = McpStreamingSelectorFilter;
        let req = make_request(Method::POST, "/mcp");

        // No marker: mode stays at the default (Buffered).
        let mut ctx = make_filter_context(&req);
        let action = filter.on_request(&mut ctx).await.unwrap();
        assert!(matches!(action, FilterAction::Continue));
        assert_eq!(ctx.subrequest_response_mode(), SubRequestResponseMode::Buffered);

        // Marker present: mode flips to Streaming.
        let mut ctx = make_filter_context(&req);
        ctx.extensions.insert(McpStreamingRequested);
        let action = filter.on_request(&mut ctx).await.unwrap();
        assert!(matches!(action, FilterAction::Continue));
        assert_eq!(ctx.subrequest_response_mode(), SubRequestResponseMode::Streaming);
    }

    #[test]
    fn from_config_rejects_non_empty_mapping() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("foo: bar").unwrap();
        assert!(McpStreamingSelectorFilter::from_config(&yaml).is_err());
    }

    #[test]
    fn from_config_accepts_null_and_empty() {
        assert!(McpStreamingSelectorFilter::from_config(&serde_yaml::Value::Null).is_ok());
        let empty: serde_yaml::Value = serde_yaml::from_str("{}").unwrap();
        assert!(McpStreamingSelectorFilter::from_config(&empty).is_ok());
    }
}
