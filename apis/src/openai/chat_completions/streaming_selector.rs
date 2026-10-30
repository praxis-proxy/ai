// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Selects the IRR streaming transport for OpenAI Chat Completions requests.

use async_trait::async_trait;
use bytes::Bytes;
use http::Method;
use praxis_filter::{
    BodyAccess, BodyMode, EmptyFilterConfig, FilterAction, FilterError, HttpFilter, HttpFilterContext,
    SubRequestResponseMode, body::MAX_JSON_BODY_BYTES, parse_filter_config,
};
use serde::Deserialize;

/// Registered filter identifier.
const FILTER_NAME: &str = "openai_chat_completions_streaming_selector";

/// The subset of the Chat Completions request that selects its response transport.
#[derive(Deserialize)]
struct StreamingRequest {
    /// Whether the client expects an SSE response.
    #[serde(default)]
    stream: bool,
}

/// Selects the typed IRR response transport from an OpenAI Chat Completions
/// request's effective `stream` bit.
///
/// Place this inside every IRR step that may terminate an OpenAI Chat
/// Completions request. It leaves the request body unchanged. Invalid or
/// non-Chat JSON stays on the buffered transport so the selected upstream or a
/// later validator retains ownership of the client-facing error.
///
/// # YAML configuration
///
/// ```yaml
/// filter: openai_chat_completions_streaming_selector
/// ```
pub struct OpenaiChatCompletionsStreamingSelectorFilter;

impl OpenaiChatCompletionsStreamingSelectorFilter {
    /// Build the selector from an empty YAML mapping.
    ///
    /// # Errors
    ///
    /// Returns an error when configuration fields are supplied.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let _: EmptyFilterConfig = parse_filter_config(FILTER_NAME, config)?;
        Ok(Box::new(Self))
    }
}

#[async_trait]
impl HttpFilter for OpenaiChatCompletionsStreamingSelectorFilter {
    fn name(&self) -> &'static str {
        "openai_chat_completions_streaming_selector"
    }

    /// IRR validates streaming-capable steps at pipeline-build time.
    fn may_select_streaming_subrequest_response(&self) -> bool {
        true
    }

    /// Read the complete JSON request before deciding its response transport.
    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    /// The selector reads only the bounded request body and never mutates it.
    fn request_body_mode(&self) -> BodyMode {
        BodyMode::StreamBuffer {
            max_bytes: Some(MAX_JSON_BODY_BYTES),
        }
    }

    /// Keep a selected upstream SSE body progressive through this step.
    fn response_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    /// Do not let the pipeline clamp an explicitly selected SSE response to a buffer.
    fn response_body_mode(&self) -> BodyMode {
        BodyMode::Stream
    }

    async fn on_request(&self, _ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        Ok(FilterAction::Continue)
    }

    async fn on_request_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if !end_of_stream {
            return Ok(FilterAction::Continue);
        }

        let stream = ctx.request.method == Method::POST
            && ctx.request.uri.path() == "/v1/chat/completions"
            && body
                .as_deref()
                .and_then(|bytes| serde_json::from_slice::<StreamingRequest>(bytes).ok())
                .is_some_and(|request| request.stream);
        ctx.set_subrequest_response_mode(if stream {
            SubRequestResponseMode::Streaming
        } else {
            SubRequestResponseMode::Buffered
        });

        Ok(FilterAction::Release)
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "tests use unwrap and expect for concise assertions"
)]
mod tests {
    use bytes::Bytes;
    use http::Method;
    use praxis_filter::{BodyAccess, BodyMode, FilterAction, SubRequestResponseMode};

    use super::OpenaiChatCompletionsStreamingSelectorFilter;

    #[test]
    fn declares_streaming_capability_and_modes() {
        let filter = filter();
        assert!(filter.may_select_streaming_subrequest_response());
        assert_eq!(filter.request_body_access(), BodyAccess::ReadOnly);
        assert!(matches!(filter.request_body_mode(), BodyMode::StreamBuffer { .. }));
        assert_eq!(filter.response_body_access(), BodyAccess::ReadOnly);
        assert!(matches!(filter.response_body_mode(), BodyMode::Stream));
    }

    #[tokio::test]
    async fn selects_streaming_only_for_a_true_stream_field() {
        let request = crate::test_utils::make_request(Method::POST, "/v1/chat/completions");

        let mut streaming_ctx = crate::test_utils::make_filter_context(&request);
        let mut streaming_body = Some(Bytes::from_static(br#"{"stream":true}"#));
        let action = filter()
            .on_request_body(&mut streaming_ctx, &mut streaming_body, true)
            .await
            .unwrap();
        assert!(matches!(action, FilterAction::Release));
        assert_eq!(
            streaming_ctx.subrequest_response_mode(),
            SubRequestResponseMode::Streaming
        );

        for body in [br#"{"stream":false}"#.as_slice(), br#"{}"#.as_slice(), b"not-json"] {
            let mut buffered_ctx = crate::test_utils::make_filter_context(&request);
            let mut buffered_body = Some(Bytes::copy_from_slice(body));
            let action = filter()
                .on_request_body(&mut buffered_ctx, &mut buffered_body, true)
                .await
                .unwrap();
            assert!(
                matches!(action, FilterAction::Release),
                "selector should release buffered body: {}",
                String::from_utf8_lossy(body)
            );
            assert_eq!(
                buffered_ctx.subrequest_response_mode(),
                SubRequestResponseMode::Buffered,
                "selector chose the wrong response mode for body: {}",
                String::from_utf8_lossy(body)
            );
        }

        let non_chat_request = crate::test_utils::make_request(Method::POST, "/v1/responses");
        let mut non_chat_ctx = crate::test_utils::make_filter_context(&non_chat_request);
        let mut non_chat_body = Some(Bytes::from_static(br#"{"stream":true}"#));
        let action = filter()
            .on_request_body(&mut non_chat_ctx, &mut non_chat_body, true)
            .await
            .unwrap();
        assert!(matches!(action, FilterAction::Release));
        assert_eq!(
            non_chat_ctx.subrequest_response_mode(),
            SubRequestResponseMode::Buffered
        );
    }

    #[test]
    fn rejects_configuration() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("unexpected: true").unwrap();
        assert!(OpenaiChatCompletionsStreamingSelectorFilter::from_config(&yaml).is_err());
    }

    fn filter() -> Box<dyn praxis_filter::HttpFilter> {
        OpenaiChatCompletionsStreamingSelectorFilter::from_config(&serde_yaml::Value::Null).unwrap()
    }
}
