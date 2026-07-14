// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Anthropic Messages to Chat Completions-compatible transformation filter.
//!
//! Rewrites Anthropic Messages request bodies to the Chat Completions
//! request shape and transforms compatible non-streaming responses back.
//! Streaming SSE transformation is handled by the separate
//! `anthropic_stream_events` filter.
//!
//! The filter name preserves the proposal/config surface. `OpenAI` here
//! means the Chat Completions wire shape, not the Responses API or
//! OpenAI-only backends.

mod config;
pub(crate) mod request;
pub(crate) mod response;

use async_trait::async_trait;
use bytes::Bytes;
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, Rejection, parse_filter_config,
};
use tracing::{debug, warn};

use self::config::{AnthropicToOpenaiConfig, build_config};

// -----------------------------------------------------------------------------
// AnthropicToOpenaiFilter
// -----------------------------------------------------------------------------

/// Transforms Anthropic Messages API requests to Chat Completions-compatible
/// request bodies and transforms compatible responses back. The filter name
/// refers to the OpenAI Chat Completions wire shape, not the Responses API;
/// non-OpenAI compatible backends are valid targets.
///
/// # YAML
///
/// ```yaml
/// filter: anthropic_to_openai
/// ```
///
/// # Full YAML
///
/// ```yaml
/// filter: anthropic_to_openai
/// max_body_bytes: 1048576
/// ```
pub struct AnthropicToOpenaiFilter {
    /// Parsed and validated configuration.
    config: AnthropicToOpenaiConfig,
}

impl AnthropicToOpenaiFilter {
    /// Create a filter from parsed YAML config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the YAML config is invalid.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: AnthropicToOpenaiConfig = parse_filter_config("anthropic_to_openai", config)?;
        let validated = build_config(cfg)?;
        Ok(Box::new(Self { config: validated }))
    }
}

#[async_trait]
impl HttpFilter for AnthropicToOpenaiFilter {
    fn name(&self) -> &'static str {
        "anthropic_to_openai"
    }

    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadWrite
    }

    fn request_body_mode(&self) -> BodyMode {
        BodyMode::StreamBuffer {
            max_bytes: Some(self.config.max_body_bytes),
        }
    }

    fn response_body_access(&self) -> BodyAccess {
        BodyAccess::ReadWrite
    }

    fn response_body_mode(&self) -> BodyMode {
        BodyMode::Stream
    }

    async fn on_response(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        if should_transform_response(ctx) {
            ctx.set_response_body_mode(BodyMode::StreamBuffer {
                max_bytes: Some(self.config.max_body_bytes),
            });
            if let Some(resp) = &mut ctx.response_header {
                resp.headers.remove(http::header::CONTENT_LENGTH);
                ctx.response_headers_modified = true;
            }
        }

        Ok(FilterAction::Continue)
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        ctx.request_headers_to_remove
            .push(http::header::HeaderName::from_static("anthropic-version"));
        ctx.request_headers_to_remove
            .push(http::header::HeaderName::from_static("x-api-key"));

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

        let bytes = match body.as_ref() {
            Some(b) if !b.is_empty() => b.as_ref(),
            _ => return Ok(FilterAction::Continue),
        };

        extract_request_metadata(ctx, bytes);
        Ok(transform_request_body(body))
    }

    fn on_response_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if !should_transform_response(ctx) {
            return Ok(FilterAction::Continue);
        }

        if !end_of_stream {
            return Ok(FilterAction::Continue);
        }

        let request_model = ctx
            .filter_metadata
            .get("anthropic_to_openai.model")
            .cloned()
            .unwrap_or_default();

        transform_non_streaming_body(ctx, body, &request_model);

        if let Some(b) = body.as_ref()
            && let Some(resp) = &mut ctx.response_header
        {
            resp.headers
                .insert(http::header::CONTENT_LENGTH, http::HeaderValue::from(b.len()));
            resp.headers.insert(
                http::header::CONTENT_TYPE,
                http::HeaderValue::from_static("application/json"),
            );
            ctx.response_headers_modified = true;
        }

        Ok(FilterAction::Continue)
    }
}

// -----------------------------------------------------------------------------
// Request Body Helpers
// -----------------------------------------------------------------------------

/// Extract streaming and model metadata from the request body.
fn extract_request_metadata(ctx: &mut HttpFilterContext<'_>, bytes: &[u8]) {
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(bytes) else {
        ctx.set_metadata("anthropic_to_openai.streaming", "false");
        return;
    };

    let is_streaming = value
        .get("stream")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    ctx.set_metadata(
        "anthropic_to_openai.streaming",
        if is_streaming { "true" } else { "false" },
    );

    let model = value
        .get("model")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .unwrap_or_default();
    ctx.set_metadata("anthropic_to_openai.model", model);
}

/// Transform the request body and return the appropriate filter action.
fn transform_request_body(body: &mut Option<Bytes>) -> FilterAction {
    let Some(bytes) = body.as_ref() else {
        return FilterAction::Continue;
    };

    match request::transform_request(bytes) {
        Ok(transformed) => {
            debug!(
                original_len = bytes.len(),
                transformed_len = transformed.len(),
                "transformed Anthropic request to Chat Completions-compatible format"
            );
            *body = Some(Bytes::from(transformed));
            FilterAction::Continue
        },
        Err(msg) => {
            warn!(error = msg.as_str(), "failed to transform Anthropic request");
            FilterAction::Reject(
                Rejection::status(400)
                    .with_header("content-type", "application/json")
                    .with_body(Bytes::from(format!(
                        r#"{{"error":{{"message":"{msg}","type":"invalid_request_error"}}}}"#
                    ))),
            )
        },
    }
}

// -----------------------------------------------------------------------------
// Response Body Helpers
// -----------------------------------------------------------------------------

/// Return true when the response should be buffered and transformed.
fn should_transform_response(ctx: &HttpFilterContext<'_>) -> bool {
    let is_streaming = ctx
        .filter_metadata
        .get("anthropic_to_openai.streaming")
        .is_some_and(|v| v == "true");
    let is_success = ctx.response_header.as_ref().is_none_or(|r| r.status.is_success());

    !is_streaming && is_success
}

/// Apply non-streaming JSON transformation to the response body.
fn transform_non_streaming_body(ctx: &mut HttpFilterContext<'_>, body: &mut Option<Bytes>, request_model: &str) {
    let bytes = match body.as_ref() {
        Some(b) => b.as_ref(),
        None => return,
    };

    if bytes.is_empty() {
        return;
    }

    match response::transform_response(bytes, request_model) {
        Ok(result) => {
            debug!(
                original_len = bytes.len(),
                transformed_len = result.body.len(),
                original_finish_reason = result.original_finish_reason.as_str(),
                "transformed Chat Completions-compatible response to Anthropic"
            );
            ctx.set_metadata("openai.finish_reason", result.original_finish_reason);
            *body = Some(Bytes::from(result.body));
        },
        Err(msg) => {
            warn!(
                error = msg.as_str(),
                "failed to transform Chat Completions-compatible response"
            );
        },
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::unwrap_used, clippy::indexing_slicing, reason = "tests")]
mod tests {
    use bytes::Bytes;
    use http::{Method, StatusCode};

    use super::*;
    use crate::test_utils::{make_filter_context, make_request, make_response};

    #[test]
    fn default_config_parses() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("{}").unwrap();
        let filter = AnthropicToOpenaiFilter::from_config(&yaml).unwrap();

        assert_eq!(filter.name(), "anthropic_to_openai", "filter name should match");
    }

    #[test]
    fn unknown_config_field_rejected() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("strip_unsupported: true").unwrap();
        let result = AnthropicToOpenaiFilter::from_config(&yaml);

        assert!(result.is_err(), "unknown config fields should be rejected");
    }

    #[test]
    fn zero_max_body_bytes_rejected() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("max_body_bytes: 0").unwrap();
        let result = AnthropicToOpenaiFilter::from_config(&yaml);

        assert!(result.is_err(), "zero max_body_bytes should be rejected");
    }

    #[test]
    fn rejects_max_body_bytes_above_ceiling() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("max_body_bytes: 67108865").unwrap();
        let result = AnthropicToOpenaiFilter::from_config(&yaml);

        assert!(
            result.is_err(),
            "max_body_bytes above 64 MiB ceiling should be rejected"
        );
    }

    #[test]
    fn non_success_response_body_is_not_transformed() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("{}").unwrap();
        let filter = AnthropicToOpenaiFilter::from_config(&yaml).unwrap();
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);
        let mut response = make_response();
        response.status = StatusCode::BAD_REQUEST;
        ctx.response_header = Some(&mut response);
        ctx.set_metadata("anthropic_to_openai.streaming", "false");
        ctx.set_metadata("anthropic_to_openai.model", "gpt-4");
        let original = Bytes::from_static(br#"{"error":{"message":"bad request","type":"invalid_request_error"}}"#);
        let mut body = Some(original.clone());

        let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();

        assert!(matches!(action, FilterAction::Continue), "filter should continue");
        assert_eq!(body, Some(original), "upstream error body should pass through");
    }

    // --- extract_request_metadata ---

    #[test]
    fn extract_request_metadata_streaming_true_with_model() {
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);
        let bytes = br#"{"stream":true,"model":"claude-opus-4-8"}"#;

        extract_request_metadata(&mut ctx, bytes);

        assert_eq!(
            ctx.filter_metadata.get("anthropic_to_openai.streaming").unwrap(),
            "true"
        );
        assert_eq!(
            ctx.filter_metadata.get("anthropic_to_openai.model").unwrap(),
            "claude-opus-4-8"
        );
    }

    #[test]
    fn extract_request_metadata_streaming_false() {
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);
        let bytes = br#"{"stream":false,"model":"gpt-4"}"#;

        extract_request_metadata(&mut ctx, bytes);

        assert_eq!(
            ctx.filter_metadata.get("anthropic_to_openai.streaming").unwrap(),
            "false"
        );
        assert_eq!(ctx.filter_metadata.get("anthropic_to_openai.model").unwrap(), "gpt-4");
    }

    #[test]
    fn extract_request_metadata_invalid_json() {
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);

        extract_request_metadata(&mut ctx, b"not json");

        assert_eq!(
            ctx.filter_metadata.get("anthropic_to_openai.streaming").unwrap(),
            "false",
            "invalid JSON should default streaming to false"
        );
        assert!(
            !ctx.filter_metadata.contains_key("anthropic_to_openai.model"),
            "invalid JSON should not set model"
        );
    }

    // --- transform_request_body ---

    #[test]
    fn transform_request_body_none_continues() {
        let mut body: Option<Bytes> = None;
        let action = transform_request_body(&mut body);

        assert!(matches!(action, FilterAction::Continue));
        assert!(body.is_none());
    }

    #[test]
    fn transform_request_body_valid_transforms() {
        let mut body = Some(Bytes::from(
            br#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"role":"user","content":"Hi"}]}"#.to_vec(),
        ));
        let action = transform_request_body(&mut body);

        assert!(matches!(action, FilterAction::Continue));
        assert!(body.is_some());
        let parsed: serde_json::Value = serde_json::from_slice(body.unwrap().as_ref()).unwrap();
        assert_eq!(parsed["messages"][0]["role"], "user");
        assert_eq!(
            parsed["max_completion_tokens"], 1024,
            "max_tokens should be mapped to max_completion_tokens"
        );
    }

    #[test]
    fn transform_request_body_invalid_rejects() {
        let mut body = Some(Bytes::from_static(b"not json"));
        let action = transform_request_body(&mut body);

        assert!(
            matches!(action, FilterAction::Reject(_)),
            "invalid body should produce a rejection"
        );
    }

    // --- should_transform_response ---

    #[test]
    fn should_transform_response_streaming_returns_false() {
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);
        ctx.set_metadata("anthropic_to_openai.streaming", "true");
        let mut response = make_response();
        ctx.response_header = Some(&mut response);

        assert!(
            !should_transform_response(&ctx),
            "streaming responses should not be transformed"
        );
    }

    #[test]
    fn should_transform_response_non_streaming_success() {
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);
        ctx.set_metadata("anthropic_to_openai.streaming", "false");
        let mut response = make_response();
        ctx.response_header = Some(&mut response);

        assert!(
            should_transform_response(&ctx),
            "non-streaming success should be transformed"
        );
    }

    // --- transform_non_streaming_body ---

    #[test]
    fn transform_non_streaming_body_none_is_noop() {
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);
        let mut body: Option<Bytes> = None;

        transform_non_streaming_body(&mut ctx, &mut body, "gpt-4");

        assert!(body.is_none());
    }

    #[test]
    fn transform_non_streaming_body_empty_bytes_is_noop() {
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);
        let mut body = Some(Bytes::new());

        transform_non_streaming_body(&mut ctx, &mut body, "gpt-4");

        assert_eq!(body.as_ref().unwrap().len(), 0, "empty bytes should not be transformed");
    }

    #[test]
    fn transform_non_streaming_body_success() {
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);
        let response_json = br#"{"id":"chatcmpl-1","model":"gpt-4","choices":[{"message":{"role":"assistant","content":"Hello!"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5}}"#;
        let mut body = Some(Bytes::from(response_json.to_vec()));

        transform_non_streaming_body(&mut ctx, &mut body, "gpt-4");

        assert!(body.is_some());
        let parsed: serde_json::Value = serde_json::from_slice(body.unwrap().as_ref()).unwrap();
        assert_eq!(parsed["type"], "message");
        assert_eq!(parsed["content"][0]["text"], "Hello!");
        assert_eq!(
            ctx.filter_metadata.get("openai.finish_reason").unwrap(),
            "stop",
            "finish_reason should be stored in metadata"
        );
    }

    #[test]
    fn transform_non_streaming_body_invalid_json_preserves_body() {
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);
        let original = Bytes::from_static(b"not json");
        let mut body = Some(original.clone());

        transform_non_streaming_body(&mut ctx, &mut body, "gpt-4");

        assert_eq!(body, Some(original), "body should not be modified on error");
    }
}
