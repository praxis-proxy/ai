// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Anthropic Messages to Chat Completions-compatible transformation filter.
//!
//! Rewrites Anthropic Messages request bodies to the Chat Completions
//! request shape, transforms compatible non-streaming successes back, and
//! normalizes pre-stream upstream errors for both request modes. Successful
//! streaming SSE transformation is handled by the separate
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
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, parse_filter_config,
};
use tracing::{debug, warn};

use self::config::{AnthropicToOpenaiConfig, build_config};
use crate::anthropic::wire;

/// Metadata key selecting success or error response transformation.
const RESPONSE_TRANSFORM_KEY: &str = "anthropic_to_openai.response_transform";
/// Response transform marker for a successful response.
const RESPONSE_TRANSFORM_SUCCESS: &str = "success";
/// Response transform marker for an upstream error.
const RESPONSE_TRANSFORM_ERROR: &str = "error";
/// Metadata key preserving the upstream error status for the body phase.
const RESPONSE_STATUS_KEY: &str = "anthropic_to_openai.response_status";
/// Metadata key preserving the upstream request ID for the body phase.
const RESPONSE_REQUEST_ID_KEY: &str = "anthropic_to_openai.response_request_id";

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
        let request_id = canonicalize_response_request_id(ctx);
        let Some(transform) = response_transform(ctx) else {
            return Ok(FilterAction::Continue);
        };

        ctx.set_metadata(RESPONSE_TRANSFORM_KEY, transform);
        if transform == RESPONSE_TRANSFORM_ERROR {
            let status = ctx
                .response_header
                .as_ref()
                .map_or(500, |response| response.status.as_u16());
            ctx.set_metadata(RESPONSE_STATUS_KEY, status.to_string());
        }
        if let Some(request_id) = request_id {
            ctx.set_metadata(RESPONSE_REQUEST_ID_KEY, request_id);
        }

        ctx.set_response_body_mode(BodyMode::StreamBuffer {
            max_bytes: Some(self.config.max_body_bytes),
        });
        prepare_transformed_response_headers(ctx);

        Ok(FilterAction::Continue)
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        ctx.request_headers_to_remove
            .push(http::header::HeaderName::from_static("anthropic-version"));
        ctx.request_headers_to_remove
            .push(http::header::HeaderName::from_static("x-api-key"));
        ctx.request_headers_to_remove.push(http::header::ACCEPT_ENCODING);

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
        let transform_error = match ctx.get_metadata(RESPONSE_TRANSFORM_KEY) {
            Some(RESPONSE_TRANSFORM_ERROR) => true,
            Some(RESPONSE_TRANSFORM_SUCCESS) => false,
            _ => return Ok(FilterAction::Continue),
        };

        if !end_of_stream {
            return Ok(FilterAction::Continue);
        }

        if transform_error {
            let status = ctx
                .get_metadata(RESPONSE_STATUS_KEY)
                .and_then(|value| value.parse::<u16>().ok())
                .and_then(|value| http::StatusCode::from_u16(value).ok())
                .unwrap_or(http::StatusCode::INTERNAL_SERVER_ERROR);
            let request_id = ctx.get_metadata(RESPONSE_REQUEST_ID_KEY);
            transform_error_body(body, status, request_id);
        } else {
            let request_model = ctx
                .filter_metadata
                .get("anthropic_to_openai.model")
                .map_or("", String::as_str);
            let request_id = ctx.get_metadata(RESPONSE_REQUEST_ID_KEY);
            if let Some(finish_reason) = transform_non_streaming_body(body, request_model, request_id) {
                ctx.set_metadata("openai.finish_reason", finish_reason);
            }
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
            FilterAction::Reject(wire::invalid_request_rejection(&msg))
        },
    }
}

// -----------------------------------------------------------------------------
// Response Body Helpers
// -----------------------------------------------------------------------------

/// Remove stale representation metadata before replacing a response body.
fn prepare_transformed_response_headers(ctx: &mut HttpFilterContext<'_>) {
    if let Some(resp) = &mut ctx.response_header {
        resp.headers.remove(http::header::CONTENT_LENGTH);
        resp.headers.remove(http::header::CONTENT_ENCODING);
        resp.headers.remove(http::header::CONTENT_RANGE);
        resp.headers.remove(http::header::ETAG);
        for header in ["content-digest", "content-md5", "digest", "repr-digest"] {
            resp.headers.remove(header);
        }
        resp.headers.insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("application/json"),
        );
        ctx.response_headers_modified = true;
    }
}

/// Expose the upstream request ID through Anthropic's canonical header.
fn canonicalize_response_request_id(ctx: &mut HttpFilterContext<'_>) -> Option<String> {
    let request_id = ctx.response_header.as_ref().and_then(|response| {
        response
            .headers
            .get("request-id")
            .or_else(|| response.headers.get("x-request-id"))
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned)
    });
    if let Some(request_id) = request_id.as_deref()
        && let Some(response) = &mut ctx.response_header
        && let Ok(value) = http::HeaderValue::from_str(request_id)
    {
        response.headers.insert("request-id", value);
        ctx.response_headers_modified = true;
    }
    request_id
}

/// Return true when the response should be buffered and transformed.
#[cfg(test)]
fn should_transform_response(ctx: &HttpFilterContext<'_>) -> bool {
    response_transform(ctx).is_some()
}

/// Select the response transformation while headers are available.
fn response_transform(ctx: &HttpFilterContext<'_>) -> Option<&'static str> {
    let is_streaming = ctx
        .filter_metadata
        .get("anthropic_to_openai.streaming")
        .is_some_and(|v| v == "true");
    let status = ctx.response_header.as_ref().map(|response| response.status);
    let is_error = status.is_some_and(|status| status.is_client_error() || status.is_server_error());
    let is_complete_success = status.is_none_or(|status| status == http::StatusCode::OK)
        && ctx.response_header.as_ref().is_none_or(|response| {
            !response.headers.contains_key(http::header::CONTENT_ENCODING)
                && !response.headers.contains_key(http::header::CONTENT_RANGE)
                && response
                    .headers
                    .get(http::header::CONTENT_TYPE)
                    .and_then(|value| value.to_str().ok())
                    .is_none_or(|value| {
                        let media_type = value.split(';').next().unwrap_or_default().trim();
                        media_type.eq_ignore_ascii_case("application/json")
                            || media_type.to_ascii_lowercase().ends_with("+json")
                    })
        });

    if is_error {
        Some(RESPONSE_TRANSFORM_ERROR)
    } else if !is_streaming && is_complete_success {
        Some(RESPONSE_TRANSFORM_SUCCESS)
    } else {
        None
    }
}

/// Normalize a buffered upstream error response.
fn transform_error_body(body: &mut Option<Bytes>, status: http::StatusCode, request_id: Option<&str>) {
    let original = body.as_deref().unwrap_or_default();
    let transformed = response::transform_error_response(original, status, request_id);

    *body = Some(Bytes::from(transformed));
}

/// Apply non-streaming JSON transformation to the response body.
fn transform_non_streaming_body(
    body: &mut Option<Bytes>,
    request_model: &str,
    request_id: Option<&str>,
) -> Option<String> {
    match response::transform_response(body.as_deref().unwrap_or_default(), request_model) {
        Ok(result) => {
            debug!(
                original_len = body.as_ref().map_or(0, Bytes::len),
                transformed_len = result.body.len(),
                original_finish_reason = result.original_finish_reason.as_str(),
                "transformed Chat Completions-compatible response to Anthropic"
            );
            *body = Some(Bytes::from(result.body));
            Some(result.original_finish_reason)
        },
        Err(msg) => {
            warn!(
                error = msg.as_str(),
                "failed to transform Chat Completions-compatible response"
            );
            *body = Some(Bytes::from(wire::error_body(
                "api_error",
                "upstream response could not be transformed",
                request_id,
            )));
            None
        },
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::too_many_lines,
    reason = "tests"
)]
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

    #[tokio::test]
    async fn error_response_state_survives_body_phase_without_headers() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("{}").unwrap();
        let filter = AnthropicToOpenaiFilter::from_config(&yaml).unwrap();
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);
        let mut response = make_response();
        response.status = StatusCode::SERVICE_UNAVAILABLE;
        response.headers.insert("x-request-id", "req_header".parse().unwrap());
        response
            .headers
            .insert(http::header::CONTENT_LENGTH, http::HeaderValue::from_static("72"));
        ctx.response_header = Some(&mut response);
        ctx.set_metadata("anthropic_to_openai.streaming", "true");
        ctx.set_metadata("anthropic_to_openai.model", "gpt-4");
        let action = filter.on_response(&mut ctx).await.unwrap();

        assert!(matches!(action, FilterAction::Continue), "filter should continue");
        assert!(
            ctx.response_header
                .as_ref()
                .is_some_and(|response| !response.headers.contains_key(http::header::CONTENT_LENGTH)),
            "buffered error should remove content-length during the header phase"
        );
        assert_eq!(
            ctx.response_header
                .as_ref()
                .and_then(|response| response.headers.get("request-id"))
                .and_then(|value| value.to_str().ok()),
            Some("req_header"),
            "OpenAI request IDs should be exposed through Anthropic's response header"
        );
        ctx.response_header = None;

        let mut body = Some(Bytes::from_static(
            br#"{"error":{"message":"unavailable","type":"server_error"}}"#,
        ));
        let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(body.as_deref().unwrap()).unwrap();

        assert!(matches!(action, FilterAction::Continue), "filter should continue");
        assert_eq!(parsed["type"], "error");
        assert_eq!(parsed["error"]["type"], "api_error");
        assert_eq!(parsed["error"]["message"], "unavailable");
        assert_eq!(parsed["request_id"], "req_header");
    }

    #[tokio::test]
    async fn rewritten_errors_remove_stale_representation_headers() {
        for content_encoding in ["gzip", "br"] {
            let yaml: serde_yaml::Value = serde_yaml::from_str("max_body_bytes: 4096").unwrap();
            let filter = AnthropicToOpenaiFilter::from_config(&yaml).unwrap();
            let request = make_request(Method::POST, "/v1/messages");
            let mut ctx = make_filter_context(&request);
            let mut response = make_response();
            response.status = StatusCode::BAD_REQUEST;
            response
                .headers
                .insert(http::header::CONTENT_TYPE, http::HeaderValue::from_static("text/plain"));
            response.headers.insert(
                http::header::CONTENT_ENCODING,
                http::HeaderValue::from_str(content_encoding).unwrap(),
            );
            response.headers.insert(
                http::header::CONTENT_RANGE,
                http::HeaderValue::from_static("bytes 0-41/42"),
            );
            response
                .headers
                .insert(http::header::ETAG, http::HeaderValue::from_static("\"upstream\""));
            response
                .headers
                .insert("content-digest", http::HeaderValue::from_static("sha-256=:abc:"));
            ctx.response_header = Some(&mut response);

            drop(filter.on_response(&mut ctx).await.unwrap());

            assert!(
                matches!(ctx.response_body_mode, BodyMode::StreamBuffer { max_bytes: Some(4096) }),
                "rewritten errors should use the configured buffer limit"
            );
            assert_eq!(
                ctx.response_header
                    .as_ref()
                    .and_then(|response| response.headers.get(http::header::CONTENT_TYPE))
                    .and_then(|value| value.to_str().ok()),
                Some("application/json"),
                "rewritten errors should advertise JSON"
            );
            for header in [
                http::header::CONTENT_ENCODING,
                http::header::CONTENT_RANGE,
                http::header::ETAG,
                http::HeaderName::from_static("content-digest"),
            ] {
                assert!(
                    ctx.response_header
                        .as_ref()
                        .is_some_and(|response| !response.headers.contains_key(&header)),
                    "{header} should be removed when rewriting a {content_encoding}-encoded error"
                );
            }
        }
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

    #[tokio::test]
    async fn on_request_prevents_upstream_response_encoding() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("{}").unwrap();
        let filter = AnthropicToOpenaiFilter::from_config(&yaml).unwrap();
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);

        drop(filter.on_request(&mut ctx).await.unwrap());

        assert!(
            ctx.request_headers_to_remove.contains(&http::header::ACCEPT_ENCODING),
            "response transformation requires an unencoded upstream representation"
        );
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

        let FilterAction::Reject(rejection) = action else {
            panic!("invalid body should produce a rejection");
        };
        let parsed: serde_json::Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();

        assert_eq!(parsed["type"], "error");
        assert_eq!(parsed["error"]["type"], "invalid_request_error");
        assert!(parsed.get("request_id").is_some());
        assert!(parsed["request_id"].is_null());
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

    #[test]
    fn should_not_transform_encoded_non_streaming_success() {
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);
        ctx.set_metadata("anthropic_to_openai.streaming", "false");
        let mut response = make_response();
        response
            .headers
            .insert(http::header::CONTENT_ENCODING, http::HeaderValue::from_static("gzip"));
        ctx.response_header = Some(&mut response);

        assert!(
            !should_transform_response(&ctx),
            "encoded success should pass through with its representation headers intact"
        );
    }

    #[tokio::test]
    async fn encoded_non_streaming_success_passes_through_unchanged() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("{}").unwrap();
        let filter = AnthropicToOpenaiFilter::from_config(&yaml).unwrap();
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);
        ctx.set_metadata("anthropic_to_openai.streaming", "false");
        let mut response = make_response();
        response
            .headers
            .insert(http::header::CONTENT_ENCODING, http::HeaderValue::from_static("gzip"));
        ctx.response_header = Some(&mut response);

        drop(filter.on_response(&mut ctx).await.unwrap());

        assert_eq!(ctx.response_body_mode, BodyMode::Stream);
        assert!(
            ctx.response_header
                .as_ref()
                .is_some_and(|response| response.headers.contains_key(http::header::CONTENT_ENCODING))
        );

        let encoded = Bytes::from_static(b"\x1f\x8bencoded-response");
        let mut body = Some(encoded.clone());
        let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();

        assert!(matches!(action, FilterAction::Continue));
        assert_eq!(body, Some(encoded));
    }

    #[tokio::test]
    async fn non_json_success_passes_through_unchanged() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("{}").unwrap();
        let filter = AnthropicToOpenaiFilter::from_config(&yaml).unwrap();
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);
        ctx.set_metadata("anthropic_to_openai.streaming", "false");
        let mut response = make_response();
        response
            .headers
            .insert(http::header::CONTENT_TYPE, http::HeaderValue::from_static("text/plain"));
        ctx.response_header = Some(&mut response);

        drop(filter.on_response(&mut ctx).await.unwrap());

        assert_eq!(ctx.response_body_mode, BodyMode::Stream);
        assert_eq!(
            ctx.response_header
                .as_ref()
                .and_then(|response| response.headers.get(http::header::CONTENT_TYPE))
                .and_then(|value| value.to_str().ok()),
            Some("text/plain")
        );

        let original = Bytes::from_static(b"upstream plaintext");
        let mut body = Some(original.clone());
        drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());
        assert_eq!(body, Some(original));
    }

    #[tokio::test]
    async fn successful_responses_canonicalize_request_id() {
        for is_streaming in ["false", "true"] {
            let yaml: serde_yaml::Value = serde_yaml::from_str("{}").unwrap();
            let filter = AnthropicToOpenaiFilter::from_config(&yaml).unwrap();
            let request = make_request(Method::POST, "/v1/messages");
            let mut ctx = make_filter_context(&request);
            ctx.set_metadata("anthropic_to_openai.streaming", is_streaming);
            let mut response = make_response();
            response.headers.insert("x-request-id", "req_success".parse().unwrap());
            ctx.response_header = Some(&mut response);

            drop(filter.on_response(&mut ctx).await.unwrap());

            assert_eq!(
                ctx.response_header
                    .as_ref()
                    .and_then(|response| response.headers.get("request-id"))
                    .and_then(|value| value.to_str().ok()),
                Some("req_success"),
                "stream={is_streaming}"
            );
        }
    }

    #[test]
    fn should_not_transform_partial_non_streaming_success() {
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);
        ctx.set_metadata("anthropic_to_openai.streaming", "false");
        let mut response = make_response();
        response.status = StatusCode::PARTIAL_CONTENT;
        response.headers.insert(
            http::header::CONTENT_RANGE,
            http::HeaderValue::from_static("bytes 0-99/200"),
        );
        ctx.response_header = Some(&mut response);

        assert!(
            !should_transform_response(&ctx),
            "partial success should pass through with its representation headers intact"
        );
    }

    #[test]
    fn should_transform_response_errors_for_both_request_modes() {
        for is_streaming in ["false", "true"] {
            for status in [StatusCode::BAD_REQUEST, StatusCode::INTERNAL_SERVER_ERROR] {
                let request = make_request(Method::POST, "/v1/messages");
                let mut ctx = make_filter_context(&request);
                ctx.set_metadata("anthropic_to_openai.streaming", is_streaming);
                let mut response = make_response();
                response.status = status;
                ctx.response_header = Some(&mut response);

                assert!(
                    should_transform_response(&ctx),
                    "{status} response should be transformed for stream={is_streaming}"
                );
            }
        }
    }

    #[test]
    fn should_not_transform_redirect_response() {
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);
        ctx.set_metadata("anthropic_to_openai.streaming", "false");
        let mut response = make_response();
        response.status = StatusCode::FOUND;
        ctx.response_header = Some(&mut response);

        assert!(!should_transform_response(&ctx), "redirect should pass through");
    }

    // --- transform_non_streaming_body ---

    #[test]
    fn transform_non_streaming_body_missing_body_returns_api_error() {
        let mut body: Option<Bytes> = None;

        let finish_reason = transform_non_streaming_body(&mut body, "gpt-4", None);
        let parsed: serde_json::Value = serde_json::from_slice(body.as_deref().unwrap()).unwrap();

        assert!(finish_reason.is_none());
        assert_eq!(parsed["type"], "error");
        assert_eq!(parsed["error"]["type"], "api_error");
        assert!(parsed["request_id"].is_null());
    }

    #[test]
    fn transform_non_streaming_body_empty_bytes_returns_api_error() {
        let mut body = Some(Bytes::new());

        let finish_reason = transform_non_streaming_body(&mut body, "gpt-4", None);
        let parsed: serde_json::Value = serde_json::from_slice(body.as_deref().unwrap()).unwrap();

        assert!(finish_reason.is_none());
        assert_eq!(parsed["type"], "error");
        assert_eq!(parsed["error"]["type"], "api_error");
    }

    #[test]
    fn transform_non_streaming_body_success() {
        let response_json = br#"{"id":"chatcmpl-1","model":"gpt-4","choices":[{"message":{"role":"assistant","content":"Hello!"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5}}"#;
        let mut body = Some(Bytes::from(response_json.to_vec()));

        let finish_reason = transform_non_streaming_body(&mut body, "gpt-4", None);

        assert!(body.is_some());
        let parsed: serde_json::Value = serde_json::from_slice(body.unwrap().as_ref()).unwrap();
        assert_eq!(parsed["type"], "message");
        assert_eq!(parsed["content"][0]["text"], "Hello!");
        assert_eq!(finish_reason.as_deref(), Some("stop"));
    }

    #[tokio::test]
    async fn malformed_non_streaming_success_returns_anthropic_api_error() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("{}").unwrap();
        let filter = AnthropicToOpenaiFilter::from_config(&yaml).unwrap();
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);
        ctx.set_metadata("anthropic_to_openai.streaming", "false");
        ctx.set_metadata("anthropic_to_openai.model", "gpt-4");
        let mut response = make_response();
        response
            .headers
            .insert("x-request-id", "req_malformed".parse().unwrap());
        ctx.response_header = Some(&mut response);

        let action = filter.on_response(&mut ctx).await.unwrap();

        assert!(matches!(action, FilterAction::Continue));
        assert!(matches!(ctx.response_body_mode, BodyMode::StreamBuffer { .. }));
        ctx.response_header = None;

        let mut body = Some(Bytes::from_static(b"not json"));
        let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(body.as_deref().unwrap()).unwrap();

        assert!(matches!(action, FilterAction::Continue));
        assert_eq!(parsed["type"], "error");
        assert_eq!(parsed["error"]["type"], "api_error");
        assert_eq!(parsed["error"]["message"], "upstream response could not be transformed");
        assert_eq!(parsed["request_id"], "req_malformed");
        assert!(!ctx.filter_metadata.contains_key("openai.finish_reason"));
    }
}
