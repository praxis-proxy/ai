// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Azure OpenAI to Chat Completions-compatible transformation filter.
//!
//! Azure OpenAI speaks the Chat Completions wire format with minor
//! differences: deployment-based paths, `api-version` query parameter,
//! `api-key` authentication header, and extra content-filter fields in
//! responses. This filter normalizes those differences so that standard
//! Chat Completions clients work transparently.
//!
//! Streaming (SSE) responses are parsed per-chunk through the shared
//! [`SseFrameParser`]; each
//! completed frame's `data:` payload is stripped of Azure-specific
//! fields and re-emitted as standard SSE.

mod config;
pub(crate) mod request;
pub(crate) mod response;

use async_trait::async_trait;
use bytes::Bytes;
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, parse_filter_config,
};
use tracing::debug;

use self::config::{AzureToOpenaiConfig, build_config};
use crate::{
    azure::wire,
    openai::sse::{SseFrame, SseFrameParser},
};

/// Metadata key selecting response transformation mode.
const RESPONSE_TRANSFORM_KEY: &str = "azure_to_openai.response_transform";
/// Response transform marker for a successful response.
const RESPONSE_TRANSFORM_SUCCESS: &str = "success";
/// Response transform marker for an upstream error.
const RESPONSE_TRANSFORM_ERROR: &str = "error";
/// Response transform marker for SSE streaming.
const RESPONSE_TRANSFORM_SSE: &str = "sse";
/// Metadata key preserving the upstream error status for the body phase.
const RESPONSE_STATUS_KEY: &str = "azure_to_openai.response_status";

/// Transforms requests targeting Azure OpenAI deployments into standard
/// Chat Completions-compatible form and normalizes responses back.
///
/// Azure OpenAI accepts Chat Completions bodies as-is; this filter
/// handles the `api-version` query parameter, strips Azure-specific
/// response fields (`prompt_filter_results`, `content_filter_results`),
/// and normalizes error responses where Azure omits the `type` field.
///
/// # YAML
///
/// ```yaml
/// filter: azure_to_openai
/// ```
///
/// # Full YAML
///
/// ```yaml
/// filter: azure_to_openai
/// api_version: "2024-10-21"
/// max_body_bytes: 1048576
/// ```
pub struct AzureToOpenaiFilter {
    /// Parsed and validated configuration.
    config: AzureToOpenaiConfig,
}

impl AzureToOpenaiFilter {
    /// Create a filter from parsed YAML config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the YAML config is invalid.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: AzureToOpenaiConfig = parse_filter_config("azure_to_openai", config)?;
        let validated = build_config(cfg)?;
        Ok(Box::new(Self { config: validated }))
    }
}

#[async_trait]
impl HttpFilter for AzureToOpenaiFilter {
    fn name(&self) -> &'static str {
        "azure_to_openai"
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

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        inject_api_version_query(ctx, &self.config.api_version);

        ctx.request_headers_to_remove.push(http::header::ACCEPT_ENCODING);

        Ok(FilterAction::Continue)
    }

    async fn on_request_body(
        &self,
        _ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if !end_of_stream {
            return Ok(FilterAction::Continue);
        }

        let Some(bytes) = body.as_ref() else {
            return Ok(FilterAction::Continue);
        };

        if bytes.is_empty() {
            return Ok(FilterAction::Continue);
        }

        if let Some(transformed) = request::strip_ignored_fields(bytes) {
            debug!(
                original_len = bytes.len(),
                transformed_len = transformed.len(),
                "stripped model field from Azure OpenAI request"
            );
            *body = Some(Bytes::from(transformed));
        }

        Ok(FilterAction::Continue)
    }

    async fn on_response(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        let transform = response_transform(ctx);

        ctx.set_metadata(RESPONSE_TRANSFORM_KEY, transform);
        if transform == RESPONSE_TRANSFORM_ERROR {
            let status = ctx
                .response_header
                .as_ref()
                .map_or(500, |response| response.status.as_u16());
            ctx.set_metadata(RESPONSE_STATUS_KEY, status.to_string());
        }

        if transform == RESPONSE_TRANSFORM_SSE {
            ctx.insert_filter_state(SseFrameParser::new(self.config.max_body_bytes));
        } else {
            ctx.set_response_body_mode(BodyMode::StreamBuffer {
                max_bytes: Some(self.config.max_body_bytes),
            });
        }
        remove_stale_content_length(ctx);

        Ok(FilterAction::Continue)
    }

    fn on_response_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        match ctx.get_metadata(RESPONSE_TRANSFORM_KEY) {
            Some(RESPONSE_TRANSFORM_SSE) => strip_sse_chunk(ctx, body, end_of_stream),
            Some(RESPONSE_TRANSFORM_ERROR) if end_of_stream => {
                transform_error_body(ctx, body);
            },
            Some(RESPONSE_TRANSFORM_SUCCESS) if end_of_stream => {
                transform_success_body(body);
            },
            _ => {},
        }
        Ok(FilterAction::Continue)
    }
}

// -----------------------------------------------------------------------------
// Request Helpers
// -----------------------------------------------------------------------------

/// Append `api-version` to the request URI query string.
fn inject_api_version_query(ctx: &mut HttpFilterContext<'_>, api_version: &str) {
    let uri = &ctx.request.uri;
    let path_and_query = uri.path_and_query().map_or_else(|| uri.path(), |pq| pq.as_str());

    let new_pq = if path_and_query.contains('?') {
        format!("{path_and_query}&api-version={api_version}")
    } else {
        format!("{path_and_query}?api-version={api_version}")
    };

    ctx.rewritten_path = Some(new_pq);
}

// -----------------------------------------------------------------------------
// Response Helpers
// -----------------------------------------------------------------------------

/// Remove stale `Content-Length` so the proxy recomputes it after body
/// transformation.
fn remove_stale_content_length(ctx: &mut HttpFilterContext<'_>) {
    if let Some(resp) = &mut ctx.response_header {
        resp.headers.remove(http::header::CONTENT_LENGTH);
        ctx.response_headers_modified = true;
    }
}

/// Select the response transformation mode while headers are available.
fn response_transform(ctx: &HttpFilterContext<'_>) -> &'static str {
    let status = ctx.response_header.as_ref().map(|r| r.status);
    let is_error = status.is_some_and(|s| s.is_client_error() || s.is_server_error());
    let is_sse = ctx
        .response_header
        .as_ref()
        .and_then(|r| r.headers.get(http::header::CONTENT_TYPE))
        .and_then(|v| v.to_str().ok())
        .is_some_and(crate::is_event_stream_content_type);

    if is_sse {
        RESPONSE_TRANSFORM_SSE
    } else if is_error {
        RESPONSE_TRANSFORM_ERROR
    } else {
        RESPONSE_TRANSFORM_SUCCESS
    }
}

/// Normalize an Azure error response body (fill `type: null`, strip `innererror`).
fn transform_error_body(ctx: &HttpFilterContext<'_>, body: &mut Option<Bytes>) {
    if let Some(bytes) = body.as_ref().filter(|b| !b.is_empty()) {
        let status = ctx
            .get_metadata(RESPONSE_STATUS_KEY)
            .and_then(|value| value.parse::<u16>().ok())
            .and_then(|value| http::StatusCode::from_u16(value).ok())
            .unwrap_or(http::StatusCode::INTERNAL_SERVER_ERROR);

        if let Some(normalized) = wire::normalize_error_response(bytes, status) {
            debug!(
                original_len = bytes.len(),
                normalized_len = normalized.len(),
                "normalized Azure error response type field"
            );
            *body = Some(Bytes::from(normalized));
        }
    }
}

/// Strip Azure content-filter fields from a buffered JSON response body.
fn transform_success_body(body: &mut Option<Bytes>) {
    if let Some(bytes) = body.as_ref().filter(|b| !b.is_empty())
        && let Some(stripped) = response::strip_azure_fields(bytes)
    {
        debug!(
            original_len = bytes.len(),
            stripped_len = stripped.len(),
            "stripped Azure content-filter fields from response"
        );
        *body = Some(Bytes::from(stripped));
    }
}

// -----------------------------------------------------------------------------
// SSE Helpers
// -----------------------------------------------------------------------------

/// Process an SSE chunk: parse frames via [`SseFrameParser`], strip
/// Azure-specific fields from each frame's data, and re-emit as SSE.
fn strip_sse_chunk(ctx: &mut HttpFilterContext<'_>, body: &mut Option<Bytes>, end_of_stream: bool) {
    let Some(bytes) = body.as_ref() else {
        if end_of_stream {
            *body = Some(Bytes::new());
        }
        return;
    };

    let Some(mut parser) = ctx.remove_filter_state::<SseFrameParser>() else {
        return;
    };

    let frames = match parser.parse_chunk(bytes) {
        Ok(frames) => frames,
        Err(e) => {
            debug!(error = %e, "SSE parse error in azure_to_openai");
            ctx.insert_filter_state(parser);
            return;
        },
    };

    if !end_of_stream {
        ctx.insert_filter_state(parser);
    }

    *body = Some(Bytes::from(rebuild_sse_frames(&frames)));
}

/// Serialize parsed [`SseFrame`]s back to SSE wire format, stripping
/// Azure-specific fields from each data payload.
fn rebuild_sse_frames(frames: &[SseFrame]) -> Vec<u8> {
    const DONE_SENTINEL: &[u8] = b"[DONE]";

    let mut output = Vec::new();
    for frame in frames {
        if frame.data.starts_with(DONE_SENTINEL) {
            output.extend_from_slice(b"data: [DONE]\n\n");
            continue;
        }

        let stripped = response::strip_azure_fields(&frame.data);
        let data = stripped.as_deref().unwrap_or(&frame.data);

        output.extend_from_slice(b"data: ");
        output.extend_from_slice(data);
        output.extend_from_slice(b"\n\n");
    }
    output
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
        let filter = AzureToOpenaiFilter::from_config(&yaml).unwrap();

        assert_eq!(filter.name(), "azure_to_openai");
    }

    #[test]
    fn explicit_config_parses() {
        let yaml: serde_yaml::Value =
            serde_yaml::from_str("api_version: \"2025-01-01\"\nmax_body_bytes: 2097152").unwrap();
        let filter = AzureToOpenaiFilter::from_config(&yaml).unwrap();

        assert_eq!(filter.name(), "azure_to_openai");
    }

    #[test]
    fn unknown_config_field_rejected() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("unknown_field: true").unwrap();
        let result = AzureToOpenaiFilter::from_config(&yaml);

        assert!(result.is_err());
    }

    #[test]
    fn empty_api_version_rejected() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("api_version: \"\"").unwrap();
        let result = AzureToOpenaiFilter::from_config(&yaml);

        assert!(result.is_err());
    }

    #[test]
    fn zero_max_body_bytes_rejected() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("max_body_bytes: 0").unwrap();
        let result = AzureToOpenaiFilter::from_config(&yaml);

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn on_request_removes_accept_encoding() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("{}").unwrap();
        let filter = AzureToOpenaiFilter::from_config(&yaml).unwrap();
        let request = make_request(Method::POST, "/openai/deployments/gpt4o/chat/completions");
        let mut ctx = make_filter_context(&request);

        drop(filter.on_request(&mut ctx).await.unwrap());

        assert!(ctx.request_headers_to_remove.contains(&http::header::ACCEPT_ENCODING));
    }

    #[tokio::test]
    async fn on_request_injects_api_version() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("{}").unwrap();
        let filter = AzureToOpenaiFilter::from_config(&yaml).unwrap();
        let request = make_request(Method::POST, "/openai/deployments/gpt4o/chat/completions");
        let mut ctx = make_filter_context(&request);

        drop(filter.on_request(&mut ctx).await.unwrap());

        let rewritten = ctx.rewritten_path.as_ref().unwrap();
        assert!(
            rewritten.contains("api-version=2024-10-21"),
            "default api-version should be injected, got: {rewritten}"
        );
    }

    #[tokio::test]
    async fn on_request_injects_api_version_with_existing_query() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("api_version: \"2025-03-01\"").unwrap();
        let filter = AzureToOpenaiFilter::from_config(&yaml).unwrap();
        let request = make_request(Method::POST, "/openai/deployments/gpt4o/chat/completions?foo=bar");
        let mut ctx = make_filter_context(&request);

        drop(filter.on_request(&mut ctx).await.unwrap());

        let rewritten = ctx.rewritten_path.as_ref().unwrap();
        assert!(
            rewritten.contains("foo=bar&api-version=2025-03-01"),
            "api-version should append to existing query, got: {rewritten}"
        );
    }

    #[tokio::test]
    async fn on_request_body_strips_model() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("{}").unwrap();
        let filter = AzureToOpenaiFilter::from_config(&yaml).unwrap();
        let request = make_request(Method::POST, "/openai/deployments/gpt4o/chat/completions");
        let mut ctx = make_filter_context(&request);
        let mut body = Some(Bytes::from(
            br#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}"#.to_vec(),
        ));

        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

        assert!(matches!(action, FilterAction::Continue));
        let parsed: serde_json::Value = serde_json::from_slice(body.unwrap().as_ref()).unwrap();
        assert!(parsed.get("model").is_none(), "model field should be stripped");
        assert_eq!(parsed["messages"][0]["role"], "user");
    }

    #[tokio::test]
    async fn on_request_body_preserves_body_without_model() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("{}").unwrap();
        let filter = AzureToOpenaiFilter::from_config(&yaml).unwrap();
        let request = make_request(Method::POST, "/openai/deployments/gpt4o/chat/completions");
        let mut ctx = make_filter_context(&request);
        let original = Bytes::from(br#"{"messages":[{"role":"user","content":"hi"}]}"#.to_vec());
        let mut body = Some(original.clone());

        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

        assert!(matches!(action, FilterAction::Continue));
        assert_eq!(body, Some(original), "body without model should pass through");
    }

    #[test]
    fn response_transform_success_for_json() {
        let request = make_request(Method::POST, "/chat/completions");
        let mut ctx = make_filter_context(&request);
        let mut response = make_response();
        ctx.response_header = Some(&mut response);

        assert_eq!(response_transform(&ctx), RESPONSE_TRANSFORM_SUCCESS);
    }

    #[test]
    fn response_transform_error_for_4xx() {
        let request = make_request(Method::POST, "/chat/completions");
        let mut ctx = make_filter_context(&request);
        let mut response = make_response();
        response.status = StatusCode::BAD_REQUEST;
        ctx.response_header = Some(&mut response);

        assert_eq!(response_transform(&ctx), RESPONSE_TRANSFORM_ERROR);
    }

    #[test]
    fn response_transform_sse_for_event_stream() {
        let request = make_request(Method::POST, "/chat/completions");
        let mut ctx = make_filter_context(&request);
        let mut response = make_response();
        response.headers.insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("text/event-stream"),
        );
        ctx.response_header = Some(&mut response);

        assert_eq!(response_transform(&ctx), RESPONSE_TRANSFORM_SSE);
    }

    #[tokio::test]
    async fn on_response_body_strips_azure_fields() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("{}").unwrap();
        let filter = AzureToOpenaiFilter::from_config(&yaml).unwrap();
        let request = make_request(Method::POST, "/chat/completions");
        let mut ctx = make_filter_context(&request);
        ctx.set_metadata(RESPONSE_TRANSFORM_KEY, RESPONSE_TRANSFORM_SUCCESS);

        let azure_response = serde_json::json!({
            "id": "chatcmpl-abc",
            "choices": [{
                "message": {"role": "assistant", "content": "Paris"},
                "finish_reason": "stop",
                "content_filter_results": {"hate": {"filtered": false}}
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3},
            "prompt_filter_results": [{"prompt_index": 0}]
        });
        let mut body = Some(Bytes::from(azure_response.to_string()));

        let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();

        assert!(matches!(action, FilterAction::Continue));
        let parsed: serde_json::Value = serde_json::from_slice(body.unwrap().as_ref()).unwrap();
        assert!(parsed.get("prompt_filter_results").is_none());
        assert!(parsed["choices"][0].get("content_filter_results").is_none());
        assert_eq!(parsed["choices"][0]["message"]["content"], "Paris");
    }

    #[test]
    fn rebuild_sse_frames_strips_content_filter() {
        let frames = vec![SseFrame {
            event_type: None,
            data: br#"{"id":"chatcmpl-abc","choices":[{"delta":{"content":"Hi"},"content_filter_results":{"hate":{"filtered":false}}}]}"#.to_vec(),
        }];
        let rebuilt = rebuild_sse_frames(&frames);
        let output = std::str::from_utf8(&rebuilt).unwrap();

        assert!(output.starts_with("data: "));
        let json_str = output.trim_start_matches("data: ").trim();
        let parsed: serde_json::Value = serde_json::from_str(json_str).unwrap();
        assert!(parsed["choices"][0].get("content_filter_results").is_none());
        assert_eq!(parsed["choices"][0]["delta"]["content"], "Hi");
    }

    #[test]
    fn rebuild_sse_frames_preserves_done_sentinel() {
        let frames = vec![SseFrame {
            event_type: None,
            data: b"[DONE]".to_vec(),
        }];
        let output = rebuild_sse_frames(&frames);
        assert_eq!(output, b"data: [DONE]\n\n");
    }

    #[test]
    fn rebuild_sse_frames_preserves_clean_event() {
        let data = br#"{"id":"chatcmpl-abc","choices":[{"delta":{"content":"Hi"}}]}"#;
        let frames = vec![SseFrame {
            event_type: None,
            data: data.to_vec(),
        }];
        let rebuilt = rebuild_sse_frames(&frames);
        let output = std::str::from_utf8(&rebuilt).unwrap();

        assert!(output.starts_with("data: "));
        let json_str = output.trim_start_matches("data: ").trim();
        assert_eq!(json_str.as_bytes(), data, "clean event should pass through unchanged");
    }

    #[test]
    fn strip_sse_chunk_processes_multi_event_chunk() {
        let request = make_request(Method::POST, "/chat/completions");
        let mut ctx = make_filter_context(&request);
        ctx.current_filter_id = Some(0);
        ctx.insert_filter_state(SseFrameParser::new(65_536));

        let chunk = b"data: {\"choices\":[{\"delta\":{\"content\":\"A\"},\"content_filter_results\":{}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"B\"},\"content_filter_results\":{}}]}\n\n";
        let mut body = Some(Bytes::from(chunk.to_vec()));

        strip_sse_chunk(&mut ctx, &mut body, false);

        let bytes = body.unwrap();
        let output = std::str::from_utf8(bytes.as_ref()).unwrap();
        assert!(
            !output.contains("content_filter_results"),
            "both events should be stripped"
        );
        assert!(output.contains("\"content\":\"A\""));
        assert!(output.contains("\"content\":\"B\""));
    }

    #[test]
    fn strip_sse_chunk_buffers_partial_line() {
        let request = make_request(Method::POST, "/chat/completions");
        let mut ctx = make_filter_context(&request);
        ctx.current_filter_id = Some(0);
        ctx.insert_filter_state(SseFrameParser::new(65_536));

        let chunk1 = b"data: {\"choices\":[{\"delta\":{\"con";
        let mut body1 = Some(Bytes::from(chunk1.to_vec()));
        strip_sse_chunk(&mut ctx, &mut body1, false);

        assert!(
            ctx.get_filter_state::<SseFrameParser>().is_some(),
            "parser should be retained in filter state"
        );
        assert!(
            body1.unwrap().is_empty(),
            "no complete frame yet — output should be empty"
        );

        let chunk2 = b"tent\":\"Hi\"},\"content_filter_results\":{}}]}\n\n";
        let mut body2 = Some(Bytes::from(chunk2.to_vec()));
        strip_sse_chunk(&mut ctx, &mut body2, false);

        let bytes2 = body2.unwrap();
        let output = std::str::from_utf8(bytes2.as_ref()).unwrap();
        assert!(!output.contains("content_filter_results"));
        assert!(output.contains("\"content\":\"Hi\""));
    }

    #[test]
    fn strip_sse_chunk_preserves_multibyte_utf8_across_boundary() {
        let request = make_request(Method::POST, "/chat/completions");
        let mut ctx = make_filter_context(&request);
        ctx.current_filter_id = Some(0);
        ctx.insert_filter_state(SseFrameParser::new(65_536));

        let full_line = "data: {\"choices\":[{\"delta\":{\"content\":\"שלום\"},\"content_filter_results\":{}}]}\n\n";
        let full_bytes = full_line.as_bytes();
        let split_at = full_bytes.len() / 2;

        let mut body1 = Some(Bytes::from(full_bytes[..split_at].to_vec()));
        strip_sse_chunk(&mut ctx, &mut body1, false);

        let mut body2 = Some(Bytes::from(full_bytes[split_at..].to_vec()));
        strip_sse_chunk(&mut ctx, &mut body2, false);

        let bytes2 = body2.unwrap();
        let output = std::str::from_utf8(bytes2.as_ref()).unwrap();
        assert!(
            output.contains("שלום"),
            "multi-byte UTF-8 should survive chunk boundary"
        );
        assert!(!output.contains("content_filter_results"));
    }

    #[tokio::test]
    async fn on_response_body_normalizes_error_type() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("{}").unwrap();
        let filter = AzureToOpenaiFilter::from_config(&yaml).unwrap();
        let request = make_request(Method::POST, "/chat/completions");
        let mut ctx = make_filter_context(&request);
        ctx.set_metadata(RESPONSE_TRANSFORM_KEY, RESPONSE_TRANSFORM_ERROR);
        ctx.set_metadata(RESPONSE_STATUS_KEY, "404");

        let azure_error = br#"{"error":{"message":"deployment not found","type":null,"code":"DeploymentNotFound"}}"#;
        let mut body = Some(Bytes::from(azure_error.to_vec()));

        let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();

        assert!(matches!(action, FilterAction::Continue));
        let parsed: serde_json::Value = serde_json::from_slice(body.unwrap().as_ref()).unwrap();
        assert_eq!(parsed["error"]["type"], "invalid_request_error");
        assert_eq!(parsed["error"]["message"], "deployment not found");
    }
}
