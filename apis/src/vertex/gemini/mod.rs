// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! OpenAI Chat Completions to Vertex AI Gemini translation filter.
//!
//! Rewrites Chat Completions request bodies to the Gemini
//! `generateContent` shape, translates non-streaming responses back
//! to Chat Completions format, and normalizes Vertex error envelopes.
//! Streaming SSE frames are translated inline using the shared
//! [`SseFrameParser`].
//!
//! [`SseFrameParser`]: crate::openai::sse::SseFrameParser

pub(crate) mod config;
pub(crate) mod request;
pub(crate) mod response;

use async_trait::async_trait;
use bytes::Bytes;
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, Rejection, parse_filter_config,
};
use tracing::{debug, warn};

use self::config::{FILTER_NAME, VertexGeminiConfig, build_config};
use crate::{is_event_stream_content_type, openai::sse::SseFrameParser, vertex::wire};

/// Metadata key: response transform mode (sse / error / success).
const RESPONSE_TRANSFORM_KEY: &str = "vertex_gemini.response_transform";
/// Response transform marker: SSE streaming.
const RESPONSE_TRANSFORM_SSE: &str = "sse";
/// Response transform marker: upstream error.
const RESPONSE_TRANSFORM_ERROR: &str = "error";
/// Response transform marker: non-streaming success.
const RESPONSE_TRANSFORM_SUCCESS: &str = "success";

/// Metadata key: upstream HTTP status for the body phase.
const RESPONSE_STATUS_KEY: &str = "vertex_gemini.response_status";
/// Metadata key: model name from the request.
const REQUEST_MODEL_KEY: &str = "vertex_gemini.model";
/// Fallback response ID when no upstream `responseId` is available.
///
/// Used in both non-streaming (`response.rs`) and streaming (`mod.rs`)
/// paths so that clients see a consistent id shape.
pub(super) const FALLBACK_RESPONSE_ID: &str = "chatcmpl-vertex";

// -----------------------------------------------------------------------------
// Filter State
// -----------------------------------------------------------------------------

/// Per-request state for tracking SSE streaming progress.
struct StreamState {
    /// Byte-level SSE frame reassembly parser.
    parser: SseFrameParser,
    /// Whether the next frame is the first one (includes `role` in delta).
    is_first_chunk: bool,
    /// Unix timestamp (seconds) fixed at stream start so every chunk
    /// reports the same `created` value, matching OpenAI behavior.
    created: u64,
}

// -----------------------------------------------------------------------------
// OpenaiChatCompletionsToVertexaiGeminiFilter
// -----------------------------------------------------------------------------

/// Transforms OpenAI Chat Completions requests into Vertex AI Gemini
/// `generateContent` format and translates responses back.
///
/// # YAML
///
/// ```yaml
/// filter: openai_chat_completions_to_vertexai_gemini
/// project: my-gcp-project
/// ```
///
/// # Full YAML
///
/// ```yaml
/// filter: openai_chat_completions_to_vertexai_gemini
/// project: my-gcp-project
/// region: us-central1
/// max_body_bytes: 1048576
/// ```
pub struct OpenaiChatCompletionsToVertexaiGeminiFilter {
    /// Parsed and validated configuration.
    config: VertexGeminiConfig,
}

impl OpenaiChatCompletionsToVertexaiGeminiFilter {
    /// Create a filter from parsed YAML config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the YAML config is invalid.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: VertexGeminiConfig = parse_filter_config(FILTER_NAME, config)?;
        let validated = build_config(cfg)?;
        Ok(Box::new(Self { config: validated }))
    }

    /// Build the Vertex AI upstream path for the given model and
    /// streaming mode.
    ///
    /// Streaming requests require `?alt=sse` to receive SSE frames;
    /// without it Vertex returns a single JSON array response.
    fn vertex_path(&self, model: &str, stream: bool) -> String {
        let (action, query) = if stream {
            ("streamGenerateContent", "?alt=sse")
        } else {
            ("generateContent", "")
        };
        format!(
            "/v1/projects/{project}/locations/{region}/publishers/google/models/{model}:{action}{query}",
            project = self.config.project,
            region = self.config.region,
        )
    }
}

#[async_trait]
impl HttpFilter for OpenaiChatCompletionsToVertexaiGeminiFilter {
    fn name(&self) -> &'static str {
        "openai_chat_completions_to_vertexai_gemini"
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
        // Remove Accept-Encoding to prevent double-compressed bodies.
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

        let Some(bytes) = body.as_ref().filter(|b| !b.is_empty()) else {
            return Ok(FilterAction::Continue);
        };

        match request::transform_request(bytes) {
            Ok(result) => {
                debug!(
                    original_len = bytes.len(),
                    transformed_len = result.body.len(),
                    model = result.model.as_str(),
                    stream = result.stream,
                    "transformed Chat Completions request to Gemini format"
                );

                ctx.rewritten_path = Some(self.vertex_path(&result.model, result.stream));
                ctx.set_metadata(REQUEST_MODEL_KEY, result.model);
                *body = Some(Bytes::from(result.body));

                Ok(FilterAction::Continue)
            },
            Err(msg) => {
                warn!(error = msg.as_str(), "failed to transform request to Gemini");
                Ok(FilterAction::Reject(build_request_rejection(&msg)))
            },
        }
    }

    async fn on_response(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        let transform = response_transform(ctx);
        ctx.set_metadata(RESPONSE_TRANSFORM_KEY, transform);

        if transform == RESPONSE_TRANSFORM_ERROR {
            let status = ctx.response_header.as_ref().map_or(500, |r| r.status.as_u16());
            ctx.set_metadata(RESPONSE_STATUS_KEY, status.to_string());

            // Buffer the full error body for normalization.
            ctx.set_response_body_mode(BodyMode::StreamBuffer {
                max_bytes: Some(self.config.max_body_bytes),
            });
            prepare_response_headers(ctx, "application/json");
        } else if transform == RESPONSE_TRANSFORM_SUCCESS {
            // Buffer the full success body for translation.
            ctx.set_response_body_mode(BodyMode::StreamBuffer {
                max_bytes: Some(self.config.max_body_bytes),
            });
            prepare_response_headers(ctx, "application/json");
        } else {
            // SSE: stream chunks, translate each frame inline.
            prepare_response_headers(ctx, "text/event-stream");
        }

        Ok(FilterAction::Continue)
    }

    fn on_response_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        match ctx.get_metadata(RESPONSE_TRANSFORM_KEY) {
            Some(RESPONSE_TRANSFORM_SSE) => {
                translate_sse_chunk(ctx, body, end_of_stream, self.config.max_body_bytes);
            },
            Some(RESPONSE_TRANSFORM_ERROR) if end_of_stream => {
                translate_error_body(ctx, body);
            },
            Some(RESPONSE_TRANSFORM_SUCCESS) if end_of_stream => {
                translate_success_body(ctx, body);
            },
            _ => {},
        }

        Ok(FilterAction::Continue)
    }
}

// -----------------------------------------------------------------------------
// Response Mode Selection
// -----------------------------------------------------------------------------

/// Determine the response transformation mode from headers.
fn response_transform(ctx: &HttpFilterContext<'_>) -> &'static str {
    let status = ctx.response_header.as_ref().map(|r| r.status);
    let is_error = status.is_some_and(|s| s.is_client_error() || s.is_server_error());
    let is_sse = ctx
        .response_header
        .as_ref()
        .and_then(|r| r.headers.get(http::header::CONTENT_TYPE))
        .and_then(|v| v.to_str().ok())
        .is_some_and(is_event_stream_content_type);

    if is_error {
        RESPONSE_TRANSFORM_ERROR
    } else if is_sse {
        RESPONSE_TRANSFORM_SSE
    } else {
        RESPONSE_TRANSFORM_SUCCESS
    }
}

// -----------------------------------------------------------------------------
// Response Header Helpers
// -----------------------------------------------------------------------------

/// Remove stale representation headers and set the response content type.
fn prepare_response_headers(ctx: &mut HttpFilterContext<'_>, content_type: &'static str) {
    if let Some(resp) = &mut ctx.response_header {
        resp.headers.remove(http::header::CONTENT_LENGTH);
        resp.headers.remove(http::header::CONTENT_ENCODING);
        resp.headers
            .insert(http::header::CONTENT_TYPE, http::HeaderValue::from_static(content_type));
        ctx.response_headers_modified = true;
    }
}

// -----------------------------------------------------------------------------
// Request Rejection
// -----------------------------------------------------------------------------

/// Build a 400 rejection with an OpenAI-shaped error envelope.
///
/// Used when request body transformation fails (e.g. invalid JSON,
/// missing `model` field) so the client receives a clear error instead
/// of a confusing upstream failure.
fn build_request_rejection(message: &str) -> Rejection {
    let body = wire::build_openai_error_body(message, "invalid_request_error", "invalid_request_error");
    Rejection::status(400)
        .with_header("content-type", "application/json")
        .with_body(Bytes::from(body))
}

// -----------------------------------------------------------------------------
// Response Body Handlers
// -----------------------------------------------------------------------------

/// The OpenAI `data: [DONE]` sentinel that signals end of stream.
///
/// Gemini does not send this — it just closes the connection. We
/// append it so OpenAI SDK clients finalize cleanly.
const SSE_DONE: &[u8] = b"data: [DONE]\n\n";

/// Translate SSE streaming chunks from Gemini to Chat Completions format.
fn translate_sse_chunk(
    ctx: &mut HttpFilterContext<'_>,
    body: &mut Option<Bytes>,
    end_of_stream: bool,
    max_body_bytes: usize,
) {
    let Some(bytes) = body.as_ref().filter(|b| !b.is_empty()) else {
        if end_of_stream {
            *body = Some(Bytes::from_static(SSE_DONE));
        }
        return;
    };

    let mut state = ctx.remove_filter_state::<StreamState>().unwrap_or_else(|| StreamState {
        parser: SseFrameParser::new(max_body_bytes),
        is_first_chunk: true,
        created: response::created_timestamp(),
    });

    let mut output = translate_sse_frames(ctx, &mut state, bytes);

    if end_of_stream {
        output.extend_from_slice(SSE_DONE);
    } else {
        ctx.insert_filter_state(state);
    }

    *body = Some(if output.is_empty() {
        Bytes::new()
    } else {
        Bytes::from(output)
    });
}

/// Parse and translate SSE frames from a raw byte chunk.
///
/// On parse error, returns an empty buffer and preserves `state`
/// for the next call.
fn translate_sse_frames(ctx: &HttpFilterContext<'_>, state: &mut StreamState, bytes: &Bytes) -> Vec<u8> {
    let model = ctx.get_metadata(REQUEST_MODEL_KEY).unwrap_or("unknown");
    let id = FALLBACK_RESPONSE_ID;

    match state.parser.parse_chunk(bytes) {
        Ok(frames) => render_sse_frames(&frames, model, id, state),
        Err(e) => {
            debug!(error = %e, "SSE parse error in vertex gemini filter");
            Vec::new()
        },
    }
}

/// Render parsed SSE frames into translated OpenAI SSE output bytes.
fn render_sse_frames(
    frames: &[crate::openai::sse::SseFrame],
    model: &str,
    id: &str,
    state: &mut StreamState,
) -> Vec<u8> {
    let mut output = Vec::new();
    for frame in frames {
        if frame.data == b"[DONE]" {
            output.extend_from_slice(b"data: [DONE]\n\n");
            continue;
        }
        match response::transform_stream_chunk(&frame.data, model, id, state.is_first_chunk, state.created) {
            Ok(translated) => {
                state.is_first_chunk = false;
                output.extend_from_slice(b"data: ");
                output.extend_from_slice(&translated);
                output.extend_from_slice(b"\n\n");
            },
            Err(e) => {
                debug!(error = e.as_str(), "failed to translate Gemini SSE frame");
            },
        }
    }
    output
}

/// Normalize a Vertex AI error response to the OpenAI error envelope.
fn translate_error_body(ctx: &HttpFilterContext<'_>, body: &mut Option<Bytes>) {
    let bytes = body.as_deref().unwrap_or_default();

    let status = ctx
        .get_metadata(RESPONSE_STATUS_KEY)
        .and_then(|v| v.parse::<u16>().ok())
        .and_then(|v| http::StatusCode::from_u16(v).ok())
        .unwrap_or(http::StatusCode::INTERNAL_SERVER_ERROR);

    let normalized = wire::normalize_error_response(bytes, status);
    debug!(
        original_len = bytes.len(),
        normalized_len = normalized.len(),
        "normalized Vertex error to OpenAI envelope"
    );
    *body = Some(Bytes::from(normalized));
}

/// Translate a non-streaming Gemini success response to Chat Completions.
///
/// When translation fails the body is replaced with an OpenAI error
/// envelope so the client receives a structured error instead of raw
/// Gemini JSON it cannot parse.
fn translate_success_body(ctx: &HttpFilterContext<'_>, body: &mut Option<Bytes>) {
    let bytes = body.as_deref().unwrap_or_default();

    let model = ctx.get_metadata(REQUEST_MODEL_KEY).unwrap_or("unknown");

    match response::transform_response(bytes, model) {
        Ok(translated) => {
            debug!(
                original_len = bytes.len(),
                translated_len = translated.len(),
                "translated Gemini response to Chat Completions"
            );
            *body = Some(Bytes::from(translated));
        },
        Err(msg) => {
            warn!(error = msg.as_str(), "failed to translate Gemini response");
            let error_body = wire::build_openai_error_body(
                "upstream response could not be translated",
                "server_error",
                "server_error",
            );
            *body = Some(Bytes::from(error_body));
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
    clippy::too_many_lines,
    clippy::panic,
    reason = "tests"
)]
mod tests {
    use bytes::Bytes;
    use http::{Method, StatusCode};

    use super::*;
    use crate::test_utils::{make_filter_context, make_request, make_response};

    /// Build a filter instance for tests, bypassing the `Box<dyn HttpFilter>`.
    fn make_filter(yaml_str: &str) -> OpenaiChatCompletionsToVertexaiGeminiFilter {
        let yaml: serde_yaml::Value = serde_yaml::from_str(yaml_str).unwrap();
        let cfg: VertexGeminiConfig = parse_filter_config(FILTER_NAME, &yaml).unwrap();
        let validated = build_config(cfg).unwrap();
        OpenaiChatCompletionsToVertexaiGeminiFilter { config: validated }
    }

    // --- from_config ---

    #[test]
    fn default_config_with_project_parses() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("project: my-project").unwrap();
        let filter = OpenaiChatCompletionsToVertexaiGeminiFilter::from_config(&yaml).unwrap();

        assert_eq!(
            filter.name(),
            FILTER_NAME,
            "fn name() must match FILTER_NAME to keep doc-gen and config in sync"
        );
    }

    #[test]
    fn missing_project_rejected() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("{}").unwrap();
        let result = OpenaiChatCompletionsToVertexaiGeminiFilter::from_config(&yaml);

        assert!(result.is_err(), "missing project should be rejected");
    }

    // --- vertex_path ---

    #[test]
    fn vertex_path_non_streaming() {
        let filter = make_filter("project: test-proj\nregion: europe-west4");

        assert_eq!(
            filter.vertex_path("gemini-2.0-flash", false),
            "/v1/projects/test-proj/locations/europe-west4/publishers/google/models/gemini-2.0-flash:generateContent"
        );
    }

    #[test]
    fn vertex_path_streaming() {
        let filter = make_filter("project: test-proj\nregion: us-central1");

        assert_eq!(
            filter.vertex_path("gemini-2.0-flash", true),
            "/v1/projects/test-proj/locations/us-central1/publishers/google/models/gemini-2.0-flash:streamGenerateContent?alt=sse"
        );
    }

    // --- response_transform ---

    #[test]
    fn response_transform_error_on_4xx() {
        let req = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        let mut resp = make_response();
        resp.status = StatusCode::BAD_REQUEST;
        ctx.response_header = Some(&mut resp);

        assert_eq!(response_transform(&ctx), RESPONSE_TRANSFORM_ERROR);
    }

    #[test]
    fn response_transform_error_on_5xx() {
        let req = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        let mut resp = make_response();
        resp.status = StatusCode::INTERNAL_SERVER_ERROR;
        ctx.response_header = Some(&mut resp);

        assert_eq!(response_transform(&ctx), RESPONSE_TRANSFORM_ERROR);
    }

    #[test]
    fn response_transform_sse_on_event_stream() {
        let req = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        let mut resp = make_response();
        resp.headers.insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("text/event-stream"),
        );
        ctx.response_header = Some(&mut resp);

        assert_eq!(response_transform(&ctx), RESPONSE_TRANSFORM_SSE);
    }

    #[test]
    fn response_transform_success_on_json() {
        let req = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        let mut resp = make_response();
        resp.headers.insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("application/json"),
        );
        ctx.response_header = Some(&mut resp);

        assert_eq!(response_transform(&ctx), RESPONSE_TRANSFORM_SUCCESS);
    }

    // --- on_request ---

    #[tokio::test]
    async fn on_request_removes_accept_encoding() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("project: p").unwrap();
        let filter = OpenaiChatCompletionsToVertexaiGeminiFilter::from_config(&yaml).unwrap();
        let req = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);

        drop(filter.on_request(&mut ctx).await.unwrap());

        assert!(
            ctx.request_headers_to_remove.contains(&http::header::ACCEPT_ENCODING),
            "response transformation requires unencoded upstream representation"
        );
    }

    // --- on_request_body ---

    #[tokio::test]
    async fn on_request_body_transforms_and_rewrites_path() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("project: my-proj\nregion: us-central1").unwrap();
        let filter = OpenaiChatCompletionsToVertexaiGeminiFilter::from_config(&yaml).unwrap();
        let req = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);

        let mut body = Some(Bytes::from(
            br#"{"model":"gemini-2.0-flash","messages":[{"role":"user","content":"Hi"}]}"#.to_vec(),
        ));
        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

        assert!(matches!(action, FilterAction::Continue));
        assert_eq!(
            ctx.rewritten_path.as_deref(),
            Some(
                "/v1/projects/my-proj/locations/us-central1/publishers/google/models/gemini-2.0-flash:generateContent"
            ),
            "non-streaming path should use generateContent"
        );
        assert_eq!(ctx.get_metadata(REQUEST_MODEL_KEY), Some("gemini-2.0-flash"),);

        let parsed: serde_json::Value = serde_json::from_slice(body.unwrap().as_ref()).unwrap();
        assert!(
            parsed.get("contents").is_some(),
            "body should be transformed to Gemini format"
        );
    }

    #[tokio::test]
    async fn on_request_body_streaming_uses_stream_endpoint() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("project: p\nregion: r").unwrap();
        let filter = OpenaiChatCompletionsToVertexaiGeminiFilter::from_config(&yaml).unwrap();
        let req = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);

        let mut body = Some(Bytes::from(
            br#"{"model":"gemini-2.0-flash","stream":true,"messages":[{"role":"user","content":"Hi"}]}"#.to_vec(),
        ));
        drop(filter.on_request_body(&mut ctx, &mut body, true).await.unwrap());

        assert!(
            ctx.rewritten_path
                .as_deref()
                .unwrap()
                .ends_with(":streamGenerateContent?alt=sse"),
            "streaming path should use streamGenerateContent?alt=sse"
        );
    }

    #[tokio::test]
    async fn on_request_body_invalid_json_rejects() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("project: p").unwrap();
        let filter = OpenaiChatCompletionsToVertexaiGeminiFilter::from_config(&yaml).unwrap();
        let req = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        let mut body = Some(Bytes::from_static(b"not json"));

        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

        let FilterAction::Reject(rejection) = action else {
            panic!("invalid body should produce a rejection");
        };
        assert_eq!(rejection.status, 400);
        let parsed: serde_json::Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();
        assert_eq!(parsed["error"]["type"], "invalid_request_error");
    }

    #[tokio::test]
    async fn on_request_body_non_eos_passthrough() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("project: p").unwrap();
        let filter = OpenaiChatCompletionsToVertexaiGeminiFilter::from_config(&yaml).unwrap();
        let req = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        let mut body = Some(Bytes::from_static(b"partial"));

        let action = filter.on_request_body(&mut ctx, &mut body, false).await.unwrap();

        assert!(matches!(action, FilterAction::Continue));
        assert!(ctx.rewritten_path.is_none(), "path should not be rewritten before EOS");
    }

    // --- on_response + on_response_body: error normalization ---

    #[tokio::test]
    async fn error_response_normalized_to_openai_envelope() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("project: p").unwrap();
        let filter = OpenaiChatCompletionsToVertexaiGeminiFilter::from_config(&yaml).unwrap();
        let req = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        let mut resp = make_response();
        resp.status = StatusCode::NOT_FOUND;
        resp.headers.insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("application/json"),
        );
        ctx.response_header = Some(&mut resp);

        drop(filter.on_response(&mut ctx).await.unwrap());

        assert!(
            matches!(ctx.response_body_mode, BodyMode::StreamBuffer { .. }),
            "error body should be buffered"
        );
        ctx.response_header = None;

        let mut body = Some(Bytes::from(
            br#"{"error":{"message":"Model not found","status":"NOT_FOUND"}}"#.to_vec(),
        ));
        let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();

        assert!(matches!(action, FilterAction::Continue));
        let parsed: serde_json::Value = serde_json::from_slice(body.unwrap().as_ref()).unwrap();
        assert!(parsed["error"]["message"].is_string());
        assert!(parsed["error"]["type"].is_string());
        assert!(parsed["error"]["code"].is_string());
    }

    // --- on_response + on_response_body: success translation ---

    #[tokio::test]
    async fn success_response_translated_to_chat_completions() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("project: p").unwrap();
        let filter = OpenaiChatCompletionsToVertexaiGeminiFilter::from_config(&yaml).unwrap();
        let req = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        ctx.set_metadata(REQUEST_MODEL_KEY, "gemini-2.0-flash".to_owned());

        let mut resp = make_response();
        resp.headers.insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("application/json"),
        );
        ctx.response_header = Some(&mut resp);

        drop(filter.on_response(&mut ctx).await.unwrap());

        assert!(
            matches!(ctx.response_body_mode, BodyMode::StreamBuffer { .. }),
            "success body should be buffered"
        );
        ctx.response_header = None;

        let gemini_response = br#"{
            "candidates": [{
                "content": {"parts": [{"text": "Hello!"}], "role": "model"},
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15
            }
        }"#;
        let mut body = Some(Bytes::from(gemini_response.to_vec()));
        let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();

        assert!(matches!(action, FilterAction::Continue));
        let parsed: serde_json::Value = serde_json::from_slice(body.unwrap().as_ref()).unwrap();
        assert_eq!(parsed["object"], "chat.completion");
        assert_eq!(parsed["model"], "gemini-2.0-flash");
        assert_eq!(parsed["choices"][0]["message"]["content"], "Hello!");
        assert_eq!(parsed["choices"][0]["finish_reason"], "stop");
        assert_eq!(parsed["usage"]["prompt_tokens"], 10);
    }

    #[tokio::test]
    async fn malformed_success_body_returns_error_envelope() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("project: p").unwrap();
        let filter = OpenaiChatCompletionsToVertexaiGeminiFilter::from_config(&yaml).unwrap();
        let req = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        ctx.set_metadata(REQUEST_MODEL_KEY, "gemini-2.0-flash".to_owned());
        ctx.set_metadata(RESPONSE_TRANSFORM_KEY, RESPONSE_TRANSFORM_SUCCESS.to_owned());

        let mut body = Some(Bytes::from_static(b"not json"));
        drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());

        let parsed: serde_json::Value = serde_json::from_slice(body.unwrap().as_ref()).unwrap();
        assert_eq!(parsed["error"]["type"], "server_error");
        assert_eq!(parsed["error"]["message"], "upstream response could not be translated");
    }

    // --- on_response + on_response_body: SSE streaming ---

    #[tokio::test]
    async fn sse_streaming_translates_frames() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("project: p").unwrap();
        let filter = OpenaiChatCompletionsToVertexaiGeminiFilter::from_config(&yaml).unwrap();
        let req = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        ctx.set_metadata(REQUEST_MODEL_KEY, "gemini-2.0-flash".to_owned());

        let mut resp = make_response();
        resp.headers.insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("text/event-stream"),
        );
        ctx.response_header = Some(&mut resp);

        drop(filter.on_response(&mut ctx).await.unwrap());

        assert_eq!(ctx.get_metadata(RESPONSE_TRANSFORM_KEY), Some(RESPONSE_TRANSFORM_SSE));
        ctx.response_header = None;

        // First SSE chunk with a complete Gemini frame.
        let sse_data = b"data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hi\"}],\"role\":\"model\"},\"finishReason\":\"STOP\"}]}\n\n";
        let mut body = Some(Bytes::from(sse_data.to_vec()));
        let action = filter.on_response_body(&mut ctx, &mut body, false).unwrap();

        assert!(matches!(action, FilterAction::Continue));
        let output_bytes = body.unwrap();
        let output = std::str::from_utf8(output_bytes.as_ref()).unwrap();
        assert!(output.starts_with("data: "), "output should be SSE-framed");

        let json_str = output.strip_prefix("data: ").unwrap().trim_end();
        let parsed: serde_json::Value = serde_json::from_str(json_str).unwrap();
        assert_eq!(parsed["object"], "chat.completion.chunk");
        assert_eq!(parsed["choices"][0]["delta"]["role"], "assistant");
    }

    #[tokio::test]
    async fn sse_end_of_stream_emits_done_sentinel() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("project: p").unwrap();
        let filter = OpenaiChatCompletionsToVertexaiGeminiFilter::from_config(&yaml).unwrap();
        let req = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        ctx.set_metadata(RESPONSE_TRANSFORM_KEY, RESPONSE_TRANSFORM_SSE.to_owned());

        // Gemini doesn't send [DONE] — the proxy emits it when the
        // connection closes (end_of_stream=true with empty body).
        let mut body = Some(Bytes::new());
        drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());

        let output_bytes = body.unwrap();
        let output = std::str::from_utf8(output_bytes.as_ref()).unwrap();
        assert_eq!(
            output, "data: [DONE]\n\n",
            "proxy must emit [DONE] for OpenAI clients when stream ends"
        );
    }

    #[tokio::test]
    async fn sse_last_chunk_includes_done_after_translation() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("project: p").unwrap();
        let filter = OpenaiChatCompletionsToVertexaiGeminiFilter::from_config(&yaml).unwrap();
        let req = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        ctx.set_metadata(REQUEST_MODEL_KEY, "gemini-2.0-flash".to_owned());
        ctx.set_metadata(RESPONSE_TRANSFORM_KEY, RESPONSE_TRANSFORM_SSE.to_owned());

        // Final chunk with data AND end_of_stream=true: should
        // translate the frame AND append [DONE].
        let sse_data = b"data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"!\"}],\"role\":\"model\"},\"finishReason\":\"STOP\"}]}\n\n";
        let mut body = Some(Bytes::from(sse_data.to_vec()));
        drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());

        let output_bytes = body.unwrap();
        let output = std::str::from_utf8(output_bytes.as_ref()).unwrap();
        assert!(output.contains("data: {"), "should contain translated frame");
        assert!(output.ends_with("data: [DONE]\n\n"), "should end with [DONE] sentinel");
    }

    // --- prepare_response_headers ---

    #[test]
    fn prepare_response_headers_sets_content_type() {
        let req = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        let mut resp = make_response();
        resp.headers
            .insert(http::header::CONTENT_LENGTH, http::HeaderValue::from_static("42"));
        resp.headers
            .insert(http::header::CONTENT_ENCODING, http::HeaderValue::from_static("gzip"));
        ctx.response_header = Some(&mut resp);

        prepare_response_headers(&mut ctx, "application/json");

        assert!(
            !ctx.response_header
                .as_ref()
                .unwrap()
                .headers
                .contains_key(http::header::CONTENT_LENGTH),
            "content-length should be removed"
        );
        assert!(
            !ctx.response_header
                .as_ref()
                .unwrap()
                .headers
                .contains_key(http::header::CONTENT_ENCODING),
            "content-encoding should be removed"
        );
        assert_eq!(
            ctx.response_header
                .as_ref()
                .unwrap()
                .headers
                .get(http::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok()),
            Some("application/json")
        );
        assert!(ctx.response_headers_modified);
    }
}
