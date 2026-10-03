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
/// OpenAI `stream_options.include_usage` — Vertex has no equivalent.
const REQUEST_INCLUDE_USAGE_KEY: &str = "vertex_gemini.include_usage";
/// Number of terminal streaming candidates expected for this request.
const REQUEST_CANDIDATE_COUNT_KEY: &str = "vertex_gemini.candidate_count";
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
    /// Sticky completion id and tool-call slots for this stream.
    translate: response::StreamTranslateState,
    /// Set when a parse or translation error was encountered during the
    /// stream. Checked at `end_of_stream` to emit an error frame.
    had_error: bool,
    /// Set when the upstream sends its own `[DONE]` sentinel.
    ///
    /// Vertex normally closes the connection instead. When it does send the
    /// sentinel, defer forwarding it until EOF validation has succeeded.
    provider_done: bool,
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
            return Ok(FilterAction::Reject(build_request_rejection("request body is empty")));
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
                if result.include_usage {
                    ctx.set_metadata(REQUEST_INCLUDE_USAGE_KEY, "true".to_owned());
                }
                ctx.set_metadata(REQUEST_CANDIDATE_COUNT_KEY, result.candidate_count.to_string());
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
                if let Some(rejection) = translate_success_body(ctx, body) {
                    return Ok(FilterAction::Reject(rejection));
                }
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
    let bytes = body.as_ref().filter(|b| !b.is_empty());

    // Nothing to parse and not closing — nothing to do.
    if bytes.is_none() && !end_of_stream {
        return;
    }

    let mut state = ctx
        .remove_filter_state::<StreamState>()
        .unwrap_or_else(|| new_stream_state(ctx, max_body_bytes));

    let mut output = match bytes {
        Some(b) => translate_sse_frames(ctx, &mut state, b),
        None => Vec::new(),
    };

    if end_of_stream {
        finish_sse_stream(ctx, &mut state, &mut output);
    } else {
        ctx.insert_filter_state(state);
    }

    *body = Some(if output.is_empty() {
        Bytes::new()
    } else {
        Bytes::from(output)
    });
}

/// Create a new stream state, extracting `include_usage` from request metadata.
fn new_stream_state(ctx: &HttpFilterContext<'_>, max_body_bytes: usize) -> StreamState {
    let mut translate = response::StreamTranslateState::new();
    translate.set_include_usage(ctx.get_metadata(REQUEST_INCLUDE_USAGE_KEY) == Some("true"));
    let candidate_count = ctx
        .get_metadata(REQUEST_CANDIDATE_COUNT_KEY)
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(1);
    translate.set_expected_candidate_count(candidate_count);
    StreamState {
        parser: SseFrameParser::new(max_body_bytes),
        translate,
        had_error: false,
        provider_done: false,
    }
}

/// Close an SSE stream.
///
/// Emits an error frame when `had_error` is set, when the byte parser has an
/// incomplete frame at EOF, or when not every requested candidate reached a
/// finish reason. Successful streams append usage (if requested) and `[DONE]`.
/// Failed streams end with the error frame and never emit a success sentinel.
fn finish_sse_stream(ctx: &HttpFilterContext<'_>, state: &mut StreamState, output: &mut Vec<u8>) {
    let truncated = !state.translate.is_complete() && !state.had_error;
    if state.had_error || state.parser.has_incomplete_frame() || truncated {
        append_stream_error_frame(output);
        return;
    }

    append_stream_usage_frame(ctx, state, output);
    output.extend_from_slice(SSE_DONE);
}

/// Append the final usage chunk to the output if `include_usage` is set and usage data exists.
fn append_stream_usage_frame(ctx: &HttpFilterContext<'_>, state: &mut StreamState, output: &mut Vec<u8>) {
    let model = ctx.get_metadata(REQUEST_MODEL_KEY).unwrap_or("unknown");
    if let Some(usage) = response::take_stream_usage_chunk(&mut state.translate, model) {
        output.extend_from_slice(b"data: ");
        output.extend_from_slice(&usage);
        output.extend_from_slice(b"\n\n");
    }
}

/// Parse and translate SSE frames from a raw byte chunk.
///
/// On parse error, marks `state.had_error` and returns an empty buffer.
/// Note: `SseFrameParser` fails closed — a `BufferOverflow` mid-chunk
/// discards the entire chunk, including any frames that completed before
/// the overflow. This is intentional: partial recovery from a corrupted
/// byte stream risks emitting truncated JSON to the client.
fn translate_sse_frames(ctx: &HttpFilterContext<'_>, state: &mut StreamState, bytes: &Bytes) -> Vec<u8> {
    // Sticky error: suppress all subsequent chunks once an error is set.
    if state.had_error {
        return Vec::new();
    }

    let model = ctx.get_metadata(REQUEST_MODEL_KEY).unwrap_or("unknown");

    match state.parser.parse_chunk(bytes) {
        Ok(frames) => render_sse_frames(&frames, model, state),
        Err(e) => {
            debug!(error = %e, "SSE parse error in vertex gemini filter");
            state.had_error = true;
            Vec::new()
        },
    }
}

/// Render parsed SSE frames into translated OpenAI SSE output bytes.
///
/// Stops on the first translation error and sets `state.had_error`; remaining
/// frames in the chunk are dropped. Later chunks are suppressed by the
/// early-return guard in `translate_sse_frames`.
fn render_sse_frames(frames: &[crate::openai::sse::SseFrame], model: &str, state: &mut StreamState) -> Vec<u8> {
    let mut output = Vec::new();
    for frame in frames {
        if frame.data == b"[DONE]" {
            state.provider_done = true;
            continue;
        }
        if state.provider_done {
            debug!("Gemini SSE frame received after provider [DONE]");
            state.had_error = true;
            break;
        }
        match response::transform_stream_chunk(&frame.data, model, &mut state.translate) {
            Ok(Some(translated)) => {
                output.extend_from_slice(b"data: ");
                output.extend_from_slice(&translated);
                output.extend_from_slice(b"\n\n");
            },
            Ok(None) => {},
            Err(e) => {
                debug!(error = e.as_str(), "failed to translate Gemini SSE frame");
                state.had_error = true;
                break;
            },
        }
    }
    output
}

/// Append an OpenAI-shaped error SSE frame to `output`.
///
/// Emitted as the terminal frame when the upstream stream contained parse or
/// translation errors, or when the connection closed before a complete
/// terminal response. Failed streams deliberately do not append `[DONE]`.
fn append_stream_error_frame(output: &mut Vec<u8>) {
    let frame = wire::build_openai_error_body(
        "upstream SSE stream ended with errors or was truncated",
        "server_error",
        "server_error",
    );
    output.extend_from_slice(b"data: ");
    output.extend_from_slice(&frame);
    output.extend_from_slice(b"\n\n");
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
/// Returns `None` on success (body is replaced in-place).
///
/// Returns `Some(Rejection)` when translation fails so the caller can
/// return a proper HTTP 500 instead of forwarding an error body with a
/// misleading HTTP 200 status. An HTTP 200 carrying an OpenAI error
/// envelope confuses SDK clients that key on the status code to decide
/// whether to raise.
fn translate_success_body(ctx: &HttpFilterContext<'_>, body: &mut Option<Bytes>) -> Option<Rejection> {
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
            None
        },
        Err(msg) => {
            warn!(error = msg.as_str(), "failed to translate Gemini response");
            Some(build_server_error_rejection(
                "upstream response could not be translated",
            ))
        },
    }
}

/// Build an OpenAI-shaped HTTP 500 rejection for internal translation failures.
fn build_server_error_rejection(message: &str) -> Rejection {
    let body = wire::build_openai_error_body(message, "server_error", "server_error");
    Rejection::status(500)
        .with_header("content-type", "application/json")
        .with_body(Bytes::from(body))
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
        assert_eq!(ctx.get_metadata(REQUEST_CANDIDATE_COUNT_KEY), Some("1"));
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

    #[tokio::test]
    async fn on_request_body_empty_at_eos_rejects() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("project: p").unwrap();
        let filter = OpenaiChatCompletionsToVertexaiGeminiFilter::from_config(&yaml).unwrap();
        let req = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        let mut body = Some(Bytes::new());

        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

        assert!(
            matches!(action, FilterAction::Reject(_)),
            "empty body at EOS should be rejected"
        );
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
    async fn malformed_success_body_returns_500_rejection() {
        // A Vertex HTTP 200 response that cannot be translated must produce
        // an HTTP 500 Rejection, not a 200 with an error body. An HTTP 200
        // carrying an error envelope confuses OpenAI SDK clients that key
        // on the status code to decide whether to raise.
        let yaml: serde_yaml::Value = serde_yaml::from_str("project: p").unwrap();
        let filter = OpenaiChatCompletionsToVertexaiGeminiFilter::from_config(&yaml).unwrap();
        let req = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        ctx.set_metadata(REQUEST_MODEL_KEY, "gemini-2.0-flash".to_owned());
        ctx.set_metadata(RESPONSE_TRANSFORM_KEY, RESPONSE_TRANSFORM_SUCCESS.to_owned());

        let mut body = Some(Bytes::from_static(b"not json"));
        let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();

        let FilterAction::Reject(rejection) = action else {
            panic!("expected Reject, got: {action:?}");
        };
        assert_eq!(rejection.status, 500, "translation failure must produce HTTP 500");
        let body_bytes = rejection.body.unwrap_or_default();
        let parsed: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
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
        ctx.set_metadata(REQUEST_MODEL_KEY, "gemini-2.0-flash".to_owned());
        ctx.set_metadata(RESPONSE_TRANSFORM_KEY, RESPONSE_TRANSFORM_SSE.to_owned());
        ctx.current_filter_id = Some(0);

        // First deliver a content frame with finishReason so the stream is
        // considered complete. Gemini doesn't send [DONE] — the proxy appends
        // it when the connection closes (end_of_stream=true with empty body).
        let sse_data =
            b"data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"hi\"}]},\"finishReason\":\"STOP\"}]}\n\n";
        let mut body = Some(Bytes::from(sse_data.to_vec()));
        drop(filter.on_response_body(&mut ctx, &mut body, false).unwrap());

        let mut body = Some(Bytes::new());
        drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());

        let output_bytes = body.unwrap();
        let output = std::str::from_utf8(output_bytes.as_ref()).unwrap();
        assert_eq!(
            output, "data: [DONE]\n\n",
            "proxy must emit [DONE] for OpenAI clients when stream ends cleanly"
        );
    }

    #[tokio::test]
    async fn sse_eof_without_finish_reason_emits_error_frame() {
        // A stream that closes cleanly (no parse error, no incomplete frame)
        // but never delivered a finishReason must be treated as truncated.
        let yaml: serde_yaml::Value = serde_yaml::from_str("project: p").unwrap();
        let filter = OpenaiChatCompletionsToVertexaiGeminiFilter::from_config(&yaml).unwrap();
        let req = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        ctx.set_metadata(REQUEST_MODEL_KEY, "gemini-2.0-flash".to_owned());
        ctx.set_metadata(RESPONSE_TRANSFORM_KEY, RESPONSE_TRANSFORM_SSE.to_owned());
        ctx.current_filter_id = Some(0);

        // Valid JSON frame, no parse error — but no finishReason.
        let mut body = Some(Bytes::from_static(
            b"data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hi\"}]}}]}\n\n",
        ));
        drop(filter.on_response_body(&mut ctx, &mut body, false).unwrap());

        // Stream closes cleanly.
        let mut body = Some(Bytes::new());
        drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());
        let output = std::str::from_utf8(body.as_ref().unwrap()).unwrap();

        assert!(
            output.contains(r#""error"#),
            "truncated stream (no finishReason) should emit an error frame, got: {output}"
        );
        assert!(output.contains("server_error"), "error type should be server_error");
        assert!(!output.contains("[DONE]"), "failed stream must not emit [DONE]");
    }

    #[tokio::test]
    async fn sse_eof_requires_every_requested_candidate_to_finish() {
        let filter = make_filter("project: p");
        let req = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        ctx.current_filter_id = Some(0);

        let mut body = Some(Bytes::from_static(
            br#"{"model":"gemini-2.0-flash","n":2,"stream":true,"messages":[{"role":"user","content":"Hi"}]}"#,
        ));
        drop(filter.on_request_body(&mut ctx, &mut body, true).await.unwrap());
        ctx.set_metadata(RESPONSE_TRANSFORM_KEY, RESPONSE_TRANSFORM_SSE);

        // Candidate zero finishes, but candidate one never arrives.
        let mut body = Some(Bytes::from_static(
            b"data: {\"candidates\":[{\"index\":0,\"content\":{\"parts\":[{\"text\":\"Hi\"}]},\"finishReason\":\"STOP\"}]}\n\n",
        ));
        drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());
        let output = std::str::from_utf8(body.as_ref().unwrap()).unwrap();

        assert!(
            output.contains(r#""error"#),
            "partial candidate set must fail: {output}"
        );
        assert!(!output.contains("[DONE]"), "partial candidate set must not succeed");
    }

    #[tokio::test]
    async fn upstream_done_is_deferred_until_stream_validation() {
        let filter = make_filter("project: p");
        let req = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        ctx.set_metadata(REQUEST_MODEL_KEY, "gemini-2.0-flash");
        ctx.set_metadata(RESPONSE_TRANSFORM_KEY, RESPONSE_TRANSFORM_SSE);
        ctx.current_filter_id = Some(0);

        let mut body = Some(Bytes::from_static(
            b"data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hi\"}]}}]}\n\ndata: [DONE]\n\n",
        ));
        drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());
        let output = std::str::from_utf8(body.as_ref().unwrap()).unwrap();

        assert!(output.contains(r#""error"#), "premature [DONE] must fail: {output}");
        assert!(
            !output.contains("data: [DONE]"),
            "provider [DONE] must not bypass validation"
        );
    }

    #[tokio::test]
    async fn sse_had_error_in_earlier_chunk_suppresses_later_chunks() {
        // Error state is sticky: a translation error in chunk N must suppress
        // all output from chunk N+1 onwards, not just the rest of chunk N.
        let yaml: serde_yaml::Value = serde_yaml::from_str("project: p").unwrap();
        let filter = OpenaiChatCompletionsToVertexaiGeminiFilter::from_config(&yaml).unwrap();
        let req = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        ctx.set_metadata(REQUEST_MODEL_KEY, "gemini-2.0-flash".to_owned());
        ctx.set_metadata(RESPONSE_TRANSFORM_KEY, RESPONSE_TRANSFORM_SSE.to_owned());
        ctx.current_filter_id = Some(0);

        // Chunk 1: invalid JSON frame — sets had_error.
        let mut body = Some(Bytes::from_static(b"data: not-json-at-all\n\n"));
        drop(filter.on_response_body(&mut ctx, &mut body, false).unwrap());

        // Chunk 2: valid frame — must be suppressed because had_error is sticky.
        let mut body = Some(Bytes::from_static(
            b"data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hi\"}],\"role\":\"model\"},\"finishReason\":\"STOP\"}]}\n\n",
        ));
        drop(filter.on_response_body(&mut ctx, &mut body, false).unwrap());
        let mid_output = std::str::from_utf8(body.as_ref().unwrap()).unwrap();
        assert!(
            !mid_output.contains("chat.completion.chunk"),
            "valid frame in a later chunk after an error must be suppressed, got: {mid_output}"
        );

        // End the stream — error frame expected.
        let mut body = Some(Bytes::new());
        drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());
        let output = std::str::from_utf8(body.as_ref().unwrap()).unwrap();
        assert!(
            output.contains(r#""error"#),
            "error frame expected after sticky error, got: {output}"
        );
        assert!(!output.contains("[DONE]"), "failed stream must not emit [DONE]");
    }

    #[tokio::test]
    async fn sse_last_chunk_includes_done_after_translation() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("project: p").unwrap();
        let filter = OpenaiChatCompletionsToVertexaiGeminiFilter::from_config(&yaml).unwrap();
        let req = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        ctx.set_metadata(REQUEST_MODEL_KEY, "gemini-2.0-flash".to_owned());
        ctx.set_metadata(RESPONSE_TRANSFORM_KEY, RESPONSE_TRANSFORM_SSE.to_owned());
        ctx.current_filter_id = Some(0);

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

    #[tokio::test]
    async fn sse_include_usage_emits_usage_chunk_before_done() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("project: p").unwrap();
        let filter = OpenaiChatCompletionsToVertexaiGeminiFilter::from_config(&yaml).unwrap();
        let req = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        ctx.current_filter_id = Some(0);

        let mut body = Some(Bytes::from_static(
            br#"{"model":"gemini-2.0-flash","stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":"Hi"}]}"#,
        ));
        drop(filter.on_request_body(&mut ctx, &mut body, true).await.unwrap());
        ctx.set_metadata(RESPONSE_TRANSFORM_KEY, RESPONSE_TRANSFORM_SSE.to_owned());

        let mut body = Some(Bytes::from_static(
            b"data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"!\"}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":3,\"candidatesTokenCount\":1,\"totalTokenCount\":4}}\n\n",
        ));
        drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());
        let output = std::str::from_utf8(body.as_ref().unwrap()).unwrap();
        let payloads = parse_sse_json_payloads(output);

        assert_eq!(payloads.len(), 2, "content chunk then usage chunk, got: {output}");
        assert_eq!(payloads[0]["choices"][0]["finish_reason"], "stop");
        assert!(payloads[0]["usage"].is_null());
        assert_eq!(payloads[1]["choices"], serde_json::json!([]));
        assert_eq!(payloads[1]["usage"]["total_tokens"], 4);
        assert!(output.ends_with("data: [DONE]\n\n"));
    }

    /// Vertex sometimes sends `usageMetadata` on a *separate* usage-only
    /// frame after the last content frame rather than piggybacking it on
    /// the final content frame.  Verify the filter correctly stashes
    /// the usage across the chunk boundary and emits the trailing chunk.
    #[tokio::test]
    async fn sse_include_usage_separate_usage_frame() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("project: p").unwrap();
        let filter = OpenaiChatCompletionsToVertexaiGeminiFilter::from_config(&yaml).unwrap();
        let req = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        ctx.current_filter_id = Some(0);

        // Request — sets REQUEST_MODEL_KEY and REQUEST_INCLUDE_USAGE_KEY in ctx.
        let mut body = Some(Bytes::from_static(
            br#"{"model":"gemini-2.0-flash","stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":"Hi"}]}"#,
        ));
        drop(filter.on_request_body(&mut ctx, &mut body, true).await.unwrap());
        ctx.set_metadata(RESPONSE_TRANSFORM_KEY, RESPONSE_TRANSFORM_SSE.to_owned());

        // First chunk — content frame without usageMetadata, not end-of-stream.
        let mut body = Some(Bytes::from_static(
            b"data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"hello\"}]},\"finishReason\":\"STOP\"}]}\n\n",
        ));
        drop(filter.on_response_body(&mut ctx, &mut body, false).unwrap());
        let first_output = std::str::from_utf8(body.as_ref().unwrap()).unwrap().to_owned();
        let first_payloads = parse_sse_json_payloads(&first_output);
        assert_eq!(
            first_payloads.len(),
            1,
            "first chunk: one content frame, got: {first_output}"
        );
        assert!(
            first_payloads[0]["usage"].is_null(),
            "content chunk must carry usage: null"
        );

        // Second chunk — usage-only frame, end-of-stream.
        let mut body = Some(Bytes::from_static(
            b"data: {\"usageMetadata\":{\"promptTokenCount\":5,\"candidatesTokenCount\":2,\"totalTokenCount\":7}}\n\n",
        ));
        drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());
        let second_output = std::str::from_utf8(body.as_ref().unwrap()).unwrap().to_owned();
        let second_payloads = parse_sse_json_payloads(&second_output);

        assert_eq!(
            second_payloads.len(),
            1,
            "second chunk: usage chunk only (usage-only frame suppressed), got: {second_output}"
        );
        assert_eq!(second_payloads[0]["choices"], serde_json::json!([]));
        assert_eq!(second_payloads[0]["usage"]["prompt_tokens"], 5);
        assert_eq!(second_payloads[0]["usage"]["completion_tokens"], 2);
        assert_eq!(second_payloads[0]["usage"]["total_tokens"], 7);
        assert_eq!(
            second_payloads[0]["id"], first_payloads[0]["id"],
            "completion id must be stable"
        );
        assert!(second_output.ends_with("data: [DONE]\n\n"));
    }

    fn parse_sse_json_payloads(output: &str) -> Vec<serde_json::Value> {
        output
            .split("\n\n")
            .filter_map(|frame| {
                let data = frame.strip_prefix("data: ")?;
                if data.trim() == "[DONE]" {
                    return None;
                }
                serde_json::from_str(data.trim()).ok()
            })
            .collect()
    }

    #[tokio::test]
    async fn sse_two_frames_assign_stable_tool_call_indices() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("project: p").unwrap();
        let filter = OpenaiChatCompletionsToVertexaiGeminiFilter::from_config(&yaml).unwrap();
        let req = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        ctx.set_metadata(REQUEST_MODEL_KEY, "gemini-2.0-flash".to_owned());
        ctx.set_metadata(RESPONSE_TRANSFORM_KEY, RESPONSE_TRANSFORM_SSE.to_owned());
        ctx.current_filter_id = Some(0);

        let mut body = Some(Bytes::from_static(
            b"data: {\"responseId\":\"resp-s\",\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"name\":\"get_weather\",\"args\":{}}}]}}]}\n\n",
        ));
        drop(filter.on_response_body(&mut ctx, &mut body, false).unwrap());
        let first = parse_sse_json_payloads(std::str::from_utf8(body.as_ref().unwrap()).unwrap());

        let mut body = Some(Bytes::from_static(
            b"data: {\"responseId\":\"resp-s\",\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"name\":\"get_calendar\",\"args\":{}}}]}}]}\n\n",
        ));
        drop(filter.on_response_body(&mut ctx, &mut body, false).unwrap());
        let second = parse_sse_json_payloads(std::str::from_utf8(body.as_ref().unwrap()).unwrap());

        let weather = &first[0]["choices"][0]["delta"]["tool_calls"][0];
        let calendar = &second[0]["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!(weather["index"], 0);
        assert_eq!(calendar["index"], 1);
        assert_ne!(weather["id"], calendar["id"]);
        assert_eq!(first[0]["id"], second[0]["id"]);
        assert_eq!(first[0]["id"], "resp-s");
    }

    #[tokio::test]
    async fn sse_google_function_call_id_survives_across_frames() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("project: p").unwrap();
        let filter = OpenaiChatCompletionsToVertexaiGeminiFilter::from_config(&yaml).unwrap();
        let req = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        ctx.set_metadata(REQUEST_MODEL_KEY, "gemini-2.0-flash".to_owned());
        ctx.set_metadata(RESPONSE_TRANSFORM_KEY, RESPONSE_TRANSFORM_SSE.to_owned());
        ctx.current_filter_id = Some(0);

        let mut body = Some(Bytes::from_static(
            b"data: {\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"id\":\"fc-keep\",\"name\":\"search\"}}]}}]}\n\n",
        ));
        drop(filter.on_response_body(&mut ctx, &mut body, false).unwrap());
        let first = parse_sse_json_payloads(std::str::from_utf8(body.as_ref().unwrap()).unwrap());

        let mut body = Some(Bytes::from_static(
            b"data: {\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"id\":\"fc-keep\",\"args\":{\"q\":\"hi\"}}}]}}]}\n\n",
        ));
        drop(filter.on_response_body(&mut ctx, &mut body, false).unwrap());
        let second = parse_sse_json_payloads(std::str::from_utf8(body.as_ref().unwrap()).unwrap());

        assert_eq!(first[0]["choices"][0]["delta"]["tool_calls"][0]["id"], "fc-keep");
        assert_eq!(first[0]["choices"][0]["delta"]["tool_calls"][0]["index"], 0);
        assert_eq!(second[0]["choices"][0]["delta"]["tool_calls"][0]["index"], 0);
        assert!(second[0]["choices"][0]["delta"]["tool_calls"][0].get("id").is_none());
        assert_eq!(
            second[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"],
            r#"{"q":"hi"}"#
        );
    }

    // --- SSE error propagation ---

    #[tokio::test]
    async fn sse_parse_error_emits_terminal_error_without_done() {
        // Use a tiny max_body_bytes so the SSE parser actually hits
        // BufferOverflow — the default (1 MiB) would never overflow
        // on a small test payload.
        let yaml: serde_yaml::Value = serde_yaml::from_str("project: p\nmax_body_bytes: 64").unwrap();
        let filter = OpenaiChatCompletionsToVertexaiGeminiFilter::from_config(&yaml).unwrap();
        let req = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        ctx.set_metadata(REQUEST_MODEL_KEY, "gemini-2.0-flash".to_owned());
        ctx.set_metadata(RESPONSE_TRANSFORM_KEY, RESPONSE_TRANSFORM_SSE.to_owned());
        ctx.current_filter_id = Some(0);

        // 100 bytes exceeds the 64-byte parser buffer → BufferOverflow.
        let overflow = "x".repeat(100);
        let huge_frame = format!("data: {overflow}\n\n");
        let mut body = Some(Bytes::from(huge_frame));
        drop(filter.on_response_body(&mut ctx, &mut body, false).unwrap());

        // Parser error was logged — no translated frames emitted this chunk.
        let output = std::str::from_utf8(body.as_ref().unwrap()).unwrap();
        assert!(
            !output.contains("chat.completion.chunk"),
            "broken frame should not be translated"
        );

        // End the stream.
        let mut body = Some(Bytes::new());
        drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());
        let output = std::str::from_utf8(body.as_ref().unwrap()).unwrap();

        assert!(
            output.contains(r#""error"#),
            "should emit an error frame, got: {output}"
        );
        assert!(output.contains("server_error"), "error type should be server_error");
        assert!(!output.contains("[DONE]"), "failed stream must not emit [DONE]");
    }

    #[tokio::test]
    async fn sse_translate_error_emits_terminal_error_without_done() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("project: p").unwrap();
        let filter = OpenaiChatCompletionsToVertexaiGeminiFilter::from_config(&yaml).unwrap();
        let req = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        ctx.set_metadata(REQUEST_MODEL_KEY, "gemini-2.0-flash".to_owned());
        ctx.set_metadata(RESPONSE_TRANSFORM_KEY, RESPONSE_TRANSFORM_SSE.to_owned());
        ctx.current_filter_id = Some(0);

        // Valid SSE frame wrapping invalid JSON — transform_stream_chunk returns Err.
        let mut body = Some(Bytes::from_static(b"data: not-json-at-all\n\n"));
        drop(filter.on_response_body(&mut ctx, &mut body, false).unwrap());

        // Close the stream.
        let mut body = Some(Bytes::new());
        drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());
        let output = std::str::from_utf8(body.as_ref().unwrap()).unwrap();

        assert!(
            output.contains(r#""error"#),
            "should emit an error frame, got: {output}"
        );
        assert!(!output.contains("[DONE]"), "failed stream must not emit [DONE]");
    }

    #[tokio::test]
    async fn sse_frames_after_error_in_same_chunk_are_not_emitted() {
        // A chunk that contains an invalid frame followed by a valid one.
        // Once the first frame errors, the loop must break — the valid
        // frame that follows must NOT be emitted as translated output.
        let yaml: serde_yaml::Value = serde_yaml::from_str("project: p").unwrap();
        let filter = OpenaiChatCompletionsToVertexaiGeminiFilter::from_config(&yaml).unwrap();
        let req = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        ctx.set_metadata(REQUEST_MODEL_KEY, "gemini-2.0-flash".to_owned());
        ctx.set_metadata(RESPONSE_TRANSFORM_KEY, RESPONSE_TRANSFORM_SSE.to_owned());
        ctx.current_filter_id = Some(0);

        // Two SSE frames in one chunk: bad JSON first, then a valid candidate.
        let chunk = b"data: not-valid-json\n\ndata: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hi\"}],\"role\":\"model\"}}]}\n\n";
        let mut body = Some(Bytes::from(chunk.as_slice()));
        drop(filter.on_response_body(&mut ctx, &mut body, false).unwrap());

        let mid_output = std::str::from_utf8(body.as_ref().unwrap()).unwrap();
        assert!(
            !mid_output.contains("chat.completion.chunk"),
            "valid frame following an error must not be emitted, got: {mid_output}"
        );

        // End the stream — must emit an error frame without a success sentinel.
        let mut body = Some(Bytes::new());
        drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());
        let output = std::str::from_utf8(body.as_ref().unwrap()).unwrap();
        assert!(output.contains(r#""error"#), "error frame expected, got: {output}");
        assert!(!output.contains("[DONE]"), "failed stream must not emit [DONE]");
    }

    #[tokio::test]
    async fn sse_incomplete_eof_emits_terminal_error_without_done() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("project: p").unwrap();
        let filter = OpenaiChatCompletionsToVertexaiGeminiFilter::from_config(&yaml).unwrap();
        let req = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        ctx.set_metadata(REQUEST_MODEL_KEY, "gemini-2.0-flash".to_owned());
        ctx.set_metadata(RESPONSE_TRANSFORM_KEY, RESPONSE_TRANSFORM_SSE.to_owned());
        ctx.current_filter_id = Some(0);

        // Send a partial SSE frame (no blank-line terminator).
        let mut body = Some(Bytes::from_static(
            b"data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hi\"}]}}]}",
        ));
        drop(filter.on_response_body(&mut ctx, &mut body, false).unwrap());

        // End the stream — parser still has bytes in its line buffer.
        let mut body = Some(Bytes::new());
        drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());
        let output = std::str::from_utf8(body.as_ref().unwrap()).unwrap();

        assert!(
            output.contains(r#""error"#),
            "incomplete frame at EOF should produce an error frame, got: {output}"
        );
        assert!(!output.contains("[DONE]"), "failed stream must not emit [DONE]");
    }

    #[tokio::test]
    async fn sse_clean_stream_has_no_error_frame() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("project: p").unwrap();
        let filter = OpenaiChatCompletionsToVertexaiGeminiFilter::from_config(&yaml).unwrap();
        let req = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        ctx.set_metadata(REQUEST_MODEL_KEY, "gemini-2.0-flash".to_owned());
        ctx.set_metadata(RESPONSE_TRANSFORM_KEY, RESPONSE_TRANSFORM_SSE.to_owned());
        ctx.current_filter_id = Some(0);

        let sse_data = b"data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hi\"}],\"role\":\"model\"},\"finishReason\":\"STOP\"}]}\n\n";
        let mut body = Some(Bytes::from(sse_data.to_vec()));
        drop(filter.on_response_body(&mut ctx, &mut body, false).unwrap());

        // Close the stream cleanly.
        let mut body = Some(Bytes::new());
        drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());
        let output = std::str::from_utf8(body.as_ref().unwrap()).unwrap();

        assert!(
            !output.contains(r#""error"#),
            "clean stream should not contain an error frame, got: {output}"
        );
        assert_eq!(output, "data: [DONE]\n\n", "clean EOF should be exactly [DONE]");
    }

    #[tokio::test]
    async fn sse_error_after_valid_frames_preserves_good_data() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("project: p").unwrap();
        let filter = OpenaiChatCompletionsToVertexaiGeminiFilter::from_config(&yaml).unwrap();
        let req = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        ctx.set_metadata(REQUEST_MODEL_KEY, "gemini-2.0-flash".to_owned());
        ctx.set_metadata(RESPONSE_TRANSFORM_KEY, RESPONSE_TRANSFORM_SSE.to_owned());
        ctx.current_filter_id = Some(0);

        // First chunk: valid frame.
        let sse_data = b"data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hi\"}],\"role\":\"model\"},\"finishReason\":\"STOP\"}]}\n\n";
        let mut body = Some(Bytes::from(sse_data.to_vec()));
        drop(filter.on_response_body(&mut ctx, &mut body, false).unwrap());
        let first_output = std::str::from_utf8(body.as_ref().unwrap()).unwrap();
        assert!(
            first_output.contains("chat.completion.chunk"),
            "valid frame should translate"
        );

        // Second chunk: Vertex error frame (quota exceeded mid-stream).
        let mut body = Some(Bytes::from_static(
            b"data: {\"error\":{\"code\":429,\"message\":\"Quota exceeded\",\"status\":\"RESOURCE_EXHAUSTED\"}}\n\n",
        ));
        drop(filter.on_response_body(&mut ctx, &mut body, false).unwrap());

        // Close the stream.
        let mut body = Some(Bytes::new());
        drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());
        let output = std::str::from_utf8(body.as_ref().unwrap()).unwrap();

        assert!(
            output.contains(r#""error"#),
            "stream with a Vertex error should end with an error frame, got: {output}"
        );
        assert!(!output.contains("[DONE]"), "failed stream must not emit [DONE]");
    }

    #[tokio::test]
    async fn sse_prompt_blocked_emits_terminal_error_without_done() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("project: p").unwrap();
        let filter = OpenaiChatCompletionsToVertexaiGeminiFilter::from_config(&yaml).unwrap();
        let req = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        ctx.set_metadata(REQUEST_MODEL_KEY, "gemini-2.0-flash".to_owned());
        ctx.set_metadata(RESPONSE_TRANSFORM_KEY, RESPONSE_TRANSFORM_SSE.to_owned());
        ctx.current_filter_id = Some(0);

        // Vertex sends promptFeedback with no candidates when the prompt is blocked.
        let mut body = Some(Bytes::from_static(
            b"data: {\"promptFeedback\":{\"blockReason\":\"SAFETY\",\"blockReasonMessage\":\"prompt violates policy\"}}\n\n",
        ));
        drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());
        let output = std::str::from_utf8(body.as_ref().unwrap()).unwrap();

        assert!(
            output.contains(r#""error"#),
            "prompt block should produce an error frame, got: {output}"
        );
        assert!(output.contains("server_error"), "error type should be server_error");
        assert!(!output.contains("[DONE]"), "failed stream must not emit [DONE]");
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
