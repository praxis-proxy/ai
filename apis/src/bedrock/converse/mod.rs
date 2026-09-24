// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! OpenAI Chat Completions to AWS Bedrock Converse translation filter.
//!
//! Rewrites Chat Completions request bodies to the Bedrock Converse
//! shape, translates non-streaming responses back to Chat Completions
//! format, and normalizes Bedrock error envelopes. Streaming responses
//! arrive as AWS `EventStream` binary frames and are translated to OpenAI
//! SSE (`text/event-stream`) inline.
//!
//! ## Request path
//!
//! 1. `on_request` — strip `Accept-Encoding` and `Authorization` so the downstream `aws_sigv4_sign` filter can inject
//!    the correct AWS credential header.
//! 2. `on_request_body` — translate the Chat Completions JSON to the Bedrock Converse shape, set `ctx.rewritten_path`
//!    to the correct `/model/{id}/converse[-stream]` endpoint, and persist the model name for the response phase.
//!
//! ## Response path (non-streaming)
//!
//! `on_response` arms the filter and switches to `StreamBuffer` mode.
//! `on_response_body` translates the full buffered body back to a Chat
//! Completions response once `end_of_stream` is true.
//!
//! ## Response path (streaming)
//!
//! `on_response` rewrites the `Content-Type` to `text/event-stream` and
//! removes `Content-Length`.  `on_response_body` feeds each chunk into
//! [`EventStreamDecoder`] and emits one `data: {json}\n\n` SSE frame
//! per decoded Bedrock event. A `data: [DONE]\n\n` sentinel is appended
//! at `end_of_stream`.
//!
//! ## Error path
//!
//! `on_response` detects non-2xx status codes, switches to
//! `StreamBuffer`, and buffers the response. `on_response_body`
//! normalizes Bedrock's `{"message":"..."}` body into an OpenAI-shaped
//! error envelope.

pub(crate) mod config;
pub(crate) mod request;
pub(crate) mod response;

use async_trait::async_trait;
use bytes::Bytes;
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, Rejection, parse_filter_config,
};
use tracing::{debug, warn};

use self::{
    config::{BedrockConverseConfig, FILTER_NAME, build_config},
    request::transform_request,
    response::{StreamState, transform_response, transform_stream_event},
};
use crate::bedrock::{
    eventstream::EventStreamDecoder,
    wire::{build_openai_error_body, normalize_error_response},
};

// -----------------------------------------------------------------------------
// Metadata keys (scoped to avoid collisions with other filters)
// -----------------------------------------------------------------------------

/// Model extracted from the Chat Completions request body.
const MODEL_KEY: &str = "bedrock_converse.model";

/// Synthetic response ID generated at request time and shared across all
/// streaming chunks so each SSE frame carries the same `id` field.
const RESPONSE_ID_KEY: &str = "bedrock_converse.response_id";

/// Discriminates which `on_response_body` branch to execute.
const RESPONSE_PATH_KEY: &str = "bedrock_converse.response_path";
/// Response-path variant for upstream error responses.
const RESPONSE_PATH_ERROR: &str = "error";
/// Response-path variant for non-streaming success responses.
const RESPONSE_PATH_SUCCESS: &str = "success";
/// Response-path variant for binary `EventStream` streaming responses.
const RESPONSE_PATH_STREAM: &str = "stream";

/// HTTP status code saved from `on_response` for use in error body
/// normalization.
const RESPONSE_STATUS_KEY: &str = "bedrock_converse.response_status";

// -----------------------------------------------------------------------------
// SSE framing
// -----------------------------------------------------------------------------

/// `data: [DONE]\n\n` — sent as the final SSE frame of a streaming response.
const SSE_DONE: &[u8] = b"data: [DONE]\n\n";

// -----------------------------------------------------------------------------
// Per-request streaming state
// -----------------------------------------------------------------------------

/// State persisted across `on_response_body` calls for a single streaming
/// response.  Stored via [`HttpFilterContext::insert_filter_state`] so it
/// survives multiple chunk deliveries.
struct BedrockStreamingState {
    /// Incremental binary frame decoder.  Holds any partial frame bytes
    /// that spanned a chunk boundary.
    decoder: EventStreamDecoder,
    /// OpenAI SSE state (role flag, tool-call index, finish reason).
    stream_state: StreamState,
    /// Whether an upstream exception or framing error has already been
    /// surfaced to the client.
    failed: bool,
}

// `BytesMut` is `Send + Sync`, and so is `EventStreamDecoder`; Rust
// derives those impls automatically for `BedrockStreamingState`.

// -----------------------------------------------------------------------------
// OpenaiChatCompletionsToBedrockConverseFilter
// -----------------------------------------------------------------------------

/// Translates OpenAI Chat Completions requests to AWS Bedrock Converse
/// format and translates responses back.
///
/// # YAML
///
/// ```yaml
/// filter: openai_chat_completions_to_bedrock_converse
/// ```
///
/// # Full YAML
///
/// ```yaml
/// filter: openai_chat_completions_to_bedrock_converse
/// max_body_bytes: 4194304
/// ```
pub struct OpenaiChatCompletionsToBedrockConverseFilter {
    /// Parsed and validated configuration.
    config: BedrockConverseConfig,
}

impl OpenaiChatCompletionsToBedrockConverseFilter {
    /// Create a filter from parsed YAML config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the YAML config is invalid.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: BedrockConverseConfig = parse_filter_config(FILTER_NAME, config)?;
        let validated = build_config(cfg)?;
        Ok(Box::new(Self { config: validated }))
    }
}

#[async_trait]
#[expect(
    clippy::too_many_lines,
    reason = "implements all four HttpFilter phases; splitting the trait impl would scatter the pipeline logic"
)]
impl HttpFilter for OpenaiChatCompletionsToBedrockConverseFilter {
    fn name(&self) -> &'static str {
        FILTER_NAME
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
        // Default to streaming; `on_response` upgrades to `StreamBuffer`
        // for non-streaming success responses and all error responses.
        BodyMode::Stream
    }

    // ── Request phase ──────────────────────────────────────────────────────

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        // Strip Accept-Encoding so the upstream binary EventStream body is
        // never compressed by a Bedrock-side proxy.
        ctx.request_headers_to_remove.push(http::header::ACCEPT_ENCODING);

        // Strip any client-side Authorization header; the pipeline's
        // aws_sigv4_sign filter will inject the correct AWS credential.
        ctx.request_headers_to_remove.push(http::header::AUTHORIZATION);

        Ok(FilterAction::Continue)
    }

    async fn on_request_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        // `StreamBuffer` mode delivers the full body on the final call.
        if !end_of_stream {
            return Ok(FilterAction::Continue);
        }

        let bytes = match body.as_ref() {
            Some(b) if !b.is_empty() => b,
            _ => return Ok(invalid_request_action("request body is empty")),
        };

        let result = match transform_request(bytes) {
            Ok(result) => result,
            Err(error) => return Ok(invalid_request_action(&error)),
        };

        // Persist model name for the response phase.
        ctx.set_metadata(MODEL_KEY, result.model.clone());

        // Generate a stable response ID used by all streaming SSE frames.
        let response_id = format!("chatcmpl-{:x}", response_id_seed());
        ctx.set_metadata(RESPONSE_ID_KEY, response_id);

        // Set the Bedrock Converse endpoint path.
        // `aws_sigv4_sign` reads `ctx.rewritten_path` to sign the
        // correct canonical URI.
        let endpoint = if result.stream { "converse-stream" } else { "converse" };
        ctx.rewritten_path = Some(format!("/model/{}/{endpoint}", result.model));

        debug!(
            model = %result.model,
            streaming = result.stream,
            path = ctx.rewritten_path.as_deref().unwrap_or(""),
            "Bedrock Converse request translated"
        );

        *body = Some(Bytes::from(result.body));
        Ok(FilterAction::Continue)
    }

    // ── Response phase ─────────────────────────────────────────────────────

    async fn on_response(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        // Read status with a short-lived immutable borrow so subsequent
        // ctx method calls (set_metadata, set_response_body_mode) can
        // take ownership of the mutable reference without a conflict.
        let status = match ctx.response_header.as_ref() {
            Some(r) => r.status,
            None => return Ok(FilterAction::Continue),
        };

        if !status.is_success() {
            // Bedrock error: buffer full body and normalize in the body phase.
            ctx.set_metadata(RESPONSE_PATH_KEY, RESPONSE_PATH_ERROR);
            ctx.set_metadata(RESPONSE_STATUS_KEY, status.as_u16().to_string());
            ctx.set_response_body_mode(BodyMode::StreamBuffer {
                max_bytes: Some(self.config.max_body_bytes),
            });
            if let Some(resp) = ctx.response_header.as_mut() {
                resp.headers.remove(http::header::CONTENT_LENGTH);
            }
            ctx.response_headers_modified = true;
            return Ok(FilterAction::Continue);
        }

        // Check content-type with another short-lived immutable borrow.
        let is_stream = ctx
            .response_header
            .as_ref()
            .and_then(|r| r.headers.get(http::header::CONTENT_TYPE))
            .and_then(|v| v.to_str().ok())
            .is_some_and(|ct| ct.contains("application/vnd.amazon.eventstream"));

        if is_stream {
            // Streaming: rewrite Content-Type to the OpenAI SSE format.
            ctx.set_metadata(RESPONSE_PATH_KEY, RESPONSE_PATH_STREAM);
            if let Some(resp) = ctx.response_header.as_mut() {
                resp.headers.remove(http::header::CONTENT_LENGTH);
                resp.headers.remove(http::header::CONTENT_ENCODING);
                resp.headers.insert(
                    http::header::CONTENT_TYPE,
                    http::HeaderValue::from_static("text/event-stream"),
                );
            }
            ctx.response_headers_modified = true;
        } else {
            // Non-streaming success: buffer full body.
            ctx.set_metadata(RESPONSE_PATH_KEY, RESPONSE_PATH_SUCCESS);
            ctx.set_response_body_mode(BodyMode::StreamBuffer {
                max_bytes: Some(self.config.max_body_bytes),
            });
            if let Some(resp) = ctx.response_header.as_mut() {
                resp.headers.remove(http::header::CONTENT_LENGTH);
            }
            ctx.response_headers_modified = true;
        }

        Ok(FilterAction::Continue)
    }

    fn on_response_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        match ctx.get_metadata(RESPONSE_PATH_KEY) {
            Some(RESPONSE_PATH_ERROR) => Ok(translate_error_body(ctx, body, end_of_stream)),
            Some(RESPONSE_PATH_SUCCESS) => Ok(translate_success_body(ctx, body, end_of_stream)),
            Some(RESPONSE_PATH_STREAM) => Ok(translate_stream_chunk(
                ctx,
                body,
                end_of_stream,
                self.config.max_body_bytes,
            )),
            _ => Ok(FilterAction::Continue),
        }
    }
}

// -----------------------------------------------------------------------------
// Response body helpers
// -----------------------------------------------------------------------------

/// Buffer-and-replace the body with an OpenAI-shaped error envelope.
fn translate_error_body(ctx: &HttpFilterContext<'_>, body: &mut Option<Bytes>, end_of_stream: bool) -> FilterAction {
    if !end_of_stream {
        return FilterAction::Continue;
    }

    let status = ctx
        .get_metadata(RESPONSE_STATUS_KEY)
        .and_then(|s| s.parse::<u16>().ok())
        .and_then(|c| http::StatusCode::from_u16(c).ok())
        .unwrap_or(http::StatusCode::INTERNAL_SERVER_ERROR);

    let bytes = body.as_deref().unwrap_or(&[]);
    let normalized = normalize_error_response(bytes, status);
    *body = Some(Bytes::from(normalized));
    FilterAction::Continue
}

/// Buffer-and-replace the body with an OpenAI Chat Completions response.
fn translate_success_body(
    ctx: &mut HttpFilterContext<'_>,
    body: &mut Option<Bytes>,
    end_of_stream: bool,
) -> FilterAction {
    if !end_of_stream {
        return FilterAction::Continue;
    }

    let model = ctx.get_metadata(MODEL_KEY).unwrap_or("unknown").to_owned();

    let bytes = body.as_deref().unwrap_or(&[]);
    match transform_response(bytes, &model) {
        Ok(translated) => {
            *body = Some(Bytes::from(translated));
        },
        Err(e) => {
            warn!(error = %e, "Bedrock Converse: non-streaming response translation failed");
            set_response_translation_error(ctx, body, &e);
        },
    }
    FilterAction::Continue
}

/// Replace a malformed upstream success response with an OpenAI-shaped 502.
fn set_response_translation_error(ctx: &mut HttpFilterContext<'_>, body: &mut Option<Bytes>, error: &str) {
    if let Some(response) = ctx.response_header.as_mut() {
        response.status = http::StatusCode::BAD_GATEWAY;
        response.headers.insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("application/json"),
        );
    }
    ctx.response_headers_modified = true;
    *body = Some(Bytes::from(build_openai_error_body(
        &format!("response translation failed: {error}"),
        "server_error",
        "response_translation_error",
    )));
}

/// Feed a binary `EventStream` chunk through the decoder and emit OpenAI SSE
/// frames for each fully-decoded Bedrock event.
#[expect(
    clippy::too_many_lines,
    reason = "handles full streaming lifecycle: decode, translate, SSE-frame, state management"
)]
fn translate_stream_chunk(
    ctx: &mut HttpFilterContext<'_>,
    body: &mut Option<Bytes>,
    end_of_stream: bool,
    max_frame_bytes: usize,
) -> FilterAction {
    let model = ctx.get_metadata(MODEL_KEY).unwrap_or("unknown").to_owned();
    let response_id = ctx
        .get_metadata(RESPONSE_ID_KEY)
        .unwrap_or(response::FALLBACK_ID)
        .to_owned();

    let chunk = body.take().unwrap_or_default();

    // Lazy-init the streaming state on the first body chunk.
    if ctx.get_filter_state::<BedrockStreamingState>().is_none() {
        ctx.insert_filter_state(BedrockStreamingState {
            decoder: EventStreamDecoder::with_max_frame_len(max_frame_bytes),
            stream_state: StreamState::default(),
            failed: false,
        });
    }

    let mut sse_output: Vec<u8> = Vec::new();

    if let Some(state) = ctx.get_filter_state_mut::<BedrockStreamingState>() {
        state.decoder.push(&chunk);

        // `decode_all` drains every complete frame from the buffer,
        // leaving any partial final frame in the decoder for the next
        // `on_response_body` call.
        match state.decoder.decode_all() {
            Ok(messages) => {
                for msg in &messages {
                    if msg.is_exception() {
                        state.failed = true;
                    }
                    if let Some(json_bytes) = transform_stream_event(msg, &model, &response_id, &mut state.stream_state)
                    {
                        sse_output.extend_from_slice(b"data: ");
                        sse_output.extend_from_slice(&json_bytes);
                        sse_output.extend_from_slice(b"\n\n");
                    }
                }
            },
            Err(e) => {
                warn!(error = %e, "Bedrock EventStream decode error");
                append_sse_error(
                    &mut sse_output,
                    &format!("invalid Bedrock event stream: {e}"),
                    "stream_decode_error",
                );
                state.failed = true;
            },
        }

        if end_of_stream && !state.failed && !state.decoder.is_empty() {
            append_sse_error(
                &mut sse_output,
                "Bedrock event stream ended with an incomplete frame",
                "incomplete_stream",
            );
            state.failed = true;
        }

        if end_of_stream && !state.failed && state.stream_state.finish_reason.is_none() {
            append_sse_error(
                &mut sse_output,
                "Bedrock event stream ended before a messageStop event",
                "incomplete_stream",
            );
            state.failed = true;
        }

        if end_of_stream && !state.failed {
            sse_output.extend_from_slice(SSE_DONE);
        }
    }
    if end_of_stream {
        // Release streaming state; the response is complete.
        ctx.remove_filter_state::<BedrockStreamingState>();
    }

    // Always set the body — even to an empty Bytes — so the framework
    // knows we consumed the upstream chunk.
    *body = if sse_output.is_empty() {
        Some(Bytes::new())
    } else {
        Some(Bytes::from(sse_output))
    };

    FilterAction::Continue
}

/// Append an OpenAI-shaped streaming error event.
fn append_sse_error(output: &mut Vec<u8>, message: &str, code: &str) {
    output.extend_from_slice(b"data: ");
    output.extend_from_slice(&build_openai_error_body(message, "server_error", code));
    output.extend_from_slice(b"\n\n");
}

/// Return an OpenAI-shaped 400 response after the complete request body has
/// been consumed, so malformed input never becomes an internal 500.
fn invalid_request_action(message: &str) -> FilterAction {
    FilterAction::Reject(
        Rejection::status(400)
            .with_header("content-type", "application/json")
            .with_body(Bytes::from(build_openai_error_body(
                message,
                "invalid_request_error",
                "invalid_request",
            )))
            .preserving_keepalive(),
    )
}

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------

/// Nanosecond-precision seed for a synthetic response ID.
///
/// Uses `SystemTime` as a cheap source of entropy; not cryptographically
/// strong, but sufficient for stable SSE chunk correlation within one
/// response.
fn response_id_seed() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos())
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    unused_must_use,
    reason = "tests"
)]
mod tests {
    use http::Method;
    use praxis_filter::parse_filter_config;
    use serde_json::Value;

    use super::*;
    use crate::{
        bedrock::eventstream::build_frame,
        test_utils::{make_filter_context, make_request, make_response},
    };

    // ── Helpers ───────────────────────────────────────────────────────────

    fn make_filter(yaml_str: &str) -> OpenaiChatCompletionsToBedrockConverseFilter {
        let yaml: serde_yaml::Value = serde_yaml::from_str(yaml_str).unwrap();
        let cfg: BedrockConverseConfig = parse_filter_config(FILTER_NAME, &yaml).unwrap();
        let validated = build_config(cfg).unwrap();
        OpenaiChatCompletionsToBedrockConverseFilter { config: validated }
    }

    fn chat_body(model: &str, stream: bool) -> Bytes {
        Bytes::from(
            serde_json::json!({
                "model": model,
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": stream
            })
            .to_string(),
        )
    }

    fn event_stream_frame(event_type: &str, payload: &[u8]) -> Vec<u8> {
        build_frame(&[(":message-type", "event"), (":event-type", event_type)], payload)
    }

    // ── Lifecycle smoke test ──────────────────────────────────────────────

    #[test]
    fn default_config_parses() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("{}").unwrap();
        let filter = OpenaiChatCompletionsToBedrockConverseFilter::from_config(&yaml).unwrap();
        assert_eq!(filter.name(), FILTER_NAME);
    }

    #[test]
    fn name_matches_filter_name_constant() {
        let filter = make_filter("{}");
        assert_eq!(filter.name(), FILTER_NAME);
    }

    // ── on_request ────────────────────────────────────────────────────────

    #[tokio::test]
    async fn on_request_removes_accept_encoding_and_authorization() {
        let filter = make_filter("{}");
        let req = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);

        filter.on_request(&mut ctx).await.unwrap();

        assert!(
            ctx.request_headers_to_remove.contains(&http::header::ACCEPT_ENCODING),
            "Accept-Encoding must be stripped"
        );
        assert!(
            ctx.request_headers_to_remove.contains(&http::header::AUTHORIZATION),
            "Authorization must be stripped so aws_sigv4_sign injects the correct credential"
        );
    }

    // ── on_request_body ───────────────────────────────────────────────────

    #[tokio::test]
    async fn on_request_body_translates_and_sets_path_non_streaming() {
        let filter = make_filter("{}");
        let req = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        ctx.current_filter_id = Some(0);

        let mut body = Some(chat_body("amazon.nova-lite-v1:0", false));
        filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

        assert_eq!(
            ctx.rewritten_path.as_deref(),
            Some("/model/amazon.nova-lite-v1:0/converse"),
            "non-streaming path must end in /converse"
        );
        assert_eq!(
            ctx.get_metadata(MODEL_KEY),
            Some("amazon.nova-lite-v1:0"),
            "model must be stored for the response phase"
        );

        // Body must be valid Bedrock Converse JSON.
        let translated: Value = serde_json::from_slice(&body.unwrap()).unwrap();
        assert!(
            translated.get("messages").is_some(),
            "translated body must contain messages"
        );
    }

    #[tokio::test]
    async fn on_request_body_sets_converse_stream_path_for_streaming() {
        let filter = make_filter("{}");
        let req = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        ctx.current_filter_id = Some(0);

        let mut body = Some(chat_body("anthropic.claude-3-sonnet-20240229-v1:0", true));
        filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

        assert_eq!(
            ctx.rewritten_path.as_deref(),
            Some("/model/anthropic.claude-3-sonnet-20240229-v1:0/converse-stream"),
            "streaming path must end in /converse-stream"
        );
    }

    #[tokio::test]
    async fn on_request_body_skips_non_final_chunk() {
        let filter = make_filter("{}");
        let req = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);

        let mut body: Option<Bytes> = None;
        // Should not error and should not rewrite the path on an interim chunk.
        filter.on_request_body(&mut ctx, &mut body, false).await.unwrap();

        assert!(ctx.rewritten_path.is_none(), "path must not be set on interim chunk");
    }

    #[tokio::test]
    async fn malformed_request_returns_openai_400() {
        let filter = make_filter("{}");
        let req = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        let mut body = Some(Bytes::from_static(b"not-json"));

        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
        let FilterAction::Reject(rejection) = action else {
            panic!("malformed input must be rejected");
        };
        assert_eq!(rejection.status, 400);
        assert!(rejection.preserve_keepalive);
        let error: Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();
        assert_eq!(error["error"]["type"], "invalid_request_error");
    }

    #[tokio::test]
    async fn unsafe_model_does_not_rewrite_upstream_path() {
        let filter = make_filter("{}");
        let req = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        let mut body = Some(chat_body("../../../other-endpoint", false));

        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

        assert!(matches!(action, FilterAction::Reject(r) if r.status == 400));
        assert!(
            ctx.rewritten_path.is_none(),
            "invalid model must not set an upstream path"
        );
        assert!(
            ctx.get_metadata(MODEL_KEY).is_none(),
            "invalid model must not be persisted"
        );
    }

    #[tokio::test]
    async fn empty_request_body_returns_openai_400() {
        let filter = make_filter("{}");
        let req = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        let mut body = None;

        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
        assert!(matches!(action, FilterAction::Reject(r) if r.status == 400));
    }

    // ── on_response — error ───────────────────────────────────────────────

    #[tokio::test]
    async fn on_response_error_sets_error_path_and_buffers_body() {
        let filter = make_filter("{}");
        let req = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);

        let mut upstream = make_response();
        upstream.status = http::StatusCode::TOO_MANY_REQUESTS;
        ctx.response_header = Some(&mut upstream);
        ctx.set_metadata(MODEL_KEY, "some-model");

        filter.on_response(&mut ctx).await.unwrap();

        assert_eq!(ctx.get_metadata(RESPONSE_PATH_KEY), Some(RESPONSE_PATH_ERROR));
        assert_eq!(ctx.get_metadata(RESPONSE_STATUS_KEY), Some("429"));
    }

    // ── on_response_body — error ──────────────────────────────────────────

    #[test]
    fn on_response_body_error_normalizes_bedrock_error() {
        let filter = make_filter("{}");
        let req = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        ctx.set_metadata(RESPONSE_PATH_KEY, RESPONSE_PATH_ERROR);
        ctx.set_metadata(RESPONSE_STATUS_KEY, "400");

        let bedrock_err = br#"{"message":"Invalid model identifier"}"#;
        let mut body: Option<Bytes> = Some(Bytes::copy_from_slice(bedrock_err));
        filter.on_response_body(&mut ctx, &mut body, true).unwrap();

        let result: Value = serde_json::from_slice(&body.unwrap()).unwrap();
        assert_eq!(result["error"]["type"], "invalid_request_error");
        assert!(
            result["error"]["message"]
                .as_str()
                .unwrap()
                .contains("Invalid model identifier")
        );
    }

    #[test]
    fn on_response_body_error_waits_for_end_of_stream() {
        let filter = make_filter("{}");
        let req = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        ctx.set_metadata(RESPONSE_PATH_KEY, RESPONSE_PATH_ERROR);
        ctx.set_metadata(RESPONSE_STATUS_KEY, "500");

        let mut body: Option<Bytes> = Some(Bytes::from_static(b"partial"));
        filter.on_response_body(&mut ctx, &mut body, false).unwrap();

        // Body must be untouched while waiting for the full buffer.
        assert_eq!(body.as_deref(), Some(b"partial".as_ref()));
    }

    // ── on_response_body — non-streaming success ──────────────────────────

    #[test]
    fn on_response_body_success_translates_full_response() {
        let filter = make_filter("{}");
        let req = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        ctx.set_metadata(RESPONSE_PATH_KEY, RESPONSE_PATH_SUCCESS);
        ctx.set_metadata(MODEL_KEY, "amazon.nova-lite-v1:0");

        let bedrock_resp = serde_json::json!({
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "Hello!"}]
                }
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 5, "outputTokens": 2, "totalTokens": 7}
        });
        let mut body = Some(Bytes::from(bedrock_resp.to_string()));
        filter.on_response_body(&mut ctx, &mut body, true).unwrap();

        let result: Value = serde_json::from_slice(&body.unwrap()).unwrap();
        assert_eq!(result["object"], "chat.completion");
        assert_eq!(result["choices"][0]["message"]["content"], "Hello!");
        assert_eq!(result["choices"][0]["finish_reason"], "stop");
    }

    #[test]
    fn malformed_success_response_becomes_openai_502() {
        let filter = make_filter("{}");
        let req = make_request(Method::POST, "/v1/chat/completions");
        let mut upstream = make_response();
        let mut ctx = make_filter_context(&req);
        ctx.response_header = Some(&mut upstream);
        ctx.set_metadata(RESPONSE_PATH_KEY, RESPONSE_PATH_SUCCESS);
        ctx.set_metadata(MODEL_KEY, "amazon.nova-lite-v1:0");

        let mut body = Some(Bytes::from_static(b"{}"));
        filter.on_response_body(&mut ctx, &mut body, true).unwrap();

        assert_eq!(
            ctx.response_header.as_ref().unwrap().status,
            http::StatusCode::BAD_GATEWAY
        );
        let result: Value = serde_json::from_slice(body.as_deref().unwrap()).unwrap();
        assert_eq!(result["error"]["type"], "server_error");
        assert_eq!(result["error"]["code"], "response_translation_error");
    }

    // ── on_response_body — streaming ──────────────────────────────────────

    #[test]
    fn on_response_body_stream_emits_sse_events_and_done_sentinel() {
        let filter = make_filter("{}");
        let req = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        ctx.current_filter_id = Some(0);
        ctx.set_metadata(RESPONSE_PATH_KEY, RESPONSE_PATH_STREAM);
        ctx.set_metadata(MODEL_KEY, "amazon.nova-lite-v1:0");
        ctx.set_metadata(RESPONSE_ID_KEY, "chatcmpl-test");

        // Frame 1: messageStart
        let start_frame = event_stream_frame("messageStart", br#"{"role":"assistant"}"#);
        // Frame 2: contentBlockDelta
        let delta_frame = event_stream_frame(
            "contentBlockDelta",
            br#"{"contentBlockIndex":0,"delta":{"text":"Hi!"}}"#,
        );
        // Frame 3: messageStop
        let stop_frame = event_stream_frame("messageStop", br#"{"stopReason":"end_turn"}"#);

        // Deliver all frames in a single chunk (common case).
        let mut combined = start_frame;
        combined.extend_from_slice(&delta_frame);
        combined.extend_from_slice(&stop_frame);

        let mut body = Some(Bytes::from(combined));
        filter.on_response_body(&mut ctx, &mut body, true).unwrap();

        let output = body.unwrap();
        let text = std::str::from_utf8(&output).unwrap();

        // Must contain at least one data frame.
        assert!(text.contains("data: "), "output must contain SSE data frames");
        // Must end with [DONE].
        assert!(text.ends_with("data: [DONE]\n\n"), "must end with [DONE] sentinel");
        // Text delta must be present.
        assert!(text.contains("Hi!"), "text delta must appear in output");
    }

    #[test]
    fn on_response_body_stream_partial_frame_emits_nothing_until_complete() {
        let filter = make_filter("{}");
        let req = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        ctx.current_filter_id = Some(0);
        ctx.set_metadata(RESPONSE_PATH_KEY, RESPONSE_PATH_STREAM);
        ctx.set_metadata(MODEL_KEY, "amazon.nova-lite-v1:0");
        ctx.set_metadata(RESPONSE_ID_KEY, "chatcmpl-test");

        let full_frame = event_stream_frame(
            "contentBlockDelta",
            br#"{"contentBlockIndex":0,"delta":{"text":"Hello"}}"#,
        );
        // Split the frame halfway.
        let split_at = full_frame.len() / 2;
        let (first_half, second_half) = full_frame.split_at(split_at);

        // First chunk: partial frame — no SSE output expected.
        let mut body1 = Some(Bytes::copy_from_slice(first_half));
        filter.on_response_body(&mut ctx, &mut body1, false).unwrap();
        let out1 = body1.unwrap();
        assert!(out1.is_empty(), "partial frame must produce no SSE output");

        // Second chunk: remainder — full frame now decoded.
        let mut final_chunk = second_half.to_vec();
        final_chunk.extend_from_slice(&event_stream_frame("messageStop", br#"{"stopReason":"end_turn"}"#));
        let mut body2 = Some(Bytes::from(final_chunk));
        filter.on_response_body(&mut ctx, &mut body2, true).unwrap();
        let out2 = body2.unwrap();
        let text2 = std::str::from_utf8(&out2).unwrap();
        assert!(text2.contains("Hello"), "completed frame must appear in second chunk");
        assert!(text2.ends_with("data: [DONE]\n\n"));
    }

    #[test]
    fn truncated_stream_emits_error_without_done() {
        let filter = make_filter("{}");
        let req = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        ctx.current_filter_id = Some(0);
        ctx.set_metadata(RESPONSE_PATH_KEY, RESPONSE_PATH_STREAM);

        let frame = event_stream_frame("messageStart", br#"{"role":"assistant"}"#);
        let mut body = Some(Bytes::copy_from_slice(&frame[..frame.len() / 2]));
        filter.on_response_body(&mut ctx, &mut body, true).unwrap();

        let text = std::str::from_utf8(body.as_deref().unwrap()).unwrap();
        assert!(text.contains("incomplete_stream"));
        assert!(!text.contains("[DONE]"));
    }

    #[test]
    fn corrupt_stream_emits_error_without_done() {
        let filter = make_filter("{}");
        let req = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        ctx.current_filter_id = Some(0);
        ctx.set_metadata(RESPONSE_PATH_KEY, RESPONSE_PATH_STREAM);

        let mut frame = event_stream_frame("messageStart", br#"{"role":"assistant"}"#);
        let last = frame.len() - 1;
        frame[last] ^= 0xFF;
        let mut body = Some(Bytes::from(frame));
        filter.on_response_body(&mut ctx, &mut body, true).unwrap();

        let text = std::str::from_utf8(body.as_deref().unwrap()).unwrap();
        assert!(text.contains("stream_decode_error"));
        assert!(!text.contains("[DONE]"));
    }
}
