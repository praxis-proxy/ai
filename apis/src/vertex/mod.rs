// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Anthropic Messages ⇄ Vertex AI `rawPredict` dialect filter.
//!
//! Vertex serves Claude through the Anthropic Messages wire format but
//! with a dialect seam on each side of the request. This filter closes
//! both so a single client-facing model id can route to either backend:
//! requests whose model does not start with the configured `model_prefix`
//! pass through unchanged, so this filter can share an Anthropic chain with
//! other suppliers. Matching requests receive the internal
//! `x-praxis-ai-vertex-route: vertex` marker for the router; the marker must
//! be removed before forwarding upstream.
//!
//! **Request** — the body's `model` moves into the URL
//! (`…/publishers/anthropic/models/{model}:rawPredict`,
//! `:streamRawPredict` when `stream` is `true`, `stream` read before the
//! body is rewritten), `anthropic_version` is injected, and the `model`
//! field is removed — Vertex rejects it with
//! `model: Extra inputs are not permitted`. `count_tokens` keeps its
//! model in the body at a distinct URL.
//!
//! **Response** — the snapshot `model` id is restored to the
//! user-facing id (top-level JSON, or `message.model` inside the
//! `message_start` SSE event, patched frame-scoped without buffering
//! the stream), and Google error envelopes are translated to Anthropic
//! error types so client retry logic keeps working. Translation is
//! pre-stream only: a mid-stream kill has no body to rewrite.
//!
//! Pair with the [`gcp_adc` filter](https://github.com/praxis-proxy/ai)
//! (`source: key_file`) and a `vertex` cluster pointing at
//! `aiplatform.googleapis.com:443` (`tls.sni` likewise); this filter
//! rewrites path and bodies, not credentials or Host.
//!
//! [`VertexFilter`]

mod config;
mod request;
mod response;

use async_trait::async_trait;
use bytes::Bytes;
use http::{HeaderName, HeaderValue};
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, parse_filter_config,
};
use tracing::debug;

use self::config::{VertexConfig, build_config};
use crate::{
    anthropic::invalid_request_rejection,
    openai::sse::SseFrameParser,
    vertex::request::{classify, transform_request},
};

/// Metadata key carrying the classified operation (`messages`,
/// `count_tokens`). Absent means "not a Vertex-handled request": no
/// request, response, or SSE transform may run.
const OPERATION_KEY: &str = "vertex.operation";
/// Metadata key carrying the user-facing model id to restore in
/// responses.
const MODEL_KEY: &str = "vertex.model";
/// Metadata key selecting the response transformation mode.
const TRANSFORM_KEY: &str = "vertex.response_transform";
/// Response transform marker for a successful JSON response.
const TRANSFORM_SUCCESS: &str = "success";
/// Response transform marker for an upstream error.
const TRANSFORM_ERROR: &str = "error";
/// Response transform marker for an SSE stream.
const TRANSFORM_SSE: &str = "sse";
/// Metadata key preserving the upstream status for the body phase.
const STATUS_KEY: &str = "vertex.response_status";

/// `anthropic-beta` header name, for the beta-flag allowlist.
const ANTHROPIC_BETA: HeaderName = HeaderName::from_static("anthropic-beta");
/// Internal route marker read by the unified router. Client-supplied
/// `x-praxis-*` headers are rejected at the protocol boundary.
pub const ROUTE_HEADER: HeaderName = HeaderName::from_static("x-praxis-ai-vertex-route");
/// Route marker value emitted for requests handled by the Vertex supplier.
const ROUTE_VALUE: HeaderValue = HeaderValue::from_static("vertex");

/// Translates Anthropic Messages requests to Vertex AI `rawPredict` and
/// Vertex responses back to the Anthropic dialect.
///
/// # YAML
///
/// ```yaml
/// filter: vertex
/// project: my-gcp-project
/// ```
///
/// # Full YAML
///
/// ```yaml
/// filter: vertex
/// project: my-gcp-project
/// location: global
/// model_prefix: "vertex/"
/// model_pin: "@20250929"
/// beta_allowlist: [context-1m-2025-08-07, interleaved-thinking-2025-05-14]
/// max_body_bytes: 33554432
/// ```
pub struct VertexFilter {
    /// Parsed and validated configuration.
    config: VertexConfig,
}

impl VertexFilter {
    /// Create a filter from parsed YAML config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the YAML config is invalid or a URL
    /// component carries unsafe characters.
    ///
    /// ```
    /// use praxis_ai_apis::vertex::VertexFilter;
    /// let filter =
    ///     VertexFilter::from_config(&serde_yaml::from_str("project: my-gcp-project").unwrap())
    ///         .unwrap();
    /// assert_eq!(filter.name(), "vertex");
    /// ```
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: VertexConfig = parse_filter_config("vertex", config)?;
        Ok(Box::new(Self {
            config: build_config(cfg)?,
        }))
    }

    /// Apply the `anthropic-beta` allowlist. An empty allowlist (the
    /// default) forwards the header untouched; otherwise unknown flags
    /// are stripped and the header removed entirely when nothing
    /// remains — Vertex rejects beta flags it does not support, and
    /// clients like Claude Code send several on every request.
    fn filter_beta_flags(&self, ctx: &mut HttpFilterContext<'_>) {
        if self.config.beta_allowlist.is_empty() {
            return;
        }
        let Some(value) = ctx.request.headers.get(&ANTHROPIC_BETA).and_then(|v| v.to_str().ok()) else {
            return;
        };

        let kept: Vec<&str> = value
            .split(',')
            .map(str::trim)
            .filter(|flag| !flag.is_empty() && self.config.beta_allowlist.iter().any(|allowed| allowed == flag))
            .collect();

        if kept.is_empty() {
            ctx.request_headers_to_remove.push(ANTHROPIC_BETA.clone());
        } else if let Ok(joined) = kept.join(", ").parse() {
            ctx.request_headers_to_set.push((ANTHROPIC_BETA.clone(), joined));
        }
    }
}

#[async_trait]
impl HttpFilter for VertexFilter {
    fn name(&self) -> &'static str {
        "vertex"
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

    fn needs_request_context(&self) -> bool {
        // The body phase classifies from the request path: with a
        // StreamBuffer pre-read the body hook can run ahead of
        // on_request, so metadata hand-off would be unreliable.
        true
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

        let Some(operation) = classify(current_path(ctx)) else {
            return Ok(FilterAction::Continue);
        };
        let Some(bytes) = body.as_ref().filter(|b| !b.is_empty()) else {
            return Ok(FilterAction::Continue);
        };
        match transform_request(bytes, operation, &self.config) {
            Ok(Some(transformed)) => {
                self.filter_beta_flags(ctx);
                debug!(
                    model = %transformed.user_model,
                    path = %transformed.path,
                    "translated Anthropic request to Vertex rawPredict"
                );
                ctx.rewritten_path = Some(transformed.path);
                ctx.request_headers_to_set
                    .push((ROUTE_HEADER.clone(), ROUTE_VALUE.clone()));
                // Marks "this request was transformed"; response-side
                // transforms only run for marked requests. Set here (not
                // in on_request) because the pre-read body phase may run
                // before the request phase.
                ctx.set_metadata(OPERATION_KEY, "handled");
                ctx.set_metadata(MODEL_KEY, transformed.user_model);
                *body = Some(Bytes::from(transformed.body));
            },
            Ok(None) => return Ok(FilterAction::Continue),
            Err(error) => return Ok(FilterAction::Reject(invalid_request_rejection(&error.to_string()))),
        }
        Ok(FilterAction::Continue)
    }

    async fn on_response(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        // A response to a request this filter did not transform (wrong
        // path, or an upstream hit via another route) must not be
        // transformed either.
        if ctx.get_metadata(OPERATION_KEY).is_none() {
            return Ok(FilterAction::Continue);
        }

        let transform = response_transform(ctx);
        ctx.set_metadata(TRANSFORM_KEY, transform);
        if transform == TRANSFORM_ERROR {
            let status = ctx
                .response_header
                .as_ref()
                .map_or(500, |response| response.status.as_u16());
            ctx.set_metadata(STATUS_KEY, status.to_string());
        }

        if transform == TRANSFORM_SSE {
            // Streaming stays Stream mode; only the (small) in-flight
            // frame is ever buffered, inside the per-request parser.
            ctx.insert_filter_state(SseFrameParser::new(self.config.max_body_bytes));
        } else {
            ctx.set_response_body_mode(BodyMode::StreamBuffer {
                max_bytes: Some(self.config.max_body_bytes),
            });
        }
        if let Some(resp) = &mut ctx.response_header {
            resp.headers.remove(http::header::CONTENT_LENGTH);
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
        match ctx.get_metadata(TRANSFORM_KEY) {
            Some(TRANSFORM_SSE) => patch_sse_chunk(ctx, body, end_of_stream),
            Some(TRANSFORM_ERROR) if end_of_stream => translate_error_body(ctx, body),
            Some(TRANSFORM_SUCCESS) if end_of_stream => restore_response_model(ctx, body),
            _ => {},
        }
        Ok(FilterAction::Continue)
    }
}

// -----------------------------------------------------------------------------
// Response Helpers
// -----------------------------------------------------------------------------

/// The path this filter should classify against: an earlier filter's
/// rewrite wins over the original URI, matching how the router and the
/// protocol layer resolve the outbound path.
fn current_path<'a>(ctx: &'a HttpFilterContext<'_>) -> &'a str {
    ctx.rewritten_path.as_deref().unwrap_or_else(|| {
        ctx.request
            .uri
            .path_and_query()
            .map_or_else(|| ctx.request.uri.path(), |pq| pq.as_str())
    })
}

/// Select the response transformation mode while headers are available.
fn response_transform(ctx: &HttpFilterContext<'_>) -> &'static str {
    let is_error = ctx
        .response_header
        .as_ref()
        .is_some_and(|r| r.status.is_client_error() || r.status.is_server_error());
    let is_sse = ctx
        .response_header
        .as_ref()
        .and_then(|r| r.headers.get(http::header::CONTENT_TYPE))
        .and_then(|v| v.to_str().ok())
        .is_some_and(crate::is_event_stream_content_type);

    if is_sse {
        TRANSFORM_SSE
    } else if is_error {
        TRANSFORM_ERROR
    } else {
        TRANSFORM_SUCCESS
    }
}

/// Patch an SSE chunk: parse complete frames, restore the model in
/// `message_start` only, and re-emit. Partial frames stay in the
/// per-request parser across chunk boundaries.
fn patch_sse_chunk(ctx: &mut HttpFilterContext<'_>, body: &mut Option<Bytes>, end_of_stream: bool) {
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
        Err(error) => {
            // A malformed or oversized frame must not kill the stream or
            // emit split duplicates; drop the chunk and let the stream
            // finish (matches the azure translation filter's behavior).
            debug!(%error, "vertex: SSE frame parse failed; dropping chunk to keep the stream alive");
            ctx.insert_filter_state(parser);
            *body = Some(Bytes::new());
            return;
        },
    };

    if !end_of_stream {
        ctx.insert_filter_state(parser);
    }

    let user_model = ctx.get_metadata(MODEL_KEY).map(str::to_owned);
    *body = Some(Bytes::from(response::rebuild_sse_frames(
        &frames,
        user_model.as_deref(),
    )));
}

/// Translate a buffered Google error envelope to the Anthropic shape,
/// preserving the HTTP status.
fn translate_error_body(ctx: &HttpFilterContext<'_>, body: &mut Option<Bytes>) {
    let Some(bytes) = body.as_ref().filter(|b| !b.is_empty()) else {
        return;
    };
    let status = ctx
        .get_metadata(STATUS_KEY)
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(500);

    if let Some(translated) = response::translate_google_error(bytes, status) {
        debug!(
            original_len = bytes.len(),
            translated_len = translated.len(),
            "vertex: translated Google error envelope to Anthropic shape"
        );
        *body = Some(Bytes::from(translated));
    }
}

/// Restore the user-facing model id in a buffered JSON response.
fn restore_response_model(ctx: &HttpFilterContext<'_>, body: &mut Option<Bytes>) {
    let (Some(bytes), Some(user_model)) = (body.as_ref().filter(|b| !b.is_empty()), ctx.get_metadata(MODEL_KEY)) else {
        return;
    };

    if let Some(restored) = response::restore_model(bytes, user_model) {
        debug!("vertex: restored user-facing model id in response");
        *body = Some(Bytes::from(restored));
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::too_many_lines,
    unused_must_use,
    reason = "tests"
)]
mod tests {
    use http::{HeaderValue, Method, StatusCode, header};

    use super::*;
    use crate::test_utils::{make_filter_context, make_request, make_response};

    fn filter(yaml: &str) -> Box<dyn HttpFilter> {
        let config: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
        VertexFilter::from_config(&config).unwrap()
    }

    fn messages_body(extra: &str) -> Bytes {
        Bytes::from(format!(
            r#"{{"model":"vertex/claude-sonnet-4-5",{extra},"messages":[{{"role":"user","content":"hi"}}]}}"#
        ))
    }

    async fn run_request_body(
        filter: &dyn HttpFilter,
        ctx: &mut HttpFilterContext<'_>,
        body: Bytes,
    ) -> Result<FilterAction, FilterError> {
        let mut body = Some(body);
        let action = filter.on_request_body(ctx, &mut body, true).await?;
        ctx.buffered_request_body = body;
        debug_assert!(matches!(filter.on_request(ctx).await?, FilterAction::Continue));
        Ok(action)
    }

    #[tokio::test]
    async fn messages_request_becomes_rawpredict_url() {
        let filter = filter("project: demo");
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);

        let action = run_request_body(filter.as_ref(), &mut ctx, messages_body(r#""max_tokens":8"#))
            .await
            .unwrap();
        assert!(matches!(action, FilterAction::Continue));
        assert_eq!(
            ctx.rewritten_path.as_deref(),
            Some("/v1/projects/demo/locations/global/publishers/anthropic/models/claude-sonnet-4-5:rawPredict")
        );
        let body: serde_json::Value =
            serde_json::from_slice(ctx.buffered_request_body.as_ref().unwrap().as_ref()).unwrap();
        assert!(body.get("model").is_none());
        assert_eq!(body["anthropic_version"], "vertex-2023-10-16");
        assert_eq!(ctx.get_metadata(MODEL_KEY), Some("vertex/claude-sonnet-4-5"));
        assert!(
            ctx.request_headers_to_set
                .iter()
                .any(|(name, value)| name == ROUTE_HEADER && value == ROUTE_VALUE)
        );
    }

    #[tokio::test]
    async fn non_vertex_model_passes_through_without_route_marker() {
        let filter = filter("project: demo");
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);
        let original = Bytes::from_static(
            br#"{"model":"claude-sonnet-5","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}"#,
        );

        let action = run_request_body(filter.as_ref(), &mut ctx, original.clone())
            .await
            .unwrap();
        assert!(matches!(action, FilterAction::Continue));
        assert_eq!(ctx.buffered_request_body.as_ref(), Some(&original));
        assert!(ctx.rewritten_path.is_none());
        assert!(ctx.get_metadata(OPERATION_KEY).is_none());
        assert!(!ctx.request_headers_to_set.iter().any(|(name, _)| name == ROUTE_HEADER));
    }

    #[tokio::test]
    async fn stream_true_selects_stream_verb() {
        let filter = filter("project: demo");
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);

        run_request_body(
            filter.as_ref(),
            &mut ctx,
            messages_body(r#""stream":true,"max_tokens":8"#),
        )
        .await
        .unwrap();
        assert!(
            ctx.rewritten_path
                .as_deref()
                .is_some_and(|p| p.ends_with(":streamRawPredict")),
            "stream flag must select the stream verb, got {:?}",
            ctx.rewritten_path
        );
    }

    #[tokio::test]
    async fn count_tokens_keeps_model_in_body() {
        let filter = filter("project: demo");
        let request = make_request(Method::POST, "/v1/messages/count_tokens");
        let mut ctx = make_filter_context(&request);

        run_request_body(filter.as_ref(), &mut ctx, messages_body(r#""max_tokens":8"#))
            .await
            .unwrap();
        assert!(
            ctx.rewritten_path
                .as_deref()
                .is_some_and(|p| p.ends_with("models/count-tokens:rawPredict"))
        );
        let body: serde_json::Value =
            serde_json::from_slice(ctx.buffered_request_body.as_ref().unwrap().as_ref()).unwrap();
        assert_eq!(
            body["model"], "claude-sonnet-4-5",
            "count_tokens needs the publisher model in the body"
        );
    }

    #[tokio::test]
    async fn unrelated_paths_pass_through() {
        let filter = filter("project: demo");
        for path in ["/v1/models", "/v1/messages/batches", "/healthz"] {
            let request = make_request(Method::POST, path);
            let mut ctx = make_filter_context(&request);
            let original = messages_body(r#""max_tokens":8"#);

            let action = run_request_body(filter.as_ref(), &mut ctx, original.clone())
                .await
                .unwrap();
            assert!(matches!(action, FilterAction::Continue));
            assert!(ctx.rewritten_path.is_none(), "{path} must not be rewritten");
            assert_eq!(
                ctx.buffered_request_body.as_ref().unwrap(),
                &original,
                "{path} body must be untouched"
            );
        }
    }

    #[tokio::test]
    async fn unsafe_model_rejected_with_anthropic_error_shape() {
        let filter = filter("project: demo");
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);
        let evil = Bytes::from_static(br#"{"model":"vertex/../../secrets","messages":[]}"#);
        let action = run_request_body(filter.as_ref(), &mut ctx, evil).await.unwrap();
        let FilterAction::Reject(rejection) = action else {
            panic!("path-injecting model must be rejected, got {action:?}");
        };
        assert_eq!(rejection.status, 400);
        let body: serde_json::Value = serde_json::from_slice(rejection.body.as_ref().unwrap().as_ref()).unwrap();
        assert_eq!(body["type"], "error");
        assert_eq!(body["error"]["type"], "invalid_request_error");
    }

    #[tokio::test]
    async fn beta_allowlist_keeps_only_allowed_flags() {
        let filter = filter("project: demo\nbeta_allowlist: [context-1m-2025-08-07]");
        let mut request = make_request(Method::POST, "/v1/messages");
        request.headers.insert(
            HeaderName::from_static("anthropic-beta"),
            "context-1m-2025-08-07, interleaved-thinking-2025-05-14"
                .parse()
                .unwrap(),
        );
        let mut ctx = make_filter_context(&request);

        run_request_body(filter.as_ref(), &mut ctx, messages_body(r#""max_tokens":8"#))
            .await
            .unwrap();
        assert!(ctx.request_headers_to_remove.is_empty());
        let beta_header = ctx
            .request_headers_to_set
            .iter()
            .find(|(name, _)| name == ANTHROPIC_BETA)
            .unwrap();
        assert_eq!(beta_header.1.to_str().unwrap(), "context-1m-2025-08-07");
    }

    #[tokio::test]
    async fn beta_allowlist_drops_header_when_nothing_remains() {
        let filter = filter("project: demo\nbeta_allowlist: [context-1m-2025-08-07]");
        let mut request = make_request(Method::POST, "/v1/messages");
        request.headers.insert(
            HeaderName::from_static("anthropic-beta"),
            "tool-search-2025-04-14".parse().unwrap(),
        );
        let mut ctx = make_filter_context(&request);

        run_request_body(filter.as_ref(), &mut ctx, messages_body(r#""max_tokens":8"#))
            .await
            .unwrap();
        assert_eq!(ctx.request_headers_to_remove.len(), 1);
        assert!(
            ctx.request_headers_to_set
                .iter()
                .all(|(name, _)| name != ANTHROPIC_BETA)
        );
    }

    #[tokio::test]
    async fn beta_header_untouched_when_allowlist_empty() {
        let filter = filter("project: demo");
        let mut request = make_request(Method::POST, "/v1/messages");
        request.headers.insert(
            HeaderName::from_static("anthropic-beta"),
            "anything-goes".parse().unwrap(),
        );
        let mut ctx = make_filter_context(&request);

        let body = Bytes::from_static(
            br#"{"model":"claude-sonnet-5","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}"#,
        );
        run_request_body(filter.as_ref(), &mut ctx, body).await.unwrap();
        assert!(
            ctx.request_headers_to_set
                .iter()
                .all(|(name, _)| name != ANTHROPIC_BETA)
        );
        assert!(ctx.request_headers_to_remove.iter().all(|name| name != ANTHROPIC_BETA));
    }

    #[tokio::test]
    async fn google_error_response_translated_on_buffered_body() {
        let filter = filter("project: demo");
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);
        ctx.set_metadata(OPERATION_KEY, "messages");

        let mut response = make_response();
        response.status = StatusCode::TOO_MANY_REQUESTS;
        response
            .headers
            .insert(header::CONTENT_TYPE, HeaderValue::from_static("application/json"));
        ctx.response_header = Some(&mut response);

        filter.on_response(&mut ctx).await.unwrap();
        assert_eq!(ctx.get_metadata(TRANSFORM_KEY), Some(TRANSFORM_ERROR));

        let mut body = Some(Bytes::from_static(
            br#"{"error":{"code":429,"message":"Quota exceeded.","status":"RESOURCE_EXHAUSTED"}}"#,
        ));
        filter.on_response_body(&mut ctx, &mut body, true).unwrap();
        let translated: serde_json::Value = serde_json::from_slice(body.unwrap().as_ref()).unwrap();
        assert_eq!(translated["type"], "error");
        assert_eq!(translated["error"]["type"], "rate_limit_error");
        assert_eq!(translated["error"]["message"], "Quota exceeded. [RESOURCE_EXHAUSTED]");
    }

    #[tokio::test]
    async fn success_response_model_restored() {
        let filter = filter("project: demo");
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);
        ctx.set_metadata(OPERATION_KEY, "messages");
        ctx.set_metadata(MODEL_KEY, "vertex/claude-sonnet-4-5");

        let mut response = make_response();
        response
            .headers
            .insert(header::CONTENT_TYPE, HeaderValue::from_static("application/json"));
        ctx.response_header = Some(&mut response);
        filter.on_response(&mut ctx).await.unwrap();

        let mut body = Some(Bytes::from_static(
            br#"{"id":"msg_vrtx_1","type":"message","model":"claude-sonnet-4-5-20250929","usage":{"input_tokens":2}}"#,
        ));
        filter.on_response_body(&mut ctx, &mut body, true).unwrap();
        let out: serde_json::Value = serde_json::from_slice(body.unwrap().as_ref()).unwrap();
        assert_eq!(out["model"], "vertex/claude-sonnet-4-5");
        assert_eq!(out["usage"]["input_tokens"], 2, "metering payload preserved");
    }

    #[tokio::test]
    async fn sse_model_split_across_chunk_boundary_is_still_restored() {
        // The plan-of-record's mandatory test: the `model` string of the
        // message_start event is split across two wire chunks. Nothing
        // may leak a half-frame, and the completed frame must carry the
        // restored model.
        let filter = filter("project: demo");
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);
        ctx.set_metadata(OPERATION_KEY, "messages");
        ctx.set_metadata(MODEL_KEY, "vertex/claude-sonnet-4-5");

        let mut response = make_response();
        response
            .headers
            .insert(header::CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
        ctx.response_header = Some(&mut response);
        // filter_state is keyed by the executing filter's id; unit tests
        // must stand in for the pipeline executor.
        ctx.current_filter_id = Some(0);
        filter.on_response(&mut ctx).await.unwrap();
        assert_eq!(ctx.get_metadata(TRANSFORM_KEY), Some(TRANSFORM_SSE));

        let event = b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"model\":\"claude-sonnet-4-5-20250929\"}}\n\n";
        // Split exactly before the `model` key name: neither half is a
        // complete frame, and the model string itself straddles chunks.
        let split = event.windows(5).position(|w| w == b"model").expect("model key present");
        let (chunk1, chunk2) = event.split_at(split);

        let mut body1 = Some(Bytes::copy_from_slice(chunk1));
        filter.on_response_body(&mut ctx, &mut body1, false).unwrap();
        let emitted1 = body1.unwrap();
        assert!(
            emitted1.is_empty(),
            "incomplete frame must stay in the parser, got {emitted1:?}"
        );

        let mut body2 = Some(Bytes::copy_from_slice(chunk2));
        filter.on_response_body(&mut ctx, &mut body2, true).unwrap();
        let emitted2 = String::from_utf8(body2.unwrap().to_vec()).unwrap();
        assert!(
            emitted2.contains("\"model\":\"vertex/claude-sonnet-4-5\""),
            "restored after reassembly: {emitted2}"
        );
        assert!(
            !emitted2.contains("20250929"),
            "snapshot id replaced across the boundary: {emitted2}"
        );
    }
}
