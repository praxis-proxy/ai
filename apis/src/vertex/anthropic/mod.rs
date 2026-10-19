// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Anthropic Messages ⇄ Vertex AI `rawPredict` dialect filter.
//!
//! **Experimental.** Requires the `vertex-anthropic-filter` cargo feature,
//! which is off by default and activates the `experimental` marker. The
//! configuration surface may change between releases.
//!
//! Vertex serves Claude through the Anthropic Messages wire format but
//! with a dialect seam on each side of the request. This filter closes
//! both after the load balancer selects a cluster whose HTTP metadata declares
//! `application_provider: vertex`. Other providers pass through unchanged.
//! Place `model_to_provider` earlier in the pipeline to select the upstream
//! and rewrite the body to its exact target model, including any snapshot.
//!
//! **Request**: the body's `model` moves into the URL
//! (`…/publishers/anthropic/models/{model}:rawPredict`,
//! `:streamRawPredict` when `stream` is `true`, `stream` read before the
//! body is rewritten), `anthropic_version` is injected, and the `model`
//! field is removed, since Vertex rejects it with
//! `model: Extra inputs are not permitted`. `count_tokens` keeps its
//! model in the body at a distinct URL.
//!
//! **Response**: the snapshot `model` id is restored to the
//! user-facing id (top-level JSON, or `message.model` inside the
//! `message_start` SSE event, patched frame-scoped without buffering
//! the stream), and Google error envelopes are translated to Anthropic
//! error types so client retry logic keeps working. Translation is
//! pre-stream only: a mid-stream kill has no body to rewrite.
//!
//! Pair with the [`gcp_adc` filter](https://github.com/praxis-proxy/ai)
//! (`source: metadata`) and a `vertex` cluster pointing at
//! `aiplatform.googleapis.com:443` (`tls.sni` likewise); this filter
//! rewrites path and bodies, not credentials or Host.
//!
//! [`AnthropicMessagesToVertexaiAnthropicFilter`]

mod config;
mod request;
mod response;
mod sse;

use async_trait::async_trait;
use bytes::Bytes;
use http::{HeaderName, HeaderValue};
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, Rejection,
    SelectedUpstreamBodyOutcome, parse_filter_config,
};
use tracing::debug;

use self::{
    config::{FILTER_NAME, VertexAnthropicConfig, build_config},
    request::{Operation, TransformedRequest, classify, transform_request},
};
use crate::{
    MODEL_PROVIDER_CLIENT_MODEL_METADATA,
    anthropic::{ErrorType, error_rejection, invalid_request_rejection},
};

/// Metadata key carrying the classified operation (`messages`,
/// `count_tokens`). Absent means "not a Vertex-handled request": no
/// request, response, or SSE transform may run.
const OPERATION_KEY: &str = "vertex_anthropic.operation";
/// Metadata key carrying the user-facing model id to restore in
/// responses.
const MODEL_KEY: &str = "vertex_anthropic.model";
/// Metadata key selecting the response transformation mode.
const TRANSFORM_KEY: &str = "vertex_anthropic.response_transform";
/// Response transform marker for a successful JSON response.
const TRANSFORM_SUCCESS: &str = "success";
/// Response transform marker for an upstream error.
const TRANSFORM_ERROR: &str = "error";
/// Response transform marker for an SSE stream.
const TRANSFORM_SSE: &str = "sse";
/// Metadata key preserving the upstream status for the body phase.
const STATUS_KEY: &str = "vertex_anthropic.response_status";

/// `anthropic-beta` header name, for the beta-flag allowlist.
const ANTHROPIC_BETA: HeaderName = HeaderName::from_static("anthropic-beta");
/// Anthropic API key header. Vertex authenticates with a Google bearer
/// token set by the credential filters, so the client's Anthropic key must
/// never reach Google.
const X_API_KEY: HeaderName = HeaderName::from_static("x-api-key");
/// Translates Anthropic Messages requests to Vertex AI `rawPredict` and
/// Vertex responses back to the Anthropic dialect.
///
/// Activates after upstream selection only for clusters whose HTTP metadata
/// declares `application_provider: vertex`. The earlier `model_to_provider`
/// filter owns upstream selection and the exact target model, including snapshot
/// suffixes. This adapter never selects a provider or maps model names.
///
/// Responses restore the original client model from
/// `model_to_provider.client_model` metadata, falling back to the target model
/// when no mapping recorded a public alias.
///
/// Streaming preserves SSE comments, event IDs, and retry fields. Oversized or
/// incomplete frames emit an Anthropic error event after complete events and
/// suppress the remaining stream.
///
/// Experimental: requires the `vertex-anthropic-filter` cargo feature,
/// which is off by default and activates the `experimental` marker. This
/// filter is a work in progress and its configuration surface may change
/// between releases.
///
/// # YAML
///
/// ```yaml
/// filter: anthropic_messages_to_vertexai_anthropic
/// project: my-gcp-project
/// ```
///
/// # Full YAML
///
/// ```yaml
/// filter: anthropic_messages_to_vertexai_anthropic
/// project: my-gcp-project
/// location: global
/// beta_allowlist: [context-1m-2025-08-07, interleaved-thinking-2025-05-14]
/// max_body_bytes: 33554432
/// ```
pub struct AnthropicMessagesToVertexaiAnthropicFilter {
    /// Parsed and validated configuration.
    config: VertexAnthropicConfig,
}

impl AnthropicMessagesToVertexaiAnthropicFilter {
    /// Create a filter from parsed YAML config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the YAML config is invalid or a URL
    /// component carries unsafe characters.
    ///
    /// ```
    /// use praxis_ai_apis::vertex::AnthropicMessagesToVertexaiAnthropicFilter;
    /// let filter = AnthropicMessagesToVertexaiAnthropicFilter::from_config(
    ///     &serde_yaml::from_str("project: my-gcp-project").unwrap(),
    /// )
    /// .unwrap();
    /// assert_eq!(filter.name(), "anthropic_messages_to_vertexai_anthropic");
    /// ```
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: VertexAnthropicConfig = parse_filter_config(FILTER_NAME, config)?;
        Ok(Box::new(Self {
            config: build_config(cfg)?,
        }))
    }

    /// Apply the `anthropic-beta` allowlist. An empty allowlist (the
    /// default) forwards the header untouched; otherwise unknown flags
    /// are stripped and the header removed entirely when nothing
    /// remains, since Vertex rejects beta flags it does not support, and
    /// clients like Claude Code send several on every request.
    fn filter_beta_flags(&self, ctx: &mut HttpFilterContext<'_>) {
        if self.config.beta_allowlist.is_empty() || !has_beta_header(ctx) {
            return;
        }
        let kept: Vec<&str> = effective_beta_values(ctx)
            .into_iter()
            .flat_map(|value| value.split(','))
            .map(str::trim)
            .filter(|flag| !flag.is_empty() && self.config.beta_allowlist.iter().any(|allowed| allowed == flag))
            .collect();

        if kept.is_empty() {
            queue_header_removal(ctx, ANTHROPIC_BETA);
        } else if let Ok(joined) = kept.join(", ").parse() {
            queue_header_set(ctx, ANTHROPIC_BETA, joined);
        }
    }

    /// Install a translated request: header hygiene, the
    /// metadata the response side keys on, and the rewritten body.
    fn apply_transformed(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        operation: Operation,
        transformed: TransformedRequest,
    ) {
        self.filter_beta_flags(ctx);
        // Responses are patched in place, so Vertex must answer uncompressed.
        queue_header_removal(ctx, http::header::ACCEPT_ENCODING);
        queue_header_removal(ctx, X_API_KEY);
        // When model_to_provider mapped a public model id onto the Vertex
        // target earlier in the chain, the client sent that public id, so it
        // is the one responses must carry back.
        let user_model = ctx
            .get_metadata(MODEL_PROVIDER_CLIENT_MODEL_METADATA)
            .map_or(transformed.target_model, str::to_owned);
        debug!(
            model = %user_model,
            path = %transformed.path,
            "translated Anthropic request to Vertex rawPredict"
        );
        ctx.rewritten_path = Some(transformed.path);
        // Marks "this request was transformed"; response-side transforms only
        // run for marked requests after selected-upstream translation succeeds.
        ctx.set_metadata(OPERATION_KEY, operation.as_str());
        ctx.set_metadata(MODEL_KEY, user_model);
        *body = Some(Bytes::from(transformed.body));
    }
}

#[async_trait]
impl HttpFilter for AnthropicMessagesToVertexaiAnthropicFilter {
    fn name(&self) -> &'static str {
        "anthropic_messages_to_vertexai_anthropic"
    }

    fn selected_upstream_request_body_access(&self) -> BodyAccess {
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
        // The selected-upstream hook classifies the original Anthropic path
        // and reads request headers for provider-specific header hygiene.
        true
    }

    async fn on_request(&self, _ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        Ok(FilterAction::Continue)
    }

    async fn on_selected_upstream_request_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
    ) -> Result<SelectedUpstreamBodyOutcome, FilterError> {
        if ctx.selected_application_provider() != Some("vertex") {
            return Ok(SelectedUpstreamBodyOutcome::Continue);
        }
        let Some(operation) = classify(current_path(ctx)) else {
            return Ok(SelectedUpstreamBodyOutcome::Continue);
        };
        let Some(bytes) = body.as_ref().filter(|bytes| !bytes.is_empty()) else {
            return Ok(SelectedUpstreamBodyOutcome::Continue);
        };
        match transform_request(bytes, operation, &self.config) {
            Ok(transformed) => {
                self.apply_transformed(ctx, body, operation, transformed);
                Ok(SelectedUpstreamBodyOutcome::Continue)
            },
            Err(error) => Ok(SelectedUpstreamBodyOutcome::Reject(invalid_request_rejection(
                &error.to_string(),
            ))),
        }
    }

    async fn on_response(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        // A response to a request this filter did not transform (wrong
        // path, or an upstream hit via another route) must not be
        // transformed either.
        if ctx.get_metadata(OPERATION_KEY).is_none() {
            return Ok(FilterAction::Continue);
        }
        if let Some(rejection) = encoded_response_rejection(ctx) {
            return Ok(FilterAction::Reject(rejection));
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
            ctx.insert_filter_state(sse::WireParser::new(self.config.max_body_bytes));
        } else {
            ctx.set_response_body_mode(BodyMode::StreamBuffer {
                max_bytes: Some(self.config.max_body_bytes),
            });
        }
        if let Some(resp) = &mut ctx.response_header {
            resp.headers.remove(http::header::CONTENT_LENGTH);
            resp.headers.remove(http::header::CONTENT_ENCODING);
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
// Request Header Helpers
// -----------------------------------------------------------------------------

/// Whether any original or pending request value supplies beta flags.
fn has_beta_header(ctx: &HttpFilterContext<'_>) -> bool {
    ctx.request.headers.contains_key(&ANTHROPIC_BETA)
        || ctx
            .request_headers_to_set
            .iter()
            .any(|(name, _)| *name == ANTHROPIC_BETA)
        || ctx
            .extra_request_headers
            .iter()
            .any(|(name, _)| name.eq_ignore_ascii_case(ANTHROPIC_BETA.as_str()))
}

/// Read beta values after the request phase's pending remove/set/add mutations.
/// Only this header's borrowed values are collected; the full header map stays borrowed.
fn effective_beta_values<'a>(ctx: &'a HttpFilterContext<'_>) -> Vec<&'a str> {
    let mut values: Vec<_> = ctx
        .request
        .headers
        .get_all(&ANTHROPIC_BETA)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .collect();
    if ctx.request_headers_to_remove.contains(&ANTHROPIC_BETA) {
        values.clear();
    }
    if let Some((_, value)) = ctx
        .request_headers_to_set
        .iter()
        .rev()
        .find(|(name, _)| *name == ANTHROPIC_BETA)
    {
        values.clear();
        values.extend(value.to_str().ok());
    }
    for (name, value) in &ctx.extra_request_headers {
        if name.eq_ignore_ascii_case(ANTHROPIC_BETA.as_str()) && HeaderValue::from_str(value).is_ok() {
            values.clear();
            values.push(value);
        }
    }
    values
}

/// Remove a selected-upstream request header, including earlier pending writes.
fn queue_header_removal(ctx: &mut HttpFilterContext<'_>, name: HeaderName) {
    ctx.request_headers_to_set.retain(|(pending, _)| *pending != name);
    ctx.extra_request_headers
        .retain(|(pending, _)| !pending.eq_ignore_ascii_case(name.as_str()));
    ctx.request_headers_to_remove.push(name);
}

/// Replace a selected-upstream request header after earlier pending mutations.
fn queue_header_set(ctx: &mut HttpFilterContext<'_>, name: HeaderName, value: HeaderValue) {
    ctx.request_headers_to_set.retain(|(pending, _)| *pending != name);
    ctx.extra_request_headers
        .retain(|(pending, _)| !pending.eq_ignore_ascii_case(name.as_str()));
    ctx.request_headers_to_set.push((name, value));
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

/// Fail closed on a response that is still content-encoded.
///
/// Handled requests drop `Accept-Encoding`, so Vertex answers uncompressed.
/// A response encoded anyway cannot have its model restored or its Google
/// error translated, and passing it through would leak the Vertex dialect,
/// so it becomes an Anthropic error instead. An upstream error status is kept
/// for client retry policies; an encoded success becomes `502`.
fn encoded_response_rejection(ctx: &HttpFilterContext<'_>) -> Option<Rejection> {
    let upstream = ctx.response_header.as_ref()?;
    let encoded = upstream
        .headers
        .get_all(http::header::CONTENT_ENCODING)
        .iter()
        .any(|value| !value.as_bytes().trim_ascii().eq_ignore_ascii_case(b"identity"));
    if !encoded {
        return None;
    }

    let status = if upstream.status.is_client_error() || upstream.status.is_server_error() {
        upstream.status.as_u16()
    } else {
        502
    };
    debug!(status, "vertex: upstream response is content-encoded; failing closed");
    Some(error_rejection(
        status,
        ErrorType::from_status(status),
        "Vertex AI returned a content-encoded response that cannot be translated",
    ))
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
    let Some(mut parser) = ctx.remove_filter_state::<sse::WireParser>() else {
        return;
    };
    let initial_frame = parser.completed_frames == 0;
    let output = if parser.failed {
        Vec::new()
    } else {
        match parser.parse_chunk(body.as_deref().unwrap_or_default(), end_of_stream) {
            Ok(frames) => response::rebuild_sse_frames(&frames, ctx.get_metadata(MODEL_KEY), initial_frame),
            Err(error) => {
                debug!(error = error.message, "vertex: SSE translation failed");
                let mut output =
                    response::rebuild_sse_frames(&error.frames, ctx.get_metadata(MODEL_KEY), initial_frame);
                output.extend_from_slice(&response::sse_error(error.message));
                output
            },
        }
    };
    if !end_of_stream {
        ctx.insert_filter_state(parser);
    }
    *body = Some(Bytes::from(output));
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
        AnthropicMessagesToVertexaiAnthropicFilter::from_config(&config).unwrap()
    }

    fn messages_body(extra: &str) -> Bytes {
        Bytes::from(format!(
            r#"{{"model":"claude-sonnet-4-5",{extra},"messages":[{{"role":"user","content":"hi"}}]}}"#
        ))
    }

    async fn run_request_body(
        filter: &dyn HttpFilter,
        ctx: &mut HttpFilterContext<'_>,
        body: Bytes,
    ) -> Result<SelectedUpstreamBodyOutcome, FilterError> {
        if ctx.upstream.is_none() {
            select_provider(ctx, Some("vertex")).await;
        }
        assert!(matches!(filter.on_request(ctx).await?, FilterAction::Continue));
        let mut body = Some(body);
        let action = filter.on_selected_upstream_request_body(ctx, &mut body).await?;
        ctx.buffered_request_body = body;
        Ok(action)
    }

    /// Publish provider metadata through the real load-balancer boundary.
    async fn select_provider(ctx: &mut HttpFilterContext<'_>, provider: Option<&str>) {
        let http = provider.map_or_else(String::new, |provider| {
            format!("\n    http: {{application_provider: {provider}}}")
        });
        let config = serde_yaml::from_str(&format!(
            "clusters:\n  - name: target{http}\n    endpoints: [127.0.0.1:3000]"
        ))
        .unwrap();
        let balancer = praxis_filter::LoadBalancerFilter::from_config(&config).unwrap();
        ctx.cluster = Some("target".into());
        ctx.upstream = None;
        balancer.on_request(ctx).await.unwrap();
        balancer.on_response(ctx).await.unwrap();
    }

    #[tokio::test]
    async fn messages_request_becomes_rawpredict_url() {
        let filter = filter("project: demo");
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);

        let action = run_request_body(filter.as_ref(), &mut ctx, messages_body(r#""max_tokens":8"#))
            .await
            .unwrap();
        assert!(matches!(action, SelectedUpstreamBodyOutcome::Continue));
        assert_eq!(
            ctx.rewritten_path.as_deref(),
            Some("/v1/projects/demo/locations/global/publishers/anthropic/models/claude-sonnet-4-5:rawPredict")
        );
        let body: serde_json::Value =
            serde_json::from_slice(ctx.buffered_request_body.as_ref().unwrap().as_ref()).unwrap();
        assert!(body.get("model").is_none());
        assert_eq!(body["anthropic_version"], "vertex-2023-10-16");
        assert_eq!(ctx.get_metadata(MODEL_KEY), Some("claude-sonnet-4-5"));
        assert_eq!(ctx.get_metadata(OPERATION_KEY), Some("messages"));
        assert!(
            ctx.request_headers_to_set.is_empty(),
            "the adapter must not select a route"
        );
    }

    #[test]
    fn body_translation_waits_for_upstream_selection() {
        let filter = filter("project: demo");
        assert_eq!(filter.request_body_access(), BodyAccess::None);
        assert_eq!(filter.selected_upstream_request_body_access(), BodyAccess::ReadWrite);
    }

    #[tokio::test]
    async fn non_vertex_and_untagged_upstreams_pass_through_even_malformed_bodies() {
        let filter = filter("project: demo");
        for provider in [Some("anthropic"), None] {
            for raw in ["not json", r#"{"model":"vertex/claude-sonnet-4-5","messages":[]}"#] {
                let request = make_request(Method::POST, "/v1/messages");
                let mut ctx = make_filter_context(&request);
                select_provider(&mut ctx, provider).await;
                let original = Bytes::from(raw);
                let action = run_request_body(filter.as_ref(), &mut ctx, original.clone())
                    .await
                    .unwrap();
                assert!(matches!(action, SelectedUpstreamBodyOutcome::Continue));
                assert_eq!(ctx.buffered_request_body.as_ref(), Some(&original));
                assert!(ctx.rewritten_path.is_none());
                assert!(ctx.get_metadata(OPERATION_KEY).is_none());
                assert!(ctx.request_headers_to_set.is_empty());
                assert!(ctx.request_headers_to_remove.is_empty());
            }
        }
    }

    #[tokio::test]
    async fn missing_selection_does_not_activate_translation() {
        let filter = filter("project: demo");
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);
        let mut body = Some(Bytes::from_static(b"not json"));
        assert!(matches!(
            filter
                .on_selected_upstream_request_body(&mut ctx, &mut body)
                .await
                .unwrap(),
            SelectedUpstreamBodyOutcome::Continue
        ));
        assert!(ctx.rewritten_path.is_none());
    }

    #[tokio::test]
    async fn handled_request_drops_accept_encoding() {
        let filter = filter("project: demo");
        let mut request = make_request(Method::POST, "/v1/messages");
        request
            .headers
            .insert(header::ACCEPT_ENCODING, HeaderValue::from_static("gzip, br"));
        let mut ctx = make_filter_context(&request);

        run_request_body(filter.as_ref(), &mut ctx, messages_body(r#""max_tokens":8"#))
            .await
            .unwrap();
        assert!(
            ctx.request_headers_to_remove.contains(&header::ACCEPT_ENCODING),
            "Vertex must answer uncompressed so the response can be patched"
        );
    }

    #[tokio::test]
    async fn handled_request_drops_anthropic_api_key() {
        let filter = filter("project: demo");
        let mut request = make_request(Method::POST, "/v1/messages");
        request
            .headers
            .insert(X_API_KEY, HeaderValue::from_static("sk-ant-client-key"));
        request
            .headers
            .insert(header::AUTHORIZATION, HeaderValue::from_static("Bearer ya29.token"));
        let mut ctx = make_filter_context(&request);

        run_request_body(filter.as_ref(), &mut ctx, messages_body(r#""max_tokens":8"#))
            .await
            .unwrap();
        assert!(
            ctx.request_headers_to_remove.contains(&X_API_KEY),
            "the client's Anthropic key must not reach Google"
        );
        assert!(
            !ctx.request_headers_to_remove.contains(&header::AUTHORIZATION),
            "Authorization belongs to the credential filters"
        );
    }

    #[tokio::test]
    async fn selected_header_hygiene_replaces_pending_writes() {
        let filter = filter("project: demo\nbeta_allowlist: [allowed]");
        let mut request = make_request(Method::POST, "/v1/messages");
        request
            .headers
            .insert(ANTHROPIC_BETA, HeaderValue::from_static("original"));
        let mut ctx = make_filter_context(&request);
        ctx.request_headers_to_set
            .push((X_API_KEY, HeaderValue::from_static("earlier-key")));
        ctx.extra_request_headers
            .push(("accept-encoding".into(), "gzip".to_owned()));
        ctx.extra_request_headers
            .push(("anthropic-beta".into(), "allowed, unknown".to_owned()));
        run_request_body(filter.as_ref(), &mut ctx, messages_body(r#""max_tokens":8"#))
            .await
            .unwrap();
        assert!(ctx.request_headers_to_set.iter().all(|(name, _)| *name != X_API_KEY));
        assert!(ctx.extra_request_headers.is_empty());
        assert!(
            ctx.request_headers_to_set
                .iter()
                .any(|(name, value)| *name == ANTHROPIC_BETA && value == "allowed")
        );
        assert!(ctx.request_headers_to_remove.contains(&X_API_KEY));
        assert!(ctx.request_headers_to_remove.contains(&header::ACCEPT_ENCODING));
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
        assert_eq!(ctx.get_metadata(OPERATION_KEY), Some("count_tokens"));
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
            assert!(matches!(action, SelectedUpstreamBodyOutcome::Continue));
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
        let SelectedUpstreamBodyOutcome::Reject(rejection) = action else {
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
        assert!(
            !ctx.request_headers_to_remove.contains(&ANTHROPIC_BETA),
            "an allowed flag keeps the header"
        );
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
        assert!(
            ctx.request_headers_to_remove.contains(&ANTHROPIC_BETA),
            "a header with no allowed flag left must be removed"
        );
        assert!(
            ctx.request_headers_to_set
                .iter()
                .all(|(name, _)| name != ANTHROPIC_BETA)
        );
    }

    /// Combine repeated field values per [RFC 9110 Section 5.3].
    ///
    /// [RFC 9110 Section 5.3]: https://datatracker.ietf.org/doc/html/rfc9110#section-5.3
    #[tokio::test]
    async fn beta_allowlist_combines_valid_instances_and_ignores_obs_text() {
        let filter = filter("project: demo\nbeta_allowlist: [context-1m-2025-08-07, interleaved-thinking-2025-05-14]");
        let mut request = make_request(Method::POST, "/v1/messages");
        for value in [
            HeaderValue::from_static("context-1m-2025-08-07, unknown"),
            HeaderValue::from_bytes(b"interleaved-thinking-2025-05-14, \xff").unwrap(),
            HeaderValue::from_static("interleaved-thinking-2025-05-14"),
        ] {
            request.headers.append(ANTHROPIC_BETA, value);
        }
        let mut ctx = make_filter_context(&request);
        run_request_body(filter.as_ref(), &mut ctx, messages_body(r#""max_tokens":8"#))
            .await
            .unwrap();
        let values: Vec<_> = ctx
            .request_headers_to_set
            .iter()
            .filter(|(name, _)| name == ANTHROPIC_BETA)
            .collect();
        assert_eq!(
            values.len(),
            1,
            "all header instances must be replaced by one filtered value"
        );
        assert_eq!(
            values[0].1.to_str().unwrap(),
            "context-1m-2025-08-07, interleaved-thinking-2025-05-14",
            "later allowed flags must survive and invalid values must contribute no flags"
        );
    }

    #[tokio::test]
    async fn beta_allowlist_removes_obs_text_only_header() {
        let filter = filter("project: demo\nbeta_allowlist: [context-1m-2025-08-07]");
        let mut request = make_request(Method::POST, "/v1/messages");
        request.headers.insert(
            ANTHROPIC_BETA,
            HeaderValue::from_bytes(b"context-1m-2025-08-07, \xff").unwrap(),
        );
        let mut ctx = make_filter_context(&request);
        run_request_body(filter.as_ref(), &mut ctx, messages_body(r#""max_tokens":8"#))
            .await
            .unwrap();
        assert!(
            ctx.request_headers_to_remove.contains(&ANTHROPIC_BETA),
            "invalid field values must not bypass the allowlist"
        );
        assert!(
            !ctx.request_headers_to_set
                .iter()
                .any(|(name, _)| name == ANTHROPIC_BETA),
            "an invalid value must never be forwarded as an allowed flag"
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
        request
            .headers
            .append(ANTHROPIC_BETA, HeaderValue::from_bytes(b"\xff").unwrap());
        let mut ctx = make_filter_context(&request);

        run_request_body(filter.as_ref(), &mut ctx, messages_body(r#""max_tokens":8"#))
            .await
            .unwrap();
        assert!(
            ctx.request_headers_to_set
                .iter()
                .all(|(name, _)| name != ANTHROPIC_BETA)
        );
        assert!(ctx.request_headers_to_remove.iter().all(|name| name != ANTHROPIC_BETA));
    }

    #[tokio::test]
    async fn beta_allowlist_keeps_an_absent_header_absent() {
        let filter = filter("project: demo\nbeta_allowlist: [context-1m-2025-08-07]");
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);
        run_request_body(filter.as_ref(), &mut ctx, messages_body(r#""max_tokens":8"#))
            .await
            .unwrap();
        assert!(
            !ctx.request_headers_to_remove.contains(&ANTHROPIC_BETA),
            "an absent beta header needs no mutation"
        );
        assert!(
            !ctx.request_headers_to_set
                .iter()
                .any(|(name, _)| name == ANTHROPIC_BETA),
            "the allowlist must never inject a beta header"
        );
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
    async fn encoded_json_response_fails_closed() {
        let filter = filter("project: demo");
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);
        ctx.set_metadata(OPERATION_KEY, "messages");
        ctx.set_metadata(MODEL_KEY, "claude-sonnet-4-5");

        let mut response = make_response();
        response
            .headers
            .insert(header::CONTENT_TYPE, HeaderValue::from_static("application/json"));
        response
            .headers
            .insert(header::CONTENT_ENCODING, HeaderValue::from_static("gzip"));
        ctx.response_header = Some(&mut response);

        let action = filter.on_response(&mut ctx).await.unwrap();
        let FilterAction::Reject(rejection) = action else {
            panic!("an encoded body cannot be restored and must not pass through, got {action:?}");
        };
        assert_eq!(rejection.status, 502, "an encoded success becomes a gateway error");
        let body: serde_json::Value = serde_json::from_slice(rejection.body.as_ref().unwrap().as_ref()).unwrap();
        assert_eq!(body["type"], "error");
        assert_eq!(body["error"]["type"], "api_error");
        assert!(
            ctx.get_metadata(TRANSFORM_KEY).is_none(),
            "no body transform may run on an encoded response"
        );
    }

    #[tokio::test]
    async fn encoded_sse_response_fails_closed() {
        let filter = filter("project: demo");
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);
        ctx.set_metadata(OPERATION_KEY, "messages");
        ctx.set_metadata(MODEL_KEY, "claude-sonnet-4-5");

        let mut response = make_response();
        response
            .headers
            .insert(header::CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
        response
            .headers
            .insert(header::CONTENT_ENCODING, HeaderValue::from_static("br"));
        ctx.response_header = Some(&mut response);
        ctx.current_filter_id = Some(0);

        let action = filter.on_response(&mut ctx).await.unwrap();
        let FilterAction::Reject(rejection) = action else {
            panic!("encoded SSE bytes yield no frames and must not pass through, got {action:?}");
        };
        assert_eq!(rejection.status, 502);
        let body: serde_json::Value = serde_json::from_slice(rejection.body.as_ref().unwrap().as_ref()).unwrap();
        assert_eq!(body["error"]["type"], "api_error");
        assert!(
            ctx.get_metadata(TRANSFORM_KEY).is_none(),
            "no SSE parser may be armed for an encoded stream"
        );
    }

    #[tokio::test]
    async fn encoded_error_response_keeps_upstream_status() {
        let filter = filter("project: demo");
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);
        ctx.set_metadata(OPERATION_KEY, "messages");

        let mut response = make_response();
        response.status = StatusCode::TOO_MANY_REQUESTS;
        response
            .headers
            .insert(header::CONTENT_ENCODING, HeaderValue::from_static("gzip"));
        ctx.response_header = Some(&mut response);

        let action = filter.on_response(&mut ctx).await.unwrap();
        let FilterAction::Reject(rejection) = action else {
            panic!("an encoded Google error cannot be translated, got {action:?}");
        };
        assert_eq!(
            rejection.status, 429,
            "retry policies must still see the upstream status"
        );
        let body: serde_json::Value = serde_json::from_slice(rejection.body.as_ref().unwrap().as_ref()).unwrap();
        assert_eq!(body["error"]["type"], "rate_limit_error");
    }

    #[tokio::test]
    async fn identity_encoded_response_is_restored() {
        let filter = filter("project: demo");
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);
        ctx.set_metadata(OPERATION_KEY, "messages");
        ctx.set_metadata(MODEL_KEY, "claude-sonnet-4-5");

        let mut response = make_response();
        response
            .headers
            .insert(header::CONTENT_TYPE, HeaderValue::from_static("application/json"));
        response
            .headers
            .insert(header::CONTENT_ENCODING, HeaderValue::from_static("identity"));
        response
            .headers
            .insert(header::CONTENT_LENGTH, HeaderValue::from_static("79"));
        ctx.response_header = Some(&mut response);

        let action = filter.on_response(&mut ctx).await.unwrap();
        assert!(matches!(action, FilterAction::Continue));
        let headers = &ctx.response_header.as_ref().unwrap().headers;
        assert!(
            !headers.contains_key(header::CONTENT_ENCODING),
            "the rewritten body carries no content coding"
        );
        assert!(
            !headers.contains_key(header::CONTENT_LENGTH),
            "the rewritten body length differs from upstream"
        );

        let mut body = Some(Bytes::from_static(
            br#"{"id":"msg_vrtx_1","type":"message","model":"claude-sonnet-4-5-20250929"}"#,
        ));
        filter.on_response_body(&mut ctx, &mut body, true).unwrap();
        let out: serde_json::Value = serde_json::from_slice(body.unwrap().as_ref()).unwrap();
        assert_eq!(out["model"], "claude-sonnet-4-5");
    }

    #[tokio::test]
    async fn success_response_model_restored() {
        let filter = filter("project: demo");
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);
        ctx.set_metadata(OPERATION_KEY, "messages");
        ctx.set_metadata(MODEL_KEY, "claude-sonnet-4-5");

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
        assert_eq!(out["model"], "claude-sonnet-4-5");
        assert_eq!(out["usage"]["input_tokens"], 2, "metering payload preserved");
    }

    #[tokio::test]
    async fn model_to_provider_client_model_is_restored_instead_of_the_target() {
        let filter = filter("project: demo");
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);
        // model_to_provider mapped a public alias before upstream selection.
        ctx.set_metadata(MODEL_PROVIDER_CLIENT_MODEL_METADATA, "public-claude");

        run_request_body(filter.as_ref(), &mut ctx, messages_body(r#""max_tokens":8"#))
            .await
            .unwrap();
        assert_eq!(
            ctx.rewritten_path.as_deref(),
            Some("/v1/projects/demo/locations/global/publishers/anthropic/models/claude-sonnet-4-5:rawPredict"),
            "the resolved target supplies the publisher model"
        );
        assert_eq!(ctx.get_metadata(MODEL_KEY), Some("public-claude"));

        let mut response = make_response();
        response
            .headers
            .insert(header::CONTENT_TYPE, HeaderValue::from_static("application/json"));
        ctx.response_header = Some(&mut response);
        filter.on_response(&mut ctx).await.unwrap();

        let mut body = Some(Bytes::from_static(
            br#"{"id":"msg_vrtx_2","type":"message","model":"claude-sonnet-4-5-20250929"}"#,
        ));
        filter.on_response_body(&mut ctx, &mut body, true).unwrap();
        let out: serde_json::Value = serde_json::from_slice(body.unwrap().as_ref()).unwrap();
        assert_eq!(
            out["model"], "public-claude",
            "the client's public id must come back, not the Vertex target"
        );
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
        ctx.set_metadata(MODEL_KEY, "public-claude");

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
            emitted2.contains("\"model\":\"public-claude\""),
            "restored after reassembly: {emitted2}"
        );
        assert!(
            !emitted2.contains("20250929"),
            "snapshot id replaced across the boundary: {emitted2}"
        );
    }
    #[tokio::test]
    async fn sse_failure_emits_anthropic_error_and_suppresses_later_success() {
        let filter = filter("project: demo\nmax_body_bytes: 64");
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);
        ctx.set_metadata(OPERATION_KEY, "messages");
        ctx.current_filter_id = Some(0);
        let mut response = make_response();
        response
            .headers
            .insert(header::CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
        ctx.response_header = Some(&mut response);
        filter.on_response(&mut ctx).await.unwrap();
        let valid = "event: ping\ndata: {\"type\":\"ping\"}\n\n";
        let mut body = Some(Bytes::from(format!("{valid}data: {}\n\n", "x".repeat(128))));
        filter.on_response_body(&mut ctx, &mut body, false).unwrap();
        let output = String::from_utf8(body.unwrap().to_vec()).unwrap();
        assert!(
            output.starts_with(valid) && output.contains("event: error\ndata: "),
            "completed events and SSE failures must both reach the client"
        );
        assert!(output.contains("api_error"));
        let mut later = Some(Bytes::from_static(b"event: message_stop\ndata: {}\n\n"));
        filter.on_response_body(&mut ctx, &mut later, true).unwrap();
        assert!(
            later.unwrap().is_empty(),
            "a failed stream must not end with a success event"
        );
    }
}
