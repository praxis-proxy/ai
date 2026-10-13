// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Anthropic Messages to Chat Completions-compatible transformation filter.
//!
//! Rewrites Anthropic Messages request bodies to the Chat Completions
//! request shape, transforms compatible non-streaming successes back, and
//! normalizes pre-stream upstream errors for both request modes. Successful
//! streaming SSE transformation is handled by the separate
//! `anthropic_messages_to_chat_completions_stream` filter.
//!
//! The name refers to the Chat Completions wire shape, not the OpenAI
//! Responses API: any Chat Completions-compatible backend is a valid
//! target, not only OpenAI.

mod config;
pub(crate) mod request;
pub(crate) mod response;

use async_trait::async_trait;
use bytes::Bytes;
use metrics::counter;
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, Rejection,
    SelectedUpstreamBodyOutcome, SubRequestResponseMode, parse_filter_config,
};
use tracing::{debug, warn};

use self::config::{AnthropicMessagesToChatCompletionsConfig, build_config};
use crate::anthropic::wire;

/// Metadata key selecting success or error response transformation.
const RESPONSE_TRANSFORM_KEY: &str = "anthropic_messages_to_chat_completions.response_transform";
/// Response transform marker for a successful response.
const RESPONSE_TRANSFORM_SUCCESS: &str = "success";
/// Response transform marker for an upstream error.
const RESPONSE_TRANSFORM_ERROR: &str = "error";
/// Metadata key preserving the upstream error status for the body phase.
const RESPONSE_STATUS_KEY: &str = "anthropic_messages_to_chat_completions.response_status";
/// Metadata key preserving the upstream request ID for the body phase.
const RESPONSE_REQUEST_ID_KEY: &str = "anthropic_messages_to_chat_completions.response_request_id";
/// Metadata key recording the raw upstream response byte count before any
/// transform shrinks it.
///
/// On the response path the managed `anthropic_web_search` loop runs after this
/// translation, so it observes the already-transformed (potentially smaller)
/// Anthropic body. Recording the pre-transform size lets it enforce its own byte
/// ceiling on the raw upstream round independently of this filter's limit.
pub(crate) const RESPONSE_RAW_BYTES_KEY: &str = "anthropic_messages_to_chat_completions.response_raw_bytes";

/// Per-request accumulator for a streaming request's non-2xx error round body.
///
/// A streaming request whose round returns a non-2xx status must still be
/// normalized to a single Anthropic error. The buffered path ratchets the
/// response body mode to [`BodyMode::StreamBuffer`] in [`on_response`], but a
/// streaming-composed IRR pipeline (for example the managed `anthropic_web_search`
/// loop) ignores that per-request ratchet and delivers the error round as raw
/// `BodyMode::Stream` chunks. Passing those chunks through and then transforming
/// an empty body at end of stream would emit the raw upstream error followed by a
/// second, generic Anthropic error — two concatenated JSON objects. Instead the
/// filter accumulates the raw bytes here, suppresses passthrough, and transforms
/// the whole body exactly once at end of stream.
///
/// [`on_response`]: AnthropicMessagesToChatCompletionsFilter::on_response
#[derive(Default)]
struct StreamingErrorBuffer {
    /// Raw upstream error bytes accumulated so far.
    buf: Vec<u8>,
    /// Set once the accumulated body exceeds `max_body_bytes`; the buffer is then
    /// cleared and end of stream emits a single JSON error instead of the body.
    overflowed: bool,
}

/// Metadata key holding the client's `stop_sequences` as a JSON array.
///
/// Chat Completions reports a matched stop string only through vLLM's
/// choice-level `stop_reason`; the response phase needs the client's list to
/// report it back truthfully as `stop_reason: stop_sequence`.
pub(crate) const STOP_SEQUENCES_KEY: &str = "anthropic_messages_to_chat_completions.stop_sequences";

/// Metadata key carrying the comma-separated degraded-feature labels from the
/// request-body phase to the response-header phase.
const DEGRADED_FEATURES_KEY: &str = "anthropic_messages_to_chat_completions.degraded_features";

/// Response header advertising which Anthropic features the translator degraded.
///
/// Logs and the per-feature counter are the operator's primary signal; this
/// header is a convenience echo for clients and gateways. It is always set from
/// the proxy's own record, replacing any backend-supplied copy.
///
/// The name avoids the Praxis reserved header prefixes (`x-praxis-`,
/// `x-ext-protocol-`, `x-ext-agent-`): the protocol layer strips reserved
/// headers from the client-bound response, so a reserved name would never reach
/// the client.
const DEGRADED_FEATURES_HEADER: &str = "x-degraded-features";

// -----------------------------------------------------------------------------
// AnthropicMessagesToChatCompletionsFilter
// -----------------------------------------------------------------------------

/// Transforms Anthropic Messages API requests to Chat Completions-compatible
/// request bodies and transforms compatible responses back. The name refers to
/// the Chat Completions wire shape, not the OpenAI Responses API; any Chat
/// Completions-compatible backend is a valid target, not only OpenAI.
///
/// Request fields the translation does not map are forwarded untouched for
/// the backend to validate. Fields whose effect the translated response could
/// not report truthfully (`service_tier`, `container`, `inference_geo`,
/// `mcp_servers`, and Chat Completions fields such as `n` or `logprobs` whose
/// output the translated response would discard) are rejected with a 400.
/// Unsupported semantic content is rejected because Chat Completions cannot
/// represent it faithfully.
///
/// `allow_lossy_features` opts specific Anthropic-only features into
/// operator-approved degradation instead of that 400. A listed feature's wire
/// markers are validated and then stripped so an unmodified client (for example
/// Claude Code, which always sends `cache_control` and `thinking`) can still
/// drive a Chat Completions backend. Each degradation is reported to the
/// operator — one `WARN` log per request, a
/// `praxis_anthropic_messages_to_chat_completions_degraded_total` counter per
/// feature, and an `x-degraded-features` response header — because the
/// request succeeds but the feature was silently dropped. `prompt_caching`
/// removes `cache_control` markers (prompt and tool content are preserved, but
/// explicit cache breakpoints are not honored, so cost and latency may differ);
/// `extended_thinking` removes `thinking` and thinking-only `context_management`
/// edits (the translated response carries no thinking blocks). Features absent
/// from the allowlist, malformed markers, and other `context_management` edits
/// are still rejected with a 400.
///
/// # YAML
///
/// ```yaml
/// filter: anthropic_messages_to_chat_completions
/// ```
///
/// # Full YAML
///
/// ```yaml
/// filter: anthropic_messages_to_chat_completions
/// max_body_bytes: 1048576
/// allow_lossy_features:
///   - prompt_caching
///   - extended_thinking
/// ```
pub struct AnthropicMessagesToChatCompletionsFilter {
    /// Parsed and validated configuration.
    config: AnthropicMessagesToChatCompletionsConfig,
}

impl AnthropicMessagesToChatCompletionsFilter {
    /// Create a filter from parsed YAML config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the YAML config is invalid.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: AnthropicMessagesToChatCompletionsConfig =
            parse_filter_config("anthropic_messages_to_chat_completions", config)?;
        let validated = build_config(cfg)?;
        if validated.allowlist().any() {
            warn!(
                "anthropic_messages_to_chat_completions: allow_lossy_features is enabled; matching \
                 requests will have unsupported Anthropic features degraded instead of rejected \
                 (see the x-degraded-features response header and the \
                 praxis_anthropic_messages_to_chat_completions_degraded_total counter)"
            );
        }
        Ok(Box::new(Self { config: validated }))
    }

    /// Accumulate a streaming request's non-2xx error round and transform it once.
    ///
    /// Raw chunks are appended to a per-request [`StreamingErrorBuffer`] and
    /// suppressed from passthrough. At end of stream the accumulated body is
    /// normalized to a single Anthropic error, so exactly one JSON object reaches
    /// the client even when the composed pipeline delivers the error round as raw
    /// `Stream` chunks rather than a single buffered body.
    ///
    /// An accumulated body that exceeds `max_body_bytes` fails closed with a single
    /// JSON error body rather than being truncated. This round is pre-SSE: its
    /// non-2xx status and `application/json` content-type are already committed, so
    /// a `Reject` here would surface as a stream termination rendered as an
    /// `event: error` SSE frame under those JSON headers — a body the client cannot
    /// parse. Emitting one valid JSON error instead keeps the response parseable and
    /// still refuses to forward an oversized upstream error.
    fn accumulate_streaming_error_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> FilterAction {
        if let Some(chunk) = body.take() {
            let buffer = ctx.extensions.get_or_insert_with(StreamingErrorBuffer::default);
            if !buffer.overflowed {
                buffer.buf.extend_from_slice(&chunk);
                if buffer.buf.len() > self.config.max_body_bytes {
                    // Free the accumulated bytes and defer the JSON error to end of
                    // stream; further chunks stay suppressed via `body.take()`.
                    buffer.overflowed = true;
                    buffer.buf = Vec::new();
                }
            }
        }

        if !end_of_stream {
            return FilterAction::Continue;
        }

        let request_id = ctx.get_metadata(RESPONSE_REQUEST_ID_KEY).map(str::to_owned);
        let buffer = ctx.extensions.remove::<StreamingErrorBuffer>().unwrap_or_default();
        if buffer.overflowed {
            *body = Some(Bytes::from(wire::error_body(
                wire::ErrorType::Api,
                "upstream response exceeded the configured max_body_bytes",
                request_id.as_deref(),
            )));
            return FilterAction::Continue;
        }

        let status = error_status(ctx);
        *body = Some(Bytes::from(response::transform_error_response(
            &buffer.buf,
            status,
            request_id.as_deref(),
        )));
        FilterAction::Continue
    }
}

#[async_trait]
impl HttpFilter for AnthropicMessagesToChatCompletionsFilter {
    fn name(&self) -> &'static str {
        "anthropic_messages_to_chat_completions"
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

    async fn on_response(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        let request_id = canonicalize_response_request_id(ctx);
        // Advertise degradation on every forwarded response (streaming or not,
        // success or error), independent of whether the body is transformed.
        apply_degraded_features_header(ctx);
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

    async fn on_selected_upstream_request_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
    ) -> Result<SelectedUpstreamBodyOutcome, FilterError> {
        // Translate only after upstream selection so a protocol-gated pipeline can
        // keep the native Anthropic body for a Messages backend.
        let bytes = match body.as_ref() {
            Some(b) if !b.is_empty() => b.as_ref(),
            _ => return Ok(SelectedUpstreamBodyOutcome::Continue),
        };

        let allow = self.config.allowlist();
        let transformed = match serde_json::from_slice::<serde_json::Value>(bytes) {
            Ok(value) => {
                extract_request_metadata(ctx, Some(&value));
                match request::transform_request_degrading(value, allow) {
                    Ok(output) => {
                        record_degraded_features(ctx, output.degraded);
                        Ok(output.body)
                    },
                    Err(msg) => Err(msg),
                }
            },
            Err(error) => {
                extract_request_metadata(ctx, None);
                Err(format!("invalid JSON: {error}"))
            },
        };

        Ok(transform_request_body(body, transformed))
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

        // Accumulate manually only when the error round is physically delivered as
        // raw `Stream` chunks. That happens under a managed streaming IRR (for
        // example the `anthropic_web_search` loop), which selects
        // `SubRequestResponseMode::Streaming` and ignores this filter's per-request
        // `StreamBuffer` ratchet. Gating on the client's `stream` flag alone would
        // be wrong: a standalone streaming request keeps the ratchet, so the
        // framework already buffers the round and re-presents the complete body at
        // end of stream. Accumulating in that case appends the framework's frozen
        // full body on top of the mid-stream chunks — a doubled, unparseable body
        // that degrades to a generic fallback error. There the buffered path below
        // transforms the framework's body exactly once.
        if transform_error && ctx.subrequest_response_mode() == SubRequestResponseMode::Streaming {
            return Ok(self.accumulate_streaming_error_body(ctx, body, end_of_stream));
        }

        if !end_of_stream {
            return Ok(FilterAction::Continue);
        }

        // Record the raw upstream byte count before any transform shrinks it, so a
        // later filter (the managed web-search loop, which runs after this
        // translation on the response path) can enforce its own byte ceiling on the
        // untransformed size rather than on the smaller body it observes.
        let raw_len = body.as_ref().map_or(0, Bytes::len);
        ctx.set_metadata(RESPONSE_RAW_BYTES_KEY, raw_len.to_string());

        // Enforce the response byte ceiling on the RAW upstream body before any
        // transform can shrink it below a downstream size check. A
        // streaming-composed pipeline (the managed web-search loop) drops the
        // executor's per-filter `StreamBuffer` response cap, so an oversized
        // buffered round would otherwise be normalized and forwarded instead of
        // rejected. Mirrors the web_search buffered ceiling with a 502 api_error.
        if raw_len > self.config.max_body_bytes {
            return Ok(FilterAction::Reject(wire::error_rejection(
                502,
                wire::ErrorType::Api,
                "upstream response exceeded the configured max_body_bytes",
            )));
        }

        Ok(transform_buffered_response(ctx, body, transform_error))
    }
}

/// Transform a fully buffered response body in place: normalize an error round
/// into a single Anthropic error, or rewrite a successful Chat Completions body
/// into the Messages shape and record the mapped finish reason.
fn transform_buffered_response(
    ctx: &mut HttpFilterContext<'_>,
    body: &mut Option<Bytes>,
    transform_error: bool,
) -> FilterAction {
    if transform_error {
        let status = error_status(ctx);
        let request_id = ctx.get_metadata(RESPONSE_REQUEST_ID_KEY);
        transform_error_body(body, status, request_id);
        FilterAction::Continue
    } else {
        let request_model = ctx
            .filter_metadata
            .get("anthropic_messages_to_chat_completions.model")
            .map_or("", String::as_str);
        let request_id = ctx.get_metadata(RESPONSE_REQUEST_ID_KEY);
        let stop_sequences = client_stop_sequences(ctx);
        if let Some(finish_reason) = transform_non_streaming_body(body, request_model, request_id, &stop_sequences) {
            ctx.set_metadata("openai.finish_reason", finish_reason);
            FilterAction::Continue
        } else {
            // Reject the invalid success. If the upstream 200 headers have
            // already been sent, the proxy aborts the response body instead
            // of sending a new HTTP 500 error.
            FilterAction::Reject(Rejection::status(500))
        }
    }
}

// -----------------------------------------------------------------------------
// Request Body Helpers
// -----------------------------------------------------------------------------

/// Extract streaming and model metadata from the parsed request body.
fn extract_request_metadata(ctx: &mut HttpFilterContext<'_>, value: Option<&serde_json::Value>) {
    let Some(value) = value else {
        ctx.set_metadata("anthropic_messages_to_chat_completions.streaming", "false");
        return;
    };

    let is_streaming = value
        .get("stream")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    ctx.set_metadata(
        "anthropic_messages_to_chat_completions.streaming",
        if is_streaming { "true" } else { "false" },
    );

    let model = value
        .get("model")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .unwrap_or_default();
    ctx.set_metadata("anthropic_messages_to_chat_completions.model", model);

    if let Some(stop_sequences) = value
        .get("stop_sequences")
        .filter(|v| v.as_array().is_some_and(|a| !a.is_empty()))
    {
        ctx.set_metadata(STOP_SEQUENCES_KEY, stop_sequences.to_string());
    }
}

/// Client `stop_sequences` retained from the request, or empty when none were sent.
///
/// A list that is not all strings is invalid Anthropic input the backend
/// rejects on its own; treating it as empty keeps the response at
/// `end_turn` rather than failing the translation.
pub(crate) fn client_stop_sequences(ctx: &HttpFilterContext<'_>) -> Vec<String> {
    ctx.filter_metadata
        .get(STOP_SEQUENCES_KEY)
        .and_then(|value| serde_json::from_str(value).ok())
        .unwrap_or_default()
}

/// Record degraded features as operator signals.
///
/// Emits a per-feature counter and one structured warning (fixed labels, no
/// prompt data), and stashes the comma-separated labels in metadata so the
/// response-header phase can advertise them. A no-op when nothing was degraded.
fn record_degraded_features(ctx: &mut HttpFilterContext<'_>, degraded: request::DegradedFeatures) {
    if !degraded.any() {
        return;
    }
    let mut features: Vec<&'static str> = Vec::new();
    if degraded.prompt_caching {
        features.push("prompt_caching");
    }
    if degraded.extended_thinking {
        features.push("extended_thinking");
    }
    for feature in &features {
        counter!(
            "praxis_anthropic_messages_to_chat_completions_degraded_total",
            "feature" => *feature
        )
        .increment(1);
    }
    let joined = features.join(",");
    warn!(
        degraded_features = joined.as_str(),
        "degraded unsupported Anthropic features to complete the Chat Completions translation"
    );
    ctx.set_metadata(DEGRADED_FEATURES_KEY, joined);
}

/// Reconcile the degraded-feature response header with the proxy's own record.
///
/// Always clears any backend-supplied `x-degraded-features` first — even when no
/// degradation occurred — so a forged upstream copy can never reach the client as
/// a false Praxis signal, then re-inserts the header only when the proxy itself
/// recorded a degradation.
fn apply_degraded_features_header(ctx: &mut HttpFilterContext<'_>) {
    let recorded = ctx.get_metadata(DEGRADED_FEATURES_KEY).map(str::to_owned);
    let Some(response) = &mut ctx.response_header else {
        return;
    };
    let mut modified = response.headers.remove(DEGRADED_FEATURES_HEADER).is_some();
    if let Some(value) = recorded
        && let Ok(header_value) = http::HeaderValue::from_str(&value)
    {
        response
            .headers
            .insert(http::HeaderName::from_static(DEGRADED_FEATURES_HEADER), header_value);
        modified = true;
    }
    if modified {
        ctx.response_headers_modified = true;
    }
}

/// Install a translated request body, or reject when translation failed.
fn transform_request_body(
    body: &mut Option<Bytes>,
    transformed: Result<Vec<u8>, String>,
) -> SelectedUpstreamBodyOutcome {
    let Some(bytes) = body.as_ref() else {
        return SelectedUpstreamBodyOutcome::Continue;
    };

    match transformed {
        Ok(transformed) => {
            debug!(
                original_len = bytes.len(),
                transformed_len = transformed.len(),
                "transformed Anthropic request to Chat Completions-compatible format"
            );
            *body = Some(Bytes::from(transformed));
            SelectedUpstreamBodyOutcome::Continue
        },
        Err(msg) => {
            warn!(error = msg.as_str(), "failed to transform Anthropic request");
            SelectedUpstreamBodyOutcome::Reject(wire::invalid_request_rejection(&msg))
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

/// Recover the upstream error status recorded in the response-header phase.
fn error_status(ctx: &HttpFilterContext<'_>) -> http::StatusCode {
    ctx.get_metadata(RESPONSE_STATUS_KEY)
        .and_then(|value| value.parse::<u16>().ok())
        .and_then(|value| http::StatusCode::from_u16(value).ok())
        .unwrap_or(http::StatusCode::INTERNAL_SERVER_ERROR)
}

/// Select the response transformation while headers are available.
fn response_transform(ctx: &HttpFilterContext<'_>) -> Option<&'static str> {
    let is_streaming = ctx
        .filter_metadata
        .get("anthropic_messages_to_chat_completions.streaming")
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
                            || media_type
                                .get(media_type.len().saturating_sub("+json".len())..)
                                .is_some_and(|suffix| suffix.eq_ignore_ascii_case("+json"))
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
    stop_sequences: &[String],
) -> Option<String> {
    match response::transform_response(body.as_deref().unwrap_or_default(), request_model, stop_sequences) {
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
                wire::ErrorType::Api,
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
    use std::sync::{Arc, Mutex};

    use bytes::Bytes;
    use http::{Method, StatusCode};

    use super::*;
    use crate::test_utils::{make_filter_context, make_request, make_response};

    #[test]
    fn default_config_parses() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("{}").unwrap();
        let filter = AnthropicMessagesToChatCompletionsFilter::from_config(&yaml).unwrap();

        assert_eq!(
            filter.name(),
            "anthropic_messages_to_chat_completions",
            "filter name should match"
        );
    }

    #[test]
    fn unknown_config_field_rejected() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("strip_unsupported: true").unwrap();
        let result = AnthropicMessagesToChatCompletionsFilter::from_config(&yaml);

        assert!(result.is_err(), "unknown config fields should be rejected");
    }

    #[test]
    fn zero_max_body_bytes_rejected() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("max_body_bytes: 0").unwrap();
        let result = AnthropicMessagesToChatCompletionsFilter::from_config(&yaml);

        assert!(result.is_err(), "zero max_body_bytes should be rejected");
    }

    #[test]
    fn rejects_max_body_bytes_above_ceiling() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("max_body_bytes: 67108865").unwrap();
        let result = AnthropicMessagesToChatCompletionsFilter::from_config(&yaml);

        assert!(
            result.is_err(),
            "max_body_bytes above 64 MiB ceiling should be rejected"
        );
    }

    #[tokio::test]
    async fn error_response_state_survives_body_phase_without_headers() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("{}").unwrap();
        let filter = AnthropicMessagesToChatCompletionsFilter::from_config(&yaml).unwrap();
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);
        let mut response = make_response();
        response.status = StatusCode::SERVICE_UNAVAILABLE;
        response.headers.insert("x-request-id", "req_header".parse().unwrap());
        response
            .headers
            .insert(http::header::CONTENT_LENGTH, http::HeaderValue::from_static("72"));
        ctx.response_header = Some(&mut response);
        ctx.set_metadata("anthropic_messages_to_chat_completions.streaming", "true");
        ctx.set_metadata("anthropic_messages_to_chat_completions.model", "gpt-4");
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
    async fn streaming_error_round_accumulates_chunks_into_single_error() {
        // A streaming-composed pipeline (for example the managed web-search IRR
        // loop) ignores the per-request `StreamBuffer` ratchet and delivers a
        // non-2xx round as raw `Stream` chunks. The filter must suppress those raw
        // chunks and emit exactly one transformed Anthropic error at end of stream,
        // never the raw body followed by a second empty-input transform.
        let yaml: serde_yaml::Value = serde_yaml::from_str("{}").unwrap();
        let filter = AnthropicMessagesToChatCompletionsFilter::from_config(&yaml).unwrap();
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);
        let mut response = make_response();
        response.status = StatusCode::TOO_MANY_REQUESTS;
        ctx.response_header = Some(&mut response);
        ctx.set_metadata("anthropic_messages_to_chat_completions.streaming", "true");
        // The managed IRR selected the streaming subrequest transport, so the error
        // round arrives as raw `Stream` chunks that this filter must accumulate.
        ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
        drop(filter.on_response(&mut ctx).await.unwrap());
        // The IRR clears the response header before the streaming body phase.
        ctx.response_header = None;

        // The raw upstream error arrives split across two non-terminal chunks.
        let raw = br#"{"error":{"message":"rate limited","type":"rate_limit_error"}}"#;
        let (head, tail) = raw.split_at(20);
        let mut first = Some(Bytes::copy_from_slice(head));
        drop(filter.on_response_body(&mut ctx, &mut first, false).unwrap());
        assert!(
            first.is_none(),
            "intermediate error chunks are suppressed, not forwarded"
        );
        let mut second = Some(Bytes::copy_from_slice(tail));
        drop(filter.on_response_body(&mut ctx, &mut second, false).unwrap());
        assert!(
            second.is_none(),
            "intermediate error chunks are suppressed, not forwarded"
        );

        // The terminal chunk flushes exactly one transformed error.
        let mut terminal = Some(Bytes::new());
        drop(filter.on_response_body(&mut ctx, &mut terminal, true).unwrap());
        let bytes = terminal.unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(parsed["type"], "error");
        assert_eq!(parsed["error"]["type"], "rate_limit_error");
        assert_eq!(
            parsed["error"]["message"], "rate limited",
            "the accumulated raw body is transformed, not an empty fallback"
        );
    }

    #[tokio::test]
    async fn standalone_streaming_error_streambuffer_delivery_yields_single_error() {
        // A standalone streaming request (no managed IRR) keeps the response body
        // in `StreamBuffer` mode via the `on_response` ratchet, so the framework
        // presents each raw chunk mid-stream AND re-presents the frozen full body
        // at end of stream. `subrequest_response_mode` stays `Buffered` (the
        // default), so this filter must NOT accumulate the chunks itself: doing so
        // would append the body twice and corrupt the JSON. It transforms the
        // framework's buffered body exactly once.
        let yaml: serde_yaml::Value = serde_yaml::from_str("{}").unwrap();
        let filter = AnthropicMessagesToChatCompletionsFilter::from_config(&yaml).unwrap();
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);
        assert_eq!(
            ctx.subrequest_response_mode(),
            SubRequestResponseMode::Buffered,
            "a standalone request has no managed IRR, so the transport is buffered"
        );
        let mut response = make_response();
        response.status = StatusCode::TOO_MANY_REQUESTS;
        ctx.response_header = Some(&mut response);
        ctx.set_metadata("anthropic_messages_to_chat_completions.streaming", "true");
        drop(filter.on_response(&mut ctx).await.unwrap());
        ctx.response_header = None;

        // `StreamBuffer` delivery: the raw upstream error is presented split across
        // two mid-stream chunks, then the complete frozen buffer is re-presented at
        // end of stream — exactly as praxis `StreamBuffer` mode drives the hook.
        let raw = br#"{"error":{"message":"rate limited: retry in 60 seconds","type":"rate_limit_error"}}"#;
        let (head, tail) = raw.split_at(30);
        let mut first = Some(Bytes::copy_from_slice(head));
        drop(filter.on_response_body(&mut ctx, &mut first, false).unwrap());
        let mut second = Some(Bytes::copy_from_slice(tail));
        drop(filter.on_response_body(&mut ctx, &mut second, false).unwrap());
        let mut terminal = Some(Bytes::copy_from_slice(raw));
        drop(filter.on_response_body(&mut ctx, &mut terminal, true).unwrap());

        let bytes = terminal.unwrap();
        // A clean parse proves the body was not appended twice: a doubled body is
        // invalid JSON and falls back to a generic "upstream request failed".
        let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(parsed["type"], "error");
        assert_eq!(parsed["error"]["type"], "rate_limit_error");
        assert_eq!(
            parsed["error"]["message"], "rate limited: retry in 60 seconds",
            "the real upstream message survives; the body is transformed once, not doubled"
        );
    }

    #[tokio::test]
    async fn buffered_error_over_max_body_bytes_is_rejected() {
        // The raw upstream body must be measured before any transform can shrink it
        // below the ceiling. A streaming-composed pipeline (the managed web-search
        // loop) drops the executor's per-filter response cap, so the translator has
        // to enforce the limit itself and reject an oversized round with a 502.
        let yaml: serde_yaml::Value = serde_yaml::from_str("max_body_bytes: 64").unwrap();
        let filter = AnthropicMessagesToChatCompletionsFilter::from_config(&yaml).unwrap();
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);
        let mut response = make_response();
        response.status = StatusCode::TOO_MANY_REQUESTS;
        ctx.response_header = Some(&mut response);
        ctx.set_metadata("anthropic_messages_to_chat_completions.streaming", "false");
        drop(filter.on_response(&mut ctx).await.unwrap());
        ctx.response_header = None;

        // An oversized upstream error body whose transformed envelope would be small.
        let padding = "x".repeat(4096);
        let raw = format!(r#"{{"error":{{"message":"{padding}","type":"rate_limit_error"}}}}"#);
        let mut body = Some(Bytes::from(raw));
        let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();

        let FilterAction::Reject(rejection) = action else {
            panic!("an oversized buffered error must be rejected, not normalized");
        };
        assert_eq!(rejection.status, 502, "the raw-size ceiling rejects with a 502");
        let parsed: serde_json::Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();
        assert_eq!(parsed["type"], "error");
        assert_eq!(parsed["error"]["type"], "api_error");
    }

    #[tokio::test]
    async fn streaming_error_over_max_body_bytes_fails_closed_with_json() {
        // The streaming accumulator must fail closed on an oversized error round
        // without truncating it. This round is pre-SSE: the non-2xx status and
        // JSON content-type are already committed, so a `Reject` here would become
        // an `event: error` SSE frame under JSON headers that the client cannot
        // parse. Instead it emits exactly one valid JSON error at end of stream.
        let yaml: serde_yaml::Value = serde_yaml::from_str("max_body_bytes: 64").unwrap();
        let filter = AnthropicMessagesToChatCompletionsFilter::from_config(&yaml).unwrap();
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);
        let mut response = make_response();
        response.status = StatusCode::TOO_MANY_REQUESTS;
        response.headers.insert("x-request-id", "req_over".parse().unwrap());
        ctx.response_header = Some(&mut response);
        ctx.set_metadata("anthropic_messages_to_chat_completions.streaming", "true");
        // The managed IRR selected the streaming subrequest transport, so the error
        // round arrives as raw `Stream` chunks that this filter must accumulate.
        ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
        drop(filter.on_response(&mut ctx).await.unwrap());
        ctx.response_header = None;

        // A single oversized chunk arrives before end of stream and is suppressed.
        let mut chunk = Some(Bytes::from("x".repeat(4096)));
        let action = filter.on_response_body(&mut ctx, &mut chunk, false).unwrap();
        assert!(
            matches!(action, FilterAction::Continue),
            "an oversized streaming chunk is suppressed, not rejected into an SSE frame"
        );
        assert!(chunk.is_none(), "the oversized chunk must not be forwarded");

        // The terminal chunk flushes exactly one valid JSON error, not SSE.
        let mut terminal = Some(Bytes::new());
        let action = filter.on_response_body(&mut ctx, &mut terminal, true).unwrap();
        assert!(matches!(action, FilterAction::Continue), "end of stream continues");
        let bytes = terminal.unwrap();
        // `from_slice` on the whole body proves it is one JSON object, never an SSE
        // event frame (which would fail to parse as JSON).
        let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(parsed["type"], "error");
        assert_eq!(parsed["error"]["type"], "api_error");
        assert!(
            parsed["error"]["message"]
                .as_str()
                .is_some_and(|message| message.contains("max_body_bytes")),
            "the JSON error names the exceeded ceiling"
        );
        assert_eq!(parsed["request_id"], "req_over", "the upstream request id is preserved");
    }

    #[tokio::test]
    async fn rewritten_errors_remove_stale_representation_headers() {
        for content_encoding in ["gzip", "br"] {
            let yaml: serde_yaml::Value = serde_yaml::from_str("max_body_bytes: 4096").unwrap();
            let filter = AnthropicMessagesToChatCompletionsFilter::from_config(&yaml).unwrap();
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

    /// Parse a raw body the way `on_selected_upstream_request_body` does, for the
    /// metadata pass.
    fn parse(body: &[u8]) -> Option<serde_json::Value> {
        serde_json::from_slice(body).ok()
    }

    /// Translate a raw body the way `on_selected_upstream_request_body` does, parse
    /// errors included.
    fn translate(body: &[u8]) -> Result<Vec<u8>, String> {
        let value: serde_json::Value =
            serde_json::from_slice(body).map_err(|error| format!("invalid JSON: {error}"))?;
        request::transform_request(value)
    }

    #[test]
    fn extract_request_metadata_streaming_true_with_model() {
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);
        let value = parse(br#"{"stream":true,"model":"claude-opus-4-8"}"#);

        extract_request_metadata(&mut ctx, value.as_ref());

        assert_eq!(
            ctx.filter_metadata
                .get("anthropic_messages_to_chat_completions.streaming")
                .unwrap(),
            "true"
        );
        assert_eq!(
            ctx.filter_metadata
                .get("anthropic_messages_to_chat_completions.model")
                .unwrap(),
            "claude-opus-4-8"
        );
    }

    #[test]
    fn extract_request_metadata_streaming_false() {
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);
        let value = parse(br#"{"stream":false,"model":"gpt-4"}"#);

        extract_request_metadata(&mut ctx, value.as_ref());

        assert_eq!(
            ctx.filter_metadata
                .get("anthropic_messages_to_chat_completions.streaming")
                .unwrap(),
            "false"
        );
        assert_eq!(
            ctx.filter_metadata
                .get("anthropic_messages_to_chat_completions.model")
                .unwrap(),
            "gpt-4"
        );
    }

    #[test]
    fn extract_request_metadata_retains_stop_sequences() {
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);
        let value = parse(br#"{"model":"gpt-4","stop_sequences":[",","END"]}"#);

        extract_request_metadata(&mut ctx, value.as_ref());

        assert_eq!(
            ctx.filter_metadata.get(STOP_SEQUENCES_KEY).unwrap(),
            r#"[",","END"]"#,
            "client stop sequences are retained for the response phase"
        );
    }

    #[test]
    fn extract_request_metadata_without_stop_sequences_sets_nothing() {
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);
        let value = parse(br#"{"model":"gpt-4","stop_sequences":[]}"#);

        extract_request_metadata(&mut ctx, value.as_ref());

        assert!(
            !ctx.filter_metadata.contains_key(STOP_SEQUENCES_KEY),
            "empty stop_sequences should not be retained"
        );
    }

    #[test]
    fn client_stop_sequences_ignores_non_string_entries() {
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);
        ctx.set_metadata(STOP_SEQUENCES_KEY, r#"[1,","]"#);

        assert!(
            client_stop_sequences(&ctx).is_empty(),
            "non-string stop_sequences should yield no client sequences"
        );
    }

    #[test]
    fn extract_request_metadata_invalid_json() {
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);

        extract_request_metadata(&mut ctx, parse(b"not json").as_ref());

        assert_eq!(
            ctx.filter_metadata
                .get("anthropic_messages_to_chat_completions.streaming")
                .unwrap(),
            "false",
            "invalid JSON should default streaming to false"
        );
        assert!(
            !ctx.filter_metadata
                .contains_key("anthropic_messages_to_chat_completions.model"),
            "invalid JSON should not set model"
        );
    }

    // --- transform_request_body ---

    #[test]
    fn transform_request_body_none_continues() {
        let mut body: Option<Bytes> = None;
        let action = transform_request_body(&mut body, translate(br#"{"model":"claude-opus-4-8"}"#));

        assert!(matches!(action, SelectedUpstreamBodyOutcome::Continue));
        assert!(body.is_none());
    }

    #[tokio::test]
    async fn on_request_prevents_upstream_response_encoding() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("{}").unwrap();
        let filter = AnthropicMessagesToChatCompletionsFilter::from_config(&yaml).unwrap();
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
        let raw = br#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"role":"user","content":"Hi"}]}"#;
        let mut body = Some(Bytes::from(raw.to_vec()));
        let action = transform_request_body(&mut body, translate(raw));

        assert!(matches!(action, SelectedUpstreamBodyOutcome::Continue));
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
        let action = transform_request_body(&mut body, translate(b"not json"));

        let SelectedUpstreamBodyOutcome::Reject(rejection) = action else {
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
        ctx.set_metadata("anthropic_messages_to_chat_completions.streaming", "true");
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
        ctx.set_metadata("anthropic_messages_to_chat_completions.streaming", "false");
        let mut response = make_response();
        ctx.response_header = Some(&mut response);

        assert!(
            should_transform_response(&ctx),
            "non-streaming success should be transformed"
        );
    }

    #[test]
    fn should_transform_json_media_types_without_changing_parameter_handling() {
        for (content_type, expected) in [
            ("application/json; charset=utf-8", true),
            ("Application/Problem+JsOn; charset=utf-8", true),
            ("application/vnd.example+JSON", true),
            ("application/problem+json-seq", false),
            ("text/plain; charset=utf-8", false),
        ] {
            let request = make_request(Method::POST, "/v1/messages");
            let mut ctx = make_filter_context(&request);
            ctx.set_metadata("anthropic_messages_to_chat_completions.streaming", "false");
            let mut response = make_response();
            response
                .headers
                .insert(http::header::CONTENT_TYPE, content_type.parse().unwrap());
            ctx.response_header = Some(&mut response);

            assert_eq!(should_transform_response(&ctx), expected, "{content_type}");
        }
    }

    #[test]
    fn should_not_transform_encoded_non_streaming_success() {
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);
        ctx.set_metadata("anthropic_messages_to_chat_completions.streaming", "false");
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
        let filter = AnthropicMessagesToChatCompletionsFilter::from_config(&yaml).unwrap();
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);
        ctx.set_metadata("anthropic_messages_to_chat_completions.streaming", "false");
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
        let filter = AnthropicMessagesToChatCompletionsFilter::from_config(&yaml).unwrap();
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);
        ctx.set_metadata("anthropic_messages_to_chat_completions.streaming", "false");
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
            let filter = AnthropicMessagesToChatCompletionsFilter::from_config(&yaml).unwrap();
            let request = make_request(Method::POST, "/v1/messages");
            let mut ctx = make_filter_context(&request);
            ctx.set_metadata("anthropic_messages_to_chat_completions.streaming", is_streaming);
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
        ctx.set_metadata("anthropic_messages_to_chat_completions.streaming", "false");
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
                ctx.set_metadata("anthropic_messages_to_chat_completions.streaming", is_streaming);
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
        ctx.set_metadata("anthropic_messages_to_chat_completions.streaming", "false");
        let mut response = make_response();
        response.status = StatusCode::FOUND;
        ctx.response_header = Some(&mut response);

        assert!(!should_transform_response(&ctx), "redirect should pass through");
    }

    // --- transform_non_streaming_body ---

    #[test]
    fn transform_non_streaming_body_missing_body_returns_api_error() {
        let mut body: Option<Bytes> = None;

        let finish_reason = transform_non_streaming_body(&mut body, "gpt-4", None, &[]);
        let parsed: serde_json::Value = serde_json::from_slice(body.as_deref().unwrap()).unwrap();

        assert!(finish_reason.is_none());
        assert_eq!(parsed["type"], "error");
        assert_eq!(parsed["error"]["type"], "api_error");
        assert!(parsed["request_id"].is_null());
    }

    #[test]
    fn transform_non_streaming_body_empty_bytes_returns_api_error() {
        let mut body = Some(Bytes::new());

        let finish_reason = transform_non_streaming_body(&mut body, "gpt-4", None, &[]);
        let parsed: serde_json::Value = serde_json::from_slice(body.as_deref().unwrap()).unwrap();

        assert!(finish_reason.is_none());
        assert_eq!(parsed["type"], "error");
        assert_eq!(parsed["error"]["type"], "api_error");
    }

    #[test]
    fn transform_non_streaming_body_success() {
        let response_json = br#"{"id":"chatcmpl-1","model":"gpt-4","choices":[{"message":{"role":"assistant","content":"Hello!"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5}}"#;
        let mut body = Some(Bytes::from(response_json.to_vec()));

        let finish_reason = transform_non_streaming_body(&mut body, "gpt-4", None, &[]);

        assert!(body.is_some());
        let parsed: serde_json::Value = serde_json::from_slice(body.unwrap().as_ref()).unwrap();
        assert_eq!(parsed["type"], "message");
        assert_eq!(parsed["content"][0]["text"], "Hello!");
        assert_eq!(finish_reason.as_deref(), Some("stop"));
    }

    #[tokio::test]
    async fn malformed_non_streaming_success_returns_anthropic_api_error() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("{}").unwrap();
        let filter = AnthropicMessagesToChatCompletionsFilter::from_config(&yaml).unwrap();
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);
        ctx.set_metadata("anthropic_messages_to_chat_completions.streaming", "false");
        ctx.set_metadata("anthropic_messages_to_chat_completions.model", "gpt-4");
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
        let FilterAction::Reject(rejection) = action else {
            panic!("malformed upstream success must reject with an HTTP error");
        };
        let parsed: serde_json::Value = serde_json::from_slice(body.as_deref().unwrap()).unwrap();

        assert_eq!(rejection.status, 500);
        assert_eq!(parsed["type"], "error");
        assert_eq!(parsed["error"]["type"], "api_error");
        assert_eq!(parsed["error"]["message"], "upstream response could not be transformed");
        assert_eq!(parsed["request_id"], "req_malformed");
        assert!(!ctx.filter_metadata.contains_key("openai.finish_reason"));
    }

    /// An allowlisted request that uses both degradable features drives the full
    /// operator-signal path: the translated body drops the wire markers, a
    /// per-feature counter is incremented, and the response header advertises
    /// the degradation on the forwarded response.
    #[test]
    fn degradation_emits_counter_metadata_and_response_header() {
        let recorder = metrics_util::debugging::DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        let yaml: serde_yaml::Value =
            serde_yaml::from_str("allow_lossy_features:\n  - prompt_caching\n  - extended_thinking").unwrap();
        let filter = AnthropicMessagesToChatCompletionsFilter::from_config(&yaml).unwrap();

        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);
        let anthropic_body = serde_json::json!({
            "model": "claude-x",
            "max_tokens": 2048,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "system": [
                {"type": "text", "text": "Be brief", "cache_control": {"type": "ephemeral"}}
            ],
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "Hi", "cache_control": {"type": "ephemeral"}}
                ]}
            ]
        });
        let mut body = Some(Bytes::from(serde_json::to_vec(&anthropic_body).unwrap()));
        let mut response = make_response();

        metrics::with_local_recorder(&recorder, || {
            runtime.block_on(async {
                let outcome = filter
                    .on_selected_upstream_request_body(&mut ctx, &mut body)
                    .await
                    .unwrap();
                assert!(
                    matches!(outcome, SelectedUpstreamBodyOutcome::Continue),
                    "a degraded-but-translatable request continues with a rewritten body"
                );

                // The forwarded Chat Completions body carries neither wire marker.
                let translated: serde_json::Value = serde_json::from_slice(body.as_deref().unwrap()).unwrap();
                assert!(translated.get("thinking").is_none(), "thinking must be stripped");
                let as_text = serde_json::to_string(&translated).unwrap();
                assert!(!as_text.contains("cache_control"), "cache_control must be stripped");
                assert!(
                    translated.get("messages").is_some(),
                    "translation must still produce messages"
                );

                // Both features are recorded for the response-header phase.
                assert_eq!(
                    ctx.get_metadata(DEGRADED_FEATURES_KEY),
                    Some("prompt_caching,extended_thinking"),
                    "both degraded features should be recorded in metadata"
                );

                ctx.response_header = Some(&mut response);
                let action = filter.on_response(&mut ctx).await.unwrap();
                assert!(
                    matches!(action, FilterAction::Continue),
                    "response phase should continue"
                );
            });
        });

        assert_eq!(
            response
                .headers
                .get(DEGRADED_FEATURES_HEADER)
                .and_then(|value| value.to_str().ok()),
            Some("prompt_caching,extended_thinking"),
            "the forwarded response should advertise the degraded features"
        );

        let snapshot: Vec<_> = snapshotter
            .snapshot()
            .into_vec()
            .into_iter()
            .map(|(key, _, _, value)| (key, value))
            .collect();
        let counter_for = |feature: &str| {
            snapshot
                .iter()
                .find(|(key, _)| {
                    key.key().name() == "praxis_anthropic_messages_to_chat_completions_degraded_total"
                        && key
                            .key()
                            .labels()
                            .any(|label| label.key() == "feature" && label.value() == feature)
                })
                .and_then(|(_, value)| match value {
                    metrics_util::debugging::DebugValue::Counter(count) => Some(*count),
                    _ => None,
                })
        };
        assert_eq!(
            counter_for("prompt_caching"),
            Some(1),
            "prompt_caching counter must fire once"
        );
        assert_eq!(
            counter_for("extended_thinking"),
            Some(1),
            "extended_thinking counter must fire once"
        );
    }

    /// Without an allowlist the filter keeps the strict #1584 behavior: a request
    /// that uses an Anthropic-only feature is rejected before any backend call,
    /// and no degradation signal is produced.
    #[test]
    fn strict_mode_rejects_degradable_feature_without_signal() {
        let recorder = metrics_util::debugging::DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        let yaml: serde_yaml::Value = serde_yaml::from_str("{}").unwrap();
        let filter = AnthropicMessagesToChatCompletionsFilter::from_config(&yaml).unwrap();

        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);
        let anthropic_body = serde_json::json!({
            "model": "claude-x",
            "max_tokens": 2048,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "messages": [{"role": "user", "content": "Hi"}]
        });
        let mut body = Some(Bytes::from(serde_json::to_vec(&anthropic_body).unwrap()));

        metrics::with_local_recorder(&recorder, || {
            runtime.block_on(async {
                let outcome = filter
                    .on_selected_upstream_request_body(&mut ctx, &mut body)
                    .await
                    .unwrap();
                assert!(
                    matches!(outcome, SelectedUpstreamBodyOutcome::Reject(_)),
                    "strict mode must reject a request that uses a degradable feature"
                );
            });
        });

        assert_eq!(
            ctx.get_metadata(DEGRADED_FEATURES_KEY),
            None,
            "strict mode records no degradation"
        );
        assert!(
            snapshotter.snapshot().into_vec().is_empty(),
            "strict mode emits no degradation counter"
        );
    }

    /// A backend-supplied `x-degraded-features` header must never reach the client
    /// as a false Praxis signal: when the proxy recorded no degradation, the
    /// response phase strips the forged copy rather than forwarding it.
    #[test]
    fn response_strips_backend_forged_degraded_header_without_degradation() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        let yaml: serde_yaml::Value = serde_yaml::from_str("{}").unwrap();
        let filter = AnthropicMessagesToChatCompletionsFilter::from_config(&yaml).unwrap();

        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);
        let mut response = make_response();
        // A non-JSON success body avoids the response-transform path; this test
        // only exercises the header reconciliation.
        response
            .headers
            .insert(http::header::CONTENT_TYPE, "text/plain".parse().unwrap());
        response.headers.insert(
            http::HeaderName::from_static(DEGRADED_FEATURES_HEADER),
            "prompt_caching,extended_thinking".parse().unwrap(),
        );
        ctx.response_header = Some(&mut response);

        runtime.block_on(async {
            let action = filter.on_response(&mut ctx).await.unwrap();
            assert!(
                matches!(action, FilterAction::Continue),
                "response phase should continue"
            );
        });

        assert_eq!(
            ctx.get_metadata(DEGRADED_FEATURES_KEY),
            None,
            "no degradation is recorded for a clean request"
        );
        assert!(
            !response.headers.contains_key(DEGRADED_FEATURES_HEADER),
            "the forged backend header must be stripped so it cannot pose as a Praxis signal"
        );
    }

    #[test]
    fn degradation_emits_operator_warn_and_metadata() {
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);
        let degraded = request::DegradedFeatures {
            prompt_caching: true,
            extended_thinking: true,
        };

        let events = capture_warn_events(|| record_degraded_features(&mut ctx, degraded));

        assert_eq!(
            events.len(),
            1,
            "degradation must emit exactly one WARN; got {events:?}"
        );
        let (message, degraded_features) = &events[0];
        assert!(
            message.contains("degraded unsupported Anthropic features to complete the Chat Completions translation"),
            "WARN must carry the documented operator message; got {message:?}"
        );
        assert_eq!(
            degraded_features.as_deref(),
            Some("prompt_caching,extended_thinking"),
            "WARN must name every degraded feature as a structured field"
        );
        assert_eq!(
            ctx.get_metadata(DEGRADED_FEATURES_KEY),
            Some("prompt_caching,extended_thinking"),
            "degraded features must be stashed for the response-header phase"
        );
    }

    #[test]
    fn no_degradation_emits_no_warn_and_no_metadata() {
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);

        let events = capture_warn_events(|| {
            record_degraded_features(&mut ctx, request::DegradedFeatures::default());
        });

        assert!(events.is_empty(), "no degradation must emit no WARN; got {events:?}");
        assert_eq!(
            ctx.get_metadata(DEGRADED_FEATURES_KEY),
            None,
            "no degradation must not stash metadata"
        );
    }

    // Test Utilities

    /// A captured WARN event: its rendered message and its `degraded_features`
    /// field value, if present.
    type CapturedWarn = (String, Option<String>);
    /// Thread-shared sink the capture layer appends each WARN event to.
    type WarnSink = Arc<Mutex<Vec<CapturedWarn>>>;

    /// Capture every WARN event emitted on the current thread while `f` runs,
    /// returning each event's message and its `degraded_features` field value.
    ///
    /// `tracing::subscriber::with_default` installs the subscriber for this thread
    /// only, so the capture is deterministic and does not race other tests.
    fn capture_warn_events<F: FnOnce()>(f: F) -> Vec<CapturedWarn> {
        use tracing_subscriber::layer::SubscriberExt as _;

        let events: WarnSink = Arc::new(Mutex::new(Vec::new()));
        let layer = WarnEventCapture(Arc::clone(&events));
        tracing::subscriber::with_default(tracing_subscriber::registry().with(layer), f);
        std::mem::take(&mut events.lock().unwrap())
    }

    struct WarnEventCapture(WarnSink);

    impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for WarnEventCapture {
        fn on_event(&self, event: &tracing::Event<'_>, _ctx: tracing_subscriber::layer::Context<'_, S>) {
            if *event.metadata().level() == tracing::Level::WARN {
                let mut visitor = WarnFieldVisitor::default();
                event.record(&mut visitor);
                self.0
                    .lock()
                    .unwrap()
                    .push((visitor.message, visitor.degraded_features));
            }
        }
    }

    #[derive(Default)]
    struct WarnFieldVisitor {
        degraded_features: Option<String>,
        message: String,
    }

    impl tracing::field::Visit for WarnFieldVisitor {
        fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
            if field.name() == "message" {
                self.message = format!("{value:?}");
            }
        }

        fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
            if field.name() == "degraded_features" {
                self.degraded_features = Some(value.to_owned());
            }
        }
    }
}
