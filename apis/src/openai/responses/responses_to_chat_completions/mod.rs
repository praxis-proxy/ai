// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Responses API to Chat Completions translation.

mod config;

/// Finite provider error normalization.
mod error;

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::too_many_lines,
    reason = "tests"
)]
mod tests;

use async_trait::async_trait;
use bytes::Bytes;
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, parse_filter_config,
};
use tracing::{debug, trace, warn};

use self::{
    config::{ResponsesToChatCompletionsConfig, build_config},
    error::normalize_provider_error,
};
use super::{
    error::{responses_error_body, responses_error_rejection},
    state::ResponsesState,
};
use crate::{
    classifier::is_responses_create,
    openai::translation::chat_completions::{
        ResponseContext, chat_response_to_response_resource, responses_state_to_chat_request,
    },
};

/// Metadata recording that request translation completed successfully.
const ARMED_KEY: &str = "responses_to_chat_completions.armed";

/// Metadata recording the Responses resource creation timestamp.
const CREATED_AT_KEY: &str = "responses_to_chat_completions.created_at";

/// Metadata recording the upstream response status while headers are mutable.
const RESPONSE_STATUS_KEY: &str = "responses_to_chat_completions.response_status";

/// Metadata selecting the finite response transformation.
const RESPONSE_TRANSFORM_KEY: &str = "responses_to_chat_completions.response_transform";

/// Marker for a successful Chat Completions response.
const RESPONSE_TRANSFORM_SUCCESS: &str = "success";

/// Marker for a finite provider error response.
const RESPONSE_TRANSFORM_ERROR: &str = "error";

/// Translates canonical Responses create requests for a Chat Completions backend.
///
/// The filter consumes the classification metadata and `ResponsesState`
/// produced by `openai_responses_format` and `openai_responses_validate`.
/// It converts the enriched request to Chat Completions wire format, converts
/// finite successful Chat responses back to Responses resources, and
/// normalizes finite provider errors while preserving their HTTP status.
/// Chat Completions SSE is left byte-for-byte unchanged for the separate
/// incremental stream converter.
///
/// Configure `path_rewrite` after this filter when the upstream endpoint must
/// change from `/v1/responses` to `/v1/chat/completions`.
///
/// # YAML
///
/// ```yaml
/// filter: responses_to_chat_completions
/// ```
///
/// # Full YAML
///
/// ```yaml
/// filter: responses_to_chat_completions
/// max_body_bytes: 67108864
/// ```
pub struct ResponsesToChatCompletionsFilter {
    /// Parsed and validated body limits.
    config: ResponsesToChatCompletionsConfig,
}

impl ResponsesToChatCompletionsFilter {
    /// Create the filter from YAML configuration.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] when the configuration contains unknown fields
    /// or an invalid body-size limit.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let parsed = if config.is_null() {
            ResponsesToChatCompletionsConfig::default()
        } else {
            parse_filter_config("responses_to_chat_completions", config)?
        };
        Ok(Box::new(Self {
            config: build_config(parsed)?,
        }))
    }

    /// Build and size-check the owned Chat Completions request body.
    fn translated_request_bytes(
        &self,
        ctx: &HttpFilterContext<'_>,
    ) -> Result<Result<Vec<u8>, FilterAction>, FilterError> {
        let streaming = request_is_streaming(ctx);
        let translated = match translate_canonical_state(ctx, streaming) {
            Ok(value) => value,
            Err(action) => return Ok(Err(action)),
        };
        let serialized = serde_json::to_vec(&translated)
            .map_err(|error| -> FilterError { format!("responses_to_chat_completions: {error}").into() })?;
        if serialized.len() > self.config.max_body_bytes {
            debug!(
                body_bytes = serialized.len(),
                max_bytes = self.config.max_body_bytes,
                "translated request body exceeds maximum size"
            );
            return Ok(Err(FilterAction::Reject(responses_error_rejection(
                413,
                "invalid_request_error",
                "request body exceeds maximum size",
                streaming,
            ))));
        }
        Ok(Ok(serialized))
    }

    /// Transform one fully buffered finite provider response.
    fn transform_finite_response(
        &self,
        ctx: &HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
    ) -> Result<(), FilterError> {
        match ctx.get_metadata(RESPONSE_TRANSFORM_KEY) {
            Some(RESPONSE_TRANSFORM_ERROR) => {
                transform_provider_error(ctx, body, self.config.max_body_bytes)?;
                Ok(())
            },
            Some(RESPONSE_TRANSFORM_SUCCESS) => self.transform_success_response(ctx, body),
            _ => Err("responses_to_chat_completions: missing finite response transform state".into()),
        }
    }

    /// Convert and size-check a successful finite Chat response.
    fn transform_success_response(
        &self,
        ctx: &HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
    ) -> Result<(), FilterError> {
        match translate_success_response(ctx, body.as_deref().unwrap_or_default()) {
            Ok(translated) if translated.len() <= self.config.max_body_bytes => {
                *body = Some(translated);
                Ok(())
            },
            Ok(translated) => {
                debug!(
                    body_bytes = translated.len(),
                    max_bytes = self.config.max_body_bytes,
                    "translated response body exceeds maximum size"
                );
                Err("responses_to_chat_completions: translated response exceeds maximum size".into())
            },
            Err(error) => {
                warn!(error = %error, "upstream provider returned an invalid Chat Completions response");
                Err(format!("responses_to_chat_completions: invalid Chat Completions response: {error}").into())
            },
        }
    }
}

#[async_trait]
impl HttpFilter for ResponsesToChatCompletionsFilter {
    fn name(&self) -> &'static str {
        "responses_to_chat_completions"
    }

    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadWrite
    }

    fn request_body_mode(&self) -> BodyMode {
        BodyMode::StreamBuffer {
            max_bytes: Some(self.config.max_body_bytes),
        }
    }

    fn response_body_mode(&self) -> BodyMode {
        BodyMode::Stream
    }

    fn response_body_access(&self) -> BodyAccess {
        BodyAccess::ReadWrite
    }

    async fn on_request(&self, _ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        Ok(FilterAction::Continue)
    }

    async fn on_response(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        if ctx.get_metadata(ARMED_KEY) != Some("true") {
            return Ok(FilterAction::Continue);
        }

        let transform = match finite_response_transform(ctx) {
            Ok(Some(transform)) => transform,
            Ok(None) => return Ok(FilterAction::Continue),
            Err(action) => return Ok(action),
        };
        let Some(status) = ctx.response_header.as_ref().map(|response| response.status) else {
            return Ok(FilterAction::Continue);
        };

        ctx.set_metadata(RESPONSE_TRANSFORM_KEY, transform);
        ctx.set_metadata(RESPONSE_STATUS_KEY, status.as_u16().to_string());
        ctx.set_response_body_mode(BodyMode::StreamBuffer {
            max_bytes: Some(self.config.max_body_bytes),
        });
        prepare_transformed_response_headers(ctx);

        Ok(FilterAction::Continue)
    }

    fn on_response_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if ctx.get_metadata(ARMED_KEY) != Some("true")
            || ctx.get_metadata(RESPONSE_TRANSFORM_KEY).is_none()
            || !end_of_stream
        {
            return Ok(FilterAction::Continue);
        }

        self.transform_finite_response(ctx, body)?;
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

        if let Some(action) = request_disposition(ctx) {
            return Ok(action);
        }

        let serialized = match self.translated_request_bytes(ctx)? {
            Ok(bytes) => bytes,
            Err(action) => return Ok(action),
        };
        ctx.request_headers_to_remove.push(http::header::ACCEPT_ENCODING);
        *body = Some(Bytes::from(serialized));
        ctx.set_metadata(ARMED_KEY, "true");
        ctx.set_metadata(CREATED_AT_KEY, ctx.time_source.now().as_secs().to_string());

        Ok(FilterAction::Continue)
    }
}

/// Decide whether the current request should translate, release, or fail closed.
fn request_disposition(ctx: &HttpFilterContext<'_>) -> Option<FilterAction> {
    if !is_responses_create(&ctx.request.method, ctx.request.uri.path()) {
        return Some(FilterAction::Continue);
    }
    match ctx.get_metadata("openai_responses_format.format") {
        Some("openai_responses") => None,
        Some(format) => {
            trace!(format, "releasing request classified as a different API format");
            Some(FilterAction::Release)
        },
        None => {
            warn!(
                prerequisite = "openai_responses_format",
                "request pipeline state is unavailable"
            );
            Some(missing_pipeline_state(false))
        },
    }
}

/// Convert the validator-owned canonical state to a Chat request value.
fn translate_canonical_state(ctx: &HttpFilterContext<'_>, streaming: bool) -> Result<serde_json::Value, FilterAction> {
    let Some(state) = ctx.extensions.get::<ResponsesState>() else {
        warn!(
            prerequisite = "openai_responses_validate",
            "request pipeline state is unavailable"
        );
        return Err(missing_pipeline_state(streaming));
    };
    responses_state_to_chat_request(&state.request_body, &state.messages, &state.tools, &state.tool_choice).map_err(
        |error| {
            debug!(error = %error, "Responses request cannot be represented by Chat Completions");
            FilterAction::Reject(responses_error_rejection(
                400,
                "invalid_request_error",
                &error.to_string(),
                streaming,
            ))
        },
    )
}

/// Return the client stream preference captured by the classifier.
fn request_is_streaming(ctx: &HttpFilterContext<'_>) -> bool {
    ctx.get_metadata("openai_responses_format.stream")
        .is_some_and(|value| value == "true")
}

/// Detect an SSE media type while response headers are still available.
fn is_sse_response(ctx: &HttpFilterContext<'_>) -> bool {
    ctx.response_header
        .as_ref()
        .and_then(|response| response.headers.get(http::header::CONTENT_TYPE))
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .is_some_and(|value| value.trim().eq_ignore_ascii_case("text/event-stream"))
}

/// Read and validate the status captured before response headers were committed.
fn captured_response_status(ctx: &HttpFilterContext<'_>) -> Result<http::StatusCode, FilterError> {
    let status = ctx
        .get_metadata(RESPONSE_STATUS_KEY)
        .ok_or_else(|| -> FilterError { "responses_to_chat_completions: missing captured response status".into() })?
        .parse::<u16>()
        .map_err(|error| -> FilterError {
            format!("responses_to_chat_completions: invalid captured response status: {error}").into()
        })?;
    http::StatusCode::from_u16(status).map_err(|error| -> FilterError {
        format!("responses_to_chat_completions: invalid captured response status: {error}").into()
    })
}

/// Normalize a finite provider error using the response-phase status snapshot.
fn transform_provider_error(
    ctx: &HttpFilterContext<'_>,
    body: &mut Option<Bytes>,
    max_body_bytes: usize,
) -> Result<(), FilterError> {
    let status = captured_response_status(ctx)?;
    let normalized = normalize_provider_error(status, body.as_deref().unwrap_or_default());
    let mut transformed = responses_error_body(&normalized.code, &normalized.message);
    if transformed.len() > max_body_bytes {
        let fallback = normalize_provider_error(status, &[]);
        transformed = responses_error_body(&fallback.code, &fallback.message);
    }
    if transformed.len() > max_body_bytes {
        return Err("responses_to_chat_completions: normalized provider error exceeds maximum size".into());
    }
    *body = Some(transformed);
    Ok(())
}

/// Select finite success or error handling while response headers are mutable.
fn finite_response_transform(ctx: &HttpFilterContext<'_>) -> Result<Option<&'static str>, FilterAction> {
    let Some(status) = ctx.response_header.as_ref().map(|response| response.status) else {
        return Ok(None);
    };
    if is_sse_response(ctx) {
        return Ok(None);
    }
    if status.is_success() {
        if status != http::StatusCode::OK || has_unsupported_success_representation(ctx) {
            return Err(FilterAction::Reject(responses_error_rejection(
                502,
                "server_error",
                "upstream provider returned an unsupported response representation",
                request_is_streaming(ctx),
            )));
        }
        return Ok(Some(RESPONSE_TRANSFORM_SUCCESS));
    }
    Ok((status.is_client_error() || status.is_server_error()).then_some(RESPONSE_TRANSFORM_ERROR))
}

/// Return whether a successful finite body cannot be safely translated.
fn has_unsupported_success_representation(ctx: &HttpFilterContext<'_>) -> bool {
    ctx.response_header.as_ref().is_some_and(|response| {
        response.headers.contains_key(http::header::CONTENT_ENCODING)
            || response.headers.contains_key(http::header::CONTENT_RANGE)
    })
}

/// Remove representation metadata invalidated by replacing a finite body.
fn prepare_transformed_response_headers(ctx: &mut HttpFilterContext<'_>) {
    if let Some(response) = &mut ctx.response_header {
        response.headers.remove(http::header::CONTENT_LENGTH);
        response.headers.remove(http::header::CONTENT_ENCODING);
        response.headers.remove(http::header::CONTENT_RANGE);
        response.headers.remove(http::header::ETAG);
        for header in ["content-digest", "content-md5", "digest", "repr-digest"] {
            response.headers.remove(header);
        }
        response.headers.insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("application/json"),
        );
        ctx.response_headers_modified = true;
    }
}

/// Convert a finite successful Chat response into a Responses resource.
fn translate_success_response(ctx: &HttpFilterContext<'_>, body: &[u8]) -> Result<Bytes, FilterError> {
    let response_id = ctx
        .get_metadata("responses.response_id")
        .ok_or_else(|| -> FilterError { "responses_to_chat_completions: missing response id".into() })?;
    let created_at = ctx
        .get_metadata(CREATED_AT_KEY)
        .and_then(|value| value.parse::<u64>().ok())
        .ok_or_else(|| -> FilterError { "responses_to_chat_completions: missing creation timestamp".into() })?;
    let state = ctx
        .extensions
        .get::<ResponsesState>()
        .ok_or_else(|| -> FilterError { "responses_to_chat_completions: missing Responses state".into() })?;
    let response_context =
        ResponseContext::from_responses_request(&state.request_body, response_id.to_owned(), created_at)
            .with_completed_at(ctx.time_source.now().as_secs());
    let provider_response: serde_json::Value = serde_json::from_slice(body)
        .map_err(|error| -> FilterError { format!("responses_to_chat_completions: {error}").into() })?;
    let translated = chat_response_to_response_resource(&provider_response, &response_context)
        .map_err(|error| -> FilterError { format!("responses_to_chat_completions: {error}").into() })?;
    let serialized = serde_json::to_vec(&translated)
        .map_err(|error| -> FilterError { format!("responses_to_chat_completions: {error}").into() })?;
    Ok(Bytes::from(serialized))
}

/// Build the fail-closed action for missing classifier or validator state.
fn missing_pipeline_state(streaming: bool) -> FilterAction {
    FilterAction::Reject(responses_error_rejection(
        500,
        "server_error",
        "request pipeline state is unavailable",
        streaming,
    ))
}
