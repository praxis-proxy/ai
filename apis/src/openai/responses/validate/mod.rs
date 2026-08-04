// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! `openai_responses_validate` filter: validate and enrich incoming Responses
//! API requests.
//!
//! Expects the upstream `openai_responses_format` classifier to have already
//! identified this request as a Responses API request and promoted
//! routing facts (`model`, `stream`, `store`, `background`) to
//! `openai_responses_format.*` metadata.
//!
//! This filter validates that the body is JSON, then does targeted field
//! extraction for `conversation.id`. It does **not** deserialize the full
//! body into a typed struct or validate provider-owned parameter
//! combinations.
//!
//! # YAML
//!
//! ```yaml
//! filter: openai_responses_validate
//! ```

use async_trait::async_trait;
use bytes::Bytes;
use praxis_filter::{
    EmptyFilterConfig, FilterAction, FilterError, HttpFilter, HttpFilterContext,
    body::{BodyAccess, BodyMode, MAX_JSON_BODY_BYTES},
    parse_filter_config,
};
use tracing::{debug, trace};

use super::{error::responses_error_rejection, extract_conversation_id, state::ResponsesState};

// -----------------------------------------------------------------------------
// OpenaiResponsesValidateFilter
// -----------------------------------------------------------------------------

/// Validates and enriches Responses API requests.
///
/// Parses the body as [`serde_json::Value`] for targeted field extraction.
/// Does not deserialize the full body into a typed struct or reject
/// provider-supported combinations such as background streaming.
///
/// Must be placed after `openai_responses_format` in the filter chain.
/// Skips non-Responses API requests (those not classified as
/// `openai_responses`).
///
/// Generates metadata: `responses.response_id` (format: `resp_` + 32
/// hex chars, CSPRNG), `responses.conversation_id`, `responses.store`,
/// `responses.background`, `responses.stream`.
///
/// This filter has no configuration, body buffering is handled by
/// the upstream `openai_responses_format` classifier.
#[derive(Default)]
pub struct OpenaiResponsesValidateFilter;

impl OpenaiResponsesValidateFilter {
    /// Create a filter from YAML config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the YAML config contains unknown fields.
    ///
    /// [`FilterError`]: praxis_filter::FilterError
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let _: EmptyFilterConfig = parse_filter_config("openai_responses_validate", config)?;
        Ok(Box::new(Self))
    }
}

#[async_trait]
impl HttpFilter for OpenaiResponsesValidateFilter {
    fn name(&self) -> &'static str {
        "openai_responses_validate"
    }

    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    fn request_body_mode(&self) -> BodyMode {
        BodyMode::StreamBuffer {
            max_bytes: Some(MAX_JSON_BODY_BYTES),
        }
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

        if ctx.get_metadata("openai_responses_format.format") != Some("openai_responses") {
            trace!("skipping non-responses request");
            return Ok(FilterAction::Release);
        }

        if is_bodyless_responses_request(&ctx.request.method, ctx.request.uri.path()) {
            trace!(
                method = %ctx.request.method,
                path = ctx.request.uri.path(),
                "skipping validation for bodyless endpoint"
            );
            return Ok(FilterAction::Release);
        }

        let parsed = match parse_request_body(ctx, body) {
            Ok(v) => v,
            Err(action) => return Ok(action),
        };

        let response_id = format!("resp_{}", ctx.id_generator.generate(ctx.time_source));
        let conversation_id = resolve_conversation_id(ctx, &parsed);

        enrich_context(ctx, &response_id, &conversation_id);
        ctx.extensions.insert(ResponsesState::from_request_body(parsed));

        debug!(
            response_id = %response_id,
            conversation_id = %conversation_id,
            "request validated, state initialized"
        );

        Ok(FilterAction::Release)
    }
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// Parse the request body as JSON.
fn parse_request_body(ctx: &HttpFilterContext<'_>, body: &Option<Bytes>) -> Result<serde_json::Value, FilterAction> {
    let streaming = ctx
        .get_metadata("openai_responses_format.stream")
        .is_some_and(|v| v == "true");
    let Some(chunk) = body.as_deref() else {
        debug!("rejecting request with missing body");
        return Err(reject_invalid("request body is required", streaming));
    };

    match serde_json::from_slice(chunk) {
        Ok(v) => Ok(v),
        Err(e) => {
            debug!(error = %e, "failed to parse request body");
            Err(reject_invalid(&format!("invalid request body: {e}"), streaming))
        },
    }
}

/// Check whether a Responses endpoint has no JSON request body to validate.
///
/// Assumes the format classifier already confirmed this is a Responses API path.
fn is_bodyless_responses_request(method: &http::Method, path: &str) -> bool {
    match *method {
        http::Method::GET | http::Method::DELETE => true,
        http::Method::POST => {
            let path = path.strip_suffix('/').unwrap_or(path);
            path.strip_prefix("/v1/responses/").is_some_and(|rest| {
                rest.strip_suffix("/cancel")
                    .is_some_and(|id| !id.is_empty() && !id.contains('/'))
            })
        },
        _ => false,
    }
}

/// Build a 400 rejection with a Responses API error body.
fn reject_invalid(message: &str, streaming: bool) -> FilterAction {
    FilterAction::Reject(responses_error_rejection(
        400,
        "invalid_request_error",
        message,
        streaming,
    ))
}

/// Extract or generate a conversation ID for the request.
fn resolve_conversation_id(ctx: &HttpFilterContext<'_>, body: &serde_json::Value) -> String {
    if let Some(id) = extract_conversation_id(body) {
        trace!(conversation_id = %id, "conversation ID extracted from request");
        id
    } else {
        let id = format!("conv_{}", ctx.id_generator.generate(ctx.time_source));
        trace!(conversation_id = %id, "conversation ID generated");
        id
    }
}

/// Enrich filter context with validated metadata for downstream filters.
///
/// Reads `stream`, `store`, `background` from `openai_responses_format.*`
/// classifier metadata and applies spec defaults.
fn enrich_context(ctx: &mut HttpFilterContext<'_>, response_id: &str, conversation_id: &str) {
    ctx.set_metadata("responses.response_id", response_id);
    ctx.set_metadata("responses.conversation_id", conversation_id);

    let store = ctx
        .get_metadata("openai_responses_format.store")
        .is_none_or(|v| v != "false");
    ctx.set_metadata("responses.store", if store { "true" } else { "false" });

    let background = ctx
        .get_metadata("openai_responses_format.background")
        .is_some_and(|v| v == "true");
    ctx.set_metadata("responses.background", if background { "true" } else { "false" });

    let stream = ctx
        .get_metadata("openai_responses_format.stream")
        .is_some_and(|v| v == "true");
    ctx.set_metadata("responses.stream", if stream { "true" } else { "false" });

    trace!(store, background, stream, "classifier metadata applied");
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::needless_raw_strings,
    clippy::needless_raw_string_hashes,
    reason = "tests"
)]
mod tests {
    use bytes::Bytes;

    use super::*;

    #[test]
    fn from_config_succeeds() {
        let filter = OpenaiResponsesValidateFilter::from_config(&serde_yaml::Value::Null).unwrap();
        assert_eq!(
            filter.name(),
            "openai_responses_validate",
            "filter name should be openai_responses_validate"
        );
    }

    #[test]
    fn from_config_rejects_unknown_fields() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("bogus: true").unwrap();
        let result = OpenaiResponsesValidateFilter::from_config(&yaml);
        assert!(result.is_err(), "unknown fields should be rejected");
    }

    #[test]
    fn body_access_is_read_only() {
        let filter = OpenaiResponsesValidateFilter;
        assert_eq!(
            filter.request_body_access(),
            BodyAccess::ReadOnly,
            "filter should use read-only body access"
        );
    }

    #[tokio::test]
    async fn valid_request_produces_metadata() {
        let ctx = run_filter(r#"{"model": "gpt-4.1", "input": "Hello"}"#, &[]).await;

        assert!(
            ctx.filter_metadata
                .get("responses.response_id")
                .is_some_and(|v| v.starts_with("resp_") && v.len() == 37),
            "response_id should be resp_ + 32 hex chars"
        );
        assert!(
            ctx.filter_metadata
                .get("responses.conversation_id")
                .is_some_and(|v| v.starts_with("conv_") && v.len() == 37),
            "conversation_id should be conv_ + 32 hex chars"
        );
        assert_eq!(
            ctx.filter_metadata.get("responses.store").map(String::as_str),
            Some("true"),
            "store should default to true when classifier has no value"
        );
        assert_eq!(
            ctx.filter_metadata.get("responses.background").map(String::as_str),
            Some("false"),
            "background should default to false"
        );
        assert_eq!(
            ctx.filter_metadata.get("responses.stream").map(String::as_str),
            Some("false"),
            "stream should default to false"
        );
    }

    #[tokio::test]
    async fn valid_request_creates_responses_state() {
        let ctx = run_filter(
            r#"{"model": "gpt-4.1", "input": "Hello", "tools": [{"type": "function", "name": "f"}]}"#,
            &[],
        )
        .await;

        let state = ctx
            .extensions
            .get::<ResponsesState>()
            .expect("validate should insert ResponsesState");
        assert_eq!(state.input.len(), 1, "input should be populated");
        assert_eq!(state.tools.len(), 1, "tools should be populated");
        assert_eq!(state.iteration, 0, "iteration should start at 0");
        assert!(state.tool_calls.is_empty(), "tool_calls should start empty");
    }

    #[tokio::test]
    async fn reads_stream_from_classifier_metadata() {
        let ctx = run_filter(r#"{"input": "Hi"}"#, &[("openai_responses_format.stream", "true")]).await;

        assert_eq!(
            ctx.filter_metadata.get("responses.stream").map(String::as_str),
            Some("true"),
            "stream should be read from classifier metadata"
        );
    }

    #[tokio::test]
    async fn reads_store_from_classifier_metadata() {
        let ctx = run_filter(r#"{"input": "Hi"}"#, &[("openai_responses_format.store", "false")]).await;

        assert_eq!(
            ctx.filter_metadata.get("responses.store").map(String::as_str),
            Some("false"),
            "store should be read from classifier metadata"
        );
    }

    #[tokio::test]
    async fn reads_background_from_classifier_metadata() {
        let ctx = run_filter(r#"{"input": "Hi"}"#, &[("openai_responses_format.background", "true")]).await;

        assert_eq!(
            ctx.filter_metadata.get("responses.background").map(String::as_str),
            Some("true"),
            "background should be read from classifier metadata"
        );
    }

    #[tokio::test]
    async fn valid_request_with_conversation_id() {
        let ctx = run_filter(r#"{"input": "Hi", "conversation": {"id": "conv_existing_123"}}"#, &[]).await;

        assert_eq!(
            ctx.filter_metadata.get("responses.conversation_id").map(String::as_str),
            Some("conv_existing_123"),
            "conversation_id should be extracted from request body"
        );
    }

    #[tokio::test]
    async fn valid_request_with_bare_string_conversation_id() {
        let ctx = run_filter(r#"{"input": "Hi", "conversation": "conv_existing_123"}"#, &[]).await;

        assert_eq!(
            ctx.filter_metadata.get("responses.conversation_id").map(String::as_str),
            Some("conv_existing_123"),
            "bare-string conversation ID should be extracted from request body"
        );
    }

    #[tokio::test]
    async fn valid_request_generates_conversation_id() {
        let ctx = run_filter(r#"{"input": "Hi"}"#, &[]).await;

        assert!(
            ctx.filter_metadata
                .get("responses.conversation_id")
                .is_some_and(|v| v.starts_with("conv_") && v.len() == 37),
            "conversation_id should be conv_ + 32 hex chars"
        );
    }

    #[tokio::test]
    async fn stream_and_background_accepted() {
        let ctx = run_filter(
            r#"{"input": "test"}"#,
            &[
                ("openai_responses_format.stream", "true"),
                ("openai_responses_format.background", "true"),
            ],
        )
        .await;
        assert_eq!(
            ctx.get_metadata("responses.stream"),
            Some("true"),
            "stream=true should be preserved"
        );
        assert_eq!(
            ctx.get_metadata("responses.background"),
            Some("true"),
            "background=true should be preserved"
        );
    }

    #[tokio::test]
    async fn background_without_store_accepted() {
        let ctx = run_filter(
            r#"{"input": "test"}"#,
            &[
                ("openai_responses_format.background", "true"),
                ("openai_responses_format.store", "false"),
            ],
        )
        .await;
        assert_eq!(
            ctx.get_metadata("responses.background"),
            Some("true"),
            "background=true should be preserved"
        );
        assert_eq!(
            ctx.get_metadata("responses.store"),
            Some("false"),
            "store=false should be preserved"
        );
    }

    #[tokio::test]
    async fn streaming_rejection_has_sse_content_type() {
        let action = run_filter_raw("not valid json", &[("openai_responses_format.stream", "true")]).await;
        if let FilterAction::Reject(rejection) = action {
            let has_content_type = rejection
                .headers
                .iter()
                .any(|(k, v)| k == "content-type" && v == "text/event-stream");
            assert!(
                has_content_type,
                "streaming rejection should have text/event-stream content-type"
            );
        } else {
            panic!("expected rejection");
        }
    }

    #[tokio::test]
    async fn non_streaming_rejection_has_json_content_type() {
        let action = run_filter_raw("not valid json", &[]).await;
        if let FilterAction::Reject(rejection) = action {
            let has_content_type = rejection
                .headers
                .iter()
                .any(|(k, v)| k == "content-type" && v == "application/json");
            assert!(
                has_content_type,
                "non-streaming rejection should have application/json content-type"
            );
        } else {
            panic!("expected rejection");
        }
    }

    #[tokio::test]
    async fn rejection_body_uses_responses_error_format() {
        let action = run_filter_raw("not valid json", &[]).await;
        if let FilterAction::Reject(rejection) = action {
            let body = rejection.body.unwrap();
            let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(
                parsed["error"]["type"].as_str(),
                Some("invalid_request_error"),
                "rejection body should have error type=invalid_request_error"
            );
            assert!(
                parsed["error"]["message"].is_string(),
                "rejection body should contain error message"
            );
            assert!(
                parsed["error"]["param"].is_null(),
                "rejection body should have error param=null"
            );
        } else {
            panic!("expected rejection");
        }
    }

    #[test]
    fn reject_invalid_escapes_control_characters() {
        let action = reject_invalid("line1\nline2", false);
        if let FilterAction::Reject(rejection) = action {
            let body = rejection.body.unwrap();
            let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(
                parsed["error"]["message"].as_str(),
                Some("line1\nline2"),
                "control characters in rejection body should remain valid JSON"
            );
        } else {
            panic!("expected rejection");
        }
    }

    #[tokio::test]
    async fn skips_chat_completions_request() {
        let filter = make_filter();
        let req = Box::leak(Box::new(crate::test_utils::make_request(
            http::Method::POST,
            "/v1/chat/completions",
        )));
        let mut ctx = crate::test_utils::make_filter_context(req);
        ctx.set_metadata("openai_responses_format.format", "openai_chat_completions");
        let mut body = Some(Bytes::from(r#"{"messages":[]}"#));

        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
        assert!(
            matches!(action, FilterAction::Release),
            "chat completions request should be released without validation"
        );
        assert!(
            !ctx.filter_metadata.contains_key("responses.response_id"),
            "responses metadata should not be set for non-responses requests"
        );
    }

    #[tokio::test]
    async fn skips_missing_format_metadata() {
        let filter = make_filter();
        let req = Box::leak(Box::new(crate::test_utils::make_request(
            http::Method::POST,
            "/v1/responses",
        )));
        let mut ctx = crate::test_utils::make_filter_context(req);
        let mut body = Some(Bytes::from(r#"{"input":"test"}"#));

        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
        assert!(
            matches!(action, FilterAction::Release),
            "request without classifier metadata should be released without validation"
        );
    }

    #[tokio::test]
    async fn not_end_of_stream_continues() {
        let filter = OpenaiResponsesValidateFilter;
        let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
        let mut ctx = crate::test_utils::make_filter_context(&req);
        let mut body = Some(Bytes::from(r#"{"input": "partial"}"#));

        let action = filter.on_request_body(&mut ctx, &mut body, false).await.unwrap();
        assert!(
            matches!(action, FilterAction::Continue),
            "non-end-of-stream should continue"
        );
    }

    #[tokio::test]
    async fn minimal_request_without_model() {
        let ctx = run_filter(r#"{"input": "Hello"}"#, &[]).await;

        assert!(
            ctx.filter_metadata.contains_key("responses.response_id"),
            "response_id should still be generated"
        );
    }

    #[tokio::test]
    async fn skips_get_response_without_body() {
        let filter = make_filter();
        let req = Box::leak(Box::new(crate::test_utils::make_request(
            http::Method::GET,
            "/v1/responses/resp_abc123",
        )));
        let mut ctx = crate::test_utils::make_filter_context(req);
        ctx.set_metadata("openai_responses_format.format", "openai_responses");
        let mut body = None;

        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
        assert!(
            matches!(action, FilterAction::Release),
            "GET request should be released without body validation"
        );
        assert!(
            !ctx.filter_metadata.contains_key("responses.response_id"),
            "responses metadata should not be set for bodyless requests"
        );
    }

    #[tokio::test]
    async fn skips_delete_response_without_body() {
        let filter = make_filter();
        let req = Box::leak(Box::new(crate::test_utils::make_request(
            http::Method::DELETE,
            "/v1/responses/resp_abc123",
        )));
        let mut ctx = crate::test_utils::make_filter_context(req);
        ctx.set_metadata("openai_responses_format.format", "openai_responses");
        let mut body = None;

        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
        assert!(
            matches!(action, FilterAction::Release),
            "DELETE request should be released without body validation"
        );
    }

    #[tokio::test]
    async fn skips_get_input_items_without_body() {
        let filter = make_filter();
        let req = Box::leak(Box::new(crate::test_utils::make_request(
            http::Method::GET,
            "/v1/responses/resp_abc123/input_items",
        )));
        let mut ctx = crate::test_utils::make_filter_context(req);
        ctx.set_metadata("openai_responses_format.format", "openai_responses");
        let mut body = None;

        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
        assert!(
            matches!(action, FilterAction::Release),
            "GET /input_items request should be released without body validation"
        );
    }

    #[tokio::test]
    async fn skips_post_cancel_without_body() {
        let filter = make_filter();
        let req = Box::leak(Box::new(crate::test_utils::make_request(
            http::Method::POST,
            "/v1/responses/resp_abc123/cancel",
        )));
        let mut ctx = crate::test_utils::make_filter_context(req);
        ctx.set_metadata("openai_responses_format.format", "openai_responses");
        let mut body = None;

        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
        assert!(
            matches!(action, FilterAction::Release),
            "POST /cancel request should be released without body validation"
        );
        assert!(
            !ctx.filter_metadata.contains_key("responses.response_id"),
            "responses metadata should not be set for bodyless requests"
        );
    }

    #[tokio::test]
    async fn skips_post_cancel_with_trailing_slash() {
        let filter = make_filter();
        let req = Box::leak(Box::new(crate::test_utils::make_request(
            http::Method::POST,
            "/v1/responses/resp_abc123/cancel/",
        )));
        let mut ctx = crate::test_utils::make_filter_context(req);
        ctx.set_metadata("openai_responses_format.format", "openai_responses");
        let mut body = None;

        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
        assert!(
            matches!(action, FilterAction::Release),
            "POST /cancel/ with trailing slash should be released without body validation"
        );
    }

    #[tokio::test]
    async fn post_input_tokens_still_validates_body() {
        let filter = make_filter();
        let req = Box::leak(Box::new(crate::test_utils::make_request(
            http::Method::POST,
            "/v1/responses/input_tokens",
        )));
        let mut ctx = crate::test_utils::make_filter_context(req);
        ctx.set_metadata("openai_responses_format.format", "openai_responses");
        let mut body = None;

        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
        assert!(
            matches!(action, FilterAction::Reject(_)),
            "POST /input_tokens without body should be rejected, not released"
        );
    }

    // -------------------------------------------------------------------------
    // Test Utilities
    // -------------------------------------------------------------------------

    fn make_filter() -> Box<dyn HttpFilter> {
        OpenaiResponsesValidateFilter::from_config(&serde_yaml::Value::Null).unwrap()
    }

    async fn run_filter(body_str: &str, classifier_metadata: &[(&str, &str)]) -> HttpFilterContext<'static> {
        let filter = make_filter();
        let req = Box::leak(Box::new(crate::test_utils::make_request(
            http::Method::POST,
            "/v1/responses",
        )));
        let mut ctx = crate::test_utils::make_filter_context(req);
        ctx.set_metadata("openai_responses_format.format", "openai_responses");
        for (k, v) in classifier_metadata {
            ctx.set_metadata(*k, *v);
        }
        let mut body = Some(Bytes::from(body_str.to_owned()));

        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
        assert!(
            matches!(action, FilterAction::Release),
            "valid request should release: got {action:?}"
        );

        ctx
    }

    async fn run_filter_raw(body_str: &str, classifier_metadata: &[(&str, &str)]) -> FilterAction {
        let filter = make_filter();
        let req = Box::leak(Box::new(crate::test_utils::make_request(
            http::Method::POST,
            "/v1/responses",
        )));
        let mut ctx = crate::test_utils::make_filter_context(req);
        ctx.set_metadata("openai_responses_format.format", "openai_responses");
        for (k, v) in classifier_metadata {
            ctx.set_metadata(*k, *v);
        }
        let mut body = Some(Bytes::from(body_str.to_owned()));

        filter.on_request_body(&mut ctx, &mut body, true).await.unwrap()
    }
}
