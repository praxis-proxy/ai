// SPDX-License-Identifier: Apache-2.0
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
//! extraction for `conversation.id` and mutually exclusive history
//! selectors. It does **not** deserialize the full body into a typed
//! struct or validate provider-owned parameter combinations.
//!
//! # YAML
//!
//! ```yaml
//! filter: openai_responses_validate
//! conditions:
//!   - unless:
//!       bound_upstream:
//!         application_provider: openai
//! ```

use async_trait::async_trait;
use bytes::Bytes;
use praxis_filter::{
    BoundUpstreamBodyOutcome, EmptyFilterConfig, FilterAction, FilterError, HttpFilter, HttpFilterContext,
    body::{BodyAccess, BodyMode, MAX_JSON_BODY_BYTES},
    parse_filter_config,
};
use tracing::{debug, trace};

use super::{
    error::{responses_error_rejection, responses_error_rejection_with_code},
    extract_conversation_id,
    state::ResponsesState,
};

// -----------------------------------------------------------------------------
// OpenaiResponsesValidateFilter
// -----------------------------------------------------------------------------

/// Validates and enriches Responses API requests.
///
/// In a legacy `openai_responses_format` chain, parses the body as
/// [`serde_json::Value`] for targeted field extraction. After
/// `openai_responses_request`, uses its initialized [`ResponsesState`] without
/// parsing again. Does not validate other provider-owned parameter
/// combinations. Rejects `background=true` because
/// locally managed pipelines cannot observe a provider-owned asynchronous
/// lifecycle. A provider-aware pipeline may condition this filter with
/// `unless: { bound_upstream: { application_provider: openai } }`; Praxis then
/// runs the same validation once at the bound-upstream body barrier and skips
/// it for OpenAI-owned passthrough.
///
/// Must be placed after `openai_responses_format` or
/// `openai_responses_request` in the filter chain.
/// Skips non-Responses API requests (those not classified as
/// `openai_responses`).
///
/// Generates metadata: `responses.response_id` (format: `resp_` + 32
/// hex chars, CSPRNG), `responses.conversation_id`, `responses.store`,
/// `responses.background`, `responses.stream`.
///
/// This filter has no filter-specific configuration. Request conditions may
/// gate it on `bound_upstream`; body buffering is shared with the preceding
/// Responses request classifier.
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

/// Validate one complete Responses request body in either body phase.
fn validate_complete_body(ctx: &mut HttpFilterContext<'_>, body: &Option<Bytes>) -> FilterAction {
    if let Some(action) = complete_body_policy(ctx) {
        return action;
    }

    match parse_request_body(body) {
        Ok(parsed) => validate_parsed_body(ctx, parsed),
        Err(action) => action,
    }
}

/// Apply classification and lifecycle guards before any JSON parsing.
#[expect(
    clippy::cognitive_complexity,
    reason = "tracing fields inflate the score for four flat lifecycle guards"
)]
fn complete_body_policy(ctx: &HttpFilterContext<'_>) -> Option<FilterAction> {
    if ctx.get_metadata("openai_responses_format.format") != Some("openai_responses") {
        trace!("skipping non-responses request");
        return Some(FilterAction::Release);
    }

    if is_bodyless_responses_request(&ctx.request.method, ctx.request.uri.path()) {
        trace!(method = %ctx.request.method, path = ctx.request.uri.path(), "skipping validation for bodyless endpoint");
        return Some(FilterAction::Release);
    }

    if background_requested(ctx) {
        debug!("rejecting unsupported Responses background mode");
        return Some(reject_invalid("background mode is not supported"));
    }

    // The consolidated request processor already parsed, validated, and
    // initialized this body. In that chain the validator is only the lifecycle
    // policy boundary, so do not parse again or regenerate request IDs.
    if ctx.extensions.get::<ResponsesState>().is_some() {
        trace!("Responses state already initialized; lifecycle policy complete");
        return Some(FilterAction::Release);
    }

    None
}

/// Validate parsed fields and initialize request-scoped Responses state.
fn validate_parsed_body(ctx: &mut HttpFilterContext<'_>, parsed: serde_json::Value) -> FilterAction {
    if let Some(action) = reject_conflicting_history_selectors(&parsed) {
        return action;
    }

    let response_id = format!("resp_{}", ctx.id_generator.generate(ctx.time_source));
    let conversation_id = resolve_conversation_id(ctx, &parsed);

    enrich_context(ctx, &response_id, &conversation_id);
    insert_responses_state(ctx, parsed, &response_id);

    debug!(
        response_id = %response_id,
        conversation_id = %conversation_id,
        "request validated, state initialized"
    );

    FilterAction::Release
}

#[async_trait]
impl HttpFilter for OpenaiResponsesValidateFilter {
    fn name(&self) -> &'static str {
        "openai_responses_validate"
    }

    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    fn bound_upstream_request_body_access(&self) -> BodyAccess {
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
        Ok(validate_complete_body(ctx, body))
    }

    async fn on_bound_upstream_request_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
    ) -> Result<BoundUpstreamBodyOutcome, FilterError> {
        Ok(match validate_complete_body(ctx, body) {
            FilterAction::Reject(rejection) => BoundUpstreamBodyOutcome::Reject(rejection),
            _ => BoundUpstreamBodyOutcome::Continue,
        })
    }
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// Initialize canonical request state, including metadata that must survive IRR steps.
fn insert_responses_state(ctx: &mut HttpFilterContext<'_>, parsed: serde_json::Value, response_id: &str) {
    let mut state = ResponsesState::from_request_body(parsed);
    state.response_id = Some(response_id.to_owned());
    ctx.extensions.insert(state);
}

/// Parse the request body as JSON.
fn parse_request_body(body: &Option<Bytes>) -> Result<serde_json::Value, FilterAction> {
    let Some(chunk) = body.as_deref() else {
        debug!("rejecting request with missing body");
        return Err(reject_invalid("request body is required"));
    };

    match serde_json::from_slice(chunk) {
        Ok(v) => Ok(v),
        Err(e) => {
            debug!(error = %e, "failed to parse request body");
            Err(reject_invalid(&format!("invalid request body: {e}")))
        },
    }
}

/// Reject requests that select both supported sources of conversation history.
fn reject_conflicting_history_selectors(body: &serde_json::Value) -> Option<FilterAction> {
    let conflicts = body.get("previous_response_id").is_some_and(|value| !value.is_null())
        && body.get("conversation").is_some_and(|value| !value.is_null());
    conflicts.then(|| {
        FilterAction::Reject(responses_error_rejection_with_code(
            400,
            "invalid_request_error",
            "mutually_exclusive_parameters",
            "Mutually exclusive parameters. Ensure you are only providing one of: 'previous_response_id' or 'conversation'.",
        ))
    })
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
fn reject_invalid(message: &str) -> FilterAction {
    FilterAction::Reject(responses_error_rejection(400, "invalid_request_error", message))
}

/// Whether the classifier observed `background: true` on this create request.
fn background_requested(ctx: &HttpFilterContext<'_>) -> bool {
    ctx.get_metadata("openai_responses_format.background") == Some("true")
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
    use praxis_filter::{FilterEntry, FilterPipeline, FilterRegistry};

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

    #[test]
    fn bound_upstream_body_access_is_read_only() {
        let filter = OpenaiResponsesValidateFilter;
        assert_eq!(
            filter.bound_upstream_request_body_access(),
            BodyAccess::ReadOnly,
            "the validator must support deferred request-scoped policy"
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
    async fn existing_responses_state_is_not_parsed_or_initialized_twice() {
        let filter = make_filter();
        let req = Box::leak(Box::new(crate::test_utils::make_request(
            http::Method::POST,
            "/v1/responses",
        )));
        let mut ctx = crate::test_utils::make_filter_context(req);
        ctx.set_metadata("openai_responses_format.format", "openai_responses");
        ctx.set_metadata("responses.response_id", "resp_existing");
        ctx.extensions
            .insert(ResponsesState::from_request_body(serde_json::json!({
                "model": "gpt-4.1",
                "input": "already parsed"
            })));
        let mut body = Some(Bytes::from_static(b"this would fail a second JSON parse"));

        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

        assert!(matches!(action, FilterAction::Release));
        assert_eq!(
            ctx.get_metadata("responses.response_id"),
            Some("resp_existing"),
            "the consolidated request processor's state must remain authoritative"
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
        assert_eq!(
            state.response_id.as_deref(),
            ctx.filter_metadata.get("responses.response_id").map(String::as_str),
            "canonical state should retain the response id across IRR steps"
        );
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
    async fn rejects_conflicting_history_selectors() {
        let action = run_filter_raw(
            r#"{"input":"next","previous_response_id":"resp_win","conversation":"conv_lose"}"#,
            &[],
        )
        .await;
        let FilterAction::Reject(rejection) = action else {
            panic!("expected rejection");
        };
        assert_eq!(rejection.status, 400);
        let body: serde_json::Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["code"], "mutually_exclusive_parameters");
        assert_eq!(
            body["error"]["message"],
            "Mutually exclusive parameters. Ensure you are only providing one of: 'previous_response_id' or 'conversation'."
        );
    }

    #[tokio::test]
    async fn streaming_selector_conflict_uses_json_validation_error() {
        let action = run_filter_raw(
            r#"{"input":"next","previous_response_id":"resp_win","conversation":"conv_lose","stream":true}"#,
            &[("openai_responses_format.stream", "true")],
        )
        .await;
        let FilterAction::Reject(rejection) = action else {
            panic!("expected rejection");
        };
        assert_eq!(rejection.status, 400);
        assert_eq!(
            rejection.headers.iter().find(|(name, _)| name == "content-type"),
            Some(&("content-type".to_owned(), "application/json".to_owned()))
        );
        let body: serde_json::Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["code"], "mutually_exclusive_parameters");
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
    async fn streaming_background_is_rejected() {
        let action = run_filter_raw(
            r#"{"input": "test"}"#,
            &[
                ("openai_responses_format.stream", "true"),
                ("openai_responses_format.background", "true"),
            ],
        )
        .await;
        assert_background_rejection(action);
    }

    #[tokio::test]
    async fn non_streaming_background_is_rejected() {
        let action = run_filter_raw(
            r#"{"input": "test"}"#,
            &[
                ("openai_responses_format.background", "true"),
                ("openai_responses_format.store", "false"),
            ],
        )
        .await;
        assert_background_rejection(action);
    }

    #[tokio::test]
    #[expect(
        clippy::too_many_lines,
        reason = "the phase assertion keeps both lifecycle checkpoints visible"
    )]
    async fn automatic_phase_selection_runs_validator_once() {
        for provider_aware in [false, true] {
            let pipeline = validator_pipeline(provider_aware, "vllm");
            let req = Box::leak(Box::new(crate::test_utils::make_request(
                http::Method::POST,
                "/v1/responses",
            )));
            let mut ctx = crate::test_utils::make_filter_context(req);
            let mut body = Some(Bytes::from_static(br#"{"model":"test","input":"hello"}"#));

            let body_action = pipeline
                .execute_http_request_body(&mut ctx, &mut body, true)
                .await
                .unwrap();
            assert!(
                matches!(body_action, FilterAction::Release | FilterAction::Continue),
                "body pre-read should complete"
            );
            let pre_read_id = ctx.get_metadata("responses.response_id").map(str::to_owned);
            assert_eq!(
                pre_read_id.is_some(),
                !provider_aware,
                "only an unconditioned validator should run during pre-read"
            );
            ctx.buffered_request_body = Some(Bytes::from_static(br#"{"model":"test","input":"hello"}"#));

            let request_action = pipeline.execute_http_request(&mut ctx).await.unwrap();
            assert!(
                matches!(request_action, FilterAction::Continue),
                "request phase should complete: {request_action:?}"
            );
            let final_id = ctx.get_metadata("responses.response_id").map(str::to_owned);
            assert!(final_id.is_some(), "the validator should execute in one body phase");
            if let Some(pre_read_id) = pre_read_id {
                assert_eq!(
                    final_id.as_deref(),
                    Some(pre_read_id.as_str()),
                    "the unconditioned validator must not execute again at the binding barrier"
                );
            }
        }
    }

    #[tokio::test]
    async fn streaming_request_rejection_uses_json_content_type() {
        let action = run_filter_raw("not valid json", &[("openai_responses_format.stream", "true")]).await;
        if let FilterAction::Reject(rejection) = action {
            let has_content_type = rejection
                .headers
                .iter()
                .any(|(k, v)| k == "content-type" && v == "application/json");
            assert!(
                has_content_type,
                "a stream:true request that fails pre-stream should still use application/json"
            );
            let body = rejection.body.expect("rejection should carry a JSON body");
            let parsed: serde_json::Value = serde_json::from_slice(&body).expect("body should be JSON");
            assert!(
                parsed.get("error").is_some_and(serde_json::Value::is_object),
                "pre-stream rejection body should be a JSON error envelope, not an SSE event: {parsed}"
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
        let action = reject_invalid("line1\nline2");
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

    fn assert_background_rejection(action: FilterAction) {
        let FilterAction::Reject(rejection) = action else {
            panic!("background=true should be rejected");
        };
        assert_eq!(rejection.status, 400);
        let body: serde_json::Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["message"], "background mode is not supported");
    }

    #[expect(
        clippy::too_many_lines,
        reason = "the inline pipeline fixture is clearer as one complete configuration"
    )]
    fn validator_pipeline(provider_aware: bool, provider: &str) -> FilterPipeline {
        let conditions = if provider_aware {
            "  conditions:\n    - unless:\n        bound_upstream:\n          application_provider: openai\n"
        } else {
            ""
        };
        let yaml = format!(
            r#"
- filter: openai_responses_format
- filter: router
  routes:
    - path_prefix: "/"
      cluster: backend
- filter: openai_responses_validate
{conditions}- filter: load_balancer
  cluster_source: bound_upstream
  clusters:
    - name: backend
      http:
        application_provider: "{provider}"
      endpoints:
        - "127.0.0.1:9"
"#
        );
        let mut registry = FilterRegistry::with_builtins();
        praxis_filter::register_filters!(
            @register registry,
            http "openai_responses_format" => crate::openai::ResponsesFormatFilter::from_config
        );
        praxis_filter::register_filters!(
            @register registry,
            http "openai_responses_validate" => OpenaiResponsesValidateFilter::from_config
        );
        let mut entries: Vec<FilterEntry> = serde_yaml::from_str(&yaml).unwrap();
        let mut pipeline = FilterPipeline::build(&mut entries, &registry).unwrap();
        pipeline.set_allow_private_upstreams(true);
        pipeline
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
