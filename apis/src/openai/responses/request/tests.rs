// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Unit tests for the consolidated Responses create request processor.

#![expect(clippy::unwrap_used, clippy::expect_used, reason = "tests")]

use bytes::Bytes;
use praxis_filter::{
    FilterAction, HttpFilter, HttpFilterContext, Request,
    body::{BodyAccess, BodyMode},
};
use serde_json::json;

use super::*;
use crate::{
    openai::responses::state::ResponsesState,
    test_utils::{make_filter_context, make_request},
};

/// Build the filter from YAML, defaulting to an empty mapping.
fn filter(yaml: &str) -> Box<dyn HttpFilter> {
    let value: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
    OpenaiResponsesRequestFilter::from_config(&value).unwrap()
}

/// Build a filter with default configuration.
fn default_filter() -> Box<dyn HttpFilter> {
    filter("{}")
}

/// Build a `POST /v1/responses` request.
fn create_request() -> Request {
    make_request(http::Method::POST, "/v1/responses")
}

/// Drive one body through the filter and return the action.
async fn run(filter: &dyn HttpFilter, request: &Request, body: &serde_json::Value) -> FilterAction {
    let mut ctx = make_filter_context(request);
    let mut bytes = Some(Bytes::from(serde_json::to_vec(body).unwrap()));
    filter.on_request_body(&mut ctx, &mut bytes, true).await.unwrap()
}

/// Assert that managed prompt templates fail with the canonical error response.
fn assert_prompt_template_rejection(action: FilterAction) {
    assert!(
        matches!(&action, FilterAction::Reject(_)),
        "managed prompt templates must be rejected before upstream contact"
    );
    if let FilterAction::Reject(rejection) = action {
        assert_eq!(
            rejection.status, 400,
            "prompt template rejection must be a client error"
        );
        let body: serde_json::Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();
        assert_eq!(
            body.pointer("/error/type").and_then(serde_json::Value::as_str),
            Some("invalid_request_error"),
            "prompt template rejection must use the OpenAI invalid-request error type"
        );
        assert_eq!(
            body.pointer("/error/message").and_then(serde_json::Value::as_str),
            Some("prompt templates are supported only for OpenAI-owned upstreams"),
            "prompt template rejection must explain the provider-binding requirement"
        );
    }
}

/// Drive one streaming create request and return its context.
async fn run_streaming_create<'a>(filter: &dyn HttpFilter, request: &'a Request) -> HttpFilterContext<'a> {
    let mut ctx = make_filter_context(request);
    let mut body = Some(Bytes::from(
        serde_json::to_vec(&json!({"model": "gpt-4.1", "input": "hi", "stream": true})).unwrap(),
    ));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(matches!(action, FilterAction::Release));
    ctx
}

#[tokio::test]
async fn a_create_request_publishes_classification_metadata() {
    let filter = default_filter();
    let request = create_request();
    let ctx = run_streaming_create(filter.as_ref(), &request).await;

    // Published under the namespace downstream filters already read.
    assert_eq!(
        ctx.filter_metadata
            .get("openai_responses_format.format")
            .map(String::as_str),
        Some("openai_responses")
    );
    assert_eq!(
        ctx.filter_metadata
            .get("openai_responses_format.stream")
            .map(String::as_str),
        Some("true")
    );
}

#[tokio::test]
async fn a_create_request_publishes_validated_facts_and_state_from_one_parse() {
    let filter = default_filter();
    let request = create_request();
    let ctx = run_streaming_create(filter.as_ref(), &request).await;

    // Derived from the same parse rather than round-tripped through metadata.
    assert_eq!(
        ctx.filter_metadata.get("responses.stream").map(String::as_str),
        Some("true")
    );
    assert_eq!(
        ctx.filter_metadata.get("responses.store").map(String::as_str),
        Some("true"),
        "store defaults to true per the OpenAI specification"
    );
    assert!(
        ctx.filter_metadata
            .get("responses.response_id")
            .is_some_and(|id| id.starts_with("resp_")),
        "a proxy-owned response ID is generated"
    );

    let state = ctx.extensions.get::<ResponsesState>().expect("state initialized");
    assert!(state.response_id.as_ref().is_some_and(|id| id.starts_with("resp_")));
}

/// Classification once moved `model` out of the parsed value, which a shared
/// parse would forward upstream as an empty string.
#[tokio::test]
async fn classification_leaves_the_body_intact_for_state() {
    let filter = default_filter();
    let request = create_request();
    let mut ctx = make_filter_context(&request);
    let mut body = Some(Bytes::from(
        serde_json::to_vec(&json!({"model": "gpt-4.1", "input": "hi"})).unwrap(),
    ));

    drop(filter.on_request_body(&mut ctx, &mut body, true).await.unwrap());

    let state = ctx.extensions.get::<ResponsesState>().expect("state initialized");
    assert_eq!(
        state.request_body.get("model").and_then(serde_json::Value::as_str),
        Some("gpt-4.1"),
        "state must retain the model the client sent"
    );
}

/// A valid create body may carry no discriminator at all. The endpoint is
/// authoritative, so it must still publish as a Responses request.
#[tokio::test]
async fn a_model_only_create_is_classified_from_the_endpoint() {
    let filter = default_filter();
    let request = create_request();
    let mut ctx = make_filter_context(&request);
    let mut body = Some(Bytes::from(serde_json::to_vec(&json!({"model": "gpt-5"})).unwrap()));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(matches!(action, FilterAction::Release));
    assert_eq!(
        ctx.filter_metadata
            .get("openai_responses_format.format")
            .map(String::as_str),
        Some("openai_responses"),
        "body heuristics find no discriminator, but the create endpoint decides"
    );
    assert!(
        ctx.extensions.get::<ResponsesState>().is_some(),
        "downstream filters gate on the published format, so state must exist too"
    );
}

/// Background rejection keys off the published format, so a body without a
/// discriminator must not slip past it.
#[tokio::test]
async fn a_model_only_background_create_is_still_rejected() {
    let filter = default_filter();
    let request = create_request();
    let action = run(
        filter.as_ref(),
        &request,
        &json!({"model": "gpt-5", "background": true}),
    )
    .await;

    assert!(
        matches!(action, FilterAction::Reject(_)),
        "an undiscriminated create body must not bypass the background rejection"
    );
}

/// A body positively identified as another format keeps that identity.
#[tokio::test]
async fn a_positively_classified_body_is_not_relabelled() {
    let filter = default_filter();
    let request = create_request();
    let mut ctx = make_filter_context(&request);
    let mut body = Some(Bytes::from(
        serde_json::to_vec(&json!({"model": "gpt-5", "messages": [{"role": "user", "content": "hi"}]})).unwrap(),
    ));

    drop(filter.on_request_body(&mut ctx, &mut body, true).await.unwrap());

    assert_eq!(
        ctx.filter_metadata
            .get("openai_responses_format.format")
            .map(String::as_str),
        Some("openai_chat_completions"),
        "only unknown bodies are upgraded by endpoint authority"
    );
    assert!(
        ctx.extensions.get::<ResponsesState>().is_none(),
        "another protocol's body must not gain Responses state, or state-driven \
         filters would pick up traffic the validation stage used to release"
    );
    assert!(
        !ctx.filter_metadata.contains_key("responses.response_id"),
        "no proxy-owned Responses identifiers for another protocol's body"
    );
}

#[test]
fn unsafe_header_targets_are_rejected_at_construction() {
    for yaml in [
        "headers:\n  format: authorization\n",
        "headers:\n  model: x-api-key\n",
        "headers:\n  stream: x-praxis-route\n",
        "headers:\n  mode: content-length\n",
    ] {
        let value: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
        assert!(
            OpenaiResponsesRequestFilter::from_config(&value).is_err(),
            "configuration should be rejected:\n{yaml}"
        );
    }
}

#[test]
fn dedicated_default_header_targets_are_accepted() {
    let value: serde_yaml::Value = serde_yaml::from_str(
        "headers:\n  format: x-praxis-ai-format\n  model: x-praxis-ai-model\n  stream: x-praxis-ai-stream\n",
    )
    .unwrap();
    assert!(OpenaiResponsesRequestFilter::from_config(&value).is_ok());
}

/// Filter results must be published under this filter's own name. A branch
/// condition has to name the filter it is attached to, and every other
/// filter's results are cleared before it is evaluated, so publishing under
/// the replaced classifier's name left every `on_result` branch unmatched —
/// a stateful request silently took the stateless path.
#[tokio::test]
async fn filter_results_are_published_under_this_filter() {
    let filter = default_filter();
    let request = create_request();
    let mut ctx = make_filter_context(&request);
    let mut body = Some(Bytes::from(
        serde_json::to_vec(&json!({
            "model": "gpt-4.1",
            "input": "hi",
            "previous_response_id": "resp_1"
        }))
        .unwrap(),
    ));

    drop(filter.on_request_body(&mut ctx, &mut body, true).await.unwrap());

    let results = ctx
        .filter_results
        .get("openai_responses_request")
        .expect("results belong to the filter that published them");
    assert_eq!(results.get("format"), Some("openai_responses"));
    assert_eq!(
        results.get("mode"),
        Some("stateful"),
        "the routing fact a branch condition matches on"
    );
    assert!(
        !ctx.filter_results.contains_key("openai_responses_format"),
        "nothing is published under the replaced filter's name"
    );
}

/// A passthrough chain consumes no state, so it can opt out of building it.
/// Classification is still published, since routing depends on it.
#[tokio::test]
async fn initialize_state_false_classifies_without_building_state() {
    let filter = filter("initialize_state: false\n");
    let request = create_request();
    let mut ctx = make_filter_context(&request);
    let mut body = Some(Bytes::from(
        serde_json::to_vec(&json!({"model": "gpt-4.1", "input": "hi", "stream": true})).unwrap(),
    ));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(matches!(action, FilterAction::Release));
    assert_eq!(
        ctx.filter_metadata
            .get("openai_responses_format.format")
            .map(String::as_str),
        Some("openai_responses"),
        "classification is still published so routing is unaffected"
    );
    assert_eq!(
        ctx.filter_metadata
            .get("openai_responses_format.stream")
            .map(String::as_str),
        Some("true"),
        "promoted routing facts are still published"
    );
    assert!(
        ctx.extensions.get::<ResponsesState>().is_none(),
        "no state is built when the chain opted out"
    );
    assert!(
        !ctx.filter_metadata.contains_key("responses.response_id"),
        "no identifier is generated when the chain opted out"
    );
}

/// The default is unchanged, so an existing chain keeps its state.
#[tokio::test]
async fn state_is_initialized_by_default() {
    let filter = default_filter();
    let request = create_request();
    let mut ctx = make_filter_context(&request);
    let mut body = Some(Bytes::from(
        serde_json::to_vec(&json!({"model": "gpt-4.1", "input": "hi"})).unwrap(),
    ));

    drop(filter.on_request_body(&mut ctx, &mut body, true).await.unwrap());

    assert!(
        ctx.extensions.get::<ResponsesState>().is_some(),
        "omitting initialize_state must keep the previous behaviour"
    );
}

#[tokio::test]
async fn bodyless_responses_operations_are_left_alone() {
    // The registry declares these as carrying no body, so there is nothing to
    // parse and nothing to publish.
    for (method, path) in [
        (http::Method::GET, "/v1/responses/resp_123"),
        (http::Method::DELETE, "/v1/responses/resp_123"),
        (http::Method::POST, "/v1/responses/resp_123/cancel"),
        (http::Method::GET, "/v1/responses/resp_123/input_items"),
    ] {
        let filter = default_filter();
        let request = make_request(method.clone(), path);
        let mut ctx = make_filter_context(&request);
        let mut body = None;

        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

        assert!(matches!(action, FilterAction::Release), "{method} {path}");
        assert!(
            ctx.extensions.get::<ResponsesState>().is_none(),
            "{method} {path} must not initialize state"
        );
        assert!(
            !ctx.filter_metadata.contains_key("openai_responses_format.format"),
            "{method} {path} publishes nothing; identity comes from the request head"
        );
    }
}

/// Compact and input-token-count declare their body optional, so a request
/// without one is complete and must not be treated as an invalid body.
#[tokio::test]
async fn an_absent_optional_body_is_not_an_invalid_body() {
    for path in ["/v1/responses/compact", "/v1/responses/input_tokens"] {
        // `reject` is the strictest setting; even there an absent optional body
        // is legitimate and must pass through.
        let filter = filter("on_invalid: reject\n");
        let request = make_request(http::Method::POST, path);
        let mut ctx = make_filter_context(&request);
        let mut body = None;

        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

        assert!(matches!(action, FilterAction::Release), "{path} should forward");
        assert_eq!(
            ctx.filter_metadata
                .get("openai_responses_format.format")
                .map(String::as_str),
            Some("openai_responses"),
            "{path} still publishes its identity so header routing can see it"
        );
        assert!(
            ctx.extensions.get::<ResponsesState>().is_none(),
            "{path} without a body initializes no state"
        );
    }
}

/// Create declares its body required, so an absent one is still an invalid
/// body and still follows `on_invalid`.
#[tokio::test]
async fn an_absent_required_body_still_follows_on_invalid() {
    let filter = filter("on_invalid: reject\n");
    let request = create_request();
    let mut ctx = make_filter_context(&request);
    let mut body = None;

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(
        matches!(action, FilterAction::Reject(_)),
        "create requires a body, so an absent one is rejected"
    );
}

/// A malformed body is an error whether or not the operation required one.
#[tokio::test]
async fn a_malformed_optional_body_still_follows_on_invalid() {
    let filter = filter("on_invalid: reject\n");
    let request = make_request(http::Method::POST, "/v1/responses/compact");
    let mut ctx = make_filter_context(&request);
    let mut body = Some(Bytes::from_static(b"{not json"));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(
        matches!(action, FilterAction::Reject(_)),
        "an optional body that was supplied but is malformed is still invalid"
    );
}

/// Compact and input-token-count declare a request body in the registry, so
/// they are parsed and published like create rather than released untouched.
#[tokio::test]
async fn other_body_bearing_responses_operations_are_processed() {
    for path in ["/v1/responses/compact", "/v1/responses/input_tokens"] {
        let filter = default_filter();
        let request = make_request(http::Method::POST, path);
        let mut ctx = make_filter_context(&request);
        let mut body = Some(Bytes::from(
            serde_json::to_vec(&json!({"model": "gpt-4.1", "previous_response_id": "resp_1"})).unwrap(),
        ));

        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

        assert!(matches!(action, FilterAction::Release), "{path}");
        assert_eq!(
            ctx.filter_metadata
                .get("openai_responses_format.format")
                .map(String::as_str),
            Some("openai_responses"),
            "{path} should publish its classification"
        );
        assert!(
            ctx.extensions.get::<ResponsesState>().is_none(),
            "{path} is not creating a response, so it must not get create state — \
             the agentic loop reads the MCP approval state that carries"
        );
    }
}

#[tokio::test]
async fn a_non_responses_path_is_left_alone() {
    let filter = default_filter();
    let request = make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = make_filter_context(&request);
    let mut body = Some(Bytes::from(
        serde_json::to_vec(&json!({"model": "gpt-4.1", "messages": []})).unwrap(),
    ));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(matches!(action, FilterAction::Release));
    assert!(
        ctx.extensions.get::<ResponsesState>().is_none(),
        "a Chat Completions body must not enter Responses processing"
    );
}

#[tokio::test]
async fn conflicting_history_selectors_are_rejected() {
    let filter = default_filter();
    let request = create_request();
    let action = run(
        filter.as_ref(),
        &request,
        &json!({"model": "m", "input": "hi", "previous_response_id": "resp_1", "conversation": "conv_1"}),
    )
    .await;

    assert!(
        matches!(action, FilterAction::Reject(_)),
        "previous_response_id and conversation are mutually exclusive"
    );
}

#[tokio::test]
async fn background_mode_is_rejected_before_upstream_contact() {
    let filter = default_filter();
    let request = create_request();
    let action = run(
        filter.as_ref(),
        &request,
        &json!({"model": "gpt-4.1", "input": "hi", "background": true}),
    )
    .await;

    assert!(
        matches!(action, FilterAction::Reject(_)),
        "Praxis does not implement the asynchronous Responses lifecycle"
    );
}

#[tokio::test]
async fn prompt_template_is_rejected_before_state_initialization() {
    let filter = default_filter();
    let request = create_request();
    let mut ctx = make_filter_context(&request);
    let mut body = Some(Bytes::from(
        serde_json::to_vec(&json!({
            "model": "gpt-4.1",
            "prompt": {"id": "pmpt_123", "variables": {"name": "Ada"}}
        }))
        .unwrap(),
    ));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert_prompt_template_rejection(action);
    assert!(
        ctx.extensions.get::<ResponsesState>().is_none(),
        "rejected prompt templates must not initialize gateway-owned state"
    );
}

#[tokio::test]
async fn null_prompt_is_allowed() {
    let filter = default_filter();
    let request = create_request();
    let mut ctx = make_filter_context(&request);
    let mut body = Some(Bytes::from(
        serde_json::to_vec(&json!({"model": "gpt-4.1", "input": "hi", "prompt": null})).unwrap(),
    ));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(
        matches!(action, FilterAction::Release),
        "a null prompt must not trigger prompt-template rejection"
    );
    assert!(
        ctx.extensions.get::<ResponsesState>().is_some(),
        "a null prompt is semantically absent and must pass validation"
    );
}

#[tokio::test]
async fn an_unclassifiable_body_follows_on_invalid_continue() {
    // The default is `continue`. The classifier this replaces forwarded such a
    // body and still published its format, so chains that route on those keys
    // keep working.
    for (label, body) in [
        ("missing", None),
        ("malformed", Some(Bytes::from_static(b"{not json"))),
        ("non-object", Some(Bytes::from_static(b"\"just a string\""))),
    ] {
        let filter = default_filter();
        let request = create_request();
        let mut ctx = make_filter_context(&request);
        let mut body = body;

        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

        assert!(matches!(action, FilterAction::Release), "{label} body should forward");
        let published = ctx
            .filter_metadata
            .get("openai_responses_format.format")
            .map(String::as_str);
        assert!(
            published == Some("non_json") || published == Some("invalid_json"),
            "{label} body should publish its format, got {published:?}"
        );
        assert!(
            ctx.extensions.get::<ResponsesState>().is_none(),
            "{label} body must not initialize Responses state"
        );
    }
}

#[tokio::test]
async fn an_unclassifiable_body_follows_on_invalid_reject() {
    for (label, body) in [
        ("missing", None),
        ("malformed", Some(Bytes::from_static(b"{not json"))),
        ("non-object", Some(Bytes::from_static(b"\"just a string\""))),
    ] {
        let filter = filter("on_invalid: reject\n");
        let request = create_request();
        let mut ctx = make_filter_context(&request);
        let mut body = body;

        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

        assert!(matches!(action, FilterAction::Reject(_)), "{label} body should reject");
    }
}

#[tokio::test]
async fn store_and_background_defaults_follow_the_specification() {
    let filter = default_filter();
    let request = create_request();
    let mut ctx = make_filter_context(&request);
    let mut body = Some(Bytes::from(
        serde_json::to_vec(&json!({"model": "gpt-4.1", "input": "hi"})).unwrap(),
    ));

    drop(filter.on_request_body(&mut ctx, &mut body, true).await.unwrap());

    assert_eq!(
        ctx.filter_metadata.get("responses.store").map(String::as_str),
        Some("true")
    );
    assert_eq!(
        ctx.filter_metadata.get("responses.background").map(String::as_str),
        Some("false")
    );
    assert_eq!(
        ctx.filter_metadata.get("responses.stream").map(String::as_str),
        Some("false")
    );
}

#[test]
fn the_filter_declares_bounded_buffering() {
    let filter = default_filter();
    assert_eq!(filter.request_body_access(), BodyAccess::ReadOnly);
    assert!(matches!(
        filter.request_body_mode(),
        BodyMode::StreamBuffer { max_bytes: Some(_) }
    ));
}

// -----------------------------------------------------------------------------
// Bound-upstream phase and response teardown
// -----------------------------------------------------------------------------

#[test]
fn declares_both_request_body_phases() {
    let filter = default_filter();
    assert_eq!(filter.request_body_access(), BodyAccess::ReadOnly);
    assert_eq!(
        filter.bound_upstream_request_body_access(),
        BodyAccess::ReadOnly,
        "a chain must be able to defer this filter until a provider is bound"
    );
}

#[test]
fn declares_a_streamed_response_body() {
    let filter = default_filter();
    assert_eq!(filter.response_body_access(), BodyAccess::ReadOnly);
    assert!(
        matches!(filter.response_body_mode(), BodyMode::Stream),
        "the response path is teardown only and must not buffer"
    );
}

/// The bound-upstream phase initializes the same state as the pre-read phase.
///
/// Core schedules one hook or the other, never both, so the two must agree:
/// a chain that defers this filter behind a `bound_upstream` condition has to
/// end up with the same `ResponsesState` a pre-read chain would produce.
#[tokio::test]
async fn the_bound_upstream_phase_initializes_state_like_the_pre_read_phase() {
    let body = json!({"model": "gpt-4.1", "input": "Hello"});

    let filter = default_filter();
    let request = create_request();
    let mut pre_read_ctx = make_filter_context(&request);
    let mut pre_read_bytes = Some(Bytes::from(serde_json::to_vec(&body).unwrap()));
    let pre_read_action = filter
        .on_request_body(&mut pre_read_ctx, &mut pre_read_bytes, true)
        .await
        .unwrap();
    assert!(matches!(pre_read_action, FilterAction::Release));

    let mut bound_ctx = make_filter_context(&request);
    let mut bound_bytes = Some(Bytes::from(serde_json::to_vec(&body).unwrap()));
    let outcome = filter
        .on_bound_upstream_request_body(&mut bound_ctx, &mut bound_bytes)
        .await
        .unwrap();
    assert!(matches!(outcome, BoundUpstreamBodyOutcome::Continue));

    let pre_read_state = pre_read_ctx.extensions.get::<ResponsesState>().unwrap();
    let bound_state = bound_ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(
        pre_read_state.input, bound_state.input,
        "both phases must derive state from the same parse"
    );
    assert!(
        bound_state
            .response_id
            .as_ref()
            .is_some_and(|id| id.starts_with("resp_")),
        "the bound-upstream phase must mint a proxy-owned response identifier"
    );
}

/// A managed `background: true` create is still rejected after binding.
#[tokio::test]
async fn the_bound_upstream_phase_still_rejects_managed_background() {
    let filter = default_filter();
    let request = create_request();
    let mut ctx = make_filter_context(&request);
    let mut bytes = Some(Bytes::from(
        serde_json::to_vec(&json!({"model": "gpt-4.1", "input": "Hello", "background": true})).unwrap(),
    ));

    let outcome = filter
        .on_bound_upstream_request_body(&mut ctx, &mut bytes)
        .await
        .unwrap();

    assert!(
        matches!(outcome, BoundUpstreamBodyOutcome::Reject(_)),
        "managed background mode is unsupported in either phase"
    );
    assert!(
        ctx.extensions.get::<ResponsesState>().is_none(),
        "a rejected create must not leave Responses state behind"
    );
}

/// A bodyless operation passes through the bound-upstream phase untouched.
#[tokio::test]
async fn the_bound_upstream_phase_releases_a_bodyless_operation() {
    let filter = default_filter();
    let request = make_request(http::Method::GET, "/v1/responses/resp_123");
    let mut ctx = make_filter_context(&request);
    let mut bytes = None;

    let outcome = filter
        .on_bound_upstream_request_body(&mut ctx, &mut bytes)
        .await
        .unwrap();

    assert!(matches!(outcome, BoundUpstreamBodyOutcome::Continue));
    assert!(
        ctx.extensions.get::<ResponsesState>().is_none(),
        "a fetch carries no body to process and must gain no state"
    );
}

#[cfg(feature = "openai-mcp-tools")]
#[tokio::test]
async fn response_teardown_drains_the_mcp_session_pool_at_eos() {
    let filter = default_filter();
    let request = create_request();
    let mut ctx = make_filter_context(&request);
    ctx.extensions.insert(crate::mcp_client::McpSessionPool::new());

    let action = filter.on_response_body(&mut ctx, &mut None, true).unwrap();

    assert!(matches!(action, FilterAction::Continue));
    assert!(
        ctx.extensions.get::<crate::mcp_client::McpSessionPool>().is_none(),
        "outer response EOS must not leave live session ownership in request extensions"
    );
}

#[cfg(feature = "openai-mcp-tools")]
#[tokio::test]
async fn response_teardown_keeps_the_pool_before_eos() {
    let filter = default_filter();
    let request = create_request();
    let mut ctx = make_filter_context(&request);
    ctx.extensions.insert(crate::mcp_client::McpSessionPool::new());

    let action = filter.on_response_body(&mut ctx, &mut None, false).unwrap();

    assert!(matches!(action, FilterAction::Continue));
    assert!(
        ctx.extensions.get::<crate::mcp_client::McpSessionPool>().is_some(),
        "streamed chunks must retain the pool for later agentic rounds"
    );
}

/// A managed create with a conversation publishes what append-back needs.
///
/// `capture_validated_append_owner` runs right after the canonical conversation
/// ID is published, and it only captures when these three facts hold. The
/// captured owner itself is private to the Conversations filter, so this asserts
/// the preconditions this filter owns; that the owner is actually captured and
/// the turn persists under it is covered by the integration suite.
#[cfg(feature = "openai-conversations")]
#[tokio::test]
async fn a_managed_create_with_a_conversation_publishes_the_append_back_facts() {
    let filter = default_filter();
    let request = create_request();
    let mut ctx = make_filter_context(&request);
    let mut bytes = Some(Bytes::from(
        serde_json::to_vec(&json!({
            "model": "gpt-4.1",
            "input": "Hello",
            "conversation": "conv_abc123",
        }))
        .unwrap(),
    ));

    let action = filter.on_request_body(&mut ctx, &mut bytes, true).await.unwrap();
    assert!(matches!(action, FilterAction::Release));

    assert_eq!(
        ctx.get_metadata("openai_responses_format.has_conversation"),
        Some("true"),
        "append-back is armed by the conversation selector"
    );
    assert_eq!(
        ctx.get_metadata("responses.conversation_id"),
        Some("conv_abc123"),
        "the canonical conversation ID must be published before the owner is captured"
    );
    assert_ne!(
        ctx.get_metadata("openai_responses_format.background"),
        Some("true"),
        "a background create does not arm local append-back"
    );
}

/// A create without a conversation does not arm append-back.
#[cfg(feature = "openai-conversations")]
#[tokio::test]
async fn a_create_without_a_conversation_does_not_arm_append_back() {
    let filter = default_filter();
    let request = create_request();
    let mut ctx = make_filter_context(&request);
    let mut bytes = Some(Bytes::from(
        serde_json::to_vec(&json!({"model": "gpt-4.1", "input": "Hello"})).unwrap(),
    ));

    drop(filter.on_request_body(&mut ctx, &mut bytes, true).await.unwrap());

    assert_ne!(
        ctx.get_metadata("openai_responses_format.has_conversation"),
        Some("true"),
        "no conversation selector means no local append-back"
    );
}
