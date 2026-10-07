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
            Some(
                "prompt templates are supported only for OpenAI-owned upstreams; send prompt content via input (OpenAI deprecated reusable prompts)"
            ),
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
            .get("openai_responses_request.format")
            .map(String::as_str),
        Some("openai_responses")
    );
    assert_eq!(
        ctx.filter_metadata
            .get("openai_responses_request.stream")
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
            .get("openai_responses_request.format")
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

/// The matched operation is authoritative over body shape: a Chat-Completions-
/// shaped body on `POST /v1/responses` is still a Responses request.
#[tokio::test]
async fn a_matched_operation_overrides_body_shape() {
    let filter = default_filter();
    let request = create_request();
    let mut ctx = make_filter_context(&request);
    let mut body = Some(Bytes::from(
        serde_json::to_vec(&json!({"model": "gpt-5", "messages": [{"role": "user", "content": "hi"}]})).unwrap(),
    ));

    drop(filter.on_request_body(&mut ctx, &mut body, true).await.unwrap());

    assert_eq!(
        ctx.filter_metadata
            .get("openai_responses_request.format")
            .map(String::as_str),
        Some("openai_responses"),
        "the endpoint identity is authoritative; JSON shape cannot override it"
    );
    assert!(
        ctx.extensions.get::<ResponsesState>().is_some(),
        "a create on the Responses endpoint builds state regardless of body shape"
    );
    assert!(
        ctx.filter_metadata.contains_key("responses.response_id"),
        "a Responses create mints its proxy-owned identifier"
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
            .get("openai_responses_request.format")
            .map(String::as_str),
        Some("openai_responses"),
        "classification is still published so routing is unaffected"
    );
    assert_eq!(
        ctx.filter_metadata
            .get("openai_responses_request.stream")
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

/// A facts publisher retains no parse unless the chain opts in. Without a
/// managed owner to consume it, a cached parse would sit in request extensions
/// alongside the original body for the whole forward — a request-sized copy
/// nothing reads.
#[tokio::test]
async fn a_facts_publisher_retains_no_parse_by_default() {
    let filter = filter("initialize_state: false\n");
    let request = create_request();
    let mut ctx = make_filter_context(&request);
    let mut body = Some(Bytes::from(
        serde_json::to_vec(&json!({"model": "gpt-4.1", "input": "hi"})).unwrap(),
    ));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(matches!(action, FilterAction::Release));
    assert_eq!(
        ctx.filter_metadata
            .get("openai_responses_request.format")
            .map(String::as_str),
        Some("openai_responses"),
        "routing facts are still published without the opt-in"
    );
    assert!(
        ctx.extensions.get::<CachedRequestParse>().is_none(),
        "no parse is retained when cache_parse_for_owner is off"
    );
}

/// The opt-in makes the facts publisher cache its parse for a managed owner.
#[tokio::test]
async fn cache_parse_for_owner_retains_the_parse() {
    let filter = filter("initialize_state: false\ncache_parse_for_owner: true\n");
    let request = create_request();
    let mut ctx = make_filter_context(&request);
    let mut body = Some(Bytes::from(
        serde_json::to_vec(&json!({"model": "gpt-4.1", "input": "hi"})).unwrap(),
    ));

    drop(filter.on_request_body(&mut ctx, &mut body, true).await.unwrap());

    assert!(
        ctx.extensions.get::<CachedRequestParse>().is_some(),
        "the opt-in hands the parse to a later managed pass"
    );
    assert!(
        ctx.extensions.get::<ResponsesState>().is_none(),
        "a facts publisher still mints no state even when it caches"
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
async fn bodyless_responses_operations_publish_identity_without_state() {
    // The registry declares these as carrying no body, so there is nothing to
    // parse. The endpoint is still authoritative that the request is Responses,
    // so the format fact is published for header-based routing — a fetch or
    // delete reaches the managed store exactly as it did under the former
    // classifier — while no response-creating state is minted.
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
        assert_eq!(
            ctx.filter_metadata
                .get("openai_responses_request.format")
                .map(String::as_str),
            Some("openai_responses"),
            "{method} {path} still publishes its identity so header routing can see it"
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
                .get("openai_responses_request.format")
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
                .get("openai_responses_request.format")
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
            .get("openai_responses_request.format")
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

/// A managed pass reuses the pre-routing parse rather than deserializing twice.
///
/// A two-entry chain publishes routing facts before binding and initializes
/// state after it. The create body must be parsed exactly once across both
/// passes, so the body bytes are corrupted between them: if the managed pass
/// re-parsed, it would fail to classify the garbage and release without state;
/// reusing the cached parse, it initializes state from the original request.
#[tokio::test]
#[expect(
    clippy::too_many_lines,
    reason = "sequential two-phase setup and cross-pass assertions"
)]
async fn the_managed_pass_reuses_the_pre_routing_parse() {
    let body = json!({"model": "gpt-4.1", "input": "deserialize once"});

    // The pre-routing publisher opts in to the handoff, so its one parse carries
    // to the managed pass rather than being dropped and re-parsed after binding.
    let facts = filter("initialize_state: false\ncache_parse_for_owner: true\n");
    let managed = default_filter();
    let request = create_request();
    let mut ctx = make_filter_context(&request);

    let mut bytes = Some(Bytes::from(serde_json::to_vec(&body).unwrap()));
    let facts_action = facts.on_request_body(&mut ctx, &mut bytes, true).await.unwrap();
    assert!(matches!(facts_action, FilterAction::Release));
    assert!(
        ctx.extensions.get::<CachedRequestParse>().is_some(),
        "the pre-routing pass must cache its parse for the managed pass"
    );
    assert!(
        ctx.extensions.get::<ResponsesState>().is_none(),
        "the pre-routing pass mints no state"
    );

    // Corrupt the body so any re-parse in the managed pass would fail.
    let mut corrupted = Some(Bytes::from_static(b"not json {{"));
    let outcome = managed
        .on_bound_upstream_request_body(&mut ctx, &mut corrupted)
        .await
        .unwrap();
    assert!(matches!(outcome, BoundUpstreamBodyOutcome::Continue));

    assert!(
        ctx.extensions.get::<CachedRequestParse>().is_none(),
        "the managed pass must consume the cached parse"
    );
    let state = ctx
        .extensions
        .get::<ResponsesState>()
        .expect("the managed pass initializes state from the cached parse");
    assert_eq!(
        state.request_body.pointer("/input").and_then(serde_json::Value::as_str),
        Some("deserialize once"),
        "state must come from the original parse, not the corrupted bytes"
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
        ctx.get_metadata("openai_responses_request.has_conversation"),
        Some("true"),
        "append-back is armed by the conversation selector"
    );
    assert_eq!(
        ctx.get_metadata("responses.conversation_id"),
        Some("conv_abc123"),
        "the canonical conversation ID must be published before the owner is captured"
    );
    assert_ne!(
        ctx.get_metadata("openai_responses_request.background"),
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
        ctx.get_metadata("openai_responses_request.has_conversation"),
        Some("true"),
        "no conversation selector means no local append-back"
    );
}

// -----------------------------------------------------------------------------
// Routing Mode
// -----------------------------------------------------------------------------

/// Drive one create body through the default filter and return its context.
async fn run_ctx<'a>(config_yaml: &str, request: &'a Request, body: &serde_json::Value) -> HttpFilterContext<'a> {
    let filter = filter(config_yaml);
    let mut ctx = make_filter_context(request);
    let mut bytes = Some(Bytes::from(serde_json::to_vec(body).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut bytes, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Release),
        "classification releases the request"
    );
    ctx
}

/// Read the routing-mode filter result published under this filter.
fn mode_result<'a>(ctx: &'a HttpFilterContext<'_>) -> Option<&'a str> {
    ctx.filter_results
        .get("openai_responses_request")
        .and_then(|r| r.get("mode"))
}

/// Collect promoted request headers for assertion.
fn collect_headers<'a>(ctx: &'a HttpFilterContext<'_>) -> std::collections::HashMap<&'a str, &'a str> {
    ctx.extra_request_headers
        .iter()
        .map(|(k, v)| (k.as_ref(), v.as_str()))
        .collect()
}

/// Mode is a pre-routing fact, so these drive the fact publisher
/// (`initialize_state: false`) that computes it before the router runs. The
/// managed owner rejects provider-owned fields such as `prompt` that are also
/// stateful markers, which would mask the mode under test.
const FACTS_ONLY: &str = "initialize_state: false\n";

/// `store=false` with no other stateful marker is the one stateless case, and
/// it must read the same across the filter result, metadata, and header.
#[tokio::test]
async fn mode_is_stateless_only_when_store_false_and_no_marker() {
    let request = create_request();
    let ctx = run_ctx(FACTS_ONLY, &request, &json!({"input": "test", "store": false})).await;

    assert_eq!(mode_result(&ctx), Some("stateless"));
    assert_eq!(
        ctx.filter_metadata
            .get("openai_responses_request.mode")
            .map(String::as_str),
        Some("stateless")
    );
    assert_eq!(collect_headers(&ctx).get("x-praxis-responses-mode"), Some(&"stateless"));
}

/// Empty `tools` is not a stateful marker.
#[tokio::test]
async fn mode_is_stateless_when_store_false_and_tools_empty() {
    let request = create_request();
    let ctx = run_ctx(
        FACTS_ONLY,
        &request,
        &json!({"input": "test", "store": false, "tools": []}),
    )
    .await;

    assert_eq!(
        mode_result(&ctx),
        Some("stateless"),
        "an empty tools array must not force the stateful path"
    );
}

/// Omitted `store` defaults to stateful, as does an explicit `store=true`.
#[tokio::test]
async fn mode_is_stateful_when_store_is_default_or_true() {
    let request = create_request();

    let omitted = run_ctx(FACTS_ONLY, &request, &json!({"input": "test"})).await;
    assert_eq!(
        mode_result(&omitted),
        Some("stateful"),
        "omitted store defaults to true (stateful)"
    );

    let explicit = run_ctx(FACTS_ONLY, &request, &json!({"input": "test", "store": true})).await;
    assert_eq!(mode_result(&explicit), Some("stateful"));
}

/// Any stateful marker keeps the request stateful even with `store=false`.
#[tokio::test]
async fn mode_is_stateful_when_store_false_but_a_marker_is_present() {
    let request = create_request();
    for marker in [
        json!({"input": "test", "store": false, "previous_response_id": "resp_1"}),
        json!({"input": "test", "store": false, "tools": [{"type": "function"}]}),
        json!({"input": "test", "store": false, "conversation": {"id": "conv_1"}}),
        json!({"input": "test", "store": false, "prompt": {"id": "pmpt_123"}}),
    ] {
        let ctx = run_ctx(FACTS_ONLY, &request, &marker).await;
        assert_eq!(
            mode_result(&ctx),
            Some("stateful"),
            "a stateful marker overrides store=false: {marker}"
        );
    }
}

/// Mode is a Responses fact, and the matched operation is authoritative: a
/// Chat-Completions-shaped body on the create endpoint is classified Responses,
/// so it still receives a mode.
#[tokio::test]
async fn mode_is_present_for_any_body_on_the_responses_endpoint() {
    let request = create_request();
    let ctx = run_ctx(
        FACTS_ONLY,
        &request,
        &json!({"messages": [{"role": "user", "content": "Hi"}]}),
    )
    .await;

    assert_eq!(
        mode_result(&ctx),
        Some("stateful"),
        "endpoint authority makes this a Responses request, which defaults to stateful"
    );
    assert_eq!(
        ctx.filter_metadata
            .get("openai_responses_request.mode")
            .map(String::as_str),
        Some("stateful"),
        "mode metadata is published for the Responses operation"
    );
    assert_eq!(
        collect_headers(&ctx).get("x-praxis-responses-mode"),
        Some(&"stateful"),
        "the mode routing header is promoted"
    );
    assert!(
        ctx.extensions
            .get::<praxis_filter::ErrorResponseFormatterHandle>()
            .is_some(),
        "the OpenAI error formatter is installed for the Responses operation"
    );
}

/// A custom mode header replaces the dedicated default name.
#[tokio::test]
async fn mode_header_honours_a_custom_name() {
    let request = create_request();
    let ctx = run_ctx(
        "initialize_state: false\nheaders:\n  mode: x-custom-mode\n",
        &request,
        &json!({"input": "test", "store": false}),
    )
    .await;
    let headers = collect_headers(&ctx);

    assert_eq!(headers.get("x-custom-mode"), Some(&"stateless"));
    assert!(
        !headers.contains_key("x-praxis-responses-mode"),
        "the default mode header is not emitted when overridden"
    );
}

/// A null mode header suppresses the header but keeps the mode metadata.
#[tokio::test]
async fn mode_header_null_suppresses_only_the_header() {
    let request = create_request();
    let ctx = run_ctx(
        "initialize_state: false\nheaders:\n  mode: null\n",
        &request,
        &json!({"input": "test", "store": false}),
    )
    .await;

    assert!(
        !collect_headers(&ctx).contains_key("x-praxis-responses-mode"),
        "a null mode header suppresses emission"
    );
    assert_eq!(
        ctx.filter_metadata
            .get("openai_responses_request.mode")
            .map(String::as_str),
        Some("stateless"),
        "metadata is still written when the header is disabled"
    );
}

// -----------------------------------------------------------------------------
// Fact Promotion
// -----------------------------------------------------------------------------

/// Every configured header carries the body fact it was derived from.
#[tokio::test]
async fn model_and_stream_facts_are_promoted_to_headers() {
    let request = create_request();
    let ctx = run_ctx(
        "{}",
        &request,
        &json!({"model": "gpt-4.1", "input": "hi", "stream": true}),
    )
    .await;
    let headers = collect_headers(&ctx);

    assert_eq!(headers.get("x-praxis-ai-format"), Some(&"openai_responses"));
    assert_eq!(headers.get("x-praxis-ai-model"), Some(&"gpt-4.1"));
    assert_eq!(headers.get("x-praxis-ai-stream"), Some(&"true"));
}

/// An oversized model value is dropped from every promotion channel so it
/// cannot smuggle an unbounded value into a header, metadata, or a result.
#[tokio::test]
async fn an_oversized_model_is_not_promoted() {
    let request = create_request();
    let oversized = "a".repeat(9000);
    let ctx = run_ctx("{}", &request, &json!({"model": oversized, "input": "hi"})).await;

    assert!(
        !collect_headers(&ctx).contains_key("x-praxis-ai-model"),
        "an oversized model is not promoted to a header"
    );
    assert!(
        !ctx.filter_metadata.contains_key("openai_responses_request.model"),
        "an oversized model is not promoted to metadata"
    );
    assert_eq!(
        ctx.filter_results
            .get("openai_responses_request")
            .and_then(|r| r.get("model")),
        None,
        "an oversized model is not promoted to a filter result"
    );
}

/// A model value carrying control characters is not a valid header value and
/// must not be promoted.
#[tokio::test]
async fn a_control_char_model_is_not_promoted() {
    let request = create_request();
    let ctx = run_ctx("{}", &request, &json!({"model": "bad\nmodel", "input": "hi"})).await;

    assert!(
        !collect_headers(&ctx).contains_key("x-praxis-ai-model"),
        "a control-character model is not promoted to a header"
    );
    assert!(
        !ctx.filter_metadata.contains_key("openai_responses_request.model"),
        "a control-character model is not promoted to metadata"
    );
}

/// Facts the body omits are not invented on any channel.
#[tokio::test]
async fn omitted_facts_are_not_promoted() {
    let request = create_request();
    // `model` and `stream` omitted; only the format is known from the endpoint.
    let ctx = run_ctx("{}", &request, &json!({"input": "hi", "store": false})).await;
    let headers = collect_headers(&ctx);

    assert_eq!(headers.get("x-praxis-ai-format"), Some(&"openai_responses"));
    assert!(!headers.contains_key("x-praxis-ai-model"), "no model fact to promote");
    assert!(!headers.contains_key("x-praxis-ai-stream"), "no stream fact to promote");
    assert!(
        !ctx.filter_metadata.contains_key("openai_responses_request.model"),
        "no model metadata when the body omits it"
    );
}

/// Null header names suppress every promoted header while metadata and results
/// stay intact.
#[tokio::test]
async fn null_header_names_suppress_all_header_promotion() {
    let request = create_request();
    let ctx = run_ctx(
        "headers:\n  format: null\n  model: null\n  stream: null\n  mode: null\n",
        &request,
        &json!({"model": "gpt-4.1", "input": "hi", "stream": true}),
    )
    .await;

    assert!(
        collect_headers(&ctx).is_empty(),
        "every promotion header is suppressed when its name is null"
    );
    assert_eq!(
        ctx.filter_metadata
            .get("openai_responses_request.model")
            .map(String::as_str),
        Some("gpt-4.1"),
        "metadata is still published when headers are disabled"
    );
    assert_eq!(
        ctx.filter_results
            .get("openai_responses_request")
            .and_then(|r| r.get("model")),
        Some("gpt-4.1"),
        "filter results are still published when headers are disabled"
    );
}
