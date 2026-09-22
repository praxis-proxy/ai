// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Unit tests for the Responses proxy filter.

use std::fmt::Write as _;

use base64::Engine as _;
use bytes::Bytes;
use http::Method;
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterEntry, FilterPipeline, FilterRegistry, HttpFilter, SubRequestResponseMode,
};
use serde_json::json;

use super::super::state::ResponsesState;
use crate::test_utils::{make_filter_context, make_request};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn from_config_accepts_null() {
    let yaml = serde_yaml::Value::Null;
    let filter = super::ResponsesProxyFilter::from_config(&yaml).unwrap();
    assert_eq!(
        filter.name(),
        "openai_responses_proxy",
        "filter name should be openai_responses_proxy"
    );
}

#[test]
fn from_config_accepts_empty_mapping() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("{}").unwrap();
    let filter = super::ResponsesProxyFilter::from_config(&yaml).unwrap();
    assert_eq!(
        filter.name(),
        "openai_responses_proxy",
        "filter name should be openai_responses_proxy"
    );
}

#[test]
fn from_config_rejects_unknown_fields() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("unknown_field: true").unwrap();
    let result = super::ResponsesProxyFilter::from_config(&yaml);
    assert!(result.is_err(), "unknown fields should be rejected");
}

#[test]
fn body_access_is_read_write() {
    let filter = make_filter();
    assert_eq!(
        filter.request_body_access(),
        BodyAccess::ReadWrite,
        "openai_responses_proxy must declare ReadWrite to modify the body"
    );
}

#[test]
fn body_mode_is_stream_buffer() {
    let filter = make_filter();
    match filter.request_body_mode() {
        BodyMode::StreamBuffer { max_bytes } => {
            assert_eq!(
                max_bytes,
                Some(67_108_864),
                "StreamBuffer should default to the 64 MiB ceiling; the raw cap is governed by body_limits"
            );
        },
        other => panic!("openai_responses_proxy must use StreamBuffer, got {other:?}"),
    }
}

#[test]
fn always_advertises_streaming_capability() {
    assert!(
        make_filter().may_select_streaming_subrequest_response(),
        "openai_responses_proxy must always declare the Praxis streaming capability so \
         transport follows the effective request, without an operator opt-in"
    );
}

#[test]
fn from_config_rejects_removed_terminal_streaming_true() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("terminal_streaming: true").unwrap();
    let result = super::ResponsesProxyFilter::from_config(&yaml);
    assert!(
        result.is_err(),
        "the removed terminal_streaming flag must be rejected so operators migrate their config"
    );
}

#[test]
fn from_config_rejects_removed_terminal_streaming_false() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("terminal_streaming: false").unwrap();
    let result = super::ResponsesProxyFilter::from_config(&yaml);
    assert!(
        result.is_err(),
        "terminal_streaming is removed entirely; even the former default value must be rejected"
    );
}

#[tokio::test]
async fn selects_streaming_from_effective_passthrough_body() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    ctx.set_metadata("openai_responses_format.stream", "false");
    let mut body = Some(Bytes::from_static(
        br#"{"model":"gpt-4.1","input":"hello","stream":true}"#,
    ));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(
        matches!(action, FilterAction::Continue),
        "streaming passthrough should continue"
    );
    assert_eq!(
        ctx.subrequest_response_mode(),
        SubRequestResponseMode::Streaming,
        "effective provider body, not descriptive classifier metadata, must select transport"
    );
}

#[tokio::test]
async fn preserves_buffered_mode_when_stream_is_false_or_absent() {
    let filter = make_filter();
    for original in [
        br#"{"model":"gpt-4.1","input":"hello","stream":false}"#.as_slice(),
        br#"{"model":"gpt-4.1","input":"hello"}"#.as_slice(),
    ] {
        let req = make_request(Method::POST, "/v1/responses");
        let mut ctx = make_filter_context(&req);
        ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
        let mut body = Some(Bytes::copy_from_slice(original));

        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

        assert!(
            matches!(action, FilterAction::Continue),
            "non-streaming request should continue"
        );
        assert_eq!(
            ctx.subrequest_response_mode(),
            SubRequestResponseMode::Buffered,
            "non-streaming effective body must keep the buffered transport"
        );
    }
}

#[tokio::test]
async fn uses_rebuilt_state_body_not_client_intent_metadata() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    ctx.set_metadata("openai_responses_format.stream", "true");
    let mut state = ResponsesState::from_request_body(json!({
        "model": "gpt-4.1",
        "input": "hello",
        "stream": false
    }));
    state
        .messages
        .push(json!({"type":"function_call_output","call_id":"call_1","output":"done"}));
    ctx.extensions.insert(state);
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
    let mut body = Some(Bytes::from_static(
        br#"{"model":"gpt-4.1","input":"hello","stream":true}"#,
    ));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(
        matches!(action, FilterAction::Continue),
        "rebuilt buffered request should continue"
    );
    let outbound: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    assert_eq!(outbound["stream"], false);
    assert_eq!(ctx.subrequest_response_mode(), SubRequestResponseMode::Buffered);
}

#[tokio::test]
async fn selects_streaming_for_rebuilt_effective_body() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    ctx.set_metadata("openai_responses_format.stream", "false");
    let mut state = ResponsesState::from_request_body(json!({
        "model": "gpt-4.1",
        "input": "hello",
        "stream": true
    }));
    state
        .messages
        .push(json!({"type":"function_call_output","call_id":"call_1","output":"done"}));
    ctx.extensions.insert(state);
    let mut body = Some(Bytes::from_static(
        br#"{"model":"gpt-4.1","input":"hello","stream":false}"#,
    ));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(
        matches!(action, FilterAction::Continue),
        "rebuilt streaming request should continue"
    );
    let outbound: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    assert_eq!(outbound["stream"], true);
    assert_eq!(ctx.subrequest_response_mode(), SubRequestResponseMode::Streaming);
}

#[tokio::test]
async fn on_request_returns_continue() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "on_request should return Continue"
    );
}

#[tokio::test]
async fn passthrough_without_state() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let original = r#"{"model":"gpt-4o","input":"hello"}"#;
    let mut body = Some(Bytes::from(original));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "should continue without ResponsesState"
    );
    assert_eq!(
        body.as_deref(),
        Some(original.as_bytes()),
        "body should be unchanged when no state is present"
    );
}

#[tokio::test]
async fn prompt_template_is_rejected_without_selected_openai_upstream() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let original = Bytes::from_static(br#"{"model":"gpt-4.1","prompt":{"id":"pmpt_123","variables":{"name":"Ada"}}}"#);
    let mut body = Some(original.clone());

    let body_action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(body_action, FilterAction::Continue),
        "prompt detection should complete before upstream validation"
    );
    let action = filter.on_request(&mut ctx).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("a prompt template must fail closed for a backend without the capability");
    };
    assert_eq!(rejection.status, 400, "unsupported prompt must return HTTP 400");
    let error: serde_json::Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();
    assert_eq!(
        error["error"]["type"], "invalid_request_error",
        "unsupported prompt must use the Responses invalid-request error type"
    );
    assert_eq!(
        error["error"]["message"],
        "prompt templates are supported only when the selected upstream declares application_protocol: openai_responses and application_provider: openai",
        "unsupported prompt must explain the protocol and provider declaration requirement"
    );
    assert_eq!(
        body.as_deref(),
        Some(original.as_ref()),
        "rejection must not mutate the request"
    );
}

#[tokio::test]
async fn prompt_template_in_canonical_state_is_rejected_by_default() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "gpt-4.1",
        "prompt": {"id": "pmpt_123"}
    })));
    let mut body = Some(Bytes::from_static(br#"{"model":"gpt-4.1","input":"rewritten"}"#));

    let body_action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(body_action, FilterAction::Continue),
        "canonical prompt detection should complete before upstream validation"
    );
    let action = filter.on_request(&mut ctx).await.unwrap();

    assert!(
        matches!(&action, FilterAction::Reject(rejection) if rejection.status == 400),
        "canonical state must remain authoritative even when the current bytes omit prompt"
    );
}

#[tokio::test]
async fn null_prompt_is_allowed_by_default() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let original = Bytes::from_static(br#"{"model":"gpt-4.1","input":"hello","prompt":null}"#);
    let mut body = Some(original.clone());

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(
        matches!(action, FilterAction::Continue),
        "a null prompt should remain a valid passthrough request"
    );
    assert!(
        matches!(filter.on_request(&mut ctx).await.unwrap(), FilterAction::Continue),
        "a null prompt must not require an OpenAI upstream"
    );
    assert_eq!(
        body.as_deref(),
        Some(original.as_ref()),
        "null prompt passthrough must preserve request bytes"
    );
}

#[test]
fn prompt_probe_validates_deep_values_without_retaining_them() {
    let prompt_body = deeply_nested_request("prompt", true);
    assert!(
        super::raw_request_has_prompt(&prompt_body),
        "a valid non-null prompt nested 256 levels deep must be detected"
    );

    let unrelated_body = deeply_nested_request("metadata", true);
    assert!(
        !super::raw_request_has_prompt(&unrelated_body),
        "a valid request with an unrelated 256-level value must prove prompt absence"
    );
}

#[test]
fn prompt_probe_ignores_malformed_json() {
    // A body that is not valid JSON cannot carry a prompt that a strict
    // OpenAI-compatible backend would parse and honor. It is not attributed to
    // prompt templates; normal request validation rejects the malformed body.
    let truncated = deeply_nested_request("metadata", false);
    assert!(
        !super::raw_request_has_prompt(&truncated),
        "truncated JSON must not be reported as a prompt template"
    );

    assert!(
        !super::raw_request_has_prompt(b"{not-json"),
        "syntactically invalid JSON must not be reported as a prompt template"
    );

    assert!(
        !super::raw_request_has_prompt(br#"{"prompt":{"id":"x"} trailing garbage"#),
        "a prompt smuggled behind trailing garbage is not valid JSON and must not be attributed here"
    );
}

#[test]
fn prompt_probe_detects_prompt_beyond_serde_recursion_limit() {
    // serde_json's `IgnoredAny` skip is iterative, so a valid non-null prompt is
    // detected far past the recursion limit that a `Value` parse would hit.
    let mut body = br#"{"metadata":"#.to_vec();
    for _ in 0..5_000 {
        body.extend_from_slice(br#"{"nested":"#);
    }
    body.extend_from_slice(b"null");
    body.resize(body.len() + 5_000, b'}');
    body.extend_from_slice(br#","prompt":{"id":"pmpt_deep"}}"#);
    assert!(
        super::raw_request_has_prompt(&body),
        "a valid prompt after 5000 levels of unrelated nesting must still be detected"
    );
}

#[test]
fn prompt_probe_allocation_is_independent_of_prompt_payload_size() {
    let small_body = br#"{"model":"gpt-4.1","prompt":{"id":"pmpt_123","variables":{"file":"x"}}}"#;
    let small_allocations = allocation_counter::measure(|| {
        std::hint::black_box(super::raw_request_has_prompt(small_body));
    });
    let payload = "x".repeat(1024 * 1024);
    let body = format!(r#"{{"model":"gpt-4.1","prompt":{{"id":"pmpt_123","variables":{{"file":"{payload}"}}}}}}"#);
    assert!(
        super::raw_request_has_prompt(body.as_bytes()),
        "the warm-up probe must detect the large prompt object"
    );
    let mut detected = false;

    let allocations = allocation_counter::measure(|| {
        detected = std::hint::black_box(super::raw_request_has_prompt(body.as_bytes()));
    });

    assert!(detected, "the large prompt object must be detected");
    assert_eq!(
        allocations.count_total, small_allocations.count_total,
        "allocation count must not grow with prompt payload size: small={small_allocations:?}, large={allocations:?}"
    );
    assert_eq!(
        allocations.bytes_total, small_allocations.bytes_total,
        "allocated bytes must not grow with prompt payload size: small={small_allocations:?}, large={allocations:?}"
    );
    assert!(
        allocations.bytes_max <= 8,
        "the visitor may use only serde_json's fixed traversal scratch, never prompt storage: {allocations:?}"
    );
}

#[tokio::test]
async fn openai_responses_metadata_preserves_prompt_template_passthrough_body() {
    let pipeline = make_prompt_pipeline(Some("openai_responses"), Some("openai"), "127.0.0.1:443");
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let original = Bytes::from_static(
        br#"{"model":"gpt-4.1","prompt":{"id":"pmpt_123","version":"2","variables":{"name":"Ada"}}}"#,
    );
    let mut body = Some(original.clone());

    let action = pipeline
        .execute_http_request_body(&mut ctx, &mut body, true)
        .await
        .unwrap();

    assert!(
        matches!(action, FilterAction::Continue),
        "OpenAI prompt passthrough body detection should continue"
    );
    assert!(
        matches!(
            pipeline.execute_http_request(&mut ctx).await.unwrap(),
            FilterAction::Continue
        ),
        "an upstream declared as OpenAI Responses must allow prompt templates"
    );
    assert_eq!(
        ctx.selected_application_protocol(),
        Some("openai_responses"),
        "the load balancer must publish the Responses protocol"
    );
    assert_eq!(
        ctx.selected_application_provider(),
        Some("openai"),
        "the load balancer must publish the OpenAI provider"
    );
    assert_eq!(
        ctx.upstream.as_ref().unwrap().address.as_ref(),
        "127.0.0.1:443",
        "endpoint identity remains independent from application capability"
    );
    assert_eq!(
        body.as_deref(),
        Some(original.as_ref()),
        "OpenAI prompt object must be byte-exact"
    );
}

#[tokio::test]
async fn openai_responses_metadata_preserves_prompt_template_in_rebuilt_state() {
    let pipeline = make_prompt_pipeline(Some("openai_responses"), Some("openai"), "127.0.0.1:443");
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let mut state = ResponsesState::from_request_body(json!({
        "model": "gpt-4.1",
        "input": "hello",
        "prompt": {"id": "pmpt_123", "variables": {"name": "Ada"}}
    }));
    state.mark_request_body_for_rebuild();
    ctx.extensions.insert(state);
    let mut body = Some(Bytes::from_static(
        br#"{"model":"gpt-4.1","input":"hello","prompt":{"id":"pmpt_123","variables":{"name":"Ada"}}}"#,
    ));

    let action = pipeline
        .execute_http_request_body(&mut ctx, &mut body, true)
        .await
        .unwrap();

    assert!(
        matches!(action, FilterAction::Continue),
        "state-backed OpenAI prompt detection should continue"
    );
    assert!(
        matches!(
            pipeline.execute_http_request(&mut ctx).await.unwrap(),
            FilterAction::Continue
        ),
        "an upstream declared as OpenAI Responses must allow a rebuilt prompt request"
    );
    let rebuilt: serde_json::Value = serde_json::from_slice(body.as_deref().unwrap()).unwrap();
    assert_eq!(
        rebuilt["prompt"]["id"], "pmpt_123",
        "rebuilt request must preserve the prompt ID"
    );
    assert_eq!(
        rebuilt["prompt"]["variables"]["name"], "Ada",
        "rebuilt request must preserve prompt variables"
    );
}

#[tokio::test]
async fn prompt_template_requires_openai_responses_protocol_and_provider() {
    for (protocol, provider) in [
        (None, Some("openai")),
        (Some("openai_chat_completions"), Some("openai")),
        (Some("openai_responses"), None),
        (Some("openai_responses"), Some("vllm")),
    ] {
        let pipeline = make_prompt_pipeline(protocol, provider, "api.openai.com:443");
        let req = make_request(Method::POST, "/v1/responses");
        let mut ctx = make_filter_context(&req);
        let mut body = Some(Bytes::from_static(br#"{"model":"gpt-4.1","prompt":{"id":"pmpt_123"}}"#));

        let body_action = pipeline
            .execute_http_request_body(&mut ctx, &mut body, true)
            .await
            .unwrap();
        assert!(
            matches!(body_action, FilterAction::Continue),
            "prompt detection should complete before provider validation"
        );
        assert!(
            matches!(
                pipeline.execute_http_request(&mut ctx).await.unwrap(),
                FilterAction::Reject(_)
            ),
            "prompt templates require exact OpenAI Responses protocol and provider metadata"
        );
    }
}

#[tokio::test]
async fn routed_openai_responses_pipeline_allows_prompt_template_for_non_openai_endpoint() {
    let pipeline = make_prompt_pipeline(Some("openai_responses"), Some("openai"), "127.0.0.1:443");
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let original = Bytes::from_static(br#"{"model":"gpt-4.1","prompt":{"id":"pmpt_123"}}"#);
    let mut body = Some(original.clone());

    let body_action = pipeline
        .execute_http_request_body(&mut ctx, &mut body, true)
        .await
        .unwrap();
    assert!(
        matches!(body_action, FilterAction::Continue),
        "the assembled pipeline must continue after processing the complete request body"
    );

    let request_action = pipeline.execute_http_request(&mut ctx).await.unwrap();
    assert!(
        matches!(request_action, FilterAction::Continue),
        "the proxy must observe and trust exact OpenAI Responses application metadata selected earlier in the real pipeline"
    );
    let upstream = ctx
        .upstream
        .as_ref()
        .expect("the load balancer must select an upstream");
    assert_eq!(
        upstream.address.as_ref(),
        "127.0.0.1:443",
        "endpoint identity must not participate in the prompt capability decision"
    );
    assert_eq!(
        ctx.selected_application_protocol(),
        Some("openai_responses"),
        "the selected cluster must declare the Responses protocol"
    );
    assert_eq!(
        ctx.selected_application_provider(),
        Some("openai"),
        "the selected cluster must declare the OpenAI provider"
    );
    assert_eq!(
        body.as_deref(),
        Some(original.as_ref()),
        "the allowed pipeline path must preserve the prompt request byte-for-byte"
    );
}

#[tokio::test]
async fn initialized_state_preserves_scalar_input_on_first_pass() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let request_body = json!({"model":"gpt-4.1","input":"hello"});
    ctx.extensions.insert(ResponsesState::from_request_body(request_body));
    let original = br#"{
  "model": "gpt-4.1",
  "input": "hello"
}"#;
    let mut body = Some(Bytes::copy_from_slice(original));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(
        matches!(action, FilterAction::Continue),
        "initialized scalar request should continue"
    );
    assert_eq!(body.as_deref(), Some(original.as_slice()));
    assert!(
        ctx.request_headers_to_set.is_empty(),
        "byte-exact first-pass forwarding must not synthesize content-length"
    );
}

#[tokio::test]
async fn provider_previous_response_id_is_byte_exact_without_rehydrate() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let original = br#"{
  "model": "gpt-4.1",
  "input": "continue",
  "previous_response_id": "resp_provider"
}"#;
    let parsed = serde_json::from_slice(original).unwrap();
    ctx.extensions.insert(ResponsesState::from_request_body(parsed));
    let mut body = Some(Bytes::copy_from_slice(original));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(
        matches!(action, FilterAction::Continue),
        "provider previous_response_id passthrough should continue"
    );
    assert_eq!(body.as_deref(), Some(original.as_slice()));
    assert!(
        ctx.request_headers_to_set.is_empty(),
        "byte-exact previous_response_id passthrough must not synthesize headers"
    );
}

#[tokio::test]
async fn rebuild_preserves_provider_previous_response_id_without_rehydrate() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let request_body = json!({
        "model": "gpt-4.1",
        "input": "continue",
        "previous_response_id": "resp_provider"
    });
    let mut state = ResponsesState::from_request_body(request_body);
    state.messages.push(json!({
        "type": "function_call_output",
        "call_id": "file_search_1",
        "output": "search context"
    }));
    ctx.extensions.insert(state);
    let mut body = Some(Bytes::from_static(
        br#"{"model":"gpt-4.1","input":"continue","previous_response_id":"resp_provider"}"#,
    ));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(
        matches!(action, FilterAction::Continue),
        "rebuilt previous_response_id request should continue"
    );
    let outbound: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    assert_eq!(outbound["previous_response_id"], "resp_provider");
    assert_eq!(outbound["input"].as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn rebuild_serializes_from_state_request_body() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let original = json!({"model":"client-model","input":"hello"});
    let mut state = ResponsesState::from_request_body(original);
    state
        .messages
        .splice(0..0, [json!({"type":"message","role":"assistant","content":"history"})]);
    ctx.extensions.insert(state);
    let mut body = Some(Bytes::from(br#"{"model":"client-model","input":"hello"}"#.as_slice()));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(
        matches!(action, FilterAction::Continue),
        "state-backed rebuild should continue"
    );
    let outbound: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    assert_eq!(outbound["model"], "client-model", "serializes from state.request_body");
    assert_eq!(outbound["input"].as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn not_end_of_stream_continues() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let mut body = Some(Bytes::from(r#"{"input":"partial"}"#));

    let action = filter.on_request_body(&mut ctx, &mut body, false).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "non-EOS should return Continue"
    );
}

#[tokio::test]
async fn rebuilds_body_with_conversation_history() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let request_body = json!({
        "model": "gpt-4o",
        "input": "What did I say?",
        "previous_response_id": "resp_abc123"
    });

    let mut state = ResponsesState::from_request_body(request_body);
    state.history_rehydrated = true;
    let stored_history = vec![
        json!({"role": "user", "content": "Hello"}),
        json!({"role": "assistant", "content": "Hi there!"}),
    ];
    state.messages.splice(0..0, stored_history);
    ctx.extensions.insert(state);

    let mut body = Some(Bytes::from(
        r#"{"model":"gpt-4o","input":"What did I say?","previous_response_id":"resp_abc123"}"#,
    ));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "should continue after rebuilding body"
    );

    let rebuilt: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    assert_eq!(rebuilt["model"], "gpt-4o", "model should be preserved");

    let input = rebuilt["input"].as_array().unwrap();
    assert_eq!(input.len(), 3, "input should contain stored history + new message");
    assert_eq!(input[0]["content"], "Hello", "first message should be stored history");
    assert_eq!(
        input[1]["content"], "Hi there!",
        "second message should be stored history"
    );

    assert!(
        rebuilt.get("previous_response_id").is_none(),
        "previous_response_id should be stripped from outbound body"
    );
}

#[tokio::test]
async fn does_not_set_content_length_header() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let request_body = json!({
        "model": "gpt-4o",
        "input": "test",
        "previous_response_id": "resp_abc123"
    });
    let mut state = ResponsesState::from_request_body(request_body);
    state.history_rehydrated = true;
    state
        .messages
        .splice(0..0, vec![json!({"role": "user", "content": "stored"})]);
    ctx.extensions.insert(state);

    let mut body = Some(Bytes::from(
        r#"{"model":"gpt-4o","input":"test","previous_response_id":"resp_abc123"}"#,
    ));
    let _action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(
        ctx.extra_request_headers
            .iter()
            .all(|(k, _)| k.as_ref() != "content-length"),
        "filter must not set content-length (core handles framing)"
    );
}

#[tokio::test]
async fn preserves_other_request_fields() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let request_body = json!({
        "model": "gpt-4o",
        "input": "test",
        "temperature": 0.7,
        "stream": true,
        "previous_response_id": "resp_abc123"
    });
    let mut state = ResponsesState::from_request_body(request_body);
    state.history_rehydrated = true;
    state
        .messages
        .splice(0..0, vec![json!({"role": "user", "content": "stored"})]);
    ctx.extensions.insert(state);

    let mut body = Some(Bytes::from(
        r#"{"model":"gpt-4o","input":"test","temperature":0.7,"stream":true,"previous_response_id":"resp_abc123"}"#,
    ));
    let _action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    let rebuilt: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    assert_eq!(rebuilt["temperature"], 0.7, "temperature should be preserved");
    assert_eq!(rebuilt["stream"], true, "stream should be preserved");
    assert_eq!(rebuilt["model"], "gpt-4o", "model should be preserved");
}

#[tokio::test]
async fn rejects_oversized_rebuilt_body_with_413() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("max_rewritten_body_bytes: 16").unwrap();
    let filter = super::ResponsesProxyFilter::from_config(&yaml).unwrap();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let request_body = json!({
        "model": "gpt-4o",
        "input": "hello",
        "previous_response_id": "resp_abc123"
    });
    let mut state = ResponsesState::from_request_body(request_body);
    state.messages.splice(
        0..0,
        vec![json!({"role": "user", "content": "a]long message that exceeds the tiny limit"})],
    );
    ctx.extensions.insert(state);

    let mut body = Some(Bytes::from(
        r#"{"model":"gpt-4o","input":"hello","previous_response_id":"resp_abc123"}"#,
    ));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(&action, FilterAction::Reject(r) if r.status == 413),
        "should reject with 413 when rebuilt body exceeds max_rewritten_body_bytes"
    );
}

#[tokio::test]
async fn strips_conversation_from_outbound_body() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let request_body = json!({
        "model": "gpt-4o",
        "input": "hello",
        "conversation": {"id": "conv_abc123"}
    });
    let mut state = ResponsesState::from_request_body(request_body);
    state.history_rehydrated = true;
    ctx.extensions.insert(state);

    let mut body = Some(Bytes::from(
        r#"{"model":"gpt-4o","input":"hello","conversation":{"id":"conv_abc123"}}"#,
    ));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "conversation stripping should continue"
    );

    let rebuilt: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    assert!(
        rebuilt.get("conversation").is_none(),
        "conversation should be stripped from outbound body"
    );
    assert_eq!(rebuilt["model"], "gpt-4o");
}

#[tokio::test]
async fn strips_locally_consumed_history_selectors() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let request_body = json!({
        "model": "gpt-4o",
        "input": "hello",
        "previous_response_id": "resp_abc123",
        "conversation": "conv_xyz789"
    });
    let mut state = ResponsesState::from_request_body(request_body);
    state.history_rehydrated = true;
    state
        .messages
        .splice(0..0, vec![json!({"role": "user", "content": "stored"})]);
    ctx.extensions.insert(state);

    let mut body = Some(Bytes::from(
        r#"{"model":"gpt-4o","input":"hello","previous_response_id":"resp_abc123","conversation":"conv_xyz789"}"#,
    ));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "identifier stripping should continue"
    );

    let rebuilt: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    assert!(
        rebuilt.get("previous_response_id").is_none(),
        "previous_response_id should be stripped"
    );
    assert!(rebuilt.get("conversation").is_none(), "conversation should be stripped");
}

#[tokio::test]
async fn passthrough_preserves_conversation_in_body() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let mut body = Some(Bytes::from(
        r#"{"model":"gpt-4.1","input":"hello","conversation":{"id":"conv_native"}}"#,
    ));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "passthrough conversation should continue"
    );

    let parsed: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    assert_eq!(
        parsed["conversation"]["id"], "conv_native",
        "passthrough must preserve the provider-owned conversation"
    );
    assert_eq!(parsed["model"], "gpt-4.1", "other fields should be preserved");
    assert!(
        ctx.extra_request_headers
            .iter()
            .all(|(k, _)| k.as_ref() != "content-length"),
        "filter must not set content-length (core handles framing)"
    );
}

#[tokio::test]
async fn rebuilt_body_preserves_conversation_without_local_rehydration() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let request_body = json!({
        "model": "gpt-4o",
        "input": "hello",
        "conversation": {"id": "conv_native"}
    });
    let mut state = ResponsesState::from_request_body(request_body);
    state.mark_request_body_for_rebuild();
    ctx.extensions.insert(state);
    let mut body = Some(Bytes::from(
        r#"{"model":"gpt-4o","input":"hello","conversation":{"id":"conv_native"}}"#,
    ));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(
        matches!(action, FilterAction::Continue),
        "rebuilt request should continue to the provider"
    );
    let rebuilt: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    assert_eq!(
        rebuilt["conversation"]["id"], "conv_native",
        "rebuilt request must preserve the provider-owned conversation"
    );
}

#[tokio::test]
async fn rebuilt_provider_conversation_continuation_sends_only_new_items() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let request_body = json!({
        "model": "gpt-4o",
        "input": "weather in SF",
        "conversation": {"id": "conv_native"}
    });
    let mut state = ResponsesState::from_request_body(request_body);
    state
        .messages
        .push(json!({"type": "function_call", "id": "fc_123", "call_id": "call_123"}));
    state.provider_history_len = state.messages.len();
    state
        .messages
        .push(json!({"type": "function_call_output", "call_id": "call_123", "output": "sunny"}));
    state.iteration = 1;
    state.mark_request_body_for_rebuild();
    ctx.extensions.insert(state);
    let mut body = Some(Bytes::from(
        r#"{"model":"gpt-4o","input":"weather in SF","conversation":{"id":"conv_native"}}"#,
    ));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(
        matches!(action, FilterAction::Continue),
        "provider-owned continuation should continue to the provider"
    );
    let rebuilt: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    assert_eq!(
        rebuilt["conversation"]["id"], "conv_native",
        "provider-owned continuation must preserve its conversation"
    );
    assert_eq!(
        rebuilt["input"],
        json!([{"type": "function_call_output", "call_id": "call_123", "output": "sunny"}]),
        "provider-owned continuation must send only the new tool result"
    );
}

#[tokio::test]
async fn passthrough_none_body() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let mut body: Option<Bytes> = None;

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "None body should return Continue"
    );
    assert!(body.is_none(), "None body should remain None");
}

#[tokio::test]
async fn passthrough_invalid_json_body() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let original = b"not valid json {{{";
    let mut body = Some(Bytes::from_static(original));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "invalid JSON body should return Continue"
    );
    assert_eq!(
        body.as_deref(),
        Some(original.as_slice()),
        "invalid JSON body should pass through unchanged"
    );
}

#[tokio::test]
async fn rebuild_non_object_request_body_passes_through() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let state = ResponsesState {
        request_body: json!(["not", "an", "object"]),
        ..Default::default()
    };
    ctx.extensions.insert(state);

    let mut body = Some(Bytes::from(r#"["not","an","object"]"#));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "non-object request_body should continue"
    );

    let rebuilt: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    assert!(
        rebuilt.is_array(),
        "non-object request_body should pass through without modification"
    );
}

#[tokio::test]
async fn rebuild_does_not_set_content_type_header() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let request_body = json!({
        "model": "gpt-4o",
        "input": "test",
        "previous_response_id": "resp_abc123"
    });
    let mut state = ResponsesState::from_request_body(request_body);
    state.history_rehydrated = true;
    state
        .messages
        .splice(0..0, vec![json!({"role": "user", "content": "stored"})]);
    ctx.extensions.insert(state);

    let mut body = Some(Bytes::from(
        r#"{"model":"gpt-4o","input":"test","previous_response_id":"resp_abc123"}"#,
    ));
    let _action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(
        ctx.extra_request_headers
            .iter()
            .all(|(k, _)| k.as_ref() != "content-type"),
        "filter must not set content-type (core handles framing)"
    );
}

// -----------------------------------------------------------------------------
// messages_for_backend / compaction_to_assistant_message
// -----------------------------------------------------------------------------

#[test]
fn messages_for_backend_borrows_when_no_compaction() {
    let msgs = vec![
        json!({"role": "user", "content": "hello"}),
        json!({"role": "assistant", "content": "hi"}),
    ];
    let result = super::messages_for_backend(&msgs);
    assert!(
        matches!(result, std::borrow::Cow::Borrowed(_)),
        "should borrow when no compaction items"
    );
    assert_eq!(result.len(), 2);
}

#[test]
fn messages_for_backend_translates_compaction_item() {
    let encoded = base64::engine::general_purpose::STANDARD.encode("summary text");
    let msgs = vec![json!({"type": "compaction", "id": "c_1", "encrypted_content": encoded})];
    let result = super::messages_for_backend(&msgs);
    assert!(
        matches!(result, std::borrow::Cow::Owned(_)),
        "summary insertion must return an owned input array"
    );
    assert_eq!(result.len(), 1);
    assert_eq!(result[0]["role"], "assistant");
    assert!(
        result[0]["content"].as_str().unwrap().contains("summary text"),
        "inserted summary message must contain the supplied summary"
    );
}

#[test]
fn messages_for_backend_mixed_items() {
    let encoded = base64::engine::general_purpose::STANDARD.encode("ctx");
    let msgs = vec![
        json!({"type": "compaction", "id": "c_1", "encrypted_content": encoded}),
        json!({"role": "user", "content": "hello"}),
    ];
    let result = super::messages_for_backend(&msgs);
    assert_eq!(result.len(), 2);
    assert_eq!(result[0]["role"], "assistant");
    assert_eq!(result[1]["role"], "user");
}

#[tokio::test]
async fn compacted_outbound_serializes_resolved_file_data_not_file_url() {
    const FILE_URL: &str = "https://files.internal/secret.bin";
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let request_body = json!({
        "model": "gpt-4o",
        "previous_response_id": "resp_prev",
        "input": [{
            "type": "message",
            "role": "user",
            "content": [{"type": "input_file", "file_url": FILE_URL}]
        }]
    });
    let mut state = ResponsesState::from_request_body(request_body);
    state.history_rehydrated = true;
    let encoded = base64::engine::general_purpose::STANDARD.encode("summary");
    state.messages = vec![
        json!({"type": "compaction", "id": "c_1", "encrypted_content": encoded}),
        json!({
            "type": "message",
            "role": "user",
            "content": [{"type": "input_file", "file_data": "SGVsbG8="}]
        }),
    ];
    ctx.extensions.insert(state);
    let mut body = Some(Bytes::from_static(
        br#"{"model":"gpt-4o","input":[{"type":"message","role":"user","content":[{"type":"input_file","file_url":"https://files.internal/secret.bin"}]}],"previous_response_id":"resp_prev"}"#,
    ));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "compaction rewrite should continue with the resolved current-turn body"
    );

    let outbound: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    let outbound_text = outbound.to_string();
    assert!(
        !outbound_text.contains(FILE_URL),
        "proxy must not serialize the unresolved file_url after compaction"
    );
    assert_eq!(
        outbound["input"][1]["content"][0]["file_data"], "SGVsbG8=",
        "proxy must serialize the resolved current-turn file_data"
    );
}

#[test]
fn compaction_to_assistant_message_decodes_encrypted_content() {
    let encoded = base64::engine::general_purpose::STANDARD.encode("decoded summary");
    let item = json!({"type": "compaction", "id": "c_1", "encrypted_content": encoded});
    let msg = super::compaction_to_assistant_message(&item);
    assert_eq!(msg["role"], "assistant");
    assert!(
        msg["content"].as_str().unwrap().contains("decoded summary"),
        "inserted summary message must contain the decoded summary"
    );
}

#[test]
fn compaction_to_assistant_message_handles_missing_content() {
    let item = json!({"type": "compaction", "id": "c_1"});
    let msg = super::compaction_to_assistant_message(&item);
    assert_eq!(msg["role"], "assistant");
    assert!(
        msg["content"]
            .as_str()
            .unwrap()
            .contains("[Previous conversation summary]")
    );
}

#[test]
fn compaction_to_assistant_message_handles_invalid_base64() {
    let item = json!({"type": "compaction", "id": "c_1", "encrypted_content": "not!valid!base64!!!"});
    let msg = super::compaction_to_assistant_message(&item);
    assert_eq!(msg["role"], "assistant");
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

/// Build a request whose selected top-level field contains 256 alternating
/// object and array levels. When `complete` is false, omit the root close.
fn deeply_nested_request(field: &str, complete: bool) -> Vec<u8> {
    let mut body = format!(r#"{{"{field}":"#).into_bytes();
    for depth in 0..256 {
        if depth % 2 == 0 {
            body.extend_from_slice(br#"{"nested":"#);
        } else {
            body.push(b'[');
        }
    }
    body.extend_from_slice(b"null");
    for depth in (0..256).rev() {
        body.push(if depth % 2 == 0 { b'}' } else { b']' });
    }
    if complete {
        body.push(b'}');
    }
    body
}

fn make_filter() -> Box<dyn HttpFilter> {
    super::ResponsesProxyFilter::from_config(&serde_yaml::Value::Null).unwrap()
}

/// Build a real routing pipeline so application metadata is selected by core.
fn make_prompt_pipeline(
    application_protocol: Option<&str>,
    application_provider: Option<&str>,
    endpoint: &str,
) -> FilterPipeline {
    let mut application = String::new();
    if application_protocol.is_some() || application_provider.is_some() {
        application.push_str("      http:\n");
    }
    if let Some(protocol) = application_protocol {
        writeln!(application, "        application_protocol: \"{protocol}\"").unwrap();
    }
    if let Some(provider) = application_provider {
        writeln!(application, "        application_provider: \"{provider}\"").unwrap();
    }
    let yaml = format!(
        r#"
- filter: router
  routes:
    - path: /v1/responses
      cluster: target
- filter: load_balancer
  clusters:
    - name: target
{application}      endpoints:
        - "{endpoint}"
- filter: openai_responses_proxy
"#
    );
    let mut registry = FilterRegistry::with_builtins();
    praxis_filter::register_filters!(
        @register registry,
        http "openai_responses_proxy" => super::ResponsesProxyFilter::from_config
    );
    let mut entries: Vec<FilterEntry> = serde_yaml::from_str(&yaml).unwrap();
    FilterPipeline::build(&mut entries, &registry).unwrap()
}
