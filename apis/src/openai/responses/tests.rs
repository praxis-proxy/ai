// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Dispatch and conformance unit tests for the Responses API filters.
//!
//! Request-processor behavior (classification, promotion, routing mode, and
//! managed-path policy) is covered in `request/tests.rs`, the module that owns
//! that logic.

#[cfg(feature = "openai-responses")]
use super::*;

// -----------------------------------------------------------------------------
// streamed_round_is_dispatchable Tests
// -----------------------------------------------------------------------------

#[cfg(feature = "openai-responses")]
#[test]
fn dispatchable_requires_terminal_completed_no_parse_error() {
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let mut state = state::ResponsesState::default();
    state.request_body = serde_json::json!({"stream": true});
    // Non-terminal stream: not dispatchable.
    assert!(!streamed_round_is_dispatchable(&ctx, &state));
    ctx.set_metadata("responses.stream_completion", "terminal");
    state.response_object = serde_json::json!({"status": "completed"});
    assert!(streamed_round_is_dispatchable(&ctx, &state));
    ctx.set_metadata("responses.stream_parse_error", "true");
    assert!(
        !streamed_round_is_dispatchable(&ctx, &state),
        "parse error blocks dispatch"
    );
}

#[cfg(feature = "openai-responses")]
#[test]
fn non_streaming_request_is_always_dispatchable() {
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let ctx = crate::test_utils::make_filter_context(&req);
    let state = state::ResponsesState::default(); // no stream flag
    assert!(streamed_round_is_dispatchable(&ctx, &state));
}

#[cfg(feature = "openai-responses")]
#[test]
fn fs_end_stream_writes_five_keys_and_is_idempotent() {
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    fs_end_stream_with_error_ctx(&mut ctx, "server_error", "boom");
    assert_eq!(ctx.get_metadata("responses.stream_error_code"), Some("server_error"));
    assert_eq!(ctx.get_metadata("responses.stream_error_message"), Some("boom"));
    assert_eq!(ctx.get_metadata("responses.skip_persist"), Some("true"));
    // After the #1046 unification the stream-stop is armed on the single
    // continuation owner (`openai_agentic_loop`), not the demoted file-search filter.
    let r = ctx.filter_results.get("openai_agentic_loop").unwrap();
    assert_eq!(r.get("action"), Some("done"));
    assert_eq!(r.get("pending"), Some("false"));
    // Idempotent: a second call with a different code does not clobber.
    fs_end_stream_with_error_ctx(&mut ctx, "other_code", "later");
    assert_eq!(ctx.get_metadata("responses.stream_error_code"), Some("server_error"));
}

// -----------------------------------------------------------------------------
// Conformance Verification Tests
// -----------------------------------------------------------------------------

#[cfg(feature = "openai-responses")]
#[test]
#[expect(clippy::print_stdout, reason = "sentinel output for xtask conformance verification")]
fn conformance_responses_routes_match_runtime_registry() {
    for spec in routes::operation_specs() {
        let path = spec.runtime_path().replace("{response_id}", "resp_test");
        let matched = routes::match_route(spec.method().as_str(), &path, spec.transport())
            .unwrap_or_else(|| panic!("failed to match route for {} {path}", spec.method().as_str()));
        assert_eq!(
            matched.spec.operation,
            spec.operation,
            "matched wrong operation for {} {path}",
            spec.method().as_str()
        );
    }
    println!("PRAXIS_CONFORMANCE_OK responses route_dispatch");
}

#[cfg(feature = "openai-responses")]
#[test]
#[expect(clippy::print_stdout, reason = "sentinel output for xtask conformance verification")]
fn conformance_responses_success_payloads_match_generated_response_schemas() {
    let spec = load_openai_spec();
    let schema = spec
        .pointer("/components/schemas/Response")
        .expect("missing Response schema");

    let chat_response = serde_json::json!({
        "id": "chatcmpl_conformance_123",
        "object": "chat.completion",
        "created": 1_700_000_000_u64,
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "Hello world"
            }
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "prompt_tokens_details": {
                "cached_tokens": 2,
                "cache_write_tokens": 0
            },
            "completion_tokens_details": {
                "reasoning_tokens": 0
            }
        }
    });
    let request = serde_json::json!({"model": "gpt-4o", "input": "Hello"});
    let context = crate::openai::translation::chat_completions::ResponseContext::from_responses_request(
        &request,
        "resp_conformance_123".to_owned(),
        1_700_000_000,
    );
    let response_resource =
        crate::openai::translation::chat_completions::chat_response_to_response_resource(&chat_response, &context)
            .expect("translator must convert chat response to response resource");

    assert_response_matches_schema(&spec, "Response", schema, &response_resource);
    println!("PRAXIS_CONFORMANCE_OK responses success_response_contract");
}

#[cfg(feature = "openai-responses")]
#[test]
#[expect(clippy::print_stdout, reason = "sentinel output for xtask conformance verification")]
fn conformance_responses_sse_lifecycle_events_match_schemas() {
    use crate::openai::responses::responses_to_chat_completions::stream::tests::{run_stream, wide_limits};

    let spec = load_openai_spec();
    let sse_event_schema = spec
        .pointer("/components/schemas/ResponseStreamEvent")
        .expect("missing ResponseStreamEvent schema");

    // Scenarios: buffered text, tool call, incomplete (length)
    let buffered_text_chunks = [
        r#"{"id":"chatcmpl_1","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello "}}]}"#,
        r#"{"id":"chatcmpl_1","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"world"}}]}"#,
        r#"{"id":"chatcmpl_1","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12,"prompt_tokens_details":{"cached_tokens":0,"cache_write_tokens":0},"completion_tokens_details":{"reasoning_tokens":0}}}"#,
    ];

    let tool_call_chunks = [
        r#"{"id":"chatcmpl_2","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_123","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Boston\"}"}}]}}]}"#,
        r#"{"id":"chatcmpl_2","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":15,"completion_tokens":10,"total_tokens":25,"prompt_tokens_details":{"cached_tokens":0,"cache_write_tokens":0},"completion_tokens_details":{"reasoning_tokens":0}}}"#,
    ];

    let incomplete_chunks = [
        r#"{"id":"chatcmpl_3","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":"Truncated..."}}]}"#,
        r#"{"id":"chatcmpl_3","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"finish_reason":"length"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,"prompt_tokens_details":{"cached_tokens":0,"cache_write_tokens":0},"completion_tokens_details":{"reasoning_tokens":0}}}"#,
    ];

    let test_streams = [
        ("buffered_text", run_stream(&buffered_text_chunks, wide_limits())),
        ("tool_call", run_stream(&tool_call_chunks, wide_limits())),
        ("incomplete", run_stream(&incomplete_chunks, wide_limits())),
    ];

    for (scenario, stream_events) in test_streams {
        assert!(!stream_events.is_empty(), "scenario {scenario} generated no events");

        let mut terminal_count = 0;
        for (idx, (event_type, event_val)) in stream_events.iter().enumerate() {
            assert_response_matches_schema(
                &spec,
                &format!("SSE event {idx} ({event_type}) in {scenario}"),
                sse_event_schema,
                event_val,
            );

            let seq = event_val
                .get("sequence_number")
                .and_then(serde_json::Value::as_u64)
                .expect("sequence_number required");
            assert_eq!(seq, idx as u64, "sequence_number mismatch in {scenario} at index {idx}");

            if idx == 0 {
                assert_eq!(
                    event_type, "response.created",
                    "first event in {scenario} must be response.created"
                );
            }

            if matches!(
                event_type.as_str(),
                "response.completed" | "response.failed" | "response.incomplete"
            ) {
                terminal_count += 1;
            }
        }

        assert_eq!(
            terminal_count, 1,
            "scenario {scenario} must have exactly 1 terminal event"
        );
        let (last_event_type, _) = stream_events.last().unwrap();
        assert!(
            matches!(
                last_event_type.as_str(),
                "response.completed" | "response.failed" | "response.incomplete"
            ),
            "last event in {scenario} must be terminal, got {last_event_type}"
        );
    }

    println!("PRAXIS_CONFORMANCE_OK responses sse_lifecycle_contract");
}

#[cfg(feature = "openai-responses")]
#[test]
#[expect(clippy::print_stdout, reason = "sentinel output for xtask conformance verification")]
fn conformance_responses_generated_schema_check_rejects_incomplete_payloads() {
    let spec = load_openai_spec();
    let schema = spec
        .pointer("/components/schemas/Response")
        .expect("missing Response schema");

    let chat_response = serde_json::json!({
        "id": "chatcmpl_conformance_123",
        "object": "chat.completion",
        "created": 1_700_000_000_u64,
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "Hello world"
            }
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "prompt_tokens_details": {
                "cached_tokens": 2,
                "cache_write_tokens": 0
            },
            "completion_tokens_details": {
                "reasoning_tokens": 0
            }
        }
    });
    let request = serde_json::json!({"model": "gpt-4o", "input": "Hello"});
    let context = crate::openai::translation::chat_completions::ResponseContext::from_responses_request(
        &request,
        "resp_conformance_123".to_owned(),
        1_700_000_000,
    );
    let valid_response =
        crate::openai::translation::chat_completions::chat_response_to_response_resource(&chat_response, &context)
            .expect("translator must convert chat response to response resource");

    // Negative 1: missing cache_write_tokens in input_tokens_details
    let mut missing_cache_write = valid_response.clone();
    if let Some(details) = missing_cache_write
        .get_mut("usage")
        .and_then(|u| u.get_mut("input_tokens_details"))
        .and_then(|d| d.as_object_mut())
    {
        details.remove("cache_write_tokens");
    }
    assert!(
        !response_schema_matches(&spec, schema, &missing_cache_write),
        "schema check must reject usage missing required cache_write_tokens"
    );

    // Negative 2: missing required top-level field `id`
    let mut missing_id = valid_response.clone();
    if let Some(obj) = missing_id.as_object_mut() {
        obj.remove("id");
    }
    assert!(
        !response_schema_matches(&spec, schema, &missing_id),
        "schema check must reject response missing required id"
    );

    // Negative 3: SSE event missing required sequence_number
    let sse_event_schema = spec
        .pointer("/components/schemas/ResponseStreamEvent")
        .expect("missing ResponseStreamEvent schema");
    let valid_sse_event = serde_json::json!({
        "type": "response.created",
        "sequence_number": 0,
        "response": valid_response
    });
    let mut missing_sequence_number = valid_sse_event.clone();
    if let Some(obj) = missing_sequence_number.as_object_mut() {
        obj.remove("sequence_number");
    }

    assert!(
        !response_schema_matches(&spec, sse_event_schema, &missing_sequence_number),
        "schema check must reject SSE event missing required sequence_number"
    );

    println!("PRAXIS_CONFORMANCE_OK responses schema_check_sensitivity");
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

#[cfg(feature = "openai-responses")]
fn load_openai_spec() -> serde_json::Value {
    let spec_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../docs/conformance/specs/openai-openapi.yaml"
    );
    let content = std::fs::read_to_string(spec_path).expect("read spec");
    let sanitized = content.replace("9223372036854776000", "9223372036854775807");
    serde_yaml::from_str(&sanitized).expect("parse spec")
}

#[cfg(feature = "openai-responses")]
fn assert_response_matches_schema(
    spec: &serde_json::Value,
    path: &str,
    schema: &serde_json::Value,
    value: &serde_json::Value,
) {
    if let Err(err) = check_schema_match(spec, schema, value) {
        panic!("{path} does not match schema: {err}; value: {value}");
    }
}

#[cfg(feature = "openai-responses")]
fn response_schema_matches(spec: &serde_json::Value, schema: &serde_json::Value, value: &serde_json::Value) -> bool {
    check_schema_match(spec, schema, value).is_ok()
}

#[cfg(feature = "openai-responses")]
fn check_schema_match(
    spec: &serde_json::Value,
    schema: &serde_json::Value,
    value: &serde_json::Value,
) -> Result<(), String> {
    let schema = resolve_spec_schema_ref(spec, schema);

    if value.is_null() && schema.get("nullable").and_then(serde_json::Value::as_bool) == Some(true) {
        return Ok(());
    }
    if let Some(variants) = schema.get("oneOf").and_then(serde_json::Value::as_array) {
        let mut matches = Vec::new();
        for (i, v) in variants.iter().enumerate() {
            if let Ok(()) = check_schema_match(spec, v, value) {
                matches.push(i);
            }
        }
        if matches.len() == 1 {
            return Ok(());
        }
        return Err(format!("oneOf matched {} variants for {value:?}", matches.len()));
    }
    if let Some(variants) = schema.get("anyOf").and_then(serde_json::Value::as_array) {
        for v in variants {
            if check_schema_match(spec, v, value).is_ok() {
                return Ok(());
            }
        }
        return Err(format!("anyOf matched 0 variants for {value:?}"));
    }
    if let Some(variants) = schema.get("allOf").and_then(serde_json::Value::as_array) {
        for (i, v) in variants.iter().enumerate() {
            if let Err(err) = check_schema_match(spec, v, value) {
                return Err(format!("allOf branch {i} failed: {err}"));
            }
        }
        return Ok(());
    }
    if let Some(enum_vals) = schema.get("enum").and_then(serde_json::Value::as_array)
        && !enum_vals.contains(value)
    {
        return Err(format!("value {value:?} not in enum {enum_vals:?}"));
    }
    if let Some(schema_type) = schema.get("type") {
        let matches_type = schema_type.as_str().map_or_else(
            || {
                schema_type.as_array().is_some_and(|types| {
                    types
                        .iter()
                        .filter_map(serde_json::Value::as_str)
                        .any(|kind| value_has_type(value, kind))
                })
            },
            |kind| value_has_type(value, kind),
        );
        if !matches_type {
            return Err(format!("type mismatch: expected {schema_type:?}, got {value:?}"));
        }
    }

    if let Some(items_schema) = schema.get("items") {
        let Some(items) = value.as_array() else {
            return Err(format!("expected array for items check, got {value:?}"));
        };
        for (idx, item) in items.iter().enumerate() {
            if let Err(err) = check_schema_match(spec, items_schema, item) {
                return Err(format!("array item [{idx}] mismatch: {err}"));
            }
        }
    }

    if let Some(object) = value.as_object() {
        if let Some(required) = schema.get("required").and_then(serde_json::Value::as_array) {
            for req in required {
                if let Some(req_str) = req.as_str()
                    && !object.contains_key(req_str)
                {
                    return Err(format!("missing required property {req_str:?}"));
                }
            }
        }
        if let Some(properties) = schema.get("properties").and_then(serde_json::Value::as_object) {
            let required_list = schema
                .get("required")
                .and_then(serde_json::Value::as_array)
                .map(|arr| arr.iter().filter_map(serde_json::Value::as_str).collect::<Vec<_>>())
                .unwrap_or_default();

            for (prop_name, prop_schema) in properties {
                if let Some(prop_val) = object.get(prop_name) {
                    if prop_val.is_null() && !required_list.contains(&prop_name.as_str()) {
                        continue;
                    }
                    if let Err(err) = check_schema_match(spec, prop_schema, prop_val) {
                        return Err(format!("property {prop_name:?} mismatch: {err}"));
                    }
                }
            }
        }
    }

    Ok(())
}

#[cfg(feature = "openai-responses")]
fn value_has_type(value: &serde_json::Value, schema_type: &str) -> bool {
    match schema_type {
        "array" => value.is_array(),
        "boolean" => value.is_boolean(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "null" => value.is_null(),
        "number" => value.as_f64().is_some(),
        "object" => value.is_object(),
        "string" => value.is_string(),
        _ => false,
    }
}

#[cfg(feature = "openai-responses")]
fn resolve_spec_schema_ref<'a>(spec: &'a serde_json::Value, schema: &'a serde_json::Value) -> &'a serde_json::Value {
    let Some(ref_path) = schema.get("$ref").and_then(serde_json::Value::as_str) else {
        return schema;
    };
    let pointer = ref_path.strip_prefix('#').unwrap_or(ref_path);
    spec.pointer(pointer)
        .unwrap_or_else(|| panic!("missing schema ref {ref_path}"))
}

#[cfg(feature = "openai-responses")]
#[test]
fn local_tool_guardrail_handoff_is_shared_by_each_configured_policy() {
    let mut extensions = RequestExtensions::default();
    let mut state = state::ResponsesState {
        messages: vec![
            serde_json::json!({"role":"user", "content":"old input"}),
            serde_json::json!({"type":"function_call", "call_id":"call_1", "name":"lookup"}),
            serde_json::json!({"type":"function_call_output", "call_id":"call_1", "output":"untrusted result"}),
            serde_json::json!({"type":"reasoning", "summary":[]}),
        ]
        .into(),
        ..state::ResponsesState::default()
    };
    state.mark_local_tool_results_from(1);
    extensions.insert(state);

    assert_eq!(
        local_tool_guardrail_messages(&extensions, 1024).unwrap(),
        vec![serde_json::json!({"role":"user", "content":"untrusted result"})]
    );
    assert_eq!(
        local_tool_guardrail_messages(&extensions, 1024).unwrap(),
        vec![serde_json::json!({"role":"user", "content":"untrusted result"})],
        "a second guardrail policy must receive the same local-result suffix"
    );
    assert_eq!(
        extensions
            .get::<state::ResponsesState>()
            .unwrap()
            .pending_local_tool_guardrail_start,
        Some(1),
        "only the loop owner clears the marker after the whole filter chain"
    );
}

#[cfg(feature = "openai-responses")]
#[test]
fn local_tool_guardrail_handoff_rejects_oversized_result_before_copying() {
    let mut extensions = RequestExtensions::default();
    let mut state = state::ResponsesState {
        messages: vec![serde_json::json!({
            "type":"function_call_output",
            "call_id":"call_1",
            "output":"untrusted result"
        })]
        .into(),
        ..state::ResponsesState::default()
    };
    state.mark_local_tool_results_from(0);
    extensions.insert(state);

    let error = local_tool_guardrail_messages(&extensions, 8)
        .expect_err("the result must be bounded before the async callout copy");
    assert!(
        error.to_string().contains("guardrail evaluation limit"),
        "oversized local tool results must be rejected before copying into the NeMo callout payload: {error}"
    );
}

#[cfg(feature = "openai-responses")]
#[test]
fn local_tool_guardrail_failure_stays_with_responses_state() {
    let mut extensions = RequestExtensions::default();
    extensions.insert(state::ResponsesState::default());

    assert!(
        record_local_tool_guardrail_failure(
            &mut extensions,
            403,
            "content_blocked",
            "blocked local result".to_owned(),
        ),
        "the Responses state must accept the guardrail failure for the loop owner to enforce"
    );
    let failure = extensions
        .get::<state::ResponsesState>()
        .and_then(|state| state.dispatch_failure.as_ref())
        .expect("the loop owner should receive the terminal failure");
    assert_eq!(failure.status, 403);
    assert_eq!(failure.code, "content_blocked");
    assert_eq!(failure.message, "blocked local result");
}
