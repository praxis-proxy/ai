// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

use serde_json::json;

use super::*;
use crate::openai::responses::config_validation::FailureMode;

// =============================================================================
// Config tests
// =============================================================================

fn base_config() -> CompactFilterConfig {
    CompactFilterConfig {
        inference_url: "http://localhost:11434/v1/chat/completions".to_owned(),
        default_model: "gpt-4o-mini".to_owned(),
        tiktoken_encoding: "cl100k_base".to_owned(),
        summary_prefix: None,
        timeout_ms: None,
        callout_failure_mode: None,
        status_on_error: None,
    }
}

#[test]
fn build_config_applies_defaults() {
    let cfg = build_config(&base_config()).unwrap();
    assert_eq!(cfg.inference_url, "http://localhost:11434/v1/chat/completions");
    assert_eq!(cfg.default_model, "gpt-4o-mini");
    assert_eq!(cfg.tiktoken_encoding, "cl100k_base");
    assert_eq!(cfg.callout.timeout_ms, 30_000);
    assert_eq!(cfg.callout.failure_mode, FailureMode::Closed);
    assert_eq!(cfg.callout.status_on_error, 502);
}

#[test]
fn build_config_rejects_empty_inference_url() {
    let mut cfg = base_config();
    cfg.inference_url = String::new();
    assert!(build_config(&cfg).is_err());
}

#[test]
fn build_config_rejects_zero_timeout() {
    let mut cfg = base_config();
    cfg.timeout_ms = Some(0);
    assert!(build_config(&cfg).is_err());
}

#[test]
fn build_config_rejects_invalid_status() {
    let mut cfg = base_config();
    cfg.status_on_error = Some(999);
    assert!(build_config(&cfg).is_err());
}

#[test]
fn build_config_rejects_unsupported_tiktoken_encoding() {
    let mut cfg = base_config();
    cfg.tiktoken_encoding = "gpt4".to_owned();
    let err = build_config(&cfg).unwrap_err();
    assert!(err.to_string().contains("unsupported tiktoken_encoding"));
}

#[test]
fn build_config_accepts_o200k_base_encoding() {
    let mut cfg = base_config();
    cfg.tiktoken_encoding = "o200k_base".to_owned();
    assert!(build_config(&cfg).is_ok());
}

#[test]
fn build_config_custom_values() {
    let mut cfg = base_config();
    cfg.timeout_ms = Some(60_000);
    cfg.callout_failure_mode = Some(FailureMode::Open);
    cfg.status_on_error = Some(503);
    let validated = build_config(&cfg).unwrap();
    assert_eq!(validated.callout.timeout_ms, 60_000);
    assert_eq!(validated.callout.failure_mode, FailureMode::Open);
    assert_eq!(validated.callout.status_on_error, 503);
}

// =============================================================================
// extract_compaction_config tests
// =============================================================================

#[test]
fn extract_compaction_config_with_compaction_entry() {
    let cm = Some(json!([{"type": "compaction", "compact_threshold": 50_000}]));
    let params = extract_compaction_config(&cm);
    assert!(params.is_some());
    let params = params.unwrap();
    assert_eq!(params.compact_threshold, 50_000);
    assert!(params.compaction_model.is_none());
}

#[test]
fn extract_compaction_config_with_model_override() {
    let cm = Some(json!([{
        "type": "compaction",
        "compact_threshold": 100_000,
        "compaction_model": "gpt-4o"
    }]));
    let params = extract_compaction_config(&cm).unwrap();
    assert_eq!(params.compact_threshold, 100_000);
    assert_eq!(params.compaction_model.as_deref(), Some("gpt-4o"));
}

#[test]
fn extract_compaction_config_no_compaction_entry() {
    let cm = Some(json!([{"type": "truncation", "max_tokens": 4096}]));
    assert!(extract_compaction_config(&cm).is_none());
}

#[test]
fn extract_compaction_config_none() {
    assert!(extract_compaction_config(&None).is_none());
}

#[test]
fn extract_compaction_config_empty_array() {
    let cm = Some(json!([]));
    assert!(extract_compaction_config(&cm).is_none());
}

#[test]
fn extract_compaction_config_missing_threshold_skips_compaction() {
    let cm = Some(json!([{"type": "compaction"}]));
    assert!(
        extract_compaction_config(&cm).is_none(),
        "missing threshold should skip compaction"
    );
}

#[test]
fn extract_compaction_config_null_threshold_skips_compaction() {
    let cm = Some(json!([{"type": "compaction", "compact_threshold": null}]));
    assert!(
        extract_compaction_config(&cm).is_none(),
        "null threshold should skip compaction"
    );
}

#[test]
fn extract_compaction_config_zero_threshold_compacts_immediately() {
    let cm = Some(json!([{"type": "compaction", "compact_threshold": 0}]));
    let params = extract_compaction_config(&cm).unwrap();
    assert_eq!(params.compact_threshold, 0, "explicit zero should still compact");
}

#[test]
fn extract_compaction_config_float_threshold_skips_compaction() {
    let cm = Some(json!([{"type": "compaction", "compact_threshold": 0.9}]));
    assert!(
        extract_compaction_config(&cm).is_none(),
        "float threshold should skip compaction (as_u64 returns None)"
    );
}

// =============================================================================
// build_compaction_item tests
// =============================================================================

#[test]
fn compaction_item_has_correct_shape() {
    use base64::Engine as _;
    let item = build_compaction_item("compact_abc123", "This is a summary.", DEFAULT_SUMMARY_PREFIX);
    assert_eq!(item["type"], "compaction");
    assert_eq!(item["id"], "compact_abc123");
    let encoded = item["encrypted_content"].as_str().unwrap();
    let decoded = base64::engine::general_purpose::STANDARD.decode(encoded).unwrap();
    assert_eq!(String::from_utf8(decoded).unwrap(), "This is a summary.");
    assert!(
        item.get("summary_prefix").is_none(),
        "default prefix should not be stored in the item"
    );
}

#[test]
fn compaction_item_with_custom_prefix() {
    let item = build_compaction_item("compact_custom", "Summary.", "Context:\n");
    assert_eq!(
        item["summary_prefix"], "Context:\n",
        "custom prefix should be stored in the item"
    );
}

#[test]
fn build_config_applies_default_summary_prefix() {
    let cfg = build_config(&base_config()).unwrap();
    assert_eq!(cfg.summary_prefix, DEFAULT_SUMMARY_PREFIX);
}

#[test]
fn build_config_applies_custom_summary_prefix() {
    let mut cfg = base_config();
    cfg.summary_prefix = Some("Summary:\n".to_owned());
    let validated = build_config(&cfg).unwrap();
    assert_eq!(validated.summary_prefix, "Summary:\n");
}

// =============================================================================
// parse_summarization_response tests
// =============================================================================

#[test]
fn parse_valid_chat_completion_response() {
    let response = json!({
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "Here is the summary."
            }
        }]
    });
    let body = serde_json::to_vec(&response).unwrap();
    let result = parse_summarization_response(&body);
    assert_eq!(result.unwrap(), "Here is the summary.");
}

#[test]
fn parse_malformed_response_returns_error() {
    let result = parse_summarization_response(b"not json");
    assert!(result.is_err());
}

#[test]
fn parse_response_missing_choices_returns_error() {
    let response = json!({"id": "chatcmpl-123"});
    let body = serde_json::to_vec(&response).unwrap();
    assert!(parse_summarization_response(&body).is_err());
}

#[test]
fn parse_response_empty_choices_returns_error() {
    let response = json!({"choices": []});
    let body = serde_json::to_vec(&response).unwrap();
    assert!(parse_summarization_response(&body).is_err());
}

// =============================================================================
// build_conversation_text tests
// =============================================================================

#[test]
fn conversation_text_simple_messages() {
    let messages = vec![
        json!({"role": "user", "content": "Hello"}),
        json!({"role": "assistant", "content": "Hi there!"}),
    ];
    let text = build_conversation_text(&messages);
    assert!(text.contains("user: Hello"));
    assert!(text.contains("assistant: Hi there!"));
}

#[test]
fn conversation_text_empty_messages() {
    let text = build_conversation_text(&[]);
    assert!(text.is_empty());
}

#[test]
fn conversation_text_skips_empty_content() {
    let messages = vec![
        json!({"role": "user", "content": "Hello"}),
        json!({"role": "assistant"}),
        json!({"role": "user", "content": "Still here"}),
    ];
    let text = build_conversation_text(&messages);
    assert!(!text.contains("assistant"));
    assert!(text.contains("user: Hello"));
    assert!(text.contains("user: Still here"));
}

#[test]
fn conversation_text_array_content() {
    let messages = vec![json!({
        "role": "user",
        "content": [
            {"type": "text", "text": "Part one"},
            {"type": "text", "text": "Part two"}
        ]
    })];
    let text = build_conversation_text(&messages);
    assert!(text.contains("user: Part one Part two"));
}

// =============================================================================
// extract_content tests
// =============================================================================

#[test]
fn extract_content_string() {
    let msg = json!({"content": "hello"});
    assert_eq!(extract_content(&msg), "hello");
}

#[test]
fn extract_content_array() {
    let msg = json!({"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]});
    assert_eq!(extract_content(&msg), "a b");
}

#[test]
fn extract_content_missing() {
    let msg = json!({"role": "user"});
    assert_eq!(extract_content(&msg), "");
}

#[test]
fn extract_content_null() {
    let msg = json!({"content": null});
    assert_eq!(extract_content(&msg), "");
}

// =============================================================================
// build_summarization_request tests
// =============================================================================

#[test]
fn summarization_request_without_instructions() {
    let messages = vec![json!({"role": "user", "content": "Hello"})];
    let conversation_text = build_conversation_text(&messages);
    let req = build_summarization_request(&conversation_text, None, "gpt-4o-mini");
    assert_eq!(req.method, http::Method::POST);
    let body: Value = serde_json::from_slice(&req.body).unwrap();
    assert_eq!(body["model"], "gpt-4o-mini");
    let msgs = body["messages"].as_array().unwrap();
    assert_eq!(msgs.len(), 2);
    assert_eq!(msgs[0]["role"], "system");
    assert!(msgs[0]["content"].as_str().unwrap().contains("Summarize"));
    assert_eq!(msgs[1]["role"], "user");
    assert!(msgs[1]["content"].as_str().unwrap().contains("user: Hello"));
}

#[test]
fn summarization_request_with_instructions() {
    let messages = vec![json!({"role": "user", "content": "Hello"})];
    let conversation_text = build_conversation_text(&messages);
    let req = build_summarization_request(&conversation_text, Some("Be concise"), "gpt-4o-mini");
    let body: Value = serde_json::from_slice(&req.body).unwrap();
    let system = body["messages"][0]["content"].as_str().unwrap();
    assert!(system.starts_with("Be concise"), "instructions should be prepended");
    assert!(system.contains("Summarize"), "system prompt should follow");
}

// =============================================================================
// replace_messages tests
// =============================================================================

#[test]
fn replace_messages_preserves_current_input() {
    let mut state = ResponsesState::from_request_body(json!({
        "model": "gpt-4o",
        "input": "What's next?"
    }));
    state.history_rehydrated = true;
    state
        .messages
        .insert(0, json!({"role": "user", "content": "old question"}));
    state
        .messages
        .insert(1, json!({"role": "assistant", "content": "old answer"}));
    state
        .persisted_messages
        .insert(0, json!({"role": "user", "content": "old question"}));
    state
        .persisted_messages
        .insert(1, json!({"role": "assistant", "content": "old answer"}));

    let compaction_item = build_compaction_item("compact_test", "Summary of old conversation.", DEFAULT_SUMMARY_PREFIX);
    replace_messages(&mut state, compaction_item);

    assert_eq!(state.messages.len(), 2, "should have compaction + current input");
    assert_eq!(state.messages[0]["type"], "compaction");
    assert_eq!(state.messages[0]["id"], "compact_test");
    assert!(state.messages[0].get("encrypted_content").is_some());
    assert_eq!(state.persisted_messages.len(), 2);
    assert_eq!(state.persisted_messages[0]["type"], "compaction");
}

#[test]
fn replace_messages_direct_input_does_not_duplicate_conversation() {
    let mut state = ResponsesState::from_request_body(json!({
        "model": "gpt-4o",
        "input": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
            {"role": "user", "content": "third"}
        ]
    }));
    assert_eq!(state.input, state.messages, "direct input: input == messages");

    let compaction_item = build_compaction_item("compact_direct", "Summary.", DEFAULT_SUMMARY_PREFIX);
    replace_messages(&mut state, compaction_item);

    assert_eq!(state.messages.len(), 1, "direct input should only have compaction item");
    assert_eq!(state.messages[0]["type"], "compaction");
    assert_eq!(state.persisted_messages.len(), 1);
}

// =============================================================================
// get_token_count tests
// =============================================================================

#[test]
fn token_count_returns_some_for_known_encoding() {
    let text = build_conversation_text(&[json!({"role": "user", "content": "Hello world"})]);
    let count = get_token_count(&text, "cl100k_base");
    assert!(count.is_some());
    assert!(count.unwrap() > 0);
}

#[test]
fn token_count_returns_none_for_unknown_encoding() {
    let text = build_conversation_text(&[json!({"role": "user", "content": "Hello"})]);
    assert!(get_token_count(&text, "unknown_encoding").is_none());
}

#[test]
fn token_count_supports_o200k() {
    let text = build_conversation_text(&[json!({"role": "user", "content": "Hello world"})]);
    let count = get_token_count(&text, "o200k_base");
    assert!(count.is_some());
    assert!(count.unwrap() > 0);
}

// =============================================================================
// build_conversation_text with tool items
// =============================================================================

#[test]
fn conversation_text_includes_function_call() {
    let messages = vec![
        json!({"role": "user", "content": "What's the weather?"}),
        json!({"type": "function_call", "name": "get_weather", "arguments": "{\"city\":\"NYC\"}"}),
    ];
    let text = build_conversation_text(&messages);
    assert!(text.contains("function_call: get_weather({\"city\":\"NYC\"})"));
}

#[test]
fn conversation_text_includes_function_call_output() {
    let messages = vec![json!({"type": "function_call_output", "call_id": "call_1", "output": "{\"temp\":72}"})];
    let text = build_conversation_text(&messages);
    assert!(text.contains("function_call_output: {\"temp\":72}"));
}

#[test]
fn conversation_text_full_tool_round_trip() {
    let messages = vec![
        json!({"role": "user", "content": "What's the weather in NYC?"}),
        json!({"type": "function_call", "name": "get_weather", "arguments": "{\"city\":\"NYC\"}"}),
        json!({"type": "function_call_output", "call_id": "call_1", "output": "{\"temp\":72}"}),
        json!({"role": "assistant", "content": "It's 72°F in NYC."}),
    ];
    let text = build_conversation_text(&messages);
    assert!(text.contains("user: What's the weather in NYC?"));
    assert!(text.contains("function_call: get_weather("));
    assert!(text.contains("function_call_output: {\"temp\":72}"));
    assert!(text.contains("assistant: It's 72°F in NYC."));
}

// =============================================================================
// build_conversation_text with compaction items
// =============================================================================

#[test]
fn conversation_text_includes_compaction_summary() {
    let item = build_compaction_item("compact_1", "Prior context about widgets.", DEFAULT_SUMMARY_PREFIX);
    let messages = vec![item, json!({"role": "user", "content": "Tell me more"})];
    let text = build_conversation_text(&messages);
    assert!(text.contains("[previous context summary]: Prior context about widgets."));
    assert!(text.contains("user: Tell me more"));
}

#[test]
fn conversation_text_skips_empty_compaction_summary() {
    let item = build_compaction_item("compact_2", "", DEFAULT_SUMMARY_PREFIX);
    let messages = vec![item, json!({"role": "user", "content": "Hello"})];
    let text = build_conversation_text(&messages);
    assert!(!text.contains("context summary"));
    assert!(text.contains("user: Hello"));
}

// =============================================================================
// on_callout_error: open/closed failure mode
// =============================================================================

fn make_filter(failure_mode: &str) -> CompactFilter {
    let yaml = serde_yaml::from_str::<serde_yaml::Value>(&format!(
        "inference_url: http://localhost/v1/chat/completions\ncallout_failure_mode: {failure_mode}"
    ))
    .unwrap();
    let cfg: CompactFilterConfig = serde_yaml::from_value(yaml).unwrap();
    let validated = build_config(&cfg).unwrap();
    CompactFilter {
        client: SubRequestClient::new(praxis_core::subrequest::SubRequestConnector::new(1, None)),
        config: validated,
    }
}

#[test]
fn callout_error_open_mode_skips_compaction() {
    let filter = make_filter("open");
    let result = filter.on_callout_error("something went wrong", false);
    assert!(result.is_ok());
    assert!(result.unwrap().is_none(), "open mode should skip compaction");
}

#[test]
fn callout_error_closed_mode_rejects_request() {
    let filter = make_filter("closed");
    let result = filter.on_callout_error("something went wrong", false);
    assert!(result.is_err(), "closed mode should reject the request");
}

#[test]
fn parse_failure_open_mode_skips_compaction() {
    let filter = make_filter("open");
    let bad_body = b"not valid json";
    let result = parse_summarization_response(bad_body)
        .map(Some)
        .or_else(|_| filter.on_callout_error("failed to parse summarization response", false));
    assert!(result.is_ok());
    assert!(result.unwrap().is_none());
}

#[test]
fn parse_failure_closed_mode_rejects_request() {
    let filter = make_filter("closed");
    let bad_body = b"not valid json";
    let result = parse_summarization_response(bad_body)
        .map(Some)
        .or_else(|_| filter.on_callout_error("failed to parse summarization response", false));
    assert!(result.is_err());
}

// =============================================================================
// non-2xx summarization response respects callout_failure_mode
// =============================================================================

#[test]
fn non_2xx_response_open_mode_skips_compaction() {
    let filter = make_filter("open");
    let resp = subrequest::SubResponse {
        status: 503,
        headers: http::HeaderMap::new(),
        body: Bytes::from_static(b"service unavailable"),
    };
    let result = filter.handle_subrequest_result(Ok(resp), false);
    assert!(result.is_ok());
    assert!(result.unwrap().is_none(), "open mode should skip compaction on non-2xx");
}

#[test]
fn non_2xx_response_closed_mode_rejects_request() {
    let filter = make_filter("closed");
    let resp = subrequest::SubResponse {
        status: 429,
        headers: http::HeaderMap::new(),
        body: Bytes::from_static(b"rate limited"),
    };
    let result = filter.handle_subrequest_result(Ok(resp), false);
    assert!(result.is_err(), "closed mode should reject on non-2xx");
}

// =============================================================================
// previous_usage fast-path
// =============================================================================

#[test]
fn previous_usage_total_returns_total_tokens() {
    let mut state = ResponsesState::from_request_body(json!({"model": "gpt-4o", "input": "Hi"}));
    state.previous_usage = Some(json!({"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}));
    assert_eq!(previous_usage_total(&state), Some(150));
}

#[test]
fn previous_usage_total_returns_none_when_absent() {
    let state = ResponsesState::from_request_body(json!({"model": "gpt-4o", "input": "Hi"}));
    assert_eq!(previous_usage_total(&state), None);
}

#[test]
fn previous_usage_total_returns_none_when_null() {
    let mut state = ResponsesState::from_request_body(json!({"model": "gpt-4o", "input": "Hi"}));
    state.previous_usage = Some(json!({"input_tokens": 100}));
    assert_eq!(previous_usage_total(&state), None);
}

#[test]
fn should_compact_uses_previous_usage_when_available() {
    let mut state = ResponsesState::from_request_body(json!({
        "model": "gpt-4o",
        "input": "Hello",
        "context_management": [{"type": "compaction", "compact_threshold": 100}]
    }));
    state.messages = vec![json!({"role": "user", "content": "Hi"})];
    state.previous_usage = Some(json!({"total_tokens": 200}));
    let result = should_compact(&state, "cl100k_base");
    assert!(result.is_some(), "should compact when previous_usage exceeds threshold");
}

#[test]
fn should_compact_skips_when_previous_usage_below_threshold() {
    let mut state = ResponsesState::from_request_body(json!({
        "model": "gpt-4o",
        "input": "Hello",
        "context_management": [{"type": "compaction", "compact_threshold": 500}]
    }));
    state.messages = vec![json!({"role": "user", "content": "Hi"})];
    state.previous_usage = Some(json!({"total_tokens": 50}));
    let result = should_compact(&state, "cl100k_base");
    assert!(result.is_none(), "should skip when previous_usage is below threshold");
}

// =============================================================================
// direct input should_compact
// =============================================================================

#[test]
fn direct_input_should_compact_uses_tiktoken() {
    let state = ResponsesState::from_request_body(json!({
        "model": "gpt-4o",
        "input": [
            {"role": "user", "content": "x".repeat(5000)}
        ],
        "context_management": [{"type": "compaction", "compact_threshold": 50}]
    }));
    let result = should_compact(&state, "cl100k_base");
    assert!(
        result.is_some(),
        "direct input exceeding threshold should trigger compaction"
    );
}

#[test]
fn direct_input_should_not_compact_below_threshold() {
    let state = ResponsesState::from_request_body(json!({
        "model": "gpt-4o",
        "input": [{"role": "user", "content": "Hi"}],
        "context_management": [{"type": "compaction", "compact_threshold": 50000}]
    }));
    let result = should_compact(&state, "cl100k_base");
    assert!(result.is_none(), "direct input below threshold should skip compaction");
}

#[test]
fn tiktoken_fallback_includes_instructions_and_tools_in_count() {
    let state = ResponsesState::from_request_body(json!({
        "model": "gpt-4o",
        "input": [{"role": "user", "content": "short"}],
        "instructions": "Very long system prompt ".repeat(5000),
        "tools": [{"type": "function", "name": "f", "description": "d ".repeat(5000)}],
        "context_management": [{"type": "compaction", "compact_threshold": 50}]
    }));
    let result = should_compact(&state, "cl100k_base");
    assert!(
        result.is_some(),
        "tiktoken path should include instructions and tool definitions in token count"
    );
}

// =============================================================================
// parse_compact_request_body
// =============================================================================

#[test]
fn parse_compact_request_body_valid() {
    let body = Some(Bytes::from(
        serde_json::to_vec(&json!({
            "response_id": "resp_abc",
            "model": "gpt-4o",
            "instructions": "Be concise"
        }))
        .unwrap(),
    ));
    let req = parse_compact_request_body(&body).unwrap();
    assert_eq!(req.response_id, "resp_abc");
    assert_eq!(req.model.as_deref(), Some("gpt-4o"));
    assert_eq!(req.instructions.as_deref(), Some("Be concise"));
}

#[test]
fn parse_compact_request_body_minimal() {
    let body = Some(Bytes::from(
        serde_json::to_vec(&json!({"response_id": "resp_xyz"})).unwrap(),
    ));
    let req = parse_compact_request_body(&body).unwrap();
    assert_eq!(req.response_id, "resp_xyz");
    assert!(req.model.is_none());
    assert!(req.instructions.is_none());
}

#[test]
fn parse_compact_request_body_empty() {
    assert!(parse_compact_request_body(&None).is_err());
}

#[test]
fn parse_compact_request_body_invalid_json() {
    let body = Some(Bytes::from_static(b"not json"));
    assert!(parse_compact_request_body(&body).is_err());
}

#[test]
fn parse_compact_request_body_missing_response_id() {
    let body = Some(Bytes::from(serde_json::to_vec(&json!({"model": "gpt-4o"})).unwrap()));
    assert!(parse_compact_request_body(&body).is_err());
}

#[test]
fn parse_compact_request_body_empty_response_id() {
    let body = Some(Bytes::from(
        serde_json::to_vec(&json!({"response_id": "", "model": "gpt-4o"})).unwrap(),
    ));
    assert!(parse_compact_request_body(&body).is_err());
}

// =============================================================================
// extract_stored_messages
// =============================================================================

#[test]
fn extract_stored_messages_returns_messages() {
    let record = ResponseRecord {
        id: "resp_1".to_owned(),
        tenant_id: "default".to_owned(),
        created_at: 0,
        model: "gpt-4o".to_owned(),
        response_object: json!({}),
        input: json!([]),
        messages: json!([
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"}
        ]),
    };
    let msgs = extract_stored_messages(record).unwrap();
    assert_eq!(msgs.len(), 2);
}

#[test]
fn extract_stored_messages_empty_array() {
    let record = ResponseRecord {
        id: "resp_1".to_owned(),
        tenant_id: "default".to_owned(),
        created_at: 0,
        model: "gpt-4o".to_owned(),
        response_object: json!({}),
        input: json!([]),
        messages: json!([]),
    };
    assert!(extract_stored_messages(record).is_err());
}

#[test]
fn extract_stored_messages_not_array() {
    let record = ResponseRecord {
        id: "resp_1".to_owned(),
        tenant_id: "default".to_owned(),
        created_at: 0,
        model: "gpt-4o".to_owned(),
        response_object: json!({}),
        input: json!([]),
        messages: json!("not an array"),
    };
    assert!(extract_stored_messages(record).is_err());
}

// =============================================================================
// canonical round-trip
// =============================================================================

#[test]
fn compaction_item_round_trips_through_canonical_replay() {
    use crate::openai::responses::canonical_openresponses_replay_item;
    let item = build_compaction_item("compact_rt", "Summary text.", DEFAULT_SUMMARY_PREFIX);
    let replayed = canonical_openresponses_replay_item(&item);
    assert!(replayed.is_some(), "compaction item should be replayable");
    let replayed = replayed.unwrap();
    assert_eq!(replayed["type"], "compaction");
    assert_eq!(replayed["id"], "compact_rt");
    assert!(replayed.get("encrypted_content").is_some());
}

// =============================================================================
// is_compactable tests
// =============================================================================

#[test]
fn is_compactable_returns_false_when_no_state() {
    assert!(!is_compactable(None));
}

#[test]
fn is_compactable_returns_true_when_rehydrated() {
    let mut state = ResponsesState::from_request_body(json!({
        "model": "gpt-4o",
        "input": "Hello"
    }));
    state.history_rehydrated = true;
    assert!(is_compactable(Some(&state)));
}

#[test]
fn is_compactable_returns_false_for_direct_input_with_compaction_config() {
    let state = ResponsesState::from_request_body(json!({
        "model": "gpt-4o",
        "input": [{"role": "user", "content": "Hello"}],
        "context_management": [{"type": "compaction", "compact_threshold": 100}]
    }));
    assert!(!state.history_rehydrated, "precondition: not rehydrated");
    assert!(!is_compactable(Some(&state)));
}

#[test]
fn is_compactable_returns_false_without_rehydration_or_config() {
    let state = ResponsesState::from_request_body(json!({
        "model": "gpt-4o",
        "input": "Hello"
    }));
    assert!(!state.history_rehydrated, "precondition: not rehydrated");
    assert!(!is_compactable(Some(&state)));
}

#[test]
fn is_compactable_returns_false_with_non_compaction_config() {
    let state = ResponsesState::from_request_body(json!({
        "model": "gpt-4o",
        "input": "Hello",
        "context_management": [{"type": "truncation", "max_tokens": 4096}]
    }));
    assert!(!is_compactable(Some(&state)));
}
