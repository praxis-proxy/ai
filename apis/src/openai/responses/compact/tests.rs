// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

use serde_json::json;

use super::*;
use crate::callout_policy::OnFailure;

// =============================================================================
// Config tests
// =============================================================================

fn base_config() -> CompactFilterConfig {
    CompactFilterConfig {
        allow_pre_security_callout: true,
        inference_url: "http://localhost:11434/v1/chat/completions".to_owned(),
        default_model: "gpt-4o-mini".to_owned(),
        tiktoken_encoding: "cl100k_base".to_owned(),
        timeout_ms: None,
        on_failure: None,
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
    assert_eq!(cfg.callout.on_failure, OnFailure::Closed);
    assert_eq!(cfg.callout.status_on_error, 502);
}

#[test]
fn build_config_rejects_missing_pre_security_ack() {
    let mut cfg = base_config();
    cfg.allow_pre_security_callout = false;
    let err = build_config(&cfg).unwrap_err();
    assert!(
        err.to_string().contains("allow_pre_security_callout"),
        "should mention allow_pre_security_callout: {err}"
    );
}

#[test]
fn from_config_missing_pre_security_ack() {
    let yaml =
        serde_yaml::from_str::<serde_yaml::Value>("inference_url: http://localhost/v1/chat/completions").unwrap();
    let err = CompactFilter::from_config(&yaml)
        .err()
        .expect("should fail without allow_pre_security_callout");
    assert!(
        err.to_string().contains("allow_pre_security_callout"),
        "should mention allow_pre_security_callout: {err}"
    );
}

#[test]
fn from_config_accepts_pre_security_ack() {
    let yaml = serde_yaml::from_str::<serde_yaml::Value>(
        "allow_pre_security_callout: true\ninference_url: http://localhost/v1/chat/completions",
    )
    .unwrap();
    assert!(
        CompactFilter::from_config(&yaml).is_ok(),
        "explicit allow_pre_security_callout should construct"
    );
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
    cfg.on_failure = Some(OnFailure::Open);
    cfg.status_on_error = Some(503);
    let validated = build_config(&cfg).unwrap();
    assert_eq!(validated.callout.timeout_ms, 60_000);
    assert_eq!(validated.callout.on_failure, OnFailure::Open);
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

// =============================================================================
// build_compaction_item tests
// =============================================================================

#[test]
fn compaction_item_has_correct_shape() {
    use base64::Engine as _;
    let item = build_compaction_item("compact_abc123", "This is a summary.");
    assert_eq!(item["type"], "compaction");
    assert_eq!(item["id"], "compact_abc123");
    let encoded = item["encrypted_content"].as_str().unwrap();
    let decoded = base64::engine::general_purpose::STANDARD.decode(encoded).unwrap();
    assert_eq!(String::from_utf8(decoded).unwrap(), "This is a summary.");
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

    let compaction_item = build_compaction_item("compact_test", "Summary of old conversation.");
    replace_messages(&mut state, compaction_item);

    assert_eq!(state.messages.len(), 2, "should have compaction + current input");
    assert_eq!(state.messages[0]["type"], "compaction");
    assert_eq!(state.messages[0]["id"], "compact_test");
    assert!(state.messages[0].get("encrypted_content").is_some());
    assert_eq!(state.persisted_messages.len(), 2);
    assert_eq!(state.persisted_messages[0]["type"], "compaction");
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
    let item = build_compaction_item("compact_1", "Prior context about widgets.");
    let messages = vec![item, json!({"role": "user", "content": "Tell me more"})];
    let text = build_conversation_text(&messages);
    assert!(text.contains("[previous context summary]: Prior context about widgets."));
    assert!(text.contains("user: Tell me more"));
}

#[test]
fn conversation_text_skips_empty_compaction_summary() {
    let item = build_compaction_item("compact_2", "");
    let messages = vec![item, json!({"role": "user", "content": "Hello"})];
    let text = build_conversation_text(&messages);
    assert!(!text.contains("context summary"));
    assert!(text.contains("user: Hello"));
}

// =============================================================================
// on_callout_error: open/closed failure mode
// =============================================================================

fn make_filter(on_failure: &str) -> CompactFilter {
    let yaml = serde_yaml::from_str::<serde_yaml::Value>(&format!(
        "allow_pre_security_callout: true\ninference_url: http://localhost/v1/chat/completions\non_failure: {on_failure}"
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
// non-2xx summarization response respects on_failure
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
// build_context_overhead_text tests
// =============================================================================

#[test]
fn overhead_text_empty_without_instructions_or_tools() {
    let state = ResponsesState::from_request_body(json!({
        "model": "gpt-4o",
        "input": "Hello"
    }));
    assert!(build_context_overhead_text(&state).is_empty());
}

#[test]
fn overhead_text_includes_instructions() {
    let state = ResponsesState::from_request_body(json!({
        "model": "gpt-4o",
        "input": "Hello",
        "instructions": "You are a helpful assistant."
    }));
    let text = build_context_overhead_text(&state);
    assert!(text.contains("You are a helpful assistant."));
}

#[test]
fn overhead_text_includes_tool_definitions() {
    let state = ResponsesState::from_request_body(json!({
        "model": "gpt-4o",
        "input": "Hello",
        "tools": [{
            "type": "function",
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        }]
    }));
    let text = build_context_overhead_text(&state);
    assert!(text.contains("get_weather"));
}

#[test]
fn overhead_text_includes_both_instructions_and_tools() {
    let state = ResponsesState::from_request_body(json!({
        "model": "gpt-4o",
        "input": "Hello",
        "instructions": "Be concise.",
        "tools": [{"type": "function", "name": "search", "parameters": {}}]
    }));
    let text = build_context_overhead_text(&state);
    assert!(text.contains("Be concise."));
    assert!(text.contains("search"));
}

#[test]
fn overhead_tokens_count_with_get_token_count() {
    let state = ResponsesState::from_request_body(json!({
        "model": "gpt-4o",
        "input": "Hello",
        "instructions": "You are a helpful assistant that answers questions concisely."
    }));
    let text = build_context_overhead_text(&state);
    let count = get_token_count(&text, "cl100k_base");
    assert!(count.is_some());
    assert!(count.unwrap() > 0);
}

#[test]
fn should_compact_accounts_for_overhead() {
    let long_instructions = "x".repeat(5000);
    let mut state = ResponsesState::from_request_body(json!({
        "model": "gpt-4o",
        "input": "Hello",
        "instructions": long_instructions,
        "context_management": [{"type": "compaction", "compact_threshold": 50}]
    }));
    state.messages = vec![json!({"role": "user", "content": "Hi"})];
    let result = should_compact(&state, "cl100k_base");
    assert!(
        result.is_some(),
        "overhead from long instructions should push total above threshold"
    );
}

// =============================================================================
// canonical round-trip
// =============================================================================

#[test]
fn compaction_item_round_trips_through_canonical_replay() {
    use crate::openai::responses::canonical_openresponses_replay_item;
    let item = build_compaction_item("compact_rt", "Summary text.");
    let replayed = canonical_openresponses_replay_item(&item);
    assert!(replayed.is_some(), "compaction item should be replayable");
    let replayed = replayed.unwrap();
    assert_eq!(replayed["type"], "compaction");
    assert_eq!(replayed["id"], "compact_rt");
    assert!(replayed.get("encrypted_content").is_some());
}
