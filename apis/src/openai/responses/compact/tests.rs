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
        allow_private_inference_url: true,
        allow_pre_security_callout: true,
        inference_url: "http://localhost:11434/v1/chat/completions".to_owned(),
        default_model: "gpt-4o-mini".to_owned(),
        tiktoken_encoding: "cl100k_base".to_owned(),
        summary_prefix: None,
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
        "allow_pre_security_callout: true\ninference_url: http://localhost/v1/chat/completions\nallow_private_inference_url: true",
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
fn private_inference_target_requires_explicit_opt_in() {
    let mut cfg = base_config();
    cfg.allow_private_inference_url = false;
    let error = build_config(&cfg).expect_err("loopback inference target must require opt-in");
    assert!(error.to_string().contains("localhost"), "unexpected error: {error}");
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
    let params = extract_compaction_config(&cm).unwrap().unwrap();
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
    let params = extract_compaction_config(&cm).unwrap().unwrap();
    assert_eq!(params.compact_threshold, 100_000);
    assert_eq!(params.compaction_model.as_deref(), Some("gpt-4o"));
}

#[test]
fn extract_compaction_config_no_compaction_entry() {
    let cm = Some(json!([{"type": "truncation", "max_tokens": 4096}]));
    assert!(extract_compaction_config(&cm).unwrap().is_none());
}

#[test]
fn extract_compaction_config_none() {
    assert!(extract_compaction_config(&None).unwrap().is_none());
}

#[test]
fn extract_compaction_config_null_is_treated_as_absent() {
    let cm = Some(json!(null));
    assert!(
        extract_compaction_config(&cm).unwrap().is_none(),
        "explicit null context_management should behave like an omitted field"
    );
}

#[test]
fn extract_compaction_config_non_array_returns_error() {
    let cm = Some(json!({"type": "compaction", "compact_threshold": 1000}));
    let err = extract_compaction_config(&cm).unwrap_err();
    assert!(
        err.contains("context_management must be an array"),
        "a present but non-array context_management must be rejected, got: {err}"
    );
}

#[test]
fn extract_compaction_config_empty_array() {
    let cm = Some(json!([]));
    assert!(extract_compaction_config(&cm).unwrap().is_none());
}

#[test]
fn extract_compaction_config_missing_threshold_returns_error() {
    let cm = Some(json!([{"type": "compaction"}]));
    let err = extract_compaction_config(&cm).unwrap_err();
    assert!(err.contains("compact_threshold"));
}

#[test]
fn extract_compaction_config_null_threshold_returns_error() {
    let cm = Some(json!([{"type": "compaction", "compact_threshold": null}]));
    let err = extract_compaction_config(&cm).unwrap_err();
    assert!(err.contains("compact_threshold"));
}

#[test]
fn extract_compaction_config_float_threshold_returns_error() {
    let cm = Some(json!([{"type": "compaction", "compact_threshold": 0.9}]));
    let err = extract_compaction_config(&cm).unwrap_err();
    assert!(err.contains("compact_threshold"));
}

#[test]
fn extract_compaction_config_string_threshold_returns_error() {
    let cm = Some(json!([{"type": "compaction", "compact_threshold": "1000"}]));
    let err = extract_compaction_config(&cm).unwrap_err();
    assert!(err.contains("compact_threshold"));
}

#[test]
fn extract_compaction_config_threshold_below_minimum_returns_error() {
    let cm = Some(json!([{"type": "compaction", "compact_threshold": 999}]));
    let err = extract_compaction_config(&cm).unwrap_err();
    assert!(err.contains("at least 1000"));
}

#[test]
fn extract_compaction_config_minimum_threshold_succeeds() {
    let cm = Some(json!([{"type": "compaction", "compact_threshold": 1000}]));
    let params = extract_compaction_config(&cm).unwrap().unwrap();
    assert_eq!(params.compact_threshold, 1000);
}

#[test]
fn extract_compaction_config_non_string_model_returns_error() {
    let cm = Some(json!([{"type": "compaction", "compact_threshold": 5000, "compaction_model": 42}]));
    let err = extract_compaction_config(&cm).unwrap_err();
    assert!(err.contains("compaction_model"));
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
    let result = parse_summarization_response(&body).unwrap();
    assert_eq!(result.content, "Here is the summary.");
    assert!(result.usage.is_none(), "no usage reported by the callout");
}

#[test]
fn parse_response_captures_usage() {
    let response = json!({
        "choices": [{"message": {"role": "assistant", "content": "Summary."}}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}
    });
    let body = serde_json::to_vec(&response).unwrap();
    let result = parse_summarization_response(&body).unwrap();
    assert_eq!(result.usage.unwrap()["total_tokens"], 60);
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
// usage mapping tests
// =============================================================================

#[test]
fn map_chat_usage_maps_all_fields() {
    let usage = json!({
        "prompt_tokens": 50,
        "completion_tokens": 10,
        "total_tokens": 60,
        "prompt_tokens_details": {"cached_tokens": 8},
        "completion_tokens_details": {"reasoning_tokens": 4}
    });
    let mapped = map_chat_usage(&usage);
    assert_eq!(mapped["input_tokens"], 50);
    assert_eq!(mapped["output_tokens"], 10);
    assert_eq!(mapped["total_tokens"], 60);
    assert_eq!(mapped["input_tokens_details"]["cached_tokens"], 8);
    assert_eq!(mapped["input_tokens_details"]["cache_write_tokens"], 0);
    assert_eq!(mapped["output_tokens_details"]["reasoning_tokens"], 4);
}

#[test]
fn map_chat_usage_defaults_missing_details_to_zero() {
    let usage = json!({"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10});
    let mapped = map_chat_usage(&usage);
    assert_eq!(mapped["input_tokens_details"]["cached_tokens"], 0);
    assert_eq!(mapped["output_tokens_details"]["reasoning_tokens"], 0);
}

#[test]
fn map_chat_usage_derives_total_when_absent() {
    let usage = json!({"prompt_tokens": 12, "completion_tokens": 5});
    let mapped = map_chat_usage(&usage);
    assert_eq!(mapped["total_tokens"], 17, "total falls back to input + output");
}

#[test]
fn build_compaction_usage_prefers_callout_usage() {
    let summary = Summarization {
        content: "short".to_owned(),
        usage: Some(json!({"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120})),
    };
    let messages = vec![json!({"role": "user", "content": "a much longer conversation body"})];
    let usage = build_compaction_usage(&messages, Some(&summary), "cl100k_base");
    // Reported usage wins over any tiktoken estimate of the message/summary text.
    assert_eq!(usage["input_tokens"], 100);
    assert_eq!(usage["output_tokens"], 20);
    assert_eq!(usage["total_tokens"], 120);
}

#[test]
fn build_compaction_usage_falls_back_to_tiktoken_when_usage_absent() {
    let summary = Summarization {
        content: "a produced summary".to_owned(),
        usage: None,
    };
    let messages = vec![json!({"role": "user", "content": "the source conversation text"})];
    let usage = build_compaction_usage(&messages, Some(&summary), "cl100k_base");
    assert!(
        usage["input_tokens"].as_u64().unwrap() > 0,
        "estimates source conversation"
    );
    assert!(
        usage["output_tokens"].as_u64().unwrap() > 0,
        "estimates produced summary"
    );
    assert_eq!(
        usage["total_tokens"].as_u64().unwrap(),
        usage["input_tokens"].as_u64().unwrap() + usage["output_tokens"].as_u64().unwrap()
    );
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
    assert_eq!(
        state.messages[1]["content"], "What's next?",
        "current-turn tail from messages must be kept"
    );
    assert_eq!(state.persisted_messages.len(), 2);
    assert_eq!(state.persisted_messages[0]["type"], "compaction");
    assert_eq!(
        state.persisted_messages[1]["content"], "What's next?",
        "current-turn tail from persisted_messages must be kept"
    );
}

#[test]
fn replace_messages_keeps_each_list_current_turn_independently() {
    let mut state = ResponsesState::from_request_body(json!({
        "model": "gpt-4o",
        "input": [{"type": "message", "role": "user", "content": "from-input"}]
    }));
    state.messages = vec![
        json!({"role": "user", "content": "hist-a"}),
        json!({"role": "user", "content": "from-messages"}),
    ];
    state.persisted_messages = vec![
        json!({"role": "user", "content": "hist-b1"}),
        json!({"role": "user", "content": "hist-b2"}),
        json!({"role": "user", "content": "from-persisted"}),
    ];

    replace_messages(&mut state, build_compaction_item("c1", "sum", DEFAULT_SUMMARY_PREFIX));

    assert_eq!(state.messages.len(), 2);
    assert_eq!(state.messages[1]["content"], "from-messages");
    assert_eq!(state.persisted_messages.len(), 2);
    assert_eq!(state.persisted_messages[1]["content"], "from-persisted");
    assert_eq!(
        state.input[0]["content"], "from-input",
        "state.input must not be used to rebuild the current turn"
    );
}

#[test]
fn compaction_preserves_resolved_file_data_instead_of_file_url() {
    const FILE_URL: &str = "https://files.internal/secret.bin";
    let mut state = ResponsesState::from_request_body(json!({
        "model": "gpt-4o",
        "previous_response_id": "resp_prev",
        "context_management": [{"type": "compaction", "compact_threshold": 1000}],
        "input": [{
            "type": "message",
            "role": "user",
            "content": [{"type": "input_file", "file_url": FILE_URL}]
        }]
    }));
    state.history_rehydrated = true;
    let history = json!({"role": "user", "content": "earlier turn long enough to compact"});
    state.messages.insert(0, history.clone());
    state.persisted_messages.insert(0, history);

    let resolved_item = json!({
        "type": "message",
        "role": "user",
        "content": [{"type": "input_file", "file_data": "SGVsbG8="}]
    });
    state.request_body["input"] = json!([resolved_item.clone()]);
    let tail = state.messages.len() - state.input.len();
    state.messages[tail] = resolved_item.clone();
    state.persisted_messages[tail] = resolved_item;

    assert_eq!(
        state.input[0]["content"][0]["file_url"], FILE_URL,
        "state.input stays the original client payload"
    );

    replace_messages(
        &mut state,
        build_compaction_item("compact_1", "summary", DEFAULT_SUMMARY_PREFIX),
    );

    assert_eq!(state.messages[0]["type"], "compaction");
    let current = &state.messages[1];
    assert_eq!(
        current["content"][0]["file_data"], "SGVsbG8=",
        "resolved file_data must survive compaction"
    );
    assert!(
        current["content"][0].get("file_url").is_none(),
        "original file_url must not be restored from state.input"
    );
    assert_eq!(
        state.persisted_messages[1]["content"][0]["file_data"], "SGVsbG8=",
        "persisted current-turn tail must keep resolved file_data"
    );
    assert_eq!(
        state.input[0]["content"][0]["file_url"], FILE_URL,
        "state.input remains the unmodified client payload"
    );
}

#[test]
fn compaction_preserves_extracted_input_text_instead_of_input_file() {
    let mut state = ResponsesState::from_request_body(json!({
        "model": "gpt-4o",
        "previous_response_id": "resp_prev",
        "context_management": [{"type": "compaction", "compact_threshold": 1000}],
        "input": [{
            "type": "message",
            "role": "user",
            "content": [{"type": "input_file", "filename": "notes.txt", "file_data": "c2VjcmV0"}]
        }]
    }));
    state.history_rehydrated = true;
    let history = json!({"role": "user", "content": "earlier turn"});
    state.messages.insert(0, history.clone());
    state.persisted_messages.insert(0, history);

    let extracted_item = json!({
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "secret"}]
    });
    state.request_body["input"] = json!([extracted_item.clone()]);
    let tail = state.messages.len() - state.input.len();
    state.messages[tail] = extracted_item.clone();
    state.persisted_messages[tail] = extracted_item;

    replace_messages(
        &mut state,
        build_compaction_item("compact_1", "summary", DEFAULT_SUMMARY_PREFIX),
    );

    let current = &state.messages[1];
    assert_eq!(
        current["content"][0]["type"], "input_text",
        "doc_extract rewrite must survive compaction"
    );
    assert_eq!(current["content"][0]["text"], "secret");
    assert_eq!(
        state.input[0]["content"][0]["type"], "input_file",
        "state.input stays the original input_file part"
    );
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

fn make_filter(on_failure: &str) -> CompactFilter {
    let yaml = serde_yaml::from_str::<serde_yaml::Value>(&format!(
        "allow_pre_security_callout: true\ninference_url: http://localhost/v1/chat/completions\nallow_private_inference_url: true\non_failure: {on_failure}"
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
    let result = filter.on_callout_error("something went wrong");
    assert!(result.is_ok());
    assert!(result.unwrap().is_none(), "open mode should skip compaction");
}

#[test]
fn callout_error_closed_mode_rejects_request() {
    let filter = make_filter("closed");
    let result = filter.on_callout_error("something went wrong");
    assert!(result.is_err(), "closed mode should reject the request");
}

#[test]
fn parse_failure_open_mode_skips_compaction() {
    let filter = make_filter("open");
    let bad_body = b"not valid json";
    let result = parse_summarization_response(bad_body)
        .map(Some)
        .or_else(|_| filter.on_callout_error("failed to parse summarization response"));
    assert!(result.is_ok());
    assert!(result.unwrap().is_none());
}

#[test]
fn parse_failure_closed_mode_rejects_request() {
    let filter = make_filter("closed");
    let bad_body = b"not valid json";
    let result = parse_summarization_response(bad_body)
        .map(Some)
        .or_else(|_| filter.on_callout_error("failed to parse summarization response"));
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
    let result = filter.handle_subrequest_result(Ok(resp));
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
    let result = filter.handle_subrequest_result(Ok(resp));
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
        "context_management": [{"type": "compaction", "compact_threshold": 1000}]
    }));
    state.messages = vec![json!({"role": "user", "content": "Hi"})];
    state.previous_usage = Some(json!({"total_tokens": 2000}));
    let result = should_compact(&state, "cl100k_base").unwrap();
    assert!(result.is_some(), "should compact when previous_usage exceeds threshold");
}

#[test]
fn should_compact_skips_when_previous_usage_below_threshold() {
    let mut state = ResponsesState::from_request_body(json!({
        "model": "gpt-4o",
        "input": "Hello",
        "context_management": [{"type": "compaction", "compact_threshold": 1000}]
    }));
    state.messages = vec![json!({"role": "user", "content": "Hi"})];
    state.previous_usage = Some(json!({"total_tokens": 500}));
    let result = should_compact(&state, "cl100k_base").unwrap();
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
            {"role": "user", "content": "word ".repeat(3000)}
        ],
        "context_management": [{"type": "compaction", "compact_threshold": 1000}]
    }));
    let result = should_compact(&state, "cl100k_base").unwrap();
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
    let result = should_compact(&state, "cl100k_base").unwrap();
    assert!(result.is_none(), "direct input below threshold should skip compaction");
}

#[test]
fn should_compact_accounts_for_overhead() {
    let long_instructions = "word ".repeat(1500);
    let mut state = ResponsesState::from_request_body(json!({
        "model": "gpt-4o",
        "input": "Hello",
        "instructions": long_instructions,
        "context_management": [{"type": "compaction", "compact_threshold": 1000}]
    }));
    state.messages = vec![json!({"role": "user", "content": "Hi"})];
    let result = should_compact(&state, "cl100k_base").unwrap();
    assert!(
        result.is_some(),
        "tiktoken path should include instructions and tool definitions in token count"
    );
}

#[test]
fn tiktoken_fallback_includes_instructions_and_tools_in_count() {
    let state = ResponsesState::from_request_body(json!({
        "model": "gpt-4o",
        "input": [{"role": "user", "content": "short"}],
        "instructions": "Very long system prompt ".repeat(5000),
        "tools": [{"type": "function", "name": "f", "description": "d ".repeat(5000)}],
        "context_management": [{"type": "compaction", "compact_threshold": 1000}]
    }));
    let result = should_compact(&state, "cl100k_base").unwrap();
    assert!(
        result.is_some(),
        "tiktoken path should include instructions and tool definitions in token count"
    );
}

// =============================================================================
// parse_compact_request_body
// =============================================================================

#[test]
fn parse_compact_request_body_with_previous_response_id() {
    let body = Some(Bytes::from(
        serde_json::to_vec(&json!({
            "model": "gpt-4o",
            "previous_response_id": "resp_abc",
            "instructions": "Be concise"
        }))
        .unwrap(),
    ));
    let req = parse_compact_request_body(&body).unwrap();
    assert_eq!(req.model, "gpt-4o");
    assert_eq!(req.previous_response_id.as_deref(), Some("resp_abc"));
    assert!(req.input.is_empty());
    assert_eq!(req.instructions.as_deref(), Some("Be concise"));
}

#[test]
fn parse_compact_request_body_with_string_input() {
    let body = Some(Bytes::from(
        serde_json::to_vec(&json!({"model": "gpt-4o", "input": "Summarize this"})).unwrap(),
    ));
    let req = parse_compact_request_body(&body).unwrap();
    assert_eq!(req.model, "gpt-4o");
    assert!(req.previous_response_id.is_none());
    assert_eq!(req.input.len(), 1, "string input coerces to one user message");
    assert_eq!(req.input[0]["role"], "user");
    assert_eq!(req.input[0]["content"], "Summarize this");
}

#[test]
fn parse_compact_request_body_with_array_input() {
    let body = Some(Bytes::from(
        serde_json::to_vec(&json!({
            "model": "gpt-4o",
            "input": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"}
            ]
        }))
        .unwrap(),
    ));
    let req = parse_compact_request_body(&body).unwrap();
    assert_eq!(req.input.len(), 2);
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
fn parse_compact_request_body_missing_model() {
    // `model` is required by the contract even when content is present.
    let body = Some(Bytes::from(serde_json::to_vec(&json!({"input": "hello"})).unwrap()));
    assert!(parse_compact_request_body(&body).is_err());
}

#[test]
fn parse_compact_request_body_empty_model() {
    let body = Some(Bytes::from(
        serde_json::to_vec(&json!({"model": "", "input": "hello"})).unwrap(),
    ));
    assert!(parse_compact_request_body(&body).is_err());
}

#[test]
fn parse_compact_request_body_missing_content() {
    // `model` alone, with neither `input` nor `previous_response_id`.
    let body = Some(Bytes::from(serde_json::to_vec(&json!({"model": "gpt-4o"})).unwrap()));
    assert!(parse_compact_request_body(&body).is_err());
}

// =============================================================================
// stored_message_array
// =============================================================================

#[test]
fn stored_message_array_returns_messages() {
    let messages = json!([
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"}
    ]);
    assert_eq!(stored_message_array(messages).len(), 2);
}

#[test]
fn stored_message_array_empty_array() {
    assert!(stored_message_array(json!([])).is_empty());
}

#[test]
fn stored_message_array_not_array() {
    assert!(stored_message_array(json!("not an array")).is_empty());
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
