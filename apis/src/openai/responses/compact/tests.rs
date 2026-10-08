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
        "prompt_tokens_details": {"cached_tokens": 8, "cache_write_tokens": 12},
        "completion_tokens_details": {"reasoning_tokens": 4}
    });
    let mapped = map_chat_usage(&usage);
    assert_eq!(mapped["input_tokens"], 50);
    assert_eq!(mapped["output_tokens"], 10);
    assert_eq!(mapped["total_tokens"], 60);
    assert_eq!(mapped["input_tokens_details"]["cached_tokens"], 8);
    assert_eq!(mapped["input_tokens_details"]["cache_write_tokens"], 12);
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
    // Parts join with no separator, matching the translation layer.
    assert!(text.contains("user: Part onePart two"), "{text}");
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
    // Parts are concatenated with no separator, matching the Responses-to-Chat
    // translation (so "pass" + "word" stays "password", not "pass word").
    let msg = json!({"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]});
    assert_eq!(extract_content(&msg), "ab");
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
    replace_messages(&mut state, &compaction_item);

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
        state.persisted_messages[0]["_praxis_local_compaction"], true,
        "private persisted history must retain local compaction provenance"
    );
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

    let compaction_item = build_compaction_item("c1", "sum", DEFAULT_SUMMARY_PREFIX);
    replace_messages(&mut state, &compaction_item);

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

    let compaction_item = build_compaction_item("compact_1", "summary", DEFAULT_SUMMARY_PREFIX);
    replace_messages(&mut state, &compaction_item);

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

    let compaction_item = build_compaction_item("compact_1", "summary", DEFAULT_SUMMARY_PREFIX);
    replace_messages(&mut state, &compaction_item);

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
fn conversation_text_includes_function_call_output_content_list() {
    // An array `output` (content-list) must contribute its text to the summary,
    // not be dropped the way an `as_str()`-only read would drop it. Parts are
    // joined with no separator, matching the translation layer ("pass" + "word"
    // stays "password", preserving the fact the backend would see).
    let messages = vec![json!({
        "type": "function_call_output",
        "call_id": "call_1",
        "output": [
            {"type": "input_text", "text": "pass"},
            {"type": "input_text", "text": "word"}
        ]
    })];
    let text = build_conversation_text(&messages);
    assert!(
        text.contains("function_call_output: password"),
        "parts join without a separator: {text}"
    );
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
        client: subrequest::isolated_client(1),
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

#[tokio::test]
#[cfg(feature = "store-sqlite")]
async fn explicit_compaction_loads_previous_response_only_for_exact_owner() {
    let backend: std::sync::Arc<dyn crate::store::PersistedStateBackend> = std::sync::Arc::new(
        crate::store::SqliteResponseStore::new("sqlite::memory:", "responses", "conversations", None, None, None)
            .await
            .unwrap(),
    );
    let owner = StateOwner::from_trusted_parts("tenant-a", "issuer-a", "alice").unwrap();
    let other = StateOwner::from_trusted_parts("tenant-a", "issuer-a", "bob").unwrap();
    backend
        .upsert_response(&ResponseRecord {
            id: "resp_private".to_owned(),
            owner: owner.clone(),
            created_at: 1_000,
            model: "gpt-4.1".to_owned(),
            response_object: json!({"id": "resp_private", "status": "completed"}),
            input: json!([]),
            messages: json!([{"role": "user", "content": "private"}]),
        })
        .await
        .unwrap();
    let registry = ResponseStoreRegistry::new();
    registry.register(&std::sync::Arc::from("default"), backend).unwrap();
    let request = parse_compact_request_body(&Some(Bytes::from_static(
        br#"{"model":"gpt-4.1","previous_response_id":"resp_private"}"#,
    )))
    .unwrap();

    let wrong_store = registry.get_scoped("default", &other).unwrap();
    let Err(FilterAction::Reject(rejection)) = collect_compact_messages(&wrong_store, &request).await else {
        panic!("wrong-owner compaction must fail before its callout");
    };
    assert_eq!(rejection.status, 404);

    let owner_store = registry.get_scoped("default", &owner).unwrap();
    let messages = collect_compact_messages(&owner_store, &request).await.unwrap();
    assert_eq!(messages, vec![json!({"role": "user", "content": "private"})]);
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
// Malformed-field validation (issue #1403)
// =============================================================================

// --- instructions ---

#[test]
fn parse_compact_instructions_accepts_string() {
    assert_eq!(
        parse_compact_instructions(Some(json!("Be concise")))
            .unwrap()
            .as_deref(),
        Some("Be concise")
    );
}

#[test]
fn parse_compact_instructions_absent_and_null_are_none() {
    assert!(parse_compact_instructions(None).unwrap().is_none());
    assert!(parse_compact_instructions(Some(Value::Null)).unwrap().is_none());
}

#[test]
fn parse_compact_instructions_wrong_type_is_rejected() {
    // Issue #1403 case 1: a numeric `instructions` must 400, not be dropped.
    assert_compact_rejected_400(
        parse_compact_request_body(&Some(compact_body(&json!({
            "model": "gpt-4o",
            "input": "INLINE-CONTENT",
            "instructions": 123
        })))),
        "instructions must be a string",
    );
}

// --- input ---

#[test]
fn parse_compact_input_wrong_type_is_rejected() {
    // Issue #1403: a wrong-typed input must 400 rather than becoming an empty list.
    assert_compact_rejected_400(
        parse_compact_request_body(&Some(compact_body(&json!({"model": "gpt-4o", "input": 123})))),
        "input must be a string or an array of items",
    );
}

#[test]
fn parse_compact_input_non_object_array_item_is_rejected() {
    // Issue #1403 case 2: `[123]` must 400 rather than contributing empty content.
    assert_compact_rejected_400(
        parse_compact_request_body(&Some(compact_body(&json!({"model": "gpt-4o", "input": [123]})))),
        "input[0] must be an object",
    );
}

#[test]
fn parse_compact_input_absent_and_null_are_empty_list() {
    assert!(parse_compact_input(None).unwrap().is_empty());
    assert!(parse_compact_input(Some(Value::Null)).unwrap().is_empty());
}

#[test]
fn parse_compact_input_empty_string_is_empty_list() {
    // An empty-content string is distinct from a wrong type; it is schema-valid
    // emptiness (tracked by #1139), not a 400.
    assert!(parse_compact_input(Some(json!(""))).unwrap().is_empty());
}

#[test]
fn parse_compact_input_object_array_items_are_accepted() {
    let items = parse_compact_input(Some(json!([
        {"role": "user", "content": "Hi"},
        {"type": "function_call", "name": "f", "arguments": "{}"}
    ])))
    .unwrap();
    assert_eq!(items.len(), 2, "well-formed object items pass validation");
}

#[test]
fn parse_compact_input_message_with_wrong_typed_content_is_rejected() {
    // Issue #1403 follow-up: an object item is not enough — a message whose
    // `content` is a number would be silently dropped to empty by the formatter,
    // so it must 400 before any callout or store write.
    assert_compact_rejected_400(
        parse_compact_request_body(&Some(compact_body(&json!({
            "model": "gpt-4o",
            "input": [{"role": "user", "content": 123}]
        })))),
        "input[0].content must be a string or an array",
    );
}

#[test]
fn parse_compact_input_content_part_non_object_is_rejected() {
    assert_compact_rejected_400(
        parse_compact_input(Some(json!([{"role": "user", "content": [123]}]))),
        "input[0].content[0] must be an object",
    );
}

#[test]
fn parse_compact_input_content_part_wrong_typed_text_is_rejected() {
    assert_compact_rejected_400(
        parse_compact_input(Some(json!([{
            "role": "user",
            "content": [{"type": "input_text", "text": 123}]
        }]))),
        "input[0].content[0].text must be a string",
    );
}

#[test]
fn parse_compact_input_text_part_with_null_or_absent_text_is_rejected() {
    // Issue #1403 follow-up: `input_text` / `output_text` require a string `text`;
    // a null or missing value would be silently dropped to empty by the formatter.
    assert_compact_rejected_400(
        parse_compact_input(Some(json!([{
            "role": "user",
            "content": [{"type": "input_text", "text": null}]
        }]))),
        "input[0].content[0].text must be a string",
    );
    assert_compact_rejected_400(
        parse_compact_input(Some(json!([{
            "role": "user",
            "content": [{"type": "input_text"}]
        }]))),
        "input[0].content[0].text must be a string",
    );
    assert_compact_rejected_400(
        parse_compact_input(Some(json!([{
            "type": "function_call_output",
            "output": [{"type": "output_text", "text": null}]
        }]))),
        "input[0].output[0].text must be a string",
    );
}

#[test]
fn parse_compact_input_multimodal_part_without_text_is_accepted() {
    // An image/file part legitimately carries no `text`; the formatter ignores
    // it, so validation must not reject it as malformed.
    let items = parse_compact_input(Some(json!([{
        "role": "user",
        "content": [
            {"type": "input_text", "text": "look"},
            {"type": "input_image", "image_url": "https://example.test/x.png"}
        ]
    }])))
    .unwrap();
    assert_eq!(items.len(), 1, "a text+image content list is well-formed");
}

#[test]
fn parse_compact_input_wrong_typed_item_type_is_rejected() {
    assert_compact_rejected_400(
        parse_compact_input(Some(json!([{"type": 123, "content": "hi"}]))),
        "input[0].type must be a string",
    );
}

#[test]
fn parse_compact_input_null_item_type_is_rejected() {
    // Issue #1403 follow-up: an explicit null `type` must not be read as "omitted"
    // and silently reinterpreted as a message; the item `type`, when present, must
    // be a concrete string.
    assert_compact_rejected_400(
        parse_compact_input(Some(json!([{"type": null, "content": "hi"}]))),
        "input[0].type must be a string",
    );
}

#[test]
fn parse_compact_input_null_type_item_reference_is_accepted() {
    // Issue #1403 follow-up: `ItemReferenceParam.type` is nullable in the schema
    // (`anyOf: [enum "item_reference", null]`), so a reference-shaped item that
    // carries an `id` but neither `role` nor `content` must be accepted with an
    // explicit null `type`, matching the already-accepted type-omitted form.
    let items = parse_compact_input(Some(json!([{"type": null, "id": "resp_123"}]))).unwrap();
    assert_eq!(
        items.len(),
        1,
        "a null-typed item_reference carries no field compact consumes"
    );
}

#[test]
fn parse_compact_input_null_type_without_id_is_rejected() {
    // Issue #1403 follow-up: the null-`type` tolerance is only for the item_reference
    // shape, which the schema requires to carry a string `id`. A null `type` with no
    // `id` is a malformed item, not a reference, and must 400 before any callout or
    // store write rather than fall through as an empty defaulted message.
    assert_compact_rejected_400(
        parse_compact_input(Some(json!([{"type": null}]))),
        "input[0].type must be a string",
    );
}

#[test]
fn parse_compact_input_null_content_part_type_is_rejected() {
    // Issue #1403 follow-up: a content part's `type` discriminator, when present,
    // must be a concrete string; a null type would escape the text-part check and
    // its null `text` would be silently dropped to empty.
    assert_compact_rejected_400(
        parse_compact_input(Some(
            json!([{"role": "user", "content": [{"type": null, "text": null}]}]),
        )),
        "input[0].content[0].type must be a string",
    );
}

#[test]
fn parse_compact_input_content_part_without_type_is_rejected() {
    // Issue #1403 follow-up: the content-part union requires a `type` discriminator;
    // a part with no type and a null `text` would escape the text-part check and be
    // dropped to empty, so the missing discriminator must 400.
    assert_compact_rejected_400(
        parse_compact_input(Some(json!([{"role": "user", "content": [{"text": null}]}]))),
        "input[0].content[0].type must be a string",
    );
}

#[test]
fn parse_compact_input_text_kind_part_with_null_text_is_rejected() {
    // Issue #1403 follow-up: the translator collapses the `text` kind (not just
    // `input_text`/`output_text`) and requires a string `text`, so a null value
    // must 400 rather than be dropped to empty.
    assert_compact_rejected_400(
        parse_compact_input(Some(
            json!([{"role": "user", "content": [{"type": "text", "text": null}]}]),
        )),
        "input[0].content[0].text must be a string",
    );
}

#[test]
fn parse_compact_input_additional_tools_item_without_content_is_accepted() {
    // Issue #1403 follow-up: an `additional_tools` item is explicitly typed and
    // carries a `role` but no `content`; it is NOT a message, so it must pass
    // through rather than be rejected for a missing required `content`.
    let items = parse_compact_input(Some(json!([
        {"type": "additional_tools", "role": "developer", "tools": []},
        {"role": "user", "content": "Hi"}
    ])))
    .unwrap();
    assert_eq!(
        items.len(),
        2,
        "a typed non-message item + a message are both well-formed"
    );
}

#[test]
fn parse_compact_input_message_with_wrong_typed_role_is_rejected() {
    assert_compact_rejected_400(
        parse_compact_input(Some(json!([{"role": 123, "content": "hi"}]))),
        "input[0].role must be a string",
    );
}

#[test]
fn parse_compact_input_function_call_wrong_typed_fields_are_rejected() {
    assert_compact_rejected_400(
        parse_compact_input(Some(json!([{"type": "function_call", "name": 1, "arguments": "{}"}]))),
        "input[0].name must be a string",
    );
    assert_compact_rejected_400(
        parse_compact_input(Some(json!([{"type": "function_call", "name": "f", "arguments": 2}]))),
        "input[0].arguments must be a string",
    );
}

#[test]
fn parse_compact_input_function_call_null_or_absent_name_and_arguments_are_rejected() {
    // Issue #1403 follow-up: `name` and `arguments` are required strings; a null or
    // missing value would be silently defaulted ("unknown"/""), losing the call.
    assert_compact_rejected_400(
        parse_compact_input(Some(
            json!([{"type": "function_call", "name": null, "arguments": "{}"}]),
        )),
        "input[0].name must be a string",
    );
    assert_compact_rejected_400(
        parse_compact_input(Some(json!([{"type": "function_call", "arguments": "{}"}]))),
        "input[0].name must be a string",
    );
    assert_compact_rejected_400(
        parse_compact_input(Some(json!([{"type": "function_call", "name": "f", "arguments": null}]))),
        "input[0].arguments must be a string",
    );
    assert_compact_rejected_400(
        parse_compact_input(Some(json!([{"type": "function_call", "name": "f"}]))),
        "input[0].arguments must be a string",
    );
}

#[test]
fn parse_compact_input_message_with_null_or_absent_role_is_rejected() {
    // Issue #1403 follow-up: a message `role` is a required string; a null or
    // absent value would be silently defaulted to "unknown", losing the speaker,
    // so it must 400 before any callout or store write.
    assert_compact_rejected_400(
        parse_compact_input(Some(json!([{"role": null, "content": "hello"}]))),
        "input[0].role must be a string",
    );
    assert_compact_rejected_400(
        parse_compact_input(Some(json!([{"type": "message", "content": "hello"}]))),
        "input[0].role must be a string",
    );
}

#[test]
fn parse_compact_input_compaction_null_or_absent_encrypted_content_is_rejected() {
    // Issue #1403 follow-up: a compaction item's `encrypted_content` is a required
    // string; a null or absent value would be summarized as empty text and stored
    // as an empty compaction, so it must 400.
    assert_compact_rejected_400(
        parse_compact_input(Some(json!([{"type": "compaction", "encrypted_content": null}]))),
        "input[0].encrypted_content must be a string",
    );
    assert_compact_rejected_400(
        parse_compact_input(Some(json!([{"type": "compaction"}]))),
        "input[0].encrypted_content must be a string",
    );
}

#[test]
fn parse_compact_input_function_call_output_wrong_typed_output_is_rejected() {
    assert_compact_rejected_400(
        parse_compact_input(Some(json!([{"type": "function_call_output", "output": 7}]))),
        "input[0].output must be a string or an array",
    );
}

#[test]
fn parse_compact_input_function_call_output_content_list_is_accepted() {
    // `output` is `string | content-list`; an array of input_text parts is a
    // valid Responses shape and must not be rejected (it is formatted, not
    // dropped).
    let items = parse_compact_input(Some(json!([{
        "type": "function_call_output",
        "call_id": "c1",
        "output": [{"type": "input_text", "text": "result"}]
    }])))
    .unwrap();
    assert_eq!(items.len(), 1, "a content-list output is well-formed");
}

#[test]
fn parse_compact_input_function_call_output_wrong_typed_part_text_is_rejected() {
    assert_compact_rejected_400(
        parse_compact_input(Some(json!([{
            "type": "function_call_output",
            "output": [{"type": "input_text", "text": 7}]
        }]))),
        "input[0].output[0].text must be a string",
    );
}

#[test]
fn parse_compact_input_unknown_item_kind_without_consumed_fields_is_accepted() {
    // An item compact does not format (no role/content it reads) is not malformed
    // and must pass through untouched rather than be rejected.
    let items = parse_compact_input(Some(json!([{"type": "item_reference", "id": "resp_123"}]))).unwrap();
    assert_eq!(items.len(), 1, "an item_reference carries no field compact consumes");
}

#[test]
fn parse_compact_input_message_with_null_content_is_rejected() {
    // Issue #1403 follow-up: a message `content` is a required `string |
    // content-list`, so an explicit null must 400 before the callout/store
    // rather than be summarized as empty text.
    assert_compact_rejected_400(
        parse_compact_input(Some(json!([{"role": "user", "content": null}]))),
        "input[0].content must be a string or an array",
    );
}

#[test]
fn parse_compact_input_message_without_content_is_rejected() {
    // A message with no `content` field is the same silent-drop as a null one.
    assert_compact_rejected_400(
        parse_compact_input(Some(json!([{"role": "user"}]))),
        "input[0].content must be a string or an array",
    );
    assert_compact_rejected_400(
        parse_compact_input(Some(json!([{"type": "message", "role": "user"}]))),
        "input[0].content must be a string or an array",
    );
}

#[test]
fn parse_compact_input_message_with_empty_content_is_accepted() {
    // Schema-valid emptiness (#1139) must not be rejected: an empty string and
    // an empty content list are both well-formed.
    assert_eq!(
        parse_compact_input(Some(json!([{"role": "user", "content": ""}])))
            .unwrap()
            .len(),
        1,
    );
    assert_eq!(
        parse_compact_input(Some(json!([{"role": "user", "content": []}])))
            .unwrap()
            .len(),
        1,
    );
}

#[test]
fn parse_compact_input_function_call_output_null_or_absent_output_is_rejected() {
    // `output` is a required `string | content-list`; null or absent must 400.
    assert_compact_rejected_400(
        parse_compact_input(Some(
            json!([{"type": "function_call_output", "call_id": "c1", "output": null}]),
        )),
        "input[0].output must be a string or an array",
    );
    assert_compact_rejected_400(
        parse_compact_input(Some(json!([{"type": "function_call_output", "call_id": "c1"}]))),
        "input[0].output must be a string or an array",
    );
}

#[test]
fn parse_compact_input_untyped_item_reference_without_role_is_accepted() {
    // An untyped item with an `id` but no `role` is an item_reference, not a
    // message, so it must not acquire a required-content check.
    let items = parse_compact_input(Some(json!([{"id": "resp_123"}]))).unwrap();
    assert_eq!(items.len(), 1, "an untyped item without a role consumes nothing");
}

#[test]
fn parse_compact_input_untyped_item_with_wrong_typed_content_is_rejected() {
    // `append_item`'s catch-all reads `content` even from an untyped, role-less
    // item, so a wrong-typed `content` would be silently dropped to empty text.
    // It must 400 like any other consumed malformed field (issue #1403).
    assert_compact_rejected_400(
        parse_compact_input(Some(json!([{"content": 123}]))),
        "input[0].content must be a string or an array",
    );
}

#[test]
fn parse_compact_input_unrecognized_type_with_malformed_content_is_rejected() {
    // An unrecognized `type` still reaches the message catch-all in `append_item`,
    // so its `content` is consumed. A null content on a role-bearing item and a
    // wrong-typed content on a role-less item must both 400 (issue #1403).
    assert_compact_rejected_400(
        parse_compact_input(Some(json!([{"type": "reasoning", "role": "user", "content": null}]))),
        "input[0].content must be a string or an array",
    );
    assert_compact_rejected_400(
        parse_compact_input(Some(json!([{"type": "reasoning", "content": 123}]))),
        "input[0].content must be a string or an array",
    );
}

#[test]
fn parse_compact_input_unrecognized_type_without_content_is_accepted() {
    // An unrecognized kind that carries neither a role nor a `content` field
    // consumes nothing and must pass through without a required-content check.
    let items = parse_compact_input(Some(json!([{"type": "reasoning"}]))).unwrap();
    assert_eq!(items.len(), 1, "an item that consumes no content is left alone");
}

// --- previous_response_id ---

#[test]
fn parse_compact_previous_response_id_wrong_type_is_rejected() {
    // Defense-in-depth: compact itself rejects a non-string prior id even when
    // rehydrate is absent from the chain.
    assert_compact_rejected_400(
        parse_compact_request_body(&Some(compact_body(
            &json!({"model": "gpt-4o", "previous_response_id": 123}),
        ))),
        "previous_response_id must be a string",
    );
}

#[test]
fn parse_compact_previous_response_id_empty_and_null_are_none() {
    assert!(parse_compact_previous_response_id(Some(json!(""))).unwrap().is_none());
    assert!(parse_compact_previous_response_id(Some(Value::Null)).unwrap().is_none());
    assert!(parse_compact_previous_response_id(None).unwrap().is_none());
}

#[test]
fn parse_compact_wrong_typed_input_rejected_even_with_previous_response_id() {
    // Issue #1403 case 3: wrong-typed input must 400 even when a valid
    // previous_response_id is present (it must never silently become empty).
    assert_compact_rejected_400(
        parse_compact_request_body(&Some(compact_body(&json!({
            "model": "gpt-4o",
            "input": 123,
            "previous_response_id": "resp_abc"
        })))),
        "input must be a string or an array of items",
    );
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

// =============================================================================
// Test Utilities
// =============================================================================

/// Assert a parse result is an HTTP 400 `invalid_request_error` whose message
/// contains `needle`. Generic over the success type so it covers both whole-body
/// parses and individual field parsers (e.g. `parse_compact_input`).
#[track_caller]
fn assert_compact_rejected_400<T>(result: Result<T, FilterAction>, needle: &str) {
    let Err(FilterAction::Reject(rejection)) = result else {
        panic!("expected a FilterAction::Reject, got a parsed request");
    };
    assert_eq!(rejection.status, 400, "malformed compact field must be HTTP 400");
    let body = rejection.body.as_ref().expect("rejection must carry an error body");
    let err: Value = serde_json::from_slice(body).expect("rejection body should be JSON");
    assert_eq!(
        err["error"]["type"], "invalid_request_error",
        "error type should be invalid_request_error: {err}"
    );
    let message = err["error"]["message"].as_str().unwrap_or_default();
    assert!(
        message.contains(needle),
        "error message {message:?} should contain {needle:?}"
    );
}

/// Serialize a JSON value into request-body bytes.
fn compact_body(value: &Value) -> Bytes {
    Bytes::from(serde_json::to_vec(value).unwrap())
}
