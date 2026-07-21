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
        timeout_ms: None,
        failure_mode: None,
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
fn build_config_custom_values() {
    let mut cfg = base_config();
    cfg.timeout_ms = Some(60_000);
    cfg.failure_mode = Some(FailureMode::Open);
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
    let cm = Some(json!([{"type": "compaction", "compact_threshold": 50000}]));
    let params = extract_compaction_config(&cm);
    assert!(params.is_some());
    let params = params.unwrap();
    assert_eq!(params.compact_threshold, 50000);
    assert!(params.compaction_model.is_none());
}

#[test]
fn extract_compaction_config_with_model_override() {
    let cm = Some(json!([{
        "type": "compaction",
        "compact_threshold": 100000,
        "compaction_model": "gpt-4o"
    }]));
    let params = extract_compaction_config(&cm).unwrap();
    assert_eq!(params.compact_threshold, 100000);
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
fn extract_compaction_config_defaults_threshold_to_zero() {
    let cm = Some(json!([{"type": "compaction"}]));
    let params = extract_compaction_config(&cm).unwrap();
    assert_eq!(params.compact_threshold, 0);
}

// =============================================================================
// build_compaction_item tests
// =============================================================================

#[test]
fn compaction_item_has_correct_shape() {
    let item = build_compaction_item("This is a summary.");
    assert_eq!(item["role"], "system");
    assert_eq!(item["content"], "This is a summary.");
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
    let req = build_summarization_request(&conversation_text, None, "gpt-4o-mini", "http://localhost/v1/chat/completions");
    assert_eq!(req.method, http::Method::POST);
    assert_eq!(req.url, "http://localhost/v1/chat/completions");
    assert!(req.body.is_some());
    let body: Value = serde_json::from_slice(&req.body.unwrap()).unwrap();
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
    let req = build_summarization_request(&conversation_text, Some("Be concise"), "gpt-4o-mini", "http://localhost/v1/chat/completions");
    let body: Value = serde_json::from_slice(&req.body.unwrap()).unwrap();
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
    state.messages.insert(0, json!({"role": "user", "content": "old question"}));
    state.messages.insert(1, json!({"role": "assistant", "content": "old answer"}));
    state.persisted_messages.insert(0, json!({"role": "user", "content": "old question"}));
    state.persisted_messages.insert(1, json!({"role": "assistant", "content": "old answer"}));

    let compaction_item = build_compaction_item("Summary of old conversation.");
    replace_messages(&mut state, compaction_item);

    assert_eq!(state.messages.len(), 2, "should have compaction + current input");
    assert_eq!(state.messages[0]["role"], "system");
    assert_eq!(state.messages[0]["content"], "Summary of old conversation.");
    assert_eq!(state.persisted_messages.len(), 2);
    assert_eq!(state.persisted_messages[0]["role"], "system");
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