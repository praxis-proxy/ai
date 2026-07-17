// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

// Uncomment as you implement and enable tests:
// use serde_json::json;
// use super::*;

// =============================================================================
// Config tests
// =============================================================================

// TODO: implement these once build_config() is done

// #[test]
// fn build_config_applies_defaults() { ... }

// #[test]
// fn build_config_rejects_empty_inference_url() { ... }

// #[test]
// fn build_config_rejects_zero_timeout() { ... }

// #[test]
// fn build_config_rejects_invalid_status() { ... }

// #[test]
// fn build_config_custom_values() { ... }

// =============================================================================
// extract_compaction_config tests
// =============================================================================

// TODO: uncomment once extract_compaction_config() is implemented

// #[test]
// fn extract_compaction_config_with_compaction_entry() {
//     let cm = Some(json!([{"type": "compaction", "compact_threshold": 50000}]));
//     let params = extract_compaction_config(&cm);
//     assert!(params.is_some());
//     let params = params.unwrap();
//     assert_eq!(params.compact_threshold, 50000);
//     assert!(params.compaction_model.is_none());
// }
//
// #[test]
// fn extract_compaction_config_with_model_override() {
//     let cm = Some(json!([{
//         "type": "compaction",
//         "compact_threshold": 100000,
//         "compaction_model": "gpt-4o"
//     }]));
//     let params = extract_compaction_config(&cm).unwrap();
//     assert_eq!(params.compact_threshold, 100000);
//     assert_eq!(params.compaction_model.as_deref(), Some("gpt-4o"));
// }
//
// #[test]
// fn extract_compaction_config_no_compaction_entry() {
//     let cm = Some(json!([{"type": "truncation", "max_tokens": 4096}]));
//     assert!(extract_compaction_config(&cm).is_none());
// }
//
// #[test]
// fn extract_compaction_config_none() {
//     assert!(extract_compaction_config(&None).is_none());
// }
//
// #[test]
// fn extract_compaction_config_empty_array() {
//     let cm = Some(json!([]));
//     assert!(extract_compaction_config(&cm).is_none());
// }

// =============================================================================
// build_compaction_item tests
// =============================================================================

// TODO: uncomment once build_compaction_item() is implemented

// #[test]
// fn compaction_item_has_correct_shape() {
//     let item = build_compaction_item("This is a summary.");
//     assert_eq!(item["type"], "compaction");
//     assert_eq!(item["encrypted_content"], "This is a summary.");
// }

// =============================================================================
// parse_summarization_response tests
// =============================================================================

// TODO: uncomment once parse_summarization_response() is implemented

// #[test]
// fn parse_valid_chat_completion_response() {
//     let response = json!({
//         "choices": [{
//             "message": {
//                 "role": "assistant",
//                 "content": "Here is the summary."
//             }
//         }]
//     });
//     let body = serde_json::to_vec(&response).unwrap();
//     let result = parse_summarization_response(&body);
//     assert_eq!(result.unwrap(), "Here is the summary.");
// }
//
// #[test]
// fn parse_malformed_response_returns_error() {
//     let result = parse_summarization_response(b"not json");
//     assert!(result.is_err());
// }
//
// #[test]
// fn parse_response_missing_choices_returns_error() {
//     let response = json!({"id": "chatcmpl-123"});
//     let body = serde_json::to_vec(&response).unwrap();
//     assert!(parse_summarization_response(&body).is_err());
// }

// =============================================================================
// build_conversation_text tests
// =============================================================================

// TODO: uncomment once build_conversation_text() is implemented

// #[test]
// fn conversation_text_simple_messages() {
//     let messages = vec![
//         json!({"role": "user", "content": "Hello"}),
//         json!({"role": "assistant", "content": "Hi there!"}),
//     ];
//     let text = build_conversation_text(&messages);
//     assert!(text.contains("user: Hello"));
//     assert!(text.contains("assistant: Hi there!"));
// }
//
// #[test]
// fn conversation_text_empty_messages() {
//     let text = build_conversation_text(&[]);
//     assert!(text.is_empty());
// }

// =============================================================================
// replace_messages tests
// =============================================================================

// TODO: uncomment once replace_messages() and build_compaction_item()
//       are implemented

// #[test]
// fn replace_messages_preserves_current_input() {
//     // Simulate state after rehydrate:
//     //   input = [current_user_msg]
//     //   messages = [old_msg1, old_msg2, current_user_msg]
//     let mut state = ResponsesState::from_request_body(json!({
//         "model": "gpt-4o",
//         "input": "What's next?"
//     }));
//     // Prepend some "rehydrated" history
//     state.messages.insert(
//         0,
//         json!({"role": "user", "content": "old question"}),
//     );
//     state.messages.insert(
//         1,
//         json!({"role": "assistant", "content": "old answer"}),
//     );
//     state.persisted_messages.insert(
//         0,
//         json!({"role": "user", "content": "old question"}),
//     );
//     state.persisted_messages.insert(
//         1,
//         json!({"role": "assistant", "content": "old answer"}),
//     );
//
//     let compaction_item = build_compaction_item("Summary of old conversation.");
//     replace_messages(&mut state, compaction_item);
//
//     // After replacement: [compaction_item, current_user_msg]
//     assert_eq!(
//         state.messages.len(),
//         2,
//         "should have compaction + current input"
//     );
//     assert_eq!(state.messages[0]["type"], "compaction");
//     assert_eq!(state.messages[1]["role"], "user");
//     assert_eq!(state.messages[1]["content"], "What's next?");
//
//     // persisted_messages should match
//     assert_eq!(state.persisted_messages.len(), 2);
//     assert_eq!(state.persisted_messages[0]["type"], "compaction");
// }

// =============================================================================
// Filter-level tests
// =============================================================================

// TODO: implement once from_config() works

// #[test]
// fn from_config_with_valid_yaml() { ... }

// #[test]
// fn from_config_rejects_missing_inference_url() { ... }
