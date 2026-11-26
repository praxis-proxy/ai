// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Ownership and reconciliation regressions for selected upstream input.

use std::borrow::Cow;

use serde_json::{Value, json, value::RawValue};

use super::{ResponsesState, equal, reconcile};
use crate::openai::responses::responses_proxy::{SelectedMessages, scan_top_level_object};

#[test]
fn unchanged_complete_history_borrows_every_payload() {
    let state = replay_state();
    let body = serde_json::to_vec(&json!({"input": state.messages})).unwrap();
    let members = scan_top_level_object(&body).unwrap();
    let selected = reconcile(&body, &members, &state, false).unwrap();
    assert_eq!(selected.len(), state.messages.len());
    for (selected, canonical) in selected.iter().zip(state.messages.iter()) {
        assert!(
            matches!(selected, Cow::Borrowed(_)),
            "unchanged history must not own copied JSON"
        );
        assert!(
            std::ptr::eq(selected.as_ref(), canonical),
            "history must borrow the original payload"
        );
    }
}

#[test]
fn full_history_edit_owns_only_the_changed_item() {
    let state = replay_state();
    let mut input = serde_json::to_value(&state.messages).unwrap();
    input[128]["content"] = json!("edited");
    let body = serde_json::to_vec(&json!({"input": input})).unwrap();
    let members = scan_top_level_object(&body).unwrap();
    let selected = reconcile(&body, &members, &state, false).unwrap();
    assert_eq!(selected.len(), state.messages.len());
    assert_eq!(selected.iter().filter(|item| matches!(item, Cow::Owned(_))).count(), 1);
    for (index, (selected, canonical)) in selected.iter().zip(state.messages.iter()).enumerate() {
        if index == 128 {
            assert_eq!(selected["content"], "edited");
        } else {
            assert!(
                std::ptr::eq(selected.as_ref(), canonical),
                "unedited items must remain borrowed"
            );
        }
    }
}

#[test]
fn current_input_edit_reattaches_borrowed_history_and_results() {
    let state = replay_state();
    let body = br#"{"input":[{"type":"message","role":"user","content":"edited"}]}"#;
    let members = scan_top_level_object(body).unwrap();
    let selected = reconcile(body, &members, &state, false).unwrap();
    assert_eq!(selected.len(), 130);
    assert_eq!(selected[128]["content"], "edited");
    assert_eq!(selected[129]["type"], "function_call_output");
    assert_eq!(selected.iter().filter(|item| matches!(item, Cow::Owned(_))).count(), 1);
    for (selected, canonical) in selected.iter().zip(state.messages.iter()).take(128) {
        assert!(std::ptr::eq(selected.as_ref(), canonical));
    }
}

#[test]
fn shorthand_edit_does_not_keep_an_old_role_or_extra_fields() {
    for input in [
        json!({"type":"message", "role":"developer", "content":"same"}),
        json!({"type":"message", "role":"user", "content":"same", "id":"old"}),
    ] {
        let state = ResponsesState::from_request_body(json!({"input": input}));
        let body = br#"{"input":"same"}"#;
        let members = scan_top_level_object(body).unwrap();
        let selected = reconcile(body, &members, &state, false).unwrap();
        assert_eq!(selected.len(), 1);
        assert_eq!(
            selected[0].as_ref(),
            &json!({"type":"message", "role":"user", "content":"same"})
        );
        assert!(matches!(selected[0], Cow::Owned(_)));
    }
}

#[test]
fn unchanged_shorthand_borrows_the_normalized_message() {
    let state = ResponsesState::from_request_body(json!({"input":"hello"}));
    let body = br#"{"input":"hello"}"#;
    let members = scan_top_level_object(body).unwrap();
    let selected = reconcile(body, &members, &state, false).unwrap();
    assert!(std::ptr::eq(selected[0].as_ref(), state.messages.get(0).unwrap()));
}

#[test]
fn invalid_scalar_number_retains_value_range_error() {
    let state = ResponsesState::from_request_body(json!({"input": null}));
    let body = br#"{"input":1e400}"#;
    let members = scan_top_level_object(body).unwrap();
    assert!(reconcile(body, &members, &state, false).is_err());
}

#[test]
fn semantic_equality_matches_value_deserialization() {
    for text in [
        r#"null"#,
        r#"true"#,
        r#"false"#,
        r#"-1"#,
        r#"18446744073709551615"#,
        r#"1.0"#,
        r#"-0.0"#,
        r#"1e100"#,
        r#""escaped\n\u0061""#,
        r#"[null, true, {"nested": [1, 2]}]"#,
        r#"{"b":2,"a":1}"#,
        r#"{"a":0,"a":1}"#,
        r#"{"nested":{"x":0},"nested":{"x":1}}"#,
        r#"{}"#,
        r#"[]"#,
    ] {
        let raw: &RawValue = serde_json::from_str(text).unwrap();
        let decoded: Value = serde_json::from_str(text).unwrap();
        for expected in [
            &decoded,
            &json!(null),
            &json!({}),
            &json!([]),
            &json!("different"),
            &json!(1),
        ] {
            assert_eq!(
                equal(raw, expected).unwrap(),
                decoded == *expected,
                "semantic equality for {text}"
            );
        }
    }
}

#[test]
fn reordered_and_duplicate_members_still_borrow_the_last_value() {
    let state = ResponsesState::from_request_body(json!({"input":[{"type":"message","role":"user","content":"a"}]}));
    let body = br#"{"input":[],"input":[{"content":"old","role":"user","content":"\u0061","type":"message"}]}"#;
    let members = scan_top_level_object(body).unwrap();
    let selected = reconcile(body, &members, &state, false).unwrap();
    assert_eq!(selected.len(), 1);
    assert!(std::ptr::eq(selected[0].as_ref(), state.messages.get(0).unwrap()));
}

#[test]
fn provider_owned_round_sends_only_the_borrowed_delta() {
    let mut state = ResponsesState::from_request_body(json!({"input":"hello", "conversation":"provider-conv"}));
    state.provider_history_len = 1;
    state.iteration = 1;
    state
        .messages
        .push(json!({"type":"function_call_output", "call_id":"call_1", "output":"result"}));
    let body = br#"{"input":"hello"}"#;
    let members = scan_top_level_object(body).unwrap();
    let selected = reconcile(body, &members, &state, true).unwrap();
    assert_eq!(selected.len(), 1);
    assert!(std::ptr::eq(selected[0].as_ref(), state.messages.get(1).unwrap()));
}

#[test]
fn preprocessed_current_input_still_appends_tool_results() {
    // Document extraction rewrites the working history in place while the
    // immutable client snapshot stays as the original input_file item.
    let original = json!({
        "type": "message",
        "role": "user",
        "content": [{
            "type": "input_file",
            "filename": "source.txt",
            "file_data": "data:text/plain;base64,aGVsbG8="
        }]
    });
    let rewritten = json!({
        "type": "message",
        "role": "user",
        "content": [{
            "type": "input_text",
            "text": "[Source: source.txt]\nhello"
        }]
    });
    let mut state = ResponsesState::from_request_body(json!({
        "model": "gpt-4.1",
        "input": [original]
    }));
    *state.messages.get_mut(0).unwrap() = rewritten.clone();
    state.messages.push(json!({
        "type": "function_call",
        "id": "fc_preprocessed",
        "call_id": "call_preprocessed",
        "name": "file_search",
        "arguments": "{\"query\":\"q\"}",
        "status": "completed"
    }));
    state.messages.push(json!({
        "type": "function_call_output",
        "call_id": "call_preprocessed",
        "output": "PREPROCESSED_TOOL_RESULT"
    }));

    let body = serde_json::to_vec(&json!({"input": [rewritten]})).unwrap();
    let members = scan_top_level_object(&body).unwrap();
    let selected = reconcile(&body, &members, &state, false).unwrap();
    assert_eq!(selected.len(), 3);
    assert_eq!(selected[0]["content"][0]["type"], "input_text");
    assert_eq!(selected[1]["type"], "function_call");
    assert_eq!(selected[2]["output"], "PREPROCESSED_TOOL_RESULT");
    assert!(
        std::ptr::eq(selected[1].as_ref(), state.messages.get(1).unwrap()),
        "unchanged tool call must remain borrowed from messages"
    );
    assert!(
        std::ptr::eq(selected[2].as_ref(), state.messages.get(2).unwrap()),
        "unchanged tool result must remain borrowed from messages"
    );
}

#[test]
fn preprocessed_then_later_edit_keeps_tool_results() {
    // Preprocessing rewrote messages; a later live-body edit diverges further.
    // Content search cannot find either snapshot — the durable start index must.
    let original = json!({
        "type": "message",
        "role": "user",
        "content": [{
            "type": "input_file",
            "filename": "s.txt",
            "file_data": "data:text/plain;base64,aGVsbG8="
        }]
    });
    let rewritten = json!({
        "type": "message",
        "role": "user",
        "content": [{
            "type": "input_text",
            "text": "[Source: s.txt]\nhello"
        }]
    });
    let edited = json!({
        "type": "message",
        "role": "user",
        "content": [{
            "type": "input_text",
            "text": "[Source: s.txt]\nREDACTED"
        }]
    });
    let mut state = ResponsesState::from_request_body(json!({
        "model": "gpt-4.1",
        "input": [original]
    }));
    *state.messages.get_mut(0).unwrap() = rewritten;
    state.messages.push(json!({
        "type": "function_call_output",
        "call_id": "c",
        "output": "R"
    }));

    let body = serde_json::to_vec(&json!({"input": [edited]})).unwrap();
    let members = scan_top_level_object(&body).unwrap();
    let selected = reconcile(&body, &members, &state, false).unwrap();
    assert_eq!(selected.len(), 2);
    assert_eq!(selected[0]["content"][0]["text"], "[Source: s.txt]\nREDACTED");
    assert_eq!(selected[1]["output"], "R");
}

#[test]
fn file_resolve_rewritten_current_input_still_appends_tool_results() {
    // openai_file_resolve rewrites file_id → file_data in place while state.input
    // keeps the client file_id snapshot — same boundary as doc_extract (#1907).
    let original = json!({
        "type": "message",
        "role": "user",
        "content": [{
            "type": "input_file",
            "file_id": "file-abc",
            "filename": "notes.pdf"
        }]
    });
    let resolved = json!({
        "type": "message",
        "role": "user",
        "content": [{
            "type": "input_file",
            "filename": "notes.pdf",
            "file_data": "data:application/pdf;base64,JVBERi0="
        }]
    });
    let mut state = ResponsesState::from_request_body(json!({
        "model": "gpt-4.1",
        "input": [original]
    }));
    *state.messages.get_mut(0).unwrap() = resolved.clone();
    state.messages.push(json!({
        "type": "function_call_output",
        "call_id": "call_file_resolve",
        "output": "FILE_RESOLVE_TOOL_RESULT"
    }));

    let body = serde_json::to_vec(&json!({"input": [resolved]})).unwrap();
    let members = scan_top_level_object(&body).unwrap();
    let selected = reconcile(&body, &members, &state, false).unwrap();
    assert_eq!(selected.len(), 2);
    assert_eq!(
        selected[0]["content"][0]["file_data"],
        "data:application/pdf;base64,JVBERi0="
    );
    assert_eq!(selected[1]["output"], "FILE_RESOLVE_TOOL_RESULT");
}

#[test]
fn preprocessed_rehydrated_history_still_appends_tool_results() {
    let original = json!({
        "type": "message",
        "role": "user",
        "content": [{
            "type": "input_file",
            "filename": "source.txt",
            "file_data": "data:text/plain;base64,aGVsbG8="
        }]
    });
    let rewritten = json!({
        "type": "message",
        "role": "user",
        "content": [{
            "type": "input_text",
            "text": "[Source: source.txt]\nhello"
        }]
    });
    let earlier = json!({"type": "message", "role": "assistant", "content": "earlier"});
    let mut state = ResponsesState::from_request_body(json!({
        "model": "gpt-4.1",
        "input": [original]
    }));
    state.messages = vec![
        earlier.clone(),
        rewritten.clone(),
        json!({
            "type": "function_call_output",
            "call_id": "call_hist",
            "output": "HISTORY_TOOL_RESULT"
        }),
    ]
    .into();
    state.history_rehydrated = true;
    state.current_input_start = 1;

    // Live body includes the rehydrated history prefix plus the rewritten
    // current input, matching the production rebuild shape.
    let body = serde_json::to_vec(&json!({"input": [earlier, rewritten]})).unwrap();
    let members = scan_top_level_object(&body).unwrap();
    let selected = reconcile(&body, &members, &state, false).unwrap();
    assert_eq!(selected.len(), 3);
    assert_eq!(selected[0]["content"], "earlier");
    assert_eq!(selected[1]["content"][0]["type"], "input_text");
    assert_eq!(selected[2]["output"], "HISTORY_TOOL_RESULT");
}

#[test]
fn compaction_projection_keeps_unedited_neighbors_borrowed() {
    let mut state = ResponsesState::from_request_body(json!({"input":"latest"}));
    state.messages = vec![
        json!({"type":"compaction", "id":"local", "encrypted_content":"c3VtbWFyeQ==", "_praxis_local_compaction":true}),
        json!({"type":"compaction", "id":"provider", "encrypted_content":"opaque"}),
        json!({"type":"message", "role":"assistant", "content":"retained"}),
        state.input[0].clone(),
    ]
    .into();
    state.provider_compaction_ids.insert("provider".into());
    state.history_rehydrated = true;
    state.current_input_start = 3;
    let projected = super::super::serialize_outbound_body(&state, true).unwrap();
    let mut input: Value = serde_json::from_slice(&projected).unwrap();
    input["input"][3]["content"] = json!("edited");
    let body = serde_json::to_vec(&input).unwrap();
    let members = scan_top_level_object(&body).unwrap();
    let selected = reconcile(&body, &members, &state, true).unwrap();
    assert_eq!(selected.len(), 4);
    assert!(
        matches!(selected[0], Cow::Owned(_)),
        "only the local summary needs translation"
    );
    assert!(std::ptr::eq(selected[1].as_ref(), state.messages.get(1).unwrap()));
    assert!(std::ptr::eq(selected[2].as_ref(), state.messages.get(2).unwrap()));
    let serialized = serde_json::to_value(SelectedMessages {
        messages: &selected,
        preserve_native_compaction: true,
        provider_compaction_ids: &state.provider_compaction_ids,
    })
    .unwrap();
    assert_eq!(serialized[0]["role"], "assistant");
    assert!(serialized[0]["content"].as_str().unwrap().ends_with("summary"));
    assert_eq!(serialized[1]["encrypted_content"], "opaque");
    assert_eq!(serialized[3]["content"], "edited");
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

fn replay_state() -> ResponsesState {
    let mut state = ResponsesState::from_request_body(json!({"input":"latest"}));
    state.messages = (0..128)
        .map(|index| {
            json!({
                "type":"message", "role":"user", "content":"x".repeat(4096), "id":format!("msg_{index}")
            })
        })
        .collect();
    state.messages.extend(state.input.iter().cloned());
    state
        .messages
        .push(json!({"type":"function_call_output", "call_id":"call_1", "output":"result"}));
    state.history_rehydrated = true;
    state.current_input_start = 128;
    state
}
