// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Service-layer unit tests: record assembly, listing, CRUD, and pending-approval
//! coordination exercised against the in-memory backend, with no request pipeline
//! and no SQL database.

use std::sync::Arc;

use praxis_ai_store::{
    PendingApprovalRecord, PersistedStateBackend, ResponseRecord, StoreError, StoreRegistry, memory::InMemoryStore,
};
use serde_json::{Value, json};

use super::{
    ListParams, Order, ResponsesService, assemble_stored_messages, list_input_items, normalize_translated_reasoning,
};
use crate::{StateOwner, openai::include::IncludeFields};

/// Build a validated owner for a fixed tenant and issuer.
fn owner(subject: &str) -> StateOwner {
    StateOwner::from_trusted_parts("tenant-a", "issuer-a", subject).unwrap()
}

/// A registry holding one in-memory backend under the default name.
fn registry() -> StoreRegistry {
    let reg = StoreRegistry::new();
    let backend: Arc<dyn PersistedStateBackend> = Arc::new(InMemoryStore::new());
    let name: Arc<str> = Arc::from("default");
    reg.register(&name, backend).unwrap();
    reg
}

/// Bind a service to the default backend for `owner`.
fn service(reg: &StoreRegistry, owner: &StateOwner) -> ResponsesService {
    ResponsesService::new(reg.get_scoped("default", owner).unwrap())
}

/// A minimal persistable record owned by `owner`.
fn sample_record(owner: &StateOwner, id: &str) -> ResponseRecord {
    ResponseRecord {
        id: id.to_owned(),
        owner: owner.clone(),
        created_at: 1_719_900_000,
        model: "gpt-4.1".to_owned(),
        response_object: json!({"id": id, "created_at": 1_719_900_000, "model": "gpt-4.1", "output": []}),
        input: json!([{"role": "user", "content": "Hello"}]),
        messages: json!([{"role": "user", "content": "Hello"}]),
    }
}

#[test]
fn build_record_returns_none_for_null_response_object() {
    let record = ResponsesService::build_record(Value::Null, owner("alice"), None, None, &[]);
    assert!(record.is_none(), "a null response object is not persistable");
}

#[test]
fn build_record_returns_none_for_missing_required_fields() {
    let record = ResponsesService::build_record(json!({"id": "resp_1"}), owner("alice"), None, None, &[]);
    assert!(record.is_none(), "missing created_at/model is not persistable");
}

#[test]
fn build_record_uses_request_input_when_state_messages_absent() {
    let request_input = json!([{"role": "user", "content": "Captured streaming input"}]);
    let response_object = json!({
        "id": "resp_stream_no_state_messages",
        "created_at": 1_719_900_000,
        "model": "gpt-4.1",
        "status": "completed",
        "output": [{"type": "message", "content": "Stored streaming output"}]
    });

    let record =
        ResponsesService::build_record(response_object, owner("alice"), Some(request_input.clone()), None, &[])
            .expect("streaming state should build a record");

    assert_eq!(
        record.input, request_input,
        "stored input comes from the original request"
    );
    assert_eq!(
        record.messages,
        json!([
            {"role": "user", "content": "Captured streaming input"},
            {"type": "message", "content": "Stored streaming output"}
        ]),
        "absent state messages must not hide request input"
    );
}

#[test]
fn build_record_preserves_mcp_metadata_from_state_messages() {
    let mcp_item = json!({
        "id": "mcpl_1",
        "type": "mcp_list_tools",
        "server_label": "weather",
        "tools": [{"name": "get_weather", "description": "d", "input_schema": {}}]
    });
    let response_object = json!({
        "id": "resp_stream_mcp",
        "created_at": 1_719_900_000,
        "model": "gpt-4.1",
        "status": "completed",
        "output": [{"type": "message", "role": "assistant", "content": "Next answer"}]
    });
    let state_messages = vec![
        json!({"role": "user", "content": "Hello"}),
        mcp_item,
        json!({"type": "message", "role": "assistant", "content": "Tools loaded"}),
        json!({"role": "user", "content": "What next?"}),
    ];

    let record = ResponsesService::build_record(
        response_object,
        owner("alice"),
        Some(json!([{"role": "user", "content": "What next?"}])),
        Some(state_messages),
        &[],
    )
    .expect("streaming state should build a record");

    assert_eq!(
        record.messages,
        json!([
            {"role": "user", "content": "Hello"},
            {"id": "mcpl_1", "type": "mcp_list_tools", "server_label": "weather",
             "tools": [{"name": "get_weather", "description": "d", "input_schema": {}}]},
            {"type": "message", "role": "assistant", "content": "Tools loaded"},
            {"role": "user", "content": "What next?"},
            {"type": "message", "role": "assistant", "content": "Next answer"}
        ]),
        "MCP metadata from state messages must be preserved, not dropped"
    );
}

#[test]
fn build_record_falls_back_to_response_object_input() {
    let response_object = json!({
        "id": "resp_buffered",
        "created_at": 1_719_900_000,
        "model": "gpt-4.1",
        "input": "Hello",
        "output": [{"type": "message", "role": "assistant", "content": "Hi"}]
    });

    let record = ResponsesService::build_record(response_object, owner("alice"), None, None, &[])
        .expect("buffered response should build a record");

    assert_eq!(
        record.input,
        json!("Hello"),
        "input falls back to the response object when no request input is captured"
    );
}

#[tokio::test]
async fn persist_and_get_round_trip() {
    let reg = registry();
    let alice = owner("alice");
    let svc = service(&reg, &alice);

    svc.persist(&sample_record(&alice, "resp_rt"), &[])
        .await
        .expect("persist should succeed");

    let fetched = svc
        .get("resp_rt")
        .await
        .expect("get should succeed")
        .expect("record should exist");
    assert_eq!(fetched.id, "resp_rt");
    assert_eq!(fetched.owner, alice);
}

#[tokio::test]
async fn delete_removes_record_and_reports_missing() {
    let reg = registry();
    let alice = owner("alice");
    let svc = service(&reg, &alice);
    svc.persist(&sample_record(&alice, "resp_del"), &[]).await.unwrap();

    assert!(
        svc.delete("resp_del").await.unwrap(),
        "deleting an existing record returns true"
    );
    assert!(
        svc.get("resp_del").await.unwrap().is_none(),
        "the record is gone after delete"
    );
    assert!(
        !svc.delete("resp_del").await.unwrap(),
        "deleting a missing record returns false"
    );
}

#[tokio::test]
async fn get_is_owner_scoped() {
    let reg = registry();
    let alice = owner("alice");
    let bob = owner("bob");
    service(&reg, &alice)
        .persist(&sample_record(&alice, "resp_iso"), &[])
        .await
        .unwrap();

    let bob_view = service(&reg, &bob).get("resp_iso").await.expect("get should succeed");
    assert!(
        bob_view.is_none(),
        "a record persisted by one owner is invisible to another"
    );
}

#[tokio::test]
async fn persist_rejects_record_owned_by_another_owner() {
    let reg = registry();
    let alice = owner("alice");
    let bob = owner("bob");

    // Service bound to alice, record built for bob: the owner-scoped facade must reject it.
    let err = service(&reg, &alice)
        .persist(&sample_record(&bob, "resp_forge"), &[])
        .await
        .expect_err("an owner mismatch must fail closed");
    assert!(
        matches!(err, StoreError::InvalidInput(_)),
        "owner mismatch is an invalid-input error"
    );

    assert!(
        service(&reg, &bob).get("resp_forge").await.unwrap().is_none(),
        "the rejected record must not be written under any owner"
    );
}

#[tokio::test]
async fn pending_approvals_persist_then_consume_and_reject_replay() {
    let reg = registry();
    let alice = owner("alice");
    let svc = service(&reg, &alice);
    let approval = PendingApprovalRecord {
        approval_id: "call_1".to_owned(),
        server_label: "weather".to_owned(),
        tool_name: "get_weather".to_owned(),
        arguments: "{}".to_owned(),
        target_fingerprint: "fp".to_owned(),
    };

    svc.persist(&sample_record(&alice, "resp_appr"), std::slice::from_ref(&approval))
        .await
        .expect("persist with pending approvals should succeed");

    let pending = svc.get_pending_approvals("resp_appr", &["call_1"]).await.unwrap();
    assert_eq!(pending.len(), 1, "the pending approval should be readable");

    let consumed = svc
        .consume_approvals("resp_appr", &["call_1"], 1_719_900_100)
        .await
        .unwrap();
    assert!(consumed.is_none(), "the first consumption claims the whole batch");

    let replay = svc
        .consume_approvals("resp_appr", &["call_1"], 1_719_900_200)
        .await
        .unwrap();
    assert_eq!(
        replay,
        Some(0),
        "a replayed approval reports its batch index and consumes nothing"
    );
}

#[test]
fn list_input_items_paginates_stored_input() {
    let alice = owner("alice");
    let record = ResponseRecord {
        input: json!([
            {"id": "a", "type": "message", "role": "user", "content": "1"},
            {"id": "b", "type": "message", "role": "user", "content": "2"},
            {"id": "c", "type": "message", "role": "user", "content": "3"}
        ]),
        ..sample_record(&alice, "resp_list")
    };
    let params = ListParams {
        cursor: None,
        limit: 2,
        order: Order::Ascending,
    };

    let page = list_input_items(&record, &params, IncludeFields::default()).expect("listing should succeed");
    assert_eq!(page.data.len(), 2, "the limit caps the page size");
    assert!(page.has_more, "more items remain beyond the first page");
}

fn types(items: &[Value]) -> Vec<&str> {
    items
        .iter()
        .map(|item| item.get("type").and_then(Value::as_str).unwrap_or("message"))
        .collect()
}

#[test]
fn trailing_reasoning_after_a_message_moves_ahead_of_its_turn() {
    // The streaming translator emits `[message, reasoning]` when the answer
    // streams before reasoning. The stored history must lead with reasoning so
    // continuation replay attaches it to the message that follows.
    let mut items = vec![
        json!({"type": "message", "role": "assistant", "id": "msg_1"}),
        json!({"type": "reasoning", "id": "rs_1", "content": [{"type": "reasoning_text", "text": "late"}]}),
    ];
    normalize_translated_reasoning(&mut items, &[0..=1]);
    assert_eq!(types(&items), ["reasoning", "message"]);
}

#[test]
fn trailing_reasoning_moves_ahead_of_a_message_and_tool_call_turn() {
    let mut items = vec![
        json!({"type": "message", "role": "assistant", "id": "msg_1"}),
        json!({"type": "function_call", "id": "fc_1"}),
        json!({"type": "reasoning", "id": "rs_1", "content": [{"type": "reasoning_text", "text": "late"}]}),
    ];
    normalize_translated_reasoning(&mut items, &[0..=2]);
    assert_eq!(types(&items), ["reasoning", "message", "function_call"]);
}

#[test]
fn reasoning_first_turns_are_left_untouched() {
    // The finite builder and native passthrough already lead with reasoning.
    let mut items = vec![
        json!({"type": "reasoning", "id": "rs_1"}),
        json!({"type": "message", "role": "assistant", "id": "msg_1"}),
    ];
    let before = items.clone();
    normalize_translated_reasoning(&mut items, &[]);
    assert_eq!(items, before);
}

#[test]
fn multi_round_agentic_reasoning_is_not_collapsed_onto_an_earlier_turn() {
    // Each round already leads with its own reasoning, and the trailing
    // reasoning here (round 2) is followed by round 2's message, so it is not
    // a trailing item: nothing moves.
    let mut items = vec![
        json!({"type": "reasoning", "id": "rs_1"}),
        json!({"type": "function_call", "id": "fc_1"}),
        json!({"type": "reasoning", "id": "rs_2"}),
        json!({"type": "message", "role": "assistant", "id": "msg_2"}),
    ];
    let before = items.clone();
    normalize_translated_reasoning(&mut items, &[]);
    assert_eq!(items, before);
}

#[test]
fn a_standalone_trailing_reasoning_turn_is_preserved() {
    // The preceding turn already leads with its own reasoning, so a trailing
    // reasoning item is a separate, standalone turn and must not be merged.
    let mut items = vec![
        json!({"type": "reasoning", "id": "rs_1"}),
        json!({"type": "message", "role": "assistant", "id": "msg_1"}),
        json!({"type": "reasoning", "id": "rs_2"}),
    ];
    let before = items.clone();
    normalize_translated_reasoning(&mut items, &[]);
    assert_eq!(items, before);
}

#[test]
fn build_record_normalizes_replay_without_changing_the_response_object() {
    let input = json!([{"role": "user", "content": "hi"}]);
    let output = json!([
        {"type": "message", "role": "assistant", "id": "msg_1"},
        {"type": "reasoning", "id": "rs_1", "content": [{"type": "reasoning_text", "text": "late"}]}
    ]);
    let response = json!({"id": "resp_late", "created_at": 1, "model": "m", "output": output});
    let record = ResponsesService::build_record(response, owner("alice"), Some(input), None, &[0..=1])
        .expect("translated response should build a record");
    assert_eq!(
        types(record.response_object["output"].as_array().unwrap()),
        ["message", "reasoning"]
    );
    let messages = record.messages.as_array().expect("assembled messages are an array");
    // The user input stays first; only the stored output turn is normalized.
    let first_role = messages
        .first()
        .and_then(|item| item.get("role"))
        .and_then(Value::as_str);
    assert_eq!(first_role, Some("user"));
    let output_turn = messages.get(1..).expect("input item precedes the stored output");
    assert_eq!(types(output_turn), ["reasoning", "message"]);
}

#[test]
fn translated_reasoning_stays_with_its_round_even_when_message_ids_repeat() {
    let output = json!([
        {"type": "message", "role": "assistant", "id": "msg_shared", "content": "first"},
        {"type": "message", "role": "assistant", "id": "msg_shared", "content": "second"},
        {"type": "reasoning", "id": "rs_second", "content": [{"type": "reasoning_text", "text": "second thought"}]},
        {"type": "message", "role": "assistant", "id": "msg_shared", "content": "third"},
        {"type": "reasoning", "id": "rs_third", "content": [{"type": "reasoning_text", "text": "third thought"}]}
    ]);
    let input = assemble_stored_messages(json!([]), Some(&output), &[1..=2, 3..=4]);
    let chat = crate::openai::translation::chat_completions::responses_request_to_chat_request(
        &json!({"model": "m", "input": input}),
        &crate::openai::translation::reasoning::ReasoningOptions {
            dialect: crate::openai::translation::reasoning::ReasoningDialect::Vllm,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        chat["messages"],
        json!([
            {"role": "assistant", "content": "first"},
            {"role": "assistant", "content": "second", "reasoning": "second thought"},
            {"role": "assistant", "content": "third", "reasoning": "third thought"}
        ])
    );
    assert_eq!(output[1]["type"], "message", "wire order must be unchanged");
}

#[test]
fn late_reasoning_before_a_tool_call_replays_with_the_answer() {
    let output = json!([
        {"type": "message", "role": "assistant", "content": "answer"},
        {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "thought"}]},
        {"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{}"}
    ]);
    let input = assemble_stored_messages(json!([]), Some(&output), &[0..=1]);
    let chat = crate::openai::translation::chat_completions::responses_request_to_chat_request(
        &json!({"model": "m", "input": input}),
        &crate::openai::translation::reasoning::ReasoningOptions {
            dialect: crate::openai::translation::reasoning::ReasoningDialect::Vllm,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        chat["messages"][0],
        json!({"role": "assistant", "content": "answer", "reasoning": "thought"})
    );
    assert_eq!(chat["messages"][1]["tool_calls"][0]["id"], "call_1");
    assert!(chat["messages"][1].get("reasoning").is_none());
}

#[test]
fn native_output_without_translator_provenance_keeps_its_order() {
    let output = json!([
        {"type": "message", "role": "assistant", "id": "msg_1"},
        {"type": "message", "role": "assistant", "id": "msg_2"},
        {"type": "reasoning", "id": "rs_native"}
    ]);
    assert_eq!(assemble_stored_messages(json!([]), Some(&output), &[]), output);
}
