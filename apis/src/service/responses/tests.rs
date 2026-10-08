// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Service-layer unit tests: record assembly, listing, CRUD, and pending-approval
//! coordination exercised against the in-memory backend, with no request pipeline
//! and no SQL database.

use std::{ops::RangeInclusive, sync::Arc};

use praxis_ai_store::{
    PendingApprovalRecord, PersistedStateBackend, ResponseRecord, StoreError, StoreRegistry, memory::InMemoryStore,
};
use serde_json::{Value, json};

use super::{
    ListParams, Order, StoredOutputPlan, assemble_stored_messages, build_record, list_input_items,
    rotate_trailing_reasoning,
};
use crate::{
    StateOwner,
    openai::{include::IncludeFields, responses::state::CollectedRound},
    store::OwnerScopedResponseStore,
};

/// Build a non-agentic plan carrying only translator replay ranges.
fn replay_plan(reasoning_replay: &[RangeInclusive<usize>]) -> StoredOutputPlan<'_> {
    StoredOutputPlan {
        reasoning_replay,
        collected_rounds: &[],
        collected_provenance: &[],
    }
}

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

/// Bind a store handle to the default backend for `owner`.
fn scoped_store(reg: &StoreRegistry, owner: &StateOwner) -> OwnerScopedResponseStore {
    reg.get_scoped("default", owner).unwrap()
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
    let record = build_record(Value::Null, owner("alice"), None, None, StoredOutputPlan::EMPTY);
    assert!(record.is_none(), "a null response object is not persistable");
}

#[test]
fn build_record_returns_none_for_missing_required_fields() {
    let record = build_record(
        json!({"id": "resp_1"}),
        owner("alice"),
        None,
        None,
        StoredOutputPlan::EMPTY,
    );
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

    let record = build_record(
        response_object,
        owner("alice"),
        Some(request_input.clone()),
        None,
        StoredOutputPlan::EMPTY,
    )
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

    let record = build_record(
        response_object,
        owner("alice"),
        Some(json!([{"role": "user", "content": "What next?"}])),
        Some(state_messages),
        StoredOutputPlan::EMPTY,
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
fn build_record_does_not_duplicate_output_already_in_state_messages() {
    let compaction = json!({
        "type": "compaction",
        "id": "cmp_provider",
        "encrypted_content": "provider-state"
    });
    let message = json!({
        "type": "message",
        "role": "assistant",
        "content": "continued"
    });
    let response_object = json!({
        "id": "resp_compaction",
        "created_at": 1_719_900_000,
        "model": "gpt-4.1",
        "status": "completed",
        "output": [compaction.clone(), message.clone()]
    });

    let record = build_record(
        response_object,
        owner("alice"),
        Some(json!([{"role": "user", "content": "Continue"}])),
        Some(vec![json!({"role": "user", "content": "Start"}), compaction.clone()]),
        StoredOutputPlan::EMPTY,
    )
    .expect("provider compaction response should build a record");

    assert_eq!(
        record.messages,
        json!([
            {"role": "user", "content": "Start"},
            {"type": "compaction", "id": "cmp_provider", "encrypted_content": "provider-state"},
            {"type": "message", "role": "assistant", "content": "continued"}
        ]),
        "provider compaction must be persisted once for replay"
    );
}

#[test]
fn build_record_does_not_duplicate_compaction_outside_overlap() {
    let compaction = json!({
        "type": "compaction",
        "id": "cmp_provider",
        "encrypted_content": "provider-state"
    });
    let response_object = json!({
        "id": "resp_compaction_outside_overlap",
        "created_at": 1_719_900_000,
        "model": "gpt-4.1",
        "status": "completed",
        "output": [
            {"type": "message", "role": "assistant", "content": "continued"},
            compaction.clone()
        ]
    });

    let record = build_record(
        response_object,
        owner("alice"),
        Some(json!([{"role": "user", "content": "Continue"}])),
        Some(vec![json!({"role": "user", "content": "Start"}), compaction.clone()]),
        StoredOutputPlan::EMPTY,
    )
    .expect("provider compaction response should build a record");

    assert_eq!(
        record.messages,
        json!([
            {"role": "user", "content": "Start"},
            {"type": "compaction", "id": "cmp_provider", "encrypted_content": "provider-state"},
            {"type": "message", "role": "assistant", "content": "continued"}
        ]),
        "a replayed compaction outside the overlap must still be persisted once"
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

    let record = build_record(response_object, owner("alice"), None, None, StoredOutputPlan::EMPTY)
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
    let store = scoped_store(&reg, &alice);

    store
        .persist_response_with_pending_approvals(&sample_record(&alice, "resp_rt"), &[])
        .await
        .expect("persist should succeed");

    let fetched = store
        .get_response("resp_rt")
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
    let store = scoped_store(&reg, &alice);
    store
        .persist_response_with_pending_approvals(&sample_record(&alice, "resp_del"), &[])
        .await
        .unwrap();

    assert!(
        store.delete_response("resp_del").await.unwrap(),
        "deleting an existing record returns true"
    );
    assert!(
        store.get_response("resp_del").await.unwrap().is_none(),
        "the record is gone after delete"
    );
    assert!(
        !store.delete_response("resp_del").await.unwrap(),
        "deleting a missing record returns false"
    );
}

#[tokio::test]
async fn get_is_owner_scoped() {
    let reg = registry();
    let alice = owner("alice");
    let bob = owner("bob");
    scoped_store(&reg, &alice)
        .persist_response_with_pending_approvals(&sample_record(&alice, "resp_iso"), &[])
        .await
        .unwrap();

    let bob_view = scoped_store(&reg, &bob)
        .get_response("resp_iso")
        .await
        .expect("get should succeed");
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

    // Store handle bound to alice, record built for bob: the owner-scoped facade must reject it.
    let err = scoped_store(&reg, &alice)
        .persist_response_with_pending_approvals(&sample_record(&bob, "resp_forge"), &[])
        .await
        .expect_err("an owner mismatch must fail closed");
    assert!(
        matches!(err, StoreError::InvalidInput(_)),
        "owner mismatch is an invalid-input error"
    );

    assert!(
        scoped_store(&reg, &bob)
            .get_response("resp_forge")
            .await
            .unwrap()
            .is_none(),
        "the rejected record must not be written under any owner"
    );
}

#[tokio::test]
async fn pending_approvals_persist_then_consume_and_reject_replay() {
    let reg = registry();
    let alice = owner("alice");
    let store = scoped_store(&reg, &alice);
    let approval = PendingApprovalRecord {
        approval_id: "call_1".to_owned(),
        server_label: "weather".to_owned(),
        tool_name: "get_weather".to_owned(),
        arguments: "{}".to_owned(),
        target_fingerprint: "fp".to_owned(),
    };

    store
        .persist_response_with_pending_approvals(&sample_record(&alice, "resp_appr"), std::slice::from_ref(&approval))
        .await
        .expect("persist with pending approvals should succeed");

    let pending = store.get_pending_approvals("resp_appr", &["call_1"]).await.unwrap();
    assert_eq!(pending.len(), 1, "the pending approval should be readable");

    let consumed = store
        .consume_approvals("resp_appr", &["call_1"], 1_719_900_100)
        .await
        .unwrap();
    assert!(consumed.is_none(), "the first consumption claims the whole batch");

    let replay = store
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
    rotate_trailing_reasoning(&mut items, &[0..=1]);
    assert_eq!(types(&items), ["reasoning", "message"]);
}

#[test]
fn trailing_reasoning_moves_ahead_of_a_message_and_tool_call_turn() {
    let mut items = vec![
        json!({"type": "message", "role": "assistant", "id": "msg_1"}),
        json!({"type": "function_call", "id": "fc_1"}),
        json!({"type": "reasoning", "id": "rs_1", "content": [{"type": "reasoning_text", "text": "late"}]}),
    ];
    rotate_trailing_reasoning(&mut items, &[0..=2]);
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
    rotate_trailing_reasoning(&mut items, &[]);
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
    rotate_trailing_reasoning(&mut items, &[]);
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
    rotate_trailing_reasoning(&mut items, &[]);
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
    let record = build_record(response, owner("alice"), Some(input), None, replay_plan(&[0..=1]))
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
    let input = assemble_stored_messages(json!([]), Some(&output), replay_plan(&[1..=2, 3..=4]));
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
    let input = assemble_stored_messages(json!([]), Some(&output), replay_plan(&[0..=1]));
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
    assert_eq!(
        assemble_stored_messages(json!([]), Some(&output), replay_plan(&[])),
        output
    );
}

// -----------------------------------------------------------------------------
// Agentic provenance reconciliation
//
// A collector persists reasoning and tool-call items (never assistant messages)
// into `persisted_messages`, so the final output must not be appended wholesale.
// These tests drive `assemble_stored_messages` with the round boundaries and item
// provenance a collector records, for both reasoning-first and late output.
// -----------------------------------------------------------------------------

/// Build one collected round's boundaries.
fn round(output_start: usize, output_end: usize, persisted_start: usize, persisted_end: usize) -> CollectedRound {
    CollectedRound {
        output_start,
        output_end,
        persisted_start,
        persisted_end,
    }
}

/// Build an agentic plan from recorded rounds, provenance, and replay ranges.
fn agentic_plan<'a>(
    reasoning_replay: &'a [RangeInclusive<usize>],
    collected_rounds: &'a [CollectedRound],
    collected_provenance: &'a [(usize, usize)],
) -> StoredOutputPlan<'a> {
    StoredOutputPlan {
        reasoning_replay,
        collected_rounds,
        collected_provenance,
    }
}

fn types_of(value: &Value) -> Vec<&str> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item.get("type").and_then(Value::as_str).unwrap_or("message"))
        .collect()
}

#[test]
fn agentic_reasoning_first_round_is_not_duplicated() {
    let history = json!([
        {"type": "message", "role": "user", "content": "q"},
        {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "thought"}]},
    ]);
    let output = json!([
        {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "thought"}]},
        {"type": "message", "role": "assistant", "content": "answer"},
    ]);
    let rounds = [round(0, 2, 1, 2)];
    let provenance = [(0, 1)];
    let stored = assemble_stored_messages(history, Some(&output), agentic_plan(&[0..=0], &rounds, &provenance));
    assert_eq!(types_of(&stored), ["message", "reasoning", "message"]);
    assert_eq!(stored.as_array().unwrap()[2]["content"], "answer");
}

#[test]
fn agentic_late_reasoning_round_is_reordered_reasoning_first() {
    let history = json!([
        {"type": "message", "role": "user", "content": "q"},
        {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "late"}]},
    ]);
    let output = json!([
        {"type": "message", "role": "assistant", "content": "answer"},
        {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "late"}]},
    ]);
    let rounds = [round(0, 2, 1, 2)];
    let provenance = [(1, 1)];
    let stored = assemble_stored_messages(history, Some(&output), agentic_plan(&[0..=1], &rounds, &provenance));
    assert_eq!(types_of(&stored), ["message", "reasoning", "message"]);
    let items = stored.as_array().unwrap();
    assert_eq!(items[1]["content"][0]["text"], "late");
    assert_eq!(items[2]["content"], "answer");
}

#[test]
fn agentic_missing_message_lands_before_its_tool_result() {
    // `[function_call, message]` round: the message is uncollected and must be
    // reinserted inside the round window, ahead of the tool result dispatch
    // appended afterwards — never after it.
    let history = json!([
        {"type": "message", "role": "user", "content": "q"},
        {"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_1", "output": "42"},
    ]);
    let output = json!([
        {"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{}"},
        {"type": "message", "role": "assistant", "content": "answer"},
    ]);
    let rounds = [round(0, 2, 1, 2)];
    let provenance = [(0, 1)];
    let stored = assemble_stored_messages(history, Some(&output), agentic_plan(&[], &rounds, &provenance));
    assert_eq!(
        types_of(&stored),
        ["message", "function_call", "message", "function_call_output"]
    );
    assert_eq!(stored.as_array().unwrap()[2]["content"], "answer");
}

#[test]
fn agentic_identical_reasoning_across_rounds_is_not_collapsed() {
    // Two rounds whose reasoning text is byte-identical.
    let same = json!([{"type": "reasoning_text", "text": "same"}]);
    let history = json!([
        {"type": "message", "role": "user", "content": "q"},
        {"type": "reasoning", "content": same},
        {"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_1", "output": "42"},
        {"type": "reasoning", "content": same},
    ]);
    let output = json!([
        {"type": "reasoning", "content": same},
        {"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{}"},
        {"type": "reasoning", "content": same},
        {"type": "message", "role": "assistant", "content": "answer"},
    ]);
    let rounds = [round(0, 2, 1, 3), round(2, 4, 4, 5)];
    let provenance = [(0, 1), (1, 2), (2, 4)];
    let stored = assemble_stored_messages(
        history,
        Some(&output),
        agentic_plan(&[0..=0, 2..=2], &rounds, &provenance),
    );
    assert_eq!(
        types_of(&stored),
        [
            "message",
            "reasoning",
            "function_call",
            "function_call_output",
            "reasoning",
            "message"
        ]
    );
}

#[test]
fn agentic_zero_persisted_all_message_round_between_two_collected_rounds() {
    // A round that persisted nothing (all-message) is still recorded, so its
    // message is placed between the surrounding rounds, not dropped or misordered.
    let history = json!([
        {"type": "message", "role": "user", "content": "q"},
        {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "one"}]},
        {"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_1", "output": "42"},
        {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "three"}]},
    ]);
    let output = json!([
        {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "one"}]},
        {"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{}"},
        {"type": "message", "role": "assistant", "content": "interlude"},
        {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "three"}]},
        {"type": "message", "role": "assistant", "content": "answer"},
    ]);
    let rounds = [round(0, 2, 1, 3), round(2, 3, 4, 4), round(3, 5, 4, 5)];
    let provenance = [(0, 1), (1, 2), (3, 4)];
    let stored = assemble_stored_messages(
        history,
        Some(&output),
        agentic_plan(&[0..=0, 3..=3], &rounds, &provenance),
    );
    let items = stored.as_array().unwrap();
    assert_eq!(
        types_of(&stored),
        [
            "message",
            "reasoning",
            "function_call",
            "function_call_output",
            "message",
            "reasoning",
            "message"
        ]
    );
    assert_eq!(items[4]["content"], "interlude");
    assert_eq!(items[6]["content"], "answer");
}

#[test]
fn agentic_uncollected_final_round_is_appended() {
    // Output items beyond the last recorded span were never collected; they are
    // appended (reordered) at the end of history.
    let history = json!([
        {"type": "message", "role": "user", "content": "q"},
        {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "one"}]},
        {"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_1", "output": "42"},
    ]);
    let output = json!([
        {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "one"}]},
        {"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{}"},
        {"type": "message", "role": "assistant", "content": "answer"},
        {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "late final"}]},
    ]);
    // Final round (output 2..4) never collected; its reasoning arrives late.
    let rounds = [round(0, 2, 1, 3)];
    let provenance = [(0, 1), (1, 2)];
    let stored = assemble_stored_messages(history, Some(&output), agentic_plan(&[2..=3], &rounds, &provenance));
    assert_eq!(
        types_of(&stored),
        [
            "message",
            "reasoning",
            "function_call",
            "function_call_output",
            "reasoning",
            "message"
        ]
    );
    let items = stored.as_array().unwrap();
    assert_eq!(items[4]["content"][0]["text"], "late final");
    assert_eq!(items[5]["content"], "answer");
}

#[test]
fn agentic_collected_file_search_item_is_refreshed_to_its_final_output() {
    // File-search dispatch updates the item's accumulated_output copy (status and
    // results) after collection. Stored history must track that final item, not the
    // collector's pre-dispatch copy. Covers a completed and an incomplete call.
    for (final_status, final_body) in [
        ("completed", json!({"queries": ["q"], "results": [{"text": "hit"}]})),
        ("incomplete", json!({"queries": ["q"]})),
    ] {
        let history = json!([
            {"type": "message", "role": "user", "content": "q"},
            // The collector's pre-dispatch copy: still searching, no results.
            {"type": "file_search_call", "id": "fs_1", "status": "searching"},
        ]);
        let mut final_call = json!({"type": "file_search_call", "id": "fs_1", "status": final_status});
        final_call
            .as_object_mut()
            .unwrap()
            .extend(final_body.as_object().unwrap().clone());
        let output = json!([final_call.clone()]);
        let rounds = [round(0, 1, 1, 2)];
        let provenance = [(0, 1)];
        let stored = assemble_stored_messages(history, Some(&output), agentic_plan(&[], &rounds, &provenance));
        let items = stored.as_array().unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(
            items[1], final_call,
            "stored file-search item must match the final output"
        );
    }
}

#[test]
fn agentic_reorder_is_a_noop_when_named_item_is_not_reasoning() {
    // Fail-safe: if the range's named end item is not a reasoning item (changed or
    // mismatched), history is left untouched rather than reordered by a guess.
    let history = json!([
        {"type": "message", "role": "user", "content": "q"},
        {"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{}"},
    ]);
    let output = json!([
        {"type": "message", "role": "assistant", "content": "answer"},
        {"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{}"},
    ]);
    let rounds = [round(0, 2, 1, 2)];
    let provenance = [(1, 1)];
    // Range names [0..=1] but item 1 is a function_call, not reasoning: no reorder.
    let stored = assemble_stored_messages(history, Some(&output), agentic_plan(&[0..=1], &rounds, &provenance));
    assert_eq!(types_of(&stored), ["message", "message", "function_call"]);
}

#[test]
fn non_agentic_dedup_remaps_late_reasoning_after_dropping_echoed_prefix() {
    // The output echoes a state item as its prefix, which the overlap dedup drops.
    // Late-reasoning ranges index the full output, so they must be remapped onto
    // the shorter appended slice; otherwise the rotation targets the wrong items
    // (or falls out of bounds and never runs).
    let echoed = json!({"type": "message", "role": "user", "content": "echo"});
    let output = json!([
        echoed.clone(),
        {"type": "message", "role": "assistant", "content": "answer"},
        {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "late"}]},
    ]);
    // Range 1..=2 spans the assistant message and its late reasoning in the output.
    let stored = assemble_stored_messages(json!([echoed]), Some(&output), replay_plan(&[1..=2]));
    assert_eq!(types_of(&stored), ["message", "reasoning", "message"]);
    let items = stored.as_array().unwrap();
    assert_eq!(items[0]["content"], "echo", "echoed prefix is stored once");
    assert_eq!(
        items[1]["content"][0]["text"], "late",
        "late reasoning rotated to the front of its turn"
    );
    assert_eq!(items[2]["content"], "answer");
}
#[test]
fn list_input_items_duplicate_ids_reach_later_items() {
    let alice = owner("alice");
    let record = ResponseRecord {
        input: json!([
            {"id": "dup", "type": "item_reference"},
            {"id": "dup", "type": "item_reference"},
            {"id": "c", "type": "item_reference"}
        ]),
        ..sample_record(&alice, "resp_duplicate_ids")
    };
    let page_params = |cursor| ListParams {
        cursor,
        limit: 1,
        order: Order::Ascending,
    };

    let first = list_input_items(&record, &page_params(None), IncludeFields::default()).unwrap();
    assert_eq!(
        first.data.first().unwrap()["id"],
        "dup",
        "the first reference target must remain unchanged"
    );
    assert!(
        first.has_more,
        "the first duplicate must not hide remaining input items"
    );
    let first_cursor = first
        .next_cursor
        .clone()
        .expect("a non-final duplicate page must expose a position cursor");

    let second = list_input_items(
        &record,
        &page_params(Some(first_cursor.clone())),
        IncludeFields::default(),
    )
    .unwrap();
    let second_cursor = second
        .next_cursor
        .clone()
        .expect("the second non-final duplicate page must expose a position cursor");
    assert_ne!(
        second_cursor, first_cursor,
        "each duplicate occurrence must have a distinct continuation cursor"
    );
    assert_eq!(
        second.data.first().unwrap()["id"],
        "dup",
        "pagination must not rewrite the repeated reference target"
    );
    assert!(second.has_more, "the later unique item must remain reachable");

    let third = list_input_items(
        &record,
        &page_params(Some(second_cursor.clone())),
        IncludeFields::default(),
    )
    .unwrap();
    assert_eq!(
        third.data.first().unwrap()["id"],
        "c",
        "pagination must reach the later unique item"
    );
    assert_eq!(
        third.last_id(),
        Some("c"),
        "the final list ID must match the final item"
    );
    assert!(!third.has_more, "pagination must terminate after the later unique item");
    assert_ne!(
        second_cursor, "c",
        "the position cursor must remain separate from item IDs"
    );
}

#[test]
fn list_input_items_duplicate_cursor_avoids_explicit_id_collision() {
    let alice = owner("alice");
    let colliding_id = "praxis_input_items_offset:1:x";
    let record = ResponseRecord {
        input: json!([
            {"id": "dup", "type": "item_reference"},
            {"id": "dup", "type": "item_reference"},
            {"id": colliding_id, "type": "item_reference"}
        ]),
        ..sample_record(&alice, "resp_cursor_collision")
    };
    let first = list_input_items(
        &record,
        &ListParams {
            cursor: None,
            limit: 1,
            order: Order::Ascending,
        },
        IncludeFields::default(),
    )
    .unwrap();
    let cursor = first
        .next_cursor
        .expect("a non-final duplicate page must expose a position cursor");

    assert_ne!(
        cursor, colliding_id,
        "a position cursor must not collide with any explicit reference ID"
    );
}
