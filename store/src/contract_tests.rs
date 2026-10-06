// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Reusable backend-contract suite, shared by the in-memory double and the SQL
//! backends. Enabled by the `test-support` feature. Each function drives one
//! backend through the trait surface and asserts the shared contract, so a new
//! backend proves conformance by running the same suite.

#![allow(clippy::expect_used, clippy::panic, reason = "reusable contract harness")]

use crate::{
    owner::StateOwner,
    traits::{ConversationItemStore, PersistedStateBackend},
    types::{
        ConversationItemRecord, ConversationRecord, EventLogStatus, PendingApprovalRecord, ResponseEventRecord,
        ResponseRecord, StoreError,
    },
};

/// Build a test owner from trusted parts.
fn owner(tenant: &str) -> StateOwner {
    StateOwner::from_trusted_parts(tenant, "issuer", "subject").unwrap_or_else(|err| panic!("owner: {err}"))
}

/// A conversation item scoped to `owner`/`conversation_id`.
fn item(owner: &StateOwner, conversation_id: &str, item_id: &str) -> ConversationItemRecord {
    ConversationItemRecord {
        item_id: item_id.to_owned(),
        owner: owner.clone(),
        conversation_id: conversation_id.to_owned(),
        item_data: serde_json::json!({ "id": item_id }),
        created_at: 0,
        position: 0,
    }
}

/// A pending approval fixture.
fn approval(id: &str) -> PendingApprovalRecord {
    PendingApprovalRecord {
        approval_id: id.to_owned(),
        server_label: "srv".to_owned(),
        tool_name: "tool".to_owned(),
        arguments: "{}".to_owned(),
        target_fingerprint: "fp".to_owned(),
    }
}

/// A normalized SSE event fixture scoped to `owner`/`response_id`.
fn event(owner: &StateOwner, response_id: &str, sequence_number: u64, terminal: bool) -> ResponseEventRecord {
    let event_type = if terminal {
        "response.completed"
    } else {
        "response.output_text.delta"
    };
    ResponseEventRecord {
        response_id: response_id.to_owned(),
        owner: owner.clone(),
        sequence_number,
        event_type: event_type.to_owned(),
        payload: serde_json::to_vec(&serde_json::json!({ "type": event_type, "sequence_number": sequence_number }))
            .expect("event payload serializes"),
        terminal,
        created_at: 1,
    }
}

/// Persist an empty parent response so its owner-scoped event log can be written.
async fn persist_parent_response(backend: &dyn PersistedStateBackend, owner: &StateOwner, response_id: &str) {
    backend
        .upsert_response(&ResponseRecord {
            id: response_id.to_owned(),
            owner: owner.clone(),
            created_at: 1,
            model: "m".to_owned(),
            response_object: serde_json::json!({}),
            input: serde_json::json!({}),
            messages: serde_json::json!([]),
        })
        .await
        .expect("upsert parent response");
}

/// Run the whole backend contract suite. Panics on the first violation.
///
/// The backend must start empty. Callers run this inside their own async test.
///
/// # Panics
///
/// Panics if the backend violates the persistence contract.
#[expect(
    clippy::large_stack_frames,
    reason = "sequentially runs the complete backend contract"
)]
pub async fn run_contract_suite(backend: &dyn PersistedStateBackend) {
    responses_are_owner_scoped(backend).await;
    response_id_is_globally_unique(backend).await;
    approvals_consume_all_or_nothing(backend).await;
    approvals_require_an_owner_matched_response(backend).await;
    persist_pairs_response_and_approvals(backend).await;
    insert_if_absent_preserves_prior_response(backend).await;
    conversation_messages_cas(backend).await;
    conversation_id_is_globally_unique(backend).await;
    items_sync_positions_and_messages(backend).await;
    bounded_item_rebuild_rolls_back(backend).await;
    bounded_rebuild_rechecks_intervening_append(backend).await;
    item_sync_delete_rolls_back_without_parent(backend).await;
    item_ids_are_owner_scoped(backend).await;
    item_positions_are_owner_scoped_and_atomic(backend).await;
    item_writes_enforce_parent_scope(backend).await;
    event_log_appends_and_lists_in_order(backend).await;
    event_log_append_is_insert_if_absent(backend).await;
    event_log_requires_owner_matched_response(backend).await;
    event_log_status_gates_on_terminal(backend).await;
    event_log_removed_with_response(backend).await;
}

/// A conversation id is globally unique in the SQL schema: another owner may
/// not overwrite the existing row.
#[expect(clippy::too_many_lines, reason = "linear contract assertions")]
async fn conversation_id_is_globally_unique(backend: &dyn PersistedStateBackend) {
    let (a, b) = (owner("conversation-id-a"), owner("conversation-id-b"));
    let base = ConversationRecord {
        conversation_id: "conv_global_id".to_owned(),
        owner: a.clone(),
        created_at: 1,
        metadata: serde_json::json!({}),
        messages: serde_json::json!([]),
    };
    backend.upsert_conversation(&base).await.expect("first owner upsert");
    let collision = ConversationRecord {
        owner: b.clone(),
        ..base
    };
    assert!(
        backend.upsert_conversation(&collision).await.is_err(),
        "cross-owner conversation id collision accepted"
    );
    assert!(
        ConversationItemStore::get_conversation(backend, &a, "conv_global_id")
            .await
            .expect("get original owner")
            .is_some(),
        "original conversation lost after rejected collision"
    );
    assert!(
        ConversationItemStore::get_conversation(backend, &b, "conv_global_id")
            .await
            .expect("get second owner")
            .is_none(),
        "conversation leaked to colliding owner"
    );
}

/// A response is visible only to its owner, and delete is scoped.
async fn responses_are_owner_scoped(backend: &dyn PersistedStateBackend) {
    let (a, b) = (owner("resp-a"), owner("resp-b"));
    let record = ResponseRecord {
        id: "resp_1".to_owned(),
        owner: a.clone(),
        created_at: 1,
        model: "m".to_owned(),
        response_object: serde_json::json!({}),
        input: serde_json::json!({}),
        messages: serde_json::json!([]),
    };
    backend.upsert_response(&record).await.expect("upsert");
    assert!(
        backend.get_response(&a, "resp_1").await.expect("get a").is_some(),
        "owner sees own response"
    );
    assert!(
        backend.get_response(&b, "resp_1").await.expect("get b").is_none(),
        "response leaked across owners"
    );
    assert!(
        !backend.delete_response(&b, "resp_1").await.expect("delete b"),
        "cross-owner delete succeeded"
    );
    assert!(
        backend.delete_response(&a, "resp_1").await.expect("delete a"),
        "owner delete failed"
    );
}

/// A response id is globally unique: a colliding id owned by another principal
/// is rejected, not overwritten.
#[expect(clippy::too_many_lines, reason = "linear contract assertions")]
async fn response_id_is_globally_unique(backend: &dyn PersistedStateBackend) {
    let (a, b) = (owner("dup-a"), owner("dup-b"));
    let base = ResponseRecord {
        id: "resp_dup".to_owned(),
        owner: a.clone(),
        created_at: 1,
        model: "m".to_owned(),
        response_object: serde_json::json!({}),
        input: serde_json::json!({}),
        messages: serde_json::json!([]),
    };
    backend.upsert_response(&base).await.expect("first owner upsert");
    backend
        .upsert_response(&base)
        .await
        .expect("same-owner re-upsert must be allowed");
    let collision = ResponseRecord {
        owner: b.clone(),
        ..base.clone()
    };
    assert!(
        backend.upsert_response(&collision).await.is_err(),
        "cross-owner id collision accepted"
    );
    assert!(
        backend.get_response(&a, "resp_dup").await.expect("get a").is_some(),
        "original owner's response lost after a rejected collision"
    );
    assert!(
        backend.get_response(&b, "resp_dup").await.expect("get b").is_none(),
        "collision leaked a row to the second owner"
    );
}

/// `consume_approvals` is single-use and all-or-nothing.
#[expect(clippy::too_many_lines, reason = "linear contract assertions")]
async fn approvals_consume_all_or_nothing(backend: &dyn PersistedStateBackend) {
    let o = owner("appr");
    // Approvals belong to a persisted response; write the parent first so a
    // backend with a foreign key from approvals to responses accepts them.
    backend
        .upsert_response(&ResponseRecord {
            id: "resp_appr".to_owned(),
            owner: o.clone(),
            created_at: 1,
            model: "m".to_owned(),
            response_object: serde_json::json!({}),
            input: serde_json::json!({}),
            messages: serde_json::json!([]),
        })
        .await
        .expect("upsert response for approvals");
    let approvals = [approval("a1"), approval("a2")];
    backend
        .record_pending_approvals(&o, "resp_appr", &approvals, 10)
        .await
        .expect("record");
    // Idempotent record: re-recording does not reset state.
    backend
        .record_pending_approvals(&o, "resp_appr", &approvals, 11)
        .await
        .expect("record twice");
    let found = backend
        .get_pending_approvals(&o, "resp_appr", &["a1", "a2"])
        .await
        .expect("get");
    assert_eq!(found.len(), 2, "both approvals retrievable");
    assert_eq!(
        backend
            .consume_approvals(&o, "resp_appr", &["a1"], 20)
            .await
            .expect("consume a1"),
        None,
        "first consume of an outstanding approval must succeed"
    );
    // A batch including the already-consumed a1 aborts wholesale.
    assert_eq!(
        backend
            .consume_approvals(&o, "resp_appr", &["a2", "a1"], 30)
            .await
            .expect("consume batch"),
        Some(1),
        "batch with a consumed id must abort at that index"
    );
    // a2 stayed outstanding through the aborted batch.
    assert_eq!(
        backend
            .consume_approvals(&o, "resp_appr", &["a2"], 40)
            .await
            .expect("consume a2"),
        None,
        "outstanding approval must survive an aborted batch"
    );
}

/// Approval rows cannot exist without an issuing response owned by the caller.
#[expect(clippy::too_many_lines, reason = "linear parent and owner contract assertions")]
async fn approvals_require_an_owner_matched_response(backend: &dyn PersistedStateBackend) {
    let issuing_owner = owner("approval-parent");
    let other = owner("approval-parent-other");
    let pending = approval("orphan-approval");

    backend
        .record_pending_approvals(&issuing_owner, "missing-response", std::slice::from_ref(&pending), 1)
        .await
        .expect("missing response makes the approval insert a no-op");
    assert!(
        backend
            .get_pending_approvals(&issuing_owner, "missing-response", &["orphan-approval"])
            .await
            .expect("read missing-response approvals")
            .is_empty(),
        "approval persisted without its issuing response"
    );
    assert_eq!(
        backend
            .consume_approvals(&issuing_owner, "missing-response", &["orphan-approval"], 2)
            .await
            .expect("consume missing-response approval"),
        Some(0),
        "orphaned approval was consumable"
    );

    backend
        .upsert_response(&ResponseRecord {
            id: "owned-response".to_owned(),
            owner: issuing_owner,
            created_at: 1,
            model: "m".to_owned(),
            response_object: serde_json::json!({}),
            input: serde_json::json!({}),
            messages: serde_json::json!([]),
        })
        .await
        .expect("upsert owner-matched response");
    backend
        .record_pending_approvals(&other, "owned-response", &[approval("wrong-owner-approval")], 3)
        .await
        .expect("wrong owner makes the approval insert a no-op");
    assert!(
        backend
            .get_pending_approvals(&other, "owned-response", &["wrong-owner-approval"])
            .await
            .expect("read wrong-owner approvals")
            .is_empty(),
        "approval persisted under an owner that does not own its issuing response"
    );
}

/// `persist_response_with_pending_approvals` writes the response and its
/// approvals together, and deleting the response takes its approvals with it.
#[expect(clippy::too_many_lines, reason = "linear contract assertions")]
async fn persist_pairs_response_and_approvals(backend: &dyn PersistedStateBackend) {
    let o = owner("persist");
    let record = ResponseRecord {
        id: "resp_persist".to_owned(),
        owner: o.clone(),
        created_at: 1,
        model: "m".to_owned(),
        response_object: serde_json::json!({}),
        input: serde_json::json!({}),
        messages: serde_json::json!([]),
    };
    backend
        .persist_response_with_pending_approvals(&record, &[approval("pa1")])
        .await
        .expect("persist response with approvals");
    assert!(
        backend.get_response(&o, "resp_persist").await.expect("get").is_some(),
        "persisted response missing"
    );
    assert_eq!(
        backend
            .get_pending_approvals(&o, "resp_persist", &["pa1"])
            .await
            .expect("get approvals")
            .len(),
        1,
        "persisted approval missing"
    );
    assert!(
        backend.delete_response(&o, "resp_persist").await.expect("delete"),
        "delete of the persisted response failed"
    );
    assert!(
        backend
            .get_pending_approvals(&o, "resp_persist", &["pa1"])
            .await
            .expect("get after delete")
            .is_empty(),
        "approval orphaned after its response was deleted"
    );
}

/// A rollback candidate must never replace a preexisting response or its
/// approvals, including when a provider reuses an ID for the same owner.
#[expect(
    clippy::too_many_lines,
    reason = "checks same-owner and cross-owner collisions with approvals"
)]
async fn insert_if_absent_preserves_prior_response(backend: &dyn PersistedStateBackend) {
    let first_owner = owner("insert-first");
    let second_owner = owner("insert-second");
    let original = ResponseRecord {
        id: "resp_insert_once".to_owned(),
        owner: first_owner.clone(),
        created_at: 1,
        model: "old".to_owned(),
        response_object: serde_json::json!({"id": "resp_insert_once", "marker": "old"}),
        input: serde_json::json!([]),
        messages: serde_json::json!([]),
    };
    assert!(
        backend
            .persist_response_with_pending_approvals_if_absent(&original, &[approval("pa_old")])
            .await
            .expect("first insert"),
        "new response must insert"
    );
    let replacement = ResponseRecord {
        model: "new".to_owned(),
        response_object: serde_json::json!({"id": "resp_insert_once", "marker": "new"}),
        ..original.clone()
    };
    assert!(
        !backend
            .persist_response_with_pending_approvals_if_absent(&replacement, &[approval("pa_new")])
            .await
            .expect("same-owner collision"),
        "same-owner collision must not replace the existing response"
    );
    assert!(
        !backend
            .persist_response_with_pending_approvals_if_absent(
                &ResponseRecord {
                    owner: second_owner.clone(),
                    ..replacement
                },
                &[approval("pa_other")],
            )
            .await
            .expect("cross-owner collision"),
        "cross-owner collision must not replace the existing response"
    );
    let stored = backend
        .get_response(&first_owner, &original.id)
        .await
        .expect("get original")
        .expect("original retained");
    assert_eq!(
        stored.response_object.get("marker"),
        Some(&serde_json::json!("old")),
        "same-owner collision changed the original response"
    );
    assert!(
        backend
            .get_response(&second_owner, &original.id)
            .await
            .expect("get other")
            .is_none(),
        "cross-owner collision leaked the original response"
    );
    assert_eq!(
        backend
            .get_pending_approvals(&first_owner, &original.id, &["pa_old", "pa_new", "pa_other"])
            .await
            .expect("get approvals")
            .len(),
        1,
        "collisions must not write replacement approvals"
    );
}

/// `compare_and_swap_conversation_messages` guards a stale write.
async fn conversation_messages_cas(backend: &dyn PersistedStateBackend) {
    let o = owner("cas");
    backend
        .upsert_conversation(&ConversationRecord {
            conversation_id: "conv_cas".to_owned(),
            owner: o.clone(),
            created_at: 1,
            metadata: serde_json::json!({}),
            messages: serde_json::json!(["v0"]),
        })
        .await
        .expect("upsert conversation");
    let v0 = serde_json::json!(["v0"]);
    let v1 = serde_json::json!(["v1"]);
    assert!(
        backend
            .compare_and_swap_conversation_messages(&o, "conv_cas", &v0, &v1)
            .await
            .expect("cas match"),
        "cas on the current value must succeed"
    );
    assert!(
        !backend
            .compare_and_swap_conversation_messages(&o, "conv_cas", &v0, &v1)
            .await
            .expect("cas stale"),
        "cas on a stale value must fail"
    );
}

/// `create_items_and_sync_messages` assigns sequential positions and rebuilds
/// the cache; `delete_item_and_sync_messages` rebuilds it again.
#[expect(clippy::too_many_lines, reason = "linear contract assertions")]
async fn items_sync_positions_and_messages(backend: &dyn PersistedStateBackend) {
    let o = owner("items");
    backend
        .upsert_conversation(&ConversationRecord {
            conversation_id: "conv_items".to_owned(),
            owner: o.clone(),
            created_at: 1,
            metadata: serde_json::json!({}),
            messages: serde_json::json!([]),
        })
        .await
        .expect("upsert conversation");
    backend
        .create_items_and_sync_messages(
            &o,
            "conv_items",
            &[item(&o, "conv_items", "it1"), item(&o, "conv_items", "it2")],
        )
        .await
        .expect("create items");
    assert_eq!(
        backend
            .conversation_item_position(&o, "conv_items", "it1")
            .await
            .expect("pos1"),
        Some(1),
        "first item assigned position 1"
    );
    assert_eq!(
        backend
            .conversation_item_position(&o, "conv_items", "it2")
            .await
            .expect("pos2"),
        Some(2),
        "second item assigned position 2"
    );
    assert_eq!(
        backend.max_item_position(&o, "conv_items").await.expect("max"),
        2,
        "max position is 2"
    );
    let listed = backend
        .list_conversation_items(&o, "conv_items", None, 10, true)
        .await
        .expect("list");
    assert_eq!(listed.len(), 2, "both items listed");
    assert!(
        backend
            .delete_item_and_sync_messages(&o, "conv_items", "it1")
            .await
            .expect("delete item"),
        "delete of an existing item must report true"
    );
    let remaining = ConversationItemStore::get_conversation(backend, &o, "conv_items")
        .await
        .expect("get conversation")
        .expect("conversation present");
    assert_eq!(
        remaining.messages.as_array().map(Vec::len),
        Some(1),
        "message cache rebuilt after delete"
    );
}

/// A failed bounded rebuild must not commit the new item or change the cache.
#[expect(clippy::too_many_lines, reason = "transactional item and cache rollback assertions")]
async fn bounded_item_rebuild_rolls_back(backend: &dyn PersistedStateBackend) {
    let o = owner("bounded-rollback");
    let conversation_id = "conv_bounded_rollback";
    backend
        .upsert_conversation(&ConversationRecord {
            conversation_id: conversation_id.to_owned(),
            owner: o.clone(),
            created_at: 1,
            metadata: serde_json::json!({}),
            messages: serde_json::json!([]),
        })
        .await
        .expect("create conversation");
    backend
        .create_items_and_sync_messages(&o, conversation_id, &[item(&o, conversation_id, "prior")])
        .await
        .expect("seed prior item and cache");
    let before = ConversationItemStore::get_conversation(backend, &o, conversation_id)
        .await
        .expect("read prior cache")
        .expect("conversation exists");

    assert!(
        matches!(
            backend
                .create_items_and_sync_messages_bounded(&o, conversation_id, &[item(&o, conversation_id, "new")], 1)
                .await,
            Err(StoreError::PayloadTooLarge)
        ),
        "oversized rebuild must fail"
    );
    let listed = backend
        .list_conversation_items(&o, conversation_id, None, 10, true)
        .await
        .expect("list after failed rebuild");
    assert_eq!(listed.len(), 1, "failed append must not commit a second item");
    assert_eq!(
        listed.first().map(|item| item.item_id.as_str()),
        Some("prior"),
        "prior item must survive the rollback"
    );
    let after = ConversationItemStore::get_conversation(backend, &o, conversation_id)
        .await
        .expect("read cache after failed rebuild")
        .expect("conversation survives");
    assert_eq!(
        after.messages, before.messages,
        "failed append must not change the cached messages"
    );
}

/// A request's earlier bounded restore cannot authorize a later cache rebuild:
/// another writer may grow the conversation between those two operations.
#[expect(
    clippy::too_many_lines,
    reason = "the restore, intervening write, and rollback assertions form one scenario"
)]
async fn bounded_rebuild_rechecks_intervening_append(backend: &dyn PersistedStateBackend) {
    let o = owner("intervening-append");
    let id = "conv_intervening_append";
    backend
        .upsert_conversation(&ConversationRecord {
            conversation_id: id.to_owned(),
            owner: o.clone(),
            created_at: 1,
            metadata: serde_json::json!({}),
            messages: serde_json::json!([]),
        })
        .await
        .expect("create conversation");
    backend
        .create_items_and_sync_messages(&o, id, &[item(&o, id, "prior")])
        .await
        .expect("seed prior item");
    let snapshot = backend
        .get_conversation_bounded(&o, id, 65_536)
        .await
        .expect("bounded restore")
        .expect("conversation exists");
    assert_eq!(
        snapshot.messages.as_array().map(Vec::len),
        Some(1),
        "restore sees the first item"
    );

    let mut concurrent = item(&o, id, "concurrent");
    concurrent.item_data = serde_json::json!({"id":"concurrent", "content":"x".repeat(10_000)});
    backend
        .create_items_and_sync_messages(&o, id, &[concurrent])
        .await
        .expect("another writer appends after restore");
    let rejected = backend
        .create_items_and_sync_messages_bounded(&o, id, &[item(&o, id, "candidate")], 65_536)
        .await;
    assert!(
        matches!(rejected, Err(StoreError::PayloadTooLarge)),
        "intervening item must exceed the bound"
    );
    let listed = backend
        .list_conversation_items(&o, id, None, 10, true)
        .await
        .expect("list items");
    assert_eq!(listed.len(), 2, "failed candidate append must roll back");
    assert!(
        listed.iter().all(|record| record.item_id != "candidate"),
        "candidate must roll back"
    );
}

/// Deleting an item and rebuilding its message cache is atomic when the parent
/// conversation has already been deleted.
#[expect(clippy::too_many_lines, reason = "linear rollback contract assertions")]
async fn item_sync_delete_rolls_back_without_parent(backend: &dyn PersistedStateBackend) {
    let o = owner("orphan-sync");
    backend
        .upsert_conversation(&ConversationRecord {
            conversation_id: "conv_orphan_sync".to_owned(),
            owner: o.clone(),
            created_at: 1,
            metadata: serde_json::json!({}),
            messages: serde_json::json!([]),
        })
        .await
        .expect("upsert conversation");
    backend
        .create_items_and_sync_messages(
            &o,
            "conv_orphan_sync",
            &[item(&o, "conv_orphan_sync", "orphan_sync_item")],
        )
        .await
        .expect("create item");
    assert!(
        ConversationItemStore::delete_conversation(backend, &o, "conv_orphan_sync")
            .await
            .expect("delete parent conversation"),
        "parent conversation must exist before deletion"
    );

    assert!(
        backend
            .delete_item_and_sync_messages(&o, "conv_orphan_sync", "orphan_sync_item")
            .await
            .is_err(),
        "item delete must fail when its cache parent is absent"
    );
    assert!(
        backend
            .get_conversation_item(&o, "conv_orphan_sync", "orphan_sync_item")
            .await
            .expect("read item after rolled-back delete")
            .is_some(),
        "failed cache synchronization must preserve the item"
    );
}

/// Provider-generated item ids may collide across owners without either row
/// shadowing the other.
#[expect(clippy::too_many_lines, reason = "linear contract assertions")]
async fn item_ids_are_owner_scoped(backend: &dyn PersistedStateBackend) {
    let (a, b) = (owner("item-owner-a"), owner("item-owner-b"));
    for (owner, conversation_id) in [(&a, "conv_owner_a"), (&b, "conv_owner_b")] {
        backend
            .upsert_conversation(&ConversationRecord {
                conversation_id: conversation_id.to_owned(),
                owner: owner.clone(),
                created_at: 1,
                metadata: serde_json::json!({}),
                messages: serde_json::json!([]),
            })
            .await
            .expect("upsert owner-scoped conversation");
        backend
            .create_conversation_items(&[item(owner, conversation_id, "shared_item_id")])
            .await
            .expect("same item id must be accepted for a different owner");
    }

    assert!(
        backend
            .get_conversation_item(&a, "conv_owner_a", "shared_item_id")
            .await
            .expect("get owner a item")
            .is_some(),
        "owner a item missing"
    );
    assert!(
        backend
            .get_conversation_item(&b, "conv_owner_b", "shared_item_id")
            .await
            .expect("get owner b item")
            .is_some(),
        "owner b item missing"
    );
}

/// Item positions are unique within an owner's conversation, and a conflicting
/// batch leaves no partial rows behind.
#[expect(clippy::too_many_lines, reason = "linear uniqueness and rollback assertions")]
async fn item_positions_are_owner_scoped_and_atomic(backend: &dyn PersistedStateBackend) {
    let owner = owner("item-position");
    backend
        .upsert_conversation(&ConversationRecord {
            conversation_id: "conv_position".to_owned(),
            owner: owner.clone(),
            created_at: 1,
            metadata: serde_json::json!({}),
            messages: serde_json::json!([]),
        })
        .await
        .expect("upsert position-test conversation");

    let mut first = item(&owner, "conv_position", "position-first");
    first.position = 7;
    let mut second = item(&owner, "conv_position", "position-second");
    second.position = 7;
    assert!(
        backend
            .create_conversation_items(&[first.clone(), second.clone()])
            .await
            .is_err(),
        "duplicate positions in one batch were accepted"
    );
    assert!(
        backend
            .list_conversation_items(&owner, "conv_position", None, 10, true)
            .await
            .expect("list after rejected position batch")
            .is_empty(),
        "a rejected position batch committed partial rows"
    );

    backend
        .create_conversation_items(std::slice::from_ref(&first))
        .await
        .expect("insert first position");
    assert!(
        backend.create_conversation_items(&[second]).await.is_err(),
        "a position already held by a stored item was accepted"
    );
    let stored = backend
        .list_conversation_items(&owner, "conv_position", None, 10, true)
        .await
        .expect("list after stored-position collision");
    assert_eq!(stored.len(), 1, "stored-position collision changed existing rows");
    assert_eq!(
        stored.first().map(|item| item.item_id.as_str()),
        Some("position-first"),
        "the original item must remain after the collision"
    );
}

/// Item writes reject an orphan (no parent conversation), a cross-owner parent,
/// and a duplicate id within one batch.
#[expect(clippy::too_many_lines, reason = "linear contract assertions")]
async fn item_writes_enforce_parent_scope(backend: &dyn PersistedStateBackend) {
    let o = owner("scope");
    let other = owner("scope-other");

    // No parent conversation exists yet: both insert paths reject the orphan.
    assert!(
        backend
            .create_conversation_items(&[item(&o, "conv_scope", "orphan_1")])
            .await
            .is_err(),
        "orphan item accepted by create_conversation_items"
    );
    assert!(
        backend
            .create_items_and_sync_messages(&o, "conv_scope", &[item(&o, "conv_scope", "orphan_2")])
            .await
            .is_err(),
        "orphan item accepted by create_items_and_sync_messages"
    );

    backend
        .upsert_conversation(&ConversationRecord {
            conversation_id: "conv_scope".to_owned(),
            owner: o.clone(),
            created_at: 1,
            metadata: serde_json::json!({}),
            messages: serde_json::json!([]),
        })
        .await
        .expect("upsert conversation");

    // An item owned by a different principal cannot attach to o's conversation.
    assert!(
        backend
            .create_conversation_items(&[item(&other, "conv_scope", "mismatch_1")])
            .await
            .is_err(),
        "cross-owner item accepted by create_conversation_items"
    );
    assert!(
        backend
            .create_items_and_sync_messages(&o, "conv_scope", &[item(&other, "conv_scope", "mismatch_2")])
            .await
            .is_err(),
        "cross-owner item accepted by create_items_and_sync_messages"
    );

    // A duplicate id inside one batch is rejected on both paths.
    assert!(
        backend
            .create_items_and_sync_messages(
                &o,
                "conv_scope",
                &[item(&o, "conv_scope", "dup"), item(&o, "conv_scope", "dup")],
            )
            .await
            .is_err(),
        "intra-batch duplicate accepted by create_items_and_sync_messages"
    );
    assert!(
        backend
            .create_conversation_items(&[item(&o, "conv_scope", "dup2"), item(&o, "conv_scope", "dup2")])
            .await
            .is_err(),
        "intra-batch duplicate accepted by create_conversation_items"
    );
}

/// Events append in any order and list ascending by `sequence_number`; the
/// `after` cursor skips sequences `<= after`; `limit` caps the page.
#[expect(clippy::too_many_lines, reason = "linear ordering and cursor assertions")]
async fn event_log_appends_and_lists_in_order(backend: &dyn PersistedStateBackend) {
    let o = owner("events-order");
    persist_parent_response(backend, &o, "resp_events_order").await;

    // Two batches, each internally out of order, prove ordering is by
    // sequence_number and not insertion order.
    backend
        .append_events(
            &o,
            "resp_events_order",
            &[
                event(&o, "resp_events_order", 2, false),
                event(&o, "resp_events_order", 0, false),
            ],
        )
        .await
        .expect("append first batch");
    backend
        .append_events(
            &o,
            "resp_events_order",
            &[
                event(&o, "resp_events_order", 1, false),
                event(&o, "resp_events_order", 3, true),
            ],
        )
        .await
        .expect("append second batch");

    let all = backend
        .list_events_after(&o, "resp_events_order", None, 10)
        .await
        .expect("list all");
    assert_eq!(
        all.iter().map(|event| event.sequence_number).collect::<Vec<_>>(),
        vec![0, 1, 2, 3],
        "events not listed in ascending sequence order"
    );
    assert!(
        all.last().is_some_and(|event| event.terminal),
        "terminal event missing from the replay tail"
    );

    let after_one = backend
        .list_events_after(&o, "resp_events_order", Some(1), 10)
        .await
        .expect("list after 1");
    assert_eq!(
        after_one.iter().map(|event| event.sequence_number).collect::<Vec<_>>(),
        vec![2, 3],
        "cursor did not skip sequences <= after"
    );

    let paged = backend
        .list_events_after(&o, "resp_events_order", None, 2)
        .await
        .expect("list limited");
    assert_eq!(
        paged.iter().map(|event| event.sequence_number).collect::<Vec<_>>(),
        vec![0, 1],
        "limit did not cap the first page in order"
    );
}

/// Re-appending a stored sequence number is a no-op: the first write wins and no
/// duplicate row is created.
async fn event_log_append_is_insert_if_absent(backend: &dyn PersistedStateBackend) {
    let o = owner("events-idem");
    persist_parent_response(backend, &o, "resp_events_idem").await;

    let mut original = event(&o, "resp_events_idem", 0, false);
    original.payload = serde_json::to_vec(&serde_json::json!({ "v": "original" })).expect("payload serializes");
    backend
        .append_events(&o, "resp_events_idem", std::slice::from_ref(&original))
        .await
        .expect("append original");

    let mut collision = event(&o, "resp_events_idem", 0, false);
    collision.payload = serde_json::to_vec(&serde_json::json!({ "v": "overwrite" })).expect("payload serializes");
    backend
        .append_events(&o, "resp_events_idem", std::slice::from_ref(&collision))
        .await
        .expect("re-append same sequence");

    let listed = backend
        .list_events_after(&o, "resp_events_idem", None, 10)
        .await
        .expect("list");
    assert_eq!(listed.len(), 1, "a duplicate sequence created a second row");
    assert_eq!(
        listed.first().map(|event| &event.payload),
        Some(&serde_json::to_vec(&serde_json::json!({ "v": "original" })).expect("payload serializes")),
        "insert-if-absent overwrote the already-stored event"
    );
}

/// The event log is owner-scoped, and an append requires a parent response owned
/// by the same principal; writes without one are silently dropped.
#[expect(clippy::too_many_lines, reason = "linear owner and parent-scope assertions")]
async fn event_log_requires_owner_matched_response(backend: &dyn PersistedStateBackend) {
    let (a, b) = (owner("events-owner-a"), owner("events-owner-b"));

    // No parent response exists: the append is a no-op.
    backend
        .append_events(&a, "resp_events_missing", &[event(&a, "resp_events_missing", 0, true)])
        .await
        .expect("append to a missing parent is a no-op");
    assert!(
        backend
            .list_events_after(&a, "resp_events_missing", None, 10)
            .await
            .expect("list missing")
            .is_empty(),
        "events attached to a response that does not exist"
    );

    // Parent owned by a: b can neither write to it nor observe its log.
    persist_parent_response(backend, &a, "resp_events_scope").await;
    backend
        .append_events(&b, "resp_events_scope", &[event(&b, "resp_events_scope", 0, true)])
        .await
        .expect("cross-owner append is a no-op");
    assert!(
        backend
            .list_events_after(&b, "resp_events_scope", None, 10)
            .await
            .expect("list as b")
            .is_empty(),
        "cross-owner append committed an event"
    );

    backend
        .append_events(&a, "resp_events_scope", &[event(&a, "resp_events_scope", 0, true)])
        .await
        .expect("owner append");
    assert_eq!(
        backend
            .list_events_after(&a, "resp_events_scope", None, 10)
            .await
            .expect("list as a")
            .len(),
        1,
        "owner cannot read its own event"
    );
    assert!(
        backend
            .list_events_after(&b, "resp_events_scope", None, 10)
            .await
            .expect("list as b after a wrote")
            .is_empty(),
        "owner-a event log leaked to owner b"
    );
    assert_eq!(
        backend
            .event_log_status(&b, "resp_events_scope")
            .await
            .expect("status as b"),
        EventLogStatus::Absent,
        "event-log status leaked to a non-owner"
    );
}

/// `event_log_status` distinguishes no-log, incomplete, and replayable, marking
/// the log replayable only once a terminal event is present.
#[expect(clippy::too_many_lines, reason = "linear contract assertions")]
async fn event_log_status_gates_on_terminal(backend: &dyn PersistedStateBackend) {
    let o = owner("events-status");
    persist_parent_response(backend, &o, "resp_events_status").await;

    assert_eq!(
        backend
            .event_log_status(&o, "resp_events_status")
            .await
            .expect("status with no log"),
        EventLogStatus::Absent,
        "a response with no events reported a log"
    );

    backend
        .append_events(
            &o,
            "resp_events_status",
            &[
                event(&o, "resp_events_status", 0, false),
                event(&o, "resp_events_status", 1, false),
            ],
        )
        .await
        .expect("append non-terminal events");
    assert_eq!(
        backend
            .event_log_status(&o, "resp_events_status")
            .await
            .expect("status incomplete"),
        EventLogStatus::Incomplete { max_sequence: 1 },
        "an incomplete log was misreported"
    );

    backend
        .append_events(&o, "resp_events_status", &[event(&o, "resp_events_status", 2, true)])
        .await
        .expect("append terminal event");
    assert_eq!(
        backend
            .event_log_status(&o, "resp_events_status")
            .await
            .expect("status replayable"),
        EventLogStatus::Replayable { max_sequence: 2 },
        "a replayable log was misreported"
    );
}

/// Deleting the parent response removes its event log so nothing stays
/// replayable after a DELETE.
#[expect(clippy::too_many_lines, reason = "linear contract assertions")]
async fn event_log_removed_with_response(backend: &dyn PersistedStateBackend) {
    let o = owner("events-delete");
    persist_parent_response(backend, &o, "resp_events_delete").await;
    backend
        .append_events(
            &o,
            "resp_events_delete",
            &[
                event(&o, "resp_events_delete", 0, false),
                event(&o, "resp_events_delete", 1, true),
            ],
        )
        .await
        .expect("append");
    assert!(
        backend
            .delete_response(&o, "resp_events_delete")
            .await
            .expect("delete parent response"),
        "delete of the parent response failed"
    );
    assert!(
        backend
            .list_events_after(&o, "resp_events_delete", None, 10)
            .await
            .expect("list after delete")
            .is_empty(),
        "event log survived its response"
    );
    assert_eq!(
        backend
            .event_log_status(&o, "resp_events_delete")
            .await
            .expect("status after delete"),
        EventLogStatus::Absent,
        "event-log status survived its response"
    );
}

#[cfg(test)]
mod tests {
    use super::run_contract_suite;
    use crate::memory::InMemoryStore;

    #[tokio::test]
    async fn in_memory_backend_satisfies_the_contract() {
        run_contract_suite(&InMemoryStore::new()).await;
    }
}
