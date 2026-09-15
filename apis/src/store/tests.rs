// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Tests for the response store persistence layer.

use std::sync::Arc;

use serde_json::json;

use super::{
    CompressionAlgorithm, ConversationItemRecord, ConversationRecord, PendingApprovalRecord, PostgresResponseStore,
    ResponseRecord, ResponseStoreRegistry, SqliteResponseStore, SslMode, StoreCompressionConfig, StoreError,
    trait_def::{ConversationItemStore, ResponseStore},
};
use crate::openai::{
    include::IncludeFields,
    responses::store::{ListParams, Order, list_input_items},
};

/// Default issuing-response scope for pending-approval tests that do not
/// exercise response scoping explicitly.
const RESP: &str = "resp_1";

// -----------------------------------------------------------------------------
// Schema Initialization
// -----------------------------------------------------------------------------

#[tokio::test]
async fn sqlite_store_initializes_schema() {
    let store = SqliteResponseStore::new(
        "sqlite::memory:",
        "test_responses",
        "test_conversation_messages",
        None,
        None,
        None,
    )
    .await
    .expect("store creation should succeed");

    let result = store
        .get_response("tenant_a", "nonexistent")
        .await
        .expect("get should succeed");

    assert!(result.is_none(), "empty store should return None");
}

// -----------------------------------------------------------------------------
// Response CRUD
// -----------------------------------------------------------------------------

#[tokio::test]
async fn upsert_and_get_response() {
    let store = make_store().await;
    let record = make_response_record("resp_1", "tenant_a", 1000);

    store.upsert_response(&record).await.expect("upsert should succeed");

    let fetched = store
        .get_response("tenant_a", "resp_1")
        .await
        .expect("get should succeed")
        .expect("record should exist");

    assert_eq!(fetched.id, "resp_1", "ID should match");
    assert_eq!(fetched.tenant_id, "tenant_a", "tenant should match");
    assert_eq!(fetched.created_at, 1000, "created_at should match");
    assert_eq!(fetched.model, "gpt-4.1", "model should match");
    assert_eq!(
        fetched.response_object,
        json!({"status": "completed"}),
        "response_object should match"
    );
    assert_eq!(
        fetched.input,
        json!("test input"),
        "input should survive JSON round-trip"
    );
    assert_eq!(
        fetched.messages,
        json!([{"role": "user", "content": "hello"}]),
        "messages should survive JSON round-trip"
    );
}

#[tokio::test]
async fn upsert_overwrites_existing_response() {
    let store = make_store().await;
    let record = make_response_record("resp_1", "tenant_a", 1000);
    store
        .upsert_response(&record)
        .await
        .expect("first upsert should succeed");

    let updated = ResponseRecord {
        model: "gpt-4.1-mini".to_owned(),
        response_object: json!({"status": "incomplete"}),
        ..make_response_record("resp_1", "tenant_a", 1000)
    };
    store
        .upsert_response(&updated)
        .await
        .expect("second upsert should succeed");

    let fetched = store
        .get_response("tenant_a", "resp_1")
        .await
        .expect("get should succeed")
        .expect("record should exist");

    assert_eq!(fetched.model, "gpt-4.1-mini", "model should be updated");
    assert_eq!(
        fetched.response_object,
        json!({"status": "incomplete"}),
        "response_object should be updated"
    );
}

#[tokio::test]
async fn get_missing_response_returns_none() {
    let store = make_store().await;

    let result = store
        .get_response("tenant_a", "nonexistent")
        .await
        .expect("get should succeed");

    assert!(result.is_none(), "missing record should return None");
}

#[tokio::test]
async fn delete_existing_response() {
    let store = make_store().await;
    let record = make_response_record("resp_1", "tenant_a", 1000);
    store.upsert_response(&record).await.expect("upsert should succeed");

    let deleted = store
        .delete_response("tenant_a", "resp_1")
        .await
        .expect("delete should succeed");

    assert!(deleted, "delete should return true for existing record");

    let fetched = store
        .get_response("tenant_a", "resp_1")
        .await
        .expect("get should succeed");

    assert!(fetched.is_none(), "deleted record should not be retrievable");
}

#[tokio::test]
async fn delete_missing_response_returns_false() {
    let store = make_store().await;

    let deleted = store
        .delete_response("tenant_a", "nonexistent")
        .await
        .expect("delete should succeed");

    assert!(!deleted, "delete should return false for missing record");
}

// -----------------------------------------------------------------------------
// Tenant Isolation
// -----------------------------------------------------------------------------

#[tokio::test]
async fn tenant_isolation_on_get() {
    let store = make_store().await;
    let record = make_response_record("resp_1", "tenant_a", 1000);
    store.upsert_response(&record).await.expect("upsert should succeed");

    let result = store
        .get_response("tenant_b", "resp_1")
        .await
        .expect("get should succeed");

    assert!(result.is_none(), "tenant_b should not see tenant_a records");
}

#[tokio::test]
async fn tenant_isolation_on_delete() {
    let store = make_store().await;
    let record = make_response_record("resp_1", "tenant_a", 1000);
    store.upsert_response(&record).await.expect("upsert should succeed");

    let deleted = store
        .delete_response("tenant_b", "resp_1")
        .await
        .expect("delete should succeed");

    assert!(!deleted, "tenant_b should not be able to delete tenant_a records");

    let still_exists = store
        .get_response("tenant_a", "resp_1")
        .await
        .expect("get should succeed");

    assert!(
        still_exists.is_some(),
        "record should still exist after cross-tenant delete attempt"
    );
}

#[tokio::test]
async fn same_response_id_can_exist_in_multiple_tenants() {
    let store = make_store().await;
    store
        .upsert_response(&make_response_record("resp_shared", "tenant_a", 1000))
        .await
        .expect("tenant_a upsert should succeed");
    store
        .upsert_response(&make_response_record("resp_shared", "tenant_b", 2000))
        .await
        .expect("tenant_b upsert should succeed");

    let tenant_a = store
        .get_response("tenant_a", "resp_shared")
        .await
        .expect("tenant_a get should succeed")
        .expect("tenant_a record should exist");
    let tenant_b = store
        .get_response("tenant_b", "resp_shared")
        .await
        .expect("tenant_b get should succeed")
        .expect("tenant_b record should exist");

    assert_eq!(tenant_a.tenant_id, "tenant_a", "tenant_a record should be isolated");
    assert_eq!(tenant_b.tenant_id, "tenant_b", "tenant_b record should be isolated");
    assert_eq!(tenant_a.created_at, 1000, "tenant_a record should not be overwritten");
    assert_eq!(tenant_b.created_at, 2000, "tenant_b record should not be overwritten");
}

// -----------------------------------------------------------------------------
// Approval Consumption (single-use)
// -----------------------------------------------------------------------------

#[tokio::test]
async fn consume_approval_first_call_claims() {
    let store = make_store().await;
    seed_pending(&store, "tenant_a", RESP, &["call_abc"]).await;

    let conflict = store
        .consume_approvals("tenant_a", RESP, &["call_abc"], 1000)
        .await
        .expect("consume should succeed");

    assert!(conflict.is_none(), "first consumption of an approval should be claimed");
}

#[tokio::test]
async fn consume_approval_without_pending_row_is_rejected() {
    let store = make_store().await;

    // No record_pending_approvals: a consume for an id the proxy never issued
    // has no server-owned row to claim and must fail closed rather than
    // conjuring consent from nothing.
    let conflict = store
        .consume_approvals("tenant_a", RESP, &["call_never_issued"], 1000)
        .await
        .expect("consume should succeed");

    assert_eq!(
        conflict,
        Some(0),
        "consuming an approval with no server-owned pending row must be rejected"
    );
}

#[tokio::test]
async fn consume_approval_replay_is_rejected() {
    let store = make_store().await;
    seed_pending(&store, "tenant_a", RESP, &["call_abc"]).await;

    let first = store
        .consume_approvals("tenant_a", RESP, &["call_abc"], 1000)
        .await
        .expect("first consume should succeed");
    let replay = store
        .consume_approvals("tenant_a", RESP, &["call_abc"], 2000)
        .await
        .expect("replay consume should succeed");

    assert!(first.is_none(), "first consumption should be claimed");
    assert_eq!(
        replay,
        Some(0),
        "replayed consumption of the same approval must be rejected (single-use)"
    );
}

#[tokio::test]
async fn consume_approval_distinct_ids_each_claim() {
    let store = make_store().await;
    seed_pending(&store, "tenant_a", RESP, &["call_1", "call_2"]).await;

    let first = store
        .consume_approvals("tenant_a", RESP, &["call_1"], 1000)
        .await
        .expect("consume should succeed");
    let second = store
        .consume_approvals("tenant_a", RESP, &["call_2"], 1000)
        .await
        .expect("consume should succeed");

    assert!(first.is_none(), "first distinct approval should be claimed");
    assert!(second.is_none(), "second distinct approval should be claimed");
}

#[tokio::test]
async fn consume_approval_is_tenant_scoped() {
    let store = make_store().await;
    seed_pending(&store, "tenant_a", RESP, &["call_shared"]).await;
    seed_pending(&store, "tenant_b", RESP, &["call_shared"]).await;

    let tenant_a = store
        .consume_approvals("tenant_a", RESP, &["call_shared"], 1000)
        .await
        .expect("tenant_a consume should succeed");
    let tenant_b = store
        .consume_approvals("tenant_b", RESP, &["call_shared"], 1000)
        .await
        .expect("tenant_b consume should succeed");

    assert!(tenant_a.is_none(), "tenant_a should claim its own approval");
    assert!(
        tenant_b.is_none(),
        "tenant_b sharing an approval id with tenant_a should still claim independently"
    );
}

#[tokio::test]
async fn consume_approvals_batch_claims_all_distinct_ids() {
    let store = make_store().await;
    seed_pending(&store, "tenant_a", RESP, &["call_1", "call_2", "call_3"]).await;

    let conflict = store
        .consume_approvals("tenant_a", RESP, &["call_1", "call_2", "call_3"], 1000)
        .await
        .expect("batch consume should succeed");

    assert!(conflict.is_none(), "a batch of distinct ids should claim every one");
}

#[tokio::test]
async fn consume_approvals_batch_is_all_or_nothing_on_replay() {
    let store = make_store().await;
    seed_pending(&store, "tenant_a", RESP, &["call_1", "call_2"]).await;

    // Claim call_1 on its own.
    store
        .consume_approvals("tenant_a", RESP, &["call_1"], 1000)
        .await
        .expect("first claim should succeed");

    // A batch that replays call_1 alongside a fresh call_2 must reject the
    // whole batch and leave call_2 unclaimed.
    let conflict = store
        .consume_approvals("tenant_a", RESP, &["call_1", "call_2"], 2000)
        .await
        .expect("batch consume should succeed");
    assert_eq!(conflict, Some(0), "the replayed id's index should be reported");

    // Proof of rollback: call_2 was never burned, so it still claims cleanly.
    let call_2 = store
        .consume_approvals("tenant_a", RESP, &["call_2"], 3000)
        .await
        .expect("call_2 consume should succeed");
    assert!(
        call_2.is_none(),
        "a rolled-back batch must not strand an otherwise-fresh sibling approval"
    );
}

#[tokio::test]
async fn consume_approvals_rejects_intra_batch_duplicate() {
    let store = make_store().await;
    seed_pending(&store, "tenant_a", RESP, &["call_dup"]).await;

    // The same id twice in one batch is a duplicate: the first occurrence claims
    // the row inside the transaction, so the second finds nothing outstanding.
    let conflict = store
        .consume_approvals("tenant_a", RESP, &["call_dup", "call_dup"], 1000)
        .await
        .expect("batch consume should succeed");
    assert_eq!(conflict, Some(1), "the duplicate's index should be reported");

    // Proof of rollback: the id was never burned by the rejected batch.
    let retry = store
        .consume_approvals("tenant_a", RESP, &["call_dup"], 2000)
        .await
        .expect("retry consume should succeed");
    assert!(
        retry.is_none(),
        "a batch rejected for an intra-batch duplicate must not burn the id"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn consume_approvals_concurrent_claims_exactly_once() {
    // A file-backed store gives the two tasks real (>1) connections so the
    // claim races at the database, not just in the pool. Exactly one caller
    // must observe the fresh claim; the other must be rejected.
    let dir = tempfile::tempdir().expect("temp dir should be created");
    let db_path = dir.path().join("concurrent_consume.db");
    let url = format!("sqlite://{}?mode=rwc", db_path.display());
    let store = Arc::new(
        SqliteResponseStore::new(&url, "test_responses", "test_conversation_messages", None, None, None)
            .await
            .expect("store creation should succeed"),
    );
    seed_pending(store.as_ref(), "tenant_a", RESP, &["call_race"]).await;

    let store_a = Arc::clone(&store);
    let store_b = Arc::clone(&store);
    let task_a = tokio::spawn(async move { store_a.consume_approvals("tenant_a", RESP, &["call_race"], 1000).await });
    let task_b = tokio::spawn(async move { store_b.consume_approvals("tenant_a", RESP, &["call_race"], 1000).await });

    let a = task_a
        .await
        .expect("task a should join")
        .expect("consume a should succeed");
    let b = task_b
        .await
        .expect("task b should join")
        .expect("consume b should succeed");

    let claimed = usize::from(a.is_none()) + usize::from(b.is_none());
    let rejected = usize::from(a == Some(0)) + usize::from(b == Some(0));
    assert_eq!(claimed, 1, "exactly one concurrent caller must claim the approval");
    assert_eq!(rejected, 1, "exactly one concurrent caller must be rejected");
}

// -----------------------------------------------------------------------------
// Pending Approvals (server-owned correlation records)
// -----------------------------------------------------------------------------

#[tokio::test]
async fn record_and_get_pending_approval_round_trips_fields() {
    let store = make_store().await;
    let record = make_pending("call_abc");
    store
        .record_pending_approvals("tenant_a", RESP, std::slice::from_ref(&record), 1000)
        .await
        .expect("record should succeed");

    let fetched = store
        .get_pending_approvals("tenant_a", RESP, &["call_abc"])
        .await
        .expect("get should succeed");

    assert_eq!(
        fetched,
        vec![record],
        "the fetched pending record must round-trip every field"
    );
}

#[tokio::test]
async fn get_pending_approvals_absent_id_returns_empty() {
    let store = make_store().await;
    seed_pending(&store, "tenant_a", RESP, &["call_present"]).await;

    let fetched = store
        .get_pending_approvals("tenant_a", RESP, &["call_absent"])
        .await
        .expect("get should succeed");

    assert!(
        fetched.is_empty(),
        "an id the proxy never issued must have no pending row"
    );
}

#[tokio::test]
async fn get_pending_approvals_is_tenant_scoped() {
    let store = make_store().await;
    seed_pending(&store, "tenant_a", RESP, &["call_shared"]).await;

    let other_tenant = store
        .get_pending_approvals("tenant_b", RESP, &["call_shared"])
        .await
        .expect("get should succeed");

    assert!(
        other_tenant.is_empty(),
        "a pending row for tenant_a must be invisible to tenant_b"
    );
}

#[tokio::test]
async fn get_pending_approvals_returns_consumed_rows() {
    let store = make_store().await;
    seed_pending(&store, "tenant_a", RESP, &["call_abc"]).await;
    store
        .consume_approvals("tenant_a", RESP, &["call_abc"], 2000)
        .await
        .expect("consume should succeed");

    // A consumed row is still returned so the resume path can distinguish
    // "already used" (row present, consume rejects) from "never issued"
    // (row absent).
    let fetched = store
        .get_pending_approvals("tenant_a", RESP, &["call_abc"])
        .await
        .expect("get should succeed");

    assert_eq!(fetched.len(), 1, "a consumed pending row must still be retrievable");
    assert_eq!(fetched[0].approval_id, "call_abc");
}

#[tokio::test]
async fn record_pending_approvals_is_idempotent_and_never_resets_consumption() {
    let store = make_store().await;
    seed_pending(&store, "tenant_a", RESP, &["call_abc"]).await;
    store
        .consume_approvals("tenant_a", RESP, &["call_abc"], 2000)
        .await
        .expect("first consume should succeed");

    // Re-recording the same approval (e.g. the response is persisted again on a
    // retried turn) must not resurrect an already-consumed row, or a client
    // could replay a used approval.
    seed_pending(&store, "tenant_a", RESP, &["call_abc"]).await;

    let replay = store
        .consume_approvals("tenant_a", RESP, &["call_abc"], 3000)
        .await
        .expect("replay consume should succeed");

    assert_eq!(
        replay,
        Some(0),
        "re-recording a consumed approval must not reset it to outstanding"
    );
}

#[tokio::test]
async fn get_pending_approvals_is_response_scoped() {
    let store = make_store().await;
    seed_pending(&store, "tenant_a", "resp_1", &["call_shared"]).await;

    let other_response = store
        .get_pending_approvals("tenant_a", "resp_2", &["call_shared"])
        .await
        .expect("get should succeed");

    assert!(
        other_response.is_empty(),
        "a pending row issued by resp_1 must be invisible under a different response id"
    );
}

#[tokio::test]
async fn consume_approvals_is_response_scoped() {
    // The same approval id issued by two different responses is two independent
    // single-use tokens. A resume that names the wrong originating response must
    // not be able to claim either, and consuming one must not consume the other.
    let store = make_store().await;
    seed_pending(&store, "tenant_a", "resp_1", &["call_shared"]).await;
    seed_pending(&store, "tenant_a", "resp_2", &["call_shared"]).await;

    // Claiming the token under an unrelated response id finds no row and rejects.
    let wrong = store
        .consume_approvals("tenant_a", "resp_other", &["call_shared"], 1000)
        .await
        .expect("consume should succeed");
    assert_eq!(
        wrong,
        Some(0),
        "an approval scoped to another response must not be claimable"
    );

    // Each issuing response's token is claimable exactly once, independently.
    let first = store
        .consume_approvals("tenant_a", "resp_1", &["call_shared"], 1000)
        .await
        .expect("consume should succeed");
    let second = store
        .consume_approvals("tenant_a", "resp_2", &["call_shared"], 1000)
        .await
        .expect("consume should succeed");
    assert!(first.is_none(), "resp_1's token should claim cleanly");
    assert!(
        second.is_none(),
        "resp_2's independent token should still claim cleanly"
    );

    // Replaying resp_1's now-consumed token rejects: it is single-use per response.
    let replay = store
        .consume_approvals("tenant_a", "resp_1", &["call_shared"], 2000)
        .await
        .expect("consume should succeed");
    assert_eq!(replay, Some(0), "resp_1's token is single-use");
}

#[tokio::test]
async fn delete_response_removes_its_pending_approvals() {
    // Deleting the originating response must not leave its approval consumable or
    // retain the sensitive arguments indefinitely: the pending rows scoped to that
    // response are removed transactionally with the response itself.
    let store = make_store().await;
    let record = make_response_record("resp_del", "tenant_a", 1000);
    store.upsert_response(&record).await.expect("upsert should succeed");
    seed_pending(&store, "tenant_a", "resp_del", &["call_abc"]).await;

    let deleted = store
        .delete_response("tenant_a", "resp_del")
        .await
        .expect("delete should succeed");
    assert!(deleted, "the response should be deleted");

    // The approval issued by the deleted response is gone: neither retrievable...
    let fetched = store
        .get_pending_approvals("tenant_a", "resp_del", &["call_abc"])
        .await
        .expect("get should succeed");
    assert!(
        fetched.is_empty(),
        "deleting the response must remove its pending approval rows"
    );

    // ...nor consumable (no server-owned row remains to claim).
    let claim = store
        .consume_approvals("tenant_a", "resp_del", &["call_abc"], 1000)
        .await
        .expect("consume should succeed");
    assert_eq!(claim, Some(0), "a deleted response's approval must not be consumable");
}

#[tokio::test]
async fn persist_response_with_pending_approvals_writes_both() {
    // The atomic write path records the response and every pending approval it
    // issued together, so the resume turn can correlate the mcp_approval_response
    // back to a durable, server-written record.
    let store = make_store().await;
    let record = make_response_record("resp_persist", "tenant_a", 1000);
    let approval = PendingApprovalRecord {
        approval_id: "call_persist".to_owned(),
        ..make_pending("call_persist")
    };

    store
        .persist_response_with_pending_approvals(&record, std::slice::from_ref(&approval))
        .await
        .expect("atomic persist should succeed");

    let fetched_response = store
        .get_response("tenant_a", "resp_persist")
        .await
        .expect("get should succeed");
    assert!(fetched_response.is_some(), "the response must be written");

    let fetched_approvals = store
        .get_pending_approvals("tenant_a", "resp_persist", &["call_persist"])
        .await
        .expect("get should succeed");
    assert_eq!(
        fetched_approvals,
        vec![approval],
        "the pending approval must be written and scoped to the issuing response"
    );
}

#[tokio::test]
async fn persist_response_with_pending_approvals_rolls_back_response_on_approval_failure() {
    // Atomicity means the response and its pending approvals commit together or not
    // at all. If the approval write fails, a separate-writes implementation leaves
    // the response committed — visible to a streaming client that could then issue a
    // DELETE while the (retried) approval insert lands, orphaning a row that holds
    // the tool arguments. A single transaction rolls the response back with the
    // failed approval, so no half-written state is ever observable.
    //
    // Force the second write to fail deterministically by dropping the derived
    // approvals table on a side connection, then assert the response did not persist.
    let dir = tempfile::tempdir().expect("temp dir should be created");
    let store = make_file_store(&dir, None).await;

    let db_path = dir.path().join("concurrent.db");
    let url = format!("sqlite://{}?mode=rwc", db_path.display());
    let side = sqlx::sqlite::SqlitePool::connect(&url)
        .await
        .expect("side connection should open");
    sqlx::query("DROP TABLE test_responses_pending_approvals")
        .execute(&side)
        .await
        .expect("dropping the approvals table should succeed");
    side.close().await;

    let record = make_response_record("resp_rollback", "tenant_a", 1000);
    let approval = make_pending("call_rollback");
    let result = store
        .persist_response_with_pending_approvals(&record, std::slice::from_ref(&approval))
        .await;
    assert!(
        result.is_err(),
        "a failed pending-approval write must surface an error, not be swallowed"
    );

    let fetched = store
        .get_response("tenant_a", "resp_rollback")
        .await
        .expect("get should succeed");
    assert!(
        fetched.is_none(),
        "the response must be rolled back when its pending-approval write fails, \
         leaving no half-written state"
    );
}

// -----------------------------------------------------------------------------
// Input Items
// -----------------------------------------------------------------------------

#[test]
fn input_items_from_array_input() {
    let record = ResponseRecord {
        input: json!([
            {"type": "message", "role": "user", "content": "Hello"},
            {"type": "message", "role": "user", "content": "World"},
            {"type": "message", "role": "user", "content": "!"}
        ]),
        ..make_response_record("resp_1", "tenant_a", 1000)
    };

    let page = list_input_items(
        &record,
        &ListParams {
            limit: 2,
            ..ListParams::default()
        },
        IncludeFields::default(),
    )
    .expect("list should succeed");

    assert_eq!(page.data.len(), 2, "should return 2 items");
    assert!(page.has_more, "should have more items");
    assert_eq!(
        page.next_cursor.as_deref(),
        Some("msg_resp_1_input_1"),
        "cursor should use the synthetic ID assigned to the last item on the page"
    );

    let page2 = list_input_items(
        &record,
        &ListParams {
            cursor: page.next_cursor,
            limit: 2,
            ..ListParams::default()
        },
        IncludeFields::default(),
    )
    .expect("list should succeed");

    assert_eq!(page2.data.len(), 1, "should return remaining 1 item");
    assert!(!page2.has_more, "should have no more items");
}

#[test]
fn input_items_uses_item_id_cursor() {
    let record = ResponseRecord {
        input: json!([
            {"id": "item_1", "type": "message", "role": "user", "content": "Hello"},
            {"id": "item_2", "type": "message", "role": "user", "content": "World"},
            {"id": "item_3", "type": "message", "role": "user", "content": "!"}
        ]),
        ..make_response_record("resp_1", "tenant_a", 1000)
    };

    let page = list_input_items(
        &record,
        &ListParams {
            limit: 2,
            order: Order::Ascending,
            ..ListParams::default()
        },
        IncludeFields::default(),
    )
    .expect("list should succeed");

    assert_eq!(
        page.next_cursor.as_deref(),
        Some("item_2"),
        "cursor should use the last item ID"
    );

    let page2 = list_input_items(
        &record,
        &ListParams {
            cursor: page.next_cursor,
            limit: 2,
            order: Order::Ascending,
        },
        IncludeFields::default(),
    )
    .expect("list should succeed");

    assert_eq!(page2.data.len(), 1, "second page should return remaining item");
    assert_eq!(page2.data[0]["id"], "item_3", "second page should start after item_2");
    assert!(!page2.has_more, "second page should complete pagination");
}

#[test]
fn input_items_from_string_input() {
    let record = ResponseRecord {
        input: json!("Hello, world!"),
        ..make_response_record("resp_1", "tenant_a", 1000)
    };

    let params = ListParams::default();
    let includes = IncludeFields::default();
    let page = list_input_items(&record, &params, includes).expect("list should succeed");

    assert_eq!(page.data.len(), 1, "string input should yield 1 item");
    assert_eq!(
        page.data[0],
        json!({
            "id": "msg_resp_1_input_0",
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Hello, world!"}]
        }),
        "string input should be normalized to a message resource"
    );
}

#[test]
fn input_items_honors_sort_order() {
    let record = ResponseRecord {
        input: json!(["first", "second", "third"]),
        ..make_response_record("resp_1", "tenant_a", 1000)
    };
    let includes = IncludeFields::default();

    let ascending = list_input_items(
        &record,
        &ListParams {
            order: Order::Ascending,
            ..ListParams::default()
        },
        IncludeFields::default(),
    )
    .expect("ascending list should succeed");
    let descending =
        list_input_items(&record, &ListParams::default(), includes).expect("descending list should succeed");

    assert_eq!(
        ascending.data,
        vec![json!("first"), json!("second"), json!("third")],
        "ascending order should preserve input order"
    );
    assert_eq!(
        descending.data,
        vec![json!("third"), json!("second"), json!("first")],
        "descending order should reverse input order"
    );
}

#[test]
fn input_items_limit_zero_clamps_to_one() {
    let record = ResponseRecord {
        input: json!([
            {"type": "message", "role": "user", "content": "Hello"},
            {"type": "message", "role": "user", "content": "World"}
        ]),
        ..make_response_record("resp_1", "tenant_a", 1000)
    };

    let page1 = list_input_items(
        &record,
        &ListParams {
            limit: 0,
            ..ListParams::default()
        },
        IncludeFields::default(),
    )
    .expect("list should succeed");

    assert_eq!(page1.data.len(), 1, "limit 0 should clamp to one item");
    assert!(page1.has_more, "first page should indicate remaining items");
    assert_eq!(
        page1.next_cursor.as_deref(),
        Some("msg_resp_1_input_1"),
        "cursor should use the synthetic ID assigned to the item on the page"
    );

    let page2 = list_input_items(
        &record,
        &ListParams {
            cursor: page1.next_cursor,
            limit: 0,
            ..ListParams::default()
        },
        IncludeFields::default(),
    )
    .expect("list should succeed");

    assert_eq!(page2.data.len(), 1, "second page should return the remaining item");
    assert!(!page2.has_more, "second page should complete pagination");
    assert!(page2.next_cursor.is_none(), "second page should not provide a cursor");
}

#[test]
fn input_items_rejects_overflowing_cursor() {
    let record = ResponseRecord {
        input: json!([{"type": "message", "role": "user", "content": "Hello"}]),
        ..make_response_record("resp_1", "tenant_a", 1000)
    };

    let result = list_input_items(
        &record,
        &ListParams {
            cursor: Some(usize::MAX.to_string()),
            limit: 1,
            ..ListParams::default()
        },
        IncludeFields::default(),
    );

    let Err(err) = result else {
        panic!("overflowing cursor should be rejected");
    };

    assert!(
        err.to_string().contains("overflow"),
        "error should explain cursor overflow: {err}"
    );
}

#[test]
fn input_items_from_empty_array() {
    let record = ResponseRecord {
        input: json!([]),
        ..make_response_record("resp_1", "tenant_a", 1000)
    };

    let params = ListParams::default();
    let includes = IncludeFields::default();
    let page = list_input_items(&record, &params, includes).expect("list should succeed");

    assert!(page.data.is_empty(), "empty array should return no items");
    assert!(!page.has_more, "should have no more items");
    assert!(page.next_cursor.is_none(), "should have no cursor");
}

// -----------------------------------------------------------------------------
// Conversation CRUD
// -----------------------------------------------------------------------------

#[tokio::test]
async fn upsert_and_get_conversation() {
    let store = make_store().await;
    let record = ConversationRecord {
        conversation_id: "conv_1".to_owned(),
        tenant_id: "tenant_a".to_owned(),
        created_at: 1000,
        metadata: json!({}),
        messages: json!([{"role": "user", "content": "Hi"}]),
    };

    store.upsert_conversation(&record).await.expect("upsert should succeed");

    let fetched = ResponseStore::get_conversation(&store, "tenant_a", "conv_1")
        .await
        .expect("get should succeed")
        .expect("record should exist");

    assert_eq!(fetched.conversation_id, "conv_1", "conversation_id should match");
    assert_eq!(
        fetched.messages,
        json!([{"role": "user", "content": "Hi"}]),
        "messages should match"
    );
}

#[tokio::test]
async fn upsert_conversation_overwrites() {
    let store = make_store().await;
    let record = ConversationRecord {
        conversation_id: "conv_1".to_owned(),
        tenant_id: "tenant_a".to_owned(),
        created_at: 1000,
        metadata: json!({}),
        messages: json!([{"role": "user", "content": "v1"}]),
    };
    store.upsert_conversation(&record).await.expect("upsert should succeed");

    let updated = ConversationRecord {
        conversation_id: "conv_1".to_owned(),
        tenant_id: "tenant_a".to_owned(),
        created_at: 2000,
        metadata: json!({"topic": "updated"}),
        messages: json!([{"role": "user", "content": "v2"}]),
    };
    store
        .upsert_conversation(&updated)
        .await
        .expect("second upsert should succeed");

    let fetched = ConversationItemStore::get_conversation(&store, "tenant_a", "conv_1")
        .await
        .expect("get should succeed")
        .expect("record should exist");

    assert_eq!(
        fetched.messages,
        json!([{"role": "user", "content": "v2"}]),
        "messages should be updated"
    );
    assert_eq!(
        fetched.metadata,
        json!({"topic": "updated"}),
        "metadata should be updated"
    );
    assert_eq!(fetched.created_at, 1000, "created_at should preserve creation time");
}

#[tokio::test]
async fn update_conversation_messages_preserves_metadata() {
    let store = make_store().await;
    let record = ConversationRecord {
        conversation_id: "conv_1".to_owned(),
        tenant_id: "tenant_a".to_owned(),
        created_at: 1000,
        metadata: json!({"version": "v1"}),
        messages: json!([{"role": "user", "content": "v1"}]),
    };
    store.upsert_conversation(&record).await.expect("upsert should succeed");

    let updated = store
        .update_conversation_messages("tenant_a", "conv_1", &json!([{"role": "assistant", "content": "v2"}]))
        .await
        .expect("message update should succeed");
    assert!(updated, "conversation should be updated");

    let fetched = ConversationItemStore::get_conversation(&store, "tenant_a", "conv_1")
        .await
        .expect("get should succeed")
        .expect("record should exist");

    assert_eq!(
        fetched.metadata,
        json!({"version": "v1"}),
        "metadata should be preserved"
    );
    assert_eq!(
        fetched.messages,
        json!([{"role": "assistant", "content": "v2"}]),
        "messages should be updated"
    );
}

#[tokio::test]
async fn update_conversation_metadata_preserves_messages() {
    let store = make_store().await;
    let record = ConversationRecord {
        conversation_id: "conv_1".to_owned(),
        tenant_id: "tenant_a".to_owned(),
        created_at: 1000,
        metadata: json!({"version": "v1"}),
        messages: json!([{"role": "user", "content": "keep me"}]),
    };
    store.upsert_conversation(&record).await.expect("upsert should succeed");

    let updated = store
        .update_conversation_metadata("tenant_a", "conv_1", &json!({"version": "v2"}))
        .await
        .expect("metadata update should succeed");
    assert!(updated, "conversation should be updated");

    let fetched = ConversationItemStore::get_conversation(&store, "tenant_a", "conv_1")
        .await
        .expect("get should succeed")
        .expect("record should exist");

    assert_eq!(fetched.metadata, json!({"version": "v2"}), "metadata should be updated");
    assert_eq!(
        fetched.messages,
        json!([{"role": "user", "content": "keep me"}]),
        "messages must be untouched by a metadata-only update"
    );
    assert_eq!(fetched.created_at, 1000, "created_at should be preserved");
}

#[tokio::test]
async fn update_conversation_metadata_nonexistent_returns_false() {
    let store = make_store().await;

    let updated = store
        .update_conversation_metadata("tenant_a", "nonexistent", &json!({"topic": "x"}))
        .await
        .expect("update should succeed");

    assert!(!updated, "updating nonexistent conversation should return false");
}

#[tokio::test]
async fn update_conversation_metadata_tenant_isolation() {
    let store = make_store().await;
    let record = ConversationRecord {
        conversation_id: "conv_1".to_owned(),
        tenant_id: "tenant_a".to_owned(),
        created_at: 1000,
        metadata: json!({"owner": "a"}),
        messages: json!([{"role": "user", "content": "original"}]),
    };
    store.upsert_conversation(&record).await.expect("upsert should succeed");

    let updated = store
        .update_conversation_metadata("tenant_b", "conv_1", &json!({"owner": "hijack"}))
        .await
        .expect("cross-tenant update should succeed");
    assert!(!updated, "tenant_b should not be able to update tenant_a metadata");

    let fetched = ConversationItemStore::get_conversation(&store, "tenant_a", "conv_1")
        .await
        .expect("get should succeed")
        .expect("record should exist");
    assert_eq!(
        fetched.metadata,
        json!({"owner": "a"}),
        "metadata should not have changed across tenants"
    );
}

#[tokio::test]
async fn compare_and_swap_conversation_messages_rejects_stale_snapshot() {
    let store = make_store().await;
    let initial = json!([{"role":"user","content":"initial"}]);
    let record = ConversationRecord {
        conversation_id: "conv_cas".to_owned(),
        tenant_id: "tenant_a".to_owned(),
        created_at: 1000,
        metadata: json!({}),
        messages: initial.clone(),
    };
    store.upsert_conversation(&record).await.expect("upsert should succeed");

    let first = json!([{"role":"assistant","content":"first"}]);
    assert!(
        store
            .compare_and_swap_conversation_messages("tenant_a", "conv_cas", &initial, &first)
            .await
            .expect("first compare-and-swap should succeed")
    );
    assert!(
        !store
            .compare_and_swap_conversation_messages("tenant_a", "conv_cas", &initial, &json!([]))
            .await
            .expect("stale compare-and-swap should be conflict-free")
    );
    let fetched = ConversationItemStore::get_conversation(&store, "tenant_a", "conv_cas")
        .await
        .expect("get should succeed")
        .expect("conversation should exist");
    assert_eq!(fetched.messages, first);
}

#[tokio::test]
async fn get_missing_conversation_returns_none() {
    let store = make_store().await;

    let result = ConversationItemStore::get_conversation(&store, "tenant_a", "nonexistent")
        .await
        .expect("get should succeed");

    assert!(result.is_none(), "missing conversation should return None");
}

#[tokio::test]
async fn conversation_tenant_isolation() {
    let store = make_store().await;
    let record = ConversationRecord {
        conversation_id: "conv_1".to_owned(),
        tenant_id: "tenant_a".to_owned(),
        created_at: 1000,
        metadata: json!({}),
        messages: json!([]),
    };
    store.upsert_conversation(&record).await.expect("upsert should succeed");

    let result = ConversationItemStore::get_conversation(&store, "tenant_b", "conv_1")
        .await
        .expect("get should succeed");

    assert!(result.is_none(), "tenant_b should not see tenant_a conversation");
}

#[tokio::test]
async fn delete_existing_conversation() {
    let store = make_store().await;
    let record = ConversationRecord {
        conversation_id: "conv_1".to_owned(),
        tenant_id: "tenant_a".to_owned(),
        created_at: 1000,
        metadata: json!({}),
        messages: json!([]),
    };
    store.upsert_conversation(&record).await.expect("upsert should succeed");

    let deleted = store
        .delete_conversation("tenant_a", "conv_1")
        .await
        .expect("delete should succeed");

    assert!(deleted, "delete should return true for existing conversation");

    let fetched = ConversationItemStore::get_conversation(&store, "tenant_a", "conv_1")
        .await
        .expect("get should succeed");

    assert!(fetched.is_none(), "deleted conversation should not be retrievable");
}

#[tokio::test]
async fn delete_missing_conversation_returns_false() {
    let store = make_store().await;

    let deleted = store
        .delete_conversation("tenant_a", "nonexistent")
        .await
        .expect("delete should succeed");

    assert!(!deleted, "delete should return false for missing conversation");
}

#[tokio::test]
async fn delete_conversation_tenant_isolation() {
    let store = make_store().await;
    let record = ConversationRecord {
        conversation_id: "conv_1".to_owned(),
        tenant_id: "tenant_a".to_owned(),
        created_at: 1000,
        metadata: json!({}),
        messages: json!([]),
    };
    store.upsert_conversation(&record).await.expect("upsert should succeed");

    let deleted = store
        .delete_conversation("tenant_b", "conv_1")
        .await
        .expect("delete should succeed");

    assert!(!deleted, "tenant_b should not be able to delete tenant_a conversation");

    let still_exists = ConversationItemStore::get_conversation(&store, "tenant_a", "conv_1")
        .await
        .expect("get should succeed");

    assert!(
        still_exists.is_some(),
        "conversation should still exist after cross-tenant delete attempt"
    );
}

// -----------------------------------------------------------------------------
// Conversation Item CRUD (SQLite)
// -----------------------------------------------------------------------------

#[tokio::test]
async fn conversation_items_paginate_ascending_and_descending() {
    let store = make_store_with_items().await;
    let items = [
        make_conversation_item("item_1", "tenant_a", "conv_1", 1),
        make_conversation_item("item_2", "tenant_a", "conv_1", 2),
        make_conversation_item("item_3", "tenant_a", "conv_1", 3),
        make_conversation_item("item_4", "tenant_a", "conv_1", 4),
    ];
    store
        .create_conversation_items(&items)
        .await
        .expect("item insert should succeed");

    let asc = store
        .list_conversation_items("tenant_a", "conv_1", None, 2, true)
        .await
        .expect("ascending list should succeed");
    assert_item_ids(&asc, &["item_1", "item_2"]);

    let asc_page2 = store
        .list_conversation_items("tenant_a", "conv_1", Some("item_2"), 2, true)
        .await
        .expect("ascending page 2 should succeed");
    assert_item_ids(&asc_page2, &["item_3", "item_4"]);

    let desc = store
        .list_conversation_items("tenant_a", "conv_1", None, 2, false)
        .await
        .expect("descending list should succeed");
    assert_item_ids(&desc, &["item_4", "item_3"]);

    let desc_page2 = store
        .list_conversation_items("tenant_a", "conv_1", Some("item_3"), 2, false)
        .await
        .expect("descending page 2 should succeed");
    assert_item_ids(&desc_page2, &["item_2", "item_1"]);
}

#[tokio::test]
async fn duplicate_position_rejected_by_unique_constraint() {
    let store = make_store_with_items().await;
    let first = [make_conversation_item("item_a", "tenant_a", "conv_1", 1)];
    store
        .create_conversation_items(&first)
        .await
        .expect("first insert should succeed");

    let duplicate = [make_conversation_item("item_b", "tenant_a", "conv_1", 1)];
    store
        .create_conversation_items(&duplicate)
        .await
        .expect_err("duplicate position should fail");
}

#[tokio::test]
async fn conversation_item_single_ops_scope_to_conversation() {
    let store = make_store_with_items().await;
    let item_conv1 = make_conversation_item("item_1", "tenant_a", "conv_1", 1);
    let item_conv2 = make_conversation_item("item_2", "tenant_a", "conv_2", 1);
    store
        .create_conversation_items(&[item_conv1, item_conv2])
        .await
        .expect("item insert should succeed");

    let get_wrong_conv = store
        .get_conversation_item("tenant_a", "conv_2", "item_1")
        .await
        .expect("get should succeed");
    assert!(get_wrong_conv.is_none(), "item_1 should not be visible in conv_2");

    let delete_wrong_conv = store
        .delete_conversation_item("tenant_a", "conv_2", "item_1")
        .await
        .expect("delete should succeed");
    assert!(!delete_wrong_conv, "deleting item_1 from conv_2 should return false");

    let still_exists = store
        .get_conversation_item("tenant_a", "conv_1", "item_1")
        .await
        .expect("get should succeed");
    assert!(still_exists.is_some(), "item_1 should still exist in conv_1");
}

#[tokio::test]
async fn max_item_position_returns_zero_when_empty() {
    let store = make_store_with_items().await;
    let max = store
        .max_item_position("tenant_a", "conv_1")
        .await
        .expect("max_item_position should succeed");
    assert_eq!(max, 0, "empty conversation should have max position 0");
}

#[tokio::test]
async fn max_item_position_returns_highest() {
    let store = make_store_with_items().await;
    let items = [
        make_conversation_item("item_1", "tenant_a", "conv_1", 5),
        make_conversation_item("item_2", "tenant_a", "conv_1", 10),
        make_conversation_item("item_3", "tenant_a", "conv_1", 3),
    ];
    store
        .create_conversation_items(&items)
        .await
        .expect("item insert should succeed");

    let max = store
        .max_item_position("tenant_a", "conv_1")
        .await
        .expect("max_item_position should succeed");
    assert_eq!(max, 10, "max position should be 10");
}

#[tokio::test]
async fn conversation_item_tenant_isolation() {
    let store = make_store_with_items().await;
    let item = make_conversation_item("item_1", "tenant_a", "conv_1", 1);
    store
        .create_conversation_items(&[item])
        .await
        .expect("item insert should succeed");

    let cross_tenant = store
        .get_conversation_item("tenant_b", "conv_1", "item_1")
        .await
        .expect("cross-tenant get should succeed");
    assert!(cross_tenant.is_none(), "tenant_b should not see tenant_a items");

    let cross_tenant_list = store
        .list_conversation_items("tenant_b", "conv_1", None, 100, true)
        .await
        .expect("cross-tenant list should succeed");
    assert!(cross_tenant_list.is_empty(), "tenant_b should see no items");
}

#[tokio::test]
async fn conversation_item_insert_rejects_existing() {
    let store = make_store_with_items().await;
    let original = make_conversation_item("item_1", "tenant_a", "conv_1", 1);
    let updated = ConversationItemRecord {
        item_data: json!({"type": "message", "role": "assistant", "content": "updated"}),
        created_at: 2000,
        position: 2,
        ..make_conversation_item("item_1", "tenant_a", "conv_1", 1)
    };

    store
        .create_conversation_items(&[original])
        .await
        .expect("initial item insert should succeed");
    store
        .create_conversation_items(&[updated])
        .await
        .expect_err("duplicate item insert should fail");

    let fetched = store
        .get_conversation_item("tenant_a", "conv_1", "item_1")
        .await
        .expect("get should succeed")
        .expect("item should exist after duplicate insert");

    assert_eq!(fetched.position, 1, "duplicate insert should preserve position");
    assert_eq!(fetched.created_at, 1000, "duplicate insert should preserve created_at");
    assert_eq!(
        fetched.item_data,
        json!({"type": "message", "role": "user", "content": "test"}),
        "duplicate insert should preserve item data"
    );
}

#[tokio::test]
async fn conversation_item_upsert_allows_same_item_id_in_different_conversations() {
    let store = make_store_with_items().await;
    let item_conv1 = ConversationItemRecord {
        item_data: json!({"conversation": "conv_1"}),
        ..make_conversation_item("item_shared", "tenant_a", "conv_1", 1)
    };
    let item_conv2 = ConversationItemRecord {
        item_data: json!({"conversation": "conv_2"}),
        ..make_conversation_item("item_shared", "tenant_a", "conv_2", 1)
    };

    store
        .create_conversation_items(&[item_conv1])
        .await
        .expect("initial item insert should succeed");
    store
        .create_conversation_items(&[item_conv2])
        .await
        .expect("same item_id in another conversation should insert");

    let conv1_item = store
        .get_conversation_item("tenant_a", "conv_1", "item_shared")
        .await
        .expect("conv_1 get should succeed")
        .expect("conv_1 item should still exist");
    let conv2_item = store
        .get_conversation_item("tenant_a", "conv_2", "item_shared")
        .await
        .expect("conv_2 get should succeed")
        .expect("conv_2 item should exist");

    assert_eq!(conv1_item.conversation_id, "conv_1", "conv_1 row should remain scoped");
    assert_eq!(conv2_item.conversation_id, "conv_2", "conv_2 row should be inserted");
    assert_eq!(
        conv1_item.item_data,
        json!({"conversation": "conv_1"}),
        "conv_1 item data should not be overwritten"
    );
    assert_eq!(
        conv2_item.item_data,
        json!({"conversation": "conv_2"}),
        "conv_2 item data should be stored separately"
    );

    let conv1_items = store
        .list_conversation_items("tenant_a", "conv_1", None, 100, true)
        .await
        .expect("conv_1 list should succeed");
    let conv2_items = store
        .list_conversation_items("tenant_a", "conv_2", None, 100, true)
        .await
        .expect("conv_2 list should succeed");
    assert_item_ids(&conv1_items, &["item_shared"]);
    assert_item_ids(&conv2_items, &["item_shared"]);
}

#[tokio::test]
async fn get_conversation_item_returns_all_fields() {
    let store = make_store_with_items().await;
    let item = ConversationItemRecord {
        item_id: "item_99".to_owned(),
        tenant_id: "tenant_a".to_owned(),
        conversation_id: "conv_1".to_owned(),
        item_data: json!({"type": "function_call", "name": "search"}),
        created_at: 5000,
        position: 42,
    };
    store
        .create_conversation_items(&[item])
        .await
        .expect("item insert should succeed");

    let fetched = store
        .get_conversation_item("tenant_a", "conv_1", "item_99")
        .await
        .expect("get should succeed")
        .expect("item should exist");

    assert_eq!(fetched.item_id, "item_99", "item_id should match");
    assert_eq!(fetched.tenant_id, "tenant_a", "tenant_id should match");
    assert_eq!(fetched.conversation_id, "conv_1", "conversation_id should match");
    assert_eq!(
        fetched.item_data,
        json!({"type": "function_call", "name": "search"}),
        "item_data should round-trip"
    );
    assert_eq!(fetched.created_at, 5000, "created_at should match");
    assert_eq!(fetched.position, 42, "position should match");
}

#[tokio::test]
async fn list_conversation_items_nonexistent_cursor_returns_empty() {
    let store = make_store_with_items().await;
    let item = make_conversation_item("item_1", "tenant_a", "conv_1", 1);
    store
        .create_conversation_items(&[item])
        .await
        .expect("item insert should succeed");

    let result = store
        .list_conversation_items("tenant_a", "conv_1", Some("nonexistent"), 10, true)
        .await
        .expect("list with nonexistent cursor should succeed");

    assert!(result.is_empty(), "nonexistent cursor item should return empty list");
}

#[tokio::test]
async fn delete_conversation_preserves_items() {
    let store = make_store_with_items().await;
    let conv = ConversationRecord {
        conversation_id: "conv_1".to_owned(),
        tenant_id: "tenant_a".to_owned(),
        created_at: 1000,
        metadata: json!({}),
        messages: json!([]),
    };
    store
        .upsert_conversation(&conv)
        .await
        .expect("conversation upsert should succeed");

    let items = [
        make_conversation_item("item_1", "tenant_a", "conv_1", 1),
        make_conversation_item("item_2", "tenant_a", "conv_1", 2),
    ];
    store
        .create_conversation_items(&items)
        .await
        .expect("item insert should succeed");

    let deleted = store
        .delete_conversation("tenant_a", "conv_1")
        .await
        .expect("delete_conversation should succeed");
    assert!(deleted, "conversation should have been deleted");

    let remaining = store
        .list_conversation_items("tenant_a", "conv_1", None, 100, true)
        .await
        .expect("list should succeed");
    assert_item_ids(&remaining, &["item_1", "item_2"]);
}

#[tokio::test]
async fn get_existing_conversation_item_ids_returns_matching() {
    let store = make_store_with_items().await;
    let items = [
        make_conversation_item("item_1", "tenant_a", "conv_1", 1),
        make_conversation_item("item_2", "tenant_a", "conv_1", 2),
        make_conversation_item("item_3", "tenant_a", "conv_1", 3),
    ];
    store
        .create_conversation_items(&items)
        .await
        .expect("item insert should succeed");

    let existing = store
        .get_existing_conversation_item_ids("tenant_a", "conv_1", &["item_1", "item_3", "item_99"])
        .await
        .expect("get_existing should succeed");

    assert_eq!(existing.len(), 2, "should find 2 of 3 queried IDs");
    assert!(existing.contains(&"item_1".to_owned()), "item_1 should be found");
    assert!(existing.contains(&"item_3".to_owned()), "item_3 should be found");
}

#[tokio::test]
async fn get_existing_conversation_item_ids_empty_input() {
    let store = make_store_with_items().await;
    let item = make_conversation_item("item_1", "tenant_a", "conv_1", 1);
    store
        .create_conversation_items(&[item])
        .await
        .expect("item insert should succeed");

    let existing = store
        .get_existing_conversation_item_ids("tenant_a", "conv_1", &[])
        .await
        .expect("get_existing with empty input should succeed");

    assert!(existing.is_empty(), "empty input should return empty result");
}

#[tokio::test]
async fn get_existing_conversation_item_ids_tenant_isolation() {
    let store = make_store_with_items().await;
    let item = make_conversation_item("item_1", "tenant_a", "conv_1", 1);
    store
        .create_conversation_items(&[item])
        .await
        .expect("item insert should succeed");

    let existing = store
        .get_existing_conversation_item_ids("tenant_b", "conv_1", &["item_1"])
        .await
        .expect("get_existing should succeed");

    assert!(existing.is_empty(), "tenant_b should not see tenant_a items");
}

#[tokio::test]
async fn delete_conversation_item_returns_true() {
    let store = make_store_with_items().await;
    let items = [
        make_conversation_item("item_1", "tenant_a", "conv_1", 1),
        make_conversation_item("item_2", "tenant_a", "conv_1", 2),
    ];
    store
        .create_conversation_items(&items)
        .await
        .expect("item insert should succeed");

    let deleted = store
        .delete_conversation_item("tenant_a", "conv_1", "item_1")
        .await
        .expect("delete should succeed");
    assert!(deleted, "delete should return true for existing item");

    let fetched = store
        .get_conversation_item("tenant_a", "conv_1", "item_1")
        .await
        .expect("get should succeed");
    assert!(fetched.is_none(), "deleted item should not be retrievable");

    let remaining = store
        .list_conversation_items("tenant_a", "conv_1", None, 100, true)
        .await
        .expect("list should succeed");
    assert_item_ids(&remaining, &["item_2"]);
}

#[tokio::test]
async fn delete_nonexistent_conversation_item_returns_false() {
    let store = make_store_with_items().await;

    let deleted = store
        .delete_conversation_item("tenant_a", "conv_1", "nonexistent")
        .await
        .expect("delete should succeed");

    assert!(!deleted, "delete should return false for nonexistent item");
}

#[tokio::test]
async fn conversation_item_position_returns_existing() {
    let store = make_store_with_items().await;
    let items = [
        make_conversation_item("item_1", "tenant_a", "conv_1", 5),
        make_conversation_item("item_2", "tenant_a", "conv_1", 10),
    ];
    store
        .create_conversation_items(&items)
        .await
        .expect("item insert should succeed");

    let position = store
        .conversation_item_position("tenant_a", "conv_1", "item_2")
        .await
        .expect("position lookup should succeed");

    assert_eq!(position, Some(10), "position should be 10");
}

#[tokio::test]
async fn conversation_item_position_returns_none_for_missing() {
    let store = make_store_with_items().await;

    let position = store
        .conversation_item_position("tenant_a", "conv_1", "nonexistent")
        .await
        .expect("position lookup should succeed");

    assert!(position.is_none(), "missing item should return None");
}

#[tokio::test]
async fn update_conversation_messages_nonexistent_returns_false() {
    let store = make_store().await;

    let updated = store
        .update_conversation_messages("tenant_a", "nonexistent", &json!({"new": "messages"}))
        .await
        .expect("update should succeed");

    assert!(!updated, "updating nonexistent conversation should return false");
}

#[tokio::test]
async fn update_conversation_messages_tenant_isolation() {
    let store = make_store().await;
    let record = ConversationRecord {
        conversation_id: "conv_1".to_owned(),
        tenant_id: "tenant_a".to_owned(),
        created_at: 1000,
        metadata: json!({}),
        messages: json!([{"role": "user", "content": "original"}]),
    };
    store.upsert_conversation(&record).await.expect("upsert should succeed");

    let updated = store
        .update_conversation_messages("tenant_b", "conv_1", &json!([{"role": "user", "content": "hijack"}]))
        .await
        .expect("cross-tenant update should succeed");
    assert!(!updated, "tenant_b should not be able to update tenant_a messages");

    let fetched = ConversationItemStore::get_conversation(&store, "tenant_a", "conv_1")
        .await
        .expect("get should succeed")
        .expect("record should exist");
    assert_eq!(
        fetched.messages,
        json!([{"role": "user", "content": "original"}]),
        "messages should not have changed"
    );
}

#[tokio::test]
async fn delete_conversation_item_tenant_isolation() {
    let store = make_store_with_items().await;
    let item = make_conversation_item("item_1", "tenant_a", "conv_1", 1);
    store
        .create_conversation_items(&[item])
        .await
        .expect("item insert should succeed");

    let deleted = store
        .delete_conversation_item("tenant_b", "conv_1", "item_1")
        .await
        .expect("cross-tenant delete should succeed");
    assert!(!deleted, "tenant_b should not be able to delete tenant_a items");

    let still_exists = store
        .get_conversation_item("tenant_a", "conv_1", "item_1")
        .await
        .expect("get should succeed");
    assert!(
        still_exists.is_some(),
        "item should still exist after cross-tenant delete attempt"
    );
}

#[tokio::test]
async fn get_existing_conversation_item_ids_conversation_isolation() {
    let store = make_store_with_items().await;
    let items = [
        make_conversation_item("item_1", "tenant_a", "conv_1", 1),
        make_conversation_item("item_2", "tenant_a", "conv_2", 1),
    ];
    store
        .create_conversation_items(&items)
        .await
        .expect("item insert should succeed");

    let existing = store
        .get_existing_conversation_item_ids("tenant_a", "conv_1", &["item_1", "item_2"])
        .await
        .expect("get_existing should succeed");

    assert_eq!(existing.len(), 1, "should find only the item in conv_1");
    assert!(
        existing.contains(&"item_1".to_owned()),
        "item_1 should be found in conv_1"
    );
}

#[tokio::test]
async fn conversation_item_position_tenant_isolation() {
    let store = make_store_with_items().await;
    let item = make_conversation_item("item_1", "tenant_a", "conv_1", 5);
    store
        .create_conversation_items(&[item])
        .await
        .expect("item insert should succeed");

    let position = store
        .conversation_item_position("tenant_b", "conv_1", "item_1")
        .await
        .expect("cross-tenant position lookup should succeed");

    assert!(position.is_none(), "tenant_b should not see tenant_a item position");
}

#[tokio::test]
async fn conversation_item_position_conversation_isolation() {
    let store = make_store_with_items().await;
    let item = make_conversation_item("item_1", "tenant_a", "conv_1", 5);
    store
        .create_conversation_items(&[item])
        .await
        .expect("item insert should succeed");

    let position = store
        .conversation_item_position("tenant_a", "conv_2", "item_1")
        .await
        .expect("cross-conversation position lookup should succeed");

    assert!(position.is_none(), "item_1 position should not be visible in conv_2");
}

#[tokio::test]
async fn conversation_item_methods_fail_without_items_table() {
    let store = make_store().await;
    let item = make_conversation_item("item_1", "tenant_a", "conv_1", 1);

    let err = store.create_conversation_items(&[item]).await.unwrap_err();
    assert!(
        matches!(err, StoreError::Unavailable(_)),
        "create should return Unavailable"
    );

    let err = store
        .list_conversation_items("tenant_a", "conv_1", None, 10, true)
        .await
        .unwrap_err();
    assert!(
        matches!(err, StoreError::Unavailable(_)),
        "list should return Unavailable"
    );

    let err = store
        .get_conversation_item("tenant_a", "conv_1", "item_1")
        .await
        .unwrap_err();
    assert!(
        matches!(err, StoreError::Unavailable(_)),
        "get should return Unavailable"
    );

    let err = store
        .delete_conversation_item("tenant_a", "conv_1", "item_1")
        .await
        .unwrap_err();
    assert!(
        matches!(err, StoreError::Unavailable(_)),
        "delete_item should return Unavailable"
    );

    let err = store
        .conversation_item_position("tenant_a", "conv_1", "item_1")
        .await
        .unwrap_err();
    assert!(
        matches!(err, StoreError::Unavailable(_)),
        "position should return Unavailable"
    );

    let err = store.max_item_position("tenant_a", "conv_1").await.unwrap_err();
    assert!(
        matches!(err, StoreError::Unavailable(_)),
        "max_position should return Unavailable"
    );

    let sync_item = make_conversation_item("item_1", "tenant_a", "conv_1", 1);
    let err = store
        .create_items_and_sync_messages("tenant_a", "conv_1", &[sync_item])
        .await
        .unwrap_err();
    assert!(
        matches!(err, StoreError::Unavailable(_)),
        "create_items_and_sync_messages should return Unavailable"
    );

    let err = store
        .delete_item_and_sync_messages("tenant_a", "conv_1", "item_1")
        .await
        .unwrap_err();
    assert!(
        matches!(err, StoreError::Unavailable(_)),
        "delete_item_and_sync_messages should return Unavailable"
    );
}

// -----------------------------------------------------------------------------
// create_items_and_sync_messages (SQLite)
// -----------------------------------------------------------------------------

#[tokio::test]
async fn create_items_and_sync_messages_assigns_positions() {
    let store = make_store_with_items().await;
    let conv = ConversationRecord {
        conversation_id: "conv_1".to_owned(),
        tenant_id: "tenant_a".to_owned(),
        created_at: 1000,
        metadata: json!({}),
        messages: json!([]),
    };
    store.upsert_conversation(&conv).await.expect("upsert should succeed");

    let items = [
        make_conversation_item("item_a", "tenant_a", "conv_1", 0),
        make_conversation_item("item_b", "tenant_a", "conv_1", 0),
    ];
    store
        .create_items_and_sync_messages("tenant_a", "conv_1", &items)
        .await
        .expect("create_items_and_sync should succeed");

    let fetched = store
        .list_conversation_items("tenant_a", "conv_1", None, 100, true)
        .await
        .expect("list should succeed");
    assert_item_ids(&fetched, &["item_a", "item_b"]);
    assert_eq!(fetched[0].position, 1, "first item should get position 1");
    assert_eq!(fetched[1].position, 2, "second item should get position 2");

    let conv_record = ConversationItemStore::get_conversation(&store, "tenant_a", "conv_1")
        .await
        .expect("get should succeed")
        .expect("conversation should exist");
    let messages = conv_record.messages.as_array().expect("messages should be an array");
    assert_eq!(messages.len(), 2, "messages cache should have 2 items");
}

#[tokio::test]
async fn create_items_and_sync_messages_continues_from_max_position() {
    let store = make_store_with_items().await;
    let conv = ConversationRecord {
        conversation_id: "conv_1".to_owned(),
        tenant_id: "tenant_a".to_owned(),
        created_at: 1000,
        metadata: json!({}),
        messages: json!([]),
    };
    store.upsert_conversation(&conv).await.expect("upsert should succeed");

    let first_batch = [make_conversation_item("item_a", "tenant_a", "conv_1", 0)];
    store
        .create_items_and_sync_messages("tenant_a", "conv_1", &first_batch)
        .await
        .expect("first batch should succeed");

    let second_batch = [
        make_conversation_item("item_b", "tenant_a", "conv_1", 0),
        make_conversation_item("item_c", "tenant_a", "conv_1", 0),
    ];
    store
        .create_items_and_sync_messages("tenant_a", "conv_1", &second_batch)
        .await
        .expect("second batch should succeed");

    let fetched = store
        .list_conversation_items("tenant_a", "conv_1", None, 100, true)
        .await
        .expect("list should succeed");
    assert_item_ids(&fetched, &["item_a", "item_b", "item_c"]);
    assert_eq!(fetched[0].position, 1);
    assert_eq!(fetched[1].position, 2);
    assert_eq!(fetched[2].position, 3);

    let conv_record = ConversationItemStore::get_conversation(&store, "tenant_a", "conv_1")
        .await
        .expect("get should succeed")
        .expect("conversation should exist");
    let messages = conv_record.messages.as_array().expect("messages should be an array");
    assert_eq!(messages.len(), 3, "messages cache should include all 3 items");
}

#[tokio::test]
async fn create_items_and_sync_messages_empty_batch_is_noop() {
    let store = make_store_with_items().await;
    let empty: [ConversationItemRecord; 0] = [];
    store
        .create_items_and_sync_messages("tenant_a", "conv_1", &empty)
        .await
        .expect("empty batch should succeed");
}

#[tokio::test]
async fn create_items_and_sync_messages_missing_conversation_errors() {
    let store = make_store_with_items().await;
    // No conversation row exists for "conv_gone" — mirrors a conversation
    // deleted between the handler's existence check and this transaction.
    let items = [make_conversation_item("item_a", "tenant_a", "conv_gone", 0)];
    let err = store
        .create_items_and_sync_messages("tenant_a", "conv_gone", &items)
        .await
        .expect_err("create against a missing conversation should error");
    assert!(
        matches!(&err, StoreError::Database(msg) if msg.contains("conversation disappeared during message sync")),
        "unexpected error: {err:?}"
    );

    // The transaction must roll back — no orphaned items may persist.
    let fetched = store
        .list_conversation_items("tenant_a", "conv_gone", None, 100, true)
        .await
        .expect("list should succeed");
    assert!(fetched.is_empty(), "items must not persist when the message sync fails");
}

// -----------------------------------------------------------------------------
// delete_item_and_sync_messages (SQLite)
// -----------------------------------------------------------------------------

#[tokio::test]
async fn delete_item_and_sync_messages_updates_cache() {
    let store = make_store_with_items().await;
    let conv = ConversationRecord {
        conversation_id: "conv_1".to_owned(),
        tenant_id: "tenant_a".to_owned(),
        created_at: 1000,
        metadata: json!({}),
        messages: json!([]),
    };
    store.upsert_conversation(&conv).await.expect("upsert should succeed");

    let items = [
        make_conversation_item("item_a", "tenant_a", "conv_1", 0),
        make_conversation_item("item_b", "tenant_a", "conv_1", 0),
    ];
    store
        .create_items_and_sync_messages("tenant_a", "conv_1", &items)
        .await
        .expect("create should succeed");

    let deleted = store
        .delete_item_and_sync_messages("tenant_a", "conv_1", "item_a")
        .await
        .expect("delete should succeed");
    assert!(deleted, "item_a should have been deleted");

    let remaining = store
        .list_conversation_items("tenant_a", "conv_1", None, 100, true)
        .await
        .expect("list should succeed");
    assert_item_ids(&remaining, &["item_b"]);

    let conv_record = ConversationItemStore::get_conversation(&store, "tenant_a", "conv_1")
        .await
        .expect("get should succeed")
        .expect("conversation should exist");
    let messages = conv_record.messages.as_array().expect("messages should be an array");
    assert_eq!(messages.len(), 1, "messages cache should reflect deletion");
}

#[tokio::test]
async fn delete_item_and_sync_messages_nonexistent_returns_false() {
    let store = make_store_with_items().await;
    let conv = ConversationRecord {
        conversation_id: "conv_1".to_owned(),
        tenant_id: "tenant_a".to_owned(),
        created_at: 1000,
        metadata: json!({}),
        messages: json!([]),
    };
    store.upsert_conversation(&conv).await.expect("upsert should succeed");

    let deleted = store
        .delete_item_and_sync_messages("tenant_a", "conv_1", "nonexistent")
        .await
        .expect("delete should succeed");
    assert!(!deleted, "nonexistent item should return false");
}

#[tokio::test]
async fn delete_item_and_sync_messages_missing_conversation_errors() {
    let store = make_store_with_items().await;
    let conv = ConversationRecord {
        conversation_id: "conv_1".to_owned(),
        tenant_id: "tenant_a".to_owned(),
        created_at: 1000,
        metadata: json!({}),
        messages: json!([]),
    };
    store.upsert_conversation(&conv).await.expect("upsert should succeed");

    let items = [make_conversation_item("item_a", "tenant_a", "conv_1", 0)];
    store
        .create_items_and_sync_messages("tenant_a", "conv_1", &items)
        .await
        .expect("create should succeed");

    // Delete the conversation row; items intentionally survive (no FK).
    ConversationItemStore::delete_conversation(&store, "tenant_a", "conv_1")
        .await
        .expect("delete conversation should succeed");

    let err = store
        .delete_item_and_sync_messages("tenant_a", "conv_1", "item_a")
        .await
        .expect_err("delete-item sync against a missing conversation should error");
    assert!(
        matches!(&err, StoreError::Database(msg) if msg.contains("conversation disappeared during message sync")),
        "unexpected error: {err:?}"
    );

    // The transaction must roll back — the item deletion must not persist.
    let remaining = store
        .list_conversation_items("tenant_a", "conv_1", None, 100, true)
        .await
        .expect("list should succeed");
    assert_item_ids(&remaining, &["item_a"]);
}

// -----------------------------------------------------------------------------
// StoreError Display & Error trait
// -----------------------------------------------------------------------------

#[test]
fn store_error_invalid_input_display() {
    let err = StoreError::InvalidInput("bad cursor".into());
    let msg = format!("{err}");
    assert!(
        msg.contains("bad cursor"),
        "InvalidInput Display should include the message: {msg}"
    );
}

#[test]
fn store_error_implements_std_error() {
    use std::error::Error as _;
    let err = StoreError::Database("connection lost".into());
    assert!(
        err.source().is_none(),
        "StoreError should implement std::error::Error with no source"
    );
}

// -----------------------------------------------------------------------------
// File-Backed Store
// -----------------------------------------------------------------------------

#[tokio::test]
async fn file_backed_store_crud() {
    let dir = tempfile::tempdir().expect("tempdir should succeed");
    let db_path = dir.path().join("test.db");
    let url = format!("sqlite://{}?mode=rwc", db_path.display());

    let store = SqliteResponseStore::new(&url, "file_responses", "file_conversations", None, None, None)
        .await
        .expect("file-backed store creation should succeed");

    let record = make_response_record("resp_1", "tenant_a", 1000);
    store.upsert_response(&record).await.expect("upsert should succeed");

    let fetched = store
        .get_response("tenant_a", "resp_1")
        .await
        .expect("get should succeed")
        .expect("record should exist");

    assert_eq!(
        fetched.id, "resp_1",
        "file-backed store should persist and retrieve records"
    );
}

// -----------------------------------------------------------------------------
// Schema Migration Idempotency
// -----------------------------------------------------------------------------

#[tokio::test]
async fn schema_migration_is_idempotent() {
    let dir = tempfile::tempdir().expect("tempdir should succeed");
    let db_path = dir.path().join("idempotent.db");
    let url = format!("sqlite://{}?mode=rwc", db_path.display());

    let store = SqliteResponseStore::new(
        &url,
        "idem_responses",
        "idem_conversations",
        Some("idem_items"),
        None,
        None,
    )
    .await
    .expect("first init should succeed");

    let record = make_response_record("resp_1", "tenant_a", 1000);
    store.upsert_response(&record).await.expect("upsert should succeed");

    drop(store);

    let store2 = SqliteResponseStore::new(
        &url,
        "idem_responses",
        "idem_conversations",
        Some("idem_items"),
        None,
        None,
    )
    .await
    .expect("second init with same tables should succeed");

    let fetched = store2
        .get_response("tenant_a", "resp_1")
        .await
        .expect("get should succeed")
        .expect("record should survive re-init");

    assert_eq!(
        fetched.id, "resp_1",
        "data should persist across schema re-initialization"
    );
}

// -----------------------------------------------------------------------------
// Schema Validation (missing columns)
// -----------------------------------------------------------------------------

#[tokio::test]
async fn sqlite_rejects_table_with_missing_columns() {
    let dir = tempfile::tempdir().expect("tempdir should succeed");
    let db_path = dir.path().join("bad_schema.db");
    let url = format!("sqlite://{}?mode=rwc", db_path.display());

    let options = url
        .parse::<sqlx::sqlite::SqliteConnectOptions>()
        .expect("url should parse")
        .create_if_missing(true);
    let pool = sqlx::SqlitePool::connect_with(options)
        .await
        .expect("pool should connect");
    sqlx::query("CREATE TABLE bad_responses (tenant_id TEXT NOT NULL, id TEXT NOT NULL, PRIMARY KEY (tenant_id, id))")
        .execute(&pool)
        .await
        .expect("manual create should succeed");
    pool.close().await;

    let result = SqliteResponseStore::new(&url, "bad_responses", "ok_conversations", None, None, None).await;
    let Err(err) = result else {
        panic!("init should fail on schema mismatch");
    };

    let msg = err.to_string();
    assert!(
        msg.contains("schema validation failed"),
        "error should mention schema validation: {msg}"
    );
    assert!(msg.contains("bad_responses"), "error should name the table: {msg}");
    assert!(msg.contains("created_at"), "error should list missing column: {msg}");
    assert!(msg.contains("model"), "error should list missing column: {msg}");
}

// -----------------------------------------------------------------------------
// Schema Validation (primary keys)
// -----------------------------------------------------------------------------

#[tokio::test]
async fn sqlite_rejects_table_with_incompatible_primary_key() {
    let dir = tempfile::tempdir().expect("tempdir should succeed");
    let db_path = dir.path().join("bad_pk.db");
    let url = format!("sqlite://{}?mode=rwc", db_path.display());

    let options = url
        .parse::<sqlx::sqlite::SqliteConnectOptions>()
        .expect("url should parse")
        .create_if_missing(true);
    let pool = sqlx::SqlitePool::connect_with(options)
        .await
        .expect("pool should connect");
    // Every expected column is present, but the table is keyed by id
    // alone. Without tenant_id in the primary key, INSERT OR REPLACE
    // collapses rows that share a response id across tenants, silently
    // destroying another tenant's record.
    sqlx::query(
        "CREATE TABLE bad_pk_responses (\
         id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, created_at BIGINT NOT NULL, \
         model TEXT NOT NULL, response_object TEXT NOT NULL, input TEXT NOT NULL, \
         messages TEXT NOT NULL)",
    )
    .execute(&pool)
    .await
    .expect("manual create should succeed");
    pool.close().await;

    let result = SqliteResponseStore::new(&url, "bad_pk_responses", "ok_conversations", None, None, None).await;
    let Err(err) = result else {
        panic!("init should fail on incompatible primary key");
    };

    let msg = err.to_string();
    assert!(
        msg.contains("schema validation failed"),
        "error should mention schema validation: {msg}"
    );
    assert!(msg.contains("bad_pk_responses"), "error should name the table: {msg}");
    assert!(
        msg.contains("primary key"),
        "error should mention the primary key: {msg}"
    );
    assert!(
        msg.contains("tenant_id"),
        "error should mention the expected tenant_id key column: {msg}"
    );
    assert!(
        msg.contains("migration"),
        "error should tell the operator a migration is required: {msg}"
    );
}

// -----------------------------------------------------------------------------
// Schema Validation (unique constraints)
// -----------------------------------------------------------------------------

#[tokio::test]
async fn sqlite_rejects_table_with_tenant_leaking_unique_constraint() {
    let dir = tempfile::tempdir().expect("tempdir should succeed");
    let db_path = dir.path().join("leak_unique.db");
    let url = format!("sqlite://{}?mode=rwc", db_path.display());

    let options = url
        .parse::<sqlx::sqlite::SqliteConnectOptions>()
        .expect("url should parse")
        .create_if_missing(true);
    let pool = sqlx::SqlitePool::connect_with(options)
        .await
        .expect("pool should connect");
    // The composite primary key is correct, but an extra UNIQUE(id) makes
    // id unique across tenants. SQLite REPLACE deletes any row violating a
    // UNIQUE or PRIMARY KEY constraint, so a second tenant's INSERT OR
    // REPLACE with the same id silently deletes the first tenant's row.
    // Checking only the primary key would miss this, so init must reject it.
    sqlx::query(
        "CREATE TABLE leak_responses (\
         tenant_id TEXT NOT NULL, id TEXT NOT NULL, created_at BIGINT NOT NULL, \
         model TEXT NOT NULL, response_object TEXT NOT NULL, input TEXT NOT NULL, \
         messages TEXT NOT NULL, PRIMARY KEY (tenant_id, id), UNIQUE (id))",
    )
    .execute(&pool)
    .await
    .expect("manual create should succeed");
    pool.close().await;

    let result = SqliteResponseStore::new(&url, "leak_responses", "leak_conversations", None, None, None).await;
    let Err(err) = result else {
        panic!("init should fail on a unique constraint that omits tenant_id");
    };

    let msg = err.to_string();
    assert!(
        msg.contains("schema validation failed"),
        "error should mention schema validation: {msg}"
    );
    assert!(msg.contains("leak_responses"), "error should name the table: {msg}");
    assert!(
        msg.contains("unexpected unique index") && msg.contains("only the primary key"),
        "error should explain a unique index beyond the primary key is rejected: {msg}"
    );
    assert!(
        msg.contains("migration"),
        "error should tell the operator a migration is required: {msg}"
    );
}

#[tokio::test]
async fn sqlite_rejects_table_with_case_insensitive_collation_on_key() {
    let dir = tempfile::tempdir().expect("tempdir should succeed");
    let db_path = dir.path().join("collate_key.db");
    let url = format!("sqlite://{}?mode=rwc", db_path.display());

    let options = url
        .parse::<sqlx::sqlite::SqliteConnectOptions>()
        .expect("url should parse")
        .create_if_missing(true);
    let pool = sqlx::SqlitePool::connect_with(options)
        .await
        .expect("pool should connect");
    // tenant_id carries COLLATE NOCASE, so the primary key treats "Tenant-A"
    // and "tenant-a" as equal. INSERT OR REPLACE then deletes one tenant's row
    // when the other writes the same id -- a cross-tenant collapse. The column
    // names still cover the primary key, so only a collation check catches it.
    sqlx::query(
        "CREATE TABLE collate_responses (\
         tenant_id TEXT NOT NULL COLLATE NOCASE, id TEXT NOT NULL, created_at BIGINT NOT NULL, \
         model TEXT NOT NULL, response_object TEXT NOT NULL, input TEXT NOT NULL, \
         messages TEXT NOT NULL, PRIMARY KEY (tenant_id, id))",
    )
    .execute(&pool)
    .await
    .expect("manual create should succeed");
    pool.close().await;

    let result = SqliteResponseStore::new(&url, "collate_responses", "collate_conversations", None, None, None).await;
    let Err(err) = result else {
        panic!("init should fail on a case-insensitive collation over a primary key column");
    };

    let msg = err.to_string();
    assert!(msg.contains("schema validation failed"), "{msg}");
    assert!(msg.contains("collate_responses"), "error should name the table: {msg}");
    assert!(
        msg.contains("tenant_id"),
        "error should name the offending column: {msg}"
    );
    assert!(
        msg.to_ascii_lowercase().contains("collation"),
        "error should explain the collation is unsafe: {msg}"
    );
    assert!(msg.contains("migration"), "{msg}");
}

#[tokio::test]
async fn sqlite_rejects_table_with_non_text_affinity_key() {
    let dir = tempfile::tempdir().expect("tempdir should succeed");
    let db_path = dir.path().join("affinity_key.db");
    let url = format!("sqlite://{}?mode=rwc", db_path.display());

    let options = url
        .parse::<sqlx::sqlite::SqliteConnectOptions>()
        .expect("url should parse")
        .create_if_missing(true);
    let pool = sqlx::SqlitePool::connect_with(options)
        .await
        .expect("pool should connect");
    // id is declared INTEGER, so it has numeric affinity: the distinct text ids
    // "1" and "01" are both stored as the integer 1 and collide, letting
    // INSERT OR REPLACE delete a distinct response. The column names and
    // collation look correct, so only an affinity check rejects it.
    sqlx::query(
        "CREATE TABLE affinity_responses (\
         tenant_id TEXT NOT NULL, id INTEGER NOT NULL, created_at BIGINT NOT NULL, \
         model TEXT NOT NULL, response_object TEXT NOT NULL, input TEXT NOT NULL, \
         messages TEXT NOT NULL, PRIMARY KEY (tenant_id, id))",
    )
    .execute(&pool)
    .await
    .expect("manual create should succeed");
    pool.close().await;

    let result = SqliteResponseStore::new(&url, "affinity_responses", "affinity_conversations", None, None, None).await;
    let Err(err) = result else {
        panic!("init should fail on a primary key column without TEXT affinity");
    };

    let msg = err.to_string();
    assert!(msg.contains("schema validation failed"), "{msg}");
    assert!(msg.contains("affinity_responses"), "error should name the table: {msg}");
    assert!(msg.contains("id"), "error should name the offending column: {msg}");
    assert!(
        msg.to_ascii_lowercase().contains("affinity"),
        "error should explain the affinity is unsafe: {msg}"
    );
    assert!(msg.contains("migration"), "{msg}");
}

// -----------------------------------------------------------------------------
// Schema Version
// -----------------------------------------------------------------------------

#[tokio::test]
async fn sqlite_stamps_schema_version_on_fresh_db() {
    let dir = tempfile::tempdir().expect("tempdir should succeed");
    let db_path = dir.path().join("version.db");
    let url = format!("sqlite://{}?mode=rwc", db_path.display());

    let _store = SqliteResponseStore::new(&url, "vr", "vc", None, None, None)
        .await
        .expect("store creation should succeed");

    let options = url
        .parse::<sqlx::sqlite::SqliteConnectOptions>()
        .expect("url should parse");
    let pool = sqlx::SqlitePool::connect_with(options)
        .await
        .expect("pool should connect");
    let version: i64 = sqlx::query_scalar("SELECT version FROM vr_schema_version")
        .fetch_one(&pool)
        .await
        .expect("version row should exist");
    assert_eq!(version, 2, "fresh store should stamp version 2");
}

#[tokio::test]
async fn sqlite_rejects_schema_version_mismatch() {
    let dir = tempfile::tempdir().expect("tempdir should succeed");
    let db_path = dir.path().join("bad_version.db");
    let url = format!("sqlite://{}?mode=rwc", db_path.display());

    let options = url
        .parse::<sqlx::sqlite::SqliteConnectOptions>()
        .expect("url should parse")
        .create_if_missing(true);
    let pool = sqlx::SqlitePool::connect_with(options)
        .await
        .expect("pool should connect");
    sqlx::query("CREATE TABLE vr_schema_version (version BIGINT NOT NULL PRIMARY KEY)")
        .execute(&pool)
        .await
        .expect("create should succeed");
    sqlx::query("INSERT INTO vr_schema_version (version) VALUES (99)")
        .execute(&pool)
        .await
        .expect("insert should succeed");
    pool.close().await;

    let result = SqliteResponseStore::new(&url, "vr", "vc", None, None, None).await;
    let Err(err) = result else {
        panic!("init should fail on version mismatch");
    };

    let msg = err.to_string();
    assert!(
        msg.contains("schema version mismatch"),
        "error should mention version mismatch: {msg}"
    );
    assert!(msg.contains("99"), "error should show stored version: {msg}");
    assert!(
        msg.contains("migration required"),
        "error should mention migration: {msg}"
    );
}

#[tokio::test]
async fn sqlite_accepts_matching_schema_version() {
    let dir = tempfile::tempdir().expect("tempdir should succeed");
    let db_path = dir.path().join("good_version.db");
    let url = format!("sqlite://{}?mode=rwc", db_path.display());

    let _store = SqliteResponseStore::new(&url, "vr", "vc", None, None, None)
        .await
        .expect("first init should succeed");

    let _store2 = SqliteResponseStore::new(&url, "vr", "vc", None, None, None)
        .await
        .expect("second init with matching version should succeed");
}

#[tokio::test]
async fn sqlite_v1_text_schema_migrates_to_v2_preserving_rows() {
    let dir = tempfile::tempdir().expect("tempdir should succeed");
    let db_path = dir.path().join("migrate.db");
    let url = format!("sqlite://{}?mode=rwc", db_path.display());

    let options = url
        .parse::<sqlx::sqlite::SqliteConnectOptions>()
        .expect("url should parse")
        .create_if_missing(true);

    // Build a legacy version-1 layout: TEXT payload columns, stamped v1,
    // with a plain-JSON row written the way the pre-bytes store would.
    let pool = sqlx::SqlitePool::connect_with(options.clone())
        .await
        .expect("pool should connect");
    for stmt in [
        "CREATE TABLE mr (tenant_id TEXT NOT NULL, id TEXT NOT NULL, created_at BIGINT NOT NULL, \
         model TEXT NOT NULL, response_object TEXT NOT NULL, input TEXT NOT NULL, messages TEXT NOT NULL, \
         PRIMARY KEY (tenant_id, id))",
        "CREATE TABLE mc (conversation_id TEXT NOT NULL, tenant_id TEXT NOT NULL, created_at BIGINT NOT NULL, \
         metadata TEXT NOT NULL, messages TEXT NOT NULL, PRIMARY KEY (conversation_id, tenant_id))",
        "CREATE TABLE mr_schema_version (version BIGINT NOT NULL PRIMARY KEY)",
        "INSERT INTO mr_schema_version (version) VALUES (1)",
        "INSERT INTO mr (tenant_id, id, created_at, model, response_object, input, messages) \
         VALUES ('tenant_a', 'legacy_resp', 1000, 'gpt-4.1', \
         '{\"id\":\"legacy_resp\",\"model\":\"gpt-4.1\"}', '\"hi\"', '[{\"role\":\"user\"}]')",
    ] {
        sqlx::query(stmt)
            .execute(&pool)
            .await
            .expect("legacy setup should succeed");
    }
    pool.close().await;

    // Before migration the store refuses to start.
    let before = SqliteResponseStore::new(&url, "mr", "mc", None, None, None).await;
    assert!(
        before.is_err_and(|e| e.to_string().contains("schema version mismatch")),
        "store must refuse a version-1 database"
    );

    // Apply the documented operator migration: CAST the responses payload
    // columns to BLOB storage class and bump the schema version.
    let pool = sqlx::SqlitePool::connect_with(options)
        .await
        .expect("pool should connect");
    for stmt in [
        "UPDATE mr SET response_object = CAST(response_object AS BLOB), \
         input = CAST(input AS BLOB), messages = CAST(messages AS BLOB)",
        "UPDATE mr_schema_version SET version = 2",
    ] {
        sqlx::query(stmt)
            .execute(&pool)
            .await
            .expect("migration should succeed");
    }
    pool.close().await;

    // After migration the store starts and the legacy row reads back intact.
    let store = SqliteResponseStore::new(&url, "mr", "mc", None, None, None)
        .await
        .expect("store should start on a migrated version-2 database");

    let fetched = store
        .get_response("tenant_a", "legacy_resp")
        .await
        .expect("get should succeed")
        .expect("legacy row should be readable after migration");
    assert_eq!(fetched.input, json!("hi"), "legacy plain-JSON input should decode");
    assert_eq!(fetched.model, "gpt-4.1", "legacy model should be intact");

    // A fresh write through the migrated store also round-trips.
    let record = make_response_record("post_mig", "tenant_a", 2000);
    store.upsert_response(&record).await.expect("upsert should succeed");
    let round = store
        .get_response("tenant_a", "post_mig")
        .await
        .expect("get should succeed")
        .expect("new row should exist");
    assert_eq!(round.response_object, record.response_object, "new write round-trips");
}

// -----------------------------------------------------------------------------
// Concurrent Access
// -----------------------------------------------------------------------------

#[tokio::test]
async fn concurrent_upserts_do_not_lose_data() {
    let dir = tempfile::tempdir().expect("tempdir should succeed");
    let store = Arc::new(make_file_store(&dir, None).await);
    let mut handles = Vec::new();

    for i in 0..20 {
        let store = Arc::clone(&store);
        handles.push(tokio::spawn(async move {
            let id = format!("resp_{i}");
            let record = make_response_record(&id, "tenant_a", i64::from(i));
            store
                .upsert_response(&record)
                .await
                .expect("concurrent upsert should succeed");
        }));
    }

    for handle in handles {
        handle.await.expect("task should not panic");
    }

    for i in 0..20 {
        let id = format!("resp_{i}");
        let fetched = store.get_response("tenant_a", &id).await.expect("get should succeed");
        assert!(fetched.is_some(), "response {id} should exist after concurrent upsert");
    }
}

#[tokio::test]
async fn concurrent_reads_and_writes() {
    let dir = tempfile::tempdir().expect("tempdir should succeed");
    let store = Arc::new(make_file_store(&dir, None).await);

    let record = make_response_record("resp_rw", "tenant_a", 1000);
    store
        .upsert_response(&record)
        .await
        .expect("seed upsert should succeed");

    let mut handles = Vec::new();

    for _ in 0..10 {
        let store = Arc::clone(&store);
        handles.push(tokio::spawn(async move {
            let fetched = store
                .get_response("tenant_a", "resp_rw")
                .await
                .expect("concurrent read should succeed");
            assert!(fetched.is_some(), "seeded record should be readable under contention");
        }));
    }

    for i in 0..10 {
        let store = Arc::clone(&store);
        handles.push(tokio::spawn(async move {
            let id = format!("resp_concurrent_{i}");
            let record = make_response_record(&id, "tenant_a", 2000 + i64::from(i));
            store
                .upsert_response(&record)
                .await
                .expect("concurrent write should succeed");
        }));
    }

    for handle in handles {
        handle.await.expect("task should not panic");
    }
}

#[tokio::test]
async fn concurrent_create_items_and_sync_messages_assigns_distinct_positions() {
    let dir = tempfile::tempdir().expect("tempdir should succeed");
    let store = Arc::new(make_file_store(&dir, Some("test_conversation_items")).await);

    let conv = ConversationRecord {
        conversation_id: "conv_1".to_owned(),
        tenant_id: "tenant_a".to_owned(),
        created_at: 1000,
        metadata: json!({}),
        messages: json!([]),
    };
    store.upsert_conversation(&conv).await.expect("upsert should succeed");

    let store_a = Arc::clone(&store);
    let store_b = Arc::clone(&store);

    let handle_a = tokio::spawn(async move {
        let items = [make_conversation_item("item_a", "tenant_a", "conv_1", 0)];
        store_a
            .create_items_and_sync_messages("tenant_a", "conv_1", &items)
            .await
            .expect("task A should succeed");
    });

    let handle_b = tokio::spawn(async move {
        let items = [make_conversation_item("item_b", "tenant_a", "conv_1", 0)];
        store_b
            .create_items_and_sync_messages("tenant_a", "conv_1", &items)
            .await
            .expect("task B should succeed");
    });

    handle_a.await.expect("task A should not panic");
    handle_b.await.expect("task B should not panic");

    let items = store
        .list_conversation_items("tenant_a", "conv_1", None, 100, true)
        .await
        .expect("list should succeed");

    assert_eq!(items.len(), 2, "both items should be present");

    let positions: Vec<i64> = items.iter().map(|i| i.position).collect();
    assert_ne!(positions[0], positions[1], "positions must be distinct");
    assert!(
        positions.contains(&1) && positions.contains(&2),
        "positions should be 1 and 2, got {positions:?}",
    );

    let conv_record = ConversationItemStore::get_conversation(&*store, "tenant_a", "conv_1")
        .await
        .expect("get should succeed")
        .expect("conversation should exist");
    let messages = conv_record.messages.as_array().expect("messages should be an array");
    assert_eq!(messages.len(), 2, "messages cache should include both items");
}

// -----------------------------------------------------------------------------
// Registry
// -----------------------------------------------------------------------------

#[tokio::test]
async fn registry_register_and_get() {
    let registry = ResponseStoreRegistry::new();
    let store: Arc<dyn ResponseStore> = Arc::new(make_store().await);
    registry
        .register(&Arc::from("primary"), Arc::clone(&store))
        .expect("register should succeed");

    let fetched = registry.get("primary");
    assert!(fetched.is_some(), "registered store should be retrievable");
}

#[test]
fn registry_get_missing_returns_none() {
    let registry = ResponseStoreRegistry::new();
    assert!(
        registry.get("nonexistent").is_none(),
        "get on empty registry should return None"
    );
}

#[tokio::test]
async fn registry_duplicate_registration_fails() {
    let registry = ResponseStoreRegistry::new();
    let store: Arc<dyn ResponseStore> = Arc::new(make_store().await);
    let name = Arc::from("dup");
    registry
        .register(&name, Arc::clone(&store))
        .expect("first register should succeed");

    let result = registry.register(&name, store);
    assert!(
        matches!(result, Err(StoreError::Unavailable(_))),
        "duplicate registration should return StoreError::Unavailable"
    );
}

#[test]
fn registry_default_is_empty() {
    let registry = ResponseStoreRegistry::default();
    assert!(
        registry.get("anything").is_none(),
        "default registry should have no stores"
    );
}

#[test]
fn registry_clone_shares_storage() {
    let registry = ResponseStoreRegistry::new();
    let cloned = registry.clone();
    assert!(
        registry.shares_storage_with(&cloned),
        "cloned registry handles should share backing storage"
    );
}

#[test]
fn registry_new_has_independent_storage() {
    let first = ResponseStoreRegistry::new();
    let second = ResponseStoreRegistry::new();
    assert!(
        !first.shares_storage_with(&second),
        "independent registries should not share backing storage"
    );
}

// -----------------------------------------------------------------------------
// PostgreSQL Backend (requires running instance, DATABASE_URL env var)
// -----------------------------------------------------------------------------

fn pg_database_url() -> String {
    std::env::var("DATABASE_URL").expect("DATABASE_URL must be set for Postgres tests")
}

fn pg_unique_suffix() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let tid = std::thread::current().id();
    format!("{id}_{tid:?}")
        .replace(|c: char| !c.is_ascii_alphanumeric() && c != '_', "_")
        .to_lowercase()
}

/// Fixture for the `PostgreSQL` schema-validation integration tests.
///
/// Each of those tests hand-crafts a pre-existing schema, points
/// `PostgresResponseStore::new` at it, and asserts that startup validation accepts
/// or rejects it. The connect -> pre-clean -> craft -> init -> assert -> clean-up
/// boilerplate is identical across them; only the crafted DDL and the expected
/// error text differ. This fixture collapses that boilerplate so each test reads as
/// "given this schema, init must reject it because <reason>".
///
/// Table names are suffix-scoped via [`pg_unique_suffix`] so tests running on
/// separate threads never collide. Because the suffix is deterministic per thread,
/// a prior run that panicked before cleanup can leave tables behind; every init
/// drops the managed set first so a stale table cannot mask the crafted one.
struct PgSchemaFixture {
    url: String,
    suffix: String,
    responses: String,
    conversations: String,
    version: String,
}

impl PgSchemaFixture {
    fn new(prefix: &str) -> Self {
        let suffix = pg_unique_suffix();
        let responses = format!("{prefix}_responses_{suffix}");
        let conversations = format!("{prefix}_conversations_{suffix}");
        let version = format!("{responses}_schema_version");
        Self {
            url: pg_database_url(),
            suffix,
            responses,
            conversations,
            version,
        }
    }

    /// A suffix-scoped name for a non-table object (a collation) so parallel
    /// threads never collide.
    fn name(&self, base: &str) -> String {
        format!("{base}_{}", self.suffix)
    }

    /// Every table this fixture owns, in drop order.
    fn all_tables(&self) -> Vec<&str> {
        vec![&self.responses, &self.conversations, &self.version]
    }

    /// Drop every owned table, then run `teardown` (drops for non-table objects such
    /// as collations, which must follow the tables that depend on them). All
    /// statements are idempotent (`IF EXISTS`).
    async fn drop_all(&self, pool: &sqlx::PgPool, teardown: &[String]) {
        use sqlx::AssertSqlSafe;
        for table in self.all_tables() {
            let sql = format!("DROP TABLE IF EXISTS {table}");
            sqlx::query(AssertSqlSafe(sql.as_str()))
                .execute(pool)
                .await
                .expect("drop table should succeed");
        }
        for stmt in teardown {
            sqlx::query(AssertSqlSafe(stmt.as_str()))
                .execute(pool)
                .await
                .expect("teardown should succeed");
        }
    }

    /// Pre-clean, run `setup` to craft the schema, run init, clean up, and return the
    /// init result. `teardown` runs as both pre-clean and post-clean.
    async fn init(&self, setup: &[String], teardown: &[String]) -> Result<PostgresResponseStore, StoreError> {
        use sqlx::AssertSqlSafe;

        let options: sqlx::postgres::PgConnectOptions = self.url.parse().expect("url should parse");
        let pool = Box::pin(sqlx::PgPool::connect_with(options))
            .await
            .expect("pool should connect");
        self.drop_all(&pool, teardown).await;
        for stmt in setup {
            sqlx::query(AssertSqlSafe(stmt.as_str()))
                .execute(&pool)
                .await
                .expect("setup should succeed");
        }
        pool.close().await;

        let result = Box::pin(PostgresResponseStore::new(
            &self.url,
            &self.responses,
            &self.conversations,
            None,
            Some(SslMode::Disable),
            None,
            None,
            None,
        ))
        .await;

        let cleanup_pool = Box::pin(sqlx::PgPool::connect(&self.url)).await.expect("cleanup pool");
        self.drop_all(&cleanup_pool, teardown).await;
        result
    }

    /// Init against the crafted schema, require rejection, and return the error text.
    async fn expect_rejected(&self, setup: &[String], teardown: &[String]) -> String {
        match self.init(setup, teardown).await {
            Ok(_) => panic!("init should reject the crafted schema"),
            Err(err) => err.to_string(),
        }
    }
}

#[test]
fn pg_ssl_mode_defaults_to_verify_full() {
    let mode = SslMode::default();
    assert!(
        matches!(mode, SslMode::VerifyFull),
        "SslMode should default to VerifyFull"
    );
}

#[test]
fn pg_ssl_mode_deserializes_verified_modes() {
    let verify_ca: SslMode = serde_json::from_str("\"verify-ca\"").expect("verify-ca should deserialize");
    let verify_full: SslMode = serde_json::from_str("\"verify-full\"").expect("verify-full should deserialize");

    assert!(matches!(verify_ca, SslMode::VerifyCa), "verify-ca should be supported");
    assert!(
        matches!(verify_full, SslMode::VerifyFull),
        "verify-full should be supported"
    );
}

#[test]
fn pg_ssl_mode_converts_to_pg_ssl_mode() {
    use sqlx::postgres::PgSslMode;

    assert!(
        matches!(PgSslMode::from(SslMode::Disable), PgSslMode::Disable),
        "Disable should map"
    );
    assert!(
        matches!(PgSslMode::from(SslMode::Prefer), PgSslMode::Prefer),
        "Prefer should map"
    );
    assert!(
        matches!(PgSslMode::from(SslMode::Require), PgSslMode::Require),
        "Require should map"
    );
    assert!(
        matches!(PgSslMode::from(SslMode::VerifyCa), PgSslMode::VerifyCa),
        "VerifyCa should map"
    );
    assert!(
        matches!(PgSslMode::from(SslMode::VerifyFull), PgSslMode::VerifyFull),
        "VerifyFull should map"
    );
}

#[tokio::test]
#[ignore]
async fn pg_nonexistent_ssl_root_cert_fails() {
    let url = pg_database_url();
    let suffix = pg_unique_suffix();
    let result = Box::pin(PostgresResponseStore::new(
        &url,
        &format!("test_responses_{suffix}"),
        &format!("test_conversations_{suffix}"),
        None,
        Some(SslMode::VerifyCa),
        Some("/nonexistent/ca.pem"),
        None,
        None,
    ))
    .await;

    let Err(err) = result else {
        panic!("nonexistent ssl_root_cert should fail");
    };
    assert!(
        matches!(err, StoreError::Database(_)),
        "error should be StoreError::Database: {err}"
    );
}

#[tokio::test]
#[ignore]
async fn pg_rejects_table_with_missing_columns() {
    let fx = PgSchemaFixture::new("missing_cols");
    let msg = fx
        .expect_rejected(
            &[format!(
                "CREATE TABLE {} (tenant_id TEXT NOT NULL, id TEXT NOT NULL, PRIMARY KEY (tenant_id, id))",
                fx.responses
            )],
            &[],
        )
        .await;

    assert!(
        msg.contains("schema validation failed"),
        "error should mention schema validation: {msg}"
    );
    assert!(msg.contains(&fx.responses), "error should name the table: {msg}");
    assert!(msg.contains("created_at"), "error should list missing column: {msg}");
}

#[tokio::test]
#[ignore]
async fn pg_rejects_table_with_incompatible_primary_key() {
    let fx = PgSchemaFixture::new("bad_pk");
    // Every column is present, but the table is keyed by id alone.
    // ON CONFLICT (tenant_id, id) would fail at runtime, so init must
    // reject the incompatible key up front.
    let msg = fx
        .expect_rejected(
            &[format!(
                "CREATE TABLE {} (\
                 id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, created_at BIGINT NOT NULL, \
                 model TEXT NOT NULL, response_object TEXT NOT NULL, input TEXT NOT NULL, \
                 messages TEXT NOT NULL)",
                fx.responses
            )],
            &[],
        )
        .await;

    assert!(
        msg.contains("schema validation failed"),
        "error should mention schema validation: {msg}"
    );
    assert!(msg.contains(&fx.responses), "error should name the table: {msg}");
    assert!(
        msg.contains("primary key"),
        "error should mention the primary key: {msg}"
    );
    assert!(
        msg.contains("migration"),
        "error should tell the operator a migration is required: {msg}"
    );
}

#[tokio::test]
#[ignore]
async fn pg_rejects_table_with_tenant_leaking_unique_constraint() {
    let fx = PgSchemaFixture::new("leak");
    // The composite primary key is correct, but the extra UNIQUE(id) makes
    // id unique across tenants. ON CONFLICT (tenant_id, id) cannot satisfy
    // that constraint, so a cross-tenant write would fail at runtime; init
    // must reject the schema up front.
    let msg = fx
        .expect_rejected(
            &[format!(
                "CREATE TABLE {} (\
                 tenant_id TEXT NOT NULL, id TEXT NOT NULL, created_at BIGINT NOT NULL, \
                 model TEXT NOT NULL, response_object TEXT NOT NULL, input TEXT NOT NULL, \
                 messages TEXT NOT NULL, PRIMARY KEY (tenant_id, id), UNIQUE (id))",
                fx.responses
            )],
            &[],
        )
        .await;

    assert!(
        msg.contains("schema validation failed"),
        "error should mention schema validation: {msg}"
    );
    assert!(msg.contains(&fx.responses), "error should name the table: {msg}");
    assert!(
        msg.contains("unexpected unique index") && msg.contains("only the primary key"),
        "error should explain a unique index beyond the primary key is rejected: {msg}"
    );
    assert!(
        msg.contains("migration"),
        "error should tell the operator a migration is required: {msg}"
    );
}

#[tokio::test]
#[ignore]
async fn pg_rejects_table_with_deferrable_primary_key() {
    let fx = PgSchemaFixture::new("defer_pk");
    // The key columns are correct, but a DEFERRABLE primary key cannot serve as
    // an ON CONFLICT arbiter: PostgreSQL rejects every upsert with "ON CONFLICT
    // does not support deferrable unique constraints ... as arbiters". A check
    // that only inspects columns would accept this and break every write, so
    // init must reject it at startup.
    let msg = fx
        .expect_rejected(
            &[format!(
                "CREATE TABLE {} (\
                 tenant_id TEXT NOT NULL, id TEXT NOT NULL, created_at BIGINT NOT NULL, \
                 model TEXT NOT NULL, response_object TEXT NOT NULL, input TEXT NOT NULL, \
                 messages TEXT NOT NULL, PRIMARY KEY (tenant_id, id) DEFERRABLE INITIALLY DEFERRED)",
                fx.responses
            )],
            &[],
        )
        .await;

    assert!(msg.contains("schema validation failed"), "{msg}");
    assert!(msg.contains(&fx.responses), "error should name the table: {msg}");
    assert!(
        msg.to_ascii_lowercase().contains("deferrable"),
        "error should explain the constraint is deferrable: {msg}"
    );
    assert!(msg.contains("migration"), "{msg}");
}

#[tokio::test]
#[ignore]
async fn pg_rejects_table_with_case_insensitive_collation_on_key() {
    let fx = PgSchemaFixture::new("ci_coll");
    let collation = fx.name("ci_coll");
    // A non-deterministic collation folds case, so the primary key index treats
    // 'Tenant-A' and 'tenant-a' as equal: ON CONFLICT (tenant_id, id) then
    // overwrites one tenant's row with another's. The key columns are correctly
    // named, so only a comparison-semantics check catches this.
    let msg = fx
        .expect_rejected(
            &[
                format!(
                    "CREATE COLLATION {collation} (provider = icu, locale = 'und-u-ks-level2', deterministic = false)"
                ),
                format!(
                    "CREATE TABLE {} (\
                     tenant_id TEXT COLLATE {collation} NOT NULL, id TEXT NOT NULL, created_at BIGINT NOT NULL, \
                     model TEXT NOT NULL, response_object TEXT NOT NULL, input TEXT NOT NULL, \
                     messages TEXT NOT NULL, PRIMARY KEY (tenant_id, id))",
                    fx.responses
                ),
            ],
            &[format!("DROP COLLATION IF EXISTS {collation}")],
        )
        .await;

    assert!(msg.contains("schema validation failed"), "{msg}");
    assert!(msg.contains(&fx.responses), "error should name the table: {msg}");
    assert!(
        msg.to_ascii_lowercase().contains("collation"),
        "error should explain the collation folds comparisons: {msg}"
    );
    assert!(msg.contains("migration"), "{msg}");
}

#[tokio::test]
#[ignore]
async fn pg_rejects_table_with_citext_key() {
    let fx = PgSchemaFixture::new("citext");
    // citext is case-insensitive by type, not by collation: the index reports a
    // deterministic default collation, so only a key-column type check catches
    // that 'Tenant-A' and 'tenant-a' collapse under ON CONFLICT. The extension is
    // shared and left in place.
    let msg = fx
        .expect_rejected(
            &[
                "CREATE EXTENSION IF NOT EXISTS citext".to_owned(),
                format!(
                    "CREATE TABLE {} (\
                     tenant_id CITEXT NOT NULL, id TEXT NOT NULL, created_at BIGINT NOT NULL, \
                     model TEXT NOT NULL, response_object TEXT NOT NULL, input TEXT NOT NULL, \
                     messages TEXT NOT NULL, PRIMARY KEY (tenant_id, id))",
                    fx.responses
                ),
            ],
            &[],
        )
        .await;

    assert!(msg.contains("schema validation failed"), "{msg}");
    assert!(msg.contains(&fx.responses), "error should name the table: {msg}");
    assert!(
        msg.to_ascii_lowercase().contains("citext"),
        "error should name the case-insensitive type: {msg}"
    );
    assert!(msg.contains("migration"), "{msg}");
}

#[tokio::test]
#[ignore]
async fn pg_accepts_varchar_key_columns() {
    let fx = PgSchemaFixture::new("varchar_key");
    // varchar is allow-listed alongside text, but PostgreSQL backs a varchar key
    // with the text_ops operator class (opcintype = text), not a varchar_ops. A
    // trusted-opclass check that compared the class input type to the column type
    // would see text != varchar and wrongly reject a valid schema, so varchar keys
    // must be accepted.
    let result = fx
        .init(
            &[format!(
                "CREATE TABLE {} (\
                 tenant_id VARCHAR NOT NULL, id VARCHAR NOT NULL, created_at BIGINT NOT NULL, \
                 model TEXT NOT NULL, response_object TEXT NOT NULL, input TEXT NOT NULL, \
                 messages TEXT NOT NULL, PRIMARY KEY (tenant_id, id))",
                fx.responses
            )],
            &[],
        )
        .await;

    assert!(
        result.is_ok(),
        "varchar keys use the text_ops operator class and must be accepted: {:?}",
        result.err()
    );
}

#[tokio::test]
#[ignore]
async fn pg_rejects_schema_version_mismatch() {
    let fx = PgSchemaFixture::new("ver");
    // Seed only the version table with an unsupported version. Init creates the
    // (valid) responses and conversations tables, so validation passes and the
    // version check is what rejects startup.
    let msg = fx
        .expect_rejected(
            &[
                format!("CREATE TABLE {} (version BIGINT NOT NULL PRIMARY KEY)", fx.version),
                format!("INSERT INTO {} (version) VALUES (99)", fx.version),
            ],
            &[],
        )
        .await;

    assert!(
        msg.contains("schema version mismatch"),
        "error should mention version mismatch: {msg}"
    );
    assert!(msg.contains("99"), "error should show stored version: {msg}");
}

#[tokio::test]
#[ignore]
async fn pg_v1_text_schema_migrates_to_v2_bytea_preserving_rows() {
    use sqlx::AssertSqlSafe;

    let url = pg_database_url();
    let suffix = pg_unique_suffix();
    let resp_table = format!("mig_r_{suffix}");
    let conv_table = format!("mig_c_{suffix}");
    let ver_table = format!("{resp_table}_schema_version");

    let options: sqlx::postgres::PgConnectOptions = url.parse().expect("url should parse");
    let pool = Box::pin(sqlx::PgPool::connect_with(options))
        .await
        .expect("pool should connect");

    // Build a legacy version-1 layout: TEXT payload columns, stamped v1,
    // with a plain-JSON row written the way the pre-bytes store would.
    for stmt in [
        format!(
            "CREATE TABLE {resp_table} (
                tenant_id TEXT NOT NULL, id TEXT NOT NULL, created_at BIGINT NOT NULL,
                model TEXT NOT NULL, response_object TEXT NOT NULL, input TEXT NOT NULL,
                messages TEXT NOT NULL, PRIMARY KEY (tenant_id, id))"
        ),
        format!(
            "CREATE TABLE {conv_table} (
                conversation_id TEXT NOT NULL, tenant_id TEXT NOT NULL, created_at BIGINT NOT NULL,
                metadata TEXT NOT NULL, messages TEXT NOT NULL, PRIMARY KEY (conversation_id, tenant_id))"
        ),
        format!("CREATE TABLE {ver_table} (version BIGINT NOT NULL PRIMARY KEY)"),
        format!("INSERT INTO {ver_table} (version) VALUES (1)"),
        format!(
            "INSERT INTO {resp_table} (tenant_id, id, created_at, model, response_object, input, messages) \
             VALUES ('tenant_a', 'legacy_resp', 1000, 'gpt-4.1', \
             '{{\"id\":\"legacy_resp\",\"model\":\"gpt-4.1\"}}', '\"hi\"', '[{{\"role\":\"user\"}}]')"
        ),
    ] {
        sqlx::query(AssertSqlSafe(stmt.as_str()))
            .execute(&pool)
            .await
            .expect("legacy setup should succeed");
    }

    // Before migration the store refuses to start.
    let before = Box::pin(PostgresResponseStore::new(
        &url,
        &resp_table,
        &conv_table,
        None,
        Some(SslMode::Disable),
        None,
        None,
        None,
    ))
    .await;
    assert!(
        before.is_err_and(|e| e.to_string().contains("schema version mismatch")),
        "store must refuse a version-1 database"
    );

    // Apply the documented operator migration. Only the responses table's
    // payload columns become BYTEA; conversations stay TEXT.
    for stmt in [
        format!(
            "ALTER TABLE {resp_table} \
             ALTER COLUMN response_object TYPE BYTEA USING convert_to(response_object, 'UTF8'), \
             ALTER COLUMN input TYPE BYTEA USING convert_to(input, 'UTF8'), \
             ALTER COLUMN messages TYPE BYTEA USING convert_to(messages, 'UTF8')"
        ),
        format!("UPDATE {ver_table} SET version = 2"),
    ] {
        sqlx::query(AssertSqlSafe(stmt.as_str()))
            .execute(&pool)
            .await
            .expect("migration should succeed");
    }
    pool.close().await;

    // After migration the store starts and the legacy row reads back intact.
    let store = Box::pin(PostgresResponseStore::new(
        &url,
        &resp_table,
        &conv_table,
        None,
        Some(SslMode::Disable),
        None,
        None,
        None,
    ))
    .await
    .expect("store should start on a migrated version-2 database");

    let fetched = store
        .get_response("tenant_a", "legacy_resp")
        .await
        .expect("get should succeed")
        .expect("legacy row should be readable after migration");
    assert_eq!(fetched.input, json!("hi"), "legacy plain-JSON input should decode");
    assert_eq!(fetched.model, "gpt-4.1", "legacy model should be intact");

    // A fresh write through the migrated store also round-trips.
    let record = make_response_record("post_mig", "tenant_a", 2000);
    store.upsert_response(&record).await.expect("upsert should succeed");
    let round = store
        .get_response("tenant_a", "post_mig")
        .await
        .expect("get should succeed")
        .expect("new row should exist");
    assert_eq!(round.response_object, record.response_object, "new write round-trips");

    let cleanup_pool = Box::pin(sqlx::PgPool::connect(&url)).await.expect("cleanup pool");
    for table in [&ver_table, &resp_table, &conv_table] {
        let drop_sql = format!("DROP TABLE IF EXISTS {table}");
        sqlx::query(AssertSqlSafe(drop_sql.as_str()))
            .execute(&cleanup_pool)
            .await
            .expect("cleanup should succeed");
    }
}

async fn make_pg_store() -> PostgresResponseStore {
    let url = pg_database_url();
    let suffix = pg_unique_suffix();
    Box::pin(PostgresResponseStore::new(
        &url,
        &format!("test_responses_{suffix}"),
        &format!("test_conversations_{suffix}"),
        None,
        Some(SslMode::Disable),
        None,
        None,
        None,
    ))
    .await
    .expect("postgres store creation should succeed")
}

async fn make_pg_compressed_store() -> PostgresResponseStore {
    let url = pg_database_url();
    let suffix = pg_unique_suffix();
    Box::pin(PostgresResponseStore::new(
        &url,
        &format!("test_responses_{suffix}"),
        &format!("test_conversations_{suffix}"),
        Some(&format!("test_conversation_items_{suffix}")),
        Some(SslMode::Disable),
        None,
        None,
        Some(&zstd_compression()),
    ))
    .await
    .expect("postgres compressed store creation should succeed")
}

#[tokio::test]
#[ignore]
async fn pg_compressed_store_roundtrips_response() {
    let store = make_pg_compressed_store().await;
    let record = make_response_record("resp_zstd", "tenant_a", 1000);
    store.upsert_response(&record).await.expect("upsert should succeed");

    let fetched = store
        .get_response("tenant_a", "resp_zstd")
        .await
        .expect("get should succeed")
        .expect("record should exist");

    assert_eq!(
        fetched.response_object, record.response_object,
        "response_object round-trips"
    );
    assert_eq!(fetched.input, record.input, "input round-trips");
    assert_eq!(fetched.messages, record.messages, "messages round-trip");
}

#[tokio::test]
#[ignore]
async fn pg_compression_is_backward_compatible_with_plain_rows() {
    let url = pg_database_url();
    let suffix = pg_unique_suffix();
    let responses_table = format!("test_responses_{suffix}");
    let conversations_table = format!("test_conversations_{suffix}");

    let plain = Box::pin(PostgresResponseStore::new(
        &url,
        &responses_table,
        &conversations_table,
        None,
        Some(SslMode::Disable),
        None,
        None,
        None,
    ))
    .await
    .expect("postgres store creation should succeed");
    let record = make_response_record("resp_plain", "tenant_a", 1000);
    plain.upsert_response(&record).await.expect("upsert should succeed");
    drop(plain);

    let compressed = Box::pin(PostgresResponseStore::new(
        &url,
        &responses_table,
        &conversations_table,
        None,
        Some(SslMode::Disable),
        None,
        None,
        Some(&zstd_compression()),
    ))
    .await
    .expect("compressed postgres store creation should succeed");

    let fetched = compressed
        .get_response("tenant_a", "resp_plain")
        .await
        .expect("get should succeed")
        .expect("plain record should remain readable");
    assert_eq!(
        fetched.response_object, record.response_object,
        "response_object readable"
    );
    assert_eq!(fetched.input, record.input, "input readable");
    assert_eq!(fetched.messages, record.messages, "messages readable");
}

#[tokio::test]
#[ignore]
async fn pg_store_initializes_schema() {
    let store = make_pg_store().await;

    let result = store
        .get_response("tenant_a", "nonexistent")
        .await
        .expect("get should succeed");

    assert!(result.is_none(), "empty store should return None");
}

#[tokio::test]
#[ignore]
async fn pg_upsert_and_get_response() {
    let store = make_pg_store().await;

    let record = make_response_record("resp_1", "tenant_a", 1000);

    store.upsert_response(&record).await.expect("upsert should succeed");

    let fetched = store
        .get_response("tenant_a", "resp_1")
        .await
        .expect("get should succeed")
        .expect("record should exist");

    assert_eq!(fetched.id, "resp_1", "ID should match");
    assert_eq!(fetched.tenant_id, "tenant_a", "tenant should match");
    assert_eq!(fetched.created_at, 1000, "created_at should match");
    assert_eq!(fetched.model, "gpt-4.1", "model should match");
    assert_eq!(
        fetched.response_object,
        json!({"status": "completed"}),
        "response_object should match"
    );
}

#[tokio::test]
#[ignore]
async fn pg_upsert_overwrites_existing_response() {
    let store = make_pg_store().await;

    let record = make_response_record("resp_1", "tenant_a", 1000);
    store
        .upsert_response(&record)
        .await
        .expect("first upsert should succeed");

    let updated = ResponseRecord {
        model: "gpt-4.1-mini".to_owned(),
        response_object: json!({"status": "incomplete"}),
        ..make_response_record("resp_1", "tenant_a", 1000)
    };
    store
        .upsert_response(&updated)
        .await
        .expect("second upsert should succeed");

    let fetched = store
        .get_response("tenant_a", "resp_1")
        .await
        .expect("get should succeed")
        .expect("record should exist");

    assert_eq!(fetched.model, "gpt-4.1-mini", "model should be updated");
    assert_eq!(
        fetched.response_object,
        json!({"status": "incomplete"}),
        "response_object should be updated"
    );
}

#[tokio::test]
#[ignore]
async fn pg_delete_existing_response() {
    let store = make_pg_store().await;

    let record = make_response_record("resp_1", "tenant_a", 1000);
    store.upsert_response(&record).await.expect("upsert should succeed");

    let deleted = store
        .delete_response("tenant_a", "resp_1")
        .await
        .expect("delete should succeed");

    assert!(deleted, "delete should return true for existing record");

    let fetched = store
        .get_response("tenant_a", "resp_1")
        .await
        .expect("get should succeed");

    assert!(fetched.is_none(), "deleted record should not be retrievable");
}

#[tokio::test]
#[ignore]
async fn pg_delete_missing_response_returns_false() {
    let store = make_pg_store().await;

    let deleted = store
        .delete_response("tenant_a", "nonexistent")
        .await
        .expect("delete should succeed");

    assert!(!deleted, "delete should return false for missing record");
}

#[tokio::test]
#[ignore]
async fn pg_tenant_isolation_on_get() {
    let store = make_pg_store().await;

    let record = make_response_record("resp_1", "tenant_a", 1000);
    store.upsert_response(&record).await.expect("upsert should succeed");

    let result = store
        .get_response("tenant_b", "resp_1")
        .await
        .expect("get should succeed");

    assert!(result.is_none(), "tenant_b should not see tenant_a records");
}

#[tokio::test]
#[ignore]
async fn pg_tenant_isolation_on_delete() {
    let store = make_pg_store().await;

    let record = make_response_record("resp_1", "tenant_a", 1000);
    store.upsert_response(&record).await.expect("upsert should succeed");

    let deleted = store
        .delete_response("tenant_b", "resp_1")
        .await
        .expect("delete should succeed");

    assert!(!deleted, "tenant_b should not be able to delete tenant_a records");

    let still_exists = store
        .get_response("tenant_a", "resp_1")
        .await
        .expect("get should succeed");

    assert!(
        still_exists.is_some(),
        "record should still exist after cross-tenant delete attempt"
    );
}

#[tokio::test]
#[ignore]
async fn pg_same_response_id_can_exist_in_multiple_tenants() {
    let store = make_pg_store().await;

    store
        .upsert_response(&make_response_record("resp_shared", "tenant_a", 1000))
        .await
        .expect("tenant_a upsert should succeed");
    store
        .upsert_response(&make_response_record("resp_shared", "tenant_b", 2000))
        .await
        .expect("tenant_b upsert should succeed");

    let tenant_a = store
        .get_response("tenant_a", "resp_shared")
        .await
        .expect("tenant_a get should succeed")
        .expect("tenant_a record should exist");
    let tenant_b = store
        .get_response("tenant_b", "resp_shared")
        .await
        .expect("tenant_b get should succeed")
        .expect("tenant_b record should exist");

    assert_eq!(tenant_a.tenant_id, "tenant_a", "tenant_a record should be isolated");
    assert_eq!(tenant_b.tenant_id, "tenant_b", "tenant_b record should be isolated");
    assert_eq!(tenant_a.created_at, 1000, "tenant_a record should not be overwritten");
    assert_eq!(tenant_b.created_at, 2000, "tenant_b record should not be overwritten");
}

#[tokio::test]
#[ignore]
async fn pg_consume_approval_replay_is_rejected() {
    let store = make_pg_store().await;
    seed_pending(&store, "tenant_a", RESP, &["call_abc"]).await;

    let first = store
        .consume_approvals("tenant_a", RESP, &["call_abc"], 1000)
        .await
        .expect("first consume should succeed");
    let replay = store
        .consume_approvals("tenant_a", RESP, &["call_abc"], 2000)
        .await
        .expect("replay consume should succeed");

    assert!(first.is_none(), "first consumption should be claimed");
    assert_eq!(
        replay,
        Some(0),
        "replayed consumption of the same approval must be rejected (single-use)"
    );
}

#[tokio::test]
#[ignore]
async fn pg_persist_response_with_pending_approvals_writes_both() {
    let store = make_pg_store().await;
    let record = make_response_record("resp_persist", "tenant_a", 1000);
    let approval = make_pending("call_persist");

    store
        .persist_response_with_pending_approvals(&record, std::slice::from_ref(&approval))
        .await
        .expect("atomic persist should succeed");

    let fetched_response = store
        .get_response("tenant_a", "resp_persist")
        .await
        .expect("get should succeed");
    assert!(fetched_response.is_some(), "the response must be written");

    let fetched_approvals = store
        .get_pending_approvals("tenant_a", "resp_persist", &["call_persist"])
        .await
        .expect("get should succeed");
    assert_eq!(
        fetched_approvals,
        vec![approval],
        "the pending approval must be written and scoped to the issuing response"
    );
}

#[tokio::test]
#[ignore]
async fn pg_consume_approval_is_tenant_scoped() {
    let store = make_pg_store().await;
    seed_pending(&store, "tenant_a", RESP, &["call_shared"]).await;
    seed_pending(&store, "tenant_b", RESP, &["call_shared"]).await;

    let tenant_a = store
        .consume_approvals("tenant_a", RESP, &["call_shared"], 1000)
        .await
        .expect("tenant_a consume should succeed");
    let tenant_b = store
        .consume_approvals("tenant_b", RESP, &["call_shared"], 1000)
        .await
        .expect("tenant_b consume should succeed");

    assert!(tenant_a.is_none(), "tenant_a should claim its own approval");
    assert!(
        tenant_b.is_none(),
        "tenant_b sharing an approval id with tenant_a should still claim independently"
    );
}

#[tokio::test]
#[ignore]
async fn pg_consume_approvals_batch_is_all_or_nothing_on_replay() {
    let store = make_pg_store().await;
    seed_pending(&store, "tenant_a", RESP, &["call_1", "call_2"]).await;

    store
        .consume_approvals("tenant_a", RESP, &["call_1"], 1000)
        .await
        .expect("first claim should succeed");

    let conflict = store
        .consume_approvals("tenant_a", RESP, &["call_1", "call_2"], 2000)
        .await
        .expect("batch consume should succeed");
    assert_eq!(conflict, Some(0), "the replayed id's index should be reported");

    let call_2 = store
        .consume_approvals("tenant_a", RESP, &["call_2"], 3000)
        .await
        .expect("call_2 consume should succeed");
    assert!(
        call_2.is_none(),
        "a rolled-back batch must not strand an otherwise-fresh sibling approval"
    );
}

#[tokio::test]
#[ignore]
async fn pg_upsert_and_get_conversation() {
    let store = make_pg_store().await;

    let record = ConversationRecord {
        conversation_id: "conv_1".to_owned(),
        tenant_id: "tenant_a".to_owned(),
        created_at: 1000,
        metadata: json!({}),
        messages: json!([{"role": "user", "content": "Hi"}]),
    };

    store.upsert_conversation(&record).await.expect("upsert should succeed");

    let fetched = ResponseStore::get_conversation(&store, "tenant_a", "conv_1")
        .await
        .expect("get should succeed")
        .expect("record should exist");

    assert_eq!(fetched.conversation_id, "conv_1", "conversation_id should match");
    assert_eq!(
        fetched.messages,
        json!([{"role": "user", "content": "Hi"}]),
        "messages should match"
    );
}

#[tokio::test]
#[ignore]
async fn pg_upsert_conversation_overwrites() {
    let store = make_pg_store().await;

    let record = ConversationRecord {
        conversation_id: "conv_1".to_owned(),
        tenant_id: "tenant_a".to_owned(),
        created_at: 1000,
        metadata: json!({}),
        messages: json!([{"role": "user", "content": "v1"}]),
    };
    store.upsert_conversation(&record).await.expect("upsert should succeed");

    let updated = ConversationRecord {
        conversation_id: "conv_1".to_owned(),
        tenant_id: "tenant_a".to_owned(),
        created_at: 2000,
        metadata: json!({"topic": "updated"}),
        messages: json!([{"role": "user", "content": "v2"}]),
    };
    store
        .upsert_conversation(&updated)
        .await
        .expect("second upsert should succeed");

    let fetched = ConversationItemStore::get_conversation(&store, "tenant_a", "conv_1")
        .await
        .expect("get should succeed")
        .expect("record should exist");

    assert_eq!(
        fetched.messages,
        json!([{"role": "user", "content": "v2"}]),
        "messages should be updated"
    );
    assert_eq!(
        fetched.metadata,
        json!({"topic": "updated"}),
        "metadata should be updated"
    );
    assert_eq!(fetched.created_at, 1000, "created_at should preserve creation time");
}

#[tokio::test]
#[ignore]
async fn pg_update_conversation_messages_preserves_metadata() {
    let store = make_pg_store().await;

    let record = ConversationRecord {
        conversation_id: "conv_1".to_owned(),
        tenant_id: "tenant_a".to_owned(),
        created_at: 1000,
        metadata: json!({"version": "v1"}),
        messages: json!([{"role": "user", "content": "v1"}]),
    };
    store.upsert_conversation(&record).await.expect("upsert should succeed");

    let updated = store
        .update_conversation_messages("tenant_a", "conv_1", &json!([{"role": "assistant", "content": "v2"}]))
        .await
        .expect("message update should succeed");
    assert!(updated, "conversation should be updated");

    let fetched = ConversationItemStore::get_conversation(&store, "tenant_a", "conv_1")
        .await
        .expect("get should succeed")
        .expect("record should exist");

    assert_eq!(
        fetched.metadata,
        json!({"version": "v1"}),
        "metadata should be preserved"
    );
    assert_eq!(
        fetched.messages,
        json!([{"role": "assistant", "content": "v2"}]),
        "messages should be updated"
    );
}

#[tokio::test]
#[ignore]
async fn pg_update_conversation_metadata_preserves_messages() {
    let store = make_pg_store().await;

    let record = ConversationRecord {
        conversation_id: "conv_1".to_owned(),
        tenant_id: "tenant_a".to_owned(),
        created_at: 1000,
        metadata: json!({"version": "v1"}),
        messages: json!([{"role": "user", "content": "keep me"}]),
    };
    store.upsert_conversation(&record).await.expect("upsert should succeed");

    let updated = store
        .update_conversation_metadata("tenant_a", "conv_1", &json!({"version": "v2"}))
        .await
        .expect("metadata update should succeed");
    assert!(updated, "conversation should be updated");

    let fetched = ConversationItemStore::get_conversation(&store, "tenant_a", "conv_1")
        .await
        .expect("get should succeed")
        .expect("record should exist");

    assert_eq!(fetched.metadata, json!({"version": "v2"}), "metadata should be updated");
    assert_eq!(
        fetched.messages,
        json!([{"role": "user", "content": "keep me"}]),
        "messages must be untouched by a metadata-only update"
    );
    assert_eq!(fetched.created_at, 1000, "created_at should be preserved");
}

#[tokio::test]
#[ignore]
async fn pg_delete_existing_conversation() {
    let store = make_pg_store().await;

    let record = ConversationRecord {
        conversation_id: "conv_1".to_owned(),
        tenant_id: "tenant_a".to_owned(),
        created_at: 1000,
        metadata: json!({}),
        messages: json!([]),
    };
    store.upsert_conversation(&record).await.expect("upsert should succeed");

    let deleted = store
        .delete_conversation("tenant_a", "conv_1")
        .await
        .expect("delete should succeed");

    assert!(deleted, "delete should return true for existing conversation");

    let fetched = ConversationItemStore::get_conversation(&store, "tenant_a", "conv_1")
        .await
        .expect("get should succeed");

    assert!(fetched.is_none(), "deleted conversation should not be retrievable");
}

#[tokio::test]
#[ignore]
async fn pg_conversation_tenant_isolation() {
    let store = make_pg_store().await;

    let record = ConversationRecord {
        conversation_id: "conv_1".to_owned(),
        tenant_id: "tenant_a".to_owned(),
        created_at: 1000,
        metadata: json!({}),
        messages: json!([]),
    };
    store.upsert_conversation(&record).await.expect("upsert should succeed");

    let result = ConversationItemStore::get_conversation(&store, "tenant_b", "conv_1")
        .await
        .expect("get should succeed");

    assert!(result.is_none(), "tenant_b should not see tenant_a conversation");
}

// -----------------------------------------------------------------------------
// Conversation Item CRUD (PostgreSQL)
// -----------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn pg_conversation_items_paginate_ascending_and_descending() {
    let store = make_pg_store_with_items().await;
    let items = [
        make_conversation_item("item_1", "tenant_a", "conv_1", 1),
        make_conversation_item("item_2", "tenant_a", "conv_1", 2),
        make_conversation_item("item_3", "tenant_a", "conv_1", 3),
        make_conversation_item("item_4", "tenant_a", "conv_1", 4),
    ];
    store
        .create_conversation_items(&items)
        .await
        .expect("item insert should succeed");

    let asc = store
        .list_conversation_items("tenant_a", "conv_1", None, 2, true)
        .await
        .expect("ascending list should succeed");
    assert_item_ids(&asc, &["item_1", "item_2"]);

    let asc_page2 = store
        .list_conversation_items("tenant_a", "conv_1", Some("item_2"), 2, true)
        .await
        .expect("ascending page 2 should succeed");
    assert_item_ids(&asc_page2, &["item_3", "item_4"]);

    let desc = store
        .list_conversation_items("tenant_a", "conv_1", None, 2, false)
        .await
        .expect("descending list should succeed");
    assert_item_ids(&desc, &["item_4", "item_3"]);

    let desc_page2 = store
        .list_conversation_items("tenant_a", "conv_1", Some("item_3"), 2, false)
        .await
        .expect("descending page 2 should succeed");
    assert_item_ids(&desc_page2, &["item_2", "item_1"]);
}

#[tokio::test]
#[ignore]
async fn pg_duplicate_position_rejected_by_unique_constraint() {
    let store = make_pg_store_with_items().await;
    let first = [make_conversation_item("item_a", "tenant_a", "conv_1", 1)];
    store
        .create_conversation_items(&first)
        .await
        .expect("first insert should succeed");

    let duplicate = [make_conversation_item("item_b", "tenant_a", "conv_1", 1)];
    store
        .create_conversation_items(&duplicate)
        .await
        .expect_err("duplicate position should fail");
}

#[tokio::test]
#[ignore]
async fn pg_conversation_item_single_ops_scope_to_conversation() {
    let store = make_pg_store_with_items().await;
    let item_conv1 = make_conversation_item("item_1", "tenant_a", "conv_1", 1);
    let item_conv2 = make_conversation_item("item_2", "tenant_a", "conv_2", 1);
    store
        .create_conversation_items(&[item_conv1, item_conv2])
        .await
        .expect("item insert should succeed");

    let get_wrong_conv = store
        .get_conversation_item("tenant_a", "conv_2", "item_1")
        .await
        .expect("get should succeed");
    assert!(get_wrong_conv.is_none(), "item_1 should not be visible in conv_2");

    let delete_wrong_conv = store
        .delete_conversation_item("tenant_a", "conv_2", "item_1")
        .await
        .expect("delete should succeed");
    assert!(!delete_wrong_conv, "deleting item_1 from conv_2 should return false");

    let still_exists = store
        .get_conversation_item("tenant_a", "conv_1", "item_1")
        .await
        .expect("get should succeed");
    assert!(still_exists.is_some(), "item_1 should still exist in conv_1");
}

#[tokio::test]
#[ignore]
async fn pg_max_item_position_returns_zero_when_empty() {
    let store = make_pg_store_with_items().await;
    let max = store
        .max_item_position("tenant_a", "conv_1")
        .await
        .expect("max_item_position should succeed");
    assert_eq!(max, 0, "empty conversation should have max position 0");
}

#[tokio::test]
#[ignore]
async fn pg_max_item_position_returns_highest() {
    let store = make_pg_store_with_items().await;
    let items = [
        make_conversation_item("item_1", "tenant_a", "conv_1", 5),
        make_conversation_item("item_2", "tenant_a", "conv_1", 10),
        make_conversation_item("item_3", "tenant_a", "conv_1", 3),
    ];
    store
        .create_conversation_items(&items)
        .await
        .expect("item insert should succeed");

    let max = store
        .max_item_position("tenant_a", "conv_1")
        .await
        .expect("max_item_position should succeed");
    assert_eq!(max, 10, "max position should be 10");
}

#[tokio::test]
#[ignore]
async fn pg_conversation_item_tenant_isolation() {
    let store = make_pg_store_with_items().await;
    let item = make_conversation_item("item_1", "tenant_a", "conv_1", 1);
    store
        .create_conversation_items(&[item])
        .await
        .expect("item insert should succeed");

    let cross_tenant = store
        .get_conversation_item("tenant_b", "conv_1", "item_1")
        .await
        .expect("cross-tenant get should succeed");
    assert!(cross_tenant.is_none(), "tenant_b should not see tenant_a items");

    let cross_tenant_list = store
        .list_conversation_items("tenant_b", "conv_1", None, 100, true)
        .await
        .expect("cross-tenant list should succeed");
    assert!(cross_tenant_list.is_empty(), "tenant_b should see no items");
}

#[tokio::test]
#[ignore]
async fn pg_conversation_item_insert_rejects_existing() {
    let store = make_pg_store_with_items().await;
    let original = make_conversation_item("item_1", "tenant_a", "conv_1", 1);
    let updated = ConversationItemRecord {
        item_data: json!({"type": "message", "role": "assistant", "content": "updated"}),
        created_at: 2000,
        position: 2,
        ..make_conversation_item("item_1", "tenant_a", "conv_1", 1)
    };

    store
        .create_conversation_items(&[original])
        .await
        .expect("initial item insert should succeed");
    store
        .create_conversation_items(&[updated])
        .await
        .expect_err("duplicate item insert should fail");

    let fetched = store
        .get_conversation_item("tenant_a", "conv_1", "item_1")
        .await
        .expect("get should succeed")
        .expect("item should exist after duplicate insert");

    assert_eq!(fetched.position, 1, "duplicate insert should preserve position");
    assert_eq!(fetched.created_at, 1000, "duplicate insert should preserve created_at");
    assert_eq!(
        fetched.item_data,
        json!({"type": "message", "role": "user", "content": "test"}),
        "duplicate insert should preserve item data"
    );
}

#[tokio::test]
#[ignore]
async fn pg_conversation_item_upsert_allows_same_item_id_in_different_conversations() {
    let store = make_pg_store_with_items().await;
    let item_conv1 = ConversationItemRecord {
        item_data: json!({"conversation": "conv_1"}),
        ..make_conversation_item("item_shared", "tenant_a", "conv_1", 1)
    };
    let item_conv2 = ConversationItemRecord {
        item_data: json!({"conversation": "conv_2"}),
        ..make_conversation_item("item_shared", "tenant_a", "conv_2", 1)
    };

    store
        .create_conversation_items(&[item_conv1])
        .await
        .expect("initial item insert should succeed");
    store
        .create_conversation_items(&[item_conv2])
        .await
        .expect("same item_id in another conversation should insert");

    let conv1_item = store
        .get_conversation_item("tenant_a", "conv_1", "item_shared")
        .await
        .expect("conv_1 get should succeed")
        .expect("conv_1 item should still exist");
    let conv2_item = store
        .get_conversation_item("tenant_a", "conv_2", "item_shared")
        .await
        .expect("conv_2 get should succeed")
        .expect("conv_2 item should exist");

    assert_eq!(conv1_item.conversation_id, "conv_1", "conv_1 row should remain scoped");
    assert_eq!(conv2_item.conversation_id, "conv_2", "conv_2 row should be inserted");
    assert_eq!(
        conv1_item.item_data,
        json!({"conversation": "conv_1"}),
        "conv_1 item data should not be overwritten"
    );
    assert_eq!(
        conv2_item.item_data,
        json!({"conversation": "conv_2"}),
        "conv_2 item data should be stored separately"
    );

    let conv1_items = store
        .list_conversation_items("tenant_a", "conv_1", None, 100, true)
        .await
        .expect("conv_1 list should succeed");
    let conv2_items = store
        .list_conversation_items("tenant_a", "conv_2", None, 100, true)
        .await
        .expect("conv_2 list should succeed");
    assert_item_ids(&conv1_items, &["item_shared"]);
    assert_item_ids(&conv2_items, &["item_shared"]);
}

#[tokio::test]
#[ignore]
async fn pg_get_conversation_item_returns_all_fields() {
    let store = make_pg_store_with_items().await;
    let item = ConversationItemRecord {
        item_id: "item_99".to_owned(),
        tenant_id: "tenant_a".to_owned(),
        conversation_id: "conv_1".to_owned(),
        item_data: json!({"type": "function_call", "name": "search"}),
        created_at: 5000,
        position: 42,
    };
    store
        .create_conversation_items(&[item])
        .await
        .expect("item insert should succeed");

    let fetched = store
        .get_conversation_item("tenant_a", "conv_1", "item_99")
        .await
        .expect("get should succeed")
        .expect("item should exist");

    assert_eq!(fetched.item_id, "item_99", "item_id should match");
    assert_eq!(fetched.tenant_id, "tenant_a", "tenant_id should match");
    assert_eq!(fetched.conversation_id, "conv_1", "conversation_id should match");
    assert_eq!(
        fetched.item_data,
        json!({"type": "function_call", "name": "search"}),
        "item_data should round-trip"
    );
    assert_eq!(fetched.created_at, 5000, "created_at should match");
    assert_eq!(fetched.position, 42, "position should match");
}

#[tokio::test]
#[ignore]
async fn pg_list_conversation_items_nonexistent_cursor_returns_empty() {
    let store = make_pg_store_with_items().await;
    let item = make_conversation_item("item_1", "tenant_a", "conv_1", 1);
    store
        .create_conversation_items(&[item])
        .await
        .expect("item insert should succeed");

    let result = store
        .list_conversation_items("tenant_a", "conv_1", Some("nonexistent"), 10, true)
        .await
        .expect("list with nonexistent cursor should succeed");

    assert!(result.is_empty(), "nonexistent cursor item should return empty list");
}

#[tokio::test]
#[ignore]
async fn pg_delete_conversation_preserves_items() {
    let store = make_pg_store_with_items().await;
    let conv = ConversationRecord {
        conversation_id: "conv_1".to_owned(),
        tenant_id: "tenant_a".to_owned(),
        created_at: 1000,
        metadata: json!({}),
        messages: json!([]),
    };
    store
        .upsert_conversation(&conv)
        .await
        .expect("conversation upsert should succeed");

    let items = [
        make_conversation_item("item_1", "tenant_a", "conv_1", 1),
        make_conversation_item("item_2", "tenant_a", "conv_1", 2),
    ];
    store
        .create_conversation_items(&items)
        .await
        .expect("item insert should succeed");

    let deleted = store
        .delete_conversation("tenant_a", "conv_1")
        .await
        .expect("delete_conversation should succeed");
    assert!(deleted, "conversation should have been deleted");

    let remaining = store
        .list_conversation_items("tenant_a", "conv_1", None, 100, true)
        .await
        .expect("list should succeed");
    assert_item_ids(&remaining, &["item_1", "item_2"]);
}

// -----------------------------------------------------------------------------
// create_items_and_sync_messages (PostgreSQL)
// -----------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn pg_create_items_and_sync_messages_assigns_positions() {
    let store = make_pg_store_with_items().await;
    let conv = ConversationRecord {
        conversation_id: "conv_sync".to_owned(),
        tenant_id: "tenant_a".to_owned(),
        created_at: 1000,
        metadata: json!({}),
        messages: json!([]),
    };
    store.upsert_conversation(&conv).await.expect("upsert should succeed");

    let items = [
        make_conversation_item("item_a", "tenant_a", "conv_sync", 0),
        make_conversation_item("item_b", "tenant_a", "conv_sync", 0),
    ];
    store
        .create_items_and_sync_messages("tenant_a", "conv_sync", &items)
        .await
        .expect("create_items_and_sync should succeed");

    let fetched = store
        .list_conversation_items("tenant_a", "conv_sync", None, 100, true)
        .await
        .expect("list should succeed");
    assert_item_ids(&fetched, &["item_a", "item_b"]);
    assert_eq!(fetched[0].position, 1, "first item should get position 1");
    assert_eq!(fetched[1].position, 2, "second item should get position 2");

    let conv_record = ConversationItemStore::get_conversation(&store, "tenant_a", "conv_sync")
        .await
        .expect("get should succeed")
        .expect("conversation should exist");
    let messages = conv_record.messages.as_array().expect("messages should be an array");
    assert_eq!(messages.len(), 2, "messages cache should have 2 items");
}

// -----------------------------------------------------------------------------
// delete_item_and_sync_messages (PostgreSQL)
// -----------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn pg_delete_item_and_sync_messages_updates_cache() {
    let store = make_pg_store_with_items().await;
    let conv = ConversationRecord {
        conversation_id: "conv_del_sync".to_owned(),
        tenant_id: "tenant_a".to_owned(),
        created_at: 1000,
        metadata: json!({}),
        messages: json!([]),
    };
    store.upsert_conversation(&conv).await.expect("upsert should succeed");

    let items = [
        make_conversation_item("item_a", "tenant_a", "conv_del_sync", 0),
        make_conversation_item("item_b", "tenant_a", "conv_del_sync", 0),
    ];
    store
        .create_items_and_sync_messages("tenant_a", "conv_del_sync", &items)
        .await
        .expect("create should succeed");

    let deleted = store
        .delete_item_and_sync_messages("tenant_a", "conv_del_sync", "item_a")
        .await
        .expect("delete should succeed");
    assert!(deleted, "item_a should have been deleted");

    let remaining = store
        .list_conversation_items("tenant_a", "conv_del_sync", None, 100, true)
        .await
        .expect("list should succeed");
    assert_item_ids(&remaining, &["item_b"]);

    let conv_record = ConversationItemStore::get_conversation(&store, "tenant_a", "conv_del_sync")
        .await
        .expect("get should succeed")
        .expect("conversation should exist");
    let messages = conv_record.messages.as_array().expect("messages should be an array");
    assert_eq!(messages.len(), 1, "messages cache should reflect deletion");
}

#[tokio::test]
#[ignore]
async fn pg_create_items_and_sync_messages_missing_conversation_errors() {
    let store = make_pg_store_with_items().await;
    // No conversation row exists — mirrors a conversation deleted between the
    // handler's existence check and this transaction.
    let items = [make_conversation_item("item_a", "tenant_a", "conv_missing", 0)];
    let err = store
        .create_items_and_sync_messages("tenant_a", "conv_missing", &items)
        .await
        .expect_err("create against a missing conversation should error");
    assert!(
        matches!(&err, StoreError::Database(msg) if msg.contains("conversation disappeared during message sync")),
        "unexpected error: {err:?}"
    );

    // The transaction must roll back — no orphaned items may persist.
    let fetched = store
        .list_conversation_items("tenant_a", "conv_missing", None, 100, true)
        .await
        .expect("list should succeed");
    assert!(fetched.is_empty(), "items must not persist when the message sync fails");
}

#[tokio::test]
#[ignore]
async fn pg_delete_item_and_sync_messages_missing_conversation_errors() {
    let store = make_pg_store_with_items().await;
    let conv = ConversationRecord {
        conversation_id: "conv_del_missing".to_owned(),
        tenant_id: "tenant_a".to_owned(),
        created_at: 1000,
        metadata: json!({}),
        messages: json!([]),
    };
    store.upsert_conversation(&conv).await.expect("upsert should succeed");

    let items = [make_conversation_item("item_a", "tenant_a", "conv_del_missing", 0)];
    store
        .create_items_and_sync_messages("tenant_a", "conv_del_missing", &items)
        .await
        .expect("create should succeed");

    // Delete the conversation row; items intentionally survive (no FK).
    ConversationItemStore::delete_conversation(&store, "tenant_a", "conv_del_missing")
        .await
        .expect("delete conversation should succeed");

    let err = store
        .delete_item_and_sync_messages("tenant_a", "conv_del_missing", "item_a")
        .await
        .expect_err("delete-item sync against a missing conversation should error");
    assert!(
        matches!(&err, StoreError::Database(msg) if msg.contains("conversation disappeared during message sync")),
        "unexpected error: {err:?}"
    );

    // The transaction must roll back — the item deletion must not persist.
    let remaining = store
        .list_conversation_items("tenant_a", "conv_del_missing", None, 100, true)
        .await
        .expect("list should succeed");
    assert_item_ids(&remaining, &["item_a"]);
}

// -----------------------------------------------------------------------------
// Compression (SQLite)
// -----------------------------------------------------------------------------

fn zstd_compression() -> StoreCompressionConfig {
    StoreCompressionConfig {
        algorithm: CompressionAlgorithm::Zstd,
        level: Some(3),
    }
}

async fn make_compressed_store() -> SqliteResponseStore {
    SqliteResponseStore::new(
        "sqlite::memory:",
        "test_responses",
        "test_conversation_messages",
        Some("test_conversation_items"),
        None,
        Some(&zstd_compression()),
    )
    .await
    .expect("compressed store creation should succeed")
}

#[tokio::test]
async fn compressed_store_roundtrips_response() {
    let store = make_compressed_store().await;
    let record = make_response_record("resp_zstd", "tenant_a", 1000);
    store.upsert_response(&record).await.expect("upsert should succeed");

    let fetched = store
        .get_response("tenant_a", "resp_zstd")
        .await
        .expect("get should succeed")
        .expect("record should exist");

    assert_eq!(
        fetched.response_object, record.response_object,
        "response_object round-trips"
    );
    assert_eq!(fetched.input, record.input, "input round-trips");
    assert_eq!(fetched.messages, record.messages, "messages round-trip");
}

#[tokio::test]
async fn compression_is_backward_compatible_with_plain_rows() {
    let dir = tempfile::tempdir().expect("tempdir should be created");
    let db_path = dir.path().join("backcompat.db");
    let url = format!("sqlite://{}?mode=rwc", db_path.display());

    let plain = SqliteResponseStore::new(&url, "bc_responses", "bc_conversations", None, None, None)
        .await
        .expect("plain store creation should succeed");
    let record = make_response_record("resp_plain", "tenant_a", 1000);
    plain.upsert_response(&record).await.expect("upsert should succeed");
    drop(plain);

    let compressed = SqliteResponseStore::new(
        &url,
        "bc_responses",
        "bc_conversations",
        None,
        None,
        Some(&zstd_compression()),
    )
    .await
    .expect("compressed store creation should succeed");

    let fetched = compressed
        .get_response("tenant_a", "resp_plain")
        .await
        .expect("get should succeed")
        .expect("plain record should remain readable");
    assert_eq!(
        fetched.response_object, record.response_object,
        "response_object readable"
    );
    assert_eq!(fetched.input, record.input, "input readable");
    assert_eq!(fetched.messages, record.messages, "messages readable");
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

async fn make_store() -> SqliteResponseStore {
    SqliteResponseStore::new(
        "sqlite::memory:",
        "test_responses",
        "test_conversation_messages",
        None,
        None,
        None,
    )
    .await
    .expect("store creation should succeed")
}

async fn make_file_store(dir: &tempfile::TempDir, items_table: Option<&str>) -> SqliteResponseStore {
    let db_path = dir.path().join("concurrent.db");
    let url = format!("sqlite://{}?mode=rwc", db_path.display());
    SqliteResponseStore::new(
        &url,
        "test_responses",
        "test_conversation_messages",
        items_table,
        None,
        None,
    )
    .await
    .expect("file-backed store creation should succeed")
}

async fn make_store_with_items() -> SqliteResponseStore {
    SqliteResponseStore::new(
        "sqlite::memory:",
        "test_responses",
        "test_conversation_messages",
        Some("test_conversation_items"),
        None,
        None,
    )
    .await
    .expect("store creation should succeed")
}

async fn make_pg_store_with_items() -> PostgresResponseStore {
    let url = pg_database_url();
    let suffix = pg_unique_suffix();
    let responses_table = format!("test_responses_{suffix}");
    let conversations_table = format!("test_conversations_{suffix}");
    let items_table = format!("test_conversation_items_{suffix}");
    PostgresResponseStore::new(
        &url,
        &responses_table,
        &conversations_table,
        Some(&items_table),
        Some(SslMode::Disable),
        None,
        None,
        None,
    )
    .await
    .expect("postgres store creation should succeed")
}

fn make_conversation_item(
    item_id: &str,
    tenant_id: &str,
    conversation_id: &str,
    position: i64,
) -> ConversationItemRecord {
    ConversationItemRecord {
        item_id: item_id.to_owned(),
        tenant_id: tenant_id.to_owned(),
        conversation_id: conversation_id.to_owned(),
        item_data: json!({"type": "message", "role": "user", "content": "test"}),
        created_at: 1000,
        position,
    }
}

fn assert_item_ids(items: &[ConversationItemRecord], expected: &[&str]) {
    let ids: Vec<&str> = items.iter().map(|i| i.item_id.as_str()).collect();
    assert_eq!(ids, expected, "item IDs should match expected order");
}

/// Build a pending-approval record for seeding the approvals table before a
/// consume test. A single fixed target is fine; consumption keys on
/// `(tenant_id, response_id, approval_id)`, and the response scope is supplied
/// separately by [`seed_pending`].
fn make_pending(approval_id: &str) -> PendingApprovalRecord {
    PendingApprovalRecord {
        approval_id: approval_id.to_owned(),
        server_label: "weather".to_owned(),
        tool_name: "get_weather".to_owned(),
        arguments: r#"{"location":"SF"}"#.to_owned(),
        target_fingerprint: "fp-test".to_owned(),
    }
}

/// Seed `approval_ids` as outstanding (unconsumed) pending rows issued by
/// `response_id` so a following [`ResponseStore::consume_approvals`] scoped to
/// the same response has real rows to claim. Takes `&dyn ResponseStore` so both
/// the SQLite and Postgres suites share it.
async fn seed_pending(store: &dyn ResponseStore, tenant_id: &str, response_id: &str, approval_ids: &[&str]) {
    let records: Vec<PendingApprovalRecord> = approval_ids.iter().map(|id| make_pending(id)).collect();
    store
        .record_pending_approvals(tenant_id, response_id, &records, 1000)
        .await
        .expect("seeding pending approvals should succeed");
}

fn make_response_record(id: &str, tenant_id: &str, created_at: i64) -> ResponseRecord {
    ResponseRecord {
        id: id.to_owned(),
        tenant_id: tenant_id.to_owned(),
        created_at,
        model: "gpt-4.1".to_owned(),
        response_object: json!({"status": "completed"}),
        input: json!("test input"),
        messages: json!([{"role": "user", "content": "hello"}]),
    }
}
