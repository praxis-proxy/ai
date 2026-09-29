// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Service-layer unit tests: item-record assembly and conversation/item CRUD
//! exercised against the in-memory backend, with no request pipeline and no SQL
//! database.

use std::sync::Arc;

use praxis_ai_store::{
    ConversationItemRecord, ConversationRecord, PersistedStateBackend, StoreError, StoreRegistry, memory::InMemoryStore,
};
use serde_json::json;

use super::{ConversationsService, build_item_records, duplicate_item_id, validate_item_count};
use crate::StateOwner;

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
fn service(reg: &StoreRegistry, owner: &StateOwner) -> ConversationsService {
    ConversationsService::new(reg.get_scoped("default", owner).unwrap())
}

/// A conversation record owned by `owner`.
fn conversation(owner: &StateOwner, id: &str) -> ConversationRecord {
    ConversationRecord {
        conversation_id: id.to_owned(),
        owner: owner.clone(),
        created_at: 1_719_900_000,
        metadata: json!({"topic": "demo"}),
        messages: json!([]),
    }
}

/// A sequential item-id generator for record assembly in tests.
fn counter() -> impl FnMut() -> String {
    let mut n = 0;
    move || {
        n += 1;
        format!("item_{n}")
    }
}

// -----------------------------------------------------------------------------
// Item-record assembly (transport-neutral, no pipeline)
// -----------------------------------------------------------------------------

#[test]
fn build_item_records_generates_ids_for_items_without_them() {
    let o = owner("alice");
    let records = build_item_records(
        &o,
        "conv_1",
        1000,
        0,
        [json!({"type": "message", "role": "user", "content": "hi"})],
        counter(),
    )
    .unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].item_id, "item_1");
    assert_eq!(records[0].conversation_id, "conv_1");
    assert_eq!(records[0].owner, o);
    assert_eq!(records[0].position, 0);
    assert_eq!(records[0].item_data["id"], "item_1");
}

#[test]
fn build_item_records_preserves_explicit_ids_and_positions() {
    let o = owner("alice");
    let records = build_item_records(
        &o,
        "conv_1",
        1000,
        5,
        [
            json!({"id": "given_a", "type": "message", "role": "user", "content": "one"}),
            json!({"id": "given_b", "type": "message", "role": "user", "content": "two"}),
        ],
        counter(),
    )
    .unwrap();
    assert_eq!(records[0].item_id, "given_a");
    assert_eq!(records[0].position, 5);
    assert_eq!(records[1].item_id, "given_b");
    assert_eq!(records[1].position, 6);
}

#[test]
fn build_item_records_normalizes_assistant_message_content() {
    let o = owner("alice");
    let records = build_item_records(
        &o,
        "conv_1",
        1000,
        0,
        [json!({"type": "message", "role": "assistant", "content": "done"})],
        counter(),
    )
    .unwrap();
    let content = &records[0].item_data["content"][0];
    assert_eq!(content["type"], "output_text");
    assert_eq!(content["text"], "done");
    assert_eq!(content["annotations"], json!([]));
    assert_eq!(content["logprobs"], json!([]));
}

#[test]
fn build_item_records_rejects_non_object_item() {
    let o = owner("alice");
    let err = build_item_records(&o, "conv_1", 1000, 0, [json!("not an object")], counter()).unwrap_err();
    assert!(matches!(err, StoreError::InvalidInput(_)));
}

#[test]
fn validate_item_count_and_duplicate_detection() {
    validate_item_count(20).unwrap();
    assert!(matches!(validate_item_count(21), Err(StoreError::InvalidInput(_))));

    let o = owner("alice");
    let records = build_item_records(
        &o,
        "conv_1",
        1000,
        0,
        [
            json!({"id": "dup", "type": "message", "role": "user", "content": "a"}),
            json!({"id": "dup", "type": "message", "role": "user", "content": "b"}),
        ],
        counter(),
    )
    .unwrap();
    assert_eq!(duplicate_item_id(&records), Some("dup"));
}

// -----------------------------------------------------------------------------
// CRUD round-trips against the in-memory backend
// -----------------------------------------------------------------------------

#[tokio::test]
async fn create_and_get_conversation_round_trip() {
    let reg = registry();
    let o = owner("alice");
    let svc = service(&reg, &o);

    svc.upsert_conversation(&conversation(&o, "conv_1")).await.unwrap();
    let fetched = svc.get_conversation("conv_1").await.unwrap().unwrap();
    assert_eq!(fetched.conversation_id, "conv_1");
    assert_eq!(fetched.owner, o);
}

#[tokio::test]
async fn update_metadata_and_delete() {
    let reg = registry();
    let o = owner("alice");
    let svc = service(&reg, &o);

    svc.upsert_conversation(&conversation(&o, "conv_1")).await.unwrap();
    assert!(
        svc.update_conversation_metadata("conv_1", &json!({"topic": "new"}))
            .await
            .unwrap()
    );
    assert_eq!(
        svc.get_conversation("conv_1").await.unwrap().unwrap().metadata,
        json!({"topic": "new"})
    );
    assert!(svc.delete_conversation("conv_1").await.unwrap());
    assert!(svc.get_conversation("conv_1").await.unwrap().is_none());
}

#[tokio::test]
async fn create_items_list_get_delete_round_trip() {
    let reg = registry();
    let o = owner("alice");
    let svc = service(&reg, &o);

    svc.upsert_conversation(&conversation(&o, "conv_1")).await.unwrap();
    let records = build_item_records(
        &o,
        "conv_1",
        1000,
        0,
        [json!({"id": "item_a", "type": "message", "role": "user", "content": "hi"})],
        counter(),
    )
    .unwrap();

    assert!(svc.existing_item_ids("conv_1", &["item_a"]).await.unwrap().is_empty());
    svc.create_items("conv_1", &records).await.unwrap();
    assert_eq!(
        svc.existing_item_ids("conv_1", &["item_a"]).await.unwrap(),
        vec!["item_a".to_owned()]
    );

    let listed = svc.list_items("conv_1", None, 10, true).await.unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].item_id, "item_a");

    assert!(svc.get_item("conv_1", "item_a").await.unwrap().is_some());
    assert!(svc.delete_item("conv_1", "item_a").await.unwrap());
    assert!(svc.get_item("conv_1", "item_a").await.unwrap().is_none());
}

// -----------------------------------------------------------------------------
// Owner scoping and forgery rejection
// -----------------------------------------------------------------------------

#[tokio::test]
async fn get_conversation_is_owner_scoped() {
    let reg = registry();
    let alice = owner("alice");
    let bob = owner("bob");
    service(&reg, &alice)
        .upsert_conversation(&conversation(&alice, "conv_1"))
        .await
        .unwrap();

    assert!(service(&reg, &bob).get_conversation("conv_1").await.unwrap().is_none());
    assert!(
        service(&reg, &alice)
            .get_conversation("conv_1")
            .await
            .unwrap()
            .is_some()
    );
}

#[tokio::test]
async fn upsert_conversation_rejects_foreign_owner_record() {
    let reg = registry();
    let alice = owner("alice");
    let bob = owner("bob");
    let svc = service(&reg, &alice);

    // A record stamped with bob's owner cannot be written through alice's handle.
    let err = svc
        .upsert_conversation(&conversation(&bob, "conv_1"))
        .await
        .unwrap_err();
    assert!(matches!(err, StoreError::InvalidInput(_)));

    // Nothing was written under either owner.
    assert!(
        service(&reg, &alice)
            .get_conversation("conv_1")
            .await
            .unwrap()
            .is_none()
    );
    assert!(service(&reg, &bob).get_conversation("conv_1").await.unwrap().is_none());
}

#[tokio::test]
async fn create_items_rejects_foreign_owner_item() {
    let reg = registry();
    let alice = owner("alice");
    let bob = owner("bob");
    let svc = service(&reg, &alice);
    svc.upsert_conversation(&conversation(&alice, "conv_1")).await.unwrap();

    let foreign: Vec<ConversationItemRecord> = build_item_records(
        &bob,
        "conv_1",
        1000,
        0,
        [json!({"id": "item_x", "type": "message", "role": "user", "content": "x"})],
        counter(),
    )
    .unwrap();
    let err = svc.create_items("conv_1", &foreign).await.unwrap_err();
    assert!(matches!(err, StoreError::InvalidInput(_)));
    assert!(svc.list_items("conv_1", None, 10, true).await.unwrap().is_empty());
}

#[tokio::test]
async fn items_are_isolated_across_owners() {
    let reg = registry();
    let alice = owner("alice");
    let bob = owner("bob");
    let alice_svc = service(&reg, &alice);
    alice_svc
        .upsert_conversation(&conversation(&alice, "conv_1"))
        .await
        .unwrap();
    let records = build_item_records(
        &alice,
        "conv_1",
        1000,
        0,
        [json!({"id": "item_a", "type": "message", "role": "user", "content": "hi"})],
        counter(),
    )
    .unwrap();
    alice_svc.create_items("conv_1", &records).await.unwrap();

    let bob_svc = service(&reg, &bob);
    assert!(bob_svc.get_conversation("conv_1").await.unwrap().is_none());
    assert!(bob_svc.list_items("conv_1", None, 10, true).await.unwrap().is_empty());
    assert!(bob_svc.get_item("conv_1", "item_a").await.unwrap().is_none());
}
