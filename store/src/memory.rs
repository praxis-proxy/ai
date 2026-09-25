// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Deterministic in-memory backend implementing both persistence traits.
//!
//! Used for service unit tests and as the reference the SQL backends
//! contract-test against. A single mutex guards all state, so the multi-step
//! trait methods that SQL backends run in a transaction (approval consumption,
//! compare-and-swap, create/delete-items-and-sync) are atomic here too.

#![expect(
    clippy::significant_drop_tightening,
    reason = "each method is one short locked critical section; holding the guard is what keeps the multi-step ops atomic"
)]

use std::{
    collections::{HashMap, HashSet},
    sync::Mutex,
};

use async_trait::async_trait;

use crate::{
    owner::StateOwner,
    traits::{ConversationItemStore, ResponseStore},
    types::{ConversationItemRecord, ConversationRecord, PendingApprovalRecord, ResponseRecord, StoreError},
};

/// Owner-qualified key for owner-scoped rows.
type OwnerKey = (StateOwner, String);

/// A stored pending approval plus its single-use consumption stamp.
#[derive(Clone)]
struct StoredApproval {
    /// The persisted approval record.
    record: PendingApprovalRecord,
    /// Epoch-ms consumption time; `None` while outstanding.
    consumed_at: Option<i64>,
}

/// All in-memory state behind one mutex.
#[derive(Default)]
struct Inner {
    /// `response_id` -> response record, carrying its owner. Globally keyed
    /// like the SQL `PRIMARY KEY (id)`, so a colliding id owned by another
    /// principal is rejected rather than shadowed.
    responses: HashMap<String, ResponseRecord>,
    /// `(owner, conversation_id)` -> conversation record.
    conversations: HashMap<OwnerKey, ConversationRecord>,
    /// `(owner, conversation_id)` -> its items (unordered; sorted on read).
    items: HashMap<OwnerKey, Vec<ConversationItemRecord>>,
    /// Globally unique item ids, enforcing cross-owner item-id uniqueness.
    item_ids: HashSet<String>,
    /// `(owner, response_id, approval_id)` -> approval + consumption stamp.
    approvals: HashMap<(StateOwner, String, String), StoredApproval>,
}

/// In-memory implementation of [`ResponseStore`] and [`ConversationItemStore`].
#[derive(Default)]
pub struct InMemoryStore {
    /// Guarded state.
    inner: Mutex<Inner>,
}

impl InMemoryStore {
    /// Create an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Lock the state, mapping poisoning to an unavailable error.
    fn lock(&self) -> Result<std::sync::MutexGuard<'_, Inner>, StoreError> {
        self.inner
            .lock()
            .map_err(|err| StoreError::Unavailable(format!("in-memory store mutex poisoned: {err}")))
    }
}

/// Rebuild a conversation's denormalized message cache from its items in order.
fn rebuild_messages(items: &[ConversationItemRecord]) -> serde_json::Value {
    let mut sorted: Vec<&ConversationItemRecord> = items.iter().collect();
    sorted.sort_by(|a, b| a.position.cmp(&b.position).then_with(|| a.item_id.cmp(&b.item_id)));
    serde_json::Value::Array(sorted.into_iter().map(|item| item.item_data.clone()).collect())
}

/// Upsert a response, rejecting an id already owned by another principal.
///
/// Mirrors the SQL owner-guarded `ON CONFLICT (id)` upsert: a same-id write from
/// a different owner fails instead of overwriting the existing row.
fn upsert_response_into(inner: &mut Inner, record: &ResponseRecord) -> Result<(), StoreError> {
    if let Some(existing) = inner.responses.get(&record.id)
        && existing.owner != record.owner
    {
        return Err(StoreError::Database("response id collision".to_owned()));
    }
    inner.responses.insert(record.id.clone(), record.clone());
    Ok(())
}

/// Reject a batch that reuses an item id, within itself or against stored rows.
///
/// Mirrors the SQL item `PRIMARY KEY (item_id)`, which is globally unique.
fn reject_duplicate_item_ids(inner: &Inner, items: &[ConversationItemRecord]) -> Result<(), StoreError> {
    let mut seen: HashSet<&str> = HashSet::with_capacity(items.len());
    for item in items {
        if !seen.insert(item.item_id.as_str()) || inner.item_ids.contains(&item.item_id) {
            return Err(StoreError::InvalidInput(format!(
                "conversation item '{}' already exists",
                item.item_id
            )));
        }
    }
    Ok(())
}

/// Reject items that leave the authorized scope or whose parent is absent.
///
/// Mirrors the SQL `require_matching_item_scope` guard plus the rebuild update
/// that matches no row when the conversation is gone.
fn require_conversation_scope(
    inner: &Inner,
    owner: &StateOwner,
    conversation_id: &str,
    items: &[ConversationItemRecord],
) -> Result<(), StoreError> {
    if !items
        .iter()
        .all(|item| &item.owner == owner && item.conversation_id == conversation_id)
    {
        return Err(StoreError::InvalidInput(
            "conversation item scope does not match its parent".to_owned(),
        ));
    }
    if !inner
        .conversations
        .contains_key(&(owner.clone(), conversation_id.to_owned()))
    {
        return Err(StoreError::Database(format!(
            "conversation disappeared during message sync: {conversation_id}"
        )));
    }
    Ok(())
}

/// Record approvals insert-if-absent, so a re-emit never resets a consumed row.
fn record_approvals_into(inner: &mut Inner, owner: &StateOwner, response_id: &str, records: &[PendingApprovalRecord]) {
    for record in records {
        let key = (owner.clone(), response_id.to_owned(), record.approval_id.clone());
        inner.approvals.entry(key).or_insert_with(|| StoredApproval {
            record: record.clone(),
            consumed_at: None,
        });
    }
}

#[async_trait]
impl ResponseStore for InMemoryStore {
    async fn upsert_response(&self, record: &ResponseRecord) -> Result<(), StoreError> {
        let mut inner = self.lock()?;
        upsert_response_into(&mut inner, record)
    }

    async fn get_response(&self, owner: &StateOwner, id: &str) -> Result<Option<ResponseRecord>, StoreError> {
        let inner = self.lock()?;
        Ok(inner.responses.get(id).filter(|record| &record.owner == owner).cloned())
    }

    async fn delete_response(&self, owner: &StateOwner, id: &str) -> Result<bool, StoreError> {
        let mut inner = self.lock()?;
        let removed = inner.responses.get(id).is_some_and(|record| &record.owner == owner);
        if removed {
            inner.responses.remove(id);
        }
        // Deleting a response removes the pending approvals it issued, so no
        // consumable approval (and no tool arguments) survive it.
        inner
            .approvals
            .retain(|(row_owner, response_id, _), _| !(row_owner == owner && response_id == id));
        Ok(removed)
    }

    async fn get_conversation(
        &self,
        owner: &StateOwner,
        conversation_id: &str,
    ) -> Result<Option<ConversationRecord>, StoreError> {
        let inner = self.lock()?;
        Ok(inner
            .conversations
            .get(&(owner.clone(), conversation_id.to_owned()))
            .cloned())
    }

    async fn record_pending_approvals(
        &self,
        owner: &StateOwner,
        response_id: &str,
        records: &[PendingApprovalRecord],
        _created_at: i64,
    ) -> Result<(), StoreError> {
        let mut inner = self.lock()?;
        record_approvals_into(&mut inner, owner, response_id, records);
        Ok(())
    }

    async fn persist_response_with_pending_approvals(
        &self,
        record: &ResponseRecord,
        pending_approvals: &[PendingApprovalRecord],
    ) -> Result<(), StoreError> {
        // One lock spans the upsert and the approval writes so a concurrent
        // delete cannot interleave and orphan an approval. The default
        // sequential impl locks twice and leaves that window open.
        let mut inner = self.lock()?;
        upsert_response_into(&mut inner, record)?;
        record_approvals_into(&mut inner, &record.owner, &record.id, pending_approvals);
        Ok(())
    }

    async fn get_pending_approvals(
        &self,
        owner: &StateOwner,
        response_id: &str,
        approval_ids: &[&str],
    ) -> Result<Vec<PendingApprovalRecord>, StoreError> {
        let inner = self.lock()?;
        let found = approval_ids
            .iter()
            .filter_map(|approval_id| {
                inner
                    .approvals
                    .get(&(owner.clone(), response_id.to_owned(), (*approval_id).to_owned()))
                    .map(|stored| stored.record.clone())
            })
            .collect();
        Ok(found)
    }

    async fn consume_approvals(
        &self,
        owner: &StateOwner,
        response_id: &str,
        approval_ids: &[&str],
        consumed_at: i64,
    ) -> Result<Option<usize>, StoreError> {
        let mut inner = self.lock()?;
        // Validate the whole batch before mutating anything (all-or-nothing):
        // a missing row, an already-consumed row, or a duplicate earlier in the
        // batch aborts the claim and leaves every row untouched.
        let mut seen: HashSet<&str> = HashSet::with_capacity(approval_ids.len());
        for (index, approval_id) in approval_ids.iter().enumerate() {
            if !seen.insert(approval_id) {
                return Ok(Some(index));
            }
            let key = (owner.clone(), response_id.to_owned(), (*approval_id).to_owned());
            match inner.approvals.get(&key) {
                Some(stored) if stored.consumed_at.is_none() => {},
                _ => return Ok(Some(index)),
            }
        }
        for approval_id in approval_ids {
            let key = (owner.clone(), response_id.to_owned(), (*approval_id).to_owned());
            if let Some(stored) = inner.approvals.get_mut(&key) {
                stored.consumed_at = Some(consumed_at);
            }
        }
        Ok(None)
    }
}

#[async_trait]
impl ConversationItemStore for InMemoryStore {
    async fn upsert_conversation(&self, record: &ConversationRecord) -> Result<(), StoreError> {
        let mut inner = self.lock()?;
        let key = (record.owner.clone(), record.conversation_id.clone());
        // Preserve the original creation time on update; refreshes of metadata
        // or messages must not rewrite created_at.
        let created_at = inner
            .conversations
            .get(&key)
            .map_or(record.created_at, |existing| existing.created_at);
        let mut stored = record.clone();
        stored.created_at = created_at;
        inner.conversations.insert(key, stored);
        Ok(())
    }

    async fn update_conversation_messages(
        &self,
        owner: &StateOwner,
        conversation_id: &str,
        messages: &serde_json::Value,
    ) -> Result<bool, StoreError> {
        let mut inner = self.lock()?;
        match inner
            .conversations
            .get_mut(&(owner.clone(), conversation_id.to_owned()))
        {
            Some(record) => {
                record.messages = messages.clone();
                Ok(true)
            },
            None => Ok(false),
        }
    }

    async fn update_conversation_metadata(
        &self,
        owner: &StateOwner,
        conversation_id: &str,
        metadata: &serde_json::Value,
    ) -> Result<bool, StoreError> {
        let mut inner = self.lock()?;
        match inner
            .conversations
            .get_mut(&(owner.clone(), conversation_id.to_owned()))
        {
            Some(record) => {
                record.metadata = metadata.clone();
                Ok(true)
            },
            None => Ok(false),
        }
    }

    async fn compare_and_swap_conversation_messages(
        &self,
        owner: &StateOwner,
        conversation_id: &str,
        expected_messages: &serde_json::Value,
        messages: &serde_json::Value,
    ) -> Result<bool, StoreError> {
        let mut inner = self.lock()?;
        match inner
            .conversations
            .get_mut(&(owner.clone(), conversation_id.to_owned()))
        {
            Some(record) if &record.messages == expected_messages => {
                record.messages = messages.clone();
                Ok(true)
            },
            _ => Ok(false),
        }
    }

    async fn get_conversation(
        &self,
        owner: &StateOwner,
        conversation_id: &str,
    ) -> Result<Option<ConversationRecord>, StoreError> {
        let inner = self.lock()?;
        Ok(inner
            .conversations
            .get(&(owner.clone(), conversation_id.to_owned()))
            .cloned())
    }

    async fn delete_conversation(&self, owner: &StateOwner, conversation_id: &str) -> Result<bool, StoreError> {
        let mut inner = self.lock()?;
        // Matches the OpenAI API: deleting a conversation leaves item rows.
        Ok(inner
            .conversations
            .remove(&(owner.clone(), conversation_id.to_owned()))
            .is_some())
    }

    async fn create_conversation_items(&self, items: &[ConversationItemRecord]) -> Result<(), StoreError> {
        let mut inner = self.lock()?;
        // Validate the whole batch before mutating (all-or-nothing). The parent
        // conversation must exist under the item's own owner, rejecting an
        // orphan and a cross-owner parent alike.
        for item in items {
            if !inner
                .conversations
                .contains_key(&(item.owner.clone(), item.conversation_id.clone()))
            {
                return Err(StoreError::InvalidInput(
                    "conversation item scope does not match its parent".to_owned(),
                ));
            }
        }
        reject_duplicate_item_ids(&inner, items)?;
        for item in items {
            inner.item_ids.insert(item.item_id.clone());
            inner
                .items
                .entry((item.owner.clone(), item.conversation_id.clone()))
                .or_default()
                .push(item.clone());
        }
        Ok(())
    }

    async fn list_conversation_items(
        &self,
        owner: &StateOwner,
        conversation_id: &str,
        after_item_id: Option<&str>,
        limit: u32,
        ascending: bool,
    ) -> Result<Vec<ConversationItemRecord>, StoreError> {
        let inner = self.lock()?;
        let mut ordered: Vec<ConversationItemRecord> = inner
            .items
            .get(&(owner.clone(), conversation_id.to_owned()))
            .cloned()
            .unwrap_or_default();
        ordered.sort_by(|a, b| a.position.cmp(&b.position).then_with(|| a.item_id.cmp(&b.item_id)));
        if !ascending {
            ordered.reverse();
        }
        let start = match after_item_id {
            Some(cursor) => match ordered.iter().position(|item| item.item_id == cursor) {
                Some(index) => index + 1,
                None => return Ok(Vec::new()),
            },
            None => 0,
        };
        Ok(ordered.into_iter().skip(start).take(limit as usize).collect())
    }

    async fn get_existing_conversation_item_ids(
        &self,
        owner: &StateOwner,
        conversation_id: &str,
        item_ids: &[&str],
    ) -> Result<Vec<String>, StoreError> {
        let inner = self.lock()?;
        let Some(items) = inner.items.get(&(owner.clone(), conversation_id.to_owned())) else {
            return Ok(Vec::new());
        };
        let present: HashSet<&str> = items.iter().map(|item| item.item_id.as_str()).collect();
        Ok(item_ids
            .iter()
            .filter(|id| present.contains(**id))
            .map(|id| (*id).to_owned())
            .collect())
    }

    async fn get_conversation_item(
        &self,
        owner: &StateOwner,
        conversation_id: &str,
        item_id: &str,
    ) -> Result<Option<ConversationItemRecord>, StoreError> {
        let inner = self.lock()?;
        Ok(inner
            .items
            .get(&(owner.clone(), conversation_id.to_owned()))
            .and_then(|items| items.iter().find(|item| item.item_id == item_id).cloned()))
    }

    async fn delete_conversation_item(
        &self,
        owner: &StateOwner,
        conversation_id: &str,
        item_id: &str,
    ) -> Result<bool, StoreError> {
        let mut inner = self.lock()?;
        let Some(items) = inner.items.get_mut(&(owner.clone(), conversation_id.to_owned())) else {
            return Ok(false);
        };
        let Some(index) = items.iter().position(|item| item.item_id == item_id) else {
            return Ok(false);
        };
        items.remove(index);
        inner.item_ids.remove(item_id);
        Ok(true)
    }

    async fn conversation_item_position(
        &self,
        owner: &StateOwner,
        conversation_id: &str,
        item_id: &str,
    ) -> Result<Option<i64>, StoreError> {
        let inner = self.lock()?;
        Ok(inner
            .items
            .get(&(owner.clone(), conversation_id.to_owned()))
            .and_then(|items| {
                items
                    .iter()
                    .find(|item| item.item_id == item_id)
                    .map(|item| item.position)
            }))
    }

    async fn max_item_position(&self, owner: &StateOwner, conversation_id: &str) -> Result<i64, StoreError> {
        let inner = self.lock()?;
        Ok(inner
            .items
            .get(&(owner.clone(), conversation_id.to_owned()))
            .and_then(|items| items.iter().map(|item| item.position).max())
            .unwrap_or(0))
    }

    async fn create_items_and_sync_messages(
        &self,
        owner: &StateOwner,
        conversation_id: &str,
        items: &[ConversationItemRecord],
    ) -> Result<(), StoreError> {
        let mut inner = self.lock()?;
        let key = (owner.clone(), conversation_id.to_owned());
        require_conversation_scope(&inner, owner, conversation_id, items)?;
        reject_duplicate_item_ids(&inner, items)?;
        let mut next = inner
            .items
            .get(&key)
            .and_then(|items| items.iter().map(|i| i.position).max())
            .unwrap_or(0);
        for item in items {
            next += 1;
            // Positions are assigned within the "transaction"; the input
            // position field is ignored.
            let mut stored = item.clone();
            stored.position = next;
            inner.item_ids.insert(stored.item_id.clone());
            inner.items.entry(key.clone()).or_default().push(stored);
        }
        let rebuilt = rebuild_messages(inner.items.get(&key).map_or(&[][..], |items| items.as_slice()));
        if let Some(conversation) = inner.conversations.get_mut(&key) {
            conversation.messages = rebuilt;
        }
        Ok(())
    }

    async fn delete_item_and_sync_messages(
        &self,
        owner: &StateOwner,
        conversation_id: &str,
        item_id: &str,
    ) -> Result<bool, StoreError> {
        let mut inner = self.lock()?;
        let key = (owner.clone(), conversation_id.to_owned());
        let Some(items) = inner.items.get_mut(&key) else {
            return Ok(false);
        };
        let Some(index) = items.iter().position(|item| item.item_id == item_id) else {
            return Ok(false);
        };
        items.remove(index);
        inner.item_ids.remove(item_id);
        let rebuilt = rebuild_messages(inner.items.get(&key).map_or(&[][..], |items| items.as_slice()));
        if let Some(conversation) = inner.conversations.get_mut(&key) {
            conversation.messages = rebuilt;
        }
        Ok(true)
    }
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::expect_used, clippy::unwrap_used, reason = "tests")]
mod tests {
    use super::*;

    fn owner(tenant: &str) -> StateOwner {
        StateOwner::from_trusted_parts(tenant, "issuer", "subject").expect("owner")
    }

    fn item(owner: &StateOwner, conversation_id: &str, item_id: &str, position: i64) -> ConversationItemRecord {
        ConversationItemRecord {
            item_id: item_id.to_owned(),
            owner: owner.clone(),
            conversation_id: conversation_id.to_owned(),
            item_data: serde_json::json!({ "id": item_id }),
            created_at: 0,
            position,
        }
    }

    fn approval(id: &str) -> PendingApprovalRecord {
        PendingApprovalRecord {
            approval_id: id.to_owned(),
            server_label: "srv".to_owned(),
            tool_name: "tool".to_owned(),
            arguments: "{}".to_owned(),
            target_fingerprint: "fp".to_owned(),
        }
    }

    #[tokio::test]
    async fn tenant_isolation_hides_other_owners() {
        let store = InMemoryStore::new();
        let a = owner("a");
        let b = owner("b");
        store
            .upsert_response(&ResponseRecord {
                id: "r1".to_owned(),
                owner: a.clone(),
                created_at: 1,
                model: "m".to_owned(),
                response_object: serde_json::json!({}),
                input: serde_json::json!({}),
                messages: serde_json::json!([]),
            })
            .await
            .unwrap();
        assert!(store.get_response(&a, "r1").await.unwrap().is_some());
        assert!(store.get_response(&b, "r1").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn consume_approvals_is_all_or_nothing() {
        let store = InMemoryStore::new();
        let o = owner("a");
        store
            .record_pending_approvals(&o, "r1", &[approval("x"), approval("y")], 10)
            .await
            .unwrap();
        // Consume x alone.
        assert_eq!(store.consume_approvals(&o, "r1", &["x"], 20).await.unwrap(), None);
        // A batch containing the already-consumed x aborts wholesale, leaving y outstanding.
        assert_eq!(
            store.consume_approvals(&o, "r1", &["y", "x"], 30).await.unwrap(),
            Some(1)
        );
        assert_eq!(store.consume_approvals(&o, "r1", &["y"], 40).await.unwrap(), None);
        // A duplicate id within one batch aborts at the second occurrence.
        store
            .record_pending_approvals(&o, "r1", &[approval("z")], 10)
            .await
            .unwrap();
        assert_eq!(
            store.consume_approvals(&o, "r1", &["z", "z"], 50).await.unwrap(),
            Some(1)
        );
        // z stays outstanding after the aborted duplicate batch.
        assert_eq!(store.consume_approvals(&o, "r1", &["z"], 60).await.unwrap(), None);
    }

    #[tokio::test]
    async fn cas_messages_guards_concurrent_update() {
        let store = InMemoryStore::new();
        let o = owner("a");
        store
            .upsert_conversation(&ConversationRecord {
                conversation_id: "c1".to_owned(),
                owner: o.clone(),
                created_at: 1,
                metadata: serde_json::json!({}),
                messages: serde_json::json!(["v0"]),
            })
            .await
            .unwrap();
        let expected = serde_json::json!(["v0"]);
        let stale = serde_json::json!(["nope"]);
        assert!(
            store
                .compare_and_swap_conversation_messages(&o, "c1", &expected, &serde_json::json!(["v1"]))
                .await
                .unwrap()
        );
        assert!(
            !store
                .compare_and_swap_conversation_messages(&o, "c1", &stale, &serde_json::json!(["v2"]))
                .await
                .unwrap()
        );
    }

    #[tokio::test]
    async fn create_items_and_sync_assigns_positions_and_rebuilds() {
        let store = InMemoryStore::new();
        let o = owner("a");
        store
            .upsert_conversation(&ConversationRecord {
                conversation_id: "c1".to_owned(),
                owner: o.clone(),
                created_at: 1,
                metadata: serde_json::json!({}),
                messages: serde_json::json!([]),
            })
            .await
            .unwrap();
        // Input positions are ignored and assigned from max+1.
        store
            .create_items_and_sync_messages(&o, "c1", &[item(&o, "c1", "i1", 999), item(&o, "c1", "i2", 999)])
            .await
            .unwrap();
        assert_eq!(store.conversation_item_position(&o, "c1", "i1").await.unwrap(), Some(1));
        assert_eq!(store.conversation_item_position(&o, "c1", "i2").await.unwrap(), Some(2));
        assert_eq!(store.max_item_position(&o, "c1").await.unwrap(), 2);
        let conversation = ConversationItemStore::get_conversation(&store, &o, "c1")
            .await
            .unwrap()
            .expect("conversation");
        assert_eq!(conversation.messages.as_array().map(Vec::len), Some(2));
    }

    #[tokio::test]
    async fn duplicate_item_id_is_rejected_across_owners() {
        let store = InMemoryStore::new();
        let a = owner("a");
        let b = owner("b");
        for (o, conversation_id) in [(&a, "c1"), (&b, "c2")] {
            store
                .upsert_conversation(&ConversationRecord {
                    conversation_id: conversation_id.to_owned(),
                    owner: o.clone(),
                    created_at: 1,
                    metadata: serde_json::json!({}),
                    messages: serde_json::json!([]),
                })
                .await
                .unwrap();
        }
        store
            .create_conversation_items(&[item(&a, "c1", "shared", 1)])
            .await
            .unwrap();
        let err = store.create_conversation_items(&[item(&b, "c2", "shared", 1)]).await;
        assert!(matches!(err, Err(StoreError::InvalidInput(_))));
    }
}
