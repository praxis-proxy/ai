// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Unified, transport-free registry of persisted-state backends.
//!
//! The registry holds a combined [`PersistedStateBackend`] handle, so a resolved
//! backend provably implements both the response and conversation-item halves.
//! Request-driven consumers take an [`OwnerScopedStore`] bound to one validated
//! owner rather than the raw backend, so a later operation cannot substitute an
//! arbitrary tenant or principal.

use std::sync::Arc;

use dashmap::{DashMap, mapref::entry::Entry};

use crate::{
    owner::StateOwner,
    traits::{PersistedStateBackend, ResponseStore},
    types::{ConversationItemRecord, ConversationRecord, PendingApprovalRecord, ResponseRecord, StoreError},
};

/// Thread-safe registry of named persisted-state backends.
///
/// Each listener owns a registry populated at startup. The registry carries no
/// transport dependency; the pipeline binding lives in the transport layer.
#[derive(Clone)]
pub struct StoreRegistry {
    /// Named combined backends.
    #[expect(clippy::type_complexity, reason = "DashMap of trait objects is inherently verbose")]
    stores: Arc<DashMap<Arc<str>, Arc<dyn PersistedStateBackend>>>,
}

/// Backend handle permanently bound to one validated owner.
///
/// Request-driven consumers obtain this facade from [`StoreRegistry::get_scoped`]
/// instead of the raw backend, so a later operation cannot substitute an
/// arbitrary owner. The facade exposes both persisted-state capabilities: the
/// Responses surface and the conversation-lifecycle and item surface, each
/// bound to the same validated owner.
#[derive(Clone)]
pub struct OwnerScopedStore {
    /// Shared combined backend hidden behind the owner-bound facade.
    store: Arc<dyn PersistedStateBackend>,
    /// Immutable scope applied to every operation.
    owner: StateOwner,
}

impl OwnerScopedStore {
    /// The validated owner every operation on this handle is bound to.
    ///
    /// Server-set at [`StoreRegistry::get_scoped`]; exposed read-only so a caller
    /// can stamp records with the bound scope rather than passing an owner that
    /// the write path would then have to re-check.
    #[must_use]
    pub fn owner(&self) -> &StateOwner {
        &self.owner
    }

    /// Retrieve a response visible to this owner.
    ///
    /// # Errors
    ///
    /// Returns a store error when the backend query fails.
    pub async fn get_response(&self, id: &str) -> Result<Option<ResponseRecord>, StoreError> {
        self.store.get_response(&self.owner, id).await
    }

    /// Delete a response visible to this owner.
    ///
    /// # Errors
    ///
    /// Returns a store error when the backend mutation fails.
    pub async fn delete_response(&self, id: &str) -> Result<bool, StoreError> {
        self.store.delete_response(&self.owner, id).await
    }

    /// Retrieve a conversation visible to this owner.
    ///
    /// # Errors
    ///
    /// Returns a store error when the backend query fails.
    pub async fn get_conversation(&self, id: &str) -> Result<Option<ConversationRecord>, StoreError> {
        // Both trait halves expose get_conversation; the facade uses the
        // ResponseStore read-only view. Upcast to disambiguate.
        let store: &dyn ResponseStore = self.store.as_ref();
        store.get_conversation(&self.owner, id).await
    }

    /// Persist a response only when its immutable owner matches this handle.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error for an owner mismatch or the backend
    /// error from persistence.
    pub async fn upsert_response(&self, record: &ResponseRecord) -> Result<(), StoreError> {
        self.require_matching_owner(&record.owner)?;
        self.store.upsert_response(record).await
    }

    /// Persist a response and the pending approvals it issued, only when the
    /// record's immutable owner matches this handle. The backend commits both in
    /// one transaction so a concurrent delete cannot orphan an approval.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error for an owner mismatch or the backend
    /// error from persistence.
    pub async fn persist_response_with_pending_approvals(
        &self,
        record: &ResponseRecord,
        pending_approvals: &[PendingApprovalRecord],
    ) -> Result<(), StoreError> {
        self.require_matching_owner(&record.owner)?;
        self.store
            .persist_response_with_pending_approvals(record, pending_approvals)
            .await
    }

    /// Retrieve pending approvals issued to this owner.
    ///
    /// # Errors
    ///
    /// Returns a store error when the backend query fails.
    pub async fn get_pending_approvals(
        &self,
        response_id: &str,
        approval_ids: &[&str],
    ) -> Result<Vec<PendingApprovalRecord>, StoreError> {
        self.store
            .get_pending_approvals(&self.owner, response_id, approval_ids)
            .await
    }

    /// Atomically consume approvals issued to this owner.
    ///
    /// # Errors
    ///
    /// Returns a store error when the backend mutation fails.
    pub async fn consume_approvals(
        &self,
        response_id: &str,
        approval_ids: &[&str],
        consumed_at: i64,
    ) -> Result<Option<usize>, StoreError> {
        self.store
            .consume_approvals(&self.owner, response_id, approval_ids, consumed_at)
            .await
    }

    /// Persist a conversation only when its immutable owner matches this handle.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error for an owner mismatch or the backend
    /// error from persistence.
    pub async fn upsert_conversation(&self, record: &ConversationRecord) -> Result<(), StoreError> {
        self.require_matching_owner(&record.owner)?;
        self.store.upsert_conversation(record).await
    }

    /// Update only this owner's conversation message cache.
    ///
    /// # Errors
    ///
    /// Returns a store error when the backend mutation fails.
    pub async fn update_conversation_messages(
        &self,
        conversation_id: &str,
        messages: &serde_json::Value,
    ) -> Result<bool, StoreError> {
        self.store
            .update_conversation_messages(&self.owner, conversation_id, messages)
            .await
    }

    /// Update only this owner's conversation metadata.
    ///
    /// # Errors
    ///
    /// Returns a store error when the backend mutation fails.
    pub async fn update_conversation_metadata(
        &self,
        conversation_id: &str,
        metadata: &serde_json::Value,
    ) -> Result<bool, StoreError> {
        self.store
            .update_conversation_metadata(&self.owner, conversation_id, metadata)
            .await
    }

    /// Compare-and-swap this owner's conversation message cache.
    ///
    /// # Errors
    ///
    /// Returns a store error when the backend mutation fails.
    pub async fn compare_and_swap_conversation_messages(
        &self,
        conversation_id: &str,
        expected_messages: &serde_json::Value,
        messages: &serde_json::Value,
    ) -> Result<bool, StoreError> {
        self.store
            .compare_and_swap_conversation_messages(&self.owner, conversation_id, expected_messages, messages)
            .await
    }

    /// Delete a conversation visible to this owner.
    ///
    /// # Errors
    ///
    /// Returns a store error when the backend mutation fails.
    pub async fn delete_conversation(&self, conversation_id: &str) -> Result<bool, StoreError> {
        self.store.delete_conversation(&self.owner, conversation_id).await
    }

    /// Insert conversation items, each of which must carry this owner.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error for an owner mismatch or the backend
    /// error from persistence.
    pub async fn create_conversation_items(&self, items: &[ConversationItemRecord]) -> Result<(), StoreError> {
        for item in items {
            self.require_matching_owner(&item.owner)?;
        }
        self.store.create_conversation_items(items).await
    }

    /// List a conversation's items visible to this owner.
    ///
    /// # Errors
    ///
    /// Returns a store error when the backend query fails.
    pub async fn list_conversation_items(
        &self,
        conversation_id: &str,
        after_item_id: Option<&str>,
        limit: u32,
        ascending: bool,
    ) -> Result<Vec<ConversationItemRecord>, StoreError> {
        self.store
            .list_conversation_items(&self.owner, conversation_id, after_item_id, limit, ascending)
            .await
    }

    /// Return the subset of `item_ids` already present for this owner.
    ///
    /// # Errors
    ///
    /// Returns a store error when the backend query fails.
    pub async fn get_existing_conversation_item_ids(
        &self,
        conversation_id: &str,
        item_ids: &[&str],
    ) -> Result<Vec<String>, StoreError> {
        self.store
            .get_existing_conversation_item_ids(&self.owner, conversation_id, item_ids)
            .await
    }

    /// Retrieve a single conversation item visible to this owner.
    ///
    /// # Errors
    ///
    /// Returns a store error when the backend query fails.
    pub async fn get_conversation_item(
        &self,
        conversation_id: &str,
        item_id: &str,
    ) -> Result<Option<ConversationItemRecord>, StoreError> {
        self.store
            .get_conversation_item(&self.owner, conversation_id, item_id)
            .await
    }

    /// Delete a single conversation item visible to this owner.
    ///
    /// # Errors
    ///
    /// Returns a store error when the backend mutation fails.
    pub async fn delete_conversation_item(&self, conversation_id: &str, item_id: &str) -> Result<bool, StoreError> {
        self.store
            .delete_conversation_item(&self.owner, conversation_id, item_id)
            .await
    }

    /// Look up an item's position within this owner's conversation.
    ///
    /// # Errors
    ///
    /// Returns a store error when the backend query fails.
    pub async fn conversation_item_position(
        &self,
        conversation_id: &str,
        item_id: &str,
    ) -> Result<Option<i64>, StoreError> {
        self.store
            .conversation_item_position(&self.owner, conversation_id, item_id)
            .await
    }

    /// Return the maximum item position in this owner's conversation.
    ///
    /// # Errors
    ///
    /// Returns a store error when the backend query fails.
    pub async fn max_item_position(&self, conversation_id: &str) -> Result<i64, StoreError> {
        self.store.max_item_position(&self.owner, conversation_id).await
    }

    /// Atomically insert items and rebuild this owner's message cache.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error for an owner mismatch or the backend
    /// error from persistence.
    pub async fn create_items_and_sync_messages(
        &self,
        conversation_id: &str,
        items: &[ConversationItemRecord],
    ) -> Result<(), StoreError> {
        for item in items {
            self.require_matching_owner(&item.owner)?;
        }
        self.store
            .create_items_and_sync_messages(&self.owner, conversation_id, items)
            .await
    }

    /// Atomically delete an item and rebuild this owner's message cache.
    ///
    /// # Errors
    ///
    /// Returns a store error when the backend mutation fails.
    pub async fn delete_item_and_sync_messages(
        &self,
        conversation_id: &str,
        item_id: &str,
    ) -> Result<bool, StoreError> {
        self.store
            .delete_item_and_sync_messages(&self.owner, conversation_id, item_id)
            .await
    }

    /// Reject records built under a different owner scope.
    fn require_matching_owner(&self, owner: &StateOwner) -> Result<(), StoreError> {
        if owner == &self.owner {
            Ok(())
        } else {
            Err(StoreError::InvalidInput(
                "record owner does not match owner-scoped store".to_owned(),
            ))
        }
    }
}

impl StoreRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            stores: Arc::new(DashMap::new()),
        }
    }

    /// Register a named combined backend.
    ///
    /// # Errors
    ///
    /// Returns `StoreError::Unavailable` if a store with the same name is
    /// already registered.
    pub fn register(&self, name: &Arc<str>, store: Arc<dyn PersistedStateBackend>) -> Result<(), StoreError> {
        match self.stores.entry(Arc::clone(name)) {
            Entry::Vacant(entry) => {
                entry.insert(store);
                Ok(())
            },
            Entry::Occupied(_) => Err(StoreError::Unavailable(format!(
                "persisted-state store '{name}' is already registered"
            ))),
        }
    }

    /// Remove a named backend, rolling back a partially provisioned attempt so a
    /// later reference's failure does not leave earlier entries registered.
    pub fn deregister(&self, name: &str) {
        self.stores.remove(name);
    }

    /// Look up a backend by name and bind all request-driven access to `owner`.
    #[must_use]
    pub fn get_scoped(&self, name: &str, owner: &StateOwner) -> Option<OwnerScopedStore> {
        self.get_backend(name).map(|store| OwnerScopedStore {
            store,
            owner: owner.clone(),
        })
    }

    /// Return whether a named backend is already registered.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.stores.contains_key(name)
    }

    /// Internal raw lookup used only to construct a constrained facade.
    fn get_backend(&self, name: &str) -> Option<Arc<dyn PersistedStateBackend>> {
        self.stores.get(name).map(|r| Arc::clone(r.value()))
    }

    /// Return whether two registry handles share the same backing storage.
    #[must_use]
    pub fn shares_storage_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.stores, &other.stores)
    }
}

impl Default for StoreRegistry {
    fn default() -> Self {
        Self::new()
    }
}
