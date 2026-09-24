// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Response store persistence layer for AI API filters.
//!
//! Provides the [`ResponseStore`] async trait, optional SQLite and `PostgreSQL`
//! backends, and supporting types. Used by AI API filters for persisting
//! response records and conversation history.

#[cfg_attr(
    not(any(feature = "_store-postgres", feature = "store-sqlite")),
    expect(clippy::allow_attributes, reason = "dead_code expect unfulfilled on module"),
    allow(
        dead_code,
        reason = "codec helpers are unused until a SQL backend feature is enabled"
    )
)]
mod compression;
#[cfg_attr(
    not(any(feature = "_store-postgres", feature = "store-sqlite")),
    expect(clippy::allow_attributes, reason = "dead_code expect unfulfilled on module"),
    allow(
        dead_code,
        reason = "backend helpers are unused until a SQL backend feature is enabled"
    )
)]
mod pool;
#[cfg(feature = "_store-postgres")]
mod postgres;
#[cfg(feature = "_store-postgres")]
mod postgres_tls;
#[cfg(feature = "_store-postgres")]
pub(crate) mod postgres_url;
#[cfg_attr(
    not(any(feature = "_store-postgres", feature = "store-sqlite")),
    expect(clippy::allow_attributes, reason = "dead_code expect unfulfilled on module"),
    allow(
        dead_code,
        reason = "backend helpers are unused until a SQL backend feature is enabled"
    )
)]
mod schemas;
#[cfg(feature = "store-sqlite")]
mod sqlite;
mod ssl_mode;
mod trait_def;
mod types;

#[cfg(test)]
#[cfg(all(feature = "_store-postgres", feature = "store-sqlite"))]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::too_many_lines,
    reason = "tests"
)]
mod tests;

use std::sync::Arc;

use dashmap::{DashMap, mapref::entry::Entry};
/// Validate response-store table identifiers.
pub(crate) use schemas::validate_identifier as validate_table_identifier;
#[cfg(feature = "_store-postgres")]
pub(crate) use schemas::validate_postgres_table_identifiers;
#[cfg(all(feature = "_store-postgres", feature = "openai-conversations"))]
pub(crate) use schemas::validate_postgres_table_set_identifiers;

#[cfg(feature = "_store-postgres")]
pub use self::postgres::PostgresResponseStore;
#[cfg(feature = "_store-postgres")]
pub use self::postgres_tls::PgTlsConfig;
#[cfg(feature = "store-sqlite")]
pub use self::sqlite::SqliteResponseStore;
pub use self::{
    compression::{CompressionAlgorithm, StoreCompressionConfig},
    pool::PoolConfig,
    ssl_mode::SslMode,
    trait_def::{ConversationItemStore, ResponseStore},
    types::{ConversationItemRecord, ConversationRecord, PendingApprovalRecord, ResponseRecord, StoreError},
};

// -----------------------------------------------------------------------------
// ResponseStoreRegistry
// -----------------------------------------------------------------------------

/// Thread-safe registry of named `ResponseStore` backends.
///
/// Each listener can own a registry populated at startup. Filters
/// look up stores by name at request time through the
/// [`HttpFilterContext`].
///
/// [`HttpFilterContext`]: praxis_filter::HttpFilterContext
#[derive(Clone)]
pub struct ResponseStoreRegistry {
    /// Named store backends.
    #[expect(clippy::type_complexity, reason = "DashMap of trait objects is inherently verbose")]
    stores: Arc<DashMap<Arc<str>, Arc<dyn ResponseStore>>>,
}

/// Response-store handle permanently bound to one validated owner.
///
/// Request-driven consumers obtain this facade from
/// [`ResponseStoreRegistry::get_scoped`] instead of receiving the raw backend,
/// so a later operation cannot accidentally substitute an arbitrary tenant or
/// principal string.
#[derive(Clone)]
pub struct OwnerScopedResponseStore {
    /// Shared backend hidden behind the owner-bound facade.
    store: Arc<dyn ResponseStore>,
    /// Immutable scope applied to every operation.
    owner: crate::StateOwner,
}

impl OwnerScopedResponseStore {
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
        self.store.get_conversation(&self.owner, id).await
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

    /// Reject records built under a different owner scope.
    fn require_matching_owner(&self, owner: &crate::StateOwner) -> Result<(), StoreError> {
        if owner == &self.owner {
            Ok(())
        } else {
            Err(StoreError::InvalidInput(
                "record owner does not match owner-scoped store".to_owned(),
            ))
        }
    }
}

impl ResponseStoreRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            stores: Arc::new(DashMap::new()),
        }
    }

    /// Register a named store backend.
    ///
    /// # Errors
    ///
    /// Returns `StoreError::Unavailable` if a store with the
    /// same name is already registered.
    pub fn register(&self, name: &Arc<str>, store: Arc<dyn ResponseStore>) -> Result<(), StoreError> {
        match self.stores.entry(Arc::clone(name)) {
            Entry::Vacant(entry) => {
                entry.insert(store);
                Ok(())
            },
            Entry::Occupied(_) => Err(StoreError::Unavailable(format!(
                "response store '{name}' is already registered"
            ))),
        }
    }

    /// Look up a store by name and bind all request-driven access to `owner`.
    pub fn get_scoped(&self, name: &str, owner: &crate::StateOwner) -> Option<OwnerScopedResponseStore> {
        self.get_backend(name).map(|store| OwnerScopedResponseStore {
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
    fn get_backend(&self, name: &str) -> Option<Arc<dyn ResponseStore>> {
        self.stores.get(name).map(|r| Arc::clone(r.value()))
    }

    /// Return whether two registry handles share the same backing storage.
    #[must_use]
    pub fn shares_storage_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.stores, &other.stores)
    }
}

impl Default for ResponseStoreRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl praxis_filter::PipelineExtension for ResponseStoreRegistry {
    fn prepare(&self, extensions: &mut praxis_filter::RequestExtensions) {
        extensions.insert(self.clone());
    }
}
