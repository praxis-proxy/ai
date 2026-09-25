// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Response store persistence layer for AI API filters.
//!
//! The persistence contracts (traits, records, owner, registry, in-memory
//! backend) live in the SQL-free `praxis-ai-store` crate and are re-exported
//! here at their original paths. This module keeps the SQL backends and the
//! transport-bound registry wrapper.

// The SQL backends moved to the praxis-ai-store-backends crate (#1260). They are
// re-exported below at their original `crate::store::*` paths so the
// conversations and responses filters keep compiling unchanged. The transitional
// dependency on praxis-ai-store-backends is removed once #1259b and #1262 take
// those filters off concrete-store construction.

#[cfg(test)]
#[cfg(all(feature = "store-postgres", feature = "store-sqlite"))]
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

use praxis_ai_store::StoreRegistry;
// Pool tuning, TLS mode, and compression config live in the SQL-free contract
// crate, re-exported here at their old paths.
pub use praxis_ai_store::{CompressionAlgorithm, PoolConfig, SslMode, StoreCompressionConfig};
// The persistence contracts moved to praxis-ai-store; re-exported at their old
// paths so existing `crate::store::*` references keep compiling. OwnerScopedStore
// keeps its former name here.
pub use praxis_ai_store::{
    ConversationItemRecord, ConversationItemStore, ConversationRecord, OwnerScopedStore as OwnerScopedResponseStore,
    PendingApprovalRecord, PersistedStateBackend, ResponseRecord, ResponseStore, StoreError,
};
#[cfg(feature = "store-sqlite")]
pub use praxis_ai_store_backends::SqliteResponseStore;
#[cfg(feature = "store-postgres")]
pub(crate) use praxis_ai_store_backends::postgres_url;
#[cfg(any(feature = "store-sqlite", feature = "store-postgres"))]
pub use praxis_ai_store_backends::store_backend_factories;
#[cfg(feature = "store-postgres")]
pub use praxis_ai_store_backends::to_pg_ssl_mode;
#[cfg(feature = "store-postgres")]
pub(crate) use praxis_ai_store_backends::validate_postgres_table_identifiers;
#[cfg(all(feature = "store-postgres", feature = "openai-conversations"))]
pub(crate) use praxis_ai_store_backends::validate_postgres_table_set_identifiers;
#[cfg(feature = "store-postgres")]
pub use praxis_ai_store_backends::{PgTlsConfig, PostgresResponseStore};

use crate::StateOwner;

// -----------------------------------------------------------------------------
// ResponseStoreRegistry
// -----------------------------------------------------------------------------

/// Transport-bound wrapper over the SQL-free [`StoreRegistry`].
///
/// The registry itself carries no transport dependency; this wrapper adds the
/// [`praxis_filter::PipelineExtension`] binding so a listener can install it and
/// filters can resolve it from the request context. Filters look up stores by
/// name at request time and take an owner-scoped handle.
#[derive(Clone, Default)]
pub struct ResponseStoreRegistry {
    /// The underlying transport-free registry.
    inner: StoreRegistry,
}

impl ResponseStoreRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: StoreRegistry::new(),
        }
    }

    /// Register a named combined backend.
    ///
    /// # Errors
    ///
    /// Returns `StoreError::Unavailable` if a store with the same name is
    /// already registered.
    pub fn register(&self, name: &Arc<str>, store: Arc<dyn PersistedStateBackend>) -> Result<(), StoreError> {
        self.inner.register(name, store)
    }

    /// Look up a backend by name and bind all request-driven access to `owner`.
    #[must_use]
    pub fn get_scoped(&self, name: &str, owner: &StateOwner) -> Option<OwnerScopedResponseStore> {
        self.inner.get_scoped(name, owner)
    }

    /// Return whether a named backend is already registered.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.inner.contains(name)
    }

    /// Return whether two registry handles share the same backing storage.
    #[must_use]
    pub fn shares_storage_with(&self, other: &Self) -> bool {
        self.inner.shares_storage_with(&other.inner)
    }
}

impl praxis_filter::PipelineExtension for ResponseStoreRegistry {
    fn prepare(&self, extensions: &mut praxis_filter::RequestExtensions) {
        extensions.insert(self.clone());
    }
}

impl From<StoreRegistry> for ResponseStoreRegistry {
    /// Wrap a registry, sharing its backing storage.
    ///
    /// The serving-runtime provisioner installs an empty registry into a pipeline
    /// and later registers backends into the same map through a clone, so the
    /// pipeline observes the backends once provisioning completes.
    fn from(inner: StoreRegistry) -> Self {
        Self { inner }
    }
}

/// Registry name of the process-default response store.
#[cfg(feature = "openai-conversations")]
pub use crate::openai::conversations::{CONVERSATIONS_STORE_NAME, store_ref_config as conversations_store_ref_config};
pub use crate::openai::responses::DEFAULT_STORE_NAME;

/// Filter type that configures the response store. The serving runtime scans
/// filter chains for this type to provision the backends the store filter reads.
pub const RESPONSE_STORE_FILTER_NAME: &str = "openai_response_store";

/// Filter type that configures the conversations store. The serving runtime
/// scans filter chains for this type to provision the conversations backend the
/// conversations filter reads.
pub const CONVERSATIONS_STORE_FILTER_NAME: &str = "openai_conversations";
