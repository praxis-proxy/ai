// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Internal, unstable persisted-state contracts for praxis-ai.
//!
//! SQL-free and transport-free: the persistence traits, record types, the owner
//! identity, a unified backend registry, and an in-memory backend. This crate
//! is a first-party workspace implementation detail with no external API and no
//! semver promise.

mod backend_config;
#[cfg(feature = "compression")]
pub mod compression;
mod factory;
mod owner;
mod registry;
mod traits;
mod types;
pub mod url_security;

#[cfg(feature = "test-support")]
pub mod memory;

#[cfg(feature = "test-support")]
pub mod contract_tests;

pub use backend_config::{DEFAULT_MAX_CONNECTIONS, MAX_IDENTIFIER_LEN, PoolConfig, SslMode, validate_table_identifier};
#[cfg(feature = "compression")]
pub use compression::{CompressionAlgorithm, StoreCompressionConfig};
pub use factory::{BackendError, EffectiveConfigKey, ProvisionedBackend, RetireBackend, StoreBackendFactory};
pub use owner::{StateOwner, StateOwnerError, validate_component};
pub use registry::{OwnerScopedStore, StoreRegistry};
pub use traits::{ConversationItemStore, PersistedStateBackend, ResponseStore};
pub use types::{ConversationItemRecord, ConversationRecord, PendingApprovalRecord, ResponseRecord, StoreError};
