// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! SQL storage backends for praxis-ai: Postgres and Sqlite.
//!
//! The concrete [`ResponseStore`] backends, their connection pool, TLS and URL
//! validation, schema management, and the [`StoreBackendFactory`] provisioning.
//! This is the only crate that carries the SQL client crypto, so a consumer
//! that omits it ships no datastore cryptography.
//!
//! [`ResponseStore`]: praxis_ai_store::ResponseStore
//! [`StoreBackendFactory`]: praxis_ai_store::StoreBackendFactory

mod pool;
#[cfg(feature = "postgres")]
mod postgres;
#[cfg(feature = "postgres")]
mod postgres_tls;
#[cfg(feature = "postgres")]
pub mod postgres_url;
mod provisioning;
mod schemas;
#[cfg(feature = "sqlite")]
mod sqlite;

// Pool tuning, TLS mode, and the compression codec live in the SQL-free
// praxis-ai-store crate; aliased here so the backends reach them through `super`.
#[cfg(feature = "postgres")]
pub use postgres::PostgresResponseStore;
#[cfg(feature = "postgres")]
pub use postgres::to_pg_ssl_mode;
#[cfg(feature = "postgres")]
pub use postgres_tls::PgTlsConfig;
#[cfg(any(feature = "sqlite", feature = "postgres"))]
pub(crate) use praxis_ai_store::PoolConfig;
#[cfg(feature = "postgres")]
pub(crate) use praxis_ai_store::SslMode;
#[cfg(any(feature = "sqlite", feature = "postgres"))]
pub(crate) use praxis_ai_store::compression;
use praxis_ai_store::{
    ConversationItemRecord, ConversationItemStore, ConversationRecord, PendingApprovalRecord, ResponseRecord,
    ResponseStore, StoreError,
};
#[cfg(any(feature = "sqlite", feature = "postgres"))]
pub use provisioning::store_backend_factories;
#[cfg(feature = "postgres")]
pub use schemas::{validate_postgres_table_identifiers, validate_postgres_table_set_identifiers};
#[cfg(feature = "sqlite")]
pub use sqlite::SqliteResponseStore;

/// Scrub a connection string, and any credentials embedded in it, from an error
/// message before it reaches [`StoreError::Database`] or a log line.
///
/// Shared by the Postgres and Sqlite backends and the provisioning factories,
/// which reach it through `super::`.
///
/// [`StoreError::Database`]: praxis_ai_store::StoreError::Database
#[cfg(any(feature = "sqlite", feature = "postgres"))]
pub(crate) fn redact_connection_error(url: &str, message: &str) -> String {
    let base = message.replace(url, "<redacted database url>");
    // Best effort for a `scheme://user:pass@host` credential echoed separately
    // from the full url: drop the userinfo segment. split_once avoids byte
    // indexing, which could split a UTF-8 character.
    let Some((before, rest)) = base.split_once("://") else {
        return base;
    };
    match rest.split_once('@') {
        Some((_userinfo, after_at)) => format!("{before}://<redacted credentials>@{after_at}"),
        None => format!("{before}://{rest}"),
    }
}
