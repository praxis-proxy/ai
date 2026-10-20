// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! SQL storage backends for praxis-ai: Postgres and Sqlite.
//!
//! Internal, unstable implementation crate: unpublished (`publish = false`), a
//! first-party workspace detail with no external API surface and no semver
//! promise.
//!
//! The concrete [`ResponseStore`] backends, their connection pool, TLS and URL
//! validation, schema management, and the [`StoreBackendFactory`] provisioning.
//! This is the only crate that carries the SQL client crypto, so a consumer
//! that omits it ships no datastore cryptography.
//!
//! [`ResponseStore`]: praxis_ai_store::ResponseStore
//! [`StoreBackendFactory`]: praxis_ai_store::StoreBackendFactory

#[cfg(any(feature = "sqlite", feature = "_postgres"))]
mod pool;
#[cfg(feature = "_postgres")]
mod postgres;
#[cfg(feature = "_postgres")]
mod postgres_tls;
#[cfg(feature = "_postgres")]
pub mod postgres_url;
#[cfg(any(feature = "sqlite", feature = "_postgres"))]
mod provisioning;
#[cfg(any(feature = "sqlite", feature = "_postgres"))]
mod schemas;
#[cfg(feature = "sqlite")]
mod sqlite;

// Pool tuning, TLS mode, and the compression codec live in the SQL-free
// praxis-ai-store crate; aliased here so the backends reach them through `super`.
#[cfg(feature = "_postgres")]
pub use postgres::PostgresResponseStore;
#[cfg(feature = "_postgres")]
pub use postgres::to_pg_ssl_mode;
#[cfg(feature = "_postgres")]
pub use postgres_tls::PgTlsConfig;
#[cfg(any(feature = "sqlite", feature = "_postgres"))]
pub(crate) use praxis_ai_store::PoolConfig;
#[cfg(feature = "_postgres")]
pub(crate) use praxis_ai_store::SslMode;
/// Bound the SQL message-cache rebuild before loading item rows. Sparse nested
/// JSON can expand far beyond its wire length across row decoding, parsed
/// values, the serialized cache, and SQL argument copies.
#[cfg(any(feature = "sqlite", feature = "_postgres"))]
const CONVERSATION_REBUILD_MULTIPLIER: usize = 512;

/// Check the complete cache projection against the caller's memory allowance.
#[cfg(any(feature = "sqlite", feature = "_postgres"))]
fn conversation_rebuild_fits(raw_bytes: i64, row_count: i64, max_bytes: usize) -> bool {
    let Ok(raw_bytes) = usize::try_from(raw_bytes) else {
        return false;
    };
    let Ok(row_count) = usize::try_from(row_count) else {
        return false;
    };
    raw_bytes
        .checked_mul(CONVERSATION_REBUILD_MULTIPLIER)
        .and_then(|bytes| bytes.checked_add(row_count.checked_mul(4)?))
        .and_then(|bytes| bytes.checked_add(4))
        .is_some_and(|bytes| bytes <= max_bytes)
}

/// Reconstruct the history from a lazy snapshot and ordered item rows.
///
/// A snapshot array is authoritative only when the item table has no rows. Once
/// item rows exist, only the explicitly wrapped legacy prefix is retained.
#[cfg(any(feature = "sqlite", feature = "_postgres"))]
fn reconstruct_conversation_history(
    snapshot: serde_json::Value,
    mut item_rows: Vec<serde_json::Value>,
) -> Vec<serde_json::Value> {
    let prefix = match snapshot {
        serde_json::Value::Array(messages) if item_rows.is_empty() => return messages,
        serde_json::Value::Object(mut object) => match object.remove("legacy_messages") {
            Some(serde_json::Value::Array(messages)) => messages,
            _ => Vec::new(),
        },
        _ => Vec::new(),
    };
    if prefix.is_empty() {
        item_rows
    } else {
        let mut messages = prefix;
        messages.append(&mut item_rows);
        messages
    }
}

#[cfg(any(feature = "sqlite", feature = "_postgres"))]
pub(crate) use praxis_ai_store::compression;
#[cfg(any(feature = "sqlite", feature = "_postgres"))]
use praxis_ai_store::{
    ConversationItemRecord, ConversationItemStore, ConversationRecord, EventLogStatus, PendingApprovalRecord,
    ResponseEventRecord, ResponseRecord, ResponseStore, StoreError, json_values_fit,
};
#[cfg(any(feature = "sqlite", feature = "_postgres"))]
pub use provisioning::store_backend_factories;
#[cfg(feature = "_postgres")]
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
#[cfg(any(feature = "sqlite", feature = "_postgres"))]
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

#[cfg(test)]
mod tests {
    #[cfg(any(feature = "sqlite", feature = "_postgres"))]
    use super::conversation_rebuild_fits;

    #[cfg(any(feature = "sqlite", feature = "_postgres"))]
    #[test]
    fn sparse_json_rebuild_needs_structural_headroom() {
        // 40 KiB of nested arrays expanded past 12 MiB across parser/copy peaks
        // in the restore allocation probe. A 26x byte reserve admitted it.
        assert!(
            !conversation_rebuild_fits(40_018, 1, 8_388_608),
            "sparse tree must reject"
        );
        assert!(conversation_rebuild_fits(128, 1, 8_388_608), "small JSON must fit");
    }
}
