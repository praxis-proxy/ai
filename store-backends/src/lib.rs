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
/// Measure work that an append actually adds. Existing item rows are not read
/// or rebuilt by append-back and therefore do not consume this operation's
/// allowance again.
#[cfg(any(feature = "sqlite", feature = "_postgres"))]
fn conversation_append_charge(
    conversation_id: &str,
    items: &[ConversationItemRecord],
    max_charge_bytes: usize,
) -> Result<usize, StoreError> {
    let mut charge = json_values_charge(std::iter::empty(), &[conversation_id], max_charge_bytes)?;
    for item in items {
        let item_charge = json_values_charge([&item.item_data], &[&item.item_id], max_charge_bytes - charge)?;
        charge = charge.checked_add(item_charge).ok_or(StoreError::PayloadTooLarge)?;
    }
    Ok(charge)
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
    BudgetedRecord, ConversationItemRecord, ConversationItemStore, ConversationRecord, EventLogStatus,
    PendingApprovalRecord, ResponseEventRecord, ResponseRecord, ResponseStore, StoreError, json_payload_charge,
    json_values_charge, json_values_fit, retained_charge,
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
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, reason = "tests")]
mod tests {
    #[cfg(any(feature = "sqlite", feature = "_postgres"))]
    use super::{ConversationItemRecord, conversation_append_charge};

    #[cfg(any(feature = "sqlite", feature = "_postgres"))]
    #[test]
    fn append_admission_counts_new_items_without_existing_history() {
        let owner = praxis_ai_store::StateOwner::from_trusted_parts("tenant", "issuer", "subject").unwrap();
        let item = ConversationItemRecord {
            item_id: "new".to_owned(),
            owner,
            conversation_id: "conversation".to_owned(),
            item_data: serde_json::json!({"text": "new turn"}),
            created_at: 0,
            position: 0,
        };
        let charge = conversation_append_charge("conversation", std::slice::from_ref(&item), 64 * 1024 * 1024).unwrap();
        assert!(charge < 64 * 1024 * 1024);
        assert!(conversation_append_charge("conversation", &[item], charge - 1).is_err());
    }
}
