// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! [`SqliteResponseStore`] — `SQLite` backend for the response store.

use async_trait::async_trait;
use sqlx::{
    AssertSqlSafe, Row as _, SqlitePool,
    sqlite::{SqliteConnectOptions, SqlitePoolOptions},
};
use tracing::info;

use super::{
    pool::{PoolConfig, apply_pool_config},
    schemas::{
        ActualKeyColumn, ActualTable, ActualUniqueIndex, SCHEMA_VERSION, SchemaCheck, TableNames, check_schema,
        expected_tables, generate_ddl, pending_approvals_table, schema_version_table, sqlite_key_column_folding,
    },
    trait_def::{ConversationItemStore, ResponseStore},
    types::{ConversationItemRecord, ConversationRecord, PendingApprovalRecord, ResponseRecord, StoreError},
};
use crate::StateOwner;

// -----------------------------------------------------------------------------
// SqliteResponseStore
// -----------------------------------------------------------------------------

/// SQLite-backed response store.
///
/// Uses [`sqlx::SqlitePool`] for async connection pooling. Table
/// names are configurable per provider (e.g., `openai_responses`,
/// `google_interactions`) to isolate data per provider.
pub struct SqliteResponseStore {
    /// Connection pool.
    pool: SqlitePool,
    /// Configured table names.
    tables: TableNames,
}

impl SqliteResponseStore {
    /// Create a new store and initialize the schema.
    ///
    /// The `database_url` is a `SQLite` connection string. Use
    /// `"sqlite::memory:"` for in-memory databases (testing) or
    /// `"sqlite:///path/to/db.sqlite?mode=rwc"` for file-backed.
    ///
    /// `responses_table` and `conversations_table` are the SQL
    /// table names to use. These come from the filter's YAML
    /// config (e.g., `openai_responses`). `items_table` is
    /// optional and enables conversation item storage.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::Database`] if the connection, schema
    /// initialization, or table name validation fails.
    pub async fn new(
        database_url: &str,
        responses_table: &str,
        conversations_table: &str,
        items_table: Option<&str>,
        pool_config: Option<&PoolConfig>,
    ) -> Result<Self, StoreError> {
        let tables = TableNames {
            responses: responses_table.to_owned(),
            conversations: conversations_table.to_owned(),
            items: items_table.map(str::to_owned),
        };
        let ddl = generate_ddl(&tables)?;

        let options: SqliteConnectOptions = database_url
            .parse()
            .map_err(|e: sqlx::Error| StoreError::Database(e.to_string()))?;

        let pool = sqlite_pool_options(database_url, pool_config)
            .connect_with(options.create_if_missing(true))
            .await
            .map_err(|e| StoreError::Database(e.to_string()))?;
        for statement in &ddl {
            sqlx::query(AssertSqlSafe(statement.as_str()))
                .execute(&pool)
                .await
                .map_err(|e| StoreError::Database(e.to_string()))?;
        }

        validate_schema(&pool, &tables).await?;
        check_schema_version(&pool, &tables).await?;

        info!(
            responses = responses_table,
            conversations = conversations_table,
            "response store initialized"
        );
        Ok(Self { pool, tables })
    }

    /// Insert or update a conversation row shared by both store traits.
    async fn upsert_conversation_record(&self, record: &ConversationRecord) -> Result<(), StoreError> {
        let messages = serde_json::to_string(&record.messages).map_err(|e| StoreError::Serialization(e.to_string()))?;
        let metadata = serde_json::to_string(&record.metadata).map_err(|e| StoreError::Serialization(e.to_string()))?;

        let sql = format!(
            "INSERT INTO {} \
             (conversation_id, tenant_id, owner_issuer, owner_subject, created_at, metadata, messages) \
             VALUES (?, ?, ?, ?, ?, ?, ?) \
             ON CONFLICT(conversation_id) \
             DO UPDATE SET messages = excluded.messages, \
             metadata = excluded.metadata \
             WHERE tenant_id = excluded.tenant_id \
               AND owner_issuer = excluded.owner_issuer \
               AND owner_subject = excluded.owner_subject",
            self.tables.conversations
        );

        let result = sqlx::query(AssertSqlSafe(sql.as_str()))
            .bind(&record.conversation_id)
            .bind(record.owner.tenant_id())
            .bind(record.owner.issuer())
            .bind(record.owner.subject())
            .bind(record.created_at)
            .bind(&metadata)
            .bind(&messages)
            .execute(&self.pool)
            .await
            .map_err(|e| StoreError::Database(e.to_string()))?;
        require_owner_preserving_write(result.rows_affected(), "conversation")
    }

    /// Retrieve a conversation row shared by both store traits.
    async fn get_conversation_record(
        &self,
        owner: &StateOwner,
        conversation_id: &str,
    ) -> Result<Option<ConversationRecord>, StoreError> {
        let sql = format!(
            "SELECT conversation_id, tenant_id, owner_issuer, owner_subject, created_at, metadata, messages \
             FROM {} \
             WHERE conversation_id = ? AND tenant_id = ? AND owner_issuer = ? AND owner_subject = ?",
            self.tables.conversations
        );

        let row = sqlx::query(AssertSqlSafe(sql.as_str()))
            .bind(conversation_id)
            .bind(owner.tenant_id())
            .bind(owner.issuer())
            .bind(owner.subject())
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| StoreError::Database(e.to_string()))?;

        row.map(|r| row_to_conversation_record(&r)).transpose()
    }

    /// Delete only a conversation row.
    async fn delete_conversation_record(&self, owner: &StateOwner, conversation_id: &str) -> Result<bool, StoreError> {
        let sql = format!(
            "DELETE FROM {} WHERE conversation_id = ? AND tenant_id = ? AND owner_issuer = ? AND owner_subject = ?",
            self.tables.conversations
        );

        let result = sqlx::query(AssertSqlSafe(sql.as_str()))
            .bind(conversation_id)
            .bind(owner.tenant_id())
            .bind(owner.issuer())
            .bind(owner.subject())
            .execute(&self.pool)
            .await
            .map_err(|e| StoreError::Database(e.to_string()))?;

        Ok(result.rows_affected() > 0)
    }
}

/// Build pool options for the requested `SQLite` database URL.
///
/// In-memory databases are pinned to a single connection regardless
/// of pool config (required to keep the database alive). For
/// file-backed databases, user-supplied [`PoolConfig`] values are
/// applied on top of sqlx defaults.
fn sqlite_pool_options(database_url: &str, pool_config: Option<&PoolConfig>) -> SqlitePoolOptions {
    if is_memory_database_url(database_url) {
        SqlitePoolOptions::new()
            .max_connections(1)
            .min_connections(1)
            .idle_timeout(None)
            .max_lifetime(None)
    } else {
        apply_pool_config(SqlitePoolOptions::new(), pool_config)
    }
}


/// Return whether the database URL targets an in-memory `SQLite` database.
fn is_memory_database_url(database_url: &str) -> bool {
    let url = database_url.trim();
    if url == "sqlite::memory:" || url == "sqlite://:memory:" {
        return true;
    }
    let query = url.split_once('?').map_or("", |(_, q)| q);
    query
        .split('&')
        .any(|param| param == "mode=memory" || param.starts_with("mode=memory&"))
}

/// Fetch column names for a `SQLite` table via `PRAGMA table_info`.
async fn table_column_names(pool: &SqlitePool, table: &str) -> Result<Vec<String>, StoreError> {
    let pragma = format!("PRAGMA table_info({table})");
    let rows = sqlx::query(AssertSqlSafe(pragma.as_str()))
        .fetch_all(pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;
    rows.iter()
        .map(|row| row.try_get::<String, _>("name"))
        .collect::<Result<_, _>>()
        .map_err(|e| StoreError::Database(e.to_string()))
}

/// Fetch ordered primary key columns for a `SQLite` table, each paired with
/// its declared type.
///
/// `PRAGMA table_info` reports `pk` as 0 for non-key columns and the
/// column's 1-based position within the primary key otherwise, so sorting
/// by it reconstructs the composite key in declaration order. The declared
/// type carries the column's affinity, which the index pragmas do not expose.
async fn primary_key_columns(pool: &SqlitePool, table: &str) -> Result<Vec<(String, String)>, StoreError> {
    let pragma = format!("PRAGMA table_info({table})");
    let rows = sqlx::query(AssertSqlSafe(pragma.as_str()))
        .fetch_all(pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;

    let mut key_columns: Vec<(i64, (String, String))> = Vec::new();
    for row in &rows {
        let position: i64 = row.try_get("pk").map_err(|e| StoreError::Database(e.to_string()))?;
        if position > 0 {
            let name: String = row.try_get("name").map_err(|e| StoreError::Database(e.to_string()))?;
            let declared_type: String = row.try_get("type").map_err(|e| StoreError::Database(e.to_string()))?;
            key_columns.push((position, (name, declared_type)));
        }
    }
    key_columns.sort_by_key(|(position, _)| *position);
    Ok(key_columns.into_iter().map(|(_, column)| column).collect())
}

/// Build a `SQLite` table's primary key columns with per-column folding
/// verdicts.
///
/// The declared type (carrying affinity) comes from `PRAGMA table_info` via
/// [`primary_key_columns`]; the collation governing each key column's equality
/// comes from the primary key's backing auto-index (`pragma_index_list` origin
/// `'pk'`, then `pragma_index_xinfo`). A single-column `INTEGER PRIMARY KEY`
/// aliases the rowid and has no auto-index, so it carries no collation and is
/// caught by the affinity half of [`sqlite_key_column_folding`].
async fn table_primary_key(pool: &SqlitePool, table: &str) -> Result<Vec<ActualKeyColumn>, StoreError> {
    let declared = primary_key_columns(pool, table).await?;
    let collations = match primary_key_index_name(pool, table).await? {
        Some(index) => index_key_collations(pool, &index).await?,
        None => Vec::new(),
    };
    Ok(declared
        .into_iter()
        .map(|(name, declared_type)| {
            let collation = collations
                .iter()
                .find(|(column, _)| column == &name)
                .and_then(|(_, collation)| collation.as_deref());
            ActualKeyColumn {
                folding: sqlite_key_column_folding(&declared_type, collation),
                name,
            }
        })
        .collect())
}

/// Fetch the name of a `SQLite` table's primary key auto-index, if any.
///
/// A composite (or any non-`INTEGER`) primary key is backed by an auto-index
/// reported by `pragma_index_list` with `origin = 'pk'`; a single-column
/// `INTEGER PRIMARY KEY` aliases the rowid and has none. The table name is
/// bound as a parameter so a quoted or special name does not break the query.
async fn primary_key_index_name(pool: &SqlitePool, table: &str) -> Result<Option<String>, StoreError> {
    let rows = sqlx::query("SELECT name, origin FROM pragma_index_list(?)")
        .bind(table)
        .fetch_all(pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;
    for row in &rows {
        let origin: String = row.try_get("origin").map_err(|e| StoreError::Database(e.to_string()))?;
        if origin == "pk" {
            return Ok(Some(
                row.try_get("name").map_err(|e| StoreError::Database(e.to_string()))?,
            ));
        }
    }
    Ok(None)
}

/// Fetch the key columns of a `SQLite` index paired with their collations via
/// the `pragma_index_xinfo` table-valued function.
///
/// The index name is bound as a parameter so a name that needs quoting does not
/// break the query. `index_xinfo` marks the columns named in the index with
/// `key = 1` and the auxiliary rowid/covering columns with `key = 0`, which are
/// skipped. A NULL name is an expression key column (never present on our
/// primary keys) and is also skipped. The collation is returned verbatim;
/// [`sqlite_key_column_folding`] decides whether it folds distinct values.
async fn index_key_collations(pool: &SqlitePool, index: &str) -> Result<Vec<(String, Option<String>)>, StoreError> {
    let rows = sqlx::query("SELECT name, coll, \"key\" AS is_key FROM pragma_index_xinfo(?)")
        .bind(index)
        .fetch_all(pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;
    let mut collations = Vec::new();
    for row in &rows {
        let is_key: i64 = row.try_get("is_key").map_err(|e| StoreError::Database(e.to_string()))?;
        if is_key == 0 {
            continue;
        }
        let name: Option<String> = row.try_get("name").map_err(|e| StoreError::Database(e.to_string()))?;
        let collation: Option<String> = row.try_get("coll").map_err(|e| StoreError::Database(e.to_string()))?;
        if let Some(name) = name {
            collations.push((name, collation));
        }
    }
    Ok(collations)
}

/// Fetch the unique indexes on a `SQLite` table other than the primary key's
/// auto-index, each with the columns it covers, via `pragma_index_list` and
/// `pragma_index_info`.
///
/// The table name is bound as a parameter so a quoted or special name does not
/// break the query. `pragma_index_list` reports every index with a `unique`
/// flag and an `origin` (`'pk'` for the primary key, `'u'` for a `UNIQUE`
/// constraint, `'c'` for a `CREATE UNIQUE INDEX`). Every unique index whose
/// origin is not `'pk'` is compared against the store's own generated unique
/// indexes by column set; anything else is rejected fail-closed rather than
/// proved a safe superset, since a narrower unique key loses rows via
/// `INSERT OR REPLACE`.
async fn table_extra_unique_indexes(pool: &SqlitePool, table: &str) -> Result<Vec<ActualUniqueIndex>, StoreError> {
    let rows = sqlx::query("SELECT name, \"unique\" AS is_unique, origin FROM pragma_index_list(?)")
        .bind(table)
        .fetch_all(pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;
    let mut indexes = Vec::new();
    for row in &rows {
        let is_unique: i64 = row
            .try_get("is_unique")
            .map_err(|e| StoreError::Database(e.to_string()))?;
        let origin: String = row.try_get("origin").map_err(|e| StoreError::Database(e.to_string()))?;
        if is_unique == 0 || origin == "pk" {
            continue;
        }
        let name: String = row.try_get("name").map_err(|e| StoreError::Database(e.to_string()))?;
        let columns = index_columns(pool, &name).await?;
        indexes.push(ActualUniqueIndex { name, columns });
    }
    indexes.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(indexes)
}

/// Fetch the columns a `SQLite` index covers, in index order, via
/// `pragma_index_info`.
///
/// The index name is bound as a parameter. `pragma_index_info` lists one row per
/// indexed column ordered by `seqno`, so the returned columns preserve the
/// index's declared order.
async fn index_columns(pool: &SqlitePool, index: &str) -> Result<Vec<String>, StoreError> {
    let rows = sqlx::query("SELECT name FROM pragma_index_info(?) ORDER BY seqno")
        .bind(index)
        .fetch_all(pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;
    let mut columns = Vec::with_capacity(rows.len());
    for row in &rows {
        let name: Option<String> = row.try_get("name").map_err(|e| StoreError::Database(e.to_string()))?;
        if let Some(name) = name {
            columns.push(name);
        }
    }
    Ok(columns)
}

/// Discover each tenant-scoped table's schema and compare it against the schema
/// this store generates before the schema version is stamped.
///
/// The single [`check_schema`] comparison fails closed on any deviation -- a
/// missing column, a primary key that is not the exact ordered contract, a key
/// column with a non-TEXT affinity or folding collation, or a unique index whose
/// column set is not one the store itself generates -- because
/// `CREATE TABLE IF NOT EXISTS` preserves a pre-existing table that could
/// silently lose data across tenants under `INSERT OR REPLACE`. The global schema
/// version table holds no tenant data and is validated by value in
/// [`check_schema_version`], not here.
async fn validate_schema(pool: &SqlitePool, tables: &TableNames) -> Result<(), StoreError> {
    let expected = expected_tables(tables);
    let mut actuals = Vec::with_capacity(expected.len());
    for (table_name, _) in &expected {
        actuals.push(ActualTable {
            columns: table_column_names(pool, table_name).await?,
            primary_key: table_primary_key(pool, table_name).await?,
            // SQLite has no deferrable constraints; a primary key is always immediate.
            primary_key_immediate: true,
            unique_indexes: table_extra_unique_indexes(pool, table_name).await?,
        });
    }

    let checks: Vec<SchemaCheck<'_>> = expected
        .iter()
        .zip(&actuals)
        .map(|((name, contract), actual)| (name.as_str(), *contract, actual))
        .collect();
    check_schema(&checks)
}

/// Stamp or validate the schema version.
async fn check_schema_version(pool: &SqlitePool, tables: &TableNames) -> Result<(), StoreError> {
    let vt = schema_version_table(&tables.responses);
    let select = format!("SELECT version FROM {vt}");
    let row: Option<i64> = sqlx::query_scalar(AssertSqlSafe(select.as_str()))
        .fetch_optional(pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;

    match row {
        None => {
            let insert = format!("INSERT OR IGNORE INTO {vt} (version) VALUES (?)");
            sqlx::query(AssertSqlSafe(insert.as_str()))
                .bind(SCHEMA_VERSION)
                .execute(pool)
                .await
                .map_err(|e| StoreError::Database(e.to_string()))?;
            Ok(())
        },
        Some(v) if v == SCHEMA_VERSION => Ok(()),
        Some(v) => Err(StoreError::Database(format!(
            "schema version mismatch in '{vt}': stored version {v}, \
             expected {SCHEMA_VERSION}; database migration required"
        ))),
    }
}

#[async_trait]
#[expect(
    clippy::too_many_lines,
    reason = "owner-scoped SQL methods keep all bindings explicit"
)]
impl ResponseStore for SqliteResponseStore {
    async fn upsert_response(&self, record: &ResponseRecord) -> Result<(), StoreError> {
        let response_object =
            serde_json::to_string(&record.response_object).map_err(|e| StoreError::Serialization(e.to_string()))?;
        let input = serde_json::to_string(&record.input).map_err(|e| StoreError::Serialization(e.to_string()))?;
        let messages = serde_json::to_string(&record.messages).map_err(|e| StoreError::Serialization(e.to_string()))?;

        let sql = format!(
            "INSERT INTO {} \
             (id, tenant_id, owner_issuer, owner_subject, created_at, model, response_object, input, messages) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) \
             ON CONFLICT(id) DO UPDATE SET created_at = excluded.created_at, model = excluded.model, \
             response_object = excluded.response_object, input = excluded.input, messages = excluded.messages \
             WHERE tenant_id = excluded.tenant_id AND owner_issuer = excluded.owner_issuer \
               AND owner_subject = excluded.owner_subject",
            self.tables.responses
        );

        let result = sqlx::query(AssertSqlSafe(sql.as_str()))
            .bind(&record.id)
            .bind(record.owner.tenant_id())
            .bind(record.owner.issuer())
            .bind(record.owner.subject())
            .bind(record.created_at)
            .bind(&record.model)
            .bind(&response_object)
            .bind(&input)
            .bind(&messages)
            .execute(&self.pool)
            .await
            .map_err(|e| StoreError::Database(e.to_string()))?;
        require_owner_preserving_write(result.rows_affected(), "response")
    }

    async fn get_response(&self, owner: &StateOwner, id: &str) -> Result<Option<ResponseRecord>, StoreError> {
        let sql = format!(
            "SELECT id, tenant_id, owner_issuer, owner_subject, created_at, model, \
                    response_object, input, messages \
             FROM {} \
             WHERE id = ? AND tenant_id = ? AND owner_issuer = ? AND owner_subject = ?",
            self.tables.responses
        );

        let row = sqlx::query(AssertSqlSafe(sql.as_str()))
            .bind(id)
            .bind(owner.tenant_id())
            .bind(owner.issuer())
            .bind(owner.subject())
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| StoreError::Database(e.to_string()))?;

        row.map(|r| row_to_response_record(&r)).transpose()
    }

    async fn delete_response(&self, owner: &StateOwner, id: &str) -> Result<bool, StoreError> {
        let delete_response_sql = format!(
            "DELETE FROM {} WHERE id = ? AND tenant_id = ? AND owner_issuer = ? AND owner_subject = ?",
            self.tables.responses
        );
        // Any pending approvals this response issued must go with it, so a
        // deleted response leaves no consumable approval behind and retains no
        // sensitive tool arguments. Both deletes commit atomically.
        let delete_approvals_sql = format!(
            "DELETE FROM {} WHERE response_id = ? AND tenant_id = ? AND owner_issuer = ? AND owner_subject = ?",
            pending_approvals_table(&self.tables.responses)
        );

        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| StoreError::Database(e.to_string()))?;

        let result = sqlx::query(AssertSqlSafe(delete_response_sql.as_str()))
            .bind(id)
            .bind(owner.tenant_id())
            .bind(owner.issuer())
            .bind(owner.subject())
            .execute(&mut *tx)
            .await
            .map_err(|e| StoreError::Database(e.to_string()))?;

        sqlx::query(AssertSqlSafe(delete_approvals_sql.as_str()))
            .bind(id)
            .bind(owner.tenant_id())
            .bind(owner.issuer())
            .bind(owner.subject())
            .execute(&mut *tx)
            .await
            .map_err(|e| StoreError::Database(e.to_string()))?;

        tx.commit().await.map_err(|e| StoreError::Database(e.to_string()))?;

        Ok(result.rows_affected() > 0)
    }

    async fn get_conversation(
        &self,
        owner: &StateOwner,
        conversation_id: &str,
    ) -> Result<Option<ConversationRecord>, StoreError> {
        self.get_conversation_record(owner, conversation_id).await
    }

    async fn record_pending_approvals(
        &self,
        owner: &StateOwner,
        response_id: &str,
        records: &[PendingApprovalRecord],
        created_at: i64,
    ) -> Result<(), StoreError> {
        if records.is_empty() {
            return Ok(());
        }
        let sql = pending_approval_insert_sql(&self.tables.responses);

        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| StoreError::Database(e.to_string()))?;

        for record in records {
            sqlx::query(AssertSqlSafe(sql.as_str()))
                .bind(owner.tenant_id())
                .bind(owner.issuer())
                .bind(owner.subject())
                .bind(response_id)
                .bind(&record.approval_id)
                .bind(&record.server_label)
                .bind(&record.tool_name)
                .bind(&record.arguments)
                .bind(&record.target_fingerprint)
                .bind(created_at)
                .bind(Option::<i64>::None)
                .bind(response_id)
                .bind(owner.tenant_id())
                .bind(owner.issuer())
                .bind(owner.subject())
                .execute(&mut *tx)
                .await
                .map_err(|e| StoreError::Database(e.to_string()))?;
        }

        tx.commit().await.map_err(|e| StoreError::Database(e.to_string()))?;
        Ok(())
    }

    #[expect(clippy::too_many_lines, reason = "sequential per-record bind within a transaction")]
    async fn persist_response_with_pending_approvals(
        &self,
        record: &ResponseRecord,
        pending_approvals: &[PendingApprovalRecord],
    ) -> Result<(), StoreError> {
        // No approvals: nothing to make atomic, so take the plain upsert path.
        if pending_approvals.is_empty() {
            return self.upsert_response(record).await;
        }

        let response_object =
            serde_json::to_string(&record.response_object).map_err(|e| StoreError::Serialization(e.to_string()))?;
        let input = serde_json::to_string(&record.input).map_err(|e| StoreError::Serialization(e.to_string()))?;
        let messages = serde_json::to_string(&record.messages).map_err(|e| StoreError::Serialization(e.to_string()))?;

        let upsert_sql = format!(
            "INSERT INTO {} \
             (id, tenant_id, owner_issuer, owner_subject, created_at, model, response_object, input, messages) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) \
             ON CONFLICT(id) DO UPDATE SET created_at = excluded.created_at, model = excluded.model, \
             response_object = excluded.response_object, input = excluded.input, messages = excluded.messages \
             WHERE tenant_id = excluded.tenant_id AND owner_issuer = excluded.owner_issuer \
               AND owner_subject = excluded.owner_subject",
            self.tables.responses
        );
        let approval_sql = pending_approval_insert_sql(&self.tables.responses);

        // One transaction so the response and its pending approvals commit
        // together or not at all. Serialized against delete_response, this closes
        // the window where a concurrent DELETE could land between the two writes
        // and orphan an approval row still holding the tool arguments.
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| StoreError::Database(e.to_string()))?;

        let result = sqlx::query(AssertSqlSafe(upsert_sql.as_str()))
            .bind(&record.id)
            .bind(record.owner.tenant_id())
            .bind(record.owner.issuer())
            .bind(record.owner.subject())
            .bind(record.created_at)
            .bind(&record.model)
            .bind(&response_object)
            .bind(&input)
            .bind(&messages)
            .execute(&mut *tx)
            .await
            .map_err(|e| StoreError::Database(e.to_string()))?;
        require_owner_preserving_write(result.rows_affected(), "response")?;

        for approval in pending_approvals {
            sqlx::query(AssertSqlSafe(approval_sql.as_str()))
                .bind(record.owner.tenant_id())
                .bind(record.owner.issuer())
                .bind(record.owner.subject())
                .bind(&record.id)
                .bind(&approval.approval_id)
                .bind(&approval.server_label)
                .bind(&approval.tool_name)
                .bind(&approval.arguments)
                .bind(&approval.target_fingerprint)
                .bind(record.created_at)
                .bind(Option::<i64>::None)
                .bind(&record.id)
                .bind(record.owner.tenant_id())
                .bind(record.owner.issuer())
                .bind(record.owner.subject())
                .execute(&mut *tx)
                .await
                .map_err(|e| StoreError::Database(e.to_string()))?;
        }

        tx.commit().await.map_err(|e| StoreError::Database(e.to_string()))?;
        Ok(())
    }

    async fn get_pending_approvals(
        &self,
        owner: &StateOwner,
        response_id: &str,
        approval_ids: &[&str],
    ) -> Result<Vec<PendingApprovalRecord>, StoreError> {
        if approval_ids.is_empty() {
            return Ok(Vec::new());
        }
        let table = pending_approvals_table(&self.tables.responses);
        let placeholders = std::iter::repeat_n("?", approval_ids.len())
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            "SELECT approval_id, server_label, tool_name, arguments, target_fingerprint \
             FROM {table} \
             WHERE tenant_id = ? AND owner_issuer = ? AND owner_subject = ? \
               AND response_id = ? AND approval_id IN ({placeholders})"
        );

        let mut query = sqlx::query(AssertSqlSafe(sql.as_str()))
            .bind(owner.tenant_id())
            .bind(owner.issuer())
            .bind(owner.subject())
            .bind(response_id);
        for approval_id in approval_ids {
            query = query.bind(*approval_id);
        }
        let rows = query
            .fetch_all(&self.pool)
            .await
            .map_err(|e| StoreError::Database(e.to_string()))?;

        rows.iter().map(row_to_pending_approval_record).collect()
    }

    async fn consume_approvals(
        &self,
        owner: &StateOwner,
        response_id: &str,
        approval_ids: &[&str],
        consumed_at: i64,
    ) -> Result<Option<usize>, StoreError> {
        if approval_ids.is_empty() {
            return Ok(None);
        }
        let table = pending_approvals_table(&self.tables.responses);
        // Conditional transition outstanding->consumed, scoped to the issuing
        // response. Callers looked the rows up first under the same response_id,
        // so a zero-row update means the approval was already consumed rather
        // than never issued.
        let sql = format!(
            "UPDATE {table} SET consumed_at = ? \
             WHERE tenant_id = ? AND owner_issuer = ? AND owner_subject = ? \
               AND response_id = ? AND approval_id = ? AND consumed_at IS NULL"
        );

        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| StoreError::Database(e.to_string()))?;

        for (index, approval_id) in approval_ids.iter().enumerate() {
            let result = sqlx::query(AssertSqlSafe(sql.as_str()))
                .bind(consumed_at)
                .bind(owner.tenant_id())
                .bind(owner.issuer())
                .bind(owner.subject())
                .bind(response_id)
                .bind(*approval_id)
                .execute(&mut *tx)
                .await
                .map_err(|e| StoreError::Database(e.to_string()))?;
            if result.rows_affected() == 0 {
                // Already consumed by a prior request, or a duplicate earlier
                // in this batch. Roll back so no id in the batch is claimed.
                tx.rollback().await.map_err(|e| StoreError::Database(e.to_string()))?;
                return Ok(Some(index));
            }
        }

        tx.commit().await.map_err(|e| StoreError::Database(e.to_string()))?;
        Ok(None)
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "async_trait counts the store method group as one expansion"
)]
#[async_trait]
impl ConversationItemStore for SqliteResponseStore {
    async fn upsert_conversation(&self, record: &ConversationRecord) -> Result<(), StoreError> {
        self.upsert_conversation_record(record).await
    }

    async fn update_conversation_messages(
        &self,
        owner: &StateOwner,
        conversation_id: &str,
        messages: &serde_json::Value,
    ) -> Result<bool, StoreError> {
        let messages = serde_json::to_string(messages).map_err(|e| StoreError::Serialization(e.to_string()))?;
        let sql = format!(
            "UPDATE {} SET messages = ? WHERE conversation_id = ? AND tenant_id = ? AND owner_issuer = ? AND owner_subject = ?",
            self.tables.conversations
        );

        let result = sqlx::query(AssertSqlSafe(sql.as_str()))
            .bind(&messages)
            .bind(conversation_id)
            .bind(owner.tenant_id())
            .bind(owner.issuer())
            .bind(owner.subject())
            .execute(&self.pool)
            .await
            .map_err(|e| StoreError::Database(e.to_string()))?;

        Ok(result.rows_affected() > 0)
    }

    async fn update_conversation_metadata(
        &self,
        owner: &StateOwner,
        conversation_id: &str,
        metadata: &serde_json::Value,
    ) -> Result<bool, StoreError> {
        let metadata = serde_json::to_string(metadata).map_err(|e| StoreError::Serialization(e.to_string()))?;
        let sql = format!(
            "UPDATE {} SET metadata = ? WHERE conversation_id = ? AND tenant_id = ? AND owner_issuer = ? AND owner_subject = ?",
            self.tables.conversations
        );

        let result = sqlx::query(AssertSqlSafe(sql.as_str()))
            .bind(&metadata)
            .bind(conversation_id)
            .bind(owner.tenant_id())
            .bind(owner.issuer())
            .bind(owner.subject())
            .execute(&self.pool)
            .await
            .map_err(|e| StoreError::Database(e.to_string()))?;

        Ok(result.rows_affected() > 0)
    }

    async fn compare_and_swap_conversation_messages(
        &self,
        owner: &StateOwner,
        conversation_id: &str,
        expected_messages: &serde_json::Value,
        messages: &serde_json::Value,
    ) -> Result<bool, StoreError> {
        let expected =
            serde_json::to_string(expected_messages).map_err(|e| StoreError::Serialization(e.to_string()))?;
        let messages = serde_json::to_string(messages).map_err(|e| StoreError::Serialization(e.to_string()))?;
        let sql = format!(
            "UPDATE {} SET messages = ? WHERE conversation_id = ? AND tenant_id = ? \
             AND owner_issuer = ? AND owner_subject = ? AND messages = ?",
            self.tables.conversations
        );
        let result = sqlx::query(AssertSqlSafe(sql.as_str()))
            .bind(&messages)
            .bind(conversation_id)
            .bind(owner.tenant_id())
            .bind(owner.issuer())
            .bind(owner.subject())
            .bind(&expected)
            .execute(&self.pool)
            .await
            .map_err(|e| StoreError::Database(e.to_string()))?;
        Ok(result.rows_affected() > 0)
    }

    async fn get_conversation(
        &self,
        owner: &StateOwner,
        conversation_id: &str,
    ) -> Result<Option<ConversationRecord>, StoreError> {
        self.get_conversation_record(owner, conversation_id).await
    }

    async fn delete_conversation(&self, owner: &StateOwner, conversation_id: &str) -> Result<bool, StoreError> {
        self.delete_conversation_record(owner, conversation_id).await
    }

    async fn create_conversation_items(&self, items: &[ConversationItemRecord]) -> Result<(), StoreError> {
        let table = self
            .tables
            .items
            .as_deref()
            .ok_or_else(|| StoreError::Unavailable("items table not configured".to_owned()))?;

        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| StoreError::Database(e.to_string()))?;

        let sql = format!(
            "INSERT INTO {table} \
             (item_id, tenant_id, owner_issuer, owner_subject, conversation_id, item_data, created_at, position) \
             SELECT ?, tenant_id, owner_issuer, owner_subject, conversation_id, ?, ?, ? \
             FROM {} \
             WHERE conversation_id = ? AND tenant_id = ? AND owner_issuer = ? AND owner_subject = ?",
            self.tables.conversations
        );

        for item in items {
            let item_data =
                serde_json::to_string(&item.item_data).map_err(|e| StoreError::Serialization(e.to_string()))?;

            let result = sqlx::query(AssertSqlSafe(sql.as_str()))
                .bind(&item.item_id)
                .bind(&item_data)
                .bind(item.created_at)
                .bind(item.position)
                .bind(&item.conversation_id)
                .bind(item.owner.tenant_id())
                .bind(item.owner.issuer())
                .bind(item.owner.subject())
                .execute(&mut *tx)
                .await
                .map_err(|e| StoreError::Database(e.to_string()))?;
            if result.rows_affected() != 1 {
                return Err(StoreError::Database(
                    "conversation item owner does not match its parent".to_owned(),
                ));
            }
        }

        tx.commit().await.map_err(|e| StoreError::Database(e.to_string()))?;
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
        let table = self
            .tables
            .items
            .as_deref()
            .ok_or_else(|| StoreError::Unavailable("items table not configured".to_owned()))?;

        let direction = if ascending { "ASC" } else { "DESC" };
        let cursor_operator = if ascending { ">" } else { "<" };

        let rows = if let Some(item_id) = after_item_id {
            let Some(position) = self.conversation_item_position(owner, conversation_id, item_id).await? else {
                return Ok(Vec::new());
            };
            let sql = format!(
                "SELECT item_id, tenant_id, owner_issuer, owner_subject, conversation_id, item_data, created_at, position \
                 FROM {table} \
                 WHERE tenant_id = ? AND owner_issuer = ? AND owner_subject = ? AND conversation_id = ? \
                   AND (position {cursor_operator} ? \
                        OR (position = ? AND item_id {cursor_operator} ?)) \
                 ORDER BY position {direction}, item_id {direction} \
                 LIMIT ?"
            );
            sqlx::query(AssertSqlSafe(sql.as_str()))
                .bind(owner.tenant_id())
                .bind(owner.issuer())
                .bind(owner.subject())
                .bind(conversation_id)
                .bind(position)
                .bind(position)
                .bind(item_id)
                .bind(limit)
                .fetch_all(&self.pool)
                .await
                .map_err(|e| StoreError::Database(e.to_string()))?
        } else {
            let sql = format!(
                "SELECT item_id, tenant_id, owner_issuer, owner_subject, conversation_id, item_data, created_at, position \
                 FROM {table} \
                 WHERE tenant_id = ? AND owner_issuer = ? AND owner_subject = ? AND conversation_id = ? \
                 ORDER BY position {direction}, item_id {direction} \
                 LIMIT ?"
            );
            sqlx::query(AssertSqlSafe(sql.as_str()))
                .bind(owner.tenant_id())
                .bind(owner.issuer())
                .bind(owner.subject())
                .bind(conversation_id)
                .bind(limit)
                .fetch_all(&self.pool)
                .await
                .map_err(|e| StoreError::Database(e.to_string()))?
        };

        rows.iter().map(row_to_conversation_item_record).collect()
    }

    async fn get_existing_conversation_item_ids(
        &self,
        owner: &StateOwner,
        conversation_id: &str,
        item_ids: &[&str],
    ) -> Result<Vec<String>, StoreError> {
        let table = self
            .tables
            .items
            .as_deref()
            .ok_or_else(|| StoreError::Unavailable("items table not configured".to_owned()))?;

        if item_ids.is_empty() {
            return Ok(Vec::new());
        }

        let placeholders: String = std::iter::repeat_n("?", item_ids.len()).collect::<Vec<_>>().join(", ");
        let sql = format!(
            "SELECT item_id FROM {table} \
             WHERE tenant_id = ? AND owner_issuer = ? AND owner_subject = ? \
               AND conversation_id = ? AND item_id IN ({placeholders})"
        );

        let mut query = sqlx::query_scalar::<_, String>(AssertSqlSafe(sql.as_str()))
            .bind(owner.tenant_id())
            .bind(owner.issuer())
            .bind(owner.subject())
            .bind(conversation_id);
        for id in item_ids {
            query = query.bind(*id);
        }

        query
            .fetch_all(&self.pool)
            .await
            .map_err(|e| StoreError::Database(e.to_string()))
    }

    async fn get_conversation_item(
        &self,
        owner: &StateOwner,
        conversation_id: &str,
        item_id: &str,
    ) -> Result<Option<ConversationItemRecord>, StoreError> {
        let table = self
            .tables
            .items
            .as_deref()
            .ok_or_else(|| StoreError::Unavailable("items table not configured".to_owned()))?;

        let sql = format!(
            "SELECT item_id, tenant_id, owner_issuer, owner_subject, conversation_id, item_data, created_at, position \
             FROM {table} \
             WHERE item_id = ? AND tenant_id = ? AND owner_issuer = ? AND owner_subject = ? AND conversation_id = ?"
        );

        let row = sqlx::query(AssertSqlSafe(sql.as_str()))
            .bind(item_id)
            .bind(owner.tenant_id())
            .bind(owner.issuer())
            .bind(owner.subject())
            .bind(conversation_id)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| StoreError::Database(e.to_string()))?;

        row.map(|r| row_to_conversation_item_record(&r)).transpose()
    }

    async fn delete_conversation_item(
        &self,
        owner: &StateOwner,
        conversation_id: &str,
        item_id: &str,
    ) -> Result<bool, StoreError> {
        let table = self
            .tables
            .items
            .as_deref()
            .ok_or_else(|| StoreError::Unavailable("items table not configured".to_owned()))?;

        let sql = format!(
            "DELETE FROM {table} WHERE item_id = ? AND tenant_id = ? AND owner_issuer = ? \
             AND owner_subject = ? AND conversation_id = ?"
        );

        let result = sqlx::query(AssertSqlSafe(sql.as_str()))
            .bind(item_id)
            .bind(owner.tenant_id())
            .bind(owner.issuer())
            .bind(owner.subject())
            .bind(conversation_id)
            .execute(&self.pool)
            .await
            .map_err(|e| StoreError::Database(e.to_string()))?;

        Ok(result.rows_affected() > 0)
    }

    async fn conversation_item_position(
        &self,
        owner: &StateOwner,
        conversation_id: &str,
        item_id: &str,
    ) -> Result<Option<i64>, StoreError> {
        let table = self
            .tables
            .items
            .as_deref()
            .ok_or_else(|| StoreError::Unavailable("items table not configured".to_owned()))?;

        let sql = format!(
            "SELECT position FROM {table} \
             WHERE item_id = ? AND tenant_id = ? AND owner_issuer = ? AND owner_subject = ? AND conversation_id = ?"
        );

        let row = sqlx::query(AssertSqlSafe(sql.as_str()))
            .bind(item_id)
            .bind(owner.tenant_id())
            .bind(owner.issuer())
            .bind(owner.subject())
            .bind(conversation_id)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| StoreError::Database(e.to_string()))?;

        row.map(|r| r.try_get("position").map_err(|e| StoreError::Database(e.to_string())))
            .transpose()
    }

    async fn max_item_position(&self, owner: &StateOwner, conversation_id: &str) -> Result<i64, StoreError> {
        let table = self
            .tables
            .items
            .as_deref()
            .ok_or_else(|| StoreError::Unavailable("items table not configured".to_owned()))?;

        let sql = format!(
            "SELECT COALESCE(MAX(position), 0) AS max_pos \
             FROM {table} \
             WHERE tenant_id = ? AND owner_issuer = ? AND owner_subject = ? AND conversation_id = ?"
        );

        let row = sqlx::query(AssertSqlSafe(sql.as_str()))
            .bind(owner.tenant_id())
            .bind(owner.issuer())
            .bind(owner.subject())
            .bind(conversation_id)
            .fetch_one(&self.pool)
            .await
            .map_err(|e| StoreError::Database(e.to_string()))?;

        row.try_get("max_pos").map_err(|e| StoreError::Database(e.to_string()))
    }

    async fn create_items_and_sync_messages(
        &self,
        owner: &StateOwner,
        conversation_id: &str,
        items: &[ConversationItemRecord],
    ) -> Result<(), StoreError> {
        if items.is_empty() {
            return Ok(());
        }
        require_matching_item_scope(owner, conversation_id, items)?;

        let items_table = self
            .tables
            .items
            .as_deref()
            .ok_or_else(|| StoreError::Unavailable("items table not configured".to_owned()))?;
        let conv_table = &self.tables.conversations;

        let mut tx = self
            .pool
            .begin_with("BEGIN IMMEDIATE")
            .await
            .map_err(|e| StoreError::Database(e.to_string()))?;

        sqlite_create_items_and_sync(&mut tx, items_table, conv_table, owner, conversation_id, items).await?;

        tx.commit().await.map_err(|e| StoreError::Database(e.to_string()))?;
        Ok(())
    }

    async fn delete_item_and_sync_messages(
        &self,
        owner: &StateOwner,
        conversation_id: &str,
        item_id: &str,
    ) -> Result<bool, StoreError> {
        let items_table = self
            .tables
            .items
            .as_deref()
            .ok_or_else(|| StoreError::Unavailable("items table not configured".to_owned()))?;
        let conv_table = &self.tables.conversations;

        let mut tx = self
            .pool
            .begin_with("BEGIN IMMEDIATE")
            .await
            .map_err(|e| StoreError::Database(e.to_string()))?;

        let deleted =
            sqlite_delete_item_and_sync(&mut tx, items_table, conv_table, owner, conversation_id, item_id).await?;

        tx.commit().await.map_err(|e| StoreError::Database(e.to_string()))?;
        Ok(deleted)
    }
}

// -----------------------------------------------------------------------------
// Transactional Helpers
// -----------------------------------------------------------------------------

/// Body of [`SqliteResponseStore::create_items_and_sync_messages`].
///
/// Runs inside a `BEGIN IMMEDIATE` transaction managed by the caller.
#[expect(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "transactional helper that threads table names and scope identifiers"
)]
async fn sqlite_create_items_and_sync(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    items_table: &str,
    conv_table: &str,
    owner: &StateOwner,
    conversation_id: &str,
    items: &[ConversationItemRecord],
) -> Result<(), StoreError> {
    let max_sql = format!(
        "SELECT COALESCE(MAX(position), 0) AS max_pos \
         FROM {items_table} \
         WHERE tenant_id = ? AND owner_issuer = ? AND owner_subject = ? AND conversation_id = ?"
    );
    let max_row = sqlx::query(AssertSqlSafe(max_sql.as_str()))
        .bind(owner.tenant_id())
        .bind(owner.issuer())
        .bind(owner.subject())
        .bind(conversation_id)
        .fetch_one(&mut **tx)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;
    let max_pos: i64 = max_row
        .try_get("max_pos")
        .map_err(|e| StoreError::Database(e.to_string()))?;

    let insert_sql = format!(
        "INSERT INTO {items_table} \
         (item_id, tenant_id, owner_issuer, owner_subject, conversation_id, item_data, created_at, position) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
    );
    for (i, item) in items.iter().enumerate() {
        let offset = i64::try_from(i).unwrap_or(i64::MAX);
        let position = max_pos.saturating_add(1).saturating_add(offset);
        let item_data = serde_json::to_string(&item.item_data).map_err(|e| StoreError::Serialization(e.to_string()))?;

        sqlx::query(AssertSqlSafe(insert_sql.as_str()))
            .bind(&item.item_id)
            .bind(owner.tenant_id())
            .bind(owner.issuer())
            .bind(owner.subject())
            .bind(conversation_id)
            .bind(&item_data)
            .bind(item.created_at)
            .bind(position)
            .execute(&mut **tx)
            .await
            .map_err(|e| StoreError::Database(e.to_string()))?;
    }

    sqlite_rebuild_messages(tx, items_table, conv_table, owner, conversation_id).await
}

/// Body of [`SqliteResponseStore::delete_item_and_sync_messages`].
#[expect(
    clippy::too_many_arguments,
    reason = "transactional helper that threads table names and scope identifiers"
)]
async fn sqlite_delete_item_and_sync(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    items_table: &str,
    conv_table: &str,
    owner: &StateOwner,
    conversation_id: &str,
    item_id: &str,
) -> Result<bool, StoreError> {
    let delete_sql = format!(
        "DELETE FROM {items_table} WHERE item_id = ? AND tenant_id = ? AND owner_issuer = ? \
         AND owner_subject = ? AND conversation_id = ?"
    );
    let result = sqlx::query(AssertSqlSafe(delete_sql.as_str()))
        .bind(item_id)
        .bind(owner.tenant_id())
        .bind(owner.issuer())
        .bind(owner.subject())
        .bind(conversation_id)
        .execute(&mut **tx)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;

    if result.rows_affected() == 0 {
        return Ok(false);
    }

    sqlite_rebuild_messages(tx, items_table, conv_table, owner, conversation_id).await?;
    Ok(true)
}

/// Read all item JSON values and overwrite the conversation message cache.
///
/// Returns [`StoreError::Database`] if the conversation row is gone by
/// the time the cache is written — i.e. it was deleted concurrently
/// between the caller's existence check and this transaction. Callers
/// run this inside a transaction, so the propagated error rolls back
/// any item mutations made in the same transaction.
#[expect(clippy::too_many_lines, reason = "sequential query pipeline within a transaction")]
async fn sqlite_rebuild_messages(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    items_table: &str,
    conv_table: &str,
    owner: &StateOwner,
    conversation_id: &str,
) -> Result<(), StoreError> {
    let select_sql = format!(
        "SELECT item_data FROM {items_table} \
         WHERE tenant_id = ? AND owner_issuer = ? AND owner_subject = ? AND conversation_id = ? \
         ORDER BY position ASC, item_id ASC"
    );
    let rows = sqlx::query(AssertSqlSafe(select_sql.as_str()))
        .bind(owner.tenant_id())
        .bind(owner.issuer())
        .bind(owner.subject())
        .bind(conversation_id)
        .fetch_all(&mut **tx)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;

    let mut messages = Vec::with_capacity(rows.len());
    for row in &rows {
        let json: String = row
            .try_get("item_data")
            .map_err(|e| StoreError::Database(e.to_string()))?;
        let value: serde_json::Value =
            serde_json::from_str(&json).map_err(|e| StoreError::Serialization(e.to_string()))?;
        messages.push(value);
    }

    let messages_json = serde_json::to_string(&serde_json::Value::Array(messages))
        .map_err(|e| StoreError::Serialization(e.to_string()))?;

    let update_sql = format!(
        "UPDATE {conv_table} SET messages = ? \
         WHERE conversation_id = ? AND tenant_id = ? AND owner_issuer = ? AND owner_subject = ?"
    );
    let updated = sqlx::query(AssertSqlSafe(update_sql.as_str()))
        .bind(&messages_json)
        .bind(conversation_id)
        .bind(owner.tenant_id())
        .bind(owner.issuer())
        .bind(owner.subject())
        .execute(&mut **tx)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;

    if updated.rows_affected() == 0 {
        return Err(StoreError::Database(format!(
            "conversation disappeared during message sync: {conversation_id}"
        )));
    }

    Ok(())
}

// -----------------------------------------------------------------------------
// Row Conversion
// -----------------------------------------------------------------------------

/// Build the insert-if-absent SQL for recording pending approvals.
///
/// `ON CONFLICT DO NOTHING` guarantees a re-emit never resets `consumed_at` back
/// to outstanding, so an already-consumed approval can never be replayed.
fn pending_approval_insert_sql(responses_table: &str) -> String {
    let table = pending_approvals_table(responses_table);
    format!(
        "INSERT INTO {table} \
         (tenant_id, owner_issuer, owner_subject, response_id, approval_id, server_label, tool_name, arguments, \
         target_fingerprint, created_at, consumed_at) \
         SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ? \
         WHERE EXISTS (SELECT 1 FROM {responses_table} WHERE id = ? AND tenant_id = ? \
           AND owner_issuer = ? AND owner_subject = ?) \
         ON CONFLICT (response_id, approval_id) DO NOTHING"
    )
}

/// Turn an owner-filtered upsert no-op into a bounded, identity-free error.
fn require_owner_preserving_write(rows_affected: u64, resource: &str) -> Result<(), StoreError> {
    if rows_affected == 1 {
        Ok(())
    } else {
        tracing::warn!(resource, "resource id is already owned by another principal");
        Err(StoreError::Database(format!("{resource} id collision")))
    }
}

/// Require every transactional item to match its authorized parent scope.
fn require_matching_item_scope(
    owner: &StateOwner,
    conversation_id: &str,
    items: &[ConversationItemRecord],
) -> Result<(), StoreError> {
    if items
        .iter()
        .all(|item| &item.owner == owner && item.conversation_id == conversation_id)
    {
        Ok(())
    } else {
        Err(StoreError::InvalidInput(
            "conversation item scope does not match its parent".to_owned(),
        ))
    }
}

/// Convert a sqlx row to a [`PendingApprovalRecord`].
fn row_to_pending_approval_record(row: &sqlx::sqlite::SqliteRow) -> Result<PendingApprovalRecord, StoreError> {
    Ok(PendingApprovalRecord {
        approval_id: row
            .try_get("approval_id")
            .map_err(|e| StoreError::Database(e.to_string()))?,
        server_label: row
            .try_get("server_label")
            .map_err(|e| StoreError::Database(e.to_string()))?,
        tool_name: row
            .try_get("tool_name")
            .map_err(|e| StoreError::Database(e.to_string()))?,
        arguments: row
            .try_get("arguments")
            .map_err(|e| StoreError::Database(e.to_string()))?,
        target_fingerprint: row
            .try_get("target_fingerprint")
            .map_err(|e| StoreError::Database(e.to_string()))?,
    })
}

/// Convert a sqlx row to a [`ResponseRecord`].
fn row_to_response_record(row: &sqlx::sqlite::SqliteRow) -> Result<ResponseRecord, StoreError> {
    let response_object_json: String = row
        .try_get("response_object")
        .map_err(|e| StoreError::Database(e.to_string()))?;
    let input_json: String = row.try_get("input").map_err(|e| StoreError::Database(e.to_string()))?;
    let messages_json: String = row
        .try_get("messages")
        .map_err(|e| StoreError::Database(e.to_string()))?;

    Ok(ResponseRecord {
        id: row.try_get("id").map_err(|e| StoreError::Database(e.to_string()))?,
        owner: row_to_owner(row)?,
        created_at: row
            .try_get("created_at")
            .map_err(|e| StoreError::Database(e.to_string()))?,
        model: row.try_get("model").map_err(|e| StoreError::Database(e.to_string()))?,
        response_object: serde_json::from_str(&response_object_json)
            .map_err(|e| StoreError::Serialization(e.to_string()))?,
        input: serde_json::from_str(&input_json).map_err(|e| StoreError::Serialization(e.to_string()))?,
        messages: serde_json::from_str(&messages_json).map_err(|e| StoreError::Serialization(e.to_string()))?,
    })
}

/// Convert a sqlx row to a [`ConversationItemRecord`].
fn row_to_conversation_item_record(row: &sqlx::sqlite::SqliteRow) -> Result<ConversationItemRecord, StoreError> {
    let item_data_json: String = row
        .try_get("item_data")
        .map_err(|e| StoreError::Database(e.to_string()))?;

    Ok(ConversationItemRecord {
        item_id: row
            .try_get("item_id")
            .map_err(|e| StoreError::Database(e.to_string()))?,
        owner: row_to_owner(row)?,
        conversation_id: row
            .try_get("conversation_id")
            .map_err(|e| StoreError::Database(e.to_string()))?,
        item_data: serde_json::from_str(&item_data_json).map_err(|e| StoreError::Serialization(e.to_string()))?,
        created_at: row
            .try_get("created_at")
            .map_err(|e| StoreError::Database(e.to_string()))?,
        position: row
            .try_get("position")
            .map_err(|e| StoreError::Database(e.to_string()))?,
    })
}

/// Convert a sqlx row to a [`ConversationRecord`].
fn row_to_conversation_record(row: &sqlx::sqlite::SqliteRow) -> Result<ConversationRecord, StoreError> {
    let messages_json: String = row
        .try_get("messages")
        .map_err(|e| StoreError::Database(e.to_string()))?;
    let metadata_json: String = row
        .try_get("metadata")
        .map_err(|e| StoreError::Database(e.to_string()))?;

    Ok(ConversationRecord {
        conversation_id: row
            .try_get("conversation_id")
            .map_err(|e| StoreError::Database(e.to_string()))?,
        owner: row_to_owner(row)?,
        created_at: row
            .try_get("created_at")
            .map_err(|e| StoreError::Database(e.to_string()))?,
        metadata: serde_json::from_str(&metadata_json).map_err(|e| StoreError::Serialization(e.to_string()))?,
        messages: serde_json::from_str(&messages_json).map_err(|e| StoreError::Serialization(e.to_string()))?,
    })
}

/// Decode and validate the immutable owner columns in a persisted row.
fn row_to_owner(row: &sqlx::sqlite::SqliteRow) -> Result<StateOwner, StoreError> {
    let tenant_id: String = row
        .try_get("tenant_id")
        .map_err(|e| StoreError::Database(e.to_string()))?;
    let issuer: String = row
        .try_get("owner_issuer")
        .map_err(|e| StoreError::Database(e.to_string()))?;
    let subject: String = row
        .try_get("owner_subject")
        .map_err(|e| StoreError::Database(e.to_string()))?;
    StateOwner::from_trusted_parts(tenant_id, issuer, subject)
        .map_err(|e| StoreError::Database(format!("invalid persisted owner: {e}")))
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, reason = "tests")]
mod tests {
    use super::*;

    #[test]
    fn memory_url_short_form() {
        assert!(
            is_memory_database_url("sqlite::memory:"),
            "short-form memory URL should be detected"
        );
    }

    #[test]
    fn memory_url_slash_form() {
        assert!(
            is_memory_database_url("sqlite://:memory:"),
            "slash-form memory URL should be detected"
        );
    }

    #[test]
    fn memory_url_query_param() {
        assert!(
            is_memory_database_url("sqlite:///test.db?mode=memory"),
            "mode=memory query param should be detected"
        );
    }

    #[test]
    fn memory_url_query_param_not_first() {
        assert!(
            is_memory_database_url("sqlite:///test.db?cache=shared&mode=memory"),
            "mode=memory should be detected even when not the first query param"
        );
    }

    #[test]
    fn memory_url_whitespace_trimmed() {
        assert!(
            is_memory_database_url("  sqlite::memory:  "),
            "leading/trailing whitespace should be trimmed"
        );
    }

    #[test]
    fn file_url_is_not_memory() {
        assert!(
            !is_memory_database_url("sqlite:///path/to/db.sqlite"),
            "file-backed URL should not be detected as memory"
        );
    }

    #[test]
    fn file_url_with_mode_rwc_is_not_memory() {
        assert!(
            !is_memory_database_url("sqlite:///test.db?mode=rwc"),
            "mode=rwc should not be detected as memory"
        );
    }

    #[test]
    fn empty_url_is_not_memory() {
        assert!(
            !is_memory_database_url(""),
            "empty URL should not be detected as memory"
        );
    }
}
