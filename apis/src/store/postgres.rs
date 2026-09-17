// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! [`PostgresResponseStore`] — `PostgreSQL` backend for the response store.

use std::path::Path;

use async_trait::async_trait;
use sqlx::{
    AssertSqlSafe, Row as _,
    postgres::{PgConnectOptions, PgPoolOptions, PgRow, PgSslMode},
};
use tracing::info;

use super::{
    SslMode,
    pool::{PoolConfig, apply_pool_config},
    schemas::{
        ActualKeyColumn, ActualTable, ActualUniqueIndex, SCHEMA_VERSION, SchemaCheck, TableNames, check_schema,
        expected_tables, generate_ddl, pending_approvals_table, pg_key_column_folding, schema_version_table,
        validate_postgres_identifiers,
    },
    trait_def::{ConversationItemStore, ResponseStore},
    types::{ConversationItemRecord, ConversationRecord, PendingApprovalRecord, ResponseRecord, StoreError},
};
use crate::StateOwner;

impl From<SslMode> for PgSslMode {
    fn from(mode: SslMode) -> Self {
        match mode {
            SslMode::Disable => Self::Disable,
            SslMode::Prefer => Self::Prefer,
            SslMode::Require => Self::Require,
            SslMode::VerifyCa => Self::VerifyCa,
            SslMode::VerifyFull => Self::VerifyFull,
        }
    }
}

// -----------------------------------------------------------------------------
// PostgresResponseStore
// -----------------------------------------------------------------------------

/// PostgreSQL-backed response store.
///
/// Uses [`sqlx::PgPool`] for async connection pooling. Table names
/// are configurable per provider (e.g., `openai_responses`,
/// `google_interactions`) to isolate data per provider.
pub struct PostgresResponseStore {
    /// Connection pool.
    pool: sqlx::PgPool,
    /// Configured table names.
    tables: TableNames,
}

impl PostgresResponseStore {
    /// Create a new store and initialize the schema.
    ///
    /// The `database_url` is a `PostgreSQL` connection string
    /// (e.g., `"postgres://user:pass@host:5432/praxis"`).
    ///
    /// `responses_table` and `conversations_table` are the SQL
    /// table names to use. These come from the filter's YAML
    /// config (e.g., `openai_responses`).
    ///
    /// `items_table`, when provided, enables the conversation items
    /// table for storing individual conversation entries.
    ///
    /// `ssl_mode` always overrides any `sslmode` in the URL —
    /// explicitly when provided, or with the [`SslMode::VerifyFull`]
    /// default when omitted. Use [`SslMode::VerifyCa`] or [`SslMode::VerifyFull`]
    /// with `ssl_root_cert` to verify the server against a custom
    /// CA. Certificate path existence is validated at connection
    /// time, not at construction.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::Database`] if the connection, schema
    /// initialization, or table name validation fails.
    #[expect(
        clippy::too_many_arguments,
        reason = "constructor mirrors SqliteResponseStore::new with SSL and pool additions"
    )]
    pub async fn new(
        database_url: &str,
        responses_table: &str,
        conversations_table: &str,
        items_table: Option<&str>,
        ssl_mode: Option<SslMode>,
        ssl_root_cert: Option<&str>,
        pool_config: Option<&PoolConfig>,
    ) -> Result<Self, StoreError> {
        let tables = TableNames {
            responses: responses_table.to_owned(),
            conversations: conversations_table.to_owned(),
            items: items_table.map(str::to_owned),
        };
        validate_postgres_identifiers(&tables)?;
        let ddl = generate_ddl(&tables)?;

        let options = pg_connect_options(database_url, ssl_mode, ssl_root_cert)?;
        let pool = Box::pin(apply_pool_config(PgPoolOptions::new(), pool_config).connect_with(options))
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
            "postgres response store initialized"
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
             VALUES ($1, $2, $3, $4, $5, $6, $7) \
             ON CONFLICT (conversation_id) DO UPDATE SET \
             messages = EXCLUDED.messages, \
             metadata = EXCLUDED.metadata \
             WHERE {}.tenant_id = EXCLUDED.tenant_id \
               AND {}.owner_issuer = EXCLUDED.owner_issuer \
               AND {}.owner_subject = EXCLUDED.owner_subject",
            self.tables.conversations, self.tables.conversations, self.tables.conversations, self.tables.conversations
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
            "SELECT conversation_id, tenant_id, owner_issuer, owner_subject, created_at, \
                    metadata, messages \
             FROM {} \
             WHERE conversation_id = $1 AND tenant_id = $2 AND owner_issuer = $3 AND owner_subject = $4",
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
            "DELETE FROM {} WHERE conversation_id = $1 AND tenant_id = $2 AND owner_issuer = $3 AND owner_subject = $4",
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

/// Build `PostgreSQL` connection options from URL and optional TLS overrides.
///
/// Always applies an SSL mode: the explicit override when provided,
/// otherwise [`SslMode::VerifyFull`]. This overrides any `sslmode`
/// embedded in the URL to ensure TLS-verified connections by default.
fn pg_connect_options(
    database_url: &str,
    ssl_mode: Option<SslMode>,
    ssl_root_cert: Option<&str>,
) -> Result<PgConnectOptions, StoreError> {
    let mut options: PgConnectOptions = database_url
        .parse()
        .map_err(|e: sqlx::Error| StoreError::Database(e.to_string()))?;

    let effective_mode = ssl_mode.unwrap_or_default();
    options = options.ssl_mode(PgSslMode::from(effective_mode));

    if let Some(cert_path) = ssl_root_cert {
        options = options.ssl_root_cert(Path::new(cert_path));
    }

    Ok(options)
}

/// Fetch column names for a `PostgreSQL` table from `information_schema`.
async fn table_column_names(pool: &sqlx::PgPool, table: &str) -> Result<Vec<String>, StoreError> {
    sqlx::query_scalar::<_, String>(
        "SELECT column_name FROM information_schema.columns \
         WHERE table_schema = current_schema() AND table_name = $1",
    )
    .bind(table)
    .fetch_all(pool)
    .await
    .map_err(|e| StoreError::Database(e.to_string()))
}

/// Fetch a `PostgreSQL` table's primary key columns with the metadata needed to
/// prove each preserves distinct key values, plus whether the key is immediate.
///
/// Reading the key from `pg_index.indisprimary` scoped to `t.relname = $1`
/// returns only *this* table's own primary key, so a same-named constraint on
/// another table cannot leak in. Only the leading `indnkeyatts` attributes are
/// key columns, excluding any `PRIMARY KEY ... INCLUDE (...)` covering column; a
/// primary key never contains an expression column, so `attname` and `typname`
/// are always present.
///
/// Each key column carries: its type OID (`pg_attribute.atttypid`, read without
/// resolving domains -- a domain reports its own OID and fails the allow-list
/// with no recursive walk); whether its collation is deterministic
/// (`pg_collation.collisdeterministic`, `NULL` for a non-collatable column); and
/// whether its operator class is the built-in default B-tree class for the
/// column type -- keyed structurally on `pg_opclass.opcnamespace = 'pg_catalog'`,
/// `pg_am.amname = 'btree'`, `pg_opclass.opcdefault`, and `pg_opclass.opcintype =
/// text` (see [`PRIMARY_KEY_QUERY`] for why the input type is `text`, not the
/// column's own OID), never on a spoofable class name, and read as `NULL`/untrusted
/// fail-closed. [`pg_key_column_folding`] turns that metadata into a verdict.
/// `pg_index.indimmediate` is constant across the index's rows; a deferrable key
/// cannot arbitrate an `ON CONFLICT` upsert.
async fn table_primary_key(pool: &sqlx::PgPool, table: &str) -> Result<(Vec<ActualKeyColumn>, bool), StoreError> {
    let rows = sqlx::query(PRIMARY_KEY_QUERY)
        .bind(table)
        .fetch_all(pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;

    let mut columns = Vec::with_capacity(rows.len());
    // A table with no primary key yields no rows; report it as immediate so the
    // empty key fails the shape check rather than the deferrable check.
    let mut immediate = true;
    for row in &rows {
        let (column, row_immediate) = pk_row_to_key_column(row)?;
        immediate = row_immediate;
        columns.push(column);
    }
    Ok((columns, immediate))
}

/// Parse one row from [`PRIMARY_KEY_QUERY`] into a key column plus the index's
/// `indimmediate` flag. That flag is index-level, so it is identical for every
/// row of the key; the caller keeps the last one read.
fn pk_row_to_key_column(row: &PgRow) -> Result<(ActualKeyColumn, bool), StoreError> {
    let name: String = row
        .try_get("column_name")
        .map_err(|e| StoreError::Database(e.to_string()))?;
    let type_oid: i64 = row
        .try_get("type_oid")
        .map_err(|e| StoreError::Database(e.to_string()))?;
    let type_name: String = row
        .try_get("type_name")
        .map_err(|e| StoreError::Database(e.to_string()))?;
    let immediate: bool = row
        .try_get("immediate")
        .map_err(|e| StoreError::Database(e.to_string()))?;
    let collation_deterministic: Option<bool> = row
        .try_get("collation_deterministic")
        .map_err(|e| StoreError::Database(e.to_string()))?;
    // A missing operator class classifies as untrusted (fail-closed).
    let operator_class_trusted: Option<bool> = row
        .try_get("operator_class_trusted")
        .map_err(|e| StoreError::Database(e.to_string()))?;
    let column = ActualKeyColumn {
        folding: pg_key_column_folding(
            type_oid,
            &type_name,
            collation_deterministic,
            operator_class_trusted.unwrap_or(false),
        ),
        name,
    };
    Ok((column, immediate))
}

/// Catalog query backing [`table_primary_key`]. `$1` binds the table name; see
/// that function's docs for how each selected column is interpreted.
///
/// `operator_class_trusted` is true only for `text_ops`, the `pg_catalog` default
/// B-tree operator class whose input type is `text`. `PostgreSQL` backs both
/// `text` and `varchar` key columns with `text_ops`: `varchar` is binary-coercible
/// to `text`, so its default B-tree class is `text_ops` and reports
/// `opcintype = text` even though the column's own type OID is `varchar`. (A
/// built-in `varchar_ops` exists but is non-default and is never auto-selected for
/// a column.) The check therefore compares against `text`'s OID rather than the
/// column's own type, so both allow-listed key types (`text`, `varchar`) pass,
/// while `bpchar_ops`, `citext_ops`, and any non-default custom class fail
/// (`opcdefault` is false for a non-default class, and no second default class can
/// exist for `text`).
const PRIMARY_KEY_QUERY: &str = "SELECT a.attname AS column_name, a.atttypid::int8 AS type_oid, \
            ty.typname AS type_name, i.indimmediate AS immediate, \
            coll.collisdeterministic AS collation_deterministic, \
            (oc.opcnamespace = 'pg_catalog'::regnamespace \
             AND am.amname = 'btree' \
             AND oc.opcdefault \
             AND oc.opcintype = 'pg_catalog.text'::regtype) AS operator_class_trusted \
     FROM pg_index i \
     JOIN pg_class t ON t.oid = i.indrelid \
     JOIN pg_namespace n ON n.oid = t.relnamespace \
     CROSS JOIN LATERAL unnest(i.indkey::int2[], i.indcollation::oid[], i.indclass::oid[]) \
         WITH ORDINALITY AS k(attnum, colloid, opclassoid, ord) \
     JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = k.attnum \
     JOIN pg_type ty ON ty.oid = a.atttypid \
     LEFT JOIN pg_collation coll ON coll.oid = k.colloid \
     JOIN pg_opclass oc ON oc.oid = k.opclassoid \
     JOIN pg_am am ON am.oid = oc.opcmethod \
     WHERE i.indisprimary \
       AND k.ord <= i.indnkeyatts \
       AND t.relname = $1 \
       AND n.nspname = current_schema() \
     ORDER BY k.ord";

/// Query for the unique indexes on a `PostgreSQL` table other than the primary
/// key, each with the columns it covers.
///
/// `pg_index.indisunique AND NOT indisprimary` covers every `UNIQUE` constraint
/// and standalone `CREATE UNIQUE INDEX`. No `indisready`/`indislive` filter is
/// applied: an invalid or half-built unique index is still an unexpected
/// deviation from the generated schema and is compared fail-closed. Each index's
/// columns are resolved through `indkey` so [`check_schema`] can accept exactly
/// the store's own generated unique indexes by column set and reject any other --
/// a narrower key that reintroduces cross-tenant data loss, or a redundant one.
///
/// Expression index members have `attnum = 0` and are dropped by the
/// `pg_attribute` join, so an expression-based unique index resolves to fewer
/// columns than any expected set and is rejected fail-closed.
const UNIQUE_INDEX_QUERY: &str = "SELECT ix.relname AS index_name, \
            array_agg(a.attname ORDER BY k.ord) AS columns \
     FROM pg_index i \
     JOIN pg_class t ON t.oid = i.indrelid \
     JOIN pg_class ix ON ix.oid = i.indexrelid \
     JOIN pg_namespace n ON n.oid = t.relnamespace \
     JOIN LATERAL unnest(i.indkey) WITH ORDINALITY AS k(attnum, ord) ON true \
     JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = k.attnum \
     WHERE i.indisunique \
       AND NOT i.indisprimary \
       AND t.relname = $1 \
       AND n.nspname = current_schema() \
     GROUP BY ix.relname \
     ORDER BY ix.relname";

/// Fetch the unique indexes on a `PostgreSQL` table other than the primary key,
/// each with the columns it covers, via [`UNIQUE_INDEX_QUERY`].
async fn table_extra_unique_indexes(pool: &sqlx::PgPool, table: &str) -> Result<Vec<ActualUniqueIndex>, StoreError> {
    let rows = sqlx::query(UNIQUE_INDEX_QUERY)
        .bind(table)
        .fetch_all(pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;
    rows.iter().map(pg_row_to_unique_index).collect()
}

/// Convert one [`UNIQUE_INDEX_QUERY`] row into an [`ActualUniqueIndex`].
fn pg_row_to_unique_index(row: &PgRow) -> Result<ActualUniqueIndex, StoreError> {
    let name: String = row
        .try_get("index_name")
        .map_err(|e| StoreError::Database(e.to_string()))?;
    let columns: Vec<String> = row
        .try_get("columns")
        .map_err(|e| StoreError::Database(e.to_string()))?;
    Ok(ActualUniqueIndex { name, columns })
}

/// Discover each tenant-scoped table's schema and compare it against the schema
/// this store generates before the schema version is stamped.
///
/// The single [`check_schema`] comparison fails closed on any deviation -- a
/// missing column, a primary key that is not the exact ordered contract, a key
/// column with a folding type/collation/operator class, a deferrable key, or a
/// unique index whose column set is not one the store itself generates -- because
/// `CREATE TABLE IF NOT EXISTS` preserves a pre-existing table that could
/// silently lose data across tenants. The global schema version table holds no
/// tenant data and is validated by value in [`check_schema_version`], not here.
async fn validate_schema(pool: &sqlx::PgPool, tables: &TableNames) -> Result<(), StoreError> {
    let expected = expected_tables(tables);
    let mut actuals = Vec::with_capacity(expected.len());
    for (table_name, _) in &expected {
        let columns = table_column_names(pool, table_name).await?;
        let (primary_key, primary_key_immediate) = table_primary_key(pool, table_name).await?;
        let unique_indexes = table_extra_unique_indexes(pool, table_name).await?;
        actuals.push(ActualTable {
            columns,
            primary_key,
            primary_key_immediate,
            unique_indexes,
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
async fn check_schema_version(pool: &sqlx::PgPool, tables: &TableNames) -> Result<(), StoreError> {
    let vt = schema_version_table(&tables.responses);
    let select = format!("SELECT version FROM {vt}");
    let row: Option<i64> = sqlx::query_scalar(AssertSqlSafe(select.as_str()))
        .fetch_optional(pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;

    match row {
        None => {
            let insert = format!("INSERT INTO {vt} (version) VALUES ($1) ON CONFLICT (version) DO NOTHING");
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
impl ResponseStore for PostgresResponseStore {
    #[expect(
        clippy::too_many_lines,
        reason = "owner-preserving SQL upsert keeps all bindings explicit"
    )]
    async fn upsert_response(&self, record: &ResponseRecord) -> Result<(), StoreError> {
        let response_object =
            serde_json::to_string(&record.response_object).map_err(|e| StoreError::Serialization(e.to_string()))?;
        let input = serde_json::to_string(&record.input).map_err(|e| StoreError::Serialization(e.to_string()))?;
        let messages = serde_json::to_string(&record.messages).map_err(|e| StoreError::Serialization(e.to_string()))?;

        let sql = format!(
            "INSERT INTO {} \
             (id, tenant_id, owner_issuer, owner_subject, created_at, model, response_object, input, messages) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9) \
             ON CONFLICT (id) DO UPDATE SET \
             created_at = EXCLUDED.created_at, \
             model = EXCLUDED.model, \
             response_object = EXCLUDED.response_object, \
             input = EXCLUDED.input, \
             messages = EXCLUDED.messages \
             WHERE {}.tenant_id = EXCLUDED.tenant_id \
               AND {}.owner_issuer = EXCLUDED.owner_issuer \
               AND {}.owner_subject = EXCLUDED.owner_subject",
            self.tables.responses, self.tables.responses, self.tables.responses, self.tables.responses
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
             WHERE id = $1 AND tenant_id = $2 AND owner_issuer = $3 AND owner_subject = $4",
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
        // Delete the response and any pending approvals it issued in one atomic
        // transaction: a deleted response must leave no consumable approval
        // behind and retain no sensitive tool arguments.
        let delete_response_sql = format!(
            "DELETE FROM {} WHERE id = $1 AND tenant_id = $2 AND owner_issuer = $3 AND owner_subject = $4",
            self.tables.responses
        );
        let delete_approvals_sql = format!(
            "DELETE FROM {} WHERE response_id = $1 AND tenant_id = $2 AND owner_issuer = $3 AND owner_subject = $4",
            pending_approvals_table(&self.tables.responses)
        );
        let mut tx = Box::pin(self.pool.begin())
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

    #[expect(clippy::too_many_lines, reason = "sequential per-record bind within a transaction")]
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
        let table = pending_approvals_table(&self.tables.responses);
        // Insert-if-absent: an approval already recorded (and possibly already
        // consumed) must never be reset back to outstanding, so a re-emit is a
        // no-op rather than a `consumed_at` reset that would enable replay.
        let sql = format!(
            "INSERT INTO {table} \
             (tenant_id, owner_issuer, owner_subject, response_id, approval_id, server_label, tool_name, arguments, \
             target_fingerprint, created_at, consumed_at) \
             SELECT $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11 \
             WHERE EXISTS (SELECT 1 FROM {} WHERE id = $12 AND tenant_id = $13 \
               AND owner_issuer = $14 AND owner_subject = $15) \
             ON CONFLICT (response_id, approval_id) DO NOTHING",
            self.tables.responses
        );

        let mut tx = Box::pin(self.pool.begin())
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
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9) \
             ON CONFLICT (id) DO UPDATE SET \
             created_at = EXCLUDED.created_at, \
             model = EXCLUDED.model, \
             response_object = EXCLUDED.response_object, \
             input = EXCLUDED.input, \
             messages = EXCLUDED.messages \
             WHERE {}.tenant_id = EXCLUDED.tenant_id \
               AND {}.owner_issuer = EXCLUDED.owner_issuer \
               AND {}.owner_subject = EXCLUDED.owner_subject",
            self.tables.responses, self.tables.responses, self.tables.responses, self.tables.responses
        );
        let approvals_table = pending_approvals_table(&self.tables.responses);
        // Insert-if-absent so a re-emit never resets an already-consumed row back
        // to outstanding (which would enable replay).
        let approval_sql = format!(
            "INSERT INTO {approvals_table} \
             (tenant_id, owner_issuer, owner_subject, response_id, approval_id, server_label, tool_name, arguments, \
             target_fingerprint, created_at, consumed_at) \
             SELECT $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11 \
             WHERE EXISTS (SELECT 1 FROM {} WHERE id = $12 AND tenant_id = $13 \
               AND owner_issuer = $14 AND owner_subject = $15) \
             ON CONFLICT (response_id, approval_id) DO NOTHING",
            self.tables.responses
        );

        // One transaction so the response and its pending approvals commit
        // together or not at all. Serialized against delete_response, this closes
        // the window where a concurrent DELETE could land between the two writes
        // and orphan an approval row still holding the tool arguments.
        let mut tx = Box::pin(self.pool.begin())
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
        // Owner occupies $1-$3 and the issuing response is $4.
        let placeholders = (0..approval_ids.len())
            .map(|i| format!("${}", i + 5))
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            "SELECT approval_id, server_label, tool_name, arguments, target_fingerprint \
             FROM {table} \
             WHERE tenant_id = $1 AND owner_issuer = $2 AND owner_subject = $3 \
               AND response_id = $4 AND approval_id IN ({placeholders})"
        );

        let mut query = sqlx::query(AssertSqlSafe(sql.as_str()))
            .bind(owner.tenant_id())
            .bind(owner.issuer())
            .bind(owner.subject())
            .bind(response_id);
        for approval_id in approval_ids {
            query = query.bind(*approval_id);
        }
        let rows = Box::pin(query.fetch_all(&self.pool))
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
            "UPDATE {table} SET consumed_at = $1 \
             WHERE tenant_id = $2 AND owner_issuer = $3 AND owner_subject = $4 \
               AND response_id = $5 AND approval_id = $6 AND consumed_at IS NULL"
        );

        let mut tx = Box::pin(self.pool.begin())
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
impl ConversationItemStore for PostgresResponseStore {
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
            "UPDATE {} SET messages = $1 WHERE conversation_id = $2 AND tenant_id = $3 \
             AND owner_issuer = $4 AND owner_subject = $5",
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
            "UPDATE {} SET metadata = $1 WHERE conversation_id = $2 AND tenant_id = $3 \
             AND owner_issuer = $4 AND owner_subject = $5",
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
            "UPDATE {} SET messages = $1 WHERE conversation_id = $2 AND tenant_id = $3 \
             AND owner_issuer = $4 AND owner_subject = $5 AND messages = $6",
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

        let mut tx = Box::pin(self.pool.begin())
            .await
            .map_err(|e| StoreError::Database(e.to_string()))?;

        let sql = format!(
            "INSERT INTO {table} \
             (item_id, tenant_id, owner_issuer, owner_subject, conversation_id, item_data, created_at, position) \
             SELECT $1, tenant_id, owner_issuer, owner_subject, conversation_id, $2, $3, $4 \
             FROM {} \
             WHERE conversation_id = $5 AND tenant_id = $6 AND owner_issuer = $7 AND owner_subject = $8",
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
                 WHERE tenant_id = $1 AND owner_issuer = $2 AND owner_subject = $3 AND conversation_id = $4 \
                   AND (position {cursor_operator} $5 \
                        OR (position = $6 AND item_id {cursor_operator} $7)) \
                 ORDER BY position {direction}, item_id {direction} \
                 LIMIT $8"
            );
            sqlx::query(AssertSqlSafe(sql.as_str()))
                .bind(owner.tenant_id())
                .bind(owner.issuer())
                .bind(owner.subject())
                .bind(conversation_id)
                .bind(position)
                .bind(position)
                .bind(item_id)
                .bind(i64::from(limit))
                .fetch_all(&self.pool)
                .await
                .map_err(|e| StoreError::Database(e.to_string()))?
        } else {
            let sql = format!(
                "SELECT item_id, tenant_id, owner_issuer, owner_subject, conversation_id, item_data, created_at, position \
                 FROM {table} \
                 WHERE tenant_id = $1 AND owner_issuer = $2 AND owner_subject = $3 AND conversation_id = $4 \
                 ORDER BY position {direction}, item_id {direction} \
                 LIMIT $5"
            );
            sqlx::query(AssertSqlSafe(sql.as_str()))
                .bind(owner.tenant_id())
                .bind(owner.issuer())
                .bind(owner.subject())
                .bind(conversation_id)
                .bind(i64::from(limit))
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

        let sql = format!(
            "SELECT item_id FROM {table} \
             WHERE tenant_id = $1 AND owner_issuer = $2 AND owner_subject = $3 \
               AND conversation_id = $4 AND item_id = ANY($5)"
        );

        let ids: Vec<&str> = item_ids.to_vec();
        sqlx::query_scalar::<_, String>(AssertSqlSafe(sql.as_str()))
            .bind(owner.tenant_id())
            .bind(owner.issuer())
            .bind(owner.subject())
            .bind(conversation_id)
            .bind(&ids)
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
             WHERE item_id = $1 AND tenant_id = $2 AND owner_issuer = $3 AND owner_subject = $4 \
               AND conversation_id = $5"
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
            "DELETE FROM {table} WHERE item_id = $1 AND tenant_id = $2 AND owner_issuer = $3 \
             AND owner_subject = $4 AND conversation_id = $5"
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
             WHERE item_id = $1 AND tenant_id = $2 AND owner_issuer = $3 AND owner_subject = $4 \
               AND conversation_id = $5"
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
             WHERE tenant_id = $1 AND owner_issuer = $2 AND owner_subject = $3 AND conversation_id = $4"
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

        let mut tx = Box::pin(self.pool.begin())
            .await
            .map_err(|e| StoreError::Database(e.to_string()))?;

        let lock_sql = format!(
            "SELECT 1 FROM {conv_table} \
             WHERE conversation_id = $1 AND tenant_id = $2 AND owner_issuer = $3 AND owner_subject = $4 \
             FOR UPDATE"
        );
        sqlx::query(AssertSqlSafe(lock_sql.as_str()))
            .bind(conversation_id)
            .bind(owner.tenant_id())
            .bind(owner.issuer())
            .bind(owner.subject())
            .fetch_optional(&mut *tx)
            .await
            .map_err(|e| StoreError::Database(e.to_string()))?;

        let max_sql = format!(
            "SELECT COALESCE(MAX(position), 0) AS max_pos \
             FROM {items_table} \
             WHERE tenant_id = $1 AND owner_issuer = $2 AND owner_subject = $3 AND conversation_id = $4"
        );
        let max_row = sqlx::query(AssertSqlSafe(max_sql.as_str()))
            .bind(owner.tenant_id())
            .bind(owner.issuer())
            .bind(owner.subject())
            .bind(conversation_id)
            .fetch_one(&mut *tx)
            .await
            .map_err(|e| StoreError::Database(e.to_string()))?;
        let max_pos: i64 = max_row
            .try_get("max_pos")
            .map_err(|e| StoreError::Database(e.to_string()))?;

        let insert_sql = format!(
            "INSERT INTO {items_table} \
             (item_id, tenant_id, owner_issuer, owner_subject, conversation_id, item_data, created_at, position) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8)"
        );
        for (i, item) in items.iter().enumerate() {
            let offset = i64::try_from(i).unwrap_or(i64::MAX);
            let position = max_pos.saturating_add(1).saturating_add(offset);
            let item_data =
                serde_json::to_string(&item.item_data).map_err(|e| StoreError::Serialization(e.to_string()))?;

            sqlx::query(AssertSqlSafe(insert_sql.as_str()))
                .bind(&item.item_id)
                .bind(owner.tenant_id())
                .bind(owner.issuer())
                .bind(owner.subject())
                .bind(conversation_id)
                .bind(&item_data)
                .bind(item.created_at)
                .bind(position)
                .execute(&mut *tx)
                .await
                .map_err(|e| StoreError::Database(e.to_string()))?;
        }

        pg_rebuild_messages(&mut tx, items_table, conv_table, owner, conversation_id).await?;

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

        let mut tx = Box::pin(self.pool.begin())
            .await
            .map_err(|e| StoreError::Database(e.to_string()))?;

        let lock_sql = format!(
            "SELECT 1 FROM {conv_table} \
             WHERE conversation_id = $1 AND tenant_id = $2 AND owner_issuer = $3 AND owner_subject = $4 \
             FOR UPDATE"
        );
        sqlx::query(AssertSqlSafe(lock_sql.as_str()))
            .bind(conversation_id)
            .bind(owner.tenant_id())
            .bind(owner.issuer())
            .bind(owner.subject())
            .fetch_optional(&mut *tx)
            .await
            .map_err(|e| StoreError::Database(e.to_string()))?;

        let delete_sql = format!(
            "DELETE FROM {items_table} \
             WHERE item_id = $1 AND tenant_id = $2 AND owner_issuer = $3 AND owner_subject = $4 \
               AND conversation_id = $5"
        );
        let result = sqlx::query(AssertSqlSafe(delete_sql.as_str()))
            .bind(item_id)
            .bind(owner.tenant_id())
            .bind(owner.issuer())
            .bind(owner.subject())
            .bind(conversation_id)
            .execute(&mut *tx)
            .await
            .map_err(|e| StoreError::Database(e.to_string()))?;

        if result.rows_affected() == 0 {
            tx.commit().await.map_err(|e| StoreError::Database(e.to_string()))?;
            return Ok(false);
        }

        pg_rebuild_messages(&mut tx, items_table, conv_table, owner, conversation_id).await?;

        tx.commit().await.map_err(|e| StoreError::Database(e.to_string()))?;
        Ok(true)
    }
}

// -----------------------------------------------------------------------------
// Transactional Helpers
// -----------------------------------------------------------------------------

/// Read all item JSON values and overwrite the conversation message cache.
///
/// Returns [`StoreError::Database`] if the conversation row is gone by
/// the time the cache is written — i.e. it was deleted concurrently
/// between the caller's existence check and this transaction. Callers
/// run this inside a transaction, so the propagated error rolls back
/// any item mutations made in the same transaction.
#[expect(clippy::too_many_lines, reason = "sequential query pipeline within a transaction")]
async fn pg_rebuild_messages(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    items_table: &str,
    conv_table: &str,
    owner: &StateOwner,
    conversation_id: &str,
) -> Result<(), StoreError> {
    let select_sql = format!(
        "SELECT item_data FROM {items_table} \
         WHERE tenant_id = $1 AND owner_issuer = $2 AND owner_subject = $3 AND conversation_id = $4 \
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
        "UPDATE {conv_table} SET messages = $1 \
         WHERE conversation_id = $2 AND tenant_id = $3 AND owner_issuer = $4 AND owner_subject = $5"
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
fn row_to_pending_approval_record(row: &PgRow) -> Result<PendingApprovalRecord, StoreError> {
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
fn row_to_response_record(row: &PgRow) -> Result<ResponseRecord, StoreError> {
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
fn row_to_conversation_item_record(row: &PgRow) -> Result<ConversationItemRecord, StoreError> {
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
fn row_to_conversation_record(row: &PgRow) -> Result<ConversationRecord, StoreError> {
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
fn row_to_owner(row: &PgRow) -> Result<StateOwner, StoreError> {
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
    fn connect_options_defaults_to_verify_full() {
        let options = pg_connect_options("postgres://user:pass@example.com/db", None, None)
            .expect("URL without sslmode should parse");

        assert!(
            matches!(options.get_ssl_mode(), PgSslMode::VerifyFull),
            "default ssl_mode should be VerifyFull"
        );
    }

    #[test]
    fn connect_options_default_overrides_url_sslmode() {
        let options = pg_connect_options("postgres://user:pass@example.com/db?sslmode=prefer", None, None)
            .expect("URL with sslmode should parse");

        assert!(
            matches!(options.get_ssl_mode(), PgSslMode::VerifyFull),
            "default VerifyFull should override URL sslmode"
        );
    }

    #[test]
    fn connect_options_uses_explicit_sslmode_override() {
        let options = pg_connect_options(
            "postgres://user:pass@example.com/db?sslmode=verify-full",
            Some(SslMode::Disable),
            None,
        )
        .expect("URL with override should parse");

        assert!(
            matches!(options.get_ssl_mode(), PgSslMode::Disable),
            "explicit ssl_mode should override URL sslmode"
        );
    }

    #[test]
    fn connect_options_applies_ssl_root_cert() {
        pg_connect_options("postgres://user:pass@example.com/db", None, Some("/path/to/ca.pem"))
            .expect("ssl_root_cert path should be accepted");
    }
}
