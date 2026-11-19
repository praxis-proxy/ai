// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! DDL, identifier validation and initialization for the file catalog schema.
//!
//! One configured table holds every file row. The schema version table and the
//! tenant index names derive from it, the same way the response store derives
//! its auxiliary names.

use praxis_ai_store::{StoreError, validate_table_identifier};

use crate::{
    postgres::{begin_initialization, check_schema_version, run_ddl, validate_expected_tables},
    schemas::{ExpectedTable, validate_postgres_identifier_case, validate_postgres_identifier_len},
};

/// Default name of the files table until configuration is wired in.
pub(crate) const DEFAULT_FILES_TABLE: &str = "praxis_files";

/// Current files schema version. Bump this when the DDL changes.
const FILES_SCHEMA_VERSION: i64 = 1;

/// Suffix appended to the files table name for the schema version table.
const VERSION_SUFFIX: &str = "_schema_version";

/// Suffix appended to the files table name for the tenant index.
const TENANT_INDEX_SUFFIX: &str = "_tenant_idx";

/// `PostgreSQL` truncates identifiers longer than this many bytes.
const POSTGRES_MAX_IDENTIFIER_LEN: usize = 63;

/// Longest files table name: the longest derived suffix must still fit.
const MAX_FILES_TABLE_LEN: usize = POSTGRES_MAX_IDENTIFIER_LEN - VERSION_SUFFIX.len();

/// Columns the files table must contain.
const FILES_COLUMNS: &[&str] = &[
    "id",
    "tenant_id",
    "owner_issuer",
    "owner_subject",
    "storage_key",
    "filename",
    "purpose",
    "state",
    "bytes",
    "created_at",
    "expires_at",
    "available_at",
];

/// The files table contract: primary key `id` and a unique `storage_key`.
const FILES_TABLE: ExpectedTable = ExpectedTable {
    columns: FILES_COLUMNS,
    primary_key: &["id"],
    unique_indexes: &[&["storage_key"]],
};

/// Derive the schema version table name from the files table name.
pub(crate) fn version_table(files: &str) -> String {
    format!("{files}{VERSION_SUFFIX}")
}

/// Derive the tenant index name from the files table name.
fn tenant_index(files: &str) -> String {
    format!("{files}{TENANT_INDEX_SUFFIX}")
}

/// Validate the files table name against shared and `PostgreSQL` rules.
///
/// The name is interpolated unquoted into DDL, so it must be a plain
/// lowercase identifier short enough for every derived name to fit.
fn validate_files_table(files: &str) -> Result<(), StoreError> {
    validate_table_identifier(files)?;
    validate_postgres_identifier_case("files table name", files)?;
    validate_postgres_identifier_len("files table name", files, MAX_FILES_TABLE_LEN)
}

/// Generate the idempotent DDL statements for the files schema.
fn generate_files_ddl(files: &str) -> Result<Vec<String>, StoreError> {
    validate_files_table(files)?;
    let versions = version_table(files);
    let tenant_idx = tenant_index(files);
    Ok(vec![
        format!(
            "CREATE TABLE IF NOT EXISTS {files} (\n\
             \x20 id            TEXT   PRIMARY KEY,\n\
             \x20 tenant_id     TEXT   NOT NULL,\n\
             \x20 owner_issuer  TEXT   NOT NULL,\n\
             \x20 owner_subject TEXT   NOT NULL,\n\
             \x20 storage_key   TEXT   NOT NULL UNIQUE,\n\
             \x20 filename      TEXT   NOT NULL,\n\
             \x20 purpose       TEXT,\n\
             \x20 state         TEXT   NOT NULL\n\
             \x20               CHECK (state IN ('pending','ready','failed','deletion_pending')),\n\
             \x20 bytes         BIGINT NOT NULL CHECK (bytes >= 0),\n\
             \x20 created_at    BIGINT NOT NULL DEFAULT extract(epoch FROM now())::bigint,\n\
             \x20 expires_at    BIGINT,\n\
             \x20 available_at  BIGINT NOT NULL DEFAULT extract(epoch FROM now())::bigint\n\
             )"
        ),
        format!("CREATE INDEX IF NOT EXISTS {tenant_idx} ON {files} (tenant_id)"),
        format!("CREATE TABLE IF NOT EXISTS {versions} (version BIGINT PRIMARY KEY)"),
    ])
}

/// Create and validate the files schema under the shared schema lock.
///
/// Runs in one transaction: take the lock, run the DDL, validate the table
/// against its contract (an existing incompatible table would otherwise be kept
/// silently by `IF NOT EXISTS`), then stamp or check the schema version.
///
/// # Errors
///
/// Returns [`StoreError::Database`] for an invalid table name, a DDL or
/// database failure, an incompatible existing table, or a version mismatch.
pub(crate) async fn initialize_files_schema(pool: &sqlx::PgPool, files: &str) -> Result<(), StoreError> {
    let ddl = generate_files_ddl(files)?;
    let mut initialization = begin_initialization(pool).await?;
    run_ddl(&mut initialization, &ddl).await?;
    validate_expected_tables(&mut initialization, &[(files.to_owned(), FILES_TABLE)]).await?;
    check_schema_version(&mut initialization, &version_table(files), FILES_SCHEMA_VERSION).await?;
    initialization
        .commit()
        .await
        .map_err(|e| StoreError::Database(e.to_string()))
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, reason = "tests")]
mod tests {
    use sqlx::{AssertSqlSafe, postgres::PgPoolOptions};

    use super::*;

    // ---- unit tests (no database) ----

    #[test]
    fn version_table_derives_name() {
        assert_eq!(version_table("praxis_files"), "praxis_files_schema_version");
    }

    #[test]
    fn ddl_has_table_index_and_version_table() {
        let ddl = generate_files_ddl("f1").unwrap();
        assert_eq!(ddl.len(), 3);
        let mut statements = ddl.iter();
        let (table, index, versions) = (
            statements.next().unwrap(),
            statements.next().unwrap(),
            statements.next().unwrap(),
        );
        assert!(table.starts_with("CREATE TABLE IF NOT EXISTS f1 ("));
        assert!(table.contains("storage_key   TEXT   NOT NULL UNIQUE"));
        assert!(table.contains("extract(epoch FROM now())::bigint"));
        assert_eq!(index, "CREATE INDEX IF NOT EXISTS f1_tenant_idx ON f1 (tenant_id)");
        assert_eq!(
            versions,
            "CREATE TABLE IF NOT EXISTS f1_schema_version (version BIGINT PRIMARY KEY)"
        );
    }

    #[test]
    fn default_table_name_is_valid() {
        generate_files_ddl(DEFAULT_FILES_TABLE).unwrap();
    }

    #[test]
    fn rejects_uppercase_name() {
        assert!(generate_files_ddl("Files").is_err());
    }

    #[test]
    fn rejects_invalid_name() {
        assert!(generate_files_ddl("bad-name").is_err());
        assert!(generate_files_ddl("").is_err());
        assert!(generate_files_ddl("1files").is_err());
    }

    #[test]
    fn rejects_name_over_48_bytes() {
        assert!(generate_files_ddl(&"a".repeat(49)).is_err());
    }

    #[test]
    fn accepts_48_byte_name() {
        let name = "a".repeat(48);
        generate_files_ddl(&name).unwrap();
        assert!(version_table(&name).len() <= POSTGRES_MAX_IDENTIFIER_LEN);
        assert!(tenant_index(&name).len() <= POSTGRES_MAX_IDENTIFIER_LEN);
    }

    // ---- PostgreSQL tests ----

    /// Connect to the disposable test database.
    async fn pool() -> sqlx::PgPool {
        let url = std::env::var("DATABASE_URL").expect("disposable test database");
        Box::pin(PgPoolOptions::new().max_connections(8).connect(&url))
            .await
            .unwrap()
    }

    /// A unique lowercase table name per test.
    fn unique_name(prefix: &str) -> String {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        format!("{prefix}_{nanos}")
    }

    async fn exec(pool: &sqlx::PgPool, sql: &str) {
        sqlx::query(AssertSqlSafe(sql)).execute(pool).await.unwrap();
    }

    #[tokio::test]
    #[ignore = "requires disposable PostgreSQL via DATABASE_URL"]
    async fn initialization_is_idempotent() {
        let pool = pool().await;
        let files = unique_name("files_idem");
        initialize_files_schema(&pool, &files).await.unwrap();
        initialize_files_schema(&pool, &files).await.unwrap();
        let versions: i64 =
            sqlx::query_scalar(AssertSqlSafe(format!("SELECT count(*) FROM {}", version_table(&files))))
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(versions, 1);
        pool.close().await;
    }

    #[tokio::test]
    #[ignore = "requires disposable PostgreSQL via DATABASE_URL"]
    async fn concurrent_initialization_succeeds() {
        let pool = pool().await;
        let files = unique_name("files_conc");
        let results = futures::future::join_all((0..8).map(|_| initialize_files_schema(&pool, &files))).await;
        for result in results {
            result.unwrap();
        }
        let versions: i64 =
            sqlx::query_scalar(AssertSqlSafe(format!("SELECT count(*) FROM {}", version_table(&files))))
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(versions, 1);
        pool.close().await;
    }

    #[tokio::test]
    #[ignore = "requires disposable PostgreSQL via DATABASE_URL"]
    async fn table_missing_columns_is_rejected() {
        let pool = pool().await;
        let files = unique_name("files_cols");
        exec(&pool, &format!("CREATE TABLE {files} (id TEXT PRIMARY KEY)")).await;
        // Fails when the DDL creates the tenant index over a missing column.
        assert!(initialize_files_schema(&pool, &files).await.is_err());
        pool.close().await;
    }

    #[tokio::test]
    #[ignore = "requires disposable PostgreSQL via DATABASE_URL"]
    async fn table_with_wrong_primary_key_is_rejected() {
        let pool = pool().await;
        let files = unique_name("files_pk");
        // Every column present, but no primary key: schema validation must catch it.
        let ddl = generate_files_ddl(&files).unwrap();
        let table = ddl.first().unwrap();
        exec(&pool, &table.replace("TEXT   PRIMARY KEY", "TEXT")).await;
        let err = initialize_files_schema(&pool, &files).await.unwrap_err();
        assert!(err.to_string().contains("primary key"), "unexpected error: {err}");
        pool.close().await;
    }

    #[tokio::test]
    #[ignore = "requires disposable PostgreSQL via DATABASE_URL"]
    async fn version_mismatch_is_rejected() {
        let pool = pool().await;
        let files = unique_name("files_ver");
        initialize_files_schema(&pool, &files).await.unwrap();
        exec(&pool, &format!("UPDATE {} SET version = 99", version_table(&files))).await;
        let err = initialize_files_schema(&pool, &files).await.unwrap_err();
        assert!(
            err.to_string().contains("schema version mismatch"),
            "unexpected error: {err}"
        );
        pool.close().await;
    }

    #[tokio::test]
    #[ignore = "requires disposable PostgreSQL via DATABASE_URL"]
    async fn quota_sum_uses_tenant_index() {
        let pool = pool().await;
        let files = unique_name("files_explain");
        initialize_files_schema(&pool, &files).await.unwrap();
        // On a near-empty table the planner prefers a sequential scan.
        let mut tx = pool.begin().await.unwrap();
        exec_tx(&mut tx, "SET LOCAL enable_seqscan = off").await;
        let rows: Vec<String> = sqlx::query_scalar(AssertSqlSafe(format!(
            "EXPLAIN SELECT coalesce(sum(bytes), 0) FROM {files} WHERE tenant_id = 't'"
        )))
        .fetch_all(&mut *tx)
        .await
        .unwrap();
        let plan = rows.join("\n");
        assert!(plan.contains(&format!("{files}_tenant_idx")), "plan: {plan}");
        tx.rollback().await.unwrap();
        pool.close().await;
    }

    async fn exec_tx(tx: &mut sqlx::Transaction<'_, sqlx::Postgres>, sql: &str) {
        sqlx::query(AssertSqlSafe(sql)).execute(&mut **tx).await.unwrap();
    }

    #[tokio::test]
    #[ignore = "requires disposable PostgreSQL via DATABASE_URL"]
    async fn coexists_with_response_store_tables() {
        use crate::{PgTlsConfig, PostgresResponseStore, SslMode};
        let url = std::env::var("DATABASE_URL").expect("disposable test database");
        let pool = pool().await;
        let responses = unique_name("coexist_responses");
        let conversations = unique_name("coexist_conversations");
        let files = unique_name("coexist_files");
        let tls = PgTlsConfig {
            ssl_mode: Some(SslMode::Disable),
            ..PgTlsConfig::default()
        };
        let store = PostgresResponseStore::new(&url, &responses, &conversations, None, &tls, None, None)
            .await
            .unwrap();
        initialize_files_schema(&pool, &files).await.unwrap();
        // Re-initializing either side still succeeds.
        initialize_files_schema(&pool, &files).await.unwrap();
        let again = PostgresResponseStore::new(&url, &responses, &conversations, None, &tls, None, None)
            .await
            .unwrap();
        store.close().await;
        again.close().await;
        pool.close().await;
    }
}
