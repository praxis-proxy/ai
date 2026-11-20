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
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "no caller until provisioning is wired in (#504)")
)]
pub(crate) const DEFAULT_FILES_TABLE: &str = "praxis_files";

/// Current files schema version. Bump this when the DDL changes.
pub(super) const FILES_SCHEMA_VERSION: i64 = 1;

/// Suffix appended to the files table name for the schema version table.
const VERSION_SUFFIX: &str = "_schema_version";

/// Suffix appended to the files table name for the tenant index.
const TENANT_INDEX_SUFFIX: &str = "_tenant_idx";

/// `PostgreSQL` truncates identifiers longer than this many bytes.
pub(super) const POSTGRES_MAX_IDENTIFIER_LEN: usize = 63;

/// Longest files table name: the longest derived suffix must still fit.
pub(super) const MAX_FILES_TABLE_LEN: usize = POSTGRES_MAX_IDENTIFIER_LEN - VERSION_SUFFIX.len();

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
    "lease_until",
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
pub(super) fn tenant_index(files: &str) -> String {
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
pub(super) fn generate_files_ddl(files: &str) -> Result<Vec<String>, StoreError> {
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
             \x20 lease_until  BIGINT NOT NULL DEFAULT extract(epoch FROM now())::bigint\n\
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
