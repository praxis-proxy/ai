// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Request-path operations of the [`FileCatalog`](praxis_ai_store::FileCatalog).
//!
//! Every function here is one attempt: it opens its own transaction, so the
//! retry wrapper can repeat it. Scoping is per tenant: lookups filter by
//! `tenant_id` and `id`, while the full creating owner is only stored.
//!
//! Each transition is a single conditional statement
//! (`UPDATE … WHERE state = … RETURNING`). When it matches no row, a follow-up
//! `SELECT` decides between an idempotent retry, a conflict and not-found.

use praxis_ai_store::{
    DeletionOutcome, FileCompletion, FileListQuery, FileRecord, FileRecordPage, NewFile, SortDirection, StateOwner,
    StoredFile,
};
use sqlx::{AssertSqlSafe, Row as _, postgres::PgRow};

use super::{Attempt, Domain, NOW, PostgresFileCatalog, invalid, retry::PgBackendError, secs_ceil, to_i64, to_u64};

/// A transaction on the catalog pool.
type Tx<'a> = sqlx::Transaction<'a, sqlx::Postgres>;

/// `(tenant_id, owner_issuer, owner_subject, storage_key, filename, state, bytes)`.
/// For a pending row `bytes` is its reservation.
type ExistingRow = (String, String, String, String, String, String, i64);

/// Columns selected for a client-visible record.
const RECORD_COLUMNS: &str = "id, bytes, created_at, expires_at, filename, purpose";

/// Visibility predicate: a ready, unexpired file.
fn visible() -> String {
    format!("state = 'ready' AND (expires_at IS NULL OR expires_at > {NOW})")
}

/// Whether `e` is a unique-constraint violation (`23505`).
fn is_unique_violation(e: &sqlx::Error) -> bool {
    e.as_database_error()
        .and_then(sqlx::error::DatabaseError::code)
        .is_some_and(|code| code == "23505")
}

/// Decode a record row. A `NULL` purpose on a ready row is corrupt.
fn record_from_row(row: &PgRow) -> Result<FileRecord, PgBackendError> {
    let decode = |e: sqlx::Error| invalid(e.to_string());
    let purpose: Option<String> = row.try_get("purpose").map_err(decode)?;
    Ok(FileRecord {
        id: row.try_get("id").map_err(decode)?,
        bytes: to_u64(row.try_get("bytes").map_err(decode)?, "bytes")?,
        created_at: row.try_get("created_at").map_err(decode)?,
        expires_at: row.try_get("expires_at").map_err(decode)?,
        filename: row.try_get("filename").map_err(decode)?,
        purpose: purpose.ok_or_else(|| invalid("ready file has no purpose"))?,
    })
}

/// One attempt of `create_pending`.
///
/// Takes the per-tenant quota lock first (the only advisory lock in the
/// transaction), treats an identical existing row as success with its stored
/// reservation, then grants `min(max_bytes, remaining quota)` and inserts.
pub(super) async fn create_pending_once(c: &PostgresFileCatalog, file: &NewFile) -> Attempt<u64> {
    let tenant = file.owner.tenant_id();
    let mut tx = c.begin().await?;

    sqlx::query("SELECT pg_advisory_xact_lock(hashtext('praxis-files-quota'), hashtext($1))")
        .bind(tenant)
        .execute(&mut *tx)
        .await
        .map_err(|e| c.pg_err(&e))?;

    if let Some(outcome) = identical_pending(c, &mut tx, file).await? {
        return Ok(outcome);
    }
    let used = tenant_usage(c, &mut tx, tenant).await?;
    let remaining = c.tenant_limit_bytes.saturating_sub(used).max(0);
    // A ceiling above i64::MAX is bounded by the quota, which fits `BIGINT`.
    let reserve = remaining.min(i64::try_from(file.max_bytes).unwrap_or(i64::MAX));
    if reserve == 0 {
        return Ok(Err(Domain::Quota));
    }
    match insert_pending(c, &mut tx, file, reserve).await {
        Ok(()) => {},
        Err(e) if is_unique_violation(&e) => return Ok(Err(Domain::Conflict)),
        Err(e) => return Err(c.pg_err(&e)),
    }
    tx.commit().await.map_err(|e| c.pg_err(&e))?;
    Ok(Ok(to_u64(reserve, "reserve")?))
}

/// If a row with the same `id` exists, its reservation when it is a retry of
/// `file` or a conflict when it is not. A retry matches a *pending* row with
/// the same owner, key and filename whose reservation does not exceed
/// `max_bytes`. The reservation depends on the quota at the first attempt, so
/// it is not compared for equality; a larger one proves a different request.
/// Other states hold an actual size or a charge being released, not a
/// reservation.
async fn identical_pending(
    c: &PostgresFileCatalog,
    tx: &mut Tx<'_>,
    file: &NewFile,
) -> Result<Option<Result<u64, Domain>>, PgBackendError> {
    let files = &c.files;
    let existing: Option<ExistingRow> = sqlx::query_as(AssertSqlSafe(format!(
        "SELECT tenant_id, owner_issuer, owner_subject, storage_key, filename, state, bytes \
         FROM {files} WHERE id = $1"
    )))
    .bind(&file.id)
    .fetch_optional(&mut **tx)
    .await
    .map_err(|e| c.pg_err(&e))?;
    let Some((tenant, issuer, subject, key, filename, state, bytes)) = existing else {
        return Ok(None);
    };
    let identical = tenant == file.owner.tenant_id()
        && issuer == file.owner.issuer()
        && subject == file.owner.subject()
        && key == file.storage_key
        && filename == file.filename
        && state == "pending"
        && i128::from(bytes) <= i128::from(file.max_bytes);
    if !identical {
        return Ok(Some(Err(Domain::Conflict)));
    }
    Ok(Some(Ok(to_u64(bytes, "bytes")?)))
}

/// The tenant's current usage: `sum(bytes)` over all its rows.
async fn tenant_usage(c: &PostgresFileCatalog, tx: &mut Tx<'_>, tenant: &str) -> Result<i64, PgBackendError> {
    sqlx::query_scalar(AssertSqlSafe(format!(
        "SELECT coalesce(sum(bytes), 0)::bigint FROM {} WHERE tenant_id = $1",
        c.files
    )))
    .bind(tenant)
    .fetch_one(&mut **tx)
    .await
    .map_err(|e| c.pg_err(&e))
}

/// Insert the pending row.
async fn insert_pending(
    c: &PostgresFileCatalog,
    tx: &mut Tx<'_>,
    file: &NewFile,
    reserve: i64,
) -> Result<(), sqlx::Error> {
    sqlx::query(AssertSqlSafe(format!(
        "INSERT INTO {} (id, tenant_id, owner_issuer, owner_subject, storage_key, filename, state, bytes) \
         VALUES ($1, $2, $3, $4, $5, $6, 'pending', $7)",
        c.files
    )))
    .bind(&file.id)
    .bind(file.owner.tenant_id())
    .bind(file.owner.issuer())
    .bind(file.owner.subject())
    .bind(&file.storage_key)
    .bind(&file.filename)
    .bind(reserve)
    .execute(&mut **tx)
    .await
    .map(|_| ())
}

/// The state of a tenant's row, if it exists.
async fn state_of(
    c: &PostgresFileCatalog,
    tx: &mut Tx<'_>,
    owner: &StateOwner,
    id: &str,
) -> Result<Option<String>, PgBackendError> {
    sqlx::query_scalar(AssertSqlSafe(format!(
        "SELECT state FROM {} WHERE tenant_id = $1 AND id = $2",
        c.files
    )))
    .bind(owner.tenant_id())
    .bind(id)
    .fetch_optional(&mut **tx)
    .await
    .map_err(|e| c.pg_err(&e))
}

/// One attempt of `complete_upload`.
pub(super) async fn complete_upload_once(
    c: &PostgresFileCatalog,
    owner: &StateOwner,
    id: &str,
    completion: &FileCompletion,
) -> Attempt<FileRecord> {
    let actual = to_i64(completion.bytes, "bytes")?;
    let expires_after = completion.expires_after.map(secs_ceil).transpose()?;
    let files = &c.files;
    let mut tx = c.begin().await?;

    let updated = sqlx::query(AssertSqlSafe(format!(
        "UPDATE {files} SET state = 'ready', bytes = $3, purpose = $4, \
         expires_at = CASE WHEN $5::bigint IS NULL THEN NULL ELSE created_at + $5::bigint END \
         WHERE tenant_id = $1 AND id = $2 AND state = 'pending' AND bytes >= $3 \
         RETURNING {RECORD_COLUMNS}"
    )))
    .bind(owner.tenant_id())
    .bind(id)
    .bind(actual)
    .bind(&completion.purpose)
    .bind(expires_after)
    .fetch_optional(&mut *tx)
    .await
    .map_err(|e| c.pg_err(&e))?;
    if let Some(row) = updated {
        let record = record_from_row(&row)?;
        tx.commit().await.map_err(|e| c.pg_err(&e))?;
        return Ok(Ok(record));
    }

    completed_retry(c, &mut tx, owner, id, completion).await
}

/// Outcome of `complete_upload` when no pending row had enough
/// reservation: an identical retry of a completed upload succeeds,
/// everything else (including a size above the reservation) conflicts.
async fn completed_retry(
    c: &PostgresFileCatalog,
    tx: &mut Tx<'_>,
    owner: &StateOwner,
    id: &str,
    completion: &FileCompletion,
) -> Attempt<FileRecord> {
    let expires_after = completion.expires_after.map(secs_ceil).transpose()?;
    let existing = sqlx::query(AssertSqlSafe(format!(
        "SELECT {RECORD_COLUMNS}, state FROM {} WHERE tenant_id = $1 AND id = $2",
        c.files
    )))
    .bind(owner.tenant_id())
    .bind(id)
    .fetch_optional(&mut **tx)
    .await
    .map_err(|e| c.pg_err(&e))?;
    let Some(row) = existing else {
        return Ok(Err(Domain::Conflict));
    };
    let state: String = row.try_get("state").map_err(|e| invalid(e.to_string()))?;
    if state != "ready" {
        return Ok(Err(Domain::Conflict));
    }
    let record = record_from_row(&row)?;
    let expected_expiry = expires_after.map(|secs| record.created_at.saturating_add(secs));
    let same = record.bytes == completion.bytes
        && record.purpose == completion.purpose
        && record.expires_at == expected_expiry;
    Ok(if same { Ok(record) } else { Err(Domain::Conflict) })
}

/// One attempt of `fail_upload`.
pub(super) async fn fail_upload_once(c: &PostgresFileCatalog, owner: &StateOwner, id: &str) -> Attempt<()> {
    let files = &c.files;
    let mut tx = c.begin().await?;
    let updated = sqlx::query(AssertSqlSafe(format!(
        "UPDATE {files} SET state = 'failed', lease_until = {NOW} \
         WHERE tenant_id = $1 AND id = $2 AND state = 'pending'"
    )))
    .bind(owner.tenant_id())
    .bind(id)
    .execute(&mut *tx)
    .await
    .map_err(|e| c.pg_err(&e))?
    .rows_affected();
    if updated == 1 {
        tx.commit().await.map_err(|e| c.pg_err(&e))?;
        return Ok(Ok(()));
    }
    let state = state_of(c, &mut tx, owner, id).await?;
    // Already failed, or already purged: nothing left to do.
    Ok(match state.as_deref() {
        None | Some("failed") => Ok(()),
        Some(_) => Err(Domain::Conflict),
    })
}

/// `get_ready`: an unexpired ready file of the tenant.
pub(super) async fn get_ready_impl(
    c: &PostgresFileCatalog,
    owner: &StateOwner,
    id: &str,
) -> Result<Option<StoredFile>, PgBackendError> {
    let files = &c.files;
    let visible = visible();
    let mut tx = c.begin().await?;
    let row = sqlx::query(AssertSqlSafe(format!(
        "SELECT {RECORD_COLUMNS}, storage_key FROM {files} WHERE tenant_id = $1 AND id = $2 AND {visible}"
    )))
    .bind(owner.tenant_id())
    .bind(id)
    .fetch_optional(&mut *tx)
    .await
    .map_err(|e| c.pg_err(&e))?;
    tx.commit().await.map_err(|e| c.pg_err(&e))?;
    row.map(|row| {
        Ok(StoredFile {
            record: record_from_row(&row)?,
            storage_key: row.try_get("storage_key").map_err(|e| invalid(e.to_string()))?,
        })
    })
    .transpose()
}

/// `list_ready`: keyset paging over the tenant's visible files.
pub(super) async fn list_ready_impl(
    c: &PostgresFileCatalog,
    owner: &StateOwner,
    query: &FileListQuery,
) -> Attempt<FileRecordPage> {
    let tenant = owner.tenant_id();
    let mut tx = c.begin().await?;
    let cursor = match &query.after {
        None => None,
        Some(after) => match resolve_cursor(c, &mut tx, tenant, after).await? {
            Some(created_at) => Some((created_at, after.as_str())),
            None => return Ok(Err(Domain::Cursor)),
        },
    };

    let rows = page_rows(c, &mut tx, tenant, query, cursor).await?;
    tx.commit().await.map_err(|e| c.pg_err(&e))?;
    let mut data = rows.iter().map(record_from_row).collect::<Result<Vec<_>, _>>()?;
    let has_more = data.len() > query.limit as usize;
    data.truncate(query.limit as usize);
    Ok(Ok(FileRecordPage { data, has_more }))
}

/// Fetch `limit + 1` visible rows after `cursor` in the requested order.
async fn page_rows(
    c: &PostgresFileCatalog,
    tx: &mut Tx<'_>,
    tenant: &str,
    query: &FileListQuery,
    cursor: Option<(i64, &str)>,
) -> Result<Vec<PgRow>, PgBackendError> {
    let files = &c.files;
    // Fixed fragments only: nothing request-controlled is interpolated.
    let (cmp, dir) = match query.order {
        SortDirection::Asc => (">", "ASC"),
        SortDirection::Desc => ("<", "DESC"),
    };
    let visible = visible();
    let limit = i64::from(query.limit);
    let (cursor_at, cursor_id) = cursor.unzip();
    sqlx::query(AssertSqlSafe(format!(
        "SELECT {RECORD_COLUMNS} FROM {files} \
         WHERE tenant_id = $1 AND {visible} \
         AND ($2::text IS NULL OR purpose = $2) \
         AND ($3::bigint IS NULL OR (created_at, id) {cmp} ($3, $4::text)) \
         ORDER BY created_at {dir}, id {dir} LIMIT $5"
    )))
    .bind(tenant)
    .bind(query.purpose.as_deref())
    .bind(cursor_at)
    .bind(cursor_id)
    .bind(limit + 1)
    .fetch_all(&mut **tx)
    .await
    .map_err(|e| c.pg_err(&e))
}

/// `created_at` of the cursor file among all the tenant's rows, so a file
/// deleted between two pages does not invalidate the cursor.
async fn resolve_cursor(
    c: &PostgresFileCatalog,
    tx: &mut Tx<'_>,
    tenant: &str,
    after: &str,
) -> Result<Option<i64>, PgBackendError> {
    sqlx::query_scalar(AssertSqlSafe(format!(
        "SELECT created_at FROM {} WHERE tenant_id = $1 AND id = $2",
        c.files
    )))
    .bind(tenant)
    .bind(after)
    .fetch_optional(&mut **tx)
    .await
    .map_err(|e| c.pg_err(&e))
}

/// One attempt of `mark_for_deletion`.
pub(super) async fn mark_for_deletion_once(
    c: &PostgresFileCatalog,
    owner: &StateOwner,
    id: &str,
) -> Result<DeletionOutcome, PgBackendError> {
    let files = &c.files;
    let visible = visible();
    let mut tx = c.begin().await?;
    let updated = sqlx::query(AssertSqlSafe(format!(
        "UPDATE {files} SET state = 'deletion_pending', lease_until = {NOW} \
         WHERE tenant_id = $1 AND id = $2 AND {visible}"
    )))
    .bind(owner.tenant_id())
    .bind(id)
    .execute(&mut *tx)
    .await
    .map_err(|e| c.pg_err(&e))?
    .rows_affected();
    if updated == 1 {
        tx.commit().await.map_err(|e| c.pg_err(&e))?;
        return Ok(DeletionOutcome::Marked);
    }
    let state = state_of(c, &mut tx, owner, id).await?;
    // Also covers an expiry sweep that marked the file first.
    Ok(match state.as_deref() {
        Some("deletion_pending") => DeletionOutcome::AlreadyPending,
        _ => DeletionOutcome::NotFound,
    })
}
