// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Cross-tenant lifecycle and cleanup operations of the [`FileCatalog`](praxis_ai_store::FileCatalog).
//!
//! These work across tenants and never change usage except the purge in
//! [`complete_cleanup_once`]. Each
//! batch is its own transaction. Claims are leases (`lease_until`): they
//! prevent duplicate byte-store deletes and double as the retry delay, because
//! a worker whose delete fails does nothing and the row is claimable again when
//! its lease expires. Only claiming writes the lease, so a stale worker cannot
//! shorten another's. Each claim gets a fresh token and the purge must match
//! it, so only the latest claim can purge. Exactly-once release comes from the
//! conditional purge: a row is deleted once, and its charge goes with it.

use std::time::Duration;

use praxis_ai_store::{CleanupClaim, CleanupState, StateOwner};
use sqlx::{AssertSqlSafe, Row as _, postgres::PgRow};

use super::{NOW, PostgresFileCatalog, invalid, retry::PgBackendError, schema::claim_sequence, secs_ceil};

/// Stored value of a cleanup state.
const fn state_name(state: CleanupState) -> &'static str {
    match state {
        CleanupState::Failed => "failed",
        CleanupState::DeletionPending => "deletion_pending",
    }
}

/// Parse a stored cleanup state; anything else is corrupt.
fn parse_state(state: &str) -> Result<CleanupState, PgBackendError> {
    match state {
        "failed" => Ok(CleanupState::Failed),
        "deletion_pending" => Ok(CleanupState::DeletionPending),
        other => Err(invalid(format!("unexpected cleanup state `{other}`"))),
    }
}

/// Decode a claimed row.
fn claim_from_row(row: &PgRow) -> Result<CleanupClaim, PgBackendError> {
    let decode = |e: sqlx::Error| invalid(e.to_string());
    let owner = StateOwner::from_trusted_parts(
        row.try_get::<String, _>("tenant_id").map_err(decode)?,
        row.try_get::<String, _>("owner_issuer").map_err(decode)?,
        row.try_get::<String, _>("owner_subject").map_err(decode)?,
    )
    .map_err(|e| invalid(format!("stored owner is invalid: {e}")))?;
    Ok(CleanupClaim {
        id: row.try_get("id").map_err(decode)?,
        owner,
        storage_key: row.try_get("storage_key").map_err(decode)?,
        state: parse_state(&row.try_get::<String, _>("state").map_err(decode)?)?,
        claim_token: row.try_get("claim_token").map_err(decode)?,
    })
}

/// Mark up to `limit` expired ready files as deletion-pending.
pub(super) async fn mark_expired_once(c: &PostgresFileCatalog, limit: u32) -> Result<u32, PgBackendError> {
    let files = &c.files;
    let mut tx = c.begin().await?;
    let affected = sqlx::query(AssertSqlSafe(format!(
        "WITH batch AS ( \
           SELECT id FROM {files} WHERE state = 'ready' AND expires_at <= {NOW} \
           LIMIT $1 FOR UPDATE SKIP LOCKED) \
         UPDATE {files} f SET state = 'deletion_pending', lease_until = {NOW} \
         FROM batch WHERE f.id = batch.id AND f.state = 'ready'"
    )))
    .bind(i64::from(limit))
    .execute(&mut *tx)
    .await
    .map_err(|e| c.pg_err(&e))?
    .rows_affected();
    tx.commit().await.map_err(|e| c.pg_err(&e))?;
    u32::try_from(affected).map_err(|e| invalid(format!("affected row count out of range: {e}")))
}

/// Mark up to `limit` abandoned pending uploads as failed.
pub(super) async fn fail_abandoned_once(
    c: &PostgresFileCatalog,
    abandon_after: Duration,
    limit: u32,
) -> Result<u32, PgBackendError> {
    let files = &c.files;
    let secs = secs_ceil(abandon_after)?;
    let mut tx = c.begin().await?;
    let affected = sqlx::query(AssertSqlSafe(format!(
        "WITH batch AS ( \
           SELECT id FROM {files} WHERE state = 'pending' AND created_at <= {NOW} - $1 \
           LIMIT $2 FOR UPDATE SKIP LOCKED) \
         UPDATE {files} f SET state = 'failed', lease_until = {NOW} \
         FROM batch WHERE f.id = batch.id AND f.state = 'pending'"
    )))
    .bind(secs)
    .bind(i64::from(limit))
    .execute(&mut *tx)
    .await
    .map_err(|e| c.pg_err(&e))?
    .rows_affected();
    tx.commit().await.map_err(|e| c.pg_err(&e))?;
    u32::try_from(affected).map_err(|e| invalid(format!("affected row count out of range: {e}")))
}

/// Lease up to `limit` due cleanup rows, oldest `lease_until` first.
pub(super) async fn claim_cleanup_once(
    c: &PostgresFileCatalog,
    lease: Duration,
    limit: u32,
) -> Result<Vec<CleanupClaim>, PgBackendError> {
    let files = &c.files;
    let lease_secs = secs_ceil(lease)?;
    let seq = claim_sequence(files);
    let mut tx = c.begin().await?;
    let rows = sqlx::query(AssertSqlSafe(format!(
        "WITH batch AS ( \
           SELECT id FROM {files} \
           WHERE state IN ('failed', 'deletion_pending') AND lease_until <= {NOW} \
           ORDER BY lease_until LIMIT $2 FOR UPDATE SKIP LOCKED) \
         UPDATE {files} f SET lease_until = {NOW} + $1, claim_token = nextval('{seq}') \
         FROM batch WHERE f.id = batch.id AND f.state IN ('failed', 'deletion_pending') \
         RETURNING f.id, f.tenant_id, f.owner_issuer, f.owner_subject, f.storage_key, f.state, f.claim_token"
    )))
    .bind(lease_secs)
    .bind(i64::from(limit))
    .fetch_all(&mut *tx)
    .await
    .map_err(|e| c.pg_err(&e))?;
    tx.commit().await.map_err(|e| c.pg_err(&e))?;

    rows.iter().map(claim_from_row).collect()
}

/// Purge the claimed row; deleting it releases its charged bytes.
///
/// `false` means no row matched: already purged, or the claim was superseded.
pub(super) async fn complete_cleanup_once(
    c: &PostgresFileCatalog,
    claim: &CleanupClaim,
) -> Result<bool, PgBackendError> {
    let files = &c.files;
    let mut tx = c.begin().await?;
    let deleted = sqlx::query(AssertSqlSafe(format!(
        "DELETE FROM {files} WHERE id = $1 AND state = $2 AND storage_key = $3 AND claim_token = $4"
    )))
    .bind(&claim.id)
    .bind(state_name(claim.state))
    .bind(&claim.storage_key)
    .bind(claim.claim_token)
    .execute(&mut *tx)
    .await
    .map_err(|e| c.pg_err(&e))?
    .rows_affected();
    tx.commit().await.map_err(|e| c.pg_err(&e))?;
    Ok(deleted == 1)
}
