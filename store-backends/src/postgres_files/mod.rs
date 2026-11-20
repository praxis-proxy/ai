// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! `PostgreSQL` backend for the [`FileCatalog`].
//!
//! [`FileCatalog`]: praxis_ai_store::FileCatalog

mod catalog;
mod reaper;
mod request;
pub(crate) mod retry;
pub(crate) mod schema;
#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::too_many_lines,
    reason = "tests"
)]
mod tests;

use std::time::Duration;

use praxis_ai_store::{CatalogError, PoolConfig};

use crate::{
    PgTlsConfig,
    pool::apply_pool_config,
    postgres::pg_connect_options,
    postgres_files::retry::{PgBackendError, PgFailure, classify},
};

/// `PostgreSQL`-backed file catalog.
///
/// One configured table holds every file row across all tenants; see
/// the `schema` module for the DDL. Usage is derived (`sum(bytes)` per tenant) rather
/// than stored, and the per-tenant quota advisory lock serializes only a
/// tenant's `create_pending` calls.
#[derive(Debug)]
pub struct PostgresFileCatalog {
    /// Connection pool.
    pool: sqlx::PgPool,
    /// Configured files table name.
    files: String,
    /// Per-tenant byte limit enforced at reservation time.
    tenant_limit_bytes: u64,
    /// Delay before a released cleanup claim becomes claimable again.
    retry_delay: Duration,
    /// Connection URL, held only to redact it from runtime query errors.
    redact_url: String,
}

impl PostgresFileCatalog {
    /// Connect and initialize the file catalog schema.
    ///
    /// The `database_url` is a `PostgreSQL` connection string;
    /// `files_table` is the SQL table name for the catalog rows, validated by
    /// the schema module against shared and `PostgreSQL` identifier rules.
    /// Schema initialization and validation run in one transaction under the
    /// same per-schema advisory lock as the response store, so the two store
    /// domains never initialize concurrently.
    ///
    /// `tls` carries the TLS mode and certificate paths; `ssl_mode` always
    /// overrides any `sslmode` embedded in the URL, defaulting to
    /// `SslMode::VerifyFull`. See [`PgTlsConfig`] for the
    /// certificate-authentication compliance profile.
    ///
    /// `tenant_limit_bytes` is the single per-tenant quota: one limit for
    /// every tenant, no overrides and no file-count limit. `retry_delay` is the
    /// fixed delay before a released (failed) cleanup claim can be claimed
    /// again.
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError::Backend`] if the connection fails (classified
    /// unavailable) or schema initialization fails (classified
    /// non-retryable), with the connection URL redacted.
    #[expect(
        clippy::too_many_arguments,
        reason = "connection settings plus the quota limit and retry delay"
    )]
    pub async fn new(
        database_url: &str,
        files_table: &str,
        tls: &PgTlsConfig<'_>,
        pool_config: Option<&PoolConfig>,
        tenant_limit_bytes: u64,
        retry_delay: Duration,
    ) -> Result<Self, CatalogError> {
        let options = pg_connect_options(database_url, tls).map_err(|e| {
            CatalogError::Backend(Box::new(PgBackendError::new(PgFailure::NonRetryable, e.to_string())))
        })?;
        let pool = Box::pin(apply_pool_config(sqlx::postgres::PgPoolOptions::new(), pool_config).connect_with(options))
            .await
            .map_err(|e| {
                // A failure here is a failed connect, not a lost query.
                CatalogError::Backend(Box::new(PgBackendError::new(
                    PgFailure::Unavailable,
                    super::redact_connection_error(database_url, &e.to_string()),
                )))
            })?;

        Box::pin(schema::initialize_files_schema(&pool, files_table))
            .await
            .map_err(|e| {
                CatalogError::Backend(Box::new(PgBackendError::new(PgFailure::NonRetryable, e.to_string())))
            })?;

        tracing::info!(files = files_table, "postgres file catalog initialized");
        Ok(Self {
            pool,
            files: files_table.to_owned(),
            tenant_limit_bytes,
            retry_delay,
            redact_url: database_url.to_owned(),
        })
    }

    /// Close the connection pool, releasing its connections.
    pub async fn close(&self) {
        self.pool.close().await;
    }

    /// Classify a runtime query error and redact the connection URL and any
    /// embedded credentials from its message.
    fn pg_err(&self, e: &sqlx::Error) -> PgBackendError {
        Self::pg_failure(&self.redact_url, e)
    }

    /// Classify and redact a query error against a connection URL.
    ///
    /// Separate from [`Self::pg_err`](Self::pg_err) so redaction is testable
    /// without a live database.
    fn pg_failure(redact_url: &str, e: &sqlx::Error) -> PgBackendError {
        let class = classify(e);
        let message = super::redact_connection_error(redact_url, &e.to_string());
        PgBackendError::new(class, message)
    }

    /// [`Self::pg_failure`] wrapped as a [`CatalogError`].
    #[cfg(test)]
    fn backend_error(redact_url: &str, e: &sqlx::Error) -> CatalogError {
        CatalogError::Backend(Box::new(Self::pg_failure(redact_url, e)))
    }

    /// Open a transaction with local `lock_timeout` and `statement_timeout`.
    ///
    /// Both settings are transaction-scoped (`SET LOCAL`), following the
    /// response store's reaper: the quota lock timeout bounds how long
    /// `create_pending` waits on a competing tenant upload, and the statement
    /// timeout bounds each statement. Values are `PostgreSQL` interval
    /// strings such as `'30s'`.
    async fn begin_tx(
        &self,
        lock_timeout: &str,
        statement_timeout: &str,
    ) -> Result<sqlx::Transaction<'_, sqlx::Postgres>, PgBackendError> {
        let mut tx = Box::pin(self.pool.begin()).await.map_err(|e| self.pg_err(&e))?;
        sqlx::query(sqlx::AssertSqlSafe(format!(
            "SET LOCAL lock_timeout = '{lock_timeout}'"
        )))
        .execute(&mut *tx)
        .await
        .map_err(|e| self.pg_err(&e))?;
        sqlx::query(sqlx::AssertSqlSafe(format!(
            "SET LOCAL statement_timeout = '{statement_timeout}'"
        )))
        .execute(&mut *tx)
        .await
        .map_err(|e| self.pg_err(&e))?;
        Ok(tx)
    }

    /// Open a transaction with the standard lock and statement timeouts.
    async fn begin(&self) -> Result<sqlx::Transaction<'_, sqlx::Postgres>, PgBackendError> {
        self.begin_tx(LOCK_TIMEOUT, STATEMENT_TIMEOUT).await
    }
}

/// How long any statement may wait for a lock. Short, so a contended tenant
/// quota lock fails fast instead of holding up an upload request.
const LOCK_TIMEOUT: &str = "1s";

/// Upper bound for any single statement, reads and reaper batches included,
/// as in the response store's `sweep_expired`.
const STATEMENT_TIMEOUT: &str = "30s";

/// SQL for the database clock in Unix seconds.
const NOW: &str = "extract(epoch FROM now())::bigint";

/// A business outcome that is not a backend failure, decided inside a
/// retried transaction.
#[derive(Debug)]
enum Domain {
    /// Maps to [`CatalogError::Conflict`].
    Conflict,
    /// Maps to [`CatalogError::QuotaExceeded`].
    Quota,
    /// Maps to [`CatalogError::InvalidCursor`].
    Cursor,
}

/// Result of one transaction attempt: a backend failure (which may be
/// retried) or a business outcome (which never is).
type Attempt<T> = Result<Result<T, Domain>, PgBackendError>;

/// Flatten an attempt (or the retry wrapper's result) into the trait error.
fn finish<T>(result: Attempt<T>) -> Result<T, CatalogError> {
    match result {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(Domain::Conflict)) => Err(CatalogError::Conflict),
        Ok(Err(Domain::Quota)) => Err(CatalogError::QuotaExceeded),
        Ok(Err(Domain::Cursor)) => Err(CatalogError::InvalidCursor),
        Err(e) => Err(CatalogError::Backend(Box::new(e))),
    }
}

/// A non-retryable error for a value that cannot be represented or decoded.
fn invalid(message: impl Into<String>) -> PgBackendError {
    PgBackendError::new(PgFailure::NonRetryable, message.into())
}

/// Convert a byte count to `BIGINT`, rejecting values above `i64::MAX`.
fn to_i64(value: u64, what: &str) -> Result<i64, PgBackendError> {
    i64::try_from(value).map_err(|e| invalid(format!("{what} exceeds the supported range: {e}")))
}

/// Convert a stored `BIGINT` to a byte count; a negative value is corrupt.
fn to_u64(value: i64, what: &str) -> Result<u64, PgBackendError> {
    u64::try_from(value).map_err(|e| invalid(format!("stored {what} is negative: {e}")))
}

/// Whole seconds of a duration, rounded up.
fn secs_ceil(d: Duration) -> Result<i64, PgBackendError> {
    let secs = d.as_secs().saturating_add(u64::from(d.subsec_nanos() > 0));
    to_i64(secs, "duration")
}
