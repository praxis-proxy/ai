// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! `PostgreSQL` backend for the [`FileCatalog`].
//!
//! [`FileCatalog`]: praxis_ai_store::FileCatalog

// Remove once the catalog operations (steps 4-5) call the constructor helpers.
#![cfg_attr(
    not(test),
    expect(dead_code, reason = "no caller until the catalog operations land in steps 4-5")
)]

mod catalog;
pub(crate) mod retry;
pub(crate) mod schema;

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
/// [`schema`] for the DDL. Usage is derived (`sum(bytes)` per tenant) rather
/// than stored, and the per-tenant quota advisory lock serializes only a
/// tenant's `create_pending` calls (steps 4–5).
#[derive(Debug)]
pub struct PostgresFileCatalog {
    /// Connection pool.
    pool: sqlx::PgPool,
    /// Configured files table name.
    files: String,
    /// Per-tenant byte limit enforced at reservation time.
    tenant_limit_bytes: u64,
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
    /// every tenant, no overrides and no file-count limit.
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError::Backend`] if the connection fails (classified
    /// unavailable) or schema initialization fails (classified
    /// non-retryable), with the connection URL redacted.
    pub async fn new(
        database_url: &str,
        files_table: &str,
        tls: &PgTlsConfig<'_>,
        pool_config: Option<&PoolConfig>,
        tenant_limit_bytes: u64,
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
            redact_url: database_url.to_owned(),
        })
    }

    /// Close the connection pool, releasing its connections.
    pub async fn close(&self) {
        self.pool.close().await;
    }

    /// Wrap a runtime query error: classify it for retry decisions and redact
    /// the connection URL and any embedded credentials from the message.
    fn db_err(&self, e: &sqlx::Error) -> CatalogError {
        Self::backend_error(&self.redact_url, e)
    }

    /// Classify and redact a query error against a connection URL.
    ///
    /// Separate from [`Self::db_err`](Self::db_err) so redaction is testable
    /// without a live database.
    fn backend_error(redact_url: &str, e: &sqlx::Error) -> CatalogError {
        let class = classify(e);
        let message = super::redact_connection_error(redact_url, &e.to_string());
        CatalogError::Backend(Box::new(PgBackendError::new(class, message)))
    }

    /// Open a transaction with local `lock_timeout` and `statement_timeout`.
    ///
    /// Both settings are transaction-scoped (`SET LOCAL`), following the
    /// response store's reaper: the quota lock timeout bounds how long
    /// `create_pending` waits on a competing tenant upload, and the statement
    /// timeout bounds each reaper batch. Values are `PostgreSQL` interval
    /// strings such as `'30s'`.
    async fn begin_tx(
        &self,
        lock_timeout: &str,
        statement_timeout: &str,
    ) -> Result<sqlx::Transaction<'_, sqlx::Postgres>, CatalogError> {
        let mut tx = Box::pin(self.pool.begin()).await.map_err(|e| self.db_err(&e))?;
        sqlx::query(sqlx::AssertSqlSafe(format!(
            "SET LOCAL lock_timeout = '{lock_timeout}'"
        )))
        .execute(&mut *tx)
        .await
        .map_err(|e| self.db_err(&e))?;
        sqlx::query(sqlx::AssertSqlSafe(format!(
            "SET LOCAL statement_timeout = '{statement_timeout}'"
        )))
        .execute(&mut *tx)
        .await
        .map_err(|e| self.db_err(&e))?;
        Ok(tx)
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, reason = "tests")]
mod tests {
    use std::error::Error as _;

    use super::*;

    /// A unique lowercase table name per test.
    fn unique_name(prefix: &str) -> String {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        format!("{prefix}_{nanos}")
    }

    /// TLS config that matches a plain local test database.
    fn tls() -> PgTlsConfig<'static> {
        PgTlsConfig {
            ssl_mode: Some(crate::SslMode::Disable),
            ..PgTlsConfig::default()
        }
    }

    #[test]
    fn backend_error_redacts_url_password() {
        let url = "postgres://user:supersecret@host.example.com/db";
        let io = std::io::Error::other("failed to reach postgres://user:supersecret@host.example.com/db");
        let err = PostgresFileCatalog::backend_error(url, &sqlx::Error::Io(io));
        let text = err.to_string();
        assert!(!text.contains("supersecret"), "leaked password: {text}");
        let source = err.source().unwrap().to_string();
        assert!(!source.contains("supersecret"), "leaked password: {source}");
    }

    #[test]
    fn backend_error_keeps_classification() {
        let io = std::io::Error::other("connection reset by peer");
        let err = PostgresFileCatalog::backend_error("postgres://user@host/db", &sqlx::Error::Io(io));
        let source = err.source().unwrap().downcast_ref::<PgBackendError>().unwrap();
        assert_eq!(source.class, PgFailure::Retryable);
    }

    #[tokio::test]
    #[ignore = "requires disposable PostgreSQL via DATABASE_URL"]
    async fn constructor_creates_schema_and_is_idempotent() {
        let url = std::env::var("DATABASE_URL").unwrap();
        let files = unique_name("catalog_new");
        let catalog = PostgresFileCatalog::new(&url, &files, &tls(), None, 1024)
            .await
            .unwrap();
        assert_eq!(catalog.files, files);
        assert_eq!(catalog.tenant_limit_bytes, 1024);
        // A second connect over the same table validates instead of creating.
        let again = PostgresFileCatalog::new(&url, &files, &tls(), None, 1024)
            .await
            .unwrap();
        catalog.close().await;
        again.close().await;
    }

    #[tokio::test]
    #[ignore = "requires disposable PostgreSQL via DATABASE_URL"]
    async fn begin_tx_sets_timeouts_and_db_err_classifies() {
        let url = std::env::var("DATABASE_URL").unwrap();
        let files = unique_name("catalog_tx");
        let catalog = PostgresFileCatalog::new(&url, &files, &tls(), None, 1024)
            .await
            .unwrap();
        let tx = catalog.begin_tx("5s", "5s").await.unwrap();
        tx.rollback().await.unwrap();
        // A query against a missing table is a non-retryable backend failure.
        let err = sqlx::query(sqlx::AssertSqlSafe(format!("SELECT * FROM {}", unique_name("missing"))))
            .execute(&catalog.pool)
            .await
            .unwrap_err();
        let mapped = catalog.db_err(&err);
        let source = mapped.source().unwrap().downcast_ref::<PgBackendError>().unwrap();
        assert_eq!(source.class, PgFailure::NonRetryable);
        catalog.close().await;
    }

    #[tokio::test]
    #[ignore = "requires disposable PostgreSQL via DATABASE_URL"]
    async fn constructor_rejects_version_mismatch() {
        let url = std::env::var("DATABASE_URL").unwrap();
        let files = unique_name("catalog_ver");
        let catalog = PostgresFileCatalog::new(&url, &files, &tls(), None, 1024)
            .await
            .unwrap();
        catalog.close().await;
        let pool = sqlx::postgres::PgPoolOptions::new().connect(&url).await.unwrap();
        sqlx::query(sqlx::AssertSqlSafe(format!(
            "UPDATE {} SET version = 99",
            schema::version_table(&files)
        )))
        .execute(&pool)
        .await
        .unwrap();
        pool.close().await;
        let err = PostgresFileCatalog::new(&url, &files, &tls(), None, 1024)
            .await
            .unwrap_err();
        let text = err.to_string();
        assert!(text.contains("backend failed"), "expected a backend error, got: {text}");
    }
}
