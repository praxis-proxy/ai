// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Tests for the `PostgreSQL` file catalog.
//!
//! Tests marked `#[ignore]` need a disposable `PostgreSQL` via `DATABASE_URL`
//! (`make test-postgres-unit` runs them). Concurrency tests end with the usage
//! oracle: every tenant's `sum(bytes)` stays within the limit.

/// DDL, validation and initialization.
mod schema {
    use sqlx::{AssertSqlSafe, postgres::PgPoolOptions};

    use crate::postgres_files::schema::*;

    // ---- unit tests (no database) ----

    #[test]
    fn table_name_validation() {
        generate_files_ddl(DEFAULT_FILES_TABLE).unwrap();
        for bad in ["Files", "bad-name", "", "1files"] {
            assert!(generate_files_ddl(bad).is_err(), "{bad:?}");
        }
        assert!(generate_files_ddl(&"a".repeat(49)).is_err());
        let name = "a".repeat(48);
        generate_files_ddl(&name).unwrap();
        assert!(version_table(&name).len() <= POSTGRES_MAX_IDENTIFIER_LEN);
        for index in [list_index, expiry_index, pending_index, lease_index] {
            assert!(index(&name).len() <= POSTGRES_MAX_IDENTIFIER_LEN);
        }
        assert!(claim_sequence(&name).len() <= POSTGRES_MAX_IDENTIFIER_LEN);
    }

    #[test]
    fn ddl_has_table_indexes_sequence_and_version_table() {
        let ddl = generate_files_ddl("f1").unwrap();
        assert_eq!(ddl.len(), 7);
        let [table, list, expiry, pending, lease, sequence, versions] = &*ddl else {
            panic!("unexpected statement count");
        };
        assert!(table.starts_with("CREATE TABLE IF NOT EXISTS f1 ("));
        assert!(table.contains("storage_key   TEXT   NOT NULL UNIQUE"));
        assert!(table.contains("extract(epoch FROM now())::bigint"));
        assert!(table.contains("claim_token   BIGINT\n"));
        assert_eq!(
            list,
            "CREATE INDEX IF NOT EXISTS f1_list_idx ON f1 (tenant_id, state, created_at, id)"
        );
        assert!(expiry.contains("f1_expiry_idx ON f1 (expires_at) WHERE state = 'ready' AND expires_at IS NOT NULL"));
        assert!(pending.contains("f1_pending_idx ON f1 (created_at) WHERE state = 'pending'"));
        assert!(lease.contains("f1_lease_idx ON f1 (lease_until) WHERE state IN ('failed', 'deletion_pending')"));
        assert_eq!(sequence, "CREATE SEQUENCE IF NOT EXISTS f1_claim_seq");
        assert_eq!(
            versions,
            "CREATE TABLE IF NOT EXISTS f1_schema_version (version BIGINT PRIMARY KEY)"
        );
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
    async fn initialization_seeds_claim_sequence_above_stored_tokens() {
        let pool = pool().await;
        let files = unique_name("files_seed");
        initialize_files_schema(&pool, &files).await.unwrap();
        exec(
            &pool,
            &format!(
                "INSERT INTO {files} (id, tenant_id, owner_issuer, owner_subject, storage_key, filename, state, \
                 bytes, claim_token) VALUES ('a', 't', 'i', 's', 'k', 'f', 'failed', 1, 500)"
            ),
        )
        .await;
        // Recreated behind the stored token.
        exec(&pool, &format!("DROP SEQUENCE {}", claim_sequence(&files))).await;
        initialize_files_schema(&pool, &files).await.unwrap();
        let next: i64 = sqlx::query_scalar(AssertSqlSafe(format!("SELECT nextval('{}')", claim_sequence(&files))))
            .fetch_one(&pool)
            .await
            .unwrap();
        assert!(next > 500, "sequence not seeded: {next}");
        // A fresh sequence next to a stored token 1 must not reissue it.
        exec(&pool, &format!("UPDATE {files} SET claim_token = 1")).await;
        exec(&pool, &format!("DROP SEQUENCE {}", claim_sequence(&files))).await;
        initialize_files_schema(&pool, &files).await.unwrap();
        let fresh: i64 = sqlx::query_scalar(AssertSqlSafe(format!("SELECT nextval('{}')", claim_sequence(&files))))
            .fetch_one(&pool)
            .await
            .unwrap();
        assert!(fresh > 1, "token 1 reissued");
        // An ahead sequence is left alone.
        initialize_files_schema(&pool, &files).await.unwrap();
        let after: i64 = sqlx::query_scalar(AssertSqlSafe(format!("SELECT nextval('{}')", claim_sequence(&files))))
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(after, fresh + 1);
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

/// Error classification and bounded retry.
mod retry {
    use crate::postgres_files::retry::*;

    #[test]
    fn sqlstate_codes_classify() {
        assert_eq!(PgFailure::from_code("40001"), PgFailure::Retryable);
        assert_eq!(PgFailure::from_code("40P01"), PgFailure::Retryable);
        assert_eq!(PgFailure::from_code("55P03"), PgFailure::Retryable);
        assert_eq!(PgFailure::from_code("57P01"), PgFailure::Retryable);
        // Unique violation, syntax error, undefined table: not retryable.
        assert_eq!(PgFailure::from_code("23505"), PgFailure::NonRetryable);
        assert_eq!(PgFailure::from_code("42601"), PgFailure::NonRetryable);
        assert_eq!(PgFailure::from_code("42P01"), PgFailure::NonRetryable);
        // Non-database errors.
        assert_eq!(classify(&sqlx::Error::PoolTimedOut), PgFailure::Unavailable);
        assert_eq!(classify(&sqlx::Error::PoolClosed), PgFailure::Unavailable);
        let io = std::io::Error::other("connection reset by peer");
        assert_eq!(classify(&sqlx::Error::Io(io)), PgFailure::Retryable);
        assert_eq!(classify(&sqlx::Error::RowNotFound), PgFailure::NonRetryable);
    }

    #[tokio::test]
    async fn retry_succeeds_after_transient_failures() {
        let attempts = std::sync::atomic::AtomicU32::new(0);
        let result = with_retry(|| {
            let count = attempts.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            async move {
                if count < 2 {
                    Err(PgBackendError::retryable("deadlock"))
                } else {
                    Ok(count)
                }
            }
        })
        .await;
        assert_eq!(result.unwrap(), 2);
        assert_eq!(attempts.load(std::sync::atomic::Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn retry_gives_up_after_max_attempts() {
        let attempts = std::sync::atomic::AtomicU32::new(0);
        let err = with_retry(|| {
            attempts.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            async { Err::<(), _>(PgBackendError::retryable("serialization failure")) }
        })
        .await
        .unwrap_err();
        assert_eq!(err.class, PgFailure::Retryable);
        assert_eq!(attempts.load(std::sync::atomic::Ordering::SeqCst), MAX_ATTEMPTS);
    }

    #[tokio::test]
    async fn non_retryable_fails_without_retry() {
        let attempts = std::sync::atomic::AtomicU32::new(0);
        let err = with_retry(|| {
            attempts.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            async { Err::<(), _>(PgBackendError::non_retryable("unique violation")) }
        })
        .await
        .unwrap_err();
        assert_eq!(err.class, PgFailure::NonRetryable);
        assert_eq!(attempts.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[test]
    fn backoff_stays_capped() {
        for attempts in 1..10 {
            let delay = backoff_delay(attempts);
            let max = MAX_BACKOFF + MAX_BACKOFF / 2;
            assert!(delay <= max, "attempt {attempts}: {delay:?} exceeds {max:?}");
            assert!(delay >= BASE_BACKOFF, "attempt {attempts}: {delay:?} below base");
        }
    }
}

/// Constructor, transaction helper and error redaction.
mod constructor {
    use std::error::Error as _;

    use crate::postgres_files::*;

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

    #[tokio::test]
    async fn constructor_rejects_limit_above_bigint_without_connecting() {
        // The check precedes the connect, so the unreachable URL is never used.
        let too_big = u64::try_from(i64::MAX).unwrap() + 1;
        let err = PostgresFileCatalog::new("postgres://unused.invalid/db", "files", &tls(), None, too_big)
            .await
            .map(|_| ())
            .unwrap_err();
        let CatalogError::Backend(source) = err else {
            panic!("expected a backend error");
        };
        let text = source.to_string();
        assert!(text.contains("tenant_limit_bytes") && text.contains("BIGINT"), "{text}");
        assert_eq!(
            validate_tenant_limit(u64::try_from(i64::MAX).unwrap()).unwrap(),
            i64::MAX
        );
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
}

/// Catalog operations: behavior, quota, races, cleanup and failures.
mod operations {
    use std::{error::Error as _, sync::Arc, time::Duration};

    use praxis_ai_store::{
        CatalogError, CleanupClaim, CleanupState, DeletionOutcome, FileCatalog as _, FileCompletion, FileListQuery,
        FileRecord, NewFile, SortDirection, StateOwner,
    };
    use tokio::task::JoinSet;

    use crate::{
        PgTlsConfig, SslMode,
        postgres_files::{
            PostgresFileCatalog,
            retry::{PgBackendError, PgFailure, with_retry},
        },
    };

    /// Per-tenant quota used by most tests.
    const LIMIT: u64 = 1_000;

    /// Long delay so a released claim stays unclaimable during a test.
    const LONG: Duration = Duration::from_secs(3_600);

    fn unique_name(prefix: &str) -> String {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        format!("{prefix}_{nanos}")
    }

    fn database_url() -> String {
        std::env::var("DATABASE_URL").unwrap()
    }

    async fn catalog_with(limit: u64) -> PostgresFileCatalog {
        let tls = PgTlsConfig {
            ssl_mode: Some(SslMode::Disable),
            ..PgTlsConfig::default()
        };
        PostgresFileCatalog::new(&database_url(), &unique_name("ops"), &tls, None, limit)
            .await
            .unwrap()
    }

    async fn catalog() -> PostgresFileCatalog {
        catalog_with(LIMIT).await
    }

    fn owner(tenant: &str, subject: &str) -> StateOwner {
        StateOwner::from_trusted_parts(tenant, "issuer", subject).unwrap()
    }

    fn new_file(id: &str, owner: &StateOwner, max_bytes: u64) -> NewFile {
        NewFile {
            id: id.to_owned(),
            owner: owner.clone(),
            storage_key: format!("key-{id}"),
            filename: format!("{id}.txt"),
            max_bytes,
        }
    }

    fn completion(bytes: u64, purpose: &str) -> FileCompletion {
        FileCompletion {
            bytes,
            purpose: purpose.to_owned(),
            expires_after: None,
        }
    }

    /// Create a pending file and complete it with its full reservation.
    async fn ready(c: &PostgresFileCatalog, owner: &StateOwner, id: &str, bytes: u64, purpose: &str) -> FileRecord {
        c.create_pending(&new_file(id, owner, bytes)).await.unwrap();
        c.complete_upload(owner, id, completion(bytes, purpose)).await.unwrap()
    }

    /// Run raw SQL against the catalog table; `{f}` is replaced by its name.
    async fn exec(c: &PostgresFileCatalog, sql: &str) {
        sqlx::query(sqlx::AssertSqlSafe(sql.replace("{f}", &c.files)))
            .execute(&c.pool)
            .await
            .unwrap();
    }

    /// The usage oracle: a tenant's `sum(bytes)`.
    async fn usage(c: &PostgresFileCatalog, tenant: &str) -> i64 {
        sqlx::query_scalar(sqlx::AssertSqlSafe(format!(
            "SELECT coalesce(sum(bytes), 0)::bigint FROM {} WHERE tenant_id = $1",
            c.files
        )))
        .bind(tenant)
        .fetch_one(&c.pool)
        .await
        .unwrap()
    }

    /// Assert no tenant is over the limit.
    async fn assert_oracle(c: &PostgresFileCatalog) {
        let over: i64 = sqlx::query_scalar(sqlx::AssertSqlSafe(format!(
            "SELECT count(*) FROM (SELECT tenant_id FROM {} GROUP BY tenant_id HAVING sum(bytes) > {}) t",
            c.files, c.tenant_limit_bytes
        )))
        .fetch_one(&c.pool)
        .await
        .unwrap();
        assert_eq!(over, 0, "a tenant is over its quota");
    }

    /// `(state, bytes)` of a row, if it exists.
    async fn row(c: &PostgresFileCatalog, id: &str) -> Option<(String, i64)> {
        sqlx::query_as(sqlx::AssertSqlSafe(format!(
            "SELECT state, bytes FROM {} WHERE id = $1",
            c.files
        )))
        .bind(id)
        .fetch_optional(&c.pool)
        .await
        .unwrap()
    }

    fn failure_class(err: &CatalogError) -> PgFailure {
        err.source().unwrap().downcast_ref::<PgBackendError>().unwrap().class
    }

    fn query(order: SortDirection, limit: u32) -> FileListQuery {
        FileListQuery {
            purpose: None,
            limit,
            order,
            after: None,
        }
    }

    fn ids(page: &praxis_ai_store::FileRecordPage) -> Vec<&str> {
        page.data.iter().map(|r| r.id.as_str()).collect()
    }

    // ---- behavior ----

    #[tokio::test]
    #[ignore = "requires disposable PostgreSQL via DATABASE_URL"]
    async fn tenant_isolation_and_same_tenant_visibility() {
        let c = catalog().await;
        let (a, b, other) = (owner("t1", "alice"), owner("t1", "bob"), owner("t2", "alice"));
        let record = ready(&c, &a, "f1", 10, "assistants").await;
        assert_eq!(record.filename, "f1.txt");

        // Another subject in the same tenant sees, lists and deletes the file.
        let stored = c.get_ready(&b, "f1").await.unwrap().unwrap();
        assert_eq!(stored.record, record);
        assert_eq!(stored.storage_key, "key-f1");
        assert_eq!(
            ids(&c.list_ready(&b, &query(SortDirection::Asc, 10)).await.unwrap()),
            ["f1"]
        );

        // Another tenant sees nothing and cannot touch it.
        assert!(c.get_ready(&other, "f1").await.unwrap().is_none());
        assert!(
            c.list_ready(&other, &query(SortDirection::Asc, 10))
                .await
                .unwrap()
                .data
                .is_empty()
        );
        assert_eq!(
            c.mark_for_deletion(&other, "f1").await.unwrap(),
            DeletionOutcome::NotFound
        );
        assert!(matches!(
            c.complete_upload(&other, "f1", completion(10, "assistants")).await,
            Err(CatalogError::Conflict)
        ));
        assert_eq!(c.mark_for_deletion(&b, "f1").await.unwrap(), DeletionOutcome::Marked);
        assert!(c.get_ready(&a, "f1").await.unwrap().is_none());
        c.close().await;
    }

    #[tokio::test]
    #[ignore = "requires disposable PostgreSQL via DATABASE_URL"]
    async fn pending_failed_and_expired_rows_are_hidden() {
        let c = catalog().await;
        let a = owner("t1", "alice");
        c.create_pending(&new_file("pending", &a, 1)).await.unwrap();
        c.create_pending(&new_file("failed", &a, 1)).await.unwrap();
        c.fail_upload(&a, "failed").await.unwrap();
        ready(&c, &a, "expired", 1, "assistants").await;
        exec(&c, "UPDATE {f} SET expires_at = 1 WHERE id = 'expired'").await;
        ready(&c, &a, "visible", 1, "assistants").await;

        for id in ["pending", "failed", "expired"] {
            assert!(c.get_ready(&a, id).await.unwrap().is_none(), "{id} must be hidden");
        }
        assert_eq!(
            ids(&c.list_ready(&a, &query(SortDirection::Asc, 10)).await.unwrap()),
            ["visible"]
        );
        c.close().await;
    }

    #[tokio::test]
    #[ignore = "requires disposable PostgreSQL via DATABASE_URL"]
    async fn paging_both_directions_purpose_filter_and_cursors() {
        let c = catalog().await;
        let a = owner("t1", "alice");
        for (i, purpose) in ["a", "b", "a", "b", "a"].iter().enumerate() {
            ready(&c, &a, &format!("f{i}"), 1, purpose).await;
            // Identical created_at for f2/f3 exercises the id tiebreak.
            let at = if i == 3 { 2 } else { i64::try_from(i).unwrap() };
            exec(&c, &format!("UPDATE {{f}} SET created_at = {at} WHERE id = 'f{i}'")).await;
        }

        let page = c.list_ready(&a, &query(SortDirection::Asc, 2)).await.unwrap();
        assert_eq!((ids(&page), page.has_more), (vec!["f0", "f1"], true));
        let mut q = query(SortDirection::Asc, 2);
        q.after = Some("f1".to_owned());
        let page = c.list_ready(&a, &q).await.unwrap();
        assert_eq!((ids(&page), page.has_more), (vec!["f2", "f3"], true));
        q.after = Some("f3".to_owned());
        let page = c.list_ready(&a, &q).await.unwrap();
        assert_eq!((ids(&page), page.has_more), (vec!["f4"], false));

        let page = c.list_ready(&a, &query(SortDirection::Desc, 2)).await.unwrap();
        assert_eq!((ids(&page), page.has_more), (vec!["f4", "f3"], true));
        let mut q = query(SortDirection::Desc, 10);
        q.after = Some("f3".to_owned());
        assert_eq!(ids(&c.list_ready(&a, &q).await.unwrap()), ["f2", "f1", "f0"]);

        let mut q = query(SortDirection::Asc, 10);
        q.purpose = Some("b".to_owned());
        assert_eq!(ids(&c.list_ready(&a, &q).await.unwrap()), ["f1", "f3"]);

        // A cursor on a file deleted between pages still resolves.
        c.mark_for_deletion(&a, "f1").await.unwrap();
        let mut q = query(SortDirection::Asc, 10);
        q.after = Some("f1".to_owned());
        assert_eq!(ids(&c.list_ready(&a, &q).await.unwrap()), ["f2", "f3", "f4"]);

        // Unknown and foreign-tenant cursors are invalid.
        q.after = Some("missing".to_owned());
        assert!(matches!(c.list_ready(&a, &q).await, Err(CatalogError::InvalidCursor)));
        q.after = Some("f0".to_owned());
        let foreign = owner("t2", "alice");
        assert!(matches!(
            c.list_ready(&foreign, &q).await,
            Err(CatalogError::InvalidCursor)
        ));
        c.close().await;
    }

    #[tokio::test]
    #[ignore = "requires disposable PostgreSQL via DATABASE_URL"]
    async fn create_pending_is_idempotent_and_conflicts() {
        let c = catalog().await;
        let a = owner("t1", "alice");
        let file = new_file("f1", &a, 100);
        let created = c.create_pending(&file).await.unwrap();
        assert_eq!(c.create_pending(&file).await.unwrap(), created);
        assert_eq!(created, 100);
        assert_eq!(usage(&c, "t1").await, 100, "an identical retry must not reserve again");

        let mut different = file.clone();
        different.filename = "other.txt".to_owned();
        assert!(matches!(
            c.create_pending(&different).await,
            Err(CatalogError::Conflict)
        ));
        // Same storage key under another id.
        let mut dup_key = new_file("f2", &a, 1);
        dup_key.storage_key = file.storage_key.clone();
        assert!(matches!(c.create_pending(&dup_key).await, Err(CatalogError::Conflict)));
        // Same id from another tenant.
        let foreign = new_file("f1", &owner("t2", "alice"), 100);
        assert!(matches!(c.create_pending(&foreign).await, Err(CatalogError::Conflict)));
        // A retry asking for less than the stored reservation is a different request.
        let smaller = new_file("f1", &a, 10);
        assert!(matches!(c.create_pending(&smaller).await, Err(CatalogError::Conflict)));
        // A ceiling above i64::MAX is bounded by the remaining quota, not rejected.
        let huge = new_file("f3", &a, u64::MAX);
        let granted = c.create_pending(&huge).await.unwrap();
        assert_eq!(granted, LIMIT - 100, "the grant is the remaining quota");
        c.close().await;
    }

    #[tokio::test]
    #[ignore = "requires disposable PostgreSQL via DATABASE_URL"]
    async fn create_pending_retry_requires_a_pending_row() {
        let c = catalog().await;
        let a = owner("t1", "alice");
        let file = new_file("f1", &a, 100);
        c.create_pending(&file).await.unwrap();
        c.complete_upload(&a, "f1", completion(40, "assistants")).await.unwrap();
        // A ready row's size is not a reservation.
        assert!(matches!(c.create_pending(&file).await, Err(CatalogError::Conflict)));

        c.create_pending(&new_file("f2", &a, 100)).await.unwrap();
        exec(&c, "UPDATE {f} SET state = 'failed' WHERE id = 'f2'").await;
        assert!(matches!(
            c.create_pending(&new_file("f2", &a, 100)).await,
            Err(CatalogError::Conflict)
        ));
        c.close().await;
    }

    #[tokio::test]
    #[ignore = "requires disposable PostgreSQL via DATABASE_URL"]
    async fn reservation_is_capped_by_the_remaining_quota() {
        let c = catalog_with(100).await;
        let a = owner("t1", "alice");
        // Below the quota the per-file maximum is granted.
        assert_eq!(c.create_pending(&new_file("f1", &a, 30)).await.unwrap(), 30);
        // The per-file maximum exceeds the remainder: the remainder is granted.
        assert_eq!(c.create_pending(&new_file("f2", &a, 500)).await.unwrap(), 70);
        assert_eq!(usage(&c, "t1").await, 100);
        // Nothing left: rejected without inserting a row.
        assert!(matches!(
            c.create_pending(&new_file("f3", &a, 500)).await,
            Err(CatalogError::QuotaExceeded)
        ));
        assert!(row(&c, "f3").await.is_none());
        // A zero per-file maximum can never reserve.
        assert!(matches!(
            c.create_pending(&new_file("f4", &owner("t2", "bob"), 0)).await,
            Err(CatalogError::QuotaExceeded)
        ));
        // A retry returns the stored grant, not a new decision, after room frees up.
        exec(&c, "DELETE FROM {f} WHERE id = 'f1'").await;
        assert_eq!(c.create_pending(&new_file("f2", &a, 500)).await.unwrap(), 70);
        assert_eq!(usage(&c, "t1").await, 70);
        c.close().await;
    }

    #[tokio::test]
    #[ignore = "requires disposable PostgreSQL via DATABASE_URL"]
    async fn complete_upload_transitions_and_retries() {
        let c = catalog().await;
        let a = owner("t1", "alice");
        c.create_pending(&new_file("f1", &a, 100)).await.unwrap();

        // Above the reservation: rejected, row untouched.
        assert!(matches!(
            c.complete_upload(&a, "f1", completion(101, "assistants")).await,
            Err(CatalogError::Conflict)
        ));
        assert_eq!(row(&c, "f1").await.unwrap(), ("pending".to_owned(), 100));

        let done = FileCompletion {
            bytes: 60,
            purpose: "assistants".to_owned(),
            expires_after: Some(Duration::from_secs(90)),
        };
        let record = c.complete_upload(&a, "f1", done.clone()).await.unwrap();
        assert_eq!(record.bytes, 60);
        assert_eq!(record.expires_at, Some(record.created_at + 90));
        // Settled: the row, and so the usage, now holds the actual size.
        assert_eq!(row(&c, "f1").await.unwrap(), ("ready".to_owned(), 60));
        assert_eq!(usage(&c, "t1").await, 60);

        // Identical retry returns the same record; any difference conflicts.
        assert_eq!(c.complete_upload(&a, "f1", done.clone()).await.unwrap(), record);
        let mut other = done.clone();
        other.purpose = "batch".to_owned();
        assert!(matches!(
            c.complete_upload(&a, "f1", other).await,
            Err(CatalogError::Conflict)
        ));
        let mut other = done;
        other.bytes = 59;
        assert!(matches!(
            c.complete_upload(&a, "f1", other).await,
            Err(CatalogError::Conflict)
        ));
        assert!(matches!(
            c.complete_upload(&a, "missing", completion(1, "x")).await,
            Err(CatalogError::Conflict)
        ));

        // Settle at exactly the reservation keeps usage; zero bytes releases it all.
        c.create_pending(&new_file("eq", &a, 100)).await.unwrap();
        c.create_pending(&new_file("zero", &a, 100)).await.unwrap();
        assert_eq!(usage(&c, "t1").await, 260);
        c.complete_upload(&a, "eq", completion(100, "p")).await.unwrap();
        assert_eq!(usage(&c, "t1").await, 260);
        c.complete_upload(&a, "zero", completion(0, "p")).await.unwrap();
        assert_eq!(usage(&c, "t1").await, 160);
        c.close().await;
    }

    #[tokio::test]
    #[ignore = "requires disposable PostgreSQL via DATABASE_URL"]
    async fn fail_upload_is_idempotent_and_never_demotes() {
        let c = catalog().await;
        let a = owner("t1", "alice");
        c.create_pending(&new_file("f1", &a, 10)).await.unwrap();
        c.fail_upload(&a, "f1").await.unwrap();
        c.fail_upload(&a, "f1").await.unwrap();
        assert_eq!(row(&c, "f1").await.unwrap().0, "failed");
        // Already purged (or never existed): nothing to do.
        c.fail_upload(&a, "missing").await.unwrap();

        ready(&c, &a, "f2", 10, "p").await;
        assert!(matches!(c.fail_upload(&a, "f2").await, Err(CatalogError::Conflict)));
        assert_eq!(row(&c, "f2").await.unwrap().0, "ready");
        c.mark_for_deletion(&a, "f2").await.unwrap();
        assert!(matches!(c.fail_upload(&a, "f2").await, Err(CatalogError::Conflict)));
        c.close().await;
    }

    #[tokio::test]
    #[ignore = "requires disposable PostgreSQL via DATABASE_URL"]
    async fn mark_for_deletion_outcomes() {
        let c = catalog().await;
        let a = owner("t1", "alice");
        ready(&c, &a, "f1", 10, "p").await;
        assert_eq!(c.mark_for_deletion(&a, "f1").await.unwrap(), DeletionOutcome::Marked);
        assert_eq!(
            c.mark_for_deletion(&a, "f1").await.unwrap(),
            DeletionOutcome::AlreadyPending
        );
        assert_eq!(
            c.mark_for_deletion(&a, "missing").await.unwrap(),
            DeletionOutcome::NotFound
        );
        // Pending and failed files are not deletable.
        c.create_pending(&new_file("p", &a, 1)).await.unwrap();
        assert_eq!(c.mark_for_deletion(&a, "p").await.unwrap(), DeletionOutcome::NotFound);
        // An expired file is invisible; once the sweep marked it, a retry succeeds.
        ready(&c, &a, "e", 10, "p").await;
        exec(&c, "UPDATE {f} SET expires_at = 1 WHERE id = 'e'").await;
        assert_eq!(c.mark_for_deletion(&a, "e").await.unwrap(), DeletionOutcome::NotFound);
        assert_eq!(c.mark_expired_for_deletion(10).await.unwrap(), 1);
        assert_eq!(
            c.mark_for_deletion(&a, "e").await.unwrap(),
            DeletionOutcome::AlreadyPending
        );
        // Marking never changes usage.
        assert_eq!(usage(&c, "t1").await, 21);
        c.close().await;
    }

    // ---- quota ----

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore = "requires disposable PostgreSQL via DATABASE_URL"]
    async fn concurrent_reservations_never_exceed_quota() {
        let c = Arc::new(catalog().await);
        let mut tasks = JoinSet::new();
        for i in 0..30 {
            let c = Arc::clone(&c);
            tasks.spawn(async move {
                // Different subjects, one tenant.
                let o = owner("t1", &format!("subject-{}", i % 5));
                c.create_pending(&new_file(&format!("f{i}"), &o, 100)).await
            });
        }
        let (mut ok, mut quota) = (0, 0);
        while let Some(result) = tasks.join_next().await {
            match result.unwrap() {
                Ok(_) => ok += 1,
                Err(CatalogError::QuotaExceeded) => quota += 1,
                Err(e) => panic!("unexpected {e:?}"),
            }
        }
        assert_eq!((ok, quota), (10, 20));
        assert_eq!(usage(&c, "t1").await, 1_000);
        assert_oracle(&c).await;
        c.close().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore = "requires disposable PostgreSQL via DATABASE_URL"]
    async fn quota_lock_does_not_block_other_tenants() {
        let c = Arc::new(catalog().await);
        // Hold tenant t1's quota lock in an open transaction.
        let mut held = c.pool.begin().await.unwrap();
        sqlx::query("SELECT pg_advisory_xact_lock(hashtext('praxis-files-quota'), hashtext('t1'))")
            .execute(&mut *held)
            .await
            .unwrap();
        let blocked = {
            let c = Arc::clone(&c);
            tokio::spawn(async move { c.create_pending(&new_file("a", &owner("t1", "s"), 10)).await })
        };
        // Another tenant is not blocked.
        tokio::time::timeout(
            Duration::from_secs(3),
            c.create_pending(&new_file("b", &owner("t2", "s"), 10)),
        )
        .await
        .expect("other tenant must not wait on t1's lock")
        .unwrap();
        assert!(!blocked.is_finished(), "t1 must wait for its own lock");
        held.rollback().await.unwrap();
        blocked.await.unwrap().unwrap();
        assert_oracle(&c).await;
        c.close().await;
    }

    #[tokio::test]
    #[ignore = "requires disposable PostgreSQL via DATABASE_URL"]
    async fn quota_frees_only_at_purge() {
        let c = catalog_with(100).await;
        let a = owner("t1", "alice");
        c.create_pending(&new_file("f1", &a, 100)).await.unwrap();
        assert!(matches!(
            c.create_pending(&new_file("f2", &a, 1)).await,
            Err(CatalogError::QuotaExceeded)
        ));
        // Crashed upload: abandoned, still reserved.
        exec(&c, "UPDATE {f} SET created_at = 1 WHERE id = 'f1'").await;
        assert_eq!(c.fail_abandoned_uploads(Duration::from_secs(60), 10).await.unwrap(), 1);
        assert_eq!(usage(&c, "t1").await, 100);
        assert!(matches!(
            c.create_pending(&new_file("f2", &a, 1)).await,
            Err(CatalogError::QuotaExceeded)
        ));
        // Purge releases it.
        let claims = c.claim_cleanup(LONG, 10).await.unwrap();
        assert_eq!(claims.len(), 1);
        assert!(c.complete_cleanup(&claims[0]).await.unwrap());
        assert_eq!(usage(&c, "t1").await, 0);
        c.create_pending(&new_file("f2", &a, 100)).await.unwrap();
        c.close().await;
    }

    // ---- races ----

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore = "requires disposable PostgreSQL via DATABASE_URL"]
    async fn delete_racing_expiry_marks_once() {
        let c = Arc::new(catalog().await);
        let a = owner("t1", "alice");
        for i in 0..10 {
            let id = format!("f{i}");
            ready(&c, &a, &id, 10, "p").await;
            exec(&c, &format!("UPDATE {{f}} SET expires_at = 1 WHERE id = '{id}'")).await;
        }
        let sweeper = {
            let c = Arc::clone(&c);
            tokio::spawn(async move { c.mark_expired_for_deletion(100).await.unwrap() })
        };
        let mut deletes = JoinSet::new();
        for i in 0..10 {
            let (c, a) = (Arc::clone(&c), a.clone());
            deletes.spawn(async move { c.mark_for_deletion(&a, &format!("f{i}")).await.unwrap() });
        }
        while let Some(outcome) = deletes.join_next().await {
            assert_ne!(
                outcome.unwrap(),
                DeletionOutcome::Marked,
                "an expired file is not deletable"
            );
        }
        assert_eq!(sweeper.await.unwrap(), 10);
        for i in 0..10 {
            assert_eq!(row(&c, &format!("f{i}")).await.unwrap().0, "deletion_pending");
        }
        assert_eq!(usage(&c, "t1").await, 100, "marking never changes usage");
        assert_oracle(&c).await;
        c.close().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore = "requires disposable PostgreSQL via DATABASE_URL"]
    async fn complete_upload_racing_abandon_has_one_winner() {
        let c = Arc::new(catalog().await);
        let a = owner("t1", "alice");
        for i in 0..20 {
            let id = format!("f{i}");
            c.create_pending(&new_file(&id, &a, 10)).await.unwrap();
            exec(&c, &format!("UPDATE {{f}} SET created_at = 1 WHERE id = '{id}'")).await;
        }
        let sweeper = {
            let c = Arc::clone(&c);
            tokio::spawn(async move { c.fail_abandoned_uploads(Duration::from_secs(60), 100).await.unwrap() })
        };
        let mut completions = JoinSet::new();
        for i in 0..20 {
            let (c, a) = (Arc::clone(&c), a.clone());
            completions.spawn(async move {
                let id = format!("f{i}");
                let result = c.complete_upload(&a, &id, completion(10, "p")).await;
                let completed = match result {
                    Ok(_) => true,
                    Err(CatalogError::Conflict) => false,
                    Err(e) => panic!("losing completion must be a conflict, got {e:?}"),
                };
                (id, completed)
            });
        }
        let mut won = 0;
        while let Some(result) = completions.join_next().await {
            let (id, completed) = result.unwrap();
            let state = row(&c, &id).await.unwrap().0;
            assert_eq!(state, if completed { "ready" } else { "failed" }, "{id}");
            won += i64::from(completed);
        }
        assert_eq!(
            i64::from(sweeper.await.unwrap()) + won,
            20,
            "each file has exactly one winner"
        );
        assert_oracle(&c).await;
        c.close().await;
    }

    // ---- cleanup ----

    async fn deletion_pending(c: &PostgresFileCatalog, a: &StateOwner, n: usize) {
        for i in 0..n {
            let id = format!("f{i}");
            ready(c, a, &id, 10, "p").await;
            c.mark_for_deletion(a, &id).await.unwrap();
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore = "requires disposable PostgreSQL via DATABASE_URL"]
    async fn competing_claimers_get_disjoint_rows() {
        let c = Arc::new(catalog_with(10_000).await);
        deletion_pending(&c, &owner("t1", "alice"), 30).await;
        let mut tasks = JoinSet::new();
        for _ in 0..4 {
            let c = Arc::clone(&c);
            tasks.spawn(async move { c.claim_cleanup(LONG, 10).await.unwrap() });
        }
        let mut seen = std::collections::HashSet::new();
        let mut total = 0;
        while let Some(claims) = tasks.join_next().await {
            for claim in claims.unwrap() {
                total += 1;
                assert!(seen.insert(claim.id), "row claimed twice");
            }
        }
        assert_eq!(total, 30);
        c.close().await;
    }

    #[tokio::test]
    #[ignore = "requires disposable PostgreSQL via DATABASE_URL"]
    async fn lease_expiry_allows_takeover() {
        let c = catalog().await;
        deletion_pending(&c, &owner("t1", "alice"), 1).await;
        let first = c.claim_cleanup(LONG, 10).await.unwrap();
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].state, CleanupState::DeletionPending);
        assert_eq!(first[0].owner, owner("t1", "alice"));
        // Leased: not claimable.
        assert!(c.claim_cleanup(LONG, 10).await.unwrap().is_empty());
        // The worker crashed and the lease ran out.
        exec(&c, "UPDATE {f} SET lease_until = 1").await;
        let second = c.claim_cleanup(LONG, 10).await.unwrap();
        assert_eq!(second.len(), 1);
        assert_eq!(second[0].id, first[0].id);
        assert_ne!(second[0].claim_token, first[0].claim_token);
        c.close().await;
    }

    #[tokio::test]
    #[ignore = "requires disposable PostgreSQL via DATABASE_URL"]
    async fn stale_complete_cleanup_releases_nothing() {
        let c = catalog().await;
        deletion_pending(&c, &owner("t1", "alice"), 2).await;
        assert_eq!(usage(&c, "t1").await, 20);
        let claims = c.claim_cleanup(LONG, 10).await.unwrap();
        let stale = claims.iter().find(|claim| claim.id == "f0").unwrap().clone();
        assert!(c.complete_cleanup(&stale).await.unwrap());
        assert_eq!(usage(&c, "t1").await, 10);
        // The stale worker finishes late: false, nothing released again.
        assert!(!c.complete_cleanup(&stale).await.unwrap());
        assert_eq!(usage(&c, "t1").await, 10);
        // A claim whose state or key no longer matches purges nothing.
        let other = claims.iter().find(|claim| claim.id == "f1").unwrap();
        let wrong_state = CleanupClaim {
            state: CleanupState::Failed,
            ..other.clone()
        };
        assert!(!c.complete_cleanup(&wrong_state).await.unwrap());
        let wrong_key = CleanupClaim {
            storage_key: "other".to_owned(),
            ..other.clone()
        };
        assert!(!c.complete_cleanup(&wrong_key).await.unwrap());
        assert_eq!(usage(&c, "t1").await, 10);
        c.close().await;
    }

    #[tokio::test]
    #[ignore = "requires disposable PostgreSQL via DATABASE_URL"]
    async fn stale_claim_cannot_purge_a_reused_id() {
        let c = catalog().await;
        let alice = owner("t1", "alice");
        deletion_pending(&c, &alice, 1).await;
        let stale = c.claim_cleanup(LONG, 10).await.unwrap().remove(0);
        // Another worker reclaims and purges.
        exec(&c, "UPDATE {f} SET lease_until = 1").await;
        let current = c.claim_cleanup(LONG, 10).await.unwrap().remove(0);
        assert!(c.complete_cleanup(&current).await.unwrap());
        // A caller breaks the contract: same ID and key, same cleanup state.
        c.create_pending(&new_file(&stale.id, &alice, 10)).await.unwrap();
        c.fail_upload(&alice, &stale.id).await.unwrap();
        exec(&c, "UPDATE {f} SET state = 'deletion_pending'").await;
        let usage_before = usage(&c, "t1").await;
        // Matches ID, key and state, but not the token.
        assert!(!c.complete_cleanup(&stale).await.unwrap());
        assert_eq!(usage(&c, "t1").await, usage_before);
        // The new row purges with its own token.
        exec(&c, "UPDATE {f} SET lease_until = 1").await;
        let fresh = c.claim_cleanup(LONG, 10).await.unwrap().remove(0);
        assert!(c.complete_cleanup(&fresh).await.unwrap());
        c.close().await;
    }

    #[tokio::test]
    #[ignore = "requires disposable PostgreSQL via DATABASE_URL"]
    async fn claim_lease_is_the_retry_delay() {
        let c = catalog().await;
        deletion_pending(&c, &owner("t1", "alice"), 1).await;
        let first = c.claim_cleanup(LONG, 10).await.unwrap();
        assert_eq!(first.len(), 1);
        // A failed worker does nothing: the held lease keeps the row hidden.
        assert!(c.claim_cleanup(LONG, 10).await.unwrap().is_empty());
        // Once the lease expires another worker reclaims it.
        exec(&c, "UPDATE {f} SET lease_until = 1").await;
        let second = c.claim_cleanup(LONG, 10).await.unwrap();
        assert_eq!(second.len(), 1);
        // The reclaim fences the first worker.
        assert_ne!(first[0].claim_token, second[0].claim_token);
        assert!(!c.complete_cleanup(&first[0]).await.unwrap());
        assert!(c.complete_cleanup(&second[0]).await.unwrap());
        assert!(!c.complete_cleanup(&second[0]).await.unwrap());
        // Failed rows are claimed too.
        c.create_pending(&new_file("p", &owner("t1", "alice"), 1))
            .await
            .unwrap();
        c.fail_upload(&owner("t1", "alice"), "p").await.unwrap();
        let failed = c.claim_cleanup(LONG, 10).await.unwrap();
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].state, CleanupState::Failed);
        c.close().await;
    }

    // ---- failures ----

    #[tokio::test]
    #[ignore = "requires disposable PostgreSQL via DATABASE_URL"]
    async fn unavailable_database_fails_fast() {
        let c = catalog().await;
        c.close().await;
        let err = c
            .create_pending(&new_file("f1", &owner("t1", "a"), 1))
            .await
            .unwrap_err();
        assert_eq!(failure_class(&err), PgFailure::Unavailable);
        let err = c.get_ready(&owner("t1", "a"), "f1").await.unwrap_err();
        assert_eq!(failure_class(&err), PgFailure::Unavailable);
    }

    #[tokio::test]
    #[ignore = "requires disposable PostgreSQL via DATABASE_URL"]
    async fn terminated_backend_is_retried() {
        let c = catalog().await;
        let attempts = std::sync::atomic::AtomicU32::new(0);
        let result = with_retry(|| async {
            let n = attempts.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let mut conn = c.pool.acquire().await.map_err(|e| c.pg_err(&e))?;
            if n == 0 {
                let pid: i32 = sqlx::query_scalar("SELECT pg_backend_pid()")
                    .fetch_one(&mut *conn)
                    .await
                    .map_err(|e| c.pg_err(&e))?;
                let admin = sqlx::postgres::PgPoolOptions::new()
                    .connect(&database_url())
                    .await
                    .unwrap();
                // The timeout makes the call wait until the backend has exited.
                let terminated: bool = sqlx::query_scalar("SELECT pg_terminate_backend($1, 5000)")
                    .bind(pid)
                    .fetch_one(&admin)
                    .await
                    .unwrap();
                assert!(terminated, "the backend must be gone before the retry is exercised");
                admin.close().await;
            }
            sqlx::query("SELECT 1")
                .execute(&mut *conn)
                .await
                .map_err(|e| c.pg_err(&e))?;
            Ok(n)
        })
        .await;
        assert_eq!(result.unwrap(), 1, "the terminated attempt must be retried");
        c.close().await;
    }

    #[tokio::test]
    #[ignore = "requires disposable PostgreSQL via DATABASE_URL"]
    async fn non_retryable_error_surfaces_as_backend() {
        let c = catalog().await;
        exec(&c, "DROP TABLE {f}").await;
        let err = c
            .create_pending(&new_file("f1", &owner("t1", "a"), 1))
            .await
            .unwrap_err();
        assert_eq!(failure_class(&err), PgFailure::NonRetryable);
        c.close().await;
    }

    // ---- indexes ----

    /// Seed a mixed-state table large enough for the planner to prefer the indexes.
    async fn seed_indexed_table(c: &PostgresFileCatalog) {
        exec(
            c,
            "INSERT INTO {f} (id, tenant_id, owner_issuer, owner_subject, storage_key, filename, state, bytes, \
                              created_at, expires_at) \
             SELECT 'id' || n, 'tenant' || (n % 100), 'i', 's', 'k' || n, 'f', \
                    CASE WHEN n % 50 = 0 THEN 'pending' WHEN n % 40 = 1 THEN 'failed' \
                         WHEN n % 40 = 2 THEN 'deletion_pending' ELSE 'ready' END, \
                    1, n, CASE WHEN n % 10 = 0 THEN n END \
             FROM generate_series(1, 500) n",
        )
        .await;
        exec(c, "ANALYZE {f}").await;
    }

    /// The plan for `sql` with sequential scans, bitmap scans and sorts discouraged,
    /// so a missing or unusable index shows up as a plan without that index.
    async fn explain(c: &PostgresFileCatalog, sql: &str) -> String {
        let mut tx = c.pool.begin().await.unwrap();
        // Sorts and bitmap scans are disabled too: on a small table the planner would
        // otherwise pick bitmap scan + sort, hiding whether the index gives the order.
        sqlx::raw_sql("SET LOCAL enable_seqscan = off; SET LOCAL enable_bitmapscan = off; SET LOCAL enable_sort = off")
            .execute(&mut *tx)
            .await
            .unwrap();
        let plan: Vec<String> =
            sqlx::query_scalar(sqlx::AssertSqlSafe(format!("EXPLAIN {}", sql.replace("{f}", &c.files))))
                .fetch_all(&mut *tx)
                .await
                .unwrap();
        tx.rollback().await.unwrap();
        plan.join("\n")
    }

    /// Assert the plan scans `index` and needs no sort.
    fn assert_uses(plan: &str, c: &PostgresFileCatalog, suffix: &str, sorted: bool) {
        assert!(plan.contains(&format!("{}{suffix}", c.files)), "plan:\n{plan}");
        assert!(!plan.contains("Seq Scan"), "plan:\n{plan}");
        assert!(!sorted || !plan.contains("Sort"), "plan:\n{plan}");
    }

    #[tokio::test]
    #[ignore = "requires disposable PostgreSQL via DATABASE_URL"]
    async fn quota_sum_uses_list_index() {
        let c = catalog().await;
        seed_indexed_table(&c).await;
        let plan = explain(
            &c,
            "SELECT coalesce(sum(bytes), 0) FROM {f} WHERE tenant_id = 'tenant7'",
        )
        .await;
        assert_uses(&plan, &c, "_list_idx", false);
        c.close().await;
    }

    #[tokio::test]
    #[ignore = "requires disposable PostgreSQL via DATABASE_URL"]
    async fn listing_uses_list_index_in_both_directions_without_sort() {
        let c = catalog().await;
        seed_indexed_table(&c).await;
        for (cmp, dir) in [(">", "ASC"), ("<", "DESC")] {
            let plan = explain(
                &c,
                &format!(
                    "SELECT id FROM {{f}} WHERE tenant_id = 'tenant7' AND state = 'ready' \
                     AND (created_at, id) {cmp} (5000, 'id5007') \
                     ORDER BY created_at {dir}, id {dir} LIMIT 11"
                ),
            )
            .await;
            assert_uses(&plan, &c, "_list_idx", true);
        }
        c.close().await;
    }

    #[tokio::test]
    #[ignore = "requires disposable PostgreSQL via DATABASE_URL"]
    async fn purpose_filtered_listing_still_uses_list_index() {
        let c = catalog().await;
        seed_indexed_table(&c).await;
        let plan = explain(
            &c,
            "SELECT id FROM {f} WHERE tenant_id = 'tenant7' AND state = 'ready' AND purpose = 'x' \
             ORDER BY created_at, id LIMIT 11",
        )
        .await;
        assert_uses(&plan, &c, "_list_idx", true);
        c.close().await;
    }

    #[tokio::test]
    #[ignore = "requires disposable PostgreSQL via DATABASE_URL"]
    async fn expiry_sweep_uses_expiry_index() {
        let c = catalog().await;
        seed_indexed_table(&c).await;
        let plan = explain(
            &c,
            "SELECT id FROM {f} WHERE state = 'ready' AND expires_at <= 100 LIMIT 100 FOR UPDATE SKIP LOCKED",
        )
        .await;
        assert_uses(&plan, &c, "_expiry_idx", false);
        c.close().await;
    }

    #[tokio::test]
    #[ignore = "requires disposable PostgreSQL via DATABASE_URL"]
    async fn abandoned_upload_sweep_uses_pending_index() {
        let c = catalog().await;
        seed_indexed_table(&c).await;
        let plan = explain(
            &c,
            "SELECT id FROM {f} WHERE state = 'pending' AND created_at <= 100 LIMIT 100 FOR UPDATE SKIP LOCKED",
        )
        .await;
        assert_uses(&plan, &c, "_pending_idx", false);
        c.close().await;
    }

    #[tokio::test]
    #[ignore = "requires disposable PostgreSQL via DATABASE_URL"]
    async fn cleanup_claim_uses_lease_index_without_sort() {
        let c = catalog().await;
        seed_indexed_table(&c).await;
        let plan = explain(
            &c,
            "SELECT id FROM {f} WHERE state IN ('failed', 'deletion_pending') \
             AND lease_until <= extract(epoch FROM now())::bigint + 1 \
             ORDER BY lease_until LIMIT 100 FOR UPDATE SKIP LOCKED",
        )
        .await;
        assert_uses(&plan, &c, "_lease_idx", true);
        c.close().await;
    }
}
