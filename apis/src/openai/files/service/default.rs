// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Default implementation of the Files service.

use std::sync::Arc;

use async_trait::async_trait;
use praxis_ai_store::{FileCatalog, StateOwner};

use super::{FileDownload, FileService, FileServiceError, FileUploadSession};
use crate::openai::files::{FileInfo, FileList, FilePage, FileStore};

/// Default implementation of the Files service.
pub struct DefaultFileService {
    /// Metadata catalog.
    catalog: Arc<dyn FileCatalog>,
    /// Byte store.
    #[expect(dead_code, reason = "used by upload and content (#500, #501)")]
    store: Arc<dyn FileStore>,
}

impl DefaultFileService {
    /// Create a service over a metadata catalog and a byte store.
    pub fn new(catalog: Arc<dyn FileCatalog>, store: Arc<dyn FileStore>) -> Self {
        Self { catalog, store }
    }
}

#[expect(
    clippy::unimplemented,
    reason = "upload and content are not implemented yet (#500, #501)"
)]
#[async_trait]
impl FileService for DefaultFileService {
    async fn begin_upload(
        &self,
        _owner: &StateOwner,
        _filename: String,
        _size_hint: Option<u64>,
    ) -> Result<Box<dyn FileUploadSession + 'static>, FileServiceError> {
        unimplemented!()
    }

    async fn retrieve(&self, owner: &StateOwner, id: &str) -> Result<FileInfo, FileServiceError> {
        let stored = self
            .catalog
            .get_ready(owner, id)
            .await?
            .ok_or(FileServiceError::NotFound)?;
        Ok(FileInfo::try_from(stored.record)?)
    }

    async fn list(&self, owner: &StateOwner, options: &FileList) -> Result<FilePage, FileServiceError> {
        let page = self.catalog.list_ready(owner, &options.into()).await?;
        Ok(FilePage::try_from(page)?)
    }

    async fn content(&self, _owner: &StateOwner, _id: &str) -> Result<FileDownload, FileServiceError> {
        unimplemented!()
    }

    async fn delete(&self, owner: &StateOwner, id: &str) -> Result<(), FileServiceError> {
        // A file already marked for deletion is reported as not found, like
        // every other hidden file: only the call that hides it succeeds.
        if self.catalog.mark_for_deletion(owner, id).await? {
            Ok(())
        } else {
            Err(FileServiceError::NotFound)
        }
    }
}

#[cfg(all(test, feature = "_store-postgres"))]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::tests_outside_test_module,
    reason = "tests"
)]
mod tests {
    //! `DefaultFileService::{retrieve, list, delete}` against a real `PostgreSQL`
    //! catalog.
    //!
    //! Every test needs a disposable `PostgreSQL` via `DATABASE_URL` and is
    //! `#[ignore]`d (`make test-postgres-unit` runs them). Files are created
    //! through the catalog because upload is not part of the service yet; the
    //! assertions go through the service only.

    use std::{sync::Arc, time::Duration};

    use praxis_ai_store::{FileCatalog, FileCompletion, NewFile, SslMode, StateOwner};
    use praxis_ai_store_backends::{PgTlsConfig, PostgresFileCatalog};
    use sqlx::{AssertSqlSafe, PgPool};

    use super::DefaultFileService;
    use crate::openai::files::{
        FileInfo, FileList, FileListOrder, FilePurpose, FileService as _, FileServiceError,
        store::unavailable::UnavailableFileStore,
    };

    /// Per-tenant quota, far above anything the tests store.
    const LIMIT: u64 = 1_000_000;

    /// A catalog, a service over it, and a side pool for fixture SQL.
    struct Fixture {
        /// Catalog used to create fixture files.
        catalog: Arc<PostgresFileCatalog>,
        /// Service under test.
        service: DefaultFileService,
        /// Side connection for backdating rows.
        pool: PgPool,
        /// Catalog table name.
        table: String,
    }

    impl Fixture {
        async fn new() -> Self {
            let url = std::env::var("DATABASE_URL").expect("disposable test database");
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let table = format!("files_svc_{nanos}");
            let tls = PgTlsConfig {
                ssl_mode: Some(SslMode::Disable),
                ..PgTlsConfig::default()
            };
            let catalog = Arc::new(PostgresFileCatalog::new(&url, &table, &tls, None, LIMIT).await.unwrap());
            let pool = PgPool::connect(&url).await.unwrap();
            let dyn_catalog: Arc<dyn FileCatalog> = Arc::<PostgresFileCatalog>::clone(&catalog);
            let service = DefaultFileService::new(dyn_catalog, Arc::new(UnavailableFileStore));
            Self {
                catalog,
                service,
                pool,
                table,
            }
        }

        /// Create a ready file with `bytes` bytes and `purpose`.
        async fn ready(&self, owner: &StateOwner, id: &str, bytes: u64, purpose: &str) {
            self.catalog.create_pending(&new_file(id, owner, bytes)).await.unwrap();
            self.catalog
                .complete_upload(owner, id, completion(bytes, purpose, None))
                .await
                .unwrap();
        }

        /// Force `created_at` so ordering does not depend on the clock.
        async fn set_created_at(&self, id: &str, created_at: i64) {
            let sql = format!("UPDATE {} SET created_at = {created_at} WHERE id = '{id}'", self.table);
            sqlx::query(AssertSqlSafe(sql)).execute(&self.pool).await.unwrap();
        }

        async fn list_ids(&self, owner: &StateOwner, list: &FileList) -> (Vec<String>, bool) {
            let page = self.service.list(owner, list).await.unwrap();
            (page.data.into_iter().map(|f| f.id).collect(), page.has_more)
        }
    }

    fn owner(tenant: &str) -> StateOwner {
        StateOwner::from_trusted_parts(tenant, "issuer", "subject").unwrap()
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

    fn completion(bytes: u64, purpose: &str, expires_after: Option<Duration>) -> FileCompletion {
        FileCompletion {
            bytes,
            purpose: purpose.to_owned(),
            expires_after,
        }
    }

    fn list(limit: u32, order: FileListOrder) -> FileList {
        FileList {
            purpose: None,
            limit,
            order,
            after: None,
        }
    }

    fn ids(values: &[&str]) -> Vec<String> {
        values.iter().map(|s| (*s).to_owned()).collect()
    }

    // ---- retrieve ----

    #[tokio::test]
    #[ignore = "requires disposable PostgreSQL via DATABASE_URL"]
    async fn retrieve_returns_ready_file() {
        let f = Fixture::new().await;
        let a = owner("a");
        f.catalog.create_pending(&new_file("f1", &a, 10)).await.unwrap();
        let record = f
            .catalog
            .complete_upload(&a, "f1", completion(7, "batch", Some(Duration::from_secs(3_600))))
            .await
            .unwrap();

        let info = f.service.retrieve(&a, "f1").await.unwrap();
        assert_eq!(
            info,
            FileInfo {
                id: "f1".to_owned(),
                bytes: 7,
                created_at: record.created_at,
                expires_at: Some(record.created_at + 3_600),
                filename: "f1.txt".to_owned(),
                purpose: FilePurpose::Batch,
            }
        );
    }

    #[tokio::test]
    #[ignore = "requires disposable PostgreSQL via DATABASE_URL"]
    async fn retrieve_hides_every_non_visible_file() {
        let f = Fixture::new().await;
        let a = owner("a");
        let b = owner("b");

        f.ready(&a, "ready", 1, "assistants").await;
        f.catalog.create_pending(&new_file("pending", &a, 1)).await.unwrap();
        f.catalog.create_pending(&new_file("failed", &a, 1)).await.unwrap();
        f.catalog.fail_upload(&a, "failed").await.unwrap();
        f.ready(&a, "deleting", 1, "assistants").await;
        f.catalog.mark_for_deletion(&a, "deleting").await.unwrap();
        f.catalog.create_pending(&new_file("expired", &a, 1)).await.unwrap();
        f.catalog
            .complete_upload(&a, "expired", completion(1, "assistants", Some(Duration::ZERO)))
            .await
            .unwrap();

        f.service.retrieve(&a, "ready").await.unwrap();
        for id in ["missing", "pending", "failed", "deleting", "expired"] {
            assert!(
                matches!(f.service.retrieve(&a, id).await, Err(FileServiceError::NotFound)),
                "{id}"
            );
        }
        assert!(
            matches!(f.service.retrieve(&b, "ready").await, Err(FileServiceError::NotFound)),
            "other tenant"
        );
    }

    // ---- list ----

    #[tokio::test]
    #[ignore = "requires disposable PostgreSQL via DATABASE_URL"]
    async fn list_orders_by_created_at_then_id() {
        let f = Fixture::new().await;
        let a = owner("a");
        for id in ["c", "a", "b", "d"] {
            f.ready(&a, id, 1, "assistants").await;
        }
        // `a`, `b` and `c` tie on created_at; `d` is newer.
        for id in ["a", "b", "c"] {
            f.set_created_at(id, 100).await;
        }
        f.set_created_at("d", 200).await;

        let (asc, has_more) = f.list_ids(&a, &list(10, FileListOrder::Asc)).await;
        assert_eq!(asc, ids(&["a", "b", "c", "d"]));
        assert!(!has_more);
        let (desc, _) = f.list_ids(&a, &list(10, FileListOrder::Desc)).await;
        assert_eq!(desc, ids(&["d", "c", "b", "a"]));
    }

    #[tokio::test]
    #[ignore = "requires disposable PostgreSQL via DATABASE_URL"]
    async fn list_limit_and_has_more() {
        let f = Fixture::new().await;
        let a = owner("a");
        for (id, ts) in ["a", "b", "c"].into_iter().zip(100_i64..) {
            f.ready(&a, id, 1, "assistants").await;
            f.set_created_at(id, ts).await;
        }

        assert_eq!(
            f.list_ids(&a, &list(2, FileListOrder::Asc)).await,
            (ids(&["a", "b"]), true)
        );
        // A page that exactly drains the list has no successor.
        assert_eq!(
            f.list_ids(&a, &list(3, FileListOrder::Asc)).await,
            (ids(&["a", "b", "c"]), false)
        );
        assert_eq!(
            f.list_ids(&owner("empty"), &list(10, FileListOrder::Asc)).await,
            (vec![], false)
        );
    }

    #[tokio::test]
    #[ignore = "requires disposable PostgreSQL via DATABASE_URL"]
    async fn list_filters_by_purpose() {
        let f = Fixture::new().await;
        let a = owner("a");
        f.ready(&a, "x", 1, "assistants").await;
        f.ready(&a, "y", 1, "batch").await;
        f.set_created_at("x", 100).await;
        f.set_created_at("y", 101).await;

        let query = FileList {
            purpose: Some("batch".to_owned()),
            ..list(10, FileListOrder::Asc)
        };
        assert_eq!(f.list_ids(&a, &query).await, (ids(&["y"]), false));
    }

    #[tokio::test]
    #[ignore = "requires disposable PostgreSQL via DATABASE_URL"]
    async fn list_after_cursor_pages_through_all_files() {
        let f = Fixture::new().await;
        let a = owner("a");
        for (id, i) in ["a", "b", "c", "d", "e"].into_iter().zip(0_i64..) {
            f.ready(&a, id, 1, "assistants").await;
            // Pairs tie on created_at so the cursor must break ties by id.
            f.set_created_at(id, 100 + i / 2).await;
        }

        let mut seen = Vec::new();
        let mut after = None;
        loop {
            let query = FileList {
                after: after.clone(),
                ..list(2, FileListOrder::Asc)
            };
            let page = f.service.list(&a, &query).await.unwrap();
            seen.extend(page.data.iter().map(|file| file.id.clone()));
            if !page.has_more {
                break;
            }
            after = page.data.last().map(|file| file.id.clone());
        }
        assert_eq!(seen, ids(&["a", "b", "c", "d", "e"]));
    }

    #[tokio::test]
    #[ignore = "requires disposable PostgreSQL via DATABASE_URL"]
    async fn list_invalid_cursor_is_invalid_input() {
        let f = Fixture::new().await;
        let a = owner("a");
        f.ready(&a, "mine", 1, "assistants").await;
        f.ready(&owner("b"), "theirs", 1, "assistants").await;

        for cursor in ["missing", "theirs"] {
            let query = FileList {
                after: Some(cursor.to_owned()),
                ..list(10, FileListOrder::Asc)
            };
            assert!(
                matches!(f.service.list(&a, &query).await, Err(FileServiceError::InvalidInput(_))),
                "{cursor}"
            );
        }
    }
    // ---- delete ----

    #[tokio::test]
    #[ignore = "requires disposable PostgreSQL via DATABASE_URL"]
    async fn delete_hides_the_file_immediately() {
        let f = Fixture::new().await;
        let a = owner("a");
        f.ready(&a, "gone", 1, "assistants").await;
        f.ready(&a, "kept", 1, "assistants").await;

        f.service.delete(&a, "gone").await.unwrap();

        assert!(matches!(
            f.service.retrieve(&a, "gone").await,
            Err(FileServiceError::NotFound)
        ));
        assert_eq!(
            f.list_ids(&a, &list(10, FileListOrder::Asc)).await,
            (ids(&["kept"]), false)
        );
    }

    #[tokio::test]
    #[ignore = "requires disposable PostgreSQL via DATABASE_URL"]
    async fn delete_twice_is_not_found() {
        let f = Fixture::new().await;
        let a = owner("a");
        f.ready(&a, "f1", 1, "assistants").await;

        f.service.delete(&a, "f1").await.unwrap();
        assert!(matches!(
            f.service.delete(&a, "f1").await,
            Err(FileServiceError::NotFound)
        ));
    }

    #[tokio::test]
    #[ignore = "requires disposable PostgreSQL via DATABASE_URL"]
    async fn delete_rejects_every_non_visible_file() {
        let f = Fixture::new().await;
        let a = owner("a");
        let b = owner("b");

        f.ready(&a, "ready", 1, "assistants").await;
        f.catalog.create_pending(&new_file("pending", &a, 1)).await.unwrap();
        f.catalog.create_pending(&new_file("failed", &a, 1)).await.unwrap();
        f.catalog.fail_upload(&a, "failed").await.unwrap();
        f.catalog.create_pending(&new_file("expired", &a, 1)).await.unwrap();
        f.catalog
            .complete_upload(&a, "expired", completion(1, "assistants", Some(Duration::ZERO)))
            .await
            .unwrap();

        for id in ["missing", "pending", "failed", "expired"] {
            assert!(
                matches!(f.service.delete(&a, id).await, Err(FileServiceError::NotFound)),
                "{id}"
            );
        }
        assert!(
            matches!(f.service.delete(&b, "ready").await, Err(FileServiceError::NotFound)),
            "other tenant"
        );
        // The rejected attempts left the owner's file untouched.
        f.service.retrieve(&a, "ready").await.unwrap();
    }
}
