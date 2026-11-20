// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! [`FileCatalog`] implementation.
//!
//! The SQL lives in [`request`](super::request) and
//! [`reaper`](super::reaper); this module wires each operation to the trait
//! and applies the bounded retry only where a repeat is a no-op by state check:
//! `create_pending`, `complete_upload`, `fail_upload`, `mark_for_deletion`,
//! `complete_cleanup` and `release_cleanup`. `claim_cleanup` and the batch
//! markers are not retried: a repeat after an ambiguous commit would duplicate
//! work, and the reaper retries on its next sweep.

use std::time::Duration;

use async_trait::async_trait;
use praxis_ai_store::{
    CatalogError, CleanupClaim, DeletionOutcome, FileCatalog, FileCompletion, FileListQuery, FileRecord,
    FileRecordPage, NewFile, StateOwner, StoredFile,
};

use super::{PostgresFileCatalog, finish, reaper, request, retry::with_retry};

/// Map a backend-only result onto the trait error.
fn backend<T>(result: Result<T, super::retry::PgBackendError>) -> Result<T, CatalogError> {
    result.map_err(|e| CatalogError::Backend(Box::new(e)))
}

#[async_trait]
impl FileCatalog for PostgresFileCatalog {
    async fn create_pending(&self, file: &NewFile) -> Result<i64, CatalogError> {
        finish(with_retry(|| request::create_pending_once(self, file)).await)
    }

    async fn complete_upload(
        &self,
        owner: &StateOwner,
        id: &str,
        completion: FileCompletion,
    ) -> Result<FileRecord, CatalogError> {
        finish(with_retry(|| request::complete_upload_once(self, owner, id, &completion)).await)
    }

    async fn fail_upload(&self, owner: &StateOwner, id: &str) -> Result<(), CatalogError> {
        finish(with_retry(|| request::fail_upload_once(self, owner, id)).await)
    }

    async fn get_ready(&self, owner: &StateOwner, id: &str) -> Result<Option<StoredFile>, CatalogError> {
        backend(request::get_ready_impl(self, owner, id).await)
    }

    async fn list_ready(&self, owner: &StateOwner, query: &FileListQuery) -> Result<FileRecordPage, CatalogError> {
        finish(request::list_ready_impl(self, owner, query).await)
    }

    async fn mark_for_deletion(&self, owner: &StateOwner, id: &str) -> Result<DeletionOutcome, CatalogError> {
        backend(with_retry(|| request::mark_for_deletion_once(self, owner, id)).await)
    }

    async fn mark_expired_for_deletion(&self, limit: u32) -> Result<u32, CatalogError> {
        backend(reaper::mark_expired_once(self, limit).await)
    }

    async fn fail_abandoned_uploads(&self, abandon_after: Duration, limit: u32) -> Result<u32, CatalogError> {
        backend(reaper::fail_abandoned_once(self, abandon_after, limit).await)
    }

    async fn claim_cleanup(&self, lease: Duration, limit: u32) -> Result<Vec<CleanupClaim>, CatalogError> {
        backend(reaper::claim_cleanup_once(self, lease, limit).await)
    }

    async fn complete_cleanup(&self, claim: &CleanupClaim) -> Result<bool, CatalogError> {
        backend(with_retry(|| reaper::complete_cleanup_once(self, claim)).await)
    }

    async fn release_cleanup(&self, claim: &CleanupClaim) -> Result<(), CatalogError> {
        backend(with_retry(|| reaper::release_cleanup_once(self, claim)).await)
    }
}
