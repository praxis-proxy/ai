// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! [`FileCatalog`] implementation stubs.
//!
//! The constructor, schema, error classification and retry wrapper are in
//! place; the operations themselves arrive in steps 4–5. Each stub fails with
//! a non-retryable backend error so the crate compiles, the trait is
//! satisfied, and an accidental early call fails loudly instead of silently.

use std::time::Duration;

use async_trait::async_trait;
use praxis_ai_store::{
    CatalogError, CleanupClaim, DeletionOutcome, FileCatalog, FileCompletion, FileListQuery, FileRecord,
    FileRecordPage, NewFile, StateOwner, StoredFile,
};

use super::{
    PostgresFileCatalog,
    retry::{PgBackendError, PgFailure},
};

/// The error every operation returns until steps 4–5 land.
fn not_implemented(operation: &str) -> CatalogError {
    CatalogError::Backend(Box::new(PgBackendError::new(
        PgFailure::NonRetryable,
        format!("`PostgresFileCatalog::{operation}` is not implemented yet"),
    )))
}

#[async_trait]
impl FileCatalog for PostgresFileCatalog {
    async fn create_pending(&self, _file: &NewFile) -> Result<i64, CatalogError> {
        Err(not_implemented("create_pending"))
    }

    async fn complete_upload(
        &self,
        _owner: &StateOwner,
        _id: &str,
        _completion: FileCompletion,
    ) -> Result<FileRecord, CatalogError> {
        Err(not_implemented("complete_upload"))
    }

    async fn fail_upload(&self, _owner: &StateOwner, _id: &str) -> Result<(), CatalogError> {
        Err(not_implemented("fail_upload"))
    }

    async fn get_ready(&self, _owner: &StateOwner, _id: &str) -> Result<Option<StoredFile>, CatalogError> {
        Err(not_implemented("get_ready"))
    }

    async fn list_ready(&self, _owner: &StateOwner, _query: &FileListQuery) -> Result<FileRecordPage, CatalogError> {
        Err(not_implemented("list_ready"))
    }

    async fn mark_for_deletion(&self, _owner: &StateOwner, _id: &str) -> Result<DeletionOutcome, CatalogError> {
        Err(not_implemented("mark_for_deletion"))
    }

    async fn mark_expired_for_deletion(&self, _limit: u32) -> Result<u32, CatalogError> {
        Err(not_implemented("mark_expired_for_deletion"))
    }

    async fn fail_abandoned_uploads(&self, _abandon_after: Duration, _limit: u32) -> Result<u32, CatalogError> {
        Err(not_implemented("fail_abandoned_uploads"))
    }

    async fn claim_cleanup(&self, _lease: Duration, _limit: u32) -> Result<Vec<CleanupClaim>, CatalogError> {
        Err(not_implemented("claim_cleanup"))
    }

    async fn complete_cleanup(&self, _claim: &CleanupClaim) -> Result<bool, CatalogError> {
        Err(not_implemented("complete_cleanup"))
    }

    async fn release_cleanup(&self, _claim: &CleanupClaim) -> Result<(), CatalogError> {
        Err(not_implemented("release_cleanup"))
    }
}
