// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! The [`FileCatalog`] trait.

use std::time::Duration;

use async_trait::async_trait;

use super::{
    CatalogError, CleanupClaim, FileCompletion, FileListQuery, FileRecord, FileRecordPage, NewFile, StoredFile,
};
use crate::owner::StateOwner;

/// Durable file metadata, lifecycle state, tenant quota and cleanup claims.
///
/// # Scoping
///
/// Scoping is **per tenant**. Any subject in a tenant can retrieve, list,
/// download and delete that tenant's files, and all subjects in a tenant share
/// one quota. Files of other tenants behave as if they did not exist. The full
/// creating [`StateOwner`] is stored for audit. Methods take `&StateOwner` so
/// the scope can change later inside the catalog without a trait change.
/// Reaper methods work across tenants.
///
/// # Quota
///
/// Usage is reserved at [`create_pending`](Self::create_pending) and released
/// only at the physical purge ([`complete_cleanup`](Self::complete_cleanup)).
/// Failing, abandoning, deleting and expiring never touch usage.
///
/// # Invariants
///
/// 1. Every ready row points to a complete, durable object.
/// 2. Every object belongs to a row, or a reaper can reach it through a failed or deletion-pending row.
/// 3. A tenant's usage equals the sum of the charged bytes of all its rows (pending, ready, failed and
///    deletion-pending).
///
/// IDs and storage keys are never reused, including after cleanup. The caller
/// ensures this by generating both randomly (for example a UUID), never from
/// client input: unique constraints cover only live rows.
///
/// Implementations make every state transition atomic and use the database
/// clock for every timestamp.
#[async_trait]
pub trait FileCatalog: Send + Sync {
    /// Insert a pending row and atomically reserve
    /// `R = min(file.max_bytes, remaining_quota)` against the tenant limit,
    /// where `remaining_quota` is the limit minus current usage, floored at
    /// zero. Returns `R`, which the service passes to the byte-store writer
    /// and retains in the upload session. Purpose and expiration arrive later.
    ///
    /// An identical retry returns the stored reservation without reserving
    /// again, even if the remaining quota has changed since. A retry matches
    /// only a pending file whose reservation does not exceed `max_bytes`.
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError::QuotaExceeded`] without inserting when `R` is
    /// zero, [`CatalogError::Conflict`] for a duplicate ID or storage key with
    /// different fields, a non-pending file, or a reservation above `max_bytes`, or [`CatalogError::Backend`] on
    /// persistence failure.
    async fn create_pending(&self, file: &NewFile) -> Result<u64, CatalogError>;

    /// Transition a pending file to ready and settle its charge from the
    /// reservation to the actual size (`used -= reserved - actual`). The
    /// expiration is `created_at + completion.expires_after`.
    ///
    /// Retrying after an ambiguous commit returns the existing ready record
    /// when all completion fields match.
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError::Conflict`] when the file is neither pending nor
    /// ready with the same completion fields, or when `completion.bytes`
    /// exceeds the reservation; otherwise [`CatalogError::Backend`].
    async fn complete_upload(
        &self,
        owner: &StateOwner,
        id: &str,
        completion: FileCompletion,
    ) -> Result<FileRecord, CatalogError>;

    /// Transition a pending upload to failed. Idempotent; a ready file is
    /// never demoted. Failed files become cleanup candidates.
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError::Conflict`] for a file already ready or marked
    /// for deletion, or [`CatalogError::Backend`] on persistence failure.
    async fn fail_upload(&self, owner: &StateOwner, id: &str) -> Result<(), CatalogError>;

    /// Fetch an unexpired ready file and its private byte-store key. Missing,
    /// expired, pending, failed, deletion-pending and other tenants' files
    /// return `None`.
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError::Backend`] on persistence failure.
    async fn get_ready(&self, owner: &StateOwner, id: &str) -> Result<Option<StoredFile>, CatalogError>;

    /// List unexpired ready files, ordered by `(created_at, id)` in the
    /// requested direction. `after` is resolved within the tenant and must
    /// name an existing row: once a deleted file is purged, a cursor pointing
    /// at it is invalid.
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError::InvalidCursor`] for an invalid cursor, or
    /// [`CatalogError::Backend`] on persistence failure.
    async fn list_ready(&self, owner: &StateOwner, query: &FileListQuery) -> Result<FileRecordPage, CatalogError>;

    /// Atomically hide a ready file and mark it for physical deletion.
    /// Returns `true` only if this call marked the row. Any other case returns
    /// `false`: no visible file exists in the tenant scope, including a file
    /// already marked (by an earlier delete or the expiry sweep) and one
    /// already purged. A retry after an ambiguous commit therefore reports
    /// `false`.
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError::Backend`] on persistence failure.
    async fn mark_for_deletion(&self, owner: &StateOwner, id: &str) -> Result<bool, CatalogError>;

    /// Mark up to `limit` expired ready files for deletion, across tenants.
    /// Returns the number newly marked. No usage change.
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError::Backend`] on persistence failure.
    async fn mark_expired_for_deletion(&self, limit: u32) -> Result<u32, CatalogError>;

    /// Mark up to `limit` pending uploads older than `abandon_after` as
    /// failed, across tenants. `abandon_after` must exceed the maximum active
    /// upload duration plus a safety margin. Returns the number newly failed.
    /// No usage change.
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError::Backend`] on persistence failure.
    async fn fail_abandoned_uploads(&self, abandon_after: Duration, limit: u32) -> Result<u32, CatalogError>;

    /// Lease up to `limit` failed and deletion-pending rows that are due, across
    /// tenants. The lease doubles as the retry delay: a worker whose byte-store
    /// deletion fails does nothing, and the row becomes claimable again when
    /// the lease expires, as it does for a crashed worker. Only claiming writes
    /// a lease, so a stale worker cannot shorten another's. Each claim carries
    /// a fresh token; a reclaim invalidates the earlier claim.
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError::Backend`] on persistence failure.
    async fn claim_cleanup(&self, lease: Duration, limit: u32) -> Result<Vec<CleanupClaim>, CatalogError>;

    /// Purge a row after byte-store deletion succeeded and release its charged
    /// bytes, in one transaction. The row must still match the claim's ID,
    /// state, storage key and token. Returns `false` if it was already purged
    /// or reclaimed, in which case nothing is released.
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError::Backend`] on persistence failure.
    async fn complete_cleanup(&self, claim: &CleanupClaim) -> Result<bool, CatalogError>;
}
