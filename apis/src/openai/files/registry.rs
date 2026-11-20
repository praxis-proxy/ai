// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Durable file metadata and lifecycle transitions.

use async_trait::async_trait;
use praxis_ai_store::StateOwner;

use super::{
    FileCleanupCandidate, FileCompletion, FileInfo, FileList, FilePage, FileUploadLimits, PendingFile, StoredFile,
};

/// Failures at the durable metadata boundary.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum FileRegistryError {
    /// An ID or key already exists, or a conditional state transition failed.
    #[error("file state conflict")]
    Conflict,
    /// No capacity can be reserved for a new upload in the owner's project.
    #[error("project file storage limit exceeded")]
    QuotaExceeded,
    /// The list cursor is invalid in this owner scope.
    #[error("invalid file list cursor")]
    InvalidCursor,
    /// The metadata backend failed; its original error remains available.
    #[error("file registry backend failed")]
    Backend(#[source] Box<dyn std::error::Error + Send + Sync>),
}

/// Owner-scoped metadata operations for a SQLite, `PostgreSQL`, or other backend.
///
/// Implementations must make every state transition and charge adjustment
/// atomic. Request-driven reads and writes use the exact [`StateOwner`] and
/// hide other owners' files as if they did not exist. Quota totals include
/// other owners in the same project without exposing their files. Reaper
/// methods deliberately scan across owners.
#[async_trait]
pub trait FileRegistry: Send + Sync {
    /// Atomically insert pending metadata and reserve upload capacity before
    /// any bytes are written. Count stored files, including hidden or expired
    /// files whose bytes remain, and all pending or failed reservations in the
    /// same project. Compute `remaining_quota` by subtracting those charges
    /// from `limits.project_quota`, flooring the result at zero. Reserve and
    /// return `R = min(limits.max_file_size, remaining_quota)`. The service
    /// passes `R` to the byte-store writer and retains it in the upload session.
    /// Purpose and expiration may arrive later in the multipart form. IDs and
    /// storage keys must not be reused, including after cleanup.
    ///
    /// # Errors
    ///
    /// Returns [`FileRegistryError::QuotaExceeded`] without inserting a row
    /// when `R` is zero, [`FileRegistryError::Conflict`] for a duplicate ID or
    /// storage key, or another [`FileRegistryError`] on persistence failure.
    async fn create_pending(&self, file: &PendingFile, limits: FileUploadLimits) -> Result<u64, FileRegistryError>;

    /// Atomically transition a pending file to ready and replace its durable
    /// reservation `R` with the actual byte count `A` from the finished writer,
    /// releasing `R - A`. For `A <= R`, do not make a fresh quota decision:
    /// capacity was reserved before writing. Record the validated purpose and
    /// expiration in the same transition. Retrying after an ambiguous commit
    /// returns the existing ready file only when all completion fields match.
    /// The resulting metadata is safe to expose.
    ///
    /// # Errors
    ///
    /// Returns [`FileRegistryError::Conflict`] when the file is neither pending
    /// nor ready with the same completion fields, or when `A > R`. Returns
    /// another [`FileRegistryError`] on persistence failure.
    async fn complete_upload(
        &self,
        owner: &StateOwner,
        id: &str,
        completion: FileCompletion,
    ) -> Result<FileInfo, FileRegistryError>;

    /// Transition a pending upload to failed after a stream, store, or
    /// reconciled completion error. Keep its reservation charged until byte
    /// deletion is confirmed by [`Self::complete_cleanup`]. A repeated failure
    /// transition is idempotent; a ready file must never be demoted. Failed
    /// files become cleanup candidates.
    ///
    /// # Errors
    ///
    /// Returns [`FileRegistryError::Conflict`] for a file already ready or
    /// marked for deletion, or another [`FileRegistryError`] on persistence
    /// failure.
    async fn fail_upload(&self, owner: &StateOwner, id: &str) -> Result<(), FileRegistryError>;

    /// Fetch an unexpired ready file and its private byte-store key. Missing,
    /// expired as of `now`, pending, failed, deletion-pending, and differently
    /// owned files return `None`. `now` is a Unix timestamp in seconds.
    ///
    /// # Errors
    ///
    /// Returns [`FileRegistryError`] on persistence failure.
    async fn get_ready(&self, owner: &StateOwner, id: &str, now: i64) -> Result<Option<StoredFile>, FileRegistryError>;

    /// List unexpired ready files only. Ordering is by `(created_at, id)` in
    /// the requested direction, with `after` interpreted in the same owner
    /// scope. Files expired as of `now` are excluded; `now` is a Unix timestamp
    /// in seconds.
    ///
    /// # Errors
    ///
    /// Returns [`FileRegistryError::InvalidCursor`] for an invalid cursor, or
    /// another [`FileRegistryError`] on persistence failure.
    async fn list_ready(&self, owner: &StateOwner, options: &FileList, now: i64)
    -> Result<FilePage, FileRegistryError>;

    /// Atomically hide a ready file and mark it for physical deletion. Keep
    /// its actual bytes charged until [`Self::complete_cleanup`].
    /// Returns `false` for missing, expired as of `now`, already hidden, or
    /// differently owned files. `now` is a Unix timestamp in seconds.
    ///
    /// # Errors
    ///
    /// Returns [`FileRegistryError`] on persistence failure.
    async fn mark_for_deletion(&self, owner: &StateOwner, id: &str, now: i64) -> Result<bool, FileRegistryError>;

    /// Mark up to `limit` expired ready files for deletion across all owners.
    /// Keep their actual bytes charged until [`Self::complete_cleanup`].
    /// Returns the number of newly marked files. A scheduled reaper sweep can
    /// call this before reading cleanup candidates.
    ///
    /// # Errors
    ///
    /// Returns [`FileRegistryError`] on persistence failure.
    async fn mark_expired_for_deletion(&self, now: i64, limit: u32) -> Result<u32, FileRegistryError>;

    /// Mark up to `limit` abandoned pending uploads as failed, across owners.
    /// The reaper must choose `created_before` older than the maximum active
    /// upload duration plus a safety margin. The service must stop writes
    /// before that cutoff can include an active upload. This recovers pending
    /// entries left by process crashes or a failed registry update. Their
    /// reservations remain charged until [`Self::complete_cleanup`].
    ///
    /// # Errors
    ///
    /// Returns [`FileRegistryError`] on persistence failure.
    async fn fail_abandoned_uploads(&self, created_before: i64, limit: u32) -> Result<u32, FileRegistryError>;

    /// List up to `limit` failed uploads and deletion-pending files for a
    /// reaper, across all owners, ordered by ID. `after` is an exclusive ID
    /// cursor so a failed early task does not hide later work. Multiple
    /// reapers may see the same task; byte-store deletion and completion must
    /// therefore be idempotent. Listed tasks still hold their quota charge.
    ///
    /// # Errors
    ///
    /// Returns [`FileRegistryError`] on persistence failure.
    async fn cleanup_candidates(
        &self,
        after: Option<&str>,
        limit: u32,
    ) -> Result<Vec<FileCleanupCandidate>, FileRegistryError>;

    /// Remove metadata and release its charge atomically only after byte-store
    /// deletion succeeded. Release the reserved bytes of a failed upload or
    /// the actual bytes of a deletion-pending file exactly once. The row must
    /// still match the candidate's owner, ID, storage key, and cleanup state.
    /// Returns `false` if another reaper already completed it. A failed purge
    /// leaves the marker and charge available for a later retry.
    ///
    /// # Errors
    ///
    /// Returns [`FileRegistryError`] on persistence failure.
    async fn complete_cleanup(&self, candidate: &FileCleanupCandidate) -> Result<bool, FileRegistryError>;
}
