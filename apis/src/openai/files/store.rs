// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Streaming byte I/O for local or remote file storage.

use async_trait::async_trait;
use bytes::Bytes;

use super::FileStream;

/// Failures while writing or reading file bytes.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum FileStoreError {
    /// The upload exceeded the writer's byte limit, which may be a quota
    /// reservation smaller than the per-file size limit.
    #[error("file exceeds the writer byte limit")]
    UploadTooLarge,
    /// The byte store failed; its original error remains available.
    #[error("file byte store failed")]
    Backend(#[source] Box<dyn std::error::Error + Send + Sync>),
}

/// An incremental byte-store upload, independent of file metadata and HTTP.
/// Only [`Self::finish`] publishes a complete object under the storage key.
/// Dropping an unfinished writer must not publish an object; implementations
/// must arrange eventual cleanup of any uncommitted backend state. The service
/// retains the registry reservation until byte deletion is confirmed.
#[async_trait]
pub trait FileStoreWriter: Send {
    /// Move one chunk to the store without buffering the entire file. Apply
    /// backpressure and reject an upload as soon as it exceeds the reserved
    /// limit passed to [`FileStore::begin_write`]. The service checks the same
    /// limit before calling this method so it can report the correct client
    /// error for either a quota or per-file bound.
    ///
    /// # Errors
    ///
    /// Returns [`FileStoreError::UploadTooLarge`] when the limit is exceeded,
    /// or [`FileStoreError::Backend`] for a storage failure. The caller must
    /// abort after an unsuccessful chunk write.
    async fn write_chunk(&mut self, chunk: Bytes) -> Result<(), FileStoreError>;

    /// Commit the complete object and return the actual number of bytes
    /// written, no greater than the reserved limit. A successful return means
    /// the object is durable. On failure, abort or remove partial bytes where
    /// possible; the pending registry entry and reservation remain eligible
    /// for reaper cleanup after an uncertain commit.
    ///
    /// # Errors
    ///
    /// Returns [`FileStoreError`] if the object could not be committed.
    async fn finish(self: Box<Self>) -> Result<u64, FileStoreError>;

    /// Discard an unfinished upload and remove partial bytes where possible.
    /// The service calls this after a malformed form or failed chunk write;
    /// its registry reservation remains charged until deletion is confirmed.
    ///
    /// # Errors
    ///
    /// Returns [`FileStoreError`] if cleanup requires a later retry.
    async fn abort(self: Box<Self>) -> Result<(), FileStoreError>;
}

/// Byte storage independent of file metadata and HTTP transport.
///
/// Keys come from trusted pending registry entries, never client filenames.
/// A store implementation may use temporary objects or multipart upload
/// internally; the writer commits them only on successful completion.
#[async_trait]
pub trait FileStore: Send + Sync {
    /// Start a byte write to `key` with the capacity `reserved_bytes` returned
    /// by [`super::FileRegistry::create_pending`]. The service creates the
    /// pending registry entry before calling this method and never opens a
    /// writer when the reservation is zero.
    ///
    /// # Errors
    ///
    /// Returns [`FileStoreError`] if the writer cannot be opened. The returned
    /// writer owns its state so it can outlive the borrow of this store.
    async fn begin_write(
        &self,
        key: &str,
        reserved_bytes: u64,
    ) -> Result<Box<dyn FileStoreWriter + 'static>, FileStoreError>;

    /// Open `key` without buffering the whole object. Failures after opening
    /// are reported as items in the returned stream.
    ///
    /// # Errors
    ///
    /// Returns [`FileStoreError`] if the object cannot be opened.
    async fn read(&self, key: &str) -> Result<FileStream<FileStoreError>, FileStoreError>;

    /// Physically remove `key`. Deleting an absent object succeeds so reapers
    /// can retry safely after a crash or a registry failure. A successful
    /// return lets the registry release the retained charge in
    /// [`super::FileRegistry::complete_cleanup`].
    ///
    /// # Errors
    ///
    /// Returns [`FileStoreError`] on a storage failure that needs a retry.
    async fn delete(&self, key: &str) -> Result<(), FileStoreError>;
}
