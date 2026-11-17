// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Streaming byte I/O for local or remote file storage.

use async_trait::async_trait;

use super::{FileStream, FileUploadStream};

/// Failures while writing or reading file bytes.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum FileStoreError {
    /// The upload exceeded the per-file size limit.
    #[error("file exceeds the upload size limit")]
    UploadTooLarge,
    /// Reading the incoming upload stream failed.
    #[error("file upload stream failed")]
    Input(#[source] Box<dyn std::error::Error + Send + Sync>),
    /// The byte store failed; its original error remains available.
    #[error("file byte store failed")]
    Backend(#[source] Box<dyn std::error::Error + Send + Sync>),
}

/// Byte storage independent of file metadata and HTTP transport.
///
/// Keys come from trusted pending registry entries, never client filenames.
/// A store implementation may use temporary objects or multipart upload
/// internally, but `write` succeeds only after the complete object is durable.
#[async_trait]
pub trait FileStore: Send + Sync {
    /// Stream an upload to `key`, counting bytes actually written. Reject an
    /// oversized stream as soon as the count exceeds `max_bytes`. On any error,
    /// abort or remove partial bytes where possible; failed metadata remains
    /// eligible for reaper cleanup if that removal is incomplete.
    ///
    /// # Errors
    ///
    /// Returns [`FileStoreError::UploadTooLarge`] when the limit is exceeded,
    /// [`FileStoreError::Input`] for a failed upload stream, or
    /// [`FileStoreError::Backend`] for byte-storage failures.
    async fn write(&self, key: &str, content: FileUploadStream, max_bytes: u64) -> Result<u64, FileStoreError>;

    /// Open `key` without buffering the whole object. Failures after opening
    /// are reported as items in the returned stream.
    ///
    /// # Errors
    ///
    /// Returns [`FileStoreError`] if the object cannot be opened.
    async fn read(&self, key: &str) -> Result<FileStream<FileStoreError>, FileStoreError>;

    /// Physically remove `key`. Deleting an absent object succeeds so reapers
    /// can retry safely after a crash or a registry failure.
    ///
    /// # Errors
    ///
    /// Returns [`FileStoreError`] on a storage failure that needs a retry.
    async fn delete(&self, key: &str) -> Result<(), FileStoreError>;
}
