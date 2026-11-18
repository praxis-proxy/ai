// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Handler-facing Files operations.

mod default;

use async_trait::async_trait;
use bytes::Bytes;
pub use default::DefaultFileService;
use praxis_ai_store::StateOwner;

use super::{
    FileInfo, FileList, FilePage, FileStream, FileUploadMetadata, registry::FileRegistryError, store::FileStoreError,
};

/// Failures exposed by the Files service to its handler.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum FileServiceError {
    /// No ready file exists in the authorized owner scope.
    #[error("file not found")]
    NotFound,
    /// A request field is invalid.
    #[error("invalid file request: {0}")]
    InvalidInput(String),
    /// The upload exceeded the per-file size limit.
    #[error("file exceeds the upload size limit")]
    UploadTooLarge,
    /// The owner's project has no remaining file capacity.
    #[error("project file storage limit exceeded")]
    QuotaExceeded,
    /// Reading the incoming HTTP upload body failed.
    #[error("file upload body failed")]
    Stream(#[source] Box<dyn std::error::Error + Send + Sync>),
    /// A metadata operation failed.
    #[error("file registry error")]
    Registry(#[source] FileRegistryError),
    /// A byte-store operation failed.
    #[error("file store error")]
    Store(#[source] FileStoreError),
}

impl From<FileRegistryError> for FileServiceError {
    fn from(error: FileRegistryError) -> Self {
        match error {
            FileRegistryError::QuotaExceeded => Self::QuotaExceeded,
            FileRegistryError::InvalidCursor => Self::InvalidInput("invalid `after` cursor".to_owned()),
            other => Self::Registry(other),
        }
    }
}

impl From<FileStoreError> for FileServiceError {
    fn from(error: FileStoreError) -> Self {
        match error {
            FileStoreError::UploadTooLarge => Self::UploadTooLarge,
            other => Self::Store(other),
        }
    }
}

/// A stream and the metadata needed to form a content response.
pub struct FileDownload {
    /// Metadata of the file being downloaded.
    pub info: FileInfo,
    /// Bytes delivered incrementally to the HTTP response. The service maps
    /// late byte-store failures into its own error type as chunks arrive.
    pub content: FileStream<FileServiceError>,
}

/// An upload whose file bytes can arrive before its other multipart fields.
/// The session owns the pending file until completion or abandonment.
#[async_trait]
pub trait FileUploadSession: Send {
    /// Move one file chunk to the byte-store writer. The writer applies
    /// backpressure and enforces the upload size limit as chunks arrive.
    ///
    /// # Errors
    ///
    /// Returns [`FileServiceError`] for an oversized upload or a byte-store
    /// failure. The pending file must remain eligible for cleanup on failure.
    async fn write_chunk(&mut self, chunk: Bytes) -> Result<(), FileServiceError>;

    /// Validate metadata, finish the byte write, and commit the file with its
    /// actual byte count. The handler calls this after parsing all form fields.
    /// Batch uploads without an explicit expiration default to 30 days from
    /// the pending file's creation time.
    ///
    /// # Errors
    ///
    /// Returns [`FileServiceError`] for invalid metadata or a failed byte or
    /// registry operation. An unsuccessful upload must remain eligible for
    /// cleanup. Ambiguous registry completion must be reconciled before
    /// marking the pending file failed.
    async fn finish(self: Box<Self>, metadata: FileUploadMetadata) -> Result<FileInfo, FileServiceError>;

    /// Abandon an upload after a malformed form or body-read error. A session
    /// dropped on cancellation can instead be recovered by the reaper.
    ///
    /// # Errors
    ///
    /// Returns [`FileServiceError`] if cleanup cannot be recorded or started.
    async fn abort(self: Box<Self>) -> Result<(), FileServiceError>;
}

/// Coordinates the metadata registry and byte store for HTTP handlers.
///
/// An upload first inserts a pending registry entry without purpose or
/// expiration, streams file bytes to the byte store, then atomically records
/// validated metadata and the actual byte count or marks the entry failed.
/// Pending and failed entries are never returned by read methods. A delete
/// only makes the registry marker durable and may signal a reaper; it does not
/// wait for physical byte deletion.
/// Implementations must bound the time an upload can remain active so a
/// scheduled reaper can safely recover abandoned pending entries.
#[async_trait]
pub trait FileService: Send + Sync {
    /// Start an upload before all multipart fields are known. The filename
    /// comes from the file part's headers and is never used as a storage key.
    ///
    /// # Errors
    ///
    /// Returns [`FileServiceError`] for invalid input or a registry or store
    /// failure while creating the pending upload. If opening the store writer
    /// fails after the pending entry is inserted, mark that entry failed. The
    /// returned session owns its state so the handler can retain it across body
    /// callbacks.
    async fn begin_upload(
        &self,
        owner: &StateOwner,
        filename: String,
    ) -> Result<Box<dyn FileUploadSession + 'static>, FileServiceError>;

    /// Retrieve completed file metadata for this exact owner.
    ///
    /// # Errors
    ///
    /// Returns [`FileServiceError::NotFound`] for missing, hidden, or
    /// differently owned files, or another [`FileServiceError`] on registry
    /// failure.
    async fn retrieve(&self, owner: &StateOwner, id: &str) -> Result<FileInfo, FileServiceError>;

    /// List completed files for this exact owner.
    ///
    /// # Errors
    ///
    /// Returns [`FileServiceError`] when the query or registry operation fails.
    async fn list(&self, owner: &StateOwner, options: &FileList) -> Result<FilePage, FileServiceError>;

    /// Open a completed file as a stream. Errors can also occur while the
    /// returned stream is being consumed.
    ///
    /// # Errors
    ///
    /// Returns [`FileServiceError::NotFound`] for missing or hidden files, or
    /// another [`FileServiceError`] if metadata or byte storage is unavailable.
    async fn content(&self, owner: &StateOwner, id: &str) -> Result<FileDownload, FileServiceError>;

    /// Mark a completed file for asynchronous deletion.
    ///
    /// Returns only after the marker is durable. The caller may then return a
    /// successful delete response; a signal can wake the reaper, while a
    /// scheduled sweep remains sufficient to process the same durable marker.
    ///
    /// # Errors
    ///
    /// Returns [`FileServiceError::NotFound`] for missing, hidden, or
    /// differently owned files, or another [`FileServiceError`] on registry
    /// failure.
    async fn delete(&self, owner: &StateOwner, id: &str) -> Result<(), FileServiceError>;
}
