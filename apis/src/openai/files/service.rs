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
    /// The upload crossed its reserved bound at the per-file size limit.
    #[error("file exceeds the upload size limit")]
    UploadTooLarge,
    /// No capacity could be reserved, or the upload crossed a reservation
    /// smaller than the per-file size limit.
    #[error("project file storage limit exceeded")]
    QuotaExceeded,
    /// Reading the incoming HTTP upload body failed.
    #[error("file upload body failed")]
    Stream(#[source] Box<dyn std::error::Error + Send + Sync>),
    /// A metadata operation failed.
    #[error("file registry error")]
    Registry(#[source] FileRegistryError),
    /// A byte-store operation failed or rejected its writer bound. The upload
    /// session classifies expected bound violations using its reservation.
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
        Self::Store(error)
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
/// The session owns the pending file and its durable reservation until
/// completion or abandonment.
#[async_trait]
pub trait FileUploadSession: Send {
    /// Before passing a chunk to the byte-store writer, reject it if the
    /// cumulative byte count would exceed the reservation `R`. Return
    /// [`FileServiceError::QuotaExceeded`] when `R < max_file_size`, otherwise
    /// [`FileServiceError::UploadTooLarge`]. This classification depends on the
    /// reserved bound even if one chunk crosses both limits. Never forward a
    /// chunk that exceeds `R`; the writer also enforces `R` as a safety bound
    /// and applies backpressure. When `R < max_file_size`, earlier chunks may
    /// already have been written before a later chunk reaches the quota bound.
    ///
    /// # Errors
    ///
    /// Returns the bound-specific error described above or
    /// [`FileServiceError::Store`] for a byte-store failure. Body-read failures
    /// remain [`FileServiceError::Stream`] at the handler boundary. On failure,
    /// stop the writer and arrange byte deletion while retaining the registry
    /// reservation until cleanup is confirmed.
    async fn write_chunk(&mut self, chunk: Bytes) -> Result<(), FileServiceError>;

    /// Validate the complete form, finish the byte write, then commit the
    /// file with the writer's actual byte count `A <= R`. The registry replaces
    /// the reservation `R` with `A` without a new quota decision. The handler
    /// calls this after parsing all form fields. Batch uploads without an
    /// explicit expiration default to 30 days from the pending file's creation
    /// time.
    ///
    /// # Errors
    ///
    /// Returns [`FileServiceError`] for invalid metadata or a failed byte or
    /// registry operation. On failure, stop the writer where possible and
    /// arrange byte deletion. Retain the reservation until cleanup is
    /// confirmed. Ambiguous registry completion must be reconciled before
    /// marking the pending file failed; a successful retry returns the ready
    /// file when all completion fields match.
    async fn finish(self: Box<Self>, metadata: FileUploadMetadata) -> Result<FileInfo, FileServiceError>;

    /// Abandon an upload after a malformed form or body-read error. Stop the
    /// writer, arrange byte deletion, and retain the reservation until cleanup
    /// is confirmed. A session dropped on cancellation can instead be
    /// recovered by the reaper.
    ///
    /// # Errors
    ///
    /// Returns [`FileServiceError`] if cleanup cannot be recorded or started.
    async fn abort(self: Box<Self>) -> Result<(), FileServiceError>;
}

/// Coordinates the metadata registry and byte store for HTTP handlers.
///
/// An upload first inserts a pending registry entry and atomically reserves
/// `R = min(max_file_size, remaining_quota)` without purpose or expiration.
/// It opens a byte-store writer limited to `R`, checks each chunk against that
/// bound before writing, then atomically replaces `R` with the actual byte
/// count after the writer publishes the file. Pending and failed entries are
/// never returned by read methods. Failed uploads and hidden or expired files
/// keep their quota charge until their bytes are deleted. A delete only makes
/// the registry marker durable and may signal a reaper; it does not wait for
/// physical byte deletion.
/// Implementations must bound the time an upload can remain active so a
/// scheduled reaper can safely recover abandoned pending entries.
#[async_trait]
pub trait FileService: Send + Sync {
    /// Start an upload before all multipart fields are known. The filename
    /// comes from the file part's headers and is never used as a storage key.
    /// Pass the configured per-file and project limits to the registry and
    /// retain its durable reservation `R` in the returned session. Reject
    /// zero capacity before opening a writer; otherwise open it with `R` as
    /// its limit.
    ///
    /// # Errors
    ///
    /// Returns [`FileServiceError::QuotaExceeded`] if no capacity can be
    /// reserved, or another [`FileServiceError`] for invalid input or a
    /// registry or store failure. If opening the store writer fails after the
    /// pending entry is inserted, mark that entry failed and arrange byte
    /// deletion; retain its reservation until cleanup is confirmed. The
    /// returned session owns its state across body callbacks.
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

    /// Mark a completed file for asynchronous deletion. Its actual bytes stay
    /// charged to the project until physical deletion is confirmed.
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
