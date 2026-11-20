// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Handler-facing Files operations.

mod default;

use async_trait::async_trait;
use bytes::Bytes;
pub use default::DefaultFileService;
use praxis_ai_store::{CatalogError, StateOwner};

use super::{FileInfo, FileList, FilePage, FileStream, FileUploadMetadata, store::FileStoreError};

/// Failures exposed by the Files service to its handler.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum FileServiceError {
    /// No ready file exists in the authorized tenant scope.
    #[error("file not found")]
    NotFound,
    /// A request field is invalid.
    #[error("invalid file request: {0}")]
    InvalidInput(String),
    /// The upload crossed its reserved bound (the size hint or the per-file
    /// limit).
    #[error("file exceeds the upload size limit")]
    UploadTooLarge,
    /// The reservation does not fit in the tenant's remaining quota.
    #[error("project file storage limit exceeded")]
    QuotaExceeded,
    /// Reading the incoming HTTP upload body failed.
    #[error("file upload body failed")]
    Stream(#[source] Box<dyn std::error::Error + Send + Sync>),
    /// A metadata operation failed.
    #[error("file catalog error")]
    Catalog(#[source] CatalogError),
    /// A byte-store operation failed or rejected its writer bound. The upload
    /// session classifies expected bound violations before the writer sees them.
    #[error("file store error")]
    Store(#[source] FileStoreError),
}

impl From<CatalogError> for FileServiceError {
    fn from(error: CatalogError) -> Self {
        match error {
            CatalogError::QuotaExceeded => Self::QuotaExceeded,
            CatalogError::InvalidCursor => Self::InvalidInput("invalid `after` cursor".to_owned()),
            other => Self::Catalog(other),
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
    /// cumulative byte count would exceed the reservation `R`. Never forward a
    /// chunk that exceeds `R`; the writer also enforces `R` as a safety bound
    /// and applies backpressure. `R` is the declared size hint when the client
    /// sent one, otherwise the per-file maximum.
    ///
    /// # Errors
    ///
    /// Returns [`FileServiceError::UploadTooLarge`] when the upload crosses
    /// `R`, or [`FileServiceError::Store`] for a byte-store failure. Body-read
    /// failures remain [`FileServiceError::Stream`] at the handler boundary.
    /// On failure, stop the writer and arrange byte deletion while retaining
    /// the catalog reservation until cleanup is confirmed.
    async fn write_chunk(&mut self, chunk: Bytes) -> Result<(), FileServiceError>;

    /// Validate the complete form, finish the byte write, then commit the
    /// file with the writer's actual byte count `A <= R`. The catalog replaces
    /// the reservation `R` with `A` without a new quota decision. The handler
    /// calls this after parsing all form fields. Batch uploads without an
    /// explicit expiration default to 30 days from the pending file's creation
    /// time.
    ///
    /// # Errors
    ///
    /// Returns [`FileServiceError`] for invalid metadata or a failed byte or
    /// catalog operation. On failure, stop the writer where possible and
    /// arrange byte deletion. Retain the reservation until cleanup is
    /// confirmed. Ambiguous catalog completion must be reconciled before
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

/// Coordinates the metadata catalog and byte store for HTTP handlers.
///
/// Scoping is per tenant: any subject in a tenant can retrieve, list, download
/// and delete that tenant's files, and all subjects share one quota.
///
/// An upload first inserts a pending catalog entry and atomically reserves
/// `R` bytes of the tenant quota: the client's size hint, or the per-file
/// maximum when there is none. The catalog rejects the upload with
/// [`FileServiceError::QuotaExceeded`] when `R` does not fit, before any bytes
/// are written. The service opens a byte-store writer limited to `R`, checks
/// each chunk against that bound before writing, then atomically replaces `R`
/// with the actual byte count after the writer publishes the file. Pending and
/// failed entries are never returned by read methods. Failed uploads and hidden
/// or expired files keep their quota charge until their bytes are deleted. A
/// delete only makes the catalog marker durable and may signal a reaper; it
/// does not wait for physical byte deletion.
/// Implementations must bound the time an upload can remain active so a
/// scheduled reaper can safely recover abandoned pending entries.
#[async_trait]
pub trait FileService: Send + Sync {
    /// Start an upload before all multipart fields are known. The filename
    /// comes from the file part's headers and is never used as a storage key.
    /// `size_hint` is the declared `Content-Length`, if any. Reserve it, or the
    /// per-file maximum when absent, in the catalog and retain the reservation
    /// `R` in the returned session. Reject a hint above the per-file maximum
    /// with [`FileServiceError::UploadTooLarge`] before reserving; otherwise
    /// open the writer with `R` as its limit.
    ///
    /// # Errors
    ///
    /// Returns [`FileServiceError::QuotaExceeded`] if the reservation does not
    /// fit, or another [`FileServiceError`] for invalid input or a
    /// catalog or store failure. If opening the store writer fails after the
    /// pending entry is inserted, mark that entry failed and arrange byte
    /// deletion; retain its reservation until cleanup is confirmed. The
    /// returned session owns its state across body callbacks.
    async fn begin_upload(
        &self,
        owner: &StateOwner,
        filename: String,
        size_hint: Option<u64>,
    ) -> Result<Box<dyn FileUploadSession + 'static>, FileServiceError>;

    /// Retrieve completed file metadata for this tenant.
    ///
    /// # Errors
    ///
    /// Returns [`FileServiceError::NotFound`] for missing, hidden, or
    /// other tenants' files, or another [`FileServiceError`] on catalog
    /// failure.
    async fn retrieve(&self, owner: &StateOwner, id: &str) -> Result<FileInfo, FileServiceError>;

    /// List completed files for this tenant.
    ///
    /// # Errors
    ///
    /// Returns [`FileServiceError`] when the query or catalog operation fails.
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
    /// other tenants' files, or another [`FileServiceError`] on catalog
    /// failure.
    async fn delete(&self, owner: &StateOwner, id: &str) -> Result<(), FileServiceError>;
}
