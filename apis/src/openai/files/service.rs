// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Handler-facing Files operations.

use async_trait::async_trait;
use praxis_ai_store::StateOwner;

use super::{
    FileInfo, FileList, FilePage, FileStream, FileUpload, FileUploadStream, registry::FileRegistryError,
    store::FileStoreError,
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
    /// Reading the incoming upload stream failed.
    #[error("file upload stream failed")]
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
            FileStoreError::Input(source) => Self::Stream(source),
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

/// Coordinates the metadata registry and byte store for HTTP handlers.
///
/// An upload first inserts a pending registry entry, streams the body to the
/// byte store, then atomically marks the entry ready with its actual byte count
/// or failed on error. Pending and failed entries are never returned by read
/// methods. A delete only makes the registry marker durable and may signal a
/// reaper; it does not wait for physical byte deletion.
/// Implementations must bound the time an upload can remain active so a
/// scheduled reaper can safely recover abandoned pending entries.
#[async_trait]
pub trait FileService: Send + Sync {
    /// Upload a file without buffering its entire body.
    ///
    /// # Errors
    ///
    /// Returns [`FileServiceError`] for invalid input, stream, registry, or store
    /// failures. The service must record a failed upload after an unsuccessful
    /// byte write. After an ambiguous metadata completion error, it must
    /// reconcile or retry the completion before attempting to mark failure.
    async fn upload(
        &self,
        owner: &StateOwner,
        file: FileUpload,
        content: FileUploadStream,
    ) -> Result<FileInfo, FileServiceError>;

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
