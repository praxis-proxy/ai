// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Transport-neutral types shared by the Files service and its two backends.

use bytes::Bytes;
use futures::stream::BoxStream;
use praxis_ai_store::StateOwner;

use super::contracts::{FilePurpose, ListOrder as FileListOrder, UploadPurpose};

/// A sendable stream of file chunks with an error type chosen by its owner.
/// Chunks pass through without copying or buffering the entire file. The
/// stream owns its state so it can cross async trait-object boundaries.
pub type FileStream<E> = BoxStream<'static, Result<Bytes, E>>;

/// Incoming upload chunks. The handler boxes its body-read errors so the
/// service and byte store can accept streams from any HTTP transport.
pub type FileUploadStream = FileStream<Box<dyn std::error::Error + Send + Sync>>;

/// Expiration measured from file creation. The service rejects `seconds`
/// outside 3600..=2592000.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FileExpiration {
    /// Seconds after `created_at` when the file expires.
    pub seconds: u32,
}

/// Fields supplied with a new upload, apart from its byte stream.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FileUpload {
    /// Original client filename; never used as a storage key.
    pub filename: String,
    /// Intended purpose.
    pub purpose: UploadPurpose,
    /// Requested expiration policy. Batch uploads default to 30 days when absent.
    pub expires_after: Option<FileExpiration>,
}

/// Metadata of a completed, client-visible file.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FileInfo {
    /// Public file identifier.
    pub id: String,
    /// Actual bytes persisted in the file store.
    pub bytes: u64,
    /// Creation timestamp in seconds since the Unix epoch.
    pub created_at: i64,
    /// Expiration timestamp in seconds since the Unix epoch, if any.
    pub expires_at: Option<i64>,
    /// Original client filename.
    pub filename: String,
    /// Recorded purpose.
    pub purpose: FilePurpose,
}

/// List options supplied by a handler. The service rejects limits outside
/// 1..=10000.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FileList {
    /// Optional purpose filter, preserved as a string for forward compatibility.
    pub purpose: Option<String>,
    /// Page size in 1..=10000.
    pub limit: u32,
    /// Sort order by creation time and file ID.
    pub order: FileListOrder,
    /// File ID after which the page begins.
    pub after: Option<String>,
}

/// One page of visible files. The HTTP handler derives `first_id` and
/// `last_id` from `data` for the wire response.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FilePage {
    /// Completed files in requested order.
    pub data: Vec<FileInfo>,
    /// Whether another page follows this one.
    pub has_more: bool,
}

/// Durable registry states. Only `Ready` is visible to HTTP reads and lists.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FileState {
    /// Metadata is committed and the byte stream is still being written.
    Pending,
    /// The byte write and metadata completion both succeeded.
    Ready,
    /// The byte write or metadata completion failed; the reaper may clean bytes.
    Failed,
    /// A delete or expiration hid the file; the reaper still owes byte deletion.
    DeletionPending,
}

/// Metadata inserted before any upload bytes are written.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PendingFile {
    /// New public file identifier, unique across all owners.
    pub id: String,
    /// Trusted owner of the file.
    pub owner: StateOwner,
    /// Immutable, globally unique and opaque byte-store key.
    pub storage_key: String,
    /// Creation timestamp in seconds since the Unix epoch.
    pub created_at: i64,
    /// Expiration timestamp in seconds since the Unix epoch, if any.
    pub expires_at: Option<i64>,
    /// Original client filename.
    pub filename: String,
    /// Recorded purpose.
    pub purpose: FilePurpose,
}

/// A ready file and the private location of its byte stream.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StoredFile {
    /// Client-visible metadata.
    pub info: FileInfo,
    /// Immutable byte-store key, never returned on the wire.
    pub storage_key: String,
}

/// A durable cleanup task returned to a background reaper.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FileCleanupCandidate {
    /// Public file identifier.
    pub id: String,
    /// Trusted owner, used for a conditional registry purge.
    pub owner: StateOwner,
    /// Immutable byte-store key to delete.
    pub storage_key: String,
    /// Either [`FileState::Failed`] or [`FileState::DeletionPending`].
    pub state: FileState,
}
