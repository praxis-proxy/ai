// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Record types exchanged with a [`FileCatalog`](super::FileCatalog).
//!
//! `purpose` is the opaque wire string: the catalog never interprets it, and
//! the API layer converts it to and from its own enums. Timestamps are Unix
//! seconds assigned by the database clock.

use std::time::Duration;

use crate::owner::StateOwner;

/// Direction of `(created_at, id)` ordering.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SortDirection {
    /// Oldest first.
    Asc,
    /// Newest first.
    Desc,
}

/// Metadata inserted before any upload bytes are written.
///
/// Purpose and expiration are recorded only when the upload completes. The
/// creation timestamp and the reservation are assigned by the catalog.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NewFile {
    /// New public file identifier, unique across all tenants, never reused.
    pub id: String,
    /// Trusted creating owner. The full owner is stored for audit.
    pub owner: StateOwner,
    /// Immutable, globally unique and opaque byte-store key, never reused.
    pub storage_key: String,
    /// Original client filename.
    pub filename: String,
    /// Most the upload may reserve: the service passes the smaller of the
    /// declared size hint and the API-wide per-file limit. The catalog
    /// reserves `min(max_bytes, remaining tenant quota)`, which is also the
    /// byte-store limit.
    pub max_bytes: u64,
}

/// Final metadata and actual byte count committed with a completed upload.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FileCompletion {
    /// Bytes persisted in the byte store. Must not exceed the reservation.
    pub bytes: u64,
    /// Validated purpose, as its wire value.
    pub purpose: String,
    /// Expiration measured from the file's creation time, if any. The catalog
    /// computes `expires_at = created_at + expires_after`.
    pub expires_after: Option<Duration>,
}

/// Metadata of a completed, client-visible file.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FileRecord {
    /// Public file identifier.
    pub id: String,
    /// Actual bytes persisted in the byte store.
    pub bytes: u64,
    /// Creation timestamp in seconds since the Unix epoch.
    pub created_at: i64,
    /// Expiration timestamp in seconds since the Unix epoch, if any.
    pub expires_at: Option<i64>,
    /// Original client filename.
    pub filename: String,
    /// Recorded purpose, as its wire value.
    pub purpose: String,
}

/// A ready file and the private location of its byte stream.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StoredFile {
    /// Client-visible metadata.
    pub record: FileRecord,
    /// Immutable byte-store key, never returned on the wire.
    pub storage_key: String,
}

/// List options for ready files.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FileListQuery {
    /// Optional purpose filter, compared as an opaque string.
    pub purpose: Option<String>,
    /// Page size.
    pub limit: u32,
    /// Direction of `(created_at, id)` ordering.
    pub order: SortDirection,
    /// File ID after which the page begins, resolved within the tenant.
    pub after: Option<String>,
}

/// One page of visible files.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FileRecordPage {
    /// Completed files in requested order.
    pub data: Vec<FileRecord>,
    /// Whether another page follows this one.
    pub has_more: bool,
}

/// Lifecycle states eligible for physical cleanup.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CleanupState {
    /// A failed upload that may have left bytes behind.
    Failed,
    /// A hidden file awaiting physical deletion.
    DeletionPending,
}

/// A leased cleanup task returned to a background reaper.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CleanupClaim {
    /// Public file identifier.
    pub id: String,
    /// Creating owner, kept for audit and logging.
    pub owner: StateOwner,
    /// Immutable byte-store key to delete.
    pub storage_key: String,
    /// Cleanup state, checked again by the conditional purge.
    pub state: CleanupState,
    /// Opaque fencing token issued with this claim; the purge must match it.
    pub claim_token: String,
}
