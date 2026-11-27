// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! The [`FileCatalog`] contract: durable file metadata, lifecycle state,
//! tenant quota and cleanup claims for the Files API.
//!
//! SQL-free, like the rest of this crate. Backends live in
//! `praxis-ai-store-backends`.

mod catalog;
mod error;
mod types;

pub use catalog::FileCatalog;
pub use error::CatalogError;
pub use types::{
    CleanupClaim, CleanupState, FileCompletion, FileListQuery, FileRecord, FileRecordPage, NewFile, SortDirection,
    StoredFile,
};
