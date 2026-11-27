// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! OpenAI Files API family.
//!
//! With `openai-files`, this module also exposes the service, metadata
//! catalog, and byte-store contracts for a local implementation. The default
//! service implements `retrieve` and `list`; upload, content and delete are
//! still placeholders. The `openai_files` filter serves list and retrieve
//! locally and passes the other operations through.

#[cfg(feature = "openai-files")]
mod config;
#[cfg(feature = "openai-files")]
pub(crate) mod contracts;
#[cfg(feature = "openai-files")]
mod filter;
#[cfg(feature = "openai-files")]
mod handlers;
#[cfg(feature = "openai-files")]
mod mapping;
#[cfg(feature = "openai-files")]
mod model;
#[cfg(feature = "openai-files")]
mod openapi;
pub(crate) mod routes;
#[cfg(feature = "openai-files")]
mod service;
#[cfg(feature = "openai-files")]
mod store;

#[cfg(feature = "openai-files")]
pub use contracts::{FilePurpose, InvalidField, ListOrder as FileListOrder, UploadPurpose};
#[cfg(feature = "openai-files")]
pub use filter::OpenaiFilesFilter;
#[cfg(feature = "openai-files")]
pub use model::{FileExpiration, FileInfo, FileList, FilePage, FileStream, FileUploadMetadata};
#[cfg(feature = "openai-files")]
pub use openapi::implementation_openapi_json;
#[cfg(feature = "openai-files")]
pub use praxis_ai_store::{CatalogError, FileCatalog};
#[cfg(feature = "openai-files")]
pub use service::{DefaultFileService, FileDownload, FileService, FileServiceError, FileUploadSession};
#[cfg(feature = "openai-files-s3")]
pub use store::S3FileStore;
#[cfg(feature = "openai-files")]
pub use store::{FileStore, FileStoreError, FileStoreWriter};

#[cfg(test)]
#[cfg(feature = "openai-files")]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::print_stdout,
    reason = "tests"
)]
mod tests;
