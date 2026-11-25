// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! OpenAI Files API family.
//!
//! With `openai-files`, this module also exposes the service, metadata
//! registry, and byte-store contracts for a local implementation. The default
//! service is currently a placeholder.

#[cfg(feature = "openai-files")]
pub(crate) mod contracts;
#[cfg(feature = "openai-files")]
mod model;
#[cfg(feature = "openai-files")]
mod openapi;
#[cfg(feature = "openai-files")]
mod registry;
pub(crate) mod routes;
#[cfg(feature = "openai-files")]
mod service;
#[cfg(feature = "openai-files")]
mod store;

#[cfg(feature = "openai-files")]
pub use contracts::{FilePurpose, InvalidField, ListOrder as FileListOrder, UploadPurpose};
#[cfg(feature = "openai-files")]
pub use model::{
    FileCleanupCandidate, FileCleanupState, FileCompletion, FileExpiration, FileInfo, FileList, FilePage, FileState,
    FileStream, FileUploadLimits, FileUploadMetadata, PendingFile, StoredFile,
};
#[cfg(feature = "openai-files")]
pub use openapi::implementation_openapi_json;
#[cfg(feature = "openai-files")]
pub use registry::{FileRegistry, FileRegistryError};
#[cfg(feature = "openai-files")]
pub use service::{DefaultFileService, FileDownload, FileService, FileServiceError, FileUploadSession};
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
