// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Temporary; replaced by `S3FileStore` once #2088 lands.

use async_trait::async_trait;

use super::{FileStore, FileStoreError, FileStoreWriter};
use crate::openai::files::FileStream;

/// A [`FileStore`] that fails every operation with a backend error.
#[expect(dead_code, reason = "wired by the openai_files filter (#499 step 4)")]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct UnavailableFileStore;

/// Build the error shared by every operation.
#[expect(dead_code, reason = "wired by the openai_files filter (#499 step 4)")]
fn unavailable() -> FileStoreError {
    FileStoreError::Backend("no file store configured".into())
}

#[async_trait]
impl FileStore for UnavailableFileStore {
    async fn begin_write(
        &self,
        _key: &str,
        _reserved_bytes: u64,
    ) -> Result<Box<dyn FileStoreWriter + 'static>, FileStoreError> {
        Err(unavailable())
    }

    async fn read(&self, _key: &str) -> Result<FileStream<FileStoreError>, FileStoreError> {
        Err(unavailable())
    }

    async fn delete(&self, _key: &str) -> Result<(), FileStoreError> {
        Err(unavailable())
    }
}
