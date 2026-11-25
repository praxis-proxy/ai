// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Default implementation of the Files service.

use std::sync::Arc;

use async_trait::async_trait;
use praxis_ai_store::{FileCatalog, StateOwner};

use super::{FileDownload, FileService, FileServiceError, FileUploadSession};
use crate::openai::files::{FileInfo, FileList, FilePage, FileStore};

/// Default implementation of the Files service.
pub struct DefaultFileService {
    /// Metadata catalog.
    catalog: Arc<dyn FileCatalog>,
    /// Byte store.
    #[expect(dead_code, reason = "used by upload, content and delete (#500, #501, delete PR)")]
    store: Arc<dyn FileStore>,
}

impl DefaultFileService {
    /// Create a service over a metadata catalog and a byte store.
    pub fn new(catalog: Arc<dyn FileCatalog>, store: Arc<dyn FileStore>) -> Self {
        Self { catalog, store }
    }
}

#[expect(
    clippy::unimplemented,
    reason = "upload, content and delete are not implemented yet (#500, #501, delete PR)"
)]
#[async_trait]
impl FileService for DefaultFileService {
    async fn begin_upload(
        &self,
        _owner: &StateOwner,
        _filename: String,
        _size_hint: Option<u64>,
    ) -> Result<Box<dyn FileUploadSession + 'static>, FileServiceError> {
        unimplemented!()
    }

    async fn retrieve(&self, owner: &StateOwner, id: &str) -> Result<FileInfo, FileServiceError> {
        let stored = self
            .catalog
            .get_ready(owner, id)
            .await?
            .ok_or(FileServiceError::NotFound)?;
        Ok(FileInfo::try_from(stored.record)?)
    }

    async fn list(&self, owner: &StateOwner, options: &FileList) -> Result<FilePage, FileServiceError> {
        let page = self.catalog.list_ready(owner, &options.into()).await?;
        Ok(FilePage::try_from(page)?)
    }

    async fn content(&self, _owner: &StateOwner, _id: &str) -> Result<FileDownload, FileServiceError> {
        unimplemented!()
    }

    async fn delete(&self, _owner: &StateOwner, _id: &str) -> Result<(), FileServiceError> {
        unimplemented!()
    }
}
