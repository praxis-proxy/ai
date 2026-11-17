// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Placeholder implementation of the Files service.

use async_trait::async_trait;
use praxis_ai_store::StateOwner;

use super::{FileDownload, FileService, FileServiceError};
use crate::openai::files::{FileInfo, FileList, FilePage, FileUpload, FileUploadStream};

/// Default implementation of the Files service.
pub struct DefaultFileService;

#[expect(clippy::unimplemented, reason = "the Files service is a placeholder")]
#[async_trait]
impl FileService for DefaultFileService {
    async fn upload(
        &self,
        _owner: &StateOwner,
        _file: FileUpload,
        _content: FileUploadStream,
    ) -> Result<FileInfo, FileServiceError> {
        unimplemented!()
    }

    async fn retrieve(&self, _owner: &StateOwner, _id: &str) -> Result<FileInfo, FileServiceError> {
        unimplemented!()
    }

    async fn list(&self, _owner: &StateOwner, _options: &FileList) -> Result<FilePage, FileServiceError> {
        unimplemented!()
    }

    async fn content(&self, _owner: &StateOwner, _id: &str) -> Result<FileDownload, FileServiceError> {
        unimplemented!()
    }

    async fn delete(&self, _owner: &StateOwner, _id: &str) -> Result<(), FileServiceError> {
        unimplemented!()
    }
}
