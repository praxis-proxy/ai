// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! S3 byte storage for Files uploads and downloads.

use async_trait::async_trait;
use object_store::aws::AmazonS3;

use super::{FileStore, FileStoreError, FileStoreWriter, object_store_adapter::ObjectStoreFileStore};
use crate::openai::files::FileStream;

/// A Files byte store backed by a caller-configured S3 bucket.
///
/// The caller configures credentials, endpoint, HTTP transport, and crypto on
/// [`AmazonS3`] before constructing this adapter. Object keys are the trusted,
/// globally unique storage keys supplied by the file registry. For recovery
/// after a process crash, configure an S3 lifecycle rule to abort incomplete
/// multipart uploads; an abandoned upload has no completed object for the
/// Files reaper to delete.
pub struct S3FileStore {
    /// Shared byte operations over the configured S3 client.
    adapter: ObjectStoreFileStore,
}

impl S3FileStore {
    /// Wrap a configured S3 bucket.
    #[must_use]
    pub fn new(store: AmazonS3) -> Self {
        Self {
            adapter: ObjectStoreFileStore::new(store),
        }
    }
}

#[async_trait]
impl FileStore for S3FileStore {
    async fn begin_write(
        &self,
        key: &str,
        reserved_bytes: u64,
    ) -> Result<Box<dyn FileStoreWriter + 'static>, FileStoreError> {
        self.adapter.begin_write(key, reserved_bytes).await
    }

    async fn read(&self, key: &str) -> Result<FileStream<FileStoreError>, FileStoreError> {
        self.adapter.read(key).await
    }

    async fn delete(&self, key: &str) -> Result<(), FileStoreError> {
        self.adapter.delete(key).await
    }
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, reason = "tests")]
mod tests {
    use bytes::Bytes;
    use futures::StreamExt as _;
    use object_store::memory::InMemory;

    use super::{FileStore as _, ObjectStoreFileStore, S3FileStore};

    #[tokio::test]
    async fn wrapper_forwards_file_store_operations() {
        let store = S3FileStore {
            adapter: ObjectStoreFileStore::new(InMemory::new()),
        };
        let mut upload = store.begin_write("files/key", 3).await.unwrap();
        upload.write_chunk(Bytes::from_static(b"abc")).await.unwrap();
        assert_eq!(upload.finish().await.unwrap(), 3);
        let chunks = store.read("files/key").await.unwrap().collect::<Vec<_>>().await;
        assert_eq!(
            chunks.into_iter().map(Result::unwrap).collect::<Vec<_>>().concat(),
            b"abc"
        );
        store.delete("files/key").await.unwrap();
        assert!(store.read("files/key").await.is_err());
    }
}
