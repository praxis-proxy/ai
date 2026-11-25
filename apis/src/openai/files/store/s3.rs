// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! S3 byte storage for Files uploads and downloads.

mod crypto;

use std::sync::Arc;

use async_trait::async_trait;
use object_store::aws::AmazonS3Builder;

use self::crypto::OpensslCryptoProvider;
use super::{FileStore, FileStoreError, FileStoreWriter, object_store_adapter::ObjectStoreFileStore};
use crate::openai::files::FileStream;

/// A Files byte store backed by a caller-configured S3 bucket.
///
/// The caller configures the bucket, credentials, endpoint, and HTTP transport
/// on an [`AmazonS3Builder`]. This adapter configures OpenSSL signing. Object keys
/// are the trusted, globally unique storage keys supplied by the file registry.
///
/// The bucket must be unversioned and stay unversioned while the store uses it.
/// [`FileStore::delete`] deletes by key. Buckets with versioning enabled or
/// suspended can retain file bytes after a successful delete.
///
/// For recovery after a process crash, configure an S3 lifecycle rule to abort
/// incomplete multipart uploads. An abandoned upload has no completed object
/// for the Files reaper to delete.
pub struct S3FileStore {
    /// Shared byte operations over the configured S3 client.
    adapter: ObjectStoreFileStore,
}

impl S3FileStore {
    /// Build the S3 client with OpenSSL signing, replacing any crypto provider
    /// supplied on `builder`.
    ///
    /// # Errors
    /// Returns an error if the S3 client could not be built.
    pub fn new(builder: AmazonS3Builder) -> object_store::Result<Self> {
        let store = builder.with_crypto_provider(Arc::new(OpensslCryptoProvider)).build()?;
        Ok(Self {
            adapter: ObjectStoreFileStore::new(store),
        })
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
