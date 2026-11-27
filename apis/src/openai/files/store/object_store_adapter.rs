// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Shared byte storage for multipart-capable object stores.

use std::{io, sync::Arc};

use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt as _;
use object_store::{GetOptions, ObjectStore, ObjectStoreExt as _, buffered::BufWriter, path::Path};
use tokio::io::AsyncWriteExt as _;

use super::{FileStore, FileStoreError, FileStoreWriter};
use crate::openai::files::FileStream;

/// Keep multipart parts above common minimums while bounding each upload's
/// buffered and in-flight bytes.
const DEFAULT_MULTIPART_PART_BYTES: usize = 8 * 1024 * 1024;

/// Minimum size for an input to keep its own zero-copy payload segment.
const MIN_ZERO_COPY_CHUNK_BYTES: usize = 8 * 1024;

/// Common Files byte operations over a configured object store.
pub(crate) struct ObjectStoreFileStore {
    /// The configured store behind a provider-independent interface.
    store: Arc<dyn ObjectStore>,
}

impl ObjectStoreFileStore {
    /// Wrap a configured object store.
    #[must_use]
    pub(crate) fn new(store: impl ObjectStore) -> Self {
        Self { store: Arc::new(store) }
    }

    /// Parse a registry key without changing the object name.
    fn location(key: &str) -> Result<Path, FileStoreError> {
        let location = Path::parse(key).map_err(backend)?;
        if location.is_root() || location.as_ref() != key {
            return Err(backend(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid file storage key",
            )));
        }
        Ok(location)
    }

    /// Remove an object whose final PUT or multipart completion may have
    /// succeeded despite shutdown returning an error.
    async fn clean_failed_commit(store: &dyn ObjectStore, location: &Path) {
        // A timed-out PUT or multipart completion may have committed
        // despite returning an error. The registry key is unique, so removing
        // it cannot affect another file. The reaper retries if this fails.
        if let Err(error) = store.delete(location).await
            && !matches!(error, object_store::Error::NotFound { .. })
        {
            tracing::warn!("failed to remove Files object after write error");
        }
    }
}

/// Preserve the storage library's error as the `Backend` source.
fn backend(error: impl std::error::Error + Send + Sync + 'static) -> FileStoreError {
    FileStoreError::Backend(Box::new(error))
}

/// One upload with its own reservation, byte count, and multipart state.
struct ObjectStoreFileWriter {
    /// Shared store for cleanup after an uncertain commit.
    store: Arc<dyn ObjectStore>,
    /// Unique registry key retained for cleanup after shutdown.
    location: Path,
    /// Buffers small files and streams larger files as multipart uploads.
    writer: BufWriter,
    /// Capacity reserved by the registry before opening this writer.
    reserved_bytes: u64,
    /// Number of bytes successfully passed to the object store writer.
    written: u64,
}

#[async_trait]
impl FileStoreWriter for ObjectStoreFileWriter {
    async fn write_chunk(&mut self, mut chunk: Bytes) -> Result<(), FileStoreError> {
        let chunk_bytes = u64::try_from(chunk.len()).map_err(|_error| FileStoreError::UploadTooLarge)?;
        let written = self
            .written
            .checked_add(chunk_bytes)
            .filter(|count| *count <= self.reserved_bytes)
            .ok_or(FileStoreError::UploadTooLarge)?;

        // BufWriter checks concurrency before each put. Splitting a large
        // input chunk keeps a single call from queuing many parts at once.
        while chunk.len() >= MIN_ZERO_COPY_CHUNK_BYTES {
            self.writer
                .put(chunk.split_to(chunk.len().min(DEFAULT_MULTIPART_PART_BYTES)))
                .await
                .map_err(backend)?;
        }
        if !chunk.is_empty() {
            // AsyncWrite coalesces small writes into blocks. Each put would
            // retain a separate Bytes entry until the payload is ready.
            self.writer.write_all(&chunk).await.map_err(backend)?;
        }
        self.written = written;
        Ok(())
    }

    async fn finish(mut self: Box<Self>) -> Result<u64, FileStoreError> {
        // shutdown publishes the object. BufWriter::abort panics after shutdown
        // starts, so an uncertain commit must be cleaned up by key instead.
        if let Err(error) = self.writer.shutdown().await {
            ObjectStoreFileStore::clean_failed_commit(self.store.as_ref(), &self.location).await;
            return Err(backend(error));
        }
        Ok(self.written)
    }

    async fn abort(mut self: Box<Self>) -> Result<(), FileStoreError> {
        self.writer.abort().await.map_err(backend)
    }
}

#[async_trait]
impl FileStore for ObjectStoreFileStore {
    async fn begin_write(
        &self,
        key: &str,
        reserved_bytes: u64,
    ) -> Result<Box<dyn FileStoreWriter + 'static>, FileStoreError> {
        let location = Self::location(key)?;
        let writer = BufWriter::with_capacity(Arc::clone(&self.store), location.clone(), DEFAULT_MULTIPART_PART_BYTES)
            .with_max_concurrency(1);
        Ok(Box::new(ObjectStoreFileWriter {
            store: Arc::clone(&self.store),
            location,
            writer,
            reserved_bytes,
            written: 0,
        }))
    }

    async fn read(&self, key: &str) -> Result<FileStream<FileStoreError>, FileStoreError> {
        let location = Self::location(key)?;
        let result = self
            .store
            .get_opts(&location, GetOptions::default())
            .await
            .map_err(backend)?;
        Ok(Box::pin(result.into_stream().map(|chunk| chunk.map_err(backend))))
    }

    async fn delete(&self, key: &str) -> Result<(), FileStoreError> {
        let location = Self::location(key)?;
        match self.store.delete(&location).await {
            Ok(()) | Err(object_store::Error::NotFound { .. }) => Ok(()),
            Err(error) => Err(backend(error)),
        }
    }
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, reason = "tests")]
mod tests;
