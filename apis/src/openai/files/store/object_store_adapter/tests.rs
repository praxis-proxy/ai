// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Object store adapter unit tests.

use std::{
    fmt, io,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use async_trait::async_trait;
use bytes::Bytes;
use futures::{
    StreamExt as _,
    stream::{self, BoxStream},
};
use object_store::{
    CopyOptions, GetOptions, GetResult, GetResultPayload, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
    PutMultipartOptions, PutOptions, PutPayload, PutResult, UploadPart, memory::InMemory, path::Path,
};

use super::{DEFAULT_MULTIPART_PART_BYTES, FileStore as _, FileStoreError, ObjectStoreFileStore};

/// Exercise the shared operations with a deterministic object store.
fn store() -> ObjectStoreFileStore {
    ObjectStoreFileStore::new(InMemory::new())
}

async fn assert_object_missing(store: &InMemory, location: &Path) {
    assert!(matches!(
        store.get_opts(location, GetOptions::default()).await,
        Err(object_store::Error::NotFound { .. })
    ));
}

/// The fault occurs only after the named storage operation has succeeded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Fault {
    PutAfterPublish,
    MultipartAfterPublish,
    ReadAfterFirstChunk,
}

fn injected_error(message: &'static str) -> object_store::Error {
    object_store::Error::Generic {
        store: "FaultStore",
        source: Box::new(io::Error::other(message)),
    }
}

/// Wrap the backing store so the adapter sees an uncertain commit or read fault.
#[derive(Debug)]
struct FaultStore {
    /// Shared backing store lets the test inspect the physical object.
    inner: InMemory,
    fault: Fault,
    /// Proves the injected failure was reached after the prior operation.
    triggered: Arc<AtomicBool>,
}

impl fmt::Display for FaultStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("FaultStore")
    }
}

#[async_trait]
impl ObjectStore for FaultStore {
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> object_store::Result<PutResult> {
        let result = self.inner.put_opts(location, payload, opts).await?;
        if self.fault == Fault::PutAfterPublish {
            self.triggered.store(true, Ordering::SeqCst);
            return Err(injected_error("injected lost PUT response"));
        }
        Ok(result)
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        let upload = self.inner.put_multipart_opts(location, opts).await?;
        if self.fault == Fault::MultipartAfterPublish {
            return Ok(Box::new(CompleteThenFailUpload {
                inner: upload,
                completed: Arc::clone(&self.triggered),
            }));
        }
        Ok(upload)
    }

    async fn get_opts(&self, location: &Path, options: GetOptions) -> object_store::Result<GetResult> {
        let mut result = self.inner.get_opts(location, options).await?;
        if self.fault == Fault::ReadAfterFirstChunk {
            result.payload = match result.payload {
                GetResultPayload::Stream(source) => {
                    let triggered = Arc::clone(&self.triggered);
                    let failure = stream::once(async move {
                        triggered.store(true, Ordering::SeqCst);
                        Err(injected_error("injected read stream failure"))
                    });
                    GetResultPayload::Stream(source.take(1).chain(failure).boxed())
                },
                #[allow(unreachable_patterns, reason = "object_store adds file payloads with its fs feature")]
                payload => payload,
            };
        }
        Ok(result)
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, object_store::Result<Path>>,
    ) -> BoxStream<'static, object_store::Result<Path>> {
        self.inner.delete_stream(locations)
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.inner.list(prefix)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> object_store::Result<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(&self, from: &Path, to: &Path, options: CopyOptions) -> object_store::Result<()> {
        self.inner.copy_opts(from, to, options).await
    }
}

/// Report a lost completion response after the backing store published the object.
#[derive(Debug)]
struct CompleteThenFailUpload {
    inner: Box<dyn MultipartUpload>,
    completed: Arc<AtomicBool>,
}

#[async_trait]
impl MultipartUpload for CompleteThenFailUpload {
    fn put_part(&mut self, data: PutPayload) -> UploadPart {
        self.inner.put_part(data)
    }

    async fn complete(&mut self) -> object_store::Result<PutResult> {
        self.inner.complete().await?;
        self.completed.store(true, Ordering::SeqCst);
        Err(injected_error("injected lost multipart completion response"))
    }

    async fn abort(&mut self) -> object_store::Result<()> {
        self.inner.abort().await
    }
}

#[test]
fn storage_keys_cannot_be_normalized_or_point_to_the_bucket_root() {
    for key in ["", "/files/key", "files/key/", "files//key", "files/../key"] {
        assert!(matches!(
            ObjectStoreFileStore::location(key),
            Err(FileStoreError::Backend(_))
        ));
    }
    assert_eq!(
        ObjectStoreFileStore::location("files/key").unwrap().as_ref(),
        "files/key"
    );
}

#[tokio::test]
async fn small_and_empty_objects_round_trip_and_delete_is_idempotent() {
    let store = store();
    let empty = store.begin_write("files/empty", 1).await.unwrap();
    assert_eq!(empty.finish().await.unwrap(), 0);
    let mut small = store.begin_write("files/small", 3).await.unwrap();
    small.write_chunk(Bytes::from_static(b"abc")).await.unwrap();
    assert!(matches!(
        store.read("files/small").await,
        Err(FileStoreError::Backend(_))
    ));
    assert_eq!(small.finish().await.unwrap(), 3);

    let empty = store.read("files/empty").await.unwrap().collect::<Vec<_>>().await;
    assert!(empty.into_iter().map(Result::unwrap).all(|chunk| chunk.is_empty()));
    let chunks = store.read("files/small").await.unwrap().collect::<Vec<_>>().await;
    assert_eq!(
        chunks.into_iter().map(Result::unwrap).collect::<Vec<_>>().concat(),
        b"abc"
    );

    store.delete("files/small").await.unwrap();
    store.delete("files/small").await.unwrap();
    assert!(matches!(
        store.read("files/small").await,
        Err(FileStoreError::Backend(_))
    ));
}

#[test]
fn fragmented_writes_bound_retained_memory() {
    const FRAGMENT_COUNT: i64 = 64 * 1024;
    let runtime = tokio::runtime::Builder::new_current_thread().build().unwrap();
    let store = store();

    // Measure while buffering the initial PUT and while filling a later part.
    for prefix_bytes in [0, DEFAULT_MULTIPART_PART_BYTES] {
        let mut upload = runtime
            .block_on(store.begin_write("files/fragmented", u64::MAX))
            .unwrap();
        if prefix_bytes > 0 {
            runtime
                .block_on(upload.write_chunk(Bytes::from(vec![0; prefix_bytes])))
                .unwrap();
        }

        let allocations = allocation_counter::measure(|| {
            runtime.block_on(async {
                for _ in 0..FRAGMENT_COUNT {
                    upload.write_chunk(Bytes::from_static(b"x")).await.unwrap();
                }
            });
        });
        assert!(
            allocations.bytes_current <= 2 * FRAGMENT_COUNT,
            "retained memory after {FRAGMENT_COUNT} one-byte writes with a {prefix_bytes}-byte prefix: {allocations:?}"
        );
        runtime.block_on(upload.abort()).unwrap();
    }
}

#[expect(clippy::too_many_lines, reason = "linear multipart upload assertions")]
#[tokio::test]
async fn multipart_upload_preserves_chunks_and_actual_count() {
    let store = store();
    let first = Bytes::from(vec![b'a'; DEFAULT_MULTIPART_PART_BYTES - 1]);
    let second = Bytes::from(vec![b'b'; DEFAULT_MULTIPART_PART_BYTES + 1]);
    let last = Bytes::from_static(b"end");
    let mut upload = store
        .begin_write(
            "files/large",
            u64::try_from(2 * DEFAULT_MULTIPART_PART_BYTES + 103).unwrap(),
        )
        .await
        .unwrap();
    upload.write_chunk(first).await.unwrap();
    upload.write_chunk(Bytes::from_static(b"ab")).await.unwrap();
    upload.write_chunk(second).await.unwrap();
    upload.write_chunk(last).await.unwrap();
    assert!(matches!(
        store.read("files/large").await,
        Err(FileStoreError::Backend(_))
    ));
    assert_eq!(
        upload.finish().await.unwrap(),
        u64::try_from(2 * DEFAULT_MULTIPART_PART_BYTES + 5).unwrap()
    );

    let chunks = store.read("files/large").await.unwrap().collect::<Vec<_>>().await;
    let data = chunks.into_iter().map(Result::unwrap).collect::<Vec<_>>().concat();
    assert_eq!(data.len(), 2 * DEFAULT_MULTIPART_PART_BYTES + 5);
    assert!(data[..DEFAULT_MULTIPART_PART_BYTES].iter().all(|byte| *byte == b'a'));
    assert!(
        data[DEFAULT_MULTIPART_PART_BYTES..2 * DEFAULT_MULTIPART_PART_BYTES + 2]
            .iter()
            .all(|byte| *byte == b'b')
    );
    assert_eq!(&data[2 * DEFAULT_MULTIPART_PART_BYTES + 2..], b"end");
}

#[tokio::test]
async fn oversized_upload_after_a_part_leaves_no_object() {
    let store = store();
    let mut upload = store
        .begin_write(
            "files/oversized",
            u64::try_from(DEFAULT_MULTIPART_PART_BYTES + 1).unwrap(),
        )
        .await
        .unwrap();
    upload
        .write_chunk(Bytes::from(vec![0; DEFAULT_MULTIPART_PART_BYTES]))
        .await
        .unwrap();
    upload.write_chunk(Bytes::from_static(b"x")).await.unwrap();
    assert!(matches!(
        upload.write_chunk(Bytes::from_static(b"y")).await,
        Err(FileStoreError::UploadTooLarge)
    ));
    upload.abort().await.unwrap();
    assert!(matches!(
        store.read("files/oversized").await,
        Err(FileStoreError::Backend(_))
    ));
}

#[tokio::test]
async fn explicit_abort_after_a_part_leaves_no_object() {
    let store = store();
    let mut upload = store.begin_write("files/aborted", u64::MAX).await.unwrap();
    upload
        .write_chunk(Bytes::from(vec![0; DEFAULT_MULTIPART_PART_BYTES]))
        .await
        .unwrap();
    upload.write_chunk(Bytes::from_static(b"x")).await.unwrap();
    upload.abort().await.unwrap();
    assert!(matches!(
        store.read("files/aborted").await,
        Err(FileStoreError::Backend(_))
    ));
}

#[tokio::test]
async fn dropping_an_unfinished_writer_does_not_publish_an_object() {
    let store = store();
    let mut upload = store.begin_write("files/dropped", 3).await.unwrap();
    upload.write_chunk(Bytes::from_static(b"abc")).await.unwrap();
    drop(upload);
    assert!(matches!(
        store.read("files/dropped").await,
        Err(FileStoreError::Backend(_))
    ));
}

#[tokio::test]
async fn dropping_an_unfinished_multipart_writer_does_not_publish_an_object() {
    let store = store();
    let key = "files/dropped-multipart";
    let mut upload = store.begin_write(key, u64::MAX).await.unwrap();
    upload
        .write_chunk(Bytes::from(vec![b'x'; DEFAULT_MULTIPART_PART_BYTES]))
        .await
        .unwrap();
    // The next write waits for the first part with max_concurrency(1).
    upload.write_chunk(Bytes::from_static(b"tail")).await.unwrap();
    assert!(matches!(store.read(key).await, Err(FileStoreError::Backend(_))));

    drop(upload);
    assert!(matches!(store.read(key).await, Err(FileStoreError::Backend(_))));
}

#[tokio::test]
async fn uncertain_commit_cleanup_removes_a_published_object() {
    let backing = InMemory::new();
    let published = Arc::new(AtomicBool::new(false));
    let store = ObjectStoreFileStore::new(FaultStore {
        inner: backing.clone(),
        fault: Fault::PutAfterPublish,
        triggered: Arc::clone(&published),
    });
    let key = "files/uncertain-commit";
    let mut upload = store.begin_write(key, 3).await.unwrap();
    upload.write_chunk(Bytes::from_static(b"abc")).await.unwrap();
    assert!(matches!(
        upload.finish().await,
        Err(FileStoreError::Backend(source))
            if source.to_string().contains("injected lost PUT response")
    ));
    assert!(published.load(Ordering::SeqCst), "fault must follow a successful PUT");
    assert!(matches!(
        backing
            .get_opts(&ObjectStoreFileStore::location(key).unwrap(), GetOptions::default())
            .await,
        Err(object_store::Error::NotFound { .. })
    ));
}

#[tokio::test]
async fn uncertain_multipart_completion_cleanup_removes_a_published_object() {
    let backing = InMemory::new();
    let completed = Arc::new(AtomicBool::new(false));
    let store = ObjectStoreFileStore::new(FaultStore {
        inner: backing.clone(),
        fault: Fault::MultipartAfterPublish,
        triggered: Arc::clone(&completed),
    });
    let key = "files/uncertain-multipart-completion";
    let location = ObjectStoreFileStore::location(key).unwrap();
    let mut upload = store.begin_write(key, u64::MAX).await.unwrap();
    upload
        .write_chunk(Bytes::from(vec![b'x'; DEFAULT_MULTIPART_PART_BYTES]))
        .await
        .unwrap();
    upload.write_chunk(Bytes::from_static(b"tail")).await.unwrap();
    assert_object_missing(&backing, &location).await;

    assert!(matches!(
        upload.finish().await,
        Err(FileStoreError::Backend(source))
            if source.to_string().contains("injected lost multipart completion response")
    ));
    assert!(
        completed.load(Ordering::SeqCst),
        "fault must follow multipart completion"
    );
    assert_object_missing(&backing, &location).await;
}

#[tokio::test]
async fn read_reports_a_stream_failure_after_the_first_chunk() {
    let triggered = Arc::new(AtomicBool::new(false));
    let store = ObjectStoreFileStore::new(FaultStore {
        inner: InMemory::new(),
        fault: Fault::ReadAfterFirstChunk,
        triggered: Arc::clone(&triggered),
    });
    let key = "files/read-stream-failure";
    let mut upload = store.begin_write(key, 3).await.unwrap();
    upload.write_chunk(Bytes::from_static(b"abc")).await.unwrap();
    assert_eq!(upload.finish().await.unwrap(), 3);

    let mut read = store.read(key).await.unwrap();
    assert!(!triggered.load(Ordering::SeqCst), "the read must open before the fault");
    assert_eq!(read.next().await.unwrap().unwrap(), Bytes::from_static(b"abc"));
    assert!(
        !triggered.load(Ordering::SeqCst),
        "the first chunk must arrive before the fault"
    );
    let error = read.next().await.unwrap().unwrap_err();
    assert!(triggered.load(Ordering::SeqCst));
    assert!(matches!(
        error,
        FileStoreError::Backend(source)
            if matches!(
                source.downcast_ref::<object_store::Error>(),
                Some(object_store::Error::Generic { store: "FaultStore", source })
                    if source.to_string() == "injected read stream failure"
            )
    ));
    assert!(read.next().await.is_none());
}
