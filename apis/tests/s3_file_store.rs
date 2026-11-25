// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Live `RustFS` contract tests for the public Files byte-store API.
//!
//! Run through `make test-s3-file-store`; live tests are ignored in ordinary
//! `cargo test` because they require a disposable `RustFS` bucket and AWS CLI.

#![allow(clippy::expect_used, clippy::panic, reason = "integration test assertions")]

#[cfg(test)]
mod tests {
    use std::process::Command;

    use async_trait::async_trait;
    use bytes::Bytes;
    use futures::StreamExt as _;
    use object_store::{
        ClientOptions,
        aws::AmazonS3Builder,
        client::{HttpClient, HttpConnector, HttpError, HttpRequest, HttpResponse, HttpService},
    };
    use praxis_ai_apis::openai::files::{FileStore as _, FileStoreError, S3FileStore};
    use serde_json::Value;

    const PART_BYTES: usize = 8 * 1024 * 1024;

    #[tokio::test]
    async fn signs_requests_through_the_supplied_connector() {
        #[derive(Debug)]
        struct TestConnector;

        impl HttpConnector for TestConnector {
            fn connect(&self, _options: &ClientOptions) -> object_store::Result<HttpClient> {
                Ok(HttpClient::new(Self))
            }
        }

        #[async_trait]
        impl HttpService for TestConnector {
            async fn call(&self, req: HttpRequest) -> Result<HttpResponse, HttpError> {
                assert_eq!(req.method(), http::Method::DELETE);
                let authorization = req
                    .headers()
                    .get(http::header::AUTHORIZATION)
                    .expect("signed request")
                    .to_str()
                    .expect("ASCII signature");
                assert!(authorization.starts_with("AWS4-HMAC-SHA256 Credential=test-access-key/"));
                Ok(HttpResponse::new(Bytes::new().into()))
            }
        }

        let builder = AmazonS3Builder::new()
            .with_bucket_name("test-bucket")
            .with_access_key_id("test-access-key")
            .with_secret_access_key("test-secret-key")
            .with_disable_bulk_delete(true)
            .with_http_connector(TestConnector);
        let store = S3FileStore::new(builder).expect("build S3 client with test connector");
        store.delete("files/key").await.expect("send signed delete");
    }

    fn listing_keys(response: &Value, field: &str) -> Vec<String> {
        let listing = response.as_object().expect("AWS CLI listing must be a JSON object");
        let Some(entries) = listing.get(field) else {
            return Vec::new();
        };
        entries
            .as_array()
            .unwrap_or_else(|| panic!("AWS CLI {field} must be an array"))
            .iter()
            .map(|entry| {
                entry
                    .get("Key")
                    .and_then(Value::as_str)
                    .unwrap_or_else(|| panic!("AWS CLI {field} entry must have a string Key"))
                    .to_owned()
            })
            .collect()
    }

    struct RustFs {
        endpoint: String,
        region: String,
        bucket: String,
        access_key: String,
        secret_key: String,
    }

    impl RustFs {
        fn from_env() -> Self {
            fn required(name: &str) -> String {
                std::env::var(name)
                    .ok()
                    .filter(|value| !value.is_empty())
                    .unwrap_or_else(|| panic!("{name} is required for RustFS E2E tests"))
            }

            Self {
                endpoint: required("RUSTFS_S3_ENDPOINT"),
                region: required("RUSTFS_S3_REGION"),
                bucket: required("RUSTFS_S3_BUCKET"),
                access_key: required("AWS_ACCESS_KEY_ID"),
                secret_key: required("AWS_SECRET_ACCESS_KEY"),
            }
        }

        fn store(&self) -> S3FileStore {
            self.store_with_credentials(&self.access_key, &self.secret_key)
        }

        fn store_with_credentials(&self, access_key: &str, secret_key: &str) -> S3FileStore {
            let builder = AmazonS3Builder::new()
                .with_endpoint(&self.endpoint)
                .with_region(&self.region)
                .with_bucket_name(&self.bucket)
                .with_access_key_id(access_key)
                .with_secret_access_key(secret_key)
                .with_virtual_hosted_style_request(false)
                .with_allow_http(true);
            S3FileStore::new(builder).expect("build RustFS S3 client")
        }

        /// AWS CLI is deliberately independent of the `object_store` S3 client.
        fn s3api(&self, operation: &str, args: &[&str]) -> Value {
            let output = Command::new("aws")
                .args([
                    "--endpoint-url",
                    &self.endpoint,
                    "--region",
                    &self.region,
                    "s3api",
                    operation,
                ])
                .args(args)
                .args(["--output", "json", "--no-cli-pager"])
                .env("AWS_ACCESS_KEY_ID", &self.access_key)
                .env("AWS_SECRET_ACCESS_KEY", &self.secret_key)
                .env("AWS_EC2_METADATA_DISABLED", "true")
                .env("AWS_MAX_ATTEMPTS", "1")
                .output()
                .expect("run AWS CLI against RustFS");
            assert!(
                output.status.success(),
                "aws s3api {operation} failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            serde_json::from_slice(&output.stdout).expect("AWS CLI JSON response")
        }

        fn objects(&self) -> Vec<String> {
            listing_keys(&self.s3api("list-objects-v2", &["--bucket", &self.bucket]), "Contents")
        }

        fn uploads(&self) -> Vec<String> {
            // List the whole isolated bucket: RustFS excludes some prefix-filter
            // listing edge cases from its default compatibility gate.
            listing_keys(
                &self.s3api("list-multipart-uploads", &["--bucket", &self.bucket]),
                "Uploads",
            )
        }

        fn assert_no_object_or_upload(&self, key: &str) {
            assert!(
                !self.objects().iter().any(|object| object == key),
                "completed object remains: {key}"
            );
            assert!(self.uploads().is_empty(), "multipart uploads remain in test bucket");
        }
    }

    #[test]
    fn listing_keys_accepts_empty_and_rejects_malformed_cli_responses() {
        use serde_json::json;

        for field in ["Contents", "Uploads"] {
            assert!(listing_keys(&json!({}), field).is_empty());
        }
        assert!(listing_keys(&json!({"Contents": []}), "Contents").is_empty());
        assert!(listing_keys(&json!({"Uploads": []}), "Uploads").is_empty());
        assert_eq!(
            listing_keys(&json!({"Contents": [{"Key": "files/a"}]}), "Contents"),
            ["files/a"]
        );
        assert_eq!(
            listing_keys(&json!({"Uploads": [{"Key": "files/b"}]}), "Uploads"),
            ["files/b"]
        );

        for (response, field) in [
            (json!(null), "Contents"),
            (json!({"Contents": null}), "Contents"),
            (json!({"Contents": {}}), "Contents"),
            (json!({"Contents": [{}]}), "Contents"),
            (json!({"Contents": [{"Key": 42}]}), "Contents"),
            (json!({"Uploads": null}), "Uploads"),
            (json!({"Uploads": {}}), "Uploads"),
            (json!({"Uploads": [{}]}), "Uploads"),
            (json!({"Uploads": [{"Key": 42}]}), "Uploads"),
        ] {
            assert!(
                std::panic::catch_unwind(|| listing_keys(&response, field)).is_err(),
                "{field} should reject {response}"
            );
        }
    }

    #[tokio::test]
    #[ignore = "requires the dedicated RustFS E2E lane"]
    #[expect(clippy::too_many_lines, reason = "checks both empty and small object lifecycles")]
    async fn empty_and_small_uploads_stream_back_and_delete_physically() {
        let config = RustFs::from_env();
        let store = config.store();

        for (key, chunk, expected) in [
            ("files/empty", None, b"".as_slice()),
            (
                "files/small",
                Some(Bytes::from_static(b"small file")),
                b"small file".as_slice(),
            ),
        ] {
            let mut upload = store.begin_write(key, 64).await.expect("begin upload");
            if let Some(chunk) = chunk {
                upload.write_chunk(chunk).await.expect("write chunk");
            }
            let written = upload.finish().await.expect("finish upload");
            assert_eq!(written, u64::try_from(expected.len()).expect("test length"));
            assert!(
                config.objects().iter().any(|object| object == key),
                "object was not committed: {key}"
            );

            let mut read = store.read(key).await.expect("open stream");
            let mut received = Vec::new();
            while let Some(chunk) = read.next().await {
                received.extend_from_slice(&chunk.expect("read chunk"));
            }
            assert_eq!(received, expected);

            store.delete(key).await.expect("delete object");
            assert!(
                !config.objects().iter().any(|object| object == key),
                "physical delete failed: {key}"
            );
            store.delete(key).await.expect("repeated delete");
            assert!(matches!(store.read(key).await, Err(FileStoreError::Backend(_))));
        }
        assert!(config.uploads().is_empty());
    }

    #[tokio::test]
    #[ignore = "requires the dedicated RustFS E2E lane"]
    #[expect(
        clippy::too_many_lines,
        reason = "checks both S3 upload branches and exact reservation"
    )]
    async fn part_size_and_exact_reservation_boundaries() {
        let config = RustFs::from_env();
        let store = config.store();
        let cases: [(&str, &[usize], usize, bool); 3] = [
            ("files/part-below", &[PART_BYTES - 1], PART_BYTES, false),
            ("files/part-exact", &[PART_BYTES], PART_BYTES + 1, true),
            ("files/reservation-exact", &[PART_BYTES, 1], PART_BYTES + 1, true),
        ];

        for (key, chunks, reservation, multipart) in cases {
            let expected_len: usize = chunks.iter().sum();
            let mut upload = store
                .begin_write(key, u64::try_from(reservation).expect("test reservation"))
                .await
                .expect("begin boundary upload");
            for &len in chunks {
                upload
                    .write_chunk(Bytes::from(vec![b'x'; len]))
                    .await
                    .expect("write boundary chunk");
            }

            let uploads = config.uploads();
            if multipart {
                assert_eq!(uploads, [key], "expected multipart upload for {key}");
            } else {
                assert!(uploads.is_empty(), "unexpected multipart upload for {key}");
            }
            assert!(
                !config.objects().iter().any(|object| object == key),
                "object was visible before finish: {key}"
            );

            let written = upload.finish().await.expect("finish boundary upload");
            let expected_bytes = u64::try_from(expected_len).expect("test length");
            assert_eq!(written, expected_bytes, "{key}");
            assert!(config.uploads().is_empty(), "multipart upload remains: {key}");
            let head = config.s3api("head-object", &["--bucket", &config.bucket, "--key", key]);
            assert_eq!(head.get("ContentLength").and_then(Value::as_u64), Some(expected_bytes));

            let mut read = store.read(key).await.expect("open boundary stream");
            let mut received = 0;
            while let Some(chunk) = read.next().await {
                let chunk = chunk.expect("read boundary chunk");
                assert!(chunk.iter().all(|byte| *byte == b'x'), "incorrect bytes in {key}");
                received += chunk.len();
            }
            assert_eq!(received, expected_len, "{key}");

            store.delete(key).await.expect("delete boundary object");
            config.assert_no_object_or_upload(key);
        }
    }

    #[tokio::test]
    #[ignore = "requires the dedicated RustFS E2E lane"]
    #[expect(
        clippy::too_many_lines,
        clippy::indexing_slicing,
        reason = "verifies every byte without buffering the read"
    )]
    async fn multipart_upload_preserves_order_and_reports_actual_byte_count() {
        let config = RustFs::from_env();
        let store = config.store();
        let key = "files/multipart";
        let segments = [(b'a', PART_BYTES - 3), (b'b', 7), (b'c', PART_BYTES + 5), (b'd', 4)];
        let expected_len: usize = segments.iter().map(|(_, len)| len).sum();
        // Give a larger allowance so the return value must be the counted bytes.
        let mut upload = store.begin_write(key, u64::MAX).await.expect("begin multipart upload");
        for &(byte, len) in &segments {
            upload
                .write_chunk(Bytes::from(vec![byte; len]))
                .await
                .expect("write multipart chunk");
        }
        assert_eq!(
            config.uploads(),
            [key],
            "expected an active multipart upload before finish"
        );
        assert!(
            !config.objects().iter().any(|object| object == key),
            "multipart object was visible before finish: {key}"
        );

        let written = upload.finish().await.expect("finish multipart upload");
        assert_eq!(written, u64::try_from(expected_len).expect("test length"));
        assert!(config.uploads().is_empty(), "multipart upload remains after finish");

        let mut read = store.read(key).await.expect("open multipart stream");
        let mut offset = 0;
        while let Some(chunk) = read.next().await {
            let chunk = chunk.expect("multipart read chunk");
            let mut remaining = chunk.as_ref();
            while !remaining.is_empty() {
                let (expected_byte, segment_end) = segments
                    .iter()
                    .scan(0, |end, (byte, len)| {
                        *end += len;
                        Some((*byte, *end))
                    })
                    .find(|(_, end)| offset < *end)
                    .expect("read more bytes than uploaded");
                let take = remaining.len().min(segment_end - offset);
                assert!(remaining[..take].iter().all(|byte| *byte == expected_byte));
                remaining = &remaining[take..];
                offset += take;
            }
        }
        assert_eq!(offset, expected_len);
        store.delete(key).await.expect("delete multipart object");
        config.assert_no_object_or_upload(key);
    }

    #[tokio::test]
    #[ignore = "requires the dedicated RustFS E2E lane"]
    async fn explicit_abort_removes_started_multipart_upload() {
        let config = RustFs::from_env();
        let store = config.store();
        let key = "files/aborted";
        let mut upload = store.begin_write(key, u64::MAX).await.expect("begin upload");
        upload
            .write_chunk(Bytes::from(vec![b'x'; PART_BYTES + 1]))
            .await
            .expect("write multipart part");

        assert_eq!(config.uploads(), [key], "expected an active multipart upload");
        upload.abort().await.expect("abort multipart upload");
        config.assert_no_object_or_upload(key);
    }

    #[tokio::test]
    #[ignore = "requires the dedicated RustFS E2E lane"]
    async fn size_limit_failure_aborts_started_multipart_upload() {
        let config = RustFs::from_env();
        let store = config.store();
        let key = "files/size-failure";
        let mut upload = store
            .begin_write(key, u64::try_from(PART_BYTES + 1).expect("test length"))
            .await
            .expect("begin upload");
        upload
            .write_chunk(Bytes::from(vec![b'x'; PART_BYTES + 1]))
            .await
            .expect("write multipart part");

        assert_eq!(config.uploads(), [key], "expected an active multipart upload");
        let result = upload.write_chunk(Bytes::from_static(b"!")).await;
        assert!(matches!(result, Err(FileStoreError::UploadTooLarge)));
        upload.abort().await.expect("abort oversized upload");
        config.assert_no_object_or_upload(key);
    }

    #[tokio::test]
    #[ignore = "requires the dedicated RustFS E2E lane"]
    async fn missing_object_and_backend_failure_are_backend_errors() {
        let config = RustFs::from_env();
        let store = config.store();
        let missing_key = "files/missing";
        config.assert_no_object_or_upload(missing_key);
        assert!(matches!(store.read(missing_key).await, Err(FileStoreError::Backend(_))));

        let existing_key = "files/backend-auth";
        let mut upload = store.begin_write(existing_key, 7).await.expect("begin seed upload");
        upload
            .write_chunk(Bytes::from_static(b"present"))
            .await
            .expect("write seed chunk");
        upload.finish().await.expect("seed object for authentication failure");
        assert!(config.objects().iter().any(|object| object == existing_key));

        let invalid_credentials = config.store_with_credentials("wrong-access-key", "wrong-secret-key");
        assert!(matches!(
            invalid_credentials.read(existing_key).await,
            Err(FileStoreError::Backend(_))
        ));
        let denied_key = "files/backend-denied";
        let denied = invalid_credentials
            .begin_write(denied_key, 1)
            .await
            .expect("begin denied upload");
        assert!(matches!(denied.finish().await, Err(FileStoreError::Backend(_))));
        config.assert_no_object_or_upload(denied_key);
        store.delete(existing_key).await.expect("delete seeded object");
        config.assert_no_object_or_upload(existing_key);
    }
}
