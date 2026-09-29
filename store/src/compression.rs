// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Optional payload compression for the responses store.

use serde::Deserialize;

use crate::{ResponseRecord, StoreError};

/// zstd frame magic number (little-endian `0xFD2FB528`).
const ZSTD_MAGIC: [u8; 4] = [0x28, 0xB5, 0x2F, 0xFD];

/// Default zstd compression level when unspecified.
const DEFAULT_ZSTD_LEVEL: i32 = 3;

/// Upper bound on the bytes for decompressing a stored zstd frame.
const MAX_DECOMPRESSED_SIZE: u64 = 256 * 1024 * 1024;

// -----------------------------------------------------------------------------
// CompressionAlgorithm
// -----------------------------------------------------------------------------

/// Compression algorithm for stored payloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompressionAlgorithm {
    /// No compression (default). Payloads are stored as raw JSON bytes.
    #[default]
    None,
    /// zstd compression.
    Zstd,
}

// -----------------------------------------------------------------------------
// StoreCompressionConfig
// -----------------------------------------------------------------------------

/// Payload compression configuration shared by all store backends.
///
/// # YAML
///
/// ```yaml
/// compression:
///   algorithm: zstd
///   level: 3
/// ```
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StoreCompressionConfig {
    /// Compression algorithm. Defaults to `none`.
    #[serde(default)]
    pub algorithm: CompressionAlgorithm,

    /// zstd compression level. Only valid when `algorithm` is `zstd`.
    #[serde(default)]
    pub level: Option<i32>,
}

impl StoreCompressionConfig {
    /// Reject values that would fail or silently no-op at runtime.
    ///
    /// # Errors
    ///
    /// Returns a message describing the first invalid field.
    pub fn validate(&self) -> Result<(), String> {
        match self.algorithm {
            CompressionAlgorithm::None => {
                if self.level.is_some() {
                    return Err("compression.level is only valid when compression.algorithm is 'zstd'".to_owned());
                }
            },
            CompressionAlgorithm::Zstd => {
                if let Some(level) = self.level {
                    let range = zstd::compression_level_range();
                    if !range.contains(&level) {
                        return Err(format!(
                            "compression.level {level} out of range {}..={}",
                            range.start(),
                            range.end()
                        ));
                    }
                }
            },
        }
        Ok(())
    }

    /// Encode a response record's JSON columns into their stored binary form.
    ///
    /// With `algorithm: none` each field is the raw UTF-8 JSON bytes. With
    /// `algorithm: zstd` the JSON is compressed into a raw zstd frame.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::Serialization`] if a field cannot be serialized,
    /// exceeds the size limit, or zstd compression fails.
    pub async fn encode(&self, record: &ResponseRecord) -> Result<[Vec<u8>; 3], StoreError> {
        let [response_object, input, messages] = [&record.response_object, &record.input, &record.messages]
            .map(|value| serde_json::to_vec(value).map_err(|e| StoreError::Serialization(e.to_string())));
        let fields = [response_object?, input?, messages?];

        if self.algorithm == CompressionAlgorithm::None {
            return Ok(fields);
        }

        let config = self.clone();
        run_blocking(move || {
            let [response_object, input, messages] = fields.map(|json| config.encode_json(json));
            Ok([response_object?, input?, messages?])
        })
        .await
    }

    /// Apply the configured codec to an owned JSON buffer.
    fn encode_json(&self, json: Vec<u8>) -> Result<Vec<u8>, StoreError> {
        match self.algorithm {
            CompressionAlgorithm::None => Ok(json),
            CompressionAlgorithm::Zstd => {
                // Refuse to compress a payload larger than decode will accept, so a
                // successful write can always be read back.
                if json.len() as u64 > MAX_DECOMPRESSED_SIZE {
                    return Err(StoreError::Serialization(
                        "zstd compress: payload exceeds size limit".to_owned(),
                    ));
                }
                zstd::bulk::compress(&json, self.zstd_level())
                    .map_err(|e| StoreError::Serialization(format!("zstd compress: {e}")))
            },
        }
    }

    /// Effective zstd compression level.
    fn zstd_level(&self) -> i32 {
        self.level.unwrap_or(DEFAULT_ZSTD_LEVEL)
    }
}

/// Run owned store codec work outside the async executor.
///
/// # Errors
///
/// Returns [`StoreError::Unavailable`] if the blocking worker panics or is
/// cancelled, or the error the codec work itself returns.
pub async fn run_blocking<T: Send + 'static>(
    work: impl FnOnce() -> Result<T, StoreError> + Send + 'static,
) -> Result<T, StoreError> {
    tokio::task::spawn_blocking(work)
        .await
        .map_err(|e| StoreError::Unavailable(format!("store codec worker failed: {e}")))?
}

/// Decode a stored binary payload back into a JSON value.
///
/// The format is auto-detected: values beginning with the zstd
/// frame magic are decompressed; all others are parsed as raw JSON
/// bytes. Reads are therefore independent of the store's configured
/// compression, which is what keeps existing uncompressed records
/// readable after compression is enabled.
///
/// # Errors
///
/// Returns [`StoreError::Serialization`] if a zstd frame is corrupt, the
/// decompressed payload exceeds the size limit, or the bytes are not valid
/// JSON.
pub fn decode(stored: &[u8]) -> Result<serde_json::Value, StoreError> {
    if stored.starts_with(&ZSTD_MAGIC) {
        use std::io::Read as _;

        let decoder =
            zstd::Decoder::new(stored).map_err(|e| StoreError::Serialization(format!("zstd decompress: {e}")))?;
        // Read one byte past the cap so a payload sitting exactly at the limit is
        // accepted while anything larger is rejected.
        let mut json = Vec::new();
        decoder
            .take(MAX_DECOMPRESSED_SIZE + 1)
            .read_to_end(&mut json)
            .map_err(|e| StoreError::Serialization(format!("zstd decompress: {e}")))?;
        if json.len() as u64 > MAX_DECOMPRESSED_SIZE {
            return Err(StoreError::Serialization(
                "zstd decompress: decompressed payload exceeds size limit".to_owned(),
            ));
        }
        serde_json::from_slice(&json).map_err(|e| StoreError::Serialization(e.to_string()))
    } else {
        serde_json::from_slice(stored).map_err(|e| StoreError::Serialization(e.to_string()))
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::expect_used, clippy::unwrap_used, reason = "tests")]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::StateOwner;

    #[tokio::test(flavor = "current_thread")]
    async fn codec_work_runs_off_the_async_worker() {
        let async_thread = std::thread::current().id();
        let codec_thread = run_blocking(|| Ok(std::thread::current().id())).await.unwrap();
        assert_ne!(codec_thread, async_thread, "codec work must leave the async worker");

        let error = run_blocking(|| Err::<(), _>(StoreError::Serialization("invalid payload".to_owned())))
            .await
            .unwrap_err();
        assert!(matches!(error, StoreError::Serialization(message) if message == "invalid payload"));
    }

    #[test]
    fn response_encoding_offloads_zstd_and_preserves_all_fields() {
        use std::sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        };

        let worker_started = Arc::new(AtomicBool::new(false));
        let started = Arc::clone(&worker_started);
        let runtime = tokio::runtime::Builder::new_current_thread()
            .on_thread_start(move || started.store(true, Ordering::SeqCst))
            .build()
            .unwrap();
        let record = response_record();

        for config in [StoreCompressionConfig::default(), zstd_config()] {
            let fields = runtime.block_on(config.encode(&record)).unwrap();
            assert_eq!(
                worker_started.load(Ordering::SeqCst),
                config.algorithm == CompressionAlgorithm::Zstd,
                "only compressed writes need a blocking worker"
            );
            for (encoded, value) in fields
                .iter()
                .zip([&record.response_object, &record.input, &record.messages])
            {
                assert_eq!(decode(encoded).unwrap(), *value);
            }
        }
    }

    #[test]
    fn default_config_is_none() {
        let cfg = StoreCompressionConfig::default();
        assert_eq!(cfg.algorithm, CompressionAlgorithm::None);
        assert!(cfg.level.is_none());
    }

    #[test]
    fn none_encode_is_raw_json_bytes() {
        let cfg = StoreCompressionConfig::default();
        let value = json!({"a": 1, "b": [1, 2, 3]});
        let encoded = encode_value(&cfg, &value);
        assert_eq!(encoded, serde_json::to_vec(&value).unwrap());
        assert!(!encoded.starts_with(&ZSTD_MAGIC));
    }

    #[test]
    fn zstd_encode_has_magic_and_roundtrips() {
        let cfg = zstd_config();
        let value = json!({"greeting": "hello world", "items": [1, 2, 3, 4, 5]});
        let encoded = encode_value(&cfg, &value);
        assert!(
            encoded.starts_with(&ZSTD_MAGIC),
            "expected zstd frame magic prefix: {encoded:?}"
        );
        assert_eq!(decode(&encoded).unwrap(), value);
    }

    #[test]
    fn decode_reads_raw_json_without_magic() {
        // Backwards compatibility: an uncompressed row parses regardless
        // of the store's configured compression.
        let value = json!({"legacy": true});
        let plain = serde_json::to_vec(&value).unwrap();
        assert_eq!(decode(&plain).unwrap(), value);
    }

    #[test]
    fn zstd_roundtrips_at_every_boundary_level() {
        let range = zstd::compression_level_range();
        let value = json!({"payload": "x".repeat(4096)});
        for level in [*range.start(), 1, DEFAULT_ZSTD_LEVEL, *range.end()] {
            let cfg = StoreCompressionConfig {
                algorithm: CompressionAlgorithm::Zstd,
                level: Some(level),
            };
            let encoded = encode_value(&cfg, &value);
            assert_eq!(decode(&encoded).unwrap(), value, "level {level}");
        }
    }

    #[test]
    fn zstd_shrinks_repetitive_payloads() {
        let cfg = zstd_config();
        let value = json!({"blob": "ababababab".repeat(1000)});
        let plain = serde_json::to_vec(&value).unwrap();
        let encoded = encode_value(&cfg, &value);
        assert!(
            encoded.len() < plain.len(),
            "compressed ({}) should be smaller than plain ({})",
            encoded.len(),
            plain.len()
        );
    }

    #[test]
    fn encode_is_deterministic() {
        let cfg = zstd_config();
        let value = json!({"stable": [1, 2, 3], "nested": {"k": "v"}});
        assert_eq!(encode_value(&cfg, &value), encode_value(&cfg, &value));
    }

    #[test]
    fn decode_rejects_corrupt_compressed_payload() {
        let mut corrupt = ZSTD_MAGIC.to_vec();
        corrupt.extend_from_slice(b"not a real frame");
        assert!(decode(&corrupt).is_err());
    }

    #[test]
    fn validate_accepts_none_without_level() {
        assert!(StoreCompressionConfig::default().validate().is_ok());
    }

    #[test]
    fn validate_rejects_level_with_none_algorithm() {
        let cfg = StoreCompressionConfig {
            algorithm: CompressionAlgorithm::None,
            level: Some(3),
        };
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("level"), "{err}");
    }

    #[test]
    fn validate_accepts_zstd_with_valid_level() {
        let cfg = StoreCompressionConfig {
            algorithm: CompressionAlgorithm::Zstd,
            level: Some(DEFAULT_ZSTD_LEVEL),
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn validate_rejects_out_of_range_level() {
        let out_of_range = *zstd::compression_level_range().end() + 1;
        let cfg = StoreCompressionConfig {
            algorithm: CompressionAlgorithm::Zstd,
            level: Some(out_of_range),
        };
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("out of range"), "{err}");
    }

    #[test]
    fn deserialize_scalar_none_default() {
        let cfg: StoreCompressionConfig = serde_yaml::from_str("{}").unwrap();
        assert_eq!(cfg.algorithm, CompressionAlgorithm::None);
    }

    #[test]
    fn deserialize_zstd_with_level() {
        let cfg: StoreCompressionConfig = serde_yaml::from_str("algorithm: zstd\nlevel: 9\n").unwrap();
        assert_eq!(cfg.algorithm, CompressionAlgorithm::Zstd);
        assert_eq!(cfg.level, Some(9));
    }

    #[test]
    fn deserialize_denies_unknown_fields() {
        let result: Result<StoreCompressionConfig, _> = serde_yaml::from_str("algorithm: zstd\nbogus: true\n");
        assert!(result.is_err(), "unknown fields should be rejected");
    }

    // -------------------------------------------------------------------------
    // Test Utilities
    // -------------------------------------------------------------------------

    fn response_record() -> ResponseRecord {
        ResponseRecord {
            id: "resp_codec".to_owned(),
            owner: StateOwner::from_trusted_parts("tenant_codec", "issuer", "subject").expect("owner"),
            created_at: 1000,
            model: "test".to_owned(),
            response_object: json!({"output": "x".repeat(65_536)}),
            input: json!([{"role": "user", "content": "hello"}]),
            messages: json!([{"role": "assistant", "content": "world"}]),
        }
    }

    /// Encode a single JSON value through [`StoreCompressionConfig::encode`] by
    /// carrying it in a record's `response_object` column, returning the stored
    /// bytes for that column so codec properties can be asserted at value level.
    fn encode_value(config: &StoreCompressionConfig, value: &serde_json::Value) -> Vec<u8> {
        let mut record = response_record();
        record.response_object = value.clone();
        let runtime = tokio::runtime::Builder::new_current_thread().build().unwrap();
        let [response_object, _, _] = runtime.block_on(config.encode(&record)).unwrap();
        response_object
    }

    fn zstd_config() -> StoreCompressionConfig {
        StoreCompressionConfig {
            algorithm: CompressionAlgorithm::Zstd,
            level: None,
        }
    }
}
