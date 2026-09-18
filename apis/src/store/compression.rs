// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Optional payload compression for the responses store.

use serde::Deserialize;

use super::StoreError;

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
    pub(crate) fn validate(&self) -> Result<(), String> {
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

    /// Effective zstd compression level.
    fn zstd_level(&self) -> i32 {
        self.level.unwrap_or(DEFAULT_ZSTD_LEVEL)
    }

    /// Encode a JSON value into its stored binary form.
    ///
    /// With `algorithm: none` this is the raw UTF-8 JSON bytes. With
    /// `algorithm: zstd` the JSON is compressed into a raw zstd frame.
    pub(crate) fn encode(&self, value: &serde_json::Value) -> Result<Vec<u8>, StoreError> {
        let json = serde_json::to_vec(value).map_err(|e| StoreError::Serialization(e.to_string()))?;
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
}

/// Decode a stored binary payload back into a JSON value.
///
/// The format is auto-detected: values beginning with the zstd
/// frame magic are decompressed; all others are parsed as raw JSON
/// bytes. Reads are therefore independent of the store's configured
/// compression, which is what keeps existing uncompressed records
/// readable after compression is enabled.
pub(crate) fn decode(stored: &[u8]) -> Result<serde_json::Value, StoreError> {
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
#[allow(clippy::unwrap_used, reason = "tests")]
mod tests {
    use serde_json::json;

    use super::*;

    fn zstd_config() -> StoreCompressionConfig {
        StoreCompressionConfig {
            algorithm: CompressionAlgorithm::Zstd,
            level: None,
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
        let encoded = cfg.encode(&value).unwrap();
        assert_eq!(encoded, serde_json::to_vec(&value).unwrap());
        assert!(!encoded.starts_with(&ZSTD_MAGIC));
    }

    #[test]
    fn zstd_encode_has_magic_and_roundtrips() {
        let cfg = zstd_config();
        let value = json!({"greeting": "hello world", "items": [1, 2, 3, 4, 5]});
        let encoded = cfg.encode(&value).unwrap();
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
            let encoded = cfg.encode(&value).unwrap();
            assert_eq!(decode(&encoded).unwrap(), value, "level {level}");
        }
    }

    #[test]
    fn zstd_shrinks_repetitive_payloads() {
        let cfg = zstd_config();
        let value = json!({"blob": "ababababab".repeat(1000)});
        let plain = serde_json::to_vec(&value).unwrap();
        let encoded = cfg.encode(&value).unwrap();
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
        assert_eq!(cfg.encode(&value).unwrap(), cfg.encode(&value).unwrap());
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
}
