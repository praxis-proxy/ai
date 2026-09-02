// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Metadata validation for conversation objects.

use std::fmt;

use serde_json::Value;

/// Maximum number of metadata keys.
const MAX_METADATA_KEYS: usize = 16;

/// Maximum length of a metadata key in bytes.
const MAX_KEY_BYTES: usize = 64;

/// Maximum length of a metadata string value in bytes.
const MAX_VALUE_BYTES: usize = 512;

/// Metadata validation failure.
#[derive(Debug)]
pub(crate) enum MetadataError {
    /// Value is not a JSON object (type mismatch).
    InvalidType(String),
    /// Constraint violation (key count, key/value length).
    ConstraintViolation(String),
}

impl fmt::Display for MetadataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidType(msg) | Self::ConstraintViolation(msg) => f.write_str(msg),
        }
    }
}

/// Validate conversation metadata.
///
/// Rules:
/// - Must be a JSON object (or null/absent → default `{}`)
/// - At most 16 keys
/// - Each key ≤ 64 bytes
/// - Each value must be a string ≤ 512 bytes
#[expect(clippy::too_many_lines, reason = "sequential validation pipeline")]
pub(crate) fn validate_metadata(metadata: &Value) -> Result<(), MetadataError> {
    let obj = match metadata {
        Value::Object(map) => map,
        Value::Null => return Ok(()),
        _ => return Err(MetadataError::InvalidType("metadata must be a JSON object".to_owned())),
    };

    if obj.len() > MAX_METADATA_KEYS {
        return Err(MetadataError::ConstraintViolation(format!(
            "metadata must have at most {MAX_METADATA_KEYS} keys, got {}",
            obj.len()
        )));
    }

    for (key, value) in obj {
        if key.len() > MAX_KEY_BYTES {
            return Err(MetadataError::ConstraintViolation(format!(
                "metadata key exceeds {MAX_KEY_BYTES} bytes: '{key}'"
            )));
        }
        match value {
            Value::String(s) => {
                if s.len() > MAX_VALUE_BYTES {
                    return Err(MetadataError::ConstraintViolation(format!(
                        "metadata value for key '{key}' exceeds {MAX_VALUE_BYTES} bytes"
                    )));
                }
            },
            _ => {
                return Err(MetadataError::InvalidType(format!(
                    "metadata value for key '{key}' must be a string"
                )));
            },
        }
    }

    Ok(())
}
