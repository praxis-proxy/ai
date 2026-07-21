// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Content extraction logic for the `openai_doc_extract` filter.
//!
//! Handles data-URI and raw base64 decoding from `file_data`,
//! MIME type determination, and text-safe content extraction.
//! This filter does not perform network I/O — `file_url` parts
//! are skipped (download belongs in `openai_file_resolve`).

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use tracing::debug;

use super::config::{DocExtractConfig, OnUnsupported, is_text_safe_mime};
use crate::openai::responses::file_resolve::resolve::infer_mime_from_filename;

/// Errors that can occur during document extraction.
#[derive(Debug, Clone)]
pub(crate) enum ExtractError {
    /// Base64 decoding failed.
    DecodeFailed {
        /// Human-readable failure reason.
        detail: String,
    },
    /// Too many `input_file` parts in one request.
    TooManyReferences {
        /// Maximum allowed references.
        limit: usize,
    },
    /// A single file or the aggregate exceeds the size limit.
    TooLarge {
        /// Human-readable size context.
        detail: String,
        /// Maximum allowed bytes.
        limit: usize,
    },
    /// The file type is not supported and `on_unsupported` is `reject`.
    Unsupported {
        /// Detected MIME type.
        mime: String,
    },
}

impl std::fmt::Display for ExtractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DecodeFailed { detail } => write!(f, "decode failed: {detail}"),
            Self::TooManyReferences { limit } => write!(f, "too many input_file references (limit {limit})"),
            Self::TooLarge { detail, limit } => write!(f, "content too large ({detail}, limit {limit})"),
            Self::Unsupported { mime } => write!(f, "unsupported file type: {mime}"),
        }
    }
}

/// Request-scoped extraction budget.
pub(crate) struct ExtractionBudget {
    /// Maximum bytes for a single file.
    pub(crate) max_content_bytes: usize,
    /// Maximum `input_file` references per request.
    pub(crate) max_file_references: usize,
    /// Maximum aggregate extracted text bytes.
    pub(crate) max_total_text_bytes: usize,
    /// Policy for unsupported MIME types.
    pub(crate) on_unsupported: OnUnsupported,
    /// Number of references processed so far.
    pub(crate) references_seen: usize,
    /// Remaining aggregate text byte budget.
    pub(crate) remaining_total_text_bytes: usize,
}

impl ExtractionBudget {
    /// Create a new budget from filter configuration.
    pub(crate) fn new(config: &DocExtractConfig) -> Self {
        Self {
            max_content_bytes: config.max_content_bytes,
            max_file_references: config.max_file_references,
            max_total_text_bytes: config.max_total_text_bytes,
            on_unsupported: config.on_unsupported,
            references_seen: 0,
            remaining_total_text_bytes: config.max_total_text_bytes,
        }
    }

    /// Increment the reference counter and reject if over the limit.
    fn register_reference(&mut self) -> Result<(), ExtractError> {
        self.references_seen += 1;
        if self.references_seen > self.max_file_references {
            return Err(ExtractError::TooManyReferences {
                limit: self.max_file_references,
            });
        }
        Ok(())
    }

    /// Deduct `bytes` from the remaining aggregate text budget.
    fn consume_text_bytes(&mut self, bytes: usize) -> Result<(), ExtractError> {
        self.remaining_total_text_bytes =
            self.remaining_total_text_bytes
                .checked_sub(bytes)
                .ok_or_else(|| ExtractError::TooLarge {
                    detail: "aggregate extracted text".to_owned(),
                    limit: self.max_total_text_bytes,
                })?;
        Ok(())
    }

    /// Start fresh count and byte accounting for a mirrored state
    /// representation (e.g. `persisted_messages`) while retaining
    /// the shared configuration.
    pub(crate) fn begin_independent_accounting(&mut self) -> ExtractionAccounting {
        let saved = ExtractionAccounting {
            references_seen: self.references_seen,
            remaining_total_text_bytes: self.remaining_total_text_bytes,
        };
        self.references_seen = 0;
        self.remaining_total_text_bytes = self.max_total_text_bytes;
        saved
    }

    /// Restore accounting for the authoritative outbound
    /// representation.
    pub(crate) fn restore_accounting(&mut self, saved: &ExtractionAccounting) {
        self.references_seen = saved.references_seen;
        self.remaining_total_text_bytes = saved.remaining_total_text_bytes;
    }
}

/// Saved budget counters for independent accounting of a mirrored
/// state representation.
pub(crate) struct ExtractionAccounting {
    /// Saved reference count.
    references_seen: usize,
    /// Saved remaining text byte budget.
    remaining_total_text_bytes: usize,
}


/// Parsed data-URI components.
pub(crate) struct DataUri<'a> {
    /// MIME type from the data URI header.
    pub(crate) mime: &'a str,
    /// Base64-encoded payload after the comma.
    pub(crate) base64_payload: &'a str,
}

/// Parse a `data:mime/type;base64,payload` string.
pub(crate) fn parse_data_uri(value: &str) -> Option<DataUri<'_>> {
    let rest = value.strip_prefix("data:")?;
    let (header, payload) = rest.split_once(',')?;
    let mime = header.strip_suffix(";base64")?;
    if mime.is_empty() {
        return None;
    }
    Some(DataUri {
        mime,
        base64_payload: payload,
    })
}

/// Try to extract text content from an `input_file` content part.
///
/// Returns:
/// - `Ok(Some(content))` if the file was text-safe and extracted
/// - `Ok(None)` if the file was skipped (unresolved `file_id`, `file_url` without inline data, unsupported MIME with
///   `on_unsupported: continue`, etc.)
/// - `Err(e)` on extraction failure or policy rejection
pub(crate) fn extract_input_file(
    part: &serde_json::Value,
    budget: &mut ExtractionBudget,
) -> Result<Option<String>, ExtractError> {
    let file_data = part.get("file_data").and_then(serde_json::Value::as_str);
    let filename = part.get("filename").and_then(serde_json::Value::as_str);

    let Some(data) = file_data else {
        return Ok(None);
    };

    budget.register_reference()?;

    let mime = determine_mime(data, filename);

    if !is_text_safe_mime(&mime) {
        return skip_or_reject_unsupported(&mime, filename, budget.on_unsupported);
    }

    let content_bytes = decode_file_data(data, budget.max_content_bytes)?;

    let text = validate_and_decode_utf8(content_bytes, &mime, filename, budget)?;
    let Some(text) = text else {
        return Ok(None);
    };

    let final_text = build_input_text(text, filename);

    if final_text.len() > budget.max_content_bytes {
        return Err(ExtractError::TooLarge {
            detail: format!("generated input_text {} bytes", final_text.len()),
            limit: budget.max_content_bytes,
        });
    }

    budget.consume_text_bytes(final_text.len())?;

    Ok(Some(final_text))
}

/// Skip or reject an unsupported MIME type based on the configured policy.
fn skip_or_reject_unsupported(
    mime: &str,
    filename: Option<&str>,
    on_unsupported: OnUnsupported,
) -> Result<Option<String>, ExtractError> {
    match on_unsupported {
        OnUnsupported::Continue => {
            debug!(mime = %mime, filename = ?filename, "skipping unsupported file type");
            Ok(None)
        },
        OnUnsupported::Reject => Err(ExtractError::Unsupported { mime: mime.to_owned() }),
    }
}

/// Decode bytes as UTF-8 and return the text.
fn validate_and_decode_utf8(
    content_bytes: Vec<u8>,
    mime: &str,
    filename: Option<&str>,
    budget: &ExtractionBudget,
) -> Result<Option<String>, ExtractError> {
    let Ok(text) = String::from_utf8(content_bytes) else {
        debug!(mime = %mime, filename = ?filename, "file content is not valid UTF-8");
        return match budget.on_unsupported {
            OnUnsupported::Continue => Ok(None),
            OnUnsupported::Reject => Err(ExtractError::Unsupported {
                mime: format!("{mime} (invalid UTF-8)"),
            }),
        };
    };

    Ok(Some(text))
}

/// Exact decoded length from a base64-encoded string, accounting
/// for trailing `=` padding.
fn base64_decoded_len(encoded: &[u8]) -> usize {
    let len = encoded.len();
    if len == 0 {
        return 0;
    }
    let padding = encoded.iter().rev().take_while(|&&b| b == b'=').count();
    let significant = len - padding;
    significant / 4 * 3
        + match significant % 4 {
            2 => 1,
            3 => 2,
            _ => 0,
        }
}

/// Reject early if the base64 payload decodes to more than the limit.
fn check_encoded_size(encoded: &str, max_content_bytes: usize) -> Result<(), ExtractError> {
    let decoded_len = base64_decoded_len(encoded.as_bytes());
    if decoded_len > max_content_bytes {
        return Err(ExtractError::TooLarge {
            detail: format!("base64 payload decodes to {decoded_len} bytes"),
            limit: max_content_bytes,
        });
    }
    Ok(())
}

/// Determine the MIME type from `file_data` without decoding the payload.
fn determine_mime(data: &str, filename: Option<&str>) -> String {
    if let Some(data_uri) = parse_data_uri(data) {
        data_uri.mime.to_owned()
    } else {
        infer_mime_from_filename(filename)
            .unwrap_or("application/octet-stream")
            .to_owned()
    }
}

/// Decode the base64 payload from `file_data` (data-URI or raw).
fn decode_file_data(data: &str, max_content_bytes: usize) -> Result<Vec<u8>, ExtractError> {
    if let Some(data_uri) = parse_data_uri(data) {
        check_encoded_size(data_uri.base64_payload, max_content_bytes)?;
        BASE64
            .decode(data_uri.base64_payload)
            .map_err(|e| ExtractError::DecodeFailed {
                detail: format!("invalid base64 in data URI: {e}"),
            })
    } else {
        check_encoded_size(data, max_content_bytes)?;
        BASE64.decode(data).map_err(|e| ExtractError::DecodeFailed {
            detail: format!("invalid base64 in file_data: {e}"),
        })
    }
}

/// Build the `input_text` content, optionally prefixed with the source
/// filename for model context.
fn build_input_text(text: String, filename: Option<&str>) -> String {
    match filename {
        Some(name) if !name.is_empty() => format!("[Source: {name}]\n{text}"),
        _ => text,
    }
}
