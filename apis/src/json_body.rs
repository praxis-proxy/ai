// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Shared JSON request-body mutation and measurement.
//!
//! Several filters buffer the request body (`BodyMode::StreamBuffer`), mutate a parsed JSON value, and then
//! re-serialize it back into the body, each re-implementing the same serialize / replace dance. This module
//! provides that shared machinery so individual filters only own their mutation logic:
//!
//! ```text
//! let mut value: serde_json::Value = serde_json::from_slice(raw)?;
//! value["model"] = "qwen-2.5-72b".into();
//! let mutation = replace_json_body(body, &value, "model_rewrite", "model")?;
//! ```
//!
//! [`serialize_json_body`] and [`SerializedJson::commit`] split the two halves for callers that must inspect the
//! serialized length before committing (for example, rejecting a rewritten body that exceeds a configured cap).
//! Callers that only need the length and never the bytes use [`serialized_len`], which counts without buffering.
//!
//! Every commit emits one consistent `tracing` event carrying the filter name, the field changed, and the size
//! delta, and returns a [`BodyMutation`] report with the same data for callers that need it programmatically.
//!
//! Request-side only: core repairs upstream `Content-Length` framing for mutated request bodies via
//! `mutated_request_body_len`, so filters must not set `Content-Length` themselves.

use std::io;

use bytes::Bytes;
use praxis_filter::json_size::{self, JsonSizeError};
use serde::Serialize;
use serde_json::Value;
use tracing::debug;

/// Cap for JSON bodies these filters buffer, size, or accumulate. Praxis 0.8.0 raised its own
/// `MAX_JSON_BODY_BYTES` to the addressable maximum and leaves the cap to `body_limits`; praxis-ai keeps 64 MiB.
pub const MAX_JSON_BODY_BYTES: usize = praxis_core::config::DEFAULT_MAX_BUFFER_BYTES;

/// Serialize `value` for a later body replacement.
///
/// Returns the serialized form without touching the buffered body, so callers can inspect the length (or
/// otherwise validate) before committing via [`SerializedJson::commit`].
///
/// # Errors
///
/// Returns [`serde_json::Error`] if `value` fails to serialize.
pub fn serialize_json_body(value: &Value) -> Result<SerializedJson, serde_json::Error> {
    Ok(SerializedJson {
        bytes: Bytes::from(serde_json::to_vec(value)?),
    })
}

/// Byte length of a JSON-serialized `value`, without allocating the serialized form.
///
/// The bytes are counted as the serializer produces them and then discarded, so the
/// value content is never duplicated on the heap just to read a length.
///
/// ```
/// use praxis_ai_apis::json_body::serialized_len;
/// use serde_json::json;
///
/// let value = json!({"model": "a"});
/// assert_eq!(serialized_len(&value)?, r#"{"model":"a"}"#.len());
/// # Ok::<(), serde_json::Error>(())
/// ```
///
/// # Errors
///
/// Returns [`serde_json::Error`] if `value` fails to serialize.
pub fn serialized_len<T: Serialize + ?Sized>(value: &T) -> Result<usize, serde_json::Error> {
    json_size::serialized_len(value).map_err(json_size_error)
}

/// Preserve serde error contracts at the AI body-mutation boundary.
pub(crate) fn json_size_error(error: JsonSizeError) -> serde_json::Error {
    match error {
        JsonSizeError::Serialization(error) => error,
        error => serde_json::Error::io(io::Error::other(error)),
    }
}

/// Serialize a body in one pass without retaining output beyond the inclusive byte limit.
///
/// Use this when a rejection does not report the exact serialized size. No partial body is returned on
/// error, so nothing can be committed.
///
/// ```
/// use praxis_ai_apis::json_body::serialize_json_body_capped;
/// use praxis_filter::json_size::JsonSizeError;
///
/// let value = serde_json::json!({"a": "é"});
/// assert_eq!(serialize_json_body_capped(&value, 10)?.len(), 10);
/// assert!(matches!(serialize_json_body_capped(&value, 9), Err(JsonSizeError::LimitExceeded)));
/// # Ok::<(), JsonSizeError>(())
/// ```
///
/// # Errors
///
/// Returns [`JsonSizeError`] on limit exhaustion, length overflow, or serialization failure.
pub fn serialize_json_body_capped<T: Serialize + ?Sized>(
    value: &T,
    limit: usize,
) -> Result<SerializedJson, JsonSizeError> {
    json_size::to_vec_capped(value, limit).map(SerializedJson::from_bytes)
}

/// A body serialized under a limit whose rejection reports the exact serialized size.
#[derive(Debug)]
pub enum BoundedJsonBody {
    /// The serialized body fits within the limit.
    Fits(SerializedJson),
    /// The exact serialized size exceeds the limit; no output was allocated.
    TooLarge {
        /// Exact serialized byte length.
        actual: usize,
    },
}

/// Serialize a body under an inclusive limit, measuring first so rejections carry the exact size.
///
/// Counts the serialized bytes into a discard-only sink, rejects oversized values without allocating
/// output, and then serializes accepted values with the same cap. Use this when the rejection reports the
/// exact size; otherwise prefer [`serialize_json_body_capped`].
///
/// ```
/// use praxis_ai_apis::json_body::{BoundedJsonBody, serialize_json_body_within};
///
/// let value = serde_json::json!({"a": "é"});
/// assert!(matches!(serialize_json_body_within(&value, 10)?, BoundedJsonBody::Fits(b) if b.len() == 10));
/// assert!(matches!(serialize_json_body_within(&value, 9)?, BoundedJsonBody::TooLarge { actual: 10 }));
/// # Ok::<(), serde_json::Error>(())
/// ```
///
/// # Errors
///
/// Returns [`serde_json::Error`] if `value` fails to serialize or its length overflows `usize`.
pub fn serialize_json_body_within<T: Serialize + ?Sized>(
    value: &T,
    limit: usize,
) -> Result<BoundedJsonBody, serde_json::Error> {
    let actual = serialized_len(value)?;
    if actual > limit {
        return Ok(BoundedJsonBody::TooLarge { actual });
    }
    serialize_json_body_capped(value, limit)
        .map(BoundedJsonBody::Fits)
        .map_err(json_size_error)
}

/// Serialize `value`, replace the buffered request `body`, and emit the mutation event.
///
/// Convenience wrapper around [`serialize_json_body`] and [`SerializedJson::commit`] for callers with no
/// pre-commit checks.
///
/// # Errors
///
/// Returns [`serde_json::Error`] if `value` fails to serialize.
pub fn replace_json_body(
    body: &mut Option<Bytes>,
    value: &Value,
    filter: &'static str,
    field: &'static str,
) -> Result<BodyMutation, serde_json::Error> {
    Ok(serialize_json_body(value)?.commit(body, filter, field))
}

/// A serialized JSON body ready to be committed.
#[derive(Debug, Clone)]
pub struct SerializedJson {
    /// The serialized body bytes.
    bytes: Bytes,
}

impl SerializedJson {
    /// Wrap already-serialized JSON bytes (produced by a state-driven serializer rather than a
    /// `serde_json::Value`) for a later body replacement.
    #[must_use]
    pub fn from_bytes(bytes: impl Into<Bytes>) -> Self {
        Self { bytes: bytes.into() }
    }

    /// Length of the serialized body in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Whether the serialized body is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// The serialized bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &Bytes {
        &self.bytes
    }

    /// Replace the buffered request body with the serialized value and emit the mutation event.
    /// Returns a [`BodyMutation`] report of what changed.
    ///
    /// Does not set `Content-Length`; core handles upstream framing via `mutated_request_body_len`.
    pub fn commit(self, body: &mut Option<Bytes>, filter: &'static str, field: &'static str) -> BodyMutation {
        let mutation = BodyMutation {
            filter,
            field,
            original_len: body.as_ref().map_or(0, Bytes::len),
            new_len: self.bytes.len(),
        };

        *body = Some(self.bytes);

        debug!(
            filter = mutation.filter(),
            field = mutation.field(),
            original_len = mutation.original_len(),
            new_len = mutation.new_len(),
            size_delta = mutation.size_delta(),
            "request body mutated"
        );

        mutation
    }
}

/// Report of a committed JSON request-body mutation.
///
/// The same fields are emitted as a structured `tracing` event at commit time; this report lets callers act
/// on them (length caps, filter-specific logging, metadata promotion).
#[derive(Debug, Clone)]
pub struct BodyMutation {
    /// Name of the filter that performed the mutation.
    filter: &'static str,
    /// Top-level JSON field the mutation targeted.
    field: &'static str,
    /// Buffered body length before the mutation, in bytes.
    original_len: usize,
    /// Body length after the mutation, in bytes.
    new_len: usize,
}

impl BodyMutation {
    /// Name of the filter that performed the mutation.
    #[must_use]
    pub fn filter(&self) -> &'static str {
        self.filter
    }

    /// Top-level JSON field the mutation targeted.
    #[must_use]
    pub fn field(&self) -> &'static str {
        self.field
    }

    /// Buffered body length before the mutation, in bytes.
    #[must_use]
    pub fn original_len(&self) -> usize {
        self.original_len
    }

    /// Body length after the mutation, in bytes.
    #[must_use]
    pub fn new_len(&self) -> usize {
        self.new_len
    }

    /// `new_len - original_len`; negative when the body shrank.
    #[must_use]
    pub fn size_delta(&self) -> i64 {
        let new = i64::try_from(self.new_len).unwrap_or(i64::MAX);
        let old = i64::try_from(self.original_len).unwrap_or(i64::MAX);
        new - old
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests"
)]
mod tests {
    use std::collections::BTreeMap;

    use serde_json::json;

    use super::*;

    fn to_vec_len(value: &Value) -> usize {
        serde_json::to_vec(value).unwrap().len()
    }

    #[test]
    fn serialized_len_matches_allocating_serialization() {
        for value in [
            json!(null),
            json!(42),
            json!(1.5),
            json!(""),
            json!({}),
            json!([]),
            json!({"model": "qwen-2.5-72b-instruct", "stream": true}),
            json!({"input": [{"type": "input_text", "text": "hello"}, {"nested": {"deep": [1, 2, 3]}}]}),
        ] {
            assert_eq!(
                serialized_len(&value).unwrap(),
                to_vec_len(&value),
                "counted length should equal the allocated length for {value}"
            );
        }
    }

    #[test]
    fn serialized_len_counts_escapes_and_multibyte_in_bytes() {
        let value = json!({"text": "caf\u{00e9} \u{2615}\n\"quoted\"\ttabbed \u{65e5}\u{672c}\u{8a9e}"});

        assert_eq!(serialized_len(&value).unwrap(), to_vec_len(&value));
    }

    #[test]
    fn serialized_len_accepts_a_slice_of_values() {
        let items = vec![json!({"role": "user", "content": "hi"}), json!({"role": "assistant"})];

        assert_eq!(
            serialized_len(items.as_slice()).unwrap(),
            serde_json::to_vec(&items).unwrap().len(),
            "a slice should measure the same as the array it serializes to"
        );
    }

    #[test]
    fn serialized_len_of_empty_slice_is_the_empty_array() {
        let items: Vec<Value> = Vec::new();

        assert_eq!(serialized_len(items.as_slice()).unwrap(), "[]".len());
    }

    #[test]
    fn serialized_len_reports_serialization_failure() {
        // JSON object keys must be strings; a tuple key fails mid-serialization.
        let unserializable: BTreeMap<(u8, u8), u8> = BTreeMap::from([((1, 2), 3)]);

        assert!(
            serialized_len(&unserializable).is_err(),
            "a serialization failure must surface as an error rather than a length"
        );
    }

    #[test]
    fn replace_grows_body() {
        let original = json!({"model": "a"});
        let mut body = Some(Bytes::from(serde_json::to_vec(&original).unwrap()));
        let mutated = json!({"model": "qwen-2.5-72b-instruct"});

        let mutation = replace_json_body(&mut body, &mutated, "model_rewrite", "model").unwrap();

        assert_eq!(mutation.filter(), "model_rewrite");
        assert_eq!(mutation.field(), "model");
        assert_eq!(mutation.original_len(), to_vec_len(&original));
        assert_eq!(mutation.new_len(), to_vec_len(&mutated));
        assert!(mutation.size_delta() > 0, "growth should report a positive delta");
        assert_eq!(
            body.as_ref().unwrap(),
            &Bytes::from(serde_json::to_vec(&mutated).unwrap()),
            "buffered body should hold the mutated value"
        );
    }

    #[test]
    fn replace_shrinks_body_reports_negative_delta() {
        let original = json!({"model": "qwen-2.5-72b-instruct"});
        let mut body = Some(Bytes::from(serde_json::to_vec(&original).unwrap()));
        let mutated = json!({"model": "a"});

        let mutation = replace_json_body(&mut body, &mutated, "model_rewrite", "model").unwrap();

        assert!(mutation.size_delta() < 0, "shrinkage should report a negative delta");
        assert_eq!(mutation.size_delta(), {
            let new = i64::try_from(to_vec_len(&mutated)).unwrap();
            let old = i64::try_from(to_vec_len(&original)).unwrap();
            new - old
        });
        assert_eq!(body.as_ref().unwrap().len(), to_vec_len(&mutated));
    }

    #[test]
    fn replace_with_absent_body_reports_zero_original_len() {
        let mut body = None;
        let mutated = json!({"model": "qwen-2.5-72b"});

        let mutation = replace_json_body(&mut body, &mutated, "model_rewrite", "model").unwrap();

        assert_eq!(mutation.original_len(), 0);
        assert_eq!(mutation.new_len(), to_vec_len(&mutated));
        assert!(body.is_some(), "body should be populated after commit");
    }

    #[test]
    fn replace_counts_multibyte_content_in_bytes() {
        let mut body = Some(Bytes::from_static(b"{}"));
        let mutated = json!({"input": "caf\u{00e9} \u{2615} \u{65e5}\u{672c}\u{8a9e}"});

        let mutation = replace_json_body(&mut body, &mutated, "prompt_enrich", "input").unwrap();

        assert_eq!(mutation.new_len(), to_vec_len(&mutated));
        assert!(
            mutation.new_len() > "caf\u{00e9} \u{2615} \u{65e5}\u{672c}\u{8a9e}".len(),
            "JSON serialization of multibyte content should produce more bytes than the rust str len"
        );
    }

    #[test]
    fn two_step_serialize_then_commit_supports_pre_commit_checks() {
        let original = json!({"input": [{"type": "input_image", "image_url": "https://example.com/a.png"}]});
        let mut body = Some(Bytes::from(serde_json::to_vec(&original).unwrap()));
        let mutated = json!({"input": [{"type": "input_text", "text": "resolved"}]});

        let serialized = serialize_json_body(&mutated).unwrap();
        assert_eq!(serialized.len(), to_vec_len(&mutated));
        assert!(!serialized.is_empty());
        assert_eq!(
            body.as_ref().unwrap().len(),
            to_vec_len(&original),
            "body must be untouched before commit"
        );

        let max_body_bytes = 1024;
        assert!(serialized.len() <= max_body_bytes);

        let mutation = serialized.commit(&mut body, "openai_file_resolve", "input");
        assert_eq!(mutation.field(), "input");
        assert_eq!(mutation.original_len(), to_vec_len(&original));
        assert_eq!(body.as_ref().unwrap().len(), to_vec_len(&mutated));
    }

    #[test]
    fn commit_with_identical_value_reports_zero_delta() {
        let value = json!({"model": "a"});
        let mut body = Some(Bytes::from(serde_json::to_vec(&value).unwrap()));

        let mutation = replace_json_body(&mut body, &value, "model_rewrite", "model").unwrap();

        assert_eq!(mutation.size_delta(), 0);
    }

    #[test]
    fn as_bytes_exposes_the_serialized_form() {
        let value = json!({"a": 1});
        let serialized = serialize_json_body(&value).unwrap();
        assert_eq!(serialized.as_bytes(), &Bytes::from(serde_json::to_vec(&value).unwrap()));
    }

    #[test]
    fn from_bytes_wraps_preserialized_json() {
        let mut body = Some(Bytes::from_static(b"{}"));
        let bytes = Bytes::from(serde_json::to_vec(&json!({"a": 1})).unwrap());

        let mutation = SerializedJson::from_bytes(bytes.clone()).commit(&mut body, "openai_responses_proxy", "body");

        assert_eq!(mutation.new_len(), bytes.len());
        assert_eq!(body.as_ref().unwrap(), &bytes);
    }
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::panic, reason = "tests")]
mod capped_tests {
    use super::*;

    #[test]
    fn capped_body_preserves_bytes_and_commits_only_on_success() -> Result<(), Box<dyn std::error::Error>> {
        let value = serde_json::json!({"z": "é 東京 🦀\n\t\"\\\u{0000}", "a": [1, 2]});
        let expected = serde_json::to_vec(&value)?;
        let original = Bytes::from_static(b"original");
        let mut body = Some(original.clone());
        assert!(matches!(
            serialize_json_body_capped(&value, expected.len() - 1),
            Err(JsonSizeError::LimitExceeded)
        ));
        assert_eq!(body.as_ref(), Some(&original));
        serialize_json_body_capped(&value, expected.len())?.commit(&mut body, "test", "body");
        assert_eq!(body.as_deref(), Some(expected.as_slice()));
        Ok(())
    }

    #[test]
    fn capped_body_does_not_allocate_oversized_output() {
        let value = serde_json::json!({"input": "\u{0000}".repeat(1_048_576)});
        let allocations = allocation_counter::measure(|| {
            assert!(matches!(
                serialize_json_body_capped(&value, 1024),
                Err(JsonSizeError::LimitExceeded)
            ));
        });
        assert!(allocations.bytes_total < 16_384, "{allocations:?}");
    }

    #[test]
    fn bounded_body_reports_exact_size_and_accepts_exact_cap() -> Result<(), Box<dyn std::error::Error>> {
        let value = serde_json::json!({"z": "é 東京 🦀\n\t\"\\\u{0000}", "a": [1, 2]});
        let expected = serde_json::to_vec(&value)?;
        match serialize_json_body_within(&value, expected.len())? {
            BoundedJsonBody::Fits(serialized) => assert_eq!(serialized.as_bytes().as_ref(), expected.as_slice()),
            BoundedJsonBody::TooLarge { actual } => panic!("exact cap rejected {actual} bytes"),
        }
        assert!(matches!(
            serialize_json_body_within(&value, expected.len() - 1)?,
            BoundedJsonBody::TooLarge { actual } if actual == expected.len()
        ));
        Ok(())
    }

    #[test]
    fn bounded_body_rejects_without_allocating_output() {
        let value = serde_json::json!({"input": "\u{0000}".repeat(1_048_576)});
        let allocations = allocation_counter::measure(|| {
            assert!(matches!(
                serialize_json_body_within(&value, 1024),
                Ok(BoundedJsonBody::TooLarge { actual }) if actual == 6 * 1_048_576 + 12
            ));
        });
        assert_eq!(allocations.bytes_total, 0, "{allocations:?}");
    }

    #[test]
    fn bounded_body_preserves_serialization_errors() {
        struct Fails;
        impl Serialize for Fails {
            fn serialize<S: serde::Serializer>(&self, _: S) -> Result<S::Ok, S::Error> {
                Err(serde::ser::Error::custom("boom"))
            }
        }
        let error = serialize_json_body_within(&Fails, 1024).unwrap_err();
        assert_eq!(error.to_string(), "boom");
    }
}
