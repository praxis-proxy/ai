// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Shared JSON request-body mutation.
//!
//! Several filters buffer the request body (`BodyMode::StreamBuffer`), mutate a parsed JSON value, and then
//! re-serialize it back into the body, each re-implementing the same serialize / replace / Content-Length fixup
//! dance. This module provides that shared machinery so individual filters only own their mutation logic:
//!
//! ```text
//! let mut value: serde_json::Value = serde_json::from_slice(raw)?;
//! value["model"] = "qwen-2.5-72b".into();
//! let mutation = replace_json_body(ctx, body, &value, "model_rewrite", "model")?;
//! ```
//!
//! [`serialize_json_body`] and [`SerializedJson::commit`] split the two halves for callers that must inspect the
//! serialized length before committing (for example, rejecting a rewritten body that exceeds a configured cap).
//!
//! Every commit emits one consistent `tracing` event carrying the filter name, the field changed, and the size
//! delta, and returns a [`BodyMutation`] report with the same data for callers that need it programmatically.
//!
//! Request-side only: core repairs upstream `Content-Length` framing for mutated request bodies, but has no
//! equivalent response-side repair, so response-body mutation needs different machinery.

use std::borrow::Cow;

use bytes::Bytes;
use praxis_filter::HttpFilterContext;
use serde_json::Value;
use tracing::debug;

/// Serialize `value` for a later body replacement.
///
/// Returns the serialized form without touching the buffered body, so callers can inspect the length (or
/// otherwise validate) before committing via [`SerializedJson::commit`].
///
/// # Errors
///
/// Returns [`JsonBodyError`] if `value` fails to serialize.
pub fn serialize_json_body(value: &Value) -> Result<SerializedJson, JsonBodyError> {
    let bytes = serde_json::to_vec(value).map_err(|e| JsonBodyError(format!("failed to serialize JSON body: {e}")))?;
    Ok(SerializedJson {
        bytes: Bytes::from(bytes),
    })
}

/// Serialize `value`, replace the buffered request `body`, fix up `Content-Length`, and emit the mutation
/// event.
///
/// Convenience wrapper around [`serialize_json_body`] and [`SerializedJson::commit`] for callers with no
/// pre-commit checks.
///
/// # Errors
///
/// Returns [`JsonBodyError`] if `value` fails to serialize.
pub fn replace_json_body(
    ctx: &mut HttpFilterContext<'_>,
    body: &mut Option<Bytes>,
    value: &Value,
    filter: &'static str,
    field: &'static str,
) -> Result<BodyMutation, JsonBodyError> {
    Ok(serialize_json_body(value)?.commit(ctx, body, filter, field))
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

    /// Replace the buffered request body with the serialized value, push the corrected `Content-Length`, and
    /// emit the mutation event. Returns a [`BodyMutation`] report of what changed.
    pub fn commit(
        self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        filter: &'static str,
        field: &'static str,
    ) -> BodyMutation {
        let mutation = BodyMutation {
            filter,
            field,
            original_len: body.as_ref().map_or(0, Bytes::len),
            new_len: self.bytes.len(),
        };

        *body = Some(self.bytes);
        ctx.extra_request_headers
            .push((Cow::Borrowed("content-length"), mutation.new_len().to_string()));

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

/// Error returned when a mutated JSON value cannot be re-serialized.
#[derive(Debug)]
pub struct JsonBodyError(String);

impl std::fmt::Display for JsonBodyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for JsonBodyError {}

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
    use http::Method;
    use serde_json::json;

    use super::*;
    use crate::test_utils::{make_filter_context, make_request};

    fn serialized_len(value: &Value) -> usize {
        serde_json::to_vec(value).unwrap().len()
    }

    #[test]
    fn replace_grows_body_and_fixes_content_length() {
        let req = make_request(Method::POST, "/v1/responses");
        let mut ctx = make_filter_context(&req);
        let original = json!({"model": "a"});
        let mut body = Some(Bytes::from(serde_json::to_vec(&original).unwrap()));
        let mutated = json!({"model": "qwen-2.5-72b-instruct"});

        let mutation = replace_json_body(&mut ctx, &mut body, &mutated, "model_rewrite", "model").unwrap();

        assert_eq!(mutation.filter(), "model_rewrite");
        assert_eq!(mutation.field(), "model");
        assert_eq!(mutation.original_len(), serialized_len(&original));
        assert_eq!(mutation.new_len(), serialized_len(&mutated));
        assert!(mutation.size_delta() > 0, "growth should report a positive delta");
        assert_eq!(
            body.as_ref().unwrap(),
            &Bytes::from(serde_json::to_vec(&mutated).unwrap()),
            "buffered body should hold the mutated value"
        );
        assert_eq!(ctx.extra_request_headers.len(), 1);
        let (name, value) = &ctx.extra_request_headers[0];
        assert_eq!(name.as_ref(), "content-length");
        assert_eq!(value, &mutation.new_len().to_string());
    }

    #[test]
    fn replace_shrinks_body_reports_negative_delta() {
        let req = make_request(Method::POST, "/v1/responses");
        let mut ctx = make_filter_context(&req);
        let original = json!({"model": "qwen-2.5-72b-instruct"});
        let mut body = Some(Bytes::from(serde_json::to_vec(&original).unwrap()));
        let mutated = json!({"model": "a"});

        let mutation = replace_json_body(&mut ctx, &mut body, &mutated, "model_rewrite", "model").unwrap();

        assert!(mutation.size_delta() < 0, "shrinkage should report a negative delta");
        assert_eq!(mutation.size_delta(), {
            let new = i64::try_from(serialized_len(&mutated)).unwrap();
            let old = i64::try_from(serialized_len(&original)).unwrap();
            new - old
        });
        assert_eq!(body.as_ref().unwrap().len(), serialized_len(&mutated));
    }

    #[test]
    fn replace_with_absent_body_reports_zero_original_len() {
        let req = make_request(Method::POST, "/v1/responses");
        let mut ctx = make_filter_context(&req);
        let mut body = None;
        let mutated = json!({"model": "qwen-2.5-72b"});

        let mutation = replace_json_body(&mut ctx, &mut body, &mutated, "model_rewrite", "model").unwrap();

        assert_eq!(mutation.original_len(), 0);
        assert_eq!(mutation.new_len(), serialized_len(&mutated));
        assert!(body.is_some(), "body should be populated after commit");
        assert_eq!(ctx.extra_request_headers.len(), 1);
    }

    #[test]
    fn replace_counts_multibyte_content_in_bytes() {
        let req = make_request(Method::POST, "/v1/responses");
        let mut ctx = make_filter_context(&req);
        let mut body = Some(Bytes::from_static(b"{}"));
        let mutated = json!({"input": "héllo ✓ 世界"});

        let mutation = replace_json_body(&mut ctx, &mut body, &mutated, "prompt_enrich", "input").unwrap();

        assert_eq!(mutation.new_len(), serialized_len(&mutated));
        assert!(
            mutation.new_len() > "héllo ✓ 世界".len(),
            "JSON escaping aside, length is measured in bytes"
        );
        assert_eq!(ctx.extra_request_headers[0].1, mutation.new_len().to_string());
    }

    #[test]
    fn two_step_serialize_then_commit_supports_pre_commit_checks() {
        let req = make_request(Method::POST, "/v1/responses");
        let mut ctx = make_filter_context(&req);
        let original = json!({"input": [{"type": "input_image", "image_url": "https://example.com/a.png"}]});
        let mut body = Some(Bytes::from(serde_json::to_vec(&original).unwrap()));
        let mutated = json!({"input": [{"type": "input_text", "text": "resolved"}]});

        let serialized = serialize_json_body(&mutated).unwrap();
        assert_eq!(serialized.len(), serialized_len(&mutated));
        assert!(!serialized.is_empty());
        assert_eq!(
            body.as_ref().unwrap().len(),
            serialized_len(&original),
            "body must be untouched before commit"
        );

        // Callers with a size cap (e.g. file_resolve / doc_extract) check here, before commit.
        let max_body_bytes = 1024;
        assert!(serialized.len() <= max_body_bytes);

        let mutation = serialized.commit(&mut ctx, &mut body, "openai_file_resolve", "input");
        assert_eq!(mutation.field(), "input");
        assert_eq!(mutation.original_len(), serialized_len(&original));
        assert_eq!(body.as_ref().unwrap().len(), serialized_len(&mutated));
        assert_eq!(ctx.extra_request_headers[0].1, serialized_len(&mutated).to_string());
    }

    #[test]
    fn commit_with_identical_value_still_repairs_content_length() {
        let req = make_request(Method::POST, "/v1/responses");
        let mut ctx = make_filter_context(&req);
        let value = json!({"model": "a"});
        let mut body = Some(Bytes::from(serde_json::to_vec(&value).unwrap()));

        let mutation = replace_json_body(&mut ctx, &mut body, &value, "model_rewrite", "model").unwrap();

        assert_eq!(mutation.size_delta(), 0);
        assert_eq!(ctx.extra_request_headers.len(), 1);
    }

    #[test]
    fn as_bytes_exposes_the_serialized_form() {
        let value = json!({"a": 1});
        let serialized = serialize_json_body(&value).unwrap();
        assert_eq!(serialized.as_bytes(), &Bytes::from(serde_json::to_vec(&value).unwrap()));
    }

    #[test]
    fn from_bytes_wraps_preserialized_json() {
        let req = make_request(Method::POST, "/v1/responses");
        let mut ctx = make_filter_context(&req);
        let mut body = Some(Bytes::from_static(b"{}"));
        let bytes = Bytes::from(serde_json::to_vec(&json!({"a": 1})).unwrap());

        let mutation =
            SerializedJson::from_bytes(bytes.clone()).commit(&mut ctx, &mut body, "openai_responses_proxy", "body");

        assert_eq!(mutation.new_len(), bytes.len());
        assert_eq!(body.as_ref().unwrap(), &bytes);
        assert_eq!(ctx.extra_request_headers[0].1, bytes.len().to_string());
    }
}
