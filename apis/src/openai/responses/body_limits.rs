// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Shared body-size enforcement for Responses API rewriter filters.
//!
//! Raw transport size is governed exclusively by the pipeline's
//! `body_limits` (merged across filters and clamped to the transport
//! ceiling by praxis core). What `body_limits` does *not* bound is the
//! body a rewriter *produces*: MCP `mcp`→`function` expansion, base64
//! file inlining, and history rebuild can all grow the body beyond the
//! raw input. Those rewritten/resolved sizes are the genuinely
//! per-filter limits enforced here.
//!
//! This module centralizes:
//! - [`validate_size_limit`]: config validation that names the actual field (unlike praxis core's
//!   `validate_max_body_bytes`, which hardcodes the string `"max_body_bytes"`).
//! - [`reject_rewritten_body_too_large`]: a streaming-aware 413 built on the shared [`responses_error_rejection`]
//!   envelope.
//!
//! [`responses_error_rejection`]: super::error::responses_error_rejection

use praxis_filter::{FilterAction, FilterError, body::MAX_JSON_BODY_BYTES};

use super::error::responses_error_rejection;

/// Validate a filter-specific byte limit, naming the actual field.
///
/// Rejects `0` and values above the absolute [`MAX_JSON_BODY_BYTES`]
/// ceiling (64 MiB), producing an error that names `field` on `filter`.
///
/// # Errors
///
/// Returns [`FilterError`] when `value` is zero or exceeds the ceiling.
pub(crate) fn validate_size_limit(filter: &str, field: &str, value: usize) -> Result<(), FilterError> {
    if value == 0 {
        return Err(format!("{filter}: '{field}' must be greater than 0").into());
    }
    if value > MAX_JSON_BODY_BYTES {
        return Err(format!("{filter}: '{field}' ({value}) exceeds maximum ({MAX_JSON_BODY_BYTES})").into());
    }
    Ok(())
}

/// Build a streaming-aware 413 for a rewritten/resolved body that
/// exceeds the filter's configured limit.
///
/// `len` is the measured serialized length, `limit` the configured
/// maximum, and `streaming` selects the SSE error envelope over the
/// JSON one.
pub(crate) fn reject_rewritten_body_too_large(len: usize, limit: usize, streaming: bool) -> FilterAction {
    FilterAction::Reject(responses_error_rejection(
        413,
        "invalid_request_error",
        &format!("rewritten request body ({len} bytes) exceeds maximum ({limit} bytes)"),
        streaming,
    ))
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, reason = "tests")]
mod tests {
    use super::*;

    #[test]
    fn validate_size_limit_rejects_zero() {
        let err = validate_size_limit("f", "max_rewritten_body_bytes", 0).unwrap_err();
        assert!(
            err.to_string().contains("max_rewritten_body_bytes"),
            "error should name the actual field"
        );
        assert!(err.to_string().contains("greater than 0"), "error should explain zero");
    }

    #[test]
    fn validate_size_limit_rejects_above_ceiling() {
        let err = validate_size_limit("f", "max_resolved_bytes", MAX_JSON_BODY_BYTES + 1).unwrap_err();
        assert!(
            err.to_string().contains("max_resolved_bytes"),
            "error should name the actual field"
        );
        assert!(
            err.to_string().contains("exceeds maximum"),
            "error should explain ceiling"
        );
    }

    #[test]
    fn validate_size_limit_accepts_ceiling() {
        assert!(
            validate_size_limit("f", "max_rewritten_body_bytes", MAX_JSON_BODY_BYTES).is_ok(),
            "the ceiling itself should be accepted"
        );
        assert!(
            validate_size_limit("f", "max_rewritten_body_bytes", 1).is_ok(),
            "a small positive value should be accepted"
        );
    }

    #[test]
    fn reject_rewritten_body_non_streaming_is_json_413() {
        let FilterAction::Reject(r) = reject_rewritten_body_too_large(100, 50, false) else {
            panic!("expected a rejection");
        };
        assert_eq!(r.status, 413, "oversized rewritten body should be 413");
        let ct = r.headers.iter().find(|(k, _)| k == "content-type");
        assert_eq!(
            ct.map(|(_, v)| v.as_str()),
            Some("application/json"),
            "non-streaming should use application/json"
        );
    }

    #[test]
    fn reject_rewritten_body_streaming_is_sse_413() {
        let FilterAction::Reject(r) = reject_rewritten_body_too_large(100, 50, true) else {
            panic!("expected a rejection");
        };
        assert_eq!(r.status, 413, "oversized rewritten body should be 413");
        let ct = r.headers.iter().find(|(k, _)| k == "content-type");
        assert_eq!(
            ct.map(|(_, v)| v.as_str()),
            Some("text/event-stream"),
            "streaming should use text/event-stream"
        );
    }
}
