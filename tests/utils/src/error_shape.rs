// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Assertions on the wire shape of a proxy-generated error response.
//!
//! When a pipeline fails before or instead of reaching a backend, Praxis core
//! renders the error through whichever [`ErrorResponseFormatter`] a filter
//! installed, falling back to RFC 9457 problem details when none did. A client
//! that only knows its own provider's error envelope cannot parse that
//! fallback, so a chain serving a given protocol must install that protocol's
//! formatter.
//!
//! These helpers assert the envelope the client actually parses rather than
//! merely the presence of a key: `{"error": null}` would satisfy a presence
//! check while breaking every real SDK.
//!
//! [`ErrorResponseFormatter`]: https://docs.rs/praxis-filter

use crate::{parse_body, parse_status};

/// Asserts `raw` is an OpenAI-shaped error for an unreachable upstream.
///
/// Mirrors what the OpenAI SDKs parse: a top-level `error` object carrying a
/// string `message`, a `type` classifying the failure, and a `code` naming it.
///
/// # Panics
///
/// Panics when the status is not 5xx, the body is not JSON, the envelope does
/// not match, or the response fell back to RFC 9457 problem details.
pub fn assert_error_is_openai_shaped(raw: &str) {
    let (body, parsed) = upstream_failure_body(raw);

    let error = parsed
        .get("error")
        .and_then(serde_json::Value::as_object)
        .unwrap_or_else(|| panic!("error must be an object, got: {body}"));
    assert!(
        error.get("message").and_then(serde_json::Value::as_str).is_some(),
        "error.message must be a string, got: {body}"
    );
    assert_eq!(
        error.get("type").and_then(serde_json::Value::as_str),
        Some("server_error"),
        "error.type should classify an upstream failure, got: {body}"
    );
    assert_eq!(
        error.get("code").and_then(serde_json::Value::as_str),
        Some("upstream_connect_refused"),
        "error.code should name the upstream failure, got: {body}"
    );
    assert_no_problem_details(&body);
}

/// Asserts `raw` is an Anthropic-shaped error for an unreachable upstream.
///
/// Mirrors what the Anthropic SDKs parse: the `"type": "error"` discriminator,
/// a nested `error` object with its own `type` and a string `message`, and the
/// `request_id` the formatter echoes from the request head.
///
/// # Panics
///
/// Panics when the status is not 5xx, the body is not JSON, the envelope does
/// not match, or the response fell back to RFC 9457 problem details.
pub fn assert_error_is_anthropic_shaped(raw: &str) {
    let (body, parsed) = upstream_failure_body(raw);

    assert_eq!(
        parsed.get("type").and_then(serde_json::Value::as_str),
        Some("error"),
        "the top-level discriminator must be \"error\", got: {body}"
    );
    let error = parsed
        .get("error")
        .and_then(serde_json::Value::as_object)
        .unwrap_or_else(|| panic!("error must be an object, got: {body}"));
    assert_eq!(
        error.get("type").and_then(serde_json::Value::as_str),
        Some("api_error"),
        "error.type should classify an upstream failure, got: {body}"
    );
    assert!(
        error.get("message").and_then(serde_json::Value::as_str).is_some(),
        "error.message must be a string, got: {body}"
    );
    assert!(
        parsed
            .get("request_id")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|id| id.starts_with("req_")),
        "request_id must be a req_-prefixed string, got: {body}"
    );
    assert_no_problem_details(&body);
}

/// Asserts `raw` is a 5xx with a JSON body, returning the body and its parse.
///
/// Both envelopes share this preamble: the failure must come from the proxy
/// (5xx for a refused upstream) and must be JSON before any shape assertion is
/// meaningful.
fn upstream_failure_body(raw: &str) -> (String, serde_json::Value) {
    let status = parse_status(raw);
    assert!(
        (500..600).contains(&status),
        "an unreachable backend should fail the request, got {status}"
    );

    let body = parse_body(raw);
    let parsed: serde_json::Value =
        serde_json::from_str(&body).unwrap_or_else(|e| panic!("error body should be JSON: {e}\n{body}"));

    (body, parsed)
}

/// Asserts the error body is not the RFC 9457 fallback core renders when no
/// filter installed a formatter.
fn assert_no_problem_details(body: &str) {
    assert!(
        !body.contains("problem+json") && !body.contains("about:blank"),
        "must not fall back to RFC 9457 problem details, got: {body}"
    );
}
