// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Unit tests for the `anthropic_validate` filter.

use super::*;

// -----------------------------------------------------------------------------
// Validation Logic
// -----------------------------------------------------------------------------

#[test]
fn valid_request_passes() {
    let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"role":"user","content":"Hi"}]}"#;
    assert!(validate_request(body).is_none(), "valid request should pass");
}

#[test]
fn whitespace_prefixed_object_passes() {
    assert!(
        validate_request(b" \t\r\n {\"nested\":{\"items\":[1,true,null]}}").is_none(),
        "JSON whitespace before an object should be ignored"
    );
}

#[test]
fn backend_owned_semantics_pass() {
    let body = br#"{"model":"","max_tokens":0,"messages":[]}"#;
    assert!(
        validate_request(body).is_none(),
        "backend-owned Anthropic semantics should be deferred"
    );
}

#[test]
fn missing_backend_owned_fields_pass() {
    let body = br#"{"metadata":{"tenant":"blue"}}"#;
    assert!(
        validate_request(body).is_none(),
        "required Anthropic fields should be validated by the backend"
    );
}

#[test]
fn invalid_json_rejected() {
    let body = b"not json {{{";
    let rejection = validate_request(body).expect("invalid JSON should be rejected");
    let parsed: serde_json::Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();

    assert_eq!(parsed["type"], "error");
    assert_eq!(parsed["error"]["type"], "invalid_request_error");
    assert!(parsed.get("request_id").is_some());
    assert!(parsed["request_id"].is_null());
}

#[test]
fn malformed_nested_json_and_trailing_data_are_rejected() {
    for body in [br#"{"outer":[{"broken":}]}"#.as_slice(), br#"{} []"#.as_slice()] {
        assert!(validate_request(body).is_some(), "malformed JSON should be rejected");
    }
}

#[test]
fn deeply_nested_json_is_fully_validated() {
    let mut body = String::from(r#"{"deep":"#);
    for depth in 0..256 {
        if depth % 2 == 0 {
            body.push_str(r#"{"deep":"#);
        } else {
            body.push('[');
        }
    }
    body.push_str("null");
    for depth in (0..256).rev() {
        body.push(if depth % 2 == 0 { '}' } else { ']' });
    }
    body.push('}');

    assert!(
        validate_request(body.as_bytes()).is_none(),
        "a valid object with 256 nested arrays and objects should pass"
    );

    body.pop();
    assert!(
        validate_request(body.as_bytes()).is_some(),
        "the same deeply nested object with a missing delimiter should fail"
    );
}

#[test]
fn non_object_json_rejected() {
    let body = br#"[]"#;
    let rejection = validate_request(body);
    assert!(rejection.is_some(), "non-object JSON should be rejected");
}

#[test]
fn scalar_and_null_json_rejected() {
    for body in [
        br#""text""#.as_slice(),
        br#"42"#.as_slice(),
        br#"true"#.as_slice(),
        br#"null"#.as_slice(),
    ] {
        assert!(validate_request(body).is_some(), "non-object JSON should be rejected");
    }
}

#[test]
fn json_object_requires_validated_object_root() {
    assert!(is_json_object(b" \t\r\n{}"));
    assert!(!is_json_object(b"[]"));
    assert!(!is_json_object(b"null"));
    assert!(!is_json_object(b""));
}

#[tokio::test]
async fn empty_body_rejected_by_filter() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("{}").unwrap();
    let filter = AnthropicValidateFilter::from_config(&yaml).unwrap();
    let req = Box::leak(Box::new(crate::test_utils::make_request(
        http::Method::POST,
        "/v1/messages",
    )));
    let mut ctx = crate::test_utils::make_filter_context(req);
    let mut body = Some(Bytes::new());

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(
        matches!(action, FilterAction::Reject(_)),
        "empty body should be rejected"
    );
}

// -----------------------------------------------------------------------------
// Config
// -----------------------------------------------------------------------------

#[test]
fn default_config_parses() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("{}").unwrap();
    let filter = AnthropicValidateFilter::from_config(&yaml).unwrap();
    assert_eq!(
        filter.name(),
        "anthropic_validate",
        "filter name should be anthropic_validate"
    );
}

#[test]
fn zero_max_body_bytes_rejected() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("max_body_bytes: 0").unwrap();
    let result = AnthropicValidateFilter::from_config(&yaml);
    assert!(result.is_err(), "zero max_body_bytes should be rejected");
}

#[test]
fn rejects_max_body_bytes_above_ceiling() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("max_body_bytes: 67108865").unwrap();
    let result = AnthropicValidateFilter::from_config(&yaml);

    assert!(
        result.is_err(),
        "max_body_bytes above 64 MiB ceiling should be rejected"
    );
}
