// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

use super::{
    config::{AiGuardrailsConfig, PhaseConfig, ProviderType},
    filter::AiGuardrailsFilter,
};

// =============================================================================
// Test helpers
// =============================================================================

/// Build an `ai_guardrails` filter configured with a `nemo` provider pointed
/// at `endpoint`. Request phase enabled, response phase disabled (default).
fn nemo_filter(endpoint: &str) -> Box<dyn praxis_filter::HttpFilter> {
    let yaml: serde_yaml::Value = serde_yaml::from_str(&format!(
        r#"
provider:
  type: nemo
  endpoint: "{endpoint}"
"#,
    ))
    .unwrap();
    AiGuardrailsFilter::from_config(&yaml).unwrap()
}

/// Build an `ai_guardrails` filter with response phase enabled.
fn nemo_filter_response(endpoint: &str) -> Box<dyn praxis_filter::HttpFilter> {
    let yaml: serde_yaml::Value = serde_yaml::from_str(&format!(
        r#"
provider:
  type: nemo
  endpoint: "{endpoint}"
phase:
  request: false
  response: true
"#,
    ))
    .unwrap();
    AiGuardrailsFilter::from_config(&yaml).unwrap()
}

/// A valid OpenAI Chat Completion response body for testing.
fn chat_completion_response(content: &str) -> bytes::Bytes {
    bytes::Bytes::from(
        serde_json::to_vec(&serde_json::json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }]
        }))
        .unwrap(),
    )
}

/// Assert that a replaced response body is valid JSON with the expected
/// guardrail error structure and that the blocked rail name appears in the
/// message.
fn assert_blocked_body_json(body: &bytes::Bytes, expected_rail: &str) {
    let trimmed = body.trim_ascii_end();
    let json: serde_json::Value = serde_json::from_slice(trimmed).expect("replacement should be valid JSON");
    let error = json.get("error").expect("body should have an 'error' key");
    assert_eq!(error.get("code").and_then(|v| v.as_str()), Some("content_blocked"),);
    let message = error.get("message").and_then(|v| v.as_str()).unwrap_or("");
    assert!(
        message.contains(expected_rail),
        "error message should include the blocked rail name '{expected_rail}', got: {message}"
    );
}

/// Assert that a replaced response body contains an error JSON payload.
///
/// When the error JSON is larger than the original body,
/// `fit_to_committed_length` truncates to match `Content-Length`, so the
/// replacement may be partial JSON. We check:
/// 1. The body starts with `{"error":` (replacement happened)
/// 2. If the body is large enough for full JSON, validate structure
fn assert_error_body_json(body: &bytes::Bytes, expected_code: &str) {
    let trimmed = body.trim_ascii_end();
    assert!(
        trimmed.starts_with(br#"{"error":"#),
        "body should start with error JSON; got: {}",
        String::from_utf8_lossy(trimmed.get(..40).unwrap_or(trimmed)),
    );
    if let Ok(json) = serde_json::from_slice::<serde_json::Value>(trimmed) {
        let error = json.get("error").expect("body should have an 'error' key");
        assert_eq!(error.get("code").and_then(|v| v.as_str()), Some(expected_code));
    }
}

/// Extract the [`praxis_filter::Rejection`] from a [`praxis_filter::FilterAction`],
/// failing the test (via `unwrap`) if the action is not `Reject`.
fn as_rejection(action: praxis_filter::FilterAction) -> praxis_filter::Rejection {
    match action {
        praxis_filter::FilterAction::Reject(rejection) => Some(rejection),
        praxis_filter::FilterAction::Continue
        | praxis_filter::FilterAction::Release
        | praxis_filter::FilterAction::BodyDone
        | praxis_filter::FilterAction::TerminalResponse(_)
        | praxis_filter::FilterAction::StreamingTerminalResponse(_) => None,
    }
    .unwrap()
}

// =============================================================================
// General config
// =============================================================================

#[test]
fn valid_config_creates_filter() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
provider:
  type: nemo
  endpoint: "http://nemo:8000/v1/guardrail/checks"
"#,
    )
    .unwrap();

    let filter = AiGuardrailsFilter::from_config(&yaml).unwrap();
    assert_eq!(filter.name(), "ai_guardrails");
}

#[test]
fn valid_config_with_all_fields() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
provider:
  type: nemo
  endpoint: "http://nemo:8000/v1/guardrail/checks"
  timeout_ms: 3000
phase:
  request: true
  response: false
"#,
    )
    .unwrap();

    let filter = AiGuardrailsFilter::from_config(&yaml).unwrap();
    assert_eq!(filter.name(), "ai_guardrails");
}

#[test]
fn phase_response_true_accepted() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
provider:
  type: nemo
  endpoint: "http://nemo:8000/v1/guardrail/checks"
phase:
  response: true
"#,
    )
    .unwrap();

    let result = AiGuardrailsFilter::from_config(&yaml);
    assert!(result.is_ok(), "phase.response: true should be accepted");
}

#[test]
fn missing_provider_rejected() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        "
phase:
  request: true
",
    )
    .unwrap();

    let result = AiGuardrailsFilter::from_config(&yaml);
    assert!(result.is_err(), "config without provider should fail");
}

#[test]
fn unknown_provider_type_rejected() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
provider:
  type: nonexistent
  endpoint: "http://localhost:8000"
"#,
    )
    .unwrap();

    let result = AiGuardrailsFilter::from_config(&yaml);
    assert!(result.is_err(), "unknown provider type should fail");
}

#[test]
fn unknown_field_rejected() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
provider:
  type: nemo
  endpoint: "http://nemo:8000/v1/guardrail/checks"
unexpected_field: true
"#,
    )
    .unwrap();

    let result = AiGuardrailsFilter::from_config(&yaml);
    assert!(result.is_err(), "unknown fields should fail with deny_unknown_fields");
}

// =============================================================================
// Pipeline acceptance
// =============================================================================

#[test]
fn registry_creates_filter_by_name() {
    let mut registry = praxis_filter::FilterRegistry::with_builtins();
    praxis_filter::register_filters!(
        @register registry,
        http "ai_guardrails" => AiGuardrailsFilter::from_config
    );
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
provider:
  type: nemo
  endpoint: "http://nemo:8000/v1/guardrail/checks"
"#,
    )
    .unwrap();

    let filter = registry.create("ai_guardrails", &yaml);
    assert!(filter.is_ok(), "pipeline should accept ai_guardrails filter");
}

// =============================================================================
// NeMo provider config
// =============================================================================

#[test]
fn nemo_missing_endpoint_rejected() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        "
provider:
  type: nemo
",
    )
    .unwrap();

    let result = AiGuardrailsFilter::from_config(&yaml);
    assert!(result.is_err(), "missing endpoint should fail");
}

#[test]
fn nemo_empty_endpoint_rejected() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
provider:
  type: nemo
  endpoint: ""
"#,
    )
    .unwrap();

    let result = AiGuardrailsFilter::from_config(&yaml);
    assert!(result.is_err(), "empty endpoint should fail");
}

#[test]
fn nemo_zero_timeout_rejected() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
provider:
  type: nemo
  endpoint: "http://nemo:8000/v1/guardrail/checks"
  timeout_ms: 0
"#,
    )
    .unwrap();

    let result = AiGuardrailsFilter::from_config(&yaml);
    assert!(result.is_err(), "zero timeout should fail");
}

// =============================================================================
// HttpFilter trait
// =============================================================================

#[test]
fn body_access_is_read_write() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
provider:
  type: nemo
  endpoint: "http://nemo:8000/v1/guardrail/checks"
"#,
    )
    .unwrap();

    let filter = AiGuardrailsFilter::from_config(&yaml).unwrap();
    assert_eq!(filter.request_body_access(), praxis_filter::body::BodyAccess::ReadWrite);
}

#[test]
fn body_mode_is_stream_buffer() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
provider:
  type: nemo
  endpoint: "http://nemo:8000/v1/guardrail/checks"
"#,
    )
    .unwrap();

    let filter = AiGuardrailsFilter::from_config(&yaml).unwrap();
    assert_eq!(
        filter.request_body_mode(),
        praxis_filter::body::BodyMode::StreamBuffer {
            max_bytes: Some(1_048_576)
        },
        "body mode should be StreamBuffer with 1 MiB limit"
    );
}

#[tokio::test]
async fn on_request_continues() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
provider:
  type: nemo
  endpoint: "http://nemo:8000/v1/guardrail/checks"
"#,
    )
    .unwrap();

    let filter = AiGuardrailsFilter::from_config(&yaml).unwrap();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let action = filter.on_request(&mut ctx).await.unwrap();
    assert!(matches!(action, praxis_filter::FilterAction::Continue));
}

#[tokio::test]
async fn on_request_body_passes_through() {
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

    let mock_server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"status": "success"})))
        .mount(&mock_server)
        .await;

    let endpoint = format!("{}/v1/guardrail/checks", mock_server.uri());
    let filter = nemo_filter(&endpoint);
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let mut body = Some(bytes::Bytes::from_static(
        br#"{"messages":[{"role":"user","content":"hello"}]}"#,
    ));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, praxis_filter::FilterAction::Continue),
        "nemo provider should pass through when status is 'success'"
    );
    assert_eq!(
        ctx.filter_results.get("ai_guardrails").unwrap().get("status"),
        Some("passed"),
        "verdict should be written to filter_results for branch-routing"
    );
}

#[tokio::test]
async fn on_request_body_blocked_writes_filter_results() {
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

    let mock_server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "status": "blocked",
            "rails_status": {"toxicity": {"status": "blocked"}}
        })))
        .mount(&mock_server)
        .await;

    let endpoint = format!("{}/v1/guardrail/checks", mock_server.uri());
    let filter = nemo_filter(&endpoint);
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let mut body = Some(bytes::Bytes::from_static(
        br#"{"messages":[{"role":"user","content":"hello"}]}"#,
    ));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    let rejection = as_rejection(action);
    assert_eq!(rejection.status, 403, "blocked verdict should reject with HTTP 403");
    let rejection_body = rejection.body.unwrap();
    let body_text = String::from_utf8_lossy(&rejection_body);
    assert!(
        body_text.contains("toxicity"),
        "rejection body should include the blocked rail name, got: {body_text}"
    );
    assert_eq!(
        ctx.filter_results.get("ai_guardrails").unwrap().get("status"),
        Some("blocked"),
        "verdict should be written to filter_results even when the request is rejected"
    );
}

#[tokio::test]
async fn on_request_body_error_status_fails_closed() {
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

    let mock_server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "status": "error",
            "rails_status": {},
            "guardrails_data": {
                "error": "Could not load guardrails configuration.",
                "details": "Invalid config path."
            }
        })))
        .mount(&mock_server)
        .await;

    let endpoint = format!("{}/v1/guardrail/checks", mock_server.uri());
    let filter = nemo_filter(&endpoint);
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let mut body = Some(bytes::Bytes::from_static(
        br#"{"messages":[{"role":"user","content":"hello"}]}"#,
    ));

    let result = filter.on_request_body(&mut ctx, &mut body, true).await;
    assert!(
        result.is_err(),
        "NeMo error status should fail closed rather than pass through"
    );
}

#[tokio::test]
async fn on_request_body_not_end_of_stream_continues_without_evaluating() {
    // The endpoint is unreachable in this test environment, so
    // the call to the provider would fail and return a `FilterError`.
    //  A `Continue` here proves evaluation was skipped entirely.
    let filter = nemo_filter("http://nemo:8000/v1/guardrail/checks");
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let mut body = Some(bytes::Bytes::from_static(
        br#"{"messages":[{"role":"user","content":"hello"}]}"#,
    ));

    let action = filter.on_request_body(&mut ctx, &mut body, false).await.unwrap();
    assert!(
        matches!(action, praxis_filter::FilterAction::Continue),
        "chunks before end_of_stream should be passed through without provider evaluation"
    );
}

#[tokio::test]
async fn on_request_body_phase_request_disabled_skips_evaluation() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
provider:
  type: nemo
  endpoint: "http://nemo:8000/v1/guardrail/checks"
phase:
  request: false
"#,
    )
    .unwrap();
    let filter = AiGuardrailsFilter::from_config(&yaml).unwrap();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let mut body = Some(bytes::Bytes::from_static(
        br#"{"messages":[{"role":"user","content":"hello"}]}"#,
    ));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, praxis_filter::FilterAction::Continue),
        "phase.request=false should skip provider evaluation entirely"
    );
    assert!(
        !ctx.filter_results.contains_key("ai_guardrails"),
        "no verdict should be recorded when request-phase evaluation is disabled"
    );
}

#[tokio::test]
async fn on_request_body_none_continues_without_evaluating() {
    let filter = nemo_filter("http://nemo:8000/v1/guardrail/checks");
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let mut body = None;

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, praxis_filter::FilterAction::Continue),
        "a missing body should be passed through without provider evaluation"
    );
}

#[tokio::test]
async fn on_request_body_empty_continues_without_evaluating() {
    let filter = nemo_filter("http://nemo:8000/v1/guardrail/checks");
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let mut body = Some(bytes::Bytes::new());

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, praxis_filter::FilterAction::Continue),
        "an empty body should be passed through without provider evaluation"
    );
}

// =============================================================================
// Request body validation (fail-closed on unsupported bodies)
// =============================================================================

#[tokio::test]
async fn on_request_body_invalid_json_rejected() {
    let filter = nemo_filter("http://nemo:8000/v1/guardrail/checks");
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let mut body = Some(bytes::Bytes::from_static(b"not json"));

    let result = filter.on_request_body(&mut ctx, &mut body, true).await;
    assert!(
        result.is_err(),
        "non-JSON body should fail closed rather than skip evaluation"
    );
}

#[tokio::test]
async fn on_request_body_missing_messages_key_rejected() {
    let filter = nemo_filter("http://nemo:8000/v1/guardrail/checks");
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let mut body = Some(bytes::Bytes::from_static(br#"{"model":"test"}"#));

    let result = filter.on_request_body(&mut ctx, &mut body, true).await;
    assert!(
        result.is_err(),
        "body without a 'messages' field should fail closed rather than skip evaluation"
    );
}

#[tokio::test]
async fn on_request_body_messages_not_array_rejected() {
    let filter = nemo_filter("http://nemo:8000/v1/guardrail/checks");
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let mut body = Some(bytes::Bytes::from_static(br#"{"messages":"hello"}"#));

    let result = filter.on_request_body(&mut ctx, &mut body, true).await;
    assert!(
        result.is_err(),
        "non-array 'messages' field should fail closed rather than skip evaluation"
    );
}

#[tokio::test]
async fn on_request_body_empty_messages_array_still_evaluated() {
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

    let mock_server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"status": "success"})))
        .mount(&mock_server)
        .await;

    let endpoint = format!("{}/v1/guardrail/checks", mock_server.uri());
    let filter = nemo_filter(&endpoint);
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let mut body = Some(bytes::Bytes::from_static(br#"{"messages":[]}"#));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, praxis_filter::FilterAction::Continue),
        "an empty (but well-formed) messages array is a recognized body shape and should still be sent to the provider, not treated as fail-closed"
    );
}

// =============================================================================
// NeMo provider HTTP behavior (fail-closed on provider errors)
// =============================================================================

#[tokio::test]
async fn on_request_body_nemo_non_2xx_rejected() {
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

    let mock_server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&mock_server)
        .await;

    let endpoint = format!("{}/v1/guardrail/checks", mock_server.uri());
    let filter = nemo_filter(&endpoint);
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let mut body = Some(bytes::Bytes::from_static(
        br#"{"messages":[{"role":"user","content":"hello"}]}"#,
    ));

    let result = filter.on_request_body(&mut ctx, &mut body, true).await;
    assert!(
        result.is_err(),
        "a non-2xx response from the provider should fail closed rather than pass through"
    );
}

#[tokio::test]
async fn on_request_body_nemo_invalid_json_response_rejected() {
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

    let mock_server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string("not json"))
        .mount(&mock_server)
        .await;

    let endpoint = format!("{}/v1/guardrail/checks", mock_server.uri());
    let filter = nemo_filter(&endpoint);
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let mut body = Some(bytes::Bytes::from_static(
        br#"{"messages":[{"role":"user","content":"hello"}]}"#,
    ));

    let result = filter.on_request_body(&mut ctx, &mut body, true).await;
    assert!(
        result.is_err(),
        "a non-JSON response body from the provider should fail closed rather than pass through"
    );
}

// =============================================================================
// ProviderType serde
// =============================================================================

#[test]
fn provider_type_nemo_parses() {
    let parsed: ProviderType = serde_yaml::from_str(r#""nemo""#).unwrap();
    assert_eq!(parsed, ProviderType::Nemo);
}

#[test]
fn provider_type_unknown_rejected() {
    let result: Result<ProviderType, _> = serde_yaml::from_str(r#""openai""#);
    assert!(result.is_err(), "unknown provider type should fail");
}

// =============================================================================
// PhaseConfig
// =============================================================================

#[test]
fn phase_config_default() {
    let phase = PhaseConfig::default();
    assert!(phase.request, "default request should be true");
    assert!(!phase.response, "default response should be false");
}

#[test]
fn phase_config_custom_values() {
    let parsed: PhaseConfig = serde_yaml::from_str(
        "
request: false
response: true
",
    )
    .unwrap();
    assert!(!parsed.request, "request should be false");
    assert!(parsed.response, "response should be true");
}

#[test]
fn phase_config_omitted_uses_defaults() {
    let parsed: PhaseConfig = serde_yaml::from_str("{}").unwrap();
    assert!(parsed.request, "omitted request should default to true");
    assert!(!parsed.response, "omitted response should default to false");
}

#[test]
fn phase_config_unknown_field_rejected() {
    let result: Result<PhaseConfig, _> = serde_yaml::from_str(
        "
request: true
unknown: 42
",
    );
    assert!(result.is_err(), "unknown fields should fail with deny_unknown_fields");
}

// =============================================================================
// AiGuardrailsConfig serde
// =============================================================================

#[test]
fn guardrails_config_minimal_valid() {
    let parsed: AiGuardrailsConfig = serde_yaml::from_str(
        r#"
provider:
  type: nemo
  endpoint: "http://nemo:8000/v1/guardrail/checks"
"#,
    )
    .unwrap();

    assert_eq!(parsed.provider.provider_type, ProviderType::Nemo);
}

#[test]
fn guardrails_config_missing_provider_rejected() {
    let result: Result<AiGuardrailsConfig, _> = serde_yaml::from_str("{}");
    assert!(result.is_err(), "missing provider should fail");
}

#[test]
fn guardrails_config_unknown_field_rejected() {
    let result: Result<AiGuardrailsConfig, _> = serde_yaml::from_str(
        r#"
provider:
  type: nemo
  endpoint: "http://nemo:8000/v1/guardrail/checks"
bogus: true
"#,
    );
    assert!(result.is_err(), "unknown fields should fail with deny_unknown_fields");
}

#[test]
fn guardrails_config_with_phase_overrides() {
    let parsed: AiGuardrailsConfig = serde_yaml::from_str(
        r#"
provider:
  type: nemo
  endpoint: "http://nemo:8000/v1/guardrail/checks"
phase:
  request: false
  response: true
"#,
    )
    .unwrap();

    assert!(!parsed.phase.request, "overridden request should be false");
    assert!(parsed.phase.response, "overridden response should be true");
}

// =============================================================================
// fit_to_committed_length
// =============================================================================

#[test]
fn fit_to_committed_length_equal_size_returns_as_is() {
    use super::filter::fit_to_committed_length;

    let original = Some(bytes::Bytes::from_static(b"1234567890"));
    let replacement = "abcdefghij".to_owned();
    let result = fit_to_committed_length(replacement, &original);
    assert_eq!(result.len(), 10);
    assert_eq!(&*result, b"abcdefghij");
}

#[test]
fn fit_to_committed_length_shorter_replacement_is_padded() {
    use super::filter::fit_to_committed_length;

    let original = Some(bytes::Bytes::from_static(b"1234567890abcdef"));
    let replacement = "short".to_owned();
    let result = fit_to_committed_length(replacement, &original);
    assert_eq!(result.len(), 16, "padded result must match original body length");
    assert!(result.starts_with(b"short"), "replacement content must be preserved");
    assert!(
        result.get(5..).unwrap_or_default().iter().all(|&b| b == b' '),
        "padding bytes must be ASCII spaces"
    );
}

#[test]
fn fit_to_committed_length_longer_replacement_is_truncated() {
    use super::filter::fit_to_committed_length;

    let original = Some(bytes::Bytes::from_static(b"tiny"));
    let replacement = "this replacement is much longer than the original".to_owned();
    let result = fit_to_committed_length(replacement, &original);
    assert_eq!(result.len(), 4, "truncated result must match original body length");
    assert_eq!(&*result, b"this");
}

#[test]
fn fit_to_committed_length_truncation_respects_utf8_boundary() {
    use super::filter::fit_to_committed_length;

    // "héllo" is 6 bytes (h=1, é=2, l=1, l=1, o=1). Truncating to 2 bytes
    // would split the é (bytes 1-2), so the function should back up to
    // byte 1 ("h") and pad with a space.
    let original = Some(bytes::Bytes::from_static(b"ab"));
    let replacement = "héllo".to_owned();
    let result = fit_to_committed_length(replacement, &original);
    assert_eq!(result.len(), 2, "must match original body length");
    let text = std::str::from_utf8(&result).expect("result must be valid UTF-8");
    assert_eq!(text, "h ", "should truncate before the multi-byte char and pad");
}

#[test]
fn fit_to_committed_length_none_body_returns_empty() {
    use super::filter::fit_to_committed_length;

    let result = fit_to_committed_length("anything".to_owned(), &None);
    assert!(
        result.is_empty(),
        "None original body means 0 committed length, so result should be empty"
    );
}

// =============================================================================
// Response body access
// =============================================================================

#[test]
fn response_body_access_none_when_phase_disabled() {
    let filter = nemo_filter("http://nemo:8000/v1/guardrail/checks");
    assert_eq!(
        filter.response_body_access(),
        praxis_filter::body::BodyAccess::None,
        "response body access should be None when response phase is disabled"
    );
}

#[test]
fn response_body_access_read_write_when_phase_enabled() {
    let filter = nemo_filter_response("http://nemo:8000/v1/guardrail/checks");
    assert_eq!(
        filter.response_body_access(),
        praxis_filter::body::BodyAccess::ReadWrite,
        "response body access should be ReadWrite when response phase is enabled"
    );
}

// =============================================================================
// on_response_body: skip conditions
// =============================================================================

#[test]
fn on_response_body_not_end_of_stream_continues() {
    let filter = nemo_filter_response("http://nemo:8000/v1/guardrail/checks");
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let mut body = Some(chat_completion_response("hello"));

    let action = filter.on_response_body(&mut ctx, &mut body, false).unwrap();
    assert!(
        matches!(action, praxis_filter::FilterAction::Continue),
        "chunks before end_of_stream should pass through without evaluation"
    );
}

#[test]
fn on_response_body_phase_disabled_skips_evaluation() {
    let filter = nemo_filter("http://nemo:8000/v1/guardrail/checks");
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let mut body = Some(chat_completion_response("hello"));

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(action, praxis_filter::FilterAction::Continue),
        "phase.response=false should skip evaluation"
    );
    assert!(
        !ctx.filter_results.contains_key("ai_guardrails"),
        "no verdict should be recorded when response-phase evaluation is disabled"
    );
}

#[test]
fn on_response_body_none_continues() {
    let filter = nemo_filter_response("http://nemo:8000/v1/guardrail/checks");
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let mut body = None;

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(action, praxis_filter::FilterAction::Continue),
        "a missing body should pass through without evaluation"
    );
}

#[test]
fn on_response_body_empty_continues() {
    let filter = nemo_filter_response("http://nemo:8000/v1/guardrail/checks");
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let mut body = Some(bytes::Bytes::new());

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(action, praxis_filter::FilterAction::Continue),
        "an empty body should pass through without evaluation"
    );
}

#[tokio::test]
async fn on_response_sse_does_not_upgrade_to_stream_buffer() {
    let filter = nemo_filter_response("http://nemo:8000/v1/guardrail/checks");
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let mut resp = crate::test_utils::make_response();
    resp.headers
        .insert("content-type", http::HeaderValue::from_static("text/event-stream"));
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());
    ctx.response_header = None;

    assert_eq!(
        ctx.response_body_mode,
        praxis_filter::BodyMode::Stream,
        "SSE responses should stay in Stream mode (no buffering)"
    );
}

#[tokio::test]
async fn on_response_json_upgrades_to_stream_buffer() {
    let filter = nemo_filter_response("http://nemo:8000/v1/guardrail/checks");
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let mut resp = crate::test_utils::make_response();
    resp.headers
        .insert("content-type", http::HeaderValue::from_static("application/json"));
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());
    ctx.response_header = None;

    assert!(
        matches!(ctx.response_body_mode, praxis_filter::BodyMode::StreamBuffer { .. }),
        "JSON responses should be upgraded to StreamBuffer for guardrail evaluation"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_body_stream_mode_skips_evaluation() {
    let filter = nemo_filter_response("http://nemo:8000/v1/guardrail/checks");
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    // Default mode is Stream - evaluation should be skipped even with valid body
    let mut body = Some(chat_completion_response("hello"));

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(matches!(action, praxis_filter::FilterAction::Continue));
    assert!(
        !ctx.filter_results.contains_key("ai_guardrails"),
        "Stream mode should skip evaluation without writing filter results"
    );
}

// =============================================================================
// on_response_body: response body validation (fail-closed)
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_body_invalid_json_replaces_body() {
    let filter = nemo_filter_response("http://nemo:8000/v1/guardrail/checks");
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.response_body_mode = praxis_filter::BodyMode::StreamBuffer { max_bytes: None };
    let mut body = Some(bytes::Bytes::from_static(
        b"this is not valid json but long enough for the error replacement to fit properly here",
    ));

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(matches!(action, praxis_filter::FilterAction::Continue));
    assert_error_body_json(&body.unwrap(), "evaluation_failed");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_body_missing_choices_replaces_body() {
    let filter = nemo_filter_response("http://nemo:8000/v1/guardrail/checks");
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.response_body_mode = praxis_filter::BodyMode::StreamBuffer { max_bytes: None };
    let mut body = Some(bytes::Bytes::from_static(br#"{"id":"test"}"#));

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(matches!(action, praxis_filter::FilterAction::Continue));
    assert_error_body_json(&body.unwrap(), "evaluation_failed");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_body_empty_choices_replaces_body() {
    let filter = nemo_filter_response("http://nemo:8000/v1/guardrail/checks");
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.response_body_mode = praxis_filter::BodyMode::StreamBuffer { max_bytes: None };
    let mut body = Some(bytes::Bytes::from_static(br#"{"choices":[]}"#));

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(matches!(action, praxis_filter::FilterAction::Continue));
    assert_error_body_json(&body.unwrap(), "evaluation_failed");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_body_mixed_choices_replaces_body() {
    let filter = nemo_filter_response("http://nemo:8000/v1/guardrail/checks");
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.response_body_mode = praxis_filter::BodyMode::StreamBuffer { max_bytes: None };
    let mut body = Some(bytes::Bytes::from(
        serde_json::to_vec(&serde_json::json!({
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "hi"}},
                {"index": 1, "finish_reason": "stop"}
            ]
        }))
        .unwrap(),
    ));

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(matches!(action, praxis_filter::FilterAction::Continue));
    assert_error_body_json(&body.unwrap(), "evaluation_failed");
}

// =============================================================================
// on_response_body: provider verdicts (via wiremock)
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_body_error_status_fails_closed() {
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

    let mock_server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "status": "error",
            "rails_status": {},
            "guardrails_data": {
                "error": "Could not load guardrails configuration.",
                "details": "Invalid config path."
            }
        })))
        .mount(&mock_server)
        .await;

    let endpoint = format!("{}/v1/guardrail/checks", mock_server.uri());
    let filter = nemo_filter_response(&endpoint);
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.response_body_mode = praxis_filter::BodyMode::StreamBuffer { max_bytes: None };
    let mut body = Some(chat_completion_response("hello"));

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(matches!(action, praxis_filter::FilterAction::Continue));
    assert_error_body_json(&body.unwrap(), "evaluation_failed");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_body_passes_through() {
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

    let mock_server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"status": "success"})))
        .mount(&mock_server)
        .await;

    let endpoint = format!("{}/v1/guardrail/checks", mock_server.uri());
    let filter = nemo_filter_response(&endpoint);
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.response_body_mode = praxis_filter::BodyMode::StreamBuffer { max_bytes: None };
    let mut body = Some(chat_completion_response("hello world"));

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(action, praxis_filter::FilterAction::Continue),
        "success verdict should continue"
    );
    assert_eq!(
        ctx.filter_results.get("ai_guardrails").unwrap().get("status"),
        Some("passed"),
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_body_blocked_replaces_body() {
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

    let mock_server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "status": "blocked",
            "rails_status": {"toxicity": {"status": "blocked"}}
        })))
        .mount(&mock_server)
        .await;

    let endpoint = format!("{}/v1/guardrail/checks", mock_server.uri());
    let filter = nemo_filter_response(&endpoint);
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.response_body_mode = praxis_filter::BodyMode::StreamBuffer { max_bytes: None };
    let original_len = chat_completion_response("something toxic").len();
    let mut body = Some(chat_completion_response("something toxic"));

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(matches!(action, praxis_filter::FilterAction::Continue));
    let replaced = body.expect("body should be replaced, not cleared");
    assert_eq!(replaced.len(), original_len, "must match original Content-Length");
    assert_eq!(
        ctx.filter_results.get("ai_guardrails").unwrap().get("status"),
        Some("blocked")
    );
    assert_blocked_body_json(&replaced, "toxicity");
}
