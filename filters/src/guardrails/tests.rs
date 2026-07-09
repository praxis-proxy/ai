// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

use super::{
    config::{AiGuardrailsConfig, PhaseConfig, ProviderType},
    filter::AiGuardrailsFilter,
};

// =============================================================================
// Test helpers
// =============================================================================

/// Build an `ai_guardrails` filter configured with a `nemo` provider pointed
/// at `endpoint`.
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

/// Extract the [`praxis_filter::Rejection`] from a [`praxis_filter::FilterAction`],
/// failing the test (via `unwrap`) if the action is not `Reject`.
fn as_rejection(action: praxis_filter::FilterAction) -> praxis_filter::Rejection {
    match action {
        praxis_filter::FilterAction::Reject(rejection) => Some(rejection),
        praxis_filter::FilterAction::Continue
        | praxis_filter::FilterAction::Release
        | praxis_filter::FilterAction::BodyDone => None,
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
  response: true
"#,
    )
    .unwrap();

    let filter = AiGuardrailsFilter::from_config(&yaml).unwrap();
    assert_eq!(filter.name(), "ai_guardrails");
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
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"status": "passed"})))
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
        "nemo provider should pass through when status is 'passed'"
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
async fn on_request_body_modified_writes_filter_results() {
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

    let mock_server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "status": "modified",
            "content": "my ssn is [REDACTED]",
            "rails_status": {"pii masking": {"status": "blocked"}}
        })))
        .mount(&mock_server)
        .await;

    let endpoint = format!("{}/v1/guardrail/checks", mock_server.uri());
    let filter = nemo_filter(&endpoint);
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let mut body = Some(bytes::Bytes::from_static(
        br#"{"messages":[{"role":"user","content":"my ssn is 123-45-6789"}]}"#,
    ));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, praxis_filter::FilterAction::Continue),
        "modified verdict should forward unchanged (redact placeholder deferred to #579)"
    );
    assert_eq!(
        ctx.filter_results.get("ai_guardrails").unwrap().get("status"),
        Some("redacted"),
        "modified verdict should record a 'redacted' status in filter_results"
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
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"status": "passed"})))
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
