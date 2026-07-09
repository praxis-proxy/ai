// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

use super::{
    config::{AiGuardrailsConfig, PhaseConfig, ProviderType},
    filter::AiGuardrailsFilter,
};

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

    let json = br#"{"messages":[{"role":"user","content":"hello"}]}"#;
    let mut body = Some(bytes::Bytes::from_static(json));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, praxis_filter::FilterAction::Continue),
        "stub provider should pass through"
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
