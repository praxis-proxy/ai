// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

use praxis_filter::{HttpFilter, HttpFilterContext};

#[cfg(feature = "openai-responses")]
use super::filter::record_local_tool_verdict;
use super::{
    config::{AiGuardrailsConfig, PhaseConfig, ProviderType},
    filter::{AiGuardrailsFilter, ResponseScreening},
    providers::GuardResult,
};

// =============================================================================
// Test helpers
// =============================================================================

/// Build an `ai_guardrails` filter configured with a `nemo` provider pointed
/// at `endpoint`. Request phase enabled, response phase disabled (default).
fn nemo_filter(endpoint: &str) -> Box<dyn HttpFilter> {
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
fn nemo_filter_response(endpoint: &str) -> Box<dyn HttpFilter> {
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

/// Build an `ai_guardrails` filter with request and response phases enabled.
fn nemo_filter_request_and_response(endpoint: &str) -> Box<dyn HttpFilter> {
    let yaml: serde_yaml::Value = serde_yaml::from_str(&format!(
        r#"
provider:
  type: nemo
  endpoint: "{endpoint}"
phase:
  request: true
  response: true
"#,
    ))
    .unwrap();
    AiGuardrailsFilter::from_config(&yaml).unwrap()
}

/// Mount a JSON `/v1/checks` response on `mock_server`.
async fn mount_nemo_checks_response(mock_server: &wiremock::MockServer, body: serde_json::Value) {
    mount_nemo_checks_responses(mock_server, vec![body]).await;
}

/// Mount sequential JSON `/v1/checks` responses, one per callout.
async fn mount_nemo_checks_responses(mock_server: &wiremock::MockServer, bodies: Vec<serde_json::Value>) {
    use std::sync::Mutex;

    use wiremock::{Mock, Request, Respond, ResponseTemplate, matchers::method};

    struct SequentialJson(Mutex<std::vec::IntoIter<serde_json::Value>>);

    impl Respond for SequentialJson {
        fn respond(&self, _: &Request) -> ResponseTemplate {
            let body = self
                .0
                .lock()
                .expect("sequence mutex")
                .next()
                .expect("unexpected extra NeMo call");
            ResponseTemplate::new(200).set_body_json(body)
        }
    }

    Mock::given(method("POST"))
        .respond_with(SequentialJson(Mutex::new(bodies.into_iter())))
        .mount(mock_server)
        .await;
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
    let json: serde_json::Value = serde_json::from_slice(body).expect("replacement should be valid JSON");
    let error = json.get("error").expect("body should have an 'error' key");
    assert_eq!(error.get("code").and_then(|v| v.as_str()), Some("content_blocked"),);
    let message = error.get("message").and_then(|v| v.as_str()).unwrap_or("");
    assert!(
        message.contains(expected_rail),
        "error message should include the blocked rail name '{expected_rail}', got: {message}"
    );
}

/// Assert that a replaced response body is an error JSON document.
fn assert_error_body_json(body: &bytes::Bytes, expected_code: &str) {
    let json: serde_json::Value = serde_json::from_slice(body).expect("replacement should be valid JSON");
    let error = json.get("error").expect("body should have an 'error' key");
    assert_eq!(error.get("code").and_then(|v| v.as_str()), Some(expected_code));
}

/// Read a JSON string at `pointer`, or `None` if the path is missing or not a string.
fn json_str<'a>(value: &'a serde_json::Value, pointer: &str) -> Option<&'a str> {
    value.pointer(pointer).and_then(serde_json::Value::as_str)
}

/// Assert request-phase redaction recorded `redacted` and rewrote every listed user turn.
fn assert_request_users_redacted(ctx: &HttpFilterContext<'_>, body: &bytes::Bytes, expected_messages: &[(&str, &str)]) {
    assert_eq!(
        ctx.filter_results.get("ai_guardrails").and_then(|r| r.get("status")),
        Some("redacted")
    );
    assert_no_injected_content_length(ctx);
    assert_forwarded_messages(body, expected_messages);
}

/// Assert the filter did not inject a `content-length` header.
fn assert_no_injected_content_length(ctx: &HttpFilterContext<'_>) {
    assert!(
        ctx.extra_request_headers
            .iter()
            .all(|(k, _)| k.as_ref() != "content-length"),
        "filter must not set content-length (core handles framing)"
    );
}

/// Assert the forwarded request body matches `expected_messages` in order.
fn assert_forwarded_messages(body: &bytes::Bytes, expected_messages: &[(&str, &str)]) {
    let forwarded: serde_json::Value = serde_json::from_slice(body).expect("forwarded JSON");
    assert_eq!(
        json_str(&forwarded, "/model"),
        Some("test"),
        "non-message fields should be preserved"
    );
    let messages = forwarded
        .get("messages")
        .and_then(|value| value.as_array())
        .expect("messages should be an array");
    assert_eq!(messages.len(), expected_messages.len());
    for (index, (role, content)) in expected_messages.iter().enumerate() {
        assert_eq!(
            messages.get(index).and_then(|message| json_str(message, "/role")),
            Some(*role),
            "message {index} role"
        );
        assert_eq!(
            messages.get(index).and_then(|message| json_str(message, "/content")),
            Some(*content),
            "message {index} content"
        );
    }
}

/// Assert response-phase redaction rewrote assistant `content` and stayed valid JSON.
fn assert_redacted_assistant_content(ctx: &HttpFilterContext<'_>, body: &bytes::Bytes, expected: &str) {
    assert_natural_json(body);
    assert_eq!(
        ctx.filter_results.get("ai_guardrails").and_then(|r| r.get("status")),
        Some("redacted")
    );
    let json: serde_json::Value = serde_json::from_slice(body).expect("redacted body should remain valid JSON");
    assert_eq!(
        json_str(&json, "/choices/0/message/content"),
        Some(expected),
        "assistant content should be replaced with NeMo content"
    );
}

/// Buffer the response body so `on_response_body` evaluates it.
fn buffer_response(ctx: &mut HttpFilterContext<'_>) {
    ctx.response_body_mode = praxis_filter::BodyMode::StreamBuffer { max_bytes: None };
}

/// Run as pipeline filter 0 and record that the request carried a body, as
/// `on_request_body` does inside a pipeline.
fn mark_request_body(ctx: &mut HttpFilterContext<'_>) {
    ctx.current_filter_id = Some(0);
    ctx.insert_filter_state(ResponseScreening::Screen);
}

/// The response-phase decision recorded for filter 0.
fn screening(ctx: &HttpFilterContext<'_>) -> Option<ResponseScreening> {
    ctx.get_filter_state::<ResponseScreening>().copied()
}

/// The body is one JSON value and is not space-padded out to a longer length.
fn assert_natural_json(body: &bytes::Bytes) {
    let json: serde_json::Value = serde_json::from_slice(body).expect("unpadded body must be valid JSON");
    let compact = serde_json::to_string(&json).expect("reserialize");
    assert_eq!(
        body.as_ref(),
        compact.as_bytes(),
        "body should be the serialized JSON with no space padding"
    );
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
  endpoint: "http://nemo:8000/v1/checks"
"#,
    )
    .unwrap();

    let filter = AiGuardrailsFilter::from_config(&yaml).unwrap();
    assert_eq!(filter.name(), "ai_guardrails");
}

#[test]
fn empty_nemo_config_ids_rejected() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
provider:
  type: nemo
  endpoint: "http://nemo:8000/v1/checks"
  guardrails:
    config_ids: []
"#,
    )
    .unwrap();

    let error = AiGuardrailsFilter::from_config(&yaml).err().unwrap();
    assert_eq!(
        error.to_string(),
        "ai_guardrails (nemo): 'guardrails.config_ids' must not be empty; omit 'guardrails' to use the service default"
    );
}

#[test]
fn nemo_token_file_must_exist_at_startup() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
provider:
  type: nemo
  endpoint: "http://127.0.0.1:8000/v1/checks"
  token_file: /nonexistent/nemo-token
"#,
    )
    .unwrap();

    let err = AiGuardrailsFilter::from_config(&yaml)
        .err()
        .expect("a missing token file must be rejected at startup");
    assert!(err.to_string().contains("cannot read"), "{err}");
}

#[test]
fn nemo_token_file_rejects_non_loopback_http() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
provider:
  type: nemo
  endpoint: "http://nemo:8000/v1/checks"
  token_file: /var/run/secrets/tokens/nemo/token
"#,
    )
    .unwrap();

    let err = AiGuardrailsFilter::from_config(&yaml)
        .err()
        .expect("a token file on a non-loopback http endpoint must be rejected");
    assert!(err.to_string().contains("loopback"), "{err}");
}

#[test]
fn valid_config_with_all_fields() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
provider:
  type: nemo
  endpoint: "http://nemo:8000/v1/checks"
  timeout_ms: 3000
  max_message_checks: 16
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
fn nemo_provider_scoped_private_endpoint_opt_in_is_rejected() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
provider:
  type: nemo
  endpoint: "http://127.0.0.1:8000/v1/checks"
  allow_private_endpoint: true
"#,
    )
    .unwrap();

    assert!(
        AiGuardrailsFilter::from_config(&yaml).is_err(),
        "private endpoint policy must be configured globally, not on the provider"
    );
}

#[test]
fn phase_response_true_accepted() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
provider:
  type: nemo
  endpoint: "http://nemo:8000/v1/checks"
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
  endpoint: "http://nemo:8000/v1/checks"
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
  endpoint: "http://nemo:8000/v1/checks"
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
  endpoint: "http://nemo:8000/v1/checks"
  timeout_ms: 0
"#,
    )
    .unwrap();

    let result = AiGuardrailsFilter::from_config(&yaml);
    assert!(result.is_err(), "zero timeout should fail");
}

#[test]
fn nemo_zero_max_message_checks_rejected() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
provider:
  type: nemo
  endpoint: "http://nemo:8000/v1/checks"
  max_message_checks: 0
"#,
    )
    .unwrap();

    let result = AiGuardrailsFilter::from_config(&yaml);
    assert!(result.is_err(), "zero max_message_checks should fail");
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
  endpoint: "http://nemo:8000/v1/checks"
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
  endpoint: "http://nemo:8000/v1/checks"
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

#[test]
fn response_only_body_mode_uses_json_body_limit() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
provider:
  type: nemo
  endpoint: "http://nemo:8000/v1/checks"
phase:
  request: false
  response: true
"#,
    )
    .unwrap();

    let filter = AiGuardrailsFilter::from_config(&yaml).unwrap();
    assert_eq!(
        filter.request_body_mode(),
        praxis_filter::body::BodyMode::StreamBuffer {
            max_bytes: Some(praxis_filter::body::MAX_JSON_BODY_BYTES)
        },
        "response-only guardrails should use the absolute JSON body limit"
    );
}

#[tokio::test]
async fn on_request_continues() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
provider:
  type: nemo
  endpoint: "http://nemo:8000/v1/checks"
"#,
    )
    .unwrap();

    let filter = AiGuardrailsFilter::from_config(&yaml).unwrap();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let action = filter.on_request(&mut ctx).await.unwrap();
    assert!(
        matches!(action, praxis_filter::FilterAction::Continue),
        "guardrails request-header phase should continue"
    );
}

#[tokio::test]
async fn on_request_body_passes_through() {
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

    let mock_server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({"status": "passed", "content": "hello"})),
        )
        .mount(&mock_server)
        .await;

    let endpoint = format!("{}/v1/checks", mock_server.uri());
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
async fn callout_timeout_starts_when_evaluation_begins() {
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

    let mock_server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({"status": "passed", "content": "hello"})),
        )
        .mount(&mock_server)
        .await;

    let yaml: serde_yaml::Value = serde_yaml::from_str(&format!(
        r#"
provider:
  type: nemo
  endpoint: "{}/v1/checks"
  timeout_ms: 1000
"#,
        mock_server.uri()
    ))
    .unwrap();
    let filter = AiGuardrailsFilter::from_config(&yaml).unwrap();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.request_start = std::time::Instant::now() - std::time::Duration::from_secs(30);
    let mut body = Some(bytes::Bytes::from_static(
        br#"{"messages":[{"role":"user","content":"hello"}]}"#,
    ));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(matches!(action, praxis_filter::FilterAction::Continue));
}

#[tokio::test]
#[expect(clippy::too_many_lines, reason = "covers the complete blocked-verdict response")]
async fn on_request_body_blocked_writes_filter_results() {
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

    let mock_server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "status": "blocked",
            "content": "blocked",
            "rail": "toxicity"
        })))
        .mount(&mock_server)
        .await;

    let endpoint = format!("{}/v1/checks", mock_server.uri());
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
async fn on_request_body_modified_rewrites_last_user_message() {
    use wiremock::MockServer;

    let mock_server = MockServer::start().await;
    mount_nemo_checks_responses(
        &mock_server,
        vec![
            serde_json::json!({"status": "passed", "content": "first"}),
            serde_json::json!({"status": "modified", "content": "masked text", "rail": "pii"}),
        ],
    )
    .await;

    let filter = nemo_filter(&format!("{}/v1/checks", mock_server.uri()));
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let mut body = Some(bytes::Bytes::from_static(
        br#"{"model":"test","messages":[{"role":"system","content":"Be helpful"},{"role":"user","content":"first"},{"role":"assistant","content":"ok"},{"role":"user","content":"my email is secret@example.com"}]}"#,
    ));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(matches!(action, praxis_filter::FilterAction::Continue));
    assert_request_users_redacted(
        &ctx,
        body.as_ref().expect("forwarded body"),
        &[
            ("system", "Be helpful"),
            ("user", "first"),
            ("assistant", "ok"),
            ("user", "masked text"),
        ],
    );
}

#[tokio::test]
async fn on_request_body_modified_rewrites_all_user_messages() {
    use wiremock::MockServer;

    let mock_server = MockServer::start().await;
    mount_nemo_checks_responses(
        &mock_server,
        vec![
            serde_json::json!({"status": "modified", "content": "My mail is <EMAIL>", "rail": "pii"}),
            serde_json::json!({"status": "modified", "content": "my credit card is <CREDIT-CARD>", "rail": "pii"}),
        ],
    )
    .await;

    let filter = nemo_filter(&format!("{}/v1/checks", mock_server.uri()));
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let mut body = Some(bytes::Bytes::from_static(
        br#"{"model":"test","messages":[{"role":"user","content":"My mail is xxx@gmail.com"},{"role":"assistant","content":"No."},{"role":"user","content":"my credit card is 1234-5678-92211"}]}"#,
    ));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(matches!(action, praxis_filter::FilterAction::Continue));
    assert_request_users_redacted(
        &ctx,
        body.as_ref().expect("forwarded body"),
        &[
            ("user", "My mail is <EMAIL>"),
            ("assistant", "No."),
            ("user", "my credit card is <CREDIT-CARD>"),
        ],
    );
}

#[tokio::test]
async fn on_request_body_modified_without_user_message_fails_closed() {
    let filter = AiGuardrailsFilter::with_verdict(
        GuardResult::redact_message(0, "masked".into(), "pii".into()),
        PhaseConfig::default(),
    )
    .expect("test filter");
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let mut body = Some(bytes::Bytes::from_static(
        br#"{"messages":[{"role":"system","content":"Be helpful"}]}"#,
    ));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    let rejection = as_rejection(action);
    assert_eq!(
        rejection.status, 502,
        "a redaction that cannot be applied must be rejected, not failed open"
    );
    let rejection_body = rejection.body.expect("rejection body");
    let body_text = String::from_utf8_lossy(&rejection_body);
    assert!(
        body_text.contains("is not a user turn"),
        "system message must not be rewritten, got: {body_text}"
    );
}

#[tokio::test]
async fn on_request_body_modified_refuses_non_string_content() {
    let filter = AiGuardrailsFilter::with_verdict(
        GuardResult::redact_message(0, "masked".into(), "pii".into()),
        PhaseConfig::default(),
    )
    .expect("test filter");
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let original = bytes::Bytes::from_static(
        br#"{"model":"test","messages":[{"role":"user","content":[{"type":"text","text":"ssn 123-45-6789"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]}]}"#,
    );
    let mut body = Some(original.clone());

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    let rejection = as_rejection(action);
    assert_eq!(
        rejection.status, 502,
        "array content must be rejected rather than replaced with a string"
    );
    let rejection_body = rejection.body.expect("rejection body");
    let body_text = String::from_utf8_lossy(&rejection_body);
    assert!(
        body_text.contains("message content is not a string"),
        "non-string content should fail closed, got: {body_text}"
    );
    assert_eq!(
        body.as_deref(),
        Some(original.as_ref()),
        "original multipart body must be kept"
    );
}

#[cfg(feature = "openai-responses")]
#[test]
fn local_tool_modified_verdict_fails_closed() {
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let action = record_local_tool_verdict(
        &mut ctx,
        GuardResult::redact_message(0, "masked result".to_owned(), "pii".to_owned()),
    )
    .unwrap();

    let rejection = as_rejection(action);
    assert_eq!(rejection.status, 502, "modified tool results must fail closed");
    let rejection_body = rejection.body.unwrap();
    let body_text = String::from_utf8_lossy(&rejection_body);
    assert!(
        body_text.contains("sanitized content cannot be safely applied"),
        "the rejection should explain why modified content is not forwarded: {body_text}"
    );
    assert_eq!(
        ctx.filter_results.get("ai_guardrails").unwrap().get("status"),
        Some("redacted"),
        "the tool-result phase should retain the provider verdict even when failing closed"
    );
}

#[tokio::test]
async fn on_request_body_unknown_status_fails_closed() {
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

    let mock_server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "status": "error",
            "content": ""
        })))
        .mount(&mock_server)
        .await;

    let endpoint = format!("{}/v1/checks", mock_server.uri());
    let filter = nemo_filter(&endpoint);
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let mut body = Some(bytes::Bytes::from_static(
        br#"{"messages":[{"role":"user","content":"hello"}]}"#,
    ));

    let result = filter.on_request_body(&mut ctx, &mut body, true).await;
    assert!(
        result.is_err(),
        "an unknown NeMo status should fail closed rather than pass through"
    );
}

#[tokio::test]
async fn on_request_body_not_end_of_stream_continues_without_evaluating() {
    // The endpoint is unreachable in this test environment, so
    // the call to the provider would fail and return a `FilterError`.
    //  A `Continue` here proves evaluation was skipped entirely.
    let filter = nemo_filter("http://nemo:8000/v1/checks");
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
  endpoint: "http://nemo:8000/v1/checks"
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
async fn on_request_body_response_guardrails_reject_streaming() {
    let filter = nemo_filter_response("http://127.0.0.1:9/v1/checks");
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let mut body = Some(bytes::Bytes::from_static(
        br#"{"messages":[{"role":"user","content":"hello"}],"stream":true}"#,
    ));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    let rejection = as_rejection(action);
    assert_eq!(
        rejection.status, 422,
        "streaming must be rejected with response guardrails enabled"
    );
    assert_eq!(
        rejection.body.as_deref(),
        Some(&b"ai_guardrails: streaming responses are not supported while a response guardrail is active"[..]),
        "streaming rejection must return the documented response-guardrail error body",
    );
    assert!(
        !ctx.filter_results.contains_key("ai_guardrails"),
        "the request must be rejected before any provider evaluation"
    );
}

#[tokio::test]
async fn on_request_body_response_guardrails_reject_malformed_json() {
    let filter = nemo_filter_response("http://127.0.0.1:9/v1/checks");
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let mut body = Some(bytes::Bytes::from_static(b"{not-json"));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    let rejection = as_rejection(action);
    assert_eq!(
        rejection.status, 400,
        "malformed request JSON must be rejected as a client error"
    );
    assert!(
        !ctx.filter_results.contains_key("ai_guardrails"),
        "malformed request JSON must be rejected before any provider evaluation"
    );
}

#[tokio::test]
async fn on_request_body_both_guardrails_reject_streaming() {
    let filter = nemo_filter_request_and_response("http://127.0.0.1:9/v1/checks");
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let mut body = Some(bytes::Bytes::from_static(
        br#"{"messages":[{"role":"user","content":"hello"}],"stream":true}"#,
    ));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    let rejection = as_rejection(action);
    assert_eq!(
        rejection.status, 422,
        "streaming must be rejected when request and response guardrails are enabled"
    );
    assert_eq!(
        rejection.body.as_deref(),
        Some(&b"ai_guardrails: streaming responses are not supported while a response guardrail is active"[..]),
        "streaming rejection must return the documented response-guardrail error body",
    );
    assert!(
        !ctx.filter_results.contains_key("ai_guardrails"),
        "the request must be rejected before any provider evaluation"
    );
}

#[tokio::test]
async fn on_request_body_none_continues_without_evaluating() {
    let filter = nemo_filter("http://nemo:8000/v1/checks");
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
    let filter = nemo_filter("http://nemo:8000/v1/checks");
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
    let filter = nemo_filter("http://nemo:8000/v1/checks");
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
    let filter = nemo_filter("http://nemo:8000/v1/checks");
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
    let filter = nemo_filter("http://nemo:8000/v1/checks");
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
async fn on_request_body_empty_messages_array_passes_without_provider_call() {
    use wiremock::MockServer;

    let mock_server = MockServer::start().await;
    let endpoint = format!("{}/v1/checks", mock_server.uri());
    let filter = nemo_filter(&endpoint);
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let mut body = Some(bytes::Bytes::from_static(br#"{"messages":[]}"#));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, praxis_filter::FilterAction::Continue),
        "an empty messages array has no user turns to evaluate"
    );
    assert!(mock_server.received_requests().await.unwrap().is_empty());
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

    let endpoint = format!("{}/v1/checks", mock_server.uri());
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

    let endpoint = format!("{}/v1/checks", mock_server.uri());
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
    assert!(!phase.tool_results, "default tool_results should be false");
}

#[test]
fn phase_config_custom_values() {
    let parsed: PhaseConfig = serde_yaml::from_str(
        "
request: false
response: true
tool_results: true
",
    )
    .unwrap();
    assert!(!parsed.request, "request should be false");
    assert!(parsed.response, "response should be true");
    assert!(parsed.tool_results, "tool_results should be true");
}

#[test]
fn phase_config_omitted_uses_defaults() {
    let parsed: PhaseConfig = serde_yaml::from_str("{}").unwrap();
    assert!(parsed.request, "omitted request should default to true");
    assert!(!parsed.response, "omitted response should default to false");
    assert!(!parsed.tool_results, "omitted tool_results should default to false");
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
  endpoint: "http://nemo:8000/v1/checks"
"#,
    )
    .unwrap();

    assert_eq!(parsed.provider.provider_type, ProviderType::Nemo);
    assert!(
        matches!(&parsed.outbound_chain, praxis_core::config::ChainRef::Inline { filters, .. } if filters.is_empty()),
        "omitted outbound_chain should default to an empty inline chain"
    );
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
  endpoint: "http://nemo:8000/v1/checks"
bogus: true
"#,
    );
    assert!(result.is_err(), "unknown fields should fail with deny_unknown_fields");
}

#[test]
fn guardrails_config_with_phase_overrides() {
    let parsed: AiGuardrailsConfig = serde_yaml::from_str(
        r#"
outbound_chain: test-outbound
provider:
  type: nemo
  endpoint: "http://nemo:8000/v1/checks"
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
// Response body access
// =============================================================================

#[test]
fn response_body_access_none_when_phase_disabled() {
    let filter = nemo_filter("http://nemo:8000/v1/checks");
    assert_eq!(
        filter.response_body_access(),
        praxis_filter::body::BodyAccess::None,
        "response body access should be None when response phase is disabled"
    );
}

#[test]
fn response_body_access_read_write_when_phase_enabled() {
    let filter = nemo_filter_response("http://nemo:8000/v1/checks");
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
    let filter = nemo_filter_response("http://nemo:8000/v1/checks");
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
    let filter = nemo_filter("http://nemo:8000/v1/checks");
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
    let filter = nemo_filter_response("http://nemo:8000/v1/checks");
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
    let filter = nemo_filter_response("http://nemo:8000/v1/checks");
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
async fn on_response_sse_is_rejected() {
    let filter = nemo_filter_response("http://nemo:8000/v1/checks");
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    mark_request_body(&mut ctx);

    let mut resp = crate::test_utils::make_response();
    resp.headers
        .insert("content-type", http::HeaderValue::from_static("text/event-stream"));
    ctx.response_header = Some(&mut resp);
    let rejection = as_rejection(filter.on_response(&mut ctx).await.unwrap());
    ctx.response_header = None;

    assert_eq!(rejection.status, 502, "SSE responses must fail closed");
    assert_eq!(
        rejection.body.as_deref(),
        Some(&b"ai_guardrails: streaming responses are not supported while a response guardrail is active"[..]),
        "SSE rejection must return the documented response-guardrail error body",
    );
    assert_eq!(
        ctx.response_body_mode,
        praxis_filter::BodyMode::Stream,
        "rejected SSE responses must not be buffered"
    );
}

#[tokio::test]
async fn on_response_json_upgrades_to_stream_buffer() {
    let filter = nemo_filter_response("http://nemo:8000/v1/checks");
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    mark_request_body(&mut ctx);

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
    assert_eq!(screening(&ctx), Some(ResponseScreening::Screen));
}

// =============================================================================
// Response phase: only model output is screened
// =============================================================================

#[tokio::test]
async fn on_request_body_marks_response_for_screening() {
    let filter = nemo_filter_response("http://nemo:8000/v1/checks");
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.current_filter_id = Some(0);
    let mut body = Some(bytes::Bytes::from_static(
        br#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#,
    ));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(matches!(action, praxis_filter::FilterAction::Continue));
    assert_eq!(
        screening(&ctx),
        Some(ResponseScreening::Screen),
        "a request with a body must have its response screened"
    );
}

/// Run `on_response` then a buffered `on_response_body` for `resp`, returning
/// the forwarded body. Buffering stands in for another filter that buffers the
/// response, so the skip must hold in `on_response_body` too.
async fn forwarded_response<'a>(
    filter: &dyn HttpFilter,
    ctx: &mut HttpFilterContext<'a>,
    resp: &'a mut praxis_filter::Response,
    original: &'static [u8],
) -> bytes::Bytes {
    ctx.response_header = Some(resp);
    let action = filter.on_response(ctx).await.unwrap();
    ctx.response_header = None;
    assert!(matches!(action, praxis_filter::FilterAction::Continue));
    assert_eq!(
        ctx.response_body_mode,
        praxis_filter::BodyMode::Stream,
        "a skipped response must not be buffered by this filter"
    );
    assert_eq!(screening(ctx), Some(ResponseScreening::Skip));

    buffer_response(ctx);
    let mut body = Some(bytes::Bytes::from_static(original));
    let action = filter.on_response_body(ctx, &mut body, true).unwrap();
    assert!(matches!(action, praxis_filter::FilterAction::Continue));
    body.expect("body is forwarded")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_skips_bodyless_request() {
    let filter = nemo_filter_response("http://127.0.0.1:9/v1/checks");
    let req = crate::test_utils::make_request(http::Method::GET, "/v1/models");
    let mut resp = crate::test_utils::make_response();
    resp.headers
        .insert("content-type", http::HeaderValue::from_static("application/json"));
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.current_filter_id = Some(0);
    let original = br#"{"object":"list","data":[{"id":"qwen"}]}"#;

    let forwarded = forwarded_response(filter.as_ref(), &mut ctx, &mut resp, original).await;
    assert_eq!(
        forwarded.as_ref(),
        original,
        "a model list must reach the client unchanged"
    );
    assert!(
        !ctx.filter_results.contains_key("ai_guardrails"),
        "no verdict is recorded for a skipped response"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_skips_error_status() {
    let filter = nemo_filter_request_and_response("http://127.0.0.1:9/v1/checks");
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut resp = crate::test_utils::make_response();
    resp.status = http::StatusCode::UNAUTHORIZED;
    resp.headers
        .insert("content-type", http::HeaderValue::from_static("text/plain"));
    let mut ctx = crate::test_utils::make_filter_context(&req);
    mark_request_body(&mut ctx);
    let original = b"Unauthorized";

    let forwarded = forwarded_response(filter.as_ref(), &mut ctx, &mut resp, original).await;
    assert_eq!(
        forwarded.as_ref(),
        original,
        "an upstream error must reach the client unchanged"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_error_status_event_stream_is_not_rejected() {
    let filter = nemo_filter_response("http://127.0.0.1:9/v1/checks");
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut resp = crate::test_utils::make_response();
    resp.status = http::StatusCode::SERVICE_UNAVAILABLE;
    resp.headers
        .insert("content-type", http::HeaderValue::from_static("text/event-stream"));
    let mut ctx = crate::test_utils::make_filter_context(&req);
    mark_request_body(&mut ctx);

    let forwarded = forwarded_response(filter.as_ref(), &mut ctx, &mut resp, b"data: overloaded\n\n").await;
    assert_eq!(forwarded.as_ref(), b"data: overloaded\n\n");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_body_stream_mode_skips_evaluation() {
    let filter = nemo_filter_response("http://nemo:8000/v1/checks");
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
    let filter = nemo_filter_response("http://nemo:8000/v1/checks");
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
    let filter = nemo_filter_response("http://nemo:8000/v1/checks");
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
    let filter = nemo_filter_response("http://nemo:8000/v1/checks");
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
    let filter = nemo_filter_response("http://nemo:8000/v1/checks");
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
async fn on_response_body_unknown_status_fails_closed() {
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

    let mock_server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "status": "error",
            "content": ""
        })))
        .mount(&mock_server)
        .await;

    let endpoint = format!("{}/v1/checks", mock_server.uri());
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
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({"status": "passed", "content": "hello"})),
        )
        .mount(&mock_server)
        .await;

    let endpoint = format!("{}/v1/checks", mock_server.uri());
    let filter = nemo_filter_response(&endpoint);
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.response_body_mode = praxis_filter::BodyMode::StreamBuffer { max_bytes: None };
    let mut body = Some(chat_completion_response("hello world"));

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(action, praxis_filter::FilterAction::Continue),
        "passed verdict should continue"
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
            "content": "blocked",
            "rail": "toxicity"
        })))
        .mount(&mock_server)
        .await;

    let endpoint = format!("{}/v1/checks", mock_server.uri());
    let filter = nemo_filter_response(&endpoint);
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    buffer_response(&mut ctx);
    let mut body = Some(chat_completion_response("something toxic"));

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(matches!(action, praxis_filter::FilterAction::Continue));
    let replaced = body.expect("body should be replaced, not cleared");
    assert_natural_json(&replaced);
    assert_eq!(
        ctx.filter_results.get("ai_guardrails").unwrap().get("status"),
        Some("blocked")
    );
    assert_blocked_body_json(&replaced, "toxicity");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_body_modified_rewrites_assistant_content() {
    use wiremock::MockServer;

    let mock_server = MockServer::start().await;
    mount_nemo_checks_response(
        &mock_server,
        serde_json::json!({"status": "modified", "content": "My SSN is [REDACTED]", "rail": "pii"}),
    )
    .await;

    let filter = nemo_filter_response(&format!("{}/v1/checks", mock_server.uri()));
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    buffer_response(&mut ctx);
    let mut body = Some(chat_completion_response("My SSN is 123-45-6789"));

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(matches!(action, praxis_filter::FilterAction::Continue));
    let replaced = body.expect("body should be rewritten, not cleared");
    assert_redacted_assistant_content(&ctx, &replaced, "My SSN is [REDACTED]");
}

/// Response-phase filter that masks choice 0 as `mask`.
fn response_mask_filter(mask: &str) -> AiGuardrailsFilter {
    AiGuardrailsFilter::with_verdict(
        GuardResult::redact_message(0, mask.into(), "pii".into()),
        PhaseConfig {
            request: false,
            response: true,
            tool_results: false,
        },
    )
    .expect("test filter")
}

/// Response-phase filter that masks choice 0 as `[REDACTED]`.
fn response_redact_filter() -> AiGuardrailsFilter {
    response_mask_filter("[REDACTED]")
}

/// Run response redaction and return the forwarded body at its natural size.
///
/// Async so callers can await it on a multi-thread runtime. `on_response_body`
/// uses `block_in_place`, which requires that runtime even though this helper
/// itself does not await.
#[expect(
    clippy::unused_async,
    reason = "callers await this on a multi-thread runtime so on_response_body can block_in_place"
)]
async fn rewritten_response(filter: &AiGuardrailsFilter, original: bytes::Bytes) -> bytes::Bytes {
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    buffer_response(&mut ctx);
    let mut body = Some(original);
    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(action, praxis_filter::FilterAction::Continue),
        "response redaction should continue"
    );
    let replaced = body.expect("body should be replaced, not cleared");
    assert_natural_json(&replaced);
    replaced
}

/// The fail-closed body parses as `evaluation_failed` and omits `absent` text.
fn assert_evaluation_failed(replaced: &bytes::Bytes, absent: &[&str]) {
    assert_natural_json(replaced);
    let json: serde_json::Value = serde_json::from_slice(replaced).expect("fail-closed body must be valid JSON");
    assert_eq!(
        json.pointer("/error/code").and_then(serde_json::Value::as_str),
        Some("evaluation_failed"),
        "failed redaction should be an evaluation_failed error"
    );
    let rendered = String::from_utf8_lossy(replaced);
    for forbidden in absent {
        assert!(!rendered.contains(forbidden), "upstream text must not reach the client");
    }
}

/// Completion whose side fields repeat the text in `message.content`.
fn completion_with_recoverable_original_text() -> bytes::Bytes {
    bytes::Bytes::from(
        serde_json::to_vec(&serde_json::json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "123-45-6789",
                    "reasoning": "the user SSN is 123-45-6789"
                },
                "logprobs": {"content": [
                    {"token": "123", "logprob": -0.1, "bytes": [49, 50, 51]},
                    {"token": "-45-", "logprob": -0.2, "bytes": [45, 52, 53, 45]},
                    {"token": "6789", "logprob": -0.3, "bytes": [54, 55, 56, 57]}
                ]},
                "token_ids": [515_051, 424_242],
                "finish_reason": "stop"
            }]
        }))
        .unwrap(),
    )
}

/// Assistant message with text plus `tool_calls`. The arguments repeat a secret.
fn completion_with_tool_calls() -> bytes::Bytes {
    bytes::Bytes::from(
        serde_json::to_vec(&serde_json::json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Looking up the record",
                    "tool_calls": [{
                        "id": "call_secret",
                        "type": "function",
                        "function": {
                            "name": "lookup_ssn",
                            "arguments": "{\"ssn\":\"123-45-6789\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }))
        .unwrap(),
    )
}

/// Assistant message with text plus a legacy `function_call`.
fn completion_with_function_call() -> bytes::Bytes {
    bytes::Bytes::from(
        serde_json::to_vec(&serde_json::json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Looking up the record",
                    "function_call": {
                        "name": "lookup_ssn",
                        "arguments": "{\"ssn\":\"123-45-6789\"}"
                    }
                },
                "finish_reason": "function_call"
            }]
        }))
        .unwrap(),
    )
}

/// Assistant message whose `content` is text plus an image part.
fn completion_with_array_content() -> bytes::Bytes {
    bytes::Bytes::from(
        serde_json::to_vec(&serde_json::json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "ssn 123-45-6789"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
                    ]
                },
                "finish_reason": "stop"
            }]
        }))
        .unwrap(),
    )
}

/// `"x"` → `"[REDACTED]"` grows the completion. Core reframes the buffered body,
/// so the longer mask is forwarded instead of an `evaluation_failed` error.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_body_expanded_redaction_is_forwarded() {
    let original = chat_completion_response("x");
    let original_len = original.len();
    let replaced = rewritten_response(&response_redact_filter(), original).await;
    assert!(
        replaced.len() > original_len,
        "a longer mask is forwarded at its natural size"
    );
    let json: serde_json::Value = serde_json::from_slice(&replaced).expect("redacted JSON");
    assert_eq!(json_str(&json, "/choices/0/message/content"), Some("[REDACTED]"));
    assert!(json.pointer("/error").is_none(), "growth must stay a completion");
}

/// A short completion whose mask is longer than the original text is forwarded
/// whole. The upstream secret must not remain.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_body_expanded_redaction_forwards_short_body() {
    let original = bytes::Bytes::from_static(br#"{"choices":[{"message":{"role":"assistant","content":"SECRET-9"}}]}"#);
    let original_len = original.len();
    let replaced = rewritten_response(&response_redact_filter(), original).await;
    assert!(replaced.len() > original_len, "the longer mask is not truncated");
    let json: serde_json::Value = serde_json::from_slice(&replaced).expect("redacted JSON");
    assert_eq!(json_str(&json, "/choices/0/message/content"), Some("[REDACTED]"));
    assert!(
        !String::from_utf8_lossy(&replaced).contains("SECRET-9"),
        "upstream secret must not reach the client"
    );
}

/// Allowlisted choice fields survive redaction.
fn assert_allowlisted_choice(json: &serde_json::Value) {
    assert_eq!(
        json_str(json, "/id"),
        Some("chatcmpl-test"),
        "fields outside the redacted choice stay"
    );
    assert_eq!(
        json_str(json, "/choices/0/message/content"),
        Some("[REDACTED]"),
        "assistant content should be the mask"
    );
    assert_eq!(
        json_str(json, "/choices/0/message/role"),
        Some("assistant"),
        "message role is on the allowlist"
    );
    assert_eq!(
        json.pointer("/choices/0/index").and_then(serde_json::Value::as_u64),
        Some(0),
        "choice index is on the allowlist"
    );
    assert_eq!(
        json_str(json, "/choices/0/finish_reason"),
        Some("stop"),
        "finish_reason is on the allowlist"
    );
}

/// Side fields that quote the original tokens are absent after redaction.
fn assert_redacted_choice_drops_original(json: &serde_json::Value, replaced: &bytes::Bytes) {
    let choice = json
        .pointer("/choices/0")
        .and_then(serde_json::Value::as_object)
        .expect("choice object");
    assert!(
        choice.get("logprobs").is_none() && choice.get("token_ids").is_none(),
        "logprobs and token_ids must not remain on a redacted choice"
    );
    assert!(
        json.pointer("/choices/0/message/reasoning").is_none(),
        "message.reasoning must not remain on a redacted choice"
    );
    let rendered = String::from_utf8_lossy(replaced);
    assert!(
        !rendered.contains("123-45-6789"),
        "original digits must not reach the client"
    );
    assert!(!rendered.contains("6789"), "logprob tokens must not reach the client");
    assert!(
        !rendered.contains("515051"),
        "return_token_ids values must not reach the client"
    );
}

/// A redacted choice keeps only `index`, `finish_reason`, and `message.role`
/// plus the masked `content`. Side fields that quote the original tokens are
/// dropped.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_body_modified_rebuilds_choice_from_allowlist() {
    let replaced = rewritten_response(&response_redact_filter(), completion_with_recoverable_original_text()).await;
    let json: serde_json::Value = serde_json::from_slice(&replaced).expect("redacted body must be valid JSON");
    assert_allowlisted_choice(&json);
    assert_redacted_choice_drops_original(&json, &replaced);
}

/// A configured redaction that cannot be applied must replace the response
/// with an `evaluation_failed` document and must not forward the original text.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_body_redaction_failure_replaces_body() {
    let replaced = rewritten_response(&response_redact_filter(), completion_with_array_content()).await;
    assert_evaluation_failed(&replaced, &["123-45-6789", "image_url"]);
}

/// Text plus `tool_calls` fails closed. Copying the calls could leak the masked
/// turn, and dropping them would leave `finish_reason: "tool_calls"` with no calls.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_body_tool_calls_fail_closed() {
    let replaced = rewritten_response(&response_redact_filter(), completion_with_tool_calls()).await;
    assert_evaluation_failed(&replaced, &["123-45-6789", "lookup_ssn", "call_secret", "tool_calls"]);
}

/// A legacy `function_call` fails closed for the same reason as `tool_calls`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_body_function_call_fails_closed() {
    let replaced = rewritten_response(&response_redact_filter(), completion_with_function_call()).await;
    assert_evaluation_failed(&replaced, &["123-45-6789", "lookup_ssn", "function_call"]);
}

/// `on_response` drops a stale `Content-Length` so core can reframe the buffered body.
#[tokio::test]
async fn on_response_removes_stale_content_length() {
    let filter = nemo_filter_response("http://127.0.0.1:9/v1/checks");
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    mark_request_body(&mut ctx);
    let mut response = crate::test_utils::make_response();
    response
        .headers
        .insert(http::header::CONTENT_LENGTH, http::HeaderValue::from_static("128"));
    ctx.response_header = Some(&mut response);

    let action = filter.on_response(&mut ctx).await.expect("on_response");
    assert!(matches!(action, praxis_filter::FilterAction::Continue));
    let removed = ctx
        .response_header
        .as_ref()
        .is_some_and(|header| !header.headers.contains_key(http::header::CONTENT_LENGTH));
    assert!(removed, "stale Content-Length must be removed");
    assert!(ctx.response_headers_modified, "header removal must be visible to core");
    ctx.response_header = None;
}

/// A longer mask is forwarded as a completion. Core reframes the body.
fn assert_forwarded_mask(ctx: &HttpFilterContext<'_>, replaced: &bytes::Bytes, original_len: usize) {
    assert!(
        replaced.len() > original_len,
        "a longer mask is forwarded at its natural size"
    );
    assert_natural_json(replaced);
    let json: serde_json::Value = serde_json::from_slice(replaced).expect("redacted JSON");
    assert_eq!(
        json_str(&json, "/choices/0/message/content"),
        Some("<PERSON>"),
        "the mask is forwarded"
    );
    assert!(
        json.pointer("/error/code").is_none(),
        "a longer mask must stay a completion"
    );
    assert_eq!(
        ctx.filter_results
            .get("ai_guardrails")
            .and_then(|result| result.get("status")),
        Some("redacted")
    );
}

/// A short name replaced by a longer mask grows the completion. The masked body
/// is forwarded instead of an `evaluation_failed` error.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_body_growth_is_forwarded() {
    let filter = response_mask_filter("<PERSON>");
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    buffer_response(&mut ctx);
    let original = chat_completion_response("Bob");
    let original_len = original.len();
    let mut body = Some(original);

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(action, praxis_filter::FilterAction::Continue),
        "an unbounded rewrite should continue"
    );
    assert_forwarded_mask(&ctx, &body.expect("body should be rewritten"), original_len);
}

/// A blocked body is the error document itself, not padded out to the upstream body.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_body_block_is_not_padded() {
    let filter = AiGuardrailsFilter::with_verdict(
        GuardResult::Block {
            reason: "toxicity".into(),
        },
        PhaseConfig {
            request: false,
            response: true,
            tool_results: false,
        },
    )
    .expect("test filter");
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    buffer_response(&mut ctx);
    let original = chat_completion_response(&"x".repeat(400));
    let original_len = original.len();
    let mut body = Some(original);

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(matches!(action, praxis_filter::FilterAction::Continue));
    let replaced = body.expect("blocked body");
    assert!(
        replaced.len() < original_len,
        "the error document is not padded out to the upstream body"
    );
    assert_natural_json(&replaced);
    assert_blocked_body_json(&replaced, "toxicity");
}

/// A redaction that cannot be applied returns the error document at its natural size.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_body_redaction_failure_is_not_padded() {
    let filter = response_redact_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    buffer_response(&mut ctx);
    let original = completion_with_array_content();
    let original_len = original.len();
    let mut body = Some(original);

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(matches!(action, praxis_filter::FilterAction::Continue));
    let replaced = body.expect("fail-closed body");
    assert_ne!(
        replaced.len(),
        original_len,
        "the error document is not fitted to the upstream body"
    );
    assert_natural_json(&replaced);
    assert_evaluation_failed(&replaced, &["123-45-6789", "image_url"]);
}
