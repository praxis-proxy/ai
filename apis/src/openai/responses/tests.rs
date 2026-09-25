// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Unit tests for the `openai_responses_format` filter.

use bytes::Bytes;

use super::*;

// -----------------------------------------------------------------------------
// Config Parsing
// -----------------------------------------------------------------------------

#[test]
fn default_config_parses() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("{}").unwrap();
    let filter = ResponsesFormatFilter::from_config(&yaml).unwrap();
    assert_eq!(
        filter.name(),
        "openai_responses_format",
        "filter name should be openai_responses_format"
    );
}

#[test]
fn full_config_parses() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
on_invalid: reject
headers:
  format: x-custom-format
  model: x-custom-model
  stream: x-custom-stream
"#,
    )
    .unwrap();

    let filter = ResponsesFormatFilter::from_config(&yaml).unwrap();
    assert_eq!(
        filter.name(),
        "openai_responses_format",
        "filter name should be openai_responses_format"
    );
}

#[test]
fn on_invalid_continue_parses() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("on_invalid: continue").unwrap();
    let filter = ResponsesFormatFilter::from_config(&yaml).unwrap();
    assert_eq!(filter.name(), "openai_responses_format", "continue mode should parse");
}

#[test]
fn on_invalid_reject_parses() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("on_invalid: reject").unwrap();
    let filter = ResponsesFormatFilter::from_config(&yaml).unwrap();
    assert_eq!(filter.name(), "openai_responses_format", "reject mode should parse");
}

#[test]
fn deny_unknown_fields_rejects_typo() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("on_invlid: continue").unwrap();
    let result = ResponsesFormatFilter::from_config(&yaml);
    assert!(result.is_err(), "typo in config field should be rejected");
}

#[test]
fn legacy_max_body_bytes_rejected() {
    // The classifier no longer carries a body-size knob; the raw cap is
    // governed by the pipeline's body_limits. The removed field must be
    // rejected rather than silently ignored.
    let yaml: serde_yaml::Value = serde_yaml::from_str("max_body_bytes: 65536").unwrap();
    let result = ResponsesFormatFilter::from_config(&yaml);
    assert!(result.is_err(), "legacy max_body_bytes should be rejected");
}

#[test]
fn invalid_header_name_rejected() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
headers:
  format: "invalid header name with spaces"
"#,
    )
    .unwrap();
    let result = ResponsesFormatFilter::from_config(&yaml);
    assert!(result.is_err(), "header name with spaces should be rejected");
}

#[test]
fn empty_header_name_rejected() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
headers:
  model: ""
"#,
    )
    .unwrap();
    let result = ResponsesFormatFilter::from_config(&yaml);
    assert!(result.is_err(), "empty header name should be rejected");
}

#[test]
fn null_header_disables_promotion() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
headers:
  format: null
  model: null
  stream: null
"#,
    )
    .unwrap();
    let filter = ResponsesFormatFilter::from_config(&yaml).unwrap();
    assert_eq!(
        filter.name(),
        "openai_responses_format",
        "null headers should disable promotion"
    );
}

// -----------------------------------------------------------------------------
// Body Access
// -----------------------------------------------------------------------------

#[test]
fn body_access_is_read_only() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("{}").unwrap();
    let filter = ResponsesFormatFilter::from_config(&yaml).unwrap();
    assert_eq!(
        filter.request_body_access(),
        BodyAccess::ReadOnly,
        "classifier must not mutate the body"
    );
}

#[test]
fn body_mode_is_stream_buffer() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("{}").unwrap();
    let filter = ResponsesFormatFilter::from_config(&yaml).unwrap();
    match filter.request_body_mode() {
        BodyMode::StreamBuffer { max_bytes } => {
            assert_eq!(
                max_bytes,
                Some(67_108_864),
                "StreamBuffer accepts up to the absolute 64 MiB ceiling; body_limits governs the raw cap"
            );
        },
        other => panic!("expected StreamBuffer, got {other:?}"),
    }
}

// -----------------------------------------------------------------------------
// Handle Invalid Format
// -----------------------------------------------------------------------------

#[test]
fn invalid_json_continue_returns_none() {
    let cfg: ResponsesFormatConfig = serde_yaml::from_str("on_invalid: continue").unwrap();
    let result = handle_invalid_format(AiRequestFormat::InvalidJson, &cfg);
    assert!(result.is_none(), "continue mode should return None for invalid JSON");
}

#[test]
fn invalid_json_reject_returns_reject() {
    let cfg: ResponsesFormatConfig = serde_yaml::from_str("on_invalid: reject").unwrap();
    let result = handle_invalid_format(AiRequestFormat::InvalidJson, &cfg);
    assert!(result.is_some(), "reject mode should return Reject for invalid JSON");
}

#[test]
fn non_json_continue_returns_none() {
    let cfg: ResponsesFormatConfig = serde_yaml::from_str("on_invalid: continue").unwrap();
    let result = handle_invalid_format(AiRequestFormat::NonJson, &cfg);
    assert!(result.is_none(), "continue mode should return None for non-JSON");
}

#[test]
fn non_json_reject_returns_reject() {
    let cfg: ResponsesFormatConfig = serde_yaml::from_str("on_invalid: reject").unwrap();
    let result = handle_invalid_format(AiRequestFormat::NonJson, &cfg);
    assert!(result.is_some(), "reject mode should return Reject for non-JSON");
}

#[test]
fn openai_responses_format_not_rejected() {
    let cfg: ResponsesFormatConfig = serde_yaml::from_str("on_invalid: reject").unwrap();
    let result = handle_invalid_format(AiRequestFormat::Responses, &cfg);
    assert!(result.is_none(), "responses format should not be rejected");
}

#[test]
fn chat_completions_not_rejected() {
    let cfg: ResponsesFormatConfig = serde_yaml::from_str("on_invalid: reject").unwrap();
    let result = handle_invalid_format(AiRequestFormat::ChatCompletions, &cfg);
    assert!(
        result.is_none(),
        "openai_chat_completions format should not be rejected"
    );
}

#[test]
fn unknown_json_rejected_in_reject_mode() {
    let cfg: ResponsesFormatConfig = serde_yaml::from_str("on_invalid: reject").unwrap();
    let result = handle_invalid_format(AiRequestFormat::UnknownJson, &cfg);
    assert!(result.is_some(), "unknown JSON should be rejected in reject mode");
}

#[test]
fn unknown_json_continues_in_continue_mode() {
    let cfg: ResponsesFormatConfig = serde_yaml::from_str("on_invalid: continue").unwrap();
    let result = handle_invalid_format(AiRequestFormat::UnknownJson, &cfg);
    assert!(result.is_none(), "unknown JSON should continue in continue mode");
}

// -----------------------------------------------------------------------------
// Full Reject-Path Tests (on_request_body)
// -----------------------------------------------------------------------------

#[tokio::test]
async fn on_request_body_rejects_unknown_json() {
    let action = run_filter_raw_with_method(
        "on_invalid: reject",
        r#"{"prompt":"hello"}"#,
        http::Method::POST,
        "/v1/chat/completions",
    )
    .await;
    assert!(
        matches!(action, FilterAction::Reject(_)),
        "unknown JSON on a non-create path should be rejected; the endpoint is not \
         authoritative there (on POST /v1/responses the same body is promoted to \
         responses -- see post_v1_responses_create_without_discriminator_*)"
    );
}

#[tokio::test]
async fn on_request_body_rejects_invalid_json() {
    let action = run_filter_raw("on_invalid: reject", "not json {{{").await;
    assert!(
        matches!(action, FilterAction::Reject(_)),
        "invalid JSON should be rejected"
    );
}

// -----------------------------------------------------------------------------
// Body Parsing Edge Cases
// -----------------------------------------------------------------------------

#[tokio::test]
async fn partial_body_before_eos_continues() {
    let filter = make_filter("{}");
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");

    let req: &'static praxis_filter::Request = Box::leak(Box::new(req));
    let mut ctx = crate::test_utils::make_filter_context(req);
    let mut body = Some(Bytes::from(r#"{"model":"gpt-4.1","inp"#));

    let action = filter.on_request_body(&mut ctx, &mut body, false).await.unwrap();

    assert!(
        matches!(action, FilterAction::Continue),
        "non-EOS body should return Continue"
    );
    assert!(
        ctx.extra_request_headers.is_empty(),
        "no headers should be promoted before EOS"
    );
}

#[tokio::test]
async fn none_body_at_eos_continues() {
    let filter = make_filter("{}");
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");

    let req: &'static praxis_filter::Request = Box::leak(Box::new(req));
    let mut ctx = crate::test_utils::make_filter_context(req);
    let mut body: Option<Bytes> = None;

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(
        !matches!(action, FilterAction::Reject(_)),
        "None body with default on_invalid:continue should not reject"
    );
}

// -----------------------------------------------------------------------------
// Promotion Tests (on_request_body)
// -----------------------------------------------------------------------------

#[tokio::test]
async fn promotes_headers_for_full_responses_request() {
    let ctx = run_filter("{}", FULL_RESPONSES_BODY).await;
    let headers = collect_headers(&ctx);

    assert_eq!(
        headers.get("x-praxis-ai-format"),
        Some(&"openai_responses"),
        "format header"
    );
    assert_eq!(headers.get("x-praxis-ai-model"), Some(&"gpt-4.1"), "model header");
    assert_eq!(headers.get("x-praxis-ai-stream"), Some(&"true"), "stream header");
    assert!(
        !headers.contains_key("x-praxis-ai-background"),
        "background should not have a default header"
    );
    assert_eq!(headers.get("x-praxis-responses-mode"), Some(&"stateful"), "mode header");
}

#[tokio::test]
async fn promotes_metadata_for_full_responses_request() {
    let ctx = run_filter("{}", FULL_RESPONSES_BODY).await;

    assert_eq!(
        ctx.filter_metadata
            .get("openai_responses_format.format")
            .map(String::as_str),
        Some("openai_responses")
    );
    assert_eq!(
        ctx.filter_metadata
            .get("openai_responses_format.model")
            .map(String::as_str),
        Some("gpt-4.1")
    );
    assert_eq!(
        ctx.filter_metadata
            .get("openai_responses_format.stream")
            .map(String::as_str),
        Some("true")
    );
    assert_eq!(
        ctx.filter_metadata
            .get("openai_responses_format.store")
            .map(String::as_str),
        Some("false")
    );
    assert_eq!(
        ctx.filter_metadata
            .get("openai_responses_format.background")
            .map(String::as_str),
        Some("false")
    );
    assert_eq!(
        ctx.filter_metadata
            .get("openai_responses_format.has_previous_response_id")
            .map(String::as_str),
        Some("true")
    );
    assert_eq!(
        ctx.filter_metadata
            .get("openai_responses_format.has_conversation")
            .map(String::as_str),
        Some("true")
    );
    assert_eq!(
        ctx.filter_metadata
            .get("openai_responses_format.has_tools")
            .map(String::as_str),
        Some("true")
    );
    assert_eq!(
        ctx.filter_metadata
            .get("openai_responses_format.has_prompt_id")
            .map(String::as_str),
        Some("true")
    );
    assert_eq!(
        ctx.filter_metadata
            .get("openai_responses_format.mode")
            .map(String::as_str),
        Some("stateful")
    );
}

#[tokio::test]
async fn promotes_filter_results_for_full_responses_request() {
    let ctx = run_filter("{}", FULL_RESPONSES_BODY).await;
    let results = ctx.filter_results.get("openai_responses_format").unwrap();

    assert_eq!(results.get("format"), Some("openai_responses"));
    assert_eq!(results.get("model"), Some("gpt-4.1"));
    assert_eq!(results.get("stream"), Some("true"));
    assert_eq!(results.get("store"), Some("false"));
    assert_eq!(results.get("background"), Some("false"));
    assert_eq!(results.get("has_previous_response_id"), Some("true"));
    assert_eq!(results.get("has_conversation"), Some("true"));
    assert_eq!(results.get("has_tools"), Some("true"));
    assert_eq!(results.get("has_prompt_id"), Some("true"));
    assert_eq!(results.get("mode"), Some("stateful"));
}

#[tokio::test]
async fn missing_optional_facts_not_promoted() {
    let ctx = run_filter("{}", r#"{"input":"test"}"#).await;
    let headers = collect_headers(&ctx);

    assert_eq!(
        headers.get("x-praxis-ai-format"),
        Some(&"openai_responses"),
        "format still promoted"
    );
    assert!(!headers.contains_key("x-praxis-ai-model"), "model header absent");
    assert!(!headers.contains_key("x-praxis-ai-stream"), "stream header absent");
    assert!(
        !ctx.filter_metadata.contains_key("openai_responses_format.model"),
        "model metadata absent"
    );
    assert!(
        !ctx.filter_metadata.contains_key("openai_responses_format.stream"),
        "stream metadata absent"
    );
    assert!(
        !ctx.filter_metadata.contains_key("openai_responses_format.store"),
        "store metadata absent"
    );
    assert!(
        !ctx.filter_metadata.contains_key("openai_responses_format.background"),
        "background metadata absent"
    );
    assert!(
        !ctx.filter_metadata
            .contains_key("openai_responses_format.has_previous_response_id"),
        "prev_id metadata absent"
    );
    assert!(
        !ctx.filter_metadata
            .contains_key("openai_responses_format.has_conversation"),
        "conversation metadata absent"
    );
    assert!(
        !ctx.filter_metadata.contains_key("openai_responses_format.has_tools"),
        "tools metadata absent"
    );
    assert!(
        !ctx.filter_metadata
            .contains_key("openai_responses_format.has_prompt_id"),
        "prompt_id metadata absent"
    );
    assert_eq!(
        ctx.filter_metadata
            .get("openai_responses_format.mode")
            .map(String::as_str),
        Some("stateful"),
        "mode should be stateful when store is omitted (defaults to true)"
    );
}

#[tokio::test]
async fn oversized_model_not_promoted_to_header_or_results_or_metadata() {
    let long_model = "x".repeat(300);
    let body_str = format!(r#"{{"model":"{long_model}","input":"test"}}"#);
    let ctx = run_filter("{}", &body_str).await;
    let headers = collect_headers(&ctx);

    assert!(
        !headers.contains_key("x-praxis-ai-model"),
        "oversized model not in header"
    );
    let results = ctx.filter_results.get("openai_responses_format").unwrap();
    assert!(results.get("model").is_none(), "oversized model not in results");
    assert!(
        !ctx.filter_metadata.contains_key("openai_responses_format.model"),
        "oversized model not in metadata"
    );
}

#[tokio::test]
async fn control_char_model_not_promoted() {
    let ctx = run_filter("{}", "{\"model\":\"bad\\nmodel\",\"input\":\"test\"}").await;
    let headers = collect_headers(&ctx);

    assert!(
        !headers.contains_key("x-praxis-ai-model"),
        "control-char model not in header"
    );
}

#[tokio::test]
async fn custom_headers_emitted_at_runtime() {
    let cfg = "headers:\n  format: x-custom-fmt\n  model: x-custom-mdl\n  stream: x-custom-strm";
    let ctx = run_filter(cfg, r#"{"model":"gpt-4.1","input":"test","stream":true}"#).await;
    let headers = collect_headers(&ctx);

    assert_eq!(headers.get("x-custom-fmt"), Some(&"openai_responses"), "custom format");
    assert_eq!(headers.get("x-custom-mdl"), Some(&"gpt-4.1"), "custom model");
    assert_eq!(headers.get("x-custom-strm"), Some(&"true"), "custom stream");
    assert!(
        !headers.contains_key("x-praxis-ai-format"),
        "default not emitted when overridden"
    );
}

#[tokio::test]
async fn null_headers_suppress_emission() {
    let cfg = "headers:\n  format: null\n  model: null\n  stream: null\n  mode: null";
    let ctx = run_filter(cfg, r#"{"model":"gpt-4.1","input":"test","stream":true}"#).await;

    assert!(
        ctx.extra_request_headers.is_empty(),
        "null headers suppress all emission"
    );
    assert_eq!(
        ctx.filter_metadata
            .get("openai_responses_format.format")
            .map(String::as_str),
        Some("openai_responses"),
        "metadata still written with null headers"
    );
}

// -----------------------------------------------------------------------------
// Path-Based Classification (method + path in on_request_body)
// -----------------------------------------------------------------------------

#[tokio::test]
async fn get_v1_responses_with_id_classifies_as_responses() {
    let ctx = run_filter_with_method("{}", "", http::Method::GET, "/v1/responses/resp_abc123").await;

    assert_eq!(
        ctx.filter_metadata
            .get("openai_responses_format.format")
            .map(String::as_str),
        Some("openai_responses"),
        "GET /v1/responses/{{id}} should classify as responses"
    );

    assert!(
        ctx.extensions.get::<ErrorResponseFormatterHandle>().is_some(),
        "openai error response formatter should be installed for responses path"
    );
}

#[tokio::test]
async fn get_v1_responses_input_items_classifies_as_responses() {
    let ctx = run_filter_with_method("{}", "", http::Method::GET, "/v1/responses/resp_abc123/input_items").await;

    assert_eq!(
        ctx.filter_metadata
            .get("openai_responses_format.format")
            .map(String::as_str),
        Some("openai_responses"),
        "GET /v1/responses/{{id}}/input_items should classify as responses"
    );
}

#[tokio::test]
async fn delete_v1_responses_with_id_classifies_as_responses() {
    let ctx = run_filter_with_method("{}", "", http::Method::DELETE, "/v1/responses/resp_abc123").await;

    assert_eq!(
        ctx.filter_metadata
            .get("openai_responses_format.format")
            .map(String::as_str),
        Some("openai_responses"),
        "DELETE /v1/responses/{{id}} should classify as responses"
    );
}

#[tokio::test]
async fn post_v1_responses_cancel_classifies_as_responses() {
    let ctx = run_filter_with_method("{}", "", http::Method::POST, "/v1/responses/resp_abc123/cancel").await;

    assert_eq!(
        ctx.filter_metadata
            .get("openai_responses_format.format")
            .map(String::as_str),
        Some("openai_responses"),
        "POST /v1/responses/{{id}}/cancel should classify as responses"
    );
    let results = ctx.filter_results.get("openai_responses_format").unwrap();
    assert_eq!(results.get("format"), Some("openai_responses"), "filter result format");
}

#[tokio::test]
async fn post_v1_responses_input_tokens_classifies_as_responses() {
    let ctx = run_filter_with_method("{}", "", http::Method::POST, "/v1/responses/input_tokens").await;

    assert_eq!(
        ctx.filter_metadata
            .get("openai_responses_format.format")
            .map(String::as_str),
        Some("openai_responses"),
        "POST /v1/responses/input_tokens should classify as responses"
    );
}

#[tokio::test]
async fn post_v1_responses_compact_classifies_as_responses() {
    let ctx = run_filter_with_method("{}", "", http::Method::POST, "/v1/responses/compact").await;

    assert_eq!(
        ctx.filter_metadata
            .get("openai_responses_format.format")
            .map(String::as_str),
        Some("openai_responses"),
        "POST /v1/responses/compact should classify as responses"
    );
}

#[tokio::test]
async fn get_path_match_promotes_filter_results() {
    let ctx = run_filter_with_method("{}", "", http::Method::GET, "/v1/responses/resp_abc123").await;
    let results = ctx.filter_results.get("openai_responses_format").unwrap();

    assert_eq!(results.get("format"), Some("openai_responses"), "filter result format");
    assert_eq!(results.get("model"), None, "no model from path-only classification");
    assert_eq!(results.get("stream"), None, "no stream from path-only classification");
}

#[tokio::test]
async fn delete_path_match_promotes_filter_results() {
    let ctx = run_filter_with_method("{}", "", http::Method::DELETE, "/v1/responses/resp_abc123").await;
    let results = ctx.filter_results.get("openai_responses_format").unwrap();

    assert_eq!(results.get("format"), Some("openai_responses"), "filter result format");
}

#[tokio::test]
async fn get_path_match_promotes_format_header() {
    let ctx = run_filter_with_method("{}", "", http::Method::GET, "/v1/responses/resp_abc").await;
    let headers = collect_headers(&ctx);

    assert_eq!(
        headers.get("x-praxis-ai-format"),
        Some(&"openai_responses"),
        "GET path match should promote format header"
    );
    assert!(
        !headers.contains_key("x-praxis-ai-model"),
        "no model for path-only classification"
    );
    assert!(
        !headers.contains_key("x-praxis-ai-stream"),
        "no stream for path-only classification"
    );
}

#[tokio::test]
async fn get_path_match_no_body_facts() {
    let ctx = run_filter_with_method("{}", "", http::Method::GET, "/v1/responses/resp_abc").await;

    assert!(
        !ctx.filter_metadata.contains_key("openai_responses_format.model"),
        "no model from path-only classification"
    );
    assert!(
        !ctx.filter_metadata.contains_key("openai_responses_format.stream"),
        "no stream from path-only classification"
    );
    assert!(
        !ctx.filter_metadata.contains_key("openai_responses_format.store"),
        "no store from path-only classification"
    );
    assert!(
        !ctx.filter_metadata.contains_key("openai_responses_format.background"),
        "no background from path-only classification"
    );
}

#[tokio::test]
async fn put_unrelated_path_classifies_body_normally() {
    let ctx = run_filter_with_method(
        "{}",
        r#"{"model":"gpt-4","messages":[]}"#,
        http::Method::PUT,
        "/v1/chat/completions",
    )
    .await;

    assert_eq!(
        ctx.filter_metadata
            .get("openai_responses_format.format")
            .map(String::as_str),
        Some("openai_chat_completions"),
        "non-path method should classify using request body"
    );
}

#[tokio::test]
async fn post_v1_responses_classifies_body_normally() {
    let ctx = run_filter_with_method(
        "{}",
        r#"{"model":"gpt-4.1","input":"test"}"#,
        http::Method::POST,
        "/v1/responses",
    )
    .await;

    assert_eq!(
        ctx.filter_metadata
            .get("openai_responses_format.format")
            .map(String::as_str),
        Some("openai_responses"),
        "POST should classify via body, not path"
    );
    assert_eq!(
        ctx.filter_metadata
            .get("openai_responses_format.model")
            .map(String::as_str),
        Some("gpt-4.1"),
        "POST should extract model from body"
    );
}

/// A valid Responses create body that omits the
/// discriminator fields (`input`, `prompt` object, `previous_response_id`,
/// `conversation`) must still classify as `openai_responses` because the
/// endpoint is authoritative — not fall through to `unknown_json`.
#[tokio::test]
async fn post_v1_responses_create_without_discriminator_classifies_as_responses() {
    let ctx = run_filter_with_method("{}", r#"{"model":"gpt-5"}"#, http::Method::POST, "/v1/responses").await;

    assert_eq!(
        ctx.filter_metadata
            .get("openai_responses_format.format")
            .map(String::as_str),
        Some("openai_responses"),
        "create body without discriminator fields should classify as responses"
    );
    assert_eq!(
        ctx.filter_metadata
            .get("openai_responses_format.model")
            .map(String::as_str),
        Some("gpt-5"),
        "model should still be extracted from the create body"
    );
}

/// The create endpoint is authoritative even when the body still carries
/// facts: body-derived fields must survive the forced format assignment.
#[tokio::test]
async fn post_v1_responses_create_preserves_body_facts() {
    let ctx = run_filter_with_method(
        "{}",
        r#"{"model":"gpt-5","stream":true,"store":false}"#,
        http::Method::POST,
        "/v1/responses",
    )
    .await;

    assert_eq!(
        ctx.filter_metadata
            .get("openai_responses_format.format")
            .map(String::as_str),
        Some("openai_responses"),
        "create endpoint should classify as responses"
    );
    assert_eq!(
        ctx.filter_metadata
            .get("openai_responses_format.stream")
            .map(String::as_str),
        Some("true"),
        "stream fact should be preserved"
    );
    assert_eq!(
        ctx.filter_metadata
            .get("openai_responses_format.store")
            .map(String::as_str),
        Some("false"),
        "store fact should be preserved"
    );
}

/// Endpoint authority must hold under `on_invalid: reject`: a valid
/// no-discriminator create body is a Responses request, so the reject path
/// must never see it.
#[tokio::test]
async fn post_v1_responses_create_without_discriminator_not_rejected() {
    let action = run_filter_raw_with_method(
        "on_invalid: reject",
        r#"{"model":"gpt-5"}"#,
        http::Method::POST,
        "/v1/responses",
    )
    .await;

    assert!(
        matches!(action, FilterAction::Release),
        "valid no-discriminator body on the create endpoint must not be rejected"
    );
}

/// A genuinely broken body on the create endpoint must not be masked as a
/// valid Responses request: parse failures are preserved so they can be
/// rejected when `on_invalid` is configured to reject.
#[tokio::test]
async fn post_v1_responses_create_preserves_invalid_json() {
    let action = run_filter_raw("on_invalid: reject", r#"{"model":"gpt-5""#).await;

    assert!(
        matches!(action, FilterAction::Reject(_)),
        "invalid JSON on the create endpoint should still be rejected, not forced to responses"
    );
}

/// Promote only the format fact for a Responses `WebSocket` handshake.
#[tokio::test]
async fn responses_websocket_handshake_promotes_only_format() {
    let ctx = run_filter_with_request_headers(
        "{}",
        "",
        http::Method::GET,
        "/v1/responses",
        &[
            (http::header::CONNECTION, "keep-alive, Upgrade"),
            (http::header::UPGRADE, "websocket"),
        ],
    )
    .await;
    let headers = collect_headers(&ctx);
    let results = ctx.filter_results.get("openai_responses_format").unwrap();

    assert_eq!(
        headers.get("x-praxis-ai-format"),
        Some(&"openai_responses"),
        "the handshake should promote the Responses format header"
    );
    assert_eq!(
        ctx.filter_metadata
            .get("openai_responses_format.format")
            .map(String::as_str),
        Some("openai_responses"),
        "the handshake should write the Responses format metadata"
    );
    assert_eq!(
        results.get("format"),
        Some("openai_responses"),
        "the handshake should publish the Responses format result"
    );

    for header in ["x-praxis-ai-model", "x-praxis-ai-stream", "x-praxis-responses-mode"] {
        assert!(!headers.contains_key(header), "handshake should not promote {header}");
    }
    for fact in [
        "model",
        "stream",
        "store",
        "background",
        "max_output_tokens",
        "has_previous_response_id",
        "has_conversation",
        "has_tools",
        "has_prompt_id",
        "mode",
    ] {
        assert!(
            !ctx.filter_metadata
                .contains_key(&format!("openai_responses_format.{fact}")),
            "handshake should not write {fact} metadata"
        );
        assert!(
            results.get(fact).is_none(),
            "handshake should not promote {fact} result"
        );
    }
}

/// Leave an ordinary bodyless Responses GET on the normal body path.
#[tokio::test]
async fn get_responses_without_websocket_headers_classifies_body_normally() {
    let ctx = run_filter_with_method("{}", "", http::Method::GET, "/v1/responses").await;

    assert_eq!(
        ctx.filter_metadata
            .get("openai_responses_format.format")
            .map(String::as_str),
        Some("non_json"),
        "an ordinary GET list request must not be promoted as a WebSocket handshake"
    );

    assert!(
        ctx.extensions.get::<ErrorResponseFormatterHandle>().is_none(),
        "openai error response formatter should not be installed for non-json"
    );
}

// -----------------------------------------------------------------------------
// Mode Computation
// -----------------------------------------------------------------------------

#[tokio::test]
async fn mode_stateless_when_store_false_no_stateful_markers() {
    let ctx = run_filter("{}", r#"{"input":"test","store":false}"#).await;
    let results = ctx.filter_results.get("openai_responses_format").unwrap();

    assert_eq!(results.get("mode"), Some("stateless"));
    assert_eq!(
        ctx.filter_metadata
            .get("openai_responses_format.mode")
            .map(String::as_str),
        Some("stateless")
    );
    let headers = collect_headers(&ctx);
    assert_eq!(headers.get("x-praxis-responses-mode"), Some(&"stateless"));
}

#[tokio::test]
async fn mode_stateful_when_store_omitted() {
    let ctx = run_filter("{}", r#"{"input":"test"}"#).await;
    let results = ctx.filter_results.get("openai_responses_format").unwrap();

    assert_eq!(
        results.get("mode"),
        Some("stateful"),
        "omitted store defaults to true (stateful)"
    );
}

#[tokio::test]
async fn mode_stateful_when_store_true() {
    let ctx = run_filter("{}", r#"{"input":"test","store":true}"#).await;
    let results = ctx.filter_results.get("openai_responses_format").unwrap();

    assert_eq!(results.get("mode"), Some("stateful"));
}

#[tokio::test]
async fn mode_stateful_when_previous_response_id() {
    let ctx = run_filter(
        "{}",
        r#"{"input":"test","store":false,"previous_response_id":"resp_1"}"#,
    )
    .await;
    let results = ctx.filter_results.get("openai_responses_format").unwrap();

    assert_eq!(results.get("mode"), Some("stateful"));
}

#[tokio::test]
async fn mode_stateful_when_tools_present() {
    let ctx = run_filter("{}", r#"{"input":"test","store":false,"tools":[{"type":"function"}]}"#).await;
    let results = ctx.filter_results.get("openai_responses_format").unwrap();

    assert_eq!(results.get("mode"), Some("stateful"));
}

#[tokio::test]
async fn background_true_is_rejected_even_when_invalid_formats_continue() {
    let action = run_filter_raw(
        "on_invalid: continue",
        r#"{"input":"test","store":false,"background":true}"#,
    )
    .await;
    let FilterAction::Reject(rejection) = action else {
        panic!("background=true should be rejected before routing");
    };
    assert_eq!(rejection.status, 400);
    let body: serde_json::Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert_eq!(body["error"]["code"], "invalid_request_error");
    assert_eq!(body["error"]["message"], "background mode is not supported");
}

#[tokio::test]
async fn mode_stateful_when_conversation_present() {
    let ctx = run_filter("{}", r#"{"input":"test","store":false,"conversation":{"id":"conv_1"}}"#).await;
    let results = ctx.filter_results.get("openai_responses_format").unwrap();

    assert_eq!(results.get("mode"), Some("stateful"));
}

#[tokio::test]
async fn mode_stateful_when_prompt_id_present() {
    let ctx = run_filter("{}", r#"{"input":"test","store":false,"prompt":{"id":"pmpt_123"}}"#).await;
    let results = ctx.filter_results.get("openai_responses_format").unwrap();

    assert_eq!(results.get("mode"), Some("stateful"));
}

#[tokio::test]
async fn mode_not_set_for_chat_completions() {
    let ctx = run_filter("{}", r#"{"messages":[{"role":"user","content":"Hi"}]}"#).await;
    let results = ctx.filter_results.get("openai_responses_format").unwrap();

    assert!(
        results.get("mode").is_none(),
        "mode should not be set for chat_completions"
    );
    assert!(
        !ctx.filter_metadata.contains_key("openai_responses_format.mode"),
        "mode metadata absent for chat_completions"
    );
    let headers = collect_headers(&ctx);
    assert!(
        !headers.contains_key("x-praxis-responses-mode"),
        "mode header absent for chat_completions"
    );

    assert!(
        ctx.extensions.get::<ErrorResponseFormatterHandle>().is_some(),
        "openai error response formatter should be installed for chat completions"
    );
}

#[tokio::test]
async fn mode_stateless_with_store_false_and_empty_tools() {
    let ctx = run_filter("{}", r#"{"input":"test","store":false,"tools":[]}"#).await;
    let results = ctx.filter_results.get("openai_responses_format").unwrap();

    assert_eq!(
        results.get("mode"),
        Some("stateless"),
        "empty tools should not trigger stateful"
    );
}

#[tokio::test]
async fn mode_header_uses_custom_name() {
    let cfg = "headers:\n  mode: x-custom-mode";
    let ctx = run_filter(cfg, r#"{"input":"test","store":false}"#).await;
    let headers = collect_headers(&ctx);

    assert_eq!(headers.get("x-custom-mode"), Some(&"stateless"));
    assert!(
        !headers.contains_key("x-praxis-responses-mode"),
        "default mode header should not be emitted when overridden"
    );
}

#[tokio::test]
async fn mode_header_suppressed_when_null() {
    let cfg = "headers:\n  mode: null";
    let ctx = run_filter(cfg, r#"{"input":"test","store":false}"#).await;
    let headers = collect_headers(&ctx);

    assert!(
        !headers.contains_key("x-praxis-responses-mode"),
        "null mode header should suppress emission"
    );
    assert_eq!(
        ctx.filter_metadata
            .get("openai_responses_format.mode")
            .map(String::as_str),
        Some("stateless"),
        "metadata still written with null mode header"
    );
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

/// Full Responses body with all optional fields for promotion tests.
const FULL_RESPONSES_BODY: &str = r#"{"model":"gpt-4.1","input":"test","stream":true,"store":false,"background":false,"previous_response_id":"resp_abc","conversation":{"id":"conv_1"},"tools":[{"type":"function"}],"prompt":{"id":"pmpt_123"}}"#;

/// Run the filter's `on_request_body` and return the resulting context.
async fn run_filter(config_yaml: &str, body_str: &str) -> HttpFilterContext<'static> {
    let filter = make_filter(config_yaml);
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");

    let req: &'static praxis_filter::Request = Box::leak(Box::new(req));
    let mut ctx = crate::test_utils::make_filter_context(req);
    let mut body = Some(Bytes::from(body_str.to_owned()));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(matches!(action, FilterAction::Release), "filter should release");
    ctx
}

/// Collect extra request headers into a map for assertion.
fn collect_headers<'a>(ctx: &'a HttpFilterContext<'_>) -> std::collections::HashMap<&'a str, &'a str> {
    ctx.extra_request_headers
        .iter()
        .map(|(k, v)| (k.as_ref(), v.as_str()))
        .collect()
}

/// Run the filter's `on_request_body` and return the raw action.
///
/// Uses `POST /v1/responses`. For other methods or paths, see
/// [`run_filter_raw_with_method`].
async fn run_filter_raw(config_yaml: &str, body_str: &str) -> FilterAction {
    run_filter_raw_with_method(config_yaml, body_str, http::Method::POST, "/v1/responses").await
}

/// Run the filter's `on_request_body` with a custom method and path and
/// return the raw action.
async fn run_filter_raw_with_method(
    config_yaml: &str,
    body_str: &str,
    method: http::Method,
    path: &str,
) -> FilterAction {
    let filter = make_filter(config_yaml);
    let req = crate::test_utils::make_request(method, path);

    let req: &'static praxis_filter::Request = Box::leak(Box::new(req));
    let mut ctx = crate::test_utils::make_filter_context(req);
    let mut body = Some(Bytes::from(body_str.to_owned()));

    filter.on_request_body(&mut ctx, &mut body, true).await.unwrap()
}

/// Run the filter's `on_request_body` with a custom method and path.
async fn run_filter_with_method(
    config_yaml: &str,
    body_str: &str,
    method: http::Method,
    path: &str,
) -> HttpFilterContext<'static> {
    let filter = make_filter(config_yaml);
    let req = crate::test_utils::make_request(method, path);

    let req: &'static praxis_filter::Request = Box::leak(Box::new(req));
    let mut ctx = crate::test_utils::make_filter_context(req);
    let mut body = Some(Bytes::from(body_str.to_owned()));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(matches!(action, FilterAction::Release), "filter should release");
    ctx
}

/// Run the filter with custom request headers.
async fn run_filter_with_request_headers(
    config_yaml: &str,
    body_str: &str,
    method: http::Method,
    path: &str,
    headers: &[(http::header::HeaderName, &'static str)],
) -> HttpFilterContext<'static> {
    let filter = make_filter(config_yaml);
    let mut req = crate::test_utils::make_request(method, path);
    for (name, value) in headers {
        req.headers.append(name, value.parse().unwrap());
    }

    let req: &'static praxis_filter::Request = Box::leak(Box::new(req));
    let mut ctx = crate::test_utils::make_filter_context(req);
    let mut body = Some(Bytes::from(body_str.to_owned()));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(matches!(action, FilterAction::Release), "filter should release");
    ctx
}

/// Build a `ResponsesFormatFilter` from a YAML snippet.
fn make_filter(yaml_str: &str) -> Box<dyn HttpFilter> {
    let yaml: serde_yaml::Value = serde_yaml::from_str(yaml_str).unwrap();
    ResponsesFormatFilter::from_config(&yaml).unwrap()
}

// -----------------------------------------------------------------------------
// streamed_round_is_dispatchable Tests
// -----------------------------------------------------------------------------

#[cfg(feature = "openai-responses")]
#[test]
fn dispatchable_requires_terminal_completed_no_parse_error() {
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let mut state = state::ResponsesState::default();
    state.request_body = serde_json::json!({"stream": true});
    // Non-terminal stream: not dispatchable.
    assert!(!streamed_round_is_dispatchable(&ctx, &state));
    ctx.set_metadata("responses.stream_completion", "terminal");
    state.response_object = serde_json::json!({"status": "completed"});
    assert!(streamed_round_is_dispatchable(&ctx, &state));
    ctx.set_metadata("responses.stream_parse_error", "true");
    assert!(
        !streamed_round_is_dispatchable(&ctx, &state),
        "parse error blocks dispatch"
    );
}

#[cfg(feature = "openai-responses")]
#[test]
fn non_streaming_request_is_always_dispatchable() {
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let ctx = crate::test_utils::make_filter_context(&req);
    let state = state::ResponsesState::default(); // no stream flag
    assert!(streamed_round_is_dispatchable(&ctx, &state));
}

#[cfg(feature = "openai-responses")]
#[test]
fn fs_end_stream_writes_five_keys_and_is_idempotent() {
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    fs_end_stream_with_error_ctx(&mut ctx, "server_error", "boom");
    assert_eq!(ctx.get_metadata("responses.stream_error_code"), Some("server_error"));
    assert_eq!(ctx.get_metadata("responses.stream_error_message"), Some("boom"));
    assert_eq!(ctx.get_metadata("responses.skip_persist"), Some("true"));
    // After the #1046 unification the stream-stop is armed on the single
    // continuation owner (`openai_agentic_loop`), not the demoted file-search filter.
    let r = ctx.filter_results.get("openai_agentic_loop").unwrap();
    assert_eq!(r.get("action"), Some("done"));
    assert_eq!(r.get("pending"), Some("false"));
    // Idempotent: a second call with a different code does not clobber.
    fs_end_stream_with_error_ctx(&mut ctx, "other_code", "later");
    assert_eq!(ctx.get_metadata("responses.stream_error_code"), Some("server_error"));
}

// -----------------------------------------------------------------------------
// Conformance Verification Tests
// -----------------------------------------------------------------------------

#[cfg(feature = "openai-responses")]
#[test]
#[expect(clippy::print_stdout, reason = "sentinel output for xtask conformance verification")]
fn conformance_responses_routes_match_runtime_registry() {
    for spec in routes::operation_specs() {
        let path = spec.runtime_path().replace("{response_id}", "resp_test");
        let matched = routes::match_route(spec.method().as_str(), &path, spec.transport())
            .unwrap_or_else(|| panic!("failed to match route for {} {path}", spec.method().as_str()));
        assert_eq!(
            matched.spec.operation,
            spec.operation,
            "matched wrong operation for {} {path}",
            spec.method().as_str()
        );
    }
    println!("PRAXIS_CONFORMANCE_OK responses route_dispatch");
}

#[cfg(feature = "openai-responses")]
#[test]
#[expect(clippy::print_stdout, reason = "sentinel output for xtask conformance verification")]
fn conformance_responses_success_payloads_match_generated_response_schemas() {
    let spec = load_openai_spec();
    let schema = spec
        .pointer("/components/schemas/Response")
        .expect("missing Response schema");

    let chat_response = serde_json::json!({
        "id": "chatcmpl_conformance_123",
        "object": "chat.completion",
        "created": 1_700_000_000_u64,
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "Hello world"
            }
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "prompt_tokens_details": {
                "cached_tokens": 2,
                "cache_write_tokens": 0
            },
            "completion_tokens_details": {
                "reasoning_tokens": 0
            }
        }
    });
    let request = serde_json::json!({"model": "gpt-4o", "input": "Hello"});
    let context = crate::openai::translation::chat_completions::ResponseContext::from_responses_request(
        &request,
        "resp_conformance_123".to_owned(),
        1_700_000_000,
    );
    let response_resource =
        crate::openai::translation::chat_completions::chat_response_to_response_resource(&chat_response, &context)
            .expect("translator must convert chat response to response resource");

    assert_response_matches_schema(&spec, "Response", schema, &response_resource);
    println!("PRAXIS_CONFORMANCE_OK responses success_response_contract");
}

#[cfg(feature = "openai-responses")]
#[test]
#[expect(clippy::print_stdout, reason = "sentinel output for xtask conformance verification")]
fn conformance_responses_sse_lifecycle_events_match_schemas() {
    use crate::openai::responses::responses_to_chat_completions::stream::tests::{run_stream, wide_limits};

    let spec = load_openai_spec();
    let sse_event_schema = spec
        .pointer("/components/schemas/ResponseStreamEvent")
        .expect("missing ResponseStreamEvent schema");

    // Scenarios: buffered text, tool call, incomplete (length)
    let buffered_text_chunks = [
        r#"{"id":"chatcmpl_1","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello "}}]}"#,
        r#"{"id":"chatcmpl_1","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"world"}}]}"#,
        r#"{"id":"chatcmpl_1","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12,"prompt_tokens_details":{"cached_tokens":0,"cache_write_tokens":0},"completion_tokens_details":{"reasoning_tokens":0}}}"#,
    ];

    let tool_call_chunks = [
        r#"{"id":"chatcmpl_2","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_123","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Boston\"}"}}]}}]}"#,
        r#"{"id":"chatcmpl_2","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":15,"completion_tokens":10,"total_tokens":25,"prompt_tokens_details":{"cached_tokens":0,"cache_write_tokens":0},"completion_tokens_details":{"reasoning_tokens":0}}}"#,
    ];

    let incomplete_chunks = [
        r#"{"id":"chatcmpl_3","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":"Truncated..."}}]}"#,
        r#"{"id":"chatcmpl_3","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"finish_reason":"length"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,"prompt_tokens_details":{"cached_tokens":0,"cache_write_tokens":0},"completion_tokens_details":{"reasoning_tokens":0}}}"#,
    ];

    let test_streams = [
        ("buffered_text", run_stream(&buffered_text_chunks, wide_limits())),
        ("tool_call", run_stream(&tool_call_chunks, wide_limits())),
        ("incomplete", run_stream(&incomplete_chunks, wide_limits())),
    ];

    for (scenario, stream_events) in test_streams {
        assert!(!stream_events.is_empty(), "scenario {scenario} generated no events");

        let mut terminal_count = 0;
        for (idx, (event_type, event_val)) in stream_events.iter().enumerate() {
            assert_response_matches_schema(
                &spec,
                &format!("SSE event {idx} ({event_type}) in {scenario}"),
                sse_event_schema,
                event_val,
            );

            let seq = event_val
                .get("sequence_number")
                .and_then(serde_json::Value::as_u64)
                .expect("sequence_number required");
            assert_eq!(seq, idx as u64, "sequence_number mismatch in {scenario} at index {idx}");

            if idx == 0 {
                assert_eq!(
                    event_type, "response.created",
                    "first event in {scenario} must be response.created"
                );
            }

            if matches!(
                event_type.as_str(),
                "response.completed" | "response.failed" | "response.incomplete"
            ) {
                terminal_count += 1;
            }
        }

        assert_eq!(
            terminal_count, 1,
            "scenario {scenario} must have exactly 1 terminal event"
        );
        let (last_event_type, _) = stream_events.last().unwrap();
        assert!(
            matches!(
                last_event_type.as_str(),
                "response.completed" | "response.failed" | "response.incomplete"
            ),
            "last event in {scenario} must be terminal, got {last_event_type}"
        );
    }

    println!("PRAXIS_CONFORMANCE_OK responses sse_lifecycle_contract");
}

#[cfg(feature = "openai-responses")]
#[test]
#[expect(clippy::print_stdout, reason = "sentinel output for xtask conformance verification")]
fn conformance_responses_generated_schema_check_rejects_incomplete_payloads() {
    let spec = load_openai_spec();
    let schema = spec
        .pointer("/components/schemas/Response")
        .expect("missing Response schema");

    let chat_response = serde_json::json!({
        "id": "chatcmpl_conformance_123",
        "object": "chat.completion",
        "created": 1_700_000_000_u64,
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "Hello world"
            }
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "prompt_tokens_details": {
                "cached_tokens": 2,
                "cache_write_tokens": 0
            },
            "completion_tokens_details": {
                "reasoning_tokens": 0
            }
        }
    });
    let request = serde_json::json!({"model": "gpt-4o", "input": "Hello"});
    let context = crate::openai::translation::chat_completions::ResponseContext::from_responses_request(
        &request,
        "resp_conformance_123".to_owned(),
        1_700_000_000,
    );
    let valid_response =
        crate::openai::translation::chat_completions::chat_response_to_response_resource(&chat_response, &context)
            .expect("translator must convert chat response to response resource");

    // Negative 1: missing cache_write_tokens in input_tokens_details
    let mut missing_cache_write = valid_response.clone();
    if let Some(details) = missing_cache_write
        .get_mut("usage")
        .and_then(|u| u.get_mut("input_tokens_details"))
        .and_then(|d| d.as_object_mut())
    {
        details.remove("cache_write_tokens");
    }
    assert!(
        !response_schema_matches(&spec, schema, &missing_cache_write),
        "schema check must reject usage missing required cache_write_tokens"
    );

    // Negative 2: missing required top-level field `id`
    let mut missing_id = valid_response.clone();
    if let Some(obj) = missing_id.as_object_mut() {
        obj.remove("id");
    }
    assert!(
        !response_schema_matches(&spec, schema, &missing_id),
        "schema check must reject response missing required id"
    );

    // Negative 3: SSE event missing required sequence_number
    let sse_event_schema = spec
        .pointer("/components/schemas/ResponseStreamEvent")
        .expect("missing ResponseStreamEvent schema");
    let valid_sse_event = serde_json::json!({
        "type": "response.created",
        "sequence_number": 0,
        "response": valid_response
    });
    let mut missing_sequence_number = valid_sse_event.clone();
    if let Some(obj) = missing_sequence_number.as_object_mut() {
        obj.remove("sequence_number");
    }

    assert!(
        !response_schema_matches(&spec, sse_event_schema, &missing_sequence_number),
        "schema check must reject SSE event missing required sequence_number"
    );

    println!("PRAXIS_CONFORMANCE_OK responses schema_check_sensitivity");
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

#[cfg(feature = "openai-responses")]
fn load_openai_spec() -> serde_json::Value {
    let spec_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../docs/conformance/specs/openai-openapi.yaml"
    );
    let content = std::fs::read_to_string(spec_path).expect("read spec");
    let sanitized = content.replace("9223372036854776000", "9223372036854775807");
    serde_yaml::from_str(&sanitized).expect("parse spec")
}

#[cfg(feature = "openai-responses")]
fn assert_response_matches_schema(
    spec: &serde_json::Value,
    path: &str,
    schema: &serde_json::Value,
    value: &serde_json::Value,
) {
    if let Err(err) = check_schema_match(spec, schema, value) {
        panic!("{path} does not match schema: {err}; value: {value}");
    }
}

#[cfg(feature = "openai-responses")]
fn response_schema_matches(spec: &serde_json::Value, schema: &serde_json::Value, value: &serde_json::Value) -> bool {
    check_schema_match(spec, schema, value).is_ok()
}

#[cfg(feature = "openai-responses")]
fn check_schema_match(
    spec: &serde_json::Value,
    schema: &serde_json::Value,
    value: &serde_json::Value,
) -> Result<(), String> {
    let schema = resolve_spec_schema_ref(spec, schema);

    if value.is_null() && schema.get("nullable").and_then(serde_json::Value::as_bool) == Some(true) {
        return Ok(());
    }
    if let Some(variants) = schema.get("oneOf").and_then(serde_json::Value::as_array) {
        let mut matches = Vec::new();
        for (i, v) in variants.iter().enumerate() {
            if let Ok(()) = check_schema_match(spec, v, value) {
                matches.push(i);
            }
        }
        if matches.len() == 1 {
            return Ok(());
        }
        return Err(format!("oneOf matched {} variants for {value:?}", matches.len()));
    }
    if let Some(variants) = schema.get("anyOf").and_then(serde_json::Value::as_array) {
        for v in variants {
            if check_schema_match(spec, v, value).is_ok() {
                return Ok(());
            }
        }
        return Err(format!("anyOf matched 0 variants for {value:?}"));
    }
    if let Some(variants) = schema.get("allOf").and_then(serde_json::Value::as_array) {
        for (i, v) in variants.iter().enumerate() {
            if let Err(err) = check_schema_match(spec, v, value) {
                return Err(format!("allOf branch {i} failed: {err}"));
            }
        }
        return Ok(());
    }
    if let Some(enum_vals) = schema.get("enum").and_then(serde_json::Value::as_array)
        && !enum_vals.contains(value)
    {
        return Err(format!("value {value:?} not in enum {enum_vals:?}"));
    }
    if let Some(schema_type) = schema.get("type") {
        let matches_type = schema_type.as_str().map_or_else(
            || {
                schema_type.as_array().is_some_and(|types| {
                    types
                        .iter()
                        .filter_map(serde_json::Value::as_str)
                        .any(|kind| value_has_type(value, kind))
                })
            },
            |kind| value_has_type(value, kind),
        );
        if !matches_type {
            return Err(format!("type mismatch: expected {schema_type:?}, got {value:?}"));
        }
    }

    if let Some(items_schema) = schema.get("items") {
        let Some(items) = value.as_array() else {
            return Err(format!("expected array for items check, got {value:?}"));
        };
        for (idx, item) in items.iter().enumerate() {
            if let Err(err) = check_schema_match(spec, items_schema, item) {
                return Err(format!("array item [{idx}] mismatch: {err}"));
            }
        }
    }

    if let Some(object) = value.as_object() {
        if let Some(required) = schema.get("required").and_then(serde_json::Value::as_array) {
            for req in required {
                if let Some(req_str) = req.as_str()
                    && !object.contains_key(req_str)
                {
                    return Err(format!("missing required property {req_str:?}"));
                }
            }
        }
        if let Some(properties) = schema.get("properties").and_then(serde_json::Value::as_object) {
            let required_list = schema
                .get("required")
                .and_then(serde_json::Value::as_array)
                .map(|arr| arr.iter().filter_map(serde_json::Value::as_str).collect::<Vec<_>>())
                .unwrap_or_default();

            for (prop_name, prop_schema) in properties {
                if let Some(prop_val) = object.get(prop_name) {
                    if prop_val.is_null() && !required_list.contains(&prop_name.as_str()) {
                        continue;
                    }
                    if let Err(err) = check_schema_match(spec, prop_schema, prop_val) {
                        return Err(format!("property {prop_name:?} mismatch: {err}"));
                    }
                }
            }
        }
    }

    Ok(())
}

#[cfg(feature = "openai-responses")]
fn value_has_type(value: &serde_json::Value, schema_type: &str) -> bool {
    match schema_type {
        "array" => value.is_array(),
        "boolean" => value.is_boolean(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "null" => value.is_null(),
        "number" => value.as_f64().is_some(),
        "object" => value.is_object(),
        "string" => value.is_string(),
        _ => false,
    }
}

#[cfg(feature = "openai-responses")]
fn resolve_spec_schema_ref<'a>(spec: &'a serde_json::Value, schema: &'a serde_json::Value) -> &'a serde_json::Value {
    let Some(ref_path) = schema.get("$ref").and_then(serde_json::Value::as_str) else {
        return schema;
    };
    let pointer = ref_path.strip_prefix('#').unwrap_or(ref_path);
    spec.pointer(pointer)
        .unwrap_or_else(|| panic!("missing schema ref {ref_path}"))
}
