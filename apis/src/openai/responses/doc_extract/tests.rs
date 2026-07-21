// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Unit tests for the `openai_doc_extract` filter.

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use bytes::Bytes;
use http::Method;
use praxis_filter::{BodyAccess, BodyMode, FilterAction, HttpFilter as _, body::MAX_JSON_BODY_BYTES};

use super::{
    config::{DocExtractConfig, OnUnsupported, is_text_safe_mime},
    extract::{ExtractError, ExtractionBudget, extract_input_file, parse_data_uri},
    *,
};
use crate::test_utils::{make_filter_context, make_request};

// -- Helpers ------------------------------------------------------------------

fn make_filter() -> DocExtractFilter {
    let cfg = DocExtractConfig {
        allow_pre_security_callout: true,
        max_body_bytes: MAX_JSON_BODY_BYTES,
        max_content_bytes: 10_485_760,
        max_file_references: 32,
        max_total_text_bytes: 67_108_864,
        on_unsupported: OnUnsupported::Continue,
    };
    DocExtractFilter { config: cfg }
}

fn make_filter_reject() -> DocExtractFilter {
    let cfg = DocExtractConfig {
        allow_pre_security_callout: true,
        max_body_bytes: MAX_JSON_BODY_BYTES,
        max_content_bytes: 10_485_760,
        max_file_references: 32,
        max_total_text_bytes: 67_108_864,
        on_unsupported: OnUnsupported::Reject,
    };
    DocExtractFilter { config: cfg }
}

fn responses_body(input: &serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "model": "test-model",
        "input": input.clone()
    })
}

fn text_file_data(text: &str) -> String {
    format!("data:text/plain;base64,{}", BASE64.encode(text.as_bytes()))
}

fn json_file_data(text: &str) -> String {
    format!("data:application/json;base64,{}", BASE64.encode(text.as_bytes()))
}

fn pdf_file_data() -> String {
    format!("data:application/pdf;base64,{}", BASE64.encode(b"%PDF-1.4"))
}

fn raw_base64(text: &str) -> String {
    BASE64.encode(text.as_bytes())
}

fn set_responses_metadata(ctx: &mut HttpFilterContext<'_>) {
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
}

// -- Config tests -------------------------------------------------------------

#[test]
fn from_config_valid() {
    let yaml = serde_yaml::from_str::<serde_yaml::Value>(
        "
        allow_pre_security_callout: true
        on_unsupported: continue
        ",
    )
    .unwrap();
    let filter = DocExtractFilter::from_config(&yaml);
    assert!(filter.is_ok(), "valid config should succeed");
}

#[test]
fn from_config_missing_pre_security_ack() {
    let yaml = serde_yaml::from_str::<serde_yaml::Value>(
        "
        on_unsupported: continue
        ",
    )
    .unwrap();
    let result = DocExtractFilter::from_config(&yaml);
    let err = result.err().expect("should fail without allow_pre_security_callout");
    assert!(
        err.to_string().contains("allow_pre_security_callout"),
        "should mention allow_pre_security_callout: {err}"
    );
}

#[test]
fn from_config_unknown_field_rejected() {
    let yaml = serde_yaml::from_str::<serde_yaml::Value>(
        "
        allow_pre_security_callout: true
        unknown_field: true
        ",
    )
    .unwrap();
    let err = DocExtractFilter::from_config(&yaml);
    assert!(err.is_err(), "unknown fields should be rejected");
}

#[test]
fn from_config_zero_max_content_bytes_rejected() {
    let yaml = serde_yaml::from_str::<serde_yaml::Value>(
        "
        allow_pre_security_callout: true
        max_content_bytes: 0
        ",
    )
    .unwrap();
    let result = DocExtractFilter::from_config(&yaml);
    let err = result.err().expect("should fail with zero max_content_bytes");
    assert!(
        err.to_string().contains("max_content_bytes"),
        "should mention max_content_bytes: {err}"
    );
}

// -- Body access tests --------------------------------------------------------

#[test]
fn body_access_is_read_write() {
    let filter = make_filter();
    assert_eq!(
        filter.request_body_access(),
        BodyAccess::ReadWrite,
        "doc_extract needs ReadWrite to rewrite the body"
    );
}

#[test]
fn body_mode_is_stream_buffer_with_default_limit() {
    let filter = make_filter();
    assert!(
        matches!(
            filter.request_body_mode(),
            BodyMode::StreamBuffer { max_bytes: Some(n) } if n == MAX_JSON_BODY_BYTES
        ),
        "expected StreamBuffer with 64 MiB default"
    );
}

// -- Request gating tests -----------------------------------------------------

#[tokio::test]
async fn skips_non_responses_request() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/chat/completions");
    let mut ctx = make_filter_context(&req);
    let body_json = serde_json::json!({"model": "test", "messages": []});
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Release),
        "chat completions should be released"
    );
}

#[tokio::test]
async fn skips_non_create_endpoint() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses/compact");
    let mut ctx = make_filter_context(&req);
    set_responses_metadata(&mut ctx);
    let body_json = serde_json::json!({"model": "test"});
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Release),
        "non-create responses endpoints should be released"
    );
}

#[tokio::test]
async fn skips_missing_format_metadata() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let body_json = serde_json::json!({"model": "test", "input": "hello"});
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Release),
        "missing format metadata should cause release"
    );
}

#[tokio::test]
async fn releases_invalid_json() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    set_responses_metadata(&mut ctx);
    let mut body = Some(Bytes::from_static(b"not json"));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Release),
        "invalid JSON should be released"
    );
}

#[tokio::test]
async fn not_end_of_stream_continues() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let mut body = Some(Bytes::from_static(b"{}"));

    let action = filter.on_request_body(&mut ctx, &mut body, false).await.unwrap();
    assert!(matches!(action, FilterAction::Continue), "partial body should continue");
}

#[tokio::test]
async fn continues_on_no_input_file() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    set_responses_metadata(&mut ctx);
    let body_json = responses_body(&serde_json::json!([
        {"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "hello"}
        ]}
    ]));
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "no input_file parts should continue unchanged"
    );
}

#[tokio::test]
async fn string_input_passes_through() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    set_responses_metadata(&mut ctx);
    let body_json = responses_body(&serde_json::json!("Hello, world!"));
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "string input should pass through"
    );
}

// -- Data URI parsing tests ---------------------------------------------------

#[test]
fn parse_data_uri_text_plain() {
    let encoded = BASE64.encode(b"hello world");
    let data = format!("data:text/plain;base64,{encoded}");
    let uri = parse_data_uri(&data).unwrap();
    assert_eq!(uri.mime, "text/plain", "MIME should be extracted from data URI");
    assert_eq!(
        uri.base64_payload, &encoded,
        "payload should be the base64 portion after the comma"
    );
}

#[test]
fn parse_data_uri_application_json() {
    let encoded = BASE64.encode(b"{\"key\": \"value\"}");
    let data = format!("data:application/json;base64,{encoded}");
    let uri = parse_data_uri(&data).unwrap();
    assert_eq!(uri.mime, "application/json", "should parse application/json MIME");
}

#[test]
fn parse_data_uri_no_data_prefix_returns_none() {
    assert!(
        parse_data_uri("not-a-data-uri").is_none(),
        "non-data-URI strings should return None"
    );
}

#[test]
fn parse_data_uri_missing_base64_returns_none() {
    assert!(
        parse_data_uri("data:text/plain,aGVsbG8=").is_none(),
        "data URI without ;base64 marker should return None"
    );
}

// -- Text-safe MIME matching tests --------------------------------------------

#[test]
fn text_plain_is_text_safe() {
    assert!(is_text_safe_mime("text/plain"), "text/plain should be text-safe");
}

#[test]
fn text_csv_is_text_safe() {
    assert!(is_text_safe_mime("text/csv"), "text/csv should be text-safe");
}

#[test]
fn text_html_is_text_safe() {
    assert!(is_text_safe_mime("text/html"), "text/html should be text-safe");
}

#[test]
fn application_json_is_text_safe() {
    assert!(
        is_text_safe_mime("application/json"),
        "application/json should be text-safe"
    );
}

#[test]
fn application_xml_is_text_safe() {
    assert!(
        is_text_safe_mime("application/xml"),
        "application/xml should be text-safe"
    );
}

#[test]
fn application_pdf_is_not_text_safe() {
    assert!(
        !is_text_safe_mime("application/pdf"),
        "application/pdf should not be text-safe"
    );
}

#[test]
fn application_octet_stream_is_not_text_safe() {
    assert!(
        !is_text_safe_mime("application/octet-stream"),
        "application/octet-stream should not be text-safe"
    );
}

#[test]
fn text_safe_with_charset_parameter() {
    assert!(
        is_text_safe_mime("text/plain; charset=utf-8"),
        "text/plain with charset parameter should be text-safe"
    );
}

// -- Extraction tests ---------------------------------------------------------

#[tokio::test]
async fn extracts_text_plain_data_uri() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    set_responses_metadata(&mut ctx);

    let body_json = responses_body(&serde_json::json!([
        {"type": "message", "role": "user", "content": [
            {"type": "input_file", "filename": "notes.txt", "file_data": text_file_data("Hello from file!")}
        ]}
    ]));
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "should continue after extraction"
    );

    let rewritten: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    let parts = &rewritten["input"][0]["content"];
    assert_eq!(parts[0]["type"], "input_text", "input_file should become input_text");
    let text = parts[0]["text"].as_str().unwrap();
    assert!(
        text.contains("[Source: notes.txt]"),
        "should have filename prefix: {text}"
    );
    assert!(text.contains("Hello from file!"), "should have file content: {text}");
}

#[tokio::test]
async fn extracts_json_data_uri() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    set_responses_metadata(&mut ctx);

    let body_json = responses_body(&serde_json::json!([
        {"type": "message", "role": "user", "content": [
            {"type": "input_file", "filename": "data.json", "file_data": json_file_data("{\"key\": \"value\"}")}
        ]}
    ]));
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "JSON data URI extraction should continue"
    );

    let rewritten: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    let parts = &rewritten["input"][0]["content"];
    assert_eq!(
        parts[0]["type"], "input_text",
        "JSON input_file should become input_text"
    );
    assert!(
        parts[0]["text"].as_str().unwrap().contains(r#"{"key": "value"}"#),
        "extracted text should contain JSON content"
    );
}

#[tokio::test]
async fn raw_base64_with_txt_filename_extracted() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    set_responses_metadata(&mut ctx);

    let body_json = responses_body(&serde_json::json!([
        {"type": "message", "role": "user", "content": [
            {"type": "input_file", "filename": "readme.txt", "file_data": raw_base64("Raw text content")}
        ]}
    ]));
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "raw base64 extraction should continue"
    );

    let rewritten: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    let text = rewritten["input"][0]["content"][0]["text"].as_str().unwrap();
    assert!(
        text.contains("Raw text content"),
        "should contain decoded raw base64 text: {text}"
    );
    assert!(
        text.contains("[Source: readme.txt]"),
        "should have filename prefix: {text}"
    );
}

#[tokio::test]
async fn unsupported_format_continue_leaves_input_file() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    set_responses_metadata(&mut ctx);

    let body_json = responses_body(&serde_json::json!([
        {"type": "message", "role": "user", "content": [
            {"type": "input_file", "filename": "doc.pdf", "file_data": pdf_file_data()}
        ]}
    ]));
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "unsupported file with on_unsupported=continue should continue"
    );

    let rewritten: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    assert_eq!(
        rewritten["input"][0]["content"][0]["type"], "input_file",
        "unsupported format should leave input_file unchanged"
    );
}

#[test]
fn non_text_data_uri_skips_without_decoding() {
    let mut budget = ExtractionBudget::new(&DocExtractConfig {
        allow_pre_security_callout: true,
        max_body_bytes: MAX_JSON_BODY_BYTES,
        max_content_bytes: 10_485_760,
        max_file_references: 32,
        max_total_text_bytes: 67_108_864,
        on_unsupported: OnUnsupported::Continue,
    });

    let part = serde_json::json!({
        "type": "input_file",
        "filename": "slide.pptx",
        "file_data": pdf_file_data()
    });

    let result = extract_input_file(&part, &mut budget).unwrap();
    assert_eq!(result, None, "non-text MIME should return None without decoding");
    assert_eq!(budget.references_seen, 1, "reference should still be counted");
    assert_eq!(
        budget.remaining_total_text_bytes, budget.max_total_text_bytes,
        "no text bytes should be consumed for skipped files"
    );
}

#[tokio::test]
async fn unsupported_format_reject_returns_400() {
    let filter = make_filter_reject();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    set_responses_metadata(&mut ctx);

    let body_json = responses_body(&serde_json::json!([
        {"type": "message", "role": "user", "content": [
            {"type": "input_file", "filename": "doc.pdf", "file_data": pdf_file_data()}
        ]}
    ]));
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    match action {
        FilterAction::Reject(rejection) => {
            assert_eq!(rejection.status, 400, "unsupported format should return 400");
        },
        _ => panic!("expected rejection for unsupported format with reject policy"),
    }
}

#[tokio::test]
async fn mixed_text_and_unsupported_extracts_text_only() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    set_responses_metadata(&mut ctx);

    let body_json = responses_body(&serde_json::json!([
        {"type": "message", "role": "user", "content": [
            {"type": "input_file", "filename": "notes.txt", "file_data": text_file_data("text content")},
            {"type": "input_file", "filename": "doc.pdf", "file_data": pdf_file_data()},
            {"type": "input_text", "text": "some question"}
        ]}
    ]));
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "mixed content extraction should continue"
    );

    let rewritten: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    let parts = &rewritten["input"][0]["content"];
    assert_eq!(parts[0]["type"], "input_text", "text file should be extracted");
    assert_eq!(parts[1]["type"], "input_file", "PDF should remain as input_file");
    assert_eq!(
        parts[2]["type"], "input_text",
        "original input_text should be preserved"
    );
}

#[tokio::test]
async fn function_call_output_input_file_extracted() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    set_responses_metadata(&mut ctx);

    let body_json = responses_body(&serde_json::json!([
        {"type": "function_call_output", "call_id": "call_123", "output": [
            {"type": "input_file", "filename": "result.txt", "file_data": text_file_data("tool output")}
        ]}
    ]));
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "function_call_output extraction should continue"
    );

    let rewritten: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    let parts = &rewritten["input"][0]["output"];
    assert_eq!(
        parts[0]["type"], "input_text",
        "tool output input_file should become input_text"
    );
    assert!(
        parts[0]["text"].as_str().unwrap().contains("tool output"),
        "extracted tool output should contain original content"
    );
}

#[tokio::test]
async fn file_id_only_skipped() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    set_responses_metadata(&mut ctx);

    let body_json = responses_body(&serde_json::json!([
        {"type": "message", "role": "user", "content": [
            {"type": "input_file", "file_id": "file_abc123"}
        ]}
    ]));
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "file_id-only part should continue (no file_data to extract)"
    );

    let rewritten: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    assert_eq!(
        rewritten["input"][0]["content"][0]["type"], "input_file",
        "file_id-only should be left unchanged"
    );
    assert_eq!(
        rewritten["input"][0]["content"][0]["file_id"], "file_abc123",
        "file_id should be preserved"
    );
}

#[tokio::test]
async fn file_url_only_skipped() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    set_responses_metadata(&mut ctx);

    let body_json = responses_body(&serde_json::json!([
        {"type": "message", "role": "user", "content": [
            {"type": "input_file", "file_url": "https://example.com/file.txt"}
        ]}
    ]));
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "file_url-only part should continue (no file_data to extract)"
    );

    let rewritten: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    assert_eq!(
        rewritten["input"][0]["content"][0]["type"], "input_file",
        "file_url-only should be left unchanged (no network I/O)"
    );
}

#[tokio::test]
async fn rejects_oversized_raw_body() {
    let filter = DocExtractFilter {
        config: DocExtractConfig {
            allow_pre_security_callout: true,
            max_body_bytes: 10,
            max_content_bytes: 10_485_760,
            max_file_references: 32,
            max_total_text_bytes: 67_108_864,
            on_unsupported: OnUnsupported::Continue,
        },
    };
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    set_responses_metadata(&mut ctx);
    let mut body = Some(Bytes::from(vec![b'x'; 100]));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Reject(_)),
        "oversized body should be rejected"
    );
}

// -- State coherence tests ----------------------------------------------------

#[tokio::test]
async fn sync_state_updates_responses_state() {
    use super::super::state::ResponsesState;

    let filter = make_filter();
    let req = Box::leak(Box::new(make_request(Method::POST, "/v1/responses")));
    let mut ctx = make_filter_context(req);
    set_responses_metadata(&mut ctx);

    let input = serde_json::json!([
        {"type": "message", "role": "user", "content": [
            {"type": "input_file", "filename": "notes.txt", "file_data": text_file_data("state content")}
        ]}
    ]);
    let body_json = responses_body(&input);

    let state = ResponsesState::from_request_body(body_json.clone());
    assert!(!state.messages.is_empty(), "state should have messages");
    ctx.extensions.insert(state);

    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "extraction with state should continue"
    );

    let state = ctx.extensions.get::<ResponsesState>().unwrap();

    let state_input = state.request_body.get("input").and_then(|v| v.as_array()).unwrap();
    assert_eq!(
        state_input[0]["content"][0]["type"], "input_text",
        "state.request_body should have extracted input_text"
    );

    assert_eq!(
        state.messages[0]["content"][0]["type"], "input_text",
        "state.messages should have extracted input_text"
    );

    assert_eq!(
        state.persisted_messages[0]["content"][0]["type"], "input_text",
        "state.persisted_messages should have extracted input_text"
    );
}

#[tokio::test]
async fn persisted_history_does_not_double_count_references() {
    use super::super::state::ResponsesState;

    let cfg = DocExtractConfig {
        allow_pre_security_callout: true,
        max_body_bytes: MAX_JSON_BODY_BYTES,
        max_content_bytes: 10_485_760,
        max_file_references: 2,
        max_total_text_bytes: 67_108_864,
        on_unsupported: OnUnsupported::Continue,
    };
    let filter = DocExtractFilter { config: cfg };

    let req = Box::leak(Box::new(make_request(Method::POST, "/v1/responses")));
    let mut ctx = make_filter_context(req);
    set_responses_metadata(&mut ctx);

    let current_input = serde_json::json!([
        {"type": "message", "role": "user", "content": [
            {"type": "input_file", "filename": "a.txt", "file_data": text_file_data("aaa")},
            {"type": "input_file", "filename": "b.txt", "file_data": text_file_data("bbb")}
        ]}
    ]);
    let body_json = responses_body(&current_input);

    let mut state = ResponsesState::from_request_body(body_json.clone());
    let history_item = serde_json::json!(
        {"type": "message", "role": "assistant", "content": [
            {"type": "output_text", "text": "prior response"}
        ]}
    );
    state.messages.insert(0, history_item.clone());
    state.persisted_messages.insert(0, history_item);
    ctx.extensions.insert(state);

    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "2 files with max_file_references=2 should succeed even with persisted mirror"
    );

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(
        state.messages.last().unwrap()["content"][0]["type"],
        "input_text",
        "messages should have extracted content"
    );
    assert_eq!(
        state.persisted_messages.last().unwrap()["content"][0]["type"],
        "input_text",
        "persisted_messages should have extracted content"
    );
}

// -- Size limit tests ---------------------------------------------------------

#[tokio::test]
async fn rejects_when_too_many_file_references() {
    let cfg = DocExtractConfig {
        allow_pre_security_callout: true,
        max_body_bytes: MAX_JSON_BODY_BYTES,
        max_content_bytes: 10_485_760,
        max_file_references: 1,
        max_total_text_bytes: 67_108_864,
        on_unsupported: OnUnsupported::Continue,
    };
    let filter = DocExtractFilter { config: cfg };

    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    set_responses_metadata(&mut ctx);

    let body_json = responses_body(&serde_json::json!([
        {"type": "message", "role": "user", "content": [
            {"type": "input_file", "filename": "a.txt", "file_data": text_file_data("one")},
            {"type": "input_file", "filename": "b.txt", "file_data": text_file_data("two")}
        ]}
    ]));
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Reject(_)),
        "should reject when exceeding file reference limit"
    );
}

#[tokio::test]
async fn rejects_oversized_base64_before_decode() {
    let cfg = DocExtractConfig {
        allow_pre_security_callout: true,
        max_body_bytes: MAX_JSON_BODY_BYTES,
        max_content_bytes: 10,
        max_file_references: 32,
        max_total_text_bytes: 67_108_864,
        on_unsupported: OnUnsupported::Continue,
    };
    let filter = DocExtractFilter { config: cfg };

    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    set_responses_metadata(&mut ctx);

    let large_text = "x".repeat(100);
    let body_json = responses_body(&serde_json::json!([
        {"type": "message", "role": "user", "content": [
            {"type": "input_file", "filename": "big.txt", "file_data": text_file_data(&large_text)}
        ]}
    ]));
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    match action {
        FilterAction::Reject(rejection) => {
            assert_eq!(
                rejection.status, 413,
                "oversized base64 should be rejected as too large"
            );
        },
        _ => panic!("expected rejection for oversized base64 payload"),
    }
}

#[test]
fn base64_precheck_does_not_reject_at_exact_limit() {
    let cfg = DocExtractConfig {
        allow_pre_security_callout: true,
        max_body_bytes: MAX_JSON_BODY_BYTES,
        max_content_bytes: 3,
        max_file_references: 32,
        max_total_text_bytes: 67_108_864,
        on_unsupported: OnUnsupported::Continue,
    };
    let mut budget = ExtractionBudget::new(&cfg);

    let part = serde_json::json!({
        "type": "input_file",
        "file_data": format!("data:text/plain;base64,{}", BASE64.encode(b"abc"))
    });

    let result = extract_input_file(&part, &mut budget);
    assert!(result.is_ok(), "3-byte file should pass with max_content_bytes=3");
    assert!(result.unwrap().is_some(), "text-safe file should be extracted");
}

#[test]
fn filename_prefix_counted_in_content_limit() {
    let limit: usize = 100;
    let cfg = DocExtractConfig {
        allow_pre_security_callout: true,
        max_body_bytes: MAX_JSON_BODY_BYTES,
        max_content_bytes: limit,
        max_file_references: 32,
        max_total_text_bytes: 67_108_864,
        on_unsupported: OnUnsupported::Continue,
    };
    let mut budget = ExtractionBudget::new(&cfg);

    let prefix = "[Source: report.txt]\n";
    let content = "x".repeat(limit - prefix.len() + 1);
    let part = serde_json::json!({
        "type": "input_file",
        "filename": "report.txt",
        "file_data": text_file_data(&content)
    });

    let result = extract_input_file(&part, &mut budget);
    assert!(
        result.is_err(),
        "should reject when prefix pushes text over max_content_bytes"
    );
}

#[test]
fn filename_prefix_fits_within_content_limit() {
    let limit: usize = 100;
    let cfg = DocExtractConfig {
        allow_pre_security_callout: true,
        max_body_bytes: MAX_JSON_BODY_BYTES,
        max_content_bytes: limit,
        max_file_references: 32,
        max_total_text_bytes: 67_108_864,
        on_unsupported: OnUnsupported::Continue,
    };
    let mut budget = ExtractionBudget::new(&cfg);

    let prefix = "[Source: report.txt]\n";
    let content = "x".repeat(limit - prefix.len());
    let part = serde_json::json!({
        "type": "input_file",
        "filename": "report.txt",
        "file_data": text_file_data(&content)
    });

    let result = extract_input_file(&part, &mut budget);
    assert!(
        result.is_ok(),
        "should succeed when prefix + content == max_content_bytes"
    );
    let text = result.unwrap().unwrap();
    assert_eq!(text.len(), limit, "final text should be exactly at the content limit");
}

#[tokio::test]
async fn input_text_without_filename_has_no_prefix() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    set_responses_metadata(&mut ctx);

    let body_json = responses_body(&serde_json::json!([
        {"type": "message", "role": "user", "content": [
            {"type": "input_file", "file_data": text_file_data("no name here")}
        ]}
    ]));
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "extraction without filename should continue"
    );

    let rewritten: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    let text = rewritten["input"][0]["content"][0]["text"].as_str().unwrap();
    assert_eq!(text, "no name here", "no filename should mean no prefix");
}

// -- Config boundary tests ----------------------------------------------------

#[test]
fn from_config_zero_max_body_bytes_rejected() {
    let yaml = serde_yaml::from_str::<serde_yaml::Value>(
        "
        allow_pre_security_callout: true
        max_body_bytes: 0
        ",
    )
    .unwrap();
    let result = DocExtractFilter::from_config(&yaml);
    let err = result.err().expect("should fail with zero max_body_bytes");
    assert!(
        err.to_string().contains("max_body_bytes"),
        "should mention max_body_bytes: {err}"
    );
}

#[test]
fn from_config_exceeds_max_body_bytes_rejected() {
    let yaml = serde_yaml::from_str::<serde_yaml::Value>(&format!(
        "
        allow_pre_security_callout: true
        max_body_bytes: {}
        ",
        MAX_JSON_BODY_BYTES + 1
    ))
    .unwrap();
    let result = DocExtractFilter::from_config(&yaml);
    let err = result.err().expect("should fail when max_body_bytes exceeds maximum");
    assert!(
        err.to_string().contains("max_body_bytes"),
        "should mention max_body_bytes: {err}"
    );
}

#[test]
fn from_config_zero_max_file_references_rejected() {
    let yaml = serde_yaml::from_str::<serde_yaml::Value>(
        "
        allow_pre_security_callout: true
        max_file_references: 0
        ",
    )
    .unwrap();
    let result = DocExtractFilter::from_config(&yaml);
    let err = result.err().expect("should fail with zero max_file_references");
    assert!(
        err.to_string().contains("max_file_references"),
        "should mention max_file_references: {err}"
    );
}

#[test]
fn from_config_exceeds_max_file_references_rejected() {
    let yaml = serde_yaml::from_str::<serde_yaml::Value>(
        "
        allow_pre_security_callout: true
        max_file_references: 999
        ",
    )
    .unwrap();
    let result = DocExtractFilter::from_config(&yaml);
    let err = result
        .err()
        .expect("should fail when max_file_references exceeds maximum");
    assert!(
        err.to_string().contains("max_file_references"),
        "should mention max_file_references: {err}"
    );
}

#[test]
fn from_config_zero_max_total_text_bytes_rejected() {
    let yaml = serde_yaml::from_str::<serde_yaml::Value>(
        "
        allow_pre_security_callout: true
        max_total_text_bytes: 0
        ",
    )
    .unwrap();
    let result = DocExtractFilter::from_config(&yaml);
    let err = result.err().expect("should fail with zero max_total_text_bytes");
    assert!(
        err.to_string().contains("max_total_text_bytes"),
        "should mention max_total_text_bytes: {err}"
    );
}

#[test]
fn from_config_exceeds_max_total_text_bytes_rejected() {
    let yaml = serde_yaml::from_str::<serde_yaml::Value>(&format!(
        "
        allow_pre_security_callout: true
        max_total_text_bytes: {}
        ",
        MAX_JSON_BODY_BYTES + 1
    ))
    .unwrap();
    let result = DocExtractFilter::from_config(&yaml);
    let err = result
        .err()
        .expect("should fail when max_total_text_bytes exceeds maximum");
    assert!(
        err.to_string().contains("max_total_text_bytes"),
        "should mention max_total_text_bytes: {err}"
    );
}

#[test]
fn from_config_exceeds_max_content_bytes_rejected() {
    let yaml = serde_yaml::from_str::<serde_yaml::Value>(
        "
        allow_pre_security_callout: true
        max_content_bytes: 10485761
        ",
    )
    .unwrap();
    let result = DocExtractFilter::from_config(&yaml);
    let err = result
        .err()
        .expect("should fail when max_content_bytes exceeds input_text schema maximum");
    assert!(
        err.to_string().contains("max_content_bytes"),
        "should mention max_content_bytes: {err}"
    );
    assert!(
        err.to_string().contains("input_text schema maximum"),
        "should reference the schema limit: {err}"
    );
}

// -- Extract edge cases -------------------------------------------------------

#[test]
fn malformed_base64_returns_decode_error() {
    let cfg = DocExtractConfig {
        allow_pre_security_callout: true,
        max_body_bytes: MAX_JSON_BODY_BYTES,
        max_content_bytes: 10_485_760,
        max_file_references: 32,
        max_total_text_bytes: 67_108_864,
        on_unsupported: OnUnsupported::Continue,
    };
    let mut budget = ExtractionBudget::new(&cfg);

    let part = serde_json::json!({
        "type": "input_file",
        "file_data": "data:text/plain;base64,!!!not-valid-base64!!!"
    });

    let result = extract_input_file(&part, &mut budget);
    assert!(result.is_err(), "malformed base64 should return an error");
    let err = result.unwrap_err();
    assert!(
        matches!(err, ExtractError::DecodeFailed { .. }),
        "error should be DecodeFailed: {err}"
    );
}

#[test]
fn invalid_utf8_continue_skips() {
    let cfg = DocExtractConfig {
        allow_pre_security_callout: true,
        max_body_bytes: MAX_JSON_BODY_BYTES,
        max_content_bytes: 10_485_760,
        max_file_references: 32,
        max_total_text_bytes: 67_108_864,
        on_unsupported: OnUnsupported::Continue,
    };
    let mut budget = ExtractionBudget::new(&cfg);

    let invalid_utf8 = BASE64.encode([0xFF, 0xFE, 0x00, 0x01]);
    let part = serde_json::json!({
        "type": "input_file",
        "file_data": format!("data:text/plain;base64,{invalid_utf8}")
    });

    let result = extract_input_file(&part, &mut budget);
    assert!(
        result.is_ok(),
        "invalid UTF-8 with on_unsupported=continue should not error"
    );
    assert!(
        result.unwrap().is_none(),
        "invalid UTF-8 should be skipped (returns None)"
    );
}

#[test]
fn invalid_utf8_reject_returns_unsupported() {
    let cfg = DocExtractConfig {
        allow_pre_security_callout: true,
        max_body_bytes: MAX_JSON_BODY_BYTES,
        max_content_bytes: 10_485_760,
        max_file_references: 32,
        max_total_text_bytes: 67_108_864,
        on_unsupported: OnUnsupported::Reject,
    };
    let mut budget = ExtractionBudget::new(&cfg);

    let invalid_utf8 = BASE64.encode([0xFF, 0xFE, 0x00, 0x01]);
    let part = serde_json::json!({
        "type": "input_file",
        "file_data": format!("data:text/plain;base64,{invalid_utf8}")
    });

    let result = extract_input_file(&part, &mut budget);
    assert!(result.is_err(), "invalid UTF-8 with on_unsupported=reject should error");
    let err = result.unwrap_err();
    assert!(
        matches!(err, ExtractError::Unsupported { .. }),
        "error should be Unsupported: {err}"
    );
}

#[test]
fn aggregate_text_bytes_overflow_rejected() {
    let cfg = DocExtractConfig {
        allow_pre_security_callout: true,
        max_body_bytes: MAX_JSON_BODY_BYTES,
        max_content_bytes: 10_485_760,
        max_file_references: 32,
        max_total_text_bytes: 10,
        on_unsupported: OnUnsupported::Continue,
    };
    let mut budget = ExtractionBudget::new(&cfg);

    let part1 = serde_json::json!({
        "type": "input_file",
        "file_data": text_file_data("abcdefghij")
    });
    let result1 = extract_input_file(&part1, &mut budget);
    assert!(result1.is_ok(), "first file fitting within total budget should succeed");

    let part2 = serde_json::json!({
        "type": "input_file",
        "file_data": text_file_data("x")
    });
    let result2 = extract_input_file(&part2, &mut budget);
    assert!(
        result2.is_err(),
        "second file should fail when aggregate exceeds max_total_text_bytes"
    );
}

// -- History and state body overflow tests ------------------------------------

#[tokio::test]
async fn history_only_extraction_converts_rehydrated_files() {
    use super::super::state::ResponsesState;

    let filter = make_filter();
    let req = Box::leak(Box::new(make_request(Method::POST, "/v1/responses")));
    let mut ctx = make_filter_context(req);
    set_responses_metadata(&mut ctx);

    let body_json = responses_body(&serde_json::json!([
        {"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "summarize the file"}
        ]}
    ]));

    let mut state = ResponsesState::from_request_body(body_json.clone());
    let history_file = serde_json::json!(
        {"type": "message", "role": "user", "content": [
            {"type": "input_file", "filename": "old.txt", "file_data": text_file_data("historical content")}
        ]}
    );
    state.messages.insert(0, history_file.clone());
    state.persisted_messages.insert(0, history_file);
    ctx.extensions.insert(state);

    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "history-only extraction should continue"
    );

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(
        state.messages[0]["content"][0]["type"], "input_text",
        "history input_file in messages should be extracted"
    );
    assert_eq!(
        state.persisted_messages[0]["content"][0]["type"], "input_text",
        "history input_file in persisted_messages should be extracted"
    );
    let text = state.messages[0]["content"][0]["text"].as_str().unwrap();
    assert!(
        text.contains("historical content"),
        "extracted history text should contain original content: {text}"
    );
}

#[tokio::test]
async fn rejects_when_state_body_exceeds_max_body_bytes() {
    use super::super::state::ResponsesState;

    let cfg = DocExtractConfig {
        allow_pre_security_callout: true,
        max_body_bytes: 200,
        max_content_bytes: 10_485_760,
        max_file_references: 32,
        max_total_text_bytes: 67_108_864,
        on_unsupported: OnUnsupported::Continue,
    };
    let filter = DocExtractFilter { config: cfg };

    let req = Box::leak(Box::new(make_request(Method::POST, "/v1/responses")));
    let mut ctx = make_filter_context(req);
    set_responses_metadata(&mut ctx);

    let body_json = responses_body(&serde_json::json!([
        {"type": "message", "role": "user", "content": [
            {"type": "input_file", "filename": "a.txt", "file_data": text_file_data("x")}
        ]}
    ]));

    let mut state = ResponsesState::from_request_body(body_json.clone());
    let large_history = serde_json::json!(
        {"type": "message", "role": "assistant", "content": [
            {"type": "output_text", "text": "x".repeat(300)}
        ]}
    );
    state.messages.insert(0, large_history.clone());
    state.persisted_messages.insert(0, large_history);
    ctx.extensions.insert(state);

    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    match action {
        FilterAction::Reject(rejection) => {
            assert_eq!(
                rejection.status, 413,
                "state body exceeding max_body_bytes should return 413"
            );
        },
        _ => panic!("expected rejection when state body (with large history) exceeds max_body_bytes"),
    }
}
