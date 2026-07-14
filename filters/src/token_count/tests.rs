// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Unit tests for the `token_count` filter.

use bytes::Bytes;
use http::header::HeaderValue;
use praxis_filter::{FilterAction, HttpFilter as _, Response};

use super::*;

// -----------------------------------------------------------------------------
// Config Parsing
// -----------------------------------------------------------------------------

#[test]
fn from_config_with_valid_provider() {
    let config: serde_yaml::Value = serde_yaml::from_str("provider: openai").unwrap();
    let filter = TokenCountFilter::from_config(&config).unwrap();

    assert_eq!(filter.name(), "token_count", "filter name should match");
}

#[test]
fn from_config_all_providers() {
    for provider in ["openai", "anthropic", "google", "bedrock", "azure"] {
        let yaml = format!("provider: {provider}");
        let config: serde_yaml::Value = serde_yaml::from_str(&yaml).unwrap();
        let result = TokenCountFilter::from_config(&config);

        assert!(result.is_ok(), "provider '{provider}' should be accepted");
    }
}

#[test]
fn from_config_rejects_unknown_provider() {
    let config: serde_yaml::Value = serde_yaml::from_str("provider: unknown").unwrap();
    let result = TokenCountFilter::from_config(&config);

    assert!(result.is_err(), "unknown provider should be rejected");
}

#[test]
fn from_config_rejects_missing_provider() {
    let config: serde_yaml::Value = serde_yaml::from_str("{}").unwrap();
    let result = TokenCountFilter::from_config(&config);

    assert!(result.is_err(), "missing provider should be rejected");
}

#[test]
fn from_config_rejects_unknown_fields() {
    let config: serde_yaml::Value = serde_yaml::from_str("provider: openai\nextra: true").unwrap();
    let result = TokenCountFilter::from_config(&config);

    assert!(result.is_err(), "unknown fields should be rejected");
}

// -----------------------------------------------------------------------------
// on_response: Content-Type Detection
// -----------------------------------------------------------------------------

#[tokio::test]
async fn on_response_sets_sse_mode_for_event_stream() {
    let filter = make_filter(TokenUsageProvider::OpenAi);
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let mut resp = make_response_with_content_type("text/event-stream");
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());
    ctx.response_header = None;

    assert_eq!(
        ctx.get_metadata(META_MODE),
        Some("sse"),
        "SSE content-type should set mode to sse"
    );
}

#[tokio::test]
async fn on_response_sets_json_mode_for_application_json() {
    let filter = make_filter(TokenUsageProvider::OpenAi);
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let mut resp = make_response_with_content_type("application/json");
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());
    ctx.response_header = None;

    assert_eq!(
        ctx.get_metadata(META_MODE),
        Some("json"),
        "JSON content-type should set mode to json"
    );
}

#[tokio::test]
async fn on_response_handles_content_type_with_charset() {
    let filter = make_filter(TokenUsageProvider::OpenAi);
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let mut resp = make_response_with_content_type("application/json; charset=utf-8");
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());
    ctx.response_header = None;

    assert_eq!(
        ctx.get_metadata(META_MODE),
        Some("json"),
        "JSON with charset should still set mode to json"
    );
}

#[tokio::test]
async fn on_response_handles_case_insensitive_content_type() {
    let filter = make_filter(TokenUsageProvider::OpenAi);
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let mut resp = make_response_with_content_type("Text/Event-Stream");
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());
    ctx.response_header = None;

    assert_eq!(
        ctx.get_metadata(META_MODE),
        Some("sse"),
        "case-insensitive content-type should be recognized"
    );
}

#[tokio::test]
async fn on_response_skips_non_success_status() {
    let filter = make_filter(TokenUsageProvider::OpenAi);
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let mut resp = make_response_with_status_and_content_type(http::StatusCode::BAD_REQUEST, "application/json");
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());
    ctx.response_header = None;

    assert!(
        ctx.get_metadata(META_MODE).is_none(),
        "non-success status should not set mode"
    );
}

#[tokio::test]
async fn on_response_skips_unknown_content_type() {
    let filter = make_filter(TokenUsageProvider::OpenAi);
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let mut resp = make_response_with_content_type("text/plain");
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());
    ctx.response_header = None;

    assert!(
        ctx.get_metadata(META_MODE).is_none(),
        "unknown content-type should not set mode"
    );
}

#[tokio::test]
async fn on_response_skips_missing_content_type() {
    let filter = make_filter(TokenUsageProvider::OpenAi);
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let mut resp = crate::test_utils::make_response();
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());
    ctx.response_header = None;

    assert!(
        ctx.get_metadata(META_MODE).is_none(),
        "missing content-type should not set mode"
    );
}

// -----------------------------------------------------------------------------
// Non-Streaming JSON: End-to-End
// -----------------------------------------------------------------------------

#[tokio::test]
async fn json_openai_extracts_tokens() {
    let json = br#"{"usage":{"prompt_tokens":15,"completion_tokens":42,"total_tokens":57}}"#;

    let (input, output, total) = run_json_extraction(TokenUsageProvider::OpenAi, json).await;

    assert_eq!(input.as_deref(), Some("15"), "OpenAI input tokens");
    assert_eq!(output.as_deref(), Some("42"), "OpenAI output tokens");
    assert_eq!(total.as_deref(), Some("57"), "OpenAI total tokens");
}

#[tokio::test]
async fn json_anthropic_extracts_tokens() {
    let json = br#"{"usage":{"input_tokens":15,"output_tokens":42}}"#;

    let (input, output, total) = run_json_extraction(TokenUsageProvider::Anthropic, json).await;

    assert_eq!(input.as_deref(), Some("15"), "Anthropic input tokens");
    assert_eq!(output.as_deref(), Some("42"), "Anthropic output tokens");
    assert_eq!(total.as_deref(), Some("57"), "Anthropic total tokens (computed)");
}

#[tokio::test]
async fn json_google_extracts_tokens() {
    let json = br#"{"usageMetadata":{"promptTokenCount":15,"candidatesTokenCount":42,"totalTokenCount":57}}"#;

    let (input, output, total) = run_json_extraction(TokenUsageProvider::Google, json).await;

    assert_eq!(input.as_deref(), Some("15"), "Google input tokens");
    assert_eq!(output.as_deref(), Some("42"), "Google output tokens");
    assert_eq!(total.as_deref(), Some("57"), "Google total tokens");
}

#[tokio::test]
async fn json_bedrock_converse_extracts_tokens() {
    let json = br#"{"usage":{"inputTokens":15,"outputTokens":42,"totalTokens":57}}"#;

    let (input, output, total) = run_json_extraction(TokenUsageProvider::Bedrock, json).await;

    assert_eq!(input.as_deref(), Some("15"), "Bedrock input tokens");
    assert_eq!(output.as_deref(), Some("42"), "Bedrock output tokens");
    assert_eq!(total.as_deref(), Some("57"), "Bedrock total tokens");
}

#[tokio::test]
async fn json_azure_extracts_tokens() {
    let json = br#"{"usage":{"prompt_tokens":5,"completion_tokens":10,"total_tokens":15}}"#;

    let (input, output, total) = run_json_extraction(TokenUsageProvider::Azure, json).await;

    assert_eq!(input.as_deref(), Some("5"), "Azure input tokens");
    assert_eq!(output.as_deref(), Some("10"), "Azure output tokens");
    assert_eq!(total.as_deref(), Some("15"), "Azure total tokens");
}

#[tokio::test]
async fn json_missing_usage_sets_nothing() {
    let json = br#"{"id":"abc","choices":[]}"#;

    let (input, output, total) = run_json_extraction(TokenUsageProvider::OpenAi, json).await;

    assert!(input.is_none(), "missing usage should not set input");
    assert!(output.is_none(), "missing usage should not set output");
    assert!(total.is_none(), "missing usage should not set total");
}

#[tokio::test]
async fn json_malformed_sets_nothing() {
    let (input, output, total) = run_json_extraction(TokenUsageProvider::OpenAi, b"not json").await;

    assert!(input.is_none(), "malformed JSON should not set input");
    assert!(output.is_none(), "malformed JSON should not set output");
    assert!(total.is_none(), "malformed JSON should not set total");
}

#[tokio::test]
async fn json_empty_body_sets_nothing() {
    let (input, output, total) = run_json_extraction(TokenUsageProvider::OpenAi, b"").await;

    assert!(input.is_none(), "empty body should not set input");
    assert!(output.is_none(), "empty body should not set output");
    assert!(total.is_none(), "empty body should not set total");
}

#[tokio::test]
async fn json_chunked_body_reassembled() {
    let filter = make_filter(TokenUsageProvider::OpenAi);
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let mut resp = make_response_with_content_type("application/json");
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());
    ctx.response_header = None;

    let chunk1 = br#"{"usage":{"prompt_tokens":15,"#;
    let chunk2 = br#""completion_tokens":42,"total_tokens":57}}"#;

    let mut body1 = Some(Bytes::from_static(chunk1));
    drop(filter.on_response_body(&mut ctx, &mut body1, false).unwrap());

    let mut body2 = Some(Bytes::from_static(chunk2));
    drop(filter.on_response_body(&mut ctx, &mut body2, true).unwrap());

    assert_eq!(ctx.get_metadata("token.input"), Some("15"), "chunked input tokens");
    assert_eq!(ctx.get_metadata("token.output"), Some("42"), "chunked output tokens");
    assert_eq!(ctx.get_metadata("token.total"), Some("57"), "chunked total tokens");
}

#[tokio::test]
async fn json_clears_working_metadata() {
    let json = br#"{"usage":{"prompt_tokens":15,"completion_tokens":42,"total_tokens":57}}"#;
    let filter = make_filter(TokenUsageProvider::OpenAi);
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let mut resp = make_response_with_content_type("application/json");
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());
    ctx.response_header = None;

    let mut body = Some(Bytes::from_static(json));
    drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());

    let working_keys: Vec<_> = ctx
        .filter_metadata
        .keys()
        .filter(|k| k.starts_with(META_PREFIX))
        .collect();

    assert!(
        working_keys.is_empty(),
        "all token_count.* working metadata should be cleared after extraction"
    );
}

// -----------------------------------------------------------------------------
// Streaming SSE: End-to-End
// -----------------------------------------------------------------------------

#[tokio::test]
async fn sse_openai_final_usage_event() {
    let events = b"data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\ndata: {\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":20,\"total_tokens\":30}}\n\ndata: [DONE]\n\n";

    let (input, output, total) = run_sse_extraction(TokenUsageProvider::OpenAi, events).await;

    assert_eq!(input.as_deref(), Some("10"), "OpenAI SSE input tokens");
    assert_eq!(output.as_deref(), Some("20"), "OpenAI SSE output tokens");
    assert_eq!(total.as_deref(), Some("30"), "OpenAI SSE total tokens");
}

#[tokio::test]
async fn sse_anthropic_accumulated_events() {
    let events = concat!(
        "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":25}}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"delta\":{\"text\":\"Hi\"}}\n\n",
        "data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":42}}\n\n",
    );

    let (input, output, total) = run_sse_extraction(TokenUsageProvider::Anthropic, events.as_bytes()).await;

    assert_eq!(
        input.as_deref(),
        Some("25"),
        "Anthropic SSE input tokens from message_start"
    );
    assert_eq!(
        output.as_deref(),
        Some("42"),
        "Anthropic SSE output tokens from message_delta"
    );
    assert_eq!(total.as_deref(), Some("67"), "Anthropic SSE total tokens (computed)");
}

#[tokio::test]
async fn sse_google_final_usage_event() {
    let events = b"data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hi\"}]}}]}\n\ndata: {\"usageMetadata\":{\"promptTokenCount\":15,\"candidatesTokenCount\":42,\"totalTokenCount\":57}}\n\n";

    let (input, output, total) = run_sse_extraction(TokenUsageProvider::Google, events).await;

    assert_eq!(input.as_deref(), Some("15"), "Google SSE input tokens");
    assert_eq!(output.as_deref(), Some("42"), "Google SSE output tokens");
    assert_eq!(total.as_deref(), Some("57"), "Google SSE total tokens");
}

#[tokio::test]
async fn sse_done_sentinel_ignored() {
    let events =
        b"data: {\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":20,\"total_tokens\":30}}\n\ndata: [DONE]\n\n";

    let (input, output, _) = run_sse_extraction(TokenUsageProvider::OpenAi, events).await;

    assert_eq!(input.as_deref(), Some("10"), "[DONE] should not overwrite usage data");
    assert_eq!(output.as_deref(), Some("20"), "[DONE] should not overwrite usage data");
}

#[tokio::test]
async fn sse_chunks_split_across_calls() {
    let filter = make_filter(TokenUsageProvider::OpenAi);
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let mut resp = make_response_with_content_type("text/event-stream");
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());
    ctx.response_header = None;

    let chunk1 = b"data: {\"usage\":{\"prompt_to";
    let chunk2 = b"kens\":10,\"completion_tokens\":20,\"total_tokens\":30}}\n\n";

    let mut body1 = Some(Bytes::from_static(chunk1));
    drop(filter.on_response_body(&mut ctx, &mut body1, false).unwrap());

    assert!(
        ctx.get_metadata("token.input").is_none(),
        "incomplete SSE frame should not set tokens"
    );

    let mut body2 = Some(Bytes::from_static(chunk2));
    drop(filter.on_response_body(&mut ctx, &mut body2, true).unwrap());

    assert_eq!(ctx.get_metadata("token.input"), Some("10"), "split SSE input tokens");
    assert_eq!(ctx.get_metadata("token.output"), Some("20"), "split SSE output tokens");
}

#[tokio::test]
async fn sse_no_usage_events_sets_nothing() {
    let events = b"data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\ndata: [DONE]\n\n";

    let (input, output, total) = run_sse_extraction(TokenUsageProvider::OpenAi, events).await;

    assert!(input.is_none(), "no usage events should not set input");
    assert!(output.is_none(), "no usage events should not set output");
    assert!(total.is_none(), "no usage events should not set total");
}

#[tokio::test]
async fn sse_clears_working_metadata() {
    let events = b"data: {\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":20,\"total_tokens\":30}}\n\n";
    let filter = make_filter(TokenUsageProvider::OpenAi);
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let mut resp = make_response_with_content_type("text/event-stream");
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());
    ctx.response_header = None;

    let mut body = Some(Bytes::copy_from_slice(events));
    drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());

    assert_no_working_metadata(&ctx);
}

// -----------------------------------------------------------------------------
// SSE: Bedrock ConverseStream Metadata
// -----------------------------------------------------------------------------

#[tokio::test]
async fn sse_bedrock_metadata_event() {
    let events = b"data: {\"contentBlockDelta\":{\"delta\":{\"text\":\"Hi\"},\"contentBlockIndex\":0}}\n\ndata: {\"metadata\":{\"usage\":{\"inputTokens\":30,\"outputTokens\":18}}}\n\n";

    let (input, output, total) = run_sse_extraction(TokenUsageProvider::Bedrock, events).await;

    assert_eq!(input.as_deref(), Some("30"), "Bedrock SSE input tokens");
    assert_eq!(output.as_deref(), Some("18"), "Bedrock SSE output tokens");
    assert_eq!(total.as_deref(), Some("48"), "Bedrock SSE total tokens (computed)");
}

// -----------------------------------------------------------------------------
// SSE: Scratch Overflow
// -----------------------------------------------------------------------------

#[tokio::test]
async fn sse_overflow_finalizes_partial_counts() {
    let filter = make_filter(TokenUsageProvider::OpenAi);
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let mut resp = make_response_with_content_type("text/event-stream");
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());
    ctx.response_header = None;

    let usage_event = b"data: {\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":20,\"total_tokens\":30}}\n\n";
    let mut body1 = Some(Bytes::from_static(usage_event));
    drop(filter.on_response_body(&mut ctx, &mut body1, false).unwrap());
    assert!(ctx.get_metadata("token.input").is_none(), "not finalized yet");

    let overflow_chunk = vec![b'x'; DEFAULT_MAX_SCRATCH_BYTES + 1];
    let mut body2 = Some(Bytes::from(overflow_chunk));
    drop(filter.on_response_body(&mut ctx, &mut body2, false).unwrap());

    assert_eq!(ctx.get_metadata("token.input"), Some("10"));
    assert_eq!(ctx.get_metadata("token.output"), Some("20"));
    assert_eq!(ctx.get_metadata("token.total"), Some("30"));
    assert_no_working_metadata(&ctx);
}

// -----------------------------------------------------------------------------
// Body Mode Without on_response
// -----------------------------------------------------------------------------

#[test]
fn on_response_body_noop_without_mode() {
    let filter = make_filter(TokenUsageProvider::OpenAi);
    let req = crate::test_utils::make_request(http::Method::GET, "/health");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let mut body = Some(Bytes::from_static(b"hello"));
    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();

    assert!(
        matches!(action, FilterAction::Continue),
        "should return Continue without mode"
    );
    assert!(
        ctx.get_metadata("token.input").is_none(),
        "should not set any token metadata without mode"
    );
}

// -----------------------------------------------------------------------------
// Buffer Overflow
// -----------------------------------------------------------------------------

#[tokio::test]
async fn json_overflow_sets_nothing() {
    let filter = make_filter(TokenUsageProvider::OpenAi);
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let mut resp = make_response_with_content_type("application/json");
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());
    ctx.response_header = None;

    let huge_body = vec![b'x'; DEFAULT_MAX_BODY_BYTES + 1];
    let mut body = Some(Bytes::from(huge_body));
    drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());

    assert!(
        ctx.get_metadata("token.input").is_none(),
        "overflow should not set tokens"
    );
    let working_keys: Vec<_> = ctx
        .filter_metadata
        .keys()
        .filter(|k| k.starts_with(META_PREFIX))
        .collect();
    assert!(working_keys.is_empty(), "overflow should clear all working metadata");
}

// -----------------------------------------------------------------------------
// Content-Type Helpers
// -----------------------------------------------------------------------------

#[test]
fn is_event_stream_recognizes_variants() {
    assert!(is_event_stream_content_type("text/event-stream"), "exact match");
    assert!(
        is_event_stream_content_type("text/event-stream; charset=utf-8"),
        "with charset"
    );
    assert!(is_event_stream_content_type("Text/Event-Stream"), "mixed case");
    assert!(is_event_stream_content_type("TEXT/EVENT-STREAM"), "uppercase");
    assert!(
        !is_event_stream_content_type("application/json"),
        "json should not match"
    );
}

#[test]
fn is_json_recognizes_variants() {
    assert!(is_json_content_type("application/json"), "exact match");
    assert!(is_json_content_type("application/json; charset=utf-8"), "with charset");
    assert!(is_json_content_type("Application/JSON"), "mixed case");
    assert!(!is_json_content_type("text/event-stream"), "SSE should not match");
}

// -----------------------------------------------------------------------------
// Hex Encoding
// -----------------------------------------------------------------------------

#[test]
fn decode_hex_roundtrips() {
    let data = b"hello world";
    let encoded = data.iter().fold(String::new(), |mut s, b| {
        _ = write!(s, "{b:02x}");
        s
    });
    let decoded = decode_hex(&encoded).unwrap();

    assert_eq!(decoded, data, "hex roundtrip should preserve data");
}

#[test]
fn decode_hex_rejects_odd_length() {
    assert!(decode_hex("abc").is_none(), "odd-length hex should return None");
}

#[test]
fn decode_hex_rejects_invalid_chars() {
    assert!(decode_hex("zz").is_none(), "invalid hex chars should return None");
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

use std::fmt::Write as _;

fn make_filter(provider: TokenUsageProvider) -> TokenCountFilter {
    TokenCountFilter { provider }
}

fn make_response_with_content_type(ct: &str) -> Response {
    let mut resp = crate::test_utils::make_response();
    resp.headers.insert("content-type", HeaderValue::from_str(ct).unwrap());
    resp
}

fn make_response_with_status_and_content_type(status: http::StatusCode, ct: &str) -> Response {
    let mut resp = Response {
        headers: http::HeaderMap::new(),
        status,
    };
    resp.headers.insert("content-type", HeaderValue::from_str(ct).unwrap());
    resp
}

fn assert_no_working_metadata(ctx: &HttpFilterContext<'_>) {
    let working_keys: Vec<_> = ctx
        .filter_metadata
        .keys()
        .filter(|k| k.starts_with(META_PREFIX))
        .collect();
    assert!(
        working_keys.is_empty(),
        "all token_count.* working metadata should be cleared"
    );
}

/// Run a full `on_response` -> `on_response_body` cycle for JSON extraction.
async fn run_json_extraction(
    provider: TokenUsageProvider,
    body_bytes: &[u8],
) -> (Option<String>, Option<String>, Option<String>) {
    let filter = make_filter(provider);
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let mut resp = make_response_with_content_type("application/json");
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());
    ctx.response_header = None;

    let mut body = Some(Bytes::copy_from_slice(body_bytes));
    drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());

    (
        ctx.get_metadata("token.input").map(str::to_owned),
        ctx.get_metadata("token.output").map(str::to_owned),
        ctx.get_metadata("token.total").map(str::to_owned),
    )
}

/// Run a full `on_response` -> `on_response_body` cycle for SSE extraction.
async fn run_sse_extraction(
    provider: TokenUsageProvider,
    sse_bytes: &[u8],
) -> (Option<String>, Option<String>, Option<String>) {
    let filter = make_filter(provider);
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let mut resp = make_response_with_content_type("text/event-stream");
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());
    ctx.response_header = None;

    let mut body = Some(Bytes::copy_from_slice(sse_bytes));
    drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());

    (
        ctx.get_metadata("token.input").map(str::to_owned),
        ctx.get_metadata("token.output").map(str::to_owned),
        ctx.get_metadata("token.total").map(str::to_owned),
    )
}
