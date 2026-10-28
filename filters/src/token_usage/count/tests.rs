// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Boundary tests for the `token_count` filter.

use std::time::Duration;

use bytes::Bytes;
use http::header::HeaderValue;
use metrics_util::debugging::{DebugValue, DebuggingRecorder};
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
    for provider in [
        "openai",
        "anthropic",
        "google",
        "bedrock",
        "bedrock_invoke_model",
        "azure",
        "auto",
    ] {
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
    let filter = make_filter(ProviderKind::OpenAi);
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
    let filter = make_filter(ProviderKind::OpenAi);
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
    let filter = make_filter(ProviderKind::OpenAi);
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
    let filter = make_filter(ProviderKind::OpenAi);
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
    let filter = make_filter(ProviderKind::OpenAi);
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
    let filter = make_filter(ProviderKind::OpenAi);
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
    let filter = make_filter(ProviderKind::OpenAi);
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

    let (input, output, total) = run_json_extraction(ProviderKind::OpenAi, json).await;

    assert_eq!(input.as_deref(), Some("15"), "OpenAI input tokens");
    assert_eq!(output.as_deref(), Some("42"), "OpenAI output tokens");
    assert_eq!(total.as_deref(), Some("57"), "OpenAI total tokens");
}

#[tokio::test]
async fn json_openai_responses_extracts_tokens() {
    let json =
        br#"{"id":"resp_123","object":"response","usage":{"input_tokens":15,"output_tokens":42,"total_tokens":57}}"#;

    let (input, output, total) = run_json_extraction(ProviderKind::OpenAi, json).await;

    assert_eq!(input.as_deref(), Some("15"), "Responses API input tokens");
    assert_eq!(output.as_deref(), Some("42"), "Responses API output tokens");
    assert_eq!(total.as_deref(), Some("57"), "Responses API total tokens");
}

#[tokio::test]
async fn json_anthropic_extracts_tokens() {
    let json = br#"{"usage":{"input_tokens":15,"output_tokens":42}}"#;

    let (input, output, total) = run_json_extraction(ProviderKind::Anthropic, json).await;

    assert_eq!(input.as_deref(), Some("15"), "Anthropic input tokens");
    assert_eq!(output.as_deref(), Some("42"), "Anthropic output tokens");
    assert_eq!(total.as_deref(), Some("57"), "Anthropic total tokens (computed)");
}

#[tokio::test]
async fn json_anthropic_includes_prompt_cache_tokens() {
    let json = br#"{"usage":{"input_tokens":50,"output_tokens":100,"cache_creation_input_tokens":1000,"cache_read_input_tokens":5000}}"#;

    let (input, output, total) = run_json_extraction(ProviderKind::Anthropic, json).await;

    assert_eq!(input.as_deref(), Some("6050"), "all Anthropic input token classes");
    assert_eq!(output.as_deref(), Some("100"), "Anthropic output tokens");
    assert_eq!(total.as_deref(), Some("6150"), "Anthropic total tokens");
}

#[tokio::test]
async fn json_anthropic_saturates_token_totals() {
    let json = format!(
        r#"{{"usage":{{"input_tokens":{},"output_tokens":1,"cache_creation_input_tokens":1}}}}"#,
        u64::MAX
    );

    let (input, output, total) = run_json_extraction(ProviderKind::Anthropic, json.as_bytes()).await;
    let max = u64::MAX.to_string();

    assert_eq!(
        input.as_deref(),
        Some(max.as_str()),
        "input token addition must saturate"
    );
    assert_eq!(output.as_deref(), Some("1"), "Anthropic output tokens");
    assert_eq!(
        total.as_deref(),
        Some(max.as_str()),
        "total token addition must saturate"
    );
}

#[tokio::test]
async fn json_google_extracts_tokens() {
    let json = br#"{"usageMetadata":{"promptTokenCount":15,"candidatesTokenCount":42,"totalTokenCount":57}}"#;

    let (input, output, total) = run_json_extraction(ProviderKind::Google, json).await;

    assert_eq!(input.as_deref(), Some("15"), "Google input tokens");
    assert_eq!(output.as_deref(), Some("42"), "Google output tokens");
    assert_eq!(total.as_deref(), Some("57"), "Google total tokens");
}

#[tokio::test]
async fn json_bedrock_converse_extracts_tokens() {
    let json = br#"{"usage":{"inputTokens":15,"outputTokens":42,"totalTokens":57}}"#;

    let (input, output, total) = run_json_extraction(ProviderKind::Bedrock, json).await;

    assert_eq!(input.as_deref(), Some("15"), "Bedrock input tokens");
    assert_eq!(output.as_deref(), Some("42"), "Bedrock output tokens");
    assert_eq!(total.as_deref(), Some("57"), "Bedrock total tokens");
}

#[tokio::test]
async fn json_bedrock_converse_includes_prompt_cache_tokens() {
    let json = br#"{"usage":{"inputTokens":9,"outputTokens":214,"cacheReadInputTokens":1066,"totalTokens":1289}}"#;

    let (input, output, total) = run_json_extraction(ProviderKind::Bedrock, json).await;

    assert_eq!(
        input.as_deref(),
        Some("1075"),
        "uncached 9 + cache read 1066, matching Anthropic normalization"
    );
    assert_eq!(output.as_deref(), Some("214"), "Bedrock output tokens");
    assert_eq!(total.as_deref(), Some("1289"), "Bedrock total tokens include the cache");
}

#[tokio::test]
async fn json_azure_extracts_tokens() {
    let json = br#"{"usage":{"prompt_tokens":5,"completion_tokens":10,"total_tokens":15}}"#;

    let (input, output, total) = run_json_extraction(ProviderKind::Azure, json).await;

    assert_eq!(input.as_deref(), Some("5"), "Azure input tokens");
    assert_eq!(output.as_deref(), Some("10"), "Azure output tokens");
    assert_eq!(total.as_deref(), Some("15"), "Azure total tokens");
}

#[tokio::test]
async fn json_missing_usage_sets_nothing() {
    let json = br#"{"id":"abc","choices":[]}"#;

    let (input, output, total) = run_json_extraction(ProviderKind::OpenAi, json).await;

    assert!(input.is_none(), "missing usage should not set input");
    assert!(output.is_none(), "missing usage should not set output");
    assert!(total.is_none(), "missing usage should not set total");
}

#[tokio::test]
async fn json_malformed_sets_nothing() {
    let (input, output, total) = run_json_extraction(ProviderKind::OpenAi, b"not json").await;

    assert!(input.is_none(), "malformed JSON should not set input");
    assert!(output.is_none(), "malformed JSON should not set output");
    assert!(total.is_none(), "malformed JSON should not set total");
}

#[tokio::test]
async fn json_empty_body_sets_nothing() {
    let (input, output, total) = run_json_extraction(ProviderKind::OpenAi, b"").await;

    assert!(input.is_none(), "empty body should not set input");
    assert!(output.is_none(), "empty body should not set output");
    assert!(total.is_none(), "empty body should not set total");
}

#[tokio::test]
async fn json_chunked_body_reassembled() {
    let filter = make_filter(ProviderKind::OpenAi);
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
    let filter = make_filter(ProviderKind::OpenAi);
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

    let (input, output, total) = run_sse_extraction(ProviderKind::OpenAi, events).await;

    assert_eq!(input.as_deref(), Some("10"), "OpenAI SSE input tokens");
    assert_eq!(output.as_deref(), Some("20"), "OpenAI SSE output tokens");
    assert_eq!(total.as_deref(), Some("30"), "OpenAI SSE total tokens");
}

#[tokio::test]
async fn sse_openai_responses_final_usage_event() {
    let events = concat!(
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hi\"}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":10,\"output_tokens\":20,\"total_tokens\":30}}}\n\n",
    );

    let (input, output, total) = run_sse_extraction(ProviderKind::OpenAi, events.as_bytes()).await;

    assert_eq!(input.as_deref(), Some("10"), "Responses API SSE input tokens");
    assert_eq!(output.as_deref(), Some("20"), "Responses API SSE output tokens");
    assert_eq!(total.as_deref(), Some("30"), "Responses API SSE total tokens");
}

#[tokio::test]
async fn sse_anthropic_accumulated_events() {
    let events = concat!(
        "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":25}}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"delta\":{\"text\":\"Hi\"}}\n\n",
        "data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":42}}\n\n",
    );

    let (input, output, total) = run_sse_extraction(ProviderKind::Anthropic, events.as_bytes()).await;

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
async fn sse_anthropic_includes_prompt_cache_tokens() {
    let events = concat!(
        "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":10,\"cache_creation_input_tokens\":100,\"cache_read_input_tokens\":500}}}\n\n",
        "data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":42}}\n\n",
    );

    let (input, output, total) = run_sse_extraction(ProviderKind::Anthropic, events.as_bytes()).await;

    assert_eq!(input.as_deref(), Some("610"), "all Anthropic SSE input token classes");
    assert_eq!(output.as_deref(), Some("42"), "Anthropic SSE output tokens");
    assert_eq!(total.as_deref(), Some("652"), "Anthropic SSE total tokens");
}

#[tokio::test]
async fn sse_google_final_usage_event() {
    let events = b"data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hi\"}]}}]}\n\ndata: {\"usageMetadata\":{\"promptTokenCount\":15,\"candidatesTokenCount\":42,\"totalTokenCount\":57}}\n\n";

    let (input, output, total) = run_sse_extraction(ProviderKind::Google, events).await;

    assert_eq!(input.as_deref(), Some("15"), "Google SSE input tokens");
    assert_eq!(output.as_deref(), Some("42"), "Google SSE output tokens");
    assert_eq!(total.as_deref(), Some("57"), "Google SSE total tokens");
}

#[tokio::test]
async fn sse_done_sentinel_ignored() {
    let events =
        b"data: {\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":20,\"total_tokens\":30}}\n\ndata: [DONE]\n\n";

    let (input, output, _) = run_sse_extraction(ProviderKind::OpenAi, events).await;

    assert_eq!(input.as_deref(), Some("10"), "[DONE] should not overwrite usage data");
    assert_eq!(output.as_deref(), Some("20"), "[DONE] should not overwrite usage data");
}

#[tokio::test]
async fn sse_chunks_split_across_calls() {
    let filter = make_filter(ProviderKind::OpenAi);
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

    let (input, output, total) = run_sse_extraction(ProviderKind::OpenAi, events).await;

    assert!(input.is_none(), "no usage events should not set input");
    assert!(output.is_none(), "no usage events should not set output");
    assert!(total.is_none(), "no usage events should not set total");
}

#[tokio::test]
async fn sse_clears_working_metadata() {
    let events = b"data: {\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":20,\"total_tokens\":30}}\n\n";
    let filter = make_filter(ProviderKind::OpenAi);
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

    let (input, output, total) = run_sse_extraction(ProviderKind::Bedrock, events).await;

    assert_eq!(input.as_deref(), Some("30"), "Bedrock SSE input tokens");
    assert_eq!(output.as_deref(), Some("18"), "Bedrock SSE output tokens");
    assert_eq!(total.as_deref(), Some("48"), "Bedrock SSE total tokens (computed)");
}

#[tokio::test]
async fn sse_bedrock_metadata_includes_prompt_cache_tokens() {
    let events =
        b"data: {\"metadata\":{\"usage\":{\"inputTokens\":9,\"outputTokens\":214,\"cacheReadInputTokens\":1066}}}\n\n";

    let (input, output, total) = run_sse_extraction(ProviderKind::Bedrock, events).await;

    assert_eq!(
        input.as_deref(),
        Some("1075"),
        "uncached 9 + cache read 1066, matching Anthropic normalization"
    );
    assert_eq!(output.as_deref(), Some("214"), "Bedrock SSE output tokens");
    assert_eq!(
        total.as_deref(),
        Some("1289"),
        "computed SSE total must include folded cache tokens"
    );
}

#[tokio::test]
async fn sse_zero_token_counts_are_written() {
    let events = b"data: {\"usage\":{\"prompt_tokens\":0,\"completion_tokens\":0,\"total_tokens\":0}}\n\n";

    let (input, output, total) = run_sse_extraction(ProviderKind::OpenAi, events).await;

    assert_eq!(input.as_deref(), Some("0"), "zero input tokens should be written");
    assert_eq!(output.as_deref(), Some("0"), "zero output tokens should be written");
    assert_eq!(total.as_deref(), Some("0"), "zero total tokens should be written");
}

#[tokio::test]
async fn sse_final_event_without_trailing_blank_line() {
    let events =
        b"data: {\"usageMetadata\":{\"promptTokenCount\":12,\"candidatesTokenCount\":34,\"totalTokenCount\":46}}";

    let (input, output, total) = run_sse_extraction(ProviderKind::Google, events).await;

    assert_eq!(input.as_deref(), Some("12"), "Google EOF usage should be extracted");
    assert_eq!(output.as_deref(), Some("34"), "Google EOF usage should be extracted");
    assert_eq!(total.as_deref(), Some("46"), "Google EOF total should be extracted");
}

// -----------------------------------------------------------------------------
// SSE: Scratch Overflow
// -----------------------------------------------------------------------------

/// Regression test for #674: an oversized event must not disable
/// extraction for the rest of the stream. The scanner recovers at the
/// next event boundary, so a terminal usage event arriving afterward is
/// still captured and treated as authoritative (no overflow status).
#[tokio::test]
async fn sse_oversized_event_recovers_and_captures_terminal_usage() {
    let mut events = b"data: ".to_vec();
    events.extend(std::iter::repeat_n(b'x', DEFAULT_MAX_SCRATCH_BYTES + 1));
    events.extend_from_slice(b"\n\n");
    events.extend_from_slice(
        b"data: {\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":20,\"total_tokens\":30}}\n\n",
    );

    let filter = make_filter(ProviderKind::OpenAi);
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let mut resp = make_response_with_content_type("text/event-stream");
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());
    ctx.response_header = None;

    let mut body = Some(Bytes::from(events));
    drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());

    assert_eq!(
        ctx.get_metadata("token.input"),
        Some("10"),
        "terminal usage should be captured after the oversized event is discarded"
    );
    assert_eq!(ctx.get_metadata("token.output"), Some("20"));
    assert_eq!(ctx.get_metadata("token.total"), Some("30"));
    assert!(
        ctx.get_metadata("token.status").is_none(),
        "recovered terminal usage after a dropped content event is authoritative"
    );
}

#[tokio::test]
async fn sse_overflow_does_not_finalize_mid_stream() {
    let filter = make_filter(ProviderKind::OpenAi);
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

    assert!(
        ctx.get_metadata("token.input").is_none(),
        "overflow mid-stream should not finalize early; already-accumulated counts wait for end_of_stream"
    );

    let mut body3 = Some(Bytes::from_static(b"\n\n"));
    drop(filter.on_response_body(&mut ctx, &mut body3, true).unwrap());

    assert_eq!(ctx.get_metadata("token.input"), Some("10"));
    assert_eq!(ctx.get_metadata("token.output"), Some("20"));
    assert_eq!(ctx.get_metadata("token.total"), Some("30"));
    assert_eq!(
        ctx.get_metadata("token.status"),
        Some("overflow"),
        "a drop after captured usage means later events (possibly a usage correction) were lost"
    );
    assert_no_working_metadata(&ctx);
}

#[tokio::test]
async fn sse_overflow_with_no_usage_sets_overflow_status() {
    let filter = make_filter(ProviderKind::OpenAi);
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let mut resp = make_response_with_content_type("text/event-stream");
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());
    ctx.response_header = None;

    let overflow_chunk = vec![b'x'; DEFAULT_MAX_SCRATCH_BYTES + 1];
    let mut body = Some(Bytes::from(overflow_chunk));
    drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());

    assert!(
        ctx.get_metadata("token.input").is_none(),
        "no usage was ever captured, so no counts should be written"
    );
    assert_eq!(
        ctx.get_metadata("token.status"),
        Some("overflow"),
        "billing consumers must not mistake this for zero usage"
    );
}

/// Anthropic/Bedrock split usage across events. If partial counts were
/// stored and the terminal usage event itself overflows, emit the
/// partials *and* `token.status = overflow` so they are not treated as
/// a complete capture.
#[tokio::test]
async fn sse_partial_then_oversized_terminal_event_sets_overflow_status() {
    let mut filter = make_filter(ProviderKind::Anthropic);
    filter.max_scratch_bytes = 256;
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/messages");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let mut resp = make_response_with_content_type("text/event-stream");
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());
    ctx.response_header = None;

    let mut events = b"data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":25}}}\n\n".to_vec();
    events.extend_from_slice(b"data: ");
    events.extend(std::iter::repeat_n(b'x', 257));
    events.extend_from_slice(b"\n\n");
    let mut body = Some(Bytes::from(events));
    drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());

    assert_eq!(ctx.get_metadata("token.input"), Some("25"), "partial input kept");
    assert_eq!(
        ctx.get_metadata("token.output"),
        Some("0"),
        "terminal output was dropped"
    );
    assert_eq!(
        ctx.get_metadata("token.status"),
        Some("overflow"),
        "partial maxima are not a complete capture when the terminal event overflowed"
    );
}

#[tokio::test]
async fn sse_responses_completed_event_larger_than_64kib_captured_at_default_scratch() {
    let filter = make_filter(ProviderKind::OpenAi);
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let mut resp = make_response_with_content_type("text/event-stream");
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());
    ctx.response_header = None;

    let mut body = Some(Bytes::from(responses_stream_with_padded_completed_event(65_536 * 3)));
    drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());

    assert_eq!(ctx.get_metadata("token.input"), Some("62000"));
    assert_eq!(ctx.get_metadata("token.output"), Some("13000"));
    assert_eq!(ctx.get_metadata("token.total"), Some("75000"));
    assert!(
        ctx.get_metadata("token.status").is_none(),
        "a complete capture at the default scratch is not an overflow"
    );
}

#[tokio::test]
async fn sse_responses_completed_event_beyond_scratch_marks_overflow() {
    let mut filter = make_filter(ProviderKind::OpenAi);
    filter.max_scratch_bytes = 65_536;
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let mut resp = make_response_with_content_type("text/event-stream");
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());
    ctx.response_header = None;

    let mut body = Some(Bytes::from(responses_stream_with_padded_completed_event(65_536 * 3)));
    drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());

    assert!(
        ctx.get_metadata("token.input").is_none(),
        "usage exists only in the oversized terminal event, which must be skipped"
    );
    assert_eq!(
        ctx.get_metadata("token.status"),
        Some("overflow"),
        "skipping the usage-carrying event must be an explicit overflow, not a silent zero"
    );
}

// -----------------------------------------------------------------------------
// Body Mode Without on_response
// -----------------------------------------------------------------------------

#[test]
fn on_response_body_noop_without_mode() {
    let filter = make_filter(ProviderKind::OpenAi);
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

/// Regression test for #674: overflow must not be indistinguishable
/// from a genuine zero-usage response.
#[tokio::test]
async fn json_overflow_sets_explicit_status_not_zero() {
    let filter = make_filter(ProviderKind::OpenAi);
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
        "overflow should not set counts"
    );
    assert_eq!(
        ctx.get_metadata("token.status"),
        Some("overflow"),
        "billing consumers must not mistake missing counts for zero usage"
    );
    assert_no_working_metadata(&ctx);
}

/// A valid response with usage near the end but larger than the
/// configured limit exhibits the same overflow signal, not silence.
#[tokio::test]
async fn json_valid_usage_beyond_configured_limit_sets_overflow_status() {
    let mut filter = make_filter(ProviderKind::OpenAi);
    filter.max_body_bytes = 64;
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let mut resp = make_response_with_content_type("application/json");
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());
    ctx.response_header = None;

    let padding = "x".repeat(200);
    let json = format!(
        r#"{{"id":"resp-1","padding":"{padding}","usage":{{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}}}}"#
    );
    let mut body = Some(Bytes::from(json.into_bytes()));
    drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());

    assert!(ctx.get_metadata("token.input").is_none());
    assert_eq!(ctx.get_metadata("token.status"), Some("overflow"));
}

#[test]
fn max_body_bytes_configurable_via_yaml() {
    let config: serde_yaml::Value = serde_yaml::from_str("provider: openai\nmax_body_bytes: 4096").unwrap();
    let filter = TokenCountFilter::from_config(&config).unwrap();

    assert_eq!(filter.name(), "token_count");
}

#[test]
fn max_scratch_bytes_configurable_via_yaml() {
    let config: serde_yaml::Value = serde_yaml::from_str("provider: openai\nmax_scratch_bytes: 512").unwrap();
    let filter = TokenCountFilter::from_config(&config).unwrap();

    assert_eq!(filter.name(), "token_count");
}

#[test]
fn from_config_rejects_zero_max_body_bytes() {
    let config: serde_yaml::Value = serde_yaml::from_str("provider: openai\nmax_body_bytes: 0").unwrap();
    match TokenCountFilter::from_config(&config) {
        Err(err) => assert!(
            err.to_string().contains("max_body_bytes"),
            "should reject zero max_body_bytes: {err}"
        ),
        Ok(_) => panic!("should reject zero max_body_bytes"),
    }
}

#[test]
fn from_config_rejects_zero_max_scratch_bytes() {
    let config: serde_yaml::Value = serde_yaml::from_str("provider: openai\nmax_scratch_bytes: 0").unwrap();
    match TokenCountFilter::from_config(&config) {
        Err(err) => assert!(
            err.to_string().contains("max_scratch_bytes"),
            "should reject zero max_scratch_bytes: {err}"
        ),
        Ok(_) => panic!("should reject zero max_scratch_bytes"),
    }
}

#[test]
fn from_config_rejects_max_body_bytes_above_ceiling() {
    let config: serde_yaml::Value = serde_yaml::from_str("provider: openai\nmax_body_bytes: 67108865").unwrap();
    match TokenCountFilter::from_config(&config) {
        Err(err) => assert!(
            err.to_string().contains("exceeds maximum"),
            "should reject max_body_bytes above 64 MiB: {err}"
        ),
        Ok(_) => panic!("should reject max_body_bytes above 64 MiB"),
    }
}

#[test]
fn from_config_rejects_max_scratch_bytes_above_ceiling() {
    let config: serde_yaml::Value = serde_yaml::from_str("provider: openai\nmax_scratch_bytes: 67108865").unwrap();
    match TokenCountFilter::from_config(&config) {
        Err(err) => assert!(
            err.to_string().contains("max_scratch_bytes") && err.to_string().contains("exceeds maximum"),
            "should reject max_scratch_bytes above 64 MiB: {err}"
        ),
        Ok(_) => panic!("should reject max_scratch_bytes above 64 MiB"),
    }
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
// Bedrock InvokeModel: Header-Only Path
// -----------------------------------------------------------------------------

#[tokio::test]
async fn bedrock_invoke_model_extracts_headers_on_response() {
    let filter = make_filter(ProviderKind::BedrockInvokeModel);
    let req = crate::test_utils::make_request(http::Method::POST, "/model/amazon.titan/invoke");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let mut resp = crate::test_utils::make_response();
    resp.headers.insert(HEADER_BEDROCK_INPUT, "25".parse().unwrap());
    resp.headers.insert(HEADER_BEDROCK_OUTPUT, "50".parse().unwrap());
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());
    ctx.response_header = None;

    assert_eq!(
        ctx.get_metadata("token.input"),
        Some("25"),
        "Bedrock InvokeModel input tokens"
    );
    assert_eq!(
        ctx.get_metadata("token.output"),
        Some("50"),
        "Bedrock InvokeModel output tokens"
    );
    assert_eq!(
        ctx.get_metadata("token.total"),
        Some("75"),
        "Bedrock InvokeModel total tokens (computed)"
    );
}

#[tokio::test]
async fn bedrock_invoke_model_headers_absent_is_noop() {
    let filter = make_filter(ProviderKind::BedrockInvokeModel);
    let req = crate::test_utils::make_request(http::Method::POST, "/model/amazon.titan/invoke");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let mut resp = crate::test_utils::make_response();
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());
    ctx.response_header = None;

    assert!(
        ctx.get_metadata("token.input").is_none(),
        "missing headers should not set input"
    );
    assert!(
        ctx.get_metadata("token.output").is_none(),
        "missing headers should not set output"
    );
}

#[tokio::test]
async fn bedrock_invoke_model_partial_headers_is_noop() {
    let filter = make_filter(ProviderKind::BedrockInvokeModel);
    let req = crate::test_utils::make_request(http::Method::POST, "/model/amazon.titan/invoke");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let mut resp = crate::test_utils::make_response();
    resp.headers.insert(HEADER_BEDROCK_INPUT, "25".parse().unwrap());
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());
    ctx.response_header = None;

    assert!(
        ctx.get_metadata("token.input").is_none(),
        "only-input header should not set tokens"
    );
}

#[tokio::test]
async fn bedrock_invoke_model_non_numeric_headers_is_noop() {
    let filter = make_filter(ProviderKind::BedrockInvokeModel);
    let req = crate::test_utils::make_request(http::Method::POST, "/model/amazon.titan/invoke");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let mut resp = crate::test_utils::make_response();
    resp.headers.insert(HEADER_BEDROCK_INPUT, "abc".parse().unwrap());
    resp.headers.insert(HEADER_BEDROCK_OUTPUT, "50".parse().unwrap());
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());
    ctx.response_header = None;

    assert!(
        ctx.get_metadata("token.input").is_none(),
        "non-numeric input header should not set tokens"
    );
    assert!(
        ctx.get_metadata("token.output").is_none(),
        "non-numeric input header should also suppress the otherwise-valid output header"
    );
}

#[tokio::test]
async fn bedrock_invoke_model_skips_non_success_status() {
    let filter = make_filter(ProviderKind::BedrockInvokeModel);
    let req = crate::test_utils::make_request(http::Method::POST, "/model/amazon.titan/invoke");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let mut resp =
        make_response_with_status_and_content_type(http::StatusCode::INTERNAL_SERVER_ERROR, "application/json");
    resp.headers.insert(HEADER_BEDROCK_INPUT, "25".parse().unwrap());
    resp.headers.insert(HEADER_BEDROCK_OUTPUT, "50".parse().unwrap());
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());
    ctx.response_header = None;

    assert!(
        ctx.get_metadata("token.input").is_none(),
        "non-success status should skip extraction even with valid token headers present"
    );
    assert!(
        ctx.get_metadata("token.output").is_none(),
        "non-success status should skip extraction even with valid token headers present"
    );
}

#[tokio::test]
async fn bedrock_invoke_model_does_not_set_content_type_mode() {
    let filter = make_filter(ProviderKind::BedrockInvokeModel);
    let req = crate::test_utils::make_request(http::Method::POST, "/model/amazon.titan/invoke");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let mut resp = make_response_with_content_type("application/json");
    resp.headers.insert(HEADER_BEDROCK_INPUT, "25".parse().unwrap());
    resp.headers.insert(HEADER_BEDROCK_OUTPUT, "50".parse().unwrap());
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());
    ctx.response_header = None;

    assert!(
        ctx.get_metadata(META_MODE).is_none(),
        "header-only path should never set the body extraction mode"
    );
}

#[test]
fn bedrock_invoke_model_response_body_access_is_none() {
    let filter = make_filter(ProviderKind::BedrockInvokeModel);
    assert_eq!(filter.response_body_access(), BodyAccess::None);
}

#[test]
fn other_providers_response_body_access_is_read_only() {
    let filter = make_filter(ProviderKind::OpenAi);
    assert_eq!(filter.response_body_access(), BodyAccess::ReadOnly);
}

#[test]
fn on_response_body_noop_for_bedrock_invoke_model() {
    let filter = make_filter(ProviderKind::BedrockInvokeModel);
    let req = crate::test_utils::make_request(http::Method::POST, "/model/amazon.titan/invoke");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let mut body: Option<Bytes> = None;
    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();

    assert!(matches!(action, FilterAction::Continue));
    assert!(ctx.get_metadata("token.input").is_none());
}

// -----------------------------------------------------------------------------
// provider: auto — Dialect Resolution by Endpoint Path
// -----------------------------------------------------------------------------

#[test]
fn for_path_resolves_openai_dialects() {
    assert_eq!(
        ProviderKind::for_path("/v1/responses"),
        Some(ProviderKind::OpenAi),
        "Responses endpoint meters with the OpenAI parser"
    );
    assert_eq!(
        ProviderKind::for_path("/v1/chat/completions"),
        Some(ProviderKind::OpenAi)
    );
    assert_eq!(ProviderKind::for_path("/v1/completions"), Some(ProviderKind::OpenAi));
    assert_eq!(
        ProviderKind::for_path("/openai/deployments/gpt/chat/completions"),
        Some(ProviderKind::OpenAi),
        "Azure OpenAI paths share the OpenAI schema"
    );
}

#[test]
fn for_path_resolves_vendor_dialects() {
    assert_eq!(ProviderKind::for_path("/v1/messages"), Some(ProviderKind::Anthropic));
    assert_eq!(
        ProviderKind::for_path("/v1beta/models/gemini:generateContent"),
        Some(ProviderKind::Google)
    );
    assert_eq!(
        ProviderKind::for_path("/v1beta/models/gemini:streamGenerateContent"),
        Some(ProviderKind::Google)
    );
    assert_eq!(
        ProviderKind::for_path("/model/claude/converse"),
        Some(ProviderKind::Bedrock)
    );
    assert_eq!(
        ProviderKind::for_path("/model/claude/converse-stream"),
        Some(ProviderKind::Bedrock)
    );
    assert_eq!(
        ProviderKind::for_path("/model/titan/invoke"),
        Some(ProviderKind::BedrockInvokeModel)
    );
    assert_eq!(
        ProviderKind::for_path("/model/titan/invoke-with-response-stream"),
        Some(ProviderKind::BedrockInvokeModel)
    );
}

#[test]
fn for_path_skips_unmeterable_paths() {
    assert_eq!(ProviderKind::for_path("/v1/models"), None, "catalog has no usage");
    assert_eq!(ProviderKind::for_path("/health"), None);
    assert_eq!(
        ProviderKind::for_path("/v1/messages/count_tokens"),
        None,
        "token estimates are not inference and must not meter"
    );
}

#[test]
fn provider_kind_meta_roundtrips() {
    for kind in [
        ProviderKind::OpenAi,
        ProviderKind::Anthropic,
        ProviderKind::Google,
        ProviderKind::Bedrock,
        ProviderKind::BedrockInvokeModel,
        ProviderKind::Azure,
    ] {
        assert_eq!(
            ProviderKind::from_meta(kind.as_str()),
            Some(kind),
            "resolved kinds must survive the metadata round-trip"
        );
    }
    assert_eq!(
        ProviderKind::from_meta("auto"),
        None,
        "auto never stashes itself as a resolved kind"
    );
}

#[tokio::test]
async fn auto_sse_responses_wrapper_metered() {
    let events = concat!(
        "data: {\"type\":\"response.output_text.delta\",\"delta\":{\"text\":\"ok\"}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":54,\"output_tokens\":9,\"total_tokens\":63}}}\n\n",
        "data: [DONE]\n\n",
    );

    let (input, output, total) = run_sse_extraction_at(ProviderKind::Auto, "/v1/responses", events.as_bytes()).await;

    assert_eq!(
        input.as_deref(),
        Some("54"),
        "Responses-API SSE wrapper must meter via the OpenAI parser on a mixed route"
    );
    assert_eq!(output.as_deref(), Some("9"));
    assert_eq!(total.as_deref(), Some("63"));
}

#[tokio::test]
async fn auto_sse_messages_anthropic_partials_metered() {
    let events = concat!(
        "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":25}}}\n\n",
        "data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":42}}\n\n",
    );

    let (input, output, total) = run_sse_extraction_at(ProviderKind::Auto, "/v1/messages", events.as_bytes()).await;

    assert_eq!(
        input.as_deref(),
        Some("25"),
        "/messages must meter through Anthropic partial accumulation"
    );
    assert_eq!(output.as_deref(), Some("42"));
    assert_eq!(total.as_deref(), Some("67"));
}

#[tokio::test]
async fn auto_json_responses_direct_metered_with_cache() {
    let json = br#"{"id":"resp_abc","usage":{"input_tokens":1000,"output_tokens":50,"total_tokens":1050,"input_tokens_details":{"cached_tokens":900}}}"#;
    let filter = make_filter(ProviderKind::Auto);
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let mut resp = make_response_with_content_type("application/json");
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());
    ctx.response_header = None;

    let mut body = Some(Bytes::copy_from_slice(json));
    drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());

    assert_eq!(ctx.get_metadata("token.input"), Some("1000"));
    assert_eq!(ctx.get_metadata("token.output"), Some("50"));
    assert_eq!(
        ctx.get_metadata("token.cache_read"),
        Some("900"),
        "cached_tokens must be attributed through the OpenAI parser"
    );
    assert_no_working_metadata(&ctx);
}

#[tokio::test]
async fn auto_unmeterable_path_sets_nothing() {
    let json = br#"{"input_tokens":123}"#;
    let filter = make_filter(ProviderKind::Auto);
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/messages/count_tokens");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let mut resp = make_response_with_content_type("application/json");
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());
    ctx.response_header = None;

    let mut body = Some(Bytes::copy_from_slice(json));
    drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());

    assert!(
        ctx.get_metadata("token.input").is_none(),
        "token-count estimates must not be metered as inference"
    );
    assert_no_working_metadata(&ctx);
}

#[tokio::test]
async fn auto_bedrock_invoke_path_meters_headers() {
    let filter = make_filter(ProviderKind::Auto);
    let req = crate::test_utils::make_request(http::Method::POST, "/model/titan/invoke");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let mut resp = make_response_with_content_type("application/json");
    resp.headers
        .insert(HEADER_BEDROCK_INPUT, HeaderValue::from_static("25"));
    resp.headers
        .insert(HEADER_BEDROCK_OUTPUT, HeaderValue::from_static("50"));
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());
    ctx.response_header = None;

    assert_eq!(
        ctx.get_metadata("token.input"),
        Some("25"),
        "/invoke must resolve to the header-only provider"
    );
    assert_eq!(ctx.get_metadata("token.output"), Some("50"));
}

// -----------------------------------------------------------------------------
// Prompt Cache Breakdown
// -----------------------------------------------------------------------------

#[tokio::test]
async fn json_anthropic_records_cache_breakdown() {
    let json = br#"{"usage":{"input_tokens":50,"output_tokens":100,"cache_creation_input_tokens":1000,"cache_read_input_tokens":5000}}"#;

    let (cache_read, cache_write) = run_cache_extraction(ProviderKind::Anthropic, "application/json", json).await;

    assert_eq!(cache_read.as_deref(), Some("5000"), "Anthropic cache read tokens");
    assert_eq!(cache_write.as_deref(), Some("1000"), "Anthropic cache write tokens");
}

#[tokio::test]
async fn json_openai_records_cached_tokens() {
    let json =
        br#"{"usage":{"prompt_tokens":1000,"completion_tokens":50,"prompt_tokens_details":{"cached_tokens":900}}}"#;

    let (cache_read, cache_write) = run_cache_extraction(ProviderKind::OpenAi, "application/json", json).await;

    assert_eq!(cache_read.as_deref(), Some("900"), "OpenAI cached prompt tokens");
    assert_eq!(
        cache_write, None,
        "OpenAI has no cache write field, so no cache write metadata is written"
    );
}

#[tokio::test]
async fn json_openai_responses_records_cache_breakdown() {
    let json = br#"{"id":"resp_123","object":"response","usage":{"input_tokens":1000,"output_tokens":50,"input_tokens_details":{"cached_tokens":900,"cache_write_tokens":80}}}"#;

    let (cache_read, cache_write) = run_cache_extraction(ProviderKind::OpenAi, "application/json", json).await;

    assert_eq!(cache_read.as_deref(), Some("900"), "Responses API cache reads");
    assert_eq!(cache_write.as_deref(), Some("80"), "Responses API cache writes");
}

#[tokio::test]
async fn json_bedrock_converse_records_cache_breakdown() {
    let json = br#"{"usage":{"inputTokens":9,"outputTokens":214,"cacheReadInputTokens":1066,"cacheWriteInputTokens":100,"totalTokens":1389}}"#;

    let (cache_read, cache_write) = run_cache_extraction(ProviderKind::Bedrock, "application/json", json).await;

    assert_eq!(cache_read.as_deref(), Some("1066"), "Converse cache read tokens");
    assert_eq!(cache_write.as_deref(), Some("100"), "Converse cache write tokens");
}

#[tokio::test]
async fn json_google_records_cached_content_tokens() {
    let json =
        br#"{"usageMetadata":{"promptTokenCount":1200,"candidatesTokenCount":40,"cachedContentTokenCount":1100}}"#;

    let (cache_read, cache_write) = run_cache_extraction(ProviderKind::Google, "application/json", json).await;

    assert_eq!(cache_read.as_deref(), Some("1100"), "Google cached content tokens");
    assert_eq!(
        cache_write, None,
        "Google has no cache write field, so no cache write metadata is written"
    );
}

#[tokio::test]
async fn json_without_cache_records_no_metadata() {
    let json = br#"{"usage":{"prompt_tokens":15,"completion_tokens":42,"total_tokens":57}}"#;

    let (cache_read, cache_write) = run_cache_extraction(ProviderKind::OpenAi, "application/json", json).await;

    assert_eq!(
        cache_read, None,
        "a response with no cache information writes no cache read metadata"
    );
    assert_eq!(
        cache_write, None,
        "a response with no cache information writes no cache write metadata"
    );
}

#[tokio::test]
async fn json_with_zero_cached_tokens_records_zero() {
    let json = br#"{"usage":{"prompt_tokens":15,"completion_tokens":42,"prompt_tokens_details":{"cached_tokens":0}}}"#;

    let (cache_read, _cache_write) = run_cache_extraction(ProviderKind::OpenAi, "application/json", json).await;

    assert_eq!(
        cache_read.as_deref(),
        Some("0"),
        "a reported cache miss is recorded as zero, distinct from absent"
    );
}

#[tokio::test]
async fn sse_anthropic_records_cache_breakdown() {
    let events = concat!(
        "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":10,\"cache_creation_input_tokens\":100,\"cache_read_input_tokens\":500}}}\n\n",
        "data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":42}}\n\n",
    );

    let (cache_read, cache_write) =
        run_cache_extraction(ProviderKind::Anthropic, "text/event-stream", events.as_bytes()).await;

    assert_eq!(cache_read.as_deref(), Some("500"), "cache read from message_start");
    assert_eq!(cache_write.as_deref(), Some("100"), "cache write from message_start");
}

#[tokio::test]
async fn sse_anthropic_without_cache_records_no_metadata() {
    let events = concat!(
        "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":25}}}\n\n",
        "data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":42}}\n\n",
    );

    let (cache_read, cache_write) =
        run_cache_extraction(ProviderKind::Anthropic, "text/event-stream", events.as_bytes()).await;

    assert_eq!(
        cache_read, None,
        "a stream with no cache fields writes no cache read metadata"
    );
    assert_eq!(
        cache_write, None,
        "a stream with no cache fields writes no cache write metadata"
    );
}

#[tokio::test]
async fn sse_openai_final_usage_records_cached_tokens() {
    let events = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\n",
        "data: {\"usage\":{\"prompt_tokens\":1000,\"completion_tokens\":20,\"prompt_tokens_details\":{\"cached_tokens\":900}}}\n\n",
        "data: [DONE]\n\n",
    );

    let (cache_read, cache_write) =
        run_cache_extraction(ProviderKind::OpenAi, "text/event-stream", events.as_bytes()).await;

    assert_eq!(
        cache_read.as_deref(),
        Some("900"),
        "cached tokens from final usage event"
    );
    assert_eq!(
        cache_write, None,
        "OpenAI has no cache write field, so no cache write metadata is written"
    );
}

#[tokio::test]
async fn sse_openai_responses_records_cache_breakdown() {
    let events = concat!(
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hi\"}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":1000,\"output_tokens\":20,\"input_tokens_details\":{\"cached_tokens\":900,\"cache_write_tokens\":80}}}}\n\n",
    );

    let (cache_read, cache_write) =
        run_cache_extraction(ProviderKind::OpenAi, "text/event-stream", events.as_bytes()).await;

    assert_eq!(cache_read.as_deref(), Some("900"), "Responses API SSE cache reads");
    assert_eq!(cache_write.as_deref(), Some("80"), "Responses API SSE cache writes");
}

#[tokio::test]
async fn sse_bedrock_metadata_records_cache_breakdown() {
    let events =
        b"data: {\"metadata\":{\"usage\":{\"inputTokens\":9,\"outputTokens\":214,\"cacheReadInputTokens\":1066,\"cacheWriteInputTokens\":100}}}\n\n";

    let (cache_read, cache_write) = run_cache_extraction(ProviderKind::Bedrock, "text/event-stream", events).await;

    assert_eq!(cache_read.as_deref(), Some("1066"), "ConverseStream cache read tokens");
    assert_eq!(cache_write.as_deref(), Some("100"), "ConverseStream cache write tokens");
}

#[tokio::test]
async fn sse_google_final_usage_records_cached_tokens() {
    let events = concat!(
        "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hi\"}]}}]}\n\n",
        "data: {\"usageMetadata\":{\"promptTokenCount\":1200,\"candidatesTokenCount\":40,\"cachedContentTokenCount\":1100}}\n\n",
    );

    let (cache_read, cache_write) =
        run_cache_extraction(ProviderKind::Google, "text/event-stream", events.as_bytes()).await;

    assert_eq!(
        cache_read.as_deref(),
        Some("1100"),
        "cached content tokens from final usage event"
    );
    assert_eq!(
        cache_write, None,
        "Google has no cache write field, so no cache write metadata is written"
    );
}

#[tokio::test]
async fn bedrock_invoke_model_headers_record_no_cache() {
    let filter = make_filter(ProviderKind::BedrockInvokeModel);
    let req = crate::test_utils::make_request(http::Method::POST, "/model/claude/invoke");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let mut resp = make_response_with_content_type("application/json");
    resp.headers
        .insert(HEADER_BEDROCK_INPUT, HeaderValue::from_static("15"));
    resp.headers
        .insert(HEADER_BEDROCK_OUTPUT, HeaderValue::from_static("42"));
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());
    ctx.response_header = None;

    assert_eq!(ctx.get_metadata("token.input"), Some("15"), "header input tokens");
    assert!(
        ctx.get_metadata("token.cache_read").is_none(),
        "response headers carry no cache breakdown"
    );
    assert!(
        ctx.get_metadata("token.cache_write").is_none(),
        "response headers carry no cache breakdown"
    );
    assert!(
        ctx.get_metadata("token.reasoning").is_none(),
        "response headers carry no reasoning breakdown"
    );
}

// -----------------------------------------------------------------------------
// Reasoning / Thinking Token Breakdown
// -----------------------------------------------------------------------------

#[tokio::test]
async fn json_openai_records_reasoning_tokens() {
    let json = br#"{"usage":{"prompt_tokens":120,"completion_tokens":800,"completion_tokens_details":{"reasoning_tokens":640}}}"#;

    let reasoning = run_reasoning_extraction(ProviderKind::OpenAi, "application/json", json).await;

    assert_eq!(reasoning.as_deref(), Some("640"), "OpenAI reasoning tokens");
}

#[tokio::test]
async fn json_openai_responses_records_reasoning_tokens() {
    let json = br#"{"id":"resp_123","object":"response","usage":{"input_tokens":120,"output_tokens":800,"output_tokens_details":{"reasoning_tokens":640}}}"#;

    let reasoning = run_reasoning_extraction(ProviderKind::OpenAi, "application/json", json).await;

    assert_eq!(reasoning.as_deref(), Some("640"), "Responses API reasoning tokens");
}

#[tokio::test]
async fn json_azure_records_reasoning_tokens() {
    let json = br#"{"usage":{"prompt_tokens":120,"completion_tokens":800,"completion_tokens_details":{"reasoning_tokens":640}}}"#;

    let reasoning = run_reasoning_extraction(ProviderKind::Azure, "application/json", json).await;

    assert_eq!(
        reasoning.as_deref(),
        Some("640"),
        "Azure shares the OpenAI usage schema"
    );
}

#[tokio::test]
async fn json_anthropic_records_thinking_tokens() {
    let json =
        br#"{"usage":{"input_tokens":200,"output_tokens":1500,"output_tokens_details":{"thinking_tokens":1200}}}"#;

    let reasoning = run_reasoning_extraction(ProviderKind::Anthropic, "application/json", json).await;

    assert_eq!(reasoning.as_deref(), Some("1200"), "Anthropic thinking tokens");
}

#[tokio::test]
async fn json_google_records_thoughts_tokens() {
    let json =
        br#"{"usageMetadata":{"promptTokenCount":50,"candidatesTokenCount":80,"thoughtsTokenCount":200,"totalTokenCount":330}}"#;

    let reasoning = run_reasoning_extraction(ProviderKind::Google, "application/json", json).await;

    assert_eq!(reasoning.as_deref(), Some("200"), "Google thoughts tokens");
}

#[tokio::test]
async fn json_google_missing_total_includes_thoughts() {
    let json = br#"{"usageMetadata":{"promptTokenCount":50,"candidatesTokenCount":80,"thoughtsTokenCount":200}}"#;

    let (input, output, total) = run_json_extraction(ProviderKind::Google, json).await;

    assert_eq!(input.as_deref(), Some("50"));
    assert_eq!(output.as_deref(), Some("80"), "output stays candidatesTokenCount");
    assert_eq!(
        total.as_deref(),
        Some("330"),
        "fallback total includes thoughts when totalTokenCount is absent"
    );
}

#[tokio::test]
async fn json_without_reasoning_records_no_metadata() {
    let json = br#"{"usage":{"prompt_tokens":15,"completion_tokens":42,"total_tokens":57}}"#;

    let reasoning = run_reasoning_extraction(ProviderKind::OpenAi, "application/json", json).await;

    assert_eq!(
        reasoning, None,
        "a response with no reasoning information writes no reasoning metadata"
    );
}

#[tokio::test]
async fn json_with_zero_reasoning_tokens_records_zero() {
    let json =
        br#"{"usage":{"prompt_tokens":15,"completion_tokens":42,"completion_tokens_details":{"reasoning_tokens":0}}}"#;

    let reasoning = run_reasoning_extraction(ProviderKind::OpenAi, "application/json", json).await;

    assert_eq!(
        reasoning.as_deref(),
        Some("0"),
        "a reported zero is recorded as zero, distinct from absent"
    );
}

#[tokio::test]
async fn json_openai_reasoning_does_not_change_output_total() {
    let json = br#"{"usage":{"prompt_tokens":120,"completion_tokens":800,"total_tokens":920,"completion_tokens_details":{"reasoning_tokens":640}}}"#;

    let (input, output, total) = run_json_extraction(ProviderKind::OpenAi, json).await;

    assert_eq!(input.as_deref(), Some("120"));
    assert_eq!(output.as_deref(), Some("800"), "output stays completion_tokens");
    assert_eq!(total.as_deref(), Some("920"));
}

#[tokio::test]
async fn sse_openai_final_usage_records_reasoning_tokens() {
    let events = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\n",
        "data: {\"usage\":{\"prompt_tokens\":120,\"completion_tokens\":800,\"completion_tokens_details\":{\"reasoning_tokens\":640}}}\n\n",
        "data: [DONE]\n\n",
    );

    let reasoning = run_reasoning_extraction(ProviderKind::OpenAi, "text/event-stream", events.as_bytes()).await;

    assert_eq!(
        reasoning.as_deref(),
        Some("640"),
        "reasoning tokens from final usage event"
    );
}

#[tokio::test]
async fn sse_openai_responses_records_reasoning_tokens() {
    let events = concat!(
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hi\"}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":120,\"output_tokens\":800,\"output_tokens_details\":{\"reasoning_tokens\":640}}}}\n\n",
    );

    let reasoning = run_reasoning_extraction(ProviderKind::OpenAi, "text/event-stream", events.as_bytes()).await;

    assert_eq!(
        reasoning.as_deref(),
        Some("640"),
        "Responses API reasoning tokens from response.completed"
    );
}

#[tokio::test]
async fn sse_anthropic_records_thinking_tokens() {
    let events = concat!(
        "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":200}}}\n\n",
        "data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":1500,\"output_tokens_details\":{\"thinking_tokens\":1200}}}\n\n",
    );

    let reasoning = run_reasoning_extraction(ProviderKind::Anthropic, "text/event-stream", events.as_bytes()).await;

    assert_eq!(
        reasoning.as_deref(),
        Some("1200"),
        "thinking tokens from terminal message_delta"
    );
}

#[tokio::test]
async fn sse_anthropic_without_thinking_records_no_metadata() {
    let events = concat!(
        "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":25}}}\n\n",
        "data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":42}}\n\n",
    );

    let reasoning = run_reasoning_extraction(ProviderKind::Anthropic, "text/event-stream", events.as_bytes()).await;

    assert_eq!(
        reasoning, None,
        "a stream with no thinking field writes no reasoning metadata"
    );
}

#[tokio::test]
async fn sse_google_final_usage_records_thoughts_tokens() {
    let events = concat!(
        "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hi\"}]}}]}\n\n",
        "data: {\"usageMetadata\":{\"promptTokenCount\":50,\"candidatesTokenCount\":80,\"thoughtsTokenCount\":200,\"totalTokenCount\":330}}\n\n",
    );

    let reasoning = run_reasoning_extraction(ProviderKind::Google, "text/event-stream", events.as_bytes()).await;
    let (input, output, total) = run_sse_extraction(ProviderKind::Google, events.as_bytes()).await;

    assert_eq!(
        reasoning.as_deref(),
        Some("200"),
        "thoughts tokens from final usage event"
    );
    assert_eq!(input.as_deref(), Some("50"));
    assert_eq!(output.as_deref(), Some("80"), "output stays candidatesTokenCount");
    assert_eq!(
        total.as_deref(),
        Some("330"),
        "SSE total must keep the provider total, which includes thoughts"
    );
}

#[tokio::test]
async fn sse_google_missing_total_includes_thoughts() {
    let events = concat!(
        "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hi\"}]}}]}\n\n",
        "data: {\"usageMetadata\":{\"promptTokenCount\":50,\"candidatesTokenCount\":80,\"thoughtsTokenCount\":200}}\n\n",
    );

    let (input, output, total) = run_sse_extraction(ProviderKind::Google, events.as_bytes()).await;
    let reasoning = run_reasoning_extraction(ProviderKind::Google, "text/event-stream", events.as_bytes()).await;

    assert_eq!(reasoning.as_deref(), Some("200"));
    assert_eq!(input.as_deref(), Some("50"));
    assert_eq!(output.as_deref(), Some("80"));
    assert_eq!(
        total.as_deref(),
        Some("330"),
        "SSE fallback total must include thoughts when totalTokenCount is absent"
    );
}

#[tokio::test]
async fn json_bedrock_converse_records_no_reasoning() {
    let json = br#"{"usage":{"inputTokens":10,"outputTokens":20}}"#;

    let reasoning = run_reasoning_extraction(ProviderKind::Bedrock, "application/json", json).await;

    assert_eq!(reasoning, None, "Bedrock Converse has no documented reasoning field");
}

#[tokio::test]
async fn json_bedrock_anthropic_fallback_records_thinking_tokens() {
    let json =
        br#"{"usage":{"input_tokens":10,"output_tokens":1500,"output_tokens_details":{"thinking_tokens":1200}}}"#;

    let reasoning = run_reasoning_extraction(ProviderKind::Bedrock, "application/json", json).await;

    assert_eq!(
        reasoning.as_deref(),
        Some("1200"),
        "Claude via InvokeModel inherits Anthropic thinking extraction"
    );
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

use std::fmt::Write as _;

/// Builds a Responses-API stream whose terminal `response.completed` event
/// embeds an output object padded to `pad_bytes`, mirroring how agentic
/// clients (reasoning summaries, tool items) inflate that event in
/// production while the smaller events stay tiny.
fn responses_stream_with_padded_completed_event(pad_bytes: usize) -> Vec<u8> {
    let mut events =
        b"event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_pad\"}}\n\n"
            .to_vec();
    events.extend_from_slice(
        b"event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\n\n",
    );
    events.extend_from_slice(b"event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_pad\",\"output\":[{\"type\":\"output_text\",\"text\":\"");
    events.extend(std::iter::repeat_n(b'x', pad_bytes));
    events.extend_from_slice(
        b"\"}],\"usage\":{\"input_tokens\":62000,\"output_tokens\":13000,\"total_tokens\":75000}}}\n\n",
    );
    events
}

fn make_filter(provider: ProviderKind) -> TokenCountFilter {
    TokenCountFilter {
        provider,
        max_body_bytes: DEFAULT_MAX_BODY_BYTES,
        max_scratch_bytes: DEFAULT_MAX_SCRATCH_BYTES,
    }
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
    provider: ProviderKind,
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

/// `StreamBuffer` chains deliver each raw chunk to filters *and* re-deliver the
/// frozen full buffer at end-of-stream. JSON extraction must read the EOS
/// buffer alone, not the concatenation of both views.
#[tokio::test]
async fn json_stream_buffer_double_delivery_extracts_once() {
    let json = br#"{"usage":{"input_tokens":15,"output_tokens":42}}"#;
    let filter = make_filter(ProviderKind::Anthropic);
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/messages");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let mut resp = make_response_with_content_type("application/json");
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());
    ctx.response_header = None;

    // The runtime upgrade a dialect-translation filter performs for JSON responses.
    ctx.response_body_mode = BodyMode::StreamBuffer { max_bytes: None };

    let raw = Bytes::copy_from_slice(json);
    let mut chunk = Some(raw.clone());
    drop(filter.on_response_body(&mut ctx, &mut chunk, false).unwrap());

    let mut frozen = Some(raw);
    drop(filter.on_response_body(&mut ctx, &mut frozen, true).unwrap());

    assert_eq!(
        ctx.get_metadata("token.input"),
        Some("15"),
        "the EOS frozen buffer must be parsed exactly once"
    );
    assert_eq!(
        ctx.get_metadata("token.output"),
        Some("42"),
        "output tokens should match"
    );
}

/// When a buffered chain releases its buffer mid-stream, the EOS call carries
/// no body; extraction must then fall back to the hex-accumulated raw chunks.
#[tokio::test]
async fn json_stream_buffer_released_falls_back_to_chunks() {
    let json = br#"{"usage":{"input_tokens":7,"output_tokens":11}}"#;
    let filter = make_filter(ProviderKind::Anthropic);
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/messages");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let mut resp = make_response_with_content_type("application/json");
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());
    ctx.response_header = None;

    ctx.response_body_mode = BodyMode::StreamBuffer { max_bytes: None };

    let mid = json.len() / 2;
    let mut first = Some(Bytes::copy_from_slice(&json[..mid]));
    drop(filter.on_response_body(&mut ctx, &mut first, false).unwrap());
    let mut second = Some(Bytes::copy_from_slice(&json[mid..]));
    drop(filter.on_response_body(&mut ctx, &mut second, false).unwrap());

    let mut eos: Option<Bytes> = None;
    drop(filter.on_response_body(&mut ctx, &mut eos, true).unwrap());

    assert_eq!(
        ctx.get_metadata("token.input"),
        Some("7"),
        "released buffers must still extract from the streamed chunks"
    );
    assert_eq!(
        ctx.get_metadata("token.output"),
        Some("11"),
        "output tokens should match"
    );
}

/// Run a full `on_response` -> `on_response_body` cycle and return the prompt
/// cache breakdown metadata for either the JSON or the SSE path.
async fn run_cache_extraction(
    provider: ProviderKind,
    content_type: &str,
    body_bytes: &[u8],
) -> (Option<String>, Option<String>) {
    let filter = make_filter(provider);
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let mut resp = make_response_with_content_type(content_type);
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());
    ctx.response_header = None;

    let mut body = Some(Bytes::copy_from_slice(body_bytes));
    drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());

    (
        ctx.get_metadata("token.cache_read").map(str::to_owned),
        ctx.get_metadata("token.cache_write").map(str::to_owned),
    )
}

/// Run a full `on_response` -> `on_response_body` cycle and return the
/// reasoning metadata for either the JSON or the SSE path.
async fn run_reasoning_extraction(provider: ProviderKind, content_type: &str, body_bytes: &[u8]) -> Option<String> {
    let filter = make_filter(provider);
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let mut resp = make_response_with_content_type(content_type);
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());
    ctx.response_header = None;

    let mut body = Some(Bytes::copy_from_slice(body_bytes));
    drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());

    ctx.get_metadata("token.reasoning").map(str::to_owned)
}

/// Run a full `on_response` -> `on_response_body` cycle for SSE extraction.
async fn run_sse_extraction(
    provider: ProviderKind,
    sse_bytes: &[u8],
) -> (Option<String>, Option<String>, Option<String>) {
    run_sse_extraction_at(provider, "/v1/chat/completions", sse_bytes).await
}

/// Run a full `on_response` -> `on_response_body` SSE cycle against a
/// specific request path, which is what `provider: auto` resolves its
/// parser from.
async fn run_sse_extraction_at(
    provider: ProviderKind,
    path: &str,
    sse_bytes: &[u8],
) -> (Option<String>, Option<String>, Option<String>) {
    let filter = make_filter(provider);
    let req = crate::test_utils::make_request(http::Method::POST, path);
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

// -----------------------------------------------------------------------------
// LLM gateway metrics
// -----------------------------------------------------------------------------

#[test]
#[expect(clippy::too_many_lines, reason = "test asserts many metrics")]
fn publish_token_usage_records_llm_metrics() {
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.cluster = Some("cb-glm".to_owned().into());
    ctx.set_metadata("openai_chat_completions_request.model", "gpt-x");

    let usage = TokenUsage::new(12, 34, Some(46))
        .with_cache(Some(5), None)
        .with_reasoning(Some(2));

    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    metrics::with_local_recorder(&recorder, || publish_token_usage(&mut ctx, usage));
    let snap = snapshotter.snapshot().into_vec();

    // Sum only series carrying the expected model/cluster labels so a
    // resolver regression (model or cluster collapsing to "unknown") fails
    // the totals below instead of passing on unlabeled counters.
    let labeled_total = |name: &str| -> u64 {
        snap.iter()
            .filter(|(k, ..)| {
                k.key().name() == name
                    && k.key().labels().any(|l| l.key() == "model" && l.value() == "gpt-x")
                    && k.key().labels().any(|l| l.key() == "cluster" && l.value() == "cb-glm")
            })
            .filter_map(|(_, _, _, v)| match v {
                DebugValue::Counter(c) => Some(*c),
                _ => None,
            })
            .sum()
    };
    let present = |name: &str| snap.iter().any(|(k, ..)| k.key().name() == name);

    assert_eq!(labeled_total(crate::llm_metrics::METRIC_PROMPT_TOKENS_TOTAL), 12);
    assert_eq!(labeled_total(crate::llm_metrics::METRIC_OUTPUT_TOKENS_TOTAL), 34);
    assert_eq!(labeled_total(crate::llm_metrics::METRIC_CACHE_READ_TOKENS_TOTAL), 5);
    assert_eq!(labeled_total(crate::llm_metrics::METRIC_REASONING_TOKENS_TOTAL), 2);
    assert!(
        !present(crate::llm_metrics::METRIC_EMPTY_COMPLETIONS_TOTAL),
        "empty_completions must not be recorded on a non-empty response"
    );
}

#[test]
fn publish_token_usage_bumps_empty_completions_on_zero_output() {
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.set_metadata(crate::llm_metrics::META_MODEL_TO_HEADER_MODEL, "gpt-x");

    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    metrics::with_local_recorder(&recorder, || {
        publish_token_usage(&mut ctx, TokenUsage::new(12, 0, Some(12)));
    });
    let snap = snapshotter.snapshot().into_vec();

    let total = |name: &str| -> u64 {
        snap.iter()
            .filter(|(k, ..)| k.key().name() == name)
            .filter_map(|(_, _, _, v)| match v {
                DebugValue::Counter(c) => Some(*c),
                _ => None,
            })
            .sum()
    };

    assert_eq!(total(crate::llm_metrics::METRIC_EMPTY_COMPLETIONS_TOTAL), 1);
}

#[test]
fn publish_token_usage_excludes_non_generation_terminations_from_empty_completions() {
    for reason in ["tool_calls", "refusal", "content_filter"] {
        let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
        let mut ctx = crate::test_utils::make_filter_context(&req);
        ctx.set_metadata(crate::llm_metrics::META_MODEL_TO_HEADER_MODEL, "gpt-x");
        ctx.set_metadata(META_FINISH_RECORDED, reason);

        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        metrics::with_local_recorder(&recorder, || {
            publish_token_usage(&mut ctx, TokenUsage::new(12, 0, Some(12)));
        });
        let snap = snapshotter.snapshot().into_vec();

        assert!(
            !snap
                .iter()
                .any(|(key, ..)| key.key().name() == crate::llm_metrics::METRIC_EMPTY_COMPLETIONS_TOTAL),
            "empty_completions must not count {reason} terminations"
        );
    }
}

#[test]
#[expect(clippy::too_many_lines, reason = "test asserts many metrics")]
fn bedrock_headers_record_llm_metrics() {
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.cluster = Some("bedrock".to_owned().into());
    ctx.set_metadata(crate::llm_metrics::META_MODEL_TO_HEADER_MODEL, "claude-x");
    let mut headers = http::HeaderMap::new();
    headers.insert(HEADER_BEDROCK_INPUT, HeaderValue::from_static("40"));
    headers.insert(HEADER_BEDROCK_OUTPUT, HeaderValue::from_static("7"));
    let mut resp = Response {
        headers,
        status: http::StatusCode::OK,
    };
    ctx.response_header = Some(&mut resp);

    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    metrics::with_local_recorder(&recorder, || extract_bedrock_headers(&mut ctx));
    let snap = snapshotter.snapshot().into_vec();

    // Sum only series carrying the expected model/cluster labels so a
    // resolver regression (model or cluster collapsing to "unknown") fails
    // the totals below instead of passing on unlabeled counters.
    let labeled_total = |name: &str| -> u64 {
        snap.iter()
            .filter(|(k, ..)| {
                k.key().name() == name
                    && k.key().labels().any(|l| l.key() == "model" && l.value() == "claude-x")
                    && k.key().labels().any(|l| l.key() == "cluster" && l.value() == "bedrock")
            })
            .filter_map(|(_, _, _, v)| match v {
                DebugValue::Counter(c) => Some(*c),
                _ => None,
            })
            .sum()
    };
    let present = |name: &str| snap.iter().any(|(k, ..)| k.key().name() == name);

    assert_eq!(labeled_total(crate::llm_metrics::METRIC_PROMPT_TOKENS_TOTAL), 40);
    assert_eq!(labeled_total(crate::llm_metrics::METRIC_OUTPUT_TOKENS_TOTAL), 7);
    assert!(
        !present(crate::llm_metrics::METRIC_CACHE_READ_TOKENS_TOTAL),
        "Bedrock headers carry no cache breakdown"
    );
    assert!(
        !present(crate::llm_metrics::METRIC_EMPTY_COMPLETIONS_TOTAL),
        "empty_completions must not be recorded on a non-empty response"
    );
}

#[test]
fn bedrock_headers_without_headers_record_nothing() {
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let mut resp = Response {
        headers: http::HeaderMap::new(),
        status: http::StatusCode::OK,
    };
    ctx.response_header = Some(&mut resp);

    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    metrics::with_local_recorder(&recorder, || extract_bedrock_headers(&mut ctx));
    let snap = snapshotter.snapshot().into_vec();

    assert!(
        snap.is_empty(),
        "missing Bedrock token headers should record nothing, got {snap:?}"
    );
}

// -----------------------------------------------------------------------------
// TPOT and finish-reason metrics
// -----------------------------------------------------------------------------

#[tokio::test]
async fn metrics_sse_openai_records_finish_reason() {
    let filter = make_filter(ProviderKind::OpenAi);
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.cluster = Some("test-cluster".to_owned().into());
    ctx.set_metadata(crate::llm_metrics::META_MODEL_TO_HEADER_MODEL, "test-model");

    let mut resp = make_response_with_content_type("text/event-stream");
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());
    ctx.response_header = None;

    let events = b"data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"},\"finish_reason\":\"stop\"}]}\n\n";
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    metrics::with_local_recorder(&recorder, || {
        let mut body = Some(Bytes::copy_from_slice(events));
        drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());
    });
    let snap = snapshotter.snapshot().into_vec();

    let total: u64 = snap
        .iter()
        .filter(|(key, ..)| {
            key.key().name() == crate::llm_metrics::METRIC_FINISH_REASONS_TOTAL
                && key.key().labels().any(|l| l.key() == "reason" && l.value() == "stop")
        })
        .filter_map(|(_, _, _, value)| match value {
            DebugValue::Counter(count) => Some(*count),
            _ => None,
        })
        .sum();
    assert_eq!(total, 1, "OpenAI finish_reason stop should be recorded once");
}

#[test]
fn output_deltas_record_itl_and_one_vllm_tpot_sample() {
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.cluster = Some("test-cluster".to_owned().into());
    ctx.set_metadata(crate::llm_metrics::META_MODEL_TO_HEADER_MODEL, "test-model");

    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    metrics::with_local_recorder(&recorder, || {
        record_output_delta(&mut ctx);
        record_output_delta(&mut ctx);
        record_output_delta(&mut ctx);
        record_streaming_tpot(&mut ctx, 4);
    });
    let snap = snapshotter.snapshot().into_vec();

    let samples = |name: &str| -> usize {
        snap.iter()
            .filter_map(|(key, _, _, value)| {
                if key.key().name() == name
                    && let DebugValue::Histogram(values) = value
                {
                    Some(values.len())
                } else {
                    None
                }
            })
            .sum()
    };
    assert_eq!(samples(crate::llm_metrics::METRIC_ITL_SECONDS), 2);
    assert_eq!(samples(crate::llm_metrics::METRIC_TPOT_SECONDS), 1);
}

#[test]
#[expect(clippy::too_many_lines, reason = "formula and skip assertions stay together")]
fn tpot_uses_vllm_formula_and_skips_single_token_responses() {
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let start = Instant::now();
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    metrics::with_local_recorder(&recorder, || {
        ctx.extensions.insert(StreamTimingState {
            first_output: start,
            last_output: start + Duration::from_secs(2),
            itl: crate::llm_metrics::itl_histogram("test-model", "test-cluster"),
            tpot: crate::llm_metrics::tpot_histogram("test-model", "test-cluster"),
        });
        record_streaming_tpot(&mut ctx, 5);
    });
    let snap = snapshotter.snapshot().into_vec();

    let values: Vec<f64> = snap
        .iter()
        .filter_map(|(key, _, _, value)| {
            if key.key().name() == crate::llm_metrics::METRIC_TPOT_SECONDS
                && let DebugValue::Histogram(values) = value
            {
                Some(values.iter().map(|value| value.into_inner()).collect::<Vec<f64>>())
            } else {
                None
            }
        })
        .flatten()
        .collect();
    assert_eq!(values, vec![0.5], "(2s) / (5-1 output tokens) == 0.5s");

    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    metrics::with_local_recorder(&recorder, || {
        ctx.extensions.insert(StreamTimingState {
            first_output: start,
            last_output: start + Duration::from_secs(2),
            itl: crate::llm_metrics::itl_histogram("test-model", "test-cluster"),
            tpot: crate::llm_metrics::tpot_histogram("test-model", "test-cluster"),
        });
        record_streaming_tpot(&mut ctx, 1);
    });
    let samples: usize = snapshotter
        .snapshot()
        .into_vec()
        .iter()
        .filter_map(|(key, _, _, value)| {
            if key.key().name() == crate::llm_metrics::METRIC_TPOT_SECONDS
                && let DebugValue::Histogram(values) = value
            {
                Some(values.len())
            } else {
                None
            }
        })
        .sum();
    assert_eq!(samples, 0, "single-token responses must not produce TPOT");
}

#[test]
fn non_output_sse_events_do_not_start_timing() {
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.cluster = Some("test-cluster".to_owned().into());
    ctx.set_metadata(crate::llm_metrics::META_MODEL_TO_HEADER_MODEL, "test-model");

    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    metrics::with_local_recorder(&recorder, || {
        process_sse_payload(
            &mut ctx,
            br#"{"choices":[{"delta":{"role":"assistant"}}]}"#,
            ProviderKind::OpenAi,
        );
        process_sse_payload(
            &mut ctx,
            br#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
            ProviderKind::OpenAi,
        );
    });

    assert!(ctx.extensions.get::<StreamTimingState>().is_none());
    assert!(
        snapshotter
            .snapshot()
            .into_vec()
            .iter()
            .all(|(key, ..)| key.key().name() != crate::llm_metrics::METRIC_ITL_SECONDS)
    );
}

#[tokio::test]
#[expect(clippy::too_many_lines, reason = "test setup and metric assertions stay together")]
async fn sse_terminal_usage_event_records_tpot_and_itl() {
    let filter = make_filter(ProviderKind::OpenAi);
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.cluster = Some("test-cluster".to_owned().into());
    ctx.set_metadata(crate::llm_metrics::META_MODEL_TO_HEADER_MODEL, "test-model");

    let mut resp = make_response_with_content_type("text/event-stream");
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());
    ctx.response_header = None;

    let events = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"A\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"B\"}}]}\n\n",
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":3,\"total_tokens\":13}}\n\n",
    );

    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    metrics::with_local_recorder(&recorder, || {
        let mut body = Some(Bytes::copy_from_slice(events.as_bytes()));
        drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());
    });

    assert_eq!(ctx.get_metadata("token.output"), Some("3"));
    let snap = snapshotter.snapshot().into_vec();
    let samples = |name: &str| -> usize {
        snap.iter()
            .filter_map(|(key, _, _, value)| {
                if key.key().name() == name
                    && let DebugValue::Histogram(values) = value
                {
                    Some(values.len())
                } else {
                    None
                }
            })
            .sum()
    };
    assert_eq!(samples(crate::llm_metrics::METRIC_TPOT_SECONDS), 1);
    assert_eq!(samples(crate::llm_metrics::METRIC_ITL_SECONDS), 1);
}

#[tokio::test]
#[expect(clippy::too_many_lines, reason = "test setup and metric assertions stay together")]
async fn sse_terminal_usage_event_with_single_output_token_skips_tpot() {
    let filter = make_filter(ProviderKind::OpenAi);
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.cluster = Some("test-cluster".to_owned().into());
    ctx.set_metadata(crate::llm_metrics::META_MODEL_TO_HEADER_MODEL, "test-model");

    let mut resp = make_response_with_content_type("text/event-stream");
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());
    ctx.response_header = None;

    let events = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"A\"}}]}\n\n",
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":1,\"total_tokens\":11}}\n\n",
    );

    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    metrics::with_local_recorder(&recorder, || {
        let mut body = Some(Bytes::copy_from_slice(events.as_bytes()));
        drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());
    });

    assert_eq!(ctx.get_metadata("token.output"), Some("1"));
    let snap = snapshotter.snapshot().into_vec();
    let tpot_samples: usize = snap
        .iter()
        .filter_map(|(key, _, _, value)| {
            if key.key().name() == crate::llm_metrics::METRIC_TPOT_SECONDS
                && let DebugValue::Histogram(values) = value
            {
                Some(values.len())
            } else {
                None
            }
        })
        .sum();
    assert_eq!(tpot_samples, 0, "output_tokens <= 1 must not produce TPOT");
}

#[tokio::test]
#[expect(clippy::too_many_lines, reason = "test setup and metric assertions stay together")]
async fn sse_without_usage_does_not_emit_empty_completion() {
    let filter = make_filter(ProviderKind::OpenAi);
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.cluster = Some("test-cluster".to_owned().into());
    ctx.set_metadata(crate::llm_metrics::META_MODEL_TO_HEADER_MODEL, "test-model");

    let mut resp = make_response_with_content_type("text/event-stream");
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());
    ctx.response_header = None;

    let events = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"A\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"B\"}}]}\n\n",
    );

    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    metrics::with_local_recorder(&recorder, || {
        let mut body = Some(Bytes::copy_from_slice(events.as_bytes()));
        drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());
    });

    assert!(ctx.get_metadata("token.output").is_none());
    let snap = snapshotter.snapshot().into_vec();
    assert!(
        snap.iter()
            .all(|(key, ..)| key.key().name() != crate::llm_metrics::METRIC_EMPTY_COMPLETIONS_TOTAL),
        "no-usage streams cannot infer an empty completion"
    );
    let itl_samples: usize = snap
        .iter()
        .filter_map(|(key, _, _, value)| {
            if key.key().name() == crate::llm_metrics::METRIC_ITL_SECONDS
                && let DebugValue::Histogram(values) = value
            {
                Some(values.len())
            } else {
                None
            }
        })
        .sum();
    assert_eq!(itl_samples, 1);
    let tpot_samples: usize = snap
        .iter()
        .filter_map(|(key, _, _, value)| {
            if key.key().name() == crate::llm_metrics::METRIC_TPOT_SECONDS
                && let DebugValue::Histogram(values) = value
            {
                Some(values.len())
            } else {
                None
            }
        })
        .sum();
    assert_eq!(tpot_samples, 0);
}

#[test]
fn extract_finish_reason_is_provider_aware() {
    assert_eq!(
        extract_finish_reason(br#"{"choices":[{"finish_reason":"stop"}]}"#, ProviderKind::OpenAi),
        Some("stop".to_owned())
    );
    assert_eq!(
        extract_finish_reason(br#"{"delta":{"stop_reason":"end_turn"}}"#, ProviderKind::Anthropic),
        Some("end_turn".to_owned())
    );
    assert_eq!(
        extract_finish_reason(
            br#"{"candidates":[{"finishReason":"MAX_TOKENS"}]}"#,
            ProviderKind::Google
        ),
        Some("MAX_TOKENS".to_owned())
    );
    assert_eq!(
        extract_finish_reason(br#"{"messageStop":{"stopReason":"tool_use"}}"#, ProviderKind::Bedrock),
        Some("tool_use".to_owned())
    );
    assert_eq!(extract_finish_reason(b"{}", ProviderKind::Auto), None);
    assert_eq!(extract_finish_reason(b"{}", ProviderKind::BedrockInvokeModel), None);
}

#[test]
fn normalize_finish_reason_bounds_labels() {
    assert_eq!(normalize_finish_reason("stop"), "stop");
    assert_eq!(normalize_finish_reason("end_turn"), "stop");
    assert_eq!(normalize_finish_reason("max_tokens"), "length");
    assert_eq!(normalize_finish_reason("tool_use"), "tool_calls");
    assert_eq!(normalize_finish_reason("SAFETY"), "content_filter");
    assert_eq!(normalize_finish_reason("stop_sequence"), "stop_sequence");
    assert_eq!(normalize_finish_reason("brand_new_reason"), "other");
}

#[test]
#[ignore = "release-mode overhead benchmark; run with --release --ignored --nocapture"]
#[expect(clippy::too_many_lines, reason = "benchmark setup and assertions stay together")]
#[expect(clippy::print_stdout, reason = "release-mode benchmark output")]
fn metrics_overhead_is_within_budget() {
    const ITERATIONS: usize = 100_000;
    const BUDGET_NANOS: u128 = 50_000;

    let payload = br#"{"choices":[{"delta":{"content":"token"}}]}"#;
    let mut detector_samples = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        std::hint::black_box(crate::llm_output_delta::payload_has_output_delta(payload));
        detector_samples.push(start.elapsed().as_nanos());
    }

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let mut itl_samples = Vec::with_capacity(ITERATIONS);
    metrics::with_local_recorder(&metrics::NoopRecorder, || {
        ctx.extensions.insert(StreamTimingState {
            first_output: Instant::now(),
            last_output: Instant::now(),
            itl: crate::llm_metrics::itl_histogram("test-model", "test-cluster"),
            tpot: crate::llm_metrics::tpot_histogram("test-model", "test-cluster"),
        });
        for _ in 0..ITERATIONS {
            let start = Instant::now();
            record_output_delta(&mut ctx);
            itl_samples.push(start.elapsed().as_nanos());
        }
    });

    detector_samples.sort_unstable();
    itl_samples.sort_unstable();
    let p99 = |samples: &[u128]| samples[ITERATIONS * 99 / 100];
    println!(
        "detector p99={}ns itl-record p99={}ns",
        p99(&detector_samples),
        p99(&itl_samples)
    );
    assert!(
        p99(&detector_samples) < BUDGET_NANOS,
        "output-delta detector exceeded 50us p99"
    );
    assert!(p99(&itl_samples) < BUDGET_NANOS, "ITL recording exceeded 50us p99");
}
