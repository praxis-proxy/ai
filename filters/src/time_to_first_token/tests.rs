// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Boundary tests for the `time_to_first_token` filter.

use bytes::Bytes;
use http::header::HeaderValue;
use praxis_ai_apis::promotion::MAX_PROMOTED_VALUE_LEN;
use praxis_filter::{HttpFilter as _, Response};

use super::*;

macro_rules! activate_ttft {
    ($filter:expr, $ctx:expr) => {
        let mut resp = make_response_with_content_type("text/event-stream");
        $ctx.response_header = Some(&mut resp);
        drop($filter.on_response($ctx).await.unwrap());
        $ctx.response_header = None;
    };
}


// -----------------------------------------------------------------------------
// on_response: Activation
// -----------------------------------------------------------------------------

#[tokio::test]
async fn on_response_activates_for_sse() {
    let filter = TimeToFirstTokenFilter;
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let mut resp = make_response_with_content_type("text/event-stream");
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());
    ctx.response_header = None;

    assert!(
        ctx.extensions.get::<TtftState>().is_some(),
        "SSE content-type should activate TTFT tracking"
    );
}

#[tokio::test]
async fn on_response_activates_for_sse_with_charset() {
    let filter = TimeToFirstTokenFilter;
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let mut resp = make_response_with_content_type("text/event-stream; charset=utf-8");
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());
    ctx.response_header = None;

    assert!(
        ctx.extensions.get::<TtftState>().is_some(),
        "SSE with charset should still activate"
    );
}

#[tokio::test]
async fn on_response_skips_non_sse() {
    let filter = TimeToFirstTokenFilter;
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let mut resp = make_response_with_content_type("application/json");
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());
    ctx.response_header = None;

    assert!(
        ctx.extensions.get::<TtftState>().is_none(),
        "JSON content-type should not activate TTFT tracking"
    );
}

#[tokio::test]
async fn on_response_skips_non_success() {
    let filter = TimeToFirstTokenFilter;
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let mut resp =
        make_response_with_status_and_content_type(http::StatusCode::INTERNAL_SERVER_ERROR, "text/event-stream");
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());
    ctx.response_header = None;

    assert!(
        ctx.extensions.get::<TtftState>().is_none(),
        "non-success status should not activate TTFT tracking"
    );
}

#[tokio::test]
async fn on_response_skips_missing_content_type() {
    let filter = TimeToFirstTokenFilter;
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let mut resp = crate::test_utils::make_response();
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());
    ctx.response_header = None;

    assert!(
        ctx.extensions.get::<TtftState>().is_none(),
        "missing content-type should not activate"
    );
}

// -----------------------------------------------------------------------------
// on_response_body: Recording
// -----------------------------------------------------------------------------

#[tokio::test]
async fn role_and_metadata_events_do_not_trigger_ttft() {
    let filter = TimeToFirstTokenFilter;
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    activate_ttft!(&filter, &mut ctx);

    let snapshotter = record_chunks(
        &filter,
        &mut ctx,
        &[
            br#"data: {"choices":[{"delta":{"role":"assistant"}}]}"#.as_slice(),
            br#"data: {"type":"response.created"}"#.as_slice(),
        ],
        false,
    );

    assert!(ctx.extensions.get::<TtftState>().is_some());
    assert!(
        snapshotter
            .snapshot()
            .into_vec()
            .iter()
            .all(|(key, ..)| key.key().name() != METRIC_TTFT_SECONDS),
        "role and metadata events must not trigger TTFT"
    );
}

#[tokio::test]
async fn first_output_delta_records_once_and_deactivates() {
    let filter = TimeToFirstTokenFilter;
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.set_metadata("openai_chat_completions_request.model", "gpt-4o");
    activate_ttft!(&filter, &mut ctx);

    let snapshotter = record_chunks(
        &filter,
        &mut ctx,
        &[
            concat!(r#"data: {"choices":[{"delta":{"role":"assistant"}}]}"#, "\n\n").as_bytes(),
            concat!(r#"data: {"choices":[{"delta":{"content":"Hi"}}]}"#, "\n\n").as_bytes(),
            concat!(r#"data: {"choices":[{"delta":{},"finish_reason":"stop"}]}"#, "\n\n").as_bytes(),
        ],
        true,
    );

    assert!(ctx.extensions.get::<TtftState>().is_none());
    let (model, samples) = assert_single_ttft_histogram(&snapshotter);
    assert_eq!(model, "gpt-4o");
    assert_eq!(samples.len(), 1);
}

#[tokio::test]
async fn comments_are_ignored() {
    let filter = TimeToFirstTokenFilter;
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    activate_ttft!(&filter, &mut ctx);

    let snapshotter = record_chunks(
        &filter,
        &mut ctx,
        &[
            b": keepalive\n\n".as_slice(),
            br#"data: {"choices":[{"delta":{"content":"Hi"}}]}"#.as_slice(),
        ],
        true,
    );

    let (_, samples) = assert_single_ttft_histogram(&snapshotter);
    assert_eq!(samples.len(), 1);
}

#[tokio::test]
async fn responses_function_arguments_do_not_trigger_ttft() {
    let filter = TimeToFirstTokenFilter;
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    activate_ttft!(&filter, &mut ctx);

    let snapshotter = record_chunks(
        &filter,
        &mut ctx,
        &[br#"data: {"type":"response.function_call_arguments.delta","delta":"{\"x\":"}"#.as_slice()],
        true,
    );

    assert!(ctx.extensions.get::<TtftState>().is_none());
    assert!(
        snapshotter
            .snapshot()
            .into_vec()
            .iter()
            .all(|(key, ..)| key.key().name() != METRIC_TTFT_SECONDS),
        "Responses function arguments must not trigger TTFT"
    );
}

#[tokio::test]
async fn chat_tool_call_delta_triggers_ttft() {
    let filter = TimeToFirstTokenFilter;
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    activate_ttft!(&filter, &mut ctx);

    let snapshotter = record_chunks(
        &filter,
        &mut ctx,
        &[br#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"f"}}]}}]}"#.as_slice()],
        true,
    );

    let (_, samples) = assert_single_ttft_histogram(&snapshotter);
    assert_eq!(samples.len(), 1);
}

#[tokio::test]
async fn split_output_delta_records() {
    let filter = TimeToFirstTokenFilter;
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    activate_ttft!(&filter, &mut ctx);

    let snapshotter = record_chunks(
        &filter,
        &mut ctx,
        &[
            br#"data: {"choices":[{"delta":{"con"#.as_slice(),
            b"tent\":\"Hi\"}}]}\n\n".as_slice(),
        ],
        false,
    );

    let (_, samples) = assert_single_ttft_histogram(&snapshotter);
    assert_eq!(samples.len(), 1);
}

#[tokio::test]
async fn no_output_delta_at_eos_does_not_record() {
    let filter = TimeToFirstTokenFilter;
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    activate_ttft!(&filter, &mut ctx);

    let snapshotter = record_chunks(
        &filter,
        &mut ctx,
        &[br#"data: {"choices":[{"delta":{"role":"assistant"}}]}"#.as_slice()],
        true,
    );

    assert!(ctx.extensions.get::<TtftState>().is_none());
    assert!(
        snapshotter
            .snapshot()
            .into_vec()
            .iter()
            .all(|(key, ..)| key.key().name() != METRIC_TTFT_SECONDS)
    );
}

#[tokio::test]
async fn body_without_on_response_is_noop() {
    let filter = TimeToFirstTokenFilter;
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let mut body = Some(Bytes::from_static(b"data: {}\n\n"));
    drop(filter.on_response_body(&mut ctx, &mut body, false).unwrap());

    assert!(
        ctx.extensions.get::<TtftState>().is_none(),
        "no TTFT state should exist without on_response"
    );
}

// -----------------------------------------------------------------------------
// Model Resolution
// -----------------------------------------------------------------------------

#[test]
fn resolve_model_openai() {
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    ctx.set_metadata("openai_responses_request.model", "gpt-4o");
    assert_eq!(resolve_model(&ctx), "gpt-4o");
}

#[test]
fn resolve_model_chat_completions() {
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    ctx.set_metadata("openai_chat_completions_request.model", "gpt-4");
    assert_eq!(resolve_model(&ctx), "gpt-4");
}

#[test]
fn resolve_model_prefers_responses_over_chat_completions() {
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    ctx.set_metadata("openai_responses_request.model", "gpt-4o");
    ctx.set_metadata("openai_chat_completions_request.model", "gpt-4");
    assert_eq!(resolve_model(&ctx), "gpt-4o");
}

#[test]
fn resolve_model_anthropic() {
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/messages");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    ctx.set_metadata("anthropic_messages_request.model", "claude-sonnet-5");
    assert_eq!(resolve_model(&ctx), "claude-sonnet-5");
}

#[test]
fn resolve_model_anthropic_messages_to_chat_completions() {
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    ctx.set_metadata("anthropic_messages_to_chat_completions.model", "claude-haiku-4-5");
    assert_eq!(resolve_model(&ctx), "claude-haiku-4-5");
}

#[test]
fn resolve_model_prefers_openai_over_anthropic() {
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    ctx.set_metadata("openai_responses_request.model", "gpt-4o");
    ctx.set_metadata("anthropic_messages_request.model", "claude-sonnet-5");
    assert_eq!(resolve_model(&ctx), "gpt-4o");
}

#[test]
fn resolve_model_unknown_fallback() {
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let ctx = crate::test_utils::make_filter_context(&req);

    assert_eq!(resolve_model(&ctx), "unknown");
}

#[test]
fn resolve_model_ignores_raw_x_model_header() {
    let mut req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    req.headers.insert(
        http::header::HeaderName::from_static("x-model"),
        HeaderValue::from_static("client-claimed-model"),
    );
    let ctx = crate::test_utils::make_filter_context(&req);

    assert_eq!(
        resolve_model(&ctx),
        "unknown",
        "a raw client X-Model request header must never become a metric label"
    );
}

#[test]
fn resolve_model_rejects_control_characters() {
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    ctx.set_metadata("anthropic_messages_to_chat_completions.model", "model\ninjection");
    assert_eq!(resolve_model(&ctx), "unknown");
}

#[test]
fn resolve_model_rejects_oversized_value() {
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let long_model = "a".repeat(MAX_PROMOTED_VALUE_LEN + 1);
    ctx.set_metadata("openai_responses_request.model", &long_model);
    assert_eq!(resolve_model(&ctx), "unknown");
}

#[test]
fn resolve_model_accepts_max_length_value() {
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let model = "a".repeat(MAX_PROMOTED_VALUE_LEN);
    ctx.set_metadata("openai_responses_request.model", &model);
    assert_eq!(resolve_model(&ctx), model);
}

// -----------------------------------------------------------------------------
// Filter Metadata
// -----------------------------------------------------------------------------

#[test]
fn from_config_succeeds() {
    let filter = TimeToFirstTokenFilter::from_config(&serde_yaml::Value::Null).unwrap();
    assert_eq!(filter.name(), "time_to_first_token", "filter name should match");
}

#[test]
fn from_config_rejects_unknown_fields() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("bogus: true").unwrap();
    let result = TimeToFirstTokenFilter::from_config(&yaml);
    assert!(result.is_err(), "unknown fields should be rejected");
}

#[test]
fn name_returns_time_to_first_token() {
    let filter = TimeToFirstTokenFilter;
    assert_eq!(filter.name(), "time_to_first_token");
}

#[test]
fn response_body_access_is_read_only() {
    let filter = TimeToFirstTokenFilter;
    assert_eq!(filter.response_body_access(), BodyAccess::ReadOnly);
}

// -----------------------------------------------------------------------------
// Test Helpers
// -----------------------------------------------------------------------------

fn record_chunks(
    filter: &TimeToFirstTokenFilter,
    ctx: &mut HttpFilterContext<'_>,
    chunks: &[&'static [u8]],
    end_of_stream: bool,
) -> metrics_util::debugging::Snapshotter {
    let recorder = metrics_util::debugging::DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    metrics::with_local_recorder(&recorder, || {
        for chunk in chunks {
            let mut body = Some(Bytes::from_static(chunk));
            drop(filter.on_response_body(ctx, &mut body, false).unwrap());
        }
        if end_of_stream {
            let mut body = None;
            drop(filter.on_response_body(ctx, &mut body, true).unwrap());
        }
    });
    snapshotter
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

fn assert_single_ttft_histogram(snapshotter: &metrics_util::debugging::Snapshotter) -> (String, Vec<f64>) {
    let snapshot = snapshotter.snapshot().into_vec();
    let histograms: Vec<_> = snapshot
        .iter()
        .filter(|(key, ..)| key.key().name() == METRIC_TTFT_SECONDS)
        .collect();
    assert_eq!(histograms.len(), 1, "exactly one TTFT histogram should be recorded");

    let (key, _, _, value) = &histograms[0];
    let model = key
        .key()
        .labels()
        .find(|l| l.key() == "model")
        .expect("histogram should have a model label")
        .value()
        .to_owned();

    let samples = match value {
        metrics_util::debugging::DebugValue::Histogram(s) => s.iter().map(|v| v.into_inner()).collect(),
        other => panic!("expected histogram value, got {other:?}"),
    };

    (model, samples)
}
