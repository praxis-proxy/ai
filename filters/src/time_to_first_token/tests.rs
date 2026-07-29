// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Boundary tests for the `time_to_first_token` filter.

use bytes::Bytes;
use http::header::HeaderValue;
use praxis_filter::{HttpFilter as _, Response};

use super::*;

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
        ctx.filter_metadata.contains_key(META_ACTIVE),
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
        ctx.filter_metadata.contains_key(META_ACTIVE),
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
        !ctx.filter_metadata.contains_key(META_ACTIVE),
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
        !ctx.filter_metadata.contains_key(META_ACTIVE),
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
        !ctx.filter_metadata.contains_key(META_ACTIVE),
        "missing content-type should not activate"
    );
}

// -----------------------------------------------------------------------------
// on_response_body: Recording
// -----------------------------------------------------------------------------

#[tokio::test]
async fn first_non_empty_chunk_records_and_deactivates() {
    let filter = TimeToFirstTokenFilter;
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let mut resp = make_response_with_content_type("text/event-stream");
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());
    ctx.response_header = None;

    assert!(ctx.filter_metadata.contains_key(META_ACTIVE));

    let snapshotter = record_body_with_metrics(&filter, &mut ctx, b"data: {}\n\n");

    assert!(
        !ctx.filter_metadata.contains_key(META_ACTIVE),
        "TTFT metadata should be removed after recording"
    );

    let (model, samples) = assert_single_ttft_histogram(&snapshotter);
    assert_eq!(model, "unknown", "model should fall back to unknown");
    assert_eq!(samples.len(), 1, "histogram should have exactly one sample");
    assert!(samples[0] > 0.0, "TTFT should be positive");
}

#[tokio::test]
async fn first_non_empty_chunk_records_with_model_label() {
    let filter = TimeToFirstTokenFilter;
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    ctx.set_metadata("openai_responses_format.model", "gpt-4o");

    let mut resp = make_response_with_content_type("text/event-stream");
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());
    ctx.response_header = None;

    let snapshotter = record_body_with_metrics(&filter, &mut ctx, b"data: {}\n\n");

    let (model, _) = assert_single_ttft_histogram(&snapshotter);
    assert_eq!(model, "gpt-4o", "model label should match metadata");
}

#[tokio::test]
async fn second_chunk_is_noop() {
    let filter = TimeToFirstTokenFilter;
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let mut resp = make_response_with_content_type("text/event-stream");
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());
    ctx.response_header = None;

    let mut body1 = Some(Bytes::from_static(b"data: first\n\n"));
    drop(filter.on_response_body(&mut ctx, &mut body1, false).unwrap());

    assert!(!ctx.filter_metadata.contains_key(META_ACTIVE));

    let mut body2 = Some(Bytes::from_static(b"data: second\n\n"));
    drop(filter.on_response_body(&mut ctx, &mut body2, false).unwrap());
}

#[tokio::test]
async fn empty_body_is_skipped() {
    let filter = TimeToFirstTokenFilter;
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let mut resp = make_response_with_content_type("text/event-stream");
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());
    ctx.response_header = None;

    let mut body_none: Option<Bytes> = None;
    drop(filter.on_response_body(&mut ctx, &mut body_none, false).unwrap());
    assert!(
        ctx.filter_metadata.contains_key(META_ACTIVE),
        "None body should not trigger recording"
    );

    let mut body_empty = Some(Bytes::new());
    drop(filter.on_response_body(&mut ctx, &mut body_empty, false).unwrap());
    assert!(
        ctx.filter_metadata.contains_key(META_ACTIVE),
        "empty Bytes should not trigger recording"
    );

    let mut body_data = Some(Bytes::from_static(b"data: hello\n\n"));
    drop(filter.on_response_body(&mut ctx, &mut body_data, false).unwrap());
    assert!(
        !ctx.filter_metadata.contains_key(META_ACTIVE),
        "non-empty body should trigger recording"
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
        !ctx.filter_metadata.contains_key(META_ACTIVE),
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

    ctx.set_metadata("openai_responses_format.model", "gpt-4o");
    assert_eq!(resolve_model(&ctx), "gpt-4o");
}

#[test]
fn resolve_model_anthropic() {
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/messages");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    ctx.set_metadata("anthropic_messages_format.model", "claude-sonnet-5");
    assert_eq!(resolve_model(&ctx), "claude-sonnet-5");
}

#[test]
fn resolve_model_anthropic_to_openai() {
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    ctx.set_metadata("anthropic_to_openai.model", "claude-haiku-4-5");
    assert_eq!(resolve_model(&ctx), "claude-haiku-4-5");
}

#[test]
fn resolve_model_prefers_openai_over_anthropic() {
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    ctx.set_metadata("openai_responses_format.model", "gpt-4o");
    ctx.set_metadata("anthropic_messages_format.model", "claude-sonnet-5");
    assert_eq!(resolve_model(&ctx), "gpt-4o");
}

#[test]
fn resolve_model_unknown_fallback() {
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let ctx = crate::test_utils::make_filter_context(&req);

    assert_eq!(resolve_model(&ctx), "unknown");
}

#[test]
fn resolve_model_rejects_control_characters() {
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    ctx.set_metadata("anthropic_to_openai.model", "model\ninjection");
    assert_eq!(resolve_model(&ctx), "unknown");
}

#[test]
fn resolve_model_rejects_oversized_value() {
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let long_model = "a".repeat(MAX_PROMOTED_VALUE_LEN + 1);
    ctx.set_metadata("openai_responses_format.model", &long_model);
    assert_eq!(resolve_model(&ctx), "unknown");
}

#[test]
fn resolve_model_accepts_max_length_value() {
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let model = "a".repeat(MAX_PROMOTED_VALUE_LEN);
    ctx.set_metadata("openai_responses_format.model", &model);
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

fn record_body_with_metrics(
    filter: &TimeToFirstTokenFilter,
    ctx: &mut HttpFilterContext<'_>,
    chunk: &'static [u8],
) -> metrics_util::debugging::Snapshotter {
    let recorder = metrics_util::debugging::DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    metrics::with_local_recorder(&recorder, || {
        let mut body = Some(Bytes::from(chunk));
        drop(filter.on_response_body(ctx, &mut body, false).unwrap());
    });
    snapshotter
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
