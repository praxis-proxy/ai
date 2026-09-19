// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

use bytes::Bytes;
use praxis_filter::{FilterAction, HttpFilter};
use serde_json::json;

use super::StreamUsageInjectFilter;

fn make_filter() -> Box<dyn HttpFilter> {
    StreamUsageInjectFilter::from_config(&serde_yaml::Value::Null).unwrap()
}

async fn run(filter: &dyn HttpFilter, json: &serde_json::Value) -> serde_json::Value {
    run_at_path(filter, "/v1/chat/completions", json).await
}

async fn run_at_path(filter: &dyn HttpFilter, path: &str, json: &serde_json::Value) -> serde_json::Value {
    let body = run_raw_at_path(filter, path, serde_json::to_vec(json).unwrap().into()).await;
    serde_json::from_slice(body.as_ref().unwrap()).unwrap()
}

/// Runs the filter over raw body bytes and returns them, so tests can
/// assert byte-for-byte pass-through when that is the contract.
async fn run_raw_at_path(filter: &dyn HttpFilter, path: &str, raw: Bytes) -> Option<Bytes> {
    let req = crate::test_utils::make_request(http::Method::POST, path);
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let mut body = Some(raw);

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "filter should always continue"
    );

    body
}

#[tokio::test]
async fn injects_when_streaming_without_stream_options() {
    let filter = make_filter();
    let input = json!({
        "model": "gpt-5.4-mini",
        "stream": true,
        "messages": [{"role": "user", "content": "hi"}]
    });

    let result = run(&*filter, &input).await;

    assert_eq!(
        result["stream_options"]["include_usage"],
        json!(true),
        "should inject include_usage"
    );
    assert_eq!(result["model"], "gpt-5.4-mini", "other fields preserved");
    assert_eq!(result["stream"], true, "stream field preserved");
}

#[tokio::test]
async fn noop_when_already_present() {
    let filter = make_filter();
    let input = json!({
        "model": "gpt-5.4",
        "stream": true,
        "stream_options": {"include_usage": true},
        "messages": [{"role": "user", "content": "hi"}]
    });

    let result = run(&*filter, &input).await;

    assert_eq!(
        result["stream_options"]["include_usage"],
        json!(true),
        "should keep existing include_usage"
    );
}

#[tokio::test]
async fn noop_when_not_streaming() {
    let filter = make_filter();
    let input = json!({
        "model": "gpt-5.4",
        "stream": false,
        "messages": [{"role": "user", "content": "hi"}]
    });

    let result = run(&*filter, &input).await;

    assert!(
        result.get("stream_options").is_none(),
        "should not inject stream_options for non-streaming requests"
    );
}

#[tokio::test]
async fn noop_when_stream_absent() {
    let filter = make_filter();
    let input = json!({
        "model": "gpt-5.4",
        "messages": [{"role": "user", "content": "hi"}]
    });

    let result = run(&*filter, &input).await;

    assert!(
        result.get("stream_options").is_none(),
        "should not inject when stream field is absent"
    );
}

#[tokio::test]
async fn preserves_existing_stream_options_fields() {
    let filter = make_filter();
    let input = json!({
        "model": "gpt-5.4",
        "stream": true,
        "stream_options": {"include_usage": false, "continuous": true},
        "messages": [{"role": "user", "content": "hi"}]
    });

    let result = run(&*filter, &input).await;

    assert_eq!(
        result["stream_options"]["include_usage"],
        json!(true),
        "should override include_usage to true"
    );
    assert_eq!(
        result["stream_options"]["continuous"],
        json!(true),
        "should preserve other stream_options fields"
    );
}

#[tokio::test]
async fn gracefully_handles_non_json() {
    let filter = make_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let mut body = Some(Bytes::from_static(b"not json at all"));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(
        matches!(action, FilterAction::Continue),
        "should continue on non-JSON body"
    );
    assert_eq!(
        body.as_ref().unwrap().as_ref(),
        b"not json at all",
        "body should be unchanged"
    );
}

#[tokio::test]
async fn noop_before_end_of_stream() {
    let filter = make_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let input = json!({"model": "gpt-5.4", "stream": true});
    let mut body = Some(Bytes::from(serde_json::to_vec(&input).unwrap()));

    let action = filter.on_request_body(&mut ctx, &mut body, false).await.unwrap();

    assert!(
        matches!(action, FilterAction::Continue),
        "should continue before end of stream"
    );
}

#[tokio::test]
async fn noop_on_responses_api() {
    let filter = make_filter();
    let input = json!({"model": "gpt-4.1", "stream": true, "input": "hi"});
    let raw = serde_json::to_vec(&input).unwrap();

    let body = run_raw_at_path(&*filter, "/v1/responses", Bytes::from(raw.clone())).await;

    assert_eq!(
        body.as_ref().unwrap().as_ref(),
        raw.as_slice(),
        "body should be passed through byte-for-byte on /v1/responses"
    );
}

#[tokio::test]
async fn injects_on_trailing_slash_path() {
    let filter = make_filter();
    let input = json!({"model": "gpt-5.4", "stream": true, "messages": []});

    let result = run_at_path(&*filter, "/v1/chat/completions/", &input).await;

    assert_eq!(
        result["stream_options"]["include_usage"],
        json!(true),
        "trailing slash must not bypass injection"
    );
}

#[tokio::test]
async fn injects_with_query_string() {
    let filter = make_filter();
    let input = json!({"model": "gpt-5.4", "stream": true, "messages": []});

    let result = run_at_path(&*filter, "/v1/chat/completions?api-version=2024-10-21", &input).await;

    assert_eq!(
        result["stream_options"]["include_usage"],
        json!(true),
        "query string must not bypass injection"
    );
}

#[tokio::test]
async fn injects_when_stream_options_null() {
    let filter = make_filter();
    let input = json!({"model": "gpt-5.4", "stream": true, "stream_options": null});

    let result = run(&*filter, &input).await;

    assert_eq!(
        result["stream_options"]["include_usage"],
        json!(true),
        "null stream_options should be treated as absent and replaced with the opt-in"
    );
}

#[tokio::test]
async fn noop_when_stream_options_not_object() {
    let filter = make_filter();
    let input = json!({"model": "gpt-5.4", "stream": true, "stream_options": "yes"});
    let raw = serde_json::to_vec(&input).unwrap();

    let body = run_raw_at_path(&*filter, "/v1/chat/completions", Bytes::from(raw.clone())).await;

    assert_eq!(
        body.as_ref().unwrap().as_ref(),
        raw.as_slice(),
        "provider-invalid stream_options should pass through untouched for upstream validation"
    );
}

#[tokio::test]
async fn noop_on_non_object_json_body() {
    let filter = make_filter();
    let raw = br#"[{"stream":true}]"#;

    let body = run_raw_at_path(&*filter, "/v1/chat/completions", Bytes::from_static(raw)).await;

    assert_eq!(
        body.as_ref().unwrap().as_ref(),
        raw,
        "non-object JSON body should pass through byte-for-byte"
    );
}

#[tokio::test]
async fn noop_on_absent_body() {
    let filter = make_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let mut body: Option<Bytes> = None;

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(
        matches!(action, FilterAction::Continue),
        "should continue when no body is buffered"
    );
    assert!(body.is_none(), "no body should be synthesized");
}

#[tokio::test]
async fn noop_on_unrelated_path() {
    let filter = make_filter();
    let input = json!({"stream": true, "something": "else"});
    let raw = serde_json::to_vec(&input).unwrap();

    let body = run_raw_at_path(&*filter, "/v1/embeddings", Bytes::from(raw.clone())).await;

    assert_eq!(
        body.as_ref().unwrap().as_ref(),
        raw.as_slice(),
        "requests to unrelated endpoints should pass through byte-for-byte"
    );
}

#[test]
fn filter_name() {
    let filter = make_filter();
    assert_eq!(
        filter.name(),
        "stream_usage_inject",
        "filter should register under its config name"
    );
}

#[test]
fn zero_max_body_bytes_rejected() {
    let config = serde_yaml::from_str::<serde_yaml::Value>("max_body_bytes: 0").unwrap();

    let result = StreamUsageInjectFilter::from_config(&config);

    assert!(result.is_err(), "zero max_body_bytes should be rejected at config time");
}

#[test]
fn oversized_max_body_bytes_rejected() {
    let config = serde_yaml::from_str::<serde_yaml::Value>("max_body_bytes: 67108865").unwrap();

    let result = StreamUsageInjectFilter::from_config(&config);

    assert!(
        result.is_err(),
        "max_body_bytes above the 64 MiB ceiling should be rejected at config time"
    );
}

#[test]
fn configured_max_body_bytes_applied() {
    let config = serde_yaml::from_str::<serde_yaml::Value>("max_body_bytes: 4096").unwrap();
    let filter = StreamUsageInjectFilter::from_config(&config).unwrap();

    assert!(
        matches!(
            filter.request_body_mode(),
            praxis_filter::BodyMode::StreamBuffer { max_bytes: Some(4096) }
        ),
        "configured max_body_bytes should flow into the body mode"
    );
}

#[test]
fn default_config() {
    let filter = make_filter();
    assert_eq!(
        filter.request_body_access(),
        praxis_filter::BodyAccess::ReadWrite,
        "should request read-write body access"
    );
    assert!(
        matches!(
            filter.request_body_mode(),
            praxis_filter::BodyMode::StreamBuffer { max_bytes: Some(limit) } if limit > 0
        ),
        "should use StreamBuffer mode"
    );
}
