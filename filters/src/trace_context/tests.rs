// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

use http::{HeaderName, HeaderValue, Method};
use praxis_filter::Request;

use super::*;
use crate::test_utils::{make_filter_context, make_request};

/// Build a filter from empty YAML config.
fn make_filter() -> Box<dyn HttpFilter> {
    let yaml: serde_yaml::Value = serde_yaml::from_str("{}").expect("valid empty config");
    TraceContextFilter::from_config(&yaml).expect("filter should build")
}

/// Build a request carrying the given headers.
fn request_with(headers: &[(&'static str, &str)]) -> Request {
    let mut req = make_request(Method::POST, "/v1/responses");
    for (name, value) in headers {
        req.headers.insert(
            HeaderName::from_static(name),
            HeaderValue::from_str(value).expect("valid test header value"),
        );
    }
    req
}

/// Read an injected header from the pending upstream mutations.
fn injected(ctx: &HttpFilterContext<'_>, name: &str) -> String {
    ctx.extra_request_headers
        .iter()
        .find(|(header, _)| header.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.clone())
        .unwrap_or_default()
}

#[tokio::test]
async fn injects_both_correlation_headers_upstream() {
    let filter = make_filter();
    let req = request_with(&[]);
    let mut ctx = make_filter_context(&req);

    let action = filter.on_request(&mut ctx).await.unwrap();

    assert!(matches!(action, FilterAction::Continue), "filter should continue");
    assert_eq!(
        injected(&ctx, "x-request-id").len(),
        32,
        "should inject a generated request ID"
    );
    let traceparent = injected(&ctx, "traceparent");
    let parts: Vec<&str> = traceparent.split('-').collect();
    assert_eq!(parts.len(), 4, "traceparent should be well-formed: {traceparent}");
    assert_eq!(parts[1].len(), 32, "trace-id should be 32 hex: {traceparent}");
    assert_eq!(parts[2].len(), 16, "span-id should be 16 hex: {traceparent}");
}

#[tokio::test]
async fn continues_client_supplied_trace() {
    let filter = make_filter();
    let req = request_with(&[("traceparent", "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01")]);
    let mut ctx = make_filter_context(&req);

    drop(filter.on_request(&mut ctx).await.unwrap());

    let traceparent = injected(&ctx, "traceparent");
    let parts: Vec<&str> = traceparent.split('-').collect();
    assert_eq!(
        parts[1], "4bf92f3577b34da6a3ce929d0e0e4736",
        "client trace-id should be continued"
    );
    assert_ne!(parts[2], "00f067aa0ba902b7", "upstream hop should get its own span-id");
}

#[tokio::test]
async fn reuses_request_id_injected_by_request_id_filter() {
    let filter = make_filter();
    let req = request_with(&[]);
    let mut ctx = make_filter_context(&req);
    ctx.extra_request_headers
        .push((Cow::Borrowed("X-Request-ID"), "from-request-id-filter".to_owned()));

    drop(filter.on_request(&mut ctx).await.unwrap());

    assert_eq!(
        injected(&ctx, "x-request-id"),
        "from-request-id-filter",
        "should reuse the ID the request_id builtin injected"
    );
    assert_eq!(
        ctx.extra_request_headers
            .iter()
            .filter(|(name, _)| name.eq_ignore_ascii_case("x-request-id"))
            .count(),
        1,
        "should not duplicate the request ID header"
    );
}

#[tokio::test]
async fn does_not_accumulate_duplicates_when_run_twice() {
    let filter = make_filter();
    let req = request_with(&[]);
    let mut ctx = make_filter_context(&req);

    drop(filter.on_request(&mut ctx).await.unwrap());
    let first = injected(&ctx, "traceparent");
    drop(filter.on_request(&mut ctx).await.unwrap());

    assert_eq!(
        ctx.extra_request_headers.len(),
        2,
        "re-running should replace, not append: {:?}",
        ctx.extra_request_headers
    );
    let second = injected(&ctx, "traceparent");
    let first_trace: Vec<&str> = first.split('-').collect();
    let second_trace: Vec<&str> = second.split('-').collect();
    assert_eq!(
        first_trace[1], second_trace[1],
        "trace-id should be stable across re-entry"
    );
}

#[tokio::test]
async fn discards_malformed_client_traceparent() {
    let filter = make_filter();
    let req = request_with(&[("traceparent", "00-not-a-valid-trace-id-01")]);
    let mut ctx = make_filter_context(&req);

    drop(filter.on_request(&mut ctx).await.unwrap());

    let traceparent = injected(&ctx, "traceparent");
    assert_ne!(
        traceparent, "00-not-a-valid-trace-id-01",
        "malformed value must not be forwarded"
    );
    let parts: Vec<&str> = traceparent.split('-').collect();
    assert_eq!(parts.len(), 4, "replacement should be well-formed: {traceparent}");
    assert_eq!(parts[1].len(), 32, "replacement trace-id should be 32 hex");
}

#[test]
fn from_config_rejects_unknown_fields() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("unexpected: true").expect("valid yaml");
    assert!(
        TraceContextFilter::from_config(&yaml).is_err(),
        "unknown config fields should be rejected"
    );
}

#[test]
fn filter_name_is_stable() {
    let filter = make_filter();
    assert_eq!(filter.name(), "trace_context", "filter name should be trace_context");
}
