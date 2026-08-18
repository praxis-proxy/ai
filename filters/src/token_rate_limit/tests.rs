// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Tests for the `token_rate_limit` filter.

use praxis_filter::FilterAction;

use super::TokenRateLimitFilter;
use crate::token_usage::META_TOKEN_TOTAL;

/// Build a request carrying a single extra header, for `bucket_key_header` tests.
fn make_request_with_header(name: &str, value: &str) -> praxis_filter::Request {
    let mut req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    req.headers.insert(
        http::header::HeaderName::from_bytes(name.as_bytes()).unwrap(),
        http::HeaderValue::from_str(value).unwrap(),
    );
    req
}

// -----------------------------------------------------------------------------
// Config Validation
// -----------------------------------------------------------------------------

#[test]
fn from_config_parses_valid_config() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("rate: 1000\nburst: 100000\nestimate_tokens: 500").unwrap();
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();
    assert_eq!(filter.name(), "token_rate_limit");
}

#[test]
fn from_config_rejects_zero_rate() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("rate: 0\nburst: 100\nestimate_tokens: 10").unwrap();
    let err = TokenRateLimitFilter::from_config(&yaml).err().expect("should error");
    assert!(err.to_string().contains("rate must be"), "got: {err}");
}

#[test]
fn from_config_rejects_zero_burst() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("rate: 10\nburst: 0\nestimate_tokens: 5").unwrap();
    let err = TokenRateLimitFilter::from_config(&yaml).err().expect("should error");
    assert!(err.to_string().contains("burst must be"), "got: {err}");
}

#[test]
fn from_config_rejects_negative_estimate() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("rate: 10\nburst: 100\nestimate_tokens: -5").unwrap();
    let err = TokenRateLimitFilter::from_config(&yaml).err().expect("should error");
    assert!(err.to_string().contains("estimate_tokens"), "got: {err}");
}

#[test]
fn from_config_rejects_estimate_exceeding_burst() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("rate: 10\nburst: 100\nestimate_tokens: 500").unwrap();
    let err = TokenRateLimitFilter::from_config(&yaml).err().expect("should error");
    assert!(err.to_string().contains("must not exceed burst"), "got: {err}");
}

#[test]
fn from_config_rejects_unknown_field() {
    let yaml: serde_yaml::Value =
        serde_yaml::from_str("rate: 10\nburst: 100\nestimate_tokens: 5\nbucket_key: header").unwrap();
    assert!(
        TokenRateLimitFilter::from_config(&yaml).is_err(),
        "composite/CEL bucket keys are still deliberately unsupported, config should reject the unknown field"
    );
}

#[test]
fn from_config_accepts_bucket_key_header() {
    let yaml: serde_yaml::Value =
        serde_yaml::from_str("rate: 10\nburst: 100\nestimate_tokens: 5\nbucket_key_header: x-app-id").unwrap();
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();
    assert_eq!(filter.name(), "token_rate_limit");
}

#[test]
fn from_config_defaults_bucket_key_header_to_none_when_absent() {
    // Existing configs without bucket_key_header must keep working unchanged
    // (single global bucket) -- covered by from_config_parses_valid_config,
    // asserted again here to pin the default explicitly as this field is added.
    let yaml: serde_yaml::Value = serde_yaml::from_str("rate: 10\nburst: 100\nestimate_tokens: 5").unwrap();
    assert!(TokenRateLimitFilter::from_config(&yaml).is_ok());
}

// -----------------------------------------------------------------------------
// Admission (on_request)
// -----------------------------------------------------------------------------

#[tokio::test]
async fn admits_request_within_budget() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("rate: 100\nburst: 1000\nestimate_tokens: 200").unwrap();
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let action = filter.on_request(&mut ctx).await.unwrap();
    assert!(matches!(action, FilterAction::Continue), "should admit within budget");
    assert_eq!(
        ctx.get_metadata("token_rate_limit.reserved"),
        Some("200"),
        "reservation amount should be stashed for reconciliation"
    );
}

#[tokio::test]
async fn rejects_with_429_when_bucket_exhausted() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("rate: 1\nburst: 100\nestimate_tokens: 60").unwrap();
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut first_ctx = crate::test_utils::make_filter_context(&req);
    let mut second_ctx = crate::test_utils::make_filter_context(&req);

    // First request consumes 60 of 100; second needs another 60, only 40 left.
    let first = filter.on_request(&mut first_ctx).await.unwrap();
    assert!(
        matches!(first, FilterAction::Continue),
        "first request should be admitted"
    );

    let second = filter.on_request(&mut second_ctx).await.unwrap();
    match second {
        FilterAction::Reject(rejection) => {
            assert_eq!(rejection.status, 429);
            let has_header = |name: &str| rejection.headers.iter().any(|(n, _)| n == name);
            assert!(has_header("Retry-After"), "429 should carry Retry-After");
            assert!(
                has_header("X-RateLimit-Limit-Tokens"),
                "429 should carry token-suffixed limit header"
            );
            assert!(
                has_header("X-RateLimit-Remaining-Tokens"),
                "429 should carry token-suffixed remaining header"
            );
            assert!(has_header("X-RateLimit-Reset-Tokens"), "429 should carry reset header");
        },
        other => panic!("second request should be rejected, insufficient tokens remain, got {other:?}"),
    }
}

#[tokio::test]
async fn rejection_does_not_consume_tokens() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("rate: 0.0001\nburst: 10\nestimate_tokens: 10").unwrap();
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    // First request consumes the entire burst; every request after that
    // should be rejected, and rejecting must not partially drain (or
    // otherwise corrupt) the already-empty bucket.
    let first = filter.on_request(&mut ctx).await.unwrap();
    assert!(
        matches!(first, FilterAction::Continue),
        "first request should exactly exhaust the burst"
    );

    for _ in 0..3 {
        let action = filter.on_request(&mut ctx).await.unwrap();
        assert!(
            matches!(action, FilterAction::Reject(_)),
            "empty bucket should always reject"
        );
    }
}

// -----------------------------------------------------------------------------
// Response Headers (on_response)
// -----------------------------------------------------------------------------

#[tokio::test]
async fn on_response_injects_token_headers() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("rate: 100\nburst: 1000\nestimate_tokens: 200").unwrap();
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut resp = crate::test_utils::make_response();
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.response_header = Some(&mut resp);

    drop(filter.on_request(&mut ctx).await.unwrap());
    drop(filter.on_response(&mut ctx).await.unwrap());

    assert!(ctx.response_headers_modified, "should flag headers as modified");
    let resp = ctx.response_header.expect("response header should be present");
    assert!(resp.headers.contains_key("x-ratelimit-limit-tokens"));
    assert!(resp.headers.contains_key("x-ratelimit-remaining-tokens"));
    assert!(resp.headers.contains_key("x-ratelimit-reset-tokens"));
}

// -----------------------------------------------------------------------------
// Reconciliation (on_response_body)
// -----------------------------------------------------------------------------

#[tokio::test]
async fn reconcile_releases_unused_tokens_on_overestimate() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("rate: 0.0001\nburst: 100\nestimate_tokens: 50").unwrap();
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    drop(filter.on_request(&mut ctx).await.unwrap()); // reserves 50, 50 left
    ctx.set_metadata(META_TOKEN_TOTAL, "30"); // actual usage was only 30

    let mut body = None;
    drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());

    // Reserved 50, actual 30 -> release 20 back -> 70 should now remain.
    // A next 50-token request should succeed (70 >= 50, leaving 20)...
    let mut second_ctx = crate::test_utils::make_filter_context(&req);
    let second = filter.on_request(&mut second_ctx).await.unwrap();
    assert!(
        matches!(second, FilterAction::Continue),
        "70 remaining should admit a 50-token request"
    );

    // ...but a third 50-token request should now fail (only 20 left).
    let mut third_ctx = crate::test_utils::make_filter_context(&req);
    let third = filter.on_request(&mut third_ctx).await.unwrap();
    assert!(
        matches!(third, FilterAction::Reject(_)),
        "only 20 remaining should reject a 50-token request"
    );
}

#[tokio::test]
async fn reconcile_draws_more_tokens_on_underestimate_and_can_starve_next_request() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("rate: 0.0001\nburst: 100\nestimate_tokens: 50").unwrap();
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    drop(filter.on_request(&mut ctx).await.unwrap()); // reserves 50, 50 left
    ctx.set_metadata(META_TOKEN_TOTAL, "90"); // actual usage exceeded the estimate

    let mut body = None;
    drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());

    // Reserved 50, actual 90 -> draw 40 more from the 50 remaining -> 10 left.
    // A next request needing more than 10 should now be rejected.
    let mut next_ctx = crate::test_utils::make_filter_context(&req);
    let next_action = filter.on_request(&mut next_ctx).await.unwrap();
    assert!(
        matches!(next_action, FilterAction::Reject(_)),
        "underestimate should have drawn down the bucket enough to starve the next 50-token request"
    );
}

#[tokio::test]
async fn reconcile_is_noop_without_token_total_metadata() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("rate: 0.0001\nburst: 100\nestimate_tokens: 50").unwrap();
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    drop(filter.on_request(&mut ctx).await.unwrap()); // reserves 50, 50 left
    // No token.total metadata set (e.g. token_count filter not configured upstream).

    let mut body = None;
    drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());

    // Reservation should stand unchanged: a second request needing the
    // remaining 50 should still succeed (no phantom charge/release happened).
    let mut next_ctx = crate::test_utils::make_filter_context(&req);
    let next_action = filter.on_request(&mut next_ctx).await.unwrap();
    assert!(
        matches!(next_action, FilterAction::Continue),
        "reservation should stand as final when token.total is absent"
    );
}

#[tokio::test]
async fn does_not_reconcile_before_end_of_stream() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("rate: 0.0001\nburst: 100\nestimate_tokens: 50").unwrap();
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    drop(filter.on_request(&mut ctx).await.unwrap());
    ctx.set_metadata(META_TOKEN_TOTAL, "5");

    let mut body = None;
    drop(filter.on_response_body(&mut ctx, &mut body, false).unwrap());

    // Reconciliation must not have run yet: 50 tokens are still reserved
    // (not released down to a 5-actual balance), so a fresh 50-token
    // request only has the remaining 50 of burst=100 to draw from, and a
    // second one on top of that should fail.
    let mut second_ctx = crate::test_utils::make_filter_context(&req);
    let second = filter.on_request(&mut second_ctx).await.unwrap();
    assert!(
        matches!(second, FilterAction::Continue),
        "50 remaining should admit one more 50-token request"
    );

    let mut third_ctx = crate::test_utils::make_filter_context(&req);
    let third = filter.on_request(&mut third_ctx).await.unwrap();
    assert!(
        matches!(third, FilterAction::Reject(_)),
        "bucket should be fully drained now (no premature release happened pre-end_of_stream)"
    );
}

// -----------------------------------------------------------------------------
// Per-app bucket keys (ai#129 bucket_key_header)
// -----------------------------------------------------------------------------

#[tokio::test]
async fn bucket_key_header_isolates_budgets_across_apps() {
    let yaml: serde_yaml::Value =
        serde_yaml::from_str("rate: 0.0001\nburst: 100\nestimate_tokens: 100\nbucket_key_header: x-app-id").unwrap();
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();

    // app-a exhausts its own 100-token budget entirely...
    let app_a_req = make_request_with_header("x-app-id", "app-a");
    let mut app_a_ctx = crate::test_utils::make_filter_context(&app_a_req);
    let app_a_first = filter.on_request(&mut app_a_ctx).await.unwrap();
    assert!(
        matches!(app_a_first, FilterAction::Continue),
        "app-a's first request should be admitted"
    );

    let mut app_a_second_ctx = crate::test_utils::make_filter_context(&app_a_req);
    let app_a_second = filter.on_request(&mut app_a_second_ctx).await.unwrap();
    assert!(
        matches!(app_a_second, FilterAction::Reject(_)),
        "app-a should now be blocked, budget exhausted"
    );

    // ...but app-b's independent 100-token budget is completely untouched.
    let app_b_req = make_request_with_header("x-app-id", "app-b");
    let mut app_b_ctx = crate::test_utils::make_filter_context(&app_b_req);
    let app_b_first = filter.on_request(&mut app_b_ctx).await.unwrap();
    assert!(
        matches!(app_b_first, FilterAction::Continue),
        "app-b's budget must be independent of app-a's exhausted one"
    );
}

#[tokio::test]
async fn bucket_key_header_falls_back_to_shared_bucket_when_header_absent() {
    let yaml: serde_yaml::Value =
        serde_yaml::from_str("rate: 0.0001\nburst: 50\nestimate_tokens: 50\nbucket_key_header: x-app-id").unwrap();
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();

    // No x-app-id header on either request: both should share one fallback bucket.
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut first_ctx = crate::test_utils::make_filter_context(&req);
    let first = filter.on_request(&mut first_ctx).await.unwrap();
    assert!(
        matches!(first, FilterAction::Continue),
        "first keyless request should be admitted"
    );

    let mut second_ctx = crate::test_utils::make_filter_context(&req);
    let second = filter.on_request(&mut second_ctx).await.unwrap();
    assert!(
        matches!(second, FilterAction::Reject(_)),
        "second keyless request should share (and exhaust) the same fallback bucket"
    );
}

#[tokio::test]
async fn bucket_key_header_reconciles_against_the_same_per_key_bucket() {
    let yaml: serde_yaml::Value =
        serde_yaml::from_str("rate: 0.0001\nburst: 100\nestimate_tokens: 50\nbucket_key_header: x-app-id").unwrap();
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();

    let app_c_req = make_request_with_header("x-app-id", "app-c");
    let mut ctx = crate::test_utils::make_filter_context(&app_c_req);
    drop(filter.on_request(&mut ctx).await.unwrap()); // reserves 50 from app-c's bucket, 50 left
    ctx.set_metadata(META_TOKEN_TOTAL, "30"); // actual usage only 30

    let mut body = None;
    drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());

    // Reconciliation must have released 20 back into app-c's *own* bucket
    // (50 remaining + 20 released = 70), not some other app's or the
    // global fallback bucket.
    let mut next_ctx = crate::test_utils::make_filter_context(&app_c_req);
    let next = filter.on_request(&mut next_ctx).await.unwrap();
    assert!(
        matches!(next, FilterAction::Continue),
        "app-c's own bucket should reflect the released tokens (70 available >= 50 requested)"
    );

    let mut after_ctx = crate::test_utils::make_filter_context(&app_c_req);
    let after = filter.on_request(&mut after_ctx).await.unwrap();
    assert!(
        matches!(after, FilterAction::Reject(_)),
        "only 20 left in app-c's bucket after the prior admission, should reject another 50-token request"
    );
}
