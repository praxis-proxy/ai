// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Tests for the `token_rate_limit` filter.

use praxis_filter::{FilterAction, HttpFilter};

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
    let yaml: serde_yaml::Value = serde_yaml::from_str("window: 1h\ncapacity: 100000\nestimate_tokens: 500").unwrap();
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();
    assert_eq!(filter.name(), "token_rate_limit");
}

#[test]
fn from_config_rejects_zero_capacity() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("window: 1h\ncapacity: 0\nestimate_tokens: 10").unwrap();
    let err = TokenRateLimitFilter::from_config(&yaml).err().expect("should error");
    assert!(err.to_string().contains("capacity must be"), "got: {err}");
}

#[test]
fn from_config_rejects_zero_estimate() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("window: 1h\ncapacity: 100\nestimate_tokens: 0").unwrap();
    let err = TokenRateLimitFilter::from_config(&yaml).err().expect("should error");
    assert!(err.to_string().contains("estimate_tokens"), "got: {err}");
}

#[test]
fn from_config_rejects_estimate_exceeding_capacity() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("window: 1h\ncapacity: 100\nestimate_tokens: 500").unwrap();
    let err = TokenRateLimitFilter::from_config(&yaml).err().expect("should error");
    assert!(err.to_string().contains("must not exceed capacity"), "got: {err}");
}

#[test]
fn from_config_rejects_invalid_window() {
    let yaml: serde_yaml::Value =
        serde_yaml::from_str("window: not-a-duration\ncapacity: 100\nestimate_tokens: 5").unwrap();
    let err = TokenRateLimitFilter::from_config(&yaml).err().expect("should error");
    assert!(err.to_string().contains("invalid duration"), "got: {err}");
}

#[test]
fn from_config_rejects_unknown_field() {
    let yaml: serde_yaml::Value =
        serde_yaml::from_str("window: 1h\ncapacity: 100\nestimate_tokens: 5\nbucket_key: header").unwrap();
    assert!(
        TokenRateLimitFilter::from_config(&yaml).is_err(),
        "composite/CEL bucket keys are still deliberately unsupported, config should reject the unknown field"
    );
}

#[test]
fn from_config_accepts_bucket_key_header() {
    let yaml: serde_yaml::Value =
        serde_yaml::from_str("window: 1h\ncapacity: 100\nestimate_tokens: 5\nbucket_key_header: x-app-id").unwrap();
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();
    assert_eq!(filter.name(), "token_rate_limit");
}

#[test]
fn from_config_defaults_bucket_key_header_to_none_when_absent() {
    // Existing configs without bucket_key_header must keep working unchanged
    // (single shared budget) -- covered by from_config_parses_valid_config,
    // asserted again here to pin the default explicitly as this field is added.
    let yaml: serde_yaml::Value = serde_yaml::from_str("window: 1h\ncapacity: 100\nestimate_tokens: 5").unwrap();
    assert!(TokenRateLimitFilter::from_config(&yaml).is_ok());
}

// -----------------------------------------------------------------------------
// Backend config (Valkey opt-in for shared/distributed state)
// -----------------------------------------------------------------------------

#[test]
fn from_config_defaults_to_memory_backend_when_backend_block_absent() {
    // No `backend:` block at all must keep working exactly like before this
    // field was added -- a config written for the in-process-only MVP
    // should never start silently expecting shared state it never asked for.
    let yaml: serde_yaml::Value = serde_yaml::from_str("window: 1h\ncapacity: 100\nestimate_tokens: 5").unwrap();
    assert!(TokenRateLimitFilter::from_config(&yaml).is_ok());
}

#[test]
fn from_config_accepts_explicit_memory_backend() {
    let yaml: serde_yaml::Value =
        serde_yaml::from_str("window: 1h\ncapacity: 100\nestimate_tokens: 5\nbackend:\n  kind: memory").unwrap();
    assert!(TokenRateLimitFilter::from_config(&yaml).is_ok());
}

#[test]
fn from_config_rejects_valkey_backend_without_url() {
    // A distributed deployment that forgets `backend.url` must fail loudly
    // at startup, not silently fall back to per-instance state -- silent
    // fallback would defeat the whole point of asking for a shared backend.
    let yaml: serde_yaml::Value =
        serde_yaml::from_str("window: 1h\ncapacity: 100\nestimate_tokens: 5\nbackend:\n  kind: valkey").unwrap();
    let err = TokenRateLimitFilter::from_config(&yaml).err().expect("should error");
    assert!(err.to_string().contains("backend.url is required"), "got: {err}");
}

#[test]
fn from_config_accepts_valkey_backend_with_url() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        "window: 1h\ncapacity: 100\nestimate_tokens: 5\nbackend:\n  kind: valkey\n  url: redis://127.0.0.1:6399",
    )
    .unwrap();
    assert!(TokenRateLimitFilter::from_config(&yaml).is_ok());
}

// `${ENV_VAR}` expansion is tested directly against `expand_backend_url_with`
// below (dependency-injected lookup) rather than through real process
// environment mutation, which is unsafe in this edition and would be
// racy across parallel test threads regardless.

#[test]
fn expand_backend_url_with_substitutes_a_resolved_env_var() {
    let expanded = super::expand_backend_url_with("${TOKEN_RATE_LIMIT_TEST_URL}", |name| {
        assert_eq!(name, "TOKEN_RATE_LIMIT_TEST_URL");
        Ok("redis://127.0.0.1:6399".to_owned())
    })
    .unwrap();
    assert_eq!(expanded, "redis://127.0.0.1:6399");
}

#[test]
fn expand_backend_url_with_passes_through_a_url_without_any_reference() {
    let expanded = super::expand_backend_url_with("redis://127.0.0.1:6399", |_name| {
        panic!("lookup should not be called when there is no ${{ENV_VAR}} reference")
    })
    .unwrap();
    assert_eq!(expanded, "redis://127.0.0.1:6399");
}

#[test]
fn expand_backend_url_with_rejects_an_unset_env_var() {
    let err = super::expand_backend_url_with("${UNSET_VAR}", |_name| Err(std::env::VarError::NotPresent))
        .expect_err("should error");
    assert!(
        err.to_string().contains("environment variable is not set"),
        "got: {err}"
    );
}

#[test]
fn expand_backend_url_with_rejects_an_embedded_reference_not_spanning_the_whole_url() {
    // A misconfigured distributed deployment silently connecting to the
    // wrong host (e.g. a typo'd literal instead of the intended env var)
    // would be a config-integrity failure that's easy to miss in review
    // -- only whole-value substitution is supported, so any other shape
    // fails config load loudly rather than doing partial/ambiguous
    // substitution.
    let err = super::expand_backend_url_with("redis://${REDIS_HOST}:6379", |_name| {
        panic!("lookup must not run for an unsupported reference shape")
    })
    .expect_err("should error");
    assert!(err.to_string().contains("one complete"), "got: {err}");
}

#[test]
fn expand_backend_url_with_rejects_multiple_references() {
    let err = super::expand_backend_url_with("${A}${B}", |_name| {
        panic!("lookup must not run for an unsupported reference shape")
    })
    .expect_err("should error");
    assert!(err.to_string().contains("one complete"), "got: {err}");
}

#[test]
fn expand_backend_url_with_rejects_an_invalid_variable_name() {
    // A misconfigured `${...}` shape (rather than a clean uppercase
    // env-var name) must fail loudly with a clear reason at startup,
    // not be silently passed through as a literal, non-functioning URL.
    let err = super::expand_backend_url_with("${lower_case}", |_name| {
        panic!("lookup must not run for an invalid variable name")
    })
    .expect_err("should error");
    assert!(
        err.to_string().contains("invalid environment variable reference"),
        "got: {err}"
    );
}

// -----------------------------------------------------------------------------
// Admission (on_request)
// -----------------------------------------------------------------------------

#[tokio::test]
async fn admits_request_within_budget() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("window: 1h\ncapacity: 1000\nestimate_tokens: 200").unwrap();
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let action = filter.on_request(&mut ctx).await.unwrap();
    assert!(matches!(action, FilterAction::Continue), "should admit within budget");
    assert!(
        ctx.get_metadata("token_rate_limit.reservation_id").is_some(),
        "reservation id should be stashed for reconciliation"
    );
}

#[tokio::test]
async fn rejects_with_429_when_budget_exhausted() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("window: 1h\ncapacity: 100\nestimate_tokens: 60").unwrap();
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
async fn rejection_does_not_consume_budget() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("window: 1h\ncapacity: 10\nestimate_tokens: 10").unwrap();
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");

    // First request consumes the entire capacity; every request after
    // that should be rejected, and rejecting must not partially drain
    // (or otherwise corrupt) the already-exhausted budget.
    let mut first_ctx = crate::test_utils::make_filter_context(&req);
    let first = filter.on_request(&mut first_ctx).await.unwrap();
    assert!(
        matches!(first, FilterAction::Continue),
        "first request should exactly exhaust the capacity"
    );

    for _ in 0..3 {
        let mut ctx = crate::test_utils::make_filter_context(&req);
        let action = filter.on_request(&mut ctx).await.unwrap();
        assert!(
            matches!(action, FilterAction::Reject(_)),
            "exhausted budget should always reject"
        );
    }
}

// -----------------------------------------------------------------------------
// Reconciliation (on_response_body)
// -----------------------------------------------------------------------------

#[tokio::test]
async fn reconcile_releases_unused_tokens_on_overestimate() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("window: 1h\ncapacity: 100\nestimate_tokens: 50").unwrap();
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
    let yaml: serde_yaml::Value = serde_yaml::from_str("window: 1h\ncapacity: 100\nestimate_tokens: 50").unwrap();
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    drop(filter.on_request(&mut ctx).await.unwrap()); // reserves 50, 50 left
    ctx.set_metadata(META_TOKEN_TOTAL, "90"); // actual usage exceeded the estimate

    let mut body = None;
    drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());

    // Reserved 50, actual 90 -> the window now holds 90 of 100 -> only 10 left.
    let mut next_ctx = crate::test_utils::make_filter_context(&req);
    let next_action = filter.on_request(&mut next_ctx).await.unwrap();
    assert!(
        matches!(next_action, FilterAction::Reject(_)),
        "underestimate should have drawn the window down enough to starve the next 50-token request"
    );
}

#[tokio::test]
async fn reconcile_charges_the_estimate_without_token_total_metadata() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("window: 1h\ncapacity: 100\nestimate_tokens: 50").unwrap();
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    drop(filter.on_request(&mut ctx).await.unwrap()); // reserves 50, 50 left
    // No token.total metadata set (e.g. token_count filter not configured upstream).

    let mut body = None;
    drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());

    // Settled at the estimate (50 of 100 used): a second 50-token request
    // should still fit exactly...
    let mut next_ctx = crate::test_utils::make_filter_context(&req);
    let next_action = filter.on_request(&mut next_ctx).await.unwrap();
    assert!(
        matches!(next_action, FilterAction::Continue),
        "50 of 100 already settled leaves exactly 50 for the next request"
    );

    // ...but a third would exceed the window's capacity.
    let mut third_ctx = crate::test_utils::make_filter_context(&req);
    let third_action = filter.on_request(&mut third_ctx).await.unwrap();
    assert!(
        matches!(third_action, FilterAction::Reject(_)),
        "window is now fully settled at 100/100"
    );
}

#[tokio::test]
async fn does_not_reconcile_before_end_of_stream() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("window: 1h\ncapacity: 100\nestimate_tokens: 50").unwrap();
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    drop(filter.on_request(&mut ctx).await.unwrap());
    ctx.set_metadata(META_TOKEN_TOTAL, "5");

    let mut body = None;
    drop(filter.on_response_body(&mut ctx, &mut body, false).unwrap());

    // Reconciliation must not have run yet: 50 tokens are still an active
    // reservation (not settled down to actual=5), so a fresh 50-token
    // request only has the remaining 50 of capacity=100 to draw from, and
    // a second one on top of that should fail.
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
        "window should be fully committed now (no premature release happened pre-end_of_stream)"
    );
}

// -----------------------------------------------------------------------------
// Per-app bucket keys (ai#129 bucket_key_header)
// -----------------------------------------------------------------------------

#[tokio::test]
async fn bucket_key_header_isolates_budgets_across_apps() {
    let yaml: serde_yaml::Value =
        serde_yaml::from_str("window: 1h\ncapacity: 100\nestimate_tokens: 100\nbucket_key_header: x-app-id").unwrap();
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
        serde_yaml::from_str("window: 1h\ncapacity: 50\nestimate_tokens: 50\nbucket_key_header: x-app-id").unwrap();
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();

    // No x-app-id header on either request: both should share one fallback budget.
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
        "second keyless request should share (and exhaust) the same fallback budget"
    );
}

#[tokio::test]
async fn bucket_key_header_reconciles_against_the_same_per_key_bucket() {
    let yaml: serde_yaml::Value =
        serde_yaml::from_str("window: 1h\ncapacity: 100\nestimate_tokens: 50\nbucket_key_header: x-app-id").unwrap();
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();

    let app_c_req = make_request_with_header("x-app-id", "app-c");
    let mut ctx = crate::test_utils::make_filter_context(&app_c_req);
    drop(filter.on_request(&mut ctx).await.unwrap()); // reserves 50 from app-c's budget, 50 left
    ctx.set_metadata(META_TOKEN_TOTAL, "30"); // actual usage only 30

    let mut body = None;
    drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());

    // Reconciliation must have settled app-c's *own* window at 30 (not
    // some other app's or the global fallback budget), leaving 70 free.
    let mut next_ctx = crate::test_utils::make_filter_context(&app_c_req);
    let next = filter.on_request(&mut next_ctx).await.unwrap();
    assert!(
        matches!(next, FilterAction::Continue),
        "app-c's own window should reflect the released tokens (70 available >= 50 requested)"
    );

    let mut after_ctx = crate::test_utils::make_filter_context(&app_c_req);
    let after = filter.on_request(&mut after_ctx).await.unwrap();
    assert!(
        matches!(after, FilterAction::Reject(_)),
        "only 20 left in app-c's window after the prior admission, should reject another 50-token request"
    );
}

#[tokio::test]
async fn bucket_key_header_with_empty_or_oversized_value_falls_back_to_the_shared_bucket() {
    // Guards two things at once: (1) correctness -- a present-but-empty
    // header value must not be treated as its own distinct key, letting
    // a client bypass per-app isolation with a technically-present but
    // meaningless value; (2) availability (FedRAMP-guidance: denial of
    // service protection) -- an oversized/garbage header value must not
    // be able to mint its own budget, which would let a client exhaust
    // the bucket_key_header cardinality bound cheaply. Both fall back to
    // the one shared budget instead.
    let yaml: serde_yaml::Value =
        serde_yaml::from_str("window: 1h\ncapacity: 50\nestimate_tokens: 50\nbucket_key_header: x-app-id").unwrap();
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();

    let empty_value_req = make_request_with_header("x-app-id", "");
    let mut ctx = crate::test_utils::make_filter_context(&empty_value_req);
    assert!(
        matches!(filter.on_request(&mut ctx).await.unwrap(), FilterAction::Continue),
        "empty-value request should be admitted from the shared fallback budget"
    );

    let oversized_value = "x".repeat(300);
    let oversized_req = make_request_with_header("x-app-id", &oversized_value);
    let mut ctx2 = crate::test_utils::make_filter_context(&oversized_req);
    let result = filter.on_request(&mut ctx2).await.unwrap();
    assert!(
        matches!(result, FilterAction::Reject(_)),
        "an oversized header value must share the fallback budget (already exhausted by the empty-value request \
         above), not mint its own independent one"
    );
}

// -----------------------------------------------------------------------------
// Lost-request handling (ai#658's still-open question, answered here via
// reservation_timeout)
// -----------------------------------------------------------------------------

#[tokio::test]
async fn lost_request_is_charged_at_its_estimate_and_cannot_bypass_the_budget() {
    // FedRAMP-guidance (denial-of-service protection): a client that
    // aborts a request before the response completes (connection reset,
    // client timeout, upstream crash) must not be able to dodge the
    // budget entirely by ensuring on_response_body/reconciliation never
    // runs -- that would make token rate limiting trivially bypassable
    // by just not waiting for the response. reservation_timeout bounds
    // how long such a reservation is trusted before being conservatively
    // charged at its estimate, matching the "lost request handling"
    // question `ai#658`'s own design doc leaves open.
    let yaml: serde_yaml::Value =
        serde_yaml::from_str("window: 300ms\ncapacity: 50\nestimate_tokens: 50\nreservation_timeout: 50ms").unwrap();
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    assert!(
        matches!(filter.on_request(&mut ctx).await.unwrap(), FilterAction::Continue),
        "first request should be admitted, reserving the entire 50-token capacity"
    );
    // Simulate an aborted request: on_response_body is deliberately never
    // called, so this reservation is never explicitly reconciled.
    drop(ctx);

    tokio::time::sleep(std::time::Duration::from_millis(80)).await; // past reservation_timeout, still inside window

    let mut second_ctx = crate::test_utils::make_filter_context(&req);
    let second = filter.on_request(&mut second_ctx).await.unwrap();
    assert!(
        matches!(second, FilterAction::Reject(_)),
        "the aborted request's reservation must still be charged against the window once it times out -- it \
         must not grant free/unmetered capacity just because the response was never observed"
    );

    tokio::time::sleep(std::time::Duration::from_millis(250)).await; // past the window's own expiry too

    let mut third_ctx = crate::test_utils::make_filter_context(&req);
    let third = filter.on_request(&mut third_ctx).await.unwrap();
    assert!(
        matches!(third, FilterAction::Continue),
        "once the window rolls over, a one-time lost request must not permanently lock the key out"
    );
}

#[test]
fn from_config_accepts_custom_reservation_timeout() {
    let yaml: serde_yaml::Value =
        serde_yaml::from_str("window: 1h\ncapacity: 100\nestimate_tokens: 5\nreservation_timeout: 10s").unwrap();
    assert!(TokenRateLimitFilter::from_config(&yaml).is_ok());
}

#[test]
fn from_config_rejects_invalid_reservation_timeout() {
    let yaml: serde_yaml::Value =
        serde_yaml::from_str("window: 1h\ncapacity: 100\nestimate_tokens: 5\nreservation_timeout: not-a-duration")
            .unwrap();
    let err = TokenRateLimitFilter::from_config(&yaml).err().expect("should error");
    assert!(err.to_string().contains("invalid duration"), "got: {err}");
}

// -----------------------------------------------------------------------------
// Valkey-backed shared state (final scenario: state shared across gateway
// instances/replicas) -- gated on a live Valkey/Redis instance via
// TOKEN_RATE_LIMIT_VALKEY_URL, skipped (not failed) when unset so
// contributors without a local Valkey aren't blocked. Set up locally with:
//   brew install valkey && valkey-server --port 6400 --daemonize yes --save ""
//   TOKEN_RATE_LIMIT_VALKEY_URL=redis://127.0.0.1:6400 cargo test -p praxis-ai-filters token_rate_limit
// -----------------------------------------------------------------------------

/// `filter.on_request` against a fresh context for one test request.
async fn request_action(filter: &dyn HttpFilter, req: &praxis_filter::Request) -> FilterAction {
    let mut ctx = crate::test_utils::make_filter_context(req);
    filter.on_request(&mut ctx).await.unwrap()
}

/// Poll `filter.on_request` for `req` up to `attempts` times, sleeping
/// briefly between each, until it's admitted. Used to await an
/// asynchronous (background-worker) Valkey reconciliation without a
/// fixed, flaky sleep.
async fn poll_until_admitted(filter: &dyn HttpFilter, req: &praxis_filter::Request, attempts: u32) -> bool {
    for _ in 0..attempts {
        if matches!(request_action(filter, req).await, FilterAction::Continue) {
            return true;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    false
}

#[tokio::test]
async fn valkey_budget_exhausted_on_one_instance_is_denied_on_another() {
    let Ok(url) = std::env::var("TOKEN_RATE_LIMIT_VALKEY_URL") else {
        tracing::warn!("skipping: TOKEN_RATE_LIMIT_VALKEY_URL not set");
        return;
    };
    let namespace = format!("praxis-test-cross-instance-{}", std::process::id());
    let yaml: serde_yaml::Value = serde_yaml::from_str(&format!(
        "window: 1h\ncapacity: 100\nestimate_tokens: 100\nbucket_key_header: x-app-id\nbackend:\n  kind: valkey\n  url: {url}\n  namespace: {namespace}\n"
    ))
    .unwrap();

    // Two independent filter instances, exactly as two gateway replicas
    // would each build their own filter from the same config.
    let instance_one = TokenRateLimitFilter::from_config(&yaml).unwrap();
    let instance_two = TokenRateLimitFilter::from_config(&yaml).unwrap();

    let app_a_req = make_request_with_header("x-app-id", "app-a");
    assert!(
        matches!(
            request_action(instance_one.as_ref(), &app_a_req).await,
            FilterAction::Continue
        ),
        "app-a's first request should be admitted on instance one"
    );

    // The *second* gateway instance must see app-a's budget as already
    // exhausted -- this is the property that makes Valkey worth the
    // added complexity over in-process state (final scenario).
    assert!(
        matches!(
            request_action(instance_two.as_ref(), &app_a_req).await,
            FilterAction::Reject(_)
        ),
        "app-a's exhausted budget must be visible on instance two via shared Valkey state"
    );

    // app-b, sharing neither app-a's identity nor its budget, must be
    // unaffected by app-a's exhaustion -- isolation holds across the
    // distributed boundary too, not just in-process. Asked of instance
    // two specifically: the same instance (and shared Valkey namespace)
    // that just correctly denied app-a.
    let app_b_req = make_request_with_header("x-app-id", "app-b");
    assert!(
        matches!(
            request_action(instance_two.as_ref(), &app_b_req).await,
            FilterAction::Continue
        ),
        "app-b must be unaffected by app-a's exhausted budget, even sharing the same Valkey namespace"
    );
}

#[tokio::test]
async fn valkey_worker_reconciles_usage_off_the_response_path() {
    let Ok(url) = std::env::var("TOKEN_RATE_LIMIT_VALKEY_URL") else {
        tracing::warn!("skipping: TOKEN_RATE_LIMIT_VALKEY_URL not set");
        return;
    };
    let namespace = format!("praxis-test-valkey-worker-{}", std::process::id());
    let yaml: serde_yaml::Value = serde_yaml::from_str(&format!(
        "window: 1h\ncapacity: 100\nestimate_tokens: 50\nbucket_key_header: x-app-id\nbackend:\n  kind: valkey\n  url: {url}\n  namespace: {namespace}\n"
    ))
    .unwrap();
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();

    let app_a_req = make_request_with_header("x-app-id", "app-a");
    let mut ctx = crate::test_utils::make_filter_context(&app_a_req);
    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    ctx.set_metadata(META_TOKEN_TOTAL, "10"); // actual usage far below the 50-token estimate

    // Reconciliation for a Valkey backend is enqueued onto a background
    // worker rather than awaited inline (the response must not be held
    // up on a network round-trip that has no bearing on this request's
    // own admission) -- so the freed budget becomes visible asynchronously.
    let mut body = None;
    assert!(matches!(
        filter.on_response_body(&mut ctx, &mut body, true).unwrap(),
        FilterAction::Continue
    ));

    // 10 (settled) + 85 should just fit under capacity=100 only once the
    // worker has actually released the 40 unused reserved tokens (50
    // estimate - 10 actual); before that, 50 (still-active reservation)
    // + 85 would exceed capacity and be denied.
    let yaml_probe: serde_yaml::Value = serde_yaml::from_str(&format!(
        "window: 1h\ncapacity: 100\nestimate_tokens: 85\nbucket_key_header: x-app-id\nbackend:\n  kind: valkey\n  url: {url}\n  namespace: {namespace}\n"
    ))
    .unwrap();
    let probe_filter = TokenRateLimitFilter::from_config(&yaml_probe).unwrap();

    let settled = poll_until_admitted(probe_filter.as_ref(), &app_a_req, 40).await;
    assert!(
        settled,
        "worker-based reconciliation should eventually release app-a's unused reservation into the shared Valkey budget"
    );
}

#[tokio::test]
async fn valkey_failure_fails_closed() {
    // Unreachable backend (no server on this port): a rate limiter that
    // silently admits everything when its backend is down defeats the
    // point of rate limiting it at all.
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        "window: 1h\ncapacity: 100\nestimate_tokens: 10\nbackend:\n  kind: valkey\n  url: redis://127.0.0.1:1\n",
    )
    .unwrap();
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    match filter.on_request(&mut ctx).await.unwrap() {
        FilterAction::Reject(rejection) => {
            assert_eq!(rejection.status, 503, "unreachable backend should fail closed with 503");
        },
        other => panic!("unreachable Valkey backend must not admit the request, got {other:?}"),
    }
}
