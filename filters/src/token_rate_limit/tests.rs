// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Tests for the `token_rate_limit` filter.

use praxis_filter::{FilterAction, HttpFilter};

use super::TokenRateLimitFilter;
use crate::token_usage::META_TOKEN_TOTAL;

/// Wrap one rule body (already-valid YAML lines, unindented) into a
/// full one-rule `rules:` config, named `"default"`. Most scenarios
/// pre-date per-rule algorithm choice and only care about one rule's
/// behavior in isolation -- multi-rule dispatch itself is covered
/// separately below.
fn single_rule(body: &str) -> String {
    let indented = body
        .lines()
        .map(|line| format!("    {line}"))
        .collect::<Vec<_>>()
        .join("\n");
    format!("rules:\n  - name: default\n{indented}\n")
}

/// [`single_rule`], parsed straight into a [`serde_yaml::Value`].
fn single_rule_yaml(body: &str) -> serde_yaml::Value {
    serde_yaml::from_str(&single_rule(body)).unwrap()
}

/// [`single_rule`], with a filter-level `top_level` block (e.g.
/// `backend: {...}`) prepended as a sibling of `rules:`.
fn single_rule_yaml_with(top_level: &str, body: &str) -> serde_yaml::Value {
    serde_yaml::from_str(&format!("{top_level}\n{}", single_rule(body))).unwrap()
}

/// Build a request carrying a single extra header, for `match` tests.
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
    let yaml = single_rule_yaml("algorithm: sliding_window\nwindow: 1h\ncapacity: 100000\nreserved_tokens: 500");
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();
    assert_eq!(filter.name(), "token_rate_limit");
}

#[test]
fn from_config_rejects_an_empty_rules_list() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("rules: []\n").unwrap();
    let err = TokenRateLimitFilter::from_config(&yaml).err().expect("should error");
    assert!(err.to_string().contains("at least one rule"), "got: {err}");
}

#[test]
fn from_config_rejects_duplicate_rule_names() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        "rules:\n\
         \x20 - name: dup\n\
         \x20   algorithm: sliding_window\n\
         \x20   window: 1h\n\
         \x20   capacity: 100\n\
         \x20   reserved_tokens: 10\n\
         \x20 - name: dup\n\
         \x20   algorithm: token_bucket\n\
         \x20   capacity: 100\n\
         \x20   refill_rate: 1\n\
         \x20   reserved_tokens: 10\n",
    )
    .unwrap();
    let err = TokenRateLimitFilter::from_config(&yaml).err().expect("should error");
    assert!(err.to_string().contains("duplicate rule name"), "got: {err}");
}

#[test]
fn from_config_rejects_zero_capacity() {
    let yaml = single_rule_yaml("algorithm: sliding_window\nwindow: 1h\ncapacity: 0\nreserved_tokens: 10");
    let err = TokenRateLimitFilter::from_config(&yaml).err().expect("should error");
    assert!(err.to_string().contains("capacity must be"), "got: {err}");
}

/// `validate_rule_bounds` is the single gate both algorithms pass
/// through in `compile_rule`, before either builds its own backend.
/// `token_bucket_ledger` happens to re-check this same bound on its
/// own construction path, but `ledger` (`sliding_window`) does not --
/// so this rejection has to come from the shared gate, not either
/// algorithm's downstream validation, to protect both.
#[test]
fn from_config_rejects_capacity_above_the_lua_safe_integer_bound() {
    let over_bound = super::token_bucket_ledger::MAX_F64_SAFE_INTEGER + 1;

    let sliding_window = single_rule_yaml(&format!(
        "algorithm: sliding_window\nwindow: 1h\ncapacity: {over_bound}\nreserved_tokens: 10"
    ));
    let err = TokenRateLimitFilter::from_config(&sliding_window)
        .err()
        .expect("sliding_window should reject a capacity beyond the f64 safe-integer bound");
    assert!(err.to_string().contains("must not exceed"), "got: {err}");

    let token_bucket = single_rule_yaml(&format!(
        "algorithm: token_bucket\ncapacity: {over_bound}\nrefill_rate: 1\nreserved_tokens: 10"
    ));
    let err = TokenRateLimitFilter::from_config(&token_bucket)
        .err()
        .expect("token_bucket should reject a capacity beyond the f64 safe-integer bound");
    assert!(err.to_string().contains("must not exceed"), "got: {err}");
}

#[test]
fn from_config_rejects_zero_estimate() {
    let yaml = single_rule_yaml("algorithm: sliding_window\nwindow: 1h\ncapacity: 100\nreserved_tokens: 0");
    let err = TokenRateLimitFilter::from_config(&yaml).err().expect("should error");
    assert!(err.to_string().contains("reserved_tokens"), "got: {err}");
}

#[test]
fn from_config_rejects_estimate_exceeding_capacity() {
    let yaml = single_rule_yaml("algorithm: sliding_window\nwindow: 1h\ncapacity: 100\nreserved_tokens: 500");
    let err = TokenRateLimitFilter::from_config(&yaml).err().expect("should error");
    assert!(err.to_string().contains("must not exceed capacity"), "got: {err}");
}

#[test]
fn from_config_rejects_invalid_window() {
    let yaml = single_rule_yaml("algorithm: sliding_window\nwindow: not-a-duration\ncapacity: 100\nreserved_tokens: 5");
    let err = TokenRateLimitFilter::from_config(&yaml).err().expect("should error");
    assert!(err.to_string().contains("invalid duration"), "got: {err}");
}

#[test]
fn from_config_rejects_unknown_field() {
    let yaml = single_rule_yaml(
        "algorithm: sliding_window\nwindow: 1h\ncapacity: 100\nreserved_tokens: 5\nbucket_key: header",
    );
    assert!(
        TokenRateLimitFilter::from_config(&yaml).is_err(),
        "composite/CEL bucket keys are still deliberately unsupported, config should reject the unknown field"
    );
}

#[test]
fn from_config_rejects_the_old_flat_pre_rules_shape() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("window: 1h\ncapacity: 100000\nreserved_tokens: 500").unwrap();
    let err = TokenRateLimitFilter::from_config(&yaml).err().expect("should error");
    assert!(err.to_string().contains("rules"), "got: {err}");
}

// -----------------------------------------------------------------------------
// Backend config (Valkey opt-in for shared/distributed state)
// -----------------------------------------------------------------------------

#[test]
fn from_config_defaults_to_memory_backend_when_backend_block_absent() {
    // No `backend:` block at all must keep working exactly like before this
    // field was added -- a config written before the Valkey backend existed
    // should never start silently expecting shared state it never asked for.
    let yaml = single_rule_yaml("algorithm: sliding_window\nwindow: 1h\ncapacity: 100\nreserved_tokens: 5");
    assert!(TokenRateLimitFilter::from_config(&yaml).is_ok());
}

#[test]
fn from_config_accepts_explicit_memory_backend() {
    let yaml = single_rule_yaml_with(
        "backend:\n  kind: memory",
        "algorithm: sliding_window\nwindow: 1h\ncapacity: 100\nreserved_tokens: 5",
    );
    assert!(TokenRateLimitFilter::from_config(&yaml).is_ok());
}

#[test]
fn from_config_rejects_valkey_backend_without_url() {
    // A distributed deployment that forgets `backend.url` must fail loudly
    // at startup, not silently fall back to per-instance state -- silent
    // fallback would defeat the whole point of asking for a shared backend.
    let yaml = single_rule_yaml_with(
        "backend:\n  kind: valkey",
        "algorithm: sliding_window\nwindow: 1h\ncapacity: 100\nreserved_tokens: 5",
    );
    let err = TokenRateLimitFilter::from_config(&yaml).err().expect("should error");
    assert!(err.to_string().contains("backend.url is required"), "got: {err}");
}

#[test]
fn from_config_accepts_valkey_backend_with_url() {
    let yaml = single_rule_yaml_with(
        "backend:\n  kind: valkey\n  url: redis://127.0.0.1:6399",
        "algorithm: sliding_window\nwindow: 1h\ncapacity: 100\nreserved_tokens: 5",
    );
    assert!(TokenRateLimitFilter::from_config(&yaml).is_ok());
}

#[test]
fn from_config_accepts_two_rules_with_different_algorithms_sharing_one_filter_level_valkey_backend() {
    // The whole point of moving `backend:` from per-rule to per-filter:
    // one `backend:` block, two rules with two different algorithms, one
    // shared Valkey connection underneath -- not one connection per rule.
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        "backend:\n\
         \x20 kind: valkey\n\
         \x20 url: redis://127.0.0.1:6399\n\
         \x20 namespace: shared\n\
         rules:\n\
         \x20 - name: team-alpha\n\
         \x20   algorithm: sliding_window\n\
         \x20   window: 1h\n\
         \x20   capacity: 100\n\
         \x20   reserved_tokens: 5\n\
         \x20 - name: team-beta\n\
         \x20   algorithm: token_bucket\n\
         \x20   capacity: 100\n\
         \x20   refill_rate: 1\n\
         \x20   reserved_tokens: 5\n",
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
    let yaml = single_rule_yaml("algorithm: sliding_window\nwindow: 1h\ncapacity: 1000\nreserved_tokens: 200");
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
    let yaml = single_rule_yaml("algorithm: sliding_window\nwindow: 1h\ncapacity: 100\nreserved_tokens: 60");
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
    let yaml = single_rule_yaml("algorithm: sliding_window\nwindow: 1h\ncapacity: 10\nreserved_tokens: 10");
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

#[tokio::test]
async fn a_request_matching_no_rule_is_not_rate_limited() {
    // Business behavior: a rule scoped to one app must not silently
    // become a global rate limiter for traffic it was never configured
    // to cover -- operators who want a catch-all budget add a trailing
    // rule with no `match`.
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        "rules:\n\
         \x20 - name: alpha-only\n\
         \x20   match:\n\
         \x20     headers:\n\
         \x20       x-app-id: alpha\n\
         \x20   algorithm: sliding_window\n\
         \x20   window: 1h\n\
         \x20   capacity: 1\n\
         \x20   reserved_tokens: 1\n",
    )
    .unwrap();
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();

    let unmatched_req = make_request_with_header("x-app-id", "beta");
    for _ in 0..5 {
        let mut ctx = crate::test_utils::make_filter_context(&unmatched_req);
        assert!(
            matches!(filter.on_request(&mut ctx).await.unwrap(), FilterAction::Continue),
            "traffic matching no configured rule must pass through, even repeatedly"
        );
    }
}

// -----------------------------------------------------------------------------
// Reconciliation (on_response_body)
// -----------------------------------------------------------------------------

#[tokio::test]
async fn reconcile_releases_unused_tokens_on_overestimate() {
    let yaml = single_rule_yaml("algorithm: sliding_window\nwindow: 1h\ncapacity: 100\nreserved_tokens: 50");
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
    let yaml = single_rule_yaml("algorithm: sliding_window\nwindow: 1h\ncapacity: 100\nreserved_tokens: 50");
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
    let yaml = single_rule_yaml("algorithm: sliding_window\nwindow: 1h\ncapacity: 100\nreserved_tokens: 50");
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
    let yaml = single_rule_yaml("algorithm: sliding_window\nwindow: 1h\ncapacity: 100\nreserved_tokens: 50");
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

/// End-of-stream on an exchange that was never admitted by this filter
/// (e.g. it matched no rule, or a prior filter already short-circuited
/// the request) must not panic or reconcile phantom state -- `reconcile`
/// no-ops when `on_request` never stashed reservation/key/rule metadata.
#[tokio::test]
async fn reconcile_is_a_noop_without_prior_admission_metadata() {
    let yaml = single_rule_yaml("algorithm: sliding_window\nwindow: 1h\ncapacity: 100\nreserved_tokens: 50");
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    // Deliberately skip `on_request` -- no reservation/key/rule metadata
    // is present on `ctx`.
    let mut body = None;
    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(matches!(action, FilterAction::Continue));

    // The full 100-token budget must still be there for a real request --
    // the no-op above must not have reserved or settled anything against it.
    let mut fresh_ctx = crate::test_utils::make_filter_context(&req);
    assert!(matches!(
        filter.on_request(&mut fresh_ctx).await.unwrap(),
        FilterAction::Continue
    ));
}

// -----------------------------------------------------------------------------
// Lost-request handling (the proposal's still-open question, answered here
// via reservation_timeout)
// -----------------------------------------------------------------------------

#[tokio::test]
async fn lost_request_is_charged_at_its_estimate_and_cannot_bypass_the_budget() {
    // A client that aborts a request before the response completes
    // (connection reset, client timeout, upstream crash) must not be
    // able to dodge the budget entirely by ensuring on_response_body/
    // reconciliation never runs -- that would make token rate limiting
    // trivially bypassable by just not waiting for the response.
    // reservation_timeout bounds how long such a reservation is trusted
    // before being conservatively
    // charged at its estimate, matching the "lost request handling"
    // question the proposal's own design doc leaves open.
    let yaml = single_rule_yaml(
        "algorithm: sliding_window\nwindow: 300ms\ncapacity: 50\nreserved_tokens: 50\nreservation_timeout: 50ms",
    );
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
fn from_config_rejects_an_invalid_match_header_name() {
    let yaml = single_rule_yaml(
        "algorithm: sliding_window\nwindow: 1h\ncapacity: 100\nreserved_tokens: 5\nmatch:\n  headers:\n    \"x \
         app\": bad\n",
    );
    let err = TokenRateLimitFilter::from_config(&yaml).err().expect("should error");
    assert!(err.to_string().contains("invalid match header"), "got: {err}");
}

#[test]
fn from_config_accepts_minute_suffix_durations() {
    let yaml = single_rule_yaml("algorithm: sliding_window\nwindow: 5m\ncapacity: 100\nreserved_tokens: 5");
    assert!(TokenRateLimitFilter::from_config(&yaml).is_ok());
}

#[test]
fn from_config_rejects_a_zero_duration_window() {
    let yaml = single_rule_yaml("algorithm: sliding_window\nwindow: 0s\ncapacity: 100\nreserved_tokens: 5");
    let err = TokenRateLimitFilter::from_config(&yaml).err().expect("should error");
    assert!(err.to_string().contains("must be positive"), "got: {err}");
}

#[test]
fn debug_format_lists_configured_rule_names() {
    // `from_config` returns `Box<dyn HttpFilter>`, which has no `Debug`
    // impl -- build the concrete type directly to exercise its own
    // `Debug` impl instead.
    let yaml = single_rule_yaml("algorithm: sliding_window\nwindow: 1h\ncapacity: 100\nreserved_tokens: 5");
    let cfg: super::config::TokenRateLimitConfig =
        praxis_filter::parse_filter_config("token_rate_limit", &yaml).unwrap();
    let backend = super::build_backend_resource(&cfg.backend).unwrap();
    let rules = cfg
        .rules
        .into_iter()
        .map(|rule| super::compile_rule(rule, &backend))
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let filter = TokenRateLimitFilter {
        rules,
        epoch: std::time::Instant::now(),
    };
    let debug = format!("{filter:?}");
    assert!(debug.contains("default"), "got: {debug}");
}

#[test]
fn from_config_accepts_custom_reservation_timeout() {
    let yaml = single_rule_yaml(
        "algorithm: sliding_window\nwindow: 1h\ncapacity: 100\nreserved_tokens: 5\nreservation_timeout: 10s",
    );
    assert!(TokenRateLimitFilter::from_config(&yaml).is_ok());
}

#[test]
fn from_config_rejects_invalid_reservation_timeout() {
    let yaml = single_rule_yaml(
        "algorithm: sliding_window\nwindow: 1h\ncapacity: 100\nreserved_tokens: 5\nreservation_timeout: \
         not-a-duration",
    );
    let err = TokenRateLimitFilter::from_config(&yaml).err().expect("should error");
    assert!(err.to_string().contains("invalid duration"), "got: {err}");
}

// -----------------------------------------------------------------------------
// Per-rule algorithm choice (ai#789/praxis#551): mixed sliding_window and
// token_bucket rules, disambiguated by a header match -- the customer
// scenario this feature exists for (each app/team picks its own
// algorithm and budget).
// -----------------------------------------------------------------------------

/// Two rules, one per algorithm, matched by `x-app-id`: `alpha` gets a
/// tiny sliding-window budget, `beta` gets a tiny token-bucket budget.
fn two_algorithm_rules_yaml() -> serde_yaml::Value {
    serde_yaml::from_str(
        "rules:\n\
         \x20 - name: team-alpha\n\
         \x20   match:\n\
         \x20     headers:\n\
         \x20       x-app-id: alpha\n\
         \x20   algorithm: sliding_window\n\
         \x20   window: 1h\n\
         \x20   capacity: 100\n\
         \x20   reserved_tokens: 100\n\
         \x20 - name: team-beta\n\
         \x20   match:\n\
         \x20     headers:\n\
         \x20       x-app-id: beta\n\
         \x20   algorithm: token_bucket\n\
         \x20   capacity: 100\n\
         \x20   refill_rate: 1\n\
         \x20   reserved_tokens: 100\n",
    )
    .unwrap()
}

#[tokio::test]
async fn dispatches_to_the_first_matching_rule_by_algorithm_and_enforces_its_own_budget() {
    let filter = TokenRateLimitFilter::from_config(&two_algorithm_rules_yaml()).unwrap();

    let alpha_req = make_request_with_header("x-app-id", "alpha");
    let mut ctx = crate::test_utils::make_filter_context(&alpha_req);
    assert!(
        matches!(filter.on_request(&mut ctx).await.unwrap(), FilterAction::Continue),
        "alpha's sliding-window rule should admit its first 100-token request"
    );
    let mut ctx = crate::test_utils::make_filter_context(&alpha_req);
    assert!(
        matches!(filter.on_request(&mut ctx).await.unwrap(), FilterAction::Reject(_)),
        "alpha's sliding-window budget is now exhausted"
    );

    // beta's independent token-bucket rule/budget is completely untouched
    // by alpha's exhaustion, proving the two rules (and algorithms) are
    // fully isolated from one another.
    let beta_req = make_request_with_header("x-app-id", "beta");
    let mut ctx = crate::test_utils::make_filter_context(&beta_req);
    assert!(
        matches!(filter.on_request(&mut ctx).await.unwrap(), FilterAction::Continue),
        "beta's token-bucket rule must be unaffected by alpha's exhausted sliding-window rule"
    );
    let mut ctx = crate::test_utils::make_filter_context(&beta_req);
    assert!(
        matches!(filter.on_request(&mut ctx).await.unwrap(), FilterAction::Reject(_)),
        "beta's token-bucket budget is now exhausted too"
    );
}

/// A request matching neither rule's `match:` condition (e.g. a readiness
/// probe with no `x-app-id`) isn't rate limited by this filter instance at
/// all -- it's admitted without reserving against *any* rule's budget.
/// This is the documented mitigation for unrelated/non-inference traffic
/// under a scoped (non-catch-all) rule set (see `on_request`'s doc comment).
#[tokio::test]
async fn on_request_with_no_matching_rule_admits_without_reserving_any_budget() {
    let filter = TokenRateLimitFilter::from_config(&two_algorithm_rules_yaml()).unwrap();

    let unmatched = crate::test_utils::make_request(http::Method::GET, "/healthz");
    for _ in 0..5 {
        let mut ctx = crate::test_utils::make_filter_context(&unmatched);
        assert!(
            matches!(filter.on_request(&mut ctx).await.unwrap(), FilterAction::Continue),
            "a request matching no rule's `match:` condition must never be rejected by this filter"
        );
    }

    // Prove the five unmatched requests above didn't silently draw down
    // alpha's budget: it must still have its full 100-token capacity.
    let alpha_req = make_request_with_header("x-app-id", "alpha");
    let mut ctx = crate::test_utils::make_filter_context(&alpha_req);
    assert!(
        matches!(filter.on_request(&mut ctx).await.unwrap(), FilterAction::Continue),
        "alpha's full budget must be untouched by requests that matched no rule"
    );
}

/// Two-rule config for [`reconciliation_settles_against_the_same_rule_that_admitted_the_request`]:
/// alpha (sliding window, capacity 5) and beta (token bucket, capacity
/// 100, `reserved_tokens` 40 -- smaller than capacity so a correct vs.
/// wrong/no-op credit-back is observably distinguishable).
fn team_alpha_sliding_and_team_beta_bucket_config() -> serde_yaml::Value {
    serde_yaml::from_str(
        "rules:\n\
         \x20 - name: team-alpha\n\
         \x20   match:\n\
         \x20     headers:\n\
         \x20       x-app-id: alpha\n\
         \x20   algorithm: sliding_window\n\
         \x20   window: 1h\n\
         \x20   capacity: 5\n\
         \x20   reserved_tokens: 5\n\
         \x20 - name: team-beta\n\
         \x20   match:\n\
         \x20     headers:\n\
         \x20       x-app-id: beta\n\
         \x20   algorithm: token_bucket\n\
         \x20   capacity: 100\n\
         \x20   refill_rate: 0.0001\n\
         \x20   reserved_tokens: 40\n",
    )
    .unwrap()
}

#[tokio::test]
async fn reconciliation_settles_against_the_same_rule_that_admitted_the_request() {
    // Regression guard for the rule-index bookkeeping: reconciling a
    // token-bucket-admitted request must credit back into *that same*
    // rule's own bucket, not silently no-op or corrupt a different rule's
    // (e.g. the sliding-window one's) state.
    let filter = TokenRateLimitFilter::from_config(&team_alpha_sliding_and_team_beta_bucket_config()).unwrap();

    let beta_req = make_request_with_header("x-app-id", "beta");
    // 100 - 40 - 40 = 20 remaining, then denied on a third 40-token ask.
    let mut first_ctx = crate::test_utils::make_filter_context(&beta_req);
    assert!(matches!(
        filter.on_request(&mut first_ctx).await.unwrap(),
        FilterAction::Continue
    ));
    assert!(matches!(
        request_action(&*filter, &beta_req).await,
        FilterAction::Continue
    ));
    assert!(matches!(
        request_action(&*filter, &beta_req).await,
        FilterAction::Reject(_)
    ));

    // Reconcile the *first* reservation down to actual usage of 10
    // (refunding 30): if this credited the wrong rule, or no-op'd, beta's
    // bucket would still be stuck at 20 and stay denied below.
    first_ctx.set_metadata(META_TOKEN_TOTAL, "10");
    let mut body = None;
    drop(filter.on_response_body(&mut first_ctx, &mut body, true).unwrap());

    // 20 + 30 refund = 50 available, enough for one more 40-token request.
    assert!(
        matches!(request_action(&*filter, &beta_req).await, FilterAction::Continue),
        "the refund from reconciling beta's own reservation must land in beta's own bucket"
    );

    // alpha's untouched sliding-window rule must still have its full
    // capacity -- proving the refund didn't leak into the wrong rule.
    let alpha_req = make_request_with_header("x-app-id", "alpha");
    assert!(
        matches!(request_action(&*filter, &alpha_req).await, FilterAction::Continue),
        "alpha's rule must be completely unaffected by beta's reconciliation"
    );
}

#[test]
fn from_config_accepts_a_token_bucket_rule() {
    let yaml = single_rule_yaml("algorithm: token_bucket\ncapacity: 100\nrefill_rate: 10\nreserved_tokens: 5");
    assert!(TokenRateLimitFilter::from_config(&yaml).is_ok());
}

/// Regression test for the actual reported vulnerability: `.nan`/`.inf`
/// parse cleanly from YAML via `serde_yaml` into an `f64` field with no
/// deserialization error, so this must be caught by `from_config`'s
/// validation, not just by unit tests that construct the Rust config
/// struct directly (which bypass the YAML layer entirely and can't catch
/// a future regression in the YAML-to-ledger wiring).
#[test]
fn from_config_rejects_non_finite_refill_rate_from_yaml() {
    for literal in [".nan", ".inf", "-.inf"] {
        let yaml = single_rule_yaml(&format!(
            "algorithm: token_bucket\ncapacity: 100\nrefill_rate: {literal}\nreserved_tokens: 5"
        ));
        let err = TokenRateLimitFilter::from_config(&yaml)
            .err()
            .unwrap_or_else(|| panic!("refill_rate: {literal} must be rejected"));
        assert!(
            err.to_string().contains("refill_rate"),
            "got: {err} for refill_rate: {literal}"
        );
    }
}

#[test]
fn from_config_rejects_a_refill_rate_that_would_overflow_the_valkey_reserve_scripts_pexpire_ttl() {
    // capacity / refill_rate = 1e11 seconds -- a config typo away from
    // plausible (e.g. an extra zero on refill_rate against a large
    // capacity meant for a generous burst rule), not a contrived extreme.
    // See MAX_CAPACITY_REFILL_RATE_RATIO_SECS's doc comment for why an
    // unbounded ratio here is a real, silently budget-draining bug on the
    // Valkey backend, not just a cosmetic validation gap.
    let yaml = single_rule_yaml("algorithm: token_bucket\ncapacity: 1000000000\nrefill_rate: 0.01\nreserved_tokens: 5");
    let err = TokenRateLimitFilter::from_config(&yaml).err().expect("should error");
    assert!(err.to_string().contains("capacity / refill_rate"), "got: {err}");
}

#[tokio::test]
async fn token_bucket_rule_admits_within_capacity_and_denies_over_it() {
    let yaml = single_rule_yaml("algorithm: token_bucket\ncapacity: 100\nrefill_rate: 1\nreserved_tokens: 100");
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));

    let mut ctx = crate::test_utils::make_filter_context(&req);
    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Reject(_)
    ));
}

// -----------------------------------------------------------------------------
// Valkey-backed shared state (final scenario: state shared across gateway
// instances/replicas) -- gated on a live Valkey/Redis instance via
// TOKEN_RATE_LIMIT_VALKEY_URL, skipped (not failed) when unset so
// contributors without a local Valkey aren't blocked. Set up locally with:
//   brew install valkey && valkey-server --port 6400 --daemonize yes --save ""
//   TOKEN_RATE_LIMIT_VALKEY_URL=redis://127.0.0.1:6400 cargo test -p praxis-ai-filters token_rate_limit
// -----------------------------------------------------------------------------

/// [`single_rule`], with a filter-level `backend: {kind: valkey}` block
/// prepended -- shared by the cross-instance/worker-reconciliation
/// scenarios below, which only vary the algorithm-specific rule body.
fn single_rule_valkey_yaml(algorithm_body: &str, url: &str, namespace: &str) -> serde_yaml::Value {
    single_rule_yaml_with(
        &format!("backend:\n  kind: valkey\n  url: {url}\n  namespace: {namespace}"),
        algorithm_body,
    )
}

/// `filter.on_request` against a fresh context for one test request.
async fn request_action(filter: &dyn HttpFilter, req: &praxis_filter::Request) -> FilterAction {
    let mut ctx = crate::test_utils::make_filter_context(req);
    filter.on_request(&mut ctx).await.unwrap()
}

/// Assert `req` is admitted by `filter`, with a business-behavior message
/// explaining why (for the many cross-instance/cross-algorithm Valkey
/// scenarios below).
async fn assert_admitted(filter: &dyn HttpFilter, req: &praxis_filter::Request, why: &str) {
    assert!(
        matches!(request_action(filter, req).await, FilterAction::Continue),
        "{why}"
    );
}

/// Assert `req` is denied (429) by `filter`, with a business-behavior
/// message explaining why.
async fn assert_denied(filter: &dyn HttpFilter, req: &praxis_filter::Request, why: &str) {
    assert!(
        matches!(request_action(filter, req).await, FilterAction::Reject(_)),
        "{why}"
    );
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
    let yaml = single_rule_valkey_yaml(
        "algorithm: sliding_window\nwindow: 1h\ncapacity: 100\nreserved_tokens: 100",
        &url,
        &namespace,
    );

    // Two independent filter instances, exactly as two gateway replicas
    // would each build their own filter from the same config.
    let instance_one = TokenRateLimitFilter::from_config(&yaml).unwrap();
    let instance_two = TokenRateLimitFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    assert_admitted(instance_one.as_ref(), &req, "admitted on instance one").await;

    // The *second* gateway instance must see the budget as already
    // exhausted -- this is the property that makes Valkey worth the
    // added complexity over in-process state (final scenario).
    assert_denied(
        instance_two.as_ref(),
        &req,
        "exhausted budget visible via shared Valkey state",
    )
    .await;
}

#[tokio::test]
async fn valkey_worker_reconciles_usage_off_the_response_path() {
    let Ok(url) = std::env::var("TOKEN_RATE_LIMIT_VALKEY_URL") else {
        tracing::warn!("skipping: TOKEN_RATE_LIMIT_VALKEY_URL not set");
        return;
    };
    let namespace = format!("praxis-test-valkey-worker-{}", std::process::id());
    let yaml = single_rule_valkey_yaml(
        "algorithm: sliding_window\nwindow: 1h\ncapacity: 100\nreserved_tokens: 50",
        &url,
        &namespace,
    );
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));
    ctx.set_metadata(META_TOKEN_TOTAL, "10"); // actual usage far below the 50-token estimate

    // Reconciliation for a Valkey backend is enqueued onto a background
    // worker rather than awaited inline (the response must not be held
    // up on a network round-trip that has no bearing on this request's
    // own admission) -- so the freed budget becomes visible asynchronously.
    let mut body = None;
    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(matches!(action, FilterAction::Continue));

    // 10 (settled) + 85 should just fit under capacity=100 only once the
    // worker has actually released the 40 unused reserved tokens (50
    // estimate - 10 actual); before that, 50 (still-active reservation)
    // + 85 would exceed capacity and be denied.
    let yaml_probe = single_rule_valkey_yaml(
        "algorithm: sliding_window\nwindow: 1h\ncapacity: 100\nreserved_tokens: 85",
        &url,
        &namespace,
    );
    let probe_filter = TokenRateLimitFilter::from_config(&yaml_probe).unwrap();

    let settled = poll_until_admitted(probe_filter.as_ref(), &req, 40).await;
    assert!(
        settled,
        "worker-based reconciliation should eventually release the unused reservation into the shared Valkey budget"
    );
}

#[tokio::test]
async fn valkey_failure_fails_closed() {
    // An unreachable backend (no server on this port) must reject, not
    // admit -- a rate limiter that silently lets every request through
    // when its state store is unavailable defeats the point of rate
    // limiting it at all, right when a backend outage makes runaway
    // spend/load most likely.
    let yaml = single_rule_yaml_with(
        "backend:\n  kind: valkey\n  url: redis://127.0.0.1:1",
        "algorithm: sliding_window\nwindow: 1h\ncapacity: 100\nreserved_tokens: 10",
    );
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

#[tokio::test]
async fn valkey_token_bucket_budget_exhausted_on_one_instance_is_denied_on_another() {
    // The token-bucket analog of `valkey_budget_exhausted_on_one_instance_is_denied_on_another`:
    // proves the *second* algorithm also gets the distributed-state
    // property that's the whole point of the Valkey backend, not just
    // the sliding-window one.
    let Ok(url) = std::env::var("TOKEN_RATE_LIMIT_VALKEY_URL") else {
        tracing::warn!("skipping: TOKEN_RATE_LIMIT_VALKEY_URL not set");
        return;
    };
    let namespace = format!("praxis-test-tb-cross-instance-{}", std::process::id());
    let yaml = single_rule_valkey_yaml(
        "algorithm: token_bucket\ncapacity: 100\nrefill_rate: 0.001\nreserved_tokens: 100",
        &url,
        &namespace,
    );

    let instance_one = TokenRateLimitFilter::from_config(&yaml).unwrap();
    let instance_two = TokenRateLimitFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    assert_admitted(instance_one.as_ref(), &req, "admitted on instance one").await;
    assert_denied(
        instance_two.as_ref(),
        &req,
        "exhausted bucket visible via shared Valkey state",
    )
    .await;
}

#[tokio::test]
async fn valkey_token_bucket_failure_fails_closed() {
    // The token-bucket analog of `valkey_failure_fails_closed`: an
    // unreachable Valkey backend must not silently admit token-bucket
    // requests either -- fail-closed has to hold for both algorithms,
    // not just the sliding-window one it was first proven on.
    let yaml = single_rule_yaml_with(
        "backend:\n  kind: valkey\n  url: redis://127.0.0.1:1",
        "algorithm: token_bucket\ncapacity: 100\nrefill_rate: 1\nreserved_tokens: 10",
    );
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    match filter.on_request(&mut ctx).await.unwrap() {
        FilterAction::Reject(rejection) => {
            assert_eq!(rejection.status, 503, "unreachable backend should fail closed with 503");
        },
        other => panic!("unreachable Valkey backend must not admit the token-bucket request, got {other:?}"),
    }
}

#[tokio::test]
async fn valkey_token_bucket_worker_reconciles_usage_off_the_response_path() {
    // Token-bucket analog of `valkey_worker_reconciles_usage_off_the_response_path`:
    // reconciliation is enqueued onto the background worker, not awaited
    // inline, and its credit becomes visible once the worker runs.
    let Ok(url) = std::env::var("TOKEN_RATE_LIMIT_VALKEY_URL") else {
        tracing::warn!("skipping: TOKEN_RATE_LIMIT_VALKEY_URL not set");
        return;
    };
    let namespace = format!("praxis-test-tb-worker-{}", std::process::id());
    let yaml = single_rule_valkey_yaml(
        "algorithm: token_bucket\ncapacity: 100\nrefill_rate: 0.0001\nreserved_tokens: 50",
        &url,
        &namespace,
    );
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));
    ctx.set_metadata(META_TOKEN_TOTAL, "10"); // actual usage far below the 50-token estimate

    let mut body = None;
    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(matches!(action, FilterAction::Continue));

    // 10 (settled) + 85 should just fit under capacity=100 only once the
    // worker has actually credited back the 40 unused reserved tokens
    // (50 estimate - 10 actual); before that, 50 (still-reserved) + 85
    // would exceed capacity and be denied.
    let yaml_probe = single_rule_valkey_yaml(
        "algorithm: token_bucket\ncapacity: 100\nrefill_rate: 0.0001\nreserved_tokens: 85",
        &url,
        &namespace,
    );
    let probe_filter = TokenRateLimitFilter::from_config(&yaml_probe).unwrap();

    let settled = poll_until_admitted(probe_filter.as_ref(), &req, 40).await;
    assert!(
        settled,
        "worker-based reconciliation should eventually credit into the shared bucket"
    );
}
