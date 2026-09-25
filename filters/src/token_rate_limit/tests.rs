// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Tests for the `token_rate_limit` filter.

use praxis_filter::{FilterAction, HttpFilter};

use super::TokenRateLimitFilter;
use crate::token_usage::{
    META_TOKEN_CACHE_READ, META_TOKEN_CACHE_WRITE, META_TOKEN_INPUT, META_TOKEN_OUTPUT, META_TOKEN_REASONING,
    META_TOKEN_STATUS, META_TOKEN_TOTAL, TOKEN_STATUS_OVERFLOW,
};

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
fn from_config_accepts_authenticated_subject_keying() {
    let yaml = single_rule_yaml_with(
        "key: authenticated_subject",
        "algorithm: sliding_window\nwindow: 1h\ncapacity: 100000\nreserved_tokens: 500",
    );
    assert!(TokenRateLimitFilter::from_config(&yaml).is_ok());
}

#[test]
fn subject_bucket_keys_are_stable_distinct_and_opaque() {
    let first = super::subject_bucket_key("application-a");
    let repeated = super::subject_bucket_key("application-a");
    let second = super::subject_bucket_key("application-b");

    assert_eq!(first, repeated);
    assert_ne!(first, second);
    assert!(!first.contains("application-a"));
    assert_eq!(first.len(), "subject:v1:".len() + 43);
}

#[tokio::test]
async fn authenticated_subject_keying_fails_closed_without_identity() {
    let yaml = single_rule_yaml_with(
        "key: authenticated_subject",
        "algorithm: sliding_window\nwindow: 1h\ncapacity: 100\nreserved_tokens: 5",
    );
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let action = filter.on_request(&mut ctx).await.unwrap();

    assert!(matches!(&action, FilterAction::Reject(rejection) if rejection.status == 401));
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

/// Missing `META_ESTIMATE` must skip settlement rather than treat the
/// estimate as 0 (which would debit `actual - 0` on top of the original
/// reservation and over-charge the window).
#[tokio::test]
async fn missing_meta_estimate_skips_reconciliation_instead_of_settling_at_zero() {
    let yaml = single_rule_yaml("algorithm: sliding_window\nwindow: 1h\ncapacity: 1000\nreserved_tokens: 500");
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    ctx.filter_metadata.remove(super::META_ESTIMATE);
    ctx.set_metadata(META_TOKEN_TOTAL, "200");
    let mut body = None;
    assert!(matches!(
        filter.on_response_body(&mut ctx, &mut body, true).unwrap(),
        FilterAction::Continue
    ));

    // Reservation of 500 still stands (no bogus +200 overage). A second
    // 500-token admit fits remaining capacity; it would not if settlement
    // had charged 700.
    let mut second = crate::test_utils::make_filter_context(&req);
    assert!(
        matches!(filter.on_request(&mut second).await.unwrap(), FilterAction::Continue),
        "skipping reconcile must leave the original 500-token reservation, not 700"
    );
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
        .map(|rule| super::compile_rule(rule, &backend, super::TokenWeights::UNITY))
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let needs_body = rules.iter().any(|r| r.estimation.needs_body());
    let filter = TokenRateLimitFilter {
        rules,
        needs_body,
        key_source: cfg.key,
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
fn from_config_rejects_non_finite_or_negative_default_weights_from_yaml() {
    for literal in [".nan", ".inf", "-.inf", "-0.1"] {
        let yaml = single_rule_yaml_with(
            &format!("default_weights:\n  cached_input: {literal}"),
            "algorithm: sliding_window\nwindow: 1h\ncapacity: 100\nreserved_tokens: 5",
        );
        let err = TokenRateLimitFilter::from_config(&yaml)
            .err()
            .unwrap_or_else(|| panic!("cached_input: {literal} must be rejected"));
        assert!(
            err.to_string().contains("cached_input"),
            "got: {err} for cached_input: {literal}"
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

#[tokio::test]
async fn input_plus_max_tokens_strategy_computes_from_body_and_content_length() {
    let yaml = single_rule_yaml(
        "algorithm: sliding_window\nwindow: 1h\ncapacity: 10000\nestimation:\n  strategy: \
         input_plus_max_tokens\n  fallback_estimate: 100\n  bytes_per_token: 4.0",
    );
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();
    let mut req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let body_bytes = br#"{"max_tokens": 200, "messages": [{"role": "user", "content": "hello"}]}"#;
    req.headers.insert(
        http::header::CONTENT_LENGTH,
        http::HeaderValue::from_str(&body_bytes.len().to_string()).unwrap(),
    );
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let action = filter.on_request(&mut ctx).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "needs_body=true, on_request should defer to on_request_body"
    );

    let mut body = Some(bytes::Bytes::from(&body_bytes[..]));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "estimate = ceil(body_len/4) + 200 should be within capacity"
    );
    assert!(
        ctx.get_metadata("token_rate_limit.reservation_id").is_some(),
        "reservation should have been created"
    );
}

#[tokio::test]
async fn malformed_json_body_uses_fallback_estimate() {
    let yaml = single_rule_yaml(
        "algorithm: sliding_window\nwindow: 1h\ncapacity: 1000\nestimation:\n  strategy: max_tokens\n  \
         fallback_estimate: 100",
    );
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    drop(filter.on_request(&mut ctx).await.unwrap());

    let mut body = Some(bytes::Bytes::from("not valid json at all"));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "malformed JSON should fall back to fallback_estimate=100 and admit"
    );
    assert!(
        ctx.get_metadata("token_rate_limit.reservation_id").is_some(),
        "fallback estimate should create a reservation"
    );
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

// -----------------------------------------------------------------------------
// Estimation strategies (M3)
// -----------------------------------------------------------------------------

#[test]
fn from_config_parses_fixed_estimation_strategy() {
    let yaml = single_rule_yaml(
        "algorithm: sliding_window\nwindow: 1h\ncapacity: 1000\nestimation:\n  strategy: fixed\n  fallback_estimate: 500",
    );
    assert!(TokenRateLimitFilter::from_config(&yaml).is_ok());
}

#[test]
fn from_config_parses_max_tokens_strategy() {
    let yaml = single_rule_yaml(
        "algorithm: sliding_window\nwindow: 1h\ncapacity: 1000\nestimation:\n  strategy: max_tokens\n  fallback_estimate: 100",
    );
    assert!(TokenRateLimitFilter::from_config(&yaml).is_ok());
}

#[test]
fn from_config_parses_input_plus_max_tokens_strategy() {
    let yaml = single_rule_yaml(
        "algorithm: sliding_window\nwindow: 1h\ncapacity: 1000\nestimation:\n  strategy: input_plus_max_tokens\n  fallback_estimate: 100\n  bytes_per_token: 3.5",
    );
    assert!(TokenRateLimitFilter::from_config(&yaml).is_ok());
}

#[test]
fn from_config_parses_model_scaled_strategy() {
    let yaml = single_rule_yaml(
        "algorithm: sliding_window\nwindow: 1h\ncapacity: 1000\nestimation:\n  strategy: model_scaled\n  fallback_estimate: 100\n  default_multiplier: 1.5\n  model_multipliers:\n    gpt-4: 2.0\n    gpt-3.5-turbo: 0.5",
    );
    assert!(TokenRateLimitFilter::from_config(&yaml).is_ok());
}

#[test]
fn from_config_rejects_both_reserved_tokens_and_estimation() {
    let yaml = single_rule_yaml(
        "algorithm: sliding_window\nwindow: 1h\ncapacity: 1000\nreserved_tokens: 50\nestimation:\n  strategy: fixed\n  fallback_estimate: 500",
    );
    let err = TokenRateLimitFilter::from_config(&yaml).err().expect("should error");
    assert!(err.to_string().contains("cannot specify both"), "got: {err}");
}

#[test]
fn from_config_rejects_neither_reserved_tokens_nor_estimation() {
    let yaml = single_rule_yaml("algorithm: sliding_window\nwindow: 1h\ncapacity: 1000");
    let err = TokenRateLimitFilter::from_config(&yaml).err().expect("should error");
    assert!(err.to_string().contains("must have either"), "got: {err}");
}

#[test]
fn from_config_rejects_fixed_strategy_without_fallback_estimate() {
    let yaml =
        single_rule_yaml("algorithm: sliding_window\nwindow: 1h\ncapacity: 1000\nestimation:\n  strategy: fixed");
    let err = TokenRateLimitFilter::from_config(&yaml).err().expect("should error");
    assert!(err.to_string().contains("fallback_estimate"), "got: {err}");
}

#[test]
fn from_config_rejects_strategy_irrelevant_estimation_fields() {
    let cases: &[(&str, &str)] = &[
        (
            "algorithm: sliding_window\nwindow: 1h\ncapacity: 1000\nestimation:\n  strategy: fixed\n  fallback_estimate: 100\n  model_multipliers:\n    gpt-4: 2.0",
            "fixed",
        ),
        (
            "algorithm: sliding_window\nwindow: 1h\ncapacity: 1000\nestimation:\n  strategy: max_tokens\n  fallback_estimate: 100\n  bytes_per_token: 3.5",
            "max_tokens",
        ),
        (
            "algorithm: sliding_window\nwindow: 1h\ncapacity: 1000\nestimation:\n  strategy: input_plus_max_tokens\n  fallback_estimate: 100\n  default_multiplier: 1.5",
            "input_plus_max_tokens",
        ),
        (
            "algorithm: sliding_window\nwindow: 1h\ncapacity: 1000\nestimation:\n  strategy: model_scaled\n  fallback_estimate: 100\n  bytes_per_token: 3.5",
            "model_scaled",
        ),
    ];
    for (body, strategy) in cases {
        let err = TokenRateLimitFilter::from_config(&single_rule_yaml(body))
            .err()
            .unwrap_or_else(|| panic!("{strategy} must reject irrelevant fields"));
        let msg = err.to_string();
        assert!(
            msg.contains("does not accept"),
            "{strategy}: expected unused-field error, got: {msg}"
        );
    }
}

#[test]
fn from_config_rejects_fixed_estimate_exceeding_capacity() {
    let yaml = single_rule_yaml(
        "algorithm: sliding_window\nwindow: 1h\ncapacity: 100\nestimation:\n  strategy: fixed\n  fallback_estimate: 200",
    );
    let err = TokenRateLimitFilter::from_config(&yaml).err().expect("should error");
    assert!(err.to_string().contains("must not exceed capacity"), "got: {err}");
}

#[test]
fn from_config_rejects_non_finite_estimation_multiplier() {
    for literal in [".nan", ".inf"] {
        let yaml = single_rule_yaml(&format!(
            "algorithm: sliding_window\nwindow: 1h\ncapacity: 1000\nestimation:\n  strategy: max_tokens\n  multiplier: {literal}\n  fallback_estimate: 100"
        ));
        let err = TokenRateLimitFilter::from_config(&yaml)
            .err()
            .unwrap_or_else(|| panic!("multiplier: {literal} must be rejected"));
        assert!(
            err.to_string().contains("multiplier"),
            "got: {err} for multiplier: {literal}"
        );
    }
}

#[test]
fn from_config_rejects_non_finite_bytes_per_token() {
    let yaml = single_rule_yaml(
        "algorithm: sliding_window\nwindow: 1h\ncapacity: 1000\nestimation:\n  strategy: input_plus_max_tokens\n  bytes_per_token: .nan\n  fallback_estimate: 100",
    );
    let err = TokenRateLimitFilter::from_config(&yaml).err().expect("should error");
    assert!(err.to_string().contains("bytes_per_token"), "got: {err}");
}

#[test]
fn from_config_rejects_non_finite_model_multiplier_entry() {
    let yaml = single_rule_yaml(
        "algorithm: sliding_window\nwindow: 1h\ncapacity: 1000\nestimation:\n  strategy: model_scaled\n  default_multiplier: 1.0\n  fallback_estimate: 100\n  model_multipliers:\n    gpt-4: .inf",
    );
    let err = TokenRateLimitFilter::from_config(&yaml).err().expect("should error");
    assert!(err.to_string().contains("model_multipliers"), "got: {err}");
}

#[test]
fn from_config_rejects_non_finite_default_multiplier() {
    let yaml = single_rule_yaml(
        "algorithm: sliding_window\nwindow: 1h\ncapacity: 1000\nestimation:\n  strategy: model_scaled\n  default_multiplier: .nan\n  fallback_estimate: 100",
    );
    let err = TokenRateLimitFilter::from_config(&yaml).err().expect("should error");
    assert!(err.to_string().contains("default_multiplier"), "got: {err}");
}

#[test]
fn from_config_rejects_fallback_estimate_exceeding_capacity() {
    let yaml = single_rule_yaml(
        "algorithm: sliding_window\nwindow: 1h\ncapacity: 100\nestimation:\n  strategy: max_tokens\n  fallback_estimate: 200",
    );
    let err = TokenRateLimitFilter::from_config(&yaml).err().expect("should error");
    assert!(
        err.to_string().contains("fallback_estimate must not exceed capacity"),
        "got: {err}"
    );
}

#[tokio::test]
async fn fixed_estimation_strategy_reserves_like_legacy_reserved_tokens() {
    let yaml = single_rule_yaml(
        "algorithm: sliding_window\nwindow: 1h\ncapacity: 1000\nestimation:\n  strategy: fixed\n  fallback_estimate: 200",
    );
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "fixed estimation should admit within budget"
    );
    assert!(ctx.get_metadata("token_rate_limit.reservation_id").is_some());
}

#[tokio::test]
async fn max_tokens_strategy_defers_to_on_request_body() {
    let yaml = single_rule_yaml(
        "algorithm: sliding_window\nwindow: 1h\ncapacity: 1000\nestimation:\n  strategy: max_tokens\n  fallback_estimate: 100",
    );
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let action = filter.on_request(&mut ctx).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "on_request should pass through when needs_body=true"
    );
    assert!(
        ctx.get_metadata("token_rate_limit.reservation_id").is_none(),
        "no reservation should be made in on_request when needs_body=true"
    );
}

#[tokio::test]
async fn max_tokens_strategy_extracts_from_body_and_reserves() {
    let yaml = single_rule_yaml(
        "algorithm: sliding_window\nwindow: 1h\ncapacity: 1000\nestimation:\n  strategy: max_tokens\n  fallback_estimate: 100",
    );
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    drop(filter.on_request(&mut ctx).await.unwrap());

    let mut body = Some(bytes::Bytes::from(r#"{"max_tokens": 500, "messages": []}"#));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "should admit with max_tokens=500 within capacity=1000"
    );
    assert!(
        ctx.get_metadata("token_rate_limit.reservation_id").is_some(),
        "reservation should be made in on_request_body"
    );

    let estimate_meta = ctx.get_metadata("token_rate_limit.estimate").unwrap();
    assert_eq!(
        estimate_meta, "500",
        "estimate metadata should reflect extracted max_tokens"
    );
}

#[tokio::test]
async fn max_tokens_strategy_prefers_max_tokens_over_max_completion_tokens() {
    let yaml =
        single_rule_yaml("algorithm: sliding_window\nwindow: 1h\ncapacity: 1000\nestimation:\n  strategy: max_tokens");
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let mut body = Some(bytes::Bytes::from(
        r#"{"max_tokens": 300, "max_completion_tokens": 500, "messages": []}"#,
    ));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));
    assert_eq!(
        ctx.get_metadata("token_rate_limit.estimate").unwrap(),
        "300",
        "max_tokens should take priority over max_completion_tokens"
    );
}

#[tokio::test]
async fn max_tokens_strategy_falls_back_to_max_completion_tokens() {
    let yaml =
        single_rule_yaml("algorithm: sliding_window\nwindow: 1h\ncapacity: 1000\nestimation:\n  strategy: max_tokens");
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let mut body = Some(bytes::Bytes::from(r#"{"max_completion_tokens": 250, "messages": []}"#));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));
    assert_eq!(
        ctx.get_metadata("token_rate_limit.estimate").unwrap(),
        "250",
        "should fall back to max_completion_tokens when max_tokens is absent"
    );
}

#[tokio::test]
async fn max_tokens_strategy_uses_fallback_when_body_has_no_tokens_field() {
    let yaml = single_rule_yaml(
        "algorithm: sliding_window\nwindow: 1h\ncapacity: 1000\nestimation:\n  strategy: max_tokens\n  fallback_estimate: 42",
    );
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let mut body = Some(bytes::Bytes::from(r#"{"messages": []}"#));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));
    assert_eq!(
        ctx.get_metadata("token_rate_limit.estimate").unwrap(),
        "42",
        "should fall back to fallback_estimate when body has no max_tokens"
    );
}

#[tokio::test]
async fn max_tokens_strategy_admits_without_reservation_when_no_fallback_and_no_body_field() {
    let yaml =
        single_rule_yaml("algorithm: sliding_window\nwindow: 1h\ncapacity: 1000\nestimation:\n  strategy: max_tokens");
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let mut body = Some(bytes::Bytes::from(r#"{"messages": []}"#));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "should admit without reservation when strategy can't extract a value and no fallback"
    );
    assert!(
        ctx.get_metadata("token_rate_limit.reservation_id").is_none(),
        "no reservation should be made when estimate is unavailable"
    );
}

#[tokio::test]
async fn max_tokens_strategy_applies_multiplier() {
    let yaml = single_rule_yaml(
        "algorithm: sliding_window\nwindow: 1h\ncapacity: 10000\nestimation:\n  strategy: max_tokens\n  multiplier: 1.5",
    );
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let mut body = Some(bytes::Bytes::from(r#"{"max_tokens": 100}"#));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));
    assert_eq!(
        ctx.get_metadata("token_rate_limit.estimate").unwrap(),
        "150",
        "100 * 1.5 = 150"
    );
}

#[tokio::test]
async fn max_tokens_strategy_denies_when_body_estimate_exceeds_budget() {
    let yaml = single_rule_yaml(
        "algorithm: sliding_window\nwindow: 1h\ncapacity: 100\nestimation:\n  strategy: max_tokens\n  fallback_estimate: 50",
    );
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut first_ctx = crate::test_utils::make_filter_context(&req);
    drop(filter.on_request(&mut first_ctx).await.unwrap());

    let mut body = Some(bytes::Bytes::from(r#"{"max_tokens": 60}"#));
    let action = filter.on_request_body(&mut first_ctx, &mut body, true).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));

    let mut second_ctx = crate::test_utils::make_filter_context(&req);
    drop(filter.on_request(&mut second_ctx).await.unwrap());

    let mut body = Some(bytes::Bytes::from(r#"{"max_tokens": 60}"#));
    let action = filter.on_request_body(&mut second_ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Reject(_)),
        "second 60-token request should be denied (only 40 of 100 remaining)"
    );
}

#[tokio::test]
async fn on_request_body_is_noop_before_end_of_stream() {
    let yaml = single_rule_yaml(
        "algorithm: sliding_window\nwindow: 1h\ncapacity: 1000\nestimation:\n  strategy: max_tokens\n  fallback_estimate: 100",
    );
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let mut body = Some(bytes::Bytes::from(r#"{"max_tokens": 500}"#));
    let action = filter.on_request_body(&mut ctx, &mut body, false).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "should continue without reservation before end_of_stream"
    );
    assert!(
        ctx.get_metadata("token_rate_limit.reservation_id").is_none(),
        "no reservation before end_of_stream"
    );
}

#[tokio::test]
async fn model_scaled_strategy_applies_per_model_multiplier() {
    let yaml = single_rule_yaml(
        "algorithm: sliding_window\nwindow: 1h\ncapacity: 10000\nestimation:\n  strategy: model_scaled\n  default_multiplier: 1.0\n  model_multipliers:\n    gpt-4: 2.0\n    gpt-3.5-turbo: 0.5",
    );
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let mut body = Some(bytes::Bytes::from(r#"{"max_tokens": 100, "model": "gpt-4"}"#));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));
    assert_eq!(
        ctx.get_metadata("token_rate_limit.estimate").unwrap(),
        "200",
        "100 * 2.0 (gpt-4 multiplier) = 200"
    );
}

#[tokio::test]
async fn model_scaled_strategy_uses_default_multiplier_for_unknown_model() {
    let yaml = single_rule_yaml(
        "algorithm: sliding_window\nwindow: 1h\ncapacity: 10000\nestimation:\n  strategy: model_scaled\n  default_multiplier: 1.5\n  model_multipliers:\n    gpt-4: 2.0",
    );
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let mut body = Some(bytes::Bytes::from(r#"{"max_tokens": 100, "model": "claude-3"}"#));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));
    assert_eq!(
        ctx.get_metadata("token_rate_limit.estimate").unwrap(),
        "150",
        "100 * 1.5 (default multiplier for unknown model) = 150"
    );
}

#[tokio::test]
async fn model_scaled_strategy_reads_model_from_x_model_header_fallback() {
    let yaml = single_rule_yaml(
        "algorithm: sliding_window\nwindow: 1h\ncapacity: 10000\nestimation:\n  strategy: model_scaled\n  default_multiplier: 1.0\n  model_multipliers:\n    gpt-4: 3.0",
    );
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();

    let mut req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    req.headers.insert(
        http::header::HeaderName::from_static("x-model"),
        http::HeaderValue::from_static("gpt-4"),
    );
    let mut ctx = crate::test_utils::make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let mut body = Some(bytes::Bytes::from(r#"{"max_tokens": 100}"#));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));
    assert_eq!(
        ctx.get_metadata("token_rate_limit.estimate").unwrap(),
        "300",
        "100 * 3.0 (gpt-4 via x-model header) = 300"
    );
}

#[tokio::test]
async fn body_dependent_reconciliation_uses_meta_estimate() {
    let yaml = single_rule_yaml(
        "algorithm: sliding_window\nwindow: 1h\ncapacity: 1000\nestimation:\n  strategy: max_tokens\n  fallback_estimate: 100",
    );
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let mut body = Some(bytes::Bytes::from(r#"{"max_tokens": 500}"#));
    drop(filter.on_request_body(&mut ctx, &mut body, true).await.unwrap());

    ctx.set_metadata(META_TOKEN_TOTAL, "200");

    let mut resp_body = None;
    drop(filter.on_response_body(&mut ctx, &mut resp_body, true).unwrap());

    // Reserved 500 via body, actual 200 -> refund 300 -> 800 remaining.
    // A second 500-token request body should be admitted (800 >= 500).
    let mut second_ctx = crate::test_utils::make_filter_context(&req);
    drop(filter.on_request(&mut second_ctx).await.unwrap());

    let mut body = Some(bytes::Bytes::from(r#"{"max_tokens": 500}"#));
    let action = filter.on_request_body(&mut second_ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "800 remaining should admit a 500-token request"
    );
}

#[test]
fn from_config_estimation_with_token_bucket_algorithm() {
    let yaml = single_rule_yaml(
        "algorithm: token_bucket\ncapacity: 1000\nrefill_rate: 10\nestimation:\n  strategy: max_tokens\n  fallback_estimate: 100",
    );
    assert!(
        TokenRateLimitFilter::from_config(&yaml).is_ok(),
        "estimation strategies must work with both algorithms"
    );
}

#[test]
fn from_config_fixed_strategy_applies_multiplier_to_estimate() {
    let yaml = single_rule_yaml(
        "algorithm: sliding_window\nwindow: 1h\ncapacity: 1000\nestimation:\n  strategy: fixed\n  fallback_estimate: 100\n  multiplier: 1.5",
    );
    assert!(TokenRateLimitFilter::from_config(&yaml).is_ok());
}

#[test]
fn request_body_access_is_none_for_fixed_strategy() {
    let yaml = single_rule_yaml(
        "algorithm: sliding_window\nwindow: 1h\ncapacity: 1000\nestimation:\n  strategy: fixed\n  fallback_estimate: 500",
    );
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();
    assert_eq!(filter.request_body_access(), praxis_filter::BodyAccess::None);
}

#[test]
fn request_body_access_is_read_only_for_body_dependent_strategy() {
    let yaml = single_rule_yaml(
        "algorithm: sliding_window\nwindow: 1h\ncapacity: 1000\nestimation:\n  strategy: max_tokens\n  fallback_estimate: 100",
    );
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();
    assert_eq!(filter.request_body_access(), praxis_filter::BodyAccess::ReadOnly);
}

#[test]
fn request_body_mode_is_stream_buffer_for_body_dependent_strategy() {
    let yaml = single_rule_yaml(
        "algorithm: sliding_window\nwindow: 1h\ncapacity: 1000\nestimation:\n  strategy: max_tokens\n  fallback_estimate: 100",
    );
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();
    assert!(
        matches!(
            filter.request_body_mode(),
            praxis_filter::BodyMode::StreamBuffer {
                max_bytes: Some(2_097_152)
            }
        ),
        "body-dependent strategies should buffer up to 2 MiB"
    );
}

// -----------------------------------------------------------------------------
// Edge cases: empty body, mixed strategies, missing Content-Length,
// model_scaled outer multiplier composition
// -----------------------------------------------------------------------------

#[tokio::test]
async fn empty_body_uses_fallback_estimate() {
    let yaml = single_rule_yaml(
        "algorithm: sliding_window\nwindow: 1h\ncapacity: 1000\nestimation:\n  strategy: max_tokens\n  \
         fallback_estimate: 75",
    );
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let mut body: Option<bytes::Bytes> = None;
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "empty body should fall back to fallback_estimate=75 and admit"
    );
    assert_eq!(
        ctx.get_metadata("token_rate_limit.estimate").unwrap(),
        "75",
        "estimate should be the fallback when body is absent"
    );
}

#[tokio::test]
async fn mixed_rules_fixed_strategy_ignores_body_when_filter_buffers() {
    let filter = TokenRateLimitFilter::from_config(&mixed_fixed_and_body_dependent_yaml()).unwrap();

    let fixed_req = make_request_with_header("x-app-id", "fixed-app");
    let mut ctx = crate::test_utils::make_filter_context(&fixed_req);
    drop(filter.on_request(&mut ctx).await.unwrap());
    let mut body = Some(bytes::Bytes::from(r#"{"max_tokens": 9999}"#));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));
    assert_eq!(
        ctx.get_metadata("token_rate_limit.estimate").unwrap(),
        "200",
        "fixed-strategy rule must use its constant (200), not the body's max_tokens"
    );
}

#[tokio::test]
async fn mixed_rules_body_dependent_strategy_extracts_from_body() {
    let filter = TokenRateLimitFilter::from_config(&mixed_fixed_and_body_dependent_yaml()).unwrap();

    let body_req = make_request_with_header("x-app-id", "body-app");
    let mut ctx = crate::test_utils::make_filter_context(&body_req);
    drop(filter.on_request(&mut ctx).await.unwrap());
    let mut body = Some(bytes::Bytes::from(r#"{"max_tokens": 300}"#));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));
    assert_eq!(
        ctx.get_metadata("token_rate_limit.estimate").unwrap(),
        "300",
        "body-dependent rule must use extracted max_tokens (300)"
    );
}

fn mixed_fixed_and_body_dependent_yaml() -> serde_yaml::Value {
    serde_yaml::from_str(
        "rules:\n\
         \x20 - name: fixed-rule\n\
         \x20   match:\n\
         \x20     headers:\n\
         \x20       x-app-id: fixed-app\n\
         \x20   algorithm: sliding_window\n\
         \x20   window: 1h\n\
         \x20   capacity: 1000\n\
         \x20   reserved_tokens: 200\n\
         \x20 - name: body-rule\n\
         \x20   match:\n\
         \x20     headers:\n\
         \x20       x-app-id: body-app\n\
         \x20   algorithm: sliding_window\n\
         \x20   window: 1h\n\
         \x20   capacity: 1000\n\
         \x20   estimation:\n\
         \x20     strategy: max_tokens\n\
         \x20     fallback_estimate: 100\n",
    )
    .unwrap()
}

#[tokio::test]
async fn input_plus_max_tokens_defaults_to_zero_input_when_content_length_absent() {
    let yaml = single_rule_yaml(
        "algorithm: sliding_window\nwindow: 1h\ncapacity: 1000\nestimation:\n  strategy: \
         input_plus_max_tokens\n  fallback_estimate: 100\n  bytes_per_token: 4.0",
    );
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();
    // No Content-Length header set.
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let mut body = Some(bytes::Bytes::from(r#"{"max_tokens": 400}"#));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));
    // input = ceil(0 / 4.0) = 0, output = 400, total = 400
    assert_eq!(
        ctx.get_metadata("token_rate_limit.estimate").unwrap(),
        "400",
        "missing Content-Length should default input to 0, estimate = 0 + max_tokens"
    );
}

#[tokio::test]
async fn model_scaled_outer_multiplier_composes_with_per_model_multiplier() {
    let yaml = single_rule_yaml(
        "algorithm: sliding_window\nwindow: 1h\ncapacity: 10000\nestimation:\n  strategy: model_scaled\n  \
         multiplier: 1.5\n  model_multipliers:\n    gpt-4: 2.0\n  default_multiplier: 1.0",
    );
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    // gpt-4: effective multiplier = 2.0 * 1.5 = 3.0, max_tokens=100 → estimate=300
    let mut body = Some(bytes::Bytes::from(r#"{"max_tokens": 100, "model": "gpt-4"}"#));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));
    assert_eq!(
        ctx.get_metadata("token_rate_limit.estimate").unwrap(),
        "300",
        "gpt-4 with outer multiplier 1.5 × model multiplier 2.0 = 3.0 × 100 = 300"
    );
}

// -----------------------------------------------------------------------------
// M4 token-type weights
// -----------------------------------------------------------------------------

fn set_typed_usage(ctx: &mut praxis_filter::HttpFilterContext<'_>, input: u64, output: u64, total: u64) {
    ctx.set_metadata(META_TOKEN_INPUT, input.to_string());
    ctx.set_metadata(META_TOKEN_OUTPUT, output.to_string());
    ctx.set_metadata(META_TOKEN_TOTAL, total.to_string());
}

#[tokio::test]
async fn weighted_cache_hit_refunds_more_than_token_total() {
    // input 100 (90 cached) + output 10, total 110.
    // weighted with cached_input 0.1: 10 + 9 + 10 = 29.
    // Reserve 50 → refund 21 → 71 remain → next 50 fits.
    let yaml = single_rule_yaml_with(
        "default_weights:\n  cached_input: 0.1",
        "algorithm: sliding_window\nwindow: 1h\ncapacity: 100\nreserved_tokens: 50",
    );
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    set_typed_usage(&mut ctx, 100, 10, 110);
    ctx.set_metadata(META_TOKEN_CACHE_READ, "90");
    let mut body = None;
    drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());

    let mut next_ctx = crate::test_utils::make_filter_context(&req);
    assert!(
        matches!(filter.on_request(&mut next_ctx).await.unwrap(), FilterAction::Continue),
        "weighted cost 29 of 100 should leave room for another 50-token reservation"
    );
}

#[tokio::test]
async fn unweighted_total_of_the_same_cached_request_would_starve_the_window() {
    // Control: same counts as weighted_cache_hit_refunds_more_than_token_total
    // but no default_weights, and only token.total=110 is published (the
    // fallback path). 50 reserved, 110 actual → window holds 110 of 100,
    // next 50 is denied.
    let yaml = single_rule_yaml("algorithm: sliding_window\nwindow: 1h\ncapacity: 100\nreserved_tokens: 50");
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());
    ctx.set_metadata(META_TOKEN_TOTAL, "110");
    let mut body = None;
    drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());

    let mut next_ctx = crate::test_utils::make_filter_context(&req);
    assert!(
        matches!(filter.on_request(&mut next_ctx).await.unwrap(), FilterAction::Reject(_)),
        "charging token.total=110 against capacity 100 must starve the next 50-token request"
    );
}

#[tokio::test]
async fn per_rule_weight_overlay_is_isolated_from_filter_defaults() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        "default_weights:\n\
         \x20 cached_input: 0.1\n\
         rules:\n\
         \x20 - name: cheap-cache\n\
         \x20   match:\n\
         \x20     headers:\n\
         \x20       x-app-id: cheap\n\
         \x20   algorithm: sliding_window\n\
         \x20   window: 1h\n\
         \x20   capacity: 100\n\
         \x20   reserved_tokens: 50\n\
         \x20   weights:\n\
         \x20     cached_input: 0.0\n\
         \x20 - name: default-cache\n\
         \x20   match:\n\
         \x20     headers:\n\
         \x20       x-app-id: full\n\
         \x20   algorithm: sliding_window\n\
         \x20   window: 1h\n\
         \x20   capacity: 75\n\
         \x20   reserved_tokens: 50\n",
    )
    .unwrap();
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();

    // cheap-cache: uncached 10 + cache 90*0 + output 10 = 20. Refund 30.
    let cheap_req = make_request_with_header("x-app-id", "cheap");
    let mut cheap_ctx = crate::test_utils::make_filter_context(&cheap_req);
    drop(filter.on_request(&mut cheap_ctx).await.unwrap());
    set_typed_usage(&mut cheap_ctx, 100, 10, 110);
    cheap_ctx.set_metadata(META_TOKEN_CACHE_READ, "90");
    let mut body = None;
    drop(filter.on_response_body(&mut cheap_ctx, &mut body, true).unwrap());

    let mut cheap_next = crate::test_utils::make_filter_context(&cheap_req);
    assert!(
        matches!(
            filter.on_request(&mut cheap_next).await.unwrap(),
            FilterAction::Continue
        ),
        "cached_input 0.0 should charge only 20, leaving room for another 50"
    );

    // default-cache uses filter cached_input 0.1 on the same typed counts:
    // cost 29. capacity 75 → remaining 46, so the next 50-token reserve
    // is denied. If the 0.0 overlay leaked onto this rule, cost would be
    // 20, remaining 55, and the next reserve would be admitted.
    let full_req = make_request_with_header("x-app-id", "full");
    let mut full_ctx = crate::test_utils::make_filter_context(&full_req);
    drop(filter.on_request(&mut full_ctx).await.unwrap());
    set_typed_usage(&mut full_ctx, 100, 10, 110);
    full_ctx.set_metadata(META_TOKEN_CACHE_READ, "90");
    drop(filter.on_response_body(&mut full_ctx, &mut body, true).unwrap());
    let mut full_next = crate::test_utils::make_filter_context(&full_req);
    assert!(
        matches!(
            filter.on_request(&mut full_next).await.unwrap(),
            FilterAction::Reject(_)
        ),
        "filter-wide cached_input 0.1 must still apply (cost 29 of 75); a leaked 0.0 overlay would admit"
    );
}

#[tokio::test]
async fn google_additive_reasoning_does_not_drop_visible_output() {
    // input 50 + output 80 + reasoning 200, total 330, reasoning weight 0.9
    // additive cost 50+80+180 = 310. Nested subtract would charge ~50+180.
    let yaml = single_rule_yaml_with(
        "default_weights:\n  reasoning: 0.9",
        "algorithm: sliding_window\nwindow: 1h\ncapacity: 400\nreserved_tokens: 50",
    );
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());
    set_typed_usage(&mut ctx, 50, 80, 330);
    ctx.set_metadata(META_TOKEN_REASONING, "200");
    let mut body = None;
    drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());

    // 400 - 310 = 90 remaining. A 50-token reserve fits; a third does not.
    let mut next_ctx = crate::test_utils::make_filter_context(&req);
    assert!(
        matches!(filter.on_request(&mut next_ctx).await.unwrap(), FilterAction::Continue),
        "310 of 400 used should admit another 50"
    );
    let mut third_ctx = crate::test_utils::make_filter_context(&req);
    assert!(
        matches!(
            filter.on_request(&mut third_ctx).await.unwrap(),
            FilterAction::Reject(_)
        ),
        "90 remaining after the second 50-token reserve should deny a third"
    );
}

#[tokio::test]
async fn openai_nested_reasoning_is_not_double_counted() {
    // input 120, output 800 (640 reasoning nested), total 920, reasoning 0.9
    // nested cost = 120 + 160 + 576 = 856. Additive would be 120+800+576 = 1496.
    // capacity 1000: nested leaves 144 (next 50 admits); additive overshoots (denies).
    let yaml = single_rule_yaml_with(
        "default_weights:\n  reasoning: 0.9",
        "algorithm: sliding_window\nwindow: 1h\ncapacity: 1000\nreserved_tokens: 50",
    );
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());
    set_typed_usage(&mut ctx, 120, 800, 920);
    ctx.set_metadata(META_TOKEN_REASONING, "640");
    let mut body = None;
    drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());

    let mut next_ctx = crate::test_utils::make_filter_context(&req);
    assert!(
        matches!(filter.on_request(&mut next_ctx).await.unwrap(), FilterAction::Continue),
        "nested reasoning cost 856 of 1000 must leave room for another 50; additive 1496 would not"
    );
}

#[tokio::test]
async fn overflow_status_keeps_the_reservation_estimate() {
    let yaml = single_rule_yaml("algorithm: sliding_window\nwindow: 1h\ncapacity: 100\nreserved_tokens: 50");
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());
    set_typed_usage(&mut ctx, 10, 10, 20);
    ctx.set_metadata(META_TOKEN_STATUS, TOKEN_STATUS_OVERFLOW);
    let mut body = None;
    drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());

    // Settled at the 50-token estimate, not the 20 typed cost: 50 of 100 used,
    // a second 50 fits exactly, a third does not.
    let mut second = crate::test_utils::make_filter_context(&req);
    assert!(matches!(
        filter.on_request(&mut second).await.unwrap(),
        FilterAction::Continue
    ));
    let mut third = crate::test_utils::make_filter_context(&req);
    assert!(matches!(
        filter.on_request(&mut third).await.unwrap(),
        FilterAction::Reject(_)
    ));
}

#[tokio::test]
async fn large_finite_weight_does_not_settle_at_zero() {
    // 1e308 is finite (config accepts it) but 2 × 1e308 overflows to +inf.
    // Saturating to COST_U64_CAP starves capacity 100. Settling at 0 would
    // refund the 50-token reservation and admit the next request.
    let yaml = single_rule_yaml_with(
        "default_weights:\n  input: 1.0e308",
        "algorithm: sliding_window\nwindow: 1h\ncapacity: 100\nreserved_tokens: 50",
    );
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());
    set_typed_usage(&mut ctx, 2, 0, 2);
    let mut body = None;
    drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());

    let mut next_ctx = crate::test_utils::make_filter_context(&req);
    assert!(
        matches!(filter.on_request(&mut next_ctx).await.unwrap(), FilterAction::Reject(_)),
        "overflowed weighted cost must saturate, not settle at zero and refund the reservation"
    );
}

#[tokio::test]
async fn anthropic_cache_write_weight_is_applied() {
    // token.input = 6050 = 50 uncached + 5000 read + 1000 write, output 100.
    // cached_input 0.1, cache_write 1.25 → 50 + 500 + 1250 + 100 = 1900.
    let yaml = single_rule_yaml_with(
        "default_weights:\n  cached_input: 0.1\n  cache_write: 1.25",
        "algorithm: sliding_window\nwindow: 1h\ncapacity: 2000\nreserved_tokens: 50",
    );
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());
    set_typed_usage(&mut ctx, 6050, 100, 6150);
    ctx.set_metadata(META_TOKEN_CACHE_READ, "5000");
    ctx.set_metadata(META_TOKEN_CACHE_WRITE, "1000");
    let mut body = None;
    drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());

    // 2000 - 1900 = 100 left. 50 fits, second 50 after that would also fit
    // (100 >= 50, then 50 left), third would fail. Two more 50s then deny.
    let mut a = crate::test_utils::make_filter_context(&req);
    assert!(matches!(
        filter.on_request(&mut a).await.unwrap(),
        FilterAction::Continue
    ));
    let mut b = crate::test_utils::make_filter_context(&req);
    assert!(matches!(
        filter.on_request(&mut b).await.unwrap(),
        FilterAction::Continue
    ));
    let mut c = crate::test_utils::make_filter_context(&req);
    assert!(matches!(
        filter.on_request(&mut c).await.unwrap(),
        FilterAction::Reject(_)
    ));
}

#[tokio::test]
async fn bedrock_cache_hidden_in_total_is_not_dropped() {
    // Pre-normalization Converse metadata: inputTokens is uncached-only,
    // cache lives only in token.total. Residual must charge 1289, not 223.
    // capacity 1300, reserved 50 → remaining 11, so the next 50 is denied.
    // Without the residual the next reserve would fit.
    let yaml = single_rule_yaml("algorithm: sliding_window\nwindow: 1h\ncapacity: 1300\nreserved_tokens: 50");
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());
    set_typed_usage(&mut ctx, 9, 214, 1289);
    let mut body = None;
    drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());

    let mut next_ctx = crate::test_utils::make_filter_context(&req);
    assert!(
        matches!(filter.on_request(&mut next_ctx).await.unwrap(), FilterAction::Reject(_)),
        "uncovered Bedrock cache in token.total must be charged; cost 223 would admit the next 50"
    );
}

#[tokio::test]
async fn bedrock_normalized_cache_breakdown_applies_cached_input_weight() {
    // After parse_bedrock folds cache into input: 9+1066=1075, cache_read 1066.
    // cached_input 0.1 → cost 330. capacity 380, reserved 50 → remaining 50,
    // so the next 50 admits and a third is denied. Residual-at-input-weight
    // would charge 1289 and deny the second request.
    let yaml = single_rule_yaml_with(
        "default_weights:\n  cached_input: 0.1",
        "algorithm: sliding_window\nwindow: 1h\ncapacity: 380\nreserved_tokens: 50",
    );
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());
    set_typed_usage(&mut ctx, 1075, 214, 1289);
    ctx.set_metadata(META_TOKEN_CACHE_READ, "1066");
    let mut body = None;
    drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());

    let mut second = crate::test_utils::make_filter_context(&req);
    assert!(
        matches!(filter.on_request(&mut second).await.unwrap(), FilterAction::Continue),
        "normalized Bedrock cache must take cached_input 0.1 (cost 330 of 380)"
    );
    let mut third = crate::test_utils::make_filter_context(&req);
    assert!(
        matches!(filter.on_request(&mut third).await.unwrap(), FilterAction::Reject(_)),
        "50 remaining after the second 50-token reserve should deny a third"
    );
}

#[tokio::test]
async fn token_bucket_applies_the_same_weighted_cost() {
    let yaml = single_rule_yaml_with(
        "default_weights:\n  cached_input: 0.1",
        "algorithm: token_bucket\ncapacity: 100\nrefill_rate: 0.0001\nreserved_tokens: 50",
    );
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());
    set_typed_usage(&mut ctx, 100, 10, 110);
    ctx.set_metadata(META_TOKEN_CACHE_READ, "90");
    let mut body = None;
    drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());

    // Bucket started at 100, reserved 50, credited back 21 (50-29) → 71.
    // Next 50 fits.
    let mut next_ctx = crate::test_utils::make_filter_context(&req);
    assert!(
        matches!(filter.on_request(&mut next_ctx).await.unwrap(), FilterAction::Continue),
        "token_bucket must apply the same partitioned weighted cost as sliding_window"
    );
}

// -----------------------------------------------------------------------------
// S1: Graduated soft-limit tiers (inject action)
// -----------------------------------------------------------------------------

/// Helper: build a config with tiers on a `sliding_window` rule.
fn tiered_rule_yaml(capacity: u64, reserved_tokens: u64, tiers_yaml: &str) -> serde_yaml::Value {
    let indented_tiers = tiers_yaml
        .lines()
        .map(|line| format!("      {line}"))
        .collect::<Vec<_>>()
        .join("\n");
    let yaml = format!(
        "rules:\n\
         \x20 - name: default\n\
         \x20   algorithm: sliding_window\n\
         \x20   window: 1h\n\
         \x20   capacity: {capacity}\n\
         \x20   reserved_tokens: {reserved_tokens}\n\
         \x20   tiers:\n\
         {indented_tiers}\n"
    );
    serde_yaml::from_str(&yaml).unwrap()
}

// -- Config validation --

#[test]
fn tier_config_parses_valid_inject_and_deny_tiers() {
    let yaml = tiered_rule_yaml(
        100,
        10,
        "- capacity: 80\n  action:\n    type: inject\n    headers:\n      X-Token-Tier: warning\n\
         - capacity: 100\n  action:\n    type: deny",
    );
    assert!(TokenRateLimitFilter::from_config(&yaml).is_ok());
}

#[test]
fn tier_config_rejects_empty_tiers_list() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        "rules:\n  - name: default\n    algorithm: sliding_window\n    window: 1h\n    capacity: 100\n    \
         reserved_tokens: 10\n    tiers: []\n",
    )
    .unwrap();
    let err = TokenRateLimitFilter::from_config(&yaml).err().expect("should error");
    assert!(err.to_string().contains("must not be empty"), "got: {err}");
}

#[test]
fn tier_config_rejects_non_ascending_capacities() {
    let yaml = tiered_rule_yaml(
        100,
        10,
        "- capacity: 90\n  action:\n    type: inject\n    headers:\n      X-Token-Tier: high\n\
         - capacity: 80\n  action:\n    type: inject\n    headers:\n      X-Token-Tier: low\n\
         - capacity: 100\n  action:\n    type: deny",
    );
    let err = TokenRateLimitFilter::from_config(&yaml).err().expect("should error");
    assert!(err.to_string().contains("strictly ascending"), "got: {err}");
}

#[test]
fn tier_config_rejects_deny_tier_not_last() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        "rules:\n\
         \x20 - name: default\n\
         \x20   algorithm: sliding_window\n\
         \x20   window: 1h\n\
         \x20   capacity: 200\n\
         \x20   reserved_tokens: 10\n\
         \x20   tiers:\n\
         \x20     - capacity: 100\n\
         \x20       action:\n\
         \x20         type: inject\n\
         \x20         headers:\n\
         \x20           X-Token-Tier: low\n\
         \x20     - capacity: 200\n\
         \x20       action:\n\
         \x20         type: deny\n\
         \x20     - capacity: 300\n\
         \x20       action:\n\
         \x20         type: inject\n\
         \x20         headers:\n\
         \x20           X-Token-Tier: over\n",
    )
    .unwrap();
    let err = TokenRateLimitFilter::from_config(&yaml).err().expect("should error");
    assert!(err.to_string().contains("deny tier must be the last"), "got: {err}");
}

#[test]
fn tier_config_rejects_deny_capacity_mismatch() {
    let yaml = tiered_rule_yaml(
        100,
        10,
        "- capacity: 80\n  action:\n    type: inject\n    headers:\n      X-Token-Tier: warning\n\
         - capacity: 90\n  action:\n    type: deny",
    );
    let err = TokenRateLimitFilter::from_config(&yaml).err().expect("should error");
    assert!(err.to_string().contains("must equal the algorithm"), "got: {err}");
}

#[test]
fn tier_config_rejects_inject_without_headers() {
    let yaml = tiered_rule_yaml(
        100,
        10,
        "- capacity: 80\n  action:\n    type: inject\n\
         - capacity: 100\n  action:\n    type: deny",
    );
    let err = TokenRateLimitFilter::from_config(&yaml).err().expect("should error");
    assert!(err.to_string().contains("at least one header"), "got: {err}");
}

#[test]
fn tier_config_rejects_zero_capacity_tier() {
    let yaml = tiered_rule_yaml(
        100,
        10,
        "- capacity: 0\n  action:\n    type: inject\n    headers:\n      X-Token-Tier: bad\n\
         - capacity: 100\n  action:\n    type: deny",
    );
    let err = TokenRateLimitFilter::from_config(&yaml).err().expect("should error");
    assert!(err.to_string().contains("capacity > 0"), "got: {err}");
}

#[test]
fn tier_config_rejects_invalid_header_name() {
    let yaml = tiered_rule_yaml(
        100,
        10,
        "- capacity: 80\n  action:\n    type: inject\n    headers:\n      \"invalid header!\": value\n\
         - capacity: 100\n  action:\n    type: deny",
    );
    let err = TokenRateLimitFilter::from_config(&yaml).err().expect("should error");
    assert!(err.to_string().contains("invalid inject header"), "got: {err}");
}

#[test]
fn tier_config_rejects_inject_tier_above_algorithm_capacity() {
    let yaml = tiered_rule_yaml(
        100,
        10,
        "- capacity: 80\n  action:\n    type: inject\n    headers:\n      X-Token-Tier: warning\n\
         - capacity: 120\n  action:\n    type: inject\n    headers:\n      X-Token-Tier: over",
    );
    let err = TokenRateLimitFilter::from_config(&yaml).err().expect("should error");
    assert!(err.to_string().contains("above the algorithm"), "got: {err}");
}

#[test]
fn tier_config_rejects_reserved_header_name() {
    let yaml = tiered_rule_yaml(
        100,
        10,
        "- capacity: 80\n  action:\n    type: inject\n    headers:\n      content-length: \"42\"",
    );
    let err = TokenRateLimitFilter::from_config(&yaml).err().expect("should error");
    assert!(err.to_string().contains("reserved/hop-by-hop"), "got: {err}");
}

#[test]
fn tier_config_rejects_hop_by_hop_header_name() {
    let yaml = tiered_rule_yaml(
        100,
        10,
        "- capacity: 80\n  action:\n    type: inject\n    headers:\n      transfer-encoding: chunked",
    );
    let err = TokenRateLimitFilter::from_config(&yaml).err().expect("should error");
    assert!(err.to_string().contains("reserved/hop-by-hop"), "got: {err}");
}

#[test]
fn tier_config_rejects_duplicate_header_after_normalization() {
    let yaml = tiered_rule_yaml(
        100,
        10,
        "- capacity: 80\n  action:\n    type: inject\n    headers:\n      X-Token-Tier: warning\n      x-token-tier: degraded",
    );
    let err = TokenRateLimitFilter::from_config(&yaml).err().expect("should error");
    assert!(err.to_string().contains("duplicate header"), "got: {err}");
}

#[test]
fn tier_config_allows_inject_only_tiers_without_deny() {
    let yaml = tiered_rule_yaml(
        100,
        10,
        "- capacity: 80\n  action:\n    type: inject\n    headers:\n      X-Token-Tier: warning\n\
         - capacity: 95\n  action:\n    type: inject\n    headers:\n      X-Token-Tier: degraded",
    );
    assert!(
        TokenRateLimitFilter::from_config(&yaml).is_ok(),
        "inject-only tiers (no deny) are valid for soft enforcement"
    );
}

#[test]
fn no_tiers_preserves_backward_compatible_behavior() {
    let yaml = single_rule_yaml("algorithm: sliding_window\nwindow: 1h\ncapacity: 100\nreserved_tokens: 50");
    assert!(
        TokenRateLimitFilter::from_config(&yaml).is_ok(),
        "a rule with no tiers must still parse as before"
    );
}

// -- Admission with tier header injection --

#[tokio::test]
async fn inject_tier_adds_headers_when_usage_exceeds_threshold() {
    // capacity 100, reserve 60. After first request: usage_after=60, which
    // exceeds the 50-token inject tier → header should be injected.
    let yaml = tiered_rule_yaml(
        100,
        60,
        "- capacity: 50\n  action:\n    type: inject\n    headers:\n      X-Token-Tier: warning\n\
         - capacity: 100\n  action:\n    type: deny",
    );
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();
    assert!(matches!(action, FilterAction::Continue), "should be admitted");
    let injected = ctx
        .request_headers_to_set
        .iter()
        .find(|(name, _)| name.as_str() == "x-token-tier");
    assert!(injected.is_some(), "inject tier header should be set");
    assert_eq!(
        injected.unwrap().1.to_str().unwrap(),
        "warning",
        "header value should match the tier config"
    );
}

#[tokio::test]
async fn no_headers_injected_when_usage_is_below_all_tiers() {
    // capacity 100, reserve 10. After first request: usage_after=10,
    // below the 50-token inject tier → no headers.
    let yaml = tiered_rule_yaml(
        100,
        10,
        "- capacity: 50\n  action:\n    type: inject\n    headers:\n      X-Token-Tier: warning\n\
         - capacity: 100\n  action:\n    type: deny",
    );
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));
    assert!(
        ctx.request_headers_to_set.is_empty(),
        "no tier headers should be injected when usage is below all thresholds"
    );
}

#[tokio::test]
async fn highest_breached_tier_header_wins_for_the_same_header_name() {
    // capacity 100, reserve 80. usage_after=80 breaches both the 50 and
    // 70 tiers. Both inject X-Token-Tier but with different values.
    // The last pushed value (70-tier's "degraded") wins.
    let yaml = tiered_rule_yaml(
        100,
        80,
        "- capacity: 50\n  action:\n    type: inject\n    headers:\n      X-Token-Tier: warning\n\
         - capacity: 70\n  action:\n    type: inject\n    headers:\n      X-Token-Tier: degraded\n\
         - capacity: 100\n  action:\n    type: deny",
    );
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));
    let tier_headers: Vec<_> = ctx
        .request_headers_to_set
        .iter()
        .filter(|(name, _)| name.as_str() == "x-token-tier")
        .collect();
    assert_eq!(tier_headers.len(), 2, "both breached tiers push their header");
    assert_eq!(
        tier_headers.last().unwrap().1.to_str().unwrap(),
        "degraded",
        "the highest breached tier's value should be last (wins for same-name headers)"
    );
}

#[tokio::test]
async fn multiple_distinct_headers_from_different_tiers_are_all_injected() {
    // capacity 100, reserve 80. usage_after=80 breaches both tiers.
    // Tier 1 injects X-Token-Hour-Tier; Tier 2 injects a different header.
    let yaml = tiered_rule_yaml(
        100,
        80,
        "- capacity: 50\n  action:\n    type: inject\n    headers:\n      X-Token-Hour-Tier: warning\n\
         - capacity: 70\n  action:\n    type: inject\n    headers:\n      X-Fairness-Id: \"85\"\n\
         - capacity: 100\n  action:\n    type: deny",
    );
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));
    let hour_tier = ctx
        .request_headers_to_set
        .iter()
        .find(|(name, _)| name.as_str() == "x-token-hour-tier");
    let fairness = ctx
        .request_headers_to_set
        .iter()
        .find(|(name, _)| name.as_str() == "x-fairness-id");
    assert_eq!(
        hour_tier.unwrap().1.to_str().unwrap(),
        "warning",
        "first tier's header should be injected"
    );
    assert_eq!(
        fairness.unwrap().1.to_str().unwrap(),
        "85",
        "second tier's header should also be injected"
    );
}

#[tokio::test]
async fn deny_tier_still_rejects_when_budget_exhausted() {
    // capacity 100, reserve 60. First request: usage_after=60, admitted
    // with warning header. Second request: needs 60 more, only 40 left → 429.
    let yaml = tiered_rule_yaml(
        100,
        60,
        "- capacity: 50\n  action:\n    type: inject\n    headers:\n      X-Token-Tier: warning\n\
         - capacity: 100\n  action:\n    type: deny",
    );
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");

    let mut first_ctx = crate::test_utils::make_filter_context(&req);
    let first = filter.on_request(&mut first_ctx).await.unwrap();
    assert!(
        matches!(first, FilterAction::Continue),
        "first request should be admitted"
    );

    let mut second_ctx = crate::test_utils::make_filter_context(&req);
    let second = filter.on_request(&mut second_ctx).await.unwrap();
    assert!(
        matches!(second, FilterAction::Reject(_)),
        "second request should be denied (429) when capacity is exhausted"
    );
}

#[tokio::test]
async fn inject_only_tiers_never_deny() {
    // No deny tier. capacity 100, reserve 60. Two requests: 60+60=120 > 100.
    // Without a deny tier in the tiers list, the backend's capacity (100)
    // still governs the deny decision.
    let yaml = tiered_rule_yaml(
        100,
        60,
        "- capacity: 50\n  action:\n    type: inject\n    headers:\n      X-Token-Tier: warning\n\
         - capacity: 80\n  action:\n    type: inject\n    headers:\n      X-Token-Tier: degraded",
    );
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");

    let mut first_ctx = crate::test_utils::make_filter_context(&req);
    let first = filter.on_request(&mut first_ctx).await.unwrap();
    assert!(
        matches!(first, FilterAction::Continue),
        "first request should be admitted with inject headers"
    );
    let tier_header = first_ctx
        .request_headers_to_set
        .iter()
        .find(|(name, _)| name.as_str() == "x-token-tier");
    assert_eq!(
        tier_header.unwrap().1.to_str().unwrap(),
        "warning",
        "usage_after 60 breaches the 50 tier but not the 80 tier"
    );
}

#[tokio::test]
async fn tier_evaluation_with_token_bucket_algorithm() {
    // Same tier behavior should work with token_bucket.
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        "rules:\n\
         \x20 - name: default\n\
         \x20   algorithm: token_bucket\n\
         \x20   capacity: 100\n\
         \x20   refill_rate: 0.0001\n\
         \x20   reserved_tokens: 60\n\
         \x20   tiers:\n\
         \x20     - capacity: 50\n\
         \x20       action:\n\
         \x20         type: inject\n\
         \x20         headers:\n\
         \x20           X-Token-Tier: warning\n\
         \x20     - capacity: 100\n\
         \x20       action:\n\
         \x20         type: deny\n",
    )
    .unwrap();
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();
    assert!(matches!(action, FilterAction::Continue), "should be admitted");
    let injected = ctx
        .request_headers_to_set
        .iter()
        .find(|(name, _)| name.as_str() == "x-token-tier");
    assert!(injected.is_some(), "token_bucket should also evaluate inject tiers");
}

#[tokio::test]
async fn tier_headers_injected_from_on_request_body_path() {
    // Body-dependent estimation with tiers.
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        "rules:\n\
         \x20 - name: default\n\
         \x20   algorithm: sliding_window\n\
         \x20   window: 1h\n\
         \x20   capacity: 200\n\
         \x20   estimation:\n\
         \x20     strategy: max_tokens\n\
         \x20     fallback_estimate: 100\n\
         \x20   tiers:\n\
         \x20     - capacity: 80\n\
         \x20       action:\n\
         \x20         type: inject\n\
         \x20         headers:\n\
         \x20           X-Token-Tier: warning\n\
         \x20     - capacity: 200\n\
         \x20       action:\n\
         \x20         type: deny\n",
    )
    .unwrap();
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let mut body = Some(bytes::Bytes::from(r#"{"max_tokens": 100}"#));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));
    let injected = ctx
        .request_headers_to_set
        .iter()
        .find(|(name, _)| name.as_str() == "x-token-tier");
    assert!(
        injected.is_some(),
        "inject tier should fire from the on_request_body path too (usage_after=100 > 80)"
    );
}

#[tokio::test]
async fn tiers_with_match_condition_only_apply_to_matching_requests() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        "rules:\n\
         \x20 - name: team-alpha\n\
         \x20   match:\n\
         \x20     headers:\n\
         \x20       x-app-id: alpha\n\
         \x20   algorithm: sliding_window\n\
         \x20   window: 1h\n\
         \x20   capacity: 100\n\
         \x20   reserved_tokens: 60\n\
         \x20   tiers:\n\
         \x20     - capacity: 50\n\
         \x20       action:\n\
         \x20         type: inject\n\
         \x20         headers:\n\
         \x20           X-Token-Tier: warning\n\
         \x20     - capacity: 100\n\
         \x20       action:\n\
         \x20         type: deny\n",
    )
    .unwrap();
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();

    // Request matching the rule: should get inject headers.
    let alpha_req = make_request_with_header("x-app-id", "alpha");
    let mut alpha_ctx = crate::test_utils::make_filter_context(&alpha_req);
    let action = filter.on_request(&mut alpha_ctx).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));
    assert!(
        alpha_ctx
            .request_headers_to_set
            .iter()
            .any(|(name, _)| name.as_str() == "x-token-tier"),
        "matching request should get inject tier headers"
    );

    // Request not matching: should pass through without tiers or rate limiting.
    let other_req = make_request_with_header("x-app-id", "beta");
    let mut other_ctx = crate::test_utils::make_filter_context(&other_req);
    let action = filter.on_request(&mut other_ctx).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));
    assert!(
        other_ctx.request_headers_to_set.is_empty(),
        "non-matching request should not get any injected headers"
    );
}

/// Client-supplied tier header is stripped on admission, preventing
/// a below-threshold client from spoofing the tier signal.
#[tokio::test]
async fn client_supplied_tier_header_is_stripped_on_admission() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        "rules:\n\
         \x20 - name: default\n\
         \x20   algorithm: sliding_window\n\
         \x20   window: 1h\n\
         \x20   capacity: 100\n\
         \x20   reserved_tokens: 10\n\
         \x20   tiers:\n\
         \x20     - capacity: 50\n\
         \x20       action:\n\
         \x20         type: inject\n\
         \x20         headers:\n\
         \x20           X-Token-Tier: warning\n\
         \x20     - capacity: 100\n\
         \x20       action:\n\
         \x20         type: deny\n",
    )
    .unwrap();
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();

    // Client sends the inject header pre-emptively.
    let mut req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    req.headers.insert("x-token-tier", "spoofed".parse().unwrap());
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));

    // The spoofed header should be in request_headers_to_remove.
    assert!(
        ctx.request_headers_to_remove
            .iter()
            .any(|n| n.as_str() == "x-token-tier"),
        "client-supplied inject header name must be stripped"
    );
}

/// When `pre_read_mutations` is already active (body phase with earlier
/// ordered producers), `evaluate_tiers` pushes inject headers into
/// the ordered log as well as `request_headers_to_set`.
#[tokio::test]
async fn tier_headers_join_pre_read_mutations_when_ordered_log_is_active() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        "rules:\n\
         \x20 - name: default\n\
         \x20   algorithm: sliding_window\n\
         \x20   window: 1h\n\
         \x20   capacity: 100\n\
         \x20   estimation:\n\
         \x20     strategy: max_tokens\n\
         \x20   tiers:\n\
         \x20     - capacity: 50\n\
         \x20       action:\n\
         \x20         type: inject\n\
         \x20         headers:\n\
         \x20           X-Token-Tier: warning\n\
         \x20     - capacity: 100\n\
         \x20       action:\n\
         \x20         type: deny\n",
    )
    .unwrap();
    let filter = TokenRateLimitFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    // Simulate an earlier filter having activated the ordered log.
    ctx.pre_read_mutations.push(praxis_filter::TrustedHeaderMutation::Set(
        http::HeaderName::from_static("x-tenant-id"),
        http::HeaderValue::from_static("acme"),
    ));

    let mut body = Some(bytes::Bytes::from_static(br#"{"max_tokens":60}"#));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));

    // The inject header should appear in both queues.
    assert!(
        ctx.request_headers_to_set
            .iter()
            .any(|(n, _)| n.as_str() == "x-token-tier"),
        "inject header should be in request_headers_to_set"
    );
    assert!(
        ctx.pre_read_mutations.iter().any(|m| matches!(
            m,
            praxis_filter::TrustedHeaderMutation::Set(name, _) if name.as_str() == "x-token-tier"
        )),
        "inject header should also be in pre_read_mutations when ordered log is active"
    );
}
