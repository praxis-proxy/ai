// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Integration tests for the `token_rate_limit` filter's example config.
//!
//! Covers the uncontested MVP core of the token rate limiting proposal
//! (`docs/proposals/00121_token-rate-limiting.md`, epic `ai#121`, PR
//! `ai#658`): reservation-based admission, 429 rejection with
//! token-denominated headers, and reconciliation against actual
//! provider-reported usage (`token_count`'s `token.total`) once the
//! response completes.

use std::collections::HashMap;

use praxis_test_utils::{
    Backend, example_config_path, free_port, http_send, json_post, load_example_config, parse_body, parse_header,
    parse_status, patch_yaml, start_proxy,
};

// -----------------------------------------------------------------------------
// Mock response bodies
// -----------------------------------------------------------------------------

/// OpenAI-shaped response reporting 10 total tokens used.
const OPENAI_LOW_USAGE_JSON: &str =
    r#"{"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":5,"total_tokens":10}}"#;

/// Plain-text response with no token usage: `token_count` extracts
/// nothing, so `token_rate_limit`'s reservation is never reconciled.
const PLAIN_TEXT_BODY: &str = "ok";

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

/// Build a YAML config for the token rate limiting pipeline using the
/// example file, substituting `burst`/`estimate_tokens` with the given
/// values so tests can exercise small, deterministic budgets.
fn token_rate_limit_config(
    proxy_port: u16,
    backend_port: u16,
    burst: u64,
    estimate_tokens: u64,
) -> praxis_core::config::Config {
    let path = example_config_path("token-rate-limit.yaml");
    let yaml = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let yaml = yaml
        .replace("burst: 100000", &format!("burst: {burst}"))
        .replace("estimate_tokens: 500", &format!("estimate_tokens: {estimate_tokens}"));
    let patched = patch_yaml(&yaml, proxy_port, &HashMap::from([("127.0.0.1:3000", backend_port)]));
    praxis_core::config::Config::from_yaml(&patched).expect("config should parse")
}

// -----------------------------------------------------------------------------
// Admission and headers
// -----------------------------------------------------------------------------

#[test]
fn admits_request_within_budget_and_injects_token_headers() {
    let backend = Backend::fixed(PLAIN_TEXT_BODY)
        .header("content-type", "text/plain")
        .start_with_shutdown();
    let proxy_port = free_port();
    let config = token_rate_limit_config(proxy_port, backend.port(), 100_000, 500);
    let proxy = start_proxy(&config);

    let raw = http_send(proxy.addr(), &json_post("/v1/chat/completions", "{}"));
    assert_eq!(parse_status(&raw), 200, "request within budget should be admitted");
    assert_eq!(parse_body(&raw), PLAIN_TEXT_BODY, "body should pass through unchanged");
    assert_eq!(
        parse_header(&raw, "x-ratelimit-limit-tokens"),
        Some("100000".to_owned()),
        "limit header should reflect configured burst"
    );
    assert!(
        parse_header(&raw, "x-ratelimit-remaining-tokens").is_some(),
        "remaining-tokens header should be present"
    );
    assert!(
        parse_header(&raw, "x-ratelimit-reset-tokens").is_some(),
        "reset header should be present"
    );
}

#[test]
fn rejects_with_429_and_retry_after_when_estimate_budget_exhausted() {
    let backend = Backend::fixed(PLAIN_TEXT_BODY)
        .header("content-type", "text/plain")
        .start_with_shutdown();
    let proxy_port = free_port();
    // burst=90, estimate=40. start_proxy's readiness probe (a real GET /
    // through the filter chain) consumes one reservation before the test
    // body runs, and this backend returns no usage info so it's never
    // reconciled: probe -40 -> 50 left, first request -40 -> 10 left
    // (200), second request needs 40 more and must be rejected.
    let config = token_rate_limit_config(proxy_port, backend.port(), 90, 40);
    let proxy = start_proxy(&config);

    let first = http_send(proxy.addr(), &json_post("/v1/chat/completions", "{}"));
    assert_eq!(parse_status(&first), 200, "first request should be admitted");

    let second = http_send(proxy.addr(), &json_post("/v1/chat/completions", "{}"));
    assert_eq!(
        parse_status(&second),
        429,
        "second request should be rejected, only 10 of 90 tokens remain (probe + first request each spent 40)"
    );
    assert!(
        parse_header(&second, "retry-after").is_some(),
        "429 should carry a Retry-After header"
    );
    assert!(
        parse_header(&second, "x-ratelimit-limit-tokens").is_some(),
        "429 should carry the token-suffixed limit header"
    );
    assert!(
        parse_header(&second, "x-ratelimit-remaining-tokens").is_some(),
        "429 should carry the token-suffixed remaining header"
    );
}

// -----------------------------------------------------------------------------
// Reconciliation against actual usage
// -----------------------------------------------------------------------------

#[test]
fn reconciliation_frees_budget_for_next_request_after_low_actual_usage() {
    let backend = Backend::fixed(OPENAI_LOW_USAGE_JSON)
        .header("content-type", "application/json")
        .start_with_shutdown();
    let proxy_port = free_port();
    // burst=60, estimate=40. Every admitted request (including
    // start_proxy's own readiness-probe GET / through the filter
    // chain) reserves 40 and then, since this backend's actual usage
    // (10) is far below the estimate, gets 30 released back on
    // reconciliation — so the bucket settles into a steady drain of
    // only 10 net tokens per request instead of monotonically losing
    // the full 40. Starting from a full bucket of 60:
    //   probe:   reserve 40 (60->20), reconcile +30 -> 50
    //   first:   reserve 40 (50->10), reconcile +30 -> 40   [200]
    //   second:  reserve 40 (40->0),  reconcile +30 -> 30   [200]
    //   third:   needs 40, only 30 remain                  [429]
    // A naive (non-reconciling) reservation scheme would already have
    // rejected "second" (it would see only 10 remaining after "first").
    let config = token_rate_limit_config(proxy_port, backend.port(), 60, 40);
    let proxy = start_proxy(&config);

    let first = http_send(proxy.addr(), &json_post("/v1/chat/completions", "{}"));
    assert_eq!(parse_status(&first), 200, "first request should be admitted");
    assert_eq!(
        parse_body(&first),
        OPENAI_LOW_USAGE_JSON,
        "body should pass through unchanged"
    );

    let second = http_send(proxy.addr(), &json_post("/v1/chat/completions", "{}"));
    assert_eq!(
        parse_status(&second),
        200,
        "reconciliation should have released enough budget (50 remaining) to admit a second 40-token request"
    );

    let third = http_send(proxy.addr(), &json_post("/v1/chat/completions", "{}"));
    assert_eq!(
        parse_status(&third),
        429,
        "after two admissions the bucket should be down to 10 remaining, rejecting a third 40-token request"
    );
}

// -----------------------------------------------------------------------------
// Example config smoke test
// -----------------------------------------------------------------------------

#[test]
fn example_config_token_rate_limit() {
    let backend = Backend::fixed(PLAIN_TEXT_BODY)
        .header("content-type", "text/plain")
        .start_with_shutdown();
    let proxy_port = free_port();

    let config = load_example_config(
        "token-rate-limit.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend.port())]),
    );
    let proxy = start_proxy(&config);

    let raw = http_send(proxy.addr(), &json_post("/v1/chat/completions", "{}"));
    assert_eq!(parse_status(&raw), 200, "example config smoke test should return 200");
    assert_eq!(parse_body(&raw), PLAIN_TEXT_BODY, "body should pass through unchanged");
}
