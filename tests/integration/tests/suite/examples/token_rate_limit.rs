// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Integration tests for the `token_rate_limit` filter's example config.
//!
//! Covers the agreed M1/M2/M6 core of the token rate limiting proposal
//! (`00121_token-rate-limiting.md` in `praxis-proxy/enhancements`, epic
//! `ai#121`): reservation-based admission, 429 rejection with
//! token-denominated headers, and reconciliation against actual
//! provider-reported usage (`token_count`'s `token.total`) once the
//! response completes.
//!
//! `mixed_algorithm_rules_valkey_backend_isolates_budgets_across_gateway_replicas`
//! additionally covers `ai#789`/`praxis#551`'s per-rule algorithm choice
//! (`rules:`/`match:`/`algorithm:`) end-to-end through the real
//! [`praxis_filter::HttpFilter`] pipeline, gated on a live Valkey/Redis
//! instance the same way `filters/src/token_rate_limit/tests.rs`'s
//! unit-level Valkey tests are.

use std::collections::HashMap;

use praxis_test_utils::{
    Backend, example_config_path, free_port, http_send, json_post, load_example_config, parse_body, parse_header,
    parse_status, patch_yaml, start_proxy,
};

/// Build a `POST` request carrying extra headers beyond the standard
/// JSON content-type/length, for `match`-based rule dispatch scenarios
/// that need to tag requests with an app identity.
fn json_post_with_headers(path: &str, body: &str, headers: &[(&str, &str)]) -> String {
    let mut extra = String::new();
    for (name, value) in headers {
        extra.push_str(&format!("{name}: {value}\r\n"));
    }
    format!(
        "POST {path} HTTP/1.1\r\n\
         Host: localhost\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         {extra}\
         Connection: close\r\n\r\n\
         {body}",
        body.len()
    )
}

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
/// example file, substituting `capacity`/`reserved_tokens` with the given
/// values so tests can exercise small, deterministic budgets.
fn token_rate_limit_config(
    proxy_port: u16,
    backend_port: u16,
    capacity: u64,
    reserved_tokens: u64,
) -> praxis_core::config::Config {
    let path = example_config_path("token-rate-limit.yaml");
    let yaml = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let yaml = yaml
        .replace("capacity: 100000", &format!("capacity: {capacity}"))
        .replace("reserved_tokens: 500", &format!("reserved_tokens: {reserved_tokens}"));
    let patched = patch_yaml(&yaml, proxy_port, &HashMap::from([("127.0.0.1:3000", backend_port)]));
    praxis_core::config::Config::from_yaml(&patched).expect("config should parse")
}

// -----------------------------------------------------------------------------
// Admission and headers
// -----------------------------------------------------------------------------

#[test]
fn admits_request_within_budget() {
    let backend = Backend::fixed(PLAIN_TEXT_BODY)
        .header("content-type", "text/plain")
        .start_with_shutdown();
    let proxy_port = free_port();
    let config = token_rate_limit_config(proxy_port, backend.port(), 100_000, 500);
    let proxy = start_proxy(&config);

    let raw = http_send(proxy.addr(), &json_post("/v1/chat/completions", "{}"));
    assert_eq!(parse_status(&raw), 200, "request within budget should be admitted");
    assert_eq!(parse_body(&raw), PLAIN_TEXT_BODY, "body should pass through unchanged");
}

#[test]
fn rejects_with_429_and_retry_after_when_estimate_budget_exhausted() {
    let backend = Backend::fixed(PLAIN_TEXT_BODY)
        .header("content-type", "text/plain")
        .start_with_shutdown();
    let proxy_port = free_port();
    // capacity=50, estimate=40, no reconciliation (this backend returns no
    // usage info): first request -40 -> 10 left (200), second request
    // needs 40 more and must be rejected.
    let config = token_rate_limit_config(proxy_port, backend.port(), 50, 40);
    let proxy = start_proxy(&config);

    let first = http_send(proxy.addr(), &json_post("/v1/chat/completions", "{}"));
    assert_eq!(parse_status(&first), 200, "first request should be admitted");

    let second = http_send(proxy.addr(), &json_post("/v1/chat/completions", "{}"));
    assert_eq!(
        parse_status(&second),
        429,
        "second request should be rejected, only 10 of 50 tokens remain after the first request spent 40"
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
    // capacity=50, estimate=40. Every admitted request reserves 40 and
    // then, since this backend's actual usage (10) is far below the
    // estimate, gets 30 released back on reconciliation — so the
    // window settles into a steady drain of only 10 net tokens per
    // request instead of monotonically losing the full 40. Starting
    // from an empty window (0/50 used):
    //   first:   reserve 40 (50->10), reconcile +30 -> 40   [200]
    //   second:  reserve 40 (40->0),  reconcile +30 -> 30   [200]
    //   third:   needs 40, only 30 remain                  [429]
    // A naive (non-reconciling) reservation scheme would already have
    // rejected "second" (it would see only 10 remaining after "first").
    let config = token_rate_limit_config(proxy_port, backend.port(), 50, 40);
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

/// Smoke-tests the real `token-rate-limit-mixed-algorithms.yaml` example
/// file itself (ai#789/praxis#551), distinct from
/// `mixed_algorithm_rules_valkey_backend_isolates_budgets_across_gateway_replicas`
/// below, which builds its own hand-rolled YAML with a Valkey backend
/// clause to prove cross-replica isolation. This confirms the example
/// file that ships in the repo actually wires up both the `team-alpha`
/// (sliding_window) and `team-beta` (token_bucket) rules correctly.
#[test]
fn example_config_token_rate_limit_mixed_algorithms() {
    let backend = Backend::fixed(PLAIN_TEXT_BODY)
        .header("content-type", "text/plain")
        .start_with_shutdown();
    let proxy_port = free_port();

    let config = load_example_config(
        "token-rate-limit-mixed-algorithms.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend.port())]),
    );
    let proxy = start_proxy(&config);

    let alpha = http_send(
        proxy.addr(),
        &json_post_with_headers("/v1/chat/completions", "{}", &[("x-app-id", "alpha")]),
    );
    assert_eq!(
        parse_status(&alpha),
        200,
        "team-alpha's sliding_window rule should admit a request within its budget"
    );

    let beta = http_send(
        proxy.addr(),
        &json_post_with_headers("/v1/chat/completions", "{}", &[("x-app-id", "beta")]),
    );
    assert_eq!(
        parse_status(&beta),
        200,
        "team-beta's token_bucket rule should admit a request within its budget"
    );
}

// -----------------------------------------------------------------------------
// Mixed algorithms, per rule (ai#789/praxis#551) -- Valkey-backed, driven
// through the real gateway pipeline across two independent proxy
// instances (simulated replicas), gated on a live Valkey/Redis instance
// via TOKEN_RATE_LIMIT_VALKEY_URL (see filters/src/token_rate_limit/
// tests.rs for local setup instructions).
// -----------------------------------------------------------------------------

/// Two `token_rate_limit` rules sharing one Valkey namespace: `team-alpha`
/// (matched on `x-app-id: alpha`) enforces a sliding-window budget,
/// `team-beta` (matched on `x-app-id: beta`) enforces a token-bucket
/// budget. A request matching neither rule (no `x-app-id` header) is a
/// catch-all-free config here, so it passes through unrated -- this
/// config is deliberately about proving per-algorithm isolation, not
/// fallback-bucket behavior (already covered above).
fn mixed_algorithm_rules_config(proxy_port: u16, backend_port: u16, valkey_url: &str, namespace: &str) -> String {
    let yaml = format!(
        "listeners:\n\
         \x20 - name: default\n\
         \x20   address: \"0.0.0.0:8080\"\n\
         \x20   filter_chains:\n\
         \x20     - main\n\
         filter_chains:\n\
         \x20 - name: main\n\
         \x20   filters:\n\
         \x20     - filter: router\n\
         \x20       routes:\n\
         \x20         - path_prefix: \"/\"\n\
         \x20           cluster: backend\n\
         \x20     - filter: token_rate_limit\n\
         \x20       backend:\n\
         \x20         kind: valkey\n\
         \x20         url: {valkey_url}\n\
         \x20         namespace: {namespace}\n\
         \x20       rules:\n\
         \x20         - name: team-alpha\n\
         \x20           match:\n\
         \x20             headers:\n\
         \x20               x-app-id: alpha\n\
         \x20           algorithm: sliding_window\n\
         \x20           window: 1h\n\
         \x20           capacity: 100\n\
         \x20           reserved_tokens: 100\n\
         \x20         - name: team-beta\n\
         \x20           match:\n\
         \x20             headers:\n\
         \x20               x-app-id: beta\n\
         \x20           algorithm: token_bucket\n\
         \x20           capacity: 100\n\
         \x20           refill_rate: 0.001\n\
         \x20           reserved_tokens: 100\n\
         \x20     - filter: access_log\n\
         \x20     - filter: load_balancer\n\
         \x20       clusters:\n\
         \x20         - name: backend\n\
         \x20           endpoints:\n\
         \x20             - \"127.0.0.1:3000\"\n"
    );
    patch_yaml(&yaml, proxy_port, &HashMap::from([("127.0.0.1:3000", backend_port)]))
}

/// Proves both algorithms get the distributed-state property that's the
/// whole point of the Valkey backend -- not just in isolation (already
/// covered at the unit-test tier in `filters/src/token_rate_limit/
/// tests.rs`), but through the real gateway pipeline, with two
/// independent proxy processes standing in for two gateway
/// replicas/instances sharing one Valkey namespace behind a load
/// balancer.
#[test]
fn mixed_algorithm_rules_valkey_backend_isolates_budgets_across_gateway_replicas() {
    let Ok(valkey_url) = std::env::var("TOKEN_RATE_LIMIT_VALKEY_URL") else {
        eprintln!("skipping: TOKEN_RATE_LIMIT_VALKEY_URL not set");
        return;
    };
    let namespace = format!("praxis-it-mixed-algorithms-{}", std::process::id());

    let backend_one = Backend::fixed(PLAIN_TEXT_BODY)
        .header("content-type", "text/plain")
        .start_with_shutdown();
    let backend_two = Backend::fixed(PLAIN_TEXT_BODY)
        .header("content-type", "text/plain")
        .start_with_shutdown();

    // Two independent proxy processes, each built from the same rules
    // config and pointed at the same Valkey namespace -- exactly as two
    // gateway replicas behind a load balancer would be.
    let proxy_one_port = free_port();
    let config_one = praxis_core::config::Config::from_yaml(&mixed_algorithm_rules_config(
        proxy_one_port,
        backend_one.port(),
        &valkey_url,
        &namespace,
    ))
    .expect("config should parse");
    let proxy_one = start_proxy(&config_one);

    let proxy_two_port = free_port();
    let config_two = praxis_core::config::Config::from_yaml(&mixed_algorithm_rules_config(
        proxy_two_port,
        backend_two.port(),
        &valkey_url,
        &namespace,
    ))
    .expect("config should parse");
    let proxy_two = start_proxy(&config_two);

    // team-alpha's sliding-window rule: admitted on replica one,
    // capacity-exhausted (100/100) on replica two via the shared Valkey
    // budget.
    let alpha_first = http_send(
        proxy_one.addr(),
        &json_post_with_headers("/v1/chat/completions", "{}", &[("x-app-id", "alpha")]),
    );
    assert_eq!(
        parse_status(&alpha_first),
        200,
        "team-alpha's first request should be admitted on replica one"
    );
    let alpha_second = http_send(
        proxy_two.addr(),
        &json_post_with_headers("/v1/chat/completions", "{}", &[("x-app-id", "alpha")]),
    );
    assert_eq!(
        parse_status(&alpha_second),
        429,
        "team-alpha's exhausted sliding-window budget must be visible on replica two via shared Valkey state"
    );

    // team-beta's token-bucket rule: admitted on replica one,
    // capacity-exhausted on replica two -- proving the *second*
    // algorithm gets the same cross-replica property, and that it
    // doesn't share state with (or get blocked by) team-alpha's budget.
    let beta_first = http_send(
        proxy_one.addr(),
        &json_post_with_headers("/v1/chat/completions", "{}", &[("x-app-id", "beta")]),
    );
    assert_eq!(
        parse_status(&beta_first),
        200,
        "team-beta's first request should be admitted on replica one, unaffected by team-alpha's exhaustion"
    );
    let beta_second = http_send(
        proxy_two.addr(),
        &json_post_with_headers("/v1/chat/completions", "{}", &[("x-app-id", "beta")]),
    );
    assert_eq!(
        parse_status(&beta_second),
        429,
        "team-beta's exhausted token-bucket budget must be visible on replica two via shared Valkey state"
    );
}
