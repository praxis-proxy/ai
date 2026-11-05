// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Tests for the external metering example configuration with model-policy
//! enforcement enabled (issue #1719).

use std::{collections::HashMap, time::Duration};

use praxis_test_utils::{
    Backend, StatefulCapturingBackend, free_port, http_send, json_post_with_header, parse_body, parse_status,
};

/// Verdict admitted by both gates.
const ALLOWED: &str = r#"{"hasAccess": true, "modelAllowed": true}"#;

/// Budget available, model explicitly denied — the headline case.
const MODEL_DENIED: &str = r#"{"hasAccess": true, "modelAllowed": false}"#;

// -----------------------------------------------------------------------------
// Config Parsing
// -----------------------------------------------------------------------------

#[test]
fn external_metering_model_policy_config_parses() {
    let metering_port = free_port();
    let config = super::load_example_config(
        "external-metering-model-policy.yaml",
        29810,
        HashMap::from([("127.0.0.1:3000", 29811_u16), ("127.0.0.1:9090", metering_port)]),
    );

    assert_eq!(config.listeners.len(), 1, "should have 1 listener");
}

// -----------------------------------------------------------------------------
// Explicit Model Denial — 403 Before Any Provider Request
// -----------------------------------------------------------------------------

#[test]
fn external_metering_model_policy_denies_model_with_403_and_zero_provider_requests() {
    // The provider mock records every request; the assertion that it stayed
    // empty is the required "zero provider requests after rejection" boundary.
    let backend = StatefulCapturingBackend::new(vec![(200, r#"{"ok":true}"#.to_owned())]).start_with_shutdown();
    let metering =
        StatefulCapturingBackend::new(vec![(200, MODEL_DENIED.to_owned()), (204, String::new())]).start_with_shutdown();
    let proxy_port = free_port();

    let config = super::load_example_config(
        "external-metering-model-policy.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend.port()), ("127.0.0.1:9090", metering.port())]),
    );

    let proxy = praxis_test_utils::start_proxy(&config);
    let raw = http_send(
        proxy.addr(),
        &json_post_with_header(
            "/v1/chat/completions",
            r#"{"model":"gpt-4o","messages":[]}"#,
            "x-tenant-username: alice",
        ),
    );

    assert_eq!(
        parse_status(&raw),
        403,
        "an explicit model denial must reject even though budget remains and billing fail_open is true"
    );
    let body = parse_body(&raw);
    assert!(
        body.contains("model not allowed for this user"),
        "rejection body should name the model denial: {body}"
    );
    assert!(
        backend.requests().is_empty(),
        "no provider request may be dispatched after a model-policy rejection"
    );
    assert!(
        metering
            .requests()
            .iter()
            .any(|r| r.method == "GET" && r.uri.contains("model=gpt-4o")),
        "the preflight should still consult the metering service with the public model"
    );
}

#[test]
fn external_metering_model_policy_budget_denial_is_still_429() {
    let backend = StatefulCapturingBackend::new(vec![(200, r#"{"ok":true}"#.to_owned())]).start_with_shutdown();
    let metering =
        StatefulCapturingBackend::new(vec![(200, r#"{"hasAccess": false, "modelAllowed": true}"#.to_owned())])
            .start_with_shutdown();
    let proxy_port = free_port();

    let config = super::load_example_config(
        "external-metering-model-policy.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend.port()), ("127.0.0.1:9090", metering.port())]),
    );

    let proxy = praxis_test_utils::start_proxy(&config);
    let raw = http_send(
        proxy.addr(),
        &json_post_with_header(
            "/v1/chat/completions",
            r#"{"model":"gpt-4o","messages":[]}"#,
            "x-tenant-username: alice",
        ),
    );

    assert_eq!(parse_status(&raw), 429, "budget exhaustion keeps its 429 contract");
    assert!(
        backend.requests().is_empty(),
        "no provider request may be dispatched after a budget rejection"
    );
}

// -----------------------------------------------------------------------------
// Allow Path — Trusted Tenant, Public Alias, Rewritten Provider Model
// -----------------------------------------------------------------------------

#[test]
fn external_metering_model_policy_allow_forwards_rewritten_model() {
    let backend = StatefulCapturingBackend::new(vec![(200, r#"{"ok":true}"#.to_owned())]).start_with_shutdown();
    let metering =
        StatefulCapturingBackend::new(vec![(200, ALLOWED.to_owned()), (204, String::new())]).start_with_shutdown();
    let proxy_port = free_port();

    // Reuse the example config's metering block but route through
    // model_to_provider, so the provider receives a rewritten target while
    // Metering must see the public alias.
    let yaml = format!(
        r#"
listeners:
  - name: default
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: router
        routes:
          - path_prefix: "/"
            cluster: backend
      - filter: external_metering
        metering_url: "http://127.0.0.1:{metering_port}"
        allow_private_endpoint: true
        feature_key: "inference-tokens"
        source: "ai-gateway"
        provider: "vertex"
        fail_open: true
        identity_header_prefix: "x-tenant-"
        model_policy_check: true
        model_policy_fail_open: false
      - filter: model_to_provider
        models:
          - model: claude-sonnet-4-5
            upstream: backend
            target_model: claude-sonnet-4-5@20250929
            paths: [/v1/messages]
      - filter: load_balancer
        clusters:
          - name: backend
            endpoints:
              - "127.0.0.1:{backend_port}"
insecure_options:
  allow_private_endpoints: true
"#,
        metering_port = metering.port(),
        backend_port = backend.port(),
    );
    let config = praxis_core::config::Config::from_yaml(&yaml).unwrap();

    let proxy = praxis_test_utils::start_proxy(&config);
    let raw = http_send(
        proxy.addr(),
        &json_post_with_header(
            "/v1/messages",
            r#"{"model":"claude-sonnet-4-5","messages":[]}"#,
            "x-tenant-username: alice",
        ),
    );
    assert_eq!(
        parse_status(&raw),
        200,
        "a request admitted by both gates must be proxied"
    );

    let forwarded = backend.requests();
    assert_eq!(forwarded.len(), 1, "the metered request must reach its provider once");
    let body: serde_json::Value = serde_json::from_str(&forwarded[0].body).expect("provider body must be JSON");
    assert_eq!(
        body["model"], "claude-sonnet-4-5@20250929",
        "the provider must receive the rewritten target"
    );

    let balance = metering
        .requests()
        .into_iter()
        .find(|r| r.method == "GET")
        .expect("the preflight should reach the metering service");
    assert!(
        balance.uri.contains("/customers/alice/"),
        "the preflight must name the trusted tenant: {}",
        balance.uri
    );
    assert!(
        balance.uri.ends_with("?model=claude-sonnet-4-5"),
        "the preflight must name the public alias, not the provider target: {}",
        balance.uri
    );
}

// -----------------------------------------------------------------------------
// Fail-Closed Availability — Independent of Billing fail_open
// -----------------------------------------------------------------------------

#[test]
fn external_metering_model_policy_fails_closed_when_metering_down_despite_billing_fail_open() {
    let backend = StatefulCapturingBackend::new(vec![(200, r#"{"ok":true}"#.to_owned())]).start_with_shutdown();
    // Nothing listens here: the callout fails at the transport level while
    // the example config keeps billing fail_open: true.
    let metering_port = free_port();
    let proxy_port = free_port();

    let config = super::load_example_config(
        "external-metering-model-policy.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend.port()), ("127.0.0.1:9090", metering_port)]),
    );

    let proxy = praxis_test_utils::start_proxy(&config);
    let raw = http_send(
        proxy.addr(),
        &json_post_with_header(
            "/v1/chat/completions",
            r#"{"model":"gpt-4o","messages":[]}"#,
            "x-tenant-username: alice",
        ),
    );

    assert_eq!(
        parse_status(&raw),
        503,
        "the model gate must fail closed (503) even though the billing gate fails open"
    );
    let body = parse_body(&raw);
    assert!(
        body.contains("metering system unavailable"),
        "rejection body should mention metering unavailability: {body}"
    );
    assert!(
        backend.requests().is_empty(),
        "no provider request may be dispatched after a fail-closed rejection"
    );
}

#[test]
fn external_metering_model_policy_fails_closed_on_legacy_payload_without_model_verdict() {
    let backend = StatefulCapturingBackend::new(vec![(200, r#"{"ok":true}"#.to_owned())]).start_with_shutdown();
    // A legacy metering service answering only `hasAccess` provides no model
    // verdict — never an implicit allow.
    let metering =
        StatefulCapturingBackend::new(vec![(200, r#"{"hasAccess": true}"#.to_owned())]).start_with_shutdown();
    let proxy_port = free_port();

    let config = super::load_example_config(
        "external-metering-model-policy.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend.port()), ("127.0.0.1:9090", metering.port())]),
    );

    let proxy = praxis_test_utils::start_proxy(&config);
    let raw = http_send(
        proxy.addr(),
        &json_post_with_header(
            "/v1/chat/completions",
            r#"{"model":"gpt-4o","messages":[]}"#,
            "x-tenant-username: alice",
        ),
    );

    assert_eq!(
        parse_status(&raw),
        503,
        "a missing model verdict must fail closed, not implicitly allow"
    );
    assert!(
        backend.requests().is_empty(),
        "no provider request may be dispatched when the model verdict is unavailable"
    );
}

#[test]
fn external_metering_model_policy_fails_closed_on_non_2xx_despite_billing_fail_open() {
    let backend = StatefulCapturingBackend::new(vec![(200, r#"{"ok":true}"#.to_owned())]).start_with_shutdown();
    // A non-2xx entitlement response is a misbehaving service, not a verdict:
    // the model gate must apply its own availability policy.
    let metering = StatefulCapturingBackend::new(vec![(500, r#"{"error":"boom"}"#.to_owned())]).start_with_shutdown();
    let proxy_port = free_port();

    let config = super::load_example_config(
        "external-metering-model-policy.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend.port()), ("127.0.0.1:9090", metering.port())]),
    );

    let proxy = praxis_test_utils::start_proxy(&config);
    let raw = http_send(
        proxy.addr(),
        &json_post_with_header(
            "/v1/chat/completions",
            r#"{"model":"gpt-4o","messages":[]}"#,
            "x-tenant-username: alice",
        ),
    );

    assert_eq!(
        parse_status(&raw),
        503,
        "a non-2xx metering response must fail the model gate closed despite billing fail_open"
    );
    let body = parse_body(&raw);
    assert!(
        body.contains("metering system unavailable"),
        "rejection body should mention metering unavailability: {body}"
    );
    assert!(
        backend.requests().is_empty(),
        "no provider request may be dispatched after a non-2xx fail-closed rejection"
    );
    assert!(
        metering.requests().iter().any(|r| r.method == "GET"),
        "the preflight must still have consulted the metering service"
    );
}

#[test]
fn external_metering_model_policy_fails_closed_on_malformed_payload_despite_billing_fail_open() {
    let backend = StatefulCapturingBackend::new(vec![(200, r#"{"ok":true}"#.to_owned())]).start_with_shutdown();
    // A 200 with an unparseable body is an unusable payload, not a verdict.
    let metering = StatefulCapturingBackend::new(vec![(200, "not json".to_owned())]).start_with_shutdown();
    let proxy_port = free_port();

    let config = super::load_example_config(
        "external-metering-model-policy.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend.port()), ("127.0.0.1:9090", metering.port())]),
    );

    let proxy = praxis_test_utils::start_proxy(&config);
    let raw = http_send(
        proxy.addr(),
        &json_post_with_header(
            "/v1/chat/completions",
            r#"{"model":"gpt-4o","messages":[]}"#,
            "x-tenant-username: alice",
        ),
    );

    assert_eq!(
        parse_status(&raw),
        503,
        "a malformed entitlement payload must fail the model gate closed despite billing fail_open"
    );
    assert!(
        backend.requests().is_empty(),
        "no provider request may be dispatched when the payload is unusable"
    );
    assert!(
        metering.requests().iter().any(|r| r.method == "GET"),
        "the preflight must still have consulted the metering service before failing closed"
    );
}

#[test]
fn external_metering_model_policy_fails_closed_on_oversized_payload_despite_billing_fail_open() {
    let backend = StatefulCapturingBackend::new(vec![(200, r#"{"ok":true}"#.to_owned())]).start_with_shutdown();
    // An entitlement body beyond the callout's response bound (64 KiB) is cut
    // off mid-payload, so even verdicts spelled out at the start cannot be
    // trusted: the truncated JSON is unusable and both gates lose their
    // verdicts. The model gate is fail-closed, so this must reject.
    let oversized = format!(
        r#"{{"hasAccess": true, "modelAllowed": true, "padding": "{}"}}"#,
        "x".repeat(70_000)
    );
    let metering = StatefulCapturingBackend::new(vec![(200, oversized)]).start_with_shutdown();
    let proxy_port = free_port();

    let config = super::load_example_config(
        "external-metering-model-policy.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend.port()), ("127.0.0.1:9090", metering.port())]),
    );

    let proxy = praxis_test_utils::start_proxy(&config);
    let raw = http_send(
        proxy.addr(),
        &json_post_with_header(
            "/v1/chat/completions",
            r#"{"model":"gpt-4o","messages":[]}"#,
            "x-tenant-username: alice",
        ),
    );

    assert_eq!(
        parse_status(&raw),
        503,
        "an oversized entitlement payload must fail the model gate closed despite billing fail_open"
    );
    assert!(
        backend.requests().is_empty(),
        "no provider request may be dispatched when the payload is cut off at the response bound"
    );
    assert!(
        metering.requests().iter().any(|r| r.method == "GET"),
        "the preflight must still have consulted the metering service before failing closed"
    );
}

#[test]
fn external_metering_model_policy_fails_closed_on_timeout_despite_billing_fail_open() {
    let backend = StatefulCapturingBackend::new(vec![(200, r#"{"ok":true}"#.to_owned())]).start_with_shutdown();
    // The metering service accepts the callout but never finishes its
    // response within the filter's timeout. A short timeout keeps the test
    // deterministic; the stall far exceeds it.
    let metering = Backend::chunked(vec![r#"{"hasAccess": true, "modelAllowed": true}"#.to_owned()])
        .chunk_delay(Duration::from_secs(10))
        .start_with_shutdown();
    let proxy_port = free_port();

    let yaml = format!(
        r#"
listeners:
  - name: default
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: router
        routes:
          - path_prefix: "/"
            cluster: backend
      - filter: external_metering
        metering_url: "http://127.0.0.1:{metering_port}"
        allow_private_endpoint: true
        feature_key: "inference-tokens"
        source: "ai-gateway"
        provider: "openai"
        timeout_seconds: 1
        fail_open: true
        identity_header_prefix: "x-tenant-"
        model_policy_check: true
        model_policy_fail_open: false
      - filter: load_balancer
        clusters:
          - name: backend
            endpoints:
              - "127.0.0.1:{backend_port}"
insecure_options:
  allow_private_endpoints: true
"#,
        metering_port = metering.port(),
        backend_port = backend.port(),
    );
    let config = praxis_core::config::Config::from_yaml(&yaml).unwrap();

    let proxy = praxis_test_utils::start_proxy(&config);
    let raw = http_send(
        proxy.addr(),
        &json_post_with_header(
            "/v1/chat/completions",
            r#"{"model":"gpt-4o","messages":[]}"#,
            "x-tenant-username: alice",
        ),
    );

    assert_eq!(
        parse_status(&raw),
        503,
        "a metering timeout must fail the model gate closed despite billing fail_open"
    );
    let body = parse_body(&raw);
    assert!(
        body.contains("metering system unavailable"),
        "rejection body should mention metering unavailability: {body}"
    );
    assert!(
        backend.requests().is_empty(),
        "no provider request may be dispatched while the verdict is still pending"
    );
}

// -----------------------------------------------------------------------------
// Model Source Conflicts — 403 Before Dispatch (review regression)
// -----------------------------------------------------------------------------

#[test]
fn external_metering_model_policy_rejects_conflicting_identity_model_before_dispatch() {
    // Regression: a trusted identity claim naming model A while the request
    // actually carries (and `model_to_provider` records, rewrites, and
    // forwards) model B must not authorize A — the preflight and the
    // provider-bound model cannot diverge.
    let backend = StatefulCapturingBackend::new(vec![(200, r#"{"ok":true}"#.to_owned())]).start_with_shutdown();
    let metering =
        StatefulCapturingBackend::new(vec![(200, ALLOWED.to_owned()), (204, String::new())]).start_with_shutdown();
    let proxy_port = free_port();

    let yaml = format!(
        r#"
listeners:
  - name: default
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: router
        routes:
          - path_prefix: "/"
            cluster: backend
      - filter: external_metering
        metering_url: "http://127.0.0.1:{metering_port}"
        allow_private_endpoint: true
        feature_key: "inference-tokens"
        source: "ai-gateway"
        provider: "vertex"
        fail_open: true
        identity_header_prefix: "x-tenant-"
        model_policy_check: true
        model_policy_fail_open: false
      - filter: model_to_provider
        models:
          - model: claude-sonnet-4-5
            upstream: backend
            target_model: claude-sonnet-4-5@20250929
            paths: [/v1/messages]
      - filter: load_balancer
        clusters:
          - name: backend
            endpoints:
              - "127.0.0.1:{backend_port}"
insecure_options:
  allow_private_endpoints: true
"#,
        metering_port = metering.port(),
        backend_port = backend.port(),
    );
    let config = praxis_core::config::Config::from_yaml(&yaml).unwrap();

    let proxy = praxis_test_utils::start_proxy(&config);
    // The identity header claims a different model than the request body;
    // `model_to_provider` records the body's model as the client-facing one.
    let raw = http_send(
        proxy.addr(),
        &json_post_with_header(
            "/v1/messages",
            r#"{"model":"claude-sonnet-4-5","messages":[]}"#,
            "x-tenant-username: alice\nx-tenant-model: gpt-4o",
        ),
    );

    assert_eq!(
        parse_status(&raw),
        403,
        "an identity model that disagrees with the requested model must reject before dispatch"
    );
    let body = parse_body(&raw);
    assert!(
        body.contains("conflicting model inputs"),
        "rejection body should name the model conflict: {body}"
    );
    assert!(
        metering.requests().is_empty(),
        "a model conflict must reject before any metering callout"
    );
    assert!(
        backend.requests().is_empty(),
        "no provider request may be dispatched when the identity and requested models diverge"
    );
}

// -----------------------------------------------------------------------------
// Missing Identity / Model — 403 Before Dispatch, No Metering Callout
// -----------------------------------------------------------------------------

#[test]
fn external_metering_model_policy_rejects_missing_identity_without_callout() {
    let backend = StatefulCapturingBackend::new(vec![(200, r#"{"ok":true}"#.to_owned())]).start_with_shutdown();
    let metering =
        StatefulCapturingBackend::new(vec![(200, ALLOWED.to_owned()), (204, String::new())]).start_with_shutdown();
    let proxy_port = free_port();

    let config = super::load_example_config(
        "external-metering-model-policy.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend.port()), ("127.0.0.1:9090", metering.port())]),
    );

    let proxy = praxis_test_utils::start_proxy(&config);
    // No x-tenant-username: the legacy skip-metering path must not bypass
    // authorization, and no default_username is configured in the example.
    let raw = http_send(
        proxy.addr(),
        &json_post_with_header(
            "/v1/chat/completions",
            r#"{"model":"gpt-4o","messages":[]}"#,
            "x-ignore: 1",
        ),
    );

    assert_eq!(
        parse_status(&raw),
        403,
        "model policy requires a trusted tenant identity before dispatch"
    );
    let body = parse_body(&raw);
    assert!(
        body.contains("tenant identity required"),
        "rejection body should name the missing identity: {body}"
    );
    assert!(
        metering.requests().is_empty(),
        "no metering callout may be made without a tenant identity"
    );
    assert!(
        backend.requests().is_empty(),
        "no provider request may be dispatched without a tenant identity"
    );
}

#[test]
fn external_metering_model_policy_rejects_unresolvable_model() {
    let backend = StatefulCapturingBackend::new(vec![(200, r#"{"ok":true}"#.to_owned())]).start_with_shutdown();
    let metering =
        StatefulCapturingBackend::new(vec![(200, ALLOWED.to_owned()), (204, String::new())]).start_with_shutdown();
    let proxy_port = free_port();

    let config = super::load_example_config(
        "external-metering-model-policy.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend.port()), ("127.0.0.1:9090", metering.port())]),
    );

    let proxy = praxis_test_utils::start_proxy(&config);
    // Identity present but no model anywhere: not in a header, and the body
    // carries no top-level "model" field.
    let raw = http_send(
        proxy.addr(),
        &json_post_with_header("/v1/chat/completions", r#"{"messages":[]}"#, "x-tenant-username: alice"),
    );

    assert_eq!(
        parse_status(&raw),
        403,
        "model policy requires a resolvable public model before dispatch"
    );
    let body = parse_body(&raw);
    assert!(
        body.contains("request model required"),
        "rejection body should name the missing model: {body}"
    );
    assert!(
        metering.requests().is_empty(),
        "no metering callout may be made without a resolvable model"
    );
    assert!(
        backend.requests().is_empty(),
        "no provider request may be dispatched without a resolvable model"
    );
}
