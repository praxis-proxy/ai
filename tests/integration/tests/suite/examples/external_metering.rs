// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Tests for the external metering example configuration.

use std::{
    collections::HashMap,
    time::{SystemTime, UNIX_EPOCH},
};

use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use openssl::{hash::MessageDigest, pkey::PKey, sign::Signer};
use praxis_test_utils::{
    Backend, RoutedBackend, StatefulCapturingBackend, StatefulCapturingGuard, free_port, http_send,
    json_post_with_header, parse_body, parse_status, start_header_echo_backend,
};
use serde_json::json;
use tempfile::TempDir;

// -----------------------------------------------------------------------------
// Config Parsing
// -----------------------------------------------------------------------------

#[test]
fn external_metering_config_parses() {
    let metering_port = free_port();
    let config = super::load_example_config(
        "external-metering.yaml",
        29800,
        HashMap::from([("127.0.0.1:3000", 29801_u16), ("127.0.0.1:9090", metering_port)]),
    );

    assert_eq!(config.listeners.len(), 1, "should have 1 listener");
}

// -----------------------------------------------------------------------------
// Balance Check — Access Granted
// -----------------------------------------------------------------------------

#[test]
fn external_metering_allows_request_when_balance_available() {
    let backend_guard = start_header_echo_backend();
    let backend_port = backend_guard.port();
    let proxy_port = free_port();

    let balance_body = r#"{"hasAccess": true, "balance": 9000.0}"#;
    let metering_port = RoutedBackend::new()
        .route("/api/v1/customers", 200, balance_body)
        .route("/api/v1/events", 204, "")
        .start();

    let config = super::load_example_config(
        "external-metering.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend_port), ("127.0.0.1:9090", metering_port)]),
    );

    let proxy = praxis_test_utils::start_proxy(&config);
    let raw = http_send(
        proxy.addr(),
        "POST /v1/chat/completions HTTP/1.1\r\n\
         Host: localhost\r\n\
         x-tenant-username: alice\r\n\
         x-tenant-group: engineering\r\n\
         Connection: close\r\n\r\n",
    );

    assert_eq!(parse_status(&raw), 200, "should proxy request when balance available");
}

// -----------------------------------------------------------------------------
// Fail-Closed — Metering Unavailable
// -----------------------------------------------------------------------------

#[test]
fn external_metering_rejects_when_fail_closed_and_metering_down() {
    let backend_guard = start_header_echo_backend();
    let backend_port = backend_guard.port();
    let proxy_port = free_port();

    // Point metering_url at a port with nothing listening
    let metering_port = free_port();

    let path = praxis_test_utils::example_config_path("external-metering.yaml");
    let yaml = std::fs::read_to_string(&path).unwrap();
    let patched = praxis_test_utils::patch_yaml(
        &yaml,
        proxy_port,
        &HashMap::from([("127.0.0.1:3000", backend_port), ("127.0.0.1:9090", metering_port)]),
    )
    .replace("fail_open: true", "fail_open: false");
    let config = praxis_core::config::Config::from_yaml(&patched).unwrap();

    let proxy = praxis_test_utils::start_proxy(&config);
    let raw = http_send(
        proxy.addr(),
        "POST /v1/chat/completions HTTP/1.1\r\n\
         Host: localhost\r\n\
         x-tenant-username: alice\r\n\
         Connection: close\r\n\r\n",
    );

    assert_eq!(
        parse_status(&raw),
        503,
        "should reject with 503 when metering is unavailable and fail_open=false"
    );
    let body = parse_body(&raw);
    assert!(
        body.contains("metering system unavailable"),
        "rejection body should mention metering unavailable: {body}"
    );
}

// -----------------------------------------------------------------------------
// Identity Headers Stripped
// -----------------------------------------------------------------------------

#[test]
fn external_metering_strips_tenant_and_auth_headers() {
    let backend_guard = start_header_echo_backend();
    let backend_port = backend_guard.port();
    let proxy_port = free_port();

    let balance_body = r#"{"hasAccess": true, "balance": 9000.0}"#;
    let metering_port = RoutedBackend::new()
        .route("/api/v1/customers", 200, balance_body)
        .route("/api/v1/events", 204, "")
        .start();

    let config = super::load_example_config(
        "external-metering.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend_port), ("127.0.0.1:9090", metering_port)]),
    );

    let proxy = praxis_test_utils::start_proxy(&config);
    let raw = http_send(
        proxy.addr(),
        "POST /v1/chat/completions HTTP/1.1\r\n\
         Host: localhost\r\n\
         x-tenant-username: alice\r\n\
         x-tenant-group: engineering\r\n\
         x-tenant-organization-id: org-9\r\n\
         x-tenant-cost-center: cc-7\r\n\
         Authorization: Bearer sk-client-secret\r\n\
         x-api-key: client-key\r\n\
         Connection: close\r\n\r\n",
    );

    assert_eq!(parse_status(&raw), 200, "should proxy successfully");
    let body = parse_body(&raw);
    assert!(
        !body.contains("x-tenant-username"),
        "tenant header should be stripped from upstream: {body}"
    );
    assert!(!body.contains("x-tenant-organization-id") && !body.contains("org-9"));
    assert!(!body.contains("x-tenant-cost-center") && !body.contains("cc-7"));
    assert!(
        !body.contains("sk-client-secret"),
        "authorization should be stripped from upstream: {body}"
    );
    assert!(
        !body.contains("client-key"),
        "x-api-key should be stripped from upstream: {body}"
    );
}

// -----------------------------------------------------------------------------
// Usage CloudEvent — Provider and Reasoning Tokens
// -----------------------------------------------------------------------------

#[test]
fn external_metering_event_carries_provider_and_reasoning_tokens() {
    // Upstream returns an OpenAI-shaped JSON body with reasoning tokens, so
    // token_count writes token.reasoning for external_metering to read back.
    let usage_body = r#"{"id":"resp-1","object":"chat.completion","usage":{"prompt_tokens":10,"completion_tokens":90,"total_tokens":100,"completion_tokens_details":{"reasoning_tokens":64}}}"#;
    let backend_guard = Backend::status(200, usage_body)
        .header("Content-Type", "application/json")
        .start_with_shutdown();
    let backend_port = backend_guard.port();
    let proxy_port = free_port();

    // Metering mock: the balance check (GET) is answered first, then the
    // fire-and-forget usage report (POST /api/v1/events), whose body we
    // capture to inspect the emitted CloudEvent.
    let metering = StatefulCapturingBackend::new(vec![
        (200, r#"{"hasAccess": true, "balance": 9000.0}"#.to_owned()),
        (204, String::new()),
    ])
    .start_with_shutdown();
    let metering_port = metering.port();

    let config = super::load_example_config(
        "external-metering.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend_port), ("127.0.0.1:9090", metering_port)]),
    );

    let proxy = praxis_test_utils::start_proxy(&config);
    let raw = http_send(
        proxy.addr(),
        "POST /v1/chat/completions HTTP/1.1\r\n\
         Host: localhost\r\n\
         x-tenant-username: alice\r\n\
         x-tenant-group: engineering\r\n\
         x-tenant-organization-id: org-9\r\n\
         x-tenant-cost-center: cc-7\r\n\
         Content-Type: application/json\r\n\
         Content-Length: 2\r\n\
         Connection: close\r\n\r\n{}",
    );
    assert_eq!(
        parse_status(&raw),
        200,
        "request should be proxied when balance is available"
    );

    let event = wait_for_usage_event(&metering);
    let json: serde_json::Value = serde_json::from_str(&event).expect("usage event body should be valid JSON");

    assert_eq!(json["type"], "inference.tokens.used", "should be a usage event");
    assert_eq!(json["subject"], "alice");
    assert_eq!(
        json["data"]["provider"], "openai",
        "the configured provider must appear on the ext-proc data path where cluster_name() is None"
    );
    assert_eq!(
        json["data"]["reasoning_tokens"], 64,
        "reasoning tokens published by token_count must reach the usage CloudEvent"
    );
    assert_eq!(json["data"]["completion_tokens"], 90);
    assert_eq!(json["data"]["prompt_tokens"], 10);
    assert_eq!(json["data"]["user"], "alice");
    assert_eq!(json["data"]["organization_id"], "org-9");
    assert_eq!(json["data"]["cost_center"], "cc-7");
}

/// Poll the metering mock until the fire-and-forget usage report POST
/// arrives, returning its body. Panics if it does not arrive in time.
fn wait_for_usage_event(metering: &StatefulCapturingGuard) -> String {
    for _ in 0..50 {
        if let Some(req) = metering
            .requests()
            .into_iter()
            .find(|r| r.method == "POST" && r.uri.starts_with("/api/v1/events"))
        {
            return req.body;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    panic!("metering usage report POST was not received within 5s");
}

// -----------------------------------------------------------------------------
// No Identity — Metering Skipped
// -----------------------------------------------------------------------------

#[test]
fn external_metering_skips_when_no_identity() {
    let backend_guard = start_header_echo_backend();
    let backend_port = backend_guard.port();
    let proxy_port = free_port();

    // No metering mock needed — filter skips entirely without identity
    let metering_port = free_port();

    let config = super::load_example_config(
        "external-metering.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend_port), ("127.0.0.1:9090", metering_port)]),
    );

    let proxy = praxis_test_utils::start_proxy(&config);
    let raw = http_send(
        proxy.addr(),
        "GET / HTTP/1.1\r\n\
         Host: localhost\r\n\
         Connection: close\r\n\r\n",
    );

    assert_eq!(
        parse_status(&raw),
        200,
        "should proxy without metering when no identity headers"
    );
}

// -----------------------------------------------------------------------------
// Provider-Resolved Model
// -----------------------------------------------------------------------------

#[test]
fn external_metering_uses_client_model_resolved_by_model_to_provider() {
    let backend = StatefulCapturingBackend::new(vec![(200, r#"{"ok":true}"#.to_owned())]).start_with_shutdown();
    let metering = StatefulCapturingBackend::new(vec![
        (200, r#"{"hasAccess": true, "balance": 9000.0}"#.to_owned()),
        (204, String::new()),
    ])
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
      - filter: external_metering
        metering_url: "http://127.0.0.1:{metering_port}"
        allow_private_endpoint: true
        feature_key: "inference-tokens"
        source: "ai-gateway"
        provider: "vertex"
        fail_open: false
        identity_header_prefix: "x-tenant-"
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
    assert_eq!(parse_status(&raw), 200, "metered request should be proxied");
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
        .expect("the balance check should reach the metering service");
    assert!(
        balance.uri.ends_with("?model=claude-sonnet-4-5"),
        "the balance check should name the client model, not the provider target: {}",
        balance.uri
    );
    let event = wait_for_usage_event(&metering);
    let json: serde_json::Value = serde_json::from_str(&event).expect("usage event body should be valid JSON");
    assert_eq!(
        json["data"]["model"], "claude-sonnet-4-5",
        "usage should be reported against the client model"
    );
}

// -----------------------------------------------------------------------------
// Tier 0 — Verified AuthenticatedIdentity from the policy engine
// -----------------------------------------------------------------------------

/// Shared HS256 secret for the policy engine's `identity/jwt` plugin.
const JWT_SECRET: &str = "praxis-metering-test-secret-not-for-production-use";
/// Trusted issuer asserted in the minted test tokens.
const JWT_ISSUER: &str = "https://idp.test.local";
/// Audience the policy engine accepts.
const JWT_AUDIENCE: &str = "metering-test";

/// Mint an HS256 JWT signed with [`JWT_SECRET`] over the given claims.
fn mint_hs256_jwt(claims: &serde_json::Value) -> String {
    let encode =
        |value: &serde_json::Value| URL_SAFE_NO_PAD.encode(serde_json::to_vec(value).expect("serialize JWT part"));
    let signing_input = format!("{}.{}", encode(&json!({"alg": "HS256", "typ": "JWT"})), encode(claims));

    let key = PKey::hmac(JWT_SECRET.as_bytes()).expect("HMAC key");
    let mut signer = Signer::new(MessageDigest::sha256(), &key).expect("HS256 signer");
    signer.update(signing_input.as_bytes()).expect("feed signing input");
    let signature = signer.sign_to_vec().expect("HMAC-SHA256 signature");

    format!("{signing_input}.{}", URL_SAFE_NO_PAD.encode(signature))
}

/// Seconds since the Unix epoch.
fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock after epoch")
        .as_secs()
}

/// Write a minimal CPEX policy document whose `identity/jwt` plugin validates
/// the test tokens and, on success, publishes an `AuthenticatedIdentity` for
/// downstream filters. The returned `TempDir` must outlive the proxy.
fn write_policy_document() -> (TempDir, String) {
    let dir = TempDir::new().expect("create tempdir");
    let path = dir.path().join("policy.yaml");
    let yaml = format!(
        r#"plugins:
  - name: jwt-user
    kind: identity/jwt
    hooks: [identity.resolve]
    mode: sequential
    on_error: fail
    config:
      header: Authorization
      trusted_issuers:
        - issuer: "{JWT_ISSUER}"
          audiences: ["{JWT_AUDIENCE}"]
          algorithms: ["HS256"]
          decoding_key:
            kind: secret
            secret: "{JWT_SECRET}"
          leeway_seconds: 60
      claim_mapper: standard
global:
  authentication:
    - jwt-user
"#
    );
    std::fs::write(&path, yaml).expect("write policy document");
    (dir, path.to_str().expect("utf8 path").to_owned())
}

/// Build a proxy config that chains the `policy` engine (which publishes the
/// verified identity) ahead of `external_metering`, so metering resolves its
/// tenant from the `AuthenticatedIdentity` extension (tier 0) rather than any
/// request header.
fn policy_metering_config(policy_path: &str, metering_port: u16, backend_port: u16, extra_metering: &str) -> String {
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
      - filter: policy
        config_path: "{policy_path}"
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
        fail_open: false
{extra_metering}      - filter: load_balancer
        clusters:
          - name: backend
            endpoints:
              - "127.0.0.1:{backend_port}"
insecure_options:
  allow_private_endpoints: true
"#
    );
    yaml
}

#[test]
fn external_metering_meters_verified_subject_id() {
    // A request carrying only a verified JWT (no x-tenant-* headers) must be
    // metered under the token's subject, proving external_metering resolves
    // identity from the AuthenticatedIdentity the policy engine published.
    let backend = Backend::status(200, r#"{"ok":true}"#)
        .header("Content-Type", "application/json")
        .start_with_shutdown();
    let metering = StatefulCapturingBackend::new(vec![
        (200, r#"{"hasAccess": true, "balance": 9000.0}"#.to_owned()),
        (204, String::new()),
    ])
    .start_with_shutdown();

    let (_policy_dir, policy_path) = write_policy_document();
    let yaml = policy_metering_config(&policy_path, metering.port(), backend.port(), "");
    let config = praxis_core::config::Config::from_yaml(&yaml).unwrap();

    let token = mint_hs256_jwt(&json!({
        "iss": JWT_ISSUER,
        "aud": JWT_AUDIENCE,
        "sub": "alice-subject",
        "exp": now_unix() + 300,
        "iat": now_unix(),
    }));

    let proxy = praxis_test_utils::start_proxy(&config);
    let raw = http_send(
        proxy.addr(),
        &format!(
            "POST /v1/chat/completions HTTP/1.1\r\n\
             Host: localhost\r\n\
             Authorization: Bearer {token}\r\n\
             Content-Type: application/json\r\n\
             Content-Length: 2\r\n\
             Connection: close\r\n\r\n{{}}"
        ),
    );
    assert_eq!(
        parse_status(&raw),
        200,
        "a request with a valid token should be proxied"
    );

    let event = wait_for_usage_event(&metering);
    let json: serde_json::Value = serde_json::from_str(&event).expect("usage event body should be valid JSON");
    assert_eq!(
        json["subject"], "alice-subject",
        "metering must meter the verified subject id, not a request header"
    );
    assert_eq!(json["data"]["user"], "alice-subject");
}

#[test]
fn external_metering_meters_configured_identity_claims() {
    // With identity_username_claim / identity_group_claim configured, metering
    // meters the named custom claims from the verified identity.
    let backend = Backend::status(200, r#"{"ok":true}"#)
        .header("Content-Type", "application/json")
        .start_with_shutdown();
    let metering = StatefulCapturingBackend::new(vec![
        (200, r#"{"hasAccess": true, "balance": 9000.0}"#.to_owned()),
        (204, String::new()),
    ])
    .start_with_shutdown();

    let (_policy_dir, policy_path) = write_policy_document();
    let extra =
        "        identity_username_claim: \"preferred_username\"\n        identity_group_claim: \"organization\"\n";
    let yaml = policy_metering_config(&policy_path, metering.port(), backend.port(), extra);
    let config = praxis_core::config::Config::from_yaml(&yaml).unwrap();

    // preferred_username / organization are non-reserved claims, so the standard
    // claim mapper passes them through to the identity's custom claims. These are
    // the exact claim names documented in the ExternalMeteringFilter end-to-end
    // example, so this test backs that example.
    let token = mint_hs256_jwt(&json!({
        "iss": JWT_ISSUER,
        "aud": JWT_AUDIENCE,
        "sub": "alice-subject",
        "preferred_username": "metered-alice",
        "organization": "metered-engineering",
        "exp": now_unix() + 300,
        "iat": now_unix(),
    }));

    let proxy = praxis_test_utils::start_proxy(&config);
    let raw = http_send(
        proxy.addr(),
        &format!(
            "POST /v1/chat/completions HTTP/1.1\r\n\
             Host: localhost\r\n\
             Authorization: Bearer {token}\r\n\
             Content-Type: application/json\r\n\
             Content-Length: 2\r\n\
             Connection: close\r\n\r\n{{}}"
        ),
    );
    assert_eq!(
        parse_status(&raw),
        200,
        "a request with a valid token should be proxied"
    );

    let event = wait_for_usage_event(&metering);
    let json: serde_json::Value = serde_json::from_str(&event).expect("usage event body should be valid JSON");
    assert_eq!(
        json["subject"], "metered-alice",
        "the configured username claim must win over the subject id"
    );
    assert_eq!(
        json["data"]["group"], "metered-engineering",
        "the configured group claim must be metered"
    );
}
