// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Tests for external metering on a verified identity published by the
//! `policy` filter. Gated on `policy-engine`, which the FIPS build omits.

use std::time::{SystemTime, UNIX_EPOCH};

use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use openssl::{hash::MessageDigest, pkey::PKey, sign::Signer};
use praxis_test_utils::{Backend, StatefulCapturingBackend, free_port, http_send, parse_status};
use serde_json::json;
use tempfile::TempDir;

use super::external_metering::wait_for_usage_event;

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
