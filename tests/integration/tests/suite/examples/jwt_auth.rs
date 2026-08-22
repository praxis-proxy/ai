// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Tests for the JWT authentication example configuration.
//!
//! Functional tests serve a real RSA JWKS from a local backend and
//! send signed tokens through the proxy end-to-end. The keypair is
//! the pre-generated test fixture shared with the unit tests.

use std::{
    collections::HashMap,
    time::{SystemTime, UNIX_EPOCH},
};

use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
use praxis_test_utils::{RoutedBackend, free_port, http_send, parse_body, parse_status, start_header_echo_backend};
use serde_json::json;

// -----------------------------------------------------------------------------
// Test Key Material (pre-generated, test-only, not a secret)
// -----------------------------------------------------------------------------

const TEST_RSA_PRIVATE_PEM: &str = include_str!("../../../../../filters/src/jwt_auth/test_fixtures/rsa_private.pem");

/// Key ID used in both the JWKS response and minted tokens.
const TEST_KID: &str = "example-config-kid";

/// JWKS path as configured in `jwt-auth.yaml`.
const JWKS_PATH: &str = "/realms/ai-gateway/protocol/openid-connect/certs";

/// Build the JWKS JSON body from the test RSA key.
fn jwks_body() -> String {
    let key = EncodingKey::from_rsa_pem(TEST_RSA_PRIVATE_PEM.as_bytes()).unwrap();
    let mut jwk = jsonwebtoken::jwk::Jwk::from_encoding_key(&key, Algorithm::RS256).unwrap();
    jwk.common.key_id = Some(TEST_KID.to_owned());
    jwk.common.key_algorithm = Some(jsonwebtoken::jwk::KeyAlgorithm::RS256);
    serde_json::to_string(&json!({ "keys": [jwk] })).unwrap()
}

/// Mint a signed JWT whose issuer matches the patched example config.
fn mint_token(issuer: &str) -> String {
    let key = EncodingKey::from_rsa_pem(TEST_RSA_PRIVATE_PEM.as_bytes()).unwrap();
    let mut header = Header::new(Algorithm::RS256);
    header.kid = Some(TEST_KID.to_owned());
    let exp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() + 3600;
    let claims = json!({
        "sub": "user-123",
        "preferred_username": "alice",
        "groups": ["ai-eng"],
        "iss": issuer,
        "exp": exp,
    });
    encode(&header, &claims, &key).unwrap()
}

/// Start a JWKS backend and return `(port, issuer)` matching the
/// example config after `127.0.0.1:8280` is remapped to that port.
fn start_jwks_backend() -> (u16, String) {
    let body = jwks_body();
    let port = RoutedBackend::new().route(JWKS_PATH, 200, &body).start();
    (port, format!("http://127.0.0.1:{port}/realms/ai-gateway"))
}

// -----------------------------------------------------------------------------
// Config Parsing
// -----------------------------------------------------------------------------

#[test]
fn jwt_auth_config_parses() {
    let config = super::load_example_config("jwt-auth.yaml", 29930, HashMap::from([("127.0.0.1:3000", 29931_u16)]));

    assert_eq!(config.listeners.len(), 1, "should have 1 listener");
    assert_eq!(&*config.listeners[0].name, "gateway", "listener name should be gateway");
}

// -----------------------------------------------------------------------------
// Valid Token — Proxied and Stripped
// -----------------------------------------------------------------------------

#[test]
fn jwt_auth_allows_valid_token_and_strips_it() {
    let backend_guard = start_header_echo_backend();
    let backend_port = backend_guard.port();
    let proxy_port = free_port();
    let (jwks_port, issuer) = start_jwks_backend();

    let config = super::load_example_config(
        "jwt-auth.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend_port), ("127.0.0.1:8280", jwks_port)]),
    );

    let token = mint_token(&issuer);
    let proxy = praxis_test_utils::start_proxy(&config);
    let raw = http_send(
        proxy.addr(),
        &format!(
            "POST /v1/messages HTTP/1.1\r\n\
             Host: localhost\r\n\
             x-api-key: {token}\r\n\
             Connection: close\r\n\r\n"
        ),
    );

    assert_eq!(parse_status(&raw), 200, "valid token should be proxied");
    let body = parse_body(&raw);
    assert!(
        !body.contains(&token),
        "token header should be stripped before the upstream request: {body}"
    );
}

// -----------------------------------------------------------------------------
// Missing / Invalid Token — Rejected
// -----------------------------------------------------------------------------

#[test]
fn jwt_auth_rejects_missing_token() {
    let backend_guard = start_header_echo_backend();
    let backend_port = backend_guard.port();
    let proxy_port = free_port();
    let (jwks_port, _issuer) = start_jwks_backend();

    let config = super::load_example_config(
        "jwt-auth.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend_port), ("127.0.0.1:8280", jwks_port)]),
    );

    let proxy = praxis_test_utils::start_proxy(&config);
    let raw = http_send(
        proxy.addr(),
        "POST /v1/messages HTTP/1.1\r\n\
         Host: localhost\r\n\
         Connection: close\r\n\r\n",
    );

    assert_eq!(parse_status(&raw), 401, "missing token should be rejected");
}

#[test]
fn jwt_auth_rejects_garbage_token() {
    let backend_guard = start_header_echo_backend();
    let backend_port = backend_guard.port();
    let proxy_port = free_port();
    let (jwks_port, _issuer) = start_jwks_backend();

    let config = super::load_example_config(
        "jwt-auth.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend_port), ("127.0.0.1:8280", jwks_port)]),
    );

    let proxy = praxis_test_utils::start_proxy(&config);
    let raw = http_send(
        proxy.addr(),
        "POST /v1/messages HTTP/1.1\r\n\
         Host: localhost\r\n\
         x-api-key: not-a-jwt\r\n\
         Connection: close\r\n\r\n",
    );

    assert_eq!(parse_status(&raw), 401, "garbage token should be rejected");
}
