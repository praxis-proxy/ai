// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Unit tests for the JWT authentication filter.

use http::{HeaderValue, Method};
use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
use praxis_filter::FilterAction;
use serde_json::json;
use wiremock::{Mock, MockServer, ResponseTemplate, matchers::path};

use crate::test_utils::{make_filter_context, make_request};

// -----------------------------------------------------------------------------
// Test Key Material (pre-generated, test-only, not a secret)
// -----------------------------------------------------------------------------

const TEST_RSA_PRIVATE_PEM: &str = include_str!("test_fixtures/rsa_private.pem");

fn test_encoding_key() -> EncodingKey {
    EncodingKey::from_rsa_pem(TEST_RSA_PRIVATE_PEM.as_bytes()).unwrap()
}

/// Build a JWKS JSON response from the test encoding key.
fn build_jwks_response(kid: &str) -> serde_json::Value {
    let key = test_encoding_key();
    let mut jwk = jsonwebtoken::jwk::Jwk::from_encoding_key(&key, Algorithm::RS256).unwrap();
    jwk.common.key_id = Some(kid.to_owned());
    jwk.common.key_algorithm = Some(jsonwebtoken::jwk::KeyAlgorithm::RS256);
    json!({ "keys": [jwk] })
}

/// Mint a JWT with the given claims.
fn mint_token(kid: &str, claims: &serde_json::Value) -> String {
    let key = test_encoding_key();
    let mut header = Header::new(Algorithm::RS256);
    header.kid = Some(kid.to_owned());
    encode(&header, claims, &key).unwrap()
}

// -----------------------------------------------------------------------------
// Config Tests
// -----------------------------------------------------------------------------

#[test]
fn config_rejects_empty_jwks_url() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
jwks_url: ""
claim_headers:
  preferred_username: "x-tenant-username"
"#,
    )
    .unwrap();
    match super::JwtAuthFilter::from_config(&yaml) {
        Err(err) => assert!(
            err.to_string().contains("jwks_url must not be empty"),
            "should reject empty jwks_url: {err}"
        ),
        Ok(_) => panic!("empty jwks_url should be rejected"),
    }
}

#[test]
fn config_rejects_empty_claim_headers() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
jwks_url: "http://keycloak:8080/certs"
claim_headers: {}
"#,
    )
    .unwrap();
    match super::JwtAuthFilter::from_config(&yaml) {
        Err(err) => assert!(
            err.to_string().contains("claim_headers must have at least one"),
            "should reject empty claim_headers: {err}"
        ),
        Ok(_) => panic!("empty claim_headers should be rejected"),
    }
}

#[test]
fn config_rejects_unknown_fields() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
jwks_url: "http://keycloak:8080/certs"
claim_headers:
  sub: "x-user"
bogus_field: true
"#,
    )
    .unwrap();
    assert!(
        super::JwtAuthFilter::from_config(&yaml).is_err(),
        "unknown fields should be rejected"
    );
}

// -----------------------------------------------------------------------------
// Token Validation Tests
// -----------------------------------------------------------------------------

#[tokio::test]
async fn valid_token_passes_and_writes_metadata() {
    let kid = "test-kid-1";

    let server = MockServer::start().await;
    Mock::given(path("/certs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(build_jwks_response(kid)))
        .mount(&server)
        .await;

    let yaml: serde_yaml::Value = serde_yaml::from_str(&format!(
        r#"
jwks_url: "{}/certs"
token_header: "x-api-key"
claim_headers:
  preferred_username: "x-tenant-username"
  groups: "x-tenant-group"
"#,
        server.uri()
    ))
    .unwrap();
    let filter = super::JwtAuthFilter::from_config(&yaml).unwrap();

    let claims = json!({
        "sub": "user-123",
        "preferred_username": "yossi",
        "groups": ["ai-eng"],
        "iss": "test",
        "exp": chrono::Utc::now().timestamp() + 3600
    });
    let token = mint_token(kid, &claims);

    let mut req = make_request(Method::POST, "/v1/messages");
    req.headers.insert("x-api-key", HeaderValue::from_str(&token).unwrap());

    let mut ctx = make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();

    assert!(matches!(action, FilterAction::Continue), "valid token should continue");
    assert_eq!(
        ctx.filter_metadata.get("x-tenant-username"),
        Some(&"yossi".to_owned()),
        "username should be in metadata"
    );
    assert_eq!(
        ctx.filter_metadata.get("x-tenant-group"),
        Some(&"ai-eng".to_owned()),
        "groups should be joined and in metadata"
    );
}

#[tokio::test]
async fn valid_token_queues_token_header_for_removal() {
    let kid = "test-kid-strip";

    let server = MockServer::start().await;
    Mock::given(path("/certs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(build_jwks_response(kid)))
        .mount(&server)
        .await;

    let yaml: serde_yaml::Value = serde_yaml::from_str(&format!(
        r#"
jwks_url: "{}/certs"
token_header: "x-api-key"
claim_headers:
  sub: "x-tenant-username"
"#,
        server.uri()
    ))
    .unwrap();
    let filter = super::JwtAuthFilter::from_config(&yaml).unwrap();

    let claims = json!({
        "sub": "user-123",
        "iss": "test",
        "exp": chrono::Utc::now().timestamp() + 3600
    });
    let token = mint_token(kid, &claims);

    let mut req = make_request(Method::POST, "/v1/messages");
    req.headers.insert("x-api-key", HeaderValue::from_str(&token).unwrap());

    let mut ctx = make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();

    assert!(matches!(action, FilterAction::Continue), "valid token should continue");
    assert!(
        ctx.request_headers_to_remove.iter().any(|h| h.as_str() == "x-api-key"),
        "token header should be queued for removal so the JWT does not leak upstream"
    );
}

#[tokio::test]
async fn expired_token_rejected() {
    let kid = "test-kid-2";

    let server = MockServer::start().await;
    Mock::given(path("/certs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(build_jwks_response(kid)))
        .mount(&server)
        .await;

    let yaml: serde_yaml::Value = serde_yaml::from_str(&format!(
        r#"
jwks_url: "{}/certs"
token_header: "x-api-key"
claim_headers:
  sub: "x-tenant-username"
"#,
        server.uri()
    ))
    .unwrap();
    let filter = super::JwtAuthFilter::from_config(&yaml).unwrap();

    let claims = json!({
        "sub": "user-123",
        "exp": chrono::Utc::now().timestamp() - 3600
    });
    let token = mint_token(kid, &claims);

    let mut req = make_request(Method::POST, "/v1/messages");
    req.headers.insert("x-api-key", HeaderValue::from_str(&token).unwrap());

    let mut ctx = make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();

    assert!(
        matches!(action, FilterAction::Reject(_)),
        "expired token should be rejected"
    );
}

#[tokio::test]
async fn wrong_issuer_rejected() {
    let kid = "test-kid-3";

    let server = MockServer::start().await;
    Mock::given(path("/certs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(build_jwks_response(kid)))
        .mount(&server)
        .await;

    let yaml: serde_yaml::Value = serde_yaml::from_str(&format!(
        r#"
jwks_url: "{}/certs"
issuer: "https://expected-issuer.com"
token_header: "x-api-key"
claim_headers:
  sub: "x-tenant-username"
"#,
        server.uri()
    ))
    .unwrap();
    let filter = super::JwtAuthFilter::from_config(&yaml).unwrap();

    let claims = json!({
        "sub": "user-123",
        "iss": "https://wrong-issuer.com",
        "exp": chrono::Utc::now().timestamp() + 3600
    });
    let token = mint_token(kid, &claims);

    let mut req = make_request(Method::POST, "/v1/messages");
    req.headers.insert("x-api-key", HeaderValue::from_str(&token).unwrap());

    let mut ctx = make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();

    assert!(
        matches!(action, FilterAction::Reject(_)),
        "wrong issuer should be rejected"
    );
}

#[tokio::test]
async fn unknown_kid_rejected() {
    let server = MockServer::start().await;
    Mock::given(path("/certs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(build_jwks_response("published-kid")))
        .mount(&server)
        .await;

    let yaml: serde_yaml::Value = serde_yaml::from_str(&format!(
        r#"
jwks_url: "{}/certs"
token_header: "x-api-key"
claim_headers:
  sub: "x-tenant-username"
"#,
        server.uri()
    ))
    .unwrap();
    let filter = super::JwtAuthFilter::from_config(&yaml).unwrap();

    let claims = json!({
        "sub": "user-123",
        "exp": chrono::Utc::now().timestamp() + 3600
    });
    let token = mint_token("unknown-kid", &claims);

    let mut req = make_request(Method::POST, "/v1/messages");
    req.headers.insert("x-api-key", HeaderValue::from_str(&token).unwrap());

    let mut ctx = make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();

    assert!(
        matches!(action, FilterAction::Reject(_)),
        "unknown kid should be rejected"
    );
}

#[tokio::test]
async fn missing_token_rejected() {
    let server = MockServer::start().await;
    Mock::given(path("/certs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(build_jwks_response("kid")))
        .mount(&server)
        .await;

    let yaml: serde_yaml::Value = serde_yaml::from_str(&format!(
        r#"
jwks_url: "{}/certs"
token_header: "x-api-key"
claim_headers:
  sub: "x-tenant-username"
"#,
        server.uri()
    ))
    .unwrap();
    let filter = super::JwtAuthFilter::from_config(&yaml).unwrap();

    let req = make_request(Method::POST, "/v1/messages");
    let mut ctx = make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();

    assert!(
        matches!(action, FilterAction::Reject(_)),
        "missing token should be rejected"
    );
}

#[tokio::test]
async fn garbage_token_rejected() {
    let server = MockServer::start().await;
    Mock::given(path("/certs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(build_jwks_response("kid")))
        .mount(&server)
        .await;

    let yaml: serde_yaml::Value = serde_yaml::from_str(&format!(
        r#"
jwks_url: "{}/certs"
token_header: "x-api-key"
claim_headers:
  sub: "x-tenant-username"
"#,
        server.uri()
    ))
    .unwrap();
    let filter = super::JwtAuthFilter::from_config(&yaml).unwrap();

    let mut req = make_request(Method::POST, "/v1/messages");
    req.headers
        .insert("x-api-key", HeaderValue::from_static("not.a.jwt.at.all"));

    let mut ctx = make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();

    assert!(
        matches!(action, FilterAction::Reject(_)),
        "garbage token should be rejected"
    );
}

#[tokio::test]
async fn bearer_prefix_extraction() {
    let kid = "test-kid-bearer";

    let server = MockServer::start().await;
    Mock::given(path("/certs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(build_jwks_response(kid)))
        .mount(&server)
        .await;

    let yaml: serde_yaml::Value = serde_yaml::from_str(&format!(
        r#"
jwks_url: "{}/certs"
token_header: "authorization"
claim_headers:
  preferred_username: "x-tenant-username"
"#,
        server.uri()
    ))
    .unwrap();
    let filter = super::JwtAuthFilter::from_config(&yaml).unwrap();

    let claims = json!({
        "preferred_username": "yossi",
        "exp": chrono::Utc::now().timestamp() + 3600
    });
    let token = mint_token(kid, &claims);
    let bearer = format!("Bearer {token}");

    let mut req = make_request(Method::POST, "/v1/messages");
    req.headers
        .insert("authorization", HeaderValue::from_str(&bearer).unwrap());

    let mut ctx = make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();

    assert!(
        matches!(action, FilterAction::Continue),
        "Bearer prefix should be stripped correctly"
    );
    assert_eq!(
        ctx.filter_metadata.get("x-tenant-username"),
        Some(&"yossi".to_owned()),
        "username should be extracted from Bearer token"
    );
}
