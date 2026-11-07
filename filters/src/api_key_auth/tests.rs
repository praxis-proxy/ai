// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Unit tests for the API key authentication filter.

use http::{HeaderValue, Method};
use praxis_filter::FilterAction;
use serde_json::json;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

use crate::test_utils::{make_filter_context, make_request};

// -----------------------------------------------------------------------------
// Config Tests
// -----------------------------------------------------------------------------

#[test]
fn config_rejects_empty_validate_url() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(r#"validate_url: """#).unwrap();
    match super::ApiKeyAuthFilter::from_config(&yaml) {
        Err(err) => assert!(
            err.to_string().contains("validate_url must not be empty"),
            "should reject empty validate_url: {err}"
        ),
        Ok(_) => panic!("empty validate_url should be rejected"),
    }
}

#[test]
fn config_rejects_unknown_fields() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
validate_url: "http://localhost:8080/validate"
bogus: true
"#,
    )
    .unwrap();
    assert!(
        super::ApiKeyAuthFilter::from_config(&yaml).is_err(),
        "unknown fields should be rejected"
    );
}

// -----------------------------------------------------------------------------
// Validation Tests
// -----------------------------------------------------------------------------

#[tokio::test]
#[expect(
    clippy::too_many_lines,
    reason = "test covers the complete authentication metadata contract"
)]
async fn valid_key_passes_and_writes_metadata() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/validate"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "valid": true,
            "username": "yossi",
            "groups": ["ai-eng"],
            "subscription": "enterprise",
            "keyId": "6e3c3851-eab5-4c10-babc-a0366a063caf"
        })))
        .mount(&server)
        .await;

    let yaml: serde_yaml::Value =
        serde_yaml::from_str(&format!(r#"validate_url: "{}/validate""#, server.uri())).unwrap();
    let filter = super::ApiKeyAuthFilter::from_config(&yaml).unwrap();

    let mut req = make_request(Method::POST, "/v1/messages");
    req.headers
        .insert("x-api-key", HeaderValue::from_static("sk-oai-test123"));

    let mut ctx = make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();

    assert!(matches!(action, FilterAction::Continue), "valid key should continue");
    assert_eq!(
        ctx.filter_metadata.get("x-tenant-username"),
        Some(&"yossi".to_owned()),
        "username should be in metadata"
    );
    assert_eq!(
        ctx.filter_metadata.get("x-tenant-group"),
        Some(&"ai-eng".to_owned()),
        "group should be in metadata"
    );
    assert_eq!(
        ctx.filter_metadata.get("x-tenant-subscription"),
        Some(&"enterprise".to_owned()),
        "subscription should be in metadata"
    );
    assert_eq!(
        ctx.filter_metadata.get("api_key.id"),
        Some(&"6e3c3851-eab5-4c10-babc-a0366a063caf".to_owned()),
        "keyId should be in metadata"
    );
}

#[tokio::test]
async fn valid_response_without_key_id_does_not_publish_one() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/validate"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "valid": true,
            "username": "no-key-id",
            "groups": []
        })))
        .mount(&server)
        .await;

    let yaml: serde_yaml::Value =
        serde_yaml::from_str(&format!(r#"validate_url: "{}/validate""#, server.uri())).unwrap();
    let filter = super::ApiKeyAuthFilter::from_config(&yaml).unwrap();

    let mut req = make_request(Method::POST, "/v1/messages");
    req.headers
        .insert("x-api-key", HeaderValue::from_static("sk-oai-no-key-id"));
    let mut ctx = make_filter_context(&req);

    let action = filter.on_request(&mut ctx).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));
    assert_eq!(ctx.filter_metadata.get("api_key.id"), None);
}

#[tokio::test]
async fn invalid_key_rejected() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/validate"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "valid": false,
            "reason": "key not found"
        })))
        .mount(&server)
        .await;

    let yaml: serde_yaml::Value =
        serde_yaml::from_str(&format!(r#"validate_url: "{}/validate""#, server.uri())).unwrap();
    let filter = super::ApiKeyAuthFilter::from_config(&yaml).unwrap();

    let mut req = make_request(Method::POST, "/v1/messages");
    req.headers.insert("x-api-key", HeaderValue::from_static("sk-oai-bad"));

    let mut ctx = make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();

    assert!(
        matches!(action, FilterAction::Reject(_)),
        "invalid key should be rejected"
    );
}

#[tokio::test]
async fn valid_response_without_username_is_rejected() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/validate"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "valid": true,
            "username": "",
            "groups": ["ai-eng"]
        })))
        .mount(&server)
        .await;

    let yaml: serde_yaml::Value =
        serde_yaml::from_str(&format!(r#"validate_url: "{}/validate""#, server.uri())).unwrap();
    let filter = super::ApiKeyAuthFilter::from_config(&yaml).unwrap();

    let mut req = make_request(Method::POST, "/v1/messages");
    req.headers
        .insert("x-api-key", HeaderValue::from_static("sk-oai-empty-user"));
    let mut ctx = make_filter_context(&req);

    let action = filter.on_request(&mut ctx).await.unwrap();
    assert!(
        matches!(action, FilterAction::Reject(_)),
        "empty identity must fail closed"
    );
}

#[tokio::test]
async fn missing_key_rejected() {
    let server = MockServer::start().await;
    let yaml: serde_yaml::Value =
        serde_yaml::from_str(&format!(r#"validate_url: "{}/validate""#, server.uri())).unwrap();
    let filter = super::ApiKeyAuthFilter::from_config(&yaml).unwrap();

    let req = make_request(Method::POST, "/v1/messages");
    let mut ctx = make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();

    assert!(
        matches!(action, FilterAction::Reject(_)),
        "missing key should be rejected"
    );
}

#[tokio::test]
#[expect(
    clippy::too_many_lines,
    reason = "test covers cache population and cache-hit behavior"
)]
async fn cache_hit_skips_callout() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/validate"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "valid": true,
            "username": "cached-user",
            "groups": ["team"],
            "keyId": "cached-key-id"
        })))
        .expect(1) // exactly one callout, not two
        .mount(&server)
        .await;

    let yaml: serde_yaml::Value =
        serde_yaml::from_str(&format!(r#"validate_url: "{}/validate""#, server.uri())).unwrap();
    let filter = super::ApiKeyAuthFilter::from_config(&yaml).unwrap();

    // First request — cache miss, calls endpoint.
    let mut req1 = make_request(Method::POST, "/v1/messages");
    req1.headers
        .insert("x-api-key", HeaderValue::from_static("sk-oai-cached"));
    let mut ctx1 = make_filter_context(&req1);
    let action = filter.on_request(&mut ctx1).await.unwrap();
    drop(action);

    // Second request — cache hit, no callout.
    let mut req2 = make_request(Method::POST, "/v1/messages");
    req2.headers
        .insert("x-api-key", HeaderValue::from_static("sk-oai-cached"));
    let mut ctx2 = make_filter_context(&req2);
    let action = filter.on_request(&mut ctx2).await.unwrap();

    assert!(matches!(action, FilterAction::Continue), "cached key should continue");
    assert_eq!(
        ctx2.filter_metadata.get("x-tenant-username"),
        Some(&"cached-user".to_owned()),
        "cached username should be in metadata"
    );
    assert_eq!(
        ctx2.filter_metadata.get("api_key.id"),
        Some(&"cached-key-id".to_owned()),
        "cached keyId should be in metadata"
    );
    // wiremock's expect(1) verifies only one callout was made.
}

#[tokio::test]
async fn key_header_stripped_from_upstream() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/validate"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "valid": true,
            "username": "stripper",
            "groups": []
        })))
        .mount(&server)
        .await;

    let yaml: serde_yaml::Value =
        serde_yaml::from_str(&format!(r#"validate_url: "{}/validate""#, server.uri())).unwrap();
    let filter = super::ApiKeyAuthFilter::from_config(&yaml).unwrap();

    let mut req = make_request(Method::POST, "/v1/messages");
    req.headers
        .insert("x-api-key", HeaderValue::from_static("sk-oai-strip"));

    let mut ctx = make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();
    drop(action);

    assert!(
        ctx.request_headers_to_remove.iter().any(|h| h.as_str() == "x-api-key"),
        "x-api-key should be marked for removal"
    );
}

#[tokio::test]
async fn endpoint_down_rejects() {
    // No mock server running — connection refused.
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
validate_url: "http://127.0.0.1:1/validate"
timeout_seconds: 1
"#,
    )
    .unwrap();
    let filter = super::ApiKeyAuthFilter::from_config(&yaml).unwrap();

    let mut req = make_request(Method::POST, "/v1/messages");
    req.headers.insert("x-api-key", HeaderValue::from_static("sk-oai-down"));

    let mut ctx = make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();

    assert!(
        matches!(action, FilterAction::Reject(_)),
        "unreachable endpoint should reject (fail-closed)"
    );
}
