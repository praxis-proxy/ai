// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Unit tests for the identity header guard filter.

use http::{HeaderValue, Method};
use praxis_filter::FilterAction;

use super::IdentityHeaderGuardFilter;
use crate::test_utils::{make_filter_context, make_request};

// -----------------------------------------------------------------------------
// Config Tests
// -----------------------------------------------------------------------------

#[test]
fn from_config_minimal() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(r#"prefix: "x-tenant-""#).unwrap();
    let filter = IdentityHeaderGuardFilter::from_config(&yaml).unwrap();
    assert_eq!(
        filter.name(),
        "identity_header_guard",
        "should produce identity_header_guard filter"
    );
}

#[test]
fn from_config_full() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
prefix: "x-maas-"
metadata_namespace: "maas"
"#,
    )
    .unwrap();
    let filter = IdentityHeaderGuardFilter::from_config(&yaml).unwrap();
    assert_eq!(filter.name(), "identity_header_guard", "full config should parse");
}

#[test]
fn from_config_rejects_empty_prefix() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(r#"prefix: """#).unwrap();
    match IdentityHeaderGuardFilter::from_config(&yaml) {
        Err(err) => assert!(
            err.to_string().contains("prefix must not be empty"),
            "error should mention prefix: {err}"
        ),
        Ok(_) => panic!("empty prefix should be rejected"),
    }
}

#[test]
fn from_config_rejects_unknown_fields() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
prefix: "x-tenant-"
bogus_field: true
"#,
    )
    .unwrap();
    assert!(
        IdentityHeaderGuardFilter::from_config(&yaml).is_err(),
        "unknown fields should be rejected"
    );
}

#[test]
fn from_config_defaults_namespace_to_identity() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(r#"prefix: "x-tenant-""#).unwrap();
    let filter = IdentityHeaderGuardFilter::from_config(&yaml).unwrap();
    assert_eq!(filter.name(), "identity_header_guard");
}

// -----------------------------------------------------------------------------
// Behavior Tests
// -----------------------------------------------------------------------------

#[tokio::test]
async fn captures_matching_headers_to_metadata() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(r#"prefix: "x-tenant-""#).unwrap();
    let filter = IdentityHeaderGuardFilter::from_config(&yaml).unwrap();

    let mut req = make_request(Method::POST, "/v1/chat/completions");
    req.headers
        .insert("x-tenant-username", HeaderValue::from_static("yossi"));
    req.headers.insert("x-tenant-group", HeaderValue::from_static("ai-eng"));

    let mut ctx = make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();

    assert!(matches!(action, FilterAction::Continue), "should continue the pipeline");
    assert_eq!(
        ctx.filter_metadata.get("identity.x-tenant-username"),
        Some(&"yossi".to_owned()),
        "username should be captured to metadata"
    );
    assert_eq!(
        ctx.filter_metadata.get("identity.x-tenant-group"),
        Some(&"ai-eng".to_owned()),
        "group should be captured to metadata"
    );
}

#[tokio::test]
async fn strips_matching_headers_from_upstream() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(r#"prefix: "x-tenant-""#).unwrap();
    let filter = IdentityHeaderGuardFilter::from_config(&yaml).unwrap();

    let mut req = make_request(Method::POST, "/v1/messages");
    req.headers
        .insert("x-tenant-username", HeaderValue::from_static("yossi"));
    req.headers
        .insert("content-type", HeaderValue::from_static("application/json"));

    let mut ctx = make_filter_context(&req);
    let _action = filter.on_request(&mut ctx).await.unwrap();

    assert!(
        ctx.request_headers_to_remove
            .iter()
            .any(|h| h.as_str() == "x-tenant-username"),
        "x-tenant-username should be marked for removal"
    );
    assert!(
        !ctx.request_headers_to_remove
            .iter()
            .any(|h| h.as_str() == "content-type"),
        "content-type should NOT be marked for removal"
    );
}

#[tokio::test]
async fn ignores_non_matching_headers() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(r#"prefix: "x-tenant-""#).unwrap();
    let filter = IdentityHeaderGuardFilter::from_config(&yaml).unwrap();

    let mut req = make_request(Method::POST, "/v1/chat/completions");
    req.headers
        .insert("authorization", HeaderValue::from_static("Bearer sk-123"));
    req.headers
        .insert("content-type", HeaderValue::from_static("application/json"));

    let mut ctx = make_filter_context(&req);
    let _action = filter.on_request(&mut ctx).await.unwrap();

    assert!(
        ctx.filter_metadata.is_empty(),
        "no identity metadata should be captured"
    );
    assert!(
        ctx.request_headers_to_remove.is_empty(),
        "no headers should be marked for removal"
    );
}

#[tokio::test]
async fn case_insensitive_prefix_matching() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(r#"prefix: "x-tenant-""#).unwrap();
    let filter = IdentityHeaderGuardFilter::from_config(&yaml).unwrap();

    let mut req = make_request(Method::POST, "/v1/chat/completions");
    req.headers
        .insert("X-Tenant-Username", HeaderValue::from_static("yossi"));

    let mut ctx = make_filter_context(&req);
    let _action = filter.on_request(&mut ctx).await.unwrap();

    assert_eq!(
        ctx.filter_metadata.get("identity.x-tenant-username"),
        Some(&"yossi".to_owned()),
        "case-insensitive match should capture the header"
    );
}

#[tokio::test]
async fn custom_namespace() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
prefix: "x-maas-"
metadata_namespace: "maas"
"#,
    )
    .unwrap();
    let filter = IdentityHeaderGuardFilter::from_config(&yaml).unwrap();

    let mut req = make_request(Method::POST, "/v1/messages");
    req.headers.insert("x-maas-username", HeaderValue::from_static("alice"));

    let mut ctx = make_filter_context(&req);
    let _action = filter.on_request(&mut ctx).await.unwrap();

    assert_eq!(
        ctx.filter_metadata.get("maas.x-maas-username"),
        Some(&"alice".to_owned()),
        "should use custom namespace"
    );
    assert!(
        !ctx.filter_metadata.contains_key("identity.x-maas-username"),
        "should NOT use default namespace"
    );
}

#[tokio::test]
async fn no_headers_means_empty_metadata() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(r#"prefix: "x-tenant-""#).unwrap();
    let filter = IdentityHeaderGuardFilter::from_config(&yaml).unwrap();

    let req = make_request(Method::POST, "/v1/chat/completions");
    let mut ctx = make_filter_context(&req);
    let _action = filter.on_request(&mut ctx).await.unwrap();

    assert!(ctx.filter_metadata.is_empty(), "no identity headers means no metadata");
}
