// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

use super::*;
use crate::test_utils::{make_filter_context, make_request};

// -----------------------------------------------------------------------------
// Config Parsing
// -----------------------------------------------------------------------------

#[test]
fn default_config_parses() {
    let filter = filter_from_yaml("{}");

    assert_eq!(filter.name(), "external_metering");
    assert_eq!(filter.identity_header_prefix, "x-tenant-");
}

#[test]
fn custom_prefix_parses() {
    let filter = filter_from_yaml("identity_header_prefix: \"x-myco-\"\n");

    assert_eq!(filter.identity_header_prefix, "x-myco-");
}

#[test]
fn config_empty_prefix_fails() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("identity_header_prefix: \"\"\n").unwrap();

    assert!(ExternalMeteringFilter::from_config(&yaml).is_err());
}

#[test]
fn config_unknown_field_fails() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("not_a_real_field: true\n").unwrap();

    assert!(ExternalMeteringFilter::from_config(&yaml).is_err());
}

// -----------------------------------------------------------------------------
// Header Removal
// -----------------------------------------------------------------------------

#[tokio::test]
async fn strips_tenant_and_credential_headers() {
    let mut req = make_request(http::Method::POST, "/v1/chat/completions");
    req.headers.insert("x-tenant-username", "alice".parse().unwrap());
    req.headers.insert("x-tenant-group", "engineering".parse().unwrap());
    req.headers.insert("authorization", "Bearer redacted".parse().unwrap());
    let mut ctx = make_filter_context(&req);

    let action = filter_from_yaml("{}").on_request(&mut ctx).await.unwrap();

    assert!(matches!(action, FilterAction::Continue));
    let removed = removed_headers(&ctx);
    assert!(removed.contains(&"x-tenant-username"));
    assert!(removed.contains(&"x-tenant-group"));
    assert!(removed.contains(&"authorization"));
    assert!(removed.contains(&"x-api-key"));
}

#[tokio::test]
async fn strips_tenant_headers_under_a_custom_prefix() {
    let mut req = make_request(http::Method::POST, "/v1/chat/completions");
    req.headers.insert("x-myco-username", "bob".parse().unwrap());
    let mut ctx = make_filter_context(&req);

    let _action = filter_from_yaml("identity_header_prefix: \"x-myco-\"\n")
        .on_request(&mut ctx)
        .await
        .unwrap();

    assert!(removed_headers(&ctx).contains(&"x-myco-username"));
}

#[tokio::test]
async fn leaves_unrelated_headers_in_place() {
    let mut req = make_request(http::Method::POST, "/v1/chat/completions");
    req.headers.insert("x-tenant-username", "alice".parse().unwrap());
    req.headers.insert("content-type", "application/json".parse().unwrap());
    let mut ctx = make_filter_context(&req);

    let _action = filter_from_yaml("{}").on_request(&mut ctx).await.unwrap();

    assert!(!removed_headers(&ctx).contains(&"content-type"));
}

#[tokio::test]
async fn header_matching_ignores_case() {
    let mut req = make_request(http::Method::POST, "/v1/chat/completions");
    req.headers.insert("X-Tenant-Username", "alice".parse().unwrap());
    let mut ctx = make_filter_context(&req);

    let _action = filter_from_yaml("identity_header_prefix: \"X-Tenant-\"\n")
        .on_request(&mut ctx)
        .await
        .unwrap();

    assert!(removed_headers(&ctx).contains(&"x-tenant-username"));
}

#[tokio::test]
async fn credentials_are_stripped_even_without_tenant_headers() {
    let req = make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = make_filter_context(&req);

    let _action = filter_from_yaml("{}").on_request(&mut ctx).await.unwrap();

    let removed = removed_headers(&ctx);
    assert!(removed.contains(&"authorization"));
    assert!(removed.contains(&"x-api-key"));
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

/// Build the concrete filter from a YAML fragment.
fn filter_from_yaml(yaml: &str) -> ExternalMeteringFilter {
    let parsed: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
    ExternalMeteringFilter::build(&parsed).unwrap()
}

/// Names of the headers the filter marked for removal.
fn removed_headers<'ctx>(ctx: &'ctx HttpFilterContext<'_>) -> Vec<&'ctx str> {
    ctx.request_headers_to_remove.iter().map(HeaderName::as_str).collect()
}
