// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use bytes::Bytes;
use http::{HeaderValue, Method};
use praxis_filter::{FilterAction, RequestExtensions, TrustedHeaderMutation};

use super::{StateOwner, StateOwnerFilter};
use crate::{
    StateOwnerHeadersFilter, project_state_owner,
    test_utils::{make_filter_context, make_request},
};

const HEADER: &str = "x-test-state-owner";
const TENANT_HEADER: &str = "x-maas-tenant";
const SUBJECT_HEADER: &str = "x-maas-user";

fn filter() -> Box<dyn praxis_filter::HttpFilter> {
    let yaml: serde_yaml::Value = serde_yaml::from_str(&format!("mode: trusted_owner\nheader: {HEADER}")).unwrap();
    StateOwnerFilter::from_config(&yaml).unwrap()
}

#[test]
fn filtered_subrequest_projection_copies_only_normalized_owner() {
    let mut parent = RequestExtensions::new();
    parent.insert(StateOwner::from_trusted_parts("tenant-a".into(), "issuer-a".into(), "alice".into()).unwrap());
    let mut child = RequestExtensions::new();

    assert!(project_state_owner(&parent, &mut child));
    assert_eq!(parent.get::<StateOwner>().unwrap().subject(), "alice");
    let projected = child.get::<StateOwner>().unwrap();
    assert_eq!(projected.tenant_id(), "tenant-a");
    assert_eq!(projected.issuer(), "issuer-a");
    assert_eq!(projected.subject(), "alice");

    let empty = RequestExtensions::new();
    assert!(!project_state_owner(&empty, &mut child));
    assert_eq!(child.get::<StateOwner>().unwrap().subject(), "alice");
}

fn mapped_filter() -> Box<dyn praxis_filter::HttpFilter> {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        "mode: trusted_headers
tenant:
  header: x-maas-tenant
issuer:
  static: https://authorino.example
subject:
  header: x-maas-user",
    )
    .unwrap();
    StateOwnerFilter::from_config(&yaml).unwrap()
}

fn projection_filter() -> Box<dyn praxis_filter::HttpFilter> {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        "tenant_header: x-tenant-id
subject_header: x-user-id
issuer_header: x-identity-issuer",
    )
    .unwrap();
    StateOwnerHeadersFilter::from_config(&yaml).unwrap()
}

fn mapped_request(tenant: &str, subject: &str) -> praxis_filter::Request {
    mapped_request_for_path("/v1/responses", tenant, subject)
}

fn mapped_request_for_path(path: &str, tenant: &str, subject: &str) -> praxis_filter::Request {
    let mut request = make_request(Method::POST, path);
    request
        .headers
        .insert(TENANT_HEADER, HeaderValue::from_str(tenant).unwrap());
    request
        .headers
        .insert(SUBJECT_HEADER, HeaderValue::from_str(subject).unwrap());
    request
}

fn rejection_body(action: FilterAction) -> serde_json::Value {
    let FilterAction::Reject(rejection) = action else {
        panic!("expected rejection");
    };
    serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap()
}

fn assertion(parts: [&str; 3]) -> String {
    let json = serde_json::to_vec(&parts).unwrap();
    format!("v1.{}", URL_SAFE_NO_PAD.encode(json))
}

fn request_with(value: &str) -> praxis_filter::Request {
    let mut request = make_request(Method::POST, "/v1/responses");
    request.headers.insert(HEADER, HeaderValue::from_str(value).unwrap());
    request
}

fn rejection_code(action: FilterAction) -> String {
    let body = rejection_body(action);
    body.pointer("/error/code")
        .and_then(serde_json::Value::as_str)
        .unwrap()
        .to_owned()
}

#[tokio::test]
async fn provider_neutral_context_supports_anthropic_messages() {
    let request = mapped_request_for_path("/v1/messages", "tenant-a", "claude-user");
    let mut ctx = make_filter_context(&request);

    let action = mapped_filter().on_request(&mut ctx).await.unwrap();

    assert!(matches!(action, FilterAction::Continue));
    assert_eq!(ctx.extensions.get::<StateOwner>().unwrap().subject(), "claude-user");

    let missing = make_request(Method::POST, "/v1/messages");
    let mut missing_ctx = make_filter_context(&missing);
    let body = rejection_body(mapped_filter().on_request(&mut missing_ctx).await.unwrap());
    assert_eq!(body.pointer("/type").and_then(serde_json::Value::as_str), Some("error"));
    assert_eq!(
        body.pointer("/error/type").and_then(serde_json::Value::as_str),
        Some("authentication_error")
    );
}

#[test]
fn configuration_requires_valid_header_name() {
    for yaml in [
        "{}",
        "mode: trusted_owner",
        "mode: trusted_owner\nheader: ''",
        "mode: trusted_owner\nheader: 'not a header'",
    ] {
        let value: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
        assert!(StateOwnerFilter::from_config(&value).is_err(), "should reject {yaml}");
    }
}

#[test]
fn configuration_rejects_unknown_fields() {
    let value: serde_yaml::Value = serde_yaml::from_str("mode: trusted_owner\nheader: x-owner\nunknown: true").unwrap();
    assert!(StateOwnerFilter::from_config(&value).is_err());
}

#[test]
fn policy_mode_is_reserved_and_fails_configuration() {
    let value: serde_yaml::Value = serde_yaml::from_str("mode: policy\nheader: x-owner").unwrap();
    let Err(error) = StateOwnerFilter::from_config(&value) else {
        panic!("policy mode must remain unavailable without PPE");
    };
    assert!(error.to_string().contains("requires the PPE integration"));
}

#[tokio::test]
async fn explicit_single_tenant_mode_installs_shared_owner_without_a_header() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("mode: single_tenant\ntenant_id: local").unwrap();
    let filter = StateOwnerFilter::from_config(&yaml).unwrap();
    let request = make_request(Method::GET, "/v1/responses/resp_known");
    let mut ctx = make_filter_context(&request);

    let action = filter.on_request(&mut ctx).await.unwrap();

    assert!(matches!(action, FilterAction::Continue));
    let owner = ctx.extensions.get::<StateOwner>().unwrap();
    assert_eq!(owner.tenant_id(), "local");
    assert_eq!(owner.issuer(), "urn:praxis:single-tenant");
    assert_eq!(owner.subject(), "shared");
    assert!(ctx.request_headers_to_remove.is_empty());
}

#[test]
fn single_tenant_mode_requires_a_valid_namespace() {
    for yaml in ["mode: single_tenant", "mode: single_tenant\ntenant_id: ''"] {
        let value: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
        assert!(StateOwnerFilter::from_config(&value).is_err());
    }
}

#[test]
fn trusted_headers_configuration_rejects_ambiguous_or_invalid_sources() {
    for yaml in [
        "mode: trusted_headers\ntenant: {header: x-owner}\nissuer: {static: issuer}\nsubject: {header: x-owner}",
        "mode: trusted_headers\ntenant: {header: x-tenant, static: tenant}\nissuer: {static: issuer}\nsubject: {header: x-user}",
        "mode: trusted_headers\ntenant: {header: ''}\nissuer: {static: issuer}\nsubject: {header: x-user}",
        "mode: trusted_headers\ntenant: {header: x-tenant}\nissuer: {static: ''}\nsubject: {header: x-user}",
    ] {
        let value: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
        assert!(StateOwnerFilter::from_config(&value).is_err(), "should reject {yaml}");
    }
}

#[tokio::test]
async fn trusted_headers_mode_maps_static_and_header_components() {
    let request = mapped_request("tenant-a", "alice");
    let mut ctx = make_filter_context(&request);

    let action = mapped_filter().on_request(&mut ctx).await.unwrap();

    assert!(matches!(action, FilterAction::Continue));
    let owner = ctx.extensions.get::<StateOwner>().unwrap();
    assert_eq!(owner.tenant_id(), "tenant-a");
    assert_eq!(owner.issuer(), "https://authorino.example");
    assert_eq!(owner.subject(), "alice");
    assert!(ctx.request_headers_to_remove.iter().any(|name| name == TENANT_HEADER));
    assert!(ctx.request_headers_to_remove.iter().any(|name| name == SUBJECT_HEADER));
}

#[test]
fn projection_configuration_rejects_invalid_or_ambiguous_headers() {
    for yaml in [
        "tenant_header: x-tenant-id",
        "tenant_header: x-tenant-id\nsubject_header: x-tenant-id",
        "tenant_header: host\nsubject_header: x-user-id",
        "tenant_header: x-praxis-owner\nsubject_header: x-user-id",
        "tenant_header: x-tenant-id\nsubject_header: 'not a header'",
    ] {
        let value: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
        assert!(
            StateOwnerHeadersFilter::from_config(&value).is_err(),
            "should reject {yaml}"
        );
    }
}

#[tokio::test]
async fn projection_overwrites_ingress_headers_from_normalized_owner() {
    let mut request = mapped_request("tenant-a", "alice");
    request
        .headers
        .insert("x-tenant-id", HeaderValue::from_static("spoofed-tenant"));
    request
        .headers
        .insert("x-user-id", HeaderValue::from_static("spoofed-user"));
    let mut ctx = make_filter_context(&request);

    assert!(matches!(
        mapped_filter().on_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    assert!(matches!(
        projection_filter().on_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));

    let projected = |name: &str| {
        ctx.request_headers_to_set
            .iter()
            .find(|(header, _)| header == name)
            .map(|(_, value)| value.to_str().unwrap())
    };
    assert_eq!(projected("x-tenant-id"), Some("tenant-a"));
    assert_eq!(projected("x-user-id"), Some("alice"));
    assert_eq!(projected("x-identity-issuer"), Some("https://authorino.example"));
}

#[tokio::test]
async fn projection_runs_during_body_pre_read_after_owner_capture() {
    let request = mapped_request("tenant-a", "alice");
    let mut ctx = make_filter_context(&request);
    let mut body = Some(Bytes::from_static(br#"{"model":"gpt-4.1","input":"hi"}"#));

    assert!(matches!(
        mapped_filter()
            .on_request_body(&mut ctx, &mut body, false)
            .await
            .unwrap(),
        FilterAction::BodyDone
    ));
    assert!(matches!(
        projection_filter()
            .on_request_body(&mut ctx, &mut body, false)
            .await
            .unwrap(),
        FilterAction::BodyDone
    ));

    for (name, expected) in [
        ("x-tenant-id", "tenant-a"),
        ("x-user-id", "alice"),
        ("x-identity-issuer", "https://authorino.example"),
    ] {
        assert!(ctx.pre_read_mutations.iter().any(|mutation| {
            matches!(mutation, TrustedHeaderMutation::Set(header, value)
                if header == name && value == HeaderValue::from_static(expected))
        }));
    }
}

#[tokio::test]
async fn projection_fails_closed_without_normalized_owner() {
    let request = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&request);

    let action = projection_filter().on_request(&mut ctx).await.unwrap();

    assert_eq!(rejection_code(action), "missing_state_owner");
    assert!(ctx.request_headers_to_set.is_empty());
}

#[tokio::test]
async fn trusted_headers_mode_queues_all_header_removals_in_body_phase() {
    let request = mapped_request("tenant-a", "alice");
    let mut ctx = make_filter_context(&request);
    let mut body = Some(Bytes::from_static(br#"{"model":"gpt-4.1","input":"hi"}"#));

    let action = mapped_filter()
        .on_request_body(&mut ctx, &mut body, false)
        .await
        .unwrap();

    assert!(matches!(action, FilterAction::BodyDone));
    for expected in [TENANT_HEADER, SUBJECT_HEADER] {
        assert!(
            ctx.pre_read_mutations
                .iter()
                .any(|mutation| matches!(mutation, TrustedHeaderMutation::Remove(name) if name == expected))
        );
    }
}

#[tokio::test]
async fn trusted_headers_mode_fails_closed_on_missing_or_duplicate_components() {
    let mut missing = mapped_request("tenant-a", "alice");
    missing.headers.remove(SUBJECT_HEADER);
    let mut missing_ctx = make_filter_context(&missing);
    let missing_action = mapped_filter().on_request(&mut missing_ctx).await.unwrap();
    assert_eq!(rejection_code(missing_action), "missing_state_owner");

    let mut duplicate = mapped_request("tenant-a", "alice");
    duplicate
        .headers
        .append(SUBJECT_HEADER, HeaderValue::from_static("bob"));
    let mut duplicate_ctx = make_filter_context(&duplicate);
    let duplicate_action = mapped_filter().on_request(&mut duplicate_ctx).await.unwrap();
    assert_eq!(rejection_code(duplicate_action), "invalid_state_owner");

    let invalid = mapped_request("", "alice");
    let mut invalid_ctx = make_filter_context(&invalid);
    let invalid_action = mapped_filter().on_request(&mut invalid_ctx).await.unwrap();
    assert_eq!(rejection_code(invalid_action), "invalid_state_owner");
}

#[tokio::test]
async fn installs_complete_owner_and_queues_removal() {
    let request = request_with(&assertion(["tenant-a", "https://issuer.example", "alice"]));
    let mut ctx = make_filter_context(&request);

    let action = filter().on_request(&mut ctx).await.unwrap();

    assert!(matches!(action, FilterAction::Continue));
    let owner = ctx.extensions.get::<StateOwner>().unwrap();
    assert_eq!(owner.tenant_id(), "tenant-a");
    assert_eq!(owner.issuer(), "https://issuer.example");
    assert_eq!(owner.subject(), "alice");
    assert!(ctx.request_headers_to_remove.iter().any(|name| name == HEADER));
}

#[tokio::test]
async fn body_phase_installs_owner_before_downstream_body_consumers() {
    let request = request_with(&assertion(["tenant-a", "issuer-a", "alice"]));
    let mut ctx = make_filter_context(&request);
    let mut body = Some(Bytes::from_static(br#"{"model":"gpt-4.1","input":"hi"}"#));

    let action = filter().on_request_body(&mut ctx, &mut body, false).await.unwrap();

    assert!(matches!(action, FilterAction::BodyDone));
    assert_eq!(ctx.extensions.get::<StateOwner>().unwrap().subject(), "alice");
    assert!(
        ctx.pre_read_mutations
            .iter()
            .any(|mutation| matches!(mutation, TrustedHeaderMutation::Remove(name) if name == HEADER))
    );
}

#[tokio::test]
async fn missing_assertion_fails_closed() {
    let request = make_request(Method::GET, "/v1/responses/resp_known");
    let mut ctx = make_filter_context(&request);

    let action = filter().on_request(&mut ctx).await.unwrap();

    assert_eq!(rejection_code(action), "missing_state_owner");
}

#[tokio::test]
async fn duplicate_assertion_is_rejected() {
    let encoded = assertion(["tenant-a", "issuer-a", "alice"]);
    let mut request = request_with(&encoded);
    request.headers.append(HEADER, HeaderValue::from_str(&encoded).unwrap());
    let mut ctx = make_filter_context(&request);

    let action = filter().on_request(&mut ctx).await.unwrap();

    assert_eq!(rejection_code(action), "invalid_state_owner");
}

#[tokio::test]
async fn unsupported_or_malformed_assertions_are_rejected() {
    let malformed_json = format!("v1.{}", URL_SAFE_NO_PAD.encode(br#"["only","two"]"#));
    for value in ["v2.abc", "v1.***", malformed_json.as_str()] {
        let request = request_with(value);
        let mut ctx = make_filter_context(&request);
        let action = filter().on_request(&mut ctx).await.unwrap();
        assert_eq!(rejection_code(action), "invalid_state_owner", "should reject {value}");
    }
}

#[tokio::test]
async fn empty_oversized_or_control_components_are_rejected() {
    let oversized = "x".repeat(1_025);
    let values = [
        assertion(["", "issuer-a", "alice"]),
        assertion(["tenant-a", "issuer-a", "line\nbreak"]),
        assertion(["tenant-a", "issuer-a", &oversized]),
    ];
    for value in values {
        let request = request_with(&value);
        let mut ctx = make_filter_context(&request);
        let action = filter().on_request(&mut ctx).await.unwrap();
        assert_eq!(rejection_code(action), "invalid_state_owner");
    }
}

#[tokio::test]
async fn oversized_assertion_is_rejected_before_decoding() {
    let request = request_with(&format!("v1.{}", "a".repeat(4_097)));
    let mut ctx = make_filter_context(&request);
    let action = filter().on_request(&mut ctx).await.unwrap();
    assert_eq!(rejection_code(action), "invalid_state_owner");
}

#[tokio::test]
async fn installed_context_is_not_reparsed_or_replaced() {
    let request = request_with("malformed");
    let mut ctx = make_filter_context(&request);
    ctx.extensions
        .insert(StateOwner::from_trusted_parts("tenant-a".into(), "issuer-a".into(), "alice".into()).unwrap());

    let action = filter().on_request(&mut ctx).await.unwrap();

    assert!(matches!(action, FilterAction::Continue));
    assert_eq!(ctx.extensions.get::<StateOwner>().unwrap().subject(), "alice");
    assert!(ctx.request_headers_to_remove.iter().any(|name| name == HEADER));
}
