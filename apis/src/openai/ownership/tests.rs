// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use bytes::Bytes;
use http::{HeaderValue, Method};
use praxis_filter::{FilterAction, TrustedHeaderMutation};

use super::{OpenAiStateOwner, OpenAiStateOwnerFilter};
use crate::test_utils::{make_filter_context, make_request};

const HEADER: &str = "x-test-state-owner";

fn filter() -> Box<dyn praxis_filter::HttpFilter> {
    let yaml: serde_yaml::Value = serde_yaml::from_str(&format!("mode: trusted_owner\nheader: {HEADER}")).unwrap();
    OpenAiStateOwnerFilter::from_config(&yaml).unwrap()
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
    let FilterAction::Reject(rejection) = action else {
        panic!("expected rejection");
    };
    let body: serde_json::Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();
    body.pointer("/error/code")
        .and_then(serde_json::Value::as_str)
        .unwrap()
        .to_owned()
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
        assert!(
            OpenAiStateOwnerFilter::from_config(&value).is_err(),
            "should reject {yaml}"
        );
    }
}

#[test]
fn configuration_rejects_unknown_fields() {
    let value: serde_yaml::Value = serde_yaml::from_str("mode: trusted_owner\nheader: x-owner\nunknown: true").unwrap();
    assert!(OpenAiStateOwnerFilter::from_config(&value).is_err());
}

#[test]
fn policy_mode_is_reserved_and_fails_configuration() {
    let value: serde_yaml::Value = serde_yaml::from_str("mode: policy\nheader: x-owner").unwrap();
    let Err(error) = OpenAiStateOwnerFilter::from_config(&value) else {
        panic!("policy mode must remain unavailable without PPE");
    };
    assert!(error.to_string().contains("requires the PPE integration"));
}

#[tokio::test]
async fn explicit_single_tenant_mode_installs_shared_owner_without_a_header() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("mode: single_tenant\ntenant_id: local").unwrap();
    let filter = OpenAiStateOwnerFilter::from_config(&yaml).unwrap();
    let request = make_request(Method::GET, "/v1/responses/resp_known");
    let mut ctx = make_filter_context(&request);

    let action = filter.on_request(&mut ctx).await.unwrap();

    assert!(matches!(action, FilterAction::Continue));
    let owner = ctx.extensions.get::<OpenAiStateOwner>().unwrap();
    assert_eq!(owner.tenant_id(), "local");
    assert_eq!(owner.issuer(), "urn:praxis:single-tenant");
    assert_eq!(owner.subject(), "shared");
    assert!(ctx.request_headers_to_remove.is_empty());
}

#[test]
fn single_tenant_mode_requires_a_valid_namespace() {
    for yaml in ["mode: single_tenant", "mode: single_tenant\ntenant_id: ''"] {
        let value: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
        assert!(OpenAiStateOwnerFilter::from_config(&value).is_err());
    }
}

#[tokio::test]
async fn installs_complete_owner_and_queues_removal() {
    let request = request_with(&assertion(["tenant-a", "https://issuer.example", "alice"]));
    let mut ctx = make_filter_context(&request);

    let action = filter().on_request(&mut ctx).await.unwrap();

    assert!(matches!(action, FilterAction::Continue));
    let owner = ctx.extensions.get::<OpenAiStateOwner>().unwrap();
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
    assert_eq!(ctx.extensions.get::<OpenAiStateOwner>().unwrap().subject(), "alice");
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
        .insert(OpenAiStateOwner::from_trusted_parts("tenant-a".into(), "issuer-a".into(), "alice".into()).unwrap());

    let action = filter().on_request(&mut ctx).await.unwrap();

    assert!(matches!(action, FilterAction::Continue));
    assert_eq!(ctx.extensions.get::<OpenAiStateOwner>().unwrap().subject(), "alice");
    assert!(ctx.request_headers_to_remove.iter().any(|name| name == HEADER));
}
