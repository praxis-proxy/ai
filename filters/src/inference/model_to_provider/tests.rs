// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Tests for stable model ID resolution.

use bytes::Bytes;
use http::Method;
use praxis_filter::FilterAction;
use serde_json::json;

use super::*;
use crate::test_utils::{make_filter_context, make_request};

fn build_filter(yaml: &str) -> Box<dyn HttpFilter> {
    let value: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
    ModelToProviderFilter::from_config(&value).unwrap()
}

fn vertex_config() -> &'static str {
    "models:\n  - model: claude-sonnet-4-5\n    provider: vertex\n    target_model: vertex/claude-sonnet-4-5\n    paths: [/v1/messages, /v1/messages/count_tokens]"
}

#[test]
fn config_rejects_empty_duplicates_and_invalid_entries() {
    for yaml in [
        "models: []",
        "models:\n  - model: same\n    provider: vertex\n    target_model: vertex/a\n    paths: [/v1/messages]\n  - model: same\n    provider: other\n    target_model: other/a\n    paths: [/v1/messages]",
        "models:\n  - model: claude\n    provider: ''\n    target_model: vertex/a\n    paths: [/v1/messages]",
        "models:\n  - model: claude\n    provider: vertex\n    target_model: vertex/a\n    paths: []",
        "models:\n  - model: claude\n    provider: vertex\n    target_model: vertex/a\n    paths: [relative/path]",
        "models:\n  - model: claude\n    provider: vertex\n    target_model: vertex/a\n    paths: [/v1/messages, /v1/messages]",
    ] {
        let value: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
        assert!(
            ModelToProviderFilter::from_config(&value).is_err(),
            "must reject: {yaml}"
        );
    }
}

#[tokio::test]
async fn mapping_sets_provider_marker_preserves_public_id_and_rewrites_target() {
    let filter = build_filter(vertex_config());
    let request = make_request(Method::POST, "/v1/messages");
    let mut ctx = make_filter_context(&request);
    let mut body = Some(Bytes::from(
        json!({"model":"claude-sonnet-4-5","messages":[]}).to_string(),
    ));

    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    assert!(matches!(
        filter.on_request_body(&mut ctx, &mut body, true).await.unwrap(),
        FilterAction::Continue
    ));

    assert!(
        ctx.request_headers_to_set
            .iter()
            .any(|(name, value)| { name == PROVIDER_HEADER && value == HeaderValue::from_static("vertex") })
    );
    assert_eq!(
        ctx.get_metadata(MODEL_PROVIDER_CLIENT_MODEL_METADATA),
        Some("claude-sonnet-4-5")
    );
    let parsed: Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    assert_eq!(parsed["model"], "vertex/claude-sonnet-4-5");
}

#[tokio::test]
async fn mapping_ignores_unknown_models_and_non_matching_paths() {
    let filter = build_filter(vertex_config());
    for (path, model) in [
        ("/v1/messages", "claude-opus-4-5"),
        ("/v1/chat/completions", "claude-sonnet-4-5"),
    ] {
        let request = make_request(Method::POST, path);
        let mut ctx = make_filter_context(&request);
        let original = Bytes::from(json!({"model":model,"messages":[]}).to_string());
        let mut body = Some(original.clone());

        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
        assert!(matches!(action, FilterAction::Continue));
        assert_eq!(body.as_ref(), Some(&original), "{path} {model}");
        assert!(ctx.request_headers_to_set.is_empty());
        assert!(ctx.get_metadata(MODEL_PROVIDER_CLIENT_MODEL_METADATA).is_none());
    }
}

#[tokio::test]
async fn mapping_matches_path_without_query_string() {
    let filter = build_filter(vertex_config());
    let request = make_request(Method::POST, "/v1/messages/count_tokens?client=cli");
    let mut ctx = make_filter_context(&request);
    let mut body = Some(Bytes::from_static(br#"{"model":"claude-sonnet-4-5"}"#));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));
    assert!(
        ctx.request_headers_to_set
            .iter()
            .any(|(name, value)| { name == PROVIDER_HEADER && value == HeaderValue::from_static("vertex") })
    );
}
