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

    assert!(
        matches!(filter.on_request(&mut ctx).await.unwrap(), FilterAction::Continue),
        "the request hook must continue until the body is available"
    );
    assert!(
        matches!(
            filter.on_request_body(&mut ctx, &mut body, true).await.unwrap(),
            FilterAction::Continue
        ),
        "a mapped request must continue with its transformed body"
    );

    assert_vertex_selector(&ctx);
    assert_eq!(
        ctx.get_metadata(MODEL_PROVIDER_CLIENT_MODEL_METADATA),
        Some("claude-sonnet-4-5"),
        "metadata must retain the original public model for metering"
    );
    let parsed: Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    assert_eq!(
        parsed["model"], "vertex/claude-sonnet-4-5",
        "the backend body must carry the configured target model"
    );
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
        assert!(
            matches!(action, FilterAction::Continue),
            "an unmapped request must continue to its fallback"
        );
        assert_eq!(
            body.as_ref(),
            Some(&original),
            "an unmapped body must remain unchanged: {path} {model}"
        );
        assert!(
            ctx.request_headers_to_set.is_empty(),
            "an unmapped request must not acquire a provider selector"
        );
        assert!(
            ctx.get_metadata(MODEL_PROVIDER_CLIENT_MODEL_METADATA).is_none(),
            "an unmapped request must not acquire mapped-model metadata"
        );
    }
}

#[tokio::test]
async fn mapping_matches_path_without_query_string() {
    let filter = build_filter(vertex_config());
    let request = make_request(Method::POST, "/v1/messages/count_tokens?client=cli");
    let mut ctx = make_filter_context(&request);
    let mut body = Some(Bytes::from_static(br#"{"model":"claude-sonnet-4-5"}"#));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "query parameters must not prevent model mapping"
    );
    assert!(
        ctx.request_headers_to_set
            .iter()
            .any(|(name, value)| { name == PROVIDER_HEADER && value == HeaderValue::from_static("vertex") }),
        "a mapped path with a query must still select the vertex provider"
    );
}

#[tokio::test]
async fn mapping_joins_an_already_active_ordered_mutation_log() {
    let filter = build_filter(vertex_config());
    let request = make_request(Method::POST, "/v1/messages");
    let mut ctx = make_filter_context(&request);
    ctx.pre_read_mutations.push(TrustedHeaderMutation::Set(
        HeaderName::from_static("x-trusted-tenant"),
        HeaderValue::from_static("acme"),
    ));
    let mut body = Some(Bytes::from_static(br#"{"model":"claude-sonnet-4-5","messages":[]}"#));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(
        matches!(action, FilterAction::Continue),
        "mapping must continue when an ordered mutation log is already active"
    );
    assert_vertex_selector(&ctx);
    assert_eq!(
        ctx.pre_read_mutations.len(),
        3,
        "the selector should be appended after the earlier ordered mutation"
    );
    assert!(
        matches!(
            ctx.pre_read_mutations.last(),
            Some(TrustedHeaderMutation::Set(name, value)) if *name == PROVIDER_HEADER && value == "vertex"
        ),
        "core ignores grouped queues once the ordered log is active, so the selector must be in it: {:?}",
        ctx.pre_read_mutations
    );
}

#[tokio::test]
async fn mapping_does_not_activate_the_ordered_mutation_log() {
    let filter = build_filter(vertex_config());
    let request = make_request(Method::POST, "/v1/messages");
    let mut ctx = make_filter_context(&request);
    let mut body = Some(Bytes::from_static(br#"{"model":"claude-sonnet-4-5","messages":[]}"#));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(
        matches!(action, FilterAction::Continue),
        "mapping must continue without activating ordered mutation mode"
    );
    assert!(
        ctx.pre_read_mutations.is_empty(),
        "writing the ordered log would make core drop other filters' grouped mutations"
    );
}

#[tokio::test]
async fn mapping_clears_stale_selector_and_metadata_before_early_returns() {
    let filter = build_filter(vertex_config());
    for (path, raw) in [
        ("/v1/messages", "not JSON"),
        ("/v1/messages", "{\"model\":\"unknown\"}"),
        ("/other", "{\"model\":\"claude-sonnet-4-5\"}"),
        ("/v1/messages", "{}"),
    ] {
        let request = make_request(Method::POST, path);
        let mut ctx = make_filter_context(&request);
        ctx.set_metadata(MODEL_PROVIDER_CLIENT_MODEL_METADATA, "previous-model");
        ctx.pre_read_mutations.push(TrustedHeaderMutation::Set(
            PROVIDER_HEADER,
            HeaderValue::from_static("spoofed"),
        ));
        let mut body = Some(Bytes::copy_from_slice(raw.as_bytes()));
        assert!(
            matches!(
                filter.on_request_body(&mut ctx, &mut body, true).await.unwrap(),
                FilterAction::Continue
            ),
            "early returns must preserve request processing: {path} {raw}"
        );
        assert_stale_mapping_removed(&ctx);
    }
}

/// Assert the selector the router needs after a mapped request.
fn assert_vertex_selector(ctx: &HttpFilterContext<'_>) {
    assert!(
        ctx.request_headers_to_set
            .iter()
            .any(|(name, value)| name == PROVIDER_HEADER && value == "vertex"),
        "a mapped request must queue the vertex provider selector"
    );
}

/// Assert stale mapping state is cleared even when body handling returns early.
fn assert_stale_mapping_removed(ctx: &HttpFilterContext<'_>) {
    assert!(
        ctx.get_metadata(MODEL_PROVIDER_CLIENT_MODEL_METADATA).is_none(),
        "every early return must clear stale public-model metadata"
    );
    assert!(
        ctx.request_headers_to_remove.contains(&PROVIDER_HEADER),
        "every early return must queue stale selector removal"
    );
    assert!(
        matches!(ctx.pre_read_mutations.last(), Some(TrustedHeaderMutation::Remove(name)) if *name == PROVIDER_HEADER),
        "stale selector removal must follow earlier ordered mutations"
    );
}
