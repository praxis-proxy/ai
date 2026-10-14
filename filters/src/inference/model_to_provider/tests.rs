// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Tests for stable model ID resolution.

use bytes::Bytes;
use http::Method;
use praxis_filter::FilterAction;
use serde_json::json;

use super::*;
use crate::test_utils::{make_filter_context, make_request};

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
        ("/v1/messages", "[]"),
        ("/v1/messages", "{\"model\":42}"),
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

#[tokio::test]
async fn mapping_clears_a_previous_grouped_selector_without_changing_other_headers() {
    let filter = build_filter(vertex_config());
    let request = make_request(Method::POST, "/v1/messages");
    let mut ctx = make_filter_context(&request);
    let tenant = HeaderName::from_static("x-trusted-tenant");
    ctx.request_headers_to_set
        .push((tenant.clone(), HeaderValue::from_static("acme")));
    let mut body = Some(Bytes::from_static(br#"{"model":"claude-sonnet-4-5"}"#));
    assert!(
        matches!(
            filter.on_request_body(&mut ctx, &mut body, true).await.unwrap(),
            FilterAction::Continue
        ),
        "the initial mapping must continue before retrying its rewritten body"
    );
    ctx.extra_request_headers
        .push((MODEL_PROVIDER_HEADER.into(), "stale".to_owned()));
    ctx.extra_request_headers
        .push(("x-trusted-mode".into(), "preserved".to_owned()));

    assert!(
        matches!(
            filter.on_request_body(&mut ctx, &mut body, true).await.unwrap(),
            FilterAction::Continue
        ),
        "an unmapped provider target must continue through the existing route"
    );

    assert_grouped_mapping_cleared(&ctx, tenant);
}

#[tokio::test]
async fn mapping_without_a_body_clears_stale_state() {
    let filter = build_filter(vertex_config());
    let request = make_request(Method::POST, "/v1/messages");
    let mut ctx = make_filter_context(&request);
    ctx.set_metadata(MODEL_PROVIDER_CLIENT_MODEL_METADATA, "previous-model");
    ctx.pre_read_mutations.push(TrustedHeaderMutation::Set(
        PROVIDER_HEADER,
        HeaderValue::from_static("stale"),
    ));

    assert!(
        matches!(
            filter.on_request_body(&mut ctx, &mut None, true).await.unwrap(),
            FilterAction::Continue
        ),
        "an absent body must continue while clearing stale state"
    );

    assert_stale_mapping_removed(&ctx);
}

#[tokio::test]
async fn mapping_defers_partial_bodies_until_end_of_stream() {
    let filter = build_filter(vertex_config());
    let request = make_request(Method::POST, "/v1/messages");
    let mut ctx = make_filter_context(&request);
    let original = Bytes::from_static(br#"{"model":"claude-sonnet-4-5"}"#);
    let mut body = Some(original.clone());

    assert!(
        matches!(
            filter.on_request_body(&mut ctx, &mut body, false).await.unwrap(),
            FilterAction::Continue
        ),
        "partial requests must continue until their buffered body is complete"
    );

    assert_eq!(body, Some(original), "a partial body must stay unchanged");
    assert!(
        ctx.request_headers_to_set.is_empty(),
        "a partial body must not set headers"
    );
    assert!(
        ctx.request_headers_to_remove.is_empty(),
        "a partial body must not remove headers"
    );
    assert!(ctx.filter_metadata.is_empty(), "a partial body must not write metadata");
}

#[tokio::test]
#[ignore = "manual latency microbenchmark; run optimized with --ignored --nocapture"]
#[expect(
    clippy::print_stderr,
    reason = "manual benchmark reports timings without a platform-dependent threshold"
)]
async fn mapping_latency_microbenchmark() {
    let filter = build_filter(vertex_config());
    let request = make_request(Method::POST, "/v1/messages");
    for (model, prompt_bytes) in [
        ("claude-sonnet-4-5", 64),
        ("claude-sonnet-4-5", 1_024),
        ("claude-sonnet-4-5", 16_384),
        ("unknown-model", 1_024),
    ] {
        let original = Bytes::from(
            json!({"model":model,"messages":[{"role":"user","content":"x".repeat(prompt_bytes)}]}).to_string(),
        );
        let started = std::time::Instant::now();
        for _iteration in 0..10_000 {
            let mut ctx = make_filter_context(&request);
            // Each invocation owns the body it may replace; Bytes shares the fixture allocation.
            let mut body = Some(original.clone());
            let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
            let (observed_action, _observed_body, _observed_metadata) =
                std::hint::black_box((action, body, ctx.filter_metadata));
            assert!(matches!(observed_action, FilterAction::Continue));
        }
        let nanos_per_request = started.elapsed().as_nanos().checked_div(10_000).unwrap();
        eprintln!(
            "model_to_provider model={model} prompt_bytes={prompt_bytes} samples=10000 ns_per_request={nanos_per_request}"
        );
    }
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

/// Build the configured filter for a unit test.
fn build_filter(yaml: &str) -> Box<dyn HttpFilter> {
    let value: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
    ModelToProviderFilter::from_config(&value).unwrap()
}

/// Example stable-model mapping used by request tests.
fn vertex_config() -> &'static str {
    "models:\n  - model: claude-sonnet-4-5\n    provider: vertex\n    target_model: vertex/claude-sonnet-4-5\n    paths: [/v1/messages, /v1/messages/count_tokens]"
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

/// Assert grouped selector removal while retaining unrelated header writes.
fn assert_grouped_mapping_cleared(ctx: &HttpFilterContext<'_>, tenant: HeaderName) {
    assert_eq!(
        ctx.pending_header_value(&PROVIDER_HEADER).unwrap(),
        praxis_filter::PendingHeaderResult::Removed,
        "a later unmapped decision must revoke earlier grouped selector sets and adds"
    );
    assert!(ctx.get_metadata(MODEL_PROVIDER_CLIENT_MODEL_METADATA).is_none());
    assert_eq!(ctx.request_headers_to_set, [(tenant, HeaderValue::from_static("acme"))]);
    assert_eq!(
        ctx.extra_request_headers,
        [("x-trusted-mode".into(), "preserved".to_owned())]
    );
    assert!(
        ctx.pre_read_mutations.is_empty(),
        "grouped cleanup must preserve grouped mode"
    );
}
