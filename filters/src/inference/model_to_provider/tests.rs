// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Tests for stable model ID resolution and staged cluster selection.

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
        "models: [{model: same, upstream: vertex, target_model: a, paths: [/v1/messages]}, {model: same, upstream: other, target_model: b, paths: [/v1/messages]}]",
        "models: [{model: claude, upstream: '', target_model: a, paths: [/v1/messages]}]",
        "models: [{model: claude, upstream: vertex, target_model: a, paths: []}]",
        "models: [{model: claude, upstream: vertex, target_model: a, paths: [relative/path]}]",
        "models: [{model: claude, upstream: vertex, target_model: a, paths: [/v1/messages, /v1/messages]}]",
        "models: [{model: claude, provider: vertex, target_model: a, paths: [/v1/messages]}]",
    ] {
        let value: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
        assert!(
            ModelToProviderFilter::from_config(&value).is_err(),
            "must reject: {yaml}"
        );
    }
    let value = serde_yaml::from_str(&format!("{}\nfallback_upstream: ''", vertex_config())).unwrap();
    assert!(
        ModelToProviderFilter::from_config(&value).is_err(),
        "an explicit fallback must name a cluster"
    );
}

#[test]
fn declares_every_mapped_and_fallback_cluster() {
    let filter = build_filter(&format!("{}\nfallback_upstream: fallback", vertex_config()));
    assert!(filter.selects_cluster(), "pipeline checks must recognize this selector");
    assert_eq!(
        filter.selected_clusters(),
        ["fallback", "vertex"],
        "validation must see every possible target"
    );
    let same = build_filter(&format!("{}\nfallback_upstream: vertex", vertex_config()));
    assert_eq!(
        same.selected_clusters(),
        ["vertex"],
        "shared targets need only one declaration"
    );
}

#[tokio::test]
async fn mapping_stages_cluster_preserves_public_id_and_rewrites_target() {
    let filter = build_filter(vertex_config());
    let request = make_request(Method::POST, "/v1/messages");
    let mut ctx = context(&request);
    let mut body = Some(Bytes::from_static(br#"{"model":"claude-sonnet-4-5","messages":[]}"#));
    assert_continue(&filter.on_request_body(&mut ctx, &mut body, true).await.unwrap());
    assert!(ctx.cluster.is_none(), "pre-read only stages selection");
    let public_model = ctx.get_metadata(MODEL_PROVIDER_CLIENT_MODEL_METADATA);
    assert_eq!(
        public_model,
        Some("claude-sonnet-4-5"),
        "keep the client model for metering"
    );
    let parsed: Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    assert_eq!(
        parsed["model"], "claude-sonnet-4-5@20250929",
        "use the exact target model"
    );
    assert_continue(&filter.on_request(&mut ctx).await.unwrap());
    assert_eq!(ctx.cluster.as_deref(), Some("vertex"), "publish the mapped cluster");
    assert!(ctx.upstream.is_none(), "the load balancer selects the endpoint");
    assert!(ctx.request_headers_to_set.is_empty(), "routing needs no header");
}

#[tokio::test]
async fn mapping_uses_explicit_fallback_without_changing_unmatched_bodies() {
    let filter = build_filter(&format!("{}\nfallback_upstream: fallback", vertex_config()));
    for (path, model) in [
        ("/v1/messages", "unknown-model"),
        ("/v1/chat/completions", "claude-sonnet-4-5"),
    ] {
        let request = make_request(Method::POST, path);
        let mut ctx = context(&request);
        let original = Bytes::from(json!({"model": model, "messages": []}).to_string());
        let mut body = Some(original.clone());
        assert_continue(&filter.on_request_body(&mut ctx, &mut body, true).await.unwrap());
        assert_continue(&filter.on_request(&mut ctx).await.unwrap());
        assert_eq!(
            ctx.cluster.as_deref(),
            Some("fallback"),
            "unmatched requests use explicit fallback"
        );
        assert_eq!(
            body.as_ref(),
            Some(&original),
            "unmatched bodies retain the client model"
        );
        assert!(
            ctx.get_metadata(MODEL_PROVIDER_CLIENT_MODEL_METADATA).is_none(),
            "no match means no alias metadata"
        );
    }
}

#[tokio::test]
async fn no_match_without_fallback_preserves_existing_selection() {
    let filter = build_filter(vertex_config());
    let request = make_request(Method::POST, "/v1/messages");
    for initial in [None, Some(Arc::from("preselected"))] {
        let mut ctx = context(&request);
        ctx.cluster.clone_from(&initial);
        ctx.metrics_route = Some("original-route".into());
        let mut body = Some(Bytes::from_static(br#"{"model":"unknown"}"#));
        assert_continue(&filter.on_request_body(&mut ctx, &mut body, true).await.unwrap());
        assert_continue(&filter.on_request(&mut ctx).await.unwrap());
        assert_eq!(ctx.cluster, initial, "no match without fallback makes no selection");
        assert_eq!(
            ctx.metrics_route.as_deref(),
            Some("original-route"),
            "an untouched route retains its metadata"
        );
    }
}

#[tokio::test]
async fn mapping_and_fallback_clear_previous_route_overrides() {
    let filter = build_filter(&format!("{}\nfallback_upstream: fallback", vertex_config()));
    let request = make_request(Method::POST, "/v1/messages");
    for model in ["claude-sonnet-4-5", "unknown"] {
        let mut ctx = context(&request);
        ctx.route_retry_policy = Some(Arc::new(serde_yaml::from_str("max_retries: 9").unwrap()));
        ctx.hedge_policy = Some(Arc::new(
            serde_yaml::from_str("initial_requests: 2\nmax_attempts: 2\nbudget_percent: 50").unwrap(),
        ));
        ctx.metrics_route = Some("old-route".into());
        let mut body = Some(Bytes::from(json!({"model": model}).to_string()));
        assert_continue(&filter.on_request_body(&mut ctx, &mut body, true).await.unwrap());
        assert_continue(&filter.on_request(&mut ctx).await.unwrap());
        assert!(
            ctx.route_retry_policy.is_none(),
            "the selected cluster must use its own retry policy"
        );
        assert!(
            ctx.hedge_policy.is_none(),
            "an old route must not impose hedging on this cluster"
        );
        assert!(
            ctx.metrics_route.is_none(),
            "the old router label must not describe this selection"
        );
    }
}

#[tokio::test]
async fn mapping_matches_path_without_query_string() {
    let filter = build_filter(vertex_config());
    let request = make_request(Method::POST, "/v1/messages/count_tokens?client=cli");
    let mut ctx = context(&request);
    let mut body = Some(Bytes::from_static(br#"{"model":"claude-sonnet-4-5"}"#));
    assert_continue(&filter.on_request_body(&mut ctx, &mut body, true).await.unwrap());
    assert_continue(&filter.on_request(&mut ctx).await.unwrap());
    assert_eq!(
        ctx.cluster.as_deref(),
        Some("vertex"),
        "queries must not prevent a configured path match"
    );
}

#[tokio::test]
async fn unmatched_complete_bodies_clear_staged_state() {
    let filter = build_filter(vertex_config());
    for (path, raw) in [
        ("/v1/messages", Some("not JSON")),
        ("/v1/messages", Some("{}")),
        ("/v1/messages", Some("[]")),
        ("/v1/messages", Some("{\"model\":42}")),
        ("/v1/messages", Some("{\"model\":\"unknown\"}")),
        ("/other", Some("{\"model\":\"claude-sonnet-4-5\"}")),
        ("/v1/messages", None),
    ] {
        let request = make_request(Method::POST, path);
        let mut ctx = context(&request);
        ctx.insert_filter_state(ResolvedUpstream(Arc::from("stale")));
        ctx.set_metadata(MODEL_PROVIDER_CLIENT_MODEL_METADATA, "stale-model");
        let mut body = raw.map(|value| Bytes::copy_from_slice(value.as_bytes()));
        assert_continue(&filter.on_request_body(&mut ctx, &mut body, true).await.unwrap());
        assert_continue(&filter.on_request(&mut ctx).await.unwrap());
        assert!(
            ctx.cluster.is_none(),
            "stale state must not select a cluster for an unmatched body"
        );
        assert!(
            ctx.get_metadata(MODEL_PROVIDER_CLIENT_MODEL_METADATA).is_none(),
            "stale aliases must be cleared"
        );
    }
}

#[tokio::test]
async fn mapping_defers_partial_bodies_until_end_of_stream() {
    let filter = build_filter(vertex_config());
    let request = make_request(Method::POST, "/v1/messages");
    let mut ctx = context(&request);
    let original = Bytes::from_static(br#"{"model":"claude-sonnet-4-5"}"#);
    let mut body = Some(original.clone());
    assert_continue(&filter.on_request_body(&mut ctx, &mut body, false).await.unwrap());
    assert_eq!(body, Some(original), "partial bodies must stay unchanged");
    assert!(
        ctx.get_filter_state::<ResolvedUpstream>().is_none(),
        "partial bodies must not stage a selection"
    );
    assert!(
        ctx.filter_metadata.is_empty(),
        "partial bodies must not publish an alias"
    );
}

#[tokio::test]
async fn mapping_preserves_last_model_and_decodes_escaped_names() {
    let filter = build_filter(vertex_config());
    for raw in [
        r#"{"mo\u0064el":"claude-sonnet-4-5"}"#,
        r#"{"model":"claude-sonnet-4-\u0035"}"#,
        r#"{"model":42,"model":"claude-sonnet-4-5"}"#,
        r#"{"model":"unknown","model":"claude-sonnet-4-5"}"#,
    ] {
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = context(&request);
        let mut body = Some(Bytes::copy_from_slice(raw.as_bytes()));
        assert_continue(&filter.on_request_body(&mut ctx, &mut body, true).await.unwrap());
        assert_continue(&filter.on_request(&mut ctx).await.unwrap());
        assert_eq!(ctx.cluster.as_deref(), Some("vertex"), "last model selects: {raw}");
        let public_model = ctx.get_metadata(MODEL_PROVIDER_CLIENT_MODEL_METADATA);
        assert_eq!(
            public_model,
            Some("claude-sonnet-4-5"),
            "preserve decoded public ID: {raw}"
        );
        let value: Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
        assert_eq!(
            value["model"], "claude-sonnet-4-5@20250929",
            "rewrite the effective model: {raw}"
        );
    }
}

#[tokio::test]
async fn invalid_or_unmatched_models_preserve_original_body_and_selection() {
    let filter = build_filter(vertex_config());
    for raw in [
        br#"{"model":"claude-sonnet-4-5","model":42}"#.as_slice(),
        br#"{"model":"claude-sonnet-4-5","model":"unknown"}"#,
        br#"{"model":"claude-sonnet-4-5"} trailing"#,
        br#"{"model":"claude-sonnet-4-5","ignored":1e999}"#,
        b"{\"model\":\"claude-sonnet-4-5\",\"ignored\":\"\xff\"}",
        br#"{"model":null}"#,
        br#"["claude-sonnet-4-5"]"#,
    ] {
        assert_passthrough(filter.as_ref(), raw).await;
    }
    let nested = format!(
        "{{\"model\":\"claude-sonnet-4-5\",\"ignored\":{}0{}}}",
        "[".repeat(128),
        "]".repeat(128)
    );
    assert_passthrough(filter.as_ref(), nested.as_bytes()).await;
}

#[tokio::test]
async fn unchanged_target_preserves_bytes_and_still_validates_entire_body() {
    let filter = build_filter(&vertex_config().replace("claude-sonnet-4-5@20250929", "claude-sonnet-4-5"));
    let original = Bytes::from_static(br#"{ "model": "unknown", "model": "claude-sonnet-4-5", "messages": [] }"#);
    let mut body = Some(original.clone());
    let request = make_request(Method::POST, "/v1/messages");
    let mut ctx = context(&request);
    assert_continue(&filter.on_request_body(&mut ctx, &mut body, true).await.unwrap());
    assert_continue(&filter.on_request(&mut ctx).await.unwrap());
    assert_eq!(
        ctx.cluster.as_deref(),
        Some("vertex"),
        "an unchanged target still selects its cluster"
    );
    assert_eq!(
        body,
        Some(original),
        "no rewrite must preserve whitespace and duplicate members"
    );
    assert_eq!(
        ctx.get_metadata(MODEL_PROVIDER_CLIENT_MODEL_METADATA),
        Some("claude-sonnet-4-5"),
        "an unchanged target retains public identity metadata"
    );
    assert_passthrough(filter.as_ref(), br#"{"model":"claude-sonnet-4-5","ignored":1e999}"#).await;
}

#[test]
fn model_preflight_borrows_plain_names_and_decodes_only_selected_strings() {
    assert!(
        matches!(
            model::peek(br#"{"model":"plain","messages":[{"content":"long"}]}"#),
            Some(std::borrow::Cow::Borrowed("plain"))
        ),
        "plain model identities must borrow the input"
    );
    assert_eq!(
        model::peek(br#"{"model":42,"mo\u0064el":"p\u006cain"}"#).as_deref(),
        Some("plain"),
        "escaped keys and the final escaped model must decode"
    );
    assert!(
        model::peek(br#"{"model":"plain","model":false}"#).is_none(),
        "the final non-string model must replace an earlier string"
    );
    assert!(
        model::peek(br#"{"model":"plain"} {}"#).is_none(),
        "trailing JSON must invalidate a candidate"
    );
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
        ("unknown-model", 16_384),
    ] {
        let original = Bytes::from(
            json!({"model":model,"messages":[{"role":"user","content":"x".repeat(prompt_bytes)}]}).to_string(),
        );
        let started = std::time::Instant::now();
        for _iteration in 0..10_000 {
            let mut ctx = context(&request);
            // Each invocation owns the body it may replace; Bytes shares the fixture allocation.
            let mut body = Some(original.clone());
            assert_continue(&filter.on_request_body(&mut ctx, &mut body, true).await.unwrap());
            let action = filter.on_request(&mut ctx).await.unwrap();
            let (observed_action, _observed_body, _observed_metadata, _observed_cluster) =
                std::hint::black_box((action, body, ctx.filter_metadata, ctx.cluster));
            assert_continue(&observed_action);
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

/// Assert that resolution permits the rest of the pipeline to execute.
fn assert_continue(action: &FilterAction) {
    assert!(
        matches!(action, FilterAction::Continue),
        "resolution must continue processing"
    );
}

/// Ensure preflight failures neither mutate the body nor publish routing metadata.
async fn assert_passthrough(filter: &dyn HttpFilter, raw: &[u8]) {
    let request = make_request(Method::POST, "/v1/messages");
    let mut ctx = context(&request);
    ctx.cluster = Some(Arc::from("existing"));
    let original = Bytes::copy_from_slice(raw);
    let mut body = Some(original.clone());
    assert_continue(&filter.on_request_body(&mut ctx, &mut body, true).await.unwrap());
    assert_continue(&filter.on_request(&mut ctx).await.unwrap());
    assert_eq!(body, Some(original), "a rejected candidate must preserve its bytes");
    assert_eq!(
        ctx.cluster.as_deref(),
        Some("existing"),
        "a rejected candidate must preserve selection"
    );
    assert!(
        ctx.get_metadata(MODEL_PROVIDER_CLIENT_MODEL_METADATA).is_none(),
        "a rejected candidate must not publish a public model"
    );
}

/// Build the configured filter for a unit test.
fn build_filter(yaml: &str) -> Box<dyn HttpFilter> {
    let value: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
    ModelToProviderFilter::from_config(&value).unwrap()
}

/// Example stable-model mapping used by request tests.
fn vertex_config() -> &'static str {
    "models: [{model: claude-sonnet-4-5, upstream: vertex, target_model: 'claude-sonnet-4-5@20250929', paths: [/v1/messages, /v1/messages/count_tokens]}]"
}

/// Give direct hook calls the same per-filter state slot the pipeline provides.
fn context(request: &praxis_filter::Request) -> HttpFilterContext<'_> {
    let mut ctx = make_filter_context(request);
    ctx.current_filter_id = Some(0);
    ctx
}
