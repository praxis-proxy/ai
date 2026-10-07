// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Unit tests for the `openai_chat_completions_format` filter.

use bytes::Bytes;
use http::Method;
use praxis_filter::{
    FilterAction, HttpFilter, HttpFilterContext, Request,
    body::{BodyAccess, BodyMode},
};

use super::*;

// -----------------------------------------------------------------------------
// Config Parsing
// -----------------------------------------------------------------------------

#[test]
fn default_config_parses() {
    let filter = make_filter("{}");
    assert_eq!(
        filter.name(),
        "openai_chat_completions_format",
        "filter name should match"
    );
}

#[test]
fn full_config_parses() {
    let filter = make_filter("max_body_bytes: 65536");
    assert_eq!(filter.name(), "openai_chat_completions_format");
}

#[test]
fn zero_max_body_bytes_rejected() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("max_body_bytes: 0").unwrap();
    assert!(
        OpenaiChatCompletionsFormatFilter::from_config(&yaml).is_err(),
        "zero max_body_bytes should be rejected"
    );
}

#[test]
fn unknown_fields_rejected() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("bogus: true").unwrap();
    assert!(
        OpenaiChatCompletionsFormatFilter::from_config(&yaml).is_err(),
        "unknown fields should be rejected"
    );
}

// -----------------------------------------------------------------------------
// Body Access Declarations
// -----------------------------------------------------------------------------

#[test]
fn declares_read_only_buffering() {
    let filter = make_filter("{}");
    assert_eq!(filter.request_body_access(), BodyAccess::ReadOnly);
    assert!(
        matches!(
            filter.request_body_mode(),
            BodyMode::StreamBuffer { max_bytes: Some(_) }
        ),
        "filter should buffer the request body"
    );
}

// -----------------------------------------------------------------------------
// Create Request: Model Fact Production
// -----------------------------------------------------------------------------

#[tokio::test]
async fn create_request_publishes_model_fact() {
    let (ctx, action) = run_filter(
        Method::POST,
        "/v1/chat/completions",
        r#"{"model":"gpt-4","messages":[{"role":"user","content":"Hi"}],"stream":true}"#,
    )
    .await;

    assert!(matches!(action, FilterAction::Release), "create should release");

    assert_eq!(
        ctx.filter_metadata
            .get("openai_chat_completions_format.format")
            .map(String::as_str),
        Some("openai_chat_completions"),
        "format fact comes from the authoritative head"
    );
    assert_eq!(
        ctx.filter_metadata
            .get("openai_chat_completions_format.model")
            .map(String::as_str),
        Some("gpt-4"),
        "model fact comes from the buffered body"
    );
    assert_eq!(
        ctx.filter_metadata
            .get("openai_chat_completions_format.stream")
            .map(String::as_str),
        Some("true"),
        "stream flag comes from the buffered body"
    );

    let results = ctx
        .filter_results
        .get("openai_chat_completions_format")
        .expect("filter results present");
    assert_eq!(results.get("format"), Some("openai_chat_completions"));
    assert_eq!(results.get("model"), Some("gpt-4"));
    assert_eq!(results.get("stream"), Some("true"));
}

#[tokio::test]
async fn create_request_without_model_publishes_format_only() {
    let (ctx, action) = run_filter(
        Method::POST,
        "/v1/chat/completions",
        r#"{"messages":[{"role":"user","content":"Hi"}]}"#,
    )
    .await;

    assert!(matches!(action, FilterAction::Release));
    assert_eq!(
        ctx.filter_metadata
            .get("openai_chat_completions_format.format")
            .map(String::as_str),
        Some("openai_chat_completions"),
    );
    assert!(
        !ctx.filter_metadata.contains_key("openai_chat_completions_format.model"),
        "no model field means no model fact"
    );
}

// -----------------------------------------------------------------------------
// Non-Create / Unmatched: No Fact
// -----------------------------------------------------------------------------

#[tokio::test]
async fn list_operation_publishes_no_fact() {
    // GET /v1/chat/completions matches listChatCompletions, not the create op.
    let (ctx, action) = run_filter(Method::GET, "/v1/chat/completions", "").await;

    assert!(matches!(action, FilterAction::Release));
    assert!(
        !ctx.filter_metadata
            .contains_key("openai_chat_completions_format.format"),
        "non-create operations produce no model fact"
    );
    assert!(
        !ctx.filter_results.contains_key("openai_chat_completions_format"),
        "non-create operations write no filter results"
    );
}

#[tokio::test]
async fn responses_create_is_not_a_chat_completion() {
    // POST /v1/responses matches createResponse under the Responses protocol; the
    // Chat producer must leave it alone so the two families do not cross-label.
    let (ctx, action) = run_filter(Method::POST, "/v1/responses", r#"{"model":"gpt-4","input":"hi"}"#).await;

    assert!(matches!(action, FilterAction::Release));
    assert!(
        !ctx.filter_metadata
            .contains_key("openai_chat_completions_format.format"),
        "a Responses create request must not get a Chat Completions fact"
    );
}

#[tokio::test]
async fn missing_operation_match_publishes_no_fact() {
    // No ai_operation ran, so no AiOperationMatch is in extensions.
    let filter = make_filter("{}");
    let req = crate::test_utils::make_request(Method::POST, "/v1/chat/completions");
    let req: &'static Request = Box::leak(Box::new(req));
    let mut ctx = crate::test_utils::make_filter_context(req);

    let mut body = Some(Bytes::from(
        r#"{"model":"gpt-4","messages":[{"role":"user","content":"Hi"}]}"#,
    ));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(matches!(action, FilterAction::Release));
    assert!(
        !ctx.filter_metadata
            .contains_key("openai_chat_completions_format.format"),
        "without an operation match the producer stays silent"
    );
}

// -----------------------------------------------------------------------------
// Streaming / Partial Body
// -----------------------------------------------------------------------------

#[tokio::test]
async fn partial_body_before_eos_continues() {
    let filter = make_filter("{}");
    let req = crate::test_utils::make_request(Method::POST, "/v1/chat/completions");
    let req: &'static Request = Box::leak(Box::new(req));
    let mut ctx = crate::test_utils::make_filter_context(req);
    stage_operation_match(&mut ctx).await;

    let mut body = Some(Bytes::from(r#"{"model":"gpt-4","mess"#));
    let action = filter.on_request_body(&mut ctx, &mut body, false).await.unwrap();

    assert!(
        matches!(action, FilterAction::Continue),
        "a partial body before EOS should continue buffering"
    );
    assert!(
        !ctx.filter_metadata
            .contains_key("openai_chat_completions_format.format"),
        "no fact should be produced before EOS"
    );
}

// -----------------------------------------------------------------------------
// Unsafe Model Values
// -----------------------------------------------------------------------------

#[tokio::test]
async fn oversized_model_not_promoted() {
    let long_model = "x".repeat(300);
    let body = format!(r#"{{"model":"{long_model}","messages":[{{"role":"user","content":"Hi"}}]}}"#);
    let (ctx, _) = run_filter(Method::POST, "/v1/chat/completions", &body).await;

    assert!(
        !ctx.filter_metadata.contains_key("openai_chat_completions_format.model"),
        "oversized model must not be promoted to metadata"
    );
    let results = ctx
        .filter_results
        .get("openai_chat_completions_format")
        .expect("filter results present");
    assert!(results.get("model").is_none(), "oversized model must not be in results");
}

#[tokio::test]
async fn control_char_model_not_promoted() {
    let (ctx, _) = run_filter(
        Method::POST,
        "/v1/chat/completions",
        "{\"model\":\"bad\\nmodel\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}",
    )
    .await;

    assert!(
        !ctx.filter_metadata.contains_key("openai_chat_completions_format.model"),
        "a model with control characters must not be promoted"
    );
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

/// Publish the operation match `ai_operation` produces for this request's head.
///
/// Runs the real classifier so the test exercises the production handoff rather
/// than a hand-built extension.
async fn stage_operation_match(ctx: &mut HttpFilterContext<'_>) {
    let classifier =
        crate::operation_classifier::AiOperationFilter::from_config(&serde_yaml::from_str("{}").unwrap()).unwrap();
    drop(classifier.on_request(ctx).await.unwrap());
}

/// Stage the operation match, run the filter body at EOS, and return the context
/// and action.
async fn run_filter(method: Method, path: &str, body_str: &str) -> (HttpFilterContext<'static>, FilterAction) {
    let filter = make_filter("{}");
    let req = crate::test_utils::make_request(method, path);
    let req: &'static Request = Box::leak(Box::new(req));
    let mut ctx = crate::test_utils::make_filter_context(req);
    stage_operation_match(&mut ctx).await;

    let mut body = Some(Bytes::from(body_str.to_owned()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    (ctx, action)
}

/// Build a filter from a YAML snippet.
fn make_filter(yaml_str: &str) -> Box<dyn HttpFilter> {
    let yaml: serde_yaml::Value = serde_yaml::from_str(yaml_str).unwrap();
    OpenaiChatCompletionsFormatFilter::from_config(&yaml).unwrap()
}
