// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Unit tests for the Anthropic Messages request processor.

use bytes::Bytes;
use praxis_filter::{
    BodyAccess, BodyMode, ErrorResponseFormatterHandle, FilterAction, HttpFilter, HttpFilterContext, Request,
};
use serde_json::json;

use super::*;
use crate::test_utils::{make_filter_context, make_request};

/// Build the filter from YAML, defaulting to an empty mapping.
fn filter(yaml: &str) -> Box<dyn HttpFilter> {
    let value: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
    AnthropicMessagesRequestFilter::from_config(&value).unwrap()
}

/// Build a filter with default configuration.
fn default_filter() -> Box<dyn HttpFilter> {
    filter("{}")
}

/// Build a `POST /v1/messages` request.
fn create_request() -> Request {
    make_request(http::Method::POST, "/v1/messages")
}

/// A well-formed Anthropic create-message body.
fn anthropic_body() -> serde_json::Value {
    json!({
        "model": "claude-opus-4-8",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hello"}],
    })
}

/// Drive one body through the filter and return the action.
async fn run_with<'a>(
    filter: &dyn HttpFilter,
    request: &'a Request,
    body: Option<&[u8]>,
) -> (FilterAction, HttpFilterContext<'a>) {
    let mut ctx = make_filter_context(request);
    let mut bytes = body.map(Bytes::copy_from_slice);
    let action = filter.on_request_body(&mut ctx, &mut bytes, true).await.unwrap();
    (action, ctx)
}

#[test]
fn declares_read_only_buffered_body_access() {
    let filter = default_filter();
    assert_eq!(filter.request_body_access(), BodyAccess::ReadOnly);
    assert!(matches!(filter.request_body_mode(), BodyMode::StreamBuffer { .. }));
}

#[tokio::test]
async fn processes_a_create_message_body_once() {
    let filter = default_filter();
    let request = create_request();
    let body = serde_json::to_vec(&anthropic_body()).unwrap();
    let (action, ctx) = run_with(filter.as_ref(), &request, Some(&body)).await;

    assert!(matches!(action, FilterAction::Release));

    let state = ctx
        .extensions
        .get::<AnthropicMessagesState>()
        .expect("a matched create-message must publish state");
    assert_eq!(state.model.as_deref(), Some("claude-opus-4-8"));
    assert_eq!(state.max_tokens, Some(1024));
    assert_eq!(state.stream, None);
    assert!(!state.has_tools);
}

#[tokio::test]
async fn forwarded_bytes_are_left_untouched() {
    let filter = default_filter();
    let request = create_request();
    let body = serde_json::to_vec(&anthropic_body()).unwrap();

    let mut ctx = make_filter_context(&request);
    let mut bytes = Some(Bytes::copy_from_slice(&body));
    drop(filter.on_request_body(&mut ctx, &mut bytes, true).await.unwrap());

    assert_eq!(
        bytes.as_deref(),
        Some(body.as_slice()),
        "this filter reads the body and must not rewrite it"
    );
}

#[tokio::test]
async fn publishes_routing_metadata_and_filter_results() {
    let filter = default_filter();
    let request = create_request();
    let body = serde_json::to_vec(&json!({
        "model": "claude-opus-4-8",
        "max_tokens": 64,
        "stream": true,
        "tools": [{"name": "lookup"}],
        "messages": [{"role": "user", "content": "Hi"}],
    }))
    .unwrap();
    let (_, ctx) = run_with(filter.as_ref(), &request, Some(&body)).await;

    assert_eq!(
        ctx.get_metadata("anthropic_messages_request.model"),
        Some("claude-opus-4-8")
    );
    assert_eq!(ctx.get_metadata("anthropic_messages_request.stream"), Some("true"));
    assert_eq!(ctx.get_metadata("anthropic_messages_request.max_tokens"), Some("64"));
    assert_eq!(ctx.get_metadata("anthropic_messages_request.has_tools"), Some("true"));

    let results = ctx
        .filter_results
        .get("anthropic_messages_request")
        .expect("results are what on_result branch conditions read");
    assert_eq!(results.get("model"), Some("claude-opus-4-8"));
    assert_eq!(results.get("stream"), Some("true"));
}

#[tokio::test]
async fn promotes_the_configured_routing_headers() {
    let filter = default_filter();
    let request = create_request();
    let body = serde_json::to_vec(&anthropic_body()).unwrap();
    let (_, ctx) = run_with(filter.as_ref(), &request, Some(&body)).await;

    let set = ctx
        .request_headers_to_set
        .iter()
        .map(|(name, value)| (name.as_str(), value.to_str().unwrap()))
        .collect::<Vec<_>>();
    assert!(set.contains(&("x-praxis-ai-model", "claude-opus-4-8")), "{set:?}");
}

#[tokio::test]
async fn installs_the_anthropic_error_formatter() {
    let filter = default_filter();
    let request = create_request();
    let body = serde_json::to_vec(&anthropic_body()).unwrap();
    let (_, ctx) = run_with(filter.as_ref(), &request, Some(&body)).await;

    assert!(
        ctx.extensions.get::<ErrorResponseFormatterHandle>().is_some(),
        "a matched Anthropic operation keeps the Anthropic error shape"
    );
}

// -----------------------------------------------------------------------------
// Envelope handling
// -----------------------------------------------------------------------------

/// `on_invalid: reject` refuses every envelope violation.
///
/// This is the policy the separate validator applied unconditionally.
#[tokio::test]
async fn a_rejecting_chain_refuses_every_bad_envelope() {
    let filter = filter("on_invalid: reject\n");
    let request = create_request();

    for body in [
        None,
        Some(b"".as_slice()),
        Some(b"{not json".as_slice()),
        Some(b"[1,2,3]".as_slice()),
        Some(b"\"text\"".as_slice()),
        Some(b"42".as_slice()),
    ] {
        let (action, ctx) = run_with(filter.as_ref(), &request, body).await;
        assert!(
            matches!(action, FilterAction::Reject(_)),
            "body {body:?} must be refused"
        );
        assert!(
            ctx.extensions.get::<AnthropicMessagesState>().is_none(),
            "a rejected request must not leave state behind"
        );
    }
}

/// `on_invalid: continue` forwards a bad envelope to the backend.
///
/// This is the policy the classifier applied, and chains that ran it without
/// the validator depend on it: a malformed body reaches the backend rather than
/// being refused at the gateway.
#[tokio::test]
async fn a_continuing_chain_forwards_a_bad_envelope() {
    let filter = filter("on_invalid: continue\n");
    let request = create_request();

    for body in [
        None,
        Some(b"".as_slice()),
        Some(b"{not json".as_slice()),
        Some(b"[1,2,3]".as_slice()),
    ] {
        let (action, ctx) = run_with(filter.as_ref(), &request, body).await;
        assert!(
            matches!(action, FilterAction::Release),
            "body {body:?} must be forwarded"
        );
        assert!(
            ctx.extensions.get::<AnthropicMessagesState>().is_none(),
            "there is no canonical state to publish for an unusable body"
        );
        assert!(
            ctx.extensions.get::<ErrorResponseFormatterHandle>().is_some(),
            "the head identified Anthropic traffic, so the error shape still applies"
        );
    }
}

/// `continue` is the default, matching the classifier it replaced.
#[tokio::test]
async fn the_default_policy_forwards_a_bad_envelope() {
    let filter = default_filter();
    let request = create_request();
    let (action, _) = run_with(filter.as_ref(), &request, Some(b"{not json")).await;
    assert!(matches!(action, FilterAction::Release));
}

// -----------------------------------------------------------------------------
// The head decides the operation
// -----------------------------------------------------------------------------

/// A Chat Completions-shaped body on `/v1/messages` is still create-message.
///
/// The old classifier inferred the protocol from body shape with an
/// `anthropic-version` tiebreak. The registry removes that guesswork, so this
/// asserts the body cannot change which operation the head resolved to.
#[tokio::test]
async fn a_chat_shaped_body_is_still_the_create_message_operation() {
    let filter = filter("on_invalid: continue\n");
    let request = create_request();
    let body = serde_json::to_vec(&json!({
        "model": "gpt-4.1",
        "messages": [{"role": "user", "content": "Hi"}],
    }))
    .unwrap();
    let (action, ctx) = run_with(filter.as_ref(), &request, Some(&body)).await;

    assert!(matches!(action, FilterAction::Release));
    assert!(
        ctx.extensions.get::<AnthropicMessagesState>().is_some(),
        "the head matched create-message, so the payload shape does not veto it"
    );
}

/// No `anthropic-version` header is needed, and its absence changes nothing.
#[tokio::test]
async fn the_anthropic_version_header_does_not_decide_the_operation() {
    let filter = default_filter();
    let body = serde_json::to_vec(&anthropic_body()).unwrap();

    let without = create_request();
    let (plain_action, plain_ctx) = run_with(filter.as_ref(), &without, Some(&body)).await;

    let mut with = create_request();
    with.headers.insert("anthropic-version", "2023-06-01".parse().unwrap());
    let (header_action, header_ctx) = run_with(filter.as_ref(), &with, Some(&body)).await;

    assert!(matches!(plain_action, FilterAction::Release));
    assert!(matches!(header_action, FilterAction::Release));
    assert_eq!(
        plain_ctx.get_metadata("anthropic_messages_request.model"),
        header_ctx.get_metadata("anthropic_messages_request.model"),
        "the version header is contract validation, not operation identity"
    );
}

/// Every Anthropic Messages operation keeps the Anthropic error shape.
///
/// Token counting and the batch family carry no body for this filter to
/// process, but they are supported endpoints and a shipped chain may hold no
/// other formatter. Without this an upstream failure would answer an Anthropic
/// client with RFC 9457 problem details.
#[tokio::test]
async fn every_messages_operation_installs_the_anthropic_error_formatter() {
    let filter = default_filter();
    for (method, path) in [
        ("POST", "/v1/messages"),
        ("POST", "/v1/messages/count_tokens"),
        ("GET", "/v1/messages/batches"),
        ("GET", "/v1/messages/batches/msgbatch_1"),
        ("POST", "/v1/messages/batches/msgbatch_1/cancel"),
    ] {
        let request = make_request(http::Method::from_bytes(method.as_bytes()).unwrap(), path);
        let (_, ctx) = run_with(filter.as_ref(), &request, Some(b"{}")).await;
        assert!(
            ctx.extensions.get::<ErrorResponseFormatterHandle>().is_some(),
            "{method} {path} should keep the Anthropic error shape"
        );
    }
}

/// A path outside the Messages surface installs no formatter.
#[tokio::test]
async fn a_non_messages_path_installs_no_formatter() {
    let filter = default_filter();
    let request = make_request(http::Method::POST, "/v1/chat/completions");
    let (_, ctx) = run_with(filter.as_ref(), &request, Some(b"{}")).await;
    assert!(
        ctx.extensions.get::<ErrorResponseFormatterHandle>().is_none(),
        "this filter owns only the Anthropic Messages surface"
    );
}

/// Other Anthropic Messages operations are released without body processing.
#[tokio::test]
async fn non_create_operations_are_released_without_processing() {
    let filter = default_filter();
    for (method, path) in [
        ("POST", "/v1/messages/count_tokens"),
        ("GET", "/v1/messages/batches"),
        ("POST", "/v1/messages/batches"),
        ("GET", "/v1/messages/batches/msgbatch_1"),
    ] {
        let request = make_request(http::Method::from_bytes(method.as_bytes()).unwrap(), path);
        let (action, ctx) = run_with(filter.as_ref(), &request, Some(b"{}")).await;

        assert!(
            matches!(action, FilterAction::Release),
            "{method} {path} must pass through"
        );
        assert!(
            ctx.extensions.get::<AnthropicMessagesState>().is_none(),
            "{method} {path} is not the create-message operation"
        );
    }
}

/// An Anthropic-shaped body on another path is not this operation.
#[tokio::test]
async fn an_anthropic_body_on_another_path_is_not_processed() {
    let filter = default_filter();
    let request = make_request(http::Method::POST, "/v1/chat/completions");
    let body = serde_json::to_vec(&anthropic_body()).unwrap();
    let (action, ctx) = run_with(filter.as_ref(), &request, Some(&body)).await;

    assert!(matches!(action, FilterAction::Release));
    assert!(
        ctx.extensions.get::<AnthropicMessagesState>().is_none(),
        "body shape must not pull another path into this processor"
    );
}

/// A bodyless operation is released without the required-body rejection.
#[tokio::test]
async fn a_bodyless_operation_is_not_rejected_for_a_missing_body() {
    let filter = default_filter();
    let request = make_request(http::Method::GET, "/v1/messages/batches");
    let (action, _) = run_with(filter.as_ref(), &request, None).await;
    assert!(
        matches!(action, FilterAction::Release),
        "the required-body rule applies only to create-message"
    );
}

/// The endpoint is the authority on protocol, not the payload's field mix.
///
/// An ordinary Anthropic body carries `messages` and no Responses or
/// Conversations markers, which the shared body classifier reads as Chat
/// Completions. The head already said create-message, so the published format
/// is Anthropic Messages regardless.
#[tokio::test]
async fn the_endpoint_decides_the_published_format() {
    let filter = filter("on_invalid: reject\n");
    let request = create_request();
    let body = serde_json::to_vec(&anthropic_body()).unwrap();
    let (action, ctx) = run_with(filter.as_ref(), &request, Some(&body)).await;

    assert!(
        matches!(action, FilterAction::Release),
        "an ordinary Anthropic body must not be refused by a rejecting chain"
    );
    assert_eq!(
        ctx.get_metadata("anthropic_messages_request.format"),
        Some("anthropic_messages")
    );
}
