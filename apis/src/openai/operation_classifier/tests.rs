// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Unit tests for the `openai_operation` classifier.

#![expect(clippy::unwrap_used, reason = "tests")]

use praxis_filter::{BodyAccess, BodyMode, Request};

use super::*;
use crate::test_utils::{make_filter_context, make_request};

/// Build the filter from YAML, defaulting to an empty mapping.
fn filter(yaml: &str) -> Box<dyn HttpFilter> {
    let value: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
    OpenaiOperationFilter::from_config(&value).unwrap()
}

/// Build a filter with default configuration.
fn default_filter() -> Box<dyn HttpFilter> {
    filter("{}")
}

/// Build a request for one method and path.
fn req(method: &str, path: &str) -> Request {
    make_request(http::Method::from_bytes(method.as_bytes()).unwrap(), path)
}

/// `WebSocket` opening handshake headers.
fn websocket_headers() -> http::HeaderMap {
    let mut headers = http::HeaderMap::new();
    headers.insert(http::header::CONNECTION, "Upgrade".parse().unwrap());
    headers.insert(http::header::UPGRADE, "websocket".parse().unwrap());
    headers
}

#[test]
fn declares_no_request_body_access_or_buffering() {
    let filter = default_filter();
    assert_eq!(filter.request_body_access(), BodyAccess::None);
    assert!(
        matches!(filter.request_body_mode(), BodyMode::Stream),
        "the classifier must not request buffering"
    );
}

#[tokio::test]
async fn classifies_a_conversations_operation() {
    let filter = default_filter();
    let request = req("GET", "/v1/conversations/conv_123");
    let mut ctx = make_filter_context(&request);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let matched = ctx.extensions.get::<OpenAiOperationMatch>().copied().unwrap();
    assert_eq!(matched.family, OpenAiApiFamily::Conversations);
    assert_eq!(matched.operation_id, "getConversation");
    assert_eq!(matched.transport, OpenAiTransport::Http);

    assert_eq!(
        ctx.filter_metadata.get("openai_operation.family").map(String::as_str),
        Some("conversations")
    );
    assert_eq!(
        ctx.filter_metadata
            .get("openai_operation.operation_id")
            .map(String::as_str),
        Some("getConversation")
    );
}

#[tokio::test]
async fn classifies_a_responses_operation() {
    let filter = default_filter();
    let request = req("POST", "/v1/responses");
    let mut ctx = make_filter_context(&request);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let matched = ctx.extensions.get::<OpenAiOperationMatch>().copied().unwrap();
    assert_eq!(matched.family, OpenAiApiFamily::Responses);
    assert_eq!(matched.operation_id, "createResponse");
}

#[tokio::test]
async fn publishes_proxy_owned_routing_headers() {
    let filter = default_filter();
    let request = req("POST", "/v1/responses");
    let mut ctx = make_filter_context(&request);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let set: Vec<(String, String)> = ctx
        .request_headers_to_set
        .iter()
        .map(|(name, value)| (name.as_str().to_owned(), value.to_str().unwrap().to_owned()))
        .collect();

    assert!(set.contains(&("x-praxis-ai-family".to_owned(), "responses".to_owned())));
    assert!(set.contains(&("x-praxis-ai-operation".to_owned(), "createResponse".to_owned())));
}

#[tokio::test]
async fn client_supplied_headers_cannot_spoof_a_matched_operation() {
    let filter = default_filter();
    let mut request = req("POST", "/v1/responses");
    request.headers.insert("x-praxis-ai-family", "files".parse().unwrap());
    request
        .headers
        .insert("x-praxis-ai-operation", "createFile".parse().unwrap());
    let mut ctx = make_filter_context(&request);

    drop(filter.on_request(&mut ctx).await.unwrap());

    // set (overwrite) semantics, so the forged values cannot survive alongside.
    let family = ctx
        .request_headers_to_set
        .iter()
        .find(|(name, _)| name.as_str() == "x-praxis-ai-family")
        .map(|(_, value)| value.to_str().unwrap().to_owned());
    assert_eq!(family.as_deref(), Some("responses"));
}

#[tokio::test]
async fn client_supplied_headers_are_stripped_when_nothing_matches() {
    let filter = default_filter();
    let mut request = req("GET", "/v1/unknown");
    request
        .headers
        .insert("x-praxis-ai-family", "responses".parse().unwrap());
    let mut ctx = make_filter_context(&request);

    drop(filter.on_request(&mut ctx).await.unwrap());

    assert!(ctx.extensions.get::<OpenAiOperationMatch>().is_none());
    let removed: Vec<&str> = ctx
        .request_headers_to_remove
        .iter()
        .map(http::HeaderName::as_str)
        .collect();
    assert!(removed.contains(&"x-praxis-ai-family"));
    assert!(removed.contains(&"x-praxis-ai-operation"));
    assert!(ctx.request_headers_to_set.is_empty());
}

#[tokio::test]
async fn websocket_handshake_selects_the_websocket_operation() {
    let filter = default_filter();
    let mut request = req("GET", "/v1/responses");
    request.headers = websocket_headers();
    let mut ctx = make_filter_context(&request);

    drop(filter.on_request(&mut ctx).await.unwrap());

    let matched = ctx.extensions.get::<OpenAiOperationMatch>().copied().unwrap();
    assert_eq!(matched.transport, OpenAiTransport::WebSocket);
    assert_eq!(matched.operation_id, "praxis_createResponseWebSocket");
}

#[tokio::test]
async fn plain_get_on_the_responses_collection_does_not_match() {
    let filter = default_filter();
    let request = req("GET", "/v1/responses");
    let mut ctx = make_filter_context(&request);
    drop(filter.on_request(&mut ctx).await.unwrap());
    assert!(
        ctx.extensions.get::<OpenAiOperationMatch>().is_none(),
        "a GET without upgrade headers is not the websocket operation"
    );
}

#[tokio::test]
async fn static_endpoints_are_not_consumed_as_identifiers() {
    let filter = default_filter();
    let request = req("POST", "/v1/responses/input_tokens");
    let mut ctx = make_filter_context(&request);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let matched = ctx.extensions.get::<OpenAiOperationMatch>().copied().unwrap();
    assert_eq!(matched.operation_id, "Getinputtokencounts");
}

#[tokio::test]
async fn unsupported_methods_publish_no_operation() {
    for (method, path) in [
        ("PUT", "/v1/responses"),
        ("PATCH", "/v1/conversations/conv_1"),
        ("DELETE", "/v1/responses"),
    ] {
        let filter = default_filter();
        let request = req(method, path);
        let mut ctx = make_filter_context(&request);
        drop(filter.on_request(&mut ctx).await.unwrap());
        assert!(
            ctx.extensions.get::<OpenAiOperationMatch>().is_none(),
            "{method} {path} must not classify"
        );
    }
}

#[tokio::test]
async fn configured_header_names_are_honored() {
    let filter = filter("\nheaders:\n  family: x-family\n  operation: x-operation\n");
    let request = req("POST", "/v1/responses");
    let mut ctx = make_filter_context(&request);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let names: Vec<&str> = ctx
        .request_headers_to_set
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();
    assert!(names.contains(&"x-family"));
    assert!(names.contains(&"x-operation"));
}

#[tokio::test]
async fn headers_can_be_disabled_while_metadata_still_publishes() {
    let filter = filter("\nheaders:\n  family: ~\n  operation: ~\n");
    let request = req("POST", "/v1/responses");
    let mut ctx = make_filter_context(&request);
    drop(filter.on_request(&mut ctx).await.unwrap());

    assert!(ctx.request_headers_to_set.is_empty());
    assert_eq!(
        ctx.filter_metadata.get("openai_operation.family").map(String::as_str),
        Some("responses")
    );
}

#[test]
fn invalid_header_name_is_rejected_at_startup() {
    let value: serde_yaml::Value = serde_yaml::from_str("headers:\n  family: \"bad header\"\n").unwrap();
    assert!(OpenaiOperationFilter::from_config(&value).is_err());
}

#[test]
fn unknown_configuration_fields_are_rejected() {
    let value: serde_yaml::Value = serde_yaml::from_str("nonsense: true\n").unwrap();
    assert!(OpenaiOperationFilter::from_config(&value).is_err());
}

#[test]
fn transport_detection_follows_the_opening_handshake() {
    assert_eq!(request_transport(&websocket_headers()), OpenAiTransport::WebSocket);
    assert_eq!(request_transport(&http::HeaderMap::new()), OpenAiTransport::Http);

    // Connection is a token list.
    let mut list = http::HeaderMap::new();
    list.insert(http::header::CONNECTION, "keep-alive, Upgrade".parse().unwrap());
    list.insert(http::header::UPGRADE, "websocket".parse().unwrap());
    assert_eq!(request_transport(&list), OpenAiTransport::WebSocket);

    // Upgrade without Connection: upgrade is not a handshake.
    let mut partial = http::HeaderMap::new();
    partial.insert(http::header::UPGRADE, "websocket".parse().unwrap());
    assert_eq!(request_transport(&partial), OpenAiTransport::Http);

    // Several nominated protocols are not treated as a websocket handshake.
    let mut multi = http::HeaderMap::new();
    multi.insert(http::header::CONNECTION, "Upgrade".parse().unwrap());
    multi.append(http::header::UPGRADE, "websocket".parse().unwrap());
    multi.append(http::header::UPGRADE, "h2c".parse().unwrap());
    assert_eq!(request_transport(&multi), OpenAiTransport::Http);
}
