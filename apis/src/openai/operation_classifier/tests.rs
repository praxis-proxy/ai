// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Unit tests for the `openai_operation` classifier.

#![expect(clippy::unwrap_used, clippy::expect_used, reason = "tests")]

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
    assert_eq!(
        matched.application_protocol,
        ApplicationProtocol::new("openai_conversations")
    );
    assert_eq!(matched.operation_id, "getConversation");
    assert_eq!(matched.transport, Transport::Http);

    assert_eq!(
        ctx.filter_metadata
            .get("openai_operation.application_protocol")
            .map(String::as_str),
        Some("openai_conversations")
    );
    assert_eq!(
        ctx.filter_metadata
            .get("openai_operation.operation_id")
            .map(String::as_str),
        Some("getConversation")
    );
}

#[tokio::test]
async fn classifies_chat_completions_operations_from_the_request_head() {
    for (method, path, operation_id) in [
        ("POST", "/v1/chat/completions", "createChatCompletion"),
        ("POST", "/v1/chat/completions/", "createChatCompletion"),
        ("POST", "/v1/chat/completions?stream=true", "createChatCompletion"),
        ("GET", "/v1/chat/completions", "listChatCompletions"),
        ("GET", "/v1/chat/completions/chatcmpl_abc", "getChatCompletion"),
        ("POST", "/v1/chat/completions/chatcmpl_abc", "updateChatCompletion"),
        ("DELETE", "/v1/chat/completions/chatcmpl_abc", "deleteChatCompletion"),
        (
            "GET",
            "/v1/chat/completions/chatcmpl_abc/messages",
            "getChatCompletionMessages",
        ),
    ] {
        let filter = default_filter();
        let request = req(method, path);
        let mut ctx = make_filter_context(&request);
        drop(filter.on_request(&mut ctx).await.unwrap());

        let matched = ctx.extensions.get::<OpenAiOperationMatch>().copied();
        assert!(matched.is_some(), "{method} {path} must classify");
        let matched = matched.unwrap();
        assert_eq!(
            matched.application_protocol,
            ApplicationProtocol::new("openai_chat_completions"),
            "{method} {path}"
        );
        assert_eq!(matched.operation_id, operation_id, "{method} {path}");
        assert_eq!(matched.transport, Transport::Http, "{method} {path}");
    }
}

#[tokio::test]
async fn websocket_handshake_on_chat_completions_does_not_match() {
    let filter = default_filter();
    let mut request = req("GET", "/v1/chat/completions");
    request.headers = websocket_headers();
    let mut ctx = make_filter_context(&request);
    drop(filter.on_request(&mut ctx).await.unwrap());
    assert!(
        ctx.extensions.get::<OpenAiOperationMatch>().is_none(),
        "Chat Completions is HTTP-only, so a websocket handshake must not classify"
    );
}

#[tokio::test]
async fn classifies_a_responses_operation() {
    let filter = default_filter();
    let request = req("POST", "/v1/responses");
    let mut ctx = make_filter_context(&request);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let matched = ctx.extensions.get::<OpenAiOperationMatch>().copied().unwrap();
    assert_eq!(
        matched.application_protocol,
        ApplicationProtocol::new("openai_responses")
    );
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

    assert!(set.contains(&(
        "x-praxis-ai-application-protocol".to_owned(),
        "openai_responses".to_owned()
    )));
    assert!(set.contains(&("x-praxis-ai-operation".to_owned(), "createResponse".to_owned())));
}

#[tokio::test]
async fn client_supplied_headers_cannot_spoof_a_matched_operation() {
    let filter = default_filter();
    let mut request = req("POST", "/v1/responses");
    request
        .headers
        .insert("x-praxis-ai-application-protocol", "openai_files".parse().unwrap());
    request
        .headers
        .insert("x-praxis-ai-operation", "createFile".parse().unwrap());
    let mut ctx = make_filter_context(&request);

    drop(filter.on_request(&mut ctx).await.unwrap());

    // set (overwrite) semantics, so the forged values cannot survive alongside.
    let protocol = ctx
        .request_headers_to_set
        .iter()
        .find(|(name, _)| name.as_str() == "x-praxis-ai-application-protocol")
        .map(|(_, value)| value.to_str().unwrap().to_owned());
    assert_eq!(protocol.as_deref(), Some("openai_responses"));
}

#[tokio::test]
async fn client_supplied_headers_are_stripped_when_nothing_matches() {
    let filter = default_filter();
    let mut request = req("GET", "/v1/unknown");
    request
        .headers
        .insert("x-praxis-ai-application-protocol", "openai_responses".parse().unwrap());
    let mut ctx = make_filter_context(&request);

    drop(filter.on_request(&mut ctx).await.unwrap());

    assert!(ctx.extensions.get::<OpenAiOperationMatch>().is_none());
    let removed: Vec<&str> = ctx
        .request_headers_to_remove
        .iter()
        .map(http::HeaderName::as_str)
        .collect();
    assert!(removed.contains(&"x-praxis-ai-application-protocol"));
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
    assert_eq!(matched.transport, Transport::WebSocket);
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
        ("PUT", "/v1/chat/completions"),
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
    let filter = filter("\nheaders:\n  application_protocol: x-protocol\n  operation: x-operation\n");
    let request = req("POST", "/v1/responses");
    let mut ctx = make_filter_context(&request);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let names: Vec<&str> = ctx
        .request_headers_to_set
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();
    assert!(names.contains(&"x-protocol"));
    assert!(names.contains(&"x-operation"));
}

#[tokio::test]
async fn headers_can_be_disabled_while_metadata_still_publishes() {
    let filter = filter("\nheaders:\n  application_protocol: ~\n  operation: ~\n");
    let request = req("POST", "/v1/responses");
    let mut ctx = make_filter_context(&request);
    drop(filter.on_request(&mut ctx).await.unwrap());

    assert!(ctx.request_headers_to_set.is_empty());
    assert_eq!(
        ctx.filter_metadata
            .get("openai_operation.application_protocol")
            .map(String::as_str),
        Some("openai_responses")
    );
}

#[test]
fn invalid_header_name_is_rejected_at_startup() {
    let value: serde_yaml::Value = serde_yaml::from_str("headers:\n  application_protocol: \"bad header\"\n").unwrap();
    assert!(OpenaiOperationFilter::from_config(&value).is_err());
}

#[test]
fn unknown_configuration_fields_are_rejected() {
    let value: serde_yaml::Value = serde_yaml::from_str("nonsense: true\n").unwrap();
    assert!(OpenaiOperationFilter::from_config(&value).is_err());
}

#[test]
fn header_targets_carrying_auth_or_framing_are_rejected() {
    for target in ["authorization", "host", "content-length", "cookie", "transfer-encoding"] {
        let value: serde_yaml::Value =
            serde_yaml::from_str(&format!("headers:\n  application_protocol: {target}\n")).unwrap();
        assert!(
            OpenaiOperationFilter::from_config(&value).is_err(),
            "{target} must not be an overwritable classifier target"
        );
    }
}

#[test]
fn both_outputs_targeting_one_header_is_rejected() {
    let value: serde_yaml::Value =
        serde_yaml::from_str("headers:\n  application_protocol: x-same\n  operation: X-Same\n").unwrap();
    assert!(
        OpenaiOperationFilter::from_config(&value).is_err(),
        "a shared target would let the operation value replace the application protocol"
    );
}

#[tokio::test]
async fn publishes_filter_results_for_branch_conditions() {
    let filter = default_filter();
    let request = req("POST", "/v1/responses");
    let mut ctx = make_filter_context(&request);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let results = ctx
        .filter_results
        .get("openai_operation")
        .expect("the classifier must publish filter results for on_result branching");
    assert_eq!(results.get("application_protocol"), Some("openai_responses"));
    assert_eq!(results.get("operation_id"), Some("createResponse"));
}

#[tokio::test]
async fn upgrade_headers_on_a_non_get_request_still_classify() {
    let filter = default_filter();
    let mut request = req("POST", "/v1/responses");
    request.headers = websocket_headers();
    let mut ctx = make_filter_context(&request);

    drop(filter.on_request(&mut ctx).await.unwrap());

    let matched = ctx
        .extensions
        .get::<OpenAiOperationMatch>()
        .copied()
        .expect("upgrade headers must not suppress classification of a non-GET operation");
    assert_eq!(matched.operation_id, "createResponse");
    assert_eq!(matched.transport, Transport::Http);
}

#[test]
fn a_websocket_handshake_must_be_a_get() {
    assert_eq!(
        request_transport("POST", &websocket_headers()),
        Transport::Http,
        "only GET carries the RFC 6455 opening handshake"
    );
}

#[test]
fn transport_detection_follows_the_opening_handshake() {
    assert_eq!(request_transport("GET", &websocket_headers()), Transport::WebSocket);
    assert_eq!(request_transport("GET", &http::HeaderMap::new()), Transport::Http);

    // Connection is a token list.
    let mut list = http::HeaderMap::new();
    list.insert(http::header::CONNECTION, "keep-alive, Upgrade".parse().unwrap());
    list.insert(http::header::UPGRADE, "websocket".parse().unwrap());
    assert_eq!(request_transport("GET", &list), Transport::WebSocket);

    // Upgrade without Connection: upgrade is not a handshake.
    let mut partial = http::HeaderMap::new();
    partial.insert(http::header::UPGRADE, "websocket".parse().unwrap());
    assert_eq!(request_transport("GET", &partial), Transport::Http);

    // Several nominated protocols are not treated as a websocket handshake.
    let mut multi = http::HeaderMap::new();
    multi.insert(http::header::CONNECTION, "Upgrade".parse().unwrap());
    multi.append(http::header::UPGRADE, "websocket".parse().unwrap());
    multi.append(http::header::UPGRADE, "h2c".parse().unwrap());
    assert_eq!(request_transport("GET", &multi), Transport::Http);
}
