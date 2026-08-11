// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

use std::time::Duration;

use bytes::Bytes;
use http::StatusCode;
use praxis_core::time::FixedTimeSource;
use praxis_filter::{BodyAccess, BodyMode, FilterAction};
use serde_json::json;

use super::{
    ARMED_KEY, CREATED_AT_KEY, RESPONSE_STATUS_KEY, ResponsesToChatCompletionsFilter, error::normalize_provider_error,
};
use crate::openai::responses::state::ResponsesState;

#[test]
fn default_config_parses() {
    let yaml = serde_yaml::from_str("{}").unwrap();
    let filter = ResponsesToChatCompletionsFilter::from_config(&yaml).unwrap();

    assert_eq!(filter.name(), "responses_to_chat_completions");
    assert_eq!(filter.request_body_access(), BodyAccess::ReadWrite);
    assert!(matches!(
        filter.request_body_mode(),
        BodyMode::StreamBuffer {
            max_bytes: Some(67_108_864)
        }
    ));
    assert!(matches!(filter.response_body_mode(), BodyMode::Stream));
}

#[tokio::test]
async fn request_headers_wait_for_successful_classification() {
    let filter = ResponsesToChatCompletionsFilter::from_config(&serde_yaml::Value::Null).unwrap();
    let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut context = crate::test_utils::make_filter_context(&request);

    let action = filter.on_request(&mut context).await.unwrap();

    assert!(matches!(action, FilterAction::Continue));
    assert!(context.request_headers_to_remove.is_empty());
}

#[tokio::test]
async fn non_create_request_continues_without_rewriting_or_arming() {
    let filter = ResponsesToChatCompletionsFilter::from_config(&serde_yaml::Value::Null).unwrap();
    let request = crate::test_utils::make_request(http::Method::GET, "/v1/responses");
    let mut context = crate::test_utils::make_filter_context(&request);
    let original = Bytes::from_static(br#"{"model":"gpt-4.1-mini","input":"hello"}"#);
    let mut body = Some(original.clone());

    let action = filter.on_request_body(&mut context, &mut body, true).await.unwrap();

    assert!(matches!(action, FilterAction::Continue));
    assert_eq!(body.as_deref(), Some(original.as_ref()));
    assert!(context.get_metadata(ARMED_KEY).is_none());
    assert!(context.get_metadata(CREATED_AT_KEY).is_none());
}

#[test]
fn custom_body_limit_parses() {
    let yaml = serde_yaml::from_str("max_body_bytes: 1048576").unwrap();
    let filter = ResponsesToChatCompletionsFilter::from_config(&yaml).unwrap();

    assert!(matches!(
        filter.request_body_mode(),
        BodyMode::StreamBuffer {
            max_bytes: Some(1_048_576)
        }
    ));
}

#[test]
fn zero_body_limit_is_rejected() {
    let yaml = serde_yaml::from_str("max_body_bytes: 0").unwrap();

    assert!(ResponsesToChatCompletionsFilter::from_config(&yaml).is_err());
}

#[test]
fn oversized_body_limit_is_rejected() {
    let yaml = serde_yaml::from_str("max_body_bytes: 67108865").unwrap();

    assert!(ResponsesToChatCompletionsFilter::from_config(&yaml).is_err());
}

#[test]
fn unknown_config_key_is_rejected() {
    let yaml = serde_yaml::from_str("unexpected: true").unwrap();

    assert!(ResponsesToChatCompletionsFilter::from_config(&yaml).is_err());
}

#[test]
fn response_body_access_is_read_write() {
    let yaml = serde_yaml::from_str("{}").unwrap();
    let filter = ResponsesToChatCompletionsFilter::from_config(&yaml).unwrap();

    assert_eq!(filter.response_body_access(), BodyAccess::ReadWrite);
}

#[tokio::test]
async fn classified_non_responses_request_is_released() {
    let filter = ResponsesToChatCompletionsFilter::from_config(&serde_yaml::Value::Null).unwrap();
    let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut context = crate::test_utils::make_filter_context(&request);
    context.set_metadata("openai_responses_format.format", "openai_chat_completions");
    let mut body = Some(Bytes::from_static(br#"{"messages":[]}"#));

    let action = filter.on_request_body(&mut context, &mut body, true).await.unwrap();

    assert!(matches!(action, FilterAction::Release));
}

#[tokio::test]
async fn responses_create_without_classifier_metadata_fails_closed() {
    let filter = ResponsesToChatCompletionsFilter::from_config(&serde_yaml::Value::Null).unwrap();
    let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut context = crate::test_utils::make_filter_context(&request);
    let mut body = Some(Bytes::from_static(br#"{"model":"m","input":"hello"}"#));

    let action = filter.on_request_body(&mut context, &mut body, true).await.unwrap();

    assert_server_error(action);
}

#[tokio::test]
async fn classified_responses_create_without_state_fails_closed() {
    let filter = ResponsesToChatCompletionsFilter::from_config(&serde_yaml::Value::Null).unwrap();
    let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut context = crate::test_utils::make_filter_context(&request);
    context.set_metadata("openai_responses_format.format", "openai_responses");
    let mut body = Some(Bytes::from_static(br#"{"model":"m","input":"hello"}"#));

    let action = filter.on_request_body(&mut context, &mut body, true).await.unwrap();

    assert_server_error(action);
}

#[tokio::test]
async fn streaming_responses_create_without_state_uses_sse_error() {
    let filter = ResponsesToChatCompletionsFilter::from_config(&serde_yaml::Value::Null).unwrap();
    let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut context = crate::test_utils::make_filter_context(&request);
    context.set_metadata("openai_responses_format.format", "openai_responses");
    context.set_metadata("openai_responses_format.stream", "true");
    let mut body = Some(Bytes::from_static(br#"{"model":"m","input":"hello","stream":true}"#));

    let action = filter.on_request_body(&mut context, &mut body, true).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected rejection");
    };
    assert_eq!(rejection.status, 500);
    assert_eq!(
        rejection.headers.iter().find(|(name, _)| name == "content-type"),
        Some(&("content-type".to_owned(), "text/event-stream".to_owned()))
    );
    let event = std::str::from_utf8(rejection.body.as_deref().unwrap()).unwrap();
    assert!(event.starts_with("event: error\ndata: "));
    let data = event.strip_prefix("event: error\ndata: ").unwrap().trim_end();
    let parsed: serde_json::Value = serde_json::from_str(data).unwrap();
    assert_eq!(parsed["error"]["code"], "server_error");
    assert_eq!(parsed["error"]["message"], "request pipeline state is unavailable");
}

fn assert_server_error(action: FilterAction) {
    let FilterAction::Reject(rejection) = action else {
        panic!("expected rejection");
    };
    assert_eq!(rejection.status, 500);
    let parsed: serde_json::Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();
    assert_eq!(parsed["error"]["code"], "server_error");
}

#[tokio::test]
async fn canonical_state_is_translated_and_arms_response() {
    let filter = ResponsesToChatCompletionsFilter::from_config(&serde_yaml::Value::Null).unwrap();
    let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let fixed_time = FixedTimeSource::new(Duration::from_secs(1_700_000_000));
    let mut context = crate::test_utils::make_filter_context(&request);
    context.time_source = &fixed_time;
    context.set_metadata("openai_responses_format.format", "openai_responses");
    context.set_metadata("openai_responses_format.stream", "false");
    let request_body = json!({
        "model": "gpt-4.1-mini",
        "input": "current input",
        "stream": false
    });
    let mut state = ResponsesState::from_request_body(request_body);
    state.messages = vec![
        json!({"role": "user", "content": "earlier history"}),
        json!({"role": "user", "content": "current input"}),
    ];
    context.extensions.insert(state);
    let mut body = Some(Bytes::from_static(
        br#"{"model":"gpt-4.1-mini","input":"current input","stream":false}"#,
    ));

    let action = filter.on_request_body(&mut context, &mut body, true).await.unwrap();

    assert!(matches!(action, FilterAction::Continue));
    let translated: serde_json::Value = serde_json::from_slice(body.as_deref().unwrap()).unwrap();
    assert_eq!(translated["model"], "gpt-4.1-mini");
    assert_eq!(translated["messages"][0]["content"], "earlier history");
    assert_eq!(translated["messages"][1]["content"], "current input");
    assert_eq!(translated["stream"], false);
    assert!(
        context
            .request_headers_to_remove
            .contains(&http::header::ACCEPT_ENCODING)
    );
    assert_eq!(context.get_metadata(ARMED_KEY), Some("true"));
    assert_eq!(context.get_metadata(CREATED_AT_KEY), Some("1700000000"));
}

#[tokio::test]
async fn translated_request_over_configured_limit_is_rejected() {
    let yaml = serde_yaml::from_str("max_body_bytes: 32").unwrap();
    let filter = ResponsesToChatCompletionsFilter::from_config(&yaml).unwrap();
    let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut context = crate::test_utils::make_filter_context(&request);
    context.set_metadata("openai_responses_format.format", "openai_responses");
    context.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "gpt-4.1-mini",
        "input": "hello"
    })));
    let mut body = Some(Bytes::from_static(br#"{"model":"gpt-4.1-mini","input":"hello"}"#));

    let action = filter.on_request_body(&mut context, &mut body, true).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected rejection");
    };
    assert_eq!(rejection.status, 413);
    let parsed: serde_json::Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();
    assert_eq!(parsed["error"]["code"], "invalid_request_error");
    assert!(context.get_metadata(ARMED_KEY).is_none());
    assert!(context.get_metadata(CREATED_AT_KEY).is_none());
}

#[tokio::test]
async fn unsupported_responses_tool_is_rejected_at_filter_boundary() {
    let filter = ResponsesToChatCompletionsFilter::from_config(&serde_yaml::Value::Null).unwrap();
    let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut context = crate::test_utils::make_filter_context(&request);
    context.set_metadata("openai_responses_format.format", "openai_responses");
    context.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "gpt-4.1-mini",
        "input": "hello",
        "tools": [{"type": "web_search"}]
    })));
    let mut body = Some(Bytes::from_static(
        br#"{"model":"gpt-4.1-mini","input":"hello","tools":[{"type":"web_search"}]}"#,
    ));

    let action = filter.on_request_body(&mut context, &mut body, true).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected rejection");
    };
    assert_eq!(rejection.status, 400);
    let parsed: serde_json::Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();
    assert_eq!(parsed["error"]["code"], "invalid_request_error");
    assert!(context.get_metadata(ARMED_KEY).is_none());
    assert!(context.get_metadata(CREATED_AT_KEY).is_none());
}

#[tokio::test]
async fn streaming_translation_error_uses_responses_sse_error_event() {
    let filter = ResponsesToChatCompletionsFilter::from_config(&serde_yaml::Value::Null).unwrap();
    let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut context = crate::test_utils::make_filter_context(&request);
    context.set_metadata("openai_responses_format.format", "openai_responses");
    context.set_metadata("openai_responses_format.stream", "true");
    context.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "gpt-4.1-mini",
        "input": "hello",
        "stream": true,
        "tools": [{"type": "web_search"}]
    })));
    let mut body = Some(Bytes::from_static(
        br#"{"model":"gpt-4.1-mini","input":"hello","stream":true,"tools":[{"type":"web_search"}]}"#,
    ));

    let action = filter.on_request_body(&mut context, &mut body, true).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected rejection");
    };
    assert_responses_sse_translation_error(&rejection);
    assert!(context.get_metadata(ARMED_KEY).is_none());
    assert!(context.get_metadata(CREATED_AT_KEY).is_none());
}

fn assert_responses_sse_translation_error(rejection: &praxis_filter::Rejection) {
    assert_eq!(rejection.status, 400);
    assert_eq!(
        rejection.headers.iter().find(|(name, _)| name == "content-type"),
        Some(&("content-type".to_owned(), "text/event-stream".to_owned()))
    );
    let event = std::str::from_utf8(rejection.body.as_deref().unwrap()).unwrap();
    assert!(event.starts_with("event: error\ndata: "));
    assert!(event.ends_with("\n\n"));
    let data = event.strip_prefix("event: error\ndata: ").unwrap().trim_end();
    let parsed: serde_json::Value = serde_json::from_str(data).unwrap();
    assert_eq!(parsed["type"], "error");
    assert_eq!(parsed["sequence_number"], 0);
    assert_eq!(parsed["error"]["type"], "invalid_request_error");
    assert_eq!(parsed["error"]["code"], "invalid_request_error");
    assert_eq!(
        parsed["error"]["message"],
        "unsupported Responses tool type for Chat Completions translation: web_search"
    );
    assert!(parsed["error"]["param"].is_null());
}

#[tokio::test]
async fn successful_sse_response_stays_streaming() {
    let yaml = serde_yaml::from_str("{}").unwrap();
    let filter = ResponsesToChatCompletionsFilter::from_config(&yaml).unwrap();
    let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut context = crate::test_utils::make_filter_context(&request);
    context.set_metadata("openai_responses_format.format", "openai_responses");
    context.set_metadata("openai_responses_format.stream", "true");
    context.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "gpt-4.1-mini",
        "input": "hello",
        "stream": true
    })));
    let mut request_body = Some(Bytes::from_static(
        br#"{"model":"gpt-4.1-mini","input":"hello","stream":true}"#,
    ));
    let request_action = filter
        .on_request_body(&mut context, &mut request_body, true)
        .await
        .unwrap();
    assert!(matches!(request_action, FilterAction::Continue));
    let response = Box::leak(Box::new(crate::test_utils::make_response()));
    response.headers.insert(
        http::header::CONTENT_TYPE,
        http::HeaderValue::from_static("text/event-stream; charset=utf-8"),
    );
    response
        .headers
        .insert(http::header::CONTENT_LENGTH, http::HeaderValue::from_static("42"));
    context.response_header = Some(response);

    let action = filter.on_response(&mut context).await.unwrap();

    assert!(matches!(action, FilterAction::Continue));
    assert!(matches!(context.response_body_mode, BodyMode::Stream));
    assert!(
        context
            .response_header
            .as_ref()
            .unwrap()
            .headers
            .contains_key(http::header::CONTENT_LENGTH)
    );
}

#[tokio::test]
async fn non_streaming_success_upgrades_to_bounded_buffer() {
    let yaml = serde_yaml::from_str("max_body_bytes: 1048576").unwrap();
    let filter = ResponsesToChatCompletionsFilter::from_config(&yaml).unwrap();
    let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut context = crate::test_utils::make_filter_context(&request);
    context.set_metadata(ARMED_KEY, "true");
    let response = Box::leak(Box::new(crate::test_utils::make_response()));
    response.headers.insert(
        http::header::CONTENT_TYPE,
        http::HeaderValue::from_static("application/json"),
    );
    response
        .headers
        .insert(http::header::CONTENT_LENGTH, http::HeaderValue::from_static("42"));
    response
        .headers
        .insert(http::header::ETAG, http::HeaderValue::from_static("\"provider\""));
    response
        .headers
        .insert("content-digest", http::HeaderValue::from_static("sha-256=:abc=:"));
    context.response_header = Some(response);

    let action = filter.on_response(&mut context).await.unwrap();

    assert!(matches!(action, FilterAction::Continue));
    assert!(matches!(
        context.response_body_mode,
        BodyMode::StreamBuffer {
            max_bytes: Some(1_048_576)
        }
    ));
    assert!(
        !context
            .response_header
            .as_ref()
            .unwrap()
            .headers
            .contains_key(http::header::CONTENT_LENGTH)
    );
    for header in [http::header::ETAG, http::HeaderName::from_static("content-digest")] {
        assert!(!context.response_header.as_ref().unwrap().headers.contains_key(header));
    }
    assert!(context.response_headers_modified);
}

#[tokio::test]
async fn encoded_success_is_rejected_before_response_headers_are_committed() {
    let filter = ResponsesToChatCompletionsFilter::from_config(&serde_yaml::Value::Null).unwrap();
    let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut context = crate::test_utils::make_filter_context(&request);
    context.set_metadata(ARMED_KEY, "true");
    let response = Box::leak(Box::new(crate::test_utils::make_response()));
    response
        .headers
        .insert(http::header::CONTENT_ENCODING, http::HeaderValue::from_static("gzip"));
    response.headers.insert(
        http::header::CONTENT_RANGE,
        http::HeaderValue::from_static("bytes 0-41/42"),
    );
    context.response_header = Some(response);

    let action = filter.on_response(&mut context).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("encoded finite success should be rejected");
    };
    assert_eq!(rejection.status, 502);
    let parsed: serde_json::Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();
    assert_eq!(parsed["error"]["code"], "server_error");
}

#[tokio::test]
async fn partial_content_success_is_rejected_before_response_headers_are_committed() {
    let filter = ResponsesToChatCompletionsFilter::from_config(&serde_yaml::Value::Null).unwrap();
    let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut context = crate::test_utils::make_filter_context(&request);
    context.set_metadata(ARMED_KEY, "true");
    let response = Box::leak(Box::new(crate::test_utils::make_response()));
    response.status = StatusCode::PARTIAL_CONTENT;
    context.response_header = Some(response);

    let action = filter.on_response(&mut context).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("partial finite success should be rejected");
    };
    assert_eq!(rejection.status, 502);
    let parsed: serde_json::Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();
    assert_eq!(parsed["error"]["code"], "server_error");
}

#[tokio::test]
async fn redirect_response_is_not_treated_as_provider_error() {
    let filter = ResponsesToChatCompletionsFilter::from_config(&serde_yaml::Value::Null).unwrap();
    let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut context = crate::test_utils::make_filter_context(&request);
    context.set_metadata(ARMED_KEY, "true");
    let response = Box::leak(Box::new(crate::test_utils::make_response()));
    response.status = StatusCode::TEMPORARY_REDIRECT;
    context.response_header = Some(response);

    let action = filter.on_response(&mut context).await.unwrap();

    assert!(matches!(action, FilterAction::Continue));
    assert!(matches!(context.response_body_mode, BodyMode::Stream));
    assert!(context.get_metadata(RESPONSE_STATUS_KEY).is_none());
    assert!(!context.response_headers_modified);
}

#[tokio::test]
async fn response_cleanup_without_headers_does_not_arm_body_processing() {
    let yaml = serde_yaml::from_str("{}").unwrap();
    let filter = ResponsesToChatCompletionsFilter::from_config(&yaml).unwrap();
    let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut context = crate::test_utils::make_filter_context(&request);
    context.set_metadata(ARMED_KEY, "true");

    let action = filter.on_response(&mut context).await.unwrap();

    assert!(matches!(action, FilterAction::Continue));
    assert!(matches!(context.response_body_mode, BodyMode::Stream));
    assert!(context.get_metadata(RESPONSE_STATUS_KEY).is_none());
}

#[tokio::test]
async fn finite_json_error_for_streaming_request_upgrades_to_buffer() {
    let yaml = serde_yaml::from_str("{}").unwrap();
    let filter = ResponsesToChatCompletionsFilter::from_config(&yaml).unwrap();
    let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut context = crate::test_utils::make_filter_context(&request);
    context.set_metadata(ARMED_KEY, "true");
    context.set_metadata("openai_responses_format.stream", "true");
    let response = Box::leak(Box::new(crate::test_utils::make_response()));
    response.status = StatusCode::BAD_REQUEST;
    response.headers.insert(
        http::header::CONTENT_TYPE,
        http::HeaderValue::from_static("application/json"),
    );
    context.response_header = Some(response);

    let action = filter.on_response(&mut context).await.unwrap();

    assert!(matches!(action, FilterAction::Continue));
    assert!(matches!(context.response_body_mode, BodyMode::StreamBuffer { .. }));
}

#[tokio::test]
async fn sse_error_response_stays_streaming() {
    let yaml = serde_yaml::from_str("{}").unwrap();
    let filter = ResponsesToChatCompletionsFilter::from_config(&yaml).unwrap();
    let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut context = crate::test_utils::make_filter_context(&request);
    context.set_metadata(ARMED_KEY, "true");
    let response = Box::leak(Box::new(crate::test_utils::make_response()));
    response.status = StatusCode::BAD_REQUEST;
    response.headers.insert(
        http::header::CONTENT_TYPE,
        http::HeaderValue::from_static("text/event-stream"),
    );
    context.response_header = Some(response);

    let action = filter.on_response(&mut context).await.unwrap();

    assert!(matches!(action, FilterAction::Continue));
    assert!(matches!(context.response_body_mode, BodyMode::Stream));
}

#[tokio::test]
async fn non_streaming_chat_response_becomes_response_resource() {
    let yaml = serde_yaml::from_str("{}").unwrap();
    let filter = ResponsesToChatCompletionsFilter::from_config(&yaml).unwrap();
    let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let fixed_time = FixedTimeSource::new(Duration::from_secs(1_700_000_000));
    let mut context = crate::test_utils::make_filter_context(&request);
    context.time_source = &fixed_time;
    context.set_metadata("openai_responses_format.format", "openai_responses");
    context.set_metadata("openai_responses_format.stream", "false");
    context.set_metadata("responses.response_id", "resp_test_123");
    let request_value = json!({
        "model": "gpt-4.1-mini",
        "input": "hello",
        "stream": false
    });
    context
        .extensions
        .insert(ResponsesState::from_request_body(request_value));
    let mut request_body = Some(Bytes::from_static(
        br#"{"model":"gpt-4.1-mini","input":"hello","stream":false}"#,
    ));
    let request_action = filter
        .on_request_body(&mut context, &mut request_body, true)
        .await
        .unwrap();
    assert!(matches!(request_action, FilterAction::Continue));
    let response = Box::leak(Box::new(crate::test_utils::make_response()));
    response.headers.insert(
        http::header::CONTENT_TYPE,
        http::HeaderValue::from_static("application/json"),
    );
    response
        .headers
        .insert(http::header::CONTENT_LENGTH, http::HeaderValue::from_static("999"));
    context.response_header = Some(response);
    let response_action = filter.on_response(&mut context).await.unwrap();
    assert!(matches!(response_action, FilterAction::Continue));
    let response = context.response_header.as_ref().unwrap();
    assert_eq!(response.status, StatusCode::OK);
    assert_eq!(
        response.headers.get(http::header::CONTENT_TYPE).unwrap(),
        "application/json"
    );
    assert!(!response.headers.contains_key(http::header::CONTENT_LENGTH));
    assert!(context.response_headers_modified);
    context.response_header = None;
    let mut response_body = Some(Bytes::from_static(
        br#"{"id":"chatcmpl_1","object":"chat.completion","model":"gpt-4.1-mini","choices":[{"index":0,"message":{"role":"assistant","content":"Hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}"#,
    ));

    let body_action = filter.on_response_body(&mut context, &mut response_body, true).unwrap();

    assert!(matches!(body_action, FilterAction::Continue));
    let translated: serde_json::Value = serde_json::from_slice(response_body.as_deref().unwrap()).unwrap();
    assert_eq!(translated["id"], "resp_test_123");
    assert_eq!(translated["object"], "response");
    assert_eq!(translated["output"][0]["content"][0]["text"], "Hello");
    assert_eq!(translated["usage"]["input_tokens"], 3);
    assert_eq!(translated["usage"]["output_tokens"], 2);
}

#[tokio::test]
async fn malformed_success_aborts_after_headers_are_sent() {
    let yaml = serde_yaml::from_str("{}").unwrap();
    let filter = ResponsesToChatCompletionsFilter::from_config(&yaml).unwrap();
    let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut context = crate::test_utils::make_filter_context(&request);
    context.set_metadata(ARMED_KEY, "true");
    context.set_metadata(CREATED_AT_KEY, "1700000000");
    context.set_metadata("responses.response_id", "resp_test_123");
    context.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "gpt-4.1-mini",
        "input": "hello"
    })));
    let response = Box::leak(Box::new(crate::test_utils::make_response()));
    response.headers.insert(
        http::header::CONTENT_TYPE,
        http::HeaderValue::from_static("application/json"),
    );
    context.response_header = Some(response);
    assert!(matches!(
        filter.on_response(&mut context).await.unwrap(),
        FilterAction::Continue
    ));
    context.response_header = None;
    let mut body = Some(Bytes::from_static(b"not-json"));

    let error = filter.on_response_body(&mut context, &mut body, true).unwrap_err();

    assert!(error.to_string().contains("invalid Chat Completions response"));
    assert_eq!(body.as_deref(), Some(b"not-json".as_slice()));
}

#[tokio::test]
async fn finite_provider_error_uses_captured_status_without_mutable_headers() {
    let yaml = serde_yaml::from_str("{}").unwrap();
    let filter = ResponsesToChatCompletionsFilter::from_config(&yaml).unwrap();
    let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut context = crate::test_utils::make_filter_context(&request);
    context.set_metadata(ARMED_KEY, "true");
    let response = Box::leak(Box::new(crate::test_utils::make_response()));
    response.status = StatusCode::TOO_MANY_REQUESTS;
    response.headers.insert(
        http::header::CONTENT_TYPE,
        http::HeaderValue::from_static("application/json"),
    );
    response
        .headers
        .insert(http::header::CONTENT_ENCODING, http::HeaderValue::from_static("gzip"));
    response.headers.insert(
        http::header::CONTENT_RANGE,
        http::HeaderValue::from_static("bytes 0-41/42"),
    );
    context.response_header = Some(response);
    assert!(matches!(
        filter.on_response(&mut context).await.unwrap(),
        FilterAction::Continue
    ));
    assert!(
        !context
            .response_header
            .as_ref()
            .unwrap()
            .headers
            .contains_key(http::header::CONTENT_ENCODING)
    );
    assert!(
        !context
            .response_header
            .as_ref()
            .unwrap()
            .headers
            .contains_key(http::header::CONTENT_RANGE)
    );
    context.response_header = None;
    let mut body = Some(Bytes::from_static(
        br#"{"error":{"code":"rate_limit_exceeded","message":"slow down"}}"#,
    ));

    let action = filter.on_response_body(&mut context, &mut body, true).unwrap();

    assert!(matches!(action, FilterAction::Continue));
    let parsed: serde_json::Value = serde_json::from_slice(body.as_deref().unwrap()).unwrap();
    assert_eq!(parsed["error"]["code"], "rate_limit_exceeded");
    assert_eq!(parsed["error"]["type"], "rate_limit_exceeded");
    assert_eq!(parsed["error"]["message"], "slow down");
    assert!(parsed["error"]["param"].is_null());
}

#[tokio::test]
async fn expanded_provider_error_falls_back_within_configured_limit() {
    let yaml = serde_yaml::from_str("max_body_bytes: 150").unwrap();
    let filter = ResponsesToChatCompletionsFilter::from_config(&yaml).unwrap();
    let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut context = crate::test_utils::make_filter_context(&request);
    context.set_metadata(ARMED_KEY, "true");
    let response = Box::leak(Box::new(crate::test_utils::make_response()));
    response.status = StatusCode::BAD_REQUEST;
    response.headers.insert(
        http::header::CONTENT_TYPE,
        http::HeaderValue::from_static("application/json"),
    );
    context.response_header = Some(response);
    assert!(matches!(
        filter.on_response(&mut context).await.unwrap(),
        FilterAction::Continue
    ));
    context.response_header = None;
    let provider_body = serde_json::to_vec(&json!({
        "error": {
            "code": "invalid_prompt",
            "message": "x".repeat(80)
        }
    }))
    .unwrap();
    assert!(provider_body.len() <= 150);
    let mut body = Some(Bytes::from(provider_body));

    let action = filter.on_response_body(&mut context, &mut body, true).unwrap();

    assert!(matches!(action, FilterAction::Continue));
    assert!(body.as_ref().unwrap().len() <= 150);
    let parsed: serde_json::Value = serde_json::from_slice(body.as_deref().unwrap()).unwrap();
    assert_eq!(parsed["error"]["message"], "upstream provider returned an error");
}

#[tokio::test]
async fn oversized_translated_success_aborts_after_headers_are_sent() {
    let yaml = serde_yaml::from_str("max_body_bytes: 512").unwrap();
    let filter = ResponsesToChatCompletionsFilter::from_config(&yaml).unwrap();
    let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut context = crate::test_utils::make_filter_context(&request);
    context.set_metadata(ARMED_KEY, "true");
    context.set_metadata(CREATED_AT_KEY, "1700000000");
    context.set_metadata("responses.response_id", "resp_test_123");
    context.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "gpt-4.1-mini",
        "input": "hello"
    })));
    let response = Box::leak(Box::new(crate::test_utils::make_response()));
    response.headers.insert(
        http::header::CONTENT_TYPE,
        http::HeaderValue::from_static("application/json"),
    );
    context.response_header = Some(response);
    assert!(matches!(
        filter.on_response(&mut context).await.unwrap(),
        FilterAction::Continue
    ));
    context.response_header = None;
    let provider_body = json!({
        "id": "chatcmpl_large",
        "object": "chat.completion",
        "model": "gpt-4.1-mini",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "x".repeat(1_024)},
            "finish_reason": "stop"
        }]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&provider_body).unwrap()));

    let error = filter.on_response_body(&mut context, &mut body, true).unwrap_err();

    assert!(error.to_string().contains("translated response exceeds maximum size"));
}

#[tokio::test]
async fn successful_sse_chunks_are_byte_exact() {
    let yaml = serde_yaml::from_str("{}").unwrap();
    let filter = ResponsesToChatCompletionsFilter::from_config(&yaml).unwrap();
    let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut context = crate::test_utils::make_filter_context(&request);
    context.set_metadata(ARMED_KEY, "true");
    let response = Box::leak(Box::new(crate::test_utils::make_response()));
    response.headers.insert(
        http::header::CONTENT_TYPE,
        http::HeaderValue::from_static("text/event-stream"),
    );
    context.response_header = Some(response);
    let response_action = filter.on_response(&mut context).await.unwrap();
    assert!(matches!(response_action, FilterAction::Continue));
    assert!(matches!(context.response_body_mode, BodyMode::Stream));
    context.response_header = None;
    let first =
        Bytes::from_static(b"data: {\"id\":\"chatcmpl_1\",\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n");
    let second = Bytes::from_static(b"data: [DONE]\n\n");
    let mut first_body = Some(first.clone());
    let mut second_body = Some(second.clone());

    let first_action = filter.on_response_body(&mut context, &mut first_body, false).unwrap();
    let second_action = filter.on_response_body(&mut context, &mut second_body, true).unwrap();

    assert!(matches!(first_action, FilterAction::Continue));
    assert!(matches!(second_action, FilterAction::Continue));
    assert_eq!(first_body.as_deref(), Some(first.as_ref()));
    assert_eq!(second_body.as_deref(), Some(second.as_ref()));
}

#[test]
fn nested_valid_response_code_and_message_are_preserved() {
    let normalized = normalize_provider_error(
        StatusCode::TOO_MANY_REQUESTS,
        br#"{"error":{"code":"rate_limit_exceeded","message":"slow down"}}"#,
    );
    assert_eq!(normalized.code, "rate_limit_exceeded");
    assert_eq!(normalized.message, "slow down");
}

#[test]
fn direct_invalid_base64_code_is_mapped() {
    let normalized = normalize_provider_error(
        StatusCode::BAD_REQUEST,
        br#"{"code":"invalid_base64","message":"bad image"}"#,
    );
    assert_eq!(normalized.code, "invalid_base64_image");
    assert_eq!(normalized.message, "bad image");
}

#[test]
fn unknown_client_error_falls_back_to_invalid_prompt() {
    let normalized = normalize_provider_error(
        StatusCode::UNPROCESSABLE_ENTITY,
        br#"{"error":{"code":"backend_specific","message":"bad request"}}"#,
    );
    assert_eq!(normalized.code, "invalid_prompt");
    assert_eq!(normalized.message, "bad request");
}

#[test]
fn code_less_rate_limit_uses_rate_limit_error_code() {
    let normalized = normalize_provider_error(StatusCode::TOO_MANY_REQUESTS, br#"{"error":{"message":"slow down"}}"#);
    assert_eq!(normalized.code, "rate_limit_exceeded");
    assert_eq!(normalized.message, "slow down");
}

#[test]
fn authentication_errors_do_not_masquerade_as_invalid_prompts() {
    for status in [StatusCode::UNAUTHORIZED, StatusCode::FORBIDDEN] {
        let normalized = normalize_provider_error(status, br#"{"error":{"message":"access denied"}}"#);
        assert_eq!(normalized.code, "server_error");
        assert_eq!(normalized.message, "access denied");
    }
}

#[test]
fn malformed_server_error_falls_back_without_reflecting_body() {
    let normalized = normalize_provider_error(StatusCode::BAD_GATEWAY, b"private upstream details");
    assert_eq!(normalized.code, "server_error");
    assert_eq!(normalized.message, "upstream provider returned an error");
}
