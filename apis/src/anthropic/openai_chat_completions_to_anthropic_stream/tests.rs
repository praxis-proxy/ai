// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

use bytes::Bytes;
use http::{HeaderValue, StatusCode};
use praxis_filter::{FilterAction, HttpFilter, Response};

use super::{ARMED, OpenaiChatCompletionsToAnthropicStreamFilter};

const EVENTS: &str = concat!(
    "event: message_start\n",
    "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"usage\":{\"input_tokens\":3}}}\n\n",
    "event: content_block_start\n",
    "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
    "event: content_block_delta\n",
    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hello\"}}\n\n",
    "event: message_delta\n",
    "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n",
    "event: message_stop\n",
    "data: {\"type\":\"message_stop\"}\n\n"
);

#[tokio::test]
async fn converts_anthropic_sse_incrementally_and_emits_openai_termination_and_usage() {
    let filter = filter("max_partial_event_bytes: 4096\nmax_tool_blocks: 10");
    let request = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut response = sse_response();
    let mut ctx = crate::test_utils::make_filter_context(&request);
    arm(&mut ctx, true);
    ctx.response_header = Some(&mut response);
    assert!(matches!(
        filter.on_response(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    assert_eq!(ctx.get_metadata(ARMED), Some("true"));
    ctx.set_metadata("intelligent_route.api_format", "openai-chat");

    let split = EVENTS.find("event: content_block_delta").unwrap();
    let mut first = Some(Bytes::copy_from_slice(&EVENTS.as_bytes()[..split]));
    assert!(matches!(
        filter.on_response_body(&mut ctx, &mut first, false).unwrap(),
        FilterAction::Continue
    ));
    let first_output = std::str::from_utf8(first.as_ref().unwrap()).unwrap();
    assert!(first_output.contains("chat.completion.chunk"));
    assert!(!first_output.contains("hello"));
    assert!(
        !first_output.contains("data: [DONE]"),
        "the first chunk must be observable before stream completion"
    );
    let mut second = Some(Bytes::copy_from_slice(&EVENTS.as_bytes()[split..]));
    assert!(matches!(
        filter.on_response_body(&mut ctx, &mut second, true).unwrap(),
        FilterAction::Continue
    ));
    let output = std::str::from_utf8(second.as_ref().unwrap()).unwrap();
    assert!(output.contains("hello"));
    assert!(output.contains("\"model\":\"client-model\""));
    assert!(output.contains("\"prompt_tokens\":3"));
    assert!(output.contains("\"completion_tokens\":2"));
    assert!(output.contains("data: [DONE]"));
}

#[tokio::test]
async fn terminal_sse_usage_includes_cached_input() {
    let filter = filter("{}");
    let request = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut response = sse_response();
    let mut ctx = crate::test_utils::make_filter_context(&request);
    arm(&mut ctx, true);
    ctx.response_header = Some(&mut response);
    assert!(matches!(
        filter.on_response(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    let events = EVENTS.replace(
        "\"input_tokens\":3",
        "\"input_tokens\":10,\"cache_read_input_tokens\":1000,\"cache_creation_input_tokens\":50",
    );
    let mut body = Some(Bytes::from(events));
    assert!(matches!(
        filter.on_response_body(&mut ctx, &mut body, true).unwrap(),
        FilterAction::Continue
    ));
    let output = std::str::from_utf8(body.as_ref().unwrap()).unwrap();
    assert!(output.contains("\"prompt_tokens\":1060"));
    assert!(output.contains("\"total_tokens\":1062"));
    assert!(output.contains("data: [DONE]"));
}

#[tokio::test]
async fn maps_refusal_stop_reason_to_content_filter_finish_reason() {
    let filter = filter("{}");
    let request = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut response = sse_response();
    let mut ctx = crate::test_utils::make_filter_context(&request);
    arm(&mut ctx, true);
    ctx.response_header = Some(&mut response);
    assert!(matches!(
        filter.on_response(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    let events = EVENTS.replace("end_turn", "refusal");
    let mut body = Some(Bytes::from(events));
    assert!(matches!(
        filter.on_response_body(&mut ctx, &mut body, true).unwrap(),
        FilterAction::Continue
    ));
    let output = std::str::from_utf8(body.as_ref().unwrap()).unwrap();
    assert!(output.contains("\"finish_reason\":\"content_filter\""));
    assert!(output.contains("data: [DONE]"));
}

#[tokio::test]
async fn maps_context_window_exhaustion_to_length_finish_reason() {
    let filter = filter("{}");
    let request = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut response = sse_response();
    let mut ctx = crate::test_utils::make_filter_context(&request);
    arm(&mut ctx, true);
    ctx.response_header = Some(&mut response);
    assert!(matches!(
        filter.on_response(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    let events = EVENTS.replace("end_turn", "model_context_window_exceeded");
    let mut body = Some(Bytes::from(events));
    assert!(matches!(
        filter.on_response_body(&mut ctx, &mut body, true).unwrap(),
        FilterAction::Continue
    ));
    let output = std::str::from_utf8(body.as_ref().unwrap()).unwrap();
    assert!(output.contains("\"finish_reason\":\"length\""));
    assert!(output.contains("data: [DONE]"));
}

#[tokio::test]
async fn converts_tool_deltas_and_stops_after_message_stop() {
    let filter = filter("{}");
    let request = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut response = sse_response();
    let mut ctx = crate::test_utils::make_filter_context(&request);
    arm(&mut ctx, false);
    ctx.response_header = Some(&mut response);
    assert!(matches!(
        filter.on_response(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));

    let events = concat!(
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_tool\",\"usage\":{\"input_tokens\":1}}}\n\n",
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":4,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"weather\",\"input\":{}}}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":4,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":\\\"Oslo\\\"}\"}}\n\n",
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":4}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
    );
    let mut body = Some(Bytes::from_static(events.as_bytes()));
    assert!(matches!(
        filter.on_response_body(&mut ctx, &mut body, true).unwrap(),
        FilterAction::Continue
    ));
    let output = std::str::from_utf8(body.as_ref().unwrap()).unwrap();
    assert!(output.contains("\"name\":\"weather\""));
    assert!(output.contains("\\\"city\\\":\\\"Oslo\\\""));
    assert!(output.contains("\"finish_reason\":\"tool_calls\""));
    assert!(output.contains("data: [DONE]"));
}

#[tokio::test]
async fn leaves_openai_candidate_response_unmodified() {
    let filter = filter("{}");
    let request = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut response = sse_response();
    let mut ctx = crate::test_utils::make_filter_context(&request);
    arm(&mut ctx, false);
    ctx.response_header = Some(&mut response);
    ctx.set_metadata("openai_chat_completions_to_anthropic.bound.api_format", "openai-chat");
    assert!(matches!(
        filter.on_response(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    assert_eq!(ctx.get_metadata(ARMED), None);
    for (chunk, eos) in [
        (b"data: {\"choices\":".as_slice(), false),
        (b"[]}\n\n".as_slice(), true),
    ] {
        let original = Bytes::copy_from_slice(chunk);
        let mut body = Some(original.clone());
        assert!(matches!(
            filter.on_response_body(&mut ctx, &mut body, eos).unwrap(),
            FilterAction::Continue
        ));
        assert_eq!(
            body,
            Some(original),
            "OpenAI SSE chunks remain unchanged and progressive"
        );
    }
}

#[tokio::test]
async fn provider_error_event_is_mapped_to_openai_sse_error_and_done() {
    let filter = filter("{}");
    let request = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut response = sse_response();
    let mut ctx = crate::test_utils::make_filter_context(&request);
    arm(&mut ctx, false);
    ctx.response_header = Some(&mut response);
    assert!(matches!(
        filter.on_response(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    let event = b"event: error\ndata: {\"type\":\"error\",\"error\":{\"message\":\"overloaded\"}}\n\n";
    let mut body = Some(Bytes::from_static(event));
    assert!(matches!(
        filter.on_response_body(&mut ctx, &mut body, true).unwrap(),
        FilterAction::Continue
    ));
    let output = std::str::from_utf8(body.as_ref().unwrap()).unwrap();
    assert!(output.contains("overloaded"));
    assert!(output.contains("data: [DONE]"));
}

#[tokio::test]
async fn partial_record_limit_fails_closed_with_terminal_error() {
    let filter = filter("max_partial_event_bytes: 32\nmax_tool_blocks: 10");
    let request = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut response = sse_response();
    let mut ctx = crate::test_utils::make_filter_context(&request);
    arm(&mut ctx, false);
    ctx.response_header = Some(&mut response);
    assert!(matches!(
        filter.on_response(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    let mut body = Some(Bytes::from_static(
        b"event: x\ndata: 123456789012345\ndata: 123456789012345\ndata: 123456789012345\n",
    ));
    assert!(matches!(
        filter.on_response_body(&mut ctx, &mut body, false).unwrap(),
        FilterAction::Continue
    ));
    let output = std::str::from_utf8(body.as_ref().unwrap()).unwrap();
    assert!(output.contains("configured streaming limit"));
    assert!(output.contains("data: [DONE]"));

    let mut trailing = Some(Bytes::from_static(EVENTS.as_bytes()));
    assert!(matches!(
        filter.on_response_body(&mut ctx, &mut trailing, true).unwrap(),
        FilterAction::Continue
    ));
    assert_eq!(trailing, Some(Bytes::new()));
}

fn filter(yaml: &str) -> Box<dyn HttpFilter> {
    let config: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
    OpenaiChatCompletionsToAnthropicStreamFilter::from_config(&config).unwrap()
}

fn arm(ctx: &mut praxis_filter::HttpFilterContext<'_>, include_usage: bool) {
    // Pipeline execution sets this before each filter callback; the test
    // context helper intentionally leaves it unset.
    ctx.current_filter_id = Some(1);
    ctx.set_metadata("openai_chat_completions_to_anthropic.bound.api_format", "messages");
    ctx.set_metadata("openai_chat_completions_to_anthropic.request_stream", "true");
    ctx.set_metadata("openai_chat_completions_to_anthropic.client_model", "client-model");
    ctx.set_metadata(
        "openai_chat_completions_to_anthropic_stream.include_usage",
        include_usage.to_string(),
    );
}

fn sse_response() -> Response {
    Response {
        status: StatusCode::OK,
        headers: [(
            http::header::CONTENT_TYPE,
            HeaderValue::from_static("text/event-stream"),
        )]
        .into_iter()
        .collect(),
    }
}
