// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional coverage for cross-provider inference failover: an
//! OpenAI-compatible primary that fails over to an Anthropic Messages
//! fallback with JSON and streaming SSE request and response translation.

use praxis_core::config::Config;
use praxis_test_utils::{
    CapturedRequest, StatefulCapturingBackend, example_config_path, free_port, http_send, parse_body, parse_status,
    start_proxy,
};
use serde_json::{Value, json};

const EXAMPLE: &str = "inference/fallback-with-provider-translation.yaml";
const RELATIVE_OVERLAY: &str = "tests/integration/fixtures/inference/routing/openai-to-anthropic.json";

const ANTHROPIC_JSON: &str = r#"{"type":"message","id":"msg_fallback","role":"assistant","model":"claude-fixture-model","content":[{"type":"text","text":"anthropic fallback"}],"stop_reason":"end_turn","usage":{"input_tokens":7,"output_tokens":5}}"#;
const ANTHROPIC_EVENTS: &str = concat!(
    "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_fallback_stream\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-fixture-model\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":7,\"output_tokens\":0}}}\n\n",
    "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
    "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"anthropic \"}}\n\n",
    "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"fallback\"}}\n\n",
    "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
    "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":5}}\n\n",
    "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
);
const OPENAI_JSON: &str = r#"{"id":"chatcmpl-primary","object":"chat.completion","model":"fixture-openai-model","choices":[{"index":0,"message":{"role":"assistant","content":"openai primary"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}"#;

#[test]
fn openai_primary_failure_fails_over_to_translated_anthropic() {
    let openai = StatefulCapturingBackend::new(vec![(503, r#"{"error":"primary unavailable"}"#.to_owned())])
        .start_with_shutdown();
    let anthropic = StatefulCapturingBackend::new(vec![(200, ANTHROPIC_JSON.to_owned())]).start_with_shutdown();

    let config = Config::from_yaml(&example_yaml(free_port(), openai.port(), anthropic.port()))
        .expect("example pipeline must parse");
    let proxy = start_proxy(&config);

    let raw = http_send(proxy.addr(), &chat_request(false));

    let primary_requests = openai.requests();
    assert_eq!(primary_requests.len(), 1, "primary should receive exactly one request");
    let primary = post_request(&primary_requests);
    assert_eq!(
        primary.uri, "/v1/chat/completions",
        "primary path should stay unchanged"
    );
    assert!(
        primary
            .headers
            .to_ascii_lowercase()
            .contains("authorization: bearer openai-key"),
        "primary upstream must receive its own bearer credential"
    );

    let fallback_requests = anthropic.requests();
    assert_eq!(
        fallback_requests.len(),
        1,
        "fallback should receive exactly one request"
    );
    let fallback = post_request(&fallback_requests);
    assert_eq!(
        fallback.uri, "/v1/messages",
        "fallback path must be rewritten to Anthropic Messages"
    );
    let fallback_body: Value = serde_json::from_str(&fallback.body).expect("translated Anthropic JSON");
    assert_eq!(
        fallback_body["model"], "claude-fixture-model",
        "target model must come from the overlay"
    );
    assert_eq!(
        fallback_body["messages"][0]["content"][0]["text"], "hi",
        "client text must be translated"
    );
    assert_eq!(fallback_body["max_tokens"], 32, "generation limits must be translated");
    assert!(
        fallback
            .headers
            .to_ascii_lowercase()
            .contains("x-api-key: anthropic-fixture-key"),
        "Anthropic upstream must receive the selected API key"
    );
    assert!(
        fallback
            .headers
            .to_ascii_lowercase()
            .contains("anthropic-version: 2023-06-01"),
        "Anthropic upstream must receive the protocol version header"
    );
    assert!(
        !fallback.headers.to_ascii_lowercase().contains("authorization: bearer"),
        "Anthropic upstream must not receive a bearer credential"
    );

    assert_eq!(
        parse_status(&raw),
        200,
        "client should receive 200 after fallback succeeds: {raw}"
    );
    let response: Value = serde_json::from_str(&parse_body(&raw)).expect("client response JSON");
    assert_eq!(
        response["object"], "chat.completion",
        "response must be OpenAI Chat Completions"
    );
    assert_eq!(
        response["choices"][0]["message"]["content"], "anthropic fallback",
        "translated response must preserve Anthropic text"
    );
    assert_eq!(
        response["choices"][0]["finish_reason"], "stop",
        "end_turn must map to stop"
    );
    assert_eq!(
        response["usage"]["prompt_tokens"], 7,
        "input usage must map to prompt_tokens"
    );
    assert_eq!(
        response["usage"]["completion_tokens"], 5,
        "output usage must map to completion_tokens"
    );
    assert_eq!(
        response["usage"]["total_tokens"], 12,
        "translated usage must include total_tokens"
    );
    assert!(
        response.get("type").is_none(),
        "translated response must not expose Anthropic's type field"
    );
}

#[test]
fn openai_primary_failure_fails_over_to_translated_anthropic_stream() {
    let openai = StatefulCapturingBackend::new(vec![(503, r#"{"error":"primary unavailable"}"#.to_owned())])
        .start_with_shutdown();
    let anthropic = StatefulCapturingBackend::new(vec![(200, ANTHROPIC_EVENTS.to_owned())]).start_with_shutdown();

    let config = Config::from_yaml(&example_yaml(free_port(), openai.port(), anthropic.port()))
        .expect("example pipeline must parse");
    let proxy = start_proxy(&config);

    let raw = http_send(proxy.addr(), &chat_request(true));

    assert_eq!(
        parse_status(&raw),
        200,
        "client should receive 200 after streaming fallback succeeds: {raw}"
    );
    assert!(
        raw.to_ascii_lowercase().contains("content-type: text/event-stream"),
        "translated response must retain SSE content type"
    );
    let body = parse_body(&raw);
    assert!(
        body.contains("anthropic ") && body.contains("fallback"),
        "translated stream must preserve Anthropic text deltas: {body}"
    );
    assert!(
        body.contains("chat.completion.chunk"),
        "translated stream must use OpenAI Chat Completions chunks: {body}"
    );
    assert!(
        body.contains("data: [DONE]"),
        "translated stream must terminate with OpenAI's DONE sentinel: {body}"
    );

    let primary_requests = openai.requests();
    assert_eq!(primary_requests.len(), 1, "primary should receive exactly one request");
    let fallback_requests = anthropic.requests();
    assert_eq!(
        fallback_requests.len(),
        1,
        "fallback should receive exactly one streaming request"
    );
    let fallback = post_request(&fallback_requests);
    assert_eq!(
        fallback.uri, "/v1/messages",
        "streaming fallback path should be rewritten to Anthropic Messages"
    );
    let fallback_body: Value =
        serde_json::from_str(&fallback.body).expect("translated Anthropic streaming request JSON");
    assert_eq!(
        fallback_body["stream"], true,
        "streaming client request must enable Anthropic streaming"
    );
}

#[test]
fn openai_primary_success_skips_anthropic_fallback() {
    let openai = StatefulCapturingBackend::new(vec![(200, OPENAI_JSON.to_owned())]).start_with_shutdown();
    let anthropic = StatefulCapturingBackend::new(vec![(200, ANTHROPIC_JSON.to_owned())]).start_with_shutdown();

    let config = Config::from_yaml(&example_yaml(free_port(), openai.port(), anthropic.port()))
        .expect("example pipeline must parse");
    let proxy = start_proxy(&config);

    let raw = http_send(proxy.addr(), &chat_request(false));

    assert_eq!(parse_status(&raw), 200, "primary success: {raw}");
    assert_eq!(openai.requests().len(), 1, "primary should receive exactly one request");
    assert_eq!(
        anthropic.requests().len(),
        0,
        "fallback must not be contacted when the primary succeeds"
    );
    let response: Value = serde_json::from_str(&parse_body(&raw)).expect("client response JSON");
    assert_eq!(
        response["choices"][0]["message"]["content"], "openai primary",
        "primary response must pass through unchanged"
    );
}

/// Render the checked-in example with an isolated listener, an absolute
/// overlay path, and test backend ports.
fn example_yaml(listener_port: u16, openai_port: u16, anthropic_port: u16) -> String {
    let example = example_config_path(EXAMPLE);
    let yaml = std::fs::read_to_string(&example).unwrap_or_else(|error| panic!("read {example}: {error}"));
    let overlay = format!(
        "{}/fixtures/inference/routing/openai-to-anthropic.json",
        env!("CARGO_MANIFEST_DIR")
    );
    yaml.replace("127.0.0.1:8080", &format!("127.0.0.1:{listener_port}"))
        .replace(RELATIVE_OVERLAY, &overlay)
        .replace("127.0.0.1:3001", &format!("127.0.0.1:{openai_port}"))
        .replace("127.0.0.1:3002", &format!("127.0.0.1:{anthropic_port}"))
}

/// Build an OpenAI Chat Completions request matching the overlay candidate.
fn chat_request(stream: bool) -> String {
    let body = json!({
        "model": "fixture-openai-model",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 32,
        "stream": stream
    })
    .to_string();
    format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nAuthorization: Bearer caller-key\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    )
}

/// Extract the recorded provider POST or report the methods observed instead.
fn post_request(requests: &[CapturedRequest]) -> CapturedRequest {
    let Some(request) = requests.iter().find(|request| request.method == "POST") else {
        let methods = requests
            .iter()
            .map(|request| request.method.as_str())
            .collect::<Vec<_>>();
        panic!("no POST request observed upstream, observed methods: {methods:?}");
    };
    CapturedRequest {
        method: request.method.clone(),
        uri: request.uri.clone(),
        headers: request.headers.clone(),
        body: request.body.clone(),
    }
}
