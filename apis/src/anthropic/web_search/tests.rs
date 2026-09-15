// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

use std::{
    io::{Read as _, Write as _},
    net::{TcpListener, TcpStream},
    sync::{Arc, Mutex},
};

use bytes::Bytes;
use http::Method;
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, HttpFilter, HttpFilterContext, Request, Response, StreamTerminationCause,
    SubRequestResponseMode,
};
use serde_json::{Value, json};

use super::*;
use crate::test_utils::{make_filter_context, make_request, make_response};

fn test_filter() -> Box<dyn HttpFilter> {
    let config = serde_yaml::from_str(
        r"
provider: you
api_key: test-key
default_context_size: medium
",
    )
    .unwrap();
    AnthropicWebSearchFilter::from_config(&config).unwrap()
}

fn terminal_streaming_filter() -> Box<dyn HttpFilter> {
    let config = serde_yaml::from_str(
        r"
provider: you
api_key: test-key
default_context_size: medium
terminal_streaming: true
",
    )
    .unwrap();
    AnthropicWebSearchFilter::from_config(&config).unwrap()
}

fn test_filter_impl_with_base_url(base_url: &str) -> AnthropicWebSearchFilter {
    let config = serde_yaml::from_str(&format!(
        r#"
provider: you
api_key: test-key
default_context_size: medium
base_url: "{base_url}"
allow_private_base_url: true
"#,
    ))
    .unwrap();
    let config: WebSearchFilterConfig = parse_filter_config(FILTER_NAME, &config).unwrap();
    let validated = build_config(FILTER_NAME, &config).unwrap();
    let client = crate::subrequest::SubRequestClient::new(praxis_core::subrequest::SubRequestConnector::new(4, None));
    let search_client = SearchClient::from_config(FILTER_NAME, &validated, client).unwrap();
    AnthropicWebSearchFilter {
        default_context_size: validated.default_context_size,
        max_body_bytes: validated.max_body_bytes,
        terminal_streaming: validated.terminal_streaming,
        search_client,
    }
}

struct SearchStub {
    base_url: String,
    requests: Arc<Mutex<Vec<String>>>,
}

impl SearchStub {
    fn base_url(&self) -> &str {
        &self.base_url
    }

    fn last_request(&self) -> String {
        self.requests.lock().unwrap().last().unwrap().clone()
    }

    fn last_json(&self) -> Value {
        let request = self.last_request();
        let (_, body) = request.split_once("\r\n\r\n").unwrap();
        serde_json::from_str(body).unwrap()
    }
}

fn start_you_search_stub(status: u16, body: String) -> SearchStub {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&requests);
    std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        captured.lock().unwrap().push(read_http_request(&mut stream));
        let response = format!(
            "HTTP/1.1 {status} Test\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        stream.write_all(response.as_bytes()).unwrap();
    });
    SearchStub {
        base_url: format!("http://{address}"),
        requests,
    }
}

fn read_http_request(stream: &mut TcpStream) -> String {
    let mut request = Vec::new();
    loop {
        let mut buffer = [0_u8; 4096];
        let count = stream.read(&mut buffer).unwrap();
        assert!(count > 0, "search request closed before its body arrived");
        request.extend_from_slice(buffer.get(..count).unwrap());

        let Some(header_end) = request.windows(4).position(|window| window == b"\r\n\r\n") else {
            continue;
        };
        let body_start = header_end + 4;
        let headers = String::from_utf8_lossy(request.get(..header_end).unwrap()).to_ascii_lowercase();
        let content_length = headers
            .lines()
            .find_map(|line| line.strip_prefix("content-length:"))
            .map(str::trim)
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0);
        if request.len() >= body_start + content_length {
            return String::from_utf8(request).unwrap();
        }
    }
}

fn valid_you_body() -> String {
    json!({
        "results": {
            "web": [{
                "title": "Potato - Wikipedia",
                "url": "https://en.wikipedia.org/wiki/Potato",
                "description": "Potato is a starchy tuber native to the Americas."
            }],
            "news": []
        }
    })
    .to_string()
}

fn empty_you_body() -> String {
    json!({"results": {"web": [], "news": []}}).to_string()
}

#[test]
fn search_stub_reads_full_content_length_body() {
    let search = start_you_search_stub(200, valid_you_body());
    let query = "q".repeat(20 * 1024);
    let body = json!({"query": query, "count": 5}).to_string();
    let address = search.base_url().strip_prefix("http://").unwrap();
    let mut stream = TcpStream::connect(address).unwrap();
    let request = format!(
        "POST /v1/search HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );

    stream.write_all(request.as_bytes()).unwrap();
    let mut response = Vec::new();
    stream.read_to_end(&mut response).unwrap();

    assert!(String::from_utf8_lossy(&response).starts_with("HTTP/1.1 200"));
    assert_eq!(search.last_json()["query"], query);
}

fn base_request() -> Value {
    json!({
        "model":"openai/gpt-oss-20b",
        "max_tokens":256,
        "system":"Answer with sources.",
        "metadata":{"user_id":"demo"},
        "tools":[{"name":"WebSearch","description":"Search the web","input_schema":{"type":"object"}}],
        "tool_choice":{"type":"tool","name":"WebSearch"},
        "messages":[{"role":"user","content":"Find potato facts"}]
    })
}

fn pending_search(query: &str) -> PendingSearch {
    PendingSearch {
        id: "toolu_search_1".to_owned(),
        query: query.to_owned(),
    }
}

fn assistant_content(query: &str) -> Vec<Value> {
    vec![json!({
        "type":"tool_use","id":"toolu_search_1","name":"WebSearch","input":{"query":query}
    })]
}

fn message_response(content: Value, stop_reason: &str) -> Bytes {
    Bytes::from(
        json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "test",
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": null,
            "usage": {"input_tokens": 10, "output_tokens": 5}
        })
        .to_string(),
    )
}

async fn initialized_context<'a>(request: &'a Request) -> HttpFilterContext<'a> {
    let filter = test_filter();
    let mut ctx = make_filter_context(request);
    let mut body = Some(Bytes::from_static(
        br#"{"model":"test","max_tokens":32,"messages":[{"role":"user","content":"search"}]}"#,
    ));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));
    ctx
}

fn result_action(ctx: &HttpFilterContext<'_>) -> Option<String> {
    ctx.filter_results.get(FILTER_NAME)?.get("action").map(str::to_owned)
}

#[tokio::test]
async fn streaming_request_is_rejected_before_reentry() {
    let filter = test_filter();
    let request = make_request(Method::POST, "/v1/messages");
    let mut ctx = make_filter_context(&request);
    let mut body = Some(Bytes::from_static(
        br#"{"model":"test","max_tokens":32,"stream":true,"messages":[]}"#,
    ));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    let FilterAction::Reject(rejection) = action else {
        panic!("expected rejection");
    };
    assert_eq!(rejection.status, 400);
    let body: Value = serde_json::from_slice(rejection.body.as_ref().unwrap()).unwrap();
    assert_eq!(body["type"], "error");
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("streaming is not supported"))
    );
}

#[test]
fn terminal_streaming_declares_streaming_capability() {
    assert!(
        !test_filter().may_select_streaming_subrequest_response(),
        "the buffered loop must not declare the Praxis streaming capability"
    );
    assert!(
        terminal_streaming_filter().may_select_streaming_subrequest_response(),
        "terminal_streaming must declare the Praxis streaming capability"
    );
}

#[test]
fn terminal_streaming_uses_incremental_response_body() {
    let buffered = test_filter();
    assert_eq!(buffered.response_body_access(), BodyAccess::ReadOnly);
    assert!(
        matches!(buffered.response_body_mode(), BodyMode::StreamBuffer { .. }),
        "the buffered loop must accumulate the whole response before classifying"
    );

    let streaming = terminal_streaming_filter();
    assert_eq!(streaming.response_body_access(), BodyAccess::ReadWrite);
    assert!(
        matches!(streaming.response_body_mode(), BodyMode::Stream),
        "terminal streaming must deliver response chunks incrementally"
    );
}

#[tokio::test]
async fn terminal_streaming_accepts_streaming_request_and_selects_streaming_transport() {
    let filter = terminal_streaming_filter();
    let request = make_request(Method::POST, "/v1/messages");
    let mut ctx = make_filter_context(&request);
    let mut body = Some(Bytes::from_static(
        br#"{"model":"test","max_tokens":32,"stream":true,"messages":[{"role":"user","content":"hi"}]}"#,
    ));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(
        matches!(action, FilterAction::Continue),
        "terminal streaming must accept an effective streaming request"
    );
    assert_eq!(
        ctx.subrequest_response_mode(),
        SubRequestResponseMode::Streaming,
        "an effective streaming request must select the streaming transport"
    );
}

#[tokio::test]
async fn terminal_streaming_keeps_buffered_transport_when_not_streaming() {
    let filter = terminal_streaming_filter();
    for body_bytes in [
        br#"{"model":"test","max_tokens":32,"stream":false,"messages":[{"role":"user","content":"hi"}]}"#.as_slice(),
        br#"{"model":"test","max_tokens":32,"messages":[{"role":"user","content":"hi"}]}"#.as_slice(),
    ] {
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);
        ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
        let mut body = Some(Bytes::copy_from_slice(body_bytes));

        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

        assert!(
            matches!(action, FilterAction::Continue),
            "a non-streaming request must continue under terminal_streaming"
        );
        assert_eq!(
            ctx.subrequest_response_mode(),
            SubRequestResponseMode::Buffered,
            "a non-streaming effective body must keep the buffered transport"
        );
    }
}

#[tokio::test]
async fn sole_web_search_tool_use_signals_loop() {
    let filter = test_filter();
    let request = make_request(Method::POST, "/v1/messages");
    let mut ctx = initialized_context(&request).await;
    let mut body = Some(message_response(
        json!([{
            "type":"tool_use","id":"toolu_search_1","name":"WebSearch",
            "input":{"query":"potato"}
        }]),
        "tool_use",
    ));

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();

    assert!(matches!(action, FilterAction::Continue));
    assert_eq!(result_action(&ctx).as_deref(), Some("loop"));
    let ResponseDecision::Managed(pending) = classify_response(body.as_ref().unwrap()) else {
        panic!("expected managed search");
    };
    assert_eq!(pending.query, "potato");
}

#[tokio::test]
async fn vllm_end_turn_web_search_signals_loop() {
    let filter = test_filter();
    let request = make_request(Method::POST, "/v1/messages");
    let mut ctx = initialized_context(&request).await;
    let mut body = Some(Bytes::from_static(
        br#"{"id":"chatcmpl-8594675bd3b17d40","type":"message","role":"assistant","content":[{"type":"tool_use","id":"chatcmpl-tool-8cb8901f3f024ffe","name":"WebSearch","input":{"query":"potato"}}],"model":"RedHatAI/Qwen3-Coder-Next-NVFP4","stop_reason":"end_turn","usage":{"input_tokens":293,"output_tokens":23}}"#,
    ));

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();

    assert!(matches!(action, FilterAction::Continue));
    assert_eq!(result_action(&ctx).as_deref(), Some("loop"));
    let ResponseDecision::Managed(pending) = classify_response(body.as_ref().unwrap()) else {
        panic!("expected vLLM WebSearch response to be managed");
    };
    assert_eq!(pending.query, "potato");
}

#[tokio::test]
async fn non_success_web_search_message_signals_done_unchanged() {
    let filter = test_filter();
    let request = make_request(Method::POST, "/v1/messages");
    let mut ctx = initialized_context(&request).await;
    let mut response = make_response();
    response.status = http::StatusCode::TOO_MANY_REQUESTS;
    ctx.response_header = Some(&mut response);
    let original = message_response(
        json!([{
            "type":"tool_use","id":"toolu_search_1","name":"WebSearch",
            "input":{"query":"potato"}
        }]),
        "tool_use",
    );
    let mut body = Some(original.clone());

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();

    assert!(matches!(action, FilterAction::Continue));
    assert_eq!(result_action(&ctx).as_deref(), Some("done"));
    assert_eq!(body, Some(original));
}

#[tokio::test]
async fn managed_query_at_utf8_byte_limit_signals_loop() {
    let filter = test_filter();
    let request = make_request(Method::POST, "/v1/messages");
    let mut ctx = initialized_context(&request).await;
    let query = "é".repeat(4096);
    let mut body = Some(message_response(
        json!([{
            "type":"tool_use","id":"toolu_search_1","name":"WebSearch",
            "input":{"query":query}
        }]),
        "tool_use",
    ));

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();

    assert!(matches!(action, FilterAction::Continue));
    assert_eq!(result_action(&ctx).as_deref(), Some("loop"));
}

#[tokio::test]
async fn escaped_managed_query_signals_loop_with_decoded_text() {
    let filter = test_filter();
    let request = make_request(Method::POST, "/v1/messages");
    let mut ctx = initialized_context(&request).await;
    let mut body = Some(message_response(
        json!([{
            "type":"tool_use","id":"toolu_search_1","name":"WebSearch",
            "input":{"query":"potato\ncultivation"}
        }]),
        "tool_use",
    ));

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();

    assert!(matches!(action, FilterAction::Continue));
    assert_eq!(result_action(&ctx).as_deref(), Some("loop"));
    let ResponseDecision::Managed(pending) = classify_response(body.as_ref().unwrap()) else {
        panic!("expected managed search");
    };
    assert_eq!(pending.query, "potato\ncultivation");
}

#[tokio::test]
async fn managed_query_over_utf8_byte_limit_is_rejected() {
    let filter = test_filter();
    let request = make_request(Method::POST, "/v1/messages");
    let mut ctx = initialized_context(&request).await;
    let query = format!("{}x", "é".repeat(4096));
    let mut body = Some(message_response(
        json!([{
            "type":"tool_use","id":"toolu_search_1","name":"WebSearch",
            "input":{"query":query}
        }]),
        "tool_use",
    ));

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected rejection");
    };
    assert_eq!(rejection.status, 400);
    assert!(String::from_utf8_lossy(rejection.body.as_ref().unwrap()).contains("8192 bytes"));
    assert_ne!(result_action(&ctx).as_deref(), Some("loop"));
}

#[tokio::test]
async fn client_owned_and_mixed_tools_signal_done() {
    for content in [
        json!([{"type":"tool_use","id":"toolu_bash","name":"Bash","input":{}}]),
        json!([
            {"type":"tool_use","id":"toolu_search","name":"WebSearch","input":{"query":"potato"}},
            {"type":"tool_use","id":"toolu_bash","name":"Bash","input":{}}
        ]),
    ] {
        let filter = test_filter();
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = initialized_context(&request).await;
        let original = message_response(content, "tool_use");
        let mut body = Some(original.clone());

        let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();

        assert!(matches!(action, FilterAction::Continue));
        assert_eq!(result_action(&ctx).as_deref(), Some("done"));
        assert_eq!(body, Some(original));
    }
}

#[tokio::test]
async fn managed_call_without_query_is_rejected() {
    let filter = test_filter();
    let request = make_request(Method::POST, "/v1/messages");
    let mut ctx = initialized_context(&request).await;
    let mut body = Some(message_response(
        json!([{"type":"tool_use","id":"toolu_search","name":"WebSearch","input":{}}]),
        "tool_use",
    ));

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected rejection");
    };
    assert_eq!(rejection.status, 400);
    assert!(String::from_utf8_lossy(rejection.body.as_ref().unwrap()).contains("query"));
}

#[tokio::test]
async fn managed_call_with_empty_id_is_rejected() {
    let filter = test_filter();
    let request = make_request(Method::POST, "/v1/messages");
    let mut ctx = initialized_context(&request).await;
    let mut body = Some(message_response(
        json!([{"type":"tool_use","id":"","name":"WebSearch","input":{"query":"potato"}}]),
        "tool_use",
    ));

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected rejection");
    };
    assert_eq!(rejection.status, 400);
}

#[tokio::test]
async fn managed_call_with_whitespace_only_query_is_rejected() {
    let filter = test_filter();
    let request = make_request(Method::POST, "/v1/messages");
    let mut ctx = initialized_context(&request).await;
    let mut body = Some(message_response(
        json!([{"type":"tool_use","id":"toolu_search","name":"WebSearch","input":{"query":"   "}}]),
        "tool_use",
    ));

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected rejection");
    };
    assert_eq!(rejection.status, 400);
}

#[tokio::test]
async fn final_text_signals_done_without_mutating_body() {
    let filter = test_filter();
    let request = make_request(Method::POST, "/v1/messages");
    let mut ctx = initialized_context(&request).await;
    let original = message_response(json!([{"type":"text","text":"Potatoes grow underground."}]), "end_turn");
    let mut body = Some(original.clone());

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();

    assert!(matches!(action, FilterAction::Continue));
    assert_eq!(result_action(&ctx).as_deref(), Some("done"));
    assert_eq!(body, Some(original));
}

#[tokio::test]
async fn non_message_error_body_signals_done() {
    let filter = test_filter();
    let request = make_request(Method::POST, "/v1/messages");
    let mut ctx = initialized_context(&request).await;
    let original = Bytes::from_static(br#"{"type":"error","error":{"type":"overloaded_error","message":"busy"}}"#);
    let mut body = Some(original.clone());

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();

    assert!(matches!(action, FilterAction::Continue));
    assert_eq!(result_action(&ctx).as_deref(), Some("done"));
    assert_eq!(body, Some(original));
}

#[tokio::test]
async fn non_end_of_stream_is_noop() {
    let filter = test_filter();
    let request = make_request(Method::POST, "/v1/messages");
    let mut ctx = initialized_context(&request).await;
    let mut body = Some(message_response(json!([]), "end_turn"));

    let action = filter.on_response_body(&mut ctx, &mut body, false).unwrap();

    assert!(matches!(action, FilterAction::Continue));
    assert!(ctx.filter_results.is_empty());
}

#[tokio::test]
async fn initial_request_body_is_not_mutated() {
    let filter = test_filter();
    let request = make_request(Method::POST, "/v1/messages");
    let mut ctx = make_filter_context(&request);
    let original = json!({
        "model":"test",
        "max_tokens":32,
        "system":"Be concise",
        "metadata":{"user_id":"demo"},
        "messages":[{"role":"user","content":"search"}]
    });
    let original = Bytes::from(original.to_string());
    let mut body = Some(original.clone());

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(matches!(action, FilterAction::Continue));
    assert_eq!(body, Some(original));
}

#[tokio::test]
async fn pending_search_executes_and_appends_tool_result() {
    let search = start_you_search_stub(200, valid_you_body());
    let filter = test_filter_impl_with_base_url(search.base_url());
    let pending = pending_search("potato");

    let outcome = filter.execute_pending_search(&pending).await;
    let mut rebuilt = base_request();
    append_search_turns(&mut rebuilt, assistant_content("potato"), pending, &outcome).unwrap();

    assert_eq!(rebuilt["model"], "openai/gpt-oss-20b");
    assert_eq!(rebuilt["system"], "Answer with sources.");
    assert_eq!(rebuilt["metadata"]["user_id"], "demo");
    assert_eq!(rebuilt["tools"][0]["name"], "WebSearch");
    assert_eq!(rebuilt["tool_choice"], json!({"type":"auto"}));
    let messages = rebuilt["messages"].as_array().unwrap();
    assert_eq!(messages[messages.len() - 2]["role"], "assistant");
    assert_eq!(messages[messages.len() - 1]["content"][0]["type"], "tool_result");
    assert_eq!(
        messages[messages.len() - 1]["content"][0]["tool_use_id"],
        "toolu_search_1"
    );
    assert!(
        messages[messages.len() - 1]["content"][0]["content"]
            .as_str()
            .unwrap()
            .contains("Potato - Wikipedia")
    );
    assert!(
        messages[messages.len() - 1]["content"][0].get("is_error").is_none(),
        "a successful search must not mark the tool result as an error"
    );
    assert_eq!(search.last_json()["query"], "potato");
    assert!(
        search
            .last_request()
            .to_ascii_lowercase()
            .contains("x-api-key: test-key")
    );
}

#[tokio::test]
async fn provider_failure_appends_is_error_tool_result() {
    let search = start_you_search_stub(503, "unavailable".to_owned());
    let filter = test_filter_impl_with_base_url(search.base_url());
    let pending = pending_search("potato");

    let outcome = filter.execute_pending_search(&pending).await;
    assert!(
        matches!(&outcome, SearchOutcome::Failed),
        "a provider 5xx must map to a failed outcome, got {outcome:?}"
    );

    let mut rebuilt = base_request();
    append_search_turns(&mut rebuilt, assistant_content("potato"), pending, &outcome).unwrap();

    let result_block = &rebuilt["messages"].as_array().unwrap().last().unwrap()["content"][0];
    assert_eq!(result_block["type"], "tool_result");
    assert_eq!(result_block["tool_use_id"], "toolu_search_1");
    assert_eq!(result_block["content"], "Web search unavailable.");
    assert_eq!(
        result_block["is_error"], true,
        "a failed search must mark the tool result with is_error"
    );
}

#[tokio::test]
async fn empty_results_appends_no_results_tool_result() {
    let search = start_you_search_stub(200, empty_you_body());
    let filter = test_filter_impl_with_base_url(search.base_url());
    let pending = pending_search("potato");

    let outcome = filter.execute_pending_search(&pending).await;
    assert!(
        matches!(&outcome, SearchOutcome::Results(results) if results.is_empty()),
        "a successful empty search must be a zero-result outcome, got {outcome:?}"
    );

    let mut rebuilt = base_request();
    append_search_turns(&mut rebuilt, assistant_content("potato"), pending, &outcome).unwrap();

    let result_block = &rebuilt["messages"].as_array().unwrap().last().unwrap()["content"][0];
    assert_eq!(result_block["content"], "No search results found.");
    assert!(
        result_block.get("is_error").is_none(),
        "a successful empty search must not mark the tool result as an error"
    );
}

/// Encode one native Messages SSE event as raw response bytes.
fn sse_event(event_type: &str, data: &Value) -> Bytes {
    Bytes::from(format!("event: {event_type}\ndata: {data}\n\n"))
}

fn sse_message_start(id: &str) -> Bytes {
    sse_event(
        "message_start",
        &json!({
            "type": "message_start",
            "message": {
                "id": id, "type": "message", "role": "assistant", "model": "test",
                "content": [], "stop_reason": null, "stop_sequence": null,
                "usage": {"input_tokens": 10, "output_tokens": 0}
            }
        }),
    )
}

fn sse_text_block(index: u64, text: &str) -> Bytes {
    let start = json!({"type":"content_block_start","index":index,"content_block":{"type":"text","text":""}});
    let delta = json!({"type":"content_block_delta","index":index,"delta":{"type":"text_delta","text":text}});
    let stop = json!({"type":"content_block_stop","index":index});
    Bytes::from(format!(
        "event: content_block_start\ndata: {start}\n\nevent: content_block_delta\ndata: {delta}\n\nevent: content_block_stop\ndata: {stop}\n\n"
    ))
}

fn sse_web_search_block(index: u64, id: &str, query: &str) -> Bytes {
    let partial = json!({"query": query}).to_string();
    let start = json!({"type":"content_block_start","index":index,"content_block":{"type":"tool_use","id":id,"name":"WebSearch","input":{}}});
    let delta =
        json!({"type":"content_block_delta","index":index,"delta":{"type":"input_json_delta","partial_json":partial}});
    let stop = json!({"type":"content_block_stop","index":index});
    Bytes::from(format!(
        "event: content_block_start\ndata: {start}\n\nevent: content_block_delta\ndata: {delta}\n\nevent: content_block_stop\ndata: {stop}\n\n"
    ))
}

fn sse_message_delta(stop_reason: &str, output_tokens: u64) -> Bytes {
    sse_event(
        "message_delta",
        &json!({
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": null},
            "usage": {"output_tokens": output_tokens}
        }),
    )
}

fn sse_message_stop() -> Bytes {
    sse_event("message_stop", &json!({"type": "message_stop"}))
}

/// The upstream round's closing `message_delta` and `message_stop` frames.
fn sse_terminal(stop_reason: &str, output_tokens: u64) -> Bytes {
    let mut bytes = sse_message_delta(stop_reason, output_tokens).to_vec();
    bytes.extend_from_slice(&sse_message_stop());
    Bytes::from(bytes)
}

/// A context whose subrequest response transport is the streaming variant.
fn streaming_response_context(request: &Request) -> HttpFilterContext<'_> {
    let mut ctx = make_filter_context(request);
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
    ctx
}

/// Run the response-header phase for one streaming round.
///
/// Mirrors the live IRR flow: `ctx.response_header` is populated only in the
/// header phase, `on_response` runs there, then the router clears the header
/// before the streaming body phase. Any decision `on_response` records must
/// therefore survive into the body phase without re-reading `ctx.response_header`.
async fn run_response_header<'a>(filter: &dyn HttpFilter, ctx: &mut HttpFilterContext<'a>, response: &'a mut Response) {
    ctx.response_header = Some(response);
    let action = filter.on_response(ctx).await.unwrap();
    assert!(matches!(action, FilterAction::Continue), "the header phase continues");
    // The IRR clears the response header before the streaming body phase.
    ctx.response_header = None;
}

/// Feed one response chunk and return the forwarded client-visible bytes.
fn feed_chunk(
    filter: &dyn HttpFilter,
    ctx: &mut HttpFilterContext<'_>,
    chunk: Bytes,
    end_of_stream: bool,
) -> Option<Bytes> {
    let mut body = Some(chunk);
    let action = filter.on_response_body(ctx, &mut body, end_of_stream).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "streaming forwarding continues the pipeline"
    );
    body
}

#[test]
fn terminal_streaming_forwards_message_start_and_suppresses_search_before_end_of_stream() {
    let filter = terminal_streaming_filter();
    let request = make_request(Method::POST, "/v1/messages");
    let mut ctx = streaming_response_context(&request);
    // One pre-terminal chunk carrying the lifecycle opener and a managed call.
    let mut chunk = sse_message_start("msg_1").to_vec();
    chunk.extend_from_slice(&sse_web_search_block(0, "toolu_ws", "potato"));

    let forwarded = feed_chunk(&*filter, &mut ctx, Bytes::from(chunk), false);

    let forwarded = String::from_utf8(forwarded.expect("message_start reaches the client").to_vec()).unwrap();
    assert!(
        forwarded.contains("event: message_start"),
        "the terminal round streams to the client before the upstream completes"
    );
    assert!(
        forwarded.contains("\"id\":\"msg_1\""),
        "the client-visible message id is forwarded"
    );
    assert!(
        !forwarded.contains("tool_use") && !forwarded.contains("WebSearch"),
        "the managed WebSearch block is suppressed rather than echoed"
    );
    assert!(
        result_action(&ctx).is_none(),
        "no loop decision is published mid-stream"
    );
}

#[test]
fn terminal_streaming_text_answer_emits_terminal_and_signals_done() {
    let filter = terminal_streaming_filter();
    let request = make_request(Method::POST, "/v1/messages");
    let mut ctx = streaming_response_context(&request);

    feed_chunk(&*filter, &mut ctx, sse_message_start("msg_1"), false);
    let mid = feed_chunk(
        &*filter,
        &mut ctx,
        sse_text_block(0, "Potatoes grow underground."),
        false,
    );
    // The round's `message_delta` is captured pre-terminal; only `message_stop`
    // arrives at end_of_stream, so a plain echo could not carry the terminal
    // `message_delta` the client-visible lifecycle requires.
    feed_chunk(&*filter, &mut ctx, sse_message_delta("end_turn", 7), false);
    let terminal = feed_chunk(&*filter, &mut ctx, sse_message_stop(), true);

    let mid = String::from_utf8(mid.expect("text is forwarded incrementally").to_vec()).unwrap();
    assert!(
        mid.contains("\"text\":\"Potatoes grow underground.\""),
        "text deltas stream to the client"
    );
    let terminal = String::from_utf8(terminal.expect("terminal frames are emitted").to_vec()).unwrap();
    assert!(
        terminal.contains("event: message_delta"),
        "the terminal message_delta is emitted once"
    );
    assert!(
        terminal.contains("\"output_tokens\":7"),
        "the aggregated output tokens are carried"
    );
    assert!(
        terminal.contains("event: message_stop"),
        "the terminal message_stop closes the lifecycle"
    );
    assert_eq!(
        result_action(&ctx).as_deref(),
        Some("done"),
        "a plain text answer finishes the loop"
    );
}

#[test]
fn terminal_streaming_managed_web_search_suppresses_block_and_signals_loop() {
    let filter = terminal_streaming_filter();
    let request = make_request(Method::POST, "/v1/messages");
    let mut ctx = streaming_response_context(&request);

    feed_chunk(&*filter, &mut ctx, sse_message_start("msg_1"), false);
    let suppressed = feed_chunk(&*filter, &mut ctx, sse_web_search_block(0, "toolu_ws", "potato"), false);
    let terminal = feed_chunk(&*filter, &mut ctx, sse_terminal("tool_use", 5), true);

    assert!(
        suppressed.is_none(),
        "the managed WebSearch tool_use block is fully suppressed"
    );
    assert!(terminal.is_none(), "no terminal frames are emitted on a managed round");
    assert_eq!(
        result_action(&ctx).as_deref(),
        Some("loop"),
        "a managed WebSearch call re-enters inference"
    );
}

#[test]
fn terminal_streaming_fails_closed_on_malformed_frame() {
    let filter = terminal_streaming_filter();
    let request = make_request(Method::POST, "/v1/messages");
    let mut ctx = streaming_response_context(&request);

    feed_chunk(&*filter, &mut ctx, sse_message_start("msg_1"), false);
    // A `content_block_start` without an index is malformed.
    let malformed = sse_event(
        "content_block_start",
        &json!({"type": "content_block_start", "content_block": {"type": "text"}}),
    );
    let out = feed_chunk(&*filter, &mut ctx, malformed, false);

    let out = String::from_utf8(out.expect("a terminal error event is emitted").to_vec()).unwrap();
    assert!(
        out.starts_with("event: error"),
        "a fail-closed error event replaces the stream"
    );
    assert!(
        out.contains("\"type\":\"api_error\""),
        "malformed upstream framing maps to api_error"
    );
    assert_eq!(
        result_action(&ctx).as_deref(),
        Some("done"),
        "a stream failure ends the loop"
    );

    // No raw bytes leak after the failure.
    let after = feed_chunk(&*filter, &mut ctx, sse_text_block(0, "late"), true);
    assert!(after.is_none(), "no bytes are forwarded after the stream fails closed");
}

#[tokio::test]
async fn terminal_streaming_passes_through_non_success_upstream() {
    let filter = terminal_streaming_filter();
    let request = make_request(Method::POST, "/v1/messages");
    let mut ctx = streaming_response_context(&request);
    let mut response = make_response();
    response.status = http::StatusCode::TOO_MANY_REQUESTS;
    run_response_header(&*filter, &mut ctx, &mut response).await;
    let original = Bytes::from_static(br#"{"type":"error","error":{"type":"overloaded_error","message":"busy"}}"#);

    let mut body = Some(original.clone());
    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();

    assert!(matches!(action, FilterAction::Continue));
    assert_eq!(
        body,
        Some(original),
        "a non-2xx upstream stream is passed through untouched"
    );
    assert_eq!(
        result_action(&ctx).as_deref(),
        Some("done"),
        "a non-success response ends the loop"
    );
}

#[tokio::test]
async fn on_request_strips_accept_encoding() {
    // A compressed response cannot be parsed into Messages SSE events, nor
    // classified as a buffered JSON body: strip the client's content-coding
    // negotiation so every backend round returns identity-encoded bytes.
    for filter in [test_filter(), terminal_streaming_filter()] {
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);

        let action = filter.on_request(&mut ctx).await.unwrap();

        assert!(matches!(action, FilterAction::Continue));
        assert!(
            ctx.request_headers_to_remove.contains(&ACCEPT_ENCODING),
            "the web-search loop must strip accept-encoding before the backend round"
        );
    }
}

#[tokio::test]
async fn terminal_streaming_declines_content_encoded_response() {
    let filter = terminal_streaming_filter();
    let request = make_request(Method::POST, "/v1/messages");
    let mut ctx = streaming_response_context(&request);
    let mut response = make_response();
    response
        .headers
        .insert(CONTENT_ENCODING, HeaderValue::from_static("gzip"));
    run_response_header(&*filter, &mut ctx, &mut response).await;
    // Opaque compressed bytes that are not a Messages SSE lifecycle; parsing
    // them as UTF-8 events would leak raw bytes into a transformed stream.
    let original = Bytes::from_static(&[0x1F, 0x8B, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00]);

    let mut body = Some(original.clone());
    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();

    assert!(matches!(action, FilterAction::Continue));
    assert_eq!(
        body,
        Some(original),
        "a content-encoded response is passed through untouched, never parsed as SSE"
    );
    assert_eq!(
        result_action(&ctx).as_deref(),
        Some("done"),
        "a content-encoded response ends the loop"
    );
}

#[tokio::test]
async fn terminal_streaming_fails_closed_on_later_round_non_success() {
    // Round 0 forwards `message_start`, so the client is mid-stream on a
    // committed 200 SSE lifecycle. A later round that returns a non-2xx status
    // cannot have its raw error body dumped into the open stream: doing so would
    // corrupt it. The stream fails closed to one terminal error event instead.
    let filter = terminal_streaming_filter();
    let request = make_request(Method::POST, "/v1/messages");
    let mut ctx = streaming_response_context(&request);

    let mut ok = make_response();
    run_response_header(&*filter, &mut ctx, &mut ok).await;
    let started = feed_chunk(&*filter, &mut ctx, sse_message_start("msg_1"), false);
    assert!(
        String::from_utf8(started.expect("message_start reaches the client").to_vec())
            .unwrap()
            .contains("event: message_start"),
        "the transformed lifecycle has started before the later round arrives"
    );

    let mut response = make_response();
    response.status = http::StatusCode::TOO_MANY_REQUESTS;
    run_response_header(&*filter, &mut ctx, &mut response).await;
    let raw = Bytes::from_static(br#"{"type":"error","error":{"type":"overloaded_error","message":"busy"}}"#);

    let out = feed_chunk(&*filter, &mut ctx, raw, true);

    let out = String::from_utf8(out.expect("a terminal error event is emitted").to_vec()).unwrap();
    assert!(
        out.starts_with("event: error"),
        "a later-round non-success response fails closed to a terminal error event"
    );
    assert!(
        out.contains("\"type\":\"api_error\""),
        "an untransformable later round maps to api_error"
    );
    assert!(
        !out.contains("overloaded_error"),
        "no raw upstream error body leaks into the already-open transformed stream"
    );
    assert_eq!(
        result_action(&ctx).as_deref(),
        Some("done"),
        "an untransformable later round ends the loop"
    );
}

#[tokio::test]
async fn terminal_streaming_fails_closed_on_later_round_content_encoded() {
    // A later round that returns a content-encoded body cannot be parsed as
    // Messages SSE; forwarding the compressed bytes into the open stream would
    // corrupt it, so the stream fails closed to one terminal error event.
    let filter = terminal_streaming_filter();
    let request = make_request(Method::POST, "/v1/messages");
    let mut ctx = streaming_response_context(&request);

    let mut ok = make_response();
    run_response_header(&*filter, &mut ctx, &mut ok).await;
    feed_chunk(&*filter, &mut ctx, sse_message_start("msg_1"), false);

    let mut response = make_response();
    response
        .headers
        .insert(CONTENT_ENCODING, HeaderValue::from_static("gzip"));
    run_response_header(&*filter, &mut ctx, &mut response).await;
    let compressed = Bytes::from_static(&[0x1F, 0x8B, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00]);

    let out = feed_chunk(&*filter, &mut ctx, compressed, true);

    let out = String::from_utf8(out.expect("a terminal error event is emitted").to_vec()).unwrap();
    assert!(
        out.starts_with("event: error"),
        "a later-round content-encoded response fails closed to a terminal error event"
    );
    assert!(
        out.contains("\"type\":\"api_error\""),
        "an untransformable later round maps to api_error"
    );
    assert_eq!(
        result_action(&ctx).as_deref(),
        Some("done"),
        "an untransformable later round ends the loop"
    );
}

#[tokio::test]
async fn on_response_marks_non_success_round_untransformable() {
    let filter = terminal_streaming_filter();
    let request = make_request(Method::POST, "/v1/messages");
    let mut ctx = streaming_response_context(&request);
    let mut response = make_response();
    response.status = http::StatusCode::TOO_MANY_REQUESTS;

    run_response_header(&*filter, &mut ctx, &mut response).await;

    assert!(
        ctx.extensions.get::<UntransformableRound>().is_some(),
        "a non-2xx header marks the round untransformable for the body phase"
    );
}

#[tokio::test]
async fn on_response_marks_content_encoded_round_untransformable() {
    let filter = terminal_streaming_filter();
    let request = make_request(Method::POST, "/v1/messages");
    let mut ctx = streaming_response_context(&request);
    let mut response = make_response();
    response
        .headers
        .insert(CONTENT_ENCODING, HeaderValue::from_static("gzip"));

    run_response_header(&*filter, &mut ctx, &mut response).await;

    assert!(
        ctx.extensions.get::<UntransformableRound>().is_some(),
        "a content-encoded header marks the round untransformable for the body phase"
    );
}

#[tokio::test]
async fn on_response_clears_stale_marker_on_success_round() {
    // A prior untransformable round must not leak its marker into a later
    // transformable round: each header phase re-decides afresh.
    let filter = terminal_streaming_filter();
    let request = make_request(Method::POST, "/v1/messages");
    let mut ctx = streaming_response_context(&request);

    let mut failing = make_response();
    failing.status = http::StatusCode::TOO_MANY_REQUESTS;
    run_response_header(&*filter, &mut ctx, &mut failing).await;
    assert!(
        ctx.extensions.get::<UntransformableRound>().is_some(),
        "the failing round sets the marker"
    );

    let mut ok = make_response();
    run_response_header(&*filter, &mut ctx, &mut ok).await;
    assert!(
        ctx.extensions.get::<UntransformableRound>().is_none(),
        "a subsequent 2xx identity round clears the stale marker"
    );
}

#[tokio::test]
async fn on_response_does_not_mark_for_buffered_filter() {
    // The buffered loop reads the response status directly in the body phase, so
    // the non-streaming filter records no streaming marker.
    let filter = test_filter();
    let request = make_request(Method::POST, "/v1/messages");
    let mut ctx = make_filter_context(&request);
    let mut response = make_response();
    response.status = http::StatusCode::TOO_MANY_REQUESTS;

    run_response_header(&*filter, &mut ctx, &mut response).await;

    assert!(
        ctx.extensions.get::<UntransformableRound>().is_none(),
        "the buffered filter records no streaming marker"
    );
}

#[test]
fn termination_deadline_maps_to_deadline_error() {
    // A transport-level deadline reuses the deadline message so the client sees a
    // consistent reason whether the deadline lapses before re-entry or mid-stream.
    assert_eq!(
        termination_stream_error(StreamTerminationCause::DeadlineExceeded),
        streaming::StreamError::DeadlineExceeded,
    );
}

#[test]
fn termination_transport_faults_map_to_upstream_terminated() {
    // Every non-deadline abnormal cause folds into one upstream-terminated error.
    for cause in [
        StreamTerminationCause::AdmissionTimeout,
        StreamTerminationCause::CircuitOpen,
        StreamTerminationCause::Connect,
        StreamTerminationCause::IdleTimeout,
        StreamTerminationCause::Io,
        StreamTerminationCause::Filter,
        StreamTerminationCause::ResponseTooLarge,
    ] {
        assert_eq!(
            termination_stream_error(cause),
            streaming::StreamError::UpstreamTerminated,
            "{cause:?} folds into an upstream-terminated error",
        );
    }
}

#[test]
fn handle_stream_termination_emits_terminal_error_and_signals_done() {
    // An abnormal termination during the completion hook is converted into one
    // terminal error event and the loop ends, so the client gets a coherent close
    // instead of an abrupt EOF.
    let request = make_request(Method::POST, "/v1/messages");
    let mut ctx = streaming_response_context(&request);
    let mut body = None;

    let action = handle_stream_termination(&mut ctx, &mut body, StreamTerminationCause::Io).unwrap();

    assert!(matches!(action, FilterAction::Continue));
    let bytes = body.expect("a terminal error event is emitted for an abnormal termination");
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    assert!(
        text.contains("event: error"),
        "exactly one terminal error event: {text}"
    );
    assert!(
        text.contains("\"type\":\"api_error\""),
        "a transport fault maps to api_error: {text}"
    );
    assert_eq!(
        result_action(&ctx).as_deref(),
        Some("done"),
        "the loop ends after an abnormal termination"
    );
}

#[test]
fn terminal_streaming_buffered_round_still_classifies() {
    // Under `terminal_streaming` a non-streaming (buffered) round keeps the
    // buffered classify behavior: mode stays Buffered, and a full JSON body is
    // classified at end_of_stream.
    let filter = terminal_streaming_filter();
    let request = make_request(Method::POST, "/v1/messages");
    let mut ctx = make_filter_context(&request);
    let original = message_response(
        json!([{"type":"tool_use","id":"toolu_search_1","name":"WebSearch","input":{"query":"potato"}}]),
        "tool_use",
    );
    let mut body = Some(original.clone());

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();

    assert!(matches!(action, FilterAction::Continue));
    assert_eq!(
        result_action(&ctx).as_deref(),
        Some("loop"),
        "a buffered managed round still loops"
    );
    assert_eq!(body, Some(original), "the buffered body is classified, not rewritten");
}

#[test]
fn terminal_streaming_reassembles_fragmented_message_start() {
    let filter = terminal_streaming_filter();
    let request = make_request(Method::POST, "/v1/messages");
    let mut ctx = streaming_response_context(&request);
    let full = sse_message_start("msg_frag");
    let split = full.len() / 2;

    let first = feed_chunk(&*filter, &mut ctx, full.slice(..split), false);
    let second = feed_chunk(&*filter, &mut ctx, full.slice(split..), false);

    assert!(first.is_none(), "an incomplete event is buffered, not forwarded");
    let second = String::from_utf8(second.expect("the completed event is forwarded").to_vec()).unwrap();
    assert!(
        second.contains("event: message_start"),
        "the reassembled message_start reaches the client"
    );
    assert!(
        second.contains("\"id\":\"msg_frag\""),
        "the reassembled payload is intact"
    );
}

#[test]
fn accounted_previous_response_recovers_complete_assistant_content() {
    let content = json!([
        {"type":"text","text":"I will search first."},
        {"type":"tool_use","id":"toolu_search_1","name":"WebSearch","input":{"query":"potato"}}
    ]);
    let response = message_response(content.clone(), "tool_use");

    let (pending, recovered) = managed_search_from_response(&response).unwrap();

    assert_eq!(pending.id, "toolu_search_1");
    assert_eq!(pending.query, "potato");
    assert_eq!(recovered, content.as_array().unwrap().clone());
}

/// Feed a managed `WebSearch` round into a fresh logical stream, returning the
/// terminal `drive_streaming_chunk` output and published action.
fn drive_managed_round(reentry: Reentry) -> (String, Option<&'static str>) {
    let mut logical = streaming::LogicalStream::new(1 << 20, 1 << 20);
    let (start, start_action) = drive_streaming_chunk(&mut logical, &sse_message_start("msg_1"), false, reentry);
    assert!(
        String::from_utf8(start).unwrap().contains("event: message_start"),
        "round 0 forwards message_start"
    );
    assert!(start_action.is_none(), "no loop decision is published mid-stream");

    let (suppressed, suppressed_action) = drive_streaming_chunk(
        &mut logical,
        &sse_web_search_block(0, "toolu_ws", "potato"),
        false,
        reentry,
    );
    assert!(
        suppressed.is_empty(),
        "the managed WebSearch block is suppressed pre-terminal"
    );
    assert!(suppressed_action.is_none(), "no loop decision is published mid-stream");

    let (output, action) = drive_streaming_chunk(&mut logical, &sse_terminal("tool_use", 5), true, reentry);
    (String::from_utf8(output).unwrap(), action)
}

#[test]
fn drive_streaming_chunk_terminates_at_iteration_ceiling() {
    // Re-entry would push the router past its iteration ceiling; opening the
    // next round would fail and abort the stream with an abrupt EOF. A managed
    // round at the ceiling must instead emit a coherent terminal error.
    let (output, action) = drive_managed_round(Reentry::IterationCeiling);

    assert!(
        output.starts_with("event: error"),
        "a managed round at the ceiling emits a terminal error, not an abrupt EOF"
    );
    assert!(
        output.contains("maximum number of search iterations"),
        "the client-safe iteration-limit reason is carried"
    );
    assert!(
        output.contains("\"type\":\"api_error\""),
        "an exhausted loop maps to api_error"
    );
    assert_eq!(
        action,
        Some(ACTION_DONE),
        "the loop ends rather than re-entering past the ceiling"
    );
}

#[test]
fn drive_streaming_chunk_loops_when_reentry_available() {
    // With headroom below the ceiling the managed round re-enters inference and
    // emits no terminal or error frames.
    let (output, action) = drive_managed_round(Reentry::Available);

    assert!(
        output.is_empty(),
        "a managed round below the ceiling emits no terminal or error frames"
    );
    assert_eq!(
        action,
        Some(ACTION_LOOP),
        "re-entry is selected when the iteration ceiling has headroom"
    );
}

#[test]
fn drive_streaming_chunk_terminates_when_deadline_exhausted() {
    // The router deadline has elapsed; opening the next round would fail and
    // abort the stream with an abrupt EOF. A managed round past the deadline must
    // instead emit a coherent terminal deadline error.
    let (output, action) = drive_managed_round(Reentry::DeadlineExceeded);

    assert!(
        output.starts_with("event: error"),
        "a managed round past the deadline emits a terminal error, not an abrupt EOF"
    );
    assert!(
        output.contains("exceeded the configured deadline"),
        "the client-safe deadline reason is carried"
    );
    assert!(
        output.contains("\"type\":\"api_error\""),
        "an exhausted deadline maps to api_error"
    );
    assert_eq!(
        action,
        Some(ACTION_DONE),
        "the loop ends rather than re-entering past the deadline"
    );
}

#[test]
fn reentry_from_state_available_with_headroom_and_future_deadline() {
    let now = Instant::now();
    let deadline = now + std::time::Duration::from_secs(60);

    assert_eq!(
        reentry_from_state(0, 5, deadline, now),
        Reentry::Available,
        "iteration headroom and a future deadline permit re-entry"
    );
}

#[test]
fn reentry_from_state_reports_iteration_ceiling() {
    let now = Instant::now();
    let deadline = now + std::time::Duration::from_secs(60);

    // iteration is zero-indexed and pre-increment: 4 -> next round would be 5,
    // which meets max_iterations.
    assert_eq!(
        reentry_from_state(4, 5, deadline, now),
        Reentry::IterationCeiling,
        "re-entry that would meet the iteration ceiling is refused"
    );
}

#[test]
fn reentry_from_state_reports_deadline_exceeded() {
    let now = Instant::now();

    // A deadline equal to now leaves zero remaining time, mirroring the IRR's
    // open_step refusal.
    assert_eq!(
        reentry_from_state(0, 5, now, now),
        Reentry::DeadlineExceeded,
        "an elapsed deadline refuses re-entry even with iteration headroom"
    );
}

#[test]
fn reentry_from_state_prefers_iteration_ceiling_over_deadline() {
    let now = Instant::now();

    // Both the iteration ceiling and the deadline are exhausted; the IRR checks
    // the iteration ceiling first, so that reason wins.
    assert_eq!(
        reentry_from_state(4, 5, now, now),
        Reentry::IterationCeiling,
        "the iteration ceiling takes precedence over an elapsed deadline"
    );
}
