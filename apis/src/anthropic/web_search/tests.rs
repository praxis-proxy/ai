// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

use std::{
    io::{Read as _, Write as _},
    net::{TcpListener, TcpStream},
    sync::{Arc, Mutex},
};

use bytes::Bytes;
use http::Method;
use praxis_filter::{FilterAction, HttpFilter, HttpFilterContext, Request};
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
