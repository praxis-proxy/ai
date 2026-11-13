// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Cross-filter request-body contracts at the actual upstream HTTP receiver.
//!
//! Same-format writers declare the fields they own. Everything else must keep
//! its values and member/array order; no-op paths must keep exact body bytes.
//! The inference fixture sweep separately covers translated request shapes.

#[cfg(feature = "openai-responses")]
mod active;
mod inventory;
#[cfg(all(
    feature = "openai-all",
    feature = "store-sqlite",
    feature = "vertex-anthropic-filter"
))]
mod translations;

use std::time::Duration;

use praxis_core::config::Config;
use praxis_test_utils::{
    Backend, HttpBackendEvent, HttpServerAction, StatefulCapturingBackend,
    body_contract::{assert_body_unchanged, assert_json_changes_only},
    free_port, start_proxy, start_scripted_http_backend,
};
use serde_json::{Value, json};
use tokio::{
    io::{AsyncReadExt as _, AsyncWriteExt as _},
    net::TcpStream,
};

/// Noncanonical key order, nested tool schemas, unknown fields, escapes,
/// Unicode and two distinct tool/message positions make serializer changes
/// observable. This literal is sent directly, never through a JSON serializer.
const BODY: &str = r#"{
  "model" : "public-model",
  "tools": [
    {"type":"function","function":{"name":"zebra","description":"caf\u00e9 ☕","parameters":{"type":"object","properties":{"z":{"type":"string"},"a":{"type":"number"}}}}},
    {"type":"function","function":{"name":"alpha","parameters":{"type":"object","properties":{"second":{"type":"boolean"},"first":{"type":"string"}}}}}
  ],
  "messages" : [{"content":"first\nmessage","role":"user"},{"content":"second message","role":"assistant"}],
  "stream": true,
  "extra": {"z":1,"a":["second","first"]}
}
"#;

/// A valid Responses request requiring no history, file or tool expansion.
#[cfg(feature = "openai-responses")]
const RESPONSES_BODY: &str = r#"{
  "model" : "public-model",
  "input" : "caf\u00e9 ☕\nhello",
  "tools": [{"type":"function","name":"zebra","parameters":{"type":"object","properties":{"z":{"type":"string"},"a":{"type":"number"}}}}],
  "metadata": {"z":"last","a":"first"},
  "store": false,
  "stream": false
}
"#;

/// Stateless broker configuration used by its mutation case.
const MCP_BROKER: &str = "- filter: mcp\n  path: /mcp\n  protocol_profile: stateless\n  servers:\n    - name: weather\n      cluster: backend\n      path: /mcp\n      tool_prefix: weather_\n      tools:\n        - name: inspect";

/// A same-format writer's permitted changes and expected result.
struct MutationCase {
    /// Pipeline fragment under test.
    filters: &'static str,
    /// Client and upstream request path.
    path: &'static str,
    /// Original request bytes, including their lexical JSON representation.
    body: String,
    /// JSON subtrees owned by this mutation.
    changes: &'static [&'static str],
    /// Location at which to verify the intended mutation.
    changed_pointer: &'static str,
    /// Replacement value, or absence for field removal.
    expected_value: Option<Value>,
}

/// Routing with an explicit model alias and a fallback for no-op cases.
const MODEL_PROVIDER: &str = "- filter: model_to_provider\n  fallback_upstream: backend\n  models:\n    - model: public-model\n      upstream: backend\n      target_model: target-model\n      paths: [/v1/chat/completions]";
/// Qualified model resolution after extracting the client model.
const LLMISVC: &str = "- filter: model_to_header\n- filter: llmisvc_model_provider_resolver";
/// Message insertion at both ends of the existing conversation.
const ENRICH: &str = "- filter: prompt_enrich\n  prepend:\n    - role: system\n      content: policy\n  append:\n    - role: user\n      content: reminder";
/// Azure's removal of the ignored body model field.
const AZURE: &str = "- filter: openai_chat_completions_to_azureai_chat_completions";
/// Model replacement independent of provider selection.
const MODEL_REWRITE: &str =
    "- filter: openai_responses_model_rewrite\n  model_aliases:\n    public-model: target-model";

/// The loop requires a bounded IRR step even for requests without Responses state.
#[cfg(feature = "openai-agentic-loop")]
const AGENTIC_BYPASS: &str = r#"- filter: iterative_request_router
  initial_step: inference
  max_iterations: 1
  steps:
    - name: inference
      filters:
        - filter: openai_agentic_loop
        - filter: router
          routes:
            - path_prefix: /
              cluster: backend
        - filter: load_balancer
          clusters:
            - name: backend
              endpoints: ['127.0.0.1:BACKEND_PORT']
      on_result:
        - default: true
          done: true"#;

/// The mutation configurations executed by the capture suite and its inventory gate.
fn mutation_cases() -> Vec<MutationCase> {
    let mut cases = vec![
        MutationCase {
            filters: "- filter: stream_usage_inject",
            path: "/v1/chat/completions",
            body: BODY.to_owned(),
            changes: &["/stream_options"],
            changed_pointer: "/stream_options/include_usage",
            expected_value: Some(json!(true)),
        },
        MutationCase {
            filters: "- filter: stream_usage_inject",
            path: "/v1/chat/completions",
            body: BODY.replace(
                "\"stream\": true",
                "\"stream\": true, \"stream_options\": {\"extra\":2,\"include_usage\":false}",
            ),
            changes: &["/stream_options/include_usage"],
            changed_pointer: "/stream_options/include_usage",
            expected_value: Some(json!(true)),
        },
        MutationCase {
            filters: MODEL_PROVIDER,
            path: "/v1/chat/completions",
            body: BODY.to_owned(),
            changes: &["/model"],
            changed_pointer: "/model",
            expected_value: Some(json!("target-model")),
        },
        MutationCase {
            filters: LLMISVC,
            path: "/v1/chat/completions",
            body: BODY.replace("public-model", "publishers/test/models/target-model"),
            changes: &["/model"],
            changed_pointer: "/model",
            expected_value: Some(json!("target-model")),
        },
        MutationCase {
            filters: AZURE,
            path: "/v1/chat/completions",
            body: BODY.to_owned(),
            changes: &["/model"],
            changed_pointer: "/model",
            expected_value: None,
        },
    ];
    cases.push(MutationCase {
        filters: MODEL_REWRITE,
        path: "/v1/chat/completions",
        body: BODY.to_owned(),
        changes: &["/model"],
        changed_pointer: "/model",
        expected_value: Some(json!("target-model")),
    });

    cases
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn same_format_mutations_change_only_owned_fields() {
    let cases = mutation_cases();
    for case in cases {
        for chunked in [false, true] {
            let forwarded = forward(case.filters, case.path, case.body.as_bytes(), chunked).await;
            assert_json_changes_only(case.body.as_bytes(), &forwarded, case.changes);
            let value: Value = serde_json::from_slice(&forwarded).expect("upstream body must be JSON");
            assert_eq!(
                value.pointer(case.changed_pointer),
                case.expected_value.as_ref(),
                "filter must perform its declared mutation: {}",
                case.filters
            );
        }
    }
}

/// The exact-byte configurations executed by the capture suite and its inventory gate.
fn no_op_cases() -> Vec<(&'static str, &'static str, String)> {
    let mut cases = vec![
        (
            "- filter: stream_usage_inject",
            "/v1/chat/completions",
            BODY.replace("\"stream\": true", "\"stream\": false"),
        ),
        (
            "- filter: stream_usage_inject",
            "/v1/chat/completions",
            BODY.replace(
                "\"stream\": true",
                "\"stream\": true, \"stream_options\": {\"include_usage\":true}",
            ),
        ),
        ("- filter: stream_usage_inject", "/v1/responses", BODY.to_owned()),
        (
            MODEL_PROVIDER,
            "/v1/chat/completions",
            BODY.replace("public-model", "unmapped-model"),
        ),
        (MODEL_PROVIDER, "/v1/messages", BODY.to_owned()),
        (LLMISVC, "/v1/chat/completions", BODY.to_owned()),
        (ENRICH, "/v1/embeddings", BODY.replace("\"messages\"", "\"input\"")),
        (
            AZURE,
            "/v1/chat/completions",
            BODY.replace("  \"model\" : \"public-model\",\n", ""),
        ),
    ];
    cases.push((
        MODEL_REWRITE,
        "/v1/chat/completions",
        BODY.replace("public-model", "unmapped-model"),
    ));
    cases
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn no_op_mutators_forward_exact_original_bytes() {
    let cases = no_op_cases();
    for (filters, path, body) in cases {
        for chunked in [false, true] {
            let forwarded = forward(filters, path, body.as_bytes(), chunked).await;
            assert_body_unchanged(body.as_bytes(), &forwarded);
        }
    }
}

/// Stateful filter configurations executed with the Responses classifier.
#[cfg(feature = "openai-responses")]
fn stateful_filters() -> Vec<&'static str> {
    #[cfg_attr(
        not(any(
            feature = "store",
            feature = "openai-mcp-tools",
            feature = "openai-file-resolve-filter"
        )),
        expect(unused_mut, reason = "optional stateful cases are feature-gated")
    )]
    let mut filters = vec![
        "openai_doc_extract\n  allow_pre_security_callout: true",
        "openai_responses_proxy",
        "openai_client_tool_compat\n- filter: openai_responses_proxy",
        "openai_file_search_dispatch\n  vector_store_url: http://127.0.0.1:9\n- filter: openai_responses_proxy",
        "openai_web_search_dispatch\n  provider: you\n  api_key:\n    value: synthetic-test-key\n  outbound_chain:\n    name: search\n    filters:\n      - filter: request_id\n- filter: openai_responses_proxy",
    ];
    #[cfg(feature = "store")]
    filters.push("openai_responses_rehydrate");
    #[cfg(feature = "openai-file-resolve-filter")]
    filters.push("openai_file_resolve\n  allow_pre_security_callout: true\n  files_api_url: http://127.0.0.1:9");
    #[cfg(feature = "openai-mcp-tools")]
    filters.extend([
        "openai_mcp_tool_resolve",
        "openai_mcp_dispatch\n- filter: openai_responses_proxy",
    ]);
    filters
}

#[cfg(feature = "openai-responses")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stateful_mutators_preserve_bytes_when_no_expansion_is_needed() {
    let filters = stateful_filters();
    for filter in filters {
        let chain = stateful_chain(filter);
        for chunked in [false, true] {
            let forwarded = forward(&chain, "/v1/responses", RESPONSES_BODY.as_bytes(), chunked).await;
            assert!(
                RESPONSES_BODY.as_bytes() == forwarded,
                "unexpected body rewrite by {filter}"
            );
        }
    }
}

/// Match production classification, which consumes the typed operation identity.
#[cfg(feature = "openai-responses")]
fn stateful_chain(filter: &str) -> String {
    format!("- filter: ai_operation\n- filter: openai_responses_request\n- filter: {filter}")
}

/// Conditional adapter configurations executed by the capture suite.
fn conditional_adapter_cases() -> Vec<(&'static str, &'static str)> {
    #[cfg_attr(
        not(any(feature = "vertex-anthropic-filter", feature = "openai-responses")),
        expect(unused_mut, reason = "optional adapter cases are feature-gated")
    )]
    let mut cases = vec![(
        "- filter: anthropic_web_search\n  provider: you\n  api_key:\n    value: synthetic-test-key\n  outbound_chain:\n    name: search\n    filters:\n      - filter: request_id",
        "/v1/messages",
    )];
    #[cfg(feature = "vertex-anthropic-filter")]
    cases.push((
        "- filter: anthropic_messages_to_vertexai_anthropic\n  project: test-project",
        "/v1/messages",
    ));
    #[cfg(feature = "openai-responses")]
    cases.push(("- filter: openai_responses_to_chat_completions", "/v1/chat/completions"));
    #[cfg(feature = "openai-agentic-loop")]
    cases.push((AGENTIC_BYPASS, "/v1/chat/completions"));
    cases
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn conditional_adapters_preserve_bytes_on_bypass() {
    let cases = conditional_adapter_cases();
    let body = BODY.replace("\"stream\": true", "\"stream\": false");
    for (filters, path) in cases {
        for chunked in [false, true] {
            let forwarded = forward(filters, path, body.as_bytes(), chunked).await;
            assert_body_unchanged(body.as_bytes(), &forwarded);
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mcp_broker_changes_only_the_routed_tool_name() {
    let body = r#"{
  "jsonrpc" : "2.0",
  "method" : "tools/call",
  "id": 1,
  "params": {"name":"weather_inspect","arguments":{"z":"caf\u00e9 ☕","a":[2,1]},"_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientInfo":{"name":"body-contract","version":"1"},"io.modelcontextprotocol/clientCapabilities":{}}}
}
"#;
    let headers = [
        ("MCP-Protocol-Version", "2026-07-28"),
        ("Mcp-Method", "tools/call"),
        ("Mcp-Name", "weather_inspect"),
    ];
    for chunked in [false, true] {
        let forwarded = forward_with_headers(MCP_BROKER, "/mcp", body.as_bytes(), chunked, &headers).await;
        assert_json_changes_only(body.as_bytes(), &forwarded, &["/params/name"]);
        let actual: Value = serde_json::from_slice(&forwarded).unwrap();
        assert_eq!(actual["params"]["name"], "inspect");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn enrichment_preserves_original_messages_and_tool_order() {
    for chunked in [false, true] {
        let forwarded = forward(ENRICH, "/v1/chat/completions", BODY.as_bytes(), chunked).await;
        assert_json_changes_only(BODY.as_bytes(), &forwarded, &["/messages"]);
        let original: Value = serde_json::from_str(BODY).expect("original body must be JSON");
        let actual: Value = serde_json::from_slice(&forwarded).expect("forwarded body must be JSON");
        let messages = actual["messages"]
            .as_array()
            .expect("enriched messages must be an array");
        assert_eq!(messages.len(), 4, "enrichment must add exactly two messages");
        assert_eq!(messages[0], json!({"role":"system","content":"policy"}));
        assert_eq!(messages[3], json!({"role":"user","content":"reminder"}));
        // Compare the original messages in their new positions, including
        // object-member order; Value equality alone cannot check that order.
        let retained = json!({"messages": &messages[1..3]});
        let expected = json!({"messages": &original["messages"]});
        assert_json_changes_only(
            &serde_json::to_vec(&expected).unwrap(),
            &serde_json::to_vec(&retained).unwrap(),
            &[],
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn guardrails_preserve_everything_except_redacted_message_content() {
    for (reply, modified) in [
        (r#"{"status":"passed","content":"first\nmessage"}"#, false),
        (r#"{"status":"modified","content":"[MASKED]","rail":"pii"}"#, true),
    ] {
        let nemo = Backend::status(200, reply)
            .header("Content-Type", "application/json")
            .start_with_shutdown();
        let filters = guardrails_filters(nemo.port());
        for chunked in [false, true] {
            let forwarded = forward(&filters, "/v1/chat/completions", BODY.as_bytes(), chunked).await;
            if modified {
                assert_json_changes_only(BODY.as_bytes(), &forwarded, &["/messages/0/content"]);
                let value: Value = serde_json::from_slice(&forwarded).expect("redacted body must be JSON");
                assert_eq!(
                    value["messages"][0]["content"], "[MASKED]",
                    "redaction must reach upstream"
                );
            } else {
                assert_body_unchanged(BODY.as_bytes(), &forwarded);
            }
        }
    }
}

/// The same guardrail configuration is used for capture and inventory.
fn guardrails_filters(port: u16) -> String {
    format!(
        "- filter: ai_operation\n- filter: ai_guardrails\n  provider:\n    type: nemo\n    endpoint: http://127.0.0.1:{port}/v1/checks\n  phase:\n    request: true\n    response: false"
    )
}

/// Send raw bytes through a real pipeline and capture them at its upstream.
async fn forward(filters: &str, path: &str, body: &[u8], chunked: bool) -> bytes::Bytes {
    forward_with_headers(filters, path, body, chunked, &[]).await
}

/// Run raw-body contracts that also need protocol-specific request headers.
async fn forward_with_headers(
    filters: &str,
    path: &str,
    body: &[u8],
    chunked: bool,
    headers: &[(&str, &str)],
) -> bytes::Bytes {
    capture_sequence(
        filters,
        path,
        body,
        chunked,
        headers,
        path,
        vec![HttpServerAction::Json {
            body: "{}".to_owned(),
            status: 200,
        }],
    )
    .await
    .remove(0)
}

/// Capture every actual inference round, including continuations after callouts.
#[expect(
    clippy::too_many_arguments,
    reason = "wire cases independently specify client framing, headers, upstream path, and scripted inference rounds"
)]
async fn capture_sequence(
    filters: &str,
    path: &str,
    body: &[u8],
    chunked: bool,
    headers: &[(&str, &str)],
    upstream_path: &str,
    responses: Vec<HttpServerAction>,
) -> Vec<bytes::Bytes> {
    let expected_requests = responses.len();
    let mut backend = start_scripted_http_backend("POST", upstream_path, responses).await;
    let config = capture_config(filters, backend.port());
    let proxy = start_proxy(&config);
    let response = send_raw(proxy.addr(), path, body, chunked, headers).await;
    assert!(
        response.starts_with(b"HTTP/1.1 200"),
        "body contract request must succeed for {path}, status {}",
        praxis_test_utils::parse_status(&String::from_utf8_lossy(&response))
    );
    let mut requests = Vec::new();
    while requests.len() < expected_requests {
        match tokio::time::timeout(Duration::from_secs(5), backend.next_event())
            .await
            .unwrap()
        {
            Some(HttpBackendEvent::Request(request)) => requests.push(request.body),
            Some(HttpBackendEvent::UnexpectedRequest { method, path }) if method == "GET" && path == "/" => {},
            event => panic!("body contract backend did not capture the request: {event:?}"),
        }
    }
    requests
}

/// Build the same isolated pipeline for single requests and setup-driven cases.
fn capture_config(filters: &str, backend_port: u16) -> Config {
    let terminal = filters.contains("filter: iterative_request_router");
    let router = if terminal || filters.contains("filter: model_to_provider") {
        ""
    } else {
        "      - filter: router\n        routes:\n          - path_prefix: /\n            cluster: backend\n"
    };
    let filters = filters.replace("BACKEND_PORT", &backend_port.to_string());
    let filters = filters
        .lines()
        .map(|line| format!("      {line}\n"))
        .collect::<String>();
    let load_balancer = if terminal {
        String::new()
    } else {
        format!(
            "      - filter: load_balancer\n        clusters:\n          - name: backend\n            endpoints: ['127.0.0.1:{backend_port}']\n"
        )
    };
    Config::from_yaml(&format!(
        "listeners:\n  - name: test\n    address: 127.0.0.1:{}\n    filter_chains: [test]\nfilter_chains:\n  - name: test\n    filters:\n{filters}{router}{load_balancer}insecure_options:\n  allow_private_endpoints: true\n  allow_private_upstreams: true\n",
        free_port()
    )).expect("body contract pipeline must parse")
}

/// Send bytes without invoking a client JSON serializer.
async fn send_raw(address: &str, path: &str, body: &[u8], chunked: bool, headers: &[(&str, &str)]) -> Vec<u8> {
    let mut stream = TcpStream::connect(address)
        .await
        .expect("body contract proxy must accept requests");
    let framing = if chunked {
        "Transfer-Encoding: chunked".to_owned()
    } else {
        format!("Content-Length: {}", body.len())
    };
    let extra_headers = headers
        .iter()
        .map(|(name, value)| format!("{name}: {value}\r\n"))
        .collect::<String>();
    let head = format!(
        "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n{framing}\r\n{extra_headers}Connection: close\r\n\r\n"
    );
    stream.write_all(head.as_bytes()).await.unwrap();
    if chunked {
        // HTTP chunk boundaries deliberately cut through JSON tokens and may
        // split multibyte UTF-8. The upstream body must be independent of them.
        for chunk in body.chunks(13) {
            stream
                .write_all(format!("{:x}\r\n", chunk.len()).as_bytes())
                .await
                .unwrap();
            stream.write_all(chunk).await.unwrap();
            stream.write_all(b"\r\n").await.unwrap();
        }
        stream.write_all(b"0\r\n\r\n").await.unwrap();
    } else {
        stream.write_all(body).await.unwrap();
    }
    let mut response = Vec::new();
    tokio::time::timeout(Duration::from_secs(10), stream.read_to_end(&mut response))
        .await
        .unwrap()
        .unwrap();
    response
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn anthropic_web_search_appends_only_messages_and_retains_tool_schemas() {
    let tools = r#"[{"name":"zebra","input_schema":{"type":"object","properties":{"z":{"type":"string"},"a":{"type":"number"}}}},{"name":"alpha","input_schema":{"type":"object","properties":{"second":{"type":"boolean"},"first":{"type":"string"}}}},{"name":"WebSearch","input_schema":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}]"#;
    let body = format!(
        r#"{{
 "model" : "public-model", "max_tokens":32,
 "tools": {tools},
 "messages":[{{"content":[{{"text":"caf\u00e9 ☕","type":"text"}}],"role":"user"}}],
 "stream":false, "extra":{{"z":1,"a":[2,1]}}
}}"#
    );
    for chunked in [false, true] {
        let search = StatefulCapturingBackend::new(vec![(200, json!({"web":{"results":[{
            "title":"ORDERED_ANTHROPIC_RESULT","url":"https://example.org/result","description":"controlled evidence"
        }]}}).to_string())])
        .start_with_shutdown();
        let filters = format!(
            r#"- filter: anthropic_messages_request
- filter: iterative_request_router
  initial_step: inference
  max_iterations: 3
  steps:
    - name: inference
      filters:
        - filter: anthropic_web_search
          provider: brave
          api_key:
            value: synthetic-test-key
          base_url: http://127.0.0.1:{}
          outbound_chain:
            name: search
            filters:
              - filter: request_id
        - filter: router
          routes:
            - path_prefix: /
              cluster: backend
        - filter: load_balancer
          clusters:
            - name: backend
              endpoints: ['127.0.0.1:BACKEND_PORT']
      on_result:
        - filter: anthropic_web_search
          key: action
          value: loop
          next: inference
        - default: true
          done: true"#,
            search.port()
        );
        let reply = |content, stop_reason| {
            HttpServerAction::Json { status:200, body:json!({
            "id":"msg_contract", "type":"message", "role":"assistant", "model":"public-model",
            "content":content, "stop_reason":stop_reason,"stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":1}
        }).to_string() }
        };
        let rounds = capture_sequence(
            &filters,
            "/v1/messages",
            body.as_bytes(),
            chunked,
            &[],
            "/v1/messages",
            vec![
                reply(
                    json!([{"type":"tool_use","id":"tu_contract","name":"WebSearch","input":{"query":"ordering"}}]),
                    "tool_use",
                ),
                reply(json!([{"type":"text","text":"done"}]), "end_turn"),
            ],
        )
        .await;
        assert_json_changes_only(body.as_bytes(), &rounds[0], &[]);
        assert_json_changes_only(body.as_bytes(), &rounds[1], &["/messages"]);
        let actual: Value = serde_json::from_slice(&rounds[1]).unwrap();
        let original: Value = serde_json::from_str(&body).unwrap();
        assert_json_changes_only(
            &serde_json::to_vec(&original["messages"][0]).unwrap(),
            &serde_json::to_vec(&actual["messages"][0]).unwrap(),
            &[],
        );
        assert_eq!(actual["messages"].as_array().unwrap().len(), 3);
        assert!(actual["messages"][2].to_string().contains("ORDERED_ANTHROPIC_RESULT"));
        assert_eq!(search.requests().len(), 1);
    }
}
