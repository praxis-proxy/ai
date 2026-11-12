// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Active request mutations, captured after the real serializer and transport.

#[cfg(feature = "openai-agentic-loop")]
use praxis_test_utils::HttpServerAction;
#[cfg(any(
    feature = "openai-agentic-loop",
    feature = "openai-file-resolve-filter",
    all(feature = "openai-conversations", feature = "store-sqlite")
))]
use praxis_test_utils::StatefulCapturingBackend;
use praxis_test_utils::body_contract::{assert_json_changes_only, retained_tool_order_mismatch};
#[cfg(feature = "openai-mcp-tools")]
use praxis_test_utils::{McpMockConfig, McpToolFixture, start_mcp_mock_server_with_config};
use serde_json::{Value, json};

/// Two existing tools with deliberately unsorted, nested parameter schemas.
const TOOLS: &str = r#"[
 {"type":"function","name":"zebra","parameters":{"type":"object","properties":{"z":{"type":"string"},"a":{"type":"number"}}}},
 {"type":"function","name":"alpha","parameters":{"type":"object","properties":{"second":{"type":"boolean"},"first":{"type":"string"}}}}
]"#;
/// Original conversation item, reused without serialization by each wire case.
const INPUT: &str = r#"[{"content":[{"text":"hello\nworld","type":"input_text"}],"role":"user","type":"message"}]"#;

/// A stored conversation exposes the rehydrator's direct reasoning-default edit.
#[cfg(all(feature = "openai-conversations", feature = "store-sqlite"))]
fn stored_chain(database_url: &str) -> String {
    format!(
        "- filter: ai_operation\n- filter: openai_responses_request\n- filter: state_owner\n  mode: single_tenant\n  tenant_id: body-contract\n- filter: openai_responses_store\n  backend: sqlite\n  database_url: '{database_url}'\n  responses_table: body_contract_responses\n  conversations_table: body_contract_conversations\n- filter: openai_conversations\n  backend: sqlite\n  database_url: '{database_url}'\n  conversations_table: body_contract_conversations\n  items_table: body_contract_items\n- filter: openai_responses_rehydrate\n- filter: openai_responses_proxy"
    )
}

/// Construct adversarial raw JSON without normalizing its lexical spelling.
fn request(input: &str, tools: &str) -> String {
    format!(
        r#"{{
 "model" : "public-model",
 "metadata": {{"z":"caf\u00e9 ☕","a":"first"}},
 "tools": {tools},
 "input" : {input},
 "store": false,
 "stream": false,
 "extra": {{"z":1,"a":[2,1]}}
}}
"#
    )
}

/// Classify and mutate a complete Responses request before native forwarding.
fn chain(filter: &str) -> String {
    format!("- filter: ai_operation\n- filter: openai_responses_request\n- filter: {filter}")
}

/// Verify both that the mutation happened and that everything else survived.
fn changed(before: &str, after: &[u8], allowed: &[&str]) -> Value {
    assert_json_changes_only(before.as_bytes(), after, allowed);
    let original: Value = serde_json::from_str(before).unwrap();
    let actual: Value = serde_json::from_slice(after).unwrap();
    assert!(original != actual, "active case must perform an actual JSON mutation");
    assert!(
        retained_tool_order_mismatch(&original, &actual).is_none(),
        "retained tools and schema keys must keep their order"
    );
    actual
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn document_extraction_changes_only_the_file_content_part() {
    let input = r#"[{"content":[{"type":"input_file","filename":"notes.txt","file_data":"data:text/plain;base64,SGVsbG8gZnJvbSBmaWxlIQ=="},{"text":"untouched","type":"input_text"}],"role":"user","type":"message"}]"#;
    let body = request(input, TOOLS);
    let filters = chain("openai_doc_extract\n  allow_pre_security_callout: true");
    for chunked in [false, true] {
        let forwarded = super::forward(&filters, "/v1/responses", body.as_bytes(), chunked).await;
        let actual = changed(&body, &forwarded, &["/input/0/content/0"]);
        assert_eq!(actual["input"][0]["content"][0]["type"], "input_text");
        let text = actual["input"][0]["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("[Source: notes.txt]") && text.contains("Hello from file!"));
    }
}

#[cfg(feature = "openai-file-resolve-filter")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn file_resolution_changes_only_reference_fields() {
    let input = r#"[{"content":[{"file_id":"file-contract","type":"input_file","opaque":{"z":1,"a":2}},{"text":"untouched","type":"input_text"}],"role":"user","type":"message"}]"#;
    let body = request(input, TOOLS);
    for chunked in [false, true] {
        let files = StatefulCapturingBackend::new(vec![
            (
                200,
                r#"{"id":"file-contract","filename":"notes.txt","content_type":"text/plain"}"#.to_owned(),
            ),
            (200, "Hello from file!".to_owned()),
        ])
        .start_with_shutdown();
        let filters = chain(&format!(
            "openai_file_resolve\n  allow_pre_security_callout: true\n  files_api_url: http://127.0.0.1:{}",
            files.port()
        ));
        let forwarded = super::forward(&filters, "/v1/responses", body.as_bytes(), chunked).await;
        let actual = changed(
            &body,
            &forwarded,
            &[
                "/input/0/content/0/file_id",
                "/input/0/content/0/file_data",
                "/input/0/content/0/filename",
            ],
        );
        assert_eq!(
            actual["input"][0]["content"][0]["file_data"],
            "SGVsbG8gZnJvbSBmaWxlIQ=="
        );
        assert_eq!(actual["input"][0]["content"][0]["filename"], "notes.txt");
        let calls = files.requests();
        assert_eq!(calls.len(), 2, "the active case must fetch metadata and content");
        assert_eq!(calls[0].uri, "/v1/files/file-contract");
        assert_eq!(calls[1].uri, "/v1/files/file-contract/content");
    }
}

#[cfg(feature = "openai-mcp-tools")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mcp_discovery_replaces_only_tools_and_retains_original_tool_order() {
    let schema: Value =
        serde_json::from_str(r#"{"type":"object","properties":{"z":{"type":"string"},"a":{"type":"number"}}}"#)
            .unwrap();
    let mcp = start_mcp_mock_server_with_config(McpMockConfig {
        tools: vec![McpToolFixture::new("inspect").with_input_schema(schema)],
        ..McpMockConfig::default()
    });
    let tools = format!(
        "{},{{\"type\":\"mcp\",\"server_label\":\"server\",\"server_url\":\"http://{}/mcp\",\"require_approval\":\"never\"}}]",
        TOOLS.trim_end().strip_suffix(']').unwrap(),
        mcp.endpoint()
    );
    let body = request(INPUT, &tools).replace(
        "\"store\": false",
        "\"tool_choice\":{\"type\":\"mcp\",\"server_label\":\"server\",\"name\":\"inspect\"},\n \"store\": false",
    );
    let filters = chain("openai_tool_parse\n- filter: openai_mcp_tool_resolve\n- filter: openai_responses_proxy");
    for chunked in [false, true] {
        let forwarded = super::forward(&filters, "/v1/responses", body.as_bytes(), chunked).await;
        let actual = changed(&body, &forwarded, &["/tools", "/tool_choice"]);
        assert_eq!(actual["tools"].as_array().unwrap().len(), 3);
        assert_eq!(
            actual["tool_choice"],
            json!({"type":"function","name":"server__inspect"})
        );
        assert_eq!(actual["tools"][0]["name"], "zebra");
        assert_eq!(actual["tools"][1]["name"], "alpha");
        assert_eq!(actual["tools"][2]["type"], "function");
        assert!(actual["tools"][2]["name"].as_str().unwrap().contains("inspect"));
        assert_json_changes_only(
            TOOLS.as_bytes(),
            &serde_json::to_vec(&actual["tools"].as_array().unwrap()[..2]).unwrap(),
            &[],
        );
        assert_eq!(
            actual["tools"][2]["parameters"]["properties"]
                .as_object()
                .unwrap()
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            ["z", "a"],
        );
    }
    assert!(mcp.method_count("tools/list") >= 2, "discovery must actually execute");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn client_tool_lowering_changes_only_its_tool_slot() {
    let tools = format!(
        "{},{{\"type\":\"custom\",\"name\":\"echo\",\"description\":\"raw echo\"}}]",
        TOOLS.trim_end().strip_suffix(']').unwrap()
    );
    let body = request(INPUT, &tools);
    let filters = chain("openai_client_tool_compat\n- filter: openai_responses_proxy");
    for chunked in [false, true] {
        let forwarded = super::forward(&filters, "/v1/responses", body.as_bytes(), chunked).await;
        let actual = changed(&body, &forwarded, &["/tools/2"]);
        assert_eq!(actual["tools"][2]["type"], "function");
        assert!(actual["tools"][2]["name"].as_str().is_some_and(|name| !name.is_empty()));
    }
}

/// A real two-round native Responses pipeline, with dispatch before loop ownership.
#[cfg(feature = "openai-agentic-loop")]
fn loop_chain(dispatch: &str, prelude: &str) -> String {
    let dispatch = dispatch
        .lines()
        .map(|line| format!("        {line}\n"))
        .collect::<String>();
    format!(
        r#"- filter: ai_operation
- filter: openai_responses_request
- filter: openai_tool_parse
{prelude}- filter: iterative_request_router
  initial_step: inference
  max_iterations: 3
  steps:
    - name: inference
      filters:
{dispatch}        - filter: openai_agentic_loop
          max_infer_iters: 2
        - filter: openai_responses_proxy
        - filter: router
          routes:
            - path_prefix: /
              cluster: backend
        - filter: load_balancer
          clusters:
            - name: backend
              endpoints: ['127.0.0.1:BACKEND_PORT']
      on_result:
        - filter: openai_agentic_loop
          key: action
          value: loop
          next: inference
        - default: true
          done: true"#
    )
}

/// Synthetic model response; no live provider or credentials are used.
#[cfg(feature = "openai-agentic-loop")]
fn round(output: &Value) -> HttpServerAction {
    HttpServerAction::Json {
        status: 200,
        body: json!({
            "id":"resp_contract", "object":"response", "status":"completed",
            "model":"public-model", "output":output,
            "usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}
        })
        .to_string(),
    }
}

/// Final model answer after the actual local tool callout has completed.
#[cfg(feature = "openai-agentic-loop")]
fn final_round() -> HttpServerAction {
    round(
        &json!([{"type":"message","id":"msg_contract","role":"assistant","status":"completed","content":[{"type":"output_text","text":"done","annotations":[]}]}]),
    )
}

/// The serializer may append context but must retain the original input verbatim.
#[cfg(feature = "openai-agentic-loop")]
fn assert_original_input_retained(before: &[u8], after: &Value) {
    let original: Value = serde_json::from_slice(before).unwrap();
    let original = original["input"].as_array().unwrap();
    let actual = after["input"].as_array().unwrap();
    assert!(actual.len() > original.len(), "a continuation must append tool context");
    assert_json_changes_only(
        &serde_json::to_vec(original).unwrap(),
        &serde_json::to_vec(&actual[..original.len()]).unwrap(),
        &[],
    );
}

#[cfg(feature = "openai-agentic-loop")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn web_dispatch_and_loop_append_results_without_reordering_the_request() {
    let tools = format!(
        "{},{{\"type\":\"web_search\"}}]",
        TOOLS.trim_end().strip_suffix(']').unwrap()
    );
    let body = request(INPUT, &tools);
    for chunked in [false, true] {
        let search =
            StatefulCapturingBackend::new(vec![(200, json!({"web":{"results":[{
            "title":"ORDERED_SEARCH_RESULT","url":"https://example.org/result","description":"controlled evidence"
        }]}}).to_string())])
            .start_with_shutdown();
        let dispatch = format!(
            "- filter: openai_web_search_dispatch\n  provider: brave\n  api_key:\n    value: synthetic-test-key\n  base_url: http://127.0.0.1:{}\n  outbound_chain:\n    name: search\n    filters:\n      - filter: request_id",
            search.port()
        );
        let filters = loop_chain(&dispatch, "");
        let rounds = super::capture_sequence(&filters, "/v1/responses", body.as_bytes(), chunked, &[], "/v1/responses", vec![
            round(&json!([{"type":"web_search_call","id":"ws_contract","status":"in_progress","action":{"type":"search","query":"ordering"}}])),
            final_round(),
        ]).await;
        assert_eq!(rounds.len(), 2);
        assert_json_changes_only(body.as_bytes(), &rounds[0], &[]);
        let actual = changed(
            std::str::from_utf8(&rounds[0]).unwrap(),
            &rounds[1],
            &["/input", "/tool_choice"],
        );
        assert_original_input_retained(&rounds[0], &actual);
        assert_eq!(actual["tool_choice"], "auto");
        assert!(actual["input"].to_string().contains("ORDERED_SEARCH_RESULT"));
        assert_eq!(search.requests().len(), 1, "the dispatcher must execute the search");
    }
}

#[cfg(all(feature = "openai-mcp-tools", feature = "openai-agentic-loop"))]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mcp_dispatch_appends_results_without_reordering_request_fields_or_tools() {
    for chunked in [false, true] {
        let mcp = start_mcp_mock_server_with_config(McpMockConfig {
            tools: vec![McpToolFixture::new("inspect")],
            ..McpMockConfig::default()
        });
        let tools = format!(
            "{},{{\"type\":\"mcp\",\"server_label\":\"server\",\"server_url\":\"http://{}/mcp\",\"require_approval\":\"never\"}}]",
            TOOLS.trim_end().strip_suffix(']').unwrap(),
            mcp.endpoint()
        );
        let body = request(INPUT, &tools);
        let filters = loop_chain("- filter: openai_mcp_dispatch", "- filter: openai_mcp_tool_resolve\n");
        let rounds = super::capture_sequence(&filters, "/v1/responses", body.as_bytes(), chunked, &[], "/v1/responses", vec![
            round(&json!([{"type":"function_call","id":"fc_contract","call_id":"call_contract","name":"server__inspect","arguments":"{\"z\":1,\"a\":2}","status":"completed"}])),
            final_round(),
        ]).await;
        changed(&body, &rounds[0], &["/tools"]);
        let actual = changed(
            std::str::from_utf8(&rounds[0]).unwrap(),
            &rounds[1],
            &["/input", "/tool_choice"],
        );
        assert_original_input_retained(&rounds[0], &actual);
        assert_eq!(actual["tool_choice"], "auto");
        assert!(
            actual["input"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["type"] == "function_call_output" && item["call_id"] == "call_contract")
        );
        assert_eq!(
            mcp.tool_call_count("inspect"),
            1,
            "the dispatcher must execute the tool"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn file_search_lowering_changes_only_the_hosted_tool_and_choice() {
    let tools = format!(
        "{},{{\"type\":\"file_search\",\"vector_store_ids\":[\"vs_contract\"]}}]",
        TOOLS.trim_end().strip_suffix(']').unwrap()
    );
    let body = request(INPUT, &tools).replace(
        "\"store\": false",
        "\"tool_choice\": {\"type\":\"file_search\"},\n \"store\": false",
    );
    let filters =
        chain("openai_file_search_dispatch\n  vector_store_url: http://127.0.0.1:9\n- filter: openai_responses_proxy");
    for chunked in [false, true] {
        let forwarded = super::forward(&filters, "/v1/responses", body.as_bytes(), chunked).await;
        let actual = changed(&body, &forwarded, &["/tools/2", "/tool_choice"]);
        assert_eq!(actual["tools"][2]["type"], "function");
        assert_eq!(actual["tools"][2]["name"], "file_search");
        assert_eq!(actual["tool_choice"], json!({"type":"function","name":"file_search"}));
    }
}

#[cfg(all(feature = "openai-conversations", feature = "store-sqlite"))]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rehydration_merges_reasoning_and_history_without_reordering_other_fields() {
    use praxis_test_utils::{TempSqlite, parse_body, parse_status, start_proxy};
    for chunked in [false, true] {
        let database = TempSqlite::new("body-contract-rehydrate");
        let backend = StatefulCapturingBackend::new(vec![(200, "{}".to_owned())]).start_with_shutdown();
        let config = super::capture_config(&stored_chain(database.url()), backend.port());
        let proxy = start_proxy(&config);
        let create = r#"{"metadata":{},"items":[{"type":"configuration_update","reasoning":{"effort":"low"}},{"type":"message","role":"user","content":[{"type":"input_text","text":"stored context"}]}]}"#;
        let response = super::send_raw(proxy.addr(), "/v1/conversations", create.as_bytes(), false, &[]).await;
        let response = String::from_utf8(response).unwrap();
        assert_eq!(parse_status(&response), 200, "conversation setup must succeed");
        let conversation: Value = serde_json::from_str(&parse_body(&response)).unwrap();
        let id = conversation["id"].as_str().unwrap();
        let body = request(INPUT, TOOLS).replace(
            "\"store\": false",
            &format!("\"conversation\":\"{id}\",\n \"store\": false"),
        );
        let response = super::send_raw(proxy.addr(), "/v1/responses", body.as_bytes(), chunked, &[]).await;
        assert_eq!(parse_status(&String::from_utf8(response).unwrap()), 200);
        let calls = backend.requests();
        assert_eq!(calls.len(), 1);
        let actual = changed(
            &body,
            calls[0].body.as_bytes(),
            &["/conversation", "/reasoning", "/input"],
        );
        assert_eq!(actual["reasoning"], json!({"effort":"low"}));
        let input = actual["input"].as_array().unwrap();
        assert!(input[0].to_string().contains("stored context"));
        assert!(input.iter().all(|item| item["type"] != "configuration_update"));
        assert_json_changes_only(
            INPUT.as_bytes(),
            &serde_json::to_vec(&input[input.len() - 1..]).unwrap(),
            &[],
        );
    }
}


/// Compile-time links to active cases, independent of bypass configurations.
#[cfg(all(feature = "openai-all", feature = "store-sqlite"))]
pub(super) fn mutation_witnesses() -> Vec<&'static str> {
    let _ = document_extraction_changes_only_the_file_content_part;
    let _ = file_resolution_changes_only_reference_fields;
    let _ = mcp_discovery_replaces_only_tools_and_retains_original_tool_order;
    let _ = client_tool_lowering_changes_only_its_tool_slot;
    let _ = file_search_lowering_changes_only_the_hosted_tool_and_choice;
    let _ = web_dispatch_and_loop_append_results_without_reordering_the_request;
    let _ = mcp_dispatch_appends_results_without_reordering_request_fields_or_tools;
    let _ = rehydration_merges_reasoning_and_history_without_reordering_other_fields;
    vec![
        "openai_doc_extract",
        "openai_file_resolve",
        "openai_mcp_tool_resolve",
        "openai_client_tool_compat",
        "openai_file_search_dispatch",
        "openai_web_search_dispatch",
        "openai_mcp_dispatch",
        "openai_agentic_loop",
        "openai_responses_proxy",
        "openai_responses_rehydrate",
    ]
}
