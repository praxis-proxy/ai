// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Full-pipeline file resolution under the agentic retained-payload ceiling.

use std::collections::HashMap;

use praxis_test_utils::{
    StatefulCapturingBackend, TempSqlite, example_config_path, free_port, http_get, http_send, json_post, parse_body,
    parse_status, patch_yaml, start_proxy,
};

use super::openai_file_resolve::start_files_api_stub;

fn agentic_file_config(
    proxy_port: u16,
    model_port: u16,
    files_port: u16,
    max_retained_bytes: usize,
    db: &TempSqlite,
) -> praxis_core::config::Config {
    let yaml = std::fs::read_to_string(example_config_path("openai/responses/agentic-loop.yaml"))
        .expect("read agentic example");
    let with_file_resolver = yaml.replacen(
        "      - filter: openai_responses_rehydrate\n",
        "      - filter: openai_responses_rehydrate\n\n      - filter: openai_file_resolve\n        files_api_url: http://127.0.0.1:9999\n        allow_pre_security_callout: true\n        on_missing: continue\n",
        1,
    );
    assert_ne!(yaml, with_file_resolver, "expected rehydrate in agentic example");
    let with_budget = with_file_resolver.replacen(
        "max_retained_bytes: 67108864",
        &format!("max_retained_bytes: {max_retained_bytes}"),
        1,
    );
    let with_store = with_budget.replace("sqlite://responses.db?mode=rwc", db.url());
    let with_key = with_store.replace("api_key: ${WEB_SEARCH_API_KEY}", "api_key: test-key");
    let patched = patch_yaml(
        &with_key,
        proxy_port,
        &HashMap::from([("127.0.0.1:3001", model_port), ("127.0.0.1:9999", files_port)]),
    );
    praxis_core::config::Config::from_yaml(&patched).expect("parse agentic file config")
}

#[test]
fn budgeted_file_id_reaches_inference_with_inline_content() {
    let files_port = start_files_api_stub();
    let response = r#"{"id":"resp_file_budget","object":"response","status":"completed","output":[]}"#;
    let model = StatefulCapturingBackend::new(vec![(200, response.to_owned())]).start_with_shutdown();
    let db = TempSqlite::new("budgeted_file_success");
    let config = agentic_file_config(free_port(), model.port(), files_port, 8_388_608, &db);
    let proxy = start_proxy(&config);
    let request = r#"{"model":"gpt-4.1","store":false,"input":[{"type":"message","role":"user","content":[{"type":"input_file","file_id":"test-file-123"}]}]}"#;

    let raw = http_send(proxy.addr(), &json_post("/v1/responses", request));

    assert_eq!(parse_status(&raw), 200, "in-budget file request failed: {raw}");
    let requests = model.requests();
    assert_eq!(requests.len(), 1, "one resolved request should reach inference");
    let body: serde_json::Value = serde_json::from_str(&requests[0].body).expect("provider request JSON");
    let part = &body["input"][0]["content"][0];
    assert_eq!(part["file_data"], "SGVsbG8sIHdvcmxkIQ==");
    assert_eq!(part["filename"], "test.txt");
    assert!(part.get("file_id").is_none());
}

#[test]
fn budgeted_file_id_exhaustion_returns_413_without_inference() {
    let files_port = start_files_api_stub();
    let model = StatefulCapturingBackend::new(vec![(200, "{}".to_owned())]).start_with_shutdown();
    let db = TempSqlite::new("budgeted_file_exhaustion");
    let config = agentic_file_config(free_port(), model.port(), files_port, 131_072, &db);
    let proxy = start_proxy(&config);
    let request = r#"{"model":"gpt-4.1","store":false,"input":[{"type":"message","role":"user","content":[{"type":"input_file","file_id":"test-file-123"}]}]}"#;

    let raw = http_send(proxy.addr(), &json_post("/v1/responses", request));

    assert_eq!(parse_status(&raw), 413, "exhausted file request must reject: {raw}");
    assert!(model.requests().is_empty(), "budget rejection must precede inference");
}

#[test]
fn budgeted_file_turn_replays_through_previous_response() {
    let files_port = start_files_api_stub();
    let first = r#"{"id":"resp_file_first","object":"response","created_at":1760000000,"model":"gpt-4.1","status":"completed","output":[{"type":"message","id":"msg_file_first","role":"assistant","content":[{"type":"output_text","text":"First answer"}]}]}"#;
    let second = r#"{"id":"resp_file_second","object":"response","created_at":1760000001,"model":"gpt-4.1","status":"completed","output":[{"type":"message","id":"msg_file_second","role":"assistant","content":[{"type":"output_text","text":"Second answer"}]}]}"#;
    let model =
        StatefulCapturingBackend::new(vec![(200, first.to_owned()), (200, second.to_owned())]).start_with_shutdown();
    let db = TempSqlite::new("budgeted_file_restore");
    let config = agentic_file_config(free_port(), model.port(), files_port, 8_388_608, &db);
    let proxy = start_proxy(&config);
    let first_request = r#"{"model":"gpt-4.1","input":[{"type":"message","role":"user","content":[{"type":"input_file","file_id":"test-file-123"}]}]}"#;

    let first_raw = http_send(proxy.addr(), &json_post("/v1/responses", first_request));
    assert_eq!(parse_status(&first_raw), 200, "first file turn: {first_raw}");
    let created: serde_json::Value = serde_json::from_str(&parse_body(&first_raw)).unwrap();
    let first_id = created["id"].as_str().expect("first response ID");
    let next_request = serde_json::json!({
        "model":"gpt-4.1",
        "input":"Next turn",
        "previous_response_id":first_id,
        "store":false
    });
    let second_raw = http_send(proxy.addr(), &json_post("/v1/responses", &next_request.to_string()));
    assert_eq!(parse_status(&second_raw), 200, "file replay: {second_raw}");

    let requests = model.requests();
    assert_eq!(requests.len(), 2, "one inference per turn");
    let replayed: serde_json::Value = serde_json::from_str(&requests[1].body).unwrap();
    let content = &replayed["input"][0]["content"][0];
    assert_eq!(
        content["file_data"], "SGVsbG8sIHdvcmxkIQ==",
        "file remains inline during replay"
    );
    assert!(content.get("file_id").is_none());
}

#[test]
fn budgeted_file_turn_replays_through_conversation() {
    let files_port = start_files_api_stub();
    let first = r#"{"id":"resp_conv_file_first","object":"response","created_at":1760000000,"model":"gpt-4.1","status":"completed","output":[{"type":"message","id":"msg_conv_file_first","role":"assistant","content":[{"type":"output_text","text":"First answer"}]}]}"#;
    let second = r#"{"id":"resp_conv_file_second","object":"response","created_at":1760000001,"model":"gpt-4.1","status":"completed","output":[{"type":"message","id":"msg_conv_file_second","role":"assistant","content":[{"type":"output_text","text":"Second answer"}]}]}"#;
    let model =
        StatefulCapturingBackend::new(vec![(200, first.to_owned()), (200, second.to_owned())]).start_with_shutdown();
    let db = TempSqlite::new("budgeted_file_conversation");
    let config = agentic_file_config(free_port(), model.port(), files_port, 8_388_608, &db);
    let proxy = start_proxy(&config);
    let created = http_send(proxy.addr(), &json_post("/v1/conversations", "{}"));
    assert_eq!(parse_status(&created), 200, "conversation create: {created}");
    let created: serde_json::Value = serde_json::from_str(&parse_body(&created)).unwrap();
    let id = created["id"].as_str().expect("conversation ID");
    let first_request = serde_json::json!({
        "model":"gpt-4.1",
        "input":[{"type":"message","role":"user","content":[{"type":"input_file","file_id":"test-file-123"}]}],
        "conversation":id,
        "store":false
    });
    let first_raw = http_send(proxy.addr(), &json_post("/v1/responses", &first_request.to_string()));
    assert_eq!(parse_status(&first_raw), 200, "first file turn: {first_raw}");
    let next_request = serde_json::json!({"model":"gpt-4.1","input":"Next turn","conversation":id,"store":false});
    let second_raw = http_send(proxy.addr(), &json_post("/v1/responses", &next_request.to_string()));
    assert_eq!(parse_status(&second_raw), 200, "conversation file replay: {second_raw}");

    let requests = model.requests();
    assert_eq!(requests.len(), 2, "one inference per turn");
    let replayed: serde_json::Value = serde_json::from_str(&requests[1].body).unwrap();
    let content = &replayed["input"][0]["content"][0];
    assert_eq!(
        content["file_data"], "SGVsbG8sIHdvcmxkIQ==",
        "file remains inline during replay"
    );
    assert!(content.get("file_id").is_none());
    let (status, items) = http_get(proxy.addr(), &format!("/v1/conversations/{id}/items"), None);
    assert_eq!(status, 200, "conversation item listing: {items}");
}
