// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Full-pipeline SSE admission and terminal failure under the agentic budget.

use std::collections::HashMap;

use praxis_test_utils::{
    StatefulCapturingBackend, TempSqlite, example_config_path, free_port, http_get, http_send, json_post, parse_body,
    parse_status, patch_yaml, start_proxy,
};

fn budgeted_agentic_config(
    proxy_port: u16,
    model_port: u16,
    max_retained_bytes: usize,
    db: &TempSqlite,
) -> praxis_core::config::Config {
    let yaml = std::fs::read_to_string(example_config_path("openai/responses/agentic-loop.yaml"))
        .expect("read agentic example");
    let with_budget = yaml.replacen(
        "max_retained_bytes: 67108864",
        &format!("max_retained_bytes: {max_retained_bytes}"),
        1,
    );
    let with_store = with_budget.replace("sqlite://responses.db?mode=rwc", db.url());
    let with_key = with_store.replace("api_key: ${WEB_SEARCH_API_KEY}", "api_key: test-key");
    let patched = patch_yaml(&with_key, proxy_port, &HashMap::from([("127.0.0.1:3001", model_port)]));
    praxis_core::config::Config::from_yaml(&patched).expect("parse budgeted agentic config")
}

#[test]
fn budgeted_stream_keeps_normal_sse_terminal() {
    let response = serde_json::json!({
        "id":"resp_stream_ok", "object":"response", "status":"completed", "output":[{
            "type":"message", "id":"msg_stream_ok", "role":"assistant",
            "content":[{"type":"output_text", "text":"hello"}]
        }]
    });
    let events = format!(
        "data: {{\"type\":\"response.created\",\"response\":{{\"id\":\"resp_stream_ok\",\"output\":[]}}}}\n\n\
         data: {{\"type\":\"response.output_text.delta\",\"delta\":\"hello\",\"output_index\":0}}\n\n\
         data: {{\"type\":\"response.completed\",\"response\":{response}}}\n\n\
         data: [DONE]\n\n"
    );
    let model = StatefulCapturingBackend::new(vec![(200, events)]).start_with_shutdown();
    let db = TempSqlite::new("budgeted_stream_success");
    let proxy = start_proxy(&budgeted_agentic_config(free_port(), model.port(), 8_388_608, &db));

    let raw = http_send(
        proxy.addr(),
        &json_post(
            "/v1/responses",
            r#"{"model":"gpt-4.1","input":"hello","stream":true,"store":false}"#,
        ),
    );

    assert_eq!(parse_status(&raw), 200, "budgeted stream failed: {raw}");
    let body = parse_body(&raw);
    assert!(body.contains("event: response.created"), "{body}");
    assert!(body.contains("event: response.completed"), "{body}");
    assert!(body.contains("\"text\":\"hello\""), "{body}");
    assert_eq!(body.matches("data: [DONE]").count(), 1, "{body}");
    assert_eq!(model.requests().len(), 1);
}

#[test]
fn committed_budget_overflow_emits_one_error_without_success_or_store() {
    let large_delta = "x".repeat(4_096);
    let events = format!(
        "data: {{\"type\":\"response.created\",\"response\":{{\"id\":\"resp_stream_overflow\",\"output\":[]}}}}\n\n\
         data: {{\"type\":\"response.output_text.delta\",\"delta\":\"ok\",\"output_index\":0}}\n\n\
         data: {{\"type\":\"response.output_text.delta\",\"delta\":\"{large_delta}\",\"output_index\":0}}\n\n\
         data: {{\"type\":\"response.completed\",\"response\":{{\"id\":\"resp_stream_overflow\",\"status\":\"completed\",\"output\":[]}}}}\n\n\
         data: [DONE]\n\n"
    );
    let model = StatefulCapturingBackend::new(vec![(200, events)]).start_with_shutdown();
    let db = TempSqlite::new("budgeted_stream_overflow");
    let proxy = start_proxy(&budgeted_agentic_config(free_port(), model.port(), 65_536, &db));
    let request = r#"{"model":"gpt-4.1","input":"hello","stream":true}"#;

    let raw = http_send(proxy.addr(), &json_post("/v1/responses", request));

    assert_eq!(
        parse_status(&raw),
        200,
        "headers committed before stream overflow: {raw}"
    );
    let body = parse_body(&raw);
    assert!(body.contains("event: response.created"), "{body}");
    assert_eq!(body.matches("event: error").count(), 1, "{body}");
    assert!(!body.contains("event: response.completed"), "{body}");
    assert!(!body.contains("data: [DONE]"), "{body}");
    assert_eq!(model.requests().len(), 1, "no second inference should dispatch");
    let (status, _) = http_get(proxy.addr(), "/v1/responses/resp_stream_overflow", None);
    assert_eq!(status, 404, "an overflow must not persist a successful response");
}
