// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Integration tests for the `token_usage_metrics` example config.
//!
//! Verifies both transparent proxying and the exported Prometheus series.

use std::collections::HashMap;

use praxis_test_utils::{
    Backend, free_port, http_send, load_example_config, parse_body, parse_status, start_proxy, wait_for_http,
};

const OPENAI_JSON: &str = r#"{"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}}"#;

fn json_post_with_headers(path: &str, body: &str, headers: &[(&str, &str)]) -> String {
    let mut extra = String::new();
    for (name, value) in headers {
        extra.push_str(&format!("{name}: {value}\r\n"));
    }
    format!(
        "POST {path} HTTP/1.1\r\n\
         Host: localhost\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         {extra}\
         Connection: close\r\n\r\n\
         {body}",
        body.len()
    )
}

#[test]
fn example_config_token_usage_metrics_passthrough() {
    let backend = Backend::fixed(OPENAI_JSON)
        .header("content-type", "application/json")
        .start_with_shutdown();
    let proxy_port = free_port();
    let admin_port = free_port();
    let mut config = load_example_config(
        "token-usage-metrics.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend.port())]),
    );
    let admin_addr = format!("127.0.0.1:{admin_port}");
    config.admin.address = Some(admin_addr.clone());
    let proxy = start_proxy(&config);
    wait_for_http(&admin_addr);

    let raw = http_send(
        proxy.addr(),
        &json_post_with_headers(
            "/v1/chat/completions",
            r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}"#,
            &[("x-tenant-username", "token-metrics-example-alice")],
        ),
    );
    assert_eq!(parse_status(&raw), 200, "example config smoke test should return 200");
    assert_eq!(parse_body(&raw), OPENAI_JSON, "body should pass through unchanged");

    let metrics = http_send(
        &admin_addr,
        "GET /metrics HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
    );
    assert_eq!(parse_status(&metrics), 200, "metrics endpoint should return 200");
    let metrics_body = parse_body(&metrics);
    assert!(
        metrics_body.lines().any(|line| {
            line.starts_with("praxis_ai_tokens_total{")
                && line.contains("tenant=\"token-metrics-example-alice\"")
                && line.contains("kind=\"total\"")
        }),
        "metrics should contain the per-tenant total-token series: {metrics_body}"
    );
}
