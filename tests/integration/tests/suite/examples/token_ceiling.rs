// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for the `token_ceiling` filter example configuration.

use std::collections::HashMap;

use praxis_test_utils::{
    Backend, free_port, http_send, json_post, load_example_config, parse_body, parse_status, start_proxy,
};

#[test]
fn example_config_enforces_output_token_ceiling() {
    let backend = Backend::fixed("upstream").start_with_shutdown();
    let proxy_port = free_port();
    let config = load_example_config(
        "token-ceiling.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:9000", backend.port())]),
    );
    let proxy = start_proxy(&config);

    let accepted = http_send(
        proxy.addr(),
        &json_post(
            "/v1/chat/completions",
            r#"{"model":"test","messages":[],"max_tokens":32}"#,
        ),
    );
    assert_eq!(
        parse_status(&accepted),
        200,
        "request within the output ceiling should be forwarded"
    );
    assert_eq!(
        parse_body(&accepted),
        "upstream",
        "accepted request should reach the backend"
    );

    let oversized_prompt = "token ".repeat(5_000);
    let rejected_input = http_send(
        proxy.addr(),
        &json_post(
            "/v1/chat/completions",
            &serde_json::json!({
                "model": "test",
                "messages": [{"role": "user", "content": oversized_prompt}],
                "max_tokens": 32,
            })
            .to_string(),
        ),
    );
    assert_eq!(
        parse_status(&rejected_input),
        400,
        "request over the input ceiling should be rejected"
    );
    assert!(
        parse_body(&rejected_input).contains("max_input_tokens_exceeded"),
        "input rejection should identify the input ceiling error"
    );

    let rejected = http_send(
        proxy.addr(),
        &json_post(
            "/v1/chat/completions",
            r#"{"model":"test","messages":[],"max_tokens":1025}"#,
        ),
    );
    assert_eq!(
        parse_status(&rejected),
        400,
        "request over the output ceiling should be rejected"
    );
    assert!(
        parse_body(&rejected).contains("token_ceiling_exceeded"),
        "rejection should identify the token ceiling error"
    );
}
