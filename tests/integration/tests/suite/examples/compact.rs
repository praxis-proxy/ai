// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for the `compact` example config.
//!
//! Verifies that the example pipeline builds, simple requests pass
//! through, and the multi-turn compaction flow works end-to-end.

use std::collections::HashMap;

use praxis_test_utils::{
    Backend, TempSqlite, example_config_path, free_port, http_send, json_post, parse_status, patch_yaml, start_proxy,
};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Backend response for the first turn — stored by response_store.
/// The output text is long enough to exceed a low compact_threshold.
const FIRST_RESPONSE_JSON: &str = r#"{"id":"resp_compact","created_at":1000,"model":"gpt-4.1","object":"response","status":"completed","input":"Explain TCP vs UDP","output":[{"type":"message","content":[{"type":"output_text","text":"TCP is a connection-oriented protocol that provides reliable, ordered delivery of data. It establishes a connection through a three-way handshake before transmitting data. UDP is a connectionless protocol that sends data without establishing a connection first. TCP guarantees delivery through acknowledgments and retransmissions while UDP does not. TCP is used for applications requiring reliability like web browsing and email while UDP is used for real-time applications like video streaming and gaming where speed matters more than reliability."}]}]}"#;

/// Chat Completions response returned by the mock backend on the second
/// turn — used by both the compact filter's summarization callout and
/// the main inference forwarded by `openai_responses_proxy`.
const CHAT_COMPLETIONS_RESPONSE: &str = r#"{"id":"chatcmpl-1","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"Summary of the conversation."},"finish_reason":"stop"}],"usage":{"prompt_tokens":50,"completion_tokens":10,"total_tokens":60}}"#;

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// Load the compact example config, replacing the SQLite URL and
/// patching listener/backend addresses.
fn load_compact_config(yaml: &str, db_url: &str, proxy_port: u16, backend_port: u16) -> praxis_core::config::Config {
    let replaced = yaml
        .replace("sqlite://responses.db?mode=rwc", db_url)
        .replace("localhost:11434", &format!("127.0.0.1:{backend_port}"));
    let patched = patch_yaml(
        &replaced,
        proxy_port,
        &HashMap::from([("127.0.0.1:11434", backend_port)]),
    );
    praxis_core::config::Config::from_yaml(&patched).expect("patched config should parse")
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn compact_passthrough() {
    let backend_guard = Backend::fixed(FIRST_RESPONSE_JSON)
        .header("content-type", "application/json")
        .start_with_shutdown();
    let proxy_port = free_port();

    let yaml = std::fs::read_to_string(example_config_path("openai/responses/compact.yaml"))
        .expect("example config should exist");
    let config = load_compact_config(&yaml, "sqlite::memory:", proxy_port, backend_guard.port());
    let proxy = start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", r#"{"model":"gpt-4.1","input":"Hello"}"#),
    );

    assert_eq!(
        parse_status(&raw),
        200,
        "request without context_management should pass through"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn compact_multi_turn_compaction() {
    let backend1 = Backend::fixed(FIRST_RESPONSE_JSON)
        .header("content-type", "application/json")
        .start_with_shutdown();
    let proxy_port = free_port();

    let db = TempSqlite::new("compact");
    let yaml = std::fs::read_to_string(example_config_path("openai/responses/compact.yaml"))
        .expect("example config should exist");

    let config1 = load_compact_config(&yaml, db.url(), proxy_port, backend1.port());
    let proxy1 = start_proxy(&config1);

    let raw1 = http_send(
        proxy1.addr(),
        &json_post("/v1/responses", r#"{"model":"gpt-4.1","input":"Explain TCP vs UDP"}"#),
    );
    assert_eq!(parse_status(&raw1), 200, "first request should succeed");

    drop(backend1);
    drop(proxy1);

    let backend2 = Backend::fixed(CHAT_COMPLETIONS_RESPONSE)
        .header("content-type", "application/json")
        .start_with_shutdown();

    let config2 = load_compact_config(&yaml, db.url(), proxy_port, backend2.port());
    let proxy2 = start_proxy(&config2);

    let raw2 = http_send(
        proxy2.addr(),
        &json_post(
            "/v1/responses",
            r#"{"model":"gpt-4.1","input":"Compare with QUIC","previous_response_id":"resp_compact","context_management":[{"type":"compaction","compact_threshold":50}]}"#,
        ),
    );
    let status2 = parse_status(&raw2);
    assert_eq!(
        status2, 200,
        "second request with compaction should succeed (callout + pipeline completed)"
    );

    drop(proxy2);
}
