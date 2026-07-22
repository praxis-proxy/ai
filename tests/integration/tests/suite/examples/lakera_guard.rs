// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Functional integration tests for the `lakera-guard` example config.

use std::collections::HashMap;

use praxis_test_utils::{
    example_config_path, free_port, http_send, parse_status, patch_yaml, start_backend_with_shutdown, start_proxy,
};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn lakera_guard_rejects_flagged() {
    let (mock_lakera_port, _lakera_guard) = start_mock_lakera(true);
    let backend_guard = start_backend_with_shutdown("upstream ok");
    let proxy_port = free_port();

    let config = load_lakera_config(proxy_port, mock_lakera_port, backend_guard.port());
    let proxy = start_proxy(&config);

    let payload = r#"{"model":"test","messages":[{"role":"user","content":"bad"}]}"#;
    let raw = http_send(
        proxy.addr(),
        &format!(
            "POST /v1/chat/completions HTTP/1.1\r\n\
             Host: localhost\r\n\
             Content-Type: application/json\r\n\
             Content-Length: {}\r\n\
             Connection: close\r\n\r\n\
             {payload}",
            payload.len()
        ),
    );

    assert_eq!(parse_status(&raw), 403, "flagged request should be rejected with 403");
}

#[test]
fn lakera_guard_passes_clean() {
    let (mock_lakera_port, _lakera_guard) = start_mock_lakera(false);
    let backend_guard = start_backend_with_shutdown("upstream ok");
    let proxy_port = free_port();

    let config = load_lakera_config(proxy_port, mock_lakera_port, backend_guard.port());
    let proxy = start_proxy(&config);

    let payload = r#"{"model":"test","messages":[{"role":"user","content":"hello"}]}"#;
    let raw = http_send(
        proxy.addr(),
        &format!(
            "POST /v1/chat/completions HTTP/1.1\r\n\
             Host: localhost\r\n\
             Content-Type: application/json\r\n\
             Content-Length: {}\r\n\
             Connection: close\r\n\r\n\
             {payload}",
            payload.len()
        ),
    );

    assert_eq!(parse_status(&raw), 200, "clean request should return 200 from upstream");
    assert!(
        raw.contains("upstream ok"),
        "clean request should receive upstream body"
    );
}

#[test]
fn lakera_guard_bypasses_non_post_requests() {
    // The example scopes the callout with `methods: [POST]`. The mock
    // flags everything, so a 200 proves the filter never fired.
    let (mock_lakera_port, _lakera_guard) = start_mock_lakera(true);
    let backend_guard = start_backend_with_shutdown("upstream ok");
    let proxy_port = free_port();

    let config = load_lakera_config(proxy_port, mock_lakera_port, backend_guard.port());
    let proxy = start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        "GET /v1/models HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
    );

    assert_eq!(
        parse_status(&raw),
        200,
        "GET should bypass the callout and reach upstream"
    );
    assert!(raw.contains("upstream ok"), "GET should receive upstream body");
}

#[test]
fn lakera_guard_rejects_flagged_with_preceding_filter() {
    // Branch evaluation clears `filter_results` after every filter in
    // the headers phase. With a filter preceding the callout in the
    // chain, body-phase extractions must still reach the callout's
    // branch evaluation (stashed and re-published by `on_request`).
    let (mock_lakera_port, _lakera_guard) = start_mock_lakera(true);
    let backend_guard = start_backend_with_shutdown("upstream ok");
    let proxy_port = free_port();

    let path = example_config_path("lakera-guard.yaml");
    let yaml = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let yaml = yaml
        .replace(
            "https://api.lakera.ai/v2/guard",
            &format!("http://127.0.0.1:{mock_lakera_port}/v2/guard"),
        )
        .replace(
            "            - name: \"Authorization\"\n              value: \"Bearer ${LAKERA_API_KEY}\"\n",
            "",
        )
        .replace(
            "      - filter: http_callout",
            "      - filter: token_usage_headers\n      - filter: http_callout",
        );

    let patched = patch_yaml(
        &yaml,
        proxy_port,
        &HashMap::from([("127.0.0.1:3000", backend_guard.port())]),
    );
    let config = praxis_core::config::Config::from_yaml(&patched).unwrap_or_else(|e| panic!("parse config: {e}"));
    let proxy = start_proxy(&config);

    let payload = r#"{"model":"test","messages":[{"role":"user","content":"bad"}]}"#;
    let raw = http_send(
        proxy.addr(),
        &format!(
            "POST /v1/chat/completions HTTP/1.1\r\n\
             Host: localhost\r\n\
             Content-Type: application/json\r\n\
             Content-Length: {}\r\n\
             Connection: close\r\n\r\n\
             {payload}",
            payload.len()
        ),
    );

    assert_eq!(
        parse_status(&raw),
        403,
        "flagged request should be rejected even with a filter preceding the callout"
    );
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

/// Load and patch the Lakera Guard example config.
fn load_lakera_config(proxy_port: u16, lakera_port: u16, backend_port: u16) -> praxis_core::config::Config {
    let path = example_config_path("lakera-guard.yaml");
    let yaml = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));

    // Replace the real Lakera API URL with our mock and remove
    // the Authorization header that references an env var.
    let yaml = yaml
        .replace(
            "https://api.lakera.ai/v2/guard",
            &format!("http://127.0.0.1:{lakera_port}/v2/guard"),
        )
        .replace(
            "            - name: \"Authorization\"\n              value: \"Bearer ${LAKERA_API_KEY}\"\n",
            "",
        );

    let patched = patch_yaml(&yaml, proxy_port, &HashMap::from([("127.0.0.1:3000", backend_port)]));
    praxis_core::config::Config::from_yaml(&patched).unwrap_or_else(|e| panic!("parse lakera-guard.yaml: {e}"))
}

/// Start a mock Lakera Guard backend that returns a fixed response.
///
/// Returns `(port, guard)`. The guard keeps the backend alive.
fn start_mock_lakera(flagged: bool) -> (u16, praxis_test_utils::net::backend::BackendGuard) {
    let body = if flagged {
        r#"{"flagged":true}"#
    } else {
        r#"{"flagged":false}"#
    };
    let guard = start_backend_with_shutdown(body);
    let port = guard.port();
    (port, guard)
}
