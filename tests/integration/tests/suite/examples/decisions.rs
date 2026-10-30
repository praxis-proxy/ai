// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for the OpenAI Decisions pass-through example config.

use std::collections::HashMap;

use praxis_test_utils::{
    free_port, http_send, json_post, load_example_config, parse_body, parse_status, start_backend_with_shutdown,
    start_proxy,
};

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

/// Running the decisions example with the decisions and default backends.
struct Harness {
    /// Decisions cluster.
    _decisions: praxis_test_utils::BackendGuard,
    /// Default cluster.
    _default: praxis_test_utils::BackendGuard,
    /// The running proxy.
    proxy: praxis_test_utils::ProxyGuard,
}

/// Start the example config against both backends.
fn start() -> Harness {
    let decisions = start_backend_with_shutdown("decisions-backend");
    let default = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();
    let config = load_example_config(
        "openai/decisions.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3001", decisions.port()), ("127.0.0.1:3002", default.port())]),
    );
    let proxy = start_proxy(&config);
    Harness {
        _decisions: decisions,
        _default: default,
        proxy,
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn decisions_example_routes_a_create_decision_request() {
    let h = start();

    let body = r#"{"model":"gpt-6-luna","input":"Order arrived damaged.","questions":[{"type":"predicate","name":"damaged","instructions":"Does the text report damage?"}]}"#;
    let raw = http_send(h.proxy.addr(), &json_post("/v1/decisions", body));

    assert_eq!(parse_status(&raw), 200, "decisions request should return 200");
    assert_eq!(
        parse_body(&raw),
        "decisions-backend",
        "POST /v1/decisions should route to the decisions cluster"
    );
}

#[test]
fn decisions_example_passes_the_body_through_unmodified() {
    let h = start();

    // The proxy does not own the Decisions contract: an unusual but
    // well-formed body must reach the backend unchanged rather than be
    // rejected or rewritten locally.
    let body = r#"{"model":"gpt-6-luna","input":"x","questions":[{"type":"score","name":"s","instructions":"?","levels":[{"label":"a","description":"a"},{"label":"b","description":"b"}]}]}"#;
    let raw = http_send(h.proxy.addr(), &json_post("/v1/decisions?foo=1", body));

    assert_eq!(parse_status(&raw), 200, "query string should not affect routing");
    assert_eq!(
        parse_body(&raw),
        "decisions-backend",
        "the request should still reach the decisions cluster"
    );
}

#[test]
fn decisions_example_leaves_unrelated_traffic_on_the_default_cluster() {
    let h = start();

    let body = r#"{"model":"gpt-6-luna","input":"Order arrived damaged.","questions":[]}"#;
    let raw = http_send(h.proxy.addr(), &json_post("/v1/chat/completions", body));

    assert_eq!(parse_status(&raw), 200, "chat completions request should return 200");
    assert_eq!(
        parse_body(&raw),
        "default-backend",
        "non-decisions traffic should fall through to the default cluster"
    );
}
