// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for the prompt-length routing example configuration.

use std::collections::HashMap;

use praxis_test_utils::{free_port, http_post, load_example_config, start_backend_with_shutdown, start_proxy};

/// Example configuration exercised by these tests.
const EXAMPLE: &str = "prompt-length-routing.yaml";

#[test]
fn large_prompt_routes_to_large_context_cluster() {
    let large_guard = start_backend_with_shutdown("large-context");
    let default_guard = start_backend_with_shutdown("default");
    let proxy_port = free_port();
    let config = load_example_config(
        EXAMPLE,
        proxy_port,
        HashMap::from([
            ("127.0.0.1:3001", large_guard.port()),
            ("127.0.0.1:3002", default_guard.port()),
        ]),
    );
    let proxy = start_proxy(&config);

    let (status, body) = http_post(
        proxy.addr(),
        "/v1/chat/completions",
        r#"{"model":"small-model","messages":[{"role":"user","content":"Explain this very long prompt in enough detail to overflow the configured input limit."}]}"#,
    );

    assert_eq!(status, 200, "large prompt should be proxied");
    assert_eq!(body, "large-context", "large prompt should use the overflow cluster");
}

#[test]
fn small_prompt_stays_on_default_cluster() {
    let large_guard = start_backend_with_shutdown("large-context");
    let default_guard = start_backend_with_shutdown("default");
    let proxy_port = free_port();
    let config = load_example_config(
        EXAMPLE,
        proxy_port,
        HashMap::from([
            ("127.0.0.1:3001", large_guard.port()),
            ("127.0.0.1:3002", default_guard.port()),
        ]),
    );
    let proxy = start_proxy(&config);

    let (status, body) = http_post(
        proxy.addr(),
        "/v1/chat/completions",
        r#"{"model":"small-model","messages":[{"role":"user","content":"hello"}]}"#,
    );

    assert_eq!(status, 200, "small prompt should be proxied");
    assert_eq!(body, "default", "small prompt should not escalate");
}
