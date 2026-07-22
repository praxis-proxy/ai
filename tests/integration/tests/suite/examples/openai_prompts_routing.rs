// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for the OpenAI Prompts routing example config.

use std::collections::HashMap;

use praxis_test_utils::{free_port, http_get, load_example_config, start_backend_with_shutdown, start_proxy};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn example_config_routes_prompts_root_to_prompts_backend() {
    let prompts_guard = start_backend_with_shutdown("prompts-api");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let config = load_example_config(
        "openai/prompts/prompts-routing.yaml",
        proxy_port,
        HashMap::from([
            ("127.0.0.1:3001", prompts_guard.port()),
            ("127.0.0.1:3002", default_guard.port()),
        ]),
    );
    let proxy = start_proxy(&config);

    let (status, body) = http_get(proxy.addr(), "/v1/prompts", None);
    assert_eq!(status, 200, "Prompts API root should be proxied");
    assert_eq!(
        body, "prompts-api",
        "GET /v1/prompts should route to prompts-api backend"
    );
}

#[test]
fn example_config_routes_prompts_subresource_to_prompts_backend() {
    let prompts_guard = start_backend_with_shutdown("prompts-api");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let config = load_example_config(
        "openai/prompts/prompts-routing.yaml",
        proxy_port,
        HashMap::from([
            ("127.0.0.1:3001", prompts_guard.port()),
            ("127.0.0.1:3002", default_guard.port()),
        ]),
    );
    let proxy = start_proxy(&config);

    let (status, body) = http_get(proxy.addr(), "/v1/prompts/prompt-abc123/versions", None);
    assert_eq!(status, 200, "Prompts API subresource should be proxied");
    assert_eq!(
        body, "prompts-api",
        "Prompts API subresources should route to prompts-api backend"
    );
}

#[test]
fn example_config_segment_boundary_falls_to_default() {
    let prompts_guard = start_backend_with_shutdown("prompts-api");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let config = load_example_config(
        "openai/prompts/prompts-routing.yaml",
        proxy_port,
        HashMap::from([
            ("127.0.0.1:3001", prompts_guard.port()),
            ("127.0.0.1:3002", default_guard.port()),
        ]),
    );
    let proxy = start_proxy(&config);

    let (status, body) = http_get(proxy.addr(), "/v1/promptsomething", None);
    assert_eq!(status, 200, "non-Prompts API path should be proxied");
    assert_eq!(
        body, "default-backend",
        "path-prefix matching must stop at a segment boundary"
    );
}
