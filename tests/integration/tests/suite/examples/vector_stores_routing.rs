// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for the vector-stores-routing example config.

use std::collections::HashMap;

use praxis_test_utils::{free_port, http_get, load_example_config, start_backend_with_shutdown, start_proxy};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn vector_stores_root_routes_to_dedicated_backend() {
    let vs_backend = start_backend_with_shutdown("vector-stores-backend");
    let default = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let config = load_example_config(
        "openai/responses/vector-stores-routing.yaml",
        proxy_port,
        HashMap::from([
            ("127.0.0.1:3001", vs_backend.port()),
            ("127.0.0.1:3002", default.port()),
        ]),
    );
    let proxy = start_proxy(&config);

    let (status, body) = http_get(proxy.addr(), "/v1/vector_stores", None);

    assert_eq!(status, 200, "root should return 200");
    assert_eq!(
        body, "vector-stores-backend",
        "/v1/vector_stores should route to vector-stores-backend"
    );
}

#[test]
fn vector_stores_nested_subresource_routes_to_dedicated_backend() {
    let vs_backend = start_backend_with_shutdown("vector-stores-backend");
    let default = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let config = load_example_config(
        "openai/responses/vector-stores-routing.yaml",
        proxy_port,
        HashMap::from([
            ("127.0.0.1:3001", vs_backend.port()),
            ("127.0.0.1:3002", default.port()),
        ]),
    );
    let proxy = start_proxy(&config);

    let (status, body) = http_get(proxy.addr(), "/v1/vector_stores/vs_abc/files", None);

    assert_eq!(status, 200, "nested subresource should return 200");
    assert_eq!(
        body, "vector-stores-backend",
        "/v1/vector_stores/vs_abc/files should route to vector-stores-backend"
    );
}

#[test]
fn vector_stores_extra_segment_boundary_routes_to_default() {
    let vs_backend = start_backend_with_shutdown("vector-stores-backend");
    let default = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let config = load_example_config(
        "openai/responses/vector-stores-routing.yaml",
        proxy_port,
        HashMap::from([
            ("127.0.0.1:3001", vs_backend.port()),
            ("127.0.0.1:3002", default.port()),
        ]),
    );
    let proxy = start_proxy(&config);

    let (status, body) = http_get(proxy.addr(), "/v1/vector_stores_extra", None);

    assert_eq!(status, 200, "segment boundary path should return 200");
    assert_eq!(
        body, "default-backend",
        "/v1/vector_stores_extra must NOT match /v1/vector_stores prefix"
    );
}
