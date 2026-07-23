// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for the OpenAI Embeddings routing example config.

use std::collections::HashMap;

use praxis_test_utils::{free_port, http_get, load_example_config, start_backend_with_shutdown, start_proxy};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn example_config_routes_embeddings_root_to_embeddings_backend() {
    let embeddings_guard = start_backend_with_shutdown("embeddings-api");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let config = load_example_config(
        "openai/embeddings/embeddings-routing.yaml",
        proxy_port,
        HashMap::from([
            ("127.0.0.1:3001", embeddings_guard.port()),
            ("127.0.0.1:3002", default_guard.port()),
        ]),
    );
    let proxy = start_proxy(&config);

    let (status, body) = http_get(proxy.addr(), "/v1/embeddings", None);
    assert_eq!(status, 200, "Embeddings API root should be proxied");
    assert_eq!(
        body, "embeddings-api",
        "GET /v1/embeddings should route to embeddings-api backend"
    );
}

#[test]
fn example_config_routes_embeddings_subresource_to_embeddings_backend() {
    let embeddings_guard = start_backend_with_shutdown("embeddings-api");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let config = load_example_config(
        "openai/embeddings/embeddings-routing.yaml",
        proxy_port,
        HashMap::from([
            ("127.0.0.1:3001", embeddings_guard.port()),
            ("127.0.0.1:3002", default_guard.port()),
        ]),
    );
    let proxy = start_proxy(&config);

    let (status, body) = http_get(proxy.addr(), "/v1/embeddings/emb-abc123", None);
    assert_eq!(status, 200, "Embeddings API subresource should be proxied");
    assert_eq!(
        body, "embeddings-api",
        "Embeddings API subresources should route to embeddings-api backend"
    );
}

#[test]
fn example_config_segment_boundary_falls_to_default() {
    let embeddings_guard = start_backend_with_shutdown("embeddings-api");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let config = load_example_config(
        "openai/embeddings/embeddings-routing.yaml",
        proxy_port,
        HashMap::from([
            ("127.0.0.1:3001", embeddings_guard.port()),
            ("127.0.0.1:3002", default_guard.port()),
        ]),
    );
    let proxy = start_proxy(&config);

    let (status, body) = http_get(proxy.addr(), "/v1/embeddingsomething", None);
    assert_eq!(status, 200, "non-Embeddings API path should be proxied");
    assert_eq!(
        body, "default-backend",
        "path-prefix matching must stop at a segment boundary"
    );
}
