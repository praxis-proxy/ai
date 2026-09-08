// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for the body-size-enforcement example config.
//!
//! Proves that `body_limits.max_request_bytes` — and only that transport
//! ceiling — governs the raw request body across a chain of body-buffering
//! OpenAI Responses filters, each of which declares a 64 MiB StreamBuffer
//! mode. A request under the cap is forwarded; a request over the cap is
//! rejected with 413 before any filter processes it.

use std::collections::HashMap;

use praxis_test_utils::{
    free_port, http_send, json_post, load_example_config, parse_body, parse_status, start_backend_with_shutdown,
    start_proxy,
};

const CONFIG: &str = "openai/responses/body-size-limits.yaml";

#[test]
fn small_responses_request_within_transport_cap_is_forwarded() {
    let inference_guard = start_backend_with_shutdown("inference-backend");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let config = load_example_config(
        CONFIG,
        proxy_port,
        HashMap::from([
            ("127.0.0.1:3001", inference_guard.port()),
            ("127.0.0.1:3002", default_guard.port()),
        ]),
    );
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"Hello, world!"}"#;
    assert!(
        body.len() < 1024,
        "fixture body must stay under the 1 KiB transport cap"
    );
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(
        parse_status(&raw),
        200,
        "request under the transport cap should be forwarded"
    );
    assert_eq!(
        parse_body(&raw),
        "inference-backend",
        "a small Responses request should reach the inference backend"
    );
}

#[test]
fn oversized_responses_request_is_rejected_with_413() {
    let inference_guard = start_backend_with_shutdown("inference-backend");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let config = load_example_config(
        CONFIG,
        proxy_port,
        HashMap::from([
            ("127.0.0.1:3001", inference_guard.port()),
            ("127.0.0.1:3002", default_guard.port()),
        ]),
    );
    let proxy = start_proxy(&config);

    // Pad the input past the 1 KiB max_request_bytes transport ceiling.
    let body = format!(r#"{{"model":"gpt-4.1","input":"{}"}}"#, "x".repeat(2000));
    assert!(body.len() > 1024, "fixture body must exceed the 1 KiB transport cap");
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &body));

    assert_eq!(
        parse_status(&raw),
        413,
        "raw body exceeding max_request_bytes should be rejected by the transport cap"
    );
}
