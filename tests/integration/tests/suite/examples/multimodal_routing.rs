// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for the multimodal routing example configuration.

use std::{
    collections::HashMap,
    sync::{Mutex, MutexGuard, PoisonError},
};

use praxis_test_utils::{
    ProxyGuard, StatefulCapturingBackend, StatefulCapturingGuard, free_port, http_post_retry, http_send,
    json_post_with_header, load_example_config, parse_body, parse_status, start_proxy,
};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Example configuration exercised by these tests.
const EXAMPLE: &str = "multimodal-routing.yaml";
/// Serializes gateways that reuse the example's cluster names in one process.
static ROUTING_TEST_LOCK: Mutex<()> = Mutex::new(());

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

/// Route URL-backed image content to an image-capable backend.
#[test]
fn image_url_routes_to_image_cluster() {
    let _lock = routing_test_lock();
    let (proxy, backends) = start_gateway();
    let body = r#"{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.test/image.png"}}]}]}"#;

    let (status, response) = http_post_retry(proxy.addr(), "/v1/chat/completions", body);

    assert_eq!(status, 200, "image request should be proxied: {response}");
    assert_eq!(response, "image", "image content should select the image cluster");
    assert_routed_unchanged(&backends, 1, body);
}

/// Route inline base64 audio to an audio-capable backend.
#[test]
fn base64_audio_routes_to_audio_cluster() {
    let _lock = routing_test_lock();
    let (proxy, backends) = start_gateway();
    let body = r#"{"messages":[{"role":"user","content":[{"type":"input_audio","input_audio":{"data":"UklGRg==","format":"wav"}}]}]}"#;

    let (status, response) = http_post_retry(proxy.addr(), "/v1/chat/completions", body);

    assert_eq!(status, 200, "audio request should be proxied: {response}");
    assert_eq!(response, "audio", "audio content should select the audio cluster");
    assert_routed_unchanged(&backends, 2, body);
}

/// Route combined image and audio content to the combined-capability backend.
#[test]
fn image_and_audio_route_to_combined_cluster() {
    let _lock = routing_test_lock();
    let (proxy, backends) = start_gateway();
    let body = r#"{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw=="}},{"type":"input_audio","input_audio":{"data":"UklGRg==","format":"wav"}}]}]}"#;

    let (status, response) = http_post_retry(proxy.addr(), "/v1/chat/completions", body);

    assert_eq!(status, 200, "combined request should be proxied: {response}");
    assert_eq!(
        response, "image-audio",
        "combined content should select the image-audio cluster"
    );
    assert_routed_unchanged(&backends, 0, body);
}

/// Keep text-only content on the example's default text backend.
#[test]
fn text_only_request_stays_on_text_cluster() {
    let _lock = routing_test_lock();
    let (proxy, _backends) = start_gateway();
    let body = r#"{"messages":[{"role":"user","content":"hello"}]}"#;

    let (status, response) = http_post_retry(proxy.addr(), "/v1/chat/completions", body);

    assert_eq!(status, 200, "text request should be proxied: {response}");
    assert_eq!(response, "text", "text-only content should use the default cluster");
}

/// Ignore an arbitrary client-supplied modality header.
#[test]
fn client_supplied_modality_header_cannot_influence_routing() {
    let _lock = routing_test_lock();
    let (proxy, _backends) = start_gateway();
    let body = r#"{"messages":[{"role":"user","content":"hello"}]}"#;
    let request = json_post_with_header("/v1/chat/completions", body, "X-Content-Modality: image_audio");

    let raw = http_send(proxy.addr(), &request);

    assert_eq!(parse_status(&raw), 200, "text request should be proxied: {raw}");
    assert_eq!(
        parse_body(&raw),
        "text",
        "client headers must not affect modality routing"
    );
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

/// Acquire the module lock without cascading failures after a poisoned test.
fn routing_test_lock() -> MutexGuard<'static, ()> {
    ROUTING_TEST_LOCK.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Start the unmodified example gateway and all modality-specific backends.
fn start_gateway() -> (ProxyGuard, [StatefulCapturingGuard; 4]) {
    let image_audio = capturing_backend("image-audio");
    let image = capturing_backend("image");
    let audio = capturing_backend("audio");
    let text = capturing_backend("text");
    let proxy_port = free_port();
    let config = load_example_config(
        EXAMPLE,
        proxy_port,
        HashMap::from([
            ("127.0.0.1:3001", image_audio.port()),
            ("127.0.0.1:3002", image.port()),
            ("127.0.0.1:3003", audio.port()),
            ("127.0.0.1:3004", text.port()),
        ]),
    );
    let proxy = start_proxy(&config);

    (proxy, [image_audio, image, audio, text])
}

/// Start a backend that captures the request and returns a fixed response.
fn capturing_backend(body: &str) -> StatefulCapturingGuard {
    StatefulCapturingBackend::new(vec![(200, body.to_owned())]).start_with_shutdown()
}

/// Assert that exactly one backend received the original request body.
fn assert_routed_unchanged(backends: &[StatefulCapturingGuard; 4], selected: usize, body: &str) {
    for (index, backend) in backends.iter().enumerate() {
        let requests = backend.requests();
        if index == selected {
            assert_eq!(requests.len(), 1, "selected backend should receive one request");
            assert_eq!(requests[0].body, body, "routing must not mutate the request body");
        } else {
            assert!(requests.is_empty(), "unselected backend {index} received a request");
        }
    }
}
