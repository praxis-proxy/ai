// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for the multimodal routing example configuration.

use std::collections::HashMap;

use praxis_test_utils::{
    BackendGuard, ProxyGuard, free_port, http_post, http_send, json_post_with_header, load_example_config,
    parse_status, start_backend_with_shutdown, start_proxy,
};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Example configuration exercised by these tests.
const EXAMPLE: &str = "multimodal-routing.yaml";

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn image_url_routes_to_image_cluster() {
    let (proxy, _backends) = start_gateway();
    let body = r#"{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.test/image.png"}}]}]}"#;

    let (status, response) = http_post(proxy.addr(), "/v1/chat/completions", body);

    assert_eq!(status, 200, "image request should be proxied");
    assert_eq!(response, "image", "image content should select the image cluster");
}

#[test]
fn base64_audio_routes_to_audio_cluster() {
    let (proxy, _backends) = start_gateway();
    let body = r#"{"messages":[{"role":"user","content":[{"type":"input_audio","input_audio":{"data":"UklGRg==","format":"wav"}}]}]}"#;

    let (status, response) = http_post(proxy.addr(), "/v1/chat/completions", body);

    assert_eq!(status, 200, "audio request should be proxied");
    assert_eq!(response, "audio", "audio content should select the audio cluster");
}

#[test]
fn image_and_audio_route_to_combined_cluster() {
    let (proxy, _backends) = start_gateway();
    let body = r#"{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw=="}},{"type":"input_audio","input_audio":{"data":"UklGRg==","format":"wav"}}]}]}"#;

    let (status, response) = http_post(proxy.addr(), "/v1/chat/completions", body);

    assert_eq!(status, 200, "combined request should be proxied");
    assert_eq!(
        response, "image-audio",
        "combined content should select the image-audio cluster"
    );
}

#[test]
fn text_only_request_stays_on_text_cluster() {
    let (proxy, _backends) = start_gateway();
    let body = r#"{"messages":[{"role":"user","content":"hello"}]}"#;

    let (status, response) = http_post(proxy.addr(), "/v1/chat/completions", body);

    assert_eq!(status, 200, "text request should be proxied");
    assert_eq!(response, "text", "text-only content should use the default cluster");
}

#[test]
fn client_supplied_internal_modality_header_is_rejected() {
    let (proxy, _backends) = start_gateway();
    let body = r#"{"messages":[{"role":"user","content":"hello"}]}"#;
    let request = json_post_with_header("/v1/chat/completions", body, "X-Praxis-AI-Modality: image_audio");

    let raw = http_send(proxy.addr(), &request);

    assert_eq!(
        parse_status(&raw),
        400,
        "client-supplied internal routing facts must fail closed: {raw}"
    );
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

/// Start the example gateway and all modality-specific backends.
fn start_gateway() -> (ProxyGuard, [BackendGuard; 4]) {
    let image_audio = start_backend_with_shutdown("image-audio");
    let image = start_backend_with_shutdown("image");
    let audio = start_backend_with_shutdown("audio");
    let text = start_backend_with_shutdown("text");
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
