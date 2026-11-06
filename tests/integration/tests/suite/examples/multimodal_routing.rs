// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for the multimodal routing example configuration.

use std::{
    collections::HashMap,
    sync::{Mutex, MutexGuard, PoisonError},
};

use praxis_core::config::Config;
use praxis_test_utils::{
    ProxyGuard, StatefulCapturingBackend, StatefulCapturingGuard, free_port, http_post_retry, http_send, json_post,
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
    let (proxy, _backends) = start_gateway();
    let body = r#"{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.test/image.png"}}]}]}"#;

    let (status, response) = http_post_retry(proxy.addr(), "/v1/chat/completions", body);

    assert_eq!(status, 200, "image request should be proxied: {response}");
    assert_eq!(response, "image", "image content should select the image cluster");
}

/// Route inline base64 audio to an audio-capable backend.
#[test]
fn base64_audio_routes_to_audio_cluster() {
    let _lock = routing_test_lock();
    let (proxy, _backends) = start_gateway();
    let body = r#"{"messages":[{"role":"user","content":[{"type":"input_audio","input_audio":{"data":"UklGRg==","format":"wav"}}]}]}"#;

    let (status, response) = http_post_retry(proxy.addr(), "/v1/chat/completions", body);

    assert_eq!(status, 200, "audio request should be proxied: {response}");
    assert_eq!(response, "audio", "audio content should select the audio cluster");
}

/// Route combined image and audio content to the combined-capability backend.
#[test]
fn image_and_audio_route_to_combined_cluster() {
    let _lock = routing_test_lock();
    let (proxy, _backends) = start_gateway();
    let body = r#"{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw=="}},{"type":"input_audio","input_audio":{"data":"UklGRg==","format":"wav"}}]}]}"#;

    let (status, response) = http_post_retry(proxy.addr(), "/v1/chat/completions", body);

    assert_eq!(status, 200, "combined request should be proxied: {response}");
    assert_eq!(
        response, "image-audio",
        "combined content should select the image-audio cluster"
    );
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

/// Remove a client-supplied custom modality value before routing.
#[test]
fn client_supplied_custom_modality_header_is_removed() {
    let _lock = routing_test_lock();
    let (proxy, backends) = start_gateway_with(|config| set_custom_modality_header(config, "X-Test-Modality"));
    let body = r#"{"messages":[{"role":"user","content":"hello"}]}"#;
    let request = json_post_with_header("/v1/chat/completions", body, "X-Test-Modality: image_audio");

    let raw = http_send(proxy.addr(), &request);

    assert_eq!(parse_status(&raw), 200, "text request should be proxied: {raw}");
    assert_eq!(parse_body(&raw), "text", "the spoofed value must not affect routing");
    let requests = backends[3].requests();
    assert_eq!(requests.len(), 1, "the text backend should receive one request");
    assert!(
        !requests[0].headers.to_ascii_lowercase().contains("x-test-modality:"),
        "the filter-owned custom header must be removed before proxying"
    );
}

/// Reject an oversized multimodal request before it can reach the text backend.
#[test]
fn oversized_multimodal_request_is_rejected_before_routing() {
    let _lock = routing_test_lock();
    let (proxy, backends) = start_gateway_with(|config| set_max_body_bytes(config, 1024));
    let body = format!(
        r#"{{"messages":[{{"role":"user","content":[{{"type":"image_url","image_url":{{"url":"data:image/png;base64,{}"}}}}]}}]}}"#,
        "A".repeat(2048)
    );
    let raw = http_send(proxy.addr(), &json_post("/v1/chat/completions", &body));

    assert_eq!(
        parse_status(&raw),
        413,
        "oversized media should fail before routing: {raw}"
    );
    assert!(
        backends.iter().all(|backend| backend.requests().is_empty()),
        "an oversized request must not reach any model backend"
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
    start_gateway_with(|_| {})
}

/// Start the example after applying a test-only configuration adjustment.
fn start_gateway_with(configure: impl FnOnce(&mut Config)) -> (ProxyGuard, [StatefulCapturingGuard; 4]) {
    let image_audio = capturing_backend("image-audio");
    let image = capturing_backend("image");
    let audio = capturing_backend("audio");
    let text = capturing_backend("text");
    let proxy_port = free_port();
    let mut config = load_example_config(
        EXAMPLE,
        proxy_port,
        HashMap::from([
            ("127.0.0.1:3001", image_audio.port()),
            ("127.0.0.1:3002", image.port()),
            ("127.0.0.1:3003", audio.port()),
            ("127.0.0.1:3004", text.port()),
        ]),
    );
    configure(&mut config);
    let proxy = start_proxy(&config);

    (proxy, [image_audio, image, audio, text])
}

/// Start a backend that captures the request and returns a fixed response.
fn capturing_backend(body: &str) -> StatefulCapturingGuard {
    StatefulCapturingBackend::new(vec![(200, body.to_owned())]).start_with_shutdown()
}

/// Override the filter body budget for the oversized-request scenario.
fn set_max_body_bytes(config: &mut Config, max_body_bytes: usize) {
    let filter = filter_config_mut(config, "multimodal_to_header");
    filter.insert(
        serde_yaml::Value::from("max_body_bytes"),
        serde_yaml::Value::from(max_body_bytes),
    );
}

/// Configure a non-reserved header on both the classifier and router.
fn set_custom_modality_header(config: &mut Config, header: &str) {
    filter_config_mut(config, "multimodal_to_header")
        .insert(serde_yaml::Value::from("header"), serde_yaml::Value::from(header));

    let router = filter_config_mut(config, "router");
    let routes = router
        .get_mut(serde_yaml::Value::from("routes"))
        .and_then(serde_yaml::Value::as_sequence_mut)
        .expect("example router should define routes");
    for route in routes {
        let Some(headers) = route
            .as_mapping_mut()
            .and_then(|mapping| mapping.get_mut(serde_yaml::Value::from("headers")))
            .and_then(serde_yaml::Value::as_mapping_mut)
        else {
            continue;
        };
        if let Some(value) = headers.remove(serde_yaml::Value::from("x-praxis-ai-modality")) {
            headers.insert(serde_yaml::Value::from(header.to_ascii_lowercase()), value);
        }
    }
}

/// Return the mutable YAML mapping for one filter in the main chain.
fn filter_config_mut<'a>(config: &'a mut Config, filter_type: &str) -> &'a mut serde_yaml::Mapping {
    config
        .filter_chains
        .iter_mut()
        .flat_map(|chain| &mut chain.filters)
        .find(|entry| entry.filter_type == filter_type)
        .unwrap_or_else(|| panic!("example should define {filter_type}"))
        .config
        .as_mapping_mut()
        .expect("filter config should be a mapping")
}
