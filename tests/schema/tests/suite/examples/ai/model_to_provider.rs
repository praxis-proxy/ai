// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional example coverage for `model_to_provider`.

use praxis_core::config::Config;
use praxis_test_utils::{free_port, http_post, start_capturing_backend, start_proxy};

#[test]
fn model_to_provider_routes_and_rewrites_stable_model_id() {
    let provider = start_capturing_backend(r#"{"ok":true}"#);
    let proxy_port = free_port();

    let yaml = make_yaml(proxy_port, provider.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let (status, response) = http_post(
        proxy.addr(),
        "/v1/messages",
        r#"{"model":"claude-sonnet-4-5","messages":[]}"#,
    );
    assert_eq!(status, 200, "mapped model should route to its provider");
    assert_eq!(response, r#"{"ok":true}"#);

    let forwarded: serde_yaml::Value = serde_yaml::from_str(&provider.body()).unwrap();
    assert_eq!(forwarded["model"], "vertex/claude-sonnet-4-5");
}

#[test]
fn model_to_provider_leaves_unmapped_model_body_unchanged() {
    let provider = start_capturing_backend(r#"{"ok":true}"#);
    let proxy_port = free_port();

    let yaml = make_yaml(proxy_port, provider.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let (status, response) = http_post(
        proxy.addr(),
        "/v1/messages",
        r#"{"model":"unknown-model","messages":[]}"#,
    );
    assert_eq!(status, 200, "unmapped models continue to the configured default route");
    assert_eq!(response, r#"{"ok":true}"#);
    let forwarded: serde_yaml::Value = serde_yaml::from_str(&provider.body()).unwrap();
    assert_eq!(forwarded["model"], "unknown-model");
}

fn make_yaml(proxy_port: u16, provider_port: u16) -> String {
    include_str!("../../../../../../examples/configs/model-to-provider.yaml")
        .replace("127.0.0.1:8080", &format!("127.0.0.1:{proxy_port}"))
        .replace("127.0.0.1:9001", &format!("127.0.0.1:{provider_port}"))
}
