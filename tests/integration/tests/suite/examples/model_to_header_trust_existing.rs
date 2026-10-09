// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Tests for the `model-to-header-trust-existing.yaml` example.
//!
//! With `unless: {headers_present: ["X-AI-Model"]}`, the `model_to_header`
//! filter is skipped when the header is already present, so the incoming value
//! reaches the router and the upstream unchanged. A request without the header
//! still has its body model promoted and routed as usual.

use std::collections::HashMap;

use praxis_test_utils::{
    CapturedRequest, StatefulCapturingBackend, StatefulCapturingGuard, free_port, http_send, parse_body, parse_status,
    start_proxy,
};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// The example config under test.
const EXAMPLE: &str = "model-to-header-trust-existing.yaml";

/// The `mistral` cluster endpoint as written in the example.
const MISTRAL_ENDPOINT: &str = "127.0.0.1:3000";

/// The `granite` cluster endpoint as written in the example.
const GRANITE_ENDPOINT: &str = "127.0.0.1:3001";

/// The `default` cluster endpoint as written in the example.
const DEFAULT_ENDPOINT: &str = "127.0.0.1:3002";

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn existing_header_routes_over_body_model() {
    let backends = Backends::start();
    let proxy = start_proxy(&backends.config());

    let raw = http_send(
        proxy.addr(),
        &chat_post(
            r#"{"model":"mistral-7b-instruct","messages":[]}"#,
            &["X-AI-Model: granite-3.1-8b"],
        ),
    );

    assert_eq!(parse_status(&raw), 200, "request should be routed: {raw}");
    assert_eq!(
        parse_body(&raw),
        "granite-backend",
        "the existing header must pick the cluster, not the body model"
    );
    assert!(
        backends.mistral.requests().is_empty(),
        "the body model must not steer routing when the header is present"
    );
    assert_eq!(
        model_header_values(&backends.granite.requests()),
        ["granite-3.1-8b"],
        "the upstream must see the incoming header once, unchanged"
    );
}

#[test]
fn missing_header_is_promoted_from_body() {
    let backends = Backends::start();
    let proxy = start_proxy(&backends.config());

    let raw = http_send(
        proxy.addr(),
        &chat_post(r#"{"model":"mistral-7b-instruct","messages":[]}"#, &[]),
    );

    assert_eq!(parse_status(&raw), 200, "request should be routed: {raw}");
    assert_eq!(
        parse_body(&raw),
        "mistral-backend",
        "without the header the body model should be promoted and routed"
    );
    assert_eq!(
        model_header_values(&backends.mistral.requests()),
        ["mistral-7b-instruct"],
        "the upstream should see the promoted body model"
    );
}

#[test]
fn existing_header_routes_without_body_model() {
    let backends = Backends::start();
    let proxy = start_proxy(&backends.config());

    let raw = http_send(
        proxy.addr(),
        &chat_post(r#"{"messages":[]}"#, &["X-AI-Model: mistral-7b-instruct"]),
    );

    assert_eq!(parse_status(&raw), 200, "request should be routed: {raw}");
    assert_eq!(
        parse_body(&raw),
        "mistral-backend",
        "an existing header should route even when the body names no model"
    );
    assert_eq!(
        model_header_values(&backends.mistral.requests()),
        ["mistral-7b-instruct"],
        "the upstream must see the incoming header once, unchanged"
    );
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

/// One capturing backend per cluster in the example.
struct Backends {
    /// Backend for the `default` cluster. The harness's readiness probe
    /// (`GET /`, no model header) lands here, so it is never asserted empty.
    default: StatefulCapturingGuard,
    /// Backend for the `granite` cluster.
    granite: StatefulCapturingGuard,
    /// Backend for the `mistral` cluster.
    mistral: StatefulCapturingGuard,
}

impl Backends {
    /// Start a capturing backend for each cluster, each answering once with
    /// its own name.
    fn start() -> Self {
        let start =
            |name: &str| StatefulCapturingBackend::new(vec![(200, format!("{name}-backend"))]).start_with_shutdown();
        Self {
            default: start("default"),
            granite: start("granite"),
            mistral: start("mistral"),
        }
    }

    fn config(&self) -> praxis_core::config::Config {
        super::load_example_config(
            EXAMPLE,
            free_port(),
            HashMap::from([
                (MISTRAL_ENDPOINT, self.mistral.port()),
                (GRANITE_ENDPOINT, self.granite.port()),
                (DEFAULT_ENDPOINT, self.default.port()),
            ]),
        )
    }
}

/// Build a raw chat-completions POST carrying `body` plus extra header lines
/// (each already `Name: value`, no CRLF).
fn chat_post(body: &str, extra_headers: &[&str]) -> String {
    let headers: String = extra_headers.iter().map(|line| format!("{line}\r\n")).collect();
    format!(
        "POST /v1/chat/completions HTTP/1.1\r\n\
         Host: localhost\r\n\
         Content-Type: application/json\r\n\
         {headers}\
         Content-Length: {}\r\n\
         Connection: close\r\n\r\n\
         {body}",
        body.len()
    )
}

/// Every `x-ai-model` value across the captured requests, in order.
fn model_header_values(requests: &[CapturedRequest]) -> Vec<String> {
    requests
        .iter()
        .flat_map(|request| request.headers.lines())
        .filter_map(|line| line.split_once(':'))
        .filter(|(name, _)| name.trim().eq_ignore_ascii_case("x-ai-model"))
        .map(|(_, value)| value.trim().to_owned())
        .collect()
}
