// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Configured inspection bounds through the intelligent-routing example.

use std::collections::HashMap;

use praxis_core::config::Config;
use praxis_test_utils::{
    StatefulCapturingBackend, free_port, http_send, load_example_config, parse_status, start_proxy,
    start_stateful_backend,
};

/// The example opts into 32 MiB request and buffer limits.
const EXAMPLE: &str = "intelligent-route-inference.yaml";

#[test]
fn model_inspection_forwards_more_than_ten_mib_without_changing_body_or_trusting_header() {
    let granite = start_stateful_backend(vec![]);
    let llama = StatefulCapturingBackend::new(vec![(200, "llama".to_owned())]).start_with_shutdown();
    let config = example_config(granite.port(), llama.port());
    let proxy = start_proxy(&config);
    let body = model_body(12_582_912);
    let raw = http_send(proxy.addr(), &request(&body, false));
    assert_eq!(parse_status(&raw), 200, "12 MiB must fit the example's bounds: {raw}");
    let requests = llama.requests();
    assert_eq!(
        requests.len(),
        1,
        "exactly one request must reach the body-selected backend"
    );
    assert_eq!(requests[0].body.len(), body.len());
    assert!(
        requests[0].body == body,
        "inspection must forward the original body intact"
    );
    assert!(
        requests[0]
            .headers
            .to_ascii_lowercase()
            .contains("x-model: llama-3.2-8b"),
        "the body-derived model must replace the spoofed header"
    );
    assert!(
        granite.requests().is_empty(),
        "the spoofed model must not select an upstream"
    );
}

#[test]
fn model_inspection_enforces_both_bounds_for_fixed_and_chunked_requests() {
    for (request_cap, buffer_cap) in [(Some(2048), 1024), (Some(1024), 2048), (None, 1024)] {
        for chunked in [false, true] {
            let granite = start_stateful_backend(vec![]);
            let llama = start_stateful_backend(vec![(200, "llama".to_owned())]);
            let mut config = example_config(granite.port(), llama.port());
            config.body_limits.max_request_bytes = request_cap;
            config.body_limits.max_buffer_bytes = buffer_cap.try_into().unwrap();
            config.insecure_options.allow_unbounded_body = request_cap.is_none();
            let proxy = start_proxy(&config);
            let body = model_body(1024);
            let raw = http_send(proxy.addr(), &request(&body, chunked));
            assert_eq!(parse_status(&raw), 200, "exactly at the effective bound: {raw}");

            let body = model_body(1025);
            let raw = http_send(proxy.addr(), &request(&body, chunked));
            assert_eq!(parse_status(&raw), 413, "one byte beyond the effective bound: {raw}");
            assert_eq!(llama.requests().len(), 1, "oversize bodies must not reach the upstream");
            assert!(granite.requests().is_empty(), "spoofed routing must remain ineffective");
        }
    }
}

#[test]
fn raising_only_buffer_cap_preserves_default_request_limit() {
    let granite = start_stateful_backend(vec![]);
    let llama = start_stateful_backend(vec![]);
    let mut config = example_config(granite.port(), llama.port());
    config.body_limits.max_request_bytes = praxis_core::config::BodyLimitsConfig::default().max_request_bytes;
    let proxy = start_proxy(&config);
    let raw = http_send(
        proxy.addr(),
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n\
         Content-Length: 10485761\r\nConnection: close\r\n\r\n",
    );
    assert_eq!(
        parse_status(&raw),
        413,
        "the default 10 MiB transfer cap still applies: {raw}"
    );
    assert!(granite.requests().is_empty());
    assert!(llama.requests().is_empty());
}

/// Load the actual example with isolated listeners and upstreams.
fn example_config(granite_port: u16, llama_port: u16) -> Config {
    load_example_config(
        EXAMPLE,
        free_port(),
        HashMap::from([("127.0.0.1:8001", granite_port), ("127.0.0.1:8002", llama_port)]),
    )
}

/// Valid JSON of exactly `size` bytes, with routing information after the padding.
fn model_body(size: usize) -> String {
    let prefix = r#"{"padding":""#;
    let suffix = r#"","model":"llama-3.2-8b","messages":[]}"#;
    let padding = size
        .checked_sub(prefix.len())
        .unwrap()
        .checked_sub(suffix.len())
        .unwrap();
    format!("{prefix}{}{suffix}", "x".repeat(padding))
}

/// A spoofed model header must not bypass body inspection, even without a length.
fn request(body: &str, chunked: bool) -> String {
    let head = "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n\
                Content-Type: application/json\r\nX-Model: granite-3.3-8b\r\nConnection: close\r\n";
    if chunked {
        let (first, second) = body.split_at(body.len() / 2);
        format!(
            "{head}Transfer-Encoding: chunked\r\n\r\n{:x}\r\n{first}\r\n{:x}\r\n{second}\r\n0\r\n\r\n",
            first.len(),
            second.len()
        )
    } else {
        format!("{head}Content-Length: {}\r\n\r\n{body}", body.len())
    }
}
