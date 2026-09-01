// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Tests for the GCP ADC example configuration.
//!
//! `GcpAdcFilter` acquires a token inline, on the request that finds the
//! cache stale (cache-through, not refresh-ahead). This test points
//! `metadata_host` at a closed local port so the inline fetch fails
//! deterministically (no real network call to `metadata.google.internal`)
//! and no token is ever cached: every request must fail closed with 503.
//! The full "token injected, reaches upstream" path is covered by the
//! unit tests in `filters/src/gcp/`.

use std::collections::HashMap;

use praxis_test_utils::{free_port, http_send, parse_status, start_header_echo_backend};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn gcp_adc_config_parses() {
    let config = super::load_example_config("gcp-adc.yaml", 29912, HashMap::from([("127.0.0.1:3000", 29913_u16)]));

    assert_eq!(config.listeners.len(), 1, "should have 1 listener");
    assert_eq!(&*config.listeners[0].name, "gateway", "listener name should be gateway");
}

#[test]
fn gcp_adc_fails_closed_without_token() {
    let backend_guard = start_header_echo_backend();
    let backend_port = backend_guard.port();
    let proxy_port = free_port();

    let path = praxis_test_utils::example_config_path("gcp-adc.yaml");
    let yaml = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let patched = praxis_test_utils::patch_yaml(&yaml, proxy_port, &HashMap::from([("127.0.0.1:3000", backend_port)]));
    // Point the metadata server at a closed port so the inline fetch
    // fails deterministically, with no real network dependency.
    let patched = patched.replace(
        "      - filter: gcp_adc",
        "      - filter: gcp_adc\n        metadata_host: 127.0.0.1:1",
    );
    let config = praxis_core::config::Config::from_yaml(&patched).unwrap_or_else(|e| panic!("parse gcp-adc.yaml: {e}"));

    let proxy = praxis_test_utils::start_proxy(&config);
    let raw = http_send(
        proxy.addr(),
        "POST /v1/models HTTP/1.1\r\n\
         Host: localhost\r\n\
         Content-Length: 15\r\n\
         Connection: close\r\n\r\n\
         {\"prompt\":\"hi\"}",
    );

    assert_eq!(
        parse_status(&raw),
        503,
        "with no acquirable token the filter must fail closed with 503: {raw}"
    );
}
