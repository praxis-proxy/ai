// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Tests for the Azure AD (Entra ID) example configuration.
//!
//! `AzureAdFilter` reads its client secret from the environment
//! variable named in `client_secret_env_var`. Since `std::env::set_var`
//! is `unsafe` in this edition and `unsafe_code` is denied
//! workspace-wide (see `aws_sigv4.rs` for the same constraint), this
//! test patches that field to name `CARGO_PKG_NAME`, which Cargo always
//! sets for a test binary — so construction succeeds with no env
//! mutation.
//!
//! The token itself is acquired in the background from the configured
//! authority. This test points `authority_host` at a closed local port
//! so no real network call is made and no token is ever cached: the
//! cache stays empty and every request must fail closed with 503. The
//! full "token injected, reaches upstream" path is covered by the unit
//! tests in `filters/src/azure/azure_ad.rs`.

use std::collections::HashMap;

use praxis_test_utils::{free_port, http_send, parse_status, start_header_echo_backend};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn azure_ad_config_parses() {
    let config = super::load_example_config("azure-ad.yaml", 29912, HashMap::from([("127.0.0.1:3000", 29913_u16)]));

    assert_eq!(config.listeners.len(), 1, "should have 1 listener");
    assert_eq!(&*config.listeners[0].name, "gateway", "listener name should be gateway");
}

#[test]
fn azure_ad_fails_closed_without_token() {
    let backend_guard = start_header_echo_backend();
    let backend_port = backend_guard.port();
    let proxy_port = free_port();

    let path = praxis_test_utils::example_config_path("azure-ad.yaml");
    let yaml = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let patched = praxis_test_utils::patch_yaml(&yaml, proxy_port, &HashMap::from([("127.0.0.1:3000", backend_port)]));
    // Use an always-set env var so filter construction succeeds, and
    // point the authority at a closed port so the background refresher
    // can never acquire a token (deterministic fail-closed).
    let patched = patched.replace(
        "        client_secret_env_var: AZURE_CLIENT_SECRET",
        "        client_secret_env_var: CARGO_PKG_NAME\n        authority_host: 127.0.0.1:1",
    );
    let config =
        praxis_core::config::Config::from_yaml(&patched).unwrap_or_else(|e| panic!("parse azure-ad.yaml: {e}"));

    let proxy = praxis_test_utils::start_proxy(&config);
    let raw = http_send(
        proxy.addr(),
        "POST /openai/deployments/gpt-4o/chat/completions HTTP/1.1\r\n\
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
