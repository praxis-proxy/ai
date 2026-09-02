// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Tests for the AWS SigV4 signing example configuration.
//!
//! `Sigv4SignFilter` sources credentials from environment variables
//! named in its config (`access_key_env_var`/`secret_key_env_var`).
//! Since `std::env::set_var` is `unsafe` in this edition and
//! `unsafe_code` is denied workspace-wide (see `web_search.rs` for
//! the same constraint), this test patches those two config fields
//! to name `CARGO_PKG_NAME`/`CARGO_MANIFEST_DIR` instead of real AWS
//! env vars — both are guaranteed to already be set by Cargo for any
//! test binary, so no env var mutation is needed at all.

use std::collections::HashMap;

use praxis_test_utils::{free_port, http_send, parse_body, parse_status, start_header_echo_backend};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn aws_sigv4_config_parses() {
    let config = super::load_example_config("aws-sigv4.yaml", 29910, HashMap::from([("127.0.0.1:3000", 29911_u16)]));

    assert_eq!(config.listeners.len(), 1, "should have 1 listener");
    assert_eq!(&*config.listeners[0].name, "gateway", "listener name should be gateway");
}

#[test]
fn aws_sigv4_signs_and_sets_host() {
    let access_key = std::env::var("CARGO_PKG_NAME").expect("CARGO_PKG_NAME is always set by cargo test");

    let backend_guard = start_header_echo_backend();
    let backend_port = backend_guard.port();
    let proxy_port = free_port();

    let path = praxis_test_utils::example_config_path("aws-sigv4.yaml");
    let yaml = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let patched = praxis_test_utils::patch_yaml(&yaml, proxy_port, &HashMap::from([("127.0.0.1:3000", backend_port)]));
    let patched = patched
        .replace(
            "access_key_env_var: AWS_ACCESS_KEY_ID",
            "access_key_env_var: CARGO_PKG_NAME",
        )
        .replace(
            "secret_key_env_var: AWS_SECRET_ACCESS_KEY",
            "secret_key_env_var: CARGO_MANIFEST_DIR",
        );
    let config =
        praxis_core::config::Config::from_yaml(&patched).unwrap_or_else(|e| panic!("parse aws-sigv4.yaml: {e}"));

    let proxy = praxis_test_utils::start_proxy(&config);
    let raw = http_send(
        proxy.addr(),
        "POST /model/anthropic.claude-3/invoke HTTP/1.1\r\n\
         Host: localhost\r\n\
         Content-Length: 15\r\n\
         Connection: close\r\n\r\n\
         {\"prompt\":\"hi\"}",
    );

    assert_eq!(parse_status(&raw), 200, "should return 200");
    let body = parse_body(&raw);
    assert!(
        body.to_lowercase()
            .contains("host: bedrock-runtime.us-east-1.amazonaws.com"),
        "upstream should receive the signed Host, not the client's Host: {body}"
    );
    assert!(
        body.contains(&format!("AWS4-HMAC-SHA256 Credential={access_key}/")),
        "upstream should receive a well-formed SigV4 Authorization header: {body}"
    );
    assert!(
        body.to_lowercase().contains("x-amz-date:"),
        "upstream should receive x-amz-date: {body}"
    );
    assert!(
        body.to_lowercase().contains("x-amz-content-sha256:"),
        "upstream should receive x-amz-content-sha256: {body}"
    );
}
