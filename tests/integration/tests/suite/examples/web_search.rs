// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for the web-search example config.
//!
//! These tests verify that the example config parses, the filter
//! pipeline builds correctly, and requests pass through unchanged.
//! The `openai_web_search` filter is a scaffolded passthrough — it validates
//! config at startup but does not execute searches at runtime.
//!
//! The example config uses `${WEB_SEARCH_API_KEY}` for the API key.
//! Since `set_var` is `unsafe` in Rust 2024 and `unsafe_code` is
//! denied, we patch the YAML to replace the env var reference with a
//! literal test key.

use std::{collections::HashMap, sync::Arc};

use praxis_core::config::Config;
use praxis_test_utils::{
    free_port, http_send, json_post, parse_body, parse_status, patch_yaml, start_backend_with_shutdown, start_proxy,
};

/// Resolve the full AI pipeline from raw YAML, surfacing any build error.
///
/// Exercises the real server pipeline-resolution path — including chain-binding
/// filters resolving their `outbound_chain` — so a config whose outbound chain
/// cannot be built fails here rather than at request time.
fn resolve(yaml: &str) -> Result<(), String> {
    let config = Config::from_yaml(yaml).map_err(|error| error.to_string())?;
    let health = Arc::new(HashMap::new());
    let kv_stores = praxis_core::kv::KvStoreRegistry::new();
    let subrequest_client =
        praxis_core::subrequest::SubRequestClient::new(praxis_core::subrequest::SubRequestConnector::new(8, None));
    let registry = praxis_ai::build_full_registry(&subrequest_client);
    praxis_ai::resolve_pipelines(&config, &registry, &health, &kv_stores, &subrequest_client)
        .map(|_pipelines| ())
        .map_err(|error| error.to_string())
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// Load the web-search example config with the env var reference
/// replaced by a literal test key.
fn load_web_search_config(proxy_port: u16, port_map: &HashMap<&str, u16>) -> Config {
    let path = praxis_test_utils::example_config_path("openai/responses/web-search.yaml");
    let yaml = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let patched = patch_yaml(&yaml, proxy_port, port_map);
    let patched = patched.replace("${WEB_SEARCH_API_KEY}", "test-key-for-config");
    Config::from_yaml(&patched).unwrap_or_else(|e| panic!("parse web-search.yaml: {e}"))
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn web_search_example_passthrough() {
    let backend_guard = start_backend_with_shutdown("inference");
    let proxy_port = free_port();

    let config = load_web_search_config(proxy_port, &HashMap::from([("127.0.0.1:3001", backend_guard.port())]));
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"Hello","tools":[{"type":"web_search"}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "request should pass through to backend");
    assert_eq!(
        parse_body(&raw),
        "inference",
        "openai_web_search filter is a passthrough — request should reach inference backend"
    );
}

/// #958: a chain-binding filter binds its `outbound_chain` at construction, so a
/// reference to an outbound chain that cannot be built must fail pipeline
/// resolution at startup rather than surfacing at request time. Pointing the
/// web-search filter at an undefined named chain must fail the build.
#[test]
fn web_search_unresolvable_outbound_chain_fails_build() {
    let path = praxis_test_utils::example_config_path("openai/responses/web-search.yaml");
    let yaml = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let yaml = yaml.replace("${WEB_SEARCH_API_KEY}", "test-key-for-config");
    let yaml = yaml.replace("outbound_chain: web-search-outbound", "outbound_chain: does-not-exist");

    let error = resolve(&yaml).expect_err("an unresolvable outbound chain must fail the build");
    assert!(
        error.contains("unknown chain") && error.contains("does-not-exist"),
        "the build error must name the unresolved outbound chain: {error}"
    );
}

#[test]
fn web_search_example_no_tools_passthrough() {
    let backend_guard = start_backend_with_shutdown("inference");
    let proxy_port = free_port();

    let config = load_web_search_config(proxy_port, &HashMap::from([("127.0.0.1:3001", backend_guard.port())]));
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"Hello"}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "no-tools request should return 200");
    assert_eq!(
        parse_body(&raw),
        "inference",
        "request without tools should route to inference backend"
    );
}
