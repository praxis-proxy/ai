// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Tests for the projected-credential-mount example configuration.
//!
//! The `intelligent-route-projected-credentials.yaml` example starts
//! `credential_inject` with an empty startup table and resolves the selected
//! credential on demand from a projected Kubernetes Secret mount. These tests
//! drive the real proxy end-to-end to prove the documented contract:
//!
//!   1. a credential mounted *after* startup is injected without reload, and
//!   2. an absent projected credential fails closed (503) without an upstream call — a selected credential-bearing
//!      route is never forwarded raw.

use std::{collections::HashMap, sync::Arc};

use praxis_core::config::Config;
use praxis_test_utils::{
    example_config_path, free_port, http_post, start_header_echo_backend, start_proxy, start_stateful_backend,
};

const EXAMPLE: &str = "intelligent-route-projected-credentials.yaml";

/// Mount base as written in the example config.
const MOUNT_BASE: &str = "/run/secrets/projected-credentials";

/// Granite cluster endpoint as written in the example config.
const GRANITE_ENDPOINT: &str = "127.0.0.1:8001";

/// Body selecting the granite candidate, whose credential reference is absent
/// from the empty startup table and resolved from the projected mount.
const GRANITE_BODY: &str = r#"{"model":"granite-3.3-8b","messages":[]}"#;

/// Read the example YAML and repoint the projected mount base, the listener,
/// the admin endpoint, and the granite cluster endpoint at test-local targets.
fn example_yaml(mount_base: &std::path::Path, listener_port: u16, backend_port: u16) -> String {
    let path = example_config_path(EXAMPLE);
    let yaml = std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {path}: {error}"));
    yaml.replace(
        MOUNT_BASE,
        mount_base.to_str().expect("temporary mount base must be UTF-8"),
    )
    .replace("0.0.0.0:8080", &format!("127.0.0.1:{listener_port}"))
    // Give the admin endpoint its own free port so parallel example tests that
    // reuse the example's fixed admin port do not contend for it.
    .replace("127.0.0.1:9901", &format!("127.0.0.1:{}", free_port()))
    .replace(GRANITE_ENDPOINT, &format!("127.0.0.1:{backend_port}"))
}

/// Resolve the full AI pipeline so filter construction with an empty startup
/// table plus `projected_credential_mount_base` is validated, not just parsed.
fn resolve(yaml: &str) -> Result<(), String> {
    let config = Config::from_yaml(yaml).map_err(|error| error.to_string())?;
    let health = Arc::new(HashMap::new());
    let kv_stores = praxis_core::kv::KvStoreRegistry::new();
    let subrequest_client = praxis_test_utils::test_subrequest_client();
    let registry = praxis_ai::build_full_registry(&subrequest_client);
    praxis_ai::resolve_pipelines(&config, &registry, &health, &kv_stores, &subrequest_client)
        .map(|_pipelines| ())
        .map_err(|error| error.to_string())
}

/// Write `token` at the projected path `{base}/{namespace}/{name}/{key}` that
/// the filter derives from the overlay credential reference.
fn mount_secret(base: &std::path::Path, namespace: &str, name: &str, key: &str, token: &str) {
    let dir = base.join(namespace).join(name);
    std::fs::create_dir_all(&dir).expect("create projected secret directory");
    std::fs::write(dir.join(key), token).expect("write projected secret token");
}

#[test]
fn projected_credentials_example_resolves_with_empty_startup_table() {
    let dir = tempfile::tempdir().expect("create mount base");
    let yaml = example_yaml(dir.path(), free_port(), free_port());
    resolve(&yaml).expect("empty startup table plus projected mount must satisfy the AI pipeline contract");
}

#[test]
fn projected_credential_mounted_after_startup_is_injected() {
    let backend = start_header_echo_backend();
    let proxy_port = free_port();
    let dir = tempfile::tempdir().expect("create mount base");

    // Start with the projected mount present but empty: no credential exists in
    // the startup table and none is mounted yet.
    let yaml = example_yaml(dir.path(), proxy_port, backend.port());
    let config = Config::from_yaml(&yaml).expect("example must parse");
    let proxy = start_proxy(&config);

    // A provider Secret is projected into the mount *after* the proxy is live.
    mount_secret(
        dir.path(),
        "grid-demo",
        "granite-provider",
        "token",
        "projected-granite-token",
    );

    let (status, body) = http_post(proxy.addr(), "/v1/chat/completions", GRANITE_BODY);
    assert_eq!(status, 200, "routed request must succeed: {body}");
    assert!(
        body.contains("authorization: Bearer projected-granite-token"),
        "credential mounted after startup must be injected from the projected mount: {body}"
    );
}

#[test]
fn absent_projected_credential_fails_closed_without_upstream_call() {
    // Capturing backend so we can assert the upstream was never contacted.
    let backend = start_stateful_backend(vec![(200, "granite-backend".to_owned())]);
    let proxy_port = free_port();
    let dir = tempfile::tempdir().expect("create mount base");

    // The mount base exists but the selected credential is never projected.
    let yaml = example_yaml(dir.path(), proxy_port, backend.port());
    let config = Config::from_yaml(&yaml).expect("example must parse");
    let proxy = start_proxy(&config);

    let (status, body) = http_post(proxy.addr(), "/v1/chat/completions", GRANITE_BODY);
    assert_eq!(
        status, 503,
        "a selected credential absent from the projected mount must fail closed: {body}"
    );
    assert!(
        backend.requests().is_empty(),
        "upstream must not be contacted when the credential cannot be injected: {:?}",
        backend.requests()
    );
}
