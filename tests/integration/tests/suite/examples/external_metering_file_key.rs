// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for the external-metering file-backed internal auth token
//! example config (issue #1721).
//!
//! Covers two layers: the example config with a `file:`-sourced
//! `internal_auth_token` parses and the pipeline builds (reading the token file
//! at startup), and, end to end, the metering balance callout carries the
//! `Authorization: Bearer <token>` header read from that file.

use std::{collections::HashMap, io::Write as _, sync::Arc};

use praxis_core::config::Config;
use praxis_test_utils::{
    StatefulCapturingBackend, allow_loopback_endpoints, example_config_path, free_port, http_send, parse_status,
    patch_yaml, start_header_echo_backend, start_proxy,
};

/// Token written to the projected secret file and expected on the wire.
const METERING_TOKEN: &str = "metering-secret-xyz";

/// Read the example config and point its `internal_auth_token.file` at `path`.
fn example_yaml_with_token_file(path: &str) -> String {
    let yaml = std::fs::read_to_string(example_config_path("external-metering-file-key.yaml")).unwrap();
    yaml.replace("/etc/praxis/metering-token", path)
}

/// Resolve the full AI pipeline from raw YAML, surfacing any build error.
fn resolve(yaml: &str) -> Result<(), String> {
    let config = Config::from_yaml(yaml).map_err(|error| error.to_string())?;
    let health = Arc::new(HashMap::new());
    let kv_stores = praxis_core::kv::KvStoreRegistry::new();
    let subrequest_client =
        praxis_core::subrequest::SubRequestClient::new(praxis_core::subrequest::SubRequestConnector::new(8, None));
    let registry = praxis_ai::build_full_registry(&subrequest_client);
    let inflight = praxis_ai::InFlightRegistry::new();
    praxis_ai::resolve_pipelines(&config, &registry, &health, &kv_stores, &inflight, &subrequest_client)
        .map(|_pipelines| ())
        .map_err(|error| error.to_string())
}

#[test]
fn external_metering_file_key_example_builds() {
    let mut token = tempfile::NamedTempFile::new().unwrap();
    write!(token, "{METERING_TOKEN}").unwrap();
    let patched = example_yaml_with_token_file(token.path().to_str().unwrap());
    resolve(&patched).expect("file-key external-metering example must build");
}

#[test]
fn external_metering_file_key_example_fails_build_without_token_file() {
    // The example references a file that does not exist; building must fail
    // loud at startup rather than silently dialing unauthenticated.
    let patched = example_yaml_with_token_file("/nonexistent/praxis-metering-token-xyz");
    assert!(
        resolve(&patched).is_err(),
        "an unreadable token file must fail the pipeline build"
    );
}

#[test]
fn external_metering_file_key_callouts_carry_bearer_token() {
    let mut token = tempfile::NamedTempFile::new().unwrap();
    write!(token, "{METERING_TOKEN}").unwrap();

    let backend = start_header_echo_backend();
    let proxy_port = free_port();

    // Metering mock: the first (non-probe) request is the balance GET, the
    // second is the fire-and-forget usage POST.
    let metering = StatefulCapturingBackend::new(vec![
        (200, r#"{"hasAccess": true, "balance": 9000.0}"#.to_owned()),
        (204, String::new()),
    ])
    .start_with_shutdown();

    let yaml = example_yaml_with_token_file(token.path().to_str().unwrap());
    let port_map = HashMap::from([("127.0.0.1:3000", backend.port()), ("127.0.0.1:9090", metering.port())]);
    let patched = allow_loopback_endpoints(&patch_yaml(&yaml, proxy_port, &port_map));
    let config = Config::from_yaml(&patched).expect("patched file-key metering config must parse");

    let proxy = start_proxy(&config);
    let raw = http_send(
        proxy.addr(),
        "POST /v1/chat/completions HTTP/1.1\r\n\
         Host: localhost\r\n\
         x-tenant-username: alice\r\n\
         x-tenant-group: engineering\r\n\
         Connection: close\r\n\r\n",
    );
    assert_eq!(parse_status(&raw), 200, "balance available should proxy the request");

    // The balance check runs synchronously during on_request, so it is captured
    // by the time the client response returns.
    let requests = metering.requests();
    let balance = requests
        .iter()
        .find(|r| r.uri.contains("/api/v1/customers"))
        .expect("metering must have received the balance check callout");
    assert_eq!(balance.method, "GET");
    assert!(
        balance.headers.to_lowercase().contains("authorization:"),
        "balance callout must carry an Authorization header: {}",
        balance.headers
    );
    assert!(
        balance.headers.contains(&format!("Bearer {METERING_TOKEN}")),
        "balance callout must carry the file-sourced bearer token: {}",
        balance.headers
    );

    // The usage report is fire-and-forget (spawned after response EOS), so poll
    // briefly for the POST to /api/v1/events and assert it carries both the
    // bearer token and the JSON content type.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
    let mut usage = None;
    while std::time::Instant::now() < deadline {
        usage = metering
            .requests()
            .into_iter()
            .find(|r| r.method == "POST" && r.uri.contains("/api/v1/events"));
        if usage.is_some() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(25));
    }
    let usage = usage.expect("metering must receive the usage-report POST");
    assert!(
        usage.headers.contains(&format!("Bearer {METERING_TOKEN}")),
        "usage report must carry the file-sourced bearer token: {}",
        usage.headers
    );
    assert!(
        usage.headers.to_lowercase().contains("content-type: application/json"),
        "usage report must keep its JSON content type alongside the auth header: {}",
        usage.headers
    );
}
