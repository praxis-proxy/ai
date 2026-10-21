// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for the external-metering file-backed internal auth token
//! example config (issue #1721).
//!
//! Covers two layers: the example config with a `file:`-sourced
//! `internal_auth_token` parses and the pipeline builds (reading the token file
//! at startup), and, end to end, the metering balance callout carries the
//! `Authorization: Bearer <token>` header read from that file, observes atomic
//! replacement, and fails closed after removal.

use std::{
    collections::HashMap,
    fs,
    io::Write as _,
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

use praxis_core::config::Config;
use praxis_test_utils::{
    StatefulCapturingBackend, allow_loopback_endpoints, example_config_path, free_port, http_send, parse_status,
    patch_yaml, start_header_echo_backend, start_proxy,
};

/// Token written to the projected secret file and expected on the wire.
const METERING_TOKEN: &str = "metering-secret-xyz";

/// Read the example config and point its `internal_auth_token.file` at `path`.
fn example_yaml_with_token_file(path: &str) -> String {
    let yaml = fs::read_to_string(example_config_path("external-metering-file-key.yaml")).unwrap();
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
    const ROTATED_TOKEN: &str = "metering-secret-rotated";
    let token_dir = tempfile::tempdir().unwrap();
    let token_path = token_dir.path().join("token");
    fs::write(&token_path, METERING_TOKEN).unwrap();

    let backend = start_header_echo_backend();
    let proxy_port = free_port();

    let metering =
        StatefulCapturingBackend::new(vec![(200, r#"{"hasAccess": true, "balance": 9000.0}"#.to_owned()); 32])
            .start_with_shutdown();

    let yaml = example_yaml_with_token_file(token_path.to_str().unwrap());
    let port_map = HashMap::from([("127.0.0.1:3000", backend.port()), ("127.0.0.1:9090", metering.port())]);
    let patched = allow_loopback_endpoints(&patch_yaml(&yaml, proxy_port, &port_map));
    let config = Config::from_yaml(&patched).expect("patched file-key metering config must parse");

    let proxy = start_proxy(&config);
    let raw = send_metered_request(proxy.addr());
    assert_eq!(parse_status(&raw), 200, "balance available should proxy the request");

    let balance = wait_for_metering_requests(&metering, "GET", "/api/v1/customers", 1)
        .into_iter()
        .next()
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
    // for it before rotating the file.
    let usage = wait_for_metering_requests(&metering, "POST", "/api/v1/events", 1)
        .into_iter()
        .next()
        .expect("metering must receive the usage-report POST");
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

    // Replace the file atomically in the same directory, as projected Secrets
    // do. Retry a balance call until the watcher has published the new token.
    let replacement_path = token_dir.path().join("token-next");
    fs::write(&replacement_path, ROTATED_TOKEN).unwrap();
    fs::rename(&replacement_path, &token_path).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    let rotated_balance_headers = loop {
        let raw = send_metered_request(proxy.addr());
        assert_eq!(parse_status(&raw), 200, "fail_open admits while metering succeeds");
        let balances = wait_for_metering_requests(&metering, "GET", "/api/v1/customers", 2);
        let latest = balances.last().expect("at least two balance callouts");
        if latest.headers.contains(&format!("Bearer {ROTATED_TOKEN}")) {
            break latest.headers.clone();
        }
        assert!(Instant::now() < deadline, "metering did not observe the rotated token");
    };
    assert!(
        rotated_balance_headers.contains(&format!("Bearer {ROTATED_TOKEN}")),
        "the next balance callout must use the replacement file token: {rotated_balance_headers}"
    );

    // Let every admitted request's fire-and-forget usage report finish before
    // checking that removal suppresses both callouts.
    let balance_count = wait_for_metering_requests(&metering, "GET", "/api/v1/customers", 2).len();
    let _usage = wait_for_metering_requests(&metering, "POST", "/api/v1/events", balance_count);

    fs::remove_file(&token_path).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let before = metering_callout_counts(&metering);
        let raw = send_metered_request(proxy.addr());
        assert_eq!(
            parse_status(&raw),
            200,
            "fail_open must admit the inference request after the token disappears"
        );
        thread::sleep(Duration::from_millis(50));
        if metering_callout_counts(&metering) == before {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "metering callouts did not stop after token removal"
        );
    }

    let settled_counts = metering_callout_counts(&metering);
    let raw = send_metered_request(proxy.addr());
    assert_eq!(parse_status(&raw), 200, "fail_open must continue admitting requests");
    thread::sleep(Duration::from_millis(50));
    assert_eq!(
        metering_callout_counts(&metering),
        settled_counts,
        "a missing token must suppress balance and usage callouts rather than send them unauthenticated"
    );
}

fn send_metered_request(proxy_addr: &str) -> String {
    http_send(
        proxy_addr,
        "POST /v1/chat/completions HTTP/1.1\r\n\
         Host: localhost\r\n\
         x-tenant-username: alice\r\n\
         x-tenant-group: engineering\r\n\
         Connection: close\r\n\r\n",
    )
}

fn wait_for_metering_requests(
    metering: &praxis_test_utils::StatefulCapturingGuard,
    method: &str,
    path: &str,
    count: usize,
) -> Vec<praxis_test_utils::CapturedRequest> {
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        let matches: Vec<_> = metering
            .requests()
            .into_iter()
            .filter(|request| request.method == method && request.uri.contains(path))
            .collect();
        if matches.len() >= count {
            return matches;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for {method} {path} x{count}"
        );
        thread::sleep(Duration::from_millis(25));
    }
}

fn metering_callout_counts(metering: &praxis_test_utils::StatefulCapturingGuard) -> (usize, usize) {
    let requests = metering.requests();
    let balances = requests
        .iter()
        .filter(|request| request.method == "GET" && request.uri.contains("/api/v1/customers"))
        .count();
    let usage_reports = requests
        .iter()
        .filter(|request| request.method == "POST" && request.uri.contains("/api/v1/events"))
        .count();
    (balances, usage_reports)
}
