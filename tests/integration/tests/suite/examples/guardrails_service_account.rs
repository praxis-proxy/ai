// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Integration tests for `nemo-guardrails-service-account.yaml`.
//!
//! The base NeMo examples call the service with no credentials. This
//! config sets `token_file` on the NeMo provider. The token file in the
//! example is a pod path, so each test that starts the proxy points that
//! field at a temp file first.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use praxis_test_utils::{
    StatefulCapturingBackend, StatefulCapturingGuard, free_port, http_post, http_send, json_post,
    start_backend_with_shutdown, start_proxy,
};
use tempfile::TempDir;

use super::load_example_config;

const EXAMPLE: &str = "nemo-guardrails-service-account.yaml";

#[test]
fn service_account_example_parses() {
    let config = load_example_config(
        EXAMPLE,
        free_port(),
        HashMap::from([("127.0.0.1:3000", 29990_u16), ("127.0.0.1:3001", 29991_u16)]),
    );
    assert_eq!(config.listeners.len(), 1);
    let mut config = config;
    let token_file = provider_config(&mut config)
        .get(serde_yaml::Value::from("token_file"))
        .and_then(serde_yaml::Value::as_str)
        .expect("the NeMo provider should name a token file");
    assert!(
        token_file.contains("/var/run/secrets/tokens/nemo/token"),
        "the example should point at the mounted pod token, got {token_file}"
    );
    let outbound = config
        .filter_chains
        .iter()
        .find(|chain| chain.name == "nemo-outbound")
        .expect("example should define the nemo-outbound chain");
    assert!(
        outbound
            .filters
            .iter()
            .all(|entry| entry.filter_type != "service_account_token"),
        "the login token is attached by the NeMo provider"
    );
}

#[test]
fn service_account_token_replaces_caller_authorization() {
    let backend = start_backend_with_shutdown("ok");
    let nemo = StatefulCapturingBackend::new(vec![(200, r#"{"status":"passed","content":"safe"}"#.to_owned())])
        .start_with_shutdown();
    let proxy_port = free_port();
    let (_token_dir, config) = load_with_token(
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend.port()), ("127.0.0.1:3001", nemo.port())]),
        "sa-jwt\n",
    );
    let proxy = start_proxy(&config);
    let raw = chat_with_caller_secret(proxy.addr());
    assert_eq!(praxis_test_utils::parse_status(&raw), 200, "{raw}");

    let requests = nemo.requests();
    assert_eq!(requests.len(), 1, "NeMo should receive exactly one callout");
    assert!(
        bearer_token(&requests[0].headers, "sa-jwt"),
        "outbound chain should send the service account token: {}",
        requests[0].headers
    );
    assert!(
        !requests[0].headers.contains("client-secret"),
        "the caller's Authorization header must not reach NeMo"
    );
}

/// A filter on the outbound chain that replaces `Authorization` still loses.
/// The pod token is written after that chain runs.
#[test]
fn outbound_chain_cannot_replace_service_account_token() {
    let backend = start_backend_with_shutdown("ok");
    let nemo = StatefulCapturingBackend::new(vec![(200, r#"{"status":"passed","content":"safe"}"#.to_owned())])
        .start_with_shutdown();
    let proxy_port = free_port();
    let (_token_dir, mut config) = load_with_token(
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend.port()), ("127.0.0.1:3001", nemo.port())]),
        "sa-jwt\n",
    );
    replace_authorization_on_outbound(&mut config);
    let proxy = start_proxy(&config);
    let raw = chat_with_caller_secret(proxy.addr());
    assert_eq!(praxis_test_utils::parse_status(&raw), 200, "{raw}");

    let requests = nemo.requests();
    assert_eq!(requests.len(), 1, "NeMo should receive exactly one callout");
    assert!(
        bearer_token(&requests[0].headers, "sa-jwt"),
        "the pod token must survive an outbound filter that replaces Authorization: {}",
        requests[0].headers
    );
    assert!(
        !requests[0].headers.contains("stolen"),
        "the outbound chain must not be able to choose the Authorization value: {}",
        requests[0].headers
    );
}

/// Replacing the token path itself is observed without a config reload.
/// A projected volume does not rewrite that path; see
/// `service_account_token_data_symlink_swap_is_observed`.
/// The request immediately after the rename can still carry the previous token.
#[test]
fn service_account_token_rotation_is_observed_without_reload() {
    let (token_dir, token_path) = write_token_file("token-a\n");
    let backend = start_backend_with_shutdown("ok");
    let passed = r#"{"status":"passed","content":"ok"}"#.to_owned();
    let nemo = StatefulCapturingBackend::new(vec![(200, passed); 128]).start_with_shutdown();
    let proxy_port = free_port();
    let mut config = load_example_config(
        EXAMPLE,
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend.port()), ("127.0.0.1:3001", nemo.port())]),
    );
    set_token_file(&mut config, &token_path);
    let proxy = start_proxy(&config);
    let _token_dir = token_dir;

    let (status, _) = http_post(
        proxy.addr(),
        "/v1/checks",
        r#"{"model":"test","messages":[{"role":"user","content":"first"}]}"#,
    );
    assert_eq!(status, 200);

    let replacement = token_path.with_extension("next");
    std::fs::write(&replacement, "token-b\n").unwrap();
    std::fs::rename(&replacement, &token_path).unwrap();
    poll_for_bearer(
        proxy.addr(),
        &nemo,
        "token-b",
        "background watcher did not publish token-b within 2s",
    );

    let requests = nemo.requests();
    let rotated = requests
        .iter()
        .position(|request| bearer_token(&request.headers, "token-b"))
        .expect("a later callout must observe the rotated token without a config reload");
    assert!(rotated >= 1, "the callout from before the rename must still be present");
    assert!(
        requests[..rotated]
            .iter()
            .all(|request| bearer_token(&request.headers, "token-a")),
        "callouts before the watcher publishes token-b must keep the previous token"
    );
}

/// Kubelet rotates a projected token by swapping `..data`. The `token`
/// symlink stays pointed at `..data/token`, so only a parent-directory
/// match reloads it. Same layout as `watcher_detects_symlink_swap`.
#[cfg(unix)]
#[test]
fn service_account_token_data_symlink_swap_is_observed() {
    let volume = projected_token_volume("token-a\n");
    let backend = start_backend_with_shutdown("ok");
    let passed = r#"{"status":"passed","content":"ok"}"#.to_owned();
    let nemo = StatefulCapturingBackend::new(vec![(200, passed); 128]).start_with_shutdown();
    let proxy_port = free_port();
    let mut config = load_example_config(
        EXAMPLE,
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend.port()), ("127.0.0.1:3001", nemo.port())]),
    );
    set_token_file(&mut config, &volume.token_path);
    let proxy = start_proxy(&config);

    let (status, _) = http_post(
        proxy.addr(),
        "/v1/checks",
        r#"{"model":"test","messages":[{"role":"user","content":"first"}]}"#,
    );
    assert_eq!(status, 200);

    swap_data_symlink(&volume, "token-b\n");
    assert_eq!(
        std::fs::read_link(&volume.token_path).unwrap(),
        Path::new("..data/token"),
        "kubelet leaves the token symlink in place"
    );
    poll_for_bearer(
        proxy.addr(),
        &nemo,
        "token-b",
        "background watcher did not publish token-b within 2s of the ..data swap",
    );
}

#[cfg(unix)]
struct ProjectedToken {
    _dir: TempDir,
    token_path: PathBuf,
    data_link: PathBuf,
}

/// A projected ServiceAccount volume: `token` -> `..data/token`, and
/// `..data` -> the directory that holds the current token bytes.
#[cfg(unix)]
fn projected_token_volume(contents: &str) -> ProjectedToken {
    let dir = tempfile::tempdir().expect("create projected token volume");
    let data_v1 = dir.path().join("data_v1");
    std::fs::create_dir(&data_v1).unwrap();
    std::fs::write(data_v1.join("token"), contents).unwrap();

    let data_link = dir.path().join("..data");
    std::os::unix::fs::symlink(&data_v1, &data_link).unwrap();
    let token_path = dir.path().join("token");
    std::os::unix::fs::symlink("..data/token", &token_path).unwrap();
    ProjectedToken {
        _dir: dir,
        token_path,
        data_link,
    }
}

/// AtomicWriter swap: point `..data` at a new directory without rewriting `token`.
#[cfg(unix)]
fn swap_data_symlink(volume: &ProjectedToken, contents: &str) {
    let data_v2 = volume
        .data_link
        .parent()
        .expect("..data lives in the volume directory")
        .join("data_v2");
    std::fs::create_dir(&data_v2).unwrap();
    std::fs::write(data_v2.join("token"), contents).unwrap();
    let tmp_link = volume.data_link.with_file_name("..data_tmp");
    std::os::unix::fs::symlink(&data_v2, &tmp_link).unwrap();
    std::fs::rename(&tmp_link, &volume.data_link).unwrap();
}

fn chat_with_caller_secret(addr: &str) -> String {
    let mut request = json_post(
        "/v1/chat/completions",
        r#"{"model":"test","messages":[{"role":"user","content":"Hello"}]}"#,
    );
    request = request.replace(
        "Connection: close",
        "Authorization: Bearer client-secret\r\nConnection: close",
    );
    http_send(addr, &request)
}

fn poll_for_bearer(addr: &str, nemo: &StatefulCapturingGuard, token: &str, timeout_msg: &str) {
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        let (status, body) = http_post(
            addr,
            "/v1/checks",
            r#"{"model":"test","messages":[{"role":"user","content":"after-rotation"}]}"#,
        );
        assert_eq!(
            status, 200,
            "callout while waiting for the watcher should still pass; body: {body}"
        );
        if nemo
            .requests()
            .iter()
            .any(|request| bearer_token(&request.headers, token))
        {
            break;
        }
        assert!(Instant::now() < deadline, "{timeout_msg}");
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn bearer_token(headers: &str, token: &str) -> bool {
    let expected = format!("Authorization: Bearer {token}");
    headers.lines().any(|line| line.eq_ignore_ascii_case(&expected))
}

fn write_token_file(contents: &str) -> (TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("create tempdir for service account token");
    let path = dir.path().join("token");
    std::fs::write(&path, contents).expect("write service account token file");
    (dir, path)
}

fn provider_config(config: &mut praxis_core::config::Config) -> &mut serde_yaml::Mapping {
    let chain = config
        .filter_chains
        .iter_mut()
        .find(|chain| chain.name == "nemo-guardrails")
        .expect("example should define the main guardrails chain");
    let filter = chain
        .filters
        .iter_mut()
        .find(|entry| entry.filter_type == "ai_guardrails")
        .expect("example should define ai_guardrails");
    filter
        .config
        .as_mapping_mut()
        .expect("guardrails config should be a mapping")
        .get_mut(serde_yaml::Value::from("provider"))
        .and_then(serde_yaml::Value::as_mapping_mut)
        .expect("NeMo provider config should be a mapping")
}

fn replace_authorization_on_outbound(config: &mut praxis_core::config::Config) {
    let outbound = config
        .filter_chains
        .iter_mut()
        .find(|chain| chain.name == "nemo-outbound")
        .expect("example should define the nemo-outbound chain");
    let strip: praxis_core::config::FilterEntry = serde_yaml::from_str(
        r#"
filter: headers
request_set:
  - name: authorization
    value: Bearer stolen
"#,
    )
    .expect("headers filter should parse");
    outbound.filters.insert(0, strip);
}

fn set_token_file(config: &mut praxis_core::config::Config, path: &Path) {
    provider_config(config).insert(
        serde_yaml::Value::from("token_file"),
        serde_yaml::Value::from(path.to_str().expect("tempdir path must be valid UTF-8")),
    );
}

fn load_with_token(
    listener_port: u16,
    port_map: HashMap<&str, u16>,
    contents: &str,
) -> (TempDir, praxis_core::config::Config) {
    let (token_dir, token_path) = write_token_file(contents);
    let mut config = load_example_config(EXAMPLE, listener_port, port_map);
    set_token_file(&mut config, &token_path);
    (token_dir, config)
}
