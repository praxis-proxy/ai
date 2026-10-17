// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Opt-in business smoke coverage for the repository Sentinel fixture.

use std::{
    collections::HashMap,
    env,
    io::Write as _,
    process::{Command, Stdio},
    sync::atomic::{AtomicU64, Ordering},
    thread,
    time::Duration,
};

use praxis_test_utils::{
    StatefulCapturingBackend, example_config_path, free_port, http_send, json_post, parse_status, patch_yaml,
    start_proxy,
};

const DEFAULT_RULE: &str = "            algorithm: sliding_window\n            window: 1h\n            capacity: 100000\n            reserved_tokens: 500";
const SLIDING_QUOTA_RULE: &str = "            algorithm: sliding_window\n            window: 1h\n            capacity: 500\n            reserved_tokens: 500";
const TOKEN_BUCKET_QUOTA_RULE: &str = "            algorithm: token_bucket\n            capacity: 500\n            refill_rate: 1\n            reserved_tokens: 500";
const OPENAI_RESPONSE_BODY: &str =
    r#"{"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":5,"total_tokens":10}}"#;

fn fixture_config(proxy_port: u16, backend_port: u16) -> praxis_core::config::Config {
    fixture_config_with_rule(proxy_port, backend_port, DEFAULT_RULE, "admission")
}

fn fixture_config_with_rule(
    proxy_port: u16,
    backend_port: u16,
    rule: &str,
    label: &str,
) -> praxis_core::config::Config {
    static NAMESPACE_COUNTER: AtomicU64 = AtomicU64::new(0);
    let endpoint_env = env::var("TOKEN_RATE_LIMIT_SENTINEL_ENDPOINTS")
        .expect("fixture must export TOKEN_RATE_LIMIT_SENTINEL_ENDPOINTS");
    assert_eq!(
        env::var("TOKEN_RATE_LIMIT_SENTINEL_REPLICATION_OFFSET_CONFIRMED").as_deref(),
        Ok("1"),
        "the fixture must acknowledge primary/replica offset convergence before tests"
    );
    assert_eq!(
        env::var("TOKEN_RATE_LIMIT_SENTINEL_PRIMARY_REPLICATION_OFFSET").ok(),
        env::var("TOKEN_RATE_LIMIT_SENTINEL_REPLICA_REPLICATION_OFFSET").ok(),
        "the fixture must export matching primary and replica replication offsets"
    );
    let endpoints = endpoint_env.split(',').map(str::trim).collect::<Vec<_>>();
    assert!(endpoints.len() >= 3, "fixture must expose at least three Sentinels");
    let product = env::var("TOKEN_RATE_LIMIT_SENTINEL_PRODUCT").unwrap_or_else(|_| "valkey".to_owned());
    let path = example_config_path("token-rate-limit-sentinel.yaml");
    let mut yaml = std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {path}: {error}"));
    yaml = yaml.replace(
        "kind: valkey # use `redis` for the Redis qualification lane",
        &format!("kind: {product}"),
    );
    yaml = yaml.replace(DEFAULT_RULE, rule);
    let namespace = format!(
        "{}          namespace: \"praxis:test:sentinel:{}:{label}:{}\"\n",
        env::var("TOKEN_RATE_LIMIT_SENTINEL_ON_FAILURE")
            .ok()
            .filter(|value| !value.is_empty())
            .map_or_else(String::new, |value| format!("          on_failure: {value}\n")),
        std::process::id(),
        NAMESPACE_COUNTER.fetch_add(1, Ordering::Relaxed),
    );
    yaml = yaml.replace("          sentinel:\n", &format!("{namespace}          sentinel:\n"));
    if env::var("TOKEN_RATE_LIMIT_SENTINEL_DATA_PASSWORD").is_ok_and(|value| !value.is_empty()) {
        yaml = yaml.replace(
            "            # redis_password: \"${TOKEN_RATE_LIMIT_REDIS_PASSWORD}\"",
            "            redis_password: \"${TOKEN_RATE_LIMIT_SENTINEL_DATA_PASSWORD}\"",
        );
    }
    if env::var("TOKEN_RATE_LIMIT_SENTINEL_PASSWORD").is_ok_and(|value| !value.is_empty()) {
        yaml = yaml.replace(
            "            # sentinel_password: \"${TOKEN_RATE_LIMIT_SENTINEL_PASSWORD}\"",
            "            sentinel_password: \"${TOKEN_RATE_LIMIT_SENTINEL_PASSWORD}\"",
        );
    }
    let endpoint_lines = endpoints
        .iter()
        .map(|endpoint| format!("              - \"{endpoint}\""))
        .collect::<Vec<_>>()
        .join("\n");
    let old_endpoints = "            endpoints:\n              - \"redis://127.0.0.1:26379\"\n              - \"redis://127.0.0.1:26380\"\n              - \"redis://127.0.0.1:26381\"";
    yaml = yaml.replace(old_endpoints, &format!("            endpoints:\n{endpoint_lines}"));
    let patched = patch_yaml(&yaml, proxy_port, &HashMap::from([("127.0.0.1:9000", backend_port)]));
    praxis_core::config::Config::from_yaml(&patched).expect("Sentinel fixture config should parse")
}

/// Run the fixture's native Sentinel CLI without putting credentials in the
/// container-engine argument list.
fn sentinel_cli(args: &[&str]) -> std::process::Output {
    let engine = env::var("TOKEN_RATE_LIMIT_SENTINEL_ENGINE").expect("fixture must export its container engine");
    let container =
        env::var("TOKEN_RATE_LIMIT_SENTINEL_SENTINEL_CONTAINER").expect("fixture must export its Sentinel container");
    let product = env::var("TOKEN_RATE_LIMIT_SENTINEL_PRODUCT").unwrap_or_else(|_| "valkey".to_owned());
    let cli = if product == "redis" { "redis-cli" } else { "valkey-cli" };
    let password = env::var("TOKEN_RATE_LIMIT_SENTINEL_PASSWORD").ok();

    let mut command = Command::new(engine);
    if let Some(password) = password {
        let mut child = command
            .args([
                "exec",
                "-i",
                &container,
                "sh",
                "-c",
                "IFS= read -r REDISCLI_AUTH; export REDISCLI_AUTH; exec \"$@\"",
                "sh",
                cli,
            ])
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("start Sentinel CLI with authentication");
        child
            .stdin
            .take()
            .expect("Sentinel CLI stdin")
            .write_all(format!("{password}\n").as_bytes())
            .expect("write Sentinel CLI authentication");
        return child.wait_with_output().expect("wait for Sentinel CLI");
    }

    command
        .args(["exec", &container, cli])
        .args(args)
        .output()
        .expect("run Sentinel CLI")
}

/// Trigger and observe Sentinel's planned promotion after the proxy has made
/// its first request, so the next request must recover an existing connection.
fn failover_warm_proxy_connection() {
    let sentinel_port = env::var("TOKEN_RATE_LIMIT_SENTINEL_SENTINEL_PORT").expect("Sentinel port");
    let replica_port = env::var("TOKEN_RATE_LIMIT_SENTINEL_REPLICA_PORT").expect("replica port");
    let requested = sentinel_cli(&["-p", &sentinel_port, "SENTINEL", "failover", "mymaster"]);
    assert!(
        requested.status.success(),
        "planned Sentinel failover failed: {}",
        String::from_utf8_lossy(&requested.stderr)
    );

    for _ in 0..30 {
        let observed = sentinel_cli(&["-p", &sentinel_port, "SENTINEL", "get-master-addr-by-name", "mymaster"]);
        if String::from_utf8_lossy(&observed.stdout).contains(&format!("127.0.0.1\n{replica_port}")) {
            return;
        }
        thread::sleep(Duration::from_secs(1));
    }
    panic!("planned Sentinel failover did not promote replica port {replica_port}");
}

#[test]
fn sentinel_fixture_admits_through_the_real_proxy_pipeline() {
    if env::var("TOKEN_RATE_LIMIT_SENTINEL_ENDPOINTS").is_err() {
        eprintln!("skipping: TOKEN_RATE_LIMIT_SENTINEL_ENDPOINTS not set");
        return;
    }
    let backend = StatefulCapturingBackend::new(vec![(200, OPENAI_RESPONSE_BODY.to_owned())]).start_with_shutdown();
    let config = fixture_config(free_port(), backend.port());
    let proxy = start_proxy(&config);
    let raw = http_send(proxy.addr(), &json_post("/v1/chat/completions", "{}"));
    assert_eq!(
        parse_status(&raw),
        200,
        "Sentinel fixture should admit through the real proxy: {raw}; upstream_requests={}",
        backend.requests().len()
    );
    assert_eq!(backend.requests().len(), 1, "admitted request invokes upstream once");
}

#[test]
fn sentinel_fixture_enforces_sliding_window_quota_through_the_real_pipeline() {
    if env::var("TOKEN_RATE_LIMIT_SENTINEL_ENDPOINTS").is_err() {
        eprintln!("skipping: TOKEN_RATE_LIMIT_SENTINEL_ENDPOINTS not set");
        return;
    }
    let backend = StatefulCapturingBackend::new(vec![
        (200, OPENAI_RESPONSE_BODY.to_owned()),
        (200, OPENAI_RESPONSE_BODY.to_owned()),
    ])
    .start_with_shutdown();
    let config = fixture_config_with_rule(free_port(), backend.port(), SLIDING_QUOTA_RULE, "sliding-quota");
    let proxy = start_proxy(&config);

    let first = http_send(proxy.addr(), &json_post("/v1/chat/completions", "{}"));
    assert_eq!(
        parse_status(&first),
        200,
        "the first sliding-window request should be admitted: {first}; upstream_requests={}",
        backend.requests().len()
    );
    assert_eq!(
        parse_status(&http_send(proxy.addr(), &json_post("/v1/chat/completions", "{}"))),
        429,
        "the replicated sliding-window quota should deny the second request"
    );
    assert_eq!(
        backend.requests().len(),
        1,
        "the denied request must not invoke upstream"
    );
}

#[test]
fn sentinel_fixture_enforces_token_bucket_quota_through_the_real_pipeline() {
    if env::var("TOKEN_RATE_LIMIT_SENTINEL_ENDPOINTS").is_err() {
        eprintln!("skipping: TOKEN_RATE_LIMIT_SENTINEL_ENDPOINTS not set");
        return;
    }
    let backend = StatefulCapturingBackend::new(vec![(200, OPENAI_RESPONSE_BODY.to_owned())]).start_with_shutdown();
    let config = fixture_config_with_rule(
        free_port(),
        backend.port(),
        TOKEN_BUCKET_QUOTA_RULE,
        "token-bucket-quota",
    );
    let proxy = start_proxy(&config);

    let first = http_send(proxy.addr(), &json_post("/v1/chat/completions", "{}"));
    assert_eq!(
        parse_status(&first),
        200,
        "the first token-bucket request should be admitted: {first}; upstream_requests={}",
        backend.requests().len()
    );
    assert_eq!(
        parse_status(&http_send(proxy.addr(), &json_post("/v1/chat/completions", "{}"))),
        429,
        "the replicated token-bucket quota should deny the second request"
    );
    assert_eq!(
        backend.requests().len(),
        1,
        "the denied request must not invoke upstream"
    );
}

#[test]
fn sentinel_fixture_warm_connection_reconnects_after_planned_failover() {
    if env::var("TOKEN_RATE_LIMIT_SENTINEL_FAILOVER_BEFORE_COMMAND").as_deref() != Ok("1") {
        eprintln!("skipping: planned failover lane is not enabled");
        return;
    }
    let backend = StatefulCapturingBackend::new(vec![
        (200, OPENAI_RESPONSE_BODY.to_owned()),
        (200, OPENAI_RESPONSE_BODY.to_owned()),
    ])
    .start_with_shutdown();
    let config = fixture_config(free_port(), backend.port());
    let proxy = start_proxy(&config);

    let first = http_send(proxy.addr(), &json_post("/v1/chat/completions", "{}"));
    assert_eq!(
        parse_status(&first),
        200,
        "the pre-failover request should warm the proxy connection: {first}"
    );
    failover_warm_proxy_connection();

    let second = http_send(proxy.addr(), &json_post("/v1/chat/completions", "{}"));
    assert_eq!(
        parse_status(&second),
        200,
        "the proxy must recover its existing Sentinel connection after failover: {second}"
    );
    assert_eq!(backend.requests().len(), 2, "both requests must reach the upstream");
}

#[test]
fn sentinel_health_noreplicas_failure() {
    if env::var("TOKEN_RATE_LIMIT_SENTINEL_ENDPOINTS").is_err() {
        eprintln!("skipping: TOKEN_RATE_LIMIT_SENTINEL_ENDPOINTS not set");
        return;
    }
    let on_failure = env::var("TOKEN_RATE_LIMIT_SENTINEL_ON_FAILURE").expect("fixture must set on_failure");
    assert!(matches!(on_failure.as_str(), "closed" | "open"));
    let backend = StatefulCapturingBackend::new(vec![(200, OPENAI_RESPONSE_BODY.to_owned())]).start_with_shutdown();
    let config = fixture_config_with_rule(free_port(), backend.port(), SLIDING_QUOTA_RULE, "noreplicas");
    let proxy = start_proxy(&config);
    let raw = http_send(proxy.addr(), &json_post("/v1/chat/completions", "{}"));
    let status = parse_status(&raw);
    if on_failure == "closed" {
        assert_eq!(status, 503, "closed NOREPLICAS admission must fail closed: {raw}");
        assert!(
            backend.requests().is_empty(),
            "closed NOREPLICAS must not invoke upstream"
        );
    } else {
        assert_eq!(status, 200, "open NOREPLICAS admission must bypass once: {raw}");
        assert_eq!(backend.requests().len(), 1, "open NOREPLICAS must invoke upstream once");
    }
}
