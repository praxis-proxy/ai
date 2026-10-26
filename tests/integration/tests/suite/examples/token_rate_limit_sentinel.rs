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
const QUOTA_RESPONSE_BODY: &str =
    r#"{"choices":[],"usage":{"prompt_tokens":250,"completion_tokens":250,"total_tokens":500}}"#;

fn fixture_config(proxy_port: u16, backend_port: u16) -> praxis_core::config::Config {
    fixture_config_with_rule(proxy_port, backend_port, DEFAULT_RULE, "admission")
}

fn replace_required(yaml: &str, from: &str, to: &str) -> String {
    assert!(
        yaml.contains(from),
        "token-rate-limit-sentinel.yaml no longer contains the fragment the fixture patches: {from:?}"
    );
    yaml.replace(from, to)
}

fn assert_replication_caught_up(primary: u64, replica: u64) {
    assert!(
        replica >= primary,
        "replica must acknowledge the observed primary offset: primary={primary}, replica={replica}"
    );
}

fn fixture_offset(name: &str) -> u64 {
    env::var(name)
        .unwrap_or_else(|error| panic!("fixture must export {name}: {error}"))
        .parse()
        .unwrap_or_else(|error| panic!("fixture offset {name} must be an unsigned integer: {error}"))
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
    assert_replication_caught_up(
        fixture_offset("TOKEN_RATE_LIMIT_SENTINEL_PRIMARY_REPLICATION_OFFSET"),
        fixture_offset("TOKEN_RATE_LIMIT_SENTINEL_REPLICA_REPLICATION_OFFSET"),
    );
    let endpoints = endpoint_env.split(',').map(str::trim).collect::<Vec<_>>();
    assert!(endpoints.len() >= 3, "fixture must expose at least three Sentinels");
    let product = env::var("TOKEN_RATE_LIMIT_SENTINEL_PRODUCT").unwrap_or_else(|_| "valkey".to_owned());
    let path = example_config_path("token-rate-limit-sentinel.yaml");
    let mut yaml = std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {path}: {error}"));
    yaml = replace_required(
        &yaml,
        "kind: valkey # use `redis` for the Redis qualification lane",
        &format!("kind: {product}"),
    );
    yaml = replace_required(&yaml, DEFAULT_RULE, rule);
    let namespace = format!(
        "{}          namespace: \"praxis:test:sentinel:{}:{label}:{}\"\n",
        env::var("TOKEN_RATE_LIMIT_SENTINEL_ON_FAILURE")
            .ok()
            .filter(|value| !value.is_empty())
            .map_or_else(String::new, |value| format!("          on_failure: {value}\n")),
        std::process::id(),
        NAMESPACE_COUNTER.fetch_add(1, Ordering::Relaxed),
    );
    yaml = replace_required(
        &yaml,
        "          sentinel:\n",
        &format!("{namespace}          sentinel:\n"),
    );
    if env::var("TOKEN_RATE_LIMIT_SENTINEL_DATA_PASSWORD").is_ok_and(|value| !value.is_empty()) {
        yaml = replace_required(
            &yaml,
            "            # redis_password: \"${TOKEN_RATE_LIMIT_REDIS_PASSWORD}\"",
            "            redis_password: \"${TOKEN_RATE_LIMIT_SENTINEL_DATA_PASSWORD}\"",
        );
    }
    if env::var("TOKEN_RATE_LIMIT_SENTINEL_PASSWORD").is_ok_and(|value| !value.is_empty()) {
        yaml = replace_required(
            &yaml,
            "            # sentinel_password: \"${TOKEN_RATE_LIMIT_SENTINEL_PASSWORD}\"",
            "            sentinel_password: \"${TOKEN_RATE_LIMIT_SENTINEL_PASSWORD}\"",
        );
    }
    let endpoint_lines = endpoints
        .iter()
        .map(|endpoint| format!("              - \"{endpoint}\""))
        .collect::<Vec<_>>()
        .join("\n");
    let old_endpoints = "            endpoints:\n              - \"redis://127.0.0.1:36379\"\n              - \"redis://127.0.0.1:36380\"\n              - \"redis://127.0.0.1:36381\"";
    yaml = replace_required(
        &yaml,
        old_endpoints,
        &format!("            endpoints:\n{endpoint_lines}"),
    );
    let patched = patch_yaml(&yaml, proxy_port, &HashMap::from([("127.0.0.1:9000", backend_port)]));
    praxis_core::config::Config::from_yaml(&patched).expect("Sentinel fixture config should parse")
}

/// Run the fixture's native Sentinel CLI without putting credentials in the
/// container-engine argument list.
fn sentinel_cli(args: &[&str]) -> std::process::Output {
    fixture_cli(
        "TOKEN_RATE_LIMIT_SENTINEL_SENTINEL_CONTAINER",
        "TOKEN_RATE_LIMIT_SENTINEL_PASSWORD",
        args,
    )
}

fn fixture_cli(container_variable: &str, password_variable: &str, args: &[&str]) -> std::process::Output {
    let engine = env::var("TOKEN_RATE_LIMIT_SENTINEL_ENGINE").expect("fixture must export its container engine");
    let container = env::var(container_variable).expect("fixture must export its container");
    let product = env::var("TOKEN_RATE_LIMIT_SENTINEL_PRODUCT").unwrap_or_else(|_| "valkey".to_owned());
    let cli = if product == "redis" { "redis-cli" } else { "valkey-cli" };
    let password = env::var(password_variable).ok().filter(|value| !value.is_empty());

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

fn replication_info(container_variable: &str, port: &str) -> String {
    let output = fixture_cli(
        container_variable,
        "TOKEN_RATE_LIMIT_SENTINEL_DATA_PASSWORD",
        &["-p", port, "INFO", "replication"],
    );
    assert!(
        output.status.success(),
        "replication INFO failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("replication INFO must be UTF-8")
}

fn info_field<'a>(info: &'a str, field: &str) -> Option<&'a str> {
    info.lines().find_map(|line| {
        let (name, value) = line.split_once(':')?;
        (name == field).then_some(value.trim())
    })
}

/// Snapshot the primary offset after admission, then wait for that offset on
/// the replica. Startup convergence alone cannot prove quota survived promotion.
fn wait_for_admitted_quota_replication() {
    let primary_port = env::var("TOKEN_RATE_LIMIT_SENTINEL_PRIMARY_PORT").expect("primary port");
    let replica_port = env::var("TOKEN_RATE_LIMIT_SENTINEL_REPLICA_PORT").expect("replica port");
    let primary = replication_info("TOKEN_RATE_LIMIT_SENTINEL_PRIMARY_CONTAINER", &primary_port);
    let target = info_field(&primary, "master_repl_offset")
        .expect("primary must report its replication offset")
        .parse::<u64>()
        .expect("primary replication offset must be an integer");
    for _ in 0..100 {
        let replica = replication_info("TOKEN_RATE_LIMIT_SENTINEL_REPLICA_CONTAINER", &replica_port);
        let offset = info_field(&replica, "slave_repl_offset").and_then(|value| value.parse::<u64>().ok());
        if info_field(&replica, "master_link_status") == Some("up") && offset.is_some_and(|offset| offset >= target) {
            return;
        }
        thread::sleep(Duration::from_millis(100));
    }
    panic!("replica did not acknowledge admitted quota at primary offset {target}");
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
#[ignore = "requires the Sentinel fixture topology"]
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
#[ignore = "requires the Sentinel fixture topology"]
fn sentinel_fixture_skips_a_dead_first_sentinel_endpoint() {
    let Some(dead) = env::var("TOKEN_RATE_LIMIT_SENTINEL_DEAD_ENDPOINT")
        .ok()
        .filter(|value| !value.is_empty())
    else {
        eprintln!("skipping: TOKEN_RATE_LIMIT_SENTINEL_DEAD_ENDPOINT not set");
        return;
    };
    assert!(
        env::var("TOKEN_RATE_LIMIT_SENTINEL_ENDPOINTS")
            .is_ok_and(|value| value.split(',').next() == Some(dead.as_str())),
        "the fixture must place the dead Sentinel first in the endpoint list"
    );
    let backend = StatefulCapturingBackend::new(vec![(200, OPENAI_RESPONSE_BODY.to_owned())]).start_with_shutdown();
    let config = fixture_config(free_port(), backend.port());
    let proxy = start_proxy(&config);
    let raw = http_send(proxy.addr(), &json_post("/v1/chat/completions", "{}"));
    assert_eq!(
        parse_status(&raw),
        200,
        "a dead first Sentinel must not prevent discovery through healthy endpoints: {raw}"
    );
    assert_eq!(
        backend.requests().len(),
        1,
        "the recovered admission invokes upstream once"
    );
}

#[test]
#[ignore = "requires the authenticated Sentinel fixture topology"]
fn sentinel_fixture_supports_separate_sentinel_and_data_credentials() {
    let has_sentinel_password = env::var("TOKEN_RATE_LIMIT_SENTINEL_PASSWORD")
        .ok()
        .is_some_and(|value| !value.is_empty());
    let has_data_password = env::var("TOKEN_RATE_LIMIT_SENTINEL_DATA_PASSWORD")
        .ok()
        .is_some_and(|value| !value.is_empty());
    if !has_sentinel_password || !has_data_password {
        eprintln!("skipping: independent Sentinel and data passwords are not set");
        return;
    }
    assert!(
        env::var("TOKEN_RATE_LIMIT_SENTINEL_PASSWORD").ok() != env::var("TOKEN_RATE_LIMIT_SENTINEL_DATA_PASSWORD").ok(),
        "Sentinel and data credentials must be independent"
    );
    let backend = StatefulCapturingBackend::new(vec![
        (200, OPENAI_RESPONSE_BODY.to_owned()),
        (200, OPENAI_RESPONSE_BODY.to_owned()),
    ])
    .start_with_shutdown();
    let sliding = start_proxy(&fixture_config_with_rule(
        free_port(),
        backend.port(),
        SLIDING_QUOTA_RULE,
        "separate-credentials-sliding",
    ));
    let token_bucket = start_proxy(&fixture_config_with_rule(
        free_port(),
        backend.port(),
        TOKEN_BUCKET_QUOTA_RULE,
        "separate-credentials-token-bucket",
    ));

    assert_eq!(
        parse_status(&http_send(sliding.addr(), &json_post("/v1/chat/completions", "{}"))),
        200,
        "sliding-window admission should work with independent credentials"
    );
    assert_eq!(
        parse_status(&http_send(
            token_bucket.addr(),
            &json_post("/v1/chat/completions", "{}")
        )),
        200,
        "token-bucket admission should work with independent credentials"
    );
    assert_eq!(
        backend.requests().len(),
        2,
        "both authenticated admissions invoke upstream"
    );
}

#[test]
fn sentinel_equal_credential_failure_does_not_log_values() {
    let marker = "controlled-equal-credential-negative-marker";
    let output = Command::new(env::current_exe().expect("locate integration test binary"))
        .args([
            "--exact",
            "examples::token_rate_limit_sentinel::sentinel_fixture_supports_separate_sentinel_and_data_credentials",
            "--ignored",
            "--nocapture",
        ])
        .env("TOKEN_RATE_LIMIT_SENTINEL_PASSWORD", marker)
        .env("TOKEN_RATE_LIMIT_SENTINEL_DATA_PASSWORD", marker)
        .env_remove("TOKEN_RATE_LIMIT_SENTINEL_ENDPOINTS")
        .output()
        .expect("run the equal-credential failure path without a topology");
    assert!(
        !output.status.success(),
        "equal credentials must fail before fixture startup"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    for diagnostic in [&stdout, &stderr] {
        assert!(
            !diagnostic.contains(marker),
            "integration failure diagnostics must not echo credential values"
        );
    }
    assert!(
        stderr.contains("Sentinel and data credentials must be independent"),
        "the child must fail on credential independence, not an unrelated startup error"
    );
}

#[test]
#[ignore = "requires the Sentinel fixture topology"]
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
#[ignore = "requires the Sentinel fixture topology"]
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
#[ignore = "requires the Sentinel fixture topology"]
fn sentinel_fixture_warm_connection_reconnects_after_planned_failover() {
    if env::var("TOKEN_RATE_LIMIT_SENTINEL_FAILOVER_BEFORE_COMMAND").as_deref() != Ok("1") {
        eprintln!("skipping: planned failover lane is not enabled");
        return;
    }
    let backend = StatefulCapturingBackend::new(vec![
        (200, QUOTA_RESPONSE_BODY.to_owned()),
        (200, QUOTA_RESPONSE_BODY.to_owned()),
        (200, QUOTA_RESPONSE_BODY.to_owned()),
        (200, QUOTA_RESPONSE_BODY.to_owned()),
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
    let sliding = start_proxy(&fixture_config_with_rule(
        free_port(),
        backend.port(),
        SLIDING_QUOTA_RULE,
        "failover-sliding-quota",
    ));
    let bucket = start_proxy(&fixture_config_with_rule(
        free_port(),
        backend.port(),
        TOKEN_BUCKET_QUOTA_RULE,
        "failover-bucket-quota",
    ));
    for quota_proxy in [&sliding, &bucket] {
        let raw = http_send(quota_proxy.addr(), &json_post("/v1/chat/completions", "{}"));
        assert_eq!(
            parse_status(&raw),
            200,
            "quota must be admitted before promotion: {raw}"
        );
    }
    wait_for_admitted_quota_replication();
    failover_warm_proxy_connection();

    let second = http_send(proxy.addr(), &json_post("/v1/chat/completions", "{}"));
    assert_eq!(
        parse_status(&second),
        200,
        "the proxy must recover its existing Sentinel connection after failover: {second}"
    );
    for quota_proxy in [&sliding, &bucket] {
        let raw = http_send(quota_proxy.addr(), &json_post("/v1/chat/completions", "{}"));
        assert_eq!(
            parse_status(&raw),
            429,
            "replicated quota must still deny after promotion: {raw}"
        );
    }
    assert_eq!(
        backend.requests().len(),
        4,
        "only the initial three admissions and recovered request may invoke upstream"
    );
}

#[test]
fn required_fixture_replacement_updates_the_expected_fragment() {
    assert_eq!(
        replace_required("sentinel: old", "old", "new"),
        "sentinel: new",
        "the required fixture replacement must apply"
    );
}

#[test]
#[should_panic(expected = "no longer contains the fragment the fixture patches")]
fn required_fixture_replacement_rejects_example_drift() {
    drop(replace_required("sentinel: changed", "old", "new"));
}

#[test]
fn replication_barrier_accepts_a_replica_ahead_of_the_primary_snapshot() {
    assert_replication_caught_up(42, 42);
    assert_replication_caught_up(42, 43);
}

#[test]
#[should_panic(expected = "replica must acknowledge the observed primary offset")]
fn replication_barrier_rejects_a_replica_behind_the_primary_snapshot() {
    assert_replication_caught_up(42, 41);
}

#[test]
fn replication_info_fields_accept_redis_and_valkey_line_endings() {
    let info = "role:slave\r\nmaster_link_status:up\r\nslave_repl_offset:123\r\n";
    assert_eq!(
        info_field(info, "slave_repl_offset"),
        Some("123"),
        "parse replica offset"
    );
    assert_eq!(
        info_field(info, "missing"),
        None,
        "absent replication fields stay absent"
    );
}

#[test]
#[ignore = "requires the Sentinel fixture topology"]
fn sentinel_health_noreplicas_failure() {
    if env::var("TOKEN_RATE_LIMIT_SENTINEL_ENDPOINTS").is_err() {
        eprintln!("skipping: TOKEN_RATE_LIMIT_SENTINEL_ENDPOINTS not set");
        return;
    }
    if env::var("TOKEN_RATE_LIMIT_SENTINEL_REPLICATION_HEALTH_TEST").as_deref() != Ok("1") {
        eprintln!("skipping: replication health lane is not enabled");
        return;
    }
    let on_failure = env::var("TOKEN_RATE_LIMIT_SENTINEL_ON_FAILURE").expect("fixture must set on_failure");
    assert!(
        matches!(on_failure.as_str(), "closed" | "open"),
        "TOKEN_RATE_LIMIT_SENTINEL_ON_FAILURE must be closed or open, got: {on_failure}"
    );
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
