// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Black-box Codex CLI acceptance tests for Responses HTTP.
//!
//! Run the pinned compatibility test locally with:
//!
//! ```console
//! PRAXIS_TEST_CODEX_BIN=/absolute/path/to/codex \
//!   cargo test -p praxis-tests-integration --test suite \
//!   codex_http::pinned_codex_completes_chat_backend_coding_workflow_over_http -- --exact
//! ```
//!
//! Required Linux CI wraps both pinned HTTP tests in a network namespace with
//! only loopback enabled, so the client cannot reach an external provider.
//!
//! The executable must report the version in [`CODEX_VERSION`]. To update the
//! pin, update that constant in both Codex acceptance modules, the fixture
//! filename and checksum, and every version, archive, cache path/key, and
//! version assertion in `.github/workflows/integration.yaml`. Run both pinned
//! HTTP and WebSocket tests with the checksum-verified replacement binary.

use std::{
    collections::HashMap,
    ffi::{OsStr, OsString},
    net::{IpAddr, Ipv4Addr},
    path::{Path, PathBuf},
    process::Stdio,
    sync::{
        Arc, LazyLock,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use jsonschema::{Draft, Validator};
#[cfg(unix)]
use nix::{
    errno::Errno,
    sys::signal::{Signal, kill},
    unistd::Pid,
};
use praxis_test_utils::{
    CapturedHttpRequest, HttpBackendEvent, HttpServerAction, example_config_path, free_port, patch_yaml, start_proxy,
    start_scripted_http_backend, start_scripted_http_backend_turns,
};
use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};

use super::harness::TempWorkspace;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Maximum time allowed for process and pipe cleanup after termination.
const CHILD_CLEANUP_TIMEOUT: Duration = Duration::from_secs(2);
/// Maximum retained bytes from each child output pipe.
const MAX_CHILD_OUTPUT_BYTES: usize = 4_194_304; // 4 MiB
/// Required output from the pinned executable.
const CODEX_VERSION: &str = "codex-cli 0.144.1";
/// Synthetic credential that must reach the test backend.
const TEST_API_KEY: &str = "praxis-http-test-key";
/// Synthetic provider credential that replaces the client credential.
const TEST_PROVIDER_API_KEY: &str = "synthetic-provider-key";
/// Fixed prompt shared by the fixture and child process.
const PROMPT: &str = "Reply with exactly PONG over HTTP. Do not call tools.";

/// Environment variable holding the real vLLM base URL for live acceptance.
const LIVE_VLLM_BASE_URL_ENV: &str = "PRAXIS_TEST_CODEX_VLLM_BASE_URL";
/// Environment variable holding the exact model name served by live vLLM.
const LIVE_VLLM_MODEL_ENV: &str = "PRAXIS_TEST_CODEX_VLLM_MODEL";
/// Environment variable holding the ephemeral bearer Praxis injects upstream.
const LIVE_BACKEND_TOKEN_ENV: &str = "CODEX_BACKEND_TOKEN";
/// Environment variable holding the PostgreSQL response-store URL.
const LIVE_DATABASE_URL_ENV: &str = "PRAXIS_TEST_CODEX_DATABASE_URL";
/// Optional Linux network namespace in which the Codex child must run.
const LIVE_NETNS_ENV: &str = "PRAXIS_TEST_CODEX_NETNS";
/// Address exposed to the isolated Codex namespace by the HTTP observer.
const LIVE_LISTEN_ADDRESS_ENV: &str = "PRAXIS_TEST_CODEX_LISTEN_ADDRESS";
/// Demands a real live run instead of allowing the test to skip.
const REQUIRE_LIVE_ENV: &str = "PRAXIS_TEST_CODEX_REQUIRE_LIVE";
/// Demands that the Codex child run inside an egress-blocked namespace.
const REQUIRE_EGRESS_ISOLATION_ENV: &str = "PRAXIS_TEST_REQUIRE_EGRESS_ISOLATION";
/// Live-model turns include inference and tool execution, so allow more time.
const LIVE_CHILD_TIMEOUT: Duration = Duration::from_secs(300);

/// Codex output item types that would indicate an attempted tool call.
const TOOL_ITEM_TYPES: &[&str] = &["command_execution", "file_change", "mcp_tool_call", "web_search"];

/// Prove the pinned Codex client completes an offline turn over HTTP.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pinned_codex_uses_responses_http_through_full_flow() {
    let Some(codex_bin) = std::env::var_os("PRAXIS_TEST_CODEX_BIN") else {
        eprintln!("skipping pinned Codex HTTP acceptance test; PRAXIS_TEST_CODEX_BIN is unset");
        return;
    };
    assert_pinned_codex_version(&codex_bin).await;

    let mut backend = start_scripted_http_backend_turns("POST", "/v1/responses", http_response_script()).await;
    let proxy_port = free_port();
    let yaml = std::fs::read_to_string(example_config_path("openai/responses/http-passthrough.yaml"))
        .expect("http-passthrough example should exist");
    let patched = patch_yaml(&yaml, proxy_port, &HashMap::from([("127.0.0.1:3001", backend.port())]));
    let config = praxis_core::config::Config::from_yaml(&patched).expect("patched config should parse");
    let _proxy = start_proxy(&config);
    let observer = HttpTransportObserver::start(proxy_port).await;

    let working_dir = tempfile::tempdir().expect("temporary working directory should be created");
    let proxy_base_url = format!("http://127.0.0.1:{}", observer.port());
    let output = run_codex(
        &codex_bin,
        CodexRunOptions {
            proxy_base_url: &proxy_base_url,
            model: "test-model",
            working_dir: working_dir.path(),
            prompt: PROMPT,
            sandbox: "read-only",
            execution_timeout: Duration::from_secs(30),
            no_proxy: "127.0.0.1,localhost",
            netns: None,
        },
    )
    .await;
    assert!(
        output.status.success(),
        "Codex failed with status {status:?}\nstdout:\n{stdout}\nstderr:\n{stderr}",
        status = output.status.code(),
        stdout = output.stdout,
        stderr = output.stderr
    );
    assert_codex_jsonl(&output.stdout);

    let requests = observe_codex_requests(&mut backend).await;
    assert!(
        !requests.is_empty(),
        "Codex should send at least one HTTP request; got none"
    );
    assert_no_websocket_upgrades(&mut backend).await;
    assert_no_unexpected_methods(&mut backend).await;
    observer.assert_http_only();
}

/// Prove the pinned client completes a translated, multi-turn coding workflow.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pinned_codex_completes_chat_backend_coding_workflow_over_http() {
    let Some(codex_bin) = std::env::var_os("PRAXIS_TEST_CODEX_BIN") else {
        eprintln!("skipping pinned Codex HTTP acceptance test; PRAXIS_TEST_CODEX_BIN is unset");
        return;
    };
    assert_pinned_codex_version(&codex_bin).await;

    let workspace = TempWorkspace::new().expect("temporary coding workspace should be created");
    let mut backend =
        start_scripted_http_backend_turns("POST", "/v1/chat/completions", chat_coding_response_script()).await;
    let proxy_port = free_port();
    let yaml = std::fs::read_to_string(example_config_path("openai/responses/codex-http-chat-translation.yaml"))
        .expect("Codex translated-provider example should exist");
    let patched = patch_yaml(&yaml, proxy_port, &HashMap::from([("127.0.0.1:3001", backend.port())]));
    let config = praxis_core::config::Config::from_yaml(&patched).expect("patched config should parse");
    let _proxy = start_proxy(&config);
    let observer = HttpTransportObserver::start(proxy_port).await;

    let prompt = "Inspect input.json, copy its expected_content value into result.txt, run ./verify.sh, then summarize with exactly TASK_COMPLETE.";
    let proxy_base_url = format!("http://127.0.0.1:{}", observer.port());
    let output = run_codex(
        &codex_bin,
        CodexRunOptions {
            proxy_base_url: &proxy_base_url,
            model: "test-model",
            working_dir: workspace.path(),
            prompt,
            sandbox: "danger-full-access",
            execution_timeout: Duration::from_secs(30),
            no_proxy: "127.0.0.1,localhost",
            netns: None,
        },
    )
    .await;
    assert!(
        output.status.success(),
        "Codex failed with status {status:?}\nstdout:\n{stdout}\nstderr:\n{stderr}",
        status = output.status.code(),
        stdout = output.stdout,
        stderr = output.stderr
    );
    workspace.assert_successful_completion();

    let requests = observe_translated_chat_requests(&mut backend).await;
    assert_translated_tool_turns(&requests);
    observer.assert_http_only();
    assert_coding_codex_jsonl(&output.stdout);
}

/// Prove pinned Codex completes a real coding/tool workflow through Praxis and
/// vLLM's native Responses endpoint on the GPU runner.
///
/// Codex streams rich client-owned tools. The client-tool compatibility filter
/// lowers those declarations to private functions for vLLM, while the shared
/// stream owner restores the canonical tool lifecycle before Codex sees it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pinned_codex_completes_native_vllm_coding_workflow_over_http() {
    let Some(live) = CodexLiveConfig::from_env() else {
        return;
    };
    assert_pinned_codex_version(&live.codex_bin).await;
    live.require_egress_isolation_if_demanded();

    let proxy_port = free_port();
    let yaml = std::fs::read_to_string(example_config_path("openai/responses/client-tool-compat.yaml"))
        .expect("Codex native Responses example should exist");
    let patched = patch_live_native_codex_config(
        &yaml,
        proxy_port,
        &live.vllm_authority,
        &live.backend_token,
        live.database_url
            .as_deref()
            .unwrap_or_else(|| panic!("{LIVE_DATABASE_URL_ENV} must be set for the native vLLM acceptance test")),
    );
    let config = praxis_core::config::Config::from_yaml(&patched).expect("live native Codex config should parse");

    run_live_codex_coding_workflow(&live, proxy_port, config).await;
}

/// Prove pinned Codex completes the same real coding/tool workflow through
/// Praxis's rich-client-tool Responses-to-Chat composition against live vLLM.
///
/// The backend requires a fresh credential from [`LIVE_BACKEND_TOKEN_ENV`],
/// while Codex only receives [`TEST_API_KEY`]. A successful turn therefore also
/// proves Praxis replaced the gateway credential before forwarding. The
/// deterministic scripted test above remains the exact provider-wire oracle;
/// this test adds real-model and real-client behavior without asserting
/// model-dependent prose or turn count.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pinned_codex_completes_translated_vllm_coding_workflow_over_http() {
    let Some(live) = CodexLiveConfig::from_env() else {
        return;
    };
    assert_pinned_codex_version(&live.codex_bin).await;
    live.require_egress_isolation_if_demanded();

    let proxy_port = free_port();
    let yaml = std::fs::read_to_string(example_config_path(
        "openai/responses/client-tool-compat-chat-completions.yaml",
    ))
    .expect("Codex client-tool Chat composition example should exist");
    let patched = patch_live_codex_config(
        &yaml,
        proxy_port,
        &live.vllm_authority,
        &live.backend_token,
        live.database_url
            .as_deref()
            .unwrap_or_else(|| panic!("{LIVE_DATABASE_URL_ENV} must be set for the translated vLLM acceptance test")),
    );
    let config = praxis_core::config::Config::from_yaml(&patched).expect("live Codex config should parse");

    run_live_codex_coding_workflow(&live, proxy_port, config).await;
}

/// Drive the common pinned-Codex coding task through one live Praxis pipeline.
async fn run_live_codex_coding_workflow(live: &CodexLiveConfig, proxy_port: u16, config: praxis_core::config::Config) {
    let workspace = TempWorkspace::new().expect("temporary coding workspace should be created");
    let _proxy = start_proxy(&config);
    let observer = HttpTransportObserver::start_on(proxy_port, live.listen_address).await;

    if let Some(namespace) = &live.netns {
        verify_egress_isolation(namespace);
    }

    let prompt = "Inspect input.json, copy its expected_content value into result.txt, run ./verify.sh, then summarize what you changed.";
    let proxy_base_url = format!("http://{}:{}", live.listen_address, observer.port());
    let no_proxy = format!("127.0.0.1,localhost,{}", live.listen_address);
    let output = run_codex(
        &live.codex_bin,
        CodexRunOptions {
            proxy_base_url: &proxy_base_url,
            model: &live.model,
            working_dir: workspace.path(),
            prompt,
            sandbox: "danger-full-access",
            execution_timeout: LIVE_CHILD_TIMEOUT,
            no_proxy: &no_proxy,
            netns: live.netns.as_deref(),
        },
    )
    .await;
    assert!(
        output.status.success(),
        "Codex failed with status {status:?}\nstdout:\n{stdout}\nstderr:\n{stderr}",
        status = output.status.code(),
        stdout = output.stdout,
        stderr = output.stderr
    );

    workspace.assert_successful_completion();
    observer.assert_http_only();
    assert_live_coding_codex_jsonl(&output.stdout);
}

/// Keep the live backend credential ephemeral rather than coupled to the example fixture.
#[test]
fn live_codex_config_replaces_the_provider_credential() {
    let yaml = std::fs::read_to_string(example_config_path(
        "openai/responses/client-tool-compat-chat-completions.yaml",
    ))
    .expect("Codex client-tool Chat composition example should exist");
    let backend_token = "ephemeral-backend-token-for-this-test";
    let database_url = "postgres://praxis:praxis@127.0.0.1:5432/praxis";
    let patched = patch_live_codex_config(&yaml, 18_080, "127.0.0.1:8000", backend_token, database_url);

    assert!(patched.contains(backend_token));
    assert!(!patched.contains(TEST_PROVIDER_API_KEY));
    assert!(patched.contains(database_url));
    assert!(patched.contains("backend: postgres"));
    assert!(patched.contains("allow_private_database_url: true"));
    assert!(!patched.contains("sqlite://responses.db?mode=rwc"));
    praxis_core::config::Config::from_yaml(&patched).expect("patched live Codex config should parse");
}

/// Keep the native Responses acceptance store private and replace the client
/// bearer before the request reaches keyed vLLM.
#[test]
fn live_native_codex_config_injects_the_provider_credential() {
    let yaml = std::fs::read_to_string(example_config_path("openai/responses/client-tool-compat.yaml"))
        .expect("Codex native Responses example should exist");
    let backend_token = "ephemeral-native-backend-token-for-this-test";
    let database_url = "postgres://praxis:praxis@127.0.0.1:5432/praxis";
    let patched = patch_live_native_codex_config(&yaml, 18_080, "127.0.0.1:8000", backend_token, database_url);

    assert!(patched.contains(backend_token));
    assert!(patched.contains("strip_client_credential: true"));
    assert!(patched.contains(database_url));
    assert!(patched.contains("backend: postgres"));
    assert!(patched.contains("allow_private_database_url: true"));
    assert!(!patched.contains("sqlite://responses.db?mode=rwc"));
    praxis_core::config::Config::from_yaml(&patched).expect("patched live native Codex config should parse");
}

/// Validate the native SSE fixture's resource snapshots against the pinned OpenResponses contract.
#[test]
fn native_sse_response_resources_match_openresponses_schema() {
    let turns = http_response_script();
    let HttpServerAction::StreamSse { events, .. } = &turns[0] else {
        panic!("native Responses fixture should stream SSE");
    };
    let mut count = 0;
    for event in events {
        let Some(payload) = event.lines().find_map(|line| line.strip_prefix("data: ")) else {
            continue;
        };
        let payload: serde_json::Value = serde_json::from_str(payload).expect("SSE fixture data should be JSON");
        let Some(resource) = payload.get("response") else {
            continue;
        };
        assert_openresponses_response_resource(resource);
        count += 1;
    }
    assert_eq!(
        count, 2,
        "native fixture should include created and completed resources"
    );
}

/// Prove the fixture validator rejects missing fields and malformed usage details.
#[test]
fn native_sse_response_schema_validation_is_sensitive() {
    let mut missing_model = http_response_resource("in_progress", Vec::new(), serde_json::Value::Null);
    missing_model
        .as_object_mut()
        .expect("fixture response should be an object")
        .remove("model");
    assert!(
        OPENRESPONSES_RESPONSE_RESOURCE_VALIDATOR
            .validate(&missing_model)
            .is_err(),
        "schema validation should reject a missing required ResponseResource field"
    );

    let invalid_usage = http_response_resource(
        "completed",
        Vec::new(),
        serde_json::json!({
            "input_tokens": 0,
            "input_tokens_details": null,
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 0
        }),
    );
    assert!(
        OPENRESPONSES_RESPONSE_RESOURCE_VALIDATOR
            .validate(&invalid_usage)
            .is_err(),
        "schema validation should reject null input token details"
    );
}

/// Prove the translated client stream starts before the Chat stream completes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn translated_chat_sse_reaches_client_before_upstream_finishes() {
    let first_chunk = serde_json::json!({
        "id": "chatcmpl-incremental",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "content": "EARLY"},
            "finish_reason": null
        }]
    });
    let terminal_chunk = serde_json::json!({
        "id": "chatcmpl-incremental",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "test-model",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    });
    let mut backend = start_scripted_http_backend_turns(
        "POST",
        "/v1/chat/completions",
        vec![HttpServerAction::StreamSse {
            events: vec![
                chat_sse_data(&first_chunk),
                chat_sse_data(&terminal_chunk),
                "data: [DONE]\n".to_owned(),
            ],
            inter_event_delay: Duration::from_secs(5),
        }],
    )
    .await;
    let proxy_port = free_port();
    let yaml = std::fs::read_to_string(example_config_path("openai/responses/codex-http-chat-translation.yaml"))
        .expect("Codex translated-provider example should exist");
    let patched = patch_yaml(&yaml, proxy_port, &HashMap::from([("127.0.0.1:3001", backend.port())]));
    let config = praxis_core::config::Config::from_yaml(&patched).expect("patched config should parse");
    let _proxy = start_proxy(&config);

    let client = tokio::spawn(read_first_translated_delta(proxy_port));
    let request = tokio::time::timeout(Duration::from_secs(5), backend.next_event())
        .await
        .expect("provider should receive translated request")
        .expect("backend event channel should remain open");
    let HttpBackendEvent::Request(request) = request else {
        panic!("provider should receive a normal HTTP request, got {request:?}");
    };
    assert_eq!(request.path, "/v1/chat/completions");

    tokio::time::timeout(Duration::from_secs(2), client)
        .await
        .expect("client should receive a translated delta before the delayed terminal Chat chunk")
        .expect("client reader task should finish");
}

/// Prove the front-door observer records an Upgrade before forwarding it.
#[tokio::test]
async fn transport_observer_detects_websocket_attempts() {
    let mut backend = start_scripted_http_backend("GET", "/v1/responses", vec![]).await;
    let observer = HttpTransportObserver::start(backend.port()).await;
    let mut client = tokio::net::TcpStream::connect((Ipv4Addr::LOCALHOST, observer.port()))
        .await
        .expect("WebSocket probe should connect to observer");
    client
        .write_all(
            b"GET /v1/responses HTTP/1.1\r\nHost: localhost\r\nConnection: Upgrade\r\nUpgrade:\twebsocket\r\nContent-Length: 0\r\n\r\n",
        )
        .await
        .expect("WebSocket probe should be written");
    let event = tokio::time::timeout(Duration::from_secs(2), backend.next_event())
        .await
        .expect("backend should receive forwarded WebSocket probe")
        .expect("backend event channel should remain open");
    assert!(
        matches!(event, HttpBackendEvent::WebSocketUpgrade { .. }),
        "scripted backend should confirm the forwarded Upgrade request: {event:?}"
    );
    assert!(
        observer.websocket_attempted(),
        "front-door observer should record the WebSocket attempt"
    );
}

/// A timed-out child must not leave descendants holding its captured pipes.
#[cfg(unix)]
#[tokio::test]
async fn timed_out_child_kills_process_group_and_closes_inherited_pipes() {
    let mut command = tokio::process::Command::new("/bin/sh");
    command
        .arg("-c")
        .arg("sleep 30 & wait")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    configure_isolated_process_group(&mut command);
    let child = command.spawn().expect("shell fixture should start");

    let output = tokio::time::timeout(
        Duration::from_secs(2),
        capture_child_output(child, Duration::from_millis(25)),
    )
    .await
    .expect("process-group cleanup and pipe collection should be bounded");

    assert!(output.timed_out, "shell fixture should hit the test timeout");
    assert!(!output.status.success(), "terminated shell fixture should fail");
}

/// Live infrastructure supplied by the GPU acceptance workflow.
struct CodexLiveConfig {
    /// Absolute path to the checksum-verified Codex executable.
    codex_bin: OsString,
    /// Host and port of the live vLLM backend.
    vllm_authority: String,
    /// Exact slash-free model alias served by vLLM.
    model: String,
    /// Ephemeral backend bearer that Praxis must inject toward vLLM.
    backend_token: String,
    /// Optional PostgreSQL response store used by the native Responses path.
    database_url: Option<String>,
    /// Host-side address exposed to the isolated Codex namespace.
    listen_address: IpAddr,
    /// Optional namespace in which the Codex child must run.
    netns: Option<String>,
}

impl CodexLiveConfig {
    /// Resolve the live gate, skipping locally but failing when CI demands it.
    fn from_env() -> Option<Self> {
        let codex_bin = std::env::var_os("PRAXIS_TEST_CODEX_BIN");
        let vllm_base = std::env::var(LIVE_VLLM_BASE_URL_ENV).ok();
        let model = std::env::var(LIVE_VLLM_MODEL_ENV).ok();
        let backend_token = std::env::var(LIVE_BACKEND_TOKEN_ENV).ok();
        let (Some(codex_bin), Some(vllm_base), Some(model), Some(backend_token)) =
            (codex_bin, vllm_base, model, backend_token)
        else {
            assert!(
                !env_is_truthy(REQUIRE_LIVE_ENV),
                "{REQUIRE_LIVE_ENV} is set but a required variable is missing; set all of \
                 PRAXIS_TEST_CODEX_BIN, {LIVE_VLLM_BASE_URL_ENV}, {LIVE_VLLM_MODEL_ENV}, \
                 and {LIVE_BACKEND_TOKEN_ENV}"
            );
            eprintln!(
                "skipping live-vLLM Codex acceptance test; set PRAXIS_TEST_CODEX_BIN, \
                 {LIVE_VLLM_BASE_URL_ENV}, {LIVE_VLLM_MODEL_ENV}, and \
                 {LIVE_BACKEND_TOKEN_ENV} to run it"
            );
            return None;
        };

        let listen_address = std::env::var(LIVE_LISTEN_ADDRESS_ENV)
            .ok()
            .and_then(|value| value.parse::<IpAddr>().ok())
            .unwrap_or(IpAddr::V4(Ipv4Addr::LOCALHOST));

        Some(Self {
            codex_bin,
            vllm_authority: authority_of(&vllm_base),
            model,
            backend_token,
            database_url: std::env::var(LIVE_DATABASE_URL_ENV).ok(),
            listen_address,
            netns: std::env::var(LIVE_NETNS_ENV).ok().filter(|value| !value.is_empty()),
        })
    }

    /// Refuse a CI live run that silently lost its namespace configuration.
    fn require_egress_isolation_if_demanded(&self) {
        if env_is_truthy(REQUIRE_EGRESS_ISOLATION_ENV) {
            assert!(
                self.netns.is_some(),
                "{REQUIRE_EGRESS_ISOLATION_ENV} is set but {LIVE_NETNS_ENV} is not; \
                 the Codex child would have external network access"
            );
        }
    }
}

/// Patch the shipped translated-provider example for a real, slower backend.
fn patch_live_codex_config(
    yaml: &str,
    proxy_port: u16,
    vllm_authority: &str,
    backend_token: &str,
    database_url: &str,
) -> String {
    patch_yaml(yaml, proxy_port, &HashMap::new())
        .replace("127.0.0.1:3001", vllm_authority)
        .replace(TEST_PROVIDER_API_KEY, backend_token)
        .replace("backend: sqlite", "backend: postgres")
        .replace(
            "database_url: \"sqlite://responses.db?mode=rwc\"",
            &format!(
                "database_url: \"{database_url}\"\n        allow_private_database_url: true\n        ssl_mode: disable"
            ),
        )
        .replace("timeout_ms: 30000", "timeout_ms: 300000")
        .replace("timeout_secs: 30", "timeout_secs: 300")
        .replace(
            "                  - name: chat-provider\n                    endpoints:",
            "                  - name: chat-provider\n                    read_timeout_ms: 300000\n                    endpoints:",
        )
}

/// Patch the native Responses example for a keyed, slower live vLLM backend.
fn patch_live_native_codex_config(
    yaml: &str,
    proxy_port: u16,
    vllm_authority: &str,
    backend_token: &str,
    database_url: &str,
) -> String {
    const LOAD_BALANCER_FILTER: &str = "              - filter: load_balancer";

    let credential_filter = format!(
        r#"              - filter: credential_injection
                clusters:
                  - name: inference-backend
                    header: Authorization
                    value: "{backend_token}"
                    header_prefix: "Bearer "
                    strip_client_credential: true
"#
    );
    assert!(
        yaml.contains(LOAD_BALANCER_FILTER),
        "client-tool-compat example should contain the inference load balancer"
    );

    let patched = patch_yaml(yaml, proxy_port, &HashMap::new())
        .replace("127.0.0.1:3001", vllm_authority)
        .replace("backend: sqlite", "backend: postgres")
        .replace(
            "database_url: \"sqlite://responses.db?mode=rwc\"",
            &format!(
                "database_url: \"{database_url}\"\n        allow_private_database_url: true\n        ssl_mode: disable"
            ),
        )
        .replace(
            "        max_iterations: 4",
            "        max_iterations: 4\n        timeout_ms: 300000\n        step_timeout_ms: 300000",
        )
        .replace(
            "                  - name: \"inference-backend\"\n                    endpoints:",
            "                  - name: \"inference-backend\"\n                    read_timeout_ms: 300000\n                    endpoints:",
        );

    patched.replacen(
        LOAD_BALANCER_FILTER,
        &format!("{credential_filter}{LOAD_BALANCER_FILTER}"),
        1,
    )
}

/// Strip scheme and trailing slash from a backend URL for Praxis endpoints.
fn authority_of(base: &str) -> String {
    base.trim()
        .trim_start_matches("http://")
        .trim_start_matches("https://")
        .trim_end_matches('/')
        .to_owned()
}

/// Reports whether a gate environment variable is explicitly truthy.
fn env_is_truthy(name: &str) -> bool {
    std::env::var(name)
        .map(|value| matches!(value.trim(), "1" | "true" | "TRUE" | "yes" | "on"))
        .unwrap_or(false)
}

/// Prove the configured child namespace has no default or public route.
fn verify_egress_isolation(namespace: &str) {
    let routes = std::process::Command::new(resolve_ip_binary())
        .args(["netns", "exec", namespace, "ip", "route", "show", "default"])
        .output()
        .expect("ip should inspect the Codex namespace");
    assert!(routes.status.success(), "default-route inspection should succeed");
    assert!(
        routes.stdout.is_empty(),
        "Codex namespace must not have a default route: {}",
        String::from_utf8_lossy(&routes.stdout)
    );

    let public_route = std::process::Command::new(resolve_ip_binary())
        .args(["netns", "exec", namespace, "ip", "route", "get", "1.1.1.1"])
        .status()
        .expect("ip should probe public routing from the Codex namespace");
    assert!(
        !public_route.success(),
        "Codex namespace unexpectedly has a route to the public internet"
    );
}

/// Resolve `ip(8)` before clearing the child environment.
///
/// GPU runner images commonly install it under `/usr/sbin`, which is absent
/// from the deliberately minimal PATH passed to the pinned Codex process.
fn resolve_ip_binary() -> PathBuf {
    ["/usr/sbin/ip", "/sbin/ip", "/usr/bin/ip", "/bin/ip"]
        .into_iter()
        .map(PathBuf::from)
        .find(|candidate| candidate.exists())
        .unwrap_or_else(|| PathBuf::from("ip"))
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

/// Captured child-process result with decoded output.
struct CodexOutput {
    /// Exit status.
    status: std::process::ExitStatus,
    /// UTF-8-lossy standard error.
    stderr: String,
    /// UTF-8-lossy standard output.
    stdout: String,
}

/// Inputs controlling one isolated Codex child process.
struct CodexRunOptions<'a> {
    /// Praxis base URL visible to the Codex child, without `/v1`.
    proxy_base_url: &'a str,
    /// Model name Codex sends through Praxis.
    model: &'a str,
    /// Coding workspace used as the child current directory.
    working_dir: &'a Path,
    /// User prompt for this acceptance turn.
    prompt: &'a str,
    /// Codex sandbox policy.
    sandbox: &'a str,
    /// Maximum wall-clock time allowed for the child.
    execution_timeout: Duration,
    /// Addresses the child may reach without using the deliberately dead proxy.
    no_proxy: &'a str,
    /// Optional egress-blocked Linux network namespace.
    netns: Option<&'a str>,
}

/// Raw child-process output plus timeout state.
struct CapturedChildOutput {
    /// Child exit status.
    status: std::process::ExitStatus,
    /// Captured standard error.
    stderr: Vec<u8>,
    /// Captured standard output.
    stdout: Vec<u8>,
    /// Whether the child exceeded its execution timeout.
    timed_out: bool,
}

/// Test-local TCP observer at the Codex-facing boundary.
struct HttpTransportObserver {
    /// Listener task forwarding traffic to Praxis.
    handle: tokio::task::JoinHandle<()>,
    /// Port exposed to Codex.
    port: u16,
    /// Shutdown signal for the listener.
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    /// Whether any connection attempted a WebSocket upgrade.
    websocket_attempted: Arc<AtomicBool>,
}

impl HttpTransportObserver {
    /// Bind a front-door observer that forwards all connections to Praxis.
    async fn start(upstream_port: u16) -> Self {
        Self::start_on(upstream_port, IpAddr::V4(Ipv4Addr::LOCALHOST)).await
    }

    /// Bind the observer on an explicit address, including a host-side veth.
    async fn start_on(upstream_port: u16, listen_address: IpAddr) -> Self {
        let listener = tokio::net::TcpListener::bind((listen_address, 0))
            .await
            .expect("transport observer should bind");
        let port = listener
            .local_addr()
            .expect("transport observer should have an address")
            .port();
        let websocket_attempted = Arc::new(AtomicBool::new(false));
        let observed = Arc::clone(&websocket_attempted);
        let (shutdown_tx, mut shutdown_rx) = tokio::sync::oneshot::channel();
        let handle = tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = &mut shutdown_rx => break,
                    accepted = listener.accept() => {
                        let Ok((client, _peer)) = accepted else {
                            break;
                        };
                        let observed = Arc::clone(&observed);
                        tokio::spawn(forward_observed_connection(client, upstream_port, observed));
                    },
                }
            }
        });
        Self {
            handle,
            port,
            shutdown: Some(shutdown_tx),
            websocket_attempted,
        }
    }

    /// Port Codex should use as its Praxis base URL.
    fn port(&self) -> u16 {
        self.port
    }

    /// Fail even when Codex recovered from an attempted WebSocket via HTTP.
    fn assert_http_only(&self) {
        assert!(
            !self.websocket_attempted(),
            "Codex attempted a WebSocket upgrade before or during the successful HTTP workflow"
        );
    }

    /// Return whether an opening request contained `Upgrade: websocket`.
    fn websocket_attempted(&self) -> bool {
        self.websocket_attempted.load(Ordering::SeqCst)
    }
}

impl Drop for HttpTransportObserver {
    fn drop(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _sent = shutdown.send(());
        }
        self.handle.abort();
    }
}

/// Inspect one connection's opening HTTP request, then forward bytes unchanged.
async fn forward_observed_connection(
    mut client: tokio::net::TcpStream,
    upstream_port: u16,
    websocket_attempted: Arc<AtomicBool>,
) {
    let mut upstream = tokio::net::TcpStream::connect((Ipv4Addr::LOCALHOST, upstream_port))
        .await
        .expect("transport observer should connect to Praxis");
    let mut opening = Vec::with_capacity(4096);
    let mut chunk = [0_u8; 4096];
    while opening.len() < 16_384 && !opening.windows(4).any(|window| window == b"\r\n\r\n") {
        let read = client
            .read(&mut chunk)
            .await
            .expect("observer should read Codex request");
        if read == 0 {
            return;
        }
        opening.extend_from_slice(&chunk[..read]);
    }
    if request_head_has_websocket_upgrade(&opening) {
        websocket_attempted.store(true, Ordering::SeqCst);
    }
    if upstream.write_all(&opening).await.is_err() {
        return;
    }
    let _forwarded = tokio::io::copy_bidirectional(&mut client, &mut upstream).await;
}

/// Parse the opening HTTP head and recognize valid WebSocket header spacing.
fn request_head_has_websocket_upgrade(opening: &[u8]) -> bool {
    String::from_utf8_lossy(opening).lines().skip(1).any(|line| {
        let Some((name, value)) = line.split_once(':') else {
            return false;
        };
        name.trim().eq_ignore_ascii_case("upgrade") && value.trim().eq_ignore_ascii_case("websocket")
    })
}

/// Send one streaming Responses request and return after its first text delta.
async fn read_first_translated_delta(proxy_port: u16) {
    let body = r#"{"model":"test-model","input":"ping","stream":true}"#;
    let request = format!(
        "POST /v1/responses HTTP/1.1\r\nHost: 127.0.0.1:{proxy_port}\r\n\
         Authorization: Bearer {TEST_API_KEY}\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let mut stream = tokio::net::TcpStream::connect((Ipv4Addr::LOCALHOST, proxy_port))
        .await
        .expect("streaming client should connect to Praxis");
    stream
        .write_all(request.as_bytes())
        .await
        .expect("streaming request should be written");

    let mut response = Vec::new();
    let mut chunk = [0_u8; 4096];
    loop {
        let read = stream
            .read(&mut chunk)
            .await
            .expect("streaming response should be readable");
        assert!(read > 0, "translated stream ended before its first text delta");
        response.extend_from_slice(&chunk[..read]);
        if response
            .windows(b"response.output_text.delta".len())
            .any(|window| window == b"response.output_text.delta")
        {
            return;
        }
    }
}

/// Build the scripted HTTP response turns.
///
/// The scripted backend delivers a `PONG` text response in SSE form for
/// each request Codex sends. The test backend delivers the same response
/// for every turn so the assertion is deterministic regardless of how
/// many times Codex polls the endpoint.
#[expect(clippy::large_stack_frames, reason = "scripted turns are test fixtures")]
fn http_response_script() -> Vec<HttpServerAction> {
    let turn = HttpServerAction::StreamSse {
        events: vec![
            sse_event(
                "response.created",
                &serde_json::json!({
                    "type": "response.created",
                    "sequence_number": 0,
                    "response": http_response_resource("in_progress", Vec::new(), serde_json::Value::Null)
                }),
            ),
            sse_event(
                "response.output_item.added",
                &serde_json::json!({
                    "type": "response.output_item.added",
                    "sequence_number": 1,
                    "output_index": 0,
                    "item": {
                        "id": "msg_http_acceptance",
                        "type": "message",
                        "role": "assistant",
                        "status": "in_progress",
                        "content": []
                    }
                }),
            ),
            sse_event(
                "response.content_part.added",
                &serde_json::json!({
                    "type": "response.content_part.added",
                    "sequence_number": 2,
                    "item_id": "msg_http_acceptance",
                    "output_index": 0,
                    "content_index": 0,
                    "part": {
                        "type": "output_text",
                        "text": "",
                        "annotations": []
                    }
                }),
            ),
            sse_event(
                "response.output_text.delta",
                &serde_json::json!({
                    "type": "response.output_text.delta",
                    "sequence_number": 3,
                    "item_id": "msg_http_acceptance",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": "PONG"
                }),
            ),
            sse_event(
                "response.output_text.done",
                &serde_json::json!({
                    "type": "response.output_text.done",
                    "sequence_number": 4,
                    "item_id": "msg_http_acceptance",
                    "output_index": 0,
                    "content_index": 0,
                    "text": "PONG"
                }),
            ),
            sse_event(
                "response.content_part.done",
                &serde_json::json!({
                    "type": "response.content_part.done",
                    "sequence_number": 5,
                    "item_id": "msg_http_acceptance",
                    "output_index": 0,
                    "content_index": 0,
                    "part": {
                        "type": "output_text",
                        "text": "PONG",
                        "annotations": []
                    }
                }),
            ),
            sse_event(
                "response.output_item.done",
                &serde_json::json!({
                    "type": "response.output_item.done",
                    "sequence_number": 6,
                    "output_index": 0,
                    "item": {
                        "id": "msg_http_acceptance",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "PONG",
                                "annotations": []
                            }
                        ]
                    }
                }),
            ),
            sse_event(
                "response.completed",
                &serde_json::json!({
                    "type": "response.completed",
                    "sequence_number": 7,
                    "response": http_response_resource(
                        "completed",
                        vec![serde_json::json!({
                            "id": "msg_http_acceptance",
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{
                                "type": "output_text",
                                "text": "PONG",
                                "annotations": []
                            }]
                        })],
                        serde_json::json!({
                            "input_tokens": 0,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 0,
                            "output_tokens_details": {"reasoning_tokens": 0},
                            "total_tokens": 0
                        }),
                    )
                }),
            ),
        ],
        inter_event_delay: Duration::ZERO,
    };
    // A handful of follow-up turns is enough to let Codex finish the
    // multi-step task. The test asserts at least one request so any
    // additional follow-up turns simply keep the script running until
    // Codex decides the turn is complete.
    let mut turns = vec![turn.clone(), turn];
    for _ in 0..6 {
        turns.push(turns[0].clone());
    }
    turns
}

/// Build a canonical OpenResponses resource snapshot for native SSE events.
fn http_response_resource(status: &str, output: Vec<serde_json::Value>, usage: serde_json::Value) -> serde_json::Value {
    let completed_at = if status == "completed" {
        serde_json::json!(2)
    } else {
        serde_json::Value::Null
    };
    let mut response = serde_json::json!({
        "id": "resp_http_acceptance",
        "object": "response",
        "created_at": 1,
        "completed_at": completed_at,
        "status": status,
        "incomplete_details": null,
        "model": "test-model",
        "previous_response_id": null,
        "instructions": null,
        "error": null,
        "tools": [],
        "tool_choice": "auto",
        "truncation": "disabled",
        "parallel_tool_calls": true,
        "text": {"format": {"type": "text"}},
        "top_p": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "top_logprobs": 0,
        "temperature": 1.0,
        "reasoning": {"effort": null, "summary": null},
        "user": null,
        "max_output_tokens": null,
        "max_tool_calls": null,
        "store": true,
        "background": false,
        "service_tier": "default",
        "metadata": {},
        "safety_identifier": null,
        "prompt_cache_key": null
    });
    response["output"] = serde_json::Value::Array(output);
    response["usage"] = usage;
    response
}

/// Schema projection of the required fields in OpenResponses 92c12d96.
///
/// The canonical schema is split across 164 files upstream. This executable
/// projection covers the ResponseResource and Usage contract used by this
/// fixture without vendoring unrelated response item variants.
static OPENRESPONSES_RESPONSE_RESOURCE_VALIDATOR: LazyLock<Validator> = LazyLock::new(|| {
    let schema = serde_json::from_str(
        r#"{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","required":["id","object","created_at","completed_at","status","incomplete_details","model","previous_response_id","instructions","output","error","tools","tool_choice","truncation","parallel_tool_calls","text","top_p","presence_penalty","frequency_penalty","top_logprobs","temperature","reasoning","user","usage","max_output_tokens","max_tool_calls","store","background","service_tier","metadata","safety_identifier","prompt_cache_key"],"properties":{"id":{"type":"string"},"object":{"const":"response"},"created_at":{"type":"integer"},"completed_at":{"type":["integer","null"]},"status":{"type":"string"},"incomplete_details":{"type":["object","null"]},"model":{"type":"string"},"previous_response_id":{"type":["string","null"]},"instructions":{"type":["string","null"]},"output":{"type":"array"},"error":{"type":["object","null"]},"tools":{"type":"array"},"tool_choice":{"type":["string","object"]},"truncation":{"type":"string"},"parallel_tool_calls":{"type":"boolean"},"text":{"type":"object"},"top_p":{"type":"number"},"presence_penalty":{"type":"number"},"frequency_penalty":{"type":"number"},"top_logprobs":{"type":"integer"},"temperature":{"type":"number"},"reasoning":{"type":["object","null"]},"user":{"type":["string","null"]},"usage":{"anyOf":[{"type":"null"},{"type":"object","required":["input_tokens","output_tokens","total_tokens","input_tokens_details","output_tokens_details"],"properties":{"input_tokens":{"type":"integer"},"output_tokens":{"type":"integer"},"total_tokens":{"type":"integer"},"input_tokens_details":{"type":"object","required":["cached_tokens"],"properties":{"cached_tokens":{"type":"integer"}}},"output_tokens_details":{"type":"object","required":["reasoning_tokens"],"properties":{"reasoning_tokens":{"type":"integer"}}}}}]},"max_output_tokens":{"type":["integer","null"]},"max_tool_calls":{"type":["integer","null"]},"store":{"type":"boolean"},"background":{"type":"boolean"},"service_tier":{"type":"string"},"metadata":{"type":"object"},"safety_identifier":{"type":["string","null"]},"prompt_cache_key":{"type":["string","null"]}}}"#,
    )
    .expect("OpenResponses fixture schema projection should parse");
    jsonschema::options()
        .with_draft(Draft::Draft202012)
        .build(&schema)
        .expect("OpenResponses fixture schema projection should compile")
});

/// Validate a fixture resource against its pinned OpenResponses contract.
fn assert_openresponses_response_resource(resource: &serde_json::Value) {
    OPENRESPONSES_RESPONSE_RESOURCE_VALIDATOR
        .validate(resource)
        .unwrap_or_else(|error| panic!("native Responses fixture violates OpenResponses 92c12d96: {error}"));
}

/// Build two Chat Completions SSE turns: one command call and one summary.
#[expect(
    clippy::large_stack_frames,
    reason = "Chat SSE JSON values are bounded test fixtures"
)]
fn chat_coding_response_script() -> Vec<HttpServerAction> {
    let command = r#"expected=$(sed -n 's/.*"expected_content": "\(.*\)".*/\1/p' input.json); test -n "$expected"; printf '%s\n' "$expected" > result.txt; ./verify.sh"#;
    let arguments = serde_json::json!({
        "cmd": command,
        "yield_time_ms": 30_000,
    })
    .to_string();
    let tool_chunk = serde_json::json!({
        "id": "chatcmpl-codex-tool",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "delta": {
                "role": "assistant",
                "tool_calls": [{
                    "index": 0,
                    "id": "call_exec_1",
                    "type": "function",
                    "function": {
                        "name": "exec_command",
                        "arguments": arguments,
                    }
                }]
            },
            "finish_reason": null
        }]
    });
    let tool_done = serde_json::json!({
        "id": "chatcmpl-codex-tool",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "test-model",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}
    });
    let summary_chunk = serde_json::json!({
        "id": "chatcmpl-codex-summary",
        "object": "chat.completion.chunk",
        "created": 2,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "content": "TASK_COMPLETE"},
            "finish_reason": null
        }]
    });
    let summary_done = serde_json::json!({
        "id": "chatcmpl-codex-summary",
        "object": "chat.completion.chunk",
        "created": 2,
        "model": "test-model",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24}
    });

    vec![
        HttpServerAction::StreamSse {
            events: vec![
                chat_sse_data(&tool_chunk),
                chat_sse_data(&tool_done),
                "data: [DONE]\n".to_owned(),
            ],
            inter_event_delay: Duration::ZERO,
        },
        HttpServerAction::StreamSse {
            events: vec![
                chat_sse_data(&summary_chunk),
                chat_sse_data(&summary_done),
                "data: [DONE]\n".to_owned(),
            ],
            inter_event_delay: Duration::ZERO,
        },
    ]
}

/// Render one Chat Completions `data:` SSE frame.
fn chat_sse_data(payload: &serde_json::Value) -> String {
    format!("data: {payload}\n")
}

/// Render a single SSE event block from an `event:` name and a JSON payload.
fn sse_event(event: &str, payload: &serde_json::Value) -> String {
    format!("event: {event}\ndata: {payload}\n")
}

/// Confirm that the explicitly provided binary matches the fixture pin.
async fn assert_pinned_codex_version(codex_bin: &OsStr) {
    let output = tokio::process::Command::new(codex_bin)
        .arg("--version")
        .output()
        .await
        .expect("PRAXIS_TEST_CODEX_BIN should execute");
    assert!(output.status.success(), "codex --version should succeed");
    assert_eq!(
        String::from_utf8_lossy(&output.stdout).trim(),
        CODEX_VERSION,
        "acceptance fixture and executable pin must match"
    );
}

/// Run Codex with isolated configuration, credentials, input, and workspace.
async fn run_codex(codex_bin: &OsStr, options: CodexRunOptions<'_>) -> CodexOutput {
    let CodexRunOptions {
        proxy_base_url,
        model,
        working_dir,
        prompt,
        sandbox,
        execution_timeout,
        no_proxy,
        netns,
    } = options;
    let codex_home = tempfile::tempdir().expect("temporary CODEX_HOME should be created");
    let config = format!(
        r#"model = "{model}"
model_provider = "praxis"
web_search = "disabled"

[features]
apps = false
browser_use = false
browser_use_external = false
computer_use = false
goals = false
image_generation = false
multi_agent = false
tool_suggest = false

[model_providers.praxis]
name = "Praxis test gateway"
base_url = "{proxy_base_url}/v1"
wire_api = "responses"
env_key = "PRAXIS_TEST_API_KEY"
"#
    );
    std::fs::write(codex_home.path().join("config.toml"), config).expect("test config should be written");

    let mut child = match netns {
        Some(namespace) => {
            let mut command = tokio::process::Command::new(resolve_ip_binary());
            command.arg("netns").arg("exec").arg(namespace).arg(codex_bin);
            command
        },
        None => tokio::process::Command::new(codex_bin),
    };
    child
        .arg("exec")
        .arg("--ephemeral")
        .arg("--strict-config")
        .arg("--skip-git-repo-check")
        .arg("--sandbox")
        .arg(sandbox)
        .arg("--json")
        .arg(prompt)
        .current_dir(working_dir)
        .env_clear()
        .env("CODEX_HOME", codex_home.path())
        .env("HOME", codex_home.path())
        .env("PATH", "/usr/bin:/bin:/usr/local/bin")
        .env("PRAXIS_TEST_API_KEY", TEST_API_KEY)
        .env("RUST_LOG", "error")
        .env("HTTP_PROXY", "http://127.0.0.1:1")
        .env("HTTPS_PROXY", "http://127.0.0.1:1")
        .env("ALL_PROXY", "http://127.0.0.1:1")
        .env("NO_PROXY", no_proxy)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    configure_isolated_process_group(&mut child);
    let child = child.spawn().expect("pinned Codex should start");
    let captured = capture_child_output(child, execution_timeout).await;

    let output = CodexOutput {
        status: captured.status,
        stderr: String::from_utf8_lossy(&captured.stderr).into_owned(),
        stdout: String::from_utf8_lossy(&captured.stdout).into_owned(),
    };
    assert!(
        !captured.timed_out,
        "Codex process exceeded {execution_timeout:?} acceptance-test timeout\nstdout:\n{}\nstderr:\n{}",
        output.stdout, output.stderr
    );
    output
}

/// Collect and validate provider-facing requests from the translated workflow.
async fn observe_translated_chat_requests(
    backend: &mut praxis_test_utils::HttpBackendGuard,
) -> Vec<CapturedHttpRequest> {
    let mut requests = Vec::new();
    while requests.len() < 2 {
        let event = tokio::time::timeout(Duration::from_secs(10), backend.next_event())
            .await
            .expect("translated provider request should arrive within ten seconds")
            .expect("backend observation channel should remain open");
        match event {
            HttpBackendEvent::Request(request) => {
                assert_eq!(request.method, "POST", "translated provider method should be POST");
                assert_eq!(
                    request.path, "/v1/chat/completions",
                    "translated provider path should be Chat Completions"
                );
                let authorization = request
                    .headers
                    .get_all(http::header::AUTHORIZATION)
                    .iter()
                    .map(|value| value.to_str().expect("provider authorization should be ASCII"))
                    .collect::<Vec<_>>();
                let expected_authorization = format!("Bearer {TEST_PROVIDER_API_KEY}");
                assert_eq!(
                    authorization,
                    vec![expected_authorization.as_str()],
                    "selected provider must receive exactly one replacement credential"
                );
                assert!(
                    !request
                        .body
                        .windows(TEST_API_KEY.len())
                        .any(|window| window == TEST_API_KEY.as_bytes()),
                    "client gateway credential must not leak into the provider body"
                );
                requests.push(request);
            },
            HttpBackendEvent::ScriptExhausted { turn } => panic!("translated workflow exhausted script at {turn}"),
            HttpBackendEvent::RequestTooLarge {
                body_bytes,
                max_body_bytes,
            } => panic!("translated request body {body_bytes} exceeded backend limit {max_body_bytes}"),
            HttpBackendEvent::WebSocketUpgrade { method, path, upgrade } => {
                panic!("translated workflow attempted WebSocket ({upgrade}) on {method} {path}");
            },
            HttpBackendEvent::UnexpectedRequest { method, path } => {
                panic!("translated workflow sent unexpected request: {method} {path}");
            },
        }
    }
    match tokio::time::timeout(Duration::from_millis(250), backend.next_event()).await {
        Err(_) => {},
        Ok(None) => panic!("translated backend observation channel closed unexpectedly"),
        Ok(Some(event)) => panic!("translated workflow emitted an unexpected event after turn two: {event:?}"),
    }
    requests
}

/// Assert the provider sees a correlated Chat tool call followed by its output.
fn assert_translated_tool_turns(requests: &[CapturedHttpRequest]) {
    assert_eq!(
        requests.len(),
        2,
        "coding workflow should use exactly two provider turns"
    );
    let first: serde_json::Value = serde_json::from_slice(&requests[0].body).expect("first Chat body should parse");
    let second: serde_json::Value = serde_json::from_slice(&requests[1].body).expect("second Chat body should parse");
    assert_eq!(first["stream"], true, "first provider turn should request streaming");
    assert_eq!(second["stream"], true, "second provider turn should request streaming");
    assert!(
        first["tools"].as_array().is_some_and(|tools| tools
            .iter()
            .any(|tool| tool.pointer("/function/name") == Some(&serde_json::json!("exec_command")))),
        "Codex exec_command declaration should be translated to a Chat function tool"
    );

    let messages = second["messages"]
        .as_array()
        .expect("second Chat turn should contain messages");
    let assistant_call_index = messages.iter().position(|message| {
        message["tool_calls"]
            .as_array()
            .is_some_and(|calls| calls.iter().any(|call| call["id"] == "call_exec_1"))
    });
    assert!(
        assistant_call_index.is_some(),
        "second turn should preserve the assistant tool call ID"
    );
    let tool_output_index = messages
        .iter()
        .position(|message| message["role"] == "tool" && message["tool_call_id"] == "call_exec_1")
        .expect("second turn should correlate command output with the tool call ID");
    let assistant_call_index = assistant_call_index.expect("assistant call index was checked above");
    assert!(
        assistant_call_index < tool_output_index,
        "assistant tool call must precede its output: call={assistant_call_index}, output={tool_output_index}"
    );
    let tool_output = &messages[tool_output_index];
    assert!(
        tool_output["content"]
            .as_str()
            .is_some_and(|content| content.contains("Process exited with code 0")),
        "provider should receive successful command output after the assistant call"
    );
}

/// Validate the stable terminal summary and usage events from Codex.
///
/// Command execution is proven independently by the workspace verifier and
/// the correlated provider-facing tool output. The pinned client does not
/// consistently include its redundant `command_execution` JSONL item for a
/// successful command with no output.
fn assert_coding_codex_jsonl(stdout: &str) {
    let mut saw_summary = false;
    let mut saw_completed_turn = false;
    let mut saw_usage = false;
    for line in stdout.lines().filter(|line| !line.trim().is_empty()) {
        let event: serde_json::Value = serde_json::from_str(line).expect("Codex --json output should be JSONL");
        let item_type = event.pointer("/item/type").and_then(serde_json::Value::as_str);
        saw_summary |= event["type"] == "item.completed"
            && item_type == Some("agent_message")
            && event.pointer("/item/text").and_then(serde_json::Value::as_str) == Some("TASK_COMPLETE");
        if event["type"] == "turn.completed" {
            saw_completed_turn = true;
            saw_usage |= event
                .pointer("/usage/input_tokens")
                .and_then(serde_json::Value::as_u64)
                .is_some_and(|tokens| tokens > 0)
                && event
                    .pointer("/usage/output_tokens")
                    .and_then(serde_json::Value::as_u64)
                    .is_some_and(|tokens| tokens > 0);
        }
    }
    assert!(
        saw_summary,
        "Codex should report the exact terminal summary; stdout:\n{stdout}"
    );
    assert!(
        saw_completed_turn,
        "Codex should report a completed coding turn; stdout:\n{stdout}"
    );
    assert!(
        saw_usage,
        "Codex should receive nonzero translated usage; stdout:\n{stdout}"
    );
}

/// Validate model-independent lifecycle invariants from the live Codex turn.
fn assert_live_coding_codex_jsonl(stdout: &str) {
    let mut started_commands = Vec::new();
    let mut saw_correlated_successful_command = false;
    let mut saw_summary = false;
    let mut saw_completed_turn = false;
    let mut saw_usage = false;

    for line in stdout.lines().filter(|line| !line.trim().is_empty()) {
        let event: serde_json::Value = serde_json::from_str(line).expect("Codex --json output should be JSONL");
        let event_type = event["type"].as_str();
        let item_type = event.pointer("/item/type").and_then(serde_json::Value::as_str);
        let item_id = event
            .pointer("/item/id")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();

        if event_type == Some("item.started") && item_type == Some("command_execution") && !item_id.is_empty() {
            started_commands.push(item_id.to_owned());
        }
        if event_type == Some("item.completed") && item_type == Some("command_execution") {
            let command = event
                .pointer("/item/command")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default();
            let exit_code = event.pointer("/item/exit_code").and_then(serde_json::Value::as_i64);
            saw_correlated_successful_command |= !command.is_empty()
                && exit_code == Some(0)
                && started_commands.iter().any(|started| started == item_id);
        }
        saw_summary |= event_type == Some("item.completed")
            && item_type == Some("agent_message")
            && event
                .pointer("/item/text")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|text| !text.trim().is_empty());
        if event_type == Some("turn.completed") {
            saw_completed_turn = true;
            saw_usage |= event
                .pointer("/usage/input_tokens")
                .and_then(serde_json::Value::as_u64)
                .is_some_and(|tokens| tokens > 0)
                && event
                    .pointer("/usage/output_tokens")
                    .and_then(serde_json::Value::as_u64)
                    .is_some_and(|tokens| tokens > 0);
        }
    }

    assert!(
        saw_correlated_successful_command,
        "Codex must emit a correlated command_execution start/completion with exit code 0; stdout:\n{stdout}"
    );
    assert!(
        saw_summary,
        "Codex must emit a non-empty terminal summary; stdout:\n{stdout}"
    );
    assert!(
        saw_completed_turn,
        "Codex must report a completed live turn; stdout:\n{stdout}"
    );
    assert!(
        saw_usage,
        "Codex must receive nonzero usage from live vLLM through Praxis; stdout:\n{stdout}"
    );
}

/// Put a child in its own process group so timeout cleanup includes descendants.
#[cfg(unix)]
fn configure_isolated_process_group(command: &mut tokio::process::Command) {
    use std::os::unix::process::CommandExt as _;

    command.as_std_mut().process_group(0);
}

/// Preserve the cross-platform direct-child behavior where process groups are unavailable.
#[cfg(not(unix))]
fn configure_isolated_process_group(_command: &mut tokio::process::Command) {}

/// Wait for a child, terminate its process group on timeout, and collect both pipes.
async fn capture_child_output(mut child: tokio::process::Child, execution_timeout: Duration) -> CapturedChildOutput {
    let process_group_id = child.id();
    let mut stdout = child.stdout.take().expect("stdout should be piped");
    let mut stderr = child.stderr.take().expect("stderr should be piped");
    let mut stdout_task = tokio::spawn(async move { read_bounded_pipe(&mut stdout).await });
    let mut stderr_task = tokio::spawn(async move { read_bounded_pipe(&mut stderr).await });

    let (status, timed_out) = if let Ok(result) = tokio::time::timeout(execution_timeout, child.wait()).await {
        (result.expect("Codex process should be waitable"), false)
    } else {
        terminate_process_group(process_group_id, &mut child);
        let status = tokio::time::timeout(CHILD_CLEANUP_TIMEOUT, child.wait())
            .await
            .expect("killed child should be reaped within the cleanup timeout")
            .expect("killed child should be waitable");
        (status, true)
    };

    let stdout = collect_pipe(&mut stdout_task, process_group_id, "stdout").await;
    let stderr = collect_pipe(&mut stderr_task, process_group_id, "stderr").await;
    CapturedChildOutput {
        status,
        stderr,
        stdout,
        timed_out,
    }
}

/// Drain a child pipe while retaining only a bounded diagnostic prefix.
async fn read_bounded_pipe<R>(reader: &mut R) -> Vec<u8>
where
    R: tokio::io::AsyncRead + Unpin,
{
    let mut retained = Vec::with_capacity(MAX_CHILD_OUTPUT_BYTES);
    let mut chunk = [0_u8; 1024];
    loop {
        let read = reader
            .read(&mut chunk)
            .await
            .expect("child output pipe should be readable");
        if read == 0 {
            return retained;
        }
        let keep = read.min(MAX_CHILD_OUTPUT_BYTES.saturating_sub(retained.len()));
        retained.extend_from_slice(&chunk[..keep]);
    }
}

/// Terminate an isolated child process group, falling back to the direct child.
fn terminate_process_group(process_group_id: Option<u32>, child: &mut tokio::process::Child) {
    #[cfg(unix)]
    if let Some(id) = process_group_id {
        let id = i32::try_from(id).expect("child PID should fit in i32");
        match kill(Pid::from_raw(-id), Signal::SIGKILL) {
            Ok(()) | Err(Errno::ESRCH) => return,
            Err(error) => panic!("timed-out child process group should be killable: {error}"),
        }
    }

    child.start_kill().expect("timed-out child process should be killable");
}

/// Collect one pipe within a bound, killing inherited descendants if necessary.
async fn collect_pipe(
    task: &mut tokio::task::JoinHandle<Vec<u8>>,
    process_group_id: Option<u32>,
    name: &str,
) -> Vec<u8> {
    if let Ok(result) = tokio::time::timeout(CHILD_CLEANUP_TIMEOUT, &mut *task).await {
        return result.unwrap_or_else(|error| panic!("{name} reader should finish: {error}"));
    }

    #[cfg(unix)]
    if let Some(id) = process_group_id {
        let id = i32::try_from(id).expect("child PID should fit in i32");
        match kill(Pid::from_raw(-id), Signal::SIGKILL) {
            Ok(()) | Err(Errno::ESRCH) => {},
            Err(error) => panic!("descendant process group holding {name} should be killable: {error}"),
        }
    }

    let result = tokio::time::timeout(CHILD_CLEANUP_TIMEOUT, &mut *task).await;
    let Ok(result) = result else {
        task.abort();
        panic!("{name} reader exceeded the cleanup timeout")
    };
    result.unwrap_or_else(|error| panic!("{name} reader should finish: {error}"))
}

/// Drain queued observations and assert that every accepted request is a
/// `POST /v1/responses` carrying the synthetic credential.
async fn observe_codex_requests(backend: &mut praxis_test_utils::HttpBackendGuard) -> Vec<CapturedHttpRequest> {
    let mut requests = Vec::new();
    let first_timeout = Duration::from_secs(10);
    let followup_timeout = Duration::from_secs(5);
    let mut wait = first_timeout;

    loop {
        let event = match tokio::time::timeout(wait, backend.next_event()).await {
            Ok(Some(event)) => event,
            Ok(None) => panic!("backend observation channel should remain open"),
            Err(_) if requests.is_empty() => panic!("backend did not observe a Codex request within ten seconds"),
            Err(_) => return requests,
        };
        match event {
            HttpBackendEvent::Request(request) => {
                assert_eq!(
                    request.method, "POST",
                    "Codex should only POST to the Responses endpoint; got {request:?}"
                );
                assert!(
                    request.path.starts_with("/v1/responses"),
                    "Codex request should target /v1/responses; got path {}",
                    request.path
                );
                assert_eq!(
                    request
                        .headers
                        .get(http::header::AUTHORIZATION)
                        .map(|v| v.to_str().unwrap_or("")),
                    Some(format!("Bearer {TEST_API_KEY}").as_str()),
                    "synthetic provider credential should reach the backend"
                );
                assert_eq!(
                    request
                        .headers
                        .get(http::header::CONTENT_TYPE)
                        .map(|v| v.to_str().unwrap_or("")),
                    Some("application/json"),
                    "Codex HTTP request should declare application/json"
                );
                let body: serde_json::Value =
                    serde_json::from_slice(&request.body).expect("native Responses request should be JSON");
                assert_eq!(
                    body["stream"], true,
                    "native request should preserve Responses streaming"
                );
                assert!(
                    body.get("input").is_some(),
                    "native request should retain Responses input"
                );
                assert!(
                    body.get("messages").is_none(),
                    "native passthrough must not transform the request into Chat Completions"
                );
                requests.push(request);
                wait = followup_timeout;
            },
            HttpBackendEvent::ScriptExhausted { turn } => {
                panic!("Codex exhausted the scripted HTTP backend at turn {turn}");
            },
            HttpBackendEvent::RequestTooLarge {
                body_bytes,
                max_body_bytes,
            } => panic!("Codex request body {body_bytes} exceeded backend limit {max_body_bytes}"),
            HttpBackendEvent::WebSocketUpgrade { method, path, upgrade } => {
                panic!("Codex attempted a WebSocket upgrade ({upgrade}) on {method} {path}");
            },
            HttpBackendEvent::UnexpectedRequest { method, path } => {
                panic!("Codex attempted unexpected HTTP request: {method} {path}");
            },
        }
        if requests.len() >= 12 {
            // Cap the loop so an unexpectedly chatty Codex cannot run forever.
            return requests;
        }
    }
}

/// Drain queued observations and reject any `WebSocketUpgrade` event.
async fn assert_no_websocket_upgrades(backend: &mut praxis_test_utils::HttpBackendGuard) {
    while let Ok(Some(event)) = tokio::time::timeout(Duration::from_millis(100), backend.next_event()).await {
        if let HttpBackendEvent::WebSocketUpgrade { method, path, upgrade } = event {
            panic!("Codex attempted a WebSocket upgrade ({upgrade}) on {method} {path}");
        }
        if let HttpBackendEvent::ScriptExhausted { turn } = event {
            panic!("Codex exhausted the scripted HTTP backend at turn {turn}");
        }
    }
}

/// Drain queued observations and reject any non-`POST` request to a
/// non-`/v1/responses*` path.
async fn assert_no_unexpected_methods(backend: &mut praxis_test_utils::HttpBackendGuard) {
    while let Ok(Some(event)) = tokio::time::timeout(Duration::from_millis(100), backend.next_event()).await {
        if let HttpBackendEvent::UnexpectedRequest { method, path } = event {
            panic!("Codex attempted unexpected HTTP request: {method} {path}");
        }
        if let HttpBackendEvent::ScriptExhausted { turn } = event {
            panic!("Codex exhausted the scripted HTTP backend at turn {turn}");
        }
    }
}

/// Validate the completed output and ensure Codex attempted no tool.
fn assert_codex_jsonl(stdout: &str) {
    let mut saw_final_message = false;
    let mut saw_completed_turn = false;
    for line in stdout.lines().filter(|line| !line.trim().is_empty()) {
        let event: serde_json::Value = serde_json::from_str(line).expect("Codex --json output should be JSONL");
        let item_type = event.pointer("/item/type").and_then(serde_json::Value::as_str);
        assert!(
            item_type.is_none_or(|kind| !TOOL_ITEM_TYPES.contains(&kind)),
            "Codex attempted a tool: {event}"
        );
        saw_final_message |= event["type"] == "item.completed"
            && item_type == Some("agent_message")
            && event.pointer("/item/text").and_then(serde_json::Value::as_str) == Some("PONG");
        saw_completed_turn |= event["type"] == "turn.completed";
    }
    assert!(
        saw_final_message,
        "Codex should complete an agent message whose exact text is PONG"
    );
    assert!(saw_completed_turn, "Codex should report a completed turn");
}
