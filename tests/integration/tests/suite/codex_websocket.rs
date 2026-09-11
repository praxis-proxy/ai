// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Black-box Codex CLI acceptance test for Responses WebSocket passthrough.
//!
//! Run the pinned compatibility test locally with:
//!
//! ```console
//! PRAXIS_TEST_CODEX_BIN=/absolute/path/to/codex \
//!   cargo test -p praxis-tests-integration --test suite \
//!   codex_websocket::pinned_codex_uses_responses_websocket_through_full_flow -- --exact
//! ```
//!
//! The executable must report the version in [`CODEX_VERSION`]. To update the
//! pin, update that constant, the fixture filename and metadata, the checksum
//! manifest, and every version, archive, cache path/key, and version assertion
//! in `.github/workflows/integration.yaml`. Recapture and sanitize the fixture
//! when the client's wire protocol changes, then run this test with the new
//! verified binary before committing the coordinated update.

use std::{collections::HashMap, ffi::OsStr, process::Stdio, time::Duration};

use praxis_test_utils::{
    CapturedWsMessage, TempSqlite, WsBackendEvent, WsServerAction, capture_child_output,
    configure_isolated_process_group, example_config_path, free_port, patch_yaml, start_proxy,
    start_scripted_websocket_backend_turns,
};
use serde::Deserialize;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Required output from the pinned executable.
const CODEX_VERSION: &str = "codex-cli 0.144.1";
/// Fixed prompt shared by the fixture and child process.
const PROMPT: &str = "Reply with exactly PONG. Do not call tools.";
/// Synthetic credential that must reach the test backend.
const TEST_API_KEY: &str = "praxis-websocket-test-key";
/// Codex output item types that would indicate an attempted tool call.
const TOOL_ITEM_TYPES: &[&str] = &["command_execution", "file_change", "mcp_tool_call", "web_search"];

/// Prove the pinned Codex client completes an offline turn over WebSocket.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pinned_codex_uses_responses_websocket_through_full_flow() {
    let Some(codex_bin) = std::env::var_os("PRAXIS_TEST_CODEX_BIN") else {
        eprintln!("skipping pinned Codex acceptance test; PRAXIS_TEST_CODEX_BIN is unset");
        return;
    };
    assert_pinned_codex_version(&codex_bin).await;

    let fixture = load_fixture();
    assert_eq!(
        fixture.metadata.fixture_version, 1,
        "the fixture schema version should match the harness"
    );
    assert_eq!(
        fixture.metadata.codex_cli_version, "0.144.1",
        "the fixture should identify the pinned Codex version"
    );
    assert_eq!(
        fixture.client.kind, "text",
        "the pinned request fixture should use a text frame"
    );
    assert_eq!(
        fixture.client.r#type, "response.create",
        "the pinned request fixture should describe a response.create frame"
    );
    assert_eq!(
        fixture.client.expected_prompt, PROMPT,
        "the fixture prompt should match the child-process prompt"
    );

    let mut script: Vec<_> = fixture
        .server_messages
        .iter()
        .map(|message| {
            assert_eq!(message.kind, "text", "fixture supports text frames only");
            WsServerAction::Text(serde_json::to_string(&message.payload).unwrap())
        })
        .collect();
    script.push(WsServerAction::Close {
        code: 1000,
        reason: "fixture complete".to_owned(),
    });
    let prewarm = vec![
        WsServerAction::Text(
            r#"{"type":"response.created","response":{"id":"resp_praxis_prewarm"}}"#.to_owned(),
        ),
        WsServerAction::Text(
            r#"{"type":"response.completed","response":{"id":"resp_praxis_prewarm","usage":{"input_tokens":0,"input_tokens_details":null,"output_tokens":0,"output_tokens_details":null,"total_tokens":0}}}"#
                .to_owned(),
        ),
    ];
    let mut backend = start_scripted_websocket_backend_turns(vec![prewarm, script]).await;
    let proxy_port = free_port();
    let db = TempSqlite::new("codex_websocket");
    let yaml = std::fs::read_to_string(example_config_path("openai/responses/full-flow.yaml"))
        .expect("full-flow example should exist");
    let patched = patch_yaml(
        &yaml.replace("sqlite://responses.db?mode=rwc", db.url()),
        proxy_port,
        &HashMap::from([("127.0.0.1:3001", backend.port())]),
    );
    let config = praxis_core::config::Config::from_yaml(&patched).expect("patched config should parse");
    let _proxy = start_proxy(&config);

    let output = run_codex(&codex_bin, proxy_port).await;
    assert!(
        output.status.success(),
        "Codex failed with status {status:?}\nstdout:\n{stdout}\nstderr:\n{stderr}",
        status = output.status.code(),
        stdout = output.stdout,
        stderr = output.stderr
    );
    assert_codex_jsonl(&output.stdout);

    let requests = observe_codex_requests(&mut backend).await;
    assert_eq!(requests.len(), 2, "Codex should send prewarm and turn requests");
    assert_eq!(
        requests[0]["type"], "response.create",
        "the prewarm frame should create a response"
    );
    assert_eq!(
        requests[0]["generate"], false,
        "the first frame should be Codex's non-generating prewarm request"
    );
    assert_eq!(
        requests[1]["type"], "response.create",
        "the turn frame should create a response"
    );
    assert_ne!(
        requests[1]["generate"], false,
        "the turn frame should request generation"
    );
    assert!(
        value_contains_exact_string(&requests[1], PROMPT),
        "the generating response.create frame should contain the exact fixed prompt"
    );
    assert_no_unexpected_http(&mut backend).await;
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

/// Typed subset of the deterministic protocol fixture.
#[derive(Debug, Deserialize)]
struct CodexWsFixture {
    /// Expected client request facts.
    client: FixtureClient,
    /// Fixture provenance and sanitization metadata.
    metadata: FixtureMetadata,
    /// Ordered server messages.
    server_messages: Vec<FixtureServerMessage>,
}

/// Captured child-process result with decoded output.
struct CodexOutput {
    /// Exit status.
    status: std::process::ExitStatus,
    /// UTF-8-lossy standard error.
    stderr: String,
    /// UTF-8-lossy standard output.
    stdout: String,
}

/// Expected facts in the Codex request frame.
#[derive(Debug, Deserialize)]
struct FixtureClient {
    /// Exact synthetic prompt.
    expected_prompt: String,
    /// WebSocket frame kind.
    kind: String,
    /// Responses request type.
    r#type: String,
}

/// Metadata that makes fixture pin updates reviewable.
#[derive(Debug, Deserialize)]
struct FixtureMetadata {
    /// Pinned Codex version.
    codex_cli_version: String,
    /// Fixture schema revision.
    fixture_version: u64,
}

/// One server-side fixture frame.
#[derive(Debug, Deserialize)]
struct FixtureServerMessage {
    /// WebSocket frame kind.
    kind: String,
    /// Responses event JSON.
    payload: serde_json::Value,
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
async fn run_codex(codex_bin: &OsStr, proxy_port: u16) -> CodexOutput {
    let codex_home = tempfile::tempdir().expect("temporary CODEX_HOME should be created");
    let working_dir = tempfile::tempdir().expect("temporary working directory should be created");
    let config = format!(
        r#"model = "test-model"
model_provider = "praxis"

[model_providers.praxis]
name = "Praxis test gateway"
base_url = "http://127.0.0.1:{proxy_port}/v1"
wire_api = "responses"
supports_websockets = true
env_key = "PRAXIS_TEST_API_KEY"
"#
    );
    std::fs::write(codex_home.path().join("config.toml"), config).expect("test config should be written");

    let mut child = tokio::process::Command::new(codex_bin);
    child
        .arg("exec")
        .arg("--ephemeral")
        .arg("--strict-config")
        .arg("--skip-git-repo-check")
        .arg("--sandbox")
        .arg("read-only")
        .arg("--json")
        .arg(PROMPT)
        .current_dir(working_dir.path())
        .env_clear()
        .env("CODEX_HOME", codex_home.path())
        .env("HOME", codex_home.path())
        .env("PATH", "/usr/bin:/bin:/usr/local/bin")
        .env("PRAXIS_TEST_API_KEY", TEST_API_KEY)
        .env("RUST_LOG", "error")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    configure_isolated_process_group(&mut child);
    let child = child.spawn().expect("pinned Codex should start");
    let captured = capture_child_output(child, Duration::from_secs(30)).await;

    let output = CodexOutput {
        status: captured.status,
        stderr: String::from_utf8_lossy(&captured.stderr).into_owned(),
        stdout: String::from_utf8_lossy(&captured.stdout).into_owned(),
    };
    assert!(
        !captured.timed_out,
        "Codex process exceeded 30-second acceptance-test timeout\nstdout:\n{}\nstderr:\n{}",
        output.stdout, output.stderr
    );
    output
}

/// Load and deserialize the committed deterministic fixture.
fn load_fixture() -> CodexWsFixture {
    let path = format!(
        "{}/fixtures/openai/responses/websocket-codex-0.144.1.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let bytes = std::fs::read(path).expect("Codex WebSocket fixture should exist");
    serde_json::from_slice(&bytes).expect("Codex WebSocket fixture should parse")
}

/// Observe one handshake and the prewarm plus generating request frames.
async fn observe_codex_requests(backend: &mut praxis_test_utils::WsBackendGuard) -> Vec<serde_json::Value> {
    let mut saw_handshake = false;
    let mut requests = Vec::new();
    loop {
        let event = tokio::time::timeout(Duration::from_secs(5), backend.next_event())
            .await
            .expect("backend observation should not time out")
            .expect("backend observation channel should remain open");
        match event {
            WsBackendEvent::Handshake { headers, path, .. } => {
                assert!(!saw_handshake, "acceptance turn should use one WebSocket connection");
                saw_handshake = true;
                assert_eq!(
                    path, "/v1/responses",
                    "Codex should use the Responses WebSocket endpoint"
                );
                assert_eq!(
                    headers.get(http::header::AUTHORIZATION).unwrap(),
                    "Bearer praxis-websocket-test-key",
                    "synthetic provider credential should reach the backend"
                );
            },
            WsBackendEvent::ClientMessage(CapturedWsMessage::Text(text)) => {
                assert!(saw_handshake, "request frame must follow the handshake");
                requests.push(serde_json::from_str(&text).expect("Codex request frame should be JSON"));
                if requests.len() == 2 {
                    return requests;
                }
            },
            WsBackendEvent::UnexpectedHttpRequest { method, path } => {
                panic!("Codex attempted unexpected HTTP fallback: {method} {path}");
            },
            WsBackendEvent::ClientMessage(_) => {},
        }
    }
}

/// Drain queued observations and reject any HTTP fallback request.
async fn assert_no_unexpected_http(backend: &mut praxis_test_utils::WsBackendGuard) {
    while let Ok(Some(event)) = tokio::time::timeout(Duration::from_millis(100), backend.next_event()).await {
        if let WsBackendEvent::UnexpectedHttpRequest { method, path } = event {
            panic!("Codex attempted unexpected HTTP fallback: {method} {path}");
        }
    }
}

/// Search a JSON value recursively for an exact string.
fn value_contains_exact_string(value: &serde_json::Value, expected: &str) -> bool {
    match value {
        serde_json::Value::String(actual) => actual == expected,
        serde_json::Value::Array(values) => values.iter().any(|value| value_contains_exact_string(value, expected)),
        serde_json::Value::Object(values) => values
            .values()
            .any(|value| value_contains_exact_string(value, expected)),
        serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::Number(_) => false,
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
