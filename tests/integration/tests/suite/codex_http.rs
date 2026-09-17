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
    ffi::OsStr,
    path::Path,
    process::Stdio,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

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
const MAX_CHILD_OUTPUT_BYTES: usize = 4 * 1024 * 1024;
/// Required output from the pinned executable.
const CODEX_VERSION: &str = "codex-cli 0.144.1";
/// Synthetic credential that must reach the test backend.
const TEST_API_KEY: &str = "praxis-http-test-key";
/// Synthetic provider credential that replaces the client credential.
const TEST_PROVIDER_API_KEY: &str = "synthetic-provider-key";
/// Fixed prompt shared by the fixture and child process.
const PROMPT: &str = "Reply with exactly PONG over HTTP. Do not call tools.";

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
    let output = run_codex(&codex_bin, observer.port(), working_dir.path(), PROMPT, "read-only").await;
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
    let output = run_codex(&codex_bin, observer.port(), workspace.path(), prompt, "workspace-write").await;
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
        vec![vec![HttpServerAction::StreamSse {
            events: vec![
                chat_sse_data(&first_chunk),
                chat_sse_data(&terminal_chunk),
                "data: [DONE]\n".to_owned(),
            ],
            inter_event_delay: Duration::from_secs(5),
        }]],
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
    let mut client = tokio::net::TcpStream::connect((std::net::Ipv4Addr::LOCALHOST, observer.port()))
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
        let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
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
    let mut upstream = tokio::net::TcpStream::connect((std::net::Ipv4Addr::LOCALHOST, upstream_port))
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
    let mut stream = tokio::net::TcpStream::connect((std::net::Ipv4Addr::LOCALHOST, proxy_port))
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
fn http_response_script() -> Vec<Vec<HttpServerAction>> {
    let turn = vec![HttpServerAction::StreamSse {
        events: vec![
            sse_event(
                "response.created",
                &serde_json::json!({
                    "type": "response.created",
                    "sequence_number": 0,
                    "response": {
                        "id": "resp_http_acceptance",
                        "object": "response",
                        "status": "in_progress",
                    }
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
                    "response": {
                        "id": "resp_http_acceptance",
                        "object": "response",
                        "status": "completed",
                        "output": [
                            {
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
                        ],
                        "usage": {
                            "input_tokens": 0,
                            "input_tokens_details": null,
                            "output_tokens": 0,
                            "output_tokens_details": null,
                            "total_tokens": 0
                        }
                    }
                }),
            ),
        ],
        inter_event_delay: Duration::ZERO,
    }];
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

/// Build two Chat Completions SSE turns: one command call and one summary.
#[expect(
    clippy::large_stack_frames,
    reason = "Chat SSE JSON values are bounded test fixtures"
)]
fn chat_coding_response_script() -> Vec<Vec<HttpServerAction>> {
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
        vec![HttpServerAction::StreamSse {
            events: vec![
                chat_sse_data(&tool_chunk),
                chat_sse_data(&tool_done),
                "data: [DONE]\n".to_owned(),
            ],
            inter_event_delay: Duration::ZERO,
        }],
        vec![HttpServerAction::StreamSse {
            events: vec![
                chat_sse_data(&summary_chunk),
                chat_sse_data(&summary_done),
                "data: [DONE]\n".to_owned(),
            ],
            inter_event_delay: Duration::ZERO,
        }],
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
async fn run_codex(codex_bin: &OsStr, proxy_port: u16, working_dir: &Path, prompt: &str, sandbox: &str) -> CodexOutput {
    let codex_home = tempfile::tempdir().expect("temporary CODEX_HOME should be created");
    let config = format!(
        r#"model = "test-model"
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
base_url = "http://127.0.0.1:{proxy_port}/v1"
wire_api = "responses"
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
        .env("NO_PROXY", "127.0.0.1,localhost")
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
