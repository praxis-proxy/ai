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
    CapturedHttpRequest, HttpBackendEvent, HttpServerAction, build_pipeline, example_config_path, free_port, http_send,
    parse_body, parse_status, patch_yaml, start_proxy, start_scripted_http_backend, start_scripted_http_backend_turns,
    start_uri_echo_backend,
};
use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};

use super::harness::{CodexCompactionWorkspace, TempWorkspace};

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

// -----------------------------------------------------------------------------
// Compaction acceptance constants
// -----------------------------------------------------------------------------
//
// Codex 0.144.1 against a provider named `praxis` does not qualify for remote
// compaction (`supports_remote_compaction()` is false unless the provider is
// `openai`/Azure), and the `token_budget` window-reset feature ships disabled by
// default. Auto-compaction therefore takes the local inline path
// (`run_inline_auto_compact_task`), which POSTs the summarization prompt to
// `/v1/responses` through Praxis and then replays a `SUMMARY_PREFIX`-headed
// compacted history on the following turn. Neither event surfaces in the
// `codex exec --json` stream, so the definite compaction signal is captured from
// the client->upstream wire by [`HttpTransportObserver`], not inferred from token
// counts or a successful final answer.

/// First line of Codex's `SUMMARIZATION_PROMPT`. Its presence in client->upstream
/// traffic proves Codex issued an inline auto-compaction request through Praxis.
/// The fragment is pure ASCII with no JSON-escaped characters, so it appears
/// verbatim inside the serialized Responses request body.
const CODEX_SUMMARIZATION_PROMPT_NEEDLE: &[u8] = b"You are performing a CONTEXT CHECKPOINT COMPACTION.";
/// First sentence of Codex's `SUMMARY_PREFIX`. It exists only in post-compaction
/// history, so seeing it on the wire proves a request carrying the compacted
/// summary traversed Praxis after compaction.
const CODEX_SUMMARY_PREFIX_NEEDLE: &[u8] =
    b"Another language model started to solve this problem and produced a summary of its thinking process.";

/// Advertised context window for the compaction run, kept under the Codex GPU
/// job's vLLM `--max-model-len 16384` so real requests are never rejected.
///
/// Tunable: the effective auto-compaction trigger is
/// `min(model_auto_compact_token_limit, 0.9 * model_context_window)` in the
/// default `Total` scope. May need one GPU-job pass to land mid-task compaction
/// against live Qwen3-8B.
const CODEX_COMPACTION_CONTEXT_WINDOW: i64 = 16_000;
/// Explicit auto-compaction token limit, set low enough that accumulated tool
/// results cross it after the early marker read but well before the backend
/// window, forcing compaction mid-task rather than rejecting an oversized turn.
const CODEX_COMPACTION_AUTO_COMPACT_LIMIT: i64 = 8_000;
/// Number of ballast chapters seeded to grow context across accepted turns.
const CODEX_COMPACTION_BALLAST_CHAPTERS: usize = 10;
/// Approximate bytes per ballast chapter (~1.5k tokens of natural-language text).
const CODEX_COMPACTION_BALLAST_BYTES: usize = 6_000;

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
            compaction_limits: None,
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
            compaction_limits: None,
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

/// Prove a pinned Codex session crosses its own context limit mid-task, performs
/// the client's inline auto-compaction through Praxis, and still finishes the
/// coding task against live vLLM.
///
/// Context grows across many accepted turns (the marker is read first, then one
/// ballast chapter per turn) until accumulated usage crosses the lowered
/// `model_auto_compact_token_limit`. Codex then runs `run_inline_auto_compact_task`
/// — POSTing its `SUMMARIZATION_PROMPT` to `/v1/responses` and replaying a
/// `SUMMARY_PREFIX`-headed compacted history — both of which
/// [`HttpTransportObserver`] captures on the client->upstream wire. The run never
/// calls an SDK compact endpoint, injects a canned summary, or enables a Praxis
/// compaction filter: compaction is driven entirely by the pinned client.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pinned_codex_compaction_crosses_context_window_over_http() {
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
            .unwrap_or_else(|| panic!("{LIVE_DATABASE_URL_ENV} must be set for the compaction vLLM acceptance test")),
    );
    let config = praxis_core::config::Config::from_yaml(&patched).expect("live compaction Codex config should parse");

    run_live_codex_compaction_workflow(&live, proxy_port, config).await;
}

/// Drive the common pinned-Codex coding task through one live Praxis pipeline.
async fn run_live_codex_coding_workflow(live: &CodexLiveConfig, proxy_port: u16, config: praxis_core::config::Config) {
    let workspace = TempWorkspace::new().expect("temporary coding workspace should be created");
    let _proxy = start_proxy(&config);
    let observer = HttpTransportObserver::start_on(proxy_port, live.listen_address).await;

    if let Some(namespace) = &live.netns {
        verify_egress_isolation(namespace);
    }

    let prompt = r#"Use exec_command immediately to run exactly this single command:
python3 -c 'import json, pathlib; data = json.load(open("input.json")); pathlib.Path("result.txt").write_text(data["expected_content"])' && ./verify.sh
Do not run a different command and do not answer before it succeeds. Then summarize what changed. /no_think"#;
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
            compaction_limits: None,
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

    observer.assert_http_only();
    assert_live_coding_codex_jsonl(&output.stdout);
    workspace.assert_successful_completion();
}

/// Drive a pinned-Codex coding task whose growing context forces the client's own
/// inline auto-compaction mid-run, then assert compaction traversed Praxis.
async fn run_live_codex_compaction_workflow(
    live: &CodexLiveConfig,
    proxy_port: u16,
    config: praxis_core::config::Config,
) {
    let workspace = CodexCompactionWorkspace::new().expect("temporary compaction workspace should be created");
    let ballast = workspace
        .seed_context_ballast(CODEX_COMPACTION_BALLAST_CHAPTERS, CODEX_COMPACTION_BALLAST_BYTES)
        .expect("ballast chapters should be seeded");
    let _proxy = start_proxy(&config);
    let observer =
        HttpTransportObserver::start_compaction_probe(proxy_port, live.listen_address, workspace.path().to_path_buf())
            .await;

    if let Some(namespace) = &live.netns {
        verify_egress_isolation(namespace);
    }

    let chapter_list = ballast.join(", ");
    let prompt = format!(
        r#"Work through these steps strictly in order, one shell command per step, and never batch steps together:
1. Run exactly: cat secret.txt && rm -f secret.txt  — memorize the token it prints verbatim. The file is deleted immediately and cannot be read again; never cat secret.txt a second time for any reason.
2. Read every ballast chapter one at a time (a separate cat command per file, no more than one file per command), in this exact order: {chapter_list}. After each file, briefly acknowledge it before reading the next.
3. Only after all chapters are read, write the memorized token (and nothing else) into result.txt with exactly: printf '%s' '<TOKEN>' > result.txt  — substitute the token you memorized in step 1. Do not read any other file to recover it.
4. Run ./verify.sh and confirm it exits 0.
5. Summarize what you changed. /no_think"#
    );
    let proxy_base_url = format!("http://{}:{}", live.listen_address, observer.port());
    let no_proxy = format!("127.0.0.1,localhost,{}", live.listen_address);
    let output = run_codex(
        &live.codex_bin,
        CodexRunOptions {
            proxy_base_url: &proxy_base_url,
            model: &live.model,
            working_dir: workspace.path(),
            prompt: &prompt,
            sandbox: "danger-full-access",
            execution_timeout: LIVE_CHILD_TIMEOUT,
            no_proxy: &no_proxy,
            netns: live.netns.as_deref(),
            compaction_limits: Some(CodexCompactionLimits {
                context_window: CODEX_COMPACTION_CONTEXT_WINDOW,
                auto_compact_token_limit: CODEX_COMPACTION_AUTO_COMPACT_LIMIT,
            }),
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

    observer.assert_http_only();
    observer.assert_compaction_traversed_praxis();
    observer.assert_compaction_preceded_result_write();
    assert_live_compaction_codex_jsonl(&output.stdout, CODEX_COMPACTION_BALLAST_CHAPTERS);
    workspace.assert_successful_completion();
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
    let request = next_translated_request(&mut backend).await;
    assert_eq!(request.path, "/v1/chat/completions");

    tokio::time::timeout(Duration::from_secs(2), client)
        .await
        .expect("client should receive a translated delta before the delayed terminal Chat chunk")
        .expect("client reader task should finish");
}

/// Prove a bodyless probe survives the translated chain.
///
/// `GET /v1/models` carries no body, and an empty body classifies as non-JSON.
/// The example leads with `openai_responses_request`, which resolves the
/// operation from the request head, so the probe must route through to the
/// provider rather than being rejected with 400 "request body is not JSON"
/// before it reaches the router.
#[test]
fn translated_chain_passes_bodyless_models_probe() {
    let backend = start_uri_echo_backend();
    let proxy_port = free_port();
    let yaml = std::fs::read_to_string(example_config_path("openai/responses/codex-http-chat-translation.yaml"))
        .expect("Codex translated-provider example should exist");
    let patched = patch_yaml(&yaml, proxy_port, &HashMap::from([("127.0.0.1:3001", backend.port())]));
    let config = praxis_core::config::Config::from_yaml(&patched).expect("patched config should parse");
    let proxy = start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        "GET /v1/models HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
    );

    assert_eq!(
        parse_status(&raw),
        200,
        "bodyless model-discovery probe should return 200: {raw}"
    );
    assert_eq!(
        parse_body(&raw),
        "/v1/models",
        "model-discovery path must reach the provider unchanged"
    );
}

/// The translated chain classifies OpenAI operations from the request head.
///
/// `openai_responses_request` resolves Responses operations only, so the
/// non-Responses traffic this chain's catch-all route also forwards leaves it
/// with no operation identity at all. `ai_operation` resolves one for every
/// OpenAI operation, from the head alone, which is what the body classifier
/// this chain used to lead with could only do after reading a body.
///
/// Pinned at the pipeline rather than on the wire, because the facts that
/// ownership produces are internal: the published application protocol and
/// operation ID, and the protocol-shaped error formatter keyed off them.
/// Nothing on this chain's own routes exercises the formatter either — every
/// route runs inside the `iterative_request_router` step, which answers an
/// upstream transport failure with its own 502 rather than through the
/// `fail_to_proxy` path that consults it. So the assertion is on chain
/// composition: nothing else here classifies non-Responses OpenAI traffic,
/// and removing the filter silently drops that.
#[test]
fn translated_chain_owns_the_openai_protocol_decision() {
    let yaml = std::fs::read_to_string(example_config_path("openai/responses/codex-http-chat-translation.yaml"))
        .expect("Codex translated-provider example should exist");
    let patched = patch_yaml(&yaml, free_port(), &HashMap::from([("127.0.0.1:3001", 19953)]));
    let config = praxis_core::config::Config::from_yaml(&patched).expect("patched config should parse");

    assert!(
        build_pipeline(&config).contains_filter("ai_operation"),
        "the chain must keep a request-head operation classifier so the \
         non-Responses traffic it forwards still resolves to an OpenAI \
         protocol"
    );
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
    /// Optional advertised context window and auto-compaction token limit.
    ///
    /// When `Some`, both values are written as top-level `config.toml` keys so the
    /// pinned client crosses its own compaction threshold mid-task. `None` keeps
    /// the default (effectively unlimited) window used by the non-compaction runs.
    compaction_limits: Option<CodexCompactionLimits>,
}

/// Context-window knobs that force the pinned client to auto-compact.
#[derive(Clone, Copy)]
struct CodexCompactionLimits {
    /// Advertised `model_context_window` (`config.toml`).
    context_window: i64,
    /// Explicit `model_auto_compact_token_limit` (`config.toml`).
    auto_compact_token_limit: i64,
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
    /// Whether client->upstream traffic carried the inline-compaction summarization prompt.
    summarization_request_seen: Arc<AtomicBool>,
    /// Whether client->upstream traffic carried the post-compaction summary prefix.
    summary_prefix_seen: Arc<AtomicBool>,
    /// At the instant the summarization request was first seen on the wire,
    /// whether `secret.txt` was already gone from the compaction workspace.
    secret_gone_at_compaction: Arc<AtomicBool>,
    /// At that same instant, whether `result.txt` was still empty (the marker not
    /// yet written). Codex drains all outstanding tool calls before it decides to
    /// compact, so an empty `result.txt` here proves the write had not run — i.e.
    /// the result write strictly follows compaction even if vLLM batched tool calls.
    result_empty_at_compaction: Arc<AtomicBool>,
}

impl HttpTransportObserver {
    /// Bind a front-door observer that forwards all connections to Praxis.
    async fn start(upstream_port: u16) -> Self {
        Self::start_inner(upstream_port, IpAddr::V4(Ipv4Addr::LOCALHOST), None).await
    }

    /// Bind the observer on an explicit address, including a host-side veth.
    async fn start_on(upstream_port: u16, listen_address: IpAddr) -> Self {
        Self::start_inner(upstream_port, listen_address, None).await
    }

    /// Bind the observer and, when it first sees Codex's summarization request on
    /// the wire, snapshot the compaction workspace's filesystem state (secret
    /// deleted, result still empty) BEFORE forwarding that request upstream.
    async fn start_compaction_probe(upstream_port: u16, listen_address: IpAddr, workspace: PathBuf) -> Self {
        Self::start_inner(upstream_port, listen_address, Some(workspace)).await
    }

    /// Shared constructor; `workspace` is `Some` only for the compaction probe.
    async fn start_inner(upstream_port: u16, listen_address: IpAddr, workspace: Option<PathBuf>) -> Self {
        let listener = tokio::net::TcpListener::bind((listen_address, 0))
            .await
            .expect("transport observer should bind");
        let port = listener
            .local_addr()
            .expect("transport observer should have an address")
            .port();
        let websocket_attempted = Arc::new(AtomicBool::new(false));
        let summarization_request_seen = Arc::new(AtomicBool::new(false));
        let summary_prefix_seen = Arc::new(AtomicBool::new(false));
        let secret_gone_at_compaction = Arc::new(AtomicBool::new(false));
        let result_empty_at_compaction = Arc::new(AtomicBool::new(false));
        let signals = ObservedConnectionSignals {
            websocket_attempted: Arc::clone(&websocket_attempted),
            summarization_request_seen: Arc::clone(&summarization_request_seen),
            summary_prefix_seen: Arc::clone(&summary_prefix_seen),
            secret_gone_at_compaction: Arc::clone(&secret_gone_at_compaction),
            result_empty_at_compaction: Arc::clone(&result_empty_at_compaction),
            compaction_workspace: workspace.map(Arc::<Path>::from),
        };
        let (shutdown_tx, mut shutdown_rx) = tokio::sync::oneshot::channel();
        let handle = tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = &mut shutdown_rx => break,
                    accepted = listener.accept() => {
                        let Ok((client, _peer)) = accepted else {
                            break;
                        };
                        tokio::spawn(forward_observed_connection(client, upstream_port, signals.clone()));
                    },
                }
            }
        });
        Self {
            handle,
            port,
            shutdown: Some(shutdown_tx),
            websocket_attempted,
            summarization_request_seen,
            summary_prefix_seen,
            secret_gone_at_compaction,
            result_empty_at_compaction,
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

    /// Whether Codex POSTed the inline-compaction summarization prompt to Praxis.
    fn summarization_request_seen(&self) -> bool {
        self.summarization_request_seen.load(Ordering::SeqCst)
    }

    /// Whether a post-compaction request carrying the summary prefix traversed Praxis.
    fn summary_prefix_seen(&self) -> bool {
        self.summary_prefix_seen.load(Ordering::SeqCst)
    }

    /// Assert the wire evidence that Codex compacted and then kept working through Praxis.
    ///
    /// The summarization prompt proves Codex itself initiated compaction (not the
    /// test, the SDK compact endpoint, or a Praxis filter); the summary prefix
    /// proves a request carrying the compacted history then traversed Praxis.
    fn assert_compaction_traversed_praxis(&self) {
        assert!(
            self.summarization_request_seen(),
            "Codex should POST its inline-compaction summarization prompt through Praxis; \
             no CONTEXT CHECKPOINT COMPACTION request was observed on the wire"
        );
        assert!(
            self.summary_prefix_seen(),
            "a post-compaction request carrying Codex's compacted summary prefix should traverse \
             Praxis; the compacted history was never replayed through the observer"
        );
    }

    /// Assert the compaction boundary preceded the result write on the wire.
    ///
    /// When the observer first saw Codex's summarization request it snapshotted the
    /// workspace: `secret.txt` was already deleted (so no post-compaction reread
    /// could relaunch a lost marker) and `result.txt` was still empty. Because
    /// Codex drains every outstanding tool call before deciding to compact, an empty
    /// `result.txt` at that instant proves the mandated `printf` write had NOT run
    /// yet — the marker therefore crossed compaction in the client's live context,
    /// not via a batched pre-compaction write. This is the wire-ordering guarantee
    /// the JSONL command order alone cannot establish.
    fn assert_compaction_preceded_result_write(&self) {
        assert!(
            self.secret_gone_at_compaction.load(Ordering::SeqCst),
            "secret.txt must already be deleted when Codex's summarization request crosses the wire, \
             so a post-compaction reread cannot relaunder the marker; the observer saw the summarization \
             request with secret.txt still present (or never saw it)"
        );
        assert!(
            self.result_empty_at_compaction.load(Ordering::SeqCst),
            "result.txt must still be empty when Codex's summarization request crosses the wire, proving \
             the marker write happens AFTER compaction (Codex drains batched tool calls before compacting); \
             the observer saw a non-empty result.txt at the compaction boundary (or never saw the request)"
        );
    }
}

/// Shared atomic signals updated while forwarding one observed connection.
#[derive(Clone)]
struct ObservedConnectionSignals {
    /// Set when an opening request declares a WebSocket upgrade.
    websocket_attempted: Arc<AtomicBool>,
    /// Set when client->upstream bytes contain the summarization prompt needle.
    summarization_request_seen: Arc<AtomicBool>,
    /// Set when client->upstream bytes contain the summary prefix needle.
    summary_prefix_seen: Arc<AtomicBool>,
    /// Snapshot of `secret.txt`-is-gone taken when the summarization needle is first seen.
    secret_gone_at_compaction: Arc<AtomicBool>,
    /// Snapshot of `result.txt`-is-empty taken when the summarization needle is first seen.
    result_empty_at_compaction: Arc<AtomicBool>,
    /// Compaction workspace to probe at that instant; `None` for non-compaction observers.
    compaction_workspace: Option<Arc<Path>>,
}

impl Drop for HttpTransportObserver {
    fn drop(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _sent = shutdown.send(());
        }
        self.handle.abort();
    }
}

/// Longest compaction needle, bounding the client->upstream sliding-window carry.
const MAX_COMPACTION_NEEDLE_LEN: usize = CODEX_SUMMARY_PREFIX_NEEDLE.len();

/// Inspect one connection's opening request and client->upstream bytes, then
/// forward both directions unchanged.
///
/// The opening head preserves WebSocket-upgrade detection. The client->upstream
/// direction is scanned for the compaction needles with a bounded sliding window
/// so no request body is buffered in full; the upstream->client direction streams
/// verbatim so SSE responses are never held back.
async fn forward_observed_connection(
    client: tokio::net::TcpStream,
    upstream_port: u16,
    signals: ObservedConnectionSignals,
) {
    let upstream = tokio::net::TcpStream::connect((Ipv4Addr::LOCALHOST, upstream_port))
        .await
        .expect("transport observer should connect to Praxis");
    let (mut client_rd, mut client_wr) = client.into_split();
    let (mut upstream_rd, mut upstream_wr) = upstream.into_split();

    let mut opening = Vec::with_capacity(4096);
    // Heap-allocated so this test observer's forwarding future stays well under
    // the workspace `large_stack_frames` threshold; inline arrays here would be
    // embedded twice (async block state plus the joined future).
    let mut chunk = vec![0_u8; 4096];
    while opening.len() < 16_384 && !opening.windows(4).any(|window| window == b"\r\n\r\n") {
        let read = match client_rd.read(&mut chunk).await {
            Ok(0) | Err(_) => break,
            Ok(read) => read,
        };
        opening.extend_from_slice(&chunk[..read]);
    }
    if opening.is_empty() {
        return;
    }
    if request_head_has_websocket_upgrade(&opening) {
        signals.websocket_attempted.store(true, Ordering::SeqCst);
    }
    scan_compaction_needles(&opening, &signals);
    if upstream_wr.write_all(&opening).await.is_err() {
        return;
    }

    // Seed the sliding window with the opening tail so a needle straddling the
    // head/body boundary is still detected.
    let carry_start = opening
        .len()
        .saturating_sub(MAX_COMPACTION_NEEDLE_LEN.saturating_sub(1));
    let carry_seed = opening[carry_start..].to_vec();
    let scan_signals = signals.clone();
    let client_to_upstream = async move {
        let mut carry = carry_seed;
        let mut buffer = vec![0_u8; 8192];
        loop {
            let read = match client_rd.read(&mut buffer).await {
                Ok(0) | Err(_) => break,
                Ok(read) => read,
            };
            let mut window = carry;
            window.extend_from_slice(&buffer[..read]);
            scan_compaction_needles(&window, &scan_signals);
            let keep = window.len().saturating_sub(MAX_COMPACTION_NEEDLE_LEN.saturating_sub(1));
            carry = window.split_off(keep);
            if upstream_wr.write_all(&buffer[..read]).await.is_err() {
                break;
            }
        }
        let _shutdown = upstream_wr.shutdown().await;
    };
    let upstream_to_client = async move {
        let _forwarded = tokio::io::copy(&mut upstream_rd, &mut client_wr).await;
        let _shutdown = client_wr.shutdown().await;
    };
    tokio::join!(client_to_upstream, upstream_to_client);
}

/// Set each compaction signal whose needle appears in `haystack`.
///
/// The first time the summarization needle is seen, snapshot the compaction
/// workspace (secret deleted, result still empty) BEFORE the caller forwards the
/// request upstream. `compare_exchange` guarantees exactly one snapshot even if
/// two connections race.
fn scan_compaction_needles(haystack: &[u8], signals: &ObservedConnectionSignals) {
    if !signals.summarization_request_seen.load(Ordering::SeqCst)
        && contains_subslice(haystack, CODEX_SUMMARIZATION_PROMPT_NEEDLE)
        && signals
            .summarization_request_seen
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        && let Some(workspace) = &signals.compaction_workspace
    {
        let (secret_gone, result_empty) = compaction_boundary_filesystem_state(workspace);
        signals.secret_gone_at_compaction.store(secret_gone, Ordering::SeqCst);
        signals.result_empty_at_compaction.store(result_empty, Ordering::SeqCst);
    }
    if !signals.summary_prefix_seen.load(Ordering::SeqCst) && contains_subslice(haystack, CODEX_SUMMARY_PREFIX_NEEDLE) {
        signals.summary_prefix_seen.store(true, Ordering::SeqCst);
    }
}

/// Snapshot the compaction workspace: `(secret.txt is gone, result.txt is empty)`.
///
/// `result.txt` seeded empty counts as empty; a missing `result.txt` counts as
/// NOT empty (fail closed — an absent file must not be read as "write pending").
fn compaction_boundary_filesystem_state(workspace: &Path) -> (bool, bool) {
    let secret_gone = !workspace.join("secret.txt").exists();
    let result_empty =
        std::fs::read_to_string(workspace.join("result.txt")).is_ok_and(|content| content.trim().is_empty());
    (secret_gone, result_empty)
}

/// Whether `needle` occurs contiguously within `haystack`.
fn contains_subslice(haystack: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty()
        && haystack.len() >= needle.len()
        && haystack.windows(needle.len()).any(|window| window == needle)
}

#[test]
fn scan_compaction_needles_sets_each_flag_once_when_its_needle_appears() {
    let signals = ObservedConnectionSignals {
        websocket_attempted: Arc::new(AtomicBool::new(false)),
        summarization_request_seen: Arc::new(AtomicBool::new(false)),
        summary_prefix_seen: Arc::new(AtomicBool::new(false)),
        secret_gone_at_compaction: Arc::new(AtomicBool::new(false)),
        result_empty_at_compaction: Arc::new(AtomicBool::new(false)),
        compaction_workspace: None,
    };

    // Ordinary request bytes leave both compaction flags clear.
    scan_compaction_needles(b"POST /v1/responses HTTP/1.1\r\n\r\n{\"input\":\"hi\"}", &signals);
    assert!(!signals.summarization_request_seen.load(Ordering::SeqCst));
    assert!(!signals.summary_prefix_seen.load(Ordering::SeqCst));

    // The summarization prompt sets only its own flag.
    let mut summarization = b"{\"instructions\":\"".to_vec();
    summarization.extend_from_slice(CODEX_SUMMARIZATION_PROMPT_NEEDLE);
    scan_compaction_needles(&summarization, &signals);
    assert!(signals.summarization_request_seen.load(Ordering::SeqCst));
    assert!(!signals.summary_prefix_seen.load(Ordering::SeqCst));

    // The replayed summary prefix sets its flag too.
    scan_compaction_needles(CODEX_SUMMARY_PREFIX_NEEDLE, &signals);
    assert!(signals.summary_prefix_seen.load(Ordering::SeqCst));
}

#[test]
fn contains_subslice_matches_only_contiguous_occurrences() {
    assert!(contains_subslice(b"abcdef", b"cde"));
    assert!(contains_subslice(b"abc", b"abc"));
    assert!(!contains_subslice(b"abc", b"abcd"));
    assert!(!contains_subslice(b"a_b_c", b"abc"));
    assert!(!contains_subslice(b"anything", b""));
}

#[test]
fn compaction_boundary_state_reports_secret_and_result_presence() {
    let dir = tempfile::TempDir::new().expect("tempdir");
    // Secret already deleted (step 1 done) and result still seeded empty: the
    // boundary we require when the summarization request crosses the wire.
    std::fs::write(dir.path().join("result.txt"), "").expect("seed result");
    assert_eq!(compaction_boundary_filesystem_state(dir.path()), (true, true));

    // A result written before compaction (a batched pre-compaction write) is not empty.
    std::fs::write(dir.path().join("result.txt"), "MARKER").expect("write result");
    assert_eq!(compaction_boundary_filesystem_state(dir.path()), (true, false));

    // secret.txt still present means the marker could still be reread post-compaction.
    std::fs::write(dir.path().join("secret.txt"), "MARKER").expect("write secret");
    assert_eq!(compaction_boundary_filesystem_state(dir.path()), (false, false));

    // A missing result.txt fails closed (treated as NOT empty, not as "write pending").
    let empty_dir = tempfile::TempDir::new().expect("tempdir");
    assert_eq!(compaction_boundary_filesystem_state(empty_dir.path()), (true, false));
}

/// Build summarization-prompt bytes that trip the first-sight snapshot.
fn summarization_needle_bytes() -> Vec<u8> {
    let mut bytes = b"{\"instructions\":\"".to_vec();
    bytes.extend_from_slice(CODEX_SUMMARIZATION_PROMPT_NEEDLE);
    bytes
}

fn compaction_probe_signals(workspace: &Path) -> ObservedConnectionSignals {
    ObservedConnectionSignals {
        websocket_attempted: Arc::new(AtomicBool::new(false)),
        summarization_request_seen: Arc::new(AtomicBool::new(false)),
        summary_prefix_seen: Arc::new(AtomicBool::new(false)),
        secret_gone_at_compaction: Arc::new(AtomicBool::new(false)),
        result_empty_at_compaction: Arc::new(AtomicBool::new(false)),
        compaction_workspace: Some(Arc::from(workspace)),
    }
}

#[test]
fn scan_compaction_needles_snapshots_a_sound_boundary_once() {
    let dir = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(dir.path().join("result.txt"), "").expect("seed result");
    let signals = compaction_probe_signals(dir.path());

    scan_compaction_needles(&summarization_needle_bytes(), &signals);
    assert!(signals.secret_gone_at_compaction.load(Ordering::SeqCst));
    assert!(signals.result_empty_at_compaction.load(Ordering::SeqCst));

    // A write that lands AFTER the first summarization must not retroactively flip
    // the snapshot: compare_exchange snapshots exactly once at the true boundary.
    std::fs::write(dir.path().join("result.txt"), "MARKER").expect("write result");
    scan_compaction_needles(&summarization_needle_bytes(), &signals);
    assert!(signals.result_empty_at_compaction.load(Ordering::SeqCst));
}

#[test]
fn scan_compaction_needles_flags_a_result_written_before_compaction() {
    let dir = tempfile::TempDir::new().expect("tempdir");
    // The marker was already written (e.g. a batched pre-compaction write) when the
    // summarization request crosses the wire: result.txt is non-empty.
    std::fs::write(dir.path().join("result.txt"), "MARKER").expect("write result");
    let signals = compaction_probe_signals(dir.path());

    scan_compaction_needles(&summarization_needle_bytes(), &signals);
    assert!(signals.secret_gone_at_compaction.load(Ordering::SeqCst));
    assert!(
        !signals.result_empty_at_compaction.load(Ordering::SeqCst),
        "a result written before compaction must fail the boundary check"
    );
}

/// Number of ballast chapter reads present in [`COMPACTION_JSONL_OK`].
const COMPACTION_JSONL_OK_CHAPTERS: usize = 2;

/// A well-formed compaction trace: marker acquired once, ballast grows before the
/// write, write after growth, verifier runs after the write, non-empty summary,
/// nonzero usage.
const COMPACTION_JSONL_OK: &str = concat!(
    r#"{"type":"item.started","item":{"type":"command_execution","id":"c1"}}"#,
    "\n",
    r#"{"type":"item.completed","item":{"type":"command_execution","id":"c1","command":"cat secret.txt && rm -f secret.txt","exit_code":0}}"#,
    "\n",
    r#"{"type":"item.started","item":{"type":"command_execution","id":"c2"}}"#,
    "\n",
    r#"{"type":"item.completed","item":{"type":"command_execution","id":"c2","command":"cat chapter_01.txt","exit_code":0}}"#,
    "\n",
    r#"{"type":"item.started","item":{"type":"command_execution","id":"c3"}}"#,
    "\n",
    r#"{"type":"item.completed","item":{"type":"command_execution","id":"c3","command":"cat chapter_02.txt","exit_code":0}}"#,
    "\n",
    r#"{"type":"item.started","item":{"type":"command_execution","id":"c4"}}"#,
    "\n",
    r#"{"type":"item.completed","item":{"type":"command_execution","id":"c4","command":"printf '%s' 'TOK' > result.txt","exit_code":0}}"#,
    "\n",
    r#"{"type":"item.started","item":{"type":"command_execution","id":"c5"}}"#,
    "\n",
    r#"{"type":"item.completed","item":{"type":"command_execution","id":"c5","command":"./verify.sh","exit_code":0}}"#,
    "\n",
    r#"{"type":"item.completed","item":{"type":"agent_message","text":"done summarizing"}}"#,
    "\n",
    r#"{"type":"turn.completed","usage":{"input_tokens":100,"output_tokens":20}}"#,
);

#[test]
fn compaction_jsonl_accepts_marker_acquired_then_grown_then_written() {
    // The happy path must not panic.
    assert_live_compaction_codex_jsonl(COMPACTION_JSONL_OK, COMPACTION_JSONL_OK_CHAPTERS);
}

#[test]
#[should_panic(expected = "all ballast growth must precede the write")]
fn compaction_jsonl_rejects_growth_after_the_write() {
    // A ballast read after the result write means compaction could be deferred
    // past the write, so retention is not proven.
    let stdout = format!(
        "{COMPACTION_JSONL_OK}\n{}\n{}",
        r#"{"type":"item.started","item":{"type":"command_execution","id":"c6"}}"#,
        r#"{"type":"item.completed","item":{"type":"command_execution","id":"c6","command":"cat chapter_03.txt","exit_code":0}}"#,
    );
    assert_live_compaction_codex_jsonl(&stdout, COMPACTION_JSONL_OK_CHAPTERS);
}

#[test]
#[should_panic(expected = "exactly once")]
fn compaction_jsonl_rejects_rereading_the_marker_source() {
    // A second read of secret.txt is the reread path that could launder a lost marker.
    let stdout = format!(
        "{COMPACTION_JSONL_OK}\n{}\n{}",
        r#"{"type":"item.started","item":{"type":"command_execution","id":"c6"}}"#,
        r#"{"type":"item.completed","item":{"type":"command_execution","id":"c6","command":"cat secret.txt","exit_code":0}}"#,
    );
    assert_live_compaction_codex_jsonl(&stdout, COMPACTION_JSONL_OK_CHAPTERS);
}

#[test]
#[should_panic(expected = "off-sequence")]
fn compaction_jsonl_rejects_source_copied_before_the_secret_read() {
    // The reviewer's pre-secret laundering path: a glob copy that never names
    // secret.txt literally (`cp s*.txt scratch.txt`) stashes the marker BEFORE the
    // mandated read. It sits at index 0 — outside the pre-write window — so only the
    // full-run allowlist can reject it. A later `printf ... "$(cat scratch.txt)" >
    // result.txt` write would then recover it, defeating retention.
    let stdout = format!(
        "{}\n{}\n{COMPACTION_JSONL_OK}",
        r#"{"type":"item.started","item":{"type":"command_execution","id":"c0"}}"#,
        r#"{"type":"item.completed","item":{"type":"command_execution","id":"c0","command":"cp s*.txt scratch.txt","exit_code":0}}"#,
    );
    assert_live_compaction_codex_jsonl(&stdout, COMPACTION_JSONL_OK_CHAPTERS);
}

#[test]
#[should_panic(expected = "off-sequence")]
fn compaction_jsonl_rejects_a_stray_command_after_the_verifier() {
    // Even post-verification, an off-sequence command (here a copy) is a stash site
    // the full-run allowlist must reject; the windowed check never reaches it.
    let stdout = format!(
        "{COMPACTION_JSONL_OK}\n{}\n{}",
        r#"{"type":"item.started","item":{"type":"command_execution","id":"c6"}}"#,
        r#"{"type":"item.completed","item":{"type":"command_execution","id":"c6","command":"ls -la","exit_code":0}}"#,
    );
    assert_live_compaction_codex_jsonl(&stdout, COMPACTION_JSONL_OK_CHAPTERS);
}

#[test]
#[should_panic(expected = "mandated `printf")]
fn compaction_jsonl_rejects_a_printf_that_stashes_to_scratch_as_the_write() {
    // The reviewer's stash path: a `printf` that NAMES result.txt (so the loose
    // `starts_with("printf ") && contains("result.txt")` shape matched) but actually
    // redirects the token to scratch.txt, leaving result.txt empty at the compaction
    // boundary. Replacing the real write with it must leave no mandated write.
    let stashed = COMPACTION_JSONL_OK.replace(
        "printf '%s' 'TOK' > result.txt",
        "printf '%s' 'TOK' > scratch.txt && cat result.txt",
    );
    assert_live_compaction_codex_jsonl(&stashed, COMPACTION_JSONL_OK_CHAPTERS);
}

#[test]
#[should_panic(expected = "off-sequence")]
fn compaction_jsonl_rejects_a_scratch_stash_printf_alongside_the_real_write() {
    // Keep the genuine write but add a stash `printf` that names result.txt while
    // redirecting elsewhere. The loose shape accepted it as a mandated command; the
    // strict `is_mandated_result_write` makes the full-run allowlist flag it.
    let stdout = format!(
        "{}\n{}\n{COMPACTION_JSONL_OK}",
        r#"{"type":"item.started","item":{"type":"command_execution","id":"c0"}}"#,
        r#"{"type":"item.completed","item":{"type":"command_execution","id":"c0","command":"printf '%s' 'TOK' > scratch.txt && cat result.txt","exit_code":0}}"#,
    );
    assert_live_compaction_codex_jsonl(&stdout, COMPACTION_JSONL_OK_CHAPTERS);
}

#[test]
fn mandated_result_write_requires_a_sole_literal_redirect_to_result() {
    assert!(is_mandated_result_write("printf '%s' 'TOK' > result.txt"));
    assert!(is_mandated_result_write("printf '%s' 'TOK'  >  result.txt"));
    // Redirects to another file, even while naming result.txt elsewhere.
    assert!(!is_mandated_result_write(
        "printf '%s' 'TOK' > scratch.txt && cat result.txt"
    ));
    // Recovers a stashed marker via command substitution.
    assert!(!is_mandated_result_write(
        "printf '%s' \"$(cat scratch.txt)\" > result.txt"
    ));
    // Appends (second redirect) or pipes rather than a sole `>` to result.txt.
    assert!(!is_mandated_result_write("printf '%s' 'TOK' >> result.txt"));
    assert!(!is_mandated_result_write("printf '%s' 'TOK' | tee result.txt"));
    // Not a printf at all (a copy-style write).
    assert!(!is_mandated_result_write("cat scratch.txt > result.txt"));
}

#[test]
#[should_panic(expected = "ballast chapters must be read")]
fn compaction_jsonl_rejects_insufficient_growth_before_the_write() {
    // Requiring more chapters than were read before the write models a
    // write-before-compaction trace: not enough context accumulated to trip the
    // 8000-token trigger, so retention across compaction is not proven.
    assert_live_compaction_codex_jsonl(COMPACTION_JSONL_OK, COMPACTION_JSONL_OK_CHAPTERS + 1);
}

#[test]
#[should_panic(expected = "EXECUTE ./verify.sh")]
fn compaction_jsonl_rejects_missing_verifier() {
    // Dropping the ./verify.sh command leaves only a client-writable
    // `.verification-ran` marker, which cannot stand in for the real run.
    let without_verify = COMPACTION_JSONL_OK
        .lines()
        .filter(|line| !line.contains("verify.sh"))
        .collect::<Vec<_>>()
        .join("\n");
    assert_live_compaction_codex_jsonl(&without_verify, COMPACTION_JSONL_OK_CHAPTERS);
}

#[test]
#[should_panic(expected = "EXECUTE ./verify.sh")]
fn compaction_jsonl_rejects_reading_the_verifier_instead_of_running_it() {
    // `cat verify.sh` mentions the script but never runs it, so verification did
    // not actually happen.
    let reads_verifier = COMPACTION_JSONL_OK.replace("./verify.sh", "cat verify.sh");
    assert_live_compaction_codex_jsonl(&reads_verifier, COMPACTION_JSONL_OK_CHAPTERS);
}

#[test]
#[should_panic(expected = "DISTINCT ballast chapters")]
fn compaction_jsonl_rejects_rereading_one_chapter_as_fake_growth() {
    // Re-reading a single small file mentions "chapter_" many times but barely
    // grows context, so it cannot prove the >8000-token accumulation that trips
    // compaction. Two reads of the SAME chapter give one DISTINCT chapter.
    let fake_growth = COMPACTION_JSONL_OK.replace("chapter_02.txt", "chapter_01.txt");
    assert_live_compaction_codex_jsonl(&fake_growth, COMPACTION_JSONL_OK_CHAPTERS);
}

#[test]
fn chapter_labels_extracts_distinct_chapter_identifiers() {
    assert_eq!(chapter_labels("cat chapter_01.txt"), ["chapter_01"]);
    assert_eq!(
        chapter_labels("cat chapter_01.txt chapter_02.txt"),
        ["chapter_01", "chapter_02"]
    );
    assert!(chapter_labels("cat secret.txt").is_empty());
}

#[test]
#[should_panic(expected = "only single-chapter `cat` reads are allowed")]
fn compaction_jsonl_rejects_echoed_chapter_filenames_as_growth() {
    // `echo chapter_NN.txt` mentions a chapter label but reads nothing, so it
    // cannot grow accepted context toward the compaction trigger — and it is not
    // the bare single-chapter `cat` the pre-write window allowlist admits.
    let echoed = COMPACTION_JSONL_OK.replace("cat chapter", "echo chapter");
    assert_live_compaction_codex_jsonl(&echoed, COMPACTION_JSONL_OK_CHAPTERS);
}

#[test]
fn unwrap_shell_command_strips_a_single_bash_lc_layer() {
    assert_eq!(unwrap_shell_command("/bin/bash -lc './verify.sh'"), "./verify.sh");
    assert_eq!(
        unwrap_shell_command("bash -c \"cat chapter_01.txt\""),
        "cat chapter_01.txt"
    );
    // Bare commands pass through untouched.
    assert_eq!(unwrap_shell_command("./verify.sh"), "./verify.sh");
    assert_eq!(unwrap_shell_command("cat chapter_01.txt"), "cat chapter_01.txt");
}

#[test]
fn single_chapter_cat_read_requires_a_bare_cat_of_one_chapter() {
    assert_eq!(single_chapter_cat_read("cat chapter_01.txt"), Some("chapter_01"));
    // A leading ./ is still a bare read.
    assert_eq!(single_chapter_cat_read("cat ./chapter_01.txt"), Some("chapter_01"));
    // Output discarded or reshaped — no context growth.
    assert_eq!(single_chapter_cat_read("cat chapter_01.txt >/dev/null"), None);
    // Glued redirect with no space still splits into two tokens but must not count.
    assert_eq!(single_chapter_cat_read("cat chapter_01.txt>/dev/null"), None);
    assert_eq!(single_chapter_cat_read("cat chapter_01.txt|wc -c"), None);
    assert_eq!(single_chapter_cat_read("cat chapter_01.txt | wc -c"), None);
    // Not a read, many files, or no chapter.
    assert_eq!(single_chapter_cat_read("echo chapter_01.txt"), None);
    assert_eq!(single_chapter_cat_read("cat chapter_01.txt chapter_02.txt"), None);
    assert_eq!(single_chapter_cat_read("cat secret.txt"), None);
}

#[test]
#[should_panic(expected = "only single-chapter `cat` reads are allowed")]
fn compaction_jsonl_rejects_reads_redirected_to_devnull() {
    // `cat chapter_NN.txt >/dev/null` discards the contents, so nothing enters
    // context — and the glued redirect is not the bare single-chapter `cat` the
    // pre-write window allowlist admits, so it is rejected before the
    // distinct-chapter count is even reached.
    let discarded = COMPACTION_JSONL_OK
        .replace("cat chapter_01.txt", "cat chapter_01.txt >/dev/null")
        .replace("cat chapter_02.txt", "cat chapter_02.txt >/dev/null");
    assert_live_compaction_codex_jsonl(&discarded, COMPACTION_JSONL_OK_CHAPTERS);
}

#[test]
#[should_panic(expected = "must be exactly")]
fn compaction_jsonl_rejects_secret_copied_to_scratch_before_deletion() {
    // Bundling a copy into the single secret command preserves the marker on disk
    // past step 1, so a post-compaction read could launder it back — the retention
    // proof is defeated even though secret.txt is still read exactly once. The
    // exact-command allowlist rejects any secret command but the mandated form.
    let preserved = COMPACTION_JSONL_OK.replace(
        "cat secret.txt && rm -f secret.txt",
        "cat secret.txt && cp secret.txt scratch && rm -f secret.txt",
    );
    assert_live_compaction_codex_jsonl(&preserved, COMPACTION_JSONL_OK_CHAPTERS);
}

#[test]
#[should_panic(expected = "must be exactly")]
fn compaction_jsonl_rejects_secret_copied_via_python_before_deletion() {
    // A Python copy matches no copy-spelling blocklist, but the exact-command
    // allowlist rejects it regardless: the secret command is no longer the
    // mandated read-then-delete form.
    let preserved = COMPACTION_JSONL_OK.replace(
        "cat secret.txt && rm -f secret.txt",
        "cat secret.txt && python3 -c 'import shutil; shutil.copyfile(secret.txt, scratch)' && rm -f secret.txt",
    );
    assert_live_compaction_codex_jsonl(&preserved, COMPACTION_JSONL_OK_CHAPTERS);
}

#[test]
#[should_panic(expected = "only single-chapter `cat` reads are allowed")]
fn compaction_jsonl_rejects_token_stashed_to_scratch_before_the_write() {
    // Even with the secret command untouched, a SEPARATE later command that writes
    // the remembered token to a scratch file preserves it on disk for a
    // post-compaction reread, so it must be rejected.
    let stashed = COMPACTION_JSONL_OK.replace(
        r#"{"type":"item.started","item":{"type":"command_execution","id":"c4"}}"#,
        concat!(
            r#"{"type":"item.started","item":{"type":"command_execution","id":"s1"}}"#,
            "\n",
            r#"{"type":"item.completed","item":{"type":"command_execution","id":"s1","command":"printf '%s' 'TOK' > scratch.txt","exit_code":0}}"#,
            "\n",
            r#"{"type":"item.started","item":{"type":"command_execution","id":"c4"}}"#,
        ),
    );
    assert_live_compaction_codex_jsonl(&stashed, COMPACTION_JSONL_OK_CHAPTERS);
}

#[test]
#[should_panic(expected = "only single-chapter `cat` reads are allowed")]
fn compaction_jsonl_rejects_python_copy_in_the_prewrite_window() {
    // A Python copy of a chapter to a scratch pad matches no copy builtin, but the
    // pre-write window allowlist admits only bare single-chapter `cat` reads, so it
    // is rejected regardless of spelling.
    let stashed = COMPACTION_JSONL_OK.replace(
        r#"{"type":"item.started","item":{"type":"command_execution","id":"c4"}}"#,
        concat!(
            r#"{"type":"item.started","item":{"type":"command_execution","id":"s1"}}"#,
            "\n",
            r#"{"type":"item.completed","item":{"type":"command_execution","id":"s1","command":"python3 -c 'import shutil; shutil.copyfile(chapter_01.txt, scratch.txt)'","exit_code":0}}"#,
            "\n",
            r#"{"type":"item.started","item":{"type":"command_execution","id":"c4"}}"#,
        ),
    );
    assert_live_compaction_codex_jsonl(&stashed, COMPACTION_JSONL_OK_CHAPTERS);
}

#[test]
#[should_panic(expected = "off-shell apply_patch/file_change")]
fn compaction_jsonl_rejects_apply_patch_scratch_write() {
    // `apply_patch` edits are reported as `file_change` items, not shell commands,
    // so the command-execution guards never see them. One could stash the marker in
    // a scratch file off-shell, to be laundered back after compaction, so any
    // file_change item must reject the whole trace.
    let patched = COMPACTION_JSONL_OK.replace(
        r#"{"type":"item.started","item":{"type":"command_execution","id":"c4"}}"#,
        concat!(
            r#"{"type":"item.completed","item":{"type":"file_change","id":"f1","changes":[{"path":"scratch.txt","kind":"add"}]}}"#,
            "\n",
            r#"{"type":"item.started","item":{"type":"command_execution","id":"c4"}}"#,
        ),
    );
    assert_live_compaction_codex_jsonl(&patched, COMPACTION_JSONL_OK_CHAPTERS);
}

#[test]
#[should_panic(expected = "mandated `printf")]
fn compaction_jsonl_rejects_copy_style_result_write() {
    // A `cat scratch > result.txt` write could launder a marker stashed off-wire
    // instead of proving the token survived compaction in context; only the
    // mandated `printf '%s' '<TOKEN>' > result.txt` is accepted as the write.
    let copied = COMPACTION_JSONL_OK.replace("printf '%s' 'TOK' > result.txt", "cat scratch.txt > result.txt");
    assert_live_compaction_codex_jsonl(&copied, COMPACTION_JSONL_OK_CHAPTERS);
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
                            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
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
        r#"{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","required":["id","object","created_at","completed_at","status","incomplete_details","model","previous_response_id","instructions","output","error","tools","tool_choice","truncation","parallel_tool_calls","text","top_p","presence_penalty","frequency_penalty","top_logprobs","temperature","reasoning","user","usage","max_output_tokens","max_tool_calls","store","background","service_tier","metadata","safety_identifier","prompt_cache_key"],"properties":{"id":{"type":"string"},"object":{"const":"response"},"created_at":{"type":"integer"},"completed_at":{"type":["integer","null"]},"status":{"type":"string"},"incomplete_details":{"type":["object","null"]},"model":{"type":"string"},"previous_response_id":{"type":["string","null"]},"instructions":{"type":["string","null"]},"output":{"type":"array"},"error":{"type":["object","null"]},"tools":{"type":"array"},"tool_choice":{"type":["string","object"]},"truncation":{"type":"string"},"parallel_tool_calls":{"type":"boolean"},"text":{"type":"object"},"top_p":{"type":"number"},"presence_penalty":{"type":"number"},"frequency_penalty":{"type":"number"},"top_logprobs":{"type":"integer"},"temperature":{"type":"number"},"reasoning":{"type":["object","null"]},"user":{"type":["string","null"]},"usage":{"anyOf":[{"type":"null"},{"type":"object","required":["input_tokens","output_tokens","total_tokens","input_tokens_details","output_tokens_details"],"properties":{"input_tokens":{"type":"integer"},"output_tokens":{"type":"integer"},"total_tokens":{"type":"integer"},"input_tokens_details":{"type":"object","required":["cached_tokens","cache_write_tokens"],"properties":{"cached_tokens":{"type":"integer"},"cache_write_tokens":{"type":"integer"}}},"output_tokens_details":{"type":"object","required":["reasoning_tokens"],"properties":{"reasoning_tokens":{"type":"integer"}}}}}]},"max_output_tokens":{"type":["integer","null"]},"max_tool_calls":{"type":["integer","null"]},"store":{"type":"boolean"},"background":{"type":"boolean"},"service_tier":{"type":"string"},"metadata":{"type":"object"},"safety_identifier":{"type":["string","null"]},"prompt_cache_key":{"type":["string","null"]}}}"#,
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
        compaction_limits,
    } = options;
    let codex_home = tempfile::tempdir().expect("temporary CODEX_HOME should be created");
    let compaction_keys = compaction_limits.map_or_else(String::new, |limits| {
        format!(
            "model_context_window = {}\nmodel_auto_compact_token_limit = {}\n",
            limits.context_window, limits.auto_compact_token_limit
        )
    });
    let config = format!(
        r#"model = "{model}"
model_provider = "praxis"
web_search = "disabled"
{compaction_keys}
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

/// Whether an observed event is the `start_proxy` readiness probe.
///
/// `start_proxy` waits for the listener by sending `GET /`, and the translated
/// chain routes every unmatched path to the selected provider, so the scripted
/// backend reports that probe as an unexpected request before any translated
/// traffic arrives. It is harness noise, not client behavior under test.
fn is_readiness_probe(event: &HttpBackendEvent) -> bool {
    matches!(event, HttpBackendEvent::UnexpectedRequest { method, path } if method == "GET" && path == "/")
}

/// The next provider-facing request, skipping the readiness probe.
async fn next_translated_request(backend: &mut praxis_test_utils::HttpBackendGuard) -> CapturedHttpRequest {
    loop {
        let event = tokio::time::timeout(Duration::from_secs(5), backend.next_event())
            .await
            .expect("provider should receive translated request")
            .expect("backend event channel should remain open");
        if is_readiness_probe(&event) {
            continue;
        }
        let HttpBackendEvent::Request(request) = event else {
            panic!("provider should receive a normal HTTP request, got {event:?}");
        };
        return request;
    }
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
        if is_readiness_probe(&event) {
            continue;
        }
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

/// Validate the compaction run's JSONL.
///
/// Pinned Codex 0.144.1 emits exactly one `turn.completed` for the whole
/// submitted prompt (covering every tool round), so accepted-turn growth is
/// proven by counting correlated successful `command_execution` items, not
/// `turn.completed` events.
///
/// Marker retention across compaction is proven structurally: the marker lives
/// only in `secret.txt`, which step 1 reads and deletes. This check asserts the
/// marker source is read exactly once, before the result write, with
/// `min_chapters_before_write` ballast chapter reads in between (the growth that
/// trips compaction) — so a lost marker cannot be laundered back by a
/// post-compaction reread. Passing the full `CODEX_COMPACTION_BALLAST_CHAPTERS`
/// count forces ALL ballast (well past the 8000-token trigger) to precede the
/// write, so a write-before-compaction trace cannot pass. The definite
/// compaction signal itself is the client->upstream wire evidence asserted by
/// [`HttpTransportObserver::assert_compaction_traversed_praxis`]. Finally, the
/// required `./verify.sh` must run successfully after the write, so the client
/// actually exercises its own result check.
/// Returns every `chapter_<digits>` label mentioned in a command, so re-reading
/// one file cannot masquerade as reading many distinct chapters.
fn chapter_labels(command: &str) -> Vec<&str> {
    const NEEDLE: &str = "chapter_";
    let bytes = command.as_bytes();
    let mut labels = Vec::new();
    for (start, _) in command.match_indices(NEEDLE) {
        let mut end = start + NEEDLE.len();
        while end < bytes.len() && bytes[end].is_ascii_digit() {
            end += 1;
        }
        if end > start + NEEDLE.len()
            && let Some(label) = command.get(start..end)
        {
            labels.push(label);
        }
    }
    labels
}

/// Unwraps a single `<shell> -c/-lc '<inner>'` layer and returns the command
/// Codex actually executed. Pinned Codex 0.144.1 records shell-wrapped
/// invocations such as `/bin/bash -lc './verify.sh'`; the unit fixtures use the
/// bare form. Only one quoting layer is stripped, which is all Codex adds.
fn unwrap_shell_command(command: &str) -> &str {
    let trimmed = command.trim();
    for flag in [" -lc ", " -c "] {
        if let Some((_, rest)) = trimmed.split_once(flag) {
            let rest = rest.trim();
            let unquoted = rest
                .strip_prefix('\'')
                .and_then(|inner| inner.strip_suffix('\''))
                .or_else(|| rest.strip_prefix('"').and_then(|inner| inner.strip_suffix('"')))
                .unwrap_or(rest);
            return unquoted.trim();
        }
    }
    trimmed
}

/// If `command` is exactly `cat <chapter_NN.txt>` — a bare read whose contents
/// reach stdout and are therefore fed back as accepted context — returns that
/// chapter's label.
///
/// The exact two-token shape is required on purpose. `echo chapter_01.txt` reads
/// nothing; `cat chapter_01.txt >/dev/null` (or any `>`, pipe, or extra argument)
/// discards or reshapes the output so it never enters context; `cat a b c` folds
/// many files into one turn. None of those grow accepted context toward the
/// compaction trigger, so none count.
///
/// The file token must be EXACTLY `<label>.txt` (an optional `./` prefix aside).
/// Whitespace splitting alone is not enough: `cat chapter_01.txt>/dev/null` has no
/// space before the redirect, so it splits into just two tokens, yet the glued
/// `>/dev/null` discards the output. Requiring the token to equal the bare chapter
/// filename rejects any suffix glued on without a space.
fn single_chapter_cat_read(command: &str) -> Option<&str> {
    if let [cat, file] = command.split_whitespace().collect::<Vec<_>>().as_slice()
        && *cat == "cat"
        && let [only] = chapter_labels(file).as_slice()
    {
        let bare = file.strip_prefix("./").unwrap_or(file);
        // `bare` starts with `only` and the only remaining characters are `.txt`.
        if bare.len() == only.len() + 4 && bare.starts_with(only) && bare.ends_with(".txt") {
            return Some(only);
        }
    }
    None
}

/// Whether a command (already shell-unwrapped) is EXACTLY the mandated marker
/// write: `printf <fmt> <arg> > result.txt`, carrying the token in its own literal
/// argument. A looser `starts_with("printf ") && contains("result.txt")` check
/// accepts a stash-and-launder pair the snapshot alone cannot catch: e.g.
/// `printf '%s' 'TOK' > scratch.txt && cat result.txt` (redirects the token to a
/// scratch file while naming result.txt, leaving result.txt empty at the
/// compaction boundary) and a later `printf '%s' "$(cat scratch.txt)" > result.txt`
/// (recovers the stashed marker via substitution after compaction). Requiring a
/// SOLE `>` redirect to `result.txt` and barring substitution/chaining/piping
/// forces the token to be a literal argument in the one write command.
fn is_mandated_result_write(command: &str) -> bool {
    let inner = command.trim();
    if !inner.starts_with("printf ") {
        return false;
    }
    // No substitution, chaining, piping, or backgrounding — the token must be a
    // literal argument, never read from or stashed to another file.
    if inner.contains(['$', '`', '|', '&', ';', '(', ')', '\n']) {
        return false;
    }
    // Exactly one `>` redirect, and its target is result.txt (so `>> result.txt`,
    // `> scratch.txt && cat result.txt`, and `> other.txt` are all rejected).
    let mut redirects = inner.split('>');
    redirects.next(); // the `printf ... ` command part
    matches!(
        (redirects.next(), redirects.next()),
        (Some(target), None) if target.trim() == "result.txt"
    )
}

/// The exact step-1 command the compaction prompt mandates: read the marker, then
/// immediately delete its only on-disk copy. The validator allowlists this exact
/// string (after unwrapping Codex's shell layer) so no bundled copy — in any
/// spelling — can preserve the marker for a post-compaction reread.
const CODEX_SECRET_COMMAND: &str = "cat secret.txt && rm -f secret.txt";

/// Counts DISTINCT ballast chapters actually READ strictly between `after` and
/// `before`, counting only genuine one-file-per-turn `cat` reads (see
/// [`single_chapter_cat_read`]) whose command also succeeded.
fn distinct_chapter_reads(commands: &[(String, bool)], after: usize, before: usize) -> usize {
    let mut seen = std::collections::HashSet::new();
    for (index, (command, ok)) in commands.iter().enumerate() {
        if index <= after || index >= before || !ok {
            continue;
        }
        if let Some(label) = single_chapter_cat_read(unwrap_shell_command(command)) {
            seen.insert(label);
        }
    }
    seen.len()
}

fn assert_live_compaction_codex_jsonl(stdout: &str, min_chapters_before_write: usize) {
    let mut started_commands = Vec::new();
    // Completed command_executions in completion order: (command, correlated-success).
    let mut completed_commands: Vec<(String, bool)> = Vec::new();
    // Any `apply_patch`/`file_change` edit Codex made OUTSIDE the shell.
    let mut file_change_items: Vec<String> = Vec::new();
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

        // `apply_patch` edits are reported as `file_change` items, NOT shell
        // commands, so the command-execution guards below never see them. Codex
        // could use one to stash the marker in a scratch file between the allowed
        // shell reads (then launder it back after compaction), so capture every
        // file_change for the hard rejection after the loop.
        if event_type == Some("item.completed") && item_type == Some("file_change") {
            file_change_items.push(
                event
                    .pointer("/item/changes")
                    .map_or_else(|| line.to_owned(), |changes| changes.to_string()),
            );
        }
        if event_type == Some("item.started") && item_type == Some("command_execution") && !item_id.is_empty() {
            started_commands.push(item_id.to_owned());
        }
        if event_type == Some("item.completed") && item_type == Some("command_execution") {
            let command = event
                .pointer("/item/command")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default();
            let exit_code = event.pointer("/item/exit_code").and_then(serde_json::Value::as_i64);
            let correlated = !command.is_empty()
                && exit_code == Some(0)
                && started_commands.iter().any(|started| started == item_id);
            completed_commands.push((command.to_owned(), correlated));
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

    // The task edits files ONLY through the mandated shell `printf`; it never uses
    // `apply_patch`. Any `file_change` item is an off-shell write that the
    // command-execution guards cannot see and could stash the marker for a
    // post-compaction reread, so reject the whole trace.
    assert!(
        file_change_items.is_empty(),
        "Codex must edit files only through the shell `printf` write; {} off-shell apply_patch/file_change edit(s) could stash the marker for a post-compaction reread: {:?}\nstdout:\n{stdout}",
        file_change_items.len(),
        file_change_items
    );

    // Multiple accepted agent rounds (not one oversized rejected request).
    let successful_commands = completed_commands.iter().filter(|(_, ok)| *ok).count();
    assert!(
        successful_commands >= 2,
        "compaction workflow should run multiple accepted command turns so context grows past the limit, saw {successful_commands}; stdout:\n{stdout}"
    );

    // The marker source is read exactly once; it is deleted in step 1 so it can
    // never be reread to recover a marker dropped by compaction.
    let secret_positions: Vec<usize> = completed_commands
        .iter()
        .enumerate()
        .filter(|(_, (command, _))| command.contains("secret.txt"))
        .map(|(index, _)| index)
        .collect();
    assert!(
        secret_positions.len() == 1,
        "Codex must read the marker source (secret.txt) exactly once; a reread could launder lost context across compaction, saw {} reads; stdout:\n{stdout}",
        secret_positions.len()
    );
    let secret_index = secret_positions[0];

    // The single secret command must be EXACTLY the permitted read-then-delete.
    // An allowlist (not a copy-spelling blocklist) is required: a bundled copy can
    // be spelled countless ways the tests cannot enumerate — `cp`, a `>` redirect,
    // `tee`, or `python -c 'import shutil; shutil.copyfile("secret.txt","scratch")'`
    // — any of which would stash the marker for a post-compaction reread. Requiring
    // the exact command the prompt mandates rejects all of them at once.
    let secret_command = unwrap_shell_command(&completed_commands[secret_index].0);
    assert!(
        secret_command == CODEX_SECRET_COMMAND,
        "the secret command must be exactly `{CODEX_SECRET_COMMAND}` so it cannot stash the marker \
         anywhere for a post-compaction reread; got: {secret_command}\nstdout:\n{stdout}"
    );

    // The remembered marker is written back only after it was acquired, and ONLY
    // via the mandated `printf '%s' '<TOKEN>' > result.txt`. Requiring the `printf`
    // literal (not merely `> result.txt`) is what forces the token to be carried in
    // the write command itself: a copy-style write like `cat scratch > result.txt`
    // would launder a marker stashed off-wire (e.g. by an apply_patch or a glob
    // copy) instead of proving the client still held the token after compaction.
    let write_index = completed_commands
        .iter()
        .position(|(command, ok)| *ok && is_mandated_result_write(unwrap_shell_command(command)))
        .unwrap_or_else(|| {
            panic!(
                "Codex must write the remembered marker into result.txt with the mandated `printf ... > result.txt`; a copy such as `cat scratch > result.txt` is not accepted because it could launder a stashed marker; stdout:\n{stdout}"
            )
        });
    assert!(
        secret_index < write_index,
        "Codex must acquire the marker before writing it back; stdout:\n{stdout}"
    );

    // Allowlist the pre-write window: between acquiring the marker and writing the
    // result, the ONLY permitted commands are genuine single-chapter `cat` reads.
    // Anything else — an `echo`/`printf` stash, a `cp`, a `tee`, or a Python copy
    // (`python -c 'import shutil; shutil.copyfile(...)'`) — could preserve the
    // marker on disk for a post-compaction reread, so reject it. Allowlisting the
    // one intended command shape (rather than blocklisting copy spellings) closes
    // every stash spelling at once. The secret command (at `secret_index`) and the
    // result write (at `write_index`) are outside this window.
    if let Some((index, (command, _))) = completed_commands.iter().enumerate().find(|(index, (command, _))| {
        *index > secret_index
            && *index < write_index
            && single_chapter_cat_read(unwrap_shell_command(command)).is_none()
    }) {
        panic!(
            "only single-chapter `cat` reads are allowed between acquiring the marker and writing the \
             result (command {index}); any other command could stash the marker for a post-compaction \
             reread; got: {command}\nstdout:\n{stdout}"
        );
    }

    // Context growth must happen AFTER acquiring the marker and BEFORE writing it
    // back. We count DISTINCT ballast chapters (chapter_NN) read in that window,
    // not raw chapter-mentioning commands: re-reading one small file N times (e.g.
    // `cat chapter_01.txt` x10) mentions "chapter_" N times but accumulates almost
    // no tokens and would not trip compaction. Requiring the FULL distinct set
    // means the whole >8000-token accumulation — and therefore the wire-confirmed
    // inline compaction it triggers — necessarily precedes the write.
    let distinct_chapters_before_write = distinct_chapter_reads(&completed_commands, secret_index, write_index);
    assert!(
        distinct_chapters_before_write >= min_chapters_before_write,
        "all {min_chapters_before_write} DISTINCT ballast chapters must be read between acquiring the marker and writing it, so the >8000-token accumulation (hence compaction) precedes the write; saw {distinct_chapters_before_write} distinct; stdout:\n{stdout}"
    );
    let growth_reads_at_or_after_write = completed_commands
        .iter()
        .enumerate()
        .filter(|(index, (command, _))| *index >= write_index && command.contains("chapter_"))
        .count();
    assert!(
        growth_reads_at_or_after_write == 0,
        "all ballast growth must precede the write so compaction cannot be deferred past it, saw {growth_reads_at_or_after_write} chapter read(s) at/after the write; stdout:\n{stdout}"
    );

    // The required verifier must actually be EXECUTED, successfully, after the
    // write. Matching `contains("verify.sh")` would accept `cat verify.sh`, so
    // require the executed command to START with `./verify.sh` — but unwrap the
    // `/bin/bash -lc '...'` layer Codex records first, so a real wrapped run is
    // accepted while reading the script is not. A client-writable
    // `.verification-ran` marker cannot stand in for it: the JSONL records the
    // real command and its exit code.
    let verify_index = completed_commands
        .iter()
        .position(|(command, ok)| *ok && unwrap_shell_command(command).starts_with("./verify.sh"))
        .unwrap_or_else(|| {
            panic!("Codex must EXECUTE ./verify.sh successfully after writing the result; stdout:\n{stdout}")
        });
    assert!(
        write_index < verify_index,
        "./verify.sh must run after the result write so it checks the written marker; stdout:\n{stdout}"
    );

    // Allowlist the ENTIRE run, not just the pre-write window. The windowed check
    // above cannot see a stash created BEFORE the secret read or AFTER the verifier:
    // e.g. `cp s*.txt scratch.txt` (a glob that never names secret.txt literally)
    // run first, then `printf '%s' "$(cat scratch.txt)" > result.txt` as the write —
    // the copy is at `index <= secret_index`, outside the window, and the write
    // still matches the `printf ... result.txt` shape. Requiring every executed
    // command across the run to be one of the four mandated shapes (the exact secret
    // read-then-delete, a single-chapter `cat`, the `printf ... > result.txt` write,
    // or `./verify.sh`) removes every off-sequence stash site at the source. Runs
    // last so the specific checks above keep their precise diagnostics.
    if let Some((index, (command, _))) = completed_commands.iter().enumerate().find(|(_, (command, _))| {
        let inner = unwrap_shell_command(command);
        let allowed = inner == CODEX_SECRET_COMMAND
            || single_chapter_cat_read(inner).is_some()
            || is_mandated_result_write(inner)
            || inner.starts_with("./verify.sh");
        !allowed
    }) {
        panic!(
            "every command in the compaction run must be one of the four mandated shapes (the exact secret \
             read-then-delete, a single-chapter `cat`, the `printf ... > result.txt` write, or `./verify.sh`); \
             command {index} is off-sequence and could stash the marker for a post-compaction reread; got: \
             {command}\nstdout:\n{stdout}"
        );
    }

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
