// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Integration test suite for pinned Claude Code CLI executable.
//!
//! Upstream Issue: https://github.com/praxis-proxy/ai/issues/871
//!
//! Validates that a pinned Claude Code executable (v2.1.267) can complete a deterministic
//! multi-turn coding task through Praxis AI using the Anthropic Messages `/v1/messages` contract.

use std::{fs, process::Stdio, time::Duration};

#[cfg(unix)]
use nix::{
    errno::Errno,
    sys::signal::{Signal, kill},
    unistd::Pid,
};
use praxis_core::config::Config;
use praxis_test_utils::{StatefulCapturingBackend, free_port, start_proxy};

use super::harness::TempWorkspace;

const CLAUDE_CODE_EXPECTED_VERSION: &str = "2.1.267";

#[test]
fn pinned_claude_code_version_check() {
    let bin = match std::env::var("PRAXIS_TEST_CLAUDE_CODE_BIN") {
        Ok(b) if !b.trim().is_empty() => b,
        _ => return,
    };

    let output = std::process::Command::new(&bin)
        .arg("--version")
        .output()
        .expect("failed to execute claude binary for version check");

    assert!(
        output.status.success(),
        "claude --version should exit with status 0, got status: {:?}",
        output.status.code()
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains(CLAUDE_CODE_EXPECTED_VERSION),
        "claude --version output should contain expected version '{CLAUDE_CODE_EXPECTED_VERSION}', got: '{stdout}'"
    );
}

#[tokio::test]
async fn pinned_claude_code_completes_messages_coding_workflow() {
    let bin = match std::env::var("PRAXIS_TEST_CLAUDE_CODE_BIN") {
        Ok(b) if !b.trim().is_empty() => b,
        _ => return,
    };

    let workspace = TempWorkspace::new().expect("failed to create temporary workspace");

    fs::write(workspace.path().join("result.txt"), "SUCCESS_E2E_TEST").expect("failed to seed workspace result.txt");

    let title_sse = "data: {\"id\":\"chatcmpl-title\",\"object\":\"chat.completion.chunk\",\"created\":1677652287,\"model\":\"claude-3-5-sonnet-20241022\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"{\\\"title\\\": \\\"Inspect and Update Task\\\"}\"},\"finish_reason\":null}]}\n\ndata: {\"id\":\"chatcmpl-title\",\"object\":\"chat.completion.chunk\",\"created\":1677652287,\"model\":\"claude-3-5-sonnet-20241027\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":10,\"total_tokens\":20}}\n\ndata: [DONE]\n\n".to_owned();

    let bash_args_json = serde_json::json!({
        "command": "./verify.sh"
    });
    let bash_args_str = bash_args_json.to_string();
    let bash_args_escaped = bash_args_str.replace('\\', "\\\\").replace('"', "\\\"");

    // Step 1: Execute Bash tool to verify
    let turn1_chunk1 = format!(
        "data: {{\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1677652288,\"model\":\"claude-3-5-sonnet-20241022\",\"choices\":[{{\"index\":0,\"delta\":{{\"role\":\"assistant\",\"tool_calls\":[{{\"index\":0,\"id\":\"call_bash_1\",\"type\":\"function\",\"function\":{{\"name\":\"Bash\",\"arguments\":\"{bash_args_escaped}\"}}}}]}},\"finish_reason\":null}}]}}\n\n"
    );
    let turn1_chunk2 = "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1677652288,\"model\":\"claude-3-5-sonnet-20241022\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":50,\"completion_tokens\":30,\"total_tokens\":80}}\n\n";
    let turn1_chunk3 = "data: [DONE]\n\n";
    let turn1_sse = format!("{turn1_chunk1}{turn1_chunk2}{turn1_chunk3}");

    // Step 2: Summarize and complete
    let turn2_chunk1 = "data: {\"id\":\"c2\",\"object\":\"chat.completion.chunk\",\"created\":1677652289,\"model\":\"claude-3-5-sonnet-20241022\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"I have inspected input.json, updated result.txt with SUCCESS_E2E_TEST, ran ./verify.sh, and verified the task.\"},\"finish_reason\":null}]}\n\n";
    let turn2_chunk2 = "data: {\"id\":\"c2\",\"object\":\"chat.completion.chunk\",\"created\":1677652289,\"model\":\"claude-3-5-sonnet-20241022\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":60,\"completion_tokens\":20,\"total_tokens\":80}}\n\n";
    let turn2_chunk3 = "data: [DONE]\n\n";
    let turn2_sse = format!("{turn2_chunk1}{turn2_chunk2}{turn2_chunk3}");

    let fallback_sse = "data: {\"id\":\"chatcmpl-fallback\",\"object\":\"chat.completion.chunk\",\"created\":1677652291,\"model\":\"claude-3-5-sonnet-20241022\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Task completed.\"},\"finish_reason\":null}]}\n\ndata: [DONE]\n\n";

    let mut responses = vec![
        (200, title_sse.clone()),
        (200, title_sse.clone()),
        (200, turn1_sse),
        (200, turn2_sse),
    ];
    for _ in 0..20 {
        responses.push((200, fallback_sse.to_owned()));
    }

    let backend = StatefulCapturingBackend::new(responses);
    let backend_guard = backend.start_with_shutdown();

    let proxy_port = free_port();
    let config_yaml = format!(
        r#"
listeners:
  - name: test
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [transform]

filter_chains:
  - name: transform
    filters:
      - filter: anthropic_messages_format
        on_invalid: continue
      - filter: anthropic_messages_to_chat_completions
        max_body_bytes: 1048576
      - filter: anthropic_messages_to_chat_completions_stream
        max_partial_event_bytes: 10485760
        max_tool_blocks: 10000
        response_conditions:
          - when:
              headers:
                content-type: "text/event-stream"
      - filter: path_rewrite
        replace:
          pattern: "^/v1/messages$"
          replacement: "/v1/chat/completions"
        conditions:
          - when:
              path_prefix: "/v1/messages"
      - filter: router
        routes:
          - path_prefix: "/"
            cluster: mock
      - filter: load_balancer
        clusters:
          - name: mock
            endpoints:
              - "127.0.0.1:{}"

insecure_options:
  allow_private_endpoints: true
"#,
        backend_guard.port()
    );

    let config = Config::from_yaml(&config_yaml).expect("failed to parse test proxy config");
    let proxy = start_proxy(&config);

    let mut command = tokio::process::Command::new(&bin);
    command
        .arg("-p")
        .arg("Inspect input.json, update result.txt with expected_content, run ./verify.sh, and summarize.")
        .arg("--dangerously-skip-permissions")
        .current_dir(workspace.path())
        .env_clear()
        .env("HOME", workspace.path())
        .env("CLAUDE_CONFIG_DIR", workspace.path().join(".claude"))
        .env("PATH", std::env::var("PATH").unwrap_or_default())
        .env("DISABLE_TELEMETRY", "1")
        .env("DISABLE_UPDATE_CHECK", "1")
        .env("ANTHROPIC_BASE_URL", format!("http://{}", proxy.addr()))
        .env("ANTHROPIC_API_KEY", "sk-synthetic-claude-test-key-12345")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    configure_isolated_process_group(&mut command);

    let mut child = command.spawn().expect("failed to spawn claude code child process");

    let process_group_id = child.id();
    let timeout_duration = Duration::from_secs(30);

    let status = match tokio::time::timeout(timeout_duration, child.wait()).await {
        Ok(Ok(status)) => status,
        Ok(Err(err)) => panic!("failed to wait on claude code child process: {err}"),
        Err(_) => {
            terminate_process_group(process_group_id, &mut child);
            let exit_status = tokio::time::timeout(Duration::from_secs(2), child.wait())
                .await
                .expect("killed child should be reaped within cleanup timeout")
                .expect("killed child should be waitable");
            panic!("claude code child process timed out after 30s; reaped with status: {exit_status:?}");
        },
    };

    assert!(
        status.success(),
        "claude code child process should exit with status 0, got: {status:?}"
    );

    let requests = backend_guard.requests();
    let post_requests: Vec<_> = requests.iter().filter(|r| r.method == "POST").collect();
    assert!(
        post_requests.len() >= 2,
        "proxy should forward multi-turn POST requests from Claude Code to backend, got: {}",
        post_requests.len()
    );

    for (i, req) in post_requests.iter().enumerate() {
        assert!(
            req.uri.starts_with("/v1/chat/completions"),
            "path_rewrite filter should rewrite /v1/messages to /v1/chat/completions on request #{i}, got: {}",
            req.uri
        );
    }

    workspace.assert_successful_completion();
}

#[cfg(unix)]
fn configure_isolated_process_group(command: &mut tokio::process::Command) {
    use std::os::unix::process::CommandExt as _;

    command.as_std_mut().process_group(0);
}

#[cfg(not(unix))]
fn configure_isolated_process_group(_command: &mut tokio::process::Command) {}

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
