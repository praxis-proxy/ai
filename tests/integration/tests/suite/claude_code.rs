// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Integration test suite for pinned Claude Code CLI executable.
//!
//! Upstream Issue: https://github.com/praxis-proxy/ai/issues/871
//!
//! Validates that a pinned Claude Code executable can complete a deterministic
//! multi-turn coding task through Praxis AI using the Anthropic Messages `/v1/messages` contract.

use std::{process::Stdio, time::Duration};

#[cfg(unix)]
use nix::{
    errno::Errno,
    sys::signal::{Signal, kill},
    unistd::Pid,
};
use praxis_core::config::Config;
use praxis_test_utils::{StatefulCapturingBackend, free_port, start_proxy};

use super::harness::{CLAUDE_CODE_PINNED_VERSION, TempWorkspace};

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
        stdout.contains(CLAUDE_CODE_PINNED_VERSION),
        "claude --version output should contain expected version '{CLAUDE_CODE_PINNED_VERSION}', got: '{stdout}'"
    );
}

#[tokio::test]
async fn pinned_claude_code_completes_messages_coding_workflow() {
    let bin = match std::env::var("PRAXIS_TEST_CLAUDE_CODE_BIN") {
        Ok(b) if !b.trim().is_empty() => b,
        _ => return,
    };

    let workspace = TempWorkspace::new().expect("failed to create temporary workspace");
    let expected_content = workspace.expected_content();

    let title_sse = "data: {\"id\":\"chatcmpl-title\",\"object\":\"chat.completion.chunk\",\"created\":1677652287,\"model\":\"claude-3-5-sonnet-20241022\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"{\\\"title\\\": \\\"Inspect and Update Task\\\"}\"},\"finish_reason\":null}]}\n\ndata: {\"id\":\"chatcmpl-title\",\"object\":\"chat.completion.chunk\",\"created\":1677652287,\"model\":\"claude-3-5-sonnet-20241027\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":10,\"total_tokens\":20}}\n\ndata: [DONE]\n\n".to_owned();

    let bash_args_json = serde_json::json!({
        "command": format!("sh -c 'echo {expected_content} > result.txt && ./verify.sh'")
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
    let turn2_chunk1 = format!(
        "data: {{\"id\":\"c2\",\"object\":\"chat.completion.chunk\",\"created\":1677652289,\"model\":\"claude-3-5-sonnet-20241022\",\"choices\":[{{\"index\":0,\"delta\":{{\"role\":\"assistant\",\"content\":\"I have inspected input.json, updated result.txt with {expected_content}, ran ./verify.sh, and verified the task.\"}},\"finish_reason\":null}}]}}\n\n"
    );
    let turn2_chunk2 = "data: {\"id\":\"c2\",\"object\":\"chat.completion.chunk\",\"created\":1677652289,\"model\":\"claude-3-5-sonnet-20241022\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":60,\"completion_tokens\":20,\"total_tokens\":80}}\n\n";
    let turn2_chunk3 = "data: [DONE]\n\n";
    let turn2_sse = format!("{turn2_chunk1}{turn2_chunk2}{turn2_chunk3}");

    let fallback_sse = "data: {\"id\":\"chatcmpl-fallback\",\"object\":\"chat.completion.chunk\",\"created\":1677652291,\"model\":\"claude-3-5-sonnet-20241022\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Task completed.\"},\"finish_reason\":null}]}\n\ndata: [DONE]\n\n";

    let mut responses = vec![(200, title_sse), (200, turn1_sse), (200, turn2_sse)];
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
        .env("HTTP_PROXY", "http://127.0.0.1:1")
        .env("HTTPS_PROXY", "http://127.0.0.1:1")
        .env("ALL_PROXY", "http://127.0.0.1:1")
        .env("NO_PROXY", "127.0.0.1,localhost")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    configure_isolated_process_group(&mut command);

    let mut child = command.spawn().expect("failed to spawn claude code child process");

    let stdout_handle = child.stdout.take().expect("child stdout should be piped");
    let stderr_handle = child.stderr.take().expect("child stderr should be piped");

    let stdout_task = tokio::spawn(async move {
        use tokio::io::AsyncReadExt as _;
        let mut buf = Vec::new();
        let mut reader = stdout_handle;
        let _res = reader.read_to_end(&mut buf).await;
        String::from_utf8_lossy(&buf).into_owned()
    });
    let stderr_task = tokio::spawn(async move {
        use tokio::io::AsyncReadExt as _;
        let mut buf = Vec::new();
        let mut reader = stderr_handle;
        let _res = reader.read_to_end(&mut buf).await;
        String::from_utf8_lossy(&buf).into_owned()
    });

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

    let stdout_str = stdout_task.await.unwrap_or_default();
    let stderr_str = stderr_task.await.unwrap_or_default();

    assert!(
        status.success(),
        "claude code child process should exit with status 0, got: {status:?}\nSTDOUT:\n{stdout_str}\nSTDERR:\n{stderr_str}"
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

    let task_posts: Vec<serde_json::Value> = post_requests
        .iter()
        .filter_map(|r| serde_json::from_str::<serde_json::Value>(&r.body).ok())
        .filter(|json| {
            let has_prompt = json["messages"].as_array().is_some_and(|msgs| {
                msgs.iter()
                    .any(|m| content_contains(&m["content"], "Inspect input.json"))
            });
            let is_main_turn = json["tools"].as_array().is_some_and(|t| !t.is_empty())
                || json["messages"]
                    .as_array()
                    .is_some_and(|msgs| msgs.iter().any(|m| m["role"].as_str() == Some("tool")));
            has_prompt && is_main_turn
        })
        .collect();

    assert!(
        task_posts.len() >= 2,
        "should capture at least 2 turns for main coding task, got: {}",
        task_posts.len()
    );

    let turn1_tools = task_posts[0]["tools"]
        .as_array()
        .expect("turn 1 request payload should contain 'tools' schema array");
    let has_bash_tool = turn1_tools
        .iter()
        .any(|t| t["function"]["name"].as_str() == Some("Bash"));
    assert!(has_bash_tool, "turn 1 request should present tool schema for 'Bash'");

    let turn2_messages = task_posts[1]["messages"]
        .as_array()
        .expect("turn 2 request payload should contain 'messages' array");

    let has_assistant_call = turn2_messages.iter().any(|m| {
        m["role"].as_str() == Some("assistant")
            && m["tool_calls"]
                .as_array()
                .is_some_and(|tc| tc.iter().any(|c| c["id"].as_str() == Some("call_bash_1")))
    });
    assert!(
        has_assistant_call,
        "turn 2 request should preserve assistant tool_call with stable ID 'call_bash_1'"
    );

    let has_tool_result = turn2_messages
        .iter()
        .any(|m| m["role"].as_str() == Some("tool") && m["tool_call_id"].as_str() == Some("call_bash_1"));
    assert!(
        has_tool_result,
        "turn 2 request should submit tool_result referencing stable call ID 'call_bash_1'"
    );

    workspace.assert_successful_completion();
}

#[cfg(unix)]
fn configure_isolated_process_group(command: &mut tokio::process::Command) {
    use std::os::unix::process::CommandExt as _;

    command.as_std_mut().process_group(0);
}

#[cfg(not(unix))]
fn configure_isolated_process_group(_command: &mut tokio::process::Command) {}

fn content_contains(val: &serde_json::Value, needle: &str) -> bool {
    if let Some(s) = val.as_str() {
        return s.contains(needle);
    }
    if let Some(arr) = val.as_array() {
        return arr
            .iter()
            .any(|item| item["text"].as_str().is_some_and(|t| t.contains(needle)));
    }
    false
}

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
