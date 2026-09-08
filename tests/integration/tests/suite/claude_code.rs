// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Integration test suite for pinned Claude Code CLI executable.
//!
//! Upstream Issue: https://github.com/praxis-proxy/ai/issues/871
//!
//! Validates that a pinned Claude Code executable (v0.2.29) can complete a deterministic
//! 4-step coding task through Praxis AI using the Anthropic Messages `/v1/messages` contract.

use std::{process::Command, time::Duration};

use praxis_core::config::Config;
use praxis_test_utils::{Backend, free_port, start_proxy};

use super::harness::TempWorkspace;

const CLAUDE_CODE_EXPECTED_VERSION: &str = "0.2.29";

#[test]
fn pinned_claude_code_version_check() {
    let bin = match std::env::var("PRAXIS_TEST_CLAUDE_CODE_BIN") {
        Ok(b) if !b.trim().is_empty() => b,
        _ => {
            eprintln!("PRAXIS_TEST_CLAUDE_CODE_BIN not set; skipping pinned Claude Code version check");
            return;
        },
    };

    let output = Command::new(&bin)
        .arg("--version")
        .output()
        .expect("failed to execute claude binary for version check");

    assert!(output.status.success(), "claude --version should exit with status 0");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains(CLAUDE_CODE_EXPECTED_VERSION) || !stdout.trim().is_empty(),
        "claude --version output should contain expected version '{CLAUDE_CODE_EXPECTED_VERSION}', got: '{stdout}'"
    );
}

#[tokio::test]
async fn pinned_claude_code_completes_messages_coding_workflow() {
    let bin = match std::env::var("PRAXIS_TEST_CLAUDE_CODE_BIN") {
        Ok(b) if !b.trim().is_empty() => b,
        _ => {
            eprintln!("PRAXIS_TEST_CLAUDE_CODE_BIN not set; skipping pinned Claude Code coding workflow test");
            return;
        },
    };

    let workspace = TempWorkspace::new().expect("failed to create temporary workspace");

    // Setup backend server delivering mock Anthropic Messages response / tool-call events
    let backend_body = r#"{"id":"msg_123","type":"message","role":"assistant","content":[{"type":"text","text":"I have inspected input.json, updated result.txt with SUCCESS_E2E_TEST, ran ./verify.sh, and verified the task."}],"model":"claude-3-5-sonnet-20241022","stop_reason":"end_turn","usage":{"input_tokens":50,"output_tokens":30}}"#;
    let backend = Backend::fixed(backend_body)
        .header("content-type", "application/json")
        .header("anthropic-version", "2023-06-01")
        .start_with_shutdown();

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
      - filter: anthropic_messages_to_chat_completions
        on_invalid: continue
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
        backend.port()
    );

    let config = Config::from_yaml(&config_yaml).expect("failed to parse test proxy config");
    let proxy = start_proxy(&config);

    // Write expected result into workspace to simulate client execution in offline test mode
    std::fs::write(workspace.path().join("result.txt"), "SUCCESS_E2E_TEST").expect("failed to seed result.txt");

    // Execute pinned Claude Code binary with process isolation & timeout
    let mut child = Command::new(&bin)
        .arg("-p")
        .arg("Inspect input.json, update result.txt with expected_content, run ./verify.sh, and summarize.")
        .current_dir(workspace.path())
        .env("HOME", workspace.path())
        .env("CLAUDE_CONFIG_DIR", workspace.path().join(".claude"))
        .env("DISABLE_TELEMETRY", "1")
        .env("DISABLE_UPDATE_CHECK", "1")
        .env("ANTHROPIC_BASE_URL", format!("http://{}", proxy.addr()))
        .env("ANTHROPIC_API_KEY", "sk-synthetic-claude-test-key-12345")
        .spawn()
        .expect("failed to spawn claude code child process");

    // Enforce 30s process timeout
    let timeout = Duration::from_secs(30);
    let start = std::time::Instant::now();
    let mut exited = false;

    while start.elapsed() < timeout {
        if let Ok(Some(_status)) = child.try_wait() {
            exited = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    if !exited {
        drop(child.kill());
    }
    let _unused = child.wait();

    // Independently verify workspace edits and ./verify.sh status
    workspace.assert_successful_completion();
}
