// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Black-box OpenCode 2.x acceptance tests against a real vLLM backend.
//!
//! ```text
//! OpenCode 2.x ─► Praxis (anthropic/messages-native-vllm.yaml) ─► vLLM /v1/chat/completions
//! ```
//!
//! The 2.x counterpart of [`super::opencode_vllm`], with the same two
//! scenarios for the same reason: a plain text turn proving the documented
//! path reaches a real model with no reasoning leak, and a tool-call round
//! trip, which is the scenario that earns GPU time. The provider-headers and
//! inversion lanes stay on CPU in [`super::opencode_v2`]; a real model adds
//! cost there without adding evidence.
//!
//! These run in the SAME CI job as the 1.x scenarios, against the same vLLM
//! container and network namespace, so they reuse the 1.x live gate and
//! workspace verbatim and differ only in which pinned binary they launch,
//! [`OPENCODE_V2_BIN_ENV`], and in the 2.x tool name, `shell`.
//!
//! ```console
//! PRAXIS_TEST_OPENCODE_V2_BIN=/path/to/package/bin/opencode \
//! PRAXIS_TEST_VLLM_BASE_URL=http://127.0.0.1:8000 \
//! PRAXIS_TEST_VLLM_MODEL=qwen3-8b \
//! VLLM_API_KEY=<backend-token> \
//!   cargo test -p praxis-tests-integration --features store-all --test suite \
//!   opencode_v2_vllm::opencode_v2_vllm_plugin_auth_completes_text_turn -- --exact
//! ```
//!
//! Pins live in `tests/integration/fixtures/opencode-cli-v2/pin.toml` and are
//! consumed by the `pins` job in `.github/workflows/vllm-integration.yaml`.

use std::path::Path;

use praxis_test_utils::CapturedChildOutput;
use tempfile::TempDir;

use super::{
    harness::OPENCODE_V2_PINNED_VERSION,
    live_vllm,
    opencode::OpenCodeTrace,
    opencode_v2::{Lane, LaunchSpec, OPENCODE_V2_BIN_ENV, PLUGIN_FAILURE_LOG, launch, pin_value},
    opencode_vllm::{
        CHILD_TIMEOUT, LiveConfig, REASONING_LEAK_MARKERS, REQUIRE_LIVE_ENV, TEXT_PROMPT, ToolWorkspace, live_config,
    },
};

/// Drive one live 2.x turn through Praxis and return the captured output.
async fn run_live(live: &LiveConfig, workspace: &Path, prompt: &str) -> CapturedChildOutput {
    let (proxy, _admin) = live_vllm::start_isolated_proxy(live.netns.as_deref(), |port| live_config(live, port));
    let base_url = format!("http://{}/v1", proxy.addr());
    let home = TempDir::new().expect("temporary HOME should be creatable");

    launch(LaunchSpec {
        bin: &live.opencode_bin,
        home: home.path(),
        workspace,
        base_url: &base_url,
        model: &live.model,
        lane: Lane::Plugin,
        prompt,
        timeout: CHILD_TIMEOUT,
    })
    .await
}

/// Shared post-run checks: the client finished, the plugin loaded, no errors.
///
/// The plugin check is not redundant with the exit status. A 2.x plugin that
/// fails to LOAD still exits 0 and still sends the turn, just unauthenticated,
/// so without this a broken plugin would surface only as a confusing 401.
fn assert_clean_run(output: &CapturedChildOutput, scenario: &str) -> OpenCodeTrace {
    live_vllm::assert_not_timed_out(output, "OpenCode 2.x", scenario, CHILD_TIMEOUT);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let trace = OpenCodeTrace::parse(&stdout);

    assert!(
        !stderr.contains(PLUGIN_FAILURE_LOG),
        "{scenario}: the documented plugin must load cleanly; found {PLUGIN_FAILURE_LOG:?}\n{stderr}"
    );
    assert!(
        trace.errors().is_empty(),
        "{scenario}: client reported an error: {:?}\nSTDERR:\n{stderr}",
        trace.errors()
    );
    assert!(
        output.status.success(),
        "{scenario}: OpenCode 2.x exited {:?}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}",
        output.status.code()
    );
    trace
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn opencode_v2_vllm_plugin_auth_completes_text_turn() {
    let Some(live) = LiveConfig::from_env_for(OPENCODE_V2_BIN_ENV) else {
        return;
    };
    let workspace = TempDir::new().expect("temporary workspace should be creatable");

    let output = run_live(&live, workspace.path(), TEXT_PROMPT).await;
    let trace = assert_clean_run(&output, "live 2.x text turn");

    let answer = trace.assistant_text();
    assert!(
        !answer.trim().is_empty(),
        "the model should produce a user-visible answer, got {:?} (events: {:?})",
        answer,
        trace.event_types()
    );
    for marker in REASONING_LEAK_MARKERS {
        assert!(
            !answer.contains(marker),
            "reasoning delimiter {marker:?} reached the user-visible answer; \
             check --reasoning-parser and --default-chat-template-kwargs against the served model: {answer:?}"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn opencode_v2_vllm_tool_call_round_trip_writes_hash_verified_marker() {
    let Some(live) = LiveConfig::from_env_for(OPENCODE_V2_BIN_ENV) else {
        return;
    };
    let workspace = ToolWorkspace::create();

    // 2.x renamed the 1.x `bash` tool to `shell`; naming the wrong tool would
    // leave the model to guess.
    let prompt = format!(
        "Use the {tool} tool. Run exactly: cp {source} {target}. \
         Then reply DONE. Do not print the file contents. /no_think",
        tool = pin_value("shell_tool"),
        source = ToolWorkspace::SOURCE,
        target = ToolWorkspace::TARGET,
    );

    let output = run_live(&live, workspace.dir.path(), &prompt).await;
    let trace = assert_clean_run(&output, "live 2.x tool-call round trip");

    // Oracle A: on-disk end state, compared in-process against the retained
    // marker rather than anything the client could have rewritten.
    workspace.assert_marker_copied();

    // Oracle B: the client actually invoked a tool, rather than the file
    // happening to exist. Lenient about event shape, strict about the fact.
    let types = trace.event_types();
    assert!(
        types.iter().any(|event| event.contains("tool")),
        "the turn should contain at least one tool event, got {types:?}"
    );
}

// -----------------------------------------------------------------------------
// Offline tests
// -----------------------------------------------------------------------------

#[test]
fn live_v2_config_skips_cleanly_when_unconfigured() {
    // This process sets none of the live variables, so the gate must skip
    // rather than panic — otherwise every PR run would fail.
    assert!(
        !live_vllm::env_is_truthy(REQUIRE_LIVE_ENV),
        "{REQUIRE_LIVE_ENV} must not be set in a normal test run"
    );
    assert!(
        LiveConfig::from_env_for(OPENCODE_V2_BIN_ENV).is_none(),
        "the live 2.x gate should skip when the environment is absent"
    );
}

#[test]
fn pinned_v2_version_matches_the_cpu_lane() {
    // Both 2.x lanes drive the same executable; a split pin would mean the
    // nightly validated a different build from the one PRs gate on.
    assert_eq!(
        pin_value("version"),
        OPENCODE_V2_PINNED_VERSION,
        "the GPU and CPU 2.x lanes must pin the same OpenCode version"
    );
}

#[test]
fn opencode_v2_gpu_lane_is_wired_in_ci() {
    // A test the workflow never names never runs, and still reports nothing.
    let workflow = super::opencode::repo_root().join(".github/workflows/vllm-integration.yaml");
    let yaml = std::fs::read_to_string(&workflow).expect("vLLM workflow should be readable");
    for needle in [
        OPENCODE_V2_BIN_ENV,
        "opencode-cli-v2/pin.toml",
        "opencode_v2_vllm::opencode_v2_vllm_plugin_auth_completes_text_turn",
        "opencode_v2_vllm::opencode_v2_vllm_tool_call_round_trip_writes_hash_verified_marker",
    ] {
        assert!(
            yaml.contains(needle),
            "{} must contain {needle:?}, or the OpenCode 2.x GPU lane is not really wired",
            workflow.display()
        );
    }
}
