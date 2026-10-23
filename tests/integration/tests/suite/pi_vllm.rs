// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Black-box pi acceptance tests against a real vLLM backend.
//!
//! ```text
//! pi (openai-completions) ─► Praxis (anthropic/messages-native-vllm.yaml) ─► vLLM /v1/chat/completions
//! ```
//!
//! The CPU lane in [`super::pi`] already proves the documented `models.json`
//! and its fail-closed inversions against a scripted backend, including a
//! scripted tool-call round trip. What only a real model shows is that a
//! tool-calling conversation survives the round trip under real token timing,
//! with streamed `tool_calls` deltas arriving across arbitrary chunk boundaries
//! and a `role: "tool"` turn going back up. So this lane mirrors the OpenCode
//! one with the same two scenarios:
//!
//! * a plain text turn, proving the documented path reaches a real model and that no reasoning channel leaks into the
//!   user-visible answer; and
//! * a tool-call round trip, which is the scenario that earns GPU time.
//!
//! It reuses the OpenCode live gate, gateway patching, and hash-verified
//! workspace verbatim: same example config, same backend variables. It differs
//! in the pinned executable, its own namespace and require-live variables, and
//! in launching the client INSIDE the namespace with `ip netns exec`.
//!
//! # Gating
//!
//! All live tests return early unless the full environment is present. Because
//! a Rust test that early-returns reports as PASSED, [`REQUIRE_LIVE_ENV`] turns
//! a missing variable into a panic so a CI nightly cannot go green by skipping.
//!
//! ```console
//! PRAXIS_TEST_PI_BIN="$PWD/pi/pi" \
//! PRAXIS_TEST_VLLM_BASE_URL=http://127.0.0.1:8000 \
//! PRAXIS_TEST_VLLM_MODEL=qwen3-8b \
//! VLLM_API_KEY=<backend-token> \
//!   cargo test -p praxis-tests-integration --features store-all --test suite \
//!   pi_vllm::pi_vllm_documented_config_completes_text_turn -- --exact
//! ```
//!
//! Pins live in `tests/integration/fixtures/pi-cli/pin.toml`; the backend pins
//! are shared with OpenCode in `tests/integration/fixtures/opencode-cli/pin.toml`.
//! Both are consumed by the `pins` job in `.github/workflows/vllm-integration.yaml`.

use std::path::Path;

use praxis_test_utils::CapturedChildOutput;
use tempfile::TempDir;

use super::{
    harness::PI_PINNED_VERSION,
    live_vllm,
    opencode_vllm::{
        CHILD_TIMEOUT, LiveConfig, LiveEnv, REASONING_LEAK_MARKERS, TEXT_PROMPT, ToolWorkspace, live_config,
    },
    pi::{Lane, LaunchSpec, PI_BIN_ENV, assert_completed_turn, launch, pin_value},
};

/// Optional Linux network namespace to launch the client in.
const NETNS_ENV: &str = "PRAXIS_TEST_PI_NETNS";
/// Demands a real live run rather than a silent skip.
const REQUIRE_LIVE_ENV: &str = "PRAXIS_TEST_PI_REQUIRE_LIVE";

/// The variables that gate this lane.
const LIVE_ENV: LiveEnv<'static> = LiveEnv {
    client: "pi",
    bin: PI_BIN_ENV,
    netns: NETNS_ENV,
    require_live: REQUIRE_LIVE_ENV,
};

/// Drive one live turn through Praxis with the documented configuration.
async fn run_live(live: &LiveConfig, workspace: &Path, prompt: &str) -> CapturedChildOutput {
    let (proxy, _admin) = live_vllm::start_isolated_proxy(live.netns.as_deref(), |port| live_config(live, port));
    let base_url = format!("http://{}/v1", proxy.addr());
    let home = TempDir::new().expect("temporary HOME should be creatable");

    launch(LaunchSpec {
        bin: &live.client_bin,
        home: home.path(),
        workspace,
        base_url: &base_url,
        model: &live.model,
        lane: Lane::Documented,
        prompt,
        timeout: CHILD_TIMEOUT,
        netns: live.netns.as_deref(),
    })
    .await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pi_vllm_documented_config_completes_text_turn() {
    let Some(live) = LiveConfig::from_env_vars(&LIVE_ENV) else {
        return;
    };
    let workspace = TempDir::new().expect("temporary workspace should be creatable");

    let output = run_live(&live, workspace.path(), TEXT_PROMPT).await;
    let trace = assert_completed_turn(&output, "live pi text turn", CHILD_TIMEOUT);

    let answer = trace.assistant_text();
    assert!(
        !answer.trim().is_empty(),
        "the model should produce a user-visible answer, got {answer:?} (events: {:?})",
        trace.event_types()
    );
    // A reasoning parser mismatched to the served model family leaves the
    // think block in the assistant message instead of a separate channel; the
    // symptom is reasoning rendered to the user as the answer.
    for marker in REASONING_LEAK_MARKERS {
        assert!(
            !answer.contains(marker),
            "reasoning delimiter {marker:?} reached the user-visible answer; \
             check --reasoning-parser and --default-chat-template-kwargs against the served model: {answer:?}"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pi_vllm_tool_call_round_trip_writes_hash_verified_marker() {
    let Some(live) = LiveConfig::from_env_vars(&LIVE_ENV) else {
        return;
    };
    let workspace = ToolWorkspace::create();

    // The marker is only obtainable by reading the file, so a model that
    // answers from the prompt alone cannot satisfy the on-disk oracle.
    let tool = pin_value("bash_tool");
    let prompt = format!(
        "Use the {tool} tool. Run exactly: cp {source} {target}. \
         Then reply DONE. Do not print the file contents. /no_think",
        source = ToolWorkspace::SOURCE,
        target = ToolWorkspace::TARGET,
    );

    let output = run_live(&live, workspace.dir.path(), &prompt).await;
    let trace = assert_completed_turn(&output, "live pi tool-call round trip", CHILD_TIMEOUT);

    // Oracle A: on-disk end state, compared in-process against the retained
    // marker rather than anything the client could have rewritten.
    workspace.assert_marker_copied();

    // Oracle B: the client actually executed the shell tool, rather than the
    // file happening to exist.
    let executed = trace.executed_tools();
    assert!(
        executed.contains(&tool.as_str()),
        "the turn should execute the {tool} tool, got {executed:?} (events: {:?})",
        trace.event_types()
    );
}

// -----------------------------------------------------------------------------
// Offline tests
// -----------------------------------------------------------------------------

#[test]
fn live_pi_config_skips_cleanly_when_unconfigured() {
    // This process sets none of the live variables, so the gate must skip
    // rather than panic — otherwise every PR run would fail.
    assert!(
        !live_vllm::env_is_truthy(REQUIRE_LIVE_ENV),
        "{REQUIRE_LIVE_ENV} must not be set in a normal test run"
    );
    assert!(
        LiveConfig::from_env_vars(&LIVE_ENV).is_none(),
        "the live pi gate should skip when the environment is absent"
    );
}

#[test]
fn pinned_pi_version_matches_the_cpu_lane() {
    // Both lanes drive the same archive; a split pin would mean the nightly
    // validated a different build from the one PRs gate on.
    assert_eq!(
        pin_value("version"),
        PI_PINNED_VERSION,
        "the GPU and CPU lanes must pin the same pi version"
    );
}

#[test]
fn pi_gpu_lane_is_wired_in_ci() {
    // A test the workflow never names never runs, and still reports nothing.
    let workflow = super::opencode::repo_root().join(".github/workflows/vllm-integration.yaml");
    let yaml = std::fs::read_to_string(&workflow).expect("vLLM workflow should be readable");
    for needle in [
        PI_BIN_ENV,
        NETNS_ENV,
        REQUIRE_LIVE_ENV,
        "pi-cli/pin.toml",
        "vllm-gpu-pi-acceptance",
        "pi_vllm::pi_vllm_documented_config_completes_text_turn",
        "pi_vllm::pi_vllm_tool_call_round_trip_writes_hash_verified_marker",
    ] {
        assert!(
            yaml.contains(needle),
            "{} must contain {needle:?}, or the pi GPU lane is not really wired",
            workflow.display()
        );
    }
}
