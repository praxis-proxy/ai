// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Black-box OpenCode acceptance tests against a real vLLM backend.
//!
//! ```text
//! OpenCode ─► Praxis (anthropic/messages-native-vllm.yaml) ─► vLLM /v1/chat/completions
//! ```
//!
//! The CPU lane in [`super::opencode`] already proves both documented
//! credential recipes against a scripted backend, and
//! `suite/examples/anthropic_messages_native_vllm.rs` proves wire fidelity
//! deterministically. Neither can prove the one thing that only a real model
//! shows: that a multi-turn tool-calling conversation survives the round trip
//! under real token timing, with streamed `tool_calls` deltas arriving across
//! arbitrary chunk boundaries and a `role:"tool"` turn going back up.
//!
//! So this lane is deliberately two scenarios, not seven:
//!
//! * a plain text turn, proving the documented path reaches a real model and
//!   that no reasoning channel leaks into the user-visible answer; and
//! * a tool-call round trip, which is the scenario that earns GPU time.
//!
//! The `--pure`/model-headers lane stays on CPU: it is a config-parsing
//! concern, and a real model adds cost without adding evidence.
//!
//! # Gating
//!
//! All live tests return early unless the full environment is present. Because
//! a Rust test that early-returns reports as PASSED, [`REQUIRE_LIVE_ENV`] turns
//! a missing variable into a panic so a CI nightly cannot go green by skipping.
//! [`live_vllm::REQUIRE_EGRESS_ISOLATION_ENV`] does the same for the namespace.
//!
//! ```console
//! PRAXIS_TEST_OPENCODE_BIN="$HOME/.opencode/bin/opencode" \
//! PRAXIS_TEST_VLLM_BASE_URL=http://127.0.0.1:8000 \
//! PRAXIS_TEST_VLLM_MODEL=qwen3-8b \
//! VLLM_API_KEY=<backend-token> \
//!   cargo test -p praxis-tests-integration --features store-all --test suite \
//!   opencode_vllm::opencode_vllm_plugin_auth_completes_text_turn -- --exact
//! ```
//!
//! Pins live in `tests/integration/fixtures/opencode-cli/pin.toml` and are
//! consumed by the `pins` job in `.github/workflows/vllm-integration.yaml`.

use std::{
    ffi::OsString,
    net::{IpAddr, Ipv4Addr},
    path::Path,
    time::Duration,
};

use praxis_core::config::Config;
use praxis_test_utils::{CapturedChildOutput, example_config_path};
use tempfile::TempDir;

use super::{
    harness::{OPENCODE_PINNED_VERSION, SHA256_DIGEST_SH},
    live_vllm::{
        self, BACKEND_TOKEN_ENV, LISTEN_ADDRESS_ENV, VLLM_BASE_URL_ENV, VLLM_MODEL_ENV, authority_of, env_is_truthy,
        non_empty_env,
    },
    opencode::{
        AuthLane, CHAT_PATH, EXAMPLE_CONFIG, GATEWAY_USER, LaunchSpec, OpenCodeTrace, PLUGIN_FAILURE_LOG,
        gateway_password, launch,
    },
};

// -----------------------------------------------------------------------------
// Pins and constants
// -----------------------------------------------------------------------------

/// Path to the pinned OpenCode executable.
const OPENCODE_BIN_ENV: &str = "PRAXIS_TEST_OPENCODE_BIN";
/// Optional Linux network namespace to launch the client in.
const NETNS_ENV: &str = "PRAXIS_TEST_OPENCODE_NETNS";
/// Demands a real live run rather than a silent skip.
const REQUIRE_LIVE_ENV: &str = "PRAXIS_TEST_OPENCODE_REQUIRE_LIVE";

/// Bound on one live turn. Generous: a cold model plus several tool turns.
const CHILD_TIMEOUT: Duration = Duration::from_secs(300);

/// Prompt for the plain text scenario.
///
/// `/no_think` suppresses Qwen3 reasoning from the user turn. Unlike the Claude
/// planning scenario, this test is about the transport, not about proving the
/// server-side reasoning configuration, so suppressing it here is deliberate.
const TEXT_PROMPT: &str = "Reply with exactly the single word PONG and nothing else. \
                           Do not call any tools. /no_think";

/// Reasoning delimiters that must never reach the user-visible answer.
const REASONING_LEAK_MARKERS: &[&str] = &["<think>", "</think>"];

// -----------------------------------------------------------------------------
// Live gate
// -----------------------------------------------------------------------------

/// Resolved live environment for one acceptance run.
struct LiveConfig {
    opencode_bin: OsString,
    vllm_authority: String,
    model: String,
    listen_address: IpAddr,
    netns: Option<String>,
}

impl LiveConfig {
    /// Resolve the live environment, or `None` when the lane is unconfigured.
    ///
    /// When [`REQUIRE_LIVE_ENV`] is truthy a missing variable is a hard failure
    /// instead of a skip, because an early return would otherwise report as a
    /// passing test — for instance if `sudo --preserve-env` dropped one.
    fn from_env() -> Option<Self> {
        let opencode_bin = std::env::var_os(OPENCODE_BIN_ENV).filter(|value| !value.is_empty());
        let base_url = non_empty_env(VLLM_BASE_URL_ENV);
        let model = non_empty_env(VLLM_MODEL_ENV);
        let token = non_empty_env(BACKEND_TOKEN_ENV);

        let (Some(opencode_bin), Some(base_url), Some(model), Some(_token)) = (opencode_bin, base_url, model, token)
        else {
            assert!(
                !env_is_truthy(REQUIRE_LIVE_ENV),
                "{REQUIRE_LIVE_ENV} is set but a required variable is missing; set all of \
                 {OPENCODE_BIN_ENV}, {VLLM_BASE_URL_ENV}, {VLLM_MODEL_ENV}, and {BACKEND_TOKEN_ENV}"
            );
            eprintln!(
                "skipping live OpenCode acceptance test; set {OPENCODE_BIN_ENV}, \
                 {VLLM_BASE_URL_ENV}, {VLLM_MODEL_ENV}, and {BACKEND_TOKEN_ENV} to run it"
            );
            return None;
        };

        let listen_address = non_empty_env(LISTEN_ADDRESS_ENV)
            .map(|value| {
                value
                    .parse::<IpAddr>()
                    .unwrap_or_else(|error| panic!("{LISTEN_ADDRESS_ENV} must be an IP address: {error}"))
            })
            .unwrap_or(IpAddr::V4(Ipv4Addr::LOCALHOST));

        let config = Self {
            opencode_bin,
            vllm_authority: authority_of(&base_url),
            model,
            listen_address,
            netns: non_empty_env(NETNS_ENV),
        };
        live_vllm::require_egress_isolation_if_demanded(config.netns.as_deref(), NETNS_ENV);
        Some(config)
    }
}

/// Build the shipped example config against the live backend.
///
/// Patches the listener, the backend authority, and the gateway password. The
/// backend credential keeps its `VLLM_API_KEY` reference: unlike the CPU lane,
/// the variable really is set here, and leaving it in place is what proves
/// Praxis injects the server-owned token the real backend demands.
fn live_config(live: &LiveConfig, proxy_port: u16) -> Config {
    let path = example_config_path(EXAMPLE_CONFIG);
    let yaml = std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {path}: {error}"));
    let listener = format!("{}:{proxy_port}", live.listen_address);
    let patched = live_vllm::replace_once(&yaml, "127.0.0.1:8080", &listener, "listener", EXAMPLE_CONFIG);
    let patched = live_vllm::replace_once(
        &patched,
        "127.0.0.1:8000",
        &live.vllm_authority,
        "backend endpoint",
        EXAMPLE_CONFIG,
    );
    let patched = live_vllm::replace_once(
        &patched,
        "env_var: GATEWAY_AUTH_PASSWORD",
        &format!("password: {}", gateway_password()),
        "gateway password",
        EXAMPLE_CONFIG,
    );
    Config::from_yaml(&patched).unwrap_or_else(|error| panic!("parse patched {EXAMPLE_CONFIG}: {error}"))
}

/// Drive one live turn through Praxis and return the captured output.
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
        lane: AuthLane::Plugin,
        prompt,
        timeout: CHILD_TIMEOUT,
    })
    .await
}

/// Shared post-run checks: the client finished, the plugin loaded, no errors.
///
/// The plugin check is not redundant with the exit status. A throwing `config`
/// hook still exits 0 and still sends the turn, just unauthenticated, so
/// without this a broken plugin would surface only as a confusing 401.
fn assert_clean_run(output: &CapturedChildOutput, scenario: &str) -> OpenCodeTrace {
    live_vllm::assert_not_timed_out(output, "OpenCode", scenario, CHILD_TIMEOUT);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let trace = OpenCodeTrace::parse(&stdout);

    assert!(
        !stderr.contains(PLUGIN_FAILURE_LOG),
        "{scenario}: the documented plugin must load cleanly; found {PLUGIN_FAILURE_LOG:?}\n{stderr}"
    );
    // Diagnose from the trace before the exit status: a turn that failed for a
    // protocol reason is far more informative than a bare non-zero code.
    assert!(
        trace.errors().is_empty(),
        "{scenario}: client reported an error: {:?}\nSTDERR:\n{stderr}",
        trace.errors()
    );
    assert!(
        output.status.success(),
        "{scenario}: OpenCode exited {:?}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}",
        output.status.code()
    );
    trace
}

// -----------------------------------------------------------------------------
// Scenarios
// -----------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn opencode_vllm_plugin_auth_completes_text_turn() {
    let Some(live) = LiveConfig::from_env() else { return };
    let workspace = TempDir::new().expect("temporary workspace should be creatable");

    let output = run_live(&live, workspace.path(), TEXT_PROMPT).await;
    let trace = assert_clean_run(&output, "live text turn");

    let answer = trace.assistant_text();
    assert!(
        !answer.trim().is_empty(),
        "the model should produce a user-visible answer, got {:?} (events: {:?})",
        answer,
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
async fn opencode_vllm_tool_call_round_trip_writes_hash_verified_marker() {
    let Some(live) = LiveConfig::from_env() else { return };
    let workspace = ToolWorkspace::create();

    // The marker is only obtainable by reading the file, so a model that
    // answers from the prompt alone cannot satisfy the on-disk oracle.
    let prompt = format!(
        "Use the bash tool. Run exactly: cp {source} {target}. \
         Then reply DONE. Do not print the file contents. /no_think",
        source = ToolWorkspace::SOURCE,
        target = ToolWorkspace::TARGET,
    );

    let output = run_live(&live, workspace.dir.path(), &prompt).await;
    let trace = assert_clean_run(&output, "live tool-call round trip");

    // Oracle A: on-disk end state, compared in-process against the retained
    // marker rather than trusting the workspace's own verify script, which the
    // client could have overwritten.
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
// Workspace
// -----------------------------------------------------------------------------

/// Workspace whose success condition requires actually running a tool.
///
/// The marker is high-entropy and lives only in `source.txt`, so it cannot be
/// guessed from the prompt. Its sha256 — never the marker itself — is written
/// to `expected.hash`, so a model that reads the verifier learns nothing.
struct ToolWorkspace {
    dir: TempDir,
    marker: String,
}

impl ToolWorkspace {
    const SOURCE: &'static str = "source.txt";
    const TARGET: &'static str = "result.txt";

    fn create() -> Self {
        let dir = TempDir::new().expect("temporary workspace should be creatable");
        let marker = format!("OPENCODE_MARKER_{:016x}", u64::from_le_bytes(rand::random::<[u8; 8]>()));
        std::fs::write(dir.path().join(Self::SOURCE), &marker).expect("source marker should be writable");
        std::fs::write(dir.path().join(Self::TARGET), "").expect("result placeholder should be writable");

        // Precompute with the same pipeline the harness compares against, so
        // the marker never appears on disk outside source.txt.
        let digest = std::process::Command::new("sh")
            .arg("-c")
            .arg(format!("tr -d '[:space:]' < {} | {SHA256_DIGEST_SH}", Self::SOURCE))
            .current_dir(dir.path())
            .output()
            .expect("sha256 digest should run");
        assert!(digest.status.success(), "sha256 digest should succeed");
        let hash = String::from_utf8_lossy(&digest.stdout).trim().to_owned();
        assert!(!hash.is_empty(), "sha256 digest should be non-empty");
        std::fs::write(dir.path().join("expected.hash"), format!("{hash}\n")).expect("hash should be writable");

        Self { dir, marker }
    }

    /// Assert the tool run reproduced the marker into the target file.
    fn assert_marker_copied(&self) {
        let content =
            std::fs::read_to_string(self.dir.path().join(Self::TARGET)).expect("result file should be readable");
        let normalized: String = content.chars().filter(|c| !c.is_whitespace()).collect();
        assert_eq!(
            normalized,
            self.marker,
            "{} must contain exactly the marker from {}; got {content:?}",
            Self::TARGET,
            Self::SOURCE
        );
    }
}

// -----------------------------------------------------------------------------
// Offline tests
// -----------------------------------------------------------------------------

#[test]
fn live_config_skips_cleanly_when_unconfigured() {
    // This process sets none of the live variables, so the gate must skip
    // rather than panic — otherwise every PR run would fail.
    assert!(
        !env_is_truthy(REQUIRE_LIVE_ENV),
        "{REQUIRE_LIVE_ENV} must not be set in a normal test run"
    );
    assert!(
        LiveConfig::from_env().is_none(),
        "the live gate should skip when the environment is absent"
    );
}

#[test]
fn live_config_patches_the_shipped_example() {
    // Exercises all three drift-guarded anchors against the real file, so an
    // example change breaks here rather than inside a nightly GPU run.
    let live = LiveConfig {
        opencode_bin: OsString::from("/nonexistent/opencode"),
        vllm_authority: "10.201.0.3:8000".to_owned(),
        model: "qwen3-8b".to_owned(),
        listen_address: IpAddr::V4(Ipv4Addr::new(10, 201, 0, 1)),
        netns: None,
    };
    let config = live_config(&live, 28_080);

    let listener = &config.listeners[0];
    assert!(
        listener.address.contains("10.201.0.1:28080"),
        "the listener should bind the configured address, got {}",
        listener.address
    );

    let chain = &config.filter_chains[0];
    let router = chain
        .filters
        .iter()
        .find(|filter| filter.filter_type == "router")
        .expect("example chain should contain a router");
    assert!(
        router.config["routes"]
            .as_sequence()
            .expect("router should declare routes")
            .iter()
            .any(|route| route["path"].as_str() == Some(CHAT_PATH)),
        "the live lane depends on an explicit {CHAT_PATH} route"
    );

    let basic_auth = chain
        .filters
        .iter()
        .find(|filter| filter.filter_type == "basic_auth")
        .expect("example chain should gate on basic_auth");
    assert_eq!(
        basic_auth.config["credentials"][0]["username"].as_str(),
        Some(GATEWAY_USER),
        "the gateway username must match what the client presents"
    );
    // The backend credential keeps its env-var reference on this lane.
    let credential = chain
        .filters
        .iter()
        .find(|filter| filter.filter_type == "credential_injection")
        .expect("example chain should inject a backend credential");
    assert_eq!(
        credential.config["clusters"][0]["env_var"].as_str(),
        Some(BACKEND_TOKEN_ENV),
        "the live lane must inject the real backend token"
    );
}

#[test]
fn tool_workspace_hides_the_marker_from_the_verifier() {
    let workspace = ToolWorkspace::create();
    let hash = std::fs::read_to_string(workspace.dir.path().join("expected.hash")).expect("hash should be readable");

    assert!(
        !hash.contains(&workspace.marker),
        "expected.hash must not leak the marker it verifies"
    );
    assert!(
        workspace.marker.starts_with("OPENCODE_MARKER_"),
        "marker should be identifiable in diagnostics: {}",
        workspace.marker
    );
    // Distinct per run, so a stale result file from an earlier run cannot
    // satisfy a later one.
    assert_ne!(
        workspace.marker,
        ToolWorkspace::create().marker,
        "each workspace should mint a fresh marker"
    );
}

#[test]
fn tool_workspace_oracle_rejects_an_unwritten_target() {
    let workspace = ToolWorkspace::create();
    // The target starts empty, so the oracle must fail before any tool runs.
    // Without this the scenario could pass on a client that did nothing.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| workspace.assert_marker_copied()));
    assert!(result.is_err(), "the oracle must reject an untouched workspace");
}

#[test]
fn pinned_version_matches_the_cpu_lane() {
    // Both lanes drive the same executable; a split pin would mean the nightly
    // validated a different build from the one PRs gate on.
    assert_eq!(
        super::opencode::pin_value("version"),
        OPENCODE_PINNED_VERSION,
        "the GPU and CPU lanes must pin the same OpenCode version"
    );
}
