// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Black-box Claude Code acceptance test for native Anthropic Messages
//! passthrough to a real vLLM backend (issue #1025).
//!
//! A pinned real Claude Code executable completes a deterministic multi-step
//! coding task while Praxis routes NATIVE Anthropic Messages traffic
//! (`/v1/messages`, `/v1/messages/count_tokens`, `/v1/models`) straight to a
//! vLLM backend that serves the Anthropic Messages API natively — with NO
//! `anthropic_messages_to_chat_completions` translation:
//!
//! ```text
//! Claude Code ─► Praxis (messages-native-vllm.yaml) ─► vLLM
//! ```
//!
//! This test asserts only what a live end-to-end run uniquely proves: the real
//! client completes the task through Praxis against a real backend. Wire
//! fidelity — native passthrough (no `chat/completions` reshaping), credential
//! isolation, and streaming semantics — is proven deterministically against
//! controlled fake backends in
//! `tests/integration/tests/suite/examples/anthropic_messages_native_vllm.rs`,
//! not observed here.
//!
//! This test is gated on live infrastructure and skips unless every required
//! variable is set. Run it locally with, e.g.:
//!
//! ```console
//! PRAXIS_TEST_CLAUDE_CODE_BIN=/absolute/path/to/claude \
//! PRAXIS_TEST_VLLM_BASE_URL=http://127.0.0.1:8000 \
//! PRAXIS_TEST_VLLM_MODEL=<exact-served-model-name> \
//! VLLM_API_KEY=<backend-bearer-token> \
//!   cargo test -p praxis-tests-integration --test suite \
//!   claude_code_vllm::pinned_claude_code_drives_native_vllm_through_full_flow -- --exact
//! ```
//!
//! Pin discipline: [`CLAUDE_CODE_VERSION`] and [`LAUNCH_FLAGS`] are part of the
//! committed pin manifest (`tests/integration/fixtures/claude-code-cli/`). They
//! MUST be re-validated against the pinned executable during the qualification
//! run described in that manifest before CI executes this test once. The served
//! model, vLLM image digest, and startup request matrix are pinned there too.

use std::{
    ffi::{OsStr, OsString},
    net::{IpAddr, Ipv4Addr},
    path::{Path, PathBuf},
    process::Stdio,
    time::{Duration, SystemTime},
};

use praxis_core::config::Config;
use praxis_test_utils::{
    CapturedChildOutput, basic_auth_header, capture_child_output, configure_isolated_process_group,
    example_config_path, free_port, start_proxy,
};
use serde_json::Value;

// -----------------------------------------------------------------------------
// Pins and constants
// -----------------------------------------------------------------------------

/// Environment variable holding the absolute path to the pinned Claude Code binary.
const CLAUDE_CODE_BIN_ENV: &str = "PRAXIS_TEST_CLAUDE_CODE_BIN";
/// Environment variable holding the native-Anthropic vLLM base URL or authority.
const VLLM_BASE_URL_ENV: &str = "PRAXIS_TEST_VLLM_BASE_URL";
/// Environment variable holding the exact served vLLM model name.
const VLLM_MODEL_ENV: &str = "PRAXIS_TEST_VLLM_MODEL";
/// Environment variable holding the backend bearer token injected by Praxis.
const BACKEND_TOKEN_ENV: &str = "VLLM_API_KEY";
/// Optional environment variable naming a Linux network namespace to launch in.
const NETNS_ENV: &str = "PRAXIS_TEST_CLAUDE_CODE_NETNS";
/// Optional environment variable demanding enforced egress isolation.
///
/// When truthy (`1`/`true`), the test refuses to run without a configured
/// [`NETNS_ENV`] namespace, so a CI acceptance run cannot silently degrade to
/// the advisory-only, non-isolated path. The client's only route is then the
/// veth to Praxis, which the test verifies actively.
const REQUIRE_EGRESS_ISOLATION_ENV: &str = "PRAXIS_TEST_REQUIRE_EGRESS_ISOLATION";
/// Optional environment variable overriding the address Praxis binds.
///
/// Under network isolation the client lives in a namespace and reaches Praxis
/// over a veth pair, so Praxis must bind the host-side veth address rather than
/// loopback, and the namespaced client can then reach only Praxis. Defaults to
/// `127.0.0.1`.
const LISTEN_ADDRESS_ENV: &str = "PRAXIS_TEST_LISTEN_ADDRESS";

/// Pinned Claude Code version substring expected from `claude --version`.
///
/// PIN: confirm the exact string against the pinned executable during the
/// qualification run and update this constant and the manifest together.
const CLAUDE_CODE_VERSION: &str = "2.0.1";

/// The native-vLLM passthrough example config under test.
const CONFIG: &str = "anthropic/messages-native-vllm.yaml";

/// The client's native Anthropic `x-api-key`. Distinct from the gateway
/// credential; the config's `headers` filter must strip it before vLLM.
const NATIVE_API_KEY: &str = "praxis-native-vllm-anthropic-key-do-not-forward";

/// The gateway `basic_auth` username the example config authenticates.
const GATEWAY_USER: &str = "gateway";

/// The gateway `basic_auth` password the client presents via
/// `ANTHROPIC_CUSTOM_HEADERS` and that `basic_auth` must strip before vLLM.
///
/// Drawn once per test process from the OS RNG rather than a source literal:
/// it is a throwaway secret scoped to this run, and both the in-process gateway
/// config and the client read the same value so they agree within a run.
fn gateway_password() -> &'static str {
    static PASSWORD: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    PASSWORD.get_or_init(random_token)
}

/// The deterministic coding-task prompt; the required value lives only on disk.
const PROMPT: &str = "Read the file `source/value.txt`. Write its contents converted to UPPERCASE \
     (with no surrounding whitespace) into `result/value.txt`. Then run `./verify.sh`. \
     Finally, summarize what you changed. The current value in `result/value.txt` is a \
     placeholder and must be replaced.";

/// Hard timeout bounding the whole child run, matching the CI documented bound.
const CHILD_TIMEOUT: Duration = Duration::from_secs(180);

/// Pinned Claude Code launch flags (excluding `-p`, the prompt, and `--model`).
///
/// PIN: these are the real print-mode headless flags accepted by the pinned
/// executable. The #1025 design sketch listed an APPROXIMATE set (`--bare`,
/// `--tools`, `--permission-mode dontAsk`, `--no-session-persistence`) that does
/// not match real Claude Code flags; do NOT reintroduce those. Re-validate the
/// full set below against the pinned executable during qualification and adjust
/// here and in the manifest together.
const LAUNCH_FLAGS: &[&str] = &[
    "--permission-mode",
    "acceptEdits",
    "--strict-mcp-config",
    "--output-format",
    "stream-json",
    "--verbose",
    "--max-turns",
    "8",
];

// -----------------------------------------------------------------------------
// Test
// -----------------------------------------------------------------------------

/// Prove the pinned Claude Code client completes a coding task through Praxis
/// against a real native-Anthropic vLLM backend.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pinned_claude_code_drives_native_vllm_through_full_flow() {
    let Some(live) = LiveConfig::from_env() else {
        return;
    };
    assert_pinned_version(&live.claude_bin).await;
    live.require_egress_isolation_if_demanded();

    // Praxis binds the configured address and forwards native Anthropic Messages
    // traffic straight to the real vLLM backend. Under network isolation Praxis
    // binds the host-side veth address so the namespaced client can reach only
    // Praxis, proving it cannot bypass the proxy.
    let proxy_port = free_port();
    let config = native_vllm_config(&live, proxy_port);
    let proxy = start_proxy(&config);
    let proxy_base_url = format!("http://{}", proxy.addr());

    // Enforced egress isolation: when a namespace is configured, actively prove
    // the isolated client can reach Praxis and CANNOT reach the public internet,
    // so all Anthropic traffic is forced through the proxy. This is real
    // enforcement, not an advisory `ANTHROPIC_BASE_URL` that a client is free to
    // ignore.
    if let Some(namespace) = &live.netns {
        verify_egress_isolation(namespace, live.listen_address, proxy_port);
    }

    let workspace = Workspace::create();
    let started = SystemTime::now();
    let output = launch_claude_code(&live, &proxy_base_url, &workspace).await;

    assert!(
        !output.timed_out,
        "Claude Code exceeded the {CHILD_TIMEOUT:?} acceptance-test timeout\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    assert!(
        output.status.success(),
        "Claude Code exited with {status:?}\nstdout:\n{stdout}\nstderr:\n{stderr}",
        status = output.status.code(),
        stdout = String::from_utf8_lossy(&output.stdout),
        stderr = String::from_utf8_lossy(&output.stderr),
    );

    // Task outcome, proven two independent ways:
    //  * end state — the exact derived file plus the harness-owned verification marker, written only when `verify.sh`
    //    confirms the compare; and
    //  * the work itself — the client's stream-json tool trace shows it Read the distinct input (receiving the per-run
    //    token back through Praxis), performed the distinct Edit writing the uppercase transform, and emitted a final
    //    summary.
    workspace.assert_task_completed(started);
    workspace.assert_task_trace(&String::from_utf8_lossy(&output.stdout));
}

// -----------------------------------------------------------------------------
// Live configuration gate
// -----------------------------------------------------------------------------

/// Resolved live-infrastructure configuration for one acceptance run.
struct LiveConfig {
    /// Absolute path to the pinned Claude Code executable.
    claude_bin: OsString,
    /// The `host:port` authority of the native-Anthropic vLLM backend.
    vllm_authority: String,
    /// The exact served vLLM model name pinned across every model surface.
    model: String,
    /// The address Praxis binds so the client can reach it.
    listen_address: IpAddr,
    /// Optional Linux network namespace to launch the child inside.
    netns: Option<String>,
}

impl LiveConfig {
    /// Reads the live gate from the environment, returning `None` (with an
    /// explanatory skip message) when any required variable is unset.
    fn from_env() -> Option<Self> {
        let claude_bin = std::env::var_os(CLAUDE_CODE_BIN_ENV);
        let vllm_base = std::env::var(VLLM_BASE_URL_ENV).ok();
        let model = std::env::var(VLLM_MODEL_ENV).ok();
        let backend_token = std::env::var(BACKEND_TOKEN_ENV).ok();

        let (Some(claude_bin), Some(vllm_base), Some(model), Some(_)) = (claude_bin, vllm_base, model, backend_token)
        else {
            eprintln!(
                "skipping native-vLLM Claude Code acceptance test; set {CLAUDE_CODE_BIN_ENV}, \
                 {VLLM_BASE_URL_ENV}, {VLLM_MODEL_ENV}, and {BACKEND_TOKEN_ENV} to run it"
            );
            return None;
        };

        let listen_address = std::env::var(LISTEN_ADDRESS_ENV)
            .ok()
            .and_then(|value| value.parse::<IpAddr>().ok())
            .unwrap_or(IpAddr::V4(Ipv4Addr::LOCALHOST));

        Some(Self {
            claude_bin,
            vllm_authority: authority_of(&vllm_base),
            model,
            listen_address,
            netns: std::env::var(NETNS_ENV).ok().filter(|value| !value.is_empty()),
        })
    }

    /// Panics if enforced egress isolation is demanded but no namespace is set.
    ///
    /// This turns the isolation contract into a hard gate for CI: with
    /// [`REQUIRE_EGRESS_ISOLATION_ENV`] truthy, the acceptance run must be
    /// network-isolated or fail loudly, never fall back to trusting the client
    /// to honour `ANTHROPIC_BASE_URL`.
    fn require_egress_isolation_if_demanded(&self) {
        if env_is_truthy(REQUIRE_EGRESS_ISOLATION_ENV) {
            assert!(
                self.netns.is_some(),
                "{REQUIRE_EGRESS_ISOLATION_ENV} is set but {NETNS_ENV} is not; \
                 egress isolation cannot be enforced without a network namespace"
            );
        }
    }
}

/// Reports whether an environment variable is set to a truthy value.
fn env_is_truthy(name: &str) -> bool {
    std::env::var(name)
        .map(|value| matches!(value.trim(), "1" | "true" | "TRUE" | "yes" | "on"))
        .unwrap_or(false)
}

/// Extracts a `host:port` authority from a base URL or an already-bare authority.
fn authority_of(base: &str) -> String {
    base.trim()
        .trim_start_matches("http://")
        .trim_start_matches("https://")
        .trim_end_matches('/')
        .to_owned()
}

/// Build the native-vLLM config with the listener and backend endpoint patched.
///
/// The listener binds [`LiveConfig::listen_address`] (loopback by default, the
/// host-side veth address under network isolation) and the backend endpoint is
/// repointed at the real vLLM authority so Praxis forwards straight to it.
/// `credential_injection` resolves `VLLM_API_KEY` from the environment at
/// pipeline build; the live gate guarantees it is set. The gateway `basic_auth`
/// password is inlined from [`gateway_password`] instead of its
/// `GATEWAY_AUTH_PASSWORD` env var, because `std::env::set_var` is `unsafe` (and
/// `unsafe_code` is denied workspace-wide) so the test cannot set it, and the
/// client must present the exact same value.
fn native_vllm_config(live: &LiveConfig, proxy_port: u16) -> Config {
    let path = example_config_path(CONFIG);
    let yaml = std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {path}: {error}"));
    let listener = format!("{}:{proxy_port}", live.listen_address);
    let patched = yaml
        .replace("0.0.0.0:8080", &listener)
        .replace("127.0.0.1:8080", &listener)
        .replace("127.0.0.1:8000", &live.vllm_authority)
        .replace(
            "env_var: GATEWAY_AUTH_PASSWORD",
            &format!("password: {}", gateway_password()),
        );
    Config::from_yaml(&patched).unwrap_or_else(|error| panic!("parse {CONFIG}: {error}"))
}

// -----------------------------------------------------------------------------
// Pinned executable
// -----------------------------------------------------------------------------

/// Confirm the explicitly provided binary matches the pinned Claude Code version.
async fn assert_pinned_version(claude_bin: &OsStr) {
    let output = tokio::process::Command::new(claude_bin)
        .arg("--version")
        .output()
        .await
        .expect("PRAXIS_TEST_CLAUDE_CODE_BIN should execute");
    assert!(output.status.success(), "claude --version should succeed");
    let reported = String::from_utf8_lossy(&output.stdout);
    assert!(
        reported.contains(CLAUDE_CODE_VERSION),
        "pinned Claude Code version {CLAUDE_CODE_VERSION} not found in `claude --version` output: {}",
        reported.trim()
    );
}

// -----------------------------------------------------------------------------
// Deterministic workspace
// -----------------------------------------------------------------------------

/// A deterministic coding-task workspace with a harness-owned verification marker.
struct Workspace {
    /// The temporary project directory the client runs inside.
    project: tempfile::TempDir,
    /// The harness-owned directory holding the verification marker, outside the
    /// project so the client cannot write it directly.
    marker_dir: tempfile::TempDir,
    /// The original lowercase source token the client must READ from disk. It is
    /// a random per-run value, so it can only appear in the client's Read result
    /// if the file's bytes were actually delivered back through Praxis.
    source_token: String,
    /// The required derived value: the uppercase source token.
    expected_value: String,
    /// The nonce `verify.sh` writes into the marker only on a successful compare.
    nonce: String,
}

impl Workspace {
    /// Creates the source/result files and an executable `verify.sh`.
    fn create() -> Self {
        let seed = unique_seed();
        let source_token = format!("praxis-native-{seed}");
        let expected_value = source_token.to_ascii_uppercase();
        let nonce = format!("verified-{seed}");

        let project = tempfile::tempdir().expect("temporary project directory should be created");
        let marker_dir = tempfile::tempdir().expect("temporary marker directory should be created");
        let root = project.path();

        std::fs::create_dir(root.join("source")).expect("source directory should be created");
        std::fs::create_dir(root.join("result")).expect("result directory should be created");
        std::fs::write(root.join("source/value.txt"), format!("{source_token}\n"))
            .expect("source value should be written");
        // Deliberately wrong so an unchanged file cannot pass verification.
        std::fs::write(root.join("result/value.txt"), "PLACEHOLDER\n").expect("result value should be written");

        let marker_path = marker_dir.path().join("verified.marker");
        write_verify_script(&root.join("verify.sh"), &marker_path, &nonce);

        Self {
            project,
            marker_dir,
            source_token,
            expected_value,
            nonce,
        }
    }

    /// The absolute path of the harness-owned verification marker.
    fn marker_path(&self) -> PathBuf {
        self.marker_dir.path().join("verified.marker")
    }

    /// Assert the derived file is exact and the marker proves verification ran.
    fn assert_task_completed(&self, started: SystemTime) {
        let result = std::fs::read_to_string(self.project.path().join("result/value.txt"))
            .expect("result/value.txt should exist after the run");
        assert_eq!(
            result.trim(),
            self.expected_value,
            "the client should write the exact uppercase-derived value"
        );

        let marker = self.marker_path();
        let marker_contents =
            std::fs::read_to_string(&marker).expect("verify.sh should have written the marker on success");
        assert_eq!(
            marker_contents.trim(),
            self.nonce,
            "the marker must contain the harness-generated nonce"
        );
        let modified = std::fs::metadata(&marker)
            .and_then(|metadata| metadata.modified())
            .expect("the marker should carry a modification time");
        assert!(
            modified >= started - Duration::from_secs(2),
            "verification must have run after the client started"
        );
    }

    /// Assert the client actually READ the distinct input and performed the
    /// distinct EDIT, then summarized — proven from its stream-json tool trace.
    ///
    /// The end-state file check in [`Self::assert_task_completed`] proves the
    /// right bytes landed on disk; this proves the client did the *work*: it did
    /// not guess the random value, it read `source/value.txt` and received the
    /// file's bytes back through Praxis (the Read `tool_result` carries the
    /// per-run [`Self::source_token`]), and it wrote the uppercase transform via
    /// the `Edit` tool (the tool that the end-state file could otherwise be
    /// produced by any means without).
    fn assert_task_trace(&self, stdout: &str) {
        let trace = ToolTrace::parse(stdout);

        let read = trace.find_tool_use("Read", "source/value.txt").unwrap_or_else(|| {
            panic!(
                "client must call Read on source/value.txt; tool calls observed: {:?}\nstdout:\n{stdout}",
                trace.tool_use_names(),
            )
        });
        let read_result = trace
            .result_for(&read.id)
            .unwrap_or_else(|| panic!("the Read of source/value.txt must produce a tool_result"));
        assert!(
            !read_result.is_error,
            "the Read tool_result must not be an error: {}",
            read_result.text,
        );
        assert!(
            read_result.text.contains(&self.source_token),
            "the Read tool_result must carry the per-run source token {} delivered back through Praxis; got: {}",
            self.source_token,
            read_result.text,
        );

        let edit = trace.find_tool_use("Edit", "result/value.txt").unwrap_or_else(|| {
            panic!(
                "client must call Edit on result/value.txt; tool calls observed: {:?}\nstdout:\n{stdout}",
                trace.tool_use_names(),
            )
        });
        let new_string = edit.input.get("new_string").and_then(Value::as_str).unwrap_or_default();
        assert!(
            new_string.contains(&self.expected_value),
            "the Edit must write the uppercase-derived value {} into result/value.txt; new_string was: {new_string}",
            self.expected_value,
        );
        let edit_result = trace
            .result_for(&edit.id)
            .unwrap_or_else(|| panic!("the Edit of result/value.txt must produce a tool_result"));
        assert!(
            !edit_result.is_error,
            "the Edit tool_result must not be an error: {}",
            edit_result.text,
        );

        assert!(
            trace.final_summary.is_some(),
            "Claude Code must emit a non-empty final summary in its stream-json output",
        );
    }
}

/// Writes an executable `verify.sh` embedding the marker path and nonce literally.
fn write_verify_script(path: &Path, marker_path: &Path, nonce: &str) {
    let marker = marker_path.display();
    let script = format!(
        "#!/bin/sh\n\
         set -eu\n\
         want=$(tr '[:lower:]' '[:upper:]' < source/value.txt)\n\
         got=$(cat result/value.txt)\n\
         if [ \"$want\" = \"$got\" ]; then\n\
         \tprintf '%s' '{nonce}' > '{marker}'\n\
         \techo 'verify: OK'\n\
         else\n\
         \techo 'verify: MISMATCH' >&2\n\
         \texit 1\n\
         fi\n"
    );
    std::fs::write(path, script).expect("verify.sh should be written");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        let mut permissions = std::fs::metadata(path)
            .expect("verify.sh metadata should be readable")
            .permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(path, permissions).expect("verify.sh should be made executable");
    }
}

/// Draws a fresh lowercase hex seed for the task value and nonce.
///
/// Sourced from the OS RNG so the round-tripped token and the success marker
/// nonce are unguessable per run and contain no hard-coded value.
fn unique_seed() -> String {
    random_token()
}

/// Returns a fresh 128-bit lowercase hex token drawn from the OS RNG.
fn random_token() -> String {
    rand::random::<[u8; 16]>()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

// -----------------------------------------------------------------------------
// Child launch
// -----------------------------------------------------------------------------

/// Runs the pinned Claude Code client with a cleared, pinned environment.
async fn launch_claude_code(live: &LiveConfig, proxy_base_url: &str, workspace: &Workspace) -> CapturedChildOutput {
    let config_dir = tempfile::tempdir().expect("temporary CLAUDE_CONFIG_DIR should be created");
    let home_dir = tempfile::tempdir().expect("temporary HOME should be created");
    let mcp_config = config_dir.path().join("empty-mcp.json");
    std::fs::write(&mcp_config, r#"{"mcpServers":{}}"#).expect("empty MCP config should be written");

    let mut command = child_command(live);
    command
        .arg("-p")
        .arg(PROMPT)
        .arg("--model")
        .arg(&live.model)
        .arg("--allowedTools")
        .arg("Read")
        .arg("Edit")
        .arg("Bash(./verify.sh:*)")
        .arg("--mcp-config")
        .arg(&mcp_config)
        .args(LAUNCH_FLAGS)
        .current_dir(workspace.project.path())
        .env_clear()
        .env("PATH", "/usr/local/bin:/usr/bin:/bin")
        .env("HOME", home_dir.path())
        .env("CLAUDE_CONFIG_DIR", config_dir.path())
        .env("ANTHROPIC_BASE_URL", proxy_base_url)
        // The native Anthropic `x-api-key` (a client credential the `headers`
        // filter must strip) and the gateway `Authorization: Basic ...` (which
        // `basic_auth` authenticates and strips) are two DISTINCT secrets, and
        // neither may reach vLLM. `ANTHROPIC_CUSTOM_HEADERS` carries the gateway
        // credential on every request the client makes.
        .env("ANTHROPIC_API_KEY", NATIVE_API_KEY)
        .env("ANTHROPIC_CUSTOM_HEADERS", gateway_auth_line())
        .env("ANTHROPIC_MODEL", &live.model)
        .env("ANTHROPIC_DEFAULT_MODEL", &live.model)
        .env("ANTHROPIC_DEFAULT_OPUS_MODEL", &live.model)
        .env("ANTHROPIC_DEFAULT_SONNET_MODEL", &live.model)
        .env("ANTHROPIC_DEFAULT_HAIKU_MODEL", &live.model)
        .env("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
        .env("DISABLE_UPDATES", "1")
        .env("DISABLE_TELEMETRY", "1")
        .env("DISABLE_ERROR_REPORTING", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    configure_isolated_process_group(&mut command);

    let child = command.spawn().expect("pinned Claude Code should start");
    capture_child_output(child, CHILD_TIMEOUT).await
}

/// Builds the base command, wrapping in a network namespace launcher if requested.
///
/// A requested namespace keeps the launcher (`ip netns exec <ns>`) inside the
/// child process group so the timeout cleanup reaps the whole descendant tree.
fn child_command(live: &LiveConfig) -> tokio::process::Command {
    match &live.netns {
        Some(namespace) => {
            // Resolve `ip` to an absolute path: the child runs with a cleared,
            // minimal `PATH` that omits `/usr/sbin`, so a bare `ip` would not
            // resolve. `ip netns exec` passes that environment through to the
            // pinned executable unchanged.
            let mut command = tokio::process::Command::new(resolve_ip_binary());
            command.arg("netns").arg("exec").arg(namespace).arg(&live.claude_bin);
            command
        },
        None => tokio::process::Command::new(&live.claude_bin),
    }
}

/// Resolves the `ip(8)` binary, which commonly lives outside a minimal `PATH`.
fn resolve_ip_binary() -> PathBuf {
    ["/usr/sbin/ip", "/sbin/ip", "/usr/bin/ip", "/bin/ip"]
        .into_iter()
        .map(PathBuf::from)
        .find(|candidate| candidate.exists())
        .unwrap_or_else(|| PathBuf::from("ip"))
}

// -----------------------------------------------------------------------------
// Enforced egress isolation
// -----------------------------------------------------------------------------

/// Public endpoints that MUST be unreachable from inside the isolated namespace.
///
/// A correctly isolated namespace has only the veth to the host-side Praxis
/// address and no default route, so any public IP is unreachable. Reaching one
/// would prove the client has general egress and could bypass Praxis to talk to
/// Anthropic (or the backend) directly. Two independent, stable anycast targets
/// guard against one being coincidentally routable.
const EGRESS_DENYLIST: &[(&str, u16)] = &[("1.1.1.1", 443), ("8.8.8.8", 53)];

/// Bound on each in-namespace connectivity probe.
const PROBE_TIMEOUT: Duration = Duration::from_secs(5);

/// Actively verify the namespace isolates the client to Praxis only.
///
/// Proves, from inside the namespace, that Praxis is reachable and that every
/// [`EGRESS_DENYLIST`] endpoint is not — i.e. the client's sole network path is
/// the proxy. Called only when a namespace is configured.
fn verify_egress_isolation(namespace: &str, praxis_host: IpAddr, praxis_port: u16) {
    assert!(
        netns_can_reach(namespace, &praxis_host.to_string(), praxis_port),
        "the isolated client must be able to reach Praxis at {praxis_host}:{praxis_port}"
    );
    for (host, port) in EGRESS_DENYLIST {
        assert!(
            !netns_can_reach(namespace, host, *port),
            "isolated client reached {host}:{port}; egress is not restricted to Praxis, \
             so the client could bypass the proxy"
        );
    }
}

/// Reports whether a TCP connection to `host:port` succeeds inside `namespace`.
///
/// Uses bash's `/dev/tcp` under `ip netns exec`, bounded by `timeout(1)`, so no
/// extra probe binary is required. A clean connect returns success; a refused,
/// unreachable, or timed-out connect returns failure. This probe is Linux-only,
/// matching the netns-gated acceptance run.
fn netns_can_reach(namespace: &str, host: &str, port: u16) -> bool {
    let seconds = PROBE_TIMEOUT.as_secs().max(1).to_string();
    let connect = format!("exec 3<>/dev/tcp/{host}/{port}");
    std::process::Command::new(resolve_ip_binary())
        .arg("netns")
        .arg("exec")
        .arg(namespace)
        .arg("timeout")
        .arg(&seconds)
        .arg("bash")
        .arg("-c")
        .arg(&connect)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// The `Authorization` header line the client presents to the gateway.
fn gateway_auth_line() -> String {
    format!("Authorization: {}", basic_auth_header(GATEWAY_USER, gateway_password()))
}

// -----------------------------------------------------------------------------
// Stream-json tool trace
// -----------------------------------------------------------------------------

/// One `tool_use` block the client emitted in its stream-json output.
struct ToolUse {
    /// The `tool_use` id, correlated with the matching `tool_result`.
    id: String,
    /// The tool name, e.g. `Read` or `Edit`.
    name: String,
    /// The tool input object (schema), e.g. `{ "file_path": "..." }`.
    input: Value,
}

/// One `tool_result` block returned to the client for a prior `tool_use`.
struct ToolResult {
    /// The id of the `tool_use` this result answers.
    tool_use_id: String,
    /// The flattened text payload of the result.
    text: String,
    /// Whether the tool reported an error.
    is_error: bool,
}

/// A parsed view of the tool calls, tool results, and final summary in Claude
/// Code `--output-format stream-json` output (one JSON object per line).
struct ToolTrace {
    tool_uses: Vec<ToolUse>,
    tool_results: Vec<ToolResult>,
    final_summary: Option<String>,
}

impl ToolTrace {
    /// Parses every JSONL line, collecting `tool_use`/`tool_result` blocks and
    /// the terminal `result` summary. Unparseable lines are ignored.
    fn parse(stdout: &str) -> Self {
        let mut tool_uses = Vec::new();
        let mut tool_results = Vec::new();
        let mut final_summary = None;

        for line in stdout.lines() {
            let Ok(value) = serde_json::from_str::<Value>(line.trim()) else {
                continue;
            };
            match value.get("type").and_then(Value::as_str) {
                Some("assistant") => {
                    for block in message_content(&value) {
                        if block.get("type").and_then(Value::as_str) == Some("tool_use")
                            && let (Some(id), Some(name)) = (
                                block.get("id").and_then(Value::as_str),
                                block.get("name").and_then(Value::as_str),
                            )
                        {
                            tool_uses.push(ToolUse {
                                id: id.to_owned(),
                                name: name.to_owned(),
                                input: block.get("input").cloned().unwrap_or(Value::Null),
                            });
                        }
                    }
                },
                Some("user") => {
                    for block in message_content(&value) {
                        if block.get("type").and_then(Value::as_str) == Some("tool_result")
                            && let Some(id) = block.get("tool_use_id").and_then(Value::as_str)
                        {
                            tool_results.push(ToolResult {
                                tool_use_id: id.to_owned(),
                                text: flatten_content(block.get("content")),
                                is_error: block.get("is_error").and_then(Value::as_bool).unwrap_or(false),
                            });
                        }
                    }
                },
                Some("result") => {
                    if let Some(result) = value.get("result").and_then(Value::as_str)
                        && result.trim().len() > 1
                    {
                        final_summary = Some(result.to_owned());
                    }
                },
                _ => {},
            }
        }

        Self {
            tool_uses,
            tool_results,
            final_summary,
        }
    }

    /// Finds the first `tool_use` for `name` whose `file_path` ends with `suffix`.
    fn find_tool_use(&self, name: &str, suffix: &str) -> Option<&ToolUse> {
        self.tool_uses.iter().find(|tool_use| {
            tool_use.name == name
                && tool_use
                    .input
                    .get("file_path")
                    .and_then(Value::as_str)
                    .is_some_and(|path| path.ends_with(suffix))
        })
    }

    /// Finds the `tool_result` correlated with a `tool_use` id.
    fn result_for(&self, tool_use_id: &str) -> Option<&ToolResult> {
        self.tool_results
            .iter()
            .find(|result| result.tool_use_id == tool_use_id)
    }

    /// The observed tool-use names, for assertion failure messages.
    fn tool_use_names(&self) -> Vec<&str> {
        self.tool_uses.iter().map(|tool_use| tool_use.name.as_str()).collect()
    }
}

/// Returns the `message.content` blocks of a stream-json envelope, if any.
fn message_content(value: &Value) -> &[Value] {
    value
        .pointer("/message/content")
        .and_then(Value::as_array)
        .map_or(&[], Vec::as_slice)
}

/// Flattens a `tool_result` `content`, which may be a bare string or an array of
/// `{ "type": "text", "text": ... }` blocks, into a single string.
fn flatten_content(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(blocks)) => blocks
            .iter()
            .filter_map(|block| block.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}
