// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Black-box acceptance tests for the pinned OpenCode CLI executable.
//!
//! OpenCode is the third coding client documented in
//! `docs/developing/cli-vllm-through-praxis.md`, and the only one that speaks
//! OpenAI Chat Completions:
//!
//! ```text
//! OpenCode ─► Praxis (anthropic/messages-native-vllm.yaml) ─► /v1/chat/completions
//! ```
//!
//! Nothing on that path translates a body, so these tests deliberately do NOT
//! re-prove wire fidelity — that is covered deterministically, and far more
//! precisely, in `suite/examples/anthropic_messages_native_vllm.rs`. What a
//! real executable proves and a synthetic request cannot is that the two
//! credential recipes the guide publishes actually work against the pinned
//! client, end to end, through the shipped example config.
//!
//! Two lanes, matching the two documented recipes:
//!
//! * **plugin** — the `praxis-auth.ts` `config` hook attaches the gateway Basic credential at runtime. This is the
//!   primary lane; the plugin under test is the fixture that [`documented_plugin_matches_fixture`] pins byte-for-byte
//!   to the code block in the guide.
//! * **headers** — the schema-native per-model `headers` map does the same thing declaratively, with `--pure` so no
//!   plugin loads at all.
//!
//! # Exit status is not an auth oracle
//!
//! When the plugin's `config` hook throws, the pinned client logs
//! [`PLUGIN_FAILURE_LOG`] and **continues**, sending the turn with no
//! `Authorization` header, exiting `0`, and ending its event stream in
//! `step_finish`. A test that only checked the exit status would pass while
//! the credential silently vanished. So the plugin lane asserts the header on
//! EVERY captured request and asserts the failure log is absent. See
//! `[opencode.auth.plugin] fail_open` in the pin manifest.
//!
//! # Running locally
//!
//! ```console
//! PRAXIS_TEST_OPENCODE_BIN="$HOME/.opencode/bin/opencode" \
//!   cargo test -p praxis-tests-integration --features store-all --test suite opencode::
//! ```
//!
//! Absent that variable the executable-backed tests return early; the offline
//! tests below still run and still catch pin, fixture, and config drift.

use std::{
    collections::HashMap,
    ffi::{OsStr, OsString},
    path::{Path, PathBuf},
    process::Stdio,
    sync::OnceLock,
    time::Duration,
};

use praxis_core::config::Config;
use praxis_test_utils::{
    CapturedChildOutput, StatefulCapturingBackend, basic_auth_header, capture_child_output,
    configure_isolated_process_group, example_config_path, free_port, patch_yaml, start_proxy,
};
use serde_json::{Value, json};
use tempfile::TempDir;

use super::harness::OPENCODE_PINNED_VERSION;

// -----------------------------------------------------------------------------
// Pins and constants
// -----------------------------------------------------------------------------

/// Environment variable holding the path to the pinned OpenCode executable.
const OPENCODE_BIN_ENV: &str = "PRAXIS_TEST_OPENCODE_BIN";
/// Shipped example config the documented OpenCode path rides.
pub(super) const EXAMPLE_CONFIG: &str = "anthropic/messages-native-vllm.yaml";
/// Pin manifest, relative to this crate's `fixtures/` directory.
pub(super) const PIN_MANIFEST: &str = "opencode-cli/pin.toml";
/// The documented plugin, mirrored from the guide.
pub(super) const PLUGIN_FIXTURE: &str = "opencode-cli/praxis-auth.ts";
/// Gateway `basic_auth` username the example config configures.
pub(super) const GATEWAY_USER: &str = "gateway";
/// Provider id the OpenCode configuration declares.
pub(super) const PROVIDER_ID: &str = "praxis";
/// Served model name, addressed by the client as `<provider>/<model>`.
const MODEL_ID: &str = "qwen3-8b";
/// Chat Completions path every request must take.
pub(super) const CHAT_PATH: &str = "/v1/chat/completions";
/// Fixed prompt. Tool use is suppressed so one scripted reply ends the turn.
const PROMPT: &str = "Reply with exactly PONG. Do not call any tools.";
/// Text the scripted backend streams back.
const REPLY: &str = "PONG";
/// Log line the client emits, only under `--print-logs`, when the hook throws.
pub(super) const PLUGIN_FAILURE_LOG: &str = "plugin config hook failed";
/// Bound on one child run. The client is a Bun binary with a slow cold start.
const CHILD_TIMEOUT: Duration = Duration::from_secs(90);
/// Scripted responses to queue. A trivial prompt costs two turns (a toolless
/// title turn and the main turn); the surplus absorbs any extra client chatter
/// so an exhausted script never masquerades as a protocol failure.
const SCRIPTED_RESPONSES: usize = 24;

/// Keys the pinned schema accepts on a provider model object.
///
/// The object is `additionalProperties: false`, so a key outside this set is a
/// hard startup error rather than a silently ignored setting. Mirrors
/// `[opencode.auth.headers] model_keys` in the pin manifest.
const MODEL_OBJECT_KEYS: &[&str] = &[
    "id",
    "name",
    "family",
    "release_date",
    "attachment",
    "reasoning",
    "temperature",
    "tool_call",
    "interleaved",
    "cost",
    "limit",
    "modalities",
    "experimental",
    "status",
    "provider",
    "options",
    "headers",
    "variants",
];

/// Which documented credential recipe a run exercises.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum AuthLane {
    /// `praxis-auth.ts` attaches the credential through the `config` hook.
    Plugin,
    /// A per-model `headers` map carries it, with plugins disabled.
    ModelHeaders,
    /// The plugin is present but `--pure` must ignore it, so no credential is
    /// attached and the gateway must reject the request.
    PureIgnoresPlugin,
}

impl AuthLane {
    /// Whether the client runs with `--pure` (no external plugins).
    pub(super) const fn is_pure(self) -> bool {
        matches!(self, Self::ModelHeaders | Self::PureIgnoresPlugin)
    }

    /// Whether a gateway credential is expected to reach the backend.
    const fn expects_credential(self) -> bool {
        matches!(self, Self::Plugin | Self::ModelHeaders)
    }
}

// -----------------------------------------------------------------------------
// Paths, secrets, and the gateway config
// -----------------------------------------------------------------------------

/// Absolute path to a file under this crate's `fixtures/` directory.
pub(super) fn fixture_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures").join(relative)
}

/// Absolute path to the repository root.
pub(super) fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .expect("repository root should resolve")
}

/// The gateway password these tests inject in place of `GATEWAY_AUTH_PASSWORD`.
///
/// Drawn once per test process from the OS RNG rather than a source literal, so
/// it is a throwaway secret scoped to this run; the config and the client read
/// the same value so they agree within a run.
pub(super) fn gateway_password() -> &'static str {
    static PASSWORD: OnceLock<String> = OnceLock::new();
    PASSWORD.get_or_init(|| {
        rand::random::<[u8; 16]>()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    })
}

/// The `Authorization` header value a trusted caller presents to the gateway.
pub(super) fn gateway_credential() -> String {
    basic_auth_header(GATEWAY_USER, gateway_password())
}

/// Replace `from` exactly once, failing loudly if the example config drifts.
///
/// A silent no-op here would build an unpatched config that still parses and
/// still serves, so the test would exercise the wrong thing rather than fail.
fn replace_once(haystack: &str, from: &str, to: &str, what: &str) -> String {
    let count = haystack.matches(from).count();
    assert_eq!(
        count, 1,
        "{EXAMPLE_CONFIG} should contain exactly one {what} anchor ({from:?}), found {count}; \
         the example drifted and this patch no longer applies"
    );
    haystack.replace(from, to)
}

/// Build the shipped example config with ports, gateway password, and backend
/// credential patched for an in-process run.
///
/// Both secrets resolve at pipeline-build time. `std::env::set_var` is `unsafe`
/// and `unsafe_code` is denied workspace-wide, so the password is inlined and
/// `VLLM_API_KEY` is repointed at `CARGO_PKG_NAME`, which Cargo always sets for
/// a test binary. Mirrors `examples/anthropic_messages_native_vllm.rs`.
fn gateway_config(proxy_port: u16, backend_port: u16) -> Config {
    let path = example_config_path(EXAMPLE_CONFIG);
    let yaml = std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {path}: {error}"));
    let patched = patch_yaml(&yaml, proxy_port, &HashMap::from([("127.0.0.1:8000", backend_port)]));
    let patched = replace_once(
        &patched,
        "env_var: VLLM_API_KEY",
        "env_var: CARGO_PKG_NAME",
        "backend credential",
    );
    let patched = replace_once(
        &patched,
        "env_var: GATEWAY_AUTH_PASSWORD",
        &format!("password: {}", gateway_password()),
        "gateway password",
    );
    Config::from_yaml(&patched).unwrap_or_else(|error| panic!("parse patched {EXAMPLE_CONFIG}: {error}"))
}

// -----------------------------------------------------------------------------
// Scripted Chat Completions backend
// -----------------------------------------------------------------------------

/// One scripted streaming reply.
///
/// A body beginning with `data: ` is served as `text/event-stream` and split on
/// blank lines into incremental chunks, so this exercises the streaming path the
/// client actually uses.
fn pong_sse() -> String {
    let role = json!({
        "id": "chatcmpl-opencode",
        "object": "chat.completion.chunk",
        "created": 1_677_652_288_u64,
        "model": MODEL_ID,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": REPLY}, "finish_reason": null}],
    });
    let stop = json!({
        "id": "chatcmpl-opencode",
        "object": "chat.completion.chunk",
        "created": 1_677_652_288_u64,
        "model": MODEL_ID,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 9, "completion_tokens": 1, "total_tokens": 10},
    });
    format!("data: {role}\n\ndata: {stop}\n\ndata: [DONE]\n\n")
}

/// A backend queued with enough identical replies to outlast client chatter.
fn scripted_backend() -> StatefulCapturingBackend {
    StatefulCapturingBackend::new((0..SCRIPTED_RESPONSES).map(|_| (200, pong_sse())).collect())
}

// -----------------------------------------------------------------------------
// Client configuration
// -----------------------------------------------------------------------------

/// The provider model object, optionally carrying a `headers` map.
fn model_object(headers: Option<Value>) -> Value {
    let mut model = json!({
        "name": "Qwen3 8B",
        "limit": {"context": 32_768, "output": 8_192},
    });
    if let Some(headers) = headers {
        model["headers"] = headers;
    }
    model
}

/// The full client configuration for a lane.
///
/// The `plugin` array is declared whenever the plugin file is seeded on disk —
/// every lane except `ModelHeaders` — so the `PureIgnoresPlugin` inversion lane
/// has a DECLARED plugin for `--pure` to ignore. OpenCode loads plugins only
/// from this array (there is no config-dir auto-discovery), so an undeclared
/// plugin never loads with or without `--pure`; declaring it only for the
/// non-pure lane would make `--pure` a no-op and the inversion prove nothing.
/// The entry is a path relative to the configuration directory the caller
/// seeded.
pub(super) fn client_config(base_url: &str, model: &str, lane: AuthLane) -> Value {
    let headers = (lane == AuthLane::ModelHeaders).then(|| json!({"Authorization": gateway_credential()}));
    let mut config = json!({
        "$schema": "https://opencode.ai/config.json",
        "autoupdate": false,
        "provider": {
            PROVIDER_ID: {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Praxis",
                "options": {"baseURL": base_url},
                "models": {model: model_object(headers)},
            },
        },
    });
    // Declare the plugin on every lane that seeds it (mirrors `seed_config_dir`),
    // NOT only the non-pure lanes: `PureIgnoresPlugin` must declare it so `--pure`
    // has a plugin to ignore, otherwise the inversion lane is vacuous.
    if lane != AuthLane::ModelHeaders {
        config["plugin"] = json!(["./praxis-auth.ts"]);
    }
    config
}

/// Seed `dir` as the client's configuration directory for `lane`.
///
/// The plugin is copied from the committed fixture rather than written inline,
/// so the executable loads exactly the source the guide publishes.
pub(super) fn seed_config_dir(dir: &Path, base_url: &str, model: &str, lane: AuthLane) {
    std::fs::create_dir_all(dir).expect("client configuration directory should be creatable");
    if lane != AuthLane::ModelHeaders {
        std::fs::copy(fixture_path(PLUGIN_FIXTURE), dir.join("praxis-auth.ts"))
            .expect("documented plugin fixture should be copyable into the configuration directory");
    }
    let config =
        serde_json::to_string_pretty(&client_config(base_url, model, lane)).expect("client config should serialize");
    std::fs::write(dir.join("opencode.jsonc"), config).expect("client configuration should be writable");
}

/// Arguments for one headless run.
///
/// The prompt is POSITIONAL and is placed after `--`. On `run`, `-p`/
/// `--password` is the OpenCode *server* Basic-auth password, so passing the
/// prompt that way would silently send it as a credential and leave the model
/// with an empty turn. [`opencode_run_uses_positional_prompt_not_dash_p`]
/// guards this.
pub(super) fn run_args(lane: AuthLane, model: &str, prompt: &str) -> Vec<String> {
    let mut args = vec!["run".to_owned()];
    if lane.is_pure() {
        args.push("--pure".to_owned());
    }
    args.extend([
        "--model".to_owned(),
        format!("{PROVIDER_ID}/{model}"),
        "--format".to_owned(),
        "json".to_owned(),
        // Mandatory: a failing plugin is otherwise entirely silent.
        "--print-logs".to_owned(),
        "--log-level".to_owned(),
        "ERROR".to_owned(),
        "--".to_owned(),
        prompt.to_owned(),
    ]);
    args
}

// -----------------------------------------------------------------------------
// Launch
// -----------------------------------------------------------------------------

/// Launch the pinned client against `base_url` and capture its output.
/// Everything one headless run needs.
///
/// Grouped rather than passed positionally because the two lanes differ in
/// four of these fields, and a bare argument list of this width invites the
/// kind of transposition that would silently test the wrong thing.
pub(super) struct LaunchSpec<'a> {
    /// The pinned executable.
    pub(super) bin: &'a OsStr,
    /// Temporary HOME; also roots the XDG directories and the config dir.
    pub(super) home: &'a Path,
    /// Working directory for the run.
    pub(super) workspace: &'a Path,
    /// Praxis base URL, including the `/v1` suffix.
    pub(super) base_url: &'a str,
    /// Served model name, addressed as `<provider>/<model>`.
    pub(super) model: &'a str,
    /// Which credential recipe to exercise.
    pub(super) lane: AuthLane,
    /// The positional prompt.
    pub(super) prompt: &'a str,
    /// Bound on the child.
    pub(super) timeout: Duration,
}

pub(super) async fn launch(spec: LaunchSpec<'_>) -> CapturedChildOutput {
    let LaunchSpec {
        bin,
        home,
        workspace,
        base_url,
        model,
        lane,
        prompt,
        timeout,
    } = spec;
    let config_home = home.join(".config");
    seed_config_dir(&config_home.join("opencode"), base_url, model, lane);

    let mut command = tokio::process::Command::new(bin);
    command
        .args(run_args(lane, model, prompt))
        .current_dir(workspace)
        .env_clear()
        .env("HOME", home)
        // Set the XDG roots explicitly rather than relying on HOME-derived
        // defaults, so a stray real credential store can never be reached.
        .env("XDG_CONFIG_HOME", &config_home)
        .env("XDG_DATA_HOME", home.join(".local").join("share"))
        .env("XDG_CACHE_HOME", home.join(".cache"))
        .env("PATH", "/usr/local/bin:/usr/bin:/bin")
        .env("CI", "1")
        .env("TERM", "dumb")
        .env("NO_COLOR", "1")
        // Suppress the catalog, update, share, and LSP-download probes. These
        // are latency hygiene, not correctness: the client completes with no
        // egress at all, verified by running this flow in a loopback-only
        // network namespace with none of them set.
        .env("OPENCODE_DISABLE_MODELS_FETCH", "1")
        .env("OPENCODE_DISABLE_AUTOUPDATE", "1")
        .env("OPENCODE_DISABLE_SHARE", "1")
        .env("OPENCODE_DISABLE_LSP_DOWNLOAD", "1")
        // Deliberately NOT setting HTTP_PROXY/HTTPS_PROXY/ALL_PROXY to a dead
        // port, which is how `claude_code.rs` fences its client in. Plugin
        // loading honors those variables, so against a refused proxy it retries
        // for ~68s: measured 4s without them and 72s with them, for an
        // otherwise identical plugin run. The CI lane gets real isolation from
        // a network namespace with no default route, which is a hard guarantee
        // rather than an advisory one, so the dead-proxy trick would buy
        // nothing here and cost a minute per run.
        //
        // STDIN MUST BE NULL. With stdin open the client logs `init` and then
        // blocks forever, before issuing a single request, even though the
        // prompt was supplied as an argument.
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    if lane != AuthLane::ModelHeaders {
        // Read by the documented plugin. `PRAXIS_BASE_URL` must match the
        // merged `baseURL` exactly or the hook throws.
        command
            .env("GATEWAY_AUTH_PASSWORD", gateway_password())
            .env("PRAXIS_BASE_URL", base_url);
    }
    configure_isolated_process_group(&mut command);

    let child = command.spawn().expect("pinned OpenCode executable should start");
    capture_child_output(child, timeout).await
}

/// Assert the child finished within its bound, surfacing both pipes on failure.
fn assert_not_timed_out(output: &CapturedChildOutput, scenario: &str) {
    assert!(
        !output.timed_out,
        "{scenario}: OpenCode exceeded {CHILD_TIMEOUT:?}. If no request reached the backend, \
         check that stdin is /dev/null — the client blocks indefinitely otherwise.\n\
         STDOUT:\n{}\nSTDERR:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

// -----------------------------------------------------------------------------
// Client event trace
// -----------------------------------------------------------------------------

/// The client's `--format json` event stream.
///
/// Deliberately lenient: it keeps only lines that parse as JSON objects and
/// asserts nothing about the schema beyond the `type` discriminant. Hard
/// assertions on event shape break on every client release and would make this
/// suite a liability rather than a guard.
pub(super) struct OpenCodeTrace {
    events: Vec<Value>,
}

impl OpenCodeTrace {
    /// Parse the event stream, discarding any non-JSON preamble.
    pub(super) fn parse(stdout: &str) -> Self {
        let events = stdout
            .lines()
            .filter_map(|line| serde_json::from_str::<Value>(line.trim()).ok())
            .filter(|value| value.is_object())
            .collect();
        Self { events }
    }

    /// Every `type` discriminant, in order.
    pub(super) fn event_types(&self) -> Vec<&str> {
        self.events.iter().filter_map(|event| event["type"].as_str()).collect()
    }

    /// Concatenated assistant text across all `text` events.
    pub(super) fn assistant_text(&self) -> String {
        self.events
            .iter()
            .filter(|event| event["type"].as_str() == Some("text"))
            .filter_map(|event| event["part"]["text"].as_str())
            .collect()
    }

    /// Error events, which the client emits with an `APIError` payload.
    pub(super) fn errors(&self) -> Vec<&Value> {
        self.events
            .iter()
            .filter(|event| event["type"].as_str() == Some("error"))
            .collect()
    }
}

// -----------------------------------------------------------------------------
// Wire assertions
// -----------------------------------------------------------------------------

/// Look up one header value from a captured newline-separated header block.
pub(super) fn header_value<'a>(headers: &'a str, name: &str) -> Option<&'a str> {
    headers.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        key.trim().eq_ignore_ascii_case(name).then(|| value.trim())
    })
}

// -----------------------------------------------------------------------------
// Executable-backed acceptance tests
// -----------------------------------------------------------------------------

/// Resolve the pinned executable, or `None` when the lane is not configured.
fn opencode_bin() -> Option<OsString> {
    match std::env::var_os(OPENCODE_BIN_ENV) {
        Some(bin) if !bin.is_empty() => Some(bin),
        _ => {
            eprintln!("skipping pinned OpenCode acceptance test; {OPENCODE_BIN_ENV} is unset");
            None
        },
    }
}

#[test]
fn pinned_opencode_version_check() {
    let Some(bin) = opencode_bin() else { return };

    let output = std::process::Command::new(&bin)
        .arg("--version")
        .output()
        .expect("pinned OpenCode executable should run --version");

    assert!(
        output.status.success(),
        "opencode --version should exit 0, got {:?}",
        output.status.code()
    );
    // The client prints the bare version with no product prefix, so this is an
    // equality check rather than a substring search.
    let reported = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    assert_eq!(
        reported, OPENCODE_PINNED_VERSION,
        "pinned OpenCode version mismatch; update OPENCODE_PINNED_VERSION, \
         fixtures/{PIN_MANIFEST}, and the CI job together"
    );
}

/// Drive one lane end to end and return the captured output and requests.
async fn run_lane(lane: AuthLane, bin: &OsStr) -> (CapturedChildOutput, Vec<praxis_test_utils::CapturedRequest>) {
    let backend = scripted_backend().start_with_shutdown();
    let proxy_port = free_port();
    let config = gateway_config(proxy_port, backend.port());
    let proxy = start_proxy(&config);

    let home = TempDir::new().expect("temporary HOME should be creatable");
    let workspace = TempDir::new().expect("temporary workspace should be creatable");
    let base_url = format!("http://{}/v1", proxy.addr());

    let output = launch(LaunchSpec {
        bin,
        home: home.path(),
        workspace: workspace.path(),
        base_url: &base_url,
        model: MODEL_ID,
        lane,
        prompt: PROMPT,
        timeout: CHILD_TIMEOUT,
    })
    .await;
    (output, backend.requests())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pinned_opencode_completes_turn_with_plugin_auth() {
    let Some(bin) = opencode_bin() else { return };
    let (output, requests) = run_lane(AuthLane::Plugin, &bin).await;

    assert_not_timed_out(&output, "plugin lane");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let trace = OpenCodeTrace::parse(&stdout);

    // Diagnose from the trace before the exit status: a turn that failed for a
    // protocol reason is far more informative than a bare non-zero code.
    assert!(
        trace.errors().is_empty(),
        "plugin lane should not surface a client error event: {:?}\nSTDERR:\n{stderr}",
        trace.errors()
    );
    // The plugin fails OPEN: a throwing hook still exits 0 and still sends the
    // turn, just unauthenticated. Without this the lane could pass while the
    // very thing it exists to prove was broken.
    assert!(
        !stderr.contains(PLUGIN_FAILURE_LOG),
        "the documented plugin must load and run cleanly; found {PLUGIN_FAILURE_LOG:?}:\n{stderr}"
    );
    assert!(
        output.status.success(),
        "plugin lane should exit 0, got {:?}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}",
        output.status.code()
    );
    assert!(
        trace.assistant_text().contains(REPLY),
        "assistant text should carry the scripted reply, got {:?} (events: {:?})",
        trace.assistant_text(),
        trace.event_types()
    );

    assert_requests_authenticated(&requests, AuthLane::Plugin);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pinned_opencode_completes_turn_with_model_headers_auth() {
    let Some(bin) = opencode_bin() else { return };
    let (output, requests) = run_lane(AuthLane::ModelHeaders, &bin).await;

    assert_not_timed_out(&output, "model-headers lane");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let trace = OpenCodeTrace::parse(&stdout);

    assert!(
        trace.errors().is_empty(),
        "model-headers lane should not surface a client error event: {:?}\nSTDERR:\n{stderr}",
        trace.errors()
    );
    assert!(
        output.status.success(),
        "model-headers lane should exit 0, got {:?}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}",
        output.status.code()
    );
    assert!(
        trace.assistant_text().contains(REPLY),
        "assistant text should carry the scripted reply, got {:?}",
        trace.assistant_text()
    );

    assert_requests_authenticated(&requests, AuthLane::ModelHeaders);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pinned_opencode_pure_mode_ignores_the_plugin_and_is_rejected() {
    let Some(bin) = opencode_bin() else { return };
    // The plugin IS present in the configuration directory and the config
    // declares it; `--pure` must ignore it. Inverting the plugin lane this way
    // proves the plugin is what supplies the credential, rather than something
    // else in the environment happening to authenticate the run.
    let (output, requests) = run_lane(AuthLane::PureIgnoresPlugin, &bin).await;

    assert_not_timed_out(&output, "pure lane");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let trace = OpenCodeTrace::parse(&stdout);

    // `basic_auth` runs before routing, so the unauthenticated turn is rejected
    // at the gateway and never reaches the backend at all. Zero captured
    // requests is the proof that the credential really was absent — a request
    // that got through would mean something other than the plugin supplied it.
    assert!(
        requests.is_empty(),
        "--pure must not authenticate, so the gateway must reject before the backend sees anything; \
         backend captured {} request(s): {:?}",
        requests.len(),
        requests.iter().map(|request| &request.uri).collect::<Vec<_>>()
    );

    // The client must surface the rejection rather than reporting success.
    let errors = trace.errors();
    assert_eq!(
        errors.len(),
        1,
        "an unauthenticated turn should surface exactly one client error event, got {:?}",
        trace.event_types()
    );
    assert_eq!(
        errors[0]["error"]["data"]["statusCode"].as_u64(),
        Some(401),
        "the gateway should reject the unauthenticated turn with 401: {}",
        errors[0]
    );
    // Confirms the 401 came from this gateway's `basic_auth` realm rather than
    // from some other hop that happened to answer 401.
    let challenge = errors[0]["error"]["data"]["responseHeaders"]["www-authenticate"]
        .as_str()
        .unwrap_or_default();
    assert!(
        challenge.contains("praxis-native-vllm-gateway"),
        "the 401 should carry the example config's basic_auth realm, got {challenge:?}"
    );
}

/// Assert the gateway credential reached the backend on every captured request.
///
/// Checks EVERY request, not one: a client sub-session that fell back to a
/// different provider would still leave the first, correct request in place and
/// a single-request assertion would miss it.
fn assert_requests_authenticated(requests: &[praxis_test_utils::CapturedRequest], lane: AuthLane) {
    assert!(
        lane.expects_credential(),
        "only credential-bearing lanes should be checked here"
    );
    assert!(
        !requests.is_empty(),
        "{lane:?} lane: the backend should have received at least one request"
    );

    let injected = std::env::var("CARGO_PKG_NAME").expect("CARGO_PKG_NAME is always set by cargo test");
    for (index, request) in requests.iter().enumerate() {
        assert_eq!(
            request.method, "POST",
            "{lane:?} lane request #{index} should be a POST, got {}",
            request.method
        );
        assert!(
            request.uri.starts_with(CHAT_PATH),
            "{lane:?} lane request #{index} should target {CHAT_PATH}, got {}",
            request.uri
        );
        // The gateway credential is verified and stripped; only the injected
        // server-owned backend token may reach the backend.
        assert_eq!(
            header_value(&request.headers, "authorization"),
            Some(format!("Bearer {injected}").as_str()),
            "{lane:?} lane request #{index} should carry the injected backend credential: {}",
            request.headers
        );
        assert!(
            !request.headers.contains(gateway_password()),
            "{lane:?} lane request #{index} must not leak the gateway password upstream"
        );
        // `anthropic_messages_protocol` is scoped to /v1/messages, so an
        // Anthropic protocol header has no business on this path.
        assert!(
            header_value(&request.headers, "anthropic-version").is_none(),
            "{lane:?} lane request #{index} should not carry anthropic-version: {}",
            request.headers
        );
    }
}

// -----------------------------------------------------------------------------
// Offline tests
//
// These run on every PR with no executable present, so pin, fixture, schema,
// and CI drift is caught even when the acceptance lane is skipped.
// -----------------------------------------------------------------------------

/// Read one scalar value from the pin manifest.
///
/// A line-oriented reader rather than a TOML parser: this crate has no `toml`
/// dependency, and the handful of pins asserted here are all top-level scalars
/// written one per line.
pub(super) fn pin_value(key: &str) -> String {
    let manifest = std::fs::read_to_string(fixture_path(PIN_MANIFEST)).expect("pin manifest should be readable");
    let needle = format!("{key} =");
    let line = manifest
        .lines()
        .map(str::trim)
        .find(|line| line.starts_with(&needle))
        .unwrap_or_else(|| panic!("pin manifest should declare {key}"));
    line.split_once('=')
        .expect("a pin line should contain =")
        .1
        .trim()
        .trim_matches('"')
        .to_owned()
}

#[test]
fn pin_toml_version_matches_harness_const() {
    assert_eq!(
        pin_value("version"),
        OPENCODE_PINNED_VERSION,
        "the pin manifest and OPENCODE_PINNED_VERSION must agree; a bump must update both"
    );
}

#[test]
fn pinned_opencode_version_is_a_one_x_release() {
    // OpenCode 2.x replaces the `config` plugin hook this suite depends on, so
    // crossing the major boundary needs a rewritten plugin, not a version bump.
    let major = OPENCODE_PINNED_VERSION
        .split('.')
        .next()
        .expect("version should be dotted");
    assert_eq!(
        major, "1",
        "pinned OpenCode must stay on the 1.x plugin contract; 2.x requires a new plugin"
    );
    assert_eq!(
        pin_value("major_series"),
        "1",
        "the pin manifest's major_series tripwire must agree with the pinned version"
    );
}

#[test]
fn documented_plugin_matches_fixture() {
    // The acceptance lane loads the fixture, so without this assertion the
    // suite would validate its own private copy of the plugin rather than the
    // one the guide tells users to create.
    let guide = repo_root().join("docs/developing/cli-vllm-through-praxis.md");
    let doc = std::fs::read_to_string(&guide).expect("client guide should be readable");

    let blocks: Vec<&str> = doc
        .split("```typescript\n")
        .skip(1)
        .filter_map(|rest| rest.split_once("```").map(|(block, _)| block))
        .collect();
    assert_eq!(
        blocks.len(),
        1,
        "the guide should publish exactly one TypeScript block (the Praxis auth plugin)"
    );

    let fixture = std::fs::read_to_string(fixture_path(PLUGIN_FIXTURE)).expect("plugin fixture should be readable");
    assert_eq!(
        fixture, blocks[0],
        "fixtures/{PLUGIN_FIXTURE} must match the plugin published in the guide byte for byte; \
         edit both or the acceptance test stops validating the documentation"
    );
}

#[test]
fn model_headers_config_is_schema_key_safe() {
    // The model object is `additionalProperties: false`, so an unknown key is a
    // hard startup error. Catching it here beats discovering it in a lane that
    // only runs when the executable is present.
    let config = client_config("http://127.0.0.1:8080/v1", MODEL_ID, AuthLane::ModelHeaders);
    let model = &config["provider"][PROVIDER_ID]["models"][MODEL_ID];
    let keys: Vec<&String> = model
        .as_object()
        .expect("model entry should be an object")
        .keys()
        .collect();

    for key in &keys {
        assert!(
            MODEL_OBJECT_KEYS.contains(&key.as_str()),
            "model object key {key:?} is not in the pinned schema allowlist {MODEL_OBJECT_KEYS:?}"
        );
    }
    let limit = &model["limit"];
    assert!(
        limit["context"].is_number() && limit["output"].is_number(),
        "the schema requires both limit.context and limit.output: {limit}"
    );
}

#[test]
fn model_headers_config_carries_the_gateway_credential() {
    let config = client_config("http://127.0.0.1:8080/v1", MODEL_ID, AuthLane::ModelHeaders);
    let model = &config["provider"][PROVIDER_ID]["models"][MODEL_ID];

    assert_eq!(
        model["headers"]["Authorization"].as_str(),
        Some(gateway_credential().as_str()),
        "the model-headers lane must carry the gateway Basic credential"
    );
    assert!(
        config["plugin"].is_null(),
        "the model-headers lane runs --pure and must not declare a plugin"
    );
    // `provider.options` has no `headers` key in the published schema; the
    // credential belongs on the model.
    assert!(
        config["provider"][PROVIDER_ID]["options"]["headers"].is_null(),
        "the credential must live on the model, not on provider.options"
    );
}

#[test]
fn plugin_config_declares_the_documented_plugin_path() {
    let config = client_config("http://127.0.0.1:8080/v1", MODEL_ID, AuthLane::Plugin);
    assert_eq!(
        config["plugin"],
        json!(["./praxis-auth.ts"]),
        "the plugin lane must reference the plugin exactly as the guide does"
    );
    assert!(
        config["provider"][PROVIDER_ID]["models"][MODEL_ID]["headers"].is_null(),
        "the plugin lane must rely on the hook alone, with no declarative header"
    );
}

#[test]
fn opencode_run_uses_positional_prompt_not_dash_p() {
    // On `run`, -p/--password is the OpenCode SERVER Basic-auth password. A
    // `claude -p "<prompt>"` habit would send the prompt as a credential and
    // leave the model with an empty turn, which is both a silent test failure
    // and a credential leak into process arguments.
    let args = run_args(AuthLane::Plugin, MODEL_ID, PROMPT);

    assert!(
        !args.iter().any(|arg| arg == "-p" || arg == "--password"),
        "the prompt must never be passed via -p/--password: {args:?}"
    );
    let separator = args
        .iter()
        .position(|arg| arg == "--")
        .expect("the prompt should be separated from flags by --");
    assert_eq!(
        args.get(separator + 1).map(String::as_str),
        Some(PROMPT),
        "the prompt must be the positional argument after --: {args:?}"
    );
    assert_eq!(
        separator + 2,
        args.len(),
        "the prompt must be the final argument: {args:?}"
    );
    assert_eq!(
        args.first().map(String::as_str),
        Some("run"),
        "subcommand should be run"
    );
    assert!(
        args.iter().any(|arg| arg == "--print-logs"),
        "--print-logs is required: a failing plugin is otherwise silent: {args:?}"
    );
}

#[test]
fn pure_lanes_pass_the_pure_flag() {
    assert!(
        run_args(AuthLane::ModelHeaders, MODEL_ID, PROMPT)
            .iter()
            .any(|arg| arg == "--pure"),
        "the model-headers lane must disable plugins"
    );
    assert!(
        run_args(AuthLane::PureIgnoresPlugin, MODEL_ID, PROMPT)
            .iter()
            .any(|arg| arg == "--pure"),
        "the inversion lane must disable plugins"
    );
    assert!(
        !run_args(AuthLane::Plugin, MODEL_ID, PROMPT)
            .iter()
            .any(|arg| arg == "--pure"),
        "the plugin lane must load the plugin"
    );
}

#[test]
fn pure_inversion_lane_declares_a_plugin_for_pure_to_ignore() {
    // The inversion lane's whole purpose is proving `--pure` ignores a DECLARED
    // plugin. OpenCode loads plugins only from the config `plugin` array, so if
    // this lane omitted the declaration the 401 would come from the plugin never
    // loading rather than from `--pure` ignoring it, and the lane would prove
    // nothing. Guard the declaration (and the still-present `--pure`) offline.
    let config = client_config("http://127.0.0.1:8080/v1", MODEL_ID, AuthLane::PureIgnoresPlugin);
    assert_eq!(
        config["plugin"],
        json!(["./praxis-auth.ts"]),
        "the inversion lane must declare the plugin so --pure has something to ignore"
    );
    assert!(
        run_args(AuthLane::PureIgnoresPlugin, MODEL_ID, PROMPT)
            .iter()
            .any(|arg| arg == "--pure"),
        "the inversion lane must still pass --pure"
    );
    assert!(
        config["provider"][PROVIDER_ID]["models"][MODEL_ID]["headers"].is_null(),
        "the inversion lane must rely on the ignored plugin alone, with no declarative header"
    );
}

#[test]
fn gateway_config_patches_the_shipped_example() {
    // Exercises both `replace_once` anchors against the real file on disk, so
    // the example drifting out from under this suite fails here rather than in
    // a lane that only runs when the executable is present.
    let config = gateway_config(29_930, 29_931);
    let chain = &config.filter_chains[0];

    let router = chain
        .filters
        .iter()
        .find(|filter| filter.filter_type == "router")
        .expect("example chain should contain a router");
    let routes = router.config["routes"]
        .as_sequence()
        .expect("router should declare routes");
    assert!(
        routes.iter().any(|route| route["path"].as_str() == Some(CHAT_PATH)),
        "the example must declare an explicit {CHAT_PATH} route for OpenCode"
    );
    assert_eq!(
        pin_value("example_config"),
        EXAMPLE_CONFIG,
        "the pin manifest should name the example config this suite drives"
    );
}

#[test]
fn trace_parser_tolerates_noise_and_finds_errors() {
    let trace = OpenCodeTrace::parse(concat!(
        "not json at all\n",
        "{\"type\":\"step_start\"}\n",
        "\n",
        "{\"type\":\"text\",\"part\":{\"text\":\"PO\"}}\n",
        "{\"type\":\"text\",\"part\":{\"text\":\"NG\"}}\n",
        "{\"type\":\"step_finish\"}\n",
    ));
    assert_eq!(trace.event_types(), ["step_start", "text", "text", "step_finish"]);
    assert_eq!(trace.assistant_text(), "PONG");
    assert!(trace.errors().is_empty(), "a clean run has no error events");

    let failed =
        OpenCodeTrace::parse("{\"type\":\"error\",\"error\":{\"name\":\"APIError\",\"data\":{\"statusCode\":401}}}\n");
    assert_eq!(failed.errors().len(), 1, "a 401 turn surfaces one error event");

    assert!(
        OpenCodeTrace::parse("").events.is_empty(),
        "an empty stream parses to no events"
    );
}

#[test]
fn header_lookup_is_case_insensitive_and_absent_safe() {
    let headers = "Host: localhost\nAuthorization: Bearer token\nContent-Type: application/json";
    assert_eq!(header_value(headers, "authorization"), Some("Bearer token"));
    assert_eq!(header_value(headers, "AUTHORIZATION"), Some("Bearer token"));
    assert_eq!(header_value(headers, "anthropic-version"), None);
}

#[test]
fn opencode_lane_is_wired_in_ci() {
    // A Rust test that early-returns reports as PASSED, so a lane whose gate
    // variable is never set would be invisibly dead. Assert CI actually sets it.
    let workflow = repo_root().join(".github/workflows/integration.yaml");
    let yaml = std::fs::read_to_string(&workflow).expect("integration workflow should be readable");

    assert!(
        yaml.contains(OPENCODE_BIN_ENV),
        "{} must set {OPENCODE_BIN_ENV}, or the OpenCode acceptance lane silently never runs",
        workflow.display()
    );
    assert!(
        yaml.contains(OPENCODE_PINNED_VERSION),
        "{} must install the pinned OpenCode version {OPENCODE_PINNED_VERSION}",
        workflow.display()
    );
}
