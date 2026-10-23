// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Black-box acceptance tests for the pinned pi coding agent executable.
//!
//! pi is the fourth coding client documented in
//! `docs/developing/cli-vllm-through-praxis.md`. Its supported lane is OpenAI
//! Chat Completions, the same untranslated path OpenCode rides:
//!
//! ```text
//! pi (openai-completions) ─► Praxis (anthropic/messages-native-vllm.yaml) ─► /v1/chat/completions
//! ```
//!
//! Nothing on that path translates a body, so these tests deliberately do NOT
//! re-prove wire fidelity — `suite/examples/anthropic_messages_native_vllm.rs`
//! covers that. What a real executable proves and a synthetic request cannot is
//! that the `models.json` the guide publishes actually authenticates the pinned
//! client through the shipped example config, end to end.
//!
//! Four lanes, all built from the documented `models.json` fixture, which
//! [`documented_models_json_matches_fixture`] pins byte for byte to the guide:
//!
//! * **documented** — the guide's configuration completes a text turn.
//! * **tool call** — the same configuration completes a scripted `bash` round trip, so the `role: "tool"` turn crosses
//!   the gateway too.
//! * **api key only** — the `headers` block removed. The placeholder `apiKey` alone must be rejected by the gateway,
//!   which proves the header, not the key, is the credential.
//! * **token unset** — the documented configuration with `PRAXIS_GATEWAY_BASIC_TOKEN` absent. pi must refuse to send
//!   anything, which is the fail-closed property the guide promises.
//!
//! # Exit status is not an error oracle
//!
//! With `--mode json` the pinned client exits `0` even when the turn fails; the
//! failure is visible only as `stopReason: "error"` on the final assistant
//! message. Every lane therefore asserts on [`PiTrace::stop_reason`]. See
//! `[pi.launch] exit_code_meaningful_in_json_mode` in the pin manifest.
//!
//! # Running locally
//!
//! ```console
//! PRAXIS_TEST_PI_BIN="$PWD/pi/pi" \
//!   cargo test -p praxis-tests-integration --features store-all --test suite pi::
//! ```
//!
//! Absent that variable the executable-backed tests return early; the offline
//! tests below still run and still catch pin, fixture, and config drift.

use std::{
    ffi::{OsStr, OsString},
    path::Path,
    process::Stdio,
    time::Duration,
};

use praxis_test_utils::{
    CapturedChildOutput, CapturedRequest, StatefulCapturingBackend, capture_child_output,
    configure_isolated_process_group, free_port, start_proxy,
};
use serde_json::{Value, json};
use tempfile::TempDir;

use super::{
    harness::PI_PINNED_VERSION,
    live_vllm::resolve_ip_binary,
    opencode::{
        CHAT_PATH, EXAMPLE_CONFIG, GUIDE, GUIDE_PI_HEADING, assert_requests_carry_injected_credential, fenced_blocks,
        fixture_path, gateway_config, gateway_credential, gateway_password, header_value, pin_value_in, repo_root,
    },
};

// -----------------------------------------------------------------------------
// Pins and constants
// -----------------------------------------------------------------------------

/// Environment variable holding the path to the pinned pi executable.
pub(super) const PI_BIN_ENV: &str = "PRAXIS_TEST_PI_BIN";
/// Pin manifest, relative to this crate's `fixtures/` directory.
pub(super) const PIN_MANIFEST: &str = "pi-cli/pin.toml";
/// The documented `models.json`, mirrored from the guide.
const MODELS_FIXTURE: &str = "pi-cli/models.json";
/// Provider id the documented `models.json` declares.
pub(super) const PROVIDER_ID: &str = "praxis";
/// Served model name, addressed by the client as `<provider>/<model>`.
const MODEL_ID: &str = "qwen3-8b";
/// The gateway URL the documented configuration points at.
const DOC_BASE_URL: &str = "http://127.0.0.1:8080/v1";
/// Environment variable the documented `Authorization` header reads.
const TOKEN_ENV: &str = "PRAXIS_GATEWAY_BASIC_TOKEN";
/// Prefix of the error pi reports when that header cannot be resolved.
const UNRESOLVED_HEADER_ERROR: &str = r#"Failed to resolve provider "praxis" header "Authorization""#;
/// Fixed prompt for the text lanes.
const PROMPT: &str = "Reply with exactly PONG. Do not call any tools.";
/// Text the scripted backend streams back.
const REPLY: &str = "PONG";
/// Bound on one child run. The client starts in well under a second.
const CHILD_TIMEOUT: Duration = Duration::from_secs(60);
/// Scripted responses to queue. A text turn costs one request and a tool round
/// trip two; the surplus absorbs a transport retry so an exhausted script never
/// masquerades as a protocol failure.
const SCRIPTED_RESPONSES: usize = 8;

/// Hermetic flags every run passes. Mirrors `[pi.launch] hermetic_flags`.
const HERMETIC_FLAGS: &[&str] = &[
    "--offline",
    "--no-extensions",
    "--no-mcp",
    "--no-skills",
    "--no-prompt-templates",
    "--no-themes",
    "--no-context-files",
    "--no-session",
    "--no-approve",
];

/// Keys pi accepts on a `models.json` provider. Mirrors
/// `[pi.provider] provider_keys`. Unknown keys are silently ignored, so this
/// allowlist is the only guard against a misspelled setting.
const PROVIDER_KEYS: &[&str] = &[
    "name",
    "baseUrl",
    "apiKey",
    "api",
    "oauth",
    "headers",
    "compat",
    "authHeader",
    "models",
    "modelOverrides",
];

/// Keys pi accepts on a `models.json` model entry. Mirrors
/// `[pi.provider] model_keys`.
const MODEL_KEYS: &[&str] = &[
    "id",
    "name",
    "api",
    "baseUrl",
    "reasoning",
    "thinkingLevelMap",
    "input",
    "inputLimits",
    "cost",
    "promptCache",
    "contextWindow",
    "maxTokens",
    "samplingParams",
    "samplingParamsByThinkingLevel",
    "headers",
    "compat",
];

/// Which variant of the documented configuration a run exercises.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Lane {
    /// The documented `models.json`, with the gateway token exported.
    Documented,
    /// The documented `models.json` without its `headers` block, so only the
    /// placeholder `apiKey` is sent and the gateway must reject it.
    ApiKeyOnly,
    /// The documented `models.json` with the gateway token NOT exported, so the
    /// header cannot resolve and pi must refuse to send.
    TokenUnset,
}

impl Lane {
    /// Whether the run exports the gateway token the header reads.
    const fn exports_token(self) -> bool {
        matches!(self, Self::Documented | Self::ApiKeyOnly)
    }
}

// -----------------------------------------------------------------------------
// Client configuration
// -----------------------------------------------------------------------------

/// Read a manifest value from `pi-cli/pin.toml`.
pub(super) fn pin_value(key: &str) -> String {
    pin_value_in(PIN_MANIFEST, key)
}

/// The Basic token the documented header reads: the gateway credential
/// without its `Basic ` scheme prefix.
fn gateway_token() -> String {
    gateway_credential()
        .strip_prefix("Basic ")
        .expect("the gateway credential should use the Basic scheme")
        .to_owned()
}

/// The documented `models.json`, parsed.
fn documented_models_json() -> Value {
    let text = std::fs::read_to_string(fixture_path(MODELS_FIXTURE))
        .unwrap_or_else(|error| panic!("read fixtures/{MODELS_FIXTURE}: {error}"));
    serde_json::from_str(&text).unwrap_or_else(|error| panic!("fixtures/{MODELS_FIXTURE} should be JSON: {error}"))
}

/// The client configuration for a lane: the documented fixture with only the
/// gateway URL and served model name patched, plus the one change a lane
/// exists to make.
///
/// Patching the parsed fixture rather than writing a parallel config is what
/// makes "the test runs the documented configuration" true. Nothing is added,
/// so a key the guide leaves out cannot be quietly relied on here.
pub(super) fn client_config(base_url: &str, model: &str, lane: Lane) -> Value {
    let mut config = documented_models_json();
    let provider = config["providers"][PROVIDER_ID]
        .as_object_mut()
        .unwrap_or_else(|| panic!("the documented models.json should declare the {PROVIDER_ID} provider"));
    assert_eq!(
        provider.get("baseUrl").and_then(Value::as_str),
        Some(DOC_BASE_URL),
        "the documented models.json should point at the local gateway the guide uses"
    );
    provider.insert("baseUrl".to_owned(), Value::from(base_url));

    let models = provider
        .get_mut("models")
        .and_then(Value::as_array_mut)
        .expect("the documented models.json should declare a models array");
    assert_eq!(
        models.len(),
        1,
        "the documented models.json should declare exactly one model"
    );
    assert_eq!(
        models[0]["id"].as_str(),
        Some(MODEL_ID),
        "the documented model should be {MODEL_ID}"
    );
    models[0]["id"] = Value::from(model);

    if lane == Lane::ApiKeyOnly {
        provider
            .remove("headers")
            .expect("the documented models.json should carry a headers block to remove");
    }
    config
}

/// Seed `agent_dir` as the client's configuration root for `lane`.
fn seed_agent_dir(agent_dir: &Path, base_url: &str, model: &str, lane: Lane) {
    std::fs::create_dir_all(agent_dir).expect("pi agent directory should be creatable");
    let config =
        serde_json::to_string_pretty(&client_config(base_url, model, lane)).expect("client config should serialize");
    std::fs::write(agent_dir.join("models.json"), config).expect("models.json should be writable");
}

/// Arguments for one headless run.
///
/// The prompt is the final POSITIONAL argument, after `--`, so a prompt that
/// begins with `-` can never be parsed as a flag.
pub(super) fn run_args(model: &str, prompt: &str) -> Vec<String> {
    let mut args = vec!["--print".to_owned(), "--mode".to_owned(), "json".to_owned()];
    args.extend(HERMETIC_FLAGS.iter().map(|flag| (*flag).to_owned()));
    args.extend([
        "--model".to_owned(),
        format!("{PROVIDER_ID}/{model}"),
        "--".to_owned(),
        prompt.to_owned(),
    ]);
    args
}

// -----------------------------------------------------------------------------
// Launch
// -----------------------------------------------------------------------------

/// Everything one headless run needs.
pub(super) struct LaunchSpec<'a> {
    /// The pinned executable.
    pub(super) bin: &'a OsStr,
    /// Temporary HOME; also roots the pi agent directory.
    pub(super) home: &'a Path,
    /// Working directory for the run, where the `bash` tool executes.
    pub(super) workspace: &'a Path,
    /// Praxis base URL, including the `/v1` suffix.
    pub(super) base_url: &'a str,
    /// Served model name, addressed as `<provider>/<model>`.
    pub(super) model: &'a str,
    /// Which configuration variant to run.
    pub(super) lane: Lane,
    /// The positional prompt.
    pub(super) prompt: &'a str,
    /// Bound on the child.
    pub(super) timeout: Duration,
    /// Linux network namespace to launch the client in, if any. The CPU lane
    /// runs the whole test inside one instead, so it passes `None`.
    pub(super) netns: Option<&'a str>,
}

/// Launch the pinned client against `base_url` and capture its output.
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
        netns,
    } = spec;
    let agent_dir = home.join(".pi").join("agent");
    seed_agent_dir(&agent_dir, base_url, model, lane);

    let mut command = match netns {
        // `ip` is resolved to an absolute path because the child's cleared PATH
        // omits /usr/sbin. The launcher stays inside the child's process group,
        // so timeout cleanup reaps the whole tree, and it passes the cleared
        // environment through to pi unchanged.
        Some(namespace) => {
            let mut command = tokio::process::Command::new(resolve_ip_binary());
            command.arg("netns").arg("exec").arg(namespace).arg(bin);
            command
        },
        None => tokio::process::Command::new(bin),
    };
    command
        .args(run_args(model, prompt))
        .current_dir(workspace)
        .env_clear()
        .env("HOME", home)
        // Pin the configuration root explicitly rather than relying on the
        // HOME-derived default, so a real `~/.pi` (stored credentials,
        // settings, extensions) can never be reached.
        .env("PI_CODING_AGENT_DIR", &agent_dir)
        .env("XDG_CONFIG_HOME", home.join(".config"))
        .env("XDG_DATA_HOME", home.join(".local").join("share"))
        .env("XDG_CACHE_HOME", home.join(".cache"))
        .env("PATH", "/usr/local/bin:/usr/bin:/bin")
        .env("CI", "1")
        .env("TERM", "dumb")
        .env("NO_COLOR", "1")
        // Latency hygiene, not correctness: the pinned client attempts no
        // egress in headless mode even without these, verified in a
        // loopback-only network namespace and against a logging proxy.
        .env("PI_OFFLINE", "1")
        .env("PI_SKIP_VERSION_CHECK", "1")
        .env("PI_TELEMETRY", "0")
        // STDIN MUST BE NULL. With stdin open `pi --print` blocks waiting for
        // EOF before issuing a single request; with stdin piped it prepends
        // whatever it reads to the prompt.
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    if lane.exports_token() {
        command.env(TOKEN_ENV, gateway_token());
    }
    configure_isolated_process_group(&mut command);

    let child = command.spawn().expect("pinned pi executable should start");
    capture_child_output(child, timeout).await
}

/// Assert the child finished within its bound, surfacing both pipes on failure.
pub(super) fn assert_not_timed_out(output: &CapturedChildOutput, scenario: &str, timeout: Duration) {
    assert!(
        !output.timed_out,
        "{scenario}: pi exceeded {timeout:?}. If no request reached the backend, check that stdin \
         is /dev/null — the client blocks waiting for EOF otherwise.\nSTDOUT:\n{}\nSTDERR:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

// -----------------------------------------------------------------------------
// Client event trace
// -----------------------------------------------------------------------------

/// The client's `--mode json` event stream.
///
/// Deliberately lenient: it keeps only lines that parse as JSON objects and
/// reads only the fields the assertions need. Hard assertions on the full
/// event schema break on every client release and would make this suite a
/// liability rather than a guard.
pub(super) struct PiTrace {
    events: Vec<Value>,
}

impl PiTrace {
    /// Parse the event stream, discarding anything that is not a JSON object.
    pub(super) fn parse(stdout: &str) -> Self {
        let events = stdout
            .lines()
            .filter_map(|line| serde_json::from_str::<Value>(line.trim()).ok())
            .filter(Value::is_object)
            .collect();
        Self { events }
    }

    /// Every `type` discriminant, in order.
    pub(super) fn event_types(&self) -> Vec<&str> {
        self.events.iter().filter_map(|event| event["type"].as_str()).collect()
    }

    /// Whether the run reached `agent_settled` without being aborted, which is
    /// how the client marks that it has no remaining work. A crash or a kill
    /// leaves this false even when an answer was streamed.
    pub(super) fn settled(&self) -> bool {
        self.events
            .iter()
            .any(|event| event["type"].as_str() == Some("agent_settled") && event["aborted"].as_bool() == Some(false))
    }

    /// The final, authoritative assistant message.
    fn final_assistant(&self) -> Option<&Value> {
        self.events
            .iter()
            .rev()
            .filter(|event| event["type"].as_str() == Some("message_end"))
            .map(|event| &event["message"])
            .find(|message| message["role"].as_str() == Some("assistant"))
    }

    /// `stopReason` of the final assistant message: `stop` on success, `error`
    /// on a failed turn.
    pub(super) fn stop_reason(&self) -> Option<&str> {
        self.final_assistant()?["stopReason"].as_str()
    }

    /// `errorMessage` of the final assistant message, empty when absent.
    pub(super) fn error_message(&self) -> &str {
        self.final_assistant()
            .and_then(|message| message["errorMessage"].as_str())
            .unwrap_or_default()
    }

    /// Text content of the final assistant message.
    pub(super) fn assistant_text(&self) -> String {
        self.final_assistant()
            .and_then(|message| message["content"].as_array())
            .into_iter()
            .flatten()
            .filter(|block| block["type"].as_str() == Some("text"))
            .filter_map(|block| block["text"].as_str())
            .collect()
    }

    /// Names of the tools the client started executing, in order.
    pub(super) fn executed_tools(&self) -> Vec<&str> {
        self.events
            .iter()
            .filter(|event| event["type"].as_str() == Some("tool_execution_start"))
            .filter_map(|event| event["toolName"].as_str())
            .collect()
    }
}

/// Shared checks for a turn that should have completed: the child finished,
/// settled, and ended on `stopReason: "stop"`.
///
/// The stop reason is checked before the exit status because it is the only
/// signal that distinguishes success from failure in JSON mode.
pub(super) fn assert_completed_turn(output: &CapturedChildOutput, scenario: &str, timeout: Duration) -> PiTrace {
    assert_not_timed_out(output, scenario, timeout);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let trace = PiTrace::parse(&stdout);

    assert_eq!(
        trace.stop_reason(),
        Some("stop"),
        "{scenario}: the final assistant message should stop cleanly; errorMessage: {:?} (events: {:?})\n\
         STDERR:\n{stderr}",
        trace.error_message(),
        trace.event_types()
    );
    assert!(
        trace.settled(),
        "{scenario}: the run should reach agent_settled without aborting (events: {:?})",
        trace.event_types()
    );
    assert!(
        output.status.success(),
        "{scenario}: pi should exit 0, got {:?}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}",
        output.status.code()
    );
    trace
}

// -----------------------------------------------------------------------------
// Scripted Chat Completions backend
// -----------------------------------------------------------------------------

/// One streamed Chat Completions chunk.
fn chunk(delta: &Value, finish_reason: Option<&str>) -> String {
    let mut chunk = json!({
        "id": "chatcmpl-pi",
        "object": "chat.completion.chunk",
        "created": 1_677_652_288_u64,
        "model": MODEL_ID,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    });
    if finish_reason.is_some() {
        chunk["usage"] = json!({"prompt_tokens": 9, "completion_tokens": 1, "total_tokens": 10});
    }
    format!("data: {chunk}\n\n")
}

/// A streamed text reply. A body beginning with `data: ` is served as
/// `text/event-stream` and split into incremental chunks, so this exercises the
/// streaming path the client actually uses.
fn pong_sse() -> String {
    let role = chunk(&json!({"role": "assistant", "content": REPLY}), None);
    let stop = chunk(&json!({}), Some("stop"));
    format!("{role}{stop}data: [DONE]\n\n")
}

/// A streamed `bash` tool call whose arguments arrive in a separate delta from
/// the call's name and id, as a real server streams them.
fn tool_call_sse(call_id: &str, command: &str) -> String {
    let start = chunk(
        &json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{"index": 0, "id": call_id, "type": "function", "function": {"name": "bash", "arguments": ""}}],
        }),
        None,
    );
    let arguments = json!({"command": command}).to_string();
    let args = chunk(
        &json!({"tool_calls": [{"index": 0, "function": {"arguments": arguments}}]}),
        None,
    );
    let stop = chunk(&json!({}), Some("tool_calls"));
    format!("{start}{args}{stop}data: [DONE]\n\n")
}

/// A backend queued with enough identical text replies to outlast any retry.
fn text_backend() -> StatefulCapturingBackend {
    StatefulCapturingBackend::new((0..SCRIPTED_RESPONSES).map(|_| (200, pong_sse())).collect())
}

// -----------------------------------------------------------------------------
// Executable-backed acceptance tests
// -----------------------------------------------------------------------------

/// Resolve the pinned executable, or `None` when the lane is not configured.
fn pi_bin() -> Option<OsString> {
    match std::env::var_os(PI_BIN_ENV) {
        Some(bin) if !bin.is_empty() => Some(bin),
        _ => {
            eprintln!("skipping pinned pi acceptance test; {PI_BIN_ENV} is unset");
            None
        },
    }
}

/// Drive one lane end to end through the shipped example and return the
/// captured output and the requests that reached the backend.
async fn run_lane(
    bin: &OsStr,
    backend: StatefulCapturingBackend,
    workspace: &Path,
    lane: Lane,
    prompt: &str,
) -> (CapturedChildOutput, Vec<CapturedRequest>) {
    let backend = backend.start_with_shutdown();
    let proxy_port = free_port();
    let config = gateway_config(proxy_port, backend.port());
    let proxy = start_proxy(&config);

    let home = TempDir::new().expect("temporary HOME should be creatable");
    let base_url = format!("http://{}/v1", proxy.addr());

    let output = launch(LaunchSpec {
        bin,
        home: home.path(),
        workspace,
        base_url: &base_url,
        model: MODEL_ID,
        lane,
        prompt,
        timeout: CHILD_TIMEOUT,
        netns: None,
    })
    .await;
    (output, backend.requests())
}

/// Assert every request reached the backend through the gateway's credential
/// swap and was sent by the pinned client.
fn assert_requests_from_pinned_client(requests: &[CapturedRequest], scenario: &str) {
    assert_requests_carry_injected_credential(requests, scenario);
    let expected = format!("pi/{PI_PINNED_VERSION} ");
    for (index, request) in requests.iter().enumerate() {
        let agent = header_value(&request.headers, "user-agent").unwrap_or_default();
        assert!(
            agent.starts_with(&expected),
            "{scenario} request #{index} should come from pi {PI_PINNED_VERSION}, got user-agent {agent:?}"
        );
        assert!(
            !request.headers.contains(&gateway_token()),
            "{scenario} request #{index} must not leak the gateway token upstream"
        );
    }
}

#[test]
fn pinned_pi_version_check() {
    let Some(bin) = pi_bin() else { return };

    let output = std::process::Command::new(&bin)
        .arg("--version")
        .stdin(Stdio::null())
        .output()
        .expect("pinned pi executable should run --version");

    assert!(
        output.status.success(),
        "pi --version should exit 0, got {:?}",
        output.status.code()
    );
    let reported = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    assert_eq!(
        reported, PI_PINNED_VERSION,
        "pinned pi version mismatch; update PI_PINNED_VERSION, fixtures/{PIN_MANIFEST}, the checksum \
         fixture, and the CI job together"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pinned_pi_completes_turn_with_documented_models_json() {
    let Some(bin) = pi_bin() else { return };
    let workspace = TempDir::new().expect("temporary workspace should be creatable");
    let (output, requests) = run_lane(&bin, text_backend(), workspace.path(), Lane::Documented, PROMPT).await;

    let trace = assert_completed_turn(&output, "documented lane", CHILD_TIMEOUT);
    assert!(
        trace.assistant_text().contains(REPLY),
        "assistant text should carry the scripted reply, got {:?} (events: {:?})",
        trace.assistant_text(),
        trace.event_types()
    );
    assert_eq!(
        requests.len(),
        1,
        "a trivial prompt should cost exactly one request on the pinned client, got {:?}",
        requests.iter().map(|request| &request.uri).collect::<Vec<_>>()
    );
    assert_requests_from_pinned_client(&requests, "documented lane");

    // The documented configuration needs no reasoning or vLLM compat fields,
    // and the guide says so. Pin that observation to the wire.
    let body: Value = serde_json::from_str(&requests[0].body).expect("the request body should be JSON");
    assert_eq!(
        body["model"].as_str(),
        Some(MODEL_ID),
        "the request should name the served model"
    );
    assert_eq!(body["stream"].as_bool(), Some(true), "the client should stream");
    for absent in ["reasoning_effort", "chat_template_kwargs", "thinking_token_budget"] {
        assert!(
            body.get(absent).is_none(),
            "with reasoning left at its default, the client should not send {absent}: {body}"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pinned_pi_runs_a_tool_call_round_trip_through_the_gateway() {
    let Some(bin) = pi_bin() else { return };
    let workspace = TempDir::new().expect("temporary workspace should be creatable");
    let marker = format!("PI_MARKER_{:016x}", u64::from_le_bytes(rand::random::<[u8; 8]>()));
    std::fs::write(workspace.path().join("source.txt"), &marker).expect("source marker should be writable");

    let call_id = "call_pi_cp";
    let responses = std::iter::once((200, tool_call_sse(call_id, "cp source.txt result.txt")))
        .chain((1..SCRIPTED_RESPONSES).map(|_| (200, pong_sse())))
        .collect();
    let (output, requests) = run_lane(
        &bin,
        StatefulCapturingBackend::new(responses),
        workspace.path(),
        Lane::Documented,
        "Copy source.txt to result.txt with the bash tool, then reply PONG.",
    )
    .await;

    let trace = assert_completed_turn(&output, "tool-call lane", CHILD_TIMEOUT);

    // Oracle A: the scripted command really executed on disk.
    let copied = std::fs::read_to_string(workspace.path().join("result.txt")).unwrap_or_default();
    assert_eq!(
        copied,
        marker,
        "the scripted bash call should have copied the marker (events: {:?})",
        trace.event_types()
    );
    // Oracle B: the client attributes the execution to the shell tool the pin
    // manifest names, rather than the file appearing some other way.
    assert_eq!(
        trace.executed_tools(),
        [pin_value("bash_tool").as_str()],
        "exactly one bash execution should be reported"
    );

    assert_eq!(
        requests.len(),
        2,
        "one tool round trip should cost two requests, got {:?}",
        requests.iter().map(|request| &request.uri).collect::<Vec<_>>()
    );
    assert_requests_from_pinned_client(&requests, "tool-call lane");

    // The follow-up turn must carry the tool result back up through the
    // gateway, correlated to the call the backend issued.
    let follow_up: Value = serde_json::from_str(&requests[1].body).expect("the follow-up body should be JSON");
    let messages = follow_up["messages"]
        .as_array()
        .expect("the follow-up should carry messages");
    assert!(
        messages.iter().any(|message| {
            message["role"].as_str() == Some("assistant") && message["tool_calls"][0]["id"].as_str() == Some(call_id)
        }),
        "the follow-up should replay the assistant tool call: {follow_up}"
    );
    assert!(
        messages.iter().any(
            |message| message["role"].as_str() == Some("tool") && message["tool_call_id"].as_str() == Some(call_id)
        ),
        "the follow-up should carry a role:tool result for {call_id}: {follow_up}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pinned_pi_api_key_alone_is_rejected_by_the_gateway() {
    let Some(bin) = pi_bin() else { return };
    // Removing the documented header leaves the client sending
    // `Bearer <apiKey>`. The gateway must reject it, which proves the header is
    // what authenticates the documented lane rather than the placeholder key.
    let workspace = TempDir::new().expect("temporary workspace should be creatable");
    let (output, requests) = run_lane(&bin, text_backend(), workspace.path(), Lane::ApiKeyOnly, PROMPT).await;

    assert_not_timed_out(&output, "api-key-only lane", CHILD_TIMEOUT);
    let trace = PiTrace::parse(&String::from_utf8_lossy(&output.stdout));

    // `basic_auth` runs before routing, so the request never reaches the
    // backend. Zero captured requests is the proof the credential was refused.
    assert!(
        requests.is_empty(),
        "the placeholder apiKey must not authenticate; backend captured {} request(s)",
        requests.len()
    );
    assert_eq!(
        trace.stop_reason(),
        Some("error"),
        "the client must surface the rejection rather than report success (events: {:?})",
        trace.event_types()
    );
    assert!(
        trace.error_message().starts_with("401"),
        "the gateway should reject the bearer-only turn with 401, got {:?}",
        trace.error_message()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pinned_pi_refuses_to_send_without_the_gateway_token() {
    let Some(bin) = pi_bin() else { return };
    let workspace = TempDir::new().expect("temporary workspace should be creatable");
    let (output, requests) = run_lane(&bin, text_backend(), workspace.path(), Lane::TokenUnset, PROMPT).await;

    assert_not_timed_out(&output, "token-unset lane", CHILD_TIMEOUT);
    let trace = PiTrace::parse(&String::from_utf8_lossy(&output.stdout));

    assert!(
        requests.is_empty(),
        "nothing should reach the backend without the gateway token; captured {}",
        requests.len()
    );
    // The error names the unresolved header, which is what distinguishes the
    // client refusing to send from the gateway rejecting an empty credential.
    assert_eq!(trace.stop_reason(), Some("error"), "events: {:?}", trace.event_types());
    let error = trace.error_message();
    assert!(
        error.contains(UNRESOLVED_HEADER_ERROR) && error.contains(TOKEN_ENV),
        "pi should fail closed on the unresolved header, naming {TOKEN_ENV}; got {error:?}"
    );
}

// -----------------------------------------------------------------------------
// Offline tests
//
// These run on every PR with no executable present, so pin, fixture, guide,
// and CI drift is caught even when the acceptance lane is skipped.
// -----------------------------------------------------------------------------

/// Heading that follows the guide's pi section.
const GUIDE_NEXT_HEADING: &str = "\n## Cleanup\n";

/// The guide's pi section.
fn guide_pi_section() -> String {
    super::opencode::guide_section(GUIDE_PI_HEADING, GUIDE_NEXT_HEADING)
}

#[test]
fn pin_toml_version_matches_harness_const() {
    assert_eq!(
        pin_value("version"),
        PI_PINNED_VERSION,
        "the pin manifest and PI_PINNED_VERSION must agree; a bump must update both"
    );
}

#[test]
fn pinned_pi_version_matches_the_major_series_tripwire() {
    let major = PI_PINNED_VERSION.split('.').next().expect("version should be dotted");
    assert_eq!(
        major,
        pin_value("major_series"),
        "crossing a pi major release must re-validate the manifest, not just bump the version"
    );
}

#[test]
fn checksum_fixture_names_the_pinned_release_archive() {
    // Both lanes verify the downloaded archive against this file, so a version
    // bump that forgets it fails here rather than as a checksum mismatch in CI.
    let platform = pin_value("platform");
    let archive = format!("pi-{platform}.tar.gz");
    assert_eq!(
        pin_value("archive"),
        archive,
        "the manifest should name the platform archive"
    );
    assert_eq!(
        pin_value("url"),
        format!("https://github.com/earendil-works/pi/releases/download/v{PI_PINNED_VERSION}/{archive}"),
        "the manifest URL should fetch the pinned version's archive"
    );

    let relative = format!("pi-cli/{PI_PINNED_VERSION}-{platform}.sha256");
    let line = std::fs::read_to_string(fixture_path(&relative))
        .unwrap_or_else(|error| panic!("fixtures/{relative} should exist for the pinned version: {error}"));
    let (digest, file) = line
        .trim()
        .split_once("  ")
        .expect("the checksum fixture should be in `sha256sum` format");
    assert_eq!(digest.len(), 64, "the checksum should be a sha256 hex digest: {digest}");
    assert!(
        digest.chars().all(|c| c.is_ascii_hexdigit()),
        "the checksum should be hex: {digest}"
    );
    assert_eq!(file, archive, "the checksum should name the archive the lanes download");
}

#[test]
fn documented_models_json_matches_fixture() {
    // Every lane builds its client config from this fixture, so this is what
    // makes the configuration in the guide, not a private copy, the thing
    // under test.
    let section = guide_pi_section();
    let blocks = fenced_blocks(&section, "json");
    assert_eq!(
        blocks.len(),
        1,
        "the guide's pi section should publish exactly one JSON block (models.json)"
    );
    let fixture =
        std::fs::read_to_string(fixture_path(MODELS_FIXTURE)).expect("models.json fixture should be readable");
    assert_eq!(
        fixture, blocks[0],
        "fixtures/{MODELS_FIXTURE} must match the models.json published in {GUIDE} byte for byte"
    );
    assert!(
        section.contains(&format!("export {TOKEN_ENV}=")),
        "the guide must tell users to export {TOKEN_ENV}, which the documented header reads"
    );
    assert!(
        section.contains(&format!("verified against pi {PI_PINNED_VERSION}")),
        "the guide must name the pinned pi version its behavioral claims were verified against"
    );
}

#[test]
fn models_json_is_schema_key_safe() {
    // The pinned client silently ignores unknown keys, so a misspelled key is
    // not an error — it just drops the setting. This is the only guard.
    let config = documented_models_json();
    let provider = config["providers"][PROVIDER_ID]
        .as_object()
        .expect("provider entry should be an object");
    for key in provider.keys() {
        assert!(
            PROVIDER_KEYS.contains(&key.as_str()),
            "provider key {key:?} is not in the pinned allowlist {PROVIDER_KEYS:?}"
        );
    }
    for model in provider["models"].as_array().expect("models should be an array") {
        for key in model.as_object().expect("model entry should be an object").keys() {
            assert!(
                MODEL_KEYS.contains(&key.as_str()),
                "model key {key:?} is not in the pinned allowlist {MODEL_KEYS:?}"
            );
        }
    }
    let top_level: Vec<&String> = config
        .as_object()
        .expect("models.json should be an object")
        .keys()
        .collect();
    assert_eq!(
        top_level,
        ["$schema", "providers"],
        "models.json should declare only $schema and providers"
    );
}

/// Read one string array from the pin manifest, which may span several lines.
///
/// Line-oriented like `pin_value_in`: this crate has no `toml` dependency.
fn pin_array(key: &str) -> Vec<String> {
    let manifest = std::fs::read_to_string(fixture_path(PIN_MANIFEST)).expect("pin manifest should be readable");
    let needle = format!("{key} = [");
    let (_, body) = manifest
        .split_once(&needle)
        .unwrap_or_else(|| panic!("pin manifest should declare {key}"));
    let (items, _) = body.split_once(']').expect("a pin array should close with ]");
    items
        .split(',')
        .map(|item| item.trim().trim_matches('"'))
        .filter(|item| !item.is_empty())
        .map(str::to_owned)
        .collect()
}

#[test]
fn schema_allowlists_match_the_manifest() {
    assert_eq!(
        pin_array("provider_keys"),
        PROVIDER_KEYS,
        "PROVIDER_KEYS must mirror [pi.provider] provider_keys"
    );
    assert_eq!(
        pin_array("model_keys"),
        MODEL_KEYS,
        "MODEL_KEYS must mirror [pi.provider] model_keys"
    );
}

#[test]
fn models_json_agrees_with_the_manifest() {
    let config = documented_models_json();
    let provider = &config["providers"][pin_value("id")];
    assert_eq!(provider["api"].as_str(), Some(pin_value("api").as_str()));
    assert_eq!(
        provider["headers"][pin_value("header_name")].as_str(),
        Some(pin_value("header_value").as_str()),
        "the documented header must read the token the manifest names"
    );
    assert_eq!(
        provider["apiKey"].as_str(),
        Some(pin_value("api_key_placeholder").as_str()),
        "pi needs a non-empty apiKey for the model to be available; the header replaces it on the wire"
    );
    let model = &provider["models"][0];
    assert_eq!(model["id"].as_str(), Some(pin_value("model_id").as_str()));
    assert_eq!(
        model["contextWindow"].as_u64().map(|value| value.to_string()),
        Some(pin_value("context_window")),
        "the documented context window must match the manifest"
    );
    assert_eq!(
        model["maxTokens"].as_u64().map(|value| value.to_string()),
        Some(pin_value("max_tokens")),
        "the documented output budget must match the manifest"
    );
    assert!(
        model.get("reasoning").is_none() && provider.get("compat").is_none(),
        "the guide documents that no reasoning or compat settings are needed for vLLM"
    );
}

#[test]
fn lanes_differ_from_the_documented_config_in_one_thing_only() {
    // Each inversion lane must change exactly one thing, or its failure could
    // come from the wrong cause.
    let documented = client_config(DOC_BASE_URL, MODEL_ID, Lane::Documented);
    assert_eq!(
        client_config(DOC_BASE_URL, MODEL_ID, Lane::TokenUnset),
        documented,
        "the token-unset lane must run the documented config unchanged"
    );
    assert!(Lane::Documented.exports_token() && !Lane::TokenUnset.exports_token());

    let mut without_headers = documented;
    without_headers["providers"][PROVIDER_ID]
        .as_object_mut()
        .expect("provider should be an object")
        .remove("headers");
    assert_eq!(
        client_config(DOC_BASE_URL, MODEL_ID, Lane::ApiKeyOnly),
        without_headers,
        "the api-key-only lane must differ only by the missing headers block"
    );
    assert!(
        Lane::ApiKeyOnly.exports_token(),
        "the api-key-only lane must still export the token, so only the header's absence explains a 401"
    );
}

#[test]
fn client_config_patches_only_the_gateway_url_and_model() {
    let config = client_config("http://127.0.0.1:9/v1", "served-name", Lane::Documented);
    let provider = &config["providers"][PROVIDER_ID];
    assert_eq!(provider["baseUrl"].as_str(), Some("http://127.0.0.1:9/v1"));
    assert_eq!(provider["models"][0]["id"].as_str(), Some("served-name"));
    assert_eq!(
        provider["headers"]["Authorization"].as_str(),
        Some(format!("Basic ${{{TOKEN_ENV}}}").as_str()),
        "the credential must stay an environment reference, never a literal"
    );
    assert!(
        !config.to_string().contains(gateway_password()),
        "the client config must not embed the gateway password"
    );
}

#[test]
fn run_args_are_hermetic_and_end_with_the_prompt() {
    let args = run_args(MODEL_ID, PROMPT);
    assert_eq!(
        &args[..3],
        ["--print", "--mode", "json"],
        "headless JSON mode comes first: {args:?}"
    );
    for flag in HERMETIC_FLAGS {
        assert!(
            args.iter().any(|arg| arg == flag),
            "{flag} is required for a hermetic run: {args:?}"
        );
    }
    let model = args
        .iter()
        .position(|arg| arg == "--model")
        .and_then(|index| args.get(index + 1));
    assert_eq!(model.map(String::as_str), Some("praxis/qwen3-8b"));
    assert_eq!(
        &args[args.len() - 2..],
        ["--", PROMPT],
        "the prompt must be the final positional argument after --: {args:?}"
    );
    let manifest = std::fs::read_to_string(fixture_path(PIN_MANIFEST)).expect("pin manifest should be readable");
    for flag in HERMETIC_FLAGS {
        assert!(
            manifest.contains(&format!("\"{flag}\"")),
            "{flag} should be listed in the manifest's hermetic_flags"
        );
    }
}

#[test]
fn gateway_config_routes_chat_completions_explicitly() {
    let config = gateway_config(29_940, 29_941);
    let router = config.filter_chains[0]
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
        "the example must declare an explicit {CHAT_PATH} route for pi"
    );
    assert_eq!(pin_value("example_config"), EXAMPLE_CONFIG);
}

#[test]
fn trace_parser_reads_the_final_assistant_message() {
    let trace = PiTrace::parse(concat!(
        "not json\n",
        "{\"type\":\"session\",\"version\":3}\n",
        "{\"type\":\"message_end\",\"message\":{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"hi\"}]}}\n",
        "{\"type\":\"tool_execution_start\",\"toolName\":\"bash\"}\n",
        "{\"type\":\"message_end\",\"message\":{\"role\":\"assistant\",\"stopReason\":\"stop\",",
        "\"content\":[{\"type\":\"text\",\"text\":\"PO\"},{\"type\":\"text\",\"text\":\"NG\"}]}}\n",
        "{\"type\":\"agent_settled\",\"aborted\":false}\n",
    ));
    assert_eq!(trace.stop_reason(), Some("stop"));
    assert_eq!(trace.assistant_text(), "PONG");
    assert_eq!(trace.executed_tools(), ["bash"]);
    assert!(trace.settled());
    assert!(trace.error_message().is_empty());

    let failed = PiTrace::parse(
        "{\"type\":\"message_end\",\"message\":{\"role\":\"assistant\",\"content\":[],\
         \"stopReason\":\"error\",\"errorMessage\":\"401: {}\"}}\n",
    );
    assert_eq!(failed.stop_reason(), Some("error"));
    assert_eq!(failed.error_message(), "401: {}");
    assert!(!failed.settled(), "a stream without agent_settled has not settled");

    let empty = PiTrace::parse("");
    assert_eq!(empty.stop_reason(), None);
    assert!(empty.assistant_text().is_empty());
}

#[test]
fn scripted_tool_call_streams_arguments_in_a_separate_delta() {
    let sse = tool_call_sse("call_x", "true");
    let chunks: Vec<Value> = sse
        .split("\n\n")
        .filter_map(|event| event.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).expect("each chunk should be JSON"))
        .collect();
    assert_eq!(chunks.len(), 3);
    assert_eq!(
        chunks[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
        "bash"
    );
    assert_eq!(
        chunks[1]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"],
        "{\"command\":\"true\"}"
    );
    assert_eq!(chunks[2]["choices"][0]["finish_reason"], "tool_calls");
    assert!(sse.ends_with("data: [DONE]\n\n"));
}

#[test]
fn pi_lane_is_wired_in_ci() {
    // A Rust test that early-returns reports as PASSED, so a lane whose gate
    // variable is never set would be invisibly dead. Assert CI actually sets it
    // and installs the pinned build.
    let workflow = repo_root().join(".github/workflows/integration.yaml");
    let yaml = std::fs::read_to_string(&workflow).expect("integration workflow should be readable");
    for needle in [
        PI_BIN_ENV,
        &format!("PI_VERSION: \"{PI_PINNED_VERSION}\""),
        "fixtures/pi-cli/",
        // `pi::` alone also matches `examples::vllm_agentic_api::`.
        "pi:: -- --skip examples::",
    ] {
        assert!(
            yaml.contains(needle),
            "{} must contain {needle:?}, or the pi acceptance lane is not really wired",
            workflow.display()
        );
    }
}
