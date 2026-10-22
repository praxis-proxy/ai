// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Black-box acceptance tests for the pinned OpenCode 2.x CLI executable.
//!
//! OpenCode 2.x is a breaking client release proven beside, not instead of,
//! the 1.x lane in [`super::opencode`]. The path is unchanged:
//!
//! ```text
//! OpenCode 2.x ─► Praxis (anthropic/messages-native-vllm.yaml) ─► /v1/chat/completions
//! ```
//!
//! Nothing on that path translates a body, so as with 1.x these tests prove
//! the credential recipes the guide publishes, against the pinned client, end
//! to end through the shipped example config. Every lane builds its client
//! config from the fixtures that [`documented_v2_config_matches_fixtures`] and
//! [`documented_v2_plugin_matches_fixture`] pin byte-for-byte to the guide.
//!
//! Four lanes:
//!
//! * **plugin** — `praxis-auth.ts`, auto-discovered from the config `plugins/` directory, attaches the gateway Basic
//!   credential from a `model.request` hook. The primary documented recipe.
//! * **provider headers** — the provider-level `headers` map does the same thing declaratively, reading a base64 token
//!   through `{env:…}` substitution, with no plugin present.
//! * **plugin absent** — the plugin lane with the plugin file removed. The gateway must reject before the backend sees
//!   anything, which proves the plugin is the credential source.
//! * **guard tripped** — the plugin is present but `PRAXIS_BASE_URL` names another gateway. The hook must throw and the
//!   client must send nothing, which proves the destination guard is enforcing.
//!
//! The last two together replace the 1.x `--pure` inversion: 2.x has no
//! `--pure` and no equivalent environment variable.
//!
//! # Exit status is still not an auth oracle
//!
//! A throwing `model.request` hook is fail-CLOSED on the pinned client: no
//! request, an `error` event, exit `1`. But a plugin that fails to LOAD is
//! fail-OPEN — the client logs [`PLUGIN_FAILURE_LOG`], sends every turn with
//! no `Authorization` header, and exits `0` — and a hook whose `providerID`
//! filter does not match never runs and logs nothing at all. So the plugin
//! lane asserts the header on EVERY captured request and asserts the failure
//! log is absent. See `[opencode.auth.plugin]` in the 2.x pin manifest.
//!
//! # Running locally
//!
//! ```console
//! PRAXIS_TEST_OPENCODE_V2_BIN=/path/to/package/bin/opencode \
//!   cargo test -p praxis-tests-integration --features store-all --test suite opencode_v2::
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
    CapturedChildOutput, CapturedRequest, capture_child_output, configure_isolated_process_group, free_port,
    start_proxy,
};
use serde_json::Value;
use tempfile::TempDir;

use super::{
    harness::OPENCODE_V2_PINNED_VERSION,
    opencode::{
        CHILD_TIMEOUT, EXAMPLE_CONFIG, GUIDE_V2_HEADING, MODEL_ID, OpenCodeTrace, PROMPT, PROVIDER_ID, REPLY,
        assert_not_timed_out, assert_requests_carry_injected_credential, fenced_blocks, fixture_path, gateway_config,
        gateway_credential, gateway_password, guide_section, pin_value_in, repo_root, scripted_backend,
    },
};

// -----------------------------------------------------------------------------
// Pins and constants
// -----------------------------------------------------------------------------

/// Environment variable holding the path to the pinned OpenCode 2.x executable.
///
/// Distinct from the 1.x lane's variable so both majors can be staged on one
/// runner and each lane resolves its own binary by absolute path.
pub(super) const OPENCODE_V2_BIN_ENV: &str = "PRAXIS_TEST_OPENCODE_V2_BIN";
/// The 2.x pin manifest, relative to this crate's `fixtures/` directory.
pub(super) const PIN_MANIFEST: &str = "opencode-cli-v2/pin.toml";
/// The documented 2.x plugin, mirrored from the guide.
const PLUGIN_FIXTURE: &str = "opencode-cli-v2/praxis-auth.ts";
/// The documented 2.x provider config, mirrored from the guide.
const CONFIG_FIXTURE: &str = "opencode-cli-v2/opencode.json";
/// The documented plugin-free provider config, mirrored from the guide.
const HEADERS_FIXTURE: &str = "opencode-cli-v2/opencode-headers.json";
/// Where the plugin lives, relative to the client configuration directory.
///
/// 2.x auto-discovers local plugins from `plugins/`, so the guide's recipe
/// needs no `plugins` config entry at all.
const PLUGIN_INSTALL_PATH: &str = "plugins/praxis-auth.ts";
/// Log line the client emits, only under `--print-logs`, when a plugin fails
/// to load. Logged at WARN, so the run must not filter above that level.
pub(super) const PLUGIN_FAILURE_LOG: &str = "failed to load plugin";
/// Message the documented plugin throws when the destination guard trips.
const GUARD_ERROR: &str = "Praxis baseURL does not match the trusted gateway";
/// Variable the documented provider-headers recipe substitutes.
const TOKEN_ENV: &str = "PRAXIS_GATEWAY_BASIC_TOKEN";
/// A gateway URL the guard must refuse: syntactically valid, never the one the
/// provider resolves.
const UNTRUSTED_BASE_URL: &str = "https://gateway.invalid/v1";
/// The gateway URL and served model the documented config blocks carry.
const DOC_BASE_URL: &str = "http://127.0.0.1:8080/v1";

/// Keys the pinned schema accepts on a provider object.
///
/// The pinned client silently ignores anything else — it does NOT enforce the
/// schema's `additionalProperties: false` — so a typo here would quietly drop
/// a setting rather than fail. Mirrors `[opencode.config] provider_keys`.
const PROVIDER_KEYS: &[&str] = &[
    "canonical",
    "name",
    "env",
    "package",
    "compatibility",
    "settings",
    "headers",
    "body",
    "models",
];

/// Keys the pinned schema accepts on a provider model object. Mirrors
/// `[opencode.config] model_keys`.
const MODEL_KEYS: &[&str] = &[
    "modelID",
    "family",
    "name",
    "releaseDate",
    "compatibility",
    "package",
    "settings",
    "headers",
    "body",
    "capabilities",
    "variants",
    "cost",
    "disabled",
    "limit",
];

/// Which documented recipe, or which inversion of it, a run exercises.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Lane {
    /// `praxis-auth.ts` attaches the credential through `model.request`.
    Plugin,
    /// The provider-level `headers` map carries it, with no plugin present.
    ProviderHeaders,
    /// The plugin lane minus the plugin file: the gateway must reject.
    PluginAbsent,
    /// The plugin is present but distrusts the resolved `baseURL`: the hook
    /// must throw and the client must send nothing.
    GuardTripped,
}

impl Lane {
    /// Whether the documented plugin is installed for this lane.
    const fn seeds_plugin(self) -> bool {
        matches!(self, Self::Plugin | Self::GuardTripped)
    }

    /// The documented config fixture this lane is built from.
    const fn config_fixture(self) -> &'static str {
        match self {
            Self::ProviderHeaders => HEADERS_FIXTURE,
            Self::Plugin | Self::PluginAbsent | Self::GuardTripped => CONFIG_FIXTURE,
        }
    }

    /// The `PRAXIS_BASE_URL` the plugin is told to trust, when the plugin's
    /// environment applies to this lane.
    ///
    /// `PluginAbsent` gets exactly the plugin lane's environment, so the plugin
    /// file is the ONLY difference between a passing and a rejected run.
    fn trusted_url(self, base_url: &str) -> Option<&str> {
        match self {
            Self::Plugin | Self::PluginAbsent => Some(base_url),
            Self::GuardTripped => Some(UNTRUSTED_BASE_URL),
            Self::ProviderHeaders => None,
        }
    }
}

// -----------------------------------------------------------------------------
// Client configuration
// -----------------------------------------------------------------------------

/// The gateway Basic token, without its `Basic ` scheme, as the documented
/// provider-headers recipe expects it in [`TOKEN_ENV`].
fn gateway_token() -> String {
    gateway_credential()
        .strip_prefix("Basic ")
        .expect("the gateway credential should use the Basic scheme")
        .to_owned()
}

/// Parse one documented config fixture.
fn read_fixture_config(relative: &str) -> Value {
    let text = std::fs::read_to_string(fixture_path(relative))
        .unwrap_or_else(|error| panic!("read fixtures/{relative}: {error}"));
    serde_json::from_str(&text).unwrap_or_else(|error| panic!("fixtures/{relative} should be JSON: {error}"))
}

/// The client configuration for a lane: the documented fixture with only the
/// gateway URL and served model name patched.
///
/// Patching the parsed fixture rather than writing a parallel config is what
/// makes "the test runs the documented configuration" true. Nothing is added,
/// so a key that the guide leaves out cannot be quietly relied on here.
pub(super) fn client_config(base_url: &str, model: &str, lane: Lane) -> Value {
    let mut config = read_fixture_config(lane.config_fixture());
    let provider = &mut config["providers"][PROVIDER_ID];
    assert_eq!(
        provider["settings"]["baseURL"].as_str(),
        Some(DOC_BASE_URL),
        "the documented config should point at the local gateway the guide uses"
    );
    provider["settings"]["baseURL"] = Value::from(base_url);

    let models = provider["models"]
        .as_object_mut()
        .expect("the documented config should declare models");
    let entry = models
        .remove(MODEL_ID)
        .unwrap_or_else(|| panic!("the documented config should declare {MODEL_ID}"));
    assert!(
        models.is_empty(),
        "the documented config should declare exactly one model"
    );
    models.insert(model.to_owned(), entry);
    config
}

/// Seed `dir` as the client's configuration directory for `lane`.
///
/// The plugin is copied from the committed fixture rather than written inline,
/// so the executable loads exactly the source the guide publishes.
pub(super) fn seed_config_dir(dir: &Path, base_url: &str, model: &str, lane: Lane) {
    std::fs::create_dir_all(dir.join("plugins")).expect("client configuration directory should be creatable");
    if lane.seeds_plugin() {
        std::fs::copy(fixture_path(PLUGIN_FIXTURE), dir.join(PLUGIN_INSTALL_PATH))
            .expect("documented plugin fixture should be copyable into the plugins directory");
    }
    let config =
        serde_json::to_string_pretty(&client_config(base_url, model, lane)).expect("client config should serialize");
    std::fs::write(dir.join("opencode.jsonc"), config).expect("client configuration should be writable");
}

/// Arguments for one headless run.
///
/// `--standalone` is mandatory: without it the client attaches to a shared
/// background service that outlives the run and can carry another test's
/// configuration. The prompt is the final POSITIONAL argument with NO `--`
/// before it — on the pinned client `run -- <prompt>` sends the prompt twice.
/// [`opencode_v2_run_has_no_double_dash_separator`] guards this.
pub(super) fn run_args(model: &str, prompt: &str) -> Vec<String> {
    vec![
        "run".to_owned(),
        "--standalone".to_owned(),
        "--model".to_owned(),
        format!("{PROVIDER_ID}/{model}"),
        "--format".to_owned(),
        "json".to_owned(),
        // Mandatory: a plugin that fails to load is otherwise silent, and the
        // failure is logged at WARN, so the level must not filter above it.
        "--print-logs".to_owned(),
        "--log-level".to_owned(),
        "warn".to_owned(),
        prompt.to_owned(),
    ]
}

// -----------------------------------------------------------------------------
// Launch
// -----------------------------------------------------------------------------

/// Everything one headless run needs.
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
    /// Which recipe or inversion to exercise.
    pub(super) lane: Lane,
    /// The positional prompt.
    pub(super) prompt: &'a str,
    /// Bound on the child.
    pub(super) timeout: Duration,
}

/// Launch the pinned 2.x client against `base_url` and capture its output.
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
        .args(run_args(model, prompt))
        .current_dir(workspace)
        .env_clear()
        .env("HOME", home)
        // Set every XDG root explicitly rather than relying on HOME-derived
        // defaults, so a stray real credential store can never be reached. 2.x
        // keeps state under XDG_STATE_HOME, which 1.x did not use.
        .env("XDG_CONFIG_HOME", &config_home)
        .env("XDG_DATA_HOME", home.join(".local").join("share"))
        .env("XDG_STATE_HOME", home.join(".local").join("state"))
        .env("XDG_CACHE_HOME", home.join(".cache"))
        .env("PATH", "/usr/local/bin:/usr/bin:/bin")
        .env("CI", "1")
        .env("TERM", "dumb")
        .env("NO_COLOR", "1")
        // Latency hygiene, not correctness: the pinned client completes with no
        // egress at all, verified in a loopback-only network namespace with
        // none of these set. The project-config switch keeps a stray
        // `opencode.json` above the temporary workspace out of the run.
        .env("OPENCODE_DISABLE_MODELS_FETCH", "1")
        .env("OPENCODE_DISABLE_AUTOUPDATE", "1")
        .env("OPENCODE_DISABLE_PROJECT_CONFIG", "1")
        // STDIN MUST BE NULL. With stdin held open the client issues no
        // request and produces no output until it is killed.
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    if let Some(trusted) = lane.trusted_url(base_url) {
        // Read by the documented plugin. `PRAXIS_BASE_URL` must equal the
        // resolved provider `baseURL` exactly or the hook throws.
        command
            .env("GATEWAY_AUTH_PASSWORD", gateway_password())
            .env("PRAXIS_BASE_URL", trusted);
    }
    if lane == Lane::ProviderHeaders {
        command.env(TOKEN_ENV, gateway_token());
    }
    configure_isolated_process_group(&mut command);

    let child = command.spawn().expect("pinned OpenCode 2.x executable should start");
    capture_child_output(child, timeout).await
}

// -----------------------------------------------------------------------------
// Executable-backed acceptance tests
// -----------------------------------------------------------------------------

/// Resolve the pinned executable, or `None` when the lane is not configured.
fn opencode_bin() -> Option<OsString> {
    match std::env::var_os(OPENCODE_V2_BIN_ENV) {
        Some(bin) if !bin.is_empty() => Some(bin),
        _ => {
            eprintln!("skipping pinned OpenCode 2.x acceptance test; {OPENCODE_V2_BIN_ENV} is unset");
            None
        },
    }
}

#[test]
fn pinned_opencode_v2_version_check() {
    let Some(bin) = opencode_bin() else { return };

    let output = std::process::Command::new(&bin)
        .arg("--version")
        .output()
        .expect("pinned OpenCode 2.x executable should run --version");

    assert!(
        output.status.success(),
        "opencode --version should exit 0, got {:?}",
        output.status.code()
    );
    // 2.x prints `opencode v<version>`, not the bare version 1.x prints.
    let reported = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    assert_eq!(
        reported,
        format!("opencode v{OPENCODE_V2_PINNED_VERSION}"),
        "pinned OpenCode 2.x version mismatch; update OPENCODE_V2_PINNED_VERSION, \
         fixtures/{PIN_MANIFEST}, and the CI jobs together"
    );
}

/// Drive one lane end to end and return the captured output and requests.
async fn run_lane(lane: Lane, bin: &OsStr) -> (CapturedChildOutput, Vec<CapturedRequest>) {
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

/// Assert a credential-bearing lane completed the scripted turn cleanly.
fn assert_completed_turn(output: &CapturedChildOutput, scenario: &str) {
    assert_not_timed_out(output, scenario);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let trace = OpenCodeTrace::parse(&stdout);

    // Diagnose from the trace before the exit status: a turn that failed for a
    // protocol reason is far more informative than a bare non-zero code.
    assert!(
        trace.errors().is_empty(),
        "{scenario}: should not surface a client error event: {:?}\nSTDERR:\n{stderr}",
        trace.errors()
    );
    assert!(
        output.status.success(),
        "{scenario}: should exit 0, got {:?}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}",
        output.status.code()
    );
    assert!(
        trace.assistant_text().contains(REPLY),
        "{scenario}: assistant text should carry the scripted reply, got {:?} (events: {:?})",
        trace.assistant_text(),
        trace.event_types()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pinned_opencode_v2_completes_turn_with_plugin_auth() {
    let Some(bin) = opencode_bin() else { return };
    let (output, requests) = run_lane(Lane::Plugin, &bin).await;

    // A plugin that fails to LOAD is fail-open: the turn still completes and
    // exits 0, just unauthenticated. Without this check the lane could pass
    // while the thing it exists to prove was broken.
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains(PLUGIN_FAILURE_LOG),
        "the documented plugin must load cleanly; found {PLUGIN_FAILURE_LOG:?}:\n{stderr}"
    );
    assert_completed_turn(&output, "2.x plugin lane");
    assert_requests_carry_injected_credential(&requests, "2.x plugin");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pinned_opencode_v2_completes_turn_with_provider_headers_auth() {
    let Some(bin) = opencode_bin() else { return };
    let (output, requests) = run_lane(Lane::ProviderHeaders, &bin).await;

    assert_completed_turn(&output, "2.x provider-headers lane");
    assert_requests_carry_injected_credential(&requests, "2.x provider-headers");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pinned_opencode_v2_without_the_plugin_is_rejected() {
    let Some(bin) = opencode_bin() else { return };
    // Same config and same environment as the plugin lane; only the plugin
    // file is missing. A rejection here is what proves the plugin, and not
    // something else in the environment, supplies the credential.
    let (output, requests) = run_lane(Lane::PluginAbsent, &bin).await;

    assert_not_timed_out(&output, "2.x plugin-absent lane");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let trace = OpenCodeTrace::parse(&stdout);

    // `basic_auth` runs before routing, so the unauthenticated turn never
    // reaches the backend. A request that got through would mean something
    // other than the plugin authenticated it.
    assert!(
        requests.is_empty(),
        "without the plugin no credential is attached, so the gateway must reject before the backend sees \
         anything; backend captured {} request(s): {:?}",
        requests.len(),
        requests.iter().map(|request| &request.uri).collect::<Vec<_>>()
    );

    let errors = trace.errors();
    assert_eq!(
        errors.len(),
        1,
        "an unauthenticated turn should surface exactly one client error event, got {:?}",
        trace.event_types()
    );
    // Unlike 1.x, the 2.x event carries no response headers, so the gateway's
    // `basic_auth` realm cannot be checked here. The empty backend capture
    // above is what places the 401 at this gateway rather than another hop.
    assert_eq!(
        errors[0]["error"]["type"].as_str(),
        Some("provider.auth"),
        "the rejection should surface as a provider auth error: {}",
        errors[0]
    );
    assert_eq!(
        errors[0]["error"]["status"].as_u64(),
        Some(401),
        "the gateway should reject the unauthenticated turn with 401: {}",
        errors[0]
    );
    assert_eq!(
        output.status.code(),
        Some(1),
        "a rejected turn should exit 1\nSTDOUT:\n{stdout}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pinned_opencode_v2_guard_refuses_an_untrusted_gateway() {
    let Some(bin) = opencode_bin() else { return };
    // The plugin IS installed, but told to trust a different gateway than the
    // one the provider resolves. On 2.x a throwing `model.request` hook is
    // fail-closed, so the guard must stop the request, not merely withhold the
    // credential as it did on 1.x.
    let (output, requests) = run_lane(Lane::GuardTripped, &bin).await;

    assert_not_timed_out(&output, "2.x guard lane");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let trace = OpenCodeTrace::parse(&stdout);

    // Zero backend requests alone could also mean a gateway rejection; the
    // error message below is what pins the refusal on the guard.
    assert!(
        requests.is_empty(),
        "the guard must stop the request before it leaves the client; backend captured {} request(s)",
        requests.len()
    );
    assert!(
        !stderr.contains(PLUGIN_FAILURE_LOG),
        "the plugin must load for the guard to run; found {PLUGIN_FAILURE_LOG:?}:\n{stderr}"
    );
    let errors = trace.errors();
    assert_eq!(
        errors.len(),
        1,
        "the refused turn should surface exactly one client error event, got {:?}",
        trace.event_types()
    );
    assert_eq!(
        errors[0]["error"]["message"].as_str(),
        Some(GUARD_ERROR),
        "the client error should carry the guard's message: {}",
        errors[0]
    );
    assert_eq!(
        output.status.code(),
        Some(1),
        "a refused turn should exit 1\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
    );
}

// -----------------------------------------------------------------------------
// Offline tests
//
// These run on every PR with no executable present, so pin, fixture, schema,
// and CI drift is caught even when the acceptance lane is skipped.
// -----------------------------------------------------------------------------

/// Read one scalar value from the 2.x pin manifest.
pub(super) fn pin_value(key: &str) -> String {
    pin_value_in(PIN_MANIFEST, key)
}

#[test]
fn pin_toml_version_matches_harness_const() {
    assert_eq!(
        pin_value("version"),
        OPENCODE_V2_PINNED_VERSION,
        "the 2.x pin manifest and OPENCODE_V2_PINNED_VERSION must agree; a bump must update both"
    );
    assert_eq!(
        pin_value("version_output"),
        format!("opencode v{OPENCODE_V2_PINNED_VERSION}"),
        "the manifest's recorded --version line must name the pinned version"
    );
}

#[test]
fn pinned_opencode_v2_version_is_a_two_x_release() {
    // The `model.request` hook and `providers` shape are 2.x contracts; a 3.x
    // client needs a re-validated manifest, not a version bump.
    let major = OPENCODE_V2_PINNED_VERSION
        .split('.')
        .next()
        .expect("version should be dotted");
    assert_eq!(major, "2", "the 2.x lane must pin a 2.x release");
    assert_eq!(
        pin_value("major_series"),
        "2",
        "the 2.x manifest's major_series tripwire must agree with the pinned version"
    );
}

#[test]
fn checksum_fixture_names_the_pinned_platform_tarball() {
    // Both lanes verify the downloaded tarball against this file, so a version
    // bump that forgets it fails here rather than as a checksum mismatch in CI.
    let platform = pin_value("platform");
    let relative = format!("opencode-cli-v2/{OPENCODE_V2_PINNED_VERSION}-{platform}.sha256");
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
    assert_eq!(
        file,
        format!("cli-{platform}-{OPENCODE_V2_PINNED_VERSION}.tgz"),
        "the checksum should name the scoped platform tarball the lanes download"
    );
}

#[test]
fn documented_v2_plugin_matches_fixture() {
    // The acceptance lane loads the fixture, so without this assertion the
    // suite would validate its own private copy of the plugin rather than the
    // one the guide tells users to create.
    let section = guide_section(GUIDE_V2_HEADING, "\n## Cleanup\n");
    let blocks = fenced_blocks(&section, "typescript");
    assert_eq!(
        blocks.len(),
        1,
        "the guide's OpenCode 2.x section should publish exactly one TypeScript block (the Praxis auth plugin)"
    );
    let fixture = std::fs::read_to_string(fixture_path(PLUGIN_FIXTURE)).expect("plugin fixture should be readable");
    assert_eq!(
        fixture, blocks[0],
        "fixtures/{PLUGIN_FIXTURE} must match the 2.x plugin published in the guide byte for byte"
    );
    assert!(
        section.contains(&format!("~/.config/opencode/{PLUGIN_INSTALL_PATH}")),
        "the guide must tell users to install the plugin where the lane installs it ({PLUGIN_INSTALL_PATH})"
    );
}

#[test]
fn documented_v2_config_matches_fixtures() {
    // Every lane builds its client config from these fixtures, so this is what
    // makes the configuration in the guide, not a private copy, the thing
    // under test.
    let section = guide_section(GUIDE_V2_HEADING, "\n## Cleanup\n");
    let blocks = fenced_blocks(&section, "json");
    assert_eq!(
        blocks.len(),
        2,
        "the guide's OpenCode 2.x section should publish exactly two JSON blocks \
         (the provider config and the plugin-free alternative)"
    );
    for (block, relative) in blocks.iter().zip([CONFIG_FIXTURE, HEADERS_FIXTURE]) {
        let fixture = std::fs::read_to_string(fixture_path(relative)).expect("config fixture should be readable");
        assert_eq!(
            &fixture, block,
            "fixtures/{relative} must match its block in the guide byte for byte"
        );
    }
}

#[test]
fn v2_config_is_schema_key_safe() {
    // The pinned client silently ignores unknown keys, so a misspelled key is
    // not an error — it just drops the setting. This is the only guard.
    for lane in [Lane::Plugin, Lane::ProviderHeaders] {
        let config = client_config(DOC_BASE_URL, MODEL_ID, lane);
        let provider = config["providers"][PROVIDER_ID]
            .as_object()
            .expect("provider entry should be an object");
        for key in provider.keys() {
            assert!(
                PROVIDER_KEYS.contains(&key.as_str()),
                "{lane:?}: provider key {key:?} is not in the pinned schema allowlist {PROVIDER_KEYS:?}"
            );
        }
        let model = provider["models"][MODEL_ID]
            .as_object()
            .expect("model entry should be an object");
        for key in model.keys() {
            assert!(
                MODEL_KEYS.contains(&key.as_str()),
                "{lane:?}: model key {key:?} is not in the pinned schema allowlist {MODEL_KEYS:?}"
            );
        }
        // 1.x keys that 2.x renamed; carrying one over is the likeliest typo.
        for stale in ["npm", "options"] {
            assert!(
                !provider.contains_key(stale),
                "{lane:?}: {stale:?} is the 1.x spelling and must not appear in a 2.x provider"
            );
        }
        assert!(
            config.get("provider").is_none() && config.get("plugin").is_none(),
            "{lane:?}: the 2.x config must use `providers`, and declare no 1.x `plugin` array"
        );
    }
}

#[test]
fn v2_config_fixtures_agree_with_the_manifest() {
    for relative in [CONFIG_FIXTURE, HEADERS_FIXTURE] {
        let config = read_fixture_config(relative);
        let provider = &config["providers"][PROVIDER_ID];
        assert_eq!(
            provider["package"].as_str(),
            Some(pin_value("package").as_str()),
            "fixtures/{relative} must use the pinned provider package"
        );
        let model = &provider["models"][pin_value("model_id")];
        assert_eq!(
            model["limit"]["context"].as_u64().map(|value| value.to_string()),
            Some(pin_value("context")),
            "fixtures/{relative} context limit must match the manifest"
        );
        assert_eq!(
            model["limit"]["output"].as_u64().map(|value| value.to_string()),
            Some(pin_value("output")),
            "fixtures/{relative} output limit must match the manifest"
        );
        assert!(
            config.get("plugins").is_none(),
            "fixtures/{relative} must not declare plugins: the guide relies on plugins/ auto-discovery"
        );
    }
}

#[test]
fn provider_headers_config_reads_the_token_from_the_environment() {
    let config = client_config(DOC_BASE_URL, MODEL_ID, Lane::ProviderHeaders);
    assert_eq!(
        config["providers"][PROVIDER_ID]["headers"]["Authorization"].as_str(),
        Some(format!("Basic {{env:{TOKEN_ENV}}}").as_str()),
        "the plugin-free recipe must substitute the token from {TOKEN_ENV}, keeping it off disk"
    );
    assert_eq!(
        pin_value("header_value"),
        format!("Basic {{env:{TOKEN_ENV}}}"),
        "the manifest must record the same substitution the fixture uses"
    );
    assert_eq!(
        format!("Basic {}", gateway_token()),
        gateway_credential(),
        "the token the lane exports must reassemble into the gateway credential"
    );
}

#[test]
fn plugin_config_relies_on_the_hook_alone() {
    for lane in [Lane::Plugin, Lane::PluginAbsent, Lane::GuardTripped] {
        let config = client_config(DOC_BASE_URL, MODEL_ID, lane);
        let provider = &config["providers"][PROVIDER_ID];
        assert!(
            provider["headers"].is_null() && provider["models"][MODEL_ID]["headers"].is_null(),
            "{lane:?}: the plugin-based lanes must carry no declarative credential"
        );
    }
}

#[test]
fn inversion_lanes_differ_from_the_plugin_lane_in_one_thing_only() {
    let base_url = "http://127.0.0.1:18080/v1";
    // Plugin absent: same config, same trusted URL, no plugin file.
    assert_eq!(
        client_config(base_url, MODEL_ID, Lane::PluginAbsent),
        client_config(base_url, MODEL_ID, Lane::Plugin),
        "the plugin-absent lane must use the plugin lane's exact config"
    );
    assert_eq!(
        Lane::PluginAbsent.trusted_url(base_url),
        Lane::Plugin.trusted_url(base_url)
    );
    assert!(Lane::Plugin.seeds_plugin() && !Lane::PluginAbsent.seeds_plugin());

    // Guard tripped: plugin present, a different trusted URL.
    assert!(Lane::GuardTripped.seeds_plugin());
    assert_eq!(
        client_config(base_url, MODEL_ID, Lane::GuardTripped),
        client_config(base_url, MODEL_ID, Lane::Plugin),
        "the guard lane must use the plugin lane's exact config"
    );
    assert_ne!(
        Lane::GuardTripped.trusted_url(base_url),
        Some(base_url),
        "the guard lane must distrust the gateway the provider resolves"
    );
    assert_eq!(
        pin_value("guard_error_message"),
        GUARD_ERROR,
        "the manifest must record the guard message the lane asserts"
    );
    let plugin = std::fs::read_to_string(fixture_path(PLUGIN_FIXTURE)).expect("plugin fixture should be readable");
    assert!(
        plugin.contains(GUARD_ERROR),
        "the documented plugin must throw the message the guard lane asserts"
    );
}

#[test]
fn opencode_v2_run_has_no_double_dash_separator() {
    // On the pinned client `run -- <prompt>` duplicates the prompt, and `-p`
    // is an unknown flag. Both are habits carried over from 1.x or Claude.
    let args = run_args(MODEL_ID, PROMPT);

    assert!(
        !args.iter().any(|arg| arg == "--"),
        "2.x duplicates a prompt that follows --: {args:?}"
    );
    assert!(
        !args.iter().any(|arg| arg == "-p" || arg == "--password"),
        "the prompt must never be passed via -p/--password: {args:?}"
    );
    assert_eq!(
        args.last().map(String::as_str),
        Some(PROMPT),
        "the prompt must be the final positional argument: {args:?}"
    );
    assert_eq!(
        args.first().map(String::as_str),
        Some("run"),
        "subcommand should be run"
    );
    assert!(
        args.iter().any(|arg| arg == "--standalone"),
        "--standalone is required, or the client attaches to a background service that outlives the run: {args:?}"
    );
    assert!(
        !args.iter().any(|arg| arg == "--pure"),
        "2.x has no --pure flag: {args:?}"
    );
    let level = args
        .iter()
        .position(|arg| arg == "--log-level")
        .and_then(|index| args.get(index + 1))
        .expect("--log-level should be set");
    assert!(
        args.iter().any(|arg| arg == "--print-logs") && level == &pin_value("log_level"),
        "--print-logs at the manifest's level is required: plugin load failures are logged at WARN: {args:?}"
    );
}

#[test]
fn gateway_config_matches_the_v2_manifest() {
    assert_eq!(
        pin_value("example_config"),
        EXAMPLE_CONFIG,
        "both OpenCode majors ride the same example config"
    );
    assert_eq!(
        pin_value("id"),
        PROVIDER_ID,
        "the 2.x manifest should declare the provider id the configs use"
    );
}

#[test]
fn opencode_v2_lane_is_wired_in_ci() {
    // A Rust test that early-returns reports as PASSED, so a lane whose gate
    // variable is never set would be invisibly dead. Assert CI actually sets it
    // and verifies the download against the committed checksum.
    let workflow = repo_root().join(".github/workflows/integration.yaml");
    let yaml = std::fs::read_to_string(&workflow).expect("integration workflow should be readable");
    for needle in [
        OPENCODE_V2_BIN_ENV,
        OPENCODE_V2_PINNED_VERSION,
        "opencode-cli-v2/",
        "opencode_v2::",
    ] {
        assert!(
            yaml.contains(needle),
            "{} must contain {needle:?}, or the OpenCode 2.x CPU lane is not really wired",
            workflow.display()
        );
    }
}
