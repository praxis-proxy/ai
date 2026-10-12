// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Black-box Claude Code acceptance tests against a real vLLM backend, both
//! ways Praxis can bridge Anthropic Messages to it (issue #1025).
//!
//! A pinned real Claude Code executable completes the same deterministic
//! multi-step coding task through Praxis against a real vLLM backend under two
//! configs that exercise the two production paths:
//!
//! ```text
//! native      Claude Code ─► Praxis (messages-native-vllm.yaml)    ─► vLLM /v1/messages
//! transformed Claude Code ─► Praxis (messages-to-openai-vllm.yaml) ─► vLLM /v1/chat/completions
//! ```
//!
//! The native path routes NATIVE Anthropic Messages traffic (`/v1/messages`,
//! `/v1/messages/count_tokens`, `/v1/models`) straight through with NO body
//! translation. The transformed path rewrites the Anthropic request into OpenAI
//! Chat Completions (and the response back) via
//! `anthropic_messages_to_chat_completions[_stream]`, so a Chat-Completions-only
//! vLLM can serve the same client. Each path runs once with deterministic
//! `acceptEdits` permissions and once with client-initiated auto-mode
//! classification, for four independent scenarios against one shared vLLM
//! container.
//!
//! A fifth scenario covers issue #1418: a read-only PLANNING turn over the
//! native path, asserting the run converges and that no reasoning-channel text
//! reaches the user-visible answer. See [`PLANNING_PROMPT`] for what that
//! scenario does and does not prove.
//!
//! These tests assert what a live end-to-end run uniquely proves: the real
//! client completes the task through Praxis against a real backend. The
//! transformed path additionally asserts the operator-degradation signal from the
//! real run — it scrapes the proxy-owned per-feature degradation counter the
//! filter emits while translating each request from the admin `/metrics` endpoint
//! (see [`assert_degradation_signaled`]), because the task only completes once
//! Praxis degrades the Anthropic-only features the client sends, yet a completed
//! CLI run cannot show that signal directly. Wire fidelity —
//! native passthrough or Chat Completions translation, credential isolation, and
//! streaming semantics — is proven deterministically against controlled fake
//! backends in
//! `tests/integration/tests/suite/examples/anthropic_messages_native_vllm.rs`
//! and `.../anthropic_messages_to_openai_vllm.rs`, not observed here.
//!
//! All five tests are gated on live infrastructure and skip unless every
//! required variable is set. Run them locally with, e.g.:
//!
//! ```console
//! PRAXIS_TEST_CLAUDE_CODE_BIN=/absolute/path/to/claude \
//! PRAXIS_TEST_VLLM_BASE_URL=http://127.0.0.1:8000 \
//! PRAXIS_TEST_VLLM_MODEL=<exact-served-model-name> \
//! VLLM_API_KEY=<backend-bearer-token> \
//!   cargo test -p praxis-tests-integration --test suite \
//!   claude_code_vllm::pinned_claude_code_drives_native_vllm_through_full_flow -- --exact
//! # and, against the same backend's Chat Completions surface:
//! #   claude_code_vllm::pinned_claude_code_drives_transformed_vllm_through_full_flow
//! ```
//!
//! Pin discipline: [`CLAUDE_CODE_VERSION`], [`PermissionScenario`], and
//! [`LAUNCH_FLAGS`] are part of the committed pin manifest
//! (`tests/integration/fixtures/claude-code-cli/`). They MUST be re-validated
//! against the pinned executable when its version changes. The served model,
//! vLLM image digest, and startup request matrix are pinned there too.

use std::{
    ffi::{OsStr, OsString},
    net::{IpAddr, Ipv4Addr},
    path::{Path, PathBuf},
    process::Stdio,
    time::{Duration, SystemTime},
};

use praxis_core::config::Config;
use praxis_test_utils::{
    CapturedChildOutput, ProxyGuard, TavilySearchCapture, basic_auth_header, capture_child_output,
    configure_isolated_process_group, example_config_path, free_port, http_get, start_tavily_relay,
};
use serde_json::Value;

use super::live_vllm::{
    self, BACKEND_TOKEN_ENV, LISTEN_ADDRESS_ENV, VLLM_BASE_URL_ENV, VLLM_MODEL_ENV, authority_of, env_is_truthy,
    resolve_ip_binary,
};

// -----------------------------------------------------------------------------
// Pins and constants
// -----------------------------------------------------------------------------

/// Environment variable holding the absolute path to the pinned Claude Code binary.
const CLAUDE_CODE_BIN_ENV: &str = "PRAXIS_TEST_CLAUDE_CODE_BIN";
/// Optional environment variable naming a Linux network namespace to launch in.
const NETNS_ENV: &str = "PRAXIS_TEST_CLAUDE_CODE_NETNS";
/// Optional environment variable demanding a real live run (no silent skip).
///
/// A Rust test that early-returns reports as PASSED — there is no distinct
/// skipped status. So when a required live variable is missing the test would
/// otherwise yield a false green, e.g. if `sudo --preserve-env` fails to
/// propagate one of them. When this is truthy (`1`/`true`), a missing required
/// variable is a hard failure instead of a skip, so CI cannot pass without
/// actually exercising the flow.
const REQUIRE_LIVE_ENV: &str = "PRAXIS_TEST_CLAUDE_CODE_REQUIRE_LIVE";

/// Environment variable holding the real Tavily API key for the managed
/// `anthropic_web_search` loop. Shared with the Anthropic SDK web-search suite.
const TAVILY_API_KEY_ENV: &str = "TAVILY_API_KEY";
/// Optional environment variable demanding a real live web-search run.
///
/// Like [`REQUIRE_LIVE_ENV`] for the coding paths: when truthy (`1`/`true`), a
/// missing [`TAVILY_API_KEY_ENV`] is a hard failure instead of a skip, so a CI
/// nightly that is supposed to exercise the real Tavily path cannot pass by
/// silently skipping it.
const REQUIRE_LIVE_WEB_SEARCH_ENV: &str = "PRAXIS_TEST_REQUIRE_LIVE_WEB_SEARCH";

/// Pinned Claude Code version substring expected from `claude --version`.
///
/// PIN: confirm the exact string against the pinned executable during the
/// qualification run and update this constant and the manifest together.
const CLAUDE_CODE_VERSION: &str = "2.1.278";

/// Output-token ceiling passed to the pinned client for the 32K vLLM context.
///
/// Claude Code otherwise requests 32K output tokens, which vLLM correctly
/// rejects before inference when the pinned model server has a 32K total
/// context. The coding task needs only short tool calls and a summary.
const CLAUDE_CODE_MAX_OUTPUT_TOKENS: &str = "2048";

/// Context window advertised to the pinned client for its compaction policy.
///
/// Keep this equal to the backend's 32K generation window. Auto-mode classifier
/// calls reserve 2,112 output tokens independently of the main-request cap and
/// include a large client-owned safety prompt, so a smaller backend window can
/// reject them before inference.
const CLAUDE_CODE_MAX_CONTEXT_TOKENS: &str = "32768";

/// Context window advertised to the pinned client for the compaction scenario.
///
/// Set under the backend's 32K generation window so that context grown across
/// accepted turns crosses the client's own auto-compaction threshold (Claude
/// Code compacts as usage approaches its context window) long before vLLM would
/// reject an oversized request. The coding task still fits because
/// [`CLAUDE_CODE_MAX_OUTPUT_TOKENS`] bounds each turn's output.
///
/// CRITICAL headroom constraint: pinned Claude Code 2.1.267 reserves 2,048
/// output tokens and 13,000 compaction tokens, so its effective auto-compaction
/// threshold is `window - 15048`. A single ballast Read must be comfortably
/// SMALLER than that headroom, otherwise one post-compaction read immediately
/// re-crosses the threshold and the client's rapid-refill breaker aborts the
/// run. At 30,000 the usable headroom is ~14,952 tokens, dwarfing a single
/// ballast chapter (~1.1k tokens), while total ballast still exceeds the
/// threshold so compaction fires mid-task.
///
/// Tunable: a single GPU-job pass may be needed to land compaction mid-task
/// against live Qwen3-8B. Keep it under the backend window, above the 15,048
/// reserve, and paired with [`COMPACTION_BALLAST_CHAPTERS`]/[`COMPACTION_BALLAST_BYTES`].
const CLAUDE_CODE_COMPACTION_CONTEXT_TOKENS: &str = "30000";

/// Number of ballast chapters seeded to grow context across accepted turns until
/// the pinned client crosses its lowered auto-compaction threshold mid-task.
///
/// Sized so total ballast (~17.6k tokens) exceeds the ~14,952-token usable
/// threshold, forcing compaction after roughly two-thirds of the reads with
/// several reads still remaining for the post-compaction turns to exercise.
const COMPACTION_BALLAST_CHAPTERS: usize = 16;

/// Approximate bytes per ballast chapter (~1.1k tokens of natural-language text),
/// kept well under the ~14,952-token post-compaction headroom so no single read
/// re-crosses the threshold and trips the client's rapid-refill breaker.
const COMPACTION_BALLAST_BYTES: usize = 4_500;

/// Hard timeout for the multi-turn compaction run. The run reads many ballast
/// chapters one per turn before the task's edit/verify/summary turns, so it
/// needs more wall-clock than the single-shot coding task's [`CHILD_TIMEOUT`].
const COMPACTION_CHILD_TIMEOUT: Duration = Duration::from_secs(300);

/// Shorter deadline for the compaction lane on the known-limited Qwen3-8B model.
/// Real self-compaction fires within the first couple of minutes (the lowered
/// window forces it after the early ballast reads), so this is ample to capture
/// the client's own `compact_boundary` event and a post-compaction request
/// through Praxis while bounding the time wasted when the 8B model then loops
/// re-reading ballast instead of finishing the downstream write. The unfinished
/// write is treated as an expected (XFAIL) model limitation — see
/// [`Workspace::assert_compaction_task_trace`].
const COMPACTION_XFAIL_CHILD_TIMEOUT: Duration = Duration::from_secs(180);

/// Model identifier (case-insensitive substring) whose self-compaction is proven
/// but whose downstream task completion is an accepted XFAIL. Qwen3-8B reliably
/// self-compacts through Praxis but cannot reliably finish the post-compaction
/// marker write: it drops the high-entropy marker across its own lossy summary,
/// or loops re-reading ballast until the deadline. The compaction + continuation
/// proofs stay hard assertions; only task completion is downgraded on this model.
/// Mirrors the Codex lane's `COMPACTION_XFAIL_MODEL`. See
/// docs/developing/gpu-nightly-suite.md.
const COMPACTION_XFAIL_MODEL: &str = "qwen3-8b";

/// Claude Code switch between its server-side and client-initiated auto-mode
/// classifier paths.
const AUTO_MODE_SERVER_ENV: &str = "CLAUDE_CODE_AUTO_MODE_SERVER";

/// The Prometheus counter the `anthropic_messages_to_chat_completions` filter
/// increments once per degraded feature while translating a request, on the
/// request path before the backend responds.
const DEGRADED_COUNTER: &str = "praxis_anthropic_messages_to_chat_completions_degraded_total";

/// Selects how the pinned client authorizes tool calls in one acceptance run.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PermissionScenario {
    /// Stable baseline: Claude Code applies its built-in `acceptEdits` mode and
    /// receives an explicit allowlist for the deterministic workspace tools.
    AcceptEdits,
    /// Exercise Claude Code's auto-mode classifier through Praxis. Setting the
    /// server toggle to `0` makes the client initiate classifier model requests;
    /// Praxis and vLLM do not implement Anthropic's server-side classifier.
    AutoClientClassifier,
    /// Read-only planning: `acceptEdits` with ONLY `Read` preapproved, so the
    /// planning scenario cannot mutate its workspace even if the model ignores
    /// the instruction not to. Paired with [`PLANNING_LAUNCH_FLAGS`], which also
    /// withholds every mutating tool from the request in the first place.
    ReadOnlyPlanning,
    /// Server-side web search: `acceptEdits` with only `WebSearch` preapproved.
    /// The managed `anthropic_web_search` loop resolves the search in Praxis and
    /// suppresses the `WebSearch` tool_use, so the client never actually executes
    /// it; preapproving the one exposed tool keeps the command well-formed.
    WebSearch,
}

impl PermissionScenario {
    /// The value accepted by Claude Code's `--permission-mode` flag.
    const fn cli_value(self) -> &'static str {
        match self {
            Self::AcceptEdits | Self::ReadOnlyPlanning | Self::WebSearch => "acceptEdits",
            Self::AutoClientClassifier => "auto",
        }
    }

    /// Adds the permission arguments for this scenario without changing the
    /// common tool exposure configured by the scenario's launch flags.
    fn configure_arguments(self, command: &mut tokio::process::Command) {
        command.arg("--permission-mode").arg(self.cli_value());
        match self {
            // Qwen may render the required verification as `./verify.sh`, invoke
            // it through a shell, or compose it with an inspection command.
            // Bash is the only shell tool exposed to this temporary,
            // egress-isolated baseline workspace, so approve it without
            // coupling the test to one command spelling.
            Self::AcceptEdits => {
                command.arg("--allowedTools").arg("Read").arg("Edit").arg("Bash");
            },
            Self::ReadOnlyPlanning => {
                command.arg("--allowedTools").arg("Read");
            },
            // The managed loop resolves WebSearch server-side and suppresses the
            // tool_use, so the client never runs it; preapproving the one exposed
            // tool keeps the command honest about what it exposes.
            Self::WebSearch => {
                command.arg("--allowedTools").arg("WebSearch");
            },
            // Auto mode deliberately preapproves nothing; see
            // [`Self::configure_environment`].
            Self::AutoClientClassifier => {},
        }
    }

    /// Applies scenario-specific environment after the command environment has
    /// been cleared. Auto mode deliberately omits `--allowedTools`, so the Edit
    /// and Bash calls cannot bypass classification through preapproval.
    fn configure_environment(self, command: &mut tokio::process::Command) {
        if self == Self::AutoClientClassifier {
            command.env(AUTO_MODE_SERVER_ENV, "0");
        }
    }
}

#[test]
fn permission_scenarios_keep_auto_mode_unapproved_and_client_classified() {
    fn configured_command(scenario: PermissionScenario) -> tokio::process::Command {
        let mut command = tokio::process::Command::new("claude");
        command.env_clear();
        scenario.configure_arguments(&mut command);
        scenario.configure_environment(&mut command);
        command
    }

    for flags in [LAUNCH_FLAGS, PLANNING_LAUNCH_FLAGS, WEB_SEARCH_LAUNCH_FLAGS] {
        assert!(!flags.contains(&"--permission-mode"));
        assert!(!flags.contains(&"--allowedTools"));
    }

    let baseline = configured_command(PermissionScenario::AcceptEdits);
    let baseline_args = baseline
        .as_std()
        .get_args()
        .map(|value| value.to_str().expect("test arguments should be UTF-8"))
        .collect::<Vec<_>>();
    assert_eq!(
        baseline_args,
        [
            "--permission-mode",
            "acceptEdits",
            "--allowedTools",
            "Read",
            "Edit",
            "Bash",
        ]
    );
    assert!(baseline.as_std().get_envs().next().is_none());

    let auto = configured_command(PermissionScenario::AutoClientClassifier);
    let auto_args = auto
        .as_std()
        .get_args()
        .map(|value| value.to_str().expect("test arguments should be UTF-8"))
        .collect::<Vec<_>>();
    assert_eq!(auto_args, ["--permission-mode", "auto"]);
    assert_eq!(
        auto.as_std()
            .get_envs()
            .find(|(name, _)| *name == OsStr::new(AUTO_MODE_SERVER_ENV))
            .and_then(|(_, value)| value),
        Some(OsStr::new("0"))
    );

    // The planning scenario preapproves only Read, so even a model that ignores
    // the prompt cannot mutate the workspace through a preapproved tool.
    let planning = configured_command(PermissionScenario::ReadOnlyPlanning);
    let planning_args = planning
        .as_std()
        .get_args()
        .map(|value| value.to_str().expect("test arguments should be UTF-8"))
        .collect::<Vec<_>>();
    assert_eq!(
        planning_args,
        ["--permission-mode", "acceptEdits", "--allowedTools", "Read"]
    );
    assert!(planning.as_std().get_envs().next().is_none());

    // The web-search scenario preapproves only WebSearch, the single tool it
    // exposes; the managed loop resolves it server-side so it is never run.
    let web_search = configured_command(PermissionScenario::WebSearch);
    let web_search_args = web_search
        .as_std()
        .get_args()
        .map(|value| value.to_str().expect("test arguments should be UTF-8"))
        .collect::<Vec<_>>();
    assert_eq!(
        web_search_args,
        ["--permission-mode", "acceptEdits", "--allowedTools", "WebSearch"]
    );
    assert!(web_search.as_std().get_envs().next().is_none());
}

#[test]
fn planning_launch_flags_withhold_every_mutating_tool() {
    let exposed = PLANNING_LAUNCH_FLAGS
        .windows(2)
        .find(|pair| pair[0] == "--tools")
        .map(|pair| pair[1])
        .expect("the planning launch flags should name the exposed built-in tools");

    assert_eq!(
        exposed, "Read",
        "the planning scenario must expose only Read: a mutating tool in the request would let \
         the run change its own workspace, and the read-only assertion would then be vacuous"
    );
    for tool in MUTATING_TOOLS {
        assert!(!exposed.split(',').any(|name| name == *tool));
    }
}

/// The native-vLLM passthrough example config under test (no body translation).
const CONFIG_NATIVE: &str = "anthropic/messages-native-vllm.yaml";

/// The transformed-vLLM example config under test: Anthropic Messages is
/// translated to OpenAI Chat Completions for a Chat-Completions-only backend.
const CONFIG_TRANSFORMED: &str = "anthropic/messages-to-openai-vllm.yaml";

/// The agentic example config exercising the managed `anthropic_web_search`
/// loop. It routes the model to the Chat Completions backend and runs a
/// server-side web search through the configured provider during the IRR loop.
const CONFIG_WEB_SEARCH: &str = "anthropic/full-flow-agentic.yaml";

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
const PROMPT: &str = "Use tools immediately; do not explain before calling them. \
     (1) Read `source/value.txt`. \
     (2) Edit `result/value.txt`: replace the exact text `PLACEHOLDER` with the source text \
     converted to UPPERCASE, with no surrounding whitespace. Do not use the source text as \
     the Edit old_string. \
     (3) You MUST use the Bash tool to run exactly `./verify.sh` and wait for `verify: OK`. \
     Do not give a final answer before that command succeeds. \
     (4) Only then give a concise final summary. /no_think";

/// Hard timeout bounding the whole child run, matching the CI documented bound.
const CHILD_TIMEOUT: Duration = Duration::from_secs(180);

/// Pinned Claude Code launch flags (excluding `-p`, the prompt, and `--model`).
///
/// PIN: these are the real print-mode headless flags accepted by the pinned
/// executable. Restricting the available built-ins to the three tools the task
/// exercises keeps unrelated tool schemas out of the prompt and makes the 32K
/// context pin representative. Re-validate the full set below against the
/// pinned executable during qualification and adjust here and in the manifest
/// together.
const LAUNCH_FLAGS: &[&str] = &[
    "--tools",
    "Read,Edit,Bash",
    "--strict-mcp-config",
    "--output-format",
    "stream-json",
    "--verbose",
    "--max-turns",
    "8",
];

/// The compaction coding prompt: establish the marker, grow context by reading
/// ballast one file per turn until the client auto-compacts, then finish the task.
///
/// The source token is read FIRST (step 1), so it is part of the pre-compaction
/// history the client must carry across its own summarization. Steps 3-5 run
/// AFTER compaction has fired mid-ballast, so completing them proves the
/// compacted session continued through Praxis. The marker is high-entropy and
/// stays on disk, so step 3 re-reads `source/value.txt` to recover the exact
/// token for the write — the test does not depend on the model retaining the
/// token verbatim across its own summarization, only on the task continuing past
/// the compaction boundary. The ballast filenames are enumerated by the caller.
const COMPACTION_PROMPT_PREFIX: &str = "Use tools immediately; do not explain before calling them. \
     (1) Read `source/value.txt` and note its exact text. Do not delete it; you may read it again later. \
     (2) Read every one of these ballast files ONE AT A TIME, a separate Read call per file, in the \
     listed order, to build up the project context — do not stop early and do not read more than one \
     per step: ";

/// The task steps appended after the enumerated ballast list in [`COMPACTION_PROMPT_PREFIX`].
const COMPACTION_PROMPT_SUFFIX: &str = ". \
     (3) After reading ALL ballast files, Read `source/value.txt` again to recover its exact text. \
     (4) Edit `result/value.txt`: replace the exact text `PLACEHOLDER` with the source text from step 3 \
     converted to UPPERCASE, with no surrounding whitespace. Do not use the source text as the Edit \
     old_string. \
     (5) You MUST use the Bash tool to run exactly `./verify.sh` and wait for `verify: OK`. Do not give a \
     final answer before that command succeeds. \
     (6) Only then give a concise final summary. /no_think";

/// Pinned launch flags for the compaction scenario.
///
/// Same exposed tools as [`LAUNCH_FLAGS`] but a higher turn budget: the run reads
/// one ballast chapter per turn before the edit/verify/summary turns, so the 8-turn
/// coding budget would exhaust before the task (and before compaction) completes.
///
/// `--no-session-persistence` keeps each run self-contained: by default Claude
/// Code writes the full conversation to a session JSONL under `~/.claude/`, and
/// disabling it keeps the scenario reproducible without leaving per-run transcript
/// state on the runner between invocations.
const COMPACTION_LAUNCH_FLAGS: &[&str] = &[
    "--tools",
    "Read,Edit,Bash",
    "--strict-mcp-config",
    "--no-session-persistence",
    "--output-format",
    "stream-json",
    "--verbose",
    "--max-turns",
    "32",
];

/// The read-only planning prompt for the issue #1418 regression.
///
/// Issue #1418 reported Claude Code's interactive plan mode looping endlessly
/// against Qwen3-8B served with `--reasoning-parser deepseek_r1` and thinking
/// left on, rendering reasoning text into the plan instead of a finished one.
///
/// SCOPE: this is NOT the interactive plan mode of that report. The pinned
/// executable exposes neither the plan-mode system prompt nor `ExitPlanMode`
/// in any headless transport (`-p`, with or without `--input-format
/// stream-json`), so plan mode cannot be driven from CI. What this scenario
/// does reproduce is the shape of the turn that broke and both of its
/// observable symptoms: a read-only, multi-file planning request whose answer
/// is prose, asserted to converge rather than exhaust the turn budget and to
/// carry no reasoning-channel text in the user-visible answer.
///
/// Deliberately NO `/no_think` suffix, unlike [`PROMPT`]. That marker suppresses
/// Qwen3 thinking from the user turn, which would mask exactly the server-side
/// misconfiguration this scenario exists to catch. Thinking must be off because
/// the backend is served correctly, not because the prompt asked.
const PLANNING_PROMPT: &str = "Read ./deploy.sh, ./backup.sh and ./README.md, then write me a \
     plan to harden the scripts and documentation present in this directory. \
     Do not modify, create, or delete any file: the plan itself is the deliverable. \
     Give the plan as your final answer.";

/// Pinned launch flags for the read-only planning scenario.
///
/// PIN: keep in sync with `[claude_code.launch.planning]` in the manifest.
/// Exposing only `Read` keeps the run incapable of mutating its workspace at
/// the request level, so the read-only assertion is about the client's
/// behaviour and not merely about permissions.
const PLANNING_LAUNCH_FLAGS: &[&str] = &[
    "--tools",
    "Read",
    "--strict-mcp-config",
    "--output-format",
    "stream-json",
    "--verbose",
    "--max-turns",
    "8",
];

/// Built-in tools that would mutate the planning workspace if ever called.
const MUTATING_TOOLS: &[&str] = &["Edit", "Write", "NotebookEdit", "Bash"];

/// Reasoning-channel delimiters that must never reach user-visible output.
///
/// vLLM's `--reasoning-parser` splits a model's thinking block out of the
/// completion into `reasoning_content`. A parser that does not match the served
/// family leaves the raw delimiters and their contents in the assistant text,
/// which is how issue #1418 surfaced: reasoning rendered as the answer.
const REASONING_LEAK_MARKERS: &[&str] = &["<think>", "</think>"];

/// The managed web-search tool name the `anthropic_web_search` filter owns.
///
/// Claude Code's `--tools WebSearch` emits a plain tool named exactly this, which
/// is the name the filter matches to take over the search server-side. It must
/// never appear as a client-visible `tool_use`: the managed loop resolves and
/// suppresses it.
const MANAGED_WEB_SEARCH_TOOL: &str = "WebSearch";

/// The server-side web-search prompt.
///
/// Directs the client to call `WebSearch` exactly once for a current, factual
/// question it cannot answer from memory, so the managed loop in Praxis is
/// exercised against real Tavily. `/no_think` suppresses Qwen3 thinking, matching
/// [`PROMPT`]; this scenario is about the search round-trip, not reasoning.
const WEB_SEARCH_PROMPT: &str = "Use the WebSearch tool to answer. \
     You MUST call the WebSearch tool exactly once before answering; do not answer from memory. \
     Search the web for: who won the most recent FIFA World Cup, and in what year. \
     After you receive the search results, reply with a single concise sentence citing the winner \
     and the year. /no_think";

/// Pinned launch flags for the server-side web-search scenario.
///
/// PIN: keep in sync with `[claude_code.launch.web_search]` in the manifest.
/// Exposing only `WebSearch` keeps the request's single tool the managed one, so
/// the trace cannot be satisfied by any other tool and the only path to an answer
/// is through the server-side search loop.
const WEB_SEARCH_LAUNCH_FLAGS: &[&str] = &[
    "--tools",
    "WebSearch",
    "--strict-mcp-config",
    "--output-format",
    "stream-json",
    "--verbose",
    "--max-turns",
    "8",
];

/// What one scenario asks the client to do, and with which capabilities.
///
/// The three travel together: a prompt is only meaningful alongside the tools
/// the request exposes and the subset of those that are preapproved. Keeping
/// them in one value means a scenario cannot be launched with another's tools.
#[derive(Clone, Copy)]
struct Turn<'a> {
    /// The `-p` prompt text. Borrowed so a scenario can pass a dynamically built
    /// prompt (e.g. the compaction run's enumerated ballast list).
    prompt: &'a str,
    /// Pinned flags, including the exposed built-in tools.
    launch_flags: &'static [&'static str],
    /// Permission mode and the preapproved subset of the exposed tools.
    permission_scenario: PermissionScenario,
    /// Value for `CLAUDE_CODE_MAX_CONTEXT_TOKENS`, the client's compaction window.
    max_context_tokens: &'static str,
    /// Hard wall-clock bound for this turn's child process.
    timeout: Duration,
}

impl Turn<'static> {
    /// The read-only planning turn of the issue #1418 regression.
    const PLANNING: Self = Self {
        prompt: PLANNING_PROMPT,
        launch_flags: PLANNING_LAUNCH_FLAGS,
        permission_scenario: PermissionScenario::ReadOnlyPlanning,
        max_context_tokens: CLAUDE_CODE_MAX_CONTEXT_TOKENS,
        timeout: CHILD_TIMEOUT,
    };
    /// The server-side web-search turn (real Tavily through the managed loop).
    const WEB_SEARCH: Self = Self {
        prompt: WEB_SEARCH_PROMPT,
        launch_flags: WEB_SEARCH_LAUNCH_FLAGS,
        permission_scenario: PermissionScenario::WebSearch,
        max_context_tokens: CLAUDE_CODE_MAX_CONTEXT_TOKENS,
        timeout: CHILD_TIMEOUT,
    };

    /// The deterministic coding task under one of its permission scenarios.
    const fn coding(permission_scenario: PermissionScenario) -> Self {
        Self {
            prompt: PROMPT,
            launch_flags: LAUNCH_FLAGS,
            permission_scenario,
            max_context_tokens: CLAUDE_CODE_MAX_CONTEXT_TOKENS,
            timeout: CHILD_TIMEOUT,
        }
    }
}

impl<'a> Turn<'a> {
    /// The compaction coding task, driven with a lowered context window and a
    /// higher turn budget so growing context forces the client's own
    /// auto-compaction mid-task.
    const fn compaction(prompt: &'a str) -> Self {
        Self {
            prompt,
            launch_flags: COMPACTION_LAUNCH_FLAGS,
            permission_scenario: PermissionScenario::AcceptEdits,
            max_context_tokens: CLAUDE_CODE_COMPACTION_CONTEXT_TOKENS,
            timeout: COMPACTION_CHILD_TIMEOUT,
        }
    }
}

// -----------------------------------------------------------------------------
// Test
// -----------------------------------------------------------------------------

/// Prove the pinned Claude Code client completes a coding task through Praxis
/// against a real native-Anthropic vLLM backend, with NO body translation.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pinned_claude_code_drives_native_vllm_through_full_flow() {
    let Some(live) = LiveConfig::from_env() else {
        return;
    };
    run_full_flow(
        &live,
        native_vllm_config,
        PermissionScenario::AcceptEdits,
        Expectation::NativePassthrough,
    )
    .await;
}

/// Prove the same client completes the same task through Praxis when Praxis
/// TRANSLATES Anthropic Messages into OpenAI Chat Completions for the same vLLM
/// backend's Chat Completions surface.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pinned_claude_code_drives_transformed_vllm_through_full_flow() {
    let Some(live) = LiveConfig::from_env() else {
        return;
    };
    run_full_flow(
        &live,
        transformed_vllm_config,
        PermissionScenario::AcceptEdits,
        Expectation::DegradedTranslation,
    )
    .await;
}

/// Prove Claude Code's client-initiated auto-mode classifier can authorize the
/// same task through the native Anthropic Messages path.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pinned_claude_code_auto_mode_drives_native_vllm_through_full_flow() {
    let Some(live) = LiveConfig::from_env() else {
        return;
    };
    run_full_flow(
        &live,
        native_vllm_config,
        PermissionScenario::AutoClientClassifier,
        Expectation::NativePassthrough,
    )
    .await;
}

/// Prove Claude Code's client-initiated auto-mode classifier can authorize the
/// same task through the translated Chat Completions path.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pinned_claude_code_auto_mode_drives_transformed_vllm_through_full_flow() {
    let Some(live) = LiveConfig::from_env() else {
        return;
    };
    run_full_flow(
        &live,
        transformed_vllm_config,
        PermissionScenario::AutoClientClassifier,
        Expectation::DegradedTranslation,
    )
    .await;
}

/// Prove a read-only planning turn over the native path converges and keeps
/// reasoning out of the user-visible answer (issue #1418).
///
/// Runs on the native path because that is where the issue was reported
/// (`Claude Code -> Praxis /v1/messages -> vLLM /v1/messages`). The defect is a
/// property of how the shared container is served, not of a Praxis filter
/// chain, so one path is enough to guard the serving pins; see
/// [`PLANNING_PROMPT`] for what this does and does not reproduce.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pinned_claude_code_planning_turn_converges_without_reasoning_leakage_on_native_vllm() {
    let Some(live) = LiveConfig::from_env() else {
        return;
    };
    run_planning_flow(&live, native_vllm_config).await;
}

/// Prove the pinned Claude Code client's built-in `WebSearch` tool is resolved
/// SERVER-SIDE by the managed `anthropic_web_search` loop against real Tavily.
///
/// This closes a coverage gap: the other scenarios exercise no web search at all,
/// and Tavily is otherwise only driven through the raw Anthropic SDK, never the
/// real CLI harness. The full chain under test is:
///
/// ```text
/// Claude Code (--tools WebSearch) -> Praxis /v1/messages
///   -> anthropic_web_search managed IRR loop -> real Tavily -> text answer -> CLI
/// ```
///
/// A loopback capture-relay sits between Praxis and Tavily. It is NOT a mock: it
/// forwards every request verbatim to real Tavily and returns Tavily's real
/// response, recording the exchange so the test can prove a live search actually
/// ran. That ground truth is necessary because the managed loop suppresses the
/// `WebSearch` tool_use and falls back to an `is_error` tool result on provider
/// failure, so a client-only "got an answer" assertion would pass even against a
/// broken integration.
///
/// Gated on the full live stack AND a real [`TAVILY_API_KEY_ENV`]; skips cleanly
/// when either is absent unless the matching require-live variable is set.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pinned_claude_code_resolves_server_side_web_search_through_tavily() {
    let Some(live) = LiveConfig::from_env() else {
        // `from_env` only guards against `REQUIRE_LIVE_ENV`, so a web-search run
        // required solely through `REQUIRE_LIVE_WEB_SEARCH_ENV` would otherwise
        // report an incomplete live stack as a PASSED skip here, before
        // `live_tavily_key` ever runs. This gate requires the FULL live stack,
        // not only the Tavily key.
        assert!(
            !env_is_truthy(REQUIRE_LIVE_WEB_SEARCH_ENV),
            "{REQUIRE_LIVE_WEB_SEARCH_ENV} is set but the live stack is incomplete; the server-side \
             web-search acceptance run requires all of {CLAUDE_CODE_BIN_ENV}, {VLLM_BASE_URL_ENV}, \
             {VLLM_MODEL_ENV}, and {BACKEND_TOKEN_ENV} — not only {TAVILY_API_KEY_ENV} — and must not \
             be skipped when a live web-search run is required"
        );
        return;
    };
    let Some(tavily_key) = live_tavily_key() else {
        return;
    };
    run_web_search_flow(&live, &tavily_key).await;
}

/// What a given acceptance path must prove beyond task completion.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Expectation {
    /// Native Anthropic passthrough: no body translation, so no degradation.
    NativePassthrough,
    /// Chat Completions translation with operator-approved degradation: the
    /// transformed config strips the Anthropic-only features Claude Code sends by
    /// default, so the degradation signal must be observable (see
    /// [`assert_degradation_signaled`]).
    DegradedTranslation,
}

/// Prove a pinned Claude Code session crosses its own context limit mid-task,
/// auto-compacts on its own, and still completes the task through Praxis against
/// live vLLM.
///
/// Runs on the native Anthropic Messages path (`Claude Code -> Praxis
/// /v1/messages -> vLLM /v1/messages`): compaction is a client-behavior property,
/// so one path suffices, and the native path is where the client's own
/// summarization call and post-compaction turns traverse Praxis most directly.
/// The client's context window is lowered to
/// [`CLAUDE_CODE_COMPACTION_CONTEXT_TOKENS`] and context is grown by reading
/// ballast one file per turn until the client's own auto-compaction fires,
/// observed from its stream-json `compact_boundary` event. Nothing calls an SDK
/// compact endpoint, injects a canned summary, or enables a Praxis compaction
/// filter.
///
/// The self-compaction and post-compaction continuation through Praxis are always
/// hard assertions. Downstream task completion (the correct post-compaction marker
/// write) is enforced only on a capable model; on [`COMPACTION_XFAIL_MODEL`]
/// (Qwen3-8B) it is an accepted XFAIL — see
/// [`Workspace::assert_compaction_task_trace`] and
/// docs/developing/gpu-nightly-suite.md.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pinned_claude_code_compaction_crosses_context_window_on_native_vllm() {
    let Some(live) = LiveConfig::from_env() else {
        return;
    };
    run_compaction_flow(&live, native_vllm_config).await;
}

/// Drives the pinned client end to end through a Praxis config built by
/// `build_config`, asserting the task completed against the real backend.
///
/// Both acceptance paths and both permission scenarios share this flow; only
/// the config filter chain, client permission setup, and post-run expectation
/// differ.
async fn run_full_flow(
    live: &LiveConfig,
    build_config: fn(&LiveConfig, u16) -> Config,
    permission_scenario: PermissionScenario,
    expectation: Expectation,
) {
    assert_pinned_version(&live.claude_bin).await;
    live.require_egress_isolation_if_demanded();

    let (proxy, admin_address) = start_isolated_proxy(live, build_config);
    let proxy_base_url = format!("http://{}", proxy.addr());

    let workspace = Workspace::create();
    let started = SystemTime::now();
    let output = launch_claude_code(
        live,
        &proxy_base_url,
        workspace.project.path(),
        Turn::coding(permission_scenario),
    )
    .await;
    assert_child_completed(&output, "coding task");

    // Task outcome, proven two independent ways:
    //  * end state — the exact derived file plus the harness-owned verification marker, written only when `verify.sh`
    //    confirms the compare; and
    //  * the work itself — the client's stream-json tool trace shows it Read the distinct input (receiving the per-run
    //    token back through Praxis), performed the distinct Edit writing the uppercase transform, and emitted a final
    //    summary.
    workspace.assert_task_trace(&String::from_utf8_lossy(&output.stdout));
    workspace.assert_task_completed(started);

    // The translated path additionally proves the operator-signal contract: the
    // task above only completes because the Anthropic-only features Claude Code
    // sends by default are degraded rather than rejected. A completed CLI run
    // cannot show the signal directly (the client consumes the response and its
    // headers), so this reads the proxy-owned per-feature degradation counter the
    // filter emitted while translating the real run's requests.
    if expectation == Expectation::DegradedTranslation {
        let admin_address = admin_address
            .as_deref()
            .expect("the transformed config must enable an admin listener for degradation metrics");
        assert_degradation_signaled(admin_address);
    }
}

/// Prove Praxis emitted its degradation signal while translating the real Claude
/// Code run's requests.
///
/// The filter increments a per-feature counter
/// (`DEGRADED_COUNTER{feature="..."}`) each time it degrades an Anthropic feature,
/// on the request-translation path. This reads that proxy-owned counter from the
/// admin `/metrics` endpoint, so the assertions reflect the markers the real
/// client actually sent during its task — not a header the test synthesizes.
///
/// An unmodified Claude Code client sends both Anthropic-only features by default
/// — `cache_control` markers (prompt caching) and a `thinking` request field
/// (extended thinking) — and the completed task above only converged because the
/// transformed config degraded them instead of rejecting the turn. Both per-feature
/// counters must therefore be non-zero. Because the filter increments them while
/// translating the request, before any backend response, these assertions do not
/// depend on the backend's first-byte latency the way asserting on the served reply
/// would.
fn assert_degradation_signaled(admin_address: &str) {
    let prompt_caching = scrape_degraded_count(admin_address, "prompt_caching");
    assert!(
        prompt_caching >= 1,
        "the real Claude Code run must have sent prompt-caching markers that Praxis degraded and \
         counted via {DEGRADED_COUNTER}{{feature=\"prompt_caching\"}}; got {prompt_caching}"
    );
    let extended_thinking = scrape_degraded_count(admin_address, "extended_thinking");
    assert!(
        extended_thinking >= 1,
        "the real Claude Code run must have sent a thinking request field that Praxis degraded and \
         counted via {DEGRADED_COUNTER}{{feature=\"extended_thinking\"}}; got {extended_thinking}"
    );
}

/// Read one labelled sample of [`DEGRADED_COUNTER`] from the admin `/metrics`
/// scrape, returning `0` when the feature has not been degraded yet.
fn scrape_degraded_count(admin_address: &str, feature: &str) -> u64 {
    let (status, body) = http_get(admin_address, "/metrics", None);
    assert_eq!(status, 200, "admin /metrics must be served; got {status}\n{body}");
    let needle = format!("{DEGRADED_COUNTER}{{feature=\"{feature}\"}} ");
    body.lines()
        .find_map(|line| line.strip_prefix(&needle))
        .map_or(0, |value| {
            value
                .trim()
                .parse::<u64>()
                .unwrap_or_else(|error| panic!("parse {DEGRADED_COUNTER} value {value:?}: {error}"))
        })
}

/// Drives one read-only planning turn and asserts it converged cleanly.
async fn run_planning_flow(live: &LiveConfig, build_config: fn(&LiveConfig, u16) -> Config) {
    assert_pinned_version(&live.claude_bin).await;
    live.require_egress_isolation_if_demanded();

    let (proxy, _admin_address) = start_isolated_proxy(live, build_config);
    let proxy_base_url = format!("http://{}", proxy.addr());

    let workspace = PlanningWorkspace::create();
    let output = launch_claude_code(live, &proxy_base_url, workspace.project.path(), Turn::PLANNING).await;
    assert_not_timed_out(&output, "planning turn");

    // Diagnose from the trace BEFORE asserting the exit status. Turn-budget
    // exhaustion — the headless form of the issue #1418 loop — exits nonzero,
    // so a plain status assertion would fail first and report only "exited
    // with 1". The trace assertions name the symptom instead.
    workspace.assert_plan_without_reasoning_leakage(&String::from_utf8_lossy(&output.stdout));
    assert_child_completed(&output, "planning turn");
}

/// Drives one web-search turn and asserts the search was resolved server-side
/// against real Tavily, observed through the capture-relay.
///
/// The relay runs on the host (where Praxis runs and makes its outbound calls),
/// not inside the client's network namespace, so egress isolation is unaffected:
/// the client still reaches only Praxis, and Praxis reaches the loopback relay
/// which reaches real Tavily.
async fn run_web_search_flow(live: &LiveConfig, tavily_key: &str) {
    assert_pinned_version(&live.claude_bin).await;
    live.require_egress_isolation_if_demanded();

    let relay = start_tavily_relay();
    let relay_port = relay.port();
    let (proxy, _admin_address) = start_isolated_proxy(live, |live, proxy_port| {
        web_search_config(live, proxy_port, relay_port, tavily_key)
    });
    let proxy_base_url = format!("http://{}", proxy.addr());

    let project = tempfile::tempdir().expect("temporary web-search project directory should be created");
    let output = launch_claude_code(live, &proxy_base_url, project.path(), Turn::WEB_SEARCH).await;
    assert_not_timed_out(&output, "web-search turn");

    // Diagnose from the trace and the observed Tavily exchange BEFORE asserting
    // the exit status: a backend that never calls WebSearch, or a managed loop
    // that silently fell back to the `is_error` tool result, still exits cleanly,
    // so a bare status assertion would miss the real failure.
    let stdout = String::from_utf8_lossy(&output.stdout);
    if let Err(reason) = check_server_side_web_search(&stdout, &relay.captures(), tavily_key) {
        panic!("{reason}\nstdout:\n{stdout}");
    }
    assert_child_completed(&output, "web-search turn");
}

/// Drives the coding task with a lowered context window and ballast reads so the
/// client crosses its own compaction threshold mid-task, then asserts it
/// auto-compacted and completed the task across the boundary through Praxis.
async fn run_compaction_flow(live: &LiveConfig, build_config: fn(&LiveConfig, u16) -> Config) {
    assert_pinned_version(&live.claude_bin).await;
    live.require_egress_isolation_if_demanded();

    let compaction_is_xfail = live.model.to_ascii_lowercase().contains(COMPACTION_XFAIL_MODEL);

    let (proxy, _admin_address) = start_isolated_proxy(live, build_config);
    let proxy_base_url = format!("http://{}", proxy.addr());

    let workspace = Workspace::create();
    let ballast = workspace.seed_ballast(COMPACTION_BALLAST_CHAPTERS, COMPACTION_BALLAST_BYTES);
    let prompt = format!(
        "{COMPACTION_PROMPT_PREFIX}{list}{COMPACTION_PROMPT_SUFFIX}",
        list = ballast.join(", "),
    );

    let started = SystemTime::now();
    let mut turn = Turn::compaction(&prompt);
    if compaction_is_xfail {
        // The 8B model loops past the full deadline once it cannot finish the
        // write; the wire compaction proof lands well before this, so bound the
        // wasted wall-clock.
        turn.timeout = COMPACTION_XFAIL_CHILD_TIMEOUT;
    }
    let output = launch_claude_code(live, &proxy_base_url, workspace.project.path(), turn).await;
    let stdout = String::from_utf8_lossy(&output.stdout);

    // HARD, always enforced (independent of model capability): the pinned client
    // self-compacted on its own (`compact_boundary` event), the pre-compaction
    // Read carried the per-run marker through Praxis, and at least one tool call
    // followed the boundary — a post-compaction request that necessarily traversed
    // Praxis. These are proven from the client's own stream-json trace and do not
    // depend on the model finishing the downstream task, so they gate the XFAIL
    // path too.
    workspace.assert_compaction_task_trace(&stdout, compaction_is_xfail);

    // XFAIL on the known-limited model: self-compaction is proven above, but
    // Qwen3-8B cannot reliably complete the post-compaction marker write. Record
    // the expected limitation and stop before the completion assertions rather
    // than failing the job. A capable model falls through to the full oracle.
    if compaction_is_xfail {
        eprintln!(
            "XFAIL (model `{model}`, documented Qwen3-8B limitation): verified the pinned Claude \
             Code client self-compacted and a post-compaction request traversed Praxis, but \
             skipping the downstream task-completion assertions. The 8B model drops the marker \
             across its own lossy summary or loops re-reading ballast past the \
             {timeout:?} deadline (timed_out={timed_out}, exit={exit:?}). See \
             docs/developing/gpu-nightly-suite.md.",
            model = live.model,
            timeout = COMPACTION_XFAIL_CHILD_TIMEOUT,
            timed_out = output.timed_out,
            exit = output.status.code(),
        );
        return;
    }

    // Capable model: enforce the full downstream completion oracle. The child must
    // have run to completion inside the deadline and the end-state check proves the
    // marker was used correctly across the boundary.
    assert_child_completed(&output, "compaction task");
    workspace.assert_task_completed(started);
}

/// Starts Praxis for one live run, proving the client cannot bypass it.
///
/// Praxis binds the configured address and forwards Anthropic Messages traffic
/// to the real vLLM backend. Under network isolation Praxis binds the host-side
/// veth address, so the namespaced client can reach only Praxis.
///
/// When a namespace is configured this actively proves the isolated client can
/// reach Praxis and CANNOT reach the public internet, so all Anthropic traffic
/// is forced through the proxy. That is real enforcement, not an advisory
/// `ANTHROPIC_BASE_URL` that a client is free to ignore.
fn start_isolated_proxy(
    live: &LiveConfig,
    build_config: impl FnOnce(&LiveConfig, u16) -> Config,
) -> (ProxyGuard, Option<String>) {
    // The admin listener (set only by the transformed config) is surfaced so
    // the caller can scrape degradation metrics.
    live_vllm::start_isolated_proxy(live.netns.as_deref(), |proxy_port| build_config(live, proxy_port))
}

/// Asserts the pinned client exited on its own rather than being reaped.
///
/// A timeout leaves the captured stdout truncated mid-stream, so every other
/// assertion about the trace would be reasoning about a partial transcript.
fn assert_not_timed_out(output: &CapturedChildOutput, scenario: &str) {
    live_vllm::assert_not_timed_out(output, "Claude Code", scenario, CHILD_TIMEOUT);
}

/// Asserts the pinned client ran to completion inside the acceptance timeout.
fn assert_child_completed(output: &CapturedChildOutput, scenario: &str) {
    assert_not_timed_out(output, scenario);
    assert!(
        output.status.success(),
        "Claude Code exited with {status:?} on the {scenario}\nstdout:\n{stdout}\nstderr:\n{stderr}",
        status = output.status.code(),
        stdout = String::from_utf8_lossy(&output.stdout),
        stderr = String::from_utf8_lossy(&output.stderr),
    );
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
            // A live run demanded by CI must never be silently skipped: an
            // early return reports as PASSED, so a required variable dropped by
            // `sudo --preserve-env` (or otherwise unset) would be a false green.
            assert!(
                !env_is_truthy(REQUIRE_LIVE_ENV),
                "{REQUIRE_LIVE_ENV} is set but a required variable is missing; set all of \
                 {CLAUDE_CODE_BIN_ENV}, {VLLM_BASE_URL_ENV}, {VLLM_MODEL_ENV}, and \
                 {BACKEND_TOKEN_ENV} — the acceptance run must not be skipped when a live run \
                 is required"
            );
            eprintln!(
                "skipping Claude Code vLLM acceptance test; set {CLAUDE_CODE_BIN_ENV}, \
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
        live_vllm::require_egress_isolation_if_demanded(self.netns.as_deref(), NETNS_ENV);
    }
}

/// Reads the real Tavily API key, returning `None` (with a skip message) when it
/// is unset — unless [`REQUIRE_LIVE_WEB_SEARCH_ENV`] is truthy, in which case a
/// missing key is a hard failure so a CI nightly cannot pass by skipping the
/// real-Tavily path.
fn live_tavily_key() -> Option<String> {
    if let Some(key) = std::env::var(TAVILY_API_KEY_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
    {
        return Some(key);
    }
    assert!(
        !env_is_truthy(REQUIRE_LIVE_WEB_SEARCH_ENV),
        "{REQUIRE_LIVE_WEB_SEARCH_ENV} is set but {TAVILY_API_KEY_ENV} is missing or empty; \
         the server-side web-search acceptance run must not be skipped when a live \
         web-search run is required"
    );
    eprintln!(
        "skipping Claude Code server-side web-search acceptance test; \
         set {TAVILY_API_KEY_ENV} (and the full live stack) to run it"
    );
    None
}

/// Build the native-vLLM passthrough config (no body translation).
fn native_vllm_config(live: &LiveConfig, proxy_port: u16) -> Config {
    live_vllm_config(CONFIG_NATIVE, live, proxy_port)
}

/// Build the transformed-vLLM config (Anthropic Messages -> Chat Completions).
///
/// Enables a loopback admin listener so the per-feature degradation counter the
/// filter increments while translating each request can be scraped from
/// `/metrics` after the live run (see [`assert_degradation_signaled`]). The
/// listener binds `127.0.0.1` rather than the proxy's isolation veth, so only the
/// host-side test can reach it — never the namespaced client — and no public-admin
/// opt-in is needed.
fn transformed_vllm_config(live: &LiveConfig, proxy_port: u16) -> Config {
    let mut config = live_vllm_config(CONFIG_TRANSFORMED, live, proxy_port);
    config.admin.address = Some(format!("127.0.0.1:{}", free_port()));
    config
}

/// Load an example config, patching the listener and backend endpoint for a live run.
///
/// The listener binds [`LiveConfig::listen_address`] (loopback by default, the
/// host-side veth address under network isolation) and the backend endpoint is
/// repointed at the real vLLM authority so Praxis forwards straight to it.
/// `credential_injection` resolves `VLLM_API_KEY` from the environment at
/// pipeline build; the live gate guarantees it is set. The gateway `basic_auth`
/// password is inlined from [`gateway_password`] instead of its
/// `GATEWAY_AUTH_PASSWORD` env var, because `std::env::set_var` is `unsafe` (and
/// `unsafe_code` is denied workspace-wide) so the test cannot set it, and the
/// client must present the exact same value. Both example configs share the same
/// listener/backend/gateway placeholders, so this loader serves both paths.
fn live_vllm_config(config: &str, live: &LiveConfig, proxy_port: u16) -> Config {
    let path = example_config_path(config);
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
    Config::from_yaml(&patched).unwrap_or_else(|error| panic!("parse {config}: {error}"))
}

/// Build the agentic web-search config for a live run against real Tavily.
///
/// Patches [`CONFIG_WEB_SEARCH`] so that:
///   * the listener binds [`LiveConfig::listen_address`] on the free proxy port;
///   * the model route marker becomes the live served model, so Claude Code's requests route to the Chat Completions
///     backend;
///   * the Chat Completions backend endpoint repoints at the real vLLM authority;
///   * the `anthropic_web_search` provider key is inlined (the env var cannot be set from the test; see
///     [`live_vllm_config`]) and its `base_url` points at the loopback capture-relay, which forwards to real Tavily;
///     and
///   * `allow_private_upstreams` is enabled so the executor permits the loopback relay callout (SSRF is enforced at
///     connect time, gated by this flag).
///
/// Each replacement is checked so the test fails loudly if the example drifts out
/// from under it rather than silently building an unpatched config.
fn web_search_config(live: &LiveConfig, proxy_port: u16, relay_port: u16, tavily_key: &str) -> Config {
    let path = example_config_path(CONFIG_WEB_SEARCH);
    let yaml = std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {path}: {error}"));
    let listener = format!("{}:{proxy_port}", live.listen_address);

    // Quote the anchor (like the `8001` backend anchor below) so it matches only
    // the listener `address:`, not the two curl-example comment URLs.
    let patched = replace_once(
        &yaml,
        "\"127.0.0.1:8080\"",
        &format!("\"{listener}\""),
        "listener address",
    );
    let patched = replace_once(
        &patched,
        "x-praxis-ai-model: \"Qwen/Qwen3-8B\"",
        &format!("x-praxis-ai-model: \"{}\"", live.model),
        "chat-route model marker",
    );
    let patched = replace_once(
        &patched,
        "\"127.0.0.1:8001\"",
        &format!("\"{}\"", live.vllm_authority),
        "chat-completions backend endpoint",
    );
    // The provider key is inlined (not an env var the test cannot set) and the
    // callout is pointed at the loopback relay. `base_url` must align with the
    // 16-space-indented `api_key` key it follows, or the YAML scanner rejects it.
    let patched = replace_once(
        &patched,
        "api_key: ${WEB_SEARCH_API_KEY}",
        &format!("api_key: {tavily_key}\n                base_url: http://127.0.0.1:{relay_port}"),
        "web-search provider key",
    );
    let patched = replace_once(
        &patched,
        "  allow_private_endpoints: true # example proxies to local backends",
        "  allow_private_endpoints: true # example proxies to local backends\n  allow_private_upstreams: true",
        "insecure_options private-upstream opt-in",
    );

    Config::from_yaml(&patched).unwrap_or_else(|error| panic!("parse {CONFIG_WEB_SEARCH}: {error}"))
}

/// Replaces the single occurrence of `from` with `to`, asserting the example
/// contains `from` exactly once.
///
/// A silent no-op replacement would build a config that still carries a
/// placeholder (an env-var token, the wrong backend, or the shipped listener),
/// so every patch the live run depends on is checked against example drift. The
/// exactly-once assertion additionally guards against an anchor that drift has
/// made ambiguous: were a second occurrence to appear, `str::replace` would
/// silently rewrite both, so the count is pinned here instead.
fn replace_once(haystack: &str, from: &str, to: &str, what: &str) -> String {
    live_vllm::replace_once(haystack, from, to, what, CONFIG_WEB_SEARCH)
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
        // The verifier nonce MUST be independent of `seed`: `verify.sh` embeds it
        // in plaintext, so a nonce derived from `seed` would let the client read
        // `verify.sh` after compaction, recover `seed`, and reconstruct the marker
        // without ever retaining it. A fresh random token shares nothing with the
        // marker, whose only on-disk trace is the preimage-resistant expected.hash.
        let nonce = format!("verified-{}", random_token());

        let project = tempfile::tempdir().expect("temporary project directory should be created");
        let marker_dir = tempfile::tempdir().expect("temporary marker directory should be created");
        let root = project.path();

        std::fs::create_dir(root.join("source")).expect("source directory should be created");
        std::fs::create_dir(root.join("result")).expect("result directory should be created");
        std::fs::write(root.join("source/value.txt"), format!("{source_token}\n"))
            .expect("source value should be written");
        // Deliberately wrong so an unchanged file cannot pass verification.
        std::fs::write(root.join("result/value.txt"), "PLACEHOLDER\n").expect("result value should be written");

        // Precompute the expected digest with the SAME whitespace-stripping
        // pipeline verify.sh uses, so the expected VALUE never appears on disk —
        // only its sha256 does. The client cannot recover the marker by reading
        // expected.hash or the script.
        std::fs::write(
            root.join("expected.hash"),
            format!("{}\n", sha256_without_whitespace(&expected_value)),
        )
        .expect("expected hash should be written");

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

    /// Seed `count` uniquely-filled ballast chapters under `ballast/` of roughly
    /// `approx_bytes` each and return their project-relative paths in order.
    ///
    /// Reading these one per turn grows the client's accumulated context until it
    /// crosses the lowered [`CLAUDE_CODE_COMPACTION_CONTEXT_TOKENS`] window. The
    /// filler never contains the source token or the derived value, so the run's
    /// marker is established only by the early `source/value.txt` Read, not leaked
    /// into ballast.
    fn seed_ballast(&self, count: usize, approx_bytes: usize) -> Vec<String> {
        const WORDS: &[&str] = &[
            "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet", "kilo",
            "lima", "mike", "november", "oscar", "papa", "quebec", "romeo", "sierra", "tango", "uniform", "victor",
            "whiskey", "xray", "yankee", "zulu", "summit", "harbor", "meridian", "quartz", "lantern", "cobalt",
        ];
        let root = self.project.path().join("ballast");
        std::fs::create_dir(&root).expect("ballast directory should be created");
        let mut names = Vec::with_capacity(count);
        for index in 1..=count {
            let name = format!("ballast/chapter_{index:02}.txt");
            let mut body = format!("# ballast chapter {index}\n");
            let mut counter = index;
            while body.len() < approx_bytes {
                body.push_str(WORDS[counter % WORDS.len()]);
                counter += 1;
                if counter % 12 == 0 {
                    body.push('\n');
                } else {
                    body.push(' ');
                }
            }
            body.push('\n');
            std::fs::write(self.project.path().join(&name), body).expect("ballast chapter should be written");
            names.push(name);
        }
        names
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

    /// Assert the client actually READ the distinct input, performed the
    /// distinct EDIT, successfully ran verification, then summarized — proven
    /// from its stream-json tool trace.
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

        trace.successful_verify_bash().unwrap_or_else(|| {
            panic!(
                "client must successfully run verify.sh; Bash commands observed: {:?}\nstdout:\n{stdout}",
                trace.bash_commands(),
            )
        });

        assert!(
            trace.final_summary.is_some(),
            "Claude Code must emit a non-empty final summary in its stream-json output",
        );
    }

    /// Assert the client crossed its context limit, auto-compacted on its own,
    /// and still completed the task correctly across the compaction boundary.
    ///
    /// The definite compaction signal is the client's own stream-json
    /// `compact_boundary` event with `trigger == "auto"` — not an inferred token
    /// count, nor an SDK compact call, nor a Praxis compaction filter. The marker
    /// (`source_token`) is read BEFORE the boundary, so it is part of the history
    /// the client had to summarize; the Edit and verification run AFTER it. Under
    /// the acceptance job's enforced egress isolation the client's only network
    /// route is Praxis, so the tool call the model requests in a post-compaction
    /// turn — and the final answer it produces from that tool's result — are
    /// requests that necessarily traversed Praxis after compaction.
    ///
    /// The oracle has two tiers. ALWAYS hard (independent of model capability):
    /// (a) the client self-compacted (boundary event), (b) the pre-compaction Read
    /// carried the per-run marker through Praxis, and (c) at least one tool call
    /// followed the boundary (continuation through Praxis). On `compaction_is_xfail`
    /// (the Qwen3-8B lane) the assertion stops there — the model self-compacts
    /// reliably but cannot reliably finish the write. On a capable model it also
    /// enforces (d) the post-compaction write is CORRECT (uppercase marker into
    /// `result/value.txt`, then a passing `./verify.sh`). The marker stays on disk
    /// and the prompt steers a post-compaction re-read to recover it, but the oracle
    /// does not mandate that re-read — a capable model may carry the token across its
    /// own summary. Mirrors the Codex lane's XFAIL; see
    /// docs/developing/gpu-nightly-suite.md.
    fn assert_compaction_task_trace(&self, stdout: &str, compaction_is_xfail: bool) {
        let trace = ToolTrace::parse(stdout);

        let compaction = trace.auto_compaction().unwrap_or_else(|| {
            panic!(
                "Claude Code must emit a compact_boundary with trigger=auto, proving the client \
                 compacted on its own after context grew past its lowered window; compaction \
                 triggers observed: {:?}\nstdout:\n{stdout}",
                trace.compaction_triggers(),
            )
        });

        // The marker was established before compaction: the first Read of the
        // source must precede the boundary so it is part of the history the
        // client had to carry across its own summarization.
        let read = trace.find_tool_use("Read", "source/value.txt").unwrap_or_else(|| {
            panic!(
                "client must Read source/value.txt; tool calls observed: {:?}\nstdout:\n{stdout}",
                trace.tool_use_names(),
            )
        });
        assert!(
            read.seq < compaction.seq,
            "the source marker must be read before compaction fires; read seq {} vs compaction seq {}\nstdout:\n{stdout}",
            read.seq,
            compaction.seq,
        );
        let read_result = trace
            .result_for(&read.id)
            .unwrap_or_else(|| panic!("the Read of source/value.txt must produce a tool_result"));
        assert!(
            read_result.text.contains(&self.source_token),
            "the Read tool_result must carry the per-run source token {} delivered back through Praxis; got: {}",
            self.source_token,
            read_result.text,
        );

        // The prompt steers the model to re-read `source/value.txt` on disk after
        // the boundary to recover the exact token for the write, but the oracle
        // does NOT mandate that specific tool call: a capable model may carry the
        // token across its own summary and write it directly. Correctness of the
        // post-compaction write (asserted below) is the requirement; the recovery
        // mechanism is the model's choice. See docs/developing/gpu-nightly-suite.md.

        // A tool call the model requested in a post-compaction turn: under
        // enforced egress isolation this request could only have reached the
        // model through Praxis, so it proves a post-compaction request traversed
        // the proxy.
        let post = trace.first_tool_use_after(compaction.seq).unwrap_or_else(|| {
            panic!(
                "a tool call must follow compaction, proving a post-compaction request traversed \
                 Praxis; tool calls observed: {:?}\nstdout:\n{stdout}",
                trace.tool_use_names(),
            )
        });
        assert!(
            post.seq > compaction.seq,
            "the post-compaction tool call must come after the boundary; got seq {} vs {}",
            post.seq,
            compaction.seq,
        );

        // XFAIL on the known-limited model: self-compaction and post-compaction
        // continuation through Praxis are proven above. Qwen3-8B cannot reliably
        // finish the downstream write, so stop before the completion oracle; the
        // caller logs the expected limitation. A capable model falls through.
        if compaction_is_xfail {
            return;
        }

        // The marker is used correctly after compaction: a SUCCESSFUL Edit writes
        // the uppercase transform after the boundary, and verification runs AFTER
        // that Edit. Requiring the Edit's own tool_result to report success — and
        // the verify to follow it — rejects a trace that carries a correct file
        // and verifier marker from before compaction and then fails the Edit.
        let edit = trace
            .successful_post_compaction_edit("result/value.txt", &self.expected_value, compaction.seq)
            .unwrap_or_else(|| {
                panic!(
                    "client must apply a SUCCESSFUL Edit writing the uppercase value {} into \
                     result/value.txt after compaction (seq {}); tool calls observed: {:?}\nstdout:\n{stdout}",
                    self.expected_value,
                    compaction.seq,
                    trace.tool_use_names(),
                )
            });

        let (verify_use, _) = trace.successful_verify_bash().unwrap_or_else(|| {
            panic!(
                "client must successfully run verify.sh after compaction; Bash commands observed: {:?}\nstdout:\n{stdout}",
                trace.bash_commands(),
            )
        });
        assert!(
            verify_use.seq > edit.seq,
            "verification must run AFTER the successful post-compaction Edit so it checks that write; \
             verify seq {} vs edit seq {}\nstdout:\n{stdout}",
            verify_use.seq,
            edit.seq,
        );

        assert!(
            trace.final_summary.is_some(),
            "Claude Code must emit a non-empty final summary after compaction\nstdout:\n{stdout}",
        );
    }
}

// -----------------------------------------------------------------------------
// Read-only planning workspace (issue #1418)
// -----------------------------------------------------------------------------

/// A small, deliberately unhardened directory for the planning scenario.
///
/// Two shell scripts and a README with obvious hardening gaps give the model
/// something concrete to plan about, mirroring the "harden the scripts and
/// documentation present in this directory" request from the report. Nothing
/// here is ever expected to change: the plan is the only deliverable.
struct PlanningWorkspace {
    /// The temporary project directory the client runs inside.
    project: tempfile::TempDir,
    /// A random per-run token embedded in a script comment. It can only reach
    /// the trace through a Read result, so it proves the client actually read
    /// the files through Praxis rather than planning from the prompt alone.
    source_token: String,
}

impl PlanningWorkspace {
    /// Creates the three files named by [`PLANNING_PROMPT`].
    fn create() -> Self {
        let source_token = format!("praxis-plan-{}", unique_seed());
        let project = tempfile::tempdir().expect("temporary planning directory should be created");
        let root = project.path();

        std::fs::write(
            root.join("deploy.sh"),
            format!("#!/bin/sh\n# {source_token}\nrm -rf $1\ncurl $2 | sh\n"),
        )
        .expect("deploy.sh should be written");
        std::fs::write(root.join("backup.sh"), "#!/bin/sh\ntar czf /tmp/backup.tgz $HOME\n")
            .expect("backup.sh should be written");
        std::fs::write(
            root.join("README.md"),
            "# Ops scripts\n\nRun deploy.sh, then backup.sh.\n",
        )
        .expect("README.md should be written");

        Self { project, source_token }
    }

    /// Assert the planning turn converged, stayed read-only, and kept the
    /// model's reasoning channel out of the user-visible plan.
    fn assert_plan_without_reasoning_leakage(&self, stdout: &str) {
        let trace = ToolTrace::parse(stdout);

        // Symptom 1 of issue #1418 — the client never converges. Headless, an
        // endless loop ends as turn-budget exhaustion rather than an answer,
        // which the client reports as `error_max_turns`.
        assert_eq!(
            trace.result_subtype.as_deref(),
            Some("success"),
            "the planning turn must converge on an answer; the client reported subtype {:?} \
             after {:?} turns, which is how an endless planning loop ends headlessly\nstdout:\n{stdout}",
            trace.result_subtype,
            trace.num_turns,
        );
        assert!(
            !trace.result_is_error,
            "the planning turn must not end in a client-reported error\nstdout:\n{stdout}"
        );

        // Symptom 2 — reasoning rendered as the answer. With a reasoning parser
        // matching the served family and thinking off, the delimiters never
        // reach the assistant text; with a mismatched parser they do.
        //
        // Asserted BEFORE the tool-trace checks below: a backend leaking its
        // reasoning channel also tends to skip the tool calls it was reasoning
        // about, and that missing Read is a consequence, not the cause. Checking
        // the leak first makes the failure name the defect.
        let leaks = trace.reasoning_leaks();
        assert!(
            leaks.is_empty(),
            "reasoning-channel text reached the user-visible answer: {leaks:?}\n\
             the backend is serving thinking the client is not meant to see — check that vLLM runs \
             with a --reasoning-parser matching the served model family and thinking disabled \
             (see tests/integration/fixtures/claude-code-cli/pin.toml)\nstdout:\n{stdout}",
        );

        // The plan must be grounded in the files: a per-run token can only
        // enter the trace through a Read result delivered back through Praxis.
        let read = trace.find_tool_use("Read", "deploy.sh").unwrap_or_else(|| {
            panic!(
                "client must Read deploy.sh before planning; tool calls observed: {:?}\nstdout:\n{stdout}",
                trace.tool_use_names(),
            )
        });
        let read_result = trace
            .result_for(&read.id)
            .unwrap_or_else(|| panic!("the Read of deploy.sh must produce a tool_result"));
        assert!(
            !read_result.is_error,
            "the Read tool_result must not be an error: {}",
            read_result.text,
        );
        assert!(
            read_result.text.contains(&self.source_token),
            "the Read tool_result must carry the per-run token {} delivered back through Praxis; got: {}",
            self.source_token,
            read_result.text,
        );

        // Planning is read-only. `--tools Read` withholds the mutating tools
        // from the request, so observing one here would mean the client
        // obtained it some other way; the files are checked directly too.
        let mutating = trace
            .tool_use_names()
            .into_iter()
            .filter(|name| MUTATING_TOOLS.contains(name))
            .collect::<Vec<_>>();
        assert!(
            mutating.is_empty(),
            "the planning turn must not call a mutating tool; observed: {mutating:?}\nstdout:\n{stdout}",
        );
        self.assert_files_unchanged();

        let plan = trace.final_summary.as_deref().unwrap_or_else(|| {
            panic!("Claude Code must emit the plan as a final summary in its stream-json output\nstdout:\n{stdout}")
        });
        // A low floor on purpose: this guards against a degenerate one-word
        // answer without asserting anything about an 8B model's prose, which
        // would make the nightly flaky for no added signal.
        assert!(
            plan.trim().len() >= 40,
            "the final answer must actually be a plan, not a stub; got: {plan:?}",
        );
    }

    /// Assert the planning turn left every workspace file byte-identical.
    fn assert_files_unchanged(&self) {
        let root = self.project.path();
        for (name, expected) in [
            (
                "deploy.sh",
                format!("#!/bin/sh\n# {}\nrm -rf $1\ncurl $2 | sh\n", self.source_token),
            ),
            ("backup.sh", "#!/bin/sh\ntar czf /tmp/backup.tgz $HOME\n".to_owned()),
            (
                "README.md",
                "# Ops scripts\n\nRun deploy.sh, then backup.sh.\n".to_owned(),
            ),
        ] {
            let actual = std::fs::read_to_string(root.join(name))
                .unwrap_or_else(|error| panic!("{name} should still be readable after a planning turn: {error}"));
            assert_eq!(actual, expected, "the planning turn must not modify {name}");
        }
        let extra = std::fs::read_dir(root)
            .expect("the planning directory should be readable")
            .filter_map(Result::ok)
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .filter(|name| !matches!(name.as_str(), "deploy.sh" | "backup.sh" | "README.md"))
            .collect::<Vec<_>>();
        assert!(
            extra.is_empty(),
            "the planning turn must not create files; found: {extra:?}",
        );
    }
}

/// Writes an executable `verify.sh` embedding the marker path and nonce literally.
/// Returns the sha256 of `value` with all whitespace stripped, computed with the
/// SAME shell pipeline `verify.sh` uses so the digests match exactly. The value
/// is passed via the environment, never interpolated into the shell command.
fn sha256_without_whitespace(value: &str) -> String {
    let output = std::process::Command::new("sh")
        .arg("-c")
        .arg(format!(
            "printf '%s' \"$VALUE\" | tr -d '[:space:]' | {}",
            super::harness::SHA256_DIGEST_SH
        ))
        .env("VALUE", value)
        .output()
        .expect("sha256 pipeline should run");
    assert!(
        output.status.success(),
        "sha256 of the expected value should succeed; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let hash = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    assert!(!hash.is_empty(), "sha256 should produce a non-empty digest");
    hash
}

/// Writes a hash-based `verify.sh`.
///
/// The script compares a sha256 of `result/value.txt` (whitespace-stripped)
/// against the precomputed `expected.hash` — it NEVER reads `source/value.txt`
/// and never prints the expected value. So `cat verify.sh`, `sh -x ./verify.sh`,
/// or any other inspection of the script or its trace cannot leak the marker
/// back into the client's context after compaction.
fn write_verify_script(path: &Path, marker_path: &Path, nonce: &str) {
    let marker = marker_path.display();
    let digest = super::harness::SHA256_DIGEST_SH;
    let script = format!(
        "#!/bin/sh\n\
         set -eu\n\
         actual=$(tr -d '[:space:]' < result/value.txt | {digest})\n\
         expected=$(tr -d '[:space:]' < expected.hash)\n\
         if [ \"$actual\" = \"$expected\" ]; then\n\
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

/// Draws a fresh decimal seed for the task value and nonce.
///
/// A numeric suffix keeps the model's required case conversion focused on the
/// fixed lowercase prefix while preserving a distinct per-run value that can
/// only enter the trace through the Read result.
fn unique_seed() -> String {
    format!("{:010}", rand::random::<u32>())
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
async fn launch_claude_code(
    live: &LiveConfig,
    proxy_base_url: &str,
    project_dir: &Path,
    turn: Turn<'_>,
) -> CapturedChildOutput {
    let config_dir = tempfile::tempdir().expect("temporary CLAUDE_CONFIG_DIR should be created");
    let home_dir = tempfile::tempdir().expect("temporary HOME should be created");
    let mcp_config = config_dir.path().join("empty-mcp.json");
    std::fs::write(&mcp_config, r#"{"mcpServers":{}}"#).expect("empty MCP config should be written");

    let mut command = child_command(live);
    command
        .arg("-p")
        .arg(turn.prompt)
        .arg("--model")
        .arg(&live.model)
        .arg("--mcp-config")
        .arg(&mcp_config)
        .args(turn.launch_flags);
    turn.permission_scenario.configure_arguments(&mut command);
    command
        .current_dir(project_dir)
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
        .env("CLAUDE_CODE_MAX_OUTPUT_TOKENS", CLAUDE_CODE_MAX_OUTPUT_TOKENS)
        .env("CLAUDE_CODE_MAX_CONTEXT_TOKENS", turn.max_context_tokens)
        .env("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
        .env("DISABLE_UPDATES", "1")
        .env("DISABLE_TELEMETRY", "1")
        .env("DISABLE_ERROR_REPORTING", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    turn.permission_scenario.configure_environment(&mut command);
    configure_isolated_process_group(&mut command);

    let child = command.spawn().expect("pinned Claude Code should start");
    capture_child_output(child, turn.timeout).await
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

// -----------------------------------------------------------------------------
// Enforced egress isolation
// -----------------------------------------------------------------------------

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
    /// Zero-based index of the stream-json line this block appeared on, used to
    /// order tool calls against the [`CompactBoundary`] in the same stream.
    seq: usize,
}

/// One `compact_boundary` system event the client emitted when it compacted.
struct CompactBoundary {
    /// `auto` when the client compacted on its own as context approached its
    /// window; `manual` for a user-invoked `/compact`.
    trigger: String,
    /// Tokens the client reported in context just before compacting, if present.
    pre_tokens: Option<u64>,
    /// Zero-based index of the stream-json line this event appeared on.
    seq: usize,
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

/// A parsed view of the tool calls, tool results, assistant text and terminal
/// outcome in Claude Code `--output-format stream-json` output (one JSON object
/// per line).
struct ToolTrace {
    tool_uses: Vec<ToolUse>,
    tool_results: Vec<ToolResult>,
    /// Every `compact_boundary` system event, in stream order.
    compact_boundaries: Vec<CompactBoundary>,
    /// Every user-visible `text` block the assistant emitted. Reasoning the
    /// backend failed to split into its own channel lands here.
    assistant_texts: Vec<String>,
    final_summary: Option<String>,
    /// The terminal `result` envelope's `subtype`, e.g. `success` or
    /// `error_max_turns`.
    result_subtype: Option<String>,
    /// Whether the terminal `result` envelope reported an error.
    result_is_error: bool,
    /// Turns the client took, reported by the terminal `result` envelope.
    num_turns: Option<u64>,
}

impl ToolTrace {
    /// Parses every JSONL line, collecting `tool_use`/`tool_result` blocks,
    /// assistant text, and the terminal `result` envelope. Unparseable lines
    /// are ignored.
    fn parse(stdout: &str) -> Self {
        let mut tool_uses = Vec::new();
        let mut tool_results = Vec::new();
        let mut compact_boundaries = Vec::new();
        let mut assistant_texts = Vec::new();
        let mut final_summary = None;
        let mut result_subtype = None;
        let mut result_is_error = false;
        let mut num_turns = None;

        for (seq, line) in stdout.lines().enumerate() {
            let Ok(value) = serde_json::from_str::<Value>(line.trim()) else {
                continue;
            };
            match value.get("type").and_then(Value::as_str) {
                Some("assistant") => {
                    for block in message_content(&value) {
                        match block.get("type").and_then(Value::as_str) {
                            Some("tool_use") => {
                                if let (Some(id), Some(name)) = (
                                    block.get("id").and_then(Value::as_str),
                                    block.get("name").and_then(Value::as_str),
                                ) {
                                    tool_uses.push(ToolUse {
                                        id: id.to_owned(),
                                        name: name.to_owned(),
                                        input: block.get("input").cloned().unwrap_or(Value::Null),
                                        seq,
                                    });
                                }
                            },
                            Some("text") => {
                                if let Some(text) = block.get("text").and_then(Value::as_str) {
                                    assistant_texts.push(text.to_owned());
                                }
                            },
                            _ => {},
                        }
                    }
                },
                Some("system") if value.get("subtype").and_then(Value::as_str) == Some("compact_boundary") => {
                    compact_boundaries.push(CompactBoundary {
                        trigger: value
                            .pointer("/compact_metadata/trigger")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_owned(),
                        pre_tokens: value.pointer("/compact_metadata/pre_tokens").and_then(Value::as_u64),
                        seq,
                    });
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
                    result_subtype = value
                        .get("subtype")
                        .and_then(Value::as_str)
                        .map(ToOwned::to_owned)
                        .or(result_subtype);
                    result_is_error |= value.get("is_error").and_then(Value::as_bool).unwrap_or(false);
                    num_turns = value.get("num_turns").and_then(Value::as_u64).or(num_turns);
                },
                _ => {},
            }
        }

        Self {
            tool_uses,
            tool_results,
            compact_boundaries,
            assistant_texts,
            final_summary,
            result_subtype,
            result_is_error,
            num_turns,
        }
    }

    /// Every reasoning-channel delimiter that reached user-visible output,
    /// paired with the offending text.
    ///
    /// Covers both the streamed assistant text blocks and the terminal summary:
    /// a mismatched `--reasoning-parser` leaves the thinking block in the
    /// assistant message, so it surfaces in both.
    fn reasoning_leaks(&self) -> Vec<(&'static str, &str)> {
        self.assistant_texts
            .iter()
            .map(String::as_str)
            .chain(self.final_summary.as_deref())
            .flat_map(|text| {
                REASONING_LEAK_MARKERS
                    .iter()
                    .filter(move |marker| text.contains(**marker))
                    .map(move |marker| (*marker, text))
            })
            .collect()
    }

    /// Finds a post-`seq` `Edit` of a file ending in `suffix` that writes
    /// `expected_value` AND whose correlated `tool_result` reports success. A
    /// failed Edit — even one leaving a coincidentally correct file — is rejected,
    /// and retries are tolerated because any successful attempt satisfies it.
    fn successful_post_compaction_edit(&self, suffix: &str, expected_value: &str, seq: usize) -> Option<&ToolUse> {
        self.tool_uses.iter().find(|tool_use| {
            tool_use.name == "Edit"
                && tool_use.seq > seq
                && tool_use
                    .input
                    .get("file_path")
                    .and_then(Value::as_str)
                    .is_some_and(|path| path.ends_with(suffix))
                && tool_use
                    .input
                    .get("new_string")
                    .and_then(Value::as_str)
                    .is_some_and(|new_string| new_string.contains(expected_value))
                && self.result_for(&tool_use.id).is_some_and(|result| !result.is_error)
        })
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

    /// Finds the Bash invocation whose correlated result proves verification
    /// succeeded. The client may first inspect (`cat ./verify.sh`) or chmod the
    /// script; those merely *mention* the path and their output can echo the
    /// script's own `verify: OK` string, so selecting on the substring would
    /// mistake an inspection for the required run and — if it precedes the Edit —
    /// false-reject a valid trace on the ordering check. Only an actual execution
    /// counts.
    fn successful_verify_bash(&self) -> Option<(&ToolUse, &ToolResult)> {
        self.tool_uses.iter().find_map(|tool_use| {
            let command = tool_use.input.get("command").and_then(Value::as_str)?;
            if tool_use.name != "Bash" || !is_verifier_execution(command) {
                return None;
            }
            let result = self.result_for(&tool_use.id)?;
            (!result.is_error && result.text.contains("verify: OK")).then_some((tool_use, result))
        })
    }

    /// The observed Bash command strings, for assertion failure messages.
    fn bash_commands(&self) -> Vec<&str> {
        self.tool_uses
            .iter()
            .filter(|tool_use| tool_use.name == "Bash")
            .filter_map(|tool_use| tool_use.input.get("command").and_then(Value::as_str))
            .collect()
    }

    /// The observed tool-use names, for assertion failure messages.
    fn tool_use_names(&self) -> Vec<&str> {
        self.tool_uses.iter().map(|tool_use| tool_use.name.as_str()).collect()
    }

    /// The first client-initiated (`trigger == "auto"`) compaction boundary, if any.
    ///
    /// A `manual` boundary would be a user-invoked `/compact`, not the client's
    /// own threshold-driven compaction, so it does not satisfy the acceptance.
    fn auto_compaction(&self) -> Option<&CompactBoundary> {
        self.compact_boundaries
            .iter()
            .find(|boundary| boundary.trigger == "auto")
    }

    /// Every compaction trigger observed, in stream order, for failure messages.
    fn compaction_triggers(&self) -> Vec<&str> {
        self.compact_boundaries
            .iter()
            .map(|boundary| boundary.trigger.as_str())
            .collect()
    }

    /// The first `tool_use` emitted strictly after the given stream position.
    fn first_tool_use_after(&self, seq: usize) -> Option<&ToolUse> {
        self.tool_uses.iter().find(|tool_use| tool_use.seq > seq)
    }

    /// The first tool call after `seq` that reaches for the persistent `source/`
    /// marker — a file tool whose `file_path` or a `Bash` command references the
    /// source directory or the marker file. The compaction task re-reads the source
    /// after the boundary to recover the exact token for the write, so this locates
    /// that required post-compaction recovery read.
    fn source_reference_after(&self, seq: usize) -> Option<&ToolUse> {
        self.tool_uses.iter().find(|tool_use| {
            if tool_use.seq <= seq {
                return false;
            }
            let field = match tool_use.name.as_str() {
                "Read" | "Edit" | "Write" => tool_use.input.get("file_path").and_then(Value::as_str),
                "Bash" => tool_use.input.get("command").and_then(Value::as_str),
                _ => None,
            };
            field.is_some_and(references_source_marker)
        })
    }
}

/// Whether a tool argument reaches for the `source/` marker: the `source`
/// directory itself (catching `source/value.txt`, `cd source && cat value.txt`,
/// `./source/...`) or the marker file `value.txt` by any path other than the
/// task's own `result/value.txt`. Broader than a literal `source/` match so `cd
/// source` and bare-`value.txt` recovery reads are all recognized.
fn references_source_marker(argument: &str) -> bool {
    argument.contains("source") || (argument.contains("value.txt") && !argument.contains("result"))
}

/// Whether a Bash command is an actual EXECUTION of the verifier script (not an
/// inspection such as `cat ./verify.sh`, nor a `chmod`). Matched by exact token
/// shape: `./verify.sh`, or `sh`/`bash` followed by the script path.
fn is_verifier_execution(command: &str) -> bool {
    match command.split_whitespace().collect::<Vec<_>>().as_slice() {
        ["./verify.sh"] => true,
        ["sh" | "bash", script] => *script == "./verify.sh" || *script == "verify.sh",
        _ => false,
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

/// Returns a copy of a forwarded Tavily request with the secret `api_key`
/// masked. Failure messages below are printed to stdout and, in CI, teed into a
/// log that is uploaded as a build artifact — GitHub scrubs secrets from the
/// live log stream but NOT from uploaded artifacts, so the real key must never
/// reach a panic or `Err` string. The small clone is on a cold failure path and
/// is necessary: the borrowed capture must not be mutated.
fn redact_api_key(request: &Value) -> Value {
    let mut redacted = request.clone();
    if let Some(api_key) = redacted.get_mut("api_key") {
        *api_key = Value::String("<redacted>".to_owned());
    }
    redacted
}

/// Asserts a web-search turn was resolved server-side against real Tavily.
///
/// Returns `Err(reason)` naming the first failure so the caller can attach the
/// full stdout. Two independent kinds of evidence must agree:
///
///   * CLIENT TRACE — the turn converged on a text answer (`result` subtype `success`, not an error) with a non-empty
///     final summary, and no `WebSearch` `tool_use` leaked to the client (the managed loop resolves and suppresses it;
///     a leak means Praxis did not take over the tool); and
///   * GROUND TRUTH — the capture-relay observed at least one real Tavily exchange: HTTP 200, the resolved API key
///     carried in the request body, a non-empty reconstructed query, and a non-empty `results` array whose every source
///     has an absolute URL. This is what a client-only assertion cannot prove, because the managed loop falls back to
///     an `is_error` tool result on provider failure and the client would still "get an answer".
fn check_server_side_web_search(
    stdout: &str,
    captures: &[TavilySearchCapture],
    tavily_key: &str,
) -> Result<(), String> {
    let trace = ToolTrace::parse(stdout);

    if trace.result_subtype.as_deref() != Some("success") {
        return Err(format!(
            "the web-search turn must converge on an answer; the client reported subtype {:?} after {:?} turns",
            trace.result_subtype, trace.num_turns,
        ));
    }
    if trace.result_is_error {
        return Err("the web-search turn must not end in a client-reported error".to_owned());
    }

    // The managed loop owns WebSearch and suppresses its tool_use; a leaked
    // WebSearch call means the client ran the search, not Praxis.
    let leaked = trace
        .tool_use_names()
        .into_iter()
        .filter(|name| *name == MANAGED_WEB_SEARCH_TOOL)
        .count();
    if leaked != 0 {
        return Err(format!(
            "the managed loop must resolve WebSearch server-side and suppress its tool_use, but {leaked} \
             WebSearch tool_use block(s) reached the client; observed tool calls: {:?}",
            trace.tool_use_names(),
        ));
    }

    match trace.final_summary.as_deref() {
        Some(summary) if summary.trim().len() >= 20 => {},
        other => {
            return Err(format!(
                "the web-search turn must emit a non-empty final answer; got {other:?}"
            ));
        },
    }

    // Ground truth: a real Tavily search actually ran through the relay.
    if captures.is_empty() {
        return Err(
            "no Tavily exchange was observed; the model never called WebSearch so the managed loop \
             dispatched no server-side search"
                .to_owned(),
        );
    }
    // Validate EVERY exchange, not just the first: the managed loop can dispatch
    // more than one search per turn (a retry or a second model search), and the
    // answer could rest on an `is_error` fallback after a later exchange failed.
    // Checking only the first would let that broken integration pass.
    for exchange in captures {
        if exchange.status != 200 {
            return Err(format!(
                "real Tavily must return HTTP 200; the relay observed status {} with body {}",
                exchange.status, exchange.response,
            ));
        }
        if exchange.request.get("api_key").and_then(Value::as_str) != Some(tavily_key) {
            return Err(format!(
                "the resolved Tavily key must travel in the forwarded request body; request was {}",
                redact_api_key(&exchange.request),
            ));
        }
        let query = exchange
            .request
            .get("query")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if query.trim().is_empty() {
            return Err(format!(
                "the reconstructed search query must be populated; request was {}",
                redact_api_key(&exchange.request),
            ));
        }
        let results = exchange.response.get("results").and_then(Value::as_array);
        match results {
            Some(results) if !results.is_empty() => {
                let all_absolute = results.iter().all(|result| {
                    result
                        .get("url")
                        .and_then(Value::as_str)
                        .is_some_and(|url| url.starts_with("http://") || url.starts_with("https://"))
                });
                if !all_absolute {
                    return Err(format!(
                        "every real Tavily source must carry an absolute URL; results were {results:?}"
                    ));
                }
            },
            _ => {
                return Err(format!(
                    "real Tavily must return a non-empty results array; response was {}",
                    exchange.response,
                ));
            },
        }
    }

    Ok(())
}

#[test]
fn tool_trace_selects_successful_verify_call_after_setup_call() {
    let stdout = r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"setup","name":"Bash","input":{"command":"chmod +x ./verify.sh"}}]}}
{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"setup","content":"Bash completed with no output"}]}}
{"type":"assistant","message":{"content":[{"type":"tool_use","id":"verify","name":"Bash","input":{"command":"./verify.sh"}}]}}
{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"verify","content":"verify: OK"}]}}
{"type":"result","result":"Task complete"}"#;

    let trace = ToolTrace::parse(stdout);
    let (tool_use, result) = trace
        .successful_verify_bash()
        .expect("the successful verify call should be selected");

    assert_eq!(tool_use.id, "verify");
    assert_eq!(result.text, "verify: OK");
    assert_eq!(trace.bash_commands(), ["chmod +x ./verify.sh", "./verify.sh"]);
}

#[test]
fn tool_trace_reports_reasoning_left_in_user_visible_text() {
    // The leaked shape: a mismatched --reasoning-parser leaves the thinking
    // block in the assistant message, so it reaches the text block and the
    // summary the client renders from it.
    let stdout = r#"{"type":"assistant","message":{"content":[{"type":"text","text":"<think>which tool?</think>Here is the plan: step 1."}]}}
{"type":"result","subtype":"success","is_error":false,"num_turns":1,"result":"<think>which tool?</think>Here is the plan: step 1."}"#;

    let trace = ToolTrace::parse(stdout);
    let markers = trace
        .reasoning_leaks()
        .into_iter()
        .map(|(marker, _)| marker)
        .collect::<Vec<_>>();

    // Both delimiters, in both the streamed text block and the summary.
    assert_eq!(markers, ["<think>", "</think>", "<think>", "</think>"]);
}

#[test]
fn tool_trace_accepts_a_clean_plan_and_records_convergence() {
    let stdout = r#"{"type":"assistant","message":{"content":[{"type":"text","text":"Here is the plan: step 1."}]}}
{"type":"result","subtype":"success","is_error":false,"num_turns":3,"result":"Here is the plan: step 1."}"#;

    let trace = ToolTrace::parse(stdout);

    assert!(trace.reasoning_leaks().is_empty());
    assert_eq!(trace.result_subtype.as_deref(), Some("success"));
    assert!(!trace.result_is_error);
    assert_eq!(trace.num_turns, Some(3));
    assert_eq!(trace.assistant_texts, ["Here is the plan: step 1."]);
}

#[test]
fn tool_trace_reports_turn_budget_exhaustion_as_the_headless_loop_symptom() {
    // The terminal envelope a looping run produces headlessly: no answer, and
    // the turn budget spent. Captured from the pinned executable against a
    // backend that never stopped requesting tools.
    let stdout = r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t1","name":"Read","input":{"file_path":"deploy.sh"}}]}}
{"type":"result","subtype":"error_max_turns","is_error":true,"num_turns":4,"result":null}"#;

    let trace = ToolTrace::parse(stdout);

    assert_eq!(trace.result_subtype.as_deref(), Some("error_max_turns"));
    assert!(trace.result_is_error);
    assert_eq!(trace.num_turns, Some(4));
    assert!(
        trace.final_summary.is_none(),
        "an exhausted run carries no answer to mistake for one"
    );
}

/// A suppressed-and-resolved web-search transcript: the client converged on a
/// text answer and NO `WebSearch` tool_use is present (the managed loop owns it).
const SUPPRESSED_WEB_SEARCH_STDOUT: &str = r#"{"type":"assistant","message":{"content":[{"type":"text","text":"Argentina won the most recent FIFA World Cup, in 2022."}]}}
{"type":"result","subtype":"success","is_error":false,"num_turns":2,"result":"Argentina won the most recent FIFA World Cup, in 2022."}"#;

/// Builds one observed Tavily exchange for the checker's ground-truth assertions.
fn tavily_capture(api_key: &str, query: &str, result_url: &str) -> TavilySearchCapture {
    TavilySearchCapture {
        request: serde_json::json!({ "api_key": api_key, "query": query }),
        status: 200,
        response: serde_json::json!({ "results": [{ "url": result_url }] }),
    }
}

#[test]
fn web_search_check_accepts_suppressed_resolution_with_real_exchange() {
    let captures = [tavily_capture(
        "tvly-key",
        "most recent FIFA World Cup winner",
        "https://fifa.com/worldcup",
    )];
    assert_eq!(
        check_server_side_web_search(SUPPRESSED_WEB_SEARCH_STDOUT, &captures, "tvly-key"),
        Ok(())
    );
}

#[test]
fn web_search_check_rejects_leaked_client_side_tool_use() {
    // The failure the relay exists to catch would still "get an answer", but a
    // WebSearch tool_use reaching the client means Praxis did NOT take over.
    let stdout = r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"ws","name":"WebSearch","input":{"query":"fifa world cup"}}]}}
{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"ws","content":"Argentina, 2022"}]}}
{"type":"assistant","message":{"content":[{"type":"text","text":"Argentina won the most recent World Cup, in 2022."}]}}
{"type":"result","subtype":"success","is_error":false,"num_turns":2,"result":"Argentina won the most recent World Cup, in 2022."}"#;
    let captures = [tavily_capture("tvly-key", "fifa world cup", "https://fifa.com")];

    let error = check_server_side_web_search(stdout, &captures, "tvly-key").expect_err("a leaked WebSearch must fail");
    assert!(error.contains("suppress its tool_use"), "unexpected reason: {error}");
}

#[test]
fn web_search_check_rejects_missing_tavily_exchange() {
    // The model converged without ever searching: no exchange was observed.
    let error = check_server_side_web_search(SUPPRESSED_WEB_SEARCH_STDOUT, &[], "tvly-key")
        .expect_err("a converged turn with no search must fail");
    assert!(error.contains("no Tavily exchange"), "unexpected reason: {error}");
}

#[test]
fn web_search_check_rejects_mismatched_provider_key() {
    let captures = [tavily_capture("wrong-key", "fifa world cup", "https://fifa.com")];
    let error = check_server_side_web_search(SUPPRESSED_WEB_SEARCH_STDOUT, &captures, "tvly-key")
        .expect_err("a mismatched provider key must fail");
    assert!(
        error.contains("must travel in the forwarded request body"),
        "unexpected reason: {error}"
    );
}

#[test]
fn web_search_check_rejects_relative_source_url() {
    let captures = [tavily_capture("tvly-key", "fifa world cup", "/relative/path")];
    let error = check_server_side_web_search(SUPPRESSED_WEB_SEARCH_STDOUT, &captures, "tvly-key")
        .expect_err("a relative source URL must fail");
    assert!(error.contains("absolute URL"), "unexpected reason: {error}");
}

#[test]
fn web_search_check_rejects_nonconvergent_subtype() {
    // The turn burned its budget without answering: a non-`success` subtype must
    // fail even though the relay did observe a real exchange.
    let stdout = r#"{"type":"result","subtype":"error_max_turns","is_error":true,"num_turns":8,"result":null}"#;
    let captures = [tavily_capture("tvly-key", "fifa world cup", "https://fifa.com")];
    let error =
        check_server_side_web_search(stdout, &captures, "tvly-key").expect_err("a non-success subtype must fail");
    assert!(
        error.contains("must converge on an answer"),
        "unexpected reason: {error}"
    );
}

#[test]
fn web_search_check_rejects_client_reported_error() {
    // A `success` subtype that still carries `is_error` must fail: the client
    // flagged the turn as errored.
    let stdout = r#"{"type":"assistant","message":{"content":[{"type":"text","text":"Argentina won the most recent FIFA World Cup, in 2022."}]}}
{"type":"result","subtype":"success","is_error":true,"num_turns":2,"result":"Argentina won the most recent FIFA World Cup, in 2022."}"#;
    let captures = [tavily_capture("tvly-key", "fifa world cup", "https://fifa.com")];
    let error =
        check_server_side_web_search(stdout, &captures, "tvly-key").expect_err("a client-reported error must fail");
    assert!(
        error.contains("must not end in a client-reported error"),
        "unexpected reason: {error}"
    );
}

#[test]
fn web_search_check_rejects_short_final_summary() {
    // A converged, suppressed turn whose answer is too short to be a real one
    // must fail the non-empty-answer threshold.
    let stdout = r#"{"type":"assistant","message":{"content":[{"type":"text","text":"Argentina"}]}}
{"type":"result","subtype":"success","is_error":false,"num_turns":2,"result":"Argentina"}"#;
    let captures = [tavily_capture("tvly-key", "fifa world cup", "https://fifa.com")];
    let error =
        check_server_side_web_search(stdout, &captures, "tvly-key").expect_err("a too-short final answer must fail");
    assert!(
        error.contains("must emit a non-empty final answer"),
        "unexpected reason: {error}"
    );
}

#[test]
fn web_search_check_rejects_non_200_tavily_status() {
    // The managed loop converged (an `is_error` tool result still "gets an
    // answer"), but the relay saw real Tavily reject the call: ground truth fails.
    let captures = [TavilySearchCapture {
        request: serde_json::json!({ "api_key": "tvly-key", "query": "fifa world cup" }),
        status: 401,
        response: serde_json::json!({ "error": "unauthorized" }),
    }];
    let error = check_server_side_web_search(SUPPRESSED_WEB_SEARCH_STDOUT, &captures, "tvly-key")
        .expect_err("a non-200 Tavily status must fail");
    assert!(error.contains("must return HTTP 200"), "unexpected reason: {error}");
}

#[test]
fn web_search_check_rejects_empty_search_query() {
    // A 200 with the right key but no reconstructed query means no real search
    // was dispatched.
    let captures = [TavilySearchCapture {
        request: serde_json::json!({ "api_key": "tvly-key", "query": "   " }),
        status: 200,
        response: serde_json::json!({ "results": [{ "url": "https://fifa.com" }] }),
    }];
    let error = check_server_side_web_search(SUPPRESSED_WEB_SEARCH_STDOUT, &captures, "tvly-key")
        .expect_err("an empty search query must fail");
    assert!(error.contains("query must be populated"), "unexpected reason: {error}");
}

#[test]
fn web_search_check_rejects_empty_results_array() {
    // A 200 with the right key and a query but no results means Tavily returned
    // nothing to ground the answer on.
    let captures = [TavilySearchCapture {
        request: serde_json::json!({ "api_key": "tvly-key", "query": "fifa world cup" }),
        status: 200,
        response: serde_json::json!({ "results": [] }),
    }];
    let error = check_server_side_web_search(SUPPRESSED_WEB_SEARCH_STDOUT, &captures, "tvly-key")
        .expect_err("an empty results array must fail");
    assert!(error.contains("non-empty results array"), "unexpected reason: {error}");
}

#[test]
fn web_search_check_rejects_a_later_failed_exchange_behind_a_valid_first() {
    // The managed loop dispatched two searches: the first succeeded, but the
    // second came back non-200. The turn still converged (an `is_error` tool
    // result "gets an answer"), so checking only the first capture would wrongly
    // pass. Every exchange must hold.
    let captures = [
        tavily_capture("tvly-key", "fifa world cup", "https://fifa.com"),
        TavilySearchCapture {
            request: serde_json::json!({ "api_key": "tvly-key", "query": "fifa world cup final score" }),
            status: 502,
            response: serde_json::json!({ "error": "bad gateway" }),
        },
    ];
    let error = check_server_side_web_search(SUPPRESSED_WEB_SEARCH_STDOUT, &captures, "tvly-key")
        .expect_err("a later failed exchange must fail even behind a valid first");
    assert!(error.contains("must return HTTP 200"), "unexpected reason: {error}");
}

#[test]
fn web_search_config_patches_the_example_into_a_valid_config() {
    // Validates offline that every anchor still exists and that the inlined key
    // plus loopback `base_url` produce YAML that parses — the 16-space `base_url`
    // must align with the `api_key` key it follows, or the scanner rejects it.
    let live = LiveConfig {
        claude_bin: OsString::from("/usr/bin/claude"),
        vllm_authority: "127.0.0.1:9000".to_owned(),
        model: "qwen3-8b".to_owned(),
        listen_address: IpAddr::V4(Ipv4Addr::LOCALHOST),
        netns: None,
    };

    // Building the config must not panic: replace_once asserts each anchor and
    // Config::from_yaml asserts the patched YAML parses.
    let _config = web_search_config(&live, 18080, 19090, "tvly-test-key");
}

#[test]
fn tool_trace_records_auto_compaction_boundary_and_orders_tool_calls_around_it() {
    // The marker is read before the boundary and the edit follows it: the exact
    // ordering the compaction acceptance asserts.
    let stdout = r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"r1","name":"Read","input":{"file_path":"source/value.txt"}}]}}
{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"r1","content":"praxis-native-0000000001"}]}}
{"type":"system","subtype":"compact_boundary","compact_metadata":{"trigger":"auto","pre_tokens":15000}}
{"type":"assistant","message":{"content":[{"type":"tool_use","id":"e1","name":"Edit","input":{"file_path":"result/value.txt","new_string":"PRAXIS-NATIVE-0000000001"}}]}}
{"type":"result","subtype":"success","is_error":false,"num_turns":5,"result":"done"}"#;

    let trace = ToolTrace::parse(stdout);
    let compaction = trace
        .auto_compaction()
        .expect("an auto compact_boundary should be parsed");
    assert_eq!(compaction.pre_tokens, Some(15000));

    let read = trace
        .find_tool_use("Read", "source/value.txt")
        .expect("the source read should be parsed");
    assert!(read.seq < compaction.seq, "the marker read precedes compaction");

    let post = trace
        .first_tool_use_after(compaction.seq)
        .expect("a tool use should follow compaction");
    assert_eq!(post.name, "Edit");
    assert!(post.seq > compaction.seq);
}

#[test]
fn tool_trace_locates_source_reread_after_compaction() {
    // The compaction task re-reads the source (via Read or Bash) after the boundary
    // to recover the exact marker for the write; the helper locates that required
    // post-compaction recovery read.
    let stdout = r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"r1","name":"Read","input":{"file_path":"source/value.txt"}}]}}
{"type":"system","subtype":"compact_boundary","compact_metadata":{"trigger":"auto","pre_tokens":15000}}
{"type":"assistant","message":{"content":[{"type":"tool_use","id":"b1","name":"Bash","input":{"command":"cat source/value.txt"}}]}}
{"type":"result","subtype":"success","is_error":false,"num_turns":5,"result":"done"}"#;

    let trace = ToolTrace::parse(stdout);
    let compaction = trace.auto_compaction().expect("auto compaction should be parsed");
    let reread = trace
        .source_reference_after(compaction.seq)
        .expect("a post-compaction Bash read of source/ should be located");
    assert_eq!(reread.name, "Bash");
    assert!(reread.seq > compaction.seq);
}

#[test]
fn tool_trace_does_not_mistake_result_edit_for_a_source_reread() {
    // A post-compaction result/ edit is not a source re-read, so the helper must
    // return None when the only post-boundary tool call touches result/value.txt.
    let stdout = r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"r1","name":"Read","input":{"file_path":"source/value.txt"}}]}}
{"type":"system","subtype":"compact_boundary","compact_metadata":{"trigger":"auto","pre_tokens":15000}}
{"type":"assistant","message":{"content":[{"type":"tool_use","id":"e1","name":"Edit","input":{"file_path":"result/value.txt","new_string":"PRAXIS-NATIVE-0000000001"}}]}}
{"type":"result","subtype":"success","is_error":false,"num_turns":5,"result":"done"}"#;

    let trace = ToolTrace::parse(stdout);
    let compaction = trace.auto_compaction().expect("auto compaction should be parsed");
    assert!(
        trace.source_reference_after(compaction.seq).is_none(),
        "a result/ edit after compaction is not a source reread"
    );
}

#[test]
fn tool_trace_locates_source_reread_spellings_after_compaction() {
    // `cd source && cat value.txt` and a bare `cat value.txt` both recover the
    // marker without the literal `source/` path; the broadened check recognizes
    // them as source re-reads while leaving the `result/value.txt` edit alone.
    for command in ["cd source && cat value.txt", "cat value.txt"] {
        let stdout = format!(
            r#"{{"type":"assistant","message":{{"content":[{{"type":"tool_use","id":"r1","name":"Read","input":{{"file_path":"source/value.txt"}}}}]}}}}
{{"type":"system","subtype":"compact_boundary","compact_metadata":{{"trigger":"auto","pre_tokens":15000}}}}
{{"type":"assistant","message":{{"content":[{{"type":"tool_use","id":"b1","name":"Bash","input":{{"command":"{command}"}}}}]}}}}
{{"type":"result","subtype":"success","is_error":false,"num_turns":5,"result":"done"}}"#
        );
        let trace = ToolTrace::parse(&stdout);
        let compaction = trace.auto_compaction().expect("auto compaction should be parsed");
        assert!(
            trace.source_reference_after(compaction.seq).is_some(),
            "post-compaction `{command}` must be recognized as a source re-read"
        );
    }
}

#[test]
fn tool_trace_ignores_manual_compaction_boundary() {
    // A user-invoked `/compact` is not the client's own threshold-driven
    // auto-compaction and must not satisfy the acceptance signal.
    let stdout = r#"{"type":"system","subtype":"compact_boundary","compact_metadata":{"trigger":"manual","pre_tokens":100}}
{"type":"result","subtype":"success","is_error":false,"num_turns":1,"result":"done"}"#;

    let trace = ToolTrace::parse(stdout);
    assert!(
        trace.auto_compaction().is_none(),
        "a manual compaction is not the client's own auto-compaction"
    );
    assert_eq!(trace.compaction_triggers(), ["manual"]);
}

#[test]
fn compaction_launch_disables_session_persistence() {
    // The compaction run keeps itself self-contained by disabling the on-disk
    // session transcript, so repeated runs don't accumulate transcript state.
    assert!(
        COMPACTION_LAUNCH_FLAGS.contains(&"--no-session-persistence"),
        "the compaction run must disable the on-disk session transcript to stay reproducible across runs"
    );
}

#[test]
fn verifier_execution_excludes_inspection_and_setup() {
    // Only an actual run of the script counts as execution.
    assert!(is_verifier_execution("./verify.sh"));
    assert!(is_verifier_execution("sh ./verify.sh"));
    assert!(is_verifier_execution("bash verify.sh"));
    // Inspection and setup are not executions, even though they name the script and
    // an inspection's stdout can echo the script's own `verify: OK` literal.
    assert!(!is_verifier_execution("cat ./verify.sh"));
    assert!(!is_verifier_execution("chmod +x ./verify.sh"));
    assert!(!is_verifier_execution("verify.sh"));
}

#[test]
fn successful_verify_bash_ignores_pre_edit_inspection_of_the_script() {
    // The script prints `verify: OK` on success, so its source contains that
    // literal. A `cat ./verify.sh` inspection before the real run must NOT be
    // selected as the verification, or an ordering check keyed on its position
    // would reject a valid run that executes the verifier afterward.
    let stdout = r##"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"peek","name":"Bash","input":{"command":"cat ./verify.sh"}}]}}
{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"peek","content":"#!/bin/sh\necho 'verify: OK'"}]}}
{"type":"assistant","message":{"content":[{"type":"tool_use","id":"run","name":"Bash","input":{"command":"./verify.sh"}}]}}
{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"run","content":"verify: OK"}]}}
{"type":"result","result":"Task complete"}"##;

    let trace = ToolTrace::parse(stdout);
    let (tool_use, result) = trace
        .successful_verify_bash()
        .expect("the executed verify call should be selected, not the inspection");

    assert_eq!(tool_use.id, "run");
    assert_eq!(result.text, "verify: OK");
}
