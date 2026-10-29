// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Typed views of the pinned-client manifests.
//!
//! Each struct models only the fields `check-pins` and `bump-pins` read.
//! Everything else in a manifest (launch flags, tool lists, isolation
//! settings) is behavioral and owned by the acceptance tests, so serde
//! ignores it.

use serde::Deserialize;

// -----------------------------------------------------------------------------
// Manifest Paths
// -----------------------------------------------------------------------------

/// Claude Code manifest, relative to the repository root.
pub(crate) const CLAUDE: &str = "tests/integration/fixtures/claude-code-cli/pin.toml";

/// `OpenCode` 1.x manifest, relative to the repository root.
pub(crate) const OPENCODE: &str = "tests/integration/fixtures/opencode-cli/pin.toml";

/// `OpenCode` 2.x manifest, relative to the repository root.
pub(crate) const OPENCODE_V2: &str = "tests/integration/fixtures/opencode-cli-v2/pin.toml";

/// pi manifest, relative to the repository root.
pub(crate) const PI: &str = "tests/integration/fixtures/pi-cli/pin.toml";

/// Codex manifest, relative to the repository root.
pub(crate) const CODEX: &str = "tests/integration/fixtures/codex-cli/pin.toml";

// -----------------------------------------------------------------------------
// Shared Tables
// -----------------------------------------------------------------------------

/// A `[vllm]` backend pin block.
#[derive(Debug, Deserialize)]
pub(crate) struct Vllm {
    /// Digest-pinned container image reference.
    pub(crate) image: String,

    /// Model name the server exposes.
    pub(crate) served_model_name: String,

    /// Tool-call parser passed to the server.
    pub(crate) tool_call_parser: String,

    /// Server context window, which must match `--max-model-len`.
    pub(crate) context_length: u64,

    /// Server launch settings.
    pub(crate) launch: VllmLaunch,
}

/// `[vllm.launch]`.
#[derive(Debug, Deserialize)]
pub(crate) struct VllmLaunch {
    /// Command-line flags passed to `vllm serve`.
    pub(crate) flags: Vec<String>,
}

impl Vllm {
    /// The value following `--max-model-len` in the launch flags.
    pub(crate) fn flag_context(&self) -> Result<u64, String> {
        let flags = &self.launch.flags;
        let index = flags
            .iter()
            .position(|flag| flag == "--max-model-len")
            .ok_or("vllm.launch.flags must pass --max-model-len")?;
        flags
            .get(index + 1)
            .and_then(|value| value.parse().ok())
            .ok_or_else(|| "--max-model-len in vllm.launch.flags must be followed by an integer".to_owned())
    }
}

/// `[gateway]`: the example config a client lane runs.
#[derive(Debug, Deserialize)]
pub(crate) struct Gateway {
    /// Example config under `examples/configs/`.
    pub(crate) example_config: String,

    /// Manifest holding the backend pins, for clients without `[vllm]`.
    #[serde(default)]
    pub(crate) backend_pins: Option<String>,
}

/// `[startup_matrix]`: request paths a client may reach.
#[derive(Debug, Deserialize)]
pub(crate) struct StartupMatrix {
    /// Path prefixes that must each have an explicit route.
    pub(crate) allowed_prefixes: Vec<String>,
}

// -----------------------------------------------------------------------------
// Claude Code
// -----------------------------------------------------------------------------

/// `claude-code-cli/pin.toml`.
#[derive(Debug, Deserialize)]
pub(crate) struct ClaudeManifest {
    /// `[claude_code]`.
    pub(crate) claude_code: Claude,

    /// `[vllm]`.
    pub(crate) vllm: Vllm,
}

/// `[claude_code]`.
#[derive(Debug, Deserialize)]
pub(crate) struct Claude {
    /// Pinned version.
    pub(crate) version: String,

    /// Platform triple of the native build.
    pub(crate) platform: String,

    /// Native build download URL.
    pub(crate) url: String,

    /// SHA-256 of the native build.
    pub(crate) sha256: String,

    /// Major-series tripwire.
    pub(crate) major_series: u64,

    /// `[claude_code.launch]`.
    pub(crate) launch: ClaudeLaunch,
}

/// `[claude_code.launch]`.
#[derive(Debug, Deserialize)]
pub(crate) struct ClaudeLaunch {
    /// Smallest backend window auto mode accepts.
    pub(crate) auto_minimum_backend_context_tokens: u64,

    /// Client context window.
    pub(crate) max_context_tokens: u64,

    /// `[claude_code.launch.compaction]`.
    pub(crate) compaction: Compaction,
}

/// `[claude_code.launch.compaction]`.
#[derive(Debug, Deserialize)]
pub(crate) struct Compaction {
    /// Lowered client window that forces auto-compaction.
    pub(crate) max_context_tokens: u64,

    /// Number of ballast chapters.
    pub(crate) ballast_chapters: u64,

    /// Size of each ballast chapter in bytes.
    pub(crate) ballast_bytes: u64,
}

// -----------------------------------------------------------------------------
// OpenCode
// -----------------------------------------------------------------------------

/// `opencode-cli/pin.toml` and `opencode-cli-v2/pin.toml`.
#[derive(Debug, Deserialize)]
pub(crate) struct OpenCodeManifest {
    /// `[opencode]`.
    pub(crate) opencode: OpenCode,

    /// `[vllm]`; only the 1.x manifest declares it.
    #[serde(default)]
    pub(crate) vllm: Option<Vllm>,

    /// `[gateway]`.
    pub(crate) gateway: Gateway,

    /// `[startup_matrix]`.
    pub(crate) startup_matrix: StartupMatrix,
}

/// `[opencode]`.
#[derive(Debug, Deserialize)]
pub(crate) struct OpenCode {
    /// Pinned version.
    pub(crate) version: String,

    /// Platform slug.
    pub(crate) platform: String,

    /// How CI obtains the executable.
    pub(crate) install: String,

    /// npm package name.
    #[serde(default)]
    pub(crate) npm_package: String,

    /// Release URL, for `install = "github-release"`.
    #[serde(default)]
    pub(crate) url: String,

    /// Release SHA-256, for `install = "github-release"`.
    #[serde(default)]
    pub(crate) sha256: String,

    /// Major-series tripwire.
    pub(crate) major_series: u64,

    /// Exact `--version` output, when it differs from the bare version.
    #[serde(default)]
    pub(crate) version_output: Option<String>,

    /// `[opencode.launch]`.
    pub(crate) launch: OpenCodeLaunch,

    /// `[opencode.provider]`.
    pub(crate) provider: OpenCodeProvider,
}

/// `[opencode.launch]`.
#[derive(Debug, Deserialize)]
pub(crate) struct OpenCodeLaunch {
    /// Whether the prompt is the final positional argument.
    pub(crate) prompt_is_positional: bool,
}

/// `[opencode.provider]`.
#[derive(Debug, Deserialize)]
pub(crate) struct OpenCodeProvider {
    /// Model the client addresses.
    pub(crate) model_id: String,

    /// Context window the client advertises.
    pub(crate) context: u64,

    /// Output budget the client advertises.
    pub(crate) output: u64,
}

// -----------------------------------------------------------------------------
// pi
// -----------------------------------------------------------------------------

/// `pi-cli/pin.toml`.
#[derive(Debug, Deserialize)]
pub(crate) struct PiManifest {
    /// `[pi]`.
    pub(crate) pi: Pi,

    /// `[vllm]`; must be absent.
    #[serde(default)]
    pub(crate) vllm: Option<toml::Value>,

    /// `[gateway]`.
    pub(crate) gateway: Gateway,

    /// `[startup_matrix]`.
    pub(crate) startup_matrix: StartupMatrix,
}

/// `[pi]`.
#[derive(Debug, Deserialize)]
pub(crate) struct Pi {
    /// Pinned version.
    pub(crate) version: String,

    /// Platform slug.
    pub(crate) platform: String,

    /// How CI obtains the executable.
    pub(crate) install: String,

    /// Release archive URL.
    pub(crate) url: String,

    /// Release archive file name.
    pub(crate) archive: String,

    /// Major-series tripwire.
    pub(crate) major_series: u64,

    /// `[pi.auth]`.
    pub(crate) auth: PiAuth,

    /// `[pi.provider]`.
    pub(crate) provider: PiProvider,
}

/// `[pi.auth]`.
#[derive(Debug, Deserialize)]
pub(crate) struct PiAuth {
    /// `models.json` fixture, relative to the manifest directory.
    pub(crate) fixture: String,
}

/// `[pi.provider]`.
#[derive(Debug, Deserialize)]
pub(crate) struct PiProvider {
    /// Provider key in `models.json`.
    pub(crate) id: String,

    /// Wire API.
    pub(crate) api: String,

    /// Model the client addresses.
    pub(crate) model_id: String,

    /// Context window the client advertises.
    pub(crate) context_window: u64,

    /// Output budget the client advertises.
    pub(crate) max_tokens: u64,
}

// -----------------------------------------------------------------------------
// Codex
// -----------------------------------------------------------------------------

/// `codex-cli/pin.toml`.
#[derive(Debug, Deserialize)]
pub(crate) struct CodexManifest {
    /// `[codex]`.
    pub(crate) codex: Codex,
}

/// `[codex]`.
#[derive(Debug, Deserialize)]
pub(crate) struct Codex {
    /// Pinned version.
    pub(crate) version: String,

    /// Target triple.
    pub(crate) platform: String,

    /// How CI obtains the executable.
    pub(crate) install: String,

    /// Release tag prefix in front of the version.
    pub(crate) release_tag_prefix: String,

    /// Release archive file name.
    pub(crate) archive: String,
}

// -----------------------------------------------------------------------------
// Loaded Set
// -----------------------------------------------------------------------------

/// Every manifest, parsed.
#[derive(Debug)]
pub(crate) struct Manifests {
    /// Claude Code.
    pub(crate) claude: ClaudeManifest,

    /// `OpenCode` 1.x.
    pub(crate) opencode: OpenCodeManifest,

    /// `OpenCode` 2.x.
    pub(crate) opencode_v2: OpenCodeManifest,

    /// pi.
    pub(crate) pi: PiManifest,

    /// Codex.
    pub(crate) codex: CodexManifest,
}

impl Manifests {
    /// Parse every manifest from `repo`.
    pub(crate) fn load(repo: &super::Repo) -> Result<Self, String> {
        Ok(Self {
            claude: parse(repo, CLAUDE)?,
            opencode: parse(repo, OPENCODE)?,
            opencode_v2: parse(repo, OPENCODE_V2)?,
            pi: parse(repo, PI)?,
            codex: parse(repo, CODEX)?,
        })
    }
}

/// Read and deserialize one manifest.
fn parse<T: serde::de::DeserializeOwned>(repo: &super::Repo, path: &str) -> Result<T, String> {
    toml::from_str(&repo.read(path)?).map_err(|e| format!("{path}: {e}"))
}
