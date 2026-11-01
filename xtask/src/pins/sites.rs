// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Where each pin is written down.
//!
//! One table drives both `check-pins` (every site must carry the manifest's
//! value) and `bump-pins` (every site is rewritten together), so the check and
//! the rewrite cannot disagree about what a bump touches.
//!
//! Only mechanical references belong here. Behavioral claims ("verified
//! against X", observed-behavior comments, replay recordings) stay out on
//! purpose: changing them is a judgment a person makes after re-validating.

use super::manifest::{self, Manifests};

// -----------------------------------------------------------------------------
// Repository Paths
// -----------------------------------------------------------------------------

/// Integration-suite harness holding the pinned-version constants.
const HARNESS: &str = "tests/integration/tests/suite/harness.rs";

/// CPU acceptance lanes.
const INTEGRATION: &str = ".github/workflows/integration.yaml";

/// GPU acceptance lanes and the `pins` job.
const VLLM_INTEGRATION: &str = ".github/workflows/vllm-integration.yaml";

/// On-demand GPU endpoint.
const VLLM_DEV_ENDPOINT: &str = ".github/workflows/vllm-dev-endpoint.yaml";

/// Baked-model GPU image.
const CONTAINERFILE: &str = "vllm/Containerfile";

// -----------------------------------------------------------------------------
// Pins
// -----------------------------------------------------------------------------

/// A pinned value that one or more files must agree on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum Pin {
    /// Claude Code version.
    Claude,
    /// `OpenCode` 1.x version.
    Opencode,
    /// `OpenCode` 2.x version.
    OpencodeV2,
    /// pi version.
    Pi,
    /// Codex version.
    Codex,
    /// vLLM GPU image reference (`tag@sha256:digest`).
    Vllm,
}

impl Pin {
    /// Every pin, in report order.
    pub(crate) const ALL: [Self; 6] = [
        Self::Claude,
        Self::Opencode,
        Self::OpencodeV2,
        Self::Pi,
        Self::Codex,
        Self::Vllm,
    ];

    /// Human-readable name.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Claude => "Claude Code",
            Self::Opencode => "OpenCode 1.x",
            Self::OpencodeV2 => "OpenCode 2.x",
            Self::Pi => "pi",
            Self::Codex => "Codex",
            Self::Vllm => "vLLM image",
        }
    }

    /// The manifest's current value, which is the source of truth.
    pub(crate) fn current(self, manifests: &Manifests) -> Result<&str, String> {
        Ok(match self {
            Self::Claude => &manifests.claude.claude_code.version,
            Self::Opencode => &manifests.opencode.opencode.version,
            Self::OpencodeV2 => &manifests.opencode_v2.opencode.version,
            Self::Pi => &manifests.pi.pi.version,
            Self::Codex => &manifests.codex.codex.version,
            Self::Vllm => {
                &manifests
                    .opencode
                    .vllm
                    .as_ref()
                    .ok_or_else(|| format!("{}: must declare [vllm]", manifest::OPENCODE))?
                    .image
            },
        })
    }

    /// Every file location that carries this pin.
    pub(crate) const fn sites(self) -> &'static [Site] {
        match self {
            Self::Claude => CLAUDE_SITES,
            Self::Opencode => OPENCODE_SITES,
            Self::OpencodeV2 => OPENCODE_V2_SITES,
            Self::Pi => PI_SITES,
            Self::Codex => CODEX_SITES,
            Self::Vllm => VLLM_SITES,
        }
    }
}

// -----------------------------------------------------------------------------
// Sites
// -----------------------------------------------------------------------------

/// Anchored text in one file that embeds a pin.
#[derive(Debug)]
pub(crate) struct Site {
    /// File, relative to the repository root.
    pub(crate) path: &'static str,

    /// The anchored text, with `{v}` standing for the pinned value.
    pub(crate) template: &'static str,

    /// How many times the rendered text must appear.
    pub(crate) count: usize,
}

impl Site {
    /// The anchored text for `value`.
    pub(crate) fn render(&self, value: &str) -> String {
        self.template.replace("{v}", value)
    }

    /// Fail unless `contents` carries the anchor for `value` exactly
    /// [`count`](Self::count) times.
    pub(crate) fn expect(&self, contents: &str, value: &str) -> Result<(), String> {
        let anchor = self.render(value);
        let found = contents.matches(&anchor).count();
        if found == self.count {
            Ok(())
        } else {
            Err(format!(
                "{}: expected `{anchor}` {} time(s), found {found}",
                self.path, self.count
            ))
        }
    }
}

/// Shorthand for a [`Site`] table entry.
const fn site(path: &'static str, template: &'static str, count: usize) -> Site {
    Site { path, template, count }
}

/// Claude Code version sites.
const CLAUDE_SITES: &[Site] = &[
    site(manifest::CLAUDE, "version = \"{v}\"", 1),
    site(manifest::CLAUDE, "claude-code-releases/{v}/linux-x64/claude\"", 1),
    site(manifest::CLAUDE, "claude-code-releases/{v}/manifest.json", 1),
    site(HARNESS, "CLAUDE_CODE_PINNED_VERSION: &str = \"{v}\";", 1),
    site(INTEGRATION, "CLAUDE_CODE_PINNED_VERSION: \"{v}\"", 1),
];

/// `OpenCode` 1.x version sites.
const OPENCODE_SITES: &[Site] = &[
    site(manifest::OPENCODE, "version = \"{v}\"", 1),
    site(HARNESS, "OPENCODE_PINNED_VERSION: &str = \"{v}\";", 1),
    site(INTEGRATION, "OPENCODE_VERSION: \"{v}\"", 1),
];

/// `OpenCode` 2.x version sites.
const OPENCODE_V2_SITES: &[Site] = &[
    site(manifest::OPENCODE_V2, "version = \"{v}\"", 1),
    site(manifest::OPENCODE_V2, "version_output = \"opencode v{v}\"", 1),
    site(HARNESS, "OPENCODE_V2_PINNED_VERSION: &str = \"{v}\";", 1),
    site(INTEGRATION, "OPENCODE_V2_VERSION: \"{v}\"", 1),
];

/// pi version sites.
const PI_SITES: &[Site] = &[
    site(manifest::PI, "version = \"{v}\"", 1),
    site(manifest::PI, "releases/download/v{v}/", 1),
    site(HARNESS, "PI_PINNED_VERSION: &str = \"{v}\";", 1),
    site(INTEGRATION, "PI_VERSION: \"{v}\"", 1),
];

/// Codex version sites.
const CODEX_SITES: &[Site] = &[
    site(manifest::CODEX, "version = \"{v}\"", 1),
    site(manifest::CODEX, "version_output = \"codex-cli {v}\"", 1),
    site(HARNESS, "CODEX_PINNED_VERSION: &str = \"{v}\";", 1),
    site(INTEGRATION, "CODEX_VERSION: \"{v}\"", 2),
    site(VLLM_INTEGRATION, "CODEX_VERSION: \"{v}\"", 1),
];

/// vLLM GPU image sites.
const VLLM_SITES: &[Site] = &[
    site(manifest::OPENCODE, "image = \"{v}\"", 1),
    site(manifest::CLAUDE, "image = \"{v}\"", 1),
    site(VLLM_INTEGRATION, "VLLM_GPU_IMAGE: \"{v}\"", 1),
    site(VLLM_DEV_ENDPOINT, "VLLM_GPU_IMAGE: {v}\n", 1),
    site(CONTAINERFILE, "FROM {v}\n", 1),
];

// -----------------------------------------------------------------------------
// Checksum Fixtures
// -----------------------------------------------------------------------------

/// A `sha256sum`-format checksum fixture for a downloaded client artifact.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Fixture {
    /// Which pin the fixture belongs to.
    pub(crate) pin: Pin,

    /// Fixture directory, relative to the repository root.
    pub(crate) dir: &'static str,

    /// Fixture file name: `<version>-<platform>.sha256`.
    pub(crate) file: String,

    /// The artifact file name the fixture's second column must name.
    pub(crate) artifact: String,
}

impl Fixture {
    /// Fixture path, relative to the repository root.
    pub(crate) fn path(&self) -> String {
        format!("{}/{}", self.dir, self.file)
    }
}

/// The checksum fixture each client's current pin requires.
///
/// Claude Code has none: its checksum is inline in the manifest, copied from
/// the upstream release manifest.
pub(crate) fn fixtures(manifests: &Manifests) -> Vec<Fixture> {
    let opencode = &manifests.opencode.opencode;
    let opencode_v2 = &manifests.opencode_v2.opencode;
    let pi = &manifests.pi.pi;
    let codex = &manifests.codex.codex;
    vec![
        fixture(
            Pin::Opencode,
            &opencode.version,
            &opencode.platform,
            format!("opencode-{}-{}.tgz", opencode.platform, opencode.version),
        ),
        fixture(
            Pin::OpencodeV2,
            &opencode_v2.version,
            &opencode_v2.platform,
            format!("cli-{}-{}.tgz", opencode_v2.platform, opencode_v2.version),
        ),
        fixture(Pin::Pi, &pi.version, &pi.platform, pi.archive.clone()),
        fixture(Pin::Codex, &codex.version, &codex.platform, codex.archive.clone()),
    ]
}

/// Build the [`Fixture`] for one pin.
fn fixture(pin: Pin, version: &str, platform: &str, artifact: String) -> Fixture {
    let dir = match pin {
        Pin::Opencode => "tests/integration/fixtures/opencode-cli",
        Pin::OpencodeV2 => "tests/integration/fixtures/opencode-cli-v2",
        Pin::Pi => "tests/integration/fixtures/pi-cli",
        Pin::Codex => "tests/integration/fixtures/codex-cli",
        Pin::Claude | Pin::Vllm => unreachable!("{} has no checksum fixture", pin.name()),
    };
    Fixture {
        pin,
        dir,
        file: format!("{version}-{platform}.sha256"),
        artifact,
    }
}
