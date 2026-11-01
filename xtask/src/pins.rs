// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! `cargo xtask check-pins` — validate the pinned coding clients and vLLM
//! backend image across every file that records them.
//!
//! The manifests under `tests/integration/fixtures/*/pin.toml` are the source
//! of truth. This checks that every harness constant, workflow env, checksum
//! fixture, and image reference agrees with them, plus the cross-manifest
//! invariants the GPU lanes rely on. The `pins` job in
//! `.github/workflows/vllm-integration.yaml` runs it with `--github-output` to
//! hand the GPU lanes their pins, so no lane hardcodes a version.
//!
//! Built under the light `pins` feature so CI can run it without compiling the
//! proxy:
//!
//! ```console
//! cargo run -p xtask --no-default-features --features pins -- check-pins
//! ```

mod check;
mod manifest;
mod sites;
#[cfg(test)]
mod tests;

use std::{
    collections::HashMap,
    fs,
    io::Write as _,
    path::{Path, PathBuf},
};

use clap::Parser;

use self::manifest::Manifests;

// -----------------------------------------------------------------------------
// CLI Arguments
// -----------------------------------------------------------------------------

/// CLI arguments for `cargo xtask check-pins`.
#[derive(Parser)]
pub(crate) struct CheckArgs {
    /// Append the GPU lanes' pins as `key=value` lines to this file (pass
    /// `$GITHUB_OUTPUT` in CI).
    #[arg(long)]
    github_output: Option<PathBuf>,
}

// -----------------------------------------------------------------------------
// Entry Point
// -----------------------------------------------------------------------------

/// Validate every pin and optionally export the GPU lanes' pins.
pub(crate) fn run_check(args: &CheckArgs) {
    let repo = Repo::new(workspace_root());
    let manifests = Manifests::load(&repo).unwrap_or_else(|e| fail(&[e]));
    let errors = check::check(&repo, &manifests);
    if !errors.is_empty() {
        fail(&errors);
    }

    if let Some(path) = &args.github_output {
        let mut out = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .unwrap_or_else(|e| fail(&[format!("failed to open {}: {e}", path.display())]));
        for (key, value) in outputs(&manifests) {
            writeln!(out, "{key}={value}")
                .unwrap_or_else(|e| fail(&[format!("failed to write {}: {e}", path.display())]));
        }
    }
    println!("all pins agree");
}

/// Print every violation and exit non-zero.
fn fail(errors: &[String]) -> ! {
    eprintln!("pin check failed:");
    for error in errors {
        eprintln!("  - {error}");
    }
    std::process::exit(1);
}

/// The repository root, one level above this crate.
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask lives inside the workspace")
        .to_owned()
}

// -----------------------------------------------------------------------------
// GitHub Outputs
// -----------------------------------------------------------------------------

/// The pins the GPU lanes read from the `pins` job's outputs.
///
/// Keys are part of the workflow contract: the `outputs:` map of the `pins` job
/// in `vllm-integration.yaml` names each one.
fn outputs(manifests: &Manifests) -> Vec<(&'static str, String)> {
    let claude = &manifests.claude.claude_code;
    let claude_vllm = &manifests.claude.vllm;
    let opencode = &manifests.opencode.opencode;
    let opencode_v2 = &manifests.opencode_v2.opencode;
    let pi = &manifests.pi.pi;
    let mut outputs = vec![
        ("claude_url", claude.url.clone()),
        ("claude_sha256", claude.sha256.clone()),
        ("claude_version", claude.version.clone()),
        ("claude_platform", claude.platform.clone()),
        ("vllm_image", claude_vllm.image.clone()),
        ("vllm_served_model", claude_vllm.served_model_name.clone()),
        ("vllm_flags", claude_vllm.launch.flags.join(" ")),
        ("opencode_v2_version", opencode_v2.version.clone()),
        ("opencode_v2_platform", opencode_v2.platform.clone()),
        ("pi_version", pi.version.clone()),
        ("pi_platform", pi.platform.clone()),
        ("pi_model", pi.provider.model_id.clone()),
        ("opencode_version", opencode.version.clone()),
        ("opencode_install", opencode.install.clone()),
        ("opencode_npm_package", opencode.npm_package.clone()),
        ("opencode_url", opencode.url.clone()),
        ("opencode_sha256", opencode.sha256.clone()),
        ("opencode_model", opencode.provider.model_id.clone()),
    ];
    if let Some(vllm) = &manifests.opencode.vllm {
        outputs.push(("opencode_vllm_image", vllm.image.clone()));
        outputs.push(("opencode_vllm_flags", vllm.launch.flags.join(" ")));
    }
    outputs
}

// -----------------------------------------------------------------------------
// Repository Access
// -----------------------------------------------------------------------------

/// Read access to the repository, with optional in-memory overrides.
///
/// The overrides let tests mutate one file of the real repository and assert
/// the matching invariant fails, without touching disk.
pub(crate) struct Repo {
    /// Repository root.
    root: PathBuf,

    /// Replacement contents keyed by repository-relative path.
    overrides: HashMap<String, String>,
}

impl Repo {
    /// Access the repository at `root`.
    pub(crate) fn new(root: PathBuf) -> Self {
        Self {
            root,
            overrides: HashMap::new(),
        }
    }

    /// Read a repository-relative file.
    pub(crate) fn read(&self, path: &str) -> Result<String, String> {
        if let Some(contents) = self.overrides.get(path) {
            return Ok(contents.clone());
        }
        fs::read_to_string(self.root.join(path)).map_err(|e| format!("{path}: {e}"))
    }

    /// Replace one file's contents in memory.
    #[cfg(test)]
    pub(crate) fn with_override(mut self, path: &str, contents: String) -> Self {
        self.overrides.insert(path.to_owned(), contents);
        self
    }

    /// File names in a repository-relative directory, sorted.
    pub(crate) fn list(&self, dir: &str) -> Result<Vec<String>, String> {
        let mut names: Vec<String> = fs::read_dir(self.root.join(dir))
            .map_err(|e| format!("{dir}: {e}"))?
            .filter_map(|entry| entry.ok()?.file_name().into_string().ok())
            .collect();
        names.sort();
        Ok(names)
    }
}
