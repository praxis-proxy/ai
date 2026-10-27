// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Offline agreement checks for the pinned Codex CLI.
//!
//! The Codex acceptance modules are feature-gated, so these checks live apart
//! from them and run on every PR with no executable present. They keep
//! `tests/integration/fixtures/codex-cli/pin.toml`, [`CODEX_PINNED_VERSION`],
//! the checksum fixture, and every CI `CODEX_VERSION` in agreement.

use super::{
    harness::CODEX_PINNED_VERSION,
    opencode::{fixture_path, pin_value_in, repo_root, workflow_job},
};

/// The Codex pin manifest, relative to this crate's `fixtures/` directory.
const PIN_MANIFEST: &str = "codex-cli/pin.toml";

/// Read one scalar value from the Codex pin manifest.
fn pin_value(key: &str) -> String {
    pin_value_in(PIN_MANIFEST, key)
}

#[test]
fn pin_toml_version_matches_harness_const() {
    assert_eq!(
        pin_value("version"),
        CODEX_PINNED_VERSION,
        "the pin manifest and CODEX_PINNED_VERSION must agree; a bump must update both"
    );
    assert_eq!(
        pin_value("version_output"),
        format!("codex-cli {CODEX_PINNED_VERSION}"),
        "the manifest's version_output must name the pinned version"
    );
}

#[test]
fn checksum_fixture_names_the_pinned_release_archive() {
    // Every Codex lane verifies the downloaded archive against this file, so a
    // version bump that forgets it fails here rather than as a checksum
    // mismatch in CI.
    let platform = pin_value("platform");
    let archive = pin_value("archive");
    assert_eq!(
        archive,
        format!("codex-{platform}.tar.gz"),
        "the manifest should name the platform archive"
    );

    let relative = format!("codex-cli/{CODEX_PINNED_VERSION}-{platform}.sha256");
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
fn websocket_recording_matches_the_pinned_version() {
    // The WebSocket lane replays a recording of the pinned client's wire
    // protocol. A bump must recapture it against the new version; renaming the
    // old recording would misstate its provenance.
    let relative = format!("openai/responses/websocket-codex-{CODEX_PINNED_VERSION}.json");
    assert!(
        fixture_path(&relative).is_file(),
        "fixtures/{relative} should exist; recapture the Codex WebSocket fixture for the pinned version"
    );
}

#[test]
fn codex_lanes_are_wired_in_ci() {
    // Each Codex job derives every path, URL, and version assertion from its
    // job-level CODEX_VERSION, so that env is the only workflow site. Any other
    // literal occurrence of the version is a site a bump would miss.
    let pinned = format!("CODEX_VERSION: \"{CODEX_PINNED_VERSION}\"");
    for (workflow, jobs) in [
        ("integration.yaml", &["codex-http", "codex-websocket"][..]),
        ("vllm-integration.yaml", &["vllm-gpu-codex-acceptance"][..]),
    ] {
        let path = repo_root().join(".github/workflows").join(workflow);
        let yaml = std::fs::read_to_string(&path).expect("workflow should be readable");
        for job in jobs {
            assert_eq!(
                workflow_job(&yaml, job).matches(&pinned).count(),
                1,
                "the `{job}` job in {} must set {pinned:?} once",
                path.display()
            );
        }
        assert_eq!(
            yaml.matches(CODEX_PINNED_VERSION).count(),
            jobs.len(),
            "{} must not hardcode the Codex version outside each Codex job's CODEX_VERSION",
            path.display()
        );
    }
}
