// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Tests for `check-pins`.
//!
//! Each mutation test starts from the real repository, rewrites one anchor in
//! one file in memory, and asserts the matching invariant reports it. A
//! mutation whose anchor has drifted fails loudly instead of passing vacuously.

use std::collections::BTreeSet;

use super::{
    Repo,
    check::{self, check_checksum_line, is_sha256, major},
    manifest::{self, Manifests},
    outputs,
    sites::{self, Pin},
    workspace_root,
};

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// Run every invariant against the real repository.
fn errors(repo: &Repo) -> Vec<String> {
    let manifests = Manifests::load(repo).expect("manifests should parse");
    check::check(repo, &manifests)
}

/// Run every invariant with `from` replaced by `to` (exactly once) in `path`.
fn errors_with(path: &str, from: &str, to: &str) -> Vec<String> {
    let repo = Repo::new(workspace_root());
    let contents = repo.read(path).expect("file should be readable");
    assert_eq!(
        contents.matches(from).count(),
        1,
        "{path} should contain the mutation anchor `{from}` exactly once"
    );
    errors(&repo.with_override(path, contents.replacen(from, to, 1)))
}

/// Assert some error contains every needle.
fn assert_reported(errors: &[String], needles: &[&str]) {
    assert!(
        errors.iter().any(|e| needles.iter().all(|n| e.contains(n))),
        "expected an error containing {needles:?}, got {errors:#?}"
    );
}

// -----------------------------------------------------------------------------
// Repository
// -----------------------------------------------------------------------------

#[test]
fn repository_pins_agree() {
    let errors = errors(&Repo::new(workspace_root()));
    assert!(errors.is_empty(), "pins should agree: {errors:#?}");
}

#[test]
fn outputs_match_the_pins_job_contract() {
    // Every key the GPU lanes read through `needs.pins.outputs.*` must be
    // exported, and nothing else: a renamed key would hand a lane an empty pin.
    let repo = Repo::new(workspace_root());
    let workflow = repo
        .read(".github/workflows/vllm-integration.yaml")
        .expect("workflow should be readable");
    let declared: BTreeSet<&str> = workflow
        .split("steps.load.outputs.")
        .skip(1)
        .filter_map(|rest| rest.split_whitespace().next())
        .collect();
    let manifests = Manifests::load(&repo).expect("manifests should parse");
    let exported: BTreeSet<&str> = outputs(&manifests).into_iter().map(|(key, _)| key).collect();
    assert_eq!(
        exported, declared,
        "check-pins outputs must match the pins job outputs map"
    );
}

// -----------------------------------------------------------------------------
// Sites and Fixtures
// -----------------------------------------------------------------------------

#[test]
fn drifted_harness_constant_is_reported() {
    let errors = errors_with(
        "tests/integration/tests/suite/harness.rs",
        "OPENCODE_PINNED_VERSION: &str = \"",
        "OPENCODE_PINNED_VERSION: &str = \"9",
    );
    assert_reported(&errors, &["OpenCode 1.x", "harness.rs", "found 0"]);
}

#[test]
fn drifted_workflow_env_is_reported() {
    let errors = errors_with(
        ".github/workflows/integration.yaml",
        "PI_VERSION: \"",
        "PI_VERSION: \"9",
    );
    assert_reported(&errors, &["pi pin", "integration.yaml"]);
}

#[test]
fn duplicated_site_is_reported() {
    // A second copy makes the anchor ambiguous; a bump would rewrite both.
    let path = ".github/workflows/integration.yaml";
    let repo = Repo::new(workspace_root());
    let contents = repo.read(path).expect("workflow should be readable");
    let manifests = Manifests::load(&repo).expect("manifests should parse");
    let duplicate = format!(
        "{contents}\n# CLAUDE_CODE_PINNED_VERSION: \"{}\"\n",
        manifests.claude.claude_code.version
    );
    let errors = errors(&repo.with_override(path, duplicate));
    assert_reported(&errors, &["Claude Code pin", "integration.yaml", "found 2"]);
}

#[test]
fn drifted_vllm_image_is_reported_per_site() {
    for path in [".github/workflows/vllm-dev-endpoint.yaml", "vllm/Containerfile"] {
        let errors = errors_with(path, "@sha256:", "@sha256:0");
        assert_reported(&errors, &["vLLM image", path]);
    }
}

#[test]
fn diverging_manifest_images_are_reported() {
    let errors = errors_with(manifest::CLAUDE, "image = \"docker.io/", "image = \"quay.io/");
    assert_reported(&errors, &["must pin the same vLLM image"]);
}

#[test]
fn checksum_line_must_name_the_artifact() {
    let digest = "a".repeat(64);
    assert!(check_checksum_line(&format!("{digest}  pi-linux-x64.tar.gz\n"), "pi-linux-x64.tar.gz").is_ok());
    assert!(check_checksum_line(&format!("{digest}  other.tar.gz\n"), "pi-linux-x64.tar.gz").is_err());
    assert!(check_checksum_line(&format!("{digest} pi-linux-x64.tar.gz\n"), "pi-linux-x64.tar.gz").is_err());
    assert!(check_checksum_line("abc  pi-linux-x64.tar.gz\n", "pi-linux-x64.tar.gz").is_err());
}

#[test]
fn drifted_checksum_fixture_is_reported() {
    let manifests = Manifests::load(&Repo::new(workspace_root())).expect("manifests should parse");
    let fixture = sites::fixtures(&manifests)
        .into_iter()
        .find(|fixture| fixture.pin == Pin::Codex)
        .expect("Codex should have a checksum fixture");
    let errors = errors_with(&fixture.path(), "  codex-", "  other-");
    assert_reported(&errors, &["codex-cli", "names `other-"]);
}

// -----------------------------------------------------------------------------
// Manifest Invariants
// -----------------------------------------------------------------------------

#[test]
fn claude_context_must_match_the_server_flag() {
    let errors = errors_with(
        manifest::CLAUDE,
        "\"--max-model-len\", \"32768\"",
        "\"--max-model-len\", \"16384\"",
    );
    assert_reported(&errors, &["vllm.context_length must match --max-model-len"]);
}

#[test]
fn claude_compaction_window_must_leave_headroom() {
    let errors = errors_with(
        manifest::CLAUDE,
        "max_context_tokens = 30000",
        "max_context_tokens = 15000",
    );
    assert_reported(&errors, &["must exceed the 15048-token Claude Code reserve"]);
}

#[test]
fn major_series_tripwire_is_enforced() {
    let errors = errors_with(manifest::PI, "major_series = 1", "major_series = 2");
    assert_reported(&errors, &["pi.major_series 2 disagrees"]);
}

#[test]
fn opencode_v2_must_not_declare_backend_pins() {
    let path = manifest::OPENCODE_V2;
    let repo = Repo::new(workspace_root());
    let contents = repo.read(path).expect("manifest should be readable");
    let vllm = repo
        .read(manifest::OPENCODE)
        .expect("manifest should be readable")
        .split_once("\n[vllm]\n")
        .and_then(|(_, rest)| rest.split_once("\n[gateway]\n"))
        .map(|(block, _)| format!("\n[vllm]\n{block}\n"))
        .expect("the 1.x manifest should have a [vllm] block before [gateway]");
    let errors = errors(&repo.with_override(path, format!("{contents}{vllm}")));
    assert_reported(&errors, &["must not declare [vllm]"]);
}

#[test]
fn pi_model_must_match_the_served_model() {
    let errors = errors_with(manifest::PI, "model_id = \"qwen3-8b\"", "model_id = \"other\"");
    assert_reported(&errors, &["pi.provider model other must match the served model"]);
}

#[test]
fn client_routes_must_be_explicit() {
    let errors = errors_with(
        "examples/configs/anthropic/messages-native-vllm.yaml",
        "- path: \"/v1/chat/completions\"",
        "- path: \"/v1/other\"",
    );
    assert_reported(&errors, &["must declare an explicit route for /v1/chat/completions"]);
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

#[test]
fn sha256_requires_lowercase_hex() {
    assert!(is_sha256(&"0f".repeat(32)));
    assert!(!is_sha256(&"0F".repeat(32)));
    assert!(!is_sha256(&"0f".repeat(31)));
}

#[test]
fn major_reads_the_leading_component() {
    assert_eq!(major("1.19.3"), Some(1));
    assert_eq!(major("0.144.1"), Some(0));
    assert_eq!(major("v2"), None);
}
