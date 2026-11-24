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
    for path in [manifest::CLAUDE, manifest::CODEX] {
        let errors = errors_with(path, "base_image = \"docker.io/", "base_image = \"quay.io/");
        assert_reported(&errors, &["must pin the same vLLM image", path]);
    }
}

#[test]
fn unknown_baked_image_is_reported() {
    let errors = errors_with(manifest::OPENCODE, "baked_image = \"text\"", "baked_image = \"other\"");
    assert_reported(
        &errors,
        &["vllm.baked_image `other` is not an image in vllm/images.json"],
    );
}

#[test]
fn launch_must_load_the_baked_weights() {
    // A Hugging Face ID would make vLLM download the weights at run time.
    for path in [manifest::CLAUDE, manifest::OPENCODE, manifest::CODEX] {
        let errors = errors_with(
            path,
            "\"--model\", \"/opt/vllm/models/Qwen/Qwen3-8B\"",
            "\"--model\", \"Qwen/Qwen3-8B\"",
        );
        assert_reported(
            &errors,
            &[path, "must be the baked weights' path `/opt/vllm/models/Qwen/Qwen3-8B`"],
        );
    }
}

#[test]
fn launch_must_follow_the_baked_model() {
    let errors = errors_with(
        manifest::IMAGES,
        "\"model\": \"Qwen/Qwen3-8B\"",
        "\"model\": \"Qwen/Qwen3-4B\"",
    );
    assert_reported(&errors, &[manifest::CODEX, "`/opt/vllm/models/Qwen/Qwen3-4B`"]);
}

#[test]
fn launch_flags_the_workflow_appends_are_rejected() {
    for flag in ["--api-key", "--served-model-name", "--revision", "--download-dir"] {
        for form in [format!(r#""{flag}", "x""#), format!(r#""{flag}=x""#)] {
            let errors = errors_with(manifest::CODEX, r#""--seed", "0""#, &form);
            assert_reported(&errors, &[manifest::CODEX, &format!("must not pass {flag}")]);
        }
    }
}

#[test]
fn launch_flags_must_survive_shell_word_splitting() {
    for separator in [" ", r"\t", r"\n"] {
        let bundled = format!(r#""--model{separator}other""#);
        let errors = errors_with(manifest::CODEX, r#""--seed", "0""#, &bundled);
        assert_reported(&errors, &[manifest::CODEX, "shell whitespace"]);
    }
}

#[test]
fn launch_rejects_a_second_model() {
    for form in [r#""--model", "other""#, r#""--model=other""#] {
        let errors = errors_with(manifest::CODEX, r#""--seed", "0""#, form);
        assert_reported(&errors, &[manifest::CODEX, "--model", "exactly once"]);
    }
}

#[test]
fn codex_launch_must_select_the_pinned_tool_parser() {
    let parser = r#""--tool-call-parser", "hermes""#;
    for replacement in ["", r#""--tool-call-parser", "other","#] {
        let errors = errors_with(manifest::CODEX, &format!("{parser},"), replacement);
        assert_reported(&errors, &[manifest::CODEX, "--tool-call-parser"]);
    }
    let errors = errors_with(manifest::CODEX, parser, r#""--tool-call-parser=hermes""#);
    assert!(errors.is_empty(), "pins should agree: {errors:#?}");
}

#[test]
fn codex_context_must_match_the_server_flag() {
    let errors = errors_with(manifest::CODEX, "context_length = 16384", "context_length = 32768");
    assert_reported(
        &errors,
        &[manifest::CODEX, "vllm.context_length must match --max-model-len"],
    );
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

#[test]
fn stale_checksum_fixture_is_reported() {
    let manifests = Manifests::load(&Repo::new(workspace_root())).expect("manifests should parse");
    let fixture = sites::fixtures(&manifests)
        .into_iter()
        .find(|fixture| fixture.pin == Pin::Pi)
        .expect("pi should have a checksum fixture");
    let mut repo = Repo::new(workspace_root());
    repo.stage(&format!("{}/0.0.1-linux-x64.sha256", fixture.dir), String::new());
    assert_reported(&errors(&repo), &["0.0.1-linux-x64.sha256 is stale"]);
}

#[test]
fn removed_checksum_fixture_is_reported() {
    let manifests = Manifests::load(&Repo::new(workspace_root())).expect("manifests should parse");
    let fixture = sites::fixtures(&manifests)
        .into_iter()
        .find(|fixture| fixture.pin == Pin::OpencodeV2)
        .expect("OpenCode 2.x should have a checksum fixture");
    let mut repo = Repo::new(workspace_root());
    repo.stage_removal(&fixture.path());
    assert_reported(&errors(&repo), &["OpenCode 2.x needs a checksum fixture"]);
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

// -----------------------------------------------------------------------------
// bump-pins
// -----------------------------------------------------------------------------

#[cfg(feature = "dev")]
mod bump {
    use serde_json::json;

    use super::super::{
        Repo,
        bump::{carry_codex_recording, checklist, display_version, names_version},
        sites::{Pin, Site},
        upstream::{
            is_stable, newest_in_series, newest_release, parse_claude_checksum, parse_hub_digest, parse_release,
            tags_latest_in_series,
        },
    };

    /// A site used only by these tests.
    const SITE: Site = Site {
        path: "test.yaml",
        template: "VERSION: \"{v}\"",
        count: 1,
    };

    #[test]
    fn site_rewrite_replaces_the_anchor() {
        let rewritten = SITE
            .rewrite("a\nVERSION: \"1.0.0\"\nb", "1.0.0", "1.1.0")
            .expect("anchor should rewrite");
        assert_eq!(rewritten, "a\nVERSION: \"1.1.0\"\nb");
    }

    #[test]
    fn site_rewrite_fails_on_a_missing_anchor() {
        let error = SITE
            .rewrite("VERSION: \"0.9.0\"", "1.0.0", "1.1.0")
            .expect_err("a drifted anchor should fail");
        assert!(error.contains("found 0"), "{error}");
    }

    #[test]
    fn site_rewrite_fails_on_a_duplicated_anchor() {
        let error = SITE
            .rewrite("VERSION: \"1.0.0\"\nVERSION: \"1.0.0\"", "1.0.0", "1.1.0")
            .expect_err("an ambiguous anchor should fail");
        assert!(error.contains("found 2"), "{error}");
    }

    #[test]
    fn npm_latest_is_used_only_within_the_series() {
        let tags = json!({"latest": "1.19.4", "beta": "0.0.0-beta-1"});
        assert_eq!(tags_latest_in_series(&tags, 1).as_deref(), Some("1.19.4"));
        assert_eq!(tags_latest_in_series(&tags, 2), None);
        assert_eq!(tags_latest_in_series(&json!({"latest": "2.0.0-rc.1"}), 2), None);
    }

    #[test]
    fn newest_in_series_compares_numerically_and_skips_prereleases() {
        let packument = json!({"versions": {
            "1.9.0": {}, "1.10.0": {}, "1.11.0-rc.1": {}, "2.0.0": {}, "0.0.0-dev-1": {}
        }});
        assert_eq!(newest_in_series(&packument, 1).as_deref(), Some("1.10.0"));
        assert_eq!(newest_in_series(&packument, 3), None);
    }

    #[test]
    fn github_release_strips_the_tag_prefix_and_reads_asset_digests() {
        let digest = "a".repeat(64);
        let release = json!({
            "tag_name": "rust-v0.164.0",
            "prerelease": false,
            "draft": false,
            "assets": [
                {"name": "codex.tar.gz", "digest": format!("sha256:{digest}")},
                {"name": "no-digest.tar.gz", "digest": null},
            ],
        });
        let parsed = parse_release(&release, "rust-v").expect("release should parse");
        assert_eq!(parsed.version, "0.164.0");
        assert_eq!(parsed.asset_sha256("codex.tar.gz"), Ok(digest.as_str()));
        assert!(parsed.asset_sha256("no-digest.tar.gz").is_err());
        assert!(parse_release(&release, "v").is_err(), "a mismatched prefix should fail");
    }

    #[test]
    fn github_prereleases_are_rejected() {
        let release = json!({"tag_name": "v1.5.0", "prerelease": true, "assets": []});
        assert!(parse_release(&release, "v").is_err());
    }

    #[test]
    fn newest_github_release_honors_prefix_series_and_version_order() {
        let releases = json!([
            {"tag_name": "v2.0.0", "prerelease": false, "assets": []},
            {"tag_name": "v1.10.0-rc.1", "prerelease": true, "assets": []},
            {"tag_name": "v1.9.0", "prerelease": false, "assets": []},
            {"tag_name": "v1.10.0", "prerelease": false, "assets": []},
            {"tag_name": "sdk-v9.0.0", "prerelease": false, "assets": []},
        ]);
        let newest = |prefix, series| {
            newest_release(&releases, prefix, series)
                .expect("listing should parse")
                .map(|release| release.tag)
        };
        assert_eq!(newest("v", None).as_deref(), Some("v2.0.0"));
        assert_eq!(newest("v", Some(1)).as_deref(), Some("v1.10.0"));
        assert_eq!(newest("sdk-v", None).as_deref(), Some("sdk-v9.0.0"));
        assert_eq!(newest("v", Some(3)), None);
        assert!(newest_release(&json!({}), "v", None).is_err());
    }

    #[test]
    fn claude_manifest_checksum_is_read_per_platform() {
        let checksum = "b".repeat(64);
        let manifest = json!({"platforms": {"linux-x64": {"checksum": checksum}}});
        assert_eq!(parse_claude_checksum(&manifest, "linux-x64"), Ok(checksum));
        assert!(parse_claude_checksum(&manifest, "darwin-arm64").is_err());
    }

    #[test]
    fn docker_hub_digest_must_be_sha256() {
        let digest = format!("sha256:{}", "c".repeat(64));
        assert_eq!(parse_hub_digest(&json!({"digest": digest})), Ok(digest));
        assert!(parse_hub_digest(&json!({"digest": "sha256:short"})).is_err());
    }

    #[test]
    fn stable_versions_are_three_numeric_components() {
        assert!(is_stable("0.164.0"));
        assert!(!is_stable("0.165.0-alpha.1"));
        assert!(!is_stable("1.2"));
    }

    #[test]
    fn version_mentions_match_whole_versions_only() {
        assert!(names_version("verified against pi 1.4.0.", "1.4.0"));
        assert!(names_version("`opencode v2.1.2`", "2.1.2"));
        assert!(!names_version("pi 11.4.0", "1.4.0"));
        assert!(!names_version("pi 1.4.0.1", "1.4.0"));
        assert!(!names_version("pi 1.4.01", "1.4.0"));
    }

    #[test]
    fn vllm_display_version_is_the_bare_tag() {
        let image = format!("docker.io/vllm/vllm-openai:v0.32.0@sha256:{}", "d".repeat(64));
        assert_eq!(display_version(Pin::Vllm, &image), "0.32.0");
        assert_eq!(display_version(Pin::Pi, "1.4.0"), "1.4.0");
    }

    #[test]
    fn codex_checklist_names_the_carried_recording() {
        let items = checklist(Pin::Codex, "0.163.0", "0.164.0");
        assert!(
            items
                .iter()
                .any(|item| item.contains("websocket-codex-0.164.0.json") && item.contains("`0.163.0`")),
            "{items:#?}"
        );
    }

    /// The pinned Codex version, its recording, and that recording carried
    /// to `9.9.9`.
    fn carried_pinned_recording() -> (String, String, String) {
        let repo = Repo::new(super::workspace_root());
        let manifests = super::Manifests::load(&repo).expect("manifests should parse");
        let old = Pin::Codex
            .current(&manifests)
            .expect("codex pin should load")
            .to_owned();
        let original = repo
            .read(&format!(
                "tests/integration/fixtures/openai/responses/websocket-codex-{old}.json"
            ))
            .expect("the pinned recording should exist");
        let carried = carry_codex_recording(&original, &old, "9.9.9", "2099-01-02").expect("recording should carry");
        (old, original, carried)
    }

    /// The string at `pointer` in `json`.
    fn metadata_str<'a>(json: &'a serde_json::Value, pointer: &str) -> Option<&'a str> {
        json.pointer(pointer).and_then(serde_json::Value::as_str)
    }

    #[test]
    fn codex_recording_metadata_names_the_new_version_and_its_source() {
        let (old, _, carried) = carried_pinned_recording();
        let after: serde_json::Value = serde_json::from_str(&carried).expect("carried should parse");
        assert_eq!(metadata_str(&after, "/metadata/codex_cli_version"), Some("9.9.9"));
        assert_eq!(metadata_str(&after, "/metadata/created_at"), Some("2099-01-02"));
        let source = metadata_str(&after, "/metadata/wire_protocol_source").expect("source should be a string");
        assert!(
            source.contains(&format!("carried unchanged from the {old} recording")) && source.contains("rust-v9.9.9"),
            "provenance should name both versions: {source}"
        );
    }

    #[test]
    fn codex_recording_wire_shapes_are_carried_unchanged() {
        let (_, original, carried) = carried_pinned_recording();
        let before: serde_json::Value = serde_json::from_str(&original).expect("original should parse");
        let after: serde_json::Value = serde_json::from_str(&carried).expect("carried should parse");
        for key in ["client", "server_messages"] {
            assert_eq!(after.get(key), before.get(key), "{key} must be carried unchanged");
        }
        assert_eq!(
            after.pointer("/metadata/sanitization"),
            before.pointer("/metadata/sanitization")
        );
        assert_eq!(
            carried.lines().count(),
            original.lines().count(),
            "formatting should be preserved"
        );
    }

    #[test]
    fn codex_recording_carry_fails_on_a_mismatched_version() {
        let recording =
            r#"{"metadata": {"codex_cli_version": "1.0.0", "created_at": "x", "wire_protocol_source": "y"}}"#;
        let error = carry_codex_recording(recording, "0.9.0", "1.1.0", "2099-01-02")
            .expect_err("a recording for another version should not carry");
        assert!(error.contains("codex_cli_version"), "{error}");
    }

    #[test]
    fn codex_recording_carry_escapes_the_provenance() {
        let recording = r#"{"metadata": {"codex_cli_version": "1.0.0", "created_at": "x", "wire_protocol_source": "has \"quotes\""}}"#;
        let carried = carry_codex_recording(recording, "1.0.0", "1.1.0", "2099-01-02").expect("recording should carry");
        let parsed: serde_json::Value = serde_json::from_str(&carried).expect("carried should parse");
        assert_eq!(
            parsed
                .pointer("/metadata/codex_cli_version")
                .and_then(serde_json::Value::as_str),
            Some("1.1.0")
        );
    }
}
