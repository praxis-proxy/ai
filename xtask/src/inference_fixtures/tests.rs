// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

use std::{
    cell::RefCell,
    collections::BTreeMap,
    error::Error as _,
    fmt::Write as _,
    fs,
    io::{Read as _, Write as _},
    net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream},
    path::{Path, PathBuf},
    sync::mpsc,
    thread,
    time::Duration,
};

use clap::Parser as _;
use praxis_test_utils::inference_fixture::{
    BodyKind, FixtureProvenance, InferenceProtocol, InferenceScenario, NormalizationMetadata, ProvenanceKind,
    RecordedBody, RecordedExchange, RecordedRequest, RecordedResponse, ScenarioExpectation, ScenarioTurn, WireFixture,
    WireTurn,
};
use serde_json::json;
use tempfile::TempDir;

use super::*;
use crate::{Cli, Command};

const MODEL: &str = "fixture-model";
const PROVIDER_BODY_SENTINEL: &str = "provider-body-must-not-print";
const SECRET_SENTINEL: &str = "cli-secret-sentinel";

#[test]
fn parser_accepts_exact_record_inference_form() {
    let cli = Cli::try_parse_from([
        "xtask",
        "record-inference",
        "--scenario",
        "messages/basic-nonstream",
        "--provider",
        "vllm",
        "--provider-base-url",
        "http://10.0.0.99:8000",
        "--model",
        "RedHatAI/Qwen3-Coder-Next-NVFP4",
        "--out",
        "fixture.json",
    ])
    .expect("the documented record command should parse");

    let Command::RecordInference(args) = cli.command else {
        panic!("record-inference must remain a flat command");
    };
    assert_eq!(args.scenario, "messages/basic-nonstream");
    assert_eq!(args.provider, "vllm");
    assert_eq!(args.provider_base_url.as_deref(), Some("http://10.0.0.99:8000"));
    assert_eq!(args.model, "RedHatAI/Qwen3-Coder-Next-NVFP4");
    assert_eq!(args.out.as_deref(), Some(Path::new("fixture.json")));
}

#[test]
fn parser_accepts_exact_import_inference_form() {
    let cli = Cli::try_parse_from([
        "xtask",
        "import-inference",
        "--recording",
        "external.json",
        "--scenario",
        "messages/basic-nonstream",
        "--provider",
        "openai",
        "--out",
        "fixture.json",
    ])
    .expect("the documented import command should parse");

    let Command::ImportInference(args) = cli.command else {
        panic!("import-inference must remain a flat command");
    };
    assert_eq!(args.recording, Path::new("external.json"));
    assert_eq!(args.scenario, "messages/basic-nonstream");
    assert_eq!(args.provider, "openai");
    assert!(!args.controlled_synthetic);
    assert_eq!(args.out.as_deref(), Some(Path::new("fixture.json")));
}

#[test]
fn parser_accepts_explicit_controlled_synthetic_import_flag() {
    let cli = Cli::try_parse_from([
        "xtask",
        "import-inference",
        "--recording",
        "external.json",
        "--scenario",
        "messages/upstream-error",
        "--provider",
        "synthetic",
        "--controlled-synthetic",
    ])
    .expect("the controlled synthetic import flag should parse");

    let Command::ImportInference(args) = cli.command else {
        panic!("import-inference must remain a flat command");
    };
    assert_eq!(args.provider, "synthetic");
    assert!(args.controlled_synthetic);
}

#[test]
fn parser_accepts_exact_check_inference_form() {
    let cli = Cli::try_parse_from([
        "xtask",
        "check-inference",
        "--root",
        "tests/integration/fixtures/inference",
    ])
    .expect("the documented check command should parse");

    let Command::CheckInference(args) = cli.command else {
        panic!("check-inference must remain a flat command");
    };
    assert_eq!(args.root, Path::new("tests/integration/fixtures/inference"));
}

#[test]
fn selected_scenario_consumes_the_discovered_snapshot_after_source_mutation() {
    let root = scenario_root("snapshot-before-mutation");
    let path = root.path().join("scenarios/messages/basic-nonstream.yaml");

    let selected = load_declared_scenario_with_discovery(root.path(), "messages/basic-nonstream", |discovery_root| {
        let snapshots = discover_scenario_snapshots(discovery_root)?;
        fs::write(
            &path,
            serde_yaml::to_string(&scenario("snapshot-after-mutation")).unwrap(),
        )
        .unwrap();
        Ok(snapshots)
    })
    .expect("selection should consume the retained scenario rather than reloading its path");

    assert_eq!(selected, scenario("snapshot-before-mutation"));
    assert_eq!(
        InferenceScenario::load(&path).unwrap(),
        scenario("snapshot-after-mutation")
    );
}

#[test]
fn openai_target_reads_only_openai_key_and_builds_sensitive_bearer() {
    let env = FakeEnv::new([
        (OPENAI_API_KEY, SECRET_SENTINEL.as_bytes()),
        (ANTHROPIC_API_KEY, b"unexpected-anthropic-key".as_slice()),
        (INFERENCE_PROVIDER_API_KEY, b"unexpected-compatible-key".as_slice()),
    ]);

    let target = provider_target("openai", "https://api.openai.com/", MODEL, &env)
        .expect("the OpenAI first-party target should accept its dedicated credential");

    assert_eq!(env.reads(), [OPENAI_API_KEY]);
    let bearer = target
        .outbound_headers
        .get(reqwest::header::AUTHORIZATION)
        .expect("bearer authorization header must be present");
    assert_eq!(bearer.as_bytes(), b"Bearer cli-secret-sentinel");
    assert!(bearer.is_sensitive());
    assert!(!format!("{target:?}").contains(SECRET_SENTINEL));
    assert!(!format!("{env:?}").contains(SECRET_SENTINEL));
}

#[test]
fn anthropic_record_reads_only_anthropic_key_and_sends_native_headers() {
    let env = FakeEnv::new([
        (ANTHROPIC_API_KEY, SECRET_SENTINEL.as_bytes()),
        (OPENAI_API_KEY, b"unexpected-openai-key".as_slice()),
        (INFERENCE_PROVIDER_API_KEY, b"unexpected-compatible-key".as_slice()),
    ]);

    let target = provider_target("anthropic", "https://api.anthropic.com/", MODEL, &env)
        .expect("the Anthropic first-party target should accept its dedicated credential");

    assert_eq!(env.reads(), [ANTHROPIC_API_KEY]);
    let key = target
        .outbound_headers
        .get("x-api-key")
        .expect("native API key header must be present");
    assert_eq!(key.as_bytes(), SECRET_SENTINEL.as_bytes());
    assert!(key.is_sensitive());
    assert_eq!(target.outbound_headers.get("anthropic-version").unwrap(), "2023-06-01");
    assert!(target.outbound_headers.get(reqwest::header::AUTHORIZATION).is_none());
    assert!(!format!("{target:?}").contains(SECRET_SENTINEL));
}

#[test]
fn anthropic_record_requires_only_the_anthropic_key() {
    let root = scenario_root("ordinary prompt");
    let env = FakeEnv::new([
        (OPENAI_API_KEY, SECRET_SENTINEL.as_bytes()),
        (INFERENCE_PROVIDER_API_KEY, SECRET_SENTINEL.as_bytes()),
    ]);
    let out = root.path().join("new-parent/fixture.json");
    let args = record_args(
        root.path(),
        "anthropic",
        "https://api.anthropic.com/",
        Some(out.clone()),
        None,
    );

    let error = run_record_with(args, &env, &mut Vec::new()).expect_err("Anthropic must require its dedicated key");

    assert_eq!(env.reads(), [ANTHROPIC_API_KEY]);
    assert!(error.contains(ANTHROPIC_API_KEY));
    assert!(!error.contains(SECRET_SENTINEL));
    assert!(!out.parent().unwrap().exists());
}

#[test]
fn anthropic_invalid_credential_header_is_opaque_and_starts_no_output() {
    for (case, credential) in [
        ("line-feed", b"bad\nsecret".as_slice()),
        ("carriage-return", b"bad\rsecret".as_slice()),
    ] {
        let root = scenario_root("ordinary prompt");
        let env = FakeEnv::new([(ANTHROPIC_API_KEY, credential)]);
        let out = root.path().join(format!("new-parent/{case}.json"));
        let args = record_args(
            root.path(),
            "anthropic",
            "https://api.anthropic.com/",
            Some(out.clone()),
            None,
        );
        let mut stdout = Vec::new();

        let error = run_record_with(args, &env, &mut stdout).expect_err("invalid header bytes must be rejected");
        let surfaces = error_surfaces(&error);

        assert_eq!(env.reads(), [ANTHROPIC_API_KEY], "case {case}");
        assert_eq!(error, "provider credential could not be used", "case {case}");
        assert!(!surfaces.contains("bad"), "case {case}");
        assert!(!surfaces.contains("secret"), "case {case}");
        assert!(stdout.is_empty(), "case {case}");
        assert!(!out.parent().unwrap().exists(), "case {case}");
    }
}

#[test]
fn anthropic_rejects_non_first_party_origins_before_environment_read() {
    for (case, base_url) in [
        ("http", "http://api.anthropic.com/"),
        ("lookalike", "https://api.anthropic.com.evil/"),
        ("subdomain", "https://evil.api.anthropic.com/"),
        ("non-default-port", "https://api.anthropic.com:8443/"),
        ("userinfo", "https://user@api.anthropic.com/"),
        ("path-prefix", "https://api.anthropic.com/v1"),
        ("query", "https://api.anthropic.com/?version=2023-06-01"),
        ("fragment", "https://api.anthropic.com/#messages"),
    ] {
        let env = FakeEnv::new([(ANTHROPIC_API_KEY, SECRET_SENTINEL.as_bytes())]);
        let error = provider_target("anthropic", base_url, MODEL, &env)
            .expect_err("only the exact Anthropic first-party origin may receive its credential");

        assert_eq!(
            error, "Anthropic provider base URL must be https://api.anthropic.com/",
            "case {case}"
        );
        assert!(env.reads().is_empty(), "case {case}");
    }
}

#[test]
fn openai_rejects_non_first_party_origins_before_environment_read() {
    for (case, base_url) in [
        ("http", "http://api.openai.com/"),
        ("lookalike", "https://api.openai.com.evil/"),
        ("subdomain", "https://evil.api.openai.com/"),
        ("non-default-port", "https://api.openai.com:8443/"),
        ("userinfo", "https://user@api.openai.com/"),
        ("path-prefix", "https://api.openai.com/v1"),
        ("query", "https://api.openai.com/?version=1"),
        ("fragment", "https://api.openai.com/#responses"),
    ] {
        let env = FakeEnv::new([(OPENAI_API_KEY, SECRET_SENTINEL.as_bytes())]);
        let error = provider_target("openai", base_url, MODEL, &env)
            .expect_err("only the exact OpenAI first-party origin may receive its credential");

        assert_eq!(
            error, "OpenAI provider base URL must be https://api.openai.com/",
            "case {case}"
        );
        assert!(env.reads().is_empty(), "case {case}");
    }
}

#[test]
fn anthropic_default_base_url_is_selected_without_an_environment_read() {
    assert_eq!(default_provider_base_url("anthropic"), "https://api.anthropic.com");
}

#[test]
fn compatible_record_rejects_credential_equal_to_model_before_creating_output() {
    let root = scenario_root("ordinary prompt");
    let provider = LocalProvider::start_with_response_model(PROVIDER_BODY_SENTINEL, "provider-safe-model");
    let env = FakeEnv::new([(INFERENCE_PROVIDER_API_KEY, MODEL.as_bytes())]);
    let out = root.path().join("must-not-exist/openai.json");
    let args = record_args(root.path(), "compatible", provider.base_url(), Some(out.clone()), None);
    let mut stdout = Vec::new();

    let error = run_record_with(args, &env, &mut stdout)
        .expect_err("a configured credential matching the model must prevent fixture output");
    provider.finish();
    let surfaces = error_surfaces(&error);

    assert_eq!(
        error,
        "fixture commit safety violation: configured credential at $/provenance/model"
    );
    assert!(!surfaces.contains(MODEL));
    assert!(stdout.is_empty());
    assert!(!out.exists());
    assert!(!out.parent().unwrap().exists());
}

#[test]
fn non_openai_record_reads_only_optional_generic_key() {
    let root = scenario_root("ordinary prompt");
    let provider = LocalProvider::start(PROVIDER_BODY_SENTINEL);
    let env = FakeEnv::new([(INFERENCE_PROVIDER_API_KEY, SECRET_SENTINEL.as_bytes())]);
    let out = root.path().join("generic.json");
    let args = record_args(root.path(), "compatible", provider.base_url(), Some(out.clone()), None);
    let mut stdout = Vec::new();

    run_record_with(args, &env, &mut stdout).expect("generic credential should authorize a compatible provider");

    assert_eq!(env.reads(), [INFERENCE_PROVIDER_API_KEY]);
    let request = provider.finish();
    assert!(
        request
            .to_ascii_lowercase()
            .contains("authorization: bearer cli-secret-sentinel")
    );
    assert_secret_absent(&stdout, &out, SECRET_SENTINEL);
}

#[test]
fn vllm_target_reads_no_credential_and_builds_no_outbound_headers() {
    let env = FakeEnv::new([(INFERENCE_PROVIDER_API_KEY, SECRET_SENTINEL.as_bytes())]);

    let target = provider_target("vllm", "http://10.0.0.99:8000", MODEL, &env)
        .expect("vLLM must build a credentialless private target");

    assert!(env.reads().is_empty());
    assert!(target.outbound_headers.is_empty());
    assert_eq!(target.provider, "vllm");
    assert_eq!(target.model, MODEL);
}

#[test]
fn vllm_record_succeeds_without_a_credential() {
    let root = scenario_root("ordinary prompt");
    let provider = LocalProvider::start(PROVIDER_BODY_SENTINEL);
    let env = FakeEnv::default();
    let out = root.path().join("vllm.json");
    let args = record_args(root.path(), "vllm", provider.base_url(), Some(out.clone()), None);
    let mut stdout = Vec::new();

    run_record_with(args, &env, &mut stdout).expect("vLLM should not require a credential");

    assert!(env.reads().is_empty());
    let request = provider.finish();
    assert!(!request.to_ascii_lowercase().contains("authorization:"));
    assert!(out.is_file());
}

#[test]
fn openai_invalid_credential_header_is_opaque() {
    let env = FakeEnv::new([(OPENAI_API_KEY, b"bad\nsecret".as_slice())]);

    let error = provider_target("openai", "https://api.openai.com/", MODEL, &env)
        .expect_err("invalid header bytes must be rejected");
    let surfaces = error_surfaces(&error);

    assert_eq!(error, "provider credential could not be used");
    assert!(!surfaces.contains("bad"));
    assert!(!surfaces.contains("secret"));
}

#[test]
fn openai_record_requires_only_the_openai_key() {
    let root = scenario_root("ordinary prompt");
    let env = FakeEnv::new([(INFERENCE_PROVIDER_API_KEY, SECRET_SENTINEL.as_bytes())]);
    let out = root.path().join("new-parent/fixture.json");
    let args = record_args(
        root.path(),
        "openai",
        "https://api.openai.com/",
        Some(out.clone()),
        None,
    );

    let error = run_record_with(args, &env, &mut Vec::new()).expect_err("OpenAI must require its dedicated key");

    assert_eq!(env.reads(), [OPENAI_API_KEY]);
    assert!(error.contains(OPENAI_API_KEY));
    assert!(!error.contains(SECRET_SENTINEL));
    assert!(!out.parent().unwrap().exists());
}

#[test]
fn compatible_json_credential_reflection_is_opaque_and_creates_no_artifact() {
    let secret = r#"cli-\"credential\"\\tail"#;
    let root = scenario_root("ordinary prompt");
    let provider = LocalProvider::start(secret);
    let env = FakeEnv::new([(INFERENCE_PROVIDER_API_KEY, secret.as_bytes())]);
    let out = root.path().join("must-not-exist/openai.json");
    let args = record_args(root.path(), "compatible", provider.base_url(), Some(out.clone()), None);
    let mut stdout = Vec::new();

    let error =
        run_record_with(args, &env, &mut stdout).expect_err("an echoed compatible credential must prevent output");
    let request = provider.finish();
    let surfaces = error_surfaces(&error);

    assert!(request.contains("authorization: Bearer"));
    assert_eq!(error, "recording provider response capture was incomplete");
    assert!(!surfaces.contains(secret));
    assert!(!String::from_utf8_lossy(&stdout).contains(secret));
    let stderr = format!("{error}\n");
    assert!(!stderr.contains(secret));
    assert!(stdout.is_empty());
    assert!(!out.exists());
    assert!(!out.parent().unwrap().exists());
}

#[test]
fn generic_header_credential_reflection_is_opaque_and_creates_no_artifact() {
    let secret = "generic-cli-credential";
    let root = scenario_root("ordinary prompt");
    let provider = LocalProvider::start_with_request_id(PROVIDER_BODY_SENTINEL, secret);
    let env = FakeEnv::new([(INFERENCE_PROVIDER_API_KEY, secret.as_bytes())]);
    let out = root.path().join("must-not-exist/generic.json");
    let args = record_args(root.path(), "compatible", provider.base_url(), Some(out.clone()), None);
    let mut stdout = Vec::new();

    let error =
        run_record_with(args, &env, &mut stdout).expect_err("an echoed generic credential must prevent fixture output");
    provider.finish();
    let surfaces = error_surfaces(&error);

    assert_eq!(error, "recording provider response capture was incomplete");
    assert!(!surfaces.contains(secret));
    assert!(!String::from_utf8_lossy(&stdout).contains(secret));
    let stderr = format!("{error}\n");
    assert!(!stderr.contains(secret));
    assert!(stdout.is_empty());
    assert!(!out.exists());
    assert!(!out.parent().unwrap().exists());
}

#[test]
fn custom_redactions_participate_in_the_runner_sanitizer() {
    let source = "custom-literal-source";
    let replacement = "<custom-redacted>";
    let root = scenario_root(source);
    let provider = LocalProvider::start(PROVIDER_BODY_SENTINEL);
    let redactions = root.path().join("redactions.json");
    fs::write(&redactions, format!(r#"{{"{source}":"{replacement}"}}"#)).unwrap();
    let out = root.path().join("custom.json");
    let args = record_args(
        root.path(),
        "vllm",
        provider.base_url(),
        Some(out.clone()),
        Some(redactions),
    );

    run_record_with(args, &FakeEnv::default(), &mut Vec::new())
        .expect("custom literal should be sanitized by the runner");
    provider.finish();

    let fixture = fs::read_to_string(out).unwrap();
    assert!(!fixture.contains(source));
    assert!(fixture.contains(replacement));
}

#[test]
fn import_custom_rules_cannot_change_protected_fixture_structure() {
    for (case, source) in [
        ("scenario", "messages/basic-nonstream"),
        ("provider", "openai"),
        ("model", MODEL),
        ("client-path", "/v1/messages"),
        ("upstream-path", "/v1/chat/completions"),
        ("turn-name", "initial"),
    ] {
        let root = scenario_root("ordinary prompt");
        let recording = root.path().join("external.json");
        write_external_recording(&recording, 64, MODEL, "source-import");
        let redactions = root.path().join("redactions.json");
        fs::write(
            &redactions,
            serde_json::to_vec(&BTreeMap::from([(source, "changed")])).unwrap(),
        )
        .unwrap();
        let out = root.path().join(format!("must-not-exist/{case}.json"));
        let args = import_args(root.path(), recording, "openai", out.clone(), Some(redactions));

        let result = run_import_with(args, &mut Vec::new());
        let Err(error) = result else {
            panic!("import rule overlapping protected {case} must fail");
        };

        assert_eq!(error, "fixture sanitizer changed protected structure");
        assert!(!out.exists());
        assert!(!out.parent().unwrap().exists());
    }
}

#[test]
fn import_one_sided_request_structure_redactions_fail_before_persistence() {
    for (case, field, source, replacement) in [
        (
            "path",
            "endpoint",
            "/imported-only-upstream-path",
            "/v1/chat/completions",
        ),
        ("method", "method", "IMPORTED-ONLY-METHOD", "POST"),
    ] {
        let root = scenario_root("ordinary prompt");
        let recording = root.path().join("external.json");
        write_external_recording(&recording, 64, MODEL, "source-import");
        let mut document: serde_json::Value = serde_json::from_slice(&fs::read(&recording).unwrap()).unwrap();
        document["request"][field] = json!(source);
        fs::write(&recording, document.to_string()).unwrap();
        let redactions = root.path().join("redactions.json");
        fs::write(
            &redactions,
            serde_json::to_vec(&BTreeMap::from([(source, replacement)])).unwrap(),
        )
        .unwrap();
        let out = root.path().join(format!("must-not-exist/{case}.json"));
        let args = import_args(root.path(), recording, "openai", out.clone(), Some(redactions));
        let mut stdout = Vec::new();

        let result = run_import_with(args, &mut stdout);
        let Err(error) = result else {
            panic!("an imported-only {case} rewrite must not mask a request mismatch");
        };
        let surfaces = error_surfaces(&error);

        assert_eq!(error, "fixture sanitizer changed protected structure");
        assert!(!surfaces.contains(source));
        assert!(!surfaces.contains(replacement));
        assert!(stdout.is_empty());
        assert!(!out.exists());
        assert!(!out.parent().unwrap().exists());
    }
}

#[test]
fn live_custom_rules_cannot_change_protected_fixture_structure() {
    for (case, source) in [
        ("scenario", "messages/basic-nonstream"),
        ("provider", "vllm"),
        ("model", MODEL),
        ("client-path", "/v1/messages"),
        ("upstream-path", "/v1/chat/completions"),
        ("turn-name", "initial"),
    ] {
        let root = scenario_root("ordinary prompt");
        let provider = LocalProvider::start(PROVIDER_BODY_SENTINEL);
        let redactions = root.path().join("redactions.json");
        fs::write(
            &redactions,
            serde_json::to_vec(&BTreeMap::from([(source, "changed")])).unwrap(),
        )
        .unwrap();
        let out = root.path().join(format!("must-not-exist/{case}.json"));
        let args = record_args(
            root.path(),
            "vllm",
            provider.base_url(),
            Some(out.clone()),
            Some(redactions),
        );

        let result = run_record_with(args, &FakeEnv::default(), &mut Vec::new());
        provider.finish();
        let Err(error) = result else {
            panic!("live rule overlapping protected {case} must fail");
        };

        assert_eq!(error, "fixture sanitizer changed protected structure");
        assert!(!out.exists());
        assert!(!out.parent().unwrap().exists());
    }
}

#[test]
fn redactions_file_rejects_wrong_shapes_duplicates_and_empty_sources_opaquely() {
    let root = tempfile::tempdir().unwrap();
    for (name, document) in [
        ("array", r#"["secret"]"#),
        ("number", r#"{"secret": 7}"#),
        ("duplicate", r#"{"secret":"one","secret":"two"}"#),
        ("empty", r#"{"":"replacement"}"#),
    ] {
        let path = root.path().join(format!("{name}.json"));
        fs::write(&path, document).unwrap();

        let error = load_redaction_rules(&path).expect_err("invalid redaction documents must fail closed");

        assert!(!error.contains("secret"));
        assert!(!error.contains("replacement"));
        assert!(!error.contains("one"));
        assert!(!error.contains("two"));
    }
}

#[test]
fn default_output_preserves_scenario_subdirectories_and_has_one_newline() {
    let root = scenario_root("ordinary prompt");
    let provider = LocalProvider::start(PROVIDER_BODY_SENTINEL);
    let args = record_args(root.path(), "vllm", provider.base_url(), None, None);
    let mut stdout = Vec::new();

    run_record_with(args, &FakeEnv::default(), &mut stdout).expect("default output should be written");
    provider.finish();

    let out = root.path().join("recordings/vllm/messages/basic-nonstream.json");
    let bytes = fs::read(&out).unwrap();
    assert!(out.parent().unwrap().is_dir());
    assert!(bytes.ends_with(b"\n"));
    assert!(!bytes.ends_with(b"\n\n"));
    let reported = fs::canonicalize(root.path())
        .unwrap()
        .join("recordings/vllm/messages/basic-nonstream.json");
    assert_eq!(
        String::from_utf8(stdout).unwrap(),
        format!("wrote inference fixture {}\n", reported.display())
    );
}

#[test]
fn scenario_and_provider_path_components_are_rejected() {
    for scenario in [
        "",
        " messages/case",
        "messages/case ",
        ".",
        "..",
        "/absolute",
        "messages//case",
        "messages/../case",
        "messages\\case",
    ] {
        assert!(
            validate_scenario_id(scenario).is_err(),
            "scenario `{scenario}` must be rejected"
        );
    }
    for provider in [
        "",
        " openai",
        "openai ",
        ".",
        "..",
        "/absolute",
        "nested/provider",
        "nested\\provider",
    ] {
        assert!(
            validate_provider(provider).is_err(),
            "provider `{provider}` must be rejected"
        );
    }
}

#[test]
fn stable_ids_require_portable_lowercase_ascii_components() {
    for scenario in [
        "/messages/basic",
        "messages/basic/",
        "messages//basic",
        "Messages/basic",
        "messages/Basic",
        "messages/-basic",
        "messages/_basic",
        "messages/.basic",
        "messages/basic.",
        "messages/basic:alternate",
        "messages/ba\0sic",
        "messages/basíc",
        "messages/／basic",
        "con/basic",
        "messages/nul.txt",
        "messages/com1.json",
        "messages/lpt9.log",
    ] {
        assert!(
            validate_scenario_id(scenario).is_err(),
            "nonportable scenario `{scenario:?}` must be rejected"
        );
    }
    for provider in [
        "OpenAI",
        "-openai",
        "_openai",
        ".openai",
        "openai.",
        "openai:profile",
        "open\0ai",
        "opénai",
        "／openai",
        "con",
        "nul.txt",
        "com1.json",
        "lpt9.log",
    ] {
        assert!(
            validate_provider(provider).is_err(),
            "nonportable provider `{provider:?}` must be rejected"
        );
    }
    for scenario in ["messages/basic-nonstream", "responses/tools.v1_case"] {
        assert!(
            validate_scenario_id(scenario).is_ok(),
            "canonical scenario must remain valid"
        );
    }
    for provider in ["openai", "vllm", "compatible", "azure-openai"] {
        assert!(
            validate_provider(provider).is_ok(),
            "canonical provider must remain valid"
        );
    }
}

#[test]
fn noncanonical_openai_is_rejected_before_any_environment_read() {
    let env = FakeEnv::new([
        (OPENAI_API_KEY, SECRET_SENTINEL.as_bytes()),
        (INFERENCE_PROVIDER_API_KEY, SECRET_SENTINEL.as_bytes()),
    ]);

    let result = provider_target("OpenAI", "http://127.0.0.1:8000", MODEL, &env);

    assert!(result.is_err());
    assert!(env.reads().is_empty());
}

#[test]
fn record_rejects_unknown_declared_scenario_before_creating_output_parent() {
    let root = scenario_root("ordinary prompt");
    let out = root.path().join("missing-parent/fixture.json");
    let mut args = record_args(root.path(), "vllm", "http://127.0.0.1:9", Some(out.clone()), None);
    args.scenario = "messages/not-declared".to_owned();

    let error = run_record_with(args, &FakeEnv::default(), &mut Vec::new())
        .expect_err("constructed paths must not substitute for scenario discovery");

    assert!(error.contains("messages/not-declared"));
    assert!(!out.parent().unwrap().exists());
}

#[test]
fn import_mismatch_creates_neither_parent_nor_output() {
    let root = scenario_root("ordinary prompt");
    let recording = root.path().join("external.json");
    write_external_recording(&recording, 999, MODEL, "source-import");
    let out = root.path().join("must-not-exist/fixture.json");
    let args = import_args(root.path(), recording, "openai", out.clone(), None);
    let mut stdout = Vec::new();

    let error = run_import_with(args, &mut stdout).expect_err("transformed request mismatch must fail");

    assert!(error.contains("replay mismatch at turns[0].upstream.request"));
    assert!(!out.exists());
    assert!(!out.parent().unwrap().exists());
    assert!(stdout.is_empty());
    assert!(!error.contains("ordinary prompt"));
    assert!(!error.contains(PROVIDER_BODY_SENTINEL));
}

#[test]
fn import_constructs_provenance_and_writes_commit_safe_fixture() {
    let root = scenario_root("ordinary prompt");
    let recording = root.path().join("external.json");
    write_external_recording(&recording, 64, MODEL, "source-import");
    let out = root.path().join("nested/imported.json");
    let args = import_args(root.path(), recording, "openai", out.clone(), None);

    run_import_with(args, &mut Vec::new()).expect("matching external recording should materialize");

    let fixture = WireFixture::load(&out).unwrap();
    assert_eq!(fixture.provenance.kind, ProvenanceKind::Imported);
    assert_eq!(fixture.provenance.provider, "openai");
    assert_eq!(fixture.provenance.model, MODEL);
    assert_eq!(fixture.provenance.source_id.as_deref(), Some("source-import"));
    assert_eq!(fixture.turns.len(), 1);
    assert_eq!(fixture.turns[0].client.request.path, "/v1/messages");
    assert_eq!(fixture.turns[0].upstream.request.path, "/v1/chat/completions");
}

#[test]
fn synthetic_provider_without_controlled_flag_remains_imported_provenance() {
    let root = scenario_root("ordinary prompt");
    let recording = root.path().join("external.json");
    write_external_recording(&recording, 64, MODEL, "source-import");
    let out = root.path().join("nested/imported.json");
    let args = import_args(root.path(), recording, "synthetic", out.clone(), None);

    run_import_with(args, &mut Vec::new()).expect("ordinary synthetic-provider import should materialize");

    let fixture = WireFixture::load(&out).unwrap();
    assert_eq!(fixture.provenance.kind, ProvenanceKind::Imported);
    assert_eq!(fixture.provenance.provider, "synthetic");
}

#[test]
fn controlled_synthetic_flag_rejects_other_providers_before_reading_inputs() {
    let root = tempfile::tempdir().unwrap();
    let mut args = import_args(
        root.path(),
        root.path().join("missing-external.json"),
        "openai",
        root.path().join("missing-output.json"),
        None,
    );
    args.controlled_synthetic = true;

    let error = run_import_with(args, &mut Vec::new())
        .expect_err("controlled synthetic provenance must require the synthetic provider");

    assert_eq!(error, "controlled synthetic imports require provider `synthetic`");
}

#[test]
fn import_materializes_explicit_external_response_status_without_post_editing() {
    let root = scenario_root("ordinary prompt");
    let scenario_path = root.path().join("scenarios/messages/basic-nonstream.yaml");
    let mut error_scenario = scenario("ordinary prompt");
    error_scenario.turns[0].expect.client_status = 429;
    fs::write(&scenario_path, serde_yaml::to_string(&error_scenario).unwrap()).unwrap();
    let recording = root.path().join("external.json");
    write_external_recording(&recording, 64, MODEL, "source-import");
    let mut envelope: serde_json::Value = serde_json::from_slice(&fs::read(&recording).unwrap()).unwrap();
    envelope["response"]["status"] = json!(429);
    envelope["response"]["body"] = json!({"error": {"message": "slow down", "type": "rate_limit_error"}});
    fs::write(&recording, envelope.to_string()).unwrap();
    let out = root.path().join("nested/imported-error.json");
    let mut args = import_args(root.path(), recording, "synthetic", out.clone(), None);
    args.controlled_synthetic = true;

    run_import_with(args, &mut Vec::new()).expect("explicit external status should materialize through the CLI");

    let fixture = WireFixture::load(&out).unwrap();
    assert_eq!(fixture.provenance.kind, ProvenanceKind::Synthetic);
    assert_eq!(fixture.provenance.provider, "synthetic");
    assert_eq!(fixture.turns[0].upstream.response.status, 429);
    assert_eq!(fixture.turns[0].client.response.status, 429);
    assert_eq!(
        fixture.turns[0].upstream.response.body,
        RecordedBody::Json {
            value: json!({"error": {"message": "slow down", "type": "rate_limit_error"}})
        }
    );
}

#[test]
fn import_rejects_model_inconsistency_opaquely() {
    let root = scenario_root("ordinary prompt");
    let recording = root.path().join("external.json");
    write_external_recording(&recording, 64, "different-model", SECRET_SENTINEL);
    let mut document: serde_json::Value = serde_json::from_slice(&fs::read(&recording).unwrap()).unwrap();
    document["request"]["body"]["model"] = json!(MODEL);
    fs::write(&recording, document.to_string()).unwrap();
    let out = root.path().join("nested/imported.json");
    let args = import_args(root.path(), recording, "openai", out.clone(), None);

    let error = run_import_with(args, &mut Vec::new()).expect_err("scenario model mismatch must fail");

    assert!(!error.contains("different-model"));
    assert!(!error.contains(SECRET_SENTINEL));
    assert!(!out.parent().unwrap().exists());
}

#[test]
fn existing_output_is_never_overwritten() {
    let root = scenario_root("ordinary prompt");
    let recording = root.path().join("external.json");
    write_external_recording(&recording, 64, MODEL, "source-import");
    let out = root.path().join("existing.json");
    fs::write(&out, "keep-me\n").unwrap();
    let args = import_args(root.path(), recording, "openai", out.clone(), None);

    let error = run_import_with(args, &mut Vec::new()).expect_err("existing output must require an explicit decision");

    assert!(error.contains("already exists"));
    assert_eq!(fs::read_to_string(out).unwrap(), "keep-me\n");
}

#[cfg(unix)]
#[test]
fn default_output_rejects_symlinks_at_every_parent_depth_without_external_artifacts() {
    use std::os::unix::fs::symlink;

    for symlink_parent in ["recordings", "recordings/openai", "recordings/openai/messages"] {
        let root = scenario_root("ordinary prompt");
        let recording = root.path().join("external.json");
        write_external_recording(&recording, 64, MODEL, "source-import");
        let external = tempfile::tempdir().unwrap();
        let link = root.path().join(symlink_parent);
        fs::create_dir_all(link.parent().unwrap()).unwrap();
        symlink(external.path(), &link).unwrap();
        let args = import_default_args(root.path(), recording, "openai");

        let result = run_import_with(args, &mut Vec::new());
        let Err(error) = result else {
            panic!("default output symlink `{symlink_parent}` must be rejected");
        };

        assert_eq!(error, "default inference fixture output path is unsafe");
        assert_eq!(fs::read_dir(external.path()).unwrap().count(), 0);
        assert!(fs::symlink_metadata(link).unwrap().file_type().is_symlink());
    }
}

#[cfg(unix)]
#[test]
fn default_output_allows_the_selected_root_itself_to_be_a_symlink() {
    use std::os::unix::fs::symlink;

    let real_root = scenario_root("ordinary prompt");
    let recording = real_root.path().join("external.json");
    write_external_recording(&recording, 64, MODEL, "source-import");
    let selection = tempfile::tempdir().unwrap();
    let selected_root = selection.path().join("fixture-root");
    symlink(real_root.path(), &selected_root).unwrap();
    let args = import_default_args(&selected_root, recording, "openai");
    let mut stdout = Vec::new();

    run_import_with(args, &mut stdout).expect("a selected root symlink should resolve once at the trust boundary");

    let output = fs::canonicalize(real_root.path())
        .unwrap()
        .join("recordings/openai/messages/basic-nonstream.json");
    assert!(output.is_file());
    assert_eq!(
        String::from_utf8(stdout).unwrap(),
        format!("wrote inference fixture {}\n", output.display())
    );
}

#[cfg(unix)]
#[test]
fn explicit_output_remains_user_authorized_through_a_symlinked_parent() {
    use std::os::unix::fs::symlink;

    let root = scenario_root("ordinary prompt");
    let recording = root.path().join("external.json");
    write_external_recording(&recording, 64, MODEL, "source-import");
    let external = tempfile::tempdir().unwrap();
    let linked_parent = root.path().join("explicit-link");
    symlink(external.path(), &linked_parent).unwrap();
    let out = linked_parent.join("explicit.json");
    let args = import_args(root.path(), recording, "openai", out.clone(), None);

    run_import_with(args, &mut Vec::new()).expect("explicit output authorizes its selected parent path");

    assert!(out.is_file());
    assert!(external.path().join("explicit.json").is_file());
}

#[test]
fn bounded_inputs_fail_without_echoing_content() {
    let root = tempfile::tempdir().unwrap();
    let recording = root.path().join("external.json");
    let redactions = root.path().join("redactions.json");
    write_oversized(&recording, MAX_EXTERNAL_RECORDING_BYTES, SECRET_SENTINEL.as_bytes());
    write_oversized(&redactions, MAX_REDACTIONS_FILE_BYTES, SECRET_SENTINEL.as_bytes());

    let recording_error = read_bounded(&recording, MAX_EXTERNAL_RECORDING_BYTES, "external recording")
        .expect_err("oversized recording must be rejected");
    let redaction_error = load_redaction_rules(&redactions).expect_err("oversized redactions must be rejected");

    assert!(!recording_error.contains(SECRET_SENTINEL));
    assert!(!redaction_error.contains(SECRET_SENTINEL));
}

#[test]
fn check_prints_one_deterministic_line_without_fixture_content() {
    let root = coverage_root();
    let mut stdout = Vec::new();

    run_check_with(
        &CheckArgs {
            root: root.path().to_path_buf(),
        },
        &mut stdout,
    )
    .expect("valid coverage tree should pass");

    let output = String::from_utf8(stdout).unwrap();
    assert_eq!(
        output,
        "inference coverage ok: features=1 scenarios=1 recordings=1 statuses=covered:1\n"
    );
    assert_eq!(output.lines().count(), 1);
    assert!(!output.contains(PROVIDER_BODY_SENTINEL));
    assert!(!output.contains(SECRET_SENTINEL));
    assert!(!output.to_ascii_lowercase().contains("authorization"));
}

#[test]
fn check_rejects_unsafe_persisted_fixture_without_printing_values() {
    const CREDENTIAL: &str = "Bearer persisted-credential-sentinel";
    const PRIVATE_PATH: &str = "/Users/fixture-owner/private/inference.json";

    let root = coverage_root();
    let path = root.path().join("recordings/openai/messages/basic-nonstream.json");
    let mut fixture = WireFixture::load(&path).expect("coverage fixture should load");
    fixture.turns[0].client.request.body = RecordedBody::Json {
        value: json!({
            "credential": CREDENTIAL,
            "private_path": PRIVATE_PATH,
        }),
    };
    fixture.write(&path).expect("unsafe fixture should remain schema-valid");
    let mut stdout = Vec::new();

    let error = run_check_with(
        &CheckArgs {
            root: root.path().to_path_buf(),
        },
        &mut stdout,
    )
    .expect_err("persisted fixtures must satisfy the default commit-safety policy");

    assert_eq!(error, "inference recording validation failed");
    assert!(!error.contains(CREDENTIAL));
    assert!(!error.contains(PRIVATE_PATH));
    assert!(stdout.is_empty());
}

#[test]
fn check_rejects_unknown_recorded_body_fields_before_lossy_projection() {
    const CREDENTIAL: &str = "Bearer raw-unknown-body-sentinel";
    const PRIVATE_PATH: &str = "/Users/raw-unknown/private/inference.json";

    let root = coverage_root();
    let path = coverage_recording_path(root.path());
    mutate_text_file(&path, |document| {
        replace_once(
            document,
            "\"kind\": \"json\",\n            \"value\": {",
            "\"kind\": \"json\",\n            \"unknown\": \"Bearer raw-unknown-body-sentinel /Users/raw-unknown/private/inference.json\",\n            \"value\": {",
        );
    });

    let error = opaque_check_error(
        root.path(),
        &[CREDENTIAL, PRIVATE_PATH, path.to_string_lossy().as_ref()],
    );

    assert_eq!(error, "inference recording validation failed");
}

#[test]
fn check_rejects_duplicate_payload_keys_before_last_value_wins() {
    const CREDENTIAL: &str = "Bearer raw-duplicate-value-sentinel";
    const PRIVATE_PATH: &str = "/Users/raw-duplicate/private/inference.json";

    let root = coverage_root();
    let path = coverage_recording_path(root.path());
    mutate_text_file(&path, |document| {
        replace_once(
            document,
            "\"secret\": \"cli-secret-sentinel\"",
            "\"secret\": \"Bearer raw-duplicate-value-sentinel /Users/raw-duplicate/private/inference.json\",\n              \"secr\\u0065t\": \"safe\"",
        );
    });

    let error = opaque_check_error(
        root.path(),
        &[CREDENTIAL, PRIVATE_PATH, path.to_string_lossy().as_ref()],
    );

    assert_eq!(error, "inference recording validation failed");
}

#[test]
fn check_rejects_unknown_recorded_body_field_names_opaquely() {
    const FIELD_NAME: &str = "Bearer raw-unknown-key-sentinel";

    let root = coverage_root();
    let path = coverage_recording_path(root.path());
    mutate_text_file(&path, |document| {
        replace_once(
            document,
            "\"kind\": \"json\",\n            \"value\": {",
            "\"kind\": \"json\",\n            \"Bearer raw-unknown-key-sentinel\": \"safe\",\n            \"value\": {",
        );
    });

    let error = opaque_check_error(root.path(), &[FIELD_NAME, path.to_string_lossy().as_ref()]);

    assert_eq!(error, "inference recording validation failed");
}

#[test]
fn check_maps_untrusted_recording_identities_to_opaque_categories() {
    const SCENARIO_ID: &str = "unsafe-scenario-id-sentinel";
    const PROVIDER: &str = "unsafe-provider-sentinel";
    const SOURCE_ID: &str = "Bearer unsafe-source-id-sentinel";

    for (needle, replacement, expected) in [
        (
            "\"scenario_id\": \"messages/basic-nonstream\"",
            "\"scenario_id\": \"unsafe-scenario-id-sentinel\"",
            "inference coverage validation failed",
        ),
        (
            "\"provider\": \"openai\"",
            "\"provider\": \"unsafe-provider-sentinel\"",
            "inference coverage validation failed",
        ),
        (
            "\"source_id\": \"coverage-source\"",
            "\"source_id\": \"Bearer unsafe-source-id-sentinel\"",
            "inference recording validation failed",
        ),
    ] {
        let root = coverage_root();
        let path = coverage_recording_path(root.path());
        mutate_text_file(&path, |document| replace_once(document, needle, replacement));

        let error = opaque_check_error(
            root.path(),
            &[SCENARIO_ID, PROVIDER, SOURCE_ID, path.to_string_lossy().as_ref()],
        );

        assert_eq!(error, expected);
    }
}

#[test]
fn check_maps_duplicate_recording_identity_to_an_opaque_category() {
    let root = coverage_root();
    let first = coverage_recording_path(root.path());
    let second = root.path().join("recordings/duplicate.json");
    fs::copy(&first, &second).expect("raw duplicate fixture should be copied");

    let error = opaque_check_error(
        root.path(),
        &[
            "messages/basic-nonstream",
            "openai",
            first.to_string_lossy().as_ref(),
            second.to_string_lossy().as_ref(),
        ],
    );

    assert_eq!(error, "inference coverage validation failed");
}

#[test]
fn check_preserves_coverage_first_precedence_without_exposing_values() {
    const UNKNOWN_FEATURE: &str = "unsafe-feature-sentinel";
    const SOURCE_ID: &str = "Bearer unsafe-precedence-source-sentinel";

    let root = coverage_root();
    let scenario_path = root.path().join("scenarios/messages/basic-nonstream.yaml");
    mutate_text_file(&scenario_path, |document| {
        replace_once(document, "- messages.basic", "- unsafe-feature-sentinel");
    });
    let recording_path = coverage_recording_path(root.path());
    mutate_text_file(&recording_path, |document| {
        replace_once(
            document,
            "\"source_id\": \"coverage-source\"",
            "\"source_id\": \"Bearer unsafe-precedence-source-sentinel\"",
        );
    });

    let error = opaque_check_error(
        root.path(),
        &[
            UNKNOWN_FEATURE,
            SOURCE_ID,
            scenario_path.to_string_lossy().as_ref(),
            recording_path.to_string_lossy().as_ref(),
        ],
    );

    assert_eq!(error, "inference coverage validation failed");
}

#[test]
fn check_maps_persisted_document_resource_failures_to_static_categories() {
    for error in [
        FixtureError::PersistedDocumentTooLarge {
            kind: "controlled-test-kind",
        },
        FixtureError::PersistedDocumentAllocation,
    ] {
        let message = super::opaque_check_error(&error);
        assert_eq!(message, "inference recording validation failed");
        assert!(!message.contains("controlled-test-kind"));
    }
}

#[test]
fn provider_target_debug_never_exposes_outbound_headers() {
    let env = FakeEnv::new([(OPENAI_API_KEY, SECRET_SENTINEL.as_bytes())]);

    let target = provider_target("openai", "https://api.openai.com/", MODEL, &env).unwrap();

    let debug = format!("{target:?}");
    assert!(!debug.contains(SECRET_SENTINEL));
    assert!(debug.contains("[redacted]"));
}

// Test Utilities
// -----------------------------------------------------------------------------

#[derive(Default)]
struct FakeEnv {
    reads: RefCell<Vec<String>>,
    values: BTreeMap<String, Vec<u8>>,
}

impl std::fmt::Debug for FakeEnv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FakeEnv")
            .field("reads", &self.reads)
            .finish_non_exhaustive()
    }
}

impl FakeEnv {
    fn new<const N: usize>(values: [(&str, &[u8]); N]) -> Self {
        Self {
            reads: RefCell::new(Vec::new()),
            values: values
                .into_iter()
                .map(|(name, value)| (name.to_owned(), value.to_vec()))
                .collect(),
        }
    }

    fn reads(&self) -> Vec<String> {
        self.reads.borrow().clone()
    }
}

impl EnvReader for FakeEnv {
    fn read(&self, name: &'static str) -> Option<Vec<u8>> {
        self.reads.borrow_mut().push(name.to_owned());
        self.values.get(name).cloned()
    }
}

struct LocalProvider {
    addr: SocketAddr,
    request: mpsc::Receiver<String>,
    thread: Option<thread::JoinHandle<()>>,
}

impl LocalProvider {
    fn start(body_text: &'static str) -> Self {
        Self::start_with_request_id_option(body_text, None)
    }

    fn start_with_response_model(body_text: &'static str, response_model: &'static str) -> Self {
        Self::start_with_options(body_text, None, response_model)
    }

    fn start_with_request_id(body_text: &'static str, request_id: &'static str) -> Self {
        Self::start_with_request_id_option(body_text, Some(request_id))
    }

    fn start_with_request_id_option(body_text: &'static str, request_id: Option<&'static str>) -> Self {
        Self::start_with_options(body_text, request_id, MODEL)
    }

    fn start_with_options(
        body_text: &'static str,
        request_id: Option<&'static str>,
        response_model: &'static str,
    ) -> Self {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let addr = listener.local_addr().unwrap();
        let (request_sender, request) = mpsc::channel();
        let thread = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let request = read_http_request(&mut stream);
            request_sender.send(request).unwrap();
            let body = json!({
                "id": "chatcmpl-cli",
                "object": "chat.completion",
                "created": 1_700_000_000_u64,
                "model": response_model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": body_text},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
            })
            .to_string();
            let reflected_header =
                request_id.map_or_else(String::new, |request_id| format!("request-id: {request_id}\r\n"));
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n{reflected_header}content-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len(),
            );
            stream.write_all(response.as_bytes()).unwrap();
        });
        Self {
            addr,
            request,
            thread: Some(thread),
        }
    }

    fn base_url(&self) -> String {
        format!("http://{}", self.addr)
    }

    fn finish(mut self) -> String {
        let request = self.request.recv_timeout(Duration::from_secs(10)).unwrap();
        self.thread.take().unwrap().join().unwrap();
        request
    }
}

impl Drop for LocalProvider {
    fn drop(&mut self) {
        if let Some(thread) = self.thread.take() {
            drop(TcpStream::connect(self.addr));
            drop(thread.join());
        }
    }
}

fn read_http_request(stream: &mut TcpStream) -> String {
    stream.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    let mut bytes = Vec::new();
    let mut buffer = [0_u8; 4096];
    let mut expected = None;
    loop {
        let count = stream.read(&mut buffer).unwrap();
        if count == 0 {
            break;
        }
        bytes.extend_from_slice(&buffer[..count]);
        if expected.is_none()
            && let Some(headers_end) = bytes.windows(4).position(|window| window == b"\r\n\r\n")
        {
            let headers = String::from_utf8_lossy(&bytes[..headers_end]);
            let content_length = headers
                .lines()
                .find_map(|line| {
                    line.to_ascii_lowercase()
                        .strip_prefix("content-length: ")
                        .map(str::to_owned)
                })
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or_default();
            expected = Some(headers_end + 4 + content_length);
        }
        if expected.is_some_and(|expected| bytes.len() >= expected) {
            break;
        }
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

fn scenario_root(prompt: &str) -> TempDir {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("scenarios/messages/basic-nonstream.yaml");
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    let scenario = scenario(prompt);
    fs::write(path, serde_yaml::to_string(&scenario).unwrap()).unwrap();
    root
}

fn scenario(prompt: &str) -> InferenceScenario {
    InferenceScenario {
        version: 1,
        id: "messages/basic-nonstream".to_owned(),
        description: "xtask local provider scenario".to_owned(),
        protocol: InferenceProtocol::AnthropicMessages,
        example_config: "anthropic/messages-to-openai.yaml".to_owned(),
        upstream_authority: "127.0.0.1:8000".to_owned(),
        features: vec!["messages.basic".to_owned()],
        turns: vec![ScenarioTurn {
            name: "initial".to_owned(),
            request: RecordedRequest {
                method: "POST".to_owned(),
                path: "/v1/messages".to_owned(),
                headers: BTreeMap::from([("content-type".to_owned(), vec!["application/json".to_owned()])]),
                body: RecordedBody::Json {
                    value: json!({
                        "model": "${MODEL}",
                        "max_tokens": 64,
                        "stream": false,
                        "messages": [{"role": "user", "content": prompt}]
                    }),
                },
            },
            expect: ScenarioExpectation {
                client_status: 200,
                client_body_kind: BodyKind::Json,
                upstream_path: "/v1/chat/completions".to_owned(),
                upstream_body_kind: BodyKind::Json,
                client_sse_events: Vec::new(),
                client_sse_repeatable_events: Vec::new(),
                client_sse_interleaved_events: Vec::new(),
                upstream_sse_events: Vec::new(),
                upstream_sse_repeatable_events: Vec::new(),
                upstream_sse_interleaved_events: Vec::new(),
            },
        }],
    }
}

fn record_args(
    root: &Path,
    provider: &str,
    provider_base_url: impl Into<String>,
    out: Option<PathBuf>,
    redactions_file: Option<PathBuf>,
) -> RecordArgs {
    RecordArgs {
        scenario: "messages/basic-nonstream".to_owned(),
        provider: provider.to_owned(),
        provider_base_url: Some(provider_base_url.into()),
        model: MODEL.to_owned(),
        root: root.to_path_buf(),
        out,
        redactions_file,
    }
}

fn import_args(
    root: &Path,
    recording: PathBuf,
    provider: &str,
    out: PathBuf,
    redactions_file: Option<PathBuf>,
) -> ImportArgs {
    ImportArgs {
        recording,
        scenario: "messages/basic-nonstream".to_owned(),
        provider: provider.to_owned(),
        controlled_synthetic: false,
        root: root.to_path_buf(),
        out: Some(out),
        redactions_file,
    }
}

fn import_default_args(root: &Path, recording: PathBuf, provider: &str) -> ImportArgs {
    ImportArgs {
        recording,
        scenario: "messages/basic-nonstream".to_owned(),
        provider: provider.to_owned(),
        controlled_synthetic: false,
        root: root.to_path_buf(),
        out: None,
        redactions_file: None,
    }
}

fn write_external_recording(path: &Path, max_tokens: u64, model: &str, source_id: &str) {
    let document = json!({
        "test_id": source_id,
        "request": {
            "method": "POST",
            "url": "http://provider.invalid/v1/chat/completions",
            "endpoint": "/v1/chat/completions",
            "headers": {"content-type": "application/json"},
            "model": model,
            "body": {
                "model": model,
                "max_completion_tokens": max_tokens,
                "stream": false,
                "messages": [{"role": "user", "content": "ordinary prompt"}]
            }
        },
        "response": {
            "body": {
                "id": "chatcmpl-import",
                "object": "chat.completion",
                "created": 1_700_000_000_u64,
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": PROVIDER_BODY_SENTINEL},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
            },
            "is_streaming": false
        }
    });
    fs::write(path, document.to_string()).unwrap();
}

fn coverage_root() -> TempDir {
    let root = scenario_root(SECRET_SENTINEL);
    fs::write(
            root.path().join("coverage.yaml"),
            "version: 1\nscope: [messages_to_chat_completions]\nfeatures:\n  - id: messages.basic\n    scopes: [messages_to_chat_completions]\n    status: covered\n    scenarios: [messages/basic-nonstream]\n    providers:\n      openai:\n        status: covered\n        reason: null\n    reason: null\n",
        )
        .unwrap();
    let path = root.path().join("recordings/openai/messages/basic-nonstream.json");
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    let fixture = WireFixture {
        version: 1,
        scenario_id: "messages/basic-nonstream".to_owned(),
        protocol: InferenceProtocol::AnthropicMessages,
        provenance: FixtureProvenance {
            kind: ProvenanceKind::Imported,
            provider: "openai".to_owned(),
            model: MODEL.to_owned(),
            source_id: Some("coverage-source".to_owned()),
        },
        normalization: NormalizationMetadata {
            version: 1,
            linked_ids: BTreeMap::new(),
        },
        turns: vec![WireTurn {
            name: "secret turn".to_owned(),
            client: RecordedExchange {
                request: RecordedRequest {
                    method: "POST".to_owned(),
                    path: "/v1/messages".to_owned(),
                    headers: BTreeMap::new(),
                    body: RecordedBody::Json {
                        value: json!({"secret": SECRET_SENTINEL}),
                    },
                },
                response: RecordedResponse {
                    status: 200,
                    headers: BTreeMap::new(),
                    body: RecordedBody::Json {
                        value: json!({"body": PROVIDER_BODY_SENTINEL}),
                    },
                },
            },
            upstream: empty_exchange(),
        }],
    };
    fs::write(path, serde_json::to_vec_pretty(&fixture).unwrap()).unwrap();
    root
}

fn coverage_recording_path(root: &Path) -> PathBuf {
    root.join("recordings/openai/messages/basic-nonstream.json")
}

fn mutate_text_file(path: &Path, mutate: impl FnOnce(&mut String)) {
    let mut document = fs::read_to_string(path).expect("raw fixture document should be readable");
    mutate(&mut document);
    fs::write(path, document).expect("raw fixture document should be writable");
}

fn replace_once(document: &mut String, needle: &str, replacement: &str) {
    let Some(offset) = document.find(needle) else {
        panic!("raw fixture mutation anchor must exist");
    };
    document.replace_range(offset..offset + needle.len(), replacement);
}

fn opaque_check_error(root: &Path, protected: &[&str]) -> String {
    let mut stdout = Vec::new();
    let error = run_check_with(
        &CheckArgs {
            root: root.to_path_buf(),
        },
        &mut stdout,
    )
    .expect_err("invalid persisted fixture trees must fail checking");
    let surfaces = error_surfaces(&error);

    assert!(stdout.is_empty());
    for value in protected {
        assert!(!surfaces.contains(value));
    }
    error
}

fn empty_exchange() -> RecordedExchange {
    RecordedExchange {
        request: RecordedRequest {
            method: "POST".to_owned(),
            path: "/v1/chat/completions".to_owned(),
            headers: BTreeMap::new(),
            body: RecordedBody::Empty,
        },
        response: RecordedResponse {
            status: 200,
            headers: BTreeMap::new(),
            body: RecordedBody::Empty,
        },
    }
}

fn write_oversized(path: &Path, limit: usize, sentinel: &[u8]) {
    let mut file = File::create(path).unwrap();
    file.write_all(&vec![b'x'; limit]).unwrap();
    file.write_all(sentinel).unwrap();
}

fn assert_secret_absent(stdout: &[u8], out: &Path, secret: &str) {
    assert!(!String::from_utf8_lossy(stdout).contains(secret));
    assert!(!fs::read_to_string(out).unwrap().contains(secret));
}

fn error_surfaces(error: &str) -> String {
    let wrapped = std::io::Error::other(error.to_owned());
    let mut surfaces = format!("{wrapped}\n{wrapped:?}");
    let mut source = wrapped.source();
    while let Some(error) = source {
        write!(&mut surfaces, "\n{error}\n{error:?}").unwrap();
        source = error.source();
    }
    surfaces
}
