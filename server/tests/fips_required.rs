// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

#![forbid(unsafe_code)]

//! `PRAXIS_REQUIRE_FIPS` must fail closed.
//!
//! Runs the real binary as a subprocess because the check ends in
//! `process::exit`. The expectation depends on the host: on a host that is
//! not in FIPS mode the binary must refuse to start and say why; on a FIPS
//! host it must start normally, unless it carries the policy engine or the
//! response store, whose dependencies do their own cryptography, in which
//! case it must refuse and name the filters. Every branch is asserted, so
//! the test is meaningful wherever it runs, and a control run without the
//! variable proves the variable is what changes the outcome.

use std::process::Command;

/// Whether both FIPS signals the binary checks are present: the kernel flag
/// and an OpenSSL provider that reports FIPS-approved algorithms.
fn host_is_fips() -> bool {
    praxis_tls::provider::install();
    praxis_tls::provider::status().unmet().is_empty()
}

/// Run `praxis-ai --validate` on the built-in default config with the given
/// environment, returning (success, stderr).
#[expect(
    clippy::expect_used,
    reason = "a test helper; a binary that cannot run is a test failure"
)]
fn validate_with(env: &[(&str, &str)]) -> (bool, String) {
    let mut command = Command::new(env!("CARGO_BIN_EXE_praxis-ai"));
    command.arg("--validate");
    command.env_remove("PRAXIS_REQUIRE_FIPS");
    command.env_remove("PRAXIS_CONFIG");
    for (key, value) in env {
        command.env(key, value);
    }
    let output = command.output().expect("the praxis-ai binary must run");
    (
        output.status.success(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

/// On a FIPS host both signals are present, so validate succeeds unless the
/// binary registers a filter whose dependencies carry their own
/// cryptography, which must be refused by name.
fn assert_fips_host_outcome(ok: bool, stderr: &str) {
    if cfg!(any(feature = "policy-engine", feature = "store")) {
        assert!(
            !ok,
            "on a FIPS host a binary carrying non-FIPS filters must refuse to start"
        );
        assert!(
            stderr.contains("PRAXIS_REQUIRE_FIPS"),
            "the refusal must name the variable, got: {stderr}"
        );
        if cfg!(feature = "policy-engine") {
            assert!(
                stderr.contains("`policy` filter"),
                "the refusal must name the policy filter, got: {stderr}"
            );
        }
        if cfg!(feature = "store") {
            assert!(
                stderr.contains("`openai_response_store` filter"),
                "the refusal must name the store filter, got: {stderr}"
            );
        }
    } else {
        assert!(
            ok,
            "on a FIPS host the requirement is met and validate must succeed: {stderr}"
        );
    }
}

#[test]
#[expect(
    clippy::tests_outside_test_module,
    reason = "integration tests are in tests/ directory, not in src"
)]
fn require_fips_fails_closed_unless_the_host_is_in_fips_mode() {
    let (control_ok, control_err) = validate_with(&[]);
    assert!(control_ok, "without the variable, validate must succeed: {control_err}");

    let (ok, stderr) = validate_with(&[("PRAXIS_REQUIRE_FIPS", "1")]);
    if host_is_fips() {
        assert_fips_host_outcome(ok, &stderr);
    } else {
        assert!(
            !ok,
            "on a non-FIPS host the requirement is unmet and the binary must refuse to start"
        );
        assert!(
            stderr.contains("PRAXIS_REQUIRE_FIPS") && stderr.contains("not in effect"),
            "the refusal must name the variable and say FIPS mode is not in effect, got: {stderr}"
        );
        assert!(
            stderr.contains("kernel is not in FIPS mode") || stderr.contains("OpenSSL provider"),
            "the refusal must say which signal is missing, got: {stderr}"
        );
    }
}

#[test]
#[expect(
    clippy::tests_outside_test_module,
    reason = "integration tests are in tests/ directory, not in src"
)]
fn a_false_value_does_not_require_fips() {
    let (ok, stderr) = validate_with(&[("PRAXIS_REQUIRE_FIPS", "false")]);
    assert!(ok, "PRAXIS_REQUIRE_FIPS=false must not require FIPS: {stderr}");
}
