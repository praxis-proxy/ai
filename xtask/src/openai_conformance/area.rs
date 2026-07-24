// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

use super::model::{CoverageMode, OperationScope, RuntimeVerificationCheck, SupportedOperation};

/// Vendored OpenAI `OpenAPI` reference used by default.
pub(super) const OPENAI_REFERENCE_SPEC: &str = "docs/conformance/specs/openai-openapi.yaml";
/// Provenance pin for the vendored complete OpenAI reference.
pub(super) const OPENAI_REFERENCE_MANIFEST: &str = "docs/conformance/specs/openai-openapi-source.json";

/// Conversations operations selected from the full OpenAI reference.
pub(super) const CONVERSATIONS_SCOPE: OperationScope =
    OperationScope::new("conversations", "Conversations", &["/conversations"]);

/// Runtime checks executed while generating the conformance report.
const CONVERSATIONS_RUNTIME_CHECKS: &[RuntimeVerificationCheck] = &[
    RuntimeVerificationCheck {
        kind: "route_dispatch",
        evidence: "openai::conversations::tests::conformance_conversations_routes_match_runtime_registry",
        success_sentinel: "PRAXIS_CONFORMANCE_OK conversations route_dispatch",
    },
    RuntimeVerificationCheck {
        kind: "success_response_contract",
        evidence: "openai::conversations::tests::conformance_conversations_success_payloads_match_generated_response_schemas",
        success_sentinel: "PRAXIS_CONFORMANCE_OK conversations success_response_contract",
    },
    RuntimeVerificationCheck {
        kind: "schema_check_sensitivity",
        evidence: "openai::conversations::tests::conformance_conversations_generated_schema_check_rejects_wrong_discriminator",
        success_sentinel: "PRAXIS_CONFORMANCE_OK conversations schema_check_sensitivity",
    },
];

/// One conformance area checked by `cargo xtask openai-conformance`.
pub(super) struct ApiArea {
    /// Operation scope loaded from the reference spec.
    pub(super) scope: OperationScope,

    /// Stable label for the generated implementation spec.
    pub(super) implementation_source: &'static str,

    /// Generate the implementation `OpenAPI` document for this area.
    pub(super) implementation_spec: fn() -> Result<String, String>,

    /// Return operations implemented locally for this area.
    pub(super) supported_operations: fn() -> Vec<SupportedOperation>,

    /// Focused test command run by the conformance task.
    pub(super) runtime_test_command: &'static str,

    /// Arguments passed to Cargo for the focused runtime checks.
    pub(super) runtime_test_args: &'static [&'static str],

    /// Runtime checks selected by the focused command.
    pub(super) runtime_checks: &'static [RuntimeVerificationCheck],
}

/// Areas included in the current OpenAI conformance suite.
pub(super) const CONFORMANCE_AREAS: &[ApiArea] = &[ApiArea {
    scope: CONVERSATIONS_SCOPE,
    implementation_source: "generated:praxis-ai-apis/openai/conversations",
    implementation_spec: conversations_implementation_spec,
    supported_operations: conversations_supported_operations,
    runtime_test_command: "cargo test -p praxis-ai-apis --lib conformance_conversations_ -- --show-output",
    runtime_test_args: &[
        "test",
        "-p",
        "praxis-ai-apis",
        "--lib",
        "conformance_conversations_",
        "--",
        "--show-output",
    ],
    runtime_checks: CONVERSATIONS_RUNTIME_CHECKS,
}];

/// Generate the Conversations implementation spec from crate code.
fn conversations_implementation_spec() -> Result<String, String> {
    praxis_ai_apis::openai::conversations_openapi_json()
        .map_err(|e| format!("failed to generate Conversations implementation OpenAPI spec: {e}"))
}

/// Return Conversations operations from the runtime route table.
fn conversations_supported_operations() -> Vec<SupportedOperation> {
    praxis_ai_apis::openai::conversations_operation_specs()
        .iter()
        .map(|spec| SupportedOperation {
            method: spec.method.to_owned(),
            path: spec.spec_path.to_owned(),
            area: "Conversations".to_owned(),
            mode: coverage_mode(spec.mode),
            evidence: format!(
                "praxis_ai_apis::openai::conversations_operation_specs::{:?}",
                spec.operation
            ),
        })
        .collect()
}

/// Convert shared runtime handling metadata into the report model.
const fn coverage_mode(mode: praxis_ai_apis::openai::OpenAiHandlingMode) -> CoverageMode {
    match mode {
        praxis_ai_apis::openai::OpenAiHandlingMode::Passthrough => CoverageMode::Passthrough,
        praxis_ai_apis::openai::OpenAiHandlingMode::Inspect => CoverageMode::Inspect,
        praxis_ai_apis::openai::OpenAiHandlingMode::Transform => CoverageMode::Transform,
        praxis_ai_apis::openai::OpenAiHandlingMode::Local => CoverageMode::Local,
    }
}
