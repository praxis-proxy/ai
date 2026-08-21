// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Drift check between the runtime Responses registry and the pinned spec.
//!
//! The registry is the runtime source of truth for Responses operation
//! identity. This check fails when a registered operation's method, path, or
//! operation ID no longer agrees with the pinned OpenAI specification, and when
//! an operation declared as a Praxis protocol extension turns out to exist in
//! the specification after all.

use super::{
    area::{OPENAI_REFERENCE_MANIFEST, OPENAI_REFERENCE_SPEC},
    model::OperationScope,
    spec::{load_reference_source, project_reference},
};

/// Responses operations selected from the pinned specification.
const RESPONSES_SCOPE: OperationScope = OperationScope::new("responses", "Responses", &["/responses"]);

/// Outcome of comparing one registered operation with the pinned specification.
enum Comparison {
    /// Operation agrees with the specification, or is an expected extension.
    Agrees,
    /// Operation disagrees; carries the human-readable reason.
    Drifted(String),
}

/// Compare one registered operation with the pinned specification.
fn compare(
    spec: &praxis_ai_apis::openai::ResponsesOperationSpec,
    found: Option<&str>,
    is_extension: bool,
) -> Comparison {
    let method = spec.method.as_str();
    if is_extension {
        return if found == Some(spec.operation_id) {
            Comparison::Drifted(format!(
                "{method} {} is declared a Praxis protocol extension but the pinned specification defines it",
                spec.spec_path
            ))
        } else {
            Comparison::Agrees
        };
    }

    match found {
        Some(operation_id) if operation_id == spec.operation_id => Comparison::Agrees,
        Some(operation_id) => Comparison::Drifted(format!(
            "{method} {} registers operation ID {} but the pinned specification says {operation_id}",
            spec.spec_path, spec.operation_id
        )),
        None => Comparison::Drifted(format!(
            "{method} {} is registered but absent from the pinned specification",
            spec.spec_path
        )),
    }
}

/// Compare the runtime Responses registry against the pinned specification.
pub(super) fn check() -> Result<String, String> {
    let reference = load_reference_source(OPENAI_REFERENCE_SPEC, Some(OPENAI_REFERENCE_MANIFEST))?;
    let operations = project_reference(&reference, RESPONSES_SCOPE)?.operations;

    let mut checked = 0_usize;
    let mut extensions = 0_usize;
    let mut failures = Vec::new();

    for spec in praxis_ai_apis::openai::responses_operation_specs() {
        let is_extension =
            praxis_ai_apis::openai::RESPONSES_PROTOCOL_EXTENSION_OPERATION_IDS.contains(&spec.operation_id);
        if is_extension {
            extensions += 1;
        } else {
            checked += 1;
        }

        let found = operations
            .iter()
            .find(|candidate| candidate.key.method == spec.method.as_str() && candidate.key.path == spec.spec_path)
            .and_then(|candidate| candidate.operation_id.as_deref());

        if let Comparison::Drifted(reason) = compare(spec, found, is_extension) {
            failures.push(reason);
        }
    }

    if failures.is_empty() {
        Ok(format!(
            "responses registry matches the pinned specification: {checked} operations checked, {extensions} protocol extensions"
        ))
    } else {
        Err(failures.join("\n"))
    }
}
