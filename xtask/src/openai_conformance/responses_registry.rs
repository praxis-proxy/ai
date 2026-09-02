// SPDX-License-Identifier: Apache-2.0
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
///
/// Takes the identifying fields rather than a spec so the comparison rules can
/// be unit tested without constructing a registry entry.
fn compare(method: &str, spec_path: &str, registered_id: &str, found: Option<&str>, is_extension: bool) -> Comparison {
    if is_extension {
        // A protocol extension exists precisely because the specification does
        // not define the operation. Any upstream definition at this
        // method/path is drift, whatever ID upstream chose — checking only for
        // our own ID would never fire, since upstream would not pick it.
        return match found {
            Some(operation_id) => Comparison::Drifted(format!(
                "{method} {spec_path} is declared a Praxis protocol extension but the pinned \
                 specification now defines it as {operation_id}"
            )),
            None => Comparison::Agrees,
        };
    }

    match found {
        Some(operation_id) if operation_id == registered_id => Comparison::Agrees,
        Some(operation_id) => Comparison::Drifted(format!(
            "{method} {spec_path} registers operation ID {registered_id} but the pinned specification says {operation_id}"
        )),
        None => Comparison::Drifted(format!(
            "{method} {spec_path} is registered but absent from the pinned specification"
        )),
    }
}

/// Counts and failures accumulated over the registry.
struct Tally {
    /// Specification-owned operations compared.
    checked: usize,
    /// Praxis protocol extensions seen.
    extensions: usize,
    /// Human-readable drift reasons.
    failures: Vec<String>,
}

/// Compare every registered operation against the projected specification.
fn tally(operations: &[super::model::SpecOperation]) -> Tally {
    let mut tally = Tally {
        checked: 0,
        extensions: 0,
        failures: Vec::new(),
    };

    for spec in praxis_ai_apis::openai::responses_operation_specs() {
        let is_extension =
            praxis_ai_apis::openai::RESPONSES_PROTOCOL_EXTENSION_OPERATION_IDS.contains(&spec.operation_id);
        if is_extension {
            tally.extensions += 1;
        } else {
            tally.checked += 1;
        }

        let found = operations
            .iter()
            .find(|candidate| candidate.key.method == spec.method.as_str() && candidate.key.path == spec.spec_path)
            .and_then(|candidate| candidate.operation_id.as_deref());

        if let Comparison::Drifted(reason) = compare(
            spec.method.as_str(),
            spec.spec_path,
            spec.operation_id,
            found,
            is_extension,
        ) {
            tally.failures.push(reason);
        }
    }

    tally
}

/// Compare the runtime Responses registry against the pinned specification.
pub(super) fn check() -> Result<String, String> {
    let reference = load_reference_source(OPENAI_REFERENCE_SPEC, Some(OPENAI_REFERENCE_MANIFEST))?;
    let operations = project_reference(&reference, RESPONSES_SCOPE)?.operations;
    let Tally {
        checked,
        extensions,
        failures,
    } = tally(&operations);

    if failures.is_empty() {
        Ok(format!(
            "responses registry matches the pinned specification: {checked} operations checked, {extensions} protocol extensions"
        ))
    } else {
        Err(failures.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::{Comparison, compare};

    /// Return the drift reason, or `None` when the comparison agrees.
    fn drift(comparison: Comparison) -> Option<String> {
        match comparison {
            Comparison::Agrees => None,
            Comparison::Drifted(reason) => Some(reason),
        }
    }

    #[test]
    fn extension_absent_from_the_specification_agrees() {
        assert!(
            drift(compare(
                "GET",
                "/responses",
                "praxis_createResponseWebSocket",
                None,
                true
            ))
            .is_none(),
            "an extension the specification does not define is the expected state"
        );
    }

    #[test]
    fn extension_defined_upstream_under_any_id_is_drift() {
        // The realistic case: upstream adds the operation under its own name.
        let reason = drift(compare(
            "GET",
            "/responses",
            "praxis_createResponseWebSocket",
            Some("createResponseWebSocket"),
            true,
        ))
        .expect("an upstream definition must be reported as drift");
        assert!(
            reason.contains("createResponseWebSocket"),
            "reason should name the upstream ID"
        );

        // The degenerate case: upstream happens to use our own ID.
        assert!(
            drift(compare(
                "GET",
                "/responses",
                "praxis_createResponseWebSocket",
                Some("praxis_createResponseWebSocket"),
                true
            ))
            .is_some(),
            "an upstream definition is drift even when the IDs coincide"
        );
    }

    #[test]
    fn registered_operation_matching_the_specification_agrees() {
        assert!(
            drift(compare(
                "POST",
                "/responses",
                "createResponse",
                Some("createResponse"),
                false
            ))
            .is_none()
        );
    }

    #[test]
    fn registered_operation_with_a_different_id_is_drift() {
        let reason = drift(compare(
            "POST",
            "/responses",
            "createResponseTypo",
            Some("createResponse"),
            false,
        ))
        .expect("a mismatched ID must be reported");
        assert!(reason.contains("createResponseTypo") && reason.contains("createResponse"));
    }

    #[test]
    fn registered_operation_absent_from_the_specification_is_drift() {
        assert!(drift(compare("POST", "/responses/invented", "inventedOperation", None, false)).is_some());
    }
}
