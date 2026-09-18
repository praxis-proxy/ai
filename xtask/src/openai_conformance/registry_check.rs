// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Shared comparison rules for family registry drift checks.
//!
//! Each family supplies its registered operations and the projected
//! specification operations. The comparison itself is family-agnostic: a
//! registered method/path/ID must match the pinned specification, and a Praxis
//! protocol extension must remain absent from that specification.

use std::ops::Deref;

use praxis_ai_apis::openai::OpenAiOperationSpec;

use super::model::SpecOperation;

/// Outcome of comparing one registered operation with the pinned specification.
pub(super) enum Comparison {
    /// Operation agrees with the specification, or is an expected extension.
    Agrees,
    /// Operation disagrees; carries the human-readable reason.
    Drifted(String),
}

/// Compare one registered operation with the pinned specification.
///
/// Takes the identifying fields rather than a spec so the comparison rules can
/// be unit tested without constructing a registry entry.
pub(super) fn compare(
    method: &str,
    spec_path: &str,
    registered_id: &str,
    found: Option<&str>,
    is_extension: bool,
) -> Comparison {
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

/// Counts and failures accumulated over a registry.
pub(super) struct Tally {
    /// Specification-owned operations compared.
    pub(super) checked: usize,
    /// Praxis protocol extensions seen.
    pub(super) extensions: usize,
    /// Human-readable drift reasons.
    pub(super) failures: Vec<String>,
}

/// Compare every registered operation against the projected specification.
pub(super) fn tally<'a, T>(
    specs: impl IntoIterator<Item = &'a T>,
    extension_ids: &[&str],
    operations: &[SpecOperation],
) -> Tally
where
    T: Deref<Target = OpenAiOperationSpec> + 'a,
{
    let mut tally = Tally {
        checked: 0,
        extensions: 0,
        failures: Vec::new(),
    };

    for spec in specs {
        let spec: &OpenAiOperationSpec = spec;
        let is_extension = extension_ids.contains(&spec.operation_id());
        if is_extension {
            tally.extensions += 1;
        } else {
            tally.checked += 1;
        }

        let found = operations
            .iter()
            .find(|candidate| candidate.key.method == spec.method().as_str() && candidate.key.path == spec.spec_path)
            .and_then(|candidate| candidate.operation_id.as_deref());

        if let Comparison::Drifted(reason) = compare(
            spec.method().as_str(),
            spec.spec_path,
            spec.operation_id(),
            found,
            is_extension,
        ) {
            tally.failures.push(reason);
        }
    }

    tally
}

/// Format a successful drift-check summary.
pub(super) fn ok_summary(family: &str, tally: &Tally) -> String {
    format!(
        "{family} registry matches the pinned specification: {} operations checked, {} protocol extensions",
        tally.checked, tally.extensions
    )
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
