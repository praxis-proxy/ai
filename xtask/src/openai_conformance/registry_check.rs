// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Shared comparison rules for family registry drift checks.
//!
//! Each family supplies its registered operations and the projected
//! specification operations. The comparison itself is family-agnostic: a
//! registered method/path/ID must match the pinned specification, and a Praxis
//! protocol extension must remain absent from that specification.

use std::{collections::BTreeMap, ops::Deref};

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

/// One operation identity: `(METHOD, path)` mapped to its operation ID.
pub(super) type OperationIds = BTreeMap<(String, String), String>;

/// Compare pinned, runtime, and generated operation identities.
///
/// Reports every disagreement rather than stopping at the first:
///
/// - runtime vs pinned: registered but absent, ID mismatch, or pinned but unregistered;
/// - generated vs runtime: missing on either side, or ID mismatch.
pub(super) fn compare_three_way(
    pinned: &OperationIds,
    runtime: &OperationIds,
    generated: &OperationIds,
) -> Vec<String> {
    let mut failures = compare_against(runtime, ("pinned specification", pinned));
    failures.extend(compare_against(runtime, ("generated document", generated)));
    failures
}

/// Compare the runtime registry with one other source, in both directions.
fn compare_against(runtime: &OperationIds, (label, other): (&str, &OperationIds)) -> Vec<String> {
    let mut failures = Vec::new();
    for (key, runtime_id) in runtime {
        let (method, path) = key;
        match other.get(key) {
            Some(other_id) if other_id == runtime_id => {},
            Some(other_id) => failures.push(format!(
                "{method} {path} registers operation ID {runtime_id} but the {label} says {other_id}"
            )),
            None => failures.push(format!("{method} {path} is registered but absent from the {label}")),
        }
    }
    for (method, path) in other.keys().filter(|key| !runtime.contains_key(*key)) {
        failures.push(format!(
            "{method} {path} is in the {label} but has no runtime registry entry"
        ));
    }
    failures
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
    use super::{Comparison, OperationIds, compare, compare_three_way};

    /// Build an identity map from `(method, path, operationId)` triples.
    fn ids(entries: &[(&str, &str, &str)]) -> OperationIds {
        entries
            .iter()
            .map(|(method, path, id)| (((*method).to_owned(), (*path).to_owned()), (*id).to_owned()))
            .collect()
    }

    /// Whether any failure contains `needle`.
    fn has(failures: &[String], needle: &str) -> bool {
        failures.iter().any(|failure| failure.contains(needle))
    }

    /// Two operations shared by every source in the agreeing baseline.
    fn baseline() -> OperationIds {
        ids(&[("GET", "/files", "listFiles"), ("POST", "/files", "createFile")])
    }

    #[test]
    fn three_way_all_sources_agree() {
        assert!(compare_three_way(&baseline(), &baseline(), &baseline()).is_empty());
    }

    #[test]
    fn three_way_pinned_only_operation_is_reported() {
        let mut pinned = baseline();
        pinned.extend(ids(&[("DELETE", "/files/{file_id}", "deleteFile")]));
        let failures = compare_three_way(&pinned, &baseline(), &baseline());
        assert_eq!(failures.len(), 1, "{failures:?}");
        assert!(has(&failures, "DELETE /files/{file_id}") && has(&failures, "no runtime registry entry"));
    }

    #[test]
    fn three_way_runtime_only_operation_is_reported_against_both() {
        let mut runtime = baseline();
        runtime.extend(ids(&[("PUT", "/files", "putFiles")]));
        let failures = compare_three_way(&baseline(), &runtime, &baseline());
        assert_eq!(failures.len(), 2, "{failures:?}");
        assert!(
            failures
                .iter()
                .any(|f| f.contains("absent from the pinned specification"))
        );
        assert!(
            failures
                .iter()
                .any(|f| f.contains("absent from the generated document"))
        );
    }

    #[test]
    fn three_way_generated_only_operation_is_reported() {
        let mut generated = baseline();
        generated.extend(ids(&[("PUT", "/files", "putFiles")]));
        let failures = compare_three_way(&baseline(), &baseline(), &generated);
        assert_eq!(failures.len(), 1, "{failures:?}");
        assert!(has(
            &failures,
            "in the generated document but has no runtime registry entry"
        ));
    }

    #[test]
    fn three_way_generated_missing_operation_is_reported() {
        let generated = ids(&[("GET", "/files", "listFiles")]);
        let failures = compare_three_way(&baseline(), &baseline(), &generated);
        assert_eq!(failures.len(), 1, "{failures:?}");
        assert!(has(&failures, "POST /files") && has(&failures, "absent from the generated document"));
    }

    #[test]
    fn three_way_id_mismatch_is_reported_per_pair() {
        let wrong = ids(&[("GET", "/files", "listFiles"), ("POST", "/files", "uploadFile")]);

        let failures = compare_three_way(&wrong, &baseline(), &baseline());
        assert_eq!(failures.len(), 1, "{failures:?}");
        assert!(has(&failures, "pinned specification says uploadFile"));

        let failures = compare_three_way(&baseline(), &baseline(), &wrong);
        assert_eq!(failures.len(), 1, "{failures:?}");
        assert!(has(&failures, "generated document says uploadFile"));

        let failures = compare_three_way(&baseline(), &wrong, &baseline());
        assert_eq!(failures.len(), 2, "{failures:?}");
    }

    #[test]
    fn three_way_reports_multiple_failures_together() {
        let pinned = ids(&[
            ("GET", "/files", "listFiles"),
            ("DELETE", "/files/{file_id}", "deleteFile"),
        ]);
        let generated = ids(&[("POST", "/files", "createFile")]);
        let failures = compare_three_way(&pinned, &baseline(), &generated);
        assert_eq!(failures.len(), 3, "every disagreement is reported: {failures:?}");
    }

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
