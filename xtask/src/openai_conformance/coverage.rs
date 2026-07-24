// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

use std::collections::{BTreeMap, BTreeSet};

use super::model::{
    CoverageReport, CoveredOperation, IgnoredCounts, OperationKey, ReferenceSourceReport, SpecOperation,
    SpecSourceReport, SupportedOperation,
};

/// Calculate coverage for extracted spec operations.
#[expect(clippy::too_many_lines, reason = "straight-line report classification")]
#[expect(clippy::too_many_arguments, reason = "explicit report inputs")]
pub(super) fn calculate_coverage(
    reference: ReferenceSourceReport,
    sources: Vec<SpecSourceReport>,
    operations: Vec<SpecOperation>,
    supported_operations: &[SupportedOperation],
    include_deprecated: bool,
    include_beta: bool,
) -> CoverageReport {
    let supported = supported_map(supported_operations);
    let mut considered = Vec::new();
    let mut ignored = IgnoredCounts::default();
    let mut covered = Vec::new();
    let mut missing = Vec::new();
    let mut seen_claims = BTreeSet::new();

    for operation in operations {
        if operation.deprecated && !include_deprecated {
            ignored.deprecated += 1;
            continue;
        }
        if operation.beta && !include_beta {
            ignored.beta += 1;
            continue;
        }

        if let Some(support) = supported.get(&operation.key).cloned() {
            seen_claims.insert(operation.key.clone());
            covered.push(CoveredOperation {
                operation: operation.clone(),
                support,
            });
        } else {
            missing.push(operation.clone());
        }
        considered.push(operation);
    }

    let stale_claims = supported_operations
        .iter()
        .filter(|support| !seen_claims.contains(&support.key()))
        .cloned()
        .collect();

    CoverageReport {
        reference,
        sources,
        considered,
        ignored,
        covered,
        missing,
        stale_claims,
        oasdiff: None,
        runtime_verifications: Vec::new(),
    }
}

/// Build the support lookup map.
fn supported_map(supported_operations: &[SupportedOperation]) -> BTreeMap<OperationKey, SupportedOperation> {
    supported_operations
        .iter()
        .cloned()
        .map(|operation| (operation.key(), operation))
        .collect()
}
