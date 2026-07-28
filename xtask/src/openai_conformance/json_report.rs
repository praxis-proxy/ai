// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

use serde_json::{Value, json};

use super::{
    Args,
    model::{
        AreaStats, CoverageMode, CoverageReport, CoveredOperation, OasdiffAreaDrift, OasdiffAreaReport,
        OasdiffAreaStats, OasdiffOperationDrift, OasdiffReport, OperationKey, RuntimeVerificationArea,
        RuntimeVerificationStatus, SpecOperation, SpecSourceReport, SupportedOperation, percent,
    },
};

/// Write the coverage report as pretty JSON.
pub(super) fn write_json_report(report: &CoverageReport, args: &Args, path: &Path) -> Result<(), String> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("failed to create JSON report directory {}: {e}", parent.display()))?;
    }

    let content = serde_json::to_string_pretty(&report_json(report, args))
        .map_err(|e| format!("failed to serialize JSON report: {e}"))?;
    std::fs::write(path, format!("{content}\n"))
        .map_err(|e| format!("failed to write JSON report {}: {e}", path.display()))
}

/// Build the machine-readable report payload.
#[expect(clippy::too_many_lines, reason = "top-level report schema is intentionally explicit")]
pub(super) fn report_json(report: &CoverageReport, args: &Args) -> Value {
    json!({
        "schema_version": 3,
        "scope": {
            "areas": report_areas(report),
            "include_deprecated": args.include_deprecated,
            "include_beta": args.include_beta,
        },
        "reference": reference_source_json(report),
        "area_projections": report.sources.iter().map(spec_source_json).collect::<Vec<_>>(),
        "capability_coverage": {
            "total_operations": report.considered.len(),
            "covered_operations_count": report.covered.len(),
            "missing_operations_count": report.missing.len(),
            "coverage_percent": report.coverage_percent(),
            "ignored": {
                "deprecated": report.ignored.deprecated,
                "beta": report.ignored.beta,
            },
            "by_area": coverage_by_area_json(report),
            "by_mode": coverage_by_mode_json(report),
            "covered_operations": report.covered.iter().map(covered_operation_json).collect::<Vec<_>>(),
            "missing_operations": report.missing.iter().map(spec_operation_json).collect::<Vec<_>>(),
            "claims_outside_reference": report
                .stale_claims
                .iter()
                .map(supported_operation_json)
                .collect::<Vec<_>>(),
        },
        "owned_contract_conformance": report
            .oasdiff
            .as_ref()
            .map_or_else(oasdiff_not_run_json, oasdiff_report_json),
        "runtime_verification": runtime_verification_json(report),
    })
}

/// Serialize the complete source shared by all selected areas.
fn reference_source_json(report: &CoverageReport) -> Value {
    json!({
        "source": report.reference.source.as_str(),
        "source_sha256": report.reference.source_sha256.as_str(),
        "upstream": report.reference.provenance.as_ref().map(|provenance| json!({
            "repository": provenance.repository.as_str(),
            "revision": provenance.revision.as_str(),
            "path": provenance.path.as_str(),
            "source_sha256": provenance.source_sha256.as_str(),
        })),
    })
}

/// Return stable selected area labels.
fn report_areas(report: &CoverageReport) -> Vec<&'static str> {
    report
        .sources
        .iter()
        .map(|source| source.area)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

/// Serialize one deterministic area projection.
fn spec_source_json(source: &SpecSourceReport) -> Value {
    json!({
        "area_id": source.area_id,
        "area": source.area,
        "projection_sha256": source.projection_sha256.as_str(),
        "operations": source.operations,
    })
}

/// Serialize per-area coverage stats.
fn coverage_by_area_json(report: &CoverageReport) -> Value {
    let mut stats: BTreeMap<&str, AreaStats> = BTreeMap::new();
    for operation in &report.considered {
        stats.entry(operation.area).or_default().total += 1;
    }
    for covered in &report.covered {
        stats.entry(covered.operation.area).or_default().covered += 1;
    }

    json!(
        stats
            .into_iter()
            .map(|(area, stat)| {
                json!({
                    "area": area,
                    "total": stat.total,
                    "covered": stat.covered,
                    "missing": stat.total.saturating_sub(stat.covered),
                    "coverage_percent": percent(stat.covered, stat.total),
                })
            })
            .collect::<Vec<_>>()
    )
}

/// Serialize coverage counts by mode.
fn coverage_by_mode_json(report: &CoverageReport) -> Value {
    let mut modes: BTreeMap<&str, usize> = BTreeMap::new();
    for covered in &report.covered {
        *modes.entry(covered.support.mode.as_str()).or_default() += 1;
    }

    json!(
        CoverageMode::ALL
            .into_iter()
            .map(|mode| json!({
                "mode": mode.as_str(),
                "operations": modes.get(mode.as_str()).copied().unwrap_or_default(),
            }))
            .collect::<Vec<_>>()
    )
}

/// Serialize one covered operation.
fn covered_operation_json(covered: &CoveredOperation) -> Value {
    json!({
        "operation": spec_operation_json(&covered.operation),
        "support": supported_operation_json(&covered.support),
    })
}

/// Serialize one operation from a spec.
fn spec_operation_json(operation: &SpecOperation) -> Value {
    json!({
        "method": operation.key.method.as_str(),
        "path": operation.key.path.as_str(),
        "operation": operation.key.to_string(),
        "area": operation.area,
        "tag": operation.tag.as_str(),
        "operation_id": operation.operation_id.as_deref(),
        "deprecated": operation.deprecated,
        "beta": operation.beta,
    })
}

/// Serialize one local support claim.
fn supported_operation_json(operation: &SupportedOperation) -> Value {
    json!({
        "method": operation.method.as_str(),
        "path": operation.path.as_str(),
        "operation": format!("{} {}", operation.method, operation.path),
        "area": operation.area.as_str(),
        "mode": operation.mode.as_str(),
        "evidence": operation.evidence.as_str(),
    })
}

/// Serialize the optional `oasdiff` report.
fn oasdiff_report_json(report: &OasdiffReport) -> Value {
    json!({
        "enabled": true,
        "applicable_modes": ["transform", "local"],
        "tool": {
            "name": "oasdiff",
            "version": report.tool_version.as_str(),
            "external_refs": false,
            "flatten_path_parameters": true,
            "excluded_elements": ["description", "examples", "extensions", "summary", "title"],
        },
        "areas": report.area_reports.iter().map(oasdiff_area_report_json).collect::<Vec<_>>(),
        "operation_contracts": {
            "total": report.total,
            "exact": report.conformant,
            "missing": report.missing.len(),
            "drifted": report.drifted.len(),
            "request_drifted": report.request_drift_count(),
            "response_drifted": report.response_drift_count(),
            "other_drifted": report.other_drift_count(),
            "exact_percent": report.conformance_percent(),
            "by_area": report.areas.iter().map(oasdiff_area_json).collect::<Vec<_>>(),
        },
        "area_contracts": report.areas.iter().map(|area| oasdiff_area_contract_json(area, &report.area_drift)).collect::<Vec<_>>(),
        "all_exact": report.all_exact(),
        "fixes_required": oasdiff_fixes_required_json(report),
    })
}

/// Serialize complete owned-contract results for one area.
fn oasdiff_area_report_json(area: &OasdiffAreaReport) -> Value {
    json!({
        "area_id": area.area_id,
        "area": area.area,
        "implementation_source": area.implementation_source.as_str(),
        "operation_contracts": {
            "total": area.total,
            "exact": area.conformant,
            "missing": area.missing.len(),
            "drifted": area.drifted.len(),
            "request_drifted": area.drifted.iter().filter(|drift| drift.has_request_drift()).count(),
            "response_drifted": area.drifted.iter().filter(|drift| drift.has_response_drift()).count(),
            "other_drifted": area.drifted.iter().filter(|drift| drift.has_other_drift()).count(),
            "exact_percent": area.conformance_percent(),
            "missing_operations": area.missing.iter().map(operation_key_json).collect::<Vec<_>>(),
            "drifted_operations": area.drifted.iter().map(oasdiff_operation_drift_json).collect::<Vec<_>>(),
        },
        "inherited_contract": {
            "exact": area.inherited_details.is_empty(),
            "details": &area.inherited_details,
        },
        "all_exact": area.all_exact(),
    })
}

/// Serialize actionable fixes derived from `oasdiff` drift.
fn oasdiff_fixes_required_json(report: &OasdiffReport) -> Value {
    let mut by_operation = Vec::new();
    for operation in &report.missing {
        by_operation.push(json!({
            "operation": operation_key_json(operation),
            "fixes": [
                fix_json(
                    "operation",
                    "missing_operation",
                    "Implement this operation and add it to the implementation OpenAPI spec.",
                    &[],
                ),
            ],
        }));
    }
    by_operation.extend(report.drifted.iter().map(oasdiff_drift_fixes_json));
    let by_area = report.area_drift.iter().map(oasdiff_area_fix_json).collect::<Vec<_>>();

    json!({
        "summary": {
            "operations_to_fix": report.missing.len() + report.drifted.len(),
            "area_contracts_to_fix": report.area_drift.len(),
            "missing_operations": report.missing.len(),
            "request_drift_operations": report.request_drift_count(),
            "response_drift_operations": report.response_drift_count(),
            "other_drift_operations": report.other_drift_count(),
        },
        "by_operation": by_operation,
        "by_area": by_area,
    })
}

/// Serialize one inherited area-level fix.
fn oasdiff_area_fix_json(drift: &OasdiffAreaDrift) -> Value {
    json!({
        "area": drift.area,
        "fixes": [fix_json(
            "area",
            "inherited_contract",
            "Align global or inherited OpenAPI contract fields with the OpenAI spec.",
            &drift.details,
        )],
    })
}

/// Serialize fix entries for one drifted operation.
fn oasdiff_drift_fixes_json(drift: &OasdiffOperationDrift) -> Value {
    let mut fixes = Vec::new();
    fixes.extend(request_fix_jsons(&drift.request_details));
    if !drift.response_details.is_empty() {
        fixes.push(fix_json(
            "response",
            "response_schema",
            "Align the response body schema with the OpenAI spec.",
            &drift.response_details,
        ));
    }
    if !drift.other_details.is_empty() {
        fixes.push(fix_json(
            "operation",
            "operation_contract",
            "Align non-request and non-response operation contract fields with the OpenAI spec.",
            &drift.other_details,
        ));
    }

    json!({
        "operation": operation_key_json(&drift.key),
        "fixes": fixes,
    })
}

/// Build request-side fix entries from categorized detail paths.
fn request_fix_jsons(details: &[String]) -> Vec<Value> {
    let parameter_details = detail_paths_with_prefix(details, "parameters");
    let body_details = detail_paths_with_prefix(details, "requestBody");
    let other_details = details
        .iter()
        .filter(|detail| !detail.starts_with("parameters") && !detail.starts_with("requestBody"))
        .cloned()
        .collect::<Vec<_>>();

    let mut fixes = Vec::new();
    push_fix_if_needed(
        &mut fixes,
        "request",
        "request_parameters",
        "Add or align OpenAI request parameters in the implementation spec and handler behavior.",
        &parameter_details,
    );
    push_fix_if_needed(
        &mut fixes,
        "request",
        "request_body_schema",
        "Align the request body schema with the OpenAI spec.",
        &body_details,
    );
    push_fix_if_needed(
        &mut fixes,
        "request",
        "request_contract",
        "Align request contract details with the OpenAI spec.",
        &other_details,
    );
    fixes
}

/// Return detail paths with the requested prefix.
fn detail_paths_with_prefix(details: &[String], prefix: &str) -> Vec<String> {
    details
        .iter()
        .filter(|detail| detail.starts_with(prefix))
        .cloned()
        .collect()
}

/// Add a fix entry when there are matching detail paths.
fn push_fix_if_needed(fixes: &mut Vec<Value>, area: &str, kind: &str, summary: &str, detail_paths: &[String]) {
    if !detail_paths.is_empty() {
        fixes.push(fix_json(area, kind, summary, detail_paths));
    }
}

/// Serialize one required fix entry.
fn fix_json(area: &str, kind: &str, summary: &str, detail_paths: &[String]) -> Value {
    json!({
        "area": area,
        "kind": kind,
        "summary": summary,
        "detail_paths": detail_paths,
    })
}

/// Serialize the `oasdiff` section when no implementation spec was provided.
fn oasdiff_not_run_json() -> Value {
    json!({
        "enabled": false,
        "reason": "owned-contract comparison was not generated",
    })
}

/// Serialize area-level exactness separately from operation counts.
fn oasdiff_area_contract_json(area: &OasdiffAreaStats, drift: &[OasdiffAreaDrift]) -> Value {
    let details = drift
        .iter()
        .find(|candidate| candidate.area == area.area)
        .map_or(&[][..], |candidate| candidate.details.as_slice());
    json!({
        "area": area.area,
        "exact": details.is_empty(),
        "details": details,
    })
}

/// Serialize one `oasdiff` area row.
fn oasdiff_area_json(area: &OasdiffAreaStats) -> Value {
    json!({
        "area": area.area,
        "total": area.total,
        "conformant": area.conformant(),
        "missing": area.missing,
        "drifted": area.drifted,
        "request_drifted": area.request_drifted,
        "response_drifted": area.response_drifted,
        "other_drifted": area.other_drifted,
        "conformance_percent": percent(area.conformant(), area.total),
    })
}

/// Serialize one `oasdiff` drifted operation.
fn oasdiff_operation_drift_json(drift: &OasdiffOperationDrift) -> Value {
    json!({
        "operation": operation_key_json(&drift.key),
        "has_request_drift": drift.has_request_drift(),
        "has_response_drift": drift.has_response_drift(),
        "has_other_drift": drift.has_other_drift(),
        "request_details": &drift.request_details,
        "response_details": &drift.response_details,
        "other_details": &drift.other_details,
    })
}

/// Serialize one operation key.
fn operation_key_json(key: &OperationKey) -> Value {
    json!({
        "method": key.method.as_str(),
        "path": key.path.as_str(),
        "operation": key.to_string(),
    })
}

/// Serialize runtime checks that were actually executed for this report.
fn runtime_verification_json(report: &CoverageReport) -> Value {
    json!({
        "all_passed": report
            .runtime_verifications
            .iter()
            .all(|verification| verification.status == RuntimeVerificationStatus::Passed),
        "areas": report
            .runtime_verifications
            .iter()
            .map(|verification| runtime_verification_area_json(report, verification))
            .collect::<Vec<_>>(),
    })
}

/// Serialize one area runtime result and its checked capability modes.
fn runtime_verification_area_json(report: &CoverageReport, verification: &RuntimeVerificationArea) -> Value {
    let modes = CoverageMode::ALL
        .into_iter()
        .filter(|mode| {
            report
                .covered
                .iter()
                .any(|covered| covered.operation.area == verification.area && covered.support.mode == *mode)
        })
        .map(CoverageMode::as_str)
        .collect::<Vec<_>>();
    json!({
        "area": verification.area,
        "modes": modes,
        "status": verification.status.as_str(),
        "command": verification.command,
        "checks": verification.checks.iter().map(|check| json!({
            "kind": check.kind,
            "evidence": check.evidence,
        })).collect::<Vec<_>>(),
    })
}
