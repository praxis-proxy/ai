// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

use std::{
    collections::{BTreeMap, BTreeSet},
    io::Write as _,
    path::Path,
    process::Command,
};

use serde_json::{Map, Value};
use tempfile::{Builder as TempFileBuilder, NamedTempFile};

use super::{
    area::ApiArea,
    model::{
        OasdiffAreaDrift, OasdiffAreaReport, OasdiffAreaStats, OasdiffOperationDrift, OasdiffReport, OperationKey,
        OperationScope, SpecOperation,
    },
    spec::read_spec,
};

/// Exact supported `oasdiff` release.
pub(super) const OASDIFF_VERSION: &str = "1.23.0";
/// Documentation-only elements excluded by `oasdiff` itself, context-aware.
const EXCLUDED_ELEMENTS: &str = "description,examples,extensions,summary,title";

/// Run `oasdiff` against the selected reference spec.
#[expect(clippy::too_many_lines, reason = "area comparison orchestration")]
pub(super) fn run_scoped_oasdiff(
    implementation_source: Option<&str>,
    considered: &[SpecOperation],
    areas: &[&ApiArea],
    reference_projections: &BTreeMap<&'static str, String>,
) -> Result<OasdiffReport, String> {
    let tool_version = checked_oasdiff_version()?;
    let mut missing = BTreeSet::new();
    let mut drifted = BTreeMap::new();
    let mut area_drift = Vec::new();
    let mut area_reports = Vec::new();

    for area in areas {
        let projection = reference_projections
            .get(area.scope.id)
            .ok_or_else(|| format!("missing reference projection for {}", area.scope.label))?;
        let reference_source = format!("generated:openai/{}", area.scope.id);
        let openai_reference = materialize_spec_content(&reference_source, projection, ".yaml")?;
        let implementation = materialize_implementation_spec(implementation_source, area)?;
        let diff = run_oasdiff_diff(openai_reference.path(), implementation.file.path(), area.scope.label)?;
        let mut area_missing = BTreeSet::new();
        let mut area_drifted = BTreeMap::new();
        let mut inherited_details = Vec::new();
        collect_oasdiff_operation_status(
            &diff,
            area.scope,
            considered,
            &mut area_missing,
            &mut area_drifted,
            &mut inherited_details,
        );
        let area_considered = considered
            .iter()
            .filter(|operation| operation.area == area.scope.label)
            .count();
        let area_missing_vec = area_missing.iter().cloned().collect::<Vec<_>>();
        let area_drifted_vec = area_drifted
            .values()
            .filter(|drift| !area_missing.contains(&drift.key))
            .cloned()
            .collect::<Vec<_>>();
        let conformant = area_considered.saturating_sub(area_missing_vec.len() + area_drifted_vec.len());
        area_reports.push(OasdiffAreaReport {
            area_id: area.scope.id,
            area: area.scope.label,
            implementation_source: implementation.source,
            total: area_considered,
            conformant,
            missing: area_missing_vec,
            drifted: area_drifted_vec,
            inherited_details: inherited_details.clone(),
        });
        missing.extend(area_missing);
        for drift in area_drifted.into_values() {
            merge_operation_drift(&mut drifted, drift);
        }
        if !inherited_details.is_empty() {
            area_drift.push(OasdiffAreaDrift {
                area: area.scope.label,
                details: inherited_details,
            });
        }
    }

    Ok(build_oasdiff_report(
        &tool_version,
        considered,
        missing,
        drifted,
        area_drift,
        area_reports,
    ))
}

/// Require a stable `oasdiff` output schema before invoking the tool.
fn checked_oasdiff_version() -> Result<String, String> {
    let output = oasdiff_command()
        .arg("--version")
        .output()
        .map_err(|e| format!("failed to run oasdiff --version: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "oasdiff --version failed: {}",
            String::from_utf8_lossy(&output.stderr).trim(),
        ));
    }
    let actual = parse_oasdiff_version(&String::from_utf8_lossy(&output.stdout))?;
    if actual != OASDIFF_VERSION {
        if actual == "main" {
            eprintln!(
                "warning: oasdiff reports version '{actual}' (go-install build); \
                 expected {OASDIFF_VERSION} — proceeding on the assumption the \
                 correct tag was installed"
            );
            return Ok(OASDIFF_VERSION.to_owned());
        }
        return Err(format!(
            "unsupported oasdiff version {actual}; install exactly {OASDIFF_VERSION}"
        ));
    }
    Ok(actual)
}

/// Parse `oasdiff version X.Y.Z` output.
pub(super) fn parse_oasdiff_version(output: &str) -> Result<String, String> {
    output
        .trim()
        .strip_prefix("oasdiff version ")
        .filter(|version| !version.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| format!("unexpected oasdiff --version output: {output:?}"))
}

/// Materialized spec plus a stable report source label.
struct MaterializedSpec {
    /// User-facing source label.
    source: String,

    /// Temporary spec file consumed by `oasdiff`.
    file: NamedTempFile,
}

/// Materialize the implementation spec for `oasdiff`.
fn materialize_implementation_spec(source: Option<&str>, area: &ApiArea) -> Result<MaterializedSpec, String> {
    if let Some(source) = source {
        return Ok(MaterializedSpec {
            source: source.to_owned(),
            file: materialize_spec(source)?,
        });
    }

    let content = (area.implementation_spec)()?;
    let file = materialize_spec_content(area.implementation_source, &content, ".json")?;
    Ok(MaterializedSpec {
        source: area.implementation_source.to_owned(),
        file,
    })
}

/// Write a URL or local spec source to a temporary file for `oasdiff`.
fn materialize_spec(source: &str) -> Result<NamedTempFile, String> {
    let content = read_spec(source)?;
    let suffix = spec_suffix(source);
    materialize_spec_content(source, &content, suffix)
}

/// Write spec content to a temporary file for `oasdiff`.
fn materialize_spec_content(source: &str, content: &str, suffix: &str) -> Result<NamedTempFile, String> {
    let mut file = TempFileBuilder::new()
        .suffix(suffix)
        .tempfile()
        .map_err(|e| format!("failed to create temporary spec file for {source}: {e}"))?;
    file.write_all(content.as_bytes())
        .map_err(|e| format!("failed to write temporary spec file for {source}: {e}"))?;
    file.flush()
        .map_err(|e| format!("failed to flush temporary spec file for {source}: {e}"))?;
    Ok(file)
}

/// Guess an `OpenAPI` file suffix from a source path.
fn spec_suffix(source: &str) -> &'static str {
    let source = source.split_once('?').map_or(source, |(path, _query)| path);
    if source.ends_with(".json") { ".json" } else { ".yaml" }
}

/// Execute `oasdiff diff` and return its JSON output.
fn run_oasdiff_diff(reference: &Path, implementation: &Path, area: &str) -> Result<Value, String> {
    let output = oasdiff_command()
        .arg("diff")
        .arg(reference)
        .arg(implementation)
        .arg("--format")
        .arg("json")
        .arg("--strip-prefix-revision")
        .arg("/v1")
        .arg("--allow-external-refs=false")
        .arg("--flatten-params")
        .arg("--exclude-elements")
        .arg(EXCLUDED_ELEMENTS)
        .output()
        .map_err(|e| format!("failed to run oasdiff for {area}: {e}"))?;

    if !output.status.success() && output.stdout.is_empty() {
        return Err(format!(
            "oasdiff failed for {area}: {}",
            String::from_utf8_lossy(&output.stderr).trim(),
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stdout = stdout.trim();
    if stdout.is_empty() {
        return Ok(Value::Object(Map::new()));
    }

    serde_json::from_str(stdout).map_err(|e| format!("failed to parse oasdiff JSON output for {area}: {e}"))
}

/// Build an `oasdiff` command, allowing an explicit checked binary path.
fn oasdiff_command() -> Command {
    Command::new(std::env::var_os("OASDIFF_BIN").unwrap_or_else(|| "oasdiff".into()))
}

/// Extract missing and drifted operation keys from `oasdiff` JSON.
#[expect(clippy::too_many_lines, reason = "straight-line traversal of oasdiff JSON")]
#[expect(clippy::too_many_arguments, reason = "separate accumulated status channels")]
pub(super) fn collect_oasdiff_operation_status(
    diff: &Value,
    scope: OperationScope,
    considered: &[SpecOperation],
    missing: &mut BTreeSet<OperationKey>,
    drifted: &mut BTreeMap<OperationKey, OasdiffOperationDrift>,
    area_details: &mut Vec<String>,
) {
    let considered_keys = considered
        .iter()
        .map(|operation| operation.key.clone())
        .collect::<BTreeSet<_>>();

    collect_inherited_area_drift(diff, area_details);

    let Some(paths) = diff.get("paths").and_then(Value::as_object) else {
        return;
    };

    if let Some(deleted_paths) = paths.get("deleted").and_then(Value::as_array) {
        for path in deleted_paths.iter().filter_map(Value::as_str) {
            if !scope.matches(path) {
                continue;
            }
            for operation in considered
                .iter()
                .filter(|operation| operation.key.path.as_str() == path)
            {
                missing.insert(operation.key.clone());
            }
        }
    }

    let Some(modified_paths) = paths.get("modified").and_then(Value::as_object) else {
        return;
    };

    for (path, path_diff) in modified_paths {
        if !scope.matches(path) {
            continue;
        }

        if let Some(parameters) = path_diff.get("parameters") {
            for operation in considered
                .iter()
                .filter(|operation| operation.key.path.as_str() == path.as_str())
            {
                let mut drift = OasdiffOperationDrift::new(operation.key.clone());
                collect_oasdiff_detail_paths("parameters", parameters, &mut drift.request_details);
                if drift.has_request_drift() {
                    merge_operation_drift(drifted, drift);
                }
            }
        }

        let Some(operations) = path_diff.get("operations").and_then(Value::as_object) else {
            continue;
        };

        if let Some(deleted_operations) = operations.get("deleted").and_then(Value::as_array) {
            for method in deleted_operations.iter().filter_map(Value::as_str) {
                insert_if_considered(missing, &considered_keys, method, path);
            }
        }

        if let Some(modified_operations) = operations.get("modified").and_then(Value::as_object) {
            for (method, operation_diff) in modified_operations {
                if operation_diff.as_object().is_some_and(Map::is_empty) {
                    continue;
                }
                let key = OperationKey::new(method.to_ascii_uppercase(), path);
                if !considered_keys.contains(&key) {
                    continue;
                }
                if let Some(drift) = operation_drift_from_diff(key, operation_diff) {
                    merge_operation_drift(drifted, drift);
                }
            }
        }
    }
}

/// Collect global contract drift that cannot be assigned to one operation.
fn collect_inherited_area_drift(diff: &Value, details: &mut Vec<String>) {
    if let Some(security) = diff.get("security") {
        collect_oasdiff_detail_paths("security", security, details);
    }
    if let Some(security_schemes) = diff.pointer("/components/securitySchemes") {
        collect_oasdiff_detail_paths("components.securitySchemes", security_schemes, details);
    }
}

/// Insert an operation key only when it belongs to the selected denominator.
fn insert_if_considered(
    operations: &mut BTreeSet<OperationKey>,
    considered: &BTreeSet<OperationKey>,
    method: &str,
    path: &str,
) {
    let key = OperationKey::new(method.to_ascii_uppercase(), path);
    if considered.contains(&key) {
        operations.insert(key);
    }
}

/// Build categorized drift details for one modified operation.
pub(super) fn operation_drift_from_diff(key: OperationKey, operation_diff: &Value) -> Option<OasdiffOperationDrift> {
    let mut drift = OasdiffOperationDrift::new(key);
    let fields = operation_diff.as_object()?;

    for (field, value) in fields {
        match field.as_str() {
            "parameters" | "requestBody" => collect_oasdiff_detail_paths(field, value, &mut drift.request_details),
            "responses" => collect_oasdiff_detail_paths(field, value, &mut drift.response_details),
            _ => collect_oasdiff_detail_paths(field, value, &mut drift.other_details),
        }
    }

    drift.has_any_drift().then_some(drift)
}

/// Merge a drift record into the accumulated operation map.
fn merge_operation_drift(drifted: &mut BTreeMap<OperationKey, OasdiffOperationDrift>, drift: OasdiffOperationDrift) {
    drifted
        .entry(drift.key.clone())
        .and_modify(|existing| {
            extend_unique(&mut existing.request_details, &drift.request_details);
            extend_unique(&mut existing.response_details, &drift.response_details);
            extend_unique(&mut existing.other_details, &drift.other_details);
        })
        .or_insert(drift);
}

/// Append detail paths that are not already present.
fn extend_unique(target: &mut Vec<String>, source: &[String]) {
    for item in source {
        if !target.contains(item) {
            target.push(item.clone());
        }
    }
}

/// Collect compact `oasdiff` detail paths from a JSON subtree.
fn collect_oasdiff_detail_paths(prefix: &str, value: &Value, details: &mut Vec<String>) {
    match value {
        Value::Object(fields) if fields.is_empty() => push_oasdiff_detail(prefix, details),
        Value::Object(fields) => {
            for (field, child) in fields {
                let path = format!("{prefix}.{field}");
                collect_oasdiff_detail_paths(&path, child, details);
            }
        },
        Value::Array(_) | Value::Bool(_) | Value::Number(_) | Value::String(_) | Value::Null => {
            push_oasdiff_detail(prefix, details);
        },
    }
}

/// Push a normalized `oasdiff` detail path.
fn push_oasdiff_detail(path: &str, details: &mut Vec<String>) {
    let path = path.replace(".modified.", ".");
    if !details.contains(&path) {
        details.push(path);
    }
}

/// Build a sorted `oasdiff` report from extracted status sets.
#[expect(clippy::too_many_arguments, reason = "explicit report inputs")]
pub(super) fn build_oasdiff_report(
    tool_version: &str,
    considered: &[SpecOperation],
    missing: BTreeSet<OperationKey>,
    drifted: BTreeMap<OperationKey, OasdiffOperationDrift>,
    area_drift: Vec<OasdiffAreaDrift>,
    area_reports: Vec<OasdiffAreaReport>,
) -> OasdiffReport {
    let missing_vec = missing.into_iter().collect::<Vec<_>>();
    let drifted_vec = drifted
        .into_iter()
        .filter(|(operation, _drift)| !missing_vec.contains(operation))
        .map(|(_operation, drift)| drift)
        .collect::<Vec<_>>();
    let conformant = considered.len().saturating_sub(missing_vec.len() + drifted_vec.len());
    let areas = build_oasdiff_area_stats(considered, &missing_vec, &drifted_vec);

    OasdiffReport {
        tool_version: tool_version.to_owned(),
        total: considered.len(),
        conformant,
        missing: missing_vec,
        drifted: drifted_vec,
        areas,
        area_reports,
        area_drift,
    }
}

/// Build per-area `oasdiff` status counts.
fn build_oasdiff_area_stats(
    considered: &[SpecOperation],
    missing: &[OperationKey],
    drifted: &[OasdiffOperationDrift],
) -> Vec<OasdiffAreaStats> {
    let mut areas = BTreeMap::<&'static str, OasdiffAreaStats>::new();
    for operation in considered {
        let stat = areas.entry(operation.area).or_insert_with(|| OasdiffAreaStats {
            area: operation.area,
            ..OasdiffAreaStats::default()
        });
        stat.total += 1;
    }

    for operation in missing {
        if let Some(area) = considered.iter().find(|candidate| candidate.key == *operation) {
            areas.entry(area.area).or_default().missing += 1;
        }
    }
    for drift in drifted {
        if let Some(area) = considered.iter().find(|candidate| candidate.key == drift.key) {
            areas.entry(area.area).or_default().drifted += 1;
            if drift.has_request_drift() {
                areas.entry(area.area).or_default().request_drifted += 1;
            }
            if drift.has_response_drift() {
                areas.entry(area.area).or_default().response_drifted += 1;
            }
            if drift.has_other_drift() {
                areas.entry(area.area).or_default().other_drifted += 1;
            }
        }
    }

    areas.into_values().collect()
}
