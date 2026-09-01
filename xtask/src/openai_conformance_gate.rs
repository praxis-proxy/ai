// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Enforce or acknowledge failures recorded in an OpenAI conformance report.

use std::{collections::BTreeSet, path::PathBuf};

use clap::Parser;
use serde::Serialize;
use serde_json::Value;

/// Generated report schema understood by this gate.
const REPORT_SCHEMA_VERSION: u64 = 3;

/// CLI arguments for `cargo xtask openai-conformance-gate`.
#[derive(Parser)]
pub(crate) struct Args {
    /// Generated conformance report to inspect.
    #[arg(long, value_name = "PATH")]
    report: PathBuf,

    /// Acknowledge recorded failures instead of requiring strict conformance.
    #[arg(long)]
    acknowledge: bool,

    /// Base-branch report used to detect newly introduced failures.
    #[arg(long, value_name = "PATH", requires = "acknowledge")]
    base_report: Option<PathBuf>,

    /// Permit failures absent from the base-branch report.
    #[arg(long, requires = "acknowledge")]
    allow_new_failures: bool,
}

/// One normalized conformance failure carried by the generated report.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
struct FailureFingerprint {
    /// Stable failure category.
    kind: String,
    /// Stable API area identifier, when applicable.
    area: Option<String>,
    /// HTTP operation label, when applicable.
    operation: Option<String>,
    /// Exact structural drift path, when applicable.
    detail: Option<String>,
}

impl FailureFingerprint {
    /// Build one stable failure fingerprint.
    fn new(kind: &str, area: Option<&str>, operation: Option<&str>, detail: Option<&str>) -> Self {
        Self {
            kind: kind.to_owned(),
            area: area.map(str::to_owned),
            operation: operation.map(str::to_owned),
            detail: detail.map(str::to_owned),
        }
    }
}

/// Run the report gate and terminate on policy failure.
pub(crate) fn run(args: &Args) {
    if let Err(error) = run_inner(args) {
        eprintln!("openai-conformance-gate failed: {error}");
        std::process::exit(1);
    }
}

/// Enforce strict conformance or validate an acknowledgement.
#[expect(
    clippy::too_many_lines,
    reason = "strict and acknowledged gate policy is kept together"
)]
fn run_inner(args: &Args) -> Result<(), String> {
    let report = read_report(&args.report)?;
    let failures = failure_fingerprints(&report)?;

    if failures.is_empty() {
        println!("OpenAI conformance is exact; no acknowledgement is required");
        return Ok(());
    }

    if !args.acknowledge {
        print_failures("strict OpenAI conformance failed", &failures);
        return Err(format!("{} conformance failure fingerprints recorded", failures.len()));
    }

    let base_failures = args
        .base_report
        .as_ref()
        .map(|path| read_report(path).and_then(|report| failure_fingerprints(&report)))
        .transpose()?
        .unwrap_or_default();
    let introduced = failures.difference(&base_failures).cloned().collect::<BTreeSet<_>>();

    if !introduced.is_empty() && !args.allow_new_failures {
        print_failures("new conformance failures require acknowledgement", &introduced);
        return Err(format!(
            "{} failure fingerprints are absent from the base report",
            introduced.len()
        ));
    }

    if introduced.is_empty() {
        println!(
            "acknowledged {} existing OpenAI conformance failure fingerprints",
            failures.len()
        );
    } else {
        print_failures("explicitly acknowledged new conformance failures", &introduced);
        println!(
            "acknowledged {} total OpenAI conformance failure fingerprints",
            failures.len()
        );
    }

    Ok(())
}

/// Load one generated JSON report.
fn read_report(path: &std::path::Path) -> Result<Value, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|error| format!("failed to read conformance report {}: {error}", path.display()))?;
    serde_json::from_str(&content)
        .map_err(|error| format!("failed to parse conformance report {}: {error}", path.display()))
}

/// Extract all exact failure identities carried by one report.
fn failure_fingerprints(report: &Value) -> Result<BTreeSet<FailureFingerprint>, String> {
    let schema_version = report
        .get("schema_version")
        .and_then(Value::as_u64)
        .ok_or_else(|| "report field schema_version must be an unsigned integer".to_owned())?;
    if schema_version != REPORT_SCHEMA_VERSION {
        return Err(format!(
            "unsupported conformance report schema version {schema_version}; expected {REPORT_SCHEMA_VERSION}"
        ));
    }

    let mut failures = BTreeSet::new();
    collect_capability_failures(report, &mut failures)?;
    collect_owned_contract_failures(report, &mut failures)?;
    Ok(failures)
}

/// Collect missing and stale capability declarations.
fn collect_capability_failures(report: &Value, failures: &mut BTreeSet<FailureFingerprint>) -> Result<(), String> {
    let coverage = required_object(report, "/capability_coverage")?;
    for operation in required_array_field(coverage, "missing_operations")? {
        failures.insert(FailureFingerprint::new(
            "capability_missing",
            optional_str(operation, "area"),
            Some(required_str(operation, "operation")?),
            None,
        ));
    }
    for operation in required_array_field(coverage, "claims_outside_reference")? {
        failures.insert(FailureFingerprint::new(
            "claim_outside_reference",
            optional_str(operation, "area"),
            Some(required_str(operation, "operation")?),
            None,
        ));
    }
    Ok(())
}

/// Collect missing operations and every operation or inherited drift detail.
#[expect(
    clippy::too_many_lines,
    reason = "straight-line traversal of the owned-contract report"
)]
fn collect_owned_contract_failures(report: &Value, failures: &mut BTreeSet<FailureFingerprint>) -> Result<(), String> {
    let initial_failure_count = failures.len();
    let owned = required_object(report, "/owned_contract_conformance")?;
    if owned.get("enabled").and_then(Value::as_bool) != Some(true) {
        failures.insert(FailureFingerprint::new(
            "owned_contract_comparison_disabled",
            None,
            None,
            None,
        ));
        return Ok(());
    }

    for area in required_array_field(owned, "areas")? {
        let area_id = required_str(area, "area_id")?;
        let contracts = required_object(area, "/operation_contracts")?;
        for operation in required_array_field(contracts, "missing_operations")? {
            failures.insert(FailureFingerprint::new(
                "owned_operation_missing",
                Some(area_id),
                Some(required_str(operation, "operation")?),
                None,
            ));
        }
        for drift in required_array_field(contracts, "drifted_operations")? {
            let drift = drift
                .as_object()
                .ok_or_else(|| "drifted_operations entries must be objects".to_owned())?;
            let operation = required_str_field(required_object_field(drift, "operation")?, "operation")?;
            collect_detail_failures(failures, area_id, operation, "request_drift", drift, "request_details")?;
            collect_detail_failures(
                failures,
                area_id,
                operation,
                "response_drift",
                drift,
                "response_details",
            )?;
            collect_detail_failures(failures, area_id, operation, "other_drift", drift, "other_details")?;
        }

        let inherited = required_object(area, "/inherited_contract")?;
        for detail in required_array_field(inherited, "details")? {
            failures.insert(FailureFingerprint::new(
                "inherited_contract_drift",
                Some(area_id),
                None,
                Some(value_str(detail, "inherited contract detail")?),
            ));
        }
    }

    let all_exact = owned
        .get("all_exact")
        .and_then(Value::as_bool)
        .ok_or_else(|| "report field /owned_contract_conformance/all_exact must be boolean".to_owned())?;
    if !all_exact && failures.len() == initial_failure_count {
        return Err("owned contract report is not exact but contains no failure details".to_owned());
    }
    Ok(())
}

/// Add one failure per normalized drift detail path.
#[expect(
    clippy::too_many_arguments,
    reason = "drift identity fields are explicit at the traversal boundary"
)]
fn collect_detail_failures(
    failures: &mut BTreeSet<FailureFingerprint>,
    area: &str,
    operation: &str,
    kind: &str,
    drift: &serde_json::Map<String, Value>,
    field: &str,
) -> Result<(), String> {
    for detail in required_array_field(drift, field)? {
        failures.insert(FailureFingerprint::new(
            kind,
            Some(area),
            Some(operation),
            Some(value_str(detail, field)?),
        ));
    }
    Ok(())
}

/// Resolve a required JSON object by pointer.
fn required_object<'a>(value: &'a Value, pointer: &str) -> Result<&'a serde_json::Map<String, Value>, String> {
    value
        .pointer(pointer)
        .and_then(Value::as_object)
        .ok_or_else(|| format!("report field {pointer} must be an object"))
}

/// Read a required array field from an object.
fn required_array_field<'a>(value: &'a serde_json::Map<String, Value>, field: &str) -> Result<&'a Vec<Value>, String> {
    value
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("report field {field} must be an array"))
}

/// Read a required object field.
fn required_object_field<'a>(
    value: &'a serde_json::Map<String, Value>,
    field: &str,
) -> Result<&'a serde_json::Map<String, Value>, String> {
    value
        .get(field)
        .and_then(Value::as_object)
        .ok_or_else(|| format!("report field {field} must be an object"))
}

/// Read a required string field from an object.
fn required_str_field<'a>(value: &'a serde_json::Map<String, Value>, field: &str) -> Result<&'a str, String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("report field {field} must be a string"))
}

/// Read a required string field.
fn required_str<'a>(value: &'a Value, field: &str) -> Result<&'a str, String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("report field {field} must be a string"))
}

/// Read an optional string field.
fn optional_str<'a>(value: &'a Value, field: &str) -> Option<&'a str> {
    value.get(field).and_then(Value::as_str)
}

/// Read a JSON string value with a contextual error.
fn value_str<'a>(value: &'a Value, field: &str) -> Result<&'a str, String> {
    value
        .as_str()
        .ok_or_else(|| format!("report field {field} must contain strings"))
}

/// Print deterministic JSON fingerprints for CI logs.
fn print_failures(heading: &str, failures: &BTreeSet<FailureFingerprint>) {
    eprintln!("{heading}:");
    for failure in failures {
        let rendered = serde_json::to_string(failure).unwrap_or_else(|_| format!("{failure:?}"));
        eprintln!("  {rendered}");
    }
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, reason = "tests")]
mod tests {
    use serde_json::json;

    use super::*;

    fn report(request_details: &[&str], response_details: &[&str]) -> Value {
        json!({
            "schema_version": REPORT_SCHEMA_VERSION,
            "capability_coverage": {
                "missing_operations": [],
                "claims_outside_reference": [],
            },
            "owned_contract_conformance": {
                "enabled": true,
                "all_exact": request_details.is_empty() && response_details.is_empty(),
                "areas": [{
                    "area_id": "conversations",
                    "operation_contracts": {
                        "missing_operations": [],
                        "drifted_operations": if request_details.is_empty() && response_details.is_empty() {
                            json!([])
                        } else {
                            json!([{
                                "operation": {"operation": "POST /conversations"},
                                "request_details": request_details,
                                "response_details": response_details,
                                "other_details": [],
                            }])
                        },
                    },
                    "inherited_contract": {"details": []},
                }],
            },
        })
    }

    #[test]
    fn exact_report_has_no_failure_fingerprints() {
        assert!(failure_fingerprints(&report(&[], &[])).unwrap().is_empty());
    }

    #[test]
    fn drift_details_are_independent_failure_fingerprints() {
        let failures = failure_fingerprints(&report(&["requestBody.required"], &["responses.200.schema"])).unwrap();
        assert_eq!(failures.len(), 2);
        assert!(failures.iter().any(|failure| failure.kind == "request_drift"));
        assert!(failures.iter().any(|failure| failure.kind == "response_drift"));
    }

    #[test]
    fn introduced_failures_are_set_difference_from_base() {
        let base = failure_fingerprints(&report(&["requestBody.required"], &[])).unwrap();
        let current = failure_fingerprints(&report(&["requestBody.required"], &["responses.200.schema"])).unwrap();
        let introduced = current.difference(&base).collect::<Vec<_>>();
        assert_eq!(introduced.len(), 1);
        assert_eq!(introduced.first().unwrap().kind, "response_drift");
    }

    #[test]
    fn missing_comparison_is_a_failure() {
        let value = json!({
            "schema_version": REPORT_SCHEMA_VERSION,
            "capability_coverage": {
                "missing_operations": [],
                "claims_outside_reference": [],
            },
            "owned_contract_conformance": {
                "enabled": false,
            },
        });
        let failures = failure_fingerprints(&value).unwrap();
        assert_eq!(failures.len(), 1);
        assert_eq!(
            failures.iter().next().unwrap().kind,
            "owned_contract_comparison_disabled"
        );
    }

    #[test]
    fn unknown_report_schema_is_rejected() {
        let mut value = report(&[], &[]);
        value
            .as_object_mut()
            .unwrap()
            .insert("schema_version".to_owned(), json!(REPORT_SCHEMA_VERSION + 1));
        assert!(failure_fingerprints(&value).is_err());
    }
}
