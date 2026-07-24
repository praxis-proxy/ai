// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Compare Praxis AI operation coverage with OpenAI's `OpenAPI` specification.

use std::{collections::BTreeMap, ffi::OsString, path::PathBuf, process::Command};

use clap::Parser;

/// Conformance areas included in this task.
mod area;
/// Operation coverage calculation.
mod coverage;
/// Machine-readable JSON report rendering.
mod json_report;
/// Shared report and operation model types.
mod model;
/// `oasdiff` execution and result extraction.
mod oasdiff;
/// Human-readable terminal report rendering.
mod print;
/// Complete reference verification and semantic area projection.
mod reference;
/// Semantic YAML tree used for full-spec projection.
mod semantic_yaml;
/// `OpenAPI` spec loading and operation extraction.
mod spec;

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, reason = "tests")]
mod tests;

use area::{ApiArea, CONFORMANCE_AREAS, OPENAI_REFERENCE_MANIFEST, OPENAI_REFERENCE_SPEC};
use coverage::calculate_coverage;
use json_report::write_json_report;
use model::{
    CoverageReport, ReferenceSourceReport, RuntimeVerificationArea, RuntimeVerificationCheck, RuntimeVerificationStatus,
};
use oasdiff::run_scoped_oasdiff;
use print::print_report;
pub(crate) use reference::{Args as ReferenceArgs, run as run_reference};
use spec::{load_reference_source, project_reference};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Default local source for OpenAI's published `OpenAPI` spec.
const DEFAULT_OPENAI_SPEC: &str = OPENAI_REFERENCE_SPEC;

/// HTTP method keys that may appear under an `OpenAPI` path item.
const HTTP_METHODS: &[&str] = &["delete", "get", "head", "options", "patch", "post", "put", "trace"];

// -----------------------------------------------------------------------------
// CLI Arguments
// -----------------------------------------------------------------------------

/// CLI arguments for `cargo xtask openai-conformance`.
#[expect(clippy::struct_excessive_bools, reason = "independent CLI flags")]
#[derive(Parser)]
pub(crate) struct Args {
    /// Local OpenAI `OpenAPI` spec path.
    #[arg(long, default_value = DEFAULT_OPENAI_SPEC)]
    openai_spec: String,

    /// Select one registered area. Repeat for multiple areas; defaults to all.
    #[arg(long = "area", value_name = "AREA")]
    areas: Vec<String>,

    /// Include operations marked `deprecated: true` in the denominator.
    #[arg(long)]
    include_deprecated: bool,

    /// Include OpenAI beta query-path operations in the denominator.
    #[arg(long)]
    include_beta: bool,

    /// Print every covered operation and the local evidence for it.
    #[arg(long)]
    list_covered: bool,

    /// Print every missing operation that is included in the denominator.
    #[arg(long)]
    list_missing: bool,

    /// Override the generated local implementation `OpenAPI` spec used for
    /// `oasdiff`.
    #[arg(long)]
    implementation_spec: Option<String>,

    /// Print every operation that `oasdiff` reports as missing or drifted.
    #[arg(long)]
    list_oasdiff: bool,

    /// Write the full conformance report as JSON.
    #[arg(long, value_name = "PATH")]
    output_json: Option<PathBuf>,

    /// Exit non-zero when covered operation percentage is below this value.
    #[arg(long, value_name = "PERCENT", value_parser = parse_percent)]
    fail_under: Option<f64>,

    /// Exit non-zero when `oasdiff` exact-operation percentage is below this
    /// value.
    #[arg(long, value_name = "PERCENT", value_parser = parse_percent)]
    fail_oasdiff_under: Option<f64>,
}

// -----------------------------------------------------------------------------
// Entry Point
// -----------------------------------------------------------------------------

/// Run OpenAI API conformance coverage calculation.
#[expect(clippy::too_many_lines, reason = "CLI orchestration and threshold handling")]
pub(crate) fn run(args: &Args) {
    let result = run_inner(args).unwrap_or_else(|e| {
        eprintln!("openai-conformance failed: {e}");
        std::process::exit(1);
    });

    print_report(&result, args);

    if let Some(path) = &args.output_json {
        write_json_report(&result, args, path).unwrap_or_else(|e| {
            eprintln!("openai-conformance failed: {e}");
            std::process::exit(1);
        });
        println!("wrote JSON report: {}", path.display());
    }

    if result
        .runtime_verifications
        .iter()
        .any(|verification| verification.status == RuntimeVerificationStatus::Failed)
    {
        eprintln!("openai-conformance failed: one or more runtime verification commands failed");
        std::process::exit(1);
    }

    if let Some(threshold) = args.fail_oasdiff_under {
        let Some(oasdiff) = &result.oasdiff else {
            eprintln!("openai-conformance failed: oasdiff report was not generated");
            std::process::exit(1);
        };
        if oasdiff.conformance_percent() < threshold {
            eprintln!(
                "oasdiff conformance {:.2}% is below --fail-oasdiff-under {:.2}%",
                oasdiff.conformance_percent(),
                threshold,
            );
            std::process::exit(1);
        }
    }

    if let Some(threshold) = args.fail_under
        && result.coverage_percent() < threshold
    {
        eprintln!(
            "OpenAI conformance coverage {:.2}% is below --fail-under {:.2}%",
            result.coverage_percent(),
            threshold,
        );
        std::process::exit(1);
    }
}

/// Run the task and return structured results.
#[expect(clippy::too_many_lines, reason = "multi-area orchestration")]
fn run_inner(args: &Args) -> Result<CoverageReport, String> {
    let areas = selected_areas(args)?;
    if args.implementation_spec.is_some() && areas.len() != 1 {
        return Err("--implementation-spec requires exactly one selected --area".to_owned());
    }

    let mut sources = Vec::new();
    let mut operations = Vec::new();
    let mut supported_operations = Vec::new();
    let mut reference_projections = BTreeMap::new();
    let reference_override = reference_spec_override(args);
    let reference_spec = reference_override.unwrap_or(OPENAI_REFERENCE_SPEC);
    let manifest = reference_override.is_none().then_some(OPENAI_REFERENCE_MANIFEST);
    let reference = load_reference_source(reference_spec, manifest)?;

    for area in &areas {
        let mut projected = project_reference(&reference, area.scope)?;
        reference_projections.insert(area.scope.id, projected.content);
        sources.push(projected.report);
        operations.append(&mut projected.operations);
        supported_operations.extend((area.supported_operations)());
    }

    let reference_report = ReferenceSourceReport {
        source: reference.source,
        source_sha256: reference.source_sha256,
        provenance: reference.provenance,
    };
    let mut report = calculate_coverage(
        reference_report,
        sources,
        operations,
        &supported_operations,
        args.include_deprecated,
        args.include_beta,
    );

    let owned_operations = report.owned_operations();
    report.oasdiff = Some(run_scoped_oasdiff(
        args.implementation_spec.as_deref(),
        &owned_operations,
        &areas,
        &reference_projections,
    )?);
    report.runtime_verifications = run_runtime_verifications(&areas);

    Ok(report)
}

/// Check whether every declared sentinel appears as an exact trimmed line.
pub(crate) fn find_missing_sentinels<'a>(stdout: &str, checks: &'a [RuntimeVerificationCheck]) -> Vec<&'a str> {
    let lines: Vec<&str> = stdout.lines().map(str::trim).collect();
    checks
        .iter()
        .filter(|check| !lines.contains(&check.success_sentinel))
        .map(|check| check.success_sentinel)
        .collect()
}

/// Execute the focused runtime contract checks for each selected area.
fn run_runtime_verifications(areas: &[&ApiArea]) -> Vec<RuntimeVerificationArea> {
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo"));
    areas
        .iter()
        .map(|area| RuntimeVerificationArea {
            area: area.scope.label,
            command: area.runtime_test_command,
            checks: area.runtime_checks,
            status: run_area_checks(&cargo, area),
        })
        .collect()
}

/// Run the focused test command for one area and check sentinels.
fn run_area_checks(cargo: &OsString, area: &ApiArea) -> RuntimeVerificationStatus {
    match Command::new(cargo).args(area.runtime_test_args).output() {
        Ok(output) if output.status.success() => {
            let missing = find_missing_sentinels(&String::from_utf8_lossy(&output.stdout), area.runtime_checks);
            if missing.is_empty() {
                RuntimeVerificationStatus::Passed
            } else {
                eprintln!("runtime verification missing sentinels: {}", missing.join(", "));
                RuntimeVerificationStatus::Failed
            }
        },
        Ok(output) => {
            eprintln!("runtime verification failed: {}", area.runtime_test_command);
            eprintln!("{}", String::from_utf8_lossy(&output.stdout).trim());
            eprintln!("{}", String::from_utf8_lossy(&output.stderr).trim());
            RuntimeVerificationStatus::Failed
        },
        Err(error) => {
            eprintln!("failed to run {}: {error}", area.runtime_test_command);
            RuntimeVerificationStatus::Failed
        },
    }
}

/// Resolve requested area IDs against the static registry.
fn selected_areas(args: &Args) -> Result<Vec<&'static ApiArea>, String> {
    if args.areas.is_empty() || matches!(args.areas.as_slice(), [area] if area == "all") {
        return Ok(CONFORMANCE_AREAS.iter().collect());
    }
    if args.areas.iter().any(|area| area == "all") {
        return Err("--area all cannot be combined with another area".to_owned());
    }

    let mut selected = Vec::new();
    for requested in &args.areas {
        let area = CONFORMANCE_AREAS
            .iter()
            .find(|area| area.scope.id == requested)
            .ok_or_else(|| {
                let available = CONFORMANCE_AREAS
                    .iter()
                    .map(|area| area.scope.id)
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("unknown conformance area {requested:?}; available areas: all, {available}")
            })?;
        if selected
            .iter()
            .any(|selected: &&ApiArea| selected.scope.id == area.scope.id)
        {
            return Err(format!("conformance area {requested:?} was selected more than once"));
        }
        selected.push(area);
    }
    Ok(selected)
}

/// Parse a finite percentage in the inclusive range 0 through 100.
fn parse_percent(value: &str) -> Result<f64, String> {
    let percent = value
        .parse::<f64>()
        .map_err(|e| format!("invalid percentage {value:?}: {e}"))?;
    if percent.is_finite() && (0.0..=100.0).contains(&percent) {
        Ok(percent)
    } else {
        Err("percentage must be a finite value between 0 and 100".to_owned())
    }
}

/// Return the explicit reference override, if the CLI provided one.
fn reference_spec_override(args: &Args) -> Option<&str> {
    if args.openai_spec == DEFAULT_OPENAI_SPEC {
        None
    } else {
        Some(&args.openai_spec)
    }
}
