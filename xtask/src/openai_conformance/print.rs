// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

use std::collections::{BTreeMap, BTreeSet};

use super::{
    Args,
    model::{AreaStats, CoverageReport, OasdiffOperationDrift, OasdiffReport, percent},
};

/// Print the coverage report.
#[expect(clippy::too_many_lines, reason = "single terminal report body")]
pub(super) fn print_report(report: &CoverageReport, args: &Args) {
    println!("OpenAI API conformance coverage");
    println!("reference: {}", report.reference.source);
    println!("area projections:");
    for source in &report.sources {
        println!(
            "  {}: {} operations ({})",
            source.area, source.operations, source.projection_sha256
        );
    }
    println!("scope: {}", report_scope(report));
    println!(
        "filters: {}, {}",
        if args.include_deprecated {
            "including deprecated operations"
        } else {
            "excluding deprecated operations"
        },
        if args.include_beta {
            "including beta operations"
        } else {
            "excluding beta operations"
        },
    );
    println!(
        "covered: {}/{} ({:.2}%)",
        report.covered.len(),
        report.considered.len(),
        report.coverage_percent(),
    );
    println!(
        "ignored: {} deprecated, {} beta",
        report.ignored.deprecated, report.ignored.beta
    );

    print_mode_summary(report);
    print_area_summary(report);
    if let Some(oasdiff) = &report.oasdiff {
        print_oasdiff_summary(oasdiff, args);
    }
    print_runtime_verification(report);

    if args.list_covered {
        print_covered(report);
    }
    if args.list_missing {
        print_missing(report);
    } else {
        print_top_missing_tags(report);
    }
    if args.list_covered && !report.stale_claims.is_empty() {
        print_stale_claims(report);
    }
}

/// Render the selected area list.
fn report_scope(report: &CoverageReport) -> String {
    report
        .sources
        .iter()
        .map(|source| source.area)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>()
        .join(", ")
}

/// Print `oasdiff` structural conformance details.
#[expect(clippy::too_many_lines, reason = "single terminal report section")]
fn print_oasdiff_summary(report: &OasdiffReport, args: &Args) {
    println!("owned contract conformance:");
    println!("  tool: oasdiff {}", report.tool_version);
    println!(
        "  exact operations: {}/{} ({:.2}%)",
        report.conformant,
        report.total,
        report.conformance_percent(),
    );
    println!("  missing operations: {}", report.missing.len());
    println!("  schema-drift operations: {}", report.drifted.len());
    println!("  request-drift operations: {}", report.request_drift_count());
    println!("  response-drift operations: {}", report.response_drift_count());
    println!("  other-drift operations: {}", report.other_drift_count());
    println!("  inherited area contracts exact: {}", report.area_drift.is_empty());
    for area in &report.area_drift {
        println!("    {}: {} drift details", area.area, area.details.len());
    }

    println!("oasdiff by area:");
    for area in &report.area_reports {
        println!(
            "  {}: {}/{} ({:.2}%), missing {}, drifted {}, implementation {}",
            area.area,
            area.conformant,
            area.total,
            area.conformance_percent(),
            area.missing.len(),
            area.drifted.len(),
            area.implementation_source,
        );
    }

    if args.list_oasdiff {
        print_oasdiff_operations(report);
    }
}

/// Print runtime verification results produced by focused tests.
fn print_runtime_verification(report: &CoverageReport) {
    println!("runtime verification:");
    for verification in &report.runtime_verifications {
        println!(
            "  {}: {} [{}]",
            verification.area,
            verification.status.as_str(),
            verification.command,
        );
    }
}

/// Print all non-conformant `oasdiff` operations.
fn print_oasdiff_operations(report: &OasdiffReport) {
    if !report.missing.is_empty() {
        println!("oasdiff missing operations:");
        for operation in &report.missing {
            println!("  {operation}");
        }
    }

    if !report.drifted.is_empty() {
        println!("oasdiff schema-drift operations:");
        for drift in &report.drifted {
            println!("  {}", drift.key);
        }
    }

    print_oasdiff_drift_group(
        "oasdiff response drift operations:",
        &report.drifted,
        OasdiffOperationDrift::has_response_drift,
        |drift| &drift.response_details,
    );
    print_oasdiff_drift_group(
        "oasdiff request drift operations:",
        &report.drifted,
        OasdiffOperationDrift::has_request_drift,
        |drift| &drift.request_details,
    );
    print_oasdiff_drift_group(
        "oasdiff other drift operations:",
        &report.drifted,
        OasdiffOperationDrift::has_other_drift,
        |drift| &drift.other_details,
    );
}

/// Print one categorized `oasdiff` drift group.
fn print_oasdiff_drift_group(
    heading: &str,
    drifted: &[OasdiffOperationDrift],
    matches: fn(&OasdiffOperationDrift) -> bool,
    details: fn(&OasdiffOperationDrift) -> &Vec<String>,
) {
    let matching = drifted.iter().filter(|drift| matches(drift)).collect::<Vec<_>>();
    if matching.is_empty() {
        return;
    }

    println!("{heading}");
    for drift in matching {
        println!("  {}", drift.key);
        print_oasdiff_detail_paths(details(drift));
    }
}

/// Print a bounded list of `oasdiff` detail paths.
fn print_oasdiff_detail_paths(details: &[String]) {
    const MAX_DETAILS: usize = 8;

    for detail in details.iter().take(MAX_DETAILS) {
        println!("    {detail}");
    }
    if details.len() > MAX_DETAILS {
        println!("    ... {} more", details.len() - MAX_DETAILS);
    }
}

/// Print coverage counts by support mode.
fn print_mode_summary(report: &CoverageReport) {
    let mut modes: BTreeMap<&str, usize> = BTreeMap::new();
    for covered in &report.covered {
        *modes.entry(covered.support.mode.as_str()).or_default() += 1;
    }

    if modes.is_empty() {
        return;
    }

    println!("covered by mode:");
    for (mode, count) in modes {
        println!("  {mode}: {count}");
    }
}

/// Print per-area coverage.
fn print_area_summary(report: &CoverageReport) {
    let mut stats: BTreeMap<&str, AreaStats> = BTreeMap::new();
    for operation in &report.considered {
        stats.entry(operation.area).or_default().total += 1;
    }
    for covered in &report.covered {
        stats.entry(covered.operation.area).or_default().covered += 1;
    }

    println!("coverage by area:");
    for (area, stat) in stats {
        println!(
            "  {area}: {}/{} ({:.2}%)",
            stat.covered,
            stat.total,
            percent(stat.covered, stat.total),
        );
    }
}

/// Print all covered operations.
fn print_covered(report: &CoverageReport) {
    println!("covered operations:");
    for covered in &report.covered {
        println!(
            "  {} [{}; {}; {}]",
            covered.operation.key,
            covered.operation.area,
            covered.support.mode.as_str(),
            covered.support.evidence,
        );
    }
}

/// Print all missing operations.
fn print_missing(report: &CoverageReport) {
    println!("missing operations:");
    for operation in &report.missing {
        match &operation.operation_id {
            Some(id) => println!("  {} [{}; {}; {id}]", operation.key, operation.area, operation.tag),
            None => println!("  {} [{}; {}]", operation.key, operation.area, operation.tag),
        }
    }
}

/// Print the areas with the largest number of missing operations.
fn print_top_missing_tags(report: &CoverageReport) {
    let mut missing_by_area: BTreeMap<&str, usize> = BTreeMap::new();
    for operation in &report.missing {
        *missing_by_area.entry(operation.area).or_default() += 1;
    }

    let mut ranked: Vec<(&str, usize)> = missing_by_area.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));

    if ranked.is_empty() {
        return;
    }

    println!("largest missing areas:");
    for (area, count) in ranked.into_iter().take(10) {
        println!("  {area}: {count}");
    }
    println!("run with --list-missing to print every missing operation");
}

/// Print support claims that no longer match the spec.
fn print_stale_claims(report: &CoverageReport) {
    println!("local support claims outside the selected specs:");
    for claim in &report.stale_claims {
        println!("  {} {} [{}]", claim.method, claim.path, claim.area);
    }
}
