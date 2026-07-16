// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Compare Praxis AI's Conversations operation coverage with OpenAI's
//! `OpenAPI` specification.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    io::Write as _,
    path::{Path, PathBuf},
    process::Command,
};

use clap::Parser;
use serde_json::{Map, Value, json};
use tempfile::{Builder as TempFileBuilder, NamedTempFile};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Default source for OpenAI's published `OpenAPI` spec.
const DEFAULT_OPENAI_SPEC: &str = "https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml";

/// HTTP method keys that may appear under an `OpenAPI` path item.
const HTTP_METHODS: &[&str] = &["delete", "get", "head", "options", "patch", "post", "put", "trace"];

/// Rust sources scanned for local support evidence.
const SUPPORT_EVIDENCE_SOURCES: &[SupportEvidenceSource] = &[SupportEvidenceSource {
    path: "apis/src/openai/conversations/handlers.rs",
    mode: CoverageMode::Local,
}];

// -----------------------------------------------------------------------------
// CLI Arguments
// -----------------------------------------------------------------------------

/// CLI arguments for `cargo xtask openai-conformance`.
#[expect(clippy::struct_excessive_bools, reason = "independent CLI flags")]
#[derive(Parser)]
pub(crate) struct Args {
    /// OpenAI `OpenAPI` spec source for Conversations: an `http(s)` URL or a
    /// local file path.
    #[arg(long, default_value = DEFAULT_OPENAI_SPEC)]
    openai_spec: String,

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

    /// Local implementation `OpenAPI` spec to compare with `oasdiff`.
    #[arg(long)]
    implementation_spec: Option<String>,

    /// Print every operation that `oasdiff` reports as missing or drifted.
    #[arg(long)]
    list_oasdiff: bool,

    /// Write the full conformance report as JSON.
    #[arg(long, value_name = "PATH")]
    output_json: Option<PathBuf>,

    /// Exit non-zero when covered operation percentage is below this value.
    #[arg(long, value_name = "PERCENT")]
    fail_under: Option<f64>,

    /// Exit non-zero when `oasdiff` exact-operation percentage is below this
    /// value. Requires `--implementation-spec`.
    #[arg(long, value_name = "PERCENT")]
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

    if let Some(threshold) = args.fail_oasdiff_under {
        let Some(oasdiff) = &result.oasdiff else {
            eprintln!("--fail-oasdiff-under requires --implementation-spec");
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
fn run_inner(args: &Args) -> Result<CoverageReport, String> {
    let mut sources = Vec::new();
    let mut operations = Vec::new();

    let (conversations_source, mut conversations_operations) =
        load_scoped_operations(&args.openai_spec, OperationScope::Conversations)?;
    sources.push(conversations_source);
    operations.append(&mut conversations_operations);

    let supported_operations = discover_supported_operations()?;
    let mut report = calculate_coverage(
        sources,
        operations,
        &supported_operations,
        args.include_deprecated,
        args.include_beta,
    );

    if let Some(implementation_spec) = &args.implementation_spec {
        report.oasdiff = Some(run_scoped_oasdiff(
            &args.openai_spec,
            implementation_spec,
            &report.considered,
        )?);
    }

    Ok(report)
}

// -----------------------------------------------------------------------------
// Report Model
// -----------------------------------------------------------------------------

/// One `OpenAPI` operation key.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct OperationKey {
    /// Uppercase HTTP method.
    method: String,

    /// `OpenAPI` path, without the server `/v1` prefix.
    path: String,
}

impl OperationKey {
    /// Build an operation key.
    fn new(method: impl Into<String>, path: impl Into<String>) -> Self {
        Self {
            method: method.into(),
            path: path.into(),
        }
    }
}

impl fmt::Display for OperationKey {
    /// Render as `METHOD /path`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.method, self.path)
    }
}

/// One operation extracted from the `OpenAPI` document.
#[derive(Clone, Debug)]
struct SpecOperation {
    /// Operation key.
    key: OperationKey,

    /// First `OpenAPI` tag, or `untagged`.
    tag: String,

    /// Conformance area for this operation.
    area: &'static str,

    /// Operation ID, if present.
    operation_id: Option<String>,

    /// Whether `OpenAPI` marks the operation as deprecated.
    deprecated: bool,

    /// Whether this operation is one of OpenAI's beta query paths.
    beta: bool,
}

/// How a supported operation is covered in this repository.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum CoverageMode {
    /// The repo serves the endpoint locally.
    Local,
}

impl CoverageMode {
    /// Stable output label.
    fn as_str(self) -> &'static str {
        match self {
            Self::Local => "local",
        }
    }
}

/// One local support claim.
#[derive(Clone, Debug)]
struct SupportedOperation {
    /// Uppercase HTTP method.
    method: String,

    /// `OpenAPI` path.
    path: String,

    /// Local feature area.
    area: String,

    /// Coverage mode.
    mode: CoverageMode,

    /// Code or test evidence for the support claim.
    evidence: String,
}

impl SupportedOperation {
    /// Operation key for map lookup.
    fn key(&self) -> OperationKey {
        OperationKey::new(&self.method, &self.path)
    }
}

/// Source file used to discover local support evidence.
#[derive(Clone, Copy)]
struct SupportEvidenceSource {
    /// Source path relative to the repository root.
    path: &'static str,

    /// Support mode implied by this source.
    mode: CoverageMode,
}

/// Selected spec source and the scoped operation count found in it.
#[derive(Debug)]
struct SpecSourceReport {
    /// Area loaded from this spec.
    area: &'static str,

    /// Spec source requested by the user.
    source: String,

    /// Number of operations extracted for this area.
    operations: usize,
}

/// Aggregate coverage report.
#[derive(Debug)]
struct CoverageReport {
    /// Spec sources that were read.
    sources: Vec<SpecSourceReport>,

    /// Operations included in the denominator.
    considered: Vec<SpecOperation>,

    /// Operations ignored because of CLI scope flags.
    ignored: IgnoredCounts,

    /// Covered operations included in the denominator.
    covered: Vec<CoveredOperation>,

    /// Missing operations included in the denominator.
    missing: Vec<SpecOperation>,

    /// Support claims whose key was not present in the spec.
    stale_claims: Vec<SupportedOperation>,

    /// Optional `oasdiff` comparison against a local implementation spec.
    oasdiff: Option<OasdiffReport>,
}

impl CoverageReport {
    /// Covered operation percentage.
    fn coverage_percent(&self) -> f64 {
        percent(self.covered.len(), self.considered.len())
    }

    /// Headline conformance metric, preferring structural `oasdiff` results
    /// when an implementation spec was provided.
    fn overall_conformance(&self) -> OverallConformance {
        if let Some(oasdiff) = &self.oasdiff {
            return OverallConformance {
                basis: "oasdiff",
                total: oasdiff.total,
                conformant: oasdiff.conformant,
                missing: oasdiff.missing.len(),
                drifted: oasdiff.drifted.len(),
                percent: oasdiff.conformance_percent(),
            };
        }

        OverallConformance {
            basis: "operation_coverage",
            total: self.considered.len(),
            conformant: self.covered.len(),
            missing: self.missing.len(),
            drifted: 0,
            percent: self.coverage_percent(),
        }
    }
}

/// Headline conformance metric used by humans and automation.
struct OverallConformance {
    /// Metric source.
    basis: &'static str,

    /// Operations in the selected denominator.
    total: usize,

    /// Operations counted as conformant by the selected basis.
    conformant: usize,

    /// Operations absent from the implementation surface.
    missing: usize,

    /// Operations present but structurally drifted.
    drifted: usize,

    /// `conformant / total` as a percentage.
    percent: f64,
}

/// One operation that was covered by a support claim.
#[derive(Debug)]
struct CoveredOperation {
    /// The spec operation.
    operation: SpecOperation,

    /// The matched local support claim.
    support: SupportedOperation,
}

/// Counts of operations ignored by scope flags.
#[derive(Default, Debug)]
struct IgnoredCounts {
    /// Deprecated operations ignored by default.
    deprecated: usize,

    /// Beta query-path operations ignored by default.
    beta: usize,
}

/// Per-area aggregate coverage.
#[derive(Default)]
struct AreaStats {
    /// Operations considered for this area.
    total: usize,

    /// Operations covered for this area.
    covered: usize,
}

/// `oasdiff` operation-level structural conformance report.
#[derive(Debug)]
struct OasdiffReport {
    /// Implementation spec compared against the references.
    implementation_source: String,

    /// Operations in the selected denominator.
    total: usize,

    /// Operations with no missing endpoint and no operation diff.
    conformant: usize,

    /// Operations absent from the implementation spec.
    missing: Vec<OperationKey>,

    /// Operations present but structurally different.
    drifted: Vec<OasdiffOperationDrift>,

    /// Per-area counts.
    areas: Vec<OasdiffAreaStats>,
}

impl OasdiffReport {
    /// Exact operation conformance percentage.
    fn conformance_percent(&self) -> f64 {
        percent(self.conformant, self.total)
    }

    /// Operations with request body or parameter drift.
    fn request_drift_count(&self) -> usize {
        self.drifted.iter().filter(|drift| drift.has_request_drift()).count()
    }

    /// Operations with response schema drift.
    fn response_drift_count(&self) -> usize {
        self.drifted.iter().filter(|drift| drift.has_response_drift()).count()
    }

    /// Operations with drift outside request or response contracts.
    fn other_drift_count(&self) -> usize {
        self.drifted.iter().filter(|drift| drift.has_other_drift()).count()
    }
}

/// `oasdiff` structural differences for one operation.
#[derive(Debug)]
struct OasdiffOperationDrift {
    /// Drifted operation key.
    key: OperationKey,

    /// Request body or parameter drift detail paths.
    request_details: Vec<String>,

    /// Response schema drift detail paths.
    response_details: Vec<String>,

    /// Other operation drift detail paths.
    other_details: Vec<String>,
}

impl OasdiffOperationDrift {
    /// Build an empty drift record for an operation.
    fn new(key: OperationKey) -> Self {
        Self {
            key,
            request_details: Vec::new(),
            response_details: Vec::new(),
            other_details: Vec::new(),
        }
    }

    /// Return whether this operation has any drift.
    fn has_any_drift(&self) -> bool {
        self.has_request_drift() || self.has_response_drift() || self.has_other_drift()
    }

    /// Return whether this operation has request drift.
    fn has_request_drift(&self) -> bool {
        !self.request_details.is_empty()
    }

    /// Return whether this operation has response drift.
    fn has_response_drift(&self) -> bool {
        !self.response_details.is_empty()
    }

    /// Return whether this operation has other operation drift.
    fn has_other_drift(&self) -> bool {
        !self.other_details.is_empty()
    }
}

/// `oasdiff` operation status counts for one area.
#[derive(Default, Debug)]
struct OasdiffAreaStats {
    /// Area label.
    area: &'static str,

    /// Operations considered in this area.
    total: usize,

    /// Missing operations in this area.
    missing: usize,

    /// Drifted operations in this area.
    drifted: usize,

    /// Operations with request drift in this area.
    request_drifted: usize,

    /// Operations with response drift in this area.
    response_drifted: usize,

    /// Operations with drift outside request and response contracts.
    other_drifted: usize,
}

impl OasdiffAreaStats {
    /// Operations with no `oasdiff` problem.
    fn conformant(&self) -> usize {
        self.total.saturating_sub(self.missing + self.drifted)
    }
}

/// API operation scope loaded from a spec source.
#[derive(Clone, Copy, Debug)]
enum OperationScope {
    /// Conversations operations from the OpenAI spec.
    Conversations,
}

impl OperationScope {
    /// Stable report label.
    fn label(self) -> &'static str {
        match self {
            Self::Conversations => "Conversations",
        }
    }

    /// Return whether the operation path is in this scope.
    fn matches(self, path: &str) -> bool {
        let path = path.split_once('?').map_or(path, |(path, _query)| path);
        match self {
            Self::Conversations => path == "/conversations" || path.starts_with("/conversations/"),
        }
    }
}

// -----------------------------------------------------------------------------
// Coverage Calculation
// -----------------------------------------------------------------------------

/// Calculate coverage for extracted spec operations.
#[expect(clippy::too_many_lines, reason = "straight-line report classification")]
fn calculate_coverage(
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
        sources,
        considered,
        ignored,
        covered,
        missing,
        stale_claims,
        oasdiff: None,
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

/// Discover locally supported operations from route documentation in source.
fn discover_supported_operations() -> Result<Vec<SupportedOperation>, String> {
    let mut operations = BTreeMap::new();
    for source in SUPPORT_EVIDENCE_SOURCES {
        collect_source_supported_operations(source, &mut operations)?;
    }
    Ok(operations.into_values().collect())
}

/// Collect support claims from one source file.
fn collect_source_supported_operations(
    source: &SupportEvidenceSource,
    operations: &mut BTreeMap<OperationKey, SupportedOperation>,
) -> Result<(), String> {
    let source_path = repo_root().join(source.path);
    let content = std::fs::read_to_string(&source_path).map_err(|e| format!("failed to read {}: {e}", source.path))?;

    for (index, line) in content.lines().enumerate() {
        for (method, path) in operation_mentions(line) {
            let Some(path) = normalize_source_operation_path(&path) else {
                continue;
            };
            let Some(area) = support_area(&path) else {
                continue;
            };

            let operation = SupportedOperation {
                method,
                path,
                area: area.to_owned(),
                mode: source.mode,
                evidence: source_evidence(source.path, index + 1, line),
            };
            operations.entry(operation.key()).or_insert(operation);
        }
    }

    Ok(())
}

/// Extract route-like operation mentions from backtick-delimited source text.
fn operation_mentions(line: &str) -> Vec<(String, String)> {
    backtick_segments(line)
        .into_iter()
        .filter_map(operation_mention)
        .collect()
}

/// Return all backtick-delimited segments from a line.
fn backtick_segments(line: &str) -> Vec<&str> {
    line.split('`')
        .enumerate()
        .filter_map(|(index, segment)| (index % 2 == 1).then_some(segment))
        .collect()
}

/// Parse one `METHOD /v1/path` segment.
fn operation_mention(segment: &str) -> Option<(String, String)> {
    let mut parts = segment.split_whitespace();
    let method = parts.next()?;
    let path = parts.next()?;

    if !HTTP_METHODS.iter().any(|known| method.eq_ignore_ascii_case(known)) {
        return None;
    }
    if !path.starts_with("/v1/conversations") {
        return None;
    }

    Some((method.to_ascii_uppercase(), path.to_owned()))
}

/// Normalize a source route path to the spec's path template.
fn normalize_source_operation_path(path: &str) -> Option<String> {
    let path = path.strip_prefix("/v1")?;
    if path == "/conversations" || path.starts_with("/conversations/") {
        return Some(path.replacen("{id}", "{conversation_id}", 1));
    }
    None
}

/// Return the support area for a normalized operation path.
fn support_area(path: &str) -> Option<&'static str> {
    (path == "/conversations" || path.starts_with("/conversations/")).then_some("Conversations")
}

/// Render compact source evidence for a discovered support claim.
fn source_evidence(path: &str, line: usize, text: &str) -> String {
    let text = text.trim().trim_start_matches("///").trim_start_matches("//").trim();
    format!("{path}:{line}: {text}")
}

/// Repository root inferred from the `xtask` manifest location.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask manifest should have a repository parent")
        .to_path_buf()
}

// -----------------------------------------------------------------------------
// `oasdiff` Comparison
// -----------------------------------------------------------------------------

/// Run `oasdiff` against the selected reference spec.
fn run_scoped_oasdiff(
    openai_source: &str,
    implementation_source: &str,
    considered: &[SpecOperation],
) -> Result<OasdiffReport, String> {
    let openai_reference = materialize_spec(openai_source)?;
    let implementation = materialize_spec(implementation_source)?;

    let mut missing = BTreeSet::new();
    let mut drifted = BTreeMap::new();

    let conversations_diff = run_oasdiff_diff(openai_reference.path(), implementation.path(), "Conversations")?;
    collect_oasdiff_operation_status(
        &conversations_diff,
        OperationScope::Conversations,
        considered,
        &mut missing,
        &mut drifted,
    );

    Ok(build_oasdiff_report(
        implementation_source,
        considered,
        missing,
        drifted,
    ))
}

/// Write a URL or local spec source to a temporary file for `oasdiff`.
fn materialize_spec(source: &str) -> Result<NamedTempFile, String> {
    let content = read_spec(source)?;
    let suffix = spec_suffix(source);
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

/// Guess an `OpenAPI` file suffix from a source path or URL.
fn spec_suffix(source: &str) -> &'static str {
    let source = source.split_once('?').map_or(source, |(path, _query)| path);
    if source.ends_with(".json") { ".json" } else { ".yaml" }
}

/// Execute `oasdiff diff` and return its JSON output.
fn run_oasdiff_diff(reference: &Path, implementation: &Path, area: &str) -> Result<Value, String> {
    let output = Command::new("oasdiff")
        .arg("diff")
        .arg(reference)
        .arg(implementation)
        .arg("--format")
        .arg("json")
        .arg("--strip-prefix-revision")
        .arg("/v1")
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

/// Extract missing and drifted operation keys from `oasdiff` JSON.
#[expect(clippy::too_many_lines, reason = "straight-line traversal of oasdiff JSON")]
fn collect_oasdiff_operation_status(
    diff: &Value,
    scope: OperationScope,
    considered: &[SpecOperation],
    missing: &mut BTreeSet<OperationKey>,
    drifted: &mut BTreeMap<OperationKey, OasdiffOperationDrift>,
) {
    let considered_keys = considered
        .iter()
        .map(|operation| operation.key.clone())
        .collect::<BTreeSet<_>>();

    let Some(paths) = diff.get("paths").and_then(Value::as_object) else {
        return;
    };

    if let Some(deleted_paths) = paths.get("deleted").and_then(Value::as_array) {
        for path in deleted_paths.iter().filter_map(Value::as_str) {
            if !scope.matches(path) {
                continue;
            }
            for operation in considered.iter().filter(|operation| operation.key.path == path) {
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
fn operation_drift_from_diff(key: OperationKey, operation_diff: &Value) -> Option<OasdiffOperationDrift> {
    let mut drift = OasdiffOperationDrift::new(key);
    let fields = operation_diff.as_object()?;

    for (field, value) in fields {
        if is_non_contract_oasdiff_field(field) {
            continue;
        }
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
                if is_non_contract_oasdiff_field(field) {
                    continue;
                }
                let path = format!("{prefix}.{field}");
                collect_oasdiff_detail_paths(&path, child, details);
            }
        },
        Value::Array(_) | Value::Bool(_) | Value::Number(_) | Value::String(_) | Value::Null => {
            push_oasdiff_detail(prefix, details);
        },
    }
}

/// Return whether an `oasdiff` field is documentation or generator metadata.
fn is_non_contract_oasdiff_field(field: &str) -> bool {
    matches!(
        field,
        "description"
            | "example"
            | "examples"
            | "extensions"
            | "operationID"
            | "operationId"
            | "summary"
            | "tags"
            | "title"
    )
}

/// Push a normalized `oasdiff` detail path.
fn push_oasdiff_detail(path: &str, details: &mut Vec<String>) {
    let path = path.replace(".modified.", ".");
    if !details.contains(&path) {
        details.push(path);
    }
}

/// Build a sorted `oasdiff` report from extracted status sets.
fn build_oasdiff_report(
    implementation_source: &str,
    considered: &[SpecOperation],
    missing: BTreeSet<OperationKey>,
    drifted: BTreeMap<OperationKey, OasdiffOperationDrift>,
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
        implementation_source: implementation_source.to_owned(),
        total: considered.len(),
        conformant,
        missing: missing_vec,
        drifted: drifted_vec,
        areas,
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

// -----------------------------------------------------------------------------
// Spec Loading and Parsing
// -----------------------------------------------------------------------------

/// Load and scope operations from a spec source.
fn load_scoped_operations(
    source: &str,
    scope: OperationScope,
) -> Result<(SpecSourceReport, Vec<SpecOperation>), String> {
    let content = read_spec(source)?;
    let operations = scope_operations(parse_openapi_operations(&content)?, scope);

    if operations.is_empty() {
        return Err(format!(
            "{} spec {source} did not contain any {} operations",
            scope.label(),
            scope.label(),
        ));
    }

    let report = SpecSourceReport {
        area: scope.label(),
        source: source.to_owned(),
        operations: operations.len(),
    };
    Ok((report, operations))
}

/// Keep only operations for the requested scope and tag them with the area.
fn scope_operations(operations: Vec<SpecOperation>, scope: OperationScope) -> Vec<SpecOperation> {
    let mut scoped: Vec<SpecOperation> = operations
        .into_iter()
        .filter(|operation| scope.matches(&operation.key.path))
        .map(|mut operation| {
            operation.area = scope.label();
            operation
        })
        .collect();
    scoped.sort_by(|a, b| a.key.cmp(&b.key));
    scoped
}

/// Read the `OpenAPI` document from a URL or local path.
fn read_spec(source: &str) -> Result<String, String> {
    if is_url(source) {
        return read_spec_url(source);
    }

    std::fs::read_to_string(Path::new(source)).map_err(|e| format!("failed to read {source}: {e}"))
}

/// Read the `OpenAPI` document from an HTTP URL.
fn read_spec_url(url: &str) -> Result<String, String> {
    let response = reqwest::blocking::get(url).map_err(|e| format!("failed to fetch {url}: {e}"))?;
    let status = response.status();
    if !status.is_success() {
        return Err(format!("failed to fetch {url}: HTTP {status}"));
    }
    response.text().map_err(|e| format!("failed to read {url}: {e}"))
}

/// Return whether a spec source is an HTTP URL.
fn is_url(source: &str) -> bool {
    source.starts_with("https://") || source.starts_with("http://")
}

/// Parse `OpenAPI` operations from a YAML or JSON document.
fn parse_openapi_operations(content: &str) -> Result<Vec<SpecOperation>, String> {
    if content.trim_start().starts_with('{') {
        return parse_json_openapi_operations(content);
    }

    parse_yaml_openapi_operations(content)
}

/// Parse `OpenAPI` operations from a JSON document.
fn parse_json_openapi_operations(content: &str) -> Result<Vec<SpecOperation>, String> {
    let spec: Value = serde_json::from_str(content).map_err(|e| format!("failed to parse OpenAPI JSON: {e}"))?;
    let paths = spec
        .get("paths")
        .and_then(Value::as_object)
        .ok_or_else(|| "OpenAPI document does not contain a paths object".to_owned())?;

    let mut operations = Vec::new();
    for (path, path_item) in paths {
        let Some(path_item) = path_item.as_object() else {
            continue;
        };
        for method in HTTP_METHODS {
            let Some(operation) = path_item.get(*method).and_then(Value::as_object) else {
                continue;
            };
            operations.push(parse_json_operation(path, method, operation));
        }
    }
    operations.sort_by(|a, b| a.key.cmp(&b.key));
    Ok(operations)
}

/// Parse `OpenAPI` operations from the top-level YAML `paths` block.
#[expect(clippy::too_many_lines, reason = "small state machine over YAML indentation")]
fn parse_yaml_openapi_operations(content: &str) -> Result<Vec<SpecOperation>, String> {
    let mut operations = Vec::new();
    let mut in_paths = false;
    let mut found_paths = false;
    let mut current_path: Option<String> = None;
    let mut current_operation: Option<YamlOperationBuilder> = None;
    let mut reading_tags = false;

    for line in content.lines() {
        let indent = leading_spaces(line);
        let trimmed = line.trim();

        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        if indent == 0 {
            if trimmed == "paths:" {
                in_paths = true;
                found_paths = true;
                continue;
            }
            if in_paths {
                break;
            }
        }
        if !in_paths {
            continue;
        }

        if indent == 2 && trimmed.starts_with('/') && trimmed.ends_with(':') {
            push_yaml_operation(&mut operations, &mut current_operation);
            current_path = trimmed.strip_suffix(':').map(str::to_owned);
            reading_tags = false;
            continue;
        }

        if indent == 4 {
            if let Some(method) = trimmed.strip_suffix(':')
                && HTTP_METHODS.contains(&method)
                && let Some(path) = &current_path
            {
                push_yaml_operation(&mut operations, &mut current_operation);
                current_operation = Some(YamlOperationBuilder::new(path, method));
                reading_tags = false;
            }
            continue;
        }

        if indent == 6 {
            reading_tags = update_yaml_operation_field(trimmed, &mut current_operation);
            continue;
        }

        if indent == 8 && reading_tags {
            if let Some(tag) = parse_yaml_list_item(trimmed)
                && let Some(operation) = current_operation.as_mut()
            {
                operation.set_tag_if_unset(tag);
            }
            continue;
        }

        if indent <= 6 {
            reading_tags = false;
        }
    }

    push_yaml_operation(&mut operations, &mut current_operation);

    if !found_paths {
        return Err("OpenAPI YAML document does not contain a paths block".to_owned());
    }

    operations.sort_by(|a, b| a.key.cmp(&b.key));
    Ok(operations)
}

/// Parse a single JSON `OpenAPI` operation object.
fn parse_json_operation(path: &str, method: &str, operation: &Map<String, Value>) -> SpecOperation {
    SpecOperation {
        key: OperationKey::new(method.to_ascii_uppercase(), path),
        tag: first_json_tag(operation),
        area: "unscoped",
        operation_id: operation.get("operationId").and_then(Value::as_str).map(str::to_owned),
        deprecated: operation.get("deprecated").and_then(Value::as_bool).unwrap_or(false),
        beta: path.contains("?beta=true"),
    }
}

/// Return the JSON operation's first tag, or `untagged`.
fn first_json_tag(operation: &Map<String, Value>) -> String {
    operation
        .get("tags")
        .and_then(Value::as_array)
        .and_then(|tags| tags.first())
        .and_then(Value::as_str)
        .unwrap_or("untagged")
        .to_owned()
}

/// Mutable YAML operation under construction.
struct YamlOperationBuilder {
    /// Operation key.
    key: OperationKey,

    /// First operation tag.
    tag: String,

    /// `OpenAPI` operation ID.
    operation_id: Option<String>,

    /// Deprecated marker.
    deprecated: bool,
}

impl YamlOperationBuilder {
    /// Build a new YAML operation accumulator.
    fn new(path: &str, method: &str) -> Self {
        Self {
            key: OperationKey::new(method.to_ascii_uppercase(), path),
            tag: "untagged".to_owned(),
            operation_id: None,
            deprecated: false,
        }
    }

    /// Set the tag only when one has not already been found.
    fn set_tag_if_unset(&mut self, tag: String) {
        if self.tag == "untagged" {
            self.tag = tag;
        }
    }

    /// Convert into a completed [`SpecOperation`].
    fn into_operation(self) -> SpecOperation {
        let beta = self.key.path.contains("?beta=true");
        SpecOperation {
            key: self.key,
            tag: self.tag,
            area: "unscoped",
            operation_id: self.operation_id,
            deprecated: self.deprecated,
            beta,
        }
    }
}

/// Push a completed YAML operation into the output list.
fn push_yaml_operation(operations: &mut Vec<SpecOperation>, current: &mut Option<YamlOperationBuilder>) {
    if let Some(operation) = current.take() {
        operations.push(operation.into_operation());
    }
}

/// Update a YAML operation field and return whether subsequent list items are
/// operation tags.
fn update_yaml_operation_field(trimmed: &str, current: &mut Option<YamlOperationBuilder>) -> bool {
    let Some(operation) = current.as_mut() else {
        return false;
    };

    if let Some(value) = trimmed.strip_prefix("operationId:") {
        operation.operation_id = parse_yaml_scalar(value);
        return false;
    }
    if let Some(value) = trimmed.strip_prefix("deprecated:") {
        operation.deprecated = value.trim() == "true";
        return false;
    }
    if let Some(value) = trimmed.strip_prefix("tags:") {
        if let Some(tag) = parse_yaml_inline_tag(value) {
            operation.set_tag_if_unset(tag);
        }
        return true;
    }

    false
}

/// Parse a YAML scalar string.
fn parse_yaml_scalar(value: &str) -> Option<String> {
    let value = value.trim();
    if value.is_empty() || value == "[]" {
        return None;
    }
    Some(value.trim_matches('"').trim_matches('\'').to_owned())
}

/// Parse an inline YAML tag value, such as `tags: [Conversations]`.
fn parse_yaml_inline_tag(value: &str) -> Option<String> {
    let value = value.trim();
    let inner = value.strip_prefix('[')?.strip_suffix(']')?;
    inner
        .split(',')
        .find_map(parse_yaml_scalar)
        .filter(|tag| !tag.is_empty())
}

/// Parse a YAML sequence item, such as `- Conversations`.
fn parse_yaml_list_item(value: &str) -> Option<String> {
    let value = value.strip_prefix('-')?;
    parse_yaml_scalar(value).filter(|tag| !tag.is_empty())
}

/// Count leading spaces in a YAML line.
fn leading_spaces(line: &str) -> usize {
    line.bytes().take_while(|b| *b == b' ').count()
}

// -----------------------------------------------------------------------------
// JSON Report Rendering
// -----------------------------------------------------------------------------

/// Write the coverage report as pretty JSON.
fn write_json_report(report: &CoverageReport, args: &Args, path: &Path) -> Result<(), String> {
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
fn report_json(report: &CoverageReport, args: &Args) -> Value {
    json!({
        "schema_version": 1,
        "scope": {
            "areas": ["Conversations"],
            "include_deprecated": args.include_deprecated,
            "include_beta": args.include_beta,
        },
        "sources": report.sources.iter().map(spec_source_json).collect::<Vec<_>>(),
        "overall_conformance": overall_conformance_json(&report.overall_conformance()),
        "operation_coverage": {
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
            "local_support_claims_outside_selected_specs": report
                .stale_claims
                .iter()
                .map(supported_operation_json)
                .collect::<Vec<_>>(),
        },
        "oasdiff": report.oasdiff.as_ref().map_or_else(oasdiff_not_run_json, oasdiff_report_json),
    })
}

/// Serialize the headline conformance metric.
fn overall_conformance_json(overall: &OverallConformance) -> Value {
    json!({
        "basis": overall.basis,
        "total_operations": overall.total,
        "conformant_operations_count": overall.conformant,
        "missing_operations_count": overall.missing,
        "drifted_operations_count": overall.drifted,
        "conformance_percent": overall.percent,
    })
}

/// Serialize one spec source.
fn spec_source_json(source: &SpecSourceReport) -> Value {
    json!({
        "area": source.area,
        "source": source.source.as_str(),
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
        modes
            .into_iter()
            .map(|(mode, count)| json!({"mode": mode, "covered": count}))
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
        "implementation_source": report.implementation_source.as_str(),
        "total_operations": report.total,
        "conformant_operations_count": report.conformant,
        "missing_operations_count": report.missing.len(),
        "drifted_operations_count": report.drifted.len(),
        "request_drift_operations_count": report.request_drift_count(),
        "response_drift_operations_count": report.response_drift_count(),
        "other_drift_operations_count": report.other_drift_count(),
        "conformance_percent": report.conformance_percent(),
        "by_area": report.areas.iter().map(oasdiff_area_json).collect::<Vec<_>>(),
        "fixes_required": oasdiff_fixes_required_json(report),
        "missing_operations": report.missing.iter().map(operation_key_json).collect::<Vec<_>>(),
        "drifted_operations": report.drifted.iter().map(oasdiff_operation_drift_json).collect::<Vec<_>>(),
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

    json!({
        "summary": {
            "operations_to_fix": report.missing.len() + report.drifted.len(),
            "missing_operations": report.missing.len(),
            "request_drift_operations": report.request_drift_count(),
            "response_drift_operations": report.response_drift_count(),
            "other_drift_operations": report.other_drift_count(),
        },
        "by_operation": by_operation,
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
        "reason": "not run; pass --implementation-spec <path-or-url> to compare against a local implementation OpenAPI spec",
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

// -----------------------------------------------------------------------------
// Report Rendering
// -----------------------------------------------------------------------------

/// Print the coverage report.
#[expect(clippy::too_many_lines, reason = "single terminal report body")]
fn print_report(report: &CoverageReport, args: &Args) {
    println!("OpenAI Conversations conformance coverage");
    println!("sources:");
    for source in &report.sources {
        println!(
            "  {}: {} ({} operations)",
            source.area, source.source, source.operations
        );
    }
    println!("scope: Conversations only");
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

    print_overall_conformance(report);
    print_mode_summary(report);
    print_area_summary(report);
    if let Some(oasdiff) = &report.oasdiff {
        print_oasdiff_summary(oasdiff, args);
    }

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

/// Print the headline conformance number.
fn print_overall_conformance(report: &CoverageReport) {
    let overall = report.overall_conformance();
    println!(
        "overall conformance: {}/{} ({:.2}%) [{}]",
        overall.conformant, overall.total, overall.percent, overall.basis,
    );
    if overall.basis == "oasdiff" {
        println!("  missing operations: {}", overall.missing);
        println!("  drifted operations: {}", overall.drifted);
    }
}

/// Print `oasdiff` structural conformance details.
#[expect(clippy::too_many_lines, reason = "single terminal report section")]
fn print_oasdiff_summary(report: &OasdiffReport, args: &Args) {
    println!("oasdiff structural conformance:");
    println!("  implementation: {}", report.implementation_source);
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

    println!("oasdiff by area:");
    for area in &report.areas {
        println!(
            "  {}: {}/{} ({:.2}%), missing {}, drifted {} (request {}, response {}, other {})",
            area.area,
            area.conformant(),
            area.total,
            percent(area.conformant(), area.total),
            area.missing,
            area.drifted,
            area.request_drifted,
            area.response_drifted,
            area.other_drifted,
        );
    }

    if args.list_oasdiff {
        print_oasdiff_operations(report);
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

/// Return `count / total` as a percentage.
fn percent(count: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        (count_as_f64(count) / count_as_f64(total)) * 100.0
    }
}

/// Convert a count to `f64` for display percentages.
fn count_as_f64(count: usize) -> f64 {
    f64::from(u32::try_from(count).unwrap_or(u32::MAX))
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, reason = "tests")]
mod tests {
    use serde_json::json;

    use super::*;

    /// Small representative `OpenAPI` document.
    const SPEC: &str = "
openapi: 3.1.0
paths:
  /conversations:
    post:
      operationId: createConversation
      tags: [Conversations]
  /conversations?beta=true:
    post:
      operationId: createConversationBeta
      tags: [Conversations]
  /conversations/{conversation_id}:
    get:
      operationId: getConversation
      tags: [Conversations]
  /conversations/{conversation_id}/items:
    get:
      operationId: listConversationItems
      tags: [Conversations]
  /conversations/legacy:
    get:
      operationId: listLegacyConversations
      tags: [Conversations]
      deprecated: true
  /chat/completions:
    post:
      operationId: createChatCompletion
      tags: [Chat]
  /models:
    get:
      operationId: listModels
      tags: [Models]
";

    fn scoped_operations() -> Vec<SpecOperation> {
        let operations = parse_openapi_operations(SPEC).unwrap();
        scope_operations(operations, OperationScope::Conversations)
    }

    fn test_sources() -> Vec<SpecSourceReport> {
        vec![SpecSourceReport {
            area: "Conversations",
            source: "openai-test".to_owned(),
            operations: 5,
        }]
    }

    fn support(method: &str, path: &str, area: &str, mode: CoverageMode) -> SupportedOperation {
        SupportedOperation {
            method: method.to_owned(),
            path: path.to_owned(),
            area: area.to_owned(),
            mode,
            evidence: "test evidence".to_owned(),
        }
    }

    fn test_supported_operations() -> Vec<SupportedOperation> {
        vec![
            support("POST", "/conversations", "Conversations", CoverageMode::Local),
            support(
                "GET",
                "/conversations/{conversation_id}",
                "Conversations",
                CoverageMode::Local,
            ),
            support(
                "GET",
                "/conversations/{conversation_id}/items",
                "Conversations",
                CoverageMode::Local,
            ),
        ]
    }

    fn test_args() -> Args {
        Args {
            openai_spec: "openai-test".to_owned(),
            include_deprecated: false,
            include_beta: false,
            list_covered: false,
            list_missing: false,
            implementation_spec: None,
            list_oasdiff: false,
            output_json: None,
            fail_under: None,
            fail_oasdiff_under: None,
        }
    }

    #[test]
    fn parses_operations() {
        let operations = parse_openapi_operations(SPEC).unwrap();
        assert_eq!(operations.len(), 7, "should parse all operation entries");
        assert!(
            operations
                .iter()
                .any(|op| op.key == OperationKey::new("POST", "/conversations")),
            "POST /conversations should be parsed"
        );
        assert!(
            operations.iter().any(|op| op.deprecated),
            "deprecated marker should be parsed"
        );
        assert!(operations.iter().any(|op| op.beta), "beta path should be marked");
    }

    #[test]
    fn scopes_only_conversations() {
        let operations = scoped_operations();

        assert_eq!(operations.len(), 5, "scope should exclude chat and model operations");
        assert!(
            operations.iter().all(|op| op.area == "Conversations"),
            "all scoped operations should receive an area"
        );
        assert!(
            operations
                .iter()
                .all(|op| !op.key.path.starts_with("/chat") && !op.key.path.starts_with("/models")),
            "unsupported API families should not enter the denominator"
        );
    }

    #[test]
    fn discovers_supported_operations_from_source_evidence() {
        let supported = discover_supported_operations().unwrap();
        let keys = supported.iter().map(SupportedOperation::key).collect::<BTreeSet<_>>();

        assert!(
            keys.contains(&OperationKey::new(
                "DELETE",
                "/conversations/{conversation_id}/items/{item_id}"
            )),
            "Conversations item delete support should come from handler source evidence"
        );
        assert!(
            supported
                .iter()
                .any(|operation| operation.evidence.contains("apis/src/openai/conversations/handlers.rs")),
            "discovered operations should carry source evidence"
        );
    }

    #[test]
    fn excludes_deprecated_and_beta_by_default() {
        let report = calculate_coverage(
            test_sources(),
            scoped_operations(),
            &test_supported_operations(),
            false,
            false,
        );

        assert_eq!(report.ignored.deprecated, 1, "one deprecated operation ignored");
        assert_eq!(report.ignored.beta, 1, "one beta operation ignored");
        assert_eq!(
            report.considered.len(),
            3,
            "stable denominator should exclude ignored operations"
        );
        assert_eq!(
            report.covered.len(),
            3,
            "all stable scoped operations should be covered"
        );
        assert!(report.missing.is_empty(), "unscoped model list should not be missing");
    }

    #[test]
    fn includes_deprecated_and_beta_when_requested() {
        let report = calculate_coverage(
            test_sources(),
            scoped_operations(),
            &test_supported_operations(),
            true,
            true,
        );

        assert_eq!(report.ignored.deprecated, 0, "deprecated should be included");
        assert_eq!(report.ignored.beta, 0, "beta should be included");
        assert_eq!(report.considered.len(), 5, "all scoped operations should be considered");
        assert_eq!(report.covered.len(), 3, "deprecated and beta entries are not claimed");
        assert_eq!(
            report.missing.len(),
            2,
            "deprecated and beta scoped entries should be missing"
        );
    }

    #[test]
    fn coverage_percent_handles_empty_denominator() {
        let report = calculate_coverage(test_sources(), Vec::new(), &test_supported_operations(), false, false);
        assert_eq!(report.coverage_percent(), 0.0, "empty denominator should be 0%");
    }

    #[test]
    #[expect(clippy::too_many_lines, reason = "representative oasdiff JSON fixture")]
    fn extracts_oasdiff_deleted_and_modified_operations() {
        let operations = scoped_operations();
        let diff = json!({
            "paths": {
                "deleted": ["/conversations/{conversation_id}"],
                "modified": {
                    "/conversations": {
                        "operations": {
                            "modified": {
                                "POST": {
                                    "requestBody": {
                                        "content": {
                                            "modified": {
                                                "application/json": {}
                                            }
                                        }
                                    },
                                    "responses": {
                                        "modified": {
                                            "200": {
                                                "content": {
                                                    "modified": {
                                                        "application/json": {
                                                            "schema": {
                                                                "properties": {
                                                                    "modified": {
                                                                        "id": {
                                                                            "type": {
                                                                                "added": ["integer"],
                                                                                "deleted": ["string"]
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });
        let mut missing = BTreeSet::new();
        let mut drifted = BTreeMap::new();

        collect_oasdiff_operation_status(
            &diff,
            OperationScope::Conversations,
            &operations,
            &mut missing,
            &mut drifted,
        );

        assert!(
            missing.contains(&OperationKey::new("GET", "/conversations/{conversation_id}")),
            "deleted path should mark its scoped operations as missing"
        );
        let drift = drifted
            .get(&OperationKey::new("POST", "/conversations"))
            .expect("modified operation should be classified as drifted");
        assert!(drift.has_request_drift(), "requestBody drift should be classified");
        assert!(drift.has_response_drift(), "responses drift should be classified");
        assert!(
            drift
                .response_details
                .iter()
                .any(|detail| detail.contains("responses.200.content.application/json.schema.properties.id.type")),
            "response type drift should be included in details"
        );
    }

    #[test]
    fn oasdiff_report_counts_missing_and_drifted_as_nonconformant() {
        let operations = scoped_operations();
        let missing = BTreeSet::from([OperationKey::new("GET", "/conversations/{conversation_id}")]);
        let drift = OasdiffOperationDrift {
            key: OperationKey::new("POST", "/conversations"),
            request_details: Vec::new(),
            response_details: vec!["responses.200.content.application/json.schema.type".to_owned()],
            other_details: Vec::new(),
        };
        let drifted = BTreeMap::from([(drift.key.clone(), drift)]);

        let report = build_oasdiff_report("implementation", &operations, missing, drifted);

        assert_eq!(
            report.total,
            operations.len(),
            "all scoped operations should be counted"
        );
        assert_eq!(report.missing.len(), 1, "missing operation should be preserved");
        assert_eq!(report.drifted.len(), 1, "drifted operation should be preserved");
        assert_eq!(report.response_drift_count(), 1, "response drift should be counted");
        assert_eq!(report.request_drift_count(), 0, "request drift should not be counted");
        assert_eq!(
            report.conformant,
            operations.len() - 2,
            "missing and drifted operations should reduce exact conformance"
        );
    }

    #[test]
    #[expect(clippy::too_many_lines, reason = "representative oasdiff JSON fixture")]
    fn ignores_oasdiff_documentation_only_response_changes() {
        let drift = operation_drift_from_diff(
            OperationKey::new("POST", "/conversations"),
            &json!({
                "responses": {
                    "modified": {
                        "200": {
                            "description": {
                                "from": "old",
                                "to": "new"
                            },
                            "content": {
                                "modified": {
                                    "application/json": {
                                        "schema": {
                                            "example": {
                                                "from": {"id": "old"},
                                                "to": {"id": "new"}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }),
        );

        assert!(
            drift.is_none(),
            "documentation-only response changes should not count as schema drift"
        );
    }

    #[test]
    #[expect(clippy::too_many_lines, reason = "JSON report fixture and assertions")]
    fn json_report_includes_oasdiff_missing_and_response_drift() {
        let operations = scoped_operations();
        let mut report = calculate_coverage(test_sources(), operations, &test_supported_operations(), false, false);
        let missing = BTreeSet::from([OperationKey::new("GET", "/conversations/{conversation_id}")]);
        let drift = OasdiffOperationDrift {
            key: OperationKey::new("POST", "/conversations"),
            request_details: Vec::new(),
            response_details: vec!["responses.200.content.application/json.schema.type".to_owned()],
            other_details: Vec::new(),
        };
        let drifted = BTreeMap::from([(drift.key.clone(), drift)]);
        report.oasdiff = Some(build_oasdiff_report(
            "implementation-spec",
            &report.considered,
            missing,
            drifted,
        ));

        let value = report_json(&report, &test_args());

        assert_eq!(value.pointer("/schema_version").and_then(Value::as_u64), Some(1));
        assert_eq!(
            value.pointer("/overall_conformance/basis").and_then(Value::as_str),
            Some("oasdiff")
        );
        assert_eq!(
            value
                .pointer("/overall_conformance/conformant_operations_count")
                .and_then(Value::as_u64),
            Some(1)
        );
        assert_eq!(
            value
                .pointer("/overall_conformance/conformance_percent")
                .and_then(Value::as_f64),
            Some(percent(1, report.considered.len()))
        );
        assert_eq!(value.pointer("/oasdiff/enabled").and_then(Value::as_bool), Some(true));
        assert_eq!(
            value
                .pointer("/oasdiff/missing_operations_count")
                .and_then(Value::as_u64),
            Some(1)
        );
        assert_eq!(
            value
                .pointer("/oasdiff/response_drift_operations_count")
                .and_then(Value::as_u64),
            Some(1)
        );
        assert_eq!(
            value
                .pointer("/oasdiff/fixes_required/summary/operations_to_fix")
                .and_then(Value::as_u64),
            Some(2)
        );
        assert_eq!(
            value
                .pointer("/oasdiff/fixes_required/by_operation/0/fixes/0/kind")
                .and_then(Value::as_str),
            Some("missing_operation")
        );
        assert_eq!(
            value
                .pointer("/oasdiff/fixes_required/by_operation/1/fixes/0/kind")
                .and_then(Value::as_str),
            Some("response_schema")
        );
        assert_eq!(
            value
                .pointer("/oasdiff/drifted_operations/0/response_details/0")
                .and_then(Value::as_str),
            Some("responses.200.content.application/json.schema.type")
        );
    }

    #[test]
    fn json_report_explains_when_oasdiff_was_not_run() {
        let report = calculate_coverage(
            test_sources(),
            scoped_operations(),
            &test_supported_operations(),
            false,
            false,
        );
        let value = report_json(&report, &test_args());

        assert_eq!(
            value.pointer("/overall_conformance/basis").and_then(Value::as_str),
            Some("operation_coverage")
        );
        assert_eq!(
            value
                .pointer("/overall_conformance/conformance_percent")
                .and_then(Value::as_f64),
            Some(100.0)
        );
        assert_eq!(value.pointer("/oasdiff/enabled").and_then(Value::as_bool), Some(false));
        assert!(
            value
                .pointer("/oasdiff/reason")
                .and_then(Value::as_str)
                .is_some_and(|reason| reason.contains("--implementation-spec")),
            "disabled oasdiff section should explain how to enable it"
        );
    }
}
