// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

use std::fmt;

/// One `OpenAPI` operation key.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct OperationKey {
    /// Uppercase HTTP method.
    pub(super) method: String,

    /// `OpenAPI` path, without the server `/v1` prefix.
    pub(super) path: String,
}

impl OperationKey {
    /// Build an operation key.
    pub(super) fn new(method: impl Into<String>, path: impl Into<String>) -> Self {
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
pub(super) struct SpecOperation {
    /// Operation key.
    pub(super) key: OperationKey,

    /// First `OpenAPI` tag, or `untagged`.
    pub(super) tag: String,

    /// Conformance area for this operation.
    pub(super) area: &'static str,

    /// Operation ID, if present.
    pub(super) operation_id: Option<String>,

    /// Whether `OpenAPI` marks the operation as deprecated.
    pub(super) deprecated: bool,

    /// Whether this operation is one of OpenAI's beta query paths.
    pub(super) beta: bool,
}

/// How a supported operation is covered in this repository.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum CoverageMode {
    /// The operation is forwarded without payload mutation.
    Passthrough,

    /// Praxis reads selected fields and preserves the rest of the payload.
    Inspect,

    /// Praxis maps between distinct input and output contracts.
    Transform,

    /// The repo serves the endpoint locally.
    Local,
}

impl CoverageMode {
    /// Stable order used by machine-readable reports.
    pub(super) const ALL: [Self; 4] = [Self::Passthrough, Self::Inspect, Self::Transform, Self::Local];

    /// Stable output label.
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Passthrough => "passthrough",
            Self::Inspect => "inspect",
            Self::Transform => "transform",
            Self::Local => "local",
        }
    }

    /// Whether Praxis owns the operation's externally visible contract.
    pub(super) fn is_owned(self) -> bool {
        matches!(self, Self::Transform | Self::Local)
    }
}

/// One local support claim.
#[derive(Clone, Debug)]
pub(super) struct SupportedOperation {
    /// Uppercase HTTP method.
    pub(super) method: String,

    /// `OpenAPI` path.
    pub(super) path: String,

    /// Local feature area.
    pub(super) area: String,

    /// Coverage mode.
    pub(super) mode: CoverageMode,

    /// Code or test evidence for the support claim.
    pub(super) evidence: String,
}

impl SupportedOperation {
    /// Operation key for map lookup.
    pub(super) fn key(&self) -> OperationKey {
        OperationKey::new(&self.method, &self.path)
    }
}

/// Immutable upstream identity recorded for the complete reference.
#[derive(Clone, Debug)]
pub(super) struct ReferenceProvenance {
    /// Upstream repository URL.
    pub(super) repository: String,

    /// Full upstream Git commit.
    pub(super) revision: String,

    /// Document path within the repository.
    pub(super) path: String,

    /// SHA-256 of the complete upstream document.
    pub(super) source_sha256: String,
}

/// Selected spec source and the scoped operation count found in it.
#[derive(Debug)]
pub(super) struct SpecSourceReport {
    /// Stable machine-readable area identifier.
    pub(super) area_id: &'static str,

    /// Area loaded from this spec.
    pub(super) area: &'static str,

    /// Number of operations extracted for this area.
    pub(super) operations: usize,

    /// SHA-256 of the deterministic area projection.
    pub(super) projection_sha256: String,
}

/// Complete OpenAI source used by every area in one report.
#[derive(Debug)]
pub(super) struct ReferenceSourceReport {
    /// Checked-in source path or explicit CLI override.
    pub(super) source: String,

    /// SHA-256 of the complete source document.
    pub(super) source_sha256: String,

    /// Immutable upstream identity, absent only for an explicit CLI override.
    pub(super) provenance: Option<ReferenceProvenance>,
}

/// Aggregate coverage report.
#[derive(Debug)]
pub(super) struct CoverageReport {
    /// One complete upstream reference shared by all area projections.
    pub(super) reference: ReferenceSourceReport,

    /// Spec sources that were read.
    pub(super) sources: Vec<SpecSourceReport>,

    /// Operations included in the denominator.
    pub(super) considered: Vec<SpecOperation>,

    /// Operations ignored because of CLI scope flags.
    pub(super) ignored: IgnoredCounts,

    /// Covered operations included in the denominator.
    pub(super) covered: Vec<CoveredOperation>,

    /// Missing operations included in the denominator.
    pub(super) missing: Vec<SpecOperation>,

    /// Support claims whose key was not present in the spec.
    pub(super) stale_claims: Vec<SupportedOperation>,

    /// Optional `oasdiff` comparison against a local implementation spec.
    pub(super) oasdiff: Option<OasdiffReport>,

    /// Runtime checks declared for the selected areas.
    pub(super) runtime_verifications: Vec<RuntimeVerificationArea>,
}

impl CoverageReport {
    /// Covered operation percentage.
    pub(super) fn coverage_percent(&self) -> f64 {
        percent(self.covered.len(), self.considered.len())
    }

    /// Operations whose contract is owned by Praxis.
    pub(super) fn owned_operations(&self) -> Vec<SpecOperation> {
        self.covered
            .iter()
            .filter(|covered| covered.support.mode.is_owned())
            .map(|covered| covered.operation.clone())
            .collect()
    }
}

/// One operation that was covered by a support claim.
#[derive(Debug)]
pub(super) struct CoveredOperation {
    /// The spec operation.
    pub(super) operation: SpecOperation,

    /// The matched local support claim.
    pub(super) support: SupportedOperation,
}

/// Counts of operations ignored by scope flags.
#[derive(Default, Debug)]
pub(super) struct IgnoredCounts {
    /// Deprecated operations ignored by default.
    pub(super) deprecated: usize,

    /// Beta query-path operations ignored by default.
    pub(super) beta: usize,
}

/// Per-area aggregate coverage.
#[derive(Default)]
pub(super) struct AreaStats {
    /// Operations considered for this area.
    pub(super) total: usize,

    /// Operations covered for this area.
    pub(super) covered: usize,
}

/// `oasdiff` operation-level structural conformance report.
#[derive(Debug)]
pub(super) struct OasdiffReport {
    /// Exact checked `oasdiff` version.
    pub(super) tool_version: String,

    /// Operations in the selected denominator.
    pub(super) total: usize,

    /// Operations with no missing endpoint and no operation diff.
    pub(super) conformant: usize,

    /// Operations absent from the implementation spec.
    pub(super) missing: Vec<OperationKey>,

    /// Operations present but structurally different.
    pub(super) drifted: Vec<OasdiffOperationDrift>,

    /// Per-area counts.
    pub(super) areas: Vec<OasdiffAreaStats>,

    /// Complete comparison results and implementation source per area.
    pub(super) area_reports: Vec<OasdiffAreaReport>,

    /// Contract drift inherited outside operation objects.
    pub(super) area_drift: Vec<OasdiffAreaDrift>,
}

/// Complete owned-contract comparison for one API area.
#[derive(Debug)]
pub(super) struct OasdiffAreaReport {
    /// Stable machine-readable area identifier.
    pub(super) area_id: &'static str,

    /// Human-readable area label.
    pub(super) area: &'static str,

    /// Generated implementation document compared for this area.
    pub(super) implementation_source: String,

    /// Number of owned operations considered.
    pub(super) total: usize,

    /// Number of exact operations.
    pub(super) conformant: usize,

    /// Operations absent from the implementation document.
    pub(super) missing: Vec<OperationKey>,

    /// Operations with structural drift.
    pub(super) drifted: Vec<OasdiffOperationDrift>,

    /// Global or inherited contract drift for this area.
    pub(super) inherited_details: Vec<String>,
}

impl OasdiffAreaReport {
    /// Exact operation percentage for this area.
    pub(super) fn conformance_percent(&self) -> f64 {
        percent(self.conformant, self.total)
    }

    /// Whether operation and inherited area contracts are exact.
    pub(super) fn all_exact(&self) -> bool {
        self.conformant == self.total && self.inherited_details.is_empty()
    }
}

impl OasdiffReport {
    /// Exact operation conformance percentage.
    pub(super) fn conformance_percent(&self) -> f64 {
        percent(self.conformant, self.total)
    }

    /// Operations with request body or parameter drift.
    pub(super) fn request_drift_count(&self) -> usize {
        self.drifted.iter().filter(|drift| drift.has_request_drift()).count()
    }

    /// Operations with response schema drift.
    pub(super) fn response_drift_count(&self) -> usize {
        self.drifted.iter().filter(|drift| drift.has_response_drift()).count()
    }

    /// Operations with drift outside request or response contracts.
    pub(super) fn other_drift_count(&self) -> usize {
        self.drifted.iter().filter(|drift| drift.has_other_drift()).count()
    }

    /// Whether both operation and inherited area contracts are exact.
    pub(super) fn all_exact(&self) -> bool {
        self.conformant == self.total && self.area_drift.is_empty()
    }
}

/// Structural drift inherited by an entire API area.
#[derive(Clone, Debug)]
pub(super) struct OasdiffAreaDrift {
    /// Area label.
    pub(super) area: &'static str,

    /// Compact `oasdiff` paths for global or inherited changes.
    pub(super) details: Vec<String>,
}

/// `oasdiff` structural differences for one operation.
#[derive(Clone, Debug)]
pub(super) struct OasdiffOperationDrift {
    /// Drifted operation key.
    pub(super) key: OperationKey,

    /// Request body or parameter drift detail paths.
    pub(super) request_details: Vec<String>,

    /// Response schema drift detail paths.
    pub(super) response_details: Vec<String>,

    /// Other operation drift detail paths.
    pub(super) other_details: Vec<String>,
}

impl OasdiffOperationDrift {
    /// Build an empty drift record for an operation.
    pub(super) fn new(key: OperationKey) -> Self {
        Self {
            key,
            request_details: Vec::new(),
            response_details: Vec::new(),
            other_details: Vec::new(),
        }
    }

    /// Return whether this operation has any drift.
    pub(super) fn has_any_drift(&self) -> bool {
        self.has_request_drift() || self.has_response_drift() || self.has_other_drift()
    }

    /// Return whether this operation has request drift.
    pub(super) fn has_request_drift(&self) -> bool {
        !self.request_details.is_empty()
    }

    /// Return whether this operation has response drift.
    pub(super) fn has_response_drift(&self) -> bool {
        !self.response_details.is_empty()
    }

    /// Return whether this operation has other operation drift.
    pub(super) fn has_other_drift(&self) -> bool {
        !self.other_details.is_empty()
    }
}

/// `oasdiff` operation status counts for one area.
#[derive(Default, Debug)]
pub(super) struct OasdiffAreaStats {
    /// Area label.
    pub(super) area: &'static str,

    /// Operations considered in this area.
    pub(super) total: usize,

    /// Missing operations in this area.
    pub(super) missing: usize,

    /// Drifted operations in this area.
    pub(super) drifted: usize,

    /// Operations with request drift in this area.
    pub(super) request_drifted: usize,

    /// Operations with response drift in this area.
    pub(super) response_drifted: usize,

    /// Operations with drift outside request and response contracts.
    pub(super) other_drifted: usize,
}

impl OasdiffAreaStats {
    /// Operations with no `oasdiff` problem.
    pub(super) fn conformant(&self) -> usize {
        self.total.saturating_sub(self.missing + self.drifted)
    }
}

/// One runtime verification check executed by the report generator.
#[derive(Clone, Debug)]
pub(crate) struct RuntimeVerificationCheck {
    /// Stable check category.
    pub(crate) kind: &'static str,

    /// Fully qualified test evidence.
    pub(crate) evidence: &'static str,

    /// Exact sentinel line emitted by the test on success.
    pub(crate) success_sentinel: &'static str,
}

/// Runtime verification declaration for one area.
#[derive(Debug)]
pub(super) struct RuntimeVerificationArea {
    /// Area label.
    pub(super) area: &'static str,

    /// Focused command executed for this report.
    pub(super) command: &'static str,

    /// Checks selected by the command.
    pub(super) checks: &'static [RuntimeVerificationCheck],

    /// Result from executing the focused command during this report run.
    pub(super) status: RuntimeVerificationStatus,
}

/// Result of one focused runtime verification command.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RuntimeVerificationStatus {
    /// Every selected test passed.
    Passed,

    /// The command ran and returned a failure status.
    Failed,
}

impl RuntimeVerificationStatus {
    /// Stable machine-readable status.
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Passed => "passed",
            Self::Failed => "failed",
        }
    }
}

/// Data-driven API operation scope loaded from the full OpenAI spec.
#[derive(Clone, Copy, Debug)]
pub(super) struct OperationScope {
    /// Stable machine-readable area identifier.
    pub(super) id: &'static str,

    /// Human-readable report label.
    pub(super) label: &'static str,

    /// Segment-aware path prefixes belonging to the area.
    pub(super) path_prefixes: &'static [&'static str],
}

impl OperationScope {
    /// Declare one area selector.
    pub(super) const fn new(id: &'static str, label: &'static str, path_prefixes: &'static [&'static str]) -> Self {
        Self {
            id,
            label,
            path_prefixes,
        }
    }

    /// Return whether the operation path is in this scope.
    pub(super) fn matches(self, path: &str) -> bool {
        let path = path.split_once('?').map_or(path, |(path, _query)| path);
        self.path_prefixes
            .iter()
            .any(|prefix| path == *prefix || path.strip_prefix(prefix).is_some_and(|suffix| suffix.starts_with('/')))
    }
}


/// Return `count / total` as a percentage.
pub(super) fn percent(count: usize, total: usize) -> f64 {
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
